//! One world slot: the game in progress, its prepared decision and replay log.
//!
//! A slot plays its own sequence of games; the next one starts in the same job
//! that finished the previous one, so a slot never idles. Every game is a pure
//! function of `(run seed, slot, game ordinal)`, the update it started in and
//! the logged actions, which lets a resumed run replay an in-flight game to
//! the exact state it had when its checkpoint was committed.

use bota_proto::ModifierSpec;

use super::episode::{
    ACTOR_DECISIONS, CONTINUE_STRIDE, CompletedAdvance, EpisodeRecord, EpisodeStream,
    RetainedChoice, TICK_CAP, retains_continue,
};
use super::game_summary::GameSummary;
use super::*;
use crate::{EncoderRow, SampledStatistics};

/// Upper bound on slots of one run; also the slot factor of per-game seeds.
pub(crate) const MAX_SLOTS: usize = crate::PPO_MAX_SLOTS;
const ACTOR_DOMAIN: u64 = 0x736c_6f74_5f61_6374;
const MIXTURE_DOMAIN: u64 = 0x736c_6f74_5f6d_6978;
const RETENTION_DOMAIN: u64 = 0x736c_6f74_5f72_6574;

/// Who plays the opponent seat of one game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpponentKind {
    /// The scripted Teacher, run on the simulation worker.
    Teacher,
    /// The HarassPush rule policy, run on the simulation worker.
    HarassPush,
    /// A rule policy drawing its preset's style per game, run on the simulation worker.
    Styled(crate::StyledScript),
    /// Frozen weights snapshot `index` of the run's opponent pool.
    Snapshot(usize),
    /// The actor weights the policy seat samples from.
    SelfPlay,
    /// This run's runtime-history milestone after `update` completed updates.
    League(u64),
}

impl OpponentKind {
    pub(crate) fn label(self) -> String {
        match self {
            Self::Teacher => "teacher".to_owned(),
            Self::HarassPush => "harass-push".to_owned(),
            Self::Styled(script) => script.label().to_owned(),
            Self::Snapshot(index) => format!("weights{index}"),
            Self::SelfPlay => "self".to_owned(),
            Self::League(update) => format!("u{update:04}"),
        }
    }
}

/// Opponent kinds with integer mixture weights (millionths), in configured order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OpponentMixture {
    entries: Vec<(OpponentKind, u64)>,
    total: u64,
}

impl OpponentMixture {
    pub(crate) fn new(entries: Vec<(OpponentKind, u64)>) -> Result<Self, PpoError> {
        if entries.is_empty() || entries.len() > super::opponents::MAX_MIXTURE_ENTRIES {
            return Err(PpoError::InvalidConfig(
                "opponent mixture needs 1..=32 entries",
            ));
        }
        let mut total = 0u64;
        for (index, &(kind, weight)) in entries.iter().enumerate() {
            if weight == 0 || weight > 1_000_000_000 {
                return Err(PpoError::InvalidConfig("opponent weight"));
            }
            if entries[..index].iter().any(|(other, _)| *other == kind) {
                return Err(PpoError::InvalidConfig("duplicate opponent mixture entry"));
            }
            total += weight;
        }
        Ok(Self { entries, total })
    }

    pub(crate) fn entries(&self) -> &[(OpponentKind, u64)] {
        &self.entries
    }

    pub(crate) const fn total(&self) -> u64 {
        self.total
    }
}

/// The opponent of game `game` on slot `slot`: one seeded weighted draw.
///
/// Kept a pure function of its arguments so a resumed run redraws every game
/// identically; schedules that adapt to results must depend only on reports of
/// updates every lane has finished (at most `update - 2`).
pub(crate) fn draw_opponent(
    mixture: &OpponentMixture,
    seed: u64,
    slot: usize,
    game: u64,
    _update: u64,
) -> Result<OpponentKind, PpoError> {
    let mut random = PpoRng::new(derive_training_seed(
        seed,
        game_key(slot, game)?,
        MIXTURE_DOMAIN,
    ));
    let mut ticket = random.below(mixture.total)?;
    for &(kind, weight) in &mixture.entries {
        if ticket < weight {
            return Ok(kind);
        }
        ticket -= weight;
    }
    unreachable!("the ticket is below the mixture total")
}

fn game_key(slot: usize, game: u64) -> Result<u64, PpoError> {
    assert!(slot < MAX_SLOTS);
    game.checked_mul(MAX_SLOTS as u64)
        .and_then(|key| key.checked_add(slot as u64))
        .ok_or(PpoError::CounterOverflow)
}

/// Everything that determines one game before its first decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GamePlan {
    pub(crate) slot: usize,
    pub(crate) game: u64,
    /// Update whose collection started the game; fixes its spawn modifiers.
    pub(crate) start_update: u64,
    pub(crate) spec: ModifierSpec,
    pub(crate) seat: usize,
    pub(crate) opponent: OpponentKind,
    /// Decisions after which the game ends as if the tick cap were reached.
    pub(crate) decision_cap: usize,
    /// Learned-potential version shaping the game; 0 is the hand potential.
    pub(crate) potential: u64,
}

impl GamePlan {
    /// The plan of a new game, drawing its opponent from the start update's mixture.
    pub(crate) fn new(
        schedule: &GameSchedule,
        slot: usize,
        game: u64,
        next: &NextGame,
    ) -> Result<Self, PpoError> {
        let opponent = draw_opponent(&next.mixture, schedule.seed, slot, game, next.update)?;
        Ok(Self::with_opponent(
            schedule,
            slot,
            game,
            (next.update, next.spec, next.potential),
            opponent,
        ))
    }

    /// The plan of a game whose opponent is already known.
    pub(crate) fn with_opponent(
        schedule: &GameSchedule,
        slot: usize,
        game: u64,
        (start_update, spec, potential): (u64, ModifierSpec, u64),
        opponent: OpponentKind,
    ) -> Self {
        Self {
            slot,
            game,
            start_update,
            spec,
            seat: (slot + game as usize % 2) % 2,
            opponent,
            decision_cap: schedule.decision_cap,
            potential,
        }
    }

    fn seed(&self, seed: u64, domain: u64) -> Result<u64, PpoError> {
        Ok(derive_training_seed(
            seed,
            game_key(self.slot, self.game)?,
            domain,
        ))
    }
}

/// Run-wide constants every game plan derives from.
#[derive(Clone, Debug)]
pub(crate) struct GameSchedule {
    pub(crate) seed: u64,
    pub(crate) decision_cap: usize,
    pub(crate) config: PpoConfig,
    /// Learned potentials games may reference; present when the run learns one.
    pub(crate) potentials: Option<super::win_model::WinModels>,
    pub(crate) shadow: Option<ShadowLabels>,
}

/// Which rule policy labels the policy seat's retained decisions, and for which games.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShadowLabels {
    pub(crate) kind: ScriptKind,
    /// Games that start collecting at or after this update carry no labels.
    pub(crate) until_update: u64,
}

/// The actions one game has taken so far; replayed on resume.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ActionLog {
    pub(crate) policy: Vec<u32>,
    /// Neural opponents only; empty against the Teacher.
    pub(crate) opponent: Vec<u32>,
}

/// Behaviour statistics of the open retained interval at a snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OpenInterval {
    pub(crate) behaviour: u64,
    pub(crate) log_probability: f32,
    pub(crate) value: f32,
}

/// Everything needed to rebuild one in-flight game exactly.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SlotSnapshot {
    pub(crate) plan: GamePlan,
    pub(crate) log: ActionLog,
    pub(crate) actor: (u64, u64),
    pub(crate) opponent: (u64, u64),
    pub(crate) open: Option<OpenInterval>,
}

/// One prepared seat decision: its frame, packed row and legal space.
pub(super) struct PreparedSeat {
    pub(super) frame: FeatureFrame,
    pub(super) row: EncoderRow,
    pub(super) space: ActionSpace,
}

impl PreparedSeat {
    fn prepare(seat: &mut ArenaSeatPolicy) -> Result<Self, PpoError> {
        let (frame, space) = prepare_neural_seat_policy_sample(seat)?;
        if !frame.matches_action_space(&space) {
            return Err(PpoError::InvalidTransition("prepared frame action space"));
        }
        let row = EncoderRow::from_frame(&frame).map_err(text_error)?;
        Ok(Self { frame, row, space })
    }

    /// Prepares the seat's next decision, reusing this row's allocation.
    fn refresh(&mut self, seat: &mut ArenaSeatPolicy) -> Result<(), PpoError> {
        let (frame, space) = prepare_neural_seat_policy_sample(seat)?;
        if !frame.matches_action_space(&space) {
            return Err(PpoError::InvalidTransition("prepared frame action space"));
        }
        self.row.pack(&frame).map_err(text_error)?;
        self.frame = frame;
        self.space = space;
        Ok(())
    }
}

/// The update, spawn modifiers and opponent mixture a slot's next game starts under.
#[derive(Clone, Debug)]
pub(crate) struct NextGame {
    pub(crate) update: u64,
    pub(crate) spec: ModifierSpec,
    pub(crate) mixture: std::sync::Arc<OpponentMixture>,
    /// Learned-potential version of games starting now; 0 is the hand potential.
    pub(crate) potential: u64,
}

/// The sampled decision a lane hands back to its slot.
pub(super) struct Decision {
    pub(super) action: StructuredAction,
    /// Present exactly when the decision begins a retained interval: statistics,
    /// value and behaviour version. Boxed because most decisions have none.
    pub(super) retained: Option<Box<(SampledStatistics, f32, u64)>>,
    pub(super) opponent: Option<StructuredAction>,
}

/// One live game on one slot.
pub(super) struct Slot {
    pub(super) plan: GamePlan,
    environment: TrainingEnvironment,
    pub(super) stream: EpisodeStream,
    pub(super) policy: PreparedSeat,
    pub(super) opponent: Option<PreparedSeat>,
    pub(super) actor_random: PpoRng,
    pub(super) opponent_random: PpoRng,
    log: ActionLog,
}

/// A game that ended during one advance.
pub(super) struct FinishedGame {
    pub(super) slot: usize,
    pub(super) game: u64,
    pub(super) terminal: Option<PpoTransition>,
    pub(super) record: EpisodeRecord,
}

impl Slot {
    /// Builds a fresh game and prepares its first decision.
    pub(super) fn start(plan: GamePlan, schedule: &GameSchedule) -> Result<Box<Self>, PpoError> {
        let seed = schedule.seed;
        let runtime = match plan.opponent {
            OpponentKind::Teacher => OpponentRuntime::Teacher,
            OpponentKind::HarassPush => OpponentRuntime::HarassPush,
            OpponentKind::Styled(script) => OpponentRuntime::Styled(script),
            OpponentKind::Snapshot(_) | OpponentKind::SelfPlay | OpponentKind::League(_) => {
                OpponentRuntime::Neural
            }
        };
        let mut environment = build_environment(
            plan.seed(seed, crate::randomization::ARENA_DOMAIN)?,
            MapId(2),
            plan.seat,
            runtime,
            annealed::spawn_modifiers_for(plan.spec),
        )?;
        let learned = match &schedule.potentials {
            Some(models) => super::win_model::lookup(models, plan.potential)?,
            None if plan.potential == 0 => None,
            None => {
                return Err(PpoError::InvalidConfig(
                    "learned potential without a registry",
                ));
            }
        };
        let policy_seat = plan.seat;
        if let Some(labels) = schedule.shadow
            && plan.start_update < labels.until_update
        {
            enable_shadow(&mut environment.seats[policy_seat], labels.kind);
        }
        let policy = PreparedSeat::prepare(&mut environment.seats[policy_seat])?;
        let opponent = match runtime {
            OpponentRuntime::Neural => Some(PreparedSeat::prepare(
                &mut environment.seats[1 - policy_seat],
            )?),
            _ => None,
        };
        let stream = EpisodeStream::new(
            &environment,
            learned,
            schedule.potentials.is_some(),
            plan.seed(seed, RETENTION_DOMAIN)?,
        );
        Ok(Box::new(Self {
            plan,
            environment,
            stream,
            policy,
            opponent,
            actor_random: PpoRng::new(plan.seed(seed, ACTOR_DOMAIN)?),
            opponent_random: PpoRng::new(plan.seed(seed, crate::randomization::OPPONENT_DOMAIN)?),
            log: ActionLog::default(),
        }))
    }

    /// Applies one decision, advances the world one interval and prepares the
    /// next decision; a finished game is replaced by the slot's next game.
    pub(super) fn advance(
        mut self: Box<Self>,
        decision: Decision,
        schedule: &GameSchedule,
        next: NextGame,
    ) -> Result<(Box<Self>, Option<FinishedGame>), PpoError> {
        let advanced = self.step(decision, schedule.config.gamma_tick)?;
        if !advanced.done {
            self.prepare_next()?;
            return Ok((self, None));
        }
        let summary = GameSummary::capture(&self.environment, advanced.outcome, advanced.end_tick);
        let terminal = match self.stream.retained_choice() {
            Some(_) => Some(self.stream.flush(None)?),
            None => None,
        };
        let record = self.stream.record(
            self.plan.slot,
            self.plan.game,
            &advanced,
            self.plan.opponent,
            &summary,
        );
        let game = self
            .plan
            .game
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        let plan = GamePlan::new(schedule, self.plan.slot, game, &next)?;
        let replacement = Self::start(plan, schedule)?;
        let finished = FinishedGame {
            slot: self.plan.slot,
            game: self.plan.game,
            terminal,
            record,
        };
        Ok((replacement, Some(finished)))
    }

    fn step(&mut self, decision: Decision, gamma: f32) -> Result<CompletedAdvance, PpoError> {
        assert!(!self.stream.done());
        assert!(!self.stream.closes_interval(decision.action.kind()));
        let seat = &mut self.environment.seats[self.plan.seat];
        let shadow = if seat.shadow {
            Some(shadow_action(seat, &self.policy.space)?)
        } else {
            None
        };
        if self.stream.retains(decision.action.kind()) {
            let (statistics, value, behaviour) = *decision
                .retained
                .ok_or(PpoError::InvalidTransition("retained decision statistics"))?;
            let shadow = shadow
                .map(|action| ActionHeadTargets::from_sampled_action(&self.policy.space, action))
                .transpose()
                .map_err(text_error)?;
            self.stream.retain(RetainedChoice {
                frame: self.policy.frame.clone(),
                target: statistics.target,
                shadow,
                action: decision.action,
                behaviour,
                log_probability: statistics.log_probability,
                value,
            });
        } else if decision.retained.is_some() {
            return Err(PpoError::InvalidTransition(
                "unretained decision statistics",
            ));
        }
        let requests = self.requests(decision.action, decision.opponent)?;
        self.log.policy.push(encode_action(decision.action)?);
        if let Some(action) = decision.opponent {
            self.log.opponent.push(encode_action(action)?);
        }
        let tick = self.environment.seats[self.plan.seat]
            .tracker
            .current()
            .ok_or(PpoError::InvalidTransition("episode snapshot"))?
            .tick;
        assert!(tick < TICK_CAP);
        let interval = crate::MAP2_DECISION_INTERVAL_TICKS.min(TICK_CAP - tick);
        let stepped = advance_interval(&mut self.environment, requests, interval)?;
        reject_production_rejection(&self.environment, "complete episode rollout")?;
        let outcome = terminal_outcome(&self.environment, stepped.winner);
        let end_tick = tick + stepped.ticks;
        let advanced = CompletedAdvance {
            end_tick,
            ticks: stepped.ticks,
            outcome,
            done: outcome.is_some()
                || end_tick >= TICK_CAP
                || self.stream.decisions() + 1 >= self.plan.decision_cap,
        };
        self.stream.record_decision(
            &mut self.environment,
            decision.action.kind(),
            &advanced,
            gamma,
        )?;
        Ok(advanced)
    }

    fn requests(
        &mut self,
        action: StructuredAction,
        opponent: Option<StructuredAction>,
    ) -> Result<Vec<Option<Request>>, PpoError> {
        let policy_seat = self.plan.seat;
        let runtime = self.environment.opponent;
        let mut requests = Vec::with_capacity(2);
        for index in 0..2 {
            let seat = &mut self.environment.seats[index];
            let request = if index == policy_seat {
                neural_policy_request_in_space(seat, action, &self.policy.space)?.1
            } else {
                match (&self.opponent, opponent) {
                    (Some(prepared), Some(action)) => {
                        neural_policy_request_in_space(seat, action, &prepared.space)?.1
                    }
                    (None, None) => scripted_request(seat, runtime)?,
                    _ => return Err(PpoError::InvalidTransition("opponent decision")),
                }
            };
            requests.push(request);
        }
        Ok(requests)
    }

    fn prepare_next(&mut self) -> Result<(), PpoError> {
        let policy_seat = self.plan.seat;
        let seats = &mut self.environment.seats;
        self.policy.refresh(&mut seats[policy_seat])?;
        if let Some(opponent) = &mut self.opponent {
            opponent.refresh(&mut seats[1 - policy_seat])?;
        }
        Ok(())
    }

    /// The replayable state of this in-flight game.
    pub(super) fn snapshot(&self) -> SlotSnapshot {
        SlotSnapshot {
            plan: self.plan,
            log: self.log.clone(),
            actor: self.actor_random.checkpoint(),
            opponent: self.opponent_random.checkpoint(),
            open: self.stream.retained_choice().map(|choice| OpenInterval {
                behaviour: choice.behaviour,
                log_probability: choice.log_probability,
                value: choice.value,
            }),
        }
    }

    /// Rebuilds a snapshot's game by replaying its logged actions.
    pub(super) fn replay(
        snapshot: &SlotSnapshot,
        schedule: &GameSchedule,
    ) -> Result<Box<Self>, PpoError> {
        let plan = snapshot.plan;
        // The mixture that drew the opponent may be gone; everything else is rederived.
        let rederived = GamePlan::with_opponent(
            schedule,
            plan.slot,
            plan.game,
            (plan.start_update, plan.spec, plan.potential),
            plan.opponent,
        );
        if rederived != plan {
            return Err(PpoError::InvalidConfig("collector snapshot game plan"));
        }
        let mut slot = Self::start(plan, schedule)?;
        let decisions = snapshot.log.policy.len();
        let neural = slot.opponent.is_some();
        if decisions >= ACTOR_DECISIONS
            || snapshot.log.opponent.len() != if neural { decisions } else { 0 }
        {
            return Err(PpoError::InvalidConfig("collector snapshot action log"));
        }
        let last_start = last_interval_start(
            &snapshot.log.policy,
            plan.seed(schedule.seed, RETENTION_DOMAIN)?,
        )?;
        if last_start.is_some() != snapshot.open.is_some() {
            return Err(PpoError::InvalidConfig("collector snapshot open interval"));
        }
        for index in 0..decisions {
            slot.replay_decision(snapshot, index, last_start, schedule.config.gamma_tick)?;
        }
        slot.actor_random = PpoRng::from_checkpoint(snapshot.actor.0, snapshot.actor.1)?;
        slot.opponent_random = PpoRng::from_checkpoint(snapshot.opponent.0, snapshot.opponent.1)?;
        if slot.snapshot() != *snapshot {
            return Err(PpoError::InvalidConfig(
                "collector snapshot replay diverged",
            ));
        }
        Ok(slot)
    }

    fn replay_decision(
        &mut self,
        snapshot: &SlotSnapshot,
        index: usize,
        last_start: Option<usize>,
        gamma: f32,
    ) -> Result<(), PpoError> {
        let action = decode_action(snapshot.log.policy[index])?;
        if self.stream.closes_interval(action.kind()) {
            // The flushed sample belonged to an already trained update.
            self.stream.flush(Some(0.0))?;
        }
        let opponent = snapshot
            .log
            .opponent
            .get(index)
            .copied()
            .map(decode_action)
            .transpose()?;
        let retained = if self.stream.retains(action.kind()) {
            let target = ActionHeadTargets::from_sampled_action(&self.policy.space, action)
                .map_err(text_error)?;
            let open = snapshot.open.filter(|_| Some(index) == last_start);
            Some(Box::new((
                SampledStatistics {
                    target,
                    log_probability: open.map_or(0.0, |open| open.log_probability),
                },
                open.map_or(0.0, |open| open.value),
                open.map_or(0, |open| open.behaviour),
            )))
        } else {
            None
        };
        let advanced = self.step(
            Decision {
                action,
                retained,
                opponent,
            },
            gamma,
        )?;
        if advanced.done {
            return Err(PpoError::InvalidConfig(
                "collector snapshot game already ended",
            ));
        }
        self.prepare_next()
    }
}

/// The decision that began the interval still open after the logged decisions,
/// by the retention rule of [`EpisodeStream`].
fn last_interval_start(log: &[u32], retention_seed: u64) -> Result<Option<usize>, PpoError> {
    let mut start = None;
    for (index, &word) in log.iter().enumerate() {
        let continued = start.map_or(0, |start| index - start);
        if start.is_none()
            || decode_action(word)?.kind() != ActionKind::Continue
            || continued == CONTINUE_STRIDE
            || retains_continue(retention_seed, index)
        {
            start = Some(index);
        }
    }
    Ok(start)
}

/// Packs one structured action into one checked word.
///
/// Bits 0..4 hold the kind, bit 4 the unit, bits 5..13 and 13..21 two small
/// indices, bits 21..23 a target tag; every other bit stays zero.
pub(crate) fn encode_action(action: StructuredAction) -> Result<u32, PpoError> {
    let unit = action.controlled_unit().map_or(0, ControlledUnit::index) as u32;
    let (first, second, tag) = action_fields(action);
    let small = |value: usize| {
        u8::try_from(value).map_err(|_| PpoError::InvalidTransition("action log index"))
    };
    Ok(action.kind().index() as u32
        | unit << 4
        | u32::from(small(first)?) << 5
        | u32::from(small(second)?) << 13
        | tag << 21)
}

fn target_fields(target: ActionTarget) -> (usize, u32) {
    match target {
        ActionTarget::None => (0, 0),
        ActionTarget::Entity(entity) => (entity.0, 1),
        ActionTarget::Point(point) => (point.0, 2),
    }
}

fn action_fields(action: StructuredAction) -> (usize, usize, u32) {
    use StructuredAction as Action;
    match action {
        Action::Continue | Action::Stop { .. } | Action::Hold { .. } => (0, 0, 0),
        Action::MovePoint { point, .. } | Action::AttackMovePoint { point, .. } => (point.0, 0, 0),
        Action::FollowUnit { target, .. } | Action::AttackUnit { target, .. } => (target.0, 0, 0),
        Action::Cast { slot, target, .. } => {
            let (index, tag) = target_fields(target);
            (usize::from(slot.0), index, tag)
        }
        Action::Use { slot, target, .. } => {
            let (index, tag) = target_fields(target);
            (usize::from(slot.0), index, tag)
        }
        Action::PutPoint { source, target, .. } => match target {
            PutPointTarget::Underfoot => (usize::from(source.0), 0, 0),
            PutPointTarget::Point(point) => (usize::from(source.0), point.0, 2),
        },
        Action::PutUnit { source, target, .. } => (usize::from(source.0), target.0, 1),
        Action::Take { loot, .. } => (loot.0, 0, 0),
        Action::Buy { item, .. } => (item.0, 0, 0),
        Action::Sell { slot, .. } => (usize::from(slot.0), 0, 0),
        Action::Swap { from, to, .. } => (usize::from(from.0), usize::from(to.0), 0),
        Action::Learn { slot } => (usize::from(slot.0), 0, 0),
    }
}

/// Inverts [`encode_action`], rejecting any word it cannot produce.
pub(crate) fn decode_action(word: u32) -> Result<StructuredAction, PpoError> {
    let invalid = PpoError::InvalidConfig("action log word");
    let fields = ActionWord {
        kind: ActionKind::from_index((word & 0xf) as usize).ok_or(invalid.clone())?,
        unit: if word >> 4 & 1 == 0 {
            ControlledUnit::Hero
        } else {
            ControlledUnit::Courier
        },
        first: (word >> 5 & 0xff) as usize,
        second: (word >> 13 & 0xff) as usize,
        tag: word >> 21,
    };
    let action = fields
        .unit_action()
        .or_else(|| fields.item_action())
        .ok_or(invalid.clone())?;
    if encode_action(action)? != word {
        return Err(invalid);
    }
    Ok(action)
}

/// The fields of one action log word before they are checked as an action.
struct ActionWord {
    kind: ActionKind,
    unit: ControlledUnit,
    first: usize,
    second: usize,
    tag: u32,
}

impl ActionWord {
    fn target(&self) -> Option<ActionTarget> {
        match self.tag {
            0 => Some(ActionTarget::None),
            1 => Some(ActionTarget::Entity(EntityIndex(self.second))),
            2 => Some(ActionTarget::Point(PointIndex(self.second))),
            _ => None,
        }
    }

    /// Kinds that name no item slot.
    fn unit_action(&self) -> Option<StructuredAction> {
        use StructuredAction as Action;
        let (unit, first) = (self.unit, self.first);
        Some(match self.kind {
            ActionKind::Continue => Action::Continue,
            ActionKind::Stop => Action::Stop { unit },
            ActionKind::MovePoint => Action::MovePoint {
                unit,
                point: PointIndex(first),
            },
            ActionKind::FollowUnit => Action::FollowUnit {
                unit,
                target: EntityIndex(first),
            },
            ActionKind::Hold => Action::Hold { unit },
            ActionKind::AttackMovePoint => Action::AttackMovePoint {
                unit,
                point: PointIndex(first),
            },
            ActionKind::AttackUnit => Action::AttackUnit {
                unit,
                target: EntityIndex(first),
            },
            ActionKind::Cast => Action::Cast {
                unit,
                slot: bota_proto::AbilitySlot(u8::try_from(first).ok()?),
                target: self.target()?,
            },
            ActionKind::Take => Action::Take {
                unit,
                loot: LootIndex(first),
            },
            ActionKind::Buy => Action::Buy {
                unit,
                item: ShopIndex(first),
            },
            ActionKind::Learn => Action::Learn {
                slot: bota_proto::AbilitySlot(u8::try_from(first).ok()?),
            },
            _ => return None,
        })
    }

    /// Kinds that name an item slot.
    fn item_action(&self) -> Option<StructuredAction> {
        use StructuredAction as Action;
        let unit = self.unit;
        let slot = bota_proto::ItemSlot(u8::try_from(self.first).ok()?);
        let second = bota_proto::ItemSlot(u8::try_from(self.second).ok()?);
        Some(match self.kind {
            ActionKind::Use => Action::Use {
                unit,
                slot,
                target: self.target()?,
            },
            ActionKind::PutPoint => Action::PutPoint {
                unit,
                source: slot,
                target: match self.tag {
                    0 => PutPointTarget::Underfoot,
                    2 => PutPointTarget::Point(PointIndex(self.second)),
                    _ => return None,
                },
            },
            ActionKind::PutUnit => Action::PutUnit {
                unit,
                source: slot,
                target: EntityIndex(self.second),
            },
            ActionKind::Sell => Action::Sell { unit, slot },
            ActionKind::Swap => Action::Swap {
                unit,
                from: slot,
                to: second,
            },
            _ => return None,
        })
    }
}

#[cfg(test)]
#[path = "../tests/slot_action_log.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/slot_episode.rs"]
mod episode_tests;
