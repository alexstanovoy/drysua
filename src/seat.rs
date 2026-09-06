use std::{io::Read, path::Path};

use bota_proto::{MatchInfo, RejectReason, ServerMsg, SlotId, Team, TickMode};

use crate::{
    ActionSpace, ActiveOrderUpdate, ActivePolicyOrder, FeatureEncoder, FeatureFrame, ItemReadiness,
    Link, LocalPolicyState, OrderPersistence, PolicyModel, SHADOW_FIEND, Seated, StateTracker,
    StructuredAction, TACTICAL_FILE_BYTES, TacticalPolicy, Teacher, TrainingArtifact, Wire,
    active_order_update_for_sent,
};

const MAX_MATCH_MESSAGES: usize = 16_777_216;
const MAX_MESSAGES_WITHOUT_SNAPSHOT: usize = 4_096;

/// Canonical deployment artifact for [`play_tactical`], distinct from Hybrid SafeTensors.
pub const TACTICAL_FILE_NAME: &str = "drysua.tactical.bin";

/// The result observed by drysua for one match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The seat held by drysua.
    pub slot: Option<SlotId>,
    /// The team of the held seat.
    pub team: Option<Team>,
    /// The winning team, or no team when a tick limit ended play.
    pub winner: Option<Team>,
    /// The latest snapshot tick received.
    pub ticks: u32,
    /// Rejected orders reported by the server.
    pub rejections: u32,
    /// The latest rejection reason, or no reason when none was rejected.
    pub last_rejection: Option<RejectReason>,
    /// Controller decisions completed at policy ticks.
    pub decisions: u32,
    /// Orders written to the connection after local deduplication.
    pub orders: u32,
}

struct LivePolicy {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    readiness: ItemReadiness,
    teacher: Teacher,
    last_decision_tick: Option<u32>,
    pending_snapshot_tick: Option<u32>,
    pending_active: Option<(u32, Option<ActivePolicyOrder>)>,
}

#[derive(Clone, Copy)]
enum LiveController<'model> {
    Hybrid(&'model PolicyModel),
    Tactical(&'model TacticalPolicy),
    Teacher,
}

/// Loads bounded tactical weights before connecting, without constructing a tensor model.
pub fn play_tactical(
    address: &str,
    name: &str,
    limit: Option<u32>,
    weights_directory: &Path,
) -> std::io::Result<Outcome> {
    let policy = load_tactical_policy(weights_directory)?;
    let (mut link, seated) = Link::join(address, name)?;
    play_tactical_on(&mut link, seated, limit, &policy)
}

/// Runs a checked tactical policy through the same live order/ACK state machine as Teacher.
pub fn play_tactical_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
    policy: &TacticalPolicy,
) -> std::io::Result<Outcome> {
    play_controller_on(wire, seated, limit, LiveController::Tactical(policy))
}

fn load_tactical_policy(directory: &Path) -> std::io::Result<TacticalPolicy> {
    let path = directory.join(TACTICAL_FILE_NAME);
    let context = |error: std::io::Error| {
        std::io::Error::new(
            error.kind(),
            format!("tactical weights {}: {error}", path.display()),
        )
    };
    let metadata = std::fs::symlink_metadata(&path).map_err(context)?;
    if !metadata.file_type().is_file() {
        return Err(context(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected a regular non-symlink file",
        )));
    }
    let mut bytes = Vec::with_capacity(TACTICAL_FILE_BYTES + 1);
    std::fs::File::open(&path)
        .map_err(context)?
        .take((TACTICAL_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(context)?;
    assert!(bytes.len() <= TACTICAL_FILE_BYTES + 1);
    let policy = TacticalPolicy::from_bytes(&bytes)
        .map_err(|error| context(std::io::Error::new(std::io::ErrorKind::InvalidData, error)))?;
    assert_eq!(bytes.len(), TACTICAL_FILE_BYTES);
    Ok(policy)
}

/// Connects and runs the deterministic Teacher on any supported map without weights.
pub fn play_teacher(address: &str, name: &str, limit: Option<u32>) -> std::io::Result<Outcome> {
    let (mut link, seated) = Link::join(address, name)?;
    play_teacher_on(&mut link, seated, limit)
}

/// Runs the deterministic Teacher on an assigned match connection without a model.
pub fn play_teacher_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
) -> std::io::Result<Outcome> {
    play_controller_on(wire, seated, limit, LiveController::Teacher)
}

/// Loads deployment weights, connects, and runs greedy policy inference for one match.
pub fn play(
    address: &str,
    name: &str,
    limit: Option<u32>,
    weights_directory: &Path,
) -> std::io::Result<Outcome> {
    let model = PolicyModel::fresh(0).map_err(std::io::Error::other)?;
    TrainingArtifact::load_runtime_weights(&model, weights_directory)
        .map_err(std::io::Error::other)?;
    let (mut link, seated) = Link::join(address, name)?;
    play_policy_on(&mut link, seated, limit, &model)
}

/// Runs greedy model inference on an assigned match connection.
pub fn play_policy_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
    model: &PolicyModel,
) -> std::io::Result<Outcome> {
    play_controller_on(wire, seated, limit, LiveController::Hybrid(model))
}

fn play_controller_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
    controller: LiveController<'_>,
) -> std::io::Result<Outcome> {
    let mut outcome = Outcome {
        slot: Some(seated.slot),
        ..Outcome::default()
    };
    let mut policy = None;
    let mut progress = MessageProgress::new();
    for _ in 0..MAX_MATCH_MESSAGES {
        let Some(message) = wire.hear()? else {
            return Err(std::io::Error::other(
                "server closed the connection before MatchOver",
            ));
        };
        progress.observe(&message)?;
        if handle_policy_message(
            wire,
            seated,
            limit,
            controller,
            &mut outcome,
            &mut policy,
            message,
        )? {
            return Ok(outcome);
        }
    }
    Err(std::io::Error::other("server match message limit exceeded"))
}

#[allow(clippy::too_many_arguments)]
fn handle_policy_message(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
    controller: LiveController<'_>,
    outcome: &mut Outcome,
    policy: &mut Option<LivePolicy>,
    message: ServerMsg,
) -> std::io::Result<bool> {
    match message {
        ServerMsg::MatchStart { info } => start_policy_match(seated, outcome, policy, &info)?,
        ServerMsg::Snapshot { view } => {
            return handle_policy_snapshot(wire, seated, limit, outcome, policy, view);
        }
        ServerMsg::OrderRejected { seq, reason } => {
            record_rejection(outcome, reason)?;
            if let Some(policy) = policy {
                policy.observe_rejection(seq)?;
            }
        }
        ServerMsg::MatchOver { winner, .. } => {
            if outcome.team.is_none() {
                return Err(std::io::Error::other(
                    "server sent MatchOver before MatchStart",
                ));
            }
            if policy
                .as_ref()
                .is_some_and(|policy| policy.pending_snapshot_tick.is_some())
            {
                return Err(std::io::Error::other(
                    "server ended the match before completing the snapshot tick",
                ));
            }
            outcome.winner = Some(winner);
            return Ok(true);
        }
        ServerMsg::Events { tick, events } => {
            handle_policy_events(wire, seated, controller, outcome, policy, tick, &events)?;
        }
        ServerMsg::Welcome { .. }
        | ServerMsg::LobbyState { .. }
        | ServerMsg::Orders { .. }
        | ServerMsg::ParticipantLeft { .. } => {}
    }
    Ok(false)
}

fn start_policy_match(
    seated: Seated,
    outcome: &mut Outcome,
    policy: &mut Option<LivePolicy>,
    info: &MatchInfo,
) -> std::io::Result<()> {
    validate_match_terms(info.tick_rate, info.mode, seated)?;
    outcome.team = Some(validate_pick(&info.picks, seated.slot)?);
    if policy.is_some() {
        return Err(std::io::Error::other("server sent repeated MatchStart"));
    }
    *policy = Some(LivePolicy::new(seated.slot, info)?);
    Ok(())
}

fn handle_policy_snapshot(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
    outcome: &mut Outcome,
    policy: &mut Option<LivePolicy>,
    view: bota_proto::WorldView,
) -> std::io::Result<bool> {
    let policy = policy
        .as_mut()
        .ok_or_else(|| std::io::Error::other("server sent Snapshot before MatchStart"))?;
    validate_snapshot(outcome, view.viewer, view.tick)?;
    outcome.ticks = view.tick;
    policy.observe_snapshot(&view)?;
    let finished = limit.is_some_and(|last_tick| view.tick >= last_tick);
    if finished && seated.mode == TickMode::Lockstep {
        wire.acknowledge(view.tick)?;
    }
    if !finished {
        policy.begin_snapshot_tick(view.tick)?;
    }
    Ok(finished)
}

fn handle_policy_events(
    wire: &mut impl Wire,
    seated: Seated,
    controller: LiveController<'_>,
    outcome: &mut Outcome,
    policy: &mut Option<LivePolicy>,
    tick: u32,
    events: &[bota_proto::EventKind],
) -> std::io::Result<()> {
    let policy = policy
        .as_mut()
        .ok_or_else(|| std::io::Error::other("server sent Events before MatchStart"))?;
    policy.observe_events(tick, events)?;
    policy.complete_snapshot_tick(wire, controller, outcome, tick)?;
    if seated.mode == TickMode::Lockstep {
        wire.acknowledge(tick)?;
    }
    Ok(())
}

fn record_rejection(outcome: &mut Outcome, reason: RejectReason) -> std::io::Result<()> {
    outcome.rejections = outcome
        .rejections
        .checked_add(1)
        .ok_or_else(|| std::io::Error::other("order rejection count overflowed"))?;
    outcome.last_rejection = Some(reason);
    Ok(())
}

/// Follows the explicit Continue policy for protocol-only tests.
pub fn play_idle_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
) -> std::io::Result<Outcome> {
    let mut outcome = Outcome {
        slot: Some(seated.slot),
        ..Outcome::default()
    };
    let mut progress = MessageProgress::new();
    for _ in 0..MAX_MATCH_MESSAGES {
        let Some(message) = wire.hear()? else {
            return Err(std::io::Error::other(
                "server closed the connection before MatchOver",
            ));
        };
        progress.observe(&message)?;
        match message {
            ServerMsg::MatchStart { info } => {
                validate_match_terms(info.tick_rate, info.mode, seated)?;
                outcome.team = Some(validate_pick(&info.picks, seated.slot)?);
            }
            ServerMsg::Snapshot { view } => {
                validate_snapshot(&outcome, view.viewer, view.tick)?;
                outcome.ticks = view.tick;
                if seated.mode == TickMode::Lockstep {
                    wire.acknowledge(view.tick)?;
                }
                if limit.is_some_and(|last_tick| view.tick >= last_tick) {
                    return Ok(outcome);
                }
            }
            ServerMsg::OrderRejected { reason, .. } => {
                outcome.rejections = outcome
                    .rejections
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("order rejection count overflowed"))?;
                outcome.last_rejection = Some(reason);
            }
            ServerMsg::MatchOver { winner, .. } => {
                if outcome.team.is_none() {
                    return Err(std::io::Error::other(
                        "server sent MatchOver before MatchStart",
                    ));
                }
                outcome.winner = Some(winner);
                return Ok(outcome);
            }
            ServerMsg::Welcome { .. }
            | ServerMsg::LobbyState { .. }
            | ServerMsg::Events { .. }
            | ServerMsg::Orders { .. }
            | ServerMsg::ParticipantLeft { .. } => {}
        }
    }
    Err(std::io::Error::other("server match message limit exceeded"))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MessageProgress {
    without_snapshot: usize,
}

impl MessageProgress {
    const fn new() -> Self {
        Self {
            without_snapshot: 0,
        }
    }

    fn observe(&mut self, message: &ServerMsg) -> std::io::Result<()> {
        if matches!(message, ServerMsg::Snapshot { .. }) {
            self.without_snapshot = 0;
            return Ok(());
        }
        self.without_snapshot = self
            .without_snapshot
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("server no-snapshot counter overflowed"))?;
        if self.without_snapshot > MAX_MESSAGES_WITHOUT_SNAPSHOT {
            return Err(std::io::Error::other(
                "server sent too many messages without a snapshot",
            ));
        }
        Ok(())
    }
}

impl LivePolicy {
    fn new(slot: SlotId, info: &MatchInfo) -> std::io::Result<Self> {
        let tracker = StateTracker::new(slot, info).map_err(std::io::Error::other)?;
        let encoder = FeatureEncoder::new(&tracker);
        Ok(Self {
            tracker,
            encoder,
            local: LocalPolicyState::new(0),
            persistence: OrderPersistence::default(),
            readiness: ItemReadiness::new(),
            teacher: Teacher::new(),
            last_decision_tick: None,
            pending_snapshot_tick: None,
            pending_active: None,
        })
    }

    fn should_decide(&self, tick: u32) -> std::io::Result<bool> {
        let Some(previous) = self.last_decision_tick else {
            return Ok(true);
        };
        let elapsed = tick
            .checked_sub(previous)
            .ok_or_else(|| std::io::Error::other("policy decision tick regressed"))?;
        Ok(elapsed >= 3)
    }

    fn begin_snapshot_tick(&mut self, tick: u32) -> std::io::Result<()> {
        if self.pending_snapshot_tick.is_some() {
            return Err(std::io::Error::other(
                "server sent Snapshot before completing the previous tick",
            ));
        }
        let current_tick = self
            .tracker
            .current()
            .ok_or_else(|| std::io::Error::other("policy snapshot is missing"))?
            .tick;
        if tick != current_tick {
            return Err(std::io::Error::other("policy snapshot tick mismatch"));
        }
        self.pending_snapshot_tick = Some(tick);
        Ok(())
    }

    fn complete_snapshot_tick(
        &mut self,
        wire: &mut impl Wire,
        controller: LiveController<'_>,
        outcome: &mut Outcome,
        tick: u32,
    ) -> std::io::Result<()> {
        if self.pending_snapshot_tick != Some(tick) {
            return Err(std::io::Error::other(
                "server Events did not complete the pending snapshot tick",
            ));
        }
        self.pending_snapshot_tick = None;
        if self.should_decide(tick)? {
            self.decide(wire, controller, outcome)?;
        }
        Ok(())
    }

    fn observe_snapshot(&mut self, view: &bota_proto::WorldView) -> std::io::Result<()> {
        let previous = self.tracker.own_hero().map(|hero| hero.id);
        self.tracker
            .observe_snapshot(view)
            .map_err(std::io::Error::other)?;
        let current = self.tracker.own_hero().map(|hero| hero.id);
        if previous != current {
            self.persistence.clear_body_for(None);
            self.local
                .set_active_order(view.tick, None)
                .map_err(std::io::Error::other)?;
            self.pending_active = None;
            assert!(self.persistence.active_body_sequence_for(None).is_none());
            assert!(self.local.active_order().is_none());
        }
        Ok(())
    }

    fn observe_events(
        &mut self,
        tick: u32,
        events: &[bota_proto::EventKind],
    ) -> std::io::Result<()> {
        self.tracker
            .observe_events(tick, events)
            .map_err(std::io::Error::other)?;
        self.encoder
            .observe(&self.tracker)
            .map_err(std::io::Error::other)
    }

    fn decide(
        &mut self,
        wire: &mut impl Wire,
        controller: LiveController<'_>,
        outcome: &mut Outcome,
    ) -> std::io::Result<()> {
        let (action, space) = self.select_action(controller)?;
        self.local
            .note_decision(space.tick(), action.kind())
            .map_err(std::io::Error::other)?;
        self.last_decision_tick = Some(space.tick());
        outcome.decisions = outcome
            .decisions
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("policy decision count overflowed"))?;
        let decoded = space.decode(action).map_err(std::io::Error::other)?;
        let Some(issued) = self.persistence.should_send(decoded) else {
            return Ok(());
        };
        let previous = self.local.active_order();
        let sequence = wire.order(issued.unit, issued.order)?;
        self.persistence
            .record_sent(sequence, issued)
            .map_err(std::io::Error::other)?;
        self.readiness.note_sent(sequence, issued, &space);
        self.teacher.note_sent(sequence, issued, space.tick());
        let update =
            active_order_update_for_sent(&self.persistence, issued.unit, sequence, action.kind());
        self.pending_active = match update {
            ActiveOrderUpdate::Preserve => self.pending_active,
            ActiveOrderUpdate::Replace(None) if previous.is_none() => None,
            ActiveOrderUpdate::Replace(None) => {
                self.local
                    .set_active_order(space.tick(), None)
                    .map_err(std::io::Error::other)?;
                Some((sequence, previous))
            }
            ActiveOrderUpdate::Replace(Some(kind)) => {
                self.local
                    .set_active_order_from_issued(space.tick(), kind, issued)
                    .map_err(std::io::Error::other)?;
                Some((sequence, previous))
            }
        };
        outcome.orders = outcome
            .orders
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("policy order count overflowed"))?;
        Ok(())
    }

    fn select_action(
        &mut self,
        controller: LiveController<'_>,
    ) -> std::io::Result<(StructuredAction, ActionSpace)> {
        let model = match controller {
            LiveController::Tactical(policy) => {
                return self
                    .teacher
                    .decide_tactical(&self.tracker, &self.persistence, &self.readiness, policy)
                    .map_err(std::io::Error::other);
            }
            LiveController::Hybrid(model)
                if self.tracker.metadata().map != bota_proto::MapId(0) =>
            {
                model
            }
            LiveController::Hybrid(_) | LiveController::Teacher => {
                return self
                    .teacher
                    .decide(&self.tracker, &self.persistence, &self.readiness)
                    .map_err(std::io::Error::other);
            }
        };
        let space = ActionSpace::from_tracker_with_readiness(&self.tracker, &self.readiness)
            .map_err(std::io::Error::other)?;
        let mut frame = FeatureFrame::new();
        self.encoder
            .encode(
                &self.tracker,
                &space,
                &self.readiness,
                &self.local,
                &mut frame,
            )
            .map_err(std::io::Error::other)?;
        let proposed = model
            .choose(&frame, &space)
            .map_err(std::io::Error::other)?
            .action;
        let action = self
            .teacher
            .deployment_action(&self.tracker, &space)
            .unwrap_or(proposed);
        Ok((action, space))
    }

    fn observe_rejection(&mut self, sequence: u32) -> std::io::Result<()> {
        self.persistence.observe_rejection(sequence);
        self.readiness.note_rejected(sequence);
        self.teacher.note_rejected(sequence);
        if let Some((pending, previous)) = self.pending_active
            && pending == sequence
        {
            let tick = self
                .tracker
                .current()
                .ok_or_else(|| std::io::Error::other("rejection before policy snapshot"))?
                .tick;
            self.local
                .restore_active_order(tick, previous)
                .map_err(std::io::Error::other)?;
            self.pending_active = None;
        }
        Ok(())
    }
}

fn validate_snapshot(outcome: &Outcome, viewer: Option<Team>, tick: u32) -> std::io::Result<()> {
    if outcome.team.is_none() {
        return Err(std::io::Error::other(
            "server sent Snapshot before MatchStart",
        ));
    }
    if viewer != outcome.team {
        return Err(std::io::Error::other(format!(
            "Snapshot viewer {viewer:?} differs from assigned team {:?}",
            outcome.team
        )));
    }
    if tick <= outcome.ticks {
        return Err(std::io::Error::other(format!(
            "Snapshot tick {tick} does not follow tick {}",
            outcome.ticks
        )));
    }
    Ok(())
}

fn validate_match_terms(tick_rate: u16, mode: TickMode, seated: Seated) -> std::io::Result<()> {
    if tick_rate != seated.tick_rate {
        return Err(std::io::Error::other(format!(
            "MatchStart tick rate {tick_rate} differs from Welcome tick rate {}",
            seated.tick_rate
        )));
    }
    if mode != seated.mode {
        return Err(std::io::Error::other(format!(
            "MatchStart mode {mode:?} differs from Welcome mode {:?}",
            seated.mode
        )));
    }
    Ok(())
}

fn validate_pick(picks: &[bota_proto::Pick], slot: SlotId) -> std::io::Result<Team> {
    let pick = picks.iter().find(|pick| pick.slot == slot).ok_or_else(|| {
        std::io::Error::other(format!(
            "MatchStart has no hero pick for assigned slot {}",
            slot.0
        ))
    })?;
    if pick.hero != SHADOW_FIEND {
        return Err(std::io::Error::other(format!(
            "assigned slot {} picked HeroId({}), expected Shadow Fiend HeroId({})",
            slot.0, pick.hero.0, SHADOW_FIEND.0
        )));
    }
    Ok(pick.team)
}
