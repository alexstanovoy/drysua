//! Native head-to-head matches between two rule policies on the builtin Map2 arena.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use bota_proto::{DamageKind, EventKind, Fixed, MapId, ServerMsg, SlotId, Team, UnitKind, Vec2};

use crate::raze_aim::{SHADOWRAZE_RADIUS, SHADOWRAZES, isqrt, raze_reach};
use crate::scripted::tactics::{DECISION_TICKS, own_fountain};
use crate::scripted::{ScriptKind, ScriptedPolicy};
use crate::{
    Arena, ArenaConfig, ItemReadiness, OrderPersistence, RazeAim, Request, StateTracker,
    tracker::map_maximum_raw,
};

/// Upper bound on duel worker threads, far above any machine this runs on.
pub const MAX_DUEL_THREADS: usize = 256;
/// Upper bound on seeds per duel; each seed is played from both sides.
pub const MAX_DUEL_SEEDS: u32 = 10_000;
/// Longer than any hero attack cycle, so an attacking hero never counts as idle.
const IDLE_ATTACK_TICKS: u32 = 60;
/// Ticks after which a Map2 match cannot still be running.
const MATCH_TICK_LIMIT: u32 = 30 * 60 * 16;

/// Which seeds and policies a duel plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuelConfig {
    /// The evaluated policy.
    pub policy: ScriptKind,
    /// The policy it plays against.
    pub opponent: ScriptKind,
    /// First arena seed.
    pub first_seed: u64,
    /// Consecutive seeds, each played once per side.
    pub seeds: u32,
    /// Worker threads, 1 through [`MAX_DUEL_THREADS`].
    pub threads: usize,
}

/// Result of one duel game for the evaluated policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuelResult {
    Win,
    Loss,
    Draw,
}

/// Why a duel game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuelEnd {
    /// A side reached its second hero death.
    Kills,
    /// A side lost a tower.
    Tower,
    /// Both sides lost on the same tick.
    Simultaneous,
    /// The 15-minute game cap.
    Cap,
}

/// Seat-observed statistics of one finished duel game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuelGame {
    pub seed: u64,
    /// Team of the evaluated policy.
    pub side: Team,
    pub result: DuelResult,
    pub end: DuelEnd,
    pub ticks: u32,
    pub policy_deaths: u16,
    pub opponent_deaths: u16,
    /// Ticks each side spent dead or in the quarter of the map around its fountain.
    pub policy_home_ticks: u32,
    pub opponent_home_ticks: u32,
    /// Longest run of ticks each live hero neither moved nor dealt damage away from home;
    /// a frozen controller shows up here long before it loses.
    pub policy_longest_idle: u32,
    pub opponent_longest_idle: u32,
    /// Razes cast by the evaluated policy with the enemy hero visible within raze reach.
    pub policy_raze_casts: u32,
    /// Razes of the evaluated policy that damaged the enemy hero.
    pub policy_raze_hits: u32,
    pub opponent_raze_casts: u32,
    pub opponent_raze_hits: u32,
    pub policy_level: u8,
    pub opponent_level: u8,
    /// Damage each hero dealt to the enemy hero.
    pub policy_hero_damage: i32,
    pub opponent_hero_damage: i32,
    /// Damage each hero dealt to enemy towers.
    pub policy_tower_damage: i32,
    pub opponent_tower_damage: i32,
    /// Orders each seat sent; a rule policy's order churn is what a clone must imitate.
    pub policy_orders: u32,
    pub opponent_orders: u32,
    pub rejections: u32,
}

/// Plays every configured seed from both sides; results are in seed then side order.
pub fn run_duel(config: DuelConfig) -> Result<Vec<DuelGame>, String> {
    if !(1..=MAX_DUEL_THREADS).contains(&config.threads) {
        return Err(format!(
            "duel threads must be 1..={MAX_DUEL_THREADS}, got {}",
            config.threads
        ));
    }
    if !(1..=MAX_DUEL_SEEDS).contains(&config.seeds) {
        return Err(format!(
            "duel seeds must be 1..={MAX_DUEL_SEEDS}, got {}",
            config.seeds
        ));
    }
    config
        .first_seed
        .checked_add(u64::from(config.seeds))
        .ok_or("duel seed range overflows")?;
    let games = config.seeds as usize * 2;
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; games]);
    std::thread::scope(|scope| {
        for _ in 0..config.threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= games {
                        break;
                    }
                    let seed = config.first_seed + (index / 2) as u64;
                    let game = play_duel_game(config.policy, config.opponent, seed, index % 2);
                    results.lock().expect("duel results lock")[index] = Some(game);
                }
            });
        }
    });
    results
        .into_inner()
        .expect("duel results lock")
        .into_iter()
        .map(|game| game.expect("every duel game ran"))
        .collect()
}

/// Plays one deterministic Map2 game with the evaluated policy in `policy_seat`.
pub fn play_duel_game(
    policy: ScriptKind,
    opponent: ScriptKind,
    seed: u64,
    policy_seat: usize,
) -> Result<DuelGame, String> {
    if policy_seat > 1 {
        return Err(format!(
            "duel policy seat must be 0 or 1, got {policy_seat}"
        ));
    }
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed,
    })
    .map_err(|error| error.to_string())?;
    let mut seats = [None, None];
    for (index, messages) in start.messages.into_iter().enumerate() {
        let kind = if index == policy_seat {
            policy
        } else {
            opponent
        };
        seats[index] = Some(DuelSeat::new(kind, index, &messages)?);
    }
    let mut seats = seats.map(|seat| seat.expect("both duel seats start"));
    let mut tally = Tally::new(seed, seats[policy_seat].tracker.team());
    loop {
        let tick = arena.tick();
        if tick > MATCH_TICK_LIMIT {
            return Err(format!("duel seed {seed} ran past tick {MATCH_TICK_LIMIT}"));
        }
        let mut requests = [None, None];
        if (tick - 1).is_multiple_of(DECISION_TICKS) {
            for (request, seat) in requests.iter_mut().zip(&mut seats) {
                *request = seat.decide()?;
            }
        }
        let step = arena.step(&requests).map_err(|error| error.to_string())?;
        let mut winner = None;
        for (index, (seat, messages)) in seats.iter_mut().zip(step.messages).enumerate() {
            tally.observe(index == policy_seat, &seat.tracker, &messages);
            winner = seat.observe(messages)?.or(winner);
        }
        tally.observe_home(&seats[policy_seat].tracker, &seats[1 - policy_seat].tracker);
        if let Some(winner) = winner {
            let rejections = seats.iter().map(|seat| seat.rejections).sum();
            return Ok(tally.finish(winner, arena.tick(), rejections, &seats, policy_seat));
        }
    }
}

struct DuelSeat {
    script: ScriptedPolicy,
    tracker: StateTracker,
    persistence: OrderPersistence,
    readiness: ItemReadiness,
    aim: RazeAim,
    sequence: u32,
    rejections: u32,
}

impl DuelSeat {
    fn new(kind: ScriptKind, index: usize, messages: &[ServerMsg]) -> Result<Self, String> {
        let info = messages.iter().find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info),
            _ => None,
        });
        let slot = SlotId(u8::try_from(index).map_err(|error| error.to_string())?);
        let tracker = StateTracker::new(slot, info.ok_or("duel seat has no MatchStart")?)
            .map_err(|error| error.to_string())?;
        let mut seat = Self {
            script: ScriptedPolicy::new(kind),
            tracker,
            persistence: OrderPersistence::default(),
            readiness: ItemReadiness::new(),
            aim: RazeAim::default(),
            sequence: 0,
            rejections: 0,
        };
        seat.observe(messages.to_vec())?;
        Ok(seat)
    }

    fn decide(&mut self) -> Result<Option<Request>, String> {
        if self.tracker.own_hero().is_none() {
            return Ok(None);
        }
        let (action, space) = self
            .script
            .decide(&self.tracker, &self.persistence, &self.readiness)
            .map_err(|error| error.to_string())?;
        let decoded = space.decode(action).map_err(|error| error.to_string())?;
        // Aimed raze intents expand into wire orders exactly as in training and live play.
        let active_body = self.persistence.active_body_order_for(None);
        let Some((issued, _)) =
            self.aim
                .resolve(&self.tracker, decoded, action.kind(), active_body)
        else {
            return Ok(None);
        };
        let Some(issued) = self.persistence.should_send(Some(issued)) else {
            return Ok(None);
        };
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or("duel sequence overflow")?;
        self.persistence
            .record_sent(self.sequence, issued)
            .map_err(|error| error.to_string())?;
        self.readiness.note_sent(self.sequence, issued, &space);
        self.script.note_sent(self.sequence, issued, space.tick());
        Ok(Some(Request {
            seq: self.sequence,
            unit: issued.unit,
            order: issued.order,
        }))
    }

    fn observe(&mut self, messages: Vec<ServerMsg>) -> Result<Option<Team>, String> {
        let mut winner = None;
        for message in messages {
            match message {
                ServerMsg::OrderRejected { seq, .. } => {
                    self.persistence.observe_rejection(seq);
                    self.readiness.note_rejected(seq);
                    self.script.note_rejected(seq);
                    self.rejections = self.rejections.saturating_add(1);
                }
                ServerMsg::Snapshot { view } => {
                    let previous = self.tracker.own_hero().map(|hero| hero.id);
                    self.tracker
                        .observe_snapshot_owned(view)
                        .map_err(|error| error.to_string())?;
                    if self.tracker.own_hero().map(|hero| hero.id) != previous {
                        self.persistence.clear_body_for(None);
                    }
                }
                ServerMsg::Events { tick, events } => self
                    .tracker
                    .observe_events(tick, &events)
                    .map_err(|error| error.to_string())?,
                ServerMsg::MatchOver { winner: result, .. } => winner = Some(result),
                _ => {}
            }
        }
        Ok(winner)
    }
}

/// Counters gathered from each seat's own message stream.
struct Tally {
    game: DuelGame,
    tower_lost: bool,
    /// Current run of still ticks per side, policy first.
    idle: [u32; 2],
}

impl Tally {
    const fn new(seed: u64, side: Team) -> Self {
        Self {
            game: DuelGame {
                seed,
                side,
                result: DuelResult::Draw,
                end: DuelEnd::Cap,
                ticks: 0,
                policy_deaths: 0,
                opponent_deaths: 0,
                policy_home_ticks: 0,
                opponent_home_ticks: 0,
                policy_longest_idle: 0,
                opponent_longest_idle: 0,
                policy_raze_casts: 0,
                policy_raze_hits: 0,
                opponent_raze_casts: 0,
                opponent_raze_hits: 0,
                policy_level: 0,
                opponent_level: 0,
                policy_hero_damage: 0,
                opponent_hero_damage: 0,
                policy_tower_damage: 0,
                opponent_tower_damage: 0,
                policy_orders: 0,
                opponent_orders: 0,
                rejections: 0,
            },
            tower_lost: false,
            idle: [0; 2],
        }
    }

    fn observe(&mut self, policy: bool, tracker: &StateTracker, messages: &[ServerMsg]) {
        let Some(hero) = tracker.own_hero() else {
            return;
        };
        let own = hero.id;
        let reach = Fixed::from_int(SHADOWRAZES[2].1 + SHADOWRAZE_RADIUS);
        let aimed = tracker.current().is_some_and(|view| {
            view.units.iter().any(|unit| {
                unit.kind == UnitKind::Hero
                    && unit.team != tracker.team()
                    && unit.hp > 0
                    && unit.pos.within(hero.pos, reach)
            })
        });
        let (game, tower_lost) = (&mut self.game, &mut self.tower_lost);
        let (casts, hits, hero_damage, tower_damage) = if policy {
            (
                &mut game.policy_raze_casts,
                &mut game.policy_raze_hits,
                &mut game.policy_hero_damage,
                &mut game.policy_tower_damage,
            )
        } else {
            (
                &mut game.opponent_raze_casts,
                &mut game.opponent_raze_hits,
                &mut game.opponent_hero_damage,
                &mut game.opponent_tower_damage,
            )
        };
        for message in messages {
            let ServerMsg::Events { events, .. } = message else {
                continue;
            };
            // Raze damage resolves in the tick of its cast, so a same-tick hero hit is a raze hit.
            let (mut razed, mut struck) = (false, false);
            for event in events {
                match *event {
                    EventKind::AbilityCast { caster, ability } => {
                        razed |= caster == own && raze_reach(ability).is_some();
                    }
                    EventKind::Damaged {
                        source: Some(source),
                        target,
                        amount,
                        kind,
                        ..
                    } if source == own => {
                        let victim = tracker.entity(target).map(|track| &track.unit);
                        let enemy = victim.is_some_and(|unit| unit.team != tracker.team());
                        match victim.map(|unit| unit.kind) {
                            Some(UnitKind::Hero) if enemy => {
                                *hero_damage = hero_damage.saturating_add(amount);
                                struck |= kind == DamageKind::Magical;
                            }
                            Some(UnitKind::Tower) if enemy => {
                                *tower_damage = tower_damage.saturating_add(amount);
                            }
                            _ => {}
                        }
                    }
                    EventKind::StructureDestroyed { .. } => *tower_lost = true,
                    _ => {}
                }
            }
            *casts += u32::from(razed && aimed);
            *hits += u32::from(razed && struck);
        }
    }

    fn observe_home(&mut self, policy: &StateTracker, opponent: &StateTracker) {
        self.game.policy_home_ticks += u32::from(at_home(policy));
        self.game.opponent_home_ticks += u32::from(at_home(opponent));
        for (index, tracker) in [policy, opponent].into_iter().enumerate() {
            self.idle[index] = if idle(tracker) {
                self.idle[index] + 1
            } else {
                0
            };
        }
        let longest = [
            &mut self.game.policy_longest_idle,
            &mut self.game.opponent_longest_idle,
        ];
        for (longest, idle) in longest.into_iter().zip(self.idle) {
            *longest = (*longest).max(idle);
        }
    }
    fn finish(
        mut self,
        winner: Team,
        ticks: u32,
        rejections: u32,
        seats: &[DuelSeat; 2],
        policy_seat: usize,
    ) -> DuelGame {
        let player = |seat: &DuelSeat| {
            let slot = seat.tracker.slot();
            seat.tracker
                .current()
                .and_then(|view| view.players.iter().find(|player| player.slot == slot))
                .map_or((0, 0), |player| (player.deaths, player.level))
        };
        let (policy, opponent) = (&seats[policy_seat], &seats[1 - policy_seat]);
        self.game.ticks = ticks;
        self.game.rejections = rejections;
        self.game.policy_orders = policy.sequence;
        self.game.opponent_orders = opponent.sequence;
        (self.game.policy_deaths, self.game.policy_level) = player(policy);
        (self.game.opponent_deaths, self.game.opponent_level) = player(opponent);
        self.game.result = if winner == Team::Neutral {
            DuelResult::Draw
        } else if winner == self.game.side {
            DuelResult::Win
        } else {
            DuelResult::Loss
        };
        self.game.end = match (winner, self.tower_lost) {
            (Team::Neutral, _) if ticks >= crate::MAP2_TICK_CAP => DuelEnd::Cap,
            (Team::Neutral, _) => DuelEnd::Simultaneous,
            (_, true) => DuelEnd::Tower,
            (_, false) => DuelEnd::Kills,
        };
        self.game
    }
}

/// Alive away from home, standing still, and dealing no damage for a whole attack cycle.
fn idle(tracker: &StateTracker) -> bool {
    let (Some(hero), Some(view)) = (tracker.own_hero(), tracker.current()) else {
        return false;
    };
    let Some(track) = tracker.entity(hero.id) else {
        return false;
    };
    !at_home(tracker)
        && track
            .velocity
            .is_some_and(|velocity| velocity.delta == Vec2::ZERO)
        && track
            .last_damage_dealt
            .is_none_or(|damage| view.tick.saturating_sub(damage.tick) > IDLE_ATTACK_TICKS)
}

/// Dead, or within the quarter of the map around the own fountain.
fn at_home(tracker: &StateTracker) -> bool {
    match (tracker.own_hero(), own_fountain(tracker)) {
        (None, _) => true,
        (Some(hero), Some(fountain)) => {
            let quarter = map_maximum_raw(tracker.metadata().terrain_cells).max(0) / 4;
            distance_raw(hero.pos, fountain) < quarter as u64
        }
        (Some(_), None) => false,
    }
}

fn distance_raw(from: Vec2, to: Vec2) -> u64 {
    isqrt(from.distance_squared(to).max(0) as u64)
}
