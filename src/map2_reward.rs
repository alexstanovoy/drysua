#![allow(
    clippy::float_arithmetic,
    reason = "Seat-only reward accounting uses bounded f64 arithmetic"
)]

mod observation;

use std::fmt;

use bota_proto::{EntityId, EventKind, MatchInfo, SlotId};
use observation::{Role, SnapshotFacts, TowerFact};

/// Version of the Map2 reward definition, recorded with checkpoints and reward reports.
pub const MAP2_REWARD_VERSION: u32 = 8;
/// Per-tick discount required for exact potential cancellation.
pub const MAP2_REWARD_GAMMA_TICK: f32 = 1.0;
/// Terminal reward for an authoritative win, before the fast-win bonus.
pub const MAP2_REWARD_WIN: f64 = 1.0;
/// Terminal reward for an authoritative loss.
pub const MAP2_REWARD_LOSS: f64 = -1.0;
/// Terminal reward for a native draw (cap or simultaneous losses) or a learner time cap.
pub const MAP2_REWARD_DRAW: f64 = 0.0;
/// Extra win reward at the end of pregame, falling linearly to zero at the native cap.
pub const MAP2_REWARD_FAST_WIN_BONUS: f64 = 0.5;
/// Potential per unit of (own - enemy) weakest-tower HP fraction; the first tower lost decides.
pub const MAP2_REWARD_TOWER_WEIGHT: f64 = 0.75;
/// Potential per hero death of lead; the second death decides.
pub const MAP2_REWARD_DEATH_WEIGHT: f64 = 0.4;
/// Potential per unit of (own - enemy) hero HP fraction; a dead hero counts as its full next life.
pub const MAP2_REWARD_HEALTH_WEIGHT: f64 = 0.2;
/// Potential of a full clamped experience lead.
pub const MAP2_REWARD_XP_WEIGHT: f64 = 0.2;
/// Experience lead at which the experience potential saturates.
pub const MAP2_REWARD_XP_SCALE: u32 = 1_000;
/// Maximum events consumed atomically in one seat-visible tick.
pub const MAP2_REWARD_MAX_EVENTS: usize = 4096;
/// Maximum visible units accepted in one snapshot.
pub const MAP2_REWARD_MAX_UNITS: usize = 4096;
/// Reward components in report and log order; `total` is their sum.
pub const MAP2_REWARD_COMPONENTS: [&str; 7] = [
    "towers", "deaths", "health", "xp", "closure", "terminal", "fast_win",
];
/// Additive seat-visible diagnostic counters in report and log order.
pub const MAP2_REWARD_COUNTERS: [&str; 9] = [
    "own_deaths",
    "enemy_deaths",
    "own_xp_gained",
    "enemy_xp_gained",
    "hero_damage_dealt",
    "hero_damage_taken",
    "tower_damage_taken",
    "other_damage_taken",
    "structure_damage_dealt",
];

const MAX_TOWERS: usize = 64;
const MAX_AMOUNT: i32 = 1_000_000;
const MAX_XP: i32 = 1_000_000_000;
const POTENTIAL_BOUND: f64 = MAP2_REWARD_TOWER_WEIGHT
    + MAP2_REWARD_DEATH_WEIGHT * crate::MAP2_DEATH_LIMIT as f64
    + MAP2_REWARD_HEALTH_WEIGHT
    + MAP2_REWARD_XP_WEIGHT;

const _: () = assert!(MAP2_REWARD_WIN + MAP2_REWARD_LOSS == 0.0);
const _: () = assert!(MAP2_REWARD_LOSS < MAP2_REWARD_DRAW);
const _: () = assert!(MAP2_REWARD_DRAW < MAP2_REWARD_WIN);
const _: () = assert!(MAP2_REWARD_FAST_WIN_BONUS >= 0.0);
const _: () = assert!(MAP2_REWARD_FAST_WIN_BONUS < MAP2_REWARD_WIN);
// A kill must stay worth more than the damage it took to land it.
const _: () = assert!(MAP2_REWARD_HEALTH_WEIGHT < MAP2_REWARD_DEATH_WEIGHT);
const _: () = assert!(MAP2_REWARD_XP_WEIGHT < MAP2_REWARD_DEATH_WEIGHT);
// Shaping redistributes credit in time; it never outweighs the outcome.
const _: () = assert!(POTENTIAL_BOUND < 2.0 * MAP2_REWARD_WIN);
const _: () = assert!(MAP2_REWARD_XP_SCALE > 0);
const _: () = assert!(crate::MAP2_TICK_CAP > crate::MAP2_PREGAME_TICKS);

/// Native game result or learner time cap; never a technical failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map2RewardEnd {
    Win,
    Loss,
    Draw,
    TimeCap,
}

/// Additive seat-visible measurements over a reward interval, for diagnostics only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Map2RewardObservations {
    pub own_deaths: u64,
    pub enemy_deaths: u64,
    pub own_xp_gained: u64,
    pub enemy_xp_gained: u64,
    /// Own hero damage to the enemy hero.
    pub hero_damage_dealt: u64,
    /// Enemy hero damage to the own hero.
    pub hero_damage_taken: u64,
    /// Tower damage to the own hero.
    pub tower_damage_taken: u64,
    /// Creep, environment and unattributed damage to the own hero.
    pub other_damage_taken: u64,
    /// Own hero damage to enemy towers.
    pub structure_damage_dealt: u64,
}

impl Map2RewardObservations {
    /// Values in [`MAP2_REWARD_COUNTERS`] order.
    pub fn counters(&self) -> [u64; MAP2_REWARD_COUNTERS.len()] {
        [
            self.own_deaths,
            self.enemy_deaths,
            self.own_xp_gained,
            self.enemy_xp_gained,
            self.hero_damage_dealt,
            self.hero_damage_taken,
            self.tower_damage_taken,
            self.other_damage_taken,
            self.structure_damage_dealt,
        ]
    }

    /// Field-wise sum, or `None` on counter overflow.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self {
            own_deaths: self.own_deaths.checked_add(other.own_deaths)?,
            enemy_deaths: self.enemy_deaths.checked_add(other.enemy_deaths)?,
            own_xp_gained: self.own_xp_gained.checked_add(other.own_xp_gained)?,
            enemy_xp_gained: self.enemy_xp_gained.checked_add(other.enemy_xp_gained)?,
            hero_damage_dealt: self
                .hero_damage_dealt
                .checked_add(other.hero_damage_dealt)?,
            hero_damage_taken: self
                .hero_damage_taken
                .checked_add(other.hero_damage_taken)?,
            tower_damage_taken: self
                .tower_damage_taken
                .checked_add(other.tower_damage_taken)?,
            other_damage_taken: self
                .other_damage_taken
                .checked_add(other.other_damage_taken)?,
            structure_damage_dealt: self
                .structure_damage_dealt
                .checked_add(other.structure_damage_dealt)?,
        })
    }
}

/// Reward emitted over an interval; `ticks` excludes the initial baseline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map2RewardBreakdown {
    pub ticks: u32,
    /// Potential steps of the weakest-tower HP lead.
    pub towers: f64,
    /// Potential steps of the hero-death lead.
    pub deaths: f64,
    /// Potential steps of the hero HP lead.
    pub health: f64,
    /// Potential steps of the clamped experience lead.
    pub xp: f64,
    /// Terminal return of the whole potential, so shaping sums to minus the initial potential.
    pub closure: f64,
    pub terminal: f64,
    /// Win-only bonus for ending the game early.
    pub fast_win: f64,
    pub total: f64,
    pub end: Option<Map2RewardEnd>,
    pub observations: Map2RewardObservations,
}

impl Map2RewardBreakdown {
    /// Values in [`MAP2_REWARD_COMPONENTS`] order.
    pub fn components(&self) -> [f64; MAP2_REWARD_COMPONENTS.len()] {
        [
            self.towers,
            self.deaths,
            self.health,
            self.xp,
            self.closure,
            self.terminal,
            self.fast_win,
        ]
    }

    fn retotal(&mut self) {
        self.total = self.components().iter().sum();
        assert!(self.total.is_finite());
        assert!(self.ticks <= crate::MAP2_TICK_CAP);
    }
}

/// ID-free, seat-visible potential inputs, used as policy features and in diagnostic logs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Map2RewardState {
    /// Own and enemy weakest-tower HP fractions, in `[0, 1]`.
    pub tower_health: [f32; 2],
    /// Own and enemy hero deaths over the Map2 death limit, in `[0, 1]`.
    pub deaths: [f32; 2],
    /// Own and enemy hero HP fractions as last seen, in `[0, 1]`; one while dead.
    pub hero_health: [f32; 2],
    /// Experience lead over [`MAP2_REWARD_XP_SCALE`], clamped to `[-1, 1]`.
    pub xp_lead: f32,
    /// Current potential in reward units.
    pub potential: f32,
    /// Last fully consumed Snapshot/Events tick; absent before the first complete pair.
    pub completed_tick: Option<u32>,
}

/// A rejected observation or lifecycle call; rejected batches do not mutate committed state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Map2RewardError {
    Invalid(&'static str),
    Limit {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for Map2RewardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "Map2 reward: {message}"),
            Self::Limit {
                field,
                actual,
                maximum,
            } => {
                write!(
                    formatter,
                    "Map2 reward: {field} has {actual} entries; maximum is {maximum}"
                )
            }
        }
    }
}

impl std::error::Error for Map2RewardError {}

/// Potential terms in reward units; their sum is the shaping potential.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Potential {
    towers: f64,
    deaths: f64,
    health: f64,
    xp: f64,
}

impl Potential {
    fn total(self) -> f64 {
        self.towers + self.deaths + self.health + self.xp
    }
}

/// A tower seen by this seat, with its side index (0 own, 1 enemy) and HP fraction.
#[derive(Clone, Copy, Debug)]
struct Tower {
    id: EntityId,
    side: usize,
    health: f64,
}

/// Complete-tick reward accounting for one Map2 seat from its own message stream only.
///
/// The reward is `terminal + fast_win` plus potential-based shaping
/// `Φ(s') - Φ(s)` (γ = 1) with `Φ(terminal) = 0`, so shaping changes no optimal policy.
#[derive(Clone, Debug)]
pub struct Map2Reward {
    roles: [Role; 2],
    current: Option<SnapshotFacts>,
    pending: Option<SnapshotFacts>,
    /// Retired snapshot tower facts, refilled in place by the next snapshot.
    towers_scratch: Vec<TowerFact>,
    towers: Vec<Tower>,
    hero_health: [f64; 2],
    potential: Potential,
    interval: Map2RewardBreakdown,
    ended: bool,
}

impl Map2Reward {
    /// Starts a Map2 1v1 observer; match identity, seeds and private economy are not stored.
    pub fn new(slot: SlotId, info: &MatchInfo) -> Result<Self, Map2RewardError> {
        let roles = observation::roles(slot, info)?;
        assert_ne!(roles[0].team, roles[1].team);
        assert_ne!(roles[0].slot, roles[1].slot);
        Ok(Self {
            roles,
            current: None,
            pending: None,
            towers_scratch: Vec::new(),
            towers: Vec::with_capacity(MAX_TOWERS),
            hero_health: [1.0; 2],
            potential: Potential::default(),
            interval: Map2RewardBreakdown::default(),
            ended: false,
        })
    }

    /// Stages one fogged snapshot; its matching Events must arrive before any further snapshot.
    pub fn observe_snapshot(
        &mut self,
        view: &bota_proto::WorldView,
    ) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        if self.pending.is_some() {
            return invalid("Snapshot arrived before pending Events");
        }
        if let Some(current) = &self.current
            && current.tick.checked_add(1) != Some(view.tick)
        {
            return invalid("Snapshot ticks must be contiguous");
        }
        let pending =
            observation::snapshot_into(view, self.roles, std::mem::take(&mut self.towers_scratch))?;
        self.check_progress(&pending)?;
        self.pending = Some(pending);
        Ok(())
    }

    /// Consumes the matching Events and commits the staged tick.
    pub fn observe_events(
        &mut self,
        tick: u32,
        events: &[EventKind],
    ) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        let Some(pending) = &self.pending else {
            return invalid("Events without pending Snapshot");
        };
        if pending.tick != tick {
            return invalid("Events tick does not match pending Snapshot");
        }
        validate_events(events)?;
        let pending = self.pending.take().expect("pending snapshot was validated");
        for event in events {
            self.observe_event(&pending, event);
        }
        self.update_towers(&pending);
        self.update_hero_health(&pending);
        let potential = self.potential_of(&pending);
        if let Some(previous) = &self.current {
            observe_scoreboard(&mut self.interval.observations, previous, &pending);
            self.interval.towers += potential.towers - self.potential.towers;
            self.interval.deaths += potential.deaths - self.potential.deaths;
            self.interval.health += potential.health - self.potential.health;
            self.interval.xp += potential.xp - self.potential.xp;
            self.interval.ticks += 1;
        }
        self.potential = potential;
        if let Some(retired) = self.current.replace(pending) {
            self.towers_scratch = retired.towers;
        }
        self.interval.retotal();
        assert!(self.potential.total().abs() <= POTENTIAL_BOUND + 1.0e-9);
        Ok(())
    }

    /// Drains completed interval reward without resetting the potential.
    pub fn take_interval(&mut self) -> Result<Map2RewardBreakdown, Map2RewardError> {
        self.ensure_complete()?;
        let result = std::mem::take(&mut self.interval);
        assert!(result.total.is_finite());
        assert!(result.end.is_none());
        Ok(result)
    }

    /// Drains the final interval with the terminal reward and returns the whole potential.
    /// `TimeCap` is the learner's episode cap and scores as a draw.
    pub fn finish(&mut self, end: Map2RewardEnd) -> Result<Map2RewardBreakdown, Map2RewardError> {
        self.ensure_complete()?;
        let completed = self
            .current
            .as_ref()
            .expect("complete interval has a completed tick")
            .tick;
        self.interval.closure = -self.potential.total();
        self.interval.terminal = match end {
            Map2RewardEnd::Win => MAP2_REWARD_WIN,
            Map2RewardEnd::Loss => MAP2_REWARD_LOSS,
            Map2RewardEnd::Draw | Map2RewardEnd::TimeCap => MAP2_REWARD_DRAW,
        };
        self.interval.fast_win = fast_win_bonus(end, completed);
        self.interval.end = Some(end);
        self.interval.retotal();
        self.ended = true;
        let result = std::mem::take(&mut self.interval);
        assert!(result.total.is_finite());
        assert!(result.end.is_some());
        Ok(result)
    }

    /// Copies the observable potential inputs; no identity handle enters it.
    pub fn state(&self) -> Map2RewardState {
        let limit = f64::from(crate::MAP2_DEATH_LIMIT);
        let deaths = self
            .current
            .as_ref()
            .map_or([0; 2], |current| current.deaths);
        let state = Map2RewardState {
            tower_health: [0, 1].map(|side| self.weakest_tower(side) as f32),
            deaths: deaths
                .map(|count| (f64::from(count.min(crate::MAP2_DEATH_LIMIT)) / limit) as f32),
            hero_health: self.hero_health.map(|health| health as f32),
            xp_lead: (self.potential.xp / MAP2_REWARD_XP_WEIGHT) as f32,
            potential: self.potential.total() as f32,
            completed_tick: self.current.as_ref().map(|current| current.tick),
        };
        assert!(state.xp_lead.abs() <= 1.0);
        state
    }

    fn ensure_live(&self) -> Result<(), Map2RewardError> {
        if self.ended {
            return invalid("episode already ended");
        }
        Ok(())
    }

    fn ensure_complete(&self) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        if self.pending.is_some() {
            return invalid("interval has pending Events");
        }
        if self.current.is_none() {
            return invalid("interval has no complete Snapshot/Events pair");
        }
        Ok(())
    }

    fn check_progress(&self, pending: &SnapshotFacts) -> Result<(), Map2RewardError> {
        if let Some(current) = &self.current {
            if (0..2).any(|role| pending.xp[role] < current.xp[role]) {
                return invalid("public cumulative XP decreased");
            }
            if (0..2).any(|role| pending.deaths[role] < current.deaths[role]) {
                return invalid("public hero deaths decreased");
            }
        }
        let new = pending
            .towers
            .iter()
            .filter(|fact| self.towers.iter().all(|tower| tower.id != fact.id))
            .count();
        limit("tower history", self.towers.len() + new, MAX_TOWERS)
    }

    /// Buildings are always visible to both sides, so a known tower missing from a
    /// snapshot has been destroyed.
    fn update_towers(&mut self, pending: &SnapshotFacts) {
        for tower in &mut self.towers {
            tower.health = pending
                .towers
                .iter()
                .find(|fact| fact.id == tower.id)
                .map_or(0.0, |fact| fact.health);
        }
        for fact in &pending.towers {
            if self.towers.iter().all(|tower| tower.id != fact.id) {
                self.towers.push(Tower {
                    id: fact.id,
                    side: usize::from(fact.team != self.roles[0].team),
                    health: fact.health,
                });
            }
        }
        assert!(self.towers.len() <= MAX_TOWERS);
    }

    /// A dead hero counts as its full next life; an unseen living hero keeps its last seen HP.
    fn update_hero_health(&mut self, pending: &SnapshotFacts) {
        for role in 0..2 {
            if pending.heroes[role].is_none() {
                self.hero_health[role] = 1.0;
            } else if let Some(health) = pending.hero_health[role] {
                self.hero_health[role] = health;
            }
            assert!((0.0..=1.0).contains(&self.hero_health[role]));
        }
        assert!(pending.hero_health[0].is_some() || pending.heroes[0].is_none());
    }

    fn weakest_tower(&self, side: usize) -> f64 {
        self.towers
            .iter()
            .filter(|tower| tower.side == side)
            .map(|tower| tower.health)
            .reduce(f64::min)
            .unwrap_or(1.0)
    }

    fn potential_of(&self, pending: &SnapshotFacts) -> Potential {
        let deaths = pending
            .deaths
            .map(|count| f64::from(count.min(crate::MAP2_DEATH_LIMIT)));
        let lead = f64::from(pending.xp[0]) - f64::from(pending.xp[1]);
        let xp = (lead / f64::from(MAP2_REWARD_XP_SCALE)).clamp(-1.0, 1.0);
        Potential {
            towers: MAP2_REWARD_TOWER_WEIGHT * (self.weakest_tower(0) - self.weakest_tower(1)),
            deaths: MAP2_REWARD_DEATH_WEIGHT * (deaths[1] - deaths[0]),
            health: MAP2_REWARD_HEALTH_WEIGHT * (self.hero_health[0] - self.hero_health[1]),
            xp: MAP2_REWARD_XP_WEIGHT * xp,
        }
    }

    /// Attributes damage by public scoreboard bodies of this or the previous tick and seen towers.
    fn observe_event(&mut self, pending: &SnapshotFacts, event: &EventKind) {
        let EventKind::Damaged {
            source,
            target,
            amount,
            ..
        } = *event
        else {
            return;
        };
        let amount = u64::try_from(amount).expect("validated damage amount");
        let hero = |role: usize, id: EntityId| {
            pending.heroes[role] == Some(id)
                || self
                    .current
                    .as_ref()
                    .is_some_and(|current| current.heroes[role] == Some(id))
        };
        let tower_side = |id: EntityId| {
            self.towers
                .iter()
                .find(|tower| tower.id == id)
                .map(|tower| tower.side)
        };
        let observations = &mut self.interval.observations;
        if hero(0, target) {
            let counter = match source {
                Some(source) if hero(1, source) => &mut observations.hero_damage_taken,
                Some(source) if tower_side(source).is_some() => {
                    &mut observations.tower_damage_taken
                }
                _ => &mut observations.other_damage_taken,
            };
            *counter += amount;
        } else if source.is_some_and(|source| hero(0, source)) {
            if hero(1, target) {
                observations.hero_damage_dealt += amount;
            } else if tower_side(target) == Some(1) {
                observations.structure_damage_dealt += amount;
            }
        }
    }
}

fn observe_scoreboard(
    observations: &mut Map2RewardObservations,
    previous: &SnapshotFacts,
    pending: &SnapshotFacts,
) {
    observations.own_deaths += u64::from(pending.deaths[0] - previous.deaths[0]);
    observations.enemy_deaths += u64::from(pending.deaths[1] - previous.deaths[1]);
    observations.own_xp_gained += u64::from(pending.xp[0] - previous.xp[0]);
    observations.enemy_xp_gained += u64::from(pending.xp[1] - previous.xp[1]);
}

/// Win-only bonus: full at the end of pregame, linear to zero at the native cap.
fn fast_win_bonus(end: Map2RewardEnd, tick: u32) -> f64 {
    if end != Map2RewardEnd::Win {
        return 0.0;
    }
    let remaining = f64::from(crate::MAP2_TICK_CAP.saturating_sub(tick));
    let span = f64::from(crate::MAP2_TICK_CAP - crate::MAP2_PREGAME_TICKS);
    let bonus = MAP2_REWARD_FAST_WIN_BONUS * (remaining / span).min(1.0);
    assert!((0.0..=MAP2_REWARD_FAST_WIN_BONUS).contains(&bonus));
    bonus
}

fn validate_events(events: &[EventKind]) -> Result<(), Map2RewardError> {
    limit("event batch", events.len(), MAP2_REWARD_MAX_EVENTS)?;
    for event in events {
        if let EventKind::Damaged { amount, .. } = event
            && !(0..=MAX_AMOUNT).contains(amount)
        {
            return invalid("damage amount outside 0..=1000000");
        }
    }
    Ok(())
}

fn invalid<T>(message: &'static str) -> Result<T, Map2RewardError> {
    Err(Map2RewardError::Invalid(message))
}

fn limit(field: &'static str, actual: usize, maximum: usize) -> Result<(), Map2RewardError> {
    if actual > maximum {
        return Err(Map2RewardError::Limit {
            field,
            actual,
            maximum,
        });
    }
    Ok(())
}
