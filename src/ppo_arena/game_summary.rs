//! End-of-game facts shared by training episode logs and frozen evaluation.
//!
//! Every seat accumulates its own casts from the events it is shown, so the
//! opponent's numbers come from the opponent seat's tracker, never from fog.

use std::fmt;

use bota_proto::{AbilityId, DamageKind, EventKind, Order, Target, Team, UnitKind};
use serde_json::{Value, json};

use super::TrainingEnvironment;
use crate::{IssuedOrder, PpoTerminalOutcome, StateTracker};

/// Shadowraze near, medium, far, then Requiem of Souls.
const TRACKED_ABILITIES: [AbilityId; 4] =
    [AbilityId(13), AbilityId(14), AbilityId(15), AbilityId(16)];
const CAST_LABELS: [&str; 4] = ["raze_near", "raze_mid", "raze_far", "requiem"];
const RAZE_COUNT: usize = 3;
/// Raze decisions by target mode, in the action schema's mode order.
const RAZE_MODE_LABELS: [&str; 3] = ["none", "entity", "point"];
/// Game-clock minutes (after the pregame) at which both seats record their standing;
/// early leads decide most games.
const MILESTONE_MINUTES: [u32; 3] = [2, 3, 5];

const fn milestone_tick(minutes: u32) -> u32 {
    crate::MAP2_PREGAME_TICKS + minutes * 60 * crate::MAP2_TICK_RATE
}

/// One seat's own standing at a milestone, from its own tracker only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Standing {
    xp: i32,
    /// Gold paid to this seat for last hits and kills; passive income is equal for both.
    bounty: i32,
    deaths: u16,
    /// Weakest own tower's HP in basis points.
    tower_bp: u16,
}

/// Own minus enemy standing at one milestone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lead {
    pub xp: i32,
    pub gold: i32,
    /// Enemy deaths minus own deaths.
    pub deaths: i32,
    /// Own minus enemy weakest-tower HP in basis points.
    pub tower_bp: i32,
}

/// Casts and structure losses one seat observed over its whole game.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SeatCombat {
    casts: [u32; TRACKED_ABILITIES.len()],
    /// Razes whose effect tick carried own magical damage to an enemy hero.
    raze_hero_hits: u32,
    /// Razes whose effect tick carried own magical damage to any hostile unit.
    raze_hits: u32,
    /// Raze decisions by target mode; an aim abandoned before its cast still counts.
    raze_modes: [u32; 3],
    tower_lost: bool,
    bounty: i32,
    standings: [Option<Standing>; MILESTONE_MINUTES.len()],
}

impl SeatCombat {
    /// Counts a decided own-hero raze by its target mode before the aim macro resolves it.
    pub(super) fn note_decision(&mut self, tracker: &StateTracker, issued: Option<IssuedOrder>) {
        let Some(IssuedOrder {
            unit: None,
            order: Order::Cast { slot, target },
        }) = issued
        else {
            return;
        };
        let raze = tracker
            .own_hero()
            .and_then(|hero| hero.abilities.get(usize::from(slot.0)))
            .is_some_and(|ability| TRACKED_ABILITIES[..RAZE_COUNT].contains(&ability.id));
        if raze {
            let mode = match target {
                Target::None => 0,
                Target::Unit(_) => 1,
                Target::Pos(_) => 2,
            };
            self.raze_modes[mode] = self.raze_modes[mode].saturating_add(1);
        }
    }

    /// Folds one tick of this seat's events; the tick's snapshot is already observed.
    pub(super) fn observe(&mut self, tracker: &StateTracker, events: &[EventKind]) {
        let own_team = tracker.team();
        let own_hero = tracker.own_hero().map(|hero| hero.id);
        let mut razes = 0u32;
        let mut hero_hit = false;
        let mut hit = false;
        for event in events {
            match event {
                EventKind::AbilityCast { caster, ability } if Some(*caster) == own_hero => {
                    if let Some(index) = TRACKED_ABILITIES.iter().position(|id| id == ability) {
                        self.casts[index] = self.casts[index].saturating_add(1);
                        razes += u32::from(index < RAZE_COUNT);
                    }
                }
                EventKind::Damaged {
                    source: Some(source),
                    target,
                    kind: DamageKind::Magical,
                    ..
                } if Some(*source) == own_hero => {
                    let hostile = tracker.entity(*target).map(|track| &track.unit);
                    let hostile = hostile.filter(|unit| unit.team != own_team);
                    hit |= hostile.is_some();
                    hero_hit |= hostile.is_some_and(|unit| unit.kind == UnitKind::Hero);
                }
                EventKind::StructureDestroyed { team, .. } if *team == own_team => {
                    self.tower_lost = true;
                }
                EventKind::Died {
                    killer: Some(killer),
                    gold,
                    ..
                } if Some(*killer) == own_hero => {
                    self.bounty = self.bounty.saturating_add(*gold);
                }
                _ => {}
            }
        }
        self.record_standings(tracker);
        // Razes land on their effect tick, so a same-tick magical hero hit belongs to them.
        if hero_hit {
            self.raze_hero_hits = self.raze_hero_hits.saturating_add(razes.min(1));
        }
        if hit {
            self.raze_hits = self.raze_hits.saturating_add(razes.min(1));
        }
        assert!(self.raze_hero_hits <= self.raze_hits);
        assert!(self.raze_hits <= self.casts[..RAZE_COUNT].iter().sum::<u32>());
    }
}

impl SeatCombat {
    /// Records the standing at every milestone this tick has reached; both seats
    /// observe every tick, so their standings share the tick.
    fn record_standings(&mut self, tracker: &StateTracker) {
        let Some(tick) = tracker.current().map(|view| view.tick) else {
            return;
        };
        for (standing, minutes) in self.standings.iter_mut().zip(MILESTONE_MINUTES) {
            if standing.is_none() && tick >= milestone_tick(minutes) {
                let player = tracker.own_player();
                let tower = if self.tower_lost {
                    0.0
                } else {
                    weakest_tower_fraction(tracker, tracker.team())
                };
                *standing = Some(Standing {
                    xp: player.map_or(0, |player| player.xp),
                    bounty: self.bounty,
                    deaths: player.map_or(0, |player| player.deaths),
                    tower_bp: basis_points(tower),
                });
            }
        }
    }
}

/// Why a Map2 game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EndReason {
    Tower,
    Deaths,
    TimeCap,
    Draw,
}

impl EndReason {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Tower => "tower",
            Self::Deaths => "deaths",
            Self::TimeCap => "timecap",
            Self::Draw => "draw",
        }
    }
}

/// One hero's end state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HeroSummary {
    pub kills: u16,
    pub deaths: u16,
    pub level: u8,
    pub xp: i32,
    /// Weakest own tower's HP fraction to four decimals; the mid tier-one tower in the mid-only map.
    pub tower_hp: f64,
    pub combat: SeatCombat,
}

/// The policy seat's view of one finished game.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GameSummary {
    pub side: Team,
    pub outcome: Option<PpoTerminalOutcome>,
    pub end_reason: EndReason,
    pub ticks: u32,
    pub own: HeroSummary,
    pub enemy: HeroSummary,
}

impl GameSummary {
    /// Reads both seats at the terminal tick; `outcome` is absent only at the tick cap.
    pub(super) fn capture(
        environment: &TrainingEnvironment,
        outcome: Option<PpoTerminalOutcome>,
        ticks: u32,
    ) -> Self {
        assert_eq!(environment.seats.len(), 2);
        assert!(ticks <= crate::MAP2_TICK_CAP);
        let own = hero_summary(&environment.seats[environment.policy_seat]);
        let enemy = hero_summary(&environment.seats[1 - environment.policy_seat]);
        let end_reason = match outcome {
            None => EndReason::TimeCap,
            Some(PpoTerminalOutcome::Draw) if ticks >= crate::MAP2_TICK_CAP => EndReason::TimeCap,
            Some(PpoTerminalOutcome::Draw) => EndReason::Draw,
            Some(PpoTerminalOutcome::Win) if enemy.combat.tower_lost => EndReason::Tower,
            Some(PpoTerminalOutcome::Loss) if own.combat.tower_lost => EndReason::Tower,
            Some(PpoTerminalOutcome::Win | PpoTerminalOutcome::Loss) => EndReason::Deaths,
        };
        Self {
            side: environment.seats[environment.policy_seat].tracker.team(),
            outcome,
            end_reason,
            ticks,
            own,
            enemy,
        }
    }

    pub(crate) const fn outcome_label(&self) -> &'static str {
        match self.outcome {
            Some(PpoTerminalOutcome::Win) => "win",
            Some(PpoTerminalOutcome::Loss) => "loss",
            Some(PpoTerminalOutcome::Draw) | None => "draw",
        }
    }

    pub(crate) const fn side_label(&self) -> &'static str {
        side_label(self.side)
    }

    /// Own minus enemy standing at each milestone the game reached.
    pub(crate) fn leads(&self) -> [Option<Lead>; MILESTONE_MINUTES.len()] {
        let (own, enemy) = (&self.own.combat.standings, &self.enemy.combat.standings);
        std::array::from_fn(|index| {
            let (own, enemy) = (own[index]?, enemy[index]?);
            Some(Lead {
                xp: own.xp - enemy.xp,
                gold: own.bounty - enemy.bounty,
                deaths: i32::from(enemy.deaths) - i32::from(own.deaths),
                tower_bp: i32::from(own.tower_bp) - i32::from(enemy.tower_bp),
            })
        })
    }

    /// The per-game fields of an evaluation JSON line.
    pub(crate) fn json(&self) -> Value {
        json!({
            "side": self.side_label(),
            "outcome": self.outcome_label(),
            "end_reason": self.end_reason.label(),
            "ticks": self.ticks,
            "own": hero_json(&self.own),
            "enemy": hero_json(&self.enemy),
            "leads": MILESTONE_MINUTES
                .iter()
                .zip(self.leads())
                .map(|(minutes, lead)| {
                    let lead = lead.map(|lead| {
                        json!({"xp": lead.xp, "gold": lead.gold, "deaths": lead.deaths,
                               "tower_bp": lead.tower_bp})
                    });
                    (format!("{minutes}m"), lead.unwrap_or(Value::Null))
                })
                .collect::<serde_json::Map<String, Value>>(),
        })
    }
}

pub(crate) const fn side_label(team: Team) -> &'static str {
    match team {
        Team::Radiant => "radiant",
        Team::Dire => "dire",
        Team::Neutral => "neutral",
    }
}

/// `key=value` fields for the episode log line.
impl fmt::Display for GameSummary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "side={} outcome={} end_reason={} ticks={}",
            self.side_label(),
            self.outcome_label(),
            self.end_reason.label(),
            self.ticks
        )?;
        for (prefix, hero) in [("own", &self.own), ("enemy", &self.enemy)] {
            write!(
                formatter,
                " {prefix}_kills={} {prefix}_deaths={} {prefix}_level={} {prefix}_xp={} {prefix}_tower_hp={:.4} {prefix}_raze_hero_hits={} {prefix}_raze_hits={}",
                hero.kills,
                hero.deaths,
                hero.level,
                hero.xp,
                hero.tower_hp,
                hero.combat.raze_hero_hits,
                hero.combat.raze_hits
            )?;
            for (label, count) in RAZE_MODE_LABELS.iter().zip(hero.combat.raze_modes) {
                write!(formatter, " {prefix}_raze_mode_{label}={count}")?;
            }
            for (label, casts) in CAST_LABELS.iter().zip(hero.combat.casts) {
                write!(formatter, " {prefix}_casts_{label}={casts}")?;
            }
        }
        for (minutes, lead) in MILESTONE_MINUTES.iter().zip(self.leads()) {
            if let Some(lead) = lead {
                write!(
                    formatter,
                    " lead_{minutes}m_xp={} lead_{minutes}m_gold={} lead_{minutes}m_deaths={} lead_{minutes}m_tower_bp={}",
                    lead.xp, lead.gold, lead.deaths, lead.tower_bp
                )?;
            }
        }
        Ok(())
    }
}

fn hero_summary(seat: &super::ArenaSeatPolicy) -> HeroSummary {
    let tracker = &seat.tracker;
    let player = tracker.own_player();
    HeroSummary {
        kills: player.map_or(0, |player| player.kills),
        deaths: player.map_or(0, |player| player.deaths),
        level: player.map_or(0, |player| player.level),
        xp: player.map_or(0, |player| player.xp),
        tower_hp: if seat.combat.tower_lost {
            0.0
        } else {
            weakest_tower_fraction(tracker, tracker.team())
        },
        combat: seat.combat,
    }
}

#[allow(
    clippy::float_arithmetic,
    reason = "a reported HP fraction; never feeds back into play"
)]
fn weakest_tower_fraction(tracker: &StateTracker, team: Team) -> f64 {
    let weakest = tracker
        .current()
        .into_iter()
        .flat_map(|view| &view.units)
        .filter(|unit| unit.kind == UnitKind::Tower && unit.team == team && unit.max_hp > 0)
        .map(|unit| f64::from(unit.hp.max(0)) / f64::from(unit.max_hp))
        .fold(1.0, f64::min);
    (weakest * 10_000.0).round() / 10_000.0
}

#[allow(
    clippy::float_arithmetic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a reported HP fraction in [0, 1]; never feeds back into play"
)]
fn basis_points(fraction: f64) -> u16 {
    assert!((0.0..=1.0).contains(&fraction));
    (fraction * 10_000.0).round() as u16
}

fn hero_json(hero: &HeroSummary) -> Value {
    let casts: serde_json::Map<String, Value> = CAST_LABELS
        .iter()
        .zip(hero.combat.casts)
        .map(|(label, casts)| ((*label).to_owned(), json!(casts)))
        .collect();
    json!({
        "kills": hero.kills,
        "deaths": hero.deaths,
        "level": hero.level,
        "xp": hero.xp,
        "tower_hp": hero.tower_hp,
        "casts": casts,
        "raze_hero_hits": hero.combat.raze_hero_hits,
        "raze_hits": hero.combat.raze_hits,
        "raze_modes": RAZE_MODE_LABELS
            .iter()
            .zip(hero.combat.raze_modes)
            .map(|(label, count)| ((*label).to_owned(), json!(count)))
            .collect::<serde_json::Map<String, Value>>(),
    })
}
