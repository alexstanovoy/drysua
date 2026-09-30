//! Deterministic rule policies that act only through the drysua action space.

#[cfg(feature = "builtin")]
mod duel;
pub(crate) mod duel_cli;
mod harass_push;
pub(crate) mod progress;
pub(crate) mod style;
pub(crate) mod tactics;

#[cfg(feature = "builtin")]
pub use duel::{DuelConfig, DuelEnd, DuelGame, DuelResult, play_duel_game, run_duel};
pub use harass_push::{HarassPush, HarassStyle};
pub use style::{StyleSpec, StyleValues, seat_seed};

use bota_proto::AbilitySlot;

use crate::scripted::style::splitmix64;
use crate::teacher::TeacherStyle;
use crate::{
    ActionError, ActionSpace, ActionTarget, ControlledUnit, EntityIndex, IssuedOrder,
    ItemReadiness, MAX_ABILITY_SLOTS, OrderPersistence, PointIndex, StateTracker, StructuredAction,
    Teacher,
};

/// A rule policy that can hold any seat: training opponent, duel side or live player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScriptKind {
    /// The original rule teacher.
    Teacher,
    /// Zones the enemy hero off the lane with stacked razes, then pushes the tower.
    HarassPush,
}

impl ScriptKind {
    /// Stable command-line spelling.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Teacher => "teacher",
            Self::HarassPush => "harass-push",
        }
    }

    /// Command-line spelling of the policy drawing a styled preset per game.
    pub const fn styled_label(self) -> &'static str {
        match self {
            Self::Teacher => "teacher-styled",
            Self::HarassPush => "harass-push-styled",
        }
    }

    /// Parses [`Self::label`] or [`Self::styled_label`]; the flag tells which.
    pub fn parse_label(text: &str) -> Option<(Self, bool)> {
        [Self::Teacher, Self::HarassPush]
            .into_iter()
            .find_map(|kind| {
                (text == kind.label())
                    .then_some((kind, false))
                    .or((text == kind.styled_label()).then_some((kind, true)))
            })
    }
}

/// Per-seat state of one rule policy: its rules plus the style noise around them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptedPolicy {
    brain: Brain,
    noise: Noise,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Brain {
    Teacher(Box<Teacher>),
    HarassPush(HarassPush),
}

/// Decision skipping and random actions of a styled policy; the canonical style has neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Noise {
    /// Decisions per acted decision.
    period: u32,
    /// Chance in a thousand that an acted decision is a random tactical action.
    epsilon_permille: u64,
    random: u64,
    decisions: u32,
}

impl ScriptedPolicy {
    /// Creates a canonical policy with empty per-match memory.
    pub fn new(kind: ScriptKind) -> Self {
        Self::styled(kind, &StyleValues::canonical(kind), 0)
    }

    /// Creates a policy of the styled preset drawn from `seed`, which also drives its noise.
    pub fn styled_preset(kind: ScriptKind, seed: u64) -> Self {
        Self::styled(kind, &StyleSpec::styled(kind).draw(seed), seed)
    }

    /// Creates a policy of one drawn style; `seed` also drives its random actions.
    pub fn styled(kind: ScriptKind, values: &StyleValues, seed: u64) -> Self {
        let brain = match kind {
            ScriptKind::Teacher => Brain::Teacher(Box::new(Teacher::with_style(
                TeacherStyle::from_values(values),
            ))),
            ScriptKind::HarassPush => {
                Brain::HarassPush(HarassPush::with_style(HarassStyle::from_values(values)))
            }
        };
        let noise = Noise {
            period: u32::try_from(values.get(kind, "period")).expect("knob bounds"),
            epsilon_permille: u64::try_from(values.get(kind, "epsilon")).expect("knob bounds"),
            random: seed,
            decisions: 0,
        };
        Self { brain, noise }
    }

    /// A Teacher built outside the style table, such as a test variant.
    #[cfg(all(test, feature = "builtin"))]
    pub(crate) fn from_teacher(teacher: Teacher) -> Self {
        let mut policy = Self::new(ScriptKind::Teacher);
        policy.brain = Brain::Teacher(Box::new(teacher));
        policy
    }

    /// Selects an action and returns the exact action space used to select it.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        let noise = &mut self.noise;
        noise.decisions = noise.decisions.wrapping_add(1);
        if !noise.decisions.is_multiple_of(noise.period) {
            let space = ActionSpace::from_tracker_with_readiness(tracker, readiness)?;
            return Ok((StructuredAction::Continue, space));
        }
        let (action, space) = match &mut self.brain {
            Brain::Teacher(teacher) => teacher.decide(tracker, persistence, readiness)?,
            Brain::HarassPush(script) => script.decide(tracker, persistence, readiness)?,
        };
        if noise.epsilon_permille > 0
            && splitmix64(&mut noise.random) % 1_000 < noise.epsilon_permille
            && let Some(random) = random_tactical_action(&space, splitmix64(&mut noise.random))
        {
            return Ok((random, space));
        }
        Ok((action, space))
    }

    /// A policy that labels another policy's seat: Teacher hands its hero razes to the
    /// aim macro as single aimed casts, the form the learner's action space expresses.
    #[cfg(feature = "builtin")]
    pub(crate) fn shadow(kind: ScriptKind) -> Self {
        let mut policy = Self::new(kind);
        if kind == ScriptKind::Teacher {
            policy.brain = Brain::Teacher(Box::new(Teacher::with_macro_hero_aim()));
        }
        policy
    }

    /// Selects an action in `space`, the space the seat's tracker and readiness build.
    #[cfg(feature = "builtin")]
    pub(crate) fn decide_in(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
    ) -> Result<StructuredAction, ActionError> {
        match &mut self.brain {
            Brain::Teacher(teacher) => teacher.decide_in(tracker, persistence, space),
            Brain::HarassPush(script) => script.decide_in(tracker, persistence, space),
        }
    }

    /// Records a sent order and the snapshot tick of the space that decoded it.
    pub fn note_sent(&mut self, sequence: u32, issued: IssuedOrder, tick: u32) {
        match &mut self.brain {
            Brain::Teacher(teacher) => teacher.note_sent(sequence, issued, tick),
            Brain::HarassPush(script) => script.note_sent(sequence, issued, tick),
        }
    }

    /// Rolls back bounded local memory created by one rejected sequence.
    pub fn note_rejected(&mut self, sequence: u32) -> bool {
        match &mut self.brain {
            Brain::Teacher(teacher) => teacher.note_rejected(sequence),
            Brain::HarassPush(script) => script.note_rejected(sequence),
        }
    }
}

/// A uniformly drawn hero order family, then a uniformly drawn legal order in it: stop or
/// hold, a walk, an attack, or an untargeted cast. Economy orders are left out, so noise
/// perturbs play without selling the build.
fn random_tactical_action(space: &ActionSpace, draw: u64) -> Option<StructuredAction> {
    let hero = ControlledUnit::Hero;
    let pick = |mask: &[bool]| -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter_map(|(index, allowed)| allowed.then_some(index))
            .collect()
    };
    let halt = [
        StructuredAction::Stop { unit: hero },
        StructuredAction::Hold { unit: hero },
    ]
    .into_iter()
    .filter(|action| space.allows(*action))
    .collect::<Vec<_>>();
    let walks = pick(space.move_point_mask(hero))
        .into_iter()
        .map(|index| StructuredAction::MovePoint {
            unit: hero,
            point: PointIndex(index),
        })
        .collect::<Vec<_>>();
    let attacks = pick(space.attack_entity_mask(hero))
        .into_iter()
        .map(|index| StructuredAction::AttackUnit {
            unit: hero,
            target: EntityIndex(index),
        })
        .collect::<Vec<_>>();
    let casts = (0..MAX_ABILITY_SLOTS)
        .filter_map(|slot| {
            let action = StructuredAction::Cast {
                unit: hero,
                slot: AbilitySlot(u8::try_from(slot).ok()?),
                target: ActionTarget::None,
            };
            space.allows(action).then_some(action)
        })
        .collect::<Vec<_>>();
    let families: Vec<&Vec<StructuredAction>> = [&halt, &walks, &attacks, &casts]
        .into_iter()
        .filter(|family| !family.is_empty())
        .collect();
    let family = families.get((draw % families.len().max(1) as u64) as usize)?;
    let action = family[((draw >> 16) % family.len() as u64) as usize];
    assert!(space.allows(action));
    Some(action)
}
