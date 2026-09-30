//! Deterministic rule policies that act only through the drysua action space.

#[cfg(feature = "builtin")]
mod duel;
pub(crate) mod duel_cli;
mod harass_push;
pub(crate) mod progress;
pub(crate) mod tactics;

#[cfg(feature = "builtin")]
pub use duel::{DuelConfig, DuelEnd, DuelGame, DuelResult, play_duel_game, run_duel};
pub use harass_push::HarassPush;

use crate::{
    ActionError, ActionSpace, IssuedOrder, ItemReadiness, OrderPersistence, StateTracker,
    StructuredAction, Teacher,
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
}

/// Per-seat state of one rule policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptedPolicy {
    Teacher(Box<Teacher>),
    HarassPush(HarassPush),
}

impl ScriptedPolicy {
    /// Creates a policy with empty per-match memory.
    pub fn new(kind: ScriptKind) -> Self {
        match kind {
            ScriptKind::Teacher => Self::Teacher(Box::default()),
            ScriptKind::HarassPush => Self::HarassPush(HarassPush::new()),
        }
    }

    /// Selects an action and returns the exact action space used to select it.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        match self {
            Self::Teacher(teacher) => teacher.decide(tracker, persistence, readiness),
            Self::HarassPush(script) => script.decide(tracker, persistence, readiness),
        }
    }

    /// Records a sent order and the snapshot tick of the space that decoded it.
    pub fn note_sent(&mut self, sequence: u32, issued: IssuedOrder, tick: u32) {
        match self {
            Self::Teacher(teacher) => teacher.note_sent(sequence, issued, tick),
            Self::HarassPush(script) => script.note_sent(sequence, issued, tick),
        }
    }

    /// Rolls back bounded local memory created by one rejected sequence.
    pub fn note_rejected(&mut self, sequence: u32) -> bool {
        match self {
            Self::Teacher(teacher) => teacher.note_rejected(sequence),
            Self::HarassPush(script) => script.note_rejected(sequence),
        }
    }
}
