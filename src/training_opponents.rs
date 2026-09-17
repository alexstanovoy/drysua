/// Versioned full-episode opponent configuration, bound to checkpoint run provenance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum TrainingOpponentSchedule {
    /// Preserve the original Teacher-only full-episode collector.
    #[default]
    Teacher,
    /// Two Weak updates, then one rotating Teacher pair out of every three pairs.
    #[value(name = "weak-warmup-v1")]
    WeakWarmupV1,
    /// Weak then Teacher, advancing on the full rolling training-win window.
    #[value(name = "mastery-v1")]
    MasteryV1,
}

impl TrainingOpponentSchedule {
    /// Stable CLI and checkpoint identity; changed schedule semantics need a new version.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Teacher => "teacher",
            Self::WeakWarmupV1 => "weak-warmup-v1",
            Self::MasteryV1 => "mastery-v1",
        }
    }

    #[cfg(feature = "builtin")]
    pub(crate) fn is_teacher(self, update: u64, pair: usize) -> bool {
        assert!(update <= crate::MAX_TRAINING_COUNTER);
        assert!(pair < 3);
        match self {
            Self::Teacher => true,
            Self::WeakWarmupV1 if update < 2 => false,
            Self::WeakWarmupV1 => ((update - 2) % 3 + pair as u64).is_multiple_of(3),
            Self::MasteryV1 => unreachable!("mastery opponent requires persisted stage"),
        }
    }
}
