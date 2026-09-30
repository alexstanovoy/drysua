use crate::PpoError;

/// Nondefault execution modes are bound by the canonical checkpoint run scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingExecutionOptions {
    /// Spread each epoch's rows across nearly equal minibatches without dropping a tail.
    pub balanced_minibatches: bool,
    /// Rows per device forward/backward and candidate-KL pass; gradients of a minibatch's
    /// microbatches are summed, so the choice only regroups floating-point reductions.
    pub training_microbatch: usize,
}

impl Default for TrainingExecutionOptions {
    fn default() -> Self {
        Self {
            balanced_minibatches: false,
            training_microbatch: DEFAULT_TRAINING_MICROBATCH,
        }
    }
}

pub(crate) const DEFAULT_TRAINING_MICROBATCH: usize = 512;
const MICROBATCH_ERROR: &str = "training microbatch must be 256, 512, 1024 or 2048";

impl TrainingExecutionOptions {
    pub fn validate(self) -> Result<Self, PpoError> {
        if !matches!(self.training_microbatch, 256 | 512 | 1024 | 2048) {
            return Err(PpoError::InvalidConfig(MICROBATCH_ERROR));
        }
        Ok(self)
    }

    #[cfg(feature = "builtin")]
    pub(crate) fn append_scope(self, command: &mut String) {
        if self.balanced_minibatches {
            command.push_str(" --balanced-minibatches");
        }
        if self.training_microbatch != DEFAULT_TRAINING_MICROBATCH {
            command.push_str(&format!(
                " --training-microbatch {}",
                self.training_microbatch
            ));
        }
    }
}

pub(crate) fn parse_actor_pipeline_groups(value: &str) -> Result<u8, &'static str> {
    match value {
        "1" => Ok(1),
        "2" => Ok(2),
        "4" => Ok(4),
        _ => Err("actor pipeline groups must be 1, 2 or 4"),
    }
}

pub(crate) fn parse_training_microbatch(value: &str) -> Result<usize, &'static str> {
    match value {
        "256" => Ok(256),
        "512" => Ok(512),
        "1024" => Ok(1024),
        "2048" => Ok(2048),
        _ => Err(MICROBATCH_ERROR),
    }
}

/// One opponent kind of the annealed per-game mixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnealedOpponent {
    /// The original scripted teacher.
    Teacher,
    /// The HarassPush rule policy.
    HarassPush,
    /// A strict runtime weights directory, loaded once and never updated.
    Weights(std::path::PathBuf),
    /// The actor weights the learner's own seat samples from.
    SelfPlay,
}
