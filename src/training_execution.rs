use crate::PpoError;

/// Experimental scheduling only; never serialized as PPO hyperparameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum ActorOverlap {
    #[default]
    Off,
    ContinueV1,
}

/// Nondefault execution modes are bound by the canonical checkpoint run scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingExecutionOptions {
    pub actor_overlap: ActorOverlap,
    /// Prefetch one whole effective minibatch, not device tensors or activations.
    pub learner_prefetch: bool,
    /// Local folding-worker ceiling; one retains the historical serial path.
    pub host_math_workers: usize,
}

impl Default for TrainingExecutionOptions {
    fn default() -> Self {
        Self {
            actor_overlap: ActorOverlap::Off,
            learner_prefetch: false,
            host_math_workers: 1,
        }
    }
}

impl TrainingExecutionOptions {
    pub fn validate(self) -> Result<Self, PpoError> {
        if !(1..=32).contains(&self.host_math_workers) {
            return Err(PpoError::InvalidConfig(
                "host math workers must be within 1..=32",
            ));
        }
        Ok(self)
    }

    pub(crate) fn append_scope(self, command: &mut String) {
        if self.actor_overlap == ActorOverlap::ContinueV1 {
            command.push_str(" --actor-overlap continue-v1");
        }
        if self.learner_prefetch {
            command.push_str(" --learner-prefetch");
        }
        if self.host_math_workers != 1 {
            command.push_str(&format!(" --host-math-workers {}", self.host_math_workers));
        }
    }
}
