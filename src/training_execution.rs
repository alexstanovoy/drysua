use crate::PpoError;

/// Nondefault execution modes are bound by the canonical checkpoint run scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingExecutionOptions {
    /// Spread each epoch's rows across nearly equal minibatches without dropping a tail.
    pub balanced_minibatches: bool,
    /// Local folding-worker ceiling; one retains the historical serial path.
    pub host_math_workers: usize,
    /// Reuse next-actor bootstrap values in the annealed collector only.
    /// Larger actor GEMMs may change bootstrap bits and subsequent PPO updates.
    pub reuse_actor_values: bool,
}

impl Default for TrainingExecutionOptions {
    fn default() -> Self {
        Self {
            balanced_minibatches: false,
            host_math_workers: 1,
            reuse_actor_values: false,
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

    #[cfg(feature = "builtin")]
    pub(crate) fn append_scope(self, command: &mut String) {
        if self.balanced_minibatches {
            command.push_str(" --balanced-minibatches");
        }
        if self.host_math_workers != 1 {
            command.push_str(&format!(" --host-math-workers {}", self.host_math_workers));
        }
        if self.reuse_actor_values {
            command.push_str(" --reuse-actor-values");
        }
    }
}
