use crate::PpoError;

// Rollout storage, the non-rollout reserve and the largest PPO tensor microbatch
// fit the 12 GiB admission budget, so every accepted microbatch mode is admitted.
const _: () = assert!(
    crate::PPO_STORAGE_PEAK_BYTES
        + 2 * 1024 * 1024 * 1024
        + crate::MODEL_PPO_MAX_MICROBATCH as u64 * crate::MODEL_PPO_GRAPH_ROW_RESERVE_BYTES
        <= 12 * 1024 * 1024 * 1024
);

/// Nondefault execution modes are bound by the canonical checkpoint run scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingExecutionOptions {
    /// Annealed actor groups per wave; one retains sequential batch collection.
    pub actor_pipeline_groups: usize,
    /// Spread each epoch's rows across nearly equal minibatches without dropping a tail.
    pub balanced_minibatches: bool,
    /// Local folding-worker ceiling; one retains the historical serial path.
    pub host_math_workers: usize,
    /// Batch frozen neural-opponent sampling in the annealed collector; changes inference numerics.
    pub neural_opponent_batching: bool,
    /// Tensor rows per PPO forward/backward and candidate-KL pass, independent of actor batching.
    /// Values above 64 change GEMM and floating reduction grouping and require a new run scope.
    pub training_microbatch: usize,
    /// Reuse next-actor bootstrap values in the annealed collector only.
    /// Larger actor GEMMs may change bootstrap bits and subsequent PPO updates.
    pub reuse_actor_values: bool,
}

impl Default for TrainingExecutionOptions {
    fn default() -> Self {
        Self {
            actor_pipeline_groups: 1,
            balanced_minibatches: false,
            host_math_workers: 1,
            neural_opponent_batching: false,
            training_microbatch: 64,
            reuse_actor_values: false,
        }
    }
}

impl TrainingExecutionOptions {
    pub fn validate(self) -> Result<Self, PpoError> {
        if !matches!(self.training_microbatch, 64 | 128 | 256) {
            return Err(PpoError::InvalidConfig(
                "training microbatch must be 64, 128 or 256",
            ));
        }
        if !matches!(self.actor_pipeline_groups, 1 | 2 | 4) {
            return Err(PpoError::InvalidConfig(
                "actor pipeline groups must be 1, 2 or 4",
            ));
        }
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
        if self.actor_pipeline_groups != 1 {
            command.push_str(&format!(
                " --actor-pipeline-groups {}",
                self.actor_pipeline_groups
            ));
        }
        if self.training_microbatch != 64 {
            command.push_str(&format!(
                " --training-microbatch {}",
                self.training_microbatch
            ));
        }
        if self.neural_opponent_batching {
            command.push_str(" --opponent-inference batched");
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
        "64" => Ok(64),
        "128" => Ok(128),
        "256" => Ok(256),
        _ => Err("training microbatch must be 64, 128 or 256"),
    }
}
