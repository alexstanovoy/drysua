use crate::PpoError;

/// Nondefault execution modes are bound by the canonical checkpoint run scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingExecutionOptions {
    /// Local folding-worker ceiling; one retains the historical serial path.
    pub host_math_workers: usize,
}

impl Default for TrainingExecutionOptions {
    fn default() -> Self {
        Self {
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

    #[cfg(feature = "builtin")]
    pub(crate) fn append_scope(self, command: &mut String) {
        if self.host_math_workers != 1 {
            command.push_str(&format!(" --host-math-workers {}", self.host_math_workers));
        }
    }
}
