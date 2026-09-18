use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{MODEL_PARAMETER_COUNT, ModelError, PolicyModel};

static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

/// Immutable finite F32 weights used by frozen opponents and accepted checkpoints.
#[derive(Clone)]
pub struct PolicySnapshot {
    id: u64,
    fingerprint: u64,
    generation: u64,
    parameters: Arc<[f32]>,
}

impl fmt::Debug for PolicySnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolicySnapshot")
            .field("id", &self.id)
            .field("fingerprint", &self.fingerprint)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl PolicySnapshot {
    /// Captures one coherent model revision into an immutable host snapshot.
    pub fn capture(model: &PolicyModel, generation: u64) -> Result<Self, SnapshotError> {
        let parameters = model.export_parameters().map_err(model_error)?;
        Self::from_parameters(parameters, generation)
    }

    fn from_parameters(parameters: Vec<f32>, generation: u64) -> Result<Self, SnapshotError> {
        if parameters.len() != MODEL_PARAMETER_COUNT {
            return Err(SnapshotError::ParameterCount {
                actual: parameters.len(),
                expected: MODEL_PARAMETER_COUNT,
            });
        }
        if let Some((index, _)) = parameters
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(SnapshotError::NonFiniteParameter { index });
        }
        let id = NEXT_SNAPSHOT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| SnapshotError::SnapshotIdentityExhausted)?;
        if id == 0 {
            return Err(SnapshotError::SnapshotIdentityExhausted);
        }
        let fingerprint = parameter_fingerprint(&parameters);
        Ok(Self {
            id,
            fingerprint,
            generation,
            parameters: parameters.into(),
        })
    }

    /// Materializes a private model instance for one frozen actor.
    pub fn instantiate(&self) -> Result<PolicyModel, SnapshotError> {
        let model = PolicyModel::fresh(self.fingerprint ^ self.generation).map_err(model_error)?;
        model
            .import_parameters(&self.parameters)
            .map_err(model_error)?;
        Ok(model)
    }

    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
}

fn parameter_fingerprint(parameters: &[f32]) -> u64 {
    let mut hash = crate::model::FNV_OFFSET;
    for value in parameters {
        hash = crate::model::fnv1a_extend(hash, &value.to_bits().to_le_bytes());
    }
    hash
}

/// Snapshot capture, identity, or materialization failure.
#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotError {
    ParameterCount { actual: usize, expected: usize },
    NonFiniteParameter { index: usize },
    SnapshotIdentityExhausted,
    Model(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParameterCount { actual, expected } => {
                write!(
                    formatter,
                    "snapshot has {actual} parameters, expected {expected}"
                )
            }
            Self::NonFiniteParameter { index } => {
                write!(formatter, "snapshot parameter {index} is non-finite")
            }
            Self::SnapshotIdentityExhausted => {
                formatter.write_str("snapshot identity is exhausted")
            }
            Self::Model(message) => write!(formatter, "snapshot model error: {message}"),
        }
    }
}

impl Error for SnapshotError {}

fn model_error(error: ModelError) -> SnapshotError {
    SnapshotError::Model(error.to_string())
}
