use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use super::{
    CheckpointError, MAX_RUNTIME_TENSOR_BYTES, RUNTIME_TENSOR_FILE, TrainingArtifact,
    decode_tensor_count, read_bounded, sha256, validate_directory, validate_names,
};
use crate::{PolicyDevice, PolicyModel};

const SOURCE: [u8; 32] = [
    0x9d, 0x0b, 0x88, 0x12, 0x8b, 0xb4, 0xa7, 0x4d, 0x63, 0x6e, 0x07, 0x74, 0xab, 0x53, 0xee, 0xed,
    0x2e, 0x2a, 0xea, 0x30, 0x6c, 0x3a, 0xb0, 0x18, 0x6a, 0x56, 0x98, 0xf4, 0x1f, 0x92, 0xaf, 0xea,
];
const M19_PARAMETERS: usize = 1_698_996;
const _: () =
    assert!(M19_PARAMETERS + (crate::GLOBAL_FEATURES - 90) * 512 == crate::MODEL_PARAMETER_COUNT);

/// Pinned M19/u162 parameter ancestry, not a resume or reward/gameplay identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Map2NonwinInitializationProvenance {
    source_sha256: [u8; 32],
}

impl TrainingArtifact {
    /// INITIALIZATION ONLY: original M19/u162 9d0b8812 parameters into a fresh current model.
    /// Checks full SHA, frozen reward3 metadata, exact names/F32 shape and finite values.
    /// Appends positive-zero global input rows while preserving all original parameter bits.
    /// No source optimizer, progress, mastery window or RNG is restored; record provenance.
    pub fn initialize_selected_m19_for_nonwin_reward(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, Map2NonwinInitializationProvenance), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let (_, metadata) = SafeTensors::read_metadata(&bytes)
            .map_err(|error| CheckpointError::Backend(error.to_string()))?;
        if metadata.metadata().as_ref() != Some(&source_metadata()) {
            return Err(CheckpointError::SchemaMismatch);
        }
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|error| CheckpointError::Backend(error.to_string()))?;
        validate_names(&tensors, &["model.parameters"])?;
        let parameters = decode_tensor_count(&tensors, "model.parameters", M19_PARAMETERS)?;
        let provenance = Map2NonwinInitializationProvenance::from_source_sha256(sha256(&bytes))?;
        Ok((initialize_model(&parameters, seed, device)?, provenance))
    }
}

fn initialize_model(
    parameters: &[f32],
    seed: u64,
    device: PolicyDevice,
) -> Result<PolicyModel, CheckpointError> {
    let model = PolicyModel::fresh_on(seed, device)
        .map_err(|error| CheckpointError::Model(error.to_string()))?;
    let target = model
        .widen_m19_reward_parameters(parameters)
        .map_err(|error| CheckpointError::Model(error.to_string()))?;
    model
        .import_parameters(&target)
        .map_err(|error| CheckpointError::Model(error.to_string()))?;
    assert_eq!(parameters.len(), M19_PARAMETERS);
    assert_eq!(model.device(), device);
    Ok(model)
}

impl Map2NonwinInitializationProvenance {
    pub const fn source_sha256(self) -> [u8; 32] {
        self.source_sha256
    }

    pub fn description(self) -> String {
        let digest: String = self
            .source_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest.len(), 64);
        format!(
            "INITIALIZATION_ONLY source_m19_sha256={digest} source_f=17 source_m=19 source_ppo=32 source_rules=27 source_reward=3 source_reward_hash=11643768462079275437 target_f={} target_m={} target_ppo={} target_rules={} target_reward={} target_reward_hash={} parameter_bits_preserved=true named_tensors=62 new_weights={}_positive_zero trunk_insertion_rows=90..{} optimizer_progress_rng_league=fresh mastery_window=fresh gameplay_equivalence=false reward_equivalence=false qualification=false",
            crate::FEATURE_SCHEMA_VERSION,
            crate::MODEL_SCHEMA_VERSION,
            crate::PPO_SCHEMA_VERSION,
            crate::PPO_RULES_AUDIT_VERSION,
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH,
            (crate::GLOBAL_FEATURES - 90) * 512,
            crate::GLOBAL_FEATURES
        )
    }

    fn from_source_sha256(source_sha256: [u8; 32]) -> Result<Self, CheckpointError> {
        if source_sha256 != SOURCE {
            return Err(CheckpointError::TensorContract(
                "selected M19 nonwin initialization source SHA-256",
            ));
        }
        assert_ne!(source_sha256, [0; 32]);
        Ok(Self { source_sha256 })
    }
}

fn source_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "10658390830565586343"),
        ("feature_schema_hash", "4298252436472980484"),
        ("model_schema_hash", "7182549121935768714"),
        ("ppo_schema_version", "32"),
        ("ppo_schema_hash", "9056229782321552319"),
        ("ppo_rules_audit_version", "27"),
        ("map2_reward_schema_version", "3"),
        ("map2_reward_schema_hash", "11643768462079275437"),
        (
            "map2_reward_schema_descriptor",
            super::legacy_reward::MAP2_REWARD_V3_DESCRIPTOR,
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

#[cfg(test)]
#[path = "tests/nonwin_initialization.rs"]
mod tests;
