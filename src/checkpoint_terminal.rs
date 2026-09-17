use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use super::{
    CheckpointError, MAX_RUNTIME_TENSOR_BYTES, RUNTIME_TENSOR_FILE, TrainingArtifact,
    decode_tensor_count, read_bounded, sha256, validate_directory, validate_names,
};
use crate::{PolicyDevice, PolicyModel};

#[path = "checkpoint_reward_v5.rs"]
mod reward_v5;

const SOURCE: [u8; 32] = [
    0xb2, 0x97, 0x52, 0xac, 0xf5, 0xc0, 0x26, 0x87, 0xa5, 0x4a, 0x63, 0xdc, 0xe9, 0xd7, 0x64, 0x80,
    0xf8, 0xe9, 0x7f, 0xc4, 0x16, 0xad, 0x9d, 0x41, 0x59, 0xc2, 0x1a, 0x59, 0xe9, 0x10, 0x97, 0x80,
];
const SOURCE_PARAMETERS: usize = 1_700_020;
const _: () = assert!(SOURCE_PARAMETERS == crate::MODEL_PARAMETER_COUNT);

impl TrainingArtifact {
    /// INITIALIZATION ONLY: pinned M21/u300 parameters into a fresh current model.
    /// Verifies frozen reward5 metadata, SHA, F32 shape and finiteness. Preserves every
    /// parameter bit, including actor and critic; imports no optimizer, RNG or mastery.
    /// The returned provenance must accompany the new artifact, not relabel the source.
    pub fn initialize_selected_m21_for_terminal_reward(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, String), CheckpointError> {
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
        let parameters = decode_tensor_count(&tensors, "model.parameters", SOURCE_PARAMETERS)?;
        if sha256(&bytes) != SOURCE {
            return Err(CheckpointError::TensorContract(
                "selected M21/u300 terminal initialization source SHA-256",
            ));
        }
        let model = initialize_parameters(&parameters, seed, device)?;
        Ok((
            model,
            format!(
                "INITIALIZATION_ONLY source_m21_u300_sha256=b29752acf5c02687a54a63dce9d76480f8e97fc416ad9d4159c21a59e9109780 source_f=19 source_m=21 source_ppo=34 source_rules=29 source_reward=5 source_reward_hash=10775256611790261869 target_f={} target_m={} target_ppo={} target_rules={} target_reward={} target_reward_hash={} parameter_bits_preserved=true parameters=1700020 new_weights=0 actor_critic_rescaled=false optimizer_progress_mastery_rng_league=fresh reward_equivalence=false qualification=false",
                crate::FEATURE_SCHEMA_VERSION,
                crate::MODEL_SCHEMA_VERSION,
                crate::PPO_SCHEMA_VERSION,
                crate::PPO_RULES_AUDIT_VERSION,
                crate::MAP2_REWARD_SCHEMA_VERSION,
                crate::MAP2_REWARD_SCHEMA_HASH,
            ),
        ))
    }
}

fn initialize_parameters(
    parameters: &[f32],
    seed: u64,
    device: PolicyDevice,
) -> Result<PolicyModel, CheckpointError> {
    assert_eq!(parameters.len(), SOURCE_PARAMETERS);
    let model = PolicyModel::fresh_on(seed, device)
        .map_err(|error| CheckpointError::Model(error.to_string()))?;
    model
        .import_parameters(parameters)
        .map_err(|error| CheckpointError::Model(error.to_string()))?;
    assert_eq!(model.device(), device);
    Ok(model)
}

fn source_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "10658390830565586343"),
        ("feature_schema_hash", "11343334068766071417"),
        ("model_schema_hash", "13521186719558157260"),
        ("ppo_schema_version", "34"),
        ("ppo_schema_hash", "12153447298094992077"),
        ("ppo_rules_audit_version", "29"),
        ("map2_reward_schema_version", "5"),
        ("map2_reward_schema_hash", "10775256611790261869"),
        ("map2_reward_schema_descriptor", reward_v5::DESCRIPTOR),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

#[cfg(test)]
#[path = "tests/terminal_initialization.rs"]
mod tests;
