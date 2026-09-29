use super::{
    CheckpointError, HashMap, MAX_RUNTIME_TENSOR_BYTES, RUNTIME_TENSOR_FILE, SafeTensors,
    TrainingArtifact, decode_runtime_tensor, decode_tensor_count, read_bounded,
    runtime_tensor_metadata_map, sha256, validate_directory, validate_names,
};
use crate::{PolicyDevice, PolicyModel, PpoSampleBudget};

const SOURCE_SHA256: [u8; 32] = [
    0x89, 0x5e, 0x66, 0xa7, 0x91, 0x86, 0x57, 0x0c, 0xe8, 0xf6, 0x53, 0xea, 0xdf, 0xad, 0x59, 0xea,
    0xce, 0x8c, 0x66, 0x4f, 0x16, 0xc5, 0xc5, 0x24, 0x80, 0x6a, 0xf4, 0x93, 0x6a, 0xc9, 0xf2, 0x1f,
];
const SOURCE_MAX_BYTES: u64 = 1_700_020 * 4 + 16 * 1024;
const LEGACY_PPO_HASH: u64 = crate::model::linked_schema_hash(
    crate::ppo::LEGACY_PPO_SCHEMA_DESCRIPTOR,
    &[
        (5, crate::ACTION_SCHEMA_HASH),
        (22, crate::FEATURE_SCHEMA_HASH),
        (24, crate::model::LEGACY_MODEL_SCHEMA_HASH),
        (7, crate::MAP2_REWARD_SCHEMA_HASH),
    ],
);
const LEGACY_ANNEALED_PPO_HASH: u64 = crate::model::linked_schema_hash(
    crate::PPO_ANNEALED_SCHEMA_DESCRIPTOR,
    &[(37, LEGACY_PPO_HASH)],
);
const _: () = assert!(crate::model::LEGACY_MODEL_PARAMETER_COUNT == 1_700_020);
const _: () = assert!(crate::MODEL_PARAMETER_COUNT == 1_812_983);

impl TrainingArtifact {
    /// Fresh-session construction; validates current or pinned source before backend allocation.
    #[cfg(any(
        feature = "builtin",
        all(
            test,
            feature = "cuda",
            any(target_os = "linux", target_os = "windows")
        )
    ))]
    pub(crate) fn initialize_from_weights(
        directory: &std::path::Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<PolicyModel, CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let parameters = decode_initial_parameters(&bytes)?;
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        Ok(model)
    }

    /// Initializes M25 from the exact selected M24/u428 runtime file only.
    /// No checkpoint, optimizer, progress, league, or RNG state is read.
    /// SHA and the complete source contract are checked before device creation.
    pub fn initialize_selected_m24_u428_for_side_actors(
        directory: &std::path::Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, String), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(&directory.join(RUNTIME_TENSOR_FILE), SOURCE_MAX_BYTES)?;
        let parameters = decode_selected_source(&bytes)?;
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        Ok((model, "INITIALIZATION_ONLY source_m24_u428_sha256=895e66a79186570ce8f653eadfad59eace8c664f16c5c524806af4936ac9f21f source_f=22 source_m=24 source_ppo=38 source_rules=32 source_action=5 source_reward=7 target_m=25 original_parameter_bits_preserved=1700020 appended_actor_parameters=112963 optimizer_progress_mastery_rng_league=fresh qualification=false".to_owned()))
    }

    /// Fresh-session weights only: strict current format or the exact pinned U428.
    /// The caller must construct fresh optimizer, progress and RNG state separately.
    /// All decoding and expansion finish before existing model parameters change.
    pub fn load_initial_weights(
        model: &PolicyModel,
        directory: &std::path::Path,
    ) -> Result<(), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let parameters = decode_initial_parameters(&bytes)?;
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))
    }
}

fn decode_initial_parameters(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    match decode_runtime_tensor(bytes) {
        Ok(parameters) => Ok(parameters),
        Err(_) if bytes.len() as u64 <= SOURCE_MAX_BYTES && sha256(bytes) == SOURCE_SHA256 => {
            decode_selected_source(bytes)
        }
        Err(error) => Err(error),
    }
}

fn decode_selected_source(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    if bytes.len() as u64 > SOURCE_MAX_BYTES || sha256(bytes) != SOURCE_SHA256 {
        return Err(CheckpointError::TensorHashMismatch);
    }
    let source = decode_source_contract(bytes)?;
    crate::expand_m24_side_actor_parameters(&source)
        .map_err(|error| CheckpointError::Model(error.to_string()))
}

fn decode_source_contract(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if metadata.metadata().as_ref() != Some(&source_metadata()) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    decode_tensor_count(
        &tensors,
        "model.parameters",
        crate::model::LEGACY_MODEL_PARAMETER_COUNT,
    )
}

fn source_metadata() -> HashMap<String, String> {
    let mut metadata = runtime_tensor_metadata_map(PpoSampleBudget::Annealed);
    for (key, value) in [
        ("model_schema_hash", crate::model::LEGACY_MODEL_SCHEMA_HASH),
        ("ppo_schema_hash", LEGACY_ANNEALED_PPO_HASH),
        ("ppo_schema_version", 38),
        ("ppo_rules_audit_version", 32),
        ("map2_reward_schema_version", 7),
    ] {
        metadata.insert(key.to_owned(), value.to_string());
    }
    metadata
}

#[cfg(test)]
use super::{Dtype, TensorView, serialize};
#[cfg(test)]
#[path = "tests/side_actor_initialization.rs"]
mod tests;
