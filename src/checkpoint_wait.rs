use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use super::{
    CheckpointError, MAX_RUNTIME_TENSOR_BYTES, RUNTIME_TENSOR_FILE, TrainingArtifact,
    decode_tensor_count, read_bounded, sha256, validate_directory, validate_names,
};
use crate::{PolicyDevice, PolicyModel};

const M17_PARAMETERS: usize = 1_696_436;
const INITIAL_SOURCE: [u8; 32] = [
    0x05, 0xe7, 0x86, 0x63, 0xdd, 0x45, 0xac, 0x23, 0xad, 0x6c, 0x02, 0x42, 0xa6, 0x9d, 0x8b, 0x8e,
    0x45, 0xc1, 0x63, 0xf7, 0x86, 0x1c, 0x59, 0x15, 0x4e, 0x3f, 0x3f, 0xd8, 0x1d, 0xe9, 0xab, 0x1f,
];
const RECOVERY_SOURCE: [u8; 32] = [
    0x10, 0x7e, 0x19, 0xe6, 0x17, 0x94, 0xc4, 0x57, 0xce, 0x8e, 0xc6, 0xad, 0xc2, 0xb3, 0xe6, 0x6c,
    0xcc, 0xe0, 0x8e, 0xe5, 0xb6, 0x54, 0x4c, 0xb2, 0x00, 0x59, 0x64, 0xf3, 0xe7, 0xdf, 0xed, 0xc8,
];
const _: () = assert!(M17_PARAMETERS + 5 * 512 == crate::MODEL_PARAMETER_COUNT);

/// Pinned M17 ancestry of a fresh wait/progress-reward policy, never a resume or release identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Map2WaitInitializationProvenance {
    source_sha256: [u8; 32],
}

impl TrainingArtifact {
    /// INITIALIZATION ONLY from pinned M17 initial05e78663 or recovery004107e19e6.
    /// Validates original nine-key reward1 metadata, names, shape, finite values and SHA.
    /// Candle inserts five positive-zero global input rows without changing source bits.
    /// No optimizer, progress, RNG, league or qualification transfers; persist provenance.
    pub fn initialize_selected_m17_for_map2_wait(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, Map2WaitInitializationProvenance), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let (source, provenance) = decode_source(&bytes)?;
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let target = model
            .widen_map2_wait_parameters(&source)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        model
            .import_parameters(&target)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        assert_eq!(target.len(), crate::MODEL_PARAMETER_COUNT);
        assert_eq!(model.device(), device);
        Ok((model, provenance))
    }
}

impl Map2WaitInitializationProvenance {
    pub const fn source_sha256(self) -> [u8; 32] {
        self.source_sha256
    }

    /// Description to store with a new run, never in the historical source metadata.
    pub fn description(self) -> String {
        let digest: String = self
            .source_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest.len(), 64);
        let description = format!(
            "INITIALIZATION_ONLY source_m17_sha256={digest} source_f=15 source_a=5 source_m=17 source_ppo=30 source_rules=25 source_reward_version=1 source_reward_hash=798798703797057220 target_f={} target_a={} target_m={} target_ppo={} target_rules={} reward_version={} reward_hash={} parameter_bits_preserved=true named_tensors=62 new_weights=2560_positive_zero trunk_insertion_rows=85..90 optimizer_progress_rng_league=fresh GAMEPLAY_EQUIVALENCE=false qualification=false",
            crate::FEATURE_SCHEMA_VERSION,
            crate::ACTION_SCHEMA_VERSION,
            crate::MODEL_SCHEMA_VERSION,
            crate::PPO_SCHEMA_VERSION,
            crate::PPO_RULES_AUDIT_VERSION,
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH,
        );
        assert!(description.len() < 4096);
        description
    }

    pub(crate) fn from_source_sha256(source_sha256: [u8; 32]) -> Result<Self, CheckpointError> {
        if ![INITIAL_SOURCE, RECOVERY_SOURCE].contains(&source_sha256) {
            return Err(CheckpointError::TensorContract(
                "selected M17 wait initialization source SHA-256",
            ));
        }
        assert_ne!(source_sha256, [0; 32]);
        Ok(Self { source_sha256 })
    }
}

fn decode_source(
    bytes: &[u8],
) -> Result<(Vec<f32>, Map2WaitInitializationProvenance), CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if metadata.metadata().as_ref() != Some(&source_metadata()) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let source = decode_tensor_count(&tensors, "model.parameters", M17_PARAMETERS)?;
    let provenance = Map2WaitInitializationProvenance::from_source_sha256(sha256(bytes))?;
    assert_eq!(source.len(), M17_PARAMETERS);
    Ok((source, provenance))
}

fn source_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "10658390830565586343"),
        ("feature_schema_hash", "1861607613534772372"),
        ("model_schema_hash", "13592057279889489276"),
        ("ppo_schema_version", "30"),
        ("ppo_schema_hash", "16275284022255703821"),
        ("ppo_rules_audit_version", "25"),
        ("map2_reward_schema_version", "1"),
        ("map2_reward_schema_hash", "798798703797057220"),
        (
            "map2_reward_schema_descriptor",
            super::legacy_reward::MAP2_REWARD_V1_DESCRIPTOR,
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}
