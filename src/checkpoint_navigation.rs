use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use super::{
    CheckpointError, MAX_RUNTIME_TENSOR_BYTES, RUNTIME_TENSOR_FILE, TrainingArtifact,
    decode_tensor_count, read_bounded, sha256, validate_directory, validate_names,
};
use crate::{PolicyDevice, PolicyModel};

const M16_PARAMETERS: usize = 1_696_436;
const INITIAL_SOURCE: [u8; 32] = [
    0x17, 0x39, 0xd2, 0x80, 0xcb, 0x6c, 0x3f, 0xbd, 0x0d, 0xf7, 0x1f, 0xfe, 0x8c, 0x4a, 0x12, 0x9e,
    0xd3, 0xe2, 0x5a, 0x0e, 0xd2, 0x94, 0xb0, 0xb5, 0x79, 0x31, 0xc4, 0x89, 0x1c, 0x56, 0x6b, 0xf4,
];
const ADVANTAGE_SOURCE: [u8; 32] = [
    0xfd, 0xd4, 0xd3, 0xf2, 0xa8, 0x5a, 0x64, 0xd9, 0xb4, 0xd1, 0x62, 0xf1, 0x60, 0x7a, 0x94, 0xa4,
    0xaa, 0xba, 0x8a, 0x8e, 0x31, 0x36, 0xe5, 0x1d, 0xab, 0x21, 0x52, 0x18, 0x1c, 0xdf, 0x63, 0xb7,
];

const _: () = assert!(M16_PARAMETERS == crate::MODEL_PARAMETER_COUNT);
const _: () = assert!(crate::MAP2_REWARD_SCHEMA_VERSION == 1);
const _: () = assert!(crate::MAP2_REWARD_SCHEMA_HASH == 798_798_703_797_057_220);

/// Whitelisted M16 parameter ancestry for an unqualified, new-navigation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Map2NavigationInitializationProvenance {
    source_sha256: [u8; 32],
}

impl TrainingArtifact {
    /// INITIALIZATION ONLY from the pinned M16 initial (1739d280...) or advantage (fdd4d3f2...).
    /// Validates exact nine-key A4/F14/M16/PPO29/rules24/reward1 metadata, full file SHA,
    /// names, F32 shape and finite values before creating a model. Candle copies all
    /// parameter bits in the audited 62-tensor layout; no source training state is read.
    /// New legal actions change behavior: GAMEPLAY_EQUIVALENCE=false, qualification=false.
    /// The caller must retain provenance and create fresh optimizer, progress, RNG and league.
    /// This reads source files only; it neither saves weights nor trains.
    pub fn initialize_selected_m16_for_map2_navigation(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, Map2NavigationInitializationProvenance), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let (parameters, provenance) = decode_selected_m16_navigation_tensor(&bytes)?;
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        model
            .initialize_m16_navigation_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        assert_eq!(parameters.len(), M16_PARAMETERS);
        assert_eq!(model.device(), device);
        Ok((model, provenance))
    }
}

impl Map2NavigationInitializationProvenance {
    /// Full immutable source-file digest; not a runtime compatibility identity.
    pub const fn source_sha256(self) -> [u8; 32] {
        self.source_sha256
    }

    /// Description to retain with a new run rather than in the historical source metadata.
    pub fn description(self) -> String {
        let digest: String = self
            .source_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest.len(), 64);
        let description = format!(
            "INITIALIZATION_ONLY source_m16_sha256={digest} source_f=14 source_a=4 source_m=16 source_ppo=29 source_rules=24 target_f={} target_a={} target_m={} target_ppo={} target_rules={} reward_version=1 reward_hash=798798703797057220 parameter_bits_preserved=true named_tensors=62 new_weights=0 optimizer_progress_rng_league=fresh GAMEPLAY_EQUIVALENCE=false new_legal_actions_change_behavior=true qualification=false",
            crate::FEATURE_SCHEMA_VERSION,
            crate::ACTION_SCHEMA_VERSION,
            crate::MODEL_SCHEMA_VERSION,
            crate::PPO_SCHEMA_VERSION,
            crate::PPO_RULES_AUDIT_VERSION
        );
        assert!(description.len() < 4_096);
        description
    }

    pub(crate) fn from_source_sha256(source_sha256: [u8; 32]) -> Result<Self, CheckpointError> {
        if ![INITIAL_SOURCE, ADVANTAGE_SOURCE].contains(&source_sha256) {
            return Err(CheckpointError::TensorContract(
                "selected M16 navigation initialization source SHA-256",
            ));
        }
        assert_ne!(source_sha256, [0; 32]);
        assert_eq!(source_sha256.len(), 32);
        Ok(Self { source_sha256 })
    }
}

fn decode_selected_m16_navigation_tensor(
    bytes: &[u8],
) -> Result<(Vec<f32>, Map2NavigationInitializationProvenance), CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if metadata.metadata().as_ref() != Some(&selected_m16_navigation_metadata()) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let parameters = decode_tensor_count(&tensors, "model.parameters", M16_PARAMETERS)?;
    let digest = sha256(bytes);
    let provenance = Map2NavigationInitializationProvenance::from_source_sha256(digest)?;
    assert_eq!(parameters.len(), M16_PARAMETERS);
    assert_eq!(provenance.source_sha256(), digest);
    Ok((parameters, provenance))
}

fn selected_m16_navigation_metadata() -> HashMap<String, String> {
    // Freeze the old execution tuple; current linked hashes must never be substituted here.
    [
        ("action_schema_hash", "281345351372519059"),
        ("feature_schema_hash", "16612223928593971806"),
        ("model_schema_hash", "16105106472474017042"),
        ("ppo_schema_version", "29"),
        ("ppo_schema_hash", "6915425029811947603"),
        ("ppo_rules_audit_version", "24"),
        ("map2_reward_schema_version", "1"),
        ("map2_reward_schema_hash", "798798703797057220"),
        (
            "map2_reward_schema_descriptor",
            crate::MAP2_REWARD_SCHEMA_DESCRIPTOR,
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}
