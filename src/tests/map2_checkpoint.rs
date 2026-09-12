use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::MapId;
use safetensors::tensor::{Dtype, TensorView, serialize};

use crate::{CheckpointError, PolicyModel, TrainingArtifact};

pub(super) use self::schema_hash as independent_linked_hash;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
pub(super) const M14_PARAMETERS: usize = 1_689_076;

pub(super) struct Directory(pub PathBuf);

impl Directory {
    pub(super) fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "drysua-map2-checkpoint-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("unique fixture directory");
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("fixture cleanup");
    }
}

pub(super) fn m14_metadata(version: u32) -> HashMap<String, String> {
    let (hash, rules) = match version {
        26 => ("4420330489262074980", "21"),
        27 => ("9274275648898675046", "22"),
        _ => panic!("fixture version"),
    };
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "1577122233561586211"),
        ("model_schema_hash", "7970187849195607202"),
        (
            "ppo_schema_version",
            if version == 26 { "26" } else { "27" },
        ),
        ("ppo_schema_hash", hash),
        ("ppo_rules_audit_version", rules),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

pub(super) fn runtime_bytes(values: &[f32], metadata: HashMap<String, String>) -> Vec<u8> {
    let data: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = TensorView::new(Dtype::F32, vec![values.len()], &data).expect("tensor");
    serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture")
}

#[test]
fn map2_semantic_versions_and_shapes_are_new_not_m14_relabels() {
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 30);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 25);
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 30);
    assert_eq!(crate::LEAGUE_RULES_AUDIT_VERSION, 25);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 5);
    assert_eq!(
        crate::MODEL_PARAMETER_COUNT,
        M14_PARAMETERS + 11 * 64 + 13 * 512
    );
}

#[test]
fn map2_runtime_rejects_both_old_m14_semantic_tuples_without_mutation() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(190).expect("model");
    let identity = model.policy_identity().expect("identity");
    let before = model.export_parameters().expect("parameters");
    for version in [26, 27] {
        let bytes = runtime_bytes(&vec![0.0; M14_PARAMETERS], m14_metadata(version));
        let path = directory.0.join("drysua.weights.safetensors");
        fs::write(&path, &bytes).expect("source");
        let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
            .expect_err("old runtime must not become Map2");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(model.policy_identity().expect("identity"), identity);
        assert_eq!(model.export_parameters().expect("parameters"), before);
        assert_eq!(fs::read(path).expect("unchanged source"), bytes);
    }
}

#[test]
fn map2_resume_rejects_checkpoint_v2_before_reading_tensors() {
    let directory = Directory::new();
    let mut manifest = b"DRYCKP18".to_vec();
    manifest.extend(2u32.to_le_bytes());
    manifest.extend(crate::CHECKPOINT_SCHEMA_HASH.to_le_bytes());
    fs::write(directory.0.join("checkpoint.meta"), manifest).expect("manifest");
    let error = TrainingArtifact::load(&directory.0).expect_err("v2 cannot resume Map2");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
}

pub(super) fn config() -> crate::PpoConfig {
    crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        rollout_decisions: 2,
        environments: 2,
        minibatch: 2,
        ..crate::PpoConfig::default()
    }
}

pub(super) fn progress() -> crate::CheckpointProgress {
    crate::CheckpointProgress {
        global_update: 0,
        policy_version: 0,
        scheduler_step: 0,
        curriculum_stage: 0,
        rollout_samples: 0,
        best_evaluation: None,
        rng_states: Vec::new(),
        league_references: Vec::new(),
    }
}

pub(super) fn run() -> crate::CheckpointRun {
    crate::CheckpointRun {
        git_commit: "map2-fixture-source".to_owned(),
        simulator_commit: "map2-fixture-simulator".to_owned(),
        enabled_features: crate::compiled_features(),
        command_line: "map2 initialization only; no gameplay equivalence".to_owned(),
        run_seed: 191,
        map: MapId(2),
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config().minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

#[test]
fn map2_checkpoint_accepts_only_map2_run_scope() {
    let model = PolicyModel::fresh(191).expect("model");
    let trainer = crate::PpoTrainer::new(&model, config(), 192).expect("fresh optimizer");
    TrainingArtifact::capture(&model, &trainer, run(), progress()).expect("Map2 capture");
    for map in [MapId(0), MapId(1), MapId(3)] {
        let mut run = run();
        run.map = map;
        let error = TrainingArtifact::capture(&model, &trainer, run, progress())
            .expect_err("non-Map2 scope");
        assert_eq!(error, CheckpointError::InvalidManifest("hero or map scope"));
        assert_eq!(
            error.to_string(),
            "checkpoint manifest has invalid hero or map scope"
        );
    }
}

#[test]
fn map2_checkpoint_rejects_discount_incompatible_with_reward_potentials() {
    let model = PolicyModel::fresh(197).expect("model");
    let config = crate::PpoConfig {
        gamma_tick: 0.99,
        ..config()
    };
    let trainer = crate::PpoTrainer::new(&model, config, 198).expect("generic PPO optimizer");
    let error = TrainingArtifact::capture(&model, &trainer, run(), progress())
        .expect_err("Map2 gamma1 reward contract");
    assert_eq!(
        error,
        CheckpointError::InvalidManifest("Map2 reward discount")
    );
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid Map2 reward discount"
    );
}

#[test]
fn map2_runtime_metadata_requires_exact_reward_version_hash_and_descriptor() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(193).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("save");
    let path = directory.0.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("runtime");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    let metadata = metadata.metadata().as_ref().expect("metadata map");
    let parameters = model.export_parameters().expect("parameters");
    for (key, value) in [
        (
            "map2_reward_schema_version",
            crate::MAP2_REWARD_SCHEMA_VERSION.to_string(),
        ),
        (
            "map2_reward_schema_hash",
            crate::MAP2_REWARD_SCHEMA_HASH.to_string(),
        ),
        (
            "map2_reward_schema_descriptor",
            crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.to_owned(),
        ),
    ] {
        assert_eq!(metadata.get(key), Some(&value));
        for replacement in [None, Some("wrong".to_owned())] {
            let mut changed = metadata.clone();
            changed.remove(key);
            if let Some(value) = replacement {
                changed.insert(key.to_owned(), value);
            }
            fs::write(&path, runtime_bytes(&parameters, changed)).expect("altered metadata");
            let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
                .expect_err("reward identity cannot be omitted or changed");
            assert_eq!(error, CheckpointError::SchemaMismatch);
        }
    }
}

#[test]
fn map2_runtime_rejects_m14_shape_even_after_metadata_is_relabelled() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(194).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("save");
    let path = directory.0.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("runtime");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    let metadata = metadata.metadata().as_ref().expect("metadata map").clone();
    fs::write(&path, runtime_bytes(&vec![0.0; M14_PARAMETERS], metadata)).expect("relabel attempt");
    let identity = model.policy_identity().expect("identity");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("no relabel");

    assert_eq!(error, CheckpointError::TensorContract("dtype or shape"));
    assert_eq!(
        error.to_string(),
        "checkpoint tensor contract has invalid dtype or shape"
    );
    assert_eq!(model.policy_identity().expect("identity"), identity);
}

#[test]
fn map2_resume_checks_reward_identity_before_tensor_access() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(195).expect("model");
    let trainer = crate::PpoTrainer::new(&model, config(), 196).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run(), progress())
        .expect("capture")
        .save(&directory.0)
        .expect("save");
    let path = directory.0.join("checkpoint.meta");
    let original = fs::read(&path).expect("manifest");
    let offset = 8 + 4 + 8 + 4 * (4 + 8);
    for index in [offset, offset + 4] {
        let mut bytes = original.clone();
        bytes[index] ^= 1;
        fs::write(&path, bytes).expect("changed reward version or hash");
        let error =
            TrainingArtifact::load_compatible(&directory.0, &run()).expect_err("reward identity");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
    }
}

#[test]
fn map2_schema_hashes_fold_imported_identities_and_complete_reward_descriptor() {
    let action = (crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH);
    let feature = (crate::FEATURE_SCHEMA_VERSION, crate::FEATURE_SCHEMA_HASH);
    let model = (crate::MODEL_SCHEMA_VERSION, crate::MODEL_SCHEMA_HASH);
    let ppo = (crate::PPO_SCHEMA_VERSION, crate::PPO_SCHEMA_HASH);
    let reward = (
        crate::MAP2_REWARD_SCHEMA_VERSION,
        crate::MAP2_REWARD_SCHEMA_HASH,
    );
    for (descriptor, schemas, expected) in [
        (
            crate::MODEL_SCHEMA_DESCRIPTOR,
            vec![action, feature, reward],
            crate::MODEL_SCHEMA_HASH,
        ),
        (
            crate::PPO_SCHEMA_DESCRIPTOR,
            vec![action, feature, model, reward],
            crate::PPO_SCHEMA_HASH,
        ),
        (
            crate::LEAGUE_SCHEMA_DESCRIPTOR,
            vec![action, feature, model, ppo, reward],
            crate::LEAGUE_SCHEMA_HASH,
        ),
        (
            crate::CHECKPOINT_SCHEMA_DESCRIPTOR,
            vec![action, feature, model, ppo, reward],
            crate::CHECKPOINT_SCHEMA_HASH,
        ),
    ] {
        assert_eq!(schema_hash(descriptor, &schemas), expected);
        for index in 0..schemas.len() {
            let mut changed = schemas.clone();
            changed[index].1 ^= 1;
            assert_ne!(schema_hash(descriptor, &changed), expected);
        }
    }
    eprintln!(
        "Map2 schemas F{}={} A{}={} M{}={} PPO{}={} rules={} league{}={} checkpoint{}={} reward{}={}",
        feature.0,
        feature.1,
        action.0,
        action.1,
        model.0,
        model.1,
        ppo.0,
        ppo.1,
        crate::PPO_RULES_AUDIT_VERSION,
        crate::LEAGUE_SCHEMA_VERSION,
        crate::LEAGUE_SCHEMA_HASH,
        crate::CHECKPOINT_SCHEMA_VERSION,
        crate::CHECKPOINT_SCHEMA_HASH,
        reward.0,
        reward.1
    );
}

pub(super) fn schema_hash(descriptor: &str, schemas: &[(u32, u64)]) -> u64 {
    let mut bytes = descriptor.as_bytes().to_vec();
    for (version, hash) in schemas {
        bytes.extend(version.to_le_bytes());
        bytes.extend(hash.to_le_bytes());
    }
    bytes.extend(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.as_bytes());
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn map2_initialization_provenance_names_ancestry_and_disclaims_gameplay_equivalence() {
    let provenance = crate::Map2InitializationProvenance {
        source_sha256: [0x63; 32],
        source_ppo_schema_version: 27,
        source_ppo_rules_audit_version: 22,
    };
    let description = provenance.description();
    assert!(description.starts_with("INITIALIZATION_ONLY "));
    for field in [
        format!("source_m14_sha256={}", "63".repeat(32)),
        "source_ppo=27 source_rules=22".to_owned(),
        "target_f=15 target_a=5 target_m=17 target_ppo=30 target_rules=25".to_owned(),
        format!(
            "reward_version={} reward_hash={}",
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH
        ),
        "old_parameter_bits_preserved=true new_weights=7360_positive_zero".to_owned(),
        "optimizer_progress_rng_league=fresh gameplay_equivalence=false qualification=false"
            .to_owned(),
    ] {
        assert!(
            description.contains(&field),
            "missing provenance field {field}"
        );
    }
    assert!(description.len() < 4_096);
}
