use super::map2_checkpoint::{Directory, runtime_bytes};
use crate::{CheckpointError, PolicyModel, TrainingArtifact};
use std::collections::HashMap;

#[test]
fn rebase_inference_versions_reject_old_effect_and_healing_interpretations() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 15);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 30);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 25);
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 30);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 5);
    assert_eq!(crate::GLOBAL_FEATURES, 85);
    assert_eq!(crate::UNIT_FEATURES, 84);
    assert!(crate::PPO_SCHEMA_DESCRIPTOR.contains("cap27900"));
    assert!(crate::LEAGUE_SCHEMA_DESCRIPTOR.contains("cap27900"));
}

#[test]
fn rebase_m15_runtime_rejects_before_mutating_parameters_identity_or_optimizer() {
    let model = PolicyModel::fresh(37).expect("model");
    let trainer = crate::PpoTrainer::new(&model, crate::PpoConfig::default(), 38).expect("trainer");
    let prior = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let directory = Directory::new();
    let bytes = runtime_bytes(&vec![0.0; 1_695_924], m15_metadata());
    std::fs::write(directory.0.join("drysua.weights.safetensors"), &bytes).expect("synthetic M15");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
        .expect_err("old effect schema");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(model.export_parameters().expect("parameters"), prior);
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert!(trainer.checkpoint_snapshot(&model).is_ok());
    assert_eq!(
        std::fs::read(directory.0.join("drysua.weights.safetensors")).expect("source"),
        bytes
    );
}

#[test]
fn rebase_m15_shape_cannot_be_relabelled_as_current_runtime() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(40).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("synthetic current");
    let current =
        std::fs::read(directory.0.join("drysua.weights.safetensors")).expect("current bytes");
    let (_, header) = safetensors::SafeTensors::read_metadata(&current).expect("metadata");
    let bytes = runtime_bytes(
        &vec![0.0; 1_695_924],
        header.metadata().clone().expect("metadata"),
    );
    std::fs::write(directory.0.join("drysua.weights.safetensors"), bytes).expect("wrong shape");
    let identity = model.policy_identity().expect("identity");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("no relabel");

    assert_eq!(
        error.to_string(),
        "checkpoint tensor contract has invalid dtype or shape"
    );
    assert_eq!(model.policy_identity().expect("identity"), identity);
}

#[test]
fn rebase_checkpoint3_rejects_before_reading_any_tensor_file() {
    let directory = Directory::new();
    let mut bytes = b"DRYCKP18".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(6_904_067_705_245_923_052u64.to_le_bytes());
    std::fs::write(directory.0.join("checkpoint.meta"), &bytes).expect("old manifest header");

    let error = TrainingArtifact::load(&directory.0).expect_err("no old resume");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        std::fs::read(directory.0.join("checkpoint.meta")).expect("unchanged header"),
        bytes
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

fn m15_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "14080316840523410707".to_owned()),
        ("feature_schema_hash", "612467982395246657".to_owned()),
        ("model_schema_hash", "149485500614302181".to_owned()),
        ("ppo_schema_version", "28".to_owned()),
        ("ppo_schema_hash", "16579842539143021978".to_owned()),
        ("ppo_rules_audit_version", "23".to_owned()),
        ("map2_reward_schema_version", "1".to_owned()),
        ("map2_reward_schema_hash", "798798703797057220".to_owned()),
        (
            "map2_reward_schema_descriptor",
            crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.to_owned(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}
