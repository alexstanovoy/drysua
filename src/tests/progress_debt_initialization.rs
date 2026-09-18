use super::map2_checkpoint::{Directory, runtime_bytes};
use super::map2_model_initialization::{assert_bits, assert_fresh_state};
use crate::{CheckpointError, PolicyModel, TrainingArtifact};
use std::collections::HashMap;

#[test]
fn current_progress_debt_shapes_and_action_contract_remain_unchanged() {
    crate::tests::support::assert_frozen_schema_versions();
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 37);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 12);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 22);
    assert_eq!(crate::MODEL_PARAMETER_COUNT, 1_700_020);
    assert_eq!(crate::ACTION_SCHEMA_HASH, 10_658_390_830_565_586_343);
    assert_eq!(crate::MAP2_REWARD_SCHEMA_HASH, 7_274_660_837_025_042_530);
}

#[test]
fn progress_debt_runtime_rejects_exact_m18_reward2_before_mutating_model_or_optimizer() {
    let directory = Directory::new();
    let path = directory.0.join("drysua.weights.safetensors");
    let bytes = runtime_bytes(&vec![0.0; 1_697_460], m18_metadata());
    std::fs::write(&path, &bytes).expect("synthetic previous runtime");
    let model = PolicyModel::fresh(9131900).expect("model");
    let before = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let trainer =
        crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 1).expect("trainer");
    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("old runtime");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_bits(&model.export_parameters().expect("after"), &before);
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_fresh_state(&model, &trainer);
    assert_eq!(std::fs::read(path).expect("source unchanged"), bytes);
}

#[test]
fn progress_debt_resume_rejects_checkpoint6_before_tensor_access() {
    let directory = Directory::new();
    let mut bytes = b"DRYCKP18".to_vec();
    bytes.extend(6u32.to_le_bytes());
    bytes.extend(16_772_919_360_388_607_733u64.to_le_bytes());
    std::fs::write(directory.0.join("checkpoint.meta"), &bytes).expect("old header");
    let error = TrainingArtifact::load(&directory.0).expect_err("old checkpoint");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        std::fs::read(directory.0.join("checkpoint.meta")).expect("unchanged"),
        bytes
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

#[test]
fn current_padding_from_m17_zeroes_seven_global_rows_and_preserves_all_old_bits() {
    let model = PolicyModel::fresh(9131901).expect("model");
    let mut source: Vec<_> = (0..1_696_436)
        .map(|index| f32::from_bits(0x3e00_0000 + index))
        .collect();
    for (index, value) in [
        (0, -0.0),
        (59_072 + 85 * 512 - 1, f32::from_bits(1)),
        (59_072 + 85 * 512, f32::from_bits(0x8000_0001)),
        (1_696_435, f32::MAX),
    ] {
        source[index] = value;
    }
    let widened = model
        .widen_map2_wait_parameters(&source)
        .expect("Candle padding");
    assert_eq!(widened.len(), 1_700_020);
    let prefix = 59_072 + 85 * 512;
    assert_bits(&widened[..prefix], &source[..prefix]);
    assert!(
        widened[prefix..prefix + 3584]
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert_bits(&widened[prefix + 3584..], &source[prefix..]);
    assert_eq!(
        model
            .parameter_schema()
            .expect("schema")
            .iter()
            .find(|(name, _)| *name == "trunk.0.weight")
            .expect("trunk")
            .1,
        [2596, 512]
    );
}

#[test]
fn progress_debt_old_reward2_literal_stays_frozen_without_asserting_a_symmetric_bound() {
    let descriptor = m18_metadata()
        .remove("map2_reward_schema_descriptor")
        .expect("old descriptor");
    let hash = descriptor
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(hash, 699_687_995_158_557_285);
    assert_ne!(descriptor, crate::MAP2_REWARD_SCHEMA_DESCRIPTOR);
    assert_eq!(crate::MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES, 8);
}

fn m18_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "10658390830565586343"),
        ("feature_schema_hash", "17888785275670453418"),
        ("model_schema_hash", "3900982062969752096"),
        ("ppo_schema_version", "31"),
        ("ppo_schema_hash", "15379677344330093698"),
        ("ppo_rules_audit_version", "26"),
        ("map2_reward_schema_version", "2"),
        ("map2_reward_schema_hash", "699687995158557285"),
        (
            "map2_reward_schema_descriptor",
            crate::checkpoint::legacy_reward::MAP2_REWARD_V2_DESCRIPTOR,
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}
