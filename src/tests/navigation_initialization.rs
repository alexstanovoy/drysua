use std::collections::HashMap;
use std::fs;

use super::map2_checkpoint::{Directory, runtime_bytes};
use super::map2_model_initialization::assert_bits;
use crate::{CheckpointError, PolicyDevice, PolicyModel, TrainingArtifact};

const M16_PARAMETERS: usize = 1_696_436;

fn m16_metadata() -> HashMap<String, String> {
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

#[test]
fn navigation_schema_changes_all_execution_identities_without_changing_shapes() {
    eprintln!(
        "navigation_schema A{}={} F{}={} M{}={} PPO{}={} rules={} League{}={} Checkpoint{}={} reward{}={}",
        crate::ACTION_SCHEMA_VERSION,
        crate::ACTION_SCHEMA_HASH,
        crate::FEATURE_SCHEMA_VERSION,
        crate::FEATURE_SCHEMA_HASH,
        crate::MODEL_SCHEMA_VERSION,
        crate::MODEL_SCHEMA_HASH,
        crate::PPO_SCHEMA_VERSION,
        crate::PPO_SCHEMA_HASH,
        crate::PPO_RULES_AUDIT_VERSION,
        crate::LEAGUE_SCHEMA_VERSION,
        crate::LEAGUE_SCHEMA_HASH,
        crate::CHECKPOINT_SCHEMA_VERSION,
        crate::CHECKPOINT_SCHEMA_HASH,
        crate::MAP2_REWARD_SCHEMA_VERSION,
        crate::MAP2_REWARD_SCHEMA_HASH
    );
    assert_eq!(crate::ACTION_SCHEMA_VERSION, 5);
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 15);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 30);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 25);
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 30);
    assert_eq!(crate::LEAGUE_RULES_AUDIT_VERSION, 25);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 5);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 15);
    assert_eq!(crate::GLOBAL_FEATURES, 85);
    assert_eq!(crate::UNIT_FEATURES, 84);
    assert_eq!(crate::MODEL_PARAMETER_COUNT, M16_PARAMETERS);
    assert_eq!(crate::MAP2_REWARD_SCHEMA_VERSION, 1);
    assert_eq!(crate::MAP2_REWARD_SCHEMA_HASH, 798_798_703_797_057_220);
}

#[test]
fn navigation_runtime_rejects_old_a4_m16_even_with_identical_shape_before_mutation() {
    let directory = Directory::new();
    let bytes = runtime_bytes(&vec![0.125; M16_PARAMETERS], m16_metadata());
    let path = directory.0.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("synthetic historical runtime");
    let model = PolicyModel::fresh(10_091_700).expect("fresh target");
    let identity = model.policy_identity().expect("identity");
    let before = model.export_parameters().expect("before");
    let trainer = crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 10_091_707)
        .expect("bound optimizer before failed load");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
        .expect_err("old legal set cannot load as current runtime");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_bits(&model.export_parameters().expect("after"), &before);
    super::map2_model_initialization::assert_fresh_state(&model, &trainer);
    assert_eq!(fs::read(path).expect("source unchanged"), bytes);
}

#[test]
fn navigation_resume_rejects_each_old_link_even_if_checkpoint_header_is_relabelled() {
    use super::map2_checkpoint::{config, progress, run};
    let directory = Directory::new();
    let model = PolicyModel::fresh(10_091_708).expect("current model");
    let trainer = crate::PpoTrainer::new(&model, config(), 10_091_709).expect("bound optimizer");
    TrainingArtifact::capture(&model, &trainer, run(), progress())
        .expect("capture")
        .save(&directory.0)
        .expect("synthetic current checkpoint");
    let path = directory.0.join("checkpoint.meta");
    let original = fs::read(&path).expect("manifest");
    let identity = model.policy_identity().expect("identity");
    for (index, version, hash) in [
        (0, 4u32, 281_345_351_372_519_059u64),
        (1, 14, 16_612_223_928_593_971_806),
        (2, 16, 16_105_106_472_474_017_042),
        (3, 29, 6_915_425_029_811_947_603),
    ] {
        let mut changed = original.clone();
        let offset = 8 + 4 + 8 + index * 12;
        changed[offset..offset + 4].copy_from_slice(&version.to_le_bytes());
        changed[offset + 4..offset + 12].copy_from_slice(&hash.to_le_bytes());
        fs::write(&path, &changed).expect("old linked identity");
        let error = TrainingArtifact::load_compatible(&directory.0, &run()).expect_err("old link");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(fs::read(&path).expect("unchanged manifest"), changed);
    }
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
    super::map2_model_initialization::assert_fresh_state(&model, &trainer);
}

#[test]
fn navigation_resume_rejects_m16_checkpoint_v4_before_tensor_access() {
    let directory = Directory::new();
    let mut manifest = b"DRYCKP18".to_vec();
    manifest.extend(4u32.to_le_bytes());
    manifest.extend(11_217_481_393_624_496_123u64.to_le_bytes());
    fs::write(directory.0.join("checkpoint.meta"), &manifest).expect("historical manifest prefix");

    let error = TrainingArtifact::load(&directory.0).expect_err("M16 cannot resume M17");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        fs::read(directory.0.join("checkpoint.meta")).expect("unchanged"),
        manifest
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

#[test]
fn navigation_initializer_rejects_missing_wrong_extra_and_current_metadata() {
    let directory = Directory::new();
    let values = vec![0.0; M16_PARAMETERS];
    let metadata = m16_metadata();
    let mut keys: Vec<_> = metadata.keys().collect();
    keys.sort();
    for key in keys {
        for replacement in [None, Some("wrong")] {
            let mut changed = metadata.clone();
            changed.remove(key);
            if let Some(value) = replacement {
                changed.insert(key.clone(), value.to_owned());
            }
            assert_initializer_error(
                &directory,
                runtime_bytes(&values, changed),
                CheckpointError::SchemaMismatch,
            );
        }
    }
    let mut extra = metadata;
    extra.insert("navigation_relabel".to_owned(), "true".to_owned());
    assert_initializer_error(
        &directory,
        runtime_bytes(&values, extra),
        CheckpointError::SchemaMismatch,
    );
    let model = PolicyModel::fresh(10_091_701).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("current test runtime");
    let bytes = fs::read(directory.0.join("drysua.weights.safetensors")).expect("current bytes");
    assert_initializer_error(&directory, bytes, CheckpointError::SchemaMismatch);
}

#[test]
fn navigation_initializer_rejects_unpinned_sha_and_nonfinite_boundary_values() {
    let directory = Directory::new();
    let mut values = vec![0.0; M16_PARAMETERS];
    assert_initializer_error(
        &directory,
        runtime_bytes(&values, m16_metadata()),
        CheckpointError::TensorContract("selected M16 navigation initialization source SHA-256"),
    );
    for index in [0, M16_PARAMETERS - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            values[index] = value;
            assert_initializer_error(
                &directory,
                runtime_bytes(&values, m16_metadata()),
                CheckpointError::NonFiniteTensor {
                    name: "model.parameters",
                    index,
                },
            );
        }
        values[index] = 0.0;
    }
}

#[test]
fn navigation_initializer_rejects_tensor_names_dtype_rank_count_and_extra_tensor() {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let directory = Directory::new();
    let data = vec![0u8; (M16_PARAMETERS + 1) * 4];
    for (name, dtype, shape, field) in [
        ("wrong", Dtype::F32, vec![M16_PARAMETERS], "names"),
        (
            "model.parameters",
            Dtype::I32,
            vec![M16_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![1, M16_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![M16_PARAMETERS - 1],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![M16_PARAMETERS + 1],
            "dtype or shape",
        ),
    ] {
        let count = shape.iter().product::<usize>();
        let tensor = TensorView::new(dtype, shape, &data[..count * 4]).expect("tensor");
        let bytes = serialize([(name, tensor)], Some(m16_metadata())).expect("fixture");
        assert_initializer_error(&directory, bytes, CheckpointError::TensorContract(field));
    }
    let tensor = TensorView::new(
        Dtype::F32,
        vec![M16_PARAMETERS],
        &data[..M16_PARAMETERS * 4],
    )
    .expect("tensor");
    let extra = TensorView::new(Dtype::F32, vec![1], &data[..4]).expect("extra");
    let bytes = serialize(
        [("model.parameters", tensor), ("extra", extra)],
        Some(m16_metadata()),
    )
    .expect("fixture");
    assert_initializer_error(&directory, bytes, CheckpointError::TensorContract("names"));
}

fn assert_initializer_error(directory: &Directory, bytes: Vec<u8>, expected: CheckpointError) {
    let path = directory.0.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("synthetic source");
    let error = TrainingArtifact::initialize_selected_m16_for_map2_navigation(
        &directory.0,
        10_091_702,
        PolicyDevice::Cpu,
    )
    .err()
    .expect("rejected initializer");
    let message = match expected {
        CheckpointError::SchemaMismatch => "checkpoint schema does not match this build".to_owned(),
        CheckpointError::TensorContract(field) => {
            format!("checkpoint tensor contract has invalid {field}")
        }
        CheckpointError::NonFiniteTensor { name, index } => {
            format!("checkpoint tensor {name} contains non-finite value at {index}")
        }
        _ => panic!("unsupported expected error"),
    };
    assert_eq!(error, expected);
    assert_eq!(error.to_string(), message);
    assert_eq!(fs::read(path).expect("unchanged source"), bytes);
}

#[test]
fn navigation_m16_copy_preserves_all_parameter_bits_and_has_fresh_training_state() {
    use super::map2_checkpoint::{config, progress, run};
    use super::map2_model_initialization::assert_fresh_state;
    let model = PolicyModel::fresh(10_091_703).expect("new owned model");
    let mut source: Vec<_> = (0..M16_PARAMETERS)
        .map(|index| f32::from_bits(0x3e00_0000 + index as u32))
        .collect();
    for (index, value) in [
        (0, -0.0),
        (84 * 64 - 1, f32::from_bits(1)),
        (84 * 64, f32::from_bits(0x8000_0001)),
        (M16_PARAMETERS - 1, f32::MAX),
    ] {
        source[index] = value;
    }

    model
        .initialize_m16_navigation_parameters(&source)
        .expect("same-shape Candle import");

    assert_eq!(model.parameter_schema().expect("layout").len(), 62);
    assert_bits(&model.export_parameters().expect("all bits"), &source);
    let trainer = crate::PpoTrainer::new(&model, config(), 10_091_704).expect("fresh optimizer");
    assert_fresh_state(&model, &trainer);
    let directory = Directory::new();
    let artifact =
        TrainingArtifact::capture(&model, &trainer, run(), progress()).expect("fresh capture");
    artifact
        .save(&directory.0)
        .expect("synthetic current checkpoint");
    let loaded = TrainingArtifact::load_compatible(&directory.0, &run()).expect("current reload");
    assert_eq!(loaded.progress(), &progress());
    let restored = PolicyModel::fresh(10_091_705).expect("new target");
    let state = loaded.restore(&restored, &run()).expect("current restore");
    assert_fresh_state(&restored, state.trainer());
    assert_bits(
        &restored.export_parameters().expect("restored bits"),
        &source,
    );
    assert_ne!(
        model.policy_identity().expect("identity"),
        restored.policy_identity().expect("identity")
    );
}

#[test]
fn navigation_m16_copy_rejects_invalid_values_before_parameter_or_identity_mutation() {
    let model = PolicyModel::fresh(10_091_706).expect("model");
    let before = model.export_parameters().expect("before");
    let identity = model.policy_identity().expect("identity");
    for count in [0, M16_PARAMETERS - 1, M16_PARAMETERS + 1] {
        let error = model
            .initialize_m16_navigation_parameters(&vec![0.0; count])
            .expect_err("length");
        assert_eq!(
            error,
            crate::ModelError::ParameterLength {
                actual: count,
                expected: M16_PARAMETERS
            }
        );
        assert_eq!(
            error.to_string(),
            format!("model parameter length {count} differs from expected {M16_PARAMETERS}")
        );
    }
    let mut values = vec![0.0; M16_PARAMETERS];
    for index in [0, M16_PARAMETERS - 1] {
        values[index] = f32::NAN;
        let error = model
            .initialize_m16_navigation_parameters(&values)
            .expect_err("nonfinite");
        assert_eq!(error, crate::ModelError::NonFiniteParameter { index });
        assert_eq!(
            error.to_string(),
            format!("model parameter {index} is non-finite")
        );
        values[index] = 0.0;
    }
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_bits(&model.export_parameters().expect("after"), &before);
}

#[test]
fn navigation_initialization_whitelists_only_two_exact_m16_digests_and_disclaims_equivalence() {
    for digest in [
        "1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4",
        "fdd4d3f2a85a64d9b4d162f1607a94a4aaba8a8e3136e51dab2152181cdf63b7",
    ] {
        let bytes: [u8; 32] = std::array::from_fn(|index| {
            u8::from_str_radix(&digest[index * 2..index * 2 + 2], 16).expect("pinned hex")
        });
        let provenance = crate::Map2NavigationInitializationProvenance::from_source_sha256(bytes)
            .expect("explicitly authorized source identity");
        assert_eq!(provenance.source_sha256(), bytes);
        let text = provenance.description();
        for field in [
            format!("INITIALIZATION_ONLY source_m16_sha256={digest}"),
            "source_f=14 source_a=4 source_m=16 source_ppo=29 source_rules=24".to_owned(),
            "target_f=15 target_a=5 target_m=17 target_ppo=30 target_rules=25".to_owned(),
            "parameter_bits_preserved=true named_tensors=62 new_weights=0".to_owned(),
            "optimizer_progress_rng_league=fresh GAMEPLAY_EQUIVALENCE=false new_legal_actions_change_behavior=true qualification=false".to_owned(),
        ] { assert!(text.contains(&field), "missing {field}"); }
        for index in 0..32 {
            let mut corrupted = bytes;
            corrupted[index] ^= 1;
            let error =
                crate::Map2NavigationInitializationProvenance::from_source_sha256(corrupted)
                    .expect_err("one changed byte is not selected");
            assert_eq!(
                error,
                CheckpointError::TensorContract(
                    "selected M16 navigation initialization source SHA-256"
                )
            );
            assert_eq!(
                error.to_string(),
                "checkpoint tensor contract has invalid selected M16 navigation initialization source SHA-256"
            );
        }
    }
}
