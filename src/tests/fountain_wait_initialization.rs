use super::map2_checkpoint::{Directory, runtime_bytes};
use super::map2_model_initialization::{assert_bits, assert_fresh_state};
use crate::{CheckpointError, PolicyDevice, PolicyModel, TrainingArtifact};
use std::collections::HashMap;

pub(super) const M17_PARAMETERS: usize = 1_696_436;

#[test]
fn wait_source_metadata_keeps_original_reward1_descriptor_independent_of_current_reward() {
    let metadata = m17_metadata();
    let descriptor = metadata
        .get("map2_reward_schema_descriptor")
        .expect("source descriptor");
    let hash = descriptor
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(hash, 798_798_703_797_057_220);
    assert_eq!(metadata["map2_reward_schema_version"], "1");
    assert!(descriptor.starts_with("drysua-map2-reward/v1;"));
    assert_ne!(descriptor, crate::MAP2_REWARD_SCHEMA_DESCRIPTOR);
    assert_eq!(crate::MAP2_REWARD_SCHEMA_HASH, 11_643_768_462_079_275_437);
}

#[test]
fn wait_resume_rejects_checkpoint5_without_reading_tensors() {
    let directory = Directory::new();
    let mut prefix = b"DRYCKP18".to_vec();
    prefix.extend(5u32.to_le_bytes());
    prefix.extend(crate::CHECKPOINT_SCHEMA_HASH.to_le_bytes());
    std::fs::write(directory.0.join("checkpoint.meta"), &prefix).expect("old header");
    let error = TrainingArtifact::load(&directory.0).expect_err("no old resume");
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        std::fs::read(directory.0.join("checkpoint.meta")).expect("unchanged"),
        prefix
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

#[test]
fn wait_m17_initializer_requires_all_original_keys_without_missing_extra_or_current_descriptor() {
    let directory = Directory::new();
    let values = vec![0.0; M17_PARAMETERS];
    let original = m17_metadata();
    let mut keys: Vec<_> = original.keys().collect();
    keys.sort();
    for key in keys {
        for value in [None, Some("wrong")] {
            let mut metadata = original.clone();
            metadata.remove(key);
            if let Some(value) = value {
                metadata.insert(key.clone(), value.into());
            }
            assert_source_error(
                &directory,
                runtime_bytes(&values, metadata),
                CheckpointError::SchemaMismatch,
            );
        }
    }
    let mut metadata = original.clone();
    metadata.insert("extra".into(), "value".into());
    assert_source_error(
        &directory,
        runtime_bytes(&values, metadata),
        CheckpointError::SchemaMismatch,
    );
    let mut metadata = original;
    metadata.insert(
        "map2_reward_schema_descriptor".into(),
        crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.into(),
    );
    assert_source_error(
        &directory,
        runtime_bytes(&values, metadata),
        CheckpointError::SchemaMismatch,
    );
}

#[test]
fn wait_m17_initializer_rejects_names_dtype_rank_count_and_extra_tensor() {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let directory = Directory::new();
    let data = vec![0; (M17_PARAMETERS + 1) * 4];
    for (name, dtype, shape, field) in [
        ("wrong", Dtype::F32, vec![M17_PARAMETERS], "names"),
        (
            "model.parameters",
            Dtype::I32,
            vec![M17_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![1, M17_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![M17_PARAMETERS - 1],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![M17_PARAMETERS + 1],
            "dtype or shape",
        ),
    ] {
        let count = shape.iter().product::<usize>();
        let tensor = TensorView::new(dtype, shape, &data[..count * 4]).expect("tensor");
        assert_source_error(
            &directory,
            serialize([(name, tensor)], Some(m17_metadata())).expect("serialize"),
            CheckpointError::TensorContract(field),
        );
    }
    let tensor = TensorView::new(
        Dtype::F32,
        vec![M17_PARAMETERS],
        &data[..M17_PARAMETERS * 4],
    )
    .expect("tensor");
    let extra = TensorView::new(Dtype::F32, vec![1], &data[..4]).expect("extra");
    assert_source_error(
        &directory,
        serialize(
            [("model.parameters", tensor), ("extra", extra)],
            Some(m17_metadata()),
        )
        .expect("serialize"),
        CheckpointError::TensorContract("names"),
    );
}

#[test]
fn wait_padded_model_roundtrips_with_fresh_optimizer_and_progress() {
    use super::map2_checkpoint::{config, progress, run};
    let directory = Directory::new();
    let model = PolicyModel::fresh(9131803).expect("model");
    let values = model
        .widen_map2_wait_parameters(&vec![0.01; M17_PARAMETERS])
        .expect("pad");
    model.import_parameters(&values).expect("import");
    let trainer = crate::PpoTrainer::new(&model, config(), 1).expect("trainer");
    let artifact = TrainingArtifact::capture(&model, &trainer, run(), progress()).expect("capture");
    artifact.save(&directory.0).expect("save synthetic");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("runtime");
    let runtime = PolicyModel::fresh(9131804).expect("runtime");
    TrainingArtifact::load_runtime_weights(&runtime, &directory.0).expect("current reload");
    assert_bits(&runtime.export_parameters().expect("runtime bits"), &values);
    let loaded = TrainingArtifact::load(&directory.0).expect("checkpoint");
    let restored = loaded.restore(&runtime, &run()).expect("restore");
    assert_fresh_state(&runtime, restored.trainer());
    assert_eq!(loaded.progress(), &progress());
    assert_bits(
        &runtime.export_parameters().expect("restored bits"),
        &values,
    );
}

#[test]
fn wait_pinned_provenance_accepts_only_exact_initial_and_recovery004_digests() {
    for text in [
        "05e78663dd45ac23ad6c0242a69d8b8e45c163f7861c59154e3f3fd81de9ab1f",
        "107e19e61794c457ce8ec6adc2b3e66ccce08ee5b6544cb2005964f3e7dfedc8",
    ] {
        let digest = std::array::from_fn(|index| {
            u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("digest")
        });
        let provenance =
            crate::Map2WaitInitializationProvenance::from_source_sha256(digest).expect("pin");
        assert_eq!(provenance.source_sha256(), digest);
        let description = provenance.description();
        for field in [
            text,
            "INITIALIZATION_ONLY",
            "source_reward_version=1",
            "reward_version=3",
            "target_f=17 target_a=5 target_m=19 target_ppo=32 target_rules=27",
            "new_weights=2560_positive_zero",
            "optimizer_progress_rng_league=fresh",
            "GAMEPLAY_EQUIVALENCE=false",
            "qualification=false",
        ] {
            assert!(description.contains(field), "{field}");
        }
        for index in 0..32 {
            let mut changed = digest;
            changed[index] ^= 1;
            let error = crate::Map2WaitInitializationProvenance::from_source_sha256(changed)
                .expect_err("only exact pins");
            assert_eq!(
                error.to_string(),
                "checkpoint tensor contract has invalid selected M17 wait initialization source SHA-256"
            );
        }
    }
}

fn assert_source_error(directory: &Directory, bytes: Vec<u8>, expected: CheckpointError) {
    let path = directory.0.join("drysua.weights.safetensors");
    std::fs::write(&path, &bytes).expect("source");
    let error =
        TrainingArtifact::initialize_selected_m17_for_map2_wait(&directory.0, 1, PolicyDevice::Cpu)
            .err()
            .expect("rejected");
    let message = match &expected {
        CheckpointError::SchemaMismatch => "checkpoint schema does not match this build".to_owned(),
        CheckpointError::TensorContract(field) => {
            format!("checkpoint tensor contract has invalid {field}")
        }
        _ => panic!("unsupported fixture"),
    };
    assert_eq!(error, expected);
    assert_eq!(error.to_string(), message);
    assert_eq!(std::fs::read(path).expect("source unchanged"), bytes);
}

#[test]
fn wait_schemas_change_actor_and_training_identity_without_changing_actions_or_units() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 17);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 19);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 32);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 27);
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 32);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 7);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 17);
    assert_eq!(crate::ACTION_SCHEMA_HASH, 10_658_390_830_565_586_343);
    assert_eq!(crate::MODEL_PARAMETER_COUNT, 1_698_996);
}

#[test]
fn wait_runtime_rejects_m17_before_mutating_parameters_identity_or_optimizer() {
    let directory = Directory::new();
    let bytes = runtime_bytes(&vec![0.0; M17_PARAMETERS], m17_metadata());
    let path = directory.0.join("drysua.weights.safetensors");
    std::fs::write(&path, &bytes).expect("synthetic M17");
    let model = PolicyModel::fresh(9131800).expect("target");
    let before = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let trainer =
        crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 1).expect("trainer");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("old runtime");

    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_bits(&model.export_parameters().expect("after"), &before);
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_fresh_state(&model, &trainer);
    assert_eq!(std::fs::read(path).expect("source unchanged"), bytes);
}

#[test]
fn wait_initialization_preserves_every_m17_bit_and_inserts_positive_zero_at_global_row85() {
    let model = PolicyModel::fresh(9131801).expect("target");
    let mut source: Vec<_> = (0..M17_PARAMETERS)
        .map(|index| f32::from_bits(0x3e00_0000 + index as u32))
        .collect();
    source[0] = -0.0;
    source[M17_PARAMETERS - 1] = f32::MIN;
    let widened = model
        .widen_map2_wait_parameters(&source)
        .expect("Candle padding");
    assert_wait_padding(&model, &source, &widened);
    model.import_parameters(&widened).expect("import");
    let trainer = crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 1)
        .expect("fresh optimizer");
    assert_fresh_state(&model, &trainer);
    assert_bits(&model.export_parameters().expect("bits"), &widened);
}

pub(super) fn assert_wait_padding(model: &PolicyModel, source: &[f32], target: &[f32]) {
    assert_eq!(source.len(), M17_PARAMETERS);
    assert_eq!(target.len(), M17_PARAMETERS + 2560);
    let mut old_offset = 0;
    let mut new_offset = 0;
    for (name, shape) in model.parameter_schema().expect("named layout") {
        let count = shape.iter().product::<usize>();
        let inserted = usize::from(name == "trunk.0.weight") * 2560;
        let old = &source[old_offset..old_offset + count - inserted];
        let new = &target[new_offset..new_offset + count];
        if inserted > 0 {
            assert_eq!(shape, [2594, 512]);
            assert_bits(&new[..85 * 512], &old[..85 * 512]);
            assert!(
                new[85 * 512..90 * 512]
                    .iter()
                    .all(|value| value.to_bits() == 0)
            );
            assert_bits(&new[90 * 512..], &old[85 * 512..]);
        } else {
            assert_bits(new, old);
        }
        old_offset += old.len();
        new_offset += new.len();
    }
    assert_eq!(old_offset, source.len());
    assert_eq!(new_offset, target.len());
}

#[test]
fn wait_initializer_rejects_unapproved_digest_and_nonfinite_synthetic_sources() {
    let directory = Directory::new();
    let mut values = vec![0.0; M17_PARAMETERS];
    for nonfinite in [None, Some(0), Some(M17_PARAMETERS - 1)] {
        if let Some(index) = nonfinite {
            values[index] = f32::NAN;
        }
        let bytes = runtime_bytes(&values, m17_metadata());
        std::fs::write(directory.0.join("drysua.weights.safetensors"), bytes)
            .expect("synthetic source");
        let error = TrainingArtifact::initialize_selected_m17_for_map2_wait(
            &directory.0,
            1,
            PolicyDevice::Cpu,
        )
        .err()
        .expect("reject");
        match nonfinite {
            Some(index) => {
                assert_eq!(
                    error,
                    CheckpointError::NonFiniteTensor {
                        name: "model.parameters",
                        index
                    }
                );
                assert_eq!(
                    error.to_string(),
                    format!(
                        "checkpoint tensor model.parameters contains non-finite value at {index}"
                    )
                );
                values[index] = 0.0;
            }
            None => assert_eq!(
                error.to_string(),
                "checkpoint tensor contract has invalid selected M17 wait initialization source SHA-256"
            ),
        }
    }
}

pub(super) fn m17_metadata() -> HashMap<String, String> {
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
            crate::checkpoint::legacy_reward::MAP2_REWARD_V1_DESCRIPTOR,
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}
