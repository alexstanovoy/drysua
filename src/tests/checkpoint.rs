use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::{MapId, Team};
use safetensors::tensor::{Dtype, TensorView, serialize};

use super::feature::{encode, tracker_with_view, world_view};
use crate::{
    ActionSpace, CheckpointDevice, CheckpointError, CheckpointProgress, CheckpointRun,
    LocalPolicyState, PolicyDevice, PolicyModel, PpoConfig, PpoOutcome, PpoRng, PpoRollout,
    PpoTrainer, RngCheckpoint, TrainingArtifact,
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

const M11_PARAMETERS: usize = 1_684_724;
#[path = "checkpoint_input_adapter.rs"]
mod checkpoint_input_adapter;
#[path = "checkpoint_order_contract.rs"]
mod checkpoint_order_contract;
#[path = "checkpoint_training_contract.rs"]
mod checkpoint_training_contract;

fn selected_m10_metadata() -> std::collections::HashMap<String, String> {
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "9669721049329356661"),
        ("model_schema_hash", "720439888929233033"),
        ("ppo_schema_version", "18"),
        ("ppo_schema_hash", "6877503070358232325"),
        ("ppo_rules_audit_version", "15"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

#[test]
fn m10_training_initialization_rejects_wrong_or_partial_metadata_and_tensor_contracts() {
    let directory = test_directory("m10-initialization-contract");
    let data = vec![0u8; M11_PARAMETERS * 4];
    for field in selected_m10_metadata().keys() {
        for replacement in [None, Some("0")] {
            let mut metadata = selected_m10_metadata();
            metadata.remove(field);
            if let Some(value) = replacement {
                metadata.insert(field.clone(), value.to_owned());
            }
            let tensor = TensorView::new(Dtype::F32, vec![M11_PARAMETERS], &data).expect("tensor");
            let bytes = serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture");
            fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");

            let error = TrainingArtifact::initialize_selected_m10_for_training(
                &directory,
                1,
                crate::PolicyDevice::Cpu,
            )
            .err()
            .expect("exact old tuple required");

            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
    }
    for (name, dtype, count, expected) in [
        ("wrong", Dtype::F32, M11_PARAMETERS, "names"),
        (
            "model.parameters",
            Dtype::I32,
            M11_PARAMETERS,
            "dtype or shape",
        ),
        ("model.parameters", Dtype::F32, 1, "dtype or shape"),
        (
            "model.parameters",
            Dtype::F32,
            M11_PARAMETERS,
            "selected M10 training source SHA-256",
        ),
    ] {
        let tensor = TensorView::new(dtype, vec![count], &data[..count * 4]).expect("tensor");
        let bytes = serialize([(name, tensor)], Some(selected_m10_metadata())).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");

        let error = TrainingArtifact::initialize_selected_m10_for_training(
            &directory,
            1,
            crate::PolicyDevice::Cpu,
        )
        .err()
        .expect("tensor contract or exact selected SHA required");

        assert_eq!(error, CheckpointError::TensorContract(expected));
        assert_eq!(
            error.to_string(),
            format!("checkpoint tensor contract has invalid {expected}")
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn m10_training_initialization_rejects_nonfinite_parameters() {
    let directory = test_directory("m10-initialization-nonfinite");
    let mut data = vec![0u8; M11_PARAMETERS * 4];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        data[..4].copy_from_slice(&value.to_le_bytes());
        let tensor = TensorView::new(Dtype::F32, vec![M11_PARAMETERS], &data).expect("tensor");
        let bytes = serialize(
            [("model.parameters", tensor)],
            Some(selected_m10_metadata()),
        )
        .expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");

        let error = TrainingArtifact::initialize_selected_m10_for_training(
            &directory,
            1,
            crate::PolicyDevice::Cpu,
        )
        .err()
        .expect("nonfinite old weights");

        assert_eq!(
            error.to_string(),
            "checkpoint tensor model.parameters contains non-finite value at 0"
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
#[ignore = "requires DRYSUA_SELECTED_M10 directory containing the immutable b3802642 artifact"]
fn selected_m10_retired_initialization_and_runtime_reject_without_source_mutation() {
    use sha2::{Digest, Sha256};
    let directory =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M10").expect("selected fixture"));
    let path = directory.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("selected artifact");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        digest,
        "b3802642b34487d66fc3f0fe526e7f8b2a84df542793ac336b58ea046b8f8b53"
    );
    let runtime = PolicyModel::fresh(510).expect("runtime");
    let before = runtime.policy_identity().expect("identity");
    let parameters = runtime.export_parameters().expect("before parameters");

    let error = TrainingArtifact::load_runtime_weights(&runtime, &directory)
        .expect_err("M10 runtime incompatible");

    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(runtime.policy_identity().expect("identity"), before);
    assert_eq!(
        runtime.export_parameters().expect("after parameters"),
        parameters
    );
    let error = TrainingArtifact::initialize_selected_m10_for_training(
        &directory,
        511,
        crate::PolicyDevice::Cpu,
    )
    .err()
    .expect("M10 authorization does not permit Map2 initialization");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(fs::read(path).expect("source unchanged"), bytes);
}

const PRIOR_PPO_SCHEMAS: [(u32, u64, u32); 3] = [
    (13, 11_103_744_726_312_279_053, 12),
    (14, 15_610_409_340_106_916_160, 13),
    (15, 13_893_101_989_595_893_928, 14),
];

#[test]
fn exact_m12_ppo23_runtime_and_training_reject_before_mutation() {
    assert_m12_order_contract_rejection(23, 765_990_392_710_687_046, 18);
}

#[test]
fn exact_m12_ppo24_runtime_and_training_reject_before_mutation() {
    assert_m12_order_contract_rejection(24, 17_486_156_843_355_673_207, 19);
}

#[test]
fn exact_m12_ppo25_runtime_and_training_reject_before_mutation() {
    assert_m12_order_contract_rejection(25, 12_302_688_747_093_836_273, 20);
}

fn m12_metadata(version: u32, hash: u64, rules: u32) -> std::collections::HashMap<String, String> {
    action_three_metadata(
        "8078516161541333175",
        "17156054387874206897",
        version,
        hash,
        rules,
    )
}

fn action_three_metadata(
    feature: &str,
    model: &str,
    version: u32,
    hash: u64,
    rules: u32,
) -> std::collections::HashMap<String, String> {
    [
        ("action_schema_hash", "1755359086494840931".to_owned()),
        ("feature_schema_hash", feature.to_owned()),
        ("model_schema_hash", model.to_owned()),
        ("ppo_schema_version", version.to_string()),
        ("ppo_schema_hash", hash.to_string()),
        ("ppo_rules_audit_version", rules.to_string()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

#[test]
fn historical_m12_metadata_keeps_exact_six_key_a3_tuple_under_map2() {
    let metadata = m12_metadata(23, 765_990_392_710_687_046, 18);
    assert_eq!(metadata.len(), 6);
    for (key, value) in [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "8078516161541333175"),
        ("model_schema_hash", "17156054387874206897"),
        ("ppo_schema_version", "23"),
        ("ppo_schema_hash", "765990392710687046"),
        ("ppo_rules_audit_version", "18"),
    ] {
        assert_eq!(metadata.get(key).map(String::as_str), Some(value));
    }
    assert_eq!(current_runtime_metadata().len(), 9);
    assert!(!metadata.contains_key("map2_reward_schema_hash"));
}

fn assert_m12_order_contract_rejection(version: u32, hash: u64, rules: u32) {
    let directory = test_directory("m12-old-order-contract");
    let source = PolicyModel::fresh(18124).expect("model");
    let parameters = vec![0.0f32; 1_689_076];
    let data: Vec<_> = parameters
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let metadata = m12_metadata(version, hash, rules);
    let view = TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("view");
    let bytes = serialize([("model.parameters", view)], Some(metadata)).expect("runtime fixture");
    let path = directory.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("fixture");
    let target = PolicyModel::fresh(18125).expect("target");
    let identity = target.policy_identity().expect("identity");
    let original = target.export_parameters().expect("original");
    let target_trainer = PpoTrainer::new(&target, checkpoint_config(), 18127).expect("optimizer");
    let error = TrainingArtifact::load_runtime_weights(&target, &directory)
        .expect_err("legacy candidate order semantics must reject");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        target.policy_identity().expect("unchanged identity"),
        identity
    );
    assert_eq!(
        target.export_parameters().expect("unchanged parameters"),
        original
    );
    assert_eq!(target_trainer.optimizer_step(), 0);
    assert!(
        TrainingArtifact::capture(
            &target,
            &target_trainer,
            run_metadata(),
            progress_metadata(0),
        )
        .is_ok(),
        "rejection must preserve optimizer binding"
    );
    assert_eq!(fs::read(path).expect("unmodified"), bytes);
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18126).expect("trainer");
    TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let path = directory.join("checkpoint.meta");
    let mut manifest = fs::read(&path).expect("manifest");
    let offset = 8 + 4 + 8 + 3 * (4 + 8);
    manifest[offset..offset + 4].copy_from_slice(&version.to_le_bytes());
    manifest[offset + 4..offset + 12].copy_from_slice(&hash.to_le_bytes());
    fs::write(&path, &manifest).expect("old training fixture");
    let Err(error) = TrainingArtifact::load(&directory) else {
        panic!("old training version {version} must reject");
    };
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        TrainingArtifact::load_compatible(&directory, &run_metadata()).expect_err("strict resume"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(fs::read(path).expect("unchanged"), manifest);
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn exact_m11_ppo19_runtime_and_training_reject_after_input_widening() {
    assert_m11_runtime_only_compatibility(19, 3_810_026_640_568_905_163, 15);
}

#[test]
fn exact_m11_ppo20_runtime_and_training_reject_after_input_widening() {
    assert_m11_runtime_only_compatibility(20, 8_812_022_392_730_398_368, 16);
}

#[test]
fn exact_m11_ppo21_runtime_and_training_reject_after_input_widening() {
    assert_m11_runtime_only_compatibility(21, 768_751_058_595_344_501, 17);
}

#[test]
fn exact_m11_ppo22_runtime_and_training_reject_after_input_widening() {
    assert_m11_runtime_only_compatibility(22, 7_033_554_372_932_156_753, 18);
}

fn assert_m11_runtime_only_compatibility(version: u32, hash: u64, rules: u32) {
    let directory = test_directory("exact-m11-ppo19");
    let source = PolicyModel::fresh(18_121).expect("source");
    let parameters = vec![0.0f32; M11_PARAMETERS];
    let data: Vec<_> = parameters
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let metadata = action_three_metadata(
        "15519817897416174399",
        "18229126264156367519",
        version,
        hash,
        rules,
    );
    let view = TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("view");
    let bytes = serialize([("model.parameters", view)], Some(metadata)).expect("P19 fixture");
    let path = directory.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("fixture");
    let target = PolicyModel::fresh(18_122).expect("target");
    let original = target.export_parameters().expect("before");
    let identity = target.policy_identity().expect("identity");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&target, &directory)
            .expect_err("old inputs incompatible"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(
        target.export_parameters().expect("loaded parameters"),
        original
    );
    assert_eq!(
        target.policy_identity().expect("unchanged identity"),
        identity
    );
    assert_eq!(fs::read(&path).expect("unchanged bytes"), bytes);
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18_123).expect("trainer");
    TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let manifest_path = directory.join("checkpoint.meta");
    let mut manifest = fs::read(&manifest_path).expect("manifest");
    let offset = 8 + 4 + 8 + 3 * (4 + 8);
    manifest[offset..offset + 4].copy_from_slice(&version.to_le_bytes());
    manifest[offset + 4..offset + 12].copy_from_slice(&hash.to_le_bytes());
    fs::write(&manifest_path, &manifest).expect("old training fixture");
    assert_eq!(
        TrainingArtifact::load(&directory).expect_err("P19 training must reject"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(
        fs::read(manifest_path).expect("unchanged manifest"),
        manifest
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

fn current_runtime_metadata() -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([
        (
            "action_schema_hash".to_owned(),
            crate::ACTION_SCHEMA_HASH.to_string(),
        ),
        (
            "feature_schema_hash".to_owned(),
            crate::FEATURE_SCHEMA_HASH.to_string(),
        ),
        (
            "model_schema_hash".to_owned(),
            crate::MODEL_SCHEMA_HASH.to_string(),
        ),
        (
            "ppo_schema_version".to_owned(),
            crate::PPO_SCHEMA_VERSION.to_string(),
        ),
        (
            "ppo_schema_hash".to_owned(),
            crate::PPO_SCHEMA_HASH.to_string(),
        ),
        (
            "ppo_rules_audit_version".to_owned(),
            crate::PPO_RULES_AUDIT_VERSION.to_string(),
        ),
        (
            "map2_reward_schema_version".to_owned(),
            crate::MAP2_REWARD_SCHEMA_VERSION.to_string(),
        ),
        (
            "map2_reward_schema_hash".to_owned(),
            crate::MAP2_REWARD_SCHEMA_HASH.to_string(),
        ),
        (
            "map2_reward_schema_descriptor".to_owned(),
            crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.to_owned(),
        ),
    ])
}

fn prior_runtime_metadata() -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([
        (
            "action_schema_hash".to_owned(),
            "1018254919734743331".to_owned(),
        ),
        (
            "feature_schema_hash".to_owned(),
            "13875648161437731669".to_owned(),
        ),
        (
            "model_schema_hash".to_owned(),
            "10644717168650027237".to_owned(),
        ),
        ("ppo_schema_version".to_owned(), "13".to_owned()),
        (
            "ppo_schema_hash".to_owned(),
            "11103744726312279053".to_owned(),
        ),
        ("ppo_rules_audit_version".to_owned(), "12".to_owned()),
    ])
}

#[test]
fn action_v2_runtime_v13_v14_v15_rejects_before_mutation_without_rewriting_metadata() {
    let directory = test_directory("v13-runtime");
    let source = PolicyModel::fresh(18_100).expect("source");
    let parameters = source.export_parameters().expect("parameters");
    let data: Vec<_> = parameters
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let path = directory.join("drysua.weights.safetensors");
    let target = PolicyModel::fresh(18_101).expect("target");
    let before = target.policy_identity().expect("identity");
    let original = target.export_parameters().expect("original");
    for (version, hash, rules) in PRIOR_PPO_SCHEMAS {
        let mut metadata = prior_runtime_metadata();
        metadata.insert("ppo_schema_version".to_owned(), version.to_string());
        metadata.insert("ppo_schema_hash".to_owned(), hash.to_string());
        metadata.insert("ppo_rules_audit_version".to_owned(), rules.to_string());
        let view = TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("view");
        let bytes = serialize([("model.parameters", view)], Some(metadata)).expect("prior runtime");
        fs::write(&path, &bytes).expect("fixture");

        let error = TrainingArtifact::load_runtime_weights(&target, &directory)
            .expect_err("action-v2 runtime is incompatible");

        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(target.policy_identity().expect("identity"), before);
        assert_eq!(target.export_parameters().expect("parameters"), original);
        assert_eq!(fs::read(&path).expect("unchanged file"), bytes);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn runtime_rejects_every_wrong_current_tuple_field_before_mutation() {
    let directory = test_directory("wrong-prior-tuple");
    let model = PolicyModel::fresh(18_102).expect("model");
    let before = model.policy_identity().expect("identity");
    let parameters = model.export_parameters().expect("parameters");
    for (field, value) in [
        ("action_schema_hash", "1018254919734743331"),
        ("feature_schema_hash", "13875648161437731669"),
        ("feature_schema_hash", "10322490384647633864"),
        ("feature_schema_hash", "9669721049329356661"),
        ("model_schema_hash", "10644717168650027237"),
        ("model_schema_hash", "3097714014199697774"),
        ("model_schema_hash", "720439888929233033"),
        ("model_schema_hash", "832872366354465423"),
        ("ppo_schema_version", "12"),
        ("ppo_schema_version", "13"),
        ("ppo_schema_version", "14"),
        ("ppo_schema_version", "15"),
        ("ppo_schema_version", "16"),
        ("ppo_schema_version", "18"),
        ("ppo_schema_hash", "6877503070358232325"),
        ("ppo_schema_version", "17"),
        ("ppo_schema_hash", "13743352113669513864"),
        ("ppo_schema_hash", "11450737853127354910"),
        ("ppo_schema_hash", "11103744726312279052"),
        ("ppo_rules_audit_version", "11"),
        ("ppo_rules_audit_version", "12"),
        ("ppo_rules_audit_version", "13"),
        ("ppo_rules_audit_version", "14"),
        ("unexpected", "13"),
        ("action_schema_hash", ""),
        ("feature_schema_hash", ""),
        ("model_schema_hash", ""),
        ("ppo_schema_version", ""),
        ("ppo_schema_hash", ""),
        ("ppo_rules_audit_version", ""),
    ] {
        let mut metadata = current_runtime_metadata();
        if value.is_empty() {
            metadata.remove(field);
        } else {
            metadata.insert(field.to_owned(), value.to_owned());
        }
        let data = 0.0f32.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).expect("view");
        let bytes = serialize([("model.parameters", view)], Some(metadata)).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");

        let error = TrainingArtifact::load_runtime_weights(&model, &directory)
            .expect_err("wrong prior tuple");

        assert_eq!(error, CheckpointError::SchemaMismatch, "{field}={value}");
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(model.policy_identity().expect("identity"), before);
        assert_eq!(model.export_parameters().expect("parameters"), parameters);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn observation_bound_change_rejects_full_feature_v8_weights_without_metadata_rewrite() {
    let directory = test_directory("feature-v8-runtime");
    let model = PolicyModel::fresh(18_106).expect("model");
    let identity = model.policy_identity().expect("identity");
    let original = model.export_parameters().expect("original parameters");
    let mut metadata = current_runtime_metadata();
    metadata.insert(
        "feature_schema_hash".to_owned(),
        "10322490384647633864".to_owned(),
    );
    metadata.insert(
        "model_schema_hash".to_owned(),
        "3097714014199697774".to_owned(),
    );
    metadata.insert("ppo_schema_version".to_owned(), "16".to_owned());
    metadata.insert(
        "ppo_schema_hash".to_owned(),
        "11450737853127354910".to_owned(),
    );
    let data = vec![0u8; crate::MODEL_PARAMETER_COUNT * 4];
    let tensor = TensorView::new(Dtype::F32, vec![crate::MODEL_PARAMETER_COUNT], &data)
        .expect("full unchanged tensor shape");
    let bytes = serialize([("model.parameters", tensor)], Some(metadata)).expect("old runtime");
    let path = directory.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("fixture");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory)
        .expect_err("old full neural weights remain incompatible");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_eq!(model.export_parameters().expect("parameters"), original);
    assert_eq!(fs::read(&path).expect("unchanged fixture"), bytes);
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn prior_training_resume_rejects_v13_v14_v15_before_model_mutation() {
    let directory = test_directory("v13-training");
    let model = PolicyModel::fresh(18_103).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_104).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let path = directory.join("checkpoint.meta");
    let current = fs::read(&path).expect("manifest");
    let before = model.policy_identity().expect("identity");
    let parameters = model.export_parameters().expect("parameters");
    // The fixed header precedes three action/feature/model version-hash pairs.
    let ppo_offset = 8 + 4 + 8 + 3 * (4 + 8);
    for (version, hash) in [
        (13, 11_103_744_726_312_279_053),
        (14, 15_610_409_340_106_916_160),
        (15, 13_893_101_989_595_893_928),
        (16, 11_450_737_853_127_354_910),
        (18, 6_877_503_070_358_232_325),
        (15, crate::PPO_SCHEMA_HASH),
        (crate::PPO_SCHEMA_VERSION, 13_893_101_989_595_893_928),
        (14, crate::PPO_SCHEMA_HASH),
        (crate::PPO_SCHEMA_VERSION, 15_610_409_340_106_916_160),
        (13, crate::PPO_SCHEMA_HASH),
        (crate::PPO_SCHEMA_VERSION, 11_103_744_726_312_279_053),
    ] {
        let mut bytes = current.clone();
        bytes[ppo_offset..ppo_offset + 4].copy_from_slice(&version.to_le_bytes());
        bytes[ppo_offset + 4..ppo_offset + 12].copy_from_slice(&hash.to_le_bytes());
        fs::write(&path, bytes).expect("prior manifest fixture");

        for result in [
            TrainingArtifact::load(&directory),
            TrainingArtifact::load_compatible(&directory, &run_metadata()),
        ] {
            let Err(error) = result else {
                panic!("old training must not resume");
            };
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
        assert_eq!(model.policy_identity().expect("identity"), before);
        assert_eq!(model.export_parameters().expect("parameters"), parameters);
        assert_eq!(trainer.optimizer_step(), 0);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[cfg(feature = "builtin")]
#[test]
fn provenance_migration_rejects_v13_v14_v15_without_rewriting_training_manifest() {
    let directory = test_directory("legacy-provenance-migration");
    let model = PolicyModel::fresh(18_109).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_110).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let path = directory.join("checkpoint.meta");
    let current = fs::read(&path).expect("manifest");
    let settings = crate::TrainingJobConfig {
        episode_time_cost: 0.0,
        terminal_only: false,
        complete_episodes: false,
        updates: 1,
        ppo: crate::PpoConfig {
            gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
            environments: 2,
            rollout_decisions: 2,
            epochs: 1,
            minibatch: 4,
            ..crate::PpoConfig::default()
        },
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::MigrateGitCommit,
        seed: 18_000,
        map: MapId(2),
        git_commit: "different-commit".to_owned(),
        simulator_commit: "b575129".to_owned(),
    };
    let ppo_offset = 8 + 4 + 8 + 3 * (4 + 8);
    for (version, hash, _) in PRIOR_PPO_SCHEMAS {
        let mut bytes = current.clone();
        bytes[ppo_offset..ppo_offset + 4].copy_from_slice(&version.to_le_bytes());
        bytes[ppo_offset + 4..ppo_offset + 12].copy_from_slice(&hash.to_le_bytes());
        fs::write(&path, &bytes).expect("prior manifest fixture");

        let error = crate::run_training_job_on(
            settings.clone(),
            PolicyDevice::Cpu,
            &directory,
            true,
            |_| panic!("legacy migration must not train"),
        )
        .expect_err("old schema is not a provenance migration");

        assert_eq!(
            error.to_string(),
            "PPO model error: checkpoint schema does not match this build"
        );
        assert_eq!(fs::read(&path).expect("unchanged manifest"), bytes);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_rejects_prior_action_feature_and_model_bindings_independently() {
    let directory = test_directory("legacy-inference-bindings");
    let model = PolicyModel::fresh(18_111).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_112).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let path = directory.join("checkpoint.meta");
    let current = fs::read(&path).expect("manifest");
    for (index, version, hash) in [
        (0, 2u32, 1_018_254_919_734_743_331u64),
        (1, 7, 13_875_648_161_437_731_669),
        (2, 7, 10_644_717_168_650_027_237),
        (2, 9, 832_872_366_354_465_423),
        (1, 11, 8_078_516_161_541_333_175),
        (2, 12, 17_156_054_387_874_206_897),
        (3, 23, 765_990_392_710_687_046),
        (3, 24, 17_486_156_843_355_673_207),
        (3, 25, 12_302_688_747_093_836_273),
    ] {
        let mut bytes = current.clone();
        let offset = 8 + 4 + 8 + index * (4 + 8);
        bytes[offset..offset + 4].copy_from_slice(&version.to_le_bytes());
        bytes[offset + 4..offset + 12].copy_from_slice(&hash.to_le_bytes());
        fs::write(&path, &bytes).expect("prior inference binding");

        let Err(error) = TrainingArtifact::load(&directory) else {
            panic!("old inference contract must not resume");
        };

        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(fs::read(&path).expect("unchanged manifest"), bytes);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn current_runtime_metadata_does_not_bypass_tensor_validation() {
    let directory = test_directory("prior-tensor-contract");
    let model = PolicyModel::fresh(18_105).expect("model");
    let before = model.policy_identity().expect("identity");
    let parameters = model.export_parameters().expect("parameters");
    for (name, dtype, count, value, expected) in [
        (
            "unknown",
            Dtype::F32,
            1,
            0.0f32,
            CheckpointError::TensorContract("names"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            1,
            0.0,
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::I32,
            crate::MODEL_PARAMETER_COUNT,
            0.0,
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            crate::MODEL_PARAMETER_COUNT,
            f32::NAN,
            CheckpointError::NonFiniteTensor {
                name: "model.parameters",
                index: 0,
            },
        ),
    ] {
        let data = value.to_le_bytes().repeat(count);
        let view = TensorView::new(dtype, vec![count], &data).expect("view");
        let bytes = serialize([(name, view)], Some(current_runtime_metadata())).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");

        let error = TrainingArtifact::load_runtime_weights(&model, &directory)
            .expect_err("invalid prior tensor");

        assert_eq!(error, expected);
        let message = match expected {
            CheckpointError::TensorContract(field) => {
                format!("checkpoint tensor contract has invalid {field}")
            }
            CheckpointError::NonFiniteTensor { name, index } => {
                format!("checkpoint tensor {name} contains non-finite value at {index}")
            }
            _ => panic!("unexpected tensor validation fixture"),
        };
        assert_eq!(error.to_string(), message);
        assert_eq!(model.policy_identity().expect("identity"), before);
        assert_eq!(model.export_parameters().expect("parameters"), parameters);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_capture_rejects_prior_rules_audit() {
    let model = PolicyModel::fresh(18_106).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_107).expect("trainer");
    for (_, _, rules) in PRIOR_PPO_SCHEMAS {
        let mut run = run_metadata();
        run.rules_audit_version = rules;

        let error = TrainingArtifact::capture(&model, &trainer, run, progress_metadata(0))
            .expect_err("prior rules audit");

        assert_eq!(
            error,
            CheckpointError::InvalidManifest("rules audit or batch size")
        );
        assert_eq!(
            error.to_string(),
            "checkpoint manifest has invalid rules audit or batch size"
        );
    }
}

#[test]
#[ignore = "requires the accepted local BC anchor and preserved pre-migration probes"]
fn legacy_anchor_runtime_and_pre_migration_probe_training_reject_without_mutation() {
    let directory = std::path::Path::new("artifacts/temp/neural-v004-default-e8");
    let path = directory.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("accepted anchor");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    assert_eq!(
        metadata.metadata().as_ref(),
        Some(&prior_runtime_metadata())
    );
    let model = PolicyModel::fresh(18_108).expect("model");

    let before = model.policy_identity().expect("identity");
    let parameters = model.export_parameters().expect("parameters");

    let error = TrainingArtifact::load_runtime_weights(&model, directory)
        .expect_err("legacy anchor cannot run with action v3");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(model.policy_identity().expect("identity"), before);
    assert_eq!(model.export_parameters().expect("parameters"), parameters);
    for probe in ["neural-v004-conservative-u1", "neural-v004-conservative-u8"] {
        let path = std::path::Path::new("artifacts/temp").join(probe);
        let error = TrainingArtifact::load(&path).expect_err("old probe cannot resume");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(model.policy_identity().expect("identity"), before);
    }
    assert_eq!(fs::read(path).expect("unchanged anchor"), bytes);
}

#[test]
fn checkpoint_schema_hash_is_stable() {
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 5);
    assert_eq!(
        crate::CHECKPOINT_SCHEMA_HASH,
        super::map2_checkpoint::schema_hash(
            crate::CHECKPOINT_SCHEMA_DESCRIPTOR,
            &[
                (crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH),
                (crate::FEATURE_SCHEMA_VERSION, crate::FEATURE_SCHEMA_HASH),
                (crate::MODEL_SCHEMA_VERSION, crate::MODEL_SCHEMA_HASH),
                (crate::PPO_SCHEMA_VERSION, crate::PPO_SCHEMA_HASH),
                (
                    crate::MAP2_REWARD_SCHEMA_VERSION,
                    crate::MAP2_REWARD_SCHEMA_HASH
                ),
            ],
        )
    );
    assert_ne!(crate::CHECKPOINT_SCHEMA_HASH, 4_581_258_024_746_721_724);
}

#[test]
fn strict_training_artifact_roundtrips_model_optimizer_and_run_state() {
    let directory = test_directory("roundtrip");
    let source = PolicyModel::fresh(18_001).expect("source model");
    let mut trainer = PpoTrainer::new(&source, checkpoint_config(), 18_002).expect("trainer");
    advance_trainer(&source, &mut trainer);
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(1))
            .expect("capture");

    artifact.save(&directory).expect("atomic save");
    let loaded =
        TrainingArtifact::load_compatible(&directory, &run_metadata()).expect("strict load");
    let restored = PolicyModel::fresh(18_003).expect("restored model");
    let mut restored_state = loaded
        .restore(&restored, &run_metadata())
        .expect("restore state");
    let restored_trainer = restored_state.trainer();

    assert_eq!(loaded.run(), &run_metadata());
    assert_eq!(loaded.progress(), &progress_metadata(1));
    assert_eq!(restored.device(), PolicyDevice::Cpu);
    assert_eq!(
        restored.export_parameters().expect("restored parameters"),
        source.export_parameters().expect("source parameters")
    );
    assert_eq!(restored_trainer.config(), trainer.config());
    assert_eq!(restored_trainer.optimizer_step(), trainer.optimizer_step());
    assert_eq!(restored_trainer.updates(), trainer.updates());
    assert_eq!(restored_trainer.rng_checkpoint(), trainer.rng_checkpoint());
    assert_eq!(
        restored_state
            .pipeline(4, 1, &restored)
            .expect("restored pipeline")
            .version()
            .expect("pipeline version")
            .get(),
        1
    );
    let (source_batch, source_actions) = checkpoint_batch(&source, 18_005);
    let (restored_batch, restored_actions) = checkpoint_batch(&restored, 18_005);
    trainer
        .train_update(&source, &source_batch)
        .expect("continued source update");
    restored_state
        .trainer_mut()
        .train_update(&restored, &restored_batch)
        .expect("continued restored update");
    assert_eq!(source_actions, restored_actions);
    assert_eq!(
        source.export_parameters().expect("continued source"),
        restored.export_parameters().expect("continued restored")
    );
    assert_eq!(
        trainer.rng_checkpoint(),
        restored_state.trainer().rng_checkpoint()
    );
    assert_eq!(
        trainer.optimizer_step(),
        restored_state.trainer().optimizer_step()
    );
    assert!(!directory.join("checkpoint.safetensors.tmp").exists());
    assert!(!directory.join("checkpoint.meta.tmp").exists());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn tensor_corruption_is_rejected_by_sha256_before_deserialization() {
    let directory = test_directory("tensor-corruption");
    let model = PolicyModel::fresh(18_010).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_011).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let tensor_path = fs::read_dir(&directory)
        .expect("artifact directory")
        .map(|entry| entry.expect("entry").path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("checkpoint.")
                        && name.ends_with(".safetensors")
                        && name != "checkpoint.safetensors"
                })
        })
        .expect("immutable tensor generation");
    let mut bytes = fs::read(&tensor_path).expect("tensor bytes");
    let index = bytes.len() - 1;
    bytes[index] ^= 0x01;
    fs::write(&tensor_path, bytes).expect("corrupt tensor");

    let error = TrainingArtifact::load(&directory).expect_err("hash mismatch");

    assert_eq!(error, CheckpointError::TensorHashMismatch);
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn truncated_manifest_is_rejected_with_exact_error() {
    let directory = test_directory("manifest-truncated");
    let model = PolicyModel::fresh(18_020).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_021).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    fs::write(directory.join("checkpoint.meta"), b"DRYSUA").expect("truncate manifest");

    let error = TrainingArtifact::load(&directory).expect_err("truncated manifest");

    assert_eq!(error.to_string(), "checkpoint manifest is truncated");
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn current_runtime_weights_roundtrip_with_current_metadata() {
    let directory = test_directory("runtime");
    let model = PolicyModel::fresh(18_030).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("runtime save");
    let bytes = fs::read(directory.join("drysua.weights.safetensors")).expect("runtime bytes");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    let expected = current_runtime_metadata();
    assert_eq!(metadata.metadata().as_ref(), Some(&expected));
    let restored = PolicyModel::fresh(18_031).expect("restored");

    TrainingArtifact::load_runtime_weights(&restored, &directory).expect("runtime load");

    assert_eq!(
        restored.export_parameters().expect("restored parameters"),
        model.export_parameters().expect("source parameters")
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn runtime_weights_reject_unknown_tensor_name_before_model_mutation() {
    let directory = test_directory("runtime-name");
    let model = PolicyModel::fresh(18_040).expect("model");
    let before = model.export_parameters().expect("before");
    let data = 0.0f32.to_le_bytes();
    let view = TensorView::new(Dtype::F32, vec![1], &data).expect("view");
    let metadata = current_runtime_metadata();
    let bytes = serialize([("unknown", view)], Some(metadata)).expect("malformed runtime file");
    fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write malformed");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory)
        .expect_err("unknown tensor name");

    assert_eq!(error, CheckpointError::TensorContract("names"));
    assert_eq!(
        error.to_string(),
        "checkpoint tensor contract has invalid names"
    );
    assert_eq!(model.export_parameters().expect("after"), before);
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn runtime_weights_reject_metadata_without_the_deployment_schema() {
    let directory = test_directory("runtime-deployment-schema");
    let model = PolicyModel::fresh(18_045).expect("model");
    let before = model.export_parameters().expect("before");
    let data = 0.0f32.to_le_bytes();
    let view = TensorView::new(Dtype::F32, vec![1], &data).expect("view");
    let metadata = std::collections::HashMap::from([
        (
            "action_schema_hash".to_owned(),
            crate::ACTION_SCHEMA_HASH.to_string(),
        ),
        (
            "feature_schema_hash".to_owned(),
            crate::FEATURE_SCHEMA_HASH.to_string(),
        ),
        (
            "model_schema_hash".to_owned(),
            crate::MODEL_SCHEMA_HASH.to_string(),
        ),
    ]);
    let bytes =
        serialize([("model.parameters", view)], Some(metadata)).expect("old runtime metadata");
    fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write malformed");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory)
        .expect_err("deployment schema is mandatory");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(model.export_parameters().expect("after"), before);
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn restore_rejects_owned_optimizer_without_parameter_mutation() {
    let source = PolicyModel::fresh(18_050).expect("source");
    let source_trainer =
        PpoTrainer::new(&source, checkpoint_config(), 18_051).expect("source trainer");
    let artifact = TrainingArtifact::capture(
        &source,
        &source_trainer,
        run_metadata(),
        progress_metadata(0),
    )
    .expect("artifact");
    let target = PolicyModel::fresh(18_052).expect("target");
    let _owner = PpoTrainer::new(&target, checkpoint_config(), 18_053).expect("existing owner");
    let before = target.export_parameters().expect("before");

    let Err(error) = artifact.restore(&target, &run_metadata()) else {
        panic!("owned optimizer was replaced");
    };

    assert_eq!(
        error.to_string(),
        "checkpoint model restore failed: model already has a behavioral optimizer owner"
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn restore_rejects_immutable_actor_model_without_parameter_mutation() {
    let source = PolicyModel::fresh(18_060).expect("source");
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18_061).expect("trainer");
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
            .expect("artifact");
    let learner = PolicyModel::fresh(18_062).expect("learner");
    let mut pipeline = crate::ActorLearnerPipeline::new(4, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let before = lease.policy().export_parameters().expect("before");

    let Err(error) = artifact.restore(lease.policy(), &run_metadata()) else {
        panic!("actor model accepted a training checkpoint");
    };

    assert_eq!(
        error.to_string(),
        "checkpoint model restore failed: immutable actor model cannot own an optimizer"
    );
    assert_eq!(lease.policy().export_parameters().expect("after"), before);
}

#[test]
fn restore_rejects_different_compatibility_scope_before_mutation() {
    let source = PolicyModel::fresh(18_070).expect("source");
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18_071).expect("trainer");
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
            .expect("artifact");
    let target = PolicyModel::fresh(18_072).expect("target");
    let before = target.export_parameters().expect("before");
    let mut expected = run_metadata();
    expected.simulator_commit = "different".to_owned();

    let Err(error) = artifact.restore(&target, &expected) else {
        panic!("incompatible simulator commit was accepted");
    };

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("compatibility scope")
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn manifest_backup_recovers_checkpoint_after_interrupted_replacement() {
    let directory = test_directory("manifest-recovery");
    let model = PolicyModel::fresh(18_080).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_081).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("artifact")
        .save(&directory)
        .expect("save");
    fs::rename(
        directory.join("checkpoint.meta"),
        directory.join("checkpoint.meta.previous"),
    )
    .expect("simulate crash after backup rename");

    let recovered = TrainingArtifact::load(&directory).expect("recover previous manifest");

    assert_eq!(recovered.run(), &run_metadata());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn documented_two_file_artifact_loads_without_immutable_generation_copy() {
    let directory = test_directory("two-file-copy");
    let model = PolicyModel::fresh(18_085).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_086).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("artifact")
        .save(&directory)
        .expect("save");
    for entry in fs::read_dir(&directory).expect("directory") {
        let path = entry.expect("entry").path();
        let is_generation = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("checkpoint.")
                    && name.ends_with(".safetensors")
                    && name != "checkpoint.safetensors"
            });
        if is_generation {
            fs::remove_file(path).expect("remove generation");
        }
    }

    let loaded = TrainingArtifact::load(&directory).expect("canonical tensor fallback");

    assert_eq!(loaded.run(), &run_metadata());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[cfg(unix)]
#[test]
fn runtime_loader_rejects_symlink_artifact() {
    use std::os::unix::fs::symlink;

    let directory = test_directory("runtime-symlink");
    let model = PolicyModel::fresh(18_090).expect("model");
    let target = directory.join("outside.safetensors");
    fs::write(&target, b"outside").expect("outside file");
    symlink(&target, directory.join("drysua.weights.safetensors")).expect("artifact symlink");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("symlink artifact");

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("artifact file type")
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}

fn checkpoint_config() -> PpoConfig {
    PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        rollout_decisions: 2,
        environments: 2,
        epochs: 1,
        minibatch: 4,
        ..PpoConfig::default()
    }
}

fn run_metadata() -> CheckpointRun {
    CheckpointRun {
        git_commit: "28e196f".to_owned(),
        simulator_commit: "b575129".to_owned(),
        enabled_features: crate::compiled_features(),
        command_line: "drysua train --device cpu".to_owned(),
        run_seed: 18_000,
        map: MapId(2),
        hero: crate::SHADOW_FIEND,
        device: CheckpointDevice::Cpu,
        batch_size: 4,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn progress_metadata(global_update: u64) -> CheckpointProgress {
    CheckpointProgress {
        global_update,
        policy_version: global_update,
        scheduler_step: 3,
        curriculum_stage: 2,
        rollout_samples: 1_024,
        best_evaluation: Some(0.75),
        rng_states: vec![
            RngCheckpoint::new("actor", 11, 12).expect("actor RNG"),
            RngCheckpoint::new("learner", 13, 14).expect("learner RNG"),
        ],
        league_references: vec![101, 102],
    }
}

fn advance_trainer(model: &PolicyModel, trainer: &mut PpoTrainer) {
    let (batch, _) = checkpoint_batch(model, 18_004);
    trainer.train_update(model, &batch).expect("update");
}

fn checkpoint_batch(
    model: &PolicyModel,
    seed: u64,
) -> (crate::PpoBatch, Vec<crate::StructuredAction>) {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let mut rng = PpoRng::new(seed);
    let policy = model.policy_identity().expect("policy");
    let mut rollout = PpoRollout::new(4, policy).expect("rollout");
    let mut actions = Vec::with_capacity(4);
    for stream in 0..4 {
        let choice = model.sample(&frame, &space, &mut rng).expect("choice");
        actions.push(choice.action());
        rollout
            .push(
                choice
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: stream as f32,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    let batch = rollout.finish(checkpoint_config()).expect("batch");
    (batch, actions)
}

fn test_directory(name: &str) -> PathBuf {
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-checkpoint-{name}-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("remove stale test directory");
    }
    fs::create_dir(&directory).expect("create test directory");
    directory
}
