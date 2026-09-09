use super::*;

fn ppo26_runtime_metadata() -> std::collections::HashMap<String, String> {
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "1577122233561586211"),
        ("model_schema_hash", "7970187849195607202"),
        ("ppo_schema_version", "26"),
        ("ppo_schema_hash", "4420330489262074980"),
        ("ppo_rules_audit_version", "21"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

#[test]
fn training_contract_versions_change_training_scope_not_actor_schema() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 12);
    assert_eq!(crate::FEATURE_SCHEMA_HASH, 1_577_122_233_561_586_211);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 14);
    assert_eq!(crate::MODEL_SCHEMA_HASH, 7_970_187_849_195_607_202);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 13);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 27);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 22);
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 27);
    assert_eq!(crate::LEAGUE_RULES_AUDIT_VERSION, 22);
    assert!(crate::PPO_SCHEMA_DESCRIPTOR.contains("observer=explicit_from_trajectory_start"));
    assert!(crate::LEAGUE_SCHEMA_DESCRIPTOR.contains("frozen_weights_not_legacy_execution"));
    assert_eq!(
        crate::TrainingScope::new(MapId(0), 12)
            .expect_err("old imitation scope")
            .to_string(),
        "imitation checkpoint has invalid rules audit version"
    );
}

#[test]
fn training_contract_ppo26_checkpoint_rejects_even_with_identical_actor_layout() {
    let directory = test_directory("training-contract-ppo26-resume");
    let model = PolicyModel::fresh(10_093_100).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 1).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("fixture");
    let path = directory.join("checkpoint.meta");
    let mut bytes = fs::read(&path).expect("manifest");
    let offset = 8 + 4 + 8 + 3 * (4 + 8);
    bytes[offset..offset + 4].copy_from_slice(&26u32.to_le_bytes());
    bytes[offset + 4..offset + 12].copy_from_slice(&4_420_330_489_262_074_980u64.to_le_bytes());
    fs::write(&path, &bytes).expect("PPO26 binding");
    for attempt in [
        TrainingArtifact::load(&directory),
        TrainingArtifact::load_compatible(&directory, &run_metadata()),
    ] {
        let Err(error) = attempt else {
            panic!("PPO26 training must not resume")
        };
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
    }
    assert_eq!(fs::read(path).expect("unchanged"), bytes);
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_exact_ppo26_runtime_preserves_parameter_bits_and_predictions() {
    let directory = test_directory("training-contract-ppo26-runtime");
    let source = PolicyModel::fresh(10_093_100).expect("source");
    let mut parameters = source.export_parameters().expect("parameters");
    parameters[0] = -0.0;
    source
        .import_parameters(&parameters)
        .expect("signed-zero boundary");
    let data: Vec<_> = parameters
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("tensor");
    let bytes = serialize(
        [("model.parameters", tensor)],
        Some(ppo26_runtime_metadata()),
    )
    .expect("fixture");
    let path = directory.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("fixture");
    let target = PolicyModel::fresh(10_093_101).expect("target");
    TrainingArtifact::load_runtime_weights(&target, &directory)
        .expect("audited identical actor contract");
    let actual: Vec<_> = target
        .export_parameters()
        .expect("parameters")
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(actual, data);
    assert_eq!(
        source.parameter_schema().expect("source layout"),
        target.parameter_schema().expect("target layout")
    );
    let frame = crate::FeatureFrame::new();
    assert_eq!(
        source.evaluate(&frame).expect("source heads"),
        target.evaluate(&frame).expect("target heads")
    );
    assert_eq!(fs::read(path).expect("unchanged source"), bytes);
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_ppo26_runtime_rejects_each_missing_wrong_or_extra_key_atomically() {
    let directory = test_directory("training-contract-ppo26-metadata");
    let model = PolicyModel::fresh(10_093_102).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 1).expect("bound optimizer");
    let identity = model.policy_identity().expect("identity");
    let original = model.export_parameters().expect("original");
    let expected = ppo26_runtime_metadata();
    let mut invalid = vec![None];
    for key in expected.keys() {
        for replacement in [None, Some("999")] {
            let mut metadata = expected.clone();
            metadata.remove(key);
            if let Some(value) = replacement {
                metadata.insert(key.clone(), value.to_owned());
            }
            invalid.push(Some(metadata));
        }
    }
    let mut extra = expected;
    extra.insert("ignored_role".to_owned(), "legacy".to_owned());
    invalid.push(Some(extra));
    for metadata in invalid {
        let tensor = TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("tensor");
        let bytes = serialize([("model.parameters", tensor)], metadata).expect("fixture");
        let path = directory.join("drysua.weights.safetensors");
        fs::write(&path, &bytes).expect("fixture");
        let error = TrainingArtifact::load_runtime_weights(&model, &directory)
            .expect_err("exact tuple only");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(model.policy_identity().expect("identity"), identity);
        assert_eq!(model.export_parameters().expect("parameters"), original);
        assert!(
            TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
                .is_ok()
        );
        assert_eq!(fs::read(path).expect("unchanged"), bytes);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_ppo26_runtime_still_validates_tensor_names_shapes_and_dtype() {
    let directory = test_directory("training-contract-ppo26-tensor");
    let model = PolicyModel::fresh(10_093_103).expect("model");
    let count = crate::MODEL_PARAMETER_COUNT;
    let data = vec![0; count * 4];
    let identity = model.policy_identity().expect("identity");
    for (name, dtype, shape, expected) in [
        (
            "wrong",
            Dtype::F32,
            vec![count],
            CheckpointError::TensorContract("names"),
        ),
        (
            "model.parameters",
            Dtype::I32,
            vec![count],
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![1, count],
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![count - 1],
            CheckpointError::TensorContract("dtype or shape"),
        ),
    ] {
        let size = shape.iter().product::<usize>() * 4;
        let tensor = TensorView::new(dtype, shape, &data[..size]).expect("tensor");
        let bytes = serialize([(name, tensor)], Some(ppo26_runtime_metadata())).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("fixture");
        let error =
            TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("invalid tensor");
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(error, expected);
        assert_eq!(
            model.policy_identity().expect("unchanged identity"),
            identity
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_ppo26_runtime_rejects_nonfinite_tensor_boundaries() {
    let directory = test_directory("training-contract-ppo26-nonfinite");
    let model = PolicyModel::fresh(10_093_103).expect("model");
    let count = crate::MODEL_PARAMETER_COUNT;
    let mut data = vec![0; count * 4];
    let identity = model.policy_identity().expect("identity");
    for index in [0, count - 1] {
        data[index * 4..index * 4 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let tensor = TensorView::new(Dtype::F32, vec![count], &data).expect("tensor");
        let bytes = serialize(
            [("model.parameters", tensor)],
            Some(ppo26_runtime_metadata()),
        )
        .expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("fixture");
        let error = TrainingArtifact::load_runtime_weights(&model, &directory)
            .expect_err("nonfinite tensor");
        assert_eq!(
            error,
            CheckpointError::NonFiniteTensor {
                name: "model.parameters",
                index
            }
        );
        assert_eq!(
            error.to_string(),
            format!("checkpoint tensor model.parameters contains non-finite value at {index}")
        );
        assert_eq!(
            model.policy_identity().expect("unchanged identity"),
            identity
        );
        data[index * 4..index * 4 + 4].fill(0);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_runtime_rejects_partial_mix_of_ppo26_and_current_training_metadata() {
    let directory = test_directory("training-contract-mixed-metadata");
    let model = PolicyModel::fresh(10_093_104).expect("model");
    let current = current_runtime_metadata();
    for key in [
        "ppo_schema_version",
        "ppo_schema_hash",
        "ppo_rules_audit_version",
    ] {
        let mut metadata = ppo26_runtime_metadata();
        metadata.insert(key.to_owned(), current[key].clone());
        let tensor = TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("tensor");
        let bytes = serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("fixture");
        let error =
            TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("mixed tuple");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
#[ignore = "read-only local M14/PPO26 artifact audit; no arena, optimizer update, output artifacts or gameplay"]
fn training_contract_local_m14_initializations_are_runtime_only_compatible_without_rewrite() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (label, expected_digest) in [
        (
            "initial",
            "ce17f6f1e3668837776b366fdcab59c250a4d6bb6de3ba3615dc3e7f23a53c4b",
        ),
        (
            "skill-alpha050",
            "22a21a13472e5f7fce308b6ddb72a03132ed711866f20d0c86ed32ce6603ded2",
        ),
    ] {
        let directory = root
            .join("artifacts/temp/neural-reset-20260908/order-contract-m14")
            .join(label);
        audit_ppo26_initialization(&directory, expected_digest);
    }
    for (source, expected) in [
        (
            "artifacts/temp/input-facts-m12-u10-init",
            "adbecb8293548ad1602b24b046b9a6f032fc62798d46102029a1f3448d764bdd",
        ),
        (
            "artifacts/temp/neural-reset-20260908/finish-skill-alpha-050",
            "d22a011f829c23593bbc6c03d22cbab9ab13bb79662959eda2d6c96b9de8098e",
        ),
    ] {
        let model = PolicyModel::fresh(10_093_106).expect("owned model");
        let error = TrainingArtifact::load_runtime_weights(&model, &root.join(source))
            .expect_err("M12 remains incompatible");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        audit_selected_m12_initializer(&root.join(source), expected);
    }
}

fn audit_ppo26_initialization(directory: &std::path::Path, expected_digest: &str) {
    use sha2::{Digest, Sha256};
    let path = directory.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("immutable initialization");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, expected_digest);
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    assert_eq!(
        metadata.metadata().as_ref(),
        Some(&ppo26_runtime_metadata())
    );
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("tensors");
    let payload = tensors.tensor("model.parameters").expect("parameters");
    let model = PolicyModel::fresh(10_093_105).expect("owned model");
    TrainingArtifact::load_runtime_weights(&model, directory).expect("unchanged actor contract");
    PolicyModel::validate_m12_parameter_schema(&model.parameter_schema().expect("62-name layout"))
        .expect("unchanged audited layout");
    let actual: Vec<_> = model
        .export_parameters()
        .expect("parameters")
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(actual, payload.data());
    let Err(error) = TrainingArtifact::load(directory) else {
        panic!("PPO26 cannot resume")
    };
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 1).expect("fresh optimizer");
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.updates(), 0);
    assert_eq!(fs::read(path).expect("unchanged source"), bytes);
    eprintln!(
        "runtime_only_compatible sha256={digest} parameter_bits_exact=true optimizer_step=0 old_resume=rejected output_artifacts=none"
    );
}

fn audit_selected_m12_initializer(source: &std::path::Path, expected: &str) {
    let path = source.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("immutable M12 source");
    let (model, digest) = TrainingArtifact::initialize_selected_m12_for_training(
        source,
        10_093_106,
        PolicyDevice::Cpu,
    )
    .expect("unchanged two-source initialization whitelist");
    let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(digest, expected);
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("M12 tensors");
    let actual: Vec<_> = model
        .export_parameters()
        .expect("initialized parameters")
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(
        actual,
        tensors
            .tensor("model.parameters")
            .expect("M12 payload")
            .data()
    );
    assert_eq!(fs::read(path).expect("unchanged M12 source"), bytes);
    eprintln!(
        "selected_m12_initialization sha256={digest} payload_exact=true source_unchanged=true output_artifacts=none"
    );
}
