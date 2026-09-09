use super::*;

fn test_directory(name: &str) -> PathBuf {
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!(
            "input-facts-{name}-{}-{sequence}",
            std::process::id()
        ));
    assert!(!path.exists());
    fs::create_dir(&path).expect("test artifact directory");
    path
}

fn m11_metadata() -> std::collections::HashMap<String, String> {
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "15519817897416174399"),
        ("model_schema_hash", "18229126264156367519"),
        ("ppo_schema_version", "21"),
        ("ppo_schema_hash", "768751058595344501"),
        ("ppo_rules_audit_version", "17"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

#[test]
fn m11_training_initializer_rejects_missing_or_wrong_metadata() {
    let directory = test_directory("m11-input-metadata");
    let data = vec![0u8; M11_PARAMETERS * 4];
    for key in m11_metadata().keys() {
        for replacement in [None, Some("0")] {
            let mut metadata = m11_metadata();
            metadata.remove(key);
            if let Some(value) = replacement {
                metadata.insert(key.clone(), value.to_owned());
            }
            let tensor = TensorView::new(Dtype::F32, vec![M11_PARAMETERS], &data).expect("tensor");
            let bytes = serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture");
            fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");
            let error = TrainingArtifact::initialize_selected_m11_for_training(
                &directory,
                1,
                PolicyDevice::Cpu,
            )
            .err()
            .expect("strict source tuple");
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn m11_input_size_rejects_under_current_runtime_metadata_without_mutation() {
    let directory = test_directory("m11-size-current-tuple");
    let data = vec![0u8; M11_PARAMETERS * 4];
    let tensor = TensorView::new(Dtype::F32, vec![M11_PARAMETERS], &data).expect("tensor");
    let bytes = serialize(
        [("model.parameters", tensor)],
        Some(current_runtime_metadata()),
    )
    .expect("fixture");
    fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");
    let model = PolicyModel::fresh(62).expect("model");
    let identity = model.policy_identity().expect("identity");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &directory)
            .expect_err("cannot disguise M11 size with new metadata"),
        CheckpointError::TensorContract("dtype or shape")
    );
    assert_eq!(model.policy_identity().expect("unchanged"), identity);
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn m11_training_initializer_rejects_wrong_tensor_shape_nonfinite_values_and_unapproved_sha() {
    let directory = test_directory("m11-input-tensor");
    for (name, count, value, expected) in [
        (
            "wrong",
            M11_PARAMETERS,
            0.0,
            CheckpointError::TensorContract("names"),
        ),
        (
            "model.parameters",
            1,
            0.0,
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            M11_PARAMETERS,
            f32::NAN,
            CheckpointError::NonFiniteTensor {
                name: "model.parameters",
                index: 0,
            },
        ),
        (
            "model.parameters",
            M11_PARAMETERS,
            0.0,
            CheckpointError::TensorContract("selected M11 training source SHA-256"),
        ),
    ] {
        let mut data = vec![0u8; count * 4];
        data[..4].copy_from_slice(&value.to_le_bytes());
        let tensor = TensorView::new(Dtype::F32, vec![count], &data).expect("tensor");
        let bytes = serialize([(name, tensor)], Some(m11_metadata())).expect("fixture");
        fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write");
        let error = TrainingArtifact::initialize_selected_m11_for_training(
            &directory,
            1,
            PolicyDevice::Cpu,
        )
        .err()
        .expect("approved finite tensor only");
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(error, expected);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
#[ignore = "requires immutable local approved u10 artifact and new DRYSUA_M11_INITIALIZATION_OUTPUT"]
fn approved_m11_initializes_zero_new_inputs_fresh_optimizer_and_runtime_rejects_old_source() {
    use sha2::{Digest, Sha256};
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/map0-ppo21-u010-u040-001/baseline-u010");
    let path = directory.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("approved artifact");
    let runtime = PolicyModel::fresh(63).expect("runtime");
    let before = runtime.export_parameters().expect("before");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&runtime, &directory)
            .expect_err("old runtime must reject"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(runtime.export_parameters().expect("after"), before);
    let (model, digest) =
        TrainingArtifact::initialize_selected_m11_for_training(&directory, 64, PolicyDevice::Cpu)
            .expect("explicit initialization");
    assert_eq!(digest, <[u8; 32]>::from(Sha256::digest(&bytes)));
    assert_eq!(fs::read(&path).expect("source untouched"), bytes);
    save_initialized_artifact(&model);
    let optimizer = model
        .claim_optimizer(crate::AdamConfig::default())
        .expect("fresh optimizer");
    assert_eq!(optimizer.step(), 0);
    assert!(optimizer.moments().0.iter().all(|value| *value == 0.0));
    assert!(optimizer.moments().1.iter().all(|value| *value == 0.0));
}

fn save_initialized_artifact(model: &PolicyModel) {
    use sha2::{Digest, Sha256};
    let output = PathBuf::from(
        std::env::var_os("DRYSUA_M11_INITIALIZATION_OUTPUT").expect("explicit new output"),
    );
    assert!(
        !output.exists(),
        "historical M12 output must never be reused"
    );
    fs::create_dir(&output).expect("new initialized output");
    TrainingArtifact::save_runtime_weights(model, &output).expect("new current-contract artifact");
    let loaded = PolicyModel::fresh(66).expect("roundtrip model");
    TrainingArtifact::load_runtime_weights(&loaded, &output)
        .expect("current runtime accepts initialization");
    assert_eq!(
        loaded.export_parameters().expect("loaded"),
        model.export_parameters().expect("initialized")
    );
    assert_eq!(loaded.parameter_count(), crate::MODEL_PARAMETER_COUNT);
    let bytes = fs::read(output.join("drysua.weights.safetensors")).expect("new weights");
    let digest: String = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    eprintln!(
        "initialized_model_version={} output={} sha256={digest}",
        crate::MODEL_SCHEMA_VERSION,
        output.display()
    );
}
