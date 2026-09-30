use super::*;

#[test]
fn training_contract_current_runtime_validates_tensor_names_shapes_and_dtype() {
    assert_invalid_runtime_contract(current_runtime_metadata(), crate::MODEL_PARAMETER_COUNT);
}

fn assert_invalid_runtime_contract(
    metadata: std::collections::HashMap<String, String>,
    count: usize,
) {
    let directory = test_directory("training-contract-runtime-tensor");
    let model = PolicyModel::fresh(10_093_103).expect("model");
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
        let bytes = serialize([(name, tensor)], Some(metadata.clone())).expect("fixture");
        let path = directory.join("drysua.weights.safetensors");
        fs::write(&path, &bytes).expect("fixture");
        let error =
            TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("invalid tensor");
        let CheckpointError::TensorContract(field) = expected else {
            panic!("tensor fixture")
        };
        assert_eq!(error, CheckpointError::TensorContract(field));
        assert_eq!(
            error.to_string(),
            format!("checkpoint tensor contract has invalid {field}")
        );
        assert_eq!(
            model.policy_identity().expect("unchanged identity"),
            identity
        );
        assert_eq!(fs::read(path).expect("unchanged source"), bytes);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn training_contract_current_runtime_rejects_nonfinite_tensor_boundaries() {
    assert_nonfinite_runtime_contract(current_runtime_metadata(), crate::MODEL_PARAMETER_COUNT);
}

fn assert_nonfinite_runtime_contract(
    metadata: std::collections::HashMap<String, String>,
    count: usize,
) {
    let directory = test_directory("training-contract-runtime-nonfinite");
    let model = PolicyModel::fresh(10_093_103).expect("model");
    let mut data = vec![0; count * 4];
    let identity = model.policy_identity().expect("identity");
    for index in [0, count - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            data[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            let tensor = TensorView::new(Dtype::F32, vec![count], &data).expect("tensor");
            let bytes =
                serialize([("model.parameters", tensor)], Some(metadata.clone())).expect("fixture");
            let path = directory.join("drysua.weights.safetensors");
            fs::write(&path, &bytes).expect("fixture");
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
            assert_eq!(fs::read(path).expect("unchanged source"), bytes);
        }
        data[index * 4..index * 4 + 4].fill(0);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}
