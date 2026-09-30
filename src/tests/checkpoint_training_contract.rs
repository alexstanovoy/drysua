use super::*;

/// Current-layout runtime tensors with the first one replaced by `(name, dtype, shape)`.
fn runtime_with_first(
    metadata: &std::collections::HashMap<String, String>,
    first: (&str, Dtype, Vec<usize>),
    data: &[u8],
) -> Vec<u8> {
    let schema = PolicyModel::fresh(0)
        .and_then(|model| model.parameter_schema())
        .expect("schema");
    let mut offset = 0;
    let mut tensors = Vec::with_capacity(schema.len());
    for (index, (name, shape)) in schema.into_iter().enumerate() {
        let (name, dtype, shape) = if index == 0 {
            first.clone()
        } else {
            (name, Dtype::F32, shape)
        };
        let size = shape.iter().product::<usize>() * 4;
        let tensor = TensorView::new(dtype, shape, &data[offset..offset + size]).expect("tensor");
        tensors.push((name.to_owned(), tensor));
        offset += size;
    }
    serialize(tensors, Some(metadata.clone())).expect("fixture")
}

#[test]
fn training_contract_current_runtime_validates_tensor_names_shapes_and_dtype() {
    let directory = test_directory("training-contract-runtime-tensor");
    let model = PolicyModel::fresh(10_093_103).expect("model");
    let data = vec![0; crate::MODEL_PARAMETER_COUNT * 4];
    let identity = model.policy_identity().expect("identity");
    let (first, shape) = model.parameter_schema().expect("schema")[0].clone();
    let count = shape.iter().product::<usize>();
    for (tensor, expected) in [
        (("wrong", Dtype::F32, shape.clone()), "names"),
        ((first, Dtype::I32, shape.clone()), "dtype or shape"),
        ((first, Dtype::F32, vec![1, count]), "dtype or shape"),
        ((first, Dtype::F32, vec![count - 1]), "dtype or shape"),
    ] {
        let bytes = runtime_with_first(&current_runtime_metadata(), tensor, &data);
        let path = directory.join("drysua.weights.safetensors");
        fs::write(&path, &bytes).expect("fixture");
        let error =
            TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("invalid tensor");
        assert_eq!(error, CheckpointError::TensorContract(expected));
        assert_eq!(
            error.to_string(),
            format!("checkpoint tensor contract has invalid {expected}")
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
    let directory = test_directory("training-contract-runtime-nonfinite");
    let model = PolicyModel::fresh(10_093_103).expect("model");
    let schema = model.parameter_schema().expect("schema");
    let (first, _) = schema[0].clone();
    let (last, last_shape) = schema[schema.len() - 1].clone();
    let count = crate::MODEL_PARAMETER_COUNT;
    let mut data = vec![0; count * 4];
    let identity = model.policy_identity().expect("identity");
    let first_shape = schema[0].1.clone();
    for (index, name, tensor_index) in [
        (0, first, 0),
        (count - 1, last, last_shape.iter().product::<usize>() - 1),
    ] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            data[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            let bytes = runtime_with_first(
                &current_runtime_metadata(),
                (first, Dtype::F32, first_shape.clone()),
                &data,
            );
            let path = directory.join("drysua.weights.safetensors");
            fs::write(&path, &bytes).expect("fixture");
            let error = TrainingArtifact::load_runtime_weights(&model, &directory)
                .expect_err("nonfinite tensor");
            assert_eq!(
                error,
                CheckpointError::NonFiniteTensor {
                    name,
                    index: tensor_index
                }
            );
            assert_eq!(
                error.to_string(),
                format!("checkpoint tensor {name} contains non-finite value at {tensor_index}")
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
