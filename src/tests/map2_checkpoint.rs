//! Shared filesystem and runtime fixtures; behavior lives in the public checkpoint tests.

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView, serialize};

/// Runtime weights of `values`: named tensors in the current layout when the
/// count matches it, otherwise one flat `model.parameters` tensor.
pub(super) fn runtime_bytes(values: &[f32], metadata: HashMap<String, String>) -> Vec<u8> {
    let data: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    if values.len() != crate::MODEL_PARAMETER_COUNT {
        let tensor = TensorView::new(Dtype::F32, vec![values.len()], &data).expect("tensor");
        return serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture");
    }
    let schema = crate::PolicyModel::fresh(0)
        .and_then(|model| model.parameter_schema())
        .expect("schema");
    let mut offset = 0;
    let mut tensors = Vec::with_capacity(schema.len());
    for (name, shape) in schema {
        let bytes = shape.iter().product::<usize>() * 4;
        let view = TensorView::new(Dtype::F32, shape, &data[offset..offset + bytes]);
        tensors.push((name, view.expect("tensor")));
        offset += bytes;
    }
    serialize(tensors, Some(metadata)).expect("fixture")
}

pub(super) fn config() -> crate::PpoConfig {
    crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        samples_per_update: 4,
        minibatch: 2,
        ..crate::PpoConfig::default()
    }
}
