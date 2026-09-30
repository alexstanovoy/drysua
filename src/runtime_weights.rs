//! Runtime weights: one named F32 tensor per model parameter plus schema metadata.
//!
//! Deployment loading is strict: the metadata, the tensor names and every shape
//! must match this build. A warm start only reuses parameters: each current
//! tensor whose name and shape the file carries is copied, every other one
//! keeps its fresh initialization, and file tensors the model no longer has
//! are dropped.

use std::collections::HashMap;

use safetensors::tensor::{Dtype, SafeTensors};

use super::{
    ACTION_SCHEMA_HASH, CheckpointError, FEATURE_SCHEMA_HASH, MAP2_REWARD_VERSION,
    MODEL_SCHEMA_HASH, PPO_RULES_AUDIT_VERSION, PPO_SCHEMA_HASH, PPO_SCHEMA_VERSION, encode_f32,
    push_json_string, validate_tensor_values,
};

/// Stable parameter names and shapes in export order.
pub(super) type ParameterSchema = [(&'static str, Vec<usize>)];

/// What a warm start took from its file.
#[cfg(feature = "builtin")]
#[derive(Debug)]
pub(super) struct WarmStart {
    pub(super) parameters: Vec<f32>,
    pub(super) reused_tensors: usize,
    pub(super) reused_parameters: usize,
    /// Current tensors the file lacks or holds with another shape.
    pub(super) reinitialized: Vec<&'static str>,
    /// File tensors the current model does not have.
    pub(super) dropped_tensors: usize,
    /// Metadata keys of this build whose stored values differ.
    pub(super) differing_metadata: Vec<String>,
}

/// Serializes named tensors with metadata in canonical key order.
///
/// The safetensors writer takes a `HashMap` and emits its iteration order,
/// which is randomized per process; this writer emits sorted metadata keys and
/// tensors in parameter order, so identical weights give identical bytes.
pub(super) fn serialize(
    schema: &ParameterSchema,
    parameters: &[f32],
) -> Result<Vec<u8>, CheckpointError> {
    if schema_elements(schema) != parameters.len() {
        return Err(CheckpointError::TensorContract("element count"));
    }
    let metadata = metadata();
    debug_assert!(metadata.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let mut header = String::with_capacity(16 * 1024);
    header.push_str("{\"__metadata__\":{");
    for (index, (key, value)) in metadata.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        push_json_string(&mut header, key);
        header.push(':');
        push_json_string(&mut header, value);
    }
    header.push('}');
    let mut offset = 0usize;
    for (name, shape) in schema {
        let bytes = shape.iter().product::<usize>() * 4;
        header.push(',');
        push_json_string(&mut header, name);
        let dimensions = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            ":{{\"dtype\":\"F32\",\"shape\":[{dimensions}],\"data_offsets\":[{offset},{}]}}",
            offset + bytes
        ));
        offset += bytes;
    }
    header.push('}');
    let mut header = header.into_bytes();
    let aligned = header.len().next_multiple_of(8);
    header.resize(aligned, b' ');
    let mut bytes = Vec::with_capacity(8 + aligned + offset);
    bytes.extend_from_slice(&(aligned as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&encode_f32(parameters));
    Ok(bytes)
}

/// Parameters of a file whose metadata, names and shapes all match this build.
pub(super) fn decode_strict(
    bytes: &[u8],
    schema: &ParameterSchema,
) -> Result<Vec<f32>, CheckpointError> {
    let (_, header) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if header.metadata().as_ref() != Some(&metadata_map()) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if tensors.len() != schema.len() {
        return Err(CheckpointError::TensorContract("names"));
    }
    let mut parameters = Vec::with_capacity(schema_elements(schema));
    for (name, shape) in schema {
        let values = tensor_values(&tensors, name, shape)?
            .ok_or(CheckpointError::TensorContract("names"))?;
        parameters.extend(values);
    }
    Ok(parameters)
}

/// Parameters for a warm start: file tensors where name and shape match, `fresh` elsewhere.
///
/// A file that shares no tensor with this model is refused: it cannot be a
/// warm start of it.
#[cfg(feature = "builtin")]
pub(super) fn decode_warm_start(
    bytes: &[u8],
    schema: &ParameterSchema,
    fresh: &[f32],
) -> Result<WarmStart, CheckpointError> {
    assert_eq!(schema_elements(schema), fresh.len());
    let (_, header) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    let differing_metadata = differing_metadata(header.metadata().clone().unwrap_or_default());
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    let mut warm = WarmStart {
        parameters: Vec::with_capacity(fresh.len()),
        reused_tensors: 0,
        reused_parameters: 0,
        reinitialized: Vec::new(),
        dropped_tensors: 0,
        differing_metadata,
    };
    let mut offset = 0usize;
    for (name, shape) in schema {
        let count = shape.iter().product::<usize>();
        match tensor_values(&tensors, name, shape) {
            Ok(Some(values)) => {
                warm.parameters.extend(values);
                warm.reused_tensors += 1;
                warm.reused_parameters += count;
            }
            Ok(None) | Err(CheckpointError::TensorContract("dtype or shape")) => {
                warm.parameters
                    .extend_from_slice(&fresh[offset..offset + count]);
                warm.reinitialized.push(name);
            }
            Err(error) => return Err(error),
        }
        offset += count;
    }
    warm.dropped_tensors = tensors
        .names()
        .into_iter()
        .filter(|name| !schema.iter().any(|(known, _)| known == name))
        .count();
    if warm.reused_tensors == 0 {
        return Err(CheckpointError::TensorContract("names"));
    }
    Ok(warm)
}

/// One named tensor's values, `None` when the file lacks it.
fn tensor_values(
    tensors: &SafeTensors<'_>,
    name: &'static str,
    shape: &[usize],
) -> Result<Option<Vec<f32>>, CheckpointError> {
    let Ok(tensor) = tensors.tensor(name) else {
        return Ok(None);
    };
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err(CheckpointError::TensorContract("dtype or shape"));
    }
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(CheckpointError::TensorContract("byte alignment"));
    }
    let values = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    validate_tensor_values(name, &values)?;
    Ok(Some(values))
}

fn schema_elements(schema: &ParameterSchema) -> usize {
    schema
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum()
}

/// This build's metadata keys whose stored values differ, plus a count of unknown keys.
///
/// Unknown keys are counted, never echoed, so log lines stay well formed.
#[cfg(feature = "builtin")]
fn differing_metadata(stored: HashMap<String, String>) -> Vec<String> {
    let mut differences = metadata()
        .into_iter()
        .filter(|(key, value)| stored.get(*key) != Some(value))
        .map(|(key, _)| key.to_owned())
        .collect::<Vec<_>>();
    let current = metadata_map();
    let unknown = stored
        .keys()
        .filter(|key| !current.contains_key(*key))
        .count();
    if unknown > 0 {
        differences.push(format!("unknown_keys:{unknown}"));
    }
    differences
}

/// The runtime weights metadata, sorted by key.
fn metadata() -> Vec<(&'static str, String)> {
    vec![
        ("action_schema_hash", ACTION_SCHEMA_HASH.to_string()),
        ("feature_schema_hash", FEATURE_SCHEMA_HASH.to_string()),
        ("map2_reward_version", MAP2_REWARD_VERSION.to_string()),
        ("model_schema_hash", MODEL_SCHEMA_HASH.to_string()),
        (
            "ppo_rules_audit_version",
            PPO_RULES_AUDIT_VERSION.to_string(),
        ),
        ("ppo_schema_hash", PPO_SCHEMA_HASH.to_string()),
        ("ppo_schema_version", PPO_SCHEMA_VERSION.to_string()),
    ]
}

/// The same metadata as an order-insensitive map, for loading.
fn metadata_map() -> HashMap<String, String> {
    metadata()
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}
