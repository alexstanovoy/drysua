use std::collections::HashMap;

use super::map2_checkpoint::runtime_bytes;
use crate::{CheckpointError, PolicyModel, TrainingArtifact};

#[test]
fn runtime_weights_with_missing_or_foreign_identity_are_rejected_before_tensor_access() {
    let model = PolicyModel::fresh(37).expect("model");
    let trainer =
        crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 38).expect("trainer");
    let prior = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let directory = crate::ppo::test_directory("map2-checkpoint");
    let path = directory.join("drysua.weights.safetensors");
    let current = current_runtime_roundtrip(&model, &directory);
    // A one-value tensor fails the tensor contract, so SchemaMismatch shows that
    // the identity is checked before any tensor is read.
    for (case, metadata) in foreign_identities(&current) {
        let bytes = runtime_bytes(&[0.0], metadata);
        std::fs::write(&path, &bytes).expect("foreign runtime");
        let error = TrainingArtifact::load_runtime_weights(&model, &directory).expect_err(&case);
        assert_eq!(error, CheckpointError::SchemaMismatch, "{case}");
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build",
            "{case}"
        );
        super::support::assert_bits(&model.export_parameters().expect("after"), &prior);
        assert_eq!(
            model.policy_identity().expect("identity"),
            identity,
            "{case}"
        );
        super::support::assert_fresh_state(&model, &trainer);
        assert_eq!(
            std::fs::read(&path).expect("source unchanged"),
            bytes,
            "{case}"
        );
    }
}

/// Saves `model`, checks that its weights load bit-exactly into another model,
/// also after re-encoding with an unordered metadata map, and returns the metadata.
fn current_runtime_roundtrip(
    model: &PolicyModel,
    directory: &std::path::Path,
) -> HashMap<String, String> {
    let path = directory.join("drysua.weights.safetensors");
    let parameters = model.export_parameters().expect("parameters");
    TrainingArtifact::save_runtime_weights(model, directory).expect("current runtime");
    let bytes = std::fs::read(&path).expect("current bytes");
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    let current = header.metadata().clone().expect("current metadata");
    assert_eq!(current, super::checkpoint::current_runtime_metadata());
    for bytes in [bytes, runtime_bytes(&parameters, current.clone())] {
        std::fs::write(&path, bytes).expect("current runtime");
        let target = PolicyModel::fresh(39).expect("target");
        TrainingArtifact::load_runtime_weights(&target, directory).expect("current import");
        super::support::assert_bits(&target.export_parameters().expect("imported"), &parameters);
    }
    current
}

/// Every identity key missing, unparsable or numerically adjacent to the current
/// value (an older or newer schema), plus one unexpected key.
fn foreign_identities(current: &HashMap<String, String>) -> Vec<(String, HashMap<String, String>)> {
    let mut cases = Vec::with_capacity(3 * current.len() + 1);
    for (key, value) in current {
        let adjacent = (value.parse::<u64>().expect("numeric identity") ^ 1).to_string();
        for replacement in [None, Some("foreign".to_owned()), Some(adjacent)] {
            let mut metadata = current.clone();
            metadata.remove(key);
            if let Some(replacement) = &replacement {
                metadata.insert(key.clone(), replacement.clone());
            }
            cases.push((format!("{key}={replacement:?}"), metadata));
        }
    }
    let mut unexpected = current.clone();
    unexpected.insert("unexpected".to_owned(), "1".to_owned());
    cases.push(("unexpected key".to_owned(), unexpected));
    cases
}
