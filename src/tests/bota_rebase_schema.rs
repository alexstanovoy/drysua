use super::map2_checkpoint::runtime_bytes;
use crate::{CheckpointError, PolicyModel, TrainingArtifact};

#[test]
fn historical_bindings_and_shapes_cannot_be_relabelled_as_current_runtime() {
    let model = PolicyModel::fresh(37).expect("model");
    let trainer =
        crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 38).expect("trainer");
    let prior = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let directory = crate::ppo::test_directory("map2-checkpoint");
    let path = directory.join("drysua.weights.safetensors");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("current runtime");
    let current = std::fs::read(&path).expect("current bytes");
    let (_, header) = safetensors::SafeTensors::read_metadata(&current).expect("metadata");
    let current = header.metadata().clone().expect("current metadata");
    let mut cases = Vec::with_capacity(18);
    // Isolate each historical binding against CURRENT instead of failing on an earlier old key.
    for (key, m15, m18) in [
        (
            "action_schema_hash",
            "14080316840523410707",
            "10658390830565586343",
        ),
        (
            "feature_schema_hash",
            "612467982395246657",
            "17888785275670453418",
        ),
        (
            "model_schema_hash",
            "149485500614302181",
            "3900982062969752096",
        ),
        ("ppo_schema_version", "28", "31"),
        (
            "ppo_schema_hash",
            "16579842539143021978",
            "15379677344330093698",
        ),
        ("ppo_rules_audit_version", "23", "26"),
        ("map2_reward_version", "1", "7"),
    ] {
        for old in [m15, m18] {
            let mut metadata = current.clone();
            let replaced = metadata
                .insert(key.to_owned(), old.to_owned())
                .expect("current binding");
            if replaced == old {
                continue;
            }
            cases.push((
                format!("{key}={old}"),
                1,
                metadata,
                CheckpointError::SchemaMismatch,
                "checkpoint schema does not match this build",
            ));
        }
    }
    for count in [1_695_924, 1_697_460, 1_812_983] {
        cases.push((
            format!("legacy flat layout {count}"),
            count,
            current.clone(),
            CheckpointError::TensorContract("names"),
            "checkpoint tensor contract has invalid names",
        ));
    }
    for (name, count, metadata, kind, expected) in cases {
        let bytes = runtime_bytes(&vec![0.0; count], metadata);
        std::fs::write(&path, &bytes).expect("incompatible runtime");
        let error = TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("reject");
        assert_eq!(error, kind, "{name}");
        assert_eq!(error.to_string(), expected, "{name}");
        super::support::assert_bits(&model.export_parameters().expect("after"), &prior);
        assert_eq!(
            model.policy_identity().expect("identity"),
            identity,
            "{name}"
        );
        super::support::assert_fresh_state(&model, &trainer);
        assert_eq!(
            std::fs::read(&path).expect("source unchanged"),
            bytes,
            "{name}"
        );
    }
}

#[test]
fn legacy_effect_and_progress_manifests_reject_before_tensor_access() {
    for (version, hash) in [
        (3u32, 6_904_067_705_245_923_052u64),
        (6, 16_772_919_360_388_607_733),
    ] {
        let directory = crate::ppo::test_directory("map2-checkpoint");
        let path = directory.join("checkpoint.meta");
        let mut bytes = b"DRYCKP21".to_vec();
        bytes.extend(version.to_le_bytes());
        bytes.extend(hash.to_le_bytes());
        std::fs::write(&path, &bytes).expect("old manifest header");
        let error = TrainingArtifact::load(&directory).expect_err("old resume");
        assert_eq!(error, CheckpointError::SchemaMismatch, "{version}");
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
        assert_eq!(std::fs::read(path).expect("unchanged header"), bytes);
        assert_eq!(std::fs::read_dir(&directory).expect("directory").count(), 1);
    }
}
