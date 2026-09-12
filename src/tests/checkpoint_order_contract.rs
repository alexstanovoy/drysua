use super::*;
use sha2::{Digest, Sha256};

const SOURCES: [(u32, u64, u32, &str); 2] = [
    (
        23,
        765_990_392_710_687_046,
        18,
        "adbecb8293548ad1602b24b046b9a6f032fc62798d46102029a1f3448d764bdd",
    ),
    (
        25,
        12_302_688_747_093_836_273,
        20,
        "d22a011f829c23593bbc6c03d22cbab9ab13bb79662959eda2d6c96b9de8098e",
    ),
];
const INITIALIZATION_SEED: u64 = 20_260_908;
const M12_PARAMETERS: usize = 1_689_076;

#[test]
fn selected_m12_initializer_rejects_each_missing_wrong_or_extra_metadata_key() {
    let directory = test_directory("m12-initializer-metadata");
    for (version, hash, rules, _) in SOURCES {
        let expected = m12_metadata(version, hash, rules);
        assert_eq!(expected.len(), 6);
        let mut invalid = vec![None, Some(current_runtime_metadata())];
        for key in expected.keys() {
            for replacement in [None, Some("0")] {
                let mut metadata = expected.clone();
                metadata.remove(key);
                if let Some(value) = replacement {
                    metadata.insert(key.clone(), value.to_owned());
                }
                invalid.push(Some(metadata));
            }
        }
        let mut extra = expected;
        extra.insert("unexpected".to_owned(), "1".to_owned());
        invalid.push(Some(extra));
        invalid.push(Some(m12_metadata(24, 17_486_156_843_355_673_207, 19)));
        for metadata in invalid {
            let data = [0; 4];
            let view = TensorView::new(Dtype::F32, vec![1], &data).expect("tensor");
            let bytes = serialize([("model.parameters", view)], metadata).expect("fixture");
            assert_initializer_rejects(&directory, &bytes, CheckpointError::SchemaMismatch);
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn selected_m12_initializer_rejects_names_dtype_shape_and_unapproved_sha() {
    let directory = test_directory("m12-initializer-tensor");
    let count = M12_PARAMETERS;
    let data = vec![0; (count + 1) * 4];
    for (version, hash, rules, _) in SOURCES {
        for (name, dtype, shape, expected) in [
            ("wrong", Dtype::F32, vec![count], "names"),
            (
                "model.parameters",
                Dtype::I32,
                vec![count],
                "dtype or shape",
            ),
            (
                "model.parameters",
                Dtype::F32,
                vec![count - 1],
                "dtype or shape",
            ),
            (
                "model.parameters",
                Dtype::F32,
                vec![count + 1],
                "dtype or shape",
            ),
            (
                "model.parameters",
                Dtype::F32,
                vec![1, count],
                "dtype or shape",
            ),
            (
                "model.parameters",
                Dtype::F32,
                vec![count],
                "selected M12 training source SHA-256",
            ),
        ] {
            let size = shape.iter().product::<usize>() * 4;
            let view = TensorView::new(dtype, shape, &data[..size]).expect("tensor");
            let bytes = serialize([(name, view)], Some(m12_metadata(version, hash, rules)))
                .expect("fixture");
            assert_initializer_rejects(
                &directory,
                &bytes,
                CheckpointError::TensorContract(expected),
            );
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn selected_m12_initializer_rejects_nonfinite_payload_at_both_boundaries() {
    let directory = test_directory("m12-initializer-nonfinite");
    let count = M12_PARAMETERS;
    for (version, hash, rules, _) in SOURCES {
        for index in [0, count - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut data = vec![0; count * 4];
                data[index * 4..(index + 1) * 4].copy_from_slice(&value.to_le_bytes());
                let view = TensorView::new(Dtype::F32, vec![count], &data).expect("tensor");
                let bytes = serialize(
                    [("model.parameters", view)],
                    Some(m12_metadata(version, hash, rules)),
                )
                .expect("fixture");
                assert_initializer_rejects(
                    &directory,
                    &bytes,
                    CheckpointError::NonFiniteTensor {
                        name: "model.parameters",
                        index,
                    },
                );
            }
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn selected_m12_initializer_rejects_m15_count_under_exact_historical_metadata() {
    let directory = test_directory("m12-no-m15-count-relabel");
    let count = crate::MODEL_PARAMETER_COUNT;
    assert_ne!(count, M12_PARAMETERS);
    let data = vec![0; count * 4];
    for (version, hash, rules, _) in SOURCES {
        let tensor = TensorView::new(Dtype::F32, vec![count], &data).expect("M15-sized tensor");
        let bytes = serialize(
            [("model.parameters", tensor)],
            Some(m12_metadata(version, hash, rules)),
        )
        .expect("historical metadata with wrong current count");
        assert_initializer_rejects(
            &directory,
            &bytes,
            CheckpointError::TensorContract("dtype or shape"),
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

fn assert_initializer_rejects(
    directory: &std::path::Path,
    bytes: &[u8],
    expected: CheckpointError,
) {
    let path = directory.join("drysua.weights.safetensors");
    fs::write(&path, bytes).expect("fixture");

    let error =
        TrainingArtifact::initialize_selected_m12_for_training(directory, 1, PolicyDevice::Cpu)
            .err()
            .expect("audited M12 source only");

    let message = match &expected {
        CheckpointError::SchemaMismatch => "checkpoint schema does not match this build".to_owned(),
        CheckpointError::TensorContract(field) => {
            format!("checkpoint tensor contract has invalid {field}")
        }
        CheckpointError::NonFiniteTensor { name, index } => {
            format!("checkpoint tensor {name} contains non-finite value at {index}")
        }
        _ => panic!("unexpected initializer error fixture"),
    };
    assert_eq!(error.to_string(), message);
    assert_eq!(error, expected);
    assert_eq!(fs::read(path).expect("source untouched"), bytes);
}

#[test]
#[ignore = "read-only DRYSUA_SELECTED_M12_SOURCE audit; legacy initialization must reject, no outputs or training"]
fn selected_m12_local_artifact_is_retired_without_writing_or_mutating_source() {
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M12_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("source directory");
    let path = source.join("drysua.weights.safetensors");
    let bytes = fs::read(&path).expect("audited source");
    let digest = hex_digest(&bytes);
    let &(version, hash, rules, _) = SOURCES
        .iter()
        .find(|source| source.3 == digest)
        .expect("precisely one of two approved digests");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    assert_eq!(
        metadata.metadata().as_ref(),
        Some(&m12_metadata(version, hash, rules))
    );
    assert_old_runtime_rejected(&source);

    let error = TrainingArtifact::initialize_selected_m12_for_training(
        &source,
        INITIALIZATION_SEED,
        PolicyDevice::Cpu,
    )
    .err()
    .expect("historical authorization cannot initialize Map2");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    let payload = parameter_payload(&bytes);
    let (values, remainder) = payload.as_chunks::<4>();
    assert!(remainder.is_empty());
    assert!(
        values
            .iter()
            .all(|bytes| f32::from_le_bytes(*bytes).is_finite())
    );
    assert_eq!(fs::read(&path).expect("source unchanged"), bytes);
    assert_eq!(hex_digest(&fs::read(&path).expect("source rehash")), digest);
    verify_source_digest_and_tuple_are_paired(&bytes, version);
    eprintln!(
        "retired_m12 source_sha256={digest} initialization=rejected runtime=rejected source_unchanged=true outputs=none"
    );
}

fn assert_old_runtime_rejected(source: &std::path::Path) {
    let runtime = PolicyModel::fresh(1).expect("runtime");
    let identity = runtime.policy_identity().expect("identity");
    let parameters = runtime.export_parameters().expect("parameters");
    let error = TrainingArtifact::load_runtime_weights(&runtime, source).expect_err("old runtime");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        runtime.policy_identity().expect("unchanged identity"),
        identity
    );
    assert_eq!(
        runtime.export_parameters().expect("unchanged parameters"),
        parameters
    );
}

fn parameter_payload(bytes: &[u8]) -> Vec<u8> {
    let tensors = safetensors::SafeTensors::deserialize(bytes).expect("tensors");
    let tensor = tensors.tensor("model.parameters").expect("parameters");
    assert_eq!(tensor.dtype(), Dtype::F32);
    assert_eq!(tensor.shape(), &[M12_PARAMETERS]);
    tensor.data().to_vec()
}

fn verify_source_digest_and_tuple_are_paired(bytes: &[u8], version: u32) {
    let directory = test_directory("m12-paired-source-sha");
    let payload = parameter_payload(bytes);
    let &(other_version, hash, rules, _) = SOURCES
        .iter()
        .find(|source| source.0 != version)
        .expect("other source");
    let tensor = TensorView::new(Dtype::F32, vec![M12_PARAMETERS], &payload).expect("tensor");
    let changed = serialize(
        [("model.parameters", tensor)],
        Some(m12_metadata(other_version, hash, rules)),
    )
    .expect("changed tuple");
    assert_initializer_rejects(
        &directory,
        &changed,
        CheckpointError::TensorContract("selected M12 training source SHA-256"),
    );
    let mut corrupted = bytes.to_vec();
    let last_value = corrupted.len() - 4;
    corrupted[last_value] ^= 1;
    assert_initializer_rejects(
        &directory,
        &corrupted,
        CheckpointError::TensorContract("selected M12 training source SHA-256"),
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
