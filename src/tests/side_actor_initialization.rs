use super::*;

#[test]
#[ignore = "explicit pinned U428 migration gate; exclusive bounded runner only"]
fn selected_u428_expands_only_for_initialization_and_preserves_source() {
    let directory = std::path::Path::new(
        "/home/alexstanovoy/Workspace/bots/drysua/artifacts/temp/annealed-teacher-20260920/attempt-008/history/update-0428",
    );
    let bytes = read_bounded(&directory.join(RUNTIME_TENSOR_FILE), SOURCE_MAX_BYTES).unwrap();
    assert_eq!(sha256(&bytes), SOURCE_SHA256);
    let legacy = decode_source_contract(&bytes).unwrap();
    let (model, provenance) = TrainingArtifact::initialize_selected_m24_u428_for_side_actors(
        directory,
        9001,
        PolicyDevice::Cpu,
    )
    .unwrap();
    let expanded = model.export_parameters().unwrap();
    assert_eq!(expanded.len(), 1_812_983);
    assert!(
        expanded[..legacy.len()]
            .iter()
            .zip(&legacy)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );
    assert!(provenance.contains("INITIALIZATION_ONLY"));
    assert!(provenance.contains("optimizer_progress_mastery_rng_league=fresh"));
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, directory),
        Err(CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        TrainingArtifact::load(directory).err().unwrap(),
        CheckpointError::SchemaMismatch
    );
    eprintln!(
        "side-actor-identities model={}:{} ppo={}:{} checkpoint={}:{} league={}:{} parameters={} tensors=86",
        crate::MODEL_SCHEMA_VERSION,
        crate::MODEL_SCHEMA_HASH,
        crate::PPO_SCHEMA_VERSION,
        crate::PPO_SCHEMA_HASH,
        crate::CHECKPOINT_SCHEMA_VERSION,
        crate::CHECKPOINT_SCHEMA_HASH,
        crate::LEAGUE_SCHEMA_VERSION,
        crate::LEAGUE_SCHEMA_HASH,
        expanded.len()
    );
    assert_eq!(
        sha256(&read_bounded(&directory.join(RUNTIME_TENSOR_FILE), SOURCE_MAX_BYTES).unwrap()),
        SOURCE_SHA256
    );
}

fn source_bytes(
    name: &str,
    dtype: Dtype,
    count: usize,
    metadata: HashMap<String, String>,
    nonfinite: bool,
) -> Vec<u8> {
    let mut data = vec![0; count * 4];
    if nonfinite {
        data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    }
    let view = TensorView::new(dtype, vec![count], &data).unwrap();
    serialize([(name, view)], Some(metadata)).unwrap()
}

#[test]
fn pinned_source_rejects_unrecognized_and_malformed_bytes_before_decode() {
    for bytes in [Vec::new(), b"not safetensors".to_vec()] {
        assert_eq!(
            decode_selected_source(&bytes),
            Err(CheckpointError::TensorHashMismatch)
        );
    }
}

#[test]
fn source_contract_rejects_wrong_name_dtype_shape_metadata_and_nonfinite() {
    let count = crate::model::LEGACY_MODEL_PARAMETER_COUNT;
    for (name, dtype, length, metadata, nonfinite, expected) in [
        (
            "wrong",
            Dtype::F32,
            count,
            source_metadata(),
            false,
            CheckpointError::TensorContract("names"),
        ),
        (
            "model.parameters",
            Dtype::I32,
            count,
            source_metadata(),
            false,
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            count - 1,
            source_metadata(),
            false,
            CheckpointError::TensorContract("dtype or shape"),
        ),
        (
            "model.parameters",
            Dtype::F32,
            count,
            HashMap::new(),
            false,
            CheckpointError::SchemaMismatch,
        ),
        (
            "model.parameters",
            Dtype::F32,
            count,
            source_metadata(),
            true,
            CheckpointError::NonFiniteTensor {
                name: "model.parameters",
                index: 0,
            },
        ),
    ] {
        let bytes = source_bytes(name, dtype, length, metadata, nonfinite);
        assert_eq!(decode_source_contract(&bytes), Err(expected));
        assert_eq!(
            decode_selected_source(&bytes),
            Err(CheckpointError::TensorHashMismatch)
        );
    }
}

#[test]
fn unpinned_legacy_is_neither_runtime_nor_initialization_compatible() {
    let bytes = source_bytes(
        "model.parameters",
        Dtype::F32,
        crate::model::LEGACY_MODEL_PARAMETER_COUNT,
        source_metadata(),
        false,
    );
    assert_eq!(
        super::super::decode_runtime_tensor(&bytes),
        Err(CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        decode_initial_parameters(&bytes),
        Err(CheckpointError::SchemaMismatch)
    );
}

#[test]
fn current_initialization_preserves_strict_runtime_errors() {
    let metadata = super::super::runtime_tensor_metadata_map(crate::PpoSampleBudget::Standard);
    let bytes = source_bytes("model.parameters", Dtype::F32, 1, metadata, false);
    assert_eq!(
        decode_initial_parameters(&bytes),
        Err(CheckpointError::TensorContract("dtype or shape"))
    );
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn invalid_initialization_is_rejected_before_invalid_cuda_device_creation() {
    let directory =
        std::env::temp_dir().join(format!("side-actors-before-device-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let bytes = source_bytes(
        "model.parameters",
        Dtype::F32,
        crate::model::LEGACY_MODEL_PARAMETER_COUNT,
        source_metadata(),
        false,
    );
    std::fs::write(directory.join(RUNTIME_TENSOR_FILE), &bytes).unwrap();
    let result = TrainingArtifact::initialize_from_weights(
        &directory,
        9001,
        PolicyDevice::Cuda {
            ordinal: usize::MAX,
        },
    );
    assert_eq!(result.err().unwrap(), CheckpointError::SchemaMismatch);
    assert_eq!(
        std::fs::read(directory.join(RUNTIME_TENSOR_FILE)).unwrap(),
        bytes
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn current_initialization_accepts_exact_current_parameters() {
    let metadata = super::super::runtime_tensor_metadata_map(crate::PpoSampleBudget::Standard);
    let bytes = source_bytes(
        "model.parameters",
        Dtype::F32,
        crate::MODEL_PARAMETER_COUNT,
        metadata,
        false,
    );
    let parameters = decode_initial_parameters(&bytes).unwrap();
    assert_eq!(parameters.len(), crate::MODEL_PARAMETER_COUNT);
    assert!(parameters.iter().all(|value| value.to_bits() == 0));
}

#[test]
fn m21_initialization_rejects_side_actors_before_directory_read() {
    let error = TrainingArtifact::initialize_selected_m21_for_terminal_reward(
        std::path::Path::new("/nonexistent/m21-side-actor-source"),
        1,
        PolicyDevice::Cpu,
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        CheckpointError::TensorContract(
            "M21 initialization unsupported with side-actors; use pinned M24/u428"
        )
    );
}

#[test]
fn expansion_preserves_legacy_bits_and_duplicates_both_actor_ranges() {
    let source: Vec<f32> = (0..crate::model::LEGACY_MODEL_PARAMETER_COUNT)
        .map(|index| index as f32)
        .collect();
    let expanded = crate::expand_m24_side_actor_parameters(&source).unwrap();
    assert_eq!(&expanded[..source.len()], source.as_slice());
    let actor: Vec<f32> = source[1_586_113..1_590_225]
        .iter()
        .chain(&source[1_591_169..1_700_020])
        .copied()
        .collect();
    assert_eq!(&expanded[source.len()..], actor.as_slice());
}

#[test]
fn rejected_initial_weights_leave_model_and_source_files_unchanged() {
    let directory = std::env::temp_dir().join(format!(
        "side-actors-rejected-initialization-{}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let model = PolicyModel::fresh(428).unwrap();
    let before = model.export_parameters().unwrap();
    let bytes = source_bytes(
        "model.parameters",
        Dtype::F32,
        crate::model::LEGACY_MODEL_PARAMETER_COUNT,
        source_metadata(),
        false,
    );
    std::fs::write(directory.join(RUNTIME_TENSOR_FILE), &bytes).unwrap();
    std::fs::write(directory.join("checkpoint.meta"), b"must not be read").unwrap();
    std::fs::write(
        directory.join("checkpoint.safetensors"),
        b"no Adam or RNG import",
    )
    .unwrap();

    let result = TrainingArtifact::load_initial_weights(&model, &directory);

    assert_eq!(result, Err(CheckpointError::SchemaMismatch));
    let after = model.export_parameters().unwrap();
    assert!(
        before
            .iter()
            .zip(&after)
            .all(|(left, right)| left.to_bits() == right.to_bits())
    );
    assert_eq!(before.len(), after.len());
    assert_eq!(
        std::fs::read(directory.join(RUNTIME_TENSOR_FILE)).unwrap(),
        bytes
    );
    assert_eq!(
        std::fs::read(directory.join("checkpoint.meta")).unwrap(),
        b"must not be read"
    );
    assert_eq!(
        std::fs::read(directory.join("checkpoint.safetensors")).unwrap(),
        b"no Adam or RNG import"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
