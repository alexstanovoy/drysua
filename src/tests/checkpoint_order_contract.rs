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
    let count = crate::MODEL_PARAMETER_COUNT;
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
    let count = crate::MODEL_PARAMETER_COUNT;
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
#[ignore = "requires DRYSUA_SELECTED_M12_SOURCE, new DRYSUA_M14_INITIALIZATION_OUTPUT, DRYSUA_INITIALIZATION_GIT_COMMIT and DRYSUA_INITIALIZATION_SIMULATOR_COMMIT; no gameplay or updates"]
fn selected_m12_local_artifact_initializes_m14_with_bit_identical_parameters_and_fresh_state() {
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M12_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("source directory");
    let output = PathBuf::from(
        std::env::var_os("DRYSUA_M14_INITIALIZATION_OUTPUT").expect("explicit new output"),
    );
    assert!(!output.exists(), "initializer never overwrites an output");
    let parent = output
        .parent()
        .expect("output parent")
        .canonicalize()
        .expect("existing parent");
    assert!(
        !parent.starts_with(&source),
        "output must be outside source"
    );
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

    let (model, returned_digest) = TrainingArtifact::initialize_selected_m12_for_training(
        &source,
        INITIALIZATION_SEED,
        PolicyDevice::Cpu,
    )
    .expect("explicit initialization, not resume");

    assert_eq!(returned_digest, <[u8; 32]>::from(Sha256::digest(&bytes)));
    let payload = parameter_payload(&bytes);
    assert_parameter_bits(&model, &payload);
    let schema = model.parameter_schema().expect("62 named tensors");
    PolicyModel::validate_m12_parameter_schema(&schema).expect("audited M12 layout");
    assert_eq!(schema.len(), 62);
    fs::create_dir(&output).expect("new output only");
    TrainingArtifact::save_runtime_weights(&model, &output).expect("M14 initialized runtime");
    let reloaded = PolicyModel::fresh(2).expect("new runtime");
    TrainingArtifact::load_runtime_weights(&reloaded, &output).expect("M14 strict reload");
    assert_parameter_bits(&reloaded, &payload);
    save_and_verify_fresh_checkpoint(&model, &output, &payload);
    assert_eq!(fs::read(&path).expect("source unchanged"), bytes);
    assert_eq!(hex_digest(&fs::read(&path).expect("source rehash")), digest);
    verify_source_digest_and_tuple_are_paired(&bytes, version);
    write_initialization_audit(&model, &source, &output, &digest);
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
    assert_eq!(tensor.shape(), &[1_689_076]);
    tensor.data().to_vec()
}

fn assert_parameter_bits(model: &PolicyModel, expected: &[u8]) {
    let values = model.export_parameters().expect("parameters");
    assert_eq!(values.len(), 1_689_076);
    assert!(values.iter().all(|value| value.is_finite()));
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(
        bytes, expected,
        "all numeric F32 parameter bits, including signed zero"
    );
}

fn fresh_progress() -> CheckpointProgress {
    CheckpointProgress {
        global_update: 0,
        policy_version: 0,
        scheduler_step: 0,
        curriculum_stage: 0,
        rollout_samples: 0,
        best_evaluation: None,
        rng_states: Vec::new(),
        league_references: Vec::new(),
    }
}

fn initialization_run() -> CheckpointRun {
    CheckpointRun {
        git_commit: std::env::var("DRYSUA_INITIALIZATION_GIT_COMMIT").expect("actual source revision"),
        simulator_commit: std::env::var("DRYSUA_INITIALIZATION_SIMULATOR_COMMIT").expect("actual simulator revision"),
        enabled_features: crate::compiled_features(),
        command_line: "cargo test --release --features builtin --lib selected_m12_local_artifact_initializes_m14_with_bit_identical_parameters_and_fresh_state --quiet -- --ignored --nocapture".to_owned(),
        run_seed: INITIALIZATION_SEED, map: MapId(0), hero: crate::SHADOW_FIEND,
        device: CheckpointDevice::Cpu, batch_size: checkpoint_config().minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn save_and_verify_fresh_checkpoint(model: &PolicyModel, output: &std::path::Path, payload: &[u8]) {
    let trainer =
        PpoTrainer::new(model, checkpoint_config(), INITIALIZATION_SEED).expect("fresh trainer");
    assert_fresh_trainer(model, &trainer);
    let run = initialization_run();
    let artifact = TrainingArtifact::capture(model, &trainer, run.clone(), fresh_progress())
        .expect("fresh capture");
    assert_eq!(
        artifact.save(output).expect("fresh save"),
        crate::CheckpointSaveOutcome::Committed
    );
    let loaded = TrainingArtifact::load_compatible(output, &run).expect("strict fresh checkpoint");
    assert_eq!(loaded.progress(), &fresh_progress());
    let restored = PolicyModel::fresh(3).expect("new owned restore target");
    let state = loaded
        .restore(&restored, &run)
        .expect("restore only newly initialized state");
    assert_fresh_trainer(&restored, state.trainer());
    assert_parameter_bits(&restored, payload);
    assert_eq!(state.trainer().rng_checkpoint(), trainer.rng_checkpoint());
    assert_ne!(
        restored.policy_identity().expect("restored identity"),
        model.policy_identity().expect("initialized identity")
    );
}

fn assert_fresh_trainer(model: &PolicyModel, trainer: &PpoTrainer) {
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.rng_checkpoint().1, 0);
    let snapshot = trainer.checkpoint_snapshot(model).expect("bound optimizer");
    assert_eq!(snapshot.adam.step(), 0);
    for moment in [snapshot.adam.moments().0, snapshot.adam.moments().1] {
        assert_eq!(moment.len(), crate::MODEL_PARAMETER_COUNT);
        assert!(moment.iter().all(|value| value.to_bits() == 0));
    }
}

fn verify_source_digest_and_tuple_are_paired(bytes: &[u8], version: u32) {
    let directory = test_directory("m12-paired-source-sha");
    let payload = parameter_payload(bytes);
    let &(other_version, hash, rules, _) = SOURCES
        .iter()
        .find(|source| source.0 != version)
        .expect("other source");
    let tensor =
        TensorView::new(Dtype::F32, vec![crate::MODEL_PARAMETER_COUNT], &payload).expect("tensor");
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
    *corrupted.last_mut().expect("nonempty payload") ^= 1;
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

fn write_initialization_audit(
    model: &PolicyModel,
    source: &std::path::Path,
    output: &std::path::Path,
    source_hash: &str,
) {
    use std::fmt::Write as _;
    let bytes = fs::read(output.join("drysua.weights.safetensors")).expect("output");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("output metadata");
    assert_eq!(
        metadata.metadata().as_ref(),
        Some(&current_runtime_metadata())
    );
    let schema = model.parameter_schema().expect("schema");
    let mut audit = format!(
        "status=initialized_policy_not_qualified_release\ncontract=candidate_order_m14_not_old_gameplay_compatible\nsource={}\nsource_sha256={source_hash}\noutput={}\noutput_sha256={}\nparameter_payload_sha256={}\nseed={INITIALIZATION_SEED}\ndtype=F32\nparameters=1689076\nnamed_layout_count=62\nparameter_bits_identical_after_runtime_and_checkpoint_reload=true\nsource_unchanged=true\noptimizer_step=0\noptimizer_moments=all_positive_zero\nprogress=all_zero_no_evaluation_rng_history_or_league\ntraining_updates=0\ngameplay_runs=0\nfeature_version={}\nmodel_version={}\nleague_version={}\nleague_hash={}\n",
        source.display(),
        output.display(),
        hex_digest(&bytes),
        hex_digest(&parameter_payload(&bytes)),
        crate::FEATURE_SCHEMA_VERSION,
        crate::MODEL_SCHEMA_VERSION,
        crate::LEAGUE_SCHEMA_VERSION,
        crate::LEAGUE_SCHEMA_HASH
    );
    let sorted: std::collections::BTreeMap<_, _> = current_runtime_metadata().into_iter().collect();
    for (name, value) in sorted {
        writeln!(audit, "{name}={value}").expect("audit");
    }
    for (name, shape) in schema {
        writeln!(audit, "layout.{name}={shape:?}").expect("layout audit");
    }
    for name in ["checkpoint.meta", "checkpoint.safetensors"] {
        writeln!(
            audit,
            "{name}.sha256={}",
            hex_digest(&fs::read(output.join(name)).expect("checkpoint"))
        )
        .expect("checkpoint audit");
    }
    fs::write(output.join("INITIALIZATION.txt"), &audit).expect("audit output");
    eprintln!("{audit}");
}
