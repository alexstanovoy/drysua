use super::*;
use crate::ppo::test_directory;

/// Offset of the 64-byte PPO config from the end of a fixed-schedule manifest:
/// config, four u64 counters, the tensor SHA-256 and the adaptive presence byte.
const CONFIG_FROM_END: usize = 64 + 4 * 8 + 32 + 1;

#[test]
fn manifest_records_current_identity_and_64_byte_config() {
    let artifact = manifest_artifact(0, 0);
    let bytes = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    assert_eq!(&bytes[..8], CHECKPOINT_MAGIC);
    assert_eq!(&bytes[8..12], &CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
    assert_eq!(&bytes[12..20], &CHECKPOINT_SCHEMA_HASH.to_le_bytes());
    assert_eq!(&bytes[56..60], &crate::PPO_SCHEMA_VERSION.to_le_bytes());
    let mut writer = ManifestWriter::default();
    encode_config(&mut writer, artifact.config).expect("config");
    assert_eq!(writer.bytes.len(), 64);
    let start = bytes.len() - CONFIG_FROM_END;
    assert_eq!(&bytes[start..start + 64], writer.bytes.as_slice());
    assert_eq!(bytes.last(), Some(&0));
    assert_eq!(
        decode_manifest(&bytes).expect("roundtrip").config(),
        artifact.config
    );
}

#[test]
fn manifests_reject_truncation_trailing_bytes_foreign_identities_and_bad_presence() {
    let encoded = encoded_manifest(0, 0);
    assert_eq!(
        decode_manifest(&encoded[..encoded.len() - 1]).expect_err("truncated"),
        CheckpointError::ManifestTruncated
    );
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert_eq!(
        decode_manifest(&trailing).expect_err("trailing"),
        CheckpointError::ManifestTrailingBytes
    );
    for range in [8..12, 12..20, 20..24, 24..32, 56..60, 60..68] {
        let mut foreign = encoded.clone();
        for byte in &mut foreign[range] {
            *byte ^= 0x5a;
        }
        let error = decode_manifest(&foreign).expect_err("foreign schema");
        assert_eq!(error, CheckpointError::SchemaMismatch);
        assert_eq!(
            error.to_string(),
            "checkpoint schema does not match this build"
        );
    }
    let mut presence = encoded;
    *presence.last_mut().expect("presence byte") = 2;
    assert_eq!(
        decode_manifest(&presence).expect_err("presence flag"),
        CheckpointError::InvalidManifest("adaptive environment presence")
    );
}

#[test]
fn invalid_config_rejects_before_tensor_io() {
    let directory = test_directory("invalid-config");
    let original = encoded_manifest(0, 0);
    for (offset, value, field) in [
        (0, 0u32, "PPO config"),
        (8, 0, "PPO config"),
        (8, crate::PPO_MAX_GAMES as u32 + 1, "PPO config"),
        (12, 17, "PPO config"),
        (16, 8_193, "PPO config"),
        (32, f32::NAN.to_bits(), "PPO config"),
        (52, 0.5f32.to_bits(), "Map2 reward discount"),
    ] {
        let mut encoded = original.clone();
        let start = encoded.len() - CONFIG_FROM_END + offset;
        encoded[start..start + 4].copy_from_slice(&value.to_le_bytes());
        fs::write(directory.join(CHECKPOINT_META_FILE), encoded).expect("manifest only");
        let error = TrainingArtifact::load(&directory).expect_err("config before missing tensors");
        assert_eq!(error, CheckpointError::InvalidManifest(field));
        assert_eq!(
            error.to_string(),
            format!("checkpoint manifest has invalid {field}")
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn sample_counters_accept_configured_boundary_and_reject_one_more() {
    for (games, updates, maximum) in [(40, 0, 0), (40, 1, 46_520), (40, 3, 139_560), (2, 3, 6_978)]
    {
        let mut artifact = manifest_artifact(updates, maximum);
        artifact.config.environments = games;
        let bytes = encode_manifest(&artifact, artifact.tensor_hash).expect("boundary");
        assert_eq!(
            decode_manifest(&bytes)
                .expect("accepted")
                .progress
                .rollout_samples,
            maximum
        );
        artifact.progress.rollout_samples += 1;
        let bytes = encode_manifest(&artifact, artifact.tensor_hash).expect("overflow");
        let error = decode_manifest(&bytes).expect_err("counter overflow");
        assert_eq!(
            error,
            CheckpointError::InvalidManifest("rollout sample counter")
        );
        assert_eq!(
            error.to_string(),
            "checkpoint manifest has invalid rollout sample counter"
        );
    }
}

#[test]
fn public_checkpoint_preserves_state_and_rejects_invalid_capture_restore() {
    let directory = test_directory("roundtrip");
    let fixture = manifest_artifact(0, 0);
    let source = PolicyModel::fresh(40_008).expect("source");
    let trainer = PpoTrainer::new(&source, fixture.config, 91).expect("trainer");
    let artifact = TrainingArtifact::capture(&source, &trainer, fixture.run, fixture.progress)
        .expect("capture");
    artifact.save(&directory).expect("save");
    let loaded = TrainingArtifact::load_compatible(&directory, artifact.run()).expect("load");
    let target = PolicyModel::fresh(40_009).expect("target");
    let state = loaded.restore(&target, artifact.run()).expect("restore");
    assert_eq!(state.trainer().config(), trainer.config());
    assert_eq!(state.progress(), artifact.progress());
    assert_eq!(state.trainer().rng_checkpoint(), trainer.rng_checkpoint());
    assert_eq!(state.trainer().optimizer_step(), trainer.optimizer_step());
    let snapshot = state
        .trainer()
        .checkpoint_snapshot(&target)
        .expect("restored snapshot");
    assert_eq!(
        snapshot.adam.moments(),
        trainer
            .checkpoint_snapshot(&source)
            .expect("source snapshot")
            .adam
            .moments()
    );
    assert_eq!(snapshot.parameters, artifact.parameters);
    drop(state);
    assert_invalid_capture_restore(&source, &trainer, &target, &loaded);
    fs::remove_dir_all(directory).expect("cleanup");
}

fn assert_invalid_capture_restore(
    source: &PolicyModel,
    trainer: &PpoTrainer,
    target: &PolicyModel,
    loaded: &TrainingArtifact,
) {
    let mut invalid = loaded.clone();
    invalid.progress.rollout_samples = 1;
    let field = "rollout sample counter";
    let error = TrainingArtifact::capture(
        source,
        trainer,
        invalid.run.clone(),
        invalid.progress.clone(),
    )
    .expect_err("invalid capture");
    assert_eq!(error, CheckpointError::InvalidManifest(field));
    assert_eq!(
        error.to_string(),
        format!("checkpoint manifest has invalid {field}")
    );
    let bytes = encode_manifest(&invalid, invalid.tensor_hash).expect("invalid fixture");
    assert_eq!(
        decode_manifest(&bytes).expect_err("invalid decode"),
        CheckpointError::InvalidManifest(field)
    );
    let before = target.export_parameters().expect("before");
    assert_eq!(
        invalid
            .restore(target, invalid.run())
            .err()
            .expect("invalid restore"),
        error
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn runtime_weights_roundtrip_and_reject_foreign_schema_without_mutation() {
    let directory = test_directory("runtime");
    let source = PolicyModel::fresh(40_012).expect("source");
    let model = PolicyModel::fresh(40_014).expect("target");
    let parameters = source.export_parameters().expect("parameters");
    let path = directory.join(RUNTIME_TENSOR_FILE);
    TrainingArtifact::save_runtime_weights(&source, &directory).expect("export");
    let bytes = fs::read(&path).expect("runtime");
    TrainingArtifact::load_runtime_weights(&model, &directory).expect("import");
    assert_eq!(model.export_parameters().expect("parameters"), parameters);
    let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
    let metadata = metadata.metadata().as_ref().expect("map");
    for change in [
        None,
        Some(("ppo_schema_version", "39".to_owned())),
        Some(("ppo_schema_hash", (PPO_SCHEMA_HASH ^ 1).to_string())),
        Some(("model_schema_hash", (MODEL_SCHEMA_HASH ^ 1).to_string())),
    ] {
        let invalid = change.is_some();
        let mut foreign = metadata.clone();
        if let Some((key, value)) = change {
            foreign.insert(key.to_owned(), value);
        }
        let data = encode_f32(&parameters);
        let tensor = TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("tensor");
        fs::write(
            &path,
            serialize([("model.parameters", tensor)], Some(foreign)).expect("fixture"),
        )
        .expect("write");
        let identity = model.policy_identity().expect("identity");
        let result = TrainingArtifact::load_runtime_weights(&model, &directory);
        if !invalid {
            result.expect("valid unordered metadata");
        } else {
            let error = result.expect_err("foreign import");
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
            assert_eq!(model.policy_identity().expect("identity"), identity);
        }
        assert_eq!(model.export_parameters().expect("parameters"), parameters);
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn public_loaders_reject_empty_and_oversized_files_before_decoding() {
    let directory = test_directory("file-bounds");
    let model = PolicyModel::fresh(40_013).expect("model");
    for (name, maximum) in [
        (CHECKPOINT_META_FILE, MAX_META_BYTES),
        (CHECKPOINT_TENSOR_FILE, MAX_TRAINING_TENSOR_BYTES),
        (RUNTIME_TENSOR_FILE, MAX_RUNTIME_TENSOR_BYTES),
    ] {
        fs::write(directory.join(CHECKPOINT_META_FILE), encoded_manifest(0, 0)).expect("manifest");
        for length in [0, maximum + 1] {
            File::create(directory.join(name))
                .expect("file")
                .set_len(length)
                .expect("sparse oversized file");
            let error = if name == RUNTIME_TENSOR_FILE {
                TrainingArtifact::load_runtime_weights(&model, &directory)
                    .expect_err("runtime size")
            } else {
                TrainingArtifact::load(&directory).expect_err("checkpoint size")
            };
            assert_eq!(error, CheckpointError::InvalidManifest("file size"));
            assert_eq!(
                error.to_string(),
                "checkpoint manifest has invalid file size"
            );
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

pub(super) fn manifest_artifact(updates: u64, samples: u64) -> TrainingArtifact {
    TrainingArtifact {
        run: CheckpointRun {
            git_commit: "capacity-drysua".to_owned(),
            simulator_commit: "capacity-simulator".to_owned(),
            enabled_features: compiled_features(),
            command_line: "train-annealed capacity-fixture".to_owned(),
            run_seed: 40_008,
            map: MapId(2),
            hero: SHADOW_FIEND,
            device: CheckpointDevice::Cpu,
            batch_size: 512,
            rules_audit_version: PPO_RULES_AUDIT_VERSION,
        },
        progress: CheckpointProgress {
            adaptive_environment: None,
            global_update: updates,
            policy_version: updates,
            scheduler_step: updates,
            curriculum_stage: 0,
            rollout_samples: samples,
            best_evaluation: None,
            rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 8, 40).expect("RNG")],
            league_references: Vec::new(),
        },
        config: PpoConfig {
            environments: 40,
            rollout_decisions: 1_163,
            decision_interval_ticks: 3,
            epochs: 1,
            minibatch: 512,
            gamma_tick: 1.0,
            ..PpoConfig::default()
        },
        trainer_updates: updates,
        shuffle: (17, 19),
        parameters: Vec::new(),
        optimizer: CheckpointOptimizer {
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            step: 0,
        },
        tensor_hash: [42; 32],
    }
}

fn encoded_manifest(updates: u64, samples: u64) -> Vec<u8> {
    let artifact = manifest_artifact(updates, samples);
    encode_manifest(&artifact, artifact.tensor_hash).expect("manifest fixture")
}
