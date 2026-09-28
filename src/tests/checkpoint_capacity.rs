use super::*;
use crate::ppo::test_directory;

#[test]
fn profile_manifests_preserve_exact_identity_and_64_byte_config() {
    let standard = encoded_manifest(PpoSampleBudget::Standard, 0, 0);
    for (budget, checkpoint_version, ppo_version) in [
        (PpoSampleBudget::Standard, 12u32, 37u32),
        (PpoSampleBudget::Annealed, 13, 38),
        (PpoSampleBudget::WideAnnealed, 14, 39),
    ] {
        let mut artifact = manifest_artifact(budget, 0, 0);
        artifact.config.environments = 2;
        let bytes = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
        assert_eq!(&bytes[8..12], &checkpoint_version.to_le_bytes());
        assert_eq!(&bytes[56..60], &ppo_version.to_le_bytes());
        assert_eq!(&bytes[80..], &standard[80..]);
        let mut writer = ManifestWriter::default();
        encode_config(&mut writer, artifact.config).expect("config");
        assert_eq!(writer.bytes.len(), 64);
        assert_eq!(
            &bytes[bytes.len() - 128..bytes.len() - 64],
            writer.bytes.as_slice()
        );
        assert_eq!(
            decode_manifest(&bytes).expect("roundtrip").config(),
            artifact.config
        );
    }
}

#[test]
fn profile_manifests_reject_truncation_trailing_bytes_and_mixed_identities() {
    let standard = encoded_manifest(PpoSampleBudget::Standard, 0, 0);
    let annealed = encoded_manifest(PpoSampleBudget::Annealed, 0, 0);
    let wide = encoded_manifest(PpoSampleBudget::WideAnnealed, 0, 0);
    for budget in [
        PpoSampleBudget::Standard,
        PpoSampleBudget::Annealed,
        PpoSampleBudget::WideAnnealed,
    ] {
        let encoded = encoded_manifest(budget, 0, 0);
        assert_eq!(
            decode_manifest(&encoded[..encoded.len() - 1]).expect_err("truncated"),
            CheckpointError::ManifestTruncated
        );
        let mut trailing = encoded;
        trailing.push(0);
        assert_eq!(
            decode_manifest(&trailing).expect_err("trailing"),
            CheckpointError::ManifestTrailingBytes
        );
    }
    for (source, other) in [
        (&standard, &annealed),
        (&annealed, &standard),
        (&standard, &wide),
        (&wide, &standard),
        (&annealed, &wide),
        (&wide, &annealed),
    ] {
        for range in [8..12, 12..20, 8..20, 56..60, 60..68, 56..68] {
            let mut mixed = source.clone();
            mixed[range.clone()].copy_from_slice(&other[range]);
            let error = decode_manifest(&mixed).expect_err("mixed schema");
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
    }
    for (source, other) in [
        (&annealed, &standard),
        (&wide, &standard),
        (&wide, &annealed),
    ] {
        let mut downgraded = source.clone();
        downgraded[..80].copy_from_slice(&other[..80]);
        assert_eq!(
            decode_manifest(&downgraded).expect_err("capacity downgrade"),
            CheckpointError::InvalidManifest("PPO config")
        );
    }
}

#[test]
fn invalid_nonstandard_config_rejects_before_tensor_io() {
    let directory = test_directory("invalid-config");
    for (budget, maximum_games) in [
        (PpoSampleBudget::Annealed, 40),
        (PpoSampleBudget::WideAnnealed, 80),
    ] {
        let original = encoded_manifest(budget, 0, 0);
        for (offset, value) in [
            (0, 0u32),
            (4, 1_162),
            (8, 1),
            (8, 3),
            (8, maximum_games + 1),
            (8, maximum_games + 2),
            (12, 17),
            (16, 8_193),
            (32, f32::NAN.to_bits()),
            (52, 0.0f32.to_bits()),
        ] {
            let mut encoded = original.clone();
            let start = encoded.len() - 128 + offset;
            encoded[start..start + 4].copy_from_slice(&value.to_le_bytes());
            fs::write(directory.join(CHECKPOINT_META_FILE), encoded).expect("manifest only");
            let error =
                TrainingArtifact::load(&directory).expect_err("config before missing tensors");
            assert_eq!(error, CheckpointError::InvalidManifest("PPO config"));
            assert_eq!(
                error.to_string(),
                "checkpoint manifest has invalid PPO config"
            );
        }
    }
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn profile_sample_counters_accept_configured_boundary_and_reject_one_more() {
    for (budget, games, updates, maximum) in [
        (PpoSampleBudget::Standard, 2, 0, 32_768),
        (PpoSampleBudget::Standard, 2, 3, 131_072),
        (PpoSampleBudget::Annealed, 40, 0, 0),
        (PpoSampleBudget::Annealed, 40, 1, 46_520),
        (PpoSampleBudget::Annealed, 40, 2, 93_040),
        (PpoSampleBudget::Annealed, 40, 3, 139_560),
        (PpoSampleBudget::Annealed, 2, 3, 6_978),
        (PpoSampleBudget::WideAnnealed, 80, 0, 0),
        (PpoSampleBudget::WideAnnealed, 80, 1, 93_040),
        (PpoSampleBudget::WideAnnealed, 80, 2, 186_080),
        (PpoSampleBudget::WideAnnealed, 80, 3, 279_120),
        (PpoSampleBudget::WideAnnealed, 40, 3, 139_560),
        (PpoSampleBudget::WideAnnealed, 2, 3, 6_978),
    ] {
        let mut artifact = manifest_artifact(budget, updates, maximum);
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
fn public_nonstandard_checkpoint_preserves_state_and_rejects_invalid_capture_restore() {
    for budget in [PpoSampleBudget::Annealed, PpoSampleBudget::WideAnnealed] {
        let directory = test_directory("roundtrip");
        let fixture = manifest_artifact(budget, 0, 0);
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
        let error = state
            .pipeline(1, 1, &target)
            .err()
            .expect("standard-only pipeline");
        assert_eq!(
            error,
            CheckpointError::InvalidManifest("annealed actor-learner pipeline")
        );
        assert_eq!(
            error.to_string(),
            "checkpoint manifest has invalid annealed actor-learner pipeline"
        );
        drop(state);
        assert_invalid_capture_restore(&source, &trainer, &target, &loaded);
        fs::remove_dir_all(directory).expect("cleanup");
    }
}

fn assert_invalid_capture_restore(
    source: &PolicyModel,
    trainer: &PpoTrainer,
    target: &PolicyModel,
    loaded: &TrainingArtifact,
) {
    for (mastery_config, mastery_progress) in
        [(false, false), (true, true), (true, false), (false, true)]
    {
        let mut invalid = loaded.clone();
        let field = if mastery_config || mastery_progress {
            invalid.run.mastery_config = mastery_config.then(crate::MasteryConfig::default);
            invalid.progress.mastery = mastery_progress.then(crate::MasteryProgress::default);
            "annealed mastery"
        } else {
            invalid.progress.rollout_samples = 1;
            "rollout sample counter"
        };
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
        let decode_field = if mastery_progress && !mastery_config {
            "mastery configuration/state mismatch"
        } else {
            field
        };
        assert_eq!(
            decode_manifest(&bytes).expect_err("invalid decode"),
            CheckpointError::InvalidManifest(decode_field)
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
}

#[test]
fn runtime_profiles_roundtrip_and_reject_mixed_schema_without_mutation() {
    let directory = test_directory("runtime");
    let source = PolicyModel::fresh(40_012).expect("source");
    let model = PolicyModel::fresh(40_014).expect("target");
    let parameters = source.export_parameters().expect("parameters");
    let path = directory.join(RUNTIME_TENSOR_FILE);
    TrainingArtifact::save_runtime_weights(&source, &directory).expect("default export");
    let standard = fs::read(&path).expect("standard");
    for (budget, other) in [
        (PpoSampleBudget::Standard, PpoSampleBudget::Annealed),
        (PpoSampleBudget::Annealed, PpoSampleBudget::Standard),
        (PpoSampleBudget::Standard, PpoSampleBudget::WideAnnealed),
        (PpoSampleBudget::WideAnnealed, PpoSampleBudget::Standard),
        (PpoSampleBudget::Annealed, PpoSampleBudget::WideAnnealed),
        (PpoSampleBudget::WideAnnealed, PpoSampleBudget::Annealed),
    ] {
        TrainingArtifact::save_runtime_weights_with_budget(&source, &directory, budget)
            .expect("export");
        let bytes = fs::read(&path).expect("runtime");
        if budget == PpoSampleBudget::Standard {
            assert_eq!(bytes, standard);
        }
        TrainingArtifact::load_runtime_weights(&model, &directory).expect("import");
        assert_eq!(model.export_parameters().expect("parameters"), parameters);
        let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
        let metadata = metadata.metadata().as_ref().expect("map");
        for change in [
            None,
            Some(("ppo_schema_version", other.schema_version().to_string())),
            Some(("ppo_schema_hash", other.schema_hash().to_string())),
        ] {
            let invalid = change.is_some();
            let mut mixed = metadata.clone();
            if let Some((key, value)) = change {
                mixed.insert(key.to_owned(), value);
            }
            let data = encode_f32(&parameters);
            let tensor =
                TensorView::new(Dtype::F32, vec![parameters.len()], &data).expect("tensor");
            fs::write(
                &path,
                serialize([("model.parameters", tensor)], Some(mixed)).expect("fixture"),
            )
            .expect("write");
            let identity = model.policy_identity().expect("identity");
            let result = TrainingArtifact::load_runtime_weights(&model, &directory);
            if !invalid {
                result.expect("valid unordered metadata");
            } else {
                let error = result.expect_err("mixed import");
                assert_eq!(error, CheckpointError::SchemaMismatch);
                assert_eq!(
                    error.to_string(),
                    "checkpoint schema does not match this build"
                );
                assert_eq!(model.policy_identity().expect("identity"), identity);
            }
            assert_eq!(model.export_parameters().expect("parameters"), parameters);
        }
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
        fs::write(
            directory.join(CHECKPOINT_META_FILE),
            encoded_manifest(PpoSampleBudget::Standard, 0, 0),
        )
        .expect("manifest");
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

pub(super) fn manifest_artifact(
    budget: PpoSampleBudget,
    updates: u64,
    samples: u64,
) -> TrainingArtifact {
    TrainingArtifact {
        run: CheckpointRun {
            mastery_config: None,
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
            mastery: None,
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
            sample_budget: budget,
            environments: match budget {
                PpoSampleBudget::Standard => 2,
                PpoSampleBudget::Annealed => 40,
                PpoSampleBudget::WideAnnealed => 80,
            },
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

fn encoded_manifest(budget: PpoSampleBudget, updates: u64, samples: u64) -> Vec<u8> {
    let artifact = manifest_artifact(budget, updates, samples);
    encode_manifest(&artifact, artifact.tensor_hash).expect("manifest fixture")
}
