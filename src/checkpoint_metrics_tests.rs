use super::*;

#[cfg(feature = "cuda")]
#[test]
#[ignore = "read-only accepted U428 qualification through the exclusive CUDA runner"]
fn accepted_u428_restores_exact_model_adam_and_rng_without_migration() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("DRYSUA_ACCEPTED_CHECKPOINT").expect("accepted checkpoint"),
    );
    let artifact = TrainingArtifact::load(&directory).expect("accepted artifact");
    assert_eq!(artifact.progress.global_update, 428);
    assert_eq!(artifact.progress.rollout_samples, 8_613_414);
    assert_eq!(artifact.optimizer.step, 17_696);
    assert_eq!(artifact.shuffle, (6_834_666_388_189_899_400, 34_451_944));
    assert_eq!(
        artifact.progress.rng_states,
        vec![RngCheckpoint::new("ppo_actor_sampling", 13_685_406_966_101_999_287, 17_120).unwrap()]
    );
    let model = PolicyModel::fresh_on(1, crate::PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    let (trainer, run, progress) = artifact
        .restore(&model, artifact.run())
        .expect("original scope restore")
        .into_parts();
    let captured = TrainingArtifact::capture(&model, &trainer, run, progress).unwrap();
    assert_eq!(
        serialize_training_tensors(&captured).unwrap(),
        serialize_training_tensors(&artifact).unwrap()
    );
    assert_eq!(captured.shuffle, artifact.shuffle);
    assert_eq!(captured.progress, artifact.progress);
    assert_eq!(captured.optimizer.step, artifact.optimizer.step);
    let runtime = PolicyModel::fresh_on(2, crate::PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    TrainingArtifact::load_runtime_weights(&runtime, &directory).unwrap();
    assert!(
        runtime
            .export_parameters()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .eq(artifact.parameters.into_iter().map(f32::to_bits))
    );
}

#[test]
fn checkpoint_profiles_bind_exact_durable_bytes_while_scope_survives_progress() {
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let mut artifact = manifest_artifact();
        artifact.config.sample_budget = budget;
        let scope = metrics_scope_identity(&artifact.run, artifact.config).unwrap();
        let original = encode_manifest(&artifact, artifact.tensor_hash).unwrap();
        artifact.progress.rng_states[0].state += 1;
        artifact.optimizer.step += 1;
        artifact.shuffle.0 += 1;
        artifact.progress.global_update += 1;
        artifact.trainer_updates += 1;
        let changed = encode_manifest(&artifact, [43; 32]).unwrap();
        assert_eq!(
            metrics_scope_identity(&artifact.run, artifact.config).unwrap(),
            scope
        );
        assert_ne!(
            metrics_manifest_identity(&original).unwrap(),
            metrics_manifest_identity(&changed).unwrap()
        );
        artifact.run.run_seed += 1;
        assert_ne!(
            metrics_scope_identity(&artifact.run, artifact.config).unwrap(),
            scope
        );
        for bytes in [original, changed] {
            let mut synced = false;
            let identity = metrics_manifest_identity_after_sync(&bytes, || {
                synced = true;
                Ok(())
            })
            .unwrap();
            assert!(synced);
            assert_eq!(identity, sha256(&bytes));
            let restored = decode_manifest(&bytes).unwrap();
            assert_eq!(
                encode_manifest(&restored, restored.tensor_hash).unwrap(),
                bytes
            );
        }
    }
}

#[test]
fn invalid_manifest_or_failed_sync_never_returns_a_durable_identity() {
    let artifact = manifest_artifact();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).unwrap();
    for invalid in [false, true] {
        let mut bytes = encoded.clone();
        if invalid {
            bytes[0] ^= 1;
        }
        let mut calls = 0;
        let error = metrics_manifest_identity_after_sync(&bytes, || {
            calls += 1;
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "metrics checkpoint directory fsync failed",
            )
            .into())
        })
        .unwrap_err();
        assert_eq!(calls, usize::from(!invalid));
        assert_eq!(
            error.to_string(),
            if invalid {
                "checkpoint manifest magic is invalid"
            } else {
                "checkpoint I/O failed: metrics checkpoint directory fsync failed"
            }
        );
    }
}

#[test]
fn metrics_manifest_identity_hashes_input_bytes_without_reencoding() {
    let artifact = manifest_artifact();
    let mut encoded = encode_manifest(&artifact, artifact.tensor_hash).unwrap();
    let mut reader = ManifestReader::new(&encoded);
    reader.take(8).unwrap();
    let budget = decode_checkpoint_identity(&mut reader).unwrap();
    decode_schema(&mut reader, budget).unwrap();
    decode_run(&mut reader).unwrap();
    reader.take(4 * 8 + 4 + 1).unwrap();
    let offset = reader.offset;
    // Accepted noncanonical zero must bind to the bytes actually saved, not a re-encoding.
    encoded[offset..offset + 8].copy_from_slice(&(-0.0f64).to_le_bytes());
    let decoded = decode_manifest(&encoded).unwrap();
    let canonical = encode_manifest(&decoded, decoded.tensor_hash).unwrap();
    let identity = metrics_manifest_identity(&encoded).unwrap();
    assert_eq!(identity, sha256(&encoded));
    assert_ne!(identity, sha256(&canonical));
}

fn manifest_artifact() -> TrainingArtifact {
    TrainingArtifact {
        run: CheckpointRun {
            mastery_config: None,
            git_commit: "metrics-drysua".to_owned(),
            simulator_commit: "metrics-simulator".to_owned(),
            enabled_features: compiled_features(),
            command_line: "train metrics-fixture --retain-every 1".to_owned(),
            run_seed: 40_008,
            map: MapId(2),
            hero: SHADOW_FIEND,
            device: CheckpointDevice::Cpu,
            batch_size: 512,
            rules_audit_version: PPO_RULES_AUDIT_VERSION,
        },
        progress: CheckpointProgress {
            mastery: None,
            global_update: 3,
            policy_version: 3,
            scheduler_step: 3,
            curriculum_stage: 0,
            rollout_samples: 2_326,
            best_evaluation: None,
            rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 8, 40).unwrap()],
            league_references: Vec::new(),
        },
        config: PpoConfig {
            decision_interval_ticks: 3,
            rollout_decisions: 1_163,
            environments: 2,
            epochs: 1,
            minibatch: 512,
            gamma_tick: 1.0,
            ..PpoConfig::default()
        },
        trainer_updates: 3,
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
