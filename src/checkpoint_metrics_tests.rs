use super::*;

type InvalidFieldCase<T> = (fn(&mut T), &'static str);

#[test]
fn metrics_scope_is_deterministic_and_uses_domain_separated_canonical_fields() {
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let mut artifact = manifest_artifact();
        artifact.config.sample_budget = budget;
        let mut writer = ManifestWriter::default();
        writer.bytes.extend(b"drysua-metrics-scope/v1");
        let (version, hash) = checkpoint_schema_identity(budget);
        writer.u32(version);
        writer.u64(hash);
        encode_schema(&mut writer, budget);
        encode_run(&mut writer, &artifact.run).expect("canonical run");
        encode_config(&mut writer, artifact.config).expect("canonical config");

        let identity = metrics_scope_identity(&artifact.run, artifact.config).expect("scope");
        let repeated =
            metrics_scope_identity(&artifact.run.clone(), artifact.config).expect("same scope");

        assert_eq!(identity, sha256(&writer.bytes));
        assert_eq!(repeated, identity);
    }
}

#[test]
fn metrics_scope_changes_with_run_provenance_seed_device_and_command_flags() {
    let artifact = manifest_artifact();
    let original = metrics_scope_identity(&artifact.run, artifact.config).expect("scope");
    let changes: [fn(&mut CheckpointRun); 9] = [
        |run| run.git_commit.push_str("-other"),
        |run| run.simulator_commit.push_str("-other"),
        |run| run.command_line.push_str(" --retain-every 2"),
        |run| run.run_seed = 0,
        |run| run.run_seed = u64::MAX,
        |run| run.device = CheckpointDevice::Cuda { ordinal: 0 },
        |run| run.device = CheckpointDevice::Cuda { ordinal: 1 },
        |run| run.device = CheckpointDevice::Metal { ordinal: 0 },
        |run| run.mastery_config = Some(crate::MasteryConfig::default()),
    ];

    for (index, change) in changes.into_iter().enumerate() {
        let mut run = artifact.run.clone();
        change(&mut run);

        let identity = metrics_scope_identity(&run, artifact.config).expect("changed scope");

        assert_ne!(identity, original, "run field variant {index}");
    }
}

#[test]
fn metrics_scope_includes_each_mastery_setting_without_command_line_changes() {
    let mut artifact = manifest_artifact();
    artifact.run.mastery_config = Some(crate::MasteryConfig::default());
    let original = metrics_scope_identity(&artifact.run, artifact.config).expect("scope");

    for (window, weak, teacher) in [(51, 80, 80), (50, 81, 80), (50, 80, 81)] {
        let mut run = artifact.run.clone();
        run.mastery_config =
            Some(crate::MasteryConfig::from_resolved(window, weak, teacher).expect("mastery"));

        let identity = metrics_scope_identity(&run, artifact.config).expect("changed mastery");

        assert_ne!(identity, original, "mastery {window}/{weak}/{teacher}");
        assert_eq!(run.command_line, artifact.run.command_line);
    }
}

#[test]
fn metrics_scope_includes_ppo_values_and_profile_without_command_line_changes() {
    let artifact = manifest_artifact();
    let original = metrics_scope_identity(&artifact.run, artifact.config).expect("scope");
    let changes: [fn(&mut PpoConfig); 16] = [
        |config| config.sample_budget = PpoSampleBudget::Annealed,
        |config| config.decision_interval_ticks = 4,
        |config| config.rollout_decisions = 1_164,
        |config| config.environments = 4,
        |config| config.epochs = 2,
        |config| config.minibatch = 256,
        |config| config.clip_epsilon = 0.3,
        |config| config.value_coefficient = 0.6,
        |config| config.entropy_coefficient = 0.02,
        |config| config.learning_rate = f32::from_bits(config.learning_rate.to_bits() + 1),
        |config| config.adam_beta1 = 0.8,
        |config| config.adam_beta2 = 0.99,
        |config| config.adam_epsilon = 2.0e-5,
        |config| config.gradient_clip = 0.6,
        |config| config.gae_lambda = 0.99,
        |config| config.target_kl = 0.03,
    ];

    for (index, change) in changes.into_iter().enumerate() {
        let mut config = artifact.config;
        change(&mut config);
        let mut run = artifact.run.clone();
        run.batch_size = config.minibatch;

        let identity = metrics_scope_identity(&run, config).expect("changed PPO config");

        assert_ne!(identity, original, "PPO field variant {index}");
        assert_eq!(run.command_line, artifact.run.command_line);
    }
}

#[test]
fn metrics_scope_rejects_invalid_run_fields_instead_of_hashing_them() {
    let artifact = manifest_artifact();
    let changes: [InvalidFieldCase<CheckpointRun>; 7] = [
        (|run| run.git_commit.clear(), "git commit"),
        (|run| run.simulator_commit.clear(), "simulator commit"),
        (|run| run.command_line.push('\n'), "command line"),
        (
            |run| run.enabled_features = "other".to_owned(),
            "enabled features",
        ),
        (|run| run.map = MapId(0), "hero or map scope"),
        (
            |run| run.rules_audit_version = 0,
            "rules audit or batch size",
        ),
        (|run| run.batch_size = 0, "rules audit or batch size"),
    ];

    for (change, field) in changes {
        let mut run = artifact.run.clone();
        change(&mut run);

        let error = metrics_scope_identity(&run, artifact.config).expect_err("invalid scope");

        assert_eq!(error, CheckpointError::InvalidManifest(field));
        assert_eq!(
            error.to_string(),
            format!("checkpoint manifest has invalid {field}")
        );
    }
}

#[test]
fn metrics_scope_accepts_bounded_text_and_rejects_one_byte_more() {
    let mut artifact = manifest_artifact();
    artifact.run.git_commit = "g".repeat(MAX_TEXT_BYTES);

    let accepted = metrics_scope_identity(&artifact.run, artifact.config);
    artifact.run.git_commit.push('g');
    let error = metrics_scope_identity(&artifact.run, artifact.config).expect_err("oversize");

    assert!(accepted.is_ok());
    assert_eq!(error, CheckpointError::InvalidManifest("git commit"));
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid git commit"
    );
}

#[test]
fn metrics_scope_rejects_invalid_ppo_values_and_wrong_map2_discount() {
    let artifact = manifest_artifact();
    let changes: [InvalidFieldCase<PpoConfig>; 3] = [
        (|config| config.learning_rate = f32::NAN, "PPO config"),
        (|config| config.minibatch = 0, "PPO config"),
        (|config| config.gamma_tick = 0.99, "Map2 reward discount"),
    ];

    for (change, field) in changes {
        let mut config = artifact.config;
        change(&mut config);

        let error = metrics_scope_identity(&artifact.run, config).expect_err("invalid config");

        assert_eq!(error, CheckpointError::InvalidManifest(field));
        assert_eq!(
            error.to_string(),
            format!("checkpoint manifest has invalid {field}")
        );
    }
}

#[test]
fn metrics_scope_rejects_annealed_mastery() {
    let mut artifact = manifest_artifact();
    artifact.run.mastery_config = Some(crate::MasteryConfig::default());
    artifact.config.sample_budget = PpoSampleBudget::Annealed;

    let error = metrics_scope_identity(&artifact.run, artifact.config).expect_err("mastery");

    assert_eq!(error, CheckpointError::InvalidManifest("annealed mastery"));
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid annealed mastery"
    );
}

#[test]
fn metrics_manifest_identity_syncs_once_before_returning_the_exact_hash() {
    let artifact = manifest_artifact();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let mut sync_calls = 0;

    let identity = metrics_manifest_identity_after_sync(&encoded, || {
        sync_calls += 1;
        Ok(())
    })
    .expect("durable manifest identity");

    assert_eq!(sync_calls, 1);
    assert_eq!(identity, sha256(&encoded));
}

#[test]
fn metrics_manifest_identity_returns_sync_failure_instead_of_an_identity() {
    let artifact = manifest_artifact();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let mut sync_calls = 0;

    let error = metrics_manifest_identity_after_sync(&encoded, || {
        sync_calls += 1;
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "metrics checkpoint directory fsync failed",
        )
        .into())
    })
    .expect_err("sync failure must prevent returning an identity");

    assert_eq!(sync_calls, 1);
    assert_eq!(
        error,
        CheckpointError::Io("metrics checkpoint directory fsync failed".to_owned())
    );
    assert_eq!(
        error.to_string(),
        "checkpoint I/O failed: metrics checkpoint directory fsync failed"
    );
}

#[test]
fn metrics_manifest_identity_rejects_invalid_metadata_before_sync() {
    let artifact = manifest_artifact();
    let mut encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    encoded[0] ^= 1;
    let mut sync_calls = 0;

    let error = metrics_manifest_identity_after_sync(&encoded, || {
        sync_calls += 1;
        Ok(())
    })
    .expect_err("invalid manifest");

    assert_eq!(error, CheckpointError::ManifestMagic);
    assert_eq!(error.to_string(), "checkpoint manifest magic is invalid");
    assert_eq!(sync_calls, 0);
}

#[test]
fn metrics_manifest_identity_is_sha256_of_exact_manifest_not_tensor_hash() {
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let mut artifact = manifest_artifact();
        artifact.config.sample_budget = budget;
        let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");

        let identity = metrics_manifest_identity(&encoded).expect("manifest identity");
        let decoded = decode_manifest(&encoded).expect("unchanged checkpoint format");
        let reencoded = encode_manifest(&decoded, decoded.tensor_hash).expect("roundtrip");

        assert_eq!(identity, <[u8; 32]>::from(Sha256::digest(&encoded)));
        assert_ne!(identity, artifact.tensor_hash);
        assert_eq!(reencoded, encoded);
    }
}

#[test]
fn metrics_manifest_identity_changes_with_progress_rng_and_optimizer_metadata() {
    let artifact = manifest_artifact();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let original = metrics_manifest_identity(&encoded).expect("identity");
    let scope = metrics_scope_identity(&artifact.run, artifact.config).expect("scope");
    let changes: [fn(&mut TrainingArtifact); 10] = [
        |artifact| {
            artifact.progress.global_update += 1;
            artifact.trainer_updates += 1;
        },
        |artifact| artifact.progress.scheduler_step += 1,
        |artifact| artifact.progress.curriculum_stage += 1,
        |artifact| artifact.progress.rollout_samples += 1,
        |artifact| artifact.progress.best_evaluation = Some(0.5),
        |artifact| artifact.progress.rng_states[0].state += 1,
        |artifact| artifact.progress.league_references.push(1),
        |artifact| artifact.shuffle.0 += 1,
        |artifact| artifact.shuffle.1 += 1,
        |artifact| artifact.optimizer.step += 1,
    ];

    for (index, change) in changes.into_iter().enumerate() {
        let mut changed = artifact.clone();
        change(&mut changed);
        let encoded = encode_manifest(&changed, changed.tensor_hash).expect("changed manifest");

        let identity = metrics_manifest_identity(&encoded).expect("changed identity");

        assert_ne!(identity, original, "metadata variant {index}");
        assert_eq!(
            metrics_scope_identity(&changed.run, changed.config).expect("scope"),
            scope
        );
    }
}

#[test]
fn metrics_manifest_identity_includes_the_existing_tensor_hash() {
    let artifact = manifest_artifact();
    let original = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let changed = encode_manifest(&artifact, [43; 32]).expect("other tensors");

    let original = metrics_manifest_identity(&original).expect("original identity");
    let changed = metrics_manifest_identity(&changed).expect("changed identity");

    assert_ne!(original, changed);
}

#[test]
fn metrics_manifest_identity_hashes_input_bytes_without_reencoding() {
    let artifact = manifest_artifact();
    let mut encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let mut reader = ManifestReader::new(&encoded);
    reader.take(8).expect("magic");
    let budget = decode_checkpoint_identity(&mut reader).expect("checkpoint schema");
    decode_schema(&mut reader, budget).expect("linked schemas");
    decode_run(&mut reader).expect("run");
    reader
        .take(4 * 8 + 4 + 1)
        .expect("progress before optional payload");
    let offset = reader.offset;
    // An absent evaluation accepts either sign of zero; re-encoding canonicalizes it.
    encoded[offset..offset + 8].copy_from_slice(&(-0.0f64).to_le_bytes());
    let decoded = decode_manifest(&encoded).expect("accepted negative zero");
    let canonical = encode_manifest(&decoded, decoded.tensor_hash).expect("canonical manifest");

    let identity = metrics_manifest_identity(&encoded).expect("exact identity");

    assert_eq!(identity, sha256(&encoded));
    assert_ne!(identity, sha256(&canonical));
}

#[test]
fn metrics_manifest_identity_rejects_truncation_and_trailing_bytes() {
    let artifact = manifest_artifact();
    let mut encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");

    for length in [0, 7, encoded.len() - 1] {
        let error = metrics_manifest_identity(&encoded[..length]).expect_err("truncated");

        assert_eq!(error, CheckpointError::ManifestTruncated);
        assert_eq!(error.to_string(), "checkpoint manifest is truncated");
    }
    encoded.push(0);

    let error = metrics_manifest_identity(&encoded).expect_err("trailing byte");

    assert_eq!(error, CheckpointError::ManifestTrailingBytes);
    assert_eq!(error.to_string(), "checkpoint manifest has trailing bytes");
}

#[test]
fn metrics_manifest_identity_rejects_magic_and_schema_mismatches() {
    let artifact = manifest_artifact();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");

    for (offset, expected) in [
        (0, CheckpointError::ManifestMagic),
        (8, CheckpointError::SchemaMismatch),
        (12, CheckpointError::SchemaMismatch),
        (20, CheckpointError::SchemaMismatch),
    ] {
        let mut changed = encoded.clone();
        changed[offset] ^= 1;

        let error = metrics_manifest_identity(&changed).expect_err("invalid header");

        assert_eq!(error, expected);
        let message = if offset == 0 {
            "checkpoint manifest magic is invalid"
        } else {
            "checkpoint schema does not match this build"
        };
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn metrics_manifest_identity_rejects_invalid_config_before_returning_a_hash() {
    let mut artifact = manifest_artifact();
    artifact.config.learning_rate = f32::NAN;
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("invalid fixture");

    let error = metrics_manifest_identity(&encoded).expect_err("invalid config");

    assert_eq!(error, CheckpointError::InvalidManifest("PPO config"));
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid PPO config"
    );
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
            rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 8, 40).expect("RNG")],
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
