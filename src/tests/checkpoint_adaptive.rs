use super::*;
use crate::ppo::test_directory;
use crate::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
    EnvironmentDecimal,
};

const BLOCK_BYTES: usize = 152;

#[test]
fn adaptive_manifest_roundtrips_fresh_active_transitioned_and_terminal_state() {
    let mut artifact = fixture();
    for update in 0..=8 {
        if update > 0 {
            observe(&mut artifact, update, 2);
        }
        let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
        let decoded = decode_manifest(&encoded).expect("decode");
        assert_eq!(&encoded[8..12], &CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
        assert_eq!(decoded.progress, artifact.progress);
        assert_eq!(decoded.run, artifact.run);
        assert_eq!(decoded.config, artifact.config);
        assert_eq!(decoded.shuffle, artifact.shuffle);
        assert_eq!(decoded.tensor_hash, artifact.tensor_hash);
        let checkpoint = decoded.progress.adaptive_environment.expect("adaptive");
        assert_eq!(checkpoint.snapshot_count, update.div_ceil(2));
    }
}

#[test]
fn adaptive_block_pins_exact_units_field_order_and_hash_bytes() {
    let mut artifact = fixture();
    observe(&mut artifact, 1, 0);
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
    let block = &encoded[encoded.len() - BLOCK_BYTES..];
    let words = [
        2u64, 800_000, 1, 200_000, 750_000, 3, 8, 2, 0, 0, 1, 0, 1, 1, 1,
    ];
    let mut golden = Vec::with_capacity(BLOCK_BYTES);
    for word in words {
        golden.extend_from_slice(&word.to_le_bytes());
    }
    golden.extend_from_slice(&[73; 32]);
    assert_eq!(block, golden);
    assert_eq!(
        artifact
            .progress
            .adaptive_environment
            .expect("adaptive")
            .config
            .scope_suffix(),
        " --environment-schedule adaptive --environment-success-updates 2 --environment-success-rate 0.8 --environment-poor-updates 1 --environment-poor-rate 0.2 --environment-extension 0.75"
    );
}

#[test]
fn adaptive_nondefault_six_decimal_config_roundtrips_without_float_conversion() {
    let mut artifact = fixture();
    let checkpoint = artifact
        .progress
        .adaptive_environment
        .as_mut()
        .expect("adaptive");
    let previous_suffix = checkpoint.config.scope_suffix();
    checkpoint.config = AdaptiveEnvironmentConfig {
        success_updates: 7,
        success_rate: EnvironmentDecimal::from_units(999_999),
        poor_updates: 3,
        poor_rate: EnvironmentDecimal::from_units(1),
        extension: EnvironmentDecimal::from_units(1_234_567),
    };
    artifact.run.command_line = artifact
        .run
        .command_line
        .replace(&previous_suffix, &checkpoint.config.scope_suffix());
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
    assert_eq!(
        decode_manifest(&encoded)
            .expect("decode")
            .progress
            .adaptive_environment,
        artifact.progress.adaptive_environment
    );
}

#[test]
fn adaptive_headers_reject_corruption_missing_block_and_trailing_bytes() {
    let artifact = fixture();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
    for end in encoded.len() - BLOCK_BYTES..encoded.len() {
        assert_eq!(
            decode_manifest(&encoded[..end]).expect_err("truncation"),
            CheckpointError::ManifestTruncated
        );
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert_eq!(
        decode_manifest(&trailing).expect_err("trailing"),
        CheckpointError::ManifestTrailingBytes
    );
    for offset in [8, 12, 56, 60] {
        let mut corrupt = encoded.clone();
        corrupt[offset] ^= 0x80;
        assert_eq!(
            decode_manifest(&corrupt).expect_err("header"),
            CheckpointError::SchemaMismatch
        );
    }
    let mut missing = encoded[..encoded.len() - BLOCK_BYTES].to_vec();
    *missing.last_mut().expect("presence byte") = 0;
    assert_manifest_error(
        &missing,
        "adaptive environment configuration/state mismatch",
    );
}

#[test]
fn adaptive_config_and_limits_are_bound_to_all_canonical_scope_values() {
    let artifact = fixture();
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
    for (word, value, field) in [
        (0, 3, "adaptive environment scope suffix"),
        (1, 799_999, "adaptive environment scope suffix"),
        (2, 2, "adaptive environment scope suffix"),
        (3, 199_999, "adaptive environment scope suffix"),
        (4, 749_999, "adaptive environment scope suffix"),
        (5, 4, "adaptive environment generation-updates scope"),
        (6, 9, "adaptive environment updates scope"),
        (7, 1, "adaptive environment zero-updates scope"),
    ] {
        let mut corrupt = encoded.clone();
        let offset = encoded.len() - BLOCK_BYTES + word * 8;
        corrupt[offset..offset + 8].copy_from_slice(&u64::to_le_bytes(value));
        assert_manifest_error(&corrupt, field);
    }
}

#[test]
fn adaptive_scope_rejects_missing_duplicate_unbounded_or_noncanonical_tokens() {
    for (source, replacement, field) in [
        ("train-annealed", "train", "adaptive environment command"),
        (
            "--updates 8",
            "--updates 08",
            "adaptive environment updates scope",
        ),
        (
            "--updates 8",
            "--updates +8",
            "adaptive environment updates scope",
        ),
        (
            "--updates 8",
            "--updates 18446744073709551616",
            "adaptive environment updates scope",
        ),
        (
            "--updates 8",
            "--updates 8 --updates 8",
            "adaptive environment updates scope",
        ),
        (
            "--updates 8",
            "--updates=8",
            "adaptive environment updates scope",
        ),
        ("--updates 8 ", "", "adaptive environment updates scope"),
        (
            "--zero-updates 2",
            "--zero-updates 1",
            "adaptive environment zero-updates scope",
        ),
        (
            "--generation-updates 3",
            "--generation-updates 4",
            "adaptive environment generation-updates scope",
        ),
        (
            "--environment-success-rate 0.8",
            "--environment-success-rate .8",
            "adaptive environment scope suffix",
        ),
        (
            "--environment-extension 0.75",
            "--environment-extension 0.75 --seed 1",
            "adaptive environment scope suffix",
        ),
    ] {
        let mut artifact = fixture();
        artifact.run.command_line = artifact.run.command_line.replace(source, replacement);
        let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("fixture");
        assert_manifest_error(&encoded, field);
    }
}

#[test]
fn adaptive_scope_requires_state_and_rejects_state_without_marker_or_league() {
    for (case, field) in [
        (0, "adaptive environment configuration/state mismatch"),
        (1, "adaptive environment scope suffix"),
        (2, "adaptive environment league"),
        (3, "adaptive environment scope suffix"),
    ] {
        let mut artifact = fixture();
        match case {
            0 => artifact.progress.adaptive_environment = None,
            1 => artifact.run.command_line = "train-annealed --updates 8".to_owned(),
            2 => artifact.progress.league_references.push(1),
            3 => artifact
                .run
                .command_line
                .insert_str(14, " --environment-schedule adaptive"),
            _ => unreachable!(),
        }
        let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("fixture");
        assert_manifest_error(&encoded, field);
    }
}

#[test]
fn adaptive_state_and_snapshot_invariants_reject_before_tensor_io() {
    let directory = test_directory("adaptive-invalid-metadata");
    let mut artifact = fixture();
    observe(&mut artifact, 1, 0);
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("encode");
    for (word, value, field) in [
        (
            0,
            0,
            "environment success updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (1, 1_000_001, "environment success rate must be in [0, 1]"),
        (
            5,
            0,
            "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            8,
            1,
            "environment generation and start update are inconsistent",
        ),
        (
            9,
            1,
            "environment start update plus spent updates must equal global update",
        ),
        (
            10,
            0,
            "environment start update plus spent updates must equal global update",
        ),
        (
            11,
            3,
            "environment success streak exceeds its window or spent updates",
        ),
        (
            12,
            2,
            "environment poor streak exceeds its window or spent updates",
        ),
        (
            13,
            2,
            "environment extension awards exceed qualifying updates",
        ),
        (14, 0, "adaptive environment snapshot count"),
        (14, u64::MAX, "adaptive environment snapshot count"),
    ] {
        let mut corrupt = encoded.clone();
        let offset = encoded.len() - BLOCK_BYTES + word * 8;
        corrupt[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        fs::write(directory.join(CHECKPOINT_META_FILE), &corrupt).expect("metadata only");
        let error = TrainingArtifact::load(&directory).expect_err("before missing tensors");
        assert_eq!(error, CheckpointError::InvalidManifest(field));
        assert_eq!(
            error.to_string(),
            format!("checkpoint manifest has invalid {field}")
        );
    }
}

#[test]
fn adaptive_snapshot_hash_is_zero_exactly_at_zero_count() {
    for update in [0, 1] {
        let mut artifact = fixture();
        if update > 0 {
            observe(&mut artifact, update, 0);
        }
        let checkpoint = artifact
            .progress
            .adaptive_environment
            .as_mut()
            .expect("adaptive");
        checkpoint.snapshot_hash = if update == 0 { [1; 32] } else { [0; 32] };
        let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("fixture");
        assert_manifest_error(&encoded, "adaptive environment snapshot hash");
    }
}

#[test]
fn adaptive_capture_save_restore_carries_model_adam_rng_and_rejects_without_mutation() {
    let directory = test_directory("adaptive-public-roundtrip");
    let fixture = fixture();
    let source = PolicyModel::fresh(51_001).expect("source");
    let trainer = PpoTrainer::new(&source, fixture.config, 91).expect("trainer");
    let artifact = TrainingArtifact::capture(
        &source,
        &trainer,
        fixture.run,
        fixture.progress,
        crate::checkpoint::collection_fixture(&source),
    )
    .expect("capture");
    artifact.save(&directory).expect("save");
    let loaded = TrainingArtifact::load_compatible(&directory, artifact.run()).expect("load");
    let target = PolicyModel::fresh(51_002).expect("target");
    let before = target.export_parameters().expect("before");
    let identity = target.policy_identity().expect("identity");
    for missing in [false, true] {
        let mut invalid = loaded.clone();
        let field = if missing {
            invalid.progress.adaptive_environment = None;
            "adaptive environment configuration/state mismatch"
        } else {
            invalid
                .progress
                .adaptive_environment
                .as_mut()
                .expect("adaptive")
                .snapshot_count = 1;
            "adaptive environment snapshot count"
        };
        let error = invalid
            .restore(&target, artifact.run())
            .err()
            .expect("reject");
        assert_eq!(error, CheckpointError::InvalidManifest(field));
        assert_eq!(target.export_parameters().expect("after"), before);
        assert_eq!(target.policy_identity().expect("identity after"), identity);
        assert_eq!(
            TrainingArtifact::capture(
                &source,
                &trainer,
                invalid.run,
                invalid.progress,
                crate::checkpoint::collection_fixture(&source)
            )
            .expect_err("capture rejects"),
            error
        );
    }
    let restored = loaded.restore(&target, artifact.run()).expect("restore");
    assert_eq!(restored.progress(), artifact.progress());
    assert_eq!(
        restored.trainer().rng_checkpoint(),
        trainer.rng_checkpoint()
    );
    assert_eq!(
        restored.trainer().optimizer_step(),
        trainer.optimizer_step()
    );
    let snapshot = restored
        .trainer()
        .checkpoint_snapshot(&target)
        .expect("snapshot");
    assert_eq!(snapshot.parameters, artifact.parameters);
    assert_eq!(snapshot.adam.moments().0, artifact.optimizer.first_moment);
    assert_eq!(snapshot.adam.moments().1, artifact.optimizer.second_moment);
}

#[test]
fn manifest_codec_rejects_global_metadata_over_64_kib() {
    let mut artifact = fixture();
    artifact.progress.rng_states = (0..32)
        .map(|index| {
            RngCheckpoint::new(format!("{index:02}{}", "x".repeat(4094)), 0, 0).expect("RNG")
        })
        .collect();
    assert_eq!(
        encode_manifest(&artifact, artifact.tensor_hash).expect_err("bounded encode"),
        CheckpointError::InvalidManifest("manifest size")
    );
    assert_eq!(
        decode_manifest(&vec![0; 65_537]).expect_err("bounded decode"),
        CheckpointError::InvalidManifest("manifest size")
    );
}

fn assert_manifest_error(bytes: &[u8], field: &'static str) {
    let error = decode_manifest(bytes).expect_err(field);
    assert_eq!(error, CheckpointError::InvalidManifest(field));
    assert_eq!(
        error.to_string(),
        format!("checkpoint manifest has invalid {field}")
    );
}

fn observe(artifact: &mut TrainingArtifact, update: u64, wins: u64) {
    let checkpoint = artifact
        .progress
        .adaptive_environment
        .as_mut()
        .expect("adaptive");
    checkpoint.state = checkpoint
        .state
        .observe(checkpoint.config, checkpoint.limits, update, wins, 2)
        .expect("controller update");
    checkpoint.snapshot_count =
        checkpoint.state.generation + u64::from(checkpoint.state.updates_in_generation > 0);
    checkpoint.snapshot_hash = [73; 32];
    artifact.progress.global_update = update;
    artifact.progress.policy_version = update;
    artifact.progress.scheduler_step = update;
    artifact.trainer_updates = update;
}

fn fixture() -> TrainingArtifact {
    let mut artifact = capacity_tests::manifest_artifact(0, 0);
    let config = AdaptiveEnvironmentConfig::default();
    artifact.run.command_line = format!(
        "train-annealed --updates 8 --generation-updates 3 --zero-updates 2{}",
        config.scope_suffix()
    );
    artifact.progress.adaptive_environment = Some(AdaptiveEnvironmentCheckpoint {
        config,
        limits: AdaptiveEnvironmentLimits {
            base_updates: 3,
            total_updates: 8,
            zero_updates: 2,
        },
        state: AdaptiveEnvironmentState::default(),
        snapshot_count: 0,
        snapshot_hash: [0; 32],
    });
    artifact
}
