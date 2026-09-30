use super::*;
use crate::adaptive_environment::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
};
use crate::randomization::{draw_generation, generation_json};

const SEED: u64 = 0x5eed_1234;
const GAMES: u64 = 8;

fn checkpoint() -> AdaptiveEnvironmentCheckpoint {
    AdaptiveEnvironmentCheckpoint {
        config: AdaptiveEnvironmentConfig::default(),
        limits: AdaptiveEnvironmentLimits {
            base_updates: 10,
            total_updates: 20,
            zero_updates: 4,
        },
        state: AdaptiveEnvironmentState::default(),
        snapshot_count: 0,
        snapshot_hash: [0; 32],
    }
}

fn recorded_prefix(directory: &Path) -> AdaptiveEnvironmentCheckpoint {
    let mut checkpoint = checkpoint();
    for (generation, start_update) in [0, 2, 5].into_iter().enumerate() {
        checkpoint.state = AdaptiveEnvironmentState {
            generation: generation as u64,
            start_update,
            ..AdaptiveEnvironmentState::default()
        };
        draw_adaptive_generation(
            directory,
            SEED,
            GAMES,
            &mut checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect("record prefix");
    }
    checkpoint
}

fn assert_config_error(error: PpoError, field: &str) {
    assert_eq!(
        error.to_string(),
        format!("invalid PPO config field: {field}")
    );
}

#[test]
fn canonical_valid_changed_historical_start_fails_prefix_hash() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let checkpoint = recorded_prefix(&directory);
    let changed = draw_generation_at_start(
        SEED,
        1,
        3,
        GAMES,
        schedule(&checkpoint, crate::randomization::AnnealScale::FULL),
    )
    .expect("different but valid historical start");
    let bytes = adaptive_generation_json(&changed);
    let path = generation_path(&directory, 1);
    std::fs::write(&path, &bytes).expect("replace with canonical alternate history");

    let error = verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("canonical validity cannot authenticate a changed start");

    assert_config_error(error, "adaptive randomization snapshot hash mismatch");
    assert_eq!(
        std::fs::read_to_string(path).expect("unchanged file"),
        bytes
    );
}

#[test]
fn missing_committed_snapshot_is_rejected_without_repair() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let checkpoint = recorded_prefix(&directory);
    let path = generation_path(&directory, 1);
    std::fs::remove_file(&path).expect("remove committed snapshot");

    let error = verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("missing prefix member");

    assert_config_error(error, "adaptive randomization snapshot is missing");
    assert!(!path.exists());
}

#[test]
fn later_orphan_is_ignored_then_verified_exactly_before_commit() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut pending = recorded_prefix(&directory);
    pending.state = AdaptiveEnvironmentState {
        generation: 3,
        start_update: 7,
        ..AdaptiveEnvironmentState::default()
    };
    let mut staged = pending;
    let expected = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut staged,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("failed collection leaves canonical orphan");
    let path = generation_path(&directory, 3);
    let bytes = std::fs::read(&path).expect("orphan bytes");
    std::fs::write(generation_path(&directory, 9), b"ignored later orphan")
        .expect("uncommitted later file");
    verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &pending,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("ignore all orphans");
    let mut replay = pending;

    let actual = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut replay,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("reuse exact orphan");

    assert_eq!(actual, expected);
    assert_eq!(replay, staged);
    assert_eq!(std::fs::read(path).expect("immutable bytes"), bytes);
}

#[test]
fn empty_pending_generation_verifies_only_completed_prefix() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = recorded_prefix(&directory);
    checkpoint.state = AdaptiveEnvironmentState {
        generation: 3,
        start_update: 7,
        ..AdaptiveEnvironmentState::default()
    };
    let saved = checkpoint;

    verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("pending prefix");

    assert_eq!(checkpoint, saved);
    assert!(!generation_path(&directory, 3).exists());
    assert_eq!(std::fs::read_dir(&directory).expect("directory").count(), 3);
}

#[test]
fn wrong_orphan_is_read_only_on_draw_error() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut staged = checkpoint();
    draw_adaptive_generation(
        &directory,
        SEED + 1,
        GAMES,
        &mut staged,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("other seed");
    let path = generation_path(&directory, 0);
    let bytes = std::fs::read(&path).expect("original orphan");
    let mut pending = checkpoint();
    let original = pending;
    verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &pending,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("ignore orphan");

    let error = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut pending,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("different orphan must not be replaced");

    assert_config_error(error, "adaptive randomization snapshot mismatch");
    assert_eq!(pending, original);
    assert_eq!(std::fs::read(path).expect("preserved orphan"), bytes);
}

#[test]
fn active_replay_preserves_fractional_awards_and_snapshot_commitment() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    let initial = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("first draw");
    for completed_update in 1..=3 {
        checkpoint.state = checkpoint
            .state
            .observe(
                checkpoint.config,
                checkpoint.limits,
                completed_update,
                0,
                GAMES,
            )
            .expect("fractional extension award");
    }
    assert_eq!(checkpoint.state.extension_awards, 3);
    assert_eq!(
        checkpoint
            .state
            .effective_budget(checkpoint.config, checkpoint.limits)
            .expect("floor of accumulated .75 credit"),
        12
    );
    let saved = checkpoint;

    verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("resume prefix");
    let replay = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("active replay");

    assert_eq!(replay, initial);
    assert_eq!(checkpoint, saved);
}

#[test]
fn real_start_controls_scale_and_global_boundary_forces_clean_draw() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    checkpoint.limits.total_updates = 10;
    checkpoint.limits.zero_updates = 2;
    draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("first");
    for completed_update in 1..=8 {
        let wins = if completed_update <= 2 { GAMES } else { 0 };
        checkpoint.state = checkpoint
            .state
            .observe(
                checkpoint.config,
                checkpoint.limits,
                completed_update,
                wins,
                GAMES,
            )
            .expect("controller update");
        if completed_update == 2 {
            let draw = draw_adaptive_generation(
                &directory,
                SEED,
                GAMES,
                &mut checkpoint,
                crate::randomization::AnnealScale::FULL,
            )
            .expect("early environment");
            assert_eq!(draw.start_update, 2);
            assert_eq!(draw.scale_bp, 5_000);
            assert_eq!(draw.start_game, 16);
            assert_eq!(draw.end_game, 80);
            assert_eq!(draw.applied_games, 48);
            let fixed = draw_generation(
                SEED,
                1,
                16,
                GAMES,
                schedule(&checkpoint, crate::randomization::AnnealScale::FULL),
            )
            .expect("same generation ordinal and actual start");
            assert_eq!(draw.deltas, fixed.deltas);
        }
    }
    assert_eq!(checkpoint.state.generation, 2);
    assert_eq!(checkpoint.state.start_update, 8);
    let state = checkpoint.state;

    let clean = draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("forced clean environment");

    assert_eq!(checkpoint.state, state);
    assert_eq!(clean.scale_bp, 0);
    assert_eq!(clean.applied_games, 0);
    assert!(clean.spec.is_nominal());
    verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("boundary prefix");
}

#[test]
fn adaptive_schema_labels_unknown_ends_as_bounds() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("draw");

    let bytes = std::fs::read(generation_path(&directory, 0)).expect("snapshot");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("canonical JSON");

    assert_eq!(json["schema"], "drysua-domain-randomization/adaptive-v3");
    assert_eq!(json["start_update"], 0);
    assert_eq!(json["end_game_bound"], 160);
    assert_eq!(json["applied_games_bound"], 128);
    assert!(json.get("end_game").is_none());
    assert!(json.get("applied_games").is_none());
    assert!(bytes.len() <= 4096);
    let mut hash = Sha256::new();
    hash.update([0; 32]);
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(&bytes);
    let expected: [u8; 32] = hash.finalize().into();
    assert_eq!(checkpoint.snapshot_hash, expected);
}

#[test]
fn rolling_commitment_includes_previous_digest_and_each_byte_length() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let checkpoint = recorded_prefix(&directory);
    let mut previous = [0; 32];
    for generation in 0..3 {
        let bytes =
            std::fs::read(generation_path(&directory, generation)).expect("canonical bytes");
        let framed = [
            previous.as_slice(),
            &(bytes.len() as u64).to_le_bytes(),
            &bytes,
        ]
        .concat();
        previous = Sha256::digest(&framed).into();
    }

    assert_eq!(checkpoint.snapshot_count, 3);
    assert_eq!(checkpoint.snapshot_hash, previous);
}

#[test]
fn fixed_draw_keeps_existing_golden_values_and_v2_hash() {
    let draw = draw_generation(
        SEED,
        3,
        4,
        GAMES,
        AnnealSchedule {
            updates: 10,
            zero_updates: 2,
            scale: crate::randomization::AnnealScale::FULL,
        },
    )
    .expect("legacy golden draw");

    assert_eq!(
        draw.deltas,
        [-2949, 0, 0, 3683, 386, 2325, 581, 0, 153, -1042, -941]
    );
    assert_eq!(
        (draw.start_game, draw.end_game, draw.start_update),
        (12, 16, 1)
    );
    assert_eq!((draw.scale_bp, draw.applied_games), (6465, 4));
    assert!(generation_json(&draw).ends_with(",\"hash\":\"fc8dbde7bc2fff27\"}\n"));
}

#[test]
fn oversized_and_noncanonical_bodies_are_rejected_without_rewrite() {
    for (bytes, field) in [
        (
            vec![b'x'; 4097],
            "adaptive randomization snapshot is oversized",
        ),
        (
            b"{}\n".to_vec(),
            "adaptive randomization snapshot start is invalid",
        ),
    ] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let mut checkpoint = checkpoint();
        draw_adaptive_generation(
            &directory,
            SEED,
            GAMES,
            &mut checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect("draw");
        let path = generation_path(&directory, 0);
        std::fs::write(&path, &bytes).expect("corrupt file");

        let error = verify_adaptive_snapshots(
            &directory,
            SEED,
            GAMES,
            &checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect_err("invalid snapshot");

        assert_config_error(error, field);
        assert_eq!(std::fs::read(path).expect("unchanged corruption"), bytes);
    }
}

#[test]
fn tampered_body_and_active_start_fail_without_changing_commitment() {
    for change_start in [false, true] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let mut checkpoint = recorded_prefix(&directory);
        let path = generation_path(&directory, 2);
        if change_start {
            checkpoint.state.start_update = 6;
        } else {
            let stored = std::fs::read_to_string(&path).expect("canonical snapshot");
            let changed = stored.replace("\"end_game_bound\":160", "\"end_game_bound\":159");
            assert_ne!(stored, changed);
            std::fs::write(&path, changed).expect("tampered body with valid start prefix");
        }
        let original = checkpoint;
        let bytes = std::fs::read(&path).expect("original bytes");

        let error = draw_adaptive_generation(
            &directory,
            SEED,
            GAMES,
            &mut checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect_err("active replay must compare against actual start and entire body");

        assert_config_error(error, "adaptive randomization snapshot mismatch");
        assert_eq!(checkpoint, original);
        assert_eq!(std::fs::read(path).expect("preserved bytes"), bytes);
    }
}

#[test]
fn fresh_draw_creates_directory_but_invalid_draw_does_not() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let child = directory.join("snapshots");
    let mut checkpoint = checkpoint();
    let error = draw_adaptive_generation(
        &child,
        SEED,
        0,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("invalid draw before directory creation");
    assert_config_error(
        error,
        "adaptive randomization games per update must be in 1..=MAX_TRAINING_COUNTER",
    );
    assert!(!child.exists());

    draw_adaptive_generation(
        &child,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("fresh directory");

    assert!(child.is_dir());
    verify_adaptive_snapshots(
        &child,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("committed first draw");
}

#[test]
fn pending_and_active_starts_must_follow_the_committed_prefix() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let original = recorded_prefix(&directory);
    for (generation, start_update, field) in [
        (2, 6, "adaptive randomization active start mismatch"),
        (
            3,
            5,
            "adaptive randomization pending start must follow snapshots",
        ),
    ] {
        let mut checkpoint = original;
        checkpoint.state.generation = generation;
        checkpoint.state.start_update = start_update;

        let error = verify_adaptive_snapshots(
            &directory,
            SEED,
            GAMES,
            &checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect_err("inconsistent start");

        assert_config_error(error, field);
    }
}

#[test]
fn first_start_and_historical_order_are_checked_before_hash() {
    for (generation, start_update) in [(0, 1), (1, 0), (1, 5)] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let checkpoint = recorded_prefix(&directory);
        let draw = draw_generation_at_start(
            SEED,
            generation,
            start_update,
            GAMES,
            schedule(&checkpoint, crate::randomization::AnnealScale::FULL),
        )
        .expect("canonical reordered draw");
        std::fs::write(
            generation_path(&directory, generation),
            adaptive_generation_json(&draw),
        )
        .expect("replace historical start");

        let error = verify_adaptive_snapshots(
            &directory,
            SEED,
            GAMES,
            &checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect_err("invalid start order");

        assert_config_error(
            error,
            "adaptive randomization snapshot starts are not strictly increasing from zero",
        );
    }
}

#[test]
fn empty_verification_does_not_create_directory_and_wrong_type_is_rejected() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let missing = directory.join("missing");
    verify_adaptive_snapshots(
        &missing,
        SEED,
        GAMES,
        &checkpoint(),
        crate::randomization::AnnealScale::FULL,
    )
    .expect("empty prefix");
    assert!(!missing.exists());
    let file = directory.join("file");
    std::fs::write(&file, b"not a directory").expect("wrong type");

    let error = verify_adaptive_snapshots(
        &file,
        SEED,
        GAMES,
        &checkpoint(),
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("even empty prefix rejects wrong directory type");

    assert_config_error(
        error,
        "adaptive randomization directory must be a real directory",
    );
}

#[test]
fn invalid_counts_and_games_fail_before_creating_files() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    for (count, games, field) in [
        (
            MAX_TRAINING_COUNTER + 1,
            GAMES,
            "adaptive randomization snapshot count is invalid",
        ),
        (
            0,
            0,
            "adaptive randomization games per update must be in 1..=MAX_TRAINING_COUNTER",
        ),
    ] {
        let mut checkpoint = checkpoint();
        checkpoint.snapshot_count = count;
        let original = checkpoint;

        let error = draw_adaptive_generation(
            &directory,
            SEED,
            games,
            &mut checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect_err("invalid inputs");

        assert_config_error(error, field);
        assert_eq!(checkpoint, original);
        assert_eq!(std::fs::read_dir(&directory).expect("directory").count(), 0);
    }
}

#[cfg(unix)]
#[test]
fn symlink_snapshots_and_directories_are_rejected_without_following() {
    use std::os::unix::fs::symlink;

    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    draw_adaptive_generation(
        &directory,
        SEED,
        GAMES,
        &mut checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect("draw");
    let path = generation_path(&directory, 0);
    let target = directory.join("target");
    std::fs::rename(&path, &target).expect("move snapshot");
    symlink(&target, &path).expect("symlink snapshot");

    let error = verify_adaptive_snapshots(
        &directory,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("snapshot symlink");
    assert_config_error(
        error,
        "adaptive randomization snapshot must be a regular file",
    );
    assert!(
        std::fs::symlink_metadata(&path)
            .expect("preserved link")
            .is_symlink()
    );

    let link = directory.join("directory-link");
    symlink(&directory, &link).expect("symlink directory");
    let error = verify_adaptive_snapshots(
        &link,
        SEED,
        GAMES,
        &checkpoint,
        crate::randomization::AnnealScale::FULL,
    )
    .expect_err("directory symlink");
    assert_config_error(
        error,
        "adaptive randomization directory must be a real directory",
    );
}
