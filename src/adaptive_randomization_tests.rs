use super::*;
use crate::adaptive_environment::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
};
use crate::randomization::{AnnealScale, draw_generation};

const SEED: u64 = 0x5eed_1234;
/// Games each update finishes, as the controller observes them.
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

fn pending(generation: u64, start_update: u64) -> AdaptiveEnvironmentState {
    AdaptiveEnvironmentState {
        generation,
        start_update,
        ..AdaptiveEnvironmentState::default()
    }
}

fn draw(
    directory: &Path,
    seed: u64,
    checkpoint: &mut AdaptiveEnvironmentCheckpoint,
) -> Result<GenerationDraw, PpoError> {
    draw_adaptive_generation(directory, seed, checkpoint, AnnealScale::FULL)
}

fn verify(directory: &Path, checkpoint: &AdaptiveEnvironmentCheckpoint) -> Result<(), PpoError> {
    verify_adaptive_snapshots(directory, SEED, checkpoint, AnnealScale::FULL)
}

/// Commits generations 0, 1 and 2 starting at updates 0, 2 and 5.
fn recorded_prefix(directory: &Path) -> AdaptiveEnvironmentCheckpoint {
    let mut checkpoint = checkpoint();
    for (generation, start_update) in [0, 2, 5].into_iter().enumerate() {
        checkpoint.state = pending(generation as u64, start_update);
        draw(directory, SEED, &mut checkpoint).expect("record prefix");
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
fn rewritten_historical_starts_fail_order_or_hash_checks_without_repair() {
    let order = "adaptive randomization snapshot starts are not strictly increasing from zero";
    for (generation, start_update, field) in [
        (1, 3, "adaptive randomization snapshot hash mismatch"),
        (0, 1, order),
        (1, 0, order),
        (1, 5, order),
    ] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let checkpoint = recorded_prefix(&directory);
        let schedule = schedule(&checkpoint, AnnealScale::FULL);
        let rewritten = draw_generation_at_start(SEED, generation, start_update, schedule)
            .expect("canonical draw at another start");
        let bytes = adaptive_generation_json(&rewritten);
        let path = generation_path(&directory, generation);
        std::fs::write(&path, &bytes).expect("rewrite history");

        let error = verify(&directory, &checkpoint).expect_err("rewritten history");

        assert_config_error(error, field);
        assert_eq!(std::fs::read_to_string(path).expect("unchanged"), bytes);
    }
}

#[test]
fn missing_oversized_or_unparseable_committed_snapshot_is_rejected_without_repair() {
    for (contents, field) in [
        (None, "adaptive randomization snapshot is missing"),
        (
            Some(vec![b'x'; MAX_SNAPSHOT_BYTES as usize + 1]),
            "adaptive randomization snapshot is oversized",
        ),
        (
            Some(b"{}\n".to_vec()),
            "adaptive randomization snapshot start is invalid",
        ),
    ] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let checkpoint = recorded_prefix(&directory);
        let path = generation_path(&directory, 1);
        match &contents {
            Some(bytes) => std::fs::write(&path, bytes).expect("corrupt snapshot"),
            None => std::fs::remove_file(&path).expect("remove snapshot"),
        }

        let error = verify(&directory, &checkpoint).expect_err("damaged prefix member");

        assert_config_error(error, field);
        assert_eq!(std::fs::read(&path).ok(), contents, "{field}");
    }
}

#[test]
fn later_orphan_is_ignored_then_verified_exactly_before_commit() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut pending_checkpoint = recorded_prefix(&directory);
    pending_checkpoint.state = pending(3, 7);
    let mut staged = pending_checkpoint;
    let expected = draw(&directory, SEED, &mut staged).expect("orphan of a failed collection");
    let path = generation_path(&directory, 3);
    let bytes = std::fs::read(&path).expect("orphan bytes");
    std::fs::write(generation_path(&directory, 9), b"ignored later orphan")
        .expect("uncommitted later file");
    verify(&directory, &pending_checkpoint).expect("ignore all orphans");
    let mut replay = pending_checkpoint;

    let actual = draw(&directory, SEED, &mut replay).expect("reuse exact orphan");

    assert_eq!(actual, expected);
    assert_eq!(replay, staged);
    assert_eq!(std::fs::read(path).expect("immutable bytes"), bytes);
}

#[test]
fn empty_pending_generation_verifies_only_completed_prefix() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = recorded_prefix(&directory);
    checkpoint.state = pending(3, 7);
    let saved = checkpoint;

    verify(&directory, &checkpoint).expect("pending prefix");

    assert_eq!(checkpoint, saved);
    assert!(!generation_path(&directory, 3).exists());
    assert_eq!(std::fs::read_dir(&directory).expect("directory").count(), 3);
}

#[test]
fn wrong_orphan_is_read_only_on_draw_error() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    draw(&directory, SEED + 1, &mut checkpoint()).expect("other seed");
    let path = generation_path(&directory, 0);
    let bytes = std::fs::read(&path).expect("original orphan");
    let mut pending = checkpoint();
    let original = pending;
    verify(&directory, &pending).expect("ignore orphan");

    let error = draw(&directory, SEED, &mut pending).expect_err("different orphan is kept");

    assert_config_error(error, "adaptive randomization snapshot mismatch");
    assert_eq!(pending, original);
    assert_eq!(std::fs::read(path).expect("preserved orphan"), bytes);
}

#[test]
fn active_replay_preserves_fractional_awards_and_snapshot_commitment() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    let initial = draw(&directory, SEED, &mut checkpoint).expect("first draw");
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
            .effective_budget(checkpoint.config, checkpoint.limits),
        Ok(12),
        "floor of three .75 awards over base ten"
    );
    let saved = checkpoint;

    verify(&directory, &checkpoint).expect("resume prefix");
    let replay = draw(&directory, SEED, &mut checkpoint).expect("active replay");

    assert_eq!(replay, initial);
    assert_eq!(checkpoint, saved);
}

/// Observes wins at updates 1-2 and losses through update 8; returns the draw made right after
/// the success at update 2.
fn early_success_then_boundary(
    directory: &Path,
    checkpoint: &mut AdaptiveEnvironmentCheckpoint,
) -> GenerationDraw {
    draw(directory, SEED, checkpoint).expect("first");
    let mut early = None;
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
            early = Some(draw(directory, SEED, checkpoint).expect("early environment"));
        }
    }
    early.expect("success at update two")
}

#[test]
fn real_start_controls_scale_and_global_boundary_forces_clean_draw() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    checkpoint.limits.total_updates = 10;
    checkpoint.limits.zero_updates = 2;
    let early = early_success_then_boundary(&directory, &mut checkpoint);
    assert_eq!(
        (early.start_update, early.scale_bp, early.start_game),
        (2, 5_000, 2)
    );
    assert_eq!((early.end_game, early.applied_games), (10, 6));
    let fixed = draw_generation(SEED, 1, 2, schedule(&checkpoint, AnnealScale::FULL))
        .expect("same generation ordinal and actual start");
    assert_eq!(early.deltas, fixed.deltas);
    assert_eq!(checkpoint.state, pending(2, 8));
    let state = checkpoint.state;

    let clean = draw(&directory, SEED, &mut checkpoint).expect("forced clean environment");

    assert_eq!(checkpoint.state, state);
    assert_eq!(clean.scale_bp, 0);
    assert_eq!(clean.applied_games, 0);
    assert!(clean.spec.is_nominal());
    verify(&directory, &checkpoint).expect("boundary prefix");
}

#[test]
fn snapshots_label_bounds_and_chain_a_length_framed_digest() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let checkpoint = recorded_prefix(&directory);
    let first = std::fs::read(generation_path(&directory, 0)).expect("snapshot");
    let json: serde_json::Value = serde_json::from_slice(&first).expect("canonical JSON");
    assert_eq!(json["schema"], "drysua-domain-randomization/adaptive-v3");
    assert_eq!(json["start_update"], 0);
    assert_eq!(json["end_game_bound"], 20);
    assert_eq!(json["applied_games_bound"], 16);
    assert!(json.get("end_game").is_none());
    assert!(json.get("applied_games").is_none());
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
fn tampered_body_and_active_start_fail_without_changing_commitment() {
    for change_start in [false, true] {
        let directory = crate::ppo::test_directory("adaptive-snapshots");
        let mut checkpoint = recorded_prefix(&directory);
        let path = generation_path(&directory, 2);
        if change_start {
            checkpoint.state.start_update = 6;
        } else {
            let stored = std::fs::read_to_string(&path).expect("canonical snapshot");
            let changed = stored.replace("\"end_game_bound\":20", "\"end_game_bound\":19");
            assert_ne!(stored, changed);
            std::fs::write(&path, changed).expect("tampered body with valid start prefix");
        }
        let original = checkpoint;
        let bytes = std::fs::read(&path).expect("original bytes");

        let error = draw(&directory, SEED, &mut checkpoint)
            .expect_err("active replay must compare against actual start and entire body");

        assert_config_error(error, "adaptive randomization snapshot mismatch");
        assert_eq!(checkpoint, original, "change start: {change_start}");
        assert_eq!(std::fs::read(path).expect("preserved bytes"), bytes);
    }
}

#[test]
fn invalid_draw_creates_nothing_and_a_valid_draw_creates_the_directory() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let child = directory.join("snapshots");
    let mut invalid = checkpoint();
    invalid.snapshot_count = MAX_TRAINING_COUNTER + 1;
    let original = invalid;
    let error = draw_adaptive_generation(&child, SEED, &mut invalid, AnnealScale::FULL)
        .expect_err("invalid snapshot count");
    assert_config_error(error, "adaptive randomization snapshot count is invalid");
    assert_eq!(invalid, original);
    assert!(!child.exists());
    let mut checkpoint = checkpoint();
    draw(&child, SEED, &mut checkpoint).expect("fresh directory");
    assert!(child.is_dir());
    verify(&child, &checkpoint).expect("committed first draw");
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

        let error = verify(&directory, &checkpoint).expect_err("inconsistent start");

        assert_config_error(error, field);
    }
}

#[test]
fn empty_verification_does_not_create_directory_and_wrong_type_is_rejected() {
    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let missing = directory.join("missing");
    verify(&missing, &checkpoint()).expect("empty prefix");
    assert!(!missing.exists());
    let file = directory.join("file");
    std::fs::write(&file, b"not a directory").expect("wrong type");

    let error = verify(&file, &checkpoint()).expect_err("even an empty prefix needs a directory");

    assert_config_error(
        error,
        "adaptive randomization directory must be a real directory",
    );
}

#[cfg(unix)]
#[test]
fn symlink_snapshots_and_directories_are_rejected_without_following() {
    use std::os::unix::fs::symlink;

    let directory = crate::ppo::test_directory("adaptive-snapshots");
    let mut checkpoint = checkpoint();
    draw(&directory, SEED, &mut checkpoint).expect("draw");
    let path = generation_path(&directory, 0);
    let target = directory.join("target");
    std::fs::rename(&path, &target).expect("move snapshot");
    symlink(&target, &path).expect("symlink snapshot");

    let error = verify(&directory, &checkpoint).expect_err("snapshot symlink");
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
    let error = verify(&link, &checkpoint).expect_err("directory symlink");
    assert_config_error(
        error,
        "adaptive randomization directory must be a real directory",
    );
}
