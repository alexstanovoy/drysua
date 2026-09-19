//! Annealing schedule, bounded draws, golden vectors and snapshot round-trips.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn test_directory(name: &str) -> PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-randomization-{name}-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        std::fs::remove_dir_all(&directory).expect("remove stale directory");
    }
    std::fs::create_dir(&directory).expect("create directory");
    directory
}

fn schedule(updates: u64, zero_updates: u64) -> AnnealSchedule {
    AnnealSchedule {
        updates,
        zero_updates,
    }
}

#[test]
fn scale_is_full_at_zero_and_zero_through_the_final_window() {
    let schedule = schedule(100, 20);
    assert_eq!(schedule.scale_bp(0), NOMINAL_BP);
    assert_eq!(schedule.scale_bp(80), 0);
    assert_eq!(schedule.scale_bp(99), 0);
    let quarter = schedule.scale_bp(20);
    assert!(
        (4_500..=5_500).contains(&quarter),
        "quarter of the annealed span is near half variance, got {quarter}"
    );
    assert!(schedule.scale_bp(40) < quarter);
}

#[test]
fn scale_has_pinned_values() {
    let schedule = schedule(10, 2);
    assert_eq!(schedule.zero_from_update(), 8);
    assert_eq!(schedule.scale_bp(0), 10_000);
    assert_eq!(schedule.scale_bp(1), 6_465);
    assert_eq!(schedule.scale_bp(2), 5_000);
    assert_eq!(schedule.scale_bp(4), 2_929);
    assert_eq!(schedule.scale_bp(7), 646);
    assert_eq!(schedule.scale_bp(8), 0);
}

#[test]
fn scale_is_monotone_over_every_update() {
    let schedule = schedule(257, 41);
    let mut previous = schedule.scale_bp(0);
    for update in 1..schedule.updates {
        let scale = schedule.scale_bp(update);
        assert!(scale <= previous, "scale rose at update {update}");
        assert!((0..=NOMINAL_BP).contains(&scale));
        previous = scale;
    }
}

#[test]
fn a_zero_window_covering_every_update_is_all_zero() {
    let full = schedule(5, 5);
    for update in 0..5 {
        assert_eq!(full.scale_bp(update), 0);
    }
    let covered = schedule(5, 9);
    for update in 0..5 {
        assert_eq!(covered.scale_bp(update), 0);
    }
}

#[test]
fn a_zero_scale_draw_is_nominal_with_no_applied_games() {
    let schedule = schedule(4, 4);
    let draw = draw_generation(11, 0, 2, 2, schedule).expect("zero-scale draw");
    assert_eq!(draw.scale_bp, 0);
    assert_eq!(draw.applied_games, 0);
    assert_eq!(draw.deltas, [0; VARIABLES.len()]);
    assert!(!draw.applies());
    assert!(draw.spec.is_nominal());
}

#[test]
fn a_generation_crossing_the_zero_window_records_applied_games() {
    // Span 3 updates, 2 games each: games 0..6 carry rules, games 6.. carry
    // none. Generation 1 (games 4..8) therefore applies to two games.
    let schedule = schedule(4, 1);
    let before = draw_generation(3, 0, 4, 2, schedule).expect("before");
    let crossing = draw_generation(3, 1, 4, 2, schedule).expect("crossing");
    let after = draw_generation(3, 2, 4, 2, schedule).expect("after");
    assert_eq!(before.applied_games, 4);
    assert_eq!(crossing.applied_games, 2);
    assert_eq!(crossing.start_game, 4);
    assert_eq!(crossing.end_game, 8);
    assert_eq!(after.applied_games, 0);
    assert!(crossing.applies());
    assert!(!after.applies());
}

#[test]
fn draws_are_deterministic_in_the_seed_and_generation() {
    let schedule = schedule(100, 20);
    let first = draw_generation(7, 3, 4, 8, schedule).expect("first draw");
    let second = draw_generation(7, 3, 4, 8, schedule).expect("second draw");
    let other_seed = draw_generation(8, 3, 4, 8, schedule).expect("other seed");
    let other_generation = draw_generation(7, 4, 4, 8, schedule).expect("other generation");
    assert_eq!(first, second);
    assert_ne!(first.spec, other_seed.spec);
    assert_ne!(first.spec, other_generation.spec);
}

#[test]
fn generation_maps_back_to_its_first_game_and_update() {
    let schedule = schedule(64, 8);
    let draw = draw_generation(3, 5, 6, 10, schedule).expect("draw");
    assert_eq!(draw.start_game, 30);
    assert_eq!(draw.end_game, 36);
    assert_eq!(draw.start_update, 3);
    assert_eq!(draw.scale_bp, schedule.scale_bp(3));
}

#[test]
fn every_delta_stays_inside_its_variable_bounds() {
    let schedule = schedule(400, 80);
    for generation in 0..600 {
        let draw = draw_generation(0x5eed, generation, 4, 8, schedule).expect("draw");
        for (index, variable) in VARIABLES.iter().enumerate() {
            let delta = draw.deltas[index];
            assert!(
                (variable.lower..=variable.upper).contains(&delta),
                "{} delta {delta} outside {}..={}",
                variable.name,
                variable.lower,
                variable.upper
            );
        }
        assert!(draw.spec.is_bounded());
    }
}

#[test]
fn full_scale_draws_have_the_expected_spread() {
    let sigma = VARIABLES[0].sigma_range / SIGMA_DIVISOR;
    let mut rng = PpoRng::new(0xabcd);
    let mut sum = 0i64;
    let mut square_sum = 0i64;
    let count = 2_000i64;
    let mut positive = 0i64;
    let mut negative = 0i64;
    for _ in 0..count {
        let delta = i64::from(normal_delta(&mut rng, sigma).expect("delta"));
        sum += delta;
        square_sum += delta * delta;
        if delta > 0 {
            positive += 1;
        }
        if delta < 0 {
            negative += 1;
        }
    }
    let mean = sum / count;
    let variance = square_sum / count - mean * mean;
    let std = (variance as f64).sqrt();
    assert!(mean.abs() < 150, "mean {mean} drifted");
    assert!(
        (3_000.0..=3_700.0).contains(&std),
        "std {std} is far from sigma {sigma}"
    );
    assert!(positive > 0 && negative > 0, "the draw is one-sided");
}

#[test]
fn per_variable_spread_matches_the_declared_sigma() {
    // Every generation starts in update 0, so every draw is at full scale.
    let schedule = schedule(2_000_000, 0);
    let mut sums = [0i64; VARIABLES.len()];
    let mut squares = [0i64; VARIABLES.len()];
    let count = 2_000i64;
    for generation in 0..count as u64 {
        let draw = draw_generation(0x5eed_5eed, generation, 1, 1_000_000, schedule)
            .expect("full-scale draw");
        assert_eq!(draw.scale_bp, NOMINAL_BP);
        for (index, delta) in draw.deltas.iter().enumerate() {
            let delta = i64::from(*delta);
            sums[index] += delta;
            squares[index] += delta * delta;
        }
    }
    // Expected standard deviations of the clamped normal: two-sided max HP
    // 3004, one-sided +100% 1940, +40pp 776, +50% 970, +30% 582.
    let expected: [f64; VARIABLES.len()] = [
        3_004.0, 1_940.0, 1_940.0, 1_940.0, 1_940.0, 1_940.0, 776.0, 970.0, 582.0, 970.0, 970.0,
    ];
    for (index, variable) in VARIABLES.iter().enumerate() {
        let mean = sums[index] / count;
        let variance = squares[index] / count - mean * mean;
        let std = (variance as f64).sqrt();
        let want = expected[index];
        assert!(
            (want * 0.88..=want * 1.12).contains(&std),
            "{} std {std} is far from {want}",
            variable.name
        );
    }
}

#[test]
fn generation_json_matches_a_golden_vector() {
    let schedule = schedule(10, 2);
    let draw = draw_generation(0x5eed_1234, 3, 4, 8, schedule).expect("draw");
    assert_eq!(generation_json(&draw), GOLDEN_GENERATION);
}

const GOLDEN_GENERATION: &str = "{\"schema\":\"drysua-domain-randomization/v2\",\"generation\":3,\"start_game\":12,\"end_game\":16,\"start_update\":1,\"scale_bp\":6465,\"applied_games\":4,\"deltas\":{\"max_hp\":-2949,\"gold_income\":0,\"max_mana\":0,\"physical_damage\":3683,\"magic_damage\":386,\"pure_damage\":2325,\"magic_resist\":581,\"status_resist\":0,\"move_speed\":153,\"mana_cost_rate\":-1042,\"cooldown_rate\":-941},\"spec\":{\"max_hp\":7051,\"gold_income\":10000,\"max_mana\":10000,\"physical_damage\":13683,\"magic_damage\":10386,\"pure_damage\":12325,\"magic_resist\":581,\"status_resist\":0,\"move_speed\":10153,\"mana_cost_rate\":8958,\"cooldown_rate\":9059},\"hash\":\"fc8dbde7bc2fff27\"}\n";

#[test]
fn json_is_canonical_and_carries_its_hash() {
    let schedule = schedule(50, 10);
    let draw = draw_generation(9, 2, 4, 8, schedule).expect("draw");
    let first = generation_json(&draw);
    let second = generation_json(&draw);
    assert_eq!(first, second);
    assert!(first.starts_with("{\"schema\":\"drysua-domain-randomization/v2\""));
    for variable in VARIABLES {
        assert!(
            first.contains(&format!("\"{}\":", variable.name)),
            "{} missing from the snapshot",
            variable.name
        );
    }
    assert!(first.contains("\"applied_games\":"));
}

#[test]
fn a_written_snapshot_is_verified_and_a_changed_file_is_rejected() {
    let directory = test_directory("snapshots");
    let schedule = schedule(20, 4);
    let draw = draw_generation(5, 1, 4, 8, schedule).expect("draw");
    write_generation_snapshot(&directory, &draw).expect("first write");
    write_generation_snapshot(&directory, &draw).expect("rewrite matches");
    let path = generation_path(&directory, 1);
    std::fs::write(&path, "{\"schema\":\"tampered\"}\n").expect("tamper");
    let error = write_generation_snapshot(&directory, &draw).expect_err("tampered snapshot");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: domain randomization snapshot mismatch"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn an_oversized_snapshot_is_rejected() {
    let directory = test_directory("oversized");
    let schedule = schedule(20, 4);
    let draw = draw_generation(6, 0, 4, 8, schedule).expect("draw");
    write_generation_snapshot(&directory, &draw).expect("write");
    let path = generation_path(&directory, 0);
    std::fs::write(&path, "x".repeat(5 * 1024)).expect("oversize");
    let error = verify_generation_snapshots(&directory, 6, 4, 8, schedule, 4)
        .expect_err("oversized snapshot");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: domain randomization snapshot is oversized"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_verification_covers_every_started_generation() {
    let directory = test_directory("resume");
    let schedule = schedule(40, 8);
    for generation in 0..3 {
        let draw = draw_generation(21, generation, 4, 8, schedule).expect("draw");
        write_generation_snapshot(&directory, &draw).expect("write");
    }
    assert_eq!(
        verify_generation_snapshots(&directory, 21, 4, 8, schedule, 8).expect("covered"),
        2
    );
    let error = verify_generation_snapshots(&directory, 21, 4, 8, schedule, 16)
        .expect_err("generation three missing");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: domain randomization snapshot is missing on resume"
    );
    let error =
        verify_generation_snapshots(&directory, 22, 4, 8, schedule, 8).expect_err("different seed");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: domain randomization snapshot mismatch"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}
