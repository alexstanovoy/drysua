//! Annealing schedule, bounded draws, golden vectors and snapshot round-trips.

use super::*;
use crate::ppo::test_directory;

fn schedule(updates: u64, zero_updates: u64) -> AnnealSchedule {
    AnnealSchedule {
        updates,
        zero_updates,
        scale: AnnealScale::FULL,
    }
}

fn scale(start_bp: i32, end_bp: i32) -> AnnealScale {
    AnnealScale { start_bp, end_bp }
}

/// Independent integer square root: brute force, unlike the kernel.
fn reference_root(value: u128) -> u128 {
    let mut root = 0u128;
    while (root + 1) * (root + 1) <= value {
        root += 1;
    }
    root
}

/// Independent reference for the whole scale ramp, in basis points.
fn reference_scale_bp(scale: AnnealScale, updates: u64, zero_updates: u64, update: u64) -> i32 {
    let span = updates.saturating_sub(zero_updates);
    if span == 0 || update >= span {
        return 0;
    }
    let root = reference_root(u128::from(update) * 100_000_000 / u128::from(span));
    ((i64::from(scale.start_bp) * (10_000 - root as i64) + i64::from(scale.end_bp) * root as i64)
        / 10_000) as i32
}

#[test]
fn scale_ramp_matches_an_independent_reference_and_is_zero_in_the_tail() {
    let scales = [
        scale(0, 0),
        AnnealScale::FULL,
        scale(0, 20_000),
        scale(5_000, 5_000),
    ];
    let runs = [
        (64, 0),
        (100, 20),
        (200, 40),
        (257, 41),
        (9, 3),
        (40, 40),
        (5, 9),
        (2, 0),
    ];
    for (scale, (updates, zero_updates)) in scales
        .into_iter()
        .flat_map(|scale| runs.map(|run| (scale, run)))
    {
        let schedule = AnnealSchedule {
            updates,
            zero_updates,
            scale,
        };
        let mut previous = scale.start_bp;
        for update in 0..=updates {
            let case = format!("{scale:?} updates={updates} zero={zero_updates} update={update}");
            let value = schedule.scale_bp(update);
            assert_eq!(
                value,
                reference_scale_bp(scale, updates, zero_updates, update),
                "{case}"
            );
            if update >= updates.saturating_sub(zero_updates) {
                assert_eq!(value, 0, "{case}");
            } else if scale.start_bp >= scale.end_bp {
                assert!(value <= previous, "{case}: a falling ramp rose");
            }
            previous = value;
        }
    }
}

#[test]
fn scale_ramp_pins_literal_values() {
    let falling = AnnealSchedule {
        updates: 10_000,
        zero_updates: 0,
        scale: AnnealScale::FULL,
    };
    let rising = AnnealSchedule {
        scale: scale(0, 20_000),
        ..falling
    };
    // A ten thousand update ramp keeps the integer root exact at 2_500.
    for (schedule, update, expected) in [
        (schedule(10, 2), 0, 10_000),
        (schedule(10, 2), 1, 6_465),
        (schedule(10, 2), 2, 5_000),
        (schedule(10, 2), 4, 2_929),
        (schedule(10, 2), 7, 646),
        (schedule(10, 2), 8, 0),
        (falling, 2_500, 5_000),
        (falling, 9_999, 1),
        (rising, 0, 0),
        (rising, 2_500, 10_000),
        (rising, 9_999, 19_998),
        (rising, 10_000, 0),
    ] {
        assert_eq!(
            schedule.scale_bp(update),
            expected,
            "{schedule:?} at {update}"
        );
    }
}

#[test]
fn draws_map_to_their_games_update_scale_and_applied_games() {
    // One game per update; schedule (4, 1) with 2 per generation: games 0..3 carry rules.
    for (generation, schedule, games, expected, applies) in [
        (0, schedule(4, 1), 2, (0, 2, 0, 2), true),
        (1, schedule(4, 1), 2, (2, 4, 2, 1), true),
        (2, schedule(4, 1), 2, (4, 6, 4, 0), false),
        (5, schedule(64, 8), 6, (30, 36, 30, 6), true),
        (0, schedule(4, 4), 2, (0, 2, 0, 0), false),
    ] {
        let draw = draw_generation(3, generation, games, schedule).expect("draw");
        let case = format!("generation {generation} of {schedule:?}");
        assert_eq!(
            (
                draw.start_game,
                draw.end_game,
                draw.start_update,
                draw.applied_games
            ),
            expected,
            "{case}"
        );
        assert_eq!(
            draw.scale_bp,
            schedule.scale_bp(draw.start_update),
            "{case}"
        );
        assert_eq!(draw.applies(), applies, "{case}");
        if draw.scale_bp == 0 {
            assert_eq!(draw.deltas, [0; VARIABLES.len()], "{case}");
            assert!(draw.spec.is_nominal(), "{case}");
        }
    }
}

#[test]
fn draws_are_deterministic_in_the_seed_and_generation() {
    let schedule = schedule(100, 20);
    let first = draw_generation(7, 3, 4, schedule).expect("first draw");
    let second = draw_generation(7, 3, 4, schedule).expect("second draw");
    let other_seed = draw_generation(8, 3, 4, schedule).expect("other seed");
    let other_generation = draw_generation(7, 4, 4, schedule).expect("other generation");
    assert_eq!(first, second);
    assert_ne!(first.spec, other_seed.spec);
    assert_ne!(first.spec, other_generation.spec);
}

#[test]
fn every_delta_stays_inside_its_variable_bounds_up_to_the_maximum_scale() {
    let maximum = AnnealScale {
        start_bp: AnnealScale::MAX_BP,
        end_bp: AnnealScale::MAX_BP,
    };
    for (generation, scale) in
        (0..600).flat_map(|generation| [(generation, AnnealScale::FULL), (generation, maximum)])
    {
        let schedule = AnnealSchedule {
            updates: 400,
            zero_updates: 80,
            scale,
        };
        let draw = draw_generation(0x5eed, generation, 4, schedule).expect("draw");
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
fn per_variable_spread_matches_the_declared_sigma() {
    // A flat full-scale ramp, so every draw is at full scale.
    let schedule = AnnealSchedule {
        scale: scale(NOMINAL_BP, NOMINAL_BP),
        ..schedule(2_000_000, 0)
    };
    let mut sums = [0i64; VARIABLES.len()];
    let mut squares = [0i64; VARIABLES.len()];
    let count = 2_000i64;
    for generation in 0..count as u64 {
        let draw = draw_generation(0x5eed_5eed, generation, 1, schedule).expect("full-scale draw");
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
    let schedule = schedule(40, 8);
    let draw = draw_generation(0x5eed_1234, 3, 4, schedule).expect("draw");
    assert_eq!(generation_json(&draw), GOLDEN_GENERATION);
}

const GOLDEN_GENERATION: &str = "{\"schema\":\"drysua-domain-randomization/v2\",\"generation\":3,\"start_game\":12,\"end_game\":16,\"start_update\":12,\"scale_bp\":3877,\"applied_games\":4,\"deltas\":{\"max_hp\":-1768,\"gold_income\":0,\"max_mana\":0,\"physical_damage\":2208,\"magic_damage\":231,\"pure_damage\":1394,\"magic_resist\":348,\"status_resist\":0,\"move_speed\":92,\"mana_cost_rate\":-625,\"cooldown_rate\":-565},\"spec\":{\"max_hp\":8232,\"gold_income\":10000,\"max_mana\":10000,\"physical_damage\":12208,\"magic_damage\":10231,\"pure_damage\":11394,\"magic_resist\":348,\"status_resist\":0,\"move_speed\":10092,\"mana_cost_rate\":9375,\"cooldown_rate\":9435},\"hash\":\"9ac73daf2c0c3f5b\"}\n";

#[test]
fn a_written_snapshot_is_verified_and_a_changed_file_is_rejected() {
    let directory = test_directory("snapshots");
    let schedule = schedule(20, 4);
    let draw = draw_generation(5, 1, 4, schedule).expect("draw");
    write_generation_snapshots(&directory, std::slice::from_ref(&draw)).expect("first write");
    write_generation_snapshots(&directory, std::slice::from_ref(&draw)).expect("rewrite matches");
    let path = generation_path(&directory, 1);
    std::fs::write(&path, "{\"schema\":\"tampered\"}\n").expect("tamper");
    let error = write_generation_snapshots(&directory, std::slice::from_ref(&draw))
        .expect_err("tampered snapshot");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: domain randomization snapshot mismatch"
    );
}

#[test]
fn resume_verification_covers_started_generations_and_rejects_any_changed_chain() {
    let directory = test_directory("resume");
    let schedule = schedule(40, 8);
    for generation in 0..3 {
        let draw = draw_generation(21, generation, 4, schedule).expect("draw");
        write_generation_snapshots(&directory, std::slice::from_ref(&draw)).expect("write");
    }
    assert_eq!(
        verify_generation_snapshots(&directory, 21, 4, schedule, 8),
        Ok(2)
    );
    let rescaled = AnnealSchedule {
        scale: scale(0, 20_000),
        ..schedule
    };
    let mismatch = "domain randomization snapshot mismatch";
    for (name, seed, schedule, completed_games, message) in [
        (
            "generation three missing",
            21,
            schedule,
            16,
            "domain randomization snapshot is missing on resume",
        ),
        ("different seed", 22, schedule, 8, mismatch),
        ("different scale", 21, rescaled, 8, mismatch),
    ] {
        assert_eq!(
            verify_generation_snapshots(&directory, seed, 4, schedule, completed_games),
            Err(PpoError::InvalidConfig(message)),
            "{name}"
        );
    }
    std::fs::write(generation_path(&directory, 0), "x".repeat(5 * 1024)).expect("oversize");
    assert_eq!(
        verify_generation_snapshots(&directory, 21, 4, schedule, 8),
        Err(PpoError::InvalidConfig(
            "domain randomization snapshot is oversized"
        ))
    );
}

/// Regression: the fixed schedule followed snapshot and directory symlinks the
/// adaptive schedule refuses.
#[cfg(unix)]
#[test]
fn symlinked_snapshots_and_directories_are_refused_without_following() {
    use std::os::unix::fs::symlink;

    let directory = test_directory("snapshot-links");
    let schedule = schedule(20, 4);
    let draw = draw_generation(5, 0, 4, schedule).expect("draw");
    write_generation_snapshots(&directory, std::slice::from_ref(&draw)).expect("write");
    let path = generation_path(&directory, 0);
    let target = directory.join("target.json");
    std::fs::rename(&path, &target).expect("move snapshot");
    symlink(&target, &path).expect("link snapshot");
    let regular = Some(PpoError::InvalidConfig(
        "domain randomization snapshot must be a regular file",
    ));
    assert_eq!(
        write_generation_snapshots(&directory, std::slice::from_ref(&draw)).err(),
        regular
    );
    assert_eq!(
        verify_generation_snapshots(&directory, 5, 4, schedule, 1).err(),
        regular
    );
    assert!(
        std::fs::symlink_metadata(&path)
            .expect("link kept")
            .is_symlink()
    );
    std::fs::rename(&target, &path).expect("restore snapshot");
    let link = directory.join("directory-link");
    symlink(&*directory, &link).expect("link directory");
    let real_directory = Some(PpoError::InvalidConfig(
        "domain randomization directory must be a real directory",
    ));
    assert_eq!(
        write_generation_snapshots(&link, std::slice::from_ref(&draw)).err(),
        real_directory
    );
    assert_eq!(
        verify_generation_snapshots(&link, 5, 4, schedule, 1).err(),
        real_directory
    );
}

#[test]
fn scale_endpoints_are_bounded_to_ten_times_full_variance() {
    for scale in [
        AnnealScale {
            start_bp: 0,
            end_bp: 0,
        },
        AnnealScale::FULL,
        AnnealScale {
            start_bp: AnnealScale::MAX_BP,
            end_bp: AnnealScale::MAX_BP,
        },
    ] {
        assert_eq!(scale.validate().expect("valid scale"), scale);
    }
    for scale in [
        AnnealScale {
            start_bp: -1,
            end_bp: 0,
        },
        AnnealScale {
            start_bp: 0,
            end_bp: AnnealScale::MAX_BP + 1,
        },
        AnnealScale {
            start_bp: 1_000_000,
            end_bp: 0,
        },
    ] {
        assert!(scale.validate().is_err(), "{scale:?} must be rejected");
    }
}

#[test]
fn the_scale_scope_suffix_is_canonical_and_only_non_default_is_split() {
    assert_eq!(
        AnnealScale::FULL.scope_suffix(),
        " --environment-scale-start 1 --environment-scale-end 0"
    );
    assert_eq!(
        AnnealScale {
            start_bp: 0,
            end_bp: 20_000,
        }
        .scope_suffix(),
        " --environment-scale-start 0 --environment-scale-end 2"
    );
    assert_eq!(
        AnnealScale {
            start_bp: 5_000,
            end_bp: 5_000,
        }
        .scope_suffix(),
        " --environment-scale-start 0.5 --environment-scale-end 0.5"
    );
    let ramp = AnnealScale {
        start_bp: 0,
        end_bp: 20_000,
    };
    let command = format!("train-annealed --updates 8{}", ramp.scope_suffix());
    assert_eq!(
        split_scale_scope(&command),
        ("train-annealed --updates 8", ramp)
    );
    assert_eq!(
        split_scale_scope("train-annealed --updates 8"),
        ("train-annealed --updates 8", AnnealScale::FULL)
    );
    // The default ramp is never recorded, so its tokens are not canonical.
    assert_eq!(
        split_scale_scope("train-annealed --environment-scale-start 1 --environment-scale-end 0"),
        (
            "train-annealed --environment-scale-start 1 --environment-scale-end 0",
            AnnealScale::FULL
        )
    );
    // A partial pair or a sub-basis-point value stays for the validator to reject.
    for text in [
        "train-annealed --environment-scale-start 0",
        "train-annealed --environment-scale-end 2",
        "train-annealed --environment-scale-start 0.00005 --environment-scale-end 0",
        "train-annealed --environment-scale-start 0 --environment-scale-end 2 --environment-unknown",
    ] {
        assert_eq!(split_scale_scope(text), (text, AnnealScale::FULL), "{text}");
    }
}
