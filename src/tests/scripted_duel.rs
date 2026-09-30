use crate::{DuelConfig, DuelResult, ScriptKind, run_duel};

#[test]
fn harass_push_beats_teacher_on_most_games_from_both_sides() {
    let games = run_duel(DuelConfig {
        policy: ScriptKind::HarassPush,
        opponent: ScriptKind::Teacher,
        first_seed: 1,
        seeds: 4,
        threads: 4,
    })
    .expect("duel plays");

    let wins = games
        .iter()
        .filter(|game| game.result == DuelResult::Win)
        .count();
    // `drysua duel --seeds 100` is the full evidence; this pins the claim on a small sample.
    assert!(
        wins >= 6,
        "HarassPush won only {wins} of {} games",
        games.len()
    );
}

#[test]
fn duel_results_do_not_depend_on_worker_count() {
    let config = |threads| DuelConfig {
        policy: ScriptKind::HarassPush,
        opponent: ScriptKind::Teacher,
        first_seed: 7,
        seeds: 1,
        threads,
    };

    let serial = run_duel(config(1)).expect("serial duel");
    let parallel = run_duel(config(2)).expect("parallel duel");

    assert_eq!(serial, parallel);
}

#[test]
fn duel_rejects_unbounded_configurations() {
    let config = DuelConfig {
        policy: ScriptKind::HarassPush,
        opponent: ScriptKind::Teacher,
        first_seed: 1,
        seeds: 1,
        threads: 1,
    };
    for (config, message) in [
        (
            DuelConfig {
                threads: 0,
                ..config
            },
            "duel threads must be 1..=256, got 0",
        ),
        (
            DuelConfig { seeds: 0, ..config },
            "duel seeds must be 1..=10000, got 0",
        ),
        (
            DuelConfig {
                first_seed: u64::MAX,
                ..config
            },
            "duel seed range overflows",
        ),
    ] {
        assert_eq!(run_duel(config).expect_err("rejected"), message);
    }
}
