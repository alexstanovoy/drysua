use crate::{DuelConfig, MAP2_TICK_RATE, ScriptKind, play_duel_game, run_duel};

#[test]
fn teacher_never_freezes_on_an_unreachable_spot() {
    // Seed 1: a walk to the landing behind its own tower ended short of it and was waited on
    // for the rest of the game. Seed 35: a Tango aimed at a tree out of reach, re-sent for 84 s.
    for (opponent, seed) in [(ScriptKind::HarassPush, 1), (ScriptKind::Teacher, 35)] {
        let game = play_duel_game(ScriptKind::Teacher, opponent, seed, 0).expect("duel plays");

        assert!(
            game.policy_longest_idle <= 10 * MAP2_TICK_RATE,
            "Teacher stood still {} ticks against {opponent:?} on seed {seed}",
            game.policy_longest_idle
        );
    }
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
