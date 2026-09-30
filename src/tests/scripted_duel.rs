use crate::{DuelConfig, MAP2_TICK_RATE, ScriptKind, StyleSpec, play_duel_game, run_duel};

#[test]
fn teacher_never_freezes_on_an_unreachable_spot() {
    // Seed 1: a walk to the landing behind its own tower ended short of it and was waited on
    // for the rest of the game. Seed 35: a Tango aimed at a tree out of reach, re-sent for 84 s.
    for (opponent, seed) in [(ScriptKind::HarassPush, 1), (ScriptKind::Teacher, 35)] {
        let game = play_duel_game(
            &StyleSpec::canonical(ScriptKind::Teacher),
            &StyleSpec::canonical(opponent),
            seed,
            0,
        )
        .expect("duel plays");

        assert!(
            game.policy_longest_idle <= 10 * MAP2_TICK_RATE,
            "Teacher stood still {} ticks against {opponent:?} on seed {seed}",
            game.policy_longest_idle
        );
    }
}

#[test]
fn duel_results_depend_only_on_seed_and_style() {
    // Styled seats draw their knobs and noise from the game seed, never from scheduling.
    let styled = |kind| StyleSpec::parse(kind, "styled,epsilon=100").expect("styled spec");
    let config = |threads| DuelConfig {
        policy: styled(ScriptKind::HarassPush),
        opponent: styled(ScriptKind::Teacher),
        first_seed: 7,
        seeds: 2,
        threads,
    };

    let serial = run_duel(config(1)).expect("serial duel");
    let parallel = run_duel(config(3)).expect("parallel duel");

    assert_eq!(serial, parallel);
    assert_ne!(serial[0].opponent_style, serial[2].opponent_style);
}

#[test]
fn duel_rejects_unbounded_configurations() {
    let config = DuelConfig {
        policy: StyleSpec::canonical(ScriptKind::HarassPush),
        opponent: StyleSpec::canonical(ScriptKind::Teacher),
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
    for (spec, message) in [
        (
            "hover=5000",
            "style knob `hover` must lie in 600..1600 with low <= high, got 5000",
        ),
        (
            "retreat=40..30",
            "style knob `retreat` must lie in 5..60 with low <= high, got 40..30",
        ),
        (
            "aggro=90",
            "harass-push has no style knob `aggro`; knobs: period, epsilon, retreat, return, salves, clarities, hover, engage, tanks, memory, razes_first",
        ),
    ] {
        assert_eq!(
            StyleSpec::parse(ScriptKind::HarassPush, spec).expect_err("rejected"),
            message
        );
    }
}
