use super::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
    EnvironmentDecimal,
};
use crate::{MAX_TRAINING_COUNTER, PpoError};

type TransitionCase = (
    &'static str,
    AdaptiveEnvironmentConfig,
    AdaptiveEnvironmentLimits,
    &'static [u64],
    u64,
);

fn decimal(text: &str) -> EnvironmentDecimal {
    text.parse().expect("valid test decimal")
}

/// An extension whose single award fills the budget to `MAX_TRAINING_COUNTER` from base one.
fn overflowing_extension() -> EnvironmentDecimal {
    EnvironmentDecimal::from_units((MAX_TRAINING_COUNTER - 1) * EnvironmentDecimal::SCALE)
}

fn limits(base: u64, total: u64, zero: u64) -> AdaptiveEnvironmentLimits {
    AdaptiveEnvironmentLimits {
        base_updates: base,
        total_updates: total,
        zero_updates: zero,
    }
    .validate()
    .expect("valid test limits")
}

fn with(change: fn(&mut AdaptiveEnvironmentConfig)) -> AdaptiveEnvironmentConfig {
    let mut config = AdaptiveEnvironmentConfig::default();
    change(&mut config);
    config
}

fn generation_start(generation: u64, start_update: u64) -> AdaptiveEnvironmentState {
    AdaptiveEnvironmentState {
        generation,
        start_update,
        ..AdaptiveEnvironmentState::default()
    }
}

fn run(
    config: AdaptiveEnvironmentConfig,
    limits: AdaptiveEnvironmentLimits,
    wins: &[u64],
) -> AdaptiveEnvironmentState {
    assert!(wins.len() <= 64);
    assert!(wins.len() as u64 <= limits.total_updates);
    let mut state = AdaptiveEnvironmentState::default();
    for (index, &wins) in wins.iter().enumerate() {
        let completed_update = index as u64 + 1;
        state = state
            .observe(config, limits, completed_update, wins, 10)
            .expect("valid complete update");
        state
            .validate(config, limits, completed_update)
            .expect("candidate is structurally valid");
    }
    state
}

fn assert_clean_generation_starts(cases: &[TransitionCase]) {
    for &(name, config, limits, wins, start_update) in cases {
        assert_eq!(
            run(config, limits, wins),
            generation_start(1, start_update),
            "{name}"
        );
    }
}

#[test]
fn seeded_sequences_match_literal_overlapping_windows_and_global_boundaries() {
    for case in 0..256u64 {
        verify_seeded_sequence(case);
    }
}

fn seeded_config(case: u64) -> AdaptiveEnvironmentConfig {
    AdaptiveEnvironmentConfig {
        success_updates: 1 + case % 4,
        success_rate: EnvironmentDecimal::from_units(
            [0, 400_000, 800_000, 1_000_000][(case / 4 % 4) as usize],
        ),
        poor_updates: 1 + case / 16 % 4,
        poor_rate: EnvironmentDecimal::from_units(
            [0, 200_000, 600_000, 1_000_000][(case / 64 % 4) as usize],
        ),
        extension: EnvironmentDecimal::from_units(
            [0, 100_000, 750_000, 1_000_000, 2_000_000][(case % 5) as usize],
        ),
    }
}

fn verify_seeded_sequence(case: u64) {
    let config = seeded_config(case);
    let limits = limits(1 + case % 7, 32, case % 9);
    let seed = 0xa9a9_2026_0928_0000 ^ case;
    let mut random = seed;
    let mut state = AdaptiveEnvironmentState::default();
    let mut window = Vec::with_capacity(32);
    let (mut generation, mut start, mut awards) = (0, 0, 0);
    for update in 1..=32 {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let wins = (random >> 32) % 11;
        window.push(wins);
        assert!(window.len() <= 32);
        let good = |wins: &u64| *wins * 1_000_000 >= config.success_rate.units() * 10;
        let poor = |wins: &u64| *wins * 1_000_000 <= config.poor_rate.units() * 10;
        let success = window.len() >= config.success_updates as usize
            && window
                .iter()
                .rev()
                .take(config.success_updates as usize)
                .all(good);
        let extension = window.len() >= config.poor_updates as usize
            && window
                .iter()
                .rev()
                .take(config.poor_updates as usize)
                .all(poor);
        if extension && !success {
            awards += 1;
        }
        let budget = limits.base_updates + awards * config.extension.units() / 1_000_000;
        let clean = update == limits.total_updates - limits.zero_updates && start < update;
        if update < limits.total_updates && (success || clean || window.len() as u64 >= budget) {
            generation += 1;
            start = update;
            awards = 0;
            window.clear();
        }
        state = state
            .observe(config, limits, update, wins, 10)
            .unwrap_or_else(|error| panic!("seed={seed} update={update}: {error}"));
        assert_eq!(
            (
                state.generation,
                state.start_update,
                state.updates_in_generation,
                state.extension_awards
            ),
            (generation, start, window.len() as u64, awards),
            "seed={seed} update={update} config={config:?}"
        );
    }
}

#[test]
fn decimal_parses_exact_millionths_and_displays_canonically() {
    for (text, units, canonical) in [
        ("0", 0, "0"),
        (".75", 750_000, "0.75"),
        ("0.1", 100_000, "0.1"),
        ("1", 1_000_000, "1"),
        ("1.250000", 1_250_000, "1.25"),
        ("001.200000", 1_200_000, "1.2"),
        ("00000000000000000000.100000", 100_000, "0.1"),
        ("0.000001", 1, "0.000001"),
        ("18446744073709.551615", u64::MAX, "18446744073709.551615"),
    ] {
        let value = decimal(text);
        assert_eq!(value, EnvironmentDecimal::from_units(units), "{text}");
        assert_eq!(value.to_string(), canonical, "{text}");
        assert_eq!(decimal(canonical), value, "{text}");
    }
}

#[test]
fn decimal_rejects_non_plain_notation_overprecision_and_overflow() {
    let notation = "environment decimal must use unsigned plain decimal notation";
    let overflow = "environment decimal exceeds u64 millionths";
    let mut cases = [
        "", ".", "1.", "-0", "-1", "NaN", "nan", "inf", "infinity", "+1", "1e0", "1E2", " 1", "1 ",
        "1..2", "１", "0_1", "0,1",
    ]
    .map(|text| (text, notation))
    .to_vec();
    cases.extend([
        (
            "0.0000001",
            "environment decimal has more than six fractional digits",
        ),
        (
            "1.0000000",
            "environment decimal has more than six fractional digits",
        ),
        ("18446744073709.551616", overflow),
        ("18446744073710", overflow),
        ("18446744073709551616", overflow),
        (
            "0000000000000000000000000000",
            "environment decimal exceeds 27 bytes",
        ),
    ]);
    for (text, message) in cases {
        assert_eq!(
            text.parse::<EnvironmentDecimal>(),
            Err(PpoError::InvalidConfig(message)),
            "{text:?}"
        );
    }
}

#[test]
fn config_validation_bounds_windows_and_rates_but_only_caps_extension_globally() {
    let success_window = "environment success updates must be in 1..=MAX_TRAINING_COUNTER";
    let poor_window = "environment poor updates must be in 1..=MAX_TRAINING_COUNTER";
    let extension = "environment extension must not exceed MAX_TRAINING_COUNTER";
    let invalid: [(&str, AdaptiveEnvironmentConfig, &str); 8] = [
        (
            "zero success window",
            with(|c| c.success_updates = 0),
            success_window,
        ),
        (
            "success window above the counter bound",
            with(|c| c.success_updates = MAX_TRAINING_COUNTER + 1),
            success_window,
        ),
        (
            "zero poor window",
            with(|c| c.poor_updates = 0),
            poor_window,
        ),
        (
            "poor window above the counter bound",
            with(|c| c.poor_updates = MAX_TRAINING_COUNTER + 1),
            poor_window,
        ),
        (
            "success rate above one",
            with(|c| c.success_rate = EnvironmentDecimal::from_units(1_000_001)),
            "environment success rate must be in [0, 1]",
        ),
        (
            "poor rate above one",
            with(|c| c.poor_rate = EnvironmentDecimal::from_units(1_000_001)),
            "environment poor rate must be in [0, 1]",
        ),
        (
            "extension one millionth above the counter bound",
            with(|c| {
                c.extension = EnvironmentDecimal::from_units(
                    MAX_TRAINING_COUNTER * EnvironmentDecimal::SCALE + 1,
                )
            }),
            extension,
        ),
        (
            "maximum extension units",
            with(|c| c.extension = EnvironmentDecimal::from_units(u64::MAX)),
            extension,
        ),
    ];
    for (name, config, message) in invalid {
        assert_eq!(
            config.validate(),
            Err(PpoError::InvalidConfig(message)),
            "{name}"
        );
    }
    for config in [
        with(|c| (c.success_rate, c.poor_rate) = (decimal("0"), decimal("1"))),
        with(|c| {
            c.success_updates = MAX_TRAINING_COUNTER;
            c.poor_updates = MAX_TRAINING_COUNTER;
            c.extension =
                EnvironmentDecimal::from_units(MAX_TRAINING_COUNTER * EnvironmentDecimal::SCALE);
        }),
    ] {
        assert_eq!(config.validate(), Ok(config));
    }
}

#[test]
fn success_requires_consecutive_individually_qualifying_updates_not_an_average() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(20, 100, 0);
    let mixed = run(config, limits, &[7, 9]);
    assert_eq!(mixed.generation, 0);
    assert_eq!(mixed.success_streak, 1);
    assert_eq!(run(config, limits, &[8, 8]), generation_start(1, 2));
    let interrupted = run(config, limits, &[8, 7, 8]);
    assert_eq!(interrupted.generation, 0);
    assert_eq!(interrupted.success_streak, 1);
}

#[test]
fn poor_streak_awards_each_overlapping_qualifying_update_not_an_average() {
    let config = with(|c| c.poor_updates = 2);
    let limits = limits(20, 100, 0);
    for (wins, awards, streak) in [
        (&[1, 3][..], 0, 0),
        (&[2][..], 0, 1),
        (&[2, 2, 2][..], 2, 2),
        (&[2, 2, 3, 2][..], 1, 1),
    ] {
        let state = run(config, limits, wins);
        assert_eq!(state.extension_awards, awards, "{wins:?}");
        assert_eq!(state.poor_streak, streak, "{wins:?}");
    }
}

#[test]
fn extension_credit_adds_the_floor_of_exact_millionth_products() {
    for (extension, base, awards, budget) in [
        (".75", 4, 1, 4),
        (".75", 4, 2, 5),
        (".75", 4, 3, 6),
        (".1", 20, 9, 20),
        (".1", 20, 10, 21),
        ("0", 2, 1, 2),
    ] {
        let config = AdaptiveEnvironmentConfig {
            extension: decimal(extension),
            ..AdaptiveEnvironmentConfig::default()
        };
        let limits = limits(base, 100, 0);
        let state = run(config, limits, &vec![0; awards as usize]);
        let case = format!("{awards} awards of {extension}");
        assert_eq!(state.extension_awards, awards, "{case}");
        assert_eq!(state.effective_budget(config, limits), Ok(budget), "{case}");
    }
}

#[test]
fn budget_exhaustion_starts_a_clean_generation() {
    assert_clean_generation_starts(&[
        (
            "two three-quarter awards extend base four to five updates",
            AdaptiveEnvironmentConfig::default(),
            limits(4, 20, 0),
            &[0, 0, 5, 5, 5],
            5,
        ),
        (
            "zero extension records awards but keeps the base budget",
            with(|c| c.extension = decimal("0")),
            limits(2, 10, 0),
            &[0, 0],
            2,
        ),
        (
            "an incomplete success window is discarded",
            with(|c| (c.success_updates, c.poor_updates, c.extension) = (3, 3, decimal("10"))),
            limits(2, 10, 0),
            &[8, 8],
            2,
        ),
        (
            "an incomplete poor window is discarded",
            with(|c| (c.success_updates, c.poor_updates, c.extension) = (3, 3, decimal("10"))),
            limits(2, 10, 0),
            &[0, 0],
            2,
        ),
    ]);
}

#[test]
fn success_and_the_zero_phase_boundary_start_a_clean_generation_once() {
    let zero_extension = with(|c| c.extension = decimal("0"));
    assert_clean_generation_starts(&[
        (
            "success discards accumulated extension credit",
            with(|c| c.extension = decimal("10")),
            limits(1, 20, 0),
            &[0, 8, 8],
            3,
        ),
        (
            "overlapping success outranks a poor award that would overflow",
            with(|c| {
                c.success_rate = decimal(".2");
                c.poor_rate = decimal(".8");
                c.extension = overflowing_extension();
            }),
            limits(1, 10, 0),
            &[5, 5],
            2,
        ),
        (
            "the boundary discards a poor award that would overflow",
            with(|c| c.extension = overflowing_extension()),
            limits(1, 3, 1),
            &[0, 0],
            2,
        ),
        (
            "boundary coinciding with success",
            zero_extension,
            limits(5, 8, 3),
            &[5, 5, 5, 8, 8],
            5,
        ),
        (
            "boundary coinciding with exhaustion",
            zero_extension,
            limits(5, 8, 3),
            &[5; 5],
            5,
        ),
    ]);
}

#[test]
fn zero_phase_boundary_forces_a_clean_environment_despite_extension_credit() {
    let config = with(|c| c.extension = decimal("10"));
    let limits = limits(10, 8, 3);
    let boundary = run(config, limits, &[0; 5]);
    assert_eq!(boundary, generation_start(1, 5));
    let after = boundary.observe(config, limits, 6, 0, 10).unwrap();
    assert_eq!(after.generation, 1);
    assert_eq!(after.updates_in_generation, 1);
    assert_eq!(after.extension_awards, 1);
}

#[test]
fn zero_phase_covering_the_whole_run_does_not_create_an_initial_transition() {
    let state = run(
        AdaptiveEnvironmentConfig::default(),
        limits(10, 3, 3),
        &[5; 3],
    );
    assert_eq!(state.generation, 0);
    assert_eq!(state.start_update, 0);
    assert_eq!(state.updates_in_generation, 3);
}

#[test]
fn extensions_of_at_least_one_award_before_exhaustion_and_retain_until_total() {
    for extension in ["1", "1.5", "10"] {
        for poor_updates in [1, 2] {
            let config = AdaptiveEnvironmentConfig {
                poor_updates,
                extension: decimal(extension),
                ..AdaptiveEnvironmentConfig::default()
            };
            let limits = limits(poor_updates, 12, 0);
            let state = run(config, limits, &[0; 12]);
            let case = format!("extension {extension}, window {poor_updates}");
            assert_eq!(state.generation, 0, "{case}");
            assert_eq!(state.updates_in_generation, 12, "{case}");
            assert_eq!(state.poor_streak, poor_updates, "{case}");
            assert_eq!(state.extension_awards, 13 - poor_updates, "{case}");
            assert!(
                state.effective_budget(config, limits).unwrap() > limits.total_updates,
                "{case}"
            );
        }
    }
}

#[test]
fn final_update_keeps_completed_counters_without_advancing() {
    let overlapping = |success_updates| {
        let mut config = with(|c| (c.success_rate, c.poor_rate) = (decimal(".2"), decimal(".8")));
        config.success_updates = success_updates;
        config.extension = overflowing_extension();
        config
    };
    for (name, config, wins, expected) in [
        (
            "completed success streak",
            with(|c| c.success_updates = 1),
            &[10][..],
            (1, 1, 0, 0),
        ),
        (
            "exhausted budget",
            AdaptiveEnvironmentConfig::default(),
            &[5][..],
            (1, 0, 0, 0),
        ),
        (
            "success suppresses the overlapping poor award",
            overlapping(1),
            &[5][..],
            (1, 1, 1, 0),
        ),
        (
            "success keeps the award of an earlier update",
            overlapping(2),
            &[5, 5][..],
            (2, 2, 1, 1),
        ),
    ] {
        let state = run(config, limits(1, wins.len() as u64, 0), wins);
        let (updates_in_generation, success_streak, poor_streak, extension_awards) = expected;
        let expected = AdaptiveEnvironmentState {
            updates_in_generation,
            success_streak,
            poor_streak,
            extension_awards,
            ..AdaptiveEnvironmentState::default()
        };
        assert_eq!(state, expected, "{name}");
    }
}

#[test]
fn terminal_states_are_invalid_before_total_and_cannot_be_observed_again() {
    let config = AdaptiveEnvironmentConfig::default();
    let success = AdaptiveEnvironmentState {
        updates_in_generation: 1,
        success_streak: 1,
        ..AdaptiveEnvironmentState::default()
    };
    let exhausted = AdaptiveEnvironmentState {
        updates_in_generation: 1,
        ..AdaptiveEnvironmentState::default()
    };
    let overspent = AdaptiveEnvironmentState {
        updates_in_generation: 2,
        ..AdaptiveEnvironmentState::default()
    };
    for (name, config, state, global, message) in [
        (
            "success before total",
            with(|c| c.success_updates = 1),
            success,
            1,
            "environment success streak requires advancement before total updates",
        ),
        (
            "exhaustion before total",
            config,
            exhausted,
            1,
            "environment budget is exhausted before total updates",
        ),
        (
            "spending beyond the budget at total",
            config,
            overspent,
            2,
            "environment spent updates exceed effective budget",
        ),
    ] {
        assert_eq!(
            state.validate(config, limits(1, 2, 0), global),
            Err(PpoError::InvalidTransition(message)),
            "{name}"
        );
    }
    assert_eq!(
        exhausted.observe(config, limits(1, 1, 0), 2, 5, 10),
        Err(PpoError::InvalidTransition(
            "environment completed update must be in 1..=total updates"
        ))
    );
}

#[test]
fn rate_endpoints_are_inclusive_and_large_game_counts_use_exact_products() {
    let config = AdaptiveEnvironmentConfig {
        success_updates: 1,
        success_rate: decimal("1"),
        poor_rate: decimal("0"),
        ..Default::default()
    };
    let limits = limits(10, 20, 0);
    for (wins, generation, awards) in [(0, 0, 1), (1, 0, 0), (u64::MAX - 1, 0, 0), (u64::MAX, 1, 0)]
    {
        let state = AdaptiveEnvironmentState::default()
            .observe(config, limits, 1, wins, u64::MAX)
            .unwrap();
        assert_eq!(state.generation, generation);
        assert_eq!(state.extension_awards, awards);
    }
}

#[test]
fn an_update_without_finished_games_spends_budget_but_breaks_both_streaks() {
    let config = AdaptiveEnvironmentConfig {
        success_updates: 2,
        poor_updates: 2,
        ..AdaptiveEnvironmentConfig::default()
    };
    let limits = limits(10, 20, 0);
    let winning = run(config, limits, &[10]);
    assert_eq!(winning.success_streak, 1);
    let empty = winning
        .observe(config, limits, 2, 0, 0)
        .expect("no finished game");
    assert_eq!(empty.updates_in_generation, 2);
    assert_eq!((empty.success_streak, empty.poor_streak), (0, 0));
    let losing = run(config, limits, &[0]);
    assert_eq!(losing.poor_streak, 1);
    let empty = losing
        .observe(config, limits, 2, 0, 0)
        .expect("no finished game");
    assert_eq!((empty.success_streak, empty.poor_streak), (0, 0));
}

#[test]
fn invalid_results_and_update_sequences_leave_the_original_state_unchanged() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(10, 20, 0);
    let state = run(config, limits, &[0]);
    let saved = state;
    for (completed, wins, games, message) in [
        (2, 11, 10, "environment wins must not exceed games"),
        (
            0,
            0,
            10,
            "environment completed update must be in 1..=total updates",
        ),
        (
            21,
            0,
            10,
            "environment completed update must be in 1..=total updates",
        ),
        (
            1,
            0,
            10,
            "environment start update plus spent updates must equal global update",
        ),
        (
            3,
            0,
            10,
            "environment start update plus spent updates must equal global update",
        ),
    ] {
        assert_eq!(
            state
                .observe(config, limits, completed, wins, games)
                .unwrap_err(),
            PpoError::InvalidTransition(message)
        );
        assert_eq!(state, saved);
    }
}

#[test]
fn effective_budget_checks_numeric_overflow_before_conversion_without_clamping() {
    let config = AdaptiveEnvironmentConfig {
        extension: EnvironmentDecimal::from_units(MAX_TRAINING_COUNTER * EnvironmentDecimal::SCALE),
        ..Default::default()
    };
    let limits = limits(1, 10, 0);
    let original = AdaptiveEnvironmentState::default();
    assert_eq!(
        original.observe(config, limits, 1, 0, 10).unwrap_err(),
        PpoError::InvalidTransition("environment effective budget exceeds MAX_TRAINING_COUNTER")
    );
    assert_eq!(original, AdaptiveEnvironmentState::default());
    let large = AdaptiveEnvironmentState {
        updates_in_generation: MAX_TRAINING_COUNTER,
        extension_awards: MAX_TRAINING_COUNTER,
        ..Default::default()
    };
    assert_eq!(
        large.effective_budget(config, limits).unwrap_err(),
        PpoError::InvalidTransition("environment effective budget exceeds MAX_TRAINING_COUNTER")
    );
    let excessive = AdaptiveEnvironmentState {
        extension_awards: u64::MAX,
        ..original
    };
    assert_eq!(
        excessive.effective_budget(config, limits).unwrap_err(),
        PpoError::InvalidTransition("environment extension awards exceed MAX_TRAINING_COUNTER")
    );
}

#[test]
fn maximum_budget_and_global_counters_are_valid_without_final_increment() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(1, MAX_TRAINING_COUNTER, 0);
    let state = AdaptiveEnvironmentState {
        generation: MAX_TRAINING_COUNTER - 2,
        start_update: MAX_TRAINING_COUNTER - 2,
        ..Default::default()
    };
    let next = state
        .observe(config, limits, MAX_TRAINING_COUNTER - 1, 5, 10)
        .unwrap();
    assert_eq!(next.generation, MAX_TRAINING_COUNTER - 1);
    let terminal = next
        .observe(config, limits, MAX_TRAINING_COUNTER, 5, 10)
        .unwrap();
    assert_eq!(terminal.generation, MAX_TRAINING_COUNTER - 1);
    assert_eq!(terminal.updates_in_generation, 1);
    let maximum = AdaptiveEnvironmentLimits {
        base_updates: MAX_TRAINING_COUNTER,
        ..limits
    };
    assert_eq!(
        state.effective_budget(config, maximum).unwrap(),
        MAX_TRAINING_COUNTER
    );
}

#[test]
fn validation_rejects_global_addition_overflow_and_mismatched_progress() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(10, 20, 0);
    for (state, global, message) in [
        (
            AdaptiveEnvironmentState {
                start_update: u64::MAX,
                updates_in_generation: 1,
                ..Default::default()
            },
            0,
            "environment global update addition overflow",
        ),
        (
            AdaptiveEnvironmentState::default(),
            1,
            "environment start update plus spent updates must equal global update",
        ),
        (
            AdaptiveEnvironmentState {
                updates_in_generation: 21,
                ..Default::default()
            },
            21,
            "environment global update exceeds total updates",
        ),
    ] {
        assert_eq!(
            state.validate(config, limits, global).unwrap_err(),
            PpoError::InvalidTransition(message)
        );
    }
}

#[test]
fn validation_rejects_impossible_generation_and_terminal_clean_state() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(10, 20, 0);
    for (generation, start_update, message) in [
        (
            1,
            0,
            "environment generation and start update are inconsistent",
        ),
        (
            0,
            1,
            "environment generation and start update are inconsistent",
        ),
        (
            3,
            2,
            "environment generation and start update are inconsistent",
        ),
        (1, 20, "environment start update must precede total updates"),
    ] {
        let state = AdaptiveEnvironmentState {
            generation,
            start_update,
            ..Default::default()
        };
        assert_eq!(
            state.validate(config, limits, start_update).unwrap_err(),
            PpoError::InvalidTransition(message)
        );
    }
}

#[test]
fn validation_rejects_streaks_exceeding_their_windows_or_spent_updates() {
    let config = AdaptiveEnvironmentConfig {
        poor_updates: 2,
        ..Default::default()
    };
    let limits = limits(10, 20, 0);
    for (spent, success_streak, poor_streak, message) in [
        (
            0,
            1,
            0,
            "environment success streak exceeds its window or spent updates",
        ),
        (
            3,
            3,
            0,
            "environment success streak exceeds its window or spent updates",
        ),
        (
            0,
            0,
            1,
            "environment poor streak exceeds its window or spent updates",
        ),
        (
            3,
            0,
            3,
            "environment poor streak exceeds its window or spent updates",
        ),
    ] {
        let state = AdaptiveEnvironmentState {
            updates_in_generation: spent,
            success_streak,
            poor_streak,
            ..Default::default()
        };
        assert_eq!(
            state.validate(config, limits, spent).unwrap_err(),
            PpoError::InvalidTransition(message)
        );
    }
}

#[test]
fn validation_rejects_awards_inconsistent_with_qualifying_updates() {
    let config = AdaptiveEnvironmentConfig {
        poor_updates: 2,
        ..Default::default()
    };
    let limits = limits(10, 20, 0);
    for (spent, poor_streak, extension_awards, message) in [
        (
            0,
            0,
            1,
            "environment extension awards exceed qualifying updates",
        ),
        (
            2,
            2,
            2,
            "environment extension awards exceed qualifying updates",
        ),
        (
            2,
            2,
            0,
            "environment qualifying poor streak is missing an extension award",
        ),
    ] {
        let state = AdaptiveEnvironmentState {
            updates_in_generation: spent,
            poor_streak,
            extension_awards,
            ..Default::default()
        };
        assert_eq!(
            state.validate(config, limits, spent).unwrap_err(),
            PpoError::InvalidTransition(message)
        );
    }
}

#[test]
fn observation_rejects_disjoint_active_streaks_without_mutating_invalid_state() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(10, 20, 0);
    let state = AdaptiveEnvironmentState {
        updates_in_generation: 1,
        success_streak: 1,
        poor_streak: 1,
        extension_awards: 1,
        ..Default::default()
    };
    let saved = state;
    let expected = PpoError::InvalidTransition(
        "environment disjoint thresholds cannot both have active streaks",
    );
    assert_eq!(state.validate(config, limits, 1).unwrap_err(), expected);
    assert_eq!(
        state.observe(config, limits, 2, 5, 10).unwrap_err(),
        expected
    );
    assert_eq!(state, saved);
}

#[test]
fn validation_rejects_a_missed_zero_phase_boundary_even_at_total() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(20, 10, 3);
    for spent in [7, 10] {
        let state = AdaptiveEnvironmentState {
            updates_in_generation: spent,
            ..Default::default()
        };
        assert_eq!(
            state.validate(config, limits, spent).unwrap_err(),
            PpoError::InvalidTransition(
                "environment state crosses the zero phase boundary without a reset"
            )
        );
    }
}

#[test]
fn invalid_config_and_limits_are_rejected_by_all_controller_entry_points() {
    let state = AdaptiveEnvironmentState::default();
    for (config, limits, message) in [
        (
            AdaptiveEnvironmentConfig {
                success_updates: 0,
                ..Default::default()
            },
            limits(2, 10, 0),
            "environment success updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            AdaptiveEnvironmentConfig::default(),
            AdaptiveEnvironmentLimits {
                base_updates: 0,
                total_updates: 10,
                zero_updates: 0,
            },
            "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
    ] {
        let expected = PpoError::InvalidConfig(message);
        assert_eq!(state.validate(config, limits, 0).unwrap_err(), expected);
        assert_eq!(
            state.effective_budget(config, limits).unwrap_err(),
            expected
        );
        assert_eq!(
            state.observe(config, limits, 1, 0, 10).unwrap_err(),
            expected
        );
    }
}
