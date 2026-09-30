use super::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
    EnvironmentDecimal, EnvironmentSchedule,
};
use crate::{MAX_TRAINING_COUNTER, PpoError};

fn decimal(text: &str) -> EnvironmentDecimal {
    text.parse().expect("valid test decimal")
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
        assert_eq!(value.units(), units, "{text}");
        assert_eq!(value.to_string(), canonical, "{text}");
    }
}

#[test]
fn decimal_codec_units_round_trip_without_config_restrictions() {
    for units in [0, 1, 100_000, 1_000_000, u64::MAX] {
        let value = EnvironmentDecimal::from_units(units);
        assert_eq!(value.units(), units);
        assert_eq!(decimal(&value.to_string()), value);
    }
}

#[test]
fn decimal_rejects_non_plain_unsigned_notation() {
    for text in [
        "", ".", "1.", "-0", "-1", "NaN", "nan", "inf", "infinity", "+1", "1e0", "1E2", " 1", "1 ",
        "1..2", "１", "0_1", "0,1",
    ] {
        let error = text.parse::<EnvironmentDecimal>().unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid PPO config field: environment decimal must use unsigned plain decimal notation",
            "{text}"
        );
    }
}

#[test]
fn decimal_rejects_overprecision_even_when_extra_digits_are_zero() {
    for text in ["0.0000001", "1.0000000"] {
        assert_eq!(
            text.parse::<EnvironmentDecimal>().unwrap_err(),
            PpoError::InvalidConfig("environment decimal has more than six fractional digits")
        );
    }
}

#[test]
fn decimal_rejects_overflow_and_bounds_input_length() {
    for text in [
        "18446744073709.551616",
        "18446744073710",
        "18446744073709551616",
    ] {
        assert_eq!(
            text.parse::<EnvironmentDecimal>().unwrap_err(),
            PpoError::InvalidConfig("environment decimal exceeds u64 millionths")
        );
    }
    assert_eq!(
        "0000000000000000000000000000"
            .parse::<EnvironmentDecimal>()
            .unwrap_err(),
        PpoError::InvalidConfig("environment decimal exceeds 27 bytes")
    );
}

#[test]
fn defaults_select_adaptive_with_the_agreed_thresholds() {
    let config = AdaptiveEnvironmentConfig::default();
    assert_eq!(config.success_updates, 2);
    assert_eq!(config.success_rate, decimal(".8"));
    assert_eq!(config.poor_updates, 1);
    assert_eq!(config.poor_rate, decimal(".2"));
    assert_eq!(config.extension, decimal(".75"));
    assert_eq!(
        EnvironmentSchedule::default(),
        EnvironmentSchedule::Adaptive(config)
    );
    assert_ne!(EnvironmentSchedule::Fixed, EnvironmentSchedule::default());
}

#[test]
fn scope_suffix_is_canonical_and_append_preserves_the_prefix() {
    let config = AdaptiveEnvironmentConfig {
        success_rate: decimal("00.800000"),
        poor_rate: decimal(".200"),
        extension: decimal("01.250000"),
        ..AdaptiveEnvironmentConfig::default()
    };
    let suffix = concat!(
        " --environment-schedule adaptive --environment-success-updates 2",
        " --environment-success-rate 0.8 --environment-poor-updates 1",
        " --environment-poor-rate 0.2 --environment-extension 1.25"
    );
    assert_eq!(config.scope_suffix(), suffix);
    let mut scope = String::from("existing scope");
    config.append_scope(&mut scope);
    assert_eq!(scope, format!("existing scope{suffix}"));
}

#[test]
fn config_rejects_zero_and_out_of_bound_windows() {
    for window in [0, MAX_TRAINING_COUNTER + 1] {
        for (success_updates, poor_updates, message) in [
            (
                window,
                1,
                "environment success updates must be in 1..=MAX_TRAINING_COUNTER",
            ),
            (
                2,
                window,
                "environment poor updates must be in 1..=MAX_TRAINING_COUNTER",
            ),
        ] {
            let config = AdaptiveEnvironmentConfig {
                success_updates,
                poor_updates,
                ..AdaptiveEnvironmentConfig::default()
            };
            assert_eq!(
                config.validate().unwrap_err(),
                PpoError::InvalidConfig(message)
            );
        }
    }
}

#[test]
fn config_rejects_rates_above_one_but_allows_overlapping_thresholds() {
    for (success_rate, poor_rate, message) in [
        (
            decimal("1.000001"),
            decimal(".2"),
            "environment success rate must be in [0, 1]",
        ),
        (
            decimal(".8"),
            decimal("1.000001"),
            "environment poor rate must be in [0, 1]",
        ),
    ] {
        let config = AdaptiveEnvironmentConfig {
            success_rate,
            poor_rate,
            ..AdaptiveEnvironmentConfig::default()
        };
        assert_eq!(
            config.validate().unwrap_err(),
            PpoError::InvalidConfig(message)
        );
    }
    let config = AdaptiveEnvironmentConfig {
        success_rate: decimal("0"),
        poor_rate: decimal("1"),
        ..AdaptiveEnvironmentConfig::default()
    };
    assert_eq!(config.validate().unwrap(), config);
}

#[test]
fn config_accepts_numeric_maxima_without_a_smaller_environment_cap() {
    let maximum_units = MAX_TRAINING_COUNTER * EnvironmentDecimal::SCALE;
    let config = AdaptiveEnvironmentConfig {
        success_updates: MAX_TRAINING_COUNTER,
        poor_updates: MAX_TRAINING_COUNTER,
        extension: EnvironmentDecimal::from_units(maximum_units),
        ..AdaptiveEnvironmentConfig::default()
    };
    assert_eq!(config.validate().unwrap(), config);
    for units in [maximum_units + 1, u64::MAX] {
        let invalid = AdaptiveEnvironmentConfig {
            extension: EnvironmentDecimal::from_units(units),
            ..config
        };
        assert_eq!(
            invalid.validate().unwrap_err(),
            PpoError::InvalidConfig("environment extension must not exceed MAX_TRAINING_COUNTER")
        );
    }
}

#[test]
fn limits_reject_invalid_counters_and_allow_base_above_total() {
    for (base_updates, total_updates, zero_updates, message) in [
        (
            0,
            10,
            0,
            "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            MAX_TRAINING_COUNTER + 1,
            10,
            0,
            "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            1,
            0,
            0,
            "environment total updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            1,
            MAX_TRAINING_COUNTER + 1,
            0,
            "environment total updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            1,
            10,
            11,
            "environment zero updates must not exceed total updates",
        ),
    ] {
        let limits = AdaptiveEnvironmentLimits {
            base_updates,
            total_updates,
            zero_updates,
        };
        assert_eq!(
            limits.validate().unwrap_err(),
            PpoError::InvalidConfig(message)
        );
    }
    assert_eq!(
        limits(MAX_TRAINING_COUNTER, 1, 1).base_updates,
        MAX_TRAINING_COUNTER
    );
}

#[test]
fn success_requires_consecutive_individually_qualifying_updates_not_an_average() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(20, 100, 0);
    let mixed = run(config, limits, &[7, 9]);
    assert_eq!(mixed.generation, 0);
    assert_eq!(mixed.success_streak, 1);
    let qualifying = run(config, limits, &[8, 8]);
    assert_eq!(
        qualifying,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 2,
            ..AdaptiveEnvironmentState::default()
        }
    );
    let interrupted = run(config, limits, &[8, 7, 8]);
    assert_eq!(interrupted.generation, 0);
    assert_eq!(interrupted.success_streak, 1);
}

#[test]
fn poor_streak_awards_each_overlapping_qualifying_update_not_an_average() {
    let config = AdaptiveEnvironmentConfig {
        poor_updates: 2,
        ..AdaptiveEnvironmentConfig::default()
    };
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
fn three_quarter_credits_extend_base_four_to_exactly_five_updates() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(4, 20, 0);
    let before = run(config, limits, &[0, 0, 5, 5]);
    assert_eq!(before.generation, 0);
    assert_eq!(before.extension_awards, 2);
    assert_eq!(before.updates_in_generation, 4);
    assert_eq!(before.effective_budget(config, limits).unwrap(), 5);
    let after = before.observe(config, limits, 5, 5, 10).unwrap();
    assert_eq!(
        after,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 5,
            ..AdaptiveEnvironmentState::default()
        }
    );
    assert_eq!(after.effective_budget(config, limits).unwrap(), 4);
}

#[test]
fn ten_tenth_credits_add_exactly_one_update_without_float_rounding() {
    let config = AdaptiveEnvironmentConfig {
        extension: decimal(".1"),
        ..AdaptiveEnvironmentConfig::default()
    };
    let limits = limits(20, 100, 0);
    let nine = run(config, limits, &[0; 9]);
    assert_eq!(nine.effective_budget(config, limits).unwrap(), 20);
    let ten = nine.observe(config, limits, 10, 0, 10).unwrap();
    assert_eq!(ten.extension_awards, 10);
    assert_eq!(ten.effective_budget(config, limits).unwrap(), 21);
}

#[test]
fn zero_extension_still_records_awards_and_exhausts_the_base_budget() {
    let config = AdaptiveEnvironmentConfig {
        extension: decimal("0"),
        ..AdaptiveEnvironmentConfig::default()
    };
    let limits = limits(2, 10, 0);
    let first = run(config, limits, &[0]);
    assert_eq!(first.extension_awards, 1);
    let second = first.observe(config, limits, 2, 0, 10).unwrap();
    assert_eq!(
        second,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 2,
            ..AdaptiveEnvironmentState::default()
        }
    );
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
            assert_eq!(state.generation, 0, "{extension}, window {poor_updates}");
            assert_eq!(state.updates_in_generation, 12);
            assert_eq!(state.poor_streak, poor_updates);
            assert_eq!(state.extension_awards, 13 - poor_updates);
            assert!(state.effective_budget(config, limits).unwrap() > limits.total_updates);
        }
    }
}

#[test]
fn exhausted_budget_resets_incomplete_success_and_poor_windows() {
    let config = AdaptiveEnvironmentConfig {
        success_updates: 3,
        poor_updates: 3,
        extension: decimal("10"),
        ..Default::default()
    };
    let limits = limits(2, 10, 0);
    for wins in [0, 8] {
        let reset = run(config, limits, &[wins; 2]);
        assert_eq!(
            reset,
            AdaptiveEnvironmentState {
                generation: 1,
                start_update: 2,
                ..Default::default()
            }
        );
        let next = reset.observe(config, limits, 3, wins, 10).unwrap();
        assert_eq!(next.success_streak, u64::from(wins == 8));
        assert_eq!(next.poor_streak, u64::from(wins == 0));
        assert_eq!(next.extension_awards, 0);
    }
}

#[test]
fn success_overrides_existing_extension_credit_and_resets_all_local_counters() {
    let config = AdaptiveEnvironmentConfig {
        extension: decimal("10"),
        ..Default::default()
    };
    let limits = limits(1, 20, 0);
    let incomplete = run(config, limits, &[0, 8]);
    assert_eq!(incomplete.success_streak, 1);
    assert_eq!(incomplete.poor_streak, 0);
    assert_eq!(incomplete.extension_awards, 1);
    let next = incomplete.observe(config, limits, 3, 8, 10).unwrap();
    assert_eq!(
        next,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 3,
            ..Default::default()
        }
    );
}

#[test]
fn overlapping_success_has_priority_over_a_poor_award_that_would_overflow() {
    let config = AdaptiveEnvironmentConfig {
        success_rate: decimal(".2"),
        poor_rate: decimal(".8"),
        extension: EnvironmentDecimal::from_units(
            (MAX_TRAINING_COUNTER - 1) * EnvironmentDecimal::SCALE,
        ),
        ..Default::default()
    };
    let state = run(config, limits(1, 10, 0), &[5, 5]);
    assert_eq!(
        state,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 2,
            ..Default::default()
        }
    );
    let terminal = run(config, limits(1, 2, 0), &[5, 5]);
    assert_eq!(terminal.success_streak, 2);
    assert_eq!(terminal.poor_streak, 1);
    assert_eq!(terminal.extension_awards, 1);
    assert_eq!(terminal.generation, 0);
}

#[test]
fn zero_phase_boundary_forces_a_clean_environment_despite_extension_credit() {
    let config = AdaptiveEnvironmentConfig {
        extension: decimal("10"),
        ..Default::default()
    };
    let limits = limits(10, 8, 3);
    let boundary = run(config, limits, &[0; 5]);
    assert_eq!(
        boundary,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 5,
            ..Default::default()
        }
    );
    let after = boundary.observe(config, limits, 6, 0, 10).unwrap();
    assert_eq!(after.generation, 1);
    assert_eq!(after.updates_in_generation, 1);
    assert_eq!(after.extension_awards, 1);
}

#[test]
fn coincident_boundary_success_and_exhaustion_advance_only_once() {
    let config = AdaptiveEnvironmentConfig {
        extension: decimal("0"),
        ..Default::default()
    };
    let limits = limits(5, 8, 3);
    for wins in [[5, 5, 5, 8, 8], [5; 5]] {
        let state = run(config, limits, &wins);
        assert_eq!(
            state,
            AdaptiveEnvironmentState {
                generation: 1,
                start_update: 5,
                ..Default::default()
            }
        );
    }
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
fn terminal_success_is_retained_without_a_needless_environment_transition() {
    let config = AdaptiveEnvironmentConfig {
        success_updates: 1,
        ..Default::default()
    };
    let state = run(config, limits(1, 1, 0), &[10]);
    assert_eq!(state.generation, 0);
    assert_eq!(state.success_streak, 1);
    assert_eq!(state.updates_in_generation, 1);
    assert_eq!(
        state.validate(config, limits(1, 2, 0), 1).unwrap_err(),
        PpoError::InvalidTransition(
            "environment success streak requires advancement before total updates"
        )
    );
}

#[test]
fn terminal_exhaustion_is_valid_only_at_total_and_cannot_be_observed_again() {
    let config = AdaptiveEnvironmentConfig::default();
    let terminal_limits = limits(1, 1, 0);
    let state = run(config, terminal_limits, &[5]);
    assert_eq!(state.generation, 0);
    assert_eq!(state.updates_in_generation, 1);
    assert_eq!(
        state.validate(config, limits(1, 2, 0), 1).unwrap_err(),
        PpoError::InvalidTransition("environment budget is exhausted before total updates")
    );
    assert_eq!(
        state
            .observe(config, terminal_limits, 2, 5, 10)
            .unwrap_err(),
        PpoError::InvalidTransition("environment completed update must be in 1..=total updates")
    );
}

#[test]
fn validation_rejects_spending_beyond_the_budget_even_at_total() {
    let config = AdaptiveEnvironmentConfig::default();
    let limits = limits(1, 2, 0);
    let state = AdaptiveEnvironmentState {
        updates_in_generation: 2,
        ..Default::default()
    };
    assert_eq!(
        state.validate(config, limits, 2).unwrap_err(),
        PpoError::InvalidTransition("environment spent updates exceed effective budget")
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
fn terminal_overlapping_success_does_not_require_a_poor_award() {
    let config = AdaptiveEnvironmentConfig {
        success_updates: 1,
        success_rate: decimal("0"),
        poor_rate: decimal("1"),
        ..Default::default()
    };
    let state = run(config, limits(1, 1, 0), &[5]);
    assert_eq!(state.generation, 0);
    assert_eq!(state.success_streak, 1);
    assert_eq!(state.poor_streak, 1);
    assert_eq!(state.extension_awards, 0);
}

#[test]
fn boundary_discards_a_poor_award_that_would_overflow_the_old_budget() {
    let config = AdaptiveEnvironmentConfig {
        extension: EnvironmentDecimal::from_units(
            (MAX_TRAINING_COUNTER - 1) * EnvironmentDecimal::SCALE,
        ),
        ..Default::default()
    };
    let state = run(config, limits(1, 3, 1), &[0, 0]);
    assert_eq!(
        state,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 2,
            ..Default::default()
        }
    );
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
