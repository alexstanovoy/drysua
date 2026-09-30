use super::*;
use crate::{AdaptiveEnvironmentConfig, EnvironmentDecimal, EnvironmentSchedule};

#[test]
fn adaptive_environment_cli_defaults_all_five_values_even_on_resume() {
    let expected = EnvironmentSchedule::Adaptive(AdaptiveEnvironmentConfig {
        success_updates: 2,
        success_rate: EnvironmentDecimal::from_units(800_000),
        poor_updates: 1,
        poor_rate: EnvironmentDecimal::from_units(200_000),
        extension: EnvironmentDecimal::from_units(750_000),
    });
    for flags in [
        vec![],
        vec!["--resume"],
        vec!["--environment-schedule", "adaptive"],
    ] {
        assert_eq!(
            schedule_for_test(&flags).expect("adaptive defaults"),
            expected
        );
    }
}

#[test]
fn adaptive_environment_cli_preserves_all_five_explicit_nondefault_values() {
    let actual = schedule_for_test(&[
        "--environment-success-updates",
        "3",
        "--environment-success-rate",
        ".875001",
        "--environment-poor-updates",
        "4",
        "--environment-poor-rate",
        ".125002",
        "--environment-extension",
        "1.250003",
    ])
    .expect("explicit adaptive configuration");
    assert_eq!(
        actual,
        EnvironmentSchedule::Adaptive(AdaptiveEnvironmentConfig {
            success_updates: 3,
            success_rate: EnvironmentDecimal::from_units(875_001),
            poor_updates: 4,
            poor_rate: EnvironmentDecimal::from_units(125_002),
            extension: EnvironmentDecimal::from_units(1_250_003),
        })
    );
}

#[test]
fn adaptive_environment_cli_accepts_extension_at_and_above_one_without_local_cap() {
    for (text, units) in [
        ("0", 0),
        ("1", 1_000_000),
        ("1.25", 1_250_000),
        ("20.000001", 20_000_001),
    ] {
        let EnvironmentSchedule::Adaptive(config) =
            schedule_for_test(&["--environment-extension", text]).expect(text)
        else {
            panic!("default schedule must be adaptive");
        };
        assert_eq!(config.extension.units(), units);
    }
}

#[test]
fn adaptive_environment_cli_equivalent_decimals_have_exact_canonical_scope() {
    let expected = concat!(
        " --environment-schedule adaptive --environment-success-updates 2",
        " --environment-success-rate 0.8 --environment-poor-updates 1",
        " --environment-poor-rate 0.2 --environment-extension 0.75"
    );
    for (success, poor, extension) in [
        (".8", ".2", ".75"),
        ("0.800000", "0.200000", "0.750000"),
        ("00.80", "00.20", "00.750"),
    ] {
        let schedule = schedule_for_test(&[
            "--environment-success-rate",
            success,
            "--environment-poor-rate",
            poor,
            "--environment-extension",
            extension,
        ])
        .expect("equivalent exact decimals");
        let EnvironmentSchedule::Adaptive(config) = schedule else {
            panic!("default schedule must be adaptive");
        };
        assert_eq!(config.scope_suffix(), expected);
        assert_eq!(
            schedule,
            EnvironmentSchedule::Adaptive(AdaptiveEnvironmentConfig::default())
        );
    }
}

#[test]
fn adaptive_environment_cli_fixed_rejects_every_explicit_knob_including_defaults() {
    assert_eq!(
        schedule_for_test(&["--environment-schedule", "fixed"]).expect("legacy fixed"),
        EnvironmentSchedule::Fixed
    );
    for (flag, value) in [
        ("--environment-success-updates", "2"),
        ("--environment-success-rate", ".8"),
        ("--environment-poor-updates", "1"),
        ("--environment-poor-rate", ".2"),
        ("--environment-extension", ".75"),
    ] {
        let error =
            schedule_for_test(&["--environment-schedule", "fixed", flag, value]).expect_err(flag);
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "environment tuning options require --environment-schedule adaptive"
        );
    }
}

#[test]
fn adaptive_environment_cli_rejects_nonexact_decimal_syntax_and_overprecision() {
    for flag in [
        "--environment-success-rate",
        "--environment-poor-rate",
        "--environment-extension",
    ] {
        for (value, reason) in [
            ("NaN", "must use unsigned plain decimal notation"),
            ("inf", "must use unsigned plain decimal notation"),
            ("-inf", "must use unsigned plain decimal notation"),
            ("+0.8", "must use unsigned plain decimal notation"),
            ("-0", "must use unsigned plain decimal notation"),
            ("-0.2", "must use unsigned plain decimal notation"),
            ("8e-1", "must use unsigned plain decimal notation"),
            ("0.8 ", "must use unsigned plain decimal notation"),
            ("1.", "must use unsigned plain decimal notation"),
            ("", "must use unsigned plain decimal notation"),
            ("0.8000000", "has more than six fractional digits"),
            ("18446744073710", "exceeds u64 millionths"),
            ("0000000000000000000000000000", "exceeds 27 bytes"),
        ] {
            let argument = format!("{flag}={value}");
            let error = schedule_for_test(&[&argument])
                .expect_err(&argument)
                .to_string();
            assert!(error.contains(flag), "{error}");
            assert!(
                error.contains(&format!("environment decimal {reason}")),
                "{error}"
            );
        }
    }
}

#[test]
fn adaptive_environment_cli_rates_accept_endpoints_and_reject_above_one() {
    for (flag, name) in [
        ("--environment-success-rate", "success"),
        ("--environment-poor-rate", "poor"),
    ] {
        for value in ["0", "1", "1.000000"] {
            schedule_for_test(&[flag, value]).expect("inclusive rate endpoints");
        }
        let error = schedule_for_test(&[flag, "1.000001"]).expect_err("rate overflow");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            format!("invalid PPO config field: environment {name} rate must be in [0, 1]")
        );
    }
}

#[test]
fn adaptive_environment_cli_windows_enforce_positive_global_counter_bounds() {
    let maximum = crate::MAX_TRAINING_COUNTER.to_string();
    let overflow = (crate::MAX_TRAINING_COUNTER + 1).to_string();
    for flag in [
        "--environment-success-updates",
        "--environment-poor-updates",
    ] {
        for value in ["1", &maximum] {
            schedule_for_test(&[flag, value]).expect("window endpoint");
        }
        for value in ["0", &overflow] {
            let error = schedule_for_test(&[flag, value])
                .expect_err("window bound")
                .to_string();
            assert!(error.contains(flag), "{error}");
            assert!(
                error.contains(&format!("{value} is not in 1..={maximum}")),
                "{error}"
            );
        }
    }
}

#[test]
fn adaptive_environment_cli_extension_enforces_full_exact_global_bound() {
    let maximum = crate::MAX_TRAINING_COUNTER.to_string();
    schedule_for_test(&["--environment-extension", &maximum]).expect("global extension bound");
    for value in [
        format!("{maximum}.000001"),
        (crate::MAX_TRAINING_COUNTER + 1).to_string(),
    ] {
        let error =
            schedule_for_test(&["--environment-extension", &value]).expect_err("extension bound");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "invalid PPO config field: environment extension must not exceed MAX_TRAINING_COUNTER"
        );
    }
}

#[test]
fn adaptive_environment_cli_requires_generation_updates_and_rejects_unknown_schedule() {
    let error = parse_annealed(&["--updates", "20"])
        .err()
        .expect("required generation updates");
    assert!(
        error
            .to_string()
            .contains("--generation-updates <GENERATION_UPDATES>"),
        "{error}"
    );
    let error =
        schedule_for_test(&["--environment-schedule", "automatic"]).expect_err("unknown schedule");
    assert!(
        error
            .to_string()
            .contains("invalid value 'automatic' for '--environment-schedule"),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("possible values: adaptive, fixed"),
        "{error}"
    );
}

#[test]
fn adaptive_environment_cli_validates_base_total_and_zero_update_limits() {
    let overflow = (crate::MAX_TRAINING_COUNTER + 1).to_string();
    let base_overflow = (crate::MAX_TRAINING_COUNTER + 1).to_string();
    for (total, generation, zero, field) in [
        (
            "0",
            "2",
            "0",
            "environment total updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            &overflow,
            "2",
            "0",
            "environment total updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            "20",
            &base_overflow,
            "0",
            "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
        ),
        (
            "20",
            "2",
            "21",
            "environment zero updates must not exceed total updates",
        ),
    ] {
        let train = parse_annealed(&[
            "--updates",
            total,
            "--generation-updates",
            generation,
            "--zero-updates",
            zero,
        ])
        .unwrap_or_else(|error| panic!("parse limits: {error}"));
        let error = train.environment_schedule().expect_err(field);
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            format!("invalid PPO config field: {field}")
        );
    }
}

#[test]
fn adaptive_environment_cli_accepts_base_above_total_and_zero_phase_endpoints() {
    for zero in ["0", "20"] {
        parse_annealed(&[
            "--updates",
            "20",
            "--generation-updates",
            "21",
            "--zero-updates",
            zero,
        ])
        .unwrap_or_else(|error| panic!("parse limits: {error}"))
        .environment_schedule()
        .expect("base may exceed total; zero may cover none or all");
    }
}

#[cfg(feature = "builtin")]
#[test]
fn adaptive_environment_cli_settings_count_generations_in_updates() {
    for schedule in ["adaptive", "fixed"] {
        let settings = annealed_settings_for_test(&[
            "--updates",
            "20",
            "--generation-updates",
            "3",
            "--environment-schedule",
            schedule,
        ])
        .expect("generation in updates");
        assert_eq!(settings.generation_updates, 3);
    }
}

fn schedule_for_test(flags: &[&str]) -> std::io::Result<EnvironmentSchedule> {
    let mut arguments = vec!["--updates", "20", "--generation-updates", "4"];
    arguments.extend_from_slice(flags);
    parse_annealed(&arguments)?.environment_schedule()
}

fn parse_annealed(flags: &[&str]) -> std::io::Result<TrainAnnealedArgs> {
    let mut arguments = vec!["drysua", "train-annealed", "--checkpoint-directory", "."];
    arguments.extend_from_slice(flags);
    let cli = Cli::try_parse_from(arguments).map_err(std::io::Error::other)?;
    let Some(Operation::TrainAnnealed(train)) = cli.operation else {
        panic!("expected annealed arguments");
    };
    Ok(train)
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_resolves_environment_scale_endpoints() {
    let base = ["--updates", "20", "--generation-updates", "4"];
    let settings = |extra: &[&str]| {
        let mut arguments = base.to_vec();
        arguments.extend_from_slice(extra);
        crate::cli::annealed_settings_for_test(&arguments)
    };
    assert_eq!(
        settings(&[]).expect("default scale").scale,
        crate::randomization::AnnealScale::FULL
    );
    assert_eq!(
        settings(&[
            "--environment-scale-start",
            "0",
            "--environment-scale-end",
            "2"
        ])
        .expect("rising ramp")
        .scale,
        crate::randomization::AnnealScale {
            start_bp: 0,
            end_bp: 20_000
        }
    );
    assert_eq!(
        settings(&[
            "--environment-scale-start",
            "0.5",
            "--environment-scale-end",
            "0.5"
        ])
        .expect("fixed half scale")
        .scale,
        crate::randomization::AnnealScale {
            start_bp: 5_000,
            end_bp: 5_000
        }
    );
    assert_eq!(
        settings(&[
            "--environment-scale-start",
            "10",
            "--environment-scale-end",
            "10"
        ])
        .expect("ten times full variance")
        .scale,
        crate::randomization::AnnealScale {
            start_bp: 100_000,
            end_bp: 100_000
        }
    );
    // The loop works in basis points, so an endpoint truncates to a hundredth.
    assert_eq!(
        settings(&["--environment-scale-start", "0.007"])
            .expect("zero point seven percent")
            .scale
            .start_bp,
        70
    );
    assert_eq!(
        settings(&["--environment-scale-start", "0.00005"])
            .expect("sub-basis-point")
            .scale
            .start_bp,
        0
    );
    for value in ["10.000001", "11"] {
        let error = settings(&["--environment-scale-start", value]).expect_err("above the bound");
        assert!(
            error.to_string().contains("must be within 0 and 10"),
            "{error}"
        );
    }
    // The equals form keeps a leading sign with the value, so the decimal
    // parser rejects it instead of clap treating it as an unknown flag.
    for value in ["-1", "1e1", "abc", "1.", ".5.5"] {
        let argument = format!("--environment-scale-end={value}");
        let error = settings(&[argument.as_str()]).expect_err("notation");
        assert!(
            error.to_string().contains("environment decimal"),
            "{value}: {error}"
        );
    }
}
