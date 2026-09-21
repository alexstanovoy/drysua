use super::*;

fn training_arguments(operation: &str) -> Vec<&str> {
    let mut arguments = vec![
        "drysua",
        operation,
        "--updates",
        "8",
        "--checkpoint-directory",
        "unused-checkpoints",
    ];
    if operation == "train-annealed" {
        arguments.extend(["--generation-games", "8", "--parallel", "2"]);
    }
    arguments
}

#[test]
fn training_metrics_default_to_disabled_for_both_trainers() {
    for operation in ["train-full", "train-annealed"] {
        let cli = Cli::try_parse_from(training_arguments(operation)).expect("training CLI");
        let metrics = match cli.operation {
            Some(Operation::TrainFull(train)) => train.metrics,
            Some(Operation::TrainAnnealed(train)) => train.metrics,
            _ => panic!("expected training arguments"),
        };

        assert!(metrics.metrics_directory.is_none());
        assert!(metrics.metrics_listen.is_none());
    }
}

#[test]
fn training_metrics_directory_and_listener_are_independent_options() {
    for operation in ["train-full", "train-annealed"] {
        for (directory, listener) in [(true, false), (false, true), (true, true)] {
            let mut arguments = training_arguments(operation);
            if directory {
                arguments.extend(["--metrics-directory", "unused-metrics"]);
            }
            if listener {
                arguments.extend(["--metrics-listen", "127.0.0.1:9470"]);
            }

            let cli = Cli::try_parse_from(arguments).expect("metrics training CLI");
            let metrics = match cli.operation {
                Some(Operation::TrainFull(train)) => train.metrics,
                Some(Operation::TrainAnnealed(train)) => train.metrics,
                _ => panic!("expected training arguments"),
            };

            assert_eq!(
                metrics.metrics_directory,
                directory.then(|| std::path::PathBuf::from("unused-metrics"))
            );
            assert_eq!(
                metrics.metrics_listen,
                listener.then(|| "127.0.0.1:9470".parse().expect("socket"))
            );
        }
    }
}

#[test]
fn metrics_serve_requires_a_directory_and_defaults_to_loopback_without_builtin_requirement() {
    let cli = Cli::try_parse_from([
        "drysua",
        "metrics-serve",
        "--metrics-directory",
        "unused-metrics",
    ])
    .expect("standalone metrics CLI with or without builtin");
    let Some(Operation::MetricsServe(serve)) = cli.operation else {
        panic!("expected metrics-serve arguments");
    };

    assert_eq!(
        serve.metrics_directory,
        std::path::Path::new("unused-metrics")
    );
    assert_eq!(
        serve.metrics_listen,
        "127.0.0.1:9464".parse().expect("socket")
    );
    let error = parse_from(["drysua", "metrics-serve"]).expect_err("directory is required");
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
    assert!(
        error
            .to_string()
            .contains("--metrics-directory <METRICS_DIRECTORY>")
    );
}

#[test]
fn metrics_serve_accepts_an_explicit_listener() {
    let cli = Cli::try_parse_from([
        "drysua",
        "metrics-serve",
        "--metrics-directory",
        "unused-metrics",
        "--metrics-listen",
        "[::1]:9470",
    ])
    .expect("explicit standalone listener");
    let Some(Operation::MetricsServe(serve)) = cli.operation else {
        panic!("expected metrics-serve arguments");
    };

    assert_eq!(serve.metrics_listen, "[::1]:9470".parse().expect("socket"));
}

#[test]
fn metrics_listener_rejects_invalid_socket_addresses_before_execution() {
    for operation in ["train-full", "train-annealed", "metrics-serve"] {
        let mut arguments = if operation == "metrics-serve" {
            vec!["drysua", operation, "--metrics-directory", "unused-metrics"]
        } else {
            training_arguments(operation)
        };
        arguments.extend(["--metrics-listen", "not-a-socket"]);

        let error = parse_from(arguments).expect_err("invalid socket");

        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
        assert!(
            error
                .to_string()
                .contains("invalid value 'not-a-socket' for '--metrics-listen <METRICS_LISTEN>'")
        );
    }
}

#[test]
fn metrics_flags_do_not_leak_into_play_smoke_or_reward_observer() {
    for operation in [None, Some("play"), Some("train"), Some("reward-observer")] {
        for (flag, value) in [
            ("--metrics-directory", "unused-metrics"),
            ("--metrics-listen", "127.0.0.1:9464"),
        ] {
            let mut arguments = vec!["drysua"];
            arguments.extend(operation);
            arguments.extend([flag, value]);

            let error = parse_from(arguments).expect_err("training-only options");

            assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
            assert!(
                error
                    .to_string()
                    .contains(&format!("unexpected argument '{flag}'"))
            );
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_metrics_options_leave_training_scope_config_and_seed_unchanged() {
    let plain = training_settings_for_test(&[]).expect("plain settings");
    let observed = training_settings_for_test(&[
        "--metrics-directory",
        "unused-metrics",
        "--metrics-listen",
        "127.0.0.1:9464",
    ])
    .expect("observed settings");
    let scope = |settings: &crate::TrainingJobConfig| {
        crate::ppo_arena::training_checkpoint_run(settings, crate::PolicyDevice::Cpu, settings.ppo)
            .expect("canonical scope")
    };

    assert_eq!(plain, observed);
    assert_eq!(scope(&plain), scope(&observed));
    let mut extended = observed;
    extended.updates += 1;
    assert_eq!(scope(&plain), scope(&extended));
}
