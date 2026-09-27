#[test]
fn public_cli_accepts_play_training_and_initialization_contracts() {
    use crate::cli::PlayPolicy::{Hybrid, Neural, Teacher};
    for (arguments, expected) in [
        ("drysua", Teacher),
        ("drysua play", Teacher),
        ("drysua --policy teacher", Teacher),
        ("drysua --weights-directory explicit-experiment", Hybrid),
        (
            "drysua play --weights-directory explicit-experiment",
            Hybrid,
        ),
        (
            "drysua play --policy neural --weights-directory artifacts/test",
            Neural,
        ),
    ] {
        assert_eq!(
            crate::cli::play_policy_for_test(arguments.split_ascii_whitespace()).unwrap(),
            expected
        );
    }
    for arguments in [
        "drysua train --updates 1 --environments 2 --rollout 8 --epochs 1 --minibatch 16 --seed 77 --map 2 --device cuda --device-ordinal 1",
        "drysua train-full --updates 10000 --environments 4 --rollout 8 --epochs 1 --minibatch 32 --checkpoint-seconds 300 --checkpoint-directory artifacts/temp/training --resume --migrate-provenance --device cuda",
        "drysua train-full --updates 8 --checkpoint-directory training/ppo-v1 --initial-weights artifacts/temp/pretrain-v1",
    ] {
        crate::cli::parse_from(arguments.split_ascii_whitespace()).expect(arguments);
    }
}

#[test]
fn public_cli_rejects_invalid_policies_and_conflicting_training_sources() {
    for (arguments, message) in [
        (
            "drysua play --policy idle",
            "invalid value 'idle' for '--policy <POLICY>'",
        ),
        ("drysua --hero 2", "unexpected argument '--hero'"),
        ("drysua --policy neural", "--weights-directory"),
        (
            "drysua train-full --updates 1 --checkpoint-directory . --migrate-provenance",
            "--resume",
        ),
        (
            "drysua train-full --updates 1 --checkpoint-directory . --initial-weights artifacts/test --resume",
            "cannot be used with '--resume'",
        ),
    ] {
        let error =
            crate::cli::parse_from(arguments.split_ascii_whitespace()).expect_err(arguments);
        assert!(error.to_string().contains(message), "{error}");
    }
    for prefix in ["drysua", "drysua play", "drysua play --policy teacher"] {
        let mut arguments: Vec<_> = prefix.split_ascii_whitespace().collect();
        arguments.extend(["--name", ""]);
        assert_eq!(
            crate::cli::run_from_for_test(arguments)
                .unwrap_err()
                .to_string(),
            "bot name must not be empty"
        );
    }
}

#[test]
fn selected_neural_weights_resolve_from_the_repository_not_the_working_directory() {
    use crate::cli::PlayPolicy::{Hybrid, Neural};
    for policy in [Hybrid, Neural] {
        let selection = crate::default_deployment::DefaultDeployment {
            policy,
            weights_directory: Some("artifacts/v9.9.9"),
        };
        assert_eq!(
            selection.resolve().unwrap(),
            (
                policy,
                Some(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/v9.9.9"))
            )
        );
    }
}

#[test]
fn selected_deployment_rejects_policy_and_weight_mismatches() {
    use crate::cli::PlayPolicy::{Hybrid, Neural, Teacher};
    for (policy, weights_directory) in [
        (Teacher, Some("artifacts/v9.9.9")),
        (Neural, None),
        (Hybrid, None),
    ] {
        let error = crate::default_deployment::DefaultDeployment {
            policy,
            weights_directory,
        }
        .resolve()
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "default Teacher must be weights-free; default neural policies must specify weights"
        );
    }
}

#[test]
fn selected_weights_cannot_escape_the_artifact_directory() {
    for directory in [
        "",
        "artifacts",
        "../artifacts/v1",
        "/tmp/weights",
        "artifacts/../weights",
    ] {
        let error = crate::default_deployment::DefaultDeployment {
            policy: crate::cli::PlayPolicy::Neural,
            weights_directory: Some(directory),
        }
        .resolve()
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "default deployment weights must be a repository-relative directory below artifacts"
        );
    }
}

#[test]
fn cli_all_training_commands_default_to_map2_and_reject_other_maps_before_execution() {
    for operation in ["train", "train-full"] {
        let help = crate::cli::parse_from(["drysua", operation, "--help"])
            .unwrap_err()
            .to_string();
        let map_help = help.split("--map <MAP>").nth(1).expect("map option");
        assert!(
            map_help
                .lines()
                .take_while(|line| !line.trim_start().starts_with('-'))
                .any(|line| line.contains("[default: 2]"))
        );
        for map in ["0", "1", "3", "65535"] {
            let mut arguments = vec!["drysua", operation, "--map", map];
            if operation == "train-full" {
                arguments.extend(["--updates", "1", "--checkpoint-directory", "."]);
            }
            let error =
                crate::cli::run_from_for_test(arguments).expect_err("reject before execution");
            assert!(
                error
                    .to_string()
                    .contains(&format!("invalid value '{map}' for '--map <MAP>'"))
            );
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_map2_reward_errors_precede_compiled_provenance_and_checkpoint_access() {
    for (flags, message) in [
        (
            "--terminal-only",
            "Map2 comprehensive reward forbids --terminal-only",
        ),
        (
            "--episode-time-cost=NaN",
            "Map2 comprehensive reward requires --episode-time-cost 0",
        ),
        (
            "--episode-time-cost=0.1",
            "Map2 comprehensive reward requires --episode-time-cost 0",
        ),
        (
            "--gamma-per-tick=0.99999994",
            "Map2 comprehensive reward requires --gamma-per-tick 1",
        ),
    ] {
        let arguments = "drysua train-full --updates 1 --checkpoint-directory artifacts/temp/not-accessed-map2-reward-validation"
            .split_ascii_whitespace().chain(flags.split_ascii_whitespace());
        let error = crate::cli::run_from_for_test(arguments).expect_err(flags);
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), message);
    }
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_configuration_accepts_boundaries_and_rejects_invalid_hyperparameters() {
    for (flags, field) in [
        ("--rollout 1162", "complete episode retained capacity"),
        (
            "--environments 3",
            "complete episodes require Map2, an even environment count up to the training maximum, and three-tick actions",
        ),
        ("--learning-rate=NaN", "learning rate"),
        ("--learning-rate=0", "learning rate"),
        ("--gae-lambda=1.01", "discount"),
        ("--entropy-coefficient=0", "entropy coefficient"),
    ] {
        let error = crate::cli::training_settings_for_test(
            &flags.split_ascii_whitespace().collect::<Vec<_>>(),
        )
        .expect_err(flags);
        assert_eq!(
            error.to_string(),
            format!("invalid PPO config field: {field}")
        );
    }
    for flags in [
        "--environments 2 --rollout 1163 --minibatch 64",
        "--environments 26 --rollout 1163 --minibatch 64",
        "--complete-episodes=false --rollout 8 --minibatch 32",
        "--learning-rate 3e-5 --gamma-per-tick 1 --gae-lambda .995 --entropy-coefficient .001 --episode-time-cost=-0",
    ] {
        crate::cli::training_settings_for_test(&flags.split_ascii_whitespace().collect::<Vec<_>>())
            .expect(flags);
    }
}
