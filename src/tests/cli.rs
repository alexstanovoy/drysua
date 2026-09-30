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
        "drysua train-annealed --updates 10000 --generation-games 40 --checkpoint-seconds 300 --checkpoint-directory training/run --resume --device cuda --device-ordinal 1",
        "drysua train-annealed --updates 8 --games 2 --parallel 2 --generation-games 2 --checkpoint-directory training/run --initial-weights training/pretrain",
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
            "drysua train-annealed --updates 1 --generation-games 40 --checkpoint-directory . --initial-weights training/pretrain --resume",
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

#[cfg(feature = "builtin")]
#[test]
fn train_annealed_configuration_rejects_invalid_hyperparameters_before_execution() {
    let base = [
        "--updates",
        "1",
        "--games",
        "2",
        "--parallel",
        "2",
        "--generation-games",
        "2",
        "--actor-pipeline-groups",
        "1",
    ];
    for (flags, field) in [
        ("--learning-rate=NaN", "learning rate"),
        ("--learning-rate=0", "learning rate"),
        ("--gae-lambda=1.01", "discount"),
        ("--entropy-coefficient=0", "entropy coefficient"),
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(flags.split_ascii_whitespace());
        let settings = crate::cli::annealed_settings_for_test(&arguments).expect(flags);
        let error =
            crate::ppo_arena::validate_annealed(&settings, Default::default()).expect_err(flags);
        assert_eq!(
            error.to_string(),
            format!("invalid PPO config field: {field}")
        );
    }
    let mut arguments = base.to_vec();
    arguments.extend(
        "--learning-rate 3e-5 --gae-lambda .995 --entropy-coefficient .001"
            .split_ascii_whitespace(),
    );
    let settings = crate::cli::annealed_settings_for_test(&arguments).expect("boundaries");
    crate::ppo_arena::validate_annealed(&settings, Default::default()).expect("valid settings");
}
