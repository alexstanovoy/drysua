#[test]
fn cli_tactical_requires_explicit_weights_directory_for_both_play_forms() {
    for arguments in [
        vec!["drysua", "--policy", "tactical"],
        vec!["drysua", "play", "--policy", "tactical"],
    ] {
        let error = crate::cli::parse_from(arguments).expect_err("tactical needs weights");
        assert!(
            error
                .to_string()
                .contains("--weights-directory <WEIGHTS_DIRECTORY>")
        );
        assert!(
            error
                .to_string()
                .contains("required arguments were not provided")
        );
    }
}

#[test]
fn cli_tactical_missing_file_fails_before_connecting_without_teacher_fallback() {
    let directory = super::seat::tactical_directory("cli-missing");
    let error = crate::cli::run_from_for_test([
        "drysua",
        "play",
        "--policy",
        "tactical",
        "--name",
        "",
        "--weights-directory",
        directory.to_str().expect("UTF-8 directory"),
    ])
    .expect_err("missing tactical weights must fail before name validation");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    assert!(error.to_string().contains("drysua.tactical.bin"));
    std::fs::remove_dir(directory).expect("remove empty directory");
}

#[test]
fn cli_tactical_loads_canonical_file_without_safetensors() {
    let directory = super::seat::tactical_directory("cli-valid");
    std::fs::write(
        directory.join("drysua.tactical.bin"),
        crate::TacticalPolicy::default().to_bytes(),
    )
    .expect("canonical tactical weights");
    let error = crate::cli::run_from_for_test([
        "drysua",
        "play",
        "--policy",
        "tactical",
        "--name",
        "",
        "--weights-directory",
        directory.to_str().expect("UTF-8 directory"),
    ])
    .expect_err("valid tactical weights reach name validation");
    assert_eq!(error.to_string(), "bot name must not be empty");
    assert!(!directory.join("drysua.weights.safetensors").exists());
    std::fs::remove_dir_all(directory).expect("remove weights");
}

#[test]
fn cli_defaults_to_selected_teacher_for_implicit_and_explicit_play() {
    for arguments in [vec!["drysua"], vec!["drysua", "play"]] {
        assert_eq!(
            crate::cli::play_policy_for_test(arguments).expect("play policy"),
            crate::cli::PlayPolicy::Teacher
        );
    }
}

#[test]
fn cli_default_play_reaches_connection_validation_without_artifact_arguments() {
    for arguments in [
        vec!["drysua", "--name", ""],
        vec!["drysua", "play", "--name", ""],
    ] {
        let error = crate::cli::run_from_for_test(arguments)
            .expect_err("the selected Teacher must validate its name without loading weights");

        assert_eq!(error.to_string(), "bot name must not be empty");
    }
}

#[test]
fn cli_explicit_weights_without_policy_retain_hybrid_experiment_behavior() {
    for arguments in [
        vec!["drysua", "--weights-directory", "explicit-experiment"],
        vec![
            "drysua",
            "play",
            "--weights-directory",
            "explicit-experiment",
        ],
    ] {
        assert_eq!(
            crate::cli::play_policy_for_test(arguments).expect("explicit weights"),
            crate::cli::PlayPolicy::Hybrid
        );
    }
}

#[test]
fn selected_neural_weights_resolve_from_the_repository_not_the_working_directory() {
    for policy in [
        crate::cli::PlayPolicy::Hybrid,
        crate::cli::PlayPolicy::Tactical,
    ] {
        let selection = crate::default_deployment::DefaultDeployment {
            policy,
            weights_directory: Some("artifacts/v9.9.9"),
        };

        let (resolved_policy, directory) = selection.resolve().expect("selected deployment");

        assert_eq!(resolved_policy, policy);
        assert_eq!(
            directory.expect("automatic weights"),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/v9.9.9")
        );
    }
}

#[test]
fn selected_deployment_rejects_policy_and_weight_mismatches() {
    for selection in [
        crate::default_deployment::DefaultDeployment {
            policy: crate::cli::PlayPolicy::Teacher,
            weights_directory: Some("artifacts/v9.9.9"),
        },
        crate::default_deployment::DefaultDeployment {
            policy: crate::cli::PlayPolicy::Tactical,
            weights_directory: None,
        },
        crate::default_deployment::DefaultDeployment {
            policy: crate::cli::PlayPolicy::Hybrid,
            weights_directory: None,
        },
    ] {
        let error = selection.resolve().expect_err("inconsistent selection");

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
        let selection = crate::default_deployment::DefaultDeployment {
            policy: crate::cli::PlayPolicy::Tactical,
            weights_directory: Some(directory),
        };

        let error = selection.resolve().expect_err("invalid artifact location");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "default deployment weights must be a repository-relative directory below artifacts"
        );
    }
}

#[test]
fn cli_teacher_reaches_connection_validation_without_loading_weights() {
    let error = crate::cli::run_from_for_test([
        "drysua",
        "play",
        "--policy",
        "teacher",
        "--name",
        "",
        "--weights-directory",
        "artifacts/temp/nonexistent-teacher-weights",
    ])
    .expect_err("empty name must fail before connecting");

    assert_eq!(error.to_string(), "bot name must not be empty");
}

#[test]
fn cli_accepts_teacher_for_implicit_play_and_rejects_unknown_policy() {
    assert_eq!(
        crate::cli::play_policy_for_test(["drysua", "--policy", "teacher"])
            .expect("teacher policy"),
        crate::cli::PlayPolicy::Teacher
    );
    let error = crate::cli::parse_from(["drysua", "play", "--policy", "idle"])
        .expect_err("unsupported policy");
    assert!(
        error
            .to_string()
            .contains("invalid value 'idle' for '--policy <POLICY>'")
    );
}

#[test]
fn cli_rejects_hero_selector() {
    let error = crate::cli::parse_from(["drysua", "--hero", "2"])
        .expect_err("drysua must not accept a hero selector");

    assert!(error.to_string().contains("unexpected argument '--hero'"));
}

#[test]
fn shadow_fiend_pick_is_hero_two() {
    assert_eq!(crate::SHADOW_FIEND, bota_proto::HeroId(2));
}

#[test]
fn cli_accepts_bounded_ppo_smoke_parameters() {
    crate::cli::parse_from([
        "drysua",
        "train",
        "--updates",
        "1",
        "--environments",
        "2",
        "--rollout",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "16",
        "--seed",
        "77",
        "--map",
        "1",
        "--device",
        "cuda",
        "--device-ordinal",
        "1",
    ])
    .expect("train CLI");
}

#[test]
fn cli_accepts_bounded_self_play_smoke_parameters() {
    crate::cli::parse_from([
        "drysua",
        "league",
        "--updates",
        "1",
        "--environments",
        "4",
        "--rollout",
        "2",
        "--epochs",
        "1",
        "--minibatch",
        "8",
        "--evaluation-pairs",
        "1",
        "--evaluation-decisions",
        "2",
        "--seed",
        "77",
        "--map",
        "1",
        "--device",
        "metal",
        "--device-ordinal",
        "0",
    ])
    .expect("league CLI");
}

#[test]
fn cli_accepts_resumable_training_job_parameters() {
    crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "10000",
        "--environments",
        "4",
        "--rollout",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "32",
        "--checkpoint-seconds",
        "300",
        "--checkpoint-directory",
        "artifacts/temp/training",
        "--resume",
        "--migrate-provenance",
        "--device",
        "cuda",
    ])
    .expect("resumable train CLI");
}

#[test]
fn cli_rejects_provenance_migration_without_resume() {
    let error = crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "10000",
        "--checkpoint-directory",
        "artifacts/temp/training",
        "--migrate-provenance",
    ])
    .expect_err("migration requires resume");

    assert!(error.to_string().contains("--resume"));
}

#[test]
fn cli_accepts_initial_weights_for_fresh_training() {
    crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "8",
        "--checkpoint-directory",
        "training/ppo-v1",
        "--initial-weights",
        "artifacts/temp/pretrain-v1",
    ])
    .expect("fresh initialized training CLI");
}

#[test]
fn cli_rejects_initial_weights_when_resuming() {
    let error = crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "8",
        "--checkpoint-directory",
        "training/ppo-v1",
        "--initial-weights",
        "artifacts/temp/pretrain-v1",
        "--resume",
    ])
    .expect_err("resume already restores exact model state");

    assert!(error.to_string().contains("cannot be used with '--resume'"));
}

#[test]
fn cli_accepts_fixed_checkpoint_evaluation_matrix() {
    crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        "training/run/checkpoint",
        "--pairs",
        "2",
        "--decisions",
        "1024",
        "--seed",
        "77",
    ])
    .expect("evaluate CLI");
}

#[test]
fn cli_rejects_partial_checkpoint_evaluation_matrix() {
    let error = crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        "training/run/checkpoint",
        "--map",
        "1",
    ])
    .expect_err("evaluation matrix must include both maps");

    assert!(error.to_string().contains("unexpected argument '--map'"));
}

#[test]
fn cli_accepts_bounded_teacher_pretraining() {
    crate::cli::parse_from([
        "drysua",
        "pretrain",
        "--output-directory",
        "artifacts/temp/pretrain-maps",
        "--epochs",
        "8",
        "--seed",
        "50001",
        "--device",
        "cuda",
    ])
    .expect("pretrain CLI");
}

#[test]
fn cli_rejects_partial_map_teacher_pretraining() {
    let error = crate::cli::parse_from([
        "drysua",
        "pretrain",
        "--output-directory",
        "artifacts/temp/pretrain-map1",
        "--map",
        "1",
    ])
    .expect_err("pretraining must cover both maps");

    assert!(error.to_string().contains("unexpected argument '--map'"));
}
#[test]
fn neural_cli_requires_explicit_weights() {
    let error = crate::cli::parse_from(["drysua", "--policy", "neural"])
        .expect_err("neural cannot fall back");
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
    assert!(error.to_string().contains("--weights-directory"));
    assert_eq!(
        crate::cli::play_policy_for_test([
            "drysua",
            "play",
            "--policy",
            "neural",
            "--weights-directory",
            "artifacts/test"
        ])
        .expect("explicit neural"),
        crate::cli::PlayPolicy::Neural
    );
}

#[test]
fn neural_default_resolves_qualified_repository_weights() {
    let selected = crate::default_deployment::DefaultDeployment {
        policy: crate::cli::PlayPolicy::Neural,
        weights_directory: Some("artifacts/qualified"),
    }
    .resolve()
    .expect("future qualified neural default");
    assert_eq!(selected.0, crate::cli::PlayPolicy::Neural);
    assert!(
        selected
            .1
            .expect("weights")
            .ends_with("artifacts/qualified")
    );
}

#[test]
fn neural_default_rejects_missing_weights() {
    let error = crate::default_deployment::DefaultDeployment {
        policy: crate::cli::PlayPolicy::Neural,
        weights_directory: None,
    }
    .resolve()
    .expect_err("neural needs weights");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        error.to_string(),
        "default Teacher must be weights-free; default neural policies must specify weights"
    );
}
#[cfg(feature = "builtin")]
#[test]
fn train_full_defaults_preserve_the_existing_ppo_config_exactly() {
    let settings = crate::cli::training_settings_for_test(&[]).expect("default settings");
    assert_eq!(
        settings.ppo,
        crate::PpoConfig {
            environments: 4,
            rollout_decisions: 2_048,
            ..crate::PpoConfig::default()
        }
    );
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_long_horizon_flags_reach_training_config() {
    let settings = crate::cli::training_settings_for_test(&[
        "--map",
        "0",
        "--learning-rate",
        "3e-5",
        "--gamma-per-tick",
        "0.9999722",
        "--gae-lambda",
        "0.995",
        "--entropy-coefficient",
        "0.001",
    ])
    .expect("long horizon settings");
    assert_eq!(settings.map, bota_proto::MapId(0));
    assert_eq!(settings.ppo.learning_rate, 3e-5);
    assert_eq!(settings.ppo.gamma_tick, 0.9999722);
    assert_eq!(settings.ppo.gae_lambda, 0.995);
    assert_eq!(settings.ppo.entropy_coefficient, 0.001);
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_rejects_nonfinite_and_out_of_range_hyperparameters() {
    for (flag, field, invalid) in [
        (
            "--learning-rate",
            "learning rate",
            vec!["NaN", "inf", "-inf", "0", "-0.1"],
        ),
        (
            "--gamma-per-tick",
            "discount",
            vec!["NaN", "inf", "-inf", "-0.1", "1", "1.01"],
        ),
        (
            "--gae-lambda",
            "discount",
            vec!["NaN", "inf", "-inf", "-0.1", "1.01"],
        ),
        (
            "--entropy-coefficient",
            "entropy coefficient",
            vec!["NaN", "inf", "-inf", "0", "-0.1"],
        ),
    ] {
        for value in invalid {
            let argument = format!("{flag}={value}");
            let error = crate::cli::training_settings_for_test(&[&argument])
                .expect_err("invalid hyperparameter");
            assert_eq!(
                error.to_string(),
                format!("invalid PPO config field: {field}")
            );
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_accepts_existing_discount_boundaries() {
    let settings = crate::cli::training_settings_for_test(&[
        "--gamma-per-tick",
        "0",
        "--gae-lambda",
        "0.99999994",
    ])
    .expect("existing valid boundaries");
    assert_eq!(settings.ppo.gamma_tick, 0.0);
    assert_eq!(settings.ppo.gae_lambda, 0.99999994);
}
#[test]
fn cli_evaluate_accepts_explicit_pure_neural_map_zero_mode() {
    crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        ".",
        "--neural-map0",
    ])
    .expect("explicit pure evaluation flag");
}
#[test]
fn neural_terminal_evaluation_accepts_teacher_only_but_hybrid_rejects_it() {
    crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        ".",
        "--neural-map0",
        "--teacher-only",
        "--decisions",
        "36300",
    ])
    .expect("bounded full-game pure evaluation");
    let error = crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        ".",
        "--teacher-only",
    ])
    .expect_err("Teacher-only requires explicit Neural");
    assert!(error.to_string().contains("--neural-map0"));
}
