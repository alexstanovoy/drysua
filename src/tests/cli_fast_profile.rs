use super::*;

fn arguments(extra: &[&str]) -> TrainAnnealedArgs {
    let mut command = vec![
        "drysua",
        "train-annealed",
        "--updates",
        "200",
        "--generation-games",
        "160",
        "--checkpoint-directory",
        "unused-fast-profile",
    ];
    command.extend_from_slice(extra);
    let cli = Cli::try_parse_from(command).expect("annealed arguments");
    let Some(Operation::TrainAnnealed(arguments)) = cli.operation else {
        panic!("annealed operation");
    };
    arguments
}

#[test]
fn annealed_cli_defaults_to_fast_execution_but_keeps_cpu_backend() {
    for extra in [&[][..], &["--resume"][..]] {
        let parsed = arguments(extra);
        assert_eq!(parsed.games, 40);
        assert_eq!(parsed.actor_pipeline_groups, 2);
        assert_eq!(parsed.training_microbatch, 256);
        assert!(parsed.reuse_actor_values);
        assert_eq!(parsed.host_math_workers, 1);
        assert!(!parsed.balanced_minibatches);
        assert!(matches!(parsed.device, LearnerDevice::Cpu));
        assert_eq!(parsed.device_ordinal, 0);
        assert_eq!(
            parsed.environment_schedule,
            EnvironmentScheduleArg::Adaptive
        );
        assert_eq!(parsed.generation_games, 160);
    }
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_resolves_hardware_independent_m40_b20_profile_without_changing_ppo() {
    let settings = arguments(&[])
        .annealed_settings("drysua".into(), "bota".into())
        .unwrap();
    assert_eq!(settings.games_per_update, 40);
    assert_eq!(settings.parallel_worlds, 20);
    assert_eq!(settings.zero_updates, 40);
    // A fresh run no longer pins a constant seed: the CLI field stays optional
    // and the resolved value is random and recorded in the run scope.
    // Seed resolution itself is covered by tests::annealed::seed_tests.
    assert_eq!(arguments(&[]).seed, None);
    assert_eq!(
        settings.environment_schedule,
        crate::EnvironmentSchedule::default()
    );
    assert_eq!(
        settings.execution,
        crate::TrainingExecutionOptions {
            actor_pipeline_groups: 2,
            training_microbatch: 256,
            reuse_actor_values: true,
            ..Default::default()
        }
    );
    assert_eq!(
        settings.ppo,
        crate::PpoConfig {
            environments: 40,
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            sample_budget: crate::PpoSampleBudget::Annealed,
            decision_interval_ticks: 3,
            gamma_tick: 1.0,
            ..Default::default()
        }
    );
    crate::ppo_arena::validate_annealed(&settings, Default::default()).unwrap();
}

#[test]
fn annealed_cli_reuse_accepts_bare_true_and_explicit_false() {
    for (extra, expected) in [
        (vec![], true),
        (vec!["--reuse-actor-values"], true),
        (vec!["--reuse-actor-values=true"], true),
        (vec!["--reuse-actor-values=false"], false),
    ] {
        assert_eq!(arguments(&extra).reuse_actor_values, expected);
    }
    for value in [
        "--reuse-actor-values=bad",
        "--reuse-actor-values=0",
        "--reuse-actor-values=FALSE",
    ] {
        let error = Cli::try_parse_from([
            "drysua",
            "train-annealed",
            "--updates",
            "200",
            "--generation-games",
            "160",
            "--checkpoint-directory",
            ".",
            value,
        ])
        .err()
        .unwrap();
        assert!(error.to_string().contains("invalid value"));
        assert!(error.to_string().contains("--reuse-actor-values"));
    }
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_old_execution_is_explicit_and_no_longer_an_implicit_default() {
    let parsed = arguments(&[
        "--games",
        "8",
        "--parallel",
        "8",
        "--actor-pipeline-groups",
        "1",
        "--training-microbatch",
        "64",
        "--reuse-actor-values=false",
        "--environment-schedule",
        "fixed",
    ]);
    let settings = parsed
        .annealed_settings("drysua".into(), "bota".into())
        .unwrap();
    assert_eq!(settings.games_per_update, 8);
    assert_eq!(settings.parallel_worlds, 8);
    assert_eq!(
        settings.execution,
        crate::TrainingExecutionOptions::default()
    );
    assert_eq!(
        settings.environment_schedule,
        crate::EnvironmentSchedule::Fixed
    );
    crate::ppo_arena::validate_annealed(&settings, Default::default()).unwrap();
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_rejects_small_games_with_default_parallel_before_directory_access() {
    let parsed = arguments(&["--games", "8"]);
    let settings = parsed
        .annealed_settings("drysua".into(), "bota".into())
        .unwrap();
    let error = run_train_annealed_with_settings(parsed, settings).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: annealed parallel worlds must divide games per update"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_requires_group_one_or_batched_inference_for_weights_and_micro64_for_wide_budget() {
    let settings = |extra: &[&str]| {
        arguments(extra)
            .annealed_settings("drysua".into(), "bota".into())
            .unwrap()
    };
    let weights = settings(&[
        "--opponent",
        "weights",
        "--opponent-weights",
        "unused-opponent",
        "--opponent-inference",
        "scalar",
    ]);
    assert_eq!(
        crate::ppo_arena::validate_annealed(&weights, Default::default())
            .unwrap_err()
            .to_string(),
        "invalid PPO config field: annealed actor pipeline weights opponent requires batched inference"
    );
    // Batched inference, the weights default, admits the grouped pipeline.
    let weights = settings(&[
        "--opponent",
        "weights",
        "--opponent-weights",
        "unused-opponent",
    ]);
    crate::ppo_arena::validate_annealed(&weights, Default::default()).unwrap();
    let weights = settings(&[
        "--opponent",
        "weights",
        "--opponent-weights",
        "unused-opponent",
        "--actor-pipeline-groups",
        "1",
    ]);
    crate::ppo_arena::validate_annealed(&weights, Default::default()).unwrap();
    let wide = settings(&["--games", "80"]);
    assert_eq!(
        crate::ppo_arena::validate_annealed(&wide, Default::default())
            .unwrap_err()
            .to_string(),
        "invalid PPO config field: training microbatch exceeds 12 GiB admission budget"
    );
    let wide = settings(&[
        "--games",
        "80",
        "--actor-pipeline-groups",
        "1",
        "--training-microbatch",
        "64",
        "--reuse-actor-values=false",
    ]);
    crate::ppo_arena::validate_annealed(&wide, Default::default()).unwrap();
}

#[test]
fn annealed_fast_defaults_do_not_change_library_or_other_command_defaults() {
    let execution = crate::TrainingExecutionOptions::default();
    assert_eq!(execution.actor_pipeline_groups, 1);
    assert_eq!(execution.training_microbatch, 64);
    assert!(!execution.reuse_actor_values);
    let Some(Operation::Train(train)) = Cli::try_parse_from(["drysua", "train"]).unwrap().operation
    else {
        panic!("train");
    };
    assert_eq!(train.environments, 2);
    let Some(Operation::TrainFull(train)) = Cli::try_parse_from([
        "drysua",
        "train-full",
        "--updates",
        "1",
        "--checkpoint-directory",
        ".",
    ])
    .unwrap()
    .operation
    else {
        panic!("train-full");
    };
    assert_eq!(train.environments, 4);
    assert_eq!(train.pipeline_groups, 1);
}
