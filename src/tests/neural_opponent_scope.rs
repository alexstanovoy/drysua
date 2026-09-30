use super::*;

#[test]
fn neural_opponent_batching_is_rejected_by_optimizer_execution_without_rng_changes() {
    let model = PolicyModel::fresh(9001).expect("model");
    let mut trainer = crate::PpoTrainer::new(&model, settings(9001, 1).ppo, 19).expect("trainer");
    let before = trainer.rng_checkpoint();
    assert_eq!(
        trainer.set_execution(crate::TrainingExecutionOptions {
            neural_opponent_batching: true,
            ..Default::default()
        }),
        Err(PpoError::InvalidConfig(
            "neural opponent batching requires the annealed collector"
        ))
    );
    assert_eq!(trainer.rng_checkpoint(), before);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn neural_opponent_cli_defaults_to_batched_weights_but_preserves_teacher_and_scalar_scopes() {
    assert!(!crate::TrainingExecutionOptions::default().neural_opponent_batching);
    let teacher = cli_options(false, None);
    assert!(!teacher.execution.neural_opponent_batching);
    for mode in ["batched", "scalar"] {
        let explicit = cli_options(false, Some(mode));
        assert!(!explicit.execution.neural_opponent_batching);
        assert_eq!(scope(&explicit), scope(&teacher));
    }
    assert!(
        !scope(&teacher)
            .command_line
            .contains("--opponent-inference")
    );
    let batched = cli_options(true, None);
    let scalar = cli_options(true, Some("scalar"));
    assert!(batched.execution.neural_opponent_batching);
    assert!(!scalar.execution.neural_opponent_batching);
    assert_eq!(scope(&batched), scope(&cli_options(true, Some("batched"))));
    assert_eq!(
        scope(&batched).command_line,
        format!(
            "{} --opponent-inference batched",
            scope(&scalar).command_line
        )
    );
    assert!(!scope(&scalar).command_line.contains("--opponent-inference"));
    for options in [&teacher, &scalar, &batched] {
        assert_eq!(
            validate_annealed(options, harness()).expect("no weights opened"),
            options.ppo
        );
    }
    let error =
        crate::cli::parse_from(["drysua", "train-annealed", "--opponent-inference", "greedy"])
            .expect_err("closed inference modes");
    assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    assert!(
        error
            .to_string()
            .contains("invalid value 'greedy' for '--opponent-inference")
    );
}

#[test]
fn neural_opponent_admission_allows_batched_groups_and_rejects_invalid_modes_before_loading() {
    for groups in [1, 2, 4] {
        let mut options = neural_options(Path::new("unused-neural-opponent"), groups, true);
        assert_eq!(
            validate_annealed(&options, harness()).expect("batched weights"),
            options.ppo
        );
        options.execution.neural_opponent_batching = false;
        if groups == 1 {
            assert_eq!(
                validate_annealed(&options, harness()).expect("legacy scalar"),
                options.ppo
            );
        } else {
            assert_eq!(
                run(
                    options.clone(),
                    Path::new("unused-neural-checkpoint"),
                    false
                ),
                Err(PpoError::InvalidConfig(
                    "annealed actor pipeline weights opponent requires batched inference"
                ))
            );
        }
        options.opponent = AnnealedOpponent::Teacher;
        assert_eq!(
            validate_annealed(&options, harness()).expect("teacher unchanged"),
            options.ppo
        );
        options.execution.neural_opponent_batching = true;
        assert_eq!(
            run(options, Path::new("unused-neural-checkpoint"), false),
            Err(PpoError::InvalidConfig(
                "neural opponent batching requires a weights opponent"
            ))
        );
    }
}

#[test]
fn neural_opponent_resume_is_exact_after_mode_rejection_and_late_abort_without_changing_weights() {
    let weights = test_directory("neural-opponent-weights");
    let model =
        PolicyModel::fresh_on(0x1111, PolicyDevice::Cpu).expect("current model architecture");
    TrainingArtifact::save_runtime_weights(&model, &weights).expect("frozen weights");
    drop(model);
    let frozen = weights_digest(&weights);
    for (groups, batched) in [(1, false), (1, true), (2, true), (4, true)] {
        assert_neural_resume(&weights, groups, batched);
        assert_eq!(weights_digest(&weights), frozen);
    }
    std::fs::remove_dir_all(weights).expect("remove own frozen weights");
}

fn assert_neural_resume(weights: &Path, groups: usize, batched: bool) {
    let uninterrupted = test_directory("neural-opponent-uninterrupted");
    let resumed = test_directory("neural-opponent-resumed");
    let mut options = neural_options(weights, groups, batched);
    assert!(harness().episode_decisions() >= 9);
    let expected = run(options.clone(), &uninterrupted, false).expect("continuous mode");
    let first = run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &resumed,
        false,
    )
    .expect("committed first update");
    assert_eq!(first.completed_updates, 1);
    assert!(first.optimizer_step > 0);
    let before = checkpoint_digests(&resumed);
    if groups == 1 {
        options.execution.neural_opponent_batching = !batched;
        let difference = if batched {
            "recorded batched, requested <absent>"
        } else {
            "recorded <absent>, requested batched"
        };
        assert_eq!(
            run(options.clone(), &resumed, true)
                .expect_err("mode mismatch")
                .to_string(),
            format!("checkpoint scope mismatch: --opponent-inference: {difference}")
        );
        assert_eq!(checkpoint_digests(&resumed), before);
        options.execution.neural_opponent_batching = batched;
    }
    assert_eq!(
        run_with(
            options.clone(),
            AnnealedHarness {
                stop_after_games: Some(2 * groups),
                ..harness()
            },
            &resumed,
            true,
        )
        .expect_err("abort before optimization")
        .to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    let actual = run(options, &resumed, true).expect("replay aborted update");
    assert_eq!(actual.completed_updates, 2);
    assert_eq!(actual.games, expected.games);
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_trajectory_equal(&uninterrupted, &resumed);
    for directory in [uninterrupted, resumed] {
        std::fs::remove_dir_all(directory).expect("remove own checkpoint");
    }
}

fn neural_options(weights: &Path, groups: usize, batched: bool) -> AnnealedJobConfig {
    let mut options = settings(9001, 2);
    options.opponent = AnnealedOpponent::Weights(weights.to_path_buf());
    options.games_per_update = 2 * groups;
    options.ppo.environments = 2 * groups;
    options.execution.actor_pipeline_groups = groups;
    options.execution.neural_opponent_batching = batched;
    options.execution.reuse_actor_values = true;
    options
}

fn cli_options(weights: bool, mode: Option<&str>) -> AnnealedJobConfig {
    let mut arguments = vec![
        "--updates",
        "2",
        "--games",
        "2",
        "--parallel",
        "2",
        "--generation-games",
        "2",
        "--actor-pipeline-groups",
        "1",
        "--training-microbatch",
        "64",
        "--environment-schedule",
        "fixed",
    ];
    if weights {
        arguments.extend([
            "--opponent",
            "weights",
            "--opponent-weights",
            "unused-neural-opponent",
        ]);
    }
    if let Some(mode) = mode {
        arguments.extend(["--opponent-inference", mode]);
    }
    crate::cli::annealed_settings_for_test(&arguments).expect("inference CLI")
}

fn scope(options: &AnnealedJobConfig) -> CheckpointRun {
    let fingerprint = matches!(options.opponent, AnnealedOpponent::Weights(_)).then_some(7);
    annealed_run(
        options,
        PolicyDevice::Cpu,
        options.ppo,
        harness(),
        fingerprint,
    )
    .expect("scope")
}

fn weights_digest(directory: &Path) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(directory.join("drysua.weights.safetensors")).expect("frozen file");
    Sha256::digest(bytes).to_vec()
}
