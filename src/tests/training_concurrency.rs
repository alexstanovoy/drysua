use super::*;
use crate::TrainingExecutionOptions;

#[test]
fn actor_value_reuse_cli_is_annealed_only_with_legacy_off_and_canonical_enabled_scope() {
    let mut arguments = vec![
        "--updates",
        "2",
        "--games",
        "2",
        "--parallel",
        "2",
        "--generation-games",
        "2",
    ];
    let baseline = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments)
        .expect("legacy execution options");
    assert!(!TrainingExecutionOptions::default().reuse_actor_values);
    assert!(!baseline.execution.reuse_actor_values);
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    assert!(!original.contains("--reuse-actor-values"));
    arguments.extend([
        "--reuse-actor-values",
        "--host-math-workers",
        "4",
        "--balanced-minibatches",
    ]);
    let enabled =
        crate::cli::legacy_fixed_annealed_settings_for_test(&arguments).expect("reuse options");
    assert!(enabled.execution.reuse_actor_values);
    assert_eq!(
        scope(&enabled),
        format!("{original} --balanced-minibatches --host-math-workers 4 --reuse-actor-values")
    );
    for operation in ["train", "train-full"] {
        let error = crate::cli::parse_from(["drysua", operation, "--reuse-actor-values"])
            .expect_err("annealed-only option");
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        assert!(
            error
                .to_string()
                .contains("unexpected argument '--reuse-actor-values'")
        );
    }
}

#[test]
fn actor_value_reuse_resumes_exactly_after_scope_rejection_and_late_abort() {
    let uninterrupted = test_directory("actor-values-uninterrupted");
    let resumed = test_directory("actor-values-resumed");
    let mut options = settings(9001, 2);
    options.execution.reuse_actor_values = true;
    options.games_per_update = 4;
    options.ppo.environments = 4;
    assert!(harness().episode_decisions() >= 9);
    let expected = run(options.clone(), &uninterrupted, false).expect("reuse updates");
    let first = run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &resumed,
        false,
    )
    .expect("first reuse update");
    assert_eq!(first.completed_updates, 1);
    assert!(first.optimizer_step > 0);
    assert_eq!(generation_files(&resumed).len(), 2);
    let before = checkpoint_digests(&resumed);
    options.execution.reuse_actor_values = false;
    assert_eq!(
        run(options.clone(), &resumed, true)
            .expect_err("scope mismatch")
            .to_string(),
        "checkpoint scope mismatch: --reuse-actor-values: recorded <present>, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    options.execution.reuse_actor_values = true;
    assert_eq!(
        run_with(
            options.clone(),
            AnnealedHarness {
                stop_after_games: Some(4),
                ..harness()
            },
            &resumed,
            true,
        )
        .expect_err("stop after all games before optimizer commit")
        .to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    assert_eq!(generation_files(&resumed).len(), 4);
    let actual = run(options, &resumed, true).expect("replay aborted reuse update");
    assert_eq!(actual.completed_updates, 2);
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_report_bits(&actual.latest, &expected.latest);
    assert_trajectory_equal(&uninterrupted, &resumed);
    for directory in [uninterrupted, resumed] {
        std::fs::remove_dir_all(directory).expect("remove own checkpoint");
    }
}

#[test]
fn actor_value_reuse_is_rejected_by_the_public_trainer_without_changing_state() {
    let model = PolicyModel::fresh(9001).expect("model");
    let mut trainer = crate::PpoTrainer::new(&model, settings(9001, 1).ppo, 19).expect("trainer");
    let random = trainer.rng_checkpoint();
    let error = trainer
        .set_execution(TrainingExecutionOptions {
            reuse_actor_values: true,
            ..Default::default()
        })
        .expect_err("collector-only option");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: reuse actor values requires the annealed collector"
    );
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn balanced_minibatches_resume_only_with_the_recorded_execution_scope() {
    let directory = test_directory("balanced-minibatch-scope");
    let mut options = settings(9001, 2);
    options.execution.balanced_minibatches = true;
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &directory,
        false,
    )
    .expect("first balanced update");
    let before = checkpoint_digests(&directory);
    options.execution.balanced_minibatches = false;
    assert_eq!(
        run(options.clone(), &directory, true)
            .unwrap_err()
            .to_string(),
        "checkpoint scope mismatch: --balanced-minibatches: recorded <present>, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&directory), before);
    options.execution.balanced_minibatches = true;
    assert_eq!(
        run(options, &directory, true)
            .expect("balanced resume")
            .completed_updates,
        2
    );
    std::fs::remove_dir_all(directory).expect("remove own checkpoint");
}

fn folding_execution() -> TrainingExecutionOptions {
    TrainingExecutionOptions {
        host_math_workers: 4,
        ..Default::default()
    }
}

#[test]
fn concurrency_cli_rejects_retired_flags_and_bounds_scope_workers() {
    for arguments in [
        vec!["--actor-overlap", "continue-v1"],
        vec!["--learner-prefetch"],
    ] {
        let error = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments)
            .expect_err("retired option");
        assert!(
            error
                .to_string()
                .contains(&format!("unexpected argument '{}'", arguments[0]))
        );
    }
    let baseline = settings(9001, 2);
    assert_eq!(baseline.execution.host_math_workers, 1);
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    assert!(!original.contains("--host-math-workers"));
    for workers in [2, 4, 32] {
        let value = workers.to_string();
        let parsed = crate::cli::legacy_fixed_annealed_settings_for_test(&[
            "--updates",
            "2",
            "--generation-games",
            "2",
            "--games",
            "2",
            "--parallel",
            "2",
            "--host-math-workers",
            &value,
        ])
        .expect("worker option");
        assert_eq!(parsed.execution.host_math_workers, workers);
        let mut candidate = baseline.clone();
        candidate.execution = parsed.execution;
        assert_eq!(
            scope(&candidate),
            format!("{original} --host-math-workers {workers}")
        );
    }
    for workers in [0, 33] {
        let error = TrainingExecutionOptions {
            host_math_workers: workers,
            ..Default::default()
        }
        .validate()
        .expect_err("worker bound");
        assert_eq!(
            error.to_string(),
            "invalid PPO config field: host math workers must be within 1..=32"
        );
    }
}

#[test]
fn concurrency_short_updates_and_resume_match_default_bits() {
    let baseline = test_directory("concurrency-base");
    let candidate = test_directory("concurrency-folding");
    let mut options = settings(9001, 2);
    let expected = run(options.clone(), &baseline, false).expect("baseline");
    options.execution = folding_execution();
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &candidate,
        false,
    )
    .expect("first update");
    let actual = run(options.clone(), &candidate, true).expect("resume C4");
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_artifact_bits(&baseline, &candidate, PolicyDevice::Cpu);
    let before = checkpoint_digests(&candidate);
    options.execution = TrainingExecutionOptions::default();
    assert_eq!(
        run(options, &candidate, true)
            .expect_err("scope mismatch")
            .to_string(),
        "checkpoint scope mismatch: --host-math-workers: recorded 4, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&candidate), before);
    for directory in [baseline, candidate] {
        std::fs::remove_dir_all(directory).expect("remove own checkpoint");
    }
}

#[test]
fn retired_execution_scopes_allow_read_only_loading_but_reject_default_and_c4_resume() {
    let directory = test_directory("retired-execution-scopes");
    let options = settings(9001, 2);
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &directory,
        false,
    )
    .expect("committed first update");
    let original = TrainingArtifact::load(&directory).expect("original artifact");
    let model = PolicyModel::fresh(9001).expect("model");
    let restored = original.restore(&model, original.run()).expect("restore");
    let expected_parameters = model.export_parameters().expect("parameters");
    let generations = generation_files(&directory);
    for suffix in [
        " --actor-overlap continue-v1",
        " --learner-prefetch",
        " --actor-overlap continue-v1 --learner-prefetch",
        " --actor-overlap continue-v1 --learner-prefetch --host-math-workers 4",
    ] {
        let mut scope = original.run().clone();
        scope.command_line.push_str(suffix);
        TrainingArtifact::capture(
            &model,
            restored.trainer(),
            scope.clone(),
            original.progress().clone(),
        )
        .expect("historical scope")
        .save(&directory)
        .expect("save historical artifact");
        let before = checkpoint_digests(&directory);
        let loaded = TrainingArtifact::load(&directory).expect("read-only historical artifact");
        assert_eq!(loaded.run(), &scope);
        assert_eq!(loaded.progress(), original.progress());
        let weights = PolicyModel::fresh(17).expect("read-only model");
        TrainingArtifact::load_runtime_weights(&weights, &directory).expect("read-only weights");
        assert_eq!(
            weights.export_parameters().expect("weights"),
            expected_parameters
        );
        assert_eq!(checkpoint_digests(&directory), before);
        for execution in [TrainingExecutionOptions::default(), folding_execution()] {
            let mut requested = options.clone();
            requested.execution = execution;
            let difference = if execution.host_math_workers == 4 {
                "--host-math-workers: recorded <absent>, requested 4"
            } else if suffix.contains("--actor-overlap") {
                "--actor-overlap: recorded continue-v1, requested <absent>"
            } else {
                "--learner-prefetch: recorded <present>, requested <absent>"
            };
            assert_eq!(
                run(requested, &directory, true)
                    .expect_err("retired resume")
                    .to_string(),
                format!("checkpoint scope mismatch: {difference}")
            );
            assert_eq!(checkpoint_digests(&directory), before);
            assert_eq!(generation_files(&directory), generations);
        }
    }
    std::fs::remove_dir_all(directory).expect("remove own historical artifacts");
}

fn assert_report_bits(source: &crate::PpoUpdateReport, target: &crate::PpoUpdateReport) {
    let bits = |report: &crate::PpoUpdateReport| {
        [
            report.policy_loss,
            report.value_loss,
            report.entropy,
            report.approximate_kl,
            report.rejected_kl,
            report.clip_fraction,
            report.gradient_norm,
            report.applied_scale,
        ]
        .map(f64::to_bits)
    };
    assert_eq!(bits(source), bits(target));
    assert_eq!(source, target);
}
