use super::*;

#[test]
fn training_microbatch_cli_is_closed_and_only_modes_above_legacy_64_change_scope() {
    let arguments = [
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
        .expect("legacy microbatch 64");
    assert_eq!(baseline.execution.training_microbatch, 64);
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    for value in ["64", "128", "256", "0", "65", "512"] {
        let mut arguments = arguments.to_vec();
        arguments.extend(["--training-microbatch", value]);
        let result = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments);
        if matches!(value, "0" | "65" | "512") {
            assert!(
                result
                    .expect_err("closed modes")
                    .to_string()
                    .contains("training microbatch must be 64, 128 or 256")
            );
        } else {
            let parsed = result.expect("mode");
            assert_eq!(parsed.ppo, baseline.ppo);
            assert_eq!(parsed.execution.training_microbatch.to_string(), value);
            let expected = if value == "64" {
                scope(&baseline)
            } else {
                format!("{} --training-microbatch {value}", scope(&baseline))
            };
            assert_eq!(scope(&parsed), expected);
        }
    }
}

#[test]
fn training_microbatch_library_modes_are_closed() {
    for microbatch in [64, 128, 256] {
        crate::TrainingExecutionOptions {
            training_microbatch: microbatch,
            ..Default::default()
        }
        .validate()
        .expect("admitted mode");
    }
    for value in [0, 65, 512] {
        assert_eq!(
            crate::TrainingExecutionOptions {
                training_microbatch: value,
                ..Default::default()
            }
            .validate()
            .expect_err("library closed modes")
            .to_string(),
            "invalid PPO config field: training microbatch must be 64, 128 or 256"
        );
    }
}

#[test]
fn training_microbatch_scope_rejects_changes_and_resume_keeps_all_execution_fields() {
    for microbatch in [128, 256] {
        let source = test_directory("microbatch-uninterrupted");
        let target = test_directory("microbatch-resumed");
        let mut options = settings(9001, 2);
        options.games_per_update = 4;
        options.ppo.environments = 4;
        options.execution.actor_pipeline_groups = 2;
        options.execution.training_microbatch = microbatch;
        options.execution.reuse_actor_values = true;
        options.execution.balanced_minibatches = true;
        options.execution.host_math_workers = 2;
        run(options.clone(), &source, false).expect("uninterrupted");
        run_with(
            options.clone(),
            AnnealedHarness {
                stop_after: Some(1),
                ..harness()
            },
            &target,
            false,
        )
        .expect("first update");
        assert_restored_execution(&options, &target);
        let before = checkpoint_digests(&target);
        let mut changed = options.clone();
        changed.execution.training_microbatch = 64;
        assert_eq!(
            run(changed, &target, true)
                .expect_err("scope rejection")
                .to_string(),
            format!(
                "checkpoint scope mismatch: --training-microbatch: recorded {microbatch}, requested <absent>"
            )
        );
        assert_eq!(checkpoint_digests(&target), before);
        run(options, &target, true).expect("same-mode resume");
        assert_trajectory_equal(&source, &target);
        for directory in [source, target] {
            std::fs::remove_dir_all(directory).expect("own cleanup");
        }
    }
}

fn assert_restored_execution(options: &AnnealedJobConfig, directory: &std::path::Path) {
    let run =
        annealed_run(options, PolicyDevice::Cpu, options.ppo, harness(), None).expect("run scope");
    let random = directory.join(RANDOMIZATION_DIRECTORY);
    let session = AnnealedSession::initialize(
        options,
        PolicyDevice::Cpu,
        directory,
        true,
        None,
        options.ppo,
        run,
        &random,
        AnnealedOpponentRuntime::Teacher,
    )
    .expect("restored session");
    assert_eq!(
        session.state.trainer.execution_for_test(),
        crate::TrainingExecutionOptions {
            actor_pipeline_groups: 1,
            reuse_actor_values: false,
            ..options.execution
        }
    );
}
