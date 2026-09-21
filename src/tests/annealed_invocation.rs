//! Operational update limits leave the training trajectory and durable scope intact.

use std::num::NonZeroU64;
use std::sync::mpsc::sync_channel;
use std::time::Duration;

use super::*;

#[test]
fn cli_rejects_zero_out_of_range_and_overflowing_invocation_updates() {
    for (value, reason) in [
        (
            "0".to_owned(),
            format!("0 is not in 1..={MAX_TRAINING_COUNTER}"),
        ),
        (
            (MAX_TRAINING_COUNTER + 1).to_string(),
            format!(
                "{} is not in 1..={MAX_TRAINING_COUNTER}",
                MAX_TRAINING_COUNTER + 1
            ),
        ),
        (
            u64::MAX.to_string(),
            format!("{} is not in 1..={MAX_TRAINING_COUNTER}", u64::MAX),
        ),
        (
            "18446744073709551616".to_owned(),
            "number too large to fit in target type".to_owned(),
        ),
    ] {
        let error = crate::cli::annealed_settings_for_test(&[
            "--updates",
            "3",
            "--generation-games",
            "2",
            "--invocation-updates",
            &value,
        ])
        .expect_err("invalid invocation limit");

        let expected = format!(
            "error: invalid value '{value}' for '--invocation-updates <INVOCATION_UPDATES>': {reason}"
        );
        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
    }
}

#[test]
fn invocation_limit_leaves_default_config_scope_and_anneal_schedule_unchanged() {
    let arguments = [
        "--updates",
        "1000",
        "--generation-games",
        "8",
        "--parallel",
        "2",
    ];
    let plain = crate::cli::annealed_settings_for_test(&arguments).expect("unlimited settings");
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(
            settings,
            PolicyDevice::Cpu,
            settings.ppo,
            AnnealedHarness::default(),
            None,
        )
        .expect("canonical scope")
    };
    assert_eq!(plain.invocation_updates, None);
    assert_eq!(plain.updates, 1_000);
    assert_eq!(plain.zero_updates, 200);

    for limit in [1, MAX_TRAINING_COUNTER] {
        let value = limit.to_string();
        let mut limited_arguments = arguments.to_vec();
        limited_arguments.extend(["--invocation-updates", &value]);
        let mut limited = crate::cli::annealed_settings_for_test(&limited_arguments)
            .expect("bounded invocation settings");

        assert_eq!(limited.invocation_updates, NonZeroU64::new(limit));
        assert_eq!(scope(&plain), scope(&limited));
        assert_eq!(anneal_schedule(&plain), anneal_schedule(&limited));
        limited.invocation_updates = None;
        assert_eq!(plain, limited);
    }
}

#[test]
fn library_rejects_invocation_limits_above_the_training_counter_bound() {
    let mut config = invocation_settings();
    config.invocation_updates = NonZeroU64::new(MAX_TRAINING_COUNTER);
    assert_eq!(
        validate_annealed(&config, harness()).expect("maximum limit"),
        config.ppo
    );

    for limit in [MAX_TRAINING_COUNTER + 1, u64::MAX] {
        config.invocation_updates = NonZeroU64::new(limit);
        let error = validate_annealed(&config, harness()).expect_err("oversized invocation limit");

        assert_eq!(
            error.to_string(),
            "invalid PPO config field: annealed invocation updates exceed MAX_TRAINING_COUNTER"
        );
    }
}

#[test]
fn fresh_invocation_forces_update_one_checkpoint_and_runtime_export_before_next_generation() {
    let directory = test_directory("invocation-fresh");
    let (sent, received) = sync_channel(1);

    let report = run_annealed_job_harnessed(
        invocation_settings(),
        harness(),
        PolicyDevice::Cpu,
        &directory,
        false,
        None,
        move |checkpoint| {
            sent.try_send(checkpoint)
                .expect("exactly one forced checkpoint")
        },
    )
    .expect("one-update invocation with a one-day checkpoint cadence");

    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.games, 2);
    assert_eq!(report.generations, 1);
    assert_eq!(generation_files(&directory).len(), 1);
    assert_eq!(
        received
            .try_recv()
            .expect("durable callback")
            .completed_updates,
        1
    );
    let artifact = TrainingArtifact::load(&directory).expect("forced checkpoint");
    assert_eq!(artifact.progress().global_update, 1);
    let runtime = PolicyModel::fresh(0).expect("runtime model");
    TrainingArtifact::load_runtime_weights(&runtime, &directory).expect("forced runtime export");
    assert_eq!(
        runtime.export_parameters().expect("runtime parameters"),
        artifact_snapshot(&artifact).0
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn relative_resumes_match_uninterrupted_boundaries_clamp_at_target_and_then_do_no_work() {
    let baseline = test_directory("invocation-baseline");
    let resumed = test_directory("invocation-resumed");
    let mut config = invocation_settings();
    config.invocation_updates = None;
    config.checkpoint_cadence = crate::TrainingCheckpointCadence::Updates(1);
    let (sent, received) = sync_channel(3);
    let snapshots = baseline.clone();
    run_annealed_job_harnessed(
        config,
        harness(),
        PolicyDevice::Cpu,
        &baseline,
        false,
        None,
        move |checkpoint| {
            sent.try_send((
                checkpoint,
                checkpoint_digests(&snapshots),
                generation_files(&snapshots),
            ))
            .expect("three baseline boundaries");
        },
    )
    .expect("unlimited baseline");

    for (update, limit) in [(1, 1), (2, 1), (3, MAX_TRAINING_COUNTER)] {
        let mut config = invocation_settings();
        config.invocation_updates = NonZeroU64::new(limit);
        let report = run(config, &resumed, update != 1).expect("additional committed update");
        let (checkpoint, digests, generations) = received.try_recv().expect("baseline boundary");

        assert_eq!(report.completed_updates, update);
        assert_eq!(report.games, update * 2);
        assert_eq!(report.generations, update);
        assert_eq!(report.rollout_samples, checkpoint.rollout_samples);
        assert_eq!(report.optimizer_step, checkpoint.optimizer_step);
        assert_eq!(report.latest.policy_loss, checkpoint.policy_loss);
        assert_eq!(report.latest.value_loss, checkpoint.value_loss);
        assert_eq!(report.latest.entropy, checkpoint.entropy);
        assert_eq!(
            crate::ppo_arena::update_kl(report.latest),
            checkpoint.approximate_kl
        );
        assert_eq!(report.latest.stopped_for_kl, checkpoint.stopped_for_kl);
        assert_eq!(
            checkpoint_digests(&resumed),
            digests,
            "parameters, Adam, RNG and runtime at U{update}"
        );
        assert_eq!(
            generation_files(&resumed),
            generations,
            "no next generation at U{update}"
        );
    }
    assert_completed_resume_is_unchanged(&resumed);
    let baseline_artifact = TrainingArtifact::load(&baseline).expect("baseline artifact");
    let resumed_artifact = TrainingArtifact::load(&resumed).expect("resumed artifact");
    assert_eq!(baseline_artifact.progress(), resumed_artifact.progress());
    assert_eq!(
        artifact_snapshot(&baseline_artifact),
        artifact_snapshot(&resumed_artifact)
    );
    std::fs::remove_dir_all(baseline).expect("remove baseline");
    std::fs::remove_dir_all(resumed).expect("remove resumed");
}

fn invocation_settings() -> AnnealedJobConfig {
    let mut config = settings(0x1234, 3);
    config.invocation_updates = NonZeroU64::new(1);
    config.zero_updates = 1;
    config.checkpoint_cadence =
        crate::TrainingCheckpointCadence::WallTime(Duration::from_secs(86_400));
    config
}

fn assert_completed_resume_is_unchanged(directory: &Path) {
    let before = checkpoint_digests(directory);
    let generations = generation_files(directory);
    let artifact = TrainingArtifact::load(directory).expect("completed artifact");

    let report = run_annealed_job_harnessed(
        invocation_settings(),
        harness(),
        PolicyDevice::Cpu,
        directory,
        true,
        None,
        |_| panic!("completed resume must not write another checkpoint"),
    )
    .expect("already-completed resume");

    assert_eq!(report.completed_updates, 3);
    assert_eq!(report.games, 6);
    assert_eq!(report.generations, 3);
    assert_eq!(report.rollout_samples, artifact.progress().rollout_samples);
    assert_eq!(report.optimizer_step, artifact_snapshot(&artifact).3);
    assert_eq!(report.elapsed_ticks, 0);
    assert_eq!(report.latest, PpoUpdateReport::default());
    assert_eq!(checkpoint_digests(directory), before);
    assert_eq!(generation_files(directory), generations);
}
