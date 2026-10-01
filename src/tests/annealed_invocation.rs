//! Invocation limits must change neither committed training state nor its scope.

use std::num::NonZeroU64;
use std::sync::mpsc::sync_channel;
use std::time::Duration;

use super::*;

#[test]
fn invocation_limits_reject_zero_overflow_and_values_above_the_counter_bound() {
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
            "18446744073709551616".to_owned(),
            "number too large to fit in target type".to_owned(),
        ),
    ] {
        let error = crate::cli::fixed_annealed_settings_for_test(&[
            "--updates",
            "3",
            "--generation-updates",
            "1",
            "--invocation-updates",
            &value,
        ])
        .expect_err("invalid invocation limit");
        let expected = format!(
            "error: invalid value '{value}' for '--invocation-updates <INVOCATION_UPDATES>': {reason}"
        );
        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
    }
    let mut config = invocation_settings();
    config.invocation_updates = NonZeroU64::new(MAX_TRAINING_COUNTER);
    assert_eq!(validate_annealed(&config, harness()), Ok(config.ppo));
    config.invocation_updates = NonZeroU64::new(MAX_TRAINING_COUNTER + 1);
    assert_eq!(
        validate_annealed(&config, harness()),
        Err(PpoError::InvalidConfig(
            "annealed invocation updates exceed MAX_TRAINING_COUNTER"
        ))
    );
}

#[test]
fn relative_resumes_match_uninterrupted_boundaries_clamp_at_target_and_then_do_no_work() {
    let baseline = test_directory("invocation-baseline");
    let resumed = test_directory("invocation-resumed");
    let mut config = invocation_settings();
    config.invocation_updates = None;
    config.checkpoint_cadence = crate::TrainingCheckpointCadence::Updates(1);
    let (sent, received) = sync_channel(3);
    let snapshots = baseline.to_path_buf();
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
        let report = run_limited_invocation(&resumed, update, limit);
        let (checkpoint, digests, generations) = received.try_recv().expect("baseline boundary");
        // Committing an update has drawn every pipelined update's generation.
        let count = usize::try_from(update + PIPELINE_STALENESS + 1)
            .expect("bounded update")
            .min(3);
        assert_eq!(generations.len(), count);
        assert!(generations[0].1.contains("\"scale_bp\":10000,"));
        assert_eq!(
            generations[count - 1].1.contains("\"scale_bp\":0,"),
            count == 3
        );
        assert_eq!(report.completed_updates, update);
        assert_eq!(report.generations, count as u64);
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
    assert_trajectory_equal(&baseline, &resumed);
}

fn run_limited_invocation(directory: &Path, update: u64, limit: u64) -> AnnealedJobReport {
    let mut config = invocation_settings();
    config.invocation_updates = NonZeroU64::new(limit);
    let (sent, forced) = sync_channel(1);
    let report = run_annealed_job_harnessed(
        config,
        harness(),
        PolicyDevice::Cpu,
        directory,
        update != 1,
        None,
        move |checkpoint| {
            sent.try_send(checkpoint)
                .expect("one forced durable checkpoint");
        },
    )
    .expect("additional committed update");
    assert_eq!(
        forced
            .try_recv()
            .expect("durable callback")
            .completed_updates,
        update
    );
    let artifact = TrainingArtifact::load(directory).expect("forced checkpoint");
    assert_eq!(artifact.progress().global_update, update);
    let runtime = PolicyModel::fresh(0).expect("runtime target");
    TrainingArtifact::load_runtime_weights(&runtime, directory).expect("forced export");
    assert_eq!(
        runtime.export_parameters().expect("runtime parameters"),
        artifact_runtime_state(&artifact).0
    );
    report
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
        |_| panic!("completed resume must not write a checkpoint"),
    )
    .expect("already-completed resume");
    assert_eq!(report.completed_updates, 3);
    assert_eq!(report.games, 0);
    assert_eq!(report.generations, 3);
    assert_eq!(report.rollout_samples, artifact.progress().rollout_samples);
    assert_eq!(report.optimizer_step, artifact_runtime_state(&artifact).1);
    assert_eq!(report.elapsed_ticks, 0);
    assert_eq!(report.latest, PpoUpdateReport::default());
    assert_eq!(checkpoint_digests(directory), before);
    assert_eq!(generation_files(directory), generations);
}
