//! Persisted mastery qualification and budget stop conditions through the job API.

use super::*;
use crate::{MasteryProgress, MasteryStage};

#[test]
fn mastery_stops_on_qualification_or_budget_without_collecting_or_rewriting() {
    for completed in [false, true] {
        let mut settings = settings(if completed {
            &["--mastery-window", "1"]
        } else {
            &[]
        });
        settings.updates = if completed { 99 } else { 2 };
        let directory = std::env::temp_dir().join(format!(
            "drysua-mastery-stop-{completed}-{}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("exclusive directory");
        checkpoint_fixture(&settings, completed)
            .save(&directory)
            .expect("checkpoint");
        let before = std::fs::read(directory.join("checkpoint.meta")).expect("manifest");
        let report = run_training_job_on_with_initial_weights(
            settings,
            PolicyDevice::Cpu,
            &directory,
            true,
            None,
            |_| panic!("no new checkpoint/update"),
        )
        .expect("bounded resume");
        assert_eq!(report.completed_updates, 2);
        assert_eq!(report.mastery_completed, completed);
        assert_eq!(report.elapsed_ticks, 0);
        assert_eq!(
            (
                report.terminal_wins,
                report.terminal_losses,
                report.terminal_draws
            ),
            (0, 0, 0)
        );
        assert_eq!(
            std::fs::read(directory.join("checkpoint.meta")).expect("unchanged manifest"),
            before
        );
        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}

fn checkpoint_fixture(settings: &TrainingJobConfig, completed: bool) -> TrainingArtifact {
    let model = PolicyModel::fresh(9140200).expect("model");
    let trainer = PpoTrainer::restore_checkpoint(
        settings.ppo,
        model
            .claim_optimizer(settings.ppo.adam())
            .expect("optimizer"),
        (123, 7),
        2,
    )
    .expect("trainer");
    let run = training_checkpoint_run(settings, PolicyDevice::Cpu, settings.ppo).expect("run");
    let (stage, games, window) = if completed {
        (MasteryStage::Completed, 2, vec![true])
    } else {
        (MasteryStage::Weak, 4, vec![true, false, true, false])
    };
    let mastery = MasteryProgress::restore(
        stage,
        games,
        window,
        settings.mastery_config.expect("config"),
    )
    .expect("mastery state");
    let progress = CheckpointProgress {
        adaptive_environment: None,
        mastery: Some(mastery),
        global_update: 2,
        policy_version: 2,
        scheduler_step: 2,
        curriculum_stage: 0,
        rollout_samples: 4,
        best_evaluation: None,
        rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 456, 4).expect("rng")],
        league_references: Vec::new(),
    };
    TrainingArtifact::capture(&model, &trainer, run, progress).expect("artifact")
}

fn settings(arguments: &[&str]) -> TrainingJobConfig {
    let mut args = vec![
        "--opponent-schedule",
        "mastery-v1",
        "--environments",
        "2",
        "--rollout",
        "1163",
        "--epochs",
        "1",
        "--minibatch",
        "512",
    ];
    args.extend(arguments);
    crate::cli::training_settings_for_test(&args).expect("mastery settings")
}
