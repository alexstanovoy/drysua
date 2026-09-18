use super::*;
use crate::{MasteryConfig, MasteryProgress, MasteryStage, TrainingGameOutcome as Outcome};

#[test]
fn mastery_cli_defaults_overrides_and_canonical_order_are_exact() {
    let defaults = settings(&[]);
    assert_eq!(defaults.mastery_config, Some(MasteryConfig::default()));
    let left = settings(&[
        "--mastery-window",
        "7",
        "--mastery-win-percent",
        "90",
        "--opponent-win-percent",
        "teacher=70",
        "--opponent-win-percent",
        "weak=80",
    ]);
    let right = settings(&[
        "--mastery-window",
        "7",
        "--opponent-win-percent",
        "weak=80",
        "--opponent-win-percent",
        "teacher=70",
    ]);
    let left = training_checkpoint_run(&left, PolicyDevice::Cpu, left.ppo).expect("run");
    let right =
        training_checkpoint_run(&right, PolicyDevice::Cpu, right.ppo).expect("same effective run");
    assert_eq!(left, right);
    assert!(left.command_line.ends_with(" --opponent-schedule mastery-v1 --mastery-window 7 --opponent-win-percent weak=80 --opponent-win-percent teacher=70"));
    for (flag, value, expected) in [
        ("--mastery-window", "0", "invalid value '0'"),
        ("--mastery-window", "1025", "invalid value '1025'"),
        ("--mastery-win-percent", "101", "invalid value '101'"),
        (
            "--opponent-win-percent",
            "other=80",
            "unknown mastery opponent; expected weak or teacher",
        ),
    ] {
        let error = crate::cli::training_settings_for_test(&[
            "--opponent-schedule",
            "mastery-v1",
            flag,
            value,
        ])
        .expect_err("invalid option");
        assert!(error.to_string().contains(expected), "{error}");
    }
    let error = crate::cli::training_settings_for_test(&[
        "--opponent-schedule",
        "mastery-v1",
        "--opponent-win-percent",
        "weak=80",
        "--opponent-win-percent",
        "weak=90",
    ])
    .expect_err("duplicate");
    assert_eq!(error.to_string(), "duplicate opponent win percent");
    let error = crate::cli::training_settings_for_test(&["--mastery-window", "50"])
        .expect_err("inactive options");
    assert_eq!(
        error.to_string(),
        "mastery options require --opponent-schedule mastery-v1"
    );
}

#[test]
fn mastery_full_batches_freeze_one_opponent_and_only_the_next_batch_changes_stage() {
    for count in [2, 4, 6] {
        let mut settings = settings(&["--mastery-window", "1"]);
        settings.ppo.environments = count;
        let mut state = MasteryProgress::default();
        let weak = environments_with_mastery(&settings, 0, Some(&state)).expect("Weak batch");
        assert_eq!(collection_opponents(&weak), ("Weak", count, 0));
        state
            .record_batch(
                settings.mastery_config.expect("config"),
                &vec![Outcome::Win; count],
            )
            .expect("completed batch");
        assert_eq!(collection_opponents(&weak), ("Weak", count, 0));
        let teacher = environments_with_mastery(&settings, 1, Some(&state)).expect("Teacher batch");
        assert_eq!(collection_opponents(&teacher), ("Teacher", 0, count));
        for seats in teacher.as_chunks::<2>().0 {
            assert_eq!(seats[0].policy_seat, 0);
            assert_eq!(seats[1].policy_seat, 1);
            assert_eq!(seats[0].next_seed, seats[1].next_seed);
            assert_eq!(seats[0].next_opponent_seed, seats[1].next_opponent_seed);
        }
    }
}

#[test]
fn mastery_completed_restore_stops_without_games_even_with_a_larger_update_budget() {
    assert_resume_stops(true);
}

#[test]
fn mastery_update_budget_stops_without_faking_a_qualified_window() {
    assert_resume_stops(false);
}

fn assert_resume_stops(completed: bool) {
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
    let artifact = checkpoint_fixture(&settings, completed);
    artifact.save(&directory).expect("save");
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
    assert_eq!(report.terminal_wins, 0);
    assert_eq!(report.terminal_losses, 0);
    assert_eq!(report.terminal_draws, 0);
    assert_eq!(
        std::fs::read(directory.join("checkpoint.meta")).expect("unchanged manifest"),
        before
    );
    std::fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn mastery_git_only_migration_cannot_change_thresholds_or_window() {
    let settings = settings(&[]);
    let stored =
        training_checkpoint_run(&settings, PolicyDevice::Cpu, settings.ppo).expect("stored");
    let mut migrated = stored.clone();
    migrated.git_commit = "new-commit".to_owned();
    validate_provenance_migration(&stored, &migrated).expect("Git only");
    for config in [
        MasteryConfig::new(51, 80, &[]).unwrap(),
        MasteryConfig::new(50, 81, &[]).unwrap(),
    ] {
        migrated.mastery_config = Some(config);
        assert_eq!(
            validate_provenance_migration(&stored, &migrated),
            Err(PpoError::InvalidConfig("provenance migration scope"))
        );
    }
}

#[test]
fn mastery_batch_proposal_is_not_installed_before_optimizer_success_or_for_errors() {
    let settings = settings(&["--mastery-window", "1"]);
    let run = training_checkpoint_run(&settings, PolicyDevice::Cpu, settings.ppo).expect("run");
    let session = TrainingSession::initialize(
        &settings,
        PolicyDevice::Cpu,
        &std::env::temp_dir(),
        false,
        None,
        settings.ppo,
        run,
    )
    .expect("session");
    let before = session.mastery.clone();
    let mut report = PpoSmokeReport::default();
    report
        .completed_episodes
        .record(30, 1, Outcome::Win)
        .expect("stream1");
    report
        .completed_episodes
        .record(29, 0, Outcome::Win)
        .expect("stream0");
    let proposed = session
        .next_mastery(&settings, &report)
        .expect("proposal")
        .expect("mastery");
    assert_eq!(proposed.stage(), MasteryStage::Teacher);
    assert_eq!(session.mastery, before);
    report.rejected_orders = 1;
    assert_eq!(
        session.next_mastery(&settings, &report),
        Err(PpoError::InvalidTransition(
            "mastery requires a complete unrejected batch"
        ))
    );
    assert_eq!(session.mastery, before);
    assert_eq!(session.trainer.optimizer_step(), 0);
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
    let mastery = if completed {
        MasteryProgress::restore(
            MasteryStage::Completed,
            2,
            vec![true],
            settings.mastery_config.expect("config"),
        )
    } else {
        MasteryProgress::restore(
            MasteryStage::Weak,
            4,
            vec![true, false, true, false],
            settings.mastery_config.expect("config"),
        )
    }
    .expect("state");
    let progress = CheckpointProgress {
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
