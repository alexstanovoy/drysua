use super::*;
use crate::TrainingOpponentSchedule;

#[test]
fn curriculum_default_teacher_preserves_the_legacy_canonical_run() {
    let implicit = crate::cli::training_settings_for_test(&[]).expect("default");
    let explicit = crate::cli::training_settings_for_test(&["--opponent-schedule", "teacher"])
        .expect("explicit Teacher");
    assert_eq!(implicit, explicit);
    assert_eq!(
        implicit.opponent_schedule,
        TrainingOpponentSchedule::Teacher
    );
    let run = training_checkpoint_run(&implicit, PolicyDevice::Cpu, implicit.ppo).expect("run");
    assert!(run.command_line.ends_with(" --complete-episodes"));
    assert!(!run.command_line.contains("opponent-schedule"));
    for update in [0, 1, 2, 8, 999_999] {
        for pair in 0..3 {
            assert!(implicit.opponent_schedule.is_teacher(update, pair));
        }
    }
}

#[test]
fn curriculum_warmup_boundary_then_rotating_pair_mix_is_seed_independent() {
    for environments_count in [2, 4, 6] {
        let mut settings = settings();
        settings.ppo.environments = environments_count;
        for update in 0..8 {
            let arenas = environments(&settings, update).expect("scheduled environments");
            let teachers = arenas
                .iter()
                .filter(|arena| opponent_name(&arena.opponent) == "Teacher")
                .count();
            assert_eq!(teachers % 2, 0);
            if update < 2 {
                assert_eq!(teachers, 0);
            } else if environments_count == 6 {
                assert_eq!(teachers, 2);
            }
            let (pairs, remainder) = arenas.as_chunks::<2>();
            assert!(remainder.is_empty());
            for (pair, seats) in pairs.iter().enumerate() {
                assert_eq!(seats[0].policy_seat, 0);
                assert_eq!(seats[1].policy_seat, 1);
                assert_eq!(seats[0].next_seed, seats[1].next_seed);
                assert_eq!(
                    opponent_name(&seats[0].opponent),
                    opponent_name(&seats[1].opponent)
                );
                let expected = update >= 2 && (update - 2 + pair as u64).is_multiple_of(3);
                assert_eq!(
                    settings.opponent_schedule.is_teacher(update, pair),
                    expected
                );
            }
            settings.seed += 1;
            let replay = environments(&settings, update).expect("different training seed");
            assert_eq!(collection_opponents(&arenas), collection_opponents(&replay));
        }
    }
}

#[test]
fn curriculum_cli_rejects_unknown_schedule_and_window_mode() {
    let error = crate::cli::training_settings_for_test(&["--opponent-schedule", "unknown"])
        .expect_err("unknown curriculum");
    assert!(error.to_string().contains("invalid value 'unknown'"));
    assert!(error.to_string().contains("--opponent-schedule"));
    let error = crate::cli::training_settings_for_test(&[
        "--opponent-schedule",
        "weak-warmup-v1",
        "--complete-episodes=false",
    ])
    .expect_err("curriculum is full-episode only");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: opponent curriculum requires complete episodes"
    );
}

#[test]
fn curriculum_resume_uses_persisted_update_and_rejects_changed_schedule_before_mutation() {
    let settings = settings();
    let path = std::env::temp_dir().join(format!("curriculum-resume-test-{}", std::process::id()));
    assert!(
        path.starts_with(std::env::temp_dir()),
        "checkpoint test must honor the system temporary directory"
    );
    std::fs::create_dir(&path).expect("exclusive test directory");
    let run = training_checkpoint_run(&settings, PolicyDevice::Cpu, settings.ppo).expect("run");
    assert!(
        run.command_line
            .ends_with(" --opponent-schedule weak-warmup-v1")
    );
    let artifact = checkpoint_fixture(&settings, run.clone());
    artifact.save(&path).expect("save");
    let restored_model = PolicyModel::fresh(9914101).expect("restore target");
    let restored = restore_training_session(
        &restored_model,
        &path,
        &run,
        settings.ppo,
        ResumeProvenance::Strict,
    )
    .expect("strict restore");
    assert_eq!(restored.completed_updates, 2);
    assert_eq!(restored.trainer.rng_checkpoint(), (123, 7));
    assert_eq!(restored.sampling.checkpoint(), (456, 12));
    assert_eq!(
        collection_opponents(
            &environments(&settings, restored.completed_updates).expect("restored phase")
        ),
        ("Mixed", 4, 2)
    );
    let mut changed = settings.clone();
    changed.opponent_schedule = TrainingOpponentSchedule::Teacher;
    let changed_run =
        training_checkpoint_run(&changed, PolicyDevice::Cpu, changed.ppo).expect("changed run");
    let untouched = PolicyModel::fresh(9914102).expect("mismatch target");
    let before = untouched.export_parameters().expect("original parameters");
    assert_eq!(
        TrainingArtifact::load_compatible(&path, &changed_run).expect_err("scope mismatch"),
        crate::CheckpointError::InvalidManifest("compatibility scope")
    );
    assert_eq!(
        artifact.restore(&untouched, &changed_run).err(),
        Some(crate::CheckpointError::InvalidManifest(
            "compatibility scope"
        ))
    );
    assert_eq!(
        untouched.export_parameters().expect("unchanged parameters"),
        before
    );
    let _optimizer = untouched
        .claim_optimizer(settings.ppo.adam())
        .expect("mismatch did not claim optimizer");
    std::fs::remove_dir_all(&path).expect("cleanup");
}

fn checkpoint_fixture(settings: &TrainingJobConfig, run: CheckpointRun) -> TrainingArtifact {
    let model = PolicyModel::fresh(9914100).expect("model");
    let trainer = PpoTrainer::restore_checkpoint(
        settings.ppo,
        model
            .claim_optimizer(settings.ppo.adam())
            .expect("optimizer"),
        (123, 7),
        2,
    )
    .expect("trainer");
    let progress = CheckpointProgress {
        mastery: None,
        global_update: 2,
        policy_version: 2,
        scheduler_step: 2,
        curriculum_stage: 0,
        rollout_samples: 12,
        best_evaluation: None,
        rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 456, 12).expect("RNG")],
        league_references: Vec::new(),
    };
    TrainingArtifact::capture(&model, &trainer, run, progress).expect("capture")
}

fn settings() -> TrainingJobConfig {
    crate::cli::training_settings_for_test(&[
        "--opponent-schedule",
        "weak-warmup-v1",
        "--environments",
        "6",
        "--rollout",
        "1163",
        "--minibatch",
        "512",
    ])
    .expect("curriculum settings")
}
