//! Sequential M40 coverage; all simulator work uses the invocation-only harness.

use super::*;
use crate::{CompletedTrainingEpisodes, TrainingGameOutcome};

fn expanded_settings(updates: u64) -> AnnealedJobConfig {
    let mut config = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "6",
        "--games",
        "40",
        "--parallel",
        "8",
        "--generation-games",
        "200",
        "--epochs",
        "1",
        "--minibatch",
        "80",
        "--zero-updates",
        "0",
        "--seed",
        "9001",
    ])
    .expect("M40 B8 K200 CLI settings");
    config.updates = updates;
    config.checkpoint_cadence = crate::TrainingCheckpointCadence::Updates(1);
    config
}

#[test]
fn m40_b8_k200_settings_validate_without_reducing_the_rollout() {
    let config = expanded_settings(1_000);

    let validated = validate_annealed(&config, AnnealedHarness::default())
        .expect("forty sequential games with eight concurrent worlds");

    assert_eq!(validated.environments, 40);
    assert_eq!(validated.rollout_decisions, 1_163);
    assert_eq!(validated.environments * validated.rollout_decisions, 46_520);
    assert_eq!(config.parallel_worlds, 8);
    assert_eq!(crate::MAX_TRAINING_ENVIRONMENTS, 26);
}

#[test]
fn completed_episodes_merge_all_forty_streams_in_tick_then_stream_order() {
    let mut completed = CompletedTrainingEpisodes::default();
    let mut expected = Vec::new();
    for batch in 0usize..5 {
        let mut part = CompletedTrainingEpisodes::default();
        for stream in batch * 8..(batch + 1) * 8 {
            let outcome = match stream % 4 {
                0 => TrainingGameOutcome::Win,
                1 => TrainingGameOutcome::Loss,
                2 => TrainingGameOutcome::Draw,
                3 => TrainingGameOutcome::TimeCap,
                _ => unreachable!("four outcomes"),
            };
            let tick = 40 - (stream / 2) as u32;
            part.record(tick, stream, outcome).expect("bounded stream");
            expected.push((tick, stream, outcome));
        }
        completed.merge(&part).expect("disjoint sequential batch");
    }

    expected.sort_by_key(|entry| (entry.0, entry.1));

    assert_eq!(completed.ordered_outcomes().len(), 40);
    assert_eq!(
        completed.ordered_outcomes(),
        expected
            .into_iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>()
    );
}

#[test]
fn completed_episodes_accept_stream39_and_reject_stream40_and_duplicates() {
    let mut completed = CompletedTrainingEpisodes::default();
    completed
        .record(crate::MAP2_TICK_CAP, 39, TrainingGameOutcome::TimeCap)
        .expect("last stream at the terminal tick");
    let original = completed;

    for (tick, stream, message) in [
        (1, 40, "completed episode tick or stream"),
        (0, 38, "completed episode tick or stream"),
        (
            crate::MAP2_TICK_CAP + 1,
            38,
            "completed episode tick or stream",
        ),
        (1, 39, "duplicate completed episode stream"),
    ] {
        assert_eq!(
            completed.record(tick, stream, TrainingGameOutcome::Win),
            Err(PpoError::InvalidTransition(message))
        );
        assert_eq!(completed, original);
    }
    assert_eq!(
        completed.merge(&original),
        Err(PpoError::InvalidTransition(
            "duplicate completed episode stream"
        ))
    );
    assert_eq!(completed, original);
}

#[test]
fn games40_is_valid_but_games41_and_games42_are_rejected() {
    for games in [0, 1, 40, 41, 42] {
        let mut config = expanded_settings(6);
        config.games_per_update = games;
        config.ppo.environments = games;
        let result = validate_annealed(&config, harness());
        if games == 40 {
            assert_eq!(result.expect("maximum games"), config.ppo);
        } else {
            assert_eq!(
                result,
                Err(PpoError::InvalidConfig(
                    "annealed games per update must be even and within 2..=40"
                ))
            );
        }
    }
}

#[test]
fn parallel_worlds26_is_valid_but_parallel_worlds41_is_rejected() {
    let mut config = settings(1, 6);
    config.games_per_update = 26;
    config.games_per_generation = 26;
    config.ppo.environments = 26;
    config.parallel_worlds = 26;
    assert_eq!(
        validate_annealed(&config, harness()).expect("B26"),
        config.ppo
    );

    for parallel in [0, 41] {
        config.parallel_worlds = parallel;
        assert_eq!(
            validate_annealed(&config, harness()),
            Err(PpoError::InvalidConfig("annealed parallel worlds"))
        );
    }
    let mut config = expanded_settings(6);
    config.parallel_worlds = 1;
    assert_eq!(
        validate_annealed(&config, harness()).expect("M40 B1"),
        config.ppo
    );
}

#[test]
fn parallel40_accepts_m40_k200_and_rejects_maximum_plus_one() {
    let mut config = expanded_settings(1_000);
    config.parallel_worlds = 40;
    assert_eq!(
        validate_annealed(&config, AnnealedHarness::default()),
        Ok(config.ppo)
    );
    config.parallel_worlds = 41;
    assert_eq!(
        validate_annealed(&config, AnnealedHarness::default()),
        Err(PpoError::InvalidConfig("annealed parallel worlds"))
    );
    assert_eq!(crate::MAX_TRAINING_ENVIRONMENTS, 26);
}

#[test]
fn parallel40_m40_k200_shortened_updates_resume_byte_identically() {
    let uninterrupted_directory = test_directory("b40-uninterrupted");
    let resumed_directory = test_directory("b40-resumed");
    let mut config = expanded_settings(2);
    config.parallel_worlds = 40;
    let uninterrupted = run(config.clone(), &uninterrupted_directory, false).expect("B40 updates");
    assert_eq!(uninterrupted.games, 80);
    assert_eq!(uninterrupted.completed_updates, 2);
    assert_eq!(uninterrupted.generations, 1);
    assert!(uninterrupted.rollout_samples >= 80);
    let stopped = AnnealedHarness {
        stop_after: Some(1),
        ..harness()
    };
    let first =
        run_with(config.clone(), stopped, &resumed_directory, false).expect("B40 checkpoint");
    assert_eq!(first.games, 40);
    assert_eq!(first.completed_updates, 1);
    let resumed = run(config, &resumed_directory, true).expect("B40 resume");
    assert_eq!(resumed.games, 80);
    assert_eq!(resumed.completed_updates, 2);
    assert_eq!(resumed.rollout_samples, uninterrupted.rollout_samples);
    assert_eq!(resumed.optimizer_step, uninterrupted.optimizer_step);
    assert_expanded_artifacts_equal(&uninterrupted_directory, &resumed_directory);
    std::fs::remove_dir_all(uninterrupted_directory).expect("cleanup uninterrupted");
    std::fs::remove_dir_all(resumed_directory).expect("cleanup resumed");
}

#[test]
fn expanded_games_keep_batch_and_generation_divisibility_requirements() {
    let mut config = expanded_settings(6);
    config.parallel_worlds = 6;
    assert_eq!(
        validate_annealed(&config, harness()),
        Err(PpoError::InvalidConfig(
            "annealed parallel worlds must divide games per update"
        ))
    );
    config.parallel_worlds = 8;
    config.games_per_generation = 202;
    assert_eq!(
        validate_annealed(&config, harness()),
        Err(PpoError::InvalidConfig(
            "annealed games per generation must be positive and divisible by parallel worlds"
        ))
    );
    assert_eq!(crate::cli::default_parallel_worlds_for(40, 200, 8), 8);
    assert_eq!(crate::cli::default_parallel_worlds_for(40, 200, 64), 20);
}

#[test]
fn library_settings_reject_noncanonical_sample_profiles_and_dimensions() {
    let mut expanded = settings(1, 6);
    expanded.games_per_update = 40;
    expanded.parallel_worlds = 8;
    expanded.games_per_generation = 200;
    expanded.ppo.environments = 40;
    expanded.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    assert_eq!(
        validate_annealed(&expanded, harness()).expect("explicit library profile"),
        expanded.ppo
    );
    expanded.ppo.sample_budget = crate::PpoSampleBudget::Standard;
    let mut legacy = settings(1, 6);
    legacy.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    for config in [expanded, legacy] {
        assert_eq!(
            validate_annealed(&config, harness()),
            Err(PpoError::InvalidConfig(
                "annealed PPO sample budget must match games per update"
            ))
        );
    }
    for field in ["environments", "rollout", "interval", "gamma"] {
        let mut config = expanded_settings(6);
        let message = match field {
            "environments" => {
                config.ppo.environments = 38;
                "annealed PPO environments must equal games per update"
            }
            "rollout" => {
                config.ppo.rollout_decisions -= 1;
                "annealed PPO rollout must be the retained decision count"
            }
            "interval" => {
                config.ppo.decision_interval_ticks = 1;
                "annealed PPO decision interval must be three ticks"
            }
            "gamma" => {
                config.ppo.gamma_tick = 0.99;
                "annealed Map2 reward requires gamma per tick one"
            }
            _ => unreachable!("fixed cases"),
        };
        assert_eq!(
            validate_annealed(&config, harness()),
            Err(PpoError::InvalidConfig(message))
        );
    }
}

#[test]
fn cli_selects_expanded_profile_only_above26_and_legacy_scope_bytes_are_unchanged() {
    for games in (2..=40).step_by(2) {
        let config = crate::cli::annealed_settings_for_test(&[
            "--updates",
            "6",
            "--games",
            &games.to_string(),
            "--parallel",
            "1",
            "--generation-games",
            "200",
            "--epochs",
            "1",
            "--minibatch",
            "80",
            "--zero-updates",
            "0",
            "--seed",
            "9001",
        ])
        .expect("bounded CLI games");
        let ppo = validate_annealed(&config, AnnealedHarness::default()).expect("valid profile");
        let run = annealed_run(
            &config,
            PolicyDevice::Cpu,
            ppo,
            AnnealedHarness::default(),
            None,
        )
        .expect("canonical scope");
        let mut expected = format!(
            "train-annealed --updates 6 --games {games} --parallel 1 --generation-games 200 --zero-updates 0 --epochs 1 --minibatch 80 --seed 9001 --map 2 --device cpu"
        );
        let profile = if games > 26 {
            expected.push_str(" --sample-budget annealed-v1");
            crate::PpoSampleBudget::Annealed
        } else {
            crate::PpoSampleBudget::Standard
        };
        expected.push_str(" --opponent teacher");
        assert_eq!(ppo.sample_budget, profile);
        assert_eq!(run.command_line, expected);
    }
}

#[test]
fn m40_four_epoch_thousand_update_budget_uses_one_shuffle_per_update() {
    let mut config = expanded_settings(1_000);
    config.ppo.epochs = 4;

    let ppo = validate_annealed(&config, AnnealedHarness::default()).expect("production budget");

    assert_eq!(ppo.environments * ppo.rollout_decisions, 46_520);
    assert_eq!(
        config.updates * ppo.epochs as u64 * (46_520 - 1),
        186_076_000
    );
}

#[test]
fn shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one() {
    for (epochs, minibatch, per_update, field) in [
        (4, 80, 4 * (46_520 - 1), "annealed shuffle RNG counter"),
        (1, 80, 46_520, "annealed sample counter"),
        (2, 1, 2 * 46_520, "annealed optimizer counter"),
    ] {
        let mut config = expanded_settings(MAX_TRAINING_COUNTER / per_update);
        config.ppo.epochs = epochs;
        config.ppo.minibatch = minibatch;
        assert_eq!(
            validate_annealed(&config, harness()).expect("maximum updates"),
            config.ppo
        );

        config.updates += 1;

        assert_eq!(
            validate_annealed(&config, harness()),
            Err(PpoError::InvalidConfig(field))
        );
    }
}

#[test]
fn counter_budget_accepts_exact_limit_and_rejects_limit_plus_one_or_overflow() {
    for field in [
        "annealed actor RNG counter",
        "annealed shuffle RNG counter",
        "annealed optimizer counter",
        "annealed sample counter",
    ] {
        assert_eq!(
            validate_counter_budget(1, MAX_TRAINING_COUNTER, MAX_TRAINING_COUNTER, field),
            Ok(())
        );
        assert_eq!(
            validate_counter_budget(1, MAX_TRAINING_COUNTER + 1, MAX_TRAINING_COUNTER, field),
            Err(PpoError::InvalidConfig(field))
        );
        assert_eq!(
            validate_counter_budget(u64::MAX, 2, MAX_TRAINING_COUNTER, field),
            Err(PpoError::InvalidConfig(field))
        );
    }
}

fn assert_expanded_artifacts_equal(first: &Path, second: &Path) {
    let first_artifact = TrainingArtifact::load(first).expect("uninterrupted artifact");
    let second_artifact = TrainingArtifact::load(second).expect("resumed artifact");
    assert_eq!(first_artifact.run(), second_artifact.run());
    assert_eq!(first_artifact.progress(), second_artifact.progress());
    assert_eq!(
        first_artifact.progress().rng_states,
        second_artifact.progress().rng_states
    );
    assert_eq!(
        artifact_snapshot(&first_artifact),
        artifact_snapshot(&second_artifact)
    );
    assert_eq!(checkpoint_digests(first), checkpoint_digests(second));
    assert_eq!(generation_files(first), generation_files(second));
}

fn assert_expanded_resume_rejects_changed_dimensions(config: &AnnealedJobConfig, directory: &Path) {
    let before = checkpoint_digests(directory);
    for (games, parallel, generation_games, message) in [
        (32, 8, 200, "--games: recorded 40, requested 32"),
        (40, 4, 200, "--parallel: recorded 8, requested 4"),
        (
            40,
            8,
            400,
            "--generation-games: recorded 200, requested 400",
        ),
    ] {
        let mut changed = config.clone();
        changed.games_per_update = games;
        changed.ppo.environments = games;
        changed.parallel_worlds = parallel;
        changed.games_per_generation = generation_games;

        let error = run(changed, directory, true).expect_err("changed M/B/K");

        assert_eq!(
            error.to_string(),
            format!("checkpoint scope mismatch: {message}")
        );
        assert_eq!(checkpoint_digests(directory), before);
    }
}

fn assert_expanded_run_completed(report: &AnnealedJobReport, directory: &Path) {
    assert_eq!(report.completed_updates, 6);
    assert_eq!(report.games, 240);
    assert_eq!(report.generations, 2);
    assert!(
        report.rollout_samples >= 240,
        "every game retains at least one row"
    );
    assert!(
        report.rollout_samples <= 480,
        "at most two full retention intervals per game"
    );
    assert!(report.optimizer_step > 0, "the learner applies updates");
    assert!(
        report.optimizer_step <= 6,
        "at most one minibatch per shortened PPO update"
    );
    let files = generation_files(directory);
    assert_eq!(files.len(), 2);
    assert!(files[0].1.contains("\"applied_games\":200"));
    assert!(files[1].1.contains("\"applied_games\":40"));
}

#[test]
fn m40_b8_k200_six_updates_resume_and_mid_update_replay_are_byte_identical() {
    let uninterrupted_directory = test_directory("m40-uninterrupted");
    let resumed_directory = test_directory("m40-resumed");
    let config = expanded_settings(6);
    let uninterrupted = run(config.clone(), &uninterrupted_directory, false).expect("six updates");
    assert_expanded_run_completed(&uninterrupted, &uninterrupted_directory);

    let stopped = AnnealedHarness {
        stop_after: Some(5),
        ..harness()
    };
    let first =
        run_with(config.clone(), stopped, &resumed_directory, false).expect("first generation");
    assert_eq!(first.completed_updates, 5);
    assert_eq!(first.games, 200);
    assert_eq!(first.generations, 1);
    assert_expanded_resume_rejects_changed_dimensions(&config, &resumed_directory);
    let committed = checkpoint_digests(&resumed_directory);
    let interrupted = AnnealedHarness {
        stop_after_games: Some(24),
        ..harness()
    };

    let error = run_with(config.clone(), interrupted, &resumed_directory, true)
        .expect_err("interrupt update six after three eight-world batches");

    assert_eq!(
        error.to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(checkpoint_digests(&resumed_directory), committed);
    assert_eq!(
        TrainingArtifact::load(&resumed_directory)
            .expect("committed artifact")
            .progress()
            .global_update,
        5
    );
    assert_eq!(
        generation_files(&resumed_directory).len(),
        2,
        "uncommitted generation is replayed"
    );
    let resumed = run(config, &resumed_directory, true).expect("replay the sixth update");
    assert_eq!(resumed.completed_updates, 6);
    assert_eq!(resumed.games, uninterrupted.games);
    assert_eq!(resumed.generations, uninterrupted.generations);
    assert_eq!(resumed.rollout_samples, uninterrupted.rollout_samples);
    assert_eq!(resumed.optimizer_step, uninterrupted.optimizer_step);
    assert_expanded_artifacts_equal(&uninterrupted_directory, &resumed_directory);
    std::fs::remove_dir_all(uninterrupted_directory).expect("remove uninterrupted directory");
    std::fs::remove_dir_all(resumed_directory).expect("remove resumed directory");
}
