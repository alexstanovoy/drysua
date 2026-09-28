//! Capacity is proved by real M40/B40 updates plus bounded failure oracles.

use super::*;
use crate::{CompletedTrainingEpisodes, TrainingGameOutcome};

#[test]
fn wide_capacity_m64_b64_and_m80_b40_collect_one_ppo_step_and_resume_exactly() {
    for (games, parallel) in [(64, 64), (80, 40)] {
        let baseline = test_directory("wide-baseline");
        let resumed = test_directory("wide-resumed");
        let config = wide_settings(games, parallel);
        let expected = run(config.clone(), &baseline, false).expect("wide collection");
        assert_eq!(expected.games, 2 * games as u64);
        assert_eq!(
            expected.optimizer_step, 2,
            "one full minibatch per shortened update"
        );
        let first = run_with(
            config.clone(),
            AnnealedHarness {
                stop_after: Some(1),
                ..harness()
            },
            &resumed,
            false,
        )
        .expect("first committed update");
        assert_eq!(first.games, games as u64);
        assert_eq!(first.optimizer_step, 1);
        let actual = run(config, &resumed, true).expect("wide resume");
        assert_eq!(actual.rollout_samples, expected.rollout_samples);
        assert_eq!(actual.optimizer_step, expected.optimizer_step);
        assert_trajectory_equal(&baseline, &resumed);
        std::fs::remove_dir_all(baseline).expect("cleanup baseline");
        std::fs::remove_dir_all(resumed).expect("cleanup resumed");
    }
}

fn wide_settings(games: usize, parallel: usize) -> AnnealedJobConfig {
    let mut config = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "2",
        "--games",
        &games.to_string(),
        "--parallel",
        &parallel.to_string(),
        "--generation-games",
        &(games * 5).to_string(),
        "--epochs",
        "1",
        "--minibatch",
        "160",
        "--zero-updates",
        "0",
        "--seed",
        "9001",
    ])
    .expect("wide CLI settings");
    config.checkpoint_cadence = crate::TrainingCheckpointCadence::Updates(1);
    config
}

#[test]
fn wide_capacity_profiles_and_partitions_admit_candidates_and_reject_overflow() {
    for (games, worlds, budget) in [
        (40, 40, crate::PpoSampleBudget::Annealed),
        (48, 48, crate::PpoSampleBudget::WideAnnealed),
        (64, 64, crate::PpoSampleBudget::WideAnnealed),
        (80, 40, crate::PpoSampleBudget::WideAnnealed),
    ] {
        let config = wide_settings(games, worlds);
        assert_eq!(
            validate_annealed(&config, harness())
                .expect("candidate")
                .sample_budget,
            budget
        );
        assert_eq!(
            config.ppo.environments * config.ppo.rollout_decisions,
            games * 1163
        );
    }
    let mut invalid = wide_settings(80, 40);
    invalid.parallel_worlds = 65;
    assert_eq!(
        validate_annealed(&invalid, harness()),
        Err(PpoError::InvalidConfig("annealed parallel worlds"))
    );
    invalid.parallel_worlds = 40;
    invalid.games_per_update = 81;
    assert_eq!(
        validate_annealed(&invalid, harness()),
        Err(PpoError::InvalidConfig(
            "annealed games per update must be even and within 2..=80"
        ))
    );
    let mut completed = CompletedTrainingEpisodes::default();
    completed
        .record(1, 79, TrainingGameOutcome::Win)
        .expect("last wide game");
    assert_eq!(
        completed.record(1, 80, TrainingGameOutcome::Win),
        Err(PpoError::InvalidTransition(
            "completed episode tick or stream"
        ))
    );
    assert_eq!(completed.ordered_outcomes(), [TrainingGameOutcome::Win]);
}

#[test]
fn m40_sequential_and_b40_updates_resume_model_optimizer_rng_and_generation_bytes() {
    for (parallel, updates, stop, interrupt) in [(8, 6, 5, Some(24)), (40, 2, 1, None)] {
        let baseline = test_directory("capacity-baseline");
        let resumed = test_directory("capacity-resumed");
        let mut config = expanded_settings(updates);
        config.parallel_worlds = parallel;
        let expected = run(config.clone(), &baseline, false).expect("complete M40 updates");
        assert_eq!(expected.games, updates * 40);
        assert_eq!(expected.completed_updates, updates);
        assert_eq!(expected.generations, (updates * 40).div_ceil(200));
        assert!((updates * 40..=updates * 80).contains(&expected.rollout_samples));
        assert!((1..=updates).contains(&expected.optimizer_step));
        let stopped = AnnealedHarness {
            stop_after: Some(stop),
            ..harness()
        };
        let first = run_with(config.clone(), stopped, &resumed, false).expect("M40 boundary");
        assert_eq!(first.games, stop * 40);
        assert_eq!(first.completed_updates, stop);
        if parallel == 8 {
            assert_expanded_resume_rejects_changed_dimensions(&config, &resumed);
        }
        if let Some(games) = interrupt {
            let before = checkpoint_digests(&resumed);
            let interrupted = AnnealedHarness {
                stop_after_games: Some(games),
                ..harness()
            };
            let error = run_with(config.clone(), interrupted, &resumed, true)
                .expect_err("uncommitted generation");
            assert_eq!(
                error.to_string(),
                "invalid PPO transition: annealed invocation stopped mid-update"
            );
            assert_eq!(checkpoint_digests(&resumed), before);
            assert_eq!(generation_files(&resumed).len(), 2);
        }
        let actual = run(config, &resumed, true).expect("replay and resume");
        assert_eq!(actual.completed_updates, updates);
        assert_eq!(actual.games, expected.games);
        assert_eq!(actual.rollout_samples, expected.rollout_samples);
        assert_eq!(actual.optimizer_step, expected.optimizer_step);
        assert_trajectory_equal(&baseline, &resumed);
        let files = generation_files(&resumed);
        if updates == 6 {
            assert!(files[0].1.contains("\"applied_games\":200"));
            assert!(files[1].1.contains("\"applied_games\":40"));
        }
        std::fs::remove_dir_all(baseline).expect("cleanup baseline");
        std::fs::remove_dir_all(resumed).expect("cleanup resumed");
    }
}

#[test]
fn expanded_jobs_enforce_capacity_and_partition_boundaries() {
    let mut maximum = expanded_settings(2);
    maximum.parallel_worlds = 40;
    let validated = validate_annealed(&maximum, AnnealedHarness::default()).expect("M40 B40");
    assert_eq!(validated.environments * validated.rollout_decisions, 46_520);
    assert_eq!(validated.sample_budget, crate::PpoSampleBudget::Annealed);
    let directory = test_directory("invalid-capacity");
    for (games, parallel, generation, message) in [
        (
            82,
            40,
            200,
            "annealed games per update must be even and within 2..=80",
        ),
        (40, 65, 200, "annealed parallel worlds"),
        (
            40,
            6,
            200,
            "annealed parallel worlds must divide games per update",
        ),
        (
            40,
            8,
            202,
            "annealed games per generation must be positive and divisible by parallel worlds",
        ),
    ] {
        let mut invalid = maximum.clone();
        invalid.games_per_update = games;
        invalid.ppo.environments = games;
        invalid.parallel_worlds = parallel;
        invalid.games_per_generation = generation;
        assert_eq!(
            run(invalid, &directory, false).expect_err("invalid job"),
            PpoError::InvalidConfig(message)
        );
        assert_eq!(std::fs::read_dir(&directory).expect("directory").count(), 0);
    }
    maximum.ppo.sample_budget = crate::PpoSampleBudget::Standard;
    assert_eq!(
        validate_annealed(&maximum, harness()),
        Err(PpoError::InvalidConfig(
            "annealed PPO sample budget must match games per update"
        ))
    );
    std::fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one() {
    for (games, epochs, minibatch, per_update, field) in [
        (40, 4, 80, 4 * (46_520 - 1), "annealed shuffle RNG counter"),
        (40, 1, 80, 46_520, "annealed sample counter"),
        (40, 2, 1, 2 * 46_520, "annealed optimizer counter"),
        (
            80,
            4,
            2048,
            4 * (93_040 - 1),
            "annealed shuffle RNG counter",
        ),
    ] {
        let mut config = expanded_settings(MAX_TRAINING_COUNTER / per_update);
        config.games_per_update = games;
        config.ppo.environments = games;
        config.ppo.sample_budget = crate::PpoSampleBudget::for_annealed_games(games);
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
        assert_eq!(
            validate_counter_budget(u64::MAX, 2, MAX_TRAINING_COUNTER, field),
            Err(PpoError::InvalidConfig(field))
        );
    }
}

#[test]
fn completed_episode_capacity_rejects_overflow_and_duplicate_merge_atomically() {
    let mut completed = CompletedTrainingEpisodes::default();
    completed
        .record(crate::MAP2_TICK_CAP, 79, TrainingGameOutcome::TimeCap)
        .expect("last valid tick and stream");
    let before = completed;
    assert_eq!(
        completed.record(1, 80, TrainingGameOutcome::Win),
        Err(PpoError::InvalidTransition(
            "completed episode tick or stream"
        ))
    );
    assert_eq!(completed, before);
    assert_eq!(
        completed.merge(&before),
        Err(PpoError::InvalidTransition(
            "duplicate completed episode stream"
        ))
    );
    assert_eq!(completed, before);
}

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
    .expect("M40 CLI settings");
    config.updates = updates;
    config.checkpoint_cadence = crate::TrainingCheckpointCadence::Updates(1);
    config
}

fn assert_expanded_resume_rejects_changed_dimensions(config: &AnnealedJobConfig, directory: &Path) {
    let before = checkpoint_digests(directory);
    for (games, parallel, generation, message) in [
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
        changed.games_per_generation = generation;
        let error = run(changed, directory, true).expect_err("changed M/B/K");
        assert_eq!(
            error.to_string(),
            format!("checkpoint scope mismatch: {message}")
        );
        assert_eq!(checkpoint_digests(directory), before);
    }
    type ConfigChange = fn(&mut AnnealedJobConfig);
    let changes: [(ConfigChange, &str); 3] = [
        (
            |config| config.updates += 1,
            "--updates: recorded 6, requested 7",
        ),
        (|config| config.seed += 1, "compatibility scope"),
        (
            |config| config.ppo.clip_epsilon = 0.1,
            "training checkpoint PPO config",
        ),
    ];
    for (change, message) in changes {
        let mut changed = config.clone();
        change(&mut changed);
        let error = run(changed, directory, true).expect_err("changed training identity");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(checkpoint_digests(directory), before);
    }
}
