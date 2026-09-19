//! Annealed loop: validation, side balance, generation rules and snapshots,
//! stop/restart determinism and opponent pinning.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::UnitKind;
use bota_server::game::{SpawnCategory, SpawnTarget};

use super::*;
use crate::randomization::{AnnealSchedule, RANDOMIZATION_DIRECTORY, draw_generation};

fn test_directory(name: &str) -> PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-annealed-{name}-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        std::fs::remove_dir_all(&directory).expect("remove stale directory");
    }
    std::fs::create_dir(&directory).expect("create directory");
    directory
}

fn settings(seed: u64, updates: u64) -> AnnealedJobConfig {
    AnnealedJobConfig {
        updates,
        games_per_update: 2,
        parallel_worlds: 2,
        games_per_generation: 2,
        zero_updates: 0,
        seed,
        opponent: AnnealedOpponent::Teacher,
        ppo: PpoConfig {
            decision_interval_ticks: MAP2_DECISION_INTERVAL_TICKS,
            environments: 2,
            rollout_decisions: MAP2_RETAINED_DECISIONS,
            epochs: 1,
            minibatch: 2,
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
        episode_decisions: 16,
        stop_after: None,
        stop_after_games: None,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        git_commit: "test-drysua-annealed".to_owned(),
        simulator_commit: "test-bota-annealed".to_owned(),
    }
}

fn generation_files(directory: &std::path::Path) -> Vec<(String, String)> {
    let random = directory.join(RANDOMIZATION_DIRECTORY);
    let mut files = std::fs::read_dir(&random)
        .expect("generation directory")
        .map(|entry| {
            let entry = entry.expect("entry");
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read_to_string(entry.path()).expect("snapshot"),
            )
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn artifact_snapshot(artifact: &TrainingArtifact) -> (Vec<f32>, Vec<f32>, Vec<f32>, u64, u64) {
    let model = PolicyModel::fresh(9_001).expect("fresh model");
    let state = artifact.restore(&model, artifact.run()).expect("restore");
    let snapshot = state
        .trainer()
        .checkpoint_snapshot(&model)
        .expect("checkpoint snapshot");
    let (first, second) = snapshot.adam.moments();
    (
        snapshot.parameters,
        first.to_vec(),
        second.to_vec(),
        state.trainer().optimizer_step(),
        state.trainer().updates(),
    )
}

fn run(
    settings: AnnealedJobConfig,
    directory: &std::path::Path,
    resume: bool,
) -> Result<AnnealedJobReport, PpoError> {
    run_annealed_job_on_with_initial_weights(
        settings,
        PolicyDevice::Cpu,
        directory,
        resume,
        None,
        |_| {},
    )
}

#[test]
fn validation_rejects_divisibility_and_budget_mistakes() {
    let mut config = settings(1, 2);
    config.games_per_update = 3;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("odd games")
            .to_string(),
        "invalid PPO config field: annealed games per update must be even and within the environment ceiling"
    );
    let mut config = settings(1, 2);
    config.parallel_worlds = 3;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("parallel does not divide games")
            .to_string(),
        "invalid PPO config field: annealed parallel worlds must divide games per update"
    );
    let mut config = settings(1, 2);
    config.games_per_generation = 3;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("generation games do not divide")
            .to_string(),
        "invalid PPO config field: annealed games per generation must be positive and divisible by parallel worlds"
    );
    let mut config = settings(1, 2);
    config.updates = 0;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("zero updates")
            .to_string(),
        "invalid PPO config field: annealed updates"
    );
    let mut config = settings(1, 2);
    config.zero_updates = 3;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("zero window past budget")
            .to_string(),
        "invalid PPO config field: annealed zero updates cannot exceed the update budget"
    );
    let mut config = settings(1, 2);
    config.episode_decisions = 0;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("empty episode")
            .to_string(),
        "invalid PPO config field: annealed episode decisions"
    );
    let mut config = settings(1, 2);
    config.ppo.environments = 4;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("environment mismatch")
            .to_string(),
        "invalid PPO config field: annealed PPO environments must equal games per update"
    );
    let mut config = settings(1, 2);
    config.ppo.gamma_tick = 0.99;
    assert_eq!(
        validate_annealed(&config)
            .expect_err("gamma mismatch")
            .to_string(),
        "invalid PPO config field: annealed Map2 reward requires gamma per tick one"
    );
}

#[test]
fn every_update_splits_sides_exactly_evenly() {
    for games in [2usize, 4, 8, 16, 26] {
        for update in 0..64u64 {
            let seats = balanced_policy_seats(0x51de, update, games).expect("seats");
            assert_eq!(seats.len(), games);
            assert_eq!(seats.iter().filter(|seat| **seat == 0).count(), games / 2);
            assert_eq!(seats.iter().filter(|seat| **seat == 1).count(), games / 2);
        }
    }
}

#[test]
fn a_zero_scale_generation_writes_a_nominal_snapshot() {
    let directory = test_directory("zero-scale");
    let schedule = AnnealSchedule {
        updates: 4,
        zero_updates: 4,
    };
    let mut cache = GenerationCache::new(
        directory.join(RANDOMIZATION_DIRECTORY),
        7,
        2,
        2,
        schedule,
        0,
    );
    let draw = cache.draw(0).expect("draw");
    assert!(!draw.applies());
    assert_eq!(draw.applied_games, 0);
    assert!(generation_rules(&draw, 0).is_empty());
    let files = generation_files(&directory);
    assert_eq!(files.len(), 1);
    assert!(files[0].1.contains("\"scale_bp\":0"));
    assert!(files[0].1.contains("\"applied_games\":0"));
    assert!(files[0].1.contains("\"max_hp\":10000"));
    assert!(files[0].1.contains("\"mana_cost_rate\":10000"));
    assert_eq!(cache.counted_through(), 1);
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn a_modified_generation_draws_a_bounded_spec() {
    let directory = test_directory("bounded");
    let schedule = AnnealSchedule {
        updates: 100,
        zero_updates: 0,
    };
    let mut cache = GenerationCache::new(
        directory.join(RANDOMIZATION_DIRECTORY),
        7,
        2,
        2,
        schedule,
        0,
    );
    let draw = cache.draw(0).expect("draw");
    assert!(draw.spec.is_bounded());
    assert!(!draw.spec.is_nominal());
    assert!(draw.applies());
    assert!(!generation_rules(&draw, 0).is_empty());
    let files = generation_files(&directory);
    assert_eq!(files.len(), 1);
    assert!(files[0].1.contains("\"scale_bp\":10000"));
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn generation_rules_select_the_documented_targets() {
    assert!(spawn_modifiers_for(ModifierSpec::NOMINAL).is_empty());
    let spec = ModifierSpec {
        max_hp: 11_000,
        physical_damage: 11_000,
        move_speed: 11_000,
        mana_cost_rate: 9_000,
        gold_income: 11_000,
        ..ModifierSpec::NOMINAL
    };
    let rules = spawn_modifiers_for(spec);
    assert_eq!(rules.len(), 3, "one rule per target set");
    let wide = &rules[0];
    assert_eq!(wide.spec.max_hp, spec.max_hp);
    for target in [
        SpawnTarget::Category(SpawnCategory::Hero),
        SpawnTarget::Category(SpawnCategory::LaneCreep),
        SpawnTarget::Category(SpawnCategory::NeutralCreep),
        SpawnTarget::Category(SpawnCategory::Structure),
    ] {
        assert!(
            wide.select.targets.contains(&target),
            "max HP takes {target:?}"
        );
    }
    let creeps = &rules[1];
    assert_eq!(creeps.spec.physical_damage, spec.physical_damage);
    assert_eq!(creeps.spec.move_speed, spec.move_speed);
    assert!(creeps.select.takes(UnitKind::Hero, None));
    assert!(creeps.select.takes(UnitKind::CreepMelee, None));
    assert!(creeps.select.takes(UnitKind::CreepNeutral, None));
    assert!(!creeps.select.takes(UnitKind::Tower, None));
    assert!(!creeps.select.takes(UnitKind::Ward, None));
    let heroes = &rules[2];
    assert_eq!(heroes.spec.mana_cost_rate, spec.mana_cost_rate);
    assert_eq!(heroes.spec.gold_income, spec.gold_income);
    assert!(heroes.select.takes(UnitKind::Hero, None));
    assert!(!heroes.select.takes(UnitKind::CreepMelee, None));
    assert!(!heroes.select.takes(UnitKind::Tower, None));
}

#[test]
fn generation_rules_follow_the_zero_window() {
    // Span 3 updates, 2 games each; generation 1 (games 4..8) applies to two.
    let schedule = AnnealSchedule {
        updates: 4,
        zero_updates: 1,
    };
    let draw = draw_generation(3, 1, 4, 2, schedule).expect("draw");
    assert_eq!(draw.applied_games, 2);
    assert!(!generation_rules(&draw, 4).is_empty());
    assert!(!generation_rules(&draw, 5).is_empty());
    assert!(generation_rules(&draw, 6).is_empty());
    let after = draw_generation(3, 2, 4, 2, schedule).expect("after");
    assert!(generation_rules(&after, 8).is_empty());
}

#[test]
fn a_zero_temperature_run_writes_only_nominal_generations() {
    let directory = test_directory("zero-run");
    let mut config = settings(0x7e10, 1);
    config.zero_updates = 1;
    let report = run(config, &directory, false).expect("zero-temperature run");
    assert_eq!(report.completed_updates, 1);
    let files = generation_files(&directory);
    assert_eq!(files.len(), 1);
    assert!(files[0].1.contains("\"scale_bp\":0"));
    assert!(files[0].1.contains("\"applied_games\":0"));
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn a_run_plays_exactly_m_games_per_update() {
    let directory = test_directory("games");
    let report = run(settings(0x600d, 2), &directory, false).expect("two updates");
    assert_eq!(report.completed_updates, 2);
    assert_eq!(report.games, 4, "two updates of two games");
    assert!(
        report.rollout_samples >= report.games,
        "every game contributes at least one sample"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn a_single_world_batch_and_an_odd_batch_both_run() {
    let directory = test_directory("batches");
    let mut config = settings(0xbeef, 1);
    config.games_per_update = 6;
    config.parallel_worlds = 3;
    config.games_per_generation = 6;
    config.ppo.environments = 6;
    let report = run(config, &directory, false).expect("odd parallel batch");
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.generations, 1);
    assert_eq!(report.games, 6);
    std::fs::remove_dir_all(directory).expect("remove directory");

    let directory = test_directory("serial-batch");
    let mut config = settings(0xcafe, 1);
    config.parallel_worlds = 1;
    let report = run(config, &directory, false).expect("single world batch");
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.games, 2);
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn stop_and_restart_reproduces_the_uninterrupted_run() {
    let uninterrupted_directory = test_directory("uninterrupted");
    let resumed_directory = test_directory("resumed");
    let uninterrupted =
        run(settings(0x1234, 2), &uninterrupted_directory, false).expect("uninterrupted run");
    let mut first_settings = settings(0x1234, 2);
    first_settings.stop_after = Some(1);
    let first = run(first_settings, &resumed_directory, false).expect("first update");
    assert_eq!(first.completed_updates, 1);
    let resumed = run(settings(0x1234, 2), &resumed_directory, true).expect("resumed run");

    assert_eq!(uninterrupted.completed_updates, 2);
    assert_eq!(resumed.completed_updates, 2);
    assert_eq!(uninterrupted.generations, 2);
    assert_eq!(resumed.generations, 2);
    assert_eq!(uninterrupted.games, resumed.games);
    assert_eq!(uninterrupted.rollout_samples, resumed.rollout_samples);

    let uninterrupted_artifact =
        TrainingArtifact::load(&uninterrupted_directory).expect("uninterrupted artifact");
    let resumed_artifact = TrainingArtifact::load(&resumed_directory).expect("resumed artifact");
    assert_eq!(uninterrupted_artifact.run(), resumed_artifact.run());
    assert_eq!(
        uninterrupted_artifact.progress(),
        resumed_artifact.progress()
    );
    assert_eq!(
        artifact_snapshot(&uninterrupted_artifact),
        artifact_snapshot(&resumed_artifact)
    );
    assert_eq!(
        generation_files(&uninterrupted_directory),
        generation_files(&resumed_directory)
    );
    std::fs::remove_dir_all(uninterrupted_directory).expect("remove uninterrupted directory");
    std::fs::remove_dir_all(resumed_directory).expect("remove resumed directory");
}

#[test]
fn a_mid_update_stop_replays_the_update() {
    let uninterrupted_directory = test_directory("mid-uninterrupted");
    let stopped_directory = test_directory("mid-stopped");
    let mut config = settings(0x9abc, 2);
    config.games_per_update = 4;
    config.parallel_worlds = 2;
    config.games_per_generation = 4;
    config.ppo.environments = 4;
    let uninterrupted =
        run(config.clone(), &uninterrupted_directory, false).expect("uninterrupted run");

    let mut first_settings = config.clone();
    first_settings.stop_after = Some(1);
    run(first_settings, &stopped_directory, false).expect("first update");

    let mut stopped = config.clone();
    stopped.stop_after_games = Some(2);
    let error = run(stopped, &stopped_directory, true).expect_err("mid-update stop");
    assert_eq!(
        error.to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(
        TrainingArtifact::load(&stopped_directory)
            .expect("stopped artifact")
            .progress()
            .global_update,
        1,
        "a mid-update stop leaves the last checkpoint untouched"
    );

    let resumed = run(config, &stopped_directory, true).expect("resumed run");
    assert_eq!(resumed.completed_updates, 2);
    assert_eq!(uninterrupted.games, resumed.games);

    let uninterrupted_artifact =
        TrainingArtifact::load(&uninterrupted_directory).expect("uninterrupted artifact");
    let resumed_artifact = TrainingArtifact::load(&stopped_directory).expect("resumed artifact");
    assert_eq!(uninterrupted_artifact.run(), resumed_artifact.run());
    assert_eq!(
        uninterrupted_artifact.progress(),
        resumed_artifact.progress()
    );
    assert_eq!(
        artifact_snapshot(&uninterrupted_artifact),
        artifact_snapshot(&resumed_artifact)
    );
    assert_eq!(
        generation_files(&uninterrupted_directory),
        generation_files(&stopped_directory)
    );
    std::fs::remove_dir_all(uninterrupted_directory).expect("remove uninterrupted directory");
    std::fs::remove_dir_all(stopped_directory).expect("remove stopped directory");
}

#[test]
fn resume_rejects_changed_loop_parameters() {
    let directory = test_directory("changed");
    run(settings(0x55aa, 1), &directory, false).expect("first update");
    let mut changed = settings(0x55aa, 1);
    changed.games_per_generation = 4;
    let error = run(changed, &directory, true).expect_err("changed generation games");
    assert!(
        error.to_string().contains("compatibility scope"),
        "unexpected error: {error}"
    );

    let directory = test_directory("changed-updates");
    let mut first_settings = settings(0x55ab, 2);
    first_settings.stop_after = Some(1);
    run(first_settings, &directory, false).expect("first update of a two-update run");
    let error = run(settings(0x55ab, 3), &directory, true).expect_err("changed update budget");
    assert!(
        error.to_string().contains("compatibility scope"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_rejects_a_changed_ppo_hyperparameter() {
    let directory = test_directory("changed-ppo");
    run(settings(0x55ac, 1), &directory, false).expect("first update");
    let mut changed = settings(0x55ac, 1);
    changed.ppo.clip_epsilon = 0.1;
    let error = run(changed, &directory, true).expect_err("changed clip epsilon");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: training checkpoint PPO config"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_rejects_a_tampered_generation_snapshot() {
    let directory = test_directory("tampered");
    run(settings(0x66bb, 1), &directory, false).expect("first update");
    let path = directory
        .join(RANDOMIZATION_DIRECTORY)
        .join("generation-000000000000.json");
    std::fs::write(&path, "{\"schema\":\"tampered\"}\n").expect("tamper");
    let error = run(settings(0x66bb, 1), &directory, true).expect_err("tampered snapshot");
    assert!(
        error
            .to_string()
            .contains("domain randomization snapshot mismatch"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_rejects_a_missing_generation_snapshot() {
    let directory = test_directory("missing-snapshot");
    let mut config = settings(0x66bc, 1);
    config.games_per_update = 6;
    config.parallel_worlds = 2;
    config.games_per_generation = 2;
    config.ppo.environments = 6;
    run(config.clone(), &directory, false).expect("first update");
    let files = generation_files(&directory);
    assert_eq!(files.len(), 3, "six games make three generations");
    let path = directory
        .join(RANDOMIZATION_DIRECTORY)
        .join("generation-000000000001.json");
    std::fs::remove_file(&path).expect("remove snapshot");
    let error = run(config, &directory, true).expect_err("missing snapshot");
    assert!(
        error
            .to_string()
            .contains("domain randomization snapshot is missing on resume"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_rejects_a_missing_randomization_directory() {
    let directory = test_directory("missing-directory");
    run(settings(0x66bd, 1), &directory, false).expect("first update");
    std::fs::remove_dir_all(directory.join(RANDOMIZATION_DIRECTORY)).expect("remove directory");
    let error = run(settings(0x66bd, 1), &directory, true).expect_err("missing directory");
    assert!(
        error
            .to_string()
            .contains("domain randomization snapshots are missing on resume"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn a_frozen_weights_opponent_is_loaded_once_and_pinned() {
    let directory = test_directory("opponent");
    let model = PolicyModel::fresh(0x7777).expect("opponent model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("save runtime weights");
    let loaded = load_opponent(
        &AnnealedOpponent::Weights(directory.clone()),
        PolicyDevice::Cpu,
    )
    .expect("weights opponent");
    assert!(loaded.fingerprint.is_some(), "weights are fingerprinted");
    let AnnealedOpponentRuntime::Weights(_) = &loaded.runtime else {
        panic!("weights opponent must be a policy runtime");
    };
    let first = loaded.runtime.spec();
    let second = loaded.runtime.spec();
    match (&first, &second) {
        (OpponentSpec::SharedPolicy(first), OpponentSpec::SharedPolicy(second)) => {
            assert!(Arc::ptr_eq(first, second), "one frozen model is shared");
        }
        _ => panic!("weights opponent must use a shared policy spec"),
    }
    let teacher =
        load_opponent(&AnnealedOpponent::Teacher, PolicyDevice::Cpu).expect("teacher opponent");
    assert!(teacher.fingerprint.is_none());
    assert!(matches!(teacher.runtime, AnnealedOpponentRuntime::Teacher));
    std::fs::remove_dir_all(directory).expect("remove directory");
}

#[test]
fn resume_rejects_swapped_opponent_weights() {
    let weights = test_directory("swap-weights");
    let directory = test_directory("swap-run");
    let first = PolicyModel::fresh(0x1111).expect("first opponent");
    TrainingArtifact::save_runtime_weights(&first, &weights).expect("save first weights");
    let mut config = settings(0x1a2c, 1);
    config.opponent = AnnealedOpponent::Weights(weights.clone());
    run(config.clone(), &directory, false).expect("first update");

    let second = PolicyModel::fresh(0x2222).expect("second opponent");
    TrainingArtifact::save_runtime_weights(&second, &weights).expect("swap weights");
    let error = run(config, &directory, true).expect_err("swapped weights");
    assert!(
        error.to_string().contains("compatibility scope"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
    std::fs::remove_dir_all(weights).expect("remove weights");
}

#[test]
fn a_weights_opponent_is_recorded_in_the_run_scope() {
    let directory = test_directory("weights-run");
    let weights = test_directory("weights-dir");
    let model = PolicyModel::fresh(0x1a2b).expect("opponent model");
    TrainingArtifact::save_runtime_weights(&model, &weights).expect("save runtime weights");
    let mut config = settings(0x1a2b, 1);
    config.opponent = AnnealedOpponent::Weights(weights.clone());
    let report = run(config, &directory, false).expect("weights opponent run");
    assert_eq!(report.completed_updates, 1);
    let artifact = TrainingArtifact::load(&directory).expect("artifact");
    assert!(
        artifact.run().command_line.contains("--opponent-weights"),
        "run scope records the frozen weights directory"
    );
    assert!(
        artifact
            .run()
            .command_line
            .contains("--opponent-fingerprint"),
        "run scope pins the frozen tensors"
    );
    std::fs::remove_dir_all(directory).expect("remove directory");
    std::fs::remove_dir_all(weights).expect("remove weights");
}

#[test]
fn cli_defaults_zero_window_and_parallel_divisor() {
    assert_eq!(crate::cli::default_parallel_worlds_for(8, 8, 6), 4);
    assert_eq!(crate::cli::default_parallel_worlds_for(8, 8, 8), 8);
    assert_eq!(crate::cli::default_parallel_worlds_for(8, 4, 3), 2);
    assert_eq!(crate::cli::default_parallel_worlds_for(6, 10, 2), 2);
    assert_eq!(crate::cli::default_parallel_worlds_for(26, 26, 1), 1);

    let settings = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "10",
        "--games",
        "4",
        "--generation-games",
        "4",
    ])
    .expect("annealed settings");
    assert_eq!(
        settings.zero_updates, 2,
        "default is one fifth of the budget"
    );
    assert!(settings.parallel_worlds >= 1);
    assert!(4usize.is_multiple_of(settings.parallel_worlds));
    assert_eq!(settings.ppo.environments, 4);
    assert_eq!(settings.episode_decisions, ANNEALED_EPISODE_DECISIONS);
    assert_eq!(settings.opponent, AnnealedOpponent::Teacher);
}

#[test]
fn cli_rejects_opponent_misuse() {
    let error = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "2",
        "--generation-games",
        "2",
        "--opponent",
        "weights",
    ])
    .expect_err("weights without a directory");
    assert_eq!(
        error.to_string(),
        "weights opponent requires --opponent-weights"
    );
    let error = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "2",
        "--generation-games",
        "2",
        "--opponent",
        "teacher",
        "--opponent-weights",
        "/tmp/nowhere",
    ])
    .expect_err("teacher with a weights directory");
    assert_eq!(
        error.to_string(),
        "teacher opponent forbids --opponent-weights"
    );
}
