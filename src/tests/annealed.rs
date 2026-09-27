//! Durable update, replay and rejection contracts; helper layouts are not contracts.

use crate::ppo::test_directory;
use std::path::PathBuf;

use super::*;
use crate::randomization::{AnnealSchedule, RANDOMIZATION_DIRECTORY, draw_generation};

#[path = "annealed_capacity.rs"]
mod capacity_tests;
#[path = "training_concurrency.rs"]
mod concurrency_tests;
#[path = "annealed_invocation.rs"]
mod invocation_tests;

#[test]
fn resume_requires_intact_generation_history_without_mutating_checkpoint() {
    for (damage, message) in [
        (0, "domain randomization snapshot mismatch"),
        (1, "domain randomization snapshot is missing on resume"),
        (2, "domain randomization snapshots are missing on resume"),
    ] {
        let directory = test_directory("generation-history");
        let mut config = settings(0x66bc, 1);
        config.games_per_update = 6;
        config.ppo.environments = 6;
        run(config.clone(), &directory, false).expect("three generations");
        assert_eq!(generation_files(&directory).len(), 3);
        let before = checkpoint_digests(&directory);
        let random = directory.join(RANDOMIZATION_DIRECTORY);
        let middle = random.join("generation-000000000001.json");
        match damage {
            0 => std::fs::write(middle, "{\"schema\":\"tampered\"}\n").expect("tamper"),
            1 => std::fs::remove_file(middle).expect("remove middle snapshot"),
            _ => std::fs::remove_dir_all(random).expect("remove history"),
        }
        let error = run(config, &directory, true).expect_err("damaged history");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(checkpoint_digests(&directory), before);
        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}

#[test]
fn cached_generation_metrics_turn_off_at_the_zero_window_boundary() {
    let schedule = AnnealSchedule {
        updates: 4,
        zero_updates: 1,
    };
    let draw = draw_generation(3, 1, 4, 2, schedule).expect("truncated draw");
    assert!(draw.scale_bp > 0);
    assert_eq!(draw.applied_games, 2);
    for (game, scale) in [
        (4, draw.scale_bp as u32),
        (5, draw.scale_bp as u32),
        (6, 0),
        (7, 0),
    ] {
        assert_eq!(generation_metrics(&draw, game), (1, scale));
        assert_eq!(generation_rules(&draw, game).is_empty(), scale == 0);
    }
}

#[test]
fn resume_rejects_swapped_opponent_weights() {
    let weights = test_directory("swap-weights");
    let directory = test_directory("swap-run");
    let first = PolicyModel::fresh(0x1111).expect("first opponent");
    TrainingArtifact::save_runtime_weights(&first, &weights).expect("first weights");
    let mut config = settings(0x1a2c, 1);
    config.opponent = AnnealedOpponent::Weights(weights.clone());
    run(config.clone(), &directory, false).expect("frozen opponent update");
    let before = checkpoint_digests(&directory);
    let second = PolicyModel::fresh(0x2222).expect("second opponent");
    TrainingArtifact::save_runtime_weights(&second, &weights).expect("swap weights");
    let error = run(config, &directory, true).expect_err("swapped weights");
    assert!(
        error.to_string().contains("--opponent-fingerprint"),
        "{error}"
    );
    assert_eq!(checkpoint_digests(&directory), before);
    std::fs::remove_dir_all(directory).expect("cleanup run");
    std::fs::remove_dir_all(weights).expect("cleanup weights");
}

#[test]
fn checkpoint_job_modes_reject_cross_resume_without_mutation() {
    for annealed in [false, true] {
        let directory = test_directory("cross-mode");
        let execute = |annealed, resume| {
            if annealed {
                return run(settings(0x7b01, 1), &directory, resume).map(|_| ());
            }
            let config = crate::cli::training_settings_for_test(&[
                "--complete-episodes=false",
                "--environments",
                "2",
                "--rollout",
                "2",
                "--epochs",
                "1",
                "--minibatch",
                "2",
                "--map",
                "2",
            ])
            .expect("window settings");
            crate::run_training_job_on_with_initial_weights(
                config,
                PolicyDevice::Cpu,
                &directory,
                resume,
                None,
                |_| {},
            )
            .map(|_| ())
        };
        execute(annealed, false).expect("source update");
        let before = checkpoint_digests(&directory);
        let error = execute(!annealed, true).expect_err("cross-mode resume");
        assert!(error.to_string().contains("compatibility scope"), "{error}");
        assert_eq!(checkpoint_digests(&directory), before);
        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}

fn settings(seed: u64, updates: u64) -> AnnealedJobConfig {
    AnnealedJobConfig {
        execution: crate::TrainingExecutionOptions::default(),
        updates,
        invocation_updates: None,
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
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        git_commit: "test-drysua-annealed".to_owned(),
        simulator_commit: "test-bota-annealed".to_owned(),
    }
}

fn harness() -> AnnealedHarness {
    AnnealedHarness {
        episode_decisions: Some(16),
        ..AnnealedHarness::default()
    }
}

fn generation_files(directory: &std::path::Path) -> Vec<(String, String)> {
    let mut files: Vec<_> = std::fs::read_dir(directory.join(RANDOMIZATION_DIRECTORY))
        .expect("generation directory")
        .map(|entry| {
            let entry = entry.expect("entry");
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read_to_string(entry.path()).expect("snapshot"),
            )
        })
        .collect();
    files.sort();
    files
}

fn artifact_runtime_state(artifact: &TrainingArtifact) -> (Vec<f32>, u64) {
    let model = PolicyModel::fresh(9_001).expect("fresh model");
    let state = artifact.restore(&model, artifact.run()).expect("restore");
    (
        model.export_parameters().expect("restored parameters"),
        state.trainer().optimizer_step(),
    )
}

fn run(
    settings: AnnealedJobConfig,
    directory: &std::path::Path,
    resume: bool,
) -> Result<AnnealedJobReport, PpoError> {
    run_with(settings, harness(), directory, resume)
}

fn run_with(
    settings: AnnealedJobConfig,
    harness: AnnealedHarness,
    directory: &std::path::Path,
    resume: bool,
) -> Result<AnnealedJobReport, PpoError> {
    run_annealed_job_harnessed(
        settings,
        harness,
        PolicyDevice::Cpu,
        directory,
        resume,
        None,
        |_| {},
    )
}

fn checkpoint_digests(directory: &std::path::Path) -> Vec<(String, String)> {
    use sha2::{Digest, Sha256};
    [
        "checkpoint.meta",
        "checkpoint.safetensors",
        "drysua.weights.safetensors",
    ]
    .into_iter()
    .map(|name| {
        let bytes = std::fs::read(directory.join(name)).expect("checkpoint file");
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        (name.to_owned(), digest)
    })
    .collect()
}

fn assert_trajectory_equal(source: &std::path::Path, target: &std::path::Path) {
    // These bytes include model, Adam, actor/shuffle RNG, progress, scope and runtime weights.
    assert_eq!(checkpoint_digests(source), checkpoint_digests(target));
    assert_eq!(generation_files(source), generation_files(target));
}
