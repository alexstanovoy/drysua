//! Native PPO collection and atomic adaptive-controller recovery contracts.

use std::num::NonZeroU64;

use super::*;
use crate::{
    AdaptiveEnvironmentCheckpoint, AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits,
    AdaptiveEnvironmentState, EnvironmentDecimal, EnvironmentSchedule,
};

#[test]
fn fractional_extension_survives_resume_and_retains_the_environment_until_update_five() {
    let baseline = test_directory("adaptive-fraction-baseline");
    let resumed = test_directory("adaptive-fraction-resumed");
    let config = adaptive_settings();
    let fixture = outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 1]);
    run_with(config.clone(), fixture, &baseline, false).expect("uninterrupted adaptive run");
    run_to(&config, fixture, &resumed, 1, false);
    let first = controller(&resumed);
    assert_eq!(first.config, AdaptiveEnvironmentConfig::default());
    assert_eq!(
        first.limits,
        AdaptiveEnvironmentLimits {
            base_updates: 4,
            total_updates: 8,
            zero_updates: 2,
        }
    );
    assert_eq!(first.state.extension_awards, 1);
    assert_eq!(first.state.updates_in_generation, 1);
    assert_eq!(first.snapshot_count, 1);
    assert_ne!(first.snapshot_hash, [0; 32]);
    run_to(&config, fixture, &resumed, 1, true);
    let second = controller(&resumed);
    assert_eq!(second.state.extension_awards, 2);
    assert_eq!(second.state.poor_streak, 1);
    assert_eq!(second.snapshot_hash, first.snapshot_hash);
    run_to(&config, fixture, &resumed, 2, true);
    let fourth = controller(&resumed);
    assert_eq!(fourth.state.generation, 0);
    assert_eq!(fourth.state.start_update, 0);
    assert_eq!(fourth.state.updates_in_generation, 4);
    assert_eq!(fourth.state.extension_awards, 2);
    assert_eq!(fourth.state.poor_streak, 0);
    run_to(&config, fixture, &resumed, 1, true);
    let fifth = controller(&resumed);
    assert_eq!(
        fifth.state,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 5,
            ..Default::default()
        }
    );
    assert_eq!(
        fifth.snapshot_count, 1,
        "no unused next-generation snapshot"
    );
    assert_eq!(fifth.snapshot_hash, first.snapshot_hash);
    run_with(config, fixture, &resumed, true).expect("finish from fractional extension boundary");
    assert_same_training(&baseline, &resumed, PolicyDevice::Cpu);
    for directory in [baseline, resumed] {
        std::fs::remove_dir_all(directory).expect("own cleanup");
    }
}

#[test]
fn early_success_at_update_two_resumes_with_the_actual_generation_start() {
    let baseline = test_directory("adaptive-success-baseline");
    let resumed = test_directory("adaptive-success-resumed");
    let mut config = adaptive_settings();
    config.updates = 4;
    config.zero_updates = 0;
    let fixture = outcome_harness(&[2, 2, 2, 2]);
    run_with(config.clone(), fixture, &baseline, false).expect("success baseline");
    run_to(&config, fixture, &resumed, 2, false);
    let boundary = controller(&resumed);
    assert_eq!(
        boundary.state,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 2,
            ..Default::default()
        }
    );
    assert_eq!(boundary.snapshot_count, 1);
    // The pipelined update three was already published under generation one.
    assert_eq!(generation_files(&resumed).len(), 2);
    run_with(config, fixture, &resumed, true).expect("resume early transition");
    let final_state = controller(&resumed);
    assert_eq!(final_state.state.start_update, 2);
    assert_eq!(final_state.state.updates_in_generation, 2);
    assert_eq!(final_state.state.success_streak, 2);
    assert_eq!(final_state.snapshot_count, 2);
    let generations = generation_files(&resumed);
    assert_eq!(
        generations[1].0,
        "adaptive-generation-0000000000000001.json"
    );
    assert!(generations[1].1.contains("\"start_update\":2"));
    assert_same_training(&baseline, &resumed, PolicyDevice::Cpu);
    for directory in [baseline, resumed] {
        std::fs::remove_dir_all(directory).expect("own cleanup");
    }
}

#[test]
fn clean_boundary_at_update_six_discards_a_long_extension_before_nominal_collection() {
    let directory = test_directory("adaptive-clean-boundary");
    let mut config = adaptive_settings();
    config.environment_schedule = EnvironmentSchedule::Adaptive(AdaptiveEnvironmentConfig {
        extension: EnvironmentDecimal::from_units(100_000_000),
        ..Default::default()
    });
    let fixture = outcome_harness(&[0, 0, 0, 0, 0, 0, 1, 1]);
    run_to(&config, fixture, &directory, 5, false);
    let extended = controller(&directory);
    assert_eq!(extended.state.generation, 0);
    assert_eq!(extended.state.extension_awards, 5);
    run_to(&config, fixture, &directory, 1, true);
    let boundary = controller(&directory);
    assert_eq!(
        boundary.state,
        AdaptiveEnvironmentState {
            generation: 1,
            start_update: 6,
            ..Default::default()
        }
    );
    assert_eq!(boundary.snapshot_count, 1);
    assert_eq!(boundary.snapshot_hash, extended.snapshot_hash);
    run_with(config, fixture, &directory, true).expect("clean final updates");
    let final_state = controller(&directory);
    assert_eq!(final_state.state.start_update, 6);
    assert_eq!(final_state.state.updates_in_generation, 2);
    assert_eq!(final_state.snapshot_count, 2);
    let generations = generation_files(&directory);
    assert_eq!(generations.len(), 2);
    assert!(generations[1].1.contains("\"start_update\":6"));
    assert!(generations[1].1.contains("\"scale_bp\":0,"));
    assert_ne!(final_state.snapshot_hash, extended.snapshot_hash);
    std::fs::remove_dir_all(directory).expect("own cleanup");
}

#[test]
fn every_adaptive_parameter_mismatch_rejects_before_mutating_the_training_tree() {
    let directory = test_directory("adaptive-config-mismatch");
    let config = adaptive_settings();
    let fixture = outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 1]);
    run_to(&config, fixture, &directory, 1, false);
    let before = tree_snapshot(&directory);
    for (field, flag) in [
        (0, "--environment-success-updates"),
        (1, "--environment-success-rate"),
        (2, "--environment-poor-updates"),
        (3, "--environment-poor-rate"),
        (4, "--environment-extension"),
    ] {
        let mut changed = config.clone();
        let mut adaptive = AdaptiveEnvironmentConfig::default();
        match field {
            0 => adaptive.success_updates = 3,
            1 => adaptive.success_rate = EnvironmentDecimal::from_units(900_000),
            2 => adaptive.poor_updates = 2,
            3 => adaptive.poor_rate = EnvironmentDecimal::from_units(100_000),
            4 => adaptive.extension = EnvironmentDecimal::from_units(500_000),
            _ => unreachable!("five bounded parameters"),
        }
        changed.environment_schedule = EnvironmentSchedule::Adaptive(adaptive);
        let message = run_with(changed, fixture, &directory, true)
            .expect_err("changed adaptive parameter")
            .to_string();
        assert!(
            message.starts_with("checkpoint scope mismatch:"),
            "{message}"
        );
        assert!(message.contains(flag), "{message}");
        assert_eq!(tree_snapshot(&directory), before, "{flag}");
    }
    let artifact = TrainingArtifact::load(&directory).expect("intact original scope");
    assert!(
        artifact
            .run()
            .command_line
            .ends_with(&AdaptiveEnvironmentConfig::default().scope_suffix())
    );
    std::fs::remove_dir_all(directory).expect("own cleanup");
}

#[test]
fn fixed_and_adaptive_cross_resume_rejects_without_mutating_either_training_tree() {
    for adaptive_source in [false, true] {
        let directory = test_directory("adaptive-fixed-cross-resume");
        let mut source = adaptive_settings();
        let mut target = source.clone();
        let adaptive_fixture = outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 1]);
        let (source_fixture, target_fixture) = if adaptive_source {
            target.environment_schedule = EnvironmentSchedule::Fixed;
            (adaptive_fixture, harness())
        } else {
            source.environment_schedule = EnvironmentSchedule::Fixed;
            (harness(), adaptive_fixture)
        };
        run_to(&source, source_fixture, &directory, 1, false);
        let before = tree_snapshot(&directory);
        let error = run_with(target, target_fixture, &directory, true)
            .expect_err("schedule kind is part of checkpoint compatibility");
        assert!(
            error.to_string().starts_with("checkpoint scope mismatch:"),
            "{error}"
        );
        assert_eq!(tree_snapshot(&directory), before);
        std::fs::remove_dir_all(directory).expect("own cleanup");
    }
}

#[test]
fn outcome_fixture_scope_binds_future_results_not_only_the_committed_prefix() {
    let directory = test_directory("adaptive-outcome-scope");
    let config = adaptive_settings();
    run_to(
        &config,
        outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 1]),
        &directory,
        1,
        false,
    );
    let before = tree_snapshot(&directory);
    let error = run_with(
        config,
        outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 2]),
        &directory,
        true,
    )
    .expect_err("changed future fixture outcome");
    assert!(
        error.to_string().starts_with("checkpoint scope mismatch:"),
        "{error}"
    );
    assert_eq!(tree_snapshot(&directory), before);
    std::fs::remove_dir_all(directory).expect("own cleanup");
}

#[test]
fn corrupt_adaptive_snapshot_resume_is_rejected_without_repairing_or_mutating_files() {
    let directory = test_directory("adaptive-corrupted-snapshot");
    let config = adaptive_settings();
    let fixture = outcome_harness(&[0, 0, 1, 1, 1, 1, 1, 1]);
    run_to(&config, fixture, &directory, 1, false);
    let checkpoint = checkpoint_digests(&directory);
    let snapshot = directory
        .join(RANDOMIZATION_DIRECTORY)
        .join("adaptive-generation-0000000000000000.json");
    assert!(snapshot.is_file());
    std::fs::write(&snapshot, "{\"schema\":\"tampered\"}\n")
        .expect("intentional snapshot corruption");
    let damaged = tree_snapshot(&directory);
    let error = run_with(config, fixture, &directory, true).expect_err("corrupt snapshot");
    let message = error.to_string();
    assert!(message.contains("adaptive"), "{message}");
    assert!(message.contains("snapshot"), "{message}");
    assert_eq!(checkpoint_digests(&directory), checkpoint);
    assert_eq!(tree_snapshot(&directory), damaged);
    std::fs::remove_dir_all(directory).expect("own cleanup");
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "exclusive CUDA: two-update M2 native adaptive replay, no outcome fixture"]
fn cuda_full_native_adaptive_resume_preserves_exact_training_state() {
    let baseline = test_directory("adaptive-cuda-native-baseline");
    let resumed = test_directory("adaptive-cuda-native-resumed");
    let mut config = adaptive_settings();
    config.updates = 2;
    config.zero_updates = 0;
    config.ppo.samples_per_update = 256;
    config.ppo.minibatch = 128;
    let device = PolicyDevice::Cuda { ordinal: 0 };
    let execute = |config: AnnealedJobConfig, directory: &Path, resume| {
        run_annealed_job_harnessed(
            config,
            AnnealedHarness::default(),
            device,
            directory,
            resume,
            None,
            |_| {},
        )
        .expect("full native adaptive CUDA invocation")
    };
    execute(config.clone(), &baseline, false);
    let mut first = config.clone();
    first.invocation_updates = NonZeroU64::new(1);
    assert_eq!(execute(first, &resumed, false).completed_updates, 1);
    assert_eq!(execute(config, &resumed, true).completed_updates, 2);
    assert_same_training(&baseline, &resumed, device);
    for directory in [baseline, resumed] {
        std::fs::remove_dir_all(directory).expect("own cleanup");
    }
}

fn adaptive_settings() -> AnnealedJobConfig {
    let mut config = settings(0xa6a9, 8);
    config.environment_schedule =
        EnvironmentSchedule::Adaptive(AdaptiveEnvironmentConfig::default());
    config.generation_updates = 4;
    config.zero_updates = 2;
    config
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "exclusive CUDA initial weights and two-update M4/G2 native replay"]
fn cuda_side_actors_update_both_heads_and_resume_exactly() {
    let baseline = test_directory("side-actor-native-baseline");
    let resumed = test_directory("side-actor-native-resumed");
    let weights = test_directory("side-actor-native-weights");
    let source = weights.as_path();
    let mut config = adaptive_settings();
    config.updates = 2;
    config.zero_updates = 0;
    config.slots = 4;
    config.ppo.samples_per_update = 256;
    config.ppo.minibatch = 128;
    config.execution.training_microbatch = 256;
    let device = PolicyDevice::Cuda { ordinal: 0 };
    let initial = PolicyModel::fresh(0x5a1d).unwrap();
    TrainingArtifact::save_runtime_weights(&initial, source).unwrap();
    let before = initial.export_parameters().unwrap();
    drop(initial);
    let execute = |config, directory: &Path, resume| {
        run_annealed_job_harnessed(
            config,
            AnnealedHarness::default(),
            device,
            directory,
            resume,
            (!resume).then_some(source),
            |_| {},
        )
        .unwrap()
    };
    execute(config.clone(), &baseline, false);
    let mut first = config.clone();
    first.invocation_updates = NonZeroU64::new(1);
    assert_eq!(execute(first, &resumed, false).completed_updates, 1);
    assert_eq!(execute(config, &resumed, true).completed_updates, 2);
    assert_same_training(&baseline, &resumed, device);
    let trained = PolicyModel::fresh(0).unwrap();
    TrainingArtifact::load_runtime_weights(&trained, &baseline).unwrap();
    let after = trained.export_parameters().unwrap();
    let changed = |range: std::ops::Range<usize>| {
        range
            .into_iter()
            .any(|index| before[index].to_bits() != after[index].to_bits())
    };
    assert!(
        changed(1_651_905..1_656_017) || changed(1_656_961..1_765_812),
        "Radiant actor must train"
    );
    assert!(changed(1_765_812..1_878_775), "Dire actor must train");
    eprintln!(
        "side-actor-native parameters=1878775 tensors=88 slots=4 lanes=2 microbatch=256 both_actor_heads_changed=true exact_model_adam_rng_controller_snapshots_resume=true"
    );
    for directory in [baseline, resumed, weights] {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

fn outcome_harness(wins: &'static [u64]) -> AnnealedHarness {
    assert!(!wins.is_empty());
    assert!(wins.len() <= 8);
    assert!(wins.iter().all(|wins| *wins <= 2));
    AnnealedHarness {
        adaptive_wins: Some(wins),
        ..harness()
    }
}

fn run_to(
    config: &AnnealedJobConfig,
    fixture: AnnealedHarness,
    directory: &Path,
    additional: u64,
    resume: bool,
) {
    assert!(additional > 0);
    assert!(additional <= config.updates);
    let mut config = config.clone();
    config.invocation_updates = NonZeroU64::new(additional);
    run_with(config, fixture, directory, resume).expect("bounded adaptive invocation");
}

fn controller(directory: &Path) -> AdaptiveEnvironmentCheckpoint {
    TrainingArtifact::load(directory)
        .expect("adaptive artifact")
        .progress()
        .adaptive_environment
        .expect("durable adaptive controller")
}

fn assert_same_training(source: &Path, target: &Path, device: PolicyDevice) {
    assert_trajectory_equal(source, target);
    assert_artifact_bits(source, target, device);
}

fn tree_snapshot(directory: &Path) -> Vec<(PathBuf, Option<[u8; 32]>)> {
    use sha2::{Digest, Sha256};
    let mut pending = vec![directory.to_path_buf()];
    let mut snapshot = Vec::new();
    for _ in 0..256 {
        let Some(path) = pending.pop() else {
            snapshot.sort();
            return snapshot;
        };
        let metadata = std::fs::symlink_metadata(&path).expect("tree entry metadata");
        assert!(!metadata.file_type().is_symlink());
        let relative = path
            .strip_prefix(directory)
            .expect("owned tree")
            .to_path_buf();
        let digest = if metadata.is_dir() {
            for entry in std::fs::read_dir(&path).expect("tree directory") {
                assert!(pending.len() + snapshot.len() < 256, "bounded fixture tree");
                pending.push(entry.expect("tree entry").path());
            }
            None
        } else {
            assert!(metadata.is_file());
            assert!(metadata.len() <= 64 * 1024 * 1024);
            Some(Sha256::digest(std::fs::read(&path).expect("tree bytes")).into())
        };
        snapshot.push((relative, digest));
    }
    panic!("fixture tree exceeds 256 entries");
}
