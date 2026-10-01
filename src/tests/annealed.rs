//! Durable update, replay and rejection contracts; helper layouts are not contracts.

use crate::ppo::test_directory;
use std::path::PathBuf;

use super::*;
use crate::randomization::{AnnealSchedule, RANDOMIZATION_DIRECTORY, draw_generation};

#[path = "adaptive_annealed.rs"]
mod adaptive_annealed;
#[path = "annealed_capacity.rs"]
mod capacity_tests;
#[path = "annealed_invocation.rs"]
mod invocation_tests;
#[path = "annealed_seed.rs"]
mod seed_tests;
#[cfg(unix)]
#[path = "annealed_session.rs"]
mod session_tests;

#[test]
fn resume_requires_intact_generation_history_without_mutating_checkpoint() {
    for (damage, message) in [
        (0, "domain randomization snapshot mismatch"),
        (1, "domain randomization snapshot is missing on resume"),
        (2, "domain randomization snapshots are missing on resume"),
    ] {
        let directory = test_directory("generation-history");
        // Committing update one has drawn the generations of every collected
        // update: one, and the pipelined two.
        let mut config = settings(0x66bc, 3);
        config.invocation_updates = std::num::NonZeroU64::new(1);
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
    }
}

#[test]
fn cached_generation_rules_turn_off_at_the_zero_window_boundary() {
    let schedule = AnnealSchedule {
        updates: 4,
        zero_updates: 1,
        scale: crate::randomization::AnnealScale::FULL,
    };
    let draw = draw_generation(3, 1, 2, schedule).expect("truncated draw");
    assert!(draw.scale_bp > 0);
    assert_eq!(draw.applied_games, 1);
    let directory = test_directory("zero-window-rules");
    let mut cache = GenerationCache::new(directory.to_path_buf(), 3, 2, schedule, 0);
    for (update, applied) in [(2, true), (3, false)] {
        let spec = cache.spec_for_update(update).expect("spec");
        assert_eq!(!spawn_modifiers_for(spec).is_empty(), applied);
    }
}

#[test]
fn resume_rejects_swapped_opponent_weights() {
    let weights = test_directory("swap-weights");
    let directory = test_directory("swap-run");
    let first = PolicyModel::fresh(0x1111).expect("first opponent");
    TrainingArtifact::save_runtime_weights(&first, &weights).expect("first weights");
    let mut config = settings(0x1a2c, 1);
    config.opponents = vec![(AnnealedOpponent::Weights(weights.to_path_buf()), one())];
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
}

#[test]
fn initial_weights_then_resume_matches_uninterrupted_parameters_optimizer_and_rng() {
    let weights = test_directory("initial-weights");
    let uninterrupted = test_directory("initial-uninterrupted");
    let resumed = test_directory("initial-resumed");
    let initial = PolicyModel::fresh(23_074).expect("initial model");
    TrainingArtifact::save_runtime_weights(&initial, &weights).expect("initial weights");
    let fingerprint = initial.parameter_fingerprint().expect("fingerprint");
    let mut config = settings(23_071, 2);
    // A styled opponent draws its style and noise from the game seed, so the split game
    // must replay it exactly too.
    config.opponents = vec![(AnnealedOpponent::Styled(crate::ScriptKind::Teacher), one())];
    let start = |directory: &std::path::Path, stop_after| {
        run_annealed_job_harnessed(
            config.clone(),
            AnnealedHarness {
                stop_after,
                ..harness()
            },
            PolicyDevice::Cpu,
            directory,
            false,
            Some(&weights),
            |_| {},
        )
        .expect("fresh run from initial weights")
    };
    let reference = start(&uninterrupted, None);
    assert_eq!(reference.starting_policy_fingerprint, fingerprint);
    assert_eq!(reference.completed_updates, 2);
    let first = start(&resumed, Some(1));
    assert_eq!(first.starting_policy_fingerprint, fingerprint);
    assert_eq!(first.completed_updates, 1);
    assert!(
        in_flight_decisions(&resumed) > 0,
        "the boundary splits a game"
    );
    let before = checkpoint_digests(&resumed);
    let error = run_annealed_job_harnessed(
        config.clone(),
        harness(),
        PolicyDevice::Cpu,
        &resumed,
        true,
        Some(&weights),
        |_| {},
    )
    .expect_err("resume cannot reload initial weights");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: resume initial weights"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    let second = run(config, &resumed, true).expect("resume");
    assert_eq!(second.completed_updates, 2);
    assert_trajectory_equal(&uninterrupted, &resumed);
    assert_artifact_bits(&uninterrupted, &resumed, PolicyDevice::Cpu);
    // `play` and frozen opponents share this strict loader.
    let exported = PolicyModel::fresh(0).expect("play model");
    TrainingArtifact::load_runtime_weights(&exported, &resumed).expect("exported runtime weights");
    let (checkpoint, _, _) = restored_state(
        &TrainingArtifact::load(&resumed).expect("checkpoint"),
        PolicyDevice::Cpu,
    );
    assert_eq!(
        exported.export_parameters().expect("exported parameters"),
        checkpoint.parameters
    );
}

/// Logged decisions of the games in flight at the committed boundary.
fn in_flight_decisions(directory: &std::path::Path) -> usize {
    let artifact = TrainingArtifact::load(directory).expect("checkpoint");
    crate::ppo_arena::collector_state::CollectorState::decode(&artifact.collection().state)
        .expect("collection state")
        .slots
        .iter()
        .map(|slot| slot.log.policy.len())
        .sum()
}

#[test]
fn simulation_thread_count_never_changes_training_bits() {
    let config = settings(23_072, 2);
    let directories: Vec<_> = [1, 4]
        .into_iter()
        .map(|threads| {
            let directory = test_directory("simulation-threads");
            let mut config = config.clone();
            config.simulation_threads = threads;
            run(config, &directory, false).expect("two updates");
            directory
        })
        .collect();
    assert_trajectory_equal(&directories[0], &directories[1]);
}

/// Runs under both environment schedules: the default adaptive one once failed
/// its first checkpoint save with a league opponent.
#[test]
fn pfsp_league_mixtures_and_learned_potential_resume_in_flight_games_exactly() {
    let adaptive =
        crate::EnvironmentSchedule::Adaptive(crate::AdaptiveEnvironmentConfig::default());
    for (name, schedule) in [
        ("fixed", crate::EnvironmentSchedule::Fixed),
        ("adaptive", adaptive),
    ] {
        league_resume_scenario(name, schedule);
    }
}

fn league_resume_scenario(name: &str, schedule: crate::EnvironmentSchedule) {
    let weights = test_directory(&format!("mixed-opponent-weights-{name}"));
    let frozen = PolicyModel::fresh(0x0dd).expect("frozen opponent");
    TrainingArtifact::save_runtime_weights(&frozen, &weights).expect("frozen weights");
    let mut config = settings(23_074, 6);
    config.environment_schedule = schedule;
    config.slots = 4;
    config.ppo.samples_per_update = 40;
    config.league_size = 4;
    config.league_every = 2;
    config.potential = Some(crate::WinModelConfig {
        every: 1,
        games: 64,
        min_games: 1,
    });
    config.opponents = vec![
        (AnnealedOpponent::Teacher, one()),
        (AnnealedOpponent::HarassPush, one()),
        (AnnealedOpponent::SelfPlay, one()),
        (AnnealedOpponent::Weights(weights.to_path_buf()), one()),
        (AnnealedOpponent::League, one()),
    ];
    let uninterrupted = test_directory(&format!("mixed-uninterrupted-{name}"));
    let resumed = test_directory(&format!("mixed-resumed-{name}"));
    run(config.clone(), &uninterrupted, false).expect("uninterrupted mixture");
    let stopped = AnnealedHarness {
        stop_after: Some(4),
        ..harness()
    };
    run_with(config.clone(), stopped, &resumed, false).expect("first updates");
    let artifact = TrainingArtifact::load(&resumed).expect("checkpoint");
    let state =
        crate::ppo_arena::collector_state::CollectorState::decode(&artifact.collection().state)
            .expect("collection state");
    assert!(
        state.slots.iter().any(|slot| !slot.log.opponent.is_empty()),
        "a neural opponent game is in flight at the boundary"
    );
    let league = |kind: &OpponentKind| matches!(kind, OpponentKind::League(_));
    assert!(state.mixture.entries().iter().any(|(kind, _)| league(kind)));
    assert!(state.outcomes.entries.iter().any(|(kind, _)| league(kind)));
    let reweighted = state
        .mixture
        .entries()
        .windows(2)
        .any(|pair| pair[0].1 != pair[1].1);
    assert!(reweighted, "PFSP reweighted the configured equal weights");
    assert!(
        state.potential > 0 && state.win.models.len() > 1,
        "new games at the boundary use a learned potential; the previous one stays for their predecessors"
    );
    let written: Vec<u64> = state.league.iter().map(|(update, _)| *update).collect();
    assert_eq!(
        written,
        [0, 2],
        "the checkpoint's own update 4 is its model"
    );
    run(config, &resumed, true).expect("resume");
    assert_trajectory_equal(&uninterrupted, &resumed);
    assert_league_directory_matches_checkpoint(&resumed);
}

/// Only the snapshots the final checkpoint records stay on disk.
fn assert_league_directory_matches_checkpoint(resumed: &std::path::Path) {
    let artifact = TrainingArtifact::load(resumed).expect("final checkpoint");
    let recorded =
        crate::ppo_arena::collector_state::CollectorState::decode(&artifact.collection().state)
            .expect("final collection state")
            .league;
    let mut kept: Vec<String> = std::fs::read_dir(resumed.join("league"))
        .expect("league")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("name")
        })
        .collect();
    kept.sort();
    let expected: Vec<String> = recorded
        .iter()
        .map(|(update, _)| format!("u{update:04}"))
        .collect();
    assert_eq!(
        kept, expected,
        "only the snapshots the final checkpoint records"
    );
}

/// HarassPush labels the learner after one critic-only update: the warm-up keeps
/// every policy parameter's bits, the labels reach the objective, and a resume
/// replays the labeled in-flight games under the recorded schedule exactly.
#[test]
fn guided_updates_freeze_the_policy_in_warmup_and_resume_exactly() {
    let weights = test_directory("guided-initial-weights");
    let initial = PolicyModel::fresh(23_075).expect("initial model");
    TrainingArtifact::save_runtime_weights(&initial, &weights).expect("initial weights");
    let mut config = settings(23_076, 3);
    config.guidance = crate::TrainingGuidance {
        imitation: Some(crate::ImitationSchedule {
            shadow: crate::ScriptKind::HarassPush,
            start: one(),
            end: crate::EnvironmentDecimal::from_units(500_000),
            updates: 2,
        }),
        critic_warmup_updates: 1,
    };
    let start = |config: &AnnealedJobConfig, directory: &std::path::Path, stop_after| {
        run_annealed_job_harnessed(
            config.clone(),
            AnnealedHarness {
                stop_after,
                ..harness()
            },
            PolicyDevice::Cpu,
            directory,
            false,
            Some(&weights),
            |_| {},
        )
        .expect("guided run")
    };
    let uninterrupted = test_directory("guided-uninterrupted");
    let reference = start(&config, &uninterrupted, None);
    assert_eq!(
        reference.latest.objective,
        crate::UpdateObjective {
            imitation: 0.5,
            critic_only: false
        }
    );
    assert!(reference.latest.imitation.labeled > 0.0);
    let resumed = test_directory("guided-resumed");
    start(&config, &resumed, Some(1));
    let (warm, _, _) = restored_state(
        &TrainingArtifact::load(&resumed).expect("warm-up checkpoint"),
        PolicyDevice::Cpu,
    );
    assert_only_the_critic_changed(&initial, &warm.parameters);
    assert!(
        in_flight_decisions(&resumed) > 0,
        "a labeled game is in flight"
    );
    let before = checkpoint_digests(&resumed);
    let mut changed = config.clone();
    changed.guidance.critic_warmup_updates = 2;
    assert!(matches!(
        run(changed, &resumed, true),
        Err(PpoError::ScopeMismatch(_))
    ));
    assert_eq!(checkpoint_digests(&resumed), before);
    let second = run(config, &resumed, true).expect("resume");
    assert_eq!(second.latest, reference.latest);
    assert_trajectory_equal(&uninterrupted, &resumed);
    assert_artifact_bits(&uninterrupted, &resumed, PolicyDevice::Cpu);
}

/// Every policy parameter keeps its bits through a critic-only warm-up.
fn assert_only_the_critic_changed(initial: &PolicyModel, warm: &[f32]) {
    let before = initial.export_parameters().expect("initial parameters");
    let mut offset = 0;
    for (name, shape) in initial.parameter_schema().expect("schema") {
        let range = offset..offset + shape.iter().product::<usize>();
        let kept = before[range.clone()]
            .iter()
            .zip(&warm[range.clone()])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert_eq!(kept, !name.starts_with("value."), "{name}");
        offset = range.end;
    }
}

/// Two lanes of one slot each; with 16-decision games an update of four
/// intervals per lane ends mid-game, so every resume replays in-flight games.
fn settings(seed: u64, updates: u64) -> AnnealedJobConfig {
    AnnealedJobConfig {
        environment_schedule: crate::EnvironmentSchedule::Fixed,
        execution: crate::TrainingExecutionOptions::default(),
        updates,
        invocation_updates: None,
        history: None,
        slots: 2,
        lanes: 2,
        simulation_threads: 2,
        simulation_groups: 1,
        pin_threads: false,
        generation_updates: 1,
        zero_updates: 0,
        scale: crate::randomization::AnnealScale::FULL,
        seed,
        opponents: vec![(AnnealedOpponent::Teacher, one())],
        opponent_schedule: crate::OpponentSchedule::Pfsp,
        league_size: 4,
        league_every: 20,
        potential: None,
        ppo: PpoConfig {
            decision_interval_ticks: MAP2_DECISION_INTERVAL_TICKS,
            samples_per_update: 6,
            epochs: 1,
            minibatch: 3,
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
        guidance: crate::TrainingGuidance::default(),
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        git_commit: "test-drysua-annealed".to_owned(),
        simulator_commit: "test-bota-annealed".to_owned(),
    }
}

fn one() -> crate::EnvironmentDecimal {
    crate::EnvironmentDecimal::from_units(crate::EnvironmentDecimal::SCALE)
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
    // The one tensor generation the manifest references survives pruning.
    let generation = std::fs::read_dir(directory)
        .expect("checkpoint directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("name")
        })
        .filter(|name| name.starts_with("checkpoint.") && name.ends_with(".safetensors"))
        .collect::<Vec<_>>();
    assert_eq!(generation.len(), 1, "{generation:?}");
    [
        "checkpoint.meta",
        &generation[0],
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

#[test]
fn the_scope_records_a_non_default_environment_scale_in_a_fixed_order() {
    let options = settings(9001, 2);
    let scope = |options: &AnnealedJobConfig| {
        let pool = load_opponents(options).expect("opponents");
        annealed_run(options, PolicyDevice::Cpu, options.ppo, harness(), &pool)
            .expect("scope")
            .command_line
    };
    let plain = scope(&options);
    assert!(!plain.contains("--environment-scale"), "{plain}");
    let ramp = crate::randomization::AnnealScale {
        start_bp: 0,
        end_bp: 20_000,
    };
    let mut scaled = options.clone();
    scaled.scale = ramp;
    assert_eq!(
        scope(&scaled),
        format!("{plain} --environment-scale-start 0 --environment-scale-end 2")
    );
    // The adaptive suffix keeps its own order and the scale comes after it.
    let mut adaptive = options.clone();
    adaptive.environment_schedule = crate::EnvironmentSchedule::Adaptive(Default::default());
    adaptive.scale = ramp;
    let text = scope(&adaptive);
    assert!(
        text.ends_with(
            "--environment-extension 0.75 --environment-scale-start 0 --environment-scale-end 2"
        ),
        "{text}"
    );
    // A fixed schedule with equal endpoints keeps one constant scale all run.
    let mut fixed = options.clone();
    fixed.scale = crate::randomization::AnnealScale {
        start_bp: 5_000,
        end_bp: 5_000,
    };
    assert!(scope(&fixed).ends_with("--environment-scale-start 0.5 --environment-scale-end 0.5"));
}

#[test]
fn resume_rejects_a_changed_environment_scale_without_committing() {
    let directory = test_directory("annealed-scale-scope");
    let options = settings(9001, 2);
    let run_result = run(options.clone(), &directory, false).expect("fresh run");
    assert_eq!(run_result.completed_updates, 2);
    let before = checkpoint_digests(&directory);
    let mut changed = options.clone();
    changed.scale = crate::randomization::AnnealScale {
        start_bp: 5_000,
        end_bp: 5_000,
    };
    let error = run(changed, &directory, true).expect_err("changed scale");
    let text = error.to_string();
    assert!(text.contains("--environment-scale-start"), "{text}");
    assert_eq!(checkpoint_digests(&directory), before);
    assert_eq!(
        run(options, &directory, true)
            .expect("recorded scale resumes")
            .completed_updates,
        2
    );
}

fn assert_artifact_bits(source: &std::path::Path, target: &std::path::Path, device: PolicyDevice) {
    let source = TrainingArtifact::load(source).expect("source artifact");
    let target = TrainingArtifact::load(target).expect("target artifact");
    assert_eq!(source.progress(), target.progress());
    let (source, source_rng, source_updates) = restored_state(&source, device);
    let (target, target_rng, target_updates) = restored_state(&target, device);
    let (source_first, source_second) = source.adam.moments().expect("source moments");
    let (target_first, target_second) = target.adam.moments().expect("target moments");
    for (source, target) in [
        (source.parameters.as_slice(), target.parameters.as_slice()),
        (&source_first, &target_first),
        (&source_second, &target_second),
    ] {
        assert_eq!(source.len(), target.len());
        assert!(
            source
                .iter()
                .zip(target)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert_eq!(source.adam.step(), target.adam.step());
    assert_eq!(source_rng, target_rng);
    assert_eq!(source_updates, target_updates);
}

fn restored_state(
    artifact: &TrainingArtifact,
    device: PolicyDevice,
) -> (crate::model::ModelAdamSnapshot, (u64, u64), u64) {
    let model = PolicyModel::fresh_on(9001, device).expect("restore device");
    let restored = artifact.restore(&model, artifact.run()).expect("restore");
    (
        restored
            .trainer()
            .checkpoint_snapshot(&model)
            .expect("snapshot"),
        restored.trainer().rng_checkpoint(),
        restored.trainer().updates(),
    )
}
