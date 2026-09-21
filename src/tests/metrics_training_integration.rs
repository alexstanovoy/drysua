use std::fs;
use std::os::unix::fs::DirBuilderExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

const CHILD_ENV: &str = "DRYSUA_METRICS_TRAINING_CHILD";

#[test]
fn metrics_enabled_trainers_resume_with_identical_checkpoints() {
    let (_, module) = module_path!().split_once("::").expect("test module path");
    let worker = format!("{module}::metrics_training_worker");
    let status = Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", &worker, "--nocapture"])
        .env(CHILD_ENV, module_path!())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .expect("launch isolated metrics training worker");
    assert!(status.success(), "metrics training worker failed: {status}");
}

#[test]
fn metrics_training_worker() {
    if std::env::var_os(CHILD_ENV) != Some(module_path!().into()) {
        return;
    }
    for annealed in [false, true] {
        assert!(!prometheus::enabled());
        assert_resume_preserves_training(annealed);
    }
}

fn assert_resume_preserves_training(annealed: bool) {
    let [baseline, resumed, metrics] = std::array::from_fn(|_| Fixture::new());
    let expected = run_job(annealed, &baseline.0, 2, false);
    assert_eq!(expected.rollout_samples, if annealed { 4 } else { 8 });
    assert_eq!(expected.games, [0; 4]);
    let options = prometheus::MetricsOptions {
        metrics_directory: Some(metrics.0.clone()),
        metrics_listen: None,
    };
    let mut games = [0; 4];
    for target in [1, 2] {
        let guard = options.start().expect("start invocation metrics");
        assert!(prometheus::enabled());
        let report = run_job(annealed, &resumed.0, target, target == 2);
        assert_eq!(report.completed_updates, target);
        for (total, session) in games.iter_mut().zip(report.games) {
            *total += session;
        }
        let updates_target = if annealed { 2 } else { target };
        assert_exposition(&metrics.0, &report, games, updates_target, true);
        let published = fs::read(metrics.0.join("metrics.state")).expect("published before finish");
        guard.finish().expect("finish invocation metrics");
        assert!(!prometheus::enabled());
        assert_exposition(&metrics.0, &report, games, updates_target, false);
        assert_eq!(
            fs::read(metrics.0.join("metrics.state")).expect("published after finish"),
            published
        );
        if target == 2 {
            assert_eq!(Counters { games, ..report }, expected);
            // The manifest binds model/Adam tensor SHA and the checkpointed RNG states.
            assert_eq!(
                fs::read(baseline.0.join("checkpoint.meta")).expect("baseline manifest"),
                fs::read(resumed.0.join("checkpoint.meta")).expect("resumed manifest"),
                "metrics must not change the checkpoint: annealed={annealed}"
            );
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Counters {
    completed_updates: u64,
    optimizer_step: u64,
    rollout_samples: u64,
    games: [u64; 4],
}

fn run_job(annealed: bool, directory: &Path, target: u64, resume: bool) -> Counters {
    assert!((1..=2).contains(&target));
    assert!(!resume || target == 2);
    let settings = full_settings(target);
    if annealed {
        let mut config = annealed_settings(settings);
        config.invocation_updates = std::num::NonZeroU64::new(if resume { 1 } else { target });
        let harness = AnnealedHarness {
            // Eight-step retention with phase 0..7 needs fifteen rounds to flush every game.
            episode_decisions: Some(15),
            stop_after: None,
            stop_after_games: None,
        };
        let report = run_annealed_job_harnessed(
            config,
            harness,
            PolicyDevice::Cpu,
            directory,
            resume,
            None,
            |_| {},
        )
        .expect("tiny annealed training invocation");
        assert_eq!(report.games, target * 2);
        Counters {
            completed_updates: report.completed_updates,
            optimizer_step: report.optimizer_step,
            rollout_samples: report.rollout_samples,
            games: [
                report.terminal_wins,
                report.terminal_losses,
                report.terminal_draws,
                report.episode_timeouts,
            ],
        }
    } else {
        let report = crate::run_training_job_on_with_initial_weights(
            settings,
            PolicyDevice::Cpu,
            directory,
            resume,
            None,
            |_| {},
        )
        .expect("tiny reset-window training invocation");
        Counters {
            completed_updates: report.completed_updates,
            optimizer_step: report.optimizer_step,
            rollout_samples: report.rollout_samples,
            games: [
                report.terminal_wins,
                report.terminal_losses,
                report.terminal_draws,
                report.episode_timeouts,
            ],
        }
    }
}

fn full_settings(updates: u64) -> crate::TrainingJobConfig {
    crate::TrainingJobConfig {
        mastery_config: None,
        opponent_schedule: crate::TrainingOpponentSchedule::Teacher,
        episode_time_cost: 0.0,
        terminal_only: false,
        complete_episodes: false,
        pipeline_groups: 1,
        updates,
        ppo: PpoConfig {
            decision_interval_ticks: MAP2_DECISION_INTERVAL_TICKS,
            environments: 2,
            rollout_decisions: 2,
            epochs: 1,
            minibatch: 2,
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed: 0x1234,
        map: MapId(2),
        git_commit: "test-drysua-metrics".to_owned(),
        simulator_commit: "test-bota-metrics".to_owned(),
    }
}

fn annealed_settings(settings: crate::TrainingJobConfig) -> AnnealedJobConfig {
    AnnealedJobConfig {
        updates: 2,
        invocation_updates: None,
        games_per_update: 2,
        parallel_worlds: 2,
        games_per_generation: 2,
        zero_updates: 0,
        seed: settings.seed,
        opponent: AnnealedOpponent::Teacher,
        ppo: PpoConfig {
            rollout_decisions: MAP2_RETAINED_DECISIONS,
            ..settings.ppo
        },
        // Neither invocation boundary is cadence-due; publication must be forced.
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(3),
        git_commit: settings.git_commit,
        simulator_commit: settings.simulator_commit,
    }
}

fn assert_exposition(
    directory: &Path,
    report: &Counters,
    games: [u64; 4],
    updates_target: u64,
    active: bool,
) {
    let text = prometheus::render_directory_for_test(directory).expect("persisted exposition");
    assert!(text.len() <= 48 * 1024);
    let metric = |name: &str, value| {
        let expected = format!("drysua_training_{name} {value}");
        assert!(text.lines().any(|line| line == expected), "{expected}");
    };
    for (name, value) in [
        ("active", u64::from(active)),
        ("metrics_available", 1),
        ("metrics_state_healthy", 1),
        ("updates_completed", report.completed_updates),
        ("updates_target", updates_target),
        ("samples_total", report.rollout_samples),
        ("optimizer_steps_total", report.optimizer_step),
        ("metrics_start_update", 0),
        ("update_duration_seconds_count", report.completed_updates),
    ] {
        metric(name, value);
    }
    for (outcome, total) in ["win", "loss", "draw", "time_cap"].into_iter().zip(games) {
        metric(&format!("games_total{{outcome=\"{outcome}\"}}"), total);
    }
    assert!(
        directory
            .join("metrics.state")
            .try_exists()
            .expect("published state")
    );
    assert!(
        !directory
            .join("metrics.pending")
            .try_exists()
            .expect("no pending publication")
    );
}

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let process = std::process::id();
        let directory =
            std::env::temp_dir().join(format!("drysua-metrics-training-{process}-{sequence}"));
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700).create(&directory).unwrap();
        Self(directory)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove metrics training fixture");
    }
}
