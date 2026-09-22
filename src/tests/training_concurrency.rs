use super::*;
use crate::{ActorOverlap, TrainingExecutionOptions};

fn all_execution() -> TrainingExecutionOptions {
    TrainingExecutionOptions {
        actor_overlap: ActorOverlap::ContinueV1,
        learner_prefetch: true,
        host_math_workers: 2,
    }
}

#[test]
fn execution_options_are_default_off_bounded_and_scope_bound() {
    let baseline = settings(9001, 2);
    assert_eq!(baseline.execution, TrainingExecutionOptions::default());
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    for execution in [
        TrainingExecutionOptions {
            actor_overlap: ActorOverlap::ContinueV1,
            ..Default::default()
        },
        TrainingExecutionOptions {
            learner_prefetch: true,
            ..Default::default()
        },
        TrainingExecutionOptions {
            host_math_workers: 2,
            ..Default::default()
        },
        all_execution(),
    ] {
        let mut candidate = baseline.clone();
        candidate.execution = execution;
        assert_ne!(original, scope(&candidate));
    }
    for workers in [0, 33] {
        let error = TrainingExecutionOptions {
            host_math_workers: workers,
            ..Default::default()
        }
        .validate()
        .expect_err("worker bound");
        assert_eq!(
            error.to_string(),
            "invalid PPO config field: host math workers must be within 1..=32"
        );
    }
    let mut weights = baseline;
    weights.execution = all_execution();
    weights.opponent = AnnealedOpponent::Weights(PathBuf::from("unused-weights"));
    assert_eq!(
        validate_annealed(&weights, harness())
            .expect_err("CPU-only opponent")
            .to_string(),
        "invalid PPO config field: Continue overlap requires a scripted opponent"
    );
}

#[test]
fn concurrency_cli_parses_flags_and_storage_bound_is_opt_in_only() {
    let options = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "2",
        "--generation-games",
        "2",
        "--games",
        "2",
        "--parallel",
        "2",
        "--actor-overlap",
        "continue-v1",
        "--learner-prefetch",
        "--host-math-workers",
        "32",
    ])
    .expect("experimental options");
    assert_eq!(options.execution.actor_overlap, ActorOverlap::ContinueV1);
    assert!(options.execution.learner_prefetch);
    assert_eq!(options.execution.host_math_workers, 32);
    const {
        assert!(crate::PPO_PREFETCH_STORAGE_PEAK_BYTES > crate::PPO_ANNEALED_STORAGE_PEAK_BYTES);
        assert!(crate::PPO_PREFETCH_STORAGE_PEAK_BYTES < 8 * 1024 * 1024 * 1024);
    }
    eprintln!(
        "prefetch-cap minibatch_max={} sample_bytes={} storage_peak_bytes={}",
        crate::MODEL_MAX_BATCH,
        std::mem::size_of::<crate::PpoPreparedSample>(),
        crate::PPO_PREFETCH_STORAGE_PEAK_BYTES
    );
}

#[test]
fn concurrency_short_updates_and_resume_match_default_bits() {
    let baseline = test_directory("concurrency-base");
    let candidate = test_directory("concurrency-all");
    let mut options = settings(9001, 2);
    let expected = run(options.clone(), &baseline, false).expect("baseline");
    options.execution = all_execution();
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &candidate,
        false,
    )
    .expect("first update");
    let actual = run(options.clone(), &candidate, true).expect("resume with execution options");
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_artifact_bits(&baseline, &candidate, PolicyDevice::Cpu);
    options.execution = TrainingExecutionOptions::default();
    assert!(
        run(options, &candidate, true)
            .expect_err("scope mismatch")
            .to_string()
            .starts_with("checkpoint scope mismatch:")
    );
    std::fs::remove_dir_all(baseline).expect("remove own baseline");
    std::fs::remove_dir_all(candidate).expect("remove own candidate");
}

#[test]
fn late_actor_failure_preserves_previous_committed_checkpoint() {
    let directory = test_directory("concurrency-late-error");
    let mut options = settings(9001, 2);
    options.execution = all_execution();
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &directory,
        false,
    )
    .expect("committed first update");
    let before = std::fs::read(directory.join("checkpoint.meta")).expect("manifest");
    let error = run_with(
        options,
        AnnealedHarness {
            fail_actor_after_dispatch: true,
            ..harness()
        },
        &directory,
        true,
    )
    .expect_err("late sampler failure");
    assert!(
        error
            .to_string()
            .contains("injected Continue overlap late decoder failure")
    );
    assert_eq!(
        std::fs::read(directory.join("checkpoint.meta")).expect("manifest"),
        before
    );
    std::fs::remove_dir_all(directory).expect("remove own checkpoint");
}

fn assert_artifact_bits(source: &std::path::Path, target: &std::path::Path, device: PolicyDevice) {
    let source = TrainingArtifact::load(source).expect("source artifact");
    let target = TrainingArtifact::load(target).expect("target artifact");
    assert_eq!(source.progress(), target.progress());
    let source = probe_state(&source, device);
    let target = probe_state(&target, device);
    let (source_first, source_second) = source.snapshot.adam.moments();
    let (target_first, target_second) = target.snapshot.adam.moments();
    for (source, target) in [
        (
            source.snapshot.parameters.as_slice(),
            target.snapshot.parameters.as_slice(),
        ),
        (source_first, target_first),
        (source_second, target_second),
    ] {
        assert_eq!(source.len(), target.len());
        assert!(
            source
                .iter()
                .zip(target)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert_eq!(source.snapshot.adam.step(), target.snapshot.adam.step());
    assert_eq!(source.shuffle, target.shuffle);
    assert_eq!(source.updates, target.updates);
}

#[test]
#[ignore = "bounded concurrency probe; run only through the authorized background runner"]
fn concurrency_probe_cpu() {
    concurrency_probe(PolicyDevice::Cpu);
}

#[test]
fn balanced_probe_flag_and_order_are_explicit_and_bounded() {
    use std::ffi::OsStr;
    assert!(!parse_probe_balanced(None).expect("default pair"));
    assert!(!parse_probe_balanced(Some(OsStr::new("0"))).expect("explicit pair"));
    assert!(parse_probe_balanced(Some(OsStr::new("1"))).expect("balanced"));
    for value in ["", "true", "2"] {
        assert_eq!(
            parse_probe_balanced(Some(OsStr::new(value))).expect_err("invalid flag"),
            "DRYSUA_PROBE_BALANCED must be 0 or 1"
        );
    }
    assert_eq!(balanced_probe_order(), [false, true, true, false]);
    assert_eq!(
        balanced_probe_order()
            .iter()
            .filter(|candidate| **candidate)
            .count(),
        2
    );
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "bounded CUDA concurrency probe; run only through the authorized background runner"]
fn concurrency_probe_cuda() {
    concurrency_probe(PolicyDevice::Cuda { ordinal: 0 });
}

fn probe_execution() -> (String, TrainingExecutionOptions) {
    let mode = std::env::var("DRYSUA_PROBE_MODE").unwrap_or_else(|_| "all".to_owned());
    let workers =
        std::env::var("DRYSUA_PROBE_WORKERS").map_or(2, |value| value.parse().expect("workers"));
    let execution = match mode.as_str() {
        "base" => TrainingExecutionOptions::default(),
        "a" => TrainingExecutionOptions {
            actor_overlap: ActorOverlap::ContinueV1,
            ..Default::default()
        },
        "b" => TrainingExecutionOptions {
            learner_prefetch: true,
            ..Default::default()
        },
        "c" => TrainingExecutionOptions {
            host_math_workers: workers,
            ..Default::default()
        },
        "all" => TrainingExecutionOptions {
            host_math_workers: workers,
            ..all_execution()
        },
        _ => panic!("DRYSUA_PROBE_MODE must be base/a/b/c/all"),
    }
    .validate()
    .expect("execution options");
    (mode, execution)
}

fn concurrency_probe(device: PolicyDevice) {
    let value = std::env::var_os("DRYSUA_PROBE_BALANCED");
    if parse_probe_balanced(value.as_deref()).expect("balanced probe flag") {
        concurrency_probe_balanced(device);
    } else {
        concurrency_probe_pair(device);
    }
}

fn parse_probe_balanced(value: Option<&std::ffi::OsStr>) -> Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("DRYSUA_PROBE_BALANCED must be 0 or 1"),
    }
}

fn balanced_probe_order() -> [bool; 4] {
    [false, true, true, false]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeWorkload {
    rounds: usize,
    epochs: usize,
    minibatch: usize,
}

fn probe_count(
    value: Option<&std::ffi::OsStr>,
    default: usize,
    maximum: usize,
) -> Result<usize, &'static str> {
    let Some(value) = value else {
        return Ok(default);
    };
    value
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=maximum).contains(value))
        .ok_or("probe workload value is outside its positive bound")
}

fn probe_workload() -> ProbeWorkload {
    let count = |name, default, maximum| {
        probe_count(std::env::var_os(name).as_deref(), default, maximum).expect(name)
    };
    ProbeWorkload {
        rounds: count("DRYSUA_PROBE_ROUNDS", 40, ANNEALED_EPISODE_DECISIONS),
        epochs: count("DRYSUA_PROBE_EPOCHS", 1, 16),
        minibatch: count("DRYSUA_PROBE_MINIBATCH", 128, crate::MODEL_MAX_BATCH),
    }
}

#[test]
fn probe_workload_bounds_are_explicit_and_defaults_are_unchanged() {
    use std::ffi::OsStr;
    assert_eq!(probe_count(None, 40, ANNEALED_EPISODE_DECISIONS), Ok(40));
    assert_eq!(probe_count(None, 1, 16), Ok(1));
    assert_eq!(probe_count(None, 128, crate::MODEL_MAX_BATCH), Ok(128));
    for value in ["1", "1024", "9300"] {
        assert_eq!(
            probe_count(Some(OsStr::new(value)), 40, 9300),
            Ok(value.parse().unwrap())
        );
    }
    for value in ["", "0", "-1", "9301", "18446744073709551616"] {
        assert_eq!(
            probe_count(Some(OsStr::new(value)), 40, 9300),
            Err("probe workload value is outside its positive bound")
        );
    }
}

fn probe_settings(workload: ProbeWorkload) -> AnnealedJobConfig {
    let mut options = settings(9001, 1);
    options.games_per_update = 40;
    options.parallel_worlds = 40;
    options.games_per_generation = 40;
    options.ppo.environments = 40;
    options.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    options.ppo.minibatch = workload.minibatch;
    options.ppo.epochs = workload.epochs;
    options
}

fn concurrency_probe_pair(device: PolicyDevice) {
    let (mode, execution) = probe_execution();
    let baseline = test_directory("concurrency-probe-base");
    let candidate = test_directory("concurrency-probe-candidate");
    let initial = std::env::var_os("DRYSUA_PROBE_WEIGHTS").map(PathBuf::from);
    let workload = probe_workload();
    let mut options = probe_settings(workload);
    eprintln!(
        "concurrency-workload worlds=40 rounds={} epochs={} effective_minibatch={} microbatch=64 sample_bytes={} two_buffer_capacity_bytes={} prefetch_storage_peak_bytes={}",
        workload.rounds,
        workload.epochs,
        workload.minibatch,
        std::mem::size_of::<crate::PpoPreparedSample>(),
        2 * workload.minibatch * std::mem::size_of::<crate::PpoPreparedSample>(),
        crate::PPO_PREFETCH_STORAGE_PEAK_BYTES,
    );
    let mut reports = Vec::new();
    for (directory, execution) in [
        (&baseline, TrainingExecutionOptions::default()),
        (&candidate, execution),
    ] {
        options.execution = execution;
        let started = Instant::now();
        let report = run_annealed_job_harnessed(
            options.clone(),
            AnnealedHarness {
                episode_decisions: Some(workload.rounds),
                ..harness()
            },
            device,
            directory,
            false,
            initial.as_deref(),
            |_| {},
        )
        .expect("short probe");
        let elapsed = started.elapsed();
        eprintln!(
            "concurrency-probe mode={mode} execution={execution:?} device={device:?} elapsed_ns={} games={} samples={} steps={} ticks={} workers_requested={} workers_resolved={} state_hash={:016x}",
            elapsed.as_nanos(),
            report.games,
            report.rollout_samples,
            report.optimizer_step,
            report.elapsed_ticks,
            execution.host_math_workers,
            crate::model::resolved_host_math_workers(execution.host_math_workers),
            artifact_hash(directory, device)
        );
        reports.push(report);
    }
    assert_eq!(reports[0].rollout_samples, reports[1].rollout_samples);
    assert_eq!(reports[0].elapsed_ticks, reports[1].elapsed_ticks);
    assert_eq!(reports[0].optimizer_step, reports[1].optimizer_step);
    assert_probe_report_bits(&reports[0].latest, &reports[1].latest);
    assert_artifact_bits(&baseline, &candidate, device);
    eprintln!(
        "concurrency-parity mode={mode} device={device:?} parameters=true moments=true rng=true report_bits=true work=true"
    );
    std::fs::remove_dir_all(baseline).expect("remove own baseline");
    std::fs::remove_dir_all(candidate).expect("remove own candidate");
}

struct BalancedTrial {
    directory: PathBuf,
    report: AnnealedJobReport,
    elapsed_ns: Option<u128>,
}

fn concurrency_probe_balanced(device: PolicyDevice) {
    let (mode, candidate) = probe_execution();
    let initial = std::env::var_os("DRYSUA_PROBE_WEIGHTS").map(PathBuf::from);
    let workload = probe_workload();
    let mut options = probe_settings(workload);
    eprintln!(
        "concurrency-balanced-workload mode={mode} device={device:?} worlds=40 rounds={} effective_minibatch={} microbatch=64 epochs={} seed=9001 order=ABBA warmups=1",
        workload.rounds, workload.minibatch, workload.epochs
    );
    let mut trials = Vec::with_capacity(5);
    trials.push(run_balanced_trial(
        &mode,
        0,
        &options,
        device,
        initial.as_deref(),
        "warmup",
        workload.rounds,
    ));
    for (index, is_candidate) in balanced_probe_order().into_iter().enumerate() {
        options.execution = if is_candidate {
            candidate
        } else {
            TrainingExecutionOptions::default()
        };
        trials.push(run_balanced_trial(
            &mode,
            index + 1,
            &options,
            device,
            initial.as_deref(),
            if is_candidate { "B" } else { "A" },
            workload.rounds,
        ));
    }
    assert_eq!(trials.len(), 5);
    let reference = &trials[1];
    for trial in &trials {
        assert_eq!(reference.report, trial.report);
        assert_probe_report_bits(&reference.report.latest, &trial.report.latest);
        assert_artifact_bits(&reference.directory, &trial.directory, device);
    }
    let times = std::array::from_fn::<_, 4, _>(|index| {
        trials[index + 1].elapsed_ns.expect("measured trial")
    });
    eprintln!(
        "concurrency-balanced-summary mode={mode} device={device:?} wall_ns={times:?} reference_mean_ns={} candidate_mean_ns={} state_hash={:016x} parameters=true moments=true rng=true report_bits=true work=true",
        (times[0] + times[3]) / 2,
        (times[1] + times[2]) / 2,
        artifact_hash(&reference.directory, device)
    );
    for trial in trials {
        std::fs::remove_dir_all(trial.directory).expect("remove own balanced trial");
    }
}

fn run_balanced_trial(
    mode: &str,
    index: usize,
    options: &AnnealedJobConfig,
    device: PolicyDevice,
    initial: Option<&std::path::Path>,
    arm: &str,
    rounds: usize,
) -> BalancedTrial {
    assert!(index <= 4);
    assert_eq!(index == 0, arm == "warmup");
    let directory = test_directory("concurrency-balanced-trial");
    // Existing collection/optimization timing lines fall between these trial markers.
    eprintln!(
        "concurrency-balanced-start mode={mode} index={index} arm={arm} measured={} execution={:?} workers_resolved={}",
        index != 0,
        options.execution,
        crate::model::resolved_host_math_workers(options.execution.host_math_workers)
    );
    let started = (index != 0).then(Instant::now);
    let report = run_annealed_job_harnessed(
        options.clone(),
        AnnealedHarness {
            episode_decisions: Some(rounds),
            ..harness()
        },
        device,
        &directory,
        false,
        initial,
        |_| {},
    )
    .expect("fresh balanced trial");
    let elapsed_ns = started.map(|started| started.elapsed().as_nanos());
    let display_time =
        elapsed_ns.map_or_else(|| "unmeasured".to_owned(), |value| value.to_string());
    eprintln!(
        "concurrency-balanced-end mode={mode} index={index} arm={arm} elapsed_ns={display_time} initial_fingerprint={:016x} samples={} steps={} ticks={}",
        report.starting_policy_fingerprint,
        report.rollout_samples,
        report.optimizer_step,
        report.elapsed_ticks
    );
    BalancedTrial {
        directory,
        report,
        elapsed_ns,
    }
}

fn assert_probe_report_bits(source: &crate::PpoUpdateReport, target: &crate::PpoUpdateReport) {
    let bits = |report: &crate::PpoUpdateReport| {
        [
            report.policy_loss,
            report.value_loss,
            report.entropy,
            report.approximate_kl,
            report.rejected_kl,
            report.clip_fraction,
            report.gradient_norm,
            report.applied_scale,
        ]
        .map(f64::to_bits)
    };
    assert_eq!(bits(source), bits(target));
    assert_eq!(source, target);
}

fn artifact_hash(directory: &std::path::Path, device: PolicyDevice) -> u64 {
    use std::hash::Hasher;
    let artifact = TrainingArtifact::load(directory).expect("probe artifact");
    let state = probe_state(&artifact, device);
    let (first, second) = state.snapshot.adam.moments();
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for values in [state.snapshot.parameters.as_slice(), first, second] {
        for value in values {
            hash.write_u32(value.to_bits());
        }
    }
    hash.write_u64(state.snapshot.adam.step());
    hash.write_u64(state.updates);
    hash.write_u64(state.shuffle.0);
    hash.write_u64(state.shuffle.1);
    hash.finish()
}

struct ProbeState {
    snapshot: crate::model::ModelAdamSnapshot,
    shuffle: (u64, u64),
    updates: u64,
}

fn probe_state(artifact: &TrainingArtifact, device: PolicyDevice) -> ProbeState {
    let model = PolicyModel::fresh_on(9001, device).expect("probe restore device");
    let restored = artifact
        .restore(&model, artifact.run())
        .expect("probe restore");
    ProbeState {
        snapshot: restored
            .trainer()
            .checkpoint_snapshot(&model)
            .expect("probe snapshot"),
        shuffle: restored.trainer().rng_checkpoint(),
        updates: restored.trainer().updates(),
    }
}
