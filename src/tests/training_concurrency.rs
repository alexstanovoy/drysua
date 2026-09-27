use super::*;
use crate::TrainingExecutionOptions;

fn folding_execution() -> TrainingExecutionOptions {
    TrainingExecutionOptions {
        host_math_workers: 4,
    }
}

#[test]
fn concurrency_cli_rejects_retired_flags_and_bounds_scope_workers() {
    for arguments in [
        vec!["--actor-overlap", "continue-v1"],
        vec!["--learner-prefetch"],
    ] {
        let error = crate::cli::annealed_settings_for_test(&arguments).expect_err("retired option");
        assert!(
            error
                .to_string()
                .contains(&format!("unexpected argument '{}'", arguments[0]))
        );
    }
    let baseline = settings(9001, 2);
    assert_eq!(baseline.execution.host_math_workers, 1);
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    assert!(!original.contains("--host-math-workers"));
    for workers in [2, 4, 32] {
        let value = workers.to_string();
        let parsed = crate::cli::annealed_settings_for_test(&[
            "--updates",
            "2",
            "--generation-games",
            "2",
            "--games",
            "2",
            "--parallel",
            "2",
            "--host-math-workers",
            &value,
        ])
        .expect("worker option");
        assert_eq!(parsed.execution.host_math_workers, workers);
        let mut candidate = baseline.clone();
        candidate.execution = parsed.execution;
        assert_eq!(
            scope(&candidate),
            format!("{original} --host-math-workers {workers}")
        );
    }
    for workers in [0, 33] {
        let error = TrainingExecutionOptions {
            host_math_workers: workers,
        }
        .validate()
        .expect_err("worker bound");
        assert_eq!(
            error.to_string(),
            "invalid PPO config field: host math workers must be within 1..=32"
        );
    }
}

#[test]
fn concurrency_short_updates_and_resume_match_default_bits() {
    let baseline = test_directory("concurrency-base");
    let candidate = test_directory("concurrency-folding");
    let mut options = settings(9001, 2);
    let expected = run(options.clone(), &baseline, false).expect("baseline");
    options.execution = folding_execution();
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
    let actual = run(options.clone(), &candidate, true).expect("resume C4");
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_artifact_bits(&baseline, &candidate, PolicyDevice::Cpu);
    let before = checkpoint_digests(&candidate);
    options.execution = TrainingExecutionOptions::default();
    assert_eq!(
        run(options, &candidate, true)
            .expect_err("scope mismatch")
            .to_string(),
        "checkpoint scope mismatch: --host-math-workers: recorded 4, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&candidate), before);
    for directory in [baseline, candidate] {
        std::fs::remove_dir_all(directory).expect("remove own checkpoint");
    }
}

#[test]
fn retired_execution_scopes_allow_read_only_loading_but_reject_default_and_c4_resume() {
    let directory = test_directory("retired-execution-scopes");
    let options = settings(9001, 2);
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
    let original = TrainingArtifact::load(&directory).expect("original artifact");
    let model = PolicyModel::fresh(9001).expect("model");
    let restored = original.restore(&model, original.run()).expect("restore");
    let expected_parameters = model.export_parameters().expect("parameters");
    let generations = generation_files(&directory);
    for suffix in [
        " --actor-overlap continue-v1",
        " --learner-prefetch",
        " --actor-overlap continue-v1 --learner-prefetch",
        " --actor-overlap continue-v1 --learner-prefetch --host-math-workers 4",
    ] {
        let mut scope = original.run().clone();
        scope.command_line.push_str(suffix);
        TrainingArtifact::capture(
            &model,
            restored.trainer(),
            scope.clone(),
            original.progress().clone(),
        )
        .expect("historical scope")
        .save(&directory)
        .expect("save historical artifact");
        let before = checkpoint_digests(&directory);
        let loaded = TrainingArtifact::load(&directory).expect("read-only historical artifact");
        assert_eq!(loaded.run(), &scope);
        assert_eq!(loaded.progress(), original.progress());
        let weights = PolicyModel::fresh(17).expect("read-only model");
        TrainingArtifact::load_runtime_weights(&weights, &directory).expect("read-only weights");
        assert_eq!(
            weights.export_parameters().expect("weights"),
            expected_parameters
        );
        assert_eq!(checkpoint_digests(&directory), before);
        for execution in [TrainingExecutionOptions::default(), folding_execution()] {
            let mut requested = options.clone();
            requested.execution = execution;
            let difference = if execution.host_math_workers == 4 {
                "--host-math-workers: recorded <absent>, requested 4"
            } else if suffix.contains("--actor-overlap") {
                "--actor-overlap: recorded continue-v1, requested <absent>"
            } else {
                "--learner-prefetch: recorded <present>, requested <absent>"
            };
            assert_eq!(
                run(requested, &directory, true)
                    .expect_err("retired resume")
                    .to_string(),
                format!("checkpoint scope mismatch: {difference}")
            );
            assert_eq!(checkpoint_digests(&directory), before);
            assert_eq!(generation_files(&directory), generations);
        }
    }
    std::fs::remove_dir_all(directory).expect("remove own historical artifacts");
}

#[test]
#[ignore = "bounded concurrency probe; run only through the authorized background runner"]
fn concurrency_probe_cpu() {
    concurrency_probe(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "bounded CUDA concurrency probe; run only through the authorized background runner"]
fn concurrency_probe_cuda() {
    concurrency_probe(PolicyDevice::Cuda { ordinal: 0 });
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

fn parse_probe_balanced(value: Option<&std::ffi::OsStr>) -> Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("DRYSUA_PROBE_BALANCED must be 0 or 1"),
    }
}

#[test]
fn probe_controls_reject_out_of_bounds_work_without_changing_defaults() {
    use std::ffi::OsStr;
    for (default, maximum) in [
        (40, ANNEALED_EPISODE_DECISIONS),
        (1, 16),
        (128, crate::MODEL_MAX_BATCH),
    ] {
        assert_eq!(probe_count(None, default, maximum), Ok(default));
        for value in [1, maximum] {
            assert_eq!(
                probe_count(Some(OsStr::new(&value.to_string())), default, maximum),
                Ok(value)
            );
        }
        for value in [
            "".to_owned(),
            "0".to_owned(),
            "-1".to_owned(),
            (maximum + 1).to_string(),
            "18446744073709551616".to_owned(),
        ] {
            assert_eq!(
                probe_count(Some(OsStr::new(&value)), default, maximum),
                Err("probe workload value is outside its positive bound")
            );
        }
    }
    assert_eq!(parse_probe_balanced(None), Ok(false));
    for (value, expected) in [("0", false), ("1", true)] {
        assert_eq!(parse_probe_balanced(Some(OsStr::new(value))), Ok(expected));
    }
    for value in ["", "true", "2"] {
        assert_eq!(
            parse_probe_balanced(Some(OsStr::new(value))),
            Err("DRYSUA_PROBE_BALANCED must be 0 or 1")
        );
    }
}

fn concurrency_probe(device: PolicyDevice) {
    let count = |name, default, maximum| {
        probe_count(std::env::var_os(name).as_deref(), default, maximum).expect(name)
    };
    let rounds = count("DRYSUA_PROBE_ROUNDS", 40, ANNEALED_EPISODE_DECISIONS);
    let mode = std::env::var("DRYSUA_PROBE_MODE").unwrap_or_else(|_| "c".to_owned());
    let execution = match mode.as_str() {
        "base" => TrainingExecutionOptions::default(),
        "c" => TrainingExecutionOptions {
            host_math_workers: count("DRYSUA_PROBE_WORKERS", 4, 32),
        },
        _ => panic!("DRYSUA_PROBE_MODE must be base/c"),
    };
    let balanced = parse_probe_balanced(std::env::var_os("DRYSUA_PROBE_BALANCED").as_deref())
        .expect("balanced flag");
    // The caller pins U376 read-only weights here; every fresh trial uses the same path.
    let initial = std::env::var_os("DRYSUA_PROBE_WEIGHTS").map(PathBuf::from);
    let mut options = settings(9001, 1);
    options.games_per_update = 40;
    options.parallel_worlds = 40;
    options.games_per_generation = 40;
    options.ppo.environments = 40;
    options.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    options.ppo.minibatch = count("DRYSUA_PROBE_MINIBATCH", 128, crate::MODEL_MAX_BATCH);
    options.ppo.epochs = count("DRYSUA_PROBE_EPOCHS", 1, 16);
    let order: &[bool] = if balanced {
        &[false, false, true, true, false]
    } else {
        &[false, true]
    };
    eprintln!(
        "concurrency-workload mode={mode} device={device:?} worlds=40 rounds={rounds} epochs={} effective_minibatch={} microbatch=64 seed=9001 balanced={balanced}",
        options.ppo.epochs, options.ppo.minibatch
    );
    let mut trials = Vec::with_capacity(order.len());
    for (index, &candidate) in order.iter().enumerate() {
        options.execution = if candidate {
            execution
        } else {
            TrainingExecutionOptions::default()
        };
        let directory = test_directory("concurrency-probe");
        let measured = !balanced || index != 0;
        eprintln!(
            "concurrency-start index={index} measured={measured} execution={:?}",
            options.execution
        );
        let started = Instant::now();
        let report = run_annealed_job_harnessed(
            options.clone(),
            AnnealedHarness {
                episode_decisions: Some(rounds),
                ..harness()
            },
            device,
            &directory,
            false,
            initial.as_deref(),
            |_| {},
        )
        .expect("fresh probe trial");
        let elapsed = started.elapsed().as_nanos();
        eprintln!(
            "concurrency-end index={index} measured={measured} elapsed_ns={elapsed} initial_fingerprint={:016x} games={} samples={} steps={} ticks={} workers_requested={} workers_resolved={} state_hash={:016x}",
            report.starting_policy_fingerprint,
            report.games,
            report.rollout_samples,
            report.optimizer_step,
            report.elapsed_ticks,
            options.execution.host_math_workers,
            crate::model::resolved_host_math_workers(options.execution.host_math_workers),
            artifact_hash(&directory, device)
        );
        assert_eq!(report.games, 40);
        trials.push((directory, report, elapsed));
    }
    for (directory, report, _) in &trials[1..] {
        assert_eq!(&trials[0].1, report);
        assert_probe_report_bits(&trials[0].1.latest, &report.latest);
        assert_artifact_bits(&trials[0].0, directory, device);
    }
    if balanced {
        eprintln!(
            "concurrency-balanced-summary reference_mean_ns={} candidate_mean_ns={}",
            (trials[1].2 + trials[4].2) / 2,
            (trials[2].2 + trials[3].2) / 2
        );
    }
    for (directory, _, _) in trials {
        std::fs::remove_dir_all(directory).expect("remove own probe trial");
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

fn assert_artifact_bits(source: &std::path::Path, target: &std::path::Path, device: PolicyDevice) {
    let source = TrainingArtifact::load(source).expect("source artifact");
    let target = TrainingArtifact::load(target).expect("target artifact");
    assert_eq!(source.progress(), target.progress());
    let (source, source_rng, source_updates) = probe_state(&source, device);
    let (target, target_rng, target_updates) = probe_state(&target, device);
    let (source_first, source_second) = source.adam.moments();
    let (target_first, target_second) = target.adam.moments();
    for (source, target) in [
        (source.parameters.as_slice(), target.parameters.as_slice()),
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
    assert_eq!(source.adam.step(), target.adam.step());
    assert_eq!(source_rng, target_rng);
    assert_eq!(source_updates, target_updates);
}

fn artifact_hash(directory: &std::path::Path, device: PolicyDevice) -> u64 {
    use std::hash::Hasher;
    let artifact = TrainingArtifact::load(directory).expect("probe artifact");
    let (snapshot, shuffle, updates) = probe_state(&artifact, device);
    let (first, second) = snapshot.adam.moments();
    // Preserve the historical state-hash byte stream (U376 CUDA: eb52a187d338a4e8).
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for values in [snapshot.parameters.as_slice(), first, second] {
        for value in values {
            hash.write_u32(value.to_bits());
        }
    }
    hash.write_u64(snapshot.adam.step());
    hash.write_u64(updates);
    hash.write_u64(shuffle.0);
    hash.write_u64(shuffle.1);
    hash.finish()
}

fn probe_state(
    artifact: &TrainingArtifact,
    device: PolicyDevice,
) -> (crate::model::ModelAdamSnapshot, (u64, u64), u64) {
    let model = PolicyModel::fresh_on(9001, device).expect("probe restore device");
    let restored = artifact
        .restore(&model, artifact.run())
        .expect("probe restore");
    (
        restored
            .trainer()
            .checkpoint_snapshot(&model)
            .expect("snapshot"),
        restored.trainer().rng_checkpoint(),
        restored.trainer().updates(),
    )
}
