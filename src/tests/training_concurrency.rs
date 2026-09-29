use super::*;
use crate::TrainingExecutionOptions;

#[test]
fn actor_value_reuse_cli_is_annealed_only_with_legacy_off_and_canonical_enabled_scope() {
    let mut arguments = vec![
        "--updates",
        "2",
        "--games",
        "2",
        "--parallel",
        "2",
        "--generation-games",
        "2",
    ];
    let baseline = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments)
        .expect("legacy execution options");
    assert!(!TrainingExecutionOptions::default().reuse_actor_values);
    assert!(!baseline.execution.reuse_actor_values);
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    assert!(!original.contains("--reuse-actor-values"));
    arguments.extend([
        "--reuse-actor-values",
        "--host-math-workers",
        "4",
        "--balanced-minibatches",
    ]);
    let enabled =
        crate::cli::legacy_fixed_annealed_settings_for_test(&arguments).expect("reuse options");
    assert!(enabled.execution.reuse_actor_values);
    assert_eq!(
        scope(&enabled),
        format!("{original} --balanced-minibatches --host-math-workers 4 --reuse-actor-values")
    );
    for operation in ["train", "train-full"] {
        let error = crate::cli::parse_from(["drysua", operation, "--reuse-actor-values"])
            .expect_err("annealed-only option");
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        assert!(
            error
                .to_string()
                .contains("unexpected argument '--reuse-actor-values'")
        );
    }
}

#[test]
fn actor_value_reuse_resumes_exactly_after_scope_rejection_and_late_abort() {
    let uninterrupted = test_directory("actor-values-uninterrupted");
    let resumed = test_directory("actor-values-resumed");
    let mut options = settings(9001, 2);
    options.execution.reuse_actor_values = true;
    options.games_per_update = 4;
    options.ppo.environments = 4;
    assert!(harness().episode_decisions() >= 9);
    let expected = run(options.clone(), &uninterrupted, false).expect("reuse updates");
    let first = run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &resumed,
        false,
    )
    .expect("first reuse update");
    assert_eq!(first.completed_updates, 1);
    assert!(first.optimizer_step > 0);
    assert_eq!(generation_files(&resumed).len(), 2);
    let before = checkpoint_digests(&resumed);
    options.execution.reuse_actor_values = false;
    assert_eq!(
        run(options.clone(), &resumed, true)
            .expect_err("scope mismatch")
            .to_string(),
        "checkpoint scope mismatch: --reuse-actor-values: recorded <present>, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    options.execution.reuse_actor_values = true;
    assert_eq!(
        run_with(
            options.clone(),
            AnnealedHarness {
                stop_after_games: Some(4),
                ..harness()
            },
            &resumed,
            true,
        )
        .expect_err("stop after all games before optimizer commit")
        .to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    assert_eq!(generation_files(&resumed).len(), 4);
    let actual = run(options, &resumed, true).expect("replay aborted reuse update");
    assert_eq!(actual.completed_updates, 2);
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_probe_report_bits(&actual.latest, &expected.latest);
    assert_trajectory_equal(&uninterrupted, &resumed);
    for directory in [uninterrupted, resumed] {
        std::fs::remove_dir_all(directory).expect("remove own checkpoint");
    }
}

#[test]
fn actor_value_reuse_is_rejected_by_the_public_trainer_without_changing_state() {
    let model = PolicyModel::fresh(9001).expect("model");
    let mut trainer = crate::PpoTrainer::new(&model, settings(9001, 1).ppo, 19).expect("trainer");
    let random = trainer.rng_checkpoint();
    let error = trainer
        .set_execution(TrainingExecutionOptions {
            reuse_actor_values: true,
            ..Default::default()
        })
        .expect_err("collector-only option");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: reuse actor values requires the annealed collector"
    );
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn balanced_minibatches_resume_only_with_the_recorded_execution_scope() {
    let directory = test_directory("balanced-minibatch-scope");
    let mut options = settings(9001, 2);
    options.execution.balanced_minibatches = true;
    run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &directory,
        false,
    )
    .expect("first balanced update");
    let before = checkpoint_digests(&directory);
    options.execution.balanced_minibatches = false;
    assert_eq!(
        run(options.clone(), &directory, true)
            .unwrap_err()
            .to_string(),
        "checkpoint scope mismatch: --balanced-minibatches: recorded <present>, requested <absent>"
    );
    assert_eq!(checkpoint_digests(&directory), before);
    options.execution.balanced_minibatches = true;
    assert_eq!(
        run(options, &directory, true)
            .expect("balanced resume")
            .completed_updates,
        2
    );
    std::fs::remove_dir_all(directory).expect("remove own checkpoint");
}

fn folding_execution() -> TrainingExecutionOptions {
    TrainingExecutionOptions {
        host_math_workers: 4,
        ..Default::default()
    }
}

#[test]
fn concurrency_cli_rejects_retired_flags_and_bounds_scope_workers() {
    for arguments in [
        vec!["--actor-overlap", "continue-v1"],
        vec!["--learner-prefetch"],
    ] {
        let error = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments)
            .expect_err("retired option");
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
        let parsed = crate::cli::legacy_fixed_annealed_settings_for_test(&[
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
            ..Default::default()
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

fn parse_probe_reuse(value: Option<&std::ffi::OsStr>) -> Result<bool, &'static str> {
    parse_probe_balanced(value).map_err(|_| "DRYSUA_PROBE_REUSE_ACTOR_VALUES must be 0 or 1")
}

#[test]
fn probe_training_microbatch_is_closed_and_rejects_invalid_values() {
    use std::ffi::OsStr;
    assert_eq!(parse_probe_training_microbatch(None), Ok(64));
    for (value, expected) in [("64", 64), ("128", 128), ("256", 256)] {
        assert_eq!(
            parse_probe_training_microbatch(Some(OsStr::new(value))),
            Ok(expected)
        );
    }
    for value in ["", "0", "65", "512", " 64", "064", "-1"] {
        assert_eq!(
            parse_probe_training_microbatch(Some(OsStr::new(value))),
            Err("training microbatch must be 64, 128 or 256")
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            parse_probe_training_microbatch(Some(OsStr::from_bytes(&[255]))),
            Err("training microbatch must be 64, 128 or 256")
        );
    }
}

pub(super) fn parse_probe_training_microbatch(
    value: Option<&std::ffi::OsStr>,
) -> Result<usize, &'static str> {
    match value {
        None => Ok(64),
        Some(value) => value
            .to_str()
            .ok_or("training microbatch must be 64, 128 or 256")
            .and_then(crate::training_execution::parse_training_microbatch),
    }
}

#[test]
fn probe_actor_value_reuse_is_default_off_and_rejects_non_boolean_values() {
    use std::ffi::OsStr;
    assert_eq!(parse_probe_reuse(None), Ok(false));
    for (value, expected) in [("0", false), ("1", true)] {
        assert_eq!(parse_probe_reuse(Some(OsStr::new(value))), Ok(expected));
    }
    for value in ["", "true", "2", "-1"] {
        assert_eq!(
            parse_probe_reuse(Some(OsStr::new(value))),
            Err("DRYSUA_PROBE_REUSE_ACTOR_VALUES must be 0 or 1")
        );
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

#[test]
fn frozen_neural_probe_cases_validate_without_loading_and_reject_scalar_g2() {
    use std::ffi::OsStr;
    let weights = Path::new("unused-frozen-parent");
    for (case, games, batch, groups, batched) in [
        ("m8-scalar", 8, 8, 1, false),
        ("m8-batched", 8, 8, 1, true),
        ("m40-scalar", 40, 20, 1, false),
        ("m40-batched", 40, 20, 1, true),
        ("m40-g2-batched", 40, 20, 2, true),
    ] {
        let options = frozen_neural_settings(OsStr::new(case), weights).expect("probe case");
        assert_eq!(options.games_per_update, games);
        assert_eq!(options.parallel_worlds, batch);
        assert_eq!(options.execution.actor_pipeline_groups, groups);
        assert_eq!(options.execution.neural_opponent_batching, batched);
        assert_eq!(validate_annealed(&options, harness()), Ok(options.ppo));
        assert_eq!((options.ppo.epochs, options.ppo.minibatch), (4, 2048));
        assert_eq!(options.execution.training_microbatch, 256);
        assert!(options.execution.reuse_actor_values);
        assert_eq!(options.ppo.gae_lambda, 0.98);
    }
    for case in ["", "m40-g2-scalar", "m8", "M8-scalar"] {
        assert_eq!(
            frozen_neural_settings(OsStr::new(case), weights).unwrap_err(),
            "DRYSUA_PROBE_NN must be m8-scalar/m8-batched/m40-scalar/m40-batched/m40-g2-batched"
        );
    }
    let mut invalid =
        frozen_neural_settings(OsStr::new("m40-g2-batched"), weights).expect("batched G2");
    invalid.execution.neural_opponent_batching = false;
    assert_eq!(
        validate_annealed(&invalid, harness()),
        Err(PpoError::InvalidConfig(
            "annealed actor pipeline weights opponent requires batched inference"
        ))
    );
}

fn concurrency_probe(device: PolicyDevice) {
    if let Some(case) = std::env::var_os("DRYSUA_PROBE_NN") {
        frozen_neural_probe(device, &case);
        return;
    }
    let count = |name, default, maximum| {
        probe_count(std::env::var_os(name).as_deref(), default, maximum).expect(name)
    };
    let rounds = count("DRYSUA_PROBE_ROUNDS", 40, ANNEALED_EPISODE_DECISIONS);
    let training_microbatch = parse_probe_training_microbatch(
        std::env::var_os("DRYSUA_PROBE_TRAINING_MICROBATCH").as_deref(),
    )
    .expect("training microbatch mode");
    let mode = std::env::var("DRYSUA_PROBE_MODE").unwrap_or_else(|_| "c".to_owned());
    let execution = match mode.as_str() {
        "base" => TrainingExecutionOptions::default(),
        "c" => TrainingExecutionOptions {
            host_math_workers: count("DRYSUA_PROBE_WORKERS", 4, 32),
            ..Default::default()
        },
        _ => panic!("DRYSUA_PROBE_MODE must be base/c"),
    };
    let balanced = parse_probe_balanced(std::env::var_os("DRYSUA_PROBE_BALANCED").as_deref())
        .expect("balanced flag");
    let reuse_actor_values =
        parse_probe_reuse(std::env::var_os("DRYSUA_PROBE_REUSE_ACTOR_VALUES").as_deref())
            .expect("reuse actor values flag");
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
        "concurrency-workload mode={mode} device={device:?} worlds=40 rounds={rounds} epochs={} effective_minibatch={} microbatch={training_microbatch} seed=9001 balanced={balanced} reuse_actor_values={reuse_actor_values}",
        options.ppo.epochs, options.ppo.minibatch
    );
    let mut trials = Vec::with_capacity(order.len());
    for (index, &candidate) in order.iter().enumerate() {
        options.execution = if candidate {
            execution
        } else {
            TrainingExecutionOptions::default()
        };
        options.execution.reuse_actor_values = reuse_actor_values;
        // Both fresh trials use one numerical mode; cross-mode bit equality is not promised.
        options.execution.training_microbatch = training_microbatch;
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

fn frozen_neural_settings(
    case: &std::ffi::OsStr,
    weights: &Path,
) -> Result<AnnealedJobConfig, &'static str> {
    let (games, batch, groups, batched) = match case.to_str() {
        Some("m8-scalar") => (8, 8, 1, false),
        Some("m8-batched") => (8, 8, 1, true),
        Some("m40-scalar") => (40, 20, 1, false),
        Some("m40-batched") => (40, 20, 1, true),
        Some("m40-g2-batched") => (40, 20, 2, true),
        _ => {
            return Err(
                "DRYSUA_PROBE_NN must be m8-scalar/m8-batched/m40-scalar/m40-batched/m40-g2-batched",
            );
        }
    };
    let mut options = settings(9001, 1);
    options.opponent = AnnealedOpponent::Weights(weights.to_path_buf());
    options.games_per_update = games;
    options.parallel_worlds = batch;
    options.games_per_generation = games as u64;
    options.ppo.environments = games;
    options.ppo.sample_budget = crate::PpoSampleBudget::for_annealed_games(games);
    options.ppo.epochs = 4;
    options.ppo.minibatch = 2048;
    options.ppo.gae_lambda = 0.98;
    options.execution.actor_pipeline_groups = groups;
    options.execution.neural_opponent_batching = batched;
    options.execution.reuse_actor_values = true;
    options.execution.training_microbatch = 256;
    Ok(options)
}

fn frozen_neural_probe(device: PolicyDevice, case: &std::ffi::OsStr) {
    let parent =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("temp/side-actors-teacher-100-20260929-0738");
    let weights = std::env::var_os("DRYSUA_PROBE_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| parent.join("checkpoint"));
    assert!(
        weights == parent.join("checkpoint") || weights == parent.join("history/u0100"),
        "NN probe requires the pinned completed U100 checkpoint or history/u0100"
    );
    let options = frozen_neural_settings(case, &weights).expect("frozen NN case");
    validate_annealed(&options, harness()).expect("NN probe admission before loading");
    eprintln!(
        "frozen-nn case={case:?} device={device:?} weights={} expected_runtime_sha256=9f8bc4acaea9a2dcbb315f7d26227b2db3d33cc41a0b0b1d662e5411ac2a6280 sha_verified=false rounds=1024 seed=9001 epochs=4 minibatch=2048 microbatch=256 reuse=true lambda=0.98 collection_only=true execution={:?}",
        weights.display(),
        options.execution
    );
    let mut previous = None;
    for index in 0..3 {
        let work = frozen_neural_trial(&options, device, &weights, index);
        if let Some(previous) = previous {
            assert_eq!(work, previous, "fresh same-mode trials changed actual work");
        }
        previous = Some(work);
    }
}

fn frozen_neural_groups(
    options: &AnnealedJobConfig,
    opponent: &AnnealedOpponentRuntime,
    local: usize,
    random: &mut PpoRng,
) -> Vec<episode::ActorGroup> {
    let seats = balanced_policy_seats(options.seed, 0, options.games_per_update).expect("seats");
    let draw = draw_generation(
        options.seed,
        0,
        options.games_per_generation,
        options.games_per_update as u64,
        anneal_schedule(options),
    )
    .expect("native generation");
    (0..options.execution.actor_pipeline_groups)
        .map(|group| {
            let base = local + group * options.parallel_worlds;
            let end = base + options.parallel_worlds;
            assert!(end <= seats.len());
            let environments =
                batch_environments(options, base as u64, &seats[base..end], opponent, &draw)
                    .expect("native annealed worlds");
            episode::ActorGroup {
                stream_base: base,
                environments,
                streams: (base..end)
                    .map(|game| {
                        episode::game_stream(options.seed, game as u64).expect("native stream")
                    })
                    .collect(),
                random: actor_stream_rngs(random, options.parallel_worlds)
                    .expect("native actor RNGs"),
            }
        })
        .collect()
}

fn frozen_neural_trial(
    options: &AnnealedJobConfig,
    device: PolicyDevice,
    weights: &Path,
    index: usize,
) -> (u64, u64, usize, u64) {
    let model = TrainingArtifact::initialize_from_weights(weights, options.seed, device)
        .expect("native frozen learner initializer");
    let opponent = load_opponent(&options.opponent, device).expect("frozen opponent");
    let fingerprint = PolicySnapshot::capture(&model, 0)
        .expect("learner fingerprint")
        .fingerprint();
    assert_eq!(
        Some(fingerprint),
        opponent.fingerprint,
        "both policies must use the same tensors"
    );
    let mut random = PpoRng::new(options.seed ^ 0xa17e);
    let mut rollout = PpoRollout::for_config(options.ppo, model.policy_identity().expect("policy"))
        .expect("bounded rollout");
    let mut report = PpoSmokeReport::default();
    let mut decisions = 0_u64;
    let mut elapsed = 0_u128;
    let wave = options.parallel_worlds * options.execution.actor_pipeline_groups;
    for local in (0..options.games_per_update).step_by(wave) {
        let mut groups = frozen_neural_groups(options, &opponent.runtime, local, &mut random);
        let starts: Vec<_> = groups
            .iter()
            .flat_map(|group| group.environments.iter().map(|world| world.arena.tick()))
            .collect();
        // Exclude loading/world construction; include native collector setup and flushing.
        let started = Instant::now();
        frozen_neural_collect(&model, options, &mut groups, &mut rollout, &mut report);
        elapsed += started.elapsed().as_nanos();
        // Only the last, terminal decision can advance fewer than three ticks.
        decisions += groups
            .iter()
            .flat_map(|group| &group.environments)
            .zip(starts)
            .map(|(world, start)| {
                u64::from((world.arena.tick() - start).div_ceil(MAP2_DECISION_INTERVAL_TICKS))
            })
            .sum::<u64>();
    }
    let expected_decisions = options.games_per_update as u64 * 1024;
    let expected_ticks = expected_decisions * u64::from(MAP2_DECISION_INTERVAL_TICKS);
    let fixed_work = report.elapsed_ticks == expected_ticks
        && decisions == expected_decisions
        && report.terminal_wins + report.terminal_losses + report.terminal_draws == 0;
    eprintln!(
        "frozen-nn-end index={index} measured={} collection_ns={elapsed} fingerprint={fingerprint:016x} ticks={} decisions={decisions} retained_samples={} wins={} losses={} draws={} timeouts={} fixed_work={fixed_work} cross_mode_equality=unverified",
        index != 0,
        report.elapsed_ticks,
        rollout.len(),
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.episode_timeouts
    );
    assert!(
        fixed_work,
        "terminal/round divergence: reject fixed-work timing comparison"
    );
    assert!(!rollout.is_empty());
    (report.elapsed_ticks, decisions, rollout.len(), fingerprint)
}

fn frozen_neural_collect(
    model: &PolicyModel,
    options: &AnnealedJobConfig,
    groups: &mut [episode::ActorGroup],
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) {
    assert_eq!(groups.len(), options.execution.actor_pipeline_groups);
    assert!(matches!(groups.len(), 1 | 2));
    if groups.len() == 1 {
        let group = &mut groups[0];
        episode::collect_batch_with_opponent_batching(
            model,
            options.ppo,
            group.stream_base,
            "frozen-nn",
            &mut group.environments,
            &mut group.streams,
            &mut group.random,
            1024,
            rollout,
            report,
            true,
            options.execution.neural_opponent_batching,
        )
        .expect("native G1 collector");
    } else {
        episode::collect_actor_pipeline_with_opponent_batching(
            model,
            options.ppo,
            groups,
            1024,
            rollout,
            report,
            true,
            true,
        )
        .expect("native G2 collector");
    }
}

pub(super) fn assert_probe_report_bits(
    source: &crate::PpoUpdateReport,
    target: &crate::PpoUpdateReport,
) {
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

pub(super) fn assert_artifact_bits(
    source: &std::path::Path,
    target: &std::path::Path,
    device: PolicyDevice,
) {
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

pub(super) fn artifact_hash(directory: &std::path::Path, device: PolicyDevice) -> u64 {
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
