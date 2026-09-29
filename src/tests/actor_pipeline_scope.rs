use super::*;

#[test]
fn actor_pipeline_cli_legacy_explicit_one_preserves_scope_and_scopes_two_or_four_groups() {
    let arguments = [
        "--updates",
        "2",
        "--games",
        "8",
        "--parallel",
        "2",
        "--generation-games",
        "2",
        "--balanced-minibatches",
        "--host-math-workers",
        "4",
        "--reuse-actor-values",
    ];
    let mut legacy_arguments = arguments.to_vec();
    legacy_arguments.extend(["--actor-pipeline-groups", "1"]);
    let baseline = crate::cli::legacy_fixed_annealed_settings_for_test(&legacy_arguments)
        .expect("legacy explicit group one");
    assert_eq!(baseline.execution.actor_pipeline_groups, 1);
    assert_eq!(
        crate::TrainingExecutionOptions::default().actor_pipeline_groups,
        1
    );
    let scope = |settings: &AnnealedJobConfig| {
        annealed_run(settings, PolicyDevice::Cpu, settings.ppo, harness(), None)
            .expect("scope")
            .command_line
    };
    let original = scope(&baseline);
    assert!(!original.contains("--actor-pipeline-groups"));
    for value in ["1", "2", "4", "0", "3", "5"] {
        let mut arguments = arguments.to_vec();
        arguments.extend(["--actor-pipeline-groups", value]);
        let result = crate::cli::legacy_fixed_annealed_settings_for_test(&arguments);
        if matches!(value, "0" | "3" | "5") {
            assert!(
                result
                    .expect_err("group bound")
                    .to_string()
                    .contains("actor pipeline groups must be 1, 2 or 4")
            );
            continue;
        }
        let parsed = result.expect("supported groups");
        assert_eq!(parsed.execution.actor_pipeline_groups.to_string(), value);
        assert!(parsed.execution.balanced_minibatches);
        assert!(parsed.execution.reuse_actor_values);
        assert_eq!(parsed.execution.host_math_workers, 4);
        let expected = if value == "1" {
            original.clone()
        } else {
            format!("{original} --actor-pipeline-groups {value}")
        };
        assert_eq!(scope(&parsed), expected);
    }
    for operation in ["train", "train-full"] {
        let error = crate::cli::parse_from(["drysua", operation, "--actor-pipeline-groups", "2"])
            .expect_err("annealed-only option");
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        assert!(
            error
                .to_string()
                .contains("unexpected argument '--actor-pipeline-groups'")
        );
    }
}

#[test]
fn actor_pipeline_cli_real_defaults_validate_m40_b20_g2_and_keep_exact_adaptive_scope() {
    let settings =
        crate::cli::annealed_settings_for_test(&["--updates", "200", "--generation-games", "160"])
            .expect("real CLI defaults without legacy injection");
    assert_eq!(settings.games_per_update, 40);
    assert_eq!(settings.parallel_worlds, 20);
    assert_eq!(
        settings.execution,
        crate::TrainingExecutionOptions {
            actor_pipeline_groups: 2,
            training_microbatch: 256,
            reuse_actor_values: true,
            ..Default::default()
        }
    );
    let config = validate_annealed(&settings, AnnealedHarness::default())
        .expect("production-sized default profile without executing games");
    assert_eq!(config, settings.ppo);
    let scope = annealed_run(
        &settings,
        PolicyDevice::Cpu,
        config,
        AnnealedHarness::default(),
        None,
    )
    .expect("canonical default scope");
    assert_eq!(
        scope.command_line,
        concat!(
            "train-annealed --updates 200 --games 40 --parallel 20 --generation-games 160",
            " --zero-updates 40 --epochs 4 --minibatch 2048 --seed 9001 --map 2 --device cpu",
            " --sample-budget annealed-v1 --opponent teacher --reuse-actor-values",
            " --actor-pipeline-groups 2 --training-microbatch 256",
            " --environment-schedule adaptive --environment-success-updates 2",
            " --environment-success-rate 0.8 --environment-poor-updates 1",
            " --environment-poor-rate 0.2 --environment-extension 0.75"
        )
    );
}

#[test]
fn actor_pipeline_cli_explicit_legacy_execution_restores_exact_fixed_scope() {
    let settings = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "200",
        "--generation-games",
        "32",
        "--games",
        "8",
        "--parallel",
        "8",
        "--actor-pipeline-groups",
        "1",
        "--training-microbatch",
        "64",
        "--reuse-actor-values=false",
        "--environment-schedule",
        "fixed",
    ])
    .expect("explicit legacy CLI execution");
    assert_eq!(
        settings.execution,
        crate::TrainingExecutionOptions::default()
    );
    assert_eq!(
        settings.environment_schedule,
        crate::EnvironmentSchedule::Fixed
    );
    let config = validate_annealed(&settings, AnnealedHarness::default())
        .expect("legacy profile without executing games");
    assert_eq!(config, settings.ppo);
    let scope = annealed_run(
        &settings,
        PolicyDevice::Cpu,
        config,
        AnnealedHarness::default(),
        None,
    )
    .expect("canonical legacy scope");
    assert_eq!(
        scope.command_line,
        concat!(
            "train-annealed --updates 200 --games 8 --parallel 8 --generation-games 32",
            " --zero-updates 40 --epochs 4 --minibatch 2048 --seed 9001 --map 2 --device cpu",
            " --opponent teacher"
        )
    );
}

#[test]
fn actor_pipeline_cli_explicit_g4_b10_validates_with_other_real_defaults() {
    let settings = crate::cli::annealed_settings_for_test(&[
        "--updates",
        "200",
        "--generation-games",
        "160",
        "--actor-pipeline-groups",
        "4",
        "--parallel",
        "10",
    ])
    .expect("explicit four-group profile");
    assert_eq!(settings.games_per_update, 40);
    assert_eq!(settings.parallel_worlds, 10);
    assert_eq!(settings.execution.actor_pipeline_groups, 4);
    assert_eq!(settings.execution.training_microbatch, 256);
    assert!(settings.execution.reuse_actor_values);
    assert_eq!(
        validate_annealed(&settings, AnnealedHarness::default()).expect("M40 B10 G4 profile"),
        settings.ppo
    );
}

#[test]
fn actor_pipeline_rejects_invalid_groups_partial_waves_and_eighty_active_worlds() {
    for (groups, games, parallel) in [(2, 4, 2), (4, 40, 10), (4, 64, 16)] {
        let mut options = pipeline_settings(groups);
        options.games_per_update = games;
        options.parallel_worlds = parallel;
        options.games_per_generation = parallel as u64;
        options.ppo.environments = games;
        options.ppo.sample_budget = PpoSampleBudget::for_annealed_games(games);
        assert_eq!(options.execution.training_microbatch, 64);
        assert_eq!(
            validate_annealed(&options, harness()).expect("teacher groups"),
            options.ppo
        );
    }
    for (groups, games, parallel, message) in [
        (0, 4, 2, "actor pipeline groups must be 1, 2 or 4"),
        (3, 4, 2, "actor pipeline groups must be 1, 2 or 4"),
        (5, 4, 2, "actor pipeline groups must be 1, 2 or 4"),
        (
            2,
            6,
            2,
            "annealed actor pipeline wave must divide games per update",
        ),
        (
            4,
            12,
            2,
            "annealed actor pipeline wave must divide games per update",
        ),
        (
            4,
            40,
            20,
            "annealed actor pipeline active worlds must not exceed 64",
        ),
    ] {
        let mut options = pipeline_settings(2);
        options.execution.actor_pipeline_groups = groups;
        options.games_per_update = games;
        options.parallel_worlds = parallel;
        options.games_per_generation = parallel as u64;
        options.ppo.environments = games;
        options.ppo.sample_budget = PpoSampleBudget::for_annealed_games(games);
        assert_eq!(
            validate_annealed(&options, harness()),
            Err(PpoError::InvalidConfig(message))
        );
    }
}

#[test]
fn actor_pipeline_rejects_weights_before_opening_the_opponent_or_checkpoint() {
    let mut options = pipeline_settings(4);
    options.opponent = AnnealedOpponent::Weights(PathBuf::from("unused-pipeline-opponent"));
    assert_eq!(
        run(
            options.clone(),
            Path::new("unused-pipeline-checkpoint"),
            false
        ),
        Err(PpoError::InvalidConfig(
            "annealed actor pipeline weights opponent requires batched inference"
        ))
    );
    options.execution.actor_pipeline_groups = 1;
    assert_eq!(
        validate_annealed(&options, harness()).expect("G1 weights supported"),
        options.ppo
    );
}

#[test]
fn actor_pipeline_resumes_exactly_after_scope_rejection_and_an_aborted_wave() {
    for groups in [2, 4] {
        assert_pipeline_resume(groups);
    }
}

fn assert_pipeline_resume(groups: usize) {
    let uninterrupted = test_directory("actor-pipeline-uninterrupted");
    let resumed = test_directory("actor-pipeline-resumed");
    let mut options = pipeline_settings(groups);
    assert!(harness().episode_decisions() >= 9);
    let expected = run(options.clone(), &uninterrupted, false).expect("pipeline updates");
    let first = run_with(
        options.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &resumed,
        false,
    )
    .expect("committed first wave");
    assert_eq!(first.completed_updates, 1);
    assert_eq!(first.games, (2 * groups) as u64);
    assert!(first.optimizer_step > 0);
    let before = checkpoint_digests(&resumed);
    options.execution.actor_pipeline_groups = 1;
    assert_eq!(
        run(options.clone(), &resumed, true)
            .expect_err("scope mismatch")
            .to_string(),
        format!(
            "checkpoint scope mismatch: --actor-pipeline-groups: recorded {groups}, requested <absent>"
        )
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    options.execution.actor_pipeline_groups = groups;
    assert_eq!(
        run_with(
            options.clone(),
            AnnealedHarness {
                stop_after_games: Some(2 * groups - 1),
                ..harness()
            },
            &resumed,
            true,
        )
        .expect_err("abort after crossing game limit within wave")
        .to_string(),
        "invalid PPO transition: annealed invocation stopped mid-update"
    );
    assert_eq!(checkpoint_digests(&resumed), before);
    assert_eq!(generation_files(&resumed).len(), 2 * groups);
    let actual = run(options, &resumed, true).expect("replay aborted wave");
    assert_eq!(actual.completed_updates, 2);
    assert_eq!(actual.games, expected.games);
    assert_eq!(actual.rollout_samples, expected.rollout_samples);
    assert_eq!(actual.optimizer_step, expected.optimizer_step);
    assert_trajectory_equal(&uninterrupted, &resumed);
    for directory in [uninterrupted, resumed] {
        std::fs::remove_dir_all(directory).expect("cleanup checkpoint");
    }
}

fn pipeline_settings(groups: usize) -> AnnealedJobConfig {
    let mut options = settings(9001, 2);
    options.games_per_update = 2 * groups;
    options.ppo.environments = 2 * groups;
    options.execution.actor_pipeline_groups = groups;
    options.execution.reuse_actor_values = true;
    options.execution.balanced_minibatches = true;
    options.execution.host_math_workers = 4;
    options
}

#[test]
fn actor_pipeline_probe_controls_preserve_work_and_reject_unsupported_profiles() {
    use std::ffi::OsStr;
    assert_eq!(parse_pipeline_probe_profile(None, None), Ok((1, 20, 40)));
    for groups in ["1", "2", "4"] {
        for (width, games) in [("10", 40), ("16", 64), ("20", 40), ("32", 64)] {
            let actual =
                parse_pipeline_probe_profile(Some(OsStr::new(groups)), Some(OsStr::new(width)));
            if groups == "4" && matches!(width, "20" | "32") {
                assert_eq!(
                    actual,
                    Err("actor pipeline probe active worlds must not exceed 64")
                );
            } else {
                assert_eq!(
                    actual,
                    Ok((groups.parse().unwrap(), width.parse().unwrap(), games))
                );
            }
        }
    }
    for value in ["", "0", "3", "5", "01", "+2", " 4", "4 "] {
        assert_eq!(
            parse_pipeline_probe_profile(Some(OsStr::new(value)), None),
            Err("actor pipeline groups must be 1, 2 or 4")
        );
    }
    for value in ["", "0", "8", "12", "24", "010", "+10", "32 "] {
        assert_eq!(
            parse_pipeline_probe_profile(None, Some(OsStr::new(value))),
            Err("actor pipeline probe width must be 10, 16, 20 or 32")
        );
    }
}

#[test]
#[ignore = "native paired probe; requires the authorized exclusive runner"]
fn actor_pipeline_probe_cpu() {
    pipeline_probe(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "native paired CUDA probe; requires the authorized exclusive runner"]
fn actor_pipeline_probe_cuda() {
    pipeline_probe(PolicyDevice::Cuda { ordinal: 0 });
}

fn pipeline_probe(device: PolicyDevice) {
    let (groups, width, games) = parse_pipeline_probe_profile(
        std::env::var_os("DRYSUA_ACTOR_PIPELINE_GROUPS").as_deref(),
        std::env::var_os("DRYSUA_ACTOR_PIPELINE_WIDTH").as_deref(),
    )
    .expect("actor pipeline probe profile");
    assert!(width * groups <= 64);
    assert!(games.is_multiple_of(width * groups));
    let rounds = pipeline_probe_count("DRYSUA_PROBE_ROUNDS", 512, ANNEALED_EPISODE_DECISIONS);
    let mut options = settings(9952600, 1);
    options.games_per_update = games;
    options.parallel_worlds = width;
    options.games_per_generation = games as u64;
    options.ppo.environments = games;
    options.ppo.sample_budget = PpoSampleBudget::for_annealed_games(games);
    options.ppo.minibatch = 2048;
    options.ppo.epochs = 4;
    options.execution.actor_pipeline_groups = groups;
    options.execution.reuse_actor_values = true;
    options.execution.training_microbatch = concurrency_tests::parse_probe_training_microbatch(
        std::env::var_os("DRYSUA_PROBE_TRAINING_MICROBATCH").as_deref(),
    )
    .expect("training microbatch mode");
    let weights = std::env::var_os("DRYSUA_PROBE_WEIGHTS").map(PathBuf::from);
    let mut trials = Vec::with_capacity(2);
    for trial in 0..2 {
        let directory = test_directory("actor-pipeline-probe");
        eprintln!(
            "actor-pipeline-start trial={trial} groups={groups} batch={width} rounds={rounds} microbatch={} epochs=4 minibatch=2048",
            options.execution.training_microbatch
        );
        let report = pipeline_probe_trial(&options, device, rounds, &directory, weights.as_deref());
        eprintln!(
            "actor-pipeline-end trial={trial} groups={groups} batch={width} rounds={rounds} games={} samples={} optimizer_steps={} ticks={} state_hash={:016x} latest={:?}",
            report.games,
            report.rollout_samples,
            report.optimizer_step,
            report.elapsed_ticks,
            concurrency_tests::artifact_hash(&directory, device),
            report.latest
        );
        trials.push((directory, report));
    }
    assert_eq!(trials[0].1, trials[1].1);
    concurrency_tests::assert_probe_report_bits(&trials[0].1.latest, &trials[1].1.latest);
    concurrency_tests::assert_artifact_bits(&trials[0].0, &trials[1].0, device);
    for (directory, _) in trials {
        std::fs::remove_dir_all(directory).expect("remove own probe checkpoint");
    }
}

fn pipeline_probe_trial(
    options: &AnnealedJobConfig,
    device: PolicyDevice,
    rounds: usize,
    directory: &Path,
    weights: Option<&Path>,
) -> AnnealedJobReport {
    let started = Instant::now();
    let report = run_annealed_job_harnessed(
        options.clone(),
        AnnealedHarness {
            episode_decisions: Some(rounds),
            ..Default::default()
        },
        device,
        directory,
        false,
        weights,
        |_| {},
    )
    .expect("bounded native pipeline probe");
    eprintln!(
        "actor-pipeline-probe games={} rounds={rounds} seed=9952600 reuse_actor_values=true device={device:?} elapsed={:?} samples={} optimizer_steps={} latest={:?}",
        options.games_per_update,
        started.elapsed(),
        report.rollout_samples,
        report.optimizer_step,
        report.latest
    );
    report
}

fn parse_pipeline_probe_profile(
    groups: Option<&std::ffi::OsStr>,
    width: Option<&std::ffi::OsStr>,
) -> Result<(usize, usize, usize), &'static str> {
    let groups = usize::from(crate::training_execution::parse_actor_pipeline_groups(
        groups
            .unwrap_or(std::ffi::OsStr::new("1"))
            .to_str()
            .ok_or("actor pipeline groups must be 1, 2 or 4")?,
    )?);
    let (width, games): (usize, usize) = match width.unwrap_or(std::ffi::OsStr::new("20")).to_str()
    {
        Some("10") => (10, 40),
        Some("16") => (16, 64),
        Some("20") => (20, 40),
        Some("32") => (32, 64),
        _ => return Err("actor pipeline probe width must be 10, 16, 20 or 32"),
    };
    if width * groups > 64 {
        return Err("actor pipeline probe active worlds must not exceed 64");
    }
    if !games.is_multiple_of(width * groups) {
        return Err("actor pipeline probe wave must divide games per update");
    }
    Ok((groups, width, games))
}

fn pipeline_probe_count(name: &str, default: usize, maximum: usize) -> usize {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => default,
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|value| (1..=maximum).contains(value))
            .unwrap_or_else(|| panic!("{name} must be within 1..={maximum}")),
        Err(error) => panic!("{name}: {error}"),
    }
}
