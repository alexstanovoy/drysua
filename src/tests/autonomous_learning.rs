#![allow(
    clippy::float_arithmetic,
    reason = "Bounded experimental learning/evaluation statistics"
)]
use super::super::{
    advance_interval, build_environment, neural_policy_request_in_space, prepare_policy_sample,
    requests_with_candidate, take_map2_reward, terminal_outcome,
};
use super::*;
use crate::{ActionSpace, FeatureFrame, Map2RewardEnd, PpoTerminalOutcome, StructuredAction};
#[path = "learning_policy.rs"]
mod learning_policy;

const ROOT: &str = "/home/alexstanovoy/Workspace/bots/drysua/temp/autonomous-20260927";
const INITIAL: &str = "/home/alexstanovoy/Workspace/bots/drysua/artifacts/temp/annealed-teacher-20260920/attempt-008/history/update-0428";
const TRAIN_SEED: u64 = 20260927;
const WIDE_INITIAL: &str = "/home/alexstanovoy/Workspace/bots/drysua/artifacts/deslop-20260922/learning-experiments/lab-final-artifacts/secondary-credit-u10";

fn wave_two() -> bool {
    std::env::var("LAB_WAVE").is_ok_and(|wave| wave == "2")
}
fn learning_wave() -> bool {
    std::env::var("LAB_WAVE").is_ok_and(|wave| wave == "3")
}
fn lab_root() -> PathBuf {
    if learning_wave() {
        let root = std::env::var_os("LAB_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(ROOT).join("learning-20260928"));
        assert!(
            root.starts_with(ROOT)
                && !root
                    .components()
                    .any(|part| matches!(part, std::path::Component::ParentDir))
        );
        assert_ne!(root, Path::new(ROOT));
        let existing = root
            .ancestors()
            .take(32)
            .find(|path| path.exists())
            .expect("existing lab ancestor");
        assert!(
            existing
                .canonicalize()
                .unwrap()
                .starts_with(Path::new(ROOT).canonicalize().unwrap()),
            "lab root escapes private tree"
        );
        root
    } else if wave_two() {
        Path::new(ROOT).join("wave2")
    } else {
        ROOT.into()
    }
}

#[test]
#[ignore = "explicit three-hour rootless experiment; never production resume"]
fn autonomous_lab() {
    std::thread::Builder::new()
        .name("autonomous-lab".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let _flush = FlushPerformanceLogs;
            assert!(!prometheus::enabled());
            if let Ok(wave) = std::env::var("LAB_WAVE") {
                assert!(matches!(wave.as_str(), "1" | "2" | "3"), "unknown LAB_WAVE");
            }
            if learning_wave() {
                validate_learning_environment();
            }
            match std::env::var("LAB_MODE").unwrap().as_str() {
                "evaluate" | "sharpen-check" => evaluate(),
                "blend" if learning_wave() => blend_initialization(),
                "sharpen" if learning_wave() => sharpen_initialization(),
                "ensemble" if learning_wave() => ensemble_initialization(),
                "policy-check" if learning_wave() => policy_parity_check(),
                "train" => train(),
                "fork-check" if learning_wave() => train(),
                "wide-smoke" => wide_smoke(),
                "memory" => memory_screen(),
                _ => panic!("LAB_MODE must be evaluate/train"),
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

fn settings(arm: &str) -> AnnealedJobConfig {
    if learning_wave() {
        return learning_settings(arm);
    }
    if wave_two() {
        assert!(matches!(
            arm,
            "width" | "wave-control" | "wave-rnd" | "wave-large"
        ));
        let games: usize = std::env::var("LAB_M").unwrap().parse().unwrap();
        let parallel: usize = std::env::var("LAB_B").unwrap().parse().unwrap();
        assert!((2..=80).contains(&games) && games.is_multiple_of(2));
        assert!((1..=64).contains(&parallel) && games.is_multiple_of(parallel));
        let mut ppo = TrainingArtifact::load(Path::new(WIDE_INITIAL))
            .unwrap()
            .config();
        ppo.environments = games;
        ppo.sample_budget = crate::PpoSampleBudget::for_annealed_games(games);
        ppo.learning_rate = 1e-5;
        ppo.gae_lambda = 1.0;
        ppo.entropy_coefficient = 0.01;
        return AnnealedJobConfig {
            environment_schedule: crate::EnvironmentSchedule::Fixed,
            execution: Default::default(),
            updates: 30,
            games_per_update: games,
            parallel_worlds: parallel,
            games_per_generation: 5 * games as u64,
            zero_updates: 30,
            seed: 20260929,
            opponent: AnnealedOpponent::Teacher,
            ppo: ppo.validate().unwrap(),
            checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
            invocation_updates: None,
            git_commit: "35f71af-uncommitted-wide-rnd-lab-v2".into(),
            simulator_commit: "8ce82af-dirty-mechanical-fixes".into(),
        };
    }
    assert!(matches!(arm, "control" | "credit" | "sil" | "precision"));
    let mut ppo = TrainingArtifact::load(Path::new(INITIAL)).unwrap().config();
    ppo.learning_rate = if arm == "control" { 3e-6 } else { 3e-5 };
    ppo.gae_lambda = if arm == "control" { 0.98 } else { 1.0 };
    ppo.entropy_coefficient = if arm == "precision" { 0.001 } else { 0.01 };
    let updates = if arm == "precision" { 10 } else { 30 };
    AnnealedJobConfig {
        environment_schedule: crate::EnvironmentSchedule::Fixed,
        execution: crate::TrainingExecutionOptions::default(),
        updates,
        games_per_update: 40,
        parallel_worlds: 40,
        games_per_generation: 200,
        zero_updates: updates,
        seed: TRAIN_SEED + u64::from(arm == "precision"),
        opponent: AnnealedOpponent::Teacher,
        ppo,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        invocation_updates: None,
        git_commit: "35f71af-experimental-uncommitted-lab-v1".into(),
        simulator_commit: "8ce82af-dirty-mechanical-fixes".into(),
    }
}

fn lab_value<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("invalid {name}: {value}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(_) => panic!("non-Unicode {name}"),
    }
}

fn learning_parent() -> PathBuf {
    std::env::var_os("LAB_PARENT")
        .map(PathBuf::from)
        .unwrap_or_else(|| WIDE_INITIAL.into())
}

fn source_hash(name: &str) -> String {
    let hash = std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"));
    assert!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid {name}"
    );
    hash
}

fn validate_learning_environment() {
    const ALLOWED: &[&str] = &[
        "LAB_WAVE",
        "LAB_MODE",
        "LAB_ARM",
        "LAB_ROOT",
        "LAB_PARENT",
        "LAB_FORK",
        "LAB_LR",
        "LAB_LAMBDA",
        "LAB_ENTROPY",
        "LAB_EPOCHS",
        "LAB_MINIBATCH",
        "LAB_TARGET_KL",
        "LAB_CLIP",
        "LAB_GRAD_CLIP",
        "LAB_VALUE_COEFFICIENT",
        "LAB_HOLD_UPDATES",
        "LAB_SCHEDULE",
        "LAB_ZERO_UPDATES",
        "LAB_UPDATES",
        "LAB_EXECUTION_PROFILE",
        "LAB_STEPS",
        "LAB_SOURCE_SHA256",
        "LAB_SIMULATOR_SHA256",
        "LAB_BINARY_SHA256",
        "LAB_WEIGHTS",
        "LAB_POLICY",
        "LAB_SET",
        "LAB_PAIR_OFFSET",
        "LAB_SNAPSHOT",
        "LAB_FINAL_MANIFEST",
        "LAB_MODEL_VARIANT",
        "LAB_RADIANT_MODEL",
        "LAB_DIRE_MODEL",
        "LAB_BLEND_ALPHA",
        "LAB_OUTPUT",
    ];
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("LAB_") {
            assert!(
                ALLOWED.iter().any(|allowed| name == *allowed),
                "unknown lab variable {name:?}"
            );
        }
    }
    let mode = std::env::var("LAB_MODE").unwrap();
    assert!(matches!(
        mode.as_str(),
        "train"
            | "evaluate"
            | "fork-check"
            | "blend"
            | "sharpen"
            | "ensemble"
            | "policy-check"
            | "sharpen-check"
    ));
    if matches!(mode.as_str(), "train" | "fork-check") {
        for name in [
            "LAB_SOURCE_SHA256",
            "LAB_SIMULATOR_SHA256",
            "LAB_BINARY_SHA256",
        ] {
            source_hash(name);
        }
        let steps = lab_value("LAB_STEPS", 2usize);
        assert!((1..=2).contains(&steps));
    }
}

fn learning_execution(profile: &str) -> (usize, crate::TrainingExecutionOptions) {
    let mut execution = crate::TrainingExecutionOptions::default();
    let parallel = match profile {
        "baseline" => 40,
        "g1-b40-m256-reuse" => {
            execution.training_microbatch = 256;
            execution.reuse_actor_values = true;
            40
        }
        "g2-b20-m256-reuse" => {
            execution.actor_pipeline_groups = 2;
            execution.training_microbatch = 256;
            execution.reuse_actor_values = true;
            20
        }
        _ => panic!("unknown LAB_EXECUTION_PROFILE: {profile}"),
    };
    (parallel, execution)
}

fn learning_settings(arm: &str) -> AnnealedJobConfig {
    assert!(
        arm.starts_with("a8-")
            && arm.len() <= 48
            && arm
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
        "invalid learning arm"
    );
    let parent =
        TrainingArtifact::load(&learning_parent()).expect("complete parent checkpoint required");
    let mut ppo = parent.config();
    assert_eq!(ppo.environments, 40, "fork requires M40 parent");
    ppo.learning_rate = lab_value("LAB_LR", 3e-6);
    ppo.gae_lambda = lab_value("LAB_LAMBDA", 1.0);
    ppo.entropy_coefficient = lab_value("LAB_ENTROPY", 0.01);
    ppo.epochs = lab_value("LAB_EPOCHS", 4);
    ppo.minibatch = lab_value("LAB_MINIBATCH", 2048);
    ppo.target_kl = lab_value("LAB_TARGET_KL", 0.02);
    ppo.clip_epsilon = lab_value("LAB_CLIP", 0.2);
    ppo.gradient_clip = lab_value("LAB_GRAD_CLIP", 0.5);
    ppo.value_coefficient = lab_value("LAB_VALUE_COEFFICIENT", 0.5);
    validate_learning_ppo(ppo);
    let updates = lab_value("LAB_UPDATES", 128u64);
    let hold = lab_value("LAB_HOLD_UPDATES", 5u64);
    assert!((1..=128).contains(&updates) && updates > parent.progress().global_update);
    assert!(matches!(hold, 5 | 20 | 40));
    let schedule = std::env::var("LAB_SCHEDULE").unwrap_or_else(|_| "clean".into());
    let zero_updates = lab_value("LAB_ZERO_UPDATES", updates);
    match schedule.as_str() {
        "clean" => {
            assert_eq!(zero_updates, updates);
            assert_eq!(hold, 5, "clean K changes are a no-op");
        }
        "annealed" => assert!(
            zero_updates >= 5
                && zero_updates < updates
                && parent.progress().global_update < updates - zero_updates,
            "annealed fork needs an active modifier window"
        ),
        _ => panic!("LAB_SCHEDULE must be clean or annealed"),
    }
    let profile = std::env::var("LAB_EXECUTION_PROFILE").unwrap_or_else(|_| "baseline".into());
    let (parallel_worlds, execution) = learning_execution(&profile);
    let settings = AnnealedJobConfig {
        environment_schedule: crate::EnvironmentSchedule::Fixed,
        execution,
        updates,
        games_per_update: 40,
        parallel_worlds,
        games_per_generation: 40 * hold,
        zero_updates,
        seed: parent.run().run_seed,
        opponent: AnnealedOpponent::Teacher,
        ppo,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        invocation_updates: None,
        git_commit: format!("lab-source-sha256:{}", source_hash("LAB_SOURCE_SHA256")),
        simulator_commit: format!("lab-source-sha256:{}", source_hash("LAB_SIMULATOR_SHA256")),
    };
    validate_annealed(&settings, AnnealedHarness::default())
        .expect("valid experimental collection");
    settings
}

fn validate_learning_ppo(config: crate::PpoConfig) {
    config.validate().expect("valid learning PPO config");
    assert!((1e-7..=3e-5).contains(&config.learning_rate));
    assert!((0.98..=1.0).contains(&config.gae_lambda));
    assert!((0.0001..=0.01).contains(&config.entropy_coefficient));
    assert!(matches!(config.epochs, 2 | 4));
    assert!(matches!(config.minibatch, 1024 | 2048 | 4096));
    assert!((0.001..=0.02).contains(&config.target_kl));
    assert!((0.1..=0.2).contains(&config.clip_epsilon));
    assert!((0.1..=0.5).contains(&config.gradient_clip));
    assert!((0.1..=1.0).contains(&config.value_coefficient));
    assert_eq!(config.gamma_tick, 1.0);
}

fn fork_scope(run: &mut CheckpointRun) -> TrainingArtifact {
    let parent = learning_parent().canonicalize().expect("parent path");
    let metadata = small_hash(&parent.join("checkpoint.meta"));
    let artifact = TrainingArtifact::load(&parent).expect("parent artifact");
    assert_eq!(
        metadata,
        small_hash(&parent.join("checkpoint.meta")),
        "parent changed while loading"
    );
    let mode = std::env::var("LAB_FORK").expect("LAB_FORK must be explicit");
    assert!(matches!(mode.as_str(), "preserve" | "reset_adam"));
    let update = artifact.progress().global_update;
    let snapshot = std::env::var("LAB_SNAPSHOT").unwrap_or_else(|_| "auto".into());
    assert!(matches!(snapshot.as_str(), "auto" | "now" | "none"));
    run.command_line.push_str(&format!(
        " --experiment autonomous-20260928 --fork {mode} --parent {} --parent-meta {} --parent-runtime {} --parent-update {update} --binary-sha256 {} --generation-prefix counterfactual-bookkeeping --generation-clock cumulative --actor-shuffle-rng preserved",
        parent.display(), metadata,
        bounded_hash(&parent.join("drysua.weights.safetensors"), 16 * 1024 * 1024),
        source_hash("LAB_BINARY_SHA256")));
    artifact
}

fn initialize_learning_child(
    settings: &AnnealedJobConfig,
    directory: &Path,
    run: CheckpointRun,
    artifact: &TrainingArtifact,
) {
    let parent = learning_parent().canonicalize().expect("parent");
    assert_ne!(
        directory.canonicalize().expect("child"),
        parent,
        "parent cannot be destination"
    );
    assert_eq!(artifact.run().run_seed, settings.seed);
    assert!(artifact.run().mastery_config.is_none());
    let mut state = restore_fork_parent(artifact);
    let reset = std::env::var("LAB_FORK").unwrap() == "reset_adam";
    let before_step = state.trainer.optimizer_step();
    let before_rng = (state.trainer.rng_checkpoint(), state.sampling.checkpoint());
    let before_parameters = parameter_hash(&state.model);
    state
        .trainer
        .fork_learning(&state.model, settings.ppo, reset)
        .expect("optimizer/config fork");
    assert_eq!(
        before_rng,
        (state.trainer.rng_checkpoint(), state.sampling.checkpoint())
    );
    assert_eq!(before_parameters, parameter_hash(&state.model));
    assert_eq!(
        state.trainer.optimizer_step(),
        if reset { 0 } else { before_step }
    );
    state.run = run;
    let games = state.completed_updates * settings.games_per_update as u64;
    let generations = games.div_ceil(settings.games_per_generation);
    assert!(generations <= 256);
    let random_directory = directory.join(RANDOMIZATION_DIRECTORY);
    for generation in 0..generations {
        let draw = draw_generation(
            settings.seed,
            generation,
            settings.games_per_generation,
            settings.games_per_update as u64,
            anneal_schedule(settings),
        )
        .expect("fork generation prefix");
        write_generation_snapshot(&random_directory, &draw).expect("child generation receipt");
    }
    eprintln!(
        "lab-fork parent={} update={} reset_adam={reset} optimizer_before={before_step} optimizer_after={} sampling={:?} shuffle={:?} parameters_sha256={before_parameters} prefix_generations={generations} prefix_executed=false",
        parent.display(),
        state.completed_updates,
        state.trainer.optimizer_step(),
        before_rng.1,
        before_rng.0
    );
    state
        .save(directory, state.checkpoint_report(None))
        .expect("truthful child checkpoint");
}

fn restore_fork_parent(artifact: &TrainingArtifact) -> TrainingSession {
    let model = PolicyModel::fresh_on(artifact.run().run_seed, PolicyDevice::Cuda { ordinal: 0 })
        .expect("fork model");
    let (trainer, run, progress) = artifact
        .restore(&model, artifact.run())
        .expect("strict parent restoration")
        .into_parts();
    assert_eq!(progress.global_update, progress.policy_version);
    let sampling =
        super::super::restore_sampling_rng(&progress.rng_states).expect("parent sampling RNG");
    let fingerprint = PolicySnapshot::capture(&model, progress.global_update)
        .expect("parent fingerprint")
        .fingerprint();
    TrainingSession {
        model,
        trainer,
        sampling,
        counters: Default::default(),
        run,
        completed_updates: progress.global_update,
        rollout_samples: progress.rollout_samples,
        latest: Default::default(),
        migrated_provenance: false,
        mastery: progress.mastery,
        adaptive_environment: progress.adaptive_environment,
        starting_policy_fingerprint: fingerprint,
    }
}

fn train() {
    let arm = std::env::var("LAB_ARM").unwrap();
    let settings = settings(&arm);
    let initial = if learning_wave() {
        learning_parent()
    } else if wave_two() {
        PathBuf::from(WIDE_INITIAL)
    } else if arm == "precision" {
        Path::new(ROOT).join("history/sil/u0010")
    } else {
        PathBuf::from(INITIAL)
    };
    let directory = lab_root().join(if arm == "width" {
        format!(
            "width-{}-{}",
            settings.games_per_update, settings.parallel_worlds
        )
    } else {
        arm.clone()
    });
    let random_directory = directory.join(RANDOMIZATION_DIRECTORY);
    let mut run = annealed_run(
        &settings,
        PolicyDevice::Cuda { ordinal: 0 },
        settings.ppo,
        AnnealedHarness::default(),
        None,
    )
    .unwrap();
    if learning_wave() {
        run.command_line.push_str(&format!(" --method {arm}"));
    } else {
        run.command_line
            .push_str(&format!(" --experiment autonomous-20260927 --method {arm}"));
    }
    if matches!(arm.as_str(), "sil" | "precision") {
        run.command_line
            .push_str(" --aux-current-mc-rows 256 --aux-batch 64 --aux-coefficient 0.05");
    }
    if arm == "precision" {
        run.command_line
            .push_str(" --initial-lab-checkpoint sil-u0010 --precision-reset-adam");
    }
    if arm == "wave-rnd" {
        run.command_line.push_str(&format!(
            " --rnd {} --rnd-schedule warm2-beta2e-5-flat10-zero15",
            crate::ppo::Rnd::SPEC
        ));
    }
    if arm == "wave-large" {
        assert_eq!(
            (settings.games_per_update, settings.parallel_worlds),
            (80, 40)
        );
        run.command_line.push_str(" --seat-schedule m40-blocks");
    }
    let parent_artifact = if learning_wave() {
        run.command_line
            .push_str(&format!(" --full-ppo {:?}", settings.ppo));
        Some(fork_scope(&mut run))
    } else {
        None
    };
    let parent_update = parent_artifact
        .as_ref()
        .map_or(0, |artifact| artifact.progress().global_update);
    let resume = directory.join("checkpoint.meta").exists();
    assert!(
        !std::fs::symlink_metadata(&directory)
            .is_ok_and(|metadata| metadata.file_type().is_symlink()),
        "lab directory cannot be a symlink"
    );
    if learning_wave() && !resume {
        assert!(
            !directory.exists(),
            "new learning fork requires an absent child directory"
        );
    }
    std::fs::create_dir_all(&random_directory).unwrap();
    let _ownership = TrainingDirectoryLock::acquire(&directory).expect("exclusive lab directory");
    if !resume && let Some(parent) = &parent_artifact {
        initialize_learning_child(&settings, &directory, run.clone(), parent);
    }
    let mut session = AnnealedSession::initialize(
        &settings,
        PolicyDevice::Cuda { ordinal: 0 },
        &directory,
        resume || learning_wave(),
        (!resume && !learning_wave()).then_some(initial.as_path()),
        settings.ppo,
        run,
        &random_directory,
        AnnealedOpponentRuntime::Teacher,
    )
    .unwrap();
    let mut rnd =
        (arm == "wave-rnd").then(|| load_rnd(&directory, session.state.completed_updates));
    if !resume && !learning_wave() {
        assert_eq!(session.state.trainer.optimizer_step(), 0);
        assert_eq!(session.state.completed_updates, 0);
        assert_eq!(session.state.sampling.draws(), 0);
    }
    eprintln!(
        "lab-start arm={arm} resume={resume} updates={} adam_steps={} initial_fingerprint={:016x} learning_rate={} lambda={} entropy={}",
        session.state.completed_updates,
        session.state.trainer.optimizer_step(),
        session.state.starting_policy_fingerprint,
        settings.ppo.learning_rate,
        settings.ppo.gae_lambda,
        settings.ppo.entropy_coefficient
    );
    if std::env::var("LAB_MODE").is_ok_and(|mode| mode == "fork-check") {
        eprintln!(
            "lab-fork-check update={} parent_update={parent_update} config={:?} execution={:?}",
            session.state.completed_updates,
            session.state.trainer.config(),
            settings.execution
        );
        return;
    }
    let mut generations = GenerationCache::new(
        random_directory,
        settings.seed,
        settings.games_per_generation,
        settings.games_per_update as u64,
        anneal_schedule(&settings),
        session.generations,
    );
    let limit: usize = std::env::var("LAB_STEPS")
        .unwrap_or_else(|_| "2".into())
        .parse()
        .unwrap();
    assert!((1..=2).contains(&limit));
    let started = Instant::now();
    for _ in 0..limit {
        if session.state.completed_updates >= settings.updates {
            break;
        }
        let update = session.state.completed_updates;
        let tick = Instant::now();
        let mut rollout =
            PpoRollout::for_config(settings.ppo, session.state.model.policy_identity().unwrap())
                .unwrap();
        let mut report = PpoSmokeReport::default();
        let seats = if arm == "wave-large" {
            [update * 2, update * 2 + 1]
                .into_iter()
                .flat_map(|block| balanced_policy_seats(settings.seed, block, 40).unwrap())
                .collect()
        } else {
            balanced_policy_seats(settings.seed, update, settings.games_per_update).unwrap()
        };
        let collecting = Instant::now();
        for local in (0..settings.games_per_update)
            .step_by(settings.parallel_worlds * settings.execution.actor_pipeline_groups)
        {
            session
                .collect_update_wave(
                    &settings,
                    AnnealedHarness::default(),
                    settings.ppo,
                    &mut generations,
                    local,
                    &seats,
                    &mut rollout,
                    &mut report,
                )
                .unwrap();
        }
        let collection_seconds = collecting.elapsed().as_secs_f64();
        assert_eq!(
            report.completed_episodes.ordered_outcomes().len(),
            settings.games_per_update
        );
        let samples = rollout.len();
        let rnd_started = Instant::now();
        if let Some(rnd) = &mut rnd {
            let coefficient = rnd_coefficient(update + 1);
            let novelty: crate::ppo::RndReport = rnd.augment(&mut rollout, coefficient).unwrap();
            assert_eq!(novelty.updates, update + 1);
            eprintln!(
                "lab-rnd update={} coefficient={coefficient} rows={} error_mean={} rms={} bonus_sum={} bonus_mean={} bonus_max={} predictor_loss_before={} predictor_loss_after={} predictor_steps={}",
                update + 1,
                novelty.rows_scored,
                novelty.error_mean,
                novelty.normalization_rms,
                novelty.bonus_sum,
                novelty.bonus_sum / novelty.rows_scored.max(1) as f64,
                novelty.max_bonus,
                novelty.predictor_loss_before,
                novelty.predictor_loss_after,
                novelty.sgd_steps
            );
        }
        let rnd_seconds = rnd_started.elapsed().as_secs_f64();
        let batch = rollout.finish(settings.ppo).unwrap();
        let optimizing = Instant::now();
        session.state.latest = session
            .state
            .trainer
            .train_update(&session.state.model, &batch)
            .unwrap();
        let optimization_seconds = optimizing.elapsed().as_secs_f64();
        let auxiliary_steps = if matches!(arm.as_str(), "sil" | "precision") {
            apply_imitation(&mut session.state.trainer, &session.state.model, &batch)
        } else {
            0
        };
        session.state.completed_updates = session.state.trainer.updates();
        session.state.rollout_samples += samples as u64;
        session.state.counters.merge(&report).unwrap();
        session.games += settings.games_per_update as u64;
        session.generations = generations.counted_through();
        if let Some(rnd) = &rnd {
            rnd.save(
                &directory.join(format!("rnd-{:04}.bin", session.state.completed_updates)),
                session.state.completed_updates,
            )
            .unwrap();
        }
        session
            .state
            .save(&directory, session.state.checkpoint_report(None))
            .unwrap();
        let completed = session.state.completed_updates;
        if rnd.is_some() {
            bind_rnd(&directory, completed);
        }
        eprintln!(
            "lab-update arm={arm} update={completed} games={} worlds={} wins={} losses={} draws={} samples={samples} adam_steps={} ppo_steps={} kl={} entropy={} reward={} seconds={} auxiliary_steps={auxiliary_steps} collection_seconds={collection_seconds} optimization_seconds={optimization_seconds} rnd_seconds={rnd_seconds} ticks={} parameters_sha256={}",
            settings.games_per_update,
            settings.parallel_worlds,
            report.terminal_wins,
            report.terminal_losses,
            report.terminal_draws,
            session.state.trainer.optimizer_step(),
            session.state.latest.minibatches,
            super::super::update_kl(session.state.latest),
            session.state.latest.entropy,
            report.map2_reward.total,
            tick.elapsed().as_secs_f64(),
            report.elapsed_ticks,
            parameter_hash(&session.state.model)
        );
        eprintln!(
            "lab-objective update={completed} local_update={} gradient_norm={} applied_scale={} clip_fraction={} policy_loss={} value_loss={} execution={:?}",
            completed - parent_update,
            session.state.latest.gradient_norm,
            session.state.latest.applied_scale,
            session.state.latest.clip_fraction,
            session.state.latest.policy_loss,
            session.state.latest.value_loss,
            settings.execution
        );
        let progress = format!(
            "{{\"arm\":\"{arm}\",\"updates\":{completed},\"parent_update\":{parent_update},\"local_updates\":{},\"games\":{},\"optimizer_steps\":{},\"checkpoint_sha256\":\"{}\",\"parameters_sha256\":\"{}\"}}\n",
            completed - parent_update,
            completed * settings.games_per_update as u64,
            session.state.trainer.optimizer_step(),
            small_hash(&directory.join("checkpoint.meta")),
            parameter_hash(&session.state.model)
        );
        std::fs::write(directory.join("lab_progress.json"), &progress).unwrap();
        if learning_wave() {
            use std::io::Write;
            let mut receipt = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(format!("lab-update-{completed:04}.json")))
                .expect("unique update receipt");
            receipt.write_all(progress.as_bytes()).unwrap();
            receipt.sync_all().unwrap();
        }
        if learning_wave() {
            let policy = std::env::var("LAB_SNAPSHOT").unwrap_or_else(|_| "auto".into());
            if policy == "now" || policy == "auto" && matches!(completed - parent_update, 5 | 10) {
                snapshot(&directory, &arm, completed);
            }
        } else if completed.is_multiple_of(10)
            || arm == "precision" && completed == 5
            || wave_two() && arm != "width" && completed.is_multiple_of(5)
        {
            snapshot(&directory, &arm, completed);
        }
        if started.elapsed().as_secs() >= 95 {
            break;
        }
    }
}

fn apply_imitation(
    trainer: &mut crate::PpoTrainer,
    model: &PolicyModel,
    batch: &crate::PpoBatch,
) -> u64 {
    let updates = trainer.updates();
    let before = trainer.optimizer_step();
    let count = batch.len().min(256);
    let mut applied = 0;
    for start in (0..count).step_by(64) {
        let samples: Vec<_> = (start..(start + 64).min(count))
            .map(|index| {
                batch
                    .sample((2 * index + 1) * batch.len() / (2 * count))
                    .unwrap()
            })
            .collect();
        let report = trainer
            .self_imitation_update(model, &samples, 0.05)
            .unwrap();
        applied += u64::from(report.applied);
        eprintln!(
            "lab-aux rows={} applied={} kl={} loss={}",
            samples.len(),
            report.applied,
            report.approximate_kl,
            report.policy_loss
        );
    }
    assert_eq!(trainer.updates(), updates);
    assert_eq!(trainer.optimizer_step(), before + applied);
    applied
}

fn rnd_coefficient(update: u64) -> f32 {
    match update {
        0..=2 => 0.0,
        3..=10 => 2e-5,
        11..=14 => 2e-5 * (15 - update) as f32 / 5.0,
        _ => 0.0,
    }
}

fn small_hash(path: &Path) -> String {
    bounded_hash(path, 65536)
}

fn bounded_hash(path: &Path, maximum: u64) -> String {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    assert!(maximum <= 16 * 1024 * 1024);
    let file = std::fs::File::open(path).unwrap();
    assert!(file.metadata().unwrap().len() <= maximum);
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() as u64 <= maximum);
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parameter_hash(model: &PolicyModel) -> String {
    use sha2::{Digest, Sha256};
    let parameters = model.export_parameters().unwrap();
    assert_eq!(parameters.len(), crate::MODEL_PARAMETER_COUNT);
    let mut digest = Sha256::new();
    for value in parameters {
        digest.update(value.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn load_rnd(directory: &Path, update: u64) -> crate::ppo::Rnd {
    if update == 0 {
        assert!(!directory.join("rnd-receipt.json").exists());
        return crate::ppo::Rnd::new(20260929);
    }
    let path = directory.join("rnd-receipt.json");
    assert!(std::fs::metadata(&path).unwrap().len() < 8192);
    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(receipt["update"].as_u64(), Some(update));
    assert_eq!(receipt["spec"].as_str(), Some(crate::ppo::Rnd::SPEC));
    assert_eq!(
        receipt["schedule"].as_str(),
        Some("warm2-beta2e-5-flat10-zero15")
    );
    assert_eq!(
        receipt["checkpoint_sha256"].as_str(),
        Some(small_hash(&directory.join("checkpoint.meta")).as_str())
    );
    let sidecar = directory.join(format!("rnd-{update:04}.bin"));
    assert_eq!(
        receipt["rnd_sha256"].as_str(),
        Some(small_hash(&sidecar).as_str())
    );
    crate::ppo::Rnd::load(&sidecar, update).unwrap()
}

fn bind_rnd(directory: &Path, update: u64) {
    use std::io::Write;
    let receipt = serde_json::json!({"update":update,"spec":crate::ppo::Rnd::SPEC,
        "schedule":"warm2-beta2e-5-flat10-zero15","checkpoint_sha256":small_hash(&directory.join("checkpoint.meta")),
        "rnd_sha256":small_hash(&directory.join(format!("rnd-{update:04}.bin")))});
    let temporary = directory.join("rnd-receipt.tmp");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .unwrap();
    file.write_all(serde_json::to_string(&receipt).unwrap().as_bytes())
        .unwrap();
    file.sync_all().unwrap();
    drop(file);
    std::fs::rename(temporary, directory.join("rnd-receipt.json")).unwrap();
    std::fs::File::open(directory).unwrap().sync_all().unwrap();
    let _ = load_rnd(directory, update);
}

fn wide_smoke() {
    let settings = settings("width");
    assert_eq!(settings.parallel_worlds, 64);
    let model = PolicyModel::fresh_on(1, PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    TrainingArtifact::load_runtime_weights(&model, Path::new(WIDE_INITIAL)).unwrap();
    let seats = balanced_policy_seats(settings.seed, 0, 64).unwrap();
    let draw = draw_generation(settings.seed, 0, 320, 64, anneal_schedule(&settings)).unwrap();
    let mut environments = batch_environments(
        &settings,
        0,
        &seats,
        &AnnealedOpponentRuntime::Teacher,
        &draw,
    )
    .unwrap();
    let mut streams = (0..64)
        .map(|game| episode::game_stream(settings.seed, game).unwrap())
        .collect::<Vec<_>>();
    let mut random = actor_stream_rngs(&mut PpoRng::new(settings.seed), 64).unwrap();
    let mut rollout =
        PpoRollout::for_config(settings.ppo, model.policy_identity().unwrap()).unwrap();
    let mut report = PpoSmokeReport::default();
    episode::collect_batch(
        &model,
        settings.ppo,
        0,
        "wide-smoke",
        &mut environments,
        &mut streams,
        &mut random,
        16,
        &mut rollout,
        &mut report,
    )
    .unwrap();
    assert!(!rollout.is_empty());
    assert_eq!(report.elapsed_ticks, 64 * 16 * 3);
    for environment in environments {
        reject_production_rejection(&environment, "wide smoke").unwrap();
    }
    eprintln!(
        "wide-cuda-smoke worlds=64 rounds=16 samples={} ticks={} rejected_orders=0",
        rollout.len(),
        report.elapsed_ticks
    );
}

fn memory_screen() {
    let settings = settings("width");
    assert_eq!(settings.games_per_update, 40);
    let model = PolicyModel::fresh_on(1, PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    TrainingArtifact::load_runtime_weights(&model, Path::new(WIDE_INITIAL)).unwrap();
    let before = model.export_parameters().unwrap();
    let seed = 20261005;
    let mut environments = (0..40)
        .map(|index| {
            build_environment(
                derive_training_seed(seed, index / 2, crate::randomization::ARENA_DOMAIN),
                derive_training_seed(seed, index / 2, crate::randomization::OPPONENT_DOMAIN),
                MapId(2),
                index as usize % 2,
                0,
                OpponentSpec::Teacher,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut streams = (0..40)
        .map(|index| episode::game_stream(seed, index).unwrap())
        .collect::<Vec<_>>();
    let mut random = (0..40)
        .map(|index| PpoRng::new(derive_training_seed(seed, index, 0x006d_656d)))
        .collect::<Vec<_>>();
    let mut rollout =
        PpoRollout::for_config(settings.ppo, model.policy_identity().unwrap()).unwrap();
    let mut report = PpoSmokeReport::default();
    episode::collect_batch(
        &model,
        settings.ppo,
        0,
        "memory-screen",
        &mut environments,
        &mut streams,
        &mut random,
        ACTOR_DECISIONS,
        &mut rollout,
        &mut report,
    )
    .unwrap();
    assert_eq!(report.completed_episodes.ordered_outcomes().len(), 40);
    let batch = rollout.finish(settings.ppo).unwrap();
    eprintln!(
        "memory-cohort seed={seed} pairs=20 games=40 wins={} losses={} draws={} samples={} production_optimizer_steps=0 intrinsic_rewards=false",
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        batch.len()
    );
    model
        .actor_snapshot()
        .unwrap()
        .memory_probe(&batch)
        .unwrap();
    crate::tests::support::assert_bits(&before, &model.export_parameters().unwrap());
}

fn snapshot(source: &Path, arm: &str, update: u64) {
    if learning_wave() {
        let history = lab_root().join("history").join(arm);
        if history.exists() && history.read_dir().unwrap().take(4).count() >= 4 {
            eprintln!(
                "lab-snapshot-cap arm={arm} update={update} retained=4 latest_checkpoint_preserved=true"
            );
            return;
        }
    }
    let target = lab_root()
        .join("history")
        .join(arm)
        .join(format!("u{update:04}"));
    assert!(!target.exists());
    std::fs::create_dir_all(target.join(RANDOMIZATION_DIRECTORY)).unwrap();
    for (index, entry) in source.read_dir().unwrap().enumerate() {
        assert!(index < 512);
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == ".training.lock" {
            continue;
        }
        if name == RANDOMIZATION_DIRECTORY {
            for (index, file) in entry.path().read_dir().unwrap().enumerate() {
                assert!(index < 256);
                let file = file.unwrap();
                assert!(file.file_type().unwrap().is_file());
                std::fs::copy(file.path(), target.join(&name).join(file.file_name())).unwrap();
            }
        } else {
            assert!(entry.file_type().unwrap().is_file());
            std::fs::copy(entry.path(), target.join(name)).unwrap();
        }
    }
}

struct EvaluationWorld {
    environment: TrainingEnvironment,
    actions: [u32; 16],
    reward: crate::Map2TrainingReward,
    action_trace: u64,
}
enum EvalRequest {
    Prepare,
    Act(StructuredAction, Box<ActionSpace>),
}
struct EvalReply {
    prepared: Option<(FeatureFrame, ActionSpace)>,
    outcome: Option<PpoTerminalOutcome>,
    tick: u32,
}

fn eval_worker(
    _: usize,
    world: &mut EvaluationWorld,
    request: EvalRequest,
) -> Result<EvalReply, PpoError> {
    let environment = &mut world.environment;
    let mut outcome = None;
    if let EvalRequest::Act(action, space) = request {
        let (_, candidate) = neural_policy_request_in_space(
            &mut environment.seats[environment.policy_seat],
            action,
            &space,
        )?;
        let requests = requests_with_candidate(environment, candidate)?;
        let before = environment.arena.tick();
        let advanced =
            advance_interval(environment, requests, 3.min(crate::MAP2_TICK_CAP - before))?;
        reject_production_rejection(environment, "lab evaluation")?;
        outcome = terminal_outcome(environment, advanced.winner);
        let end = match outcome {
            Some(PpoTerminalOutcome::Win) => Some(Map2RewardEnd::Win),
            Some(PpoTerminalOutcome::Loss) => Some(Map2RewardEnd::Loss),
            Some(PpoTerminalOutcome::Draw) => Some(Map2RewardEnd::Draw),
            None => None,
        };
        world
            .reward
            .record(take_map2_reward(environment, end, advanced.ticks)?)?;
        world.actions[action.kind().index()] += 1;
        world.action_trace =
            crate::model::fnv1a_extend(world.action_trace, format!("{action:?}").as_bytes());
    }
    let prepared = if outcome.is_none() {
        Some(prepare_policy_sample(environment)?)
    } else {
        None
    };
    Ok(EvalReply {
        prepared,
        outcome,
        tick: environment.arena.tick(),
    })
}

fn evaluation_seed(set: &str, offset: u64) -> u64 {
    match set {
        "autonomous-validation" => {
            assert!(matches!(offset, 0 | 20));
            2026092802
        }
        "autonomous-final" => {
            assert!(offset <= 180 && offset.is_multiple_of(20));
            2026092803
        }
        "selection" => {
            assert_eq!(offset, 0);
            20261001
        }
        "confirmation" => {
            assert!(matches!(offset, 0 | 20));
            20261002
        }
        "wave-selection" => {
            assert_eq!(offset, 0);
            20261003
        }
        "wave-confirmation" => {
            assert!(matches!(offset, 0 | 20));
            20261004
        }
        _ => panic!("invalid evaluation set"),
    }
}

struct EvaluationPolicy {
    radiant: PolicyModel,
    dire: Option<PolicyModel>,
    identity: String,
    experts: Option<(String, String)>,
    description: String,
}

impl EvaluationPolicy {
    fn load(default: &Path, device: PolicyDevice) -> Self {
        let variant = std::env::var("LAB_MODEL_VARIANT").unwrap_or_else(|_| "single".into());
        let radiant = std::env::var_os("LAB_RADIANT_MODEL").map(PathBuf::from);
        let dire = std::env::var_os("LAB_DIRE_MODEL").map(PathBuf::from);
        let (radiant_path, dire_path) = match variant.as_str() {
            "single" => {
                assert!(
                    radiant.is_none() && dire.is_none(),
                    "single forbids expert paths"
                );
                (default.to_path_buf(), None)
            }
            "side-experts" => {
                assert!(
                    std::env::var_os("LAB_WEIGHTS").is_none(),
                    "side-experts forbids LAB_WEIGHTS"
                );
                (
                    radiant.expect("missing Radiant expert"),
                    Some(dire.expect("missing Dire expert")),
                )
            }
            _ => panic!("unknown LAB_MODEL_VARIANT"),
        };
        let radiant_hash = bounded_hash(
            &radiant_path.join("drysua.weights.safetensors"),
            16 * 1024 * 1024,
        );
        let dire_hash = dire_path
            .as_ref()
            .map(|path| bounded_hash(&path.join("drysua.weights.safetensors"), 16 * 1024 * 1024));
        let identity = dire_hash.as_ref().map_or_else(
            || radiant_hash.clone(),
            |dire| learning_policy::ensemble_id(&radiant_hash, dire).unwrap(),
        );
        let radiant = load_evaluation_model(&radiant_path, device);
        let dire = dire_path
            .as_ref()
            .map(|path| load_evaluation_model(path, device));
        let experts = dire_hash.map(|dire| (radiant_hash, dire));
        Self {
            radiant,
            dire,
            identity,
            experts,
            description: if variant == "single" {
                radiant_path.display().to_string()
            } else {
                format!("team-ensemble:{}", learning_policy::ROUTING)
            },
        }
    }

    /// Strict ensemble contract: both full-batch experts must succeed, even on
    /// unselected rows. No error-equivalence promise to a standalone expert.
    /// Commit caller RNG only after every model and cached route is validated.
    fn actions(
        &self,
        frames: &[FeatureFrame],
        spaces: &[ActionSpace],
        random: &mut [PpoRng],
        routes: &[usize],
        sampled: bool,
    ) -> Result<Vec<StructuredAction>, crate::ModelError> {
        if frames.is_empty()
            || frames.len() > 40
            || frames.len() != routes.len()
            || frames.len() != random.len()
            || frames.len() != spaces.len()
            || routes.iter().any(|route| *route > 1)
        {
            return Err(crate::ModelError::InvalidModelState(
                "evaluation batch or cached route",
            ));
        }
        let mut radiant_random = random.to_vec();
        let radiant =
            evaluation_actions(&self.radiant, frames, spaces, &mut radiant_random, sampled)?;
        let Some(dire_model) = &self.dire else {
            random.clone_from_slice(&radiant_random);
            return Ok(radiant);
        };
        let mut dire_random = random.to_vec();
        let dire = evaluation_actions(dire_model, frames, spaces, &mut dire_random, sampled)?;
        let actions = routes
            .iter()
            .enumerate()
            .map(|(index, route)| match route {
                0 => {
                    dire_random[index] = radiant_random[index].clone();
                    radiant[index]
                }
                1 => dire[index],
                _ => panic!("invalid cached own-team route"),
            })
            .collect();
        random.clone_from_slice(&dire_random);
        Ok(actions)
    }

    fn parameter_hashes(&self) -> (String, Option<String>) {
        (
            parameter_hash(&self.radiant),
            self.dire.as_ref().map(parameter_hash),
        )
    }
}

fn load_evaluation_model(path: &Path, device: PolicyDevice) -> PolicyModel {
    let model = PolicyModel::fresh_on(1, device).unwrap();
    TrainingArtifact::load_runtime_weights(&model, path).unwrap();
    model
}

fn evaluation_actions(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    random: &mut [PpoRng],
    sampled: bool,
) -> Result<Vec<StructuredAction>, crate::ModelError> {
    if sampled {
        Ok(model
            .sample_batch(frames, spaces, random)?
            .into_iter()
            .map(|choice| choice.action())
            .collect())
    } else {
        Ok(model
            .choose_batch(frames, spaces)?
            .into_iter()
            .map(|choice| choice.action)
            .collect())
    }
}

#[test]
fn ensemble_unselected_expert_failure_rejects_batch_without_consuming_caller_rng() {
    let mut worlds = parity_worlds(0);
    let (frame, space) = prepare_policy_sample(&mut worlds[0].environment).unwrap();
    let healthy = PolicyModel::fresh(391).unwrap();
    let poisoned = PolicyModel::fresh(392).unwrap();
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in poisoned.parameter_schema().unwrap() {
        let end = offset + shape.iter().product::<usize>();
        if name == "trunk.2.bias" {
            parameters[offset..end].fill(1.0);
        }
        if name == "kind.weight" {
            parameters[offset..end].fill(f32::MAX);
        }
        offset = end;
    }
    poisoned.import_parameters(&parameters).unwrap();
    let policy = EvaluationPolicy {
        radiant: healthy,
        dire: Some(poisoned),
        identity: "error-contract-test".into(),
        experts: None,
        description: "strict-ensemble-test".into(),
    };
    let mut random = vec![PpoRng::new(881)];
    let before = random.clone();
    let frames = [frame];
    let spaces = [space];
    let mut standalone = random.clone();
    assert_eq!(
        evaluation_actions(&policy.radiant, &frames, &spaces, &mut standalone, true)
            .unwrap()
            .len(),
        1
    );
    assert_ne!(standalone, before);
    let error = policy
        .actions(&frames, &spaces, &mut random, &[0], true)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "model kind output at batch 0 index 0 is non-finite"
    );
    assert_eq!(random, before);
    let error = policy
        .actions(&frames, &spaces, &mut random, &[2], true)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "model produced invalid evaluation batch or cached route"
    );
    assert_eq!(random, before);
}

#[test]
fn ensemble_mixed_routes_preserve_selected_full_batch_actions_and_rng() {
    let mut worlds = [parity_worlds(0).remove(0), parity_worlds(1).remove(0)];
    let routes: Vec<_> = worlds
        .iter()
        .map(|world| {
            learning_policy::observed_team(
                &world.environment.seats[world.environment.policy_seat].tracker,
            )
            .unwrap()
        })
        .collect();
    assert_eq!(routes, [0, 1]);
    let (frames, spaces): (Vec<_>, Vec<_>) = worlds
        .iter_mut()
        .map(|world| prepare_policy_sample(&mut world.environment).unwrap())
        .unzip();
    let policy = EvaluationPolicy {
        radiant: routing_fixture_model(0),
        dire: Some(routing_fixture_model(1)),
        identity: "mixed-route-test".into(),
        experts: None,
        description: "mixed-route-test".into(),
    };
    for sampled in [false, true] {
        let mut actual_rng = vec![PpoRng::new(981), PpoRng::new(982)];
        let mut radiant_rng = actual_rng.clone();
        let mut dire_rng = actual_rng.clone();
        let radiant =
            evaluation_actions(&policy.radiant, &frames, &spaces, &mut radiant_rng, sampled)
                .unwrap();
        let dire = evaluation_actions(
            policy.dire.as_ref().unwrap(),
            &frames,
            &spaces,
            &mut dire_rng,
            sampled,
        )
        .unwrap();
        assert_ne!(
            radiant, dire,
            "mixed-route fixture must distinguish experts"
        );
        let actual = policy
            .actions(&frames, &spaces, &mut actual_rng, &routes, sampled)
            .unwrap();
        assert_eq!(actual, [radiant[0], dire[1]]);
        assert_eq!(actual_rng, [radiant_rng[0].clone(), dire_rng[1].clone()]);
    }
}

fn routing_fixture_model(kind: usize) -> PolicyModel {
    assert!(kind < 2);
    let model = PolicyModel::fresh(491 + kind as u64).unwrap();
    let mut parameters = model.export_parameters().unwrap();
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().unwrap() {
        let end = offset + shape.iter().product::<usize>();
        if name == "kind.bias" {
            parameters[offset..end].fill(-1000.0);
            parameters[offset + kind] = 1000.0;
        }
        offset = end;
    }
    assert_eq!(offset, crate::MODEL_PARAMETER_COUNT);
    model.import_parameters(&parameters).unwrap();
    model
}

fn runtime_budget(path: &Path) -> crate::PpoSampleBudget {
    use std::io::Read;
    let mut file = std::fs::File::open(path.join("drysua.weights.safetensors")).unwrap();
    let mut length = [0u8; 8];
    file.read_exact(&mut length).unwrap();
    let length = u64::from_le_bytes(length);
    assert!((2..=65536).contains(&length), "bounded runtime header");
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let version: u32 = header["__metadata__"]["ppo_schema_version"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    let hash: u64 = header["__metadata__"]["ppo_schema_hash"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    [
        crate::PpoSampleBudget::Standard,
        crate::PpoSampleBudget::Annealed,
        crate::PpoSampleBudget::WideAnnealed,
    ]
    .into_iter()
    .find(|budget| budget.schema_version() == version && budget.schema_hash() == hash)
    .expect("runtime capacity profile must match a supported exact identity")
}

fn sharpen_directory() -> PathBuf {
    let name = std::env::var("LAB_OUTPUT").expect("missing calibration output name");
    assert!(
        !name.is_empty()
            && name.len() <= 48
            && name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    );
    let base = lab_root().join("sharpened");
    let directory = base.join(name);
    assert!(!directory.exists(), "calibration destination must be new");
    std::fs::create_dir_all(&base).unwrap();
    assert!(
        base.canonicalize()
            .unwrap()
            .starts_with(lab_root().canonicalize().unwrap())
    );
    std::fs::create_dir(&directory).unwrap();
    directory
}

fn sharpen_initialization() {
    assert_eq!(std::env::var("LAB_MODEL_VARIANT").unwrap(), "side-experts");
    let policy = EvaluationPolicy::load(Path::new(WIDE_INITIAL), PolicyDevice::Cpu);
    let before = policy.parameter_hashes();
    let radiant_parameters =
        learning_policy::sharpen_four(&policy.radiant).expect("finite Radiant calibration");
    let dire_parameters = learning_policy::sharpen_four(policy.dire.as_ref().unwrap())
        .expect("finite Dire calibration");
    let sources = [
        PathBuf::from(std::env::var_os("LAB_RADIANT_MODEL").unwrap()),
        PathBuf::from(std::env::var_os("LAB_DIRE_MODEL").unwrap()),
    ];
    let budgets = sources.each_ref().map(|source| runtime_budget(source));
    let (radiant_parent, dire_parent) = policy.experts.as_ref().unwrap();
    for (source, expected) in sources.iter().zip([radiant_parent, dire_parent]) {
        assert_eq!(
            &bounded_hash(&source.join("drysua.weights.safetensors"), 16 * 1024 * 1024),
            expected,
            "source changed during calibration"
        );
    }
    let radiant = PolicyModel::fresh(1).unwrap();
    let dire = PolicyModel::fresh(1).unwrap();
    radiant.import_parameters(&radiant_parameters).unwrap();
    dire.import_parameters(&dire_parameters).unwrap();
    assert_eq!(before, policy.parameter_hashes());
    let directory = sharpen_directory();
    for ((model, team), budget) in [(&radiant, "radiant"), (&dire, "dire")]
        .into_iter()
        .zip(budgets)
    {
        let child = directory.join(team);
        std::fs::create_dir(&child).unwrap();
        TrainingArtifact::save_runtime_weights_with_budget(model, &child, budget).unwrap();
    }
    let radiant_hash = bounded_hash(
        &directory.join("radiant/drysua.weights.safetensors"),
        16 * 1024 * 1024,
    );
    let dire_hash = bounded_hash(
        &directory.join("dire/drysua.weights.safetensors"),
        16 * 1024 * 1024,
    );
    let manifest = serde_json::json!({"schema":"team-ensemble-sharpen4/v1", "kind":"posttraining-calibration-not-trained-checkpoint",
        "factor":4, "conditional_temperature":0.25, "optimizer":"absent", "affected_parameters_per_expert":learning_policy::SHARPENED_PARAMETERS,
        "unchanged_parameters_per_expert":crate::MODEL_PARAMETER_COUNT-learning_policy::SHARPENED_PARAMETERS,
        "parent_radiant_runtime_sha256":radiant_parent, "parent_dire_runtime_sha256":dire_parent,
        "radiant_runtime_sha256":radiant_hash, "dire_runtime_sha256":dire_hash, "radiant_directory":directory.join("radiant"), "dire_directory":directory.join("dire"),
        "routing":learning_policy::ROUTING, "policy_sha256":learning_policy::ensemble_id(&radiant_hash, &dire_hash).unwrap(), "primary_candidate_replaced":false});
    std::fs::write(
        directory.join("ensemble.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    eprintln!(
        "lab-sharpen output={} manifest={manifest}",
        directory.display()
    );
}

fn blend_initialization() {
    assert_eq!(std::env::var("LAB_MODEL_VARIANT").unwrap(), "side-experts");
    let policy = EvaluationPolicy::load(Path::new(WIDE_INITIAL), PolicyDevice::Cpu);
    let alpha: f32 = std::env::var("LAB_BLEND_ALPHA")
        .expect("missing blend alpha")
        .parse()
        .expect("invalid blend alpha");
    let parameters = learning_policy::blend_parameters(
        &policy.radiant.export_parameters().unwrap(),
        &policy.dire.as_ref().unwrap().export_parameters().unwrap(),
        alpha,
    )
    .unwrap();
    let name = std::env::var("LAB_OUTPUT").expect("missing blend output name");
    assert!(
        !name.is_empty()
            && name.len() <= 48
            && name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    );
    let base = lab_root().join("blends");
    std::fs::create_dir_all(&base).unwrap();
    assert!(
        base.canonicalize()
            .unwrap()
            .starts_with(lab_root().canonicalize().unwrap())
    );
    let directory = base.join(name);
    assert!(!directory.exists(), "blend output must be new");
    let model = PolicyModel::fresh(1).unwrap();
    model.import_parameters(&parameters).unwrap();
    std::fs::create_dir(&directory).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &directory).unwrap();
    let (radiant, dire) = policy.experts.unwrap();
    let receipt = serde_json::json!({"schema":"weight-blend-v1", "kind":"single-nn-weight-initialization-not-trained-checkpoint", "radiant_runtime_sha256":radiant,
        "dire_runtime_sha256":dire, "radiant_weight":alpha, "alpha_bits":alpha.to_bits(), "optimizer":"absent", "runtime_sha256":bounded_hash(&directory.join("drysua.weights.safetensors"), 16*1024*1024)});
    std::fs::write(
        directory.join("blend.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    eprintln!("lab-blend output={} receipt={receipt}", directory.display());
}

fn ensemble_initialization() {
    let policy = EvaluationPolicy::load(Path::new(WIDE_INITIAL), PolicyDevice::Cpu);
    let (radiant, dire) = policy
        .experts
        .as_ref()
        .expect("ensemble requires both experts");
    let name = std::env::var("LAB_OUTPUT").expect("missing ensemble output name");
    assert!(
        !name.is_empty()
            && name.len() <= 48
            && name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    );
    let directory = lab_root().join("ensembles").join(name);
    assert!(!directory.exists(), "ensemble artifact must be new");
    std::fs::create_dir_all(directory.parent().unwrap()).unwrap();
    assert!(
        directory
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .starts_with(lab_root().canonicalize().unwrap())
    );
    std::fs::create_dir(&directory).unwrap();
    let manifest = serde_json::json!({"schema":"team-ensemble-v1", "kind":"two-full-neural-policies-not-single-runtime", "routing":learning_policy::ROUTING,
        "policy_sha256":policy.identity, "radiant_runtime_sha256":radiant, "dire_runtime_sha256":dire,
        "radiant_directory":std::env::var("LAB_RADIANT_MODEL").unwrap(), "dire_directory":std::env::var("LAB_DIRE_MODEL").unwrap()});
    std::fs::write(
        directory.join("ensemble.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    eprintln!(
        "lab-ensemble output={} manifest={manifest}",
        directory.display()
    );
}

fn parity_worlds(seat: usize) -> Vec<EvaluationWorld> {
    (0..4)
        .map(|index| EvaluationWorld {
            environment: build_environment(
                derive_training_seed(2026092811, index, crate::randomization::ARENA_DOMAIN),
                derive_training_seed(2026092811, index, crate::randomization::OPPONENT_DOMAIN),
                MapId(2),
                seat,
                0,
                OpponentSpec::Teacher,
            )
            .unwrap(),
            actions: [0; 16],
            reward: Default::default(),
            action_trace: crate::model::FNV_OFFSET,
        })
        .collect()
}

fn policy_parity_check() {
    let ensemble =
        EvaluationPolicy::load(Path::new(WIDE_INITIAL), PolicyDevice::Cuda { ordinal: 0 });
    assert!(
        ensemble.dire.is_some(),
        "policy-check requires side-experts"
    );
    let before = ensemble.parameter_hashes();
    let sampled = match std::env::var("LAB_POLICY")
        .unwrap_or_else(|_| "greedy".into())
        .as_str()
    {
        "greedy" => false,
        "sampled" => true,
        _ => panic!("invalid LAB_POLICY"),
    };
    for seat in 0..2 {
        let mut routed = parity_worlds(seat);
        let route =
            learning_policy::observed_team(&routed[0].environment.seats[seat].tracker).unwrap();
        let selected = if route == 0 {
            &ensemble.radiant
        } else {
            ensemble.dire.as_ref().unwrap()
        };
        let reference = PolicyModel::fresh_on(1, PolicyDevice::Cuda { ordinal: 0 }).unwrap();
        reference
            .import_parameters(&selected.export_parameters().unwrap())
            .unwrap();
        let reference = EvaluationPolicy {
            radiant: reference,
            dire: None,
            identity: "parity-reference".into(),
            experts: None,
            description: "parity-reference".into(),
        };
        let mut standalone = parity_worlds(seat);
        let mut routed_rng: Vec<_> = (0..4).map(|index| PpoRng::new(719 + index)).collect();
        let mut standalone_rng = routed_rng.clone();
        let actual = eval_rounds(&ensemble, &mut routed, &mut routed_rng, sampled, None);
        let expected = eval_rounds(
            &reference,
            &mut standalone,
            &mut standalone_rng,
            sampled,
            None,
        );
        assert_eq!(actual, expected);
        assert_eq!(routed_rng, standalone_rng);
        for (actual, expected) in routed.iter().zip(&standalone) {
            assert_eq!(actual.actions, expected.actions);
            assert_eq!(actual.action_trace, expected.action_trace);
            assert_eq!(actual.reward, expected.reward);
        }
        eprintln!(
            "lab-policy-parity games=4 observed_route={route} sampled={sampled} exact_actions_rng_reports=true model_id={}",
            ensemble.identity
        );
    }
    assert_eq!(before, ensemble.parameter_hashes());
}

fn validate_final_declaration(policy: &EvaluationPolicy, offset: u64, mode: &str) {
    let path = PathBuf::from(
        std::env::var_os("LAB_FINAL_MANIFEST")
            .expect("final requires preselected immutable manifest"),
    );
    assert!(std::fs::metadata(&path).unwrap().len() <= 8192);
    let declaration: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    learning_policy::validate_declaration(
        &declaration,
        &policy.identity,
        policy
            .experts
            .as_ref()
            .map(|(radiant, dire)| (radiant.as_str(), dire.as_str())),
        offset,
        mode,
    )
    .expect("declared policy identity");
}

fn calibration_reference(set: &str, offset: u64, mode: &str) -> Option<EvaluationPolicy> {
    if !std::env::var("LAB_MODE").is_ok_and(|mode| mode == "sharpen-check") {
        return None;
    }
    assert_eq!(set, "autonomous-validation");
    assert_eq!(offset, 0);
    assert_eq!(mode, "greedy");
    let root = Path::new(WIDE_INITIAL)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("a8-side-experts-inputs");
    Some(EvaluationPolicy {
        radiant: load_evaluation_model(
            &root.join("radiant-carrier-u30"),
            PolicyDevice::Cuda { ordinal: 0 },
        ),
        dire: Some(load_evaluation_model(
            &root.join("dire-entropy003-u15"),
            PolicyDevice::Cuda { ordinal: 0 },
        )),
        identity: "immutable-sharpen-parent-reference".into(),
        experts: None,
        description: "immutable-sharpen-parent-reference".into(),
    })
}

fn checked_evaluation_actions(
    model: &EvaluationPolicy,
    reference: Option<&EvaluationPolicy>,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    random: &mut [PpoRng],
    routes: &[usize],
    sampled: bool,
) -> Vec<StructuredAction> {
    let Some(reference) = reference else {
        return model
            .actions(frames, spaces, random, routes, sampled)
            .expect("strict ensemble batch failed before dispatch or RNG commit");
    };
    assert!(!sampled);
    assert_eq!(frames.len(), routes.len());
    let mut choices = Vec::with_capacity(2);
    for (actual, expected) in [
        (&model.radiant, &reference.radiant),
        (
            model.dire.as_ref().unwrap(),
            reference.dire.as_ref().unwrap(),
        ),
    ] {
        let actual = actual.choose_batch(frames, spaces).unwrap();
        let expected = expected.choose_batch(frames, spaces).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                actual.action, expected.action,
                "sharpen changed greedy action"
            );
            assert_eq!(
                actual.value.to_bits(),
                expected.value.to_bits(),
                "sharpen changed critic bits"
            );
        }
        choices.push(actual);
    }
    routes
        .iter()
        .enumerate()
        .map(|(index, &route)| choices[route][index].action)
        .collect()
}

#[test]
fn calibration_comparison_rejects_changed_actions_and_critic_bits_without_rng_commit() {
    for value_only in [false, true] {
        let mut worlds = parity_worlds(0);
        let (frame, space) = prepare_policy_sample(&mut worlds[0].environment).unwrap();
        let reference = EvaluationPolicy {
            radiant: routing_fixture_model(0),
            dire: Some(routing_fixture_model(0)),
            identity: "reference".into(),
            experts: None,
            description: "reference".into(),
        };
        let model = EvaluationPolicy {
            radiant: routing_fixture_model(usize::from(!value_only)),
            dire: Some(routing_fixture_model(0)),
            identity: "changed".into(),
            experts: None,
            description: "changed".into(),
        };
        if value_only {
            let mut parameters = model.radiant.export_parameters().unwrap();
            let mut offset = 0;
            for (name, shape) in model.radiant.parameter_schema().unwrap() {
                if name == "value.bias" {
                    parameters[offset] += 1.0;
                }
                offset += shape.iter().product::<usize>();
            }
            model.radiant.import_parameters(&parameters).unwrap();
        }
        let mut random = vec![PpoRng::new(441)];
        let before = random.clone();
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checked_evaluation_actions(
                &model,
                Some(&reference),
                &[frame],
                &[space],
                &mut random,
                &[0],
                false,
            )
        }))
        .unwrap_err();
        let message = failure.downcast_ref::<String>().unwrap();
        assert!(message.contains(if value_only {
            "sharpen changed critic bits"
        } else {
            "sharpen changed greedy action"
        }));
        assert_eq!(random, before);
    }
}

fn evaluate() {
    let weights = PathBuf::from(std::env::var_os("LAB_WEIGHTS").unwrap_or_else(|| {
        if wave_two() || learning_wave() {
            WIDE_INITIAL.into()
        } else {
            INITIAL.into()
        }
    }));
    let mode = std::env::var("LAB_POLICY").unwrap_or_else(|_| "greedy".into());
    assert!(matches!(mode.as_str(), "greedy" | "sampled"));
    let set = std::env::var("LAB_SET").unwrap_or_else(|_| {
        if learning_wave() {
            "autonomous-validation"
        } else {
            "selection"
        }
        .into()
    });
    let offset: u64 = std::env::var("LAB_PAIR_OFFSET")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let seed = evaluation_seed(&set, offset);
    let model = EvaluationPolicy::load(&weights, PolicyDevice::Cuda { ordinal: 0 });
    if set == "autonomous-final" {
        validate_final_declaration(&model, offset, &mode);
    }
    let before = model.parameter_hashes();
    let reference = calibration_reference(&set, offset, &mode);
    let reference_before = reference.as_ref().map(EvaluationPolicy::parameter_hashes);
    let mut worlds = (0..40)
        .map(|index| {
            let pair = offset + index as u64 / 2;
            EvaluationWorld {
                environment: build_environment(
                    derive_training_seed(seed, pair, crate::randomization::ARENA_DOMAIN),
                    derive_training_seed(seed, pair, crate::randomization::OPPONENT_DOMAIN),
                    MapId(2),
                    index % 2,
                    0,
                    OpponentSpec::Teacher,
                )
                .unwrap(),
                actions: [0; 16],
                reward: Default::default(),
                action_trace: crate::model::FNV_OFFSET,
            }
        })
        .collect::<Vec<_>>();
    let mut random = (0..40)
        .map(|index| PpoRng::new(derive_training_seed(seed, offset * 2 + index, 0x6576_616c)))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let outcomes = eval_rounds(
        &model,
        &mut worlds,
        &mut random,
        mode == "sampled",
        reference.as_ref(),
    );
    let mut counts = [[0u64; 3]; 2];
    for (index, (outcome, tick)) in outcomes.into_iter().enumerate() {
        let slot = match outcome {
            PpoTerminalOutcome::Win => 0,
            PpoTerminalOutcome::Loss => 1,
            PpoTerminalOutcome::Draw => 2,
        };
        counts[index % 2][slot] += 1;
        eprintln!(
            "lab-game set={set} policy={mode} pair={} seat={} outcome={outcome:?} tick={tick} actions={:?} model_id={} {}",
            offset + index as u64 / 2,
            index % 2,
            worlds[index].actions,
            model.identity,
            worlds[index].reward
        );
    }
    assert_eq!(before, model.parameter_hashes());
    assert_eq!(
        reference_before,
        reference.as_ref().map(EvaluationPolicy::parameter_hashes)
    );
    if reference.is_some() {
        eprintln!(
            "lab-sharpen-parity games=40 greedy_actions_all_steps=true critic_value_bits_all_steps=true parameters_unchanged=true"
        );
    }
    eprintln!(
        "lab-evaluation set={set} policy={mode} offset={offset} games=40 radiant={:?} dire={:?} wins={} losses={} draws={} seconds={} weights={} model_id={} parameters_unchanged=true",
        counts[0],
        counts[1],
        counts[0][0] + counts[1][0],
        counts[0][1] + counts[1][1],
        counts[0][2] + counts[1][2],
        started.elapsed().as_secs_f64(),
        model.description,
        model.identity
    );
}

fn eval_rounds(
    model: &EvaluationPolicy,
    worlds: &mut [EvaluationWorld],
    random: &mut [PpoRng],
    sampled: bool,
    reference: Option<&EvaluationPolicy>,
) -> Vec<(PpoTerminalOutcome, u32)> {
    let count = worlds.len();
    assert!((1..=40).contains(&count));
    let routes: Vec<_> = worlds
        .iter()
        .map(|world| {
            learning_policy::observed_team(
                &world.environment.seats[world.environment.policy_seat].tracker,
            )
            .expect("observed learner team")
        })
        .collect();
    std::thread::scope(|scope| {
        let workers =
            super::super::parallel::StreamWorkers::spawn(scope, worlds, "lab-eval", eval_worker)
                .unwrap();
        let mut active: Vec<_> = (0..count).collect();
        for &index in &active {
            workers.submit(index, EvalRequest::Prepare).unwrap();
        }
        let mut replies = workers.receive(&active).unwrap();
        let mut outcomes = vec![None; count];
        for _ in 0..ACTOR_DECISIONS {
            if active.is_empty() {
                break;
            }
            let (frames, spaces): (Vec<_>, Vec<_>) = replies
                .into_iter()
                .map(|reply| reply.prepared.unwrap())
                .unzip();
            let mut selected: Vec<_> = active.iter().map(|&index| random[index].clone()).collect();
            let selected_routes: Vec<_> = active.iter().map(|&index| routes[index]).collect();
            let actions = checked_evaluation_actions(
                model,
                reference,
                &frames,
                &spaces,
                &mut selected,
                &selected_routes,
                sampled,
            );
            for (&index, rng) in active.iter().zip(selected) {
                random[index] = rng;
            }
            for ((&index, action), space) in active.iter().zip(actions).zip(spaces) {
                workers
                    .submit(index, EvalRequest::Act(action, Box::new(space)))
                    .unwrap();
            }
            replies = workers.receive(&active).unwrap();
            let mut next = Vec::new();
            let mut remaining = Vec::new();
            for (index, reply) in active.into_iter().zip(replies) {
                if let Some(outcome) = reply.outcome {
                    outcomes[index] = Some((outcome, reply.tick));
                } else {
                    next.push(index);
                    remaining.push(reply);
                }
            }
            active = next;
            replies = remaining;
        }
        workers.finish().unwrap();
        assert!(active.is_empty());
        outcomes.into_iter().map(Option::unwrap).collect()
    })
}

#[test]
fn learning_lab_profiles_validate_real_annealed_wave_and_memory_admission() {
    for profile in ["baseline", "g1-b40-m256-reuse", "g2-b20-m256-reuse"] {
        let (parallel_worlds, execution) = learning_execution(profile);
        let settings = AnnealedJobConfig {
            environment_schedule: crate::EnvironmentSchedule::Fixed,
            execution,
            updates: 128,
            games_per_update: 40,
            parallel_worlds,
            games_per_generation: 200,
            zero_updates: 128,
            seed: TRAIN_SEED,
            opponent: AnnealedOpponent::Teacher,
            ppo: crate::PpoConfig {
                sample_budget: crate::PpoSampleBudget::Annealed,
                environments: 40,
                rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
                gamma_tick: 1.0,
                ..Default::default()
            },
            checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
            invocation_updates: None,
            git_commit: "test-profile".into(),
            simulator_commit: "test-simulator".into(),
        };
        validate_annealed(&settings, AnnealedHarness::default())
            .expect("admitted collector profile");
        assert_eq!(parallel_worlds * execution.actor_pipeline_groups, 40);
        let run = annealed_run(
            &settings,
            PolicyDevice::Cpu,
            settings.ppo,
            AnnealedHarness::default(),
            None,
        )
        .unwrap();
        if profile.starts_with("g2") {
            assert!(run.command_line.contains("--actor-pipeline-groups 2"));
        }
        if profile != "baseline" {
            assert!(run.command_line.contains("--training-microbatch 256"));
            assert!(run.command_line.contains("--reuse-actor-values"));
        }
    }
}

#[test]
fn learning_lab_final_chunks_cover_one_hundred_unique_paired_worlds() {
    let mut pairs = std::collections::BTreeSet::new();
    for offset in [0, 20, 40, 60, 80] {
        assert_eq!(evaluation_seed("autonomous-final", offset), 2026092803);
        for pair in offset..offset + 20 {
            assert!(pairs.insert(pair));
        }
    }
    assert_eq!(pairs.len(), 100);
    assert_ne!(
        evaluation_seed("autonomous-validation", 0),
        evaluation_seed("autonomous-final", 0)
    );
    assert!(std::panic::catch_unwind(|| evaluation_seed("autonomous-final", 1)).is_err());
    assert!(std::panic::catch_unwind(|| evaluation_seed("autonomous-final", 200)).is_err());
    assert!(std::panic::catch_unwind(|| learning_execution("unqualified-graph")).is_err());
}
