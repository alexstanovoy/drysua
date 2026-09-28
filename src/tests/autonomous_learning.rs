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

const ROOT: &str = "/home/alexstanovoy/Workspace/bots/drysua/temp/autonomous-20260927";
const INITIAL: &str = "/home/alexstanovoy/Workspace/bots/drysua/artifacts/temp/annealed-teacher-20260920/attempt-008/history/update-0428";
const TRAIN_SEED: u64 = 20260927;
const WIDE_INITIAL: &str = "/home/alexstanovoy/Workspace/bots/drysua/artifacts/deslop-20260922/learning-experiments/lab-final-artifacts/secondary-credit-u10";

fn wave_two() -> bool {
    std::env::var("LAB_WAVE").is_ok_and(|wave| wave == "2")
}
fn lab_root() -> PathBuf {
    if wave_two() {
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
            match std::env::var("LAB_MODE").unwrap().as_str() {
                "evaluate" => evaluate(),
                "train" => train(),
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

fn train() {
    let arm = std::env::var("LAB_ARM").unwrap();
    let settings = settings(&arm);
    let initial = if wave_two() {
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
    std::fs::create_dir_all(&directory).unwrap();
    let resume = directory.join("checkpoint.meta").exists();
    let random_directory = directory.join(RANDOMIZATION_DIRECTORY);
    std::fs::create_dir_all(&random_directory).unwrap();
    let mut run = annealed_run(
        &settings,
        PolicyDevice::Cuda { ordinal: 0 },
        settings.ppo,
        AnnealedHarness::default(),
        None,
    )
    .unwrap();
    run.command_line
        .push_str(&format!(" --experiment autonomous-20260927 --method {arm}"));
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
    let mut session = AnnealedSession::initialize(
        &settings,
        PolicyDevice::Cuda { ordinal: 0 },
        &directory,
        resume,
        (!resume).then_some(initial.as_path()),
        settings.ppo,
        run,
        &random_directory,
        AnnealedOpponentRuntime::Teacher,
    )
    .unwrap();
    let mut rnd =
        (arm == "wave-rnd").then(|| load_rnd(&directory, session.state.completed_updates));
    if !resume {
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
        for local in (0..settings.games_per_update).step_by(settings.parallel_worlds) {
            session
                .collect_update_batch(
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
        std::fs::write(directory.join("lab_progress.json"), format!("{{\"arm\":\"{arm}\",\"updates\":{completed},\"games\":{},\"optimizer_steps\":{}}}\n", completed*settings.games_per_update as u64,session.state.trainer.optimizer_step())).unwrap();
        if completed.is_multiple_of(10)
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
    use sha2::{Digest, Sha256};
    assert!(std::fs::metadata(path).unwrap().len() <= 65536);
    Sha256::digest(std::fs::read(path).unwrap())
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
    let target = lab_root()
        .join("history")
        .join(arm)
        .join(format!("u{update:04}"));
    assert!(!target.exists());
    std::fs::create_dir_all(target.join(RANDOMIZATION_DIRECTORY)).unwrap();
    for (index, entry) in source.read_dir().unwrap().enumerate() {
        assert!(index < 128);
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == ".training.lock" {
            continue;
        }
        if name == RANDOMIZATION_DIRECTORY {
            for (index, file) in entry.path().read_dir().unwrap().enumerate() {
                assert!(index < 32);
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

fn evaluate() {
    let weights = PathBuf::from(std::env::var_os("LAB_WEIGHTS").unwrap_or_else(|| {
        if wave_two() {
            WIDE_INITIAL.into()
        } else {
            INITIAL.into()
        }
    }));
    let mode = std::env::var("LAB_POLICY").unwrap_or_else(|_| "greedy".into());
    assert!(matches!(mode.as_str(), "greedy" | "sampled"));
    let set = std::env::var("LAB_SET").unwrap_or_else(|_| "selection".into());
    let seed = match set.as_str() {
        "selection" => 20261001,
        "confirmation" => 20261002,
        "wave-selection" => 20261003,
        "wave-confirmation" => 20261004,
        _ => panic!("invalid evaluation set"),
    };
    let offset: u64 = std::env::var("LAB_PAIR_OFFSET")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    assert!(offset == 0 || set.ends_with("confirmation") && offset == 20);
    let model = PolicyModel::fresh_on(1, PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &weights).unwrap();
    let before = model.export_parameters().unwrap();
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
            }
        })
        .collect::<Vec<_>>();
    let mut random = (0..40)
        .map(|index| PpoRng::new(derive_training_seed(seed, offset * 2 + index, 0x6576_616c)))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let outcomes = eval_rounds(&model, &mut worlds, &mut random, mode == "sampled");
    let mut counts = [[0u64; 3]; 2];
    for (index, (outcome, tick)) in outcomes.into_iter().enumerate() {
        let slot = match outcome {
            PpoTerminalOutcome::Win => 0,
            PpoTerminalOutcome::Loss => 1,
            PpoTerminalOutcome::Draw => 2,
        };
        counts[index % 2][slot] += 1;
        eprintln!(
            "lab-game set={set} policy={mode} pair={} seat={} outcome={outcome:?} tick={tick} actions={:?} {}",
            offset + index as u64 / 2,
            index % 2,
            worlds[index].actions,
            worlds[index].reward
        );
    }
    crate::tests::support::assert_bits(&before, &model.export_parameters().unwrap());
    eprintln!(
        "lab-evaluation set={set} policy={mode} offset={offset} games=40 radiant={:?} dire={:?} wins={} losses={} draws={} seconds={} weights={} parameters_unchanged=true",
        counts[0],
        counts[1],
        counts[0][0] + counts[1][0],
        counts[0][1] + counts[1][1],
        counts[0][2] + counts[1][2],
        started.elapsed().as_secs_f64(),
        weights.display()
    );
}

fn eval_rounds(
    model: &PolicyModel,
    worlds: &mut [EvaluationWorld],
    random: &mut [PpoRng],
    sampled: bool,
) -> Vec<(PpoTerminalOutcome, u32)> {
    std::thread::scope(|scope| {
        let workers =
            super::super::parallel::StreamWorkers::spawn(scope, worlds, "lab-eval", eval_worker)
                .unwrap();
        let mut active: Vec<_> = (0..40).collect();
        for &index in &active {
            workers.submit(index, EvalRequest::Prepare).unwrap();
        }
        let mut replies = workers.receive(&active).unwrap();
        let mut outcomes = vec![None; 40];
        for _ in 0..ACTOR_DECISIONS {
            if active.is_empty() {
                break;
            }
            let (frames, spaces): (Vec<_>, Vec<_>) = replies
                .into_iter()
                .map(|reply| reply.prepared.unwrap())
                .unzip();
            let actions: Vec<_> = if sampled {
                let mut selected: Vec<_> =
                    active.iter().map(|&index| random[index].clone()).collect();
                let choices = model.sample_batch(&frames, &spaces, &mut selected).unwrap();
                for (&index, rng) in active.iter().zip(selected) {
                    random[index] = rng;
                }
                choices.into_iter().map(|choice| choice.action()).collect()
            } else {
                model
                    .choose_batch(&frames, &spaces)
                    .unwrap()
                    .into_iter()
                    .map(|choice| choice.action)
                    .collect()
            };
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
