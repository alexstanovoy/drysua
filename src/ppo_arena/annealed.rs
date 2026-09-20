//! The annealed domain-randomization training loop.
//!
//! One update plays `games_per_update` complete Map2 episodes in batches of
//! `parallel_worlds` worlds, applies one generation's trusted spawn modifiers
//! per `games_per_generation` global games, and runs one PPO optimizer step
//! over every episode's samples. The opponent is frozen for the whole run.
//!
//! Every random choice is a pure function of `(seed, update, game index)`
//! through dedicated streams, so a stopped run resumed from its last
//! committed update replays the interrupted update byte for byte.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use bota_proto::{MapId, ModifierSpec};
use bota_server::game::{
    ModifierDuration, SpawnCategory, SpawnModifier, SpawnSelector, SpawnTarget,
    check_spawn_modifier,
};

use super::episode::{self, ACTOR_DECISIONS};
use super::{
    OpponentSpec, SessionCheckpoint, SmokeCounters, TRAINING_MAX_ENVIRONMENTS,
    TrainingCheckpointSchedule, TrainingDirectoryLock, actor_stream_rngs,
    build_environment_with_spawn_modifiers, device_name, reject_production_rejection,
    restore_strict_session, save_training_artifact, text_error, training_checkpoint_report,
    validate_checkpoint_cadence, validate_initial_weights_directory, validate_training_directory,
};
use crate::randomization::{
    AnnealSchedule, GenerationDraw, RANDOMIZATION_DIRECTORY, derive_training_seed, draw_generation,
    verify_generation_snapshots, write_generation_snapshot,
};
use crate::telemetry::{FlushPerformanceLogs, TrainingTimingScope, time_training_scope};
use crate::{
    CheckpointDevice, CheckpointRun, MAP2_DECISION_INTERVAL_TICKS, MAP2_RETAINED_DECISIONS,
    MAP2_REWARD_GAMMA_TICK, MAX_TRAINING_COUNTER, PPO_MAX_SAMPLES, PPO_RULES_AUDIT_VERSION,
    PolicyDevice, PolicyModel, PolicySnapshot, PpoConfig, PpoError, PpoRng, PpoRollout,
    PpoSmokeReport, PpoTrainer, PpoUpdateReport, SHADOW_FIEND, TrainingArtifact,
    TrainingCheckpointReport, compiled_features,
};

/// The environment ceiling and the retained-decision count bound one update's
/// rollout, so no runtime capacity check is needed.
const _: () = assert!(TRAINING_MAX_ENVIRONMENTS * MAP2_RETAINED_DECISIONS <= PPO_MAX_SAMPLES);

/// Domain separating the per-update balanced side shuffle.
const SEAT_DOMAIN: u64 = 0x7365_6174_5f62_616c;
/// Trainer seed mixing, matching the production training job.
const TRAINER_SALT: u64 = 0x51a9;
/// Actor sampling seed mixing, matching the production training job.
const SAMPLING_SALT: u64 = 0xa17e;

/// Decisions one production annealed episode runs: the full Map2 ceiling.
pub(crate) const ANNEALED_EPISODE_DECISIONS: usize = ACTOR_DECISIONS;

/// The opponent frozen for the whole annealed run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnealedOpponent {
    /// The original scripted teacher.
    Teacher,
    /// A strict runtime weights directory, loaded once and never updated.
    Weights(PathBuf),
}

/// Bounded, resumable annealed-loop settings.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnealedJobConfig {
    /// Total updates in the run.
    pub updates: u64,
    /// Games per update; always even so sides split exactly.
    pub games_per_update: usize,
    /// Worlds advanced in parallel per batch; divides games and generation games.
    pub parallel_worlds: usize,
    /// Games per environment generation, on the global game counter.
    pub games_per_generation: u64,
    /// Final updates played with no modifiers.
    pub zero_updates: u64,
    /// Deterministic run seed.
    pub seed: u64,
    /// Opponent, frozen for the whole run.
    pub opponent: AnnealedOpponent,
    /// PPO dimensions and hyperparameters; `environments` and `rollout_decisions`
    /// are the loop's, not free choices.
    pub ppo: PpoConfig,
    /// When to write a durable checkpoint.
    pub checkpoint_cadence: crate::TrainingCheckpointCadence,
    /// Drysua commit recorded in the run scope.
    pub git_commit: String,
    /// Simulator commit recorded in the run scope.
    pub simulator_commit: String,
}

/// Invocation-only bounds the test harness may set.
///
/// Production never sets any: the CLI always runs the full episode ceiling and
/// stops only on the update budget. The harness is crate-private, so it cannot
/// appear on the command line, and none of its fields shape the run scope
/// except the effective episode ceiling, which is recorded only when a test
/// shortens it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AnnealedHarness {
    /// Decisions per episode; `None` runs the production ceiling.
    pub(crate) episode_decisions: Option<usize>,
    /// Stop after this many completed updates in one invocation.
    pub(crate) stop_after: Option<u64>,
    /// Stop inside an update after this many games, before any optimizer step.
    pub(crate) stop_after_games: Option<usize>,
}

impl AnnealedHarness {
    /// The episode ceiling this invocation runs.
    pub(crate) fn episode_decisions(&self) -> usize {
        self.episode_decisions.unwrap_or(ANNEALED_EPISODE_DECISIONS)
    }
}

/// Final durable and gameplay telemetry of one annealed invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnealedJobReport {
    pub starting_policy_fingerprint: u64,
    pub completed_updates: u64,
    pub optimizer_step: u64,
    pub rollout_samples: u64,
    /// Games collected up to the completed update, resumed runs included;
    /// `updates * games_per_update` over a completed run.
    pub games: u64,
    pub generations: u64,
    pub map2_reward: crate::Map2TrainingReward,
    pub episode_timeouts: u64,
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
    pub elapsed_ticks: u64,
    pub latest: PpoUpdateReport,
}

/// The frozen opponent runtime.
enum AnnealedOpponentRuntime {
    Teacher,
    Weights(Arc<PolicyModel>),
}

impl AnnealedOpponentRuntime {
    fn spec(&self) -> OpponentSpec {
        match self {
            Self::Teacher => OpponentSpec::Teacher,
            Self::Weights(model) => OpponentSpec::SharedPolicy(Arc::clone(model)),
        }
    }
}

/// One loaded opponent and the fingerprint of the weights it was loaded from.
struct LoadedOpponent {
    runtime: AnnealedOpponentRuntime,
    fingerprint: Option<u64>,
}

/// Runs the annealed loop, optionally loading deployment weights first.
pub fn run_annealed_job_on_with_initial_weights<F>(
    settings: AnnealedJobConfig,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
    checkpointed: F,
) -> Result<AnnealedJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport) + Send + 'static,
{
    run_annealed_job_harnessed(
        settings,
        AnnealedHarness::default(),
        device,
        checkpoint_directory,
        resume,
        initial_weights_directory,
        checkpointed,
    )
}

/// Runs the annealed loop under an invocation-only test harness.
pub(crate) fn run_annealed_job_harnessed<F>(
    settings: AnnealedJobConfig,
    harness: AnnealedHarness,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
    checkpointed: F,
) -> Result<AnnealedJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport) + Send + 'static,
{
    let directory = checkpoint_directory.to_path_buf();
    let initial_weights = initial_weights_directory.map(Path::to_path_buf);
    std::thread::Builder::new()
        .name("drysua-annealed".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let _log_flush = FlushPerformanceLogs;
            run_annealed_inner(
                settings,
                harness,
                device,
                &directory,
                resume,
                initial_weights.as_deref(),
                checkpointed,
            )
        })
        .map_err(|error| PpoError::Model(format!("annealed worker spawn failed: {error}")))?
        .join()
        .map_err(|_| PpoError::Model("annealed worker panicked".to_owned()))?
}

fn run_annealed_inner<F>(
    settings: AnnealedJobConfig,
    harness: AnnealedHarness,
    device: PolicyDevice,
    directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
    mut checkpointed: F,
) -> Result<AnnealedJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport),
{
    let config = validate_annealed(&settings, harness)?;
    validate_initial_weights_directory(directory, resume, initial_weights_directory)?;
    if let AnnealedOpponent::Weights(weights) = &settings.opponent
        && weights == directory
    {
        return Err(PpoError::InvalidConfig(
            "annealed opponent weights must not be the checkpoint directory",
        ));
    }
    validate_training_directory(directory, resume)?;
    let _lock = TrainingDirectoryLock::acquire(directory)?;
    let opponent = load_opponent(&settings.opponent, device)?;
    let run = annealed_run(&settings, device, config, harness, opponent.fingerprint)?;
    if resume {
        let stored = TrainingArtifact::load_run_scope(directory).map_err(text_error)?;
        if let Some(difference) = scope_command_line_difference(&stored, &run) {
            return Err(PpoError::ScopeMismatch(difference));
        }
    }
    let random_directory = directory.join(RANDOMIZATION_DIRECTORY);
    if !resume {
        std::fs::create_dir_all(&random_directory)
            .map_err(|error| PpoError::Model(format!("randomization directory: {error}")))?;
    }
    let mut session =
        time_training_scope(TrainingTimingScope::SessionInitialization, None, || {
            AnnealedSession::initialize(
                &settings,
                device,
                directory,
                resume,
                initial_weights_directory,
                config,
                run,
                &random_directory,
                opponent.runtime,
            )
        })?;
    session.run_updates(&settings, harness, config, directory, &mut checkpointed)?;
    Ok(session.report())
}

struct AnnealedSession {
    model: PolicyModel,
    trainer: PpoTrainer,
    sampling: PpoRng,
    opponent: AnnealedOpponentRuntime,
    run: CheckpointRun,
    random_directory: PathBuf,
    generations: u64,
    starting_policy_fingerprint: u64,
    completed_updates: u64,
    rollout_samples: u64,
    games: u64,
    latest: PpoUpdateReport,
    counters: SmokeCounters,
}

impl AnnealedSession {
    #[allow(clippy::too_many_arguments)]
    fn initialize(
        settings: &AnnealedJobConfig,
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        initial_weights_directory: Option<&Path>,
        config: PpoConfig,
        run: CheckpointRun,
        random_directory: &Path,
        opponent: AnnealedOpponentRuntime,
    ) -> Result<Self, PpoError> {
        let model = PolicyModel::fresh_on(settings.seed, device).map_err(text_error)?;
        if !resume && let Some(initial_weights_directory) = initial_weights_directory {
            TrainingArtifact::load_runtime_weights(&model, initial_weights_directory)
                .map_err(text_error)?;
        }
        let (trainer, sampling, completed_updates, rollout_samples, generations) = if resume {
            let parts = restore_strict_session(&model, directory, &run, config)?;
            std::fs::metadata(random_directory).map_err(|_| {
                PpoError::InvalidConfig("domain randomization snapshots are missing on resume")
            })?;
            let completed_updates = parts.progress.global_update;
            let rollout_samples = parts.progress.rollout_samples;
            let completed_games = completed_updates
                .checked_mul(settings.games_per_update as u64)
                .ok_or(PpoError::CounterOverflow)?;
            let schedule = anneal_schedule(settings);
            let generations = verify_generation_snapshots(
                random_directory,
                settings.seed,
                settings.games_per_generation,
                settings.games_per_update as u64,
                schedule,
                completed_games,
            )?;
            (
                parts.trainer,
                parts.sampling,
                completed_updates,
                rollout_samples,
                generations,
            )
        } else {
            let trainer = PpoTrainer::new(&model, config, settings.seed ^ TRAINER_SALT)?;
            let sampling = PpoRng::new(settings.seed ^ SAMPLING_SALT);
            (trainer, sampling, 0, 0, 0)
        };
        let starting_policy_fingerprint = PolicySnapshot::capture(&model, completed_updates)
            .map_err(text_error)?
            .fingerprint();
        let games = completed_updates
            .checked_mul(settings.games_per_update as u64)
            .ok_or(PpoError::CounterOverflow)?;
        Ok(Self {
            model,
            trainer,
            sampling,
            opponent,
            run,
            random_directory: random_directory.to_path_buf(),
            generations,
            starting_policy_fingerprint,
            completed_updates,
            rollout_samples,
            games,
            latest: PpoUpdateReport::default(),
            counters: SmokeCounters::default(),
        })
    }

    fn run_updates(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        config: PpoConfig,
        directory: &Path,
        checkpointed: &mut impl FnMut(TrainingCheckpointReport),
    ) -> Result<(), PpoError> {
        let started = Instant::now();
        let mut schedule = TrainingCheckpointSchedule::new(settings.checkpoint_cadence)?;
        let anneal = anneal_schedule(settings);
        let mut generations = GenerationCache::new(
            self.random_directory.clone(),
            settings.seed,
            settings.games_per_generation,
            settings.games_per_update as u64,
            anneal,
            self.generations,
        );
        while self.completed_updates < settings.updates {
            self.train_update(settings, harness, config, &mut generations)?;
            let final_update = self.completed_updates == settings.updates;
            if schedule.is_due(self.completed_updates, started.elapsed()) || final_update {
                let report = self.checkpoint_report(None);
                let durable =
                    crate::telemetry::time_training_checkpoint(self.completed_updates, || {
                        self.save(directory, report)
                    })?;
                checkpointed(durable);
                schedule.mark_committed(started.elapsed())?;
            }
            if harness
                .stop_after
                .is_some_and(|stop| self.completed_updates >= stop)
            {
                break;
            }
        }
        self.generations = generations.counted_through();
        Ok(())
    }

    fn train_update(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        config: PpoConfig,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        let update = self.completed_updates;
        let games = settings.games_per_update;
        let capacity = games
            .checked_mul(MAP2_RETAINED_DECISIONS)
            .ok_or(PpoError::CounterOverflow)?;
        let policy_identity = self.model.policy_identity().map_err(text_error)?;
        let mut rollout = PpoRollout::new(capacity, policy_identity)?;
        let mut report = PpoSmokeReport::default();
        let seats = balanced_policy_seats(settings.seed, update, games)?;
        let mut local = 0usize;
        while local < games {
            let global_game = update
                .checked_mul(games as u64)
                .and_then(|base| base.checked_add(local as u64))
                .ok_or(PpoError::CounterOverflow)?;
            let generation = global_game / settings.games_per_generation;
            let generation_end = generation
                .checked_add(1)
                .and_then(|next| next.checked_mul(settings.games_per_generation))
                .ok_or(PpoError::CounterOverflow)?;
            let draw = generations.draw(generation)?;
            let rules = generation_rules(&draw, global_game);
            let batch_len = settings.parallel_worlds;
            assert!(local + batch_len <= games, "batch stays inside the update");
            assert!(
                global_game + batch_len as u64 <= generation_end,
                "batch stays inside one generation"
            );
            let mut environments = Vec::with_capacity(batch_len);
            for offset in 0..batch_len {
                let game = global_game + offset as u64;
                let seed =
                    derive_training_seed(settings.seed, game, crate::randomization::ARENA_DOMAIN);
                let opponent_seed = derive_training_seed(
                    settings.seed,
                    game,
                    crate::randomization::OPPONENT_DOMAIN,
                );
                environments.push(build_environment_with_spawn_modifiers(
                    seed,
                    opponent_seed,
                    MapId(2),
                    seats[local + offset],
                    0,
                    self.opponent.spec(),
                    rules.clone(),
                )?);
            }
            let mut streams = (0..batch_len)
                .map(|offset| episode::game_stream(settings.seed, global_game + offset as u64))
                .collect::<Result<Vec<_>, _>>()?;
            let mut random = actor_stream_rngs(&mut self.sampling, batch_len)?;
            episode::collect_batch(
                &self.model,
                config,
                local,
                "annealed",
                &mut environments,
                &mut streams,
                &mut random,
                harness.episode_decisions(),
                &mut rollout,
                &mut report,
            )?;
            for environment in &environments {
                reject_production_rejection(environment, "annealed collection")?;
            }
            self.games = self
                .games
                .checked_add(batch_len as u64)
                .ok_or(PpoError::CounterOverflow)?;
            local += batch_len;
            if harness.stop_after_games.is_some_and(|stop| local >= stop) {
                return Err(PpoError::InvalidTransition(
                    "annealed invocation stopped mid-update",
                ));
            }
        }
        let samples = rollout.len();
        let batch = rollout.finish(config)?;
        let update_report = self.trainer.train_update(&self.model, &batch)?;
        self.latest = update_report;
        self.completed_updates = self.trainer.updates();
        self.rollout_samples = self
            .rollout_samples
            .checked_add(samples as u64)
            .ok_or(PpoError::CounterOverflow)?;
        self.counters.merge(&report)?;
        Ok(())
    }

    fn save(
        &self,
        directory: &Path,
        report: TrainingCheckpointReport,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        save_training_artifact(
            SessionCheckpoint {
                model: &self.model,
                trainer: &self.trainer,
                run: &self.run,
                sampling: &self.sampling,
                completed_updates: self.completed_updates,
                rollout_samples: self.rollout_samples,
                mastery: None,
            },
            directory,
            report,
        )
    }

    fn checkpoint_report(&self, cleanup_warning: Option<String>) -> TrainingCheckpointReport {
        training_checkpoint_report(
            &self.counters,
            self.completed_updates,
            self.trainer.optimizer_step(),
            self.rollout_samples,
            self.latest,
            cleanup_warning,
        )
    }

    fn report(&self) -> AnnealedJobReport {
        AnnealedJobReport {
            starting_policy_fingerprint: self.starting_policy_fingerprint,
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            games: self.games,
            generations: self.generations,
            map2_reward: self.counters.map2_reward,
            episode_timeouts: self.counters.episode_timeouts,
            terminal_wins: self.counters.terminal_wins,
            terminal_losses: self.counters.terminal_losses,
            terminal_draws: self.counters.terminal_draws,
            elapsed_ticks: self.counters.elapsed_ticks,
            latest: self.latest,
        }
    }
}

/// The rules one generation's draw puts on a batch starting at `global_game`.
///
/// A generation that crosses the zero window carries rules only for the games
/// it still covers; a fully truncated generation carries none.
fn generation_rules(draw: &GenerationDraw, global_game: u64) -> Vec<SpawnModifier> {
    let applied_through = draw.start_game.saturating_add(draw.applied_games);
    if draw.applies() && global_game < applied_through {
        spawn_modifiers_for(draw.spec)
    } else {
        Vec::new()
    }
}

/// The trusted spawn rules one generation's spec turns into.
///
/// Target sets, all for the whole match:
/// - max HP: heroes, lane creeps, neutral creeps and structures;
/// - damage amplification, magic resistance, status resistance and movement
///   speed: heroes, lane creeps and neutral creeps;
/// - max mana, mana cost, cooldown and gold income: heroes.
///
/// Structures take only the max HP rule: their blows are not amplified and
/// they take no resistance deltas, matching the transport's documented scope.
///
/// A rule whose fields are all neutral is left out, so a nominal spec yields
/// no rules at all and a zero-temperature update never touches a world.
fn spawn_modifiers_for(spec: ModifierSpec) -> Vec<SpawnModifier> {
    if spec.is_nominal() {
        return Vec::new();
    }
    let mut rules = Vec::with_capacity(3);
    push_spawn_rule(
        &mut rules,
        &[
            SpawnTarget::Category(SpawnCategory::Hero),
            SpawnTarget::Category(SpawnCategory::LaneCreep),
            SpawnTarget::Category(SpawnCategory::NeutralCreep),
            SpawnTarget::Category(SpawnCategory::Structure),
        ],
        ModifierSpec {
            max_hp: spec.max_hp,
            ..ModifierSpec::NOMINAL
        },
    );
    push_spawn_rule(
        &mut rules,
        &[
            SpawnTarget::Category(SpawnCategory::Hero),
            SpawnTarget::Category(SpawnCategory::LaneCreep),
            SpawnTarget::Category(SpawnCategory::NeutralCreep),
        ],
        ModifierSpec {
            physical_damage: spec.physical_damage,
            magic_damage: spec.magic_damage,
            pure_damage: spec.pure_damage,
            magic_resist: spec.magic_resist,
            status_resist: spec.status_resist,
            move_speed: spec.move_speed,
            ..ModifierSpec::NOMINAL
        },
    );
    push_spawn_rule(
        &mut rules,
        &[SpawnTarget::Category(SpawnCategory::Hero)],
        ModifierSpec {
            max_mana: spec.max_mana,
            mana_cost_rate: spec.mana_cost_rate,
            cooldown_rate: spec.cooldown_rate,
            gold_income: spec.gold_income,
            ..ModifierSpec::NOMINAL
        },
    );
    rules
}

/// Adds one rule when its spec changes anything, refusing an invalid one.
fn push_spawn_rule(rules: &mut Vec<SpawnModifier>, targets: &[SpawnTarget], spec: ModifierSpec) {
    if spec.is_nominal() {
        return;
    }
    let rule = SpawnModifier {
        select: SpawnSelector {
            team: None,
            targets: targets.to_vec(),
        },
        spec,
        duration: ModifierDuration::MatchLong,
    };
    assert!(
        check_spawn_modifier(&rule).is_ok(),
        "a generated spawn rule is bounded"
    );
    rules.push(rule);
}

/// One generation's draw, cached between the games that share it.
struct GenerationCache {
    directory: PathBuf,
    seed: u64,
    games_per_generation: u64,
    games_per_update: u64,
    schedule: AnnealSchedule,
    last: Option<GenerationDraw>,
    /// One past the highest generation whose snapshot is recorded.
    counted_through: u64,
}

impl GenerationCache {
    fn new(
        directory: PathBuf,
        seed: u64,
        games_per_generation: u64,
        games_per_update: u64,
        schedule: AnnealSchedule,
        verified: u64,
    ) -> Self {
        Self {
            directory,
            seed,
            games_per_generation,
            games_per_update,
            schedule,
            last: None,
            counted_through: verified,
        }
    }

    /// The draw for one generation, written and verified on first use.
    fn draw(&mut self, generation: u64) -> Result<GenerationDraw, PpoError> {
        if self
            .last
            .as_ref()
            .is_none_or(|draw| draw.generation != generation)
        {
            let draw = draw_generation(
                self.seed,
                generation,
                self.games_per_generation,
                self.games_per_update,
                self.schedule,
            )?;
            write_generation_snapshot(&self.directory, &draw)?;
            let next = generation.checked_add(1).ok_or(PpoError::CounterOverflow)?;
            self.counted_through = self.counted_through.max(next);
            // One line per generation, so a long run's active scale and rules
            // are visible while it runs and not only in the snapshot files.
            eprintln!(
                "annealed: generation={generation} scale_bp={} start_game={} applied_games={} rules={}",
                draw.scale_bp,
                draw.start_game,
                draw.applied_games,
                spawn_modifiers_for(draw.spec).len(),
            );
            self.last = Some(draw);
        }
        Ok(self.last.expect("drawn above"))
    }

    fn counted_through(&self) -> u64 {
        self.counted_through
    }
}

/// One balanced seat assignment per update: exactly half of the games per side.
fn balanced_policy_seats(seed: u64, update: u64, games: usize) -> Result<Vec<usize>, PpoError> {
    assert!(games.is_multiple_of(2), "sides need an even game count");
    let mut seats = (0..games).map(|index| index % 2).collect::<Vec<_>>();
    let mut rng = PpoRng::new(derive_training_seed(seed, update, SEAT_DOMAIN));
    rng.shuffle(&mut seats)?;
    assert_eq!(seats.iter().filter(|seat| **seat == 0).count(), games / 2);
    assert_eq!(seats.iter().filter(|seat| **seat == 1).count(), games / 2);
    Ok(seats)
}

fn anneal_schedule(settings: &AnnealedJobConfig) -> AnnealSchedule {
    AnnealSchedule {
        updates: settings.updates,
        zero_updates: settings.zero_updates,
    }
}

fn load_opponent(
    opponent: &AnnealedOpponent,
    device: PolicyDevice,
) -> Result<LoadedOpponent, PpoError> {
    match opponent {
        AnnealedOpponent::Teacher => Ok(LoadedOpponent {
            runtime: AnnealedOpponentRuntime::Teacher,
            fingerprint: None,
        }),
        AnnealedOpponent::Weights(directory) => {
            let model = PolicyModel::fresh_on(0, device).map_err(text_error)?;
            TrainingArtifact::load_runtime_weights(&model, directory).map_err(text_error)?;
            let fingerprint = PolicySnapshot::capture(&model, 0)
                .map_err(text_error)?
                .fingerprint();
            Ok(LoadedOpponent {
                runtime: AnnealedOpponentRuntime::Weights(Arc::new(model)),
                fingerprint: Some(fingerprint),
            })
        }
    }
}

/// The canonical command line recorded as the run scope.
///
/// Resume compares it byte for byte, so every loop parameter that shapes the
/// run is part of it and a mismatched parameter rejects before any game. The
/// frozen weights fingerprint pins the tensors, not only the directory path.
fn annealed_run(
    settings: &AnnealedJobConfig,
    device: PolicyDevice,
    config: PpoConfig,
    harness: AnnealedHarness,
    opponent_fingerprint: Option<u64>,
) -> Result<CheckpointRun, PpoError> {
    let device_name = device_name(device);
    let mut command_line = format!(
        "train-annealed --updates {} --games {} --parallel {} --generation-games {} --zero-updates {} --epochs {} --minibatch {} --seed {} --map 2 --device {device_name}",
        settings.updates,
        settings.games_per_update,
        settings.parallel_worlds,
        settings.games_per_generation,
        settings.zero_updates,
        config.epochs,
        config.minibatch,
        settings.seed,
    );
    if harness.episode_decisions() != ACTOR_DECISIONS {
        command_line.push_str(&format!(
            " --episode-decisions {}",
            harness.episode_decisions()
        ));
    }
    match &settings.opponent {
        AnnealedOpponent::Teacher => command_line.push_str(" --opponent teacher"),
        AnnealedOpponent::Weights(directory) => {
            command_line.push_str(&format!(" --opponent-weights {}", directory.display()));
            let fingerprint = opponent_fingerprint
                .ok_or(PpoError::InvalidConfig("annealed opponent fingerprint"))?;
            command_line.push_str(&format!(" --opponent-fingerprint {fingerprint:016x}"));
        }
    }
    let defaults = PpoConfig::default();
    if config.learning_rate != defaults.learning_rate {
        command_line.push_str(&format!(" --learning-rate {}", config.learning_rate));
    }
    if config.gae_lambda != defaults.gae_lambda {
        command_line.push_str(&format!(" --gae-lambda {}", config.gae_lambda));
    }
    if config.entropy_coefficient != defaults.entropy_coefficient {
        command_line.push_str(&format!(
            " --entropy-coefficient {}",
            config.entropy_coefficient
        ));
    }
    Ok(CheckpointRun {
        mastery_config: None,
        git_commit: settings.git_commit.clone(),
        simulator_commit: settings.simulator_commit.clone(),
        enabled_features: compiled_features(),
        command_line,
        run_seed: settings.seed,
        map: MapId(2),
        hero: SHADOW_FIEND,
        device: CheckpointDevice::from_policy(device).map_err(text_error)?,
        batch_size: config.minibatch,
        rules_audit_version: PPO_RULES_AUDIT_VERSION,
    })
}

/// Validates every annealed parameter and returns the PPO config it forces.
fn validate_annealed(
    settings: &AnnealedJobConfig,
    harness: AnnealedHarness,
) -> Result<PpoConfig, PpoError> {
    if settings.updates == 0 || settings.updates > MAX_TRAINING_COUNTER {
        return Err(PpoError::InvalidConfig("annealed updates"));
    }
    if settings.games_per_update < 2
        || settings.games_per_update > TRAINING_MAX_ENVIRONMENTS
        || !settings.games_per_update.is_multiple_of(2)
    {
        return Err(PpoError::InvalidConfig(
            "annealed games per update must be even and within the environment ceiling",
        ));
    }
    if settings.parallel_worlds == 0 || settings.parallel_worlds > TRAINING_MAX_ENVIRONMENTS {
        return Err(PpoError::InvalidConfig("annealed parallel worlds"));
    }
    if !settings
        .games_per_update
        .is_multiple_of(settings.parallel_worlds)
    {
        return Err(PpoError::InvalidConfig(
            "annealed parallel worlds must divide games per update",
        ));
    }
    if settings.games_per_generation == 0
        || settings.games_per_generation > MAX_TRAINING_COUNTER
        || !settings
            .games_per_generation
            .is_multiple_of(settings.parallel_worlds as u64)
    {
        return Err(PpoError::InvalidConfig(
            "annealed games per generation must be positive and divisible by parallel worlds",
        ));
    }
    if settings.zero_updates > settings.updates {
        return Err(PpoError::InvalidConfig(
            "annealed zero updates cannot exceed the update budget",
        ));
    }
    if harness.episode_decisions() == 0 || harness.episode_decisions() > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("annealed episode decisions"));
    }
    if settings.git_commit.is_empty() || settings.git_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("annealed git commit"));
    }
    if settings.simulator_commit.is_empty() || settings.simulator_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("annealed simulator commit"));
    }
    validate_checkpoint_cadence(settings.checkpoint_cadence)?;
    if let AnnealedOpponent::Weights(directory) = &settings.opponent
        && directory.as_os_str().is_empty()
    {
        return Err(PpoError::InvalidConfig("annealed opponent weights"));
    }
    let config = settings.ppo;
    if config.environments != settings.games_per_update {
        return Err(PpoError::InvalidConfig(
            "annealed PPO environments must equal games per update",
        ));
    }
    if config.rollout_decisions != MAP2_RETAINED_DECISIONS {
        return Err(PpoError::InvalidConfig(
            "annealed PPO rollout must be the retained decision count",
        ));
    }
    if config.decision_interval_ticks != MAP2_DECISION_INTERVAL_TICKS {
        return Err(PpoError::InvalidConfig(
            "annealed PPO decision interval must be three ticks",
        ));
    }
    if config.gamma_tick != MAP2_REWARD_GAMMA_TICK {
        return Err(PpoError::InvalidConfig(
            "annealed Map2 reward requires gamma per tick one",
        ));
    }
    config.validate()?;
    Ok(config)
}

/// The differing command-line token when two runs differ only there.
///
/// A run that differs in any other scope field is left to the strict loader,
/// which reports the generic `compatibility scope`.
fn scope_command_line_difference(
    stored: &CheckpointRun,
    expected: &CheckpointRun,
) -> Option<String> {
    if stored == expected {
        return None;
    }
    let mut stored_scope = stored.clone();
    stored_scope.command_line.clear();
    let mut expected_scope = expected.clone();
    expected_scope.command_line.clear();
    if stored_scope != expected_scope {
        return None;
    }
    Some(command_line_difference(
        &stored.command_line,
        &expected.command_line,
    ))
}

/// Names the first differing token, preferring the `--flag` it belongs to.
fn command_line_difference(stored: &str, expected: &str) -> String {
    let stored: Vec<&str> = stored.split_whitespace().collect();
    let expected: Vec<&str> = expected.split_whitespace().collect();
    for index in 0..stored.len().max(expected.len()) {
        let left = stored.get(index).copied().unwrap_or("<absent>");
        let right = expected.get(index).copied().unwrap_or("<absent>");
        if left == right {
            continue;
        }
        if right.starts_with("--") {
            let value = command_line_option_value(&expected, index);
            return format!("{right}: recorded <absent>, requested {value}");
        }
        if left.starts_with("--") {
            let value = command_line_option_value(&stored, index);
            return format!("{left}: recorded {value}, requested <absent>");
        }
        if index > 0 {
            let flag = stored[index - 1];
            if flag.starts_with("--") && expected.get(index - 1).copied() == Some(flag) {
                return format!("{flag}: recorded {left}, requested {right}");
            }
        }
        return format!("token {index}: recorded {left}, requested {right}");
    }
    "command lines differ".to_owned()
}

/// The value after one option, or its marker when the option is a switch.
fn command_line_option_value<'a>(tokens: &[&'a str], index: usize) -> &'a str {
    tokens
        .get(index + 1)
        .copied()
        .filter(|token| !token.starts_with("--"))
        .unwrap_or("<present>")
}

#[cfg(test)]
#[path = "../tests/annealed.rs"]
mod tests;
