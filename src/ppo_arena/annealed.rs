//! The annealed domain-randomization training loop with continuous collection.
//!
//! Slots play games back to back on a fixed simulation pool while inference
//! lanes sample their decisions; an update is due once every lane has closed
//! its share of `samples_per_update` retained intervals. Games in flight at an
//! update boundary continue under the next update's actor weights, which lag
//! the learner by [`PIPELINE_STALENESS`] updates so collection never waits for
//! optimization. Each update's spawn modifiers follow the annealing schedule
//! per update. Every random choice is a pure function of the run seed, slot,
//! game ordinal and round sequence, so a stopped run resumed from its last
//! committed update replays in-flight games and continues byte for byte.

use std::path::{Path, PathBuf};
#[path = "annealed_adaptive.rs"]
mod adaptive;
#[path = "annealed_session.rs"]
mod session;

use bota_proto::{MapId, ModifierSpec};
use bota_server::game::{
    ModifierDuration, SpawnCategory, SpawnModifier, SpawnSelector, SpawnTarget,
    check_spawn_modifier,
};

use super::episode::ACTOR_DECISIONS;
use super::opponents::{League, MAX_LEAGUE_SIZE, OpponentSchedule};
use super::slot::{MAX_SLOTS, OpponentKind};
use super::{
    TrainingCheckpointReport, TrainingDirectoryLock, device_name, text_error,
    validate_checkpoint_cadence, validate_initial_weights_directory, validate_training_directory,
};
use crate::randomization::{
    AnnealSchedule, GenerationDraw, RANDOMIZATION_DIRECTORY, draw_generation,
    write_generation_snapshots,
};
use crate::telemetry::{FlushPerformanceLogs, TrainingTimingScope, time_training_scope};
use crate::{
    AnnealedOpponent, CheckpointDevice, CheckpointRun, EnvironmentDecimal,
    MAP2_DECISION_INTERVAL_TICKS, MAP2_REWARD_GAMMA_TICK, MAX_TRAINING_COUNTER,
    MODEL_MAX_OPTIMIZER_STEP, PPO_MAX_POLICY_SAMPLE_DRAWS, PPO_RULES_AUDIT_VERSION, PolicyDevice,
    PolicyModel, PpoConfig, PpoError, PpoUpdateReport, SHADOW_FIEND, TrainingArtifact,
    compiled_features,
};

/// Updates the actor weights of a collected update lag the learner.
pub(crate) const PIPELINE_STALENESS: u64 = 1;
const _: () = assert!(PIPELINE_STALENESS == crate::adaptive_environment::ADAPTIVE_COLLECTION_LAG);
/// Decisions one production annealed episode runs: the full Map2 ceiling.
pub(crate) const ANNEALED_EPISODE_DECISIONS: usize = ACTOR_DECISIONS;
/// Largest retained-interval target of one update.
pub const MAX_SAMPLES_PER_UPDATE: usize = 32_768;
const _: () = assert!(MAX_SAMPLES_PER_UPDATE + 2 * MAX_SLOTS <= crate::PPO_MAX_SAMPLES);
const _: () = assert!(ACTOR_DECISIONS as u64 * PPO_MAX_POLICY_SAMPLE_DRAWS <= MAX_TRAINING_COUNTER);

/// Bounded, resumable annealed-loop settings.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnealedJobConfig {
    /// Adaptive transitions or the fixed per-generation schedule.
    pub environment_schedule: crate::EnvironmentSchedule,
    /// Learner execution choices, bound to the run scope.
    pub execution: crate::TrainingExecutionOptions,
    /// Total updates in the run.
    pub updates: u64,
    /// Concurrent world slots; each always holds a live game.
    pub slots: usize,
    /// Inference lanes; each owns `slots / lanes` slots, a thread and a weight replica.
    pub lanes: usize,
    /// Simulation worker threads; never changes results, so excluded from the run scope.
    pub simulation_threads: usize,
    /// Groups of consecutive lanes with their own share of the simulation
    /// workers, one per cache domain; never changes results.
    pub simulation_groups: usize,
    /// Pins each group's threads to its cache domain; never changes results.
    pub pin_threads: bool,
    /// Updates per environment generation.
    pub generation_updates: u64,
    /// Final updates played with no modifiers.
    pub zero_updates: u64,
    /// Deterministic run seed.
    pub seed: u64,
    /// Environment scale ramp endpoints; `AnnealScale::FULL` is the default ramp.
    pub scale: crate::randomization::AnnealScale,
    /// Per-game opponent mixture with positive weights, in configured order.
    pub opponents: Vec<(AnnealedOpponent, EnvironmentDecimal)>,
    /// How the configured weights become each update's mixture.
    pub opponent_schedule: OpponentSchedule,
    /// League milestones playing at once when the mixture has a league entry.
    pub league_size: usize,
    /// Updates between league snapshots of the learner.
    pub league_every: u64,
    /// Reward 9: shape with a learned win-probability potential refitted this way;
    /// `None` keeps the hand potential (reward 8).
    pub potential: Option<crate::WinModelConfig>,
    /// PPO dimensions and hyperparameters.
    pub ppo: PpoConfig,
    /// Fading imitation and critic warm-up.
    pub guidance: super::TrainingGuidance,
    /// When to write a durable checkpoint.
    pub checkpoint_cadence: crate::TrainingCheckpointCadence,
    /// Additional committed updates in this invocation, capped by `updates`.
    /// Excluded from the run scope; finishing at this boundary forces a
    /// durable checkpoint and runtime export.
    pub invocation_updates: Option<std::num::NonZeroU64>,
    /// Milestone runtime-weight snapshots; excluded from the run scope.
    pub history: Option<crate::RuntimeHistory>,
    /// Drysua commit recorded in the run scope.
    pub git_commit: String,
    /// Simulator commit recorded in the run scope.
    pub simulator_commit: String,
}

/// Invocation-only bounds the test harness may set.
///
/// Production never sets any. The harness is crate-private, so it cannot
/// appear on the command line; only a shortened episode ceiling is recorded
/// in the run scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AnnealedHarness {
    /// Explicit controller-only outcome fixture; never exposed by production entry points.
    #[cfg(test)]
    pub(crate) adaptive_wins: Option<&'static [u64]>,
    /// Decisions per episode; `None` runs the production ceiling.
    pub(crate) episode_decisions: Option<usize>,
    /// Stop after this many completed updates in one invocation.
    pub(crate) stop_after: Option<u64>,
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
    /// Games finished during this invocation's committed updates.
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

/// Configured opponents with their weights, frozen snapshot parameters and
/// fingerprints in mixture order, and the league.
pub(crate) struct OpponentPool {
    /// Configured entries except the league, in configured order.
    pub(crate) entries: Vec<(OpponentKind, u64)>,
    pub(crate) league: Option<League>,
    pub(crate) schedule: OpponentSchedule,
    pub(crate) snapshots: Vec<std::sync::Arc<Vec<f32>>>,
    fingerprints: Vec<u64>,
}

impl OpponentPool {
    /// Configured entries plus the league members playing `update`.
    pub(crate) fn entries_for(&self, update: u64) -> Vec<(OpponentKind, u64)> {
        let mut entries = self.entries.clone();
        if let Some(league) = self.league {
            entries.extend(
                league
                    .members(update)
                    .into_iter()
                    .map(|milestone| (OpponentKind::League(milestone), league.weight)),
            );
        }
        entries
    }
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

/// Read-only CLI admission before the checkpoint directory is locked.
pub(crate) fn preflight_annealed_resume(
    settings: &AnnealedJobConfig,
    device: PolicyDevice,
    directory: &Path,
) -> Result<(), PpoError> {
    let harness = AnnealedHarness::default();
    let config = validate_annealed(settings, harness)?;
    validate_training_directory(directory, true)?;
    let pool = load_opponents(settings)?;
    let run = annealed_run(settings, device, config, harness, &pool)?;
    check_resume_scope(settings, &run, directory)
}

fn check_resume_scope(
    settings: &AnnealedJobConfig,
    run: &CheckpointRun,
    directory: &Path,
) -> Result<(), PpoError> {
    let stored = TrainingArtifact::load_run_scope(directory).map_err(text_error)?;
    if let Some(difference) = scope_command_line_difference(&stored, run) {
        return Err(PpoError::ScopeMismatch(difference));
    }
    adaptive::preflight_resume(settings, run, directory)
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
    if settings
        .opponents
        .iter()
        .any(|(opponent, _)| *opponent == AnnealedOpponent::Weights(directory.to_path_buf()))
    {
        return Err(PpoError::InvalidConfig(
            "annealed opponent weights must not be the checkpoint directory",
        ));
    }
    validate_training_directory(directory, resume)?;
    let pool = load_opponents(&settings)?;
    let run = annealed_run(&settings, device, config, harness, &pool)?;
    if resume {
        check_resume_scope(&settings, &run, directory)?;
    }
    let _lock = TrainingDirectoryLock::acquire(directory)?;
    let random_directory = directory.join(RANDOMIZATION_DIRECTORY);
    if !resume {
        std::fs::create_dir_all(&random_directory)
            .map_err(|error| PpoError::Model(format!("randomization directory: {error}")))?;
    }
    let mut session =
        time_training_scope(TrainingTimingScope::SessionInitialization, None, || {
            session::AnnealedSession::initialize(
                &settings,
                device,
                directory,
                resume,
                initial_weights_directory,
                config,
                run,
                &random_directory,
            )
        })?;
    session.run_updates(
        &settings,
        harness,
        config,
        directory,
        &pool,
        &mut checkpointed,
    )?;
    Ok(session.report())
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
pub(super) fn spawn_modifiers_for(spec: ModifierSpec) -> Vec<SpawnModifier> {
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
    adaptive: Option<crate::AdaptiveEnvironmentCheckpoint>,
    directory: PathBuf,
    seed: u64,
    games_per_generation: u64,
    games_per_update: u64,
    schedule: AnnealSchedule,
    last: Option<GenerationDraw>,
    /// Draws whose snapshots are written with the next checkpoint.
    pending: Vec<GenerationDraw>,
    /// One past the highest generation whose snapshot is recorded.
    counted_through: u64,
}

/// Pending snapshots written early, bounding memory under very long checkpoint intervals.
const MAX_PENDING_GENERATIONS: usize = 1024;

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
            adaptive: None,
            directory,
            seed,
            games_per_generation,
            games_per_update,
            schedule,
            last: None,
            pending: Vec::new(),
            counted_through: verified,
        }
    }

    /// The spawn modifiers of games that start while `update` is collected.
    ///
    /// A generation crossing into the zero window applies only to the updates
    /// it still covers; a fully truncated generation applies to none.
    fn spec_for_update(&mut self, update: u64) -> Result<ModifierSpec, PpoError> {
        let draw = self.draw_for_game(update)?;
        let applied_through = draw.start_game.saturating_add(draw.applied_games);
        Ok(if draw.applies() && update < applied_through {
            draw.spec
        } else {
            ModifierSpec::NOMINAL
        })
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
            self.pending.push(draw);
            if self.pending.len() >= MAX_PENDING_GENERATIONS {
                self.write_pending()?;
            }
            let next = generation.checked_add(1).ok_or(PpoError::CounterOverflow)?;
            self.counted_through = self.counted_through.max(next);
            crate::telemetry::log_line!(
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

    /// Writes the snapshots of every generation drawn since the last call; a
    /// checkpoint calls it before committing so its generations are on disk.
    fn write_pending(&mut self) -> Result<(), PpoError> {
        if !self.pending.is_empty() {
            write_generation_snapshots(&self.directory, &self.pending)?;
            self.pending.clear();
        }
        Ok(())
    }

    fn counted_through(&self) -> u64 {
        self.adaptive
            .map_or(self.counted_through, |state| state.snapshot_count)
    }
}

fn anneal_schedule(settings: &AnnealedJobConfig) -> AnnealSchedule {
    AnnealSchedule {
        updates: settings.updates,
        zero_updates: settings.zero_updates,
        scale: settings.scale,
    }
}

/// Loads every frozen snapshot of the mixture once, on the host.
fn load_opponents(settings: &AnnealedJobConfig) -> Result<OpponentPool, PpoError> {
    let mut entries = Vec::with_capacity(settings.opponents.len());
    let mut league = None;
    let mut snapshots = Vec::new();
    let mut fingerprints = Vec::new();
    for (opponent, weight) in &settings.opponents {
        let kind = match opponent {
            AnnealedOpponent::Teacher => OpponentKind::Teacher,
            AnnealedOpponent::HarassPush => OpponentKind::HarassPush,
            AnnealedOpponent::Styled(kind) => OpponentKind::Styled(*kind),
            AnnealedOpponent::SelfPlay => OpponentKind::SelfPlay,
            AnnealedOpponent::League => {
                league = Some(League {
                    weight: weight.units(),
                    size: settings.league_size,
                    every: settings.league_every,
                });
                continue;
            }
            AnnealedOpponent::Weights(directory) => {
                let model = PolicyModel::fresh_on(0, PolicyDevice::Cpu).map_err(text_error)?;
                TrainingArtifact::load_runtime_weights(&model, directory).map_err(text_error)?;
                fingerprints.push(model.parameter_fingerprint().map_err(text_error)?);
                snapshots.push(std::sync::Arc::new(
                    model.export_parameters().map_err(text_error)?,
                ));
                OpponentKind::Snapshot(snapshots.len() - 1)
            }
        };
        entries.push((kind, weight.units()));
    }
    if entries.is_empty() && league.is_none() {
        return Err(PpoError::InvalidConfig(
            "annealed opponent mixture is empty",
        ));
    }
    Ok(OpponentPool {
        entries,
        league,
        schedule: settings.opponent_schedule,
        snapshots,
        fingerprints,
    })
}

/// The canonical command line recorded as the run scope.
///
/// Resume compares it byte for byte, so every loop parameter that shapes the
/// run is part of it and a mismatched parameter rejects before any game.
/// Frozen weights fingerprints pin the tensors, not only the directory paths.
fn annealed_run(
    settings: &AnnealedJobConfig,
    device: PolicyDevice,
    config: PpoConfig,
    harness: AnnealedHarness,
    pool: &OpponentPool,
) -> Result<CheckpointRun, PpoError> {
    settings.execution.validate()?;
    let device_name = device_name(device);
    let mut command_line = format!(
        "train-annealed --updates {} --samples-per-update {} --slots {} --lanes {} --generation-updates {} --zero-updates {} --epochs {} --minibatch {} --seed {} --map 2 --device {device_name} --pipeline-staleness {PIPELINE_STALENESS}",
        settings.updates,
        config.samples_per_update,
        settings.slots,
        settings.lanes,
        settings.generation_updates,
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
    append_opponent_scope(settings, pool, &mut command_line)?;
    if let Some(potential) = settings.potential {
        command_line.push_str(&format!(
            " --potential learned --win-model-every {} --win-model-games {} --win-model-min-games {}",
            potential.every, potential.games, potential.min_games
        ));
    }
    let defaults = PpoConfig::default();
    if config.learning_rate != defaults.learning_rate {
        command_line.push_str(&format!(" --learning-rate {}", config.learning_rate));
    }
    if config.gae_lambda_tick != defaults.gae_lambda_tick {
        command_line.push_str(&format!(" --gae-lambda-tick {}", config.gae_lambda_tick));
    }
    if config.entropy_coefficient != defaults.entropy_coefficient {
        command_line.push_str(&format!(
            " --entropy-coefficient {}",
            config.entropy_coefficient
        ));
    }
    if config.target_kl != defaults.target_kl {
        command_line.push_str(&format!(" --target-kl {}", config.target_kl));
    }
    if config.value_coefficient != defaults.value_coefficient {
        command_line.push_str(&format!(
            " --value-coefficient {}",
            config.value_coefficient
        ));
    }
    settings.execution.append_scope(&mut command_line);
    settings.guidance.append_scope(&mut command_line);
    adaptive::append_scope(settings, harness, &mut command_line);
    Ok(CheckpointRun {
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

/// Records the opponent mixture in order, pinning frozen snapshots by fingerprint.
fn append_opponent_scope(
    settings: &AnnealedJobConfig,
    pool: &OpponentPool,
    command_line: &mut String,
) -> Result<(), PpoError> {
    let mut fingerprints = pool.fingerprints.iter();
    for (opponent, weight) in &settings.opponents {
        match opponent {
            AnnealedOpponent::Teacher => {
                command_line.push_str(&format!(" --opponent teacher:{weight}"))
            }
            AnnealedOpponent::SelfPlay => {
                command_line.push_str(&format!(" --opponent self:{weight}"))
            }
            AnnealedOpponent::HarassPush => {
                command_line.push_str(&format!(" --opponent harass-push:{weight}"))
            }
            AnnealedOpponent::Styled(kind) => {
                command_line.push_str(&format!(" --opponent {}:{weight}", kind.styled_label()))
            }
            AnnealedOpponent::League => {
                let league = pool
                    .league
                    .ok_or(PpoError::InvalidConfig("annealed league"))?;
                command_line.push_str(&format!(
                    " --opponent league:{weight} --league-size {} --league-every {}",
                    league.size, league.every
                ));
            }
            AnnealedOpponent::Weights(directory) => {
                let fingerprint = fingerprints
                    .next()
                    .ok_or(PpoError::InvalidConfig("annealed opponent fingerprint"))?;
                command_line.push_str(&format!(
                    " --opponent weights:{}:{weight} --opponent-fingerprint {fingerprint:016x}",
                    directory.display()
                ));
            }
        }
    }
    command_line.push_str(&format!(" --opponent-schedule {}", pool.schedule.name()));
    Ok(())
}

/// Validates every annealed parameter.
pub(crate) fn validate_annealed(
    settings: &AnnealedJobConfig,
    harness: AnnealedHarness,
) -> Result<PpoConfig, PpoError> {
    settings.execution.validate()?;
    settings.scale.validate()?;
    settings.guidance.validate()?;
    if settings.updates == 0 || settings.updates > MAX_TRAINING_COUNTER {
        return Err(PpoError::InvalidConfig("annealed updates"));
    }
    if settings
        .invocation_updates
        .is_some_and(|limit| limit.get() > MAX_TRAINING_COUNTER)
    {
        return Err(PpoError::InvalidConfig(
            "annealed invocation updates exceed MAX_TRAINING_COUNTER",
        ));
    }
    validate_collection(settings)?;
    adaptive::validate(settings, harness)?;
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
    validate_opponents(settings)?;
    let config = validate_annealed_ppo(settings)?;
    validate_annealed_counters(settings, config)?;
    Ok(config)
}

fn validate_collection(settings: &AnnealedJobConfig) -> Result<(), PpoError> {
    if !(1..=MAX_SLOTS).contains(&settings.slots) {
        return Err(PpoError::InvalidConfig(
            "annealed slots must be within 1..=256",
        ));
    }
    if settings.lanes == 0 || !settings.slots.is_multiple_of(settings.lanes) {
        return Err(PpoError::InvalidConfig("annealed lanes must divide slots"));
    }
    // A lane batches its slots plus any self-play opponent rows in one call.
    if 2 * (settings.slots / settings.lanes) > crate::MODEL_SAMPLING_BATCH {
        return Err(PpoError::InvalidConfig(
            "annealed lanes hold at most 64 slots each",
        ));
    }
    if !(1..=256).contains(&settings.simulation_threads) {
        return Err(PpoError::InvalidConfig("annealed simulation threads"));
    }
    let groups = settings.simulation_groups;
    if groups == 0 || !settings.lanes.is_multiple_of(groups) || settings.simulation_threads < groups
    {
        return Err(PpoError::InvalidConfig(
            "annealed simulation groups must divide lanes and have a thread each",
        ));
    }
    if settings.generation_updates == 0 || settings.generation_updates > MAX_TRAINING_COUNTER {
        return Err(PpoError::InvalidConfig("annealed generation updates"));
    }
    Ok(())
}

fn validate_opponents(settings: &AnnealedJobConfig) -> Result<(), PpoError> {
    settings
        .potential
        .map_or(Ok(()), crate::WinModelConfig::validate)?;
    if settings.opponents.is_empty() || settings.opponents.len() > 16 {
        return Err(PpoError::InvalidConfig(
            "annealed opponent mixture needs 1..=16 entries",
        ));
    }
    let leagues = settings
        .opponents
        .iter()
        .filter(|(opponent, _)| *opponent == AnnealedOpponent::League)
        .count();
    if leagues > 1
        || !(1..=MAX_LEAGUE_SIZE).contains(&settings.league_size)
        || !(1..=MAX_TRAINING_COUNTER).contains(&settings.league_every)
    {
        return Err(PpoError::InvalidConfig(
            "annealed league: at most one entry of 1..=16 snapshots, taken every 1.. updates",
        ));
    }
    for (opponent, weight) in &settings.opponents {
        if weight.units() == 0 || weight.units() > 1_000 * EnvironmentDecimal::SCALE {
            return Err(PpoError::InvalidConfig("annealed opponent weight"));
        }
        if let AnnealedOpponent::Weights(directory) = opponent
            && directory.as_os_str().is_empty()
        {
            return Err(PpoError::InvalidConfig("annealed opponent weights"));
        }
    }
    Ok(())
}

fn validate_annealed_ppo(settings: &AnnealedJobConfig) -> Result<PpoConfig, PpoError> {
    let config = settings.ppo;
    if !(1..=MAX_SAMPLES_PER_UPDATE).contains(&config.samples_per_update)
        || !config
            .samples_per_update
            .is_multiple_of(settings.lanes.max(1))
    {
        return Err(PpoError::InvalidConfig(
            "annealed samples per update must be a multiple of lanes within 1..=32768",
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
    config.validate()
}

fn validate_annealed_counters(
    settings: &AnnealedJobConfig,
    config: PpoConfig,
) -> Result<(), PpoError> {
    let samples = config.rollout_capacity(settings.slots) as u64;
    let optimizer_steps = samples
        .div_ceil(config.minibatch as u64)
        .checked_mul(config.epochs as u64)
        .ok_or(PpoError::InvalidConfig("annealed optimizer counter"))?;
    validate_counter_budget(
        settings.updates,
        optimizer_steps,
        MODEL_MAX_OPTIMIZER_STEP,
        "annealed optimizer counter",
    )?;
    validate_counter_budget(
        settings.updates,
        samples,
        MAX_TRAINING_COUNTER,
        "annealed sample counter",
    )?;
    // All samples share one shuffle per epoch.
    let shuffle_draws = samples
        .saturating_sub(1)
        .checked_mul(config.epochs as u64)
        .ok_or(PpoError::InvalidConfig("annealed shuffle RNG counter"))?;
    validate_counter_budget(
        settings.updates,
        shuffle_draws,
        MAX_TRAINING_COUNTER,
        "annealed shuffle RNG counter",
    )
}

fn validate_counter_budget(
    count: u64,
    per_count: u64,
    maximum: u64,
    field: &'static str,
) -> Result<(), PpoError> {
    assert!(maximum > 0);
    assert!(!field.is_empty());
    count
        .checked_mul(per_count)
        .filter(|total| *total <= maximum)
        .map(|_| ())
        .ok_or(PpoError::InvalidConfig(field))
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

fn log_ppo_update(
    report: &crate::PpoUpdateReport,
    explained_variance: crate::ExplainedVariance,
    optimizer_steps: u64,
    samples: usize,
    learning_rate: f32,
) {
    crate::telemetry::log_line!(
        "level=INFO event=ppo_update update={} policy_loss={:.6} value_loss={:.6} entropy={:.6} approx_kl={:.8} clip_fraction={:.6} explained_variance={:.6} explained_variance_mc={:.6} kl_stop={} optimizer_steps={optimizer_steps} samples={samples} learning_rate={learning_rate:e} critic_only={}{}",
        report.update,
        report.policy_loss,
        report.value_loss,
        report.entropy,
        crate::ppo_arena::update_kl(*report),
        report.clip_fraction,
        explained_variance.lambda,
        explained_variance.monte_carlo,
        report.stopped_for_kl,
        report.objective.critic_only,
        imitation_fields(report),
    );
}

/// The imitation coefficient, mean cross entropy per labeled row and the
/// agreement of the legal argmax with the labels: whole actions, then per head.
#[allow(
    clippy::float_arithmetic,
    reason = "rates of summed imitation statistics"
)]
fn imitation_fields(report: &crate::PpoUpdateReport) -> String {
    const HEADS: [&str; crate::MODEL_BEHAVIORAL_HEADS] = [
        "kind",
        "unit",
        "ability",
        "item",
        "swap",
        "learn",
        "shop",
        "loot",
        "target_mode",
        "put_mode",
        "entity",
        "point",
    ];
    let imitation = &report.imitation;
    if report.objective.imitation == 0.0 || imitation.labeled == 0.0 {
        return String::new();
    }
    let mut fields = format!(
        " imitation_coefficient={} imitation_loss={:.6} imitation_labeled={} imitation_agree={:.6}",
        report.objective.imitation,
        imitation.cross_entropy / imitation.labeled,
        imitation.labeled,
        imitation.action_agreements / imitation.labeled,
    );
    for (head, name) in HEADS.iter().enumerate() {
        if imitation.head_labels[head] > 0.0 {
            fields.push_str(&format!(
                " imitation_agree_{name}={:.6}",
                imitation.head_agreements[head] / imitation.head_labels[head]
            ));
        }
    }
    fields
}
