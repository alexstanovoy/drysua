use std::collections::VecDeque;
mod annealed;
pub use annealed::{
    AnnealedJobConfig, AnnealedJobReport, AnnealedOpponent,
    run_annealed_job_on_with_initial_weights,
};
pub(crate) mod episode;
mod parallel;
mod reward;
pub use reward::Map2TrainingReward;
#[cfg(test)]
#[path = "tests/ppo_arena_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;
#[cfg(test)]
#[path = "tests/training_order_contract.rs"]
pub(crate) mod training_order_contract;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{Duration, Instant};

use bota_proto::{EventKind, MapId, RejectReason, ServerMsg, SlotId, Team};
use bota_server::game::SpawnModifier;

use crate::persistence::training::PolicyOrderBookkeeping;
use crate::telemetry::{
    FlushPerformanceLogs, TrainingStage, TrainingTimingScope, TrainingUpdateMode,
    TrainingUpdateTimer, time_training_checkpoint, time_training_scope,
};

use crate::{
    ACTOR_LEARNER_BUFFERS, ActionKind, ActionSpace, ActivePolicyOrder, ActorLearnerPipeline, Arena,
    ArenaConfig, ArenaStart, CheckpointDevice, CheckpointProgress, CheckpointRun,
    CheckpointSaveOutcome, FeatureEncoder, FeatureFrame, ItemReadiness, LocalPolicyState,
    MAX_TRAINING_COUNTER, MODEL_MAX_OPTIMIZER_STEP, OrderPersistence, PPO_MAX_POLICY_SAMPLE_DRAWS,
    PPO_MAX_ROLLOUT_DECISIONS, PPO_RULES_AUDIT_VERSION, PolicyDevice, PolicyModel, PolicySnapshot,
    PpoConfig, PpoError, PpoOutcome, PpoPolicyChoice, PpoRng, PpoRollout, PpoTerminalOutcome,
    PpoTrainer, PpoUpdateReport, Request, RewardTracker, RngCheckpoint, SHADOW_FIEND, StateTracker,
    Teacher, TrainingArtifact, compiled_features, tick_discount,
};
use crate::{MAP2_REWARD_GAMMA_TICK, Map2RewardBreakdown, Map2RewardEnd};

pub(crate) use crate::map2_contract::MAX_TRAINING_ENVIRONMENTS as TRAINING_MAX_ENVIRONMENTS;
const READINESS_ORDER_HISTORY: usize = 32;
const _: () = assert!(PPO_MAX_ROLLOUT_DECISIONS <= crate::PPO_MAX_SAMPLES);

/// Bounded builtin smoke-run settings for the complete actor-to-learner path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PpoSmokeConfig {
    pub updates: u32,
    pub environments: usize,
    pub rollout_decisions: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub seed: u64,
    pub map: MapId,
}

impl Default for PpoSmokeConfig {
    fn default() -> Self {
        Self {
            updates: 1,
            environments: 2,
            rollout_decisions: 8,
            epochs: 1,
            minibatch: 16,
            seed: 9_001,
            map: MapId(2),
        }
    }
}

/// Aggregate result of a short real-simulator PPO run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PpoSmokeReport {
    pub completed_episodes: crate::CompletedTrainingEpisodes,
    pub map2_reward: Map2TrainingReward,
    pub episode_timeouts: u64,
    pub updates: u32,
    pub transitions: usize,
    pub optimizer_step: u64,
    pub final_policy_loss: f64,
    pub final_value_loss: f64,
    pub final_entropy: f64,
    pub final_kl: f64,
    pub rejected_orders: u64,
    pub elapsed_ticks: u64,
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
}

/// Bounded, resumable production PPO settings.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingJobConfig {
    /// Resolved mastery window/thresholds; present only for mastery-v1.
    pub mastery_config: Option<crate::MasteryConfig>,
    /// Full-episode opponent schedule, persisted in the canonical checkpoint run command.
    pub opponent_schedule: crate::TrainingOpponentSchedule,
    /// Legacy override retained for explicit rejection; Map2 requires zero.
    pub episode_time_cost: f32,
    /// Legacy override retained for explicit rejection; Map2 requires comprehensive reward.
    pub terminal_only: bool,
    pub complete_episodes: bool,
    /// Fixed complete-episode collection groups, one barrier each: 1 (default
    /// global barrier), 2, or 4. Recorded in the canonical run scope because
    /// batch composition differs from the one-group contract.
    pub pipeline_groups: usize,
    pub updates: u64,
    /// Single source of truth for rollout dimensions and checkpointed hyperparameters.
    pub ppo: PpoConfig,
    pub checkpoint_cadence: TrainingCheckpointCadence,
    pub resume_provenance: ResumeProvenance,
    pub seed: u64,
    pub map: MapId,
    pub git_commit: String,
    pub simulator_commit: String,
}

/// A deterministic test cadence or a production monotonic wall-time cadence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCheckpointCadence {
    Updates(u64),
    WallTime(Duration),
}

/// Strict resume by default, with an explicit one-time Git provenance migration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResumeProvenance {
    #[default]
    Strict,
    MigrateGitCommit,
}

/// Durable progress plus invocation-local gameplay telemetry emitted after a committed checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingCheckpointReport {
    pub map2_reward: Map2TrainingReward,
    pub episode_timeouts: u64,
    pub completed_updates: u64,
    pub optimizer_step: u64,
    pub rollout_samples: u64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub stopped_for_kl: bool,
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
    pub rejected_orders: u64,
    pub elapsed_ticks: u64,
    pub cleanup_warning: Option<String>,
}

/// Final durable progress and gameplay telemetry from the current bounded invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainingJobReport {
    pub mastery_completed: bool,
    pub map2_reward: Map2TrainingReward,
    pub episode_timeouts: u64,
    pub starting_policy_fingerprint: u64,
    pub completed_updates: u64,
    pub optimizer_step: u64,
    pub rollout_samples: u64,
    pub final_policy_loss: f64,
    pub final_value_loss: f64,
    pub final_entropy: f64,
    pub final_kl: f64,
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
    pub rejected_orders: u64,
    pub elapsed_ticks: u64,
}

/// Independent opponent used by one fixed evaluation cohort.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointEvaluationBaseline {
    Teacher,
    Weak,
}

struct ArenaSeatPolicy {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    order_bookkeeping: PolicyOrderBookkeeping,
    readiness: ItemReadiness,
    teacher: Teacher,
    sequence: u32,
    rejections: u64,
    pending_active: Option<(u32, Option<ActivePolicyOrder>)>,
    readiness_orders: VecDeque<(u32, crate::IssuedOrder, u32)>,
    last_issued: Option<(u32, crate::IssuedOrder, ActionKind)>,
    last_rejection: Option<(u32, RejectReason)>,
}

struct TrainingEnvironment {
    arena: Arena,
    seats: Vec<ArenaSeatPolicy>,
    policy_seat: usize,
    reward: RewardTracker,
    decision: u32,
    map: MapId,
    next_seed: u64,
    next_opponent_seed: u64,
    opponent_spec: OpponentSpec,
    opponent: OpponentRuntime,
    retired_rejections: u64,
}

#[derive(Clone)]
enum OpponentSpec {
    #[cfg(test)]
    Policy(PolicySnapshot),
    SharedPolicy(Arc<PolicyModel>),
    Teacher,
    Weak,
}

enum OpponentRuntime {
    Policy {
        model: Arc<PolicyModel>,
        rng: PpoRng,
    },
    Teacher,
    Weak,
}

struct ArenaAdvance {
    winner: Option<Team>,
    ticks: u32,
}

struct PendingTransition {
    choice: PpoPolicyChoice,
    reward: f32,
    map2_reward: Option<Map2RewardBreakdown>,
    ticks: u32,
    terminal: bool,
    terminal_outcome: Option<PpoTerminalOutcome>,
    next_frame: Option<FeatureFrame>,
}

struct TrainingDirectoryLock {
    _file: File,
}

pub(crate) struct TrainingCheckpointSchedule {
    cadence: TrainingCheckpointCadence,
    next_wall_deadline: Option<Duration>,
}

impl TrainingCheckpointSchedule {
    pub(crate) fn new(cadence: TrainingCheckpointCadence) -> Result<Self, PpoError> {
        validate_checkpoint_cadence(cadence)?;
        let next_wall_deadline = match cadence {
            TrainingCheckpointCadence::Updates(_) => None,
            TrainingCheckpointCadence::WallTime(interval) => Some(interval),
        };
        Ok(Self {
            cadence,
            next_wall_deadline,
        })
    }

    pub(crate) fn is_due(&self, completed_updates: u64, elapsed: Duration) -> bool {
        match self.cadence {
            TrainingCheckpointCadence::Updates(interval) => {
                completed_updates.is_multiple_of(interval)
            }
            TrainingCheckpointCadence::WallTime(_) => self
                .next_wall_deadline
                .is_some_and(|deadline| elapsed >= deadline),
        }
    }

    pub(crate) fn mark_committed(&mut self, elapsed: Duration) -> Result<(), PpoError> {
        let TrainingCheckpointCadence::WallTime(interval) = self.cadence else {
            return Ok(());
        };
        let deadline = self
            .next_wall_deadline
            .ok_or(PpoError::InvalidTransition("wall checkpoint deadline"))?;
        let lag = elapsed.saturating_sub(deadline);
        let periods = lag.as_nanos() / interval.as_nanos() + 1;
        let periods = u32::try_from(periods).map_err(|_| PpoError::CounterOverflow)?;
        let advance = interval
            .checked_mul(periods)
            .ok_or(PpoError::CounterOverflow)?;
        self.next_wall_deadline = deadline.checked_add(advance);
        Ok(())
    }
}

pub fn run_ppo_smoke_on(
    settings: PpoSmokeConfig,
    device: PolicyDevice,
) -> Result<PpoSmokeReport, PpoError> {
    validate_smoke(settings)?;
    let config = smoke_ppo_config(settings).validate()?;
    let model = PolicyModel::fresh_on(settings.seed, device).map_err(text_error)?;
    let mut trainer = PpoTrainer::new(&model, config, settings.seed ^ 0x51a9)?;
    let mut smoke = PpoSmokeReport::default();
    let capacity = settings
        .environments
        .checked_mul(settings.rollout_decisions)
        .ok_or(PpoError::InvalidConfig("smoke samples"))?;
    let mut pipeline = ActorLearnerPipeline::new(capacity, 1, &model).map_err(pipeline_error)?;
    let actor = pipeline.take_actor(0).map_err(pipeline_error)?;
    let (report_sender, report_receiver) = sync_channel(ACTOR_LEARNER_BUFFERS);
    let worker = thread::Builder::new()
        .name("drysua-ppo-actor".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || -> Result<(), PpoError> {
            let mut sampling = PpoRng::new(settings.seed ^ 0xa17e);
            let mut environments = build_environments(settings)?;
            for _ in 0..settings.updates {
                let (lease, mut rollout) = actor.wait_rollout().map_err(pipeline_error)?;
                let mut report = PpoSmokeReport::default();
                collect_update(
                    lease.policy(),
                    &mut sampling,
                    &mut environments,
                    config,
                    settings.rollout_decisions,
                    &mut rollout,
                    &mut report,
                )?;
                lease.try_submit(rollout).map_err(pipeline_error)?;
                report_sender
                    .send(report)
                    .map_err(|_| PpoError::Model("actor report channel disconnected".to_owned()))?;
            }
            Ok(())
        })
        .map_err(|error| PpoError::Model(format!("actor worker spawn failed: {error}")))?;
    let learner_result = (|| -> Result<(), PpoError> {
        for _ in 0..settings.updates {
            let batch = pipeline.accept().map_err(pipeline_error)?.finish(config)?;
            let actor_report = report_receiver
                .recv()
                .map_err(|_| PpoError::Model("actor report channel disconnected".to_owned()))?;
            merge_actor_report(&mut smoke, actor_report)?;
            let report = trainer.train_pipeline_update(&model, &batch)?;
            record_update(&mut smoke, capacity, report)?;
            pipeline.publish(&model).map_err(pipeline_error)?;
        }
        Ok(())
    })();
    drop(pipeline);
    drop(report_receiver);
    let worker_result = worker
        .join()
        .map_err(|_| PpoError::Model("actor worker panicked".to_owned()))?;
    learner_result?;
    worker_result?;
    Ok(smoke)
}

/// Maximum decision rounds one [`TrainingCollectionSlice`] window can run.
pub const TRAINING_COLLECTION_SLICE_MAX_ROUNDS: usize = episode::ACTOR_DECISIONS;

/// Fixed settings for one bounded [`TrainingCollectionSlice`].
///
/// One environment selects the serial baseline; even counts from two to
/// [`MAX_TRAINING_ENVIRONMENTS`](crate::MAX_TRAINING_ENVIRONMENTS) select the
/// production worker pool. `update` seeds the opponent schedule and the
/// retention phases exactly as a production update does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCollectionSliceConfig {
    pub seed: u64,
    pub environments: usize,
    pub update: u64,
    /// Decision rounds executed during construction, outside every measured
    /// window; zero starts the first measured window at the first decision.
    pub warmup_rounds: usize,
    /// Fixed complete-episode collection groups: 1 is the global barrier,
    /// 2/4 partition the same streams into independent pipelined batches.
    pub pipeline_groups: usize,
}

/// Work performed by one bounded collection window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingCollectionSliceReport {
    pub environments: usize,
    pub warmup_rounds: usize,
    pub rounds: usize,
    pub decisions: u64,
    pub ticks: u64,
    pub start_tick: u32,
    pub end_tick: u32,
    pub retained_samples: usize,
}

/// Opt-in per-phase wall accounting for one bounded window.
///
impl episode::CollectionPhases {
    /// Adds one group's phase accounting into the run-wide report.
    fn add_to(&self, phases: &mut TrainingCollectionPhaseReport) {
        phases.prepare_ns += self.prepare_ns.load(std::sync::atomic::Ordering::Relaxed);
        phases.advance_ns += self.advance_ns.load(std::sync::atomic::Ordering::Relaxed);
        phases.forward_ns += self.forward_ns.load(std::sync::atomic::Ordering::Relaxed);
        phases.barrier_ns += self.barrier_ns.load(std::sync::atomic::Ordering::Relaxed);
        phases.apply_ns += self.apply_ns.load(std::sync::atomic::Ordering::Relaxed);
        phases.flush_wait_ns += self
            .flush_wait_ns
            .load(std::sync::atomic::Ordering::Relaxed);
        phases.evaluator_ns += self.evaluator_ns.load(std::sync::atomic::Ordering::Relaxed);
    }
}

/// `flush_wait_ns` is a subset of `apply_ns`; `prepare_ns`, `advance_ns` and
/// `evaluator_ns` run concurrently with the orchestrator phases, so their sums
/// are worker occupancy rather than wall time. Instrumentation only observes,
/// so a phased window runs the identical collector operations as a fast one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingCollectionPhaseReport {
    pub prepare_ns: u64,
    pub advance_ns: u64,
    pub forward_ns: u64,
    pub barrier_ns: u64,
    pub apply_ns: u64,
    pub flush_wait_ns: u64,
    pub evaluator_ns: u64,
}

/// A bounded, self-contained slice of the real complete-episode collector.
///
/// Construction builds fresh Map2 worlds plus a fresh model and runs the
/// optional warmup through the same collector that later measures, so warmup
/// and window share one trajectory. [`Self::run`] then advances exactly the
/// requested decision rounds and returns the work performed. The slice is
/// single-use, which keeps criterion's batched setup and every measured window
/// identical.
///
/// This is benchmark support code. Production training never constructs a
/// slice, requires even paired stream counts, and keeps running whole
/// episodes; the slice only exposes the same per-decision work under a caller
/// owned bound. Opponents follow the production mastery-v1 schedule at its
/// deterministic fresh stage (Weak), the only schedule valid across the full
/// environment range.
pub struct TrainingCollectionSlice {
    settings: TrainingJobConfig,
    model: PolicyModel,
    environments: Vec<TrainingEnvironment>,
    rollout: PpoRollout,
    sampling: PpoRng,
    update: u64,
    warmup_rounds: usize,
    report: PpoSmokeReport,
    fresh: bool,
}

impl TrainingCollectionSlice {
    /// Builds fresh worlds and model for one bounded collection slice.
    pub fn new(
        config: TrainingCollectionSliceConfig,
        device: PolicyDevice,
    ) -> Result<Self, PpoError> {
        let settings = collection_slice_settings(config)?;
        let model = PolicyModel::fresh_on(config.seed, device).map_err(text_error)?;
        let mastery = crate::MasteryProgress::default();
        let environments = if config.environments == 1 {
            vec![episode::single_environment(
                &settings,
                config.update,
                Some(&mastery),
            )?]
        } else {
            episode::environments_with_mastery(&settings, config.update, Some(&mastery))?
        };
        assert_eq!(environments.len(), config.environments);
        let capacity = config
            .environments
            .checked_mul(settings.ppo.rollout_decisions)
            .ok_or(PpoError::InvalidConfig("collection slice capacity"))?;
        let rollout = PpoRollout::new(capacity, model.policy_identity().map_err(text_error)?)?;
        let mut slice = Self {
            settings,
            model,
            environments,
            rollout,
            sampling: PpoRng::new(config.seed ^ 0x6265_6e63_686d_6172),
            update: config.update,
            warmup_rounds: config.warmup_rounds,
            report: PpoSmokeReport::default(),
            fresh: false,
        };
        if config.warmup_rounds > 0 {
            // The warmup is its own window: a fresh rollout and report keep it
            // out of the measured sample sequence, exactly as a production
            // update would start one.
            let mut warmup_rollout =
                PpoRollout::new(capacity, slice.model.policy_identity().map_err(text_error)?)?;
            let mut warmup_report = PpoSmokeReport::default();
            let mut warmup_phases = None;
            let warmup = collection_advance(
                &slice.settings,
                &slice.model,
                &mut slice.environments,
                &mut slice.sampling,
                slice.update,
                config.warmup_rounds,
                config.warmup_rounds,
                &mut warmup_phases,
                &mut warmup_rollout,
                &mut warmup_report,
            )?;
            assert_eq!(warmup.rounds, config.warmup_rounds);
        }
        slice.fresh = true;
        Ok(slice)
    }

    /// Advances one measured window of exactly `rounds` decision rounds.
    pub fn run(&mut self, rounds: usize) -> Result<TrainingCollectionSliceReport, PpoError> {
        assert!(self.fresh, "collection slice is single-use");
        self.fresh = false;
        let mut phases = None;
        collection_advance(
            &self.settings,
            &self.model,
            &mut self.environments,
            &mut self.sampling,
            self.update,
            self.warmup_rounds,
            rounds,
            &mut phases,
            &mut self.rollout,
            &mut self.report,
        )
    }

    /// Advances one measured window while recording per-phase wall time.
    ///
    /// The serial one-environment reference does not run the worker pool and
    /// therefore has no phases to attribute.
    pub fn run_phased(
        &mut self,
        rounds: usize,
    ) -> Result<(TrainingCollectionSliceReport, TrainingCollectionPhaseReport), PpoError> {
        assert!(self.fresh, "collection slice is single-use");
        if self.environments.len() == 1 {
            return Err(PpoError::InvalidConfig("collection slice phases"));
        }
        self.fresh = false;
        let mut phases = Some(TrainingCollectionPhaseReport::default());
        let report = collection_advance(
            &self.settings,
            &self.model,
            &mut self.environments,
            &mut self.sampling,
            self.update,
            self.warmup_rounds,
            rounds,
            &mut phases,
            &mut self.rollout,
            &mut self.report,
        )?;
        Ok((
            report,
            phases.expect("phase accounting stays enabled for the window"),
        ))
    }
}

/// Advances `rounds` decision rounds over the already warm environments.
#[allow(clippy::too_many_arguments)]
fn collection_advance(
    settings: &TrainingJobConfig,
    model: &PolicyModel,
    environments: &mut [TrainingEnvironment],
    sampling: &mut PpoRng,
    update: u64,
    warmup_rounds: usize,
    rounds: usize,
    phases: &mut Option<TrainingCollectionPhaseReport>,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<TrainingCollectionSliceReport, PpoError> {
    validate_slice_window(warmup_rounds, rounds)?;
    let start_tick = slice_tick(environments)?;
    let ticks_before = report.elapsed_ticks;
    let retained_before = rollout.len();
    let count = phases.is_some().then_some(settings.pipeline_groups);
    let counters: Vec<episode::CollectionPhases> = (0..count.unwrap_or(0))
        .map(|_| episode::CollectionPhases::default())
        .collect();
    if environments.len() == 1 {
        episode::collect_serial_bounded(
            model,
            sampling,
            &mut environments[0],
            settings,
            update,
            rounds,
            rollout,
            report,
        )?;
    } else {
        episode::collect_groups_bounded(
            model,
            sampling,
            environments,
            settings,
            update,
            settings.pipeline_groups,
            rounds,
            false,
            (!counters.is_empty()).then_some(counters.as_slice()),
            rollout,
            report,
        )?;
    }
    if let Some(phases) = phases {
        for counter in &counters {
            counter.add_to(phases);
        }
    }
    let decisions = u64::try_from(rounds)
        .ok()
        .and_then(|rounds| rounds.checked_mul(environments.len() as u64))
        .ok_or(PpoError::CounterOverflow)?;
    let ticks = report
        .elapsed_ticks
        .checked_sub(ticks_before)
        .ok_or(PpoError::CounterOverflow)?;
    assert_eq!(
        ticks,
        decisions * u64::from(crate::MAP2_DECISION_INTERVAL_TICKS),
        "bounded collection tick count"
    );
    assert_eq!(
        report.terminal_wins
            + report.terminal_losses
            + report.terminal_draws
            + report.episode_timeouts,
        0,
        "bounded collection ended an episode"
    );
    Ok(TrainingCollectionSliceReport {
        environments: environments.len(),
        warmup_rounds,
        rounds,
        decisions,
        ticks,
        start_tick,
        end_tick: slice_tick(environments)?,
        retained_samples: rollout.len() - retained_before,
    })
}

fn slice_tick(environments: &[TrainingEnvironment]) -> Result<u32, PpoError> {
    let environment = environments
        .first()
        .ok_or(PpoError::InvalidConfig("collection slice environments"))?;
    Ok(environment.seats[environment.policy_seat]
        .tracker
        .current()
        .ok_or(PpoError::InvalidTransition("collection slice snapshot"))?
        .tick)
}

/// Bounds one slice window and its untimed warmup against the actor decision
/// ceiling. Each window owns a fresh rollout, so the bounds are per window.
fn validate_slice_window(warmup_rounds: usize, rounds: usize) -> Result<(), PpoError> {
    if rounds == 0 || rounds > TRAINING_COLLECTION_SLICE_MAX_ROUNDS {
        return Err(PpoError::InvalidConfig("collection slice rounds"));
    }
    if warmup_rounds > TRAINING_COLLECTION_SLICE_MAX_ROUNDS {
        return Err(PpoError::InvalidConfig("collection slice warmup"));
    }
    Ok(())
}

fn collection_slice_settings(
    config: TrainingCollectionSliceConfig,
) -> Result<TrainingJobConfig, PpoError> {
    if config.environments == 0 || config.environments > TRAINING_MAX_ENVIRONMENTS {
        return Err(PpoError::InvalidConfig("collection slice environments"));
    }
    if config.environments > 1 && !episode::valid_environment_count(config.environments) {
        return Err(PpoError::InvalidConfig(
            "collection slice paired environment count",
        ));
    }
    if config.warmup_rounds > TRAINING_COLLECTION_SLICE_MAX_ROUNDS {
        return Err(PpoError::InvalidConfig("collection slice warmup"));
    }
    let settings = TrainingJobConfig {
        mastery_config: Some(crate::MasteryConfig::default()),
        opponent_schedule: crate::TrainingOpponentSchedule::MasteryV1,
        episode_time_cost: 0.0,
        terminal_only: false,
        complete_episodes: true,
        pipeline_groups: config.pipeline_groups,
        updates: 1,
        ppo: PpoConfig {
            decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
            environments: config.environments,
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            epochs: 1,
            minibatch: 512,
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            target_kl: 1.0,
            ..PpoConfig::default()
        },
        checkpoint_cadence: TrainingCheckpointCadence::Updates(1),
        resume_provenance: ResumeProvenance::Strict,
        seed: config.seed,
        map: MapId(2),
        git_commit: String::new(),
        simulator_commit: String::new(),
    };
    episode::validate_pipeline_groups(&settings)?;
    Ok(settings)
}

/// Runs bounded PPO after optionally loading deployment weights for a fresh run.
pub fn run_training_job_on_with_initial_weights<F>(
    settings: TrainingJobConfig,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
    checkpointed: F,
) -> Result<TrainingJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport) + Send + 'static,
{
    let directory = checkpoint_directory.to_path_buf();
    let initial_weights = initial_weights_directory.map(Path::to_path_buf);
    thread::Builder::new()
        .name("drysua-training".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let _log_flush = FlushPerformanceLogs;
            run_training_job_inner(
                settings,
                device,
                &directory,
                resume,
                initial_weights.as_deref(),
                checkpointed,
            )
        })
        .map_err(|error| PpoError::Model(format!("training worker spawn failed: {error}")))?
        .join()
        .map_err(|_| PpoError::Model("training worker panicked".to_owned()))?
}

fn run_training_job_inner<F>(
    settings: TrainingJobConfig,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
    mut checkpointed: F,
) -> Result<TrainingJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport),
{
    let config = validate_training_job(&settings, resume)?;
    validate_initial_weights_directory(checkpoint_directory, resume, initial_weights_directory)?;
    validate_training_directory(checkpoint_directory, resume)?;
    let _directory_lock = TrainingDirectoryLock::acquire(checkpoint_directory)?;
    let run = training_checkpoint_run(&settings, device, config)?;
    let capacity = config
        .environments
        .checked_mul(config.rollout_decisions)
        .ok_or(PpoError::InvalidConfig("training samples"))?;
    let mut session =
        time_training_scope(TrainingTimingScope::SessionInitialization, None, || {
            TrainingSession::initialize(
                &settings,
                device,
                checkpoint_directory,
                resume,
                initial_weights_directory,
                config,
                run,
            )
        })?;
    let start = session.completed_updates;
    if start > settings.updates {
        return Err(PpoError::InvalidConfig(
            "training update target precedes checkpoint",
        ));
    }
    if session.migrated_provenance {
        let migration = session.checkpoint_report(None);
        let durable = time_training_checkpoint(session.completed_updates, || {
            session.save(checkpoint_directory, migration)
        })?;
        checkpointed(durable);
    } else if resume {
        time_training_scope(
            TrainingTimingScope::ResumeRuntimeExport,
            Some(session.completed_updates),
            || {
                TrainingArtifact::save_runtime_weights(&session.model, checkpoint_directory)
                    .map_err(text_error)
            },
        )?;
    }
    session.run_updates(
        &settings,
        config,
        capacity,
        checkpoint_directory,
        &mut checkpointed,
    )?;
    Ok(session.report())
}

/// Session counters merged from actor reports and read by checkpoints.
///
/// Both training sessions keep the same set, so merging and reporting live
/// here once.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SmokeCounters {
    pub(crate) map2_reward: Map2TrainingReward,
    pub(crate) episode_timeouts: u64,
    pub(crate) terminal_wins: u64,
    pub(crate) terminal_losses: u64,
    pub(crate) terminal_draws: u64,
    pub(crate) rejected_orders: u64,
    pub(crate) elapsed_ticks: u64,
}

impl SmokeCounters {
    /// Adds one actor report's counters into the running totals.
    pub(crate) fn merge(&mut self, report: &PpoSmokeReport) -> Result<(), PpoError> {
        self.map2_reward.merge(report.map2_reward)?;
        self.episode_timeouts = self
            .episode_timeouts
            .checked_add(report.episode_timeouts)
            .ok_or(PpoError::CounterOverflow)?;
        self.terminal_wins = self
            .terminal_wins
            .checked_add(report.terminal_wins)
            .ok_or(PpoError::CounterOverflow)?;
        self.terminal_losses = self
            .terminal_losses
            .checked_add(report.terminal_losses)
            .ok_or(PpoError::CounterOverflow)?;
        self.terminal_draws = self
            .terminal_draws
            .checked_add(report.terminal_draws)
            .ok_or(PpoError::CounterOverflow)?;
        self.rejected_orders = self
            .rejected_orders
            .checked_add(report.rejected_orders)
            .ok_or(PpoError::CounterOverflow)?;
        self.elapsed_ticks = self
            .elapsed_ticks
            .checked_add(report.elapsed_ticks)
            .ok_or(PpoError::CounterOverflow)?;
        Ok(())
    }
}

/// One checkpoint report from the running counters and the latest update.
pub(crate) fn training_checkpoint_report(
    counters: &SmokeCounters,
    completed_updates: u64,
    optimizer_step: u64,
    rollout_samples: u64,
    latest: PpoUpdateReport,
    cleanup_warning: Option<String>,
) -> TrainingCheckpointReport {
    TrainingCheckpointReport {
        map2_reward: counters.map2_reward,
        episode_timeouts: counters.episode_timeouts,
        completed_updates,
        optimizer_step,
        rollout_samples,
        policy_loss: latest.policy_loss,
        value_loss: latest.value_loss,
        entropy: latest.entropy,
        approximate_kl: update_kl(latest),
        stopped_for_kl: latest.stopped_for_kl,
        terminal_wins: counters.terminal_wins,
        terminal_losses: counters.terminal_losses,
        terminal_draws: counters.terminal_draws,
        rejected_orders: counters.rejected_orders,
        elapsed_ticks: counters.elapsed_ticks,
        cleanup_warning,
    }
}

/// Everything one durable checkpoint is captured from.
pub(crate) struct SessionCheckpoint<'a> {
    pub(crate) model: &'a PolicyModel,
    pub(crate) trainer: &'a PpoTrainer,
    pub(crate) run: &'a CheckpointRun,
    pub(crate) sampling: &'a PpoRng,
    pub(crate) completed_updates: u64,
    pub(crate) rollout_samples: u64,
    pub(crate) mastery: Option<crate::MasteryProgress>,
}

/// Captures and commits one durable checkpoint plus the runtime weights.
pub(crate) fn save_training_artifact(
    session: SessionCheckpoint<'_>,
    directory: &Path,
    report: TrainingCheckpointReport,
) -> Result<TrainingCheckpointReport, PpoError> {
    let (state, draws) = session.sampling.checkpoint();
    let progress = CheckpointProgress {
        mastery: session.mastery,
        global_update: session.completed_updates,
        policy_version: session.completed_updates,
        scheduler_step: session.completed_updates,
        curriculum_stage: 0,
        rollout_samples: session.rollout_samples,
        best_evaluation: None,
        rng_states: vec![
            RngCheckpoint::new("ppo_actor_sampling", state, draws).map_err(text_error)?,
        ],
        league_references: Vec::new(),
    };
    let artifact = TrainingArtifact::capture(
        session.model,
        session.trainer,
        session.run.clone(),
        progress,
    )
    .map_err(text_error)?;
    let outcome = artifact.save(directory).map_err(text_error)?;
    TrainingArtifact::save_runtime_weights(session.model, directory).map_err(text_error)?;
    let cleanup_warning = match outcome {
        CheckpointSaveOutcome::Committed => None,
        CheckpointSaveOutcome::CommittedWithCleanupError(message) => Some(message),
    };
    Ok(TrainingCheckpointReport {
        cleanup_warning,
        ..report
    })
}

/// The device name recorded in a run scope.
pub(crate) fn device_name(device: PolicyDevice) -> &'static str {
    match device {
        PolicyDevice::Cpu => "cpu",
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        PolicyDevice::Cuda { .. } => "cuda",
        #[cfg(all(feature = "metal", target_os = "macos"))]
        PolicyDevice::Metal { .. } => "metal",
    }
}

struct RestoredTrainingSession {
    trainer: PpoTrainer,
    sampling: PpoRng,
    completed_updates: u64,
    rollout_samples: u64,
    migrated_provenance: bool,
    mastery: Option<crate::MasteryProgress>,
}

struct TrainingSession {
    mastery: Option<crate::MasteryProgress>,
    counters: SmokeCounters,
    model: PolicyModel,
    trainer: PpoTrainer,
    sampling: PpoRng,
    run: CheckpointRun,
    completed_updates: u64,
    rollout_samples: u64,
    latest: PpoUpdateReport,
    migrated_provenance: bool,
    starting_policy_fingerprint: u64,
}

impl TrainingSession {
    fn run_updates(
        &mut self,
        settings: &TrainingJobConfig,
        config: PpoConfig,
        capacity: usize,
        directory: &Path,
        checkpointed: &mut impl FnMut(TrainingCheckpointReport),
    ) -> Result<(), PpoError> {
        let started = Instant::now();
        let mut schedule = TrainingCheckpointSchedule::new(settings.checkpoint_cadence)?;
        for update in self.completed_updates..settings.updates {
            if self
                .mastery
                .as_ref()
                .is_some_and(crate::MasteryProgress::completed)
            {
                break;
            }
            let stage = self.mastery.as_ref().map(crate::MasteryProgress::stage);
            let report = self.train_update(settings, update, config, capacity)?;
            let changed = stage != self.mastery.as_ref().map(crate::MasteryProgress::stage);
            let final_update = report.completed_updates == settings.updates;
            if schedule.is_due(report.completed_updates, started.elapsed())
                || final_update
                || changed
            {
                let durable = time_training_checkpoint(self.completed_updates, || {
                    self.save(directory, report)
                })?;
                checkpointed(durable);
                schedule.mark_committed(started.elapsed())?;
            }
        }
        Ok(())
    }

    fn initialize(
        settings: &TrainingJobConfig,
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        initial_weights_directory: Option<&Path>,
        config: PpoConfig,
        run: CheckpointRun,
    ) -> Result<Self, PpoError> {
        let model = PolicyModel::fresh_on(settings.seed, device).map_err(text_error)?;
        if !resume && let Some(initial_weights_directory) = initial_weights_directory {
            TrainingArtifact::load_runtime_weights(&model, initial_weights_directory)
                .map_err(text_error)?;
        }
        let restored = if resume {
            restore_training_session(&model, directory, &run, config, settings.resume_provenance)?
        } else {
            let trainer = PpoTrainer::new(&model, config, settings.seed ^ 0x51a9)?;
            RestoredTrainingSession {
                trainer,
                sampling: PpoRng::new(settings.seed ^ 0xa17e),
                completed_updates: 0,
                rollout_samples: 0,
                migrated_provenance: false,
                mastery: settings
                    .mastery_config
                    .map(|_| crate::MasteryProgress::default()),
            }
        };
        let starting_policy_fingerprint =
            PolicySnapshot::capture(&model, restored.completed_updates)
                .map_err(text_error)?
                .fingerprint();
        Ok(Self {
            mastery: restored.mastery,
            model,
            trainer: restored.trainer,
            counters: SmokeCounters::default(),
            sampling: restored.sampling,
            run,
            completed_updates: restored.completed_updates,
            rollout_samples: restored.rollout_samples,
            latest: PpoUpdateReport::default(),
            migrated_provenance: restored.migrated_provenance,
            starting_policy_fingerprint,
        })
    }

    fn train_update(
        &mut self,
        settings: &TrainingJobConfig,
        update: u64,
        config: PpoConfig,
        capacity: usize,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        let mode = if settings.complete_episodes {
            TrainingUpdateMode::CompleteEpisodes
        } else {
            TrainingUpdateMode::ResetWindow
        };
        let mut timing = TrainingUpdateTimer::new(update, mode, self.trainer.optimizer_step());
        assert_eq!(update, self.completed_updates);
        assert_eq!(self.completed_updates, self.trainer.updates());
        let mut rollout =
            PpoRollout::new(capacity, self.model.policy_identity().map_err(text_error)?)?;
        let mut environments = build_training_environments(
            settings,
            update,
            config,
            &self.model,
            self.mastery.as_ref(),
        )?;
        let mut actor_report = PpoSmokeReport::default();
        timing.enter(TrainingStage::Collection);
        let collected = self.collect_rollout(
            settings,
            update,
            &mut environments,
            &mut rollout,
            &mut actor_report,
        );
        timing.set_samples(rollout.len());
        timing.observe_result(collected)?;
        timing.enter(TrainingStage::BatchPreparation);
        let next_mastery = self.next_mastery(settings, &actor_report)?;
        self.accumulate_actor_counters(&actor_report)?;
        let samples = rollout.len();
        if !settings.complete_episodes {
            assert_eq!(samples, capacity);
        }
        assert!(samples <= capacity);
        let batch = rollout.finish(config)?;
        timing.enter(TrainingStage::Optimization);
        let optimized = self.trainer.train_update(&self.model, &batch);
        timing.set_optimizer_step(self.trainer.optimizer_step());
        self.latest = timing.observe_result(optimized)?;
        self.mastery = next_mastery;
        timing.enter(TrainingStage::Finalization);
        self.completed_updates = self.trainer.updates();
        self.rollout_samples = self
            .rollout_samples
            .checked_add(samples as u64)
            .ok_or(PpoError::CounterOverflow)?;
        self.log_mastery_progress();
        let report = self.checkpoint_report(None);
        timing.complete();
        Ok(report)
    }

    fn collect_rollout(
        &mut self,
        settings: &TrainingJobConfig,
        update: u64,
        environments: &mut [TrainingEnvironment],
        rollout: &mut PpoRollout,
        actor_report: &mut PpoSmokeReport,
    ) -> Result<(), PpoError> {
        if settings.complete_episodes && settings.pipeline_groups == 1 {
            episode::collect_bounded(
                &self.model,
                &mut self.sampling,
                environments,
                settings,
                update,
                episode::ACTOR_DECISIONS,
                true,
                rollout,
                actor_report,
            )
        } else if settings.complete_episodes {
            episode::collect_groups_bounded(
                &self.model,
                &mut self.sampling,
                environments,
                settings,
                update,
                settings.pipeline_groups,
                episode::ACTOR_DECISIONS,
                true,
                None,
                rollout,
                actor_report,
            )
        } else {
            collect_update(
                &self.model,
                &mut self.sampling,
                environments,
                settings.ppo,
                settings.ppo.rollout_decisions,
                rollout,
                actor_report,
            )
        }
    }

    fn accumulate_actor_counters(&mut self, actor_report: &PpoSmokeReport) -> Result<(), PpoError> {
        self.counters.merge(actor_report)
    }

    fn next_mastery(
        &self,
        settings: &TrainingJobConfig,
        report: &PpoSmokeReport,
    ) -> Result<Option<crate::MasteryProgress>, PpoError> {
        let Some(mut progress) = self.mastery.clone() else {
            return Ok(None);
        };
        let config = settings
            .mastery_config
            .ok_or(PpoError::InvalidConfig("missing mastery config"))?;
        let outcomes = report.completed_episodes.ordered_outcomes();
        if outcomes.len() != settings.ppo.environments || report.rejected_orders != 0 {
            return Err(PpoError::InvalidTransition(
                "mastery requires a complete unrejected batch",
            ));
        }
        progress
            .record_batch(config, &outcomes)
            .map_err(PpoError::InvalidTransition)?;
        Ok(Some(progress))
    }

    fn log_mastery_progress(&self) {
        if let (Some(config), Some(progress)) = (self.run.mastery_config, &self.mastery) {
            crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default()).emit(
                &format_args!("level=INFO event=training_mastery_progress updates={} stage={:?} stage_games={} recent_games={} recent_wins={} window={} win_percent={} completed={}",
                    self.completed_updates, progress.stage(), progress.games(), progress.recent().len(),
                    progress.wins(), config.window(), config.threshold(progress.stage()), progress.completed()),
            );
        }
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
                mastery: self.mastery.clone(),
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

    fn report(&self) -> TrainingJobReport {
        TrainingJobReport {
            mastery_completed: self
                .mastery
                .as_ref()
                .is_some_and(crate::MasteryProgress::completed),
            map2_reward: self.counters.map2_reward,
            episode_timeouts: self.counters.episode_timeouts,
            starting_policy_fingerprint: self.starting_policy_fingerprint,
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            final_policy_loss: self.latest.policy_loss,
            final_value_loss: self.latest.value_loss,
            final_entropy: self.latest.entropy,
            final_kl: update_kl(self.latest),
            terminal_wins: self.counters.terminal_wins,
            terminal_losses: self.counters.terminal_losses,
            terminal_draws: self.counters.terminal_draws,
            rejected_orders: self.counters.rejected_orders,
            elapsed_ticks: self.counters.elapsed_ticks,
        }
    }
}

fn merge_actor_report(
    aggregate: &mut PpoSmokeReport,
    actor: PpoSmokeReport,
) -> Result<(), PpoError> {
    aggregate
        .completed_episodes
        .merge(&actor.completed_episodes)?;
    aggregate.map2_reward.merge(actor.map2_reward)?;
    aggregate.episode_timeouts = aggregate
        .episode_timeouts
        .checked_add(actor.episode_timeouts)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.rejected_orders = aggregate
        .rejected_orders
        .checked_add(actor.rejected_orders)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.elapsed_ticks = aggregate
        .elapsed_ticks
        .checked_add(actor.elapsed_ticks)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.terminal_wins = aggregate
        .terminal_wins
        .checked_add(actor.terminal_wins)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.terminal_losses = aggregate
        .terminal_losses
        .checked_add(actor.terminal_losses)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.terminal_draws = aggregate
        .terminal_draws
        .checked_add(actor.terminal_draws)
        .ok_or(PpoError::CounterOverflow)?;
    Ok(())
}

fn record_update(
    report: &mut PpoSmokeReport,
    transitions: usize,
    update: PpoUpdateReport,
) -> Result<(), PpoError> {
    report.updates = report
        .updates
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    report.transitions = report
        .transitions
        .checked_add(transitions)
        .ok_or(PpoError::CounterOverflow)?;
    report.optimizer_step = update.optimizer_step;
    report.final_policy_loss = update.policy_loss;
    report.final_value_loss = update.value_loss;
    report.final_entropy = update.entropy;
    report.final_kl = update_kl(update);
    Ok(())
}

pub(crate) fn update_kl(update: PpoUpdateReport) -> f64 {
    if update.stopped_for_kl {
        update.rejected_kl
    } else {
        update.approximate_kl
    }
}

fn validate_training_job(
    settings: &TrainingJobConfig,
    resume: bool,
) -> Result<PpoConfig, PpoError> {
    if settings.updates == 0 || settings.updates > 1_000_000 {
        return Err(PpoError::InvalidConfig("training updates"));
    }
    validate_checkpoint_cadence(settings.checkpoint_cadence)?;
    if !resume && settings.resume_provenance != ResumeProvenance::Strict {
        return Err(PpoError::InvalidConfig(
            "fresh training provenance migration",
        ));
    }
    if settings.ppo.environments == 0
        || settings.ppo.environments > TRAINING_MAX_ENVIRONMENTS
        || !settings.ppo.environments.is_multiple_of(2)
    {
        return Err(PpoError::InvalidConfig("training environments"));
    }
    if settings.ppo.rollout_decisions == 0
        || settings.ppo.rollout_decisions > PPO_MAX_ROLLOUT_DECISIONS
    {
        return Err(PpoError::InvalidConfig("training rollout decisions"));
    }
    if settings.map != MapId(2) {
        return Err(PpoError::InvalidConfig("production training requires Map2"));
    }
    if settings.git_commit.is_empty() || settings.git_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("training git commit"));
    }
    if settings.simulator_commit.is_empty() || settings.simulator_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("training simulator commit"));
    }
    let config = settings.ppo.validate()?;
    episode::validate(settings)?;
    validate_training_counters(settings, config)?;
    Ok(config)
}

fn validate_initial_weights_directory(
    checkpoint_directory: &Path,
    resume: bool,
    initial_weights_directory: Option<&Path>,
) -> Result<(), PpoError> {
    if resume && initial_weights_directory.is_some() {
        return Err(PpoError::InvalidConfig("resume initial weights"));
    }
    if initial_weights_directory.is_some_and(|initial| initial == checkpoint_directory) {
        return Err(PpoError::InvalidConfig(
            "training initial weights directory",
        ));
    }
    Ok(())
}

fn validate_checkpoint_cadence(cadence: TrainingCheckpointCadence) -> Result<(), PpoError> {
    match cadence {
        TrainingCheckpointCadence::Updates(interval) if (1..=1_000).contains(&interval) => Ok(()),
        TrainingCheckpointCadence::WallTime(interval)
            if (Duration::from_secs(1)..=Duration::from_secs(86_400)).contains(&interval) =>
        {
            Ok(())
        }
        TrainingCheckpointCadence::Updates(_) => {
            Err(PpoError::InvalidConfig("training checkpoint updates"))
        }
        TrainingCheckpointCadence::WallTime(_) => {
            Err(PpoError::InvalidConfig("training checkpoint wall time"))
        }
    }
}

fn validate_training_counters(
    settings: &TrainingJobConfig,
    config: PpoConfig,
) -> Result<(), PpoError> {
    let samples = (config.environments as u64)
        .checked_mul(config.rollout_decisions as u64)
        .ok_or(PpoError::InvalidConfig("training sample counter"))?;
    let actor_seed_draws = settings
        .updates
        .checked_mul(config.environments as u64)
        .ok_or(PpoError::InvalidConfig("training actor RNG counter"))?;
    let actor_decisions = if settings.complete_episodes {
        episode::ACTOR_DECISIONS
    } else {
        config.rollout_decisions
    };
    let actor_stream_draws = (actor_decisions as u64)
        .checked_mul(PPO_MAX_POLICY_SAMPLE_DRAWS)
        .ok_or(PpoError::InvalidConfig("training actor RNG counter"))?;
    if actor_seed_draws > MAX_TRAINING_COUNTER || actor_stream_draws > MAX_TRAINING_COUNTER {
        return Err(PpoError::InvalidConfig("training actor RNG counter"));
    }
    let shuffle_draws = settings
        .updates
        .checked_mul(config.epochs as u64)
        .and_then(|count| count.checked_mul(samples.saturating_sub(1)))
        .ok_or(PpoError::InvalidConfig("training shuffle RNG counter"))?;
    if shuffle_draws > MAX_TRAINING_COUNTER {
        return Err(PpoError::InvalidConfig("training shuffle RNG counter"));
    }
    let minibatches = samples.div_ceil(config.minibatch as u64);
    let optimizer_steps = settings
        .updates
        .checked_mul(config.epochs as u64)
        .and_then(|count| count.checked_mul(minibatches))
        .ok_or(PpoError::InvalidConfig("training optimizer counter"))?;
    if optimizer_steps > MODEL_MAX_OPTIMIZER_STEP {
        return Err(PpoError::InvalidConfig("training optimizer counter"));
    }
    Ok(())
}

fn validate_training_directory(directory: &Path, resume: bool) -> Result<(), PpoError> {
    let metadata = directory
        .symlink_metadata()
        .map_err(|_| PpoError::InvalidConfig("training checkpoint directory"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PpoError::InvalidConfig("training checkpoint directory"));
    }
    let mut entries = directory
        .read_dir()
        .map_err(|error| PpoError::Model(format!("checkpoint directory read failed: {error}")))?;
    if !resume && entries.next().is_some() {
        return Err(PpoError::InvalidConfig(
            "fresh training checkpoint directory",
        ));
    }
    Ok(())
}

impl TrainingDirectoryLock {
    fn acquire(directory: &Path) -> Result<Self, PpoError> {
        let path = directory.join(".training.lock");
        if path
            .symlink_metadata()
            .is_ok_and(|metadata| metadata.file_type().is_symlink() || !metadata.is_file())
        {
            return Err(PpoError::InvalidConfig("training lock file"));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| PpoError::Model(format!("training lock open failed: {error}")))?;
        validate_open_lock_file(&file, &path)?;
        file.try_lock().map_err(|error| {
            PpoError::Model(format!("training checkpoint directory is locked: {error}"))
        })?;
        Ok(Self { _file: file })
    }
}

#[cfg(unix)]
fn validate_open_lock_file(file: &File, path: &Path) -> Result<(), PpoError> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let opened = file
        .metadata()
        .map_err(|error| PpoError::Model(format!("training lock metadata failed: {error}")))?;
    let linked = path
        .symlink_metadata()
        .map_err(|error| PpoError::Model(format!("training lock metadata failed: {error}")))?;
    if linked.file_type().is_symlink()
        || !linked.is_file()
        || opened.dev() != linked.dev()
        || opened.ino() != linked.ino()
    {
        return Err(PpoError::InvalidConfig("training lock file"));
    }
    file.set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|error| PpoError::Model(format!("training lock permissions failed: {error}")))
}

#[cfg(not(unix))]
fn validate_open_lock_file(file: &File, path: &Path) -> Result<(), PpoError> {
    let opened = file
        .metadata()
        .map_err(|error| PpoError::Model(format!("training lock metadata failed: {error}")))?;
    let linked = path
        .symlink_metadata()
        .map_err(|error| PpoError::Model(format!("training lock metadata failed: {error}")))?;
    if linked.file_type().is_symlink() || !opened.is_file() || !linked.is_file() {
        return Err(PpoError::InvalidConfig("training lock file"));
    }
    Ok(())
}

pub(crate) fn training_checkpoint_run(
    settings: &TrainingJobConfig,
    device: PolicyDevice,
    config: PpoConfig,
) -> Result<CheckpointRun, PpoError> {
    let device_name = device_name(device);
    let mut command_line = format!(
        "train-full --environments {} --rollout {} --epochs {} --minibatch {} --seed {} --map {} --device {device_name}",
        config.environments,
        config.rollout_decisions,
        config.epochs,
        config.minibatch,
        settings.seed,
        settings.map.0,
    );
    if settings.complete_episodes {
        command_line.push_str(" --complete-episodes");
    } else {
        command_line.push_str(" --complete-episodes=false");
    }
    if settings.opponent_schedule != crate::TrainingOpponentSchedule::Teacher {
        command_line.push_str(" --opponent-schedule ");
        command_line.push_str(settings.opponent_schedule.as_str());
    }
    if let Some(config) = settings.mastery_config {
        command_line.push_str(&config.canonical_suffix());
    }
    if settings.terminal_only {
        command_line.push_str(" --terminal-only");
    }
    if settings.episode_time_cost > 0.0 {
        command_line.push_str(&format!(
            " --episode-time-cost {}",
            settings.episode_time_cost
        ));
    }
    // The default one-group contract keeps the historical command line
    // byte-identical; a grouped run is a distinct scope and cannot resume a
    // default checkpoint (or the reverse) under strict provenance.
    if settings.pipeline_groups > 1 {
        command_line.push_str(&format!(" --pipeline-groups {}", settings.pipeline_groups));
    }
    Ok(CheckpointRun {
        mastery_config: settings.mastery_config,
        git_commit: settings.git_commit.clone(),
        simulator_commit: settings.simulator_commit.clone(),
        enabled_features: compiled_features(),
        command_line,
        run_seed: settings.seed,
        map: settings.map,
        hero: SHADOW_FIEND,
        device: CheckpointDevice::from_policy(device).map_err(text_error)?,
        batch_size: config.minibatch,
        rules_audit_version: PPO_RULES_AUDIT_VERSION,
    })
}

fn restore_training_session(
    model: &PolicyModel,
    directory: &Path,
    run: &CheckpointRun,
    config: PpoConfig,
    provenance: ResumeProvenance,
) -> Result<RestoredTrainingSession, PpoError> {
    let parts = match provenance {
        ResumeProvenance::Strict => restore_strict_session(model, directory, run, config)?,
        ResumeProvenance::MigrateGitCommit => {
            let artifact = TrainingArtifact::load(directory).map_err(text_error)?;
            let restore_run = artifact.run().clone();
            validate_provenance_migration(&restore_run, run)?;
            restore_loaded_session(model, artifact, &restore_run, config, true)?
        }
    };
    let progress = parts.progress;
    Ok(RestoredTrainingSession {
        trainer: parts.trainer,
        sampling: parts.sampling,
        completed_updates: progress.global_update,
        rollout_samples: progress.rollout_samples,
        migrated_provenance: parts.migrated,
        mastery: progress.mastery,
    })
}

/// One restored strict session's parts.
pub(crate) struct RestoredSessionParts {
    pub(crate) trainer: PpoTrainer,
    pub(crate) sampling: PpoRng,
    pub(crate) progress: CheckpointProgress,
    pub(crate) migrated: bool,
}

/// Restores a checkpoint whose run scope must match exactly.
pub(crate) fn restore_strict_session(
    model: &PolicyModel,
    directory: &Path,
    run: &CheckpointRun,
    config: PpoConfig,
) -> Result<RestoredSessionParts, PpoError> {
    let artifact = TrainingArtifact::load_compatible(directory, run).map_err(text_error)?;
    restore_loaded_session(model, artifact, run, config, false)
}

/// Restores one already loaded artifact and checks the session invariants.
fn restore_loaded_session(
    model: &PolicyModel,
    artifact: TrainingArtifact,
    restore_run: &CheckpointRun,
    config: PpoConfig,
    migrated: bool,
) -> Result<RestoredSessionParts, PpoError> {
    let restored = artifact.restore(model, restore_run).map_err(text_error)?;
    if restored.trainer().config() != config {
        return Err(PpoError::InvalidConfig("training checkpoint PPO config"));
    }
    let progress = restored.progress();
    if progress.policy_version != progress.global_update {
        return Err(PpoError::InvalidConfig(
            "training checkpoint policy version",
        ));
    }
    let sampling = restore_sampling_rng(&progress.rng_states)?;
    let progress = progress.clone();
    let (trainer, _, _) = restored.into_parts();
    Ok(RestoredSessionParts {
        trainer,
        sampling,
        progress,
        migrated,
    })
}

fn validate_provenance_migration(
    stored: &CheckpointRun,
    expected: &CheckpointRun,
) -> Result<(), PpoError> {
    if stored.git_commit == expected.git_commit {
        return Err(PpoError::InvalidConfig(
            "provenance migration requires a changed Git commit",
        ));
    }
    let mut migrated = stored.clone();
    migrated.git_commit.clone_from(&expected.git_commit);
    if &migrated != expected {
        return Err(PpoError::InvalidConfig("provenance migration scope"));
    }
    Ok(())
}

fn restore_sampling_rng(states: &[RngCheckpoint]) -> Result<PpoRng, PpoError> {
    let mut matching = states
        .iter()
        .filter(|checkpoint| checkpoint.name() == "ppo_actor_sampling");
    let checkpoint = matching
        .next()
        .ok_or(PpoError::InvalidConfig("training checkpoint actor RNG"))?;
    if matching.next().is_some() {
        return Err(PpoError::InvalidConfig(
            "training checkpoint duplicate actor RNG",
        ));
    }
    PpoRng::from_checkpoint(checkpoint.state(), checkpoint.draws())
}

fn validate_smoke(settings: PpoSmokeConfig) -> Result<(), PpoError> {
    if settings.updates == 0 || settings.updates > 10 {
        return Err(PpoError::InvalidConfig("smoke updates"));
    }
    if settings.environments == 0 || settings.environments > 16 {
        return Err(PpoError::InvalidConfig("smoke environments"));
    }
    if settings.rollout_decisions == 0 || settings.rollout_decisions > 64 {
        return Err(PpoError::InvalidConfig("smoke rollout decisions"));
    }
    if settings.map != MapId(2) {
        return Err(PpoError::InvalidConfig("production training requires Map2"));
    }
    Ok(())
}

fn smoke_ppo_config(settings: PpoSmokeConfig) -> PpoConfig {
    PpoConfig {
        decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
        environments: settings.environments,
        rollout_decisions: settings.rollout_decisions,
        epochs: settings.epochs,
        minibatch: settings.minibatch,
        target_kl: 1.0,
        gamma_tick: MAP2_REWARD_GAMMA_TICK,
        ..PpoConfig::default()
    }
}

fn build_environments(settings: PpoSmokeConfig) -> Result<Vec<TrainingEnvironment>, PpoError> {
    let mut environments = Vec::with_capacity(settings.environments);
    for index in 0..settings.environments {
        let seed = settings
            .seed
            .checked_add(index as u64)
            .ok_or(PpoError::InvalidConfig("environment seed"))?;
        let policy_seat = index % 2;
        environments.push(build_environment(
            seed,
            (settings.seed ^ crate::randomization::OPPONENT_DOMAIN)
                .checked_add(index as u64)
                .ok_or(PpoError::InvalidConfig("opponent seed"))?,
            settings.map,
            policy_seat,
            0,
            OpponentSpec::Teacher,
        )?);
    }
    Ok(environments)
}

fn build_training_environments(
    settings: &TrainingJobConfig,
    update: u64,
    config: PpoConfig,
    model: &PolicyModel,
    mastery: Option<&crate::MasteryProgress>,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    if settings.complete_episodes {
        return if mastery.is_some() {
            episode::environments_with_mastery(settings, update, mastery)
        } else {
            episode::environments(settings, update)
        };
    }
    let decision = update
        .checked_mul(config.rollout_decisions as u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(PpoError::InvalidConfig("training decision counter"))?;
    let offset = update
        .checked_mul(config.environments as u64)
        .ok_or(PpoError::CounterOverflow)?;
    let mut environments = Vec::with_capacity(config.environments);
    let mut warmup_decisions = Vec::with_capacity(config.environments);
    for index in 0..config.environments {
        let stream = offset
            .checked_add(index as u64)
            .ok_or(PpoError::CounterOverflow)?;
        let pair = training_pair_index(stream);
        let opponent = match training_opponent_baseline(pair) {
            CheckpointEvaluationBaseline::Weak => OpponentSpec::Weak,
            CheckpointEvaluationBaseline::Teacher => OpponentSpec::Teacher,
        };
        let environment = build_environment(
            derive_training_seed(settings.seed, pair, crate::randomization::ARENA_DOMAIN),
            derive_training_seed(settings.seed, pair, crate::randomization::OPPONENT_DOMAIN),
            settings.map,
            training_policy_seat(stream),
            decision,
            opponent,
        )?;
        environments.push(environment);
        warmup_decisions.push(training_warmup_decisions(pair));
    }
    warmup_training_environments(
        &mut environments,
        &warmup_decisions,
        model,
        config.decision_interval_ticks,
    )?;
    Ok(environments)
}

pub(crate) use crate::randomization::derive_training_seed;

pub(crate) const fn training_policy_seat(stream: u64) -> usize {
    (stream % 2) as usize
}

pub(crate) const fn training_pair_index(stream: u64) -> u64 {
    stream / 2
}

const fn training_opponent_baseline(pair: u64) -> CheckpointEvaluationBaseline {
    if (pair / 8).is_multiple_of(2) {
        CheckpointEvaluationBaseline::Weak
    } else {
        CheckpointEvaluationBaseline::Teacher
    }
}

pub(crate) const fn training_warmup_decisions(phase: u64) -> usize {
    const PHASES: [usize; 8] = [0, 300, 450, 600, 900, 1_200, 1_800, 2_400];
    PHASES[(phase % PHASES.len() as u64) as usize]
}

fn warmup_training_environments(
    environments: &mut [TrainingEnvironment],
    decisions: &[usize],
    model: &PolicyModel,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    if environments.is_empty()
        || environments.len() != decisions.len()
        || environments.len() > TRAINING_MAX_ENVIRONMENTS
    {
        return Err(PpoError::InvalidConfig("training warmup environments"));
    }
    // Even a zero-decision warmup sends cleanup requests before the first sampled action.
    for environment in environments.iter_mut() {
        let seat = &mut environment.seats[environment.policy_seat];
        seat.order_bookkeeping
            .enable_candidate(&seat.persistence)
            .map_err(PpoError::InvalidTransition)?;
    }
    let maximum = decisions.iter().copied().max().unwrap_or(0);
    for decision in 0..maximum {
        let active = decisions
            .iter()
            .enumerate()
            .filter_map(|(index, limit)| (decision < *limit).then_some(index))
            .collect::<Vec<_>>();
        let requests = requests_for_batched_greedy_decisions(environments, &active, model)?;
        advance_warmup_environments(environments, &active, requests, decision_interval_ticks)?;
    }
    for environment in environments {
        finish_warmup_environment(environment, decision_interval_ticks)?;
    }
    Ok(())
}

fn advance_warmup_environments(
    environments: &mut [TrainingEnvironment],
    active: &[usize],
    requests: Vec<Vec<Option<Request>>>,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    if active.len() != requests.len() {
        return Err(PpoError::InvalidTransition("batched warmup request count"));
    }
    for (&index, requests) in active.iter().zip(requests) {
        let environment = environments
            .get_mut(index)
            .ok_or(PpoError::InvalidTransition(
                "batched warmup environment index",
            ))?;
        let advanced = advance_interval(environment, requests, decision_interval_ticks)?;
        reject_production_rejection(environment, "production warmup")?;
        if advanced.winner.is_some() {
            restart_environment(environment)?;
        }
    }
    Ok(())
}

fn finish_warmup_environment(
    environment: &mut TrainingEnvironment,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    clear_warmup_orders(environment, decision_interval_ticks)?;
    if environment.map == MapId(2) {
        for seat in &mut environment.seats {
            seat.tracker
                .take_map2_reward_interval()
                .map_err(text_error)?;
        }
        return Ok(());
    }
    environment.reward = RewardTracker::default();
    let summary = environment.seats[environment.policy_seat]
        .tracker
        .latest_summary()
        .ok_or(PpoError::InvalidTransition("warmup summary"))?;
    environment.reward.observe(summary, 1.0, None)?;
    Ok(())
}

fn clear_warmup_orders(
    environment: &mut TrainingEnvironment,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    for unit in [crate::ControlledUnit::Hero, crate::ControlledUnit::Courier] {
        stop_warmup_unit(environment, decision_interval_ticks, unit)?;
    }
    for seat in &mut environment.seats {
        seat.persistence.clear_body();
        if seat.order_bookkeeping.is_neural() {
            seat.order_bookkeeping.clear_body();
            seat.pending_active = None;
        }
        seat.teacher = Teacher::new();
        let tick = seat
            .tracker
            .current()
            .ok_or(PpoError::InvalidTransition("warmup boundary snapshot"))?
            .tick;
        seat.local = LocalPolicyState::new(tick);
        assert!(seat.persistence.active_body_sequence().is_none());
    }
    Ok(())
}

fn stop_warmup_unit(
    environment: &mut TrainingEnvironment,
    decision_interval_ticks: u32,
    unit: crate::ControlledUnit,
) -> Result<(), PpoError> {
    let mut requests = Vec::with_capacity(environment.seats.len());
    for seat in &mut environment.seats {
        let space = ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness)
            .map_err(|error| PpoError::Model(error.to_string()))?;
        let action = crate::StructuredAction::Stop { unit };
        let issued = if space.allows(action) {
            space
                .decode(action)
                .map_err(|error| PpoError::Model(error.to_string()))?
        } else {
            None
        };
        requests.push(issue_request(
            seat,
            issued,
            &space,
            ActionKind::Stop,
            false,
        )?);
    }
    let advanced = advance_interval(environment, requests, decision_interval_ticks)?;
    reject_production_rejection(environment, "production warmup cleanup")?;
    if advanced.winner.is_some() {
        restart_environment(environment)?;
    }
    Ok(())
}

fn build_environment(
    seed: u64,
    opponent_seed: u64,
    map: MapId,
    policy_seat: usize,
    decision: u32,
    opponent_spec: OpponentSpec,
) -> Result<TrainingEnvironment, PpoError> {
    build_environment_with_spawn_modifiers(
        seed,
        opponent_seed,
        map,
        policy_seat,
        decision,
        opponent_spec,
        Vec::new(),
    )
}

/// Builds one environment whose units carry trusted spawn modifiers.
///
/// The annealed loop puts one generation's rules on the units their selectors
/// name at world construction and on every later spawn; the default training
/// paths pass none.
fn build_environment_with_spawn_modifiers(
    seed: u64,
    opponent_seed: u64,
    map: MapId,
    policy_seat: usize,
    decision: u32,
    opponent_spec: OpponentSpec,
    spawn_modifiers: Vec<SpawnModifier>,
) -> Result<TrainingEnvironment, PpoError> {
    if policy_seat >= 2 {
        return Err(PpoError::InvalidConfig("training policy seat"));
    }
    let config = ArenaConfig {
        seats: 2,
        map,
        seed,
    };
    let (arena, start) = if spawn_modifiers.is_empty() {
        Arena::new(config)
    } else {
        Arena::new_with_spawn_modifiers(config, spawn_modifiers)
    }
    .map_err(|error| PpoError::Model(error.to_string()))?;
    let seats = setup_seats(start)?;
    let mut reward = RewardTracker::default();
    if map != MapId(2) {
        reward.observe(
            seats[policy_seat]
                .tracker
                .latest_summary()
                .ok_or(PpoError::InvalidTransition("initial summary"))?,
            1.0,
            None,
        )?;
    }
    let opponent = build_opponent(&opponent_spec, opponent_seed)?;
    Ok(TrainingEnvironment {
        arena,
        seats,
        policy_seat,
        reward,
        decision,
        map,
        next_seed: seed.wrapping_add(1u64 << 32),
        next_opponent_seed: opponent_seed.wrapping_add(1u64 << 32),
        opponent_spec,
        opponent,
        retired_rejections: 0,
    })
}

fn build_opponent(spec: &OpponentSpec, _seed: u64) -> Result<OpponentRuntime, PpoError> {
    match spec {
        #[cfg(test)]
        OpponentSpec::Policy(snapshot) => Ok(OpponentRuntime::Policy {
            model: Arc::new(
                snapshot
                    .instantiate()
                    .map_err(|error| PpoError::Model(error.to_string()))?,
            ),
            rng: PpoRng::new(_seed),
        }),
        OpponentSpec::SharedPolicy(model) => Ok(OpponentRuntime::Policy {
            model: Arc::clone(model),
            rng: PpoRng::new(_seed),
        }),
        OpponentSpec::Teacher => Ok(OpponentRuntime::Teacher),
        OpponentSpec::Weak => Ok(OpponentRuntime::Weak),
    }
}

fn setup_seats(start: ArenaStart) -> Result<Vec<ArenaSeatPolicy>, PpoError> {
    start
        .messages
        .into_iter()
        .enumerate()
        .map(|(index, messages)| setup_seat(index, &messages))
        .collect()
}

fn setup_seat(index: usize, messages: &[ServerMsg]) -> Result<ArenaSeatPolicy, PpoError> {
    let info = messages.iter().find_map(|message| match message {
        ServerMsg::MatchStart { info } => Some(info),
        _ => None,
    });
    let snapshot = messages.iter().find_map(|message| match message {
        ServerMsg::Snapshot { view } => Some(view),
        _ => None,
    });
    let slot = SlotId(u8::try_from(index).map_err(|_| PpoError::InvalidTransition("seat"))?);
    let mut tracker =
        StateTracker::new(slot, info.ok_or(PpoError::InvalidTransition("match info"))?)
            .map_err(|error| PpoError::Model(error.to_string()))?;
    if tracker.metadata().map == MapId(2) {
        if !matches!(messages.first(), Some(ServerMsg::MatchStart { .. })) {
            return Err(PpoError::InvalidTransition("initial MatchStart ordering"));
        }
        validate_arena_tick_messages(&messages[1..])?;
    }
    tracker
        .observe_snapshot(snapshot.ok_or(PpoError::InvalidTransition("initial snapshot"))?)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let mut event_batches = messages.iter().filter_map(|message| match message {
        ServerMsg::Events { tick, events } => Some((*tick, events)),
        _ => None,
    });
    let (event_tick, events) = event_batches
        .next()
        .ok_or(PpoError::InvalidTransition("initial events"))?;
    if event_batches.next().is_some() {
        return Err(PpoError::InvalidTransition(
            "multiple initial event batches",
        ));
    }
    tracker
        .observe_events(event_tick, events)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    if tracker.map2_reward_state().is_some() {
        let baseline = tracker.take_map2_reward_interval().map_err(text_error)?;
        assert_eq!(baseline.ticks, 0);
        assert!(baseline.end.is_none());
    }
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).map_err(text_error)?;
    Ok(ArenaSeatPolicy {
        tracker,
        encoder,
        local: LocalPolicyState::new(1),
        persistence: OrderPersistence::default(),
        order_bookkeeping: PolicyOrderBookkeeping::Legacy,
        readiness: ItemReadiness::new(),
        teacher: Teacher::new(),
        sequence: 0,
        rejections: 0,
        pending_active: None,
        readiness_orders: VecDeque::with_capacity(READINESS_ORDER_HISTORY),
        last_issued: None,
        last_rejection: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_update(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environments: &mut [TrainingEnvironment],
    config: PpoConfig,
    decisions: usize,
    rollout: &mut PpoRollout,
    smoke: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    let rejections_before = environment_rejections(environments)?;
    let mut stream_rngs = actor_stream_rngs(sampling, environments.len())?;
    for _ in 0..decisions {
        let pending = collect_round(model, &mut stream_rngs, environments, config)?;
        let bootstrap = bootstrap_values(model, &pending)?;
        commit_round(environments, pending, bootstrap, rollout, smoke)?;
    }
    smoke.rejected_orders =
        rejection_delta(rejections_before, environment_rejections(environments)?)?;
    Ok(())
}

fn actor_stream_rngs(master: &mut PpoRng, count: usize) -> Result<Vec<PpoRng>, PpoError> {
    if count == 0 || count > TRAINING_MAX_ENVIRONMENTS {
        return Err(PpoError::InvalidConfig("actor RNG streams"));
    }
    (0..count)
        .map(|_| master.next_word().map(PpoRng::new))
        .collect()
}

fn environment_rejections(environments: &[TrainingEnvironment]) -> Result<u64, PpoError> {
    let mut total = 0u64;
    for environment in environments {
        total = total
            .checked_add(environment.retired_rejections)
            .ok_or(PpoError::CounterOverflow)?;
        for seat in &environment.seats {
            total = total
                .checked_add(seat.rejections)
                .ok_or(PpoError::CounterOverflow)?;
        }
    }
    Ok(total)
}

fn rejection_delta(before: u64, after: u64) -> Result<u64, PpoError> {
    after.checked_sub(before).ok_or(PpoError::InvalidTransition(
        "arena rejection counter regressed",
    ))
}

fn collect_round(
    model: &PolicyModel,
    sampling: &mut [PpoRng],
    environments: &mut [TrainingEnvironment],
    config: PpoConfig,
) -> Result<Vec<PendingTransition>, PpoError> {
    if sampling.len() != environments.len() {
        return Err(PpoError::InvalidConfig("actor RNG stream count"));
    }
    if config.gamma_tick != MAP2_REWARD_GAMMA_TICK
        && environments
            .iter()
            .any(|environment| environment.map == MapId(2))
    {
        return Err(PpoError::InvalidConfig(
            "Map2 comprehensive reward requires gamma per tick one",
        ));
    }
    let mut frames = Vec::with_capacity(environments.len());
    let mut spaces = Vec::with_capacity(environments.len());
    for environment in environments.iter_mut() {
        let (frame, space) = prepare_policy_sample(environment)?;
        frames.push(frame);
        spaces.push(space);
    }
    let choices = model
        .sample_batch(&frames, &spaces, sampling)
        .map_err(text_error)?;
    let mut pending = Vec::with_capacity(environments.len());
    for ((environment, choice), space) in environments.iter_mut().zip(choices).zip(&spaces) {
        let requests = requests_for_decision_in_space(environment, &choice, space)?;
        let advanced = advance_interval(environment, requests, config.decision_interval_ticks)?;
        reject_production_rejection(environment, "production rollout")?;
        let terminal = advanced.winner.is_some();
        let outcome = terminal_outcome(environment, advanced.winner);
        let map2_reward = (environment.map == MapId(2))
            .then(|| take_map2_reward(environment, map2_reward_end(outcome), advanced.ticks))
            .transpose()?;
        let reward = if let Some(reward) = map2_reward {
            reward.total as f32
        } else {
            let summary = environment.seats[environment.policy_seat]
                .tracker
                .latest_summary()
                .ok_or(PpoError::InvalidTransition("next summary"))?;
            environment
                .reward
                .observe(
                    summary,
                    tick_discount(config.gamma_tick, advanced.ticks)?,
                    outcome,
                )?
                .total
        };
        let next_frame = (!terminal)
            .then(|| encode_next_frame(environment))
            .transpose()?;
        pending.push(PendingTransition {
            choice,
            reward,
            map2_reward,
            ticks: advanced.ticks,
            terminal,
            terminal_outcome: outcome,
            next_frame,
        });
    }
    Ok(pending)
}

fn prepare_policy_sample(
    environment: &mut TrainingEnvironment,
) -> Result<(FeatureFrame, ActionSpace), PpoError> {
    let seat = &mut environment.seats[environment.policy_seat];
    prepare_neural_seat_policy_sample(seat)
}

fn prepare_neural_seat_policy_sample(
    seat: &mut ArenaSeatPolicy,
) -> Result<(FeatureFrame, ActionSpace), PpoError> {
    seat.order_bookkeeping
        .enable_candidate(&seat.persistence)
        .map_err(PpoError::InvalidTransition)?;
    seat.order_bookkeeping
        .reconcile(&seat.tracker, &mut seat.local, &mut seat.pending_active)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    prepare_seat_policy_sample(seat)
}

/// Starts model-facing observation before any actual send; Teacher transport remains legacy.
#[cfg(test)]
fn observe_neural_seat_orders(seat: &mut ArenaSeatPolicy) -> Result<(), PpoError> {
    seat.order_bookkeeping
        .enable_observer(&seat.persistence)
        .map_err(PpoError::InvalidTransition)?;
    seat.order_bookkeeping
        .reconcile(&seat.tracker, &mut seat.local, &mut seat.pending_active)
        .map_err(|error| PpoError::Model(error.to_string()))
}

#[cfg(test)]
fn prepare_neural_observer_sample(
    seat: &mut ArenaSeatPolicy,
) -> Result<(FeatureFrame, ActionSpace), PpoError> {
    observe_neural_seat_orders(seat)?;
    prepare_seat_policy_sample(seat)
}

fn prepare_seat_policy_sample(
    seat: &mut ArenaSeatPolicy,
) -> Result<(FeatureFrame, ActionSpace), PpoError> {
    let space = ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let mut frame = FeatureFrame::new();
    seat.encoder
        .encode(
            &seat.tracker,
            &space,
            &seat.readiness,
            &seat.local,
            &mut frame,
        )
        .map_err(text_error)?;
    Ok((frame, space))
}

fn terminal_outcome(
    environment: &TrainingEnvironment,
    winner: Option<Team>,
) -> Option<PpoTerminalOutcome> {
    let team = environment.seats[environment.policy_seat].tracker.team();
    winner.map(|winner| {
        if winner == Team::Neutral {
            PpoTerminalOutcome::Draw
        } else if winner == team {
            PpoTerminalOutcome::Win
        } else {
            PpoTerminalOutcome::Loss
        }
    })
}

fn map2_reward_end(outcome: Option<PpoTerminalOutcome>) -> Option<Map2RewardEnd> {
    outcome.map(|outcome| match outcome {
        PpoTerminalOutcome::Win => Map2RewardEnd::Win,
        PpoTerminalOutcome::Loss => Map2RewardEnd::Loss,
        PpoTerminalOutcome::Draw => Map2RewardEnd::Draw,
    })
}

fn take_map2_reward(
    environment: &mut TrainingEnvironment,
    end: Option<Map2RewardEnd>,
    ticks: u32,
) -> Result<Map2RewardBreakdown, PpoError> {
    assert_eq!(environment.map, MapId(2));
    assert_eq!(environment.seats.len(), 2);
    assert!(ticks > 0);
    let mut candidate = None;
    for (side, seat) in environment.seats.iter_mut().enumerate() {
        let seat_end = if side == environment.policy_seat {
            end
        } else {
            end.map(|end| match end {
                Map2RewardEnd::Win => Map2RewardEnd::Loss,
                Map2RewardEnd::Loss => Map2RewardEnd::Win,
                Map2RewardEnd::Draw | Map2RewardEnd::TimeCap => end,
            })
        };
        let reward = match seat_end {
            Some(end) => seat.tracker.finish_map2_reward(end),
            None => seat.tracker.take_map2_reward_interval(),
        }
        .map_err(text_error)?;
        if reward.ticks != ticks {
            return Err(PpoError::InvalidTransition(
                "Map2 reward interval tick count",
            ));
        }
        assert_eq!(reward.end, seat_end);
        assert!(reward.total.is_finite());
        if side == environment.policy_seat {
            candidate = Some(reward);
        }
    }
    candidate.ok_or(PpoError::InvalidTransition("Map2 reward candidate seat"))
}

fn bootstrap_values(
    model: &PolicyModel,
    pending: &[PendingTransition],
) -> Result<Vec<f32>, PpoError> {
    let frames = pending
        .iter()
        .filter_map(|transition| transition.next_frame.clone())
        .collect::<Vec<_>>();
    if frames.is_empty() {
        return Ok(vec![0.0; pending.len()]);
    }
    let mut values = model
        .evaluate_batch(&frames)
        .map_err(text_error)?
        .into_iter()
        .map(|output| output.value);
    pending
        .iter()
        .map(|transition| {
            if transition.terminal {
                Ok(0.0)
            } else {
                values
                    .next()
                    .ok_or(PpoError::InvalidTransition("bootstrap value"))
            }
        })
        .collect()
}

fn commit_round(
    environments: &mut [TrainingEnvironment],
    pending: Vec<PendingTransition>,
    bootstrap: Vec<f32>,
    rollout: &mut PpoRollout,
    smoke: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    for (stream, ((environment, pending), next_value)) in environments
        .iter_mut()
        .zip(pending)
        .zip(bootstrap)
        .enumerate()
    {
        if let Some(reward) = pending.map2_reward {
            smoke.map2_reward.record(reward)?;
        }
        match pending.terminal_outcome {
            Some(PpoTerminalOutcome::Win) => {
                smoke.terminal_wins = smoke
                    .terminal_wins
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
            Some(PpoTerminalOutcome::Loss) => {
                smoke.terminal_losses = smoke
                    .terminal_losses
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
            Some(PpoTerminalOutcome::Draw) => {
                smoke.terminal_draws = smoke
                    .terminal_draws
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
            None => {}
        }
        rollout.push(pending.choice.finish(PpoOutcome {
            stream,
            decision: environment.decision,
            ticks: pending.ticks,
            next_value,
            reward: pending.reward,
            terminal: pending.terminal,
        })?)?;
        environment.decision = environment
            .decision
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        smoke.elapsed_ticks = smoke
            .elapsed_ticks
            .checked_add(u64::from(pending.ticks))
            .ok_or(PpoError::CounterOverflow)?;
        if pending.terminal {
            restart_environment(environment)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn sample_policy(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environment: &mut TrainingEnvironment,
) -> Result<PpoPolicyChoice, PpoError> {
    let (frame, space) = prepare_policy_sample(environment)?;
    model.sample(&frame, &space, sampling).map_err(text_error)
}

fn encode_next_frame(environment: &mut TrainingEnvironment) -> Result<FeatureFrame, PpoError> {
    let seat = &mut environment.seats[environment.policy_seat];
    let space = ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let mut frame = FeatureFrame::new();
    seat.encoder
        .encode(
            &seat.tracker,
            &space,
            &seat.readiness,
            &seat.local,
            &mut frame,
        )
        .map_err(text_error)?;
    Ok(frame)
}

fn requests_for_decision_in_space(
    environment: &mut TrainingEnvironment,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
) -> Result<Vec<Option<Request>>, PpoError> {
    let mut requests = Vec::with_capacity(environment.seats.len());
    for index in 0..environment.seats.len() {
        let request = if index == environment.policy_seat {
            policy_request_in_space(&mut environment.seats[index], choice, space)?
        } else {
            opponent_request(&mut environment.seats[index], &mut environment.opponent)?
        };
        requests.push(request);
    }
    Ok(requests)
}

#[cfg(test)]
fn requests_for_neural_greedy_decision(
    environment: &mut TrainingEnvironment,
    model: &PolicyModel,
) -> Result<(Vec<Option<Request>>, ActionKind), PpoError> {
    assert!(environment.policy_seat < environment.seats.len());
    let (frame, space) = prepare_policy_sample(environment)?;
    let choice = model.choose(&frame, &space).map_err(text_error)?;
    assert!(space.allows(choice.action));
    let (action, request) = neural_policy_request_in_space(
        &mut environment.seats[environment.policy_seat],
        choice.action,
        &space,
    )?;
    Ok((requests_with_candidate(environment, request)?, action))
}

fn neural_policy_request_in_space(
    seat: &mut ArenaSeatPolicy,
    proposed: crate::StructuredAction,
    space: &ActionSpace,
) -> Result<(ActionKind, Option<Request>), PpoError> {
    assert!(space.allows(proposed));
    let action = proposed.kind();
    seat.local
        .note_decision(space.tick(), action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(proposed)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let request = issue_request(seat, issued, space, action, false)?;
    Ok((action, request))
}

fn requests_for_batched_greedy_decisions(
    environments: &mut [TrainingEnvironment],
    active: &[usize],
    model: &PolicyModel,
) -> Result<Vec<Vec<Option<Request>>>, PpoError> {
    if active.is_empty() {
        return Ok(Vec::new());
    }
    if active.windows(2).any(|pair| pair[0] >= pair[1])
        || active.iter().any(|index| *index >= environments.len())
    {
        return Err(PpoError::InvalidTransition(
            "batched warmup active environments",
        ));
    }
    requests_for_batched_model_greedy_decisions(environments, active, model)
}

fn requests_for_batched_model_greedy_decisions(
    environments: &mut [TrainingEnvironment],
    active: &[usize],
    model: &PolicyModel,
) -> Result<Vec<Vec<Option<Request>>>, PpoError> {
    let mut frames = Vec::with_capacity(active.len());
    let mut spaces = Vec::with_capacity(active.len());
    for &index in active {
        let (frame, space) = prepare_policy_sample(&mut environments[index])?;
        frames.push(frame);
        spaces.push(space);
    }
    let choices = model.choose_batch(&frames, &spaces).map_err(text_error)?;
    let mut output = Vec::with_capacity(active.len());
    for ((&index, choice), space) in active.iter().zip(choices).zip(&spaces) {
        let environment = &mut environments[index];
        let policy_seat = environment.policy_seat;
        let (_, candidate_request) = neural_policy_request_in_space(
            &mut environment.seats[policy_seat],
            choice.action,
            space,
        )?;
        output.push(requests_with_candidate(environment, candidate_request)?);
    }
    Ok(output)
}

fn requests_with_candidate(
    environment: &mut TrainingEnvironment,
    mut candidate_request: Option<Request>,
) -> Result<Vec<Option<Request>>, PpoError> {
    let policy_seat = environment.policy_seat;
    let mut requests = Vec::with_capacity(environment.seats.len());
    for index in 0..environment.seats.len() {
        let request = if index == policy_seat {
            candidate_request.take()
        } else {
            opponent_request(&mut environment.seats[index], &mut environment.opponent)?
        };
        requests.push(request);
    }
    Ok(requests)
}

fn opponent_request(
    seat: &mut ArenaSeatPolicy,
    opponent: &mut OpponentRuntime,
) -> Result<Option<Request>, PpoError> {
    match opponent {
        OpponentRuntime::Policy { model, rng } => {
            let (frame, space) = prepare_neural_seat_policy_sample(seat)?;
            let choice = model.sample(&frame, &space, rng).map_err(text_error)?;
            policy_request(seat, &choice)
        }
        OpponentRuntime::Teacher => teacher_request(seat),
        OpponentRuntime::Weak => {
            let tick = seat
                .tracker
                .current()
                .ok_or(PpoError::InvalidTransition("weak snapshot"))?
                .tick;
            seat.local
                .note_decision(tick, ActionKind::Continue)
                .map_err(|error| PpoError::Model(error.to_string()))?;
            Ok(None)
        }
    }
}

fn policy_request(
    seat: &mut ArenaSeatPolicy,
    choice: &PpoPolicyChoice,
) -> Result<Option<Request>, PpoError> {
    let space = ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    policy_request_in_space(seat, choice, &space)
}

fn policy_request_in_space(
    seat: &mut ArenaSeatPolicy,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
) -> Result<Option<Request>, PpoError> {
    if !choice.frame.matches_action_space(space) {
        return Err(PpoError::InvalidTransition("prepared actor action space"));
    }
    seat.local
        .note_decision(space.tick(), choice.action.kind())
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(choice.action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    issue_request(seat, issued, space, choice.action.kind(), false)
}

fn teacher_request(seat: &mut ArenaSeatPolicy) -> Result<Option<Request>, PpoError> {
    teacher_request_with_action(seat).map(|(_, request)| request)
}

fn teacher_request_with_action(
    seat: &mut ArenaSeatPolicy,
) -> Result<(ActionKind, Option<Request>), PpoError> {
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let action_kind = action.kind();
    seat.local
        .note_decision(space.tick(), action_kind)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let request = issue_request(seat, issued, &space, action_kind, true)?;
    Ok((action_kind, request))
}

#[cfg(test)]
const fn deployment_uses_teacher(map: MapId) -> bool {
    matches!(map, MapId(0))
}

fn issue_request(
    seat: &mut ArenaSeatPolicy,
    issued: Option<crate::IssuedOrder>,
    space: &ActionSpace,
    action_kind: ActionKind,
    synchronize_teacher: bool,
) -> Result<Option<Request>, PpoError> {
    let persistence = seat.order_bookkeeping.transport(&seat.persistence);
    let Some(issued) = persistence.should_send(issued) else {
        return Ok(None);
    };
    let previous = seat.local.active_order();
    seat.sequence = seat
        .sequence
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    let preserves = seat
        .order_bookkeeping
        .record_sent(&mut seat.persistence, seat.sequence, issued, &seat.tracker)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    seat.readiness.note_sent(seat.sequence, issued, space);
    remember_readiness_order(seat, issued, space.tick());
    let update = if preserves {
        crate::ActiveOrderUpdate::Preserve
    } else {
        crate::active_order_update_for_sent(
            seat.order_bookkeeping.effective(&seat.persistence),
            issued.unit,
            seat.sequence,
            action_kind,
        )
    };
    seat.pending_active = match update {
        crate::ActiveOrderUpdate::Preserve => seat.pending_active,
        crate::ActiveOrderUpdate::Replace(None) if previous.is_none() => None,
        crate::ActiveOrderUpdate::Replace(None) => {
            seat.local
                .set_active_order(space.tick(), None)
                .map_err(|error| PpoError::Model(error.to_string()))?;
            Some((seat.sequence, previous))
        }
        crate::ActiveOrderUpdate::Replace(Some(kind)) => {
            seat.local
                .set_active_order_from_issued(space.tick(), kind, issued)
                .map_err(|error| PpoError::Model(error.to_string()))?;
            Some((seat.sequence, previous))
        }
    };
    if synchronize_teacher {
        seat.teacher.note_sent(seat.sequence, issued, space.tick());
    }
    seat.last_issued = Some((seat.sequence, issued, action_kind));
    Ok(Some(Request {
        seq: seat.sequence,
        unit: issued.unit,
        order: issued.order,
    }))
}

fn remember_readiness_order(seat: &mut ArenaSeatPolicy, issued: crate::IssuedOrder, tick: u32) {
    assert!(seat.readiness_orders.len() <= READINESS_ORDER_HISTORY);
    if matches!(
        issued.order,
        bota_proto::Order::Swap { .. } | bota_proto::Order::Use { .. }
    ) {
        if seat.readiness_orders.len() == READINESS_ORDER_HISTORY {
            seat.readiness_orders.pop_front();
        }
        seat.readiness_orders
            .push_back((seat.sequence, issued, tick));
    }
    assert!(seat.readiness_orders.len() <= READINESS_ORDER_HISTORY);
}

fn advance_interval(
    environment: &mut TrainingEnvironment,
    requests: Vec<Option<Request>>,
    ticks: u32,
) -> Result<ArenaAdvance, PpoError> {
    if ticks == 0 || ticks > episode::TICK_CAP {
        return Err(PpoError::InvalidConfig("arena interval ticks"));
    }
    let mut winner = None;
    let mut elapsed = 0u32;
    for tick in 0..ticks {
        let empty = vec![None; environment.seats.len()];
        let step = environment
            .arena
            .step(if tick == 0 { &requests } else { &empty })
            .map_err(|error| PpoError::Model(error.to_string()))?;
        elapsed = elapsed.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        if step.messages.len() != environment.seats.len() {
            return Err(PpoError::InvalidTransition("arena seat stream count"));
        }
        for (index, (seat, messages)) in environment.seats.iter_mut().zip(step.messages).enumerate()
        {
            let observed = observe_messages_owned(seat, messages)?;
            if index > 0 && observed != winner {
                return Err(PpoError::InvalidTransition(
                    "arena seats disagree on MatchOver",
                ));
            }
            winner = observed;
        }
        if winner.is_some() {
            break;
        }
    }
    Ok(ArenaAdvance {
        winner,
        ticks: elapsed,
    })
}

fn reject_production_rejection(
    environment: &TrainingEnvironment,
    context: &'static str,
) -> Result<(), PpoError> {
    for (index, seat) in environment.seats.iter().enumerate() {
        if let Some((sequence, reason)) = seat.last_rejection {
            return Err(PpoError::Model(format!(
                "{context} seat {index} sequence {sequence} rejected as {reason:?}; tick={:?} mana={:?}; last issued {:?}; readiness orders {:?}",
                seat.tracker.current().map(|view| view.tick),
                seat.tracker.own_hero().map(|hero| hero.mana),
                seat.last_issued,
                seat.readiness_orders,
            )));
        }
    }
    Ok(())
}

fn restart_environment(environment: &mut TrainingEnvironment) -> Result<(), PpoError> {
    let retired = environment
        .retired_rejections
        .checked_add(environment.seats.iter().map(|seat| seat.rejections).sum())
        .ok_or(PpoError::CounterOverflow)?;
    let mut replacement = build_environment(
        environment.next_seed,
        environment.next_opponent_seed,
        environment.map,
        environment.policy_seat,
        environment.decision,
        environment.opponent_spec.clone(),
    )?;
    for (previous, next) in environment.seats.iter().zip(&mut replacement.seats) {
        next.order_bookkeeping = previous.order_bookkeeping.for_new_trajectory();
    }
    replacement.retired_rejections = retired;
    *environment = replacement;
    Ok(())
}

/// Owned variant used by the production tick loop: snapshot views move into
/// the tracker instead of being cloned.
fn observe_messages_owned(
    seat: &mut ArenaSeatPolicy,
    messages: Vec<ServerMsg>,
) -> Result<Option<Team>, PpoError> {
    if seat.tracker.metadata().map == MapId(2) {
        validate_arena_tick_messages(&messages)?;
    }
    let mut winner = None;
    for message in messages {
        match message {
            ServerMsg::OrderRejected { seq, reason } => {
                seat.persistence.observe_rejection(seq);
                seat.order_bookkeeping.observe_rejection(seq);
                seat.readiness.note_rejected(seq);
                seat.teacher.note_rejected(seq);
                if let Some((pending, previous)) = seat.pending_active
                    && pending == seq
                {
                    let tick = seat
                        .tracker
                        .current()
                        .ok_or(PpoError::InvalidTransition(
                            "rejection before arena snapshot",
                        ))?
                        .tick;
                    seat.local
                        .restore_active_order(tick, previous)
                        .map_err(|error| PpoError::Model(error.to_string()))?;
                    seat.pending_active = None;
                }
                seat.rejections = seat
                    .rejections
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
                seat.last_rejection = Some((seq, reason));
            }
            ServerMsg::Snapshot { view } => observe_arena_snapshot_owned(seat, view)?,
            ServerMsg::Events { tick, events } => observe_arena_events(seat, tick, &events)?,
            ServerMsg::MatchOver { winner: result, .. } => winner = Some(result),
            ServerMsg::MatchStart { .. }
            | ServerMsg::Welcome { .. }
            | ServerMsg::LobbyState { .. }
            | ServerMsg::Orders { .. }
            | ServerMsg::ParticipantLeft { .. } => {}
        }
    }
    seat.order_bookkeeping
        .reconcile(&seat.tracker, &mut seat.local, &mut seat.pending_active)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    seat.encoder.observe(&seat.tracker).map_err(text_error)?;
    Ok(winner)
}

fn validate_arena_tick_messages(messages: &[ServerMsg]) -> Result<(), PpoError> {
    let messages = if matches!(messages.first(), Some(ServerMsg::OrderRejected { .. })) {
        &messages[1..]
    } else {
        messages
    };
    let [
        ServerMsg::Snapshot { view },
        ServerMsg::Events { tick, .. },
        terminal @ ..,
    ] = messages
    else {
        return Err(PpoError::InvalidTransition(
            "arena Snapshot/Events/MatchOver ordering",
        ));
    };
    let valid_terminal = match terminal {
        [] => true,
        [ServerMsg::MatchOver { stats, .. }] => stats.duration == *tick,
        _ => false,
    };
    if view.tick != *tick || !valid_terminal {
        return Err(PpoError::InvalidTransition(
            "arena Snapshot/Events/MatchOver ordering",
        ));
    }
    assert!(messages.len() <= 3);
    assert_eq!(view.tick, *tick);
    Ok(())
}

fn observe_arena_events(
    seat: &mut ArenaSeatPolicy,
    tick: u32,
    events: &[EventKind],
) -> Result<(), PpoError> {
    seat.tracker
        .observe_events(tick, events)
        .map_err(|error| PpoError::Model(error.to_string()))
}

/// Owned variant: moves the snapshot into the tracker and keeps its tick for
/// the body-change bookkeeping.
fn observe_arena_snapshot_owned(
    seat: &mut ArenaSeatPolicy,
    view: bota_proto::WorldView,
) -> Result<(), PpoError> {
    let tick = view.tick;
    let previous = seat.tracker.own_hero().map(|hero| hero.id);
    seat.tracker
        .observe_snapshot_owned(view)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let current = seat.tracker.own_hero().map(|hero| hero.id);
    if previous != current {
        seat.persistence.clear_body_for(None);
        seat.order_bookkeeping.clear_body_for(None);
        seat.local
            .set_active_order(tick, None)
            .map_err(|error| PpoError::Model(error.to_string()))?;
        seat.pending_active = None;
        assert!(seat.persistence.active_body_sequence_for(None).is_none());
        assert!(seat.local.active_order().is_none());
    }
    Ok(())
}

pub(crate) fn text_error(error: impl std::fmt::Display) -> PpoError {
    PpoError::Model(error.to_string())
}

fn pipeline_error(error: impl std::fmt::Display) -> PpoError {
    PpoError::Model(format!("actor-learner pipeline: {error}"))
}

#[cfg(test)]
#[path = "tests/training_collection_slice.rs"]
mod training_collection_slice_tests;
