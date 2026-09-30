use std::collections::VecDeque;
mod annealed;
pub(crate) use annealed::preflight_annealed_resume;
pub(crate) use annealed::validate_annealed;
pub use annealed::{
    AnnealedJobConfig, AnnealedJobReport, run_annealed_job_on_with_initial_weights,
};
mod collector;
pub(crate) mod collector_state;
pub(crate) mod episode;
mod evaluation;
pub(crate) use evaluation::{EvaluationOpponent, EvaluationSettings, run_evaluation};
mod game_summary;
mod lane;
mod pool;
mod reward;
mod slot;
pub use reward::Map2TrainingReward;
#[cfg(test)]
#[path = "tests/ppo_arena_test_support.rs"]
mod test_support;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::Duration;
#[cfg(test)]
pub(crate) use test_support::*;

use bota_proto::{EventKind, MapId, RejectReason, ServerMsg, SlotId, Team};
use bota_server::game::SpawnModifier;

use crate::persistence::training::PolicyOrderBookkeeping;

use crate::{
    ActionKind, ActionSpace, ActionTarget, ActivePolicyOrder, Arena, ArenaConfig, ArenaStart,
    BehavioralTarget, CheckpointProgress, CheckpointRun, CheckpointSaveOutcome,
    CollectionCheckpoint, ControlledUnit, EntityIndex, FeatureEncoder, FeatureFrame, ItemReadiness,
    LocalPolicyState, LootIndex, OrderPersistence, PointIndex, PolicyDevice, PolicyModel,
    PpoConfig, PpoError, PpoRng, PpoTerminalOutcome, PpoTrainer, PpoTransition, PpoUpdateReport,
    PutPointTarget, Request, ScriptKind, ScriptedPolicy, ShopIndex, StateTracker, StructuredAction,
    TrainingArtifact, tick_discount,
};
use crate::{MAP2_REWARD_GAMMA_TICK, Map2RewardBreakdown, Map2RewardEnd};

const READINESS_ORDER_HISTORY: usize = 32;

/// Gameplay counters of one collected update batch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CollectionReport {
    pub map2_reward: Map2TrainingReward,
    pub episode_timeouts: u64,
    pub rejected_orders: u64,
    pub elapsed_ticks: u64,
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
}

/// A deterministic test cadence or a production monotonic wall-time cadence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingCheckpointCadence {
    Updates(u64),
    WallTime(Duration),
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

struct ArenaSeatPolicy {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    order_bookkeeping: PolicyOrderBookkeeping,
    aim: crate::RazeAim,
    readiness: ItemReadiness,
    script: ScriptedPolicy,
    combat: game_summary::SeatCombat,
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
    map: MapId,
    opponent: OpponentRuntime,
}

/// Who decides the opponent seat's orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpponentRuntime {
    /// A policy whose actions the caller supplies each decision.
    Neural,
    Teacher,
    HarassPush,
    /// Test fixture that always continues, keeping native worlds deterministic.
    #[cfg(test)]
    Idle,
}

struct ArenaAdvance {
    winner: Option<Team>,
    ticks: u32,
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

/// Session counters merged from actor reports and read by checkpoints.
///
/// Both training sessions keep the same set, so merging and reporting live
/// here once.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SessionCounters {
    pub(crate) map2_reward: Map2TrainingReward,
    pub(crate) episode_timeouts: u64,
    pub(crate) terminal_wins: u64,
    pub(crate) terminal_losses: u64,
    pub(crate) terminal_draws: u64,
    pub(crate) rejected_orders: u64,
    pub(crate) elapsed_ticks: u64,
}

impl SessionCounters {
    /// Adds one actor report's counters into the running totals.
    pub(crate) fn merge(&mut self, report: &CollectionReport) -> Result<(), PpoError> {
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

/// The device name recorded in a run scope.
pub(crate) fn device_name(device: PolicyDevice) -> &'static str {
    match device {
        PolicyDevice::Cpu => "cpu",
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        PolicyDevice::Cuda { .. } => "cuda",
    }
}

/// Checkpoint-owned learner state: model, optimizer, progress and collection.
struct TrainingSession {
    adaptive_environment: Option<crate::AdaptiveEnvironmentCheckpoint>,
    counters: SessionCounters,
    model: PolicyModel,
    trainer: PpoTrainer,
    /// Collection state the next update starts from; required by every save.
    collection: Option<CollectionCheckpoint>,
    run: CheckpointRun,
    completed_updates: u64,
    rollout_samples: u64,
    latest: PpoUpdateReport,
    starting_policy_fingerprint: u64,
}

impl TrainingSession {
    fn initialize(
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        initial_weights_directory: Option<&Path>,
        config: PpoConfig,
        run: CheckpointRun,
    ) -> Result<Self, PpoError> {
        let model = match (resume, initial_weights_directory) {
            (false, Some(directory)) => {
                TrainingArtifact::initialize_from_weights(directory, run.run_seed, device)
                    .map_err(text_error)?
            }
            _ => PolicyModel::fresh_on(run.run_seed, device).map_err(text_error)?,
        };
        let restored = if resume {
            restore_training_session(&model, directory, &run, config)?
        } else {
            let trainer = PpoTrainer::new(&model, config, run.run_seed ^ 0x51a9)?;
            RestoredTrainingSession {
                adaptive_environment: None,
                trainer,
                collection: None,
                completed_updates: 0,
                rollout_samples: 0,
            }
        };
        let starting_policy_fingerprint = model.parameter_fingerprint().map_err(text_error)?;
        Ok(Self {
            adaptive_environment: restored.adaptive_environment,
            model,
            trainer: restored.trainer,
            counters: SessionCounters::default(),
            collection: restored.collection,
            run,
            completed_updates: restored.completed_updates,
            rollout_samples: restored.rollout_samples,
            latest: PpoUpdateReport::default(),
            starting_policy_fingerprint,
        })
    }

    fn save(
        &self,
        directory: &Path,
        report: TrainingCheckpointReport,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        let collection = self
            .collection
            .clone()
            .ok_or(PpoError::InvalidTransition("checkpoint collection state"))?;
        let progress = CheckpointProgress {
            adaptive_environment: self.adaptive_environment,
            global_update: self.completed_updates,
            policy_version: self.completed_updates,
            scheduler_step: self.completed_updates,
            curriculum_stage: 0,
            rollout_samples: self.rollout_samples,
            best_evaluation: None,
            rng_states: Vec::new(),
            league_references: Vec::new(),
        };
        let artifact = TrainingArtifact::capture(
            &self.model,
            &self.trainer,
            self.run.clone(),
            progress,
            collection,
        )
        .map_err(text_error)?;
        let outcome = artifact.save(directory).map_err(text_error)?;
        TrainingArtifact::save_runtime_weights(&self.model, directory).map_err(text_error)?;
        let cleanup_warning = match outcome {
            CheckpointSaveOutcome::Committed => None,
            CheckpointSaveOutcome::CommittedWithCleanupError(message) => Some(message),
        };
        Ok(TrainingCheckpointReport {
            cleanup_warning,
            ..report
        })
    }

    fn checkpoint_report(&self, cleanup_warning: Option<String>) -> TrainingCheckpointReport {
        TrainingCheckpointReport {
            map2_reward: self.counters.map2_reward,
            episode_timeouts: self.counters.episode_timeouts,
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            policy_loss: self.latest.policy_loss,
            value_loss: self.latest.value_loss,
            entropy: self.latest.entropy,
            approximate_kl: update_kl(self.latest),
            stopped_for_kl: self.latest.stopped_for_kl,
            terminal_wins: self.counters.terminal_wins,
            terminal_losses: self.counters.terminal_losses,
            terminal_draws: self.counters.terminal_draws,
            rejected_orders: self.counters.rejected_orders,
            elapsed_ticks: self.counters.elapsed_ticks,
            cleanup_warning,
        }
    }
}

pub(crate) fn update_kl(update: PpoUpdateReport) -> f64 {
    if update.stopped_for_kl {
        update.rejected_kl
    } else {
        update.approximate_kl
    }
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

fn restore_training_session(
    model: &PolicyModel,
    directory: &Path,
    run: &CheckpointRun,
    config: PpoConfig,
) -> Result<RestoredTrainingSession, PpoError> {
    let artifact = TrainingArtifact::load_compatible(directory, run).map_err(text_error)?;
    if artifact.config() != config {
        return Err(PpoError::InvalidConfig("training checkpoint PPO config"));
    }
    let restored = artifact.restore(model, run).map_err(text_error)?;
    let progress = restored.progress();
    if progress.policy_version != progress.global_update {
        return Err(PpoError::InvalidConfig(
            "training checkpoint policy version",
        ));
    }
    let progress = progress.clone();
    let (trainer, _, _) = restored.into_parts();
    Ok(RestoredTrainingSession {
        adaptive_environment: progress.adaptive_environment,
        trainer,
        collection: Some(artifact.collection().clone()),
        completed_updates: progress.global_update,
        rollout_samples: progress.rollout_samples,
    })
}

struct RestoredTrainingSession {
    adaptive_environment: Option<crate::AdaptiveEnvironmentCheckpoint>,
    trainer: PpoTrainer,
    collection: Option<CollectionCheckpoint>,
    completed_updates: u64,
    rollout_samples: u64,
}

pub(crate) use crate::randomization::derive_training_seed;

/// Builds one environment whose units carry trusted spawn modifiers.
///
/// The annealed loop puts one generation's rules on the units their selectors
/// name at world construction and on every later spawn.
fn build_environment(
    seed: u64,
    map: MapId,
    policy_seat: usize,
    opponent: OpponentRuntime,
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
    let mut seats = setup_seats(start)?;
    if opponent == OpponentRuntime::HarassPush {
        seats[1 - policy_seat].script = ScriptedPolicy::new(ScriptKind::HarassPush);
    }
    Ok(TrainingEnvironment {
        arena,
        seats,
        policy_seat,
        map,
        opponent,
    })
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
        aim: crate::RazeAim::default(),
        readiness: ItemReadiness::new(),
        script: ScriptedPolicy::new(ScriptKind::Teacher),
        combat: game_summary::SeatCombat::default(),
        sequence: 0,
        rejections: 0,
        pending_active: None,
        readiness_orders: VecDeque::with_capacity(READINESS_ORDER_HISTORY),
        last_issued: None,
        last_rejection: None,
    })
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

/// The request of a scripted opponent seat; neural seats supply their actions.
fn scripted_request(
    seat: &mut ArenaSeatPolicy,
    runtime: OpponentRuntime,
) -> Result<Option<Request>, PpoError> {
    match runtime {
        OpponentRuntime::Teacher | OpponentRuntime::HarassPush => teacher_request(seat),
        OpponentRuntime::Neural => Err(PpoError::InvalidTransition(
            "neural opponent without a decision",
        )),
        #[cfg(test)]
        OpponentRuntime::Idle => {
            let tick = seat
                .tracker
                .current()
                .ok_or(PpoError::InvalidTransition("idle snapshot"))?
                .tick;
            seat.local
                .note_decision(tick, ActionKind::Continue)
                .map_err(|error| PpoError::Model(error.to_string()))?;
            Ok(None)
        }
    }
}

fn teacher_request(seat: &mut ArenaSeatPolicy) -> Result<Option<Request>, PpoError> {
    teacher_request_with_action(seat).map(|(_, request)| request)
}

fn teacher_request_with_action(
    seat: &mut ArenaSeatPolicy,
) -> Result<(ActionKind, Option<Request>), PpoError> {
    let (action, space) = seat
        .script
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

fn issue_request(
    seat: &mut ArenaSeatPolicy,
    issued: Option<crate::IssuedOrder>,
    space: &ActionSpace,
    action_kind: ActionKind,
    synchronize_teacher: bool,
) -> Result<Option<Request>, PpoError> {
    let active_body = seat
        .order_bookkeeping
        .effective(&seat.persistence)
        .active_body_order_for(None);
    let Some((issued, action_kind)) =
        seat.aim
            .resolve(&seat.tracker, issued, action_kind, active_body)
    else {
        return Ok(None);
    };
    let persistence = seat.order_bookkeeping.transport(&seat.persistence);
    let Some(issued) = persistence.should_send(Some(issued)) else {
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
        seat.script.note_sent(seat.sequence, issued, space.tick());
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
                seat.script.note_rejected(seq);
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
        .map_err(|error| PpoError::Model(error.to_string()))?;
    seat.combat.observe(&seat.tracker, events);
    Ok(())
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
