use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::thread;

use bota_proto::{MapId, ServerMsg, SlotId, Team};

use crate::{
    ACTOR_LEARNER_BUFFERS, ActionKind, ActionSpace, ActorLearnerPipeline, Arena, ArenaConfig,
    ArenaStart, CheckpointDevice, CheckpointProgress, CheckpointRun, CheckpointSaveOutcome,
    CrossPlayProfile, FeatureEncoder, FeatureFrame, ItemReadiness, LEAGUE_MIN_PROMOTION_ACTIONS,
    LEAGUE_MIN_PROMOTION_PAIRS, League, LeagueEvaluation, LeagueExploitAudit, LeagueMatchResult,
    LeagueOpponent, LeagueOpponentKind, LeaguePairedResult, LeaguePromotionDecision, LeagueSampler,
    LocalPolicyState, MAX_TRAINING_COUNTER, MODEL_MAX_OPTIMIZER_STEP, OrderPersistence,
    PPO_MAX_POLICY_SAMPLE_DRAWS, PPO_RULES_AUDIT_VERSION, PolicyDevice, PolicyModel,
    PolicySnapshot, PpoConfig, PpoError, PpoOutcome, PpoPolicyChoice, PpoRng, PpoRollout,
    PpoTerminalOutcome, PpoTrainer, PpoUpdateReport, Request, RewardTracker, RngCheckpoint,
    SHADOW_FIEND, StateTracker, Teacher, TrainingArtifact, compiled_features, tick_discount,
};

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
            map: MapId(1),
        }
    }
}

/// Aggregate result of a short real-simulator PPO run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PpoSmokeReport {
    pub updates: u32,
    pub transitions: usize,
    pub optimizer_step: u64,
    pub final_policy_loss: f64,
    pub final_value_loss: f64,
    pub final_entropy: f64,
    pub final_kl: f64,
    pub rejected_orders: u64,
    pub elapsed_ticks: u64,
}

/// Bounded, resumable production PPO settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingJobConfig {
    pub updates: u64,
    pub environments: usize,
    pub rollout_decisions: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub checkpoint_interval: u64,
    pub seed: u64,
    pub map: MapId,
    pub git_commit: String,
    pub simulator_commit: String,
}

/// Durable progress emitted only after a checkpoint and runtime weights are committed.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingCheckpointReport {
    pub completed_updates: u64,
    pub optimizer_step: u64,
    pub rollout_samples: u64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub stopped_for_kl: bool,
    pub cleanup_warning: Option<String>,
}

/// Final state of a bounded production training invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainingJobReport {
    pub completed_updates: u64,
    pub optimizer_step: u64,
    pub rollout_samples: u64,
    pub final_policy_loss: f64,
    pub final_value_loss: f64,
    pub final_entropy: f64,
    pub final_kl: f64,
}

/// Bounded settings for one complete self-play scheduling and evaluation smoke run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeagueSmokeConfig {
    pub updates: u32,
    pub environments: usize,
    pub rollout_decisions: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub evaluation_pairs: usize,
    pub evaluation_decisions: usize,
    pub seed: u64,
    pub map: MapId,
}

impl Default for LeagueSmokeConfig {
    fn default() -> Self {
        Self {
            updates: 1,
            environments: 4,
            rollout_decisions: 8,
            epochs: 1,
            minibatch: 32,
            evaluation_pairs: 2,
            evaluation_decisions: 8,
            seed: 10_001,
            map: MapId(1),
        }
    }
}

/// Aggregate result of bounded league training and held-out paired evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LeagueSmokeReport {
    pub ppo: PpoSmokeReport,
    pub opponent_counts: [u32; 5],
    pub paired_evaluations: usize,
    pub profile_evaluations: usize,
    pub exploit_evaluations: usize,
    pub evaluation_actions: u64,
    pub evaluation_rejections: u64,
    pub league_policies: usize,
    pub promotions: u32,
    pub accepted_before: u64,
    pub accepted_after: u64,
}

struct ArenaSeatPolicy {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    readiness: ItemReadiness,
    teacher: Teacher,
    sequence: u32,
    rejections: u64,
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
    ticks: u32,
    terminal: bool,
    next_frame: Option<FeatureFrame>,
}

struct EvaluationMatch {
    result: LeagueMatchResult,
    actions: u64,
    rejections: u64,
}

struct TrainingDirectoryLock {
    _file: File,
}

/// Runs real seat-projected arenas, GAE, clipped PPO, value loss, entropy, and Adam briefly.
pub fn run_ppo_smoke(settings: PpoSmokeConfig) -> Result<PpoSmokeReport, PpoError> {
    run_ppo_smoke_on(settings, PolicyDevice::Cpu)
}

/// Runs the complete actor-to-learner path on one selected backend.
pub fn run_ppo_smoke_on(
    settings: PpoSmokeConfig,
    device: PolicyDevice,
) -> Result<PpoSmokeReport, PpoError> {
    validate_smoke(settings)?;
    let config = smoke_ppo_config(settings).validate()?;
    let model = PolicyModel::fresh_on(settings.seed, device).map_err(model_error)?;
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
            smoke.rejected_orders = actor_report.rejected_orders;
            smoke.elapsed_ticks = smoke
                .elapsed_ticks
                .checked_add(actor_report.elapsed_ticks)
                .ok_or(PpoError::CounterOverflow)?;
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

/// Runs bounded PPO updates and commits exact, resumable state at fixed intervals.
pub fn run_training_job_on<F>(
    settings: TrainingJobConfig,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    checkpointed: F,
) -> Result<TrainingJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport) + Send + 'static,
{
    let directory = checkpoint_directory.to_path_buf();
    thread::Builder::new()
        .name("drysua-training".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || run_training_job_inner(settings, device, &directory, resume, checkpointed))
        .map_err(|error| PpoError::Model(format!("training worker spawn failed: {error}")))?
        .join()
        .map_err(|_| PpoError::Model("training worker panicked".to_owned()))?
}

fn run_training_job_inner<F>(
    settings: TrainingJobConfig,
    device: PolicyDevice,
    checkpoint_directory: &Path,
    resume: bool,
    mut checkpointed: F,
) -> Result<TrainingJobReport, PpoError>
where
    F: FnMut(TrainingCheckpointReport),
{
    validate_training_directory(checkpoint_directory, resume)?;
    let _directory_lock = TrainingDirectoryLock::acquire(checkpoint_directory)?;
    let config = validate_training_job(&settings)?;
    let run = training_checkpoint_run(&settings, device, config)?;
    let capacity = settings
        .environments
        .checked_mul(settings.rollout_decisions)
        .ok_or(PpoError::InvalidConfig("training samples"))?;
    let mut session = TrainingSession::initialize(
        &settings,
        device,
        checkpoint_directory,
        resume,
        config,
        run,
        capacity,
    )?;
    if resume {
        TrainingArtifact::save_runtime_weights(&session.model, checkpoint_directory)
            .map_err(checkpoint_error)?;
    }
    let start = session.completed_updates;
    if start > settings.updates {
        return Err(PpoError::InvalidConfig(
            "training update target precedes checkpoint",
        ));
    }
    for update in start..settings.updates {
        let report = session.train_update(&settings, update, config, capacity)?;
        if report.completed_updates % settings.checkpoint_interval == 0
            || report.completed_updates == settings.updates
        {
            let durable = session.save(checkpoint_directory, report)?;
            checkpointed(durable);
        }
    }
    Ok(session.report())
}

struct TrainingSession {
    model: PolicyModel,
    trainer: PpoTrainer,
    pipeline: ActorLearnerPipeline,
    actor: crate::ActorRolloutSender,
    sampling: PpoRng,
    run: CheckpointRun,
    completed_updates: u64,
    rollout_samples: u64,
    latest: PpoUpdateReport,
}

impl TrainingSession {
    #[allow(clippy::too_many_arguments)]
    fn initialize(
        settings: &TrainingJobConfig,
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        config: PpoConfig,
        run: CheckpointRun,
        capacity: usize,
    ) -> Result<Self, PpoError> {
        let model = PolicyModel::fresh_on(settings.seed, device).map_err(model_error)?;
        let (trainer, mut pipeline, sampling, completed_updates, rollout_samples) = if resume {
            restore_training_session(&model, directory, &run, config, capacity)?
        } else {
            let trainer = PpoTrainer::new(&model, config, settings.seed ^ 0x51a9)?;
            let pipeline =
                ActorLearnerPipeline::new(capacity, 1, &model).map_err(pipeline_error)?;
            (trainer, pipeline, PpoRng::new(settings.seed ^ 0xa17e), 0, 0)
        };
        let actor = pipeline.take_actor(0).map_err(pipeline_error)?;
        Ok(Self {
            model,
            trainer,
            pipeline,
            actor,
            sampling,
            run,
            completed_updates,
            rollout_samples,
            latest: PpoUpdateReport::default(),
        })
    }

    fn train_update(
        &mut self,
        settings: &TrainingJobConfig,
        update: u64,
        config: PpoConfig,
        capacity: usize,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        let mut environments = build_training_environments(settings, update, config)?;
        let (lease, mut rollout) = self.actor.wait_rollout().map_err(pipeline_error)?;
        let mut actor_report = PpoSmokeReport::default();
        collect_update(
            lease.policy(),
            &mut self.sampling,
            &mut environments,
            config,
            settings.rollout_decisions,
            &mut rollout,
            &mut actor_report,
        )?;
        lease.try_submit(rollout).map_err(pipeline_error)?;
        let batch = self
            .pipeline
            .accept()
            .map_err(pipeline_error)?
            .finish(config)?;
        self.latest = self.trainer.train_pipeline_update(&self.model, &batch)?;
        self.pipeline.publish(&self.model).map_err(pipeline_error)?;
        self.completed_updates = self.trainer.updates();
        self.rollout_samples = self
            .rollout_samples
            .checked_add(capacity as u64)
            .ok_or(PpoError::CounterOverflow)?;
        Ok(self.checkpoint_report(None))
    }

    fn save(
        &self,
        directory: &Path,
        report: TrainingCheckpointReport,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        let (state, draws) = self.sampling.checkpoint();
        let progress = CheckpointProgress {
            global_update: self.completed_updates,
            policy_version: self.pipeline.version().map_err(pipeline_error)?.get(),
            scheduler_step: self.completed_updates,
            curriculum_stage: 0,
            rollout_samples: self.rollout_samples,
            best_evaluation: None,
            rng_states: vec![
                RngCheckpoint::new("ppo_actor_sampling", state, draws).map_err(checkpoint_error)?,
            ],
            league_references: Vec::new(),
        };
        let artifact =
            TrainingArtifact::capture(&self.model, &self.trainer, self.run.clone(), progress)
                .map_err(checkpoint_error)?;
        let outcome = artifact.save(directory).map_err(checkpoint_error)?;
        TrainingArtifact::save_runtime_weights(&self.model, directory).map_err(checkpoint_error)?;
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
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            policy_loss: self.latest.policy_loss,
            value_loss: self.latest.value_loss,
            entropy: self.latest.entropy,
            approximate_kl: update_kl(self.latest),
            stopped_for_kl: self.latest.stopped_for_kl,
            cleanup_warning,
        }
    }

    fn report(&self) -> TrainingJobReport {
        TrainingJobReport {
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            final_policy_loss: self.latest.policy_loss,
            final_value_loss: self.latest.value_loss,
            final_entropy: self.latest.entropy,
            final_kl: update_kl(self.latest),
        }
    }
}

/// Runs bounded self-play PPO, frozen-opponent scheduling, and held-out paired evaluation.
pub fn run_league_smoke(settings: LeagueSmokeConfig) -> Result<LeagueSmokeReport, PpoError> {
    run_league_smoke_on(settings, PolicyDevice::Cpu)
}

/// Runs league training with CPU actors and one selected learner backend.
pub fn run_league_smoke_on(
    settings: LeagueSmokeConfig,
    device: PolicyDevice,
) -> Result<LeagueSmokeReport, PpoError> {
    validate_league_smoke(settings)?;
    let ppo_settings = league_ppo_settings(settings);
    let config = smoke_ppo_config(ppo_settings).validate()?;
    let model = PolicyModel::fresh_on(settings.seed, device).map_err(model_error)?;
    let accepted = PolicySnapshot::capture(&model, 0).map_err(league_error)?;
    let accepted_before = accepted.fingerprint();
    let mut league = League::new(32, accepted).map_err(league_error)?;
    let mut scheduler = LeagueSampler::new(settings.seed ^ 0x1ea9);
    let mut opponent_rng = PpoRng::new(settings.seed ^ 0x6f70_706f_6e65_6e74);
    let mut trainer = PpoTrainer::new(&model, config, settings.seed ^ 0x51a9)?;
    let capacity = settings
        .environments
        .checked_mul(settings.rollout_decisions)
        .ok_or(PpoError::InvalidConfig("league samples"))?;
    let mut pipeline = ActorLearnerPipeline::new(capacity, 1, &model).map_err(pipeline_error)?;
    let actor = pipeline.take_actor(0).map_err(pipeline_error)?;
    let (job_sender, job_receiver) = sync_channel::<Vec<TrainingEnvironment>>(1);
    let (report_sender, report_receiver) = sync_channel(ACTOR_LEARNER_BUFFERS);
    let worker = thread::Builder::new()
        .name("drysua-league-actor".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || -> Result<(), PpoError> {
            let mut sampling = PpoRng::new(settings.seed ^ 0xa17e);
            for _ in 0..settings.updates {
                let mut environments = job_receiver.recv().map_err(|_| {
                    PpoError::Model("league actor job channel disconnected".to_owned())
                })?;
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
                    .map_err(|_| PpoError::Model("league actor report disconnected".to_owned()))?;
            }
            Ok(())
        })
        .map_err(|error| PpoError::Model(format!("league actor worker spawn failed: {error}")))?;
    let mut training_seeds = Vec::with_capacity(settings.updates as usize * settings.environments);
    let mut report = LeagueSmokeReport {
        accepted_before,
        ..LeagueSmokeReport::default()
    };
    let initial = PolicySnapshot::capture(&model, 0).map_err(league_error)?;
    let environments = build_league_environments(
        settings,
        0,
        &initial,
        &league,
        &mut scheduler,
        &mut opponent_rng,
        &mut training_seeds,
        &mut report,
    )?;
    job_sender
        .send(environments)
        .map_err(|_| PpoError::Model("league actor job channel disconnected".to_owned()))?;
    let learner_result = (|| -> Result<(), PpoError> {
        for update in 0..settings.updates {
            let batch = pipeline.accept().map_err(pipeline_error)?.finish(config)?;
            let actor_report = report_receiver.recv().map_err(|_| {
                PpoError::Model("league actor report channel disconnected".to_owned())
            })?;
            merge_actor_report(&mut report.ppo, actor_report)?;
            if update + 1 < settings.updates {
                let next = update + 1;
                let current =
                    PolicySnapshot::capture(&model, u64::from(next)).map_err(league_error)?;
                let environments = build_league_environments(
                    settings,
                    next,
                    &current,
                    &league,
                    &mut scheduler,
                    &mut opponent_rng,
                    &mut training_seeds,
                    &mut report,
                )?;
                job_sender.send(environments).map_err(|_| {
                    PpoError::Model("league actor job channel disconnected".to_owned())
                })?;
            }
            let update_report = trainer.train_pipeline_update(&model, &batch)?;
            record_update(&mut report.ppo, capacity, update_report)?;
            pipeline.publish(&model).map_err(pipeline_error)?;
            evaluate_and_retain(
                &model,
                settings,
                update,
                &training_seeds,
                &mut league,
                &mut report,
            )?;
        }
        Ok(())
    })();
    drop(job_sender);
    drop(pipeline);
    drop(report_receiver);
    let worker_result = worker
        .join()
        .map_err(|_| PpoError::Model("league actor worker panicked".to_owned()))?;
    learner_result?;
    worker_result?;
    report.league_policies = league.len();
    report.accepted_after = league.accepted().fingerprint();
    Ok(report)
}

fn merge_actor_report(
    aggregate: &mut PpoSmokeReport,
    actor: PpoSmokeReport,
) -> Result<(), PpoError> {
    aggregate.rejected_orders = aggregate
        .rejected_orders
        .checked_add(actor.rejected_orders)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.elapsed_ticks = aggregate
        .elapsed_ticks
        .checked_add(actor.elapsed_ticks)
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

fn update_kl(update: PpoUpdateReport) -> f64 {
    if update.stopped_for_kl {
        update.rejected_kl
    } else {
        update.approximate_kl
    }
}

fn validate_training_job(settings: &TrainingJobConfig) -> Result<PpoConfig, PpoError> {
    if settings.updates == 0 || settings.updates > 1_000_000 {
        return Err(PpoError::InvalidConfig("training updates"));
    }
    if settings.checkpoint_interval == 0 || settings.checkpoint_interval > 1_000 {
        return Err(PpoError::InvalidConfig("training checkpoint interval"));
    }
    if settings.environments == 0 || settings.environments > 16 {
        return Err(PpoError::InvalidConfig("training environments"));
    }
    if settings.rollout_decisions == 0 || settings.rollout_decisions > 64 {
        return Err(PpoError::InvalidConfig("training rollout decisions"));
    }
    if !matches!(settings.map, MapId(0) | MapId(1)) {
        return Err(PpoError::InvalidConfig("training map"));
    }
    if settings.git_commit.is_empty() || settings.git_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("training git commit"));
    }
    if settings.simulator_commit.is_empty() || settings.simulator_commit.len() > 4_096 {
        return Err(PpoError::InvalidConfig("training simulator commit"));
    }
    let config = training_ppo_config(settings).validate()?;
    validate_training_counters(settings, config)?;
    Ok(config)
}

fn training_ppo_config(settings: &TrainingJobConfig) -> PpoConfig {
    PpoConfig {
        environments: settings.environments,
        rollout_decisions: settings.rollout_decisions,
        epochs: settings.epochs,
        minibatch: settings.minibatch,
        ..PpoConfig::default()
    }
}

fn validate_training_counters(
    settings: &TrainingJobConfig,
    config: PpoConfig,
) -> Result<(), PpoError> {
    let samples = (settings.environments as u64)
        .checked_mul(settings.rollout_decisions as u64)
        .ok_or(PpoError::InvalidConfig("training sample counter"))?;
    let actor_draws = settings
        .updates
        .checked_mul(samples)
        .and_then(|count| count.checked_mul(PPO_MAX_POLICY_SAMPLE_DRAWS))
        .ok_or(PpoError::InvalidConfig("training actor RNG counter"))?;
    if actor_draws > MAX_TRAINING_COUNTER {
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

fn training_checkpoint_run(
    settings: &TrainingJobConfig,
    device: PolicyDevice,
    config: PpoConfig,
) -> Result<CheckpointRun, PpoError> {
    let device_name = match device {
        PolicyDevice::Cpu => "cpu",
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        PolicyDevice::Cuda { .. } => "cuda",
        #[cfg(all(feature = "metal", target_os = "macos"))]
        PolicyDevice::Metal { .. } => "metal",
    };
    let command_line = format!(
        "train-full --environments {} --rollout {} --epochs {} --minibatch {} --seed {} --map {} --device {device_name}",
        settings.environments,
        settings.rollout_decisions,
        settings.epochs,
        settings.minibatch,
        settings.seed,
        settings.map.0,
    );
    Ok(CheckpointRun {
        git_commit: settings.git_commit.clone(),
        simulator_commit: settings.simulator_commit.clone(),
        enabled_features: compiled_features(),
        command_line,
        run_seed: settings.seed,
        map: settings.map,
        hero: SHADOW_FIEND,
        device: CheckpointDevice::from_policy(device).map_err(checkpoint_error)?,
        batch_size: config.minibatch,
        rules_audit_version: PPO_RULES_AUDIT_VERSION,
    })
}

fn restore_training_session(
    model: &PolicyModel,
    directory: &Path,
    run: &CheckpointRun,
    config: PpoConfig,
    capacity: usize,
) -> Result<(PpoTrainer, ActorLearnerPipeline, PpoRng, u64, u64), PpoError> {
    let artifact = TrainingArtifact::load_compatible(directory, run).map_err(checkpoint_error)?;
    let restored = artifact.restore(model, run).map_err(checkpoint_error)?;
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
    let completed_updates = progress.global_update;
    let rollout_samples = progress.rollout_samples;
    let pipeline = restored
        .pipeline(capacity, 1, model)
        .map_err(checkpoint_error)?;
    let (trainer, _, _) = restored.into_parts();
    Ok((
        trainer,
        pipeline,
        sampling,
        completed_updates,
        rollout_samples,
    ))
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
    if !matches!(settings.map, MapId(0) | MapId(1)) {
        return Err(PpoError::InvalidConfig("smoke map"));
    }
    Ok(())
}

fn validate_league_smoke(settings: LeagueSmokeConfig) -> Result<(), PpoError> {
    validate_smoke(league_ppo_settings(settings))?;
    if settings.evaluation_pairs == 0 || settings.evaluation_pairs > 64 {
        return Err(PpoError::InvalidConfig("league evaluation pairs"));
    }
    if settings.evaluation_decisions == 0 || settings.evaluation_decisions > 1_024 {
        return Err(PpoError::InvalidConfig("league evaluation decisions"));
    }
    settings
        .seed
        .checked_add(1u64 << 40)
        .ok_or(PpoError::InvalidConfig("league evaluation seed"))?;
    Ok(())
}

const fn league_ppo_settings(settings: LeagueSmokeConfig) -> PpoSmokeConfig {
    PpoSmokeConfig {
        updates: settings.updates,
        environments: settings.environments,
        rollout_decisions: settings.rollout_decisions,
        epochs: settings.epochs,
        minibatch: settings.minibatch,
        seed: settings.seed,
        map: settings.map,
    }
}

fn smoke_ppo_config(settings: PpoSmokeConfig) -> PpoConfig {
    PpoConfig {
        environments: settings.environments,
        rollout_decisions: settings.rollout_decisions,
        epochs: settings.epochs,
        minibatch: settings.minibatch,
        target_kl: 1.0,
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
            (settings.seed ^ 0x6f70_706f_6e65_6e74)
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
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    let decision = update
        .checked_mul(settings.rollout_decisions as u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(PpoError::InvalidConfig("training decision counter"))?;
    let offset = update
        .checked_mul(settings.environments as u64)
        .ok_or(PpoError::CounterOverflow)?;
    let mut environments = Vec::with_capacity(settings.environments);
    for index in 0..settings.environments {
        let stream = offset
            .checked_add(index as u64)
            .ok_or(PpoError::CounterOverflow)?;
        let mut environment = build_environment(
            derive_training_seed(settings.seed, stream, 0x6172_656e_615f_7365),
            derive_training_seed(settings.seed, stream, 0x6f70_706f_6e65_6e74),
            settings.map,
            training_policy_seat(stream),
            decision,
            OpponentSpec::Teacher,
        )?;
        warmup_environment(
            &mut environment,
            training_warmup_decisions(stream),
            config.decision_interval_ticks,
        )?;
        environments.push(environment);
    }
    Ok(environments)
}

pub(crate) const fn derive_training_seed(base: u64, stream: u64, domain: u64) -> u64 {
    let mut value = base ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) const fn training_policy_seat(stream: u64) -> usize {
    (stream % 2) as usize
}

pub(crate) const fn training_warmup_decisions(phase: u64) -> usize {
    const PHASES: [usize; 8] = [0, 300, 450, 600, 900, 1_200, 1_800, 2_400];
    PHASES[(phase % PHASES.len() as u64) as usize]
}

fn warmup_environment(
    environment: &mut TrainingEnvironment,
    decisions: usize,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    for _ in 0..decisions {
        let requests = environment
            .seats
            .iter_mut()
            .map(teacher_request)
            .collect::<Result<Vec<_>, _>>()?;
        let advanced = advance_interval(environment, requests, decision_interval_ticks)?;
        if advanced.winner.is_some() {
            restart_environment(environment)?;
        }
    }
    environment.reward = RewardTracker::default();
    let summary = environment.seats[environment.policy_seat]
        .tracker
        .latest_summary()
        .ok_or(PpoError::InvalidTransition("warmup summary"))?;
    environment.reward.observe(summary, 1.0, None)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_league_environments(
    settings: LeagueSmokeConfig,
    update: u32,
    current: &PolicySnapshot,
    league: &League,
    scheduler: &mut LeagueSampler,
    opponent_rng: &mut PpoRng,
    training_seeds: &mut Vec<u64>,
    report: &mut LeagueSmokeReport,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    let mut environments = Vec::with_capacity(settings.environments);
    let offset = update as usize * settings.environments;
    for index in 0..settings.environments {
        let seed = settings
            .seed
            .checked_add((offset + index) as u64)
            .ok_or(PpoError::InvalidConfig("league training seed"))?;
        let opponent = scheduler.sample(league, current).map_err(league_error)?;
        report.opponent_counts[opponent.kind().index()] = report.opponent_counts
            [opponent.kind().index()]
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
        training_seeds.push(seed);
        environments.push(build_environment(
            seed,
            opponent_rng.next_word()?,
            settings.map,
            (offset + index) % 2,
            update * settings.rollout_decisions as u32,
            opponent_spec(&opponent)?,
        )?);
    }
    Ok(environments)
}

fn evaluate_and_retain(
    model: &PolicyModel,
    settings: LeagueSmokeConfig,
    update: u32,
    training_seeds: &[u64],
    league: &mut League,
    report: &mut LeagueSmokeReport,
) -> Result<(), PpoError> {
    let candidate = PolicySnapshot::capture(model, u64::from(update) + 1).map_err(league_error)?;
    let accepted = league.accepted().clone();
    let (pairs, actions, rejections) = evaluate_pairs(model, &accepted, settings, update)?;
    let (profile, profile_pairs, profile_actions, profile_rejections) =
        evaluate_cross_play_profile(model, &candidate, league, &pairs, settings, update)?;
    record_evaluation(
        report,
        pairs.len(),
        profile_pairs,
        actions
            .checked_add(profile_actions)
            .ok_or(PpoError::CounterOverflow)?,
        rejections
            .checked_add(profile_rejections)
            .ok_or(PpoError::CounterOverflow)?,
    )?;
    let score = paired_score(&pairs);
    if league.contains(candidate.fingerprint()) {
        return Ok(());
    }
    if pairs.len() < LEAGUE_MIN_PROMOTION_PAIRS
        || actions < LEAGUE_MIN_PROMOTION_ACTIONS
        || rejections.saturating_mul(1_000) >= actions
    {
        return league
            .insert_historical(candidate, score, profile)
            .map_err(league_error);
    }
    let promotion_seeds = pairs.iter().map(|pair| pair.seed).collect::<Vec<_>>();
    let (exploit_audit, exploit_actions, exploit_rejections) = evaluate_exploit_audit(
        model,
        &candidate,
        &accepted,
        settings,
        update,
        training_seeds,
        &promotion_seeds,
    )?;
    record_exploit_evaluation(report, exploit_actions, exploit_rejections)?;
    let Some(exploit_audit) = exploit_audit else {
        return league
            .insert_historical(candidate, score, profile)
            .map_err(league_error);
    };
    let evidence = LeagueEvaluation::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        pairs,
        training_seeds,
        rejections,
        actions,
        profile,
        exploit_audit,
    )
    .map_err(league_error)?;
    if league
        .try_promote(candidate, evidence)
        .map_err(league_error)?
        == LeaguePromotionDecision::Accepted
    {
        report.promotions = report
            .promotions
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
    }
    Ok(())
}

fn record_evaluation(
    report: &mut LeagueSmokeReport,
    promotion_pairs: usize,
    profile_pairs: usize,
    actions: u64,
    rejections: u64,
) -> Result<(), PpoError> {
    report.paired_evaluations = report
        .paired_evaluations
        .checked_add(promotion_pairs)
        .ok_or(PpoError::CounterOverflow)?;
    report.profile_evaluations = report
        .profile_evaluations
        .checked_add(profile_pairs)
        .ok_or(PpoError::CounterOverflow)?;
    report.evaluation_actions = report
        .evaluation_actions
        .checked_add(actions)
        .ok_or(PpoError::CounterOverflow)?;
    report.evaluation_rejections = report
        .evaluation_rejections
        .checked_add(rejections)
        .ok_or(PpoError::CounterOverflow)?;
    Ok(())
}

fn record_exploit_evaluation(
    report: &mut LeagueSmokeReport,
    actions: u64,
    rejections: u64,
) -> Result<(), PpoError> {
    report.exploit_evaluations = report
        .exploit_evaluations
        .checked_add(2)
        .ok_or(PpoError::CounterOverflow)?;
    report.evaluation_actions = report
        .evaluation_actions
        .checked_add(actions)
        .ok_or(PpoError::CounterOverflow)?;
    report.evaluation_rejections = report
        .evaluation_rejections
        .checked_add(rejections)
        .ok_or(PpoError::CounterOverflow)?;
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
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map,
        seed,
    })
    .map_err(|error| PpoError::Model(error.to_string()))?;
    let seats = setup_seats(start)?;
    let mut reward = RewardTracker::default();
    reward.observe(
        seats[policy_seat]
            .tracker
            .latest_summary()
            .ok_or(PpoError::InvalidTransition("initial summary"))?,
        1.0,
        None,
    )?;
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

fn build_opponent(spec: &OpponentSpec, seed: u64) -> Result<OpponentRuntime, PpoError> {
    match spec {
        OpponentSpec::Policy(snapshot) => Ok(OpponentRuntime::Policy {
            model: Arc::new(
                snapshot
                    .instantiate()
                    .map_err(|error| PpoError::Model(error.to_string()))?,
            ),
            rng: PpoRng::new(seed),
        }),
        OpponentSpec::SharedPolicy(model) => Ok(OpponentRuntime::Policy {
            model: Arc::clone(model),
            rng: PpoRng::new(seed),
        }),
        OpponentSpec::Teacher => Ok(OpponentRuntime::Teacher),
        OpponentSpec::Weak => Ok(OpponentRuntime::Weak),
    }
}

fn opponent_spec(opponent: &LeagueOpponent) -> Result<OpponentSpec, PpoError> {
    match opponent.kind() {
        LeagueOpponentKind::CurrentMirror
        | LeagueOpponentKind::Accepted
        | LeagueOpponentKind::Historical => opponent
            .snapshot()
            .cloned()
            .map(OpponentSpec::Policy)
            .ok_or(PpoError::InvalidTransition("model opponent snapshot")),
        LeagueOpponentKind::Teacher => Ok(OpponentSpec::Teacher),
        LeagueOpponentKind::Weak => Ok(OpponentSpec::Weak),
    }
}

fn evaluate_pairs(
    candidate: &PolicyModel,
    accepted: &PolicySnapshot,
    settings: LeagueSmokeConfig,
    update: u32,
) -> Result<(Vec<LeaguePairedResult>, u64, u64), PpoError> {
    let base = evaluation_seed_base(settings.seed, update)?;
    let mut pairs = Vec::with_capacity(settings.evaluation_pairs);
    let opponent =
        OpponentSpec::SharedPolicy(Arc::new(accepted.instantiate().map_err(league_error)?));
    let mut actor_rng =
        PpoRng::new(settings.seed ^ 0x6576_616c_6163_746f ^ (u64::from(update) << 32));
    let mut actions = 0u64;
    let mut rejections = 0u64;
    for index in 0..settings.evaluation_pairs {
        let seed = base
            .checked_add(index as u64)
            .ok_or(PpoError::InvalidConfig("league evaluation seed"))?;
        let evaluated = evaluate_pair(candidate, &opponent, settings, seed, &mut actor_rng)?;
        actions = actions
            .checked_add(evaluated.1)
            .ok_or(PpoError::CounterOverflow)?;
        rejections = rejections
            .checked_add(evaluated.2)
            .ok_or(PpoError::CounterOverflow)?;
        pairs.push(evaluated.0);
    }
    Ok((pairs, actions, rejections))
}

fn evaluate_cross_play_profile(
    candidate_model: &PolicyModel,
    candidate: &PolicySnapshot,
    league: &League,
    accepted_pairs: &[LeaguePairedResult],
    settings: LeagueSmokeConfig,
    update: u32,
) -> Result<(CrossPlayProfile, usize, u64, u64), PpoError> {
    let base = evaluation_seed_base(settings.seed, update)?;
    let mut actor_rng =
        PpoRng::new(settings.seed ^ 0x6372_6f73_7370_6c79 ^ (u64::from(update) << 32));
    let pair_count = accepted_pairs.len().min(2);
    let teacher = evaluate_profile_pairs(
        candidate_model,
        &OpponentSpec::Teacher,
        settings,
        base.checked_add(128)
            .ok_or(PpoError::InvalidConfig("teacher evaluation seed"))?,
        pair_count,
        &mut actor_rng,
    )?;
    let historical = league.select_bucket(55, candidate).map_err(league_error)?;
    let history = evaluate_profile_pairs(
        candidate_model,
        &opponent_spec(&historical)?,
        settings,
        base.checked_add(192)
            .ok_or(PpoError::InvalidConfig("history evaluation seed"))?,
        pair_count,
        &mut actor_rng,
    )?;
    let profile = CrossPlayProfile::new(
        paired_score(&teacher.0) as f32,
        paired_score(&accepted_pairs[..pair_count]) as f32,
        paired_score(&history.0) as f32,
    )
    .map_err(league_error)?;
    let actions = teacher
        .1
        .checked_add(history.1)
        .ok_or(PpoError::CounterOverflow)?;
    let rejections = teacher
        .2
        .checked_add(history.2)
        .ok_or(PpoError::CounterOverflow)?;
    Ok((profile, pair_count * 2, actions, rejections))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_exploit_audit(
    candidate_model: &PolicyModel,
    candidate: &PolicySnapshot,
    accepted: &PolicySnapshot,
    settings: LeagueSmokeConfig,
    update: u32,
    training_seeds: &[u64],
    promotion_seeds: &[u64],
) -> Result<(Option<LeagueExploitAudit>, u64, u64), PpoError> {
    let base = evaluation_seed_base(settings.seed, update)?
        .checked_add(256)
        .ok_or(PpoError::InvalidConfig("exploit evaluation seed"))?;
    let mut actor_rng =
        PpoRng::new(settings.seed ^ 0x6578_706c_6f69_7421 ^ (u64::from(update) << 32));
    let opponent =
        OpponentSpec::SharedPolicy(Arc::new(accepted.instantiate().map_err(league_error)?));
    let evaluated = evaluate_profile_pairs(
        candidate_model,
        &opponent,
        settings,
        base,
        2,
        &mut actor_rng,
    )?;
    let audit = match LeagueExploitAudit::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        &evaluated.0,
        training_seeds,
        promotion_seeds,
        evaluated.2,
        evaluated.1,
    ) {
        Ok(audit) => Some(audit),
        Err(crate::LeagueError::ExploitRegression | crate::LeagueError::InvalidRejectionRate) => {
            None
        }
        Err(error) => return Err(league_error(error)),
    };
    Ok((audit, evaluated.1, evaluated.2))
}

fn evaluate_profile_pairs(
    candidate: &PolicyModel,
    opponent: &OpponentSpec,
    settings: LeagueSmokeConfig,
    first_seed: u64,
    pair_count: usize,
    actor_rng: &mut PpoRng,
) -> Result<(Vec<LeaguePairedResult>, u64, u64), PpoError> {
    let mut pairs = Vec::with_capacity(pair_count);
    let mut actions = 0u64;
    let mut rejections = 0u64;
    for index in 0..pair_count {
        let seed = first_seed
            .checked_add(index as u64)
            .ok_or(PpoError::InvalidConfig("profile evaluation seed"))?;
        let evaluated = evaluate_pair(candidate, opponent, settings, seed, actor_rng)?;
        pairs.push(evaluated.0);
        actions = actions
            .checked_add(evaluated.1)
            .ok_or(PpoError::CounterOverflow)?;
        rejections = rejections
            .checked_add(evaluated.2)
            .ok_or(PpoError::CounterOverflow)?;
    }
    Ok((pairs, actions, rejections))
}

fn evaluation_seed_base(seed: u64, update: u32) -> Result<u64, PpoError> {
    seed.checked_add(1u64 << 40)
        .and_then(|seed| seed.checked_add(u64::from(update) * 1_024))
        .ok_or(PpoError::InvalidConfig("league evaluation seed"))
}

fn evaluate_pair(
    candidate: &PolicyModel,
    opponent: &OpponentSpec,
    settings: LeagueSmokeConfig,
    seed: u64,
    actor_rng: &mut PpoRng,
) -> Result<(LeaguePairedResult, u64, u64), PpoError> {
    let candidate_seed = actor_rng.next_word()?;
    let opponent_seed = actor_rng.next_word()?;
    let radiant = evaluate_match(
        candidate,
        opponent,
        settings,
        seed,
        candidate_seed,
        opponent_seed,
        0,
    )?;
    let dire = evaluate_match(
        candidate,
        opponent,
        settings,
        seed,
        candidate_seed,
        opponent_seed,
        1,
    )?;
    let actions = radiant
        .actions
        .checked_add(dire.actions)
        .ok_or(PpoError::CounterOverflow)?;
    let rejections = radiant
        .rejections
        .checked_add(dire.rejections)
        .ok_or(PpoError::CounterOverflow)?;
    Ok((
        LeaguePairedResult {
            seed,
            candidate_radiant: radiant.result,
            candidate_dire: dire.result,
        },
        actions,
        rejections,
    ))
}

fn evaluate_match(
    candidate: &PolicyModel,
    opponent: &OpponentSpec,
    settings: LeagueSmokeConfig,
    seed: u64,
    candidate_seed: u64,
    opponent_seed: u64,
    candidate_seat: usize,
) -> Result<EvaluationMatch, PpoError> {
    let mut environment = build_environment(
        seed,
        opponent_seed,
        settings.map,
        candidate_seat,
        0,
        opponent.clone(),
    )?;
    let mut sampling = PpoRng::new(candidate_seed);
    let mut winner = None;
    let mut actions = 0u64;
    for _ in 0..settings.evaluation_decisions {
        let choice = sample_policy(candidate, &mut sampling, &mut environment)?;
        let requests = requests_for_decision(&mut environment, &choice)?;
        let advanced = advance_interval(&mut environment, requests, 3)?;
        actions = actions.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        if advanced.winner.is_some() {
            winner = advanced.winner;
            break;
        }
    }
    let result = evaluation_result(&environment, winner)?;
    let rejections = environment.seats[candidate_seat].rejections;
    Ok(EvaluationMatch {
        result,
        actions,
        rejections,
    })
}

fn evaluation_result(
    environment: &TrainingEnvironment,
    winner: Option<Team>,
) -> Result<LeagueMatchResult, PpoError> {
    let seat = &environment.seats[environment.policy_seat];
    if let Some(winner) = winner {
        return Ok(if winner == seat.tracker.team() {
            LeagueMatchResult::Win
        } else {
            LeagueMatchResult::Loss
        });
    }
    Ok(LeagueMatchResult::Draw)
}

#[allow(
    clippy::float_arithmetic,
    reason = "paired held-out scores are normalized by the bounded game count"
)]
fn paired_score(pairs: &[LeaguePairedResult]) -> f64 {
    let score = pairs
        .iter()
        .map(|pair| pair_score(pair.candidate_radiant) + pair_score(pair.candidate_dire))
        .sum::<i32>();
    f64::from(score) / (pairs.len() * 2) as f64
}

const fn pair_score(result: LeagueMatchResult) -> i32 {
    match result {
        LeagueMatchResult::Win => 1,
        LeagueMatchResult::Draw => 0,
        LeagueMatchResult::Loss => -1,
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
    tracker
        .observe_snapshot(snapshot.ok_or(PpoError::InvalidTransition("initial snapshot"))?)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).map_err(feature_error)?;
    Ok(ArenaSeatPolicy {
        tracker,
        encoder,
        local: LocalPolicyState::new(1),
        persistence: OrderPersistence::default(),
        readiness: ItemReadiness::new(),
        teacher: Teacher::new(),
        sequence: 0,
        rejections: 0,
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
    for _ in 0..decisions {
        let pending = collect_round(model, sampling, environments, config)?;
        let bootstrap = bootstrap_values(model, &pending)?;
        commit_round(environments, pending, bootstrap, rollout, smoke)?;
    }
    smoke.rejected_orders = environments
        .iter()
        .map(|environment| {
            environment.retired_rejections
                + environment
                    .seats
                    .iter()
                    .map(|seat| seat.rejections)
                    .sum::<u64>()
        })
        .sum();
    Ok(())
}

fn collect_round(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environments: &mut [TrainingEnvironment],
    config: PpoConfig,
) -> Result<Vec<PendingTransition>, PpoError> {
    let mut pending = Vec::with_capacity(environments.len());
    for environment in environments {
        let choice = sample_policy(model, sampling, environment)?;
        let requests = requests_for_decision(environment, &choice)?;
        let advanced = advance_interval(environment, requests, config.decision_interval_ticks)?;
        let terminal = advanced.winner.is_some();
        let outcome = terminal_outcome(environment, advanced.winner);
        let summary = environment.seats[environment.policy_seat]
            .tracker
            .latest_summary()
            .ok_or(PpoError::InvalidTransition("next summary"))?;
        let discount = tick_discount(config.gamma_tick, advanced.ticks)?;
        let reward = environment
            .reward
            .observe(summary, discount, outcome)?
            .total;
        let next_frame = (!terminal)
            .then(|| encode_next_frame(environment))
            .transpose()?;
        pending.push(PendingTransition {
            choice,
            reward,
            ticks: advanced.ticks,
            terminal,
            next_frame,
        });
    }
    Ok(pending)
}

fn terminal_outcome(
    environment: &TrainingEnvironment,
    winner: Option<Team>,
) -> Option<PpoTerminalOutcome> {
    let team = environment.seats[environment.policy_seat].tracker.team();
    winner.map(|winner| {
        if winner == team {
            PpoTerminalOutcome::Win
        } else {
            PpoTerminalOutcome::Loss
        }
    })
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
        .map_err(model_error)?
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

fn sample_policy(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environment: &mut TrainingEnvironment,
) -> Result<PpoPolicyChoice, PpoError> {
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
        .map_err(feature_error)?;
    model.sample(&frame, &space, sampling).map_err(model_error)
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
        .map_err(feature_error)?;
    Ok(frame)
}

fn requests_for_decision(
    environment: &mut TrainingEnvironment,
    choice: &PpoPolicyChoice,
) -> Result<Vec<Option<Request>>, PpoError> {
    let mut requests = Vec::with_capacity(environment.seats.len());
    for index in 0..environment.seats.len() {
        let request = if index == environment.policy_seat {
            policy_request(&mut environment.seats[index], choice)?
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
                .map_err(feature_error)?;
            let choice = model.sample(&frame, &space, rng).map_err(model_error)?;
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
    seat.local
        .note_decision(space.tick(), choice.action.kind())
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(choice.action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    issue_request(seat, issued, &space, None)
}

fn teacher_request(seat: &mut ArenaSeatPolicy) -> Result<Option<Request>, PpoError> {
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    issue_request(seat, issued, &space, Some(action.kind()))
}

fn issue_request(
    seat: &mut ArenaSeatPolicy,
    issued: Option<crate::IssuedOrder>,
    space: &ActionSpace,
    teacher_kind: Option<ActionKind>,
) -> Result<Option<Request>, PpoError> {
    let Some(issued) = seat.persistence.should_send(issued) else {
        return Ok(None);
    };
    seat.sequence = seat
        .sequence
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    seat.persistence
        .record_sent(seat.sequence, issued)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    seat.readiness.note_sent(seat.sequence, issued, space);
    if teacher_kind.is_some() {
        seat.teacher.note_sent(seat.sequence, issued, space.tick());
    }
    Ok(Some(Request {
        seq: seat.sequence,
        unit: issued.unit,
        order: issued.order,
    }))
}

fn advance_interval(
    environment: &mut TrainingEnvironment,
    requests: Vec<Option<Request>>,
    ticks: u32,
) -> Result<ArenaAdvance, PpoError> {
    let mut winner = None;
    let mut elapsed = 0u32;
    for tick in 0..ticks {
        let empty = vec![None; environment.seats.len()];
        let step = environment
            .arena
            .step(if tick == 0 { &requests } else { &empty })
            .map_err(|error| PpoError::Model(error.to_string()))?;
        elapsed = elapsed.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        for (seat, messages) in environment.seats.iter_mut().zip(step.messages) {
            winner = observe_messages(seat, &messages)?.or(winner);
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
    replacement.retired_rejections = retired;
    *environment = replacement;
    Ok(())
}

fn observe_messages(
    seat: &mut ArenaSeatPolicy,
    messages: &[ServerMsg],
) -> Result<Option<Team>, PpoError> {
    let mut winner = None;
    for message in messages {
        match message {
            ServerMsg::OrderRejected { seq, .. } => {
                seat.persistence.observe_rejection(*seq);
                seat.readiness.note_rejected(*seq);
                seat.teacher.note_rejected(*seq);
                seat.rejections = seat
                    .rejections
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
            ServerMsg::Snapshot { view } => seat
                .tracker
                .observe_snapshot(view)
                .map_err(|error| PpoError::Model(error.to_string()))?,
            ServerMsg::Events { tick, events } => seat
                .tracker
                .observe_events(*tick, events)
                .map_err(|error| PpoError::Model(error.to_string()))?,
            ServerMsg::MatchOver { winner: result, .. } => winner = Some(*result),
            ServerMsg::MatchStart { .. }
            | ServerMsg::Welcome { .. }
            | ServerMsg::LobbyState { .. }
            | ServerMsg::Orders { .. }
            | ServerMsg::ParticipantLeft { .. } => {}
        }
    }
    seat.encoder.observe(&seat.tracker).map_err(feature_error)?;
    Ok(winner)
}

fn model_error(error: crate::ModelError) -> PpoError {
    PpoError::Model(error.to_string())
}

fn checkpoint_error(error: crate::CheckpointError) -> PpoError {
    PpoError::Model(error.to_string())
}

fn pipeline_error(error: impl std::fmt::Display) -> PpoError {
    PpoError::Model(format!("actor-learner pipeline: {error}"))
}

fn feature_error(error: crate::FeatureError) -> PpoError {
    PpoError::Model(error.to_string())
}

fn league_error(error: crate::LeagueError) -> PpoError {
    PpoError::Model(error.to_string())
}
