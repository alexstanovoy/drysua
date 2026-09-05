use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{Duration, Instant};

use bota_proto::{EventKind, MapId, RejectReason, ServerMsg, SlotId, Team};

use crate::{
    ACTOR_LEARNER_BUFFERS, ActionKind, ActionSpace, ActivePolicyOrder, ActorLearnerPipeline,
    AdamConfig, Arena, ArenaConfig, ArenaStart, BehavioralTrainer, CheckpointDevice,
    CheckpointProgress, CheckpointRun, CheckpointSaveOutcome, CrossPlayProfile,
    EarlyStoppingConfig, FeatureEncoder, FeatureFrame, IMITATION_RULES_AUDIT_VERSION,
    ImitationPool, ImitationSide, ItemReadiness, LEAGUE_MIN_PROMOTION_ACTIONS,
    LEAGUE_MIN_PROMOTION_PAIRS, League, LeagueEvaluation, LeagueExploitAudit, LeagueMatchResult,
    LeagueOpponent, LeagueOpponentKind, LeaguePairedResult, LeaguePromotionDecision, LeagueSampler,
    LocalPolicyState, MAX_TRAINING_COUNTER, MODEL_MAX_OPTIMIZER_STEP, OfflineEvaluation,
    OrderPersistence, PPO_MAX_POLICY_SAMPLE_DRAWS, PPO_MAX_ROLLOUT_DECISIONS,
    PPO_RULES_AUDIT_VERSION, PolicyDevice, PolicyModel, PolicySnapshot, PpoConfig, PpoError,
    PpoOutcome, PpoPolicyChoice, PpoRng, PpoRollout, PpoTerminalOutcome, PpoTrainer,
    PpoUpdateReport, Request, RewardTracker, RngCheckpoint, SHADOW_FIEND, SampleIdentity,
    SeedNamespace, SeedNamespaces, StateTracker, Teacher, TeacherCoverage, TrainingArtifact,
    TrainingScope, compiled_features, tick_discount,
};

const TRAINING_MAX_ENVIRONMENTS: usize = 16;
const READINESS_ORDER_HISTORY: usize = 32;
const _: () =
    assert!(TRAINING_MAX_ENVIRONMENTS * PPO_MAX_ROLLOUT_DECISIONS <= crate::PPO_MAX_SAMPLES);

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
    pub terminal_wins: u64,
    pub terminal_losses: u64,
    pub terminal_draws: u64,
}

/// Bounded, resumable production PPO settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingJobConfig {
    pub updates: u64,
    pub environments: usize,
    pub rollout_decisions: usize,
    pub epochs: usize,
    pub minibatch: usize,
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

/// Fixed held-out checkpoint evaluation settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointEvaluationConfig {
    pub pairs: usize,
    pub decisions: usize,
    pub seed: u64,
}

/// Independent opponent used by one fixed evaluation cohort.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointEvaluationBaseline {
    Teacher,
    Weak,
}

/// A timeout is distinct from an authoritative simulator result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointEvaluationOutcome {
    Win,
    Loss,
    Timeout,
}

/// Seat-visible activity and gameplay result for one deterministic match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointEvaluationGame {
    pub map: MapId,
    pub baseline: CheckpointEvaluationBaseline,
    pub seed: u64,
    pub candidate_team: Team,
    pub outcome: CheckpointEvaluationOutcome,
    pub decisions: u32,
    pub wire_orders: u32,
    pub rejected_orders: u32,
    pub baseline_wire_orders: u32,
    pub baseline_rejected_orders: u32,
    pub elapsed_ticks: u32,
    pub action_counts: [u32; ActionKind::COUNT],
    pub final_summary: crate::GlobalSummary,
}

/// Complete fixed matrix over both maps, baselines, sides, and paired seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointEvaluationReport {
    pub fingerprint: u64,
    pub games: Vec<CheckpointEvaluationGame>,
}

/// Hard failures that prevent a checkpoint from continuing to train or ship.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointEvaluationQuality {
    pub passed: bool,
    pub win_games: usize,
    pub timeout_games: usize,
    pub idle_games: usize,
    pub collapsed_games: usize,
    pub baseline_failure_games: usize,
    pub weak_loss_games: usize,
    pub weak_stalled_games: usize,
    pub rejected_orders: u64,
}

/// Bounded teacher behavioral-cloning bootstrap settings for both supported maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BehavioralPretrainingConfig {
    pub epochs: u32,
    pub seed: u64,
}

/// Dataset, optimizer, agreement, and artifact result from teacher pretraining.
#[derive(Clone, Debug, PartialEq)]
pub struct BehavioralPretrainingReport {
    pub training_samples: usize,
    pub validation_samples: usize,
    pub held_out_samples: usize,
    pub optimizer_steps: u64,
    pub final_loss: f64,
    pub held_out_kind_agreement: f64,
    pub held_out_full_agreement: f64,
    pub action_counts: [u64; ActionKind::COUNT],
    pub gameplay_validation_map_zero_progress: usize,
    pub gameplay_validation_map_one_wins: usize,
    pub gameplay_validation_failures: usize,
    pub fingerprint: u64,
}

impl CheckpointEvaluationReport {
    pub fn quality(&self) -> CheckpointEvaluationQuality {
        let win_games = self
            .games
            .iter()
            .filter(|game| game.outcome == CheckpointEvaluationOutcome::Win)
            .count();
        let timeout_games = self
            .games
            .iter()
            .filter(|game| game.outcome == CheckpointEvaluationOutcome::Timeout)
            .count();
        let idle_games = self
            .games
            .iter()
            .filter(|game| game.wire_orders == 0)
            .count();
        let collapsed_games = self
            .games
            .iter()
            .filter(|game| action_distribution_collapsed(game))
            .count();
        let baseline_failure_games = self
            .games
            .iter()
            .filter(|game| {
                game.baseline == CheckpointEvaluationBaseline::Teacher
                    && (game.baseline_wire_orders == 0 || game.baseline_rejected_orders > 0)
            })
            .count();
        let weak_loss_games = self
            .games
            .iter()
            .filter(|game| {
                game.baseline == CheckpointEvaluationBaseline::Weak
                    && game.outcome == CheckpointEvaluationOutcome::Loss
            })
            .count();
        let weak_stalled_games = self
            .games
            .iter()
            .filter(|game| {
                game.baseline == CheckpointEvaluationBaseline::Weak
                    && game.outcome == CheckpointEvaluationOutcome::Timeout
                    && game.final_summary.enemy_structures_destroyed == 0
            })
            .count();
        let rejected_orders = self
            .games
            .iter()
            .map(|game| u64::from(game.rejected_orders))
            .sum();
        CheckpointEvaluationQuality {
            passed: !self.games.is_empty()
                && win_games > 0
                && timeout_games < self.games.len()
                && idle_games == 0
                && collapsed_games == 0
                && baseline_failure_games == 0
                && weak_loss_games == 0
                && weak_stalled_games == 0
                && rejected_orders == 0,
            win_games,
            timeout_games,
            idle_games,
            collapsed_games,
            baseline_failure_games,
            weak_loss_games,
            weak_stalled_games,
            rejected_orders,
        }
    }
}

fn action_distribution_collapsed(game: &CheckpointEvaluationGame) -> bool {
    let maximum = game.action_counts.iter().copied().max().unwrap_or(0);
    u64::from(maximum).saturating_mul(100) >= u64::from(game.decisions).saturating_mul(95)
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
    terminal_outcome: Option<PpoTerminalOutcome>,
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

/// Loads deployment weights and evaluates every map, baseline, side, and paired seed.
pub fn evaluate_runtime_checkpoint(
    settings: CheckpointEvaluationConfig,
    checkpoint_directory: &Path,
) -> Result<CheckpointEvaluationReport, PpoError> {
    validate_checkpoint_evaluation(settings)?;
    let model = PolicyModel::fresh(settings.seed).map_err(model_error)?;
    TrainingArtifact::load_runtime_weights(&model, checkpoint_directory)
        .map_err(checkpoint_error)?;
    let fingerprint = PolicySnapshot::capture(&model, 0)
        .map_err(league_error)?
        .fingerprint();
    let match_count = settings
        .pairs
        .checked_mul(8)
        .ok_or(PpoError::InvalidConfig("checkpoint evaluation matches"))?;
    let mut games = Vec::with_capacity(match_count);
    for map in [MapId(0), MapId(1)] {
        for baseline in [
            CheckpointEvaluationBaseline::Teacher,
            CheckpointEvaluationBaseline::Weak,
        ] {
            for pair in 0..settings.pairs {
                let seed = settings
                    .seed
                    .checked_add(pair as u64)
                    .ok_or(PpoError::InvalidConfig("checkpoint evaluation seed"))?;
                for candidate_seat in 0..2 {
                    games.push(evaluate_checkpoint_game(
                        &model,
                        settings,
                        map,
                        baseline,
                        seed,
                        candidate_seat,
                    )?);
                }
            }
        }
    }
    Ok(CheckpointEvaluationReport { fingerprint, games })
}

const PRETRAINING_TRAIN_SEEDS: usize = 4;
const PRETRAINING_DAGGER_SEEDS: usize = 4;
const PRETRAINING_VALIDATION_SEEDS: usize = 2;
const PRETRAINING_HELD_OUT_SEEDS: usize = 2;
const PRETRAINING_TRAINING_MAP: MapId = MapId(1);
const PRETRAINING_DECISIONS: usize = 4_096;
const PRETRAINING_WINDOWS: usize = 4;
const PRETRAINING_WINDOW_DECISIONS: usize = PRETRAINING_DECISIONS / PRETRAINING_WINDOWS;
const PRETRAINING_DAGGER_DECISIONS: usize = PRETRAINING_DECISIONS;
const PRETRAINING_CONTINUE_STRIDE: usize = 128;
const PRETRAINING_EFFECTIVE_BATCH: usize = 64;
const PRETRAINING_TRAIN_CONTINUE_WINDOW_CAP: u64 = 48;
const PRETRAINING_TRAIN_OTHER_WINDOW_CAP: u64 = 6;
const PRETRAINING_HELD_OUT_CONTINUE_WINDOW_CAP: u64 = 21;
const PRETRAINING_HELD_OUT_OTHER_WINDOW_CAP: u64 = 4;
const PRETRAINING_DAGGER_WINDOW_SIDE_KIND_CAP: u64 = 3;
const PRETRAINING_GAMEPLAY_DECISIONS: usize = 4_096;
const PRETRAINING_GAMEPLAY_VALIDATION_SEEDS: [u64; 3] = [9_100_001, 9_100_002, 9_100_003];
const PRETRAINING_GAMEPLAY_ACCEPTANCE_SEED: u64 = 9_000_001;
const PRETRAINING_OVERALL_KIND_PERCENT: usize = 60;
const PRETRAINING_OVERALL_FULL_PERCENT: usize = 55;
const PRETRAINING_SIDE_KIND_PERCENT: usize = 55;
const PRETRAINING_SIDE_FULL_PERCENT: usize = 50;

const PRETRAINING_MAX_BASE_TRAIN_SAMPLES: usize = PRETRAINING_TRAIN_SEEDS
    * 2
    * PRETRAINING_WINDOWS
    * (PRETRAINING_TRAIN_CONTINUE_WINDOW_CAP as usize
        + (ActionKind::COUNT - 1) * PRETRAINING_TRAIN_OTHER_WINDOW_CAP as usize);
const PRETRAINING_MAX_HELD_OUT_SAMPLES: usize = PRETRAINING_HELD_OUT_SEEDS
    * 2
    * PRETRAINING_WINDOWS
    * (PRETRAINING_HELD_OUT_CONTINUE_WINDOW_CAP as usize
        + (ActionKind::COUNT - 1) * PRETRAINING_HELD_OUT_OTHER_WINDOW_CAP as usize);
const PRETRAINING_MAX_VALIDATION_SAMPLES: usize = PRETRAINING_VALIDATION_SEEDS
    * 2
    * PRETRAINING_WINDOWS
    * (PRETRAINING_HELD_OUT_CONTINUE_WINDOW_CAP as usize
        + (ActionKind::COUNT - 1) * PRETRAINING_HELD_OUT_OTHER_WINDOW_CAP as usize);
const PRETRAINING_MAX_DAGGER_SAMPLES: usize = PRETRAINING_DAGGER_SEEDS
    * 2
    * PRETRAINING_WINDOWS
    * ActionKind::COUNT
    * PRETRAINING_DAGGER_WINDOW_SIDE_KIND_CAP as usize;
const _: () = assert!(PRETRAINING_DECISIONS.is_multiple_of(PRETRAINING_WINDOWS));
const PRETRAINING_SAMPLE_CAPACITY: usize = PRETRAINING_MAX_BASE_TRAIN_SAMPLES
    + PRETRAINING_MAX_VALIDATION_SAMPLES
    + PRETRAINING_MAX_HELD_OUT_SAMPLES
    + PRETRAINING_MAX_DAGGER_SAMPLES;
const _: () = assert!(PRETRAINING_SAMPLE_CAPACITY <= crate::MAX_IMITATION_SAMPLES);

struct PretrainingCollection {
    pool: ImitationPool,
    validation_coverage: TeacherCoverage,
    held_out_coverage: TeacherCoverage,
    training_action_counts: [u64; ActionKind::COUNT],
    split_counts: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PretrainingAgreement {
    kind_matching: usize,
    kind_total: usize,
    full_matching: usize,
    full_total: usize,
    radiant_kind: crate::AgreementCount,
    dire_kind: crate::AgreementCount,
    families: [crate::AgreementCount; ActionKind::COUNT],
}

struct PretrainingCandidate {
    agreement: PretrainingAgreement,
    gameplay: PretrainingGameplayMatrix,
    parameters: Vec<f32>,
    optimizer_steps: u64,
    loss: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PretrainingGameplay {
    pub(crate) games: usize,
    pub(crate) failures: usize,
    pub(crate) wins: usize,
    pub(crate) structure_progress_games: usize,
    pub(crate) structures: u64,
    pub(crate) deaths: u64,
    pub(crate) rejections: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PretrainingGameplayMatrix {
    map_zero: PretrainingGameplay,
    map_one: PretrainingGameplay,
}

/// Collects paired teacher trajectories, clones behavior, gates agreement, and saves weights.
pub fn run_behavioral_pretraining_on(
    settings: BehavioralPretrainingConfig,
    device: PolicyDevice,
    output_directory: &Path,
) -> Result<BehavioralPretrainingReport, PpoError> {
    validate_behavioral_pretraining(settings)?;
    validate_training_directory(output_directory, false)?;
    let _directory_lock = TrainingDirectoryLock::acquire(output_directory)?;
    let mut collection = collect_pretraining_samples(settings)?;
    validate_action_diversity(&collection.training_action_counts)?;
    let model = PolicyModel::fresh_on(settings.seed, device).map_err(model_error)?;
    let mut trainer = BehavioralTrainer::new(
        PRETRAINING_EFFECTIVE_BATCH,
        settings.seed ^ 0x7368_7566_666c_6521,
        AdamConfig {
            learning_rate: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            gradient_clip: 0.5,
        },
        EarlyStoppingConfig::default(),
        &model,
        &collection.pool,
    )
    .map_err(imitation_error)?;
    let bootstrap_epochs = settings.epochs.div_ceil(2);
    let mut stage_loss =
        train_behavioral_epochs(&model, &collection.pool, &mut trainer, bootstrap_epochs)?;
    let mut validation = evaluate_pretraining_validation(&model, &collection)?;
    let mut agreement_trace = Vec::with_capacity(PRETRAINING_DAGGER_SEEDS + 1);
    let mut gameplay_trace = Vec::with_capacity(PRETRAINING_DAGGER_SEEDS + 1);
    let agreement = pretraining_agreement(validation.metrics());
    let gameplay = evaluate_pretraining_gameplay_validation(&model)?;
    agreement_trace.push(agreement);
    gameplay_trace.push(gameplay);
    let mut best = capture_pretraining_candidate(
        &model,
        agreement,
        gameplay,
        trainer.counters().global_update,
        stage_loss,
    )?;
    let remaining_epochs = settings.epochs - bootstrap_epochs;
    for dagger_index in 0..PRETRAINING_DAGGER_SEEDS {
        let dagger_samples = collect_dagger_samples(
            &model,
            settings,
            dagger_index,
            &mut collection.pool,
            &mut collection.training_action_counts,
        )?;
        collection.split_counts[0] = collection.split_counts[0]
            .checked_add(dagger_samples)
            .ok_or(PpoError::CounterOverflow)?;
        trainer
            .rebind_pool(&collection.pool)
            .map_err(imitation_error)?;
        let phase_epochs = remaining_epochs / PRETRAINING_DAGGER_SEEDS as u32;
        stage_loss = train_behavioral_epochs(&model, &collection.pool, &mut trainer, phase_epochs)?;
        validation = evaluate_pretraining_validation(&model, &collection)?;
        let agreement = pretraining_agreement(validation.metrics());
        let gameplay = evaluate_pretraining_gameplay_validation(&model)?;
        agreement_trace.push(agreement);
        gameplay_trace.push(gameplay);
        if pretraining_candidate_better(gameplay, agreement, &best) {
            best = capture_pretraining_candidate(
                &model,
                agreement,
                gameplay,
                trainer.counters().global_update,
                stage_loss,
            )?;
        }
    }
    model
        .import_parameters(&best.parameters)
        .map_err(model_error)?;
    validation = evaluate_pretraining_validation(&model, &collection)?;
    let restored_agreement = pretraining_agreement(validation.metrics());
    assert_eq!(restored_agreement, best.agreement);
    let held_out = evaluate_pretraining_held_out(&model, &collection)?;
    if let Err(error) = validate_pretraining_agreement(held_out.metrics()) {
        return Err(PpoError::Model(format!(
            "{error}; agreement stages={agreement_trace:?}; gameplay stages={gameplay_trace:?}"
        )));
    }
    let acceptance = evaluate_pretraining_gameplay(&model, PRETRAINING_GAMEPLAY_ACCEPTANCE_SEED)?;
    validate_pretraining_gameplay_acceptance(acceptance, &gameplay_trace)?;
    TrainingArtifact::save_runtime_weights(&model, output_directory).map_err(checkpoint_error)?;
    let fingerprint = PolicySnapshot::capture(&model, 0)
        .map_err(league_error)?
        .fingerprint();
    Ok(BehavioralPretrainingReport {
        training_samples: collection.split_counts[0],
        validation_samples: collection.split_counts[1],
        held_out_samples: collection.split_counts[2],
        optimizer_steps: best.optimizer_steps,
        final_loss: best.loss,
        held_out_kind_agreement: held_out
            .metrics()
            .overall
            .kind_agreement()
            .ok_or(PpoError::InvalidTransition("held-out kind agreement"))?,
        held_out_full_agreement: held_out
            .metrics()
            .overall
            .full_agreement()
            .ok_or(PpoError::InvalidTransition("held-out full agreement"))?,
        action_counts: collection.training_action_counts,
        gameplay_validation_map_zero_progress: best.gameplay.map_zero.structure_progress_games,
        gameplay_validation_map_one_wins: best.gameplay.map_one.wins,
        gameplay_validation_failures: best
            .gameplay
            .map_zero
            .failures
            .checked_add(best.gameplay.map_one.failures)
            .ok_or(PpoError::CounterOverflow)?,
        fingerprint,
    })
}

fn evaluate_pretraining_validation(
    model: &PolicyModel,
    collection: &PretrainingCollection,
) -> Result<crate::ValidationEvaluation, PpoError> {
    OfflineEvaluation::evaluate_validation(
        model,
        &collection.pool,
        collection.validation_coverage.clone(),
    )
    .map_err(imitation_error)
}

fn evaluate_pretraining_held_out(
    model: &PolicyModel,
    collection: &PretrainingCollection,
) -> Result<crate::HeldOutEvaluation, PpoError> {
    OfflineEvaluation::evaluate_held_out(
        model,
        &collection.pool,
        collection.held_out_coverage.clone(),
    )
    .map_err(imitation_error)
}

fn pretraining_agreement(metrics: &crate::OfflineEvaluation) -> PretrainingAgreement {
    PretrainingAgreement {
        kind_matching: metrics.overall.kind.matching,
        kind_total: metrics.overall.kind.total,
        full_matching: metrics.overall.full.matching,
        full_total: metrics.overall.full.total,
        radiant_kind: metrics.radiant.kind,
        dire_kind: metrics.dire.kind,
        families: metrics.overall.families,
    }
}

fn capture_pretraining_candidate(
    model: &PolicyModel,
    agreement: PretrainingAgreement,
    gameplay: PretrainingGameplayMatrix,
    optimizer_steps: u64,
    loss: f64,
) -> Result<PretrainingCandidate, PpoError> {
    if !loss.is_finite() {
        return Err(PpoError::InvalidTransition("pretraining candidate loss"));
    }
    let parameters = model.export_parameters().map_err(model_error)?;
    assert_eq!(parameters.len(), crate::MODEL_PARAMETER_COUNT);
    Ok(PretrainingCandidate {
        agreement,
        gameplay,
        parameters,
        optimizer_steps,
        loss,
    })
}

fn pretraining_candidate_better(
    gameplay: PretrainingGameplayMatrix,
    agreement: PretrainingAgreement,
    current: &PretrainingCandidate,
) -> bool {
    pretraining_gameplay_better(gameplay, current.gameplay)
        || (gameplay == current.gameplay
            && pretraining_agreement_better(agreement, current.agreement))
}

const fn pretraining_gameplay_better(
    candidate: PretrainingGameplayMatrix,
    current: PretrainingGameplayMatrix,
) -> bool {
    let candidate_rejections = candidate
        .map_zero
        .rejections
        .saturating_add(candidate.map_one.rejections);
    let current_rejections = current
        .map_zero
        .rejections
        .saturating_add(current.map_one.rejections);
    if candidate_rejections != current_rejections {
        return candidate_rejections < current_rejections;
    }
    if candidate.map_one.failures != current.map_one.failures {
        return candidate.map_one.failures < current.map_one.failures;
    }
    if candidate.map_one.wins != current.map_one.wins {
        return candidate.map_one.wins > current.map_one.wins;
    }
    if candidate.map_zero.failures != current.map_zero.failures {
        return candidate.map_zero.failures < current.map_zero.failures;
    }
    if candidate.map_zero.structure_progress_games != current.map_zero.structure_progress_games {
        return candidate.map_zero.structure_progress_games
            > current.map_zero.structure_progress_games;
    }
    if candidate.map_zero.structures != current.map_zero.structures {
        return candidate.map_zero.structures > current.map_zero.structures;
    }
    if candidate.map_one.structures != current.map_one.structures {
        return candidate.map_one.structures > current.map_one.structures;
    }
    candidate
        .map_zero
        .deaths
        .saturating_add(candidate.map_one.deaths)
        < current
            .map_zero
            .deaths
            .saturating_add(current.map_one.deaths)
}

const fn pretraining_agreement_better(
    candidate: PretrainingAgreement,
    current: PretrainingAgreement,
) -> bool {
    candidate.full_matching > current.full_matching
        || (candidate.full_matching == current.full_matching
            && candidate.kind_matching > current.kind_matching)
}

#[cfg(test)]
pub(crate) fn pretraining_best_stage_for_test(agreements: &[(usize, usize)]) -> usize {
    assert!(!agreements.is_empty());
    assert!(agreements.len() <= PRETRAINING_DAGGER_SEEDS + 1);
    let mut selected = 0usize;
    for index in 1..agreements.len() {
        let candidate = agreements[index];
        let current = agreements[selected];
        if candidate.1 > current.1 || (candidate.1 == current.1 && candidate.0 > current.0) {
            selected = index;
        }
    }
    selected
}

#[cfg(test)]
pub(crate) fn pretraining_gameplay_stage_for_test(stages: &[[PretrainingGameplay; 2]]) -> usize {
    assert!(!stages.is_empty());
    assert!(stages.len() <= PRETRAINING_DAGGER_SEEDS + 1);
    let gameplay = |stage: [PretrainingGameplay; 2]| PretrainingGameplayMatrix {
        map_zero: stage[0],
        map_one: stage[1],
    };
    let mut selected = 0usize;
    for index in 1..stages.len() {
        if pretraining_gameplay_better(gameplay(stages[index]), gameplay(stages[selected])) {
            selected = index;
        }
    }
    selected
}

fn evaluate_pretraining_gameplay(
    model: &PolicyModel,
    seed: u64,
) -> Result<PretrainingGameplayMatrix, PpoError> {
    Ok(PretrainingGameplayMatrix {
        map_zero: evaluate_pretraining_map_gameplay(model, MapId(0), seed)?,
        map_one: evaluate_pretraining_map_gameplay(model, MapId(1), seed)?,
    })
}

fn evaluate_pretraining_gameplay_validation(
    model: &PolicyModel,
) -> Result<PretrainingGameplayMatrix, PpoError> {
    let mut output = PretrainingGameplayMatrix::default();
    for seed in PRETRAINING_GAMEPLAY_VALIDATION_SEEDS {
        output.merge(evaluate_pretraining_gameplay(model, seed)?)?;
    }
    Ok(output)
}

impl PretrainingGameplayMatrix {
    fn merge(&mut self, other: Self) -> Result<(), PpoError> {
        self.map_zero.merge(other.map_zero)?;
        self.map_one.merge(other.map_one)
    }
}

impl PretrainingGameplay {
    fn merge(&mut self, other: Self) -> Result<(), PpoError> {
        self.games = self
            .games
            .checked_add(other.games)
            .ok_or(PpoError::CounterOverflow)?;
        self.failures = self
            .failures
            .checked_add(other.failures)
            .ok_or(PpoError::CounterOverflow)?;
        self.wins = self
            .wins
            .checked_add(other.wins)
            .ok_or(PpoError::CounterOverflow)?;
        self.structure_progress_games = self
            .structure_progress_games
            .checked_add(other.structure_progress_games)
            .ok_or(PpoError::CounterOverflow)?;
        self.structures = self
            .structures
            .checked_add(other.structures)
            .ok_or(PpoError::CounterOverflow)?;
        self.deaths = self
            .deaths
            .checked_add(other.deaths)
            .ok_or(PpoError::CounterOverflow)?;
        self.rejections = self
            .rejections
            .checked_add(other.rejections)
            .ok_or(PpoError::CounterOverflow)?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) const fn pretraining_gameplay_validation_seeds_for_test() -> [u64; 3] {
    PRETRAINING_GAMEPLAY_VALIDATION_SEEDS
}

fn evaluate_pretraining_map_gameplay(
    model: &PolicyModel,
    map: MapId,
    seed: u64,
) -> Result<PretrainingGameplay, PpoError> {
    let settings = CheckpointEvaluationConfig {
        pairs: 1,
        decisions: PRETRAINING_GAMEPLAY_DECISIONS,
        seed,
    };
    let mut output = PretrainingGameplay::default();
    for candidate_seat in 0..2 {
        let game = evaluate_checkpoint_game(
            model,
            settings,
            map,
            CheckpointEvaluationBaseline::Weak,
            seed,
            candidate_seat,
        )?;
        if game.baseline_wire_orders != 0 || game.baseline_rejected_orders != 0 {
            return Err(PpoError::InvalidTransition(
                "pretraining weak baseline activity",
            ));
        }
        if !game.final_summary.destroyed_structures_present {
            return Err(PpoError::InvalidTransition(
                "pretraining structure baseline incomplete",
            ));
        }
        output.games = output
            .games
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        output.rejections = output
            .rejections
            .checked_add(u64::from(game.rejected_orders))
            .ok_or(PpoError::CounterOverflow)?;
        output.structures = output
            .structures
            .checked_add(u64::from(game.final_summary.enemy_structures_destroyed))
            .ok_or(PpoError::CounterOverflow)?;
        if game.final_summary.enemy_structures_destroyed > 0 {
            output.structure_progress_games = output
                .structure_progress_games
                .checked_add(1)
                .ok_or(PpoError::CounterOverflow)?;
        }
        output.deaths = output
            .deaths
            .checked_add(game.final_summary.allied.deaths)
            .ok_or(PpoError::CounterOverflow)?;
        match game.outcome {
            CheckpointEvaluationOutcome::Win => output.wins += 1,
            CheckpointEvaluationOutcome::Loss => output.failures += 1,
            CheckpointEvaluationOutcome::Timeout
                if game.final_summary.enemy_structures_destroyed == 0 =>
            {
                output.failures += 1;
            }
            CheckpointEvaluationOutcome::Timeout => {}
        }
    }
    Ok(output)
}

fn validate_pretraining_gameplay_acceptance(
    gameplay: PretrainingGameplayMatrix,
    stages: &[PretrainingGameplayMatrix],
) -> Result<(), PpoError> {
    let map_zero = gameplay.map_zero;
    let map_one = gameplay.map_one;
    if map_zero.games == 2
        && map_zero.failures == 0
        && map_zero.structure_progress_games == 2
        && map_zero.structures >= 2
        && map_zero.rejections == 0
        && map_one.games == 2
        && map_one.failures == 0
        && map_one.wins == 2
        && map_one.structures >= 2
        && map_one.rejections == 0
    {
        return Ok(());
    }
    Err(PpoError::Model(format!(
        "pretraining gameplay acceptance failed: {gameplay:?}; validation stages={stages:?}"
    )))
}

#[cfg(test)]
pub(crate) fn validate_pretraining_gameplay_acceptance_for_test(
    map_zero: PretrainingGameplay,
    map_one: PretrainingGameplay,
) -> Result<(), PpoError> {
    validate_pretraining_gameplay_acceptance(PretrainingGameplayMatrix { map_zero, map_one }, &[])
}

fn validate_behavioral_pretraining(settings: BehavioralPretrainingConfig) -> Result<(), PpoError> {
    if !(8..=32).contains(&settings.epochs) || !settings.epochs.is_multiple_of(8) {
        return Err(PpoError::InvalidConfig("pretraining epochs"));
    }
    settings
        .seed
        .checked_add(
            (PRETRAINING_TRAIN_SEEDS
                + PRETRAINING_DAGGER_SEEDS
                + PRETRAINING_VALIDATION_SEEDS
                + PRETRAINING_HELD_OUT_SEEDS) as u64,
        )
        .ok_or(PpoError::InvalidConfig("pretraining seed"))?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_behavioral_pretraining_for_test(
    settings: BehavioralPretrainingConfig,
) -> Result<(), PpoError> {
    validate_behavioral_pretraining(settings)
}

fn collect_pretraining_samples(
    settings: BehavioralPretrainingConfig,
) -> Result<PretrainingCollection, PpoError> {
    let training = (0..PRETRAINING_TRAIN_SEEDS)
        .map(|offset| settings.seed + offset as u64)
        .collect::<Vec<_>>();
    let dagger = (0..PRETRAINING_DAGGER_SEEDS)
        .map(|offset| settings.seed + PRETRAINING_TRAIN_SEEDS as u64 + offset as u64)
        .collect::<Vec<_>>();
    let validation = (0..PRETRAINING_VALIDATION_SEEDS)
        .map(|offset| {
            settings.seed
                + (PRETRAINING_TRAIN_SEEDS + PRETRAINING_DAGGER_SEEDS) as u64
                + offset as u64
        })
        .collect::<Vec<_>>();
    let promotion = (0..PRETRAINING_HELD_OUT_SEEDS)
        .map(|offset| {
            settings.seed
                + (PRETRAINING_TRAIN_SEEDS
                    + PRETRAINING_DAGGER_SEEDS
                    + PRETRAINING_VALIDATION_SEEDS) as u64
                + offset as u64
        })
        .collect::<Vec<_>>();
    let mut optimization_seeds = training.clone();
    optimization_seeds.extend_from_slice(&dagger);
    let namespaces = SeedNamespaces::new(optimization_seeds, validation.clone(), promotion.clone())
        .map_err(imitation_error)?;
    let scope = TrainingScope::new(PRETRAINING_TRAINING_MAP, IMITATION_RULES_AUDIT_VERSION)
        .map_err(imitation_error)?;
    let capacity = PRETRAINING_SAMPLE_CAPACITY;
    let mut pool = ImitationPool::new(capacity, settings.seed | 1, namespaces, scope)
        .map_err(imitation_error)?;
    let mut training_coverage = TeacherCoverage::new();
    let mut validation_coverage = TeacherCoverage::new();
    let mut held_out_coverage = TeacherCoverage::new();
    let mut training_action_counts = [0u64; ActionKind::COUNT];
    let mut validation_action_counts = [0u64; ActionKind::COUNT];
    let mut held_out_action_counts = [0u64; ActionKind::COUNT];
    let mut side_counts = [0usize; 2];
    let mut split_counts = [0usize; 3];
    for seed in training {
        collect_pretraining_seed(
            PRETRAINING_TRAINING_MAP,
            seed,
            SeedNamespace::Training,
            &mut pool,
            &mut training_coverage,
            &mut training_action_counts,
            &mut side_counts,
            &mut split_counts,
        )?;
    }
    for seed in validation {
        collect_pretraining_seed(
            PRETRAINING_TRAINING_MAP,
            seed,
            SeedNamespace::Validation,
            &mut pool,
            &mut validation_coverage,
            &mut validation_action_counts,
            &mut side_counts,
            &mut split_counts,
        )?;
    }
    for seed in promotion {
        collect_pretraining_seed(
            PRETRAINING_TRAINING_MAP,
            seed,
            SeedNamespace::Promotion,
            &mut pool,
            &mut held_out_coverage,
            &mut held_out_action_counts,
            &mut side_counts,
            &mut split_counts,
        )?;
    }
    let counted_splits = [
        usize::try_from(training_action_counts.iter().copied().sum::<u64>())
            .map_err(|_| PpoError::CounterOverflow)?,
        usize::try_from(validation_action_counts.iter().copied().sum::<u64>())
            .map_err(|_| PpoError::CounterOverflow)?,
        usize::try_from(held_out_action_counts.iter().copied().sum::<u64>())
            .map_err(|_| PpoError::CounterOverflow)?,
    ];
    if pool.is_empty()
        || pool.len() != split_counts.iter().copied().sum::<usize>()
        || counted_splits != split_counts
        || training_coverage.attempted() != training_coverage.represented()
        || validation_coverage.attempted() != validation_coverage.represented()
        || held_out_coverage.attempted() != held_out_coverage.represented()
        || side_counts[0].abs_diff(side_counts[1]).saturating_mul(100)
            > pool.len().saturating_mul(5)
    {
        return Err(PpoError::Model(format!(
            "pretraining collection coverage failed: pool={}, splits={split_counts:?}, sides={side_counts:?}, train={}/{}, validation={}/{}, held_out={}/{}",
            pool.len(),
            training_coverage.represented(),
            training_coverage.attempted(),
            validation_coverage.represented(),
            validation_coverage.attempted(),
            held_out_coverage.represented(),
            held_out_coverage.attempted(),
        )));
    }
    Ok(PretrainingCollection {
        pool,
        validation_coverage,
        held_out_coverage,
        training_action_counts,
        split_counts,
    })
}

fn train_behavioral_epochs(
    model: &PolicyModel,
    pool: &ImitationPool,
    trainer: &mut BehavioralTrainer,
    epochs: u32,
) -> Result<f64, PpoError> {
    let mut final_loss = 0.0;
    for _ in 0..epochs {
        final_loss = trainer
            .train_epoch(model, pool)
            .map_err(imitation_error)?
            .average_loss;
    }
    Ok(final_loss)
}

fn collect_dagger_samples(
    model: &PolicyModel,
    settings: BehavioralPretrainingConfig,
    dagger_index: usize,
    pool: &mut ImitationPool,
    training_action_counts: &mut [u64; ActionKind::COUNT],
) -> Result<usize, PpoError> {
    let mut retained = 0usize;
    let seed = settings.seed + PRETRAINING_TRAIN_SEEDS as u64 + dagger_index as u64;
    let baseline = pretraining_dagger_baseline(dagger_index);
    let mut window_counts = [[[0u64; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2];
    for policy_seat in 0..2 {
        let side_retained = collect_dagger_side(
            model,
            PRETRAINING_TRAINING_MAP,
            seed,
            baseline,
            policy_seat,
            pool,
            training_action_counts,
            &mut window_counts,
        )?;
        retained = retained
            .checked_add(side_retained)
            .ok_or(PpoError::CounterOverflow)?;
    }
    Ok(retained)
}

#[allow(clippy::too_many_arguments)]
fn collect_dagger_side(
    model: &PolicyModel,
    map: MapId,
    seed: u64,
    baseline: CheckpointEvaluationBaseline,
    policy_seat: usize,
    pool: &mut ImitationPool,
    training_action_counts: &mut [u64; ActionKind::COUNT],
    dagger_window_counts: &mut [[[u64; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2],
) -> Result<usize, PpoError> {
    let opponent = match baseline {
        CheckpointEvaluationBaseline::Teacher => OpponentSpec::Teacher,
        CheckpointEvaluationBaseline::Weak => OpponentSpec::Weak,
    };
    let mut environment = build_environment(
        seed,
        derive_training_seed(seed, policy_seat as u64, baseline_domain(baseline)),
        map,
        policy_seat,
        0,
        opponent,
    )?;
    assert_eq!(environment.policy_seat, policy_seat);
    warmup_checkpoint_evaluation(&mut environment)?;
    let mut retained = 0usize;
    let mut trajectory = 0u64;
    for decision in 0..PRETRAINING_DAGGER_DECISIONS {
        let retained_sample = dagger_decision(
            &mut environment.seats[policy_seat],
            model,
            seed,
            trajectory,
            decision,
            pool,
            training_action_counts,
            dagger_window_counts,
        )?;
        retained = retained
            .checked_add(retained_sample)
            .ok_or(PpoError::CounterOverflow)?;
        let mut learner_request =
            dagger_learner_request(&mut environment.seats[policy_seat], model)?;
        let mut requests = Vec::with_capacity(environment.seats.len());
        for index in 0..environment.seats.len() {
            let request = if index == policy_seat {
                learner_request.take()
            } else {
                opponent_request(&mut environment.seats[index], &mut environment.opponent)?
            };
            requests.push(request);
        }
        let advanced = advance_interval(&mut environment, requests, 3)?;
        reject_production_rejection(&environment, "DAgger collection")?;
        if advanced.winner.is_some() {
            trajectory = trajectory.checked_add(1).ok_or(PpoError::CounterOverflow)?;
            restart_environment(&mut environment)?;
            warmup_checkpoint_evaluation(&mut environment)?;
        }
    }
    Ok(retained)
}

const fn pretraining_dagger_baseline(index: usize) -> CheckpointEvaluationBaseline {
    assert!(index < PRETRAINING_DAGGER_SEEDS);
    CheckpointEvaluationBaseline::Weak
}

#[cfg(test)]
pub(crate) const fn pretraining_dagger_baseline_for_test(
    index: usize,
) -> CheckpointEvaluationBaseline {
    pretraining_dagger_baseline(index)
}

#[allow(clippy::too_many_arguments)]
fn dagger_decision(
    seat: &mut ArenaSeatPolicy,
    model: &PolicyModel,
    seed: u64,
    trajectory: u64,
    decision: usize,
    pool: &mut ImitationPool,
    training_action_counts: &mut [u64; ActionKind::COUNT],
    dagger_window_counts: &mut [[[u64; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2],
) -> Result<usize, PpoError> {
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
    let learner = model.choose(&frame, &space).map_err(model_error)?.action;
    let (teacher, _) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let learner = if deployment_uses_teacher(seat.tracker.metadata().map) {
        teacher
    } else {
        seat.teacher
            .safety_action(&seat.tracker, &space)
            .unwrap_or(learner)
    };
    let side = imitation_side_index(seat.tracker.team())?;
    let window = decision / PRETRAINING_WINDOW_DECISIONS;
    if dagger_window_counts[side][window][teacher.kind().index()]
        >= PRETRAINING_DAGGER_WINDOW_SIDE_KIND_CAP
    {
        return Ok(0);
    }
    if learner == teacher && !(decision + 1).is_multiple_of(PRETRAINING_CONTINUE_STRIDE) {
        return Ok(0);
    }
    let identity = SampleIdentity::from_frame(
        SeedNamespace::Training,
        seed,
        trajectory,
        space.tick(),
        &frame,
    )
    .map_err(imitation_error)?;
    let sample = crate::ImitationSample::dagger(frame, &space, learner, teacher, identity)
        .map_err(imitation_error)?;
    if pool.push(sample).map_err(imitation_error)?.is_some() {
        return Err(PpoError::InvalidTransition("DAgger pool eviction"));
    }
    training_action_counts[teacher.kind().index()] = training_action_counts[teacher.kind().index()]
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    dagger_window_counts[side][window][teacher.kind().index()] = dagger_window_counts[side][window]
        [teacher.kind().index()]
    .checked_add(1)
    .ok_or(PpoError::CounterOverflow)?;
    Ok(1)
}

fn dagger_learner_request(
    seat: &mut ArenaSeatPolicy,
    model: &PolicyModel,
) -> Result<Option<Request>, PpoError> {
    if deployment_uses_teacher(seat.tracker.metadata().map) {
        return teacher_request(seat);
    }
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
    let action = model.choose(&frame, &space).map_err(model_error)?.action;
    let action = seat
        .teacher
        .safety_action(&seat.tracker, &space)
        .unwrap_or(action);
    seat.local
        .note_decision(space.tick(), action.kind())
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    issue_request(seat, issued, &space, action.kind(), true)
}

#[cfg(test)]
type PretrainingSummary = (usize, [[u64; ActionKind::COUNT]; 3], [usize; 3]);

#[cfg(test)]
pub(crate) fn collect_pretraining_summary_for_test(
    seed: u64,
) -> Result<PretrainingSummary, PpoError> {
    let collection = collect_pretraining_samples(BehavioralPretrainingConfig { epochs: 1, seed })?;
    summarize_pretraining_map(&collection.pool, PRETRAINING_TRAINING_MAP)
}

#[cfg(test)]
fn summarize_pretraining_map(
    pool: &ImitationPool,
    map: MapId,
) -> Result<PretrainingSummary, PpoError> {
    assert_eq!(map, PRETRAINING_TRAINING_MAP);
    let mut split_actions = [[0u64; ActionKind::COUNT]; 3];
    let mut split_counts = [0usize; 3];
    for index in 0..pool.len() {
        let sample = pool
            .get(index)
            .ok_or(PpoError::InvalidTransition("pretraining test sample"))?;
        if sample.identity().map() != map {
            continue;
        }
        let split = match sample.split() {
            crate::ImitationSplit::Train => 0,
            crate::ImitationSplit::Validation => 1,
            crate::ImitationSplit::HeldOut => 2,
        };
        split_actions[split][sample.teacher_action().kind().index()] += 1;
        split_counts[split] += 1;
    }
    let samples = split_counts.iter().sum();
    Ok((samples, split_actions, split_counts))
}

#[allow(clippy::too_many_arguments)]
fn collect_pretraining_seed(
    map: MapId,
    seed: u64,
    namespace: SeedNamespace,
    pool: &mut ImitationPool,
    coverage: &mut TeacherCoverage,
    action_counts: &mut [u64; ActionKind::COUNT],
    side_counts: &mut [usize; 2],
    split_counts: &mut [usize; 3],
) -> Result<(), PpoError> {
    let mut environment = build_environment(
        seed,
        derive_training_seed(seed, 0, 0x7072_6574_7261_696e),
        map,
        0,
        0,
        OpponentSpec::Teacher,
    )?;
    let mut trajectory = 0u64;
    let mut window_counts = [[[0u64; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2];
    for decision in 0..PRETRAINING_DECISIONS {
        let mut requests = Vec::with_capacity(environment.seats.len());
        for seat in &mut environment.seats {
            let (action, space, sample) = pretraining_teacher_action(
                seat,
                namespace,
                seed,
                trajectory,
                decision,
                &window_counts,
                coverage,
            )?;
            seat.local
                .note_decision(space.tick(), action.kind())
                .map_err(|error| PpoError::Model(error.to_string()))?;
            let issued = space
                .decode(action)
                .map_err(|error| PpoError::Model(error.to_string()))?;
            requests.push(issue_request(seat, issued, &space, action.kind(), true)?);
            if let Some(sample) = sample {
                let side = match sample.side() {
                    ImitationSide::Radiant => 0,
                    ImitationSide::Dire => 1,
                };
                if pool.push(sample).map_err(imitation_error)?.is_some() {
                    return Err(PpoError::InvalidTransition("pretraining pool eviction"));
                }
                action_counts[action.kind().index()] = action_counts[action.kind().index()]
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
                let window = decision / PRETRAINING_WINDOW_DECISIONS;
                window_counts[side][window][action.kind().index()] = window_counts[side][window]
                    [action.kind().index()]
                .checked_add(1)
                .ok_or(PpoError::CounterOverflow)?;
                side_counts[side] = side_counts[side]
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
                let split = match namespace {
                    SeedNamespace::Training => 0,
                    SeedNamespace::Validation => 1,
                    SeedNamespace::Promotion => 2,
                };
                split_counts[split] = split_counts[split]
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
        }
        let advanced = advance_interval(&mut environment, requests, 3)?;
        if environment.seats.iter().any(|seat| seat.rejections != 0) {
            return Err(PpoError::InvalidTransition("pretraining order rejection"));
        }
        if advanced.winner.is_some() {
            trajectory = trajectory.checked_add(1).ok_or(PpoError::CounterOverflow)?;
            restart_environment(&mut environment)?;
        }
    }
    Ok(())
}

fn pretraining_teacher_action(
    seat: &mut ArenaSeatPolicy,
    namespace: SeedNamespace,
    seed: u64,
    trajectory: u64,
    decision: usize,
    window_counts: &[[[u64; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2],
    coverage: &mut TeacherCoverage,
) -> Result<
    (
        crate::StructuredAction,
        ActionSpace,
        Option<crate::ImitationSample>,
    ),
    PpoError,
> {
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let side = imitation_side_index(seat.tracker.team())?;
    let window = decision / PRETRAINING_WINDOW_DECISIONS;
    let cap = pretraining_base_window_cap(namespace, action.kind())?;
    if window_counts[side][window][action.kind().index()] >= cap {
        return Ok((action, space, None));
    }
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
    let identity = SampleIdentity::from_frame(namespace, seed, trajectory, space.tick(), &frame)
        .map_err(imitation_error)?;
    let sample = crate::ImitationSample::teacher(frame, &space, action, identity)
        .map_err(imitation_error)?;
    coverage
        .record_represented_for(&sample)
        .map_err(imitation_error)?;
    Ok((action, space, Some(sample)))
}

fn pretraining_base_window_cap(
    namespace: SeedNamespace,
    kind: ActionKind,
) -> Result<u64, PpoError> {
    match (namespace, kind) {
        (SeedNamespace::Training, ActionKind::Continue) => {
            Ok(PRETRAINING_TRAIN_CONTINUE_WINDOW_CAP)
        }
        (SeedNamespace::Training, _) => Ok(PRETRAINING_TRAIN_OTHER_WINDOW_CAP),
        (SeedNamespace::Validation | SeedNamespace::Promotion, ActionKind::Continue) => {
            Ok(PRETRAINING_HELD_OUT_CONTINUE_WINDOW_CAP)
        }
        (SeedNamespace::Validation | SeedNamespace::Promotion, _) => {
            Ok(PRETRAINING_HELD_OUT_OTHER_WINDOW_CAP)
        }
    }
}

fn imitation_side_index(team: Team) -> Result<usize, PpoError> {
    match team {
        Team::Radiant => Ok(0),
        Team::Dire => Ok(1),
        Team::Neutral => Err(PpoError::InvalidTransition("pretraining neutral seat")),
    }
}

fn validate_action_diversity(counts: &[u64; ActionKind::COUNT]) -> Result<(), PpoError> {
    let total = counts.iter().copied().sum::<u64>();
    let families = counts.iter().filter(|count| **count != 0).count();
    let maximum = counts.iter().copied().max().unwrap_or(0);
    if total == 0 || families < 2 || maximum.saturating_mul(100) >= total.saturating_mul(95) {
        return Err(PpoError::InvalidTransition(
            "pretraining teacher action diversity",
        ));
    }
    Ok(())
}

fn validate_pretraining_agreement(metrics: &crate::OfflineEvaluation) -> Result<(), PpoError> {
    let requirements = [
        (
            "overall",
            &metrics.overall,
            PRETRAINING_OVERALL_KIND_PERCENT,
            PRETRAINING_OVERALL_FULL_PERCENT,
        ),
        (
            "radiant",
            &metrics.radiant,
            PRETRAINING_SIDE_KIND_PERCENT,
            PRETRAINING_SIDE_FULL_PERCENT,
        ),
        (
            "dire",
            &metrics.dire,
            PRETRAINING_SIDE_KIND_PERCENT,
            PRETRAINING_SIDE_FULL_PERCENT,
        ),
    ];
    for (scope, aggregate, kind_percent, full_percent) in requirements {
        if aggregate.kind.total == 0
            || aggregate.full.total == 0
            || aggregate.kind.matching.saturating_mul(100)
                < aggregate.kind.total.saturating_mul(kind_percent)
            || aggregate.full.matching.saturating_mul(100)
                < aggregate.full.total.saturating_mul(full_percent)
        {
            return Err(PpoError::Model(format!(
                "pretraining held-out {scope} agreement failed: kind={}/{}, full={}/{}",
                aggregate.kind.matching,
                aggregate.kind.total,
                aggregate.full.matching,
                aggregate.full.total,
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_pretraining_agreement_for_test(
    metrics: &crate::OfflineEvaluation,
) -> Result<(), PpoError> {
    validate_pretraining_agreement(metrics)
}

fn validate_checkpoint_evaluation(settings: CheckpointEvaluationConfig) -> Result<(), PpoError> {
    if !(1..=8).contains(&settings.pairs) {
        return Err(PpoError::InvalidConfig("checkpoint evaluation pairs"));
    }
    if !(1..=4_096).contains(&settings.decisions) {
        return Err(PpoError::InvalidConfig("checkpoint evaluation decisions"));
    }
    settings
        .seed
        .checked_add(settings.pairs as u64)
        .ok_or(PpoError::InvalidConfig("checkpoint evaluation seed"))?;
    Ok(())
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
    run_training_job_on_with_initial_weights(
        settings,
        device,
        checkpoint_directory,
        resume,
        None,
        checkpointed,
    )
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
    validate_initial_weights_directory(checkpoint_directory, resume, initial_weights_directory)?;
    validate_training_directory(checkpoint_directory, resume)?;
    let _directory_lock = TrainingDirectoryLock::acquire(checkpoint_directory)?;
    let config = validate_training_job(&settings, resume)?;
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
        initial_weights_directory,
        config,
        run,
    )?;
    let start = session.completed_updates;
    if start > settings.updates {
        return Err(PpoError::InvalidConfig(
            "training update target precedes checkpoint",
        ));
    }
    if session.migrated_provenance {
        let migration = session.checkpoint_report(None);
        let durable = session.save(checkpoint_directory, migration)?;
        checkpointed(durable);
    } else if resume {
        TrainingArtifact::save_runtime_weights(&session.model, checkpoint_directory)
            .map_err(checkpoint_error)?;
    }
    let started = Instant::now();
    let mut checkpoint_schedule = TrainingCheckpointSchedule::new(settings.checkpoint_cadence)?;
    for update in start..settings.updates {
        let report = session.train_update(&settings, update, config, capacity)?;
        let elapsed = started.elapsed();
        let final_update = report.completed_updates == settings.updates;
        if checkpoint_schedule.is_due(report.completed_updates, elapsed) || final_update {
            let durable = session.save(checkpoint_directory, report)?;
            checkpointed(durable);
            checkpoint_schedule.mark_committed(started.elapsed())?;
        }
    }
    Ok(session.report())
}

struct TrainingSession {
    model: PolicyModel,
    trainer: PpoTrainer,
    sampling: PpoRng,
    run: CheckpointRun,
    completed_updates: u64,
    rollout_samples: u64,
    latest: PpoUpdateReport,
    migrated_provenance: bool,
    starting_policy_fingerprint: u64,
    terminal_wins: u64,
    terminal_losses: u64,
    terminal_draws: u64,
    rejected_orders: u64,
    elapsed_ticks: u64,
}

impl TrainingSession {
    fn initialize(
        settings: &TrainingJobConfig,
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        initial_weights_directory: Option<&Path>,
        config: PpoConfig,
        run: CheckpointRun,
    ) -> Result<Self, PpoError> {
        let model = PolicyModel::fresh_on(settings.seed, device).map_err(model_error)?;
        if !resume && let Some(initial_weights_directory) = initial_weights_directory {
            TrainingArtifact::load_runtime_weights(&model, initial_weights_directory)
                .map_err(checkpoint_error)?;
        }
        let (trainer, sampling, completed_updates, rollout_samples, migrated_provenance) = if resume
        {
            restore_training_session(&model, directory, &run, config, settings.resume_provenance)?
        } else {
            let trainer = PpoTrainer::new(&model, config, settings.seed ^ 0x51a9)?;
            (trainer, PpoRng::new(settings.seed ^ 0xa17e), 0, 0, false)
        };
        let starting_policy_fingerprint = PolicySnapshot::capture(&model, completed_updates)
            .map_err(league_error)?
            .fingerprint();
        Ok(Self {
            model,
            trainer,
            sampling,
            run,
            completed_updates,
            rollout_samples,
            latest: PpoUpdateReport::default(),
            migrated_provenance,
            starting_policy_fingerprint,
            terminal_wins: 0,
            terminal_losses: 0,
            terminal_draws: 0,
            rejected_orders: 0,
            elapsed_ticks: 0,
        })
    }

    fn train_update(
        &mut self,
        settings: &TrainingJobConfig,
        update: u64,
        config: PpoConfig,
        capacity: usize,
    ) -> Result<TrainingCheckpointReport, PpoError> {
        assert_eq!(update, self.completed_updates);
        assert_eq!(self.completed_updates, self.trainer.updates());
        let mut rollout =
            PpoRollout::new(capacity, self.model.policy_identity().map_err(model_error)?)?;
        let mut environments = build_training_environments(settings, update, config, &self.model)?;
        let mut actor_report = PpoSmokeReport::default();
        collect_update(
            &self.model,
            &mut self.sampling,
            &mut environments,
            config,
            settings.rollout_decisions,
            &mut rollout,
            &mut actor_report,
        )?;
        self.terminal_wins = self
            .terminal_wins
            .checked_add(actor_report.terminal_wins)
            .ok_or(PpoError::CounterOverflow)?;
        self.terminal_losses = self
            .terminal_losses
            .checked_add(actor_report.terminal_losses)
            .ok_or(PpoError::CounterOverflow)?;
        self.terminal_draws = self
            .terminal_draws
            .checked_add(actor_report.terminal_draws)
            .ok_or(PpoError::CounterOverflow)?;
        self.rejected_orders = self
            .rejected_orders
            .checked_add(actor_report.rejected_orders)
            .ok_or(PpoError::CounterOverflow)?;
        self.elapsed_ticks = self
            .elapsed_ticks
            .checked_add(actor_report.elapsed_ticks)
            .ok_or(PpoError::CounterOverflow)?;
        assert_eq!(rollout.len(), capacity);
        let batch = rollout.finish(config)?;
        self.latest = self.trainer.train_update(&self.model, &batch)?;
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
            policy_version: self.completed_updates,
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
            terminal_wins: self.terminal_wins,
            terminal_losses: self.terminal_losses,
            terminal_draws: self.terminal_draws,
            rejected_orders: self.rejected_orders,
            elapsed_ticks: self.elapsed_ticks,
            cleanup_warning,
        }
    }

    fn report(&self) -> TrainingJobReport {
        TrainingJobReport {
            starting_policy_fingerprint: self.starting_policy_fingerprint,
            completed_updates: self.completed_updates,
            optimizer_step: self.trainer.optimizer_step(),
            rollout_samples: self.rollout_samples,
            final_policy_loss: self.latest.policy_loss,
            final_value_loss: self.latest.value_loss,
            final_entropy: self.latest.entropy,
            final_kl: update_kl(self.latest),
            terminal_wins: self.terminal_wins,
            terminal_losses: self.terminal_losses,
            terminal_draws: self.terminal_draws,
            rejected_orders: self.rejected_orders,
            elapsed_ticks: self.elapsed_ticks,
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

#[cfg(test)]
pub(crate) fn merge_actor_report_for_test(
    aggregate: &mut PpoSmokeReport,
    actor: PpoSmokeReport,
) -> Result<(), PpoError> {
    merge_actor_report(aggregate, actor)
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
    if settings.environments == 0
        || settings.environments > TRAINING_MAX_ENVIRONMENTS
        || !settings.environments.is_multiple_of(2)
    {
        return Err(PpoError::InvalidConfig("training environments"));
    }
    if settings.rollout_decisions == 0 || settings.rollout_decisions > PPO_MAX_ROLLOUT_DECISIONS {
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
    let actor_seed_draws = settings
        .updates
        .checked_mul(settings.environments as u64)
        .ok_or(PpoError::InvalidConfig("training actor RNG counter"))?;
    let actor_stream_draws = (settings.rollout_decisions as u64)
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
    provenance: ResumeProvenance,
) -> Result<(PpoTrainer, PpoRng, u64, u64, bool), PpoError> {
    let (artifact, restore_run, migrated) = match provenance {
        ResumeProvenance::Strict => (
            TrainingArtifact::load_compatible(directory, run).map_err(checkpoint_error)?,
            run.clone(),
            false,
        ),
        ResumeProvenance::MigrateGitCommit => {
            let artifact = TrainingArtifact::load(directory).map_err(checkpoint_error)?;
            let restore_run = artifact.run().clone();
            validate_provenance_migration(&restore_run, run)?;
            (artifact, restore_run, true)
        }
    };
    let restored = artifact
        .restore(model, &restore_run)
        .map_err(checkpoint_error)?;
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
    let (trainer, _, _) = restored.into_parts();
    Ok((
        trainer,
        sampling,
        completed_updates,
        rollout_samples,
        migrated,
    ))
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
    model: &PolicyModel,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    let decision = update
        .checked_mul(settings.rollout_decisions as u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(PpoError::InvalidConfig("training decision counter"))?;
    let offset = update
        .checked_mul(settings.environments as u64)
        .ok_or(PpoError::CounterOverflow)?;
    let mut environments = Vec::with_capacity(settings.environments);
    let mut warmup_decisions = Vec::with_capacity(settings.environments);
    for index in 0..settings.environments {
        let stream = offset
            .checked_add(index as u64)
            .ok_or(PpoError::CounterOverflow)?;
        let pair = training_pair_index(stream);
        let opponent = match training_opponent_baseline(pair) {
            CheckpointEvaluationBaseline::Weak => OpponentSpec::Weak,
            CheckpointEvaluationBaseline::Teacher => OpponentSpec::Teacher,
        };
        let environment = build_environment(
            derive_training_seed(settings.seed, pair, 0x6172_656e_615f_7365),
            derive_training_seed(settings.seed, pair, 0x6f70_706f_6e65_6e74),
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

pub(crate) const fn derive_training_seed(base: u64, stream: u64, domain: u64) -> u64 {
    let mut value = base ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

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

#[cfg(test)]
pub(crate) const fn training_opponent_baseline_for_test(pair: u64) -> CheckpointEvaluationBaseline {
    training_opponent_baseline(pair)
}

#[cfg(test)]
pub(crate) const fn training_warmup_phase_index(stream: u64) -> usize {
    (training_pair_index(stream) % 8) as usize
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
    environment.reward = RewardTracker::default();
    let summary = environment.seats[environment.policy_seat]
        .tracker
        .latest_summary()
        .ok_or(PpoError::InvalidTransition("warmup summary"))?;
    environment.reward.observe(summary, 1.0, None)?;
    Ok(())
}

#[cfg(test)]
fn run_warmup_decisions(
    environment: &mut TrainingEnvironment,
    model: &PolicyModel,
    decisions: usize,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    for _ in 0..decisions {
        let (requests, _) = requests_for_greedy_decision(environment, model)?;
        let advanced = advance_interval(environment, requests, decision_interval_ticks)?;
        reject_production_rejection(environment, "production warmup")?;
        if advanced.winner.is_some() {
            restart_environment(environment)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn production_warmup_order_counts_for_test(
    model: &PolicyModel,
    decisions: usize,
) -> Result<[u32; 2], PpoError> {
    let mut environment = build_environment(23_078, 23_079, MapId(1), 0, 0, OpponentSpec::Weak)?;
    run_warmup_decisions(&mut environment, model, decisions, 3)?;
    assert_eq!(environment.policy_seat, 0);
    Ok([environment.seats[0].sequence, environment.seats[1].sequence])
}

#[cfg(test)]
pub(crate) fn production_batched_warmup_order_counts_for_test(
    model: &PolicyModel,
    decisions: usize,
) -> Result<[[u32; 2]; 2], PpoError> {
    let mut environments = vec![
        build_environment(23_082, 23_083, MapId(1), 0, 0, OpponentSpec::Weak)?,
        build_environment(23_082, 23_083, MapId(1), 1, 0, OpponentSpec::Weak)?,
    ];
    for _ in 0..decisions {
        let active = [0, 1];
        let requests = requests_for_batched_greedy_decisions(&mut environments, &active, model)?;
        advance_warmup_environments(&mut environments, &active, requests, 3)?;
    }
    Ok(std::array::from_fn(|environment| {
        std::array::from_fn(|seat| environments[environment].seats[seat].sequence)
    }))
}

#[cfg(test)]
pub(crate) fn production_warmup_cleanup_preserves_readiness_for_test() -> Result<bool, PpoError> {
    let mut environment = build_environment(23_080, 23_081, MapId(1), 0, 0, OpponentSpec::Weak)?;
    for seat in &mut environment.seats {
        seat.readiness
            .note_shared_wait_for_test(crate::ControlledUnit::Hero, 7, 2_100);
    }
    let before = environment
        .seats
        .iter()
        .map(|seat| seat.readiness)
        .collect::<Vec<_>>();
    clear_warmup_orders(&mut environment, 3)?;
    Ok(environment
        .seats
        .iter()
        .zip(before)
        .all(|(seat, readiness)| seat.readiness == readiness))
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
    if pairs.iter().any(pair_timed_out) {
        return league
            .insert_historical(candidate, score, profile)
            .map_err(league_error);
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

const fn pair_timed_out(pair: &LeaguePairedResult) -> bool {
    matches!(pair.candidate_radiant, LeagueMatchResult::Timeout)
        || matches!(pair.candidate_dire, LeagueMatchResult::Timeout)
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
        Err(
            crate::LeagueError::ExploitRegression
            | crate::LeagueError::InvalidRejectionRate
            | crate::LeagueError::EvaluationTimeout,
        ) => None,
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

fn evaluate_checkpoint_game(
    candidate: &PolicyModel,
    settings: CheckpointEvaluationConfig,
    map: MapId,
    baseline: CheckpointEvaluationBaseline,
    seed: u64,
    candidate_seat: usize,
) -> Result<CheckpointEvaluationGame, PpoError> {
    let opponent = match baseline {
        CheckpointEvaluationBaseline::Teacher => OpponentSpec::Teacher,
        CheckpointEvaluationBaseline::Weak => OpponentSpec::Weak,
    };
    let opponent_seed = derive_training_seed(seed, map.0 as u64, baseline_domain(baseline));
    let mut environment = build_environment(seed, opponent_seed, map, candidate_seat, 0, opponent)?;
    warmup_checkpoint_evaluation(&mut environment)?;
    let candidate_team = environment.seats[candidate_seat].tracker.team();
    let mut action_counts = [0u32; ActionKind::COUNT];
    let mut decisions = 0u32;
    let mut elapsed_ticks = 0u32;
    let mut winner = None;
    for _ in 0..settings.decisions {
        let (requests, action) = requests_for_greedy_decision(&mut environment, candidate)?;
        let advanced = advance_interval(&mut environment, requests, 3)?;
        action_counts[action.index()] = action_counts[action.index()]
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        decisions = decisions.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        elapsed_ticks = elapsed_ticks
            .checked_add(advanced.ticks)
            .ok_or(PpoError::CounterOverflow)?;
        if advanced.winner.is_some() {
            winner = advanced.winner;
            break;
        }
    }
    let seat = &environment.seats[candidate_seat];
    let baseline_seat = &environment.seats[1 - candidate_seat];
    let rejected_orders = u32::try_from(seat.rejections)
        .map_err(|_| PpoError::InvalidTransition("evaluation rejection count"))?;
    let final_summary = seat
        .tracker
        .latest_summary()
        .ok_or(PpoError::InvalidTransition("evaluation final summary"))?;
    Ok(CheckpointEvaluationGame {
        map,
        baseline,
        seed,
        candidate_team,
        outcome: checkpoint_evaluation_outcome(candidate_team, winner),
        decisions,
        wire_orders: seat.sequence,
        rejected_orders,
        baseline_wire_orders: baseline_seat.sequence,
        baseline_rejected_orders: u32::try_from(baseline_seat.rejections)
            .map_err(|_| PpoError::InvalidTransition("baseline rejection count"))?,
        elapsed_ticks,
        action_counts,
        final_summary,
    })
}

#[cfg(test)]
pub(crate) fn evaluate_teacher_against_weak_for_test(
    seed: u64,
    decisions: usize,
) -> Result<Vec<CheckpointEvaluationGame>, PpoError> {
    validate_checkpoint_evaluation(CheckpointEvaluationConfig {
        pairs: 1,
        decisions,
        seed,
    })?;
    let mut games = Vec::with_capacity(4);
    for map in [MapId(0), MapId(1)] {
        for teacher_seat in 0..2 {
            games.push(evaluate_teacher_against_weak_game(
                seed,
                decisions,
                map,
                teacher_seat,
            )?);
        }
    }
    Ok(games)
}

#[cfg(test)]
fn evaluate_teacher_against_weak_game(
    seed: u64,
    decision_limit: usize,
    map: MapId,
    teacher_seat: usize,
) -> Result<CheckpointEvaluationGame, PpoError> {
    let opponent_seed = derive_training_seed(
        seed,
        map.0 as u64,
        baseline_domain(CheckpointEvaluationBaseline::Weak),
    );
    let mut environment = build_environment(
        seed,
        opponent_seed,
        map,
        teacher_seat,
        0,
        OpponentSpec::Weak,
    )?;
    warmup_checkpoint_evaluation(&mut environment)?;
    let teacher_team = environment.seats[teacher_seat].tracker.team();
    let mut action_counts = [0u32; ActionKind::COUNT];
    let mut decisions = 0u32;
    let mut elapsed_ticks = 0u32;
    let mut winner = None;
    for _ in 0..decision_limit {
        let (requests, action) = requests_for_teacher_decision(&mut environment)?;
        let advanced = advance_interval(&mut environment, requests, 3)?;
        action_counts[action.index()] = action_counts[action.index()]
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        decisions = decisions.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        elapsed_ticks = elapsed_ticks
            .checked_add(advanced.ticks)
            .ok_or(PpoError::CounterOverflow)?;
        if advanced.winner.is_some() {
            winner = advanced.winner;
            break;
        }
    }
    teacher_weak_game_report(
        &environment,
        map,
        seed,
        teacher_team,
        winner,
        decisions,
        elapsed_ticks,
        action_counts,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn teacher_weak_game_report(
    environment: &TrainingEnvironment,
    map: MapId,
    seed: u64,
    teacher_team: Team,
    winner: Option<Team>,
    decisions: u32,
    elapsed_ticks: u32,
    action_counts: [u32; ActionKind::COUNT],
) -> Result<CheckpointEvaluationGame, PpoError> {
    let teacher = &environment.seats[environment.policy_seat];
    let weak = &environment.seats[1 - environment.policy_seat];
    Ok(CheckpointEvaluationGame {
        map,
        baseline: CheckpointEvaluationBaseline::Weak,
        seed,
        candidate_team: teacher_team,
        outcome: checkpoint_evaluation_outcome(teacher_team, winner),
        decisions,
        wire_orders: teacher.sequence,
        rejected_orders: u32::try_from(teacher.rejections)
            .map_err(|_| PpoError::InvalidTransition("teacher rejection count"))?,
        baseline_wire_orders: weak.sequence,
        baseline_rejected_orders: u32::try_from(weak.rejections)
            .map_err(|_| PpoError::InvalidTransition("weak rejection count"))?,
        elapsed_ticks,
        action_counts,
        final_summary: teacher
            .tracker
            .latest_summary()
            .ok_or(PpoError::InvalidTransition("teacher final summary"))?,
    })
}

const fn baseline_domain(baseline: CheckpointEvaluationBaseline) -> u64 {
    match baseline {
        CheckpointEvaluationBaseline::Teacher => 0x7465_6163_6865_7221,
        CheckpointEvaluationBaseline::Weak => 0x7765_616b_5f5f_5f5f,
    }
}

fn warmup_checkpoint_evaluation(environment: &mut TrainingEnvironment) -> Result<(), PpoError> {
    for _ in 0..300 {
        let requests = vec![None; environment.seats.len()];
        let advanced = advance_interval(environment, requests, 3)?;
        if advanced.winner.is_some() {
            return Err(PpoError::InvalidTransition(
                "checkpoint evaluation warmup terminated",
            ));
        }
    }
    Ok(())
}

fn checkpoint_evaluation_outcome(
    candidate: Team,
    winner: Option<Team>,
) -> CheckpointEvaluationOutcome {
    match winner {
        Some(winner) if winner == candidate => CheckpointEvaluationOutcome::Win,
        Some(_) => CheckpointEvaluationOutcome::Loss,
        None => CheckpointEvaluationOutcome::Timeout,
    }
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
    Ok(LeagueMatchResult::Timeout)
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
        LeagueMatchResult::Timeout => -1,
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

#[cfg(test)]
pub(crate) fn rejection_delta_for_test(before: u64, after: u64) -> Result<u64, PpoError> {
    rejection_delta(before, after)
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
    let mut frames = Vec::with_capacity(environments.len());
    let mut spaces = Vec::with_capacity(environments.len());
    for environment in environments.iter_mut() {
        let (frame, space) = prepare_policy_sample(environment)?;
        frames.push(frame);
        spaces.push(space);
    }
    let choices = model
        .sample_batch(&frames, &spaces, sampling)
        .map_err(model_error)?;
    let mut pending = Vec::with_capacity(environments.len());
    for ((environment, choice), space) in environments.iter_mut().zip(choices).zip(&spaces) {
        let requests = requests_for_decision_in_space(environment, &choice, space)?;
        let advanced = advance_interval(environment, requests, config.decision_interval_ticks)?;
        reject_production_rejection(environment, "production rollout")?;
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
        .map_err(feature_error)?;
    Ok((frame, space))
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
    let seat = &environment.seats[environment.policy_seat];
    let space = ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    requests_for_decision_in_space(environment, choice, &space)
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

fn requests_for_greedy_decision(
    environment: &mut TrainingEnvironment,
    model: &PolicyModel,
) -> Result<(Vec<Option<Request>>, ActionKind), PpoError> {
    let policy_seat = environment.policy_seat;
    let (action, candidate_request) =
        greedy_policy_request(&mut environment.seats[policy_seat], model)?;
    let requests = requests_with_candidate(environment, candidate_request)?;
    Ok((requests, action))
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
    if active
        .iter()
        .all(|index| deployment_uses_teacher(environments[*index].map))
    {
        return active
            .iter()
            .map(|index| {
                requests_for_greedy_decision(&mut environments[*index], model).map(|v| v.0)
            })
            .collect();
    }
    requests_for_batched_model_greedy_decisions(environments, active, model)
}

fn requests_for_batched_model_greedy_decisions(
    environments: &mut [TrainingEnvironment],
    active: &[usize],
    model: &PolicyModel,
) -> Result<Vec<Vec<Option<Request>>>, PpoError> {
    if active
        .iter()
        .any(|index| deployment_uses_teacher(environments[*index].map))
    {
        return Err(PpoError::InvalidTransition("mixed batched warmup policy"));
    }
    let mut frames = Vec::with_capacity(active.len());
    let mut spaces = Vec::with_capacity(active.len());
    for &index in active {
        let (frame, space) = prepare_policy_sample(&mut environments[index])?;
        frames.push(frame);
        spaces.push(space);
    }
    let choices = model.choose_batch(&frames, &spaces).map_err(model_error)?;
    let mut output = Vec::with_capacity(active.len());
    for ((&index, choice), space) in active.iter().zip(choices).zip(&spaces) {
        let environment = &mut environments[index];
        let policy_seat = environment.policy_seat;
        let (_, candidate_request) = greedy_policy_request_in_space(
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

#[cfg(test)]
fn requests_for_teacher_decision(
    environment: &mut TrainingEnvironment,
) -> Result<(Vec<Option<Request>>, ActionKind), PpoError> {
    let teacher_seat = environment.policy_seat;
    let (action, mut teacher_request) =
        teacher_request_with_action(&mut environment.seats[teacher_seat])?;
    let mut requests = Vec::with_capacity(environment.seats.len());
    for index in 0..environment.seats.len() {
        let request = if index == teacher_seat {
            teacher_request.take()
        } else {
            opponent_request(&mut environment.seats[index], &mut environment.opponent)?
        };
        requests.push(request);
    }
    Ok((requests, action))
}

fn greedy_policy_request(
    seat: &mut ArenaSeatPolicy,
    model: &PolicyModel,
) -> Result<(ActionKind, Option<Request>), PpoError> {
    if deployment_uses_teacher(seat.tracker.metadata().map) {
        return teacher_request_with_action(seat);
    }
    let (frame, space) = prepare_seat_policy_sample(seat)?;
    let action = model.choose(&frame, &space).map_err(model_error)?.action;
    greedy_policy_request_in_space(seat, action, &space)
}

fn greedy_policy_request_in_space(
    seat: &mut ArenaSeatPolicy,
    proposed: crate::StructuredAction,
    space: &ActionSpace,
) -> Result<(ActionKind, Option<Request>), PpoError> {
    let action = seat
        .teacher
        .deployment_action(&seat.tracker, space)
        .unwrap_or(proposed);
    seat.local
        .note_decision(space.tick(), action.kind())
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = space
        .decode(action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let request = issue_request(seat, issued, space, action.kind(), true)?;
    Ok((action.kind(), request))
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

const fn deployment_uses_teacher(map: MapId) -> bool {
    matches!(map, MapId(0))
}

#[cfg(test)]
pub(crate) const fn deployment_uses_teacher_for_test(map: MapId) -> bool {
    deployment_uses_teacher(map)
}

fn issue_request(
    seat: &mut ArenaSeatPolicy,
    issued: Option<crate::IssuedOrder>,
    space: &ActionSpace,
    action_kind: ActionKind,
    synchronize_teacher: bool,
) -> Result<Option<Request>, PpoError> {
    let Some(issued) = seat.persistence.should_send(issued) else {
        return Ok(None);
    };
    let previous = seat.local.active_order();
    seat.sequence = seat
        .sequence
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    seat.persistence
        .record_sent(seat.sequence, issued)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    seat.readiness.note_sent(seat.sequence, issued, space);
    if matches!(
        issued.order,
        bota_proto::Order::Swap { .. } | bota_proto::Order::Use { .. }
    ) {
        if seat.readiness_orders.len() == READINESS_ORDER_HISTORY {
            seat.readiness_orders.pop_front();
        }
        seat.readiness_orders
            .push_back((seat.sequence, issued, space.tick()));
    }
    let update = crate::active_order_update_for_sent(
        &seat.persistence,
        issued.unit,
        seat.sequence,
        action_kind,
    );
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

fn reject_production_rejection(
    environment: &TrainingEnvironment,
    context: &'static str,
) -> Result<(), PpoError> {
    for (index, seat) in environment.seats.iter().enumerate() {
        if let Some((sequence, reason)) = seat.last_rejection {
            return Err(PpoError::Model(format!(
                "{context} seat {index} sequence {sequence} rejected as {reason:?}; last issued {:?}; readiness orders {:?}",
                seat.last_issued, seat.readiness_orders,
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
            ServerMsg::OrderRejected { seq, reason } => {
                seat.persistence.observe_rejection(*seq);
                seat.readiness.note_rejected(*seq);
                seat.teacher.note_rejected(*seq);
                if let Some((pending, previous)) = seat.pending_active
                    && pending == *seq
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
                seat.last_rejection = Some((*seq, *reason));
            }
            ServerMsg::Snapshot { view } => observe_arena_snapshot(seat, view)?,
            ServerMsg::Events { tick, events } => observe_arena_events(seat, *tick, events)?,
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

fn observe_arena_events(
    seat: &mut ArenaSeatPolicy,
    tick: u32,
    events: &[EventKind],
) -> Result<(), PpoError> {
    seat.tracker
        .observe_events(tick, events)
        .map_err(|error| PpoError::Model(error.to_string()))
}

fn observe_arena_snapshot(
    seat: &mut ArenaSeatPolicy,
    view: &bota_proto::WorldView,
) -> Result<(), PpoError> {
    let previous = seat.tracker.own_hero().map(|hero| hero.id);
    seat.tracker
        .observe_snapshot(view)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let current = seat.tracker.own_hero().map(|hero| hero.id);
    if previous != current {
        seat.persistence.clear_body_for(None);
        seat.local
            .set_active_order(view.tick, None)
            .map_err(|error| PpoError::Model(error.to_string()))?;
        seat.pending_active = None;
        assert!(seat.persistence.active_body_sequence_for(None).is_none());
        assert!(seat.local.active_order().is_none());
    }
    Ok(())
}

fn model_error(error: crate::ModelError) -> PpoError {
    PpoError::Model(error.to_string())
}

fn checkpoint_error(error: crate::CheckpointError) -> PpoError {
    PpoError::Model(error.to_string())
}

fn imitation_error(error: crate::ImitationError) -> PpoError {
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
