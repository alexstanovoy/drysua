#![allow(
    clippy::float_arithmetic,
    reason = "PPO optimization and metrics use floating-point arithmetic"
)]

use std::error::Error;
use std::fmt;

use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, AdamConfig, AdamState, BehavioralTarget,
    FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION, FeatureFrame, GlobalSummary, MODEL_MAX_BATCH,
    MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, PolicyIdentity, PolicyModel, StructuredAction,
};

/// Maximum concurrently interleaved environment-seat rollout streams.
pub const PPO_MAX_STREAMS: usize = 1_280;
/// Maximum transitions retained for one policy update.
pub const PPO_MAX_SAMPLES: usize = 32_768;
/// Absolute shaping budget for one episode.
pub const PPO_SHAPING_BUDGET: f32 = 100.0;
/// Terminal reward for winning; losing is its negation.
pub const PPO_TERMINAL_REWARD: f32 = 1_000.0;
/// Version of rollout, GAE, objective, optimizer, and reward semantics.
pub const PPO_SCHEMA_VERSION: u32 = 1;
/// Audited simulator rules required by stage-nine rollouts.
pub const PPO_RULES_AUDIT_VERSION: u32 = 2;
/// Canonical stage-nine learner contract covered by [`PPO_SCHEMA_HASH`].
pub const PPO_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-ppo/v1;",
    "action_schema_version=1;action_schema_hash=17797499074169920257;",
    "feature_schema_version=4;feature_schema_hash=508444194896722448;",
    "model_schema_version=3;model_schema_hash=6172692684479642043;rules_audit=2;",
    "bounds=rollout32768,streams1280,environments128,decisions256,epochs16,minibatch8192,microbatch64;",
    "actor=frozen_exact_policy_identity,legal_masked_gumbel_max_open_f64_uniform,exact_autoregressive_log_probability_and_entropy;",
    "gae=gamma_tick_pow_elapsed_ticks,lambda0.98,terminal_reset,bootstrap_truncation,normalized_advantages;",
    "objective=clipped_surrogate0.2,value_mse0.5,entropy0.01,target_kl0.02;",
    "optimizer=adam_lr3e-4_beta1_0.9_beta2_0.999_epsilon1e-5_global_clip0.5,weighted_host_microbatch_accumulation,transactional_parameters_moments_shuffle;",
    "reward=seat_safe_global_summary,potential_shaping_budget100,terminal_win1000_loss-1000_draw0,separate_breakdown;",
    "arena=one_learner_seat_against_independent_teacher,snapshot_and_events_every_tick,decision_interval3,batched_bootstrap,restart_on_terminal;"
);

const PPO_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PPO_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn ppo_fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = PPO_FNV_OFFSET;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(PPO_FNV_PRIME);
        index += 1;
    }
    hash
}

/// Stable FNV-1a hash of [`PPO_SCHEMA_DESCRIPTOR`].
pub const PPO_SCHEMA_HASH: u64 = ppo_fnv1a(PPO_SCHEMA_DESCRIPTOR.as_bytes());

const _: () = assert!(ACTION_SCHEMA_VERSION == 1);
const _: () = assert!(ACTION_SCHEMA_HASH == 17_797_499_074_169_920_257);
const _: () = assert!(FEATURE_SCHEMA_VERSION == 4);
const _: () = assert!(FEATURE_SCHEMA_HASH == 508_444_194_896_722_448);
const _: () = assert!(MODEL_SCHEMA_VERSION == 3);
const _: () = assert!(MODEL_SCHEMA_HASH == 6_172_692_684_479_642_043);

/// Stage-nine PPO hyperparameters and bounded rollout dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoConfig {
    pub decision_interval_ticks: u32,
    pub rollout_decisions: usize,
    pub environments: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub clip_epsilon: f32,
    pub value_coefficient: f32,
    pub entropy_coefficient: f32,
    pub learning_rate: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_epsilon: f32,
    pub gradient_clip: f32,
    pub gamma_tick: f32,
    pub gae_lambda: f32,
    pub target_kl: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            decision_interval_ticks: 3,
            rollout_decisions: 256,
            environments: 32,
            epochs: 4,
            minibatch: 2_048,
            clip_epsilon: 0.2,
            value_coefficient: 0.5,
            entropy_coefficient: 0.01,
            learning_rate: 3.0e-4,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1.0e-5,
            gradient_clip: 0.5,
            gamma_tick: 0.996_655_5,
            gae_lambda: 0.98,
            target_kl: 0.02,
        }
    }
}

impl PpoConfig {
    /// Validates every finite range and fixed upper bound.
    pub fn validate(self) -> Result<Self, PpoError> {
        if self.decision_interval_ticks == 0 {
            return Err(PpoError::InvalidConfig("decision interval"));
        }
        if !(1..=256).contains(&self.rollout_decisions) {
            return Err(PpoError::InvalidConfig("rollout decisions"));
        }
        if !(1..=128).contains(&self.environments) {
            return Err(PpoError::InvalidConfig("environments"));
        }
        let samples = self
            .rollout_decisions
            .checked_mul(self.environments)
            .ok_or(PpoError::InvalidConfig("samples per update"))?;
        if samples > PPO_MAX_SAMPLES
            || self.minibatch == 0
            || self.minibatch > samples
            || self.minibatch > MODEL_MAX_BATCH
        {
            return Err(PpoError::InvalidConfig("minibatch"));
        }
        if !(1..=16).contains(&self.epochs) {
            return Err(PpoError::InvalidConfig("epochs"));
        }
        validate_probabilities(self)?;
        Ok(self)
    }

    pub(crate) fn adam(self) -> AdamConfig {
        AdamConfig {
            learning_rate: self.learning_rate,
            beta1: self.adam_beta1,
            beta2: self.adam_beta2,
            epsilon: self.adam_epsilon,
            gradient_clip: self.gradient_clip,
        }
    }
}

fn validate_probabilities(config: PpoConfig) -> Result<(), PpoError> {
    let inside_unit = |value: f32| value.is_finite() && (0.0..1.0).contains(&value);
    if !inside_unit(config.gamma_tick) || !inside_unit(config.gae_lambda) {
        return Err(PpoError::InvalidConfig("discount"));
    }
    for (value, field) in [
        (config.clip_epsilon, "clip epsilon"),
        (config.value_coefficient, "value coefficient"),
        (config.entropy_coefficient, "entropy coefficient"),
        (config.target_kl, "target KL"),
        (config.learning_rate, "learning rate"),
        (config.adam_epsilon, "Adam epsilon"),
        (config.gradient_clip, "gradient clip"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PpoError::InvalidConfig(field));
        }
    }
    if !inside_unit(config.adam_beta1) || !inside_unit(config.adam_beta2) {
        return Err(PpoError::InvalidConfig("Adam beta"));
    }
    Ok(())
}

/// Stage-nine rollout, advantage, optimizer, or model integration failure.
#[derive(Clone, Debug, PartialEq)]
pub enum PpoError {
    InvalidConfig(&'static str),
    InvalidDiscount,
    InvalidTransition(&'static str),
    Capacity {
        capacity: usize,
    },
    RolloutFull {
        capacity: usize,
    },
    EmptyRollout,
    PolicyMismatch,
    StreamOutOfRange {
        stream: usize,
    },
    DecisionSequence {
        stream: usize,
        expected: u32,
        got: u32,
    },
    NonFinite(&'static str),
    CounterOverflow,
    Rollback {
        cause: String,
        rollback: String,
    },
    Model(String),
}

impl fmt::Display for PpoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(field) => write!(formatter, "invalid PPO config field: {field}"),
            Self::InvalidDiscount => formatter.write_str("invalid tick discount"),
            Self::InvalidTransition(field) => write!(formatter, "invalid PPO transition: {field}"),
            Self::Capacity { capacity } => write!(
                formatter,
                "PPO rollout capacity {capacity} is outside 1..={PPO_MAX_SAMPLES}"
            ),
            Self::RolloutFull { capacity } => {
                write!(formatter, "PPO rollout reached capacity {capacity}")
            }
            Self::EmptyRollout => formatter.write_str("PPO rollout is empty"),
            Self::PolicyMismatch => formatter.write_str("PPO rollout policy identity is stale"),
            Self::StreamOutOfRange { stream } => {
                write!(formatter, "PPO rollout stream {stream} is out of range")
            }
            Self::DecisionSequence {
                stream,
                expected,
                got,
            } => write!(
                formatter,
                "PPO stream {stream} expected decision {expected}, got {got}"
            ),
            Self::NonFinite(field) => write!(formatter, "PPO {field} is non-finite"),
            Self::CounterOverflow => formatter.write_str("PPO update counter overflow"),
            Self::Rollback { cause, rollback } => {
                write!(
                    formatter,
                    "PPO update failed ({cause}); rollback failed ({rollback})"
                )
            }
            Self::Model(message) => write!(formatter, "PPO model error: {message}"),
        }
    }
}

impl Error for PpoError {}

/// Deterministic bounded policy-sampling and minibatch-shuffle generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PpoRng {
    state: u64,
    draws: u64,
}

impl PpoRng {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            draws: 0,
        }
    }

    pub fn next_u64(&mut self) -> Result<u64, PpoError> {
        self.draws = self
            .draws
            .checked_add(1)
            .ok_or(PpoError::InvalidTransition("RNG draw overflow"))?;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Ok(value ^ (value >> 31))
    }

    /// Uniform value strictly inside `(0, 1)` using stable top 52 bits.
    pub fn uniform_open(&mut self) -> Result<f64, PpoError> {
        Ok(open_unit_from_bits(self.next_u64()? >> 12))
    }

    pub const fn draws(&self) -> u64 {
        self.draws
    }

    /// Unbiased integer in `0..bound`.
    pub fn below(&mut self, bound: u64) -> Result<u64, PpoError> {
        if bound == 0 {
            return Err(PpoError::InvalidTransition("zero random bound"));
        }
        let zone = ((1u128 << 64) / u128::from(bound)) * u128::from(bound);
        loop {
            let value = u128::from(self.next_u64()?);
            if value < zone {
                return Ok((value % u128::from(bound)) as u64);
            }
        }
    }

    #[cfg(feature = "builtin")]
    pub(crate) fn next_word(&mut self) -> Result<u64, PpoError> {
        self.next_u64()
    }

    pub(crate) fn shuffle(&mut self, order: &mut [usize]) -> Result<(), PpoError> {
        for index in (1..order.len()).rev() {
            let bound = u64::try_from(index + 1)
                .map_err(|_| PpoError::InvalidTransition("shuffle bound"))?;
            let selected = usize::try_from(self.next_u64()? % bound)
                .map_err(|_| PpoError::InvalidTransition("shuffle index"))?;
            order.swap(index, selected);
        }
        Ok(())
    }
}

fn open_unit_from_bits(bits: u64) -> f64 {
    const SCALE: f64 = 4_503_599_627_370_496.0;
    debug_assert!(bits < 1u64 << 52);
    (bits as f64 + 0.5) / SCALE
}

#[cfg(test)]
pub(crate) fn open_unit_bounds_for_test() -> (f64, f64) {
    (
        open_unit_from_bits(0),
        open_unit_from_bits((1u64 << 52) - 1),
    )
}

/// One sampled legal action before rewards and advantages are assembled.
#[derive(Clone, Debug)]
pub struct PpoTransition {
    pub(crate) frame: FeatureFrame,
    pub(crate) target: BehavioralTarget,
    pub(crate) action: StructuredAction,
    pub(crate) policy: PolicyIdentity,
    pub(crate) stream: usize,
    pub(crate) decision: u32,
    pub(crate) ticks: u32,
    pub(crate) old_log_probability: f32,
    pub(crate) old_value: f32,
    pub(crate) next_value: f32,
    pub(crate) reward: f32,
    pub(crate) terminal: bool,
}

/// Arguments that close one sampled policy decision into a rollout transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoOutcome {
    pub stream: usize,
    pub decision: u32,
    pub ticks: u32,
    pub next_value: f32,
    pub reward: f32,
    pub terminal: bool,
}

/// Sampled action and old-policy statistics returned by the model.
#[derive(Clone, Debug)]
pub struct PpoPolicyChoice {
    pub(crate) frame: FeatureFrame,
    pub(crate) target: BehavioralTarget,
    pub(crate) action: StructuredAction,
    pub(crate) policy: PolicyIdentity,
    pub(crate) log_probability: f32,
    pub(crate) entropy: f32,
    pub(crate) value: f32,
}

impl PpoPolicyChoice {
    pub const fn action(&self) -> StructuredAction {
        self.action
    }

    pub const fn policy(&self) -> PolicyIdentity {
        self.policy
    }

    pub const fn log_probability(&self) -> f32 {
        self.log_probability
    }

    pub const fn entropy(&self) -> f32 {
        self.entropy
    }

    pub const fn value(&self) -> f32 {
        self.value
    }

    /// Adds only bounded observable outcome data to this policy sample.
    pub fn finish(self, outcome: PpoOutcome) -> Result<PpoTransition, PpoError> {
        let transition = PpoTransition {
            frame: self.frame,
            target: self.target,
            action: self.action,
            policy: self.policy,
            stream: outcome.stream,
            decision: outcome.decision,
            ticks: outcome.ticks,
            old_log_probability: self.log_probability,
            old_value: self.value,
            next_value: outcome.next_value,
            reward: outcome.reward,
            terminal: outcome.terminal,
        };
        validate_transition(&transition)?;
        Ok(transition)
    }
}

fn validate_transition(transition: &PpoTransition) -> Result<(), PpoError> {
    if transition.stream >= PPO_MAX_STREAMS {
        return Err(PpoError::StreamOutOfRange {
            stream: transition.stream,
        });
    }
    if transition.ticks == 0 {
        return Err(PpoError::InvalidTransition("zero elapsed ticks"));
    }
    if transition.terminal && transition.next_value != 0.0 {
        return Err(PpoError::InvalidTransition("terminal bootstrap value"));
    }
    for (value, field) in [
        (transition.old_log_probability, "old log probability"),
        (transition.old_value, "old value"),
        (transition.next_value, "next value"),
        (transition.reward, "reward"),
    ] {
        if !value.is_finite() {
            return Err(PpoError::NonFinite(field));
        }
    }
    if transition.old_log_probability > 1.0e-5 || !transition.frame.is_finite() {
        return Err(PpoError::InvalidTransition("policy statistics or frame"));
    }
    transition
        .target
        .validate()
        .map_err(|error| PpoError::Model(error.to_string()))
}

/// Fixed-capacity rollout tied to one immutable actor policy version.
pub struct PpoRollout {
    policy: PolicyIdentity,
    capacity: usize,
    transitions: Vec<PpoTransition>,
    next_decision: [Option<u32>; PPO_MAX_STREAMS],
}

impl PpoRollout {
    pub fn new(capacity: usize, policy: PolicyIdentity) -> Result<Self, PpoError> {
        if !(1..=PPO_MAX_SAMPLES).contains(&capacity) {
            return Err(PpoError::Capacity { capacity });
        }
        Ok(Self {
            policy,
            capacity,
            transitions: Vec::with_capacity(capacity),
            next_decision: [None; PPO_MAX_STREAMS],
        })
    }

    pub fn push(&mut self, transition: PpoTransition) -> Result<(), PpoError> {
        validate_transition(&transition)?;
        if transition.policy != self.policy {
            return Err(PpoError::PolicyMismatch);
        }
        if self.transitions.len() >= self.capacity {
            return Err(PpoError::RolloutFull {
                capacity: self.capacity,
            });
        }
        let expected = self.next_decision[transition.stream].unwrap_or(transition.decision);
        if transition.decision != expected {
            return Err(PpoError::DecisionSequence {
                stream: transition.stream,
                expected,
                got: transition.decision,
            });
        }
        self.next_decision[transition.stream] = Some(
            transition
                .decision
                .checked_add(1)
                .ok_or(PpoError::CounterOverflow)?,
        );
        self.transitions.push(transition);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn finish(self, config: PpoConfig) -> Result<PpoBatch, PpoError> {
        if self.transitions.is_empty() {
            return Err(PpoError::EmptyRollout);
        }
        let config = config.validate()?;
        prepare_batch(self.policy, self.transitions, config)
    }
}

/// One transition with normalized GAE and lambda return.
#[derive(Clone, Debug)]
pub struct PpoPreparedSample {
    pub(crate) transition: PpoTransition,
    pub(crate) advantage: f32,
    pub(crate) return_value: f32,
}

impl PpoPreparedSample {
    pub const fn action(&self) -> StructuredAction {
        self.transition.action
    }
    pub const fn advantage(&self) -> f32 {
        self.advantage
    }
    pub const fn return_value(&self) -> f32 {
        self.return_value
    }
}

/// Immutable normalized update batch from one actor policy revision.
pub struct PpoBatch {
    policy: PolicyIdentity,
    samples: Vec<PpoPreparedSample>,
}

/// One model minibatch result before trainer-level aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PpoMinibatchReport {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub clip_fraction: f64,
    pub gradient_norm: f64,
    pub applied_scale: f64,
    pub samples: usize,
    pub applied: bool,
}

/// One complete PPO update report across epochs and minibatches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PpoUpdateReport {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub clip_fraction: f64,
    pub gradient_norm: f64,
    pub applied_scale: f64,
    pub samples_optimized: usize,
    pub minibatches: usize,
    pub epochs_completed: usize,
    pub stopped_for_kl: bool,
    pub optimizer_step: u64,
    pub update: u64,
}

/// Exclusive PPO optimizer owner with deterministic bounded shuffling.
pub struct PpoTrainer {
    config: PpoConfig,
    adam: AdamState,
    shuffle: PpoRng,
    updates: u64,
}

impl PpoTrainer {
    pub fn new(model: &PolicyModel, config: PpoConfig, seed: u64) -> Result<Self, PpoError> {
        let config = config.validate()?;
        let adam = model
            .claim_optimizer(config.adam())
            .map_err(|error| PpoError::Model(error.to_string()))?;
        Ok(Self {
            config,
            adam,
            shuffle: PpoRng::new(seed),
            updates: 0,
        })
    }

    pub const fn config(&self) -> PpoConfig {
        self.config
    }

    pub const fn optimizer_step(&self) -> u64 {
        self.adam.step()
    }

    #[cfg(test)]
    pub(crate) const fn shuffle_draws_for_test(&self) -> u64 {
        self.shuffle.draws()
    }

    pub fn train_update(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
    ) -> Result<PpoUpdateReport, PpoError> {
        let current = model
            .policy_identity()
            .map_err(|error| PpoError::Model(error.to_string()))?;
        if current != batch.policy || self.adam.policy_identity() != current {
            return Err(PpoError::PolicyMismatch);
        }
        let snapshot = model
            .coherent_snapshot(&self.adam)
            .map_err(|error| PpoError::Model(error.to_string()))?;
        let shuffle = self.shuffle.clone();
        let optimizer_step = self.adam.step();
        match self.train_update_inner(model, batch) {
            Ok(report) => Ok(report),
            Err(error) => {
                self.shuffle = shuffle;
                if self.adam.step() == optimizer_step {
                    return Err(error);
                }
                let expected = self.adam.binding();
                model
                    .restore_snapshot(&snapshot, &mut self.adam, expected)
                    .map_err(|rollback| PpoError::Rollback {
                        cause: error.to_string(),
                        rollback: rollback.to_string(),
                    })?;
                Err(error)
            }
        }
    }

    fn train_update_inner(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
    ) -> Result<PpoUpdateReport, PpoError> {
        let mut aggregate = PpoUpdateReport::default();
        let mut order = (0..batch.samples.len()).collect::<Vec<_>>();
        'epochs: for epoch in 0..self.config.epochs {
            self.shuffle.shuffle(&mut order)?;
            for indices in order.chunks(self.config.minibatch) {
                let samples = indices
                    .iter()
                    .map(|index| &batch.samples[*index])
                    .collect::<Vec<_>>();
                let report = model
                    .ppo_update(&samples, &mut self.adam, self.config)
                    .map_err(|error| PpoError::Model(error.to_string()))?;
                if !report.applied {
                    aggregate.stopped_for_kl = true;
                    break 'epochs;
                }
                aggregate_minibatch(&mut aggregate, report)?;
            }
            aggregate.epochs_completed = epoch + 1;
        }
        finish_update_report(&mut aggregate, self.adam.step())?;
        self.updates = self
            .updates
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        aggregate.update = self.updates;
        Ok(aggregate)
    }
}

fn aggregate_minibatch(
    aggregate: &mut PpoUpdateReport,
    report: PpoMinibatchReport,
) -> Result<(), PpoError> {
    aggregate.policy_loss += report.policy_loss * report.samples as f64;
    aggregate.value_loss += report.value_loss * report.samples as f64;
    aggregate.entropy += report.entropy * report.samples as f64;
    aggregate.approximate_kl += report.approximate_kl * report.samples as f64;
    aggregate.clip_fraction += report.clip_fraction * report.samples as f64;
    aggregate.gradient_norm += report.gradient_norm;
    aggregate.applied_scale += report.applied_scale;
    aggregate.samples_optimized = aggregate
        .samples_optimized
        .checked_add(report.samples)
        .ok_or(PpoError::CounterOverflow)?;
    aggregate.minibatches = aggregate
        .minibatches
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    Ok(())
}

fn finish_update_report(report: &mut PpoUpdateReport, optimizer_step: u64) -> Result<(), PpoError> {
    if report.samples_optimized == 0 || report.minibatches == 0 {
        return Err(PpoError::InvalidTransition("no PPO minibatch applied"));
    }
    let samples = report.samples_optimized as f64;
    report.policy_loss /= samples;
    report.value_loss /= samples;
    report.entropy /= samples;
    report.approximate_kl /= samples;
    report.clip_fraction /= samples;
    report.gradient_norm /= report.minibatches as f64;
    report.applied_scale /= report.minibatches as f64;
    report.optimizer_step = optimizer_step;
    Ok(())
}

impl PpoBatch {
    pub const fn policy(&self) -> PolicyIdentity {
        self.policy
    }
    pub fn samples(&self) -> &[PpoPreparedSample] {
        &self.samples
    }

    #[cfg(test)]
    pub(crate) fn replace_advantage_for_test(&mut self, index: usize, value: f32) -> f32 {
        std::mem::replace(&mut self.samples[index].advantage, value)
    }
}

fn prepare_batch(
    policy: PolicyIdentity,
    transitions: Vec<PpoTransition>,
    config: PpoConfig,
) -> Result<PpoBatch, PpoError> {
    let mut next_advantage = [0.0f32; PPO_MAX_STREAMS];
    let mut prepared = Vec::with_capacity(transitions.len());
    for transition in transitions.into_iter().rev() {
        let discount = tick_discount(config.gamma_tick, transition.ticks)?;
        let continuation = if transition.terminal { 0.0 } else { 1.0 };
        let delta = transition.reward + discount * transition.next_value * continuation
            - transition.old_value;
        let advantage =
            delta + discount * config.gae_lambda * next_advantage[transition.stream] * continuation;
        if !advantage.is_finite() {
            return Err(PpoError::NonFinite("advantage"));
        }
        next_advantage[transition.stream] = advantage;
        prepared.push(PpoPreparedSample {
            return_value: transition.old_value + advantage,
            transition,
            advantage,
        });
    }
    prepared.reverse();
    normalize_advantages(&mut prepared)?;
    Ok(PpoBatch {
        policy,
        samples: prepared,
    })
}

fn normalize_advantages(samples: &mut [PpoPreparedSample]) -> Result<(), PpoError> {
    let count = samples.len() as f64;
    let mean = samples
        .iter()
        .map(|sample| f64::from(sample.advantage))
        .sum::<f64>()
        / count;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = f64::from(sample.advantage) - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    let deviation = variance.sqrt();
    let divisor = deviation.max(1.0e-8);
    for sample in samples {
        sample.advantage = ((f64::from(sample.advantage) - mean) / divisor) as f32;
        if !sample.advantage.is_finite() || !sample.return_value.is_finite() {
            return Err(PpoError::NonFinite("normalized advantage or return"));
        }
    }
    Ok(())
}

/// Discount over an exact positive number of elapsed simulation ticks.
pub fn tick_discount(gamma_tick: f32, ticks: u32) -> Result<f32, PpoError> {
    if !gamma_tick.is_finite() || !(0.0..1.0).contains(&gamma_tick) || ticks == 0 {
        return Err(PpoError::InvalidDiscount);
    }
    let exponent = i32::try_from(ticks).map_err(|_| PpoError::InvalidDiscount)?;
    let discount = gamma_tick.powi(exponent);
    discount
        .is_finite()
        .then_some(discount)
        .ok_or(PpoError::InvalidDiscount)
}

/// Scalar PPO clipped surrogate used by reference tests and diagnostics.
pub fn clipped_surrogate(ratio: f32, advantage: f32, epsilon: f32) -> f32 {
    assert!(ratio.is_finite());
    assert!(advantage.is_finite());
    assert!(epsilon.is_finite());
    assert!(epsilon > 0.0);
    (ratio * advantage).min(ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage)
}

/// Observable shaping and terminal reward components for one seat transition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardBreakdown {
    pub experience: f32,
    pub last_hits: f32,
    pub denies: f32,
    pub combat: f32,
    pub structures: f32,
    pub wealth: f32,
    pub terminal: f32,
    pub total: f32,
}

/// Explicit terminal adjudication; a draw is not a nonterminal step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpoTerminalOutcome {
    Win,
    Loss,
    Draw,
}

/// Bounded seat-only reward state; no simulator state or hidden identifiers enter it.
#[derive(Clone, Debug, Default)]
pub struct RewardTracker {
    previous: Option<GlobalSummary>,
    shaping_total: f32,
}

impl RewardTracker {
    pub fn observe(
        &mut self,
        next: GlobalSummary,
        discount: f32,
        outcome: Option<PpoTerminalOutcome>,
    ) -> Result<RewardBreakdown, PpoError> {
        if !discount.is_finite() || !(0.0..=1.0).contains(&discount) {
            return Err(PpoError::InvalidDiscount);
        }
        let mut reward = self
            .previous
            .map_or_else(RewardBreakdown::default, |previous| {
                shaping_delta(previous, next, discount)
            });
        let proposed = reward.total;
        let allowed = (self.shaping_total + proposed)
            .clamp(-PPO_SHAPING_BUDGET, PPO_SHAPING_BUDGET)
            - self.shaping_total;
        scale_shaping(&mut reward, proposed, allowed);
        self.shaping_total += allowed;
        reward.terminal = outcome.map_or(0.0, |outcome| match outcome {
            PpoTerminalOutcome::Win => PPO_TERMINAL_REWARD,
            PpoTerminalOutcome::Loss => -PPO_TERMINAL_REWARD,
            PpoTerminalOutcome::Draw => 0.0,
        });
        reward.total += reward.terminal;
        self.previous = Some(next);
        Ok(reward)
    }
}

fn shaping_delta(previous: GlobalSummary, next: GlobalSummary, discount: f32) -> RewardBreakdown {
    let potential = |next: f32, previous: f32| discount * next - previous;
    let experience = potential(score_xp(next), score_xp(previous)) * 0.01;
    let last_hits = potential(score_last_hits(next), score_last_hits(previous)) * 0.2;
    let denies = potential(score_denies(next), score_denies(previous)) * 0.1;
    let combat = potential(score_combat(next), score_combat(previous)) * 2.0;
    let structures = potential(score_structures(next), score_structures(previous)) * 5.0;
    let wealth = potential(next.own_gold as f32, previous.own_gold as f32) * 0.001;
    RewardBreakdown {
        experience,
        last_hits,
        denies,
        combat,
        structures,
        wealth,
        terminal: 0.0,
        total: experience + last_hits + denies + combat + structures + wealth,
    }
}

fn score_xp(summary: GlobalSummary) -> f32 {
    (summary.allied.xp - summary.enemy.xp) as f32
}

fn score_last_hits(summary: GlobalSummary) -> f32 {
    (summary.allied.last_hits as i64 - summary.enemy.last_hits as i64) as f32
}

fn score_denies(summary: GlobalSummary) -> f32 {
    (summary.allied.denies as i64 - summary.enemy.denies as i64) as f32
}

fn score_combat(summary: GlobalSummary) -> f32 {
    let allied = summary.allied.kills as i64 - summary.allied.deaths as i64;
    let enemy = summary.enemy.kills as i64 - summary.enemy.deaths as i64;
    (allied - enemy) as f32
}

fn score_structures(summary: GlobalSummary) -> f32 {
    (i64::from(summary.enemy_structures_destroyed) - i64::from(summary.allied_structures_destroyed))
        as f32
}

fn scale_shaping(reward: &mut RewardBreakdown, proposed: f32, allowed: f32) {
    if proposed == 0.0 || proposed == allowed {
        reward.total = allowed;
        return;
    }
    let scale = allowed / proposed;
    reward.experience *= scale;
    reward.last_hits *= scale;
    reward.denies *= scale;
    reward.combat *= scale;
    reward.structures *= scale;
    reward.wealth *= scale;
    reward.total = allowed;
}
