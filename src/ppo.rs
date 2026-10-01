#![allow(
    clippy::float_arithmetic,
    reason = "PPO optimization and metrics use floating-point arithmetic"
)]

mod minibatch;

use std::error::Error;
use std::fmt;

use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, ActionHeadTargets, AdamConfig, AdamState,
    FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION, FeatureFrame, MODEL_MAX_BATCH, MODEL_SCHEMA_HASH,
    MODEL_SCHEMA_VERSION, PackedActionHeadTargets, PolicyModel, RaggedFeatureArena,
    RaggedFeatureHeader, StagedPpoBatch, StructuredAction,
};

#[cfg(test)]
#[path = "tests/ppo_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;

/// Maximum epoch, optimizer-step, and global-update counter value.
pub const MAX_TRAINING_COUNTER: u64 = 1_000_000_000;
/// Maximum concurrently interleaved environment-seat rollout streams.
pub const PPO_MAX_STREAMS: usize = 1_280;
/// Maximum transitions retained for one policy update: the largest target plus
/// the two intervals each slot may close in the round that reaches it.
pub const PPO_MAX_SAMPLES: usize = PPO_MAX_UPDATE_SAMPLES + 2 * PPO_MAX_SLOTS;
/// Maximum simultaneously active worlds of one evaluation batch.
pub const PPO_MAX_PARALLEL_WORLDS: usize = 64;
/// Maximum concurrent training world slots.
pub const PPO_MAX_SLOTS: usize = 256;
/// Largest retained-interval target of one update.
pub const PPO_MAX_UPDATE_SAMPLES: usize = 32_768;
const _: () = assert!(PPO_MAX_UPDATE_SAMPLES + 2 * PPO_MAX_SLOTS <= PPO_MAX_SAMPLES);
const _: () = assert!(
    std::mem::size_of::<CompactPpoTransition>()
        + std::mem::size_of::<CompactPreparedSample>()
        + std::mem::size_of::<usize>()
        <= 8_448
);
const _: () = assert!(std::mem::size_of::<PpoPreparedSample>() <= 71_000);
const _: () = assert!(PPO_MAX_PARALLEL_WORLDS <= crate::MODEL_TRAINING_BATCH);
/// Conservative rollout storage plus one fully materialized effective minibatch.
/// Includes arena reallocation overlap, both preparation vectors and shuffle order;
/// excludes allocator overhead, model/optimizer tensors and native simulator worlds.
pub const PPO_STORAGE_PEAK_BYTES: u64 = crate::feature::FEATURE_ARENA_PEAK_BYTES
    + PPO_MAX_SAMPLES as u64
        * (std::mem::size_of::<CompactPpoTransition>()
            + std::mem::size_of::<CompactPreparedSample>()
            + std::mem::size_of::<usize>()) as u64
    + MODEL_MAX_BATCH as u64
        * (std::mem::size_of::<PpoPreparedSample>() + crate::FEATURE_FRAME_HEAP_BYTES) as u64;
const _: () = assert!(PPO_MAX_SAMPLES == 33_280);
const _: () = assert!(PPO_STORAGE_PEAK_BYTES < 10 * 1024 * 1024 * 1024);
/// Maximum updates between a sample's behaviour weights and the learner: one
/// pipelined update plus an interval that straddles an update boundary.
pub const PPO_MAX_STALENESS: u64 = 2;
/// Maximum random draws made by one autoregressive policy sample.
pub const PPO_MAX_POLICY_SAMPLE_DRAWS: u64 = 132;
/// Version of rollout, GAE, objective, optimizer, and reward semantics.
pub const PPO_SCHEMA_VERSION: u32 = 44;
/// Version of the simulator and learner rules rollouts assume.
pub const PPO_RULES_AUDIT_VERSION: u32 = 32;
/// Learner contract covered by [`PPO_SCHEMA_HASH`].
pub const PPO_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-ppo/v44;",
    "linked_schemas=action,feature,model,map2_reward;linked_hash=fnv1a_descriptor_then_ordered_version_le32_hash_le64_then_map2_reward_descriptor_utf8;rules_audit=32;",
    "scope=map2_mid_only_dota_geometry_mid_waves_second_hero_death_or_first_tower_loss_simultaneous_draw_cap27900_including900_pregame_cap_tick_draw;",
    "collection=continuous_slots1to256_back_to_back_games,lanes_divide_slots_max64_slots_per_lane,update_due_after_whole_lane_rounds_reaching_samples_per_update_over_lanes,in_flight_intervals_continue_under_next_weights,actor_weights_lag_learner_by_pipeline_staleness_at_most2_with_boundary_intervals,per_update_opponent_mixture_teacher_frozen_weights_selfplay_league_snapshots_every_n_updates_pfsp_weight_times_one_minus_laplace_winrate_draw_is_nonwin_over_last100_games_squared_integer_from_outcomes_of_updates_every_lane_finished;",
    "candidate_order=live_neural_ppo_learner_and_neural_opponents_by_prepared_action,effective_directive_ledger_follows_all_actual_sends;teacher=original_strategy_no_learner_override;",
    "bounds=streams1280,samples_per_update32768,max_samples33280,slots256,epochs16,minibatch8192,microbatch64_128_256;",
    "complete_episodes=map2_balanced_sides,retain_first_every_noncontinue_continue_after8_decision_interval_else_with_probability1_8_from_game_seed_and_decision,inverse_retention_probability_weight,retain_original_action_logprob_exact_elapsed_ticks_and_all_intervening_reward,terminal_zero_bootstrap_partial_flush,no_synthetic_zero_tick_samples,empty_optimizer_batch_rejected;lambda1=full_monte_carlo_f64_return_recurrence;",
    "terminal=win1_loss-1_draw-.5_timecap-.5,victory_time=win_only_native_ticks_full.2_to9000_linear_to0_at21600,draw_and_timecap_are_nonwins_distinct_labels,infrastructure_failure_invalidates_not_fabricated_outcome;",
    "wire_rebase=bota78427bb_missed_event_ignored_without_damage_or_healing_cheat_order_never_issued_or_honoured_NoCheats_rejected_without_reward,attack_time_ms_converted_to_ticks,bound_combat_and_collision_clearance_no_terminal_or_shaping_change;",
    "actor=per_lane_weight_replica_recorded_behaviour_version,batch_max128_single_shared_trunk_forward_policy_and_selfplay_rows,side_selected_radiant_dire_actor_heads,per_game_rng_from_seed_slot_game,transactional_batch_rng,legal_masked_gumbel_max_open_f64_uniform,exact_autoregressive_log_probability_and_entropy_for_retained_rows;",
    "gae=map2_gamma_tick1_required,lambda_per_elapsed_tick0.9997912,terminal_reset,bootstrap_collector_truncation_not_task_terminal,normalized_advantages_times_inverse_retention_probability_over_batch_mean;",
    "objective=clipped_surrogate0.2,value_mse0.5,entropy0.004,target_kl0.02;",
    "critic=value_mlp_on_shared_trunk,value_loss_trains_trunk;",
    "optimizer=adam_lr1e-5_beta1_0.9_beta2_0.999_epsilon1e-5_global_clip0.5,weighted_host_microbatch_accumulation,transactional_parameters_moments_shuffle;",
    "kl_guard=pre_step_rejection,post_step_sample_weighted_complete_effective_minibatch_rollout_policy_kl,candidate_exceeds_target_or_evaluation_error_restores_exact_parameters_adam_moments_step_policy_revision_under_exclusive_parameter_lock,applied_report_post_step_kl,rejected_report_candidate_kl;",
    "reward=linked_map2_reward_schema_version_hash_and_full_descriptor,seat_only_full_contiguous_snapshot_events_before_retention_or_tracker_journal;no_action_or_Teacher_override;",
    "arena=one_learner_seat_against_independent_frozen_opponent,snapshot_then_explicit_events_including_empty_complete_every_visible_tick,decision_after_tick_complete,decision_interval3,pregame_enabled,batched_bootstrap,hero_identity_change_invalidates_local_body_order,hero_active_order_feature_ignores_courier_orders;",
    "navigation=existing_walkable_building_landing_points_allow_MovePoint_only,AttackMovePoint_source_veto_unchanged,no_goal_features_or_forced_retreat,seat_visible_channel_masks_cast_and_use,seat_visible_item_mute_masks_use,put_point_underfoot_only;",
    "deployment=raw_map2_mid_neural_policy_no_teacher_override_or_strategic_masks;",
    "teacher_economy=custom_bota_wraith_band_tango_boots_optional_stick_gloves_belt_once_only;",
    "initialization=runtime_weights_named_tensors_equal_name_and_shape_reused_others_fresh_any_linked_schema_metadata_parameters_only_fresh_optimizer_progress_rng;"
);

/// FNV-1a of the descriptor, ordered linked identities, and reward version.
pub const PPO_SCHEMA_HASH: u64 = crate::model::linked_schema_hash(
    PPO_SCHEMA_DESCRIPTOR,
    &[
        (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
        (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
    ],
);

const _: () = assert!(ACTION_SCHEMA_VERSION == 8);
const _: () = assert!(FEATURE_SCHEMA_VERSION == 26);
const _: () = assert!(MODEL_SCHEMA_VERSION == 27);
const _: () = assert!(PPO_RULES_AUDIT_VERSION == 32);

/// PPO hyperparameters and bounded rollout dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoConfig {
    pub decision_interval_ticks: u32,
    /// Retained intervals that make one update due.
    pub samples_per_update: usize,
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
    /// GAE trace decay per simulation tick, so the credit horizon is fixed in
    /// game time however densely decisions are retained.
    pub gae_lambda_tick: f32,
    pub target_kl: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            decision_interval_ticks: 3,
            // About sixteen games when every order and each eighth Continue is retained.
            samples_per_update: 24_000,
            epochs: 4,
            minibatch: 2_048,
            clip_epsilon: 0.2,
            value_coefficient: 0.5,
            entropy_coefficient: 0.004,
            learning_rate: 1.0e-5,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1.0e-5,
            gradient_clip: 0.5,
            gamma_tick: 0.996_655_5,
            // 0.995 per 24-tick interval: a 4,800-tick (160 s) credit horizon.
            gae_lambda_tick: 0.999_791_2,
            target_kl: 0.02,
        }
    }
}

impl PpoConfig {
    pub fn validate(self) -> Result<Self, PpoError> {
        if self.decision_interval_ticks == 0 {
            return Err(PpoError::InvalidConfig("decision interval"));
        }
        if !(1..=PPO_MAX_UPDATE_SAMPLES).contains(&self.samples_per_update) {
            return Err(PpoError::InvalidConfig("samples per update"));
        }
        if self.minibatch == 0
            || self.minibatch > self.samples_per_update
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

    /// Retained intervals one update can hold: every lane closes whole rounds,
    /// and one round closes at most two intervals per slot.
    pub const fn rollout_capacity(self, slots: usize) -> usize {
        assert!(slots <= PPO_MAX_SLOTS);
        self.samples_per_update + 2 * slots
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
    if !config.gamma_tick.is_finite()
        || !(0.0..=1.0).contains(&config.gamma_tick)
        || !config.gae_lambda_tick.is_finite()
        || !(0.0..=1.0).contains(&config.gae_lambda_tick)
    {
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

/// Rollout, advantage, optimizer or model failure.
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
    /// A checkpoint's run scope differs from the requested one, by field.
    ScopeMismatch(String),
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
            Self::ScopeMismatch(message) => {
                write!(formatter, "checkpoint scope mismatch: {message}")
            }
        }
    }
}

impl Error for PpoError {}

/// SplitMix64 generator for policy sampling and minibatch shuffles; counts its draws.
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

    pub(crate) fn from_checkpoint(state: u64, draws: u64) -> Result<Self, PpoError> {
        if draws > crate::MAX_TRAINING_COUNTER {
            return Err(PpoError::CounterOverflow);
        }
        Ok(Self { state, draws })
    }

    pub const fn checkpoint(&self) -> (u64, u64) {
        (self.state, self.draws)
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

/// One sampled legal action before rewards and advantages are assembled.
#[derive(Clone, Debug)]
pub struct PpoTransition {
    pub(crate) frame: FeatureFrame,
    pub(crate) target: ActionHeadTargets,
    /// A shadow rule policy's label of the same decision, for the imitation term.
    pub(crate) shadow: Option<ActionHeadTargets>,
    pub(crate) action: StructuredAction,
    /// Completed updates of the actor weights that sampled `action`.
    pub(crate) behaviour: u64,
    pub(crate) stream: usize,
    pub(crate) decision: u32,
    pub(crate) ticks: u32,
    pub(crate) old_log_probability: f32,
    pub(crate) old_value: f32,
    pub(crate) next_value: f32,
    pub(crate) reward: f32,
    pub(crate) terminal: bool,
    /// Inverse probability that the decision was retained; scales its policy gradient.
    pub(crate) weight: f32,
}

pub(crate) fn validate_transition(transition: &PpoTransition) -> Result<(), PpoError> {
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
    if !(1.0..=crate::MAP2_CONTINUE_STRIDE as f32).contains(&transition.weight) {
        return Err(PpoError::InvalidTransition("retention weight"));
    }
    if transition.old_log_probability > 1.0e-5 || !transition.frame.is_finite() {
        return Err(PpoError::InvalidTransition("policy statistics or frame"));
    }
    let invalid = |error: crate::TargetError| PpoError::Model(error.to_string());
    transition.target.validate().map_err(invalid)?;
    if let Some(shadow) = &transition.shadow {
        shadow.validate().map_err(invalid)?;
    }
    Ok(())
}

/// Fixed-capacity rollout of one update; samples carry their behaviour versions.
pub struct PpoRollout {
    capacity: usize,
    transitions: Vec<CompactPpoTransition>,
    frames: RaggedFeatureArena,
    next_decision: [Option<u32>; PPO_MAX_STREAMS],
}

struct CompactPpoTransition {
    frame: RaggedFeatureHeader,
    target: PackedActionHeadTargets,
    shadow: Option<PackedActionHeadTargets>,
    action: StructuredAction,
    behaviour: u64,
    stream: usize,
    decision: u32,
    ticks: u32,
    old_log_probability: f32,
    old_value: f32,
    next_value: f32,
    reward: f32,
    terminal: bool,
    weight: f32,
    radiant: bool,
}

impl PpoRollout {
    /// A rollout holding at most `capacity` transitions.
    pub fn new(capacity: usize) -> Result<Self, PpoError> {
        if !(1..=PPO_MAX_SAMPLES).contains(&capacity) {
            return Err(PpoError::Capacity { capacity });
        }
        Ok(Self {
            capacity,
            transitions: rollout_storage(capacity)?,
            frames: RaggedFeatureArena::new(capacity).map_err(PpoError::InvalidTransition)?,
            next_decision: [None; PPO_MAX_STREAMS],
        })
    }

    pub fn push(&mut self, transition: PpoTransition) -> Result<(), PpoError> {
        validate_transition(&transition)?;
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
        let next_decision = transition
            .decision
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        let frame = self
            .frames
            .push(&transition.frame)
            .map_err(PpoError::InvalidTransition)?;
        self.next_decision[transition.stream] = Some(next_decision);
        self.transitions.push(CompactPpoTransition {
            frame,
            target: transition.target.pack(),
            shadow: transition.shadow.as_ref().map(ActionHeadTargets::pack),
            action: transition.action,
            behaviour: transition.behaviour,
            stream: transition.stream,
            decision: transition.decision,
            ticks: transition.ticks,
            old_log_probability: transition.old_log_probability,
            old_value: transition.old_value,
            next_value: transition.next_value,
            reward: transition.reward,
            terminal: transition.terminal,
            weight: transition.weight,
            radiant: transition.frame.global[crate::global_feature::SIDE_RADIANT] == 1.0,
        });
        Ok(())
    }

    #[cfg(any(feature = "builtin", test))]
    pub(crate) fn len(&self) -> usize {
        self.transitions.len()
    }

    /// The normalized batch; separate side networks normalize each side's
    /// advantages on their own, as each network's own batch.
    pub fn finish(
        self,
        config: PpoConfig,
        side_networks: crate::SideNetworks,
    ) -> Result<PpoBatch, PpoError> {
        if self.transitions.is_empty() {
            return Err(PpoError::EmptyRollout);
        }
        let config = config.validate()?;
        prepare_batch(self.transitions, self.frames, config, side_networks)
    }
}

/// One transition with its normalized, retention-weighted GAE and lambda return.
#[derive(Clone, Debug)]
pub struct PpoPreparedSample {
    pub(crate) transition: PpoTransition,
    pub(crate) advantage: f32,
    pub(crate) return_value: f32,
}

/// Immutable normalized update batch.
pub struct PpoBatch {
    samples: Vec<CompactPreparedSample>,
    frames: RaggedFeatureArena,
    /// Monte Carlo return of each sample whose game ended in this batch.
    outcome_returns: Vec<Option<f32>>,
    /// Radiant, then dire.
    sides: [BatchSideStatistics; 2],
}

/// Rollout statistics of one side's samples, before the update trains on them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchSideStatistics {
    pub samples: usize,
    /// Mean and standard deviation of the raw GAE advantages.
    pub advantage_mean: f64,
    pub advantage_deviation: f64,
    /// Mean advantage after the batch-wide normalization the update trains on.
    pub normalized_advantage_mean: f64,
    pub return_mean: f64,
    pub value_mean: f64,
    /// Explained variance of the lambda returns by the behaviour values.
    pub explained_variance: f64,
    /// Mean negative behaviour log-probability, a sampled estimate of the
    /// behaviour policy's entropy summed over its active heads.
    pub behaviour_entropy: f64,
    /// Share of samples that begin with Continue, and the mean normalized
    /// advantage of those and of the others.
    pub continue_share: f64,
    pub continue_advantage_mean: f64,
    pub action_advantage_mean: f64,
}

/// How much return variance the rollout critic explained before an update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplainedVariance {
    /// Against the lambda returns it trains on, which bootstrap from itself.
    pub lambda: f64,
    /// Against the Monte Carlo return of the samples whose game ended in the
    /// batch: the outcome it has to predict, never its own bootstrap.
    pub monte_carlo: f64,
}

struct CompactPreparedSample {
    transition: CompactPpoTransition,
    advantage: f32,
    return_value: f32,
}

/// Auxiliary terms of one update: a pure function of the update index and the
/// run scope, so a resumed run recomputes them exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UpdateObjective {
    /// Weight of the cross entropy to the shadow labels; zero leaves PPO unchanged.
    pub imitation: f32,
    /// Power in `[0, 1]` of the inverse imitation-class frequency each label is
    /// weighted by; 0 weighs every label alike.
    pub imitation_balance: f32,
    /// Trains only the critic head; every other parameter keeps its bits.
    pub critic_only: bool,
}

/// Imitation statistics summed over optimized rows, in shadow-label head order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImitationReport {
    /// Summed cross entropy of every labeled head.
    pub cross_entropy: f64,
    /// Rows with a shadow label.
    pub labeled: f64,
    /// Labeled rows whose legal argmax matches the label on every labeled head.
    pub action_agreements: f64,
    /// Per head: labeled rows whose legal argmax matches the label.
    pub head_agreements: [f64; crate::MODEL_ACTION_HEADS],
    /// Per head: rows the label defines.
    pub head_labels: [f64; crate::MODEL_ACTION_HEADS],
    /// Per imitation class: labeled rows agreeing on every labeled head.
    pub class_agreements: [f64; crate::IMITATION_CLASSES.len()],
    /// Per imitation class: labeled rows.
    pub class_labels: [f64; crate::IMITATION_CLASSES.len()],
}

impl ImitationReport {
    fn add(&mut self, other: &Self) {
        self.cross_entropy += other.cross_entropy;
        self.labeled += other.labeled;
        self.action_agreements += other.action_agreements;
        for head in 0..crate::MODEL_ACTION_HEADS {
            self.head_agreements[head] += other.head_agreements[head];
            self.head_labels[head] += other.head_labels[head];
        }
        for class in 0..crate::IMITATION_CLASSES.len() {
            self.class_agreements[class] += other.class_agreements[class];
            self.class_labels[class] += other.class_labels[class];
        }
    }
}

/// Per-row quantities [`SideReport`] sums besides the head entropies: rows,
/// policy loss, value loss, entropy, approximate KL, clipped rows, imitation
/// cross entropy and imitation-labeled rows.
pub const SIDE_QUANTITIES: usize = 8;

/// Columns of one [`SideReport`] row: the side quantities, then every head's entropy.
const SIDE_COLUMNS: usize = SIDE_QUANTITIES + crate::MODEL_ACTION_HEADS;

/// Sums over optimized rows of every row and of the dire rows alone; a radiant
/// sum is the difference. The KL is the gradient forward's, before the step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideReport {
    pub all: [f64; SIDE_COLUMNS],
    pub dire: [f64; SIDE_COLUMNS],
}

impl Default for SideReport {
    fn default() -> Self {
        Self {
            all: [0.0; SIDE_COLUMNS],
            dire: [0.0; SIDE_COLUMNS],
        }
    }
}

impl SideReport {
    pub(crate) fn from_sums(sums: &[f32]) -> Self {
        assert_eq!(sums.len(), 2 * SIDE_COLUMNS);
        Self {
            all: std::array::from_fn(|column| f64::from(sums[column])),
            dire: std::array::from_fn(|column| f64::from(sums[SIDE_COLUMNS + column])),
        }
    }

    fn add(&mut self, other: &Self) {
        for column in 0..SIDE_COLUMNS {
            self.all[column] += other.all[column];
            self.dire[column] += other.dire[column];
        }
    }

    /// Per-row means of one side (`dire` false is radiant), or `None` without rows.
    pub fn means(&self, dire: bool) -> Option<[f64; SIDE_COLUMNS]> {
        let sums: [f64; SIDE_COLUMNS] = std::array::from_fn(|column| {
            if dire {
                self.dire[column]
            } else {
                self.all[column] - self.dire[column]
            }
        });
        let rows = sums[0];
        (rows > 0.0).then(|| {
            std::array::from_fn(|column| match column {
                0 => rows,
                6 if sums[7] > 0.0 => sums[6] / sums[7],
                6 => f64::NAN,
                7 => sums[7],
                _ => sums[column] / rows,
            })
        })
    }
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
    pub imitation: ImitationReport,
    pub sides: SideReport,
}

/// One complete PPO update report across epochs and minibatches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PpoUpdateReport {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub rejected_kl: f64,
    pub clip_fraction: f64,
    pub gradient_norm: f64,
    pub applied_scale: f64,
    pub samples_optimized: usize,
    pub samples_rejected: usize,
    pub minibatches: usize,
    pub epochs_completed: usize,
    pub stopped_for_kl: bool,
    pub optimizer_step: u64,
    pub update: u64,
    /// The auxiliary terms this update trained with.
    pub objective: UpdateObjective,
    pub imitation: ImitationReport,
    pub sides: SideReport,
}

/// Exclusive PPO optimizer owner with deterministic bounded shuffling.
pub struct PpoTrainer {
    execution: crate::TrainingExecutionOptions,
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
            execution: crate::TrainingExecutionOptions::default(),
        })
    }

    pub const fn config(&self) -> PpoConfig {
        self.config
    }

    /// Reapply operational choices after strict checkpoint restoration.
    pub fn set_execution(
        &mut self,
        execution: crate::TrainingExecutionOptions,
    ) -> Result<(), PpoError> {
        self.execution = execution.validate()?;
        Ok(())
    }

    pub const fn optimizer_step(&self) -> u64 {
        self.adam.step()
    }

    pub const fn updates(&self) -> u64 {
        self.updates
    }

    pub const fn rng_checkpoint(&self) -> (u64, u64) {
        (self.shuffle.state, self.shuffle.draws)
    }

    pub(crate) fn checkpoint_snapshot(
        &self,
        model: &PolicyModel,
    ) -> Result<crate::ModelAdamSnapshot, PpoError> {
        model
            .coherent_snapshot(&self.adam)
            .map_err(|error| PpoError::Model(error.to_string()))
    }

    pub(crate) fn restore_checkpoint(
        config: PpoConfig,
        adam: AdamState,
        shuffle: (u64, u64),
        updates: u64,
    ) -> Result<Self, PpoError> {
        let config = config.validate()?;
        if updates > crate::MAX_TRAINING_COUNTER {
            return Err(PpoError::CounterOverflow);
        }
        if adam.config() != config.adam() {
            return Err(PpoError::InvalidConfig("checkpoint Adam config"));
        }
        Ok(Self {
            config,
            adam,
            shuffle: PpoRng::from_checkpoint(shuffle.0, shuffle.1)?,
            updates,
            execution: crate::TrainingExecutionOptions::default(),
        })
    }

    /// Trains one update; every sample's behaviour weights must be at most
    /// [`PPO_MAX_STALENESS`] updates older than the learner.
    pub fn train_update(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
        objective: UpdateObjective,
    ) -> Result<PpoUpdateReport, PpoError> {
        let current = model
            .policy_identity()
            .map_err(|error| PpoError::Model(error.to_string()))?;
        if self.adam.policy_identity() != current
            || batch.samples.iter().any(|sample| {
                let behaviour = sample.transition.behaviour;
                behaviour > self.updates || self.updates - behaviour > PPO_MAX_STALENESS
            })
        {
            return Err(PpoError::PolicyMismatch);
        }
        if !objective.imitation.is_finite() || objective.imitation < 0.0 {
            return Err(PpoError::InvalidConfig("imitation coefficient"));
        }
        if !(0.0..=1.0).contains(&objective.imitation_balance) {
            return Err(PpoError::InvalidConfig("imitation balance"));
        }
        self.train_accepted_update(model, batch, objective)
    }

    fn train_accepted_update(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
        objective: UpdateObjective,
    ) -> Result<PpoUpdateReport, PpoError> {
        self.validate_batch_dimensions(batch)?;
        let snapshot = model
            .coherent_snapshot(&self.adam)
            .map_err(|error| PpoError::Model(error.to_string()))?;
        let shuffle = self.shuffle.clone();
        let optimizer_step = self.adam.step();
        match self.train_update_inner(model, batch, objective) {
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

    fn validate_batch_dimensions(&self, batch: &PpoBatch) -> Result<(), PpoError> {
        if batch.len() > self.config.rollout_capacity(PPO_MAX_SLOTS) {
            return Err(PpoError::InvalidConfig("batch dimensions"));
        }
        Ok(())
    }

    fn train_update_inner(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
        objective: UpdateObjective,
    ) -> Result<PpoUpdateReport, PpoError> {
        let mut aggregate = PpoUpdateReport {
            objective,
            ..PpoUpdateReport::default()
        };
        let staged = batch.stage(
            model,
            imitates(objective).then_some(objective.imitation_balance),
        )?;
        let mut order = (0..batch.samples.len()).collect::<Vec<_>>();
        'epochs: for epoch in 0..self.config.epochs {
            self.shuffle.shuffle(&mut order)?;
            for indices in minibatch::partition(
                &order,
                self.config.minibatch,
                self.execution.balanced_minibatches,
            ) {
                let report = model
                    .ppo_update_staged(
                        &staged,
                        indices,
                        &mut self.adam,
                        (self.config, objective),
                        (self.execution.training_microbatch, self.execution.kl_guard),
                    )
                    .map_err(|error| PpoError::Model(error.to_string()))?;
                if !report.applied {
                    aggregate.stopped_for_kl = true;
                    record_kl_rejection(&mut aggregate, report)?;
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

/// Whether an update with `objective` trains the imitation term.
pub(crate) fn imitates(objective: UpdateObjective) -> bool {
    objective.imitation > 0.0 && !objective.critic_only
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
    aggregate.imitation.add(&report.imitation);
    aggregate.sides.add(&report.sides);
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

fn record_kl_rejection(
    aggregate: &mut PpoUpdateReport,
    report: PpoMinibatchReport,
) -> Result<(), PpoError> {
    if report.samples == 0 || report.samples > MODEL_MAX_BATCH {
        return Err(PpoError::InvalidTransition("KL rejection samples"));
    }
    aggregate.rejected_kl = report.approximate_kl;
    aggregate.samples_rejected = report.samples;
    Ok(())
}

fn finish_update_report(report: &mut PpoUpdateReport, optimizer_step: u64) -> Result<(), PpoError> {
    if report.samples_optimized == 0 || report.minibatches == 0 {
        if report.stopped_for_kl && report.samples_rejected > 0 {
            report.optimizer_step = optimizer_step;
            return Ok(());
        }
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
    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }

    /// Both explained variances; NaN where the returns are constant or absent.
    pub fn explained_variance(&self) -> ExplainedVariance {
        let value = |sample: &CompactPreparedSample| f64::from(sample.transition.old_value);
        ExplainedVariance {
            lambda: explained(
                self.samples
                    .iter()
                    .map(|sample| (f64::from(sample.return_value), value(sample))),
            ),
            monte_carlo: explained(self.samples.iter().zip(&self.outcome_returns).filter_map(
                |(sample, outcome)| outcome.map(|outcome| (f64::from(outcome), value(sample))),
            )),
        }
    }

    /// Radiant, then dire rollout statistics.
    pub const fn side_statistics(&self) -> [BatchSideStatistics; 2] {
        self.sides
    }

    pub fn sample(&self, index: usize) -> Result<PpoPreparedSample, PpoError> {
        let sample = self
            .samples
            .get(index)
            .ok_or(PpoError::InvalidTransition("PPO sample index"))?;
        Ok(PpoPreparedSample {
            transition: expand_transition(&self.frames, &sample.transition)?,
            advantage: sample.advantage,
            return_value: sample.return_value,
        })
    }

    /// Uploads every sample once, in batch order, to the learner device, with
    /// the shadow labels when the update imitates.
    /// `imitation` is the class balance power of an update that imitates.
    fn stage(
        &self,
        model: &PolicyModel,
        imitation: Option<f32>,
    ) -> Result<StagedPpoBatch, PpoError> {
        let error = |error: crate::ModelError| PpoError::Model(error.to_string());
        let mut staging = model
            .ppo_staging(self.samples.len(), imitation)
            .map_err(error)?;
        for index in 0..self.samples.len() {
            staging.push(&self.sample(index)?).map_err(error)?;
        }
        model.stage_ppo_batch(staging).map_err(error)
    }
}

/// `1 - Var(target - prediction) / Var(target)` over `(target, prediction)` pairs.
fn explained(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> f64 {
    let count = pairs.clone().count() as f64;
    let mean = |values: &dyn Fn((f64, f64)) -> f64| pairs.clone().map(values).sum::<f64>() / count;
    let variance = |values: &dyn Fn((f64, f64)) -> f64| {
        let center = mean(values);
        mean(&|pair| (values(pair) - center).powi(2))
    };
    let targets = variance(&|(target, _)| target);
    if count == 0.0 || targets == 0.0 {
        return f64::NAN;
    }
    1.0 - variance(&|(target, prediction)| target - prediction) / targets
}

fn rollout_storage<T>(capacity: usize) -> Result<Vec<T>, PpoError> {
    assert!(std::mem::size_of::<T>() > 0);
    assert!(capacity <= PPO_MAX_SAMPLES);
    let mut storage = Vec::new();
    storage
        .try_reserve_exact(capacity)
        .map_err(|_| PpoError::InvalidTransition("rollout allocation failed"))?;
    Ok(storage)
}

fn prepare_batch(
    transitions: Vec<CompactPpoTransition>,
    frames: RaggedFeatureArena,
    config: PpoConfig,
    side_networks: crate::SideNetworks,
) -> Result<PpoBatch, PpoError> {
    let mut next_advantage = [0.0f32; PPO_MAX_STREAMS];
    let mut next_return = [None; PPO_MAX_STREAMS];
    let mut next_outcome: [Option<f64>; PPO_MAX_STREAMS] = [None; PPO_MAX_STREAMS];
    let mut prepared = rollout_storage(transitions.len())?;
    let mut outcome_returns = Vec::with_capacity(transitions.len());
    for transition in transitions.into_iter().rev() {
        let later = if transition.terminal {
            Some(0.0)
        } else {
            next_outcome[transition.stream]
        };
        let outcome = later.map(|later| {
            f64::from(transition.reward)
                + f64::from(config.gamma_tick).powi(transition.ticks as i32) * later
        });
        next_outcome[transition.stream] = outcome;
        outcome_returns.push(outcome.map(|outcome| outcome as f32));
        let discount = tick_discount(config.gamma_tick, transition.ticks)?;
        let trace = tick_discount(config.gae_lambda_tick, transition.ticks)?;
        let continuation = if transition.terminal { 0.0 } else { 1.0 };
        let delta = transition.reward + discount * transition.next_value * continuation
            - transition.old_value;
        let mut advantage =
            delta + discount * trace * next_advantage[transition.stream] * continuation;
        let mut return_value = transition.old_value + advantage;
        if config.gae_lambda_tick == 1.0 {
            let value = monte_carlo_return(&transition, config.gamma_tick, &mut next_return)?;
            return_value = value as f32;
            advantage = (value - f64::from(transition.old_value)) as f32;
        }
        if !advantage.is_finite() {
            return Err(PpoError::NonFinite("advantage"));
        }
        next_advantage[transition.stream] = advantage;
        prepared.push(CompactPreparedSample {
            return_value,
            transition,
            advantage,
        });
    }
    prepared.reverse();
    outcome_returns.reverse();
    let mut sides = [true, false].map(|radiant| side_statistics(&prepared, radiant));
    match side_networks {
        crate::SideNetworks::Shared => {
            normalize_advantages(&mut prepared, |_| true)?;
            weigh_advantages(&mut prepared, |_| true);
        }
        crate::SideNetworks::Separate => {
            for radiant in [true, false] {
                let side = |sample: &CompactPreparedSample| sample.transition.radiant == radiant;
                normalize_advantages(&mut prepared, side)?;
                weigh_advantages(&mut prepared, side);
            }
        }
    }
    for (side, radiant) in sides.iter_mut().zip([true, false]) {
        let normalized = side_statistics(&prepared, radiant);
        side.normalized_advantage_mean = normalized.advantage_mean;
        side.continue_advantage_mean = normalized.continue_advantage_mean;
        side.action_advantage_mean = normalized.action_advantage_mean;
    }
    Ok(PpoBatch {
        samples: prepared,
        frames,
        outcome_returns,
        sides,
    })
}

fn side_statistics(samples: &[CompactPreparedSample], radiant: bool) -> BatchSideStatistics {
    let side = || {
        samples
            .iter()
            .filter(move |sample| sample.transition.radiant == radiant)
    };
    let count = side().count();
    if count == 0 {
        return BatchSideStatistics::default();
    }
    let mean = |value: &dyn Fn(&CompactPreparedSample) -> f64| {
        side().map(value).sum::<f64>() / count as f64
    };
    let advantage_mean = mean(&|sample| f64::from(sample.advantage));
    BatchSideStatistics {
        samples: count,
        advantage_mean,
        advantage_deviation: mean(&|sample| (f64::from(sample.advantage) - advantage_mean).powi(2))
            .sqrt(),
        normalized_advantage_mean: advantage_mean,
        return_mean: mean(&|sample| f64::from(sample.return_value)),
        value_mean: mean(&|sample| f64::from(sample.transition.old_value)),
        explained_variance: explained(side().map(|sample| {
            (
                f64::from(sample.return_value),
                f64::from(sample.transition.old_value),
            )
        })),
        behaviour_entropy: mean(&|sample| -f64::from(sample.transition.old_log_probability)),
        continue_share: mean(&|sample| f64::from(u8::from(continues(sample)))),
        continue_advantage_mean: conditional_mean(side().filter(|sample| continues(sample))),
        action_advantage_mean: conditional_mean(side().filter(|sample| !continues(sample))),
    }
}

fn continues(sample: &CompactPreparedSample) -> bool {
    sample.transition.action.kind() == crate::ActionKind::Continue
}

fn conditional_mean<'a>(samples: impl Iterator<Item = &'a CompactPreparedSample>) -> f64 {
    let (count, sum) = samples.fold((0usize, 0.0f64), |(count, sum), sample| {
        (count + 1, sum + f64::from(sample.advantage))
    });
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn monte_carlo_return(
    transition: &CompactPpoTransition,
    gamma: f32,
    next: &mut [Option<f64>; PPO_MAX_STREAMS],
) -> Result<f64, PpoError> {
    assert!(transition.ticks > 0);
    assert!(transition.stream < PPO_MAX_STREAMS);
    let bootstrap = if transition.terminal {
        0.0
    } else {
        next[transition.stream].unwrap_or(f64::from(transition.next_value))
    };
    let ticks = i32::try_from(transition.ticks).map_err(|_| PpoError::InvalidDiscount)?;
    let value = f64::from(transition.reward) + f64::from(gamma).powi(ticks) * bootstrap;
    if !value.is_finite() {
        return Err(PpoError::NonFinite("Monte Carlo return"));
    }
    next[transition.stream] = Some(value);
    Ok(value)
}

/// Normalizes the advantages of the `selected` samples over those samples.
fn normalize_advantages(
    samples: &mut [CompactPreparedSample],
    selected: impl Fn(&CompactPreparedSample) -> bool + Copy,
) -> Result<(), PpoError> {
    let count = samples.iter().filter(|sample| selected(sample)).count();
    if count == 0 {
        return Ok(());
    }
    let count = count as f64;
    let mean = samples
        .iter()
        .filter(|sample| selected(sample))
        .map(|sample| f64::from(sample.advantage))
        .sum::<f64>()
        / count;
    let variance = samples
        .iter()
        .filter(|sample| selected(sample))
        .map(|sample| {
            let delta = f64::from(sample.advantage) - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    let deviation = variance.sqrt();
    let divisor = deviation.max(1.0e-8);
    for sample in samples.iter_mut().filter(|sample| selected(sample)) {
        sample.advantage = ((f64::from(sample.advantage) - mean) / divisor) as f32;
        if !sample.advantage.is_finite() || !sample.return_value.is_finite() {
            return Err(PpoError::NonFinite("normalized advantage or return"));
        }
    }
    Ok(())
}

/// Scales each `selected` normalized advantage by its decision's inverse
/// retention probability over the mean of those samples; a positive factor on
/// the advantage weighs the clipped surrogate exactly as it would weigh the sample.
fn weigh_advantages(
    samples: &mut [CompactPreparedSample],
    selected: impl Fn(&CompactPreparedSample) -> bool + Copy,
) {
    let count = samples.iter().filter(|sample| selected(sample)).count();
    if count == 0 {
        return;
    }
    let mean = samples
        .iter()
        .filter(|sample| selected(sample))
        .map(|sample| f64::from(sample.transition.weight))
        .sum::<f64>()
        / count as f64;
    for sample in samples.iter_mut().filter(|sample| selected(sample)) {
        sample.advantage =
            (f64::from(sample.advantage) * f64::from(sample.transition.weight) / mean) as f32;
    }
}

fn expand_transition(
    frames: &RaggedFeatureArena,
    compact: &CompactPpoTransition,
) -> Result<PpoTransition, PpoError> {
    Ok(PpoTransition {
        frame: frames
            .expand(&compact.frame)
            .map_err(PpoError::InvalidTransition)?,
        target: compact.target.unpack(),
        shadow: compact.shadow.as_ref().map(PackedActionHeadTargets::unpack),
        action: compact.action,
        behaviour: compact.behaviour,
        stream: compact.stream,
        decision: compact.decision,
        ticks: compact.ticks,
        old_log_probability: compact.old_log_probability,
        old_value: compact.old_value,
        next_value: compact.next_value,
        reward: compact.reward,
        terminal: compact.terminal,
        weight: compact.weight,
    })
}

/// Discount over an exact positive number of elapsed simulation ticks.
pub fn tick_discount(gamma_tick: f32, ticks: u32) -> Result<f32, PpoError> {
    if !gamma_tick.is_finite() || !(0.0..=1.0).contains(&gamma_tick) || ticks == 0 {
        return Err(PpoError::InvalidDiscount);
    }
    let exponent = i32::try_from(ticks).map_err(|_| PpoError::InvalidDiscount)?;
    let discount = gamma_tick.powi(exponent);
    discount
        .is_finite()
        .then_some(discount)
        .ok_or(PpoError::InvalidDiscount)
}

/// Terminal result of a game; a draw is terminal too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpoTerminalOutcome {
    Win,
    Loss,
    Draw,
}
