#![allow(
    clippy::float_arithmetic,
    reason = "policy tensors use f32 outside the deterministic simulation"
)]

use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use candle_core::{DType, Device, Tensor, Var};

mod device_learner;
mod ppo_objective;
mod rows;
mod sampling;
mod side_actors;
pub use device_learner::MODEL_PPO_MAX_MICROBATCH;
pub(crate) use device_learner::StagedPpoBatch;
pub use rows::{ENCODER_ROW_ELEMENTS, EncoderRow};
#[cfg(test)]
pub(crate) use side_actors::take_encoder_forwards_for_test;
use side_actors::{ActorHead, ActorRouting, QueuedPair, StageRequest, StageValues};
#[cfg(test)]
#[path = "tests/model_side_actors.rs"]
mod side_actor_tests;
#[cfg(test)]
pub(crate) use sampling::{take_sampling_dispatches_for_test, with_eager_sampling_for_test};

#[cfg(test)]
#[path = "tests/model_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;

use crate::{
    ABILITY_FEATURE_TOKENS, ABILITY_FEATURES, ActionKind, ActionSpace, ActionTarget,
    BehavioralTarget, ControlledUnit, EntityIndex, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION,
    FeatureFrame, GLOBAL_FEATURES, HISTORY_FEATURES, HISTORY_SAMPLES, HeadTarget,
    ITEM_FEATURE_TOKENS, ITEM_FEATURES, LOOT_FEATURE_TOKENS, LOOT_FEATURES, LootIndex,
    MAP_FEATURES, MAX_POLICY_HISTORY, OWN_UNIT_FEATURE_TOKENS, POINT_FEATURE_TOKENS,
    POINT_FEATURES, POLICY_HISTORY_FEATURES, PROJECTILE_FEATURE_TOKENS, PROJECTILE_FEATURES,
    PointIndex, PpoConfig, PpoMinibatchReport, PpoPolicyChoice, PpoPreparedSample, PpoRng,
    PutPointTarget, REMEMBERED_UNIT_FEATURE_TOKENS, ShopIndex, StructuredAction,
    UNIT_FEATURE_TOKENS, UNIT_FEATURES, ability_feature, item_feature, loot_feature, point_feature,
    projectile_feature, unit_feature,
};

/// Version of the fixed policy-model layout and linked candidate execution contract.
pub const MODEL_SCHEMA_VERSION: u32 = 26;
/// Maximum frame count accepted by one public batch call.
pub const MODEL_MAX_BATCH: usize = 8_192;
/// Frame count evaluated by one bounded host inference tensor graph.
pub const MODEL_EVALUATION_MICROBATCH: usize = 64;
/// Maximum frame count in one autograd-preserving tensor forward pass.
pub const MODEL_TRAINING_BATCH: usize = 64;
/// Maximum rows in one sampling or greedy selection call.
pub const MODEL_SAMPLING_BATCH: usize = 128;
const _: () = assert!(MODEL_SAMPLING_BATCH <= MODEL_PPO_MAX_MICROBATCH);
const _: () = assert!(MODEL_TRAINING_BATCH <= MODEL_PPO_MAX_MICROBATCH);
const _: () = assert!(MODEL_PPO_MAX_MICROBATCH <= MODEL_MAX_BATCH);
/// Number of append-only action-kind logits.
pub const MODEL_KIND_HEAD: usize = 16;
/// Number of controlled-unit logits.
pub const MODEL_UNIT_HEAD: usize = 2;
/// Maximum number of ability-slot logits.
pub const MODEL_ABILITY_HEAD: usize = 8;
/// Maximum number of item or source-slot logits.
pub const MODEL_ITEM_HEAD: usize = 15;
/// Number of swap-destination logits.
pub const MODEL_SWAP_HEAD: usize = 15;
/// Maximum number of learn-slot logits.
pub const MODEL_LEARN_HEAD: usize = 6;
/// Maximum number of shop logits.
pub const MODEL_SHOP_HEAD: usize = 64;
/// Maximum number of loot logits.
pub const MODEL_LOOT_HEAD: usize = 16;
/// Maximum number of current-entity pointer logits.
pub const MODEL_ENTITY_POINTER_HEAD: usize = 96;
/// Maximum number of point-candidate pointer logits.
pub const MODEL_POINT_POINTER_HEAD: usize = 64;
/// Maximum checked Adam optimizer step.
pub const MODEL_MAX_OPTIMIZER_STEP: u64 = 1_000_000_000;
/// Number of behavioral heads represented in update activity counts.
pub const MODEL_BEHAVIORAL_HEADS: usize = 12;

const UNIT_HIDDEN: usize = 64;
const UNIT_EMBEDDING: usize = 128;
const TOKEN_HIDDEN: usize = 64;
const TOKEN_EMBEDDING: usize = 64;
const UNIT_GROUPS: usize = 5;
const TRUNK_INPUT: usize = GLOBAL_FEATURES
    + HISTORY_SAMPLES * HISTORY_FEATURES
    + MAX_POLICY_HISTORY * POLICY_HISTORY_FEATURES
    + MAP_FEATURES
    + OWN_UNIT_FEATURE_TOKENS * UNIT_EMBEDDING
    + UNIT_GROUPS * UNIT_EMBEDDING * 2
    + 5 * TOKEN_EMBEDDING * 2;
const TRUNK_WIDE: usize = 512;
const TRUNK_WIDTH: usize = 256;
const VALUE_HIDDEN: usize = 256;
const VALUE_OUTPUT_SCALE: f64 = 16.0;
const KIND_EMBEDDING: usize = 32;
const UNIT_SELECTION_EMBEDDING: usize = 32;
const SLOT_EMBEDDING: usize = 16;
const DECODER_CONTEXT: usize = 336;
const TARGET_MODE_HEAD: usize = 3;
const PUT_MODE_HEAD: usize = 2;
pub(crate) const MODEL_PARAMETER_TENSORS: usize = 88;
static NEXT_MODEL_LINEAGE: AtomicU64 = AtomicU64::new(1);
static NEXT_OPTIMIZER_LINEAGE: AtomicU64 = AtomicU64::new(1);

/// Canonical model shapes, parameter order, side routing and linked action/feature semantics.
pub const MODEL_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-model/v26;",
    "linked_schemas=action,feature,map2_reward;linked_hash=fnv1a_descriptor_then_ordered_version_le32_hash_le64_then_map2_reward_descriptor_utf8;",
    "scope=map2_mid_only_cap27900_including900_pregame;candidate_execution=feature19_candidate_order_bookkeeping_action7_walkable_building_landing_move_only_mango_unchanged_point_pointer64_raze_only_points;layout=88_named_tensors_1878775_f32;",
    "map2_inputs=global92_unit84,wire_rebase_unit_bound_and_collision_and_attack_time;",
    "dtype=f32;device=cpu_actor,cpu_or_cuda_learner,one_learner_per_device;architecture=deepsets;activations=relu_after_every_encoder_and_trunk_linear;",
    "input_conditioning=host_before_tensor_after_presence_mask,feature_v9_unchanged;category_divisors=global10:5,12:3,32:16,55:12;policy_history3:16;unit5:12;ability1:2,2:8,11:5;item1:5,2:64,9:5,13:3;point10:8_sources9..13_exceed_one,12:8,16:12;semantic_ids=ability5_and_projectile6:ln1p(x)/ln(65548),item4_and_loot1:ln1p(x)/ln(65537);all_other_features_identity;",
    "output_initialization=all_linear_outside_relu_mlps_and_pointer_queries:he_uniform_times0.01,value_readout_times0.01_over16,value_hidden_he_uniform,bias_zero,no_extra_rng_draws;pointer_scaling=dot_div_sqrt_embedding_width_all_actor_batch_and_training_paths;",
    "numeric_semantics=semantic_id_signed_ln1p_abs_extension_preserves_zero,bc_and_ppo_masked_cross_entropy_center_legal_logits_by_detached_row_max_before_logsumexp_and_selected_subtraction;",
    "unit_mlp=84x64,64x128,128x128;",
    "ability_mlp=24x64,64x64;item_mlp=28x64,64x64;",
    "point_mlp=32x64,64x64;projectile_mlp=20x64,64x64;loot_mlp=16x64,64x64;",
    "unit_groups=hero,creep,structure,neutral,courier_ward;",
    "pool=token_present_and_semantic_group_mask,mean=sum_over_selected/divide_by_positive_count,max=where_selected_embedding_else_negative_infinity_then_argmax_lowest_token_tie_per_channel_then_differentiable_gather_original_embedding,one_token_receives_max_gradient,empty_mean_and_max_exact_zero,cross_group_rows_never_enter_reduction;",
    "token_pools=ability,item,point,projectile,loot;own_units=hero,courier;",
    "trunk=2596x512,512x256,256x256;",
    "embeddings=kind:16x32,unit:2x32,ability:8x16,item:15x16;",
    "heads=value:256x256_relu_256x1_times16,kind:16,unit:2,ability:8,item_source_from:15,swap_to:15,learn:6,shop:64,loot:16,target_mode:3,put_mode:2,entity_query:128,point_query:64;",
    "action_kind=0Continue,1Stop,2MovePoint,3FollowUnit,4Hold,5AttackMovePoint,6AttackUnit,7Cast,8Use,9PutPoint,10PutUnit,11Take,12Buy,13Sell,14Swap,15Learn;",
    "decoder=kind_then_optional_controlled_unit_then_family_slot_or_source_then_optional_target;branches=Continue:none,Stop:unit,MovePoint:unit_point,FollowUnit:unit_entity,Hold:unit,AttackMovePoint:unit_point,AttackUnit:unit_entity,Cast:unit_ability_target_mode_target,Use:unit_item_target_mode_target,PutPoint:unit_source_put_mode_optional_point,PutUnit:unit_source_entity,Take:unit_loot,Buy:unit_shop,Sell:unit_item,Swap:unit_from_to,Learn:ability;",
    "pointer=entity_query_dot_current_unit_embedding_in_frame_unit_order,point_query_dot_point_embedding_in_frame_point_order;target_mode=masked_argmax_None_Entity_Point_before_selected_pointer_argmax;put_mode=masked_argmax_Underfoot_Point_before_point_argmax;pointer_values_never_offset_mode_logits;",
    "head_context=controlled_and_learn_kind_prefix,ability_item_shop_loot_kind_unit_prefix,swap_target_put_and_pointer_kind_unit_slot_prefix;",
    "selection=choose_requires_private_exact_frame_action_space_lineage_revision_tick_readiness_provenance_before_tensor_work,provenance_excluded_from_tensor_and_frame_equality;mask_before_argmax,all_logits_finite_required,highest_legal_logit,lowest_stable_index_tie,no_legal_exact_error,final_action_allows_and_decode_required;",
    "nonfinite=finite_parameters_required_on_import,finite_frame_required,all_public_host_outputs_and_every_traversed_decoder_head_checked_with_batch_and_index,error_on_overflow_no_policy_choice,training_output_exposes_optional_graph_preserving_finite_validation;",
    "initialization=splitmix64_state_plus_9e3779b97f4a7c15_then_mix_bf58476d1ce4e5b9_94d049bb133111eb_top24_to_symmetric_closed_interval;linear_weight_scale=sqrt(6/fan_in),linear_bias_zero,embedding_scale=sqrt(3/columns);draw_order=unit_ability_item_point_projectile_loot_trunk_value_hidden_value_readout_kind_kind_embedding_unit_embedding_ability_embedding_item_embedding_controlled_ability_head_item_head_swap_head_learn_head_shop_head_loot_head_target_mode_put_mode_entity_query_point_query;seed_is_not_input_or_parameter;",
    "batch=public_host_limit8192,evaluation_microbatch64_under_one_parameter_read_lock,training_tensor_limit64,larger_effective_training_batches_require_gradient_accumulation;",
    "runtime_identity=checked_process_local_nonzero_model_lineage_plus_monotonic_parameter_revision,one_internal_learner_optimizer_lineage_bound_to_exact_policy_identity,raw_import_advances_revision_and_unbinds_optimizer,evidence_never_enters_tensors;",
    "updates=single_model_rwlock,all_inference_and_export_reads_hold_one_shared_lock,training_output_owns_shared_lock_for_full_forward_loss_backward_lifetime,named_backward_requires_same_model_guarded_output_and_returns88_stable_named_optional_gradient_tensors,no_unlocked_vars_exposed,parameter_import_deep_copies_originals_and_builds_and_replaces_all88_vars_under_one_exclusive_lock_with_exact_rollback_on_failure,readers_observe_complete_old_or_complete_new_parameter_set;",
    "parameter_order=unit_mlp,ability_mlp,item_mlp,point_mlp,projectile_mlp,loot_mlp,trunk,value.0,value.1,kind,kind_embedding,unit_embedding,ability_embedding,item_embedding,unit_head,ability_head,item_head,swap_head,learn_head,shop_head,loot_head,target_mode_head,put_mode_head,entity_query,point_query;",
    "reward7=win.2_loss_neg.2_draw0_completed_taskcap_neg.2_win_only_victory_time_bonus;",
    "actor=radiant12_linear_heads_then_dire12_independent_linear_heads_in_the_same_order,shared_encoder_trunk_value_and_embeddings;ppo_value_loss_trains_shared_trunk;",
    "dire_order=kind,controlled,ability_head,item_head,swap_head,learn_head,shop_head,loot_head,target_mode,put_mode,entity_query,point_query;",
    "routing=observed_global4_Radiant_global5_Dire_exact_numeric_onehot_only,validate_before_tensors,negative_zero_is_zero;",
    "geometry=existing_team_canonical_position_and_delta_unchanged;no_seat_stream_seed_outcome_or_modifier_routing;",
    "math=both_actor_linears_full_original_batch_then_u8_where,select_queries_before_pointer_dot,no_row_compaction;",
    "finite=both_raw_branches_all_rows_of_exercised_heads,all_training_heads_and_selected_pointer_scores_before_backward,unused_inference_families_skipped;",
    "optimizer=one_shared_parameter_lock_global_Adam_norm_and_transaction_over88_tensors;",
    "initialization_suffix=dire_heads_drawn_after_the_radiant_parameter_order;"
);

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub(crate) const fn fnv1a_extend(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        index += 1;
    }
    hash
}

/// FNV-1a over the little-endian bits of exported parameters.
pub(crate) fn parameter_fingerprint_of(parameters: &[f32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for value in parameters {
        hash = fnv1a_extend(hash, &value.to_bits().to_le_bytes());
    }
    hash
}

/// Folds ordered imported identities and the Map2 reward version.
pub(crate) const fn linked_schema_hash(descriptor: &str, schemas: &[(u32, u64)]) -> u64 {
    let mut hash = fnv1a_extend(FNV_OFFSET, descriptor.as_bytes());
    let mut index = 0;
    while index < schemas.len() {
        hash = fnv1a_extend(hash, &schemas[index].0.to_le_bytes());
        hash = fnv1a_extend(hash, &schemas[index].1.to_le_bytes());
        index += 1;
    }
    fnv1a_extend(hash, &crate::MAP2_REWARD_VERSION.to_le_bytes())
}

const fn linear_parameters(input: usize, output: usize) -> usize {
    input * output + output
}

/// FNV-1a of the descriptor, ordered linked versions/hashes, and reward version.
pub const MODEL_SCHEMA_HASH: u64 = linked_schema_hash(
    MODEL_SCHEMA_DESCRIPTOR,
    &[
        (crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
    ],
);

/// Exact number of F32 parameters in the model layout.
pub const MODEL_PARAMETER_COUNT: usize = 1_878_775;

const _: () = assert!(FEATURE_SCHEMA_VERSION == 25);
const _: () = assert!(crate::ACTION_SCHEMA_VERSION == 8);
const _: () = assert!(GLOBAL_FEATURES == 92);
const _: () = assert!(UNIT_FEATURES == 84);
const _: () = assert!(TRUNK_INPUT == 2_596);
const _: () = assert!(
    DECODER_CONTEXT == TRUNK_WIDTH + KIND_EMBEDDING + UNIT_SELECTION_EMBEDDING + SLOT_EMBEDDING
);
const _: () = assert!(MODEL_PARAMETER_COUNT == parameter_count_from_shapes());

const fn parameter_count_from_shapes() -> usize {
    let unit = linear_parameters(UNIT_FEATURES, UNIT_HIDDEN)
        + linear_parameters(UNIT_HIDDEN, UNIT_EMBEDDING)
        + linear_parameters(UNIT_EMBEDDING, UNIT_EMBEDDING);
    let tokens = linear_parameters(ABILITY_FEATURES, TOKEN_HIDDEN)
        + linear_parameters(TOKEN_HIDDEN, TOKEN_EMBEDDING)
        + linear_parameters(ITEM_FEATURES, TOKEN_HIDDEN)
        + linear_parameters(TOKEN_HIDDEN, TOKEN_EMBEDDING)
        + linear_parameters(POINT_FEATURES, TOKEN_HIDDEN)
        + linear_parameters(TOKEN_HIDDEN, TOKEN_EMBEDDING)
        + linear_parameters(PROJECTILE_FEATURES, TOKEN_HIDDEN)
        + linear_parameters(TOKEN_HIDDEN, TOKEN_EMBEDDING)
        + linear_parameters(LOOT_FEATURES, TOKEN_HIDDEN)
        + linear_parameters(TOKEN_HIDDEN, TOKEN_EMBEDDING);
    unit + tokens + trunk_parameter_count() + decoder_parameter_count()
}

const fn trunk_parameter_count() -> usize {
    linear_parameters(TRUNK_INPUT, TRUNK_WIDE)
        + linear_parameters(TRUNK_WIDE, TRUNK_WIDTH)
        + linear_parameters(TRUNK_WIDTH, TRUNK_WIDTH)
}

const fn decoder_parameter_count() -> usize {
    let embeddings = MODEL_KIND_HEAD * KIND_EMBEDDING
        + MODEL_UNIT_HEAD * UNIT_SELECTION_EMBEDDING
        + MODEL_ABILITY_HEAD * SLOT_EMBEDDING
        + MODEL_ITEM_HEAD * SLOT_EMBEDDING;
    let value = linear_parameters(TRUNK_WIDTH, VALUE_HIDDEN) + linear_parameters(VALUE_HIDDEN, 1);
    let direct = value + linear_parameters(TRUNK_WIDTH, MODEL_KIND_HEAD);
    let conditional = linear_parameters(DECODER_CONTEXT, MODEL_UNIT_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_ABILITY_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_ITEM_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_SWAP_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_LEARN_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_SHOP_HEAD)
        + linear_parameters(DECODER_CONTEXT, MODEL_LOOT_HEAD)
        + linear_parameters(DECODER_CONTEXT, TARGET_MODE_HEAD)
        + linear_parameters(DECODER_CONTEXT, PUT_MODE_HEAD)
        + linear_parameters(DECODER_CONTEXT, UNIT_EMBEDDING)
        + linear_parameters(DECODER_CONTEXT, TOKEN_EMBEDDING);
    // Radiant and Dire own independent kind and conditional actor heads.
    embeddings
        + direct
        + conditional
        + linear_parameters(TRUNK_WIDTH, MODEL_KIND_HEAD)
        + conditional
}

fn allocate_lineage(counter: &AtomicU64, exhausted: ModelError) -> Result<NonZeroU64, ModelError> {
    let value = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| exhausted)?;
    NonZeroU64::new(value).ok_or(ModelError::InvalidModelState("zero lineage"))
}

/// Model construction, evaluation, selection, or parameter-validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    InvalidSideOneHot {
        index: usize,
        radiant_bits: u32,
        dire_bits: u32,
    },
    EmptyBatch,
    BatchTooLarge {
        count: usize,
        maximum: usize,
    },
    EmptyTrainingBatch,
    TrainingBatchTooLarge {
        count: usize,
        maximum: usize,
    },
    TrainingPrefixCount {
        prefixes: usize,
        frames: usize,
    },
    BatchActionSpaceCount {
        action_spaces: usize,
        frames: usize,
    },
    SamplingRngCount {
        rngs: usize,
        frames: usize,
    },
    BatchFrameActionSpaceMismatch {
        index: usize,
    },
    TrainingSlotIndex {
        family: &'static str,
        index: usize,
        maximum: usize,
    },
    NonFiniteFrame {
        index: usize,
    },
    ParameterLength {
        actual: usize,
        expected: usize,
    },
    NonFiniteParameter {
        index: usize,
    },
    BehavioralTarget {
        head: &'static str,
        label: usize,
    },
    BehavioralExampleCount {
        count: usize,
        maximum: usize,
    },
    NonTrainingExample {
        index: usize,
    },
    InvalidAdamConfig(&'static str),
    OptimizerVectorLength {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    OptimizerStepOverflow,
    NonFiniteLoss,
    NonFiniteGradient {
        index: usize,
    },
    NonFiniteMoment {
        field: &'static str,
        index: usize,
    },
    NonFiniteOptimizerNorm,
    NonFiniteOptimizerUpdate {
        index: usize,
    },
    EmptyMask,
    SelectionShape {
        logits: usize,
        mask: usize,
    },
    SelectionNonFinite {
        index: usize,
    },
    NoLegalContinuation,
    NonFiniteOutput {
        field: &'static str,
        batch: usize,
        index: usize,
    },
    FrameActionSpaceMismatch,
    TrainingOutputModelMismatch,
    OptimizerAlreadyOwned,
    OptimizerOwnershipMismatch,
    ModelLineageUnavailable,
    OptimizerLineageUnavailable,
    ParameterRevisionOverflow,
    InjectedParameterFailure {
        index: usize,
    },
    ParameterLockPoisoned,
    Backend(String),
    InvalidModelState(&'static str),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch
            | Self::BatchTooLarge { .. }
            | Self::EmptyTrainingBatch
            | Self::TrainingBatchTooLarge { .. }
            | Self::TrainingPrefixCount { .. }
            | Self::BatchActionSpaceCount { .. }
            | Self::SamplingRngCount { .. }
            | Self::BatchFrameActionSpaceMismatch { .. }
            | Self::TrainingSlotIndex { .. }
            | Self::NonFiniteFrame { .. }
            | Self::ParameterLength { .. }
            | Self::NonFiniteParameter { .. }
            | Self::BehavioralTarget { .. }
            | Self::BehavioralExampleCount { .. }
            | Self::NonTrainingExample { .. } => self.fmt_input(formatter),
            Self::InvalidAdamConfig(_)
            | Self::OptimizerVectorLength { .. }
            | Self::OptimizerStepOverflow
            | Self::NonFiniteLoss
            | Self::NonFiniteGradient { .. }
            | Self::NonFiniteMoment { .. }
            | Self::NonFiniteOptimizerNorm
            | Self::NonFiniteOptimizerUpdate { .. } => self.fmt_optimizer(formatter),
            _ => self.fmt_runtime(formatter),
        }
    }
}

impl ModelError {
    fn fmt_input(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("model batch must contain at least one frame"),
            Self::BatchTooLarge { count, maximum } => write!(
                formatter,
                "model batch count {count} exceeds maximum {maximum}"
            ),
            Self::EmptyTrainingBatch => {
                formatter.write_str("model training batch must contain at least one frame")
            }
            Self::TrainingBatchTooLarge { count, maximum } => write!(
                formatter,
                "model training batch count {count} exceeds maximum {maximum}"
            ),
            Self::TrainingPrefixCount { prefixes, frames } => write!(
                formatter,
                "model training prefix count {prefixes} differs from frame count {frames}"
            ),
            Self::BatchActionSpaceCount {
                action_spaces,
                frames,
            } => write!(
                formatter,
                "model batch action-space count {action_spaces} differs from frame count {frames}"
            ),
            Self::SamplingRngCount { rngs, frames } => write!(
                formatter,
                "model sampling RNG count {rngs} differs from frame count {frames}"
            ),
            Self::BatchFrameActionSpaceMismatch { index } => write!(
                formatter,
                "model batch frame {index} does not belong to its action space"
            ),
            Self::TrainingSlotIndex {
                family,
                index,
                maximum,
            } => write!(
                formatter,
                "model training {family} slot index {index} exceeds maximum {maximum}"
            ),
            Self::NonFiniteFrame { index } => {
                write!(formatter, "model frame {index} contains a non-finite value")
            }
            Self::ParameterLength { actual, expected } => write!(
                formatter,
                "model parameter length {actual} differs from expected {expected}"
            ),
            Self::NonFiniteParameter { index } => {
                write!(formatter, "model parameter {index} is non-finite")
            }
            Self::BehavioralTarget { head, label } => write!(
                formatter,
                "model behavioral target label {label} is illegal for head {head}"
            ),
            Self::BehavioralExampleCount { count, maximum } => write!(
                formatter,
                "model behavioral example count {count} is outside 1..={maximum}"
            ),
            Self::NonTrainingExample { index } => {
                write!(
                    formatter,
                    "model behavioral training example {index} is not Train"
                )
            }
            _ => self.fmt_runtime(formatter),
        }
    }

    fn fmt_optimizer(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAdamConfig(field) => write!(formatter, "model Adam {field} is invalid"),
            Self::OptimizerVectorLength {
                field,
                actual,
                expected,
            } => write!(
                formatter,
                "model optimizer {field} length {actual} differs from expected {expected}"
            ),
            Self::OptimizerStepOverflow => {
                formatter.write_str("model Adam step exceeds its maximum")
            }
            Self::NonFiniteLoss => formatter.write_str("model behavioral loss is non-finite"),
            Self::NonFiniteGradient { index } => {
                write!(formatter, "model gradient {index} is non-finite")
            }
            Self::NonFiniteMoment { field, index } => {
                write!(formatter, "model Adam {field} moment {index} is non-finite")
            }
            Self::NonFiniteOptimizerNorm => {
                formatter.write_str("model gradient norm is non-finite")
            }
            Self::NonFiniteOptimizerUpdate { index } => {
                write!(formatter, "model Adam update {index} is non-finite")
            }
            _ => self.fmt_runtime(formatter),
        }
    }

    fn fmt_runtime(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSideOneHot {
                index,
                radiant_bits,
                dire_bits,
            } => write!(
                formatter,
                "model frame {index} has invalid side one-hot: radiant={}, dire={}; expected (1,0) or (0,1)",
                f32::from_bits(*radiant_bits),
                f32::from_bits(*dire_bits)
            ),
            Self::EmptyMask => formatter.write_str("model selection mask is empty"),
            Self::SelectionShape { logits, mask } => write!(
                formatter,
                "model selection logits length {logits} differs from mask length {mask}"
            ),
            Self::SelectionNonFinite { index } => {
                write!(formatter, "model selection logit {index} is non-finite")
            }
            Self::NoLegalContinuation => {
                formatter.write_str("model selection has no legal continuation")
            }
            Self::NonFiniteOutput {
                field,
                batch,
                index,
            } => write!(
                formatter,
                "model {field} output at batch {batch} index {index} is non-finite"
            ),
            Self::FrameActionSpaceMismatch => formatter
                .write_str("model feature frame does not belong to the supplied action space"),
            Self::TrainingOutputModelMismatch => {
                formatter.write_str("model training output belongs to a different policy model")
            }
            Self::OptimizerAlreadyOwned => {
                formatter.write_str("model already has a behavioral optimizer owner")
            }
            Self::OptimizerOwnershipMismatch => {
                formatter.write_str("model optimizer owner or parameter revision does not match")
            }
            Self::ModelLineageUnavailable => {
                formatter.write_str("model lineage allocation is exhausted")
            }
            Self::OptimizerLineageUnavailable => {
                formatter.write_str("model optimizer lineage allocation is exhausted")
            }
            Self::ParameterRevisionOverflow => {
                formatter.write_str("model parameter revision is exhausted")
            }
            Self::InjectedParameterFailure { index } => write!(
                formatter,
                "model injected parameter replacement failure after tensor {index}"
            ),
            Self::ParameterLockPoisoned => formatter.write_str("model parameter lock is poisoned"),
            Self::Backend(message) => write!(formatter, "model tensor operation failed: {message}"),
            Self::InvalidModelState(field) => write!(formatter, "model produced invalid {field}"),
            _ => formatter.write_str("model error category is invalid"),
        }
    }
}

impl Error for ModelError {}

impl From<candle_core::Error> for ModelError {
    fn from(error: candle_core::Error) -> Self {
        Self::Backend(error.to_string())
    }
}

/// Public value and append-only action-kind logits for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyOutput {
    /// Unbounded scalar state-value prediction.
    pub value: f32,
    /// Logits in append-only [`ActionKind`] order.
    pub kind_logits: [f32; MODEL_KIND_HEAD],
}

impl PolicyOutput {
    /// Whether the value and every action-kind logit are finite.
    pub fn is_finite(&self) -> bool {
        self.value.is_finite() && self.kind_logits.iter().all(|value| value.is_finite())
    }
}

/// Greedy legal structured action paired with its state value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyChoice {
    /// Greedy structured action allowed by the supplied action space.
    pub action: StructuredAction,
    /// Finite state-value prediction for the selected frame.
    pub value: f32,
}

/// Process-local model lineage and exact installed-parameter revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyIdentity {
    lineage: NonZeroU64,
    revision: u64,
}

impl PolicyIdentity {
    pub const fn lineage(self) -> NonZeroU64 {
        self.lineage
    }

    pub const fn revision(self) -> u64 {
        self.revision
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OptimizerBinding {
    pub(crate) lineage: NonZeroU64,
    pub(crate) policy: PolicyIdentity,
}

/// Standard Adam hyperparameters with global gradient-norm clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub gradient_clip: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            gradient_clip: 0.5,
        }
    }
}

/// One optimizer owner with exact moments, checked step, and bound policy revision.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamState {
    binding: OptimizerBinding,
    config: AdamConfig,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    step: u64,
}

/// Coherent host snapshot captured under one model parameter guard.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelAdamSnapshot {
    pub parameters: Vec<f32>,
    pub adam: AdamState,
}

impl AdamState {
    pub(crate) fn new(config: AdamConfig, binding: OptimizerBinding) -> Result<Self, ModelError> {
        validate_adam_config(config)?;
        Ok(Self {
            binding,
            config,
            first_moment: vec![0.0; MODEL_PARAMETER_COUNT],
            second_moment: vec![0.0; MODEL_PARAMETER_COUNT],
            step: 0,
        })
    }

    pub(crate) fn from_parts(
        config: AdamConfig,
        first_moment: Vec<f32>,
        second_moment: Vec<f32>,
        step: u64,
        binding: OptimizerBinding,
    ) -> Result<Self, ModelError> {
        validate_adam_parts(
            config,
            &first_moment,
            &second_moment,
            step,
            MODEL_PARAMETER_COUNT,
        )?;
        Ok(Self {
            binding,
            config,
            first_moment,
            second_moment,
            step,
        })
    }

    pub const fn config(&self) -> AdamConfig {
        self.config
    }
    pub const fn step(&self) -> u64 {
        self.step
    }
    pub const fn policy_identity(&self) -> PolicyIdentity {
        self.binding.policy
    }
    pub(crate) const fn binding(&self) -> OptimizerBinding {
        self.binding
    }
    pub fn moments(&self) -> (&[f32], &[f32]) {
        (&self.first_moment, &self.second_moment)
    }
}

/// Pre-update behavioral loss and optimizer diagnostics for one effective batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelUpdateReport {
    pub average_loss: f64,
    pub active_head_counts: [usize; MODEL_BEHAVIORAL_HEADS],
    pub unclipped_norm: f64,
    pub applied_scale: f64,
    pub sample_count: usize,
    pub optimizer_step: u64,
}

/// Valid zero-based ability slot selected in one training prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingAbilitySlot(u8);

impl TrainingAbilitySlot {
    /// Builds a slot inside the fixed eight-logit ability head.
    pub fn new(index: usize) -> Result<Self, ModelError> {
        if index >= MODEL_ABILITY_HEAD {
            return Err(ModelError::TrainingSlotIndex {
                family: "ability",
                index,
                maximum: MODEL_ABILITY_HEAD - 1,
            });
        }
        Ok(Self(index as u8))
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Valid zero-based item slot selected in one training prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingItemSlot(u8);

impl TrainingItemSlot {
    /// Builds a slot inside the fixed fifteen-logit item head.
    pub fn new(index: usize) -> Result<Self, ModelError> {
        if index >= MODEL_ITEM_HEAD {
            return Err(ModelError::TrainingSlotIndex {
                family: "item",
                index,
                maximum: MODEL_ITEM_HEAD - 1,
            });
        }
        Ok(Self(index as u8))
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Family-specific slot selected before conditional training heads are evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingSlot {
    /// Ability slot selected before target heads.
    Ability(TrainingAbilitySlot),
    /// Item slot selected before target or swap heads.
    Item(TrainingItemSlot),
}

/// Teacher-selected autoregressive prefix for one training frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingPrefix {
    kind: ActionKind,
    unit: Option<ControlledUnit>,
    slot: Option<TrainingSlot>,
}

impl TrainingPrefix {
    /// Builds one bounded kind, controlled-unit, and family-slot prefix.
    pub const fn new(
        kind: ActionKind,
        unit: Option<ControlledUnit>,
        slot: Option<TrainingSlot>,
    ) -> Self {
        Self { kind, unit, slot }
    }

    /// Top-level action family used by this teacher-forced prefix.
    pub const fn kind(self) -> ActionKind {
        self.kind
    }

    /// Controlled unit selected before a conditional family head.
    pub const fn unit(self) -> Option<ControlledUnit> {
        self.unit
    }

    /// Ability or item slot selected before a target or swap head.
    pub const fn slot(self) -> Option<TrainingSlot> {
        self.slot
    }
}

struct PolicyTensorTensors {
    side_raw: [[Tensor; 2]; 12],
    value: Tensor,
    kind: Tensor,
    controlled: Tensor,
    ability: Tensor,
    item: Tensor,
    swap: Tensor,
    learn: Tensor,
    shop: Tensor,
    loot: Tensor,
    target_mode: Tensor,
    put_mode: Tensor,
    entity_pointer: Tensor,
    point_pointer: Tensor,
}

/// Autograd-preserving output holding one complete parameter read session.
pub struct PolicyTensorOutput<'model> {
    model_identity: usize,
    tensors: PolicyTensorTensors,
    _parameter_guard: RwLockReadGuard<'model, ()>,
}

impl fmt::Debug for PolicyTensorOutput<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolicyTensorOutput")
            .field("shapes", &self.shapes())
            .finish_non_exhaustive()
    }
}

/// Exact tensor dimensions returned by [`PolicyModel::training_forward`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyTensorShapes {
    /// State-value tensor dimensions.
    pub value: Vec<usize>,
    /// Action-kind tensor dimensions.
    pub kind: Vec<usize>,
    /// Controlled-unit tensor dimensions.
    pub controlled: Vec<usize>,
    /// Ability-slot tensor dimensions.
    pub ability: Vec<usize>,
    /// Item-slot tensor dimensions.
    pub item: Vec<usize>,
    /// Swap-destination tensor dimensions.
    pub swap: Vec<usize>,
    /// Learn-slot tensor dimensions.
    pub learn: Vec<usize>,
    /// Shop tensor dimensions.
    pub shop: Vec<usize>,
    /// Loot tensor dimensions.
    pub loot: Vec<usize>,
    /// Target-mode tensor dimensions.
    pub target_mode: Vec<usize>,
    /// Put-mode tensor dimensions.
    pub put_mode: Vec<usize>,
    /// Entity-pointer tensor dimensions.
    pub entity_pointer: Vec<usize>,
    /// Point-pointer tensor dimensions.
    pub point_pointer: Vec<usize>,
}

impl PolicyTensorOutput<'_> {
    /// Shape `[batch, 1]` state values.
    pub const fn value(&self) -> &Tensor {
        &self.tensors.value
    }

    /// Shape `[batch, 16]` action-kind logits.
    pub const fn kind(&self) -> &Tensor {
        &self.tensors.kind
    }

    /// Shape `[batch, 2]` controlled-unit logits.
    pub const fn controlled(&self) -> &Tensor {
        &self.tensors.controlled
    }

    /// Shape `[batch, 8]` ability-slot logits.
    pub const fn ability(&self) -> &Tensor {
        &self.tensors.ability
    }

    /// Shape `[batch, 15]` item or source-slot logits.
    pub const fn item(&self) -> &Tensor {
        &self.tensors.item
    }

    /// Shape `[batch, 15]` swap-destination logits.
    pub const fn swap(&self) -> &Tensor {
        &self.tensors.swap
    }

    /// Shape `[batch, 6]` learn-slot logits.
    pub const fn learn(&self) -> &Tensor {
        &self.tensors.learn
    }

    /// Shape `[batch, 64]` shop logits.
    pub const fn shop(&self) -> &Tensor {
        &self.tensors.shop
    }

    /// Shape `[batch, 16]` loot logits.
    pub const fn loot(&self) -> &Tensor {
        &self.tensors.loot
    }

    /// Shape `[batch, 3]` None, Entity, and Point mode logits.
    pub const fn target_mode(&self) -> &Tensor {
        &self.tensors.target_mode
    }

    /// Shape `[batch, 2]` Underfoot and Point mode logits.
    pub const fn put_mode(&self) -> &Tensor {
        &self.tensors.put_mode
    }

    /// Shape `[batch, 96]` current-unit pointer logits.
    pub const fn entity_pointer(&self) -> &Tensor {
        &self.tensors.entity_pointer
    }

    /// Shape `[batch, 64]` point-candidate pointer logits.
    pub const fn point_pointer(&self) -> &Tensor {
        &self.tensors.point_pointer
    }

    /// Returns every head shape without converting tensor values to host storage.
    pub fn shapes(&self) -> PolicyTensorShapes {
        PolicyTensorShapes {
            value: self.value().dims().to_vec(),
            kind: self.kind().dims().to_vec(),
            controlled: self.controlled().dims().to_vec(),
            ability: self.ability().dims().to_vec(),
            item: self.item().dims().to_vec(),
            swap: self.swap().dims().to_vec(),
            learn: self.learn().dims().to_vec(),
            shop: self.shop().dims().to_vec(),
            loot: self.loot().dims().to_vec(),
            target_mode: self.target_mode().dims().to_vec(),
            put_mode: self.put_mode().dims().to_vec(),
            entity_pointer: self.entity_pointer().dims().to_vec(),
            point_pointer: self.point_pointer().dims().to_vec(),
        }
    }

    /// Checks every tensor value while preserving the existing autograd graph.
    pub fn validate_finite(&self) -> Result<(), ModelError> {
        side_actors::validate_training(&self.tensors)
    }

    /// Sums all heads into one scalar graph-connected probe loss.
    pub fn sum_all_heads(&self) -> Result<Tensor, ModelError> {
        sum_training_tensors(&self.tensors)
    }
}

/// One gradient in stable parameter export order.
pub struct NamedPolicyGradient {
    name: &'static str,
    parameter_shape: Vec<usize>,
    gradient: Option<Tensor>,
}

impl NamedPolicyGradient {
    /// Stable parameter name covered by the model schema descriptor.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Parameter dimensions expected by an optimizer update.
    pub fn parameter_shape(&self) -> &[usize] {
        &self.parameter_shape
    }

    /// Read-only gradient tensor, absent when the loss did not use this parameter.
    pub const fn gradient(&self) -> Option<&Tensor> {
        self.gradient.as_ref()
    }

    /// Gradient dimensions, absent when the parameter was outside the loss graph.
    pub fn gradient_shape(&self) -> Option<&[usize]> {
        self.gradient.as_ref().map(Tensor::dims)
    }
}

impl fmt::Debug for NamedPolicyGradient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NamedPolicyGradient")
            .field("name", &self.name)
            .field("parameter_shape", &self.parameter_shape)
            .field("gradient_shape", &self.gradient_shape())
            .finish()
    }
}

struct Linear {
    weight: Var,
    bias: Var,
}

impl Linear {
    fn fresh(
        input: usize,
        output: usize,
        generator: &mut Initializer,
        device: &Device,
    ) -> Result<Self, ModelError> {
        Self::fresh_with_gain(input, output, generator, device, 0.01)
    }

    fn fresh_with_gain(
        input: usize,
        output: usize,
        generator: &mut Initializer,
        device: &Device,
        gain: f32,
    ) -> Result<Self, ModelError> {
        assert!(input > 0);
        assert!(output > 0);
        let scale = (6.0f32 / input as f32).sqrt() * gain;
        let values = (0..input * output)
            .map(|_| generator.symmetric() * scale)
            .collect::<Vec<_>>();
        Ok(Self {
            weight: Var::from_tensor(&Tensor::from_vec(values, (input, output), device)?)?,
            bias: Var::from_tensor(&Tensor::zeros(output, DType::F32, device)?)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        Ok(input
            .matmul(self.weight.as_tensor())?
            .broadcast_add(self.bias.as_tensor())?)
    }

    fn parameters<'a>(
        &'a self,
        names: (&'static str, &'static str),
        output: &mut Vec<NamedParameter<'a>>,
    ) {
        output.push(NamedParameter {
            name: names.0,
            value: &self.weight,
        });
        output.push(NamedParameter {
            name: names.1,
            value: &self.bias,
        });
    }
}

struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn fresh(
        shapes: &[(usize, usize)],
        generator: &mut Initializer,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(shapes.len());
        for &(input, output) in shapes {
            layers.push(Linear::fresh_with_gain(
                input, output, generator, device, 1.0,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?.relu()?;
        }
        Ok(output)
    }

    fn parameters<'a>(
        &'a self,
        names: &[(&'static str, &'static str)],
        output: &mut Vec<NamedParameter<'a>>,
    ) {
        debug_assert_eq!(self.layers.len(), names.len());
        for (layer, name) in self.layers.iter().zip(names) {
            layer.parameters(*name, output);
        }
    }
}

/// The critic: one hidden ReLU layer over the shared trunk, then a linear
/// read-out multiplied by [`VALUE_OUTPUT_SCALE`].
///
/// Adam moves every parameter by about the learning rate per step whatever the
/// gradient scale, so a freshly initialized read-out needs thousands of steps to
/// reach returns of order one at actor learning rates. The fixed output scale is
/// a critic-only learning-rate multiplier that leaves the optimizer untouched.
struct ValueHead {
    hidden: Linear,
    output: Linear,
}

impl ValueHead {
    fn fresh(generator: &mut Initializer, device: &Device) -> Result<Self, ModelError> {
        Ok(Self {
            hidden: Linear::fresh_with_gain(TRUNK_WIDTH, VALUE_HIDDEN, generator, device, 1.0)?,
            output: Linear::fresh_with_gain(
                VALUE_HIDDEN,
                1,
                generator,
                device,
                0.01 / VALUE_OUTPUT_SCALE as f32,
            )?,
        })
    }

    fn forward(&self, trunk: &Tensor) -> Result<Tensor, ModelError> {
        Ok(self
            .output
            .forward(&self.hidden.forward(trunk)?.relu()?)?
            .affine(VALUE_OUTPUT_SCALE, 0.0)?)
    }

    fn parameters<'a>(&'a self, output: &mut Vec<NamedParameter<'a>>) {
        self.hidden
            .parameters(("value.0.weight", "value.0.bias"), output);
        self.output
            .parameters(("value.1.weight", "value.1.bias"), output);
    }
}

struct Embedding {
    value: Var,
}

impl Embedding {
    fn fresh(
        rows: usize,
        columns: usize,
        generator: &mut Initializer,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let scale = (3.0f32 / columns as f32).sqrt();
        let values = (0..rows * columns)
            .map(|_| generator.symmetric() * scale)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (rows, columns), device)?;
        Ok(Self {
            value: Var::from_tensor(&tensor)?,
        })
    }

    fn row(&self, index: usize) -> Result<Tensor, ModelError> {
        Ok(self.value.as_tensor().get(index)?.unsqueeze(0)?)
    }
}

struct Initializer {
    state: u64,
}

impl Initializer {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn symmetric(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let fraction = (value >> 40) as f32 / ((1u32 << 24) - 1) as f32;
        fraction * 2.0 - 1.0
    }
}

struct NamedParameter<'a> {
    name: &'static str,
    value: &'a Var,
}

fn apply_parameter_tensors(
    parameters: &[NamedParameter<'_>],
    replacements: &[Tensor],
    originals: &[Tensor],
    fail_after: Option<usize>,
) -> Result<(), ModelError> {
    for (index, (parameter, replacement)) in parameters.iter().zip(replacements).enumerate() {
        if let Err(error) = parameter.value.set(replacement) {
            restore_parameter_tensors(parameters, originals, &error.to_string())?;
            return Err(error.into());
        }
        if fail_after == Some(index) {
            restore_parameter_tensors(parameters, originals, "injected replacement failure")?;
            return Err(ModelError::InjectedParameterFailure { index });
        }
    }
    Ok(())
}

fn restore_parameter_tensors(
    parameters: &[NamedParameter<'_>],
    originals: &[Tensor],
    cause: &str,
) -> Result<(), ModelError> {
    for (parameter, original) in parameters.iter().zip(originals) {
        parameter.value.set(original).map_err(|rollback| {
            ModelError::Backend(format!(
                "parameter replacement failed ({cause}); rollback failed ({rollback})"
            ))
        })?;
    }
    Ok(())
}

/// Backend selected for policy parameters and tensor execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDevice {
    Cpu,
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda {
        ordinal: usize,
    },
}

impl PolicyDevice {
    fn candle(self) -> Result<Device, ModelError> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Self::Cuda { ordinal } => cuda_device(ordinal),
        }
    }
}

/// A CUDA device on the calling thread's per-thread stream, without cudarc's
/// per-allocation event pairs.
///
/// With event tracking on, cudarc creates and destroys two events for every
/// allocation but consults them only in multi-stream mode, which a context
/// that only uses candle's per-thread stream never enters. The events were
/// about half of every learner and lane thread's CUDA API time.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_device(ordinal: usize) -> Result<Device, ModelError> {
    let device = Device::new_cuda(ordinal)?;
    let cuda = device.as_cuda_device()?;
    // SAFETY: the context is fresh and never gets a second stream (drysua
    // creates none; candle uses the per-thread stream), so cudarc records and
    // waits on no event either way.
    unsafe { cuda.disable_event_tracking() };
    Ok(device)
}

/// F32 DeepSets policy with an autoregressive masked decoder.
pub struct PolicyModel {
    dire: side_actors::ActorHeads,
    parameter_lock: RwLock<()>,
    lineage: NonZeroU64,
    parameter_revision: AtomicU64,
    optimizer_lineage: AtomicU64,
    device_kind: PolicyDevice,
    tensor_device: Device,
    unit: Mlp,
    ability: Mlp,
    item: Mlp,
    point: Mlp,
    projectile: Mlp,
    loot: Mlp,
    trunk: Mlp,
    value: ValueHead,
    kind: Linear,
    kind_embedding: Embedding,
    unit_embedding: Embedding,
    ability_embedding: Embedding,
    item_embedding: Embedding,
    controlled: Linear,
    ability_head: Linear,
    item_head: Linear,
    swap_head: Linear,
    learn_head: Linear,
    shop_head: Linear,
    loot_head: Linear,
    target_mode: Linear,
    put_mode: Linear,
    entity_query: Linear,
    point_query: Linear,
}

struct PolicyEncoders {
    unit: Mlp,
    ability: Mlp,
    item: Mlp,
    point: Mlp,
    projectile: Mlp,
    loot: Mlp,
}

#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct PpoTestFaults {
    candidate_evaluation: bool,
    rollback_import: bool,
}

impl PolicyModel {
    /// Constructs fixed parameters from one explicit deterministic seed.
    pub fn fresh(seed: u64) -> Result<Self, ModelError> {
        Self::fresh_on(seed, PolicyDevice::Cpu)
    }

    /// Constructs deterministic parameters on one explicitly selected backend.
    pub fn fresh_on(seed: u64, device: PolicyDevice) -> Result<Self, ModelError> {
        let tensor_device = device.candle()?;
        let mut generator = Initializer::new(seed);
        let unit = Mlp::fresh(
            &[(UNIT_FEATURES, 64), (64, 128), (128, 128)],
            &mut generator,
            &tensor_device,
        )?;
        let ability = Mlp::fresh(
            &[(ABILITY_FEATURES, 64), (64, 64)],
            &mut generator,
            &tensor_device,
        )?;
        let item = Mlp::fresh(
            &[(ITEM_FEATURES, 64), (64, 64)],
            &mut generator,
            &tensor_device,
        )?;
        let point = Mlp::fresh(
            &[(POINT_FEATURES, 64), (64, 64)],
            &mut generator,
            &tensor_device,
        )?;
        let projectile = Mlp::fresh(
            &[(PROJECTILE_FEATURES, 64), (64, 64)],
            &mut generator,
            &tensor_device,
        )?;
        let loot = Mlp::fresh(
            &[(LOOT_FEATURES, 64), (64, 64)],
            &mut generator,
            &tensor_device,
        )?;
        Self::fresh_from_encoders(
            generator,
            PolicyEncoders {
                unit,
                ability,
                item,
                point,
                projectile,
                loot,
            },
            device,
            tensor_device,
        )
    }

    fn fresh_from_encoders(
        mut generator: Initializer,
        encoders: PolicyEncoders,
        device_kind: PolicyDevice,
        tensor_device: Device,
    ) -> Result<Self, ModelError> {
        let trunk = Mlp::fresh(
            &[(TRUNK_INPUT, 512), (512, 256), (256, 256)],
            &mut generator,
            &tensor_device,
        )?;
        let lineage = allocate_lineage(&NEXT_MODEL_LINEAGE, ModelError::ModelLineageUnavailable)?;
        Ok(Self {
            parameter_lock: RwLock::new(()),
            lineage,
            parameter_revision: AtomicU64::new(0),
            optimizer_lineage: AtomicU64::new(0),
            device_kind,
            tensor_device: tensor_device.clone(),
            unit: encoders.unit,
            ability: encoders.ability,
            item: encoders.item,
            point: encoders.point,
            projectile: encoders.projectile,
            loot: encoders.loot,
            trunk,
            value: ValueHead::fresh(&mut generator, &tensor_device)?,
            kind: Linear::fresh(256, 16, &mut generator, &tensor_device)?,
            kind_embedding: Embedding::fresh(16, 32, &mut generator, &tensor_device)?,
            unit_embedding: Embedding::fresh(2, 32, &mut generator, &tensor_device)?,
            ability_embedding: Embedding::fresh(8, 16, &mut generator, &tensor_device)?,
            item_embedding: Embedding::fresh(15, 16, &mut generator, &tensor_device)?,
            controlled: Linear::fresh(336, 2, &mut generator, &tensor_device)?,
            ability_head: Linear::fresh(336, 8, &mut generator, &tensor_device)?,
            item_head: Linear::fresh(336, 15, &mut generator, &tensor_device)?,
            swap_head: Linear::fresh(336, 15, &mut generator, &tensor_device)?,
            learn_head: Linear::fresh(336, 6, &mut generator, &tensor_device)?,
            shop_head: Linear::fresh(336, 64, &mut generator, &tensor_device)?,
            loot_head: Linear::fresh(336, 16, &mut generator, &tensor_device)?,
            target_mode: Linear::fresh(336, 3, &mut generator, &tensor_device)?,
            put_mode: Linear::fresh(336, 2, &mut generator, &tensor_device)?,
            entity_query: Linear::fresh(336, 128, &mut generator, &tensor_device)?,
            point_query: Linear::fresh(336, 64, &mut generator, &tensor_device)?,
            dire: side_actors::ActorHeads::fresh(&mut generator, &tensor_device)?,
        })
    }

    /// Backend currently owning every parameter tensor.
    pub const fn device(&self) -> PolicyDevice {
        self.device_kind
    }

    fn tensor_device(&self) -> &Device {
        &self.tensor_device
    }

    /// Exact number of scalar F32 parameters.
    pub const fn parameter_count(&self) -> usize {
        MODEL_PARAMETER_COUNT
    }

    /// Returns the process-local lineage and exact current parameter revision.
    pub fn policy_identity(&self) -> Result<PolicyIdentity, ModelError> {
        let _guard = self.read_parameter_lock()?;
        Ok(self.policy_identity_locked())
    }

    pub(crate) fn claim_optimizer(&self, config: AdamConfig) -> Result<AdamState, ModelError> {
        validate_adam_config(config)?;
        let _guard = self.write_parameter_lock()?;
        if self.optimizer_lineage.load(Ordering::Relaxed) != 0 {
            return Err(ModelError::OptimizerAlreadyOwned);
        }
        let lineage = allocate_lineage(
            &NEXT_OPTIMIZER_LINEAGE,
            ModelError::OptimizerLineageUnavailable,
        )?;
        let binding = OptimizerBinding {
            lineage,
            policy: self.policy_identity_locked(),
        };
        let adam = AdamState::new(config, binding)?;
        self.optimizer_lineage
            .store(lineage.get(), Ordering::Relaxed);
        Ok(adam)
    }

    pub(crate) fn install_training_checkpoint(
        &self,
        parameters: &[f32],
        config: AdamConfig,
        first_moment: Vec<f32>,
        second_moment: Vec<f32>,
        step: u64,
    ) -> Result<AdamState, ModelError> {
        validate_parameter_values(parameters)?;
        validate_adam_parts(
            config,
            &first_moment,
            &second_moment,
            step,
            MODEL_PARAMETER_COUNT,
        )?;
        let _guard = self.write_parameter_lock()?;
        if self.optimizer_lineage.load(Ordering::Relaxed) != 0 {
            return Err(ModelError::OptimizerAlreadyOwned);
        }
        let policy = self.next_policy_identity_locked()?;
        let lineage = allocate_lineage(
            &NEXT_OPTIMIZER_LINEAGE,
            ModelError::OptimizerLineageUnavailable,
        )?;
        let adam = AdamState::from_parts(
            config,
            first_moment,
            second_moment,
            step,
            OptimizerBinding { lineage, policy },
        )?;
        self.import_parameters_locked(parameters, None)?;
        self.parameter_revision
            .store(policy.revision, Ordering::Relaxed);
        self.optimizer_lineage
            .store(lineage.get(), Ordering::Relaxed);
        Ok(adam)
    }

    /// Evaluates one frame without mutating model state.
    pub fn evaluate(&self, frame: &FeatureFrame) -> Result<PolicyOutput, ModelError> {
        let mut outputs = self.evaluate_batch(std::slice::from_ref(frame))?;
        outputs
            .pop()
            .ok_or(ModelError::InvalidModelState("single-frame output"))
    }

    /// Evaluates a bounded nonempty batch in input order.
    pub fn evaluate_batch(&self, frames: &[FeatureFrame]) -> Result<Vec<PolicyOutput>, ModelError> {
        validate_batch(frames)?;
        let _guard = self.read_parameter_lock()?;
        let mut output = Vec::with_capacity(frames.len());
        for (chunk_index, chunk) in frames.chunks(MODEL_EVALUATION_MICROBATCH).enumerate() {
            let offset = chunk_index * MODEL_EVALUATION_MICROBATCH;
            output.extend(self.evaluate_chunk(chunk, offset)?);
        }
        Ok(output)
    }

    fn evaluate_chunk(
        &self,
        frames: &[FeatureFrame],
        batch_offset: usize,
    ) -> Result<Vec<PolicyOutput>, ModelError> {
        let routing = ActorRouting::new(frames, self.tensor_device(), false)?;
        let state = self.forward_frames(frames)?;
        let base = self.base_logits(&state, &routing, batch_offset)?;
        collect_outputs(base.value, base.kind, batch_offset)
    }

    /// Selects one greedy legal structured action and returns its state value.
    pub fn choose(
        &self,
        frame: &FeatureFrame,
        space: &ActionSpace,
    ) -> Result<PolicyChoice, ModelError> {
        if !frame.matches_action_space(space) {
            return Err(ModelError::FrameActionSpaceMismatch);
        }
        validate_batch(std::slice::from_ref(frame))?;
        let _guard = self.read_parameter_lock()?;
        let routing = ActorRouting::new(std::slice::from_ref(frame), self.tensor_device(), false)?;
        let state = self.forward_frames(std::slice::from_ref(frame))?;
        let values = self
            .value
            .forward(&state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let value = *values
            .first()
            .ok_or(ModelError::InvalidModelState("value head shape"))?;
        if !value.is_finite() {
            return Err(ModelError::NonFiniteOutput {
                field: "value",
                batch: 0,
                index: 0,
            });
        }
        let mut source = ModelDecoder {
            model: self,
            state,
            routing,
            rng: None,
            observed: None,
        };
        let action = decode_from_source(space, &mut source)?;
        if !space.allows(action) {
            return Err(ModelError::InvalidModelState("illegal decoded action"));
        }
        space
            .decode(action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        Ok(PolicyChoice { action, value })
    }

    /// Selects at most [`MODEL_SAMPLING_BATCH`] greedy legal actions in input order.
    pub fn choose_batch(
        &self,
        frames: &[FeatureFrame],
        action_spaces: &[ActionSpace],
    ) -> Result<Vec<PolicyChoice>, ModelError> {
        validate_policy_batch(frames, action_spaces)?;
        let rows = packed_rows(frames)?;
        let rows = rows.iter().collect::<Vec<_>>();
        let spaces = action_spaces.iter().collect::<Vec<_>>();
        let _guard = self.read_parameter_lock()?;
        Ok(self
            .selection_rows_locked(&rows, &spaces, None)?
            .into_iter()
            .map(|choice| PolicyChoice {
                action: choice.action,
                value: choice.value,
            })
            .collect())
    }

    /// Samples one legal autoregressive action and records exact old-policy statistics.
    pub fn sample(
        &self,
        frame: &FeatureFrame,
        space: &ActionSpace,
        rng: &mut PpoRng,
    ) -> Result<PpoPolicyChoice, ModelError> {
        if !frame.matches_action_space(space) {
            return Err(ModelError::FrameActionSpaceMismatch);
        }
        validate_batch(std::slice::from_ref(frame))?;
        let _guard = self.read_parameter_lock()?;
        let routing = ActorRouting::new(std::slice::from_ref(frame), self.tensor_device(), false)?;
        let state = self.forward_frames(std::slice::from_ref(frame))?;
        let value = self
            .value
            .forward(&state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        validate_value_rows(std::slice::from_ref(&value), 0)?;
        let mut source = ModelDecoder {
            model: self,
            state,
            routing,
            rng: Some(rng),
            observed: Some(SampledPathLogits::default()),
        };
        let action = decode_from_source(space, &mut source)?;
        let target = BehavioralTarget::from_action(frame, space, action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let (log_probability, entropy) = source
            .observed
            .as_ref()
            .ok_or(ModelError::InvalidModelState("sampled path logits"))?
            .statistics(&target)?;
        Ok(PpoPolicyChoice {
            frame: frame.clone(),
            target,
            action,
            policy: self.policy_identity_locked(),
            log_probability,
            entropy,
            value,
        })
    }

    /// Samples at most [`MODEL_SAMPLING_BATCH`] rows with one transactional RNG per row.
    pub fn sample_batch(
        &self,
        frames: &[FeatureFrame],
        action_spaces: &[ActionSpace],
        rngs: &mut [PpoRng],
    ) -> Result<Vec<PpoPolicyChoice>, ModelError> {
        validate_policy_batch(frames, action_spaces)?;
        validate_sampling_rng_count(frames.len(), rngs.len())?;
        let rows = packed_rows(frames)?;
        let rows = rows.iter().collect::<Vec<_>>();
        let spaces = action_spaces.iter().collect::<Vec<_>>();
        let statistics = vec![true; frames.len()];
        let _guard = self.read_parameter_lock()?;
        let policy = self.policy_identity_locked();
        let sampled = self.sample_rows_locked(&rows, &spaces, rngs, &statistics)?;
        sampled
            .into_iter()
            .zip(frames)
            .map(|(row, frame)| {
                let statistics = row
                    .statistics
                    .ok_or(ModelError::InvalidModelState("sampled row statistics"))?;
                Ok(PpoPolicyChoice {
                    frame: frame.clone(),
                    target: statistics.target,
                    action: row.action,
                    policy,
                    log_probability: statistics.log_probability,
                    entropy: statistics.entropy,
                    value: row.value,
                })
            })
            .collect()
    }

    /// Samples packed rows with one transactional RNG per row; behaviour statistics
    /// are computed only for rows that request them.
    #[cfg(feature = "builtin")]
    pub(crate) fn sample_rows(
        &self,
        rows: &[&EncoderRow],
        spaces: &[&ActionSpace],
        rngs: &mut [PpoRng],
        statistics: &[bool],
    ) -> Result<Vec<SampledRow>, ModelError> {
        let _guard = self.read_parameter_lock()?;
        self.sample_rows_locked(rows, spaces, rngs, statistics)
    }

    fn sample_rows_locked(
        &self,
        rows: &[&EncoderRow],
        spaces: &[&ActionSpace],
        rngs: &mut [PpoRng],
        statistics: &[bool],
    ) -> Result<Vec<SampledRow>, ModelError> {
        validate_row_batch(rows.len(), spaces.len())?;
        validate_sampling_rng_count(rows.len(), rngs.len())?;
        if statistics.len() != rows.len() {
            return Err(ModelError::InvalidModelState(
                "sampled row statistics count",
            ));
        }
        let mut staged_rngs = rngs.to_vec();
        let selected =
            self.selection_rows_locked(rows, spaces, Some(staged_rngs.as_mut_slice()))?;
        let sampled = selected
            .into_iter()
            .zip(spaces)
            .zip(statistics)
            .map(|((selection, space), &wanted)| finish_sampled_row(space, selection, wanted))
            .collect::<Result<Vec<_>, _>>()?;
        rngs.clone_from_slice(&staged_rngs);
        Ok(sampled)
    }

    fn selection_rows_locked(
        &self,
        rows: &[&EncoderRow],
        action_spaces: &[&ActionSpace],
        mut rngs: Option<&mut [PpoRng]>,
    ) -> Result<Vec<BatchSelection>, ModelError> {
        let state = self.forward_rows(rows)?;
        let routing = ActorRouting::from_rows(rows, self.tensor_device())?;
        let base = self.base_logits(&state, &routing, 0)?;
        let mut rows = initialize_sampling_rows(&base, action_spaces, &mut rngs)?;
        let kind = self.sampling_kind_logits(&state, &sampling_prefixes(&rows), &routing)?;
        select_sampling_units(&mut rows, &kind, action_spaces, &mut rngs)?;
        let unit = self.sampling_unit_logits(&state, &sampling_prefixes(&rows), &routing)?;
        select_sampling_slots(&mut rows, &unit, action_spaces, &mut rngs)?;
        let slot = self.sampling_slot_logits(&state, &sampling_prefixes(&rows), &routing)?;
        decode_batch_rows(
            action_spaces,
            &mut rngs,
            rows,
            SamplingLogits {
                base,
                kind,
                unit,
                slot,
            },
        )
    }

    pub fn action_statistics(
        &self,
        frame: &FeatureFrame,
        space: &ActionSpace,
        action: StructuredAction,
    ) -> Result<(f32, f32, f32), ModelError> {
        if !frame.matches_action_space(space) {
            return Err(ModelError::FrameActionSpaceMismatch);
        }
        let target = BehavioralTarget::from_action(frame, space, action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let _guard = self.read_parameter_lock()?;
        let statistics = self.policy_path_statistics_locked(frame, &target)?;
        Ok((
            statistics.log_probability,
            statistics.entropy,
            statistics.value,
        ))
    }

    fn policy_path_statistics_locked(
        &self,
        frame: &FeatureFrame,
        target: &BehavioralTarget,
    ) -> Result<PolicyPathStatistics, ModelError> {
        let output = self.training_forward_locked(
            std::slice::from_ref(frame),
            std::slice::from_ref(&target.prefix()),
        )?;
        validate_training_tensors_finite(&output)?;
        let value = output.value.flatten_all()?.to_vec1::<f32>()?[0];
        let logits = BehavioralHostLogits::from_tensors(&output)?;
        let (log_probability, entropy) = logits.statistics(0, target)?;
        Ok(PolicyPathStatistics {
            log_probability,
            entropy,
            value,
        })
    }

    /// Exports parameters in stable descriptor order.
    pub fn export_parameters(&self) -> Result<Vec<f32>, ModelError> {
        let _guard = self.read_parameter_lock()?;
        self.export_parameters_locked()
    }

    /// FNV-1a over the little-endian bits of one coherent parameter export.
    /// Run scopes record it to pin frozen opponent weights; reports log it.
    pub fn parameter_fingerprint(&self) -> Result<u64, ModelError> {
        Ok(parameter_fingerprint_of(&self.export_parameters()?))
    }

    fn export_parameters_locked(&self) -> Result<Vec<f32>, ModelError> {
        let parameters = self.parameters();
        let mut output = Vec::with_capacity(MODEL_PARAMETER_COUNT);
        for parameter in parameters {
            output.extend(parameter.value.flatten_all()?.to_vec1::<f32>()?);
        }
        if output.len() != MODEL_PARAMETER_COUNT {
            return Err(ModelError::InvalidModelState("parameter count"));
        }
        Ok(output)
    }

    /// Atomically imports finite parameters in stable descriptor order.
    pub fn import_parameters(&self, values: &[f32]) -> Result<(), ModelError> {
        self.import_parameters_inner(values, None)
    }

    fn import_parameters_inner(
        &self,
        values: &[f32],
        fail_after: Option<usize>,
    ) -> Result<(), ModelError> {
        validate_parameter_values(values)?;
        let _guard = self.write_parameter_lock()?;
        let next = self.next_policy_identity_locked()?;
        self.import_parameters_locked(values, fail_after)?;
        self.parameter_revision
            .store(next.revision, Ordering::Relaxed);
        self.optimizer_lineage.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn import_parameters_locked(
        &self,
        values: &[f32],
        fail_after: Option<usize>,
    ) -> Result<(), ModelError> {
        let parameters = self.parameters();
        let originals = parameters
            .iter()
            .map(|parameter| Ok(parameter.value.as_tensor().copy()?.detach()))
            .collect::<Result<Vec<_>, ModelError>>()?;
        let mut tensors = Vec::with_capacity(parameters.len());
        let mut offset = 0usize;
        for parameter in &parameters {
            let count = parameter.value.elem_count();
            let shape = parameter.value.shape().clone();
            tensors.push(Tensor::from_vec(
                values[offset..offset + count].to_vec(),
                shape,
                self.tensor_device(),
            )?);
            offset += count;
        }
        apply_parameter_tensors(&parameters, &tensors, &originals, fail_after)
    }

    pub(crate) fn coherent_snapshot(
        &self,
        adam: &AdamState,
    ) -> Result<ModelAdamSnapshot, ModelError> {
        let _guard = self.read_parameter_lock()?;
        self.validate_optimizer_binding_locked(adam.binding)?;
        validate_adam_parts(
            adam.config,
            &adam.first_moment,
            &adam.second_moment,
            adam.step,
            MODEL_PARAMETER_COUNT,
        )?;
        Ok(ModelAdamSnapshot {
            parameters: self.export_parameters_locked()?,
            adam: adam.clone(),
        })
    }

    pub(crate) fn restore_snapshot(
        &self,
        snapshot: &ModelAdamSnapshot,
        adam: &mut AdamState,
        expected: OptimizerBinding,
    ) -> Result<OptimizerBinding, ModelError> {
        self.restore_snapshot_inner(snapshot, adam, expected, None)
    }

    fn restore_snapshot_inner(
        &self,
        snapshot: &ModelAdamSnapshot,
        adam: &mut AdamState,
        expected: OptimizerBinding,
        fail_after: Option<usize>,
    ) -> Result<OptimizerBinding, ModelError> {
        validate_parameter_values(&snapshot.parameters)?;
        validate_adam_parts(
            snapshot.adam.config,
            &snapshot.adam.first_moment,
            &snapshot.adam.second_moment,
            snapshot.adam.step,
            MODEL_PARAMETER_COUNT,
        )?;
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(expected)?;
        let next = self.next_policy_identity_locked()?;
        self.import_parameters_locked(&snapshot.parameters, fail_after)?;
        let binding = OptimizerBinding {
            lineage: expected.lineage,
            policy: next,
        };
        let mut restored = snapshot.adam.clone();
        restored.binding = binding;
        *adam = restored;
        self.parameter_revision
            .store(next.revision, Ordering::Relaxed);
        Ok(binding)
    }

    /// Stable names and shapes in parameter export order.
    pub fn parameter_schema(&self) -> Result<Vec<(&'static str, Vec<usize>)>, ModelError> {
        let _guard = self.read_parameter_lock()?;
        Ok(self
            .parameters()
            .into_iter()
            .map(|parameter| (parameter.name, parameter.value.dims().to_vec()))
            .collect())
    }

    /// Evaluates every trainable head for bounded teacher-selected prefixes.
    pub fn training_forward<'model>(
        &'model self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorOutput<'model>, ModelError> {
        validate_training_batch(frames, prefixes)?;
        let guard = self.read_parameter_lock()?;
        let tensors = self.training_forward_locked(frames, prefixes)?;
        validate_training_tensors_finite(&tensors)?;
        Ok(PolicyTensorOutput {
            model_identity: std::ptr::from_ref(self).addr(),
            tensors,
            _parameter_guard: guard,
        })
    }

    /// Every head over one shared trunk; the value loss trains the trunk too.
    fn training_forward_locked(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorTensors, ModelError> {
        let routing = ActorRouting::new(frames, self.tensor_device(), true)?;
        let state = self.forward_frames(frames)?;
        let prefixes = PrefixUpload::new(prefixes, self.tensor_device())?;
        self.training_heads(state, routing, &prefixes)
    }

    /// Training outputs of one staged microbatch.
    fn training_forward_inputs(
        &self,
        inputs: &device_learner::StagedInputs,
    ) -> Result<PolicyTensorTensors, ModelError> {
        let routing = ActorRouting::from_mask(inputs.sides.clone());
        let state = self.forward_encoder_inputs(&inputs.encoder)?;
        self.training_heads(state, routing, &inputs.prefixes)
    }

    fn training_heads(
        &self,
        state: ForwardState,
        routing: ActorRouting,
        prefixes: &PrefixUpload,
    ) -> Result<PolicyTensorTensors, ModelError> {
        let contexts = self.training_contexts(&state.trunk, prefixes)?;
        let entity_query = routing
            .forward(self, ActorHead::EntityQuery, &contexts.slot)?
            .unsqueeze(1)?;
        let point_query = routing
            .forward(self, ActorHead::PointQuery, &contexts.slot)?
            .unsqueeze(1)?;
        Ok(PolicyTensorTensors {
            value: self.value.forward(&state.trunk)?,
            kind: routing.forward(self, ActorHead::Kind, &state.trunk)?,
            controlled: routing.forward(self, ActorHead::Controlled, &contexts.kind)?,
            ability: routing.forward(self, ActorHead::Ability, &contexts.unit)?,
            item: routing.forward(self, ActorHead::Item, &contexts.unit)?,
            swap: routing.forward(self, ActorHead::Swap, &contexts.slot)?,
            learn: routing.forward(self, ActorHead::Learn, &contexts.kind)?,
            shop: routing.forward(self, ActorHead::Shop, &contexts.unit)?,
            loot: routing.forward(self, ActorHead::Loot, &contexts.unit)?,
            target_mode: routing.forward(self, ActorHead::TargetMode, &contexts.slot)?,
            put_mode: routing.forward(self, ActorHead::PutMode, &contexts.slot)?,
            entity_pointer: scaled_pointer_dot(&state.current_units, &entity_query)?,
            point_pointer: scaled_pointer_dot(&state.points, &point_query)?,
            side_raw: routing.into_raw()?,
        })
    }

    /// Backpropagates a scalar loss tied to one live guarded training output.
    pub fn backward_named(
        &self,
        output: &PolicyTensorOutput<'_>,
        loss: &Tensor,
    ) -> Result<Vec<NamedPolicyGradient>, ModelError> {
        if output.model_identity != std::ptr::from_ref(self).addr() {
            return Err(ModelError::TrainingOutputModelMismatch);
        }
        self.backward_named_locked(loss)
    }

    fn backward_named_locked(&self, loss: &Tensor) -> Result<Vec<NamedPolicyGradient>, ModelError> {
        let gradients = loss.backward()?;
        Ok(self
            .parameters()
            .into_iter()
            .map(|parameter| NamedPolicyGradient {
                name: parameter.name,
                parameter_shape: parameter.value.dims().to_vec(),
                gradient: gradients.get(parameter.value.as_tensor()).cloned(),
            })
            .collect())
    }

    /// Value and side-selected kind logits; errors report rows offset by `batch_offset`.
    fn base_logits(
        &self,
        state: &ForwardState,
        routing: &ActorRouting,
        batch_offset: usize,
    ) -> Result<SamplingBaseLogits, ModelError> {
        let mut request = StageRequest::new(state.trunk.dim(0)?);
        let value = request.push(self.value.forward(&state.trunk)?)?;
        let kind = routing.queue_pair(self, ActorHead::Kind, &state.trunk, &mut request)?;
        let values = request.read()?;
        let value = values.column(value);
        validate_value_rows(&value, batch_offset)?;
        let kind = routing.select(&values, kind).map_err(|error| match error {
            ModelError::NonFiniteOutput {
                field,
                batch,
                index,
            } => ModelError::NonFiniteOutput {
                field,
                batch: batch_offset + batch,
                index,
            },
            error => error,
        })?;
        Ok(SamplingBaseLogits { value, kind })
    }

    fn sampling_kind_logits(
        &self,
        state: &ForwardState,
        prefixes: &[TrainingPrefix],
        routing: &ActorRouting,
    ) -> Result<SamplingKindLogits, ModelError> {
        let [controlled, learn] = sampling::kind_needed(prefixes);
        if !controlled && !learn {
            return Ok(SamplingKindLogits::default());
        }
        let context = self.sampling_context(&state.trunk, prefixes, SamplingContext::Kind)?;
        let mut request = StageRequest::new(prefixes.len());
        let mut queue =
            |needed, head| self.queue_actor_head(needed, head, &context, routing, &mut request);
        let controlled = queue(controlled, ActorHead::Controlled)?;
        let learn = queue(learn, ActorHead::Learn)?;
        let values = request.read()?;
        Ok(SamplingKindLogits {
            controlled: select_queued(routing, &values, controlled)?,
            learn: select_queued(routing, &values, learn)?,
        })
    }

    fn sampling_unit_logits(
        &self,
        state: &ForwardState,
        prefixes: &[TrainingPrefix],
        routing: &ActorRouting,
    ) -> Result<SamplingUnitLogits, ModelError> {
        let needed = sampling::needed(prefixes, |kind| {
            use ActionKind::*;
            [
                kind == Cast,
                matches!(kind, Use | PutPoint | PutUnit | Sell | Swap),
                kind == Buy,
                kind == Take,
                matches!(kind, FollowUnit | AttackUnit),
                matches!(kind, MovePoint | AttackMovePoint),
            ]
        });
        if !needed.contains(&true) {
            return Ok(SamplingUnitLogits::default());
        }
        let [ability, item, shop, loot, entity, point] = needed;
        let context = self.sampling_context(&state.trunk, prefixes, SamplingContext::Unit)?;
        let mut request = StageRequest::new(prefixes.len());
        let mut queue =
            |needed, head| self.queue_actor_head(needed, head, &context, routing, &mut request);
        let ability = queue(ability, ActorHead::Ability)?;
        let item = queue(item, ActorHead::Item)?;
        let shop = queue(shop, ActorHead::Shop)?;
        let loot = queue(loot, ActorHead::Loot)?;
        let mut pointer = |needed, tokens, head| {
            self.queue_pointer_head(needed, &context, tokens, head, routing, &mut request)
        };
        let entity = pointer(entity, &state.current_units, ActorHead::EntityQuery)?;
        let point = pointer(point, &state.points, ActorHead::PointQuery)?;
        let values = request.read()?;
        Ok(SamplingUnitLogits {
            ability: select_queued(routing, &values, ability)?,
            item: select_queued(routing, &values, item)?,
            shop: select_queued(routing, &values, shop)?,
            loot: select_queued(routing, &values, loot)?,
            entity: pointer_queued(routing, &values, entity)?,
            point: pointer_queued(routing, &values, point)?,
        })
    }

    fn sampling_slot_logits(
        &self,
        state: &ForwardState,
        prefixes: &[TrainingPrefix],
        routing: &ActorRouting,
    ) -> Result<SamplingSlotLogits, ModelError> {
        let needed = sampling::needed(prefixes, |kind| {
            use ActionKind::*;
            // Target/put mode has not been sampled: keep every potentially traversed pointer.
            [
                kind == Swap,
                matches!(kind, Cast | Use),
                kind == PutPoint,
                matches!(kind, Cast | Use | PutUnit),
                matches!(kind, Cast | Use | PutPoint),
            ]
        });
        if !needed.contains(&true) {
            return Ok(SamplingSlotLogits::default());
        }
        let [swap, target_mode, put_mode, entity, point] = needed;
        let context = self.sampling_context(&state.trunk, prefixes, SamplingContext::Slot)?;
        let mut request = StageRequest::new(prefixes.len());
        let mut queue =
            |needed, head| self.queue_actor_head(needed, head, &context, routing, &mut request);
        let swap = queue(swap, ActorHead::Swap)?;
        let target_mode = queue(target_mode, ActorHead::TargetMode)?;
        let put_mode = queue(put_mode, ActorHead::PutMode)?;
        let mut pointer = |needed, tokens, head| {
            self.queue_pointer_head(needed, &context, tokens, head, routing, &mut request)
        };
        let entity = pointer(entity, &state.current_units, ActorHead::EntityQuery)?;
        let point = pointer(point, &state.points, ActorHead::PointQuery)?;
        let values = request.read()?;
        Ok(SamplingSlotLogits {
            swap: select_queued(routing, &values, swap)?,
            target_mode: select_queued(routing, &values, target_mode)?,
            put_mode: select_queued(routing, &values, put_mode)?,
            entity: pointer_queued(routing, &values, entity)?,
            point: pointer_queued(routing, &values, point)?,
        })
    }

    fn queue_actor_head(
        &self,
        needed: bool,
        head: ActorHead,
        context: &Tensor,
        routing: &ActorRouting,
        request: &mut StageRequest,
    ) -> Result<Option<QueuedPair>, ModelError> {
        if !needed {
            return Ok(None);
        }
        let (batch, width) = context.dims2()?;
        assert!((1..=MODEL_SAMPLING_BATCH).contains(&batch));
        assert_eq!(width, DECODER_CONTEXT);
        #[cfg(test)]
        sampling::record_dispatch(batch);
        routing.queue_pair(self, head, context, request).map(Some)
    }

    fn queue_pointer_head(
        &self,
        needed: bool,
        context: &Tensor,
        tokens: &Tensor,
        head: ActorHead,
        routing: &ActorRouting,
        request: &mut StageRequest,
    ) -> Result<Option<(QueuedPair, usize)>, ModelError> {
        if !needed {
            return Ok(None);
        }
        #[cfg(test)]
        sampling::record_dispatch(context.dim(0)?);
        routing
            .queue_pointer(self, head, context, tokens, request)
            .map(Some)
    }

    fn sampling_context(
        &self,
        trunk: &Tensor,
        prefixes: &[TrainingPrefix],
        depth: SamplingContext,
    ) -> Result<Tensor, ModelError> {
        let batch = prefixes.len();
        let upload = PrefixUpload::new(prefixes, self.tensor_device())?;
        let kind = self.kind_embeddings(&upload)?;
        if matches!(depth, SamplingContext::Kind) {
            return self.kind_context_from_embedding(trunk, &kind);
        }
        let unit = self.unit_embeddings(&upload)?;
        let slot = match depth {
            SamplingContext::Slot => self.slot_embeddings(&upload)?,
            SamplingContext::Kind | SamplingContext::Unit => {
                Tensor::zeros((batch, SLOT_EMBEDDING), DType::F32, self.tensor_device())?
            }
        };
        Ok(Tensor::cat(&[trunk, &kind, &unit, &slot], 1)?)
    }

    fn kind_context_from_embedding(
        &self,
        trunk: &Tensor,
        kind: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let batch = trunk.dim(0)?;
        assert_eq!(kind.dims(), &[batch, KIND_EMBEDDING]);
        assert_eq!(trunk.dim(1)?, TRUNK_WIDTH);
        let unit = Tensor::zeros(
            (batch, UNIT_SELECTION_EMBEDDING),
            DType::F32,
            self.tensor_device(),
        )?;
        let slot = Tensor::zeros((batch, SLOT_EMBEDDING), DType::F32, self.tensor_device())?;
        Ok(Tensor::cat(&[trunk, kind, &unit, &slot], 1)?)
    }

    fn training_contexts(
        &self,
        trunk: &Tensor,
        upload: &PrefixUpload,
    ) -> Result<TrainingContexts, ModelError> {
        let kind = self.kind_embeddings(upload)?;
        let unit = self.unit_embeddings(upload)?;
        let slot = self.slot_embeddings(upload)?;
        let zero_unit = Tensor::zeros(unit.shape(), DType::F32, self.tensor_device())?;
        let zero_slot = Tensor::zeros(slot.shape(), DType::F32, self.tensor_device())?;
        Ok(TrainingContexts {
            kind: Tensor::cat(&[trunk, &kind, &zero_unit, &zero_slot], 1)?,
            unit: Tensor::cat(&[trunk, &kind, &unit, &zero_slot], 1)?,
            slot: Tensor::cat(&[trunk, &kind, &unit, &slot], 1)?,
        })
    }

    fn kind_embeddings(&self, upload: &PrefixUpload) -> Result<Tensor, ModelError> {
        Ok(self
            .kind_embedding
            .value
            .as_tensor()
            .index_select(upload.indices(PrefixIndex::Kind), 0)?)
    }

    fn unit_embeddings(&self, upload: &PrefixUpload) -> Result<Tensor, ModelError> {
        Ok(self
            .unit_embedding
            .value
            .as_tensor()
            .index_select(upload.indices(PrefixIndex::Unit), 0)?
            .broadcast_mul(upload.mask(PrefixMask::Unit))?)
    }

    fn slot_embeddings(&self, upload: &PrefixUpload) -> Result<Tensor, ModelError> {
        let ability = self
            .ability_embedding
            .value
            .as_tensor()
            .index_select(upload.indices(PrefixIndex::Ability), 0)?
            .broadcast_mul(upload.mask(PrefixMask::Ability))?;
        let item = self
            .item_embedding
            .value
            .as_tensor()
            .index_select(upload.indices(PrefixIndex::Item), 0)?
            .broadcast_mul(upload.mask(PrefixMask::Item))?;
        Ok((ability + item)?)
    }

    fn read_parameter_lock(&self) -> Result<RwLockReadGuard<'_, ()>, ModelError> {
        let guard = self
            .parameter_lock
            .read()
            .map_err(|_| ModelError::ParameterLockPoisoned)?;
        Ok(guard)
    }

    fn write_parameter_lock(&self) -> Result<RwLockWriteGuard<'_, ()>, ModelError> {
        let guard = self
            .parameter_lock
            .write()
            .map_err(|_| ModelError::ParameterLockPoisoned)?;
        Ok(guard)
    }

    fn policy_identity_locked(&self) -> PolicyIdentity {
        PolicyIdentity {
            lineage: self.lineage,
            revision: self.parameter_revision.load(Ordering::Relaxed),
        }
    }

    fn next_policy_identity_locked(&self) -> Result<PolicyIdentity, ModelError> {
        let revision = self
            .parameter_revision
            .load(Ordering::Relaxed)
            .checked_add(1)
            .ok_or(ModelError::ParameterRevisionOverflow)?;
        Ok(PolicyIdentity {
            lineage: self.lineage,
            revision,
        })
    }

    fn validate_optimizer_binding_locked(
        &self,
        binding: OptimizerBinding,
    ) -> Result<(), ModelError> {
        if binding.policy != self.policy_identity_locked()
            || self.optimizer_lineage.load(Ordering::Relaxed) != binding.lineage.get()
        {
            return Err(ModelError::OptimizerOwnershipMismatch);
        }
        Ok(())
    }

    fn forward_frames(&self, frames: &[FeatureFrame]) -> Result<ForwardState, ModelError> {
        let rows = frames
            .iter()
            .map(EncoderRow::from_frame)
            .collect::<Result<Vec<_>, _>>()?;
        self.forward_rows(&rows.iter().collect::<Vec<_>>())
    }

    fn forward_rows(&self, rows: &[&EncoderRow]) -> Result<ForwardState, ModelError> {
        let (host, lengths) = rows::assemble(rows);
        let total = host.len();
        let flat = Tensor::from_vec(host, total, self.tensor_device())?;
        let inputs = EncoderInputs::from_buffer(&flat, &lengths, rows.len())?;
        self.forward_encoder_inputs(&inputs)
    }

    fn forward_encoder_inputs(&self, inputs: &EncoderInputs) -> Result<ForwardState, ModelError> {
        #[cfg(test)]
        side_actors::record_encoder_forward();
        let batch = inputs.batch;
        let units = encode_units(
            self,
            &inputs.units.0,
            &inputs.units.1,
            &inputs.unit_groups,
            batch,
        )?;
        let own_units = encode_own_units(self, &inputs.own.0, &inputs.own.1, batch)?;
        let encode = |encoder: &Mlp, pair: &(Tensor, Tensor), tokens| {
            encode_tokens(encoder, &pair.0, &pair.1, tokens, batch)
        };
        let abilities = encode(&self.ability, &inputs.abilities, ABILITY_FEATURE_TOKENS)?;
        let items = encode(&self.item, &inputs.items, ITEM_FEATURE_TOKENS)?;
        let points = encode(&self.point, &inputs.points, POINT_FEATURE_TOKENS)?;
        let projectiles = encode(
            &self.projectile,
            &inputs.projectiles,
            PROJECTILE_FEATURE_TOKENS,
        )?;
        let loot = encode(&self.loot, &inputs.loot, LOOT_FEATURE_TOKENS)?;
        let trunk_input = Tensor::cat(
            &[
                &inputs.scalars,
                &own_units.fixed,
                &units.pooled,
                &abilities.pooled,
                &items.pooled,
                &points.pooled,
                &projectiles.pooled,
                &loot.pooled,
            ],
            1,
        )?;
        if trunk_input.dims() != [batch, TRUNK_INPUT] {
            return Err(ModelError::InvalidModelState("trunk input shape"));
        }
        let trunk = self.trunk.forward(&trunk_input)?;
        Ok(ForwardState {
            trunk,
            current_units: units.current,
            points: points.encoded,
        })
    }

    fn parameters(&self) -> Vec<NamedParameter<'_>> {
        let mut output = Vec::with_capacity(MODEL_PARAMETER_TENSORS);
        self.unit.parameters(
            &[
                ("unit.0.weight", "unit.0.bias"),
                ("unit.1.weight", "unit.1.bias"),
                ("unit.2.weight", "unit.2.bias"),
            ],
            &mut output,
        );
        self.ability.parameters(
            &[
                ("ability.0.weight", "ability.0.bias"),
                ("ability.1.weight", "ability.1.bias"),
            ],
            &mut output,
        );
        self.item.parameters(
            &[
                ("item.0.weight", "item.0.bias"),
                ("item.1.weight", "item.1.bias"),
            ],
            &mut output,
        );
        self.point.parameters(
            &[
                ("point.0.weight", "point.0.bias"),
                ("point.1.weight", "point.1.bias"),
            ],
            &mut output,
        );
        self.projectile.parameters(
            &[
                ("projectile.0.weight", "projectile.0.bias"),
                ("projectile.1.weight", "projectile.1.bias"),
            ],
            &mut output,
        );
        self.loot.parameters(
            &[
                ("loot.0.weight", "loot.0.bias"),
                ("loot.1.weight", "loot.1.bias"),
            ],
            &mut output,
        );
        self.trunk.parameters(
            &[
                ("trunk.0.weight", "trunk.0.bias"),
                ("trunk.1.weight", "trunk.1.bias"),
                ("trunk.2.weight", "trunk.2.bias"),
            ],
            &mut output,
        );
        self.decoder_parameters(&mut output);
        self.dire.parameters(&mut output);
        assert_eq!(output.len(), MODEL_PARAMETER_TENSORS);
        output
    }

    fn decoder_parameters<'a>(&'a self, output: &mut Vec<NamedParameter<'a>>) {
        self.value.parameters(output);
        self.kind.parameters(("kind.weight", "kind.bias"), output);
        output.push(NamedParameter {
            name: "kind_embedding.weight",
            value: &self.kind_embedding.value,
        });
        output.push(NamedParameter {
            name: "unit_embedding.weight",
            value: &self.unit_embedding.value,
        });
        output.push(NamedParameter {
            name: "ability_embedding.weight",
            value: &self.ability_embedding.value,
        });
        output.push(NamedParameter {
            name: "item_embedding.weight",
            value: &self.item_embedding.value,
        });
        self.controlled
            .parameters(("controlled.weight", "controlled.bias"), output);
        self.ability_head
            .parameters(("ability_head.weight", "ability_head.bias"), output);
        self.item_head
            .parameters(("item_head.weight", "item_head.bias"), output);
        self.swap_head
            .parameters(("swap_head.weight", "swap_head.bias"), output);
        self.learn_head
            .parameters(("learn_head.weight", "learn_head.bias"), output);
        self.shop_head
            .parameters(("shop_head.weight", "shop_head.bias"), output);
        self.loot_head
            .parameters(("loot_head.weight", "loot_head.bias"), output);
        self.target_mode
            .parameters(("target_mode.weight", "target_mode.bias"), output);
        self.put_mode
            .parameters(("put_mode.weight", "put_mode.bias"), output);
        self.entity_query
            .parameters(("entity_query.weight", "entity_query.bias"), output);
        self.point_query
            .parameters(("point_query.weight", "point_query.bias"), output);
    }
}

struct PolicyPathStatistics {
    log_probability: f32,
    entropy: f32,
    value: f32,
}

struct AdamDiagnostics {
    unclipped_norm: f64,
    applied_scale: f64,
}

struct BehavioralHostLogits {
    kind: Vec<Vec<f32>>,
    controlled: Vec<Vec<f32>>,
    ability: Vec<Vec<f32>>,
    item: Vec<Vec<f32>>,
    swap: Vec<Vec<f32>>,
    learn: Vec<Vec<f32>>,
    shop: Vec<Vec<f32>>,
    loot: Vec<Vec<f32>>,
    target_mode: Vec<Vec<f32>>,
    put_mode: Vec<Vec<f32>>,
    entity_pointer: Vec<Vec<f32>>,
    point_pointer: Vec<Vec<f32>>,
}

impl BehavioralHostLogits {
    fn from_tensors(output: &PolicyTensorTensors) -> Result<Self, ModelError> {
        Ok(Self {
            kind: output.kind.to_vec2()?,
            controlled: output.controlled.to_vec2()?,
            ability: output.ability.to_vec2()?,
            item: output.item.to_vec2()?,
            swap: output.swap.to_vec2()?,
            learn: output.learn.to_vec2()?,
            shop: output.shop.to_vec2()?,
            loot: output.loot.to_vec2()?,
            target_mode: output.target_mode.to_vec2()?,
            put_mode: output.put_mode.to_vec2()?,
            entity_pointer: output.entity_pointer.to_vec2()?,
            point_pointer: output.point_pointer.to_vec2()?,
        })
    }

    fn statistics(
        &self,
        index: usize,
        target: &BehavioralTarget,
    ) -> Result<(f32, f32), ModelError> {
        macro_rules! add_head {
            ($logp:ident, $entropy:ident, $values:ident, $field:ident) => {
                let (head_logp, head_entropy) =
                    host_head_statistics(self.row(&self.$values, index)?, &target.$field)?;
                $logp += head_logp;
                $entropy += head_entropy;
            };
        }
        let (mut log_probability, mut entropy) =
            host_head_statistics(self.row(&self.kind, index)?, &target.kind)?;
        add_head!(log_probability, entropy, controlled, controlled);
        add_head!(log_probability, entropy, ability, ability);
        add_head!(log_probability, entropy, item, item);
        add_head!(log_probability, entropy, swap, swap);
        add_head!(log_probability, entropy, learn, learn);
        add_head!(log_probability, entropy, shop, shop);
        add_head!(log_probability, entropy, loot, loot);
        add_head!(log_probability, entropy, target_mode, target_mode);
        add_head!(log_probability, entropy, put_mode, put_mode);
        add_head!(log_probability, entropy, entity_pointer, entity_pointer);
        add_head!(log_probability, entropy, point_pointer, point_pointer);
        if !log_probability.is_finite() || !entropy.is_finite() || entropy < 0.0 {
            return Err(ModelError::InvalidModelState("policy path statistics"));
        }
        Ok((log_probability, entropy))
    }

    fn row<'a>(&self, values: &'a [Vec<f32>], index: usize) -> Result<&'a [f32], ModelError> {
        values
            .get(index)
            .map(Vec::as_slice)
            .ok_or(ModelError::InvalidModelState(
                "behavioral output batch shape",
            ))
    }
}

fn host_head_statistics<const WIDTH: usize>(
    logits: &[f32],
    target: &HeadTarget<WIDTH>,
) -> Result<(f32, f32), ModelError> {
    if !target.active {
        return Ok((0.0, 0.0));
    }
    if logits.len() != WIDTH || !target.is_selected_legal() {
        return Err(ModelError::InvalidModelState("policy statistics head"));
    }
    let maximum = logits
        .iter()
        .zip(target.mask)
        .filter_map(|(value, legal)| legal.then_some(*value))
        .reduce(f32::max)
        .ok_or(ModelError::NoLegalContinuation)?;
    let sum = logits
        .iter()
        .zip(target.mask)
        .filter_map(|(value, legal)| legal.then_some((*value - maximum).exp()))
        .sum::<f32>();
    let log_normalizer = maximum + sum.ln();
    let log_probability = logits[target.selected] - log_normalizer;
    let entropy = logits
        .iter()
        .zip(target.mask)
        .filter(|(_, legal)| *legal)
        .map(|(value, _)| {
            let log_probability = *value - log_normalizer;
            -log_probability.exp() * log_probability
        })
        .sum();
    Ok((log_probability, entropy))
}

fn validate_adam_config(config: AdamConfig) -> Result<(), ModelError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(ModelError::InvalidAdamConfig("learning rate"));
    }
    if !config.beta1.is_finite() || !(0.0..1.0).contains(&config.beta1) {
        return Err(ModelError::InvalidAdamConfig("beta1"));
    }
    if !config.beta2.is_finite() || !(0.0..1.0).contains(&config.beta2) {
        return Err(ModelError::InvalidAdamConfig("beta2"));
    }
    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(ModelError::InvalidAdamConfig("epsilon"));
    }
    if !config.gradient_clip.is_finite() || config.gradient_clip <= 0.0 {
        return Err(ModelError::InvalidAdamConfig("gradient clip"));
    }
    Ok(())
}

fn validate_adam_parts(
    config: AdamConfig,
    first_moment: &[f32],
    second_moment: &[f32],
    step: u64,
    expected: usize,
) -> Result<(), ModelError> {
    validate_adam_config(config)?;
    validate_optimizer_length("first moment", first_moment.len(), expected)?;
    validate_optimizer_length("second moment", second_moment.len(), expected)?;
    if step > MODEL_MAX_OPTIMIZER_STEP {
        return Err(ModelError::OptimizerStepOverflow);
    }
    validate_moments("first", first_moment, false)?;
    validate_moments("second", second_moment, true)
}

fn validate_optimizer_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::OptimizerVectorLength {
            field,
            actual,
            expected,
        });
    }
    Ok(())
}

fn validate_moments(
    field: &'static str,
    values: &[f32],
    nonnegative: bool,
) -> Result<(), ModelError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || (nonnegative && **value < 0.0))
    {
        return Err(ModelError::NonFiniteMoment { field, index });
    }
    Ok(())
}

fn collect_outputs(
    values: Vec<f32>,
    kinds: Vec<Vec<f32>>,
    batch_offset: usize,
) -> Result<Vec<PolicyOutput>, ModelError> {
    if values.len() != kinds.len() {
        return Err(ModelError::InvalidModelState("batch output shape"));
    }
    let mut output = Vec::with_capacity(values.len());
    for (batch, (value, logits)) in values.into_iter().zip(kinds).enumerate() {
        if !value.is_finite() {
            return Err(ModelError::NonFiniteOutput {
                field: "value",
                batch: batch_offset + batch,
                index: 0,
            });
        }
        if let Some((index, _)) = logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ModelError::NonFiniteOutput {
                field: "kind",
                batch: batch_offset + batch,
                index,
            });
        }
        let kind_logits = logits
            .try_into()
            .map_err(|_| ModelError::InvalidModelState("kind head shape"))?;
        output.push(PolicyOutput { value, kind_logits });
    }
    Ok(output)
}

fn validate_parameter_values(values: &[f32]) -> Result<(), ModelError> {
    if values.len() != MODEL_PARAMETER_COUNT {
        return Err(ModelError::ParameterLength {
            actual: values.len(),
            expected: MODEL_PARAMETER_COUNT,
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteParameter { index });
    }
    Ok(())
}

fn validate_batch(frames: &[FeatureFrame]) -> Result<(), ModelError> {
    validate_batch_count(frames.len())?;
    if let Some((index, _)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| !frame.is_finite())
    {
        return Err(ModelError::NonFiniteFrame { index });
    }
    side_actors::validate_sides(frames)?;
    Ok(())
}

fn validate_policy_batch(
    frames: &[FeatureFrame],
    action_spaces: &[ActionSpace],
) -> Result<(), ModelError> {
    validate_batch_count(frames.len())?;
    if frames.len() > MODEL_SAMPLING_BATCH {
        return Err(ModelError::BatchTooLarge {
            count: frames.len(),
            maximum: MODEL_SAMPLING_BATCH,
        });
    }
    if action_spaces.len() != frames.len() {
        return Err(ModelError::BatchActionSpaceCount {
            action_spaces: action_spaces.len(),
            frames: frames.len(),
        });
    }
    if let Some((index, _)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| !frame.is_finite())
    {
        return Err(ModelError::NonFiniteFrame { index });
    }
    if let Some((index, _)) = frames
        .iter()
        .zip(action_spaces)
        .enumerate()
        .find(|(_, (frame, space))| !frame.matches_action_space(space))
    {
        return Err(ModelError::BatchFrameActionSpaceMismatch { index });
    }
    side_actors::validate_sides(frames)?;
    Ok(())
}

fn validate_row_batch(rows: usize, spaces: usize) -> Result<(), ModelError> {
    validate_batch_count(rows)?;
    if rows > MODEL_SAMPLING_BATCH {
        return Err(ModelError::BatchTooLarge {
            count: rows,
            maximum: MODEL_SAMPLING_BATCH,
        });
    }
    if spaces != rows {
        return Err(ModelError::BatchActionSpaceCount {
            action_spaces: spaces,
            frames: rows,
        });
    }
    Ok(())
}

fn validate_sampling_rng_count(frame_count: usize, rng_count: usize) -> Result<(), ModelError> {
    if rng_count != frame_count {
        return Err(ModelError::SamplingRngCount {
            rngs: rng_count,
            frames: frame_count,
        });
    }
    Ok(())
}

pub(crate) fn validate_batch_count(count: usize) -> Result<(), ModelError> {
    if count == 0 {
        return Err(ModelError::EmptyBatch);
    }
    if count > MODEL_MAX_BATCH {
        return Err(ModelError::BatchTooLarge {
            count,
            maximum: MODEL_MAX_BATCH,
        });
    }
    Ok(())
}

fn validate_training_batch(
    frames: &[FeatureFrame],
    prefixes: &[TrainingPrefix],
) -> Result<(), ModelError> {
    validate_training_batch_count(frames.len())?;
    validate_training_batch_inputs(frames, prefixes)
}

fn validate_training_batch_inputs(
    frames: &[FeatureFrame],
    prefixes: &[TrainingPrefix],
) -> Result<(), ModelError> {
    if prefixes.len() != frames.len() {
        return Err(ModelError::TrainingPrefixCount {
            prefixes: prefixes.len(),
            frames: frames.len(),
        });
    }
    if let Some((index, _)) = frames
        .iter()
        .enumerate()
        .find(|(_, frame)| !frame.is_finite())
    {
        return Err(ModelError::NonFiniteFrame { index });
    }
    side_actors::validate_sides(frames)?;
    Ok(())
}

pub(crate) fn validate_training_batch_count(count: usize) -> Result<(), ModelError> {
    if count == 0 {
        return Err(ModelError::EmptyTrainingBatch);
    }
    if count > MODEL_TRAINING_BATCH {
        return Err(ModelError::TrainingBatchTooLarge {
            count,
            maximum: MODEL_TRAINING_BATCH,
        });
    }
    Ok(())
}

fn training_slot_indices(prefixes: &[TrainingPrefix], ability: bool) -> (Vec<u32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(prefixes.len());
    let mut presence = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let selected = match prefix.slot {
            Some(TrainingSlot::Ability(slot)) if ability => Some(slot.index()),
            Some(TrainingSlot::Item(slot)) if !ability => Some(slot.index()),
            _ => None,
        };
        indices.push(selected.unwrap_or(0) as u32);
        presence.push(selected.is_some() as u8 as f32);
    }
    (indices, presence)
}

fn sampling_prefixes(rows: &[SamplingRow]) -> Vec<TrainingPrefix> {
    rows.iter().map(|row| row.prefix).collect()
}

fn batch_row_rng<'a>(rngs: &'a mut Option<&mut [PpoRng]>, index: usize) -> Option<&'a mut PpoRng> {
    rngs.as_deref_mut().map(|rngs| {
        assert!(index < rngs.len());
        &mut rngs[index]
    })
}

fn initialize_sampling_rows(
    logits: &SamplingBaseLogits,
    action_spaces: &[&ActionSpace],
    rngs: &mut Option<&mut [PpoRng]>,
) -> Result<Vec<SamplingRow>, ModelError> {
    if logits.value.len() != action_spaces.len() || logits.kind.len() != action_spaces.len() {
        return Err(ModelError::InvalidModelState("sampling base batch shape"));
    }
    let mut rows = Vec::with_capacity(action_spaces.len());
    for (index, space) in action_spaces.iter().enumerate() {
        let value = logits.value[index];
        if !value.is_finite() {
            return Err(ModelError::NonFiniteOutput {
                field: "value",
                batch: index,
                index: 0,
            });
        }
        let raw = finite_sampling_array("kind", index, &logits.kind)?;
        let perturbed = perturb_logits(raw, batch_row_rng(rngs, index))?;
        let selected = masked_argmax(&perturbed, space.kind_mask().as_array())?;
        let kind = ActionKind::from_index(selected)
            .ok_or(ModelError::InvalidModelState("sampling action kind"))?;
        let mut observed = SampledPathLogits::default();
        let mut sampled = SampledPathLogits::default();
        observed.kind = Some(raw);
        sampled.kind = Some(perturbed);
        rows.push(SamplingRow {
            value,
            prefix: TrainingPrefix::new(kind, None, None),
            observed,
            perturbed: sampled,
        });
    }
    Ok(rows)
}

fn select_sampling_units(
    rows: &mut [SamplingRow],
    logits: &SamplingKindLogits,
    action_spaces: &[&ActionSpace],
    rngs: &mut Option<&mut [PpoRng]>,
) -> Result<(), ModelError> {
    for index in 0..rows.len() {
        let kind = rows[index].prefix.kind();
        if matches!(kind, ActionKind::Continue | ActionKind::Learn) {
            continue;
        }
        let scores = sample_sampling_head(
            "controlled",
            index,
            logits.controlled.as_deref(),
            batch_row_rng(rngs, index),
            &mut rows[index].observed.controlled,
            &mut rows[index].perturbed.controlled,
        )?;
        let mask = action_spaces[index].controlled_unit_mask(kind);
        let selected = masked_argmax(&scores, mask.as_array())?;
        let unit = [ControlledUnit::Hero, ControlledUnit::Courier]
            .get(selected)
            .copied()
            .ok_or(ModelError::InvalidModelState("sampling controlled unit"))?;
        rows[index].prefix = TrainingPrefix::new(kind, Some(unit), None);
    }
    Ok(())
}

fn select_sampling_slots(
    rows: &mut [SamplingRow],
    logits: &SamplingUnitLogits,
    action_spaces: &[&ActionSpace],
    rngs: &mut Option<&mut [PpoRng]>,
) -> Result<(), ModelError> {
    for index in 0..rows.len() {
        let Some(slot) = select_sampling_slot(
            index,
            &mut rows[index],
            logits,
            action_spaces[index],
            batch_row_rng(rngs, index),
        )?
        else {
            continue;
        };
        let prefix = rows[index].prefix;
        rows[index].prefix = TrainingPrefix::new(prefix.kind(), prefix.unit(), Some(slot));
    }
    Ok(())
}

fn select_sampling_slot(
    batch: usize,
    row: &mut SamplingRow,
    logits: &SamplingUnitLogits,
    space: &ActionSpace,
    rng: Option<&mut PpoRng>,
) -> Result<Option<TrainingSlot>, ModelError> {
    let kind = row.prefix.kind();
    let unit = row.prefix.unit();
    match (kind, unit) {
        (ActionKind::Cast, Some(unit)) => {
            let scores = sample_sampling_head(
                "ability",
                batch,
                logits.ability.as_deref(),
                rng,
                &mut row.observed.ability,
                &mut row.perturbed.ability,
            )?;
            let mask = padded_mask::<MODEL_ABILITY_HEAD>(&space.ability_slot_mask(unit))?;
            let selected = masked_argmax(&scores, &mask)?;
            Ok(Some(TrainingSlot::Ability(TrainingAbilitySlot::new(
                selected,
            )?)))
        }
        (kind, Some(_)) if kind_has_item_slot_context(kind) => {
            select_sampling_item_slot(batch, row, logits, space, rng).map(Some)
        }
        _ => Ok(None),
    }
}

const fn kind_has_item_slot_context(kind: ActionKind) -> bool {
    matches!(
        kind,
        ActionKind::Use | ActionKind::PutPoint | ActionKind::PutUnit | ActionKind::Swap
    )
}

fn select_sampling_item_slot(
    batch: usize,
    row: &mut SamplingRow,
    logits: &SamplingUnitLogits,
    space: &ActionSpace,
    rng: Option<&mut PpoRng>,
) -> Result<TrainingSlot, ModelError> {
    let kind = row.prefix.kind();
    let unit = row
        .prefix
        .unit()
        .ok_or(ModelError::InvalidModelState("sampling item-slot unit"))?;
    let scores = sample_sampling_head(
        "item",
        batch,
        logits.item.as_deref(),
        rng,
        &mut row.observed.item,
        &mut row.perturbed.item,
    )?;
    let mask = sampling_item_slot_mask(space, kind, unit)?;
    let selected = masked_argmax(&scores, &mask)?;
    Ok(TrainingSlot::Item(TrainingItemSlot::new(selected)?))
}

fn sampling_item_slot_mask(
    space: &ActionSpace,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<[bool; MODEL_ITEM_HEAD], ModelError> {
    match kind {
        ActionKind::Use => padded_mask(&space.item_slot_mask(unit)),
        ActionKind::PutPoint => put_point_source_mask(space, unit),
        ActionKind::PutUnit => put_unit_source_mask(space, unit),
        ActionKind::Swap => Ok(swap_source_mask(space, unit)),
        _ => Err(ModelError::InvalidModelState(
            "sampling item-slot action kind",
        )),
    }
}

fn finite_sampling_array<const WIDTH: usize>(
    field: &'static str,
    batch: usize,
    rows: &[Vec<f32>],
) -> Result<[f32; WIDTH], ModelError> {
    let values = rows
        .get(batch)
        .ok_or(ModelError::InvalidModelState("sampling head batch shape"))?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteOutput {
            field,
            batch,
            index,
        });
    }
    values
        .as_slice()
        .try_into()
        .map_err(|_| ModelError::InvalidModelState("sampling decoder head shape"))
}

fn sample_sampling_head<const WIDTH: usize>(
    field: &'static str,
    batch: usize,
    rows: Option<&[Vec<f32>]>,
    rng: Option<&mut PpoRng>,
    observed: &mut Option<[f32; WIDTH]>,
    perturbed: &mut Option<[f32; WIDTH]>,
) -> Result<[f32; WIDTH], ModelError> {
    if let Some(values) = *perturbed {
        if observed.is_none() {
            return Err(ModelError::InvalidModelState(
                "sampling perturbed head without raw logits",
            ));
        }
        return Ok(values);
    }
    let rows = rows.ok_or(ModelError::InvalidModelState(
        "requested skipped sampling head",
    ))?;
    let raw = finite_sampling_array(field, batch, rows)?;
    let sampled = perturb_logits(raw, rng)?;
    *observed = Some(raw);
    *perturbed = Some(sampled);
    Ok(sampled)
}

fn validate_value_rows(values: &[f32], batch_offset: usize) -> Result<(), ModelError> {
    assert!(!values.is_empty());
    assert!(batch_offset + values.len() <= MODEL_MAX_BATCH);
    if let Some(batch) = values.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteOutput {
            field: "value",
            batch: batch_offset + batch,
            index: 0,
        });
    }
    Ok(())
}

fn validate_tensor_finite(field: &'static str, tensor: &Tensor) -> Result<(), ModelError> {
    let width = tensor.dims().last().copied().unwrap_or(1);
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    if let Some((flat, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteOutput {
            field,
            batch: flat / width,
            index: flat % width,
        });
    }
    Ok(())
}

fn validate_training_tensors_finite(output: &PolicyTensorTensors) -> Result<(), ModelError> {
    side_actors::validate_training(output)
}

fn sum_training_tensors(output: &PolicyTensorTensors) -> Result<Tensor, ModelError> {
    let tensors = [
        &output.kind,
        &output.controlled,
        &output.ability,
        &output.item,
        &output.swap,
        &output.learn,
        &output.shop,
        &output.loot,
        &output.target_mode,
        &output.put_mode,
        &output.entity_pointer,
        &output.point_pointer,
    ];
    let mut loss = output.value.sum_all()?;
    for tensor in tensors {
        loss = (loss + tensor.sum_all()?)?;
    }
    Ok(loss)
}

/// Validated side-selected rows of one optionally queued actor head.
fn select_queued(
    routing: &ActorRouting,
    values: &StageValues,
    pair: Option<QueuedPair>,
) -> Result<Option<Vec<Vec<f32>>>, ModelError> {
    pair.map(|pair| routing.select(values, pair)).transpose()
}

/// Pointer scores of one optionally queued pointer head after validating its raw query pair.
fn pointer_queued(
    routing: &ActorRouting,
    values: &StageValues,
    pointer: Option<(QueuedPair, usize)>,
) -> Result<Option<Vec<Vec<f32>>>, ModelError> {
    pointer
        .map(|(pair, scores)| {
            routing.validate(values, pair)?;
            Ok(values.rows(scores))
        })
        .transpose()
}

#[derive(Clone, Copy)]
enum PrefixIndex {
    Kind,
    Unit,
    Ability,
    Item,
}

#[derive(Clone, Copy)]
enum PrefixMask {
    Unit,
    Ability,
    Item,
}

/// Embedding indices and presence masks of one prefix batch.
///
/// Host prefixes upload with two copies instead of one per field; each field is
/// a view of those copies holding the same values the per-field uploads held.
struct PrefixUpload {
    indices: [Tensor; 4],
    masks: [Tensor; 3],
}

impl PrefixUpload {
    fn new(prefixes: &[TrainingPrefix], device: &Device) -> Result<Self, ModelError> {
        let batch = prefixes.len();
        assert!((1..=device_learner::MODEL_MAX_STAGED_ROWS).contains(&batch));
        let (ability, ability_mask) = training_slot_indices(prefixes, true);
        let (item, item_mask) = training_slot_indices(prefixes, false);
        let mut indices = Vec::with_capacity(4 * batch);
        indices.extend(prefixes.iter().map(|prefix| prefix.kind.index() as u32));
        indices.extend(
            prefixes
                .iter()
                .map(|prefix| prefix.unit.map_or(0, ControlledUnit::index) as u32),
        );
        indices.extend(ability);
        indices.extend(item);
        let mut masks = Vec::with_capacity(3 * batch);
        masks.extend(
            prefixes
                .iter()
                .map(|prefix| prefix.unit.is_some() as u8 as f32),
        );
        masks.extend(ability_mask);
        masks.extend(item_mask);
        assert_eq!(indices.len(), 4 * batch);
        assert_eq!(masks.len(), 3 * batch);
        let indices = Tensor::from_vec(indices, 4 * batch, device)?;
        let masks = Tensor::from_vec(masks, 3 * batch, device)?;
        let index = |field: usize| indices.narrow(0, field * batch, batch);
        let mask = |field: usize| masks.narrow(0, field * batch, batch)?.reshape((batch, 1));
        Ok(Self {
            indices: [index(0)?, index(1)?, index(2)?, index(3)?],
            masks: [mask(0)?, mask(1)?, mask(2)?],
        })
    }

    /// The rows at `rows`, gathered on the device.
    fn gather(&self, rows: &Tensor) -> Result<Self, ModelError> {
        let index = |tensor: &Tensor| tensor.index_select(rows, 0);
        Ok(Self {
            indices: [
                index(&self.indices[0])?,
                index(&self.indices[1])?,
                index(&self.indices[2])?,
                index(&self.indices[3])?,
            ],
            masks: [
                index(&self.masks[0])?,
                index(&self.masks[1])?,
                index(&self.masks[2])?,
            ],
        })
    }

    fn indices(&self, field: PrefixIndex) -> &Tensor {
        &self.indices[field as usize]
    }

    fn mask(&self, field: PrefixMask) -> &Tensor {
        &self.masks[field as usize]
    }
}

struct ForwardState {
    trunk: Tensor,
    current_units: Tensor,
    points: Tensor,
}

struct EncoderInputs {
    batch: usize,
    units: (Tensor, Tensor),
    unit_groups: Vec<Tensor>,
    own: (Tensor, Tensor),
    abilities: (Tensor, Tensor),
    items: (Tensor, Tensor),
    points: (Tensor, Tensor),
    projectiles: (Tensor, Tensor),
    loot: (Tensor, Tensor),
    scalars: Tensor,
}

impl EncoderInputs {
    fn from_buffer(flat: &Tensor, lengths: &[usize], batch: usize) -> Result<Self, ModelError> {
        assert_eq!(lengths.len(), 3 + UNIT_GROUPS + 12);
        let mut offset = 0usize;
        let mut views = Vec::with_capacity(lengths.len());
        for &length in lengths {
            views.push(flat.narrow(0, offset, length)?);
            offset = offset
                .checked_add(length)
                .ok_or(ModelError::InvalidModelState("staged input overflow"))?;
        }
        assert_eq!(offset, flat.elem_count());
        Self::from_parts(views, batch)
    }

    /// Encoder inputs from the 20 per-part tensors, each holding `batch` rows.
    fn from_parts(parts: Vec<Tensor>, batch: usize) -> Result<Self, ModelError> {
        assert_eq!(parts.len(), 3 + UNIT_GROUPS + 12);
        let mut views = parts.into_iter();
        let units = encoder_input_pair(&mut views, batch, ENCODER_UNIT_TOKENS, UNIT_FEATURES)?;
        let mut unit_groups = Vec::with_capacity(UNIT_GROUPS);
        for _ in 0..UNIT_GROUPS {
            unit_groups.push(views.next().expect("unit group view").reshape((
                batch,
                ENCODER_UNIT_TOKENS,
                1,
            ))?);
        }
        let own = encoder_input_pair(&mut views, batch, OWN_UNIT_FEATURE_TOKENS, UNIT_FEATURES)?;
        let abilities =
            encoder_input_pair(&mut views, batch, ABILITY_FEATURE_TOKENS, ABILITY_FEATURES)?;
        let items = encoder_input_pair(&mut views, batch, ITEM_FEATURE_TOKENS, ITEM_FEATURES)?;
        let points = encoder_input_pair(&mut views, batch, POINT_FEATURE_TOKENS, POINT_FEATURES)?;
        let projectiles = encoder_input_pair(
            &mut views,
            batch,
            PROJECTILE_FEATURE_TOKENS,
            PROJECTILE_FEATURES,
        )?;
        let loot = encoder_input_pair(&mut views, batch, LOOT_FEATURE_TOKENS, LOOT_FEATURES)?;
        let scalars = views
            .next()
            .expect("scalars view")
            .reshape((batch, ENCODER_SCALARS))?;
        assert!(views.next().is_none());
        Ok(Self {
            batch,
            units,
            unit_groups,
            own,
            abilities,
            items,
            points,
            projectiles,
            loot,
            scalars,
        })
    }
}

fn encoder_input_pair(
    views: &mut std::vec::IntoIter<Tensor>,
    batch: usize,
    tokens: usize,
    features: usize,
) -> Result<(Tensor, Tensor), ModelError> {
    let rows = views
        .next()
        .expect("encoder rows view")
        .reshape((batch * tokens, features))?;
    let presence = views
        .next()
        .expect("encoder presence view")
        .reshape((batch, tokens, 1))?;
    Ok((rows, presence))
}

struct SamplingBaseLogits {
    value: Vec<f32>,
    kind: Vec<Vec<f32>>,
}

#[derive(Default)]
struct SamplingKindLogits {
    controlled: Option<Vec<Vec<f32>>>,
    learn: Option<Vec<Vec<f32>>>,
}

#[derive(Default)]
struct SamplingUnitLogits {
    ability: Option<Vec<Vec<f32>>>,
    item: Option<Vec<Vec<f32>>>,
    shop: Option<Vec<Vec<f32>>>,
    loot: Option<Vec<Vec<f32>>>,
    entity: Option<Vec<Vec<f32>>>,
    point: Option<Vec<Vec<f32>>>,
}

#[derive(Default)]
struct SamplingSlotLogits {
    swap: Option<Vec<Vec<f32>>>,
    target_mode: Option<Vec<Vec<f32>>>,
    put_mode: Option<Vec<Vec<f32>>>,
    entity: Option<Vec<Vec<f32>>>,
    point: Option<Vec<Vec<f32>>>,
}

struct SamplingLogits {
    base: SamplingBaseLogits,
    kind: SamplingKindLogits,
    unit: SamplingUnitLogits,
    slot: SamplingSlotLogits,
}

#[derive(Clone, Copy)]
enum SamplingContext {
    Kind,
    Unit,
    Slot,
}

struct SamplingRow {
    value: f32,
    prefix: TrainingPrefix,
    observed: SampledPathLogits,
    perturbed: SampledPathLogits,
}

struct BatchSelection {
    action: StructuredAction,
    value: f32,
    observed: SampledPathLogits,
}

struct TrainingContexts {
    kind: Tensor,
    unit: Tensor,
    slot: Tensor,
}

struct UnitEncoding {
    pooled: Tensor,
    current: Tensor,
}

struct OwnUnitEncoding {
    fixed: Tensor,
}

struct TokenEncoding {
    pooled: Tensor,
    encoded: Tensor,
}

pub(crate) fn scaled_pointer_dot(tokens: &Tensor, query: &Tensor) -> Result<Tensor, ModelError> {
    let width = tokens.dim(2)?;
    assert!(matches!(width, UNIT_EMBEDDING | TOKEN_EMBEDDING));
    assert_eq!(query.dim(2)?, width);
    Ok(tokens
        .broadcast_mul(query)?
        .sum(2)?
        .affine(1.0 / (width as f64).sqrt(), 0.0)?)
}

pub(crate) fn condition_rows(
    values: &mut [f32],
    features: usize,
    divisors: &[(usize, f32)],
    semantic_id: Option<(usize, f32)>,
) {
    assert!(features > 0);
    assert!(values.len().is_multiple_of(features));
    assert!(
        divisors
            .iter()
            .all(|&(index, divisor)| index < features && divisor >= 1.0)
    );
    if let Some((index, maximum)) = semantic_id {
        assert!(index < features);
        assert!(maximum >= 1.0);
    }
    for row in values.chunks_exact_mut(features) {
        for &(index, divisor) in divisors {
            row[index] /= divisor;
        }
        if let Some((index, maximum)) = semantic_id {
            // Log scaling preserves useful separation of common low IDs without huge unknown-ID activations.
            row[index] = row[index].signum() * row[index].abs().ln_1p() / maximum.ln_1p();
        }
    }
}

/// Scalar encoder input width for one frame.
const ENCODER_SCALARS: usize = GLOBAL_FEATURES
    + HISTORY_SAMPLES * HISTORY_FEATURES
    + MAX_POLICY_HISTORY * POLICY_HISTORY_FEATURES
    + MAP_FEATURES;

/// Unit encoder input rows for one frame.
const ENCODER_UNIT_TOKENS: usize = UNIT_FEATURE_TOKENS + REMEMBERED_UNIT_FEATURE_TOKENS;

fn encode_units(
    model: &PolicyModel,
    rows: &Tensor,
    presence: &Tensor,
    groups: &[Tensor],
    batch: usize,
) -> Result<UnitEncoding, ModelError> {
    let tokens = ENCODER_UNIT_TOKENS;
    assert_eq!(rows.dims(), [batch * tokens, UNIT_FEATURES]);
    assert_eq!(presence.dims(), [batch, tokens, 1]);
    let encoded = model
        .unit
        .forward(rows)?
        .reshape((batch, tokens, UNIT_EMBEDDING))?;
    let encoded = encoded.broadcast_mul(presence)?;
    let pooled = pool_groups(&encoded, groups, batch, tokens, UNIT_EMBEDDING)?;
    let current = encoded.narrow(1, 0, UNIT_FEATURE_TOKENS)?;
    Ok(UnitEncoding { pooled, current })
}

pub(crate) fn unit_group(kind: f32) -> Option<usize> {
    match kind as u8 {
        1 => Some(0),
        2..=5 => Some(1),
        7..=10 => Some(2),
        6 => Some(3),
        11 | 12 => Some(4),
        _ => None,
    }
}

fn encode_own_units(
    model: &PolicyModel,
    rows: &Tensor,
    mask: &Tensor,
    batch: usize,
) -> Result<OwnUnitEncoding, ModelError> {
    assert_eq!(
        rows.dims(),
        [batch * OWN_UNIT_FEATURE_TOKENS, UNIT_FEATURES]
    );
    assert_eq!(mask.dims(), [batch, OWN_UNIT_FEATURE_TOKENS, 1]);
    let encoded =
        model
            .unit
            .forward(rows)?
            .reshape((batch, OWN_UNIT_FEATURE_TOKENS, UNIT_EMBEDDING))?;
    let fixed = encoded.broadcast_mul(mask)?.flatten_from(1)?;
    Ok(OwnUnitEncoding { fixed })
}

fn encode_tokens(
    encoder: &Mlp,
    rows: &Tensor,
    presence: &Tensor,
    tokens: usize,
    batch: usize,
) -> Result<TokenEncoding, ModelError> {
    let encoded = encoder
        .forward(rows)?
        .reshape((batch, tokens, TOKEN_EMBEDDING))?;
    let encoded = encoded.broadcast_mul(presence)?;
    let pooled = pool_groups(
        &encoded,
        std::slice::from_ref(presence),
        batch,
        tokens,
        TOKEN_EMBEDDING,
    )?;
    Ok(TokenEncoding { pooled, encoded })
}

fn pool_groups(
    encoded: &Tensor,
    masks: &[Tensor],
    batch: usize,
    tokens: usize,
    width: usize,
) -> Result<Tensor, ModelError> {
    assert!(!masks.is_empty());
    let mut pools = Vec::with_capacity(masks.len() * 2);
    let device = encoded.device();
    for mask in masks {
        assert_eq!(mask.dims(), [batch, tokens, 1]);
        let masked = encoded.broadcast_mul(mask)?;
        let counts = mask.sum(1)?;
        let denominator = counts.clamp(1.0f32, tokens as f32)?;
        let mean = masked.sum(1)?.broadcast_div(&denominator)?;
        let selected = mask.eq(1.0)?.broadcast_as((batch, tokens, width))?;
        let negative_infinity = Tensor::full(f32::NEG_INFINITY, (batch, tokens, width), device)?;
        let candidates = selected.where_cond(encoded, &negative_infinity)?;
        let indices = candidates.argmax_keepdim(1)?.contiguous()?;
        let maximum = encoded.gather(&indices, 1)?.squeeze(1)?;
        let present = counts.gt(0.0)?.broadcast_as((batch, width))?;
        let zeros = Tensor::zeros((batch, width), DType::F32, device)?;
        pools.push(mean);
        pools.push(present.where_cond(&maximum, &zeros)?);
    }
    let refs = pools.iter().collect::<Vec<_>>();
    let pooled = Tensor::cat(&refs, 1)?;
    if pooled.dims() != [batch, masks.len() * width * 2] {
        return Err(ModelError::InvalidModelState("pool shape"));
    }
    Ok(pooled)
}

pub(crate) fn masked_argmax(logits: &[f32], mask: &[bool]) -> Result<usize, ModelError> {
    if mask.is_empty() {
        return Err(ModelError::EmptyMask);
    }
    if logits.len() != mask.len() {
        return Err(ModelError::SelectionShape {
            logits: logits.len(),
            mask: mask.len(),
        });
    }
    if let Some((index, _)) = logits
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::SelectionNonFinite { index });
    }
    let mut selected = None;
    for (index, (&score, &allowed)) in logits.iter().zip(mask).enumerate() {
        if allowed && selected.is_none_or(|(_, best)| score > best) {
            selected = Some((index, score));
        }
    }
    selected
        .map(|(index, _)| index)
        .ok_or(ModelError::NoLegalContinuation)
}

trait DecoderSource {
    fn kind(&mut self) -> Result<[f32; 16], ModelError>;
    fn controlled(&mut self, kind: ActionKind) -> Result<[f32; 2], ModelError>;
    fn ability(
        &mut self,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
    ) -> Result<[f32; 8], ModelError>;
    fn item(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 15], ModelError>;
    fn swap(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: usize,
    ) -> Result<[f32; 15], ModelError>;
    fn learn(&mut self, kind: ActionKind) -> Result<[f32; 6], ModelError>;
    fn shop(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 64], ModelError>;
    fn loot(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 16], ModelError>;
    fn target_mode(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: SlotSelection,
    ) -> Result<[f32; 3], ModelError>;
    fn put_mode(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: usize,
    ) -> Result<[f32; 2], ModelError>;
    fn entity(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; 96], ModelError>;
    fn point(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; MODEL_POINT_POINTER_HEAD], ModelError>;
}

#[derive(Clone, Copy)]
enum SlotSelection {
    Ability(usize),
    Item(usize),
}

fn decode_from_source(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
) -> Result<StructuredAction, ModelError> {
    let kind_index = masked_argmax(&source.kind()?, space.kind_mask().as_array())?;
    let kind =
        ActionKind::from_index(kind_index).ok_or(ModelError::InvalidModelState("action kind"))?;
    if kind == ActionKind::Continue {
        return Ok(StructuredAction::Continue);
    }
    if kind == ActionKind::Learn {
        return decode_learn(space, source, kind);
    }
    let unit_mask = space.controlled_unit_mask(kind);
    let unit_index = masked_argmax(&source.controlled(kind)?, unit_mask.as_array())?;
    let unit = if unit_index == 0 {
        ControlledUnit::Hero
    } else {
        ControlledUnit::Courier
    };
    decode_controlled(space, source, kind, unit)
}

fn decode_controlled(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    match kind {
        ActionKind::Stop => Ok(StructuredAction::Stop { unit }),
        ActionKind::MovePoint => Ok(StructuredAction::MovePoint {
            unit,
            point: choose_point(space.move_point_mask(unit), source.point(kind, unit, None)?)?,
        }),
        ActionKind::FollowUnit => Ok(StructuredAction::FollowUnit {
            unit,
            target: choose_entity(
                space.follow_entity_mask(unit),
                source.entity(kind, unit, None)?,
            )?,
        }),
        ActionKind::Hold => Ok(StructuredAction::Hold { unit }),
        ActionKind::AttackMovePoint => Ok(StructuredAction::AttackMovePoint {
            unit,
            point: choose_point(
                space.attack_move_point_mask(unit),
                source.point(kind, unit, None)?,
            )?,
        }),
        ActionKind::AttackUnit => Ok(StructuredAction::AttackUnit {
            unit,
            target: choose_entity(
                space.attack_entity_mask(unit),
                source.entity(kind, unit, None)?,
            )?,
        }),
        ActionKind::Cast => decode_cast(space, source, kind, unit),
        ActionKind::Use => decode_use(space, source, kind, unit),
        ActionKind::PutPoint => decode_put_point(space, source, kind, unit),
        ActionKind::PutUnit => decode_put_unit(space, source, kind, unit),
        ActionKind::Take => decode_take(space, source, kind, unit),
        ActionKind::Buy => decode_buy(space, source, kind, unit),
        ActionKind::Sell => decode_sell(space, source, kind, unit),
        ActionKind::Swap => decode_swap(space, source, kind, unit),
        ActionKind::Continue | ActionKind::Learn => {
            Err(ModelError::InvalidModelState("controlled action kind"))
        }
    }
}

fn decode_cast(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let mask = padded_mask::<8>(&space.ability_slot_mask(unit))?;
    let slot = masked_argmax(&source.ability(kind, Some(unit))?, &mask)?;
    let slot_wire = bota_proto::AbilitySlot(slot as u8);
    let target_mask = space
        .cast_target_mask(unit, slot_wire)
        .ok_or(ModelError::NoLegalContinuation)?;
    let target = choose_target(
        source,
        kind,
        unit,
        SlotSelection::Ability(slot),
        target_mask,
    )?;
    Ok(StructuredAction::Cast {
        unit,
        slot: slot_wire,
        target,
    })
}

fn decode_use(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let mask = padded_mask::<15>(&space.item_slot_mask(unit))?;
    let slot = masked_argmax(&source.item(kind, unit)?, &mask)?;
    let slot_wire = bota_proto::ItemSlot(slot as u8);
    let target_mask = space
        .use_target_mask(unit, slot_wire)
        .ok_or(ModelError::NoLegalContinuation)?;
    let target = choose_target(source, kind, unit, SlotSelection::Item(slot), target_mask)?;
    Ok(StructuredAction::Use {
        unit,
        slot: slot_wire,
        target,
    })
}

fn choose_target(
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
    slot: SlotSelection,
    mask: &crate::TargetMask,
) -> Result<ActionTarget, ModelError> {
    let selected = select_target_mode(
        &source.target_mode(kind, unit, slot)?,
        mask.allows_none(),
        mask.entities(),
        mask.points(),
    )?;
    match selected {
        0 => Ok(ActionTarget::None),
        1 => Ok(ActionTarget::Entity(choose_entity(
            mask.entities(),
            source.entity(kind, unit, Some(slot))?,
        )?)),
        2 => Ok(ActionTarget::Point(choose_point(
            mask.points(),
            source.point(kind, unit, Some(slot))?,
        )?)),
        _ => Err(ModelError::InvalidModelState("target mode")),
    }
}

fn select_target_mode(
    scores: &[f32; 3],
    none: bool,
    entities: &[bool],
    points: &[bool],
) -> Result<usize, ModelError> {
    masked_argmax(
        scores,
        &[none, entities.contains(&true), points.contains(&true)],
    )
}

fn decode_put_point(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let source_mask = put_point_source_mask(space, unit)?;
    let selected = masked_argmax(&source.item(kind, unit)?, &source_mask)?;
    let slot = bota_proto::ItemSlot(selected as u8);
    let points = space
        .put_point_target_mask(unit, slot)
        .ok_or(ModelError::NoLegalContinuation)?;
    let modes = source.put_mode(kind, unit, selected)?;
    let underfoot = space
        .put_underfoot_mask(unit)
        .get(selected)
        .copied()
        .unwrap_or(false);
    let mode = masked_argmax(&modes, &[underfoot, points.contains(&true)])?;
    let target = if mode == 0 {
        PutPointTarget::Underfoot
    } else {
        let scores = source.point(kind, unit, Some(SlotSelection::Item(selected)))?;
        PutPointTarget::Point(choose_point(points, scores)?)
    };
    Ok(StructuredAction::PutPoint {
        unit,
        source: slot,
        target,
    })
}

fn decode_put_unit(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let source_mask = put_unit_source_mask(space, unit)?;
    let selected = masked_argmax(&source.item(kind, unit)?, &source_mask)?;
    let slot = bota_proto::ItemSlot(selected as u8);
    let mask = space
        .put_entity_target_mask(unit, slot)
        .ok_or(ModelError::NoLegalContinuation)?;
    let scores = source.entity(kind, unit, Some(SlotSelection::Item(selected)))?;
    let target = choose_entity(mask, scores)?;
    Ok(StructuredAction::PutUnit {
        unit,
        source: slot,
        target,
    })
}

fn put_point_source_mask(
    space: &ActionSpace,
    unit: ControlledUnit,
) -> Result<[bool; 15], ModelError> {
    let underfoot = space.put_underfoot_mask(unit);
    let mut mask = [false; 15];
    for index in 0..underfoot.len() {
        let slot = bota_proto::ItemSlot(index as u8);
        mask[index] = underfoot[index]
            || space
                .put_point_target_mask(unit, slot)
                .is_some_and(|points| points.contains(&true));
    }
    Ok(mask)
}

fn put_unit_source_mask(
    space: &ActionSpace,
    unit: ControlledUnit,
) -> Result<[bool; 15], ModelError> {
    let slots = space.put_source_slot_mask(unit);
    if slots.len() > 15 {
        return Err(ModelError::SelectionShape {
            logits: 15,
            mask: slots.len(),
        });
    }
    let mut mask = [false; 15];
    for (index, allowed) in mask.iter_mut().enumerate().take(slots.len()) {
        let slot = bota_proto::ItemSlot(index as u8);
        *allowed = space
            .put_entity_target_mask(unit, slot)
            .is_some_and(|entities| entities.contains(&true));
    }
    Ok(mask)
}

fn decode_take(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let mask = padded_mask::<16>(space.take_mask(unit))?;
    let index = masked_argmax(&source.loot(kind, unit)?, &mask)?;
    Ok(StructuredAction::Take {
        unit,
        loot: LootIndex(index),
    })
}

fn decode_buy(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let mask = padded_mask::<64>(space.buy_mask(unit))?;
    let index = masked_argmax(&source.shop(kind, unit)?, &mask)?;
    Ok(StructuredAction::Buy {
        unit,
        item: ShopIndex(index),
    })
}

fn decode_sell(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let index = masked_argmax(&source.item(kind, unit)?, space.sell_slot_mask(unit))?;
    Ok(StructuredAction::Sell {
        unit,
        slot: bota_proto::ItemSlot(index as u8),
    })
}

fn decode_swap(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
    unit: ControlledUnit,
) -> Result<StructuredAction, ModelError> {
    let sources = swap_source_mask(space, unit);
    let from = masked_argmax(&source.item(kind, unit)?, &sources)?;
    let row = space
        .swap_destination_mask(unit, bota_proto::ItemSlot(from as u8))
        .ok_or(ModelError::NoLegalContinuation)?;
    let to = masked_argmax(&source.swap(kind, unit, from)?, row)?;
    Ok(StructuredAction::Swap {
        unit,
        from: bota_proto::ItemSlot(from as u8),
        to: bota_proto::ItemSlot(to as u8),
    })
}

fn swap_source_mask(space: &ActionSpace, unit: ControlledUnit) -> [bool; MODEL_ITEM_HEAD] {
    std::array::from_fn(|index| {
        space
            .swap_destination_mask(unit, bota_proto::ItemSlot(index as u8))
            .is_some_and(|row| row.contains(&true))
    })
}

fn decode_learn(
    space: &ActionSpace,
    source: &mut impl DecoderSource,
    kind: ActionKind,
) -> Result<StructuredAction, ModelError> {
    let mask = padded_mask::<6>(space.learn_slot_mask())?;
    let slot = masked_argmax(&source.learn(kind)?, &mask)?;
    Ok(StructuredAction::Learn {
        slot: bota_proto::AbilitySlot(slot as u8),
    })
}

fn choose_entity(mask: &[bool], scores: [f32; 96]) -> Result<EntityIndex, ModelError> {
    let scores = scores.get(..mask.len()).ok_or(ModelError::SelectionShape {
        logits: 96,
        mask: mask.len(),
    })?;
    Ok(EntityIndex(masked_argmax(scores, mask)?))
}

fn choose_point(
    mask: &[bool],
    scores: [f32; MODEL_POINT_POINTER_HEAD],
) -> Result<PointIndex, ModelError> {
    let scores = scores.get(..mask.len()).ok_or(ModelError::SelectionShape {
        logits: MODEL_POINT_POINTER_HEAD,
        mask: mask.len(),
    })?;
    Ok(PointIndex(masked_argmax(scores, mask)?))
}

fn padded_mask<const SIZE: usize>(mask: &[bool]) -> Result<[bool; SIZE], ModelError> {
    if mask.len() > SIZE {
        return Err(ModelError::SelectionShape {
            logits: SIZE,
            mask: mask.len(),
        });
    }
    let mut output = [false; SIZE];
    output[..mask.len()].copy_from_slice(mask);
    Ok(output)
}

#[derive(Default)]
struct SampledPathLogits {
    kind: Option<[f32; MODEL_KIND_HEAD]>,
    controlled: Option<[f32; MODEL_UNIT_HEAD]>,
    ability: Option<[f32; MODEL_ABILITY_HEAD]>,
    item: Option<[f32; MODEL_ITEM_HEAD]>,
    swap: Option<[f32; MODEL_SWAP_HEAD]>,
    learn: Option<[f32; MODEL_LEARN_HEAD]>,
    shop: Option<[f32; MODEL_SHOP_HEAD]>,
    loot: Option<[f32; MODEL_LOOT_HEAD]>,
    target_mode: Option<[f32; TARGET_MODE_HEAD]>,
    put_mode: Option<[f32; PUT_MODE_HEAD]>,
    entity_pointer: Option<[f32; MODEL_ENTITY_POINTER_HEAD]>,
    point_pointer: Option<[f32; MODEL_POINT_POINTER_HEAD]>,
}

impl SampledPathLogits {
    fn statistics(&self, target: &BehavioralTarget) -> Result<(f32, f32), ModelError> {
        macro_rules! add_head {
            ($logp:ident, $entropy:ident, $field:ident) => {
                let (head_logp, head_entropy) =
                    sampled_head_statistics(self.$field.as_ref(), &target.$field)?;
                $logp += head_logp;
                $entropy += head_entropy;
            };
        }
        let (mut log_probability, mut entropy) =
            sampled_head_statistics(self.kind.as_ref(), &target.kind)?;
        add_head!(log_probability, entropy, controlled);
        add_head!(log_probability, entropy, ability);
        add_head!(log_probability, entropy, item);
        add_head!(log_probability, entropy, swap);
        add_head!(log_probability, entropy, learn);
        add_head!(log_probability, entropy, shop);
        add_head!(log_probability, entropy, loot);
        add_head!(log_probability, entropy, target_mode);
        add_head!(log_probability, entropy, put_mode);
        add_head!(log_probability, entropy, entity_pointer);
        add_head!(log_probability, entropy, point_pointer);
        Ok((log_probability, entropy))
    }
}

fn sampled_head_statistics<const WIDTH: usize>(
    logits: Option<&[f32; WIDTH]>,
    target: &HeadTarget<WIDTH>,
) -> Result<(f32, f32), ModelError> {
    if !target.active {
        return Ok((0.0, 0.0));
    }
    let logits = logits.ok_or(ModelError::InvalidModelState("missing sampled head"))?;
    host_head_statistics(logits, target)
}

fn decode_batch_rows(
    action_spaces: &[&ActionSpace],
    rngs: &mut Option<&mut [PpoRng]>,
    rows: Vec<SamplingRow>,
    logits: SamplingLogits,
) -> Result<Vec<BatchSelection>, ModelError> {
    let mut selections = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let mut source = SamplingDecoder {
            batch: index,
            logits: &logits,
            rng: batch_row_rng(rngs, index),
            observed: row.observed,
            perturbed: row.perturbed,
        };
        let action = decode_from_source(action_spaces[index], &mut source)?;
        if !action_spaces[index].allows(action) {
            return Err(ModelError::InvalidModelState("illegal decoded action"));
        }
        action_spaces[index]
            .decode(action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        selections.push(BatchSelection {
            action,
            value: row.value,
            observed: source.observed,
        });
    }
    Ok(selections)
}

fn finish_sampled_row(
    space: &ActionSpace,
    selection: BatchSelection,
    statistics: bool,
) -> Result<SampledRow, ModelError> {
    let statistics = if statistics {
        let target = BehavioralTarget::from_sampled_action(space, selection.action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let (log_probability, entropy) = selection.observed.statistics(&target)?;
        Some(SampledStatistics {
            target,
            log_probability,
            entropy,
        })
    } else {
        None
    };
    Ok(SampledRow {
        action: selection.action,
        value: selection.value,
        statistics,
    })
}

/// Packs a validated frame batch into encoder rows.
fn packed_rows(frames: &[FeatureFrame]) -> Result<Vec<EncoderRow>, ModelError> {
    frames.iter().map(EncoderRow::from_frame).collect()
}

/// One sampled row: action, state value and, when requested, exact behaviour statistics.
pub(crate) struct SampledRow {
    pub(crate) action: StructuredAction,
    pub(crate) value: f32,
    pub(crate) statistics: Option<SampledStatistics>,
}

/// Behavioural target and old-policy statistics of one sampled row.
pub(crate) struct SampledStatistics {
    pub(crate) target: BehavioralTarget,
    pub(crate) log_probability: f32,
    pub(crate) entropy: f32,
}

struct SamplingDecoder<'logits, 'rng> {
    batch: usize,
    logits: &'logits SamplingLogits,
    rng: Option<&'rng mut PpoRng>,
    observed: SampledPathLogits,
    perturbed: SampledPathLogits,
}

macro_rules! sampling_decoder_head {
    ($source:ident, $name:literal, $rows:expr, $field:ident) => {
        sample_sampling_head(
            $name,
            $source.batch,
            $rows,
            $source.rng.as_deref_mut(),
            &mut $source.observed.$field,
            &mut $source.perturbed.$field,
        )
    };
}

impl DecoderSource for SamplingDecoder<'_, '_> {
    fn kind(&mut self) -> Result<[f32; 16], ModelError> {
        sampling_decoder_head!(self, "kind", Some(self.logits.base.kind.as_slice()), kind)
    }

    fn controlled(&mut self, _: ActionKind) -> Result<[f32; 2], ModelError> {
        sampling_decoder_head!(
            self,
            "controlled",
            self.logits.kind.controlled.as_deref(),
            controlled
        )
    }

    fn ability(
        &mut self,
        _: ActionKind,
        _: Option<ControlledUnit>,
    ) -> Result<[f32; 8], ModelError> {
        sampling_decoder_head!(
            self,
            "ability",
            self.logits.unit.ability.as_deref(),
            ability
        )
    }

    fn item(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 15], ModelError> {
        sampling_decoder_head!(self, "item", self.logits.unit.item.as_deref(), item)
    }

    fn swap(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 15], ModelError> {
        sampling_decoder_head!(self, "swap", self.logits.slot.swap.as_deref(), swap)
    }

    fn learn(&mut self, _: ActionKind) -> Result<[f32; 6], ModelError> {
        sampling_decoder_head!(self, "learn", self.logits.kind.learn.as_deref(), learn)
    }

    fn shop(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 64], ModelError> {
        sampling_decoder_head!(self, "shop", self.logits.unit.shop.as_deref(), shop)
    }

    fn loot(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 16], ModelError> {
        sampling_decoder_head!(self, "loot", self.logits.unit.loot.as_deref(), loot)
    }

    fn target_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: SlotSelection,
    ) -> Result<[f32; 3], ModelError> {
        sampling_decoder_head!(
            self,
            "target mode",
            self.logits.slot.target_mode.as_deref(),
            target_mode
        )
    }

    fn put_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 2], ModelError> {
        sampling_decoder_head!(
            self,
            "put mode",
            self.logits.slot.put_mode.as_deref(),
            put_mode
        )
    }

    fn entity(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; 96], ModelError> {
        let rows = if slot.is_some() {
            &self.logits.slot.entity
        } else {
            &self.logits.unit.entity
        };
        sampling_decoder_head!(self, "entity pointer", rows.as_deref(), entity_pointer)
    }

    fn point(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; MODEL_POINT_POINTER_HEAD], ModelError> {
        let rows = if slot.is_some() {
            &self.logits.slot.point
        } else {
            &self.logits.unit.point
        };
        sampling_decoder_head!(self, "point pointer", rows.as_deref(), point_pointer)
    }
}

struct ModelDecoder<'model, 'rng> {
    model: &'model PolicyModel,
    state: ForwardState,
    routing: ActorRouting,
    rng: Option<&'rng mut PpoRng>,
    observed: Option<SampledPathLogits>,
}

fn perturb_logits<const SIZE: usize>(
    mut logits: [f32; SIZE],
    rng: Option<&mut PpoRng>,
) -> Result<[f32; SIZE], ModelError> {
    let Some(rng) = rng else {
        return Ok(logits);
    };
    for logit in &mut logits {
        let uniform = rng
            .uniform_open()
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let noise = -(-uniform.ln()).ln();
        *logit += noise as f32;
        if !logit.is_finite() {
            return Err(ModelError::InvalidModelState("sampling noise"));
        }
    }
    Ok(logits)
}

fn finite_array<const SIZE: usize>(
    field: &'static str,
    values: Vec<f32>,
) -> Result<[f32; SIZE], ModelError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteOutput {
            field,
            batch: 0,
            index,
        });
    }
    values
        .try_into()
        .map_err(|_| ModelError::InvalidModelState("decoder head shape"))
}

impl ModelDecoder<'_, '_> {
    fn perturb<const SIZE: usize>(
        &mut self,
        logits: [f32; SIZE],
    ) -> Result<[f32; SIZE], ModelError> {
        perturb_logits(logits, self.rng.as_deref_mut())
    }

    fn context(
        &self,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
        slot: Option<SlotSelection>,
    ) -> Result<Tensor, ModelError> {
        let kind = self.model.kind_embedding.row(kind.index())?;
        let unit = match unit {
            Some(unit) => self.model.unit_embedding.row(unit.index())?,
            None => Tensor::zeros((1, 32), DType::F32, self.model.tensor_device())?,
        };
        let slot = match slot {
            Some(SlotSelection::Ability(index)) => self.model.ability_embedding.row(index)?,
            Some(SlotSelection::Item(index)) => self.model.item_embedding.row(index)?,
            None => Tensor::zeros((1, 16), DType::F32, self.model.tensor_device())?,
        };
        Ok(Tensor::cat(&[&self.state.trunk, &kind, &unit, &slot], 1)?)
    }

    fn head<const SIZE: usize>(
        &self,
        field: &'static str,
        head: ActorHead,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; SIZE], ModelError> {
        let values = self
            .routing
            .forward(self.model, head, &self.context(kind, unit, slot)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        finite_array(field, values)
    }

    fn pointer<const SIZE: usize>(
        &self,
        field: &'static str,
        head: ActorHead,
        tokens: &Tensor,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; SIZE], ModelError> {
        let query = self
            .routing
            .forward(self.model, head, &self.context(kind, Some(unit), slot)?)?
            .unsqueeze(1)?;
        let scores = scaled_pointer_dot(tokens, &query)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        finite_array(field, scores)
    }
}

impl DecoderSource for ModelDecoder<'_, '_> {
    fn kind(&mut self) -> Result<[f32; 16], ModelError> {
        let values = self
            .routing
            .forward(self.model, ActorHead::Kind, &self.state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let logits = finite_array("kind", values)?;
        if let Some(observed) = &mut self.observed {
            observed.kind = Some(logits);
        }
        self.perturb(logits)
    }
    fn controlled(&mut self, kind: ActionKind) -> Result<[f32; 2], ModelError> {
        let logits = self.head("controlled", ActorHead::Controlled, kind, None, None)?;
        if let Some(observed) = &mut self.observed {
            observed.controlled = Some(logits);
        }
        self.perturb(logits)
    }
    fn ability(
        &mut self,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
    ) -> Result<[f32; 8], ModelError> {
        let logits = self.head("ability", ActorHead::Ability, kind, unit, None)?;
        if let Some(observed) = &mut self.observed {
            observed.ability = Some(logits);
        }
        self.perturb(logits)
    }
    fn item(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 15], ModelError> {
        let logits = self.head("item", ActorHead::Item, kind, Some(unit), None)?;
        if let Some(observed) = &mut self.observed {
            observed.item = Some(logits);
        }
        self.perturb(logits)
    }
    fn swap(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: usize,
    ) -> Result<[f32; 15], ModelError> {
        let logits = self.head(
            "swap",
            ActorHead::Swap,
            kind,
            Some(unit),
            Some(SlotSelection::Item(slot)),
        )?;
        if let Some(observed) = &mut self.observed {
            observed.swap = Some(logits);
        }
        self.perturb(logits)
    }
    fn learn(&mut self, kind: ActionKind) -> Result<[f32; 6], ModelError> {
        let logits = self.head("learn", ActorHead::Learn, kind, None, None)?;
        if let Some(observed) = &mut self.observed {
            observed.learn = Some(logits);
        }
        self.perturb(logits)
    }
    fn shop(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 64], ModelError> {
        let logits = self.head("shop", ActorHead::Shop, kind, Some(unit), None)?;
        if let Some(observed) = &mut self.observed {
            observed.shop = Some(logits);
        }
        self.perturb(logits)
    }
    fn loot(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 16], ModelError> {
        let logits = self.head("loot", ActorHead::Loot, kind, Some(unit), None)?;
        if let Some(observed) = &mut self.observed {
            observed.loot = Some(logits);
        }
        self.perturb(logits)
    }
    fn target_mode(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: SlotSelection,
    ) -> Result<[f32; 3], ModelError> {
        let logits = self.head(
            "target mode",
            ActorHead::TargetMode,
            kind,
            Some(unit),
            Some(slot),
        )?;
        if let Some(observed) = &mut self.observed {
            observed.target_mode = Some(logits);
        }
        self.perturb(logits)
    }
    fn put_mode(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: usize,
    ) -> Result<[f32; 2], ModelError> {
        let logits = self.head(
            "put mode",
            ActorHead::PutMode,
            kind,
            Some(unit),
            Some(SlotSelection::Item(slot)),
        )?;
        if let Some(observed) = &mut self.observed {
            observed.put_mode = Some(logits);
        }
        self.perturb(logits)
    }
    fn entity(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; 96], ModelError> {
        let logits = self.pointer(
            "entity pointer",
            ActorHead::EntityQuery,
            &self.state.current_units,
            kind,
            unit,
            slot,
        )?;
        if let Some(observed) = &mut self.observed {
            observed.entity_pointer = Some(logits);
        }
        self.perturb(logits)
    }
    fn point(
        &mut self,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; MODEL_POINT_POINTER_HEAD], ModelError> {
        let logits = self.pointer(
            "point pointer",
            ActorHead::PointQuery,
            &self.state.points,
            kind,
            unit,
            slot,
        )?;
        if let Some(observed) = &mut self.observed {
            observed.point_pointer = Some(logits);
        }
        self.perturb(logits)
    }
}
