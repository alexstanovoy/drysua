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

use crate::{
    ABILITY_FEATURE_TOKENS, ABILITY_FEATURES, ActionKind, ActionSpace, ActionTarget,
    BehavioralPrediction, BehavioralTarget, ControlledUnit, EntityIndex, FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION, FeatureFrame, GLOBAL_FEATURES, HISTORY_FEATURES, HISTORY_SAMPLES,
    HeadTarget, ITEM_FEATURE_TOKENS, ITEM_FEATURES, ImitationSample, ImitationSplit,
    LOOT_FEATURE_TOKENS, LOOT_FEATURES, LootIndex, MAP_FEATURES, MAX_POLICY_HISTORY,
    OWN_UNIT_FEATURE_TOKENS, POINT_FEATURE_TOKENS, POINT_FEATURES, POLICY_HISTORY_FEATURES,
    PROJECTILE_FEATURE_TOKENS, PROJECTILE_FEATURES, PointIndex, PpoConfig, PpoMinibatchReport,
    PpoPolicyChoice, PpoPreparedSample, PpoRng, PutPointTarget, REMEMBERED_UNIT_FEATURE_TOKENS,
    ShopIndex, StructuredAction, UNIT_FEATURE_TOKENS, UNIT_FEATURES, ability_feature, item_feature,
    loot_feature, point_feature, projectile_feature, unit_feature,
};

/// Version of the fixed policy-model parameter schema.
pub const MODEL_SCHEMA_VERSION: u32 = 3;
/// Maximum frame count accepted by one public batch call.
pub const MODEL_MAX_BATCH: usize = 8_192;
/// Frame count evaluated by one bounded host inference tensor graph.
pub const MODEL_EVALUATION_MICROBATCH: usize = 64;
/// Maximum frame count in one autograd-preserving tensor forward pass.
pub const MODEL_TRAINING_BATCH: usize = 64;
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
pub const MODEL_POINT_POINTER_HEAD: usize = 48;
/// Maximum checked Adam optimizer step.
pub const MODEL_MAX_OPTIMIZER_STEP: u64 = 1_000_000_000;
/// Number of behavioral heads represented in update activity counts.
pub const MODEL_BEHAVIORAL_HEADS: usize = 12;

const UNIT_HIDDEN: usize = 64;
const UNIT_EMBEDDING: usize = 128;
const TOKEN_HIDDEN: usize = 64;
const TOKEN_EMBEDDING: usize = 64;
const UNIT_GROUPS: usize = 5;
const TRUNK_INPUT: usize = 2_568;
const TRUNK_WIDE: usize = 512;
const TRUNK_WIDTH: usize = 256;
const KIND_EMBEDDING: usize = 32;
const UNIT_SELECTION_EMBEDDING: usize = 32;
const SLOT_EMBEDDING: usize = 16;
const DECODER_CONTEXT: usize = 336;
const TARGET_MODE_HEAD: usize = 3;
const PUT_MODE_HEAD: usize = 2;
static NEXT_MODEL_LINEAGE: AtomicU64 = AtomicU64::new(1);
static NEXT_OPTIMIZER_LINEAGE: AtomicU64 = AtomicU64::new(1);

/// Canonical model shapes, parameter order, and linked feature schema.
pub const MODEL_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-model/v3;",
    "feature_schema_version=4;feature_schema_hash=508444194896722448;",
    "dtype=f32;device=cpu_stage7_intentional,accelerator_training_deferred;architecture=deepsets;activations=relu_after_every_encoder_and_trunk_linear;",
    "unit_mlp=69x64,64x128,128x128;",
    "ability_mlp=24x64,64x64;item_mlp=28x64,64x64;",
    "point_mlp=32x64,64x64;projectile_mlp=20x64,64x64;loot_mlp=16x64,64x64;",
    "unit_groups=hero,creep,structure,neutral,courier_ward;",
    "pool=token_present_and_semantic_group_mask,mean=sum_over_selected/divide_by_positive_count,max=where_selected_embedding_else_negative_infinity_then_argmax_lowest_token_tie_per_channel_then_differentiable_gather_original_embedding,one_token_receives_max_gradient,empty_mean_and_max_exact_zero,cross_group_rows_never_enter_reduction;",
    "token_pools=ability,item,point,projectile,loot;own_units=hero,courier;",
    "trunk=2568x512,512x256,256x256;",
    "embeddings=kind:16x32,unit:2x32,ability:8x16,item:15x16;",
    "heads=value:1,kind:16,unit:2,ability:8,item_source_from:15,swap_to:15,learn:6,shop:64,loot:16,target_mode:3,put_mode:2,entity_query:128,point_query:64;",
    "action_kind=0Continue,1Stop,2MovePoint,3FollowUnit,4Hold,5AttackMovePoint,6AttackUnit,7Cast,8Use,9PutPoint,10PutUnit,11Take,12Buy,13Sell,14Swap,15Learn;",
    "decoder=kind_then_optional_controlled_unit_then_family_slot_or_source_then_optional_target;branches=Continue:none,Stop:unit,MovePoint:unit_point,FollowUnit:unit_entity,Hold:unit,AttackMovePoint:unit_point,AttackUnit:unit_entity,Cast:unit_ability_target_mode_target,Use:unit_item_target_mode_target,PutPoint:unit_source_put_mode_optional_point,PutUnit:unit_source_entity,Take:unit_loot,Buy:unit_shop,Sell:unit_item,Swap:unit_from_to,Learn:ability;",
    "pointer=entity_query_dot_current_unit_embedding_in_frame_unit_order,point_query_dot_point_embedding_in_frame_point_order;target_mode=masked_argmax_None_Entity_Point_before_selected_pointer_argmax;put_mode=masked_argmax_Underfoot_Point_before_point_argmax;pointer_values_never_offset_mode_logits;",
    "head_context=controlled_and_learn_kind_prefix,ability_item_shop_loot_kind_unit_prefix,swap_target_put_and_pointer_kind_unit_slot_prefix;",
    "selection=choose_requires_private_exact_frame_action_space_lineage_revision_tick_readiness_provenance_before_tensor_work,provenance_excluded_from_tensor_and_frame_equality;mask_before_argmax,all_logits_finite_required,highest_legal_logit,lowest_stable_index_tie,no_legal_exact_error,final_action_allows_and_decode_required;",
    "nonfinite=finite_parameters_required_on_import,finite_frame_required,all_public_host_outputs_and_every_traversed_decoder_head_checked_with_batch_and_index,error_on_overflow_no_policy_choice,training_output_exposes_optional_graph_preserving_finite_validation;",
    "initialization=splitmix64_state_plus_9e3779b97f4a7c15_then_mix_bf58476d1ce4e5b9_94d049bb133111eb_top24_to_symmetric_closed_interval;linear_weight_scale=sqrt(6/fan_in),linear_bias_zero,embedding_scale=sqrt(3/columns);draw_order=unit_ability_item_point_projectile_loot_trunk_value_kind_kind_embedding_unit_embedding_ability_embedding_item_embedding_controlled_ability_head_item_head_swap_head_learn_head_shop_head_loot_head_target_mode_put_mode_entity_query_point_query;seed_is_not_input_or_parameter;",
    "batch=public_host_limit8192,evaluation_microbatch64_under_one_parameter_read_lock,training_tensor_limit64,larger_effective_training_batches_require_gradient_accumulation;",
    "runtime_identity=checked_process_local_nonzero_model_lineage_plus_monotonic_parameter_revision,one_internal_optimizer_lineage_bound_to_exact_policy_identity,raw_import_advances_revision_and_unbinds_optimizer,evidence_never_enters_tensors;",
    "updates=single_model_rwlock,all_inference_and_export_reads_hold_one_shared_lock,training_output_owns_shared_lock_for_full_forward_loss_backward_lifetime,named_backward_requires_same_model_guarded_output_and_returns62_stable_named_optional_gradient_tensors,no_unlocked_vars_exposed,parameter_import_deep_copies_originals_and_builds_and_replaces_all62_vars_under_one_exclusive_lock_with_exact_rollback_on_failure,readers_observe_complete_old_or_complete_new_parameter_set;",
    "parameter_order=unit_mlp,ability_mlp,item_mlp,point_mlp,projectile_mlp,loot_mlp,trunk,value,kind,kind_embedding,unit_embedding,ability_embedding,item_embedding,unit_head,ability_head,item_head,swap_head,learn_head,shop_head,loot_head,target_mode_head,put_mode_head,entity_query,point_query;"
);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        index += 1;
    }
    hash
}

/// Stable FNV-1a hash of [`MODEL_SCHEMA_DESCRIPTOR`].
pub const MODEL_SCHEMA_HASH: u64 = fnv1a(MODEL_SCHEMA_DESCRIPTOR.as_bytes());

const fn linear_parameters(input: usize, output: usize) -> usize {
    input * output + output
}

/// Exact number of F32 parameters in the version-one policy model.
pub const MODEL_PARAMETER_COUNT: usize = 1_684_724;

const _: () = assert!(FEATURE_SCHEMA_VERSION == 4);
const _: () = assert!(FEATURE_SCHEMA_HASH == 508_444_194_896_722_448);
const _: () = assert!(TRUNK_INPUT == 2_568);
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
    let direct =
        linear_parameters(TRUNK_WIDTH, 1) + linear_parameters(TRUNK_WIDTH, MODEL_KIND_HEAD);
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
    embeddings + direct + conditional
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

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaskedCrossEntropyTestResult {
    pub loss: f32,
    pub gradients: Vec<f32>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdamStepTestResult {
    pub parameters: Vec<f32>,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    pub unclipped_norm: f64,
    pub applied_scale: f64,
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

    /// Shape `[batch, 48]` point-candidate pointer logits.
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
        validate_tensor_finite("value", self.value())?;
        validate_tensor_finite("kind", self.kind())?;
        validate_tensor_finite("controlled", self.controlled())?;
        validate_tensor_finite("ability", self.ability())?;
        validate_tensor_finite("item", self.item())?;
        validate_tensor_finite("swap", self.swap())?;
        validate_tensor_finite("learn", self.learn())?;
        validate_tensor_finite("shop", self.shop())?;
        validate_tensor_finite("loot", self.loot())?;
        validate_tensor_finite("target mode", self.target_mode())?;
        validate_tensor_finite("put mode", self.put_mode())?;
        validate_tensor_finite("entity pointer", self.entity_pointer())?;
        validate_tensor_finite("point pointer", self.point_pointer())
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

#[cfg(test)]
pub(crate) struct PolicyTensorSnapshot {
    pub controlled: Vec<f32>,
    pub ability: Vec<f32>,
    pub target_mode: Vec<f32>,
    pub entity_pointer: Vec<f32>,
    pub point_pointer: Vec<f32>,
}

struct Linear {
    weight: Var,
    bias: Var,
}

impl Linear {
    fn fresh(input: usize, output: usize, generator: &mut Initializer) -> Result<Self, ModelError> {
        let scale = (6.0f32 / input as f32).sqrt();
        let values = (0..input * output)
            .map(|_| generator.symmetric() * scale)
            .collect::<Vec<_>>();
        let device = Device::Cpu;
        Ok(Self {
            weight: Var::from_tensor(&Tensor::from_vec(values, (input, output), &device)?)?,
            bias: Var::from_tensor(&Tensor::zeros(output, DType::F32, &device)?)?,
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
    fn fresh(shapes: &[(usize, usize)], generator: &mut Initializer) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(shapes.len());
        for &(input, output) in shapes {
            layers.push(Linear::fresh(input, output, generator)?);
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

struct Embedding {
    value: Var,
}

impl Embedding {
    fn fresh(rows: usize, columns: usize, generator: &mut Initializer) -> Result<Self, ModelError> {
        let scale = (3.0f32 / columns as f32).sqrt();
        let values = (0..rows * columns)
            .map(|_| generator.symmetric() * scale)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (rows, columns), &Device::Cpu)?;
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

/// F32 CPU DeepSets policy with an autoregressive masked decoder.
pub struct PolicyModel {
    parameter_lock: RwLock<()>,
    lineage: NonZeroU64,
    parameter_revision: AtomicU64,
    optimizer_lineage: AtomicU64,
    unit: Mlp,
    ability: Mlp,
    item: Mlp,
    point: Mlp,
    projectile: Mlp,
    loot: Mlp,
    trunk: Mlp,
    value: Linear,
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

impl PolicyModel {
    /// Constructs fixed parameters from one explicit deterministic seed.
    pub fn fresh(seed: u64) -> Result<Self, ModelError> {
        let mut generator = Initializer::new(seed);
        let unit = Mlp::fresh(
            &[(UNIT_FEATURES, 64), (64, 128), (128, 128)],
            &mut generator,
        )?;
        let ability = Mlp::fresh(&[(ABILITY_FEATURES, 64), (64, 64)], &mut generator)?;
        let item = Mlp::fresh(&[(ITEM_FEATURES, 64), (64, 64)], &mut generator)?;
        let point = Mlp::fresh(&[(POINT_FEATURES, 64), (64, 64)], &mut generator)?;
        let projectile = Mlp::fresh(&[(PROJECTILE_FEATURES, 64), (64, 64)], &mut generator)?;
        let loot = Mlp::fresh(&[(LOOT_FEATURES, 64), (64, 64)], &mut generator)?;
        Self::fresh_from_encoders(generator, unit, ability, item, point, projectile, loot)
    }

    fn fresh_from_encoders(
        mut generator: Initializer,
        unit: Mlp,
        ability: Mlp,
        item: Mlp,
        point: Mlp,
        projectile: Mlp,
        loot: Mlp,
    ) -> Result<Self, ModelError> {
        let trunk = Mlp::fresh(
            &[(TRUNK_INPUT, 512), (512, 256), (256, 256)],
            &mut generator,
        )?;
        let lineage = allocate_lineage(&NEXT_MODEL_LINEAGE, ModelError::ModelLineageUnavailable)?;
        Ok(Self {
            parameter_lock: RwLock::new(()),
            lineage,
            parameter_revision: AtomicU64::new(0),
            optimizer_lineage: AtomicU64::new(0),
            unit,
            ability,
            item,
            point,
            projectile,
            loot,
            trunk,
            value: Linear::fresh(256, 1, &mut generator)?,
            kind: Linear::fresh(256, 16, &mut generator)?,
            kind_embedding: Embedding::fresh(16, 32, &mut generator)?,
            unit_embedding: Embedding::fresh(2, 32, &mut generator)?,
            ability_embedding: Embedding::fresh(8, 16, &mut generator)?,
            item_embedding: Embedding::fresh(15, 16, &mut generator)?,
            controlled: Linear::fresh(336, 2, &mut generator)?,
            ability_head: Linear::fresh(336, 8, &mut generator)?,
            item_head: Linear::fresh(336, 15, &mut generator)?,
            swap_head: Linear::fresh(336, 15, &mut generator)?,
            learn_head: Linear::fresh(336, 6, &mut generator)?,
            shop_head: Linear::fresh(336, 64, &mut generator)?,
            loot_head: Linear::fresh(336, 16, &mut generator)?,
            target_mode: Linear::fresh(336, 3, &mut generator)?,
            put_mode: Linear::fresh(336, 2, &mut generator)?,
            entity_query: Linear::fresh(336, 128, &mut generator)?,
            point_query: Linear::fresh(336, 64, &mut generator)?,
        })
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

    pub(crate) fn with_policy_identity<T>(
        &self,
        operation: impl FnOnce(PolicyIdentity) -> T,
    ) -> Result<T, ModelError> {
        let _guard = self.read_parameter_lock()?;
        Ok(operation(self.policy_identity_locked()))
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

    #[cfg(test)]
    pub(crate) fn claim_adam_for_test(&self, config: AdamConfig) -> Result<AdamState, ModelError> {
        self.claim_optimizer(config)
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
        let state = self.forward_frames(frames)?;
        let values = self
            .value
            .forward(&state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let kinds = self.kind.forward(&state.trunk)?.to_vec2::<f32>()?;
        collect_outputs(values, kinds, batch_offset)
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
        let state = self.forward_frames(std::slice::from_ref(frame))?;
        let value = self
            .value
            .forward(&state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let mut source = ModelDecoder {
            model: self,
            state,
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

    /// Evaluates one exact legal action path without sampling or mutation.
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
                &Device::Cpu,
            )?);
            offset += count;
        }
        apply_parameter_tensors(&parameters, &tensors, &originals, fail_after)
    }

    #[cfg(test)]
    pub(crate) fn import_parameters_with_failure(
        &self,
        values: &[f32],
        fail_after: usize,
    ) -> Result<(), ModelError> {
        self.import_parameters_inner(values, Some(fail_after))
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

    #[cfg(test)]
    pub(crate) fn restore_snapshot_with_failure(
        &self,
        snapshot: &ModelAdamSnapshot,
        adam: &mut AdamState,
        expected: OptimizerBinding,
    ) -> Result<OptimizerBinding, ModelError> {
        self.restore_snapshot_inner(snapshot, adam, expected, Some(0))
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

    fn training_forward_locked(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorTensors, ModelError> {
        let state = self.forward_frames(frames)?;
        let contexts = self.training_contexts(&state.trunk, prefixes)?;
        let entity_query = self.entity_query.forward(&contexts.slot)?.unsqueeze(1)?;
        let point_query = self.point_query.forward(&contexts.slot)?.unsqueeze(1)?;
        Ok(PolicyTensorTensors {
            value: self.value.forward(&state.trunk)?,
            kind: self.kind.forward(&state.trunk)?,
            controlled: self.controlled.forward(&contexts.kind)?,
            ability: self.ability_head.forward(&contexts.unit)?,
            item: self.item_head.forward(&contexts.unit)?,
            swap: self.swap_head.forward(&contexts.slot)?,
            learn: self.learn_head.forward(&contexts.kind)?,
            shop: self.shop_head.forward(&contexts.unit)?,
            loot: self.loot_head.forward(&contexts.unit)?,
            target_mode: self.target_mode.forward(&contexts.slot)?,
            put_mode: self.put_mode.forward(&contexts.slot)?,
            entity_pointer: state.current_units.broadcast_mul(&entity_query)?.sum(2)?,
            point_pointer: state.points.broadcast_mul(&point_query)?.sum(2)?,
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

    #[cfg(test)]
    pub(crate) fn behavioral_update_with_barrier(
        &self,
        examples: &[&ImitationSample],
        adam: &mut AdamState,
        entered: &std::sync::Barrier,
        release: &std::sync::Barrier,
    ) -> Result<ModelUpdateReport, ModelError> {
        validate_behavioral_examples(examples)?;
        let _guard = self.write_parameter_lock()?;
        entered.wait();
        release.wait();
        self.behavioral_update_locked(examples, adam)
    }

    pub(crate) fn behavioral_update(
        &self,
        examples: &[&ImitationSample],
        adam: &mut AdamState,
    ) -> Result<ModelUpdateReport, ModelError> {
        validate_behavioral_examples(examples)?;
        let _guard = self.write_parameter_lock()?;
        self.behavioral_update_locked(examples, adam)
    }

    fn behavioral_update_locked(
        &self,
        examples: &[&ImitationSample],
        adam: &mut AdamState,
    ) -> Result<ModelUpdateReport, ModelError> {
        self.validate_optimizer_binding_locked(adam.binding)?;
        validate_adam_parts(
            adam.config,
            &adam.first_moment,
            &adam.second_moment,
            adam.step,
            MODEL_PARAMETER_COUNT,
        )?;
        let mut gradients = vec![0.0f32; MODEL_PARAMETER_COUNT];
        let mut loss_sum = 0.0f64;
        let mut active_head_counts = [0usize; MODEL_BEHAVIORAL_HEADS];
        for microbatch in examples.chunks(MODEL_TRAINING_BATCH) {
            let result = self.behavioral_microbatch_locked(microbatch)?;
            loss_sum += result.loss_sum;
            accumulate_gradients(&mut gradients, &result.gradients)?;
            accumulate_head_counts(&mut active_head_counts, result.active_head_counts)?;
        }
        let divisor = examples.len() as f32;
        for gradient in &mut gradients {
            *gradient /= divisor;
        }
        let average_loss = loss_sum / examples.len() as f64;
        if !average_loss.is_finite() {
            return Err(ModelError::NonFiniteLoss);
        }
        let diagnostics = self.apply_adam_locked(adam, &gradients)?;
        Ok(ModelUpdateReport {
            average_loss,
            active_head_counts,
            unclipped_norm: diagnostics.unclipped_norm,
            applied_scale: diagnostics.applied_scale,
            sample_count: examples.len(),
            optimizer_step: adam.step,
        })
    }

    pub(crate) fn ppo_update(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_update_with_microbatch(examples, adam, config, MODEL_TRAINING_BATCH)
    }

    fn ppo_update_with_microbatch(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch_size: usize,
    ) -> Result<PpoMinibatchReport, ModelError> {
        if examples.is_empty() || examples.len() > MODEL_MAX_BATCH {
            return Err(ModelError::InvalidModelState("PPO minibatch count"));
        }
        if !(1..=MODEL_TRAINING_BATCH).contains(&microbatch_size) {
            return Err(ModelError::InvalidModelState("PPO microbatch size"));
        }
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(adam.binding)?;
        let mut gradients = vec![0.0f32; MODEL_PARAMETER_COUNT];
        let mut report = PpoMinibatchReport::default();
        for microbatch in examples.chunks(microbatch_size) {
            let mut result = self.ppo_microbatch_locked(microbatch, config)?;
            scale_gradients(&mut result.gradients, microbatch.len() as f32)?;
            accumulate_gradients(&mut gradients, &result.gradients)?;
            accumulate_ppo_report(&mut report, result.report)?;
        }
        let divisor = examples.len() as f32;
        for gradient in &mut gradients {
            *gradient /= divisor;
        }
        average_ppo_report(&mut report, examples.len())?;
        if report.approximate_kl > f64::from(config.target_kl) {
            return Ok(report);
        }
        let diagnostics = self.apply_adam_locked(adam, &gradients)?;
        report.gradient_norm = diagnostics.unclipped_norm;
        report.applied_scale = diagnostics.applied_scale;
        report.applied = true;
        Ok(report)
    }

    #[cfg(test)]
    pub(crate) fn ppo_update_with_microbatch_for_test(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch_size: usize,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_update_with_microbatch(examples, adam, config, microbatch_size)
    }

    fn ppo_microbatch_locked(
        &self,
        examples: &[&PpoPreparedSample],
        config: PpoConfig,
    ) -> Result<PpoMicrobatch, ModelError> {
        let frames = examples
            .iter()
            .map(|sample| sample.transition.frame.clone())
            .collect::<Vec<_>>();
        let prefixes = examples
            .iter()
            .map(|sample| sample.transition.target.prefix())
            .collect::<Vec<_>>();
        validate_training_batch(&frames, &prefixes)?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let (loss, report) = ppo_loss(&output, examples, config)?;
        let named = self.backward_named_locked(&loss)?;
        Ok(PpoMicrobatch {
            gradients: collect_host_gradients(named)?,
            report,
        })
    }

    fn behavioral_microbatch_locked(
        &self,
        examples: &[&ImitationSample],
    ) -> Result<BehavioralMicrobatch, ModelError> {
        let frames = examples
            .iter()
            .map(|sample| sample.frame().clone())
            .collect::<Vec<_>>();
        let prefixes = examples
            .iter()
            .map(|sample| sample.target().prefix())
            .collect::<Vec<_>>();
        validate_training_batch(&frames, &prefixes)?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let (loss, active_head_counts) = behavioral_loss(&output, examples)?;
        let loss_sum = loss.to_scalar::<f32>()?;
        if !loss_sum.is_finite() {
            return Err(ModelError::NonFiniteLoss);
        }
        let named = self.backward_named_locked(&loss)?;
        let gradients = collect_host_gradients(named)?;
        Ok(BehavioralMicrobatch {
            loss_sum: f64::from(loss_sum),
            gradients,
            active_head_counts,
        })
    }

    fn apply_adam_locked(
        &self,
        adam: &mut AdamState,
        gradients: &[f32],
    ) -> Result<AdamDiagnostics, ModelError> {
        let next = self.next_policy_identity_locked()?;
        let parameters = self.export_parameters_locked()?;
        let replacement = compute_adam_step(
            &parameters,
            gradients,
            &adam.first_moment,
            &adam.second_moment,
            adam.step,
            adam.config,
        )?;
        self.import_parameters_locked(&replacement.parameters, None)?;
        adam.first_moment = replacement.first_moment;
        adam.second_moment = replacement.second_moment;
        adam.step = replacement.step;
        adam.binding.policy = next;
        self.parameter_revision
            .store(next.revision, Ordering::Relaxed);
        Ok(AdamDiagnostics {
            unclipped_norm: replacement.unclipped_norm,
            applied_scale: replacement.applied_scale,
        })
    }

    /// Returns greedy legal labels from real teacher-forced conditional tensors.
    pub fn behavioral_predictions(
        &self,
        examples: &[&ImitationSample],
    ) -> Result<Vec<BehavioralPrediction>, ModelError> {
        self.behavioral_predictions_with_identity(examples)
            .map(|(predictions, _)| predictions)
    }

    pub(crate) fn behavioral_predictions_with_identity(
        &self,
        examples: &[&ImitationSample],
    ) -> Result<(Vec<BehavioralPrediction>, PolicyIdentity), ModelError> {
        if examples.is_empty() || examples.len() > MODEL_MAX_BATCH {
            return Err(ModelError::BehavioralExampleCount {
                count: examples.len(),
                maximum: MODEL_MAX_BATCH,
            });
        }
        validate_behavioral_targets(examples)?;
        let _guard = self.read_parameter_lock()?;
        let identity = self.policy_identity_locked();
        let mut predictions = Vec::with_capacity(examples.len());
        for microbatch in examples.chunks(MODEL_TRAINING_BATCH) {
            predictions.extend(self.behavioral_prediction_microbatch_locked(microbatch)?);
        }
        Ok((predictions, identity))
    }

    fn behavioral_prediction_microbatch_locked(
        &self,
        examples: &[&ImitationSample],
    ) -> Result<Vec<BehavioralPrediction>, ModelError> {
        let frames = examples
            .iter()
            .map(|sample| sample.frame().clone())
            .collect::<Vec<_>>();
        let prefixes = examples
            .iter()
            .map(|sample| sample.target().prefix())
            .collect::<Vec<_>>();
        validate_training_batch(&frames, &prefixes)?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let logits = BehavioralHostLogits::from_tensors(&output)?;
        examples
            .iter()
            .enumerate()
            .map(|(index, sample)| logits.predict(index, sample.target()))
            .collect()
    }

    fn training_contexts(
        &self,
        trunk: &Tensor,
        prefixes: &[TrainingPrefix],
    ) -> Result<TrainingContexts, ModelError> {
        let batch = prefixes.len();
        let kind_indices = prefixes
            .iter()
            .map(|prefix| prefix.kind.index() as u32)
            .collect::<Vec<_>>();
        let kind_indices = Tensor::from_vec(kind_indices, batch, &Device::Cpu)?;
        let kind = self
            .kind_embedding
            .value
            .as_tensor()
            .index_select(&kind_indices, 0)?;
        let unit = self.training_unit_embeddings(prefixes)?;
        let slot = self.training_slot_embeddings(prefixes)?;
        let zero_unit = Tensor::zeros(unit.shape(), DType::F32, &Device::Cpu)?;
        let zero_slot = Tensor::zeros(slot.shape(), DType::F32, &Device::Cpu)?;
        Ok(TrainingContexts {
            kind: Tensor::cat(&[trunk, &kind, &zero_unit, &zero_slot], 1)?,
            unit: Tensor::cat(&[trunk, &kind, &unit, &zero_slot], 1)?,
            slot: Tensor::cat(&[trunk, &kind, &unit, &slot], 1)?,
        })
    }

    fn training_unit_embeddings(&self, prefixes: &[TrainingPrefix]) -> Result<Tensor, ModelError> {
        let indices = prefixes
            .iter()
            .map(|prefix| prefix.unit.map_or(0, ControlledUnit::index) as u32)
            .collect::<Vec<_>>();
        let presence = prefixes
            .iter()
            .map(|prefix| prefix.unit.is_some() as u8 as f32)
            .collect::<Vec<_>>();
        let indices = Tensor::from_vec(indices, prefixes.len(), &Device::Cpu)?;
        let presence = Tensor::from_vec(presence, (prefixes.len(), 1), &Device::Cpu)?;
        Ok(self
            .unit_embedding
            .value
            .as_tensor()
            .index_select(&indices, 0)?
            .broadcast_mul(&presence)?)
    }

    fn training_slot_embeddings(&self, prefixes: &[TrainingPrefix]) -> Result<Tensor, ModelError> {
        let ability = training_slot_indices(prefixes, true);
        let item = training_slot_indices(prefixes, false);
        let ability_indices = Tensor::from_vec(ability.0, prefixes.len(), &Device::Cpu)?;
        let item_indices = Tensor::from_vec(item.0, prefixes.len(), &Device::Cpu)?;
        let ability_mask = Tensor::from_vec(ability.1, (prefixes.len(), 1), &Device::Cpu)?;
        let item_mask = Tensor::from_vec(item.1, (prefixes.len(), 1), &Device::Cpu)?;
        let ability = self
            .ability_embedding
            .value
            .as_tensor()
            .index_select(&ability_indices, 0)?
            .broadcast_mul(&ability_mask)?;
        let item = self
            .item_embedding
            .value
            .as_tensor()
            .index_select(&item_indices, 0)?
            .broadcast_mul(&item_mask)?;
        Ok((ability + item)?)
    }

    fn read_parameter_lock(&self) -> Result<RwLockReadGuard<'_, ()>, ModelError> {
        self.parameter_lock
            .read()
            .map_err(|_| ModelError::ParameterLockPoisoned)
    }

    fn write_parameter_lock(&self) -> Result<RwLockWriteGuard<'_, ()>, ModelError> {
        self.parameter_lock
            .write()
            .map_err(|_| ModelError::ParameterLockPoisoned)
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

    #[cfg(test)]
    pub(crate) fn training_snapshot(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorSnapshot, ModelError> {
        let output = self.training_forward(frames, prefixes)?;
        Ok(PolicyTensorSnapshot {
            controlled: output.controlled().flatten_all()?.to_vec1()?,
            ability: output.ability().flatten_all()?.to_vec1()?,
            target_mode: output.target_mode().flatten_all()?.to_vec1()?,
            entity_pointer: output.entity_pointer().flatten_all()?.to_vec1()?,
            point_pointer: output.point_pointer().flatten_all()?.to_vec1()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn gradient_probe(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<Vec<(&'static str, bool)>, ModelError> {
        let output = self.training_forward(frames, prefixes)?;
        let loss = output.sum_all_heads()?;
        Ok(self
            .backward_named(&output, &loss)?
            .into_iter()
            .map(|gradient| (gradient.name, gradient.gradient.is_some()))
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn parameter_write_available_for_test(&self) -> bool {
        self.parameter_lock.try_write().is_ok()
    }

    fn forward_frames(&self, frames: &[FeatureFrame]) -> Result<ForwardState, ModelError> {
        let batch = frames.len();
        let units = encode_units(self, frames)?;
        let own_units = encode_own_units(self, frames)?;
        let abilities = encode_tokens(frames, TokenField::Ability, &self.ability)?;
        let items = encode_tokens(frames, TokenField::Item, &self.item)?;
        let points = encode_tokens(frames, TokenField::Point, &self.point)?;
        let projectiles = encode_tokens(frames, TokenField::Projectile, &self.projectile)?;
        let loot = encode_tokens(frames, TokenField::Loot, &self.loot)?;
        let scalars = scalar_tensor(frames)?;
        let trunk_input = Tensor::cat(
            &[
                &scalars,
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
        let mut output = Vec::with_capacity(62);
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
        output
    }

    fn decoder_parameters<'a>(&'a self, output: &mut Vec<NamedParameter<'a>>) {
        self.value
            .parameters(("value.weight", "value.bias"), output);
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

struct BehavioralMicrobatch {
    loss_sum: f64,
    gradients: Vec<f32>,
    active_head_counts: [usize; MODEL_BEHAVIORAL_HEADS],
}

struct PpoMicrobatch {
    gradients: Vec<f32>,
    report: PpoMinibatchReport,
}

struct PolicyPathStatistics {
    log_probability: f32,
    entropy: f32,
    value: f32,
}

struct AdamReplacement {
    parameters: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    step: u64,
    unclipped_norm: f64,
    applied_scale: f64,
}

struct AdamDiagnostics {
    unclipped_norm: f64,
    applied_scale: f64,
}

struct AdamCalculation {
    step: u64,
    norm: f64,
    scale: f64,
    beta1_correction: f64,
    beta2_correction: f64,
    config: AdamConfig,
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

    fn predict(
        &self,
        index: usize,
        target: &BehavioralTarget,
    ) -> Result<BehavioralPrediction, ModelError> {
        Ok(BehavioralPrediction {
            kind: select_active_head(self.row(&self.kind, index)?, &target.kind)?
                .ok_or(ModelError::InvalidModelState("inactive kind target"))?,
            controlled: select_active_head(self.row(&self.controlled, index)?, &target.controlled)?,
            ability: select_active_head(self.row(&self.ability, index)?, &target.ability)?,
            item: select_active_head(self.row(&self.item, index)?, &target.item)?,
            swap: select_active_head(self.row(&self.swap, index)?, &target.swap)?,
            learn: select_active_head(self.row(&self.learn, index)?, &target.learn)?,
            shop: select_active_head(self.row(&self.shop, index)?, &target.shop)?,
            loot: select_active_head(self.row(&self.loot, index)?, &target.loot)?,
            target_mode: select_active_head(
                self.row(&self.target_mode, index)?,
                &target.target_mode,
            )?,
            put_mode: select_active_head(self.row(&self.put_mode, index)?, &target.put_mode)?,
            entity_pointer: select_active_head(
                self.row(&self.entity_pointer, index)?,
                &target.entity_pointer,
            )?,
            point_pointer: select_active_head(
                self.row(&self.point_pointer, index)?,
                &target.point_pointer,
            )?,
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

fn select_active_head<const WIDTH: usize>(
    logits: &[f32],
    target: &HeadTarget<WIDTH>,
) -> Result<Option<usize>, ModelError> {
    if !target.active {
        return Ok(None);
    }
    Ok(Some(masked_argmax(logits, &target.mask)?))
}

fn validate_behavioral_examples(examples: &[&ImitationSample]) -> Result<(), ModelError> {
    if examples.is_empty() || examples.len() > MODEL_MAX_BATCH {
        return Err(ModelError::BehavioralExampleCount {
            count: examples.len(),
            maximum: MODEL_MAX_BATCH,
        });
    }
    if let Some((index, _)) = examples
        .iter()
        .enumerate()
        .find(|(_, sample)| sample.split() != ImitationSplit::Train)
    {
        return Err(ModelError::NonTrainingExample { index });
    }
    validate_behavioral_targets(examples)
}

fn validate_behavioral_targets(examples: &[&ImitationSample]) -> Result<(), ModelError> {
    for (index, sample) in examples.iter().enumerate() {
        if !sample.frame().is_finite() {
            return Err(ModelError::NonFiniteFrame { index });
        }
        sample
            .target()
            .validate()
            .map_err(|error| ModelError::Backend(error.to_string()))?;
    }
    Ok(())
}

fn behavioral_loss(
    output: &PolicyTensorTensors,
    examples: &[&ImitationSample],
) -> Result<(Tensor, [usize; MODEL_BEHAVIORAL_HEADS]), ModelError> {
    macro_rules! add_head {
        ($loss:ident, $tensor:expr, $name:literal, $field:ident) => {
            $loss = ($loss + masked_head_loss($tensor, examples, $name, |target| &target.$field)?)?;
        };
    }
    let mut loss = masked_head_loss(&output.kind, examples, "kind", |target| &target.kind)?;
    add_head!(loss, &output.controlled, "controlled", controlled);
    add_head!(loss, &output.ability, "ability", ability);
    add_head!(loss, &output.item, "item", item);
    add_head!(loss, &output.swap, "swap", swap);
    add_head!(loss, &output.learn, "learn", learn);
    add_head!(loss, &output.shop, "shop", shop);
    add_head!(loss, &output.loot, "loot", loot);
    add_head!(loss, &output.target_mode, "target mode", target_mode);
    add_head!(loss, &output.put_mode, "put mode", put_mode);
    add_head!(
        loss,
        &output.entity_pointer,
        "entity pointer",
        entity_pointer
    );
    add_head!(loss, &output.point_pointer, "point pointer", point_pointer);
    Ok((loss.sum_all()?, behavioral_head_counts(examples)))
}

fn ppo_loss(
    output: &PolicyTensorTensors,
    examples: &[&PpoPreparedSample],
    config: PpoConfig,
) -> Result<(Tensor, PpoMinibatchReport), ModelError> {
    let negative_log_probability = ppo_negative_log_probability(output, examples)?;
    let new_log_probability = negative_log_probability.neg()?;
    let old_log_probability = Tensor::from_vec(
        examples
            .iter()
            .map(|sample| sample.transition.old_log_probability)
            .collect::<Vec<_>>(),
        examples.len(),
        &Device::Cpu,
    )?;
    let advantages = Tensor::from_vec(
        examples
            .iter()
            .map(|sample| sample.advantage)
            .collect::<Vec<_>>(),
        examples.len(),
        &Device::Cpu,
    )?;
    let log_ratio = (&new_log_probability - &old_log_probability)?;
    let ratio = log_ratio.exp()?;
    let clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)?;
    let unclipped = ratio.mul(&advantages)?;
    let clipped = clipped_ratio.mul(&advantages)?;
    let policy_loss = unclipped.minimum(&clipped)?.mean_all()?.neg()?;
    let returns = Tensor::from_vec(
        examples
            .iter()
            .map(|sample| sample.return_value)
            .collect::<Vec<_>>(),
        examples.len(),
        &Device::Cpu,
    )?;
    let values = output.value.squeeze(1)?;
    let value_loss = (&values - &returns)?.sqr()?.mean_all()?;
    let entropy = ppo_entropy(output, examples)?.mean_all()?;
    let loss = (&policy_loss + &value_loss.affine(f64::from(config.value_coefficient), 0.0)?)?;
    let loss = (&loss - &entropy.affine(f64::from(config.entropy_coefficient), 0.0)?)?;
    let report = ppo_loss_report(
        &policy_loss,
        &value_loss,
        &entropy,
        &log_ratio,
        &ratio,
        config,
        examples.len(),
    )?;
    Ok((loss, report))
}

fn ppo_loss_report(
    policy_loss: &Tensor,
    value_loss: &Tensor,
    entropy: &Tensor,
    log_ratio: &Tensor,
    ratio: &Tensor,
    config: PpoConfig,
    samples: usize,
) -> Result<PpoMinibatchReport, ModelError> {
    let approximate_kl = (&ratio.affine(1.0, -1.0)? - log_ratio)?
        .mean_all()?
        .to_scalar::<f32>()?;
    let ratios = ratio.to_vec1::<f32>()?;
    let clipped = ratios
        .iter()
        .filter(|ratio| **ratio < 1.0 - config.clip_epsilon || **ratio > 1.0 + config.clip_epsilon)
        .count();
    Ok(PpoMinibatchReport {
        policy_loss: f64::from(policy_loss.to_scalar::<f32>()?),
        value_loss: f64::from(value_loss.to_scalar::<f32>()?),
        entropy: f64::from(entropy.to_scalar::<f32>()?),
        approximate_kl: f64::from(approximate_kl),
        clip_fraction: clipped as f64 / samples as f64,
        gradient_norm: 0.0,
        applied_scale: 0.0,
        samples,
        applied: false,
    })
}

fn ppo_negative_log_probability(
    output: &PolicyTensorTensors,
    examples: &[&PpoPreparedSample],
) -> Result<Tensor, ModelError> {
    macro_rules! add_head {
        ($loss:ident, $tensor:expr, $name:literal, $field:ident) => {
            $loss =
                ($loss + masked_ppo_head_loss($tensor, examples, $name, |target| &target.$field)?)?;
        };
    }
    let mut loss = masked_ppo_head_loss(&output.kind, examples, "kind", |target| &target.kind)?;
    add_head!(loss, &output.controlled, "controlled", controlled);
    add_head!(loss, &output.ability, "ability", ability);
    add_head!(loss, &output.item, "item", item);
    add_head!(loss, &output.swap, "swap", swap);
    add_head!(loss, &output.learn, "learn", learn);
    add_head!(loss, &output.shop, "shop", shop);
    add_head!(loss, &output.loot, "loot", loot);
    add_head!(loss, &output.target_mode, "target mode", target_mode);
    add_head!(loss, &output.put_mode, "put mode", put_mode);
    add_head!(
        loss,
        &output.entity_pointer,
        "entity pointer",
        entity_pointer
    );
    add_head!(loss, &output.point_pointer, "point pointer", point_pointer);
    Ok(loss)
}

fn masked_ppo_head_loss<const WIDTH: usize>(
    logits: &Tensor,
    examples: &[&PpoPreparedSample],
    name: &'static str,
    target: fn(&BehavioralTarget) -> &HeadTarget<WIDTH>,
) -> Result<Tensor, ModelError> {
    let mut masks = Vec::with_capacity(examples.len() * WIDTH);
    let mut labels = Vec::with_capacity(examples.len());
    let mut active = Vec::with_capacity(examples.len());
    for sample in examples {
        append_tensor_target(
            target(&sample.transition.target),
            name,
            &mut masks,
            &mut labels,
            &mut active,
        )?;
    }
    masked_loss_from_parts(logits, examples.len(), WIDTH, masks, labels, active)
}

fn masked_loss_from_parts(
    logits: &Tensor,
    batch: usize,
    width: usize,
    masks: Vec<u8>,
    labels: Vec<u32>,
    active: Vec<f32>,
) -> Result<Tensor, ModelError> {
    if logits.dims() != [batch, width] {
        return Err(ModelError::InvalidModelState("PPO head shape"));
    }
    let masks = Tensor::from_vec(masks, (batch, width), &Device::Cpu)?;
    let labels = Tensor::from_vec(labels, (batch, 1), &Device::Cpu)?;
    let active = Tensor::from_vec(active, batch, &Device::Cpu)?;
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), &Device::Cpu)?;
    let legal_logits = masks.where_cond(logits, &negative)?;
    let selected = legal_logits.gather(&labels, 1)?.squeeze(1)?;
    Ok((legal_logits.log_sum_exp(1)? - selected)?.mul(&active)?)
}

fn ppo_entropy(
    output: &PolicyTensorTensors,
    examples: &[&PpoPreparedSample],
) -> Result<Tensor, ModelError> {
    macro_rules! add_head {
        ($entropy:ident, $tensor:expr, $field:ident) => {
            $entropy =
                ($entropy + masked_ppo_head_entropy($tensor, examples, |target| &target.$field)?)?;
        };
    }
    let mut entropy = masked_ppo_head_entropy(&output.kind, examples, |target| &target.kind)?;
    add_head!(entropy, &output.controlled, controlled);
    add_head!(entropy, &output.ability, ability);
    add_head!(entropy, &output.item, item);
    add_head!(entropy, &output.swap, swap);
    add_head!(entropy, &output.learn, learn);
    add_head!(entropy, &output.shop, shop);
    add_head!(entropy, &output.loot, loot);
    add_head!(entropy, &output.target_mode, target_mode);
    add_head!(entropy, &output.put_mode, put_mode);
    add_head!(entropy, &output.entity_pointer, entity_pointer);
    add_head!(entropy, &output.point_pointer, point_pointer);
    Ok(entropy)
}

fn masked_ppo_head_entropy<const WIDTH: usize>(
    logits: &Tensor,
    examples: &[&PpoPreparedSample],
    target: fn(&BehavioralTarget) -> &HeadTarget<WIDTH>,
) -> Result<Tensor, ModelError> {
    if logits.dims() != [examples.len(), WIDTH] {
        return Err(ModelError::InvalidModelState("PPO entropy head shape"));
    }
    let mut masks = Vec::with_capacity(examples.len() * WIDTH);
    let mut active = Vec::with_capacity(examples.len());
    for sample in examples {
        let head = target(&sample.transition.target);
        masks.extend(head.mask.map(u8::from));
        active.push(f32::from(head.active));
    }
    let masks = Tensor::from_vec(masks, (examples.len(), WIDTH), &Device::Cpu)?;
    let active = Tensor::from_vec(active, examples.len(), &Device::Cpu)?;
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), &Device::Cpu)?;
    let legal = masks.where_cond(logits, &negative)?;
    let log_normalizer = legal.log_sum_exp(1)?.unsqueeze(1)?;
    let log_probability = legal.broadcast_sub(&log_normalizer)?;
    let zeros = Tensor::zeros(logits.shape(), DType::F32, &Device::Cpu)?;
    let safe_log_probability = masks.where_cond(&log_probability, &zeros)?;
    let probability = masks.where_cond(&safe_log_probability.exp()?, &zeros)?;
    let entropy = probability
        .mul(&safe_log_probability)?
        .sum(1)?
        .neg()?
        .mul(&active)?;
    Ok(entropy)
}

fn masked_head_loss<const WIDTH: usize>(
    logits: &Tensor,
    examples: &[&ImitationSample],
    name: &'static str,
    target: fn(&BehavioralTarget) -> &HeadTarget<WIDTH>,
) -> Result<Tensor, ModelError> {
    if logits.dims() != [examples.len(), WIDTH] {
        return Err(ModelError::InvalidModelState("behavioral head shape"));
    }
    let mut masks = Vec::with_capacity(examples.len() * WIDTH);
    let mut labels = Vec::with_capacity(examples.len());
    let mut active = Vec::with_capacity(examples.len());
    for sample in examples {
        let head = target(sample.target());
        append_tensor_target(head, name, &mut masks, &mut labels, &mut active)?;
    }
    let masks = Tensor::from_vec(masks, (examples.len(), WIDTH), &Device::Cpu)?;
    let labels = Tensor::from_vec(labels, (examples.len(), 1), &Device::Cpu)?;
    let active = Tensor::from_vec(active, examples.len(), &Device::Cpu)?;
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), &Device::Cpu)?;
    let legal_logits = masks.where_cond(logits, &negative)?;
    let selected = legal_logits.gather(&labels, 1)?.squeeze(1)?;
    Ok((legal_logits.log_sum_exp(1)? - selected)?.mul(&active)?)
}

fn append_tensor_target<const WIDTH: usize>(
    target: &HeadTarget<WIDTH>,
    name: &'static str,
    masks: &mut Vec<u8>,
    labels: &mut Vec<u32>,
    active: &mut Vec<f32>,
) -> Result<(), ModelError> {
    if target.active && !target.mask.get(target.selected).copied().unwrap_or(false) {
        return Err(ModelError::BehavioralTarget {
            head: name,
            label: target.selected,
        });
    }
    if target.active {
        masks.extend(target.mask.map(u8::from));
        labels.push(target.selected as u32);
        active.push(1.0);
    } else {
        masks.push(1);
        masks.resize(masks.len() + WIDTH - 1, 0);
        labels.push(0);
        active.push(0.0);
    }
    Ok(())
}

fn behavioral_head_counts(examples: &[&ImitationSample]) -> [usize; MODEL_BEHAVIORAL_HEADS] {
    let mut output = [0usize; MODEL_BEHAVIORAL_HEADS];
    for sample in examples {
        let target = sample.target();
        let active = [
            target.kind.active,
            target.controlled.active,
            target.ability.active,
            target.item.active,
            target.swap.active,
            target.learn.active,
            target.shop.active,
            target.loot.active,
            target.target_mode.active,
            target.put_mode.active,
            target.entity_pointer.active,
            target.point_pointer.active,
        ];
        for (count, active) in output.iter_mut().zip(active) {
            *count += usize::from(active);
        }
    }
    output
}

fn collect_host_gradients(named: Vec<NamedPolicyGradient>) -> Result<Vec<f32>, ModelError> {
    let mut output = Vec::with_capacity(MODEL_PARAMETER_COUNT);
    for gradient in named {
        let count = gradient.parameter_shape.iter().product::<usize>();
        if let Some(tensor) = gradient.gradient {
            let values = tensor.flatten_all()?.to_vec1::<f32>()?;
            if values.len() != count {
                return Err(ModelError::InvalidModelState("gradient shape"));
            }
            output.extend(values);
        } else {
            output.resize(output.len() + count, 0.0);
        }
    }
    if output.len() != MODEL_PARAMETER_COUNT {
        return Err(ModelError::InvalidModelState("gradient parameter count"));
    }
    validate_gradients(&output)?;
    Ok(output)
}

fn accumulate_gradients(total: &mut [f32], addition: &[f32]) -> Result<(), ModelError> {
    if total.len() != addition.len() {
        return Err(ModelError::OptimizerVectorLength {
            field: "gradient",
            actual: addition.len(),
            expected: total.len(),
        });
    }
    for (index, (total, addition)) in total.iter_mut().zip(addition).enumerate() {
        *total += addition;
        if !total.is_finite() {
            return Err(ModelError::NonFiniteGradient { index });
        }
    }
    Ok(())
}

fn scale_gradients(gradients: &mut [f32], scale: f32) -> Result<(), ModelError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ModelError::InvalidModelState("gradient accumulation scale"));
    }
    for (index, gradient) in gradients.iter_mut().enumerate() {
        *gradient *= scale;
        if !gradient.is_finite() {
            return Err(ModelError::NonFiniteGradient { index });
        }
    }
    Ok(())
}

fn accumulate_ppo_report(
    total: &mut PpoMinibatchReport,
    addition: PpoMinibatchReport,
) -> Result<(), ModelError> {
    total.policy_loss += addition.policy_loss * addition.samples as f64;
    total.value_loss += addition.value_loss * addition.samples as f64;
    total.entropy += addition.entropy * addition.samples as f64;
    total.approximate_kl += addition.approximate_kl * addition.samples as f64;
    total.clip_fraction += addition.clip_fraction * addition.samples as f64;
    total.samples = total
        .samples
        .checked_add(addition.samples)
        .ok_or(ModelError::InvalidModelState("PPO sample count"))?;
    Ok(())
}

fn average_ppo_report(
    report: &mut PpoMinibatchReport,
    expected_samples: usize,
) -> Result<(), ModelError> {
    if report.samples != expected_samples || report.samples == 0 {
        return Err(ModelError::InvalidModelState("PPO report sample count"));
    }
    let divisor = report.samples as f64;
    report.policy_loss /= divisor;
    report.value_loss /= divisor;
    report.entropy /= divisor;
    report.approximate_kl /= divisor;
    report.clip_fraction /= divisor;
    for value in [
        report.policy_loss,
        report.value_loss,
        report.entropy,
        report.approximate_kl,
        report.clip_fraction,
    ] {
        if !value.is_finite() {
            return Err(ModelError::InvalidModelState("PPO report finite"));
        }
    }
    Ok(())
}

fn accumulate_head_counts(
    total: &mut [usize; MODEL_BEHAVIORAL_HEADS],
    addition: [usize; MODEL_BEHAVIORAL_HEADS],
) -> Result<(), ModelError> {
    for (total, addition) in total.iter_mut().zip(addition) {
        *total = total
            .checked_add(addition)
            .ok_or(ModelError::InvalidModelState(
                "behavioral active-head count",
            ))?;
    }
    Ok(())
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

fn validate_gradients(gradients: &[f32]) -> Result<(), ModelError> {
    if let Some((index, _)) = gradients
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteGradient { index });
    }
    Ok(())
}

fn compute_adam_step(
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    step: u64,
    config: AdamConfig,
) -> Result<AdamReplacement, ModelError> {
    validate_adam_parts(config, first_moment, second_moment, step, parameters.len())?;
    validate_optimizer_length("gradient", gradients.len(), parameters.len())?;
    validate_gradients(gradients)?;
    if let Some((index, _)) = parameters
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ModelError::NonFiniteParameter { index });
    }
    let next_step = step
        .checked_add(1)
        .filter(|value| *value <= MODEL_MAX_OPTIMIZER_STEP)
        .ok_or(ModelError::OptimizerStepOverflow)?;
    let norm = gradient_norm(gradients)?;
    let scale = if norm > f64::from(config.gradient_clip) {
        f64::from(config.gradient_clip) / norm
    } else {
        1.0
    };
    let calculation = AdamCalculation {
        step: next_step,
        norm,
        scale,
        beta1_correction: 1.0 - f64::from(config.beta1).powi(next_step as i32),
        beta2_correction: 1.0 - f64::from(config.beta2).powi(next_step as i32),
        config,
    };
    build_adam_replacement(
        parameters,
        gradients,
        first_moment,
        second_moment,
        &calculation,
    )
}

fn gradient_norm(gradients: &[f32]) -> Result<f64, ModelError> {
    let mut squared = 0.0f64;
    for gradient in gradients {
        squared += f64::from(*gradient) * f64::from(*gradient);
        if !squared.is_finite() {
            return Err(ModelError::NonFiniteOptimizerNorm);
        }
    }
    let norm = squared.sqrt();
    if !norm.is_finite() {
        return Err(ModelError::NonFiniteOptimizerNorm);
    }
    Ok(norm)
}

fn build_adam_replacement(
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    calculation: &AdamCalculation,
) -> Result<AdamReplacement, ModelError> {
    let mut next_parameters = Vec::with_capacity(parameters.len());
    let mut next_first = Vec::with_capacity(parameters.len());
    let mut next_second = Vec::with_capacity(parameters.len());
    for index in 0..parameters.len() {
        let values = adam_scalar(
            parameters[index],
            gradients[index],
            first_moment[index],
            second_moment[index],
            calculation,
        )?;
        if !values.0.is_finite() || !values.1.is_finite() || !values.2.is_finite() {
            return Err(ModelError::NonFiniteOptimizerUpdate { index });
        }
        next_parameters.push(values.0);
        next_first.push(values.1);
        next_second.push(values.2);
    }
    Ok(AdamReplacement {
        parameters: next_parameters,
        first_moment: next_first,
        second_moment: next_second,
        step: calculation.step,
        unclipped_norm: calculation.norm,
        applied_scale: calculation.scale,
    })
}

fn adam_scalar(
    parameter: f32,
    gradient: f32,
    first: f32,
    second: f32,
    calculation: &AdamCalculation,
) -> Result<(f32, f32, f32), ModelError> {
    let config = calculation.config;
    let gradient = (f64::from(gradient) * calculation.scale) as f32;
    let first = config.beta1 * first + (1.0 - config.beta1) * gradient;
    let second_value = f64::from(config.beta2) * f64::from(second)
        + f64::from(1.0 - config.beta2) * f64::from(gradient) * f64::from(gradient);
    let second = second_value as f32;
    let first_hat = f64::from(first) / calculation.beta1_correction;
    let second_hat = f64::from(second) / calculation.beta2_correction;
    let delta = f64::from(config.learning_rate) * first_hat
        / (second_hat.sqrt() + f64::from(config.epsilon));
    let parameter = (f64::from(parameter) - delta) as f32;
    if second < 0.0 {
        return Err(ModelError::NonFiniteOptimizerUpdate { index: 0 });
    }
    Ok((parameter, first, second))
}

#[cfg(test)]
pub(crate) fn adam_step_for_test(
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    step: u64,
    config: AdamConfig,
) -> Result<AdamStepTestResult, ModelError> {
    let update = compute_adam_step(
        parameters,
        gradients,
        first_moment,
        second_moment,
        step,
        config,
    )?;
    Ok(AdamStepTestResult {
        parameters: update.parameters,
        first_moment: update.first_moment,
        second_moment: update.second_moment,
        unclipped_norm: update.unclipped_norm,
        applied_scale: update.applied_scale,
    })
}

#[cfg(test)]
pub(crate) fn masked_cross_entropy_for_test(
    logits: &[f32],
    mask: &[bool],
    selected: usize,
    active: bool,
) -> Result<MaskedCrossEntropyTestResult, ModelError> {
    if logits.len() != mask.len() || logits.is_empty() {
        return Err(ModelError::SelectionShape {
            logits: logits.len(),
            mask: mask.len(),
        });
    }
    let variable = Var::from_tensor(&Tensor::from_slice(
        logits,
        (1, logits.len()),
        &Device::Cpu,
    )?)?;
    let target = HeadTarget::<1> {
        active,
        mask: [active],
        selected: 0,
    };
    let dynamic = DynamicTestTarget {
        active: target.active,
        mask,
        selected,
    };
    let loss = dynamic_masked_loss(variable.as_tensor(), dynamic)?;
    let loss_value = loss.to_scalar::<f32>()?;
    let gradients = loss.backward()?;
    let gradient = match gradients.get(variable.as_tensor()) {
        Some(gradient) => gradient.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0.0; logits.len()],
    };
    Ok(MaskedCrossEntropyTestResult {
        loss: loss_value,
        gradients: gradient,
    })
}

#[cfg(test)]
struct DynamicTestTarget<'a> {
    active: bool,
    mask: &'a [bool],
    selected: usize,
}

#[cfg(test)]
fn dynamic_masked_loss(
    logits: &Tensor,
    target: DynamicTestTarget<'_>,
) -> Result<Tensor, ModelError> {
    if target.active && !target.mask.get(target.selected).copied().unwrap_or(false) {
        return Err(ModelError::BehavioralTarget {
            head: "test",
            label: target.selected,
        });
    }
    let mask = if target.active {
        target.mask.iter().copied().map(u8::from).collect()
    } else {
        std::iter::once(1)
            .chain(std::iter::repeat_n(0, target.mask.len() - 1))
            .collect()
    };
    let mask = Tensor::from_vec(mask, (1, target.mask.len()), &Device::Cpu)?;
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), &Device::Cpu)?;
    let legal = mask.where_cond(logits, &negative)?;
    let selected = if target.active { target.selected } else { 0 };
    let index = Tensor::from_vec(vec![selected as u32], (1, 1), &Device::Cpu)?;
    let loss = (legal.log_sum_exp(1)? - legal.gather(&index, 1)?.squeeze(1)?)?;
    Ok(loss
        .affine(if target.active { 1.0 } else { 0.0 }, 0.0)?
        .sum_all()?)
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
    validate_tensor_finite("value", &output.value)?;
    validate_tensor_finite("kind", &output.kind)?;
    validate_tensor_finite("controlled", &output.controlled)?;
    validate_tensor_finite("ability", &output.ability)?;
    validate_tensor_finite("item", &output.item)?;
    validate_tensor_finite("swap", &output.swap)?;
    validate_tensor_finite("learn", &output.learn)?;
    validate_tensor_finite("shop", &output.shop)?;
    validate_tensor_finite("loot", &output.loot)?;
    validate_tensor_finite("target mode", &output.target_mode)?;
    validate_tensor_finite("put mode", &output.put_mode)?;
    validate_tensor_finite("entity pointer", &output.entity_pointer)?;
    validate_tensor_finite("point pointer", &output.point_pointer)
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

struct ForwardState {
    trunk: Tensor,
    current_units: Tensor,
    points: Tensor,
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

fn encode_units(model: &PolicyModel, frames: &[FeatureFrame]) -> Result<UnitEncoding, ModelError> {
    let tokens = UNIT_FEATURE_TOKENS + REMEMBERED_UNIT_FEATURE_TOKENS;
    let mut values = Vec::with_capacity(frames.len() * tokens * UNIT_FEATURES);
    let mut masks = (0..UNIT_GROUPS)
        .map(|_| Vec::with_capacity(frames.len() * tokens))
        .collect::<Vec<_>>();
    for frame in frames {
        append_unit_rows(&mut values, &mut masks, &frame.units);
        append_unit_rows(&mut values, &mut masks, &frame.remembered_units);
    }
    let rows = Tensor::from_vec(values, (frames.len() * tokens, UNIT_FEATURES), &Device::Cpu)?;
    let encoded = model
        .unit
        .forward(&rows)?
        .reshape((frames.len(), tokens, UNIT_EMBEDDING))?;
    let presence = (0..frames.len() * tokens)
        .map(|index| masks.iter().any(|mask| mask[index] == 1.0) as u8 as f32)
        .collect::<Vec<_>>();
    let presence = Tensor::from_vec(presence, (frames.len(), tokens, 1), &Device::Cpu)?;
    let encoded = encoded.broadcast_mul(&presence)?;
    let pooled = pool_groups(&encoded, &masks, frames.len(), tokens, UNIT_EMBEDDING)?;
    let current = encoded.narrow(1, 0, UNIT_FEATURE_TOKENS)?;
    Ok(UnitEncoding { pooled, current })
}

fn append_unit_rows<const TOKENS: usize>(
    values: &mut Vec<f32>,
    masks: &mut [Vec<f32>],
    rows: &[[f32; UNIT_FEATURES]; TOKENS],
) {
    for row in rows {
        let present = row[unit_feature::TOKEN_PRESENT] == 1.0;
        if present {
            values.extend(row);
        } else {
            values.resize(values.len() + UNIT_FEATURES, 0.0);
        }
        let group = unit_group(row[unit_feature::KIND_TOKEN]);
        for (index, mask) in masks.iter_mut().enumerate() {
            mask.push((present && group == Some(index)) as u8 as f32);
        }
    }
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
    frames: &[FeatureFrame],
) -> Result<OwnUnitEncoding, ModelError> {
    let mut values = Vec::with_capacity(frames.len() * OWN_UNIT_FEATURE_TOKENS * UNIT_FEATURES);
    let mut mask = Vec::with_capacity(frames.len() * OWN_UNIT_FEATURE_TOKENS);
    for frame in frames {
        append_present_rows(
            &mut values,
            &mut mask,
            &frame.own_units,
            unit_feature::TOKEN_PRESENT,
        );
    }
    let rows = Tensor::from_vec(
        values,
        (frames.len() * OWN_UNIT_FEATURE_TOKENS, UNIT_FEATURES),
        &Device::Cpu,
    )?;
    let encoded = model.unit.forward(&rows)?.reshape((
        frames.len(),
        OWN_UNIT_FEATURE_TOKENS,
        UNIT_EMBEDDING,
    ))?;
    let mask = Tensor::from_vec(
        mask,
        (frames.len(), OWN_UNIT_FEATURE_TOKENS, 1),
        &Device::Cpu,
    )?;
    let fixed = encoded.broadcast_mul(&mask)?.flatten_from(1)?;
    Ok(OwnUnitEncoding { fixed })
}

enum TokenField {
    Ability,
    Item,
    Point,
    Projectile,
    Loot,
}

impl TokenField {
    const fn shape(&self) -> (usize, usize, usize) {
        match self {
            Self::Ability => (
                ABILITY_FEATURE_TOKENS,
                ABILITY_FEATURES,
                ability_feature::TOKEN_PRESENT,
            ),
            Self::Item => (
                ITEM_FEATURE_TOKENS,
                ITEM_FEATURES,
                item_feature::TOKEN_PRESENT,
            ),
            Self::Point => (
                POINT_FEATURE_TOKENS,
                POINT_FEATURES,
                point_feature::TOKEN_PRESENT,
            ),
            Self::Projectile => (
                PROJECTILE_FEATURE_TOKENS,
                PROJECTILE_FEATURES,
                projectile_feature::TOKEN_PRESENT,
            ),
            Self::Loot => (
                LOOT_FEATURE_TOKENS,
                LOOT_FEATURES,
                loot_feature::TOKEN_PRESENT,
            ),
        }
    }
}

fn encode_tokens(
    frames: &[FeatureFrame],
    field: TokenField,
    encoder: &Mlp,
) -> Result<TokenEncoding, ModelError> {
    let (tokens, features, presence) = field.shape();
    let mut values = Vec::with_capacity(frames.len() * tokens * features);
    let mut mask = Vec::with_capacity(frames.len() * tokens);
    for frame in frames {
        append_token_field(&mut values, &mut mask, frame, &field, presence);
    }
    let rows = Tensor::from_vec(values, (frames.len() * tokens, features), &Device::Cpu)?;
    let encoded = encoder
        .forward(&rows)?
        .reshape((frames.len(), tokens, TOKEN_EMBEDDING))?;
    let presence = Tensor::from_vec(mask.clone(), (frames.len(), tokens, 1), &Device::Cpu)?;
    let encoded = encoded.broadcast_mul(&presence)?;
    let pooled = pool_groups(&encoded, &[mask], frames.len(), tokens, TOKEN_EMBEDDING)?;
    Ok(TokenEncoding { pooled, encoded })
}

fn append_token_field(
    values: &mut Vec<f32>,
    mask: &mut Vec<f32>,
    frame: &FeatureFrame,
    field: &TokenField,
    presence: usize,
) {
    match field {
        TokenField::Ability => append_present_rows(values, mask, &frame.abilities, presence),
        TokenField::Item => append_present_rows(values, mask, &frame.items, presence),
        TokenField::Point => append_present_rows(values, mask, &frame.points, presence),
        TokenField::Projectile => append_present_rows(values, mask, &frame.projectiles, presence),
        TokenField::Loot => append_present_rows(values, mask, &frame.loot, presence),
    }
}

fn append_present_rows<const TOKENS: usize, const FEATURES: usize>(
    values: &mut Vec<f32>,
    mask: &mut Vec<f32>,
    rows: &[[f32; FEATURES]; TOKENS],
    presence: usize,
) {
    for row in rows {
        let present = row[presence] == 1.0;
        if present {
            values.extend(row);
        } else {
            values.resize(values.len() + FEATURES, 0.0);
        }
        mask.push(present as u8 as f32);
    }
}

fn pool_groups(
    encoded: &Tensor,
    masks: &[Vec<f32>],
    batch: usize,
    tokens: usize,
    width: usize,
) -> Result<Tensor, ModelError> {
    let mut pools = Vec::with_capacity(masks.len() * 2);
    for values in masks {
        let mask = Tensor::from_vec(values.clone(), (batch, tokens, 1), &Device::Cpu)?;
        let masked = encoded.broadcast_mul(&mask)?;
        let counts = mask.sum(1)?;
        let denominator = counts.clamp(1.0f32, tokens as f32)?;
        let mean = masked.sum(1)?.broadcast_div(&denominator)?;
        let selected = mask.eq(1.0)?.broadcast_as((batch, tokens, width))?;
        let negative_infinity =
            Tensor::full(f32::NEG_INFINITY, (batch, tokens, width), &Device::Cpu)?;
        let candidates = selected.where_cond(encoded, &negative_infinity)?;
        let indices = candidates.argmax_keepdim(1)?.contiguous()?;
        let maximum = encoded.gather(&indices, 1)?.squeeze(1)?;
        let present = counts.gt(0.0)?.broadcast_as((batch, width))?;
        let zeros = Tensor::zeros((batch, width), DType::F32, &Device::Cpu)?;
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

#[cfg(test)]
pub(crate) fn pool_groups_for_test(
    encoded: &[Vec<f32>],
    masks: &[Vec<bool>],
) -> Result<Vec<f32>, ModelError> {
    let tokens = encoded.len();
    let width = encoded.first().map_or(0, Vec::len);
    if tokens == 0 || width == 0 || encoded.iter().any(|row| row.len() != width) {
        return Err(ModelError::InvalidModelState("pool test shape"));
    }
    if masks.iter().any(|mask| mask.len() != tokens) {
        return Err(ModelError::InvalidModelState("pool test mask shape"));
    }
    let values = encoded.iter().flatten().copied().collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (1, tokens, width), &Device::Cpu)?;
    let masks = masks
        .iter()
        .map(|mask| mask.iter().map(|value| *value as u8 as f32).collect())
        .collect::<Vec<_>>();
    Ok(pool_groups(&tensor, &masks, 1, tokens, width)?
        .flatten_all()?
        .to_vec1()?)
}

#[cfg(test)]
pub(crate) fn pool_max_gradient_for_test(
    encoded: &[Vec<f32>],
    masks: &[Vec<bool>],
) -> Result<Vec<f32>, ModelError> {
    if masks.len() != 1 {
        return Err(ModelError::InvalidModelState("pool gradient group count"));
    }
    let tokens = encoded.len();
    let width = encoded.first().map_or(0, Vec::len);
    if tokens == 0 || width == 0 || encoded.iter().any(|row| row.len() != width) {
        return Err(ModelError::InvalidModelState("pool gradient shape"));
    }
    let values = encoded.iter().flatten().copied().collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (1, tokens, width), &Device::Cpu)?;
    let variable = Var::from_tensor(&tensor)?;
    let masks = masks
        .iter()
        .map(|mask| mask.iter().map(|value| *value as u8 as f32).collect())
        .collect::<Vec<_>>();
    let pooled = pool_groups(variable.as_tensor(), &masks, 1, tokens, width)?;
    let loss = pooled.narrow(1, width, width)?.sum_all()?;
    let gradients = loss.backward()?;
    let gradient = gradients
        .get(variable.as_tensor())
        .ok_or(ModelError::InvalidModelState("pool gradient"))?;
    Ok(gradient.flatten_all()?.to_vec1()?)
}

fn scalar_tensor(frames: &[FeatureFrame]) -> Result<Tensor, ModelError> {
    const SCALARS: usize = GLOBAL_FEATURES
        + HISTORY_SAMPLES * HISTORY_FEATURES
        + MAX_POLICY_HISTORY * POLICY_HISTORY_FEATURES
        + MAP_FEATURES;
    let mut values = Vec::with_capacity(frames.len() * SCALARS);
    for frame in frames {
        values.extend(frame.global);
        values.extend(frame.history.iter().flatten().copied());
        values.extend(frame.policy_history.iter().flatten().copied());
        values.extend(frame.map);
    }
    Ok(Tensor::from_vec(
        values,
        (frames.len(), SCALARS),
        &Device::Cpu,
    )?)
}

/// Fixed scripted logits used to verify the real legality decoder.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct DecoderLogits {
    pub kind: [f32; MODEL_KIND_HEAD],
    pub controlled: [f32; MODEL_UNIT_HEAD],
    pub ability: [f32; MODEL_ABILITY_HEAD],
    pub item: [f32; MODEL_ITEM_HEAD],
    pub swap: [f32; MODEL_SWAP_HEAD],
    pub learn: [f32; MODEL_LEARN_HEAD],
    pub shop: [f32; MODEL_SHOP_HEAD],
    pub loot: [f32; MODEL_LOOT_HEAD],
    pub target_mode: [f32; TARGET_MODE_HEAD],
    pub put_mode: [f32; PUT_MODE_HEAD],
    pub entity: [f32; MODEL_ENTITY_POINTER_HEAD],
    pub point: [f32; MODEL_POINT_POINTER_HEAD],
}

#[cfg(test)]
impl DecoderLogits {
    pub(crate) fn favor(kind: ActionKind) -> Self {
        let mut output = Self::default();
        output.kind[kind.index()] = 1.0;
        output
    }
}

#[cfg(test)]
impl Default for DecoderLogits {
    fn default() -> Self {
        Self {
            kind: [0.0; MODEL_KIND_HEAD],
            controlled: [0.0; MODEL_UNIT_HEAD],
            ability: [0.0; MODEL_ABILITY_HEAD],
            item: [0.0; MODEL_ITEM_HEAD],
            swap: [0.0; MODEL_SWAP_HEAD],
            learn: [0.0; MODEL_LEARN_HEAD],
            shop: [0.0; MODEL_SHOP_HEAD],
            loot: [0.0; MODEL_LOOT_HEAD],
            target_mode: [0.0; TARGET_MODE_HEAD],
            put_mode: [0.0; PUT_MODE_HEAD],
            entity: [0.0; MODEL_ENTITY_POINTER_HEAD],
            point: [0.0; MODEL_POINT_POINTER_HEAD],
        }
    }
}

#[cfg(test)]
pub(crate) fn decode_with_logits(
    space: &ActionSpace,
    logits: &DecoderLogits,
) -> Result<StructuredAction, ModelError> {
    let mut source = ScriptedDecoder(logits);
    decode_from_source(space, &mut source)
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
    ) -> Result<[f32; 48], ModelError>;
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

#[cfg(test)]
pub(crate) fn select_target_for_test(
    modes: [f32; 3],
    entities: &[f32],
    points: &[f32],
) -> Result<ActionTarget, ModelError> {
    let entity_mask = vec![true; entities.len()];
    let point_mask = vec![true; points.len()];
    match select_target_mode(&modes, true, &entity_mask, &point_mask)? {
        0 => Ok(ActionTarget::None),
        1 => Ok(ActionTarget::Entity(EntityIndex(masked_argmax(
            entities,
            &entity_mask,
        )?))),
        2 => Ok(ActionTarget::Point(PointIndex(masked_argmax(
            points,
            &point_mask,
        )?))),
        _ => Err(ModelError::InvalidModelState("target mode")),
    }
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
    let mut sources = [false; 15];
    for (index, allowed) in sources.iter_mut().enumerate() {
        *allowed = space
            .swap_destination_mask(unit, bota_proto::ItemSlot(index as u8))
            .is_some_and(|row| row.contains(&true));
    }
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

fn choose_point(mask: &[bool], scores: [f32; 48]) -> Result<PointIndex, ModelError> {
    let scores = scores.get(..mask.len()).ok_or(ModelError::SelectionShape {
        logits: 48,
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

#[cfg(test)]
struct ScriptedDecoder<'a>(&'a DecoderLogits);

#[cfg(test)]
impl DecoderSource for ScriptedDecoder<'_> {
    fn kind(&mut self) -> Result<[f32; 16], ModelError> {
        Ok(self.0.kind)
    }
    fn controlled(&mut self, _: ActionKind) -> Result<[f32; 2], ModelError> {
        Ok(self.0.controlled)
    }
    fn ability(
        &mut self,
        _: ActionKind,
        _: Option<ControlledUnit>,
    ) -> Result<[f32; 8], ModelError> {
        Ok(self.0.ability)
    }
    fn item(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 15], ModelError> {
        Ok(self.0.item)
    }
    fn swap(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 15], ModelError> {
        Ok(self.0.swap)
    }
    fn learn(&mut self, _: ActionKind) -> Result<[f32; 6], ModelError> {
        Ok(self.0.learn)
    }
    fn shop(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 64], ModelError> {
        Ok(self.0.shop)
    }
    fn loot(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 16], ModelError> {
        Ok(self.0.loot)
    }
    fn target_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: SlotSelection,
    ) -> Result<[f32; 3], ModelError> {
        Ok(self.0.target_mode)
    }
    fn put_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 2], ModelError> {
        Ok(self.0.put_mode)
    }
    fn entity(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: Option<SlotSelection>,
    ) -> Result<[f32; 96], ModelError> {
        Ok(self.0.entity)
    }
    fn point(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: Option<SlotSelection>,
    ) -> Result<[f32; 48], ModelError> {
        Ok(self.0.point)
    }
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

struct ModelDecoder<'model, 'rng> {
    model: &'model PolicyModel,
    state: ForwardState,
    rng: Option<&'rng mut PpoRng>,
    observed: Option<SampledPathLogits>,
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
        mut logits: [f32; SIZE],
    ) -> Result<[f32; SIZE], ModelError> {
        let Some(rng) = self.rng.as_deref_mut() else {
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

    fn context(
        &self,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
        slot: Option<SlotSelection>,
    ) -> Result<Tensor, ModelError> {
        let kind = self.model.kind_embedding.row(kind.index())?;
        let unit = match unit {
            Some(unit) => self.model.unit_embedding.row(unit.index())?,
            None => Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?,
        };
        let slot = match slot {
            Some(SlotSelection::Ability(index)) => self.model.ability_embedding.row(index)?,
            Some(SlotSelection::Item(index)) => self.model.item_embedding.row(index)?,
            None => Tensor::zeros((1, 16), DType::F32, &Device::Cpu)?,
        };
        Ok(Tensor::cat(&[&self.state.trunk, &kind, &unit, &slot], 1)?)
    }

    fn head<const SIZE: usize>(
        &self,
        field: &'static str,
        head: &Linear,
        kind: ActionKind,
        unit: Option<ControlledUnit>,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; SIZE], ModelError> {
        let values = head
            .forward(&self.context(kind, unit, slot)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        finite_array(field, values)
    }

    fn pointer<const SIZE: usize>(
        &self,
        field: &'static str,
        head: &Linear,
        tokens: &Tensor,
        kind: ActionKind,
        unit: ControlledUnit,
        slot: Option<SlotSelection>,
    ) -> Result<[f32; SIZE], ModelError> {
        let query = head
            .forward(&self.context(kind, Some(unit), slot)?)?
            .unsqueeze(1)?;
        let scores = tokens
            .broadcast_mul(&query)?
            .sum(2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        finite_array(field, scores)
    }
}

impl DecoderSource for ModelDecoder<'_, '_> {
    fn kind(&mut self) -> Result<[f32; 16], ModelError> {
        let values = self
            .model
            .kind
            .forward(&self.state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let logits = finite_array("kind", values)?;
        if let Some(observed) = &mut self.observed {
            observed.kind = Some(logits);
        }
        self.perturb(logits)
    }
    fn controlled(&mut self, kind: ActionKind) -> Result<[f32; 2], ModelError> {
        let logits = self.head("controlled", &self.model.controlled, kind, None, None)?;
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
        let logits = self.head("ability", &self.model.ability_head, kind, unit, None)?;
        if let Some(observed) = &mut self.observed {
            observed.ability = Some(logits);
        }
        self.perturb(logits)
    }
    fn item(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 15], ModelError> {
        let logits = self.head("item", &self.model.item_head, kind, Some(unit), None)?;
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
            &self.model.swap_head,
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
        let logits = self.head("learn", &self.model.learn_head, kind, None, None)?;
        if let Some(observed) = &mut self.observed {
            observed.learn = Some(logits);
        }
        self.perturb(logits)
    }
    fn shop(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 64], ModelError> {
        let logits = self.head("shop", &self.model.shop_head, kind, Some(unit), None)?;
        if let Some(observed) = &mut self.observed {
            observed.shop = Some(logits);
        }
        self.perturb(logits)
    }
    fn loot(&mut self, kind: ActionKind, unit: ControlledUnit) -> Result<[f32; 16], ModelError> {
        let logits = self.head("loot", &self.model.loot_head, kind, Some(unit), None)?;
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
            &self.model.target_mode,
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
            &self.model.put_mode,
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
            &self.model.entity_query,
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
    ) -> Result<[f32; 48], ModelError> {
        let logits = self.pointer(
            "point pointer",
            &self.model.point_query,
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
