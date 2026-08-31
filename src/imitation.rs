#![allow(
    clippy::float_arithmetic,
    reason = "behavioral-training metrics and orchestration use floating-point values"
)]

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::{AbilitySlot, HeroId, ItemSlot, MapId};

use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, ActionError, ActionKind, ActionSpace, ActionTarget,
    AdamConfig, AdamState, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION, FeatureFrame,
    ItemReadiness, MODEL_ABILITY_HEAD, MODEL_ENTITY_POINTER_HEAD, MODEL_ITEM_HEAD, MODEL_KIND_HEAD,
    MODEL_LEARN_HEAD, MODEL_LOOT_HEAD, MODEL_MAX_BATCH, MODEL_PARAMETER_COUNT,
    MODEL_POINT_POINTER_HEAD, MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, MODEL_SHOP_HEAD,
    MODEL_SWAP_HEAD, MODEL_UNIT_HEAD, ModelAdamSnapshot, ModelError, ModelUpdateReport,
    OrderPersistence, PolicyModel, PutPointTarget, SHADOW_FIEND, StateTracker, StructuredAction,
    Teacher, TrainingAbilitySlot, TrainingItemSlot, TrainingPrefix, TrainingSlot, global_feature,
};

/// Maximum number of owned samples retained by one imitation pool.
pub const MAX_IMITATION_SAMPLES: usize = 8_192;
/// Maximum seed count in each training, validation, or promotion namespace.
pub const MAX_SEED_NAMESPACE: usize = 8_192;
/// Maximum epoch, optimizer-step, and global-update counter value.
pub const MAX_TRAINING_COUNTER: u64 = 1_000_000_000;
/// Maximum early-stopping patience in gameplay evaluations.
pub const MAX_EARLY_STOPPING_PATIENCE: u32 = 1_000_000;
/// Minimum rollout action count accepted by the promotion gate.
pub const MIN_PROMOTION_ROLLOUT_ACTIONS: u64 = 1_000;
/// Current behavioral optimizer ownership schema.
pub const IMITATION_OPTIMIZER_VERSION: u32 = 2;
/// Current audited game-rules scope for stage-eight artifacts.
pub const IMITATION_RULES_AUDIT_VERSION: u32 = 2;

const TARGET_MODE_HEAD: usize = 3;
const PUT_MODE_HEAD: usize = 2;
static NEXT_POOL_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_SAMPLE_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// Construction, orchestration, evaluation, or checkpoint validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImitationError {
    FrameActionSpaceMismatch,
    NonFiniteFrame,
    ActionNotAllowed {
        role: &'static str,
        kind: ActionKind,
    },
    TargetInactiveMask {
        head: &'static str,
    },
    TargetEmptyMask {
        head: &'static str,
    },
    TargetLabel {
        head: &'static str,
        label: usize,
        width: usize,
    },
    TargetIllegalLabel {
        head: &'static str,
        label: usize,
    },
    TargetPathMismatch(&'static str),
    MaskOversize {
        actual: usize,
        maximum: usize,
    },
    InvalidFrameSide,
    InvalidFrameMap,
    SampleIdentityMismatch(&'static str),
    DaggerSplit,
    SeedMembership {
        namespace: &'static str,
        seed: u64,
    },
    DuplicateSampleIdentity,
    NoEvictableTrainSample,
    PoolBindingMismatch,
    HeldOutContamination,
    HeldOutSampleNotInPool,
    Capacity {
        value: usize,
        maximum: usize,
    },
    EmptyTrainSet,
    EffectiveBatch {
        value: usize,
        maximum: usize,
    },
    CounterOverflow {
        counter: &'static str,
        maximum: u64,
    },
    SeedCapacity {
        namespace: &'static str,
        count: usize,
        maximum: usize,
    },
    DuplicateSeed {
        namespace: &'static str,
        seed: u64,
    },
    SeedOverlap {
        first: &'static str,
        second: &'static str,
        seed: u64,
    },
    InvalidEarlyStopping(&'static str),
    EvaluationEpochOrder {
        epoch: u64,
        previous: u64,
    },
    NonFiniteEvaluation,
    InvalidEvaluationCounts,
    InvalidRolloutCounts,
    InvalidGameplayReport(&'static str),
    PolicyIdentityMismatch,
    InvalidTeacherCoverage,
    CheckpointSchema,
    CheckpointState(&'static str),
    Rollback {
        cause: String,
        rollback: String,
    },
    InjectedEpochFailure {
        update: usize,
    },
    Model(String),
}

impl fmt::Display for ImitationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameActionSpaceMismatch
            | Self::NonFiniteFrame
            | Self::ActionNotAllowed { .. }
            | Self::TargetInactiveMask { .. }
            | Self::TargetEmptyMask { .. }
            | Self::TargetLabel { .. }
            | Self::TargetIllegalLabel { .. }
            | Self::TargetPathMismatch(_)
            | Self::MaskOversize { .. }
            | Self::InvalidFrameSide
            | Self::InvalidFrameMap
            | Self::SampleIdentityMismatch(_)
            | Self::DaggerSplit => self.fmt_target(formatter),
            _ => self.fmt_training(formatter),
        }
    }
}

impl ImitationError {
    fn fmt_target(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameActionSpaceMismatch => formatter
                .write_str("imitation feature frame does not belong to the supplied action space"),
            Self::NonFiniteFrame => formatter.write_str("imitation feature frame is non-finite"),
            Self::ActionNotAllowed { role, kind } => write!(
                formatter,
                "imitation {role} action {kind:?} is not allowed by the supplied action space"
            ),
            Self::TargetInactiveMask { head } => write!(
                formatter,
                "imitation inactive head {head} has a nonempty legal mask"
            ),
            Self::TargetEmptyMask { head } => {
                write!(formatter, "imitation active head {head} has no legal label")
            }
            Self::TargetLabel { head, label, width } => write!(
                formatter,
                "imitation target label {label} is outside width {width} for head {head}"
            ),
            Self::TargetIllegalLabel { head, label } => write!(
                formatter,
                "imitation target label {label} is illegal for head {head}"
            ),
            Self::TargetPathMismatch(field) => {
                write!(formatter, "imitation target path does not match {field}")
            }
            Self::MaskOversize { actual, maximum } => write!(
                formatter,
                "imitation mask width {actual} exceeds head width {maximum}"
            ),
            Self::InvalidFrameSide => {
                formatter.write_str("imitation frame has invalid absolute side features")
            }
            Self::InvalidFrameMap => {
                formatter.write_str("imitation frame has invalid absolute map features")
            }
            Self::SampleIdentityMismatch(field) => write!(
                formatter,
                "imitation sample identity {field} does not match its frame or action space"
            ),
            Self::DaggerSplit => {
                formatter.write_str("imitation DAgger sample must belong to Train")
            }
            _ => self.fmt_training(formatter),
        }
    }

    fn fmt_training(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { value, maximum } => write!(
                formatter,
                "imitation pool capacity {value} is outside 1..={maximum}"
            ),
            Self::EmptyTrainSet => formatter.write_str("imitation train set is empty"),
            Self::EffectiveBatch { value, maximum } => write!(
                formatter,
                "imitation effective batch {value} is outside 1..={maximum}"
            ),
            Self::CounterOverflow { counter, maximum } => write!(
                formatter,
                "imitation {counter} counter exceeds maximum {maximum}"
            ),
            Self::SeedCapacity {
                namespace,
                count,
                maximum,
            } => write!(
                formatter,
                "imitation {namespace} seed count {count} exceeds maximum {maximum}"
            ),
            Self::DuplicateSeed { namespace, seed } => {
                write!(formatter, "imitation {namespace} seed {seed} is duplicated")
            }
            Self::SeedOverlap {
                first,
                second,
                seed,
            } => write!(
                formatter,
                "imitation seed {seed} appears in both {first} and {second} namespaces"
            ),
            Self::SeedMembership { namespace, seed } => write!(
                formatter,
                "imitation {namespace} seed {seed} is absent from its seed namespace"
            ),
            Self::DuplicateSampleIdentity => {
                formatter.write_str("imitation sample identity is duplicated")
            }
            Self::NoEvictableTrainSample => {
                formatter.write_str("imitation pool is full and has no evictable Train sample")
            }
            Self::PoolBindingMismatch => formatter.write_str(
                "imitation trainer pool lineage, revision, scope, or seeds do not match",
            ),
            Self::HeldOutContamination => formatter
                .write_str("imitation held-out evaluation received non-HeldOut or DAgger data"),
            Self::HeldOutSampleNotInPool => formatter
                .write_str("imitation held-out evaluation received a sample outside the pool"),
            _ => self.fmt_training_state(formatter),
        }
    }

    fn fmt_training_state(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEarlyStopping(field) => {
                write!(formatter, "imitation early-stopping {field} is invalid")
            }
            Self::EvaluationEpochOrder { epoch, previous } => write!(
                formatter,
                "imitation evaluation epoch {epoch} is not strictly increasing after {previous}"
            ),
            Self::NonFiniteEvaluation => {
                formatter.write_str("imitation evaluation value is non-finite")
            }
            Self::InvalidEvaluationCounts => {
                formatter.write_str("imitation evaluation counts are inconsistent")
            }
            Self::InvalidRolloutCounts => {
                formatter.write_str("imitation rollout rejection counts are inconsistent")
            }
            Self::InvalidGameplayReport(field) => {
                write!(formatter, "imitation paired gameplay {field} is invalid")
            }
            Self::PolicyIdentityMismatch => formatter
                .write_str("imitation promotion evidence policy identity does not match candidate"),
            Self::InvalidTeacherCoverage => {
                formatter.write_str("imitation teacher coverage counts are inconsistent")
            }
            Self::CheckpointSchema => {
                formatter.write_str("imitation checkpoint schema does not match this build")
            }
            Self::CheckpointState(field) => {
                write!(formatter, "imitation checkpoint has invalid {field}")
            }
            Self::Rollback { cause, rollback } => write!(
                formatter,
                "imitation epoch failed ({cause}); rollback failed ({rollback})"
            ),
            Self::InjectedEpochFailure { update } => write!(
                formatter,
                "imitation injected epoch failure before update {update}"
            ),
            Self::Model(message) => {
                write!(formatter, "imitation model operation failed: {message}")
            }
            _ => formatter.write_str("imitation error category is invalid"),
        }
    }
}

impl Error for ImitationError {}

impl From<ModelError> for ImitationError {
    fn from(error: ModelError) -> Self {
        Self::Model(error.to_string())
    }
}

/// One fixed-width legal mask and selected class for a policy head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadTarget<const WIDTH: usize> {
    /// Whether this head lies on the teacher action path.
    pub active: bool,
    /// Exact legal classes in stable model order; all false when inactive.
    pub mask: [bool; WIDTH],
    /// Teacher-selected class; zero when inactive.
    pub selected: usize,
}

impl<const WIDTH: usize> HeadTarget<WIDTH> {
    const fn inactive() -> Self {
        Self {
            active: false,
            mask: [false; WIDTH],
            selected: 0,
        }
    }

    fn active(mask: [bool; WIDTH], selected: usize) -> Self {
        Self {
            active: true,
            mask,
            selected,
        }
    }

    /// Whether the selected class is inside this active head's legal mask.
    pub fn is_selected_legal(&self) -> bool {
        self.active && self.mask.get(self.selected).copied().unwrap_or(false)
    }

    fn validate(&self, name: &'static str) -> Result<(), ImitationError> {
        if !self.active {
            if self.mask.contains(&true) {
                return Err(ImitationError::TargetInactiveMask { head: name });
            }
            return Ok(());
        }
        if !self.mask.contains(&true) {
            return Err(ImitationError::TargetEmptyMask { head: name });
        }
        if self.selected >= WIDTH {
            return Err(ImitationError::TargetLabel {
                head: name,
                label: self.selected,
                width: WIDTH,
            });
        }
        if !self.mask[self.selected] {
            return Err(ImitationError::TargetIllegalLabel {
                head: name,
                label: self.selected,
            });
        }
        Ok(())
    }
}

/// Fixed, identifier-free behavioral labels for every autoregressive policy head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BehavioralTarget {
    pub kind: HeadTarget<MODEL_KIND_HEAD>,
    pub controlled: HeadTarget<MODEL_UNIT_HEAD>,
    pub ability: HeadTarget<MODEL_ABILITY_HEAD>,
    pub item: HeadTarget<MODEL_ITEM_HEAD>,
    pub swap: HeadTarget<MODEL_SWAP_HEAD>,
    pub learn: HeadTarget<MODEL_LEARN_HEAD>,
    pub shop: HeadTarget<MODEL_SHOP_HEAD>,
    pub loot: HeadTarget<MODEL_LOOT_HEAD>,
    pub target_mode: HeadTarget<TARGET_MODE_HEAD>,
    pub put_mode: HeadTarget<PUT_MODE_HEAD>,
    pub entity_pointer: HeadTarget<MODEL_ENTITY_POINTER_HEAD>,
    pub point_pointer: HeadTarget<MODEL_POINT_POINTER_HEAD>,
    prefix: TrainingPrefix,
}

impl BehavioralTarget {
    /// Reverse-maps one exact legal teacher action into fixed head labels and masks.
    pub fn from_action(
        frame: &FeatureFrame,
        space: &ActionSpace,
        action: StructuredAction,
    ) -> Result<Self, ImitationError> {
        validate_action_input(frame, space, action, "teacher")?;
        let mut target = Self::base(space, action);
        target.map_action(space, action)?;
        target.validate()?;
        Ok(target)
    }

    fn base(space: &ActionSpace, action: StructuredAction) -> Self {
        Self {
            kind: HeadTarget::active(*space.kind_mask().as_array(), action.kind().index()),
            controlled: HeadTarget::inactive(),
            ability: HeadTarget::inactive(),
            item: HeadTarget::inactive(),
            swap: HeadTarget::inactive(),
            learn: HeadTarget::inactive(),
            shop: HeadTarget::inactive(),
            loot: HeadTarget::inactive(),
            target_mode: HeadTarget::inactive(),
            put_mode: HeadTarget::inactive(),
            entity_pointer: HeadTarget::inactive(),
            point_pointer: HeadTarget::inactive(),
            prefix: TrainingPrefix::new(action.kind(), action.controlled_unit(), None),
        }
    }

    fn map_action(
        &mut self,
        space: &ActionSpace,
        action: StructuredAction,
    ) -> Result<(), ImitationError> {
        self.map_controlled(space, action);
        match action {
            StructuredAction::Continue
            | StructuredAction::Stop { .. }
            | StructuredAction::Hold { .. } => {}
            StructuredAction::MovePoint { unit, point } => {
                self.point_pointer = pointer_target(space.move_point_mask(unit), point.0)?;
            }
            StructuredAction::FollowUnit { unit, target } => {
                self.entity_pointer = pointer_target(space.follow_entity_mask(unit), target.0)?;
            }
            StructuredAction::AttackMovePoint { unit, point } => {
                self.point_pointer = pointer_target(space.attack_move_point_mask(unit), point.0)?;
            }
            StructuredAction::AttackUnit { unit, target } => {
                self.entity_pointer = pointer_target(space.attack_entity_mask(unit), target.0)?;
            }
            StructuredAction::Cast { unit, slot, target } => {
                self.map_cast(space, unit, slot, target)?
            }
            StructuredAction::Use { unit, slot, target } => {
                self.map_use(space, unit, slot, target)?
            }
            StructuredAction::PutPoint {
                unit,
                source,
                target,
            } => self.map_put_point(space, unit, source, target)?,
            StructuredAction::PutUnit {
                unit,
                source,
                target,
            } => {
                let source_mask = std::array::from_fn(|index| {
                    space
                        .put_entity_target_mask(unit, ItemSlot(index as u8))
                        .is_some_and(|mask| mask.contains(&true))
                });
                self.item = HeadTarget::active(source_mask, usize::from(source.0));
                self.prefix = item_prefix(ActionKind::PutUnit, unit, source)?;
                self.entity_pointer = pointer_target(
                    required_mask(space.put_entity_target_mask(unit, source))?,
                    target.0,
                )?;
            }
            StructuredAction::Take { unit, loot } => {
                self.loot = pointer_target(space.take_mask(unit), loot.0)?;
            }
            StructuredAction::Buy { unit, item } => {
                self.shop = pointer_target(space.buy_mask(unit), item.0)?;
            }
            StructuredAction::Sell { unit, slot } => {
                self.item = HeadTarget::active(*space.sell_slot_mask(unit), usize::from(slot.0));
            }
            StructuredAction::Swap { unit, from, to } => self.map_swap(space, unit, from, to)?,
            StructuredAction::Learn { slot } => {
                self.learn =
                    HeadTarget::active(padded_mask(space.learn_slot_mask())?, usize::from(slot.0));
            }
        }
        Ok(())
    }

    fn map_controlled(&mut self, space: &ActionSpace, action: StructuredAction) {
        if let Some(unit) = action.controlled_unit() {
            self.controlled = HeadTarget::active(
                *space.controlled_unit_mask(action.kind()).as_array(),
                unit.index(),
            );
        }
    }

    fn map_cast(
        &mut self,
        space: &ActionSpace,
        unit: crate::ControlledUnit,
        slot: AbilitySlot,
        selected: ActionTarget,
    ) -> Result<(), ImitationError> {
        self.ability = HeadTarget::active(
            padded_mask(&space.ability_slot_mask(unit))?,
            usize::from(slot.0),
        );
        self.prefix = TrainingPrefix::new(
            ActionKind::Cast,
            Some(unit),
            Some(TrainingSlot::Ability(TrainingAbilitySlot::new(
                usize::from(slot.0),
            )?)),
        );
        self.map_target(
            required_target(space.cast_target_mask(unit, slot))?,
            selected,
        )?;
        Ok(())
    }

    fn map_use(
        &mut self,
        space: &ActionSpace,
        unit: crate::ControlledUnit,
        slot: ItemSlot,
        selected: ActionTarget,
    ) -> Result<(), ImitationError> {
        self.item = HeadTarget::active(
            padded_mask(&space.item_slot_mask(unit))?,
            usize::from(slot.0),
        );
        self.prefix = item_prefix(ActionKind::Use, unit, slot)?;
        self.map_target(
            required_target(space.use_target_mask(unit, slot))?,
            selected,
        )?;
        Ok(())
    }

    fn map_target(
        &mut self,
        mask: &crate::TargetMask,
        selected: ActionTarget,
    ) -> Result<(), ImitationError> {
        let modes = [
            mask.allows_none(),
            mask.entities().contains(&true),
            mask.points().contains(&true),
        ];
        let mode = match selected {
            ActionTarget::None => 0,
            ActionTarget::Entity(_) => 1,
            ActionTarget::Point(_) => 2,
        };
        self.target_mode = HeadTarget::active(modes, mode);
        match selected {
            ActionTarget::None => {}
            ActionTarget::Entity(index) => {
                self.entity_pointer = pointer_target(mask.entities(), index.0)?
            }
            ActionTarget::Point(index) => {
                self.point_pointer = pointer_target(mask.points(), index.0)?
            }
        }
        Ok(())
    }

    fn map_put_point(
        &mut self,
        space: &ActionSpace,
        unit: crate::ControlledUnit,
        source: ItemSlot,
        target: PutPointTarget,
    ) -> Result<(), ImitationError> {
        let underfoot = space.put_underfoot_mask(unit);
        let source_mask = std::array::from_fn(|index| {
            underfoot.get(index).copied().unwrap_or(false)
                || space
                    .put_point_target_mask(unit, ItemSlot(index as u8))
                    .is_some_and(|mask| mask.contains(&true))
        });
        self.item = HeadTarget::active(source_mask, usize::from(source.0));
        self.prefix = item_prefix(ActionKind::PutPoint, unit, source)?;
        let points = required_mask(space.put_point_target_mask(unit, source))?;
        let allows_underfoot =
            underfoot
                .get(usize::from(source.0))
                .copied()
                .ok_or(ImitationError::TargetLabel {
                    head: "item",
                    label: usize::from(source.0),
                    width: MODEL_ITEM_HEAD,
                })?;
        let modes = [allows_underfoot, points.contains(&true)];
        let mode = usize::from(matches!(target, PutPointTarget::Point(_)));
        self.put_mode = HeadTarget::active(modes, mode);
        if let PutPointTarget::Point(point) = target {
            self.point_pointer = pointer_target(points, point.0)?;
        }
        Ok(())
    }

    fn map_swap(
        &mut self,
        space: &ActionSpace,
        unit: crate::ControlledUnit,
        from: ItemSlot,
        to: ItemSlot,
    ) -> Result<(), ImitationError> {
        let source_mask = std::array::from_fn(|index| {
            space
                .swap_destination_mask(unit, ItemSlot(index as u8))
                .is_some_and(|mask| mask.contains(&true))
        });
        self.item = HeadTarget::active(source_mask, usize::from(from.0));
        self.swap = HeadTarget::active(
            *space
                .swap_destination_mask(unit, from)
                .ok_or(ImitationError::TargetEmptyMask { head: "swap" })?,
            usize::from(to.0),
        );
        self.prefix = item_prefix(ActionKind::Swap, unit, from)?;
        Ok(())
    }

    /// Validates every active and inactive head boundary.
    pub fn validate(&self) -> Result<(), ImitationError> {
        self.kind.validate("kind")?;
        self.controlled.validate("controlled")?;
        self.ability.validate("ability")?;
        self.item.validate("item")?;
        self.swap.validate("swap")?;
        self.learn.validate("learn")?;
        self.shop.validate("shop")?;
        self.loot.validate("loot")?;
        self.target_mode.validate("target mode")?;
        self.put_mode.validate("put mode")?;
        self.entity_pointer.validate("entity pointer")?;
        self.point_pointer.validate("point pointer")?;
        self.validate_path()
    }

    fn validate_path(&self) -> Result<(), ImitationError> {
        let kind = ActionKind::from_index(self.kind.selected)
            .ok_or(ImitationError::TargetPathMismatch("kind"))?;
        if !self.kind.active {
            return Err(ImitationError::TargetPathMismatch("kind active"));
        }
        if self.prefix.kind() != kind {
            return Err(ImitationError::TargetPathMismatch("prefix kind"));
        }
        let expected_unit = (kind != ActionKind::Continue && kind != ActionKind::Learn)
            .then(|| selected_unit(self.controlled.selected))
            .transpose()?;
        if self.controlled.active != expected_unit.is_some() || self.prefix.unit() != expected_unit
        {
            return Err(ImitationError::TargetPathMismatch("controlled unit"));
        }
        let expected_slot = match kind {
            ActionKind::Cast => Some(TrainingSlot::Ability(training_ability_slot(
                self.ability.selected,
            )?)),
            ActionKind::Use | ActionKind::PutPoint | ActionKind::PutUnit | ActionKind::Swap => {
                Some(TrainingSlot::Item(training_item_slot(self.item.selected)?))
            }
            _ => None,
        };
        if self.prefix.slot() != expected_slot {
            return Err(ImitationError::TargetPathMismatch("slot"));
        }
        let expected = expected_activity(self, kind);
        let actual = [
            self.controlled.active,
            self.ability.active,
            self.item.active,
            self.swap.active,
            self.learn.active,
            self.shop.active,
            self.loot.active,
            self.target_mode.active,
            self.put_mode.active,
            self.entity_pointer.active,
            self.point_pointer.active,
        ];
        if actual != expected {
            return Err(ImitationError::TargetPathMismatch("active heads"));
        }
        Ok(())
    }

    /// Exact teacher-forced context used by all conditional model tensors.
    pub const fn prefix(&self) -> TrainingPrefix {
        self.prefix
    }

    /// Reconstructs the exact structured action represented by selected labels.
    pub fn reconstruct_action(&self) -> Result<StructuredAction, ImitationError> {
        self.validate()?;
        let kind =
            ActionKind::from_index(self.kind.selected).ok_or(ImitationError::TargetLabel {
                head: "kind",
                label: self.kind.selected,
                width: MODEL_KIND_HEAD,
            })?;
        if kind == ActionKind::Continue {
            return Ok(StructuredAction::Continue);
        }
        if kind == ActionKind::Learn {
            return Ok(StructuredAction::Learn {
                slot: AbilitySlot(self.learn.selected as u8),
            });
        }
        self.reconstruct_controlled(kind)
    }

    fn reconstruct_controlled(&self, kind: ActionKind) -> Result<StructuredAction, ImitationError> {
        let unit = selected_unit(self.controlled.selected)?;
        let action = match kind {
            ActionKind::Stop => StructuredAction::Stop { unit },
            ActionKind::MovePoint => StructuredAction::MovePoint {
                unit,
                point: crate::PointIndex(self.point_pointer.selected),
            },
            ActionKind::FollowUnit => StructuredAction::FollowUnit {
                unit,
                target: crate::EntityIndex(self.entity_pointer.selected),
            },
            ActionKind::Hold => StructuredAction::Hold { unit },
            ActionKind::AttackMovePoint => StructuredAction::AttackMovePoint {
                unit,
                point: crate::PointIndex(self.point_pointer.selected),
            },
            ActionKind::AttackUnit => StructuredAction::AttackUnit {
                unit,
                target: crate::EntityIndex(self.entity_pointer.selected),
            },
            ActionKind::Cast | ActionKind::Use => return self.reconstruct_targeted(kind, unit),
            ActionKind::PutPoint | ActionKind::PutUnit => return self.reconstruct_put(kind, unit),
            ActionKind::Take => StructuredAction::Take {
                unit,
                loot: crate::LootIndex(self.loot.selected),
            },
            ActionKind::Buy => StructuredAction::Buy {
                unit,
                item: crate::ShopIndex(self.shop.selected),
            },
            ActionKind::Sell => StructuredAction::Sell {
                unit,
                slot: ItemSlot(self.item.selected as u8),
            },
            ActionKind::Swap => StructuredAction::Swap {
                unit,
                from: ItemSlot(self.item.selected as u8),
                to: ItemSlot(self.swap.selected as u8),
            },
            ActionKind::Continue | ActionKind::Learn => {
                return Err(ImitationError::TargetEmptyMask { head: "controlled" });
            }
        };
        Ok(action)
    }

    fn reconstruct_targeted(
        &self,
        kind: ActionKind,
        unit: crate::ControlledUnit,
    ) -> Result<StructuredAction, ImitationError> {
        let target = match self.target_mode.selected {
            0 => ActionTarget::None,
            1 => ActionTarget::Entity(crate::EntityIndex(self.entity_pointer.selected)),
            2 => ActionTarget::Point(crate::PointIndex(self.point_pointer.selected)),
            label => {
                return Err(ImitationError::TargetLabel {
                    head: "target mode",
                    label,
                    width: 3,
                });
            }
        };
        Ok(if kind == ActionKind::Cast {
            StructuredAction::Cast {
                unit,
                slot: AbilitySlot(self.ability.selected as u8),
                target,
            }
        } else {
            StructuredAction::Use {
                unit,
                slot: ItemSlot(self.item.selected as u8),
                target,
            }
        })
    }

    fn reconstruct_put(
        &self,
        kind: ActionKind,
        unit: crate::ControlledUnit,
    ) -> Result<StructuredAction, ImitationError> {
        let source = ItemSlot(self.item.selected as u8);
        if kind == ActionKind::PutUnit {
            return Ok(StructuredAction::PutUnit {
                unit,
                source,
                target: crate::EntityIndex(self.entity_pointer.selected),
            });
        }
        let target = match self.put_mode.selected {
            0 => PutPointTarget::Underfoot,
            1 => PutPointTarget::Point(crate::PointIndex(self.point_pointer.selected)),
            label => {
                return Err(ImitationError::TargetLabel {
                    head: "put mode",
                    label,
                    width: 2,
                });
            }
        };
        Ok(StructuredAction::PutPoint {
            unit,
            source,
            target,
        })
    }

    /// Number of heads contributing cross entropy for this target.
    pub fn active_head_count(&self) -> usize {
        [
            self.kind.active,
            self.controlled.active,
            self.ability.active,
            self.item.active,
            self.swap.active,
            self.learn.active,
            self.shop.active,
            self.loot.active,
            self.target_mode.active,
            self.put_mode.active,
            self.entity_pointer.active,
            self.point_pointer.active,
        ]
        .into_iter()
        .map(usize::from)
        .sum()
    }
}

/// Canonical player side retained only as sample metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImitationSide {
    Radiant,
    Dire,
}

/// Dataset partition with optimizer access restricted to `Train`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImitationSplit {
    Train,
    Validation,
    HeldOut,
}

/// Provenance of a teacher label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImitationSource {
    Teacher,
    Dagger,
}

/// Seed namespace encoded in immutable sample identity metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeedNamespace {
    Training,
    Validation,
    Promotion,
}

impl SeedNamespace {
    const fn split(self) -> ImitationSplit {
        match self {
            Self::Training => ImitationSplit::Train,
            Self::Validation => ImitationSplit::Validation,
            Self::Promotion => ImitationSplit::HeldOut,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Validation => "validation",
            Self::Promotion => "promotion",
        }
    }
}

/// Seed, trajectory, tick, side, and namespace identity kept outside model tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleIdentity {
    namespace: SeedNamespace,
    seed: u64,
    trajectory: u64,
    tick: u32,
    side: ImitationSide,
}

impl SampleIdentity {
    /// Builds identity metadata while deriving side from absolute frame features.
    pub fn from_frame(
        namespace: SeedNamespace,
        seed: u64,
        trajectory: u64,
        tick: u32,
        frame: &FeatureFrame,
    ) -> Result<Self, ImitationError> {
        Ok(Self {
            namespace,
            seed,
            trajectory,
            tick,
            side: frame_side(frame)?,
        })
    }

    pub const fn namespace(self) -> SeedNamespace {
        self.namespace
    }
    pub const fn seed(self) -> u64 {
        self.seed
    }
    pub const fn trajectory(self) -> u64 {
        self.trajectory
    }
    pub const fn tick(self) -> u32 {
        self.tick
    }
    pub const fn side(self) -> ImitationSide {
        self.side
    }
}

/// Fixed hero, map, and audited-rules identity for one pool and checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingScope {
    pub hero: HeroId,
    pub map: MapId,
    pub rules_audit_version: u32,
}

impl TrainingScope {
    pub fn new(map: MapId, rules_audit_version: u32) -> Result<Self, ImitationError> {
        if !matches!(map, MapId(0) | MapId(1)) {
            return Err(ImitationError::InvalidFrameMap);
        }
        if rules_audit_version != IMITATION_RULES_AUDIT_VERSION {
            return Err(ImitationError::CheckpointState("rules audit version"));
        }
        Ok(Self {
            hero: SHADOW_FIEND,
            map,
            rules_audit_version,
        })
    }
}

/// One owned frame and identifier-free teacher label with dataset metadata.
#[derive(Clone, Debug)]
pub struct ImitationSample {
    instance: NonZeroU64,
    frame: FeatureFrame,
    target: BehavioralTarget,
    teacher_action: StructuredAction,
    learner_action: Option<StructuredAction>,
    side: ImitationSide,
    split: ImitationSplit,
    source: ImitationSource,
    identity: SampleIdentity,
}

impl ImitationSample {
    /// Builds a regular teacher sample from one exact frame/action-space pair.
    pub fn teacher(
        frame: FeatureFrame,
        space: &ActionSpace,
        teacher_action: StructuredAction,
        identity: SampleIdentity,
    ) -> Result<Self, ImitationError> {
        validate_identity(&frame, space, identity)?;
        let target = BehavioralTarget::from_action(&frame, space, teacher_action)?;
        let instance = allocate_sample_instance()?;
        Ok(Self {
            instance,
            frame,
            target,
            teacher_action,
            learner_action: None,
            side: identity.side,
            split: identity.namespace.split(),
            source: ImitationSource::Teacher,
            identity,
        })
    }

    /// Builds a learner-visited sample relabeled by a teacher in the same exact space.
    pub fn dagger(
        frame: FeatureFrame,
        space: &ActionSpace,
        learner_action: StructuredAction,
        teacher_action: StructuredAction,
        identity: SampleIdentity,
    ) -> Result<Self, ImitationError> {
        validate_identity(&frame, space, identity)?;
        if identity.namespace != SeedNamespace::Training {
            return Err(ImitationError::DaggerSplit);
        }
        validate_action_input(&frame, space, learner_action, "learner")?;
        let target = BehavioralTarget::from_action(&frame, space, teacher_action)?;
        let instance = allocate_sample_instance()?;
        Ok(Self {
            instance,
            frame,
            target,
            teacher_action,
            learner_action: Some(learner_action),
            side: identity.side,
            split: ImitationSplit::Train,
            source: ImitationSource::Dagger,
            identity,
        })
    }

    pub const fn frame(&self) -> &FeatureFrame {
        &self.frame
    }
    pub const fn target(&self) -> &BehavioralTarget {
        &self.target
    }
    pub const fn teacher_action(&self) -> StructuredAction {
        self.teacher_action
    }
    pub const fn learner_action(&self) -> Option<StructuredAction> {
        self.learner_action
    }
    pub const fn side(&self) -> ImitationSide {
        self.side
    }
    pub const fn split(&self) -> ImitationSplit {
        self.split
    }
    pub const fn source(&self) -> ImitationSource {
        self.source
    }
    pub const fn identity(&self) -> SampleIdentity {
        self.identity
    }
    pub fn is_disagreement(&self) -> bool {
        self.learner_action
            .is_some_and(|action| action != self.teacher_action)
    }
}

/// Bounded source and learner-disagreement counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DaggerStatistics {
    pub teacher: usize,
    pub dagger: usize,
    pub disagreements: usize,
}

/// Exact mutable pool lineage and revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolProvenance {
    pub lineage: u64,
    pub revision: u64,
}

/// Exact pool identity bound into trainers and checkpoints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolBinding {
    instance: NonZeroU64,
    pub capacity: usize,
    pub provenance: PoolProvenance,
    pub seeds: SeedNamespaces,
    pub scope: TrainingScope,
}

/// Fixed-capacity pool with protected evaluation samples and exact seed namespaces.
pub struct ImitationPool {
    capacity: usize,
    samples: VecDeque<ImitationSample>,
    seen_identities: VecDeque<SampleIdentity>,
    binding: PoolBinding,
}

impl ImitationPool {
    pub fn new(
        capacity: usize,
        lineage: u64,
        seeds: SeedNamespaces,
        scope: TrainingScope,
    ) -> Result<Self, ImitationError> {
        if !(1..=MAX_IMITATION_SAMPLES).contains(&capacity) {
            return Err(ImitationError::Capacity {
                value: capacity,
                maximum: MAX_IMITATION_SAMPLES,
            });
        }
        validate_scope(scope)?;
        if lineage == 0 {
            return Err(ImitationError::CheckpointState("pool lineage"));
        }
        let instance = allocate_pool_instance()?;
        Ok(Self {
            capacity,
            samples: VecDeque::with_capacity(capacity),
            seen_identities: VecDeque::with_capacity(MAX_IMITATION_SAMPLES),
            binding: PoolBinding {
                instance,
                capacity,
                provenance: PoolProvenance {
                    lineage,
                    revision: 0,
                },
                seeds,
                scope,
            },
        })
    }

    /// Appends one sample and returns the whole oldest sample when full.
    pub fn push(
        &mut self,
        sample: ImitationSample,
    ) -> Result<Option<ImitationSample>, ImitationError> {
        self.validate_sample(&sample)?;
        let identity = sample.identity;
        if self.seen_identities.contains(&identity)
            || self.samples.iter().any(|held| held.identity == identity)
        {
            return Err(ImitationError::DuplicateSampleIdentity);
        }
        let next_revision = self
            .binding
            .provenance
            .revision
            .checked_add(1)
            .filter(|revision| *revision <= MAX_TRAINING_COUNTER)
            .ok_or(ImitationError::CounterOverflow {
                counter: "pool revision",
                maximum: MAX_TRAINING_COUNTER,
            })?;
        let evicted = if self.samples.len() == self.capacity {
            let index = self
                .samples
                .iter()
                .position(|held| held.split == ImitationSplit::Train)
                .ok_or(ImitationError::NoEvictableTrainSample)?;
            self.samples.remove(index)
        } else {
            None
        };
        self.samples.push_back(sample);
        if self.seen_identities.len() == MAX_IMITATION_SAMPLES {
            self.seen_identities.pop_front();
        }
        self.seen_identities.push_back(identity);
        self.binding.provenance.revision = next_revision;
        Ok(evicted)
    }

    fn validate_sample(&self, sample: &ImitationSample) -> Result<(), ImitationError> {
        let identity = sample.identity;
        if !sample.frame.is_finite() {
            return Err(ImitationError::NonFiniteFrame);
        }
        sample.target.validate()?;
        if identity.namespace.split() != sample.split {
            return Err(ImitationError::SampleIdentityMismatch("namespace"));
        }
        if !self
            .binding
            .seeds
            .contains(identity.namespace, identity.seed)
        {
            return Err(ImitationError::SeedMembership {
                namespace: identity.namespace.name(),
                seed: identity.seed,
            });
        }
        if frame_map(&sample.frame)? != self.binding.scope.map {
            return Err(ImitationError::SampleIdentityMismatch("map"));
        }
        if sample.source == ImitationSource::Dagger && sample.split != ImitationSplit::Train {
            return Err(ImitationError::DaggerSplit);
        }
        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<&ImitationSample> {
        self.samples.get(index)
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    pub const fn provenance(&self) -> PoolProvenance {
        self.binding.provenance
    }
    pub const fn binding(&self) -> &PoolBinding {
        &self.binding
    }

    #[cfg(test)]
    pub(crate) fn clear_identity_history_for_test(&mut self) {
        self.seen_identities.clear();
    }

    /// Returns shuffled indices of only Train samples without changing sample order.
    pub fn training_order(&self, shuffle: &mut ShuffleState) -> Result<Vec<usize>, ImitationError> {
        let mut order = self
            .samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| (sample.split == ImitationSplit::Train).then_some(index))
            .collect::<Vec<_>>();
        if order.is_empty() {
            return Err(ImitationError::EmptyTrainSet);
        }
        shuffle.shuffle(&mut order)?;
        Ok(order)
    }

    fn held_out(&self) -> Result<Vec<&ImitationSample>, ImitationError> {
        let output = self
            .samples
            .iter()
            .filter(|sample| sample.split == ImitationSplit::HeldOut)
            .collect::<Vec<_>>();
        if output
            .iter()
            .any(|sample| sample.source == ImitationSource::Dagger)
        {
            return Err(ImitationError::HeldOutContamination);
        }
        Ok(output)
    }

    pub fn statistics(&self) -> DaggerStatistics {
        let mut output = DaggerStatistics::default();
        for sample in &self.samples {
            match sample.source {
                ImitationSource::Teacher => output.teacher += 1,
                ImitationSource::Dagger => output.dagger += 1,
            }
            output.disagreements += usize::from(sample.is_disagreement());
        }
        output
    }
}

/// Owned SplitMix64 state used only for deterministic Fisher-Yates ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShuffleState {
    state: u64,
    draws: u64,
}

impl ShuffleState {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            draws: 0,
        }
    }
    pub const fn state(self) -> u64 {
        self.state
    }
    pub const fn draws(self) -> u64 {
        self.draws
    }

    fn next(&mut self) -> Result<u64, ImitationError> {
        self.draws = self
            .draws
            .checked_add(1)
            .filter(|draws| *draws <= MAX_TRAINING_COUNTER)
            .ok_or(ImitationError::CounterOverflow {
                counter: "shuffle draw",
                maximum: MAX_TRAINING_COUNTER,
            })?;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Ok(value ^ (value >> 31))
    }

    fn shuffle(&mut self, values: &mut [usize]) -> Result<(), ImitationError> {
        let required = values.len().saturating_sub(1) as u64;
        if self
            .draws
            .checked_add(required)
            .is_none_or(|draws| draws > MAX_TRAINING_COUNTER)
        {
            return Err(ImitationError::CounterOverflow {
                counter: "shuffle draw",
                maximum: MAX_TRAINING_COUNTER,
            });
        }
        for index in (1..values.len()).rev() {
            let selected = (self.next()? % (index as u64 + 1)) as usize;
            values.swap(index, selected);
        }
        Ok(())
    }
}

fn validate_action_input(
    frame: &FeatureFrame,
    space: &ActionSpace,
    action: StructuredAction,
    role: &'static str,
) -> Result<(), ImitationError> {
    if !frame.matches_action_space(space) {
        return Err(ImitationError::FrameActionSpaceMismatch);
    }
    if !frame.is_finite() {
        return Err(ImitationError::NonFiniteFrame);
    }
    if !space.allows(action) || space.decode(action).is_err() {
        return Err(ImitationError::ActionNotAllowed {
            role,
            kind: action.kind(),
        });
    }
    Ok(())
}

fn validate_identity(
    frame: &FeatureFrame,
    space: &ActionSpace,
    identity: SampleIdentity,
) -> Result<(), ImitationError> {
    if frame_side(frame)? != identity.side {
        return Err(ImitationError::SampleIdentityMismatch("side"));
    }
    if space.tick() != identity.tick {
        return Err(ImitationError::SampleIdentityMismatch("tick"));
    }
    Ok(())
}

fn frame_side(frame: &FeatureFrame) -> Result<ImitationSide, ImitationError> {
    let radiant = frame.global()[global_feature::SIDE_RADIANT];
    let dire = frame.global()[global_feature::SIDE_DIRE];
    match (radiant, dire) {
        (1.0, 0.0) => Ok(ImitationSide::Radiant),
        (0.0, 1.0) => Ok(ImitationSide::Dire),
        _ => Err(ImitationError::InvalidFrameSide),
    }
}

fn frame_map(frame: &FeatureFrame) -> Result<MapId, ImitationError> {
    let zero = frame.global()[global_feature::MAP_ZERO];
    let one = frame.global()[global_feature::MAP_ONE];
    match (zero, one) {
        (1.0, 0.0) => Ok(MapId(0)),
        (0.0, 1.0) => Ok(MapId(1)),
        _ => Err(ImitationError::InvalidFrameMap),
    }
}

fn selected_unit(index: usize) -> Result<crate::ControlledUnit, ImitationError> {
    match index {
        0 => Ok(crate::ControlledUnit::Hero),
        1 => Ok(crate::ControlledUnit::Courier),
        label => Err(ImitationError::TargetLabel {
            head: "controlled",
            label,
            width: 2,
        }),
    }
}

fn training_ability_slot(index: usize) -> Result<TrainingAbilitySlot, ImitationError> {
    TrainingAbilitySlot::new(index).map_err(|_| ImitationError::TargetLabel {
        head: "ability",
        label: index,
        width: MODEL_ABILITY_HEAD,
    })
}

fn training_item_slot(index: usize) -> Result<TrainingItemSlot, ImitationError> {
    TrainingItemSlot::new(index).map_err(|_| ImitationError::TargetLabel {
        head: "item",
        label: index,
        width: MODEL_ITEM_HEAD,
    })
}

fn expected_activity(target: &BehavioralTarget, kind: ActionKind) -> [bool; 11] {
    let mut expected = [false; 11];
    expected[0] = kind != ActionKind::Continue && kind != ActionKind::Learn;
    match kind {
        ActionKind::MovePoint | ActionKind::AttackMovePoint => expected[10] = true,
        ActionKind::FollowUnit | ActionKind::AttackUnit => expected[9] = true,
        ActionKind::Cast => {
            expected[1] = true;
            expected[7] = true;
            expected[9] = target.target_mode.selected == 1;
            expected[10] = target.target_mode.selected == 2;
        }
        ActionKind::Use => {
            expected[2] = true;
            expected[7] = true;
            expected[9] = target.target_mode.selected == 1;
            expected[10] = target.target_mode.selected == 2;
        }
        ActionKind::PutPoint => {
            expected[2] = true;
            expected[8] = true;
            expected[10] = target.put_mode.selected == 1;
        }
        ActionKind::PutUnit => {
            expected[2] = true;
            expected[9] = true;
        }
        ActionKind::Take => expected[6] = true,
        ActionKind::Buy => expected[5] = true,
        ActionKind::Sell => expected[2] = true,
        ActionKind::Swap => {
            expected[2] = true;
            expected[3] = true;
        }
        ActionKind::Learn => expected[4] = true,
        ActionKind::Continue | ActionKind::Stop | ActionKind::Hold => {}
    }
    expected
}

fn padded_mask<const WIDTH: usize>(mask: &[bool]) -> Result<[bool; WIDTH], ImitationError> {
    if mask.len() > WIDTH {
        return Err(ImitationError::MaskOversize {
            actual: mask.len(),
            maximum: WIDTH,
        });
    }
    let mut output = [false; WIDTH];
    output[..mask.len()].copy_from_slice(mask);
    Ok(output)
}

fn pointer_target<const WIDTH: usize>(
    mask: &[bool],
    selected: usize,
) -> Result<HeadTarget<WIDTH>, ImitationError> {
    Ok(HeadTarget::active(padded_mask(mask)?, selected))
}

#[cfg(test)]
pub(crate) fn padded_mask_for_test<const WIDTH: usize>(
    mask: &[bool],
) -> Result<[bool; WIDTH], ImitationError> {
    padded_mask(mask)
}

fn required_mask(mask: Option<&[bool]>) -> Result<&[bool], ImitationError> {
    mask.ok_or(ImitationError::TargetEmptyMask { head: "pointer" })
}

fn required_target(mask: Option<&crate::TargetMask>) -> Result<&crate::TargetMask, ImitationError> {
    mask.ok_or(ImitationError::TargetEmptyMask {
        head: "target mode",
    })
}

fn item_prefix(
    kind: ActionKind,
    unit: crate::ControlledUnit,
    slot: ItemSlot,
) -> Result<TrainingPrefix, ImitationError> {
    Ok(TrainingPrefix::new(
        kind,
        Some(unit),
        Some(TrainingSlot::Item(TrainingItemSlot::new(usize::from(
            slot.0,
        ))?)),
    ))
}

/// Greedy teacher-forced class selected independently for every active target head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BehavioralPrediction {
    pub kind: usize,
    pub controlled: Option<usize>,
    pub ability: Option<usize>,
    pub item: Option<usize>,
    pub swap: Option<usize>,
    pub learn: Option<usize>,
    pub shop: Option<usize>,
    pub loot: Option<usize>,
    pub target_mode: Option<usize>,
    pub put_mode: Option<usize>,
    pub entity_pointer: Option<usize>,
    pub point_pointer: Option<usize>,
}

/// Failure from a teacher decision or exact target construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TeacherSampleError {
    Decision(ActionError),
    Target(ImitationError),
}

impl fmt::Display for TeacherSampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decision(error) => write!(formatter, "teacher decision failed: {error}"),
            Self::Target(error) => write!(formatter, "teacher target construction failed: {error}"),
        }
    }
}

impl Error for TeacherSampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Decision(error) => Some(error),
            Self::Target(error) => Some(error),
        }
    }
}

/// Error from an identity-bound teacher collector, preserving the source error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverageError<E> {
    Capacity(ImitationError),
    Operation(E),
}

impl<E: fmt::Display> fmt::Display for CoverageError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity(error) => error.fmt(formatter),
            Self::Operation(error) => write!(formatter, "teacher operation failed: {error}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> Error for CoverageError<E> {}

/// Bounded real teacher-attempt denominator including failed attempts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TeacherCoverage {
    attempted: usize,
    represented: usize,
    failed: usize,
    attempted_identities: Vec<SampleIdentity>,
    represented_samples: Vec<(SampleIdentity, NonZeroU64)>,
    namespace: Option<SeedNamespace>,
    attempted_by_side: [usize; 2],
    represented_by_side: [usize; 2],
}

impl TeacherCoverage {
    pub const fn new() -> Self {
        Self {
            attempted: 0,
            represented: 0,
            failed: 0,
            attempted_identities: Vec::new(),
            represented_samples: Vec::new(),
            namespace: None,
            attempted_by_side: [0; 2],
            represented_by_side: [0; 2],
        }
    }

    #[cfg(test)]
    pub(crate) fn record_represented(&mut self) -> Result<(), ImitationError> {
        self.increment(true)
    }

    pub fn record_failed(&mut self) -> Result<(), ImitationError> {
        self.increment(false)
    }

    /// Records a successful teacher label for one exact sample identity.
    #[cfg(test)]
    pub(crate) fn record_represented_for(
        &mut self,
        sample: &ImitationSample,
    ) -> Result<(), ImitationError> {
        self.record_identity(sample.identity, true, Some(sample.instance))
    }

    /// Records a failed teacher attempt for one exact state identity.
    pub fn record_failed_for(&mut self, identity: SampleIdentity) -> Result<(), ImitationError> {
        self.record_identity(identity, false, None)
    }

    /// Runs one complete teacher-decision/target-construction operation as one attempt.
    #[cfg(test)]
    pub(crate) fn collect<T, E>(
        &mut self,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.ensure_capacity().map_err(CoverageError::Capacity)?;
        let result = operation();
        self.attempted += 1;
        if result.is_ok() {
            self.represented += 1;
        } else {
            self.failed += 1;
        }
        result.map_err(CoverageError::Operation)
    }

    /// Performs one identity-bound teacher attempt while retaining failures.
    #[cfg(test)]
    fn collect_for<T, E>(
        &mut self,
        identity: SampleIdentity,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.ensure_capacity().map_err(CoverageError::Capacity)?;
        self.validate_identity_metadata(identity)
            .map_err(CoverageError::Capacity)?;
        let result = operation();
        self.record_identity_after_capacity(identity, result.is_ok(), None);
        result.map_err(CoverageError::Operation)
    }

    #[cfg(test)]
    pub(crate) fn collect_for_test<T, E>(
        &mut self,
        identity: SampleIdentity,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.collect_for(identity, operation)
    }

    /// Runs `Teacher::decide` and exact target construction as one counted attempt.
    pub fn collect_teacher_sample(
        &mut self,
        identity: SampleIdentity,
        frame: FeatureFrame,
        teacher: &mut Teacher,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<ImitationSample, CoverageError<TeacherSampleError>> {
        self.ensure_capacity().map_err(CoverageError::Capacity)?;
        self.validate_identity_metadata(identity)
            .map_err(CoverageError::Capacity)?;
        let result = (|| {
            let (action, space) = teacher
                .decide(tracker, persistence, readiness)
                .map_err(TeacherSampleError::Decision)?;
            ImitationSample::teacher(frame, &space, action, identity)
                .map_err(TeacherSampleError::Target)
        })();
        let instance = result.as_ref().ok().map(|sample| sample.instance);
        self.record_identity_after_capacity(identity, result.is_ok(), instance);
        result.map_err(CoverageError::Operation)
    }

    fn increment(&mut self, represented: bool) -> Result<(), ImitationError> {
        self.ensure_capacity()?;
        self.attempted += 1;
        if represented {
            self.represented += 1;
        } else {
            self.failed += 1;
        }
        Ok(())
    }

    fn record_identity(
        &mut self,
        identity: SampleIdentity,
        represented: bool,
        instance: Option<NonZeroU64>,
    ) -> Result<(), ImitationError> {
        self.ensure_capacity()?;
        self.validate_identity_metadata(identity)?;
        self.record_identity_after_capacity(identity, represented, instance);
        Ok(())
    }

    fn validate_identity_metadata(&self, identity: SampleIdentity) -> Result<(), ImitationError> {
        if self
            .namespace
            .is_some_and(|namespace| namespace != identity.namespace)
        {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        if self.attempted_identities.contains(&identity) {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        Ok(())
    }

    fn record_identity_after_capacity(
        &mut self,
        identity: SampleIdentity,
        represented: bool,
        instance: Option<NonZeroU64>,
    ) {
        self.attempted += 1;
        self.namespace = self.namespace.or(Some(identity.namespace));
        self.attempted_identities.push(identity);
        let side = identity.side.index();
        self.attempted_by_side[side] += 1;
        if represented {
            self.represented += 1;
            self.represented_by_side[side] += 1;
            if let Some(instance) = instance {
                self.represented_samples.push((identity, instance));
            }
        } else {
            self.failed += 1;
        }
    }

    fn ensure_capacity(&self) -> Result<(), ImitationError> {
        if self.attempted >= MAX_IMITATION_SAMPLES {
            return Err(ImitationError::CounterOverflow {
                counter: "teacher coverage attempt",
                maximum: MAX_IMITATION_SAMPLES as u64,
            });
        }
        Ok(())
    }

    pub const fn attempted(&self) -> usize {
        self.attempted
    }
    pub const fn represented(&self) -> usize {
        self.represented
    }
    pub const fn failed(&self) -> usize {
        self.failed
    }
    pub fn ratio(&self) -> Option<f64> {
        (self.attempted != 0).then(|| self.represented as f64 / self.attempted as f64)
    }

    fn validate(&self) -> Result<(), ImitationError> {
        let attempted_by_side = self.attempted_by_side[0].checked_add(self.attempted_by_side[1]);
        let represented_by_side =
            self.represented_by_side[0].checked_add(self.represented_by_side[1]);
        if self.attempted > MAX_IMITATION_SAMPLES
            || self.represented.checked_add(self.failed) != Some(self.attempted)
            || self.attempted_identities.len() != self.attempted
            || self.represented_samples.len() != self.represented
            || attempted_by_side != Some(self.attempted)
            || represented_by_side != Some(self.represented)
        {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        let represented_identities = self
            .represented_samples
            .iter()
            .map(|(identity, _)| *identity)
            .collect::<Vec<_>>();
        if !coverage_identities_valid(&self.attempted_identities, &represented_identities) {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        Ok(())
    }

    fn matches_samples(&self, samples: &[&ImitationSample]) -> bool {
        if self.namespace != Some(SeedNamespace::Promotion) {
            return false;
        }
        let mut expected = samples
            .iter()
            .map(|sample| (sample.identity, sample.instance))
            .collect::<Vec<_>>();
        let mut represented = self.represented_samples.clone();
        expected.sort_unstable();
        represented.sort_unstable();
        expected == represented
    }

    fn side_counts(&self, side: ImitationSide) -> (usize, usize) {
        let index = side.index();
        (
            self.represented_by_side[index],
            self.attempted_by_side[index],
        )
    }
}

fn coverage_identities_valid(attempted: &[SampleIdentity], represented: &[SampleIdentity]) -> bool {
    let mut attempted = attempted.to_vec();
    let mut represented = represented.to_vec();
    attempted.sort_unstable();
    represented.sort_unstable();
    let unique = !attempted.windows(2).any(|pair| pair[0] == pair[1])
        && !represented.windows(2).any(|pair| pair[0] == pair[1]);
    unique
        && represented
            .iter()
            .all(|identity| attempted.binary_search(identity).is_ok())
}

impl ImitationSide {
    const fn index(self) -> usize {
        match self {
            Self::Radiant => 0,
            Self::Dire => 1,
        }
    }
}

/// Matching and total count with an explicit absent ratio for an empty denominator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgreementCount {
    pub matching: usize,
    pub total: usize,
}

impl AgreementCount {
    pub fn agreement(self) -> Option<f64> {
        (self.total != 0).then(|| self.matching as f64 / self.total as f64)
    }
}

/// Offline behavioral metrics for one side selection or the complete sample set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluationAggregate {
    pub samples: usize,
    pub kind: AgreementCount,
    pub full: AgreementCount,
    pub teacher_covered: usize,
    pub teacher_attempted: usize,
    pub families: [AgreementCount; ActionKind::COUNT],
    pub action_distribution: [usize; ActionKind::COUNT],
}

impl EvaluationAggregate {
    pub fn kind_agreement(&self) -> Option<f64> {
        self.kind.agreement()
    }
    pub fn full_agreement(&self) -> Option<f64> {
        self.full.agreement()
    }
    pub fn teacher_coverage(&self) -> Option<f64> {
        (self.teacher_attempted != 0)
            .then(|| self.teacher_covered as f64 / self.teacher_attempted as f64)
    }
    pub fn continue_ratio(&self) -> Option<f64> {
        (self.samples != 0).then(|| {
            self.action_distribution[ActionKind::Continue.index()] as f64 / self.samples as f64
        })
    }

    fn note(&mut self, sample: &ImitationSample, prediction: BehavioralPrediction) {
        let target = sample.target();
        let kind_matches = prediction.kind == target.kind.selected;
        let full_matches = kind_matches && prediction_matches(target, prediction);
        self.samples += 1;
        self.kind.total += 1;
        self.kind.matching += usize::from(kind_matches);
        self.full.total += 1;
        self.full.matching += usize::from(full_matches);
        let family = sample.teacher_action.kind().index();
        self.families[family].total += 1;
        self.families[family].matching += usize::from(full_matches);
        self.action_distribution[family] += 1;
    }
}

/// Overall, Radiant, and Dire offline teacher-forced behavioral evaluation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OfflineEvaluation {
    pub overall: EvaluationAggregate,
    pub radiant: EvaluationAggregate,
    pub dire: EvaluationAggregate,
}

/// Typed promotion-set evaluation tied to one exact pool revision and real coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldOutEvaluation {
    metrics: OfflineEvaluation,
    coverage: TeacherCoverage,
    pool: PoolBinding,
    candidate: crate::PolicyIdentity,
}

impl HeldOutEvaluation {
    pub const fn metrics(&self) -> &OfflineEvaluation {
        &self.metrics
    }
    pub const fn coverage(&self) -> &TeacherCoverage {
        &self.coverage
    }
    pub const fn pool(&self) -> &PoolBinding {
        &self.pool
    }
    pub const fn candidate(&self) -> crate::PolicyIdentity {
        self.candidate
    }
}

impl OfflineEvaluation {
    /// Evaluates only promotion/HeldOut samples from one exact pool revision.
    pub fn evaluate_held_out(
        model: &PolicyModel,
        pool: &ImitationPool,
        coverage: TeacherCoverage,
    ) -> Result<HeldOutEvaluation, ImitationError> {
        let samples = pool.held_out()?;
        Self::evaluate_held_out_samples(model, pool, &samples, coverage)
    }

    /// Evaluates a caller-selected set only when every sample is clean HeldOut data.
    pub fn evaluate_held_out_samples(
        model: &PolicyModel,
        pool: &ImitationPool,
        samples: &[&ImitationSample],
        coverage: TeacherCoverage,
    ) -> Result<HeldOutEvaluation, ImitationError> {
        if samples.len() > MODEL_MAX_BATCH {
            return Err(ImitationError::EffectiveBatch {
                value: samples.len(),
                maximum: MODEL_MAX_BATCH,
            });
        }
        for sample in samples {
            if sample.split != ImitationSplit::HeldOut || sample.source == ImitationSource::Dagger {
                return Err(ImitationError::HeldOutContamination);
            }
            if !pool.samples.iter().any(|held| std::ptr::eq(held, *sample)) {
                return Err(ImitationError::HeldOutSampleNotInPool);
            }
        }
        let (metrics, candidate) = Self::evaluate_samples(model, samples, &coverage)?;
        Ok(HeldOutEvaluation {
            metrics,
            coverage,
            pool: pool.binding.clone(),
            candidate,
        })
    }

    fn evaluate_samples(
        model: &PolicyModel,
        samples: &[&ImitationSample],
        coverage: &TeacherCoverage,
    ) -> Result<(Self, crate::PolicyIdentity), ImitationError> {
        coverage.validate()?;
        if samples.len() > MODEL_MAX_BATCH {
            return Err(ImitationError::EffectiveBatch {
                value: samples.len(),
                maximum: MODEL_MAX_BATCH,
            });
        }
        if coverage.represented != samples.len() {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        if !coverage.matches_samples(samples) {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        for sample in samples {
            sample.target.validate()?;
        }
        let (predictions, candidate) = model.behavioral_predictions_with_identity(samples)?;
        if predictions.len() != samples.len() {
            return Err(ImitationError::CheckpointState("prediction count"));
        }
        let mut output = Self::default();
        for (sample, prediction) in samples.iter().zip(predictions) {
            if sample.split != ImitationSplit::HeldOut || sample.source == ImitationSource::Dagger {
                return Err(ImitationError::HeldOutContamination);
            }
            output.overall.note(sample, prediction);
            match sample.side {
                ImitationSide::Radiant => output.radiant.note(sample, prediction),
                ImitationSide::Dire => output.dire.note(sample, prediction),
            }
        }
        output.overall.teacher_covered = coverage.represented;
        output.overall.teacher_attempted = coverage.attempted;
        set_side_coverage(&mut output.radiant, coverage, ImitationSide::Radiant);
        set_side_coverage(&mut output.dire, coverage, ImitationSide::Dire);
        Ok((output, candidate))
    }
}

fn set_side_coverage(
    aggregate: &mut EvaluationAggregate,
    coverage: &TeacherCoverage,
    side: ImitationSide,
) {
    if coverage
        .attempted_by_side
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
        != Some(coverage.attempted)
    {
        aggregate.teacher_covered = 0;
        aggregate.teacher_attempted = 0;
        return;
    }
    let (represented, attempted) = coverage.side_counts(side);
    aggregate.teacher_covered = represented;
    aggregate.teacher_attempted = attempted;
}

/// Learner outcome for one learner-versus-teacher match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearnerMatchOutcome {
    Win,
    Loss,
    Draw,
}

/// One finite learner-versus-teacher outcome and score pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnerTeacherResult {
    outcome: LearnerMatchOutcome,
    learner_score: f64,
    teacher_score: f64,
}

impl LearnerTeacherResult {
    pub fn new(
        outcome: LearnerMatchOutcome,
        learner_score: f64,
        teacher_score: f64,
    ) -> Result<Self, ImitationError> {
        if !learner_score.is_finite() || !teacher_score.is_finite() {
            return Err(ImitationError::NonFiniteEvaluation);
        }
        Ok(Self {
            outcome,
            learner_score,
            teacher_score,
        })
    }

    pub const fn outcome(self) -> LearnerMatchOutcome {
        self.outcome
    }
    pub const fn learner_score(self) -> f64 {
        self.learner_score
    }
    pub const fn teacher_score(self) -> f64 {
        self.teacher_score
    }
}

/// One seed played once with the learner as Radiant and once as Dire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedSeedResult {
    seed: u64,
    radiant: LearnerTeacherResult,
    dire: LearnerTeacherResult,
}

impl PairedSeedResult {
    pub const fn new(seed: u64, radiant: LearnerTeacherResult, dire: LearnerTeacherResult) -> Self {
        Self {
            seed,
            radiant,
            dire,
        }
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }
    pub const fn radiant(self) -> LearnerTeacherResult {
        self.radiant
    }
    pub const fn dire(self) -> LearnerTeacherResult {
        self.dire
    }
}

/// Aggregate derived only from structural per-seed results for one side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideMatchReport {
    games: u32,
    wins: u32,
    losses: u32,
    draws: u32,
    learner_score: f64,
    teacher_score: f64,
}

impl SideMatchReport {
    const fn empty() -> Self {
        Self {
            games: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            learner_score: 0.0,
            teacher_score: 0.0,
        }
    }

    fn note(&mut self, result: LearnerTeacherResult) -> Result<(), ImitationError> {
        self.games += 1;
        match result.outcome {
            LearnerMatchOutcome::Win => self.wins += 1,
            LearnerMatchOutcome::Loss => self.losses += 1,
            LearnerMatchOutcome::Draw => self.draws += 1,
        }
        self.learner_score += result.learner_score;
        self.teacher_score += result.teacher_score;
        if !self.learner_score.is_finite() || !self.teacher_score.is_finite() {
            return Err(ImitationError::NonFiniteEvaluation);
        }
        Ok(())
    }

    pub const fn games(self) -> u32 {
        self.games
    }
    pub const fn wins(self) -> u32 {
        self.wins
    }
    pub const fn losses(self) -> u32 {
        self.losses
    }
    pub const fn draws(self) -> u32 {
        self.draws
    }
    pub const fn learner_score(self) -> f64 {
        self.learner_score
    }
    pub const fn teacher_score(self) -> f64 {
        self.teacher_score
    }
    pub fn learner_win_rate(self) -> f64 {
        f64::from(self.wins) / f64::from(self.games)
    }
}

/// Bounded structural paired matches for one exact candidate policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedGameplayReport {
    candidate: crate::PolicyIdentity,
    results: Vec<PairedSeedResult>,
    paired_seeds: Vec<u64>,
    radiant: SideMatchReport,
    dire: SideMatchReport,
}

impl PairedGameplayReport {
    pub fn new(
        candidate: crate::PolicyIdentity,
        mut results: Vec<PairedSeedResult>,
    ) -> Result<Self, ImitationError> {
        if results.is_empty() || results.len() > MAX_SEED_NAMESPACE {
            return Err(ImitationError::InvalidGameplayReport("paired seed count"));
        }
        results.sort_unstable_by_key(|result| result.seed);
        if results.windows(2).any(|pair| pair[0].seed >= pair[1].seed) {
            return Err(ImitationError::InvalidGameplayReport(
                "paired seed identity",
            ));
        }
        let mut radiant = SideMatchReport::empty();
        let mut dire = SideMatchReport::empty();
        for result in &results {
            radiant.note(result.radiant)?;
            dire.note(result.dire)?;
        }
        let paired_seeds = results.iter().map(|result| result.seed).collect();
        Ok(Self {
            candidate,
            results,
            paired_seeds,
            radiant,
            dire,
        })
    }

    pub const fn candidate(&self) -> crate::PolicyIdentity {
        self.candidate
    }
    pub fn results(&self) -> &[PairedSeedResult] {
        &self.results
    }
    pub fn paired_seeds(&self) -> &[u64] {
        &self.paired_seeds
    }
    pub const fn radiant(&self) -> SideMatchReport {
        self.radiant
    }
    pub const fn dire(&self) -> SideMatchReport {
        self.dire
    }

    fn validate(&self) -> Result<(), ImitationError> {
        let rebuilt = Self::new(self.candidate, self.results.clone())?;
        if &rebuilt != self {
            return Err(ImitationError::InvalidGameplayReport(
                "derived paired aggregates",
            ));
        }
        Ok(())
    }

    fn learner_not_worse(&self) -> bool {
        let learner = self.radiant.learner_score + self.dire.learner_score;
        let teacher = self.radiant.teacher_score + self.dire.teacher_score;
        let radiant_has_score =
            self.radiant.learner_score != 0.0 || self.radiant.teacher_score != 0.0;
        let dire_has_score = self.dire.learner_score != 0.0 || self.dire.teacher_score != 0.0;
        radiant_has_score
            && dire_has_score
            && learner.is_finite()
            && teacher.is_finite()
            && self.radiant.learner_score >= self.radiant.teacher_score
            && self.dire.learner_score >= self.dire.teacher_score
            && learner >= teacher
    }
}

/// Bounded rollout rejection counts and mandatory safety audits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutAudit {
    candidate: crate::PolicyIdentity,
    rejected_actions: u64,
    total_actions: u64,
    side_audit_passed: bool,
    exploit_audit_passed: bool,
}

impl RolloutAudit {
    pub fn new(
        candidate: crate::PolicyIdentity,
        rejected_actions: u64,
        total_actions: u64,
        side_audit_passed: bool,
        exploit_audit_passed: bool,
    ) -> Result<Self, ImitationError> {
        if total_actions > MAX_TRAINING_COUNTER
            || rejected_actions > MAX_TRAINING_COUNTER
            || rejected_actions > total_actions
        {
            return Err(ImitationError::InvalidRolloutCounts);
        }
        Ok(Self {
            candidate,
            rejected_actions,
            total_actions,
            side_audit_passed,
            exploit_audit_passed,
        })
    }

    pub const fn candidate(self) -> crate::PolicyIdentity {
        self.candidate
    }
    pub const fn rejected_actions(self) -> u64 {
        self.rejected_actions
    }
    pub const fn total_actions(self) -> u64 {
        self.total_actions
    }
}

/// Typed held-out, paired-gameplay, and rollout inputs to the promotion gate.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionGateInput {
    pub held_out: HeldOutEvaluation,
    pub rollout: RolloutAudit,
    pub gameplay: PairedGameplayReport,
}

/// Individual promotion requirements and their conjunction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionGateResult {
    pub full_agreement: bool,
    pub teacher_coverage: bool,
    pub rejection_rate: bool,
    pub gameplay_score: bool,
    pub both_sides: bool,
    pub side_audit: bool,
    pub exploit_audit: bool,
    pub nontrivial_action_distribution: bool,
    pub radiant_win_rate: f64,
    pub dire_win_rate: f64,
    pub passed: bool,
}

impl PromotionGateInput {
    pub fn evaluate(&self, model: &PolicyModel) -> Result<PromotionGateResult, ImitationError> {
        model.with_policy_identity(|candidate| self.evaluate_candidate(candidate))?
    }

    fn evaluate_candidate(
        &self,
        candidate: crate::PolicyIdentity,
    ) -> Result<PromotionGateResult, ImitationError> {
        self.validate_inputs(candidate)?;
        let metrics = &self.held_out.metrics;
        let full_agreement = metrics
            .overall
            .full_agreement()
            .is_some_and(|value| value >= 0.95);
        let teacher_coverage = self.held_out.coverage.ratio() == Some(1.0)
            && self.held_out.coverage.represented == metrics.overall.samples;
        let rejection = self.rollout.rejected_actions as f64 / self.rollout.total_actions as f64;
        let rejection_rate = rejection < 0.001;
        let gameplay_score = self.gameplay.learner_not_worse();
        let both_sides = metrics.radiant.samples != 0 && metrics.dire.samples != 0;
        let side_audit = self.rollout.side_audit_passed;
        let exploit_audit = self.rollout.exploit_audit_passed;
        let nontrivial_action_distribution = metrics
            .overall
            .action_distribution
            .iter()
            .filter(|count| **count != 0)
            .count()
            >= 2;
        Ok(PromotionGateResult {
            full_agreement,
            teacher_coverage,
            rejection_rate,
            gameplay_score,
            both_sides,
            side_audit,
            exploit_audit,
            nontrivial_action_distribution,
            radiant_win_rate: self.gameplay.radiant.learner_win_rate(),
            dire_win_rate: self.gameplay.dire.learner_win_rate(),
            passed: full_agreement
                && teacher_coverage
                && rejection_rate
                && gameplay_score
                && both_sides
                && side_audit
                && exploit_audit
                && nontrivial_action_distribution,
        })
    }

    fn validate_inputs(&self, candidate: crate::PolicyIdentity) -> Result<(), ImitationError> {
        let metrics = &self.held_out.metrics;
        validate_evaluation_aggregate(&metrics.overall)?;
        validate_evaluation_aggregate(&metrics.radiant)?;
        validate_evaluation_aggregate(&metrics.dire)?;
        if metrics.radiant.samples.checked_add(metrics.dire.samples)
            != Some(metrics.overall.samples)
        {
            return Err(ImitationError::InvalidEvaluationCounts);
        }
        validate_pool_binding(&self.held_out.pool)?;
        validate_scope(self.held_out.pool.scope)?;
        if self.held_out.candidate != candidate
            || self.rollout.candidate != candidate
            || self.gameplay.candidate != candidate
        {
            return Err(ImitationError::PolicyIdentityMismatch);
        }
        self.held_out.coverage.validate()?;
        let coverage = &self.held_out.coverage;
        if metrics.overall.teacher_covered != coverage.represented
            || metrics.overall.teacher_attempted != coverage.attempted
        {
            return Err(ImitationError::InvalidTeacherCoverage);
        }
        for (aggregate, side) in [
            (&metrics.radiant, ImitationSide::Radiant),
            (&metrics.dire, ImitationSide::Dire),
        ] {
            let (represented, attempted) = coverage.side_counts(side);
            if aggregate.teacher_covered != represented || aggregate.teacher_attempted != attempted
            {
                return Err(ImitationError::InvalidTeacherCoverage);
            }
        }
        self.gameplay.validate()?;
        if self.gameplay.paired_seeds != self.held_out.pool.seeds.promotion {
            return Err(ImitationError::InvalidGameplayReport(
                "promotion seed namespace",
            ));
        }
        if self.rollout.total_actions < MIN_PROMOTION_ROLLOUT_ACTIONS
            || self.rollout.total_actions > MAX_TRAINING_COUNTER
            || self.rollout.rejected_actions > MAX_TRAINING_COUNTER
            || self.rollout.rejected_actions > self.rollout.total_actions
        {
            return Err(ImitationError::InvalidRolloutCounts);
        }
        Ok(())
    }
}

fn validate_evaluation_aggregate(aggregate: &EvaluationAggregate) -> Result<(), ImitationError> {
    let family_total = aggregate
        .families
        .iter()
        .map(|family| family.total)
        .try_fold(0usize, usize::checked_add);
    let distribution_total = aggregate
        .action_distribution
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add);
    if aggregate.samples > MAX_IMITATION_SAMPLES
        || aggregate.full.total != aggregate.samples
        || aggregate.full.matching > aggregate.full.total
        || aggregate.kind.matching > aggregate.kind.total
        || aggregate.kind.total > aggregate.samples
        || aggregate.teacher_covered > aggregate.teacher_attempted
        || aggregate.teacher_attempted > MAX_IMITATION_SAMPLES
        || family_total != Some(aggregate.samples)
        || distribution_total != Some(aggregate.samples)
    {
        return Err(ImitationError::InvalidEvaluationCounts);
    }
    for family in aggregate.families {
        if family.matching > family.total || family.total > aggregate.samples {
            return Err(ImitationError::InvalidEvaluationCounts);
        }
    }
    Ok(())
}

/// Sorted unique and pairwise-disjoint optimization/evaluation seed metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedNamespaces {
    training: Vec<u64>,
    validation: Vec<u64>,
    promotion: Vec<u64>,
}

impl SeedNamespaces {
    pub fn new(
        training: Vec<u64>,
        validation: Vec<u64>,
        promotion: Vec<u64>,
    ) -> Result<Self, ImitationError> {
        let training = validate_seed_namespace("training", training)?;
        let validation = validate_seed_namespace("validation", validation)?;
        let promotion = validate_seed_namespace("promotion", promotion)?;
        validate_seed_disjoint("training", &training, "validation", &validation)?;
        validate_seed_disjoint("training", &training, "promotion", &promotion)?;
        validate_seed_disjoint("validation", &validation, "promotion", &promotion)?;
        Ok(Self {
            training,
            validation,
            promotion,
        })
    }

    pub fn training(&self) -> &[u64] {
        &self.training
    }
    pub fn validation(&self) -> &[u64] {
        &self.validation
    }
    pub fn promotion(&self) -> &[u64] {
        &self.promotion
    }

    fn contains(&self, namespace: SeedNamespace, seed: u64) -> bool {
        let seeds = match namespace {
            SeedNamespace::Training => &self.training,
            SeedNamespace::Validation => &self.validation,
            SeedNamespace::Promotion => &self.promotion,
        };
        seeds.binary_search(&seed).is_ok()
    }
}

fn validate_seed_namespace(
    name: &'static str,
    mut seeds: Vec<u64>,
) -> Result<Vec<u64>, ImitationError> {
    if seeds.len() > MAX_SEED_NAMESPACE {
        return Err(ImitationError::SeedCapacity {
            namespace: name,
            count: seeds.len(),
            maximum: MAX_SEED_NAMESPACE,
        });
    }
    seeds.sort_unstable();
    for pair in seeds.windows(2) {
        if pair[0] == pair[1] {
            return Err(ImitationError::DuplicateSeed {
                namespace: name,
                seed: pair[0],
            });
        }
    }
    Ok(seeds)
}

fn validate_seed_disjoint(
    first_name: &'static str,
    first: &[u64],
    second_name: &'static str,
    second: &[u64],
) -> Result<(), ImitationError> {
    let (mut left, mut right) = (0usize, 0usize);
    while left < first.len() && right < second.len() {
        match first[left].cmp(&second[right]) {
            std::cmp::Ordering::Less => left += 1,
            std::cmp::Ordering::Greater => right += 1,
            std::cmp::Ordering::Equal => {
                return Err(ImitationError::SeedOverlap {
                    first: first_name,
                    second: second_name,
                    seed: first[left],
                });
            }
        }
    }
    Ok(())
}

fn prediction_matches(target: &BehavioralTarget, prediction: BehavioralPrediction) -> bool {
    head_matches(&target.controlled, prediction.controlled)
        && head_matches(&target.ability, prediction.ability)
        && head_matches(&target.item, prediction.item)
        && head_matches(&target.swap, prediction.swap)
        && head_matches(&target.learn, prediction.learn)
        && head_matches(&target.shop, prediction.shop)
        && head_matches(&target.loot, prediction.loot)
        && head_matches(&target.target_mode, prediction.target_mode)
        && head_matches(&target.put_mode, prediction.put_mode)
        && head_matches(&target.entity_pointer, prediction.entity_pointer)
        && head_matches(&target.point_pointer, prediction.point_pointer)
}

fn head_matches<const WIDTH: usize>(target: &HeadTarget<WIDTH>, prediction: Option<usize>) -> bool {
    !target.active || prediction == Some(target.selected)
}

/// Gameplay early-stopping configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyStoppingConfig {
    pub minimum_improvement: f64,
    pub patience: u32,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            minimum_improvement: 0.0,
            patience: 10,
        }
    }
}

/// Bounded best gameplay snapshot and no-improvement state.
#[derive(Clone, Debug, PartialEq)]
pub struct EarlyStopper {
    config: EarlyStoppingConfig,
    best_epoch: Option<u64>,
    best_score: Option<f64>,
    best_state: Option<CompleteTrainingState>,
    stale_evaluations: u32,
    last_evaluation_epoch: Option<u64>,
    stopped: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct CompleteTrainingState {
    model: ModelAdamSnapshot,
    counters: TrainerCounters,
    shuffle: ShuffleState,
}

impl EarlyStopper {
    pub fn new(config: EarlyStoppingConfig) -> Result<Self, ImitationError> {
        validate_early_config(config)?;
        Ok(Self {
            config,
            best_epoch: None,
            best_score: None,
            best_state: None,
            stale_evaluations: 0,
            last_evaluation_epoch: None,
            stopped: false,
        })
    }

    /// Records finite gameplay score and returns whether patience is exhausted.
    fn observe(
        &mut self,
        epoch: u64,
        score: f64,
        state: CompleteTrainingState,
    ) -> Result<bool, ImitationError> {
        if self.stopped {
            return Ok(true);
        }
        if !score.is_finite() {
            return Err(ImitationError::NonFiniteEvaluation);
        }
        if epoch > MAX_TRAINING_COUNTER {
            return Err(ImitationError::CounterOverflow {
                counter: "epoch",
                maximum: MAX_TRAINING_COUNTER,
            });
        }
        if let Some(previous) = self.last_evaluation_epoch
            && epoch <= previous
        {
            return Err(ImitationError::EvaluationEpochOrder { epoch, previous });
        }
        let improved = self
            .best_score
            .is_none_or(|best| score > best + self.config.minimum_improvement);
        if improved {
            self.best_epoch = Some(epoch);
            self.best_score = Some(score);
            self.best_state = Some(state);
            self.stale_evaluations = 0;
            self.last_evaluation_epoch = Some(epoch);
            return Ok(false);
        }
        self.stale_evaluations =
            self.stale_evaluations
                .checked_add(1)
                .ok_or(ImitationError::CounterOverflow {
                    counter: "early-stopping patience",
                    maximum: u64::from(MAX_EARLY_STOPPING_PATIENCE),
                })?;
        self.last_evaluation_epoch = Some(epoch);
        self.stopped = self.stale_evaluations >= self.config.patience;
        Ok(self.stopped)
    }

    pub const fn best_epoch(&self) -> Option<u64> {
        self.best_epoch
    }
}

fn validate_early_config(config: EarlyStoppingConfig) -> Result<(), ImitationError> {
    if !config.minimum_improvement.is_finite() || config.minimum_improvement < 0.0 {
        return Err(ImitationError::InvalidEarlyStopping("minimum improvement"));
    }
    if !(1..=MAX_EARLY_STOPPING_PATIENCE).contains(&config.patience) {
        return Err(ImitationError::InvalidEarlyStopping("patience"));
    }
    Ok(())
}

/// Trainer counters persisted in strict checkpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainerCounters {
    pub epoch: u64,
    pub global_update: u64,
}

/// One deterministic complete pass over all selected Train samples.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingEpochReport {
    pub epoch: u64,
    pub order: Vec<usize>,
    pub updates: Vec<ModelUpdateReport>,
    pub average_loss: f64,
}

/// Deterministic effective-batch and optimizer orchestration state.
pub struct BehavioralTrainer {
    effective_batch: usize,
    shuffle: ShuffleState,
    counters: TrainerCounters,
    adam: AdamState,
    early_stopper: EarlyStopper,
    pool: PoolBinding,
    model_identity: crate::PolicyIdentity,
}

impl BehavioralTrainer {
    pub fn new(
        effective_batch: usize,
        shuffle_seed: u64,
        adam: AdamConfig,
        early_stopping: EarlyStoppingConfig,
        model: &PolicyModel,
        pool: &ImitationPool,
    ) -> Result<Self, ImitationError> {
        validate_effective_batch(effective_batch)?;
        let early_stopper = EarlyStopper::new(early_stopping)?;
        let adam = model.claim_optimizer(adam)?;
        let model_identity = adam.binding().policy;
        Ok(Self {
            effective_batch,
            shuffle: ShuffleState::new(shuffle_seed),
            counters: TrainerCounters::default(),
            adam,
            early_stopper,
            pool: pool.binding.clone(),
            model_identity,
        })
    }

    /// Visits each Train sample exactly once and permits one final partial batch.
    pub fn train_epoch(
        &mut self,
        model: &PolicyModel,
        pool: &ImitationPool,
    ) -> Result<TrainingEpochReport, ImitationError> {
        self.train_epoch_atomic(model, pool, None, false)
    }

    /// Accepts the current or a newer revision of the same in-memory pool.
    pub fn rebind_pool(&mut self, pool: &ImitationPool) -> Result<(), ImitationError> {
        validate_pool_binding(&pool.binding)?;
        validate_scope(pool.binding.scope)?;
        validate_trainer_counters(self)?;
        let current = &self.pool;
        let candidate = &pool.binding;
        if current.instance != candidate.instance
            || current.capacity != candidate.capacity
            || current.provenance.lineage != candidate.provenance.lineage
            || current.seeds != candidate.seeds
            || current.scope != candidate.scope
            || candidate.provenance.revision < current.provenance.revision
        {
            return Err(ImitationError::PoolBindingMismatch);
        }
        self.pool = candidate.clone();
        Ok(())
    }

    fn train_epoch_atomic(
        &mut self,
        model: &PolicyModel,
        pool: &ImitationPool,
        fail_update: Option<usize>,
        rollback_failure: bool,
    ) -> Result<TrainingEpochReport, ImitationError> {
        self.validate_pool(pool)?;
        validate_trainer_counters(self)?;
        let rollback = self.complete_state(model)?;
        let early = self.early_stopper.clone();
        match self.train_epoch_inner(model, pool, fail_update) {
            Ok(report) => Ok(report),
            Err(cause) => {
                let restored = self.rollback(model, rollback, early, rollback_failure);
                match restored {
                    Ok(()) => Err(cause),
                    Err(rollback) => Err(ImitationError::Rollback {
                        cause: cause.to_string(),
                        rollback: rollback.to_string(),
                    }),
                }
            }
        }
    }

    fn train_epoch_inner(
        &mut self,
        model: &PolicyModel,
        pool: &ImitationPool,
        fail_update: Option<usize>,
    ) -> Result<TrainingEpochReport, ImitationError> {
        let order = pool.training_order(&mut self.shuffle)?;
        let update_count = order.len().div_ceil(self.effective_batch);
        validate_counter_increment("global update", self.counters.global_update, update_count)?;
        validate_counter_increment("epoch", self.counters.epoch, 1)?;
        let mut updates = Vec::with_capacity(update_count);
        let mut loss_sum = 0.0f64;
        for (update_index, batch) in order.chunks(self.effective_batch).enumerate() {
            if fail_update == Some(update_index) {
                return Err(ImitationError::InjectedEpochFailure {
                    update: update_index,
                });
            }
            let examples = batch
                .iter()
                .map(|index| {
                    pool.get(*index)
                        .ok_or(ImitationError::CheckpointState("training order index"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let report = model.behavioral_update(&examples, &mut self.adam)?;
            self.model_identity = self.adam.binding().policy;
            loss_sum += report.average_loss * report.sample_count as f64;
            self.counters.global_update += 1;
            updates.push(report);
        }
        self.counters.epoch += 1;
        Ok(TrainingEpochReport {
            epoch: self.counters.epoch,
            order,
            updates,
            average_loss: loss_sum / pool_train_count(pool) as f64,
        })
    }

    fn complete_state(&self, model: &PolicyModel) -> Result<CompleteTrainingState, ImitationError> {
        Ok(CompleteTrainingState {
            model: model.coherent_snapshot(&self.adam)?,
            counters: self.counters,
            shuffle: self.shuffle,
        })
    }

    fn rollback(
        &mut self,
        model: &PolicyModel,
        state: CompleteTrainingState,
        early: EarlyStopper,
        rollback_failure: bool,
    ) -> Result<(), ImitationError> {
        #[cfg(test)]
        if rollback_failure {
            let expected = self.adam.binding();
            let binding =
                model.restore_snapshot_with_failure(&state.model, &mut self.adam, expected)?;
            self.model_identity = binding.policy;
        }
        #[cfg(not(test))]
        let _ = rollback_failure;
        if !rollback_failure {
            let expected = self.adam.binding();
            let binding = model.restore_snapshot(&state.model, &mut self.adam, expected)?;
            self.model_identity = binding.policy;
        }
        self.counters = state.counters;
        self.shuffle = state.shuffle;
        self.early_stopper = early;
        Ok(())
    }

    fn validate_pool(&self, pool: &ImitationPool) -> Result<(), ImitationError> {
        if self.pool != pool.binding {
            return Err(ImitationError::PoolBindingMismatch);
        }
        Ok(())
    }

    pub const fn adam(&self) -> &AdamState {
        &self.adam
    }
    pub const fn counters(&self) -> TrainerCounters {
        self.counters
    }
    pub const fn model_identity(&self) -> crate::PolicyIdentity {
        self.model_identity
    }
    pub const fn shuffle_draws(&self) -> u64 {
        self.shuffle.draws
    }

    /// Records gameplay evaluation at the current completed epoch.
    pub fn observe_gameplay(
        &mut self,
        score: f64,
        model: &PolicyModel,
    ) -> Result<bool, ImitationError> {
        if !score.is_finite() {
            return Err(ImitationError::NonFiniteEvaluation);
        }
        if self.early_stopper.stopped {
            return Ok(true);
        }
        validate_trainer_counters(self)?;
        let state = self.complete_state(model)?;
        self.early_stopper
            .observe(self.counters.epoch, score, state)
    }

    /// Atomically restores the best gameplay-evaluated model parameters.
    pub fn restore_best(&mut self, model: &PolicyModel) -> Result<(), ImitationError> {
        let state = self
            .early_stopper
            .best_state
            .clone()
            .ok_or(ImitationError::CheckpointState("best training snapshot"))?;
        let expected = self.adam.binding();
        let binding = model.restore_snapshot(&state.model, &mut self.adam, expected)?;
        self.model_identity = binding.policy;
        self.counters = state.counters;
        self.shuffle = state.shuffle;
        self.early_stopper.stale_evaluations = 0;
        self.early_stopper.last_evaluation_epoch = self.early_stopper.best_epoch;
        self.early_stopper.stopped = false;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn train_epoch_with_failure(
        &mut self,
        model: &PolicyModel,
        pool: &ImitationPool,
        update: usize,
        rollback_failure: bool,
    ) -> Result<TrainingEpochReport, ImitationError> {
        self.train_epoch_atomic(model, pool, Some(update), rollback_failure)
    }

    #[cfg(test)]
    pub(crate) fn set_shuffle_draws_for_test(&mut self, draws: u64) {
        self.shuffle.draws = draws;
    }

    #[cfg(test)]
    pub(crate) fn set_counters_for_test(
        &mut self,
        counters: TrainerCounters,
    ) -> Result<(), ImitationError> {
        let (first, second) = self.adam.moments();
        self.adam = AdamState::from_parts(
            self.adam.config(),
            first.to_vec(),
            second.to_vec(),
            counters.global_update,
            self.adam.binding(),
        )?;
        self.counters = counters;
        Ok(())
    }
}

fn pool_train_count(pool: &ImitationPool) -> usize {
    pool.samples
        .iter()
        .filter(|sample| sample.split == ImitationSplit::Train)
        .count()
}

fn validate_effective_batch(value: usize) -> Result<(), ImitationError> {
    if !(1..=MODEL_MAX_BATCH).contains(&value) {
        return Err(ImitationError::EffectiveBatch {
            value,
            maximum: MODEL_MAX_BATCH,
        });
    }
    Ok(())
}

fn validate_counter_increment(
    counter: &'static str,
    value: u64,
    increment: usize,
) -> Result<(), ImitationError> {
    let increment = u64::try_from(increment).map_err(|_| ImitationError::CounterOverflow {
        counter,
        maximum: MAX_TRAINING_COUNTER,
    })?;
    if value
        .checked_add(increment)
        .is_none_or(|next| next > MAX_TRAINING_COUNTER)
    {
        return Err(ImitationError::CounterOverflow {
            counter,
            maximum: MAX_TRAINING_COUNTER,
        });
    }
    Ok(())
}

/// Complete strict in-memory behavioral-training checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingCheckpoint {
    pub action_schema_version: u32,
    pub action_schema_hash: u64,
    pub model_schema_version: u32,
    pub model_schema_hash: u64,
    pub feature_schema_version: u32,
    pub feature_schema_hash: u64,
    pub pool: PoolBinding,
    pub optimizer_version: u32,
    pub model_identity: crate::PolicyIdentity,
    pub optimizer_lineage: u64,
    pub parameters: Vec<f32>,
    pub adam_config: AdamConfig,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    pub adam_step: u64,
    pub effective_batch: usize,
    pub epoch: u64,
    pub global_update: u64,
    pub shuffle_state: u64,
    pub shuffle_draws: u64,
    pub early_stopping_config: EarlyStoppingConfig,
    pub best_epoch: Option<u64>,
    pub best_score: Option<f64>,
    pub best_model_identity: Option<crate::PolicyIdentity>,
    pub best_optimizer_lineage: Option<u64>,
    pub best_parameters: Option<Vec<f32>>,
    pub best_first_moment: Option<Vec<f32>>,
    pub best_second_moment: Option<Vec<f32>>,
    pub best_adam_step: Option<u64>,
    pub best_counters: Option<TrainerCounters>,
    pub best_shuffle_state: Option<u64>,
    pub best_shuffle_draws: Option<u64>,
    pub stale_evaluations: u32,
    pub last_evaluation_epoch: Option<u64>,
    pub stopped: bool,
}

impl TrainingCheckpoint {
    pub fn capture(
        model: &PolicyModel,
        trainer: &BehavioralTrainer,
        pool: &ImitationPool,
    ) -> Result<Self, ImitationError> {
        trainer.validate_pool(pool)?;
        validate_trainer_counters(trainer)?;
        let current = model.coherent_snapshot(&trainer.adam)?;
        let (first_moment, second_moment) = current.adam.moments();
        let best = trainer.early_stopper.best_state.as_ref();
        Ok(Self {
            action_schema_version: ACTION_SCHEMA_VERSION,
            action_schema_hash: ACTION_SCHEMA_HASH,
            model_schema_version: MODEL_SCHEMA_VERSION,
            model_schema_hash: MODEL_SCHEMA_HASH,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            feature_schema_hash: FEATURE_SCHEMA_HASH,
            pool: trainer.pool.clone(),
            optimizer_version: IMITATION_OPTIMIZER_VERSION,
            model_identity: current.adam.binding().policy,
            optimizer_lineage: current.adam.binding().lineage.get(),
            parameters: current.parameters,
            adam_config: current.adam.config(),
            first_moment: first_moment.to_vec(),
            second_moment: second_moment.to_vec(),
            adam_step: current.adam.step(),
            effective_batch: trainer.effective_batch,
            epoch: trainer.counters.epoch,
            global_update: trainer.counters.global_update,
            shuffle_state: trainer.shuffle.state(),
            shuffle_draws: trainer.shuffle.draws(),
            early_stopping_config: trainer.early_stopper.config,
            best_epoch: trainer.early_stopper.best_epoch,
            best_score: trainer.early_stopper.best_score,
            best_model_identity: best.map(|state| state.model.adam.binding().policy),
            best_optimizer_lineage: best.map(|state| state.model.adam.binding().lineage.get()),
            best_parameters: best.map(|state| state.model.parameters.clone()),
            best_first_moment: best.map(|state| state.model.adam.moments().0.to_vec()),
            best_second_moment: best.map(|state| state.model.adam.moments().1.to_vec()),
            best_adam_step: best.map(|state| state.model.adam.step()),
            best_counters: best.map(|state| state.counters),
            best_shuffle_state: best.map(|state| state.shuffle.state()),
            best_shuffle_draws: best.map(|state| state.shuffle.draws()),
            stale_evaluations: trainer.early_stopper.stale_evaluations,
            last_evaluation_epoch: trainer.early_stopper.last_evaluation_epoch,
            stopped: trainer.early_stopper.stopped,
        })
    }

    /// Strictly validates all state before atomically replacing model and trainer state.
    pub fn restore(
        &self,
        model: &PolicyModel,
        trainer: &mut BehavioralTrainer,
        pool: &ImitationPool,
    ) -> Result<(), ImitationError> {
        let candidate = self.validate(trainer, pool)?;
        let mut adam = candidate.current.adam.clone();
        let expected = trainer.adam.binding();
        let binding = model.restore_snapshot(&candidate.current, &mut adam, expected)?;
        trainer.effective_batch = self.effective_batch;
        trainer.shuffle = candidate.shuffle;
        trainer.counters = candidate.counters;
        trainer.adam = adam;
        trainer.early_stopper = candidate.early;
        trainer.pool = self.pool.clone();
        trainer.model_identity = binding.policy;
        Ok(())
    }

    fn validate(
        &self,
        trainer: &BehavioralTrainer,
        pool: &ImitationPool,
    ) -> Result<CheckpointCandidate, ImitationError> {
        self.precheck_lengths()?;
        if self.action_schema_version != ACTION_SCHEMA_VERSION
            || self.action_schema_hash != ACTION_SCHEMA_HASH
            || self.model_schema_version != MODEL_SCHEMA_VERSION
            || self.model_schema_hash != MODEL_SCHEMA_HASH
            || self.feature_schema_version != FEATURE_SCHEMA_VERSION
            || self.feature_schema_hash != FEATURE_SCHEMA_HASH
        {
            return Err(ImitationError::CheckpointSchema);
        }
        if self.optimizer_version != IMITATION_OPTIMIZER_VERSION {
            return Err(ImitationError::CheckpointState(
                "optimizer ownership or version",
            ));
        }
        if self.pool != pool.binding || self.pool != trainer.pool {
            return Err(ImitationError::PoolBindingMismatch);
        }
        validate_pool_binding(&self.pool)?;
        validate_scope(self.pool.scope)?;
        validate_parameter_vector(&self.parameters, "parameters")?;
        validate_effective_batch(self.effective_batch)?;
        validate_checkpoint_counters(self)?;
        let optimizer_lineage = NonZeroU64::new(self.optimizer_lineage)
            .ok_or(ImitationError::CheckpointState("optimizer lineage"))?;
        let adam = AdamState::from_parts(
            self.adam_config,
            self.first_moment.clone(),
            self.second_moment.clone(),
            self.adam_step,
            crate::OptimizerBinding {
                lineage: optimizer_lineage,
                policy: self.model_identity,
            },
        )?;
        let early = validate_checkpoint_early(self)?;
        Ok(CheckpointCandidate {
            current: ModelAdamSnapshot {
                parameters: self.parameters.clone(),
                adam,
            },
            counters: TrainerCounters {
                epoch: self.epoch,
                global_update: self.global_update,
            },
            shuffle: ShuffleState {
                state: self.shuffle_state,
                draws: self.shuffle_draws,
            },
            early,
        })
    }

    fn precheck_lengths(&self) -> Result<(), ImitationError> {
        precheck_vector_length(&self.parameters, "parameters")?;
        precheck_vector_length(&self.first_moment, "first moment")?;
        precheck_vector_length(&self.second_moment, "second moment")?;
        for (values, field) in [
            (&self.best_parameters, "best parameters"),
            (&self.best_first_moment, "best first moment"),
            (&self.best_second_moment, "best second moment"),
        ] {
            if let Some(values) = values {
                precheck_vector_length(values, field)?;
            }
        }
        Ok(())
    }
}

fn validate_pool_binding(binding: &PoolBinding) -> Result<(), ImitationError> {
    if !(1..=MAX_IMITATION_SAMPLES).contains(&binding.capacity)
        || binding.provenance.lineage == 0
        || binding.provenance.revision > MAX_TRAINING_COUNTER
    {
        return Err(ImitationError::CheckpointState("pool binding"));
    }
    Ok(())
}

fn allocate_pool_instance() -> Result<NonZeroU64, ImitationError> {
    let value = NEXT_POOL_INSTANCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| ImitationError::CheckpointState("pool instance availability"))?;
    NonZeroU64::new(value).ok_or(ImitationError::CheckpointState("pool instance identity"))
}

fn allocate_sample_instance() -> Result<NonZeroU64, ImitationError> {
    let value = NEXT_SAMPLE_INSTANCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| ImitationError::CheckpointState("sample instance availability"))?;
    NonZeroU64::new(value).ok_or(ImitationError::CheckpointState("sample instance identity"))
}

fn validate_trainer_counters(trainer: &BehavioralTrainer) -> Result<(), ImitationError> {
    if trainer.counters.epoch > MAX_TRAINING_COUNTER
        || trainer.counters.global_update > MAX_TRAINING_COUNTER
        || trainer.shuffle.draws > MAX_TRAINING_COUNTER
        || trainer.adam.step() != trainer.counters.global_update
        || trainer.counters.epoch > trainer.counters.global_update
        || trainer.adam.binding().policy != trainer.model_identity
    {
        return Err(ImitationError::CheckpointState(
            "trainer counter relationship",
        ));
    }
    Ok(())
}

struct CheckpointCandidate {
    current: ModelAdamSnapshot,
    counters: TrainerCounters,
    shuffle: ShuffleState,
    early: EarlyStopper,
}

fn validate_checkpoint_counters(checkpoint: &TrainingCheckpoint) -> Result<(), ImitationError> {
    if checkpoint.epoch > MAX_TRAINING_COUNTER || checkpoint.global_update > MAX_TRAINING_COUNTER {
        return Err(ImitationError::CheckpointState("training counters"));
    }
    if checkpoint.shuffle_draws > MAX_TRAINING_COUNTER {
        return Err(ImitationError::CheckpointState("shuffle draw counter"));
    }
    if checkpoint.adam_step != checkpoint.global_update
        || checkpoint.epoch > checkpoint.global_update
    {
        return Err(ImitationError::CheckpointState(
            "optimizer counter relationship",
        ));
    }
    Ok(())
}

fn validate_checkpoint_early(
    checkpoint: &TrainingCheckpoint,
) -> Result<EarlyStopper, ImitationError> {
    validate_early_config(checkpoint.early_stopping_config)?;
    if checkpoint.stale_evaluations > checkpoint.early_stopping_config.patience
        || checkpoint.stopped
            != (checkpoint.stale_evaluations >= checkpoint.early_stopping_config.patience)
    {
        return Err(ImitationError::CheckpointState("early-stopping patience"));
    }
    validate_checkpoint_evaluation_history(checkpoint)?;
    let best_state = build_best_state(checkpoint)?;
    Ok(EarlyStopper {
        config: checkpoint.early_stopping_config,
        best_epoch: checkpoint.best_epoch,
        best_score: checkpoint.best_score,
        best_state,
        stale_evaluations: checkpoint.stale_evaluations,
        last_evaluation_epoch: checkpoint.last_evaluation_epoch,
        stopped: checkpoint.stopped,
    })
}

fn validate_checkpoint_evaluation_history(
    checkpoint: &TrainingCheckpoint,
) -> Result<(), ImitationError> {
    let states_present = [
        checkpoint.best_epoch.is_some(),
        checkpoint.best_score.is_some(),
        checkpoint.best_model_identity.is_some(),
        checkpoint.best_optimizer_lineage.is_some(),
        checkpoint.best_parameters.is_some(),
        checkpoint.best_first_moment.is_some(),
        checkpoint.best_second_moment.is_some(),
        checkpoint.best_adam_step.is_some(),
        checkpoint.best_counters.is_some(),
        checkpoint.best_shuffle_state.is_some(),
        checkpoint.best_shuffle_draws.is_some(),
    ];
    if states_present.iter().any(|value| *value) && !states_present.iter().all(|value| *value) {
        return Err(ImitationError::CheckpointState("best evaluation snapshot"));
    }
    if checkpoint
        .best_score
        .is_some_and(|score| !score.is_finite())
    {
        return Err(ImitationError::CheckpointState("best score"));
    }
    match (checkpoint.best_epoch, checkpoint.last_evaluation_epoch) {
        (None, None) if checkpoint.stale_evaluations == 0 => Ok(()),
        (Some(best), Some(last)) => {
            let stale = u64::from(checkpoint.stale_evaluations);
            let epochs_valid = best <= last
                && last <= checkpoint.epoch
                && last <= MAX_TRAINING_COUNTER
                && if stale == 0 {
                    best == last
                } else {
                    last.checked_sub(best)
                        .is_some_and(|elapsed| elapsed >= stale)
                };
            if epochs_valid {
                Ok(())
            } else {
                Err(ImitationError::CheckpointState(
                    "evaluation epoch relationship",
                ))
            }
        }
        _ => Err(ImitationError::CheckpointState("best evaluation state")),
    }
}

fn build_best_state(
    checkpoint: &TrainingCheckpoint,
) -> Result<Option<CompleteTrainingState>, ImitationError> {
    let Some(parameters) = &checkpoint.best_parameters else {
        return Ok(None);
    };
    validate_parameter_vector(parameters, "best parameters")?;
    let optimizer_lineage = checkpoint
        .best_optimizer_lineage
        .and_then(NonZeroU64::new)
        .ok_or(ImitationError::CheckpointState("best optimizer lineage"))?;
    let policy = checkpoint
        .best_model_identity
        .ok_or(ImitationError::CheckpointState("best model identity"))?;
    let adam = AdamState::from_parts(
        checkpoint.adam_config,
        checkpoint
            .best_first_moment
            .clone()
            .ok_or(ImitationError::CheckpointState("best first moment"))?,
        checkpoint
            .best_second_moment
            .clone()
            .ok_or(ImitationError::CheckpointState("best second moment"))?,
        checkpoint
            .best_adam_step
            .ok_or(ImitationError::CheckpointState("best Adam step"))?,
        crate::OptimizerBinding {
            lineage: optimizer_lineage,
            policy,
        },
    )?;
    let counters = checkpoint
        .best_counters
        .ok_or(ImitationError::CheckpointState("best counters"))?;
    if counters.epoch > MAX_TRAINING_COUNTER
        || counters.global_update > MAX_TRAINING_COUNTER
        || adam.step() != counters.global_update
        || counters.epoch > counters.global_update
        || counters.epoch > checkpoint.epoch
        || counters.global_update > checkpoint.global_update
        || adam.step() > checkpoint.adam_step
    {
        return Err(ImitationError::CheckpointState("best counter relationship"));
    }
    if checkpoint.best_epoch != Some(counters.epoch) {
        return Err(ImitationError::CheckpointState(
            "best epoch snapshot relationship",
        ));
    }
    let shuffle = ShuffleState {
        state: checkpoint
            .best_shuffle_state
            .ok_or(ImitationError::CheckpointState("best shuffle state"))?,
        draws: checkpoint
            .best_shuffle_draws
            .ok_or(ImitationError::CheckpointState("best shuffle draws"))?,
    };
    if shuffle.draws > MAX_TRAINING_COUNTER {
        return Err(ImitationError::CheckpointState("best shuffle draw counter"));
    }
    if shuffle.draws > checkpoint.shuffle_draws {
        return Err(ImitationError::CheckpointState(
            "best shuffle draw relationship",
        ));
    }
    Ok(Some(CompleteTrainingState {
        model: ModelAdamSnapshot {
            parameters: parameters.clone(),
            adam,
        },
        counters,
        shuffle,
    }))
}

fn validate_scope(scope: TrainingScope) -> Result<(), ImitationError> {
    if scope.hero != SHADOW_FIEND
        || !matches!(scope.map, MapId(0) | MapId(1))
        || scope.rules_audit_version != IMITATION_RULES_AUDIT_VERSION
    {
        return Err(ImitationError::CheckpointState("training scope"));
    }
    Ok(())
}

fn precheck_vector_length(values: &[f32], field: &'static str) -> Result<(), ImitationError> {
    if values.len() != MODEL_PARAMETER_COUNT {
        return Err(ImitationError::CheckpointState(field));
    }
    Ok(())
}

fn validate_parameter_vector(values: &[f32], field: &'static str) -> Result<(), ImitationError> {
    if values.len() != MODEL_PARAMETER_COUNT || values.iter().any(|value| !value.is_finite()) {
        return Err(ImitationError::CheckpointState(field));
    }
    Ok(())
}
