use std::error::Error;
use std::fmt;

use bota_proto::{AbilitySlot, ItemSlot, MapId};

use crate::{
    ActionKind, ActionSpace, ActionTarget, FeatureFrame, MODEL_ABILITY_HEAD,
    MODEL_BEHAVIORAL_HEADS, MODEL_ENTITY_POINTER_HEAD, MODEL_ITEM_HEAD, MODEL_KIND_HEAD,
    MODEL_LEARN_HEAD, MODEL_LOOT_HEAD, MODEL_POINT_POINTER_HEAD, MODEL_SHOP_HEAD, MODEL_SWAP_HEAD,
    MODEL_UNIT_HEAD, ModelError, PutPointTarget, StructuredAction, TrainingAbilitySlot,
    TrainingItemSlot, TrainingPrefix, TrainingSlot,
};

/// Maximum epoch, optimizer-step, and global-update counter value.
pub const MAX_TRAINING_COUNTER: u64 = 1_000_000_000;
/// Current sample audit: F17 wait/progress-accounting globals with unchanged A5 legality.
/// Prior samples and reports cannot be relabelled as reward-v3 observations.
pub const IMITATION_RULES_AUDIT_VERSION: u32 = 22;

const TARGET_MODE_HEAD: usize = 3;
const PUT_MODE_HEAD: usize = 2;

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
    SampleMapOutsideScope {
        map: MapId,
    },
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
    InvalidTeacherCoverage,
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
            | Self::SampleMapOutsideScope { .. }
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
            Self::SampleMapOutsideScope { map } => write!(
                formatter,
                "imitation sample map MapId({}) is outside its training scope",
                map.0
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
            Self::InvalidTeacherCoverage => {
                formatter.write_str("imitation teacher coverage counts are inconsistent")
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

const BEHAVIORAL_MASK_BITS: usize = MODEL_KIND_HEAD
    + MODEL_UNIT_HEAD
    + MODEL_ABILITY_HEAD
    + MODEL_ITEM_HEAD
    + MODEL_SWAP_HEAD
    + MODEL_LEARN_HEAD
    + MODEL_SHOP_HEAD
    + MODEL_LOOT_HEAD
    + TARGET_MODE_HEAD
    + PUT_MODE_HEAD
    + MODEL_ENTITY_POINTER_HEAD
    + MODEL_POINT_POINTER_HEAD;
const BEHAVIORAL_MASK_BYTES: usize = BEHAVIORAL_MASK_BITS.div_ceil(8);

#[derive(Clone, Debug)]
pub(crate) struct PackedBehavioralTarget {
    active: u16,
    selected: [u8; MODEL_BEHAVIORAL_HEADS],
    masks: [u8; BEHAVIORAL_MASK_BYTES],
    prefix: TrainingPrefix,
}

impl BehavioralTarget {
    pub(crate) fn pack(&self) -> PackedBehavioralTarget {
        let mut packed = PackedBehavioralTarget {
            active: 0,
            selected: [0; MODEL_BEHAVIORAL_HEADS],
            masks: [0; BEHAVIORAL_MASK_BYTES],
            prefix: self.prefix,
        };
        let mut bit = 0usize;
        pack_head(&self.kind, 0, &mut bit, &mut packed);
        pack_head(&self.controlled, 1, &mut bit, &mut packed);
        pack_head(&self.ability, 2, &mut bit, &mut packed);
        pack_head(&self.item, 3, &mut bit, &mut packed);
        pack_head(&self.swap, 4, &mut bit, &mut packed);
        pack_head(&self.learn, 5, &mut bit, &mut packed);
        pack_head(&self.shop, 6, &mut bit, &mut packed);
        pack_head(&self.loot, 7, &mut bit, &mut packed);
        pack_head(&self.target_mode, 8, &mut bit, &mut packed);
        pack_head(&self.put_mode, 9, &mut bit, &mut packed);
        pack_head(&self.entity_pointer, 10, &mut bit, &mut packed);
        pack_head(&self.point_pointer, 11, &mut bit, &mut packed);
        debug_assert_eq!(bit, BEHAVIORAL_MASK_BITS);
        packed
    }
}

impl PackedBehavioralTarget {
    pub(crate) fn unpack(&self) -> BehavioralTarget {
        let mut bit = 0usize;
        let target = BehavioralTarget {
            kind: unpack_head(self, 0, &mut bit),
            controlled: unpack_head(self, 1, &mut bit),
            ability: unpack_head(self, 2, &mut bit),
            item: unpack_head(self, 3, &mut bit),
            swap: unpack_head(self, 4, &mut bit),
            learn: unpack_head(self, 5, &mut bit),
            shop: unpack_head(self, 6, &mut bit),
            loot: unpack_head(self, 7, &mut bit),
            target_mode: unpack_head(self, 8, &mut bit),
            put_mode: unpack_head(self, 9, &mut bit),
            entity_pointer: unpack_head(self, 10, &mut bit),
            point_pointer: unpack_head(self, 11, &mut bit),
            prefix: self.prefix,
        };
        debug_assert_eq!(bit, BEHAVIORAL_MASK_BITS);
        target
    }
}

fn pack_head<const WIDTH: usize>(
    head: &HeadTarget<WIDTH>,
    head_index: usize,
    bit: &mut usize,
    packed: &mut PackedBehavioralTarget,
) {
    if head.active {
        packed.active |= 1 << head_index;
    }
    packed.selected[head_index] = head.selected as u8;
    for value in head.mask {
        if value {
            packed.masks[*bit / 8] |= 1 << (*bit % 8);
        }
        *bit += 1;
    }
}

fn unpack_head<const WIDTH: usize>(
    packed: &PackedBehavioralTarget,
    head_index: usize,
    bit: &mut usize,
) -> HeadTarget<WIDTH> {
    let mut mask = [false; WIDTH];
    for value in &mut mask {
        *value = packed.masks[*bit / 8] & (1 << (*bit % 8)) != 0;
        *bit += 1;
    }
    HeadTarget {
        active: packed.active & (1 << head_index) != 0,
        mask,
        selected: packed.selected[head_index] as usize,
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
