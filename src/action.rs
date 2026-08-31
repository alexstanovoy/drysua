use std::error::Error;
use std::fmt;

use bota_proto::{
    AbilitySlot, Aim, EntityId, Fixed, ItemId, ItemSlot, ItemView, Order, ShopEntry, StatusFlags,
    Target, Team, UnitKind, UnitView, Vec2, WorldView,
};

use crate::tracker::{
    COURIER_ITEM_SLOTS as COURIER_BAG_SLOTS, HERO_ITEM_SLOTS as HERO_BAG_SLOTS,
    STASH_ITEM_SLOTS as STASH_SLOTS, TrackerProvenance, is_structure,
};
use crate::{
    ItemReadiness, MAX_ABILITY_SLOTS, MAX_POINT_CANDIDATES, StateTracker, TERRAIN_CELL_SIZE,
    UNIT_TOKENS,
};
/// Distance at which drysua permits stash swaps around the own fountain.
pub const STASH_ACCESS_RANGE: i32 = 1_000;
/// Version of the append-only structured-action schema.
pub const ACTION_SCHEMA_VERSION: u32 = 1;
/// Canonical action families, head widths, and autoregressive branch order.
pub const ACTION_SCHEMA_DESCRIPTOR: &str = "bota-drysua-action/v1;kinds=Continue,Stop,MovePoint,FollowUnit,Hold,AttackMovePoint,AttackUnit,Cast,Use,PutPoint,PutUnit,Take,Buy,Sell,Swap,Learn;heads=kind16,controlled2,ability8,item15,swap15,learn6,shop64,loot16,target_mode3,put_mode2,entity96,point48;target_modes=None,Entity,Point;put_modes=Underfoot,Point;";
/// Stable FNV-1a identity of [`ACTION_SCHEMA_DESCRIPTOR`].
pub const ACTION_SCHEMA_HASH: u64 = action_schema_hash(ACTION_SCHEMA_DESCRIPTOR.as_bytes());

const fn action_schema_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut index = 0usize;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

const ACTION_KIND_COUNT: usize = 16;
const ACTIVE_ITEM_SLOTS: usize = 6;
const STASH_SLOT_START: usize = HERO_BAG_SLOTS;
const WIRE_ITEM_SLOTS: usize = HERO_BAG_SLOTS + STASH_SLOTS;
const NEARBY_TREE_POINTS: usize = 8;
const PREDICTED_HERO_POINTS: usize = 4;
const STATIC_TREE_CLEARANCE: i32 = 48 + 24 + 8;
const STRUCTURE_CLEARANCE: i32 = 24 + 8;
const MAX_PURCHASE_SLOTS: usize = WIRE_ITEM_SLOTS;
const TACTICAL_RADII: [i32; 3] = [200, 600, 1_200];
/// Cells scanned around a structure for a teleport landing; covers its range.
const LANDING_SEARCH_CELLS: usize = 10;

/// Stable append-only top-level action discriminator.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionKind {
    Continue = 0,
    Stop = 1,
    MovePoint = 2,
    FollowUnit = 3,
    Hold = 4,
    AttackMovePoint = 5,
    AttackUnit = 6,
    Cast = 7,
    Use = 8,
    PutPoint = 9,
    PutUnit = 10,
    Take = 11,
    Buy = 12,
    Sell = 13,
    Swap = 14,
    Learn = 15,
}

impl ActionKind {
    /// Number of append-only action kinds in this schema.
    pub const COUNT: usize = ACTION_KIND_COUNT;
    /// All action kinds in stable model-index order.
    pub const ALL: [Self; ACTION_KIND_COUNT] = [
        Self::Continue,
        Self::Stop,
        Self::MovePoint,
        Self::FollowUnit,
        Self::Hold,
        Self::AttackMovePoint,
        Self::AttackUnit,
        Self::Cast,
        Self::Use,
        Self::PutPoint,
        Self::PutUnit,
        Self::Take,
        Self::Buy,
        Self::Sell,
        Self::Swap,
        Self::Learn,
    ];

    /// Stable zero-based model index.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Converts a stable model index to its action kind.
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self::ALL[index])
        } else {
            None
        }
    }
}

/// Unit selected after the top-level action kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControlledUnit {
    Hero,
    Courier,
}

impl ControlledUnit {
    /// Stable conditional-head index.
    pub const fn index(self) -> usize {
        match self {
            Self::Hero => 0,
            Self::Courier => 1,
        }
    }
}

/// Index into the current visible entity candidate list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityIndex(pub usize);

/// Index into the current point candidate list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointIndex(pub usize);

/// Index into the current visible loot candidate list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LootIndex(pub usize);

/// Index into the current static shop candidate list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShopIndex(pub usize);

/// Typed target selected for an ability or active item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionTarget {
    None,
    Entity(EntityIndex),
    Point(PointIndex),
}

/// Ground destination for `PutPoint`, including the wire `Target::None` form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PutPointTarget {
    Underfoot,
    Point(PointIndex),
}

/// One structured action accepted by the autoregressive action schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StructuredAction {
    Continue,
    Stop {
        unit: ControlledUnit,
    },
    MovePoint {
        unit: ControlledUnit,
        point: PointIndex,
    },
    FollowUnit {
        unit: ControlledUnit,
        target: EntityIndex,
    },
    Hold {
        unit: ControlledUnit,
    },
    AttackMovePoint {
        unit: ControlledUnit,
        point: PointIndex,
    },
    AttackUnit {
        unit: ControlledUnit,
        target: EntityIndex,
    },
    Cast {
        unit: ControlledUnit,
        slot: AbilitySlot,
        target: ActionTarget,
    },
    Use {
        unit: ControlledUnit,
        slot: ItemSlot,
        target: ActionTarget,
    },
    PutPoint {
        unit: ControlledUnit,
        source: ItemSlot,
        target: PutPointTarget,
    },
    PutUnit {
        unit: ControlledUnit,
        source: ItemSlot,
        target: EntityIndex,
    },
    Take {
        unit: ControlledUnit,
        loot: LootIndex,
    },
    Buy {
        unit: ControlledUnit,
        item: ShopIndex,
    },
    Sell {
        unit: ControlledUnit,
        slot: ItemSlot,
    },
    Swap {
        unit: ControlledUnit,
        from: ItemSlot,
        to: ItemSlot,
    },
    Learn {
        slot: AbilitySlot,
    },
}

/// Short public name for one strongly typed structured action.
pub type Action = StructuredAction;

impl StructuredAction {
    /// Top-level discriminator used by the kind mask.
    pub const fn kind(self) -> ActionKind {
        match self {
            Self::Continue => ActionKind::Continue,
            Self::Stop { .. } => ActionKind::Stop,
            Self::MovePoint { .. } => ActionKind::MovePoint,
            Self::FollowUnit { .. } => ActionKind::FollowUnit,
            Self::Hold { .. } => ActionKind::Hold,
            Self::AttackMovePoint { .. } => ActionKind::AttackMovePoint,
            Self::AttackUnit { .. } => ActionKind::AttackUnit,
            Self::Cast { .. } => ActionKind::Cast,
            Self::Use { .. } => ActionKind::Use,
            Self::PutPoint { .. } => ActionKind::PutPoint,
            Self::PutUnit { .. } => ActionKind::PutUnit,
            Self::Take { .. } => ActionKind::Take,
            Self::Buy { .. } => ActionKind::Buy,
            Self::Sell { .. } => ActionKind::Sell,
            Self::Swap { .. } => ActionKind::Swap,
            Self::Learn { .. } => ActionKind::Learn,
        }
    }

    /// Controlled unit, absent for action families that do not select one.
    pub const fn controlled_unit(self) -> Option<ControlledUnit> {
        match self {
            Self::Continue | Self::Learn { .. } => None,
            Self::Stop { unit }
            | Self::MovePoint { unit, .. }
            | Self::FollowUnit { unit, .. }
            | Self::Hold { unit }
            | Self::AttackMovePoint { unit, .. }
            | Self::AttackUnit { unit, .. }
            | Self::Cast { unit, .. }
            | Self::Use { unit, .. }
            | Self::PutPoint { unit, .. }
            | Self::PutUnit { unit, .. }
            | Self::Take { unit, .. }
            | Self::Buy { unit, .. }
            | Self::Sell { unit, .. }
            | Self::Swap { unit, .. } => Some(unit),
        }
    }
}

/// Relation of a visible entity candidate to the controlled seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityRelation {
    Own,
    Allied,
    Enemy,
    Neutral,
}

/// Model-safe visible entity metadata; the opaque handle remains private.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityCandidate {
    id: EntityId,
    unit: UnitView,
    /// Unit category from the current `WorldView`.
    pub kind: UnitKind,
    /// Relation to this seat.
    pub relation: EntityRelation,
    /// Current visible position.
    pub position: Vec2,
}

impl EntityCandidate {
    pub(crate) const fn id(&self) -> EntityId {
        self.id
    }

    pub(crate) fn unit(&self) -> &UnitView {
        &self.unit
    }
}

/// One of the eight deterministic tactical directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PointDirection {
    East,
    NorthEast,
    North,
    NorthWest,
    West,
    SouthWest,
    South,
    SouthEast,
}

/// Relation attached to a public map landmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LandmarkRelation {
    Own,
    Enemy,
}

/// Provenance of one point candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointSource {
    Tactical {
        direction: PointDirection,
        radius: i32,
    },
    StaticTree,
    PlantedTree,
    BuildingLanding(UnitKind),
    Fountain(LandmarkRelation),
    Tower(LandmarkRelation),
    PredictedHero(EntityRelation),
    PredictedCreep(EntityRelation),
}

/// Deterministic model-visible point candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointCandidate {
    /// Candidate position clamped to the terrain grid.
    pub position: Vec2,
    /// Candidate provenance used by Tree and Building target masks.
    pub source: PointSource,
    /// Whether public terrain and candidate provenance permit standing here.
    pub walkable: bool,
    /// Whether a currently standing static or planted tree occupies this point.
    pub standing_tree: bool,
    /// Whether this point is a landing candidate near a visible allied structure.
    pub allied_building: bool,
}

/// Model-safe visible ground-item candidate; its opaque handle remains private.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LootCandidate {
    id: EntityId,
    /// Current visible position.
    pub position: Vec2,
    /// Item carried by this ground entity.
    pub item: ItemId,
    /// Current charge count, absent for items without charges.
    pub charges: Option<u8>,
}

impl LootCandidate {
    pub(crate) const fn id(&self) -> EntityId {
        self.id
    }
}

/// One bounded shop output in stable item-id order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShopCandidate {
    /// Item sent in a decoded buy order.
    pub item: ItemId,
    /// Full purchase cost from match metadata.
    pub cost: i32,
}

/// Boolean mask for all stable top-level kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindMask([bool; ACTION_KIND_COUNT]);

impl KindMask {
    /// Whether the action family has at least one valid full action.
    pub const fn allows(&self, kind: ActionKind) -> bool {
        self.0[kind.index()]
    }

    /// Fixed model-head representation in stable kind order.
    pub const fn as_array(&self) -> &[bool; ACTION_KIND_COUNT] {
        &self.0
    }
}

/// Boolean controlled-unit mask in Hero, Courier order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlledUnitMask([bool; 2]);

impl ControlledUnitMask {
    /// Whether a controlled unit has at least one valid continuation.
    pub const fn allows(&self, unit: ControlledUnit) -> bool {
        self.0[unit.index()]
    }

    /// Fixed model-head representation in Hero, Courier order.
    pub const fn as_array(&self) -> &[bool; 2] {
        &self.0
    }
}

/// Conditional target mask for an ability or active item slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetMask {
    none: bool,
    entities: Vec<bool>,
    points: Vec<bool>,
}

impl TargetMask {
    /// Whether the no-target form is valid.
    pub const fn allows_none(&self) -> bool {
        self.none
    }

    /// Entity pointer mask in current candidate order.
    pub fn entities(&self) -> &[bool] {
        &self.entities
    }

    /// Point pointer mask in current candidate order.
    pub fn points(&self) -> &[bool] {
        &self.points
    }

    /// Whether a typed target is valid for this slot.
    pub fn allows(&self, target: ActionTarget) -> bool {
        match target {
            ActionTarget::None => self.none,
            ActionTarget::Entity(index) => self.entities.get(index.0).copied().unwrap_or(false),
            ActionTarget::Point(index) => self.points.get(index.0).copied().unwrap_or(false),
        }
    }

    fn any(&self) -> bool {
        self.none || self.entities.contains(&true) || self.points.contains(&true)
    }
}

/// A decoded order paired with the wire controlled-unit selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IssuedOrder {
    /// `None` selects the own hero; `Some` selects the current courier.
    pub unit: Option<EntityId>,
    /// Exact wire order built only from current candidates.
    pub order: Order,
}

/// Construction or decoding failure for the structured action space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionError {
    SnapshotRequired,
    Arithmetic(&'static str),
    InvalidSchema(&'static str),
    CandidateIndex {
        field: &'static str,
        index: usize,
        count: usize,
    },
    NotAllowed(ActionKind),
}

impl fmt::Display for ActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SnapshotRequired => {
                formatter.write_str("action space requires a validated snapshot")
            }
            Self::Arithmetic(field) => {
                write!(formatter, "action space arithmetic overflow in {field}")
            }
            Self::InvalidSchema(field) => {
                write!(
                    formatter,
                    "action space received invalid wire schema in {field}"
                )
            }
            Self::CandidateIndex {
                field,
                index,
                count,
            } => write!(
                formatter,
                "{field} index {index} is outside candidate count {count}"
            ),
            Self::NotAllowed(kind) => {
                write!(
                    formatter,
                    "action {kind:?} is masked by the current action space"
                )
            }
        }
    }
}

impl Error for ActionError {}

#[derive(Clone, Debug)]
struct ControlledState {
    id: EntityId,
    unit: UnitView,
    underfoot_walkable: bool,
}

#[derive(Clone, Copy, Debug)]
struct BuyRequirement {
    missing_cost: i32,
    missing_slots: usize,
}

#[derive(Clone, Debug)]
struct StaticPassability {
    axis: usize,
    open: Vec<bool>,
}

#[derive(Clone, Debug)]
struct PutPointMask {
    underfoot: bool,
    points: Vec<bool>,
}

impl PutPointMask {
    fn any(&self) -> bool {
        self.underfoot || self.points.contains(&true)
    }
}

#[derive(Clone, Debug)]
struct ControlledMasks {
    stop: bool,
    move_points: Vec<bool>,
    follow_entities: Vec<bool>,
    hold: bool,
    attack_move_points: Vec<bool>,
    attack_entities: Vec<bool>,
    casts: Vec<TargetMask>,
    uses: Vec<TargetMask>,
    put_points: Vec<PutPointMask>,
    put_units: Vec<Vec<bool>>,
    take: Vec<bool>,
    buy: Vec<bool>,
    sell: [bool; WIRE_ITEM_SLOTS],
    swap: [[bool; WIRE_ITEM_SLOTS]; WIRE_ITEM_SLOTS],
    learn: Vec<bool>,
}

impl ControlledMasks {
    fn empty(
        entity_count: usize,
        point_count: usize,
        loot_count: usize,
        shop_count: usize,
    ) -> Self {
        Self {
            stop: false,
            move_points: vec![false; point_count],
            follow_entities: vec![false; entity_count],
            hold: false,
            attack_move_points: vec![false; point_count],
            attack_entities: vec![false; entity_count],
            casts: Vec::new(),
            uses: Vec::new(),
            put_points: Vec::new(),
            put_units: Vec::new(),
            take: vec![false; loot_count],
            buy: vec![false; shop_count],
            sell: [false; WIRE_ITEM_SLOTS],
            swap: [[false; WIRE_ITEM_SLOTS]; WIRE_ITEM_SLOTS],
            learn: Vec::new(),
        }
    }

    fn family_allowed(&self, kind: ActionKind) -> bool {
        match kind {
            ActionKind::Continue => false,
            ActionKind::Stop => self.stop,
            ActionKind::MovePoint => self.move_points.contains(&true),
            ActionKind::FollowUnit => self.follow_entities.contains(&true),
            ActionKind::Hold => self.hold,
            ActionKind::AttackMovePoint => self.attack_move_points.contains(&true),
            ActionKind::AttackUnit => self.attack_entities.contains(&true),
            ActionKind::Cast => self.casts.iter().any(TargetMask::any),
            ActionKind::Use => self.uses.iter().any(TargetMask::any),
            ActionKind::PutPoint => self.put_points.iter().any(PutPointMask::any),
            ActionKind::PutUnit => self.put_units.iter().any(|mask| mask.contains(&true)),
            ActionKind::Take => self.take.contains(&true),
            ActionKind::Buy => self.buy.contains(&true),
            ActionKind::Sell => self.sell.contains(&true),
            ActionKind::Swap => self.swap.iter().any(|row| row.contains(&true)),
            ActionKind::Learn => self.learn.contains(&true),
        }
    }
}

/// Bounded candidates and all conditional legality masks for one snapshot.
pub struct ActionSpace {
    provenance: TrackerProvenance,
    controlled: [Option<ControlledState>; 2],
    entities: Vec<EntityCandidate>,
    points: Vec<PointCandidate>,
    loot: Vec<LootCandidate>,
    shop: Vec<ShopCandidate>,
    tick: u32,
    readiness: ItemReadiness,
    own_gold: i32,
    own_stash: Vec<Option<bota_proto::ItemView>>,
    own_fountain: Option<Vec2>,
    allied_buildings: Vec<Vec2>,
    buy_requirements: Vec<BuyRequirement>,
    masks: [ControlledMasks; 2],
    kind_mask: KindMask,
}

impl ActionSpace {
    /// Builds candidates only from the latest validated seat-specific snapshot.
    ///
    /// Without recorded local timers this trusts the wire readiness fields.
    pub fn from_tracker(tracker: &StateTracker) -> Result<Self, ActionError> {
        Self::from_tracker_with_readiness(tracker, &ItemReadiness::new())
    }

    /// Builds candidates and masks with locally tracked item timers applied.
    pub fn from_tracker_with_readiness(
        tracker: &StateTracker,
        readiness: &ItemReadiness,
    ) -> Result<Self, ActionError> {
        let current = tracker.current().ok_or(ActionError::SnapshotRequired)?;
        validate_dynamic_schema(current.units.as_slice(), tracker.shop())?;
        validate_shop_recipes(tracker.shop())?;
        validate_positions(tracker, current)?;
        let passability = reconstruct_static_passability(tracker, current)?;
        let center = tracker
            .own_hero()
            .or_else(|| tracker.own_courier())
            .map(|unit| unit.pos);
        let entities = build_entity_candidates(tracker, current, center);
        let points = build_point_candidates(tracker, current, &passability, center)?;
        let loot = build_loot_candidates(current);
        let shop = build_shop_candidates(tracker);
        let own_fountain = center.and_then(|position| {
            nearest_landmark(tracker, current, position, UnitKind::Fountain, true)
        });
        let controlled = [
            live_state(&passability, tracker.own_hero()),
            live_state(&passability, tracker.own_courier()),
        ];
        let own_player = tracker.own_player().ok_or(ActionError::SnapshotRequired)?;
        let own_gold = own_player
            .gold
            .ok_or(ActionError::InvalidSchema("own gold"))?;
        let own_stash = own_player
            .stash
            .clone()
            .ok_or(ActionError::InvalidSchema("own stash"))?;
        let allied_buildings = allied_building_positions(tracker, current);
        let buy_requirements = build_buy_requirements(tracker, &shop, &own_stash)?;
        let placeholder =
            ControlledMasks::empty(entities.len(), points.len(), loot.len(), shop.len());
        let mut space = Self {
            provenance: tracker.provenance(),
            controlled,
            entities,
            points,
            loot,
            shop,
            tick: current.tick,
            readiness: *readiness,
            own_gold,
            own_stash,
            own_fountain,
            allied_buildings,
            buy_requirements,
            masks: [placeholder.clone(), placeholder],
            kind_mask: KindMask([false; ACTION_KIND_COUNT]),
        };
        space.masks = [ControlledUnit::Hero, ControlledUnit::Courier]
            .map(|unit| build_controlled_masks(&space, unit));
        space.kind_mask = build_kind_mask(&space.masks);
        Ok(space)
    }

    /// Tick of the snapshot this space was built from.
    pub const fn tick(&self) -> u32 {
        self.tick
    }

    pub(crate) fn matches_tracker(&self, tracker: &StateTracker) -> bool {
        self.provenance.matches(tracker)
    }

    pub(crate) fn matches_readiness(&self, readiness: &ItemReadiness) -> bool {
        self.readiness == *readiness
    }

    pub(crate) const fn feature_frame_provenance(&self) -> crate::FeatureFrameProvenance {
        crate::FeatureFrameProvenance::new(
            self.provenance.lineage(),
            self.provenance.revision(),
            self.tick,
            self.readiness,
        )
    }

    /// Item held by a controlled unit in one of its bag slots, if any.
    pub fn controlled_item(&self, unit: ControlledUnit, slot: ItemSlot) -> Option<ItemView> {
        self.controlled[unit.index()]
            .as_ref()?
            .unit
            .items
            .get(usize::from(slot.0))
            .copied()
            .flatten()
    }

    /// Finds a current candidate by an opaque handle without exposing it as a feature.
    pub fn entity_index(&self, id: EntityId) -> Option<EntityIndex> {
        self.entities
            .iter()
            .position(|candidate| candidate.id == id)
            .map(EntityIndex)
    }

    /// Top-level mask in stable append-only kind order.
    pub const fn kind_mask(&self) -> &KindMask {
        &self.kind_mask
    }

    /// Conditional Hero/Courier mask for one action family.
    pub fn controlled_unit_mask(&self, kind: ActionKind) -> ControlledUnitMask {
        if kind == ActionKind::Continue {
            return ControlledUnitMask([false, false]);
        }
        ControlledUnitMask(std::array::from_fn(|index| {
            self.masks[index].family_allowed(kind)
        }))
    }

    /// Current visible entity candidates, capped at `UNIT_TOKENS`.
    pub fn entity_candidates(&self) -> &[EntityCandidate] {
        &self.entities
    }

    /// Deterministic deduplicated point candidates.
    pub fn point_candidates(&self) -> &[PointCandidate] {
        &self.points
    }

    /// Current visible loot candidates.
    pub fn loot_candidates(&self) -> &[LootCandidate] {
        &self.loot
    }

    /// Static shop candidates in item-id order.
    pub fn shop_candidates(&self) -> &[ShopCandidate] {
        &self.shop
    }

    /// Number of allowed sell slots whose ownership cannot be proved on wire.
    ///
    /// Sell is masked unless `ItemView::for_sale` proves prior server acceptance,
    /// so this compatibility diagnostic is always zero.
    pub const fn sell_ownership_uncertain_count(&self) -> usize {
        0
    }

    /// Whether any allowed sell action has wire-absent ownership.
    ///
    /// This compatibility diagnostic is always false.
    pub const fn has_sell_ownership_uncertainty(&self) -> bool {
        false
    }

    /// Ability-slot mask after selecting a controlled unit.
    pub fn ability_slot_mask(&self, unit: ControlledUnit) -> Vec<bool> {
        self.masks[unit.index()]
            .casts
            .iter()
            .map(TargetMask::any)
            .collect()
    }

    /// Active inventory-slot mask after selecting a controlled unit.
    pub fn item_slot_mask(&self, unit: ControlledUnit) -> Vec<bool> {
        self.masks[unit.index()]
            .uses
            .iter()
            .map(TargetMask::any)
            .collect()
    }

    /// Bag source-slot mask shared by put target modes.
    pub fn put_source_slot_mask(&self, unit: ControlledUnit) -> Vec<bool> {
        let masks = &self.masks[unit.index()];
        masks
            .put_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                point.any()
                    || masks
                        .put_units
                        .get(index)
                        .is_some_and(|units| units.contains(&true))
            })
            .collect()
    }

    /// Underfoot target mask in bag source-slot order.
    pub fn put_underfoot_mask(&self, unit: ControlledUnit) -> Vec<bool> {
        self.masks[unit.index()]
            .put_points
            .iter()
            .map(|mask| mask.underfoot)
            .collect()
    }

    /// Point pointer mask after selecting a bag source slot.
    pub fn put_point_target_mask(&self, unit: ControlledUnit, source: ItemSlot) -> Option<&[bool]> {
        self.masks[unit.index()]
            .put_points
            .get(usize::from(source.0))
            .map(|mask| mask.points.as_slice())
    }

    /// Allied bag-carrier pointer mask after selecting a bag source slot.
    pub fn put_entity_target_mask(
        &self,
        unit: ControlledUnit,
        source: ItemSlot,
    ) -> Option<&[bool]> {
        self.masks[unit.index()]
            .put_units
            .get(usize::from(source.0))
            .map(Vec::as_slice)
    }

    /// Target mask after selecting a controlled unit and ability slot.
    pub fn cast_target_mask(&self, unit: ControlledUnit, slot: AbilitySlot) -> Option<&TargetMask> {
        self.masks[unit.index()].casts.get(usize::from(slot.0))
    }

    /// Target mask after selecting a controlled unit and active item slot.
    pub fn use_target_mask(&self, unit: ControlledUnit, slot: ItemSlot) -> Option<&TargetMask> {
        self.masks[unit.index()].uses.get(usize::from(slot.0))
    }

    /// Entity target mask for follow orders.
    pub fn follow_entity_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].follow_entities
    }

    /// Entity target mask for attack orders.
    pub fn attack_entity_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].attack_entities
    }

    /// Point target mask for move orders.
    pub fn move_point_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].move_points
    }

    /// Point target mask for attack-move orders.
    pub fn attack_move_point_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].attack_move_points
    }

    /// Loot mask after selecting a controlled unit.
    pub fn take_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].take
    }

    /// Shop mask after selecting a controlled unit.
    ///
    /// The current server sends remote buys to stash. The selected body only
    /// receives a purchase directly while it is within the local shop range.
    pub fn buy_mask(&self, unit: ControlledUnit) -> &[bool] {
        &self.masks[unit.index()].buy
    }

    /// Sell slots whose `for_sale` state proves prior server ownership acceptance.
    pub const fn sell_slot_mask(&self, unit: ControlledUnit) -> &[bool; WIRE_ITEM_SLOTS] {
        &self.masks[unit.index()].sell
    }

    /// Destination slot mask after selecting a nonempty swap source.
    pub fn swap_destination_mask(
        &self,
        unit: ControlledUnit,
        from: ItemSlot,
    ) -> Option<&[bool; WIRE_ITEM_SLOTS]> {
        self.masks[unit.index()].swap.get(usize::from(from.0))
    }

    /// Hero ability slots that currently report `can_level`.
    pub fn learn_slot_mask(&self) -> &[bool] {
        &self.masks[ControlledUnit::Hero.index()].learn
    }

    /// Whether the exact complete structured action is legal in this snapshot.
    pub fn allows(&self, action: StructuredAction) -> bool {
        if !self.kind_mask.allows(action.kind()) {
            return false;
        }
        self.allows_conditionals(action)
    }

    /// Validates the exact action and converts candidate indices to wire values.
    pub fn decode(&self, action: StructuredAction) -> Result<Option<IssuedOrder>, ActionError> {
        self.validate_candidate_indices(action)?;
        if !self.allows(action) {
            return Err(ActionError::NotAllowed(action.kind()));
        }
        if action == StructuredAction::Continue {
            return Ok(None);
        }
        let unit = action
            .controlled_unit()
            .map(|selected| self.wire_unit(selected))
            .transpose()?
            .flatten();
        let order = self.decode_order(action)?;
        Ok(Some(IssuedOrder { unit, order }))
    }

    fn allows_conditionals(&self, action: StructuredAction) -> bool {
        match action {
            StructuredAction::Continue => true,
            StructuredAction::Stop { unit } => self.masks[unit.index()].stop,
            StructuredAction::MovePoint { unit, point } => {
                mask_at(&self.masks[unit.index()].move_points, point.0)
            }
            StructuredAction::FollowUnit { unit, target } => {
                mask_at(&self.masks[unit.index()].follow_entities, target.0)
            }
            StructuredAction::Hold { unit } => self.masks[unit.index()].hold,
            StructuredAction::AttackMovePoint { unit, point } => {
                mask_at(&self.masks[unit.index()].attack_move_points, point.0)
            }
            StructuredAction::AttackUnit { unit, target } => {
                mask_at(&self.masks[unit.index()].attack_entities, target.0)
            }
            StructuredAction::Cast { unit, slot, target } => self
                .cast_target_mask(unit, slot)
                .is_some_and(|mask| mask.allows(target)),
            StructuredAction::Use { unit, slot, target } => self
                .use_target_mask(unit, slot)
                .is_some_and(|mask| mask.allows(target)),
            StructuredAction::PutPoint {
                unit,
                source,
                target,
            } => allows_put_point(&self.masks[unit.index()], source, target),
            StructuredAction::PutUnit {
                unit,
                source,
                target,
            } => allows_put_unit(&self.masks[unit.index()], source, target),
            StructuredAction::Take { unit, loot } => {
                mask_at(&self.masks[unit.index()].take, loot.0)
            }
            StructuredAction::Buy { unit, item } => mask_at(&self.masks[unit.index()].buy, item.0),
            StructuredAction::Sell { unit, slot } => self.masks[unit.index()]
                .sell
                .get(usize::from(slot.0))
                .copied()
                .unwrap_or(false),
            StructuredAction::Swap { unit, from, to } => self.masks[unit.index()]
                .swap
                .get(usize::from(from.0))
                .and_then(|row| row.get(usize::from(to.0)))
                .copied()
                .unwrap_or(false),
            StructuredAction::Learn { slot } => self
                .learn_slot_mask()
                .get(usize::from(slot.0))
                .copied()
                .unwrap_or(false),
        }
    }

    fn validate_candidate_indices(&self, action: StructuredAction) -> Result<(), ActionError> {
        match action {
            StructuredAction::MovePoint { point, .. }
            | StructuredAction::AttackMovePoint { point, .. } => {
                check_candidate("point target", point.0, self.points.len())
            }
            StructuredAction::FollowUnit { target, .. }
            | StructuredAction::AttackUnit { target, .. }
            | StructuredAction::PutUnit { target, .. } => {
                check_candidate("entity target", target.0, self.entities.len())
            }
            StructuredAction::Cast { target, .. } | StructuredAction::Use { target, .. } => {
                self.validate_target_index(target)
            }
            StructuredAction::PutPoint {
                target: PutPointTarget::Point(point),
                ..
            } => check_candidate("point target", point.0, self.points.len()),
            StructuredAction::Take { loot, .. } => {
                check_candidate("loot target", loot.0, self.loot.len())
            }
            StructuredAction::Buy { item, .. } => {
                check_candidate("shop item", item.0, self.shop.len())
            }
            _ => Ok(()),
        }
    }

    fn validate_target_index(&self, target: ActionTarget) -> Result<(), ActionError> {
        match target {
            ActionTarget::None => Ok(()),
            ActionTarget::Entity(index) => {
                check_candidate("entity target", index.0, self.entities.len())
            }
            ActionTarget::Point(index) => {
                check_candidate("point target", index.0, self.points.len())
            }
        }
    }

    fn wire_unit(&self, unit: ControlledUnit) -> Result<Option<EntityId>, ActionError> {
        match unit {
            ControlledUnit::Hero => Ok(None),
            ControlledUnit::Courier => self.controlled[unit.index()]
                .as_ref()
                .map(|state| Some(state.id))
                .ok_or(ActionError::NotAllowed(ActionKind::Stop)),
        }
    }

    fn decode_order(&self, action: StructuredAction) -> Result<Order, ActionError> {
        let order = match action {
            StructuredAction::Continue => {
                return Err(ActionError::NotAllowed(ActionKind::Continue));
            }
            StructuredAction::Stop { .. } => Order::Move {
                target: Target::None,
            },
            StructuredAction::MovePoint { point, .. } => Order::Move {
                target: Target::Pos(self.points[point.0].position),
            },
            StructuredAction::FollowUnit { target, .. } => Order::Move {
                target: Target::Unit(self.entities[target.0].id),
            },
            StructuredAction::Hold { .. } => Order::Attack {
                target: Target::None,
            },
            StructuredAction::AttackMovePoint { point, .. } => Order::Attack {
                target: Target::Pos(self.points[point.0].position),
            },
            StructuredAction::AttackUnit { target, .. } => Order::Attack {
                target: Target::Unit(self.entities[target.0].id),
            },
            StructuredAction::Cast { slot, target, .. } => Order::Cast {
                slot,
                target: self.decode_target(target),
            },
            StructuredAction::Use { slot, target, .. } => Order::Use {
                slot,
                target: self.decode_target(target),
            },
            StructuredAction::PutPoint { source, target, .. } => Order::Put {
                slot: source,
                target: self.decode_put_point(target),
            },
            StructuredAction::PutUnit { source, target, .. } => Order::Put {
                slot: source,
                target: Target::Unit(self.entities[target.0].id),
            },
            StructuredAction::Take { loot, .. } => Order::Take {
                target: Target::Unit(self.loot[loot.0].id),
            },
            StructuredAction::Buy { item, .. } => Order::Buy {
                item: self.shop[item.0].item,
            },
            StructuredAction::Sell { slot, .. } => Order::Sell { slot },
            StructuredAction::Swap { from, to, .. } => Order::Swap { from, to },
            StructuredAction::Learn { slot } => Order::Learn { slot },
        };
        Ok(order)
    }

    fn decode_target(&self, target: ActionTarget) -> Target {
        match target {
            ActionTarget::None => Target::None,
            ActionTarget::Entity(index) => Target::Unit(self.entities[index.0].id),
            ActionTarget::Point(index) => Target::Pos(self.points[index.0].position),
        }
    }

    fn decode_put_point(&self, target: PutPointTarget) -> Target {
        match target {
            PutPointTarget::Underfoot => Target::None,
            PutPointTarget::Point(index) => Target::Pos(self.points[index.0].position),
        }
    }
}

fn validate_dynamic_schema(
    units: &[UnitView],
    shop: &[bota_proto::ShopEntry],
) -> Result<(), ActionError> {
    for unit in units {
        if unit.radius.raw < 0
            || unit.radius.raw > Fixed::MAX.raw - Fixed::from_int(STRUCTURE_CLEARANCE).raw
        {
            return Err(ActionError::InvalidSchema("UnitView radius"));
        }
        for ability in &unit.abilities {
            if !(0..=Fixed::MAX.to_int()).contains(&ability.range) || ability.mana_cost < 0 {
                return Err(ActionError::InvalidSchema("AbilityView range or mana cost"));
            }
        }
        for item in unit.items.iter().flatten() {
            if !(0..=Fixed::MAX.to_int()).contains(&item.range) || item.mana_cost < 0 {
                return Err(ActionError::InvalidSchema("ItemView range or mana cost"));
            }
        }
    }
    if shop.iter().any(|entry| entry.cost < 0) {
        return Err(ActionError::InvalidSchema("ShopEntry cost"));
    }
    Ok(())
}

fn validate_shop_recipes(shop: &[ShopEntry]) -> Result<(), ActionError> {
    if shop
        .iter()
        .enumerate()
        .any(|(index, entry)| shop[..index].iter().any(|other| other.id == entry.id))
    {
        return Err(ActionError::InvalidSchema("duplicate shop item"));
    }
    let mut colors = vec![0u8; shop.len()];
    for root in 0..shop.len() {
        if colors[root] == 0 {
            validate_recipe_from(shop, root, &mut colors)?;
        }
    }
    Ok(())
}

fn validate_recipe_from(
    shop: &[ShopEntry],
    root: usize,
    colors: &mut [u8],
) -> Result<(), ActionError> {
    let mut stack = vec![(root, 0usize)];
    colors[root] = 1;
    while let Some((entry_index, component_index)) = stack.last_mut() {
        let components = &shop[*entry_index].components;
        if *component_index == components.len() {
            colors[*entry_index] = 2;
            stack.pop();
            continue;
        }
        let component = components[*component_index];
        *component_index += 1;
        let Some(next) = shop.iter().position(|entry| entry.id == component) else {
            return Err(ActionError::InvalidSchema("unknown recipe component"));
        };
        match colors[next] {
            0 => {
                colors[next] = 1;
                stack.push((next, 0));
            }
            1 => return Err(ActionError::InvalidSchema("cyclic shop recipe")),
            2 => {}
            _ => return Err(ActionError::InvalidSchema("recipe traversal state")),
        }
    }
    Ok(())
}

fn validate_positions(tracker: &StateTracker, current: &WorldView) -> Result<(), ActionError> {
    let maximum = map_maximum_raw(tracker)?;
    for position in tracker.static_trees() {
        validate_position(*position, maximum, "MatchInfo tree position")?;
    }
    for unit in &current.units {
        validate_position(unit.pos, maximum, "UnitView position")?;
    }
    for position in &current.planted_trees {
        validate_position(*position, maximum, "planted tree position")?;
    }
    for loot in &current.loot {
        validate_position(loot.pos, maximum, "LootView position")?;
    }
    Ok(())
}

fn validate_position(position: Vec2, maximum: i64, field: &'static str) -> Result<(), ActionError> {
    let x = i64::from(position.x.raw);
    let y = i64::from(position.y.raw);
    if x < 0 || y < 0 || x > maximum || y > maximum {
        return Err(ActionError::InvalidSchema(field));
    }
    Ok(())
}

fn reconstruct_static_passability(
    tracker: &StateTracker,
    current: &WorldView,
) -> Result<StaticPassability, ActionError> {
    let axis = usize::try_from(tracker.metadata().terrain_cells)
        .map_err(|_| ActionError::InvalidSchema("terrain axis"))?;
    let mut terrain = Vec::with_capacity(axis * axis);
    for &(run, cell) in tracker.terrain_rle() {
        terrain.resize(terrain.len() + usize::from(run), cell);
    }
    debug_assert_eq!(terrain.len(), axis * axis);
    let open = terrain.iter().map(|cell| cell & 0x80 != 0).collect();
    let mut passability = StaticPassability { axis, open };
    for (index, position) in tracker.static_trees().iter().copied().enumerate() {
        let index = u32::try_from(index).map_err(|_| ActionError::Arithmetic("tree index"))?;
        let locally_felled = current.felled_trees.contains(&index)
            && tracker.position_locally_observable_to_own_seat(position);
        if !locally_felled {
            passability.block_circle(position, Fixed::from_int(STATIC_TREE_CLEARANCE));
        }
    }
    for position in current.planted_trees.iter().copied() {
        if tracker.position_locally_observable_to_own_seat(position) {
            passability.block_circle(position, Fixed::from_int(STATIC_TREE_CLEARANCE));
        }
    }
    for structure in current.units.iter().filter(|unit| is_structure(unit.kind)) {
        let clearance_raw = structure.radius.raw + Fixed::from_int(STRUCTURE_CLEARANCE).raw;
        passability.block_circle(structure.pos, Fixed { raw: clearance_raw });
    }
    Ok(passability)
}

impl StaticPassability {
    fn walkable(&self, position: Vec2) -> bool {
        let Some((cell_x, cell_y)) = self.cell_of(position) else {
            return false;
        };
        self.open[cell_y * self.axis + cell_x]
    }

    fn cell_of(&self, position: Vec2) -> Option<(usize, usize)> {
        if position.x.raw < 0 || position.y.raw < 0 {
            return None;
        }
        let cell_x = usize::try_from(position.x.to_int() / TERRAIN_CELL_SIZE).ok()?;
        let cell_y = usize::try_from(position.y.to_int() / TERRAIN_CELL_SIZE).ok()?;
        (cell_x < self.axis && cell_y < self.axis).then_some((cell_x, cell_y))
    }

    fn cell_center(&self, cell_x: usize, cell_y: usize) -> Vec2 {
        let half = TERRAIN_CELL_SIZE / 2;
        Vec2::from_ints(
            i32::try_from(cell_x).unwrap_or(i32::MAX) * TERRAIN_CELL_SIZE + half,
            i32::try_from(cell_y).unwrap_or(i32::MAX) * TERRAIN_CELL_SIZE + half,
        )
    }

    fn block_circle(&mut self, center: Vec2, radius: Fixed) {
        let span = usize::try_from(radius.to_int() / TERRAIN_CELL_SIZE + 1).unwrap_or(self.axis);
        let Some((center_x, center_y)) = self.cell_of(center) else {
            return;
        };
        let start_x = center_x.saturating_sub(span);
        let start_y = center_y.saturating_sub(span);
        let end_x = center_x.saturating_add(span).min(self.axis - 1);
        let end_y = center_y.saturating_add(span).min(self.axis - 1);
        for cell_y in start_y..=end_y {
            for cell_x in start_x..=end_x {
                if self.cell_center(cell_x, cell_y).within(center, radius) {
                    self.open[cell_y * self.axis + cell_x] = false;
                }
            }
        }
    }
}

fn live_state(passability: &StaticPassability, unit: Option<&UnitView>) -> Option<ControlledState> {
    let unit = unit?;
    if unit.hp <= 0 || has_status(unit, StatusFlags::DEAD) {
        return None;
    }
    Some(ControlledState {
        id: unit.id,
        unit: unit.clone(),
        underfoot_walkable: passability.walkable(unit.pos),
    })
}

fn build_entity_candidates(
    tracker: &StateTracker,
    current: &WorldView,
    center: Option<Vec2>,
) -> Vec<EntityCandidate> {
    let mut selected: Vec<&UnitView> = current
        .units
        .iter()
        .filter(|unit| unit.hp > 0 && !has_status(unit, StatusFlags::DEAD))
        .collect();
    selected.sort_by_key(|unit| entity_priority(tracker, unit, center));
    selected.truncate(UNIT_TOKENS);
    let mut output = Vec::with_capacity(UNIT_TOKENS);
    for unit in selected {
        output.push(EntityCandidate {
            id: unit.id,
            unit: unit.clone(),
            kind: unit.kind,
            relation: entity_relation(tracker, unit),
            position: unit.pos,
        });
    }
    output
}

fn entity_priority<'a>(
    tracker: &'a StateTracker,
    unit: &'a UnitView,
    center: Option<Vec2>,
) -> EntityPriority<'a> {
    let own_body = unit.owner == Some(tracker.slot())
        && matches!(unit.kind, UnitKind::Hero | UnitKind::Courier);
    let structure = is_structure(unit.kind);
    let distance = center.map_or(i64::MAX, |origin| origin.distance_squared(unit.pos));
    let combat_relevant = distance <= Fixed::from_int(1_200).squared_raw();
    let priority = if own_body {
        0
    } else if unit.kind == UnitKind::Hero {
        1
    } else if structure {
        2
    } else if combat_relevant {
        3
    } else {
        4
    };
    EntityPriority {
        tracker,
        priority,
        distance,
        relation: entity_relation(tracker, unit),
        owner_relation: candidate_owner_relation(tracker, unit),
        unit,
        track: tracker.entity(unit.id),
    }
}

struct EntityPriority<'a> {
    tracker: &'a StateTracker,
    priority: u8,
    distance: i64,
    relation: EntityRelation,
    owner_relation: Option<EntityRelation>,
    unit: &'a UnitView,
    track: Option<&'a crate::EntityTrack>,
}

impl Ord for EntityPriority<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.distance.cmp(&other.distance))
            .then_with(|| relation_index(self.relation).cmp(&relation_index(other.relation)))
            .then_with(|| {
                self.owner_relation
                    .map(relation_index)
                    .cmp(&other.owner_relation.map(relation_index))
            })
            .then_with(|| compare_units(self.tracker, self.unit, other.unit))
            .then_with(|| compare_track_features(self.tracker, self.track, other.track))
            .then_with(|| self.unit.id.cmp(&other.unit.id))
    }
}

impl PartialOrd for EntityPriority<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EntityPriority<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for EntityPriority<'_> {}

fn compare_units(tracker: &StateTracker, left: &UnitView, right: &UnitView) -> std::cmp::Ordering {
    left.kind
        .cmp(&right.kind)
        .then_with(|| {
            canonical_position_key(tracker, left.pos)
                .cmp(&canonical_position_key(tracker, right.pos))
        })
        .then_with(|| canonical_facing(tracker, left).cmp(&canonical_facing(tracker, right)))
        .then_with(|| left.hp.cmp(&right.hp))
        .then_with(|| left.max_hp.cmp(&right.max_hp))
        .then_with(|| left.mana.cmp(&right.mana))
        .then_with(|| left.max_mana.cmp(&right.max_mana))
        .then_with(|| left.move_speed.cmp(&right.move_speed))
        .then_with(|| left.attack_damage.cmp(&right.attack_damage))
        .then_with(|| left.attack_range.cmp(&right.attack_range))
        .then_with(|| left.attack_interval.cmp(&right.attack_interval))
        .then_with(|| left.attack_speed.cmp(&right.attack_speed))
        .then_with(|| left.armor.cmp(&right.armor))
        .then_with(|| left.magic_resist.cmp(&right.magic_resist))
        .then_with(|| left.radius.cmp(&right.radius))
        .then_with(|| left.vision_radius.cmp(&right.vision_radius))
        .then_with(|| left.true_sight_radius.cmp(&right.true_sight_radius))
        .then_with(|| left.statuses.cmp(&right.statuses))
        .then_with(|| item_capacity_key(left).cmp(&item_capacity_key(right)))
}

fn item_capacity_key(unit: &UnitView) -> (usize, usize, bool) {
    let free = unit.items.iter().filter(|slot| slot.is_none()).count();
    (unit.items.len(), free, free > 0)
}

fn compare_track_features(
    tracker: &StateTracker,
    left: Option<&crate::EntityTrack>,
    right: Option<&crate::EntityTrack>,
) -> std::cmp::Ordering {
    let left = left.map(|track| track_semantic_key(tracker, track));
    let right = right.map(|track| track_semantic_key(tracker, track));
    left.cmp(&right)
}

type TrackSemanticKey = (
    Option<(i64, i64, u32)>,
    i64,
    i64,
    Option<(u32, i32, bota_proto::DamageKind, bool)>,
    Option<(u32, i32, bota_proto::DamageKind, bool)>,
    Option<(u32, bota_proto::AbilityId)>,
    Option<u32>,
);

fn track_semantic_key(tracker: &StateTracker, track: &crate::EntityTrack) -> TrackSemanticKey {
    (
        track.velocity.map(|velocity| {
            let side = if tracker.team() == Team::Dire {
                -1i64
            } else {
                1
            };
            (
                i64::from(velocity.delta.x.raw) * side,
                i64::from(velocity.delta.y.raw) * side,
                velocity.elapsed_ticks,
            )
        }),
        track.hp_delta,
        track.mana_delta,
        track
            .last_damage_dealt
            .map(|value| (value.tick, value.amount, value.kind, value.crit)),
        track
            .last_damage_taken
            .map(|value| (value.tick, value.amount, value.kind, value.crit)),
        track
            .last_ability_cast
            .map(|value| (value.tick, value.ability)),
        track.last_possible_attack_landed.map(|value| value.tick),
    )
}

fn build_point_candidates(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    center: Option<Vec2>,
) -> Result<Vec<PointCandidate>, ActionError> {
    let mut points = Vec::with_capacity(MAX_POINT_CANDIDATES);
    if let Some(center) = center {
        add_tactical_points(tracker, passability, center, &mut points)?;
        add_building_landing_points(tracker, current, passability, center, &mut points);
        add_tree_points(tracker, current, passability, center, &mut points)?;
        add_landmark_points(tracker, current, passability, center, &mut points);
        add_predicted_points(tracker, current, passability, center, &mut points)?;
    }
    Ok(points)
}

fn add_tactical_points(
    tracker: &StateTracker,
    passability: &StaticPassability,
    center: Vec2,
    points: &mut Vec<PointCandidate>,
) -> Result<(), ActionError> {
    for radius in TACTICAL_RADII {
        for direction in all_directions() {
            let offset = direction_offset(tracker.team(), direction, radius)?;
            let position = clamp_offset(tracker, center, offset)?;
            push_point(
                points,
                PointCandidate {
                    position,
                    source: PointSource::Tactical { direction, radius },
                    walkable: passability.walkable(position),
                    standing_tree: false,
                    allied_building: false,
                },
            );
        }
    }
    Ok(())
}

fn add_building_landing_points(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    center: Vec2,
    points: &mut Vec<PointCandidate>,
) {
    let mut landings = current
        .units
        .iter()
        .filter(|unit| unit.team == tracker.team() && is_structure(unit.kind))
        .filter_map(|unit| {
            let position = nearest_landing_cell(passability, unit.pos, tracker.team())?;
            Some((
                center.distance_squared(position),
                unit.kind,
                position,
                unit.id,
            ))
        })
        .collect::<Vec<_>>();
    landings.sort_by_key(|(distance, kind, position, id)| {
        (
            *distance,
            *kind,
            canonical_position_key(tracker, *position),
            *id,
        )
    });
    for (_, kind, position, _) in landings {
        push_point(
            points,
            PointCandidate {
                position,
                source: PointSource::BuildingLanding(kind),
                walkable: true,
                standing_tree: false,
                allied_building: true,
            },
        );
        if points.len() == MAX_POINT_CANDIDATES {
            break;
        }
    }
}

fn nearest_landing_cell(passability: &StaticPassability, center: Vec2, team: Team) -> Option<Vec2> {
    let (center_x, center_y) = passability.cell_of(center)?;
    let mut best: Option<(i64, usize, Vec2)> = None;
    let start_y = center_y.saturating_sub(LANDING_SEARCH_CELLS);
    let start_x = center_x.saturating_sub(LANDING_SEARCH_CELLS);
    let end_y = center_y
        .saturating_add(LANDING_SEARCH_CELLS)
        .min(passability.axis - 1);
    let end_x = center_x
        .saturating_add(LANDING_SEARCH_CELLS)
        .min(passability.axis - 1);
    for cell_y in start_y..=end_y {
        for cell_x in start_x..=end_x {
            let position = canonical_cell_position(passability, cell_x, cell_y, team);
            if !passability.walkable(position) {
                continue;
            }
            let distance = center.distance_squared(position);
            let cell_index = cell_y * passability.axis + cell_x;
            let cell_index = if team == Team::Dire {
                passability.axis * passability.axis - 1 - cell_index
            } else {
                cell_index
            };
            if best.is_none_or(|current| (distance, cell_index) < (current.0, current.1)) {
                best = Some((distance, cell_index, position));
            }
        }
    }
    best.map(|(_, _, position)| position)
}

fn canonical_cell_position(
    passability: &StaticPassability,
    cell_x: usize,
    cell_y: usize,
    team: Team,
) -> Vec2 {
    if team != Team::Dire {
        return passability.cell_center(cell_x, cell_y);
    }
    let canonical =
        passability.cell_center(passability.axis - 1 - cell_x, passability.axis - 1 - cell_y);
    let maximum = i64::try_from(passability.axis).expect("validated terrain axis")
        * i64::from(TERRAIN_CELL_SIZE)
        * i64::from(Fixed::ONE.raw)
        - 1;
    Vec2 {
        x: Fixed {
            raw: i32::try_from(maximum - i64::from(canonical.x.raw)).expect("validated map extent"),
        },
        y: Fixed {
            raw: i32::try_from(maximum - i64::from(canonical.y.raw)).expect("validated map extent"),
        },
    }
}

fn add_tree_points(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    center: Vec2,
    points: &mut Vec<PointCandidate>,
) -> Result<(), ActionError> {
    let mut trees = Vec::with_capacity(tracker.static_trees().len() + current.planted_trees.len());
    for (index, position) in tracker.static_trees().iter().copied().enumerate() {
        let index = u32::try_from(index).map_err(|_| ActionError::Arithmetic("tree index"))?;
        let locally_observable = tracker.position_locally_observable_to_own_seat(position);
        if !current.felled_trees.contains(&index) || !locally_observable {
            trees.push((center.distance_squared(position), position, false));
        }
    }
    for position in current.planted_trees.iter().copied() {
        if tracker.position_locally_observable_to_own_seat(position) {
            trees.push((center.distance_squared(position), position, true));
        }
    }
    trees.sort_by_key(|(distance, position, planted)| {
        (
            *distance,
            canonical_position_key(tracker, *position),
            *planted,
        )
    });
    for (_, position, planted) in trees.into_iter().take(NEARBY_TREE_POINTS) {
        let position = clamp_position(tracker, position)?;
        push_point(
            points,
            PointCandidate {
                position,
                source: if planted {
                    PointSource::PlantedTree
                } else {
                    PointSource::StaticTree
                },
                walkable: passability.walkable(position),
                standing_tree: true,
                allied_building: false,
            },
        );
        if points.len() == MAX_POINT_CANDIDATES {
            break;
        }
    }
    Ok(())
}

fn add_landmark_points(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    center: Vec2,
    points: &mut Vec<PointCandidate>,
) {
    for (kind, source) in [
        (UnitKind::Fountain, true),
        (UnitKind::Fountain, false),
        (UnitKind::Tower, true),
        (UnitKind::Tower, false),
    ] {
        let landmark = nearest_landmark(tracker, current, center, kind, source);
        if let Some(position) = landmark {
            let relation = if source {
                LandmarkRelation::Own
            } else {
                LandmarkRelation::Enemy
            };
            let source = if kind == UnitKind::Fountain {
                PointSource::Fountain(relation)
            } else {
                PointSource::Tower(relation)
            };
            push_point(
                points,
                PointCandidate {
                    position,
                    source,
                    walkable: passability.walkable(position),
                    standing_tree: false,
                    allied_building: false,
                },
            );
        }
    }
}

fn add_predicted_points(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    center: Vec2,
    points: &mut Vec<PointCandidate>,
) -> Result<(), ActionError> {
    let mut predicted = Vec::with_capacity(current.units.len());
    for unit in &current.units {
        let Some(velocity) = tracker.entity(unit.id).and_then(|track| track.velocity) else {
            continue;
        };
        if velocity.delta == Vec2::ZERO {
            continue;
        }
        let source = if unit.kind == UnitKind::Hero {
            Some(PointSource::PredictedHero(entity_relation(tracker, unit)))
        } else if is_creep(unit.kind) {
            Some(PointSource::PredictedCreep(entity_relation(tracker, unit)))
        } else {
            None
        };
        let Some(source) = source else {
            continue;
        };
        debug_assert_ne!(velocity.elapsed_ticks, 0);
        let position = clamp_offset(tracker, unit.pos, velocity.delta)?;
        predicted.push((
            center.distance_squared(position),
            predicted_source_index(source),
            position,
            unit.id,
            source,
        ));
    }
    predicted.sort_by_key(|(distance, source, position, id, _)| {
        (
            *distance,
            *source,
            canonical_position_key(tracker, *position),
            *id,
        )
    });
    let mut heroes = 0usize;
    for (_, _, position, _, source) in predicted {
        if matches!(source, PointSource::PredictedHero(_)) {
            if heroes == PREDICTED_HERO_POINTS {
                continue;
            }
            heroes += 1;
        }
        push_point(
            points,
            PointCandidate {
                position,
                source,
                walkable: passability.walkable(position),
                standing_tree: false,
                allied_building: false,
            },
        );
        if points.len() == MAX_POINT_CANDIDATES {
            break;
        }
    }
    Ok(())
}

const fn predicted_source_index(source: PointSource) -> u8 {
    match source {
        PointSource::PredictedHero(relation) => relation_index(relation),
        PointSource::PredictedCreep(relation) => 4 + relation_index(relation),
        _ => u8::MAX,
    }
}

fn all_directions() -> [PointDirection; 8] {
    [
        PointDirection::East,
        PointDirection::NorthEast,
        PointDirection::North,
        PointDirection::NorthWest,
        PointDirection::West,
        PointDirection::SouthWest,
        PointDirection::South,
        PointDirection::SouthEast,
    ]
}

fn direction_offset(
    team: Team,
    direction: PointDirection,
    radius: i32,
) -> Result<Vec2, ActionError> {
    let diagonal = i64::from(radius)
        .checked_mul(46_341)
        .ok_or(ActionError::Arithmetic("diagonal point"))?
        / 65_536;
    let diagonal =
        i32::try_from(diagonal).map_err(|_| ActionError::Arithmetic("diagonal point"))?;
    let (x, y) = match direction {
        PointDirection::East => (radius, 0),
        PointDirection::NorthEast => (diagonal, diagonal),
        PointDirection::North => (0, radius),
        PointDirection::NorthWest => (-diagonal, diagonal),
        PointDirection::West => (-radius, 0),
        PointDirection::SouthWest => (-diagonal, -diagonal),
        PointDirection::South => (0, -radius),
        PointDirection::SouthEast => (diagonal, -diagonal),
    };
    let side = if team == Team::Dire { -1 } else { 1 };
    Ok(Vec2::from_ints(x * side, y * side))
}

fn clamp_offset(tracker: &StateTracker, position: Vec2, offset: Vec2) -> Result<Vec2, ActionError> {
    let x = i64::from(position.x.raw)
        .checked_add(i64::from(offset.x.raw))
        .ok_or(ActionError::Arithmetic("point x"))?;
    let y = i64::from(position.y.raw)
        .checked_add(i64::from(offset.y.raw))
        .ok_or(ActionError::Arithmetic("point y"))?;
    clamp_raw_position(tracker, x, y)
}

fn clamp_position(tracker: &StateTracker, position: Vec2) -> Result<Vec2, ActionError> {
    clamp_raw_position(
        tracker,
        i64::from(position.x.raw),
        i64::from(position.y.raw),
    )
}

fn clamp_raw_position(tracker: &StateTracker, x: i64, y: i64) -> Result<Vec2, ActionError> {
    let maximum = map_maximum_raw(tracker)?;
    let x = i32::try_from(x.clamp(0, maximum))
        .map_err(|_| ActionError::Arithmetic("clamped point x"))?;
    let y = i32::try_from(y.clamp(0, maximum))
        .map_err(|_| ActionError::Arithmetic("clamped point y"))?;
    Ok(Vec2 {
        x: Fixed { raw: x },
        y: Fixed { raw: y },
    })
}

fn map_maximum_raw(tracker: &StateTracker) -> Result<i64, ActionError> {
    let world_units = i64::from(tracker.metadata().terrain_cells)
        .checked_mul(i64::from(TERRAIN_CELL_SIZE))
        .ok_or(ActionError::Arithmetic("map world size"))?;
    let exclusive_raw = world_units
        .checked_mul(i64::from(Fixed::ONE.raw))
        .ok_or(ActionError::Arithmetic("map fixed size"))?;
    let maximum = exclusive_raw
        .checked_sub(1)
        .ok_or(ActionError::InvalidSchema("terrain extent"))?;
    Ok(maximum.min(i64::from(i32::MAX)))
}

fn canonical_position_key(tracker: &StateTracker, position: Vec2) -> (i64, i64) {
    if tracker.team() == Team::Dire {
        let maximum = crate::tracker::map_maximum_raw(tracker.metadata().terrain_cells);
        (
            maximum - i64::from(position.x.raw),
            maximum - i64::from(position.y.raw),
        )
    } else {
        (i64::from(position.x.raw), i64::from(position.y.raw))
    }
}

fn canonical_facing(tracker: &StateTracker, unit: &UnitView) -> u16 {
    if tracker.team() == Team::Dire {
        unit.facing.brads.wrapping_add(1 << 15)
    } else {
        unit.facing.brads
    }
}

fn push_point(points: &mut Vec<PointCandidate>, candidate: PointCandidate) {
    if let Some(existing) = points
        .iter_mut()
        .find(|point| point.position == candidate.position)
    {
        existing.walkable = existing.walkable && candidate.walkable;
        existing.standing_tree |= candidate.standing_tree;
        existing.allied_building |= candidate.allied_building;
        return;
    }
    if points.len() == MAX_POINT_CANDIDATES {
        return;
    }
    points.push(candidate);
}

fn nearest_landmark(
    tracker: &StateTracker,
    current: &WorldView,
    center: Vec2,
    kind: UnitKind,
    own: bool,
) -> Option<Vec2> {
    current
        .units
        .iter()
        .filter(|unit| unit.kind == kind)
        .filter(|unit| (unit.team == tracker.team()) == own)
        .min_by_key(|unit| {
            (
                center.distance_squared(unit.pos),
                canonical_position_key(tracker, unit.pos),
                unit.id,
            )
        })
        .map(|unit| unit.pos)
}

fn allied_building_positions(tracker: &StateTracker, current: &WorldView) -> Vec<Vec2> {
    current
        .units
        .iter()
        .filter(|unit| unit.team == tracker.team() && is_structure(unit.kind))
        .map(|unit| unit.pos)
        .collect()
}

fn build_loot_candidates(current: &bota_proto::WorldView) -> Vec<LootCandidate> {
    let mut output: Vec<_> = current
        .loot
        .iter()
        .map(|candidate| LootCandidate {
            id: candidate.id,
            position: candidate.pos,
            item: candidate.item,
            charges: candidate.charges,
        })
        .collect();
    output.sort_by_key(|candidate| {
        (
            candidate.item,
            candidate.charges,
            candidate.position,
            candidate.id,
        )
    });
    output
}

fn build_shop_candidates(tracker: &StateTracker) -> Vec<ShopCandidate> {
    let mut output: Vec<_> = tracker
        .shop()
        .iter()
        .map(|entry| ShopCandidate {
            item: entry.id,
            cost: entry.cost,
        })
        .collect();
    output.sort_by_key(|candidate| candidate.item);
    output
}

fn build_buy_requirements(
    tracker: &StateTracker,
    candidates: &[ShopCandidate],
    stash: &[Option<ItemView>],
) -> Result<Vec<BuyRequirement>, ActionError> {
    let mut held: Vec<ItemId> = tracker
        .own_hero()
        .into_iter()
        .flat_map(|hero| hero.items.iter().take(HERO_BAG_SLOTS).flatten())
        .chain(stash.iter().flatten())
        .map(|item| item.id)
        .collect();
    let held_template = held.clone();
    let mut output = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        held.clone_from(&held_template);
        output.push(missing_buy_requirement(
            tracker.shop(),
            candidate.item,
            &mut held,
        )?);
    }
    Ok(output)
}

fn missing_buy_requirement(
    shop: &[ShopEntry],
    item: ItemId,
    held: &mut Vec<ItemId>,
) -> Result<BuyRequirement, ActionError> {
    let root = shop_entry(shop, item)?;
    if root.components.is_empty() {
        return Ok(BuyRequirement {
            missing_cost: root.cost,
            missing_slots: 1,
        });
    }
    let mut stack = root.components.iter().rev().copied().collect();
    let edge_count = shop.iter().try_fold(0usize, |count, entry| {
        count
            .checked_add(entry.components.len())
            .ok_or(ActionError::Arithmetic("recipe edge count"))
    })?;
    let expansion_limit = (MAX_PURCHASE_SLOTS + held.len() + 1)
        .checked_mul(shop.len() + edge_count + 1)
        .ok_or(ActionError::Arithmetic("recipe expansion limit"))?;
    expand_missing_parts(shop, held, &mut stack, expansion_limit)
}

fn expand_missing_parts(
    shop: &[ShopEntry],
    held: &mut Vec<ItemId>,
    stack: &mut Vec<ItemId>,
    expansion_limit: usize,
) -> Result<BuyRequirement, ActionError> {
    let mut requirement = BuyRequirement {
        missing_cost: 0,
        missing_slots: 0,
    };
    for _ in 0..expansion_limit {
        let Some(item) = stack.pop() else {
            return Ok(requirement);
        };
        if let Some(index) = held.iter().position(|held_item| *held_item == item) {
            held.remove(index);
            continue;
        }
        let entry = shop_entry(shop, item)?;
        if entry.components.is_empty() {
            requirement.missing_slots += 1;
            requirement.missing_cost = requirement
                .missing_cost
                .checked_add(entry.cost)
                .ok_or(ActionError::Arithmetic("missing recipe cost"))?;
            if requirement.missing_slots > MAX_PURCHASE_SLOTS {
                return Ok(requirement);
            }
            continue;
        }
        stack.extend(entry.components.iter().rev().copied());
    }
    Err(ActionError::InvalidSchema("recipe expansion limit"))
}

fn shop_entry(shop: &[ShopEntry], item: ItemId) -> Result<&ShopEntry, ActionError> {
    shop.iter()
        .find(|entry| entry.id == item)
        .ok_or(ActionError::InvalidSchema("unknown recipe component"))
}

fn build_controlled_masks(space: &ActionSpace, unit: ControlledUnit) -> ControlledMasks {
    let mut masks = ControlledMasks::empty(
        space.entities.len(),
        space.points.len(),
        space.loot.len(),
        space.shop.len(),
    );
    let Some(state) = &space.controlled[unit.index()] else {
        return masks;
    };
    let movement_enabled = !has_status(&state.unit, StatusFlags::STUNNED)
        && !has_status(&state.unit, StatusFlags::ROOTED);
    let attack_enabled = !has_status(&state.unit, StatusFlags::STUNNED)
        && !has_status(&state.unit, StatusFlags::DISARMED);
    masks.stop = true;
    fill_body_masks(space, state, movement_enabled, attack_enabled, &mut masks);
    fill_cast_masks(space, state, &mut masks);
    fill_use_masks(space, state, unit, &mut masks);
    fill_put_masks(space, state, unit, &mut masks);
    fill_inventory_masks(space, state, unit, &mut masks);
    masks
}

fn fill_body_masks(
    space: &ActionSpace,
    state: &ControlledState,
    movement_enabled: bool,
    attack_enabled: bool,
    masks: &mut ControlledMasks,
) {
    for (index, point) in space.points.iter().enumerate() {
        masks.move_points[index] = movement_enabled && point.walkable;
        masks.attack_move_points[index] = attack_enabled && point.walkable;
    }
    for (index, target) in space.entities.iter().enumerate() {
        let other = target.id != state.id;
        masks.follow_entities[index] = movement_enabled && other;
        masks.attack_entities[index] = attack_enabled && other;
    }
    masks.hold = attack_enabled;
}

fn fill_cast_masks(space: &ActionSpace, state: &ControlledState, masks: &mut ControlledMasks) {
    let disabled = has_status(&state.unit, StatusFlags::STUNNED)
        || has_status(&state.unit, StatusFlags::SILENCED);
    masks.casts.reserve(state.unit.abilities.len());
    for ability in state.unit.abilities.iter().take(MAX_ABILITY_SLOTS) {
        let ready = !disabled
            && !ability.passive
            && ability.level > 0
            && ability.cooldown_left == 0
            && state.unit.mana >= ability.mana_cost;
        masks.casts.push(target_mask(
            space,
            state,
            ability.aim,
            ability.range,
            ready,
            false,
        ));
    }
}

fn fill_use_masks(
    space: &ActionSpace,
    state: &ControlledState,
    unit: ControlledUnit,
    masks: &mut ControlledMasks,
) {
    masks.uses.reserve(ACTIVE_ITEM_SLOTS);
    for slot in 0..ACTIVE_ITEM_SLOTS {
        let item = state.unit.items.get(slot).and_then(Option::as_ref);
        let Some(item) = item else {
            masks.uses.push(empty_target_mask(space));
            continue;
        };
        let ready = item.cooldown_left == 0
            && item.charges != Some(0)
            && state.unit.mana >= item.mana_cost
            && !has_status(&state.unit, StatusFlags::STUNNED)
            && !space
                .readiness
                .inventory_muted(unit, ItemSlot(slot as u8), space.tick)
            && !space.readiness.shared_waiting(unit, item.id, space.tick);
        let Some(aim) = item.aim else {
            masks.uses.push(empty_target_mask(space));
            continue;
        };
        masks
            .uses
            .push(target_mask(space, state, aim, item.range, ready, true));
    }
}

fn target_mask(
    space: &ActionSpace,
    state: &ControlledState,
    aim: Aim,
    range: i32,
    ready: bool,
    active_item: bool,
) -> TargetMask {
    let mut mask = empty_target_mask(space);
    if !ready {
        return mask;
    }
    match aim {
        Aim::Own => mask.none = true,
        Aim::Unit => fill_unit_targets(space, state, range, active_item, &mut mask.entities),
        Aim::Point => fill_point_targets(space, state, range, active_item, &mut mask.points),
        Aim::Tree => fill_tree_targets(space, state, range, &mut mask.points),
        Aim::Building => fill_building_targets(space, range, &mut mask.points),
    }
    mask
}

fn fill_unit_targets(
    space: &ActionSpace,
    state: &ControlledState,
    range: i32,
    allied_only: bool,
    mask: &mut [bool],
) {
    for (index, target) in space.entities.iter().enumerate() {
        let relation_allowed = !allied_only
            || matches!(
                target.relation,
                EntityRelation::Own | EntityRelation::Allied
            );
        mask[index] = relation_allowed && in_range(state.unit.pos, target.position, range);
    }
}

fn fill_point_targets(
    space: &ActionSpace,
    state: &ControlledState,
    range: i32,
    require_walkable: bool,
    mask: &mut [bool],
) {
    for (index, target) in space.points.iter().enumerate() {
        mask[index] = (!require_walkable || target.walkable)
            && in_range(state.unit.pos, target.position, range);
    }
}

fn fill_tree_targets(space: &ActionSpace, state: &ControlledState, range: i32, mask: &mut [bool]) {
    for (index, target) in space.points.iter().enumerate() {
        mask[index] = target.standing_tree && in_range(state.unit.pos, target.position, range);
    }
}

fn fill_building_targets(space: &ActionSpace, range: i32, mask: &mut [bool]) {
    for (index, target) in space.points.iter().enumerate() {
        mask[index] = target.allied_building
            && target.walkable
            && space
                .allied_buildings
                .iter()
                .any(|building| in_range(*building, target.position, range));
    }
}

fn fill_put_masks(
    space: &ActionSpace,
    state: &ControlledState,
    unit: ControlledUnit,
    masks: &mut ControlledMasks,
) {
    let bag_limit = bag_slot_limit(unit);
    let underfoot = state.underfoot_walkable;
    masks.put_points.reserve(bag_limit);
    masks.put_units.reserve(bag_limit);
    for source in 0..bag_limit {
        let held = state.unit.items.get(source).is_some_and(Option::is_some);
        masks.put_points.push(PutPointMask {
            underfoot: held && underfoot,
            points: space
                .points
                .iter()
                .map(|point| held && point.walkable)
                .collect(),
        });
        masks.put_units.push(
            space
                .entities
                .iter()
                .map(|target| held && valid_put_unit_target(state, target))
                .collect(),
        );
    }
}

fn valid_put_unit_target(state: &ControlledState, target: &EntityCandidate) -> bool {
    target.id != state.id
        && matches!(
            target.relation,
            EntityRelation::Own | EntityRelation::Allied
        )
        && matches!(target.kind, UnitKind::Hero | UnitKind::Courier)
        && target.unit.items.iter().any(Option::is_none)
}

fn fill_inventory_masks(
    space: &ActionSpace,
    state: &ControlledState,
    unit: ControlledUnit,
    masks: &mut ControlledMasks,
) {
    let bag_has_space = state.unit.items.iter().any(Option::is_none);
    masks.take.fill(bag_has_space);
    fill_buy_mask(space, state, unit, masks);
    let near_fountain = near_own_fountain(space, state.unit.pos);
    for slot in 0..WIRE_ITEM_SLOTS {
        let held = item_at(space, state, unit, slot);
        masks.sell[slot] = held.is_some_and(|item| item.for_sale);
        for to in 0..WIRE_ITEM_SLOTS {
            masks.swap[slot][to] = slot != to
                && held.is_some()
                && slot_exists(space, state, unit, to)
                && (!(is_stash(slot) || is_stash(to)) || near_fountain);
        }
    }
    if unit == ControlledUnit::Hero {
        masks.learn = state
            .unit
            .abilities
            .iter()
            .take(MAX_ABILITY_SLOTS)
            .map(|ability| ability.can_level)
            .collect();
    }
}

fn fill_buy_mask(
    space: &ActionSpace,
    state: &ControlledState,
    unit: ControlledUnit,
    masks: &mut ControlledMasks,
) {
    if unit != ControlledUnit::Hero {
        return;
    }
    let stash_slots = space.own_stash.iter().filter(|slot| slot.is_none()).count();
    let hero_slots = if near_own_fountain(space, state.unit.pos) {
        state
            .unit
            .items
            .iter()
            .filter(|slot| slot.is_none())
            .count()
    } else {
        0
    };
    let free_slots = stash_slots + hero_slots;
    for (index, item) in space.shop.iter().enumerate() {
        let requirement = space.buy_requirements[index];
        masks.buy[index] = free_slots > 0
            && free_slots >= requirement.missing_slots
            && space.own_gold >= item.cost
            && space.own_gold >= requirement.missing_cost;
    }
}

fn item_at<'a>(
    space: &'a ActionSpace,
    state: &'a ControlledState,
    unit: ControlledUnit,
    slot: usize,
) -> Option<&'a bota_proto::ItemView> {
    if slot < bag_slot_limit(unit) {
        return state.unit.items.get(slot).and_then(Option::as_ref);
    }
    if is_stash(slot) {
        return space
            .own_stash
            .get(slot - STASH_SLOT_START)
            .and_then(Option::as_ref);
    }
    None
}

fn slot_exists(
    space: &ActionSpace,
    state: &ControlledState,
    unit: ControlledUnit,
    slot: usize,
) -> bool {
    if slot < bag_slot_limit(unit) {
        return slot < state.unit.items.len();
    }
    is_stash(slot) && slot - STASH_SLOT_START < space.own_stash.len()
}

const fn bag_slot_limit(unit: ControlledUnit) -> usize {
    match unit {
        ControlledUnit::Hero => HERO_BAG_SLOTS,
        ControlledUnit::Courier => COURIER_BAG_SLOTS,
    }
}

const fn is_stash(slot: usize) -> bool {
    slot >= STASH_SLOT_START && slot < WIRE_ITEM_SLOTS
}

fn near_own_fountain(space: &ActionSpace, position: Vec2) -> bool {
    space
        .own_fountain
        .is_some_and(|fountain| position.within(fountain, Fixed::from_int(STASH_ACCESS_RANGE)))
}

fn build_kind_mask(masks: &[ControlledMasks; 2]) -> KindMask {
    let mut output = [false; ACTION_KIND_COUNT];
    output[ActionKind::Continue.index()] = true;
    for kind in ActionKind::ALL.into_iter().skip(1) {
        output[kind.index()] = masks.iter().any(|mask| mask.family_allowed(kind));
    }
    KindMask(output)
}

fn allows_put_point(masks: &ControlledMasks, source: ItemSlot, target: PutPointTarget) -> bool {
    let Some(mask) = masks.put_points.get(usize::from(source.0)) else {
        return false;
    };
    match target {
        PutPointTarget::Underfoot => mask.underfoot,
        PutPointTarget::Point(point) => mask.points.get(point.0).copied().unwrap_or(false),
    }
}

fn allows_put_unit(masks: &ControlledMasks, source: ItemSlot, target: EntityIndex) -> bool {
    masks
        .put_units
        .get(usize::from(source.0))
        .and_then(|mask| mask.get(target.0))
        .copied()
        .unwrap_or(false)
}

fn empty_target_mask(space: &ActionSpace) -> TargetMask {
    TargetMask {
        none: false,
        entities: vec![false; space.entities.len()],
        points: vec![false; space.points.len()],
    }
}

fn entity_relation(tracker: &StateTracker, unit: &UnitView) -> EntityRelation {
    if unit.owner == Some(tracker.slot()) {
        EntityRelation::Own
    } else if unit.team == tracker.team() {
        EntityRelation::Allied
    } else if unit.team == Team::Neutral {
        EntityRelation::Neutral
    } else {
        EntityRelation::Enemy
    }
}

fn candidate_owner_relation(tracker: &StateTracker, unit: &UnitView) -> Option<EntityRelation> {
    let owner = unit.owner?;
    if owner == tracker.slot() {
        return Some(EntityRelation::Own);
    }
    let team = tracker
        .current()?
        .players
        .iter()
        .find(|player| player.slot == owner)?
        .team;
    Some(if team == tracker.team() {
        EntityRelation::Allied
    } else {
        EntityRelation::Enemy
    })
}

const fn relation_index(relation: EntityRelation) -> u8 {
    match relation {
        EntityRelation::Own => 0,
        EntityRelation::Allied => 1,
        EntityRelation::Enemy => 2,
        EntityRelation::Neutral => 3,
    }
}

fn in_range(from: Vec2, to: Vec2, range: i32) -> bool {
    range >= 0 && from.within(to, Fixed::from_int(range))
}

const fn has_status(unit: &UnitView, status: u16) -> bool {
    unit.statuses.bits & status != 0
}

const fn is_creep(kind: UnitKind) -> bool {
    matches!(
        kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
            | UnitKind::CreepNeutral
    )
}

fn mask_at(mask: &[bool], index: usize) -> bool {
    mask.get(index).copied().unwrap_or(false)
}

fn check_candidate(field: &'static str, index: usize, count: usize) -> Result<(), ActionError> {
    if index >= count {
        return Err(ActionError::CandidateIndex {
            field,
            index,
            count,
        });
    }
    Ok(())
}
