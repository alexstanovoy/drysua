//! Fixtures shared by the test modules whose copies differed only in values.
//!
//! Each builder starts from neutral terms and takes the values that varied
//! between the replaced copies through its constructor or setters; a caller
//! that leaves a setter out gets only the neutral term, never another
//! scenario's value.

use bota_proto::{
    Angle, Attribute, Attributes, EntityId, Fixed, HeroId, MapId, MatchInfo, Pick, ShopEntry,
    SlotId, StatusFlags, Team, TickMode, UnitKind, UnitView, Vec2,
};

use crate::{FeatureFrame, SHADOW_FIEND};

/// A unit view from the lockstep worlds, with every differing stat explicit.
pub(super) struct UnitFixture {
    pub(super) id: EntityId,
    pub(super) kind: UnitKind,
    pub(super) team: Team,
    pub(super) pos: Vec2,
    /// Current and maximum mana, which every replaced copy kept equal.
    pub(super) mana: i32,
    pub(super) attack_damage: i32,
    pub(super) attack_time: u32,
    pub(super) attributes: Attributes,
    pub(super) primary: Option<Attribute>,
    pub(super) hero: Option<HeroId>,
    pub(super) owner: Option<SlotId>,
    pub(super) level: u8,
}

impl UnitFixture {
    pub(super) fn build(self) -> UnitView {
        UnitView {
            id: self.id,
            kind: self.kind,
            team: self.team,
            pos: self.pos,
            facing: Angle { brads: 0 },
            hp: 1_000,
            max_hp: 1_000,
            mana: self.mana,
            max_mana: self.mana,
            move_speed: Fixed::from_int(300),
            attack_damage: self.attack_damage,
            attack_range: Fixed::from_int(500),
            attack_time: self.attack_time,
            attack_point: 0,
            attack_speed: 100,
            armor: Fixed::ZERO,
            magic_resist: Fixed::ZERO,
            collision: Fixed::from_int(24),
            bound: Fixed::from_int(24),
            vision_radius: Fixed::from_int(1_800),
            true_sight_radius: Fixed::ZERO,
            statuses: StatusFlags { bits: 0 },
            attributes: self.attributes,
            primary: self.primary,
            hero: self.hero,
            owner: self.owner,
            level: self.level,
            abilities: Vec::new(),
            items: Vec::new(),
            effects: Vec::new(),
        }
    }
}

/// The two-seat pick layout: slot 0 plays `team`, slot 1 the opposing side and
/// both pick Shadow Fiend.
pub(super) fn two_seat_picks(team: Team) -> Vec<Pick> {
    let opponent = match team {
        Team::Radiant => Team::Dire,
        Team::Dire => Team::Radiant,
        Team::Neutral => Team::Neutral,
    };
    vec![
        Pick {
            slot: SlotId(0),
            team,
            hero: SHADOW_FIEND,
        },
        Pick {
            slot: SlotId(1),
            team: opponent,
            hero: SHADOW_FIEND,
        },
    ]
}

/// A match description that starts from the neutral terms and replaces exactly
/// what each scenario changes: thirty ticks per second and a lockstep mode.
pub(super) struct MatchInfoFixture {
    info: MatchInfo,
}

impl MatchInfoFixture {
    /// `picks` is explicit because the seat count and sides vary per scenario.
    pub(super) fn new(match_id: u64, map: MapId, picks: Vec<Pick>) -> Self {
        Self {
            info: MatchInfo {
                match_id,
                map,
                tick_rate: 30,
                pregame_ticks: 0,
                trees: Vec::new(),
                terrain_cells: 0,
                terrain_rle: Vec::new(),
                opaque_cells: Vec::new(),
                mode: TickMode::Lockstep,
                picks,
                shop: Vec::new(),
            },
        }
    }

    /// The clock counts up from minus `ticks`; zero starts the game at once.
    pub(super) fn pregame_ticks(mut self, ticks: u32) -> Self {
        self.info.pregame_ticks = ticks;
        self
    }

    pub(super) fn trees(mut self, trees: Vec<Vec2>) -> Self {
        self.info.trees = trees;
        self
    }

    pub(super) fn terrain_cells(mut self, cells: u32) -> Self {
        self.info.terrain_cells = cells;
        self
    }

    pub(super) fn terrain_rle(mut self, rle: Vec<(u16, u8)>) -> Self {
        self.info.terrain_rle = rle;
        self
    }

    pub(super) fn opaque_cells(mut self, cells: Vec<(u16, u16)>) -> Self {
        self.info.opaque_cells = cells;
        self
    }

    pub(super) fn mode(mut self, mode: TickMode) -> Self {
        self.info.mode = mode;
        self
    }

    pub(super) fn shop(mut self, shop: Vec<ShopEntry>) -> Self {
        self.info.shop = shop;
        self
    }

    pub(super) fn build(self) -> MatchInfo {
        self.info
    }
}

/// Flattens a frame in encoder order, limiting a category to its leading terms
/// when a frozen layout captured only those.
pub(super) fn feature_values(
    frame: &FeatureFrame,
    global_limit: Option<usize>,
    unit_limit: Option<usize>,
) -> impl Iterator<Item = f32> + '_ {
    fn prefix(values: &[f32], limit: Option<usize>) -> &[f32] {
        match limit {
            Some(limit) => &values[..limit],
            None => values,
        }
    }
    prefix(&frame.global, global_limit)
        .iter()
        .copied()
        .chain(frame.history.iter().flatten().copied())
        .chain(frame.policy_history.iter().flatten().copied())
        .chain(
            frame
                .units
                .iter()
                .flat_map(move |row| prefix(row, unit_limit).iter().copied()),
        )
        .chain(
            frame
                .own_units
                .iter()
                .flat_map(move |row| prefix(row, unit_limit).iter().copied()),
        )
        .chain(
            frame
                .remembered_units
                .iter()
                .flat_map(move |row| prefix(row, unit_limit).iter().copied()),
        )
        .chain(frame.points.iter().flatten().copied())
        .chain(frame.abilities.iter().flatten().copied())
        .chain(frame.items.iter().flatten().copied())
        .chain(frame.projectiles.iter().flatten().copied())
        .chain(frame.loot.iter().flatten().copied())
        .chain(frame.map.iter().copied())
}
