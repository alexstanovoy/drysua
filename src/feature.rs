#![allow(
    clippy::float_arithmetic,
    reason = "policy tensors use bounded f32 values outside the deterministic simulation"
)]

use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;

use bota_proto::{
    AbilityId, AbilityView, Aim, Angle, Attribute, DamageKind, EntityId, EventKind, Fixed, ItemId,
    ItemSlot, ItemView, Order, PlayerView, ProjectileView, ShopEntry, StatusFlags, Target, Team,
    UnitKind, UnitView, Vec2,
};

use crate::tracker::{StaticTrackerProvenance, TrackerProvenance};
use crate::{
    ActionKind, ActionSpace, ControlledUnit, EntityRelation, HISTORY_AGES, IssuedOrder,
    ItemReadiness, LandmarkRelation, MAX_LOOT, MAX_POINT_CANDIDATES, MAX_PROJECTILES,
    MAX_SHOP_ITEMS, OWN_ITEM_SLOTS, PointCandidate, PointDirection, PointSource,
    SHADOW_FIEND_ABILITY_SLOTS, StateTracker, TERRAIN_CELL_SIZE, UNIT_TOKENS,
};

mod combat;

#[cfg(test)]
#[path = "tests/feature_capacity.rs"]
mod capacity_tests;

#[cfg(test)]
#[path = "tests/feature_test_support.rs"]
mod test_support;

/// Version of the policy feature layout.
pub const FEATURE_SCHEMA_VERSION: u32 = 26;
/// Number of scalar global features.
pub const GLOBAL_FEATURES: usize = global_feature::WIDTH;
/// Number of scalar features in one global-history sample.
pub const HISTORY_FEATURES: usize = history_feature::WIDTH;
/// Number of global-history samples.
pub const HISTORY_SAMPLES: usize = HISTORY_AGES.len();
/// Number of scalar features in one local policy-history sample.
pub const POLICY_HISTORY_FEATURES: usize = policy_history_feature::WIDTH;
/// Maximum number of local policy-history samples.
pub const MAX_POLICY_HISTORY: usize = 16;
/// Number of scalar features in one unit token.
pub const UNIT_FEATURES: usize = unit_feature::WIDTH;
/// Number of unit tokens in exact ActionSpace entity-candidate order.
pub const UNIT_FEATURE_TOKENS: usize = UNIT_TOKENS;
/// Number of fixed own hero and courier unit tokens.
pub const OWN_UNIT_FEATURE_TOKENS: usize = 2;
/// Maximum number of non-targetable remembered unit tokens.
pub const REMEMBERED_UNIT_FEATURE_TOKENS: usize = 32;
/// Number of scalar features in one point-candidate token.
pub const POINT_FEATURES: usize = point_feature::WIDTH;
/// Number of point tokens in exact ActionSpace point-candidate order.
pub const POINT_FEATURE_TOKENS: usize = MAX_POINT_CANDIDATES;
/// Number of scalar features in one ability token.
pub const ABILITY_FEATURES: usize = ability_feature::WIDTH;
/// Number of fixed own hero and courier ability tokens.
pub const ABILITY_FEATURE_TOKENS: usize = SHADOW_FIEND_ABILITY_SLOTS + 8;
/// Number of scalar features in one item token.
pub const ITEM_FEATURES: usize = item_feature::WIDTH;
/// Inventory and backpack slots of the nearest enemy hero, after the shop rows.
pub const ENEMY_ITEM_SLOTS: usize = 9;
/// Number of fixed own inventory, shop and enemy-hero item tokens.
pub const ITEM_FEATURE_TOKENS: usize = OWN_ITEM_SLOTS + MAX_SHOP_ITEMS + ENEMY_ITEM_SLOTS;
/// Number of scalar features in one projectile token.
pub const PROJECTILE_FEATURES: usize = projectile_feature::WIDTH;
/// Number of fixed projectile tokens.
pub const PROJECTILE_FEATURE_TOKENS: usize = 32;
/// Number of scalar features in one loot token.
pub const LOOT_FEATURES: usize = loot_feature::WIDTH;
/// Number of fixed loot tokens.
pub const LOOT_FEATURE_TOKENS: usize = MAX_LOOT;
/// Number of scalar local map-context features.
pub const MAP_FEATURES: usize = map_feature::WIDTH;
/// One-hot ability identifier classes: ids below the last class, then "other".
pub const ABILITY_ID_CLASSES: usize = 25;
/// One-hot item identifier classes: ids below the last class, then "other".
pub const ITEM_ID_CLASSES: usize = 65;

/// Whole ticks of an attack cadence the wire reports in milliseconds.
///
/// Tick rate is pinned to the validated 30 by every accepted match.
pub(crate) fn attack_interval_ticks(attack_time_ms: u32) -> u32 {
    ((u64::from(attack_time_ms) * 30) / 1000) as u32
}

/// Normalisers in world units, ticks, whole points and counts. Tactical
/// quantities saturate where the fight stops caring, so raze reaches, attack
/// ranges, cooldowns and last-hit health keep resolution inside [-1, 1].
mod scale {
    pub const NEAR_DISTANCE: f32 = 2_000.0;
    pub const LOG_DISTANCE_UNIT: f32 = 100.0;
    pub const LOG_DISTANCE_MAX: f32 = 26_000.0;
    pub const HEALTH_NEAR: f32 = 1_000.0;
    pub const HEALTH_LOG_MAX: f32 = 5_000.0;
    pub const EFFECTIVE_HEALTH_LOG_MAX: f32 = 10_000.0;
    pub const MANA_NEAR: f32 = 1_000.0;
    pub const POOL_DELTA: f32 = 100.0;
    pub const DAMAGE: f32 = 300.0;
    pub const RANGE: f32 = 1_000.0;
    pub const ATTACK_TIME_MS: f32 = 3_000.0;
    pub const ATTACK_POINT_MS: f32 = 1_000.0;
    pub const ATTACK_SPEED: f32 = 700.0;
    pub const MOVE_SPEED: f32 = 550.0;
    pub const ARMOR: f32 = 30.0;
    pub const VISION: f32 = 2_000.0;
    pub const BOUND: f32 = 300.0;
    pub const VELOCITY: f32 = 600.0;
    pub const HITS: f32 = 20.0;
    pub const REACH_SECONDS: f32 = 10.0;
    pub const MARGIN: f32 = 1_000.0;
    pub const RAZE_DAMAGE: f32 = 500.0;
    pub const RAZES_TO_KILL: u32 = 5;
    pub const RAZE_STACKS: u32 = 4;
    pub const LEVEL: f32 = 25.0;
    pub const COOLDOWN_NEAR_TICKS: f32 = 300.0;
    pub const COOLDOWN_LOG_TICKS: f32 = 3_600.0;
    pub const RESTORED_HEALTH: f32 = 500.0;
    pub const RESTORED_MANA: f32 = 300.0;
    pub const GOLD: f32 = 5_000.0;
    pub const XP: f32 = 5_000.0;
    pub const LAST_HITS: f32 = 100.0;
    pub const DENIES: f32 = 50.0;
    pub const SNAPSHOT_DAMAGE: f32 = 200.0;
    pub const ORDER_AGE_TICKS: f32 = 900.0;
    pub const RESPAWN_TICKS: f32 = 1_800.0;
    pub const VISIBLE_UNITS: f32 = 64.0;
    pub const CREEPS: f32 = 8.0;
    pub const CREEP_BALANCE_RADIUS: f32 = 1_200.0;
    pub const ABILITY_MANA: f32 = 200.0;
    pub const ITEM_VALUE: f32 = 5_000.0;
    pub const CHARGES: f32 = 10.0;
    pub const MUTE_TICKS: f32 = 180.0;
    pub const SHARED_WAIT_TICKS: f32 = 2_100.0;
    pub const LAST_CAST_TICKS: f32 = 1_800.0;
    pub const PROJECTILE_RELATIVE: f32 = 2_000.0;
    pub const PROJECTILE_VELOCITY: f32 = 1_500.0;
    pub const PROJECTILE_AGE_TICKS: f32 = 60.0;
    pub const PROJECTILE_APPROACH: f32 = 500.0;
    pub const LOOT_AGE_TICKS: f32 = 300.0;
    pub const REQUIEM_COOLDOWN_TICKS: f32 = 3_600.0;
    pub const RAZE_COOLDOWN_TICKS: f32 = 300.0;
    pub const RECENT_ATTACK_TICKS: u32 = 60;
}

/// Journal events older than this never feed recent-damage and cast inputs.
const RECENT_EVENT_TICKS: u32 = 4_800;
const MAP_RAY_CELLS: usize = 20;
const MAP_DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Stable indices in each global vector; widths above one start a one-hot or fixed block.
pub mod global_feature {
    pub const TICK: usize = 0;
    pub const PREGAME_PROGRESS: usize = 1;
    pub const WAVE_PHASE: usize = 2;
    pub const JUNGLE_PHASE: usize = 3;
    pub const SIDE_RADIANT: usize = 4;
    pub const SIDE_DIRE: usize = 5;
    pub const OWN_ASSET_VALUE: usize = 6;
    pub const ACTIVE_ORDER_PRESENT: usize = 7;
    pub const ACTIVE_ORDER_KIND_START: usize = 8;
    pub const ACTIVE_ORDER_AGE: usize = 24;
    pub const LAST_DECISION_PRESENT: usize = 25;
    pub const SNAPSHOT_DAMAGE_DEALT: usize = 26;
    pub const SNAPSHOT_DAMAGE_TAKEN: usize = 27;
    pub const OWN_LEVEL: usize = 28;
    pub const OWN_XP: usize = 29;
    pub const OWN_LAST_HITS: usize = 30;
    pub const OWN_DENIES: usize = 31;
    pub const ACTIVE_TARGET_PRESENT: usize = 32;
    pub const ACTIVE_TARGET_POINT: usize = 33;
    pub const ACTIVE_TARGET_UNIT: usize = 34;
    pub const ACTIVE_TARGET_VISIBLE: usize = 35;
    pub const ACTIVE_TARGET_RELATIVE_X: usize = 36;
    pub const ACTIVE_TARGET_RELATIVE_Y: usize = 37;
    pub const ACTIVE_TARGET_DISTANCE_NEAR: usize = 38;
    pub const ACTIVE_TARGET_DISTANCE_LOG: usize = 39;
    pub const ACTIVE_TARGET_KIND_START: usize = 40;
    pub const ACTIVE_TARGET_ALLIED: usize = 52;
    pub const ACTIVE_TARGET_ENEMY: usize = 53;
    pub const ACTIVE_TARGET_NEUTRAL: usize = 54;
    pub const OWN_ANCIENT_RELATIVE_X: usize = 55;
    pub const OWN_ANCIENT_RELATIVE_Y: usize = 56;
    pub const OWN_ANCIENT_DISTANCE_LOG: usize = 57;
    pub const ENEMY_ANCIENT_RELATIVE_X: usize = 58;
    pub const ENEMY_ANCIENT_RELATIVE_Y: usize = 59;
    pub const ENEMY_ANCIENT_DISTANCE_LOG: usize = 60;
    pub const MAP2_OWN_TOWER_HEALTH: usize = 61;
    pub const MAP2_ENEMY_TOWER_HEALTH: usize = 62;
    pub const MAP2_OWN_DEATHS: usize = 63;
    pub const MAP2_ENEMY_DEATHS: usize = 64;
    pub const MAP2_OWN_HERO_HEALTH: usize = 65;
    pub const MAP2_ENEMY_HERO_HEALTH: usize = 66;
    pub const MAP2_XP_LEAD: usize = 67;
    pub const MAP2_REWARD_POTENTIAL: usize = 68;
    pub const ENEMY_HERO_VISIBLE: usize = 69;
    pub const SECONDS_TO_KILL_ENEMY: usize = 70;
    pub const SECONDS_TO_BE_KILLED: usize = 71;
    pub const ENEMY_CREEPS_ACQUIRING: usize = 72;
    pub const ENEMY_CREEPS_ATTACKING: usize = 73;
    pub const CREEP_BALANCE: usize = 74;
    pub const ALLIED_CREEPS_UNDER_ENEMY_TOWER: usize = 75;
    pub const ENEMY_CREEPS_UNDER_OWN_TOWER: usize = 76;
    pub const IN_ENEMY_TOWER_RANGE: usize = 77;
    pub const ENEMY_TOWER_TARGETING: usize = 78;
    pub const RAZES_READY: usize = 79;
    pub const RAZES_AFFORDABLE: usize = 80;
    pub(crate) const WIDTH: usize = 81;
}

/// Stable indices in each global-history sample; widths above one start a one-hot or fixed block.
pub mod history_feature {
    pub const SAMPLE_PRESENT: usize = 0;
    pub const AGE: usize = 1;
    pub const HP_PRESENT: usize = 2;
    pub const HP_RATIO: usize = 3;
    pub const MANA_PRESENT: usize = 4;
    pub const MANA_RATIO: usize = 5;
    pub const OWN_LEVEL: usize = 6;
    pub const OWN_GOLD: usize = 7;
    pub const OWN_ALIVE: usize = 8;
    pub const RESPAWN_LEFT: usize = 9;
    pub const VISIBLE_ALLIED_UNITS: usize = 10;
    pub const VISIBLE_ENEMY_UNITS: usize = 11;
    pub(crate) const WIDTH: usize = 12;
}

/// Stable indices in each local policy-history sample; widths above one start a one-hot or fixed block.
pub mod policy_history_feature {
    pub const KIND_START: usize = 0;
    pub(crate) const WIDTH: usize = 16;
}

/// Stable indices in each local map context; widths above one start a one-hot or fixed block.
pub mod map_feature {
    pub const WALKABLE: usize = 0;
    pub const WATER: usize = 1;
    pub const ELEVATION: usize = 2;
    pub const OWN_FOUNTAIN_DISTANCE_LOG: usize = 3;
    pub const ENEMY_FOUNTAIN_DISTANCE_LOG: usize = 4;
    pub const OWN_TOWER_DISTANCE_LOG: usize = 5;
    pub const ENEMY_TOWER_DISTANCE_LOG: usize = 6;
    pub const RAYS_START: usize = 7;
    pub(crate) const WIDTH: usize = 87;
}

/// Stable indices in each unit token; widths above one start a one-hot or fixed block.
pub mod unit_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const RELATION_START: usize = 1;
    pub const KIND_START: usize = 5;
    pub const ENEMY_OWNED: usize = 17;
    pub const VISIBLE: usize = 18;
    pub const AGE: usize = 19;
    pub const POSITION_X: usize = 20;
    pub const POSITION_Y: usize = 21;
    pub const RELATIVE_X: usize = 22;
    pub const RELATIVE_Y: usize = 23;
    pub const DIRECTION_X: usize = 24;
    pub const DIRECTION_Y: usize = 25;
    pub const DISTANCE_NEAR: usize = 26;
    pub const DISTANCE_LOG: usize = 27;
    pub const BEARING_COS: usize = 28;
    pub const BEARING_SIN: usize = 29;
    pub const FACING_COS: usize = 30;
    pub const FACING_SIN: usize = 31;
    pub const BOUND: usize = 32;
    pub const MOTION_PRESENT: usize = 33;
    pub const VELOCITY_X: usize = 34;
    pub const VELOCITY_Y: usize = 35;
    pub const ELEVATION: usize = 36;
    pub const WALKABLE: usize = 37;
    pub const HP_RATIO: usize = 38;
    pub const HP_NEAR: usize = 39;
    pub const HP_LOG: usize = 40;
    pub const MAX_HP_LOG: usize = 41;
    pub const MANA_PRESENT: usize = 42;
    pub const MANA_RATIO: usize = 43;
    pub const MANA_NEAR: usize = 44;
    pub const MAX_MANA_NEAR: usize = 45;
    pub const HP_DELTA: usize = 46;
    pub const MANA_DELTA: usize = 47;
    pub const EFFECTIVE_HP_PHYSICAL: usize = 48;
    pub const EFFECTIVE_HP_MAGICAL: usize = 49;
    pub const LEVEL: usize = 50;
    pub const ATTACK_DAMAGE: usize = 51;
    pub const ATTACK_RANGE: usize = 52;
    pub const ATTACK_TIME: usize = 53;
    pub const ATTACK_POINT: usize = 54;
    pub const ATTACK_SPEED: usize = 55;
    pub const MOVE_SPEED: usize = 56;
    pub const ARMOR: usize = 57;
    pub const MAGIC_RESISTANCE: usize = 58;
    pub const VISION: usize = 59;
    pub const HERO_RELATIVE_PRESENT: usize = 60;
    pub const OWN_HIT: usize = 61;
    pub const OWN_HITS_TO_KILL: usize = 62;
    pub const ITS_HIT: usize = 63;
    pub const ITS_HITS_TO_KILL_OWN: usize = 64;
    pub const TIME_TO_REACH: usize = 65;
    pub const OWN_IN_ATTACK_RANGE: usize = 66;
    pub const UNIT_IN_ATTACK_RANGE: usize = 67;
    pub const OWN_RANGE_MARGIN: usize = 68;
    pub const UNIT_RANGE_MARGIN: usize = 69;
    pub const OWN_IN_ACQUISITION: usize = 70;
    pub const KILLABLE_NOW: usize = 71;
    pub const OWN_RAZE_DAMAGE: usize = 72;
    pub const OWN_RAZES_TO_KILL: usize = 73;
    pub const IN_OWN_RAZE_START: usize = 74;
    pub const SLOWED: usize = 77;
    pub const INVULNERABLE: usize = 78;
    pub const CHANNELLING: usize = 79;
    pub const RECENT_DAMAGE_TAKEN: usize = 80;
    pub const RECENT_DAMAGE_DEALT_PRESENT: usize = 81;
    pub const RECENT_DAMAGE_DEALT: usize = 82;
    pub const ATTACK_PHASE_PRESENT: usize = 83;
    pub const ATTACK_PHASE: usize = 84;
    pub const ATTACKING_OWN_HERO: usize = 85;
    pub const ATTACKING_OWN_SIDE: usize = 86;
    pub const RAZE_EFFECT_PRESENT: usize = 87;
    pub const RAZE_STACKS: usize = 88;
    pub const RAZE_TICKS_LEFT: usize = 89;
    pub const GUARDED_TICKS_LEFT: usize = 90;
    pub const INSPIRED_TICKS_LEFT: usize = 91;
    pub const HEALTH_RESTORE_REPORT_PRESENT: usize = 92;
    pub const HEALTH_RESTORE_REPORT_AMOUNT: usize = 93;
    pub const HEALTH_RESTORE_REPORT_AGE: usize = 94;
    pub const MANA_RESTORE_REPORT_PRESENT: usize = 95;
    pub const MANA_RESTORE_REPORT_AMOUNT: usize = 96;
    pub const MANA_RESTORE_REPORT_AGE: usize = 97;
    pub const KIT_PRESENT: usize = 98;
    pub const RAZE_COOLDOWN_START: usize = 99;
    pub const RAZE_LEVEL: usize = 102;
    pub const REQUIEM_COOLDOWN: usize = 103;
    pub const REQUIEM_LEVEL: usize = 104;
    pub const CAN_AFFORD_RAZE: usize = 105;
    pub const RAZE_THREAT_START: usize = 106;
    pub(crate) const WIDTH: usize = 109;
}

/// Stable indices in each point-candidate token; widths above one start a one-hot or fixed block.
pub mod point_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const POSITION_X: usize = 1;
    pub const POSITION_Y: usize = 2;
    pub const RELATIVE_X: usize = 3;
    pub const RELATIVE_Y: usize = 4;
    pub const DIRECTION_X: usize = 5;
    pub const DIRECTION_Y: usize = 6;
    pub const DISTANCE_NEAR: usize = 7;
    pub const DISTANCE_LOG: usize = 8;
    pub const BEARING_COS: usize = 9;
    pub const BEARING_SIN: usize = 10;
    pub const SOURCE_START: usize = 11;
    pub const SOURCE_DIRECTION_START: usize = 24;
    pub const SOURCE_RADIUS_PRESENT: usize = 32;
    pub const SOURCE_RADIUS: usize = 33;
    pub const SOURCE_KIND_START: usize = 34;
    pub const SOURCE_RELATION_START: usize = 46;
    pub const WALKABLE: usize = 50;
    pub const STANDING_TREE: usize = 51;
    pub const ALLIED_BUILDING: usize = 52;
    pub const SIGHTING_AGE: usize = 53;
    pub const RAZE_UNITS_START: usize = 54;
    pub const RAZE_HEROES_START: usize = 57;
    pub(crate) const WIDTH: usize = 60;
}

/// Stable indices in each ability token; widths above one start a one-hot or fixed block.
pub mod ability_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const BODY_START: usize = 1;
    pub const SLOT_START: usize = 3;
    pub const OBSERVED: usize = 11;
    pub const ID_START: usize = 12;
    pub const LEVEL: usize = 37;
    pub const MAX_LEVEL: usize = 38;
    pub const COOLDOWN_NEAR: usize = 39;
    pub const COOLDOWN_LOG: usize = 40;
    pub const MANA_COST: usize = 41;
    pub const RANGE: usize = 42;
    pub const AIM_START: usize = 43;
    pub const PASSIVE: usize = 48;
    pub const TOGGLE_ON: usize = 49;
    pub const CAN_LEVEL: usize = 50;
    pub const LEGAL: usize = 51;
    pub const LAST_CAST_PRESENT: usize = 52;
    pub const LAST_CAST_AGE: usize = 53;
    pub const SCOREBOARD_KIT_SOURCE: usize = 54;
    pub const MANA_SUFFICIENT: usize = 55;
    pub(crate) const WIDTH: usize = 56;
}

/// Stable indices in each item token; widths above one start a one-hot or fixed block.
pub mod item_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const LOCATION_START: usize = 1;
    pub const SLOT: usize = 6;
    pub const ITEM_PRESENT: usize = 7;
    pub const ITEM_START: usize = 8;
    pub const CHARGES_PRESENT: usize = 73;
    pub const CHARGES: usize = 74;
    pub const COOLDOWN_NEAR: usize = 75;
    pub const COOLDOWN_LOG: usize = 76;
    pub const AIM_PRESENT: usize = 77;
    pub const AIM_START: usize = 78;
    pub const RANGE: usize = 83;
    pub const MANA_COST: usize = 84;
    pub const ATTRIBUTE_PRESENT: usize = 85;
    pub const ATTRIBUTE_START: usize = 86;
    pub const FOR_SALE: usize = 89;
    pub const MUTED: usize = 90;
    pub const VALUE_PRESENT: usize = 91;
    pub const VALUE: usize = 92;
    pub const RECIPE_COMPONENT: usize = 93;
    pub const COMPOSITE: usize = 94;
    pub const LEGAL: usize = 95;
    pub const SHOP_CANDIDATE: usize = 96;
    pub const MUTE_REMAINING_PRESENT: usize = 97;
    pub const MUTE_REMAINING: usize = 98;
    pub const SHARED_WAIT_PRESENT: usize = 99;
    pub const SHARED_WAIT_REMAINING: usize = 100;
    pub const SCOREBOARD_KIT_SOURCE: usize = 101;
    pub(crate) const WIDTH: usize = 102;
}

/// Stable indices in each projectile token; widths above one start a one-hot or fixed block.
pub mod projectile_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const RELATION_START: usize = 1;
    pub const ABILITY_PRESENT: usize = 5;
    pub const ABILITY_START: usize = 6;
    pub const RELATIVE_X: usize = 31;
    pub const RELATIVE_Y: usize = 32;
    pub const FACING_COS: usize = 33;
    pub const FACING_SIN: usize = 34;
    pub const VELOCITY_PRESENT: usize = 35;
    pub const VELOCITY_X: usize = 36;
    pub const VELOCITY_Y: usize = 37;
    pub const AGE: usize = 38;
    pub const CLOSEST_APPROACH_PRESENT: usize = 39;
    pub const CLOSEST_APPROACH: usize = 40;
    pub(crate) const WIDTH: usize = 41;
}

/// Stable indices in each ground-loot token; widths above one start a one-hot or fixed block.
pub mod loot_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const ITEM_START: usize = 1;
    pub const CHARGES_PRESENT: usize = 66;
    pub const CHARGES: usize = 67;
    pub const RELATIVE_X: usize = 68;
    pub const RELATIVE_Y: usize = 69;
    pub const DISTANCE_NEAR: usize = 70;
    pub const PATH_DISTANCE_PRESENT: usize = 71;
    pub const PATH_DISTANCE: usize = 72;
    pub const VISIBLE_AGE_PRESENT: usize = 73;
    pub const VISIBLE_AGE: usize = 74;
    pub(crate) const WIDTH: usize = 75;
}

/// Canonical schema text covered by [`FEATURE_SCHEMA_HASH`].
pub const FEATURE_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-feature/v26;",
    "shapes=global:81,history:7x12,policy_history:16x16,unit:96x109,own_unit:2x109,remembered_unit:32x109,point:64x60,ability:14x56,item:94x102,projectile:32x41,loot:16x75,map:87;",
    "scalar_ranges=all_finite_within_minus1_1,one_hot_blocks_one_hot_or_all_zero,no_host_conditioning;",
    "history_ages=480,240,120,60,30,15,0;",
    "normalizers=distance:euclidean_centres_near_min(d,2000)/2000_log_ln(1+d/100)/ln(261),position_and_relative:extent,health:ratio_near_min(hp,1000)/1000_log_ln1p(hp)/ln1p(5000),effective_health:ln1p(hp/kept)/ln1p(10000),mana:ratio_near1000,pool_delta:100,damage:300,range:1000,attack_time_ms:3000,attack_point_ms:1000,attack_speed:700,move_speed:550,armor:30,vision:2000,bound:300,velocity_units_per_second:600,hits:20_ceil,reach_seconds:10,margin:1000,raze_damage:500,razes_to_kill:cap5,raze_stacks:4,level:25,cooldown_ticks:near300_log3600,restored_health:500,restored_mana:300,gold_assets_xp:5000,last_hits:100,denies:50,snapshot_damage:200,order_age_ticks:900,respawn_ticks:1800,visible_units:64,creeps:8,ability_mana:200,item_value:5000,charges:10,mute_ticks:180,shared_wait_ticks:2100,last_cast_ticks:1800,projectile:relative2000_velocity1500_age60_approach500,loot_age_ticks:300,raze_cooldown_ticks:300,requiem_cooldown_ticks:3600,tick:27900;all_clamped;",
    "one_hot=unit_kind:Hero,CreepMelee,CreepFlagbearer,CreepRanged,CreepSiege,CreepNeutral,Tower,Ancient,Barracks,Fountain,Ward,Courier;action_kind:append_only16;relation:own,allied,enemy,neutral;ability_id:0..23_then_other;item_id:0..63_then_other;aim:Own,Point,Unit,Tree,Building;attribute:Strength,Agility,Intelligence;body:hero,courier;ability_slot:0..7;item_location:hero,stash,courier,shop,enemy_hero;point_source:Tactical,StaticTree,PlantedTree,BuildingLanding,Fountain,Tower,PredictedHero,PredictedCreep,RazeFacing,RazeCluster,LastSeenHero,ExtrapolatedHero,RazeRing;point_direction:E,NE,N,NW,W,SW,S,SE;",
    "global_layout=tick,pregame_progress,wave_phase,jungle_phase,side_radiant,side_dire,own_asset_value,active_order_present,active_order_kind16,active_order_age,last_decision_present,snapshot_damage_dealt,snapshot_damage_taken,own_level,own_xp,own_last_hits,own_denies,active_target_present,active_target_point,active_target_unit,active_target_visible,active_target_relative_x,active_target_relative_y,active_target_distance_near,active_target_distance_log,active_target_kind12,active_target_allied,active_target_enemy,active_target_neutral,own_ancient_relative_x,own_ancient_relative_y,own_ancient_distance_log,enemy_ancient_relative_x,enemy_ancient_relative_y,enemy_ancient_distance_log,map2_own_tower_health,map2_enemy_tower_health,map2_own_deaths,map2_enemy_deaths,map2_own_hero_health,map2_enemy_hero_health,map2_xp_lead,map2_reward_potential,enemy_hero_visible,seconds_to_kill_enemy,seconds_to_be_killed,enemy_creeps_acquiring,enemy_creeps_attacking,creep_balance,allied_creeps_under_enemy_tower,enemy_creeps_under_own_tower,in_enemy_tower_range,enemy_tower_targeting,razes_ready,razes_affordable;",
    "history_layout=sample_present,age,hp_present,hp_ratio,mana_present,mana_ratio,own_level,own_gold,own_alive,respawn_left,visible_allied_units,visible_enemy_units;",
    "policy_history_layout=kind16;",
    "map_layout=walkable,water,elevation,own_fountain_distance_log,enemy_fountain_distance_log,own_tower_distance_log,enemy_tower_distance_log,rays80;",
    "unit_layout=token_present,relation4,kind12,enemy_owned,visible,age,position_x,position_y,relative_x,relative_y,direction_x,direction_y,distance_near,distance_log,bearing_cos,bearing_sin,facing_cos,facing_sin,bound,motion_present,velocity_x,velocity_y,elevation,walkable,hp_ratio,hp_near,hp_log,max_hp_log,mana_present,mana_ratio,mana_near,max_mana_near,hp_delta,mana_delta,effective_hp_physical,effective_hp_magical,level,attack_damage,attack_range,attack_time,attack_point,attack_speed,move_speed,armor,magic_resistance,vision,hero_relative_present,own_hit,own_hits_to_kill,its_hit,its_hits_to_kill_own,time_to_reach,own_in_attack_range,unit_in_attack_range,own_range_margin,unit_range_margin,own_in_acquisition,killable_now,own_raze_damage,own_razes_to_kill,in_own_raze3,slowed,invulnerable,channelling,recent_damage_taken,recent_damage_dealt_present,recent_damage_dealt,attack_phase_present,attack_phase,attacking_own_hero,attacking_own_side,raze_effect_present,raze_stacks,raze_ticks_left,guarded_ticks_left,inspired_ticks_left,health_restore_report_present,health_restore_report_amount,health_restore_report_age,mana_restore_report_present,mana_restore_report_amount,mana_restore_report_age,kit_present,raze_cooldown3,raze_level,requiem_cooldown,requiem_level,can_afford_raze,raze_threat3;",
    "point_layout=token_present,position_x,position_y,relative_x,relative_y,direction_x,direction_y,distance_near,distance_log,bearing_cos,bearing_sin,source13,source_direction8,source_radius_present,source_radius,source_kind12,source_relation4,walkable,standing_tree,allied_building,sighting_age,raze_units3,raze_heroes3;",
    "ability_layout=token_present,body2,slot8,observed,id25,level,max_level,cooldown_near,cooldown_log,mana_cost,range,aim5,passive,toggle_on,can_level,legal,last_cast_present,last_cast_age,scoreboard_kit_source,mana_sufficient;",
    "item_layout=token_present,location5,slot,item_present,item65,charges_present,charges,cooldown_near,cooldown_log,aim_present,aim5,range,mana_cost,attribute_present,attribute3,for_sale,muted,value_present,value,recipe_component,composite,legal,shop_candidate,mute_remaining_present,mute_remaining,shared_wait_present,shared_wait_remaining,scoreboard_kit_source;",
    "projectile_layout=token_present,relation4,ability_present,ability25,relative_x,relative_y,facing_cos,facing_sin,velocity_present,velocity_x,velocity_y,age,closest_approach_present,closest_approach;",
    "loot_layout=token_present,item65,charges_present,charges,relative_x,relative_y,distance_near,path_distance_present,path_distance,visible_age_present,visible_age;",
    "unit_leading_when_present=own-hero-then-own-courier;item_fixed=hero9,stash6,courier6,shop64,enemy_hero9;",
    "derived=own_hero_perspective_zero_without_live_own_hero,armor_kept:100/(100+6max(armor,0)),magic_kept:1-magic_resist,own_hit_and_its_hit_after_target_mitigation,hits_to_kill_ceil,time_to_reach_euclidean_over_own_move_speed,range_margins_centre_distance_minus_range_minus_both_bounds,acquisition:melee_flagbearer_neutral500_ranged600_siege800_tower_fountain_attack_range_plus_both_bounds,killable_now:last_hit_or_deny_below50pct_building10pct,raze_damage:90_160_230_300_plus_stack50_60_70_80_per_held_stack_times_magic_kept,raze_strikes:centre_within250_of_landing_along_current_facing_live_uninvulnerable_nonstructure;",
    "hero_kit=every_hero_token_visible_or_remembered,raze_and_requiem_cooldown_minus_age,shared_raze_level,requiem_level,can_afford_next_raze;raze_threat=hostile_hero_landing_along_its_facing_strikes_own_hero;recent_attack=last_damage_dealt_within60_ticks_on_own_hero_or_other_own_side_unit;",
    "global_derived=seconds_to_kill_enemy_and_to_be_killed:ready_affordable_razes_stacking_then_attacks_cap30_div30,enemy_hero_visible,enemy_creeps_acquiring_and_attacking_own_hero,creep_balance_within1200,allied_creeps_in_enemy_tower_range,enemy_creeps_in_own_tower_range,own_hero_in_enemy_tower_range,enemy_tower_hit_own_hero_within60,razes_ready_and_affordable_div3;",
    "enemy_items=nearest_visible_else_most_recent_remembered_enemy_hero_inventory9;",
    "active_target=opaque_local_point_or_full_generation_unit_key_never_encoded_as_identifier,canonical_relative_extent,euclidean_near_and_log_distance,point_or_unit,visibility,unit_kind_one_hot_and_relation;",
    "policy_history=16_newest_first_selected_kind_one_hot_only,no_ages_targets_slots_outcomes;fallback_bookkeeping=no_wire_request_no_selected_action_no_sent_history_insertion;",
    "ancient_geometry=seat_observed_current_or_remembered_Ancient_kind_exact_team,only_unique_known_position_per_team,requires_live_own_hero_origin,canonical_delta_extent_log_distance,missing_or_ambiguous_all_zero;",
    "map2_slots=own_and_enemy_weakest_tower_hp,deaths_div_limit,hero_hp_last_seen_dead1,xp_lead_clamped,reward_potential;",
    "raze_effect=anonymous_effect15_present,stacks_div4_clamped,ticks_left_div240;pair=max_lexicographic_valid_active_row_stacks_then_remaining_ticks,never_sum_or_source_attribution;memory_timer=observed_ticks_minus_age_expired_zero,no_fog_refresh,known_death_clears;",
    "manual_restoration=Healed_amount_health_and_mana_independent_reports,latest_positive_per_channel,amount_health_div500_mana_div300_clamped,age_div480;",
    "navigation_contract=existing_walkable_building_landing_move_pointers_allowed,attack_move_veto_and_tp_provenance_unchanged;frame_legality_and_action_space_provenance=new_contract,raw_dimensions_field_ids_point_order_and_sources_unchanged,no_old_frame_or_corpus_relabel;",
    "coordinates=raw_extent:terrain_cells*64*65536,dire_position:(extent_raw-1)-raw,dire_delta:negated,dire_facing:brads+32768_wrapping,absolute_side:global[4:6];",
    "unit_candidates=current_visible_live_only,cap96,priority:own_body_then_hero_then_structure_then_within1200_then_other,distance_then_relation_then_owner_relation_then_canonical_model_semantics_then_entity_id_only_for_semantically_identical_ties;",
    "unit_semantic_order=kind,canonical_position,canonical_facing,hp,max_hp,mana,max_mana,move_speed,attack_damage,attack_range,attack_time,attack_point,attack_speed,armor,magic_resistance,bound,collision,vision,true_sight,statuses,item_slot_count,free_item_slots,item_capacity_available,canonical_velocity,hp_delta,mana_delta,recent_damage,recent_cast,recent_attack;",
    "unit_memory=units_exact_current_pointer_order,own_units_fixed_hero_courier_current_or_remembered,remembered_units_nonown_hidden_cap32_lexicographic_complete_encoded_token,tracker_cap4096_evict_complete_oldest_invisible_last_seen_tick_cohorts,no_target_handles;",
    "point_candidates=cap64,deduplicate_position_keep_first_source,canonical_team_directions,generate:general_cap48_tactical_radii200_600_1200_in_E_NE_N_NW_W_SW_S_SE_order_then_allied_building_landings_then_nearest8_visible_or_static-baseline_trees_then_own_fountain_enemy_fountain_own_tower_enemy_tower_then_predicted_units,then_live_own_hero_raze_only_cap16_facing_then_best_landing_near_mid_far_then_fogged_enemy_heroes_last_seen_extrapolated_then_blind_ring8;",
    "point_order=building:distance_kind_canonical_landing_position_entity_id_identical_tie,tree:distance_canonical_position_planted,predicted:distance_source_relation_canonical_position_entity_id_identical_tie,landmark:distance_canonical_position_entity_id_identical_tie;shop_order=item_id;",
    "loot_candidates=current_visible,cap16,order:item_then_charges_then_position_then_entity_id_only_for_semantically_identical_ties;",
    "projectile_order=lexicographic_encoded_semantics,select_first32,feature_identical_ties_indistinguishable;",
    "projectile_history=continuous_full_handle_observation,cap4096,first_age,second_velocity,closest_approach,disappearance_or_generation_resets;loot_history=continuous_full_handle_observation,cap16,visible_age,duplicate_current_semantics_suppress_identity_age;",
    "mutual_attack_range=distance_squared_le_square_attack_range_plus_both_bound_radii,fixed_saturating_sum,inclusive,no_continuation_leeway;",
    "observation_journal=16_states,strictly_increasing_observe,exact_tracker_lineage_slot_static_snapshot_predecessor_and_tracker_history_provenance,rollback_at_or_before,eviction_horizon_exact_error_and_atomic,reset_empty,failed_observe_and_encode_atomic;",
    "loot_path=static-terrain-and-static-tree-grid;",
    "map_rays=E_NE_N_NW_W_SW_S_SE,20_cells,64_world_units_per_step,first_nonwalkable_water_opaque_tree_and_endpoint_elevation_walkability;",
    "visibility=allied_current_units,positive_vision_radius,within_radius,target_elevation_not_above_viewer,exact_fixed_point_supercover_line,intermediate_opaque_or_higher_cell_blocks,corner_touch_checks_both_cells;",
    "trees=opaque_cells_include_static_map_tree_cells,static_occupancy_baseline,dynamic_delta_proof_requires_live_allied_body_in_same_or_adjacent_cell,proof_ignores_all_dynamic_tree_entries,local_felled_unblocks_passability_and_tree_mask,local_planted_blocks_passability_and_enters_tree_mask,remote_dynamic_changes_invariant,no_hidden_dynamic_blocker_channel;",
    "events=input_batch_cap:payload_len_div_2:2097152_reject_before_mutation,snapshot_event_journal_cap64,only_ticks_strictly_before_snapshot,same_tick_delivery_cannot_overwrite_or_evict_prior_snapshot_features,ability_cast_age_per_caster_and_ability,combat_phase_for_any_tracked_source;",
    "readiness=wire_item_mute_exact,local_backpack_mute:180_from_apply_tick,effective_mute_max_wire_and_local,teleport_shared_wait:2100_from_apply_tick,hero_inventory_journals6,body_shared_journals2,recent_request_cap8,effective_evicted_base_retained,rejection_exact_for_retained_sequences,evicted_sequence_rejection_unsupported,retained_rejections_restore_base;",
    "local_rollback=active_transition_cap16,evicted_effective_base_retained,earliest_supported_tick_tracked,rollback_before_horizon_exact_error_and_atomic,decision_eviction_advances_horizon_to_incoming_tick;",
    "candidate_order=opt_in_live_pure_neural_ppo_learner_and_greedy_candidate_neuralseat_learner;legacy_order=teacher_tactical_hybrid_frozen_opponents_expert_collection_and_dagger_labeler_ledger_unchanged;",
    "own_cast=implicit_own_shadow_fiend_nonpassive_visible_aimOwn_ability13_14_15_16_targetNone,preserve_body_directive_sequence_kind_target_start_and_pending_rollback_advance_actual_sent_sequence,not_attack_animation_phase;other_casts_use_put_take_legacy_interruption;",
    "reconcile=complete_snapshot_and_current_tick_events_before_features,full_validated_snapshot_living_handles_not_cap96;fallback=absent_full_handle_attack_or_move_unit_to_last_observed_point_only_immediately_previous_tick_without_observed_death,keep_request_sequence,set_hero_AttackMovePoint_or_MovePoint_start_at_fallback_tick,reappearance_does_not_restore_unit_or_reset_age;unknown_target_or_unproved_final_position_clear_not_Stop;",
    "candidate_ledgers=separate_legacy_actual_requests_and_candidate_effective_directives,two_bodies_one_previous_transition_each_one_hero_pending_rollback,reconcile_current_and_rollback_atomically_after_local_chronology_check,older_rejections_noop,hero_lifecycle_clears_both_hero_states,courier_death_disappearance_generation_invalidates_only_candidate_courier;prefix_replay=observations_actual_sends_rejections_and_explicit_candidate_role_both_ledgers;",
    "own_payloads=live_body_current,hero_scoreboard_kit_current_with_source_bit,absent_courier_ability_and_item_payloads_missing,no_remembered_body_payload_fallback;",
    "provenance=private_nonzero_checked_tracker_lineage_clone_gets_fresh_lineage_move_preserves_lineage,action_space_exact_bounded_lineage_slot_static_snapshot_tracker_comparison,readiness_exact_bounded_comparison,observation_exact_bounded_lineage_slot_static_snapshot_tracker_comparison,encoder_static_exact_bounded_comparison,no_correctness_claim_for_fnv_schema_hash;",
    "ids=entity_match_seed_tracker_lineage_excluded_from_frame,entity_full_handle_only_memory_key_and_final_identical_tie,ability_and_item_semantic_categories_visible;",
    "map2_reward=only_map2_tracker_owned_complete_contiguous_seat_snapshot_events,public_scoreboard_buildings_and_visible_heroes_only,complete_pair_required_before_observe_and_encode,drain_does_not_invalidate_features,finish_advances_revision;reward_schema=linked_version;",
    "rebase_effects=Guarded13_Inspired14_Shadowraze15;auras=unit76:guarded_remaining_div15,77:inspired_remaining_div15,positive_means_presence,anonymous_max_timer_valid1..15_no_stacks,age_subtracted_no_hidden_refresh_or_source,expired_absent_dead_zero;",
    "map2_scope=cap27900_including900_pregame_15minute_gameplay;",
    "mango=item42_category43_existing_item_and_loot_token_fields,no_new_item_rows;map2_geometry=map0_public_geometry_no_metadata_relabel;",
    "mastery_stage_thresholds_and_rolling_window_excluded_from_features;",
);

/// FNV-1a of the descriptor, action version-le32/hash-le64 pair, and reward version.
pub const FEATURE_SCHEMA_HASH: u64 = crate::model::linked_schema_hash(
    FEATURE_SCHEMA_DESCRIPTOR,
    &[(crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH)],
);

const _: () = assert!(crate::ACTION_SCHEMA_VERSION == 8);
const _: () = assert!(StatusFlags::INVULNERABLE == 1 << 9);
const _: () = assert!(StatusFlags::CHANNELLING == 1 << 10);
const _: () = assert!(crate::MAP2_TICK_CAP == 27_900);
// The descriptor's shapes line spells these widths out.
const _: () = assert!(
    GLOBAL_FEATURES == 81
        && HISTORY_FEATURES == 12
        && POLICY_HISTORY_FEATURES == ActionKind::COUNT
        && UNIT_FEATURES == 109
        && POINT_FEATURES == 60
        && ABILITY_FEATURES == 56
        && ITEM_FEATURES == 102
        && ITEM_FEATURE_TOKENS == 94
        && PROJECTILE_FEATURES == 41
        && LOOT_FEATURES == 75
        && MAP_FEATURES == 87
);
const _: () = assert!(
    global_feature::ACTIVE_ORDER_KIND_START + ActionKind::COUNT == global_feature::ACTIVE_ORDER_AGE
);
const _: () = assert!(map_feature::RAYS_START + 8 * 10 == MAP_FEATURES);
const _: () = assert!(ability_feature::ID_START + ABILITY_ID_CLASSES == ability_feature::LEVEL);
const _: () = assert!(item_feature::ITEM_START + ITEM_ID_CLASSES == item_feature::CHARGES_PRESENT);
const _: () = assert!(crate::MAX_TRACKED_ENTITIES == 4_096);
const _: () = assert!(MAX_PROJECTILES == 4_096);
const _: () = assert!(PROJECTILE_FEATURE_TOKENS == 32);
const _: () = assert!(PROJECTILE_FEATURE_TOKENS < MAX_PROJECTILES);

/// One fixed-shape policy input frame owned by its caller.
///
/// Encoding writes into this storage without allocation. The arrays can be
/// passed directly to a model backend or retained as bounded local history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FeatureFrameProvenance {
    lineage: NonZeroU64,
    revision: u64,
    tick: u32,
    readiness: ItemReadiness,
}

impl FeatureFrameProvenance {
    pub(crate) const fn new(
        lineage: NonZeroU64,
        revision: u64,
        tick: u32,
        readiness: ItemReadiness,
    ) -> Self {
        Self {
            lineage,
            revision,
            tick,
            readiness,
        }
    }
}

/// Token and scalar storage lives on the heap: a frame is about 126 KiB, and
/// decisions move several of them through threads with small stacks.
#[derive(Clone, Debug)]
pub struct FeatureFrame {
    provenance: Option<FeatureFrameProvenance>,
    pub(crate) global: Box<[f32; GLOBAL_FEATURES]>,
    pub(crate) history: Box<[[f32; HISTORY_FEATURES]; HISTORY_SAMPLES]>,
    pub(crate) policy_history: Box<[[f32; POLICY_HISTORY_FEATURES]; MAX_POLICY_HISTORY]>,
    pub(crate) units: Box<[[f32; UNIT_FEATURES]; UNIT_FEATURE_TOKENS]>,
    pub(crate) own_units: Box<[[f32; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS]>,
    pub(crate) remembered_units: Box<[[f32; UNIT_FEATURES]; REMEMBERED_UNIT_FEATURE_TOKENS]>,
    pub(crate) points: Box<[[f32; POINT_FEATURES]; POINT_FEATURE_TOKENS]>,
    pub(crate) abilities: Box<[[f32; ABILITY_FEATURES]; ABILITY_FEATURE_TOKENS]>,
    pub(crate) items: Box<[[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS]>,
    pub(crate) projectiles: Box<[[f32; PROJECTILE_FEATURES]; PROJECTILE_FEATURE_TOKENS]>,
    pub(crate) loot: Box<[[f32; LOOT_FEATURES]; LOOT_FEATURE_TOKENS]>,
    pub(crate) map: Box<[f32; MAP_FEATURES]>,
}

/// Heap bytes one frame owns beside its inline handles.
pub const FEATURE_FRAME_HEAP_BYTES: usize = 4
    * (GLOBAL_FEATURES
        + HISTORY_SAMPLES * HISTORY_FEATURES
        + MAX_POLICY_HISTORY * POLICY_HISTORY_FEATURES
        + (UNIT_FEATURE_TOKENS + OWN_UNIT_FEATURE_TOKENS + REMEMBERED_UNIT_FEATURE_TOKENS)
            * UNIT_FEATURES
        + POINT_FEATURE_TOKENS * POINT_FEATURES
        + ABILITY_FEATURE_TOKENS * ABILITY_FEATURES
        + ITEM_FEATURE_TOKENS * ITEM_FEATURES
        + PROJECTILE_FEATURE_TOKENS * PROJECTILE_FEATURES
        + LOOT_FEATURE_TOKENS * LOOT_FEATURES
        + MAP_FEATURES);

/// A zeroed fixed-size array built directly on the heap.
fn zeroed<T: Copy + Default, const N: usize>() -> Box<[T; N]> {
    vec![T::default(); N]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("vector has exactly N elements"))
}

fn zeroed_rows<const F: usize, const N: usize>() -> Box<[[f32; F]; N]> {
    vec![[0.0; F]; N]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("vector has exactly N rows"))
}

impl FeatureFrame {
    /// Creates a zeroed fixed-shape frame.
    pub fn new() -> Self {
        Self {
            provenance: None,
            global: zeroed(),
            history: zeroed_rows(),
            policy_history: zeroed_rows(),
            units: zeroed_rows(),
            own_units: zeroed_rows(),
            remembered_units: zeroed_rows(),
            points: zeroed_rows(),
            abilities: zeroed_rows(),
            items: zeroed_rows(),
            projectiles: zeroed_rows(),
            loot: zeroed_rows(),
            map: zeroed(),
        }
    }

    /// Zeroes every value in place, keeping the allocations.
    fn clear(&mut self) {
        self.provenance = None;
        self.global.fill(0.0);
        self.history.as_flattened_mut().fill(0.0);
        self.policy_history.as_flattened_mut().fill(0.0);
        self.units.as_flattened_mut().fill(0.0);
        self.own_units.as_flattened_mut().fill(0.0);
        self.remembered_units.as_flattened_mut().fill(0.0);
        self.points.as_flattened_mut().fill(0.0);
        self.abilities.as_flattened_mut().fill(0.0);
        self.items.as_flattened_mut().fill(0.0);
        self.projectiles.as_flattened_mut().fill(0.0);
        self.loot.as_flattened_mut().fill(0.0);
        self.map.fill(0.0);
    }

    /// Global scalar features in stable schema order.
    #[cfg(test)]
    pub fn global(&self) -> &[f32; GLOBAL_FEATURES] {
        &self.global
    }

    /// Fixed own hero and courier unit tokens.
    #[cfg(test)]
    pub fn own_units(&self) -> &[[f32; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS] {
        &self.own_units
    }

    /// Non-targetable remembered unit tokens.
    #[cfg(test)]
    pub fn remembered_units(&self) -> &[[f32; UNIT_FEATURES]; REMEMBERED_UNIT_FEATURE_TOKENS] {
        &self.remembered_units
    }

    /// Point tokens in exact point-pointer order.
    #[cfg(test)]
    pub fn points(&self) -> &[[f32; POINT_FEATURES]; POINT_FEATURE_TOKENS] {
        &self.points
    }

    /// Whether every scalar in the frame is finite.
    pub fn is_finite(&self) -> bool {
        let fields: [&[f32]; 12] = [
            &self.global[..],
            self.history.as_flattened(),
            self.policy_history.as_flattened(),
            self.units.as_flattened(),
            self.own_units.as_flattened(),
            self.remembered_units.as_flattened(),
            self.points.as_flattened(),
            self.abilities.as_flattened(),
            self.items.as_flattened(),
            self.projectiles.as_flattened(),
            self.loot.as_flattened(),
            &self.map[..],
        ];
        fields
            .into_iter()
            .all(|field| field.iter().all(|value| value.is_finite()))
    }

    pub(crate) fn matches_action_space(&self, action_space: &ActionSpace) -> bool {
        self.provenance == Some(action_space.feature_frame_provenance())
    }
}

impl PartialEq for FeatureFrame {
    fn eq(&self, other: &Self) -> bool {
        self.global == other.global
            && self.history == other.history
            && self.policy_history == other.policy_history
            && self.units == other.units
            && self.own_units == other.own_units
            && self.remembered_units == other.remembered_units
            && self.points == other.points
            && self.abilities == other.abilities
            && self.items == other.items
            && self.projectiles == other.projectiles
            && self.loot == other.loot
            && self.map == other.map
    }
}

impl Default for FeatureFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
struct IndexedFeatureRow<const FEATURES: usize> {
    index: u16,
    values: [f32; FEATURES],
}

const FEATURE_ARENA_BYTES: [u64; 7] = [
    feature_row_bytes::<UNIT_FEATURE_TOKENS, UNIT_FEATURES>(),
    feature_row_bytes::<REMEMBERED_UNIT_FEATURE_TOKENS, UNIT_FEATURES>(),
    feature_row_bytes::<POINT_FEATURE_TOKENS, POINT_FEATURES>(),
    feature_row_bytes::<ABILITY_FEATURE_TOKENS, ABILITY_FEATURES>(),
    feature_row_bytes::<ITEM_FEATURE_TOKENS, ITEM_FEATURES>(),
    feature_row_bytes::<PROJECTILE_FEATURE_TOKENS, PROJECTILE_FEATURES>(),
    feature_row_bytes::<LOOT_FEATURE_TOKENS, LOOT_FEATURES>(),
];

/// Capped row-vector bytes plus one largest arena's moving-reallocation overlap.
/// Excludes headers, transitions, materialized minibatches, and allocator overhead;
/// this is not a bound on the process's total memory consumption.
pub(crate) const FEATURE_ARENA_PEAK_BYTES: u64 = {
    let mut total = 0;
    let mut largest = 0;
    let mut index = 0;
    while index < FEATURE_ARENA_BYTES.len() {
        let bytes = FEATURE_ARENA_BYTES[index];
        total += bytes;
        if bytes > largest {
            largest = bytes;
        }
        index += 1;
    }
    total + largest
};

const fn feature_row_bytes<const TOKENS: usize, const FEATURES: usize>() -> u64 {
    assert!(TOKENS > 0);
    assert!(TOKENS <= u16::MAX as usize);
    let rows = crate::PPO_MAX_SAMPLES as u64 * TOKENS as u64;
    assert!(rows <= u32::MAX as u64);
    rows * std::mem::size_of::<IndexedFeatureRow<FEATURES>>() as u64
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeatureRowRange {
    offset: u32,
    count: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct RaggedFeatureHeader {
    provenance: Option<FeatureFrameProvenance>,
    global: [f32; GLOBAL_FEATURES],
    history: [[f32; HISTORY_FEATURES]; HISTORY_SAMPLES],
    policy_history: [[f32; POLICY_HISTORY_FEATURES]; MAX_POLICY_HISTORY],
    own_units: [[f32; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS],
    map: [f32; MAP_FEATURES],
    units: FeatureRowRange,
    remembered_units: FeatureRowRange,
    points: FeatureRowRange,
    abilities: FeatureRowRange,
    items: FeatureRowRange,
    projectiles: FeatureRowRange,
    loot: FeatureRowRange,
}

pub(crate) struct RaggedFeatureArena {
    sample_capacity: usize,
    units: Vec<IndexedFeatureRow<UNIT_FEATURES>>,
    remembered_units: Vec<IndexedFeatureRow<UNIT_FEATURES>>,
    points: Vec<IndexedFeatureRow<POINT_FEATURES>>,
    abilities: Vec<IndexedFeatureRow<ABILITY_FEATURES>>,
    items: Vec<IndexedFeatureRow<ITEM_FEATURES>>,
    projectiles: Vec<IndexedFeatureRow<PROJECTILE_FEATURES>>,
    loot: Vec<IndexedFeatureRow<LOOT_FEATURES>>,
}

impl RaggedFeatureArena {
    /// Every row vector grows fallibly within its row cap.
    pub(crate) fn new(sample_capacity: usize) -> Result<Self, &'static str> {
        if !(1..=crate::PPO_MAX_SAMPLES).contains(&sample_capacity) {
            return Err("ragged feature sample capacity is outside 1..=33280");
        }
        Ok(Self {
            sample_capacity,
            units: Vec::new(),
            remembered_units: Vec::new(),
            points: Vec::new(),
            abilities: Vec::new(),
            items: Vec::new(),
            projectiles: Vec::new(),
            loot: Vec::new(),
        })
    }

    pub(crate) fn push(
        &mut self,
        frame: &FeatureFrame,
    ) -> Result<RaggedFeatureHeader, &'static str> {
        self.reserve_frame(frame)?;
        let units = append_feature_rows(
            &mut self.units,
            &frame.units,
            unit_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let remembered_units = append_feature_rows(
            &mut self.remembered_units,
            &frame.remembered_units,
            unit_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let points = append_feature_rows(
            &mut self.points,
            &frame.points,
            point_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let abilities = append_feature_rows(
            &mut self.abilities,
            &frame.abilities,
            ability_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let items = append_feature_rows(
            &mut self.items,
            &frame.items,
            item_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let projectiles = append_feature_rows(
            &mut self.projectiles,
            &frame.projectiles,
            projectile_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        let loot = append_feature_rows(
            &mut self.loot,
            &frame.loot,
            loot_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        Ok(RaggedFeatureHeader {
            provenance: frame.provenance,
            global: *frame.global,
            history: *frame.history,
            policy_history: *frame.policy_history,
            own_units: *frame.own_units,
            map: *frame.map,
            units,
            remembered_units,
            points,
            abilities,
            items,
            projectiles,
            loot,
        })
    }

    fn reserve_frame(&mut self, frame: &FeatureFrame) -> Result<(), &'static str> {
        assert!(self.sample_capacity <= crate::PPO_MAX_SAMPLES);
        // Reserve all seven arenas first so an allocation failure cannot append partial rows.
        reserve_feature_rows(
            &mut self.units,
            &frame.units,
            unit_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.remembered_units,
            &frame.remembered_units,
            unit_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.points,
            &frame.points,
            point_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.abilities,
            &frame.abilities,
            ability_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.items,
            &frame.items,
            item_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.projectiles,
            &frame.projectiles,
            projectile_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )?;
        reserve_feature_rows(
            &mut self.loot,
            &frame.loot,
            loot_feature::TOKEN_PRESENT,
            self.sample_capacity,
        )
    }

    pub(crate) fn expand(
        &self,
        header: &RaggedFeatureHeader,
    ) -> Result<FeatureFrame, &'static str> {
        let mut frame = FeatureFrame {
            provenance: header.provenance,
            global: Box::new(header.global),
            history: Box::new(header.history),
            policy_history: Box::new(header.policy_history),
            own_units: Box::new(header.own_units),
            map: Box::new(header.map),
            ..FeatureFrame::new()
        };
        restore_feature_rows(&self.units, header.units, &mut frame.units)?;
        restore_feature_rows(
            &self.remembered_units,
            header.remembered_units,
            &mut frame.remembered_units,
        )?;
        restore_feature_rows(&self.points, header.points, &mut frame.points)?;
        restore_feature_rows(&self.abilities, header.abilities, &mut frame.abilities)?;
        restore_feature_rows(&self.items, header.items, &mut frame.items)?;
        restore_feature_rows(
            &self.projectiles,
            header.projectiles,
            &mut frame.projectiles,
        )?;
        restore_feature_rows(&self.loot, header.loot, &mut frame.loot)?;
        Ok(frame)
    }
}

fn reserve_feature_rows<const TOKENS: usize, const FEATURES: usize>(
    arena: &mut Vec<IndexedFeatureRow<FEATURES>>,
    rows: &[[f32; FEATURES]; TOKENS],
    presence: usize,
    sample_capacity: usize,
) -> Result<(), &'static str> {
    let maximum = bounded_feature_row_capacity::<TOKENS>(sample_capacity)?;
    if presence >= FEATURES {
        return Err("ragged feature presence index out of range");
    }
    let count = rows.iter().filter(|row| row[presence] == 1.0).count();
    let end = arena
        .len()
        .checked_add(count)
        .filter(|end| *end <= maximum)
        .ok_or("ragged feature arena capacity exceeded")?;
    reserve_feature_capacity(arena, end, maximum)
}

fn bounded_feature_row_capacity<const TOKENS: usize>(
    sample_capacity: usize,
) -> Result<usize, &'static str> {
    if TOKENS == 0 || TOKENS > u16::MAX as usize {
        return Err("ragged feature token count is outside 1..=65535");
    }
    let maximum = sample_capacity
        .checked_mul(TOKENS)
        .ok_or("ragged feature arena capacity overflow")?;
    u32::try_from(maximum).map_err(|_| "ragged feature arena offset capacity exceeds u32")?;
    Ok(maximum)
}

fn reserve_feature_capacity<const FEATURES: usize>(
    arena: &mut Vec<IndexedFeatureRow<FEATURES>>,
    end: usize,
    maximum: usize,
) -> Result<(), &'static str> {
    if end < arena.len() || end > maximum {
        return Err("ragged feature arena capacity exceeded");
    }
    if arena.capacity() > maximum {
        return Err("ragged feature allocated capacity exceeds maximum");
    }
    if end <= arena.capacity() {
        return Ok(());
    }
    let capacity = arena.capacity().saturating_mul(2).max(end).min(maximum);
    assert!(capacity >= end);
    assert!(capacity <= maximum);
    arena
        .try_reserve_exact(capacity - arena.len())
        .map_err(|_| "ragged feature arena allocation failed")?;
    if arena.capacity() > maximum {
        return Err("ragged feature allocated capacity exceeds maximum");
    }
    assert!(arena.capacity() >= end);
    Ok(())
}

fn append_feature_rows<const TOKENS: usize, const FEATURES: usize>(
    arena: &mut Vec<IndexedFeatureRow<FEATURES>>,
    rows: &[[f32; FEATURES]; TOKENS],
    presence: usize,
    sample_capacity: usize,
) -> Result<FeatureRowRange, &'static str> {
    let count = rows.iter().filter(|row| row[presence] == 1.0).count();
    let maximum = sample_capacity
        .checked_mul(TOKENS)
        .ok_or("ragged feature arena capacity overflow")?;
    let end = arena
        .len()
        .checked_add(count)
        .filter(|end| *end <= maximum)
        .ok_or("ragged feature arena capacity exceeded")?;
    let offset = u32::try_from(arena.len()).map_err(|_| "ragged feature offset overflow")?;
    for (index, values) in rows.iter().enumerate() {
        if values[presence] == 1.0 {
            arena.push(IndexedFeatureRow {
                index: u16::try_from(index).map_err(|_| "ragged feature index overflow")?,
                values: *values,
            });
        }
    }
    if arena.len() != end {
        return Err("ragged feature row count mismatch");
    }
    Ok(FeatureRowRange {
        offset,
        count: u16::try_from(count).map_err(|_| "ragged feature count overflow")?,
    })
}

fn restore_feature_rows<const TOKENS: usize, const FEATURES: usize>(
    arena: &[IndexedFeatureRow<FEATURES>],
    range: FeatureRowRange,
    rows: &mut [[f32; FEATURES]; TOKENS],
) -> Result<(), &'static str> {
    let offset = range.offset as usize;
    let end = offset
        .checked_add(range.count as usize)
        .filter(|end| *end <= arena.len())
        .ok_or("ragged feature range is invalid")?;
    for row in &arena[offset..end] {
        let target = rows
            .get_mut(row.index as usize)
            .ok_or("ragged feature token index is invalid")?;
        *target = row.values;
    }
    Ok(())
}

/// One local decision retained for policy-history features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyDecision {
    /// Snapshot tick used for the decision.
    pub tick: u32,
    /// Selected top-level action family.
    pub kind: ActionKind,
}

/// One locally active order and its deterministic start tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivePolicyOrder {
    /// Tick at which the policy selected the order.
    pub started_tick: u32,
    /// Selected top-level action family.
    pub kind: ActionKind,
    /// Identifier-free target semantics retained outside model tensors.
    pub target: ActivePolicyTarget,
}

/// Target of one locally active persistent body order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivePolicyTarget {
    None,
    Point(Vec2),
    Unit(EntityId),
}

/// A tick older than the latest one supplied to [`LocalPolicyState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPolicyError {
    TickRegression { incoming: u32, latest: u32 },
}

impl fmt::Display for LocalPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TickRegression { incoming, latest } => write!(
                formatter,
                "local policy tick {incoming} is older than latest tick {latest}"
            ),
        }
    }
}

impl Error for LocalPolicyError {}

/// Local decision history and active order behind the policy-history and active-order inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalPolicyState {
    latest_tick: u32,
    decisions: [Option<PolicyDecision>; MAX_POLICY_HISTORY],
    decision_count: usize,
    active: Option<ActivePolicyOrder>,
}

impl LocalPolicyState {
    /// Creates empty local state at `tick`.
    pub const fn new(tick: u32) -> Self {
        Self {
            latest_tick: tick,
            decisions: [None; MAX_POLICY_HISTORY],
            decision_count: 0,
            active: None,
        }
    }

    /// Records one decision, evicting the oldest entry at the fixed bound.
    pub fn note_decision(&mut self, tick: u32, kind: ActionKind) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        if self.decision_count == MAX_POLICY_HISTORY {
            self.decisions.copy_within(1..MAX_POLICY_HISTORY, 0);
            self.decision_count -= 1;
        }
        self.decisions[self.decision_count] = Some(PolicyDecision { tick, kind });
        self.decision_count += 1;
        self.latest_tick = tick;
        Ok(())
    }

    /// Replaces or clears the current active order at one local tick.
    pub fn set_active_order(
        &mut self,
        tick: u32,
        kind: Option<ActionKind>,
    ) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        let order = kind.map(|kind| ActivePolicyOrder {
            started_tick: tick,
            kind,
            target: ActivePolicyTarget::None,
        });
        self.active = order;
        self.latest_tick = tick;
        Ok(())
    }

    /// Replaces the active order, keeping the move or attack target of `issued`.
    pub fn set_active_order_from_issued(
        &mut self,
        tick: u32,
        kind: ActionKind,
        issued: IssuedOrder,
    ) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        let target = match issued.order {
            Order::Move { target } | Order::Attack { target } => match target {
                Target::None => ActivePolicyTarget::None,
                Target::Pos(position) => ActivePolicyTarget::Point(position),
                Target::Unit(unit) => ActivePolicyTarget::Unit(unit),
            },
            _ => ActivePolicyTarget::None,
        };
        let order = Some(ActivePolicyOrder {
            started_tick: tick,
            kind,
            target,
        });
        self.active = order;
        self.latest_tick = tick;
        Ok(())
    }

    /// Restores a previously captured active order at one rejection-observation tick.
    pub fn restore_active_order(
        &mut self,
        tick: u32,
        order: Option<ActivePolicyOrder>,
    ) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        self.active = order;
        self.latest_tick = tick;
        Ok(())
    }

    /// Decisions in oldest-to-newest order.
    pub fn decisions(&self) -> impl DoubleEndedIterator<Item = &PolicyDecision> {
        self.decisions[..self.decision_count]
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Current active order.
    pub const fn active_order(&self) -> Option<ActivePolicyOrder> {
        self.active
    }

    fn check_tick(&self, tick: u32) -> Result<(), LocalPolicyError> {
        if tick < self.latest_tick {
            return Err(LocalPolicyError::TickRegression {
                incoming: tick,
                latest: self.latest_tick,
            });
        }
        Ok(())
    }
}

/// Feature construction failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureError {
    Map2RewardIncomplete { snapshot: u32 },
    SnapshotRequired,
    TickMismatch { snapshot: u32, action_space: u32 },
    ActionSpaceMismatch,
    ReadinessMismatch,
    MapMismatch,
    LocalStateAhead { snapshot: u32, local: u32 },
    ObservationRequired { snapshot: u32 },
    ObservationMismatch { snapshot: u32 },
    ObservationTickNotIncreasing { incoming: u32, latest: u32 },
    ObservationPredecessorMismatch { incoming: u32 },
    NonFinite,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Map2RewardIncomplete { snapshot } => write!(
                formatter,
                "feature Map2 reward requires complete Snapshot/Events for tick {snapshot}"
            ),
            Self::SnapshotRequired => formatter.write_str("feature encoding requires a snapshot"),
            Self::TickMismatch {
                snapshot,
                action_space,
            } => write!(
                formatter,
                "feature snapshot tick {snapshot} differs from action-space tick {action_space}"
            ),
            Self::ActionSpaceMismatch => {
                formatter.write_str("feature action space belongs to a different snapshot")
            }
            Self::ReadinessMismatch => {
                formatter.write_str("feature item readiness differs from action space")
            }
            Self::MapMismatch => {
                formatter.write_str("feature encoder map context differs from tracker")
            }
            Self::LocalStateAhead { snapshot, local } => write!(
                formatter,
                "local policy tick {local} is newer than snapshot tick {snapshot}"
            ),
            Self::ObservationRequired { snapshot } => write!(
                formatter,
                "feature observation for snapshot tick {snapshot} is required"
            ),
            Self::ObservationMismatch { snapshot } => write!(
                formatter,
                "feature observation belongs to a different snapshot at tick {snapshot}"
            ),
            Self::ObservationTickNotIncreasing { incoming, latest } => write!(
                formatter,
                "feature observation tick {incoming} must be greater than latest tick {latest}"
            ),
            Self::ObservationPredecessorMismatch { incoming } => write!(
                formatter,
                "feature observation snapshot tick {incoming} does not extend its exact predecessor"
            ),
            Self::NonFinite => formatter.write_str("feature encoder produced a non-finite value"),
        }
    }
}

impl Error for FeatureError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProjectileObservation {
    id: bota_proto::EntityId,
    first_tick: u32,
    last_tick: u32,
    previous_tick: Option<u32>,
    position: Vec2,
    previous_position: Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LootObservation {
    id: bota_proto::EntityId,
    first_tick: u32,
    last_tick: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct FeatureObservationState {
    tick: Option<u32>,
    /// Shared structural proof; every clone is an Arc bump.
    provenance: Option<std::sync::Arc<TrackerProvenance>>,
    /// One entry per live projectile in wire order; bounded by
    /// [`MAX_PROJECTILES`].
    projectiles: Vec<ProjectileObservation>,
    /// One entry per live loot crate in wire order; bounded by [`MAX_LOOT`].
    loot: Vec<LootObservation>,
}

impl FeatureObservationState {
    fn new() -> Self {
        Self {
            tick: None,
            provenance: None,
            projectiles: Vec::new(),
            loot: Vec::new(),
        }
    }
}

/// Static public map data decoded once for repeated feature encoding.
///
/// Terrain, static tree sight blockers, and tree locations come from `MatchInfo`.
/// A dynamic tree delta is accepted only in the same or an adjacent terrain cell
/// to a current allied body. This proof uses no dynamic tree-list entry; remote
/// entries preserve the static baseline and cannot change policy observations.
pub struct FeatureEncoder {
    map: bota_proto::MapId,
    static_provenance: std::sync::Arc<StaticTrackerProvenance>,
    lineage: NonZeroU64,
    axis: usize,
    extent_raw: i64,
    terrain: Vec<u8>,
    opaque: Vec<bool>,
    static_tree_index: Vec<(usize, u32)>,
    path_distances: Vec<u32>,
    path_queue: Vec<usize>,
    path_origin: Option<usize>,
    /// Target cells of the last flood fill; with `path_exhausted` this decides
    /// whether the cached distances answer a new query without refilling.
    path_targets: Vec<u32>,
    path_exhausted: bool,
    /// Static impassability table for the loot path flood fill: non-walkable
    /// terrain or a static tree. Built once because terrain and the static
    /// tree index never change for an encoder.
    path_impassable: Vec<bool>,
    observation: std::sync::Arc<FeatureObservationState>,
}

impl FeatureEncoder {
    /// Decodes the static map context of `tracker`'s match.
    pub fn new(tracker: &StateTracker) -> Self {
        let axis = usize::try_from(tracker.metadata().terrain_cells)
            .expect("validated terrain axis fits usize");
        let mut terrain = Vec::with_capacity(axis * axis);
        for &(run, cell) in tracker.terrain_rle() {
            terrain.resize(terrain.len() + usize::from(run), cell);
        }
        let mut opaque = vec![false; axis * axis];
        for &(x, y) in tracker.opaque_cells() {
            opaque[usize::from(y) * axis + usize::from(x)] = true;
        }
        let static_trees = tracker.static_trees().to_vec();
        let mut static_tree_index = Vec::with_capacity(static_trees.len());
        for (index, position) in static_trees.iter().copied().enumerate() {
            if let Some(cell) = cell_index(axis, position) {
                static_tree_index.push((
                    cell,
                    u32::try_from(index).expect("bounded static tree index fits u32"),
                ));
            }
        }
        static_tree_index.sort_unstable();
        let mut path_impassable = vec![false; axis * axis];
        for (cell, _) in &static_tree_index {
            path_impassable[*cell] = true;
        }
        for (cell, terrain_byte) in terrain.iter().enumerate() {
            if terrain_byte & 0x80 == 0 {
                path_impassable[cell] = true;
            }
        }
        Self {
            map: tracker.metadata().map,
            static_provenance: tracker.static_provenance(),
            lineage: tracker.lineage(),
            axis,
            extent_raw: map_extent_raw(axis),
            terrain,
            opaque,
            static_tree_index,
            path_distances: vec![u32::MAX; axis * axis],
            path_queue: Vec::with_capacity(axis * axis),
            path_origin: None,
            path_targets: Vec::new(),
            path_exhausted: false,
            path_impassable,
            observation: std::sync::Arc::new(FeatureObservationState::new()),
        }
    }

    /// Records projectile and loot observations exactly once for one snapshot.
    pub fn observe(&mut self, tracker: &StateTracker) -> Result<(), FeatureError> {
        let current = tracker.current().ok_or(FeatureError::SnapshotRequired)?;
        self.validate_map(tracker)?;
        validate_map2_reward(tracker, current.tick)?;
        if tracker.lineage() != self.lineage {
            return Err(FeatureError::ObservationPredecessorMismatch {
                incoming: current.tick,
            });
        }
        if let Some(latest) = self.observation.tick
            && current.tick <= latest
        {
            return Err(FeatureError::ObservationTickNotIncreasing {
                incoming: current.tick,
                latest,
            });
        }
        if self
            .observation
            .provenance
            .as_ref()
            .is_some_and(|previous| !previous.snapshot_precedes(tracker))
        {
            return Err(FeatureError::ObservationPredecessorMismatch {
                incoming: current.tick,
            });
        }
        let next = next_observation_state(&self.observation, tracker, current);
        self.push_observation(next);
        Ok(())
    }

    /// Encodes one frame from seat-observable state and local history.
    pub fn encode(
        &mut self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        readiness: &ItemReadiness,
        local: &LocalPolicyState,
        output: &mut FeatureFrame,
    ) -> Result<(), FeatureError> {
        let current = tracker.current().ok_or(FeatureError::SnapshotRequired)?;
        self.validate_inputs(tracker, action_space, readiness, local, current.tick)?;
        output.clear();
        let view = Perspective::new(tracker, current.tick);
        self.encode_global(tracker, local, &view, output);
        self.encode_history(tracker, output);
        self.encode_policy_history(local, output);
        self.encode_units(tracker, action_space, &view, output);
        self.encode_own_units(tracker, &view, output);
        self.encode_remembered_units(tracker, &view, output);
        self.encode_points(action_space, &view, output);
        self.encode_abilities(tracker, action_space, output);
        self.encode_items(tracker, action_space, readiness, &view, output);
        self.encode_projectiles(tracker, &view, output);
        self.encode_loot(tracker, action_space, output);
        self.encode_map(tracker, output);
        if !output.is_finite() {
            return Err(FeatureError::NonFinite);
        }
        output.provenance = Some(action_space.feature_frame_provenance());
        Ok(())
    }

    fn validate_inputs(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        readiness: &ItemReadiness,
        local: &LocalPolicyState,
        tick: u32,
    ) -> Result<(), FeatureError> {
        validate_map2_reward(tracker, tick)?;
        if action_space.tick() != tick {
            return Err(FeatureError::TickMismatch {
                snapshot: tick,
                action_space: action_space.tick(),
            });
        }
        if !action_space.matches_tracker(tracker) {
            return Err(FeatureError::ActionSpaceMismatch);
        }
        if !action_space.matches_readiness(readiness) {
            return Err(FeatureError::ReadinessMismatch);
        }
        self.validate_map(tracker)?;
        if self.observation.tick != Some(tick) {
            return Err(FeatureError::ObservationRequired { snapshot: tick });
        }
        if !self
            .observation
            .provenance
            .as_ref()
            .is_some_and(|provenance| provenance.matches(tracker))
        {
            return Err(FeatureError::ObservationMismatch { snapshot: tick });
        }
        if local.latest_tick > tick {
            return Err(FeatureError::LocalStateAhead {
                snapshot: tick,
                local: local.latest_tick,
            });
        }
        Ok(())
    }

    fn validate_map(&self, tracker: &StateTracker) -> Result<(), FeatureError> {
        let metadata = tracker.metadata();
        if metadata.map != self.map
            || metadata.terrain_cells as usize != self.axis
            || !self.static_provenance.matches(tracker)
        {
            return Err(FeatureError::MapMismatch);
        }
        Ok(())
    }

    fn push_observation(&mut self, observation: std::sync::Arc<FeatureObservationState>) {
        self.observation = observation;
        self.path_origin = None;
    }

    fn encode_global(
        &self,
        tracker: &StateTracker,
        local: &LocalPolicyState,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        use global_feature as index;
        let current = tracker.current().expect("snapshot was checked");
        let metadata = tracker.metadata();
        let own = tracker.own_player().expect("own player was validated");
        let global = &mut output.global;
        global[index::TICK] = unit_ratio(current.tick, crate::MAP2_TICK_CAP);
        global[index::PREGAME_PROGRESS] = pregame_progress(current.tick, metadata.pregame_ticks);
        global[index::WAVE_PHASE] =
            periodic_phase(current.tick, metadata.pregame_ticks, metadata.tick_rate, 30);
        global[index::JUNGLE_PHASE] =
            periodic_phase(current.tick, metadata.pregame_ticks, metadata.tick_rate, 60);
        global[index::SIDE_RADIANT] = bool_feature(tracker.team() == Team::Radiant);
        global[index::SIDE_DIRE] = bool_feature(tracker.team() == Team::Dire);
        global[index::OWN_ASSET_VALUE] = fraction(own_asset_value(tracker) as f32, scale::GOLD);
        encode_local_global(self, tracker, current.tick, local, global);
        encode_own_score(own, global);
        let (dealt, taken) = snapshot_damage(tracker);
        global[index::SNAPSHOT_DAMAGE_DEALT] = fraction(dealt as f32, scale::SNAPSHOT_DAMAGE);
        global[index::SNAPSHOT_DAMAGE_TAKEN] = fraction(taken as f32, scale::SNAPSHOT_DAMAGE);
        self.encode_ancient_geometry(tracker, global);
        encode_map2_reward(tracker, global);
        encode_global_combat(tracker, view, global);
    }

    fn encode_ancient_geometry(&self, tracker: &StateTracker, global: &mut [f32; GLOBAL_FEATURES]) {
        use global_feature as index;
        let Some(origin) = tracker.own_hero().map(|unit| unit.pos) else {
            return;
        };
        for (team, offset) in [
            (tracker.team(), index::OWN_ANCIENT_RELATIVE_X),
            (opposing(tracker.team()), index::ENEMY_ANCIENT_RELATIVE_X),
        ] {
            let mut positions = tracker
                .entities()
                .iter()
                .filter(|track| track.unit.kind == UnitKind::Ancient && track.unit.team == team)
                .map(|track| track.unit.pos);
            let Some(position) = positions.next() else {
                continue;
            };
            if positions.any(|other| other != position) {
                continue;
            }
            let delta = self.canonical_delta(tracker.team(), position, origin);
            global[offset] = signed_raw_ratio(delta.0, self.extent_raw);
            global[offset + 1] = signed_raw_ratio(delta.1, self.extent_raw);
            global[offset + 2] = log_distance(combat::distance(origin, position));
        }
    }

    fn encode_history(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let current_tick = tracker.current().expect("snapshot was checked").tick;
        let summaries = tracker.history();
        for (sample_index, age) in HISTORY_AGES.iter().copied().enumerate() {
            let summary = summaries[sample_index];
            let target = current_tick.saturating_sub(age);
            if age == 0 || summary.tick <= target {
                encode_history_sample(&mut output.history[sample_index], summary, current_tick);
            }
        }
    }

    fn encode_policy_history(&self, local: &LocalPolicyState, output: &mut FeatureFrame) {
        for (output_index, decision) in local.decisions().rev().enumerate() {
            one_hot(
                &mut output.policy_history[output_index],
                policy_history_feature::KIND_START,
                POLICY_HISTORY_FEATURES,
                decision.kind.index(),
            );
        }
    }

    fn encode_units(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        for (index, candidate) in action_space.entity_candidates().iter().enumerate() {
            let track = tracker
                .entity(candidate.id())
                .expect("action candidate has a visible tracker record");
            output.units[index] = self.encode_unit_track(tracker, track, view);
        }
    }

    fn encode_own_units(
        &self,
        tracker: &StateTracker,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        for (index, kind) in [UnitKind::Hero, UnitKind::Courier].into_iter().enumerate() {
            if let Some(track) = own_body_track(tracker, kind) {
                output.own_units[index] = self.encode_unit_track(tracker, track, view);
            }
        }
    }

    fn encode_remembered_units(
        &self,
        tracker: &StateTracker,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        let mut count = 0usize;
        for track in tracker.entities() {
            if snapshot_visible(tracker, track) || is_own_body_track(tracker, track) {
                continue;
            }
            let token = self.encode_unit_track(tracker, track, view);
            insert_sorted_token(&mut output.remembered_units, &mut count, token);
        }
    }

    fn encode_points(
        &self,
        action_space: &ActionSpace,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        for (index, point) in action_space.point_candidates().iter().enumerate() {
            output.points[index] = self.encode_point(point, view);
        }
    }

    fn encode_point(
        &self,
        point: &PointCandidate,
        view: &Perspective<'_>,
    ) -> [f32; POINT_FEATURES] {
        use point_feature as index;
        let mut token = [0.0; POINT_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        let position = self.canonical_position(view.team, point.position);
        token[index::POSITION_X] = coordinate_ratio(position.x.raw, self.extent_raw);
        token[index::POSITION_Y] = coordinate_ratio(position.y.raw, self.extent_raw);
        self.encode_relative(&mut token, index::RELATIVE_X, point.position, view);
        encode_point_source(&mut token, point.source);
        token[index::WALKABLE] = bool_feature(point.walkable);
        token[index::STANDING_TREE] = bool_feature(point.standing_tree);
        token[index::ALLIED_BUILDING] = bool_feature(point.allied_building);
        if let PointSource::LastSeenHero { age } | PointSource::ExtrapolatedHero { age } =
            point.source
        {
            token[index::SIGHTING_AGE] = unit_ratio(age, crate::FOG_GUESS_MAX_AGE_TICKS);
        }
        for (reach, coverage) in point.raze_coverage.iter().enumerate() {
            token[index::RAZE_UNITS_START + reach] = unit_ratio(u32::from(coverage.units), 4);
            token[index::RAZE_HEROES_START + reach] = unit_ratio(u32::from(coverage.heroes), 1);
        }
        token
    }

    /// Writes relative x, y, direction x, y, near and log distance, and bearing
    /// cos, sin into the eight slots from `start`, in that order.
    fn encode_relative(
        &self,
        token: &mut [f32],
        start: usize,
        position: Vec2,
        view: &Perspective<'_>,
    ) {
        let Some(origin) = view.origin else {
            return;
        };
        let delta = self.canonical_delta(view.team, position, origin);
        token[start] = signed_raw_ratio(delta.0, self.extent_raw);
        token[start + 1] = signed_raw_ratio(delta.1, self.extent_raw);
        let distance = combat::distance(origin, position);
        if distance > 0.0 {
            let raw_distance = distance * 65_536.0;
            token[start + 2] = (delta.0 as f32 / raw_distance).clamp(-1.0, 1.0);
            token[start + 3] = (delta.1 as f32 / raw_distance).clamp(-1.0, 1.0);
        }
        token[start + 4] = near_distance(distance);
        token[start + 5] = log_distance(distance);
        if let Some(hero) = view.hero
            && distance > 0.0
            && hero.pos == origin
        {
            let (cosine, sine) = bearing(delta, canonical_facing(view.team, hero.facing), distance);
            token[start + 6] = cosine;
            token[start + 7] = sine;
        }
    }

    fn encode_unit_track(
        &self,
        tracker: &StateTracker,
        track: &crate::EntityTrack,
        view: &Perspective<'_>,
    ) -> [f32; UNIT_FEATURES] {
        use unit_feature as index;
        let mut token = [0.0; UNIT_FEATURES];
        let unit = &track.unit;
        token[index::TOKEN_PRESENT] = 1.0;
        let relation = unit_relation(tracker, unit);
        one_hot(
            &mut token,
            index::RELATION_START,
            4,
            relation_index(relation),
        );
        one_hot(
            &mut token,
            index::KIND_START,
            12,
            unit_kind_index(unit.kind),
        );
        token[index::ENEMY_OWNED] =
            bool_feature(owner_relation(tracker, unit) == Some(EntityRelation::Enemy));
        token[index::VISIBLE] = bool_feature(snapshot_visible(tracker, track));
        let age = view.tick.saturating_sub(track.last_seen_tick);
        token[index::AGE] = unit_ratio(age, crate::HISTORY_TICKS);
        self.encode_unit_geometry(&mut token, unit, view);
        self.encode_unit_motion(&mut token, view.team, track);
        self.encode_unit_terrain(&mut token, unit.pos);
        encode_unit_resources(&mut token, track);
        encode_unit_combat(&mut token, unit);
        encode_statuses(&mut token, unit.statuses);
        encode_unit_recent(&mut token, tracker, track, view);
        encode_unit_effects(&mut token, track, view.tick);
        encode_restoration_reports(&mut token, tracker, track.id, view.tick);
        encode_hero_kit(&mut token, unit, age, relation, view);
        if let Some(hero) = view.hero {
            encode_unit_tactics(&mut token, unit, hero, relation, age, view.team);
        }
        token
    }

    fn encode_unit_geometry(
        &self,
        token: &mut [f32; UNIT_FEATURES],
        unit: &UnitView,
        view: &Perspective<'_>,
    ) {
        use unit_feature as index;
        let position = self.canonical_position(view.team, unit.pos);
        token[index::POSITION_X] = coordinate_ratio(position.x.raw, self.extent_raw);
        token[index::POSITION_Y] = coordinate_ratio(position.y.raw, self.extent_raw);
        let (cosine, sine) = canonical_facing(view.team, unit.facing);
        token[index::FACING_COS] = cosine;
        token[index::FACING_SIN] = sine;
        token[index::BOUND] = fraction(unit.bound.to_f32(), scale::BOUND);
        self.encode_relative(token, index::RELATIVE_X, unit.pos, view);
    }

    fn encode_unit_motion(
        &self,
        token: &mut [f32; UNIT_FEATURES],
        team: Team,
        track: &crate::EntityTrack,
    ) {
        let Some(velocity) = track.velocity else {
            return;
        };
        token[unit_feature::MOTION_PRESENT] = 1.0;
        let sign = if team == Team::Dire { -1.0 } else { 1.0 };
        let per_second = TICKS_PER_SECOND / velocity.elapsed_ticks.max(1) as f32;
        for (offset, raw) in [velocity.delta.x.raw, velocity.delta.y.raw]
            .into_iter()
            .enumerate()
        {
            let units = sign * raw as f32 / 65_536.0 * per_second;
            token[unit_feature::VELOCITY_X + offset] = signed_fraction(units, scale::VELOCITY);
        }
    }

    fn encode_unit_terrain(&self, token: &mut [f32; UNIT_FEATURES], position: Vec2) {
        let Some(cell) = self.cell(position) else {
            return;
        };
        let terrain = self.terrain[cell];
        token[unit_feature::ELEVATION] = ratio(i64::from(terrain & 0x3f), 0, 63);
        token[unit_feature::WALKABLE] = bool_feature(terrain & 0x80 != 0);
    }

    fn encode_abilities(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        output: &mut FeatureFrame,
    ) {
        let tick = tracker.current().expect("snapshot was checked").tick;
        for (body, unit, abilities, first, count) in [
            (
                ControlledUnit::Hero,
                own_body_track(tracker, UnitKind::Hero),
                own_hero_abilities(tracker),
                0,
                SHADOW_FIEND_ABILITY_SLOTS,
            ),
            (
                ControlledUnit::Courier,
                own_body_track(tracker, UnitKind::Courier),
                own_courier_abilities(tracker),
                SHADOW_FIEND_ABILITY_SLOTS,
                8,
            ),
        ] {
            let mana = unit.map(|track| track.unit.mana);
            for slot in 0..count {
                let ability = abilities.and_then(|(abilities, _)| abilities.get(slot));
                let legal = ability_legal(action_space, body, slot);
                let mut token = encode_ability_token(body, slot, ability, legal, mana);
                if body == ControlledUnit::Hero {
                    token[ability_feature::SCOREBOARD_KIT_SOURCE] =
                        bool_feature(abilities.is_some_and(|(_, kit)| kit));
                }
                encode_ability_history(&mut token, tracker, unit, ability, tick);
                output.abilities[first + slot] = token;
            }
        }
    }

    fn encode_items(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        readiness: &ItemReadiness,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        let context = OwnedItemContext {
            tracker,
            action_space,
            readiness,
            tick: view.tick,
        };
        let hero_items = own_hero_items(tracker);
        let stash = tracker
            .own_player()
            .and_then(|player| player.stash.as_deref());
        let courier_items = own_courier_items(tracker);
        for (first, count, body, location, items) in [
            (
                0,
                9,
                ControlledUnit::Hero,
                0,
                hero_items.map(|(items, _)| items),
            ),
            (9, 6, ControlledUnit::Hero, 1, stash),
            (15, 6, ControlledUnit::Courier, 2, courier_items),
        ] {
            for slot in 0..count {
                let item = items.and_then(|items| items.get(slot)).copied().flatten();
                // Stash rows number their slots after the bag's nine.
                let slot_token = if location == 1 { first + slot } else { slot };
                output.items[first + slot] = context.encode(body, location, slot_token, slot, item);
            }
        }
        let kit = bool_feature(hero_items.is_some_and(|(_, kit)| kit));
        for token in &mut output.items[..9] {
            token[item_feature::SCOREBOARD_KIT_SOURCE] = kit;
        }
        encode_shop_items(tracker, action_space, &mut output.items);
        encode_enemy_items(tracker, &mut output.items);
    }

    fn encode_projectiles(
        &self,
        tracker: &StateTracker,
        view: &Perspective<'_>,
        output: &mut FeatureFrame,
    ) {
        let current = tracker.current().expect("snapshot was checked");
        assert!(current.projectiles.len() <= MAX_PROJECTILES);
        let mut count = 0usize;
        for projectile in &current.projectiles {
            let history = self.projectile_observation(projectile.id);
            let token = self.projectile_token(projectile, view, history);
            insert_sorted_token(&mut output.projectiles, &mut count, token);
        }
        assert!(count <= PROJECTILE_FEATURE_TOKENS);
    }

    fn projectile_token(
        &self,
        projectile: &ProjectileView,
        view: &Perspective<'_>,
        history: Option<ProjectileObservation>,
    ) -> [f32; PROJECTILE_FEATURES] {
        use projectile_feature as index;
        let team = view.team;
        let mut token = [0.0; PROJECTILE_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        let relation = team_relation(team, projectile.team);
        one_hot(
            &mut token,
            index::RELATION_START,
            4,
            relation_index(relation),
        );
        if let Some(ability) = projectile.ability {
            token[index::ABILITY_PRESENT] = 1.0;
            let class = ability_id_class(ability);
            one_hot(&mut token, index::ABILITY_START, ABILITY_ID_CLASSES, class);
        }
        if let Some(origin) = view.origin {
            let delta = self.canonical_delta(team, projectile.pos, origin);
            token[index::RELATIVE_X] = raw_units(delta.0, scale::PROJECTILE_RELATIVE);
            token[index::RELATIVE_Y] = raw_units(delta.1, scale::PROJECTILE_RELATIVE);
        }
        let (cosine, sine) = canonical_facing(team, projectile.facing);
        token[index::FACING_COS] = cosine;
        token[index::FACING_SIN] = sine;
        let Some(history) = history else {
            return token;
        };
        let age = (history.last_tick - history.first_tick) as f32;
        token[index::AGE] = fraction(age, scale::PROJECTILE_AGE_TICKS);
        let Some(previous_tick) = history.previous_tick else {
            return token;
        };
        let elapsed = i64::from(history.last_tick - previous_tick);
        let sign = if team == Team::Dire { -1 } else { 1 };
        let delta_x =
            sign * (i64::from(history.position.x.raw) - i64::from(history.previous_position.x.raw));
        let delta_y =
            sign * (i64::from(history.position.y.raw) - i64::from(history.previous_position.y.raw));
        let per_second = TICKS_PER_SECOND / elapsed as f32;
        token[index::VELOCITY_PRESENT] = 1.0;
        token[index::VELOCITY_X] = raw_units(delta_x, scale::PROJECTILE_VELOCITY / per_second);
        token[index::VELOCITY_Y] = raw_units(delta_y, scale::PROJECTILE_VELOCITY / per_second);
        if let Some(origin) = view.origin {
            let relative = self.canonical_delta(team, projectile.pos, origin);
            token[index::CLOSEST_APPROACH_PRESENT] = 1.0;
            token[index::CLOSEST_APPROACH] = closest_approach(relative, delta_x, delta_y);
        }
        token
    }

    fn encode_loot(
        &mut self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        output: &mut FeatureFrame,
    ) {
        let current = tracker.current().expect("snapshot was checked");
        let origin = own_origin(tracker);
        if let Some(origin) = origin
            && !current.loot.is_empty()
        {
            // Impassable targets can never be settled, so excluding them from
            // the pending set keeps the bounded fill from exhausting the map.
            let targets: Vec<usize> = action_space
                .loot_candidates()
                .iter()
                .filter_map(|candidate| self.cell(candidate.position))
                .filter(|target| !self.path_impassable[*target])
                .collect();
            self.prepare_path_distances(origin, &targets);
        }
        for (candidate_index, candidate) in action_space.loot_candidates().iter().enumerate() {
            let loot = current
                .loot
                .iter()
                .find(|loot| loot.id == candidate.id())
                .expect("action loot candidate belongs to current snapshot");
            let duplicate_semantics = action_space.loot_candidates().iter().any(|other| {
                other.id() != candidate.id()
                    && other.item == candidate.item
                    && other.charges == candidate.charges
                    && other.position == candidate.position
            });
            let history = (!duplicate_semantics)
                .then(|| self.loot_observation(loot.id))
                .flatten();
            output.loot[candidate_index] = self.loot_token(tracker.team(), loot, origin, history);
        }
    }

    fn loot_token(
        &self,
        team: Team,
        loot: &bota_proto::LootView,
        origin: Option<Vec2>,
        history: Option<LootObservation>,
    ) -> [f32; LOOT_FEATURES] {
        use loot_feature as index;
        let mut token = [0.0; LOOT_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        one_hot(
            &mut token,
            index::ITEM_START,
            ITEM_ID_CLASSES,
            item_id_class(loot.item),
        );
        token[index::CHARGES_PRESENT] = bool_feature(loot.charges.is_some());
        token[index::CHARGES] = fraction(f32::from(loot.charges.unwrap_or(0)), scale::CHARGES);
        if let Some(history) = history {
            token[index::VISIBLE_AGE_PRESENT] = 1.0;
            let age = (history.last_tick - history.first_tick) as f32;
            token[index::VISIBLE_AGE] = fraction(age, scale::LOOT_AGE_TICKS);
        }
        let Some(origin) = origin else {
            return token;
        };
        let delta = self.canonical_delta(team, loot.pos, origin);
        token[index::RELATIVE_X] = raw_units(delta.0, scale::NEAR_DISTANCE);
        token[index::RELATIVE_Y] = raw_units(delta.1, scale::NEAR_DISTANCE);
        token[index::DISTANCE_NEAR] = near_distance(combat::distance(origin, loot.pos));
        if let Some(steps) = self.path_steps(loot.pos) {
            token[index::PATH_DISTANCE_PRESENT] = 1.0;
            token[index::PATH_DISTANCE] =
                ratio(i64::from(steps), 0, self.path_distances.len() as i64);
        }
        token
    }

    fn projectile_observation(&self, id: bota_proto::EntityId) -> Option<ProjectileObservation> {
        self.observation
            .projectiles
            .iter()
            .find(|history| history.id == id)
            .copied()
    }

    fn loot_observation(&self, id: bota_proto::EntityId) -> Option<LootObservation> {
        self.observation
            .loot
            .iter()
            .find(|history| history.id == id)
            .copied()
    }

    fn encode_map(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        use map_feature as index;
        let Some(origin) = own_origin(tracker) else {
            return;
        };
        if let Some(cell) = self.cell(origin) {
            let terrain = self.terrain[cell];
            output.map[index::WALKABLE] = bool_feature(terrain & 0x80 != 0);
            output.map[index::WATER] = bool_feature(terrain & 0x40 != 0);
            output.map[index::ELEVATION] = ratio(i64::from(terrain & 0x3f), 0, 63);
        }
        encode_landmark_distances(tracker, origin, &mut output.map);
        for (direction_index, direction) in MAP_DIRECTIONS.into_iter().enumerate() {
            let start = index::RAYS_START + direction_index * 10;
            self.encode_map_ray(
                tracker,
                origin,
                direction,
                &mut output.map[start..start + 10],
            );
        }
    }

    fn encode_map_ray(
        &self,
        tracker: &StateTracker,
        origin: Vec2,
        canonical_direction: (i32, i32),
        output: &mut [f32],
    ) {
        let world_direction = if tracker.team() == Team::Dire {
            (-canonical_direction.0, -canonical_direction.1)
        } else {
            canonical_direction
        };
        let mut hits = [None; 4];
        let mut endpoint = None;
        for step in 1..=MAP_RAY_CELLS {
            let position = ray_position(origin, world_direction, step);
            let Some(cell) = self.cell(position) else {
                hits[0].get_or_insert(step);
                break;
            };
            let terrain = self.terrain[cell];
            set_first_hit(&mut hits[0], terrain & 0x80 == 0, step);
            set_first_hit(&mut hits[1], terrain & 0x40 != 0, step);
            set_first_hit(&mut hits[2], self.opaque[cell], step);
            set_first_hit(
                &mut hits[3],
                self.tree_at_visible_context(tracker, position),
                step,
            );
            endpoint = Some(terrain);
        }
        for pair in 0..4 {
            if let Some(step) = hits[pair] {
                output[pair * 2] = 1.0;
                output[pair * 2 + 1] = ratio(step as i64, 1, MAP_RAY_CELLS as i64);
            }
        }
        if let Some(terrain) = endpoint {
            output[8] = ratio(i64::from(terrain & 0x3f), 0, 63);
            output[9] = bool_feature(terrain & 0x80 != 0);
        }
    }

    fn tree_at_visible_context(&self, tracker: &StateTracker, position: Vec2) -> bool {
        let current = tracker.current().expect("snapshot was checked");
        let locally_observable = tracker.position_locally_observable_to_own_seat(position);
        let Some(cell) = self.cell(position) else {
            return false;
        };
        let start = self
            .static_tree_index
            .partition_point(|(tree_cell, _)| *tree_cell < cell);
        let end = self
            .static_tree_index
            .partition_point(|(tree_cell, _)| *tree_cell <= cell);
        if start < end && !locally_observable {
            return true;
        }
        if self.static_tree_index[start..end]
            .iter()
            .any(|(_, index)| !current.felled_trees.contains(index))
        {
            return true;
        }
        locally_observable
            && current
                .planted_trees
                .iter()
                .copied()
                .any(|tree| same_cell(self, tree, position))
    }

    fn prepare_path_distances(&mut self, origin: Vec2, targets: &[usize]) {
        let Some(origin) = self.cell(origin) else {
            self.path_origin = None;
            return;
        };
        if self.path_origin == Some(origin) && self.cached_paths_cover(targets) {
            return;
        }
        assert!(self.axis <= 1 << 16);
        let axis = self.axis;
        // Work on local buffers for the whole fill: the compiler keeps the
        // grid pointers in registers instead of reloading `self` fields, and
        // the queue is the exact visited list used for the reset below.
        let mut distances = std::mem::take(&mut self.path_distances);
        let mut queue = std::mem::take(&mut self.path_queue);
        let impassable = std::mem::take(&mut self.path_impassable);
        for &packed in &queue {
            let x = packed & 0xffff;
            let y = packed >> 16;
            distances[y * axis + x] = u32::MAX;
        }
        queue.clear();
        self.path_origin = Some(origin);
        self.path_targets.clear();
        self.path_targets
            .extend(targets.iter().map(|target| *target as u32));
        self.path_targets.sort_unstable();
        self.path_targets.dedup();
        self.path_exhausted = false;
        distances[origin] = 0;
        queue.push(((origin / axis) << 16) | (origin % axis));
        let (cursor, pending) = flood_fill(
            axis,
            &impassable,
            &self.path_targets,
            &mut distances,
            &mut queue,
        );
        // An exhausted queue proves the whole reachable component is known;
        // with pending targets settled the field is complete for them too.
        self.path_exhausted =
            pending == 0 && !self.path_targets.is_empty() && cursor == queue.len();
        self.path_distances = distances;
        self.path_queue = queue;
        self.path_impassable = impassable;
    }

    fn cached_paths_cover(&self, targets: &[usize]) -> bool {
        self.path_exhausted
            || targets.iter().all(|target| {
                let packed = *target as u32;
                self.path_targets.binary_search(&packed).is_ok()
            })
    }

    fn path_steps(&self, position: Vec2) -> Option<u32> {
        let distance = self.path_distances[self.cell(position)?];
        (distance != u32::MAX).then_some(distance)
    }

    fn canonical_position(&self, team: Team, position: Vec2) -> Vec2 {
        if team == Team::Dire {
            let maximum = self.extent_raw - 1;
            Vec2 {
                x: Fixed {
                    raw: clamp_raw(maximum - i64::from(position.x.raw)),
                },
                y: Fixed {
                    raw: clamp_raw(maximum - i64::from(position.y.raw)),
                },
            }
        } else {
            position
        }
    }

    fn canonical_delta(&self, team: Team, position: Vec2, origin: Vec2) -> (i64, i64) {
        let mut x = i64::from(position.x.raw) - i64::from(origin.x.raw);
        let mut y = i64::from(position.y.raw) - i64::from(origin.y.raw);
        if team == Team::Dire {
            x = -x;
            y = -y;
        }
        (x, y)
    }

    fn cell(&self, position: Vec2) -> Option<usize> {
        let (x, y) = self.cell_xy(position)?;
        Some(y * self.axis + x)
    }

    fn cell_xy(&self, position: Vec2) -> Option<(usize, usize)> {
        if position.x.raw < 0 || position.y.raw < 0 {
            return None;
        }
        let x = usize::try_from(position.x.to_int() / TERRAIN_CELL_SIZE).ok()?;
        let y = usize::try_from(position.y.to_int() / TERRAIN_CELL_SIZE).ok()?;
        (x < self.axis && y < self.axis).then_some((x, y))
    }
}

/// Breadth-first 8-neighbour fill from the queued origin until every sorted
/// `target` cell is settled or the reachable component is exhausted. Queue
/// entries pack a cell as `y << 16 | x`. Returns the queue cursor and the
/// number of targets still unsettled.
fn flood_fill(
    axis: usize,
    impassable: &[bool],
    targets: &[u32],
    distances: &mut [u32],
    queue: &mut Vec<usize>,
) -> (usize, usize) {
    // Neighbour order matches [`MAP_DIRECTIONS`] exactly; the deltas are row
    // strides computed once, so the inner loop needs one add per neighbour
    // instead of a multiply.
    const DIRECTIONS: [(isize, isize); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    let row = axis as isize;
    let deltas: [isize; 8] = [1, row + 1, row, row - 1, -1, -row - 1, -row, -row + 1];
    let mut pending = targets.len();
    let mut cursor = 0usize;
    while cursor < queue.len() && pending > 0 {
        let packed = queue[cursor];
        cursor += 1;
        let x = (packed & 0xffff) as isize;
        let y = (packed >> 16) as isize;
        let cell = y as usize * axis + x as usize;
        if targets.binary_search(&(cell as u32)).is_ok() {
            pending -= 1;
        }
        let next_distance = distances[cell].saturating_add(1);
        for (index, (delta_x, delta_y)) in DIRECTIONS.iter().enumerate() {
            let next_x = x + delta_x;
            let next_y = y + delta_y;
            if next_x < 0 || next_y < 0 {
                continue;
            }
            let (next_x, next_y) = (next_x as usize, next_y as usize);
            if next_x >= axis || next_y >= axis {
                continue;
            }
            let next = (cell as isize + deltas[index]) as usize;
            if impassable[next] || distances[next] != u32::MAX {
                continue;
            }
            distances[next] = next_distance;
            queue.push((next_y << 16) | next_x);
        }
    }
    (cursor, pending)
}

fn validate_map2_reward(tracker: &StateTracker, tick: u32) -> Result<(), FeatureError> {
    if let Some(state) = tracker.map2_reward_state()
        && state.completed_tick != Some(tick)
    {
        return Err(FeatureError::Map2RewardIncomplete { snapshot: tick });
    }
    Ok(())
}

/// The own hero's point of view shared by every token of one frame.
struct Perspective<'a> {
    team: Team,
    tick: u32,
    /// Own hero position, else own courier position.
    origin: Option<Vec2>,
    /// The live own hero.
    hero: Option<&'a UnitView>,
    kit: Option<combat::HeroKit>,
}

impl<'a> Perspective<'a> {
    fn new(tracker: &'a StateTracker, tick: u32) -> Self {
        let hero = tracker.own_hero().filter(|hero| hero.hp > 0);
        Self {
            team: tracker.team(),
            tick,
            origin: own_origin(tracker),
            hero,
            kit: hero.and_then(|hero| combat::HeroKit::of(hero, 0)),
        }
    }

    /// Whether `track` dealt its latest damage to the own hero, or to another own-side unit.
    fn recent_victim(&self, tracker: &StateTracker, track: &crate::EntityTrack) -> [bool; 2] {
        let Some(damage) = track.last_damage_dealt else {
            return [false; 2];
        };
        if self.tick.saturating_sub(damage.tick) > scale::RECENT_ATTACK_TICKS {
            return [false; 2];
        }
        let Some(victim) = damage.counterpart else {
            return [false; 2];
        };
        if self.hero.is_some_and(|hero| hero.id == victim) {
            return [true, false];
        }
        let own_side = tracker
            .entity(victim)
            .is_some_and(|victim| victim.unit.team == self.team);
        [false, own_side]
    }
}

fn encode_map2_reward(tracker: &StateTracker, global: &mut [f32; GLOBAL_FEATURES]) {
    use global_feature as index;
    let Some(state) = tracker.map2_reward_state() else {
        return;
    };
    assert_eq!(tracker.metadata().map, crate::MAP2_ID);
    assert_eq!(
        state.completed_tick,
        tracker.current().map(|view| view.tick)
    );
    global[index::MAP2_OWN_TOWER_HEALTH] = state.tower_health[0];
    global[index::MAP2_ENEMY_TOWER_HEALTH] = state.tower_health[1];
    global[index::MAP2_OWN_DEATHS] = state.deaths[0];
    global[index::MAP2_ENEMY_DEATHS] = state.deaths[1];
    global[index::MAP2_OWN_HERO_HEALTH] = state.hero_health[0];
    global[index::MAP2_ENEMY_HERO_HEALTH] = state.hero_health[1];
    global[index::MAP2_XP_LEAD] = state.xp_lead;
    global[index::MAP2_REWARD_POTENTIAL] = state.potential;
}

/// Creep, tower and duel summaries from the own hero's point of view.
fn encode_global_combat(
    tracker: &StateTracker,
    view: &Perspective<'_>,
    global: &mut [f32; GLOBAL_FEATURES],
) {
    use global_feature as index;
    let Some(hero) = view.hero else {
        return;
    };
    let current = tracker.current().expect("snapshot was checked");
    let enemy = opposing(view.team);
    let mut counts = [0.0f32; 5];
    for track in tracker.entities().iter().filter(|track| track.visible) {
        let unit = &track.unit;
        if unit.hp <= 0 {
            continue;
        }
        if unit.team == enemy && is_lane_creep(unit.kind) {
            counts[0] += f32::from(u8::from(own_in_acquisition(hero, unit)));
            counts[1] += f32::from(u8::from(view.recent_victim(tracker, track)[0]));
        }
        if is_lane_creep(unit.kind)
            && combat::distance(hero.pos, unit.pos) <= scale::CREEP_BALANCE_RADIUS
        {
            counts[2] += if unit.team == view.team {
                1.0
            } else if unit.team == enemy {
                -1.0
            } else {
                0.0
            };
        }
        if unit.kind == UnitKind::Tower && unit.team == enemy {
            global[index::IN_ENEMY_TOWER_RANGE] =
                global[index::IN_ENEMY_TOWER_RANGE].max(bool_feature(in_tower_range(unit, hero)));
            let targeting = view.recent_victim(tracker, track)[0];
            global[index::ENEMY_TOWER_TARGETING] =
                global[index::ENEMY_TOWER_TARGETING].max(bool_feature(targeting));
        }
        if is_lane_creep(unit.kind) {
            let (tower_team, slot) = if unit.team == view.team {
                (enemy, 3)
            } else {
                (view.team, 4)
            };
            let covered = current.units.iter().any(|tower| {
                tower.kind == UnitKind::Tower
                    && tower.team == tower_team
                    && tower.hp > 0
                    && in_tower_range(tower, unit)
            });
            counts[slot] += f32::from(u8::from(covered));
        }
    }
    global[index::ENEMY_CREEPS_ACQUIRING] = fraction(counts[0], scale::CREEPS);
    global[index::ENEMY_CREEPS_ATTACKING] = fraction(counts[1], scale::CREEPS);
    global[index::CREEP_BALANCE] = signed_fraction(counts[2], scale::CREEPS);
    global[index::ALLIED_CREEPS_UNDER_ENEMY_TOWER] = fraction(counts[3], scale::CREEPS);
    global[index::ENEMY_CREEPS_UNDER_OWN_TOWER] = fraction(counts[4], scale::CREEPS);
    encode_duel(current, view, hero, global);
}

/// Seconds each hero needs to kill the other and the own hero's ready razes.
fn encode_duel(
    current: &bota_proto::WorldView,
    view: &Perspective<'_>,
    hero: &UnitView,
    global: &mut [f32; GLOBAL_FEATURES],
) {
    use global_feature as index;
    if let Some(kit) = view.kit {
        let ready = kit.razes.iter().filter(|left| **left == Some(0)).count();
        global[index::RAZES_READY] = ready as f32 / 3.0;
        if kit.raze_level > 0 && kit.raze_mana > 0 {
            global[index::RAZES_AFFORDABLE] = fraction((kit.mana / kit.raze_mana) as f32, 3.0);
        }
    }
    let enemy_team = opposing(view.team);
    let Some(enemy) = current
        .units
        .iter()
        .filter(|unit| unit.kind == UnitKind::Hero && unit.team == enemy_team && unit.hp > 0)
        .min_by_key(|unit| hero.pos.distance_squared(unit.pos))
    else {
        return;
    };
    global[index::ENEMY_HERO_VISIBLE] = 1.0;
    if let Some(kit) = view.kit {
        let seconds = kit.seconds_to_kill(hero, enemy, raze_stacks(enemy));
        global[index::SECONDS_TO_KILL_ENEMY] = fraction(seconds, combat::MAX_TTK_SECONDS);
    }
    if let Some(kit) = combat::HeroKit::of(enemy, 0) {
        let seconds = kit.seconds_to_kill(enemy, hero, raze_stacks(hero));
        global[index::SECONDS_TO_BE_KILLED] = fraction(seconds, combat::MAX_TTK_SECONDS);
    }
}

fn raze_stacks(unit: &UnitView) -> u32 {
    crate::tracker::shadowraze_effect(unit, 0).map_or(0, |(stacks, _)| stacks)
}

const fn is_lane_creep(kind: UnitKind) -> bool {
    matches!(
        kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
    )
}

/// Whether `unit` stands inside `tower`'s attack reach, bound to bound.
fn in_tower_range(tower: &UnitView, unit: &UnitView) -> bool {
    let reach = tower.attack_range + tower.bound + unit.bound;
    tower.pos.distance_squared(unit.pos) <= reach.squared_raw()
}

/// Whether the own hero stands where a hostile `unit` acquires targets on its own.
fn own_in_acquisition(hero: &UnitView, unit: &UnitView) -> bool {
    combat::acquisition(unit).is_some_and(|range| {
        combat::distance(hero.pos, unit.pos) <= range + hero.bound.to_f32() + unit.bound.to_f32()
    })
}

fn encode_landmark_distances(
    tracker: &StateTracker,
    origin: Vec2,
    output: &mut [f32; MAP_FEATURES],
) {
    use map_feature as index;
    let current = tracker.current().expect("snapshot was checked");
    for (slot, kind, own) in [
        (index::OWN_FOUNTAIN_DISTANCE_LOG, UnitKind::Fountain, true),
        (
            index::ENEMY_FOUNTAIN_DISTANCE_LOG,
            UnitKind::Fountain,
            false,
        ),
        (index::OWN_TOWER_DISTANCE_LOG, UnitKind::Tower, true),
        (index::ENEMY_TOWER_DISTANCE_LOG, UnitKind::Tower, false),
    ] {
        let team = if own {
            tracker.team()
        } else {
            opposing(tracker.team())
        };
        let nearest = current
            .units
            .iter()
            .filter(|unit| unit.kind == kind && unit.team == team)
            .map(|unit| origin.distance_squared(unit.pos))
            .min();
        if let Some(distance_squared) = nearest {
            let distance = (distance_squared as f64).sqrt() as f32 / 65_536.0;
            output[slot] = log_distance(distance);
        }
    }
}

fn encode_unit_effects(token: &mut [f32; UNIT_FEATURES], track: &crate::EntityTrack, tick: u32) {
    use unit_feature as index;
    if track.unit.hp <= 0
        || track
            .last_death
            .is_some_and(|death| death.tick >= track.last_seen_tick)
    {
        return;
    }
    let age = tick.saturating_sub(track.last_seen_tick);
    if let Some((stacks, ticks)) = crate::tracker::shadowraze_effect(&track.unit, age) {
        token[index::RAZE_EFFECT_PRESENT] = 1.0;
        token[index::RAZE_STACKS] = unit_ratio(stacks, scale::RAZE_STACKS);
        token[index::RAZE_TICKS_LEFT] = unit_ratio(ticks, crate::SHADOWRAZE_EFFECT_TICKS);
    }
    let [guarded, inspired] = crate::tracker::aura_effects(&track.unit, age);
    token[index::GUARDED_TICKS_LEFT] = unit_ratio(guarded, crate::AURA_EFFECT_TICKS);
    token[index::INSPIRED_TICKS_LEFT] = unit_ratio(inspired, crate::AURA_EFFECT_TICKS);
}

fn encode_restoration_reports(
    token: &mut [f32; UNIT_FEATURES],
    tracker: &StateTracker,
    target: EntityId,
    tick: u32,
) {
    let reports = crate::tracker::recent_restoration_reports(tracker, target);
    for (report, offset, maximum) in [
        (
            reports[0],
            unit_feature::HEALTH_RESTORE_REPORT_PRESENT,
            scale::RESTORED_HEALTH,
        ),
        (
            reports[1],
            unit_feature::MANA_RESTORE_REPORT_PRESENT,
            scale::RESTORED_MANA,
        ),
    ] {
        if let Some((reported_tick, amount)) = report {
            assert!(reported_tick < tick);
            assert!(tick - reported_tick <= crate::HISTORY_TICKS);
            token[offset] = 1.0;
            token[offset + 1] = fraction(amount as f32, maximum);
            token[offset + 2] = unit_ratio(tick - reported_tick, crate::HISTORY_TICKS);
        }
    }
}

/// Razes, Requiem and mana of any hero token; raze threat of a hostile one.
fn encode_hero_kit(
    token: &mut [f32; UNIT_FEATURES],
    unit: &UnitView,
    age: u32,
    relation: EntityRelation,
    view: &Perspective<'_>,
) {
    use unit_feature as index;
    let Some(kit) = combat::HeroKit::of(unit, age) else {
        return;
    };
    token[index::KIT_PRESENT] = 1.0;
    for (reach, left) in kit.razes.iter().enumerate() {
        if let Some(left) = left {
            token[index::RAZE_COOLDOWN_START + reach] =
                fraction(*left as f32, scale::RAZE_COOLDOWN_TICKS);
        }
    }
    token[index::RAZE_LEVEL] = f32::from(kit.raze_level) / 4.0;
    if let Some(left) = kit.requiem {
        token[index::REQUIEM_COOLDOWN] = fraction(left as f32, scale::REQUIEM_COOLDOWN_TICKS);
    }
    token[index::REQUIEM_LEVEL] = f32::from(kit.requiem_level) / 3.0;
    token[index::CAN_AFFORD_RAZE] = bool_feature(kit.can_afford_raze());
    let hostile = matches!(relation, EntityRelation::Enemy | EntityRelation::Neutral);
    let Some(hero) = view
        .hero
        .filter(|hero| hostile && age == 0 && combat::razeable(hero))
    else {
        return;
    };
    let offset = canonical_offset(view.team, unit.pos, hero.pos);
    let facing = canonical_facing(view.team, unit.facing);
    for (reach, (_, distance)) in crate::raze_aim::SHADOWRAZES.iter().enumerate() {
        let strikes = kit.razes[reach].is_some() && combat::raze_strikes(offset, facing, *distance);
        token[index::RAZE_THREAT_START + reach] = bool_feature(strikes);
    }
}

fn encode_local_global(
    encoder: &FeatureEncoder,
    tracker: &StateTracker,
    tick: u32,
    local: &LocalPolicyState,
    global: &mut [f32; GLOBAL_FEATURES],
) {
    use global_feature as index;
    if let Some(active) = local.active_order() {
        global[index::ACTIVE_ORDER_PRESENT] = 1.0;
        one_hot(
            global,
            index::ACTIVE_ORDER_KIND_START,
            ActionKind::COUNT,
            active.kind.index(),
        );
        let age = tick.saturating_sub(active.started_tick) as f32;
        global[index::ACTIVE_ORDER_AGE] = fraction(age, scale::ORDER_AGE_TICKS);
        encode_active_target(encoder, tracker, active.target, global);
    }
    if local.decisions().next_back().is_some() {
        global[index::LAST_DECISION_PRESENT] = 1.0;
    }
}

fn encode_active_target(
    encoder: &FeatureEncoder,
    tracker: &StateTracker,
    target: ActivePolicyTarget,
    global: &mut [f32; GLOBAL_FEATURES],
) {
    use global_feature as index;
    let (position, visible, unit) = match target {
        ActivePolicyTarget::None => return,
        ActivePolicyTarget::Point(position) => (position, true, None),
        ActivePolicyTarget::Unit(id) => {
            let Some(entity) = tracker.entity(id) else {
                global[index::ACTIVE_TARGET_PRESENT] = 1.0;
                global[index::ACTIVE_TARGET_UNIT] = 1.0;
                return;
            };
            (entity.unit.pos, entity.visible, Some(&entity.unit))
        }
    };
    global[index::ACTIVE_TARGET_PRESENT] = 1.0;
    global[index::ACTIVE_TARGET_POINT] = bool_feature(unit.is_none());
    global[index::ACTIVE_TARGET_UNIT] = bool_feature(unit.is_some());
    global[index::ACTIVE_TARGET_VISIBLE] = bool_feature(visible);
    if let Some(hero) = tracker.own_hero() {
        let delta = encoder.canonical_delta(tracker.team(), position, hero.pos);
        global[index::ACTIVE_TARGET_RELATIVE_X] = signed_raw_ratio(delta.0, encoder.extent_raw);
        global[index::ACTIVE_TARGET_RELATIVE_Y] = signed_raw_ratio(delta.1, encoder.extent_raw);
        let distance = combat::distance(hero.pos, position);
        global[index::ACTIVE_TARGET_DISTANCE_NEAR] = near_distance(distance);
        global[index::ACTIVE_TARGET_DISTANCE_LOG] = log_distance(distance);
    }
    if let Some(unit) = unit {
        one_hot(
            global,
            index::ACTIVE_TARGET_KIND_START,
            12,
            unit_kind_index(unit.kind),
        );
        if unit.team == Team::Neutral {
            global[index::ACTIVE_TARGET_NEUTRAL] = 1.0;
        } else if unit.team == tracker.team() {
            global[index::ACTIVE_TARGET_ALLIED] = 1.0;
        } else {
            global[index::ACTIVE_TARGET_ENEMY] = 1.0;
        }
    }
}

fn encode_own_score(own: &PlayerView, global: &mut [f32; GLOBAL_FEATURES]) {
    use global_feature as index;
    global[index::OWN_LEVEL] = fraction(f32::from(own.level), scale::LEVEL);
    global[index::OWN_XP] = fraction(own.xp as f32, scale::XP);
    global[index::OWN_LAST_HITS] = fraction(f32::from(own.last_hits), scale::LAST_HITS);
    global[index::OWN_DENIES] = fraction(f32::from(own.denies), scale::DENIES);
}

fn encode_history_sample(
    output: &mut [f32; HISTORY_FEATURES],
    summary: crate::GlobalSummary,
    current_tick: u32,
) {
    use history_feature as index;
    output[index::SAMPLE_PRESENT] = 1.0;
    output[index::AGE] = unit_ratio(
        current_tick.saturating_sub(summary.tick),
        crate::HISTORY_TICKS,
    );
    output[index::HP_PRESENT] = bool_feature(summary.own_hp_present);
    output[index::HP_RATIO] = safe_fraction(summary.own_hp, summary.own_max_hp);
    output[index::MANA_PRESENT] = bool_feature(summary.own_mana_present);
    output[index::MANA_RATIO] = safe_fraction(summary.own_mana, summary.own_max_mana);
    output[index::OWN_LEVEL] = fraction(f32::from(summary.own_level), scale::LEVEL);
    output[index::OWN_GOLD] = fraction(summary.own_gold as f32, scale::GOLD);
    output[index::OWN_ALIVE] = bool_feature(summary.own_hp_present);
    output[index::RESPAWN_LEFT] = fraction(summary.own_respawn_left as f32, scale::RESPAWN_TICKS);
    output[index::VISIBLE_ALLIED_UNITS] =
        fraction(summary.visible_allied_units as f32, scale::VISIBLE_UNITS);
    output[index::VISIBLE_ENEMY_UNITS] =
        fraction(summary.visible_enemy_units as f32, scale::VISIBLE_UNITS);
}

fn encode_point_source(token: &mut [f32; POINT_FEATURES], source: PointSource) {
    use point_feature as index;
    let (category, direction, radius, kind, relation) = point_source_semantics(source);
    one_hot(token, index::SOURCE_START, 13, category);
    if let Some(direction) = direction {
        one_hot(
            token,
            index::SOURCE_DIRECTION_START,
            8,
            point_direction_index(direction),
        );
    }
    if let Some(radius) = radius {
        token[index::SOURCE_RADIUS_PRESENT] = 1.0;
        token[index::SOURCE_RADIUS] = ratio(i64::from(radius), 0, 1_200);
    }
    if let Some(kind) = kind {
        one_hot(token, index::SOURCE_KIND_START, 12, unit_kind_index(kind));
    }
    if let Some(relation) = relation {
        one_hot(
            token,
            index::SOURCE_RELATION_START,
            4,
            relation_index(relation),
        );
    }
}

fn encode_unit_resources(token: &mut [f32; UNIT_FEATURES], track: &crate::EntityTrack) {
    use unit_feature as index;
    let unit = &track.unit;
    if unit.max_hp > 0 {
        token[index::HP_RATIO] = safe_fraction(unit.hp, unit.max_hp);
        token[index::HP_NEAR] = fraction(combat::health(unit), scale::HEALTH_NEAR);
        token[index::HP_LOG] = log_fraction(combat::health(unit), scale::HEALTH_LOG_MAX);
        token[index::MAX_HP_LOG] = log_fraction(unit.max_hp as f32, scale::HEALTH_LOG_MAX);
        let [physical, magical] = combat::effective_health(unit);
        token[index::EFFECTIVE_HP_PHYSICAL] =
            log_fraction(physical, scale::EFFECTIVE_HEALTH_LOG_MAX);
        token[index::EFFECTIVE_HP_MAGICAL] = log_fraction(magical, scale::EFFECTIVE_HEALTH_LOG_MAX);
    }
    if unit.max_mana > 0 {
        token[index::MANA_PRESENT] = 1.0;
        token[index::MANA_RATIO] = safe_fraction(unit.mana, unit.max_mana);
        token[index::MANA_NEAR] = fraction(unit.mana.max(0) as f32, scale::MANA_NEAR);
        token[index::MAX_MANA_NEAR] = fraction(unit.max_mana as f32, scale::MANA_NEAR);
    }
    if track.velocity.is_some() {
        token[index::HP_DELTA] = signed_fraction(track.hp_delta as f32, scale::POOL_DELTA);
        token[index::MANA_DELTA] = signed_fraction(track.mana_delta as f32, scale::POOL_DELTA);
    }
    token[index::LEVEL] = fraction(f32::from(unit.level), scale::LEVEL);
}

fn encode_unit_combat(token: &mut [f32; UNIT_FEATURES], unit: &UnitView) {
    use unit_feature as index;
    token[index::ATTACK_DAMAGE] = fraction(unit.attack_damage as f32, scale::DAMAGE);
    token[index::ATTACK_RANGE] = fraction(unit.attack_range.to_f32(), scale::RANGE);
    token[index::ATTACK_TIME] = fraction(unit.attack_time as f32, scale::ATTACK_TIME_MS);
    token[index::ATTACK_POINT] = fraction(unit.attack_point as f32, scale::ATTACK_POINT_MS);
    token[index::ATTACK_SPEED] = fraction(unit.attack_speed as f32, scale::ATTACK_SPEED);
    token[index::MOVE_SPEED] = fraction(unit.move_speed.to_f32(), scale::MOVE_SPEED);
    token[index::ARMOR] = signed_fraction(unit.armor.to_f32(), scale::ARMOR);
    token[index::MAGIC_RESISTANCE] = unit.magic_resist.to_f32().clamp(-1.0, 1.0);
    token[index::VISION] = fraction(unit.vision_radius.to_f32(), scale::VISION);
}

/// Fight arithmetic between the live own hero and `unit` from the hero's side.
fn encode_unit_tactics(
    token: &mut [f32; UNIT_FEATURES],
    unit: &UnitView,
    hero: &UnitView,
    relation: EntityRelation,
    age: u32,
    team: Team,
) {
    use unit_feature as index;
    token[index::HERO_RELATIVE_PRESENT] = 1.0;
    let distance = combat::distance(hero.pos, unit.pos);
    let hulls = hero.bound + unit.bound;
    let distance_squared = hero.pos.distance_squared(unit.pos);
    token[index::OWN_IN_ATTACK_RANGE] =
        bool_feature(distance_squared <= (hero.attack_range + hulls).squared_raw());
    token[index::UNIT_IN_ATTACK_RANGE] =
        bool_feature(distance_squared <= (unit.attack_range + hulls).squared_raw());
    let edge = distance - hulls.to_f32();
    token[index::OWN_RANGE_MARGIN] =
        signed_fraction(edge - hero.attack_range.to_f32(), scale::MARGIN);
    token[index::UNIT_RANGE_MARGIN] =
        signed_fraction(edge - unit.attack_range.to_f32(), scale::MARGIN);
    let speed = hero.move_speed.to_f32();
    if speed > 0.0 {
        token[index::TIME_TO_REACH] = fraction(distance / speed, scale::REACH_SECONDS);
    }
    if relation == EntityRelation::Own {
        return;
    }
    token[index::OWN_HIT] = fraction(combat::hit_damage(hero, unit), scale::DAMAGE);
    token[index::OWN_HITS_TO_KILL] = fraction(combat::hits_to_kill(hero, unit), scale::HITS);
    let allied = relation == EntityRelation::Allied;
    token[index::KILLABLE_NOW] = bool_feature(age == 0 && combat::killable_now(hero, unit, allied));
    if allied {
        return;
    }
    token[index::ITS_HIT] = fraction(combat::hit_damage(unit, hero), scale::DAMAGE);
    token[index::ITS_HITS_TO_KILL_OWN] = fraction(combat::hits_to_kill(unit, hero), scale::HITS);
    token[index::OWN_IN_ACQUISITION] = bool_feature(own_in_acquisition(hero, unit));
    encode_own_raze(token, unit, hero, age, team);
}

/// What the own hero's next raze does to a hostile `unit`, and which reaches strike it now.
fn encode_own_raze(
    token: &mut [f32; UNIT_FEATURES],
    unit: &UnitView,
    hero: &UnitView,
    age: u32,
    team: Team,
) {
    use unit_feature as index;
    if !combat::razeable(unit) {
        return;
    }
    if let Some(kit) = combat::HeroKit::of(hero, 0) {
        let stacks = crate::tracker::shadowraze_effect(unit, age).map_or(0, |(stacks, _)| stacks);
        token[index::OWN_RAZE_DAMAGE] = fraction(kit.raze_damage(unit, stacks), scale::RAZE_DAMAGE);
        if kit.raze_level > 0 {
            let razes = kit.razes_to_kill(unit, stacks, scale::RAZES_TO_KILL);
            token[index::OWN_RAZES_TO_KILL] = unit_ratio(razes, scale::RAZES_TO_KILL);
        }
    }
    if age > 0 {
        return;
    }
    let offset = canonical_offset(team, hero.pos, unit.pos);
    let facing = canonical_facing(team, hero.facing);
    for (reach, (_, distance)) in crate::raze_aim::SHADOWRAZES.iter().enumerate() {
        let strikes = combat::raze_strikes(offset, facing, *distance);
        token[index::IN_OWN_RAZE_START + reach] = bool_feature(strikes);
    }
}

fn encode_unit_recent(
    token: &mut [f32; UNIT_FEATURES],
    tracker: &StateTracker,
    track: &crate::EntityTrack,
    view: &Perspective<'_>,
) {
    use unit_feature as index;
    let current_tick = view.tick;
    let [own_hero, own_side] = view.recent_victim(tracker, track);
    token[index::ATTACKING_OWN_HERO] = bool_feature(own_hero);
    token[index::ATTACKING_OWN_SIDE] = bool_feature(own_side);
    if recent_damage(tracker, track.id, current_tick, false).is_some() {
        token[index::RECENT_DAMAGE_TAKEN] = 1.0;
    }
    if let Some((_, amount)) = recent_damage(tracker, track.id, current_tick, true) {
        token[index::RECENT_DAMAGE_DEALT_PRESENT] = 1.0;
        token[index::RECENT_DAMAGE_DEALT] = fraction(amount as f32, scale::DAMAGE);
    }
    let Some(attack_tick) = recent_possible_attack(tracker, track.id, current_tick) else {
        return;
    };
    let age = current_tick - attack_tick;
    let interval = attack_interval_ticks(track.unit.attack_time).max(1);
    if age > interval {
        return;
    }
    token[index::ATTACK_PHASE_PRESENT] = 1.0;
    token[index::ATTACK_PHASE] = unit_ratio(age, interval);
}

fn encode_ability_history(
    token: &mut [f32; ABILITY_FEATURES],
    tracker: &StateTracker,
    track: Option<&crate::EntityTrack>,
    ability: Option<&AbilityView>,
    current_tick: u32,
) {
    let (Some(track), Some(ability)) = (track, ability) else {
        return;
    };
    let Some(cast_tick) = recent_ability_cast(tracker, track.id, ability.id, current_tick) else {
        return;
    };
    token[ability_feature::LAST_CAST_PRESENT] = 1.0;
    token[ability_feature::LAST_CAST_AGE] =
        fraction((current_tick - cast_tick) as f32, scale::LAST_CAST_TICKS);
}

fn encode_statuses(token: &mut [f32; UNIT_FEATURES], statuses: StatusFlags) {
    for (index, flag) in [
        (unit_feature::SLOWED, StatusFlags::SLOWED),
        (unit_feature::INVULNERABLE, StatusFlags::INVULNERABLE),
        (unit_feature::CHANNELLING, StatusFlags::CHANNELLING),
    ] {
        token[index] = bool_feature(statuses.bits & flag != 0);
    }
}

fn encode_ability_token(
    unit: ControlledUnit,
    slot: usize,
    ability: Option<&AbilityView>,
    legal: bool,
    mana: Option<i32>,
) -> [f32; ABILITY_FEATURES] {
    use ability_feature as index;
    let mut token = [0.0; ABILITY_FEATURES];
    token[index::TOKEN_PRESENT] = 1.0;
    one_hot(&mut token, index::BODY_START, 2, unit.index());
    one_hot(&mut token, index::SLOT_START, 8, slot);
    let Some(ability) = ability else {
        return token;
    };
    token[index::OBSERVED] = 1.0;
    one_hot(
        &mut token,
        index::ID_START,
        ABILITY_ID_CLASSES,
        ability_id_class(ability.id),
    );
    token[index::LEVEL] = f32::from(ability.level.min(4)) / 4.0;
    token[index::MAX_LEVEL] = f32::from(ability.max_level.min(4)) / 4.0;
    encode_cooldown(&mut token, index::COOLDOWN_NEAR, ability.cooldown_left);
    token[index::MANA_COST] = fraction(ability.mana_cost as f32, scale::ABILITY_MANA);
    token[index::RANGE] = fraction(ability.range as f32, scale::RANGE);
    one_hot(&mut token, index::AIM_START, 5, aim_index(ability.aim));
    token[index::PASSIVE] = bool_feature(ability.passive);
    token[index::TOGGLE_ON] = bool_feature(ability.on);
    token[index::CAN_LEVEL] = bool_feature(ability.can_level);
    token[index::LEGAL] = bool_feature(legal);
    token[index::MANA_SUFFICIENT] =
        bool_feature(mana.is_some_and(|mana| mana >= ability.mana_cost));
    token
}

/// Near (saturating at ten seconds) and log cooldown in the two slots from `start`.
fn encode_cooldown(token: &mut [f32], start: usize, left: u32) {
    token[start] = fraction(left as f32, scale::COOLDOWN_NEAR_TICKS);
    token[start + 1] = log_fraction(left as f32, scale::COOLDOWN_LOG_TICKS);
}

/// Read-only context shared by the own inventory, stash and courier rows.
struct OwnedItemContext<'a> {
    tracker: &'a StateTracker,
    action_space: &'a ActionSpace,
    readiness: &'a ItemReadiness,
    tick: u32,
}

impl OwnedItemContext<'_> {
    fn encode(
        &self,
        unit: ControlledUnit,
        location: usize,
        slot_token: usize,
        slot: usize,
        item: Option<ItemView>,
    ) -> [f32; ITEM_FEATURES] {
        use item_feature as index;
        let mut token = [0.0; ITEM_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        one_hot(&mut token, index::LOCATION_START, 5, location);
        token[index::SLOT] = slot_value(slot_token);
        let Some(item) = item else {
            return token;
        };
        encode_item_view(&mut token, self.tracker.shop(), item);
        let wire_slot = ItemSlot(u8::try_from(slot).unwrap_or(u8::MAX));
        let local_mute = self
            .readiness
            .inventory_mute_left(unit, wire_slot, self.tick)
            .unwrap_or(0);
        let mute_left = item.mute_left.max(local_mute);
        token[index::MUTE_REMAINING_PRESENT] = 1.0;
        token[index::MUTE_REMAINING] = fraction(mute_left as f32, scale::MUTE_TICKS);
        token[index::MUTED] = bool_feature(mute_left > 0);
        // Stash rows never wait on a shared scroll cooldown.
        if location != 1
            && let Some(left) = self.readiness.shared_wait_left(unit, item.id, self.tick)
        {
            token[index::SHARED_WAIT_PRESENT] = 1.0;
            token[index::SHARED_WAIT_REMAINING] = fraction(left as f32, scale::SHARED_WAIT_TICKS);
        }
        if slot < 6 {
            token[index::LEGAL] = bool_feature(item_legal(self.action_space, unit, slot));
        }
        token
    }
}

/// One-based slot scalar shared by every item row.
fn slot_value(slot: usize) -> f32 {
    (slot + 1) as f32 / 64.0
}

fn encode_item_view(token: &mut [f32; ITEM_FEATURES], shop: &[ShopEntry], item: ItemView) {
    use item_feature as index;
    token[index::ITEM_PRESENT] = 1.0;
    one_hot(
        token,
        index::ITEM_START,
        ITEM_ID_CLASSES,
        item_id_class(item.id),
    );
    token[index::CHARGES_PRESENT] = bool_feature(item.charges.is_some());
    token[index::CHARGES] = fraction(f32::from(item.charges.unwrap_or(0)), scale::CHARGES);
    encode_cooldown(token, index::COOLDOWN_NEAR, item.cooldown_left);
    if let Some(aim) = item.aim {
        token[index::AIM_PRESENT] = 1.0;
        one_hot(token, index::AIM_START, 5, aim_index(aim));
    }
    token[index::RANGE] = fraction(item.range as f32, scale::RANGE);
    token[index::MANA_COST] = fraction(item.mana_cost as f32, scale::ABILITY_MANA);
    if let Some(mode) = item.mode {
        token[index::ATTRIBUTE_PRESENT] = 1.0;
        one_hot(token, index::ATTRIBUTE_START, 3, attribute_index(mode));
    }
    token[index::FOR_SALE] = bool_feature(item.for_sale);
    if let Some(entry) = shop.iter().find(|entry| entry.id == item.id) {
        token[index::VALUE_PRESENT] = 1.0;
        token[index::VALUE] = fraction(entry.cost as f32, scale::ITEM_VALUE);
        token[index::COMPOSITE] = bool_feature(!entry.components.is_empty());
    }
    token[index::RECIPE_COMPONENT] =
        bool_feature(shop.iter().any(|entry| entry.components.contains(&item.id)));
}

fn encode_shop_items(
    tracker: &StateTracker,
    action_space: &ActionSpace,
    output: &mut [[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS],
) {
    use item_feature as index;
    for (slot, candidate) in action_space.shop_candidates().iter().enumerate() {
        let mut token = [0.0; ITEM_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        one_hot(&mut token, index::LOCATION_START, 5, 3);
        token[index::SLOT] = slot_value(slot);
        token[index::ITEM_PRESENT] = 1.0;
        one_hot(
            &mut token,
            index::ITEM_START,
            ITEM_ID_CLASSES,
            item_id_class(candidate.item),
        );
        token[index::VALUE_PRESENT] = 1.0;
        token[index::VALUE] = fraction(candidate.cost as f32, scale::ITEM_VALUE);
        token[index::SHOP_CANDIDATE] = 1.0;
        token[index::LEGAL] = bool_feature(action_space.buy_mask(ControlledUnit::Hero)[slot]);
        if let Some(entry) = tracker
            .shop()
            .iter()
            .find(|entry| entry.id == candidate.item)
        {
            token[index::COMPOSITE] = bool_feature(!entry.components.is_empty());
        }
        output[OWN_ITEM_SLOTS + slot] = token;
    }
}

/// The bag of the nearest visible enemy hero, else of the most recently seen one.
fn encode_enemy_items(
    tracker: &StateTracker,
    output: &mut [[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS],
) {
    let enemy = opposing(tracker.team());
    let origin = own_origin(tracker);
    let Some(hero) = tracker
        .entities()
        .iter()
        .filter(|track| track.unit.kind == UnitKind::Hero && track.unit.team == enemy)
        .min_by_key(|track| {
            let distance = origin.map_or(0, |origin| origin.distance_squared(track.unit.pos));
            (!track.visible, u32::MAX - track.last_seen_tick, distance)
        })
    else {
        return;
    };
    let first = OWN_ITEM_SLOTS + MAX_SHOP_ITEMS;
    for (slot, item) in hero.unit.items.iter().take(ENEMY_ITEM_SLOTS).enumerate() {
        let token = &mut output[first + slot];
        token[item_feature::TOKEN_PRESENT] = 1.0;
        one_hot(token, item_feature::LOCATION_START, 5, 4);
        token[item_feature::SLOT] = slot_value(slot);
        if let Some(item) = item {
            encode_item_view(token, tracker.shop(), *item);
        }
    }
}

const TICKS_PER_SECOND: f32 = 30.0;

/// Sets `token[start + index]` when `index` lies inside the `width`-wide block.
fn one_hot(token: &mut [f32], start: usize, width: usize, index: usize) {
    if index < width {
        token[start + index] = 1.0;
    }
}

fn fraction(value: f32, scale: f32) -> f32 {
    (value / scale).clamp(0.0, 1.0)
}

fn signed_fraction(value: f32, scale: f32) -> f32 {
    (value / scale).clamp(-1.0, 1.0)
}

/// ln(1 + value) against ln(1 + maximum), saturating at one.
fn log_fraction(value: f32, maximum: f32) -> f32 {
    (value.max(0.0).ln_1p() / maximum.ln_1p()).min(1.0)
}

fn near_distance(distance: f32) -> f32 {
    fraction(distance, scale::NEAR_DISTANCE)
}

fn log_distance(distance: f32) -> f32 {
    log_fraction(
        distance / scale::LOG_DISTANCE_UNIT,
        scale::LOG_DISTANCE_MAX / scale::LOG_DISTANCE_UNIT,
    )
}

/// A raw fixed-point length in world units over `scale`, clamped to [-1, 1].
fn raw_units(raw: i64, scale: f32) -> f32 {
    signed_fraction(raw as f32 / 65_536.0, scale)
}

/// Canonical cosine and sine of a facing: Dire facings turn half round.
fn canonical_facing(team: Team, angle: Angle) -> (f32, f32) {
    let brads = if team == Team::Dire {
        angle.brads.wrapping_add(1 << 15)
    } else {
        angle.brads
    };
    let radians = f32::from(brads) * (std::f32::consts::TAU / 65_536.0);
    let (sine, cosine) = radians.sin_cos();
    (cosine, sine)
}

/// World-unit offset from `from` to `to`, turned half round on the Dire side.
fn canonical_offset(team: Team, from: Vec2, to: Vec2) -> (f32, f32) {
    let sign = if team == Team::Dire { -1.0 } else { 1.0 };
    let dx = (i64::from(to.x.raw) - i64::from(from.x.raw)) as f32 / 65_536.0;
    let dy = (i64::from(to.y.raw) - i64::from(from.y.raw)) as f32 / 65_536.0;
    (sign * dx, sign * dy)
}

/// Cosine and sine of a canonical delta's direction relative to a canonical facing.
fn bearing(delta: (i64, i64), facing: (f32, f32), distance: f32) -> (f32, f32) {
    let dx = delta.0 as f32 / 65_536.0;
    let dy = delta.1 as f32 / 65_536.0;
    let (cosine, sine) = facing;
    (
        ((dx * cosine + dy * sine) / distance).clamp(-1.0, 1.0),
        ((dy * cosine - dx * sine) / distance).clamp(-1.0, 1.0),
    )
}

/// Chebyshev closest approach of a projectile, in world units over the approach scale.
fn closest_approach(relative: (i64, i64), velocity_x: i64, velocity_y: i64) -> f32 {
    let relative_x = relative.0 as f32;
    let relative_y = relative.1 as f32;
    let velocity_x = velocity_x as f32;
    let velocity_y = velocity_y as f32;
    let speed_squared = velocity_x * velocity_x + velocity_y * velocity_y;
    let time = if speed_squared > 0.0 {
        (-(relative_x * velocity_x + relative_y * velocity_y) / speed_squared).max(0.0)
    } else {
        0.0
    };
    let closest_x = relative_x + velocity_x * time;
    let closest_y = relative_y + velocity_y * time;
    fraction(
        closest_x.abs().max(closest_y.abs()) / 65_536.0,
        scale::PROJECTILE_APPROACH,
    )
}

const fn unit_kind_index(kind: UnitKind) -> usize {
    match kind {
        UnitKind::Hero => 0,
        UnitKind::CreepMelee => 1,
        UnitKind::CreepFlagbearer => 2,
        UnitKind::CreepRanged => 3,
        UnitKind::CreepSiege => 4,
        UnitKind::CreepNeutral => 5,
        UnitKind::Tower => 6,
        UnitKind::Ancient => 7,
        UnitKind::Barracks => 8,
        UnitKind::Fountain => 9,
        UnitKind::Ward => 10,
        UnitKind::Courier => 11,
    }
}

fn ability_id_class(id: AbilityId) -> usize {
    usize::from(id.0).min(ABILITY_ID_CLASSES - 1)
}

fn item_id_class(id: ItemId) -> usize {
    usize::from(id.0).min(ITEM_ID_CLASSES - 1)
}

const fn aim_index(aim: Aim) -> usize {
    match aim {
        Aim::Own => 0,
        Aim::Point => 1,
        Aim::Unit => 2,
        Aim::Tree => 3,
        Aim::Building => 4,
    }
}

const fn attribute_index(attribute: Attribute) -> usize {
    match attribute {
        Attribute::Strength => 0,
        Attribute::Agility => 1,
        Attribute::Intelligence => 2,
    }
}

fn next_observation_state(
    previous: &FeatureObservationState,
    tracker: &StateTracker,
    current: &bota_proto::WorldView,
) -> std::sync::Arc<FeatureObservationState> {
    assert!(current.projectiles.len() <= MAX_PROJECTILES);
    assert!(current.loot.len() <= MAX_LOOT);
    let mut next = FeatureObservationState::new();
    next.tick = Some(current.tick);
    next.provenance = Some(std::sync::Arc::new(tracker.provenance()));
    next.projectiles.reserve(current.projectiles.len());
    for projectile in &current.projectiles {
        let prior = previous
            .projectiles
            .iter()
            .find(|entry| entry.id == projectile.id);
        next.projectiles.push(prior.map_or(
            ProjectileObservation {
                id: projectile.id,
                first_tick: current.tick,
                last_tick: current.tick,
                previous_tick: None,
                position: projectile.pos,
                previous_position: projectile.pos,
            },
            |entry| ProjectileObservation {
                id: projectile.id,
                first_tick: entry.first_tick,
                last_tick: current.tick,
                previous_tick: Some(entry.last_tick),
                position: projectile.pos,
                previous_position: entry.position,
            },
        ));
    }
    next.loot.reserve(current.loot.len());
    for loot in &current.loot {
        let first_tick = previous
            .loot
            .iter()
            .find(|entry| entry.id == loot.id)
            .map_or(current.tick, |entry| entry.first_tick);
        next.loot.push(LootObservation {
            id: loot.id,
            first_tick,
            last_tick: current.tick,
        });
    }
    std::sync::Arc::new(next)
}

type PointSourceSemantics = (
    usize,
    Option<PointDirection>,
    Option<i32>,
    Option<UnitKind>,
    Option<EntityRelation>,
);

fn point_source_semantics(source: PointSource) -> PointSourceSemantics {
    match source {
        PointSource::Tactical { direction, radius } => {
            (0, Some(direction), Some(radius), None, None)
        }
        PointSource::StaticTree => (1, None, None, None, None),
        PointSource::PlantedTree => (2, None, None, None, None),
        PointSource::BuildingLanding(kind) => (3, None, None, Some(kind), None),
        PointSource::Fountain(relation) => (4, None, None, None, Some(landmark_relation(relation))),
        PointSource::Tower(relation) => (5, None, None, None, Some(landmark_relation(relation))),
        PointSource::PredictedHero(relation) => (6, None, None, None, Some(relation)),
        PointSource::PredictedCreep(relation) => (7, None, None, None, Some(relation)),
        PointSource::RazeFacing => (8, None, Some(crate::RAZE_RING_RADIUS), None, None),
        PointSource::RazeCluster { reach } => (9, None, Some(reach), None, None),
        PointSource::LastSeenHero { .. } => (
            10,
            None,
            None,
            Some(UnitKind::Hero),
            Some(EntityRelation::Enemy),
        ),
        PointSource::ExtrapolatedHero { .. } => (
            11,
            None,
            None,
            Some(UnitKind::Hero),
            Some(EntityRelation::Enemy),
        ),
        PointSource::RazeRing => (12, None, Some(crate::RAZE_RING_RADIUS), None, None),
    }
}

const fn landmark_relation(relation: LandmarkRelation) -> EntityRelation {
    match relation {
        LandmarkRelation::Own => EntityRelation::Own,
        LandmarkRelation::Enemy => EntityRelation::Enemy,
    }
}

const fn point_direction_index(direction: PointDirection) -> usize {
    match direction {
        PointDirection::East => 0,
        PointDirection::NorthEast => 1,
        PointDirection::North => 2,
        PointDirection::NorthWest => 3,
        PointDirection::West => 4,
        PointDirection::SouthWest => 5,
        PointDirection::South => 6,
        PointDirection::SouthEast => 7,
    }
}

fn ability_legal(action_space: &ActionSpace, unit: ControlledUnit, slot: usize) -> bool {
    let Some(slot) = u8::try_from(slot).ok() else {
        return false;
    };
    action_space
        .cast_target_mask(unit, bota_proto::AbilitySlot(slot))
        .is_some_and(|mask| {
            mask.allows_none() || mask.entities().contains(&true) || mask.points().contains(&true)
        })
}

fn item_legal(action_space: &ActionSpace, unit: ControlledUnit, slot: usize) -> bool {
    let Some(slot) = u8::try_from(slot).ok() else {
        return false;
    };
    action_space
        .use_target_mask(unit, ItemSlot(slot))
        .is_some_and(|mask| {
            mask.allows_none() || mask.entities().contains(&true) || mask.points().contains(&true)
        })
}

fn own_hero_abilities(tracker: &StateTracker) -> Option<(&[AbilityView], bool)> {
    if let Some(hero) = tracker.own_hero() {
        return Some((hero.abilities.as_slice(), false));
    }
    let kit = tracker.own_player()?.kit.as_ref()?;
    Some((kit.abilities.as_slice(), true))
}

fn own_courier_abilities(tracker: &StateTracker) -> Option<(&[AbilityView], bool)> {
    tracker
        .own_courier()
        .map(|unit| (unit.abilities.as_slice(), false))
}

fn own_hero_items(tracker: &StateTracker) -> Option<(&[Option<ItemView>], bool)> {
    if let Some(hero) = tracker.own_hero() {
        return Some((hero.items.as_slice(), false));
    }
    let kit = tracker.own_player()?.kit.as_ref()?;
    Some((kit.items.as_slice(), true))
}

fn own_courier_items(tracker: &StateTracker) -> Option<&[Option<ItemView>]> {
    tracker.own_courier().map(|unit| unit.items.as_slice())
}

fn own_body_track(tracker: &StateTracker, kind: UnitKind) -> Option<&crate::EntityTrack> {
    let mut best = None;
    let mut tied = false;
    for track in tracker.entities().iter().filter(|track| {
        track.unit.kind == kind
            && track.unit.team == tracker.team()
            && track.unit.owner == Some(tracker.slot())
    }) {
        match best {
            None => best = Some(track),
            Some(current) if track.last_seen_tick > current.last_seen_tick => {
                best = Some(track);
                tied = false;
            }
            Some(current) if track.last_seen_tick == current.last_seen_tick => tied = true,
            Some(_) => {}
        }
    }
    (!tied).then_some(best).flatten()
}

fn is_own_body_track(tracker: &StateTracker, track: &crate::EntityTrack) -> bool {
    track.unit.owner == Some(tracker.slot())
        && track.unit.team == tracker.team()
        && matches!(track.unit.kind, UnitKind::Hero | UnitKind::Courier)
}

fn snapshot_visible(tracker: &StateTracker, track: &crate::EntityTrack) -> bool {
    tracker.current().is_some_and(|current| {
        current
            .units
            .binary_search_by_key(&track.id, |unit| unit.id)
            .is_ok()
    })
}

fn insert_sorted_token<const FEATURES: usize, const TOKENS: usize>(
    output: &mut [[f32; FEATURES]; TOKENS],
    count: &mut usize,
    token: [f32; FEATURES],
) {
    let position = output[..*count]
        .iter()
        .position(|current| token_order(&token, current) == std::cmp::Ordering::Less)
        .unwrap_or(*count);
    if position >= TOKENS {
        return;
    }
    let end = (*count).min(TOKENS - 1);
    for index in (position..end).rev() {
        output[index + 1] = output[index];
    }
    output[position] = token;
    *count = (*count + 1).min(TOKENS);
}

fn token_order<const FEATURES: usize>(
    left: &[f32; FEATURES],
    right: &[f32; FEATURES],
) -> std::cmp::Ordering {
    for index in 0..FEATURES {
        let ordering = left[index].total_cmp(&right[index]);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

fn own_origin(tracker: &StateTracker) -> Option<Vec2> {
    tracker
        .own_hero()
        .or_else(|| tracker.own_courier())
        .map(|unit| unit.pos)
}

fn unit_relation(tracker: &StateTracker, unit: &UnitView) -> EntityRelation {
    if unit.owner == Some(tracker.slot()) {
        EntityRelation::Own
    } else if unit.team == tracker.team() {
        EntityRelation::Allied
    } else if unit.team == opposing(tracker.team()) {
        EntityRelation::Enemy
    } else {
        EntityRelation::Neutral
    }
}

fn owner_relation(tracker: &StateTracker, unit: &UnitView) -> Option<EntityRelation> {
    let owner = unit.owner?;
    if owner == tracker.slot() {
        return Some(EntityRelation::Own);
    }
    let player = tracker
        .current()?
        .players
        .iter()
        .find(|player| player.slot == owner)?;
    Some(if player.team == tracker.team() {
        EntityRelation::Allied
    } else {
        EntityRelation::Enemy
    })
}

fn snapshot_damage(tracker: &StateTracker) -> (i64, i64) {
    let tick = tracker.current().expect("snapshot was checked").tick;
    let mut damage = (0i64, 0i64);
    for kind in [UnitKind::Hero, UnitKind::Courier] {
        let Some(track) = own_body_track(tracker, kind) else {
            continue;
        };
        if let Some((_, amount)) = recent_damage(tracker, track.id, tick, true) {
            damage.0 = damage.0.saturating_add(i64::from(amount.max(0)));
        }
        if let Some((_, amount)) = recent_damage(tracker, track.id, tick, false) {
            damage.1 = damage.1.saturating_add(i64::from(amount.max(0)));
        }
    }
    damage
}

fn recent_damage(
    tracker: &StateTracker,
    entity: bota_proto::EntityId,
    current_tick: u32,
    dealt: bool,
) -> Option<(u32, i32)> {
    tracker.snapshot_events().iter().rev().find_map(|observed| {
        if !recent_before(Some(observed.tick), current_tick) {
            return None;
        }
        let EventKind::Damaged {
            source,
            target,
            amount,
            ..
        } = observed.kind
        else {
            return None;
        };
        let matches = if dealt {
            source == Some(entity)
        } else {
            target == entity
        };
        matches.then_some((observed.tick, amount))
    })
}

fn recent_ability_cast(
    tracker: &StateTracker,
    caster: bota_proto::EntityId,
    ability: AbilityId,
    current_tick: u32,
) -> Option<u32> {
    tracker.snapshot_events().iter().rev().find_map(|observed| {
        if !recent_before(Some(observed.tick), current_tick) {
            return None;
        }
        match observed.kind {
            EventKind::AbilityCast {
                caster: event_caster,
                ability: event_ability,
            } if event_caster == caster && event_ability == ability => Some(observed.tick),
            _ => None,
        }
    })
}

fn recent_possible_attack(
    tracker: &StateTracker,
    source: bota_proto::EntityId,
    current_tick: u32,
) -> Option<u32> {
    tracker.snapshot_events().iter().rev().find_map(|observed| {
        if !recent_before(Some(observed.tick), current_tick) {
            return None;
        }
        let EventKind::Damaged {
            source: event_source,
            kind,
            crit,
            ..
        } = observed.kind
        else {
            return None;
        };
        if event_source != Some(source) || kind != DamageKind::Physical || crit {
            return None;
        }
        let cast_same_tick = tracker.snapshot_events().iter().any(|other| {
            other.tick == observed.tick
                && matches!(other.kind, EventKind::AbilityCast { caster, .. } if caster == source)
        });
        (!cast_same_tick).then_some(observed.tick)
    })
}

/// Every item the seat owns: carried (or held by its dead hero), stashed, or on its courier.
pub(crate) fn own_items(tracker: &StateTracker) -> impl Iterator<Item = &ItemView> {
    own_hero_items(tracker)
        .into_iter()
        .flat_map(|(items, _)| items)
        .chain(
            tracker
                .own_player()
                .and_then(|player| player.stash.as_deref())
                .into_iter()
                .flatten(),
        )
        .chain(own_courier_items(tracker).into_iter().flatten())
        .flatten()
}

/// Whole shop cost of every item the seat owns.
pub(crate) fn own_asset_value(tracker: &StateTracker) -> i64 {
    let mut total = 0i64;
    for item in own_items(tracker) {
        if let Some(entry) = tracker.shop().iter().find(|entry| entry.id == item.id) {
            total = total.saturating_add(i64::from(entry.cost));
        }
    }
    total
}

fn pregame_progress(tick: u32, pregame_ticks: u32) -> f32 {
    if pregame_ticks == 0 || tick >= pregame_ticks {
        return 1.0;
    }
    ratio(i64::from(tick), 0, i64::from(pregame_ticks))
}

fn periodic_phase(tick: u32, pregame: u32, tick_rate: u16, seconds: u32) -> f32 {
    let period = u32::from(tick_rate).saturating_mul(seconds).max(1);
    let game_tick = tick.saturating_sub(pregame);
    ratio(i64::from(game_tick % period), 0, i64::from(period))
}

fn map_extent_raw(axis: usize) -> i64 {
    let units =
        i64::try_from(axis).expect("bounded terrain axis fits i64") * i64::from(TERRAIN_CELL_SIZE);
    units << Fixed::FRAC_BITS
}

fn ray_position(origin: Vec2, direction: (i32, i32), step: usize) -> Vec2 {
    let distance = (i64::try_from(step).expect("bounded ray step") * i64::from(TERRAIN_CELL_SIZE))
        << Fixed::FRAC_BITS;
    Vec2 {
        x: Fixed {
            raw: clamp_raw(i64::from(origin.x.raw) + i64::from(direction.0) * distance),
        },
        y: Fixed {
            raw: clamp_raw(i64::from(origin.y.raw) + i64::from(direction.1) * distance),
        },
    }
}

fn set_first_hit(hit: &mut Option<usize>, condition: bool, step: usize) {
    if condition && hit.is_none() {
        *hit = Some(step);
    }
}

fn same_cell(context: &FeatureEncoder, left: Vec2, right: Vec2) -> bool {
    context
        .cell_xy(left)
        .is_some_and(|cell| Some(cell) == context.cell_xy(right))
}

fn cell_index(axis: usize, position: Vec2) -> Option<usize> {
    if position.x.raw < 0 || position.y.raw < 0 {
        return None;
    }
    let x = usize::try_from(position.x.to_int() / TERRAIN_CELL_SIZE).ok()?;
    let y = usize::try_from(position.y.to_int() / TERRAIN_CELL_SIZE).ok()?;
    (x < axis && y < axis).then_some(y * axis + x)
}

fn relation_index(relation: EntityRelation) -> usize {
    match relation {
        EntityRelation::Own => 0,
        EntityRelation::Allied => 1,
        EntityRelation::Enemy => 2,
        EntityRelation::Neutral => 3,
    }
}

fn team_relation(own: Team, other: Team) -> EntityRelation {
    if own == other {
        EntityRelation::Allied
    } else if opposing(own) == other {
        EntityRelation::Enemy
    } else {
        EntityRelation::Neutral
    }
}

fn safe_fraction(value: i32, maximum: i32) -> f32 {
    if maximum <= 0 {
        return 0.0;
    }
    ratio(i64::from(value), 0, i64::from(maximum))
}

fn coordinate_ratio(raw: i32, extent_raw: i64) -> f32 {
    ratio(i64::from(raw), 0, extent_raw.saturating_sub(1).max(1))
}

fn signed_raw_ratio(raw: i64, maximum: i64) -> f32 {
    (raw as f32 / maximum.max(1) as f32).clamp(-1.0, 1.0)
}

fn recent_before(event_tick: Option<u32>, current_tick: u32) -> bool {
    event_tick.is_some_and(|tick| tick < current_tick && current_tick - tick <= RECENT_EVENT_TICKS)
}

fn unit_ratio(value: u32, maximum: u32) -> f32 {
    ratio(i64::from(value), 0, i64::from(maximum.max(1)))
}

fn ratio(value: i64, minimum: i64, maximum: i64) -> f32 {
    assert!(minimum < maximum);
    let clamped = value.clamp(minimum, maximum);
    (clamped - minimum) as f32 / (maximum - minimum) as f32
}

fn bool_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn clamp_raw(raw: i64) -> i32 {
    raw.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

const fn opposing(team: Team) -> Team {
    match team {
        Team::Radiant => Team::Dire,
        Team::Dire => Team::Radiant,
        Team::Neutral => Team::Neutral,
    }
}
