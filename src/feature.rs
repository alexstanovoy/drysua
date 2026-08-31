#![allow(
    clippy::float_arithmetic,
    reason = "policy tensors use bounded f32 values outside the deterministic simulation"
)]

use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;

use bota_proto::{
    AbilityId, AbilityView, Aim, Angle, Attribute, DamageKind, EventKind, Fixed, ItemId, ItemSlot,
    ItemView, PlayerView, ProjectileView, ShopEntry, StatusFlags, Team, UnitKind, UnitView, Vec2,
};

use crate::tracker::{StaticTrackerProvenance, TrackerProvenance};
use crate::{
    ActionKind, ActionSpace, ControlledUnit, EntityRelation, HISTORY_AGES, ItemReadiness,
    LandmarkRelation, MAX_LOOT, MAX_POINT_CANDIDATES, MAX_PROJECTILES, MAX_SHOP_ITEMS,
    OWN_ITEM_SLOTS, PointCandidate, PointDirection, PointSource, SHADOW_FIEND_ABILITY_SLOTS,
    StateTracker, TERRAIN_CELL_SIZE, UNIT_TOKENS,
};

/// Version of the append-only policy feature schema.
pub const FEATURE_SCHEMA_VERSION: u32 = 3;
/// Number of scalar global features.
pub const GLOBAL_FEATURES: usize = 64;
/// Number of scalar features in one global-history sample.
pub const HISTORY_FEATURES: usize = 24;
/// Number of global-history samples.
pub const HISTORY_SAMPLES: usize = HISTORY_AGES.len();
/// Number of scalar features in one local policy-history sample.
pub const POLICY_HISTORY_FEATURES: usize = 4;
/// Maximum number of local policy-history samples.
pub const MAX_POLICY_HISTORY: usize = 16;
/// Number of scalar features in one unit token.
pub const UNIT_FEATURES: usize = 69;
/// Number of unit tokens in exact ActionSpace entity-candidate order.
pub const UNIT_FEATURE_TOKENS: usize = UNIT_TOKENS;
/// Number of fixed own hero and courier unit tokens.
pub const OWN_UNIT_FEATURE_TOKENS: usize = 2;
/// Maximum number of non-targetable remembered unit tokens.
pub const REMEMBERED_UNIT_FEATURE_TOKENS: usize = 32;
/// Number of scalar features in one point-candidate token.
pub const POINT_FEATURES: usize = 32;
/// Number of point tokens in exact ActionSpace point-candidate order.
pub const POINT_FEATURE_TOKENS: usize = MAX_POINT_CANDIDATES;
/// Number of scalar features in one ability token.
pub const ABILITY_FEATURES: usize = 24;
/// Number of fixed own hero and courier ability tokens.
pub const ABILITY_FEATURE_TOKENS: usize = SHADOW_FIEND_ABILITY_SLOTS + 8;
/// Number of scalar features in one item token.
pub const ITEM_FEATURES: usize = 28;
/// Number of fixed own inventory and shop item tokens.
pub const ITEM_FEATURE_TOKENS: usize = OWN_ITEM_SLOTS + MAX_SHOP_ITEMS;
/// Number of scalar features in one projectile token.
pub const PROJECTILE_FEATURES: usize = 20;
/// Number of fixed projectile tokens.
pub const PROJECTILE_FEATURE_TOKENS: usize = MAX_PROJECTILES;
/// Number of scalar features in one loot token.
pub const LOOT_FEATURES: usize = 16;
/// Number of fixed loot tokens.
pub const LOOT_FEATURE_TOKENS: usize = MAX_LOOT;
/// Number of scalar local map-context features.
pub const MAP_FEATURES: usize = 96;
/// Maximum encoded observation states retained for deterministic rollback.
pub const MAX_FEATURE_OBSERVATION_HISTORY: usize = 16;

const MAX_TICK: u32 = 3_600_000;
const MAX_AGE: u32 = 4_800;
const MAX_GOLD: i64 = 100_000;
const MAX_SCORE: i64 = 1_000;
const MAX_XP: i64 = 100_000;
const MAX_HP: i64 = 100_000;
const MAX_MANA: i64 = 20_000;
const MAX_DAMAGE: i64 = 10_000;
const MAX_ATTACK_INTERVAL: i64 = 600;
const MAX_SPEED: i64 = 2_000;
const MAX_ARMOR_RAW: i64 = 100 << Fixed::FRAC_BITS;
const MAX_LEVEL: i64 = 30;
const MAX_CHARGES: i64 = 255;
const MAX_COOLDOWN: i64 = 36_000;
const MAX_ITEM_COST: i64 = 100_000;
const MAX_STRUCTURE_COUNT: i64 = 64;
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

/// Stable indices in the global feature vector.
pub mod global_feature {
    pub const TICK: usize = 0;
    pub const PREGAME_PROGRESS: usize = 1;
    pub const WAVE_PHASE: usize = 2;
    pub const JUNGLE_PHASE: usize = 3;
    pub const SIDE_RADIANT: usize = 4;
    pub const SIDE_DIRE: usize = 5;
    pub const MAP_ZERO: usize = 6;
    pub const MAP_ONE: usize = 7;
    pub const SEAT_COUNT: usize = 8;
    pub const ROLE_PRESENT: usize = 9;
    pub const ROLE_TOKEN: usize = 10;
    pub const LANE_PRESENT: usize = 11;
    pub const LANE_TOKEN: usize = 12;
    pub const KILL_ADVANTAGE: usize = 13;
    pub const DEATH_ADVANTAGE: usize = 14;
    pub const ASSIST_ADVANTAGE: usize = 15;
    pub const XP_ADVANTAGE: usize = 16;
    pub const LEVEL_ADVANTAGE: usize = 17;
    pub const LAST_HIT_ADVANTAGE: usize = 18;
    pub const DENY_ADVANTAGE: usize = 19;
    pub const OWN_GOLD: usize = 20;
    pub const OWN_ASSET_VALUE: usize = 21;
    pub const RESPAWN_PRESENT: usize = 22;
    pub const RESPAWN_LEFT: usize = 23;
    pub const OWN_ALIVE: usize = 24;
    pub const ALLIED_ALIVE_HEROES: usize = 25;
    pub const ENEMY_ALIVE_HEROES: usize = 26;
    pub const ALLIED_STRUCTURE_HP: usize = 27;
    pub const ENEMY_STRUCTURE_HP: usize = 28;
    pub const DESTROYED_STRUCTURES_PRESENT: usize = 29;
    pub const DESTROYED_STRUCTURES: usize = 30;
    pub const ACTIVE_ORDER_PRESENT: usize = 31;
    pub const ACTIVE_ORDER_KIND: usize = 32;
    pub const ACTIVE_ORDER_AGE: usize = 33;
    pub const LAST_DECISION_PRESENT: usize = 34;
    pub const TICKS_SINCE_DECISION: usize = 35;
    pub const SNAPSHOT_DAMAGE_DEALT: usize = 36;
    pub const SNAPSHOT_DAMAGE_TAKEN: usize = 37;
    pub const OWN_LEVEL: usize = 38;
    pub const OWN_XP: usize = 39;
    pub const OWN_KILLS: usize = 40;
    pub const OWN_DEATHS: usize = 41;
    pub const OWN_ASSISTS: usize = 42;
    pub const OWN_LAST_HITS: usize = 43;
    pub const OWN_DENIES: usize = 44;
    pub const VISIBLE_ALLIED_UNITS: usize = 45;
    pub const VISIBLE_ENEMY_UNITS: usize = 46;
    pub const ENEMY_SCOREBOARD_ENABLED: usize = 47;
}

/// Stable indices in each unit token.
pub mod unit_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const RELATION_START: usize = 1;
    pub const KIND_TOKEN: usize = 5;
    pub const OWNER_START: usize = 6;
    pub const OWNER_PRESENT: usize = 9;
    pub const OBSERVATION_PRESENT: usize = 10;
    pub const VISIBLE: usize = 11;
    pub const REMEMBERED: usize = 12;
    pub const ORIGIN_PRESENT: usize = 13;
    pub const AGE: usize = 14;
    pub const POSITION_X: usize = 15;
    pub const POSITION_Y: usize = 16;
    pub const RELATIVE_X: usize = 17;
    pub const RELATIVE_Y: usize = 18;
    pub const DISTANCE: usize = 19;
    pub const DIRECTION_X: usize = 20;
    pub const DIRECTION_Y: usize = 21;
    pub const FACING: usize = 22;
    pub const RADIUS: usize = 23;
    pub const VELOCITY_PRESENT: usize = 24;
    pub const VELOCITY_X: usize = 25;
    pub const VELOCITY_Y: usize = 26;
    pub const HP_PRESENT: usize = 27;
    pub const ELEVATION: usize = 28;
    pub const WALKABLE: usize = 29;
    pub const HP_RATIO: usize = 30;
    pub const MANA_PRESENT: usize = 31;
    pub const MANA_RATIO: usize = 32;
    pub const HP_DELTA_PRESENT: usize = 33;
    pub const HP_DELTA: usize = 34;
    pub const MANA_DELTA_PRESENT: usize = 35;
    pub const MANA_DELTA: usize = 36;
    pub const ATTACK_DAMAGE: usize = 37;
    pub const ATTACK_RANGE: usize = 38;
    pub const ATTACK_INTERVAL: usize = 39;
    pub const ATTACK_SPEED: usize = 40;
    pub const MOVE_SPEED: usize = 41;
    pub const ARMOR: usize = 42;
    pub const MAGIC_RESISTANCE: usize = 43;
    pub const VISION: usize = 44;
    pub const TRUE_SIGHT: usize = 45;
    pub const ATTACKS_TO_KILL_PRESENT: usize = 46;
    pub const ATTACKS_TO_KILL: usize = 47;
    pub const TIME_TO_REACH_PRESENT: usize = 48;
    pub const TIME_TO_REACH: usize = 49;
    pub const OWN_IN_ATTACK_RANGE: usize = 50;
    pub const UNIT_IN_ATTACK_RANGE: usize = 51;
    pub const STATUS_START: usize = 52;
    pub const RECENT_DAMAGE_TAKEN: usize = 61;
    pub const RECENT_DAMAGE_DEALT_PRESENT: usize = 62;
    pub const RECENT_DAMAGE_DEALT: usize = 63;
    pub const ATTACK_PHASE_PRESENT: usize = 64;
    pub const ATTACK_PHASE: usize = 65;
    pub const ITEM_SLOT_COUNT: usize = 66;
    pub const FREE_ITEM_SLOTS: usize = 67;
    pub const ITEM_CAPACITY_AVAILABLE: usize = 68;
}

/// Stable indices in each point-candidate token.
pub mod point_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const POINTER_VALID: usize = 1;
    pub const POSITION_X: usize = 2;
    pub const POSITION_Y: usize = 3;
    pub const ORIGIN_PRESENT: usize = 4;
    pub const RELATIVE_X: usize = 5;
    pub const RELATIVE_Y: usize = 6;
    pub const DISTANCE: usize = 7;
    pub const DIRECTION_X: usize = 8;
    pub const DIRECTION_Y: usize = 9;
    pub const SOURCE_TOKEN: usize = 10;
    pub const SOURCE_DIRECTION_PRESENT: usize = 11;
    pub const SOURCE_DIRECTION_TOKEN: usize = 12;
    pub const SOURCE_RADIUS_PRESENT: usize = 13;
    pub const SOURCE_RADIUS: usize = 14;
    pub const SOURCE_KIND_PRESENT: usize = 15;
    pub const SOURCE_KIND_TOKEN: usize = 16;
    pub const SOURCE_RELATION_PRESENT: usize = 17;
    pub const SOURCE_RELATION_START: usize = 18;
    pub const WALKABLE: usize = 22;
    pub const STANDING_TREE: usize = 23;
    pub const ALLIED_BUILDING: usize = 24;
}

/// Stable indices in each ability token.
pub mod ability_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const BODY_TOKEN: usize = 1;
    pub const SEMANTIC_SLOT_TOKEN: usize = 2;
    pub const OBSERVATION_PRESENT: usize = 3;
    pub const ID_PRESENT: usize = 4;
    pub const ID_TOKEN: usize = 5;
    pub const LEVEL: usize = 6;
    pub const MAX_LEVEL: usize = 7;
    pub const COOLDOWN: usize = 8;
    pub const MANA_COST: usize = 9;
    pub const RANGE: usize = 10;
    pub const AIM_TOKEN: usize = 11;
    pub const PASSIVE: usize = 12;
    pub const TOGGLE_ON: usize = 13;
    pub const CAN_LEVEL: usize = 14;
    pub const LEGAL: usize = 15;
    pub const LAST_CAST_PRESENT: usize = 16;
    pub const LAST_CAST_AGE: usize = 17;
    pub const SCOREBOARD_KIT_SOURCE: usize = 18;
}

/// Stable indices in each item token.
pub mod item_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const LOCATION_TOKEN: usize = 1;
    pub const SLOT_TOKEN: usize = 2;
    pub const ITEM_PRESENT: usize = 3;
    pub const ITEM_TOKEN: usize = 4;
    pub const CHARGES_PRESENT: usize = 5;
    pub const CHARGES: usize = 6;
    pub const COOLDOWN: usize = 7;
    pub const AIM_PRESENT: usize = 8;
    pub const AIM_TOKEN: usize = 9;
    pub const RANGE: usize = 10;
    pub const MANA_COST: usize = 11;
    pub const ATTRIBUTE_PRESENT: usize = 12;
    pub const ATTRIBUTE_TOKEN: usize = 13;
    pub const FOR_SALE: usize = 14;
    pub const MUTED: usize = 15;
    pub const VALUE_PRESENT: usize = 16;
    pub const VALUE: usize = 17;
    pub const RECIPE_COMPONENT: usize = 18;
    pub const COMPOSITE: usize = 19;
    pub const LEGAL: usize = 20;
    pub const SHOP_CANDIDATE: usize = 21;
    pub const MUTE_REMAINING_PRESENT: usize = 22;
    pub const MUTE_REMAINING: usize = 23;
    pub const SHARED_WAIT_PRESENT: usize = 24;
    pub const SHARED_WAIT_REMAINING: usize = 25;
    pub const SCOREBOARD_KIT_SOURCE: usize = 26;
}

/// Stable indices in each global-history sample.
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
    pub const XP_ADVANTAGE: usize = 12;
    pub const LEVEL_ADVANTAGE: usize = 13;
    pub const KILL_ADVANTAGE: usize = 14;
    pub const DEATH_ADVANTAGE: usize = 15;
    pub const ASSIST_ADVANTAGE: usize = 16;
    pub const LAST_HIT_ADVANTAGE: usize = 17;
    pub const DENY_ADVANTAGE: usize = 18;
    pub const ALLIED_STRUCTURE_HP: usize = 19;
    pub const ENEMY_STRUCTURE_HP: usize = 20;
    pub const DESTROYED_STRUCTURES_PRESENT: usize = 21;
    pub const DESTROYED_STRUCTURES: usize = 22;
    pub const ENEMY_SCOREBOARD_ENABLED: usize = 23;
}

/// Stable indices in each projectile token.
pub mod projectile_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const RELATION_START: usize = 1;
    pub const ABILITY_PRESENT: usize = 5;
    pub const ABILITY_TOKEN: usize = 6;
    pub const RELATIVE_X: usize = 7;
    pub const RELATIVE_Y: usize = 8;
    pub const FACING: usize = 9;
    pub const VELOCITY_PRESENT: usize = 10;
    pub const VELOCITY_X: usize = 11;
    pub const VELOCITY_Y: usize = 12;
    pub const AGE_PRESENT: usize = 13;
    pub const AGE: usize = 14;
    pub const CLOSEST_APPROACH_PRESENT: usize = 15;
    pub const CLOSEST_APPROACH: usize = 16;
    pub const ORIGIN_PRESENT: usize = 17;
}

/// Stable indices in each ground-loot token.
pub mod loot_feature {
    pub const TOKEN_PRESENT: usize = 0;
    pub const ITEM_TOKEN: usize = 1;
    pub const CHARGES_PRESENT: usize = 2;
    pub const CHARGES: usize = 3;
    pub const RELATIVE_X: usize = 4;
    pub const RELATIVE_Y: usize = 5;
    pub const DIRECT_DISTANCE: usize = 6;
    pub const PATH_DISTANCE_PRESENT: usize = 7;
    pub const PATH_DISTANCE: usize = 8;
    pub const VISIBLE_AGE_PRESENT: usize = 9;
    pub const VISIBLE_AGE: usize = 10;
    pub const ORIGIN_PRESENT: usize = 11;
}

/// Canonical schema text covered by [`FEATURE_SCHEMA_HASH`].
pub const FEATURE_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-feature/v3;",
    "shapes=global:64,history:7x24,policy_history:16x4,unit:96x69,own_unit:2x69,remembered_unit:32x69,point:48x32,ability:14x24,item:85x28,projectile:32x20,loot:16x16,map:96;",
    "scalar_ranges=presence_and_one_hot:[0,1],unsigned_continuous:[0,1],signed_continuous:[-1,1],category:positive_exact_integer,all_finite;",
    "history_ages=480,240,120,60,30,15,0;",
    "normalizers=tick:3600000,age:4800,history_age:480,gold_asset_item_cost:100000,score:1000,xp:100000,hp:100000,mana:20000,damage:10000,attack_interval:600,speed:2000,armor_raw:6553600,level:30,charges:255,cooldown:36000,structures:64;",
    "coordinates=raw_extent:terrain_cells*64*65536,dire_position:(extent_raw-1)-raw,dire_delta:negated,dire_facing:brads+32768_wrapping,absolute_side:global[4:6];",
    "categories=unit_kind:Hero1,CreepMelee2,CreepFlagbearer3,CreepRanged4,CreepSiege5,CreepNeutral6,Roshan7,Tower8,Ancient9,Fountain10,Ward11,Courier12;",
    "ability_category=id8:1,id9:2,id10:3,id11:4,id12:5,id13:6,id14:7,id15:8,id16:9,id17:10,id18:11,other:raw+12,range:1..65547;",
    "item_category=raw+1,range:1..65536;aim=Own1,Point2,Unit3,Tree4,Building5;attribute=Strength1,Agility2,Intelligence3;",
    "action_category=Continue1,Stop2,MovePoint3,FollowUnit4,Hold5,AttackMovePoint6,AttackUnit7,Cast8,Use9,PutPoint10,PutUnit11,Take12,Buy13,Sell14,Swap15,Learn16;",
    "role_category=Carry1,Mid2,Offlane3,Support4,HardSupport5;lane_category=Safe1,Mid2,Offlane3;body_category=Hero1,Courier2;item_location=hero2,stash3,courier4,shop5;slots=zero_based_plus_one;",
    "unit_leading_when_present=own-hero-then-own-courier;item_fixed=hero9,stash6,courier6,shop64;",
    "unit_candidates=current_visible_live_only,cap96,priority:own_body_then_hero_then_structure_then_within1200_then_other,distance_then_relation_then_owner_relation_then_canonical_model_semantics_then_entity_id_only_for_semantically_identical_ties;",
    "unit_semantic_order=kind,canonical_position,canonical_facing,hp,max_hp,mana,max_mana,move_speed,attack_damage,attack_range,attack_interval,attack_speed,armor,magic_resistance,radius,vision,true_sight,statuses,item_slot_count,free_item_slots,item_capacity_available,canonical_velocity,hp_delta,mana_delta,recent_damage,recent_cast,recent_attack;",
    "unit_memory=units_exact_current_pointer_order,own_units_fixed_hero_courier_current_or_remembered,remembered_units_nonown_hidden_cap32_lexicographic_complete_encoded_token,tracker_cap256_evict_complete_oldest_invisible_last_seen_tick_cohorts,no_target_handles;",
    "point_candidates=cap48,deduplicate_position_keep_first_source,canonical_team_directions,generate:tactical_radii200_600_1200_in_E_NE_N_NW_W_SW_S_SE_order_then_allied_building_landings_then_nearest8_visible_or_static-baseline_trees_then_own_fountain_enemy_fountain_own_tower_enemy_tower_then_predicted_units;",
    "point_features=exact_action_pointer_prefix,present,pointer_valid,canonical_position,relative_position,distance,direction,source_category,direction_radius_kind_relation_parameters,walkable,standing_tree,allied_building;",
    "point_order=building:distance_kind_canonical_landing_position_entity_id_identical_tie,tree:distance_canonical_position_planted,predicted:distance_source_relation_canonical_position_entity_id_identical_tie,landmark:distance_canonical_position_entity_id_identical_tie;shop_order=item_id;",
    "loot_candidates=current_visible,cap16,order:item_then_charges_then_position_then_entity_id_only_for_semantically_identical_ties;",
    "projectile_order=lexicographic_encoded_semantics,feature_identical_ties_indistinguishable;",
    "projectile_history=continuous_full_handle_observation,cap32,first_age,second_velocity,closest_approach,disappearance_or_generation_resets;loot_history=continuous_full_handle_observation,cap16,visible_age,duplicate_current_semantics_suppress_identity_age;",
    "observation_journal=16_states,strictly_increasing_observe,exact_tracker_lineage_slot_static_snapshot_predecessor_and_tracker_history_provenance,rollback_at_or_before,eviction_horizon_exact_error_and_atomic,reset_empty,failed_observe_and_encode_atomic;",
    "loot_path=static-terrain-and-static-tree-grid;",
    "map_rays=E_NE_N_NW_W_SW_S_SE,20_cells,64_world_units_per_step,first_nonwalkable_water_opaque_tree_and_endpoint_elevation_walkability;",
    "visibility=allied_current_units,positive_vision_radius,within_radius,target_elevation_not_above_viewer,exact_fixed_point_supercover_line,intermediate_opaque_or_higher_cell_blocks,corner_touch_checks_both_cells;",
    "trees=opaque_cells_include_static_map_tree_cells,static_occupancy_baseline,dynamic_delta_proof_requires_live_allied_body_in_same_or_adjacent_cell,proof_ignores_all_dynamic_tree_entries,local_felled_unblocks_passability_and_tree_mask,local_planted_blocks_passability_and_enters_tree_mask,remote_dynamic_changes_invariant,no_hidden_dynamic_blocker_channel;",
    "events=input_batch_cap:payload_len_div_2:2097152_reject_before_mutation,snapshot_event_journal_cap64,only_ticks_strictly_before_snapshot,same_tick_delivery_cannot_overwrite_or_evict_prior_snapshot_features,ability_cast_age_per_caster_and_ability,combat_phase_for_any_tracked_source;",
    "readiness=backpack_mute:180_from_apply_tick,teleport_shared_wait:2100_from_apply_tick,hero_inventory_journals6,body_shared_journals2,recent_request_cap8,effective_evicted_base_retained,rejection_exact_for_retained_sequences,evicted_sequence_rejection_unsupported,retained_rejections_restore_base;",
    "local_rollback=active_assignment_transition_cap16,evicted_effective_base_retained,earliest_supported_tick_tracked,rollback_before_horizon_exact_error_and_atomic,decision_eviction_advances_horizon_to_incoming_tick;",
    "own_payloads=live_body_current,hero_scoreboard_kit_current_with_source_bit,absent_courier_ability_and_item_payloads_missing,no_remembered_body_payload_fallback;",
    "provenance=private_nonzero_checked_tracker_lineage_clone_gets_fresh_lineage_move_preserves_lineage,action_space_exact_bounded_lineage_slot_static_snapshot_tracker_comparison,readiness_exact_bounded_comparison,observation_exact_bounded_lineage_slot_static_snapshot_tracker_comparison,encoder_static_exact_bounded_comparison,no_correctness_claim_for_fnv_schema_hash;",
    "audit=enemy_scoreboard_disabled_by_default,global47_and_history23_presence,disabled_zeros_enemy_alive_and_score_xp_level_kda_last_hit_deny_advantages;",
    "ids=entity_match_seed_tracker_lineage_excluded_from_frame,entity_full_handle_only_memory_key_and_final_identical_tie,ability_and_item_semantic_categories_visible;",
    "global_indices=0:tick,1:pregame,2:wave,3:jungle,4:radiant,5:dire,6:map0,7:map1,8:seats,9:role_present,10:role,11:lane_present,12:lane,13:kill_adv,14:death_adv,15:assist_adv,16:xp_adv,17:level_adv,18:lh_adv,19:deny_adv,20:gold,21:assets,22:respawn_present,23:respawn,24:alive,25:allied_alive,26:enemy_alive,27:allied_structure_hp,28:enemy_structure_hp,29:destroyed_present,30:destroyed,31:order_present,32:order,33:order_age,34:decision_present,35:decision_age,36:damage_dealt,37:damage_taken,38:level,39:xp,40:kills,41:deaths,42:assists,43:lh,44:denies,45:allied_visible,46:enemy_visible,47:enemy_scoreboard_enabled,48-63:reserved;",
    "history_indices=0:present,1:age,2:hp_present,3:hp,4:mana_present,5:mana,6:level,7:gold,8:alive,9:respawn,10:allied_visible,11:enemy_visible,12:xp_adv,13:level_adv,14:kill_adv,15:death_adv,16:assist_adv,17:lh_adv,18:deny_adv,19:allied_structure_hp,20:enemy_structure_hp,21:destroyed_present,22:destroyed,23:enemy_scoreboard_enabled;",
    "policy_history_indices=0:present,1:age,2:kind_present,3:kind;",
    "unit_indices=0:present,1-4:relation,5:kind,6-8:owner_relation,9:owner_present,10:observation,11:visible,12:remembered,13:origin_present,14:age,15-16:position,17-18:relative,19:distance,20-21:direction,22:facing,23:radius,24:velocity_present,25-26:velocity,27:hp_present,28:elevation,29:walkable,30:hp,31:mana_present,32:mana,33:hp_delta_present,34:hp_delta,35:mana_delta_present,36:mana_delta,37:attack_damage,38:attack_range,39:attack_interval,40:attack_speed,41:move_speed,42:armor,43:magic_resistance,44:vision,45:true_sight,46:attacks_present,47:attacks,48:reach_present,49:reach,50-51:mutual_range,52-60:statuses,61:damage_taken,62:damage_dealt_present,63:damage_dealt,64:attack_phase_present,65:attack_phase,66:item_slot_count,67:free_item_slots,68:item_capacity_available;",
    "point_indices=0:present,1:pointer_valid,2-3:position,4:origin_present,5-6:relative,7:distance,8-9:direction,10:source,11:source_direction_present,12:source_direction,13:source_radius_present,14:source_radius,15:source_kind_present,16:source_kind,17:source_relation_present,18-21:source_relation,22:walkable,23:standing_tree,24:allied_building,25-31:reserved;",
    "ability_indices=0:present,1:body,2:slot,3:observation,4:id_present,5:id,6:level,7:max_level,8:cooldown,9:mana,10:range,11:aim,12:passive,13:toggle,14:can_level,15:legal,16:last_cast_present,17:last_cast_age,18:scoreboard_kit_source,19-23:reserved;",
    "item_indices=0:present,1:location,2:slot,3:item_present,4:item,5:charges_present,6:charges,7:cooldown,8:aim_present,9:aim,10:range,11:mana,12:attribute_present,13:attribute,14:for_sale,15:muted,16:value_present,17:value,18:recipe_component,19:composite,20:legal,21:shop,22:mute_present,23:mute_left,24:shared_wait_present,25:shared_wait_left,26:scoreboard_kit_source,27:reserved;",
    "projectile_indices=0:present,1-4:relation,5:ability_present,6:ability,7-8:relative,9:facing,10:velocity_present,11-12:velocity,13:age_present,14:age,15:approach_present,16:approach,17:origin_present,18-19:reserved;",
    "loot_indices=0:present,1:item,2:charges_present,3:charges,4-5:relative,6:direct_distance,7:path_present,8:path,9:age_present,10:age,11:origin_present,12-15:reserved;",
    "map_indices=0:present,1:walkable,2:water,3:elevation,4:opaque,5:tree,6-13:landmark_presence_distance_pairs,14-15:reserved,16-95:eight_rays_each_four_presence_distance_pairs_plus_endpoint_elevation_walkable;",
    "global_scalars=normalizers:tick3600000_pregame_ticks_wave30s_jungle60s_seats10_score1000_xp100000_gold100000_age4800_damage10000_hp100000_structures64_visible256_level30,categories:role1..5_lane1..3_action1..16_map_onehot_side_onehot,reserved:48..63;",
    "history_scalars=normalizers:age480_hp_ratio_mana_ratio_level30_gold100000_visible256_score1000_xp100000_hp100000_structures64,categories:none,reserved:none;",
    "policy_history_scalars=normalizers:age4800,categories:action1..16,reserved:none;",
    "unit_scalars=normalizers:age480_position_extent_delta_extent_distance_extent_facing65535_radius_extent_hp_ratio_mana_ratio_damage10000_attack_range_fixed_max_attack_interval600_attack_speed2000_move_speed2000_armor_raw6553600_magic_resistance_fixed_max_vision_fixed_max_attacks100_reach4800_item_slots9,categories:relation_onehot4_kind1..12_owner_relation_onehot3_status_bits9,reserved:none;",
    "point_scalars=normalizers:position_extent_relative_extent_distance_extent_radius1200,categories:source1..8_direction1..8_kind1..12_relation_onehot4,reserved:25..31;",
    "ability_scalars=normalizers:level30_cooldown36000_mana20000_range_fixed_max_age4800,categories:body1..2_slot1..8_ability1..65547_aim1..5,reserved:19..23;",
    "item_scalars=normalizers:charges255_cooldown36000_range_fixed_max_mana20000_value100000_mute36000_shared_wait36000,categories:location_hero2_stash3_courier4_shop5_with1_reserved_slot1..64_item1..65536_aim1..5_attribute1..3,reserved:27;",
    "projectile_scalars=normalizers:relative_extent_facing65535_velocity_extent_per_tick_age4800_closest_approach_extent,categories:relation_onehot4_ability1..65547,reserved:18..19;",
    "loot_scalars=normalizers:charges255_relative_extent_direct_extent_path_axis_squared_age4800,categories:item1..65536,reserved:12..15;",
    "map_scalars=normalizers:elevation63_landmark_squared_extent_ray_step20,categories:direction_fixed_E_NE_N_NW_W_SW_S_SE_hit_kind_walkable_water_opaque_tree,reserved:14..15;"
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

/// Stable FNV-1a hash of the complete version-three schema descriptor.
pub const FEATURE_SCHEMA_HASH: u64 = fnv1a(FEATURE_SCHEMA_DESCRIPTOR.as_bytes());

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

#[derive(Clone, Debug)]
pub struct FeatureFrame {
    provenance: Option<FeatureFrameProvenance>,
    pub(crate) global: [f32; GLOBAL_FEATURES],
    pub(crate) history: [[f32; HISTORY_FEATURES]; HISTORY_SAMPLES],
    pub(crate) policy_history: [[f32; POLICY_HISTORY_FEATURES]; MAX_POLICY_HISTORY],
    pub(crate) units: [[f32; UNIT_FEATURES]; UNIT_FEATURE_TOKENS],
    pub(crate) own_units: [[f32; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS],
    pub(crate) remembered_units: [[f32; UNIT_FEATURES]; REMEMBERED_UNIT_FEATURE_TOKENS],
    pub(crate) points: [[f32; POINT_FEATURES]; POINT_FEATURE_TOKENS],
    pub(crate) abilities: [[f32; ABILITY_FEATURES]; ABILITY_FEATURE_TOKENS],
    pub(crate) items: [[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS],
    pub(crate) projectiles: [[f32; PROJECTILE_FEATURES]; PROJECTILE_FEATURE_TOKENS],
    pub(crate) loot: [[f32; LOOT_FEATURES]; LOOT_FEATURE_TOKENS],
    pub(crate) map: [f32; MAP_FEATURES],
}

impl FeatureFrame {
    /// Creates a zeroed fixed-shape frame.
    pub const fn new() -> Self {
        Self {
            provenance: None,
            global: [0.0; GLOBAL_FEATURES],
            history: [[0.0; HISTORY_FEATURES]; HISTORY_SAMPLES],
            policy_history: [[0.0; POLICY_HISTORY_FEATURES]; MAX_POLICY_HISTORY],
            units: [[0.0; UNIT_FEATURES]; UNIT_FEATURE_TOKENS],
            own_units: [[0.0; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS],
            remembered_units: [[0.0; UNIT_FEATURES]; REMEMBERED_UNIT_FEATURE_TOKENS],
            points: [[0.0; POINT_FEATURES]; POINT_FEATURE_TOKENS],
            abilities: [[0.0; ABILITY_FEATURES]; ABILITY_FEATURE_TOKENS],
            items: [[0.0; ITEM_FEATURES]; ITEM_FEATURE_TOKENS],
            projectiles: [[0.0; PROJECTILE_FEATURES]; PROJECTILE_FEATURE_TOKENS],
            loot: [[0.0; LOOT_FEATURES]; LOOT_FEATURE_TOKENS],
            map: [0.0; MAP_FEATURES],
        }
    }

    /// Global scalar features in stable schema order.
    pub const fn global(&self) -> &[f32; GLOBAL_FEATURES] {
        &self.global
    }

    /// Global-history samples in oldest-to-newest schema order.
    pub const fn history(&self) -> &[[f32; HISTORY_FEATURES]; HISTORY_SAMPLES] {
        &self.history
    }

    /// Local policy-history samples in newest-first encoded order.
    pub const fn policy_history(&self) -> &[[f32; POLICY_HISTORY_FEATURES]; MAX_POLICY_HISTORY] {
        &self.policy_history
    }

    /// Current unit tokens in exact entity-pointer order.
    pub const fn units(&self) -> &[[f32; UNIT_FEATURES]; UNIT_FEATURE_TOKENS] {
        &self.units
    }

    /// Fixed own hero and courier unit tokens.
    pub const fn own_units(&self) -> &[[f32; UNIT_FEATURES]; OWN_UNIT_FEATURE_TOKENS] {
        &self.own_units
    }

    /// Non-targetable remembered unit tokens.
    pub const fn remembered_units(
        &self,
    ) -> &[[f32; UNIT_FEATURES]; REMEMBERED_UNIT_FEATURE_TOKENS] {
        &self.remembered_units
    }

    /// Point tokens in exact point-pointer order.
    pub const fn points(&self) -> &[[f32; POINT_FEATURES]; POINT_FEATURE_TOKENS] {
        &self.points
    }

    /// Fixed own-body ability tokens.
    pub const fn abilities(&self) -> &[[f32; ABILITY_FEATURES]; ABILITY_FEATURE_TOKENS] {
        &self.abilities
    }

    /// Fixed inventory and shop item tokens.
    pub const fn items(&self) -> &[[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS] {
        &self.items
    }

    /// Current projectile tokens in deterministic semantic order.
    pub const fn projectiles(&self) -> &[[f32; PROJECTILE_FEATURES]; PROJECTILE_FEATURE_TOKENS] {
        &self.projectiles
    }

    /// Current loot tokens in exact loot-pointer order.
    pub const fn loot(&self) -> &[[f32; LOOT_FEATURES]; LOOT_FEATURE_TOKENS] {
        &self.loot
    }

    /// Fixed local map-context scalars.
    pub const fn map(&self) -> &[f32; MAP_FEATURES] {
        &self.map
    }

    /// Whether every scalar in the frame is finite.
    pub fn is_finite(&self) -> bool {
        self.global.iter().all(|value| value.is_finite())
            && self.history.iter().flatten().all(|value| value.is_finite())
            && self
                .policy_history
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && self.units.iter().flatten().all(|value| value.is_finite())
            && self
                .own_units
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && self
                .remembered_units
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && self.points.iter().flatten().all(|value| value.is_finite())
            && self
                .abilities
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && self.items.iter().flatten().all(|value| value.is_finite())
            && self
                .projectiles
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && self.loot.iter().flatten().all(|value| value.is_finite())
            && self.map.iter().all(|value| value.is_finite())
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
}

/// Local strategic role supplied by the policy configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyRole {
    Carry,
    Mid,
    Offlane,
    Support,
    HardSupport,
}

/// Local lane assignment supplied by the policy configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyLane {
    Safe,
    Mid,
    Offlane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PolicyAssignment {
    tick: u32,
    role: Option<PolicyRole>,
    lane: Option<PolicyLane>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActivePolicyTransition {
    tick: u32,
    kind: Option<ActionKind>,
}

/// Invalid chronology or unsupported rollback supplied to [`LocalPolicyState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPolicyError {
    TickRegression { incoming: u32, latest: u32 },
    RollbackBeforeHorizon { requested: u32, earliest: u32 },
}

impl fmt::Display for LocalPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TickRegression { incoming, latest } => write!(
                formatter,
                "local policy tick {incoming} is older than latest tick {latest}"
            ),
            Self::RollbackBeforeHorizon {
                requested,
                earliest,
            } => write!(
                formatter,
                "local policy rollback tick {requested} is older than earliest supported tick {earliest}"
            ),
        }
    }
}

impl Error for LocalPolicyError {}

/// Explicit bounded local policy state used by active-order and history features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalPolicyState {
    earliest_rollback_tick: u32,
    latest_tick: u32,
    decisions: [Option<PolicyDecision>; MAX_POLICY_HISTORY],
    decision_count: usize,
    active_base: Option<ActivePolicyOrder>,
    active_transitions: [Option<ActivePolicyTransition>; MAX_POLICY_HISTORY],
    active_transition_count: usize,
    assignment_base: Option<PolicyAssignment>,
    assignments: [Option<PolicyAssignment>; MAX_POLICY_HISTORY],
    assignment_count: usize,
}

impl LocalPolicyState {
    /// Creates empty local state at one rollback-safe baseline tick.
    pub const fn new(tick: u32) -> Self {
        Self {
            earliest_rollback_tick: tick,
            latest_tick: tick,
            decisions: [None; MAX_POLICY_HISTORY],
            decision_count: 0,
            active_base: None,
            active_transitions: [None; MAX_POLICY_HISTORY],
            active_transition_count: 0,
            assignment_base: None,
            assignments: [None; MAX_POLICY_HISTORY],
            assignment_count: 0,
        }
    }

    /// Clears all local policy data and starts a new epoch at `tick`.
    pub fn reset(&mut self, tick: u32) {
        *self = Self::new(tick);
    }

    /// Removes local data newer than `tick` when the bounded journal can prove it.
    pub fn rollback(&mut self, tick: u32) -> Result<(), LocalPolicyError> {
        if tick < self.earliest_rollback_tick {
            return Err(LocalPolicyError::RollbackBeforeHorizon {
                requested: tick,
                earliest: self.earliest_rollback_tick,
            });
        }
        let mut kept = [None; MAX_POLICY_HISTORY];
        let mut count = 0usize;
        for decision in self.decisions().copied() {
            if decision.tick <= tick {
                kept[count] = Some(decision);
                count += 1;
            }
        }
        self.decisions = kept;
        self.decision_count = count;
        self.rollback_active(tick);
        self.rollback_assignment(tick);
        self.latest_tick = tick;
        Ok(())
    }

    /// Records one decision, evicting the oldest entry at the fixed bound.
    pub fn note_decision(&mut self, tick: u32, kind: ActionKind) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        if self.decision_count == MAX_POLICY_HISTORY {
            self.decisions.copy_within(1..MAX_POLICY_HISTORY, 0);
            self.decision_count -= 1;
            self.earliest_rollback_tick = self.earliest_rollback_tick.max(tick);
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
        self.push_active_transition(ActivePolicyTransition { tick, kind });
        self.latest_tick = tick;
        Ok(())
    }

    /// Replaces the explicit role and lane assignment at one local tick.
    pub fn set_assignment(
        &mut self,
        tick: u32,
        role: Option<PolicyRole>,
        lane: Option<PolicyLane>,
    ) -> Result<(), LocalPolicyError> {
        self.check_tick(tick)?;
        self.push_assignment(PolicyAssignment { tick, role, lane });
        self.latest_tick = tick;
        Ok(())
    }

    /// Decisions in oldest-to-newest order.
    pub fn decisions(&self) -> impl DoubleEndedIterator<Item = &PolicyDecision> {
        self.decisions[..self.decision_count]
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Current active order, if local policy state supplies one.
    pub const fn active_order(&self) -> Option<ActivePolicyOrder> {
        let mut active = self.active_base;
        let mut index = 0usize;
        while index < self.active_transition_count {
            if let Some(transition) = self.active_transitions[index] {
                active = match transition.kind {
                    Some(kind) => Some(ActivePolicyOrder {
                        started_tick: transition.tick,
                        kind,
                    }),
                    None => None,
                };
            }
            index += 1;
        }
        active
    }

    fn assignment(&self) -> Option<PolicyAssignment> {
        self.assignments[..self.assignment_count]
            .iter()
            .rev()
            .find_map(|assignment| *assignment)
            .or(self.assignment_base)
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

    fn push_active_transition(&mut self, transition: ActivePolicyTransition) {
        if self.active_transition_count == MAX_POLICY_HISTORY {
            let evicted = self.active_transitions[0].expect("filled active transition");
            self.active_base = apply_active_transition(self.active_base, evicted);
            self.earliest_rollback_tick = self.earliest_rollback_tick.max(evicted.tick);
            self.active_transitions
                .copy_within(1..MAX_POLICY_HISTORY, 0);
            self.active_transition_count -= 1;
        }
        self.active_transitions[self.active_transition_count] = Some(transition);
        self.active_transition_count += 1;
    }

    fn push_assignment(&mut self, assignment: PolicyAssignment) {
        if self.assignment_count == MAX_POLICY_HISTORY {
            let evicted = self.assignments[0].expect("filled assignment transition");
            self.assignment_base = Some(evicted);
            self.earliest_rollback_tick = self.earliest_rollback_tick.max(evicted.tick);
            self.assignments.copy_within(1..MAX_POLICY_HISTORY, 0);
            self.assignment_count -= 1;
        }
        self.assignments[self.assignment_count] = Some(assignment);
        self.assignment_count += 1;
    }

    fn rollback_active(&mut self, tick: u32) {
        self.active_transition_count = self.active_transitions[..self.active_transition_count]
            .iter()
            .take_while(|entry| entry.is_some_and(|transition| transition.tick <= tick))
            .count();
        self.active_transitions[self.active_transition_count..].fill(None);
    }

    fn rollback_assignment(&mut self, tick: u32) {
        self.assignment_count = self.assignments[..self.assignment_count]
            .iter()
            .take_while(|entry| entry.is_some_and(|assignment| assignment.tick <= tick))
            .count();
        self.assignments[self.assignment_count..].fill(None);
    }
}

const fn apply_active_transition(
    _active: Option<ActivePolicyOrder>,
    transition: ActivePolicyTransition,
) -> Option<ActivePolicyOrder> {
    match transition.kind {
        Some(kind) => Some(ActivePolicyOrder {
            started_tick: transition.tick,
            kind,
        }),
        None => None,
    }
}

/// Feature construction failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureError {
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
    ObservationRollbackBeforeHorizon { requested: u32, earliest: u32 },
    NonFinite,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::ObservationRollbackBeforeHorizon {
                requested,
                earliest,
            } => write!(
                formatter,
                "feature observation rollback tick {requested} is older than earliest supported tick {earliest}"
            ),
            Self::NonFinite => formatter.write_str("feature encoder produced a non-finite value"),
        }
    }
}

impl Error for FeatureError {}

/// Explicit controls for policy inputs that can expose audited wire data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureAuditConfig {
    /// Whether enemy scoreboard values and derived history enter the model frame.
    pub enemy_scoreboard: bool,
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct FeatureObservationState {
    tick: Option<u32>,
    provenance: Option<TrackerProvenance>,
    projectiles: [Option<ProjectileObservation>; MAX_PROJECTILES],
    projectile_count: usize,
    loot: [Option<LootObservation>; MAX_LOOT],
    loot_count: usize,
}

impl FeatureObservationState {
    const fn new() -> Self {
        Self {
            tick: None,
            provenance: None,
            projectiles: [None; MAX_PROJECTILES],
            projectile_count: 0,
            loot: [None; MAX_LOOT],
            loot_count: 0,
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
    audit: FeatureAuditConfig,
    map: bota_proto::MapId,
    static_provenance: StaticTrackerProvenance,
    lineage: NonZeroU64,
    axis: usize,
    extent_raw: i64,
    terrain: Vec<u8>,
    opaque: Vec<bool>,
    static_tree_index: Vec<(usize, u32)>,
    path_distances: Vec<u32>,
    path_queue: Vec<usize>,
    path_origin: Option<usize>,
    observation: FeatureObservationState,
    observation_history: [Option<FeatureObservationState>; MAX_FEATURE_OBSERVATION_HISTORY],
    observation_history_count: usize,
    earliest_observation_rollback_tick: Option<u32>,
}

impl FeatureEncoder {
    /// Builds bounded static map context from validated public tracker inputs.
    pub fn new(tracker: &StateTracker) -> Self {
        Self::new_with_audit(tracker, FeatureAuditConfig::default())
    }

    /// Builds map context with explicit enemy-scoreboard audit controls.
    pub fn new_with_audit(tracker: &StateTracker, audit: FeatureAuditConfig) -> Self {
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
        Self {
            audit,
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
            observation: FeatureObservationState::new(),
            observation_history: std::array::from_fn(|_| None),
            observation_history_count: 0,
            earliest_observation_rollback_tick: None,
        }
    }

    /// Records projectile and loot observations exactly once for one snapshot.
    pub fn observe(&mut self, tracker: &StateTracker) -> Result<(), FeatureError> {
        let current = tracker.current().ok_or(FeatureError::SnapshotRequired)?;
        self.validate_map(tracker)?;
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

    /// Restores the newest retained observation at or before `tick`.
    pub fn rollback(&mut self, tick: u32) -> Result<(), FeatureError> {
        if let Some(earliest) = self.earliest_observation_rollback_tick
            && tick < earliest
        {
            return Err(FeatureError::ObservationRollbackBeforeHorizon {
                requested: tick,
                earliest,
            });
        }
        let kept = self.observation_history[..self.observation_history_count]
            .iter()
            .take_while(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|state| state.tick.is_some_and(|t| t <= tick))
            })
            .count();
        self.observation_history_count = kept;
        self.observation_history[kept..].fill(None);
        self.observation = self.observation_history[..kept]
            .iter()
            .rev()
            .find_map(|entry| entry.clone())
            .unwrap_or_else(FeatureObservationState::new);
        self.path_origin = None;
        Ok(())
    }

    /// Clears all dynamic observations while retaining the static map allocation.
    pub fn reset(&mut self) {
        self.observation = FeatureObservationState::new();
        self.observation_history.fill(None);
        self.observation_history_count = 0;
        self.earliest_observation_rollback_tick = None;
        self.path_origin = None;
    }

    /// Encodes one fixed frame from seat-safe state and explicit local history.
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
        *output = FeatureFrame::new();
        self.encode_global(tracker, local, output);
        self.encode_history(tracker, output);
        self.encode_policy_history(current.tick, local, output);
        self.encode_units(tracker, action_space, output);
        self.encode_own_units(tracker, output);
        self.encode_remembered_units(tracker, output);
        self.encode_points(tracker, action_space, output);
        self.encode_abilities(tracker, action_space, output);
        self.encode_items(tracker, action_space, readiness, output);
        self.encode_projectiles(tracker, output);
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

    fn push_observation(&mut self, observation: FeatureObservationState) {
        if self.observation_history_count == MAX_FEATURE_OBSERVATION_HISTORY {
            self.earliest_observation_rollback_tick = self.observation_history[1]
                .as_ref()
                .and_then(|state| state.tick);
            self.observation_history.rotate_left(1);
            self.observation_history[MAX_FEATURE_OBSERVATION_HISTORY - 1] = None;
            self.observation_history_count -= 1;
        }
        self.observation = observation.clone();
        self.observation_history[self.observation_history_count] = Some(observation);
        self.observation_history_count += 1;
        self.path_origin = None;
    }

    fn encode_global(
        &self,
        tracker: &StateTracker,
        local: &LocalPolicyState,
        output: &mut FeatureFrame,
    ) {
        use global_feature as index;
        let current = tracker.current().expect("snapshot was checked");
        let metadata = tracker.metadata();
        let own = tracker.own_player().expect("own player was validated");
        let summary = tracker.history()[HISTORY_SAMPLES - 1];
        output.global[index::TICK] = unit_ratio(current.tick, MAX_TICK);
        output.global[index::PREGAME_PROGRESS] =
            pregame_progress(current.tick, metadata.pregame_ticks);
        output.global[index::WAVE_PHASE] =
            periodic_phase(current.tick, metadata.pregame_ticks, metadata.tick_rate, 30);
        output.global[index::JUNGLE_PHASE] =
            periodic_phase(current.tick, metadata.pregame_ticks, metadata.tick_rate, 60);
        output.global[index::SIDE_RADIANT] = bool_feature(tracker.team() == Team::Radiant);
        output.global[index::SIDE_DIRE] = bool_feature(tracker.team() == Team::Dire);
        output.global[index::MAP_ZERO] = bool_feature(metadata.map.0 == 0);
        output.global[index::MAP_ONE] = bool_feature(metadata.map.0 == 1);
        output.global[index::SEAT_COUNT] = ratio(i64::from(metadata.seats), 1, 10);
        encode_assignment(local, &mut output.global);
        output.global[index::ENEMY_SCOREBOARD_ENABLED] = bool_feature(self.audit.enemy_scoreboard);
        if self.audit.enemy_scoreboard {
            encode_score_advantages(&mut output.global, &summary);
        }
        output.global[index::OWN_GOLD] = signed_ratio(i64::from(own.gold.unwrap_or(0)), MAX_GOLD);
        output.global[index::OWN_ASSET_VALUE] = signed_ratio(own_asset_value(tracker), MAX_GOLD);
        output.global[index::RESPAWN_PRESENT] = bool_feature(own.unit.is_none());
        output.global[index::RESPAWN_LEFT] = unit_ratio(own.respawn_left, MAX_AGE);
        output.global[index::OWN_ALIVE] = bool_feature(own.unit.is_some());
        let (allied_alive, enemy_alive) =
            alive_hero_counts(current.players.as_slice(), tracker.team());
        output.global[index::ALLIED_ALIVE_HEROES] = ratio(allied_alive, 0, 5);
        if self.audit.enemy_scoreboard {
            output.global[index::ENEMY_ALIVE_HEROES] = ratio(enemy_alive, 0, 5);
        }
        output.global[index::ALLIED_STRUCTURE_HP] =
            signed_ratio(summary.allied_structure_hp, MAX_HP);
        output.global[index::ENEMY_STRUCTURE_HP] = signed_ratio(summary.enemy_structure_hp, MAX_HP);
        output.global[index::DESTROYED_STRUCTURES_PRESENT] =
            bool_feature(summary.destroyed_structures_present);
        let destroyed = summary
            .allied_structures_destroyed
            .saturating_add(summary.enemy_structures_destroyed);
        output.global[index::DESTROYED_STRUCTURES] =
            ratio(i64::from(destroyed), 0, MAX_STRUCTURE_COUNT);
        encode_local_global(current.tick, local, &mut output.global);
        encode_own_score(own, &mut output.global);
        output.global[index::VISIBLE_ALLIED_UNITS] =
            ratio(i64::from(summary.visible_allied_units), 0, 256);
        output.global[index::VISIBLE_ENEMY_UNITS] =
            ratio(i64::from(summary.visible_enemy_units), 0, 256);
        let (dealt, taken) = snapshot_damage(tracker);
        output.global[index::SNAPSHOT_DAMAGE_DEALT] = signed_ratio(dealt, MAX_DAMAGE);
        output.global[index::SNAPSHOT_DAMAGE_TAKEN] = signed_ratio(taken, MAX_DAMAGE);
    }

    fn encode_history(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let current_tick = tracker.current().expect("snapshot was checked").tick;
        let summaries = tracker.history();
        for (sample_index, age) in HISTORY_AGES.iter().copied().enumerate() {
            let summary = summaries[sample_index];
            let target = current_tick.saturating_sub(age);
            let present = age == 0 || summary.tick <= target;
            if present {
                encode_history_sample(
                    &mut output.history[sample_index],
                    summary,
                    current_tick,
                    self.audit.enemy_scoreboard,
                );
            }
        }
    }

    fn encode_policy_history(
        &self,
        tick: u32,
        local: &LocalPolicyState,
        output: &mut FeatureFrame,
    ) {
        for (output_index, decision) in local.decisions().rev().enumerate() {
            let token = &mut output.policy_history[output_index];
            token[0] = 1.0;
            token[1] = unit_ratio(tick.saturating_sub(decision.tick), MAX_AGE);
            token[2] = 1.0;
            token[3] = category_token(decision.kind.index());
        }
    }

    fn encode_units(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        output: &mut FeatureFrame,
    ) {
        let origin = own_origin(tracker);
        for (index, candidate) in action_space.entity_candidates().iter().enumerate() {
            let track = tracker
                .entity(candidate.id())
                .expect("action candidate has a visible tracker record");
            output.units[index] = self.encode_unit_track(tracker, track, origin);
        }
    }

    fn encode_own_units(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let origin = own_origin(tracker);
        for (index, kind) in [UnitKind::Hero, UnitKind::Courier].into_iter().enumerate() {
            if let Some(track) = own_body_track(tracker, kind) {
                output.own_units[index] = self.encode_unit_track(tracker, track, origin);
            }
        }
    }

    fn encode_remembered_units(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let origin = own_origin(tracker);
        let mut count = 0usize;
        for track in tracker.entities() {
            if snapshot_visible(tracker, track) || is_own_body_track(tracker, track) {
                continue;
            }
            let token = self.encode_unit_track(tracker, track, origin);
            insert_sorted_token(&mut output.remembered_units, &mut count, token);
        }
    }

    fn encode_points(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        output: &mut FeatureFrame,
    ) {
        let origin = own_origin(tracker);
        for (index, point) in action_space.point_candidates().iter().enumerate() {
            output.points[index] = self.encode_point(tracker.team(), point, origin);
        }
    }

    fn encode_point(
        &self,
        team: Team,
        point: &PointCandidate,
        origin: Option<Vec2>,
    ) -> [f32; POINT_FEATURES] {
        use point_feature as index;
        let mut token = [0.0; POINT_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        token[index::POINTER_VALID] = 1.0;
        let position = self.canonical_position(team, point.position);
        token[index::POSITION_X] = coordinate_ratio(position.x.raw, self.extent_raw);
        token[index::POSITION_Y] = coordinate_ratio(position.y.raw, self.extent_raw);
        encode_point_origin(self, &mut token, team, point.position, origin);
        encode_point_source(&mut token, point.source);
        token[index::WALKABLE] = bool_feature(point.walkable);
        token[index::STANDING_TREE] = bool_feature(point.standing_tree);
        token[index::ALLIED_BUILDING] = bool_feature(point.allied_building);
        token
    }

    fn encode_unit_track(
        &self,
        tracker: &StateTracker,
        track: &crate::EntityTrack,
        origin: Option<Vec2>,
    ) -> [f32; UNIT_FEATURES] {
        let mut token = [0.0; UNIT_FEATURES];
        let unit = &track.unit;
        let tick = tracker.current().expect("snapshot was checked").tick;
        token[unit_feature::TOKEN_PRESENT] = 1.0;
        encode_relation(&mut token, unit_relation(tracker, unit));
        token[unit_feature::KIND_TOKEN] = unit_kind_token(unit.kind);
        encode_owner_relation(&mut token, owner_relation(tracker, unit));
        token[unit_feature::OBSERVATION_PRESENT] = 1.0;
        let visible = snapshot_visible(tracker, track);
        token[unit_feature::VISIBLE] = bool_feature(visible);
        token[unit_feature::REMEMBERED] = bool_feature(!visible);
        token[unit_feature::AGE] = unit_ratio(
            tick.saturating_sub(track.last_seen_tick),
            crate::HISTORY_TICKS,
        );
        self.encode_unit_geometry(&mut token, tracker.team(), unit, origin);
        self.encode_unit_motion(&mut token, tracker.team(), track);
        self.encode_unit_terrain(&mut token, unit.pos);
        encode_unit_resources(&mut token, track);
        encode_unit_combat(&mut token, unit);
        encode_unit_inventory(&mut token, unit);
        encode_unit_tactics(
            &mut token,
            unit,
            origin,
            tracker.own_hero(),
            tracker.metadata().tick_rate,
        );
        encode_statuses(&mut token, unit.statuses);
        encode_unit_recent(&mut token, tracker, track, tick);
        token
    }

    fn encode_unit_geometry(
        &self,
        token: &mut [f32; UNIT_FEATURES],
        team: Team,
        unit: &UnitView,
        origin: Option<Vec2>,
    ) {
        let position = self.canonical_position(team, unit.pos);
        token[unit_feature::POSITION_X] = coordinate_ratio(position.x.raw, self.extent_raw);
        token[unit_feature::POSITION_Y] = coordinate_ratio(position.y.raw, self.extent_raw);
        token[unit_feature::FACING] = facing_feature(team, unit.facing);
        token[unit_feature::RADIUS] = raw_distance_ratio(unit.radius.raw, self.extent_raw);
        if let Some(origin) = origin {
            token[unit_feature::ORIGIN_PRESENT] = 1.0;
            let delta = self.canonical_delta(team, unit.pos, origin);
            token[unit_feature::RELATIVE_X] = signed_raw_ratio(delta.0, self.extent_raw);
            token[unit_feature::RELATIVE_Y] = signed_raw_ratio(delta.1, self.extent_raw);
            let distance = delta.0.abs().max(delta.1.abs());
            token[unit_feature::DISTANCE] = raw_distance_ratio_i64(distance, self.extent_raw);
            if distance > 0 {
                token[unit_feature::DIRECTION_X] = delta.0 as f32 / distance as f32;
                token[unit_feature::DIRECTION_Y] = delta.1 as f32 / distance as f32;
            }
        }
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
        token[unit_feature::VELOCITY_PRESENT] = 1.0;
        let mut x = i64::from(velocity.delta.x.raw);
        let mut y = i64::from(velocity.delta.y.raw);
        if team == Team::Dire {
            x = -x;
            y = -y;
        }
        let divisor = i64::from(velocity.elapsed_ticks.max(1));
        token[unit_feature::VELOCITY_X] = signed_raw_ratio(x / divisor, self.extent_raw);
        token[unit_feature::VELOCITY_Y] = signed_raw_ratio(y / divisor, self.extent_raw);
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
        let hero_abilities = own_hero_abilities(tracker);
        let courier_abilities = own_courier_abilities(tracker);
        let tick = tracker.current().expect("snapshot was checked").tick;
        let hero_track = own_body_track(tracker, UnitKind::Hero);
        let courier_track = own_body_track(tracker, UnitKind::Courier);
        for slot in 0..SHADOW_FIEND_ABILITY_SLOTS {
            let ability = hero_abilities.and_then(|(abilities, _)| abilities.get(slot));
            let mut token = encode_ability_token(
                ControlledUnit::Hero,
                slot,
                ability,
                ability_legal(action_space, ControlledUnit::Hero, slot),
            );
            token[ability_feature::SCOREBOARD_KIT_SOURCE] =
                bool_feature(hero_abilities.is_some_and(|(_, kit)| kit));
            encode_ability_history(&mut token, tracker, hero_track, ability, tick);
            output.abilities[slot] = token;
        }
        for slot in 0..8 {
            let ability = courier_abilities.and_then(|(abilities, _)| abilities.get(slot));
            let mut token = encode_ability_token(
                ControlledUnit::Courier,
                slot,
                ability,
                ability_legal(action_space, ControlledUnit::Courier, slot),
            );
            encode_ability_history(&mut token, tracker, courier_track, ability, tick);
            output.abilities[SHADOW_FIEND_ABILITY_SLOTS + slot] = token;
        }
    }

    fn encode_items(
        &self,
        tracker: &StateTracker,
        action_space: &ActionSpace,
        readiness: &ItemReadiness,
        output: &mut FeatureFrame,
    ) {
        let tick = tracker.current().expect("snapshot was checked").tick;
        let hero_items = own_hero_items(tracker);
        let stash = tracker
            .own_player()
            .and_then(|player| player.stash.as_deref());
        let courier_items = own_courier_items(tracker);
        for slot in 0..9 {
            let item = hero_items
                .and_then(|(items, _)| items.get(slot))
                .copied()
                .flatten();
            output.items[slot] = encode_owned_item(
                tracker,
                action_space,
                readiness,
                tick,
                ControlledUnit::Hero,
                1,
                slot,
                item,
            );
            output.items[slot][item_feature::SCOREBOARD_KIT_SOURCE] =
                bool_feature(hero_items.is_some_and(|(_, kit)| kit));
        }
        for slot in 0..6 {
            let item = stash.and_then(|items| items.get(slot)).copied().flatten();
            output.items[9 + slot] = encode_owned_item(
                tracker,
                action_space,
                readiness,
                tick,
                ControlledUnit::Hero,
                2,
                9 + slot,
                item,
            );
        }
        for slot in 0..6 {
            let item = courier_items
                .and_then(|items| items.get(slot))
                .copied()
                .flatten();
            output.items[15 + slot] = encode_owned_item(
                tracker,
                action_space,
                readiness,
                tick,
                ControlledUnit::Courier,
                3,
                slot,
                item,
            );
        }
        encode_shop_items(tracker, action_space, &mut output.items);
    }

    fn encode_projectiles(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let current = tracker.current().expect("snapshot was checked");
        let origin = own_origin(tracker);
        let mut count = 0usize;
        for projectile in &current.projectiles {
            let history = self.projectile_observation(projectile.id);
            let token = self.projectile_token(tracker.team(), projectile, origin, history);
            insert_sorted_token(&mut output.projectiles, &mut count, token);
        }
    }

    fn projectile_token(
        &self,
        team: Team,
        projectile: &ProjectileView,
        origin: Option<Vec2>,
        history: Option<ProjectileObservation>,
    ) -> [f32; PROJECTILE_FEATURES] {
        use projectile_feature as index;
        let mut token = [0.0; PROJECTILE_FEATURES];
        token[index::TOKEN_PRESENT] = 1.0;
        encode_small_relation(
            &mut token[index::RELATION_START..index::RELATION_START + 4],
            team_relation(team, projectile.team),
        );
        token[index::ABILITY_PRESENT] = bool_feature(projectile.ability.is_some());
        token[index::ABILITY_TOKEN] = projectile.ability.map_or(0.0, ability_id_token);
        if let Some(origin) = origin {
            token[index::ORIGIN_PRESENT] = 1.0;
            let delta = self.canonical_delta(team, projectile.pos, origin);
            token[index::RELATIVE_X] = signed_raw_ratio(delta.0, self.extent_raw);
            token[index::RELATIVE_Y] = signed_raw_ratio(delta.1, self.extent_raw);
        }
        token[index::FACING] = facing_feature(team, projectile.facing);
        if let Some(history) = history {
            token[index::AGE_PRESENT] = 1.0;
            token[index::AGE] = unit_ratio(history.last_tick - history.first_tick, MAX_AGE);
            if let Some(previous_tick) = history.previous_tick {
                let elapsed = history.last_tick - previous_tick;
                let mut delta_x =
                    i64::from(history.position.x.raw) - i64::from(history.previous_position.x.raw);
                let mut delta_y =
                    i64::from(history.position.y.raw) - i64::from(history.previous_position.y.raw);
                if team == Team::Dire {
                    delta_x = -delta_x;
                    delta_y = -delta_y;
                }
                token[index::VELOCITY_PRESENT] = 1.0;
                token[index::VELOCITY_X] =
                    signed_raw_ratio(delta_x / i64::from(elapsed), self.extent_raw);
                token[index::VELOCITY_Y] =
                    signed_raw_ratio(delta_y / i64::from(elapsed), self.extent_raw);
                if let Some(origin) = origin {
                    let relative = self.canonical_delta(team, projectile.pos, origin);
                    token[index::CLOSEST_APPROACH_PRESENT] = 1.0;
                    token[index::CLOSEST_APPROACH] =
                        closest_approach_ratio(relative, delta_x, delta_y, self.extent_raw);
                }
            }
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
            self.prepare_path_distances(origin);
        }
        for (candidate_index, candidate) in action_space.loot_candidates().iter().enumerate() {
            let loot = current
                .loot
                .iter()
                .find(|loot| loot.id == candidate.id())
                .expect("action loot candidate belongs to current snapshot");
            use loot_feature as index;
            let mut token = [0.0; LOOT_FEATURES];
            token[index::TOKEN_PRESENT] = 1.0;
            token[index::ITEM_TOKEN] = item_id_token(loot.item);
            token[index::CHARGES_PRESENT] = bool_feature(loot.charges.is_some());
            token[index::CHARGES] = ratio(i64::from(loot.charges.unwrap_or(0)), 0, MAX_CHARGES);
            let duplicate_semantics = action_space.loot_candidates().iter().any(|other| {
                other.id() != candidate.id()
                    && other.item == candidate.item
                    && other.charges == candidate.charges
                    && other.position == candidate.position
            });
            if !duplicate_semantics && let Some(history) = self.loot_observation(loot.id) {
                token[index::VISIBLE_AGE_PRESENT] = 1.0;
                token[index::VISIBLE_AGE] =
                    unit_ratio(history.last_tick - history.first_tick, MAX_AGE);
            }
            if let Some(origin) = origin {
                token[index::ORIGIN_PRESENT] = 1.0;
                let delta = self.canonical_delta(tracker.team(), loot.pos, origin);
                token[index::RELATIVE_X] = signed_raw_ratio(delta.0, self.extent_raw);
                token[index::RELATIVE_Y] = signed_raw_ratio(delta.1, self.extent_raw);
                token[index::DIRECT_DISTANCE] =
                    raw_distance_ratio_i64(delta.0.abs().max(delta.1.abs()), self.extent_raw);
                if let Some(steps) = self.path_steps(loot.pos) {
                    token[index::PATH_DISTANCE_PRESENT] = 1.0;
                    token[index::PATH_DISTANCE] =
                        ratio(i64::from(steps), 0, self.path_distances.len() as i64);
                }
            }
            output.loot[candidate_index] = token;
        }
    }

    fn projectile_observation(&self, id: bota_proto::EntityId) -> Option<ProjectileObservation> {
        self.observation.projectiles[..self.observation.projectile_count]
            .iter()
            .flatten()
            .find(|history| history.id == id)
            .copied()
    }

    fn loot_observation(&self, id: bota_proto::EntityId) -> Option<LootObservation> {
        self.observation.loot[..self.observation.loot_count]
            .iter()
            .flatten()
            .find(|history| history.id == id)
            .copied()
    }

    fn encode_map(&self, tracker: &StateTracker, output: &mut FeatureFrame) {
        let Some(origin) = own_origin(tracker) else {
            return;
        };
        output.map[0] = 1.0;
        if let Some(cell) = self.cell(origin) {
            let terrain = self.terrain[cell];
            output.map[1] = bool_feature(terrain & 0x80 != 0);
            output.map[2] = bool_feature(terrain & 0x40 != 0);
            output.map[3] = ratio(i64::from(terrain & 0x3f), 0, 63);
            output.map[4] = bool_feature(self.opaque[cell]);
        }
        output.map[5] = bool_feature(self.tree_at_visible_context(tracker, origin));
        self.encode_landmark_distances(tracker, origin, &mut output.map);
        for (direction_index, direction) in MAP_DIRECTIONS.into_iter().enumerate() {
            let start = 16 + direction_index * 10;
            self.encode_map_ray(
                tracker,
                origin,
                direction,
                &mut output.map[start..start + 10],
            );
        }
    }

    fn encode_landmark_distances(
        &self,
        tracker: &StateTracker,
        origin: Vec2,
        output: &mut [f32; MAP_FEATURES],
    ) {
        let current = tracker.current().expect("snapshot was checked");
        for (pair, kind, own) in [
            (0usize, UnitKind::Fountain, true),
            (1, UnitKind::Fountain, false),
            (2, UnitKind::Tower, true),
            (3, UnitKind::Tower, false),
        ] {
            let relation_team = if own {
                tracker.team()
            } else {
                opposing(tracker.team())
            };
            let nearest = current
                .units
                .iter()
                .filter(|unit| unit.kind == kind && unit.team == relation_team)
                .map(|unit| origin.distance_squared(unit.pos))
                .min();
            if let Some(distance_squared) = nearest {
                output[6 + pair * 2] = 1.0;
                output[7 + pair * 2] = squared_distance_ratio(distance_squared, self.extent_raw);
            }
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

    fn prepare_path_distances(&mut self, origin: Vec2) {
        let Some(origin) = self.cell(origin) else {
            self.path_origin = None;
            return;
        };
        if self.path_origin == Some(origin) {
            return;
        }
        self.path_distances.fill(u32::MAX);
        self.path_queue.clear();
        self.path_origin = Some(origin);
        self.path_distances[origin] = 0;
        self.path_queue.push(origin);
        let mut cursor = 0usize;
        while cursor < self.path_queue.len() {
            let cell = self.path_queue[cursor];
            cursor += 1;
            self.visit_path_neighbors(cell);
        }
    }

    fn visit_path_neighbors(&mut self, cell: usize) {
        let x = cell % self.axis;
        let y = cell / self.axis;
        let next_distance = self.path_distances[cell].saturating_add(1);
        for (delta_x, delta_y) in MAP_DIRECTIONS {
            let delta_x = isize::try_from(delta_x).expect("map direction fits isize");
            let delta_y = isize::try_from(delta_y).expect("map direction fits isize");
            let Some(next_x) = x.checked_add_signed(delta_x) else {
                continue;
            };
            let Some(next_y) = y.checked_add_signed(delta_y) else {
                continue;
            };
            if next_x >= self.axis || next_y >= self.axis {
                continue;
            }
            let next = next_y * self.axis + next_x;
            if self.terrain[next] & 0x80 == 0
                || self.static_tree_at_cell(next)
                || self.path_distances[next] != u32::MAX
            {
                continue;
            }
            self.path_distances[next] = next_distance;
            self.path_queue.push(next);
        }
    }

    fn path_steps(&self, position: Vec2) -> Option<u32> {
        let distance = self.path_distances[self.cell(position)?];
        (distance != u32::MAX).then_some(distance)
    }

    fn static_tree_at_cell(&self, cell: usize) -> bool {
        let start = self
            .static_tree_index
            .partition_point(|(tree_cell, _)| *tree_cell < cell);
        self.static_tree_index
            .get(start)
            .is_some_and(|(tree_cell, _)| *tree_cell == cell)
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

fn next_observation_state(
    previous: &FeatureObservationState,
    tracker: &StateTracker,
    current: &bota_proto::WorldView,
) -> FeatureObservationState {
    let mut next = FeatureObservationState::new();
    next.tick = Some(current.tick);
    next.provenance = Some(tracker.provenance());
    for projectile in &current.projectiles {
        let prior = previous.projectiles[..previous.projectile_count]
            .iter()
            .flatten()
            .find(|entry| entry.id == projectile.id);
        let observation = prior.map_or(
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
        );
        next.projectiles[next.projectile_count] = Some(observation);
        next.projectile_count += 1;
    }
    for loot in &current.loot {
        let first_tick = previous.loot[..previous.loot_count]
            .iter()
            .flatten()
            .find(|entry| entry.id == loot.id)
            .map_or(current.tick, |entry| entry.first_tick);
        next.loot[next.loot_count] = Some(LootObservation {
            id: loot.id,
            first_tick,
            last_tick: current.tick,
        });
        next.loot_count += 1;
    }
    next
}

fn encode_score_advantages(global: &mut [f32; GLOBAL_FEATURES], summary: &crate::GlobalSummary) {
    use global_feature as index;
    global[index::KILL_ADVANTAGE] = signed_ratio(
        difference(summary.allied.kills, summary.enemy.kills),
        MAX_SCORE,
    );
    global[index::DEATH_ADVANTAGE] = signed_ratio(
        difference(summary.enemy.deaths, summary.allied.deaths),
        MAX_SCORE,
    );
    global[index::ASSIST_ADVANTAGE] = signed_ratio(
        difference(summary.allied.assists, summary.enemy.assists),
        MAX_SCORE,
    );
    global[index::XP_ADVANTAGE] =
        signed_ratio(summary.allied.xp.saturating_sub(summary.enemy.xp), MAX_XP);
    global[index::LEVEL_ADVANTAGE] = signed_ratio(
        difference(summary.allied.levels, summary.enemy.levels),
        MAX_SCORE,
    );
    global[index::LAST_HIT_ADVANTAGE] = signed_ratio(
        difference(summary.allied.last_hits, summary.enemy.last_hits),
        MAX_SCORE,
    );
    global[index::DENY_ADVANTAGE] = signed_ratio(
        difference(summary.allied.denies, summary.enemy.denies),
        MAX_SCORE,
    );
}

fn encode_assignment(local: &LocalPolicyState, global: &mut [f32; GLOBAL_FEATURES]) {
    use global_feature as index;
    let Some(assignment) = local.assignment() else {
        return;
    };
    if let Some(role) = assignment.role {
        global[index::ROLE_PRESENT] = 1.0;
        global[index::ROLE_TOKEN] = category_token(role_index(role));
    }
    if let Some(lane) = assignment.lane {
        global[index::LANE_PRESENT] = 1.0;
        global[index::LANE_TOKEN] = category_token(lane_index(lane));
    }
}

fn encode_local_global(tick: u32, local: &LocalPolicyState, global: &mut [f32; GLOBAL_FEATURES]) {
    use global_feature as index;
    if let Some(active) = local.active_order() {
        global[index::ACTIVE_ORDER_PRESENT] = 1.0;
        global[index::ACTIVE_ORDER_KIND] = category_token(active.kind.index());
        global[index::ACTIVE_ORDER_AGE] =
            unit_ratio(tick.saturating_sub(active.started_tick), MAX_AGE);
    }
    if let Some(decision) = local.decisions().next_back() {
        global[index::LAST_DECISION_PRESENT] = 1.0;
        global[index::TICKS_SINCE_DECISION] =
            unit_ratio(tick.saturating_sub(decision.tick), MAX_AGE);
    }
}

fn encode_own_score(own: &PlayerView, global: &mut [f32; GLOBAL_FEATURES]) {
    use global_feature as index;
    global[index::OWN_LEVEL] = ratio(i64::from(own.level), 0, MAX_LEVEL);
    global[index::OWN_XP] = signed_ratio(i64::from(own.xp), MAX_XP);
    global[index::OWN_KILLS] = ratio(i64::from(own.kills), 0, MAX_SCORE);
    global[index::OWN_DEATHS] = ratio(i64::from(own.deaths), 0, MAX_SCORE);
    global[index::OWN_ASSISTS] = ratio(i64::from(own.assists), 0, MAX_SCORE);
    global[index::OWN_LAST_HITS] = ratio(i64::from(own.last_hits), 0, MAX_SCORE);
    global[index::OWN_DENIES] = ratio(i64::from(own.denies), 0, MAX_SCORE);
}

fn encode_history_sample(
    output: &mut [f32; HISTORY_FEATURES],
    summary: crate::GlobalSummary,
    current_tick: u32,
    enemy_scoreboard: bool,
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
    output[index::OWN_LEVEL] = ratio(i64::from(summary.own_level), 0, MAX_LEVEL);
    output[index::OWN_GOLD] = signed_ratio(i64::from(summary.own_gold), MAX_GOLD);
    output[index::OWN_ALIVE] = bool_feature(summary.own_hp_present);
    output[index::RESPAWN_LEFT] = unit_ratio(summary.own_respawn_left, MAX_AGE);
    output[index::VISIBLE_ALLIED_UNITS] = ratio(i64::from(summary.visible_allied_units), 0, 256);
    output[index::VISIBLE_ENEMY_UNITS] = ratio(i64::from(summary.visible_enemy_units), 0, 256);
    output[index::ENEMY_SCOREBOARD_ENABLED] = bool_feature(enemy_scoreboard);
    if enemy_scoreboard {
        encode_history_advantages(output, summary);
    }
    output[index::ALLIED_STRUCTURE_HP] = signed_ratio(summary.allied_structure_hp, MAX_HP);
    output[index::ENEMY_STRUCTURE_HP] = signed_ratio(summary.enemy_structure_hp, MAX_HP);
    output[index::DESTROYED_STRUCTURES_PRESENT] =
        bool_feature(summary.destroyed_structures_present);
    let destroyed = summary
        .allied_structures_destroyed
        .saturating_add(summary.enemy_structures_destroyed);
    output[index::DESTROYED_STRUCTURES] = ratio(i64::from(destroyed), 0, MAX_STRUCTURE_COUNT);
}

fn encode_history_advantages(output: &mut [f32; HISTORY_FEATURES], summary: crate::GlobalSummary) {
    use history_feature as index;
    output[index::XP_ADVANTAGE] =
        signed_ratio(summary.allied.xp.saturating_sub(summary.enemy.xp), MAX_XP);
    output[index::LEVEL_ADVANTAGE] = signed_ratio(
        difference(summary.allied.levels, summary.enemy.levels),
        MAX_SCORE,
    );
    output[index::KILL_ADVANTAGE] = signed_ratio(
        difference(summary.allied.kills, summary.enemy.kills),
        MAX_SCORE,
    );
    output[index::DEATH_ADVANTAGE] = signed_ratio(
        difference(summary.enemy.deaths, summary.allied.deaths),
        MAX_SCORE,
    );
    output[index::ASSIST_ADVANTAGE] = signed_ratio(
        difference(summary.allied.assists, summary.enemy.assists),
        MAX_SCORE,
    );
    output[index::LAST_HIT_ADVANTAGE] = signed_ratio(
        difference(summary.allied.last_hits, summary.enemy.last_hits),
        MAX_SCORE,
    );
    output[index::DENY_ADVANTAGE] = signed_ratio(
        difference(summary.allied.denies, summary.enemy.denies),
        MAX_SCORE,
    );
}

fn encode_relation(token: &mut [f32; UNIT_FEATURES], relation: EntityRelation) {
    token[unit_feature::RELATION_START + relation_index(relation)] = 1.0;
}

fn encode_owner_relation(token: &mut [f32; UNIT_FEATURES], relation: Option<EntityRelation>) {
    if let Some(relation) = relation {
        token[unit_feature::OWNER_PRESENT] = 1.0;
        let index = relation_index(relation);
        if index < 3 {
            token[unit_feature::OWNER_START + index] = 1.0;
        }
    }
}

fn encode_point_origin(
    encoder: &FeatureEncoder,
    token: &mut [f32; POINT_FEATURES],
    team: Team,
    position: Vec2,
    origin: Option<Vec2>,
) {
    use point_feature as index;
    let Some(origin) = origin else {
        return;
    };
    token[index::ORIGIN_PRESENT] = 1.0;
    let delta = encoder.canonical_delta(team, position, origin);
    token[index::RELATIVE_X] = signed_raw_ratio(delta.0, encoder.extent_raw);
    token[index::RELATIVE_Y] = signed_raw_ratio(delta.1, encoder.extent_raw);
    let distance = delta.0.abs().max(delta.1.abs());
    token[index::DISTANCE] = raw_distance_ratio_i64(distance, encoder.extent_raw);
    if distance > 0 {
        token[index::DIRECTION_X] = delta.0 as f32 / distance as f32;
        token[index::DIRECTION_Y] = delta.1 as f32 / distance as f32;
    }
}

fn encode_point_source(token: &mut [f32; POINT_FEATURES], source: PointSource) {
    use point_feature as index;
    let (category, direction, radius, kind, relation) = point_source_semantics(source);
    token[index::SOURCE_TOKEN] = category_token(category);
    if let Some(direction) = direction {
        token[index::SOURCE_DIRECTION_PRESENT] = 1.0;
        token[index::SOURCE_DIRECTION_TOKEN] = category_token(point_direction_index(direction));
    }
    if let Some(radius) = radius {
        token[index::SOURCE_RADIUS_PRESENT] = 1.0;
        token[index::SOURCE_RADIUS] = ratio(i64::from(radius), 0, 1_200);
    }
    if let Some(kind) = kind {
        token[index::SOURCE_KIND_PRESENT] = 1.0;
        token[index::SOURCE_KIND_TOKEN] = unit_kind_token(kind);
    }
    if let Some(relation) = relation {
        token[index::SOURCE_RELATION_PRESENT] = 1.0;
        token[index::SOURCE_RELATION_START + relation_index(relation)] = 1.0;
    }
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

fn encode_unit_resources(token: &mut [f32; UNIT_FEATURES], track: &crate::EntityTrack) {
    let unit = &track.unit;
    if unit.max_hp > 0 {
        token[unit_feature::HP_PRESENT] = 1.0;
        token[unit_feature::HP_RATIO] = safe_fraction(unit.hp, unit.max_hp);
    }
    if unit.max_mana > 0 {
        token[unit_feature::MANA_PRESENT] = 1.0;
        token[unit_feature::MANA_RATIO] = safe_fraction(unit.mana, unit.max_mana);
    }
    if track.velocity.is_some() {
        token[unit_feature::HP_DELTA_PRESENT] = 1.0;
        token[unit_feature::HP_DELTA] = signed_ratio(track.hp_delta, MAX_HP);
        token[unit_feature::MANA_DELTA_PRESENT] = 1.0;
        token[unit_feature::MANA_DELTA] = signed_ratio(track.mana_delta, MAX_MANA);
    }
}

fn encode_unit_combat(token: &mut [f32; UNIT_FEATURES], unit: &UnitView) {
    token[unit_feature::ATTACK_DAMAGE] = signed_ratio(i64::from(unit.attack_damage), MAX_DAMAGE);
    token[unit_feature::ATTACK_RANGE] =
        raw_distance_ratio_i64(i64::from(unit.attack_range.raw), i64::from(Fixed::MAX.raw));
    token[unit_feature::ATTACK_INTERVAL] =
        ratio(i64::from(unit.attack_interval), 0, MAX_ATTACK_INTERVAL);
    token[unit_feature::ATTACK_SPEED] = signed_ratio(i64::from(unit.attack_speed), MAX_SPEED);
    token[unit_feature::MOVE_SPEED] = signed_ratio(
        i64::from(unit.move_speed.raw),
        MAX_SPEED << Fixed::FRAC_BITS,
    );
    token[unit_feature::ARMOR] = signed_ratio(i64::from(unit.armor.raw), MAX_ARMOR_RAW);
    token[unit_feature::MAGIC_RESISTANCE] =
        signed_ratio(i64::from(unit.magic_resist.raw), i64::from(Fixed::ONE.raw));
    token[unit_feature::VISION] =
        raw_distance_ratio_i64(i64::from(unit.vision_radius.raw), i64::from(Fixed::MAX.raw));
    token[unit_feature::TRUE_SIGHT] = raw_distance_ratio_i64(
        i64::from(unit.true_sight_radius.raw),
        i64::from(Fixed::MAX.raw),
    );
}

fn encode_unit_inventory(token: &mut [f32; UNIT_FEATURES], unit: &UnitView) {
    let free = unit.items.iter().filter(|slot| slot.is_none()).count();
    token[unit_feature::ITEM_SLOT_COUNT] = ratio(unit.items.len() as i64, 0, 9);
    token[unit_feature::FREE_ITEM_SLOTS] = ratio(free as i64, 0, 9);
    token[unit_feature::ITEM_CAPACITY_AVAILABLE] = bool_feature(free > 0);
}

fn encode_unit_tactics(
    token: &mut [f32; UNIT_FEATURES],
    unit: &UnitView,
    origin: Option<Vec2>,
    own_hero: Option<&UnitView>,
    tick_rate: u16,
) {
    let (Some(origin), Some(hero)) = (origin, own_hero) else {
        return;
    };
    let distance_squared = origin.distance_squared(unit.pos);
    if hero.attack_damage > 0 && unit.hp > 0 {
        let attacks = (i64::from(unit.hp) + i64::from(hero.attack_damage) - 1)
            / i64::from(hero.attack_damage);
        token[unit_feature::ATTACKS_TO_KILL_PRESENT] = 1.0;
        token[unit_feature::ATTACKS_TO_KILL] = ratio(attacks, 1, 100);
    }
    if hero.move_speed.raw > 0 {
        token[unit_feature::TIME_TO_REACH_PRESENT] = 1.0;
        let distance = maximum_axis_distance(origin, unit.pos);
        let ticks = distance
            .saturating_mul(i64::from(tick_rate))
            .checked_div(i64::from(hero.move_speed.raw))
            .unwrap_or(i64::MAX);
        token[unit_feature::TIME_TO_REACH] = ratio(ticks, 0, i64::from(MAX_AGE));
    }
    token[unit_feature::OWN_IN_ATTACK_RANGE] =
        bool_feature(distance_squared <= hero.attack_range.squared_raw());
    token[unit_feature::UNIT_IN_ATTACK_RANGE] =
        bool_feature(distance_squared <= unit.attack_range.squared_raw());
}

fn encode_unit_recent(
    token: &mut [f32; UNIT_FEATURES],
    tracker: &StateTracker,
    track: &crate::EntityTrack,
    current_tick: u32,
) {
    let taken = recent_damage(tracker, track.id, current_tick, false);
    if taken.is_some() {
        token[unit_feature::RECENT_DAMAGE_TAKEN] = 1.0;
    }
    if let Some((_, amount)) = recent_damage(tracker, track.id, current_tick, true) {
        token[unit_feature::RECENT_DAMAGE_DEALT_PRESENT] = 1.0;
        token[unit_feature::RECENT_DAMAGE_DEALT] = signed_ratio(i64::from(amount), MAX_DAMAGE);
    }
    let Some(attack_tick) = recent_possible_attack(tracker, track.id, current_tick) else {
        return;
    };
    let age = current_tick - attack_tick;
    if age > track.unit.attack_interval {
        return;
    }
    token[unit_feature::ATTACK_PHASE_PRESENT] = 1.0;
    token[unit_feature::ATTACK_PHASE] = unit_ratio(age, track.unit.attack_interval.max(1));
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
    token[ability_feature::LAST_CAST_AGE] = unit_ratio(current_tick - cast_tick, MAX_AGE);
}

fn encode_statuses(token: &mut [f32; UNIT_FEATURES], statuses: StatusFlags) {
    for (offset, flag) in [
        StatusFlags::STUNNED,
        StatusFlags::SILENCED,
        StatusFlags::ROOTED,
        StatusFlags::DISARMED,
        StatusFlags::SLOWED,
        StatusFlags::DOT,
        StatusFlags::INVISIBLE,
        StatusFlags::MAGIC_IMMUNE,
        StatusFlags::DEAD,
    ]
    .into_iter()
    .enumerate()
    {
        token[unit_feature::STATUS_START + offset] = bool_feature(statuses.bits & flag != 0);
    }
}

fn encode_ability_token(
    unit: ControlledUnit,
    slot: usize,
    ability: Option<&AbilityView>,
    legal: bool,
) -> [f32; ABILITY_FEATURES] {
    use ability_feature as index;
    let mut token = [0.0; ABILITY_FEATURES];
    token[index::TOKEN_PRESENT] = 1.0;
    token[index::BODY_TOKEN] = category_token(unit.index());
    token[index::SEMANTIC_SLOT_TOKEN] = category_token(slot);
    let Some(ability) = ability else {
        return token;
    };
    token[index::OBSERVATION_PRESENT] = 1.0;
    token[index::ID_PRESENT] = 1.0;
    token[index::ID_TOKEN] = ability_id_token(ability.id);
    token[index::LEVEL] = ratio(i64::from(ability.level), 0, 30);
    token[index::MAX_LEVEL] = ratio(i64::from(ability.max_level), 0, 30);
    token[index::COOLDOWN] = unit_ratio(ability.cooldown_left, MAX_COOLDOWN as u32);
    token[index::MANA_COST] = signed_ratio(i64::from(ability.mana_cost), MAX_MANA);
    token[index::RANGE] = signed_ratio(i64::from(ability.range), Fixed::MAX.to_int() as i64);
    token[index::AIM_TOKEN] = aim_token(ability.aim);
    token[index::PASSIVE] = bool_feature(ability.passive);
    token[index::TOGGLE_ON] = bool_feature(ability.on);
    token[index::CAN_LEVEL] = bool_feature(ability.can_level);
    token[index::LEGAL] = bool_feature(legal);
    token
}

#[allow(clippy::too_many_arguments)]
fn encode_owned_item(
    tracker: &StateTracker,
    action_space: &ActionSpace,
    readiness: &ItemReadiness,
    tick: u32,
    unit: ControlledUnit,
    location: usize,
    slot: usize,
    item: Option<ItemView>,
) -> [f32; ITEM_FEATURES] {
    let mut token = [0.0; ITEM_FEATURES];
    token[item_feature::TOKEN_PRESENT] = 1.0;
    token[item_feature::LOCATION_TOKEN] = category_token(location);
    token[item_feature::SLOT_TOKEN] = category_token(slot);
    let Some(item) = item else {
        return token;
    };
    encode_item_view(&mut token, tracker.shop(), item);
    let wire_slot = ItemSlot(u8::try_from(slot).unwrap_or(u8::MAX));
    if let Some(left) = readiness.inventory_mute_left(unit, wire_slot, tick) {
        token[item_feature::MUTE_REMAINING_PRESENT] = 1.0;
        token[item_feature::MUTE_REMAINING] = unit_ratio(left, MAX_COOLDOWN as u32);
        token[item_feature::MUTED] = bool_feature(left > 0);
    }
    if location != 2
        && let Some(left) = readiness.shared_wait_left(unit, item.id, tick)
    {
        token[item_feature::SHARED_WAIT_PRESENT] = 1.0;
        token[item_feature::SHARED_WAIT_REMAINING] = unit_ratio(left, MAX_COOLDOWN as u32);
    }
    if slot < 6 {
        token[item_feature::LEGAL] = bool_feature(item_legal(action_space, unit, slot));
    }
    token
}

fn encode_item_view(token: &mut [f32; ITEM_FEATURES], shop: &[ShopEntry], item: ItemView) {
    token[item_feature::ITEM_PRESENT] = 1.0;
    token[item_feature::ITEM_TOKEN] = item_id_token(item.id);
    token[item_feature::CHARGES_PRESENT] = bool_feature(item.charges.is_some());
    token[item_feature::CHARGES] = ratio(i64::from(item.charges.unwrap_or(0)), 0, MAX_CHARGES);
    token[item_feature::COOLDOWN] = unit_ratio(item.cooldown_left, MAX_COOLDOWN as u32);
    token[item_feature::AIM_PRESENT] = bool_feature(item.aim.is_some());
    token[item_feature::AIM_TOKEN] = item.aim.map_or(0.0, aim_token);
    token[item_feature::RANGE] = signed_ratio(i64::from(item.range), Fixed::MAX.to_int() as i64);
    token[item_feature::MANA_COST] = signed_ratio(i64::from(item.mana_cost), MAX_MANA);
    token[item_feature::ATTRIBUTE_PRESENT] = bool_feature(item.mode.is_some());
    token[item_feature::ATTRIBUTE_TOKEN] = item.mode.map_or(0.0, attribute_token);
    token[item_feature::FOR_SALE] = bool_feature(item.for_sale);
    if let Some(entry) = shop.iter().find(|entry| entry.id == item.id) {
        token[item_feature::VALUE_PRESENT] = 1.0;
        token[item_feature::VALUE] = signed_ratio(i64::from(entry.cost), MAX_ITEM_COST);
        token[item_feature::COMPOSITE] = bool_feature(!entry.components.is_empty());
    }
    token[item_feature::RECIPE_COMPONENT] =
        bool_feature(shop.iter().any(|entry| entry.components.contains(&item.id)));
}

fn encode_shop_items(
    tracker: &StateTracker,
    action_space: &ActionSpace,
    output: &mut [[f32; ITEM_FEATURES]; ITEM_FEATURE_TOKENS],
) {
    for (slot, candidate) in action_space.shop_candidates().iter().enumerate() {
        let mut token = [0.0; ITEM_FEATURES];
        token[item_feature::TOKEN_PRESENT] = 1.0;
        token[item_feature::LOCATION_TOKEN] = category_token(4);
        token[item_feature::SLOT_TOKEN] = category_token(slot);
        token[item_feature::ITEM_PRESENT] = 1.0;
        token[item_feature::ITEM_TOKEN] = item_id_token(candidate.item);
        token[item_feature::VALUE_PRESENT] = 1.0;
        token[item_feature::VALUE] = signed_ratio(i64::from(candidate.cost), MAX_ITEM_COST);
        token[item_feature::SHOP_CANDIDATE] = 1.0;
        token[item_feature::LEGAL] =
            bool_feature(action_space.buy_mask(ControlledUnit::Hero)[slot]);
        if let Some(entry) = tracker
            .shop()
            .iter()
            .find(|entry| entry.id == candidate.item)
        {
            token[item_feature::COMPOSITE] = bool_feature(!entry.components.is_empty());
        }
        output[OWN_ITEM_SLOTS + slot] = token;
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

fn own_asset_value(tracker: &StateTracker) -> i64 {
    let mut total = 0i64;
    for item in own_hero_items(tracker)
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
    {
        if let Some(entry) = tracker.shop().iter().find(|entry| entry.id == item.id) {
            total = total.saturating_add(i64::from(entry.cost));
        }
    }
    total
}

fn alive_hero_counts(players: &[PlayerView], own_team: Team) -> (i64, i64) {
    let mut allied = 0i64;
    let mut enemy = 0i64;
    for player in players {
        if player.unit.is_none() {
            continue;
        }
        if player.team == own_team {
            allied += 1;
        } else if player.team == opposing(own_team) {
            enemy += 1;
        }
    }
    (allied, enemy)
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

fn facing_feature(team: Team, angle: Angle) -> f32 {
    let brads = if team == Team::Dire {
        angle.brads.wrapping_add(1 << 15)
    } else {
        angle.brads
    };
    brads as f32 / u16::MAX as f32
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

fn encode_small_relation(output: &mut [f32], relation: EntityRelation) {
    output[relation_index(relation)] = 1.0;
}

fn unit_kind_token(kind: UnitKind) -> f32 {
    category_token(match kind {
        UnitKind::Hero => 0,
        UnitKind::CreepMelee => 1,
        UnitKind::CreepFlagbearer => 2,
        UnitKind::CreepRanged => 3,
        UnitKind::CreepSiege => 4,
        UnitKind::CreepNeutral => 5,
        UnitKind::Roshan => 6,
        UnitKind::Tower => 7,
        UnitKind::Ancient => 8,
        UnitKind::Fountain => 9,
        UnitKind::Ward => 10,
        UnitKind::Courier => 11,
    })
}

fn ability_id_token(id: AbilityId) -> f32 {
    category_token(match id.0 {
        8 => 0,
        9 => 1,
        10 => 2,
        11 => 3,
        12 => 4,
        13 => 5,
        14 => 6,
        15 => 7,
        16 => 8,
        17 => 9,
        18 => 10,
        other => 11 + usize::from(other),
    })
}

fn item_id_token(id: ItemId) -> f32 {
    category_token(usize::from(id.0))
}

fn aim_token(aim: Aim) -> f32 {
    category_token(match aim {
        Aim::Own => 0,
        Aim::Point => 1,
        Aim::Unit => 2,
        Aim::Tree => 3,
        Aim::Building => 4,
    })
}

fn attribute_token(attribute: Attribute) -> f32 {
    category_token(match attribute {
        Attribute::Strength => 0,
        Attribute::Agility => 1,
        Attribute::Intelligence => 2,
    })
}

const fn role_index(role: PolicyRole) -> usize {
    match role {
        PolicyRole::Carry => 0,
        PolicyRole::Mid => 1,
        PolicyRole::Offlane => 2,
        PolicyRole::Support => 3,
        PolicyRole::HardSupport => 4,
    }
}

const fn lane_index(lane: PolicyLane) -> usize {
    match lane {
        PolicyLane::Safe => 0,
        PolicyLane::Mid => 1,
        PolicyLane::Offlane => 2,
    }
}

fn category_token(index: usize) -> f32 {
    (index + 1) as f32
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

fn raw_distance_ratio(raw: i32, maximum: i64) -> f32 {
    raw_distance_ratio_i64(i64::from(raw), maximum)
}

fn raw_distance_ratio_i64(raw: i64, maximum: i64) -> f32 {
    ratio(raw, 0, maximum)
}

fn signed_raw_ratio(raw: i64, maximum: i64) -> f32 {
    signed_ratio(raw, maximum)
}

fn squared_distance_ratio(distance_squared: i64, extent_raw: i64) -> f32 {
    let maximum = extent_raw.saturating_mul(extent_raw).max(1);
    ratio(distance_squared, 0, maximum)
}

fn closest_approach_ratio(
    relative: (i64, i64),
    velocity_x: i64,
    velocity_y: i64,
    extent_raw: i64,
) -> f32 {
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
    let distance = closest_x.abs().max(closest_y.abs());
    (distance / extent_raw.max(1) as f32).clamp(0.0, 1.0)
}

fn maximum_axis_distance(left: Vec2, right: Vec2) -> i64 {
    let x = i64::from(left.x.raw).saturating_sub(i64::from(right.x.raw));
    let y = i64::from(left.y.raw).saturating_sub(i64::from(right.y.raw));
    x.abs().max(y.abs())
}

fn recent_before(event_tick: Option<u32>, current_tick: u32) -> bool {
    event_tick.is_some_and(|tick| tick < current_tick && current_tick - tick <= MAX_AGE)
}

fn unit_ratio(value: u32, maximum: u32) -> f32 {
    ratio(i64::from(value), 0, i64::from(maximum.max(1)))
}

fn signed_ratio(value: i64, maximum: i64) -> f32 {
    ratio(value, -maximum.abs().max(1), maximum.abs().max(1)) * 2.0 - 1.0
}

fn ratio(value: i64, minimum: i64, maximum: i64) -> f32 {
    assert!(minimum < maximum);
    let clamped = value.clamp(minimum, maximum);
    (clamped - minimum) as f32 / (maximum - minimum) as f32
}

fn bool_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn difference(left: u64, right: u64) -> i64 {
    i64::try_from(left)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(right).unwrap_or(i64::MAX))
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
