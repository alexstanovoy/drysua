use bota_proto::{
    AbilityId, AbilityView, Aim, Angle, Attribute, Attributes, DamageKind, EntityId, EventKind,
    Fixed, ItemId, ItemSlot, ItemView, LootView, MapId, MatchInfo, Order, Pick, PlayerView,
    ProjectileView, ShopEntry, SlotId, StatusFlags, Team, TickMode, UnitKind, UnitView, Vec2,
    WorldView,
};

use crate::{
    ABILITY_FEATURE_TOKENS, ABILITY_FEATURES, ActionKind, ActionSpace, FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION, FeatureAuditConfig, FeatureEncoder, FeatureFrame, GLOBAL_FEATURES,
    HISTORY_FEATURES, HISTORY_SAMPLES, ITEM_FEATURE_TOKENS, ITEM_FEATURES, IssuedOrder,
    ItemReadiness, LOOT_FEATURE_TOKENS, LOOT_FEATURES, LocalPolicyState, MAP_FEATURES,
    MAX_POLICY_HISTORY, OWN_UNIT_FEATURE_TOKENS, POINT_FEATURE_TOKENS, POINT_FEATURES,
    POLICY_HISTORY_FEATURES, PROJECTILE_FEATURE_TOKENS, PROJECTILE_FEATURES, PolicyLane,
    PolicyRole, REMEMBERED_UNIT_FEATURE_TOKENS, SHADOW_FIEND, StateTracker, UNIT_FEATURE_TOKENS,
    UNIT_FEATURES, ability_feature, global_feature, history_feature, item_feature, loot_feature,
    point_feature, projectile_feature, unit_feature,
};

const AXIS: u32 = 128;
const EXTENT: i32 = 8_192;
const HERO: EntityId = entity(10, 1);
const COURIER: EntityId = entity(11, 1);
const ENEMY: EntityId = entity(20, 1);

#[test]
fn feature_schema_dimensions_and_hash_are_stable() {
    assert_eq!(FEATURE_SCHEMA_VERSION, 4);
    assert_eq!(FEATURE_SCHEMA_HASH, 508_444_194_896_722_448);
    assert_eq!(GLOBAL_FEATURES, 64);
    assert_eq!((HISTORY_SAMPLES, HISTORY_FEATURES), (7, 24));
    assert_eq!((MAX_POLICY_HISTORY, POLICY_HISTORY_FEATURES), (16, 4));
    assert_eq!((UNIT_FEATURE_TOKENS, UNIT_FEATURES), (96, 69));
    assert_eq!((OWN_UNIT_FEATURE_TOKENS, UNIT_FEATURES), (2, 69));
    assert_eq!((REMEMBERED_UNIT_FEATURE_TOKENS, UNIT_FEATURES), (32, 69));
    assert_eq!((POINT_FEATURE_TOKENS, POINT_FEATURES), (48, 32));
    assert_eq!((ABILITY_FEATURE_TOKENS, ABILITY_FEATURES), (14, 24));
    assert_eq!((ITEM_FEATURE_TOKENS, ITEM_FEATURES), (85, 28));
    assert_eq!((PROJECTILE_FEATURE_TOKENS, PROJECTILE_FEATURES), (32, 20));
    assert_eq!((LOOT_FEATURE_TOKENS, LOOT_FEATURES), (16, 16));
    assert_eq!(MAP_FEATURES, 96);
}

#[test]
fn every_continuous_and_category_field_stays_inside_its_declared_range() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, u32::MAX));
    let mut local = LocalPolicyState::new(0);
    local
        .set_assignment(1, Some(PolicyRole::HardSupport), Some(PolicyLane::Offlane))
        .expect("assignment");
    local
        .set_active_order(1, Some(ActionKind::Learn))
        .expect("active order");
    local.note_decision(1, ActionKind::Learn).expect("decision");
    let frame = encode(&tracker, &local);

    assert_field_ranges("global", &frame.global, &[(10, 5), (12, 3), (32, 16)]);
    for (row, token) in frame.history.iter().enumerate() {
        assert_field_ranges(&format!("history[{row}]"), token, &[]);
    }
    for (row, token) in frame.policy_history.iter().enumerate() {
        assert_field_ranges(&format!("policy_history[{row}]"), token, &[(3, 16)]);
    }
    assert_matrix_field_ranges("units", &frame.units, &[(unit_feature::KIND_TOKEN, 12)]);
    assert_matrix_field_ranges(
        "own_units",
        &frame.own_units,
        &[(unit_feature::KIND_TOKEN, 12)],
    );
    assert_matrix_field_ranges(
        "remembered_units",
        &frame.remembered_units,
        &[(unit_feature::KIND_TOKEN, 12)],
    );
    assert_matrix_field_ranges(
        "points",
        &frame.points,
        &[
            (point_feature::SOURCE_TOKEN, 8),
            (point_feature::SOURCE_DIRECTION_TOKEN, 8),
            (point_feature::SOURCE_KIND_TOKEN, 12),
        ],
    );
    assert_matrix_field_ranges(
        "abilities",
        &frame.abilities,
        &[
            (ability_feature::BODY_TOKEN, 2),
            (ability_feature::SEMANTIC_SLOT_TOKEN, 8),
            (ability_feature::ID_TOKEN, 65_547),
            (ability_feature::AIM_TOKEN, 5),
        ],
    );
    assert_matrix_field_ranges(
        "items",
        &frame.items,
        &[
            (item_feature::LOCATION_TOKEN, 5),
            (item_feature::SLOT_TOKEN, 64),
            (item_feature::ITEM_TOKEN, 65_536),
            (item_feature::AIM_TOKEN, 5),
            (item_feature::ATTRIBUTE_TOKEN, 3),
        ],
    );
    assert_matrix_field_ranges(
        "projectiles",
        &frame.projectiles,
        &[(projectile_feature::ABILITY_TOKEN, 65_547)],
    );
    assert_matrix_field_ranges("loot", &frame.loot, &[(loot_feature::ITEM_TOKEN, 65_536)]);
    assert_field_ranges("map", &frame.map, &[]);
}

#[test]
fn dire_position_and_angle_transform_is_exact_at_raw_boundaries() {
    let maximum = ((i64::from(AXIS) * i64::from(crate::TERRAIN_CELL_SIZE)) << Fixed::FRAC_BITS) - 1;
    let low_radiant = boundary_frame(Team::Radiant, 0, 0);
    let low_dire = boundary_frame(
        Team::Dire,
        i32::try_from(maximum).expect("fixture extent"),
        1 << 15,
    );
    assert_eq!(low_radiant.units[0][unit_feature::POSITION_X], 0.0);
    assert_eq!(low_dire.units[0][unit_feature::POSITION_X], 0.0);
    assert_eq!(low_radiant.units[0][unit_feature::FACING], 0.0);
    assert_eq!(low_dire.units[0][unit_feature::FACING], 0.0);

    let high_radiant = boundary_frame(
        Team::Radiant,
        i32::try_from(maximum).expect("fixture extent"),
        u16::MAX,
    );
    let high_dire = boundary_frame(Team::Dire, 0, (1 << 15) - 1);
    assert_eq!(high_radiant.units[0][unit_feature::POSITION_X], 1.0);
    assert_eq!(high_dire.units[0][unit_feature::POSITION_X], 1.0);
    assert_eq!(high_radiant.units[0][unit_feature::FACING], 1.0);
    assert_eq!(high_dire.units[0][unit_feature::FACING], 1.0);
}

#[test]
fn radiant_and_dire_frames_are_canonically_equal_except_absolute_side() {
    let radiant = encoded_frame(Team::Radiant, world_view(Team::Radiant, 100));
    let dire = encoded_frame(Team::Dire, world_view(Team::Dire, 100));
    assert_eq!(radiant.global[global_feature::SIDE_RADIANT], 1.0);
    assert_eq!(dire.global[global_feature::SIDE_DIRE], 1.0);

    let mut radiant_canonical = radiant;
    let mut dire_canonical = dire;
    for index in [global_feature::SIDE_RADIANT, global_feature::SIDE_DIRE] {
        radiant_canonical.global[index] = 0.0;
        dire_canonical.global[index] = 0.0;
    }
    assert_eq!(radiant_canonical.global, dire_canonical.global);
    assert_eq!(radiant_canonical.history, dire_canonical.history);
    assert_eq!(
        radiant_canonical.policy_history,
        dire_canonical.policy_history
    );
    assert_matrix_eq("units", &radiant_canonical.units, &dire_canonical.units);
    assert_eq!(radiant_canonical.own_units, dire_canonical.own_units);
    assert_eq!(
        radiant_canonical.remembered_units,
        dire_canonical.remembered_units
    );
    assert_eq!(radiant_canonical.points, dire_canonical.points);
    assert_eq!(radiant_canonical.abilities, dire_canonical.abilities);
    assert_eq!(radiant_canonical.items, dire_canonical.items);
    assert_eq!(radiant_canonical.projectiles, dire_canonical.projectiles);
    assert_eq!(radiant_canonical.loot, dire_canonical.loot);
    assert_eq!(radiant_canonical.map, dire_canonical.map);
}

#[test]
fn mirrored_candidates_and_masks_match_at_unit_and_point_capacity() {
    let radiant_tracker = tracker_with_view(Team::Radiant, capacity_view(Team::Radiant));
    let dire_tracker = tracker_with_view(Team::Dire, capacity_view(Team::Dire));
    let radiant_space = ActionSpace::from_tracker(&radiant_tracker).expect("radiant space");
    let dire_space = ActionSpace::from_tracker(&dire_tracker).expect("dire space");
    let radiant = encode(&radiant_tracker, &LocalPolicyState::new(0));
    let dire = encode(&dire_tracker, &LocalPolicyState::new(0));

    assert_eq!(radiant_space.entity_candidates().len(), UNIT_FEATURE_TOKENS);
    assert_eq!(radiant_space.point_candidates().len(), POINT_FEATURE_TOKENS);
    assert_eq!(dire_space.entity_candidates().len(), UNIT_FEATURE_TOKENS);
    assert_eq!(dire_space.point_candidates().len(), POINT_FEATURE_TOKENS);
    assert_eq!(radiant.units, dire.units);
    assert_eq!(radiant.points, dire.points);
    assert_action_masks_equal(&radiant_space, &dire_space);

    let radiant_east = tactical_candidate(&radiant_space, crate::PointDirection::East, 200);
    let dire_east = tactical_candidate(&dire_space, crate::PointDirection::East, 200);
    assert_eq!(radiant_east.source, dire_east.source);
    assert_eq!(
        dire_east.position,
        canonical_position(Team::Dire, radiant_east.position)
    );
    assert_decoded_move_position(&radiant_space, radiant_east.position);
    assert_decoded_move_position(&dire_space, dire_east.position);
}

#[test]
fn feature_values_clamp_at_boundaries_and_remain_finite() {
    let mut view = world_view(Team::Radiant, u32::MAX);
    let hero_index = unit_index(&view, HERO);
    view.players[0].gold = Some(i32::MAX);
    view.players[0].xp = i32::MAX;
    view.units[hero_index].hp = i32::MIN;
    view.units[hero_index].max_hp = i32::MAX;
    view.units[hero_index].mana = i32::MAX;
    view.units[hero_index].max_mana = i32::MAX;
    let frame = encoded_frame(Team::Radiant, view);

    assert_eq!(frame.global[global_feature::TICK], 1.0);
    assert_eq!(frame.global[global_feature::OWN_GOLD], 1.0);
    assert!(
        frame
            .units
            .iter()
            .all(|token| { (0.0..=1.0).contains(&token[unit_feature::HP_RATIO]) })
    );
    assert!(frame.is_finite());
    assert!(all_values(&frame).all(|value| (-1.0..=65_547.0).contains(&value)));
}

#[test]
fn maximum_semantic_identifiers_stay_exact_and_inside_schema_ranges() {
    let mut view = world_view(Team::Radiant, 1);
    let hero_index = unit_index(&view, HERO);
    view.units[hero_index].abilities[0].id = AbilityId(u16::MAX);
    view.units[hero_index].items[0] = Some(item(ItemId(u16::MAX)));
    view.projectiles[0].ability = Some(AbilityId(u16::MAX));
    view.loot[0].item = ItemId(u16::MAX);
    let frame = encoded_frame(Team::Radiant, view);

    assert_eq!(frame.abilities[0][ability_feature::ID_TOKEN], 65_547.0);
    assert_eq!(frame.items[0][item_feature::ITEM_TOKEN], 65_536.0);
    assert_eq!(
        frame.projectiles[0][projectile_feature::ABILITY_TOKEN],
        65_547.0
    );
    assert_eq!(frame.loot[0][loot_feature::ITEM_TOKEN], 65_536.0);
}

#[test]
fn explicit_presence_distinguishes_missing_values_from_legitimate_zero() {
    let mut view = world_view(Team::Radiant, 1);
    let hero_index = unit_index(&view, HERO);
    view.units[hero_index].mana = 0;
    view.units[hero_index].max_mana = 100;
    let courier_index = unit_index(&view, COURIER);
    view.units[courier_index].mana = 0;
    view.units[courier_index].max_mana = 0;
    let frame = encoded_frame(Team::Radiant, view);

    assert_eq!(frame.units[0][unit_feature::MANA_PRESENT], 1.0);
    assert_eq!(frame.units[0][unit_feature::MANA_RATIO], 0.0);
    assert_eq!(frame.units[1][unit_feature::MANA_PRESENT], 0.0);
    assert_eq!(frame.units[1][unit_feature::MANA_RATIO], 0.0);
    assert_eq!(frame.abilities[0][ability_feature::LAST_CAST_PRESENT], 0.0);
    assert_eq!(frame.global[global_feature::ROLE_PRESENT], 0.0);
    assert_eq!(frame.global[global_feature::LANE_PRESENT], 0.0);
}

#[test]
fn resource_delta_presence_distinguishes_first_observation_from_measured_zero() {
    let info = match_info(Team::Radiant);
    let first_view = world_view(Team::Radiant, 1);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&first_view)
        .expect("first snapshot");
    let first = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(first.units[0][unit_feature::HP_DELTA_PRESENT], 0.0);
    assert_eq!(first.units[0][unit_feature::MANA_DELTA_PRESENT], 0.0);

    tracker
        .observe_snapshot(&WorldView {
            tick: 2,
            ..first_view
        })
        .expect("unchanged second snapshot");
    let second = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(second.units[0][unit_feature::HP_DELTA_PRESENT], 1.0);
    assert_eq!(second.units[0][unit_feature::HP_DELTA], 0.0);
    assert_eq!(second.units[0][unit_feature::MANA_DELTA_PRESENT], 1.0);
    assert_eq!(second.units[0][unit_feature::MANA_DELTA], 0.0);
}

#[test]
fn every_feature_group_contains_observable_data_and_semantic_tokens() {
    let frame = encoded_frame(Team::Radiant, world_view(Team::Radiant, 100));

    assert_eq!(frame.global[global_feature::MAP_ZERO], 1.0);
    assert_eq!(frame.history[HISTORY_SAMPLES - 1][0], 1.0);
    assert_eq!(frame.units[0][unit_feature::OBSERVATION_PRESENT], 1.0);
    assert_eq!(frame.abilities[0][ability_feature::ID_TOKEN], 6.0);
    assert_eq!(frame.items[0][item_feature::ITEM_TOKEN], 2.0);
    assert_eq!(frame.items[21][item_feature::SHOP_CANDIDATE], 1.0);
    assert_eq!(frame.projectiles[0][0], 1.0);
    assert_eq!(frame.loot[0][0], 1.0);
    assert_eq!(frame.map[0], 1.0);
}

#[test]
fn enemy_scoreboard_is_suppressed_by_default_and_explicitly_audited_when_enabled() {
    let mut changed = world_view(Team::Radiant, 10);
    changed.players[1].xp = 99_999;
    changed.players[1].level = 30;
    changed.players[1].kills = 100;
    changed.players[1].deaths = 100;
    changed.players[1].assists = 100;
    changed.players[1].last_hits = 500;
    changed.players[1].denies = 500;
    changed.players[1].unit = None;
    let baseline = encoded_frame(Team::Radiant, world_view(Team::Radiant, 10));
    let suppressed = encoded_frame(Team::Radiant, changed.clone());

    assert_eq!(baseline, suppressed);
    for index in 13..=19 {
        assert_eq!(baseline.global[index], suppressed.global[index]);
        assert_eq!(suppressed.global[index], 0.0);
    }
    assert_eq!(suppressed.global[global_feature::ENEMY_ALIVE_HEROES], 0.0);
    assert_eq!(
        suppressed.global[global_feature::ENEMY_SCOREBOARD_ENABLED],
        0.0
    );
    for sample in &suppressed.history {
        assert_eq!(sample[history_feature::ENEMY_SCOREBOARD_ENABLED], 0.0);
        for value in &sample[12..=18] {
            assert_eq!(*value, 0.0);
        }
    }

    let tracker = tracker_with_view(Team::Radiant, changed);
    let audited = encode_with_audit(
        &tracker,
        FeatureAuditConfig {
            enemy_scoreboard: true,
        },
    );
    assert_eq!(
        audited.global[global_feature::ENEMY_SCOREBOARD_ENABLED],
        1.0
    );
    assert_ne!(audited.global[global_feature::XP_ADVANTAGE], 0.0);
    assert_eq!(
        audited.history[HISTORY_SAMPLES - 1][history_feature::ENEMY_SCOREBOARD_ENABLED],
        1.0
    );
}

#[test]
fn unit_selection_is_deterministic_bounded_and_entity_id_independent() {
    let mut first = world_view(Team::Radiant, 1);
    for index in 0..150u32 {
        first
            .units
            .push(creep(entity(100 + index, 1), 2_500 + index as i32, 2_500));
    }
    first.units.sort_by_key(|unit| unit.id);
    let mut second = first.clone();
    reverse_entity_ids_and_generations(&mut second, 10_000, 77);

    let first = encoded_frame(Team::Radiant, first);
    let second = encoded_frame(Team::Radiant, second);

    assert_eq!(first.units, second.units);
    assert_eq!(present_unit_count(&first), UNIT_FEATURE_TOKENS);
}

#[test]
fn unit_and_loot_tokens_align_exactly_with_action_pointer_candidates() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("action space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let maximum_raw =
        ((i64::from(AXIS) * i64::from(crate::TERRAIN_CELL_SIZE)) << Fixed::FRAC_BITS) - 1;

    assert_eq!(space.entity_candidates()[0].kind, UnitKind::Hero);
    assert_eq!(
        space.entity_candidates()[0].relation,
        crate::EntityRelation::Own
    );
    assert_eq!(space.entity_candidates()[1].kind, UnitKind::Courier);
    assert_eq!(
        space.entity_candidates()[1].relation,
        crate::EntityRelation::Own
    );
    assert_eq!(present_unit_count(&frame), space.entity_candidates().len());
    for (index, candidate) in space.entity_candidates().iter().enumerate() {
        let token = &frame.units[index];
        assert_eq!(token[unit_feature::TOKEN_PRESENT], 1.0);
        assert_eq!(
            token[unit_feature::POSITION_X],
            candidate.position.x.raw as f32 / maximum_raw as f32
        );
        assert_eq!(
            token[unit_feature::RELATION_START + relation_offset(candidate.relation)],
            1.0
        );
    }
    assert_eq!(
        frame
            .loot
            .iter()
            .filter(|token| token[loot_feature::TOKEN_PRESENT] == 1.0)
            .count(),
        space.loot_candidates().len()
    );
    for (index, candidate) in space.loot_candidates().iter().enumerate() {
        assert_eq!(
            frame.loot[index][loot_feature::ITEM_TOKEN],
            f32::from(candidate.item.0) + 1.0
        );
        assert_eq!(
            frame.loot[index][loot_feature::CHARGES_PRESENT],
            if candidate.charges.is_some() {
                1.0
            } else {
                0.0
            }
        );
    }
}

#[test]
fn point_tokens_are_a_fixed_valid_prefix_aligned_with_action_candidates() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("action space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let maximum_raw =
        ((i64::from(AXIS) * i64::from(crate::TERRAIN_CELL_SIZE)) << Fixed::FRAC_BITS) - 1;

    assert!(space.point_candidates().len() <= POINT_FEATURE_TOKENS);
    for (index, candidate) in space.point_candidates().iter().enumerate() {
        let token = frame.points[index];
        assert_eq!(token[point_feature::TOKEN_PRESENT], 1.0);
        assert_eq!(token[point_feature::POINTER_VALID], 1.0);
        assert_eq!(
            token[point_feature::POSITION_X],
            candidate.position.x.raw as f32 / maximum_raw as f32
        );
        assert_eq!(
            token[point_feature::WALKABLE],
            candidate.walkable as u8 as f32
        );
        assert_eq!(
            token[point_feature::STANDING_TREE],
            candidate.standing_tree as u8 as f32
        );
        assert_eq!(
            token[point_feature::ALLIED_BUILDING],
            candidate.allied_building as u8 as f32
        );
    }
    for token in &frame.points[space.point_candidates().len()..] {
        assert_eq!(token[point_feature::TOKEN_PRESENT], 0.0);
        assert_eq!(token[point_feature::POINTER_VALID], 0.0);
    }
    let tactical = space
        .point_candidates()
        .iter()
        .position(|candidate| matches!(candidate.source, crate::PointSource::Tactical { .. }))
        .expect("tactical point");
    assert_eq!(frame.points[tactical][point_feature::SOURCE_TOKEN], 1.0);
    assert_eq!(
        frame.points[tactical][point_feature::SOURCE_DIRECTION_PRESENT],
        1.0
    );
    assert_eq!(
        frame.points[tactical][point_feature::SOURCE_RADIUS_PRESENT],
        1.0
    );
}

#[test]
fn missing_own_bodies_do_not_create_unaligned_entity_tokens() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("first snapshot");
    let mut hidden = world_view(Team::Radiant, 2);
    hidden.players[0].unit = None;
    hidden.players[0].respawn_left = 10;
    hidden.players[0].kit = Some(bota_proto::Kit {
        abilities: abilities(),
        items: hero_items(),
    });
    hidden
        .units
        .retain(|unit| !matches!(unit.id, HERO | COURIER));
    tracker.observe_snapshot(&hidden).expect("hidden bodies");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert!(frame.units.iter().all(|token| {
        token[unit_feature::TOKEN_PRESENT] == 0.0 || token[unit_feature::RELATION_START] == 0.0
    }));
    assert_eq!(
        frame.abilities[0][ability_feature::OBSERVATION_PRESENT],
        1.0
    );
    assert_eq!(
        frame.abilities[0][ability_feature::SCOREBOARD_KIT_SOURCE],
        1.0
    );
    assert_eq!(frame.items[0][item_feature::ITEM_PRESENT], 1.0);
    assert_eq!(frame.items[0][item_feature::SCOREBOARD_KIT_SOURCE], 1.0);
    assert!(
        frame.abilities[crate::SHADOW_FIEND_ABILITY_SLOTS..]
            .iter()
            .all(|token| token[ability_feature::OBSERVATION_PRESENT] == 0.0)
    );
    assert!(
        frame.items[15..21]
            .iter()
            .all(|token| token[item_feature::ITEM_PRESENT] == 0.0)
    );
}

#[test]
fn hidden_enemy_removal_enters_only_bounded_nontargetable_memory() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("visible enemy");
    let mut hidden = world_view(Team::Radiant, 2);
    hidden.units.retain(|unit| unit.id != ENEMY);
    tracker.observe_snapshot(&hidden).expect("hidden enemy");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert!(
        frame
            .units
            .iter()
            .all(|token| token[unit_feature::REMEMBERED] == 0.0)
    );
    assert_eq!(present_unit_count(&frame), 6);
    assert!(frame.remembered_units.iter().any(|token| {
        token[unit_feature::TOKEN_PRESENT] == 1.0
            && token[unit_feature::REMEMBERED] == 1.0
            && token[unit_feature::RELATION_START + 2] == 1.0
    }));
}

#[test]
fn own_body_memory_uses_fixed_slots_and_never_restores_body_payloads() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("visible bodies");
    let mut absent = world_view(Team::Radiant, 2);
    absent.players[0].unit = None;
    absent.players[0].kit = Some(bota_proto::Kit {
        abilities: abilities(),
        items: hero_items(),
    });
    absent
        .units
        .retain(|unit| !matches!(unit.id, HERO | COURIER));
    tracker.observe_snapshot(&absent).expect("absent bodies");

    let frame = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(frame.own_units[0][unit_feature::TOKEN_PRESENT], 1.0);
    assert_eq!(frame.own_units[0][unit_feature::REMEMBERED], 1.0);
    assert_eq!(frame.own_units[1][unit_feature::TOKEN_PRESENT], 1.0);
    assert_eq!(frame.own_units[1][unit_feature::REMEMBERED], 1.0);
    assert_eq!(
        frame.abilities[0][ability_feature::SCOREBOARD_KIT_SOURCE],
        1.0
    );
    assert!(
        frame.abilities[crate::SHADOW_FIEND_ABILITY_SLOTS..]
            .iter()
            .all(|token| token[ability_feature::OBSERVATION_PRESENT] == 0.0)
    );
    assert!(
        frame.items[15..21]
            .iter()
            .all(|token| token[item_feature::ITEM_PRESENT] == 0.0)
    );
}

#[test]
fn remembered_units_are_capacity_bounded_id_invariant_and_expire() {
    let info = match_info(Team::Radiant);
    let mut first_view = world_view(Team::Radiant, 1);
    for index in 0..40u32 {
        first_view.units.push(creep(
            entity(100 + index, 1),
            2_500 + i32::try_from(index).expect("small index"),
            2_500,
        ));
    }
    first_view.units.sort_by_key(|unit| unit.id);
    let mut remapped = first_view.clone();
    for unit in remapped.units.iter_mut().filter(|unit| unit.id.idx >= 100) {
        unit.id = entity(20_139 - unit.id.idx, 91 + unit.id.idx % 3);
    }
    remapped.units.sort_by_key(|unit| unit.id);
    let mut first = StateTracker::new(SlotId(0), &info).expect("first tracker");
    first.observe_snapshot(&first_view).expect("first visible");
    first
        .observe_snapshot(&world_view(Team::Radiant, 2))
        .expect("first hidden");
    let mut second = StateTracker::new(SlotId(0), &info).expect("second tracker");
    second
        .observe_snapshot(&remapped)
        .expect("remapped visible");
    second
        .observe_snapshot(&world_view(Team::Radiant, 2))
        .expect("remapped hidden");

    let first_frame = encode(&first, &LocalPolicyState::new(0));
    let second_frame = encode(&second, &LocalPolicyState::new(0));
    assert_eq!(first_frame.remembered_units, second_frame.remembered_units);
    assert_eq!(
        first_frame
            .remembered_units
            .iter()
            .filter(|token| token[unit_feature::TOKEN_PRESENT] == 1.0)
            .count(),
        REMEMBERED_UNIT_FEATURE_TOKENS
    );

    first
        .observe_snapshot(&world_view(Team::Radiant, 481))
        .expect("memory boundary");
    assert!(
        encode(&first, &LocalPolicyState::new(0))
            .remembered_units
            .iter()
            .any(|token| token[unit_feature::TOKEN_PRESENT] == 1.0)
    );
    first
        .observe_snapshot(&world_view(Team::Radiant, 482))
        .expect("memory expiry");
    assert!(
        encode(&first, &LocalPolicyState::new(0))
            .remembered_units
            .iter()
            .all(|token| token[unit_feature::TOKEN_PRESENT] == 0.0)
    );
}

#[test]
fn remembered_units_stay_id_invariant_after_more_than_tracker_capacity_were_seen() {
    let first = tracker_after_capacity_eviction(false);
    let remapped = tracker_after_capacity_eviction(true);

    let first_frame = encode(&first, &LocalPolicyState::new(0));
    let remapped_frame = encode(&remapped, &LocalPolicyState::new(0));

    assert_eq!(
        first_frame.remembered_units,
        remapped_frame.remembered_units
    );
    assert_eq!(
        first_frame
            .remembered_units
            .iter()
            .filter(|token| token[unit_feature::TOKEN_PRESENT] == 1.0)
            .count(),
        REMEMBERED_UNIT_FEATURE_TOKENS
    );
}

#[test]
fn remote_tree_changes_do_not_change_policy_features() {
    let mut baseline = world_view(Team::Radiant, 1);
    baseline.felled_trees.clear();
    let mut remote_change = baseline.clone();
    remote_change.felled_trees.push(1);

    let baseline = encoded_frame(Team::Radiant, baseline);
    let remote_change = encoded_frame(Team::Radiant, remote_change);

    assert_eq!(baseline, remote_change);
}

#[test]
fn same_tick_event_delivery_order_does_not_change_snapshot_features() {
    let info = match_info(Team::Radiant);
    let mut without_events = StateTracker::new(SlotId(0), &info).expect("tracker");
    without_events
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot");
    let mut with_events = StateTracker::new(SlotId(0), &info).expect("tracker");
    with_events
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot");
    with_events
        .observe_events(
            1,
            &[EventKind::Died {
                unit: ENEMY,
                killer: Some(HERO),
                denied: false,
                gold: 0,
            }],
        )
        .expect("event");

    let plain = encode(&without_events, &LocalPolicyState::new(0));
    let eventful = encode(&with_events, &LocalPolicyState::new(0));
    assert_eq!(plain, eventful);
}

#[test]
fn local_policy_history_is_bounded_and_rollback_reset_are_deterministic() {
    let mut local = LocalPolicyState::new(0);
    for tick in 5..=20 {
        local
            .note_decision(tick, ActionKind::MovePoint)
            .expect("ordered decision");
    }
    local
        .set_active_order(20, Some(ActionKind::AttackUnit))
        .expect("active order");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 25));
    let full = encode(&tracker, &local);
    assert_eq!(
        full.policy_history
            .iter()
            .filter(|token| token[0] == 1.0)
            .count(),
        16
    );
    assert_eq!(full.global[global_feature::ACTIVE_ORDER_PRESENT], 1.0);

    local
        .rollback(10)
        .expect("rollback within retained history");
    let rolled_back = encode(&tracker, &local);
    assert_eq!(
        rolled_back
            .policy_history
            .iter()
            .filter(|token| token[0] == 1.0)
            .count(),
        6
    );
    assert_eq!(
        rolled_back.global[global_feature::ACTIVE_ORDER_PRESENT],
        0.0
    );
    local.reset(25);
    let reset = encode(&tracker, &local);
    assert!(reset.policy_history.iter().all(|token| token[0] == 0.0));
    assert_eq!(reset.global[global_feature::LAST_DECISION_PRESENT], 0.0);
}

#[test]
fn local_policy_state_rejects_time_regression_with_exact_error() {
    let mut local = LocalPolicyState::new(10);
    local
        .note_decision(12, ActionKind::Continue)
        .expect("decision");
    let error = local
        .set_active_order(11, Some(ActionKind::MovePoint))
        .expect_err("older tick rejected");
    assert_eq!(
        error.to_string(),
        "local policy tick 11 is older than latest tick 12"
    );
}

#[test]
fn explicit_role_and_lane_assignment_is_versioned_local_state() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let mut local = LocalPolicyState::new(0);
    local
        .set_assignment(5, Some(PolicyRole::Mid), Some(PolicyLane::Mid))
        .expect("assignment");

    let assigned = encode(&tracker, &local);
    assert_eq!(assigned.global[global_feature::ROLE_PRESENT], 1.0);
    assert_eq!(assigned.global[global_feature::ROLE_TOKEN], 2.0);
    assert_eq!(assigned.global[global_feature::LANE_PRESENT], 1.0);
    assert_eq!(assigned.global[global_feature::LANE_TOKEN], 2.0);

    local.rollback(4).expect("assignment rollback");
    let rolled_back = encode(&tracker, &local);
    assert_eq!(rolled_back.global[global_feature::ROLE_PRESENT], 0.0);
    assert_eq!(rolled_back.global[global_feature::LANE_PRESENT], 0.0);
}

#[test]
fn local_policy_rollback_restores_replaced_order_and_assignment() {
    let mut local = LocalPolicyState::new(0);
    local
        .set_active_order(2, Some(ActionKind::MovePoint))
        .expect("first order");
    local
        .set_assignment(2, Some(PolicyRole::Carry), Some(PolicyLane::Safe))
        .expect("first assignment");
    local
        .set_active_order(4, Some(ActionKind::AttackUnit))
        .expect("replacement order");
    local
        .set_assignment(4, Some(PolicyRole::Mid), Some(PolicyLane::Mid))
        .expect("replacement assignment");

    local.rollback(3).expect("transition rollback");

    assert_eq!(
        local.active_order(),
        Some(crate::ActivePolicyOrder {
            started_tick: 2,
            kind: ActionKind::MovePoint,
        })
    );
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 5));
    let frame = encode(&tracker, &local);
    assert_eq!(frame.global[global_feature::ROLE_TOKEN], 1.0);
    assert_eq!(frame.global[global_feature::LANE_TOKEN], 1.0);
}

#[test]
fn local_policy_journals_restore_across_eviction_and_reset() {
    let mut local = LocalPolicyState::new(0);
    for tick in 1..=20 {
        local
            .set_active_order(tick, Some(ActionKind::MovePoint))
            .expect("active transition");
        local
            .set_assignment(tick, Some(PolicyRole::Carry), Some(PolicyLane::Safe))
            .expect("assignment transition");
    }

    local.rollback(4).expect("rollback to base boundary");
    assert_eq!(local.active_order().expect("evicted base").started_tick, 4);
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 5));
    let restored = encode(&tracker, &local);
    assert_eq!(restored.global[global_feature::ROLE_PRESENT], 1.0);

    local.reset(5);
    assert!(local.active_order().is_none());
    let reset = encode(&tracker, &local);
    assert_eq!(reset.global[global_feature::ROLE_PRESENT], 0.0);
}

#[test]
fn local_policy_rollback_before_eviction_horizon_is_exact_and_atomic() {
    let mut local = LocalPolicyState::new(0);
    for tick in 1..=20 {
        local
            .set_active_order(tick, Some(ActionKind::MovePoint))
            .expect("active transition");
        local
            .set_assignment(tick, Some(PolicyRole::Carry), Some(PolicyLane::Safe))
            .expect("assignment transition");
    }
    let unchanged = local;
    let error = local
        .rollback(3)
        .expect_err("tick before base is unsupported");
    assert_eq!(
        error.to_string(),
        "local policy rollback tick 3 is older than earliest supported tick 4"
    );
    assert_eq!(local, unchanged);

    local.rollback(4).expect("base boundary is retained");
    assert_eq!(local.active_order().expect("active base").started_tick, 4);
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 5));
    let frame = encode(&tracker, &local);
    assert_eq!(frame.global[global_feature::ROLE_TOKEN], 1.0);
    assert_eq!(frame.global[global_feature::LANE_TOKEN], 1.0);
}

#[test]
fn local_policy_interleaved_journals_use_the_strictest_exact_horizon() {
    let mut local = LocalPolicyState::new(0);
    for tick in 1..=17 {
        local
            .note_decision(tick, ActionKind::Continue)
            .expect("decision transition");
        local
            .set_active_order(tick, Some(ActionKind::MovePoint))
            .expect("active transition");
        local
            .set_assignment(tick, Some(PolicyRole::Carry), Some(PolicyLane::Safe))
            .expect("assignment transition");
    }

    let unchanged = local;
    let error = local
        .rollback(16)
        .expect_err("decision eviction makes tick sixteen inexact");
    assert_eq!(
        error.to_string(),
        "local policy rollback tick 16 is older than earliest supported tick 17"
    );
    assert_eq!(local, unchanged);

    local.rollback(17).expect("exact current boundary");
    assert_eq!(local.decisions().count(), MAX_POLICY_HISTORY);
    assert_eq!(local.active_order().expect("active order").started_tick, 17);
}

#[test]
fn history_uses_exact_ages_and_marks_dead_hero_resources_missing() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("live snapshot");
    let mut dead = world_view(Team::Radiant, 16);
    dead.players[0].unit = None;
    dead.players[0].respawn_left = 30;
    dead.players[0].kit = Some(bota_proto::Kit {
        abilities: abilities(),
        items: hero_items(),
    });
    dead.units.retain(|unit| unit.id != HERO);
    tracker.observe_snapshot(&dead).expect("dead snapshot");

    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let prior = HISTORY_SAMPLES - 2;
    let current = HISTORY_SAMPLES - 1;
    assert_eq!(frame.history[prior][history_feature::SAMPLE_PRESENT], 1.0);
    assert_eq!(frame.history[prior][history_feature::HP_PRESENT], 1.0);
    assert_eq!(frame.history[prior][history_feature::MANA_PRESENT], 1.0);
    assert_eq!(frame.history[current][history_feature::HP_PRESENT], 0.0);
    assert_eq!(frame.history[current][history_feature::HP_RATIO], 0.0);
    assert_eq!(frame.history[current][history_feature::MANA_PRESENT], 0.0);
    assert_eq!(frame.history[current][history_feature::MANA_RATIO], 0.0);
}

#[test]
fn structure_destruction_and_level_advantage_follow_snapshot_history() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("baseline");
    let mut destroyed = world_view(Team::Radiant, 2);
    destroyed.units.retain(|unit| unit.id != entity(33, 1));
    tracker
        .observe_snapshot(&destroyed)
        .expect("destroyed tower");

    let frame = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(
        frame.global[global_feature::DESTROYED_STRUCTURES_PRESENT],
        1.0
    );
    assert!(frame.global[global_feature::DESTROYED_STRUCTURES] > 0.0);
    assert_eq!(frame.global[global_feature::LEVEL_ADVANTAGE], 0.0);
    assert_eq!(
        frame.history[HISTORY_SAMPLES - 1][history_feature::DESTROYED_STRUCTURES_PRESENT],
        1.0
    );
}

#[test]
fn prior_tick_events_encode_combat_and_cast_history_but_same_tick_events_do_not() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot one");
    tracker
        .observe_events(
            1,
            &[EventKind::Damaged {
                source: Some(HERO),
                target: ENEMY,
                amount: 75,
                kind: DamageKind::Physical,
                crit: false,
            }],
        )
        .expect("damage event");
    tracker
        .observe_events(
            2,
            &[EventKind::AbilityCast {
                caster: HERO,
                ability: AbilityId(13),
            }],
        )
        .expect("cast event");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 3))
        .expect("snapshot three");

    let prior = encode(&tracker, &LocalPolicyState::new(0));
    assert!(prior.global[global_feature::SNAPSHOT_DAMAGE_DEALT] > 0.0);
    assert_eq!(
        prior.units[0][unit_feature::RECENT_DAMAGE_DEALT_PRESENT],
        1.0
    );
    assert_eq!(prior.units[0][unit_feature::ATTACK_PHASE_PRESENT], 1.0);
    assert_eq!(prior.abilities[0][ability_feature::LAST_CAST_PRESENT], 1.0);

    tracker
        .observe_events(
            3,
            &[EventKind::Damaged {
                source: Some(ENEMY),
                target: HERO,
                amount: 999,
                kind: DamageKind::Magical,
                crit: false,
            }],
        )
        .expect("same-tick event");
    let same_tick = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(
        prior.global[global_feature::SNAPSHOT_DAMAGE_TAKEN],
        same_tick.global[global_feature::SNAPSHOT_DAMAGE_TAKEN]
    );
}

#[test]
fn prior_combat_phase_is_encoded_for_enemy_and_ownerless_creep_tokens() {
    let info = match_info(Team::Radiant);
    let creep_id = entity(70, 3);
    let mut first_view = world_view(Team::Radiant, 1);
    first_view.units.push(creep(creep_id, 2_250, 2_000));
    first_view.units.sort_by_key(|unit| unit.id);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&first_view).expect("snapshot one");
    tracker
        .observe_events(
            2,
            &[
                EventKind::Damaged {
                    source: Some(ENEMY),
                    target: HERO,
                    amount: 31,
                    kind: DamageKind::Physical,
                    crit: false,
                },
                EventKind::Damaged {
                    source: Some(creep_id),
                    target: HERO,
                    amount: 19,
                    kind: DamageKind::Physical,
                    crit: false,
                },
            ],
        )
        .expect("prior attacks");
    let mut current = first_view;
    current.tick = 3;
    tracker.observe_snapshot(&current).expect("snapshot three");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let enemy_index = space
        .entity_candidates()
        .iter()
        .position(|candidate| candidate.position == current.units[unit_index(&current, ENEMY)].pos)
        .expect("enemy token");
    let creep_index = space
        .entity_candidates()
        .iter()
        .position(|candidate| {
            candidate.position == current.units[unit_index(&current, creep_id)].pos
        })
        .expect("creep token");

    assert_eq!(
        frame.units[enemy_index][unit_feature::ATTACK_PHASE_PRESENT],
        1.0
    );
    assert!(frame.units[enemy_index][unit_feature::RECENT_DAMAGE_DEALT] > 0.0);
    assert_eq!(
        frame.units[creep_index][unit_feature::ATTACK_PHASE_PRESENT],
        1.0
    );
    assert!(frame.units[creep_index][unit_feature::RECENT_DAMAGE_DEALT] > 0.0);
}

#[test]
fn same_tick_damage_and_cast_cannot_overwrite_strictly_prior_observations() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot one");
    tracker
        .observe_events(
            2,
            &[
                EventKind::Damaged {
                    source: Some(HERO),
                    target: ENEMY,
                    amount: 25,
                    kind: DamageKind::Magical,
                    crit: false,
                },
                EventKind::AbilityCast {
                    caster: HERO,
                    ability: AbilityId(13),
                },
            ],
        )
        .expect("prior events");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 3))
        .expect("snapshot three");
    let before = encode(&tracker, &LocalPolicyState::new(0));

    let same_tick = vec![
        EventKind::AbilityCast {
            caster: HERO,
            ability: AbilityId(13),
        },
        EventKind::Damaged {
            source: Some(HERO),
            target: ENEMY,
            amount: 999,
            kind: DamageKind::Magical,
            crit: false,
        },
    ];
    tracker
        .observe_events(3, &same_tick)
        .expect("same-tick events");
    let after = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(
        before.global[global_feature::SNAPSHOT_DAMAGE_DEALT],
        after.global[global_feature::SNAPSHOT_DAMAGE_DEALT]
    );
    assert_eq!(
        before.abilities[0][ability_feature::LAST_CAST_AGE],
        after.abilities[0][ability_feature::LAST_CAST_AGE]
    );
}

#[test]
fn ability_slots_use_independent_recent_cast_events() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot");
    tracker
        .observe_events(
            2,
            &[EventKind::AbilityCast {
                caster: HERO,
                ability: AbilityId(13),
            }],
        )
        .expect("first cast");
    tracker
        .observe_events(
            4,
            &[EventKind::AbilityCast {
                caster: HERO,
                ability: AbilityId(14),
            }],
        )
        .expect("second cast");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 5))
        .expect("encoding snapshot");

    let frame = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(frame.abilities[0][ability_feature::LAST_CAST_PRESENT], 1.0);
    assert_eq!(frame.abilities[1][ability_feature::LAST_CAST_PRESENT], 1.0);
    assert!(
        frame.abilities[0][ability_feature::LAST_CAST_AGE]
            > frame.abilities[1][ability_feature::LAST_CAST_AGE]
    );
}

#[test]
fn readiness_timers_encode_known_zero_active_and_boundary_values() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    let mut first = world_view(Team::Radiant, 1);
    let hero_index = unit_index(&first, HERO);
    first.units[hero_index].items[0] = Some(item(crate::TOWN_PORTAL_SCROLL));
    first.units[hero_index].items[7] = Some(item(ItemId(2)));
    tracker.observe_snapshot(&first).expect("first snapshot");
    let initial_space = ActionSpace::from_tracker(&tracker).expect("initial space");
    let mut readiness = ItemReadiness::new();
    readiness.note_sent(
        7,
        IssuedOrder {
            unit: None,
            order: Order::Swap {
                from: ItemSlot(7),
                to: ItemSlot(2),
            },
        },
        &initial_space,
    );
    let mut second = world_view(Team::Radiant, 2);
    let hero_index = unit_index(&second, HERO);
    second.units[hero_index].items[0] = Some(item(crate::TOWN_PORTAL_SCROLL));
    second.units[hero_index].items[2] = Some(item(ItemId(2)));
    tracker.observe_snapshot(&second).expect("second snapshot");
    let frame = encode_with_readiness(&tracker, &readiness);

    assert_eq!(frame.items[2][item_feature::MUTE_REMAINING_PRESENT], 1.0);
    assert_eq!(frame.items[2][item_feature::MUTED], 1.0);
    assert!(frame.items[2][item_feature::MUTE_REMAINING] > 0.0);
    assert_eq!(frame.items[0][item_feature::SHARED_WAIT_PRESENT], 1.0);
    assert_eq!(frame.items[0][item_feature::SHARED_WAIT_REMAINING], 0.0);
    assert_eq!(
        readiness.inventory_mute_left(crate::ControlledUnit::Hero, ItemSlot(2), 182),
        Some(0)
    );
}

#[test]
fn first_projectile_and_loot_observation_marks_age_and_velocity_missing() {
    let frame = encoded_frame(Team::Radiant, world_view(Team::Radiant, 100));

    assert_eq!(
        frame.projectiles[0][projectile_feature::VELOCITY_PRESENT],
        0.0
    );
    assert_eq!(frame.projectiles[0][projectile_feature::AGE_PRESENT], 1.0);
    assert_eq!(frame.projectiles[0][projectile_feature::AGE], 0.0);
    assert_eq!(
        frame.projectiles[0][projectile_feature::CLOSEST_APPROACH_PRESENT],
        0.0
    );
    assert_eq!(frame.loot[0][loot_feature::PATH_DISTANCE_PRESENT], 1.0);
    assert!(frame.loot[0][loot_feature::PATH_DISTANCE] > 0.0);
    assert_eq!(frame.loot[0][loot_feature::VISIBLE_AGE_PRESENT], 1.0);
    assert_eq!(frame.loot[0][loot_feature::VISIBLE_AGE], 0.0);
    assert!(frame.loot[0][loot_feature::DIRECT_DISTANCE] > 0.0);
}

#[test]
fn projectile_and_loot_history_handles_motion_zero_disappearance_and_generation() {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("first snapshot");
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("first observation");

    let mut moved = world_view(Team::Radiant, 3);
    moved.projectiles[0].pos.x += Fixed::from_int(64);
    tracker.observe_snapshot(&moved).expect("moved snapshot");
    encoder.observe(&tracker).expect("moved observation");
    let moved_frame = encode_with_encoder(&tracker, &mut encoder);
    assert_eq!(
        moved_frame.projectiles[0][projectile_feature::VELOCITY_PRESENT],
        1.0
    );
    assert!(moved_frame.projectiles[0][projectile_feature::VELOCITY_X] > 0.0);
    assert!(moved_frame.projectiles[0][projectile_feature::AGE] > 0.0);
    assert_eq!(
        moved_frame.projectiles[0][projectile_feature::CLOSEST_APPROACH_PRESENT],
        1.0
    );
    assert!(moved_frame.projectiles[0][projectile_feature::CLOSEST_APPROACH] > 0.0);
    assert!(moved_frame.loot[0][loot_feature::VISIBLE_AGE] > 0.0);

    let mut stopped = moved.clone();
    stopped.tick = 4;
    tracker
        .observe_snapshot(&stopped)
        .expect("stopped snapshot");
    encoder.observe(&tracker).expect("stopped observation");
    let stopped_frame = encode_with_encoder(&tracker, &mut encoder);
    assert_eq!(
        stopped_frame.projectiles[0][projectile_feature::VELOCITY_PRESENT],
        1.0
    );
    assert_eq!(
        stopped_frame.projectiles[0][projectile_feature::VELOCITY_X],
        0.0
    );

    let mut absent = stopped.clone();
    absent.tick = 5;
    absent.projectiles.clear();
    absent.loot.clear();
    tracker.observe_snapshot(&absent).expect("absent snapshot");
    encoder.observe(&tracker).expect("absent observation");
    let absent_frame = encode_with_encoder(&tracker, &mut encoder);
    assert_eq!(
        absent_frame.projectiles[0][projectile_feature::TOKEN_PRESENT],
        0.0
    );
    assert_eq!(absent_frame.loot[0][loot_feature::TOKEN_PRESENT], 0.0);

    let mut replacement = stopped;
    replacement.tick = 6;
    replacement.projectiles[0].id.generation += 1;
    replacement.loot[0].id.generation += 1;
    tracker
        .observe_snapshot(&replacement)
        .expect("replacement snapshot");
    encoder.observe(&tracker).expect("replacement observation");
    let replacement_frame = encode_with_encoder(&tracker, &mut encoder);
    assert_eq!(
        replacement_frame.projectiles[0][projectile_feature::VELOCITY_PRESENT],
        0.0
    );
    assert_eq!(
        replacement_frame.projectiles[0][projectile_feature::AGE],
        0.0
    );
    assert_eq!(replacement_frame.loot[0][loot_feature::VISIBLE_AGE], 0.0);
}

#[test]
fn cloned_tracker_branch_has_fresh_lineage_and_cannot_extend_observation_history() {
    let info = match_info(Team::Radiant);
    let first_view = world_view(Team::Radiant, 1);
    let mut first_tracker = StateTracker::new(SlotId(0), &info).expect("first tracker");
    first_tracker
        .observe_snapshot(&first_view)
        .expect("first snapshot");
    let mut encoder = FeatureEncoder::new(&first_tracker);
    encoder.observe(&first_tracker).expect("first observation");

    let mut second_tracker = first_tracker.clone();
    let mut second_view = world_view(Team::Radiant, 2);
    second_view.projectiles[0].pos.x += Fixed::from_int(32);
    second_tracker
        .observe_snapshot(&second_view)
        .expect("second snapshot");
    let error = encoder
        .observe(&second_tracker)
        .expect_err("clone branch has distinct lineage");
    assert_eq!(
        error.to_string(),
        "feature observation snapshot tick 2 does not extend its exact predecessor"
    );

    let restored = encode_with_encoder(&first_tracker, &mut encoder);

    let mut changed_info = info.clone();
    changed_info.map = MapId(1);
    let mut invalid = StateTracker::new(SlotId(0), &changed_info).expect("invalid tracker");
    invalid
        .observe_snapshot(&second_view)
        .expect("invalid snapshot");
    let error = encoder.observe(&invalid).expect_err("map mismatch");
    assert_eq!(
        error.to_string(),
        "feature encoder map context differs from tracker"
    );
    assert_eq!(encode_with_encoder(&first_tracker, &mut encoder), restored);

    encoder.reset();
    let space = ActionSpace::from_tracker(&first_tracker).expect("space");
    let mut output = FeatureFrame::new();
    let error = encoder
        .encode(
            &first_tracker,
            &space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("reset requires observation");
    assert_eq!(
        error.to_string(),
        "feature observation for snapshot tick 1 is required"
    );
}

#[test]
fn feature_observation_history_has_an_exact_fixed_capacity() {
    assert_eq!(crate::MAX_FEATURE_OBSERVATION_HISTORY, 16);
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("first snapshot");
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("first observation");
    for tick in 2..=18 {
        tracker
            .observe_snapshot(&world_view(Team::Radiant, tick))
            .expect("new snapshot");
        encoder.observe(&tracker).expect("new observation");
    }

    let latest_space = ActionSpace::from_tracker(&tracker).expect("latest space");
    let latest = encode_with_encoder(&tracker, &mut encoder);
    let error = encoder
        .rollback(2)
        .expect_err("rollback before retained observation horizon");
    assert_eq!(
        error.to_string(),
        "feature observation rollback tick 2 is older than earliest supported tick 3"
    );
    let mut after_error = FeatureFrame::new();
    encoder
        .encode(
            &tracker,
            &latest_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut after_error,
        )
        .expect("failed rollback preserves latest observation");
    assert_eq!(after_error, latest);
    encoder.rollback(3).expect("oldest retained observation");
}

#[test]
fn visible_local_tree_delta_changes_only_local_map_context() {
    let mut standing = world_view(Team::Radiant, 1);
    standing.felled_trees.clear();
    let mut felled = standing.clone();
    felled.felled_trees.push(0);

    let standing = encoded_frame(Team::Radiant, standing);
    let felled = encoded_frame(Team::Radiant, felled);
    assert_ne!(standing.map, felled.map);
    assert_eq!(standing.global, felled.global);
    assert_eq!(standing.units, felled.units);
}

#[test]
fn map_extent_uses_terrain_axis_and_accepts_multi_run_real_axis() {
    let mut info = match_info(Team::Radiant);
    info.terrain_cells = 288;
    info.terrain_rle = terrain_rle(288, 0x80);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let expected = 2_000.0 / (288.0 * crate::TERRAIN_CELL_SIZE as f32);

    assert_eq!(frame.units[0][unit_feature::POSITION_X], expected);
    assert!(frame.map[0] > 0.0);
}

#[test]
fn feature_encoder_rejects_mismatched_inputs_with_exact_errors() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut output = FeatureFrame::new();
    output.global[0] = 0.25;
    let unchanged = output.clone();
    let mut local_ahead = LocalPolicyState::new(11);
    local_ahead
        .note_decision(11, ActionKind::Continue)
        .expect("local decision");
    let error = encoder
        .encode(
            &tracker,
            &space,
            &ItemReadiness::new(),
            &local_ahead,
            &mut output,
        )
        .expect_err("local state ahead");
    assert_eq!(
        error.to_string(),
        "local policy tick 11 is newer than snapshot tick 10"
    );
    assert_eq!(output, unchanged);

    let mut changed = world_view(Team::Radiant, 10);
    changed.players[0].gold = Some(2_000);
    let changed_tracker = tracker_with_view(Team::Radiant, changed);
    let error = encoder
        .encode(
            &changed_tracker,
            &space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("space mismatch");
    assert_eq!(
        error.to_string(),
        "feature action space belongs to a different snapshot"
    );

    assert_static_and_slot_provenance(&tracker, &mut encoder, &mut output);
}

#[test]
fn feature_observation_is_bound_to_the_exact_same_tick_snapshot_atomically() {
    let first = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let mut changed_view = world_view(Team::Radiant, 10);
    changed_view.projectiles[0].pos.x += Fixed::from_int(1);
    let changed = tracker_with_view(Team::Radiant, changed_view);
    let changed_space = ActionSpace::from_tracker(&changed).expect("changed space");
    let mut encoder = FeatureEncoder::new(&first);
    encoder.observe(&first).expect("first observation");
    let mut output = FeatureFrame::new();
    output.global[0] = 0.5;
    let unchanged = output.clone();

    let error = encoder
        .encode(
            &changed,
            &changed_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("same tick does not establish provenance");
    assert_eq!(
        error.to_string(),
        "feature observation belongs to a different snapshot at tick 10"
    );
    assert_eq!(output, unchanged);

    let first_space = ActionSpace::from_tracker(&first).expect("first space");
    encoder
        .encode(
            &first,
            &first_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect("failed encode preserves observation");
}

#[test]
fn feature_observation_rejects_a_different_exact_predecessor_atomically() {
    let info = match_info(Team::Radiant);
    let first_view = world_view(Team::Radiant, 1);
    let mut first = StateTracker::new(SlotId(0), &info).expect("first tracker");
    first.observe_snapshot(&first_view).expect("first snapshot");
    let mut encoder = FeatureEncoder::new(&first);
    encoder.observe(&first).expect("first observation");

    let mut different_previous = first_view;
    different_previous.projectiles[0].pos.x += Fixed::from_int(64);
    let mut branch = StateTracker::new(SlotId(0), &info).expect("branch tracker");
    branch
        .observe_snapshot(&different_previous)
        .expect("different predecessor");
    branch
        .observe_snapshot(&world_view(Team::Radiant, 2))
        .expect("branch current snapshot");

    let error = encoder
        .observe(&branch)
        .expect_err("different predecessor cannot extend observation history");
    assert_eq!(
        error.to_string(),
        "feature observation snapshot tick 2 does not extend its exact predecessor"
    );
    let first_space = ActionSpace::from_tracker(&first).expect("first space");
    let mut output = FeatureFrame::new();
    encoder
        .encode(
            &first,
            &first_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect("failed branch observation preserves predecessor");
}

#[test]
fn feature_observation_rejects_reconstructed_equal_tracker_branch_by_lineage() {
    let info = match_info(Team::Radiant);
    let first_view = world_view(Team::Radiant, 1);
    let mut original = StateTracker::new(SlotId(0), &info).expect("original tracker");
    original
        .observe_snapshot(&first_view)
        .expect("original predecessor");
    let mut encoder = FeatureEncoder::new(&original);
    encoder.observe(&original).expect("original observation");

    let mut branch = StateTracker::new(SlotId(0), &info).expect("branch tracker");
    branch
        .observe_snapshot(&first_view)
        .expect("equal predecessor");
    branch
        .observe_snapshot(&world_view(Team::Radiant, 2))
        .expect("branch successor");
    let error = encoder
        .observe(&branch)
        .expect_err("equal reconstructed predecessor has different lineage");
    assert_eq!(
        error.to_string(),
        "feature observation snapshot tick 2 does not extend its exact predecessor"
    );
}

#[test]
fn match_and_tracker_lineage_never_change_policy_values() {
    let first = encoded_frame(Team::Radiant, world_view(Team::Radiant, 10));
    let second = encoded_frame(Team::Radiant, world_view(Team::Radiant, 10));
    assert_eq!(first, second);

    let mut changed_info = match_info(Team::Radiant);
    changed_info.match_id = u64::MAX;
    let mut tracker = StateTracker::new(SlotId(0), &changed_info).expect("changed match tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 10))
        .expect("changed match snapshot");
    assert_eq!(first, encode(&tracker, &LocalPolicyState::new(0)));
}

#[test]
fn allied_item_capacity_keeps_entity_tokens_and_put_masks_id_invariant() {
    let target_position = Vec2::from_ints(2_100, 2_200);
    let mut first_view = world_view(Team::Radiant, 1);
    let hero_index = unit_index(&first_view, HERO);
    first_view.units[hero_index].items[0] = Some(item(ItemId(2)));
    let mut available = courier(Team::Radiant, target_position);
    available.id = entity(80, 1);
    available.owner = None;
    available.items = vec![None; 6];
    let mut full = available.clone();
    full.id = entity(81, 1);
    full.items = vec![Some(item(ItemId(2))); 6];
    first_view.units.extend([available, full]);
    first_view.units.sort_by_key(|unit| unit.id);
    let mut remapped = first_view.clone();
    reverse_entity_ids_and_generations(&mut remapped, 30_000, 44);

    let first_tracker = tracker_with_view(Team::Radiant, first_view);
    let second_tracker = tracker_with_view(Team::Radiant, remapped);
    let first_space = ActionSpace::from_tracker(&first_tracker).expect("first space");
    let second_space = ActionSpace::from_tracker(&second_tracker).expect("second space");
    let first_frame = encode(&first_tracker, &LocalPolicyState::new(0));
    let second_frame = encode(&second_tracker, &LocalPolicyState::new(0));

    assert_eq!(first_frame.units, second_frame.units);
    assert_eq!(
        first_space
            .put_entity_target_mask(crate::ControlledUnit::Hero, ItemSlot(0))
            .expect("first put mask"),
        second_space
            .put_entity_target_mask(crate::ControlledUnit::Hero, ItemSlot(0))
            .expect("second put mask")
    );
    let available_index = first_frame
        .units
        .iter()
        .position(|token| {
            token[unit_feature::KIND_TOKEN] == 12.0
                && token[unit_feature::RELATION_START + 1] == 1.0
                && token[unit_feature::ITEM_CAPACITY_AVAILABLE] == 1.0
        })
        .expect("available target");
    let full_index = first_frame
        .units
        .iter()
        .enumerate()
        .find(|(index, token)| {
            *index != available_index
                && token[unit_feature::TOKEN_PRESENT] == 1.0
                && token[unit_feature::KIND_TOKEN] == 12.0
                && token[unit_feature::RELATION_START + 1] == 1.0
                && token[unit_feature::ITEM_CAPACITY_AVAILABLE] == 0.0
        })
        .map(|(index, _)| index)
        .expect("full target");
    assert_eq!(
        first_frame.units[available_index][unit_feature::ITEM_CAPACITY_AVAILABLE],
        1.0
    );
    assert_eq!(
        first_frame.units[full_index][unit_feature::ITEM_CAPACITY_AVAILABLE],
        0.0
    );
    assert!(
        first_space
            .put_entity_target_mask(crate::ControlledUnit::Hero, ItemSlot(0))
            .expect("put mask")[available_index]
    );
    assert!(
        !first_space
            .put_entity_target_mask(crate::ControlledUnit::Hero, ItemSlot(0))
            .expect("put mask")[full_index]
    );
}

fn assert_static_and_slot_provenance(
    tracker: &StateTracker,
    encoder: &mut FeatureEncoder,
    output: &mut FeatureFrame,
) {
    let mut changed_info = match_info(Team::Radiant);
    changed_info.trees.push(Vec2::from_ints(3_000, 3_000));
    let mut changed_static = StateTracker::new(SlotId(0), &changed_info).expect("static tracker");
    changed_static
        .observe_snapshot(&world_view(Team::Radiant, 10))
        .expect("static snapshot");
    let static_space = ActionSpace::from_tracker(&changed_static).expect("static space");
    let error = encoder
        .encode(
            tracker,
            &static_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            output,
        )
        .expect_err("static provenance mismatch");
    assert_eq!(
        error.to_string(),
        "feature action space belongs to a different snapshot"
    );

    let slot_one_info = match_info(Team::Radiant);
    let mut slot_one_view = world_view(Team::Dire, 10);
    slot_one_view.players[0].slot = SlotId(1);
    slot_one_view.players[1].slot = SlotId(0);
    slot_one_view.players.sort_by_key(|player| player.slot);
    for unit in &mut slot_one_view.units {
        if unit.id == HERO {
            unit.owner = Some(SlotId(1));
        } else if unit.id == ENEMY {
            unit.owner = Some(SlotId(0));
        }
    }
    let mut dire = StateTracker::new(SlotId(1), &slot_one_info).expect("slot one tracker");
    dire.observe_snapshot(&slot_one_view)
        .expect("slot one snapshot");
    let dire_space = ActionSpace::from_tracker(&dire).expect("other slot space");
    let error = encoder
        .encode(
            tracker,
            &dire_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            output,
        )
        .expect_err("slot provenance mismatch");
    assert_eq!(
        error.to_string(),
        "feature action space belongs to a different snapshot"
    );
}

#[test]
fn feature_encoder_reports_snapshot_tick_map_and_readiness_errors_exactly() {
    let info = match_info(Team::Radiant);
    let empty = StateTracker::new(SlotId(0), &info).expect("empty tracker");
    let full = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&full).expect("space");
    let mut output = FeatureFrame::new();
    let error = FeatureEncoder::new(&empty)
        .encode(
            &empty,
            &space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("snapshot required");
    assert_eq!(error.to_string(), "feature encoding requires a snapshot");

    let mut newer = full;
    newer
        .observe_snapshot(&world_view(Team::Radiant, 11))
        .expect("newer snapshot");
    let error = FeatureEncoder::new(&newer)
        .encode(
            &newer,
            &space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("tick mismatch");
    assert_eq!(
        error.to_string(),
        "feature snapshot tick 11 differs from action-space tick 10"
    );

    assert_map_and_readiness_errors(&mut output);
}

#[test]
fn readiness_provenance_covers_bounded_rejection_journal_not_only_current_deadline() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let baseline = ActionSpace::from_tracker(&tracker).expect("baseline space");
    let issued = IssuedOrder {
        unit: None,
        order: Order::Swap {
            from: ItemSlot(7),
            to: ItemSlot(2),
        },
    };
    let mut first = ItemReadiness::new();
    first.note_sent(1, issued, &baseline);
    let mut second = ItemReadiness::new();
    second.note_sent(2, issued, &baseline);
    assert_eq!(
        first.inventory_mute_left(crate::ControlledUnit::Hero, ItemSlot(2), 11),
        second.inventory_mute_left(crate::ControlledUnit::Hero, ItemSlot(2), 11)
    );

    let space = ActionSpace::from_tracker_with_readiness(&tracker, &first).expect("first space");
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    let mut output = FeatureFrame::new();
    let error = encoder
        .encode(
            &tracker,
            &space,
            &second,
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("journal provenance mismatch");
    assert_eq!(
        error.to_string(),
        "feature item readiness differs from action space"
    );
}

fn assert_map_and_readiness_errors(output: &mut FeatureFrame) {
    let base = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let mut encoder = FeatureEncoder::new(&base);
    encoder.observe(&base).expect("observation");
    let mut changed_info = match_info(Team::Radiant);
    changed_info.map = MapId(1);
    let mut changed = StateTracker::new(SlotId(0), &changed_info).expect("map tracker");
    changed
        .observe_snapshot(&world_view(Team::Radiant, 10))
        .expect("map snapshot");
    let changed_space = ActionSpace::from_tracker(&changed).expect("map space");
    let error = encoder
        .encode(
            &changed,
            &changed_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            output,
        )
        .expect_err("map mismatch");
    assert_eq!(
        error.to_string(),
        "feature encoder map context differs from tracker"
    );

    let space = ActionSpace::from_tracker(&base).expect("space");
    let mut readiness = ItemReadiness::new();
    readiness.note_sent(
        1,
        IssuedOrder {
            unit: None,
            order: Order::Swap {
                from: ItemSlot(7),
                to: ItemSlot(2),
            },
        },
        &space,
    );
    let error = encoder
        .encode(&base, &space, &readiness, &LocalPolicyState::new(0), output)
        .expect_err("readiness mismatch");
    assert_eq!(
        error.to_string(),
        "feature item readiness differs from action space"
    );
}

fn assert_action_masks_equal(first: &ActionSpace, second: &ActionSpace) {
    assert_eq!(first.kind_mask(), second.kind_mask());
    for unit in [crate::ControlledUnit::Hero, crate::ControlledUnit::Courier] {
        for kind in ActionKind::ALL {
            assert_eq!(
                first.controlled_unit_mask(kind),
                second.controlled_unit_mask(kind)
            );
        }
        assert_eq!(
            first.follow_entity_mask(unit),
            second.follow_entity_mask(unit)
        );
        assert_eq!(
            first.attack_entity_mask(unit),
            second.attack_entity_mask(unit)
        );
        assert_eq!(first.move_point_mask(unit), second.move_point_mask(unit));
        assert_eq!(
            first.attack_move_point_mask(unit),
            second.attack_move_point_mask(unit)
        );
        assert_eq!(first.take_mask(unit), second.take_mask(unit));
        assert_eq!(first.buy_mask(unit), second.buy_mask(unit));
        assert_eq!(first.sell_slot_mask(unit), second.sell_slot_mask(unit));
        for slot in 0..8 {
            let slot = bota_proto::AbilitySlot(slot);
            assert_eq!(
                first.cast_target_mask(unit, slot),
                second.cast_target_mask(unit, slot)
            );
        }
        for slot in 0..9 {
            let slot = ItemSlot(slot);
            assert_eq!(
                first.use_target_mask(unit, slot),
                second.use_target_mask(unit, slot)
            );
            assert_eq!(
                first.put_point_target_mask(unit, slot),
                second.put_point_target_mask(unit, slot)
            );
            assert_eq!(
                first.put_entity_target_mask(unit, slot),
                second.put_entity_target_mask(unit, slot)
            );
            assert_eq!(
                first.swap_destination_mask(unit, slot),
                second.swap_destination_mask(unit, slot)
            );
        }
    }
    assert_eq!(first.learn_slot_mask(), second.learn_slot_mask());
}

fn tactical_candidate(
    space: &ActionSpace,
    direction: crate::PointDirection,
    radius: i32,
) -> crate::PointCandidate {
    space
        .point_candidates()
        .iter()
        .copied()
        .find(|candidate| candidate.source == crate::PointSource::Tactical { direction, radius })
        .expect("tactical candidate")
}

fn assert_decoded_move_position(space: &ActionSpace, expected: Vec2) {
    let point = space
        .point_candidates()
        .iter()
        .position(|candidate| candidate.position == expected)
        .expect("point index");
    let issued = space
        .decode(crate::StructuredAction::MovePoint {
            unit: crate::ControlledUnit::Hero,
            point: crate::PointIndex(point),
        })
        .expect("decoded move")
        .expect("move order");
    assert_eq!(
        issued.order,
        Order::Move {
            target: bota_proto::Target::Pos(expected)
        }
    );
}

fn capacity_view(team: Team) -> WorldView {
    let mut view = world_view(team, 1);
    for index in 0..130u32 {
        let canonical = Vec2::from_ints(
            2_500 + (index % 20) as i32 * 80,
            2_500 + (index / 20) as i32 * 80,
        );
        view.units
            .push(capacity_creep(team, entity(1_000 + index, 1), canonical));
    }
    for index in 0..30u32 {
        let canonical = Vec2::from_ints(
            3_000 + (index % 6) as i32 * 700,
            3_500 + (index / 6) as i32 * 700,
        );
        view.units.push(building(
            entity(2_000 + index, 1),
            team,
            UnitKind::Tower,
            canonical_position(team, canonical),
        ));
    }
    view.units.sort_by_key(|unit| unit.id);
    view
}

fn capacity_creep(team: Team, id: EntityId, canonical: Vec2) -> UnitView {
    let mut unit = base_unit(
        id,
        UnitKind::CreepMelee,
        opposing(team),
        canonical_position(team, canonical),
    );
    unit.hp = 500;
    unit.max_hp = 500;
    unit.attack_damage = 20;
    unit.attack_range = Fixed::from_int(100);
    unit.attack_interval = 40;
    unit.radius = Fixed::from_int(20);
    unit.vision_radius = Fixed::from_int(600);
    unit
}

fn tracker_after_capacity_eviction(remapped: bool) -> StateTracker {
    let info = match_info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    let mut first = world_view(Team::Radiant, 1);
    for index in 0..249u32 {
        let id = if remapped {
            50_000 - index
        } else {
            1_000 + index
        };
        first
            .units
            .push(creep(entity(id, 1 + index % 3), 3_000, 3_000));
    }
    first.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&first).expect("capacity snapshot");

    let mut second = world_view(Team::Radiant, 2);
    for index in 0..40u32 {
        let id = if remapped {
            70_000 - index
        } else {
            2_000 + index
        };
        second.units.push(creep(
            entity(id, 7 + index % 5),
            2_500 + index as i32,
            2_500,
        ));
    }
    second.units.sort_by_key(|unit| unit.id);
    tracker
        .observe_snapshot(&second)
        .expect("evicting snapshot");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 3))
        .expect("hidden snapshot");
    tracker
}

fn encoded_frame(team: Team, view: WorldView) -> FeatureFrame {
    let tracker = tracker_with_view(team, view);
    encode(&tracker, &LocalPolicyState::new(0))
}

pub(super) fn encode(tracker: &StateTracker, local: &LocalPolicyState) -> FeatureFrame {
    let action_space = ActionSpace::from_tracker(tracker).expect("action space");
    let mut encoder = FeatureEncoder::new(tracker);
    encoder.observe(tracker).expect("feature observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            &action_space,
            &ItemReadiness::new(),
            local,
            &mut frame,
        )
        .expect("feature frame");
    frame
}

fn encode_with_readiness(tracker: &StateTracker, readiness: &ItemReadiness) -> FeatureFrame {
    let action_space =
        ActionSpace::from_tracker_with_readiness(tracker, readiness).expect("action space");
    let mut encoder = FeatureEncoder::new(tracker);
    encoder.observe(tracker).expect("feature observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            &action_space,
            readiness,
            &LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("feature frame");
    frame
}

fn encode_with_audit(tracker: &StateTracker, audit: FeatureAuditConfig) -> FeatureFrame {
    let action_space = ActionSpace::from_tracker(tracker).expect("action space");
    let mut encoder = FeatureEncoder::new_with_audit(tracker, audit);
    encoder.observe(tracker).expect("feature observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            &action_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("feature frame");
    frame
}

fn encode_with_encoder(tracker: &StateTracker, encoder: &mut FeatureEncoder) -> FeatureFrame {
    let action_space = ActionSpace::from_tracker(tracker).expect("action space");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            &action_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("feature frame");
    frame
}

pub(super) fn tracker_with_view(team: Team, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &match_info(team)).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

fn match_info(team: Team) -> MatchInfo {
    let enemy = opposing(team);
    let trees = canonical_positions(
        team,
        [Vec2::from_ints(2_100, 2_000), Vec2::from_ints(7_000, 7_000)],
    )
    .to_vec();
    let opaque_cells = trees
        .iter()
        .map(|position| {
            (
                u16::try_from(position.x.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree x cell"),
                u16::try_from(position.y.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree y cell"),
            )
        })
        .collect();
    MatchInfo {
        match_id: 99,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 90,
        trees,
        terrain_cells: AXIS,
        terrain_rle: vec![((AXIS * AXIS) as u16, 0x80)],
        opaque_cells,
        mode: TickMode::Lockstep,
        picks: vec![
            Pick {
                slot: SlotId(0),
                team,
                hero: SHADOW_FIEND,
            },
            Pick {
                slot: SlotId(1),
                team: enemy,
                hero: SHADOW_FIEND,
            },
        ],
        shop: shop(),
    }
}

pub(super) fn world_view(team: Team, tick: u32) -> WorldView {
    let enemy = opposing(team);
    let positions = canonical_positions(
        team,
        [
            Vec2::from_ints(2_000, 2_000),
            Vec2::from_ints(1_800, 1_800),
            Vec2::from_ints(2_300, 2_000),
            Vec2::from_ints(1_000, 1_000),
            Vec2::from_ints(1_400, 1_400),
            Vec2::from_ints(7_000, 7_000),
            Vec2::from_ints(6_600, 6_600),
        ],
    );
    let mut units = vec![
        hero(team, positions[0]),
        courier(team, positions[1]),
        enemy_hero(enemy, positions[2]),
        building(entity(30, 1), team, UnitKind::Fountain, positions[3]),
        building(entity(31, 1), team, UnitKind::Tower, positions[4]),
        building(entity(32, 1), enemy, UnitKind::Fountain, positions[5]),
        building(entity(33, 1), enemy, UnitKind::Tower, positions[6]),
    ];
    units.sort_by_key(|unit| unit.id);
    WorldView {
        tick,
        viewer: Some(team),
        units,
        projectiles: vec![ProjectileView {
            id: entity(40, 3),
            pos: canonical_position(team, Vec2::from_ints(2_050, 2_000)),
            facing: canonical_angle(team, Angle { brads: 0 }),
            team,
            ability: Some(AbilityId(13)),
        }],
        players: vec![own_player(team), enemy_player(enemy)],
        felled_trees: Vec::new(),
        planted_trees: vec![canonical_position(team, Vec2::from_ints(2_200, 2_000))],
        loot: vec![LootView {
            id: entity(50, 9),
            pos: canonical_position(team, Vec2::from_ints(2_000, 2_080)),
            item: ItemId(2),
            charges: Some(1),
        }],
    }
}

fn hero(team: Team, position: Vec2) -> UnitView {
    let mut unit = base_unit(HERO, UnitKind::Hero, team, position);
    unit.facing = canonical_angle(team, Angle { brads: 1_000 });
    unit.hp = 900;
    unit.max_hp = 1_000;
    unit.mana = 300;
    unit.max_mana = 400;
    unit.attack_damage = 60;
    unit.attack_range = Fixed::from_int(500);
    unit.attack_interval = 30;
    unit.attack_speed = 110;
    unit.armor = Fixed::from_int(3);
    unit.magic_resist = Fixed::from_ratio(1, 4);
    unit.radius = Fixed::from_int(24);
    unit.vision_radius = Fixed::from_int(1_800);
    unit.attributes = Attributes::all(20);
    unit.primary = Some(Attribute::Agility);
    unit.hero = Some(SHADOW_FIEND);
    unit.owner = Some(SlotId(0));
    unit.level = 5;
    unit.abilities = abilities();
    unit.items = hero_items();
    unit
}

fn courier(team: Team, position: Vec2) -> UnitView {
    let mut unit = base_unit(COURIER, UnitKind::Courier, team, position);
    unit.hp = 250;
    unit.max_hp = 250;
    unit.move_speed = Fixed::from_int(380);
    unit.radius = Fixed::from_int(16);
    unit.vision_radius = Fixed::from_int(500);
    unit.owner = Some(SlotId(0));
    unit.abilities = (8..=12)
        .map(|id| ability(AbilityId(id), Aim::Own))
        .collect();
    unit.items = vec![Some(item(ItemId(2))); 6];
    unit
}

fn enemy_hero(team: Team, position: Vec2) -> UnitView {
    let mut unit = base_unit(ENEMY, UnitKind::Hero, team, position);
    unit.hp = 800;
    unit.max_hp = 1_000;
    unit.mana = 200;
    unit.max_mana = 400;
    unit.attack_damage = 55;
    unit.attack_range = Fixed::from_int(500);
    unit.attack_interval = 32;
    unit.radius = Fixed::from_int(24);
    unit.vision_radius = Fixed::from_int(1_800);
    unit.hero = Some(SHADOW_FIEND);
    unit.owner = Some(SlotId(1));
    unit.level = 4;
    unit.abilities = abilities();
    unit.items = vec![None; 9];
    unit
}

fn creep(id: EntityId, x: i32, y: i32) -> UnitView {
    let mut unit = base_unit(id, UnitKind::CreepMelee, Team::Dire, Vec2::from_ints(x, y));
    unit.hp = 500;
    unit.max_hp = 500;
    unit.attack_damage = 20;
    unit.attack_range = Fixed::from_int(100);
    unit.attack_interval = 40;
    unit.radius = Fixed::from_int(20);
    unit.vision_radius = Fixed::from_int(600);
    unit
}

fn building(id: EntityId, team: Team, kind: UnitKind, position: Vec2) -> UnitView {
    let mut unit = base_unit(id, kind, team, position);
    unit.hp = 1_000;
    unit.max_hp = 1_000;
    unit.attack_damage = 100;
    unit.attack_range = Fixed::from_int(700);
    unit.attack_interval = 30;
    unit.radius = Fixed::from_int(80);
    unit.vision_radius = Fixed::from_int(1_800);
    unit
}

fn base_unit(id: EntityId, kind: UnitKind, team: Team, pos: Vec2) -> UnitView {
    UnitView {
        id,
        kind,
        team,
        pos,
        facing: canonical_angle(team, Angle { brads: 0 }),
        hp: 0,
        max_hp: 0,
        mana: 0,
        max_mana: 0,
        move_speed: Fixed::from_int(300),
        attack_damage: 0,
        attack_range: Fixed::ZERO,
        attack_interval: 0,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::ZERO,
        vision_radius: Fixed::ZERO,
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::ZERO,
        primary: None,
        hero: None,
        owner: None,
        level: 0,
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}

fn own_player(team: Team) -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team,
        hero: SHADOW_FIEND,
        unit: Some(HERO),
        level: 5,
        xp: 500,
        gold: Some(1_000),
        stash: Some(vec![Some(item(ItemId(2))); 6]),
        kit: None,
        kills: 2,
        deaths: 1,
        assists: 3,
        last_hits: 20,
        denies: 4,
        respawn_left: 0,
    }
}

fn enemy_player(team: Team) -> PlayerView {
    PlayerView {
        slot: SlotId(1),
        team,
        hero: SHADOW_FIEND,
        unit: Some(ENEMY),
        level: 4,
        xp: 400,
        gold: None,
        stash: None,
        kit: None,
        kills: 1,
        deaths: 2,
        assists: 2,
        last_hits: 15,
        denies: 2,
        respawn_left: 0,
    }
}

fn abilities() -> Vec<AbilityView> {
    (13..=18)
        .map(|id| ability(AbilityId(id), if id <= 15 { Aim::Point } else { Aim::Own }))
        .collect()
}

fn ability(id: AbilityId, aim: Aim) -> AbilityView {
    AbilityView {
        id,
        level: 1,
        max_level: 4,
        cooldown_left: 0,
        mana_cost: 75,
        range: 600,
        aim,
        passive: matches!(id.0, 17 | 18),
        on: false,
        can_level: false,
    }
}

fn hero_items() -> Vec<Option<ItemView>> {
    let mut items = vec![None; 9];
    items[0] = Some(item(ItemId(1)));
    items
}

fn item(id: ItemId) -> ItemView {
    ItemView {
        id,
        charges: Some(2),
        cooldown_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: true,
    }
}

fn shop() -> Vec<ShopEntry> {
    vec![
        ShopEntry {
            id: ItemId(0),
            cost: 500,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(1),
            cost: 50,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(2),
            cost: 100,
            components: vec![ItemId(1)],
        },
    ]
}

fn terrain_rle(axis: u32, cell: u8) -> Vec<(u16, u8)> {
    let mut left = u64::from(axis) * u64::from(axis);
    let mut runs = Vec::new();
    while left > 0 {
        let run = left.min(u64::from(u16::MAX));
        runs.push((run as u16, cell));
        left -= run;
    }
    runs
}

pub(super) fn reverse_entity_ids_and_generations(
    view: &mut WorldView,
    start: u32,
    generation: u32,
) {
    let old_hero = view.players[0].unit;
    let old_enemy = view.players[1].unit;
    let count = u32::try_from(view.units.len()).expect("bounded units");
    for (offset, unit) in view.units.iter_mut().enumerate() {
        let old = unit.id;
        let offset = u32::try_from(offset).expect("bounded unit index");
        unit.id = entity(start + count - offset, generation + offset % 3);
        if Some(old) == old_hero {
            view.players[0].unit = Some(unit.id);
        }
        if Some(old) == old_enemy {
            view.players[1].unit = Some(unit.id);
        }
    }
    view.units.sort_by_key(|unit| unit.id);
    for (offset, projectile) in view.projectiles.iter_mut().enumerate() {
        projectile.id = entity(start + 1_000 + offset as u32, generation);
    }
    for (offset, loot) in view.loot.iter_mut().enumerate() {
        loot.id = entity(start + 2_000 + offset as u32, generation);
    }
}

fn boundary_frame(team: Team, hero_x_raw: i32, facing: u16) -> FeatureFrame {
    let mut view = world_view(team, 1);
    let hero_index = unit_index(&view, HERO);
    view.units[hero_index].pos.x = Fixed { raw: hero_x_raw };
    view.units[hero_index].facing = Angle { brads: facing };
    encoded_frame(team, view)
}

fn assert_matrix_field_ranges<const ROWS: usize, const COLUMNS: usize>(
    name: &str,
    matrix: &[[f32; COLUMNS]; ROWS],
    categories: &[(usize, usize)],
) {
    for (row, fields) in matrix.iter().enumerate() {
        assert_field_ranges(&format!("{name}[{row}]"), fields, categories);
    }
}

fn assert_field_ranges<const FIELDS: usize>(
    name: &str,
    fields: &[f32; FIELDS],
    categories: &[(usize, usize)],
) {
    for (index, value) in fields.iter().copied().enumerate() {
        if let Some((_, maximum)) = categories.iter().find(|(field, _)| *field == index) {
            assert_eq!(value.fract(), 0.0, "{name}[{index}] category integer");
            assert!(
                (0.0..=*maximum as f32).contains(&value),
                "{name}[{index}] category {value} exceeds {maximum}"
            );
        } else {
            assert!(
                (-1.0..=1.0).contains(&value),
                "{name}[{index}] continuous value {value}"
            );
        }
    }
}

fn all_values(frame: &FeatureFrame) -> impl Iterator<Item = f32> + '_ {
    frame
        .global
        .iter()
        .copied()
        .chain(frame.history.iter().flatten().copied())
        .chain(frame.policy_history.iter().flatten().copied())
        .chain(frame.units.iter().flatten().copied())
        .chain(frame.own_units.iter().flatten().copied())
        .chain(frame.remembered_units.iter().flatten().copied())
        .chain(frame.points.iter().flatten().copied())
        .chain(frame.abilities.iter().flatten().copied())
        .chain(frame.items.iter().flatten().copied())
        .chain(frame.projectiles.iter().flatten().copied())
        .chain(frame.loot.iter().flatten().copied())
        .chain(frame.map.iter().copied())
}

fn present_unit_count(frame: &FeatureFrame) -> usize {
    frame
        .units
        .iter()
        .filter(|token| token[unit_feature::TOKEN_PRESENT] == 1.0)
        .count()
}

const fn relation_offset(relation: crate::EntityRelation) -> usize {
    match relation {
        crate::EntityRelation::Own => 0,
        crate::EntityRelation::Allied => 1,
        crate::EntityRelation::Enemy => 2,
        crate::EntityRelation::Neutral => 3,
    }
}

fn assert_matrix_eq<const ROWS: usize, const COLUMNS: usize>(
    name: &str,
    left: &[[f32; COLUMNS]; ROWS],
    right: &[[f32; COLUMNS]; ROWS],
) {
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            assert_eq!(
                left[row][column], right[row][column],
                "{name}[{row}][{column}]"
            );
        }
    }
}

fn canonical_positions<const COUNT: usize>(team: Team, values: [Vec2; COUNT]) -> [Vec2; COUNT] {
    values.map(|position| canonical_position(team, position))
}

fn canonical_position(team: Team, position: Vec2) -> Vec2 {
    if team == Team::Dire {
        let maximum = (i64::from(EXTENT) << Fixed::FRAC_BITS) - 1;
        Vec2 {
            x: Fixed {
                raw: (maximum - i64::from(position.x.raw)) as i32,
            },
            y: Fixed {
                raw: (maximum - i64::from(position.y.raw)) as i32,
            },
        }
    } else {
        position
    }
}

fn canonical_angle(team: Team, angle: Angle) -> Angle {
    if team == Team::Dire {
        Angle {
            brads: angle.brads.wrapping_sub(1 << 15),
        }
    } else {
        angle
    }
}

fn unit_index(view: &WorldView, id: EntityId) -> usize {
    view.units
        .iter()
        .position(|unit| unit.id == id)
        .expect("fixture unit")
}

const fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}

const fn opposing(team: Team) -> Team {
    match team {
        Team::Radiant => Team::Dire,
        Team::Dire => Team::Radiant,
        Team::Neutral => Team::Neutral,
    }
}
