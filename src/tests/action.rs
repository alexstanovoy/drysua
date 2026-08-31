use std::collections::BTreeSet;

use bota_proto::{
    AbilityId, AbilitySlot, AbilityView, Aim, Angle, Attribute, Attributes, EntityId, Fixed,
    HeroId, ItemId, ItemSlot, ItemView, LootView, MapId, MatchInfo, Order, Pick, PlayerView,
    ShopEntry, SlotId, StatusFlags, Target, Team, TickMode, UnitKind, UnitView, Vec2, WorldView,
};

use crate::{
    ActionError, ActionKind, ActionSpace, ActionTarget, ControlledUnit, EntityIndex,
    EntityRelation, LandmarkRelation, LootIndex, MAX_POINT_CANDIDATES, PointDirection, PointIndex,
    PointSource, PutPointTarget, SHADOW_FIEND, ShopIndex, StateTracker, StructuredAction,
    UNIT_TOKENS,
};

const HERO_ID: EntityId = entity(10, 1);
const COURIER_ID: EntityId = entity(11, 1);
const ENEMY_ID: EntityId = entity(20, 1);
const HIDDEN_ENEMY_ID: EntityId = entity(999, 7);
const LOOT_ID: EntityId = entity(50, 1);

#[test]
fn action_kind_indices_and_roundtrip_are_stable_for_all_sixteen_kinds() {
    let expected = [
        ActionKind::Continue,
        ActionKind::Stop,
        ActionKind::MovePoint,
        ActionKind::FollowUnit,
        ActionKind::Hold,
        ActionKind::AttackMovePoint,
        ActionKind::AttackUnit,
        ActionKind::Cast,
        ActionKind::Use,
        ActionKind::PutPoint,
        ActionKind::PutUnit,
        ActionKind::Take,
        ActionKind::Buy,
        ActionKind::Sell,
        ActionKind::Swap,
        ActionKind::Learn,
    ];

    assert_eq!(ActionKind::COUNT, 16);
    assert_eq!(ActionKind::ALL, expected);
    for (index, kind) in expected.into_iter().enumerate() {
        assert_eq!(kind.index(), index);
        assert_eq!(ActionKind::from_index(index), Some(kind));
    }
    assert_eq!(ActionKind::from_index(16), None);
}

#[test]
fn action_space_requires_a_snapshot_with_exact_error_message() {
    let tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");

    let error = ActionSpace::from_tracker(&tracker)
        .err()
        .expect("snapshot required");

    assert_eq!(error, ActionError::SnapshotRequired);
    assert_eq!(
        error.to_string(),
        "action space requires a validated snapshot"
    );
}

#[test]
fn entity_candidates_never_include_hidden_tracks_or_enemy_scoreboard_handles() {
    let mut tracker = tracker_with_view(world_view(1));
    let mut hidden_once = world_view(2);
    hidden_once
        .units
        .push(enemy_creep(HIDDEN_ENEMY_ID, 2_300, 2_000));
    hidden_once.units.sort_by_key(|unit| unit.id);
    tracker
        .observe_snapshot(&hidden_once)
        .expect("visible once");
    tracker
        .observe_snapshot(&world_view(3))
        .expect("hidden now");

    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let decoded_ids = decoded_candidate_ids(&space);

    assert!(!decoded_ids.contains(&HIDDEN_ENEMY_ID));
    assert!(!decoded_ids.contains(&entity(777, 4)));
    assert!(decoded_ids.contains(&ENEMY_ID));
}

#[test]
fn entity_truncation_is_deterministic_and_preserves_bodies_heroes_and_structures() {
    let mut view = world_view(1);
    for index in 100..=220 {
        view.units.push(enemy_creep(
            entity(index, 1),
            3_000 + i32::try_from(index).expect("small id"),
            3_000,
        ));
    }
    let important_hero = entity(900, 1);
    let important_tower = entity(901, 1);
    view.units.push(unit(
        important_hero,
        UnitKind::Hero,
        Team::Dire,
        7_000,
        7_000,
    ));
    view.units.push(unit(
        important_tower,
        UnitKind::Tower,
        Team::Dire,
        7_100,
        7_100,
    ));
    view.units.sort_by_key(|unit| unit.id);
    let first = ActionSpace::from_tracker(&tracker_with_view(view.clone())).expect("first");
    let second = ActionSpace::from_tracker(&tracker_with_view(view)).expect("second");

    assert_eq!(first.entity_candidates().len(), UNIT_TOKENS);
    assert_eq!(
        decoded_candidate_ids(&first),
        decoded_candidate_ids(&second)
    );
    let ids = decoded_candidate_ids(&first);
    assert!(ids.contains(&HERO_ID));
    assert!(ids.contains(&COURIER_ID));
    assert!(ids.contains(&important_hero));
    assert!(ids.contains(&important_tower));
}

#[test]
fn point_candidates_are_bounded_deduplicated_and_cover_directions_landmarks_and_trees() {
    let mut tracker = tracker_with_view(world_view(1));
    let mut moved = world_view(3);
    moved
        .units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_ID)
        .expect("enemy")
        .pos = Vec2::from_ints(2_364, 2_000);
    tracker.observe_snapshot(&moved).expect("velocity snapshot");

    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let points = space.point_candidates();
    let unique: BTreeSet<Vec2> = points.iter().map(|point| point.position).collect();

    assert!(points.len() <= MAX_POINT_CANDIDATES);
    assert_eq!(unique.len(), points.len());
    assert!(points.iter().any(|point| {
        point.source
            == PointSource::Tactical {
                direction: PointDirection::East,
                radius: 200,
            }
            && point.position == Vec2::from_ints(2_200, 2_000)
    }));
    assert!(points.iter().any(|point| {
        point.source
            == PointSource::Tactical {
                direction: PointDirection::NorthEast,
                radius: 200,
            }
            && point.position == Vec2::from_ints(2_141, 2_141)
    }));
    assert!(points.iter().any(|point| {
        point.source == PointSource::StaticTree && point.position == Vec2::from_ints(2_100, 2_100)
    }));
    assert!(!points.iter().any(|point| {
        matches!(point.source, PointSource::StaticTree)
            && point.position == Vec2::from_ints(2_050, 2_050)
    }));
    assert!(points.iter().any(|point| {
        point.source == PointSource::PredictedHero(EntityRelation::Enemy)
            && point.position == Vec2::from_ints(2_428, 2_000)
    }));
    assert!(
        points
            .iter()
            .any(|point| { point.source == PointSource::Fountain(LandmarkRelation::Own) })
    );
    assert!(
        points
            .iter()
            .any(|point| { point.source == PointSource::Tower(LandmarkRelation::Enemy) })
    );
}

#[test]
fn terrain_rle_controls_point_walkability_at_the_explicit_cell_boundary() {
    let blocked_cell = 31 * 128 + 34;
    let mut info = match_info();
    info.terrain_rle = terrain_with_blocked_cell(blocked_cell);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&world_view(1)).expect("snapshot");

    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let east = space
        .point_candidates()
        .iter()
        .find(|point| {
            point.source
                == PointSource::Tactical {
                    direction: PointDirection::East,
                    radius: 200,
                }
        })
        .expect("east point");

    assert!(!east.walkable);
}

#[test]
fn remote_tree_deltas_preserve_static_passability_while_structures_still_block() {
    let mut info = match_info();
    info.trees = vec![Vec2::from_ints(2_208, 2_016)];
    let mut view = world_view(1);
    view.felled_trees = vec![0];
    view.planted_trees = vec![Vec2::from_ints(2_016, 2_208)];
    view.units.push(unit(
        entity(40, 1),
        UnitKind::Tower,
        Team::Dire,
        1_824,
        2_016,
    ));
    view.units.sort_by_key(|unit| unit.id);
    let hero_index = hero_index(&view);
    view.units[hero_index].items[0] = Some(item(Some(Aim::Point), 1_200));
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");

    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let east = tactical_point(&space, PointDirection::East, 200);
    let north = tactical_point(&space, PointDirection::North, 200);
    let west = tactical_point(&space, PointDirection::West, 200);

    assert!(!space.point_candidates()[east.0].walkable);
    assert!(!space.move_point_mask(ControlledUnit::Hero)[east.0]);
    assert!(
        !space
            .put_point_target_mask(ControlledUnit::Hero, ItemSlot(0))
            .expect("put mask")[east.0]
    );
    assert!(space.point_candidates()[north.0].walkable);
    assert!(space.move_point_mask(ControlledUnit::Hero)[north.0]);
    assert!(!space.point_candidates()[west.0].walkable);
}

#[test]
fn locally_proven_felled_static_tree_unblocks_walkability_and_tree_mask() {
    let mut info = match_info();
    info.trees = vec![Vec2::from_ints(2_000, 2_000)];
    info.opaque_cells = vec![(31, 31)];
    let mut standing = world_view(1);
    standing.felled_trees.clear();
    standing.planted_trees.clear();
    let hero_index = hero_index(&standing);
    standing.units[hero_index].items[0] = Some(item(Some(Aim::Point), 1_200));
    let standing_space =
        ActionSpace::from_tracker(&tracker_with_info_and_view(&info, standing.clone()))
            .expect("standing tree space");
    assert!(!standing_space.put_underfoot_mask(ControlledUnit::Hero)[0]);
    assert!(
        standing_space
            .cast_target_mask(ControlledUnit::Hero, AbilitySlot(3))
            .expect("tree mask")
            .points()
            .contains(&true)
    );

    standing.felled_trees.push(0);
    let felled_space = ActionSpace::from_tracker(&tracker_with_info_and_view(&info, standing))
        .expect("felled tree space");
    assert!(felled_space.put_underfoot_mask(ControlledUnit::Hero)[0]);
    assert!(
        !felled_space
            .cast_target_mask(ControlledUnit::Hero, AbilitySlot(3))
            .expect("tree mask")
            .points()
            .contains(&true)
    );
}

#[test]
fn locally_proven_planted_tree_blocks_walkability_and_enters_tree_mask() {
    let mut info = match_info();
    info.trees.clear();
    info.opaque_cells.clear();
    let mut open = world_view(1);
    open.felled_trees.clear();
    open.planted_trees.clear();
    let hero_index = hero_index(&open);
    open.units[hero_index].items[0] = Some(item(Some(Aim::Point), 1_200));
    let open_space = ActionSpace::from_tracker(&tracker_with_info_and_view(&info, open.clone()))
        .expect("open space");
    assert!(open_space.put_underfoot_mask(ControlledUnit::Hero)[0]);

    open.planted_trees.push(Vec2::from_ints(2_000, 2_000));
    let planted_space = ActionSpace::from_tracker(&tracker_with_info_and_view(&info, open))
        .expect("planted tree space");
    assert!(!planted_space.put_underfoot_mask(ControlledUnit::Hero)[0]);
    assert!(
        planted_space
            .cast_target_mask(ControlledUnit::Hero, AbilitySlot(3))
            .expect("tree mask")
            .points()
            .contains(&true)
    );
}

#[test]
fn remote_tree_deltas_leave_candidates_and_masks_invariant_until_visibility_is_proven() {
    let mut info = match_info();
    info.trees.push(Vec2::from_ints(7_000, 7_000));
    let mut baseline = world_view(1);
    baseline.felled_trees.clear();
    baseline.planted_trees.clear();
    let baseline_space =
        ActionSpace::from_tracker(&tracker_with_info_and_view(&info, baseline.clone()))
            .expect("baseline");

    let mut remote = baseline.clone();
    remote.felled_trees.push(3);
    remote.planted_trees.push(Vec2::from_ints(7_100, 7_100));
    let remote_space =
        ActionSpace::from_tracker(&tracker_with_info_and_view(&info, remote)).expect("remote");
    assert_eq!(
        baseline_space.point_candidates(),
        remote_space.point_candidates()
    );
    assert_eq!(
        baseline_space
            .cast_target_mask(ControlledUnit::Hero, AbilitySlot(3))
            .expect("tree mask")
            .points(),
        remote_space
            .cast_target_mask(ControlledUnit::Hero, AbilitySlot(3))
            .expect("tree mask")
            .points()
    );

    baseline.felled_trees.push(1);
    let local_space = ActionSpace::from_tracker(&tracker_with_info_and_view(&info, baseline))
        .expect("local delta");
    assert_ne!(
        baseline_space.point_candidates(),
        local_space.point_candidates()
    );
}

#[test]
fn town_portal_targets_walkable_landings_by_building_range_not_user_range() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].pos = Vec2::from_ints(6_000, 6_000);
    view.units[hero_index].items[0] =
        Some(item_with_id(ItemId(8), Some(Aim::Building), 600, false));
    let space = ActionSpace::from_tracker(&tracker_with_view(view)).expect("space");
    let mask = space
        .use_target_mask(ControlledUnit::Hero, ItemSlot(0))
        .expect("town portal mask");

    let allowed: Vec<usize> = mask
        .points()
        .iter()
        .enumerate()
        .filter_map(|(index, allowed)| allowed.then_some(index))
        .collect();
    assert!(!allowed.is_empty());
    for index in allowed {
        let candidate = space.point_candidates()[index];
        assert!(candidate.walkable);
        assert!(candidate.allied_building);
        assert!(matches!(candidate.source, PointSource::BuildingLanding(_)));
        assert!(
            !candidate
                .position
                .within(Vec2::from_ints(6_000, 6_000), Fixed::from_int(600))
        );
    }
    for (index, candidate) in space.point_candidates().iter().enumerate() {
        if matches!(
            candidate.source,
            PointSource::Tower(LandmarkRelation::Own)
                | PointSource::Fountain(LandmarkRelation::Own)
        ) {
            assert!(!mask.points()[index]);
            assert!(!candidate.walkable);
        }
    }
}

#[test]
fn hero_and_courier_masks_follow_availability_and_status_boundaries() {
    let baseline = ActionSpace::from_tracker(&tracker_with_view(world_view(1))).expect("baseline");
    assert!(
        baseline
            .controlled_unit_mask(ActionKind::Stop)
            .allows(ControlledUnit::Hero)
    );
    assert!(
        baseline
            .controlled_unit_mask(ActionKind::Stop)
            .allows(ControlledUnit::Courier)
    );

    let mut dead_hero = world_view(1);
    dead_hero.players[0].unit = None;
    dead_hero.players[0].kit = Some(bota_proto::Kit {
        abilities: hero().abilities,
        items: hero().items,
    });
    dead_hero.units.retain(|unit| unit.id != HERO_ID);
    let dead = ActionSpace::from_tracker(&tracker_with_view(dead_hero)).expect("dead hero space");
    assert!(
        !dead
            .controlled_unit_mask(ActionKind::Stop)
            .allows(ControlledUnit::Hero)
    );
    assert!(
        dead.controlled_unit_mask(ActionKind::Stop)
            .allows(ControlledUnit::Courier)
    );

    let mut disabled = world_view(1);
    let disabled_hero = hero_index(&disabled);
    disabled.units[disabled_hero].statuses.bits =
        StatusFlags::STUNNED | StatusFlags::ROOTED | StatusFlags::DISARMED;
    let disabled = ActionSpace::from_tracker(&tracker_with_view(disabled)).expect("disabled");
    assert!(disabled.allows(StructuredAction::Stop {
        unit: ControlledUnit::Hero,
    }));
    assert!(
        !disabled
            .controlled_unit_mask(ActionKind::MovePoint)
            .allows(ControlledUnit::Hero)
    );
    assert!(
        !disabled
            .controlled_unit_mask(ActionKind::Hold)
            .allows(ControlledUnit::Hero)
    );
}

#[test]
fn ability_masks_apply_aim_passive_learning_cooldown_mana_silence_and_strict_range() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].abilities = vec![
        ability(Aim::Own, 0),
        ability(Aim::Unit, 300),
        ability(Aim::Point, 1_200),
        AbilityView {
            passive: true,
            ..ability(Aim::Own, 0)
        },
        AbilityView {
            level: 0,
            ..ability(Aim::Own, 0)
        },
        AbilityView {
            cooldown_left: 1,
            mana_cost: 10_000,
            ..ability(Aim::Own, 0)
        },
    ];
    let space = ActionSpace::from_tracker(&tracker_with_view(view.clone())).expect("space");
    assert_eq!(
        space.ability_slot_mask(ControlledUnit::Hero),
        [true, true, true, false, false, false]
    );
    let enemy = entity_candidate(&space, EntityRelation::Enemy, UnitKind::Hero);
    assert!(space.allows(StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(1),
        target: ActionTarget::Entity(enemy),
    }));

    view.units[hero_index].abilities[1].range = 299;
    let short = ActionSpace::from_tracker(&tracker_with_view(view.clone())).expect("short range");
    let enemy = entity_candidate(&short, EntityRelation::Enemy, UnitKind::Hero);
    assert!(!short.allows(StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(1),
        target: ActionTarget::Entity(enemy),
    }));

    view.units[hero_index].statuses.bits = StatusFlags::SILENCED;
    let silenced = ActionSpace::from_tracker(&tracker_with_view(view)).expect("silenced");
    assert!(
        silenced
            .ability_slot_mask(ControlledUnit::Hero)
            .iter()
            .all(|allowed| !allowed)
    );
}

#[test]
fn item_masks_cover_all_aims_and_reject_backpack_cooldown_charges_mana_and_range() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].items = action_items();
    let space = ActionSpace::from_tracker(&tracker_with_view(view.clone())).expect("space");

    assert_eq!(
        space.item_slot_mask(ControlledUnit::Hero),
        [true, true, true, true, true, false]
    );
    let ally = entity_candidate(&space, EntityRelation::Own, UnitKind::Courier);
    assert!(space.allows(StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(1),
        target: ActionTarget::Entity(ally),
    }));
    assert!(!space.allows(StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(1),
        target: ActionTarget::Entity(entity_candidate(
            &space,
            EntityRelation::Enemy,
            UnitKind::Hero,
        )),
    }));
    assert!(!space.allows(StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(6),
        target: ActionTarget::None,
    }));

    view.units[hero_index].items[0]
        .as_mut()
        .expect("item")
        .charges = Some(0);
    view.units[hero_index].items[1]
        .as_mut()
        .expect("item")
        .cooldown_left = 1;
    view.units[hero_index].items[2]
        .as_mut()
        .expect("item")
        .mana_cost = 10_000;
    let blocked = ActionSpace::from_tracker(&tracker_with_view(view)).expect("blocked");
    assert!(!blocked.item_slot_mask(ControlledUnit::Hero)[0]);
    assert!(!blocked.item_slot_mask(ControlledUnit::Hero)[1]);
    assert!(!blocked.item_slot_mask(ControlledUnit::Hero)[2]);
}

#[test]
fn inventory_economy_and_learning_masks_cover_positive_and_negative_boundaries() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].items = action_items();
    view.units[hero_index].items[8] = None;
    view.players[0].stash = Some(vec![Some(item(Some(Aim::Own), 0)); 6]);
    let full_stash = ActionSpace::from_tracker(&tracker_with_view(view.clone())).expect("space");
    assert!(full_stash.take_mask(ControlledUnit::Hero)[0]);
    assert!(!full_stash.has_sell_ownership_uncertainty());
    assert_eq!(full_stash.sell_ownership_uncertain_count(), 0);
    assert!(full_stash.sell_slot_mask(ControlledUnit::Hero)[0]);
    assert!(!full_stash.sell_slot_mask(ControlledUnit::Hero)[1]);
    assert!(
        full_stash
            .swap_destination_mask(ControlledUnit::Hero, ItemSlot(0))
            .expect("source")[8]
    );
    assert!(full_stash.learn_slot_mask()[0]);
    assert!(full_stash.buy_mask(ControlledUnit::Hero)[0]);

    view.units[hero_index].items[8] = Some(item(Some(Aim::Own), 0));
    view.units[hero_index].items[7] = Some(item(Some(Aim::Own), 0));
    view.players[0].gold = Some(0);
    let no_capacity_or_gold =
        ActionSpace::from_tracker(&tracker_with_view(view)).expect("blocked economy");
    assert!(
        no_capacity_or_gold
            .take_mask(ControlledUnit::Hero)
            .iter()
            .all(|allowed| !allowed)
    );
    assert!(
        no_capacity_or_gold
            .buy_mask(ControlledUnit::Hero)
            .iter()
            .all(|allowed| !allowed)
    );
    assert!(
        !no_capacity_or_gold
            .swap_destination_mask(ControlledUnit::Hero, ItemSlot(0))
            .expect("source")[0]
    );
}

#[test]
fn sell_requires_for_sale_as_prior_ownership_proof() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].items[0] = Some(item_with_id(ItemId(0), None, 0, false));
    view.units[hero_index].items[1] = Some(item_with_id(ItemId(1), None, 0, true));

    let space = ActionSpace::from_tracker(&tracker_with_view(view)).expect("space");

    assert!(!space.sell_slot_mask(ControlledUnit::Hero)[0]);
    assert!(space.sell_slot_mask(ControlledUnit::Hero)[1]);
    assert!(!space.has_sell_ownership_uncertainty());
}

#[test]
fn composite_buy_requires_each_missing_part_slot_and_courier_buy_is_masked() {
    let mut info = match_info();
    info.shop = vec![
        ShopEntry {
            id: ItemId(0),
            cost: 400,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(1),
            cost: 400,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(2),
            cost: 900,
            components: vec![ItemId(0), ItemId(1)],
        },
    ];
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].items = vec![Some(item_with_id(ItemId(9), None, 0, false)); 9];
    view.players[0].stash = Some(vec![Some(item_with_id(ItemId(9), None, 0, false)); 5]);
    view.players[0].stash.as_mut().expect("stash").push(None);
    view.players[0].gold = Some(1_000);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");

    let no_parts = ActionSpace::from_tracker(&tracker).expect("space");
    assert!(!no_parts.buy_mask(ControlledUnit::Hero)[2]);
    assert!(
        no_parts
            .buy_mask(ControlledUnit::Courier)
            .iter()
            .all(|allowed| !allowed)
    );

    view.units[hero_index].items[0] = Some(item_with_id(ItemId(0), None, 0, false));
    tracker
        .observe_snapshot(&WorldView { tick: 2, ..view })
        .expect("second snapshot");
    let one_part_held = ActionSpace::from_tracker(&tracker).expect("space");
    assert!(one_part_held.buy_mask(ControlledUnit::Hero)[2]);
}

#[test]
fn composite_buy_requires_full_price_and_missing_leaf_price() {
    let mut info = match_info();
    // Server validation checks the 250 full cost, but execution pays the 300
    // missing leaf sum: gold 280 passes validation yet silently buys nothing.
    info.shop = vec![
        ShopEntry {
            id: ItemId(0),
            cost: 150,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(1),
            cost: 150,
            components: Vec::new(),
        },
        ShopEntry {
            id: ItemId(2),
            cost: 250,
            components: vec![ItemId(0), ItemId(1)],
        },
    ];
    let mut view = world_view(1);
    view.players[0].gold = Some(200);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");

    let below_full = ActionSpace::from_tracker(&tracker).expect("space");
    assert!(!below_full.buy_mask(ControlledUnit::Hero)[2]);

    view.players[0].gold = Some(280);
    tracker
        .observe_snapshot(&WorldView {
            tick: 2,
            ..view.clone()
        })
        .expect("second snapshot");
    let validation_only = ActionSpace::from_tracker(&tracker).expect("space");
    assert!(!validation_only.buy_mask(ControlledUnit::Hero)[2]);

    view.players[0].gold = Some(300);
    tracker
        .observe_snapshot(&WorldView { tick: 3, ..view })
        .expect("third snapshot");
    let enough_for_both = ActionSpace::from_tracker(&tracker).expect("space");
    assert!(enough_for_both.buy_mask(ControlledUnit::Hero)[2]);
}

#[test]
fn cyclic_and_unknown_recipe_schemas_return_action_error() {
    let mut cyclic_info = match_info();
    cyclic_info.shop = vec![
        ShopEntry {
            id: ItemId(0),
            cost: 100,
            components: vec![ItemId(1)],
        },
        ShopEntry {
            id: ItemId(1),
            cost: 100,
            components: vec![ItemId(0)],
        },
    ];
    let cyclic = tracker_with_info_and_view(&cyclic_info, world_view(1));
    assert_eq!(
        ActionSpace::from_tracker(&cyclic).err(),
        Some(ActionError::InvalidSchema("cyclic shop recipe"))
    );

    let mut unknown_info = match_info();
    unknown_info.shop[0].components = vec![ItemId(99)];
    let unknown = tracker_with_info_and_view(&unknown_info, world_view(1));
    assert_eq!(
        ActionSpace::from_tracker(&unknown).err(),
        Some(ActionError::InvalidSchema("unknown recipe component"))
    );
}

#[test]
fn schema_decode_maps_every_action_family_to_wire_order_and_rejects_injected_indices() {
    let mut view = world_view(1);
    let hero_index = hero_index(&view);
    view.units[hero_index].items = action_items();
    let space = ActionSpace::from_tracker(&tracker_with_view(view)).expect("space");
    let point = walkable_point(&space);
    let courier = entity_candidate(&space, EntityRelation::Own, UnitKind::Courier);
    let enemy = entity_candidate(&space, EntityRelation::Enemy, UnitKind::Hero);

    assert_eq!(
        space.decode(StructuredAction::Continue).expect("continue"),
        None
    );
    assert_order(
        &space,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero,
        },
        None,
        Order::Move {
            target: Target::None,
        },
    );
    assert_order(
        &space,
        StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point,
        },
        None,
        Order::Move {
            target: Target::Pos(space.point_candidates()[point.0].position),
        },
    );
    assert_order(
        &space,
        StructuredAction::FollowUnit {
            unit: ControlledUnit::Hero,
            target: courier,
        },
        None,
        Order::Move {
            target: Target::Unit(COURIER_ID),
        },
    );
    assert_order(
        &space,
        StructuredAction::Hold {
            unit: ControlledUnit::Hero,
        },
        None,
        Order::Attack {
            target: Target::None,
        },
    );
    assert_order(
        &space,
        StructuredAction::AttackMovePoint {
            unit: ControlledUnit::Hero,
            point,
        },
        None,
        Order::Attack {
            target: Target::Pos(space.point_candidates()[point.0].position),
        },
    );
    assert_order(
        &space,
        StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target: enemy,
        },
        None,
        Order::Attack {
            target: Target::Unit(ENEMY_ID),
        },
    );
    assert_order(
        &space,
        StructuredAction::Cast {
            unit: ControlledUnit::Hero,
            slot: AbilitySlot(0),
            target: ActionTarget::None,
        },
        None,
        Order::Cast {
            slot: AbilitySlot(0),
            target: Target::None,
        },
    );
    assert_order(
        &space,
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::None,
        },
        None,
        Order::Use {
            slot: ItemSlot(0),
            target: Target::None,
        },
    );
    assert_order(
        &space,
        StructuredAction::PutPoint {
            unit: ControlledUnit::Hero,
            source: ItemSlot(0),
            target: PutPointTarget::Point(point),
        },
        None,
        Order::Put {
            slot: ItemSlot(0),
            target: Target::Pos(space.point_candidates()[point.0].position),
        },
    );
    assert_order(
        &space,
        StructuredAction::PutUnit {
            unit: ControlledUnit::Hero,
            source: ItemSlot(0),
            target: courier,
        },
        None,
        Order::Put {
            slot: ItemSlot(0),
            target: Target::Unit(COURIER_ID),
        },
    );
    assert_order(
        &space,
        StructuredAction::Take {
            unit: ControlledUnit::Hero,
            loot: LootIndex(0),
        },
        None,
        Order::Take {
            target: Target::Unit(LOOT_ID),
        },
    );
    assert_order(
        &space,
        StructuredAction::Buy {
            unit: ControlledUnit::Hero,
            item: ShopIndex(0),
        },
        None,
        Order::Buy { item: ItemId(0) },
    );
    assert_order(
        &space,
        StructuredAction::Sell {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
        },
        None,
        Order::Sell { slot: ItemSlot(0) },
    );
    assert_order(
        &space,
        StructuredAction::Swap {
            unit: ControlledUnit::Hero,
            from: ItemSlot(0),
            to: ItemSlot(8),
        },
        None,
        Order::Swap {
            from: ItemSlot(0),
            to: ItemSlot(8),
        },
    );
    assert_order(
        &space,
        StructuredAction::Learn {
            slot: AbilitySlot(0),
        },
        None,
        Order::Learn {
            slot: AbilitySlot(0),
        },
    );
    assert_order(
        &space,
        StructuredAction::Stop {
            unit: ControlledUnit::Courier,
        },
        Some(COURIER_ID),
        Order::Move {
            target: Target::None,
        },
    );

    let error = space
        .decode(StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target: EntityIndex(UNIT_TOKENS),
        })
        .expect_err("fabricated target");
    assert_eq!(
        error.to_string(),
        format!(
            "entity target index 96 is outside candidate count {}",
            space.entity_candidates().len()
        )
    );
}

#[cfg(feature = "builtin")]
#[test]
fn real_builtin_map_zero_and_one_snapshots_build_and_safe_allowed_actions_validate() {
    for map in [MapId(0), MapId(1)] {
        let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
            seats: 2,
            map,
            seed: 919,
        })
        .expect("arena");
        let (info, view) = start_info_and_view(&start.messages[0]);
        let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
        tracker.observe_snapshot(view).expect("snapshot");
        let space = ActionSpace::from_tracker(&tracker).expect("real action space");
        let actions = [
            StructuredAction::Continue,
            StructuredAction::Stop {
                unit: ControlledUnit::Hero,
            },
            StructuredAction::Hold {
                unit: ControlledUnit::Hero,
            },
        ];
        for (sequence, action) in actions.into_iter().enumerate() {
            assert!(space.allows(action));
            let decoded = space.decode(action).expect("allowed action decodes");
            let request = decoded.map(|issued| crate::Request {
                seq: u32::try_from(sequence).expect("small sequence"),
                unit: issued.unit,
                order: issued.order,
            });
            let step = arena.step(&[request, None]).expect("arena step");
            assert!(
                !step.messages[0]
                    .iter()
                    .any(|message| matches!(message, bota_proto::ServerMsg::OrderRejected { .. }))
            );
        }
    }
}

fn decoded_candidate_ids(space: &ActionSpace) -> Vec<EntityId> {
    let mut ids = Vec::new();
    for unit in [ControlledUnit::Hero, ControlledUnit::Courier] {
        for index in 0..space.entity_candidates().len() {
            let decoded = space
                .decode(StructuredAction::FollowUnit {
                    unit,
                    target: EntityIndex(index),
                })
                .ok()
                .flatten();
            if let Some(crate::IssuedOrder {
                order: Order::Move {
                    target: Target::Unit(id),
                },
                ..
            }) = decoded
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
    }
    ids.sort_unstable();
    ids
}

fn entity_candidate(space: &ActionSpace, relation: EntityRelation, kind: UnitKind) -> EntityIndex {
    EntityIndex(
        space
            .entity_candidates()
            .iter()
            .position(|candidate| candidate.relation == relation && candidate.kind == kind)
            .expect("entity candidate"),
    )
}

fn walkable_point(space: &ActionSpace) -> PointIndex {
    PointIndex(
        space
            .point_candidates()
            .iter()
            .position(|point| point.walkable)
            .expect("walkable point"),
    )
}

fn assert_order(
    space: &ActionSpace,
    action: StructuredAction,
    unit: Option<EntityId>,
    order: Order,
) {
    assert!(space.allows(action), "{action:?}");
    let issued = space.decode(action).expect("decode").expect("wire order");
    assert_eq!(issued.unit, unit);
    assert_eq!(issued.order, order);
}

fn tracker_with_view(view: WorldView) -> StateTracker {
    tracker_with_info_and_view(&match_info(), view)
}

fn tracker_with_info_and_view(info: &MatchInfo, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

fn tactical_point(space: &ActionSpace, direction: PointDirection, radius: i32) -> PointIndex {
    PointIndex(
        space
            .point_candidates()
            .iter()
            .position(|point| point.source == PointSource::Tactical { direction, radius })
            .expect("tactical point"),
    )
}

fn item_with_id(id: ItemId, aim: Option<Aim>, range: i32, for_sale: bool) -> ItemView {
    ItemView {
        id,
        for_sale,
        ..item(aim, range)
    }
}

fn match_info() -> MatchInfo {
    MatchInfo {
        match_id: 1,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 90,
        trees: vec![
            Vec2::from_ints(2_050, 2_050),
            Vec2::from_ints(2_100, 2_100),
            Vec2::from_ints(2_200, 1_900),
        ],
        terrain_cells: 128,
        terrain_rle: vec![(16_384, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: SHADOW_FIEND,
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: SHADOW_FIEND,
            },
        ],
        shop: vec![
            ShopEntry {
                id: ItemId(0),
                cost: 50,
                components: Vec::new(),
            },
            ShopEntry {
                id: ItemId(1),
                cost: 700,
                components: Vec::new(),
            },
        ],
    }
}

fn world_view(tick: u32) -> WorldView {
    let mut units = vec![
        hero(),
        courier(),
        unit(ENEMY_ID, UnitKind::Hero, Team::Dire, 2_300, 2_000),
        unit(
            entity(30, 1),
            UnitKind::Fountain,
            Team::Radiant,
            1_500,
            1_500,
        ),
        unit(entity(31, 1), UnitKind::Fountain, Team::Dire, 7_000, 7_000),
        unit(entity(32, 1), UnitKind::Tower, Team::Radiant, 2_500, 2_000),
        unit(entity(33, 1), UnitKind::Tower, Team::Dire, 3_000, 2_000),
    ];
    units.sort_by_key(|unit| unit.id);
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units,
        projectiles: Vec::new(),
        players: vec![own_player(), enemy_player()],
        felled_trees: vec![0],
        planted_trees: vec![Vec2::from_ints(1_900, 2_100)],
        loot: vec![LootView {
            id: LOOT_ID,
            pos: Vec2::from_ints(2_050, 2_000),
            item: ItemId(0),
            charges: Some(1),
        }],
    }
}

fn hero() -> UnitView {
    let mut hero = unit(HERO_ID, UnitKind::Hero, Team::Radiant, 2_000, 2_000);
    hero.hero = Some(SHADOW_FIEND);
    hero.owner = Some(SlotId(0));
    hero.mana = 500;
    hero.max_mana = 500;
    hero.level = 6;
    hero.abilities = vec![
        AbilityView {
            can_level: true,
            ..ability(Aim::Own, 0)
        },
        ability(Aim::Unit, 300),
        ability(Aim::Point, 1_200),
        ability(Aim::Tree, 1_200),
        ability(Aim::Building, 1_200),
        ability(Aim::Own, 0),
    ];
    hero.items = vec![None; 9];
    hero
}

fn courier() -> UnitView {
    let mut courier = unit(COURIER_ID, UnitKind::Courier, Team::Radiant, 2_100, 2_000);
    courier.owner = Some(SlotId(0));
    courier.items = vec![None; 6];
    courier.attack_damage = 0;
    courier
}

fn enemy_creep(id: EntityId, x: i32, y: i32) -> UnitView {
    unit(id, UnitKind::CreepMelee, Team::Dire, x, y)
}

fn unit(id: EntityId, kind: UnitKind, team: Team, x: i32, y: i32) -> UnitView {
    UnitView {
        id,
        kind,
        team,
        pos: Vec2::from_ints(x, y),
        facing: Angle { brads: 0 },
        hp: 1_000,
        max_hp: 1_000,
        mana: 0,
        max_mana: 0,
        move_speed: Fixed::from_int(300),
        attack_damage: 50,
        attack_range: Fixed::from_int(500),
        attack_interval: 30,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(24),
        vision_radius: Fixed::from_int(1_800),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::all(20),
        primary: Some(Attribute::Agility),
        hero: (kind == UnitKind::Hero).then_some(HeroId(2)),
        owner: None,
        level: 0,
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}

fn ability(aim: Aim, range: i32) -> AbilityView {
    AbilityView {
        id: AbilityId(13),
        level: 1,
        max_level: 4,
        cooldown_left: 0,
        mana_cost: 75,
        range,
        aim,
        passive: false,
        on: false,
        can_level: false,
    }
}

fn action_items() -> Vec<Option<ItemView>> {
    let mut first = item(Some(Aim::Own), 0);
    first.for_sale = true;
    vec![
        Some(first),
        Some(item(Some(Aim::Unit), 250)),
        Some(item(Some(Aim::Point), 1_200)),
        Some(item(Some(Aim::Tree), 1_200)),
        Some(item(Some(Aim::Building), 1_200)),
        Some(item(None, 0)),
        Some(item(Some(Aim::Own), 0)),
        None,
        None,
    ]
}

fn item(aim: Option<Aim>, range: i32) -> ItemView {
    ItemView {
        id: ItemId(0),
        charges: Some(1),
        cooldown_left: 0,
        mode: None,
        mana_cost: 0,
        range,
        aim,
        for_sale: false,
    }
}

fn own_player() -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team: Team::Radiant,
        hero: SHADOW_FIEND,
        unit: Some(HERO_ID),
        level: 6,
        xp: 0,
        gold: Some(600),
        stash: Some(vec![None; 6]),
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

fn enemy_player() -> PlayerView {
    PlayerView {
        slot: SlotId(1),
        team: Team::Dire,
        hero: SHADOW_FIEND,
        unit: Some(entity(777, 4)),
        level: 1,
        xp: 0,
        gold: None,
        stash: None,
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

fn hero_index(view: &WorldView) -> usize {
    view.units
        .iter()
        .position(|unit| unit.id == HERO_ID)
        .expect("hero")
}

fn terrain_with_blocked_cell(blocked: usize) -> Vec<(u16, u8)> {
    let before = u16::try_from(blocked).expect("before fits");
    let after = u16::try_from(16_384 - blocked - 1).expect("after fits");
    vec![(before, 0x80), (1, 0), (after, 0x80)]
}

const fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}

#[cfg(feature = "builtin")]
fn start_info_and_view(messages: &[bota_proto::ServerMsg]) -> (&MatchInfo, &WorldView) {
    let info = messages
        .iter()
        .find_map(|message| match message {
            bota_proto::ServerMsg::MatchStart { info } => Some(info),
            _ => None,
        })
        .expect("match info");
    let view = messages
        .iter()
        .find_map(|message| match message {
            bota_proto::ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .expect("snapshot");
    (info, view)
}
