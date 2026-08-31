use bota_proto::{
    AbilityId, AbilityView, Aim, Angle, Attribute, Attributes, DamageKind, EffectId, EffectView,
    EntityId, EventKind, Fixed, HeroId, ItemId, ItemView, Kit, LootView, MapId, MatchInfo, Pick,
    PlayerView, ProjectileView, ShopEntry, SlotId, StatusFlags, Team, TickMode, UnitKind, UnitView,
    Vec2, WorldView,
};

use crate::{
    HISTORY_AGES, HISTORY_TICKS, MAX_ABILITY_SLOTS, MAX_EFFECTS_PER_UNIT, MAX_EVENTS_PER_BATCH,
    MAX_HISTORY_SUMMARIES, MAX_LOOT, MAX_OPAQUE_CELLS, MAX_PLANTED_TREES, MAX_PROJECTILES,
    MAX_RECENT_EVENTS, MAX_SEATS, MAX_SHOP_COMPONENTS, MAX_SHOP_ITEMS, MAX_STATIC_TREES,
    MAX_TERRAIN_AXIS, MAX_TERRAIN_RUNS, MAX_TRACKED_ENTITIES, OWN_ITEM_SLOTS, SHADOW_FIEND,
    SHADOW_FIEND_ABILITY_SLOTS, StateTracker, UNIT_TOKENS,
};

#[test]
fn tracker_initializes_with_valid_shadow_fiend_match_terms() {
    let info = match_info();

    let tracker = StateTracker::new(SlotId(0), &info).expect("valid tracker");

    assert_eq!(tracker.slot(), SlotId(0));
    assert_eq!(tracker.team(), Team::Radiant);
    assert_eq!(tracker.metadata().map, MapId(0));
    assert_eq!(tracker.metadata().seats, 2);
    assert_eq!(tracker.static_trees(), [Vec2::from_ints(4, 5)]);
    assert_eq!(tracker.terrain_rle(), [(64, 0x80)]);
    assert_eq!(tracker.opaque_cells(), [(0, 0)]);
    assert_eq!(tracker.shop().len(), 1);
    assert!(tracker.current().is_none());
}

#[test]
fn match_start_opaque_baseline_contains_the_static_tree_cell() {
    let tracker = new_tracker();
    let tree = tracker.static_trees()[0];
    let cell = (
        u16::try_from(tree.x.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree x cell"),
        u16::try_from(tree.y.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree y cell"),
    );

    assert!(tracker.opaque_cells().contains(&cell));
}

#[test]
fn tracker_public_hard_limits_match_stage_three_contract() {
    assert_eq!(MAX_SEATS, 10);
    assert_eq!(UNIT_TOKENS, 96);
    assert_eq!(MAX_TRACKED_ENTITIES, 256);
    assert_eq!(MAX_PROJECTILES, 32);
    assert_eq!(MAX_LOOT, 16);
    assert_eq!(SHADOW_FIEND_ABILITY_SLOTS, 6);
    assert_eq!(MAX_ABILITY_SLOTS, 8);
    assert_eq!(OWN_ITEM_SLOTS, 21);
    assert_eq!(MAX_SHOP_ITEMS, 64);
    assert_eq!(crate::MAX_POINT_CANDIDATES, 48);
    assert_eq!(MAX_EVENTS_PER_BATCH, bota_proto::MAX_PAYLOAD_LEN / 2);
    assert_eq!(MAX_RECENT_EVENTS, 64);
    assert_eq!(HISTORY_TICKS, 480);
    assert_eq!(MAX_HISTORY_SUMMARIES, 481);
    assert_eq!(HISTORY_AGES, [480, 240, 120, 60, 30, 15, 0]);
}

#[test]
fn tracker_rejects_invalid_hero_team_map_seats_and_tick_rate_with_exact_messages() {
    let mut info = match_info();
    info.picks[0].hero = HeroId(1);
    assert_new_error(
        &info,
        "own slot picked HeroId(1), expected Shadow Fiend HeroId(2)",
    );

    let mut info = match_info();
    info.picks[0].team = Team::Neutral;
    assert_new_error(&info, "own slot has non-playable team Neutral");

    let mut info = match_info();
    info.map = MapId(2);
    assert_new_error(
        &info,
        "unsupported map MapId(2); expected MapId(0) or MapId(1)",
    );

    let mut info = match_info();
    info.picks.clear();
    assert_new_error(&info, "MatchInfo.picks has 0 seats; expected 1..=10");

    let mut info = match_info();
    info.picks = (0..=10)
        .map(|slot| Pick {
            slot: SlotId(slot),
            team: if slot % 2 == 0 {
                Team::Radiant
            } else {
                Team::Dire
            },
            hero: SHADOW_FIEND,
        })
        .collect();
    assert_new_error(&info, "MatchInfo.picks has 11 seats; expected 1..=10");

    let mut info = match_info();
    info.tick_rate = 0;
    assert_new_error(&info, "MatchInfo.tick_rate must be positive");
}

#[test]
fn tracker_rejects_bounded_static_inputs_before_cloning() {
    let mut info = match_info();
    info.trees = vec![Vec2::ZERO; MAX_STATIC_TREES + 1];
    assert_new_error(&info, "MatchInfo.trees has 4097 entries; limit is 4096");

    let mut info = match_info();
    info.shop = vec![shop_entry(); MAX_SHOP_ITEMS + 1];
    assert_new_error(&info, "MatchInfo.shop has 65 entries; limit is 64");

    let mut info = match_info();
    info.shop[0].components = vec![ItemId(1); MAX_SHOP_COMPONENTS + 1];
    assert_new_error(&info, "ShopEntry.components has 17 entries; limit is 16");

    let mut info = match_info();
    info.terrain_cells = MAX_TERRAIN_AXIS + 1;
    assert_new_error(&info, "MatchInfo.terrain_cells is 513; expected 1..=512");

    let mut info = match_info();
    info.terrain_rle = vec![(1, 0); MAX_TERRAIN_RUNS + 1];
    assert_new_error(
        &info,
        "MatchInfo.terrain_rle has 262145 entries; limit is 262144",
    );

    let mut info = match_info();
    info.opaque_cells = vec![(0, 0); MAX_OPAQUE_CELLS + 1];
    assert_new_error(
        &info,
        "MatchInfo.opaque_cells has 262145 entries; limit is 262144",
    );

    let mut info = match_info();
    info.terrain_rle = vec![(0, 0), (1, 0)];
    assert_new_error(&info, "MatchInfo.terrain_rle entry 0 has zero run length");

    let mut info = match_info();
    info.opaque_cells = vec![(8, 0)];
    assert_new_error(
        &info,
        "MatchInfo.opaque_cells contains (8, 0) outside 8x8 terrain",
    );
}

#[test]
fn tracker_accepts_exact_map_boundary_and_rejects_every_position_source_atomically() {
    let maximum = ((8 * crate::TERRAIN_CELL_SIZE) << Fixed::FRAC_BITS) - 1;
    let boundary = Vec2 {
        x: Fixed { raw: maximum },
        y: Fixed { raw: maximum },
    };
    let mut info = match_info();
    info.trees = vec![boundary];
    StateTracker::new(SlotId(0), &info).expect("inclusive static boundary");
    info.trees[0].x.raw += 1;
    assert_new_error(
        &info,
        "MatchInfo.trees[0] position raw (33554432, 33554431) is outside 0..=33554431",
    );

    let mut boundary_view = world_view(1);
    for unit in &mut boundary_view.units {
        unit.pos = boundary;
    }
    boundary_view.projectiles[0].pos = boundary;
    boundary_view.planted_trees[0] = boundary;
    boundary_view.loot[0].pos = boundary;
    new_tracker()
        .observe_snapshot(&boundary_view)
        .expect("inclusive dynamic boundaries");

    for (change, field) in [
        (set_unit_outside as fn(&mut WorldView), "WorldView.units"),
        (set_projectile_outside, "WorldView.projectiles"),
        (set_planted_tree_outside, "WorldView.planted_trees"),
        (set_loot_outside, "WorldView.loot"),
    ] {
        let mut tracker = tracker_with_first_tick(1);
        let mut invalid = world_view(2);
        change(&mut invalid);
        let expected = format!("{field}[0] position raw (-1, 0) is outside 0..=33554431");
        assert_observe_error(&mut tracker, &invalid, &expected);
        assert_eq!(tracker.current().expect("prior snapshot").tick, 1);
        assert_eq!(tracker.history()[6].tick, 1);
        assert_eq!(
            tracker
                .entity(entity(1, 1))
                .expect("prior hero")
                .last_seen_tick,
            1
        );
    }

    for (change, field) in [
        (set_unit_above as fn(&mut WorldView), "WorldView.units"),
        (set_projectile_above, "WorldView.projectiles"),
        (set_planted_tree_above, "WorldView.planted_trees"),
        (set_loot_above, "WorldView.loot"),
    ] {
        let mut tracker = tracker_with_first_tick(1);
        let mut invalid = world_view(2);
        change(&mut invalid);
        let expected =
            format!("{field}[0] position raw (33554432, 33554431) is outside 0..=33554431");
        assert_observe_error(&mut tracker, &invalid, &expected);
        assert_eq!(tracker.current().expect("prior snapshot").tick, 1);
    }
}

#[test]
fn tracker_rejects_ambiguous_own_hero_and_courier_bodies_with_exact_errors() {
    let mut hero_view = world_view(1);
    let mut extra_hero = hero();
    extra_hero.id = entity(3, 7);
    hero_view.units.push(extra_hero);
    hero_view.units.sort_by_key(|unit| unit.id);
    assert_observe_error(
        &mut new_tracker(),
        &hero_view,
        "own slot has visible hero EntityId(3, 7) besides scoreboard hero EntityId(1, 1)",
    );

    let mut dead_view = world_view(1);
    dead_view.players[0].unit = None;
    dead_view.players[0].kit = Some(dead_kit());
    assert_observe_error(
        &mut new_tracker(),
        &dead_view,
        "own slot has visible hero EntityId(1, 1) while scoreboard body is absent",
    );

    let mut courier_view = world_view(1);
    let mut extra_courier = courier();
    extra_courier.id = entity(4, 9);
    courier_view.units.push(extra_courier);
    courier_view.units.sort_by_key(|unit| unit.id);
    assert_observe_error(
        &mut new_tracker(),
        &courier_view,
        "own slot has ambiguous couriers EntityId(2, 1) and EntityId(4, 9)",
    );
}

#[test]
fn public_visibility_uses_terrain_opaque_line_and_exact_radius_boundary() {
    let mut view = world_view(1);
    view.units[0].vision_radius = Fixed::from_int(200);
    view.units[1].vision_radius = Fixed::ZERO;
    let mut clear_info = match_info();
    clear_info.opaque_cells.clear();
    let mut clear = StateTracker::new(SlotId(0), &clear_info).expect("clear tracker");
    clear.observe_snapshot(&view).expect("snapshot");
    let target = Vec2::from_ints(130, 20);
    assert!(clear.position_visible_to_own_seat(target));

    let mut blocked_info = clear_info;
    blocked_info.opaque_cells = vec![(1, 0)];
    let mut blocked = StateTracker::new(SlotId(0), &blocked_info).expect("blocked tracker");
    blocked.observe_snapshot(&view).expect("snapshot");
    assert!(!blocked.position_visible_to_own_seat(target));

    let mut elevated_info = match_info();
    elevated_info.opaque_cells.clear();
    elevated_info.terrain_rle = vec![(1, 0x80), (1, 0x81), (62, 0x80)];
    let mut elevated = StateTracker::new(SlotId(0), &elevated_info).expect("elevated tracker");
    elevated.observe_snapshot(&view).expect("snapshot");
    assert!(!elevated.position_visible_to_own_seat(target));

    let mut corner_view = view.clone();
    corner_view.units[0].pos = Vec2::from_ints(32, 32);
    let mut corner_info = match_info();
    corner_info.opaque_cells = vec![(1, 0)];
    let mut corner = StateTracker::new(SlotId(0), &corner_info).expect("corner tracker");
    corner.observe_snapshot(&corner_view).expect("snapshot");
    assert!(!corner.position_visible_to_own_seat(Vec2::from_ints(96, 96)));

    let hero = blocked.own_hero().expect("hero");
    let radius_edge = hero.pos + Vec2::from_ints(hero.vision_radius.to_int(), 0);
    assert!(clear.position_visible_to_own_seat(radius_edge));
    assert!(!clear.position_visible_to_own_seat(radius_edge + Vec2::from_ints(1, 0)));
}

fn set_unit_outside(view: &mut WorldView) {
    view.units[0].pos = raw_position(-1, 0);
}

fn set_projectile_outside(view: &mut WorldView) {
    view.projectiles[0].pos = raw_position(-1, 0);
}

fn set_planted_tree_outside(view: &mut WorldView) {
    view.planted_trees[0] = raw_position(-1, 0);
}

fn set_loot_outside(view: &mut WorldView) {
    view.loot[0].pos = raw_position(-1, 0);
}

fn set_unit_above(view: &mut WorldView) {
    view.units[0].pos = raw_position(33_554_432, 33_554_431);
}

fn set_projectile_above(view: &mut WorldView) {
    view.projectiles[0].pos = raw_position(33_554_432, 33_554_431);
}

fn set_planted_tree_above(view: &mut WorldView) {
    view.planted_trees[0] = raw_position(33_554_432, 33_554_431);
}

fn set_loot_above(view: &mut WorldView) {
    view.loot[0].pos = raw_position(33_554_432, 33_554_431);
}

const fn raw_position(x: i32, y: i32) -> Vec2 {
    Vec2 {
        x: Fixed { raw: x },
        y: Fixed { raw: y },
    }
}

#[test]
fn tracker_rejects_wrong_viewer_tick_and_missing_player_with_exact_messages() {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.viewer = Some(Team::Dire);
    assert_observe_error(
        &mut tracker,
        &view,
        "snapshot viewer Some(Dire) differs from tracker team Radiant",
    );

    tracker
        .observe_snapshot(&world_view(1))
        .expect("first snapshot");
    assert_observe_error(
        &mut tracker,
        &world_view(1),
        "snapshot tick 1 does not follow current tick 1",
    );

    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.players.remove(0);
    assert_observe_error(
        &mut tracker,
        &view,
        "snapshot players has no MatchInfo pick SlotId(0)",
    );
}

#[test]
fn tracker_rejects_scoreboard_rows_not_in_exact_match_roster_atomically() {
    assert_roster_error(
        |view| {
            view.players.remove(1);
        },
        "snapshot players has no MatchInfo pick SlotId(1)",
    );
    assert_roster_error(
        |view| {
            let mut extra = enemy_player();
            extra.slot = SlotId(2);
            view.players.push(extra);
        },
        "snapshot players contains unknown SlotId(2)",
    );
    assert_roster_error(
        |view| view.players[1].slot = SlotId(2),
        "snapshot players contains unknown SlotId(2)",
    );
    assert_roster_error(
        |view| view.players[1].team = Team::Radiant,
        "snapshot player SlotId(1) team Radiant differs from MatchInfo pick Dire",
    );
    assert_roster_error(
        |view| view.players[1].hero = HeroId(1),
        "snapshot player SlotId(1) HeroId(1) differs from MatchInfo pick HeroId(2)",
    );
    assert_roster_error(
        |view| view.players.swap(0, 1),
        "snapshot players are not sorted by SlotId",
    );
}

#[test]
fn tracker_distinguishes_entity_generations_and_sorts_by_full_handle() {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.units.push(creep(entity(20, 1), Team::Dire, 100, 200));
    view.units.push(creep(entity(20, 2), Team::Dire, 100, 200));
    view.units.sort_by_key(|unit| unit.id);

    tracker.observe_snapshot(&view).expect("snapshot");

    assert!(tracker.entity(entity(20, 1)).is_some());
    assert!(tracker.entity(entity(20, 2)).is_some());
    assert!(
        tracker
            .entities()
            .windows(2)
            .all(|pair| pair[0].id < pair[1].id)
    );
}

#[test]
fn tracker_records_exact_velocity_and_resource_deltas_across_skipped_ticks() {
    let mut tracker = new_tracker();
    tracker
        .observe_snapshot(&world_view(1))
        .expect("first snapshot");
    let mut next = world_view(6);
    next.units[0].pos = Vec2::from_ints(16, 24);
    next.units[0].hp = 875;
    next.units[0].mana = 360;

    tracker.observe_snapshot(&next).expect("skipped snapshot");

    let hero = tracker.entity(entity(1, 1)).expect("hero track");
    let velocity = hero.velocity.expect("velocity estimate");
    assert_eq!(velocity.delta, Vec2::from_ints(6, 4));
    assert_eq!(velocity.elapsed_ticks, 5);
    assert_eq!(hero.hp_delta, -125);
    assert_eq!(hero.mana_delta, -40);
    assert_eq!(hero.previous_seen_tick, 1);
    assert_eq!(hero.last_seen_tick, 6);
}

#[test]
fn tracker_remembers_invisible_entity_through_history_boundary_then_expires_it() {
    let remembered = entity(20, 1);
    let mut tracker = new_tracker();
    let mut first = world_view(1);
    first.units.push(creep(remembered, Team::Dire, 100, 200));
    first.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&first).expect("visible snapshot");

    tracker
        .observe_snapshot(&world_view(481))
        .expect("boundary snapshot");
    assert!(
        !tracker
            .entity(remembered)
            .expect("remembered at age 480")
            .visible
    );

    tracker
        .observe_snapshot(&world_view(482))
        .expect("expiry snapshot");
    assert!(tracker.entity(remembered).is_none());
}

#[test]
fn tracker_evicts_the_complete_oldest_invisible_tick_cohort() {
    let mut tracker = new_tracker();
    let mut first = world_view(1);
    for index in 3..=MAX_TRACKED_ENTITIES as u32 {
        first
            .units
            .push(creep(entity(index, 1), Team::Dire, 100, 100));
    }
    first.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&first).expect("full tracker");

    let mut second = world_view(2);
    second
        .units
        .push(creep(entity(300, 1), Team::Dire, 100, 100));
    second.units.sort_by_key(|unit| unit.id);
    tracker
        .observe_snapshot(&second)
        .expect("evicting snapshot");

    assert_eq!(tracker.entities().len(), 3);
    for index in 3..=MAX_TRACKED_ENTITIES as u32 {
        assert!(tracker.entity(entity(index, 1)).is_none());
    }
    assert!(tracker.entity(entity(300, 1)).expect("new track").visible);
    assert!(tracker.entity(entity(1, 1)).expect("own hero").visible);
}

#[test]
fn tracker_evicts_only_complete_oldest_cohorts_needed_for_the_incoming_snapshot() {
    let mut tracker = new_tracker();
    let mut first = world_view(1);
    for index in 3..=MAX_TRACKED_ENTITIES as u32 {
        first
            .units
            .push(creep(entity(index, 1), Team::Dire, 100, 100));
    }
    first.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&first).expect("full tracker");

    let mut second = world_view(2);
    second.units.extend(
        (100..=MAX_TRACKED_ENTITIES as u32)
            .map(|index| creep(entity(index, 1), Team::Dire, 100, 100)),
    );
    second.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&second).expect("newer cohort");

    let mut third = second;
    third.tick = 3;
    third
        .units
        .extend((300..397).map(|index| creep(entity(index, 1), Team::Dire, 100, 100)));
    third.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&third).expect("evicting snapshot");

    assert_eq!(tracker.entities().len(), MAX_TRACKED_ENTITIES);
    assert!(tracker.entity(entity(3, 1)).is_none());
    assert!(tracker.entity(entity(99, 1)).is_none());
    assert!(
        tracker
            .entity(entity(100, 1))
            .expect("newer cohort")
            .velocity
            .is_some()
    );
    assert!(
        tracker
            .entity(entity(396, 1))
            .expect("incoming cohort")
            .visible
    );
}

#[test]
fn tracker_rejects_visible_unit_input_above_hard_safe_cap() {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    for index in 3..=257 {
        view.units
            .push(creep(entity(index, 1), Team::Dire, 100, 100));
    }
    view.units.sort_by_key(|unit| unit.id);

    assert_observe_error(
        &mut tracker,
        &view,
        "WorldView.units has 257 entries; limit is 256",
    );
    assert!(tracker.current().is_none());
}

#[test]
fn tracker_updates_damage_heal_death_cast_and_possible_attack_observations() {
    let attacker = entity(1, 1);
    let target = entity(20, 1);
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.units.push(creep(target, Team::Dire, 100, 100));
    view.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&view).expect("snapshot");
    let events = [
        EventKind::Damaged {
            source: Some(attacker),
            target,
            amount: 75,
            kind: DamageKind::Physical,
            crit: false,
        },
        EventKind::Healed {
            source: Some(attacker),
            target: attacker,
            amount: 20,
        },
        EventKind::Died {
            unit: target,
            killer: Some(attacker),
            denied: false,
        },
        EventKind::AbilityCast {
            caster: attacker,
            ability: AbilityId(13),
        },
    ];

    tracker.observe_events(1, &events).expect("events");

    let attacker_track = tracker.entity(attacker).expect("attacker");
    assert_eq!(
        attacker_track
            .last_damage_dealt
            .expect("damage")
            .counterpart,
        Some(target)
    );
    assert_eq!(attacker_track.last_heal_received.expect("heal").amount, 20);
    assert_eq!(
        attacker_track.last_ability_cast.expect("cast").ability,
        AbilityId(13)
    );
    assert!(
        attacker_track.last_possible_attack_landed.is_none(),
        "a same-tick ability cast keeps physical damage ambiguous rather than calling it an attack"
    );
    let target_track = tracker.entity(target).expect("target");
    assert_eq!(target_track.last_damage_taken.expect("damage").amount, 75);
    assert_eq!(
        target_track.last_death.expect("death").killer,
        Some(attacker)
    );
    assert!(!target_track.visible);
}

#[test]
fn tracker_marks_only_unattributed_physical_noncritical_damage_as_possible_attack() {
    let attacker = entity(1, 1);
    let target = entity(20, 1);
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.units.push(creep(target, Team::Dire, 100, 100));
    view.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&view).expect("snapshot");

    tracker
        .observe_events(
            1,
            &[EventKind::Damaged {
                source: Some(attacker),
                target,
                amount: 30,
                kind: DamageKind::Physical,
                crit: false,
            }],
        )
        .expect("damage");

    let possible = tracker
        .entity(attacker)
        .expect("attacker")
        .last_possible_attack_landed
        .expect("possible attack");
    assert_eq!(possible.tick, 1);
    assert_eq!(possible.target, target);
}

#[test]
fn tracker_retains_unknown_event_ids_without_creating_entities() {
    let mut tracker = new_tracker();
    tracker.observe_snapshot(&world_view(1)).expect("snapshot");
    let unknown = entity(999, 7);

    tracker
        .observe_events(
            1,
            &[EventKind::Died {
                unit: unknown,
                killer: None,
                denied: false,
            }],
        )
        .expect("unknown event");

    assert!(tracker.entity(unknown).is_none());
    assert_eq!(tracker.recent_events().len(), 1);
    assert_eq!(
        tracker.recent_events()[0].kind,
        EventKind::Died {
            unit: unknown,
            killer: None,
            denied: false,
        }
    );
}

#[test]
fn tracker_accepts_event_stream_independently_of_snapshots_and_rejects_old_ticks() {
    let mut tracker = new_tracker();
    let one = EventKind::ItemBought {
        slot: SlotId(0),
        item: ItemId(1),
    };
    tracker
        .observe_events(10, std::slice::from_ref(&one))
        .expect("event before snapshot");
    tracker
        .observe_snapshot(&world_view(20))
        .expect("newer snapshot");
    tracker
        .observe_events(11, &[])
        .expect("empty event batch older than snapshot");
    tracker
        .observe_events(30, std::slice::from_ref(&one))
        .expect("event batch newer than snapshot");

    let error = tracker
        .observe_events(30, &[])
        .expect_err("equal event tick rejected");
    assert_eq!(
        error.to_string(),
        "event batch tick 30 must be greater than last event tick 30"
    );
    let error = tracker
        .observe_events(29, std::slice::from_ref(&one))
        .expect_err("older event tick rejected");
    assert_eq!(
        error.to_string(),
        "event batch tick 29 must be greater than last event tick 30"
    );
    assert_eq!(tracker.recent_events().len(), 2);
}

#[test]
fn event_batch_limit_accepts_the_wire_bound_and_rejects_the_next_count() {
    crate::tracker::validate_event_batch_limit(MAX_EVENTS_PER_BATCH).expect("exact boundary");

    let error = crate::tracker::validate_event_batch_limit(MAX_EVENTS_PER_BATCH + 1)
        .expect_err("count above boundary");

    assert_eq!(
        error.to_string(),
        "event batch has 2097153 entries; limit is 2097152"
    );
}

#[test]
fn tracker_processes_a_small_functional_event_batch() {
    let mut tracker = new_tracker();
    tracker.observe_snapshot(&world_view(1)).expect("snapshot");
    let events: Vec<_> = (0..3)
        .map(|index| EventKind::ItemBought {
            slot: SlotId(0),
            item: ItemId(index as u16),
        })
        .collect();

    tracker.observe_events(2, &events).expect("small batch");

    assert_eq!(tracker.recent_events().len(), 3);
    assert_eq!(
        tracker.recent_events().front().expect("front").kind,
        EventKind::ItemBought {
            slot: SlotId(0),
            item: ItemId(0),
        }
    );
    assert_eq!(
        tracker.recent_events().back().expect("back").kind,
        EventKind::ItemBought {
            slot: SlotId(0),
            item: ItemId(2),
        }
    );
}

#[test]
fn tracker_history_selects_exact_tick_ages_and_falls_back_to_oldest() {
    let mut tracker = new_tracker();
    for tick in 1..=481 {
        tracker
            .observe_snapshot(&world_view(tick))
            .expect("snapshot");
    }

    assert_eq!(
        tracker.history().map(|summary| summary.tick),
        [1, 241, 361, 421, 451, 466, 481]
    );

    let mut early = tracker_with_first_tick(7);
    assert_eq!(early.history().map(|summary| summary.tick), [7; 7]);
    early
        .observe_snapshot(&world_view(22))
        .expect("second snapshot");
    assert_eq!(
        early.history().map(|summary| summary.tick),
        [7, 7, 7, 7, 7, 7, 22]
    );
}

#[test]
fn tracker_history_uses_game_ticks_across_snapshot_call_gaps() {
    let mut tracker = new_tracker();
    for tick in 1..=600 {
        tracker
            .observe_snapshot(&world_view(tick))
            .expect("snapshot");
    }
    tracker
        .observe_snapshot(&world_view(901))
        .expect("gap snapshot");

    assert_eq!(
        tracker.history().map(|summary| summary.tick),
        [421, 600, 600, 600, 600, 600, 901]
    );
}

#[test]
fn tracker_global_summary_aggregates_scoreboard_units_and_structures() {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.players[0].xp = 120;
    view.players[0].kills = 2;
    view.players[0].last_hits = 8;
    view.players[1].xp = 90;
    view.players[1].deaths = 2;
    view.players[1].denies = 3;
    view.units.push(building(entity(30, 1), Team::Radiant, 700));
    view.units.push(building(entity(31, 1), Team::Dire, 650));
    view.units.sort_by_key(|unit| unit.id);

    tracker.observe_snapshot(&view).expect("summary snapshot");

    let summary = tracker.history()[6];
    assert_eq!(summary.own_hp, 1_000);
    assert_eq!(summary.own_max_hp, 1_000);
    assert!(summary.own_hp_present);
    assert_eq!(summary.own_mana, 400);
    assert_eq!(summary.own_max_mana, 400);
    assert!(summary.own_mana_present);
    assert_eq!(summary.own_level, 3);
    assert_eq!(summary.own_gold, 600);
    assert_eq!(summary.visible_allied_units, 3);
    assert_eq!(summary.visible_enemy_units, 1);
    assert_eq!(summary.allied.xp, 120);
    assert_eq!(summary.allied.levels, 3);
    assert_eq!(summary.allied.kills, 2);
    assert_eq!(summary.allied.last_hits, 8);
    assert_eq!(summary.enemy.xp, 90);
    assert_eq!(summary.enemy.levels, 2);
    assert_eq!(summary.enemy.deaths, 2);
    assert_eq!(summary.enemy.denies, 3);
    assert_eq!(summary.allied_structure_hp, 700);
    assert_eq!(summary.enemy_structure_hp, 650);
    assert!(summary.destroyed_structures_present);
    assert_eq!(summary.allied_structures_destroyed, 0);
    assert_eq!(summary.enemy_structures_destroyed, 0);
}

#[test]
fn tracker_summary_marks_live_resources_missing_and_counts_observed_structure_removal() {
    let mut tracker = new_tracker();
    let mut first = world_view(1);
    first
        .units
        .push(building(entity(30, 1), Team::Radiant, 700));
    first.units.push(building(entity(31, 1), Team::Dire, 650));
    first.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&first).expect("baseline");

    let mut second = world_view(2);
    second.players[0].unit = None;
    second.players[0].respawn_left = 30;
    second.players[0].kit = Some(bota_proto::Kit {
        abilities: Vec::new(),
        items: vec![None; 9],
    });
    second.units.retain(|unit| unit.id != entity(1, 1));
    second
        .units
        .push(building(entity(30, 1), Team::Radiant, 700));
    second.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&second).expect("removal");

    let summary = tracker.history()[6];
    assert!(!summary.own_hp_present);
    assert!(!summary.own_mana_present);
    assert_eq!(summary.own_hp, 0);
    assert_eq!(summary.own_mana, 0);
    assert_eq!(summary.allied_structures_destroyed, 0);
    assert_eq!(summary.enemy_structures_destroyed, 1);
}

#[test]
fn tracker_marks_destroyed_structure_count_missing_without_pregame_baseline() {
    let mut tracker = new_tracker();
    let mut joined = world_view(100);
    joined
        .units
        .push(building(entity(30, 1), Team::Radiant, 700));
    joined.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&joined).expect("joined snapshot");

    tracker
        .observe_snapshot(&world_view(101))
        .expect("later snapshot");

    let summary = tracker.history()[6];
    assert!(!summary.destroyed_structures_present);
    assert_eq!(summary.allied_structures_destroyed, 0);
    assert_eq!(summary.enemy_structures_destroyed, 0);
}

#[test]
fn tracker_rejects_snapshot_projectile_loot_tree_and_nested_vector_bounds() {
    assert_snapshot_limit(
        |view| view.projectiles = vec![projectile(); MAX_PROJECTILES + 1],
        "WorldView.projectiles has 33 entries; limit is 32",
    );
    assert_snapshot_limit(
        |view| view.loot = vec![loot(); MAX_LOOT + 1],
        "WorldView.loot has 17 entries; limit is 16",
    );
    assert_snapshot_limit(
        |view| view.planted_trees = vec![Vec2::ZERO; MAX_PLANTED_TREES + 1],
        "WorldView.planted_trees has 4097 entries; limit is 4096",
    );
    assert_snapshot_limit(
        |view| view.units[0].abilities = vec![ability(); SHADOW_FIEND_ABILITY_SLOTS + 1],
        "UnitView.abilities has 7 entries; limit is 6",
    );
    assert_snapshot_limit(
        |view| view.units[0].items = vec![None; 10],
        "UnitView.items has 10 entries; limit is 9",
    );
    assert_snapshot_limit(
        |view| view.units[0].effects = vec![effect(); MAX_EFFECTS_PER_UNIT + 1],
        "UnitView.effects has 33 entries; limit is 32",
    );
}

#[test]
fn tracker_exposes_own_player_hero_courier_and_current_map_state() {
    let mut tracker = new_tracker();
    let view = world_view(1);

    tracker.observe_snapshot(&view).expect("snapshot");

    assert_eq!(tracker.own_player().expect("player").slot, SlotId(0));
    assert_eq!(tracker.own_hero().expect("hero").id, entity(1, 1));
    assert_eq!(tracker.own_courier().expect("courier").id, entity(2, 1));
    let current = tracker.current().expect("current");
    assert_eq!(current.projectiles.len(), 1);
    assert_eq!(current.loot.len(), 1);
    assert_eq!(current.felled_trees, [0]);
    assert_eq!(current.planted_trees, [Vec2::from_ints(6, 7)]);
}

#[test]
fn tracker_returns_no_own_hero_while_dead_and_keeps_courier_available() {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    view.players[0].unit = None;
    view.players[0].kit = Some(dead_kit());
    view.players[0].respawn_left = 20;
    view.units.retain(|unit| unit.id != entity(1, 1));

    tracker.observe_snapshot(&view).expect("dead snapshot");

    assert!(tracker.own_hero().is_none());
    assert_eq!(tracker.own_courier().expect("courier").id, entity(2, 1));
    assert_eq!(tracker.history()[6].own_respawn_left, 20);
}

fn assert_new_error(info: &MatchInfo, expected: &str) {
    let error = StateTracker::new(SlotId(0), info)
        .err()
        .expect("invalid match info must fail");
    assert_eq!(error.to_string(), expected);
}

fn assert_observe_error(tracker: &mut StateTracker, view: &WorldView, expected: &str) {
    let error = tracker
        .observe_snapshot(view)
        .expect_err("invalid snapshot must fail");
    assert_eq!(error.to_string(), expected);
}

fn assert_snapshot_limit(change: impl FnOnce(&mut WorldView), expected: &str) {
    let mut tracker = new_tracker();
    let mut view = world_view(1);
    change(&mut view);
    assert_observe_error(&mut tracker, &view, expected);
}

fn assert_roster_error(change: impl FnOnce(&mut WorldView), expected: &str) {
    let mut tracker = tracker_with_first_tick(1);
    let mut view = world_view(2);
    change(&mut view);

    assert_observe_error(&mut tracker, &view, expected);
    assert_eq!(tracker.current().expect("unchanged snapshot").tick, 1);
    assert_eq!(tracker.history()[6].tick, 1);
}

fn new_tracker() -> StateTracker {
    StateTracker::new(SlotId(0), &match_info()).expect("fixture tracker")
}

fn tracker_with_first_tick(tick: u32) -> StateTracker {
    let mut tracker = new_tracker();
    tracker
        .observe_snapshot(&world_view(tick))
        .expect("first snapshot");
    tracker
}

fn match_info() -> MatchInfo {
    MatchInfo {
        match_id: 77,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 90,
        trees: vec![Vec2::from_ints(4, 5)],
        terrain_cells: 8,
        terrain_rle: vec![(64, 0x80)],
        opaque_cells: vec![(0, 0)],
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
        shop: vec![shop_entry()],
    }
}

fn shop_entry() -> ShopEntry {
    ShopEntry {
        id: ItemId(1),
        cost: 50,
        components: Vec::new(),
    }
}

fn world_view(tick: u32) -> WorldView {
    let mut units = vec![hero(), courier()];
    units.sort_by_key(|unit| unit.id);
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units,
        projectiles: vec![projectile()],
        players: vec![own_player(), enemy_player()],
        felled_trees: vec![0],
        planted_trees: vec![Vec2::from_ints(6, 7)],
        loot: vec![loot()],
    }
}

fn hero() -> UnitView {
    UnitView {
        id: entity(1, 1),
        kind: UnitKind::Hero,
        team: Team::Radiant,
        pos: Vec2::from_ints(10, 20),
        facing: Angle { brads: 0 },
        hp: 1_000,
        max_hp: 1_000,
        mana: 400,
        max_mana: 400,
        move_speed: Fixed::from_int(300),
        attack_damage: 55,
        attack_range: Fixed::from_int(500),
        attack_interval: 30,
        attack_speed: 100,
        armor: Fixed::from_int(2),
        magic_resist: Fixed::from_ratio(1, 4),
        radius: Fixed::from_int(24),
        vision_radius: Fixed::from_int(1_800),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::all(20),
        primary: Some(Attribute::Agility),
        hero: Some(SHADOW_FIEND),
        owner: Some(SlotId(0)),
        level: 3,
        abilities: vec![ability(); SHADOW_FIEND_ABILITY_SLOTS],
        items: vec![None; 9],
        effects: vec![effect()],
    }
}

fn courier() -> UnitView {
    UnitView {
        id: entity(2, 1),
        kind: UnitKind::Courier,
        team: Team::Radiant,
        pos: Vec2::from_ints(8, 8),
        facing: Angle { brads: 1 },
        hp: 100,
        max_hp: 100,
        mana: 0,
        max_mana: 0,
        move_speed: Fixed::from_int(350),
        attack_damage: 0,
        attack_range: Fixed::ZERO,
        attack_interval: 0,
        attack_speed: 0,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(16),
        vision_radius: Fixed::from_int(400),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::ZERO,
        primary: None,
        hero: None,
        owner: Some(SlotId(0)),
        level: 0,
        abilities: Vec::new(),
        items: vec![None; 6],
        effects: Vec::new(),
    }
}

fn creep(id: EntityId, team: Team, hp: i32, mana: i32) -> UnitView {
    UnitView {
        id,
        kind: UnitKind::CreepMelee,
        team,
        pos: Vec2::from_ints(100, 200),
        facing: Angle { brads: 2 },
        hp,
        max_hp: 500,
        mana,
        max_mana: 200,
        move_speed: Fixed::from_int(300),
        attack_damage: 20,
        attack_range: Fixed::from_int(100),
        attack_interval: 40,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(20),
        vision_radius: Fixed::from_int(600),
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

fn building(id: EntityId, team: Team, hp: i32) -> UnitView {
    UnitView {
        id,
        kind: UnitKind::Tower,
        team,
        pos: Vec2::from_ints(300, 400),
        facing: Angle { brads: 3 },
        hp,
        max_hp: 1_000,
        mana: 0,
        max_mana: 0,
        move_speed: Fixed::ZERO,
        attack_damage: 100,
        attack_range: Fixed::from_int(700),
        attack_interval: 30,
        attack_speed: 100,
        armor: Fixed::from_int(10),
        magic_resist: Fixed::from_ratio(1, 4),
        radius: Fixed::from_int(80),
        vision_radius: Fixed::from_int(1_800),
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

fn own_player() -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team: Team::Radiant,
        hero: SHADOW_FIEND,
        unit: Some(entity(1, 1)),
        level: 3,
        xp: 100,
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
        unit: None,
        level: 2,
        xp: 80,
        gold: None,
        stash: None,
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 10,
    }
}

fn ability() -> AbilityView {
    AbilityView {
        id: AbilityId(13),
        level: 1,
        max_level: 4,
        cooldown_left: 0,
        mana_cost: 75,
        range: 200,
        aim: Aim::Point,
        passive: false,
        on: false,
        can_level: false,
    }
}

fn item() -> ItemView {
    ItemView {
        id: ItemId(1),
        charges: Some(1),
        cooldown_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: false,
    }
}

fn effect() -> EffectView {
    EffectView {
        id: EffectId(11),
        ticks_left: None,
        stacks: Some(3),
    }
}

fn projectile() -> ProjectileView {
    ProjectileView {
        id: entity(40, 1),
        pos: Vec2::from_ints(20, 30),
        facing: Angle { brads: 4 },
        team: Team::Radiant,
        ability: None,
    }
}

fn loot() -> LootView {
    LootView {
        id: entity(50, 1),
        pos: Vec2::from_ints(30, 40),
        item: ItemId(1),
        charges: Some(1),
    }
}

fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}

fn dead_kit() -> Kit {
    Kit {
        abilities: vec![ability(); SHADOW_FIEND_ABILITY_SLOTS],
        items: vec![Some(item()); 9],
    }
}
