use bota_proto::{
    AbilityId, AbilitySlot, AbilityView, Aim, Angle, Attribute, Attributes, EffectId, EffectView,
    EntityId, Fixed, ItemId, ItemSlot, ItemView, MapId, MatchInfo, Order, Pick, PlayerView,
    ShopEntry, SlotId, StatusFlags, Target, Team, TickMode, UnitKind, UnitView, Vec2, WorldView,
};

use crate::{
    ActionError, ActionKind, ActionSpace, ActionTarget, ControlledUnit, EntityIndex, ItemReadiness,
    PointIndex, SHADOW_FIEND, ShopIndex, StateTracker, StructuredAction,
};

const MANGO: ItemId = ItemId(42);
const HERO: EntityId = EntityId {
    idx: 1,
    generation: 1,
};

#[test]
fn schema_five_preserves_mango_mana_and_anonymous_effect_ordering() {
    assert_eq!(crate::ACTION_SCHEMA_VERSION, 5);
    assert_eq!(crate::ACTION_SCHEMA_HASH, 10_658_390_830_565_586_343);
    for semantics in [
        "buy_decode=root_or_first_missing_leaf",
        "buy_mango=item42_one_charge_repeatable",
        "target_none_provable_positive_own_mana_deficit",
        "mana_legality=all_casts_and_uses_conservative_own_wire_mana_affordability",
        "raze_legality=mechanical_only_empty_and_beneficial_allowed",
        "entity_order=active_effect15_max_lexicographic_stacks_remaining_then_guarded13_inspired14_timers_then_prior_received_manual_hp_mana_report_semantics_before_opaque_id",
    ] {
        assert!(
            crate::ACTION_SCHEMA_DESCRIPTOR.contains(semantics),
            "{semantics}"
        );
    }
}

#[test]
fn mango_catalog_entry_43_decodes_without_reindexing_older_items() {
    let space = action_space(&view());

    assert_eq!(space.shop_candidates().len(), 43);
    for (index, candidate) in space.shop_candidates().iter().enumerate() {
        assert_eq!(usize::from(candidate.item.0), index);
    }
    assert_eq!(space.shop_candidates()[42].cost, 65);
    assert_eq!(
        space.decode(buy()).expect("buy").expect("order").order,
        Order::Buy { item: MANGO }
    );
}

#[test]
fn mango_purchase_remains_repeatable_with_existing_full_or_partial_stacks() {
    for charges in 1..=3 {
        let mut view = view();
        view.units[0].items[0] = Some(mango(Some(charges)));
        view.players[0].stash.as_mut().expect("stash")[0] = Some(mango(Some(charges)));

        let space = action_space(&view);

        assert!(space.allows(buy()), "charges={charges}");
        assert_eq!(
            space.decode(buy()).expect("buy").expect("order").order,
            Order::Buy { item: MANGO }
        );
    }
}

#[test]
fn mango_purchase_merges_into_full_home_inventory_including_backpack() {
    for slot in 0..9 {
        let mut view = full_view();
        view.units[0].items[slot] = Some(mango(Some(2)));

        let space = action_space(&view);

        assert!(space.allows(buy()), "slot={slot}");
    }
}

#[test]
fn mango_purchase_merges_into_full_stash_at_home_or_remotely() {
    for remote in [false, true] {
        for slot in 0..6 {
            let mut view = full_view();
            if remote {
                view.units[0].pos = Vec2::from_ints(5_000, 5_000);
            }
            view.players[0].stash.as_mut().expect("stash")[slot] = Some(mango(Some(1)));

            assert!(
                action_space(&view).allows(buy()),
                "remote={remote}, slot={slot}"
            );
        }
    }
}

#[test]
fn mango_remote_purchase_cannot_use_bag_space_or_bag_merge_room() {
    for partial_stack in [false, true] {
        let mut view = full_view();
        view.units[0].pos = Vec2::from_ints(5_000, 5_000);
        view.units[0].items[0] = partial_stack.then(|| mango(Some(1)));

        assert_masked(&action_space(&view), buy(), ActionKind::Buy);
    }
}

#[test]
fn mango_purchase_uses_empty_stash_when_home_bag_cannot_receive() {
    let mut view = full_view();
    view.units[0].items.fill(Some(mango(Some(3))));
    view.players[0].stash.as_mut().expect("stash")[5] = None;

    assert!(action_space(&view).allows(buy()));
}

#[test]
fn mango_purchase_rejects_full_or_invalid_merge_stacks_without_empty_slots() {
    for charges in [None, Some(0), Some(3), Some(4), Some(u8::MAX)] {
        let mut view = full_view();
        view.units[0].items[0] = Some(mango(charges));
        view.players[0].stash.as_mut().expect("stash")[0] = Some(mango(charges));

        assert_masked(&action_space(&view), buy(), ActionKind::Buy);
    }
}

#[test]
fn mango_purchase_cannot_merge_sale_marked_or_mode_incompatible_stacks() {
    for mode in [None, Some(Attribute::Strength)] {
        let mut view = full_view();
        let mut held = mango(Some(1));
        held.mode = mode;
        held.for_sale = mode.is_none();
        view.units[0].items[0] = Some(held);
        view.players[0].stash.as_mut().expect("stash")[0] = Some(held);

        assert_masked(&action_space(&view), buy(), ActionKind::Buy);
    }
}

#[test]
fn mango_purchase_can_merge_muted_or_cooling_stacks_without_laundering_use() {
    let mut view = full_view();
    let mut held = mango(Some(2));
    held.mute_left = 180;
    held.cooldown_left = 30;
    view.units[0].items[0] = Some(held);

    let space = action_space(&view);

    assert!(space.allows(buy()));
    assert_masked(&space, use_mango(0), ActionKind::Use);
}

#[test]
fn mango_purchase_requires_65_gold_even_when_a_stack_can_merge() {
    for gold in [0, 64, 65, 66] {
        let mut view = full_view();
        view.units[0].items[0] = Some(mango(Some(1)));
        view.players[0].gold = Some(gold);

        assert_eq!(action_space(&view).allows(buy()), gold >= 65, "gold={gold}");
    }
}

#[test]
fn mango_purchase_never_uses_courier_capacity_or_courier_control() {
    let mut view = full_view();
    view.units[1].items[0] = Some(mango(Some(1)));
    let space = action_space(&view);

    assert_masked(&space, buy(), ActionKind::Buy);
    assert_masked(
        &space,
        StructuredAction::Buy {
            unit: ControlledUnit::Courier,
            item: ShopIndex(42),
        },
        ActionKind::Buy,
    );
}

#[test]
fn mango_use_allows_every_provable_positive_deficit_without_strategy_threshold() {
    for mana in [0, 1, 400, 499] {
        let mut view = view();
        view.units[0].mana = mana;

        assert!(action_space(&view).allows(use_mango(0)), "mana={mana}");
    }
}

#[test]
fn mango_use_rejects_full_overfull_absent_nonpositive_or_invalid_mana_pools() {
    for (mana, maximum) in [(500, 500), (501, 500), (0, 0), (0, -1), (-1, 500), (1, 1)] {
        let mut view = view();
        view.units[0].mana = mana;
        view.units[0].max_mana = maximum;

        assert_masked(&action_space(&view), use_mango(0), ActionKind::Use);
    }
}

#[test]
fn mango_use_only_decodes_target_none() {
    let space = action_space(&view());
    let mask = space
        .use_target_mask(ControlledUnit::Hero, ItemSlot(0))
        .expect("mask");

    assert!(mask.allows_none());
    assert!(!mask.entities().contains(&true));
    assert!(!mask.points().contains(&true));
    assert_eq!(
        space
            .decode(use_mango(0))
            .expect("use")
            .expect("order")
            .order,
        Order::Use {
            slot: ItemSlot(0),
            target: Target::None
        }
    );
    for target in [
        ActionTarget::Entity(EntityIndex(0)),
        ActionTarget::Point(PointIndex(0)),
    ] {
        assert_masked(
            &space,
            StructuredAction::Use {
                unit: ControlledUnit::Hero,
                slot: ItemSlot(0),
                target,
            },
            ActionKind::Use,
        );
    }
}

#[test]
fn mango_use_is_active_only_in_inventory_slots_zero_through_five() {
    for slot in 0..15u8 {
        let mut view = view();
        view.units[0].items.fill(None);
        if slot < 9 {
            view.units[0].items[usize::from(slot)] = Some(mango(Some(1)));
        } else {
            view.players[0].stash.as_mut().expect("stash")[usize::from(slot - 9)] =
                Some(mango(Some(1)));
        }

        assert_eq!(
            action_space(&view).allows(use_mango(slot)),
            slot < 6,
            "slot={slot}"
        );
    }
}

#[test]
fn mango_use_requires_a_valid_positive_charge_count() {
    for charges in [
        None,
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(u8::MAX),
    ] {
        let mut view = view();
        view.units[0].items[0] = Some(mango(charges));

        assert_eq!(
            action_space(&view).allows(use_mango(0)),
            charges.is_some_and(|value| (1..=3).contains(&value)),
            "charges={charges:?}"
        );
    }
}

#[test]
fn mango_use_obeys_wire_mute_and_cooldown() {
    for (mute, cooldown) in [(1, 0), (180, 0), (0, 1)] {
        let mut view = view();
        let held = view.units[0].items[0].as_mut().expect("mango");
        held.mute_left = mute;
        held.cooldown_left = cooldown;

        assert_masked(&action_space(&view), use_mango(0), ActionKind::Use);
    }
}

#[test]
fn mango_use_obeys_local_backpack_mute_and_resumes_at_exact_expiry() {
    let mut view = view();
    view.units[0].items.swap(0, 6);
    let space = action_space(&view);
    let issued = space
        .decode(StructuredAction::Swap {
            unit: ControlledUnit::Hero,
            from: ItemSlot(6),
            to: ItemSlot(0),
        })
        .expect("swap")
        .expect("order");
    let mut readiness = ItemReadiness::new();
    readiness.note_sent(1, issued, &space);
    view.units[0].items.swap(0, 6);
    for (tick, allowed) in [(190, false), (191, true)] {
        view.tick = tick;
        let tracker = tracker(&view);
        let space = ActionSpace::from_tracker_with_readiness(&tracker, &readiness).expect("space");

        assert_eq!(space.allows(use_mango(0)), allowed, "tick={tick}");
    }
}

#[test]
fn mango_use_never_enables_courier_even_with_a_synthetic_mana_deficit() {
    let mut view = view();
    view.units[1].items[0] = Some(mango(Some(1)));
    view.units[1].max_mana = 500;
    let space = action_space(&view);

    assert_masked(
        &space,
        StructuredAction::Use {
            unit: ControlledUnit::Courier,
            slot: ItemSlot(0),
            target: ActionTarget::None,
        },
        ActionKind::Use,
    );
}

#[test]
fn mango_use_is_disabled_by_death_stun_and_channel_but_not_silence_root_or_disarm() {
    for (status, allowed) in [
        (StatusFlags::DEAD, false),
        (StatusFlags::STUNNED, false),
        (StatusFlags::CHANNELLING, false),
        (StatusFlags::SILENCED, true),
        (StatusFlags::ROOTED, true),
        (StatusFlags::DISARMED, true),
    ] {
        let mut view = view();
        view.units[0].statuses.bits = status;

        assert_eq!(
            action_space(&view).allows(use_mango(0)),
            allowed,
            "status={status}"
        );
    }
}

#[test]
fn empty_and_beneficial_razes_are_both_legal_without_a_stack_or_hit_shield() {
    for (slot, distance) in [(0, 200), (1, 450), (2, 700)] {
        for beneficial in [false, true] {
            let mut view = view();
            if beneficial {
                let mut enemy = unit(4, UnitKind::Hero);
                enemy.team = Team::Dire;
                enemy.pos = Vec2::from_ints(2_000 + distance, 2_000);
                enemy.effects = vec![effect(Some(2), Some(240))];
                view.units.push(enemy);
            }
            let space = action_space(&view);

            assert!(
                space.allows(raze(slot)),
                "beneficial={beneficial}, slot={slot}"
            );
            assert_eq!(
                space
                    .decode(raze(slot))
                    .expect("cast")
                    .expect("order")
                    .order,
                Order::Cast {
                    slot: AbilitySlot(slot),
                    target: Target::None
                }
            );
        }
    }
}

#[test]
fn razes_require_own_mana_at_each_level() {
    for cost in [75, 80, 85, 90] {
        for mana in [cost - 1, cost] {
            let mut view = view();
            view.units[0].mana = mana;
            for ability in &mut view.units[0].abilities {
                ability.mana_cost = cost;
            }
            let space = action_space(&view);

            for slot in 0..3 {
                assert_eq!(
                    space.allows(raze(slot)),
                    mana >= cost,
                    "cost={cost}, mana={mana}"
                );
            }
        }
    }
}

#[test]
fn razes_obey_learning_cooldown_passive_silence_stun_channel_and_death() {
    for blocked in 0..7 {
        let mut view = view();
        match blocked {
            0 => view.units[0].abilities[0].level = 0,
            1 => view.units[0].abilities[0].cooldown_left = 1,
            2 => view.units[0].abilities[0].passive = true,
            3 => view.units[0].statuses.bits = StatusFlags::SILENCED,
            4 => view.units[0].statuses.bits = StatusFlags::STUNNED,
            5 => view.units[0].statuses.bits = StatusFlags::CHANNELLING,
            6 => view.units[0].hp = 0,
            _ => unreachable!("bounded cases"),
        }

        assert_masked(&action_space(&view), raze(0), ActionKind::Cast);
    }
}

#[test]
fn all_abilities_and_items_require_provable_own_mana_including_rounded_one() {
    for (mana, cost, allowed) in [
        (0, 0, true),
        (0, 1, false),
        (1, 1, false),
        (2, 1, true),
        (74, 75, false),
        (75, 75, true),
    ] {
        let mut view = view();
        view.units[0].mana = mana;
        view.units[0].abilities[0].mana_cost = cost;
        let mut item = mango(None);
        item.id = ItemId(7);
        item.mana_cost = cost;
        view.units[0].items[0] = Some(item);
        let space = action_space(&view);

        assert_eq!(
            space.allows(raze(0)),
            allowed,
            "ability mana={mana}, cost={cost}"
        );
        assert_eq!(
            space.allows(use_mango(0)),
            allowed,
            "item mana={mana}, cost={cost}"
        );
    }
}

#[test]
fn effect_semantics_sort_before_opaque_entity_handles() {
    let mut view = view();
    let mut first = unit(4, UnitKind::CreepMelee);
    first.team = Team::Dire;
    first.effects = vec![effect(Some(3), Some(100))];
    let mut second = first.clone();
    second.id.idx = 5;
    second.effects = vec![effect(Some(2), Some(240))];
    view.units.extend([first, second]);
    let before = action_space(&view);
    view.units[3].id.idx = 5;
    view.units[4].id.idx = 4;
    view.units.sort_by_key(|unit| unit.id);
    let after = action_space(&view);
    let effects = |space: &ActionSpace| {
        space
            .entity_candidates()
            .iter()
            .filter(|candidate| candidate.kind == UnitKind::CreepMelee)
            .map(|candidate| candidate.unit().effects.clone())
            .collect::<Vec<_>>()
    };

    assert_eq!(effects(&before), effects(&after));
}

#[test]
fn effect_order_uses_one_active_pair_and_ignores_expired_incomplete_or_row_order() {
    let mut view = view();
    let mut first = unit(5, UnitKind::CreepMelee);
    first.team = Team::Dire;
    first.effects = vec![
        effect(Some(2), Some(100)),
        effect(Some(1), Some(240)),
        effect(Some(255), Some(0)),
        effect(None, Some(240)),
        effect(Some(255), None),
    ];
    let mut second = first.clone();
    second.id.idx = 4;
    second.effects = vec![effect(Some(2), Some(200))];
    let first_id = first.id;
    let second_id = second.id;
    view.units.extend([first, second]);
    view.units.sort_by_key(|unit| unit.id);
    let before = action_space(&view);
    for unit in &mut view.units[3..] {
        unit.effects.reverse();
    }
    let after = action_space(&view);

    assert!(
        before.entity_index(first_id).expect("first").0
            < before.entity_index(second_id).expect("second").0
    );
    for unit in &view.units[3..] {
        assert_eq!(before.entity_index(unit.id), after.entity_index(unit.id));
    }
}

#[cfg(feature = "builtin")]
#[test]
fn server_fractional_mana_deficit_below_one_is_usable_when_wire_proves_it() {
    let (mut world, info) = server_world();
    let hero = world.seats[0].unit.expect("hero");
    world.mana.get_mut(hero).expect("mana").mana = Fixed {
        raw: Fixed::from_int(500).raw - 1,
    };
    let space = server_space(&world, &info);
    let issued = space.decode(use_mango(0)).expect("use").expect("order");

    assert_eq!(
        world.validate_order(SlotId(0), issued.unit, &issued.order),
        Ok(())
    );
    let mut events = Vec::new();
    assert!(world.use_item(hero, 0, Target::None, &mut events));
    assert!(
        events.is_empty(),
        "sub-one restoration has no whole-unit report"
    );
    assert_eq!(
        world.mana.get(hero).expect("mana").mana,
        Fixed::from_int(500)
    );
    assert!(world.inventory.get(hero).expect("bag").slots[0].is_none());
}

#[cfg(feature = "builtin")]
#[test]
fn server_fractional_deficit_with_equal_rounded_values_is_conservatively_masked() {
    for (mana, maximum) in [
        (Fixed::from_ratio(1_997, 4), Fixed::from_ratio(999, 2)),
        (Fixed { raw: 1 }, Fixed::from_int(1)),
    ] {
        let (mut world, info) = server_world();
        let hero = world.seats[0].unit.expect("hero");
        world.stats.get_mut(hero).expect("stats").max_mana = maximum;
        world.mana.get_mut(hero).expect("mana").mana = mana;
        let visible = world.view(Team::Radiant);
        let own = visible
            .units
            .iter()
            .find(|unit| unit.owner == Some(SlotId(0)) && unit.kind == UnitKind::Hero)
            .expect("visible hero");
        let space = server_space(&world, &info);

        assert_eq!(own.mana, own.max_mana);
        assert_eq!(
            world.validate_order(
                SlotId(0),
                None,
                &Order::Use {
                    slot: ItemSlot(0),
                    target: Target::None
                }
            ),
            Ok(())
        );
        assert_masked(&space, use_mango(0), ActionKind::Use);
    }
}

#[cfg(feature = "builtin")]
#[test]
fn server_full_mana_rejects_mango_without_spending_a_charge() {
    let (mut world, info) = server_world();
    let hero = world.seats[0].unit.expect("hero");
    world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(500);
    let space = server_space(&world, &info);

    assert_masked(&space, use_mango(0), ActionKind::Use);
    assert_eq!(
        world.validate_order(
            SlotId(0),
            None,
            &Order::Use {
                slot: ItemSlot(0),
                target: Target::None
            }
        ),
        Err(bota_proto::RejectReason::NotReady)
    );
    let mut events = Vec::new();
    assert!(!world.use_item(hero, 0, Target::None, &mut events));
    assert!(events.is_empty(), "ineffective use reports no restoration");
    assert_eq!(
        world.inventory.get(hero).expect("bag").slots[0]
            .expect("mango")
            .charges,
        1
    );
}

#[cfg(feature = "builtin")]
#[test]
fn server_repeated_purchases_merge_to_three_then_stop_only_for_capacity() {
    let (mut world, info) = server_world();
    let hero = world.seats[0].unit.expect("hero");
    let filler = bota_server::game::ItemStack::bought(ItemId(9), SlotId(0), 0);
    world.inventory.get_mut(hero).expect("bag").slots[1..].fill(filler);
    world.seats[0].stash.slots.fill(filler);
    for expected_charges in 2..=3 {
        world.seats[0].gold = 65;
        let space = server_space(&world, &info);
        let issued = space.decode(buy()).expect("buy").expect("order");

        assert_eq!(
            world.validate_order(SlotId(0), issued.unit, &issued.order),
            Ok(())
        );
        assert!(world.buy(SlotId(0), MANGO, &mut Vec::new()));
        assert_eq!(
            world.inventory.get(hero).expect("bag").slots[0]
                .expect("mango")
                .charges,
            expected_charges
        );
        assert_eq!(world.seats[0].gold, 0);
    }
    world.seats[0].gold = 65;
    assert_masked(&server_space(&world, &info), buy(), ActionKind::Buy);
    assert_eq!(
        world.validate_order(SlotId(0), None, &Order::Buy { item: MANGO }),
        Err(bota_proto::RejectReason::InventoryFull)
    );
}

#[cfg(feature = "builtin")]
pub(super) fn server_world() -> (bota_server::game::World, MatchInfo) {
    let config = bota_server::game::MatchConfig {
        match_id: 92_010_042,
        master_key: [0; 32],
        map: MapId(2),
        tick_rate: 30,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 30,
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
    };
    let mut world = bota_server::game::World::for_match(&config, config.rng());
    world.tick = 1;
    let hero = world.seats[0].unit.expect("hero");
    world.inventory.get_mut(hero).expect("bag").slots[0] =
        bota_server::game::ItemStack::bought(MANGO, SlotId(0), world.tick);
    world.stats.get_mut(hero).expect("stats").max_mana = Fixed::from_int(500);
    world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(499);
    world.seats[0].gold = 65;
    (world, config.info())
}

#[cfg(feature = "builtin")]
fn server_space(world: &bota_server::game::World, info: &MatchInfo) -> ActionSpace {
    let mut tracker = StateTracker::new(SlotId(0), info).expect("Map2 tracker");
    tracker
        .observe_snapshot(&world.view(Team::Radiant))
        .expect("snapshot");
    tracker
        .observe_events(world.tick, &[])
        .expect("matching events");
    ActionSpace::from_tracker(&tracker).expect("Map2 action space")
}

fn buy() -> StructuredAction {
    StructuredAction::Buy {
        unit: ControlledUnit::Hero,
        item: ShopIndex(42),
    }
}

fn use_mango(slot: u8) -> StructuredAction {
    StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(slot),
        target: ActionTarget::None,
    }
}

fn raze(slot: u8) -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot),
        target: ActionTarget::None,
    }
}

fn assert_masked(space: &ActionSpace, action: StructuredAction, kind: ActionKind) {
    assert!(!space.allows(action), "{action:?}");
    let error = space.decode(action).expect_err("masked action");
    assert_eq!(error, ActionError::NotAllowed(kind));
    assert_eq!(
        error.to_string(),
        format!("action {kind:?} is masked by the current action space")
    );
}

fn action_space(view: &WorldView) -> ActionSpace {
    ActionSpace::from_tracker(&tracker(view)).expect("action space")
}

fn tracker(view: &WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &info()).expect("tracker");
    tracker.observe_snapshot(view).expect("snapshot");
    tracker
}

fn info() -> MatchInfo {
    MatchInfo {
        match_id: 92_010_042,
        map: MapId(1),
        tick_rate: 30,
        pregame_ticks: 90,
        trees: Vec::new(),
        terrain_cells: 128,
        terrain_rle: vec![(16_384, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![Pick {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: SHADOW_FIEND,
        }],
        shop: (0..43)
            .map(|id| ShopEntry {
                id: ItemId(id),
                cost: if id == 42 { 65 } else { 50 },
                components: Vec::new(),
            })
            .collect(),
    }
}

fn view() -> WorldView {
    let mut hero = unit(1, UnitKind::Hero);
    hero.owner = Some(SlotId(0));
    hero.hero = Some(SHADOW_FIEND);
    hero.mana = 499;
    hero.max_mana = 500;
    hero.items = vec![None; 9];
    hero.items[0] = Some(mango(Some(1)));
    hero.abilities = (13..16).map(ability).collect();
    let mut courier = unit(2, UnitKind::Courier);
    courier.owner = Some(SlotId(0));
    courier.items = vec![None; 6];
    let mut fountain = unit(3, UnitKind::Fountain);
    fountain.pos = Vec2::from_ints(1_500, 1_500);
    WorldView {
        tick: 10,
        viewer: Some(Team::Radiant),
        units: vec![hero, courier, fountain],
        projectiles: Vec::new(),
        players: vec![player()],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn full_view() -> WorldView {
    let mut view = view();
    let mut filler = mango(None);
    filler.id = ItemId(9);
    filler.aim = None;
    view.units[0].items.fill(Some(filler));
    view.players[0]
        .stash
        .as_mut()
        .expect("stash")
        .fill(Some(filler));
    view
}

fn unit(index: u32, kind: UnitKind) -> UnitView {
    UnitView {
        id: EntityId {
            idx: index,
            generation: 1,
        },
        kind,
        team: Team::Radiant,
        pos: Vec2::from_ints(2_000, 2_000),
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
        hero: None,
        owner: None,
        level: 1,
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}

fn player() -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team: Team::Radiant,
        hero: SHADOW_FIEND,
        unit: Some(HERO),
        level: 1,
        xp: 0,
        gold: Some(65),
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

fn mango(charges: Option<u8>) -> ItemView {
    ItemView {
        id: MANGO,
        charges,
        cooldown_left: 0,
        mute_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: false,
    }
}

fn ability(id: u16) -> AbilityView {
    AbilityView {
        id: AbilityId(id),
        level: 1,
        max_level: 4,
        cooldown_left: 0,
        mana_cost: 75,
        range: 0,
        aim: Aim::Own,
        passive: false,
        on: false,
        can_level: false,
    }
}

fn effect(stacks: Option<u32>, ticks_left: Option<u32>) -> EffectView {
    EffectView {
        id: EffectId(15),
        stacks,
        ticks_left,
    }
}
