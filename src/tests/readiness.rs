use bota_proto::{
    Aim, Angle, Attribute, Attributes, EntityId, Fixed, HeroId, ItemSlot, ItemView, MapId,
    MatchInfo, Order, Pick, PlayerView, ShopEntry, SlotId, StatusFlags, Team, TickMode, UnitKind,
    UnitView, Vec2, WorldView,
};

use crate::{
    ActionSpace, ActionTarget, BACKPACK_MUTE_TICKS, ControlledUnit, ItemReadiness,
    MAX_READINESS_TIMER_HISTORY, PointIndex, SHADOW_FIEND, SHARED_WAITS, StateTracker,
    StructuredAction, TOWN_PORTAL_SCROLL,
};

const HERO_ID: EntityId = entity(10, 1);
const COURIER_ID: EntityId = entity(11, 1);

#[test]
fn backpack_swap_mutes_the_landing_inventory_slot_until_mute_expiry() {
    let mut readiness = ItemReadiness::new();
    let mut hero_items: Vec<Option<ItemView>> = vec![None; 9];
    hero_items[7] = Some(usable_item());
    let space = space_at(1, &hero_items, &[None; 6], &readiness);
    let swap = space
        .decode(StructuredAction::Swap {
            unit: ControlledUnit::Hero,
            from: ItemSlot(7),
            to: ItemSlot(2),
        })
        .expect("swap decodes")
        .expect("swap is a wire order");
    assert_eq!(
        swap.order,
        Order::Swap {
            from: ItemSlot(7),
            to: ItemSlot(2),
        }
    );
    readiness.note_sent(1, swap, &space);
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), 2),
        Some(BACKPACK_MUTE_TICKS)
    );
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Courier, ItemSlot(2), 2),
        None
    );

    hero_items[2] = Some(usable_item());
    hero_items[7] = None;
    let muted = space_at(2, &hero_items, &[None; 6], &readiness);
    assert!(!muted.item_slot_mask(ControlledUnit::Hero)[2]);
    let untracked = ActionSpace::from_tracker(&tracker_at(2, &hero_items, &[None; 6]))
        .expect("wire-only space trusts the stack cooldown");
    assert!(untracked.item_slot_mask(ControlledUnit::Hero)[2]);

    let boundary = 2 + BACKPACK_MUTE_TICKS;
    let still_muted = space_at(boundary - 1, &hero_items, &[None; 6], &readiness);
    assert!(!still_muted.item_slot_mask(ControlledUnit::Hero)[2]);
    let awake = space_at(boundary, &hero_items, &[None; 6], &readiness);
    assert!(awake.item_slot_mask(ControlledUnit::Hero)[2]);
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), boundary),
        Some(0)
    );
}

#[test]
fn rejected_swap_rolls_back_the_mute_by_sequence() {
    let mut readiness = ItemReadiness::new();
    let mut hero_items: Vec<Option<ItemView>> = vec![None; 9];
    hero_items[7] = Some(usable_item());
    let space = space_at(1, &hero_items, &[None; 6], &readiness);
    let swap = space
        .decode(StructuredAction::Swap {
            unit: ControlledUnit::Hero,
            from: ItemSlot(7),
            to: ItemSlot(2),
        })
        .expect("swap decodes")
        .expect("swap is a wire order");
    readiness.note_sent(1, swap, &space);

    assert!(!readiness.note_rejected(2));
    assert!(readiness.note_rejected(1));

    hero_items[2] = Some(usable_item());
    hero_items[7] = None;
    let space = space_at(2, &hero_items, &[None; 6], &readiness);
    assert!(space.item_slot_mask(ControlledUnit::Hero)[2]);
}

#[test]
fn rejected_newer_overlapping_mute_restores_exact_older_timer() {
    let mut readiness = ItemReadiness::new();
    note_swap_timer(&mut readiness, 10, 1, ItemSlot(2));
    note_swap_timer(&mut readiness, 11, 20, ItemSlot(2));

    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), 21),
        Some(BACKPACK_MUTE_TICKS)
    );
    assert!(readiness.note_rejected(11));
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), 21),
        Some(161)
    );
    assert!(!readiness.note_rejected(99));
}

#[test]
fn rejected_newer_overlapping_shared_wait_restores_exact_older_timer() {
    let mut readiness = ItemReadiness::new();
    note_shared_wait(&mut readiness, 10, 1);
    note_shared_wait(&mut readiness, 11, 20);

    assert_eq!(
        readiness.shared_wait_left(ControlledUnit::Hero, TOWN_PORTAL_SCROLL, 21),
        Some(SHARED_WAITS[0].1)
    );
    assert!(readiness.note_rejected(11));
    assert_eq!(
        readiness.shared_wait_left(ControlledUnit::Hero, TOWN_PORTAL_SCROLL, 21),
        Some(2_081)
    );
}

#[test]
fn readiness_timer_journal_is_bounded_and_restores_newest_retained_entry() {
    let mut readiness = ItemReadiness::new();
    for offset in 0..=MAX_READINESS_TIMER_HISTORY {
        note_swap_timer(
            &mut readiness,
            u32::try_from(offset + 1).expect("small sequence"),
            u32::try_from(offset + 1).expect("small tick"),
            ItemSlot(2),
        );
    }

    assert!(!readiness.note_rejected(1));
    assert!(readiness.note_rejected(9));
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), 10),
        Some(179)
    );

    for sequence in (2..=8).rev() {
        assert!(readiness.note_rejected(sequence));
    }
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), 10),
        Some(172)
    );
    assert!(!readiness.note_rejected(1));
}

#[test]
fn readiness_base_composes_over_multiple_complete_history_evictions() {
    let mut readiness = ItemReadiness::new();
    let total = MAX_READINESS_TIMER_HISTORY * 3 + 1;
    for offset in 1..=total {
        let sequence = u32::try_from(offset).expect("small sequence");
        note_swap_timer(&mut readiness, sequence, sequence, ItemSlot(2));
    }

    let first_retained =
        u32::try_from(total - MAX_READINESS_TIMER_HISTORY + 1).expect("small retained sequence");
    for sequence in (first_retained..=u32::try_from(total).expect("small total")).rev() {
        assert!(readiness.note_rejected(sequence));
    }

    let latest_evicted = first_retained - 1;
    let query_tick = u32::try_from(total + 1).expect("small query tick");
    assert_eq!(
        readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(2), query_tick),
        Some(
            latest_evicted
                .saturating_add(1)
                .saturating_add(BACKPACK_MUTE_TICKS)
                .saturating_sub(query_tick)
        )
    );
    assert!(!readiness.note_rejected(latest_evicted));
}

#[test]
fn rejection_does_not_change_other_readiness_channels() {
    let mut readiness = ItemReadiness::new();
    note_swap_timer(&mut readiness, 1, 1, ItemSlot(2));
    note_swap_timer(&mut readiness, 2, 2, ItemSlot(3));

    assert!(readiness.note_rejected(2));
    assert!(readiness.inventory_muted(ControlledUnit::Hero, ItemSlot(2), 3));
    assert!(!readiness.inventory_muted(ControlledUnit::Hero, ItemSlot(3), 3));
}

#[test]
fn inventory_to_inventory_and_courier_swaps_never_mute() {
    let mut readiness = ItemReadiness::new();
    let mut hero_items: Vec<Option<ItemView>> = vec![None; 9];
    hero_items[1] = Some(usable_item());
    hero_items[8] = Some(usable_item());
    let space = space_at(
        1,
        &hero_items,
        &[Some(usable_item()), None, None, None, None, None],
        &readiness,
    );
    let inventory_swap = space
        .decode(StructuredAction::Swap {
            unit: ControlledUnit::Hero,
            from: ItemSlot(1),
            to: ItemSlot(2),
        })
        .expect("inventory swap decodes")
        .expect("wire order");
    readiness.note_sent(1, inventory_swap, &space);
    readiness.note_sent(
        2,
        crate::IssuedOrder {
            unit: Some(COURIER_ID),
            order: Order::Swap {
                from: ItemSlot(5),
                to: ItemSlot(0),
            },
        },
        &space,
    );
    readiness.note_sent(
        3,
        crate::IssuedOrder {
            unit: None,
            order: Order::Swap {
                from: ItemSlot(8),
                to: ItemSlot(7),
            },
        },
        &space,
    );

    let mut next_items: Vec<Option<ItemView>> = vec![None; 9];
    next_items[2] = Some(usable_item());
    let space = space_at(
        2,
        &next_items,
        &[Some(usable_item()), None, None, None, None, None],
        &readiness,
    );
    assert!(space.item_slot_mask(ControlledUnit::Hero)[2]);
    assert!(space.item_slot_mask(ControlledUnit::Courier)[0]);
}

#[test]
fn town_portal_use_blocks_every_stack_on_that_body_until_the_shared_wait_expires() {
    let mut readiness = ItemReadiness::new();
    let hero_items = vec![
        Some(town_portal()),
        Some(town_portal()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let space = space_at(1, &hero_items, &[None; 6], &readiness);
    let landing = PointIndex(
        space
            .point_candidates()
            .iter()
            .position(|point| point.allied_building && point.walkable)
            .expect("building landing"),
    );
    let teleport = space
        .decode(StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::Point(landing),
        })
        .expect("teleport decodes")
        .expect("wire order");
    readiness.note_sent(7, teleport, &space);
    assert_eq!(
        readiness.shared_wait_left(ControlledUnit::Hero, TOWN_PORTAL_SCROLL, 2),
        Some(SHARED_WAITS[0].1)
    );

    let hero_items = vec![
        Some(town_portal()),
        Some(town_portal()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let waiting = space_at(2, &hero_items, &[None; 6], &readiness);
    assert!(!waiting.item_slot_mask(ControlledUnit::Hero)[0]);
    assert!(!waiting.item_slot_mask(ControlledUnit::Hero)[1]);
    let untracked = ActionSpace::from_tracker(&tracker_at(2, &hero_items, &[None; 6]))
        .expect("wire reports shared-wait stacks as ready");
    assert!(untracked.item_slot_mask(ControlledUnit::Hero)[0]);

    let wait = SHARED_WAITS
        .iter()
        .find(|(item, _)| *item == TOWN_PORTAL_SCROLL)
        .expect("scroll wait")
        .1;
    let boundary = 2 + wait;
    let still_waiting = space_at(boundary - 1, &hero_items, &[None; 6], &readiness);
    assert!(!still_waiting.item_slot_mask(ControlledUnit::Hero)[1]);
    let expired = space_at(boundary, &hero_items, &[None; 6], &readiness);
    assert!(expired.item_slot_mask(ControlledUnit::Hero)[1]);
    assert_eq!(
        readiness.shared_wait_left(ControlledUnit::Hero, TOWN_PORTAL_SCROLL, boundary),
        Some(0)
    );
}

#[test]
fn rejected_town_portal_use_rolls_back_the_shared_wait() {
    let mut readiness = ItemReadiness::new();
    let hero_items = vec![
        Some(town_portal()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let space = space_at(1, &hero_items, &[None; 6], &readiness);
    let landing = PointIndex(
        space
            .point_candidates()
            .iter()
            .position(|point| point.allied_building && point.walkable)
            .expect("building landing"),
    );
    let teleport = space
        .decode(StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::Point(landing),
        })
        .expect("teleport decodes")
        .expect("wire order");
    readiness.note_sent(7, teleport, &space);
    assert!(readiness.note_rejected(7));

    let space = space_at(2, &hero_items, &[None; 6], &readiness);
    assert!(space.item_slot_mask(ControlledUnit::Hero)[0]);
}

#[test]
fn shared_waits_are_tracked_per_body() {
    let mut readiness = ItemReadiness::new();
    let hero_items = vec![
        Some(town_portal()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let courier_items = [Some(town_portal()), None, None, None, None, None];
    let space = space_at(1, &hero_items, &courier_items, &readiness);
    readiness.note_sent(
        4,
        crate::IssuedOrder {
            unit: Some(COURIER_ID),
            order: Order::Use {
                slot: ItemSlot(0),
                target: bota_proto::Target::None,
            },
        },
        &space,
    );

    let space = space_at(2, &hero_items, &courier_items, &readiness);
    assert!(!space.item_slot_mask(ControlledUnit::Courier)[0]);
    assert!(space.item_slot_mask(ControlledUnit::Hero)[0]);
}

fn space_at(
    tick: u32,
    hero_items: &[Option<ItemView>],
    courier_items: &[Option<ItemView>],
    readiness: &ItemReadiness,
) -> ActionSpace {
    ActionSpace::from_tracker_with_readiness(
        &tracker_at(tick, hero_items, courier_items),
        readiness,
    )
    .expect("action space")
}

fn note_swap_timer(readiness: &mut ItemReadiness, sequence: u32, tick: u32, target: ItemSlot) {
    let mut hero_items = vec![None; 9];
    hero_items[7] = Some(usable_item());
    let space = space_at(tick, &hero_items, &[None; 6], readiness);
    readiness.note_sent(
        sequence,
        crate::IssuedOrder {
            unit: None,
            order: Order::Swap {
                from: ItemSlot(7),
                to: target,
            },
        },
        &space,
    );
}

fn note_shared_wait(readiness: &mut ItemReadiness, sequence: u32, tick: u32) {
    let hero_items = [
        Some(town_portal()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let space = space_at(tick, &hero_items, &[None; 6], readiness);
    readiness.note_sent(
        sequence,
        crate::IssuedOrder {
            unit: None,
            order: Order::Use {
                slot: ItemSlot(0),
                target: bota_proto::Target::None,
            },
        },
        &space,
    );
}

fn tracker_at(
    tick: u32,
    hero_items: &[Option<ItemView>],
    courier_items: &[Option<ItemView>],
) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    tracker
        .observe_snapshot(&world_view(tick, hero_items, courier_items))
        .expect("snapshot");
    tracker
}

fn match_info() -> MatchInfo {
    MatchInfo {
        match_id: 1,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 90,
        trees: Vec::new(),
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
        shop: vec![ShopEntry {
            id: TOWN_PORTAL_SCROLL,
            cost: 100,
            components: Vec::new(),
        }],
    }
}

fn world_view(
    tick: u32,
    hero_items: &[Option<ItemView>],
    courier_items: &[Option<ItemView>],
) -> WorldView {
    let mut hero = unit(HERO_ID, UnitKind::Hero, Team::Radiant, 2_000, 2_000);
    hero.hero = Some(SHADOW_FIEND);
    hero.owner = Some(SlotId(0));
    hero.mana = 500;
    hero.max_mana = 500;
    hero.items = hero_items.to_vec();
    let mut courier = unit(COURIER_ID, UnitKind::Courier, Team::Radiant, 2_100, 2_000);
    courier.owner = Some(SlotId(0));
    courier.items = courier_items.to_vec();
    let mut fountain = unit(
        entity(30, 1),
        UnitKind::Fountain,
        Team::Radiant,
        1_500,
        1_500,
    );
    fountain.radius = Fixed::from_int(60);
    let mut tower = unit(entity(32, 1), UnitKind::Tower, Team::Radiant, 2_500, 2_000);
    tower.radius = Fixed::from_int(40);
    let mut units = vec![hero, courier, fountain, tower];
    units.sort_by_key(|unit| unit.id);
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units,
        projectiles: Vec::new(),
        players: vec![
            PlayerView {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: SHADOW_FIEND,
                unit: Some(HERO_ID),
                level: 1,
                xp: 0,
                gold: Some(0),
                stash: Some(vec![None; 6]),
                kit: None,
                kills: 0,
                deaths: 0,
                assists: 0,
                last_hits: 0,
                denies: 0,
                respawn_left: 0,
            },
            PlayerView {
                slot: SlotId(1),
                team: Team::Dire,
                hero: SHADOW_FIEND,
                unit: None,
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
            },
        ],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
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
        attack_damage: 0,
        attack_range: Fixed::from_int(500),
        attack_interval: 30,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(24),
        vision_radius: Fixed::from_int(1_800),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::all(0),
        primary: Some(Attribute::Agility),
        hero: (kind == UnitKind::Hero).then_some(HeroId(2)),
        owner: None,
        level: 0,
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}

fn usable_item() -> ItemView {
    ItemView {
        id: TOWN_PORTAL_SCROLL,
        charges: Some(1),
        cooldown_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: false,
    }
}

fn town_portal() -> ItemView {
    ItemView {
        range: 600,
        aim: Some(Aim::Building),
        ..usable_item()
    }
}

const fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}
