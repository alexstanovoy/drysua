use super::fixtures;
use crate::{
    ActionSpace, BACKPACK_MUTE_TICKS, ControlledUnit, IssuedOrder, ItemReadiness,
    MAX_READINESS_TIMER_HISTORY, SHADOW_FIEND, SHARED_WAITS, StateTracker, TOWN_PORTAL_SCROLL,
};
use bota_proto::{
    Aim, Attributes, EntityId, ItemSlot, ItemView, MapId, MatchInfo, Order, PlayerView, ShopEntry,
    SlotId, Team, UnitKind, Vec2, WorldView,
};

const HERO_ID: EntityId = EntityId {
    idx: 10,
    generation: 1,
};
const COURIER_ID: EntityId = EntityId {
    idx: 11,
    generation: 1,
};

#[test]
fn readiness_lifecycle_masks_exact_expiry_and_rejection_restores_older_body_local_timers() {
    for shared in [false, true] {
        let mut readiness = ItemReadiness::new();
        note(&mut readiness, 10, 1, shared);
        let older = readiness;
        note(&mut readiness, 11, 20, shared);
        let duration = if shared {
            SHARED_WAITS[0].1
        } else {
            BACKPACK_MUTE_TICKS
        };
        for (tick, allowed) in [(21, false), (20 + duration, false), (21 + duration, true)] {
            let space = space_at(tick, &readiness);
            assert_eq!(space.item_slot_mask(ControlledUnit::Hero)[0], allowed);
            assert_eq!(
                space.item_slot_mask(ControlledUnit::Hero)[1],
                !shared || allowed
            );
            assert!(space.item_slot_mask(ControlledUnit::Courier)[0]);
        }
        assert!(!readiness.note_rejected(99));
        assert!(readiness.note_rejected(11));
        assert_eq!(readiness, older);
        assert!(!space_at(21, &readiness).item_slot_mask(ControlledUnit::Hero)[0]);
        assert!(readiness.note_rejected(10));
        assert!(space_at(21, &readiness).item_slot_mask(ControlledUnit::Hero)[0]);
    }
}

#[test]
fn bounded_readiness_journal_preserves_evicted_baseline_across_multiple_wraps() {
    let mut readiness = ItemReadiness::new();
    let total = MAX_READINESS_TIMER_HISTORY as u32 * 3 + 1;
    for sequence in 1..=total {
        note(&mut readiness, sequence, sequence, false);
    }
    let first = total - MAX_READINESS_TIMER_HISTORY as u32 + 1;
    assert!(!readiness.note_rejected(first - 1));
    for sequence in (first..=total).rev() {
        assert!(readiness.note_rejected(sequence));
        assert_eq!(
            readiness.inventory_mute_left(ControlledUnit::Hero, ItemSlot(0), total + 1),
            Some(sequence + BACKPACK_MUTE_TICKS - (total + 1))
        );
    }
    assert!(!readiness.note_rejected(first - 1));
}

#[test]
fn inventory_swaps_propagate_mutes_without_muting_courier_or_backpack() {
    for (from, to) in [(7, 0), (0, 7)] {
        let mut readiness = ItemReadiness::new();
        let space = space_at(1, &readiness);
        let swap = |unit, from, to| IssuedOrder {
            unit,
            order: Order::Swap {
                from: ItemSlot(from),
                to: ItemSlot(to),
            },
        };
        readiness.note_sent(1, swap(None, from, to), &space);
        let space = space_at(2, &readiness);
        readiness.note_sent(2, swap(None, 0, 1), &space);
        readiness.note_sent(3, swap(Some(COURIER_ID), 0, 1), &space);
        readiness.note_sent(4, swap(None, 7, 8), &space);
        let next = space_at(3, &readiness);
        assert!(!next.item_slot_mask(ControlledUnit::Hero)[0]);
        assert!(!next.item_slot_mask(ControlledUnit::Hero)[1]);
        assert!(next.item_slot_mask(ControlledUnit::Courier)[1]);
        assert_eq!(
            readiness.inventory_mute_left(
                ControlledUnit::Hero,
                ItemSlot(1),
                2 + BACKPACK_MUTE_TICKS
            ),
            Some(0)
        );
    }
}

fn note(readiness: &mut ItemReadiness, sequence: u32, tick: u32, shared: bool) {
    let space = space_at(tick, readiness);
    let order = if shared {
        Order::Use {
            slot: ItemSlot(0),
            target: bota_proto::Target::None,
        }
    } else {
        Order::Swap {
            from: ItemSlot(7),
            to: ItemSlot(0),
        }
    };
    readiness.note_sent(sequence, IssuedOrder { unit: None, order }, &space);
}

fn space_at(tick: u32, readiness: &ItemReadiness) -> ActionSpace {
    let item = Some(ItemView {
        id: TOWN_PORTAL_SCROLL,
        charges: Some(1),
        cooldown_left: 0,
        mute_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: false,
    });
    let mut tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    tracker
        .observe_snapshot(&world_view(tick, &[item; 9], &[item; 6]))
        .expect("snapshot");
    ActionSpace::from_tracker_with_readiness(&tracker, readiness).expect("readiness masks")
}

pub(super) fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(1, MapId(0), fixtures::two_seat_picks(Team::Radiant))
        .pregame_ticks(90)
        .terrain_cells(128)
        .terrain_rle(vec![(16_384, 0x80)])
        .shop(vec![ShopEntry {
            id: TOWN_PORTAL_SCROLL,
            cost: 100,
            components: Vec::new(),
        }])
        .build()
}

pub(super) fn world_view(
    tick: u32,
    hero_items: &[Option<ItemView>],
    courier_items: &[Option<ItemView>],
) -> WorldView {
    let mut units = Vec::new();
    for (idx, kind, x, y, items) in [
        (HERO_ID.idx, UnitKind::Hero, 2_000, 2_000, hero_items),
        (
            COURIER_ID.idx,
            UnitKind::Courier,
            2_100,
            2_000,
            courier_items,
        ),
        (30, UnitKind::Fountain, 1_500, 1_500, &[][..]),
        (32, UnitKind::Tower, 2_500, 2_000, &[][..]),
    ] {
        let mut unit = fixtures::UnitFixture {
            id: EntityId { idx, generation: 1 },
            kind,
            team: Team::Radiant,
            pos: Vec2::from_ints(x, y),
            mana: if kind == UnitKind::Hero { 500 } else { 0 },
            attack_damage: 0,
            attack_time: 1000,
            attributes: Attributes::all(0),
            primary: Some(bota_proto::Attribute::Agility),
            hero: (kind == UnitKind::Hero).then_some(SHADOW_FIEND),
            owner: matches!(kind, UnitKind::Hero | UnitKind::Courier).then_some(SlotId(0)),
            level: 0,
        }
        .build();
        unit.items = items.to_vec();
        if matches!(kind, UnitKind::Fountain | UnitKind::Tower) {
            let radius = if kind == UnitKind::Fountain { 60 } else { 40 };
            unit.collision = bota_proto::Fixed::from_int(radius);
            unit.bound = unit.collision;
        }
        units.push(unit);
    }
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units,
        players: players(),
        projectiles: Vec::new(),
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn players() -> Vec<PlayerView> {
    [Team::Radiant, Team::Dire]
        .into_iter()
        .enumerate()
        .map(|(slot, team)| PlayerView {
            slot: SlotId(slot as u8),
            team,
            hero: SHADOW_FIEND,
            unit: (slot == 0).then_some(HERO_ID),
            level: 1,
            xp: 0,
            gold: (slot == 0).then_some(0),
            stash: (slot == 0).then(|| vec![None; 6]),
            kit: None,
            kills: 0,
            deaths: 0,
            assists: 0,
            last_hits: 0,
            denies: 0,
            respawn_left: 0,
        })
        .collect()
}
