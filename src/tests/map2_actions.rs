use bota_proto::{Aim, ItemId, ItemSlot, ItemView, MapId, MatchInfo, Order, Target, WorldView};

#[cfg(feature = "builtin")]
use bota_proto::{Fixed, SlotId, Team};

use super::action;
#[cfg(feature = "builtin")]
use super::fixtures;
#[cfg(feature = "builtin")]
use crate::ShopIndex;
use crate::{ActionSpace, ActionTarget, ControlledUnit, ItemReadiness, StructuredAction};

const MANGO: ItemId = ItemId(42);

#[test]
fn casts_require_provable_mana_at_fractional_and_whole_unit_boundaries() {
    for (mana, cost, allowed) in [(1, 1, false), (2, 1, true), (74, 75, false), (75, 75, true)] {
        let mut view = view();
        view.units[0].mana = mana;
        view.units[0].abilities[0].mana_cost = cost;
        let slot = bota_proto::AbilitySlot(0);
        action::assert_gate(
            &space(view),
            StructuredAction::Cast {
                unit: ControlledUnit::Hero,
                slot,
                target: ActionTarget::None,
            },
            allowed.then_some(Order::Cast {
                slot,
                target: Target::None,
            }),
        );
    }
}

#[test]
fn malformed_mango_stacks_and_unprovable_mana_are_rejected_at_public_decode() {
    for charges in [None, Some(0), Some(4), Some(u8::MAX)] {
        let mut view = view();
        view.units[0].items[0] = Some(mango(charges));
        action::assert_gate(&space(view), use_mango(0), None);
    }
    for (mana, maximum) in [(-1, 500), (0, 0), (0, -1), (500, 500), (501, 500)] {
        let mut view = view();
        view.units[0].mana = mana;
        view.units[0].max_mana = maximum;
        action::assert_gate(&space(view), use_mango(0), None);
    }
    for mana in [0, 499] {
        let mut view = view();
        view.units[0].mana = mana;
        action::assert_gate(&space(view), use_mango(0), Some(use_order()));
    }
}

#[test]
fn backpack_mango_readiness_resumes_at_exact_expiry() {
    let mut view = view();
    view.units[0].items.swap(0, 6);
    let space = space(view.clone());
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
        let tracker = action::tracker_with_info_and_view(&info(), view.clone());
        let space = ActionSpace::from_tracker_with_readiness(&tracker, &readiness).expect("space");
        action::assert_gate(&space, use_mango(0), allowed.then_some(use_order()));
    }
}

#[cfg(feature = "builtin")]
#[test]
fn native_mango_wire_proof_validation_and_consumption_preserve_fractional_boundaries() {
    for (mana, maximum, wire_allowed, native_allowed) in [
        (
            Fixed {
                raw: Fixed::from_int(500).raw - 1,
            },
            Fixed::from_int(500),
            true,
            true,
        ),
        (
            Fixed::from_ratio(1_997, 4),
            Fixed::from_ratio(999, 2),
            false,
            true,
        ),
        (Fixed { raw: 1 }, Fixed::from_int(1), false, true),
        (Fixed::from_int(500), Fixed::from_int(500), false, false),
    ] {
        let (mut world, info) = server_world();
        let hero = world.seats[0].unit.expect("hero");
        world.stats.get_mut(hero).expect("stats").max_mana = maximum;
        world.mana.get_mut(hero).expect("mana").mana = mana;
        let space = action::space(&info, world.view(Team::Radiant));
        action::assert_gate(&space, use_mango(0), wire_allowed.then_some(use_order()));
        assert_eq!(
            world.validate_order(SlotId(0), None, &use_order()),
            if native_allowed {
                Ok(())
            } else {
                Err(bota_proto::RejectReason::NotReady)
            }
        );
        let mut events = Vec::new();
        assert_eq!(
            world.use_item(hero, 0, Target::None, &mut events),
            native_allowed
        );
        assert!(
            events.is_empty(),
            "fractional restoration reports no whole units"
        );
        assert_eq!(world.mana.get(hero).expect("mana").mana, maximum);
        let held = world.inventory.get(hero).expect("bag").slots[0];
        assert_eq!(held.is_none(), native_allowed);
        if !native_allowed {
            assert_eq!(held.expect("unspent mango").charges, 1);
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn native_mango_purchases_execute_decoded_orders_merge_to_three_and_stop_at_capacity() {
    let (mut world, info) = server_world();
    let hero = world.seats[0].unit.expect("hero");
    let filler = bota_server::game::ItemStack::bought(ItemId(9), SlotId(0), 0);
    world.inventory.get_mut(hero).expect("bag").slots[1..].fill(filler);
    world.seats[0].stash.slots.fill(filler);
    for expected in 2..=3 {
        world.seats[0].gold = 64;
        action::assert_gate(
            &action::space(&info, world.view(Team::Radiant)),
            buy(),
            None,
        );
        world.seats[0].gold = 65;
        let space = action::space(&info, world.view(Team::Radiant));
        let issued = space.decode(buy()).expect("decode").expect("buy");
        assert_eq!(issued.order, Order::Buy { item: MANGO });
        assert_eq!(
            world.validate_order(SlotId(0), issued.unit, &issued.order),
            Ok(())
        );
        world.advance(&[bota_server::game::Command {
            slot: SlotId(0),
            unit: issued.unit,
            order: issued.order,
        }]);
        assert_eq!(
            world.inventory.get(hero).expect("bag").slots[0]
                .expect("mango")
                .charges,
            expected
        );
        assert_eq!(world.seats[0].gold, 0);
    }
    world.seats[0].gold = 65;
    action::assert_gate(
        &action::space(&info, world.view(Team::Radiant)),
        buy(),
        None,
    );
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
        mode: bota_proto::TickMode::Lockstep,
        ack_timeout_ticks: 30,
        cheats: false,
        spawn_modifiers: Vec::new(),
        picks: fixtures::two_seat_picks(Team::Radiant),
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

fn space(view: WorldView) -> ActionSpace {
    action::space(&info(), view)
}

fn info() -> MatchInfo {
    let mut info = action::match_info();
    info.map = MapId(1);
    info.shop = (0..43)
        .map(|id| bota_proto::ShopEntry {
            id: ItemId(id),
            cost: if id == 42 { 65 } else { 50 },
            components: Vec::new(),
        })
        .collect();
    info
}

fn view() -> WorldView {
    let mut view = action::world_view(10);
    view.units[0].mana = 499;
    view.units[0].items[0] = Some(mango(Some(1)));
    view.players[0].gold = Some(65);
    view
}

pub(super) fn mango(charges: Option<u8>) -> ItemView {
    ItemView {
        id: MANGO,
        charges,
        ..action::item(Some(Aim::Own), 0)
    }
}

#[cfg(feature = "builtin")]
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

fn use_order() -> Order {
    Order::Use {
        slot: ItemSlot(0),
        target: Target::None,
    }
}
