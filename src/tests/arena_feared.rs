use bota_proto::{AbilitySlot, Fixed, ItemSlot, StatusFlags, Target};
use bota_server::game::{ItemStack, Modifier, ModifierKind, Modifiers};

use super::*;
use crate::{
    ActionError, ActionSpace, ActionTarget, ControlledUnit, StateTracker, StructuredAction,
};

#[test]
fn feared_item_mask_matches_arena_disabled_rejection_and_expiry_acceptance() {
    assert_fear_gate_and_expiry(
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(2),
            target: ActionTarget::None,
        },
        Order::Use {
            slot: ItemSlot(2),
            target: Target::None,
        },
    );
}

#[test]
fn feared_cast_mask_matches_arena_disabled_rejection_and_expiry_acceptance() {
    assert_fear_gate_and_expiry(
        StructuredAction::Cast {
            unit: ControlledUnit::Hero,
            slot: AbilitySlot(0),
            target: ActionTarget::None,
        },
        Order::Cast {
            slot: AbilitySlot(0),
            target: Target::None,
        },
    );
}

fn assert_fear_gate_and_expiry(action: StructuredAction, order: Order) {
    let (mut arena, mut tracker) = feared_arena();
    assert_eq!(tracker.current().expect("view").tick, 15_646);
    assert_eq!(
        tracker.own_hero().expect("hero").statuses.bits,
        StatusFlags::FEARED
    );
    let feared = ActionSpace::from_tracker(&tracker).expect("feared space");
    let request = Request {
        seq: 294,
        unit: None,
        order,
    };

    let rejected = arena
        .step(&[None, Some(request)])
        .expect("fear expiry tick");

    assert_eq!(
        rejected.messages[1][0],
        ServerMsg::OrderRejected {
            seq: 294,
            reason: RejectReason::Disabled
        }
    );
    assert!(
        !feared.allows(action),
        "wire-visible fear must mask the server-disabled action"
    );
    assert_eq!(
        feared.decode(action),
        Err(ActionError::NotAllowed(action.kind()))
    );
    observe_tick(&mut tracker, &rejected.messages[1]);
    assert_eq!(
        tracker.own_hero().expect("hero").statuses.bits & StatusFlags::FEARED,
        0
    );
    let ready = ActionSpace::from_tracker(&tracker).expect("ready space");
    assert!(ready.allows(action));
    let issued = ready
        .decode(action)
        .expect("unmasked action")
        .expect("order");
    assert_eq!(issued.order, order);

    let accepted = arena
        .step(&[
            None,
            Some(Request {
                seq: 295,
                unit: issued.unit,
                order: issued.order,
            }),
        ])
        .expect("accepted action after fear expiry");

    assert!(
        accepted
            .messages
            .iter()
            .flatten()
            .all(|message| !matches!(message, ServerMsg::OrderRejected { .. }))
    );
    assert_eq!(snapshot(&accepted.messages[1]).tick, 15_648);
}

fn feared_arena() -> (Arena, StateTracker) {
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 9_001,
    })
    .expect("arena");
    let ServerMsg::MatchStart { info } = &start.messages[1][0] else {
        panic!("match start first");
    };
    let mut tracker = StateTracker::new(SlotId(1), info).expect("seat one tracker");
    let configured = arena.configure_for_test(|world| {
        world.tick = 15_646;
        let hero = world.seats[1].unit.expect("seat one hero");
        let mut stick = ItemStack::bought(ItemId(35), SlotId(1), world.tick).expect("magic stick");
        stick.charges = 6;
        world.inventory.get_mut(hero).expect("bag").slots[2] = Some(stick);
        world.level.get_mut(hero).expect("level").0 = 6;
        world.abilities.get_mut(hero).expect("abilities").slots[0].level = 1;
        world.settle();
        world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(292);
        world.health.get_mut(hero).expect("health").hp = Fixed::from_int(200);
        world.modifiers.insert(
            hero,
            Modifiers(vec![Modifier {
                kind: ModifierKind::Feared,
                source: None,
                ticks_left: Some(1),
            }]),
        );
    });
    observe_tick(&mut tracker, &configured.messages[1]);
    assert_eq!(arena.tick(), 15_646);
    assert_eq!(tracker.own_hero().expect("hero").mana, 292);
    (arena, tracker)
}

fn observe_tick(tracker: &mut StateTracker, messages: &[ServerMsg]) {
    assert!((2..=4).contains(&messages.len()));
    let view = snapshot(messages);
    tracker.observe_snapshot(view).expect("arena snapshot");
    let (tick, events) = messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Events { tick, events } => Some((*tick, events)),
            _ => None,
        })
        .expect("arena events");
    assert_eq!(tick, view.tick);
    tracker
        .observe_events(tick, events)
        .expect("matching arena events");
}
