//! Server-side pregame contract tests salvaged from the retired tactical pregame module.
//!
//! They exercise builtin `Arena` order validation and masking during the 900-tick
//! pregame window without any tactical policy; only the seat projection helper was
//! reduced to the tracker and order persistence the assertions use.

use bota_proto::{
    AbilitySlot, EventKind, ItemId, MapId, Order, RejectReason, ServerMsg, SlotId, Target, Vec2,
};

use crate::{
    ActionSpace, ActionTarget, Arena, ArenaConfig, ControlledUnit, OrderPersistence, Request,
    StateTracker, StructuredAction,
};

const OPENING_SEED: u64 = 9_204_100;
const LANE_CENTER: Vec2 = bota_server::game::rules::DEMO_LANE_CORNERS[1];

struct Seat {
    tracker: StateTracker,
    persistence: OrderPersistence,
}

fn new_seat(index: u8, messages: &[ServerMsg]) -> Seat {
    let info = messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info),
            _ => None,
        })
        .expect("pregame MatchStart");
    let mut seat = Seat {
        tracker: StateTracker::new(SlotId(index), info).expect("pregame tracker"),
        persistence: OrderPersistence::default(),
    };
    observe_messages(&mut seat, messages);
    seat
}

fn observe_messages(seat: &mut Seat, messages: &[ServerMsg]) {
    for message in messages {
        match message {
            ServerMsg::Snapshot { view } => {
                let previous = seat.tracker.own_hero().map(|hero| hero.id);
                seat.tracker.observe_snapshot(view).expect("snapshot");
                if previous != seat.tracker.own_hero().map(|hero| hero.id) {
                    seat.persistence.clear_body_for(None);
                }
            }
            ServerMsg::Events { tick, events } => {
                seat.tracker.observe_events(*tick, events).expect("events");
            }
            ServerMsg::MatchStart { .. } => {}
            _ => {}
        }
    }
}

#[test]
fn pregame_server_rejects_an_unlearned_cast_and_the_action_mask_excludes_it() {
    let (mut arena, seat) = pregame_arena();
    let cast = StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(0),
        target: ActionTarget::None,
    };
    let space = ActionSpace::from_tracker(&seat.tracker).expect("pregame action space");
    assert!(
        !space.allows(cast),
        "unlearned spells stay masked before the horn"
    );
    let invalid = arena
        .step(&[
            Some(Request {
                seq: 1,
                unit: None,
                order: Order::Cast {
                    slot: AbilitySlot(0),
                    target: Target::None,
                },
            }),
            None,
        ])
        .expect("illegal cast tick");
    assert_eq!(
        invalid.messages[0][0],
        ServerMsg::OrderRejected {
            seq: 1,
            reason: RejectReason::NotLearned
        }
    );
}

#[test]
fn pregame_server_applies_shop_learn_and_movement_orders() {
    let (mut arena, mut seat) = pregame_arena();
    let origin = seat.tracker.own_hero().expect("hero").pos;
    for order in [
        Order::Buy { item: ItemId(0) },
        Order::Learn {
            slot: AbilitySlot(0),
        },
        Order::Move {
            target: Target::Pos(LANE_CENTER),
        },
    ] {
        pregame_order(&mut arena, &mut seat, order);
    }
    for _ in 0..3 {
        let step = arena.step(&[None; 2]).expect("movement tick");
        observe_messages(&mut seat, &step.messages[0]);
    }

    let hero = seat.tracker.own_hero().expect("hero");
    assert!(hero.items.iter().flatten().any(|item| item.id == ItemId(0)));
    assert_eq!(hero.abilities[0].level, 1);
    assert_eq!(
        seat.tracker.current().expect("view").players[0].gold,
        Some(100)
    );
    assert_ne!(hero.pos, origin);
    assert!(arena.tick() < 900);
}

#[test]
fn pregame_server_executes_a_learned_cast_and_masks_its_cooldown() {
    let (mut arena, mut seat) = pregame_arena();
    pregame_order(
        &mut arena,
        &mut seat,
        Order::Learn {
            slot: AbilitySlot(0),
        },
    );
    let cast = StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(0),
        target: ActionTarget::None,
    };
    let space = ActionSpace::from_tracker(&seat.tracker).expect("learned space");
    assert!(space.allows(cast));

    let cast_completed = pregame_order(
        &mut arena,
        &mut seat,
        Order::Cast {
            slot: AbilitySlot(0),
            target: Target::None,
        },
    );

    assert!(
        cast_completed,
        "server executes the pregame spell, not just accepts it"
    );
    let space = ActionSpace::from_tracker(&seat.tracker).expect("cooldown space");
    assert!(!space.allows(cast));
    assert!(arena.tick() < 900);
}

fn pregame_arena() -> (Arena, Seat) {
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: OPENING_SEED,
    })
    .expect("arena");
    let seat = new_seat(0, &start.messages[0]);
    assert_eq!(seat.tracker.metadata().pregame_ticks, 900);
    assert_eq!(arena.tick(), 1);
    (arena, seat)
}

fn pregame_order(arena: &mut Arena, seat: &mut Seat, order: Order) -> bool {
    assert!(arena.tick() < 900);
    assert_eq!(arena.seat_count(), 2);
    let sequence = arena.tick();
    let step = arena
        .step(&[
            Some(Request {
                seq: sequence,
                unit: None,
                order,
            }),
            None,
        ])
        .expect("pregame order tick");
    observe_messages(seat, &step.messages[0]);
    has_cast_event(&step.messages[0])
}

fn has_cast_event(messages: &[ServerMsg]) -> bool {
    messages.iter().any(|message| matches!(message,
        ServerMsg::Events { events, .. } if events.iter().any(|event| matches!(event, EventKind::AbilityCast { .. }))))
}
