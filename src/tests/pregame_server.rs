//! Builtin `Arena` order validation and action masking during the 900-tick pregame.

use bota_proto::{
    AbilitySlot, EventKind, ItemId, MapId, Order, RejectReason, ServerMsg, SlotId, Target, Vec2,
};

use crate::{ActionSpace, Arena, ArenaConfig, ControlledUnit, Request, StateTracker};

const OPENING_SEED: u64 = 9_204_100;
const LANE_CENTER: Vec2 = bota_server::game::rules::DEMO_LANE_CORNERS[1];

#[test]
fn pregame_server_rejects_an_unlearned_cast_then_executes_it_once_learned_and_masks_its_cooldown() {
    let (mut arena, mut tracker) = pregame_arena();
    let cast = Order::Cast {
        slot: AbilitySlot(0),
        target: Target::None,
    };
    assert!(
        !cast_ready(&tracker),
        "unlearned spells stay masked before the horn"
    );
    let invalid = arena
        .step(&[Some(request(1, cast)), None])
        .expect("illegal cast tick");
    assert_eq!(
        invalid.messages[0][0],
        ServerMsg::OrderRejected {
            seq: 1,
            reason: RejectReason::NotLearned
        }
    );
    observe(&mut tracker, &invalid.messages[0]);
    let learn = Order::Learn {
        slot: AbilitySlot(0),
    };
    assert!(
        !pregame_order(&mut arena, &mut tracker, learn),
        "learning is not reported as a cast"
    );
    assert!(cast_ready(&tracker));
    assert!(
        pregame_order(&mut arena, &mut tracker, cast),
        "server executes the pregame spell, not just accepts it"
    );
    assert!(!cast_ready(&tracker));
    assert!(arena.tick() < 900);
}

#[test]
fn pregame_server_applies_shop_learn_and_movement_orders() {
    let (mut arena, mut tracker) = pregame_arena();
    let origin = tracker.own_hero().expect("hero").pos;
    for order in [
        Order::Buy { item: ItemId(0) },
        Order::Learn {
            slot: AbilitySlot(0),
        },
        Order::Move {
            target: Target::Pos(LANE_CENTER),
        },
    ] {
        pregame_order(&mut arena, &mut tracker, order);
    }
    for _ in 0..3 {
        let step = arena.step(&[None; 2]).expect("movement tick");
        observe(&mut tracker, &step.messages[0]);
    }

    let hero = tracker.own_hero().expect("hero");
    assert!(hero.items.iter().flatten().any(|item| item.id == ItemId(0)));
    assert_eq!(hero.abilities[0].level, 1);
    assert_eq!(tracker.current().expect("view").players[0].gold, Some(100));
    assert_ne!(hero.pos, origin);
    assert!(arena.tick() < 900);
}

fn pregame_arena() -> (Arena, StateTracker) {
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: OPENING_SEED,
    })
    .expect("arena");
    let ServerMsg::MatchStart { info } = &start.messages[0][0] else {
        panic!("pregame MatchStart first");
    };
    let mut tracker = StateTracker::new(SlotId(0), info).expect("pregame tracker");
    observe(&mut tracker, &start.messages[0]);
    assert_eq!(tracker.metadata().pregame_ticks, 900);
    assert_eq!(arena.tick(), 1);
    (arena, tracker)
}

fn observe(tracker: &mut StateTracker, messages: &[ServerMsg]) {
    for message in messages {
        match message {
            ServerMsg::Snapshot { view } => tracker.observe_snapshot(view).expect("snapshot"),
            ServerMsg::Events { tick, events } => {
                tracker.observe_events(*tick, events).expect("events");
            }
            _ => {}
        }
    }
}

fn request(seq: u32, order: Order) -> Request {
    Request {
        seq,
        unit: None,
        order,
    }
}

fn cast_ready(tracker: &StateTracker) -> bool {
    ActionSpace::from_tracker(tracker)
        .expect("pregame action space")
        .cast_ready(ControlledUnit::Hero, AbilitySlot(0))
}

/// Sends `order` from seat zero for one tick and reports whether a cast event followed.
fn pregame_order(arena: &mut Arena, tracker: &mut StateTracker, order: Order) -> bool {
    assert!(arena.tick() < 900);
    let step = arena
        .step(&[Some(request(arena.tick(), order)), None])
        .expect("pregame order tick");
    observe(tracker, &step.messages[0]);
    step.messages[0].iter().any(|message| {
        matches!(message, ServerMsg::Events { events, .. }
            if events.iter().any(|event| matches!(event, EventKind::AbilityCast { .. })))
    })
}
