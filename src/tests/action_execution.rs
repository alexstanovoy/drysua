use bota_proto::{EffectId, EventKind, Fixed, ItemSlot, StatusFlags, Target, Vec2};

use super::*;
use crate::{
    ActionSpace, ActionTarget, ControlledUnit, PointIndex, StateTracker, StructuredAction,
};

const TREE: Vec2 = Vec2::from_ints(8192, 10304);
const ORIGIN: Vec2 = Vec2::from_ints(8320, 10080);

#[test]
fn tango_native_and_decoded_orders_approach_consume_and_heal_beyond_cast_range() {
    // Prove the native contract first, independently of the adapter's mask.
    for decoded in [false, true] {
        let (mut arena, mut tracker) = tango_arena();
        let initial_hp = tracker.own_hero().unwrap().hp;
        wait_ticks(&mut arena, &mut tracker, 60);
        let baseline_gain = tracker.own_hero().unwrap().hp - initial_hp;
        let space = ActionSpace::from_tracker(&tracker).unwrap();
        let hero = tracker.own_hero().unwrap();
        let tango = hero.items[0].unwrap();
        assert!(!hero.pos.within(TREE, Fixed::from_int(tango.range)));
        let action = tango_action(tree_target(&space));
        let native = Order::Use {
            slot: ItemSlot(0),
            target: Target::Pos(TREE),
        };
        let order = if decoded {
            let issued = space.decode(action).expect("distant Tango target").unwrap();
            assert_eq!(issued.unit, None);
            assert_eq!(issued.order, native);
            issued.order
        } else {
            native
        };

        step(&mut arena, &mut tracker, Some(order));
        wait_ticks(&mut arena, &mut tracker, 2);
        let hero = tracker.own_hero().unwrap();
        assert_ne!(hero.pos, ORIGIN, "Use must approach without Move");
        assert_eq!(hero.items[0].unwrap().charges, Some(3));
        assert_tango_consumes_and_heals(&mut arena, &mut tracker, baseline_gain);

        // Local observation must remove the consumed tree, not choose a replacement.
        let movement = Order::Move {
            target: Target::Pos(TREE),
        };
        step(&mut arena, &mut tracker, Some(movement));
        wait_ticks(&mut arena, &mut tracker, 32);
        let hero = tracker.own_hero().unwrap();
        assert!(hero.pos.within(TREE, Fixed::from_int(32)));
        let after = ActionSpace::from_tracker(&tracker).unwrap();
        let points = after.point_candidates();
        assert!(
            !points
                .iter()
                .any(|point| point.position == TREE && point.standing_tree)
        );
    }
}

fn assert_tango_consumes_and_heals(arena: &mut Arena, tracker: &mut StateTracker, baseline: i32) {
    let hero_id = tracker.own_hero().unwrap().id;
    let trees = tracker.static_trees();
    let tree_index = trees.iter().position(|position| *position == TREE).unwrap() as u32;
    let mut consumed = false;
    for _ in 0..60 {
        let advanced = step(arena, tracker, None);
        let hero = tracker.own_hero().unwrap();
        if hero.items[0].unwrap().charges == Some(2) {
            let view = tracker.current().unwrap();
            assert!(view.felled_trees.contains(&tree_index));
            assert!(hero.effects.iter().any(|effect| effect.id == EffectId(1)));
            assert!(advanced.messages[0].iter().any(|message| matches!(message,
                ServerMsg::Events { events, .. } if events.iter().any(|event| matches!(event,
                    EventKind::Healed { source: Some(source), target, amount, mana: 0 }
                    if *source == hero_id && *target == hero_id && *amount > 0)))));
            consumed = true;
            break;
        }
    }
    assert!(consumed, "Tango did not consume a charge");
    let before = tracker.own_hero().unwrap().hp;
    wait_ticks(arena, tracker, 60);
    let hero = tracker.own_hero().unwrap();
    assert!(hero.hp < hero.max_hp, "healing must not clip at full HP");
    assert!(hero.hp - before > baseline + 10, "Tango did not heal");
    assert_eq!(hero.items[0].unwrap().charges, Some(2));
}

#[test]
fn tango_rejects_non_tree_and_missing_targets_without_broadening_quelling_range() {
    let (_, tracker) = tango_arena();
    let space = ActionSpace::from_tracker(&tracker).unwrap();
    let points = space.point_candidates();
    assert!(space.allows(tango_action(tree_target(&space))));
    let invalid = points
        .iter()
        .position(|point| !point.standing_tree)
        .unwrap();
    let invalid = tango_action(ActionTarget::Point(PointIndex(invalid)));
    assert_eq!(
        space.decode(invalid).unwrap_err().to_string(),
        "action Use is masked by the current action space"
    );
    let count = points.len();
    let missing = tango_action(ActionTarget::Point(PointIndex(count)));
    assert_eq!(
        space.decode(missing).unwrap_err().to_string(),
        format!("point target index {count} is outside candidate count {count}")
    );
    let quelling = tracker.own_hero().unwrap().items[1].unwrap();
    let beyond_quelling = points
        .iter()
        .position(|point| {
            point.standing_tree && !ORIGIN.within(point.position, Fixed::from_int(quelling.range))
        })
        .unwrap();
    let mask = space
        .use_target_mask(ControlledUnit::Hero, ItemSlot(1))
        .unwrap();
    assert!(!mask.allows(ActionTarget::Point(PointIndex(beyond_quelling))));
}

#[test]
fn tango_distant_tree_stays_masked_when_dead_disabled_or_item_not_ready() {
    // Mutate only the projected snapshot to exercise each existing local guard.
    let (_, mut tracker) = tango_arena();
    let original = tracker.current().unwrap().clone();
    let cases = [
        (StatusFlags::DEAD, 0, 0, 0, 3),
        (StatusFlags::STUNNED, 200, 0, 0, 3),
        (StatusFlags::FEARED, 200, 0, 0, 3),
        (StatusFlags::CHANNELLING, 200, 0, 0, 3),
        (0, 200, 1, 0, 3),
        (0, 200, 0, 1, 3),
        (0, 200, 0, 0, 0),
    ];
    for (index, (status, hp, cooldown, mute, charges)) in cases.into_iter().enumerate() {
        let mut view = original.clone();
        view.tick += index as u32 + 1;
        let hero = view
            .units
            .iter_mut()
            .find(|unit| unit.id == original.players[0].unit.unwrap())
            .unwrap();
        hero.statuses.bits = status;
        hero.hp = hp;
        let item = hero.items[0].as_mut().unwrap();
        item.cooldown_left = cooldown;
        item.mute_left = mute;
        item.charges = Some(charges);
        tracker.observe_snapshot(&view).unwrap();
        let blocked = ActionSpace::from_tracker(&tracker).unwrap();
        let slots = blocked.item_slot_mask(ControlledUnit::Hero);
        assert!(!slots.first().copied().unwrap_or(false));
    }
}

fn tree_target(space: &ActionSpace) -> ActionTarget {
    let points = space.point_candidates();
    let index = points
        .iter()
        .position(|point| point.position == TREE)
        .unwrap();
    assert!(points[index].standing_tree);
    ActionTarget::Point(PointIndex(index))
}

fn tango_action(target: ActionTarget) -> StructuredAction {
    StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(0),
        target,
    }
}

fn tango_arena() -> (Arena, StateTracker) {
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 35_071,
    })
    .unwrap();
    let ServerMsg::MatchStart { info } = &start.messages[0][0] else {
        panic!("MatchStart first")
    };
    let configured = arena.configure_for_test(|world| {
        for item in [ItemId(7), ItemId(5)] {
            assert!(world.buy(SlotId(0), item, &mut Vec::new()));
        }
        let hero = world.seats[0].unit.unwrap();
        world.transform.get_mut(hero).unwrap().pos = ORIGIN;
        world.modifiers.get_mut(hero).unwrap().0.clear();
        world.health.get_mut(hero).unwrap().hp = Fixed::from_int(200);
    });
    let mut tracker = StateTracker::new(SlotId(0), info).unwrap();
    let view = snapshot(&configured.messages[0]);
    tracker.observe_snapshot(view).unwrap();
    assert_eq!(tracker.own_hero().unwrap().pos, ORIGIN);
    (arena, tracker)
}

fn step(arena: &mut Arena, tracker: &mut StateTracker, order: Option<Order>) -> crate::ArenaStep {
    let request = order.map(|order| Request {
        seq: arena.tick(),
        unit: None,
        order,
    });
    let advanced = arena.step(&[request, None]).unwrap();
    for message in &advanced.messages[0] {
        match message {
            ServerMsg::Snapshot { view } => tracker.observe_snapshot(view).unwrap(),
            ServerMsg::Events { tick, events } => tracker.observe_events(*tick, events).unwrap(),
            ServerMsg::OrderRejected { reason, .. } => {
                panic!("native order {order:?} rejected: {reason:?}")
            }
            _ => {}
        }
    }
    assert!(arena.tick() < 900, "fixture must finish before lane waves");
    for effect in &tracker.own_hero().unwrap().effects {
        assert!(
            !matches!(effect.id, EffectId(3 | 13)),
            "aura confounds healing"
        );
    }
    advanced
}

fn wait_ticks(arena: &mut Arena, tracker: &mut StateTracker, ticks: u32) {
    assert!(ticks > 0);
    assert!(ticks <= 60);
    for _ in 0..ticks {
        step(arena, tracker, None);
    }
}
