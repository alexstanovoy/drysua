use super::neural_order_contract::DiagnosticPolicy::{self, ActiveOrderReadout, Constant};
use super::neural_order_contract::{Scenario, combat_start, install_policy, seated};
use super::support::RecordingWire;
use crate::{
    ActionKind, IssuedOrder, LocalPolicyState, OrderPersistence, PolicyModel, StateTracker, Wire,
    play_neural_on,
};
use bota_proto::{EntityId, EventKind, MapId, Order, RejectReason, ServerMsg, SlotId, Target};

#[test]
fn public_neural_reconciles_fog_death_generations_and_exact_rejections_before_readout() {
    for side in 0..2 {
        for scenario in [
            "fog",
            "death",
            "target generation",
            "body generation",
            "rejection",
            "stale rejection",
            "rollback",
            "missing snapshots",
            "expired history",
        ] {
            assert_reconciliation(side, scenario);
        }
    }
}

fn assert_reconciliation(side: usize, scenario: &str) {
    let (_, mut messages) = combat_start(MapId(0), 10_092_100, side, Some(Scenario::OwnCast));
    let ServerMsg::Snapshot { view } = &messages[1] else {
        panic!("snapshot")
    };
    let view = view.clone();
    let target = view.players[1 - side].unit.expect("target");
    let attack = Order::Attack {
        target: Target::Unit(target),
    };
    let stop = Order::Move {
        target: Target::None,
    };
    let mut expected = vec![(None, attack)];
    let start = match scenario {
        "missing snapshots" => 4,
        "expired history" => crate::HISTORY_TICKS + 2,
        _ => 2,
    };
    let mut schedule = vec![
        (1, Constant(ActionKind::AttackUnit)),
        (start.max(4), ActiveOrderReadout),
    ];
    extend_reconciliation(&mut messages, view, side, scenario, start);
    if matches!(
        scenario,
        "death"
            | "body generation"
            | "rejection"
            | "rollback"
            | "missing snapshots"
            | "expired history"
    ) {
        expected.push((None, stop));
    }
    if scenario == "rollback" {
        schedule[1].1 = Constant(ActionKind::Stop);
        schedule.push((7, ActiveOrderReadout));
    }
    if scenario == "target generation" {
        schedule.push((7, Constant(ActionKind::AttackUnit)));
        expected.push((
            None,
            Order::Attack {
                target: Target::Unit(bota_proto::EntityId {
                    generation: target.generation + 1,
                    ..target
                }),
            },
        ));
    }
    let wire = replay(messages, side, &schedule);
    assert_eq!(wire.orders, expected, "whole request trace: {scenario}");
    assert_eq!(
        wire.acknowledgements,
        std::iter::once(1)
            .chain(start..=start + 6)
            .collect::<Vec<_>>()
    );
}

/// Replays `messages` through the production neural seat; only the policy schedule is synthetic.
fn replay(
    mut messages: Vec<ServerMsg>,
    side: usize,
    schedule: &[(u32, DiagnosticPolicy)],
) -> RecordingWire {
    struct ScheduledWire<'a> {
        wire: RecordingWire,
        model: &'a PolicyModel,
        schedule: &'a [(u32, DiagnosticPolicy)],
    }
    impl Wire for ScheduledWire<'_> {
        fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
            let message = self.wire.hear()?;
            if let Some(ServerMsg::Events { tick, .. }) = &message {
                for (scheduled, policy) in self.schedule {
                    if tick == scheduled {
                        install_policy(self.model, *policy);
                    }
                }
            }
            Ok(message)
        }
        fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
            self.wire.order(unit, order)
        }
        fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
            self.wire.acknowledge(tick)
        }
    }
    assert!(messages.len() <= 32);
    assert!(schedule.len() <= 8);
    let mut final_view = messages
        .iter()
        .rev()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("final snapshot");
    final_view.tick += 1;
    let limit = final_view.tick;
    messages.push(ServerMsg::Snapshot { view: final_view });
    let model = PolicyModel::fresh(10_092_101).expect("diagnostic model");
    let mut wire = ScheduledWire {
        model: &model,
        schedule,
        wire: RecordingWire {
            messages: messages.into(),
            orders: Vec::new(),
            acknowledgements: Vec::new(),
        },
    };
    play_neural_on(&mut wire, seated(side), Some(limit), &model).expect("production Neural replay");
    assert!(wire.wire.messages.is_empty());
    wire.wire
}

fn extend_reconciliation(
    messages: &mut Vec<ServerMsg>,
    mut view: bota_proto::WorldView,
    side: usize,
    scenario: &str,
    start: u32,
) {
    let target = view.players[1 - side].unit.expect("target");
    for tick in start..=start + 5 {
        view.tick = tick;
        let mut events = Vec::new();
        if tick == start {
            match scenario {
                "fog" | "death" | "missing snapshots" | "expired history" => {
                    view.units.retain(|unit| unit.id != target)
                }
                "target generation" | "body generation" => {
                    replace_generation(
                        &mut view,
                        if scenario == "body generation" {
                            side
                        } else {
                            1 - side
                        },
                    );
                }
                "rejection" | "stale rejection" => messages.push(ServerMsg::OrderRejected {
                    seq: if scenario == "rejection" { 1 } else { 99 },
                    reason: RejectReason::UnknownTarget,
                }),
                _ => {}
            }
            if scenario == "death" {
                events.push(EventKind::Died {
                    unit: target,
                    killer: None,
                    denied: false,
                    gold: 0,
                });
            }
        }
        if scenario == "rollback" && tick == 5 {
            messages.push(ServerMsg::OrderRejected {
                seq: 2,
                reason: RejectReason::UnknownTarget,
            });
        }
        messages.push(ServerMsg::Snapshot { view: view.clone() });
        messages.push(ServerMsg::Events { tick, events });
    }
}

fn replace_generation(view: &mut bota_proto::WorldView, side: usize) {
    let id = view.players[side].unit.expect("body");
    let unit = view
        .units
        .iter_mut()
        .find(|unit| unit.id == id)
        .expect("visible body");
    unit.id.generation += 1;
    view.players[side].unit = Some(unit.id);
}

fn crowd_target(view: &mut bota_proto::WorldView) -> bota_proto::EntityId {
    let hero = view.players[0].unit.expect("hero");
    let mut creep = view
        .units
        .iter()
        .find(|unit| unit.id == hero)
        .expect("template")
        .clone();
    creep.kind = bota_proto::UnitKind::CreepMelee;
    creep.owner = None;
    creep.hero = None;
    creep.abilities.clear();
    creep.items.clear();
    for idx in 10_000..10_100 {
        creep.id = bota_proto::EntityId { idx, generation: 1 };
        view.units.push(creep.clone());
    }
    creep.id.idx = 20_000;
    creep.pos = bota_proto::Vec2::from_ints(18_000, 18_000);
    let target = creep.id;
    view.units.push(creep);
    view.units.sort_by_key(|unit| unit.id);
    target
}

#[test]
fn nonmonotonic_sequence_is_rejected_atomically_across_request_and_neural_ledgers() {
    let (_, messages) = combat_start(MapId(0), 10_092_100, 0, None);
    let ServerMsg::MatchStart { info } = &messages[0] else {
        panic!("start")
    };
    let ServerMsg::Snapshot { view } = &messages[1] else {
        panic!("snapshot")
    };
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(view).expect("snapshot");
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::None,
        },
    };
    let mut requests = OrderPersistence::default();
    let mut candidate = Some(OrderPersistence::default());
    crate::record_sent_for_policy(&mut requests, &mut candidate, 5, issued, &tracker)
        .expect("first send");
    let before = (requests, candidate);
    for sequence in [0, 4, 5] {
        let error = crate::record_sent_for_policy(
            &mut requests,
            &mut candidate,
            sequence,
            issued,
            &tracker,
        )
        .expect_err("nonmonotonic sequence");
        assert_eq!(
            error.to_string(),
            format!("order sequence {sequence} must be greater than last sent sequence 5")
        );
        assert_eq!((requests, candidate), before);
    }
    assert_chronology_atomic(tracker, view);
}

fn assert_chronology_atomic(mut tracker: StateTracker, view: &bota_proto::WorldView) {
    let target = view.players[1].unit.expect("target");
    let attack = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(target),
        },
    };
    let mut orders = OrderPersistence::default();
    orders.record_sent(1, attack).expect("attack");
    let mut local = LocalPolicyState::new(0);
    local
        .set_active_order_from_issued(1, ActionKind::AttackUnit, attack)
        .expect("active");
    local
        .note_decision(10, ActionKind::Continue)
        .expect("future decision");
    let mut missing = view.clone();
    missing.tick = 2;
    missing.units.retain(|unit| unit.id != target);
    tracker.observe_snapshot(&missing).expect("missing target");
    let mut pending = None;
    let before = (orders, local, pending);
    assert_eq!(
        orders
            .reconcile_neural_snapshot(&tracker, &mut local, &mut pending)
            .expect_err("chronology")
            .to_string(),
        "local policy tick 2 is older than latest tick 10"
    );
    assert_eq!((orders, local, pending), before);
}

#[test]
fn visible_target_outside_candidate_cap_does_not_lose_persistence() {
    let (_, messages) = combat_start(MapId(0), 10_092_100, 0, None);
    let ServerMsg::MatchStart { info } = &messages[0] else {
        panic!("start")
    };
    let ServerMsg::Snapshot { view } = &messages[1] else {
        panic!("snapshot")
    };
    let mut view = view.clone();
    let target = crowd_target(&mut view);
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(&view).expect("crowded snapshot");
    let attack = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(target),
        },
    };
    let mut orders = OrderPersistence::default();
    orders.record_sent(1, attack).expect("attack");
    let mut local = LocalPolicyState::new(0);
    local
        .set_active_order_from_issued(1, ActionKind::AttackUnit, attack)
        .expect("active");
    let active = local.active_order();
    view.tick = 2;
    tracker.observe_snapshot(&view).expect("next snapshot");
    orders
        .reconcile_neural_snapshot(&tracker, &mut local, &mut None)
        .expect("reconcile");
    let space = crate::ActionSpace::from_tracker(&tracker).expect("capped space");
    assert_eq!(space.entity_candidates().len(), 96);
    assert_eq!(space.entity_index(target), None);
    assert_eq!(orders.should_send(Some(attack)), None);
    assert_eq!(local.active_order(), active);
}
