use std::collections::VecDeque;

use bota_proto::{
    AbilityId, AbilitySlot, Aim, DamageKind, EntityId, EventKind, MapId, MatchInfo, Order,
    PlayerId, ServerMsg, SlotId, Target, Team, TickMode, UnitKind, Vec2, WorldView,
};
use bota_server::game::{UnitOrder, World, wire_id};

use crate::{
    ActionKind, ActionSpace, Arena, ArenaConfig, ControlledUnit, FeatureFrame, IssuedOrder,
    MODEL_PARAMETER_COUNT, Outcome, PolicyModel, Request, Seated, StateTracker, StructuredAction,
    Wire, global_feature, play_neural_on,
};

const SEED: u64 = 10_092_000;
const MAX_TICKS: u32 = 121;
const READOUT_TICK: u32 = 73;
const MAX_ORDERS: usize = 8;
const MAX_EVENTS: usize = 64;
const REATTACK_TICK: u32 = 10;

// The archived order-contract report records these assertions failing before the fix.
#[test]
fn radiant_reappearance_must_allow_the_same_attack_to_be_sent_again() {
    assert_reappearance_contract(0);
}

#[test]
fn dire_reappearance_must_allow_the_same_attack_to_be_sent_again() {
    assert_reappearance_contract(1);
}

#[test]
fn radiant_own_cast_then_continue_must_preserve_active_attack_tracking() {
    assert_own_cast_tracking(0);
}

#[test]
fn dire_own_cast_then_continue_must_preserve_active_attack_tracking() {
    assert_own_cast_tracking(1);
}

#[test]
fn stable_visibility_suppresses_only_a_redundant_attack_without_changing_gameplay() {
    for side in 0..2 {
        let actual = run_fixture(side, Scenario::Stable, false);
        let reference = run_fixture(side, Scenario::Stable, true);

        assert!(
            actual
                .trace
                .snapshots
                .iter()
                .all(|view| target_visible(view, actual.target))
        );
        assert_eq!(actual.trace.sent, vec![(1, attack(actual.target))]);
        assert_eq!(
            reference.trace.applied.len(),
            actual.trace.applied.len() + 1
        );
        assert_eq!(actual.trace.snapshots, reference.trace.snapshots);
        assert_eq!(actual.trace.events, reference.trace.events);
        assert!(
            actual
                .trace
                .physical_damage(actual.hero, actual.target, 1, MAX_TICKS)
                > 0
        );
        assert_ne!(
            actual.trace.hero_position(1, actual.hero),
            actual.trace.hero_position(MAX_TICKS, actual.hero)
        );
        print_evidence("stable_visibility", &actual);
    }
}

#[test]
fn explicit_wire_reattack_after_visibility_gap_restores_chase_and_damage() {
    for side in 0..2 {
        let reference = run_fixture(side, Scenario::VisibilityGap, true);

        assert_visibility_gap(&reference);
        assert_eq!(reference.trace.order_at(11), attack(reference.target));
        assert!(
            reference
                .trace
                .applied
                .contains(&(REATTACK_TICK, attack(reference.target)))
        );
        assert!(
            reference
                .trace
                .physical_damage(reference.hero, reference.target, 11, MAX_TICKS)
                > 0
        );
        print_evidence("explicit_wire_reattack", &reference);
    }
}

#[test]
fn own_cast_then_continue_keeps_the_server_attack_and_deals_visible_damage() {
    for side in 0..2 {
        let actual = run_fixture(side, Scenario::OwnCast, false);

        assert_cast_and_continue(&actual);
        assert!(
            actual
                .trace
                .snapshots
                .iter()
                .all(|view| target_visible(view, actual.target))
        );
        assert!(
            actual
                .trace
                .physical_damage(actual.hero, actual.target, 6, READOUT_TICK)
                > 0
        );
        print_evidence("own_cast_server_control", &actual);
    }
}

#[test]
fn continue_without_cast_preserves_neural_active_order_tracking() {
    for side in 0..2 {
        let actual = run_fixture(side, Scenario::NoCast, false);

        assert_eq!(actual.trace.order_at(7), attack(actual.target));
        assert_eq!(actual.trace.sent, vec![(1, attack(actual.target))]);
        assert!(
            actual
                .trace
                .physical_damage(actual.hero, actual.target, 6, READOUT_TICK)
                > 0
        );
        print_evidence("no_cast_tracking_control", &actual);
    }
}

#[test]
fn active_order_readout_depends_only_on_the_existing_presence_feature() {
    let model = PolicyModel::fresh(SEED).expect("diagnostic model");
    install_policy(&model, DiagnosticPolicy::ActiveOrderReadout);
    let mut frame = FeatureFrame::new();
    let absent = model.evaluate(&frame).expect("absent order readout");
    frame.global[global_feature::ACTIVE_ORDER_PRESENT] = 1.0;
    let present = model.evaluate(&frame).expect("present order readout");

    assert_eq!(absent.kind_logits[ActionKind::Continue.index()], 0.0);
    assert_eq!(absent.kind_logits[ActionKind::Stop.index()], 1.0);
    assert_eq!(present.kind_logits[ActionKind::Continue.index()], 2.0);
    assert_eq!(present.kind_logits[ActionKind::Stop.index()], 1.0);
}

fn assert_reappearance_contract(side: usize) {
    let actual = run_fixture(side, Scenario::VisibilityGap, false);
    let reference = run_fixture(side, Scenario::VisibilityGap, true);
    assert_visibility_gap(&actual);
    assert_visibility_gap(&reference);
    assert_eq!(
        actual.trace.snapshots[..10],
        reference.trace.snapshots[..10]
    );
    assert_eq!(reference.trace.order_at(11), attack(reference.target));
    assert!(
        reference
            .trace
            .physical_damage(reference.hero, reference.target, 11, MAX_TICKS)
            > 0
    );
    print_evidence("reappearance_actual", &actual);
    print_evidence("reappearance_wire_reference", &reference);

    assert!(
        actual
            .trace
            .sent
            .contains(&(REATTACK_TICK, attack(actual.target))),
        "Neural must resend the same AttackUnit after observed target disappearance and reappearance; the server no longer holds that attack"
    );
    assert_eq!(actual.trace.order_at(11), attack(actual.target));
    assert_eq!(actual.trace.snapshots, reference.trace.snapshots);
}

fn assert_own_cast_tracking(side: usize) {
    let actual = run_fixture(side, Scenario::OwnCast, false);
    assert_cast_and_continue(&actual);
    assert!(
        actual
            .trace
            .physical_damage(actual.hero, actual.target, 6, READOUT_TICK)
            > 0
    );
    print_evidence("own_cast_tracking_actual", &actual);

    assert!(
        !actual
            .trace
            .sent
            .iter()
            .any(|(tick, _)| *tick == READOUT_TICK),
        "Aim::Own cast preserves the server AttackUnit: after Continue, Neural ACTIVE_ORDER_PRESENT must remain one (readout Continue), not zero (readout Stop)"
    );
    assert_eq!(actual.trace.order_at(76), attack(actual.target));
}

fn assert_visibility_gap(fixture: &CompletedFixture) {
    assert!(target_visible(fixture.trace.snapshot(1), fixture.target));
    for tick in 4..=8 {
        assert!(
            !target_visible(fixture.trace.snapshot(tick), fixture.target),
            "target visible at {tick}"
        );
    }
    for tick in [9, 10, 11] {
        assert!(
            target_visible(fixture.trace.snapshot(tick), fixture.target),
            "same handle missing at {tick}"
        );
    }
    assert!(matches!(
        fixture.trace.order_at(4),
        Order::Attack {
            target: Target::Pos(_)
        }
    ));
    assert!(matches!(
        fixture.trace.order_at(10),
        Order::Attack {
            target: Target::Pos(_)
        }
    ));
    assert_eq!(fixture.trace.sent[0], (1, attack(fixture.target)));
}

fn assert_cast_and_continue(fixture: &CompletedFixture) {
    assert_eq!(fixture.trace.sent[0], (1, attack(fixture.target)));
    assert_eq!(
        fixture.trace.sent[1],
        (
            4,
            Order::Cast {
                slot: AbilitySlot(0),
                target: Target::None
            }
        )
    );
    assert!(fixture.trace.events.contains(&(
        5,
        EventKind::AbilityCast {
            caster: fixture.hero,
            ability: AbilityId(13),
        }
    )));
    assert!(
        !fixture
            .trace
            .sent
            .iter()
            .any(|(tick, _)| (7..READOUT_TICK).contains(tick))
    );
    assert_eq!(fixture.trace.order_at(7), attack(fixture.target));
    assert_eq!(fixture.trace.order_at(70), attack(fixture.target));
    let ability = &fixture
        .trace
        .snapshot(1)
        .units
        .iter()
        .find(|unit| unit.id == fixture.hero)
        .expect("seat-visible caster")
        .abilities[0];
    assert_eq!(ability.aim, Aim::Own);
    assert_eq!(ability.id, AbilityId(13));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Stable,
    VisibilityGap,
    OwnCast,
    NoCast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiagnosticPolicy {
    Constant(ActionKind),
    ActiveOrderReadout,
}

struct CompletedFixture {
    info: MatchInfo,
    side: usize,
    hero: EntityId,
    target: EntityId,
    outcome: Outcome,
    trace: ProtocolTrace,
}

#[derive(Default)]
struct ProtocolTrace {
    snapshots: Vec<WorldView>,
    events: Vec<(u32, EventKind)>,
    sent: Vec<(u32, Order)>,
    applied: Vec<(u32, Order)>,
    server_orders: Vec<(u32, Order)>,
}

impl ProtocolTrace {
    fn snapshot(&self, tick: u32) -> &WorldView {
        assert!((1..=MAX_TICKS).contains(&tick));
        let view = &self.snapshots[tick as usize - 1];
        assert_eq!(view.tick, tick);
        view
    }

    fn order_at(&self, tick: u32) -> Order {
        assert!((1..=MAX_TICKS).contains(&tick));
        assert!(self.server_orders.len() <= 8);
        self.server_orders
            .iter()
            .find(|(observed, _)| *observed == tick)
            .expect("fixture order inspection")
            .1
    }

    fn hero_position(&self, tick: u32, hero: EntityId) -> Vec2 {
        let view = self.snapshot(tick);
        assert!(view.units.len() <= 4);
        view.units
            .iter()
            .find(|unit| unit.id == hero)
            .expect("seat-visible hero")
            .pos
    }

    fn physical_damage(&self, hero: EntityId, target: EntityId, start: u32, end: u32) -> i32 {
        assert!(start <= end);
        assert!(self.events.len() <= MAX_EVENTS);
        self.events
            .iter()
            .filter_map(|(tick, event)| match event {
                EventKind::Damaged {
                    source: Some(source),
                    target: damaged,
                    amount,
                    kind: DamageKind::Physical,
                    ..
                } if *source == hero && *damaged == target && (start..=end).contains(tick) => {
                    Some(*amount)
                }
                _ => None,
            })
            .sum()
    }
}

struct ArenaWire<'model> {
    arena: Arena,
    info: MatchInfo,
    model: &'model PolicyModel,
    policy: DiagnosticPolicy,
    scenario: Scenario,
    side: usize,
    limit: u32,
    target: EntityId,
    force_reattack: bool,
    queue: VecDeque<ServerMsg>,
    pending: Option<Request>,
    sequence: u32,
    trace: ProtocolTrace,
}

impl Wire for ArenaWire<'_> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        assert!(self.queue.len() <= 3);
        let message = self.queue.pop_front().expect("fixture protocol message");
        if let ServerMsg::Events { tick, .. } = &message {
            let policy = scheduled_policy(self.scenario, *tick);
            if policy == DiagnosticPolicy::Constant(ActionKind::AttackUnit) {
                assert_targeted_attack_is_legal(
                    &self.info,
                    self.side,
                    self.trace.snapshot(*tick),
                    self.target,
                );
            }
            if policy != self.policy {
                install_policy(self.model, policy);
                self.policy = policy;
            }
        }
        self.record_message(&message);
        Ok(Some(message))
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        assert_eq!(unit, None, "diagnostic policy must control its hero");
        assert!(self.pending.is_none());
        assert!(self.trace.sent.len() < MAX_ORDERS);
        self.sequence += 1;
        self.trace.sent.push((self.arena.tick(), order));
        self.pending = Some(Request {
            seq: self.sequence,
            unit,
            order,
        });
        Ok(self.sequence)
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        assert_eq!(tick, self.arena.tick());
        assert!(tick <= self.limit);
        if tick == self.limit {
            return Ok(());
        }
        assert!(self.queue.is_empty());
        let mut request = self.pending.take();
        if self.force_reattack && tick == REATTACK_TICK && request.is_none() {
            request = Some(Request {
                seq: 1_000,
                unit: None,
                order: attack(self.target),
            });
        }
        if let Some(request) = request {
            assert!(self.trace.applied.len() < MAX_ORDERS);
            self.trace.applied.push((tick, request.order));
        }
        self.configure_visibility_transition(tick);
        let mut requests = [None, None];
        requests[self.side] = request;
        let step = self.arena.step(&requests).expect("short fixture tick");
        for messages in &step.messages {
            assert_eq!(
                messages.len(),
                2,
                "fixture must have only Snapshot and Events, never rejection or MatchOver"
            );
        }
        self.queue.extend(step.messages[self.side].iter().cloned());
        Ok(())
    }
}

impl ArenaWire<'_> {
    fn record_message(&mut self, message: &ServerMsg) {
        match message {
            ServerMsg::Snapshot { view } => {
                assert!(self.trace.snapshots.len() < MAX_TICKS as usize);
                assert_eq!(view.tick as usize, self.trace.snapshots.len() + 1);
                if [1, 4, 7, 10, 11, 70, 76, 121].contains(&view.tick) {
                    let order = inspect_server_order(&mut self.arena, self.side, view);
                    self.trace.server_orders.push((view.tick, order));
                }
                self.trace.snapshots.push(view.clone());
            }
            ServerMsg::Events { tick, events } => {
                assert!(self.trace.events.len() + events.len() <= MAX_EVENTS);
                self.trace
                    .events
                    .extend(events.iter().cloned().map(|event| (*tick, event)));
            }
            ServerMsg::MatchStart { .. } => {}
            _ => panic!("unexpected fixture protocol message: {message:?}"),
        }
    }

    fn configure_visibility_transition(&mut self, tick: u32) {
        assert!(tick < self.limit);
        assert!(self.side < 2);
        if self.scenario != Scenario::VisibilityGap || ![3, 8].contains(&tick) {
            return;
        }
        let position = if tick == 3 {
            Vec2::from_ints(13_000, 13_000)
        } else {
            Vec2::from_ints(8_700, 8_900)
        };
        self.arena.configure_for_test(|world| {
            let target = world
                .of_wire(self.target)
                .expect("same live target generation");
            world
                .transform
                .get_mut(target)
                .expect("target transform")
                .pos = position;
        });
    }
}

fn run_fixture(side: usize, scenario: Scenario, force_reattack: bool) -> CompletedFixture {
    assert!(side < 2);
    assert!(!force_reattack || matches!(scenario, Scenario::Stable | Scenario::VisibilityGap));
    let model = PolicyModel::fresh(SEED).expect("diagnostic model, no trained weights");
    let policy = DiagnosticPolicy::Constant(ActionKind::AttackUnit);
    install_policy(&model, policy);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(0),
        seed: SEED,
    })
    .expect("real Map0 Arena");
    let configured = arena.configure_for_test(|world| configure_scene(world, side, scenario));
    let ServerMsg::MatchStart { info } = start.messages[side][0].clone() else {
        panic!("match metadata")
    };
    start.messages[side].truncate(1);
    start.messages[side].extend(configured.messages[side].iter().cloned());
    let ServerMsg::Snapshot { view } = &start.messages[side][1] else {
        panic!("initial snapshot")
    };
    let hero = view.players[side].unit.expect("controlled hero");
    let target = view.players[1 - side].unit.expect("target hero");
    assert!(target_visible(view, target));
    let limit = if matches!(scenario, Scenario::OwnCast | Scenario::NoCast) {
        76
    } else {
        MAX_TICKS
    };
    let mut wire = ArenaWire {
        arena,
        info,
        model: &model,
        policy,
        scenario,
        side,
        limit,
        target,
        force_reattack,
        queue: start.messages[side].clone().into(),
        pending: None,
        sequence: 0,
        trace: ProtocolTrace::default(),
    };
    let seated = Seated {
        player: PlayerId(1),
        slot: SlotId(side as u8),
        tick_rate: 30,
        mode: TickMode::Lockstep,
    };
    let outcome =
        play_neural_on(&mut wire, seated, Some(limit), &model).expect("actual pure Neural runtime");
    assert_eq!(
        outcome.winner, None,
        "mechanics fixture, not a scored match"
    );
    assert_eq!(outcome.rejections, 0);
    assert_eq!(outcome.ticks, limit);
    assert_eq!(outcome.decisions, (limit - 1) / 3);
    CompletedFixture {
        info: wire.info,
        side,
        hero,
        target,
        outcome,
        trace: wire.trace,
    }
}

fn configure_scene(world: &mut World, side: usize, scenario: Scenario) {
    assert_eq!(world.tick, 1);
    assert_eq!(world.seats.len(), 2);
    let heroes = [
        world.seats[side].unit.expect("actor"),
        world.seats[1 - side].unit.expect("target"),
    ];
    let removed: Vec<_> = world
        .entities
        .iter()
        .filter(|entity| {
            !heroes.contains(entity) && world.kind.get(*entity) != Some(&UnitKind::Ancient)
        })
        .collect();
    assert!(removed.len() < 64);
    for entity in removed {
        assert!(world.despawn(entity));
    }
    for (index, hero) in heroes.into_iter().enumerate() {
        world.statuses.remove(hero);
        world.set_order(hero, UnitOrder::Stand);
        let transform = world.transform.get_mut(hero).expect("hero transform");
        transform.pos = match (scenario, index) {
            (Scenario::Stable | Scenario::VisibilityGap, 0) => Vec2::from_ints(7_800, 8_000),
            (Scenario::Stable, 1) => Vec2::from_ints(8_700, 8_900),
            (Scenario::VisibilityGap, 1) => Vec2::from_ints(7_950, 8_150),
            _ => Vec2::from_ints(8_600 + index as i32 * 200, 8_900),
        };
        transform.facing.brads = if index == 0 { 0 } else { 32_768 };
    }
    world
        .abilities
        .get_mut(heroes[0])
        .expect("actor abilities")
        .slots[0]
        .level = 1;
}

fn inspect_server_order(arena: &mut Arena, side: usize, before: &WorldView) -> Order {
    assert_eq!(arena.tick(), before.tick);
    let mut order = None;
    let inspected = arena.configure_for_test(|world| {
        let hero = world.seats[side].unit.expect("live fixture hero");
        order = Some(
            match world.orders.get(hero).expect("server body order").current {
                UnitOrder::Attack { target, .. } => attack(wire_id(target)),
                UnitOrder::AttackMove { pos } => Order::Attack {
                    target: Target::Pos(pos),
                },
                UnitOrder::Stand => Order::Move {
                    target: Target::None,
                },
                other => panic!("unexpected server fixture order: {other:?}"),
            },
        );
    });
    assert_eq!(
        inspected.messages[side][0],
        ServerMsg::Snapshot {
            view: before.clone()
        },
        "inspection must not change seat-visible state"
    );
    order.expect("test-only server order evidence")
}

fn assert_targeted_attack_is_legal(
    info: &MatchInfo,
    side: usize,
    view: &WorldView,
    target: EntityId,
) {
    let mut tracker = StateTracker::new(SlotId(side as u8), info).expect("seat-only mask check");
    tracker
        .observe_snapshot(view)
        .expect("real seat projection");
    let space = ActionSpace::from_tracker(&tracker).expect("current legal action space");
    let index = space
        .entity_index(target)
        .expect("same visible target in candidates");
    assert_eq!(
        space
            .attack_entity_mask(ControlledUnit::Hero)
            .iter()
            .position(|allowed| *allowed),
        Some(index.0),
        "zero pointer logits must select the intended target, not a structure or another body"
    );
    let action = StructuredAction::AttackUnit {
        unit: ControlledUnit::Hero,
        target: index,
    };
    assert!(space.allows(action));
    assert_eq!(
        space.decode(action).expect("legal attack decode"),
        Some(IssuedOrder {
            unit: None,
            order: attack(target)
        })
    );
}

fn scheduled_policy(scenario: Scenario, tick: u32) -> DiagnosticPolicy {
    assert!((1..=MAX_TICKS).contains(&tick));
    if tick == 1 {
        return DiagnosticPolicy::Constant(ActionKind::AttackUnit);
    }
    if matches!(scenario, Scenario::Stable | Scenario::VisibilityGap) && tick == REATTACK_TICK {
        return DiagnosticPolicy::Constant(ActionKind::AttackUnit);
    }
    if scenario == Scenario::OwnCast && tick == 4 {
        return DiagnosticPolicy::Constant(ActionKind::Cast);
    }
    if matches!(scenario, Scenario::OwnCast | Scenario::NoCast) && tick == READOUT_TICK {
        return DiagnosticPolicy::ActiveOrderReadout;
    }
    DiagnosticPolicy::Constant(ActionKind::Continue)
}

fn install_policy(model: &PolicyModel, policy: DiagnosticPolicy) {
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("model schema") {
        let count = shape.iter().product::<usize>();
        let values = &mut parameters[offset..offset + count];
        match (policy, name) {
            (DiagnosticPolicy::Constant(kind), "kind.bias") => values[kind.index()] = 10.0,
            (DiagnosticPolicy::ActiveOrderReadout, "trunk.0.weight") => {
                assert_eq!(shape, [2576, 512]);
                values[global_feature::ACTIVE_ORDER_PRESENT * shape[1]] = 1.0;
            }
            (DiagnosticPolicy::ActiveOrderReadout, "trunk.1.weight" | "trunk.2.weight") => {
                values[0] = 1.0
            }
            (DiagnosticPolicy::ActiveOrderReadout, "kind.weight") => {
                values[ActionKind::Continue.index()] = 2.0
            }
            (DiagnosticPolicy::ActiveOrderReadout, "kind.bias") => {
                values[ActionKind::Stop.index()] = 1.0
            }
            _ => {}
        }
        offset += count;
    }
    assert_eq!(offset, MODEL_PARAMETER_COUNT);
    // Only constant choices or the existing seat-visible active-order bit drive
    // this instrument. No fixture World, hidden target or seed enters the model.
    model
        .import_parameters(&parameters)
        .expect("finite diagnostic parameters");
}

fn attack(target: EntityId) -> Order {
    Order::Attack {
        target: Target::Unit(target),
    }
}

fn target_visible(view: &WorldView, target: EntityId) -> bool {
    assert!(view.viewer.is_some());
    assert!(view.units.len() <= 4);
    view.units.iter().any(|unit| unit.id == target)
}

fn print_evidence(label: &str, fixture: &CompletedFixture) {
    let last = fixture.outcome.ticks;
    assert!(last <= MAX_TICKS);
    assert_eq!(fixture.outcome.rejections, 0);
    eprintln!(
        "order_contract label={label} seed={SEED} side={:?} ticks={last} rejections=0 winner=None",
        if fixture.side == 0 {
            Team::Radiant
        } else {
            Team::Dire
        }
    );
    eprintln!(
        "sent={:?} applied={:?}",
        fixture.trace.sent, fixture.trace.applied
    );
    eprintln!("server_orders={:?}", fixture.trace.server_orders);
    eprintln!(
        "position_at10={:?} position_at_end={:?} physical_damage_after10={}",
        fixture.trace.hero_position(10, fixture.hero),
        fixture.trace.hero_position(last, fixture.hero),
        fixture
            .trace
            .physical_damage(fixture.hero, fixture.target, 11, last)
    );
    eprintln!("events={:?}", fixture.trace.events);
}

#[test]
fn candidate_training_paths_preserve_own_cast_target_and_start_and_match_live_requests() {
    for side in 0..2 {
        for scenario in [Scenario::OwnCast, Scenario::VisibilityGap, Scenario::Stable] {
            let fixture = run_fixture(side, scenario, false);
            assert_candidate_training_replay(&fixture, scenario);
        }
    }
}

fn messages_at(fixture: &CompletedFixture, tick: u32) -> Vec<ServerMsg> {
    assert!(tick < fixture.outcome.ticks);
    let mut messages = Vec::with_capacity(3);
    if tick == 1 {
        messages.push(ServerMsg::MatchStart {
            info: fixture.info.clone(),
        });
    }
    messages.push(ServerMsg::Snapshot {
        view: fixture.trace.snapshot(tick).clone(),
    });
    messages.push(ServerMsg::Events {
        tick,
        events: fixture
            .trace
            .events
            .iter()
            .filter(|(observed, _)| *observed == tick)
            .map(|(_, event)| event.clone())
            .collect(),
    });
    assert!(messages.len() <= 3);
    messages
}

fn assert_candidate_training_replay(fixture: &CompletedFixture, scenario: Scenario) {
    use crate::{
        ActivePolicyOrder, ActivePolicyTarget, NeuralSeatOrderContractProbe, PpoOrderContractProbe,
    };
    let start = messages_at(fixture, 1);
    let mut ppo = PpoOrderContractProbe::new(fixture.side, &start);
    let mut neural = NeuralSeatOrderContractProbe::new(fixture.side, &start);
    let model = PolicyModel::fresh(SEED).expect("replay instrument");
    for tick in 1..fixture.outcome.ticks {
        if tick > 1 {
            let messages = messages_at(fixture, tick);
            ppo.observe(&messages);
            neural.observe(&messages);
        }
        if !(tick - 1).is_multiple_of(3) {
            continue;
        }
        install_policy(&model, scheduled_policy(scenario, tick));
        let (ppo_frame, ppo_request, ppo_active) = ppo.decide(&model, true);
        let (neural_frame, neural_request, neural_active) = neural.decide(&model, true);
        assert!(
            ppo_frame == neural_frame,
            "candidate input parity: {scenario:?} side={} tick={tick}",
            fixture.side
        );
        assert_eq!(
            ppo_request, neural_request,
            "candidate request parity at tick {tick}"
        );
        assert_eq!(
            ppo_active, neural_active,
            "candidate active-order parity at tick {tick}"
        );
        let expected = fixture
            .trace
            .sent
            .iter()
            .find(|(observed, _)| *observed == tick)
            .map(|(_, order)| *order);
        assert_eq!(
            ppo_request.map(|request| request.order),
            expected,
            "live/PPO delivery parity at tick {tick}"
        );
        assert_eq!(
            ppo.legacy_persistence(),
            neural.legacy_persistence(),
            "legacy shadow parity"
        );
        if scenario == Scenario::OwnCast && tick >= 4 {
            assert_eq!(
                ppo_active,
                Some(ActivePolicyOrder {
                    started_tick: 1,
                    kind: ActionKind::AttackUnit,
                    target: ActivePolicyTarget::Unit(fixture.target)
                }),
                "candidate cast must preserve the exact original target and start tick"
            );
            assert_eq!(
                ppo.legacy_persistence().active_body_order_for(None),
                None,
                "Teacher label collection must retain the legacy cast-clears-persistence contract"
            );
        }
    }
}

#[test]
fn noncandidate_training_seats_keep_legacy_cast_and_visibility_behavior() {
    use crate::{NeuralSeatOrderContractProbe, PpoOrderContractProbe};
    for scenario in [Scenario::OwnCast, Scenario::VisibilityGap] {
        let fixture = run_fixture(0, scenario, false);
        let start = messages_at(&fixture, 1);
        let mut ppo = PpoOrderContractProbe::new(0, &start);
        let mut expert = NeuralSeatOrderContractProbe::new(0, &start);
        let model = PolicyModel::fresh(SEED).expect("legacy instrument");
        for tick in 1..fixture.outcome.ticks {
            if tick > 1 {
                let messages = messages_at(&fixture, tick);
                ppo.observe(&messages);
                expert.observe(&messages);
            }
            if !(tick - 1).is_multiple_of(3) {
                continue;
            }
            install_policy(&model, scheduled_policy(scenario, tick));
            let (ppo_frame, request, active) = ppo.decide(&model, false);
            let (expert_frame, expert_request, expert_active) = expert.decide(&model, false);
            assert!(ppo_frame == expert_frame, "legacy input parity");
            assert_eq!(request, expert_request);
            assert_eq!(active, expert_active);
            if scenario == Scenario::OwnCast && tick == 4 {
                assert_eq!(active, None);
            }
            if scenario == Scenario::OwnCast && tick == READOUT_TICK {
                assert_eq!(
                    request.expect("legacy absent-order readout").order,
                    Order::Move {
                        target: Target::None
                    }
                );
            }
            if scenario == Scenario::VisibilityGap && tick == REATTACK_TICK {
                assert_eq!(request, None);
            }
        }
    }
}

#[test]
fn candidate_training_paths_wait_for_current_tick_death_evidence_before_reconciling() {
    use crate::{NeuralSeatOrderContractProbe, PpoOrderContractProbe};
    let fixture = run_fixture(0, Scenario::NoCast, false);
    let start = messages_at(&fixture, 1);
    let mut ppo = PpoOrderContractProbe::new(0, &start);
    let mut neural = NeuralSeatOrderContractProbe::new(0, &start);
    let model = PolicyModel::fresh(SEED).expect("probe");
    install_policy(&model, DiagnosticPolicy::Constant(ActionKind::AttackUnit));
    assert!(ppo.decide(&model, true).1.is_some());
    assert!(neural.decide(&model, true).1.is_some());
    let mut view = fixture.trace.snapshot(2).clone();
    view.units.retain(|unit| unit.id != fixture.target);
    view.players[1].unit = None;
    let messages = [
        ServerMsg::Snapshot { view: view.clone() },
        ServerMsg::Events {
            tick: 2,
            events: vec![EventKind::Died {
                unit: fixture.target,
                killer: None,
                denied: false,
                gold: 0,
            }],
        },
    ];
    ppo.observe(&messages);
    neural.observe(&messages);
    view.tick = 4;
    let messages = [
        ServerMsg::Snapshot { view },
        ServerMsg::Events {
            tick: 4,
            events: Vec::new(),
        },
    ];
    ppo.observe(&messages);
    neural.observe(&messages);
    install_policy(&model, DiagnosticPolicy::Constant(ActionKind::Continue));
    let (ppo_frame, _, ppo_active) = ppo.decide(&model, true);
    let (neural_frame, _, neural_active) = neural.decide(&model, true);
    assert_eq!(
        ppo_active, None,
        "final death position is not in the seat snapshot"
    );
    assert_eq!(neural_active, None);
    assert!(ppo_frame == neural_frame);
}
