use super::support::RecordingWire;
use crate::{
    ActionKind, Arena, ArenaConfig, MODEL_PARAMETER_COUNT, PolicyModel, Request, Seated, Wire,
    global_feature, play_neural_on,
};
use bota_proto::{
    AbilityId, AbilitySlot, DamageKind, EntityId, EventKind, MapId, Order, PlayerId, ServerMsg,
    SlotId, Target, TickMode, UnitKind, Vec2,
};
use bota_server::game::{UnitOrder, World};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scenario {
    Stable,
    VisibilityGap,
    OwnCast,
}

impl Scenario {
    const fn tick_limit(self) -> u32 {
        // Stable and fog traces need the baseline chase horizon to observe damage.
        match self {
            Self::Stable | Self::VisibilityGap => 121,
            Self::OwnCast => 76,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum DiagnosticPolicy {
    Constant(ActionKind),
    ActiveOrderReadout,
}

struct NativeWire<'a> {
    arena: Arena,
    wire: RecordingWire,
    model: &'a PolicyModel,
    side: usize,
    scenario: Scenario,
    target: EntityId,
    tick: u32,
    pending: Option<Request>,
    sent: Vec<(u32, Order)>,
    trace: Vec<ServerMsg>,
}

#[test]
fn public_neural_native_trace_preserves_attack_through_own_cast_and_resends_after_fog() {
    for side in 0..2 {
        for scenario in [Scenario::Stable, Scenario::VisibilityGap, Scenario::OwnCast] {
            assert_native_trace(side, scenario);
        }
    }
}

fn assert_native_trace(side: usize, scenario: Scenario) {
    let limit = scenario.tick_limit();
    let model = PolicyModel::fresh(10_092_000).expect("untrained diagnostic model");
    let (arena, messages) = combat_start(MapId(0), 10_092_000, side, Some(scenario));
    let ServerMsg::Snapshot { view } = &messages[1] else {
        panic!("snapshot")
    };
    let hero = view.players[side].unit.expect("hero");
    let target = view.players[1 - side].unit.expect("target");
    let mut wire = NativeWire {
        arena,
        model: &model,
        side,
        scenario,
        target,
        tick: 0,
        pending: None,
        wire: RecordingWire {
            messages: messages.into(),
            orders: Vec::new(),
            acknowledgements: Vec::new(),
        },
        sent: Vec::new(),
        trace: Vec::new(),
    };
    let outcome = play_neural_on(&mut wire, seated(side), Some(limit), &model)
        .expect("native Neural, no Teacher");
    let attack = Order::Attack {
        target: Target::Unit(target),
    };
    let mut expected = vec![(1, attack)];
    match scenario {
        Scenario::Stable => {}
        Scenario::VisibilityGap => expected.push((10, attack)),
        Scenario::OwnCast => expected.push((
            4,
            Order::Cast {
                slot: AbilitySlot(0),
                target: Target::None,
            },
        )),
    }
    assert_eq!(wire.sent, expected, "complete public request trace");
    assert_training_trace(side, scenario, &wire.trace, &expected);
    let summary = (outcome.ticks, outcome.decisions, outcome.rejections);
    assert_eq!(summary, (limit, (limit - 1) / 3, 0));
    assert_eq!(outcome.winner, None);
    assert_eq!(wire.wire.acknowledgements, (1..=limit).collect::<Vec<_>>());
    assert_eq!(wire.trace.len(), 2 * limit as usize);
    assert!(wire.trace.iter().any(|message| matches!(message,
        ServerMsg::Events { tick, events } if *tick > 10 && events.iter().any(|event| matches!(event,
            EventKind::Damaged { source: Some(source), target: damaged, kind: DamageKind::Physical, amount, .. }
            if *source == hero && *damaged == target && *amount > 0)))), "attack must actually damage its target: side={side} scenario={scenario:?} limit={limit}");
    if scenario == Scenario::OwnCast {
        assert!(wire.trace.iter().any(
            |message| matches!(message, ServerMsg::Events { tick: 5, events }
            if events.contains(&EventKind::AbilityCast { caster: hero, ability: AbilityId(13) }))
        ));
    }
    for message in &wire.trace {
        if let ServerMsg::Snapshot { view } = message {
            assert_eq!(
                view.units.iter().any(|unit| unit.id == target),
                scenario != Scenario::VisibilityGap || !(4..=8).contains(&view.tick)
            );
        }
    }
}

pub(super) fn seated(side: usize) -> Seated {
    Seated {
        player: PlayerId(1),
        slot: SlotId(side as u8),
        tick_rate: 30,
        mode: TickMode::Lockstep,
    }
}

fn scheduled_policy(scenario: Scenario, tick: u32) -> DiagnosticPolicy {
    match tick {
        1 | 10 => DiagnosticPolicy::Constant(ActionKind::AttackUnit),
        4 if scenario == Scenario::OwnCast => DiagnosticPolicy::Constant(ActionKind::Cast),
        73 => DiagnosticPolicy::ActiveOrderReadout,
        _ => DiagnosticPolicy::Constant(ActionKind::Continue),
    }
}

fn assert_training_trace(
    side: usize,
    scenario: Scenario,
    trace: &[ServerMsg],
    expected: &[(u32, Order)],
) {
    let mut probe = crate::PpoOrderContractProbe::new(side, &trace[..3]);
    let model = PolicyModel::fresh(10_092_000).expect("training instrument");
    let mut sent = Vec::new();
    for messages in trace[1..].as_chunks::<2>().0 {
        let ServerMsg::Snapshot { view } = &messages[0] else {
            panic!("snapshot")
        };
        if view.tick > 1 {
            probe.observe(messages);
        }
        if !(view.tick - 1).is_multiple_of(3) {
            continue;
        }
        install_policy(&model, scheduled_policy(scenario, view.tick));
        let (_, request, _) = probe.decide(&model, true);
        if let Some(request) = request {
            sent.push((view.tick, request.order));
        }
        if scenario == Scenario::OwnCast && view.tick >= 4 {
            assert_eq!(
                probe.request_persistence().active_body_order_for(None),
                None,
                "the request ledger keeps no body order after the cast"
            );
        }
    }
    assert_eq!(sent, expected, "PPO candidate complete delivery trace");
}

impl Wire for NativeWire<'_> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        let message = self.wire.hear()?;
        if let Some(ServerMsg::Snapshot { view }) = &message {
            self.tick = view.tick;
        }
        if let Some(ServerMsg::Events { tick, .. }) = &message {
            let policy = scheduled_policy(self.scenario, *tick);
            if (*tick - 1).is_multiple_of(3) {
                install_policy(self.model, policy);
            }
        }
        if let Some(message) = &message {
            assert!(self.trace.len() < 2 * self.scenario.tick_limit() as usize);
            self.trace.push(message.clone());
        }
        Ok(message)
    }
    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        assert_eq!(unit, None);
        assert!(self.pending.is_none());
        assert!(self.sent.len() < 8);
        let seq = self.wire.order(unit, order)?;
        self.sent.push((self.tick, order));
        self.pending = Some(Request { seq, unit, order });
        Ok(seq)
    }
    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        assert_eq!(tick, self.tick);
        self.wire.acknowledge(tick)?;
        if tick == self.scenario.tick_limit() {
            return Ok(());
        }
        assert!(self.wire.messages.is_empty());
        if self.scenario == Scenario::VisibilityGap && [3, 8].contains(&tick) {
            self.arena.configure_for_test(|world| {
                let target = world.of_wire(self.target).expect("live generation");
                world.transform.get_mut(target).expect("position").pos = if tick == 3 {
                    Vec2::from_ints(4_000, 13_000)
                } else {
                    Vec2::from_ints(8_700, 8_900)
                };
            });
        }
        let mut requests = [None, None];
        requests[self.side] = self.pending.take();
        let step = self.arena.step(&requests).expect("native execution");
        self.wire
            .messages
            .extend(step.messages[self.side].iter().cloned());
        Ok(())
    }
}

pub(super) fn combat_start(
    map: MapId,
    seed: u64,
    side: usize,
    scenario: Option<Scenario>,
) -> (Arena, Vec<ServerMsg>) {
    assert!(side < 2);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map,
        seed,
    })
    .expect("arena");
    let configured = arena.configure_for_test(|world| configure_scene(world, side, scenario));
    let mut messages = std::mem::take(&mut start.messages[side]);
    messages.truncate(1);
    messages.extend(configured.messages[side].iter().cloned());
    assert_eq!(messages.len(), 3);
    (arena, messages)
}

fn configure_scene(world: &mut World, side: usize, scenario: Option<Scenario>) {
    assert_eq!(world.tick, 1);
    let heroes = [
        world.seats[side].unit.expect("actor"),
        world.seats[1 - side].unit.expect("target"),
    ];
    let removed: Vec<_> = world
        .entities
        .iter()
        .filter(|entity| {
            scenario.is_some()
                && !heroes.contains(entity)
                && world.kind.get(*entity) != Some(&UnitKind::Ancient)
        })
        .collect();
    assert!(removed.len() < 64);
    for entity in removed {
        assert!(world.despawn(entity));
    }
    for (index, hero) in heroes.into_iter().enumerate() {
        world
            .modifiers
            .insert(hero, bota_server::game::Modifiers::default());
        world.set_order(hero, UnitOrder::Stand);
        let transform = world.transform.get_mut(hero).expect("transform");
        transform.pos = match (scenario.unwrap_or(Scenario::OwnCast), index) {
            (Scenario::Stable | Scenario::VisibilityGap, 0) => Vec2::from_ints(7_800, 8_000),
            (Scenario::Stable, 1) => Vec2::from_ints(8_700, 8_900),
            (Scenario::VisibilityGap, 1) => Vec2::from_ints(7_950, 8_150),
            _ => Vec2::from_ints(8_600 + index as i32 * 200, 8_900),
        };
        transform.facing.brads = if index == 0 { 0 } else { 32_768 };
    }
    world.abilities.get_mut(heroes[0]).expect("abilities").slots[0].level = 1;
}

pub(super) fn install_policy(model: &PolicyModel, policy: DiagnosticPolicy) {
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        let values = &mut parameters[offset..offset + count];
        let name = name.strip_prefix("dire.").unwrap_or(name);
        match (policy, name) {
            (DiagnosticPolicy::Constant(kind), "kind.bias") => values[kind.index()] = 10.0,
            (DiagnosticPolicy::ActiveOrderReadout, "trunk.0.weight") => {
                assert_eq!(shape, [2812, 512]);
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
    model
        .import_parameters(&parameters)
        .expect("finite seat-visible diagnostic weights");
}
