use std::collections::VecDeque;
use std::io;

use crate::{
    Arena, ArenaConfig, MAP2_ID, MAP2_TICK_CAP, MAP2_TICK_RATE, PolicyModel, Request, Seated, Wire,
    play_neural_on, play_script_on,
};
use bota_proto::{
    DamageKind, EntityId, EventKind, Order, PlayerId, ServerMsg, SlotId, Team, TickMode, Vec2,
    decode_payload, encode_frame_to_vec,
};

#[derive(Debug, PartialEq, Eq)]
enum Boundary {
    Start,
    Snapshot(u32),
    Events(u32),
    Ack(u32),
    Over(u32, Team),
}

struct NativeWire {
    arena: Arena,
    side: usize,
    witness: EventKind,
    request: Option<Request>,
    messages: VecDeque<ServerMsg>,
    boundaries: Vec<Boundary>,
    orders: u32,
}

#[test]
fn native_cap_trace_keeps_complete_ticks_and_leaves_terminal_messages_after_early_exit() {
    let model = PolicyModel::fresh(10_092_900).expect("fresh CPU model");
    for controller in [None, Some(&model)] {
        for side in 0..2 {
            for terminal in [false, true] {
                native_session(side, controller, terminal);
            }
        }
    }
}

fn native_session(side: usize, model: Option<&PolicyModel>, terminal: bool) {
    let (arena, messages, witness) = primed_arena(side);
    let mut wire = NativeWire {
        arena,
        side,
        witness,
        request: None,
        messages: wire_messages(&messages),
        boundaries: Vec::new(),
        orders: 0,
    };
    let seated = Seated {
        player: PlayerId(71),
        slot: SlotId(side as u8),
        tick_rate: MAP2_TICK_RATE as u16,
        mode: TickMode::Lockstep,
    };
    let limit = Some(MAP2_TICK_CAP + u32::from(terminal));
    let outcome = match model {
        Some(model) => play_neural_on(&mut wire, seated, limit, model),
        None => play_script_on(&mut wire, seated, limit, crate::ScriptKind::Teacher),
    }
    .expect("production controller");
    assert_eq!(outcome.slot, Some(SlotId(side as u8)));
    assert_eq!(
        outcome.team,
        Some(if side == 0 { Team::Radiant } else { Team::Dire })
    );
    assert_eq!(outcome.ticks, MAP2_TICK_CAP);
    assert_eq!(outcome.winner, terminal.then_some(Team::Neutral));
    assert_eq!(outcome.rejections, 0);
    assert_eq!(outcome.decisions, 1);
    assert_eq!(outcome.orders, wire.orders);
    assert_trace(wire, terminal);
}

fn wire_messages(messages: &[ServerMsg]) -> VecDeque<ServerMsg> {
    assert!((1..=3).contains(&messages.len()));
    messages
        .iter()
        .map(|message| {
            let bytes = encode_frame_to_vec(message).expect("native frame");
            decode_payload(&bytes[4..]).expect("decoded native frame")
        })
        .collect()
}

fn assert_trace(mut wire: NativeWire, terminal: bool) {
    let mut expected = vec![
        Boundary::Start,
        Boundary::Snapshot(MAP2_TICK_CAP - 1),
        Boundary::Events(MAP2_TICK_CAP - 1),
        Boundary::Ack(MAP2_TICK_CAP - 1),
        Boundary::Snapshot(MAP2_TICK_CAP),
    ];
    if terminal {
        expected.push(Boundary::Events(MAP2_TICK_CAP));
    }
    expected.push(Boundary::Ack(MAP2_TICK_CAP));
    if terminal {
        expected.push(Boundary::Over(MAP2_TICK_CAP, Team::Neutral));
    } else {
        assert!(matches!(
            wire.messages.pop_front(),
            Some(ServerMsg::Events {
                tick: MAP2_TICK_CAP,
                ..
            })
        ));
        assert!(matches!(
            wire.messages.pop_front(),
            Some(ServerMsg::MatchOver {
                winner: Team::Neutral,
                ..
            })
        ));
    }
    assert!(wire.messages.is_empty());
    assert_eq!(wire.boundaries, expected);
}

impl Wire for NativeWire {
    fn hear(&mut self) -> io::Result<Option<ServerMsg>> {
        let message = self.messages.pop_front();
        if let Some(message) = &message {
            let boundary = match message {
                ServerMsg::MatchStart { info } => {
                    assert_eq!(info.map, MAP2_ID);
                    Boundary::Start
                }
                ServerMsg::Snapshot { view } => Boundary::Snapshot(view.tick),
                ServerMsg::Events { tick, .. } => Boundary::Events(*tick),
                ServerMsg::MatchOver { winner, stats } => Boundary::Over(stats.duration, *winner),
                other => panic!("unexpected native message: {other:?}"),
            };
            assert!(self.boundaries.len() < 8);
            self.boundaries.push(boundary);
        }
        Ok(message)
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> io::Result<u32> {
        assert_eq!(
            self.boundaries.last(),
            Some(&Boundary::Events(MAP2_TICK_CAP - 1))
        );
        assert_eq!(self.orders, 0);
        self.orders += 1;
        self.request = Some(Request {
            seq: 1,
            unit,
            order,
        });
        Ok(1)
    }

    fn acknowledge(&mut self, tick: u32) -> io::Result<()> {
        assert!(self.boundaries.len() < 8);
        self.boundaries.push(Boundary::Ack(tick));
        if tick == MAP2_TICK_CAP - 1 {
            assert!(self.messages.is_empty());
            let mut requests = [None, None];
            requests[self.side] = self.request.take();
            let final_tick = self.arena.step(&requests).expect("native final tick");
            assert_final_tick(&final_tick.messages[self.side], self.witness.clone());
            self.messages = wire_messages(&final_tick.messages[self.side]);
        }
        Ok(())
    }
}

fn primed_arena(side: usize) -> (Arena, Vec<ServerMsg>, EventKind) {
    assert!(side < 2);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MAP2_ID,
        seed: 10_092_901,
    })
    .expect("native Map2 start");
    let mut hero = None;
    let projected = arena.configure_for_test(|world| {
        world.tick = MAP2_TICK_CAP - 1;
        let unit = world.seats[side].unit.expect("assigned hero");
        world
            .modifiers
            .insert(unit, bota_server::game::Modifiers::default());
        world.transform.get_mut(unit).expect("position").pos = Vec2::from_ints(9_216, 9_216);
        hero = Some(bota_server::game::wire_id(unit));
        world.push_hit(None, unit, 1, DamageKind::Pure);
    });
    let mut initial = start.messages.swap_remove(side);
    initial.truncate(1);
    initial.extend(projected.messages[side].iter().cloned());
    assert_eq!(arena.tick(), MAP2_TICK_CAP - 1);
    (
        arena,
        initial,
        EventKind::Damaged {
            source: None,
            target: hero.expect("witness hero"),
            amount: 1,
            kind: DamageKind::Pure,
            crit: false,
        },
    )
}

fn assert_final_tick(messages: &[ServerMsg], witness: EventKind) {
    let [
        ServerMsg::Snapshot { view },
        ServerMsg::Events { tick, events },
        ServerMsg::MatchOver { winner, stats },
    ] = messages
    else {
        panic!("native final tick must be Snapshot, complete Events, MatchOver without rejection");
    };
    assert_eq!(view.tick, MAP2_TICK_CAP);
    assert_eq!(*tick, MAP2_TICK_CAP);
    assert_eq!(stats.duration, MAP2_TICK_CAP);
    assert_eq!(*winner, Team::Neutral);
    assert!(
        events.contains(&witness),
        "final damage must precede the draw"
    );
}
