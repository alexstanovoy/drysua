use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

use bota_proto::{
    AbilitySlot, EntityId, EventKind, Fixed, ItemId, Order, RejectReason, Target, TickMode,
    UnitKind, Vec2, WorldView,
};

use super::*;
use crate::{
    ActionSpace, ActionTarget, ControlledUnit, Link, Outcome, Seated, StructuredAction, Wire,
};

const OPENING_SEED: u64 = 9_204_100;
const OPENING_LIMIT: u32 = 1_402;
const LANE_CENTER: Vec2 = bota_server::game::rules::DEMO_LANE_CORNERS[1];

#[test]
fn pregame_teacher_and_tactical_reach_lane_before_first_creep_meet_with_tcp_builtin_parity() {
    assert_teacher_opponent_parity(&TacticalPolicy::default(), 1);
}

#[test]
fn pregame_nondefault_policy_preserves_arrival_and_tcp_builtin_parity_on_both_seats() {
    let policy = nondefault_policy();
    for candidate in 0..2 {
        assert_teacher_opponent_parity(&policy, candidate);
    }
}

#[test]
fn pregame_teacher_and_tactical_policy_are_in_lane_at_first_creep_meet_without_early_combat() {
    let policy = nondefault_policy();
    for controller in [
        OpeningController::Teacher,
        OpeningController::Tactical(&policy),
    ] {
        for candidate in 0..2 {
            let mut controllers = [OpeningController::Passive; 2];
            controllers[candidate] = controller;

            let authoritative = assert_opening_parity(controllers);

            for trace in &authoritative {
                assert_eq!(
                    trace.pregame_damage, 0,
                    "scenario must exclude pregame combat"
                );
                assert_eq!(trace.first_damage_tick, None);
            }
            assert!(authoritative[1 - candidate].orders.is_empty());
            assert_eq!(
                authoritative[candidate].creep_meet_in_lane_reach,
                Some(true)
            );
        }
    }
}

fn nondefault_policy() -> TacticalPolicy {
    let policy = tactical_search_founders()[TacticalMode::Fight.index()].clone();
    assert_ne!(policy, TacticalPolicy::default());
    assert_eq!(
        TacticalPolicy::from_bytes(&policy.to_bytes()).expect("current schema"),
        policy
    );
    let features =
        crate::TacticalFeatures::from_values([0.0; TACTICAL_FEATURES]).expect("features");
    assert_eq!(
        policy.choose(&features, [true; crate::TACTICAL_MODES]),
        TacticalMode::Fight
    );
    policy
}

#[derive(Clone, Copy, Debug)]
enum OpeningController<'policy> {
    Passive,
    Teacher,
    Tactical(&'policy TacticalPolicy),
}

fn assert_teacher_opponent_parity(policy: &TacticalPolicy, candidate: u8) {
    assert!(candidate < 2);
    let mut controllers = [OpeningController::Teacher; 2];
    controllers[usize::from(candidate)] = OpeningController::Tactical(policy);
    let authoritative = assert_opening_parity(controllers);
    let evaluated = evaluate_tactical_match(
        Some(policy),
        TacticalMatchConfig {
            seed: OPENING_SEED,
            candidate_seat: candidate,
            tick_limit: OPENING_LIMIT,
        },
    )
    .expect("production builtin evaluation");
    assert_eq!(
        evaluated.order_fingerprints,
        authoritative
            .each_ref()
            .map(|trace| trace.order_hash.finish())
    );
    assert_eq!(evaluated.decisions, [(OPENING_LIMIT - 1).div_ceil(3); 2]);
}

fn assert_opening_parity(controllers: [OpeningController<'_>; 2]) -> [OpeningTrace; 2] {
    let old = builtin_opening(controllers, 901);
    let expected = builtin_opening(controllers, 1);
    let (live, authoritative) = tcp_opening(controllers);
    println!(
        "pregame seed={OPENING_SEED} controllers={:?}",
        controllers.map(|controller| match controller {
            OpeningController::Passive => "Passive",
            OpeningController::Teacher => "Teacher",
            OpeningController::Tactical(_) => "Tactical",
        })
    );
    for index in 0..2 {
        expected[index].print_comparison(&old[index], &live[index].0, &authoritative[index]);
    }
    for index in 0..2 {
        assert_eq!(
            live[index].0.snapshot_hash.finish(),
            expected[index].snapshot_hash.finish()
        );
        assert_eq!(live[index].0.orders, expected[index].orders);
        assert_eq!(authoritative[index].orders, expected[index].orders);
        assert_eq!(live[index].1.rejections, 0);
        assert_eq!(expected[index].wave_spawn, Some(900));
        assert_eq!(authoritative[index].wave_spawn, Some(900));
        assert!(authoritative[index].creep_meet.expect("wave meeting") >= 900);
        if matches!(controllers[index], OpeningController::Passive) {
            assert_eq!(live[index].1.decisions, 0);
            assert!(authoritative[index].orders.is_empty());
            continue;
        }
        assert_eq!(live[index].1.decisions, (OPENING_LIMIT - 1).div_ceil(3));
        assert!(expected[index].arrival.expect("arrived") < 900);
        assert!(
            old[index].arrival.expect("late arrival")
                > old[index].creep_meet.expect("old creep meet")
        );
        assert!(
            expected[index].arrival.expect("arrival")
                < authoritative[index]
                    .creep_meet
                    .expect("actual first creep meet")
        );
    }
    authoritative
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
        observe_messages(&mut seat, &step.messages[0]).expect("movement stream");
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
    let seat = new_seat(0, &start.messages[0]).expect("real MatchStart");
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
    observe_messages(seat, &step.messages[0]).expect("pregame order accepted");
    has_cast_event(&step.messages[0])
}

fn has_cast_event(messages: &[ServerMsg]) -> bool {
    messages.iter().any(|message| matches!(message,
        ServerMsg::Events { events, .. } if events.iter().any(|event| matches!(event, EventKind::AbilityCast { .. }))))
}

struct OpeningTrace {
    slot: SlotId,
    view: Option<WorldView>,
    snapshot_hash: DefaultHasher,
    order_hash: DefaultHasher,
    orders: Vec<(u32, Option<EntityId>, Order)>,
    arrival: Option<u32>,
    horn_position: Option<Vec2>,
    wave_spawn: Option<u32>,
    creep_meet: Option<u32>,
    creep_meet_position: Option<Vec2>,
    creep_meet_in_lane_reach: Option<bool>,
    creep_meet_health: Option<(i32, i32, i32, i32)>,
    pregame_damage: u32,
    first_damage_tick: Option<u32>,
    minimum_pregame_health: Option<(u32, i32, i32, Vec2)>,
}

impl OpeningTrace {
    fn new(slot: SlotId) -> Self {
        assert!(slot.0 < 2);
        Self {
            slot,
            view: None,
            snapshot_hash: DefaultHasher::new(),
            order_hash: DefaultHasher::new(),
            orders: Vec::new(),
            arrival: None,
            horn_position: None,
            wave_spawn: None,
            creep_meet: None,
            creep_meet_position: None,
            creep_meet_in_lane_reach: None,
            creep_meet_health: None,
            pregame_damage: 0,
            first_damage_tick: None,
            minimum_pregame_health: None,
        }
    }

    fn observe(&mut self, message: &ServerMsg) {
        match message {
            ServerMsg::MatchStart { info } => {
                assert_eq!(info.pregame_ticks, 900);
                assert_eq!(info.map, MapId(1));
                assert_eq!(info.match_id, OPENING_SEED);
            }
            ServerMsg::Snapshot { view } => self.snapshot(view),
            ServerMsg::Events { tick, events } => self.observe_damage(*tick, events),
            ServerMsg::OrderRejected { seq, reason } => {
                panic!("opening rejected seq={seq} reason={reason:?}")
            }
            ServerMsg::MatchOver { .. } => panic!("opening must not end the match"),
            _ => {}
        }
    }

    fn observe_damage(&mut self, tick: u32, events: &[EventKind]) {
        let view = self.view.as_ref().expect("snapshot before Events");
        assert_eq!(tick, view.tick);
        assert!(events.len() <= 4096);
        let own = view.players[usize::from(self.slot.0)].unit;
        for event in events {
            let EventKind::Damaged {
                source,
                target,
                amount,
                ..
            } = event
            else {
                continue;
            };
            if Some(*target) == own && tick < 900 && *amount > 0 {
                self.pregame_damage = self
                    .pregame_damage
                    .checked_add(*amount as u32)
                    .expect("bounded pregame damage");
                self.first_damage_tick.get_or_insert(tick);
            }
            if self.creep_meet.is_some() || *amount <= 0 {
                continue;
            }
            let source = view.units.iter().find(|unit| Some(unit.id) == *source);
            let target = view.units.iter().find(|unit| unit.id == *target);
            let (Some(source), Some(target)) = (source, target) else {
                continue;
            };
            if !lane_creep(source.kind) || !lane_creep(target.kind) || source.team == target.team {
                continue;
            }
            let hero = view
                .units
                .iter()
                .find(|unit| Some(unit.id) == own)
                .expect("hero at creep meet");
            self.creep_meet = Some(tick);
            self.creep_meet_position = Some(hero.pos);
            self.creep_meet_health = Some((hero.hp, hero.max_hp, hero.mana, hero.max_mana));
            let enemy = if source.team == hero.team {
                target
            } else {
                source
            };
            // The far raze reaches 700 units with a 250-unit radius.
            self.creep_meet_in_lane_reach = Some(hero.pos.within(enemy.pos, Fixed::from_int(950)));
        }
    }

    fn print_comparison(&self, old: &Self, live: &Self, authoritative: &Self) {
        assert_eq!(self.slot, live.slot);
        assert_eq!(self.slot, authoritative.slot);
        println!(
            "pregame slot={} old_arrival={:?} new_arrival={:?} live_arrival={:?} old_creep_meet={:?} visible_creep_meet={:?} horn_position={:?} first_orders={:?}",
            self.slot.0,
            old.arrival,
            self.arrival,
            live.arrival,
            old.creep_meet,
            self.creep_meet,
            self.horn_position,
            &self.orders[..self.orders.len().min(5)],
        );
        println!(
            "pregame authoritative slot={} first_creep_meet={:?} position={:?} in_lane_reach={:?} hp_max_mana_max={:?} pregame_damage={} first_damage_tick={:?} minimum_pregame_health={:?}",
            self.slot.0,
            authoritative.creep_meet,
            authoritative.creep_meet_position,
            authoritative.creep_meet_in_lane_reach,
            authoritative.creep_meet_health,
            authoritative.pregame_damage,
            authoritative.first_damage_tick,
            authoritative.minimum_pregame_health,
        );
        let recovery_orders = self
            .orders
            .iter()
            .filter(|(tick, ..)| *tick >= 850 && Some(*tick) <= authoritative.creep_meet)
            .take(32)
            .collect::<Vec<_>>();
        println!(
            "pregame slot={} orders_before_meet={recovery_orders:?}",
            self.slot.0
        );
    }

    fn snapshot(&mut self, view: &WorldView) {
        assert!(view.tick <= OPENING_LIMIT);
        assert_eq!(
            view.tick,
            self.view.as_ref().map_or(1, |previous| previous.tick + 1)
        );
        bota_proto::encode_frame_to_vec(&ServerMsg::Snapshot { view: view.clone() })
            .expect("snapshot frame")
            .hash(&mut self.snapshot_hash);
        let hero = view
            .units
            .iter()
            .find(|unit| unit.owner == Some(self.slot) && unit.kind == UnitKind::Hero)
            .expect("opening hero alive");
        if view.tick < 900
            && self
                .minimum_pregame_health
                .is_none_or(|(_, hp, ..)| hero.hp < hp)
        {
            self.minimum_pregame_health = Some((view.tick, hero.hp, hero.max_hp, hero.pos));
        }
        if hero.pos.within(LANE_CENTER, Fixed::from_int(600)) {
            self.arrival.get_or_insert(view.tick);
        }
        if view.tick == 900 {
            self.horn_position = Some(hero.pos);
            let spawn = if self.slot == SlotId(0) {
                bota_server::game::rules::DEMO_RADIANT_CREEP_SPAWN
            } else {
                bota_server::game::rules::DEMO_DIRE_CREEP_SPAWN
            };
            let mut wave = view
                .units
                .iter()
                .filter(|unit| unit.team == hero.team && lane_creep(unit.kind));
            assert_eq!(wave.clone().count(), 4);
            assert!(wave.all(|unit| unit.pos.within(spawn, Fixed::from_int(200))));
        }
        if view.units.iter().any(|unit| lane_creep(unit.kind)) {
            self.wave_spawn.get_or_insert(view.tick);
        }
        self.view = Some(view.clone());
    }

    fn order(&mut self, tick: u32, unit: Option<EntityId>, order: Order) {
        assert!(self.orders.len() < OPENING_LIMIT.div_ceil(3) as usize);
        assert!((tick - 1).is_multiple_of(3));
        assert!(
            self.orders
                .last()
                .is_none_or(|(previous, ..)| tick >= previous + 3)
        );
        (tick, unit, order).hash(&mut self.order_hash);
        self.orders.push((tick, unit, order));
    }
}

fn lane_creep(kind: UnitKind) -> bool {
    matches!(kind, UnitKind::CreepMelee | UnitKind::CreepRanged)
}

fn builtin_opening(
    controllers: [OpeningController<'_>; 2],
    first_decision: u32,
) -> [OpeningTrace; 2] {
    assert!(matches!(first_decision, 1 | 901));
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: OPENING_SEED,
    })
    .expect("arena");
    let mut seats = [
        new_seat(0, &start.messages[0]).expect("Radiant"),
        new_seat(1, &start.messages[1]).expect("Dire"),
    ];
    let mut traces = [OpeningTrace::new(SlotId(0)), OpeningTrace::new(SlotId(1))];
    for (trace, messages) in traces.iter_mut().zip(&start.messages) {
        for message in messages {
            trace.observe(message);
        }
    }
    for _ in 1..OPENING_LIMIT {
        let tick = arena.tick();
        let mut requests = [None; 2];
        if tick >= first_decision && (tick - first_decision).is_multiple_of(3) {
            for index in 0..2 {
                requests[index] = match controllers[index] {
                    OpeningController::Passive => None,
                    OpeningController::Teacher => {
                        seat_request(&mut seats[index], None).expect("Teacher request")
                    }
                    OpeningController::Tactical(policy) => {
                        seat_request(&mut seats[index], Some(policy)).expect("Tactical request")
                    }
                };
                if let Some(request) = requests[index] {
                    traces[index].order(tick, request.unit, request.order);
                }
            }
        }
        let step = arena.step(&requests).expect("opening step");
        for index in 0..2 {
            assert_eq!(
                observe_messages(&mut seats[index], &step.messages[index]).expect("complete tick"),
                None
            );
            for message in &step.messages[index] {
                traces[index].observe(message);
            }
        }
    }
    traces
}

struct OpeningWire {
    link: Link,
    trace: OpeningTrace,
    completed: u32,
    acknowledged: u32,
}

impl Wire for OpeningWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        let message = self.link.hear()?;
        if let Some(message) = &message {
            if let ServerMsg::Snapshot { view } = message {
                assert_eq!(self.acknowledged + 1, view.tick);
            }
            self.trace.observe(message);
            if let ServerMsg::Events { tick, .. } = message {
                self.completed = *tick;
            }
        }
        Ok(message)
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        let tick = self
            .trace
            .view
            .as_ref()
            .expect("snapshot before order")
            .tick;
        assert_eq!(self.completed, tick, "orders must wait for explicit Events");
        assert_eq!(self.acknowledged + 1, tick, "orders must precede ACK");
        self.trace.order(tick, unit, order);
        self.link.order(unit, order)
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        assert_eq!(self.acknowledged + 1, tick);
        if tick != OPENING_LIMIT {
            assert_eq!(self.completed, tick);
        }
        self.acknowledged = tick;
        self.link.acknowledge(tick)
    }
}

fn tcp_opening(
    controllers: [OpeningController<'_>; 2],
) -> ([(OpeningTrace, Outcome); 2], [OpeningTrace; 2]) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind real match server");
    let address = listener.local_addr().expect("address").to_string();
    let timeout = Duration::from_secs(10);
    let root = directory("pregame");
    std::fs::create_dir(&root).expect("unique replay directory");
    let replay = root.join("opening.brp");
    let server_replay = replay.clone();
    let (finished, completion) = mpsc::sync_channel(1);
    let server = thread::spawn(move || {
        finished
            .send(bota_server::game_loop::run(
                listener,
                bota_server::game_loop::ServerOpts {
                    mode: TickMode::Lockstep,
                    tick_rate: 30,
                    players: 2,
                    replay: Some(server_replay),
                    seed: OPENING_SEED,
                    map: MapId(1),
                    ack_timeout_ticks: 600,
                },
            ))
            .expect("server completion receiver");
    });
    let (radiant, radiant_seat) =
        Link::join_with_timeout(&address, "opening-radiant", timeout).expect("Radiant");
    let (dire, dire_seat) =
        Link::join_with_timeout(&address, "opening-dire", timeout).expect("Dire");
    assert_eq!(radiant_seat.slot, SlotId(0));
    assert_eq!(dire_seat.slot, SlotId(1));
    let output = thread::scope(|scope| {
        let radiant = scope.spawn(move || play_opening(radiant, radiant_seat, controllers[0]));
        let dire = play_opening(dire, dire_seat, controllers[1]);
        [radiant.join().expect("Radiant thread"), dire]
    });
    completion
        .recv_timeout(timeout)
        .expect("bounded server shutdown")
        .expect("server result");
    server.join().expect("server thread");
    let authoritative = replay_opening(&replay);
    std::fs::remove_dir_all(root).expect("replay cleanup");
    (output, authoritative)
}

fn replay_opening(path: &Path) -> [OpeningTrace; 2] {
    use bota_proto::{FrameReader, ReplayRecord};
    use std::io::Read;

    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .expect("server replay")
        .take(16_777_217)
        .read_to_end(&mut bytes)
        .expect("bounded replay read");
    assert!(bytes.len() <= 16_777_216);
    assert!(!bytes.is_empty());
    let mut reader = FrameReader::new();
    reader.push(&bytes);
    let mut traces = [OpeningTrace::new(SlotId(0)), OpeningTrace::new(SlotId(1))];
    for _ in 0..OPENING_LIMIT * 4 + 64 {
        let Some(record) = reader.next_message::<ReplayRecord>().expect("replay frame") else {
            break;
        };
        match record {
            ReplayRecord::Msg(message) => {
                if matches!(&message, ServerMsg::Snapshot { view } if view.tick > OPENING_LIMIT) {
                    break;
                }
                for trace in &mut traces {
                    trace.observe(&message);
                }
            }
            ReplayRecord::Orders { tick, orders } => {
                for order in orders {
                    traces[usize::from(order.slot.0)].order(tick - 1, order.unit, order.order);
                }
            }
        }
    }
    for trace in &traces {
        assert_eq!(trace.view.as_ref().expect("full view").tick, OPENING_LIMIT);
    }
    assert_eq!(traces[0].creep_meet, traces[1].creep_meet);
    traces
}

fn play_opening(
    link: Link,
    seated: Seated,
    controller: OpeningController<'_>,
) -> (OpeningTrace, Outcome) {
    let mut wire = OpeningWire {
        link,
        trace: OpeningTrace::new(seated.slot),
        completed: 0,
        acknowledged: 0,
    };
    let outcome = match controller {
        OpeningController::Passive => play_passive_opening(&mut wire, seated),
        OpeningController::Tactical(policy) => {
            crate::play_tactical_on(&mut wire, seated, Some(OPENING_LIMIT), policy)
        }
        OpeningController::Teacher => {
            crate::play_teacher_on(&mut wire, seated, Some(OPENING_LIMIT))
        }
    }
    .expect("real opening deployment");
    assert_eq!(wire.acknowledged, OPENING_LIMIT);
    (wire.trace, outcome)
}

fn play_passive_opening(wire: &mut OpeningWire, seated: Seated) -> std::io::Result<Outcome> {
    assert_eq!(seated.mode, TickMode::Lockstep);
    assert!(seated.slot.0 < 2);
    let mut outcome = Outcome {
        slot: Some(seated.slot),
        ..Outcome::default()
    };
    for _ in 0..OPENING_LIMIT * 4 + 64 {
        let message = wire
            .hear()?
            .ok_or_else(|| std::io::Error::other("passive opening disconnected"))?;
        match message {
            ServerMsg::MatchStart { info } => {
                outcome.team = Some(info.picks[usize::from(seated.slot.0)].team);
            }
            ServerMsg::Snapshot { view } => {
                outcome.ticks = view.tick;
                if view.tick == OPENING_LIMIT {
                    wire.acknowledge(view.tick)?;
                    return Ok(outcome);
                }
            }
            ServerMsg::Events { tick, .. } => wire.acknowledge(tick)?,
            _ => {}
        }
    }
    Err(std::io::Error::other(
        "passive opening message limit exceeded",
    ))
}
