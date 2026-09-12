use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use bota_proto::{
    ClientMsg, DamageKind, EntityId, EventKind, Order, PlayerId, Role, ServerMsg, SlotId, Team,
    TickMode, Vec2, decode_payload, encode_frame_to_vec,
};

use crate::{
    Arena, ArenaConfig, Link, MAP2_ID, MAP2_TICK_CAP, MAP2_TICK_RATE, Outcome, PolicyModel,
    Request, SHADOW_FIEND, Wire, play_neural_on, play_teacher_on,
};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
const ACCEPT_ATTEMPTS: u32 = 100;
const MAX_CLIENT_MESSAGES: usize = 6;
const MAX_CLIENT_PAYLOAD: usize = 256;
const MAX_SERVER_FRAME: usize = 1024 * 1024;
const MAX_NATIVE_MESSAGES: usize = 6;
const NAME: &str = "native-cap-contract";

#[derive(Clone, Copy)]
enum Controller<'model> {
    Teacher,
    Neural(&'model PolicyModel),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    Start,
    Snapshot(u32),
    Events(u32),
    Ack(u32),
    Over(u32, Team),
}

struct TracedLink {
    link: Link,
    messages: Vec<ServerMsg>,
    boundaries: Vec<Boundary>,
}

struct Peer {
    stream: TcpStream,
    deadline: Instant,
    received: Vec<ClientMsg>,
    written_frames: usize,
}

struct Session {
    outcome: Outcome,
    observed: Vec<ServerMsg>,
    boundaries: Vec<Boundary>,
    unread_at_return: Vec<ServerMsg>,
    native: Vec<ServerMsg>,
    received: Vec<ClientMsg>,
}

#[test]
fn teacher_receives_native_cap_draw_after_complete_events_with_safety_limit_on_both_sides() {
    for side in 0..2 {
        let session = run_session(side, Controller::Teacher, MAP2_TICK_CAP + 1);
        assert_cap_session(session, side, true);
    }
}

#[test]
fn neural_receives_native_cap_draw_after_complete_events_with_safety_limit_on_both_sides() {
    let model = PolicyModel::fresh(10_092_900).expect("synthetic fresh CPU model, no artifact");
    for side in 0..2 {
        let session = run_session(side, Controller::Neural(&model), MAP2_TICK_CAP + 1);
        assert_cap_session(session, side, true);
    }
}

#[test]
fn teacher_exact_cap_limit_returns_before_events_and_match_over_on_both_sides() {
    for side in 0..2 {
        let session = run_session(side, Controller::Teacher, MAP2_TICK_CAP);
        assert_cap_session(session, side, false);
    }
}

#[test]
fn neural_exact_cap_limit_returns_before_events_and_match_over_on_both_sides() {
    let model = PolicyModel::fresh(10_092_900).expect("synthetic fresh CPU model, no artifact");
    for side in 0..2 {
        let session = run_session(side, Controller::Neural(&model), MAP2_TICK_CAP);
        assert_cap_session(session, side, false);
    }
}

fn run_session(side: usize, controller: Controller<'_>, limit: u32) -> Session {
    assert!(side < 2);
    assert!(matches!(limit, MAP2_TICK_CAP) || limit == MAP2_TICK_CAP + 1);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind bounded peer");
    listener.set_nonblocking(true).expect("bounded accept");
    let address = listener.local_addr().expect("peer address").to_string();
    let (cancel, cancelled) = mpsc::sync_channel(0);
    let (finished, completion) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let result = serve(listener, cancelled, side);
        let _ = finished.send(result);
    });
    let (link, seated) =
        Link::join_with_timeout(&address, NAME, IO_TIMEOUT).expect("real bot join");
    assert_eq!(seated.slot, SlotId(side as u8));
    let mut wire = TracedLink {
        link,
        messages: Vec::with_capacity(MAX_NATIVE_MESSAGES),
        boundaries: Vec::with_capacity(8),
    };
    let result = match controller {
        Controller::Teacher => play_teacher_on(&mut wire, seated, Some(limit)),
        Controller::Neural(model) => play_neural_on(&mut wire, seated, Some(limit), model),
    };
    // Only the intentional early-exit case drains outside the controller to avoid a teardown RST.
    let remainder = if limit == MAP2_TICK_CAP {
        drain_after_return(&mut wire.link)
    } else {
        Ok(Vec::new())
    };
    let TracedLink {
        link,
        messages,
        boundaries,
    } = wire;
    drop(link);
    drop(cancel);
    let completed = completion
        .recv_timeout(IO_TIMEOUT * 2)
        .expect("bounded peer completion");
    worker.join().expect("peer worker did not panic");
    let (native, received) = completed.expect("native TCP peer");
    Session {
        outcome: result.expect("production play loop"),
        observed: messages,
        boundaries,
        unread_at_return: remainder.expect("drain after controller return"),
        native,
        received,
    }
}

fn assert_cap_session(session: Session, side: usize, terminal: bool) {
    let team = if side == 0 { Team::Radiant } else { Team::Dire };
    assert_eq!(session.outcome.slot, Some(SlotId(side as u8)));
    assert_eq!(session.outcome.team, Some(team));
    assert_eq!(session.outcome.ticks, MAP2_TICK_CAP);
    assert_eq!(session.outcome.winner, terminal.then_some(Team::Neutral));
    assert_eq!(session.outcome.rejections, 0);
    assert_eq!(session.outcome.last_rejection, None);
    assert_eq!(session.outcome.decisions, 1);
    assert!(session.outcome.orders <= 1);
    assert_eq!(session.received.len(), 5 + session.outcome.orders as usize);
    let expected = if terminal { MAX_NATIVE_MESSAGES } else { 4 };
    assert_eq!(session.observed, session.native[..expected]);
    assert_eq!(session.unread_at_return, session.native[expected..]);
    let mut boundaries = vec![
        Boundary::Start,
        Boundary::Snapshot(MAP2_TICK_CAP - 1),
        Boundary::Events(MAP2_TICK_CAP - 1),
        Boundary::Ack(MAP2_TICK_CAP - 1),
        Boundary::Snapshot(MAP2_TICK_CAP),
    ];
    if terminal {
        boundaries.push(Boundary::Events(MAP2_TICK_CAP));
    }
    boundaries.push(Boundary::Ack(MAP2_TICK_CAP));
    if terminal {
        boundaries.push(Boundary::Over(MAP2_TICK_CAP, Team::Neutral));
    }
    assert_eq!(session.boundaries, boundaries);
    assert_eq!(
        session
            .received
            .iter()
            .filter_map(|message| match message {
                ClientMsg::Ack { tick } => Some(*tick),
                _ => None,
            })
            .collect::<Vec<_>>(),
        [MAP2_TICK_CAP - 1, MAP2_TICK_CAP]
    );
}

fn serve(
    listener: TcpListener,
    cancelled: Receiver<()>,
    side: usize,
) -> io::Result<(Vec<ServerMsg>, Vec<ClientMsg>)> {
    let stream = accept_peer(&listener, &cancelled)?;
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    let mut peer = Peer {
        stream,
        deadline: Instant::now() + IO_TIMEOUT,
        received: Vec::with_capacity(MAX_CLIENT_MESSAGES),
        written_frames: 0,
    };
    peer.handshake(side)?;
    let (mut arena, mut native, witness) = primed_arena(side);
    peer.send_batch(&native)?;
    let request = peer.acknowledged(MAP2_TICK_CAP - 1, true)?;
    let mut requests = [None, None];
    requests[side] = request;
    let final_tick = arena.step(&requests).expect("one real tick after priming");
    assert_eq!(arena.tick(), MAP2_TICK_CAP);
    assert_final_tick(&final_tick.messages[side], witness);
    peer.send_batch(&final_tick.messages[side])?;
    native.extend(final_tick.messages[side].iter().cloned());
    assert_eq!(native.len(), MAX_NATIVE_MESSAGES);
    assert!(peer.acknowledged(MAP2_TICK_CAP, false)?.is_none());
    assert_eq!(peer.written_frames, MAX_NATIVE_MESSAGES + 1);
    peer.stream.shutdown(Shutdown::Write)?;
    peer.set_read_deadline()?;
    let mut extra = [0; 1];
    assert_eq!(
        peer.stream.read(&mut extra)?,
        0,
        "no extra orders/ACKs after the terminal tick"
    );
    Ok((native, peer.received))
}

fn accept_peer(listener: &TcpListener, cancelled: &Receiver<()>) -> io::Result<TcpStream> {
    for _ in 0..ACCEPT_ATTEMPTS {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                match cancelled.recv_timeout(IO_TIMEOUT / ACCEPT_ATTEMPTS) {
                    Err(RecvTimeoutError::Timeout) => {}
                    _ => return Err(io::Error::other("client setup cancelled before accept")),
                }
            }
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "native-cap peer accept deadline",
    ))
}

impl Peer {
    fn handshake(&mut self, side: usize) -> io::Result<()> {
        assert_eq!(
            self.receive()?,
            ClientMsg::Hello {
                role: Role::Bot,
                name: NAME.to_owned()
            }
        );
        self.send_batch(&[ServerMsg::Welcome {
            player_id: PlayerId(71),
            slot: Some(SlotId(side as u8)),
            tick_rate: MAP2_TICK_RATE as u16,
            mode: TickMode::Lockstep,
        }])?;
        assert_eq!(self.receive()?, ClientMsg::PickHero { hero: SHADOW_FIEND });
        assert_eq!(self.receive()?, ClientMsg::SetReady(true));
        Ok(())
    }

    fn acknowledged(&mut self, tick: u32, allow_order: bool) -> io::Result<Option<Request>> {
        assert!((MAP2_TICK_CAP - 1..=MAP2_TICK_CAP).contains(&tick));
        let mut request = None;
        for _ in 0..=usize::from(allow_order) {
            match self.receive()? {
                ClientMsg::Order { seq, unit, order } if allow_order && request.is_none() => {
                    assert_eq!(seq, 1);
                    request = Some(Request { seq, unit, order });
                }
                ClientMsg::Ack { tick: acknowledged } => {
                    assert_eq!(acknowledged, tick);
                    return Ok(request);
                }
                message => {
                    return Err(io::Error::other(format!(
                        "unexpected client message before Ack({tick}): {message:?}"
                    )));
                }
            }
        }
        Err(io::Error::other(format!("missing bounded Ack({tick})")))
    }

    fn receive(&mut self) -> io::Result<ClientMsg> {
        assert!(self.received.len() < MAX_CLIENT_MESSAGES);
        let mut prefix = [0; 4];
        self.set_read_deadline()?;
        self.stream.read_exact(&mut prefix)?;
        let length = u32::from_le_bytes(prefix) as usize;
        if length == 0 || length > MAX_CLIENT_PAYLOAD {
            return Err(io::Error::other(format!(
                "client payload outside 1..={MAX_CLIENT_PAYLOAD}: {length}"
            )));
        }
        let mut payload = [0; MAX_CLIENT_PAYLOAD];
        self.set_read_deadline()?;
        self.stream.read_exact(&mut payload[..length])?;
        let message = decode_payload::<ClientMsg>(&payload[..length]).map_err(io::Error::other)?;
        self.received.push(message.clone());
        assert!(self.received.len() <= MAX_CLIENT_MESSAGES);
        Ok(message)
    }

    fn send_batch(&mut self, messages: &[ServerMsg]) -> io::Result<()> {
        assert!((1..=3).contains(&messages.len()));
        for message in messages {
            assert!(self.written_frames < MAX_NATIVE_MESSAGES + 1);
            let bytes = encode_frame_to_vec(message).map_err(io::Error::other)?;
            assert!(bytes.len() > 4);
            assert!(bytes.len() <= MAX_SERVER_FRAME);
            // Split the length prefix too; TCP is still free to coalesce these writes.
            for fragment in [&bytes[..2], &bytes[2..4], &bytes[4..]] {
                self.stream.set_write_timeout(Some(self.remaining()?))?;
                self.stream.write_all(fragment)?;
            }
            self.written_frames += 1;
        }
        Ok(())
    }

    fn remaining(&self) -> io::Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "native-cap peer I/O deadline"))
    }

    fn set_read_deadline(&self) -> io::Result<()> {
        self.stream.set_read_timeout(Some(self.remaining()?))
    }
}

impl Wire for TracedLink {
    fn hear(&mut self) -> io::Result<Option<ServerMsg>> {
        let message = self.link.hear()?;
        if let Some(message) = &message {
            assert!(self.messages.len() < MAX_NATIVE_MESSAGES);
            let boundary = match message {
                ServerMsg::MatchStart { info } => {
                    assert_eq!(info.map, MAP2_ID);
                    Boundary::Start
                }
                ServerMsg::Snapshot { view } => Boundary::Snapshot(view.tick),
                ServerMsg::Events { tick, .. } => Boundary::Events(*tick),
                ServerMsg::MatchOver { winner, stats } => Boundary::Over(stats.duration, *winner),
                other => {
                    return Err(io::Error::other(format!(
                        "unexpected native-cap server message: {other:?}"
                    )));
                }
            };
            assert!(self.boundaries.len() < 8);
            self.boundaries.push(boundary);
            self.messages.push(message.clone());
        }
        Ok(message)
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> io::Result<u32> {
        assert_eq!(
            self.boundaries.last(),
            Some(&Boundary::Events(MAP2_TICK_CAP - 1))
        );
        self.link.order(unit, order)
    }

    fn acknowledge(&mut self, tick: u32) -> io::Result<()> {
        assert!(self.boundaries.len() < 8);
        self.link.acknowledge(tick)?;
        self.boundaries.push(Boundary::Ack(tick));
        Ok(())
    }

    fn take_receive_wait(&mut self) -> Option<Duration> {
        self.link.take_receive_wait()
    }
}

fn drain_after_return(link: &mut Link) -> io::Result<Vec<ServerMsg>> {
    let mut messages = Vec::with_capacity(2);
    for _ in 0..=2 {
        let Some(message) = link.hear()? else {
            return Ok(messages);
        };
        messages.push(message);
    }
    Err(io::Error::other(
        "more than two native messages remained after controller return",
    ))
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
        world.statuses.remove(unit);
        world.transform.get_mut(unit).expect("position").pos = Vec2::from_ints(9_216, 9_216);
        hero = Some(bota_server::game::wire_id(unit));
        world.push_hit(None, unit, 1, DamageKind::Pure);
    });
    let mut initial = start.messages.swap_remove(side);
    initial.truncate(1);
    initial.extend(projected.messages[side].iter().cloned());
    assert_eq!(arena.tick(), MAP2_TICK_CAP - 1);
    let witness = EventKind::Damaged {
        source: None,
        target: hero.expect("witness hero"),
        amount: 1,
        kind: DamageKind::Pure,
        crit: false,
    };
    (arena, initial, witness)
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
        "nonempty final damage must precede the draw"
    );
}
