use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use bota_proto::{MapId, ServerMsg, TickMode};
use bota_server::game_loop::{ServerOpts, run};

use crate::{Arena, ArenaConfig, Link, Wire};

#[test]
fn builtin_first_two_snapshots_match_tcp_server() {
    let seed = 79;
    let map = MapId(1);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let address = listener.local_addr().expect("test server address");
    let (finished_tx, finished_rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _unused = finished_tx.send(run(listener, server_options(seed, map)));
    });

    let timeout = Duration::from_secs(5);
    let (mut radiant, radiant_seat) =
        Link::join_with_timeout(&address.to_string(), "radiant", timeout).expect("seat 0");
    let (mut dire, dire_seat) =
        Link::join_with_timeout(&address.to_string(), "dire", timeout).expect("seat 1");
    assert_eq!(radiant_seat.slot.0, 0);
    assert_eq!(dire_seat.slot.0, 1);

    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map,
        seed,
    })
    .expect("local arena");
    let local_second = arena.step(&[None, None]).expect("local second tick");

    assert_eq!(next_match_start(&mut radiant), start.messages[0][0]);
    assert_eq!(next_match_start(&mut dire), start.messages[1][0]);
    assert_eq!(next_snapshot(&mut radiant), start.messages[0][1]);
    assert_eq!(next_snapshot(&mut dire), start.messages[1][1]);

    radiant.acknowledge(1).expect("ack radiant tick one");
    dire.acknowledge(1).expect("ack dire tick one");
    assert_eq!(next_snapshot(&mut radiant), local_second.messages[0][0]);
    assert_eq!(next_snapshot(&mut dire), local_second.messages[1][0]);

    drop(radiant);
    drop(dire);
    finished_rx
        .recv_timeout(timeout)
        .expect("server exits within the test timeout")
        .expect("server match exits cleanly");
}

fn server_options(seed: u64, map: MapId) -> ServerOpts {
    ServerOpts {
        mode: TickMode::Lockstep,
        tick_rate: 30,
        players: 2,
        replay: None,
        seed,
        map,
        ack_timeout_ticks: 300,
    }
}

fn next_match_start(link: &mut Link) -> ServerMsg {
    next_matching(link, |message| {
        matches!(message, ServerMsg::MatchStart { .. })
    })
}

fn next_snapshot(link: &mut Link) -> ServerMsg {
    next_matching(link, |message| {
        matches!(message, ServerMsg::Snapshot { .. })
    })
}

fn next_matching(link: &mut Link, wanted: impl Fn(&ServerMsg) -> bool) -> ServerMsg {
    for _ in 0..64 {
        let message = link
            .hear()
            .expect("read server message")
            .expect("server remains connected");
        if wanted(&message) {
            return message;
        }
    }
    panic!("server sent 64 messages without the expected match message");
}
