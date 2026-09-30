use std::cell::Cell;
use std::collections::VecDeque;
use std::io::Cursor;
use std::time::Duration;

use bota_proto::{MapId, MatchStats, PlayerId};

use super::*;
use crate::telemetry::{Clock, LiveConfig, PerformanceOutput};
use crate::tests::support::RecordingWire;

#[derive(Default)]
struct TestClock(Cell<Duration>);

impl Clock for TestClock {
    fn now(&self) -> Duration {
        let now = self.0.get();
        self.0.set(now + Duration::from_nanos(100));
        now
    }
}

fn fixture(mode: TickMode) -> RecordingWire {
    let (_, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_101,
    })
    .unwrap();
    let ServerMsg::MatchStart { mut info } = start.messages[0][0].clone() else {
        panic!("MatchStart first");
    };
    info.mode = mode;
    let mut messages = VecDeque::from([ServerMsg::MatchStart { info }]);
    let view = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .unwrap();
    for tick in 1..=4 {
        let mut view = view.clone();
        view.tick = tick;
        messages.push_back(ServerMsg::Snapshot { view });
        messages.push_back(ServerMsg::Events {
            tick,
            events: Vec::new(),
        });
    }
    messages.push_back(ServerMsg::MatchOver {
        winner: Team::Radiant,
        stats: MatchStats {
            duration: 4,
            slots: Vec::new(),
        },
    });
    RecordingWire {
        messages,
        orders: Vec::new(),
        acknowledgements: Vec::new(),
    }
}

#[test]
fn seat_reports_completed_work_and_preserves_terminal_outcomes_with_debug_enabled_or_disabled() {
    let model = PolicyModel::fresh(70_102).unwrap();
    for mode in [TickMode::Lockstep, TickMode::Realtime] {
        for controller in [
            LiveController::Script(crate::ScriptKind::Teacher),
            LiveController::Neural(&model),
        ] {
            for (limit, disconnected, updates, reason) in [
                (None, false, 4, "match_over"),
                (Some(4), false, 3, "limit"),
                (None, true, 0, "error"),
            ] {
                for debug in [0, 1] {
                    let clock = TestClock::default();
                    let mut wire = fixture(mode);
                    if disconnected {
                        wire.messages.truncate(2);
                    }
                    let seated = Seated {
                        player: PlayerId(1),
                        slot: SlotId(0),
                        tick_rate: 30,
                        mode,
                    };
                    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 16_384]));
                    let result = play_controller_with_telemetry(
                        &mut wire,
                        seated,
                        limit,
                        controller,
                        LiveConfig::new(2, debug).unwrap(),
                        &clock,
                        &mut output,
                    );
                    if disconnected {
                        assert_eq!(
                            result.unwrap_err().to_string(),
                            "server closed the connection before MatchOver"
                        );
                        assert!(wire.orders.is_empty());
                    } else {
                        let outcome = result.unwrap();
                        assert_eq!(outcome.ticks, 4);
                        assert_eq!(outcome.winner, limit.is_none().then_some(Team::Radiant));
                        assert_eq!(outcome.orders as usize, wire.orders.len());
                    }
                    let acknowledgements = if mode == TickMode::Lockstep && !disconnected {
                        vec![1, 2, 3, 4]
                    } else {
                        Vec::new()
                    };
                    assert_eq!(wire.acknowledgements, acknowledgements);
                    assert_report(output, updates, reason, debug);
                }
            }
        }
    }
}

fn assert_report(
    output: PerformanceOutput<Cursor<[u8; 16_384]>>,
    updates: usize,
    reason: &str,
    debug: u32,
) {
    assert!(!output.failed());
    let output = output.into_inner();
    let text = std::str::from_utf8(&output.get_ref()[..output.position() as usize]).unwrap();
    assert_eq!(
        text.matches("event=live_performance scope=total").count(),
        1
    );
    assert_eq!(
        text.matches("event=live_performance scope=window").count(),
        updates / 2
    );
    assert!(text.contains(&format!("reason={reason} updates={updates}")));
    assert!(text.contains(&format!("pending_update={}", reason != "match_over")));
    assert_eq!(
        text.contains("level=DEBUG event=live_decision"),
        debug != 0 && reason != "error"
    );
    assert!(!text.contains("level=WARN"));
    if reason == "error" {
        assert!(text.contains(
            "compute_p50_upper_ns=unknown compute_p95_upper_ns=unknown compute_max_ns=0"
        ));
    }
}
