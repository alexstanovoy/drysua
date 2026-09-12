use std::cell::Cell;
use std::collections::VecDeque;
use std::io::Cursor;
use std::time::Duration;

use bota_proto::{ClientMsg, EntityId, MapId, MatchStats, Order, PlayerId};

use super::*;
use crate::telemetry::{Clock, LiveConfig, PerformanceOutput};

#[derive(Default)]
struct TestClock(Cell<Duration>);

impl Clock for TestClock {
    fn now(&self) -> Duration {
        let now = self.0.get();
        self.0.set(now + Duration::from_nanos(100));
        now
    }
}

struct FixtureWire<'a> {
    messages: VecDeque<ServerMsg>,
    sent: Vec<ClientMsg>,
    clock: &'a TestClock,
    last_events: u32,
    early_limit: Option<u32>,
}

impl Wire for FixtureWire<'_> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        self.clock
            .0
            .set(self.clock.0.get() + Duration::from_millis(50));
        let message = self.messages.pop_front();
        if let Some(ServerMsg::Events { tick, .. }) = message.as_ref() {
            self.last_events = *tick;
        }
        Ok(message)
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        assert!(
            !matches!(self.sent.last(), Some(ClientMsg::Ack { tick }) if *tick == self.last_events)
        );
        assert!(self.sent.len() < 32);
        let seq = self
            .sent
            .iter()
            .filter(|message| matches!(message, ClientMsg::Order { .. }))
            .count() as u32
            + 1;
        self.sent.push(ClientMsg::Order { seq, unit, order });
        self.clock
            .0
            .set(self.clock.0.get() + Duration::from_micros(3));
        Ok(seq)
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        assert!(
            tick == self.last_events || Some(tick) == self.early_limit,
            "ACK must follow Events"
        );
        assert!(self.sent.len() < 32);
        self.sent.push(ClientMsg::Ack { tick });
        self.clock
            .0
            .set(self.clock.0.get() + Duration::from_micros(5));
        Ok(())
    }

    fn take_receive_wait(&mut self) -> Option<Duration> {
        Some(Duration::from_millis(50))
    }
}

fn fixture(clock: &TestClock, mode: TickMode) -> FixtureWire<'_> {
    let (_, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_101,
    })
    .unwrap();
    let mut messages = VecDeque::new();
    let ServerMsg::MatchStart { mut info } = start.messages[0][0].clone() else {
        panic!("MatchStart first");
    };
    info.mode = mode;
    messages.push_back(ServerMsg::MatchStart { info });
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
    FixtureWire {
        messages,
        sent: Vec::new(),
        clock,
        last_events: 0,
        early_limit: None,
    }
}

fn terms(mode: TickMode) -> Seated {
    Seated {
        player: PlayerId(1),
        slot: SlotId(0),
        tick_rate: 30,
        mode,
    }
}

fn baseline(
    wire: &mut FixtureWire<'_>,
    mode: TickMode,
    controller: LiveController<'_>,
    clock: &impl Clock,
) -> Outcome {
    let mut outcome = Outcome {
        slot: Some(SlotId(0)),
        ..Outcome::default()
    };
    let mut policy = None;
    for _ in 0..16 {
        let message = wire.hear().unwrap().unwrap();
        if handle_policy_message(
            wire,
            terms(mode),
            None,
            controller,
            &mut outcome,
            &mut policy,
            message,
            clock,
        )
        .unwrap()
        {
            return outcome;
        }
    }
    panic!("bounded fixture must complete");
}

#[test]
fn live_telemetry_preserves_teacher_and_pure_neural_orders_outcomes_and_ack_order() {
    let model = PolicyModel::fresh(70_102).unwrap();
    for mode in [TickMode::Lockstep, TickMode::Realtime] {
        for controller in [LiveController::Teacher, LiveController::Neural(&model)] {
            let baseline_clock = TestClock::default();
            let mut plain = fixture(&baseline_clock, mode);
            let expected = baseline(&mut plain, mode, controller, &baseline_clock);
            for debug_every in [0, 1] {
                let clock = TestClock::default();
                let mut measured = fixture(&clock, mode);
                let mut output = PerformanceOutput::new(Cursor::new([0_u8; 16_384]));
                let actual = play_controller_with_telemetry(
                    &mut measured,
                    terms(mode),
                    None,
                    controller,
                    LiveConfig::new(2, debug_every).unwrap(),
                    &clock,
                    &mut output,
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(measured.sent, plain.sent);
                assert!(!output.failed());
            }
        }
    }
}

#[test]
fn live_telemetry_logs_completed_windows_and_final_totals_without_wait_budget_warnings() {
    let clock = TestClock::default();
    let mut wire = fixture(&clock, TickMode::Lockstep);
    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 16_384]));
    play_controller_with_telemetry(
        &mut wire,
        terms(TickMode::Lockstep),
        None,
        LiveController::Teacher,
        LiveConfig::new(2, 1).unwrap(),
        &clock,
        &mut output,
    )
    .unwrap();
    let output = output.into_inner();
    let text = std::str::from_utf8(&output.get_ref()[..output.position() as usize]).unwrap();
    assert_eq!(
        text.matches("event=live_performance scope=window").count(),
        2
    );
    assert_eq!(
        text.matches("event=live_performance scope=total").count(),
        1
    );
    assert!(text.contains("reason=match_over updates=4 progress_ticks=3"));
    assert!(text.contains("receive_wait_scope=socket_read"));
    assert!(text.contains("level=DEBUG event=live_decision"));
    assert!(!text.contains("level=WARN"));
}

#[test]
fn legacy_limit_snapshot_is_not_counted_as_a_completed_snapshot_events_update() {
    let clock = TestClock::default();
    let mut wire = fixture(&clock, TickMode::Lockstep);
    wire.early_limit = Some(4);
    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 8192]));
    let outcome = play_controller_with_telemetry(
        &mut wire,
        terms(TickMode::Lockstep),
        Some(4),
        LiveController::Teacher,
        LiveConfig::default(),
        &clock,
        &mut output,
    )
    .unwrap();
    let output = output.into_inner();
    let text = std::str::from_utf8(&output.get_ref()[..output.position() as usize]).unwrap();
    assert_eq!(outcome.ticks, 4);
    assert!(text.contains("reason=limit updates=3 progress_ticks=2"));
    assert!(text.contains("pending_update=true"));
    assert!(matches!(wire.sent.last(), Some(ClientMsg::Ack { tick: 4 })));
}

#[test]
fn missing_events_error_remains_exact_and_final_report_does_not_claim_completion() {
    let clock = TestClock::default();
    let mut wire = fixture(&clock, TickMode::Lockstep);
    wire.messages.truncate(2);
    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 8192]));
    let error = play_controller_with_telemetry(
        &mut wire,
        terms(TickMode::Lockstep),
        None,
        LiveController::Teacher,
        LiveConfig::default(),
        &clock,
        &mut output,
    )
    .unwrap_err();
    let output = output.into_inner();
    let text = std::str::from_utf8(&output.get_ref()[..output.position() as usize]).unwrap();
    assert_eq!(
        error.to_string(),
        "server closed the connection before MatchOver"
    );
    assert!(wire.sent.is_empty());
    assert!(text.contains("reason=error updates=0"));
    assert!(text.contains("pending_update=true"));
}
