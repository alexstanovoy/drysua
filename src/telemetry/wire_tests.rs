use std::cell::Cell;
use std::time::Duration;

use bota_proto::{EntityId, Order, ServerMsg};

use super::*;
use crate::Wire;

#[derive(Default)]
struct FakeClock(Cell<Duration>);

impl Clock for FakeClock {
    fn now(&self) -> Duration {
        self.0.get()
    }
}

struct TimedFixture<'a> {
    clock: &'a FakeClock,
    socket_wait: Option<Duration>,
    fail: bool,
    calls: [u32; 3],
}

impl TimedFixture<'_> {
    fn advance(&self, micros: u64) {
        self.clock
            .0
            .set(self.clock.now() + Duration::from_micros(micros));
    }
}

impl Wire for TimedFixture<'_> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        self.calls[0] += 1;
        self.advance(10);
        Ok(Some(ServerMsg::Events {
            tick: 7,
            events: Vec::new(),
        }))
    }

    fn order(&mut self, _: Option<EntityId>, _: Order) -> std::io::Result<u32> {
        self.calls[1] += 1;
        self.advance(3);
        Ok(41)
    }

    fn acknowledge(&mut self, _: u32) -> std::io::Result<()> {
        self.calls[2] += 1;
        self.advance(2);
        if self.fail {
            Err(std::io::Error::other("fixture ACK failure"))
        } else {
            Ok(())
        }
    }

    fn take_receive_wait(&mut self) -> Option<Duration> {
        self.socket_wait
    }
}

#[test]
fn measured_wire_splits_socket_wait_from_decode_and_preserves_results_and_call_counts() {
    let clock = FakeClock::default();
    let mut fixture = TimedFixture {
        clock: &clock,
        socket_wait: Some(Duration::from_micros(7)),
        fail: false,
        calls: [0; 3],
    };
    let mut wire = MeasuredWire::new(&mut fixture, &clock);
    assert!(matches!(
        wire.hear().unwrap(),
        Some(ServerMsg::Events { tick: 7, .. })
    ));
    assert_eq!(
        wire.order(
            None,
            Order::Move {
                target: bota_proto::Target::None
            }
        )
        .unwrap(),
        41
    );
    wire.acknowledge(7).unwrap();
    let timing = wire.take_timing();
    assert_eq!(timing.receive_wait, Duration::from_micros(7));
    assert_eq!(timing.compute, Duration::from_micros(3));
    assert_eq!(timing.order_send, Some(Duration::from_micros(3)));
    assert_eq!(timing.ack_send, Some(Duration::from_micros(2)));
    assert_eq!(wire.receive_scope(), "socket_read");
    assert_eq!(wire.take_timing().receive_wait, Duration::ZERO);
    assert_eq!(fixture.calls, [1, 1, 1]);
}

#[test]
fn generic_wire_wait_is_honestly_labeled_as_whole_hear_call() {
    let clock = FakeClock::default();
    let mut fixture = TimedFixture {
        clock: &clock,
        socket_wait: None,
        fail: false,
        calls: [0; 3],
    };
    let mut wire = MeasuredWire::new(&mut fixture, &clock);
    wire.hear().unwrap();
    let timing = wire.take_timing();
    assert_eq!(timing.receive_wait, Duration::from_micros(10));
    assert_eq!(timing.compute, Duration::ZERO);
    assert_eq!(wire.receive_scope(), "wire_hear_including_decode");
}

#[test]
fn measured_ack_failure_is_returned_unchanged_and_timed_without_retry() {
    let clock = FakeClock::default();
    let mut fixture = TimedFixture {
        clock: &clock,
        socket_wait: None,
        fail: true,
        calls: [0; 3],
    };
    let mut wire = MeasuredWire::new(&mut fixture, &clock);
    let error = wire.acknowledge(7).unwrap_err();
    assert_eq!(error.to_string(), "fixture ACK failure");
    assert_eq!(wire.take_timing().ack_send, Some(Duration::from_micros(2)));
    assert_eq!(fixture.calls, [0, 0, 1]);
}
