use std::cell::Cell;
use std::time::Duration;

use bota_proto::{EntityId, Order, ServerMsg};

use super::*;
use crate::Wire;

#[derive(Default)]
struct FakeClock(Cell<u64>);

impl Clock for FakeClock {
    fn now(&self) -> Duration {
        Duration::from_micros(self.0.get())
    }
}

struct TimedFixture<'a> {
    clock: &'a FakeClock,
    socket_wait: Option<Duration>,
    fail: bool,
    calls: [u32; 3],
}

impl Wire for TimedFixture<'_> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        self.calls[0] += 1;
        self.clock.0.set(self.clock.0.get() + 10);
        Ok(Some(ServerMsg::Events {
            tick: 7,
            events: Vec::new(),
        }))
    }

    fn order(&mut self, _: Option<EntityId>, _: Order) -> std::io::Result<u32> {
        self.calls[1] += 1;
        self.clock.0.set(self.clock.0.get() + 3);
        Ok(41)
    }

    fn acknowledge(&mut self, _: u32) -> std::io::Result<()> {
        self.calls[2] += 1;
        self.clock.0.set(self.clock.0.get() + 2);
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
fn measured_wire_preserves_results_errors_and_accounts_each_call_once() {
    for socket_wait in [None, Some(Duration::from_micros(7))] {
        for fail in [false, true] {
            let clock = FakeClock::default();
            let mut fixture = TimedFixture {
                clock: &clock,
                socket_wait,
                fail,
                calls: [0; 3],
            };
            let mut wire = MeasuredWire::new(&mut fixture, &clock);
            assert!(matches!(
                wire.hear().unwrap(),
                Some(ServerMsg::Events { tick: 7, .. })
            ));
            let order = Order::Move {
                target: bota_proto::Target::None,
            };
            assert_eq!(wire.order(None, order).unwrap(), 41);
            let result = wire.acknowledge(7).map_err(|error| error.to_string());
            assert_eq!(
                result.err().as_deref(),
                fail.then_some("fixture ACK failure")
            );
            let timing = wire.take_timing();
            let wait = socket_wait.unwrap_or(Duration::from_micros(10));
            assert_eq!(timing.receive_wait, wait);
            assert_eq!(timing.compute, Duration::from_micros(10) - wait);
            assert_eq!(timing.order_send, Some(Duration::from_micros(3)));
            assert_eq!(timing.ack_send, Some(Duration::from_micros(2)));
            assert_eq!(
                wire.receive_scope(),
                if socket_wait.is_some() {
                    "socket_read"
                } else {
                    "wire_hear_including_decode"
                }
            );
            let cleared = wire.take_timing();
            assert_eq!(cleared.receive_wait, Duration::ZERO);
            assert_eq!(cleared.compute, Duration::ZERO);
            assert_eq!(cleared.order_send, None);
            assert_eq!(cleared.ack_send, None);
            assert_eq!(fixture.calls, [1, 1, 1]);
        }
    }
}
