use bota_proto::{EntityId, Order, ServerMsg};

use super::{Clock, UpdateTiming, add_duration, subtract_duration};
use crate::Wire;

pub(crate) struct MeasuredWire<'a, W, C> {
    wire: &'a mut W,
    clock: &'a C,
    timing: UpdateTiming,
    receive_scope: &'static str,
}

impl<'a, W: Wire, C: Clock> MeasuredWire<'a, W, C> {
    pub(crate) fn new(wire: &'a mut W, clock: &'a C) -> Self {
        Self {
            wire,
            clock,
            timing: UpdateTiming::default(),
            receive_scope: "unavailable",
        }
    }

    pub(crate) fn take_timing(&mut self) -> UpdateTiming {
        std::mem::take(&mut self.timing)
    }

    pub(crate) const fn receive_scope(&self) -> &'static str {
        self.receive_scope
    }
}

impl<W: Wire, C: Clock> Wire for MeasuredWire<'_, W, C> {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        let started = self.clock.now();
        let result = self.wire.hear();
        let elapsed = subtract_duration(self.clock.now(), started, &mut self.timing.saturated);
        let wait = self.wire.take_receive_wait();
        let scope = if wait.is_some() {
            "socket_read"
        } else {
            "wire_hear_including_decode"
        };
        self.receive_scope = if self.receive_scope == "unavailable" || self.receive_scope == scope {
            scope
        } else {
            "mixed_socket_read_and_wire_hear"
        };
        let wait = wait.unwrap_or(elapsed);
        self.timing.receive_wait =
            add_duration(self.timing.receive_wait, wait, &mut self.timing.saturated);
        let compute = subtract_duration(elapsed, wait, &mut self.timing.saturated);
        self.timing.compute =
            add_duration(self.timing.compute, compute, &mut self.timing.saturated);
        result
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        let started = self.clock.now();
        let result = self.wire.order(unit, order);
        let elapsed = subtract_duration(self.clock.now(), started, &mut self.timing.saturated);
        self.timing.order_send = Some(add_duration(
            self.timing.order_send.unwrap_or_default(),
            elapsed,
            &mut self.timing.saturated,
        ));
        result
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        let started = self.clock.now();
        let result = self.wire.acknowledge(tick);
        let elapsed = subtract_duration(self.clock.now(), started, &mut self.timing.saturated);
        self.timing.ack_send = Some(add_duration(
            self.timing.ack_send.unwrap_or_default(),
            elapsed,
            &mut self.timing.saturated,
        ));
        result
    }
}
