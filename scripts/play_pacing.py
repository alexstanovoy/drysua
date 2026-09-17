"""Shared 30 Hz deadlines for original native Lockstep ACKs; no generated input."""

import time


PERIOD = 1 / 30
STALL_TIMEOUT = 10
ACK_TIMEOUT_TICKS = 900
MAX_TICK = 27900
assert STALL_TIMEOUT < ACK_TIMEOUT_TICKS / 30


def read_u32(payload, offset):
    start, value = offset, 0
    for index in range(5):
        if offset >= len(payload):
            raise ValueError("truncated pacing u32")
        byte = payload[offset]
        offset += 1
        if index == 4 and byte > 15:
            raise ValueError("pacing integer exceeds u32")
        value |= (byte & 127) << (7 * index)
        if byte < 128:
            if offset - start != max(1, (value.bit_length() + 6) // 7):
                raise ValueError("noncanonical pacing integer")
            return value, offset
    raise ValueError("pacing integer exceeds u32")


def ack_tick(payload):
    kind, offset = read_u32(payload, 0)
    if kind != 4:
        return None
    tick, offset = read_u32(payload, offset)
    if offset != len(payload):
        raise ValueError("trailing bytes in paced ACK")
    return tick


class PacedClock:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = [False, False]
        self.finished = [False, False]
        self.snapshots = [0, 0]
        self.completed = [0, 0]
        self.acked = [0, 0]
        self.tick = 0
        self.due = None
        self.progress = None

    def observe_server(self, slot, payload):
        assert slot in (0, 1)
        kind, offset = read_u32(payload, 0)
        if kind == 2:
            if self.started[slot]:
                raise ValueError("duplicate paced MatchStart")
            self.started[slot] = True
            if self.progress is None:
                self.progress = self.clock()
        elif kind in (3, 4):
            tick, _ = read_u32(payload, offset)
            if not self.started[slot]:
                raise ValueError("paced tick before MatchStart")
            if kind == 3:
                self.observe_snapshot(slot, tick)
            else:
                if tick != self.snapshots[slot] or tick != self.completed[slot] + 1:
                    raise ValueError("paced Events do not complete the next Snapshot")
                self.completed[slot] = tick
        elif kind == 7:
            self.finished[slot] = True

    def observe_snapshot(self, slot, tick):
        if not 1 <= tick <= MAX_TICK or tick != self.snapshots[slot] + 1:
            raise ValueError("paced Snapshot ticks must start at one and be contiguous")
        if self.snapshots[slot] != self.completed[slot]:
            raise ValueError("paced Snapshot arrived before previous Events")
        if tick == self.tick + 1:
            if self.acked != [self.tick, self.tick]:
                raise ValueError("server advanced before both paced ACKs")
            self.tick = tick
            self.progress = self.clock()
            # Rebase on the actual next tick; a late client never accumulates catch-up credit.
            self.due = self.progress + PERIOD
        elif tick != self.tick:
            raise ValueError("paced seat Snapshot differs from shared tick")
        self.snapshots[slot] = tick

    def allow_ack(self, slot, payload):
        assert slot in (0, 1)
        tick = ack_tick(payload)
        if tick is None:
            return True
        if not self.started[slot] or self.snapshots[slot] == 0:
            raise ValueError("paced ACK before first Snapshot")
        if tick < self.snapshots[slot]:
            raise ValueError("stale paced ACK tick")
        if tick > self.snapshots[slot]:
            raise ValueError("future paced ACK tick")
        if tick <= self.acked[slot]:
            raise ValueError("duplicate paced ACK tick")
        if tick != self.acked[slot] + 1:
            raise ValueError("paced ACK ticks must be contiguous")
        if self.completed != [tick, tick] or self.clock() < self.due:
            return False
        self.acked[slot] = tick
        return True

    def check_deadline(self):
        if (self.progress is not None and not all(self.finished)
                and self.clock() - self.progress >= STALL_TIMEOUT):
            raise RuntimeError(f"paced lockstep stalled at tick {self.tick} for {STALL_TIMEOUT} seconds; "
                               "aborting before native ACK timeout; reward INCOMPLETE")

    def wait_timeout(self, timeout):
        if self.due is not None:
            remaining = self.due - self.clock()
            if remaining > 0:
                return min(timeout, remaining)
        return timeout
