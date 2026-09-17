"""Fake-clock pacing tests; no subprocess, simulator or wall-clock sleep."""

import socket
import struct
import unittest

from play_admission import Admission, Endpoint, SeatRelay
from play_match import parse_arguments, transport_arguments
from play_pacing import PacedClock, PERIOD, STALL_TIMEOUT, ack_tick


def integer(value):
    output = bytearray()
    for _ in range(5):
        output.append((value & 127) | (128 if value > 127 else 0))
        value >>= 7
        if not value:
            return bytes(output)
    raise ValueError("fixture integer exceeds u32")


def frame(payload):
    return struct.pack("<I", len(payload)) + payload


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.pacer = PacedClock(self.clock)
        for slot in (0, 1):
            self.pacer.observe_server(slot, b"\x02")

    def observe_tick(self, tick):
        for slot in (0, 1):
            self.pacer.observe_server(slot, b"\x03" + integer(tick))
            self.pacer.observe_server(slot, b"\x04" + integer(tick) + b"\x00")

    def test_both_original_acks_share_one_30hz_deadline(self):
        anchor = self.clock.now
        for tick in range(1, 91):
            self.observe_tick(tick)
            payload = b"\x04" + integer(tick)
            for slot in (0, 1):
                self.assertFalse(self.pacer.allow_ack(slot, payload))
            self.clock.now = self.pacer.due
            for slot in (1, 0):
                self.assertTrue(self.pacer.allow_ack(slot, payload))
        self.assertAlmostEqual(self.clock.now - anchor, 3.0)

    def test_late_frame_rebases_clock_without_fastforward_catchup(self):
        self.observe_tick(1)
        self.clock.now += 2.0
        for slot in (0, 1):
            self.assertTrue(self.pacer.allow_ack(slot, b"\x04\x01"))
        self.observe_tick(2)
        for slot in (0, 1):
            self.assertFalse(self.pacer.allow_ack(slot, b"\x04\x02"))
        self.assertAlmostEqual(self.pacer.due - self.clock.now, PERIOD)

    def test_ack_waits_for_both_complete_server_pairs_even_after_deadline(self):
        self.pacer.observe_server(0, b"\x03\x01")
        self.clock.now += 1.0
        self.assertFalse(self.pacer.allow_ack(0, b"\x04\x01"))
        self.pacer.observe_server(0, b"\x04\x01\x00")
        self.pacer.observe_server(1, b"\x03\x01")
        self.assertFalse(self.pacer.allow_ack(0, b"\x04\x01"))
        self.pacer.observe_server(1, b"\x04\x01\x00")
        self.assertTrue(self.pacer.allow_ack(0, b"\x04\x01"))

    def test_stale_future_duplicate_and_overwide_acks_cannot_reach_server_max_tick(self):
        self.observe_tick(1)
        for payload, message in ((b"\x04\x00", "stale"), (b"\x04\x02", "future"),
                                 (b"\x04\xff\xff\xff\xff\x1f", "u32"),
                                 (b"\x84\x00\x01", "canonical"),
                                 (b"\x04\x81\x00", "canonical"),
                                 (b"\x04\x01\x00", "trailing")):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                self.pacer.allow_ack(0, payload)
        self.clock.now = self.pacer.due
        self.assertTrue(self.pacer.allow_ack(0, b"\x04\x01"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.pacer.allow_ack(0, b"\x04\x01")

    def test_width_bounded_ack_parser_and_nonack_orders(self):
        self.assertEqual(ack_tick(b"\x04" + integer(2**32 - 1)), 2**32 - 1)
        self.assertIsNone(ack_tick(b"\x03\x01\x00\x00"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            ack_tick(b"\x04\x80")

    def test_native_timeout_advance_and_pacing_stall_fail_explicitly(self):
        self.observe_tick(1)
        with self.assertRaisesRegex(ValueError, "before both paced ACKs"):
            self.pacer.observe_server(0, b"\x03\x02")
        self.clock.now += STALL_TIMEOUT
        with self.assertRaisesRegex(RuntimeError, "paced lockstep stalled"):
            self.pacer.check_deadline()

    def test_coalesced_orders_and_ack_remain_fifo_with_no_extra_ack(self):
        for slot, role in ((0, "human"), (1, "bot")):
            self.setUp()
            self.observe_tick(1)
            relay = SeatRelay(1, slot, role, 1, pacer=self.pacer)
            pairs = [socket.socketpair(), socket.socketpair()]
            relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
            relay.hello = relay.welcomed = True
            before = frame(b"\x03\x01\x00\x00")
            ack = frame(b"\x04\x01")
            after = frame(b"\x03\x02\x00\x00")
            try:
                self.clock.now = self.pacer.due - PERIOD / 2
                for fragment in ((before + ack + after)[:3], (before + ack + after)[3:]):
                    relay.endpoints[0].incoming.extend(fragment)
                    relay.forward_frames(0)
                self.assertEqual(bytes(relay.endpoints[1].outgoing), before)
                self.assertEqual(bytes(relay.endpoints[0].incoming), ack + after)
                self.clock.now = self.pacer.due
                relay.forward_frames(0)
                self.assertEqual(bytes(relay.endpoints[1].outgoing), before + ack + after)
                self.assertEqual(bytes(relay.endpoints[0].incoming), b"")
                self.assertEqual(relay.endpoints[0].frames, 3)
            finally:
                relay.close()
                for pair in pairs:
                    pair[1].close()

    def test_only_reward_report_uses_paced_native_lockstep(self):
        for opponent in ("neural", "teacher"):
            arguments = parse_arguments(["--opponent", opponent])
            self.assertEqual(transport_arguments(arguments), ["--mode", "realtime"])
            arguments.reward_report = True
            self.assertEqual(transport_arguments(arguments),
                             ["--mode", "lockstep", "--ack-timeout-ticks", "900"])
        with self.assertRaisesRegex(ValueError, "pacing requires native Lockstep"):
            Admission(1, "radiant", paced=True)

    def test_held_ack_with_many_later_orders_drains_in_bounded_fifo_batches(self):
        self.observe_tick(1)
        relay = SeatRelay(1, 0, "human", 1, pacer=self.pacer)
        pairs = [socket.socketpair(), socket.socketpair()]
        relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
        relay.hello = relay.welcomed = True
        wire = frame(b"\x04\x01") + frame(b"\x03\x01\x00\x00") * 14000
        try:
            relay.endpoints[0].incoming.extend(wire)
            self.assertFalse(relay.forward_frames(0))
            self.clock.now = self.pacer.due
            for _ in range(3):
                relay.forward_frames(0)
            self.assertEqual(bytes(relay.endpoints[1].outgoing), wire)
            self.assertEqual(relay.endpoints[0].frames, 14001)
            self.assertEqual(relay.endpoints[0].incoming, b"")
        finally:
            relay.close()
            for pair in pairs:
                pair[1].close()

    def test_complete_ack_held_at_client_eof_is_not_mislabelled_truncated(self):
        self.observe_tick(1)
        relay = SeatRelay(1, 0, "human", 1, pacer=self.pacer)
        pairs = [socket.socketpair(), socket.socketpair()]
        relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
        relay.hello = relay.welcomed = True
        wire = frame(b"\x04\x01") + frame(b"\x03\x01\x00\x00")
        try:
            relay.endpoints[0].incoming.extend(wire)
            pairs[0][1].shutdown(socket.SHUT_WR)
            relay.receive(0)
            self.assertTrue(relay.endpoints[0].eof)
            self.assertEqual(bytes(relay.endpoints[0].incoming), wire)
            self.clock.now = self.pacer.due
            relay.forward_frames(0)
            self.assertEqual(bytes(relay.endpoints[1].outgoing), wire)
            self.assertFalse(relay.endpoints[1].write_closed)
        finally:
            relay.close()
            for pair in pairs:
                pair[1].close()

    def test_poll_timeout_uses_due_time_without_spinning_on_late_client(self):
        self.observe_tick(1)
        self.clock.now = self.pacer.due - 0.001
        self.assertAlmostEqual(self.pacer.wait_timeout(0.01), 0.001)
        self.clock.now = self.pacer.due + 1
        self.assertEqual(self.pacer.wait_timeout(0.01), 0.01)

    def test_pump_releases_due_ack_without_new_socket_read_and_rejects_truncated_eof(self):
        self.observe_tick(1)
        relay = SeatRelay(1, 0, "human", 1, pacer=self.pacer)
        pairs = [socket.socketpair(), socket.socketpair()]
        relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
        relay.hello = relay.welcomed = True
        wire = frame(b"\x04\x01")
        try:
            relay.endpoints[0].incoming.extend(wire)
            self.assertFalse(relay.pump())
            self.clock.now = self.pacer.due
            self.assertTrue(relay.pump())
            self.assertEqual(relay.endpoints[1].outgoing, b"")
            pairs[1][1].setblocking(False)
            self.assertEqual(pairs[1][1].recv(64), wire)
            relay.endpoints[0].incoming.extend(struct.pack("<I", 6) + b"\x04")
            relay.endpoints[0].eof = True
            with self.assertRaisesRegex(ValueError, "truncated relay frame at EOF"):
                relay.forward_frames(0)
        finally:
            relay.close()
            for pair in pairs:
                pair[1].close()

    def test_both_human_sides_share_clock_and_keep_original_admission_roles(self):
        for side in ("radiant", "dire"):
            admission = Admission(1, side, mode=1, paced=True, clock=self.clock)
            try:
                self.assertTrue(all(relay.pacer is admission.pacer for relay in admission.relays))
                self.assertEqual([relay.slot for relay in admission.relays], [0, 1])
                self.assertEqual([relay.role for relay in admission.relays],
                                 ["human", "bot"] if side == "radiant" else ["bot", "human"])
                with self.assertRaisesRegex(ValueError, "before first Snapshot"):
                    admission.pacer.allow_ack(0, b"\x04\x01")
            finally:
                admission.close()


if __name__ == "__main__":
    unittest.main()
