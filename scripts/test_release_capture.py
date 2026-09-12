"""Deterministic terminal-tail streams and process-exit ordering; no real games or sleeps."""

from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import release_crossplay as crossplay
import release_wire as wire
from test_release_crossplay import client
from test_release_wire import frame, integer, observer


def order(sequence):
    return frame(b"\x03" + integer(sequence) + b"\x00\x09\x00")


class CapturedRelay(wire.Relay):
    def filter_client(self, data):
        parsed = super().filter_client(data)
        self.captured.extend(parsed)
        return parsed

    def close(self):
        super().close()
        self.finalized_capture = bytes(self.captured)


def terminal_relay(slot=0, finished=True):
    relay = CapturedRelay.__new__(CapturedRelay)
    relay.__dict__.update(observer(27900).__dict__)
    relay.observed.update(slot=slot, last_snapshot=100)
    relay.stop = threading.Event()
    relay.welcomed = threading.Event()
    relay.welcomed.set()
    relay.listener = Mock()
    relay.address = "127.0.0.1:4455"
    relay.captured = bytearray()
    relay.prefix = b"".join(order(sequence) for sequence in range(1, 282))
    relay.filter_client(relay.prefix)
    if finished:
        relay.observe(frame(b"\x07\x00\x64\x02" + bytes(9) + b"\x01" + bytes(8)))
        assert relay.observed["winner"] == "Radiant"
    return relay


def tail_stream(relay, chunks=None, server_eof_first=True):
    tail = order(282) + frame(b"\x04\x64")
    chunks = [tail] if chunks is None else chunks
    client_socket = Mock(recv=Mock(side_effect=chunks + [b""]))
    server_socket = Mock(recv=Mock(return_value=b""))
    sources = ([server_socket] if server_eof_first else []) + [client_socket] * len(chunks)
    sources += ([] if server_eof_first else [server_socket]) + [client_socket]

    def readable(readers, writers, errors, timeout):
        assert readers and sources, "poll beyond finite terminal fixture"
        assert timeout == 0.1
        source = sources.pop(0)
        assert source in readers
        return [source], [], []

    def drain():
        with patch.object(wire.select, "select", side_effect=readable):
            relay.pump(client_socket, server_socket)

    return drain, client_socket, server_socket, sources


def queued_thread(relay, drain, stuck=False):
    completed = []

    def join(timeout):
        assert 0 <= timeout <= 6
        if not completed and not stuck:
            try:
                drain()
            except (OSError, ValueError) as error:
                relay.observed["errors"].append(str(error))
            completed.append(True)

    relay.thread = Mock(join=Mock(side_effect=join),
                        is_alive=Mock(side_effect=lambda: not completed and not relay.stop.is_set()))


def executed_terminal_game(stuck=False, malformed=False):
    relays = [terminal_relay(slot) for slot in range(2)]
    servers = []
    for slot, relay in enumerate(relays):
        chunks = [order(282)[:-1]] if malformed and slot == 0 else None
        drain, _, server, _ = tail_stream(relay, chunks)
        queued_thread(relay, drain, stuck and slot == 0)
        servers.append(server)

    def launch(command, path, processes, handles, environment=None):
        index = len(processes)
        process = Mock(pid=101 + index, returncode=0, poll=Mock(return_value=0))
        processes.append(process)
        text = "listening" if index == 0 else client(crossplay.SIDES[index - 1])["stdout"]
        path.write_text(text.replace("10 decisions, 5 orders", "282 decisions, 282 orders"))
        return process

    with tempfile.TemporaryDirectory() as temporary, \
            patch.object(crossplay, "validate_runtime_contract", return_value=wire.CURRENT_SIMULATOR), \
            patch.object(crossplay, "launch", side_effect=launch), \
            patch.object(crossplay, "server_port", return_value=4455), \
            patch.object(crossplay, "Relay", side_effect=relays), \
            patch.object(crossplay.time, "monotonic", return_value=0), \
            patch.object(crossplay.time, "sleep", side_effect=AssertionError("unexpected wall sleep")):
        game = crossplay.execute_game(Path(temporary), Path("unused-server"),
            [dict(binary=Path("unused-bot"), policy="teacher")] * 2, 7,
            crossplay.current_map2_registry("a" * 64))
    return game, relays, servers


class CaptureTests(unittest.TestCase):
    def test_order_282_and_ack_after_verified_terminal_and_server_eof_are_accounted(self):
        relay = terminal_relay()
        drain, client_socket, server_socket, sources = tail_stream(relay)
        drain()
        self.assertEqual(relay.captured, relay.prefix + order(282) + frame(b"\x04\x64"))
        self.assertEqual(relay.client_frames, 283)
        self.assertEqual(relay.client_bytes, len(relay.captured))
        self.assertEqual(sources, [])
        server_socket.sendall.assert_not_called()
        client_socket.shutdown.assert_called_once_with(socket.SHUT_WR)

    def test_verified_matchover_blocks_forwarding_even_before_server_eof(self):
        relay = terminal_relay()
        drain, _, server_socket, _ = tail_stream(relay, server_eof_first=False)
        drain()
        self.assertEqual(relay.client_frames, 283)
        server_socket.sendall.assert_not_called()

    def test_fragmented_late_order_and_ack_are_observed_once_at_every_split(self):
        tail = order(282) + frame(b"\x04\x64")
        for split in range(len(tail) + 1):
            with self.subTest(split=split):
                relay = terminal_relay()
                drain, _, server_socket, _ = tail_stream(relay, [part for part in (tail[:split], tail[split:]) if part])
                drain()
                self.assertEqual(relay.captured, relay.prefix + tail)
                self.assertEqual(relay.client_buffer, b"")
                server_socket.sendall.assert_not_called()

    def test_real_evaluator_drains_after_clients_exit_before_finalizing_observation(self):
        game, relays, servers = executed_terminal_game()
        self.assertEqual((game["result"], game["errors"], game["wall_timeout"]), ("win", [], False))
        self.assertEqual(game["killed_pids"], [])
        for relay, server in zip(relays, servers):
            self.assertEqual(relay.finalized_capture, relay.prefix + order(282) + frame(b"\x04\x64"))
            self.assertTrue(relay.observed["client_eof"])
            server.sendall.assert_not_called()

    def test_truncated_client_tail_is_error_despite_native_win_and_zero_process_exits(self):
        game, _, _ = executed_terminal_game(malformed=True)
        self.assertEqual((game["result"], game["errors"], game["wall_timeout"]),
                         ("error", ["truncated client frame"], False))

    def test_natural_drain_timeout_is_explicit_error_not_match_wall_timeout(self):
        game, _, _ = executed_terminal_game(stuck=True)
        self.assertEqual((game["result"], game["errors"], game["wall_timeout"]),
                         ("error", ["relay drain timeout"], False))

    def test_finish_is_bounded_and_does_not_force_stop(self):
        relay = terminal_relay()
        relay.thread = Mock(is_alive=Mock(return_value=True))
        self.assertFalse(relay.finish(timeout=0.25))
        relay.thread.join.assert_called_once_with(timeout=0.25)
        self.assertFalse(relay.stop.is_set())
        self.assertEqual(relay.observed["errors"], ["relay drain timeout"])
        for timeout in (-1, 2.01, float("nan")):
            with self.assertRaisesRegex(ValueError, "relay drain timeout must be 0..2 seconds"):
                relay.finish(timeout=timeout)

    def test_thread_completion_without_client_eof_cannot_silently_pass(self):
        relay = terminal_relay()
        relay.thread = Mock(is_alive=Mock(return_value=False))
        self.assertFalse(relay.finish(timeout=0))
        self.assertEqual(relay.observed["errors"], ["relay finished before client EOF"])

    def test_forced_timeout_cleanup_skips_natural_drain(self):
        relay = Mock()
        crossplay.stop_game([], [relay], [], set(), True)
        relay.finish.assert_not_called()
        relay.close.assert_called_once_with()

    def test_two_relays_share_one_two_second_natural_drain_budget(self):
        relays = [Mock(), Mock()]
        with patch.object(crossplay.time, "monotonic", side_effect=[10, 10.5, 12.5]):
            crossplay.stop_game([], relays, [], set(), False)
        relays[0].finish.assert_called_once_with(timeout=1.5)
        relays[1].finish.assert_called_once_with(timeout=0)

    def test_closed_client_read_half_does_not_discard_its_late_write_half(self):
        relay = terminal_relay()
        relay.tick_limit = 100
        relay.observed["cap_events"] = True
        self.assertEqual(relay.filter_client(frame(b"\x04\x64")), b"")
        tail = order(282)
        client_socket = Mock(sendall=Mock(side_effect=BrokenPipeError()),
                             recv=Mock(side_effect=[tail, b""]))
        server_socket = Mock(recv=Mock(side_effect=[frame(b"\x08\x00\x00"), b""]))
        schedule = iter([server_socket, server_socket, client_socket, client_socket])
        with patch.object(wire.select, "select", side_effect=lambda *_: ([next(schedule)], [], [])):
            relay.pump(client_socket, server_socket)
        self.assertEqual(relay.captured, relay.prefix + tail)
        self.assertTrue(relay.observed["client_eof"])
        server_socket.sendall.assert_not_called()

    def test_preterminal_order_is_forwarded_but_later_terminal_tail_is_only_accounted(self):
        relay = terminal_relay(finished=False)
        before, after = order(282), order(283) + frame(b"\x04\x64")
        client_socket = Mock(recv=Mock(side_effect=[before, after, b""]))
        terminal = frame(b"\x07\x00\x64\x02" + bytes(9) + b"\x01" + bytes(8))
        server_socket = Mock(recv=Mock(side_effect=[terminal, b""]))
        schedule = iter([client_socket, server_socket, server_socket, client_socket, client_socket])
        with patch.object(wire.select, "select", side_effect=lambda *_: ([next(schedule)], [], [])):
            relay.pump(client_socket, server_socket)
        server_socket.sendall.assert_called_once_with(before)
        self.assertEqual(relay.captured, relay.prefix + before + after)
        self.assertTrue(relay.observed["client_eof"])


class TailValidationTests(unittest.TestCase):
    def test_truncated_late_header_or_body_fails_at_client_eof(self):
        for tail in (b"\x05", order(282)[:-1]):
            relay = terminal_relay()
            drain, _, _, _ = tail_stream(relay, [tail])
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, "truncated client frame"):
                drain()

    def test_late_client_bounds_remain_enforced(self):
        for size in (0, wire.FRAME_LIMIT + 1):
            relay = terminal_relay()
            drain, _, _, _ = tail_stream(relay, [struct.pack("<I", size)])
            with self.assertRaisesRegex(ValueError, "invalid client frame length"):
                drain()
        for field, value, message in (("client_bytes", wire.CLIENT_BYTE_LIMIT, "client wire byte limit exceeded"),
                                      ("client_frames", 27900 * 4 + 1000, "client wire frame limit exceeded")):
            relay = terminal_relay()
            setattr(relay, field, value)
            drain, _, _, _ = tail_stream(relay)
            with self.assertRaisesRegex(ValueError, message):
                drain()

    def test_late_ack_and_order_payloads_fail_closed_when_malformed(self):
        cases = ((b"\x04\x65", "ACK exceeds observed snapshot or malformed ACK"),
                 (b"\x04\x64\x00", "ACK exceeds observed snapshot or malformed ACK"),
                 (b"\x03\x01\x00\x09", "truncated postcard byte"),
                 (b"\x03\x01\x00\x0a", "unknown client order"),
                 (b"\x03\x01\x00\x00\x03", "unknown client target"),
                 (b"\x03\x01\x00\x09\x00\x00", "trailing client message bytes"),
                 (b"\x06", "unknown client message"))
        for payload, message in cases:
            relay = terminal_relay()
            drain, _, _, _ = tail_stream(relay, [frame(payload)])
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                drain()

    def test_all_client_variants_are_wire_valid_under_both_protocol_pins(self):
        orders = (b"\x00\x00", b"\x01\x01\x01\x02", b"\x02\xff\x00", b"\x03\x0e\x02\x01\x02",
                  b"\x04\x01\x00", b"\x05\x02\x01\x02", b"\x06\xac\x02", b"\x07\x02",
                  b"\x08\x00\x0e", b"\x09\x00")
        payloads = [b"\x00\x01\x03bot", b"\x01\x02", b"\x02\x01", b"\x04\x64", b"\x05\x00", b"\x05\x01\x01"]
        payloads += [b"\x03\x01\x01\x05\x06" + item for item in orders]
        for simulator in (wire.CURRENT_SIMULATOR, wire.HISTORICAL_SIMULATOR):
            relay = terminal_relay()
            relay.simulator_commit = simulator
            for payload in payloads:
                with self.subTest(simulator=simulator, payload=payload):
                    self.assertEqual(relay.filter_client(frame(payload)), frame(payload))

    def test_invalid_hello_ready_and_view_tails_are_not_silently_discarded(self):
        cases = ((b"\x00\x03\x00", "invalid client role"), (b"\x00\x01\x02x", "truncated client name"),
                 (b"\x00\x01\x01\xff", "invalid client name UTF-8"), (b"\x02\x02", "invalid postcard ready"),
                 (b"\x05\x02", "invalid postcard option"), (b"\x05\x00\x00", "trailing client message bytes"))
        for payload, message in cases:
            relay = terminal_relay()
            drain, _, _, _ = tail_stream(relay, [frame(payload)])
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                drain()


if __name__ == "__main__":
    unittest.main()
