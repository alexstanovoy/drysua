"""Portable diagnostic regressions; no native binaries or weights required."""

import contextlib
import io
import json
from pathlib import Path
import socket
import signal
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import play_match
from play_admission import Admission, Endpoint, SeatRelay, varint, verify_hello
from play_reward import RewardPipe, print_overview


def integer(value):
    """Encode one canonical postcard unsigned integer fixture."""
    assert 0 <= value < 2**64
    data = bytearray()
    for _ in range(10):
        data.append((value & 127) | (128 if value >= 128 else 0))
        value >>= 7
        if not value:
            return bytes(data)
    raise AssertionError("fixture integer exceeds u64")


def frame(payload):
    assert 0 < len(payload) <= 4 * 1024 * 1024
    return struct.pack("<I", len(payload)) + payload


def relay_fixture(test, slot=0, role="human", mode=0, pacer=None):
    relay = SeatRelay(1, slot, role, mode, pacer=pacer)
    test.addCleanup(relay.close)
    pairs = []
    for _ in range(2):
        pair = socket.socketpair()
        pairs.append(pair)
        for connection in pair:
            test.addCleanup(connection.close)
    relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
    return relay, pairs


class VarintTests(unittest.TestCase):
    def test_varint_u64_maximum_is_valid_but_overflow_and_negative_offset_fail(self):
        self.assertEqual(varint(integer(2**64 - 1), 0), (2**64 - 1, 10))
        for payload, offset, message in ((b"\xff" * 9 + b"\x02", 0, "exceeds u64"),
                                         (b"\x00", -1, "invalid postcard offset"),
                                         (b"\x80\x00", 0, "noncanonical postcard integer")):
            with self.assertRaisesRegex(ValueError, message):
                varint(payload, offset)


class OpponentTests(unittest.TestCase):
    def test_teacher_needs_no_weights_and_exact_explicit_policy(self):
        for side in ("radiant", "dire"):
            arguments = play_match.parse_arguments(["--opponent", "teacher", "--human-side", side])
            with patch("play_match.current_paths", side_effect=AssertionError("weights touched")):
                binary, weights = play_match.opponent_paths(Path("/repo"), arguments)
            self.assertIsNone(weights)
            self.assertEqual(play_match.bot_command(binary, "127.0.0.1:1", arguments.opponent, weights),
                             [str(binary), "--addr", "127.0.0.1:1", "--name", "drysua", "--policy", "teacher"])

    def test_watch_seats_a_named_rule_bot_where_the_human_would_sit(self):
        for side, slot in (("radiant", 0), ("dire", 1)):
            arguments = play_match.parse_arguments(
                ["--watch", "harass-push", "--opponent", "teacher", "--human-side", side])
            admission = Admission(1, arguments.human_side, watch=arguments.watch is not None)
            try:
                self.assertEqual([relay.role for relay in admission.relays][slot], "watched")
                verify_hello(b"\x00\x01\x07watched", "watched")
                with self.assertRaisesRegex(ValueError, "expected watched identity"):
                    verify_hello(b"\x00\x00\x05human", "watched")
                self.assertEqual(
                    play_match.bot_command(Path("/drysua"), "127.0.0.1:2", arguments.watch, None, name="watched"),
                    ["/drysua", "--addr", "127.0.0.1:2", "--name", "watched", "--policy", "harass-push"])
            finally:
                admission.close()

    def test_default_neural_still_fails_closed_without_weights(self):
        arguments = play_match.parse_arguments([])
        self.assertEqual(arguments.opponent, "neural")
        with self.assertRaisesRegex(RuntimeError, "no Teacher fallback"):
            play_match.opponent_paths(Path("/repo"), arguments)

    def test_ambiguous_teacher_weights_and_bad_flags_rejected(self):
        for flags in (["--opponent", "teacher", "--weights-directory", "."],
                      ["--opponent", "harass-push", "--weights-directory", "."], ["--watch", "neural"],
                      ["--opponent", "hybrid"], ["--reward-interval", "0"],
                      ["--human-side", "dire", "--bot-side", "dire"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                play_match.parse_arguments(flags)


class TeeTests(unittest.TestCase):
    def reward_pipe(self, role="human"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        reader, writer = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        pipe = RewardPipe(writer, Path(directory.name) / role, role)
        self.addCleanup(pipe.close)
        return pipe, reader

    def test_fragmented_welcome_reaches_player_and_passive_pipe_without_extra_ack(self):
        for slot, role in ((0, "human"), (1, "bot")):
            with self.subTest(role=role):
                relay, _ = relay_fixture(self, slot, role)
                pipe, reader = self.reward_pipe(role)
                reader.setblocking(False)
                relay.observer = Mock(wraps=pipe.offer)
                name = b"human" if role == "human" else b"drysua"
                hello = frame(bytes([0, slot, len(name)]) + name)
                relay.endpoints[0].incoming.extend(hello)
                relay.forward_frames(0)
                self.assertEqual(relay.endpoints[1].outgoing, hello)
                relay.observer.assert_not_called()
                with self.assertRaises(BlockingIOError):
                    reader.recv(64)
                welcome = frame(bytes([0, slot + 1, 1, slot, 30, 0]))
                for fragment in (welcome[:2], welcome[2:5], welcome[5:]):
                    relay.endpoints[1].incoming.extend(fragment)
                    relay.forward_frames(1)
                pipe.finish()
                relay.observer.assert_called_once_with(welcome)
                self.assertEqual(reader.recv(64), welcome)
                self.assertEqual(reader.recv(64), b"")
                self.assertEqual(relay.endpoints[0].outgoing, welcome)
                self.assertEqual(relay.endpoints[1].outgoing, hello)
                self.assertIsNone(pipe.failure)

    def test_failed_report_repeated_overview_preserves_original_observer_error(self):
        pipe, _ = self.reward_pipe()
        path = pipe.prefix.with_suffix(".json")
        report = {"valid": False, "complete": False, "error": "EOF before MatchOver",
                  "components": {}, "total": 0, "total_without_terminal": 0}
        path.write_text(json.dumps(report))
        pipe.fail("observer rejected stream")
        with contextlib.redirect_stdout(io.StringIO()):
            print_overview([(pipe, None, None)])
            first = json.loads(path.read_text())
            print_overview([(pipe, None, None)])
        second = json.loads(path.read_text())
        self.assertEqual(first, second)
        self.assertEqual(second["observer_error"], "EOF before MatchOver")

    def test_marker_io_failure_during_overflow_cannot_drop_original_server_frame(self):
        pipe, _ = self.reward_pipe()
        relay, _ = relay_fixture(self)
        relay.hello = True
        relay.observer = pipe.offer
        welcome = frame(bytes([0, 1, 1, 0, 30, 0]))
        pipe.pending.extend(b"x" * pipe.limit)
        relay.endpoints[1].incoming.extend(welcome)
        with patch.object(pipe, "pump"), patch.object(Path, "open", side_effect=OSError("disk full")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            relay.forward_frames(1)
        self.assertEqual(bytes(relay.endpoints[0].outgoing), welcome)
        self.assertIn("queue limit", pipe.failure)
        self.assertIn("disk full", pipe.persistence_error)
        self.assertIn("persistence failed", output.getvalue())
        self.assertTrue(pipe.closed)

    def test_final_report_rewrite_io_failure_is_reported_without_raising(self):
        pipe, _ = self.reward_pipe()
        path = pipe.prefix.with_suffix(".json")
        path.write_text(json.dumps({"error": "EOF before MatchOver", "components": {}}))
        original_open = Path.open

        def fail_writes(path, mode="r", *args, **kwargs):
            if mode == "w":
                raise PermissionError("read-only report")
            return original_open(path, mode, *args, **kwargs)

        pipe.fail("observer rejected stream")
        with patch.object(Path, "open", fail_writes), contextlib.redirect_stdout(io.StringIO()) as output:
            print_overview([(pipe, None, None)])
        self.assertIn("persistence failed", output.getvalue())
        self.assertIn("read-only report", pipe.persistence_error)
        self.assertIn("INVALID/INCOMPLETE", output.getvalue())

    def test_reporting_exception_cannot_skip_child_group_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = play_match.Supervisor(Path(directory))
            with patch("play_match.finish_observers", side_effect=TypeError("malformed report")), \
                    patch.object(supervisor, "signal_groups", return_value=[]) as groups:
                with self.assertRaisesRegex(RuntimeError, "finalizing reward observation: malformed report"):
                    supervisor.close()
                self.assertEqual([call.args[0] for call in groups.call_args_list],
                                 [signal.SIGTERM, signal.SIGKILL])

    def test_backpressure_invalidates_instead_of_dropping_silently(self):
        pipe, _ = self.reward_pipe()
        with patch.object(pipe, "pump"):
            pipe.offer(b"x" * pipe.limit)
            pipe.offer(b"x")
        self.assertIn("queue limit", pipe.failure)
        self.assertEqual(len(pipe.pending), 0)
        self.assertTrue(pipe.failure_path.is_file())

    def test_observer_crash_is_incomplete_not_terminal(self):
        pipe, reader = self.reward_pipe()
        reader.close()
        pipe.offer(b"frame")
        self.assertIn("observer write failed", pipe.failure)


if __name__ == "__main__":
    unittest.main()
