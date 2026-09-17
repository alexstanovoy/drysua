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
from unittest.mock import patch

import play_match
from play_admission import Endpoint, SeatRelay
from play_reward import RewardPipe, print_overview


class OpponentTests(unittest.TestCase):
    def test_teacher_needs_no_weights_and_exact_explicit_policy(self):
        for side in ("radiant", "dire"):
            arguments = play_match.parse_arguments(["--opponent", "teacher", "--human-side", side])
            with patch("play_match.read_runtime_metadata", side_effect=AssertionError("weights touched")):
                binary, weights = play_match.opponent_paths(Path("/repo"), arguments)
            self.assertIsNone(weights)
            self.assertEqual(play_match.bot_command(binary, "127.0.0.1:1", arguments, weights),
                             [str(binary), "--addr", "127.0.0.1:1", "--name", "drysua", "--policy", "teacher"])

    def test_default_neural_still_fails_closed_without_weights(self):
        arguments = play_match.parse_arguments([])
        self.assertEqual(arguments.opponent, "neural")
        with self.assertRaisesRegex(RuntimeError, "no Teacher fallback"):
            play_match.opponent_paths(Path("/repo"), arguments)

    def test_ambiguous_teacher_weights_and_bad_flags_rejected(self):
        for flags in (["--opponent", "teacher", "--weights-directory", "."],
                      ["--opponent", "hybrid"], ["--reward-interval", "0"],
                      ["--human-side", "dire", "--bot-side", "dire"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                play_match.parse_arguments(flags)


class TeeTests(unittest.TestCase):
    def test_failed_report_repeated_overview_preserves_original_observer_error(self):
        with tempfile.TemporaryDirectory() as directory:
            reader, writer = socket.socketpair()
            pipe = RewardPipe(writer, Path(directory) / "human", "human")
            path = pipe.prefix.with_suffix(".json")
            report = {"valid": False, "complete": False, "error": "EOF before MatchOver",
                      "components": {}, "total": 0, "total_without_terminal": 0}
            path.write_text(json.dumps(report))
            try:
                pipe.fail("observer rejected stream")
                with contextlib.redirect_stdout(io.StringIO()):
                    print_overview([(pipe, None, None)])
                    first = json.loads(path.read_text())
                    print_overview([(pipe, None, None)])
                second = json.loads(path.read_text())
                self.assertEqual(first, second)
                self.assertEqual(second["observer_error"], "EOF before MatchOver")
            finally:
                pipe.close()
                reader.close()

    def test_marker_io_failure_during_overflow_cannot_drop_original_server_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            reader, writer = socket.socketpair()
            pipe = RewardPipe(writer, Path(directory) / "human", "human")
            relay = SeatRelay(1, 0, "human", 0)
            pairs = [socket.socketpair(), socket.socketpair()]
            relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
            relay.hello = True
            relay.observer = pipe.offer
            payload = bytes([0, 1, 1, 0, 30, 0])
            frame = struct.pack("<I", len(payload)) + payload
            try:
                pipe.pending.extend(b"x" * pipe.limit)
                relay.endpoints[1].incoming.extend(frame)
                with patch.object(pipe, "pump"), patch.object(Path, "open", side_effect=OSError("disk full")), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    relay.forward_frames(1)
                self.assertEqual(bytes(relay.endpoints[0].outgoing), frame)
                self.assertIn("queue limit", pipe.failure)
                self.assertIn("disk full", pipe.persistence_error)
                self.assertIn("persistence failed", output.getvalue())
                self.assertTrue(pipe.closed)
            finally:
                pipe.close()
                reader.close()
                relay.close()
                for pair in pairs:
                    pair[1].close()

    def test_final_report_rewrite_io_failure_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            reader, writer = socket.socketpair()
            pipe = RewardPipe(writer, Path(directory) / "human", "human")
            path = pipe.prefix.with_suffix(".json")
            path.write_text(json.dumps({"error": "EOF before MatchOver", "components": {}}))
            original_open = Path.open

            def fail_writes(path, mode="r", *args, **kwargs):
                if mode == "w":
                    raise PermissionError("read-only report")
                return original_open(path, mode, *args, **kwargs)

            try:
                pipe.fail("observer rejected stream")
                with patch.object(Path, "open", fail_writes), contextlib.redirect_stdout(io.StringIO()) as output:
                    print_overview([(pipe, None, None)])
                self.assertIn("persistence failed", output.getvalue())
                self.assertIn("read-only report", pipe.persistence_error)
                self.assertIn("INVALID/INCOMPLETE", output.getvalue())
            finally:
                pipe.close()
                reader.close()

    def test_reporting_exception_cannot_skip_child_group_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = play_match.Supervisor(Path(directory))
            with patch("play_match.finish_observers", side_effect=TypeError("malformed report")), \
                    patch.object(supervisor, "signal_groups", return_value=[]) as groups:
                with self.assertRaisesRegex(RuntimeError, "finalizing reward observation: malformed report"):
                    supervisor.close()
                self.assertEqual([call.args[0] for call in groups.call_args_list],
                                 [signal.SIGTERM, signal.SIGKILL])

    def test_complete_server_frames_copied_without_order_changes_or_ack(self):
        for role, slot in (("human", 0), ("bot", 1)):
            relay = SeatRelay(1, slot, role, 0)
            pairs = [socket.socketpair(), socket.socketpair()]
            copies = []
            relay.observer = copies.append
            relay.endpoints = [Endpoint(pair[0]) for pair in pairs]
            name = b"human" if role == "human" else b"drysua"
            hello = bytes([0, 0 if role == "human" else 1, len(name)]) + name
            wire = struct.pack("<I", len(hello)) + hello
            try:
                relay.endpoints[0].incoming.extend(wire)
                relay.forward_frames(0)
                self.assertEqual(bytes(relay.endpoints[1].outgoing), wire)
                self.assertEqual(copies, [])
                payload = bytes([0, slot + 1, 1, slot, 30, 0])
                frame = struct.pack("<I", len(payload)) + payload
                for fragment in (frame[:2], frame[2:5], frame[5:]):
                    relay.endpoints[1].incoming.extend(fragment)
                    relay.forward_frames(1)
                self.assertEqual(copies, [frame])
                self.assertEqual(bytes(relay.endpoints[0].outgoing), frame)
                self.assertEqual(bytes(relay.endpoints[1].outgoing), wire)
            finally:
                relay.close()
                for pair in pairs:
                    pair[1].close()

    def test_backpressure_invalidates_instead_of_dropping_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            reader, writer = socket.socketpair()
            pipe = RewardPipe(writer, Path(directory) / "human", "human")
            try:
                with patch.object(pipe, "pump"):
                    pipe.offer(b"x" * pipe.limit)
                    pipe.offer(b"x")
                self.assertIn("queue limit", pipe.failure)
                self.assertEqual(len(pipe.pending), 0)
                self.assertTrue(pipe.failure_path.is_file())
            finally:
                pipe.close()
                reader.close()

    def test_observer_crash_is_incomplete_not_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            reader, writer = socket.socketpair()
            reader.close()
            pipe = RewardPipe(writer, Path(directory) / "human", "human")
            pipe.offer(b"frame")
            self.assertIn("observer write failed", pipe.failure)
            pipe.close()


if __name__ == "__main__":
    unittest.main()
