"""Release gate regression tests (stdlib only)."""

import unittest
from unittest.mock import Mock, patch
import socket
import struct
import threading
from pathlib import Path

from release_build import read_registry, SIMULATOR

from release_crossplay import check_resources, evaluate_gate, validate_game
from release_wire import Relay


def games(tag="v0.0.1", wins=11):
    return [dict(opponent=tag, seed=seed, side=side,
                 result="win" if index < wins else "loss", errors=[])
            for index, (seed, side) in enumerate(
                (seed, side) for seed in range(10) for side in ("Radiant", "Dire"))]


class GateTests(unittest.TestCase):
    def test_ten_of_twenty_fails_strict_majority(self):
        self.assertFalse(evaluate_gate(games(wins=10), ["v0.0.1"], list(range(10)))["passed"])

    def test_eleven_of_twenty_passes_strict_majority(self):
        self.assertTrue(evaluate_gate(games(), ["v0.0.1"], list(range(10)))["passed"])

    def test_each_opponent_must_pass_not_pooled(self):
        report = evaluate_gate(games(wins=20) + games("v0.0.2", 10),
                               ["v0.0.1", "v0.0.2"], list(range(10)))
        self.assertFalse(report["passed"])

    def test_missing_unknown_duplicate_results_fail_closed(self):
        for records in (games()[:-1], games() + [games()[0]],
                        games() + games("unknown")):
            with self.subTest(records=len(records)):
                self.assertFalse(evaluate_gate(records, ["v0.0.1"], list(range(10)))["passed"])
        records = games()
        records[0]["result"] = "unknown"
        self.assertFalse(evaluate_gate(records, ["v0.0.1"], list(range(10)))["passed"])

    def test_draws_and_timeouts_stay_in_denominator(self):
        for outcome in ("draw", "timeout"):
            records = games(wins=10)
            for record in records[10:]:
                record["result"] = outcome
            report = evaluate_gate(records, ["v0.0.1"], list(range(10)))
            self.assertFalse(report["passed"])
            self.assertEqual(report["opponents"]["v0.0.1"]["games"], 20)

    def test_errors_fail_even_with_twenty_wins(self):
        for error in ("order rejection", "process failure", "identity mismatch"):
            records = games(wins=20)
            records[0]["errors"] = [error]
            self.assertFalse(evaluate_gate(records, ["v0.0.1"], list(range(10)))["passed"])


def client(side, winner="Radiant", rejected=0, exit_code=0):
    return dict(exit=exit_code, stdout=(f"played 100 ticks as Some({side}); "
                f"winner Some({winner}); 10 decisions, 5 orders, {rejected} rejected orders\n"),
                wire=dict(slot=0 if side == "Radiant" else 1, winner=winner,
                          rejected=0, errors=[]))


class GameTests(unittest.TestCase):
    def test_replay_not_created_until_match_start_is_allowed(self):
        directory = Mock()
        directory.__truediv__ = Mock(return_value=Mock())
        replay = directory / "match.brp"
        replay.exists.return_value = False
        check_resources(directory, [])
        replay.stat.assert_not_called()

    def test_consistent_matchover_is_a_win(self):
        result, errors = validate_game([client("Radiant"), client("Dire")], 0, False, 1000)
        self.assertEqual(result, "win")
        self.assertEqual(errors, [])

    def test_rejection_process_identity_and_missing_summary_fail(self):
        for field, value, message in (("stdout", "", "missing or invalid outcome"),
                                      ("exit", 2, "client process failure")):
            clients = [client("Radiant"), client("Dire")]
            clients[0][field] = value
            self.assertIn(message, validate_game(clients, 0, False, 1000)[1])
        self.assertIn("order rejection", validate_game(
            [client("Radiant", rejected=1), client("Dire")], 0, False, 1000)[1])
        self.assertIn("identity mismatch", validate_game(
            [client("Dire"), client("Radiant")], 0, False, 1000)[1])

    def test_claimed_winner_without_wire_matchover_fails(self):
        clients = [client("Radiant"), client("Dire")]
        clients[0]["wire"]["winner"] = None
        self.assertIn("MatchOver mismatch", validate_game(clients, 0, False, 1000)[1])

    def test_wall_timeout_is_not_a_win(self):
        result, _ = validate_game([client("Radiant"), client("Dire")], 0, True, 1000)
        self.assertEqual(result, "timeout")


class WireTests(unittest.TestCase):
    def test_fragmented_welcome_and_matchover_are_observed(self):
        relay = Relay.__new__(Relay)
        relay.tick_limit = 10
        relay.buffer = bytearray()
        relay.frames = relay.bytes = 0
        relay.welcomed = threading.Event()
        relay.observed = dict(slot=None, winner=None, rejected=0)
        welcome = bytes([0, 1, 1, 0, 30, 1])
        over = bytes([7, 1, 5, 2] + [0] * 9 + [1] + [0] * 8)
        framed = struct.pack("<I", len(welcome)) + welcome + struct.pack("<I", len(over)) + over
        for byte in framed:
            relay.observe(bytes([byte]))
        self.assertEqual(relay.observed["slot"], 0)
        self.assertEqual(relay.observed["winner"], "Dire")
        self.assertTrue(relay.welcomed.is_set())

    def test_truncated_matchover_is_not_a_winner(self):
        relay = Relay.__new__(Relay)
        relay.tick_limit = 10
        relay.observed = dict(winner=None)
        with self.assertRaisesRegex(ValueError, "truncated postcard integer"):
            relay.observe_message(bytes([7, 1]))
        self.assertIsNone(relay.observed["winner"])

    def test_wire_rejection_is_observed_without_trusting_client_summary(self):
        relay = Relay.__new__(Relay)
        relay.observed = dict(rejected=0)
        relay.observe_message(bytes([5, 0, 0]))
        self.assertEqual(relay.observed["rejected"], 1)

    def test_unknown_message_and_oversized_frame_fail(self):
        relay = Relay.__new__(Relay)
        with self.assertRaisesRegex(ValueError, "unknown server message"):
            relay.observe_message(bytes([9]))
        relay.bytes = 0
        relay.buffer = bytearray()
        with self.assertRaisesRegex(ValueError, "invalid wire frame length"):
            relay.observe(struct.pack("<I", 4 * 1024 * 1024 + 1))

    def test_server_eof_half_closes_and_drains_final_client_ack(self):
        relay = Relay.__new__(Relay)
        relay.tick_limit = 10
        relay.stop = Mock()
        relay.stop.is_set.return_value = False
        relay.buffer = bytearray()
        relay.observed = dict(winner="Radiant")
        client_socket, server_socket = Mock(), Mock()
        server_socket.recv.return_value = b""
        client_socket.recv.side_effect = [b"final ack", b""]
        with patch("release_wire.select.select", side_effect=[
                ([server_socket], [], []), ([client_socket], [], []), ([client_socket], [], [])]):
            relay.pump(client_socket, server_socket)
        client_socket.shutdown.assert_called_once_with(socket.SHUT_WR)
        self.assertEqual(client_socket.recv.call_count, 2)
        server_socket.sendall.assert_not_called()

    def test_snapshot_beyond_tick_limit_is_rejected(self):
        relay = Relay.__new__(Relay)
        relay.tick_limit = 10
        with self.assertRaisesRegex(ValueError, "server exceeded tick limit"):
            relay.observe_message(bytes([3, 11, 1, 0]))

    def test_snapshot_wrong_viewer_fails_identity(self):
        relay = Relay.__new__(Relay)
        relay.tick_limit = 10
        relay.observed = dict(slot=0)
        with self.assertRaisesRegex(ValueError, "snapshot identity mismatch"):
            relay.observe_message(bytes([3, 1, 1, 1]))


class RegistryTests(unittest.TestCase):
    def test_registered_annotated_tag_identity_is_required(self):
        registry_path = Path(__file__).resolve().parents[1] / "releases.json"
        commit = "2cd104c8b8f0c5d1bed9988dfad4ddf4defd23f6"
        with patch("release_build.git", side_effect=["v0.0.1", "tag", commit, SIMULATOR]):
            registry = read_registry(registry_path, Path("bot"), Path("simulator"))
        self.assertEqual(registry["releases"][0]["commit"], commit)

    def test_unregistered_release_lightweight_tag_or_moved_tag_fails(self):
        registry_path = Path(__file__).resolve().parents[1] / "releases.json"
        for replies, message in ((["v0.0.1\nv0.0.2"], "release tags and registry differ"),
                                 (["v0.0.1", "commit"], "must be an annotated tag"),
                                 (["v0.0.1", "tag", "0" * 40], "commit identity mismatch")):
            with patch("release_build.git", side_effect=replies):
                with self.assertRaisesRegex(ValueError, message):
                    read_registry(registry_path, Path("bot"), Path("simulator"))


if __name__ == "__main__":
    unittest.main()
