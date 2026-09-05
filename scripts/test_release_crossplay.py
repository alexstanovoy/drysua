"""Release gate regression tests (stdlib only)."""

import json
import io
import unittest
from unittest.mock import Mock, patch
import socket
import struct
import threading
import tempfile
import argparse
from pathlib import Path

from release_build import read_registry, SIMULATOR

from release_crossplay import check_resources, evaluate_gate, validate_game
from release_wire import Relay
import release_crossplay as crossplay
import release_build as release_build


class CandidateTests(unittest.TestCase):
    def test_tactical_cli_requires_weights_and_emits_explicit_tactical_command(self):
        parser = argparse.ArgumentParser()
        crossplay.add_candidate_arguments(parser)
        args = parser.parse_args(["--candidate-policy", "tactical"])
        with self.assertRaisesRegex(ValueError, "tactical requires --candidate-weights"):
            crossplay.validate_candidate_arguments(args)
        args.candidate_weights = Path("/weights")
        crossplay.validate_candidate_arguments(args)
        bot = dict(binary=Path("/bot"), policy="tactical", weights=args.candidate_weights)
        command = crossplay.bot_command(bot, "127.0.0.1:1", 0, 30000)
        self.assertEqual(command[2:4], ["--policy", "tactical"])
        self.assertEqual(command[-2:], ["--weights-directory", "/weights"])

    def test_tactical_snapshot_selects_canonical_file_and_isolates_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            weights = source / "drysua.tactical.bin"
            weights.write_bytes(b"tactical deployment")
            (source / release_build.WEIGHTS_NAME).write_bytes(b"unrelated hybrid weights")
            metadata = release_build.snapshot_weights(source, source / "snapshot", policy="tactical")
            self.assertEqual(metadata["sha256"], crossplay.digest(weights))
            self.assertEqual(Path(metadata["snapshot"]).name, "drysua.tactical.bin")
            weights.write_bytes(b"trained replacement")
            self.assertEqual(Path(metadata["snapshot"]).read_bytes(), b"tactical deployment")
            self.assertFalse((source / "snapshot" / release_build.WEIGHTS_NAME).exists())

    def test_tactical_snapshot_rejects_missing_empty_link_and_large_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            weights = source / "drysua.tactical.bin"
            (source / release_build.WEIGHTS_NAME).write_bytes(b"not a tactical fallback")
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                release_build.snapshot_weights(source, source / "missing", policy="tactical")
            for value in (b"", bytes(16385)):
                weights.write_bytes(value)
                with self.assertRaisesRegex(ValueError, "weights size"):
                    release_build.snapshot_weights(source, source / "invalid", policy="tactical")
            weights.unlink()
            weights.symlink_to(source / release_build.WEIGHTS_NAME)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                release_build.snapshot_weights(source, source / "link", policy="tactical")

    def test_tactical_registry_requires_version_bound_sha_weights(self):
        release = dict(tag="v0.0.4", policy="tactical")
        with self.assertRaisesRegex(ValueError, "tactical requires weights"):
            release_build.validate_release_weights(release)
        release["weights"] = dict(path="artifacts/v0.0.4", sha256="a" * 64)
        release_build.validate_release_weights(release)
        release["weights"]["path"] = "artifacts/v0.0.4/../temp"
        with self.assertRaisesRegex(ValueError, "weights path"):
            release_build.validate_release_weights(release)

    def test_cli_defaults_teacher_and_requires_explicit_hybrid_weights(self):
        parser = argparse.ArgumentParser()
        crossplay.add_candidate_arguments(parser)
        self.assertEqual(parser.parse_args([]).candidate_policy, "teacher")
        for arguments, message in (([], None), (["--candidate-policy", "hybrid"],
                "hybrid requires --candidate-weights"),
                (["--candidate-weights", "."], "teacher forbids --candidate-weights")):
            args = parser.parse_args(arguments)
            if message:
                with self.assertRaisesRegex(ValueError, message):
                    crossplay.validate_candidate_arguments(args)
            else:
                crossplay.validate_candidate_arguments(args)
        args = parser.parse_args(["--candidate-policy", "hybrid", "--candidate-weights", "."])
        crossplay.validate_candidate_arguments(args)
        with patch("sys.stderr", new_callable=io.StringIO) as errors, self.assertRaises(SystemExit):
            parser.parse_args(["--candidate-policy", "network"])
        self.assertIn("invalid choice: 'network'", errors.getvalue())

    def test_snapshot_isolated_and_sha_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            weights = source / release_build.WEIGHTS_NAME
            weights.write_bytes(b"accepted runtime weights")
            target = Path(temporary) / "snapshot"
            metadata = release_build.snapshot_weights(source, target)
            self.assertEqual(metadata["sha256"], crossplay.digest(target / weights.name))
            weights.write_bytes(b"later training output")
            self.assertEqual((target / weights.name).read_bytes(), b"accepted runtime weights")
            self.assertNotEqual(metadata["sha256"], crossplay.digest(weights))

    def test_missing_empty_symlink_and_oversized_weights_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            weights = source / release_build.WEIGHTS_NAME
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                release_build.snapshot_weights(source, source / "missing")
            weights.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "weights size"):
                release_build.snapshot_weights(source, source / "empty")
            weights.unlink()
            weights.symlink_to("absent")
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                release_build.snapshot_weights(source, source / "link")
            weights.unlink()
            weights.write_bytes(b"12")
            with patch.object(release_build, "MAX_WEIGHTS_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "weights size"):
                    release_build.snapshot_weights(source, source / "large")

    def test_wrong_digest_and_copy_race_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / release_build.WEIGHTS_NAME).write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "release weights SHA256 mismatch"):
                release_build.snapshot_weights(source, source / "wrong", "0" * 64)
            with patch.object(release_build, "digest", side_effect=["a", "b"]):
                with self.assertRaisesRegex(ValueError, "weights changed while being snapshotted"):
                    release_build.snapshot_weights(source, source / "race")

    def test_run_wires_snapshot_metadata_and_does_not_change_gate(self):
        self.run_wires_snapshot("hybrid", release_build.WEIGHTS_NAME)

    def test_tactical_run_binds_canonical_snapshot_and_keeps_complete_schedule(self):
        self.run_wires_snapshot("tactical", "drysua.tactical.bin")

    def run_wires_snapshot(self, policy, filename):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(__file__).resolve().parents[1]
            output = Path(temporary)
            binary = output / "input-binary"
            binary.write_bytes(b"binary")
            weights = output / filename
            weights.write_bytes(b"weights")
            args = argparse.Namespace(candidate_policy=policy, candidate_weights=output,
                candidate_binary=binary, candidate_metadata="test provenance", bota_repository=root)
            registry = dict(seeds=list(range(10)))
            teacher = dict(binary=binary, policy="teacher")
            with patch.object(crossplay, "read_registry", return_value=registry), \
                    patch.object(crossplay, "prepare", return_value=(binary, {"v0.0.1": teacher})), \
                    patch.object(crossplay, "execute_game", return_value=dict(result="draw", errors=[])) as execute, \
                    patch("builtins.print"):
                status = crossplay.run(args, root, output)
            self.assertEqual(status, 1)
            self.assertEqual(execute.call_count, 20)
            for call in execute.call_args_list:
                bots = call.args[2]
                candidate = next(bot for bot in bots if bot["policy"] == policy)
                self.assertEqual(candidate["weights"], output / "candidate-weights")
                self.assertIn(teacher, bots)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["candidate"]["weights"]["sha256"], crossplay.digest(weights))
            self.assertEqual(report["candidate"]["policy"], policy)
            self.assertEqual(report["gate"]["opponents"]["v0.0.1"]["games"], 20)

    def test_commands_keep_hybrid_weights_and_teacher_separate_in_either_seat(self):
        hybrid = dict(binary=Path("/candidate"), policy="hybrid", weights=Path("/snapshot"))
        teacher = dict(binary=Path("/historical"), policy="teacher")
        for bots in ([hybrid, teacher], [teacher, hybrid]):
            for index, bot in enumerate(bots):
                command = crossplay.bot_command(bot, "127.0.0.1:1234", index, 30000)
                self.assertEqual(command[command.index("--policy") + 1], bot["policy"])
                self.assertEqual("--weights-directory" in command, bot is hybrid)
                if bot is hybrid:
                    self.assertEqual(command[-2:], ["--weights-directory", "/snapshot"])

    def test_bot_command_rejects_missing_or_ambiguous_weights(self):
        for bot in (dict(binary="bot", policy="hybrid"),
                    dict(binary="bot", policy="teacher", weights="ignored"),
                    dict(binary="bot", policy="unknown")):
            with self.assertRaisesRegex(ValueError, "bot policy/weights contract mismatch"):
                crossplay.bot_command(bot, "127.0.0.1:1234", 0, 30000)

    def test_changed_snapshot_invalidates_otherwise_passing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(__file__).resolve().parents[1]
            output = Path(temporary)
            binary = output / "input-binary"
            binary.write_bytes(b"binary")
            (output / release_build.WEIGHTS_NAME).write_bytes(b"weights")
            args = argparse.Namespace(candidate_policy="hybrid", candidate_weights=output,
                candidate_binary=binary, candidate_metadata="test", bota_repository=root)

            def game(directory, server, bots, seed, registry):
                snapshot = output / "candidate-weights" / release_build.WEIGHTS_NAME
                snapshot.unlink()
                snapshot.write_bytes(b"changed")
                return dict(result="win" if bots[0]["policy"] == "hybrid" else "loss", errors=[])

            with patch.object(crossplay, "read_registry", return_value=dict(seeds=list(range(10)))), \
                    patch.object(crossplay, "prepare", return_value=(binary, {
                        "v0.0.1": dict(binary=binary, policy="teacher")})), \
                    patch.object(crossplay, "execute_game", side_effect=game), patch("builtins.print"):
                status = crossplay.run(args, root, output)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["gate"]["opponents"]["v0.0.1"]["wins"], 20)
            self.assertEqual(status, 1)
            self.assertEqual(report["gate"]["errors"], ["binary or weights identity changed during run"])

    def test_future_release_weights_contract_rejects_ambiguity_and_unsafe_paths(self):
        teacher = dict(tag="v0.0.4", policy="teacher")
        release_build.validate_release_weights(teacher)
        with self.assertRaisesRegex(ValueError, "teacher forbids weights"):
            release_build.validate_release_weights(dict(teacher, weights={}))
        hybrid = dict(tag="v0.0.4", policy="hybrid")
        with self.assertRaisesRegex(ValueError, "hybrid requires weights"):
            release_build.validate_release_weights(hybrid)
        for path in ("/artifacts/v0.0.4", "artifacts/temp", "artifacts/v0.0.4/../escape",
                     "artifacts/v0.0.3", "artifacts/v0.0.4/subdir"):
            with self.assertRaisesRegex(ValueError, "weights path"):
                release_build.validate_release_weights(dict(hybrid, weights=dict(path=path, sha256="a" * 64)))
        valid = dict(hybrid, weights=dict(path="artifacts/v0.0.4", sha256="a" * 64))
        release_build.validate_release_weights(valid)
        valid["weights"]["sha256"] = "not a digest"
        with self.assertRaisesRegex(ValueError, "weights SHA256"):
            release_build.validate_release_weights(valid)


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
    def registry_path(self):
        release = dict(tag="v0.0.1", commit="2cd104c8b8f0c5d1bed9988dfad4ddf4defd23f6",
                       policy="teacher", simulator_commit=SIMULATOR, map=1)
        registry = dict(schema_version=1, simulator_commit=SIMULATOR, map=1,
                        policy="teacher", tick_limit=30000, process_timeout_seconds=90,
                        gate="candidate_wins * 2 > all_scheduled_games_per_opponent",
                        seeds=list(range(10)), releases=[release])
        return Mock(read_text=Mock(return_value=json.dumps(registry)))

    def test_registered_annotated_tag_identity_is_required(self):
        registry_path = self.registry_path()
        commit = "2cd104c8b8f0c5d1bed9988dfad4ddf4defd23f6"
        with patch("release_build.git", side_effect=["v0.0.1", "tag", commit, SIMULATOR]):
            registry = read_registry(registry_path, Path("bot"), Path("simulator"))
        self.assertEqual(registry["releases"][0]["commit"], commit)

    def test_hybrid_registry_keeps_annotated_source_and_simulator_requirements(self):
        registry = json.loads(self.registry_path().read_text())
        release = registry["releases"][0]
        release.update(policy="hybrid", weights=dict(path="artifacts/v0.0.1", sha256="a" * 64))
        path = Mock(read_text=Mock(return_value=json.dumps(registry)))
        with patch("release_build.git", side_effect=["v0.0.1", "tag", release["commit"], SIMULATOR]):
            self.assertEqual(read_registry(path, Path("bot"), Path("simulator")), registry)

    def test_hybrid_prepare_uses_weights_from_immutable_tag_archive(self):
        self.prepare_uses_archived_weights("hybrid")

    def test_tactical_prepare_uses_weights_from_immutable_tag_archive(self):
        self.prepare_uses_archived_weights("tactical")

    def prepare_uses_archived_weights(self, policy):
        registry = json.loads(self.registry_path().read_text())
        release = registry["releases"][0]
        release.update(policy=policy, weights=dict(path="artifacts/v0.0.1", sha256="a" * 64))
        with patch.object(release_build, "archive") as archive, \
                patch.object(release_build, "build"), \
                patch.object(release_build, "snapshot_weights", return_value={}) as snapshot:
            _, opponents = release_build.prepare(Path("bot"), Path("simulator"), Path("run"), registry)
        archive.assert_any_call(Path("bot"), release["commit"], Path("run/sources/v0.0.1"), Path("run"))
        snapshot.assert_called_once_with(Path("run/sources/v0.0.1/artifacts/v0.0.1"),
                                         Path("run/weights-v0.0.1"), "a" * 64, policy=policy)
        self.assertEqual(opponents["v0.0.1"]["policy"], policy)

    def test_unregistered_release_lightweight_tag_or_moved_tag_fails(self):
        registry_path = self.registry_path()
        for replies, message in ((["v0.0.1\nv0.0.2"], "release tags and registry differ"),
                                 (["v0.0.1", "commit"], "must be an annotated tag"),
                                 (["v0.0.1", "tag", "0" * 40], "commit identity mismatch")):
            with patch("release_build.git", side_effect=replies):
                with self.assertRaisesRegex(ValueError, message):
                    read_registry(registry_path, Path("bot"), Path("simulator"))


if __name__ == "__main__":
    unittest.main()
