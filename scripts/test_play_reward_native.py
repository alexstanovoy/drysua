"""Opt-in current-native fixtures; sole heavy ownership and resource guard required."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from play_admission import Admission
from play_match import Supervisor
from play_pacing import ACK_TIMEOUT_TICKS
from play_reward import start_observers


class NativeFixtureMixin:
    def paths(self):
        root = Path(__file__).resolve().parents[2]
        server = root / "bota/target/release/bota-server"
        expected = os.environ.get("DRYSUA_REWARD_SERVER_SHA256", "")
        self.assertEqual(len(expected), 64, "supply recorded current server SHA-256, not historical fixtures")
        with server.open("rb") as stream:
            self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), expected)
        return root, server

    @contextlib.contextmanager
    def directory(self, side):
        parent = os.environ.get("DRYSUA_REWARD_FIXTURE_DIRECTORY")
        if parent:
            self.assertTrue(Path(parent).is_dir(), "fixture evidence parent must already exist")
            directory = Path(tempfile.mkdtemp(prefix=f"native-{side}-", dir=parent))
            print(f"native reward TEST FIXTURE evidence: {directory}", flush=True)
            yield directory
        else:
            with tempfile.TemporaryDirectory(prefix=f"reward-native-{side}-") as temporary:
                yield Path(temporary)

    def match(self, root, directory, executable, side, *, complete):
        supervisor = Supervisor(directory)
        binary = root / "drysua/target/release/drysua"
        try:
            server = supervisor.spawn("server", [str(executable), "--port", "0", "--mode", "lockstep",
                "--ack-timeout-ticks", str(ACK_TIMEOUT_TICKS), "--players", "2", "--map", "2",
                "--seed", "9000001"], root)
            port = supervisor.wait_ready(server, 0)
            # Only full TEST FIXTURES are unpaced; normal --reward-report is always paced.
            supervisor.admission = Admission(port, side, mode=1, paced=not complete)
            start_observers(supervisor, binary, root, SimpleNamespace(reward_interval=300, opponent="teacher"))
            slot = 0 if side == "radiant" else 1
            ending = ["--complete"] if complete else ["--stop-tick", "60"]
            client = supervisor.spawn("client", [sys.executable, "-B", str(root / "drysua/scripts/play_reward_fixture.py"),
                "--addr", supervisor.admission.addresses["human"], "--slot", str(slot), *ending], root)
            supervisor.spawn("bot", [str(binary), "--addr", supervisor.admission.addresses["bot"],
                "--name", "drysua", "--policy", "teacher"], root)
            self.wait_fixture(supervisor, client, complete)
            self.assertEqual(client.exit_status(), 0)
            self.assertTrue(supervisor.admission.welcomed)
        finally:
            supervisor.close()
        with (directory / "client.log").open("rb") as stream:
            output = stream.read(1024 * 1024 + 1)
        self.assertLessEqual(len(output), 1024 * 1024, "fixture client log exceeds 1 MiB")
        result = json.loads(output)
        reports = [json.loads((directory / f"reward-{role}.json").read_text()) for role in ("human", "bot")]
        return result, reports

    def wait_fixture(self, supervisor, client, complete):
        deadline = time.monotonic() + (240 if complete else 20)
        for _ in range(240_000):
            status = client.exit_status()
            self.assertIn(status, (None, 0), f"fixture Player exited {status}; see {supervisor.directory}/client.log")
            for child in supervisor.children:
                if child.name in ("server", "bot"):
                    self.assertIn(child.exit_status(), (None, 0),
                                  f"native fixture {child.name} failed; see {supervisor.directory}/{child.name}.log")
            if status is not None:
                if not complete or (all(relay.match_over for relay in supervisor.admission.relays)
                        and all(child.exit_status() is not None for _, child, _ in supervisor.reward_pipes)):
                    return
            self.assertLess(time.monotonic(), deadline, "native fixture exceeded bounded deadline")
            supervisor.pump(0.001 if complete else 0.005)
        self.fail("native fixture pump count exceeded")


@unittest.skipUnless(os.environ.get("DRYSUA_REWARD_NATIVE") == "1",
                     "paced short native fixtures require explicit heavy ownership and resource guard")
class CurrentNativeRewardTests(NativeFixtureMixin, unittest.TestCase):
    def test_both_human_seats_keep_paced_60_tick_cut_incomplete(self):
        root, server = self.paths()
        for side in ("radiant", "dire"):
            with self.subTest(side=side), self.directory(side) as directory:
                result, reports = self.match(root, directory, server, side, complete=False)
                self.assertGreaterEqual(result["elapsed_snapshot_seconds"], 1.9)
                self.assertLess(result["elapsed_snapshot_seconds"], 10)
                self.assertFalse(result["terminal"])
                for report in reports:
                    self.assertFalse(report["complete"])
                    self.assertFalse(report["valid"])
                    self.assertEqual(report["ticks"], 60)
                    self.assertEqual(report["components"]["terminal"], 0)
                    self.assertIsNone(report["outcome"])


@unittest.skipUnless(os.environ.get("DRYSUA_REWARD_NATIVE_FULL") == "1",
                     "full native TEST FIXTURES require separate explicit heavy ownership and resource guard")
class CompleteNativeRewardTests(NativeFixtureMixin, unittest.TestCase):
    def test_radiant_passive_player_genuine_terminal_reports_both_seats(self):
        self.complete_match("radiant")

    def test_dire_passive_player_genuine_terminal_reports_both_seats(self):
        self.complete_match("dire")

    def complete_match(self, side):
        root, server = self.paths()
        with self.directory(side) as directory:
            result, reports = self.match(root, directory, server, side, complete=True)
            self.assertTrue(result["terminal"], "manual cut is not a genuine native end")
            self.assertEqual(result["snapshots"], result["events"])
            self.assertGreater(result["ticks"], 0)
            self.assertLessEqual(result["ticks"], 27900)
            self.assertEqual({report["slot"] for report in reports}, {0, 1})
            for report in reports:
                self.assertTrue(report["valid"], report)
                self.assertTrue(report["complete"], report)
                self.assertEqual(report["ticks"], result["ticks"])
                self.assertEqual(report["reward_ticks"], result["ticks"] - 1)
                self.assertFalse(report["pending_events"])
                self.assertEqual(len(report["components"]), 17)
                self.assertEqual(report["profile_version"], 6)
                self.assertEqual(report["profile_hash"], "1084583101075978392")
                self.assertEqual(report["team"], ("Radiant", "Dire")[report["slot"]])
                outcome = "Draw" if result["winner"] == 2 else (
                    "Win" if report["slot"] == result["winner"] else "Loss")
                self.assertEqual(report["outcome"], outcome)
                self.assertEqual(report["components"]["terminal"], {"Draw": 0, "Win": 0.2, "Loss": -0.2}[outcome])
                self.assertAlmostEqual(report["total"], sum(report["components"].values()), places=10)
                self.assertAlmostEqual(report["total_without_terminal"],
                                       report["total"] - report["components"]["terminal"], places=10)
            self.assertFalse(list(directory.glob("*.invalid.json")))


if __name__ == "__main__":
    unittest.main()
