"""Campaign controller contracts against a fake rootless Docker whose container is a fake trainer.

The controller runs in-process; only the host cgroup and health probes are
replaced, because a fake container has no cgroup of its own. The real probes
are exercised by the documented smoke run.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(REPOSITORY / "scripts"))
import train  # noqa: E402
import train_report  # noqa: E402
import train_session  # noqa: E402

IMAGE = "docker.io/library/ubuntu:26.04@sha256:" + "1" * 64
GIB = 1024 ** 3
HEALTHY = {"memory_available": 40 * GIB, "disk_free": 500 * GIB, "cpu_celsius": 50.0}


def cgroup(state, pid, memory_gib=12):
    (state / f"cgroup-read-{pid}").touch()
    return {"memory.max": str(memory_gib * GIB), "memory.swap.max": "0", "pids.max": "1024", "cpu.max": "max 100000",
            "memory.current": "1000", "memory.peak": "2000", "pids.current": "12", "cpu_usage_usec": 5,
            "memory.events.max": 0, "memory.events.oom": 0, "memory.events.oom_kill": 0, "pids.events.max": 0,
            "uid_map": ["0", str(os.getuid()), "1"]}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.docker = self.root / "docker-bin"
        (self.docker / "state").mkdir(parents=True)
        self.executable(self.docker / "docker", FIXTURES / "fake_docker.py")
        self.inspector = self.executable(self.root / "inspector", FIXTURES / "fake_inspector.py")
        self.trainer = self.root / "trainer"
        self.trainer.write_bytes(b"\x7fELF" + bytes(60))
        self.trainer.chmod(0o700)
        self.lock = self.root / "heavy.lock"
        self.lock.touch()
        self.campaign = self.root / "campaign"
        state = self.docker / "state"
        patches = [patch.dict(os.environ, {"PATH": f"{self.docker}:{os.environ['PATH']}"}),
                   patch.object(train_session, "read_cgroup", side_effect=lambda pid: cgroup(state, pid)),
                   patch.object(train_session, "host_sample", side_effect=lambda config, directory: dict(HEALTHY)),
                   patch.object(train_session, "POLL_SECONDS", 0.02)]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        self.scenario()

    def executable(self, path, source):
        path.write_text(f"#!{sys.executable}\n" + source.read_text())
        path.chmod(0o700)
        return path

    def scenario(self, **values):
        values.update(repository=str(REPOSITORY), campaign=str(self.campaign), controller_pid=os.getpid())
        (self.docker / "state" / "scenario.json").write_text(json.dumps(values))

    def create(self, **overrides):
        config = {"schema": 2, "trainer": str(self.trainer), "inspector": str(self.inspector), "total_updates": 5,
                  "history_every": 2, "image": IMAGE, "lock_paths": [str(self.lock)],
                  "training_args": ["--slots", "2", "--samples-per-update", "64"]}
        config.update(overrides)
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return train.create(path, self.campaign)

    def creates(self):
        return json.loads((self.docker / "state" / "creates.json").read_text())

    def receipt(self, session):
        return json.loads((self.campaign / "sessions" / f"{session:04d}" / "receipt.json").read_text())

    def test_run_trains_every_update_in_one_container_with_history_and_a_receipt(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / train.RUNTIME_FILE).write_bytes(b"frozen opponent")
        self.create(opponents=[{"kind": "teacher", "weight": "2"}, {"kind": "weights", "path": str(snapshot),
                                                                     "weight": "0.5"}, {"kind": "league"}])
        self.scenario(checkpoint_every=3)
        result = train.run_campaign(self.campaign)
        self.assertEqual((result["phase"], result["updates"], result["sessions"]), ("completed", 5, 1))
        [create] = self.creates()
        command = create[create.index("--") + 1:]
        self.assertEqual(command[:2], [str(self.campaign / "bin/trainer"), "train-annealed"])
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--checkpoint-interval-seconds") + 1], "600")
        opponents = [command[index + 1] for index, token in enumerate(command) if token == "--opponent"]
        self.assertEqual(opponents, ["teacher:2", f"weights:{self.campaign / 'inputs/opponent-1'}:0.5", "league:1"])
        self.assertEqual((self.campaign / "inputs/opponent-1" / train.RUNTIME_FILE).read_bytes(), b"frozen opponent")
        self.assertIn(f"CUDA_CACHE_PATH={self.campaign / 'cuda-cache'}", create)
        self.assertIn(f"type=bind,src={self.campaign / 'bin'},dst={self.campaign / 'bin'},readonly", create)
        self.assertEqual(sorted(path.name for path in (self.campaign / "history").iterdir()),
                         ["u0003", "u0005"])  # Milestones exist only at checkpoints.
        receipt = self.receipt(1)
        self.assertEqual((receipt["start_updates"], receipt["end_updates"]), (0, 5))
        self.assertTrue(receipt["outcome"]["verified_limits"])
        self.assertEqual(json.loads((self.docker / "state" / "containers.json").read_text()), {})
        self.assertFalse((self.campaign / "owner.json").exists())

    def test_eval_rates_each_snapshot_once_and_the_dashboard_shows_the_rating_curve(self):
        self.create()
        self.scenario(checkpoint_every=3)
        train.run_campaign(self.campaign)
        drysua = self.executable(self.root / "drysua", FIXTURES / "fake_eval.py")

        def evaluate():
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                train.main(["eval", str(self.campaign), "--drysua", str(drysua), "--seeds", "1:4", "--average", "2"])
            return json.loads(output.getvalue())

        first = evaluate()
        self.assertEqual(first["candidates"], ["u0003", "u0005", "u0005-avg2"])
        self.assertEqual((first["new_runs"], first["store"]), (3, str(self.campaign / "eval")))
        self.assertEqual(evaluate()["new_runs"], 0)
        html = self.root / "dashboard.html"
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["report", str(self.campaign), "--html", str(html)])
        embedded = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>',
                             html.read_text(), re.S)
        [section] = [section for section in json.loads(embedded.group(1))["sections"]
                     if section["title"].startswith("Frozen pool evaluation")]
        charts = {chart["title"]: chart for chart in section["charts"]}
        rating = charts["Pool Elo (anchor teacher = 0)"]
        self.assertEqual(rating["x"], [3, 5])
        self.assertEqual([entry["name"] for entry in rating["series"]], ["snapshot", "avg2"])
        self.assertIsNone(rating["series"][1]["values"][0])
        robustness = [entry["name"] for entry in charts["Robustness: worst case and held-out"]["series"]]
        self.assertEqual(robustness, ["worst case", "held-out", "train"])

    def test_pause_commits_the_in_flight_update_and_resume_continues_in_a_new_container(self):
        self.create()
        self.scenario(pause_at=2)
        paused = train.run_campaign(self.campaign)
        self.assertEqual((paused["phase"], paused["updates"], paused["last_error"]), ("paused", 2, None))
        self.scenario()
        completed = train.run_campaign(self.campaign, resume=True)
        self.assertEqual((completed["phase"], completed["updates"], completed["sessions"]), ("completed", 5, 2))
        self.assertIn("--resume", self.creates()[1])
        self.assertEqual((self.receipt(2)["start_updates"], self.receipt(2)["end_updates"]), (2, 5))
        report, _ = train_report.build_report(self.campaign)
        self.assertEqual((report["last_update"], report["games"], report["processes"]), (5, 10, 2))

    def test_immediate_stop_abandons_the_in_flight_update_and_keeps_the_last_commit(self):
        self.create()
        self.scenario(stop_at=3)
        result = train.run_campaign(self.campaign)
        self.assertEqual((result["phase"], result["updates"]), ("paused", 2))
        self.assertEqual(self.receipt(1)["outcome"]["request"], "stop")
        self.assertEqual(train_report.build_report(self.campaign)[0]["last_update"], 2)

    def test_controller_signal_is_a_graceful_pause(self):
        self.create()
        self.scenario(signal_controller_at=1)
        result = train.run_campaign(self.campaign)
        self.assertEqual((result["phase"], result["updates"]), ("paused", 1))

    def test_trainer_crash_fails_the_session_without_retry_and_resume_is_explicit(self):
        self.create(checkpoint_seconds=60)
        self.scenario(crash_at=5, checkpoint_every=3)
        failed = train.run_campaign(self.campaign)
        self.assertEqual((failed["phase"], failed["updates"]), ("failed", 3))
        command = self.creates()[0]
        self.assertEqual(command[command.index("--checkpoint-interval-seconds") + 1], "60")
        self.assertEqual(failed["last_error"], "trainer exited with code 3 (oom_killed=False)")
        self.assertEqual(len(self.creates()), 1)
        with self.assertRaisesRegex(ValueError, r"run requires phase \['prepared'\], campaign is failed"):
            train.run_campaign(self.campaign)
        self.scenario()
        self.assertEqual(train.run_campaign(self.campaign, resume=True)["phase"], "completed")
        # Updates 4 and 5 of the crashed process were never durable; the resume replays them once.
        report, _ = train_report.build_report(self.campaign)
        self.assertEqual((report["last_update"], report["games"], report["updates"]), (5, 10, 5))

    def test_host_health_violation_stops_gracefully_with_the_reason(self):
        self.create()
        samples = iter([dict(HEALTHY)] + [dict(HEALTHY, memory_available=GIB)] * 1000)
        with patch.object(train_session, "host_sample", side_effect=lambda *_: next(samples)), \
                patch.object(train_session, "SAMPLE_SECONDS", 0):
            self.scenario(update_seconds=0.2)
            result = train.run_campaign(self.campaign)
        self.assertEqual(result["phase"], "paused")
        self.assertEqual(result["last_error"], "host MemAvailable below 16 GiB")
        self.assertLess(result["updates"], 5)

    def test_session_deadline_stops_gracefully(self):
        self.create(max_seconds=1, total_updates=50)
        self.scenario(update_seconds=0.2)
        result = train.run_campaign(self.campaign)
        self.assertEqual((result["phase"], result["last_error"]), ("paused", "session deadline reached"))
        self.assertTrue(0 < result["updates"] < 50)

    def test_wrong_actual_cgroup_limits_kill_the_container_and_fail_the_session(self):
        self.create()
        self.scenario(update_seconds=0.2)
        wrong = lambda pid: cgroup(self.docker / "state", pid, memory_gib=16)  # noqa: E731
        with patch.object(train_session, "read_cgroup", side_effect=wrong):
            result = train.run_campaign(self.campaign)
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(result["last_error"], "actual container cgroup limits were never verified")
        self.assertEqual(self.receipt(1)["outcome"]["reason"],
                         f"container verification failed: container cgroup memory.max is '{16 * GIB}', "
                         f"expected {12 * GIB}")

    def test_recover_removes_the_orphaned_container_of_a_dead_controller(self):
        self.create()
        self.scenario(pause_at=2)
        train.run_campaign(self.campaign)
        dead = subprocess.Popen(["true"])
        dead.wait()
        identity = {"pid": dead.pid, "start": "0", "uid": os.getuid(), "boot": "gone"}
        container = train_session.create_container(self.campaign, train.load_campaign(self.campaign)[2]["config"],
                                                   train.status(self.campaign)["campaign_id"], 2, ["true"], None)
        owner = dict(identity, token="f" * 32, session=2, container_id=container)
        (self.campaign / "sessions" / "0002").mkdir()
        (self.campaign / "owner.json").write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "a previous session was not closed; run recover"):
            train.run_campaign(self.campaign, resume=True)
        with self.assertRaisesRegex(ValueError, "recover requires explicit --confirm-offline"):
            train.recover(self.campaign)
        recovered = train.recover(self.campaign, confirm=True)
        self.assertEqual((recovered["phase"], recovered["updates"]), ("paused", 2))
        self.assertEqual(json.loads((self.docker / "state" / "containers.json").read_text()), {})
        self.assertEqual(self.receipt(2)["outcome"]["reason"], "recovered after controller loss")

    def test_control_requests_need_a_live_owner(self):
        self.create()
        for operation in ("pause", "stop"):
            with self.assertRaisesRegex(ValueError, "no live controller owns this campaign"):
                train.request_control(self.campaign, operation)

    def test_detach_reports_the_startup_error_of_the_frozen_controller(self):
        self.create()
        self.lock.unlink()
        with self.assertRaisesRegex(ValueError, "detached controller did not start: error: .*heavy.lock"):
            train.detach(self.campaign)
        self.assertEqual(train.status(self.campaign)["phase"], "prepared")

    def test_frozen_inputs_are_verified_and_never_executed_by_status(self):
        self.create()
        frozen = self.campaign / "bin" / "trainer"
        frozen.chmod(0o700)
        frozen.write_bytes(b"\x7fELF changed")
        frozen.chmod(0o500)
        with self.assertRaisesRegex(ValueError, "frozen artifact integrity mismatch: bin/trainer"):
            train.status(self.campaign)

    def test_changed_controller_source_must_use_the_frozen_snapshot(self):
        self.create()
        frozen = self.campaign / "frozen" / "train.py"
        with patch.object(train, "SOURCE_DIRECTORY", self.root):
            for name in train.SOURCES:
                shutil.copy(REPOSITORY / "scripts" / name, self.root / name)
            (self.root / "train.py").write_text(frozen.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(ValueError, "controller source differs from the frozen snapshot"):
                train.run_campaign(self.campaign)
        self.assertEqual(train.status(self.campaign)["phase"], "prepared")

    def test_invalid_configurations_are_rejected_before_creation(self):
        for overrides, message in (
                ({"schema": 1}, "unsupported configuration schema; expected 2"),
                ({"invocation_updates": 1}, "unknown configuration fields: \\['invocation_updates'\\]"),
                ({"image": "ubuntu:26.04"}, "image must be pinned by sha256 digest"),
                ({"training_args": ["--updates", "9"]}, "controller-owned or unreviewed argument: --updates"),
                ({"training_args": ["--resume"]}, "controller-owned or unreviewed argument: --resume"),
                ({"mode": "gpu"}, "gpu_uuid must be a full GPU UUID"),
                ({"vram_budget_mib": 4096}, "vram_budget_mib needs gpu mode"),
                ({"stop_seconds": 1}, "stop_seconds must be an integer in 5..3600"),
                ({"checkpoint_seconds": 59}, "checkpoint_seconds must be an integer in 60..86400"),
                ({"training_args": ["--checkpoint-interval-seconds", "60"]},
                 "controller-owned or unreviewed argument: --checkpoint-interval-seconds"),
                ({"opponents": [{"kind": "weights", "weight": "1"}]}, "opponent fields must be"),
                ({"opponents": [{"kind": "teacher", "weight": 1}]}, "opponent weight must be a decimal string")):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                self.create(**overrides)
            self.assertFalse(self.campaign.exists())

    def test_trainer_must_be_native_code_not_a_script(self):
        self.trainer.write_text("#!/bin/sh\n")
        with self.assertRaisesRegex(ValueError, "trainer must be a native ELF executable"):
            self.create()

    def test_example_configuration_is_valid(self):
        value = json.loads((REPOSITORY / "docs" / "training_controller.example.json").read_text())
        self.assertEqual(train.validate_config(value)["total_updates"], value["total_updates"])

    def test_campaign_is_private_and_status_rejects_corruption(self):
        self.create()
        self.assertEqual(stat.S_IMODE(self.campaign.stat().st_mode), 0o700)
        status_path = self.campaign / "status.json"
        value = json.loads(status_path.read_text())
        status_path.write_text(json.dumps(dict(value, schema="drysua-training-campaign/v1")))
        with self.assertRaisesRegex(ValueError, "schema 2 required"):
            train.status(self.campaign)


if __name__ == "__main__":
    unittest.main()
