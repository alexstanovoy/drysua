"""Small offline fixtures; run only in an approved non-training window."""

import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runtime_guard as guard


class RuntimeGuardTests(unittest.TestCase):
    def test_cgroup_accepts_hard_twelve_gib_without_cpu_quota(self):
        limits = {"memory.max": str(12 * guard.GIB), "memory.swap.max": "0",
                  "pids.max": "1024", "cpu.max": "max 100000"}
        guard.validate_limits(limits)

    def test_cgroup_rejects_unlimited_memory_swap_pids_and_cpu_quota(self):
        valid = {"memory.max": str(12 * guard.GIB), "memory.swap.max": "0",
                 "pids.max": "1024", "cpu.max": "max 100000"}
        for name, value in [("memory.max", "max"), ("memory.max", str(16 * guard.GIB)),
                            ("memory.swap.max", "1"), ("pids.max", "max"),
                            ("cpu.max", "90000 100000")]:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(guard.Refusal, name):
                    guard.validate_limits(dict(valid, **{name: value}))

    def test_resource_thresholds_accept_equality_reject_crossing_and_missing_gpu(self):
        guard.validate_sample(24 * guard.GIB, 4096, 90, 85, admission=True)
        guard.validate_sample(16 * guard.GIB, 4096, None, 85, admission=False)
        cases = [(16 * guard.GIB - 1, 4096, 90, 85, "MemAvailable"),
                 (16 * guard.GIB, 4095, 90, 85, "VRAM"),
                 (16 * guard.GIB, 4096, 91, 85, "CPU"),
                 (16 * guard.GIB, 4096, 90, 86, "GPU"),
                 (16 * guard.GIB, None, 90, 85, "GPU telemetry")]
        for memory, free, cpu, gpu, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(guard.Refusal, message):
                    guard.validate_sample(memory, free, cpu, gpu, admission=False)

    def test_event_delta_fails_even_if_counter_is_not_oom_kill(self):
        guard.validate_events({"memory.high": 2}, {"memory.high": 2})
        with self.assertRaisesRegex(guard.Refusal, "memory.high"):
            guard.validate_events({"memory.high": 2}, {"memory.high": 3})
        with self.assertRaisesRegex(guard.Refusal, "event keys"):
            guard.validate_events({"pids.max": 0}, {})

    def test_checkpoint_scalar_is_bounded_and_does_not_claim_tensor_validation(self):
        header = b"DRYCKP18" + struct.pack("<IQ", 13, 0x9D251E7C6725BB43)
        linked = [(5, 0x93EA35FD652475A7), (22, 0x92723C71B52788B0),
                  (24, 0xA799157E02A95F4E), (38, 0xEAC28E84438D6A75),
                  (7, 0x64F4C97D0C52D062)]
        data = header + b"".join(struct.pack("<IQ", *item) for item in linked)
        data += b"".join(struct.pack("<H", len(text)) + text
                         for text in [b"source", b"simulator", b"builtin,cuda", b"command"])
        data += struct.pack("<QHHBIIIB", 9001, 2, 2, 1, 0, 2048, 32, 0)
        data += struct.pack("<QQQ", 180, 180, 180)
        self.assertEqual(guard.checkpoint_update(data), 180)
        for invalid in [data[:79], data[:-1], b"wrong" + data[5:], data + bytes(65536)]:
            with self.assertRaisesRegex(guard.Refusal, "checkpoint"):
                guard.checkpoint_update(invalid)

    def test_native_argv_keeps_total_target_and_never_relabels_resume(self):
        with patch.dict("os.environ", {}, clear=True):
            config = guard.run_config()
        command = guard.native_command(config, resume=True)
        self.assertEqual(command[command.index("--updates") + 1], "1000")
        self.assertEqual(command[command.index("--parallel") + 1], "40")
        self.assertEqual(command[command.index("--invocation-updates") + 1], "1")
        self.assertIn("--resume", command)
        self.assertNotIn("--initial-weights", command)

    def test_config_rejects_deadline_over_300_and_invalid_parallel(self):
        for values, message in [({"INVOCATION_SECONDS": "301"}, "INVOCATION_SECONDS"),
                                ({"TRAIN_GAMES": "26"}, "TRAIN_GAMES"),
                                ({"TRAIN_PARALLEL": "7"}, "parallel must divide"),
                                ({"TRAIN_GAMES": "39"}, "games must be even")]:
            with patch.dict("os.environ", values, clear=True):
                with self.assertRaisesRegex(guard.Refusal, message):
                    guard.run_config()

    def test_lock_contention_refuses_without_replacing_or_signalling_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heavy.lock"
            path.touch()
            with guard.shared_locks([path]):
                with self.assertRaisesRegex(guard.Refusal, "heavy lock busy"):
                    with guard.shared_locks([path]):
                        self.fail("admitted second heavy job")

    def test_payload_overflow_refuses_before_writing_beyond_cap(self):
        with tempfile.TemporaryFile() as output:
            with self.assertRaisesRegex(guard.Refusal, "payload log limit"):
                guard.append_payload(output, b"xx", guard.PAYLOAD_LIMIT - 1)
            self.assertEqual(output.tell(), 0)

    def test_each_invocation_gets_its_own_log_and_full_payload_budget(self):
        with patch.dict("os.environ", {}, clear=True):
            config = guard.run_config()
        filesystem = SimpleNamespace(f_bavail=200 * guard.GIB, f_frsize=1)
        def complete(_command, _locks, output, written, *_arguments):
            return guard.append_payload(output, b"12345678", written)

        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"
            logs.mkdir()
            with patch.object(guard, "DATA", Path(directory)), \
                    patch.object(guard, "PAYLOAD_LIMIT", 8), \
                    patch.object(guard.os, "statvfs", return_value=filesystem), \
                    patch.object(guard, "supervise", side_effect=complete):
                first = guard.run_invocation(config, 0, False, (), "fixture", {})
                second = guard.run_invocation(config, 1, True, (), "fixture", {})

            self.assertEqual((first, second), (8, 8))
            self.assertEqual((logs / "payload-0001.log").read_bytes(), b"12345678")
            self.assertEqual((logs / "payload-0002.log").read_bytes(), b"12345678")

    def test_disk_floor_reserves_remaining_logs_and_rejects_one_byte_below(self):
        with patch.dict("os.environ", {}, clear=True):
            config = guard.run_config()
        for update in (0, 999):
            required = 100 * guard.GIB + (1000 - update) * guard.PAYLOAD_LIMIT
            for deficit in (0, 1):
                with self.subTest(update=update, deficit=deficit), tempfile.TemporaryDirectory() as directory:
                    logs = Path(directory) / "logs"
                    logs.mkdir()
                    filesystem = SimpleNamespace(f_bavail=required - deficit, f_frsize=1)
                    with patch.object(guard, "DATA", Path(directory)), \
                            patch.object(guard.os, "statvfs", return_value=filesystem), \
                            patch.object(guard, "supervise", return_value=0) as supervise:
                        if deficit:
                            with self.assertRaisesRegex(guard.Refusal, "disk available below 100 GiB plus remaining payload budget"):
                                guard.run_invocation(config, update, True, (), "fixture", {})
                            supervise.assert_not_called()
                            self.assertEqual(list(logs.iterdir()), [])
                        else:
                            guard.run_invocation(config, update, True, (), "fixture", {})
                            supervise.assert_called_once()

    def test_failed_invocation_evidence_is_preserved_and_duplicate_attempt_refused(self):
        with patch.dict("os.environ", {}, clear=True):
            config = guard.run_config()
        filesystem = SimpleNamespace(f_bavail=200 * guard.GIB, f_frsize=1)
        def fail(_command, _locks, output, written, *_arguments):
            guard.append_payload(output, b"failure evidence", written)
            raise guard.Refusal("native fixture failed")

        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"
            logs.mkdir()
            with patch.object(guard, "DATA", Path(directory)), \
                    patch.object(guard.os, "statvfs", return_value=filesystem), \
                    patch.object(guard, "supervise", side_effect=fail) as supervise:
                with self.assertRaisesRegex(guard.Refusal, "native fixture failed"):
                    guard.run_invocation(config, 180, True, (), "fixture", {})
                with self.assertRaisesRegex(FileExistsError, "payload-0181.log"):
                    guard.run_invocation(config, 180, True, (), "fixture", {})

                supervise.assert_called_once()
            self.assertEqual((logs / "payload-0181.log").read_bytes(), b"failure evidence")

    def test_invocation_log_namespace_rejects_completed_negative_or_over_1000_updates(self):
        for update, target in [(1000, 1000), (-1, 1000), (0, 1001), (0, 0)]:
            with self.subTest(update=update, target=target), patch.object(guard, "supervise") as supervise:
                with self.assertRaisesRegex(guard.Refusal, "invocation log update outside 1..1000 target"):
                    guard.run_invocation({"updates": target}, update, True, (), "fixture", {})
                supervise.assert_not_called()

    def test_fresh_metrics_accepts_precreated_lock_but_not_old_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".metrics.writer.lock").touch(mode=0o600)
            guard.validate_fresh_metrics(path)
            (path / "metrics.state").touch()
            with self.assertRaisesRegex(guard.Refusal, "fresh metrics"):
                guard.validate_fresh_metrics(path)

    def test_fresh_weights_hash_rejects_substituted_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.write_bytes(b"approved fixture")
            expected = guard.file_sha256(path)
            path.write_bytes(b"substituted fixture")
            with self.assertRaisesRegex(guard.Refusal, "initial weights hash"):
                guard.verify_initial_weights(path, expected)

    def test_shutdown_before_spawn_never_starts_payload(self):
        with patch.object(guard, "STOP_REQUESTED", True), patch.object(guard.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(guard.Refusal, "shutdown requested before payload"):
                guard.supervise([], (), None, 0, {}, "", {})
            spawn.assert_not_called()

    def test_short_payload_write_is_failure_not_claimed_complete_output(self):
        class ShortWriter:
            def write(self, data):
                return len(data) - 1

        with self.assertRaisesRegex(guard.Refusal, "short payload log write"):
            guard.append_payload(ShortWriter(), b"fixture", 0)


if __name__ == "__main__":
    unittest.main()
