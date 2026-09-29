"""Bounded filesystem fixtures and mocked Docker/native processes; no GPU work."""

import contextlib
import json
import os
from pathlib import Path
import selectors
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import train_runner as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        workspace = self.root / "workspace/repository"
        campaign = workspace / "temp/campaign"
        job = campaign / "invocations/000001"
        for path in (workspace / "scripts", campaign / "bin", campaign / "inputs", job):
            path.mkdir(parents=True, mode=0o700)
        campaign.chmod(0o700)
        binary = campaign / "bin/drysua"
        binary.write_bytes(b"\x7fELFfixture")
        binary.chmod(0o500)
        lock = workspace / "heavy.lock"
        lock.touch(mode=0o600)
        self.spec = {
            "version": 1, "workspace_root": str(workspace),
            "campaign_directory": str(campaign), "job_directory": str(job),
            "command": [str(binary), "train-annealed", "--invocation-updates", "1",
                        "--device", "cpu"], "mode": "cpu", "docker_context": "rootless",
            "image": "ubuntu:26.04@sha256:" + "a" * 64, "gpu_uuid": None,
            "lock_paths": [str(lock)], "campaign_id": "campaign", "invocation_id": "000001",
        }
        self.container_id = "c" * 64
        self.sample = {"memory_available": 24 * runner.GIB, "cpu_celsius": None,
                       "gpu_free_mib": None, "gpu_celsius": None}
        self.limits = {"memory.max": str(12 * runner.GIB), "memory.swap.max": "0",
                       "pids.max": "1024", "cpu.max": "max 100000"}
        self.events = {"memory.max": 0, "memory.high": 0, "memory.oom": 0,
                       "memory.oom_kill": 0, "pids.max": 0}

    def intent(self):
        intent = runner.make_intent(self.spec)
        runner.write_json(Path(self.spec["job_directory"]) / "intent.json", intent)
        runner.write_json(Path(self.spec["job_directory"]) / "receipt.json", {
            "schema": "drysua-training-runner-receipt/v1", "container_id": self.container_id,
            "intent": intent, "runner_sha256": "0" * 64})
        return intent

    def inspected(self, running=False):
        intent = runner.make_intent(self.spec)
        return {"Id": self.container_id, "Name": "/" + intent["name"],
                "Labels": intent["labels"], "ImageReference": self.spec["image"],
                "HostConfig": {"Memory": 12 * runner.GIB, "MemorySwap": 12 * runner.GIB,
                               "PidsLimit": 1024, "CpuQuota": 0, "NanoCpus": 0, "CpusetCpus": ""},
                "State": {"Running": running, "OOMKilled": False,
                          "ExitCode": 0, "Status": "running" if running else "exited"}}

    def test_spec_accepts_cpu_without_cuda_and_rejects_shell_unbounded_or_unpinned_input(self):
        self.assertEqual(runner.validate_spec(self.spec), self.spec)
        cases = [("version", True, "version"), ("image", "ubuntu:latest", "digest"),
                 ("campaign_id", "../other", "campaign_id"),
                 ("command", ["/bin/sh", "-c", "true"], "binary"),
                 ("command", self.spec["command"] + ["x" * 4097], "argument"),
                 ("lock_paths", [], "lock_paths"), ("lock_paths", [[]], "lock_paths")]
        for key, value, message in cases:
            with self.subTest(key=key), self.assertRaisesRegex(runner.Refusal, message):
                runner.validate_spec(dict(self.spec, **{key: value}))

    def test_spec_refuses_mutable_binary_symlink_and_unbounded_training(self):
        binary = Path(self.spec["command"][0])
        binary.chmod(0o777)
        with self.assertRaisesRegex(runner.Refusal, "binary"):
            runner.validate_spec(self.spec)
        binary.chmod(0o500)
        command = self.spec["command"].copy()
        command[3] = "1000"
        with self.assertRaisesRegex(runner.Refusal, "invocation-updates"):
            runner.validate_spec(dict(self.spec, command=command))
        alias = binary.with_name("alias")
        alias.symlink_to(binary)
        command[0], command[3] = str(alias), "1"
        with self.assertRaisesRegex(runner.Refusal, "canonical"):
            runner.validate_spec(dict(self.spec, command=command))

    def test_mounts_protect_workspace_binary_inputs_and_do_not_inherit_secrets(self):
        command = runner.create_command(self.spec, runner.make_intent(self.spec))
        mounts = [command[index + 1] for index, item in enumerate(command) if item == "--mount"]
        workspace = str(Path(self.spec["workspace_root"]).parent)
        self.assertIn(f"type=bind,src={workspace},dst={workspace},readonly", mounts)
        for name in ("bin", "inputs"):
            path = str(Path(self.spec["campaign_directory"]) / name)
            self.assertIn(f"type=bind,src={path},dst={path},readonly", mounts)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("docker.sock", " ".join(command))
        self.assertIn("--pids-limit=1024", command)
        self.assertNotIn("--pids-limit=10000", command)
        with patch.dict(os.environ, {"PRIVATE_TOKEN": "never-forward-this"}):
            self.assertNotIn("never-forward-this", " ".join(runner.create_command(self.spec, runner.make_intent(self.spec))))

    def test_shared_lock_contention_and_symlink_refuse_without_replacing_inode(self):
        lock = Path(self.spec["lock_paths"][0])
        inode = lock.stat().st_ino
        with runner.shared_locks(self.spec["lock_paths"]):
            with self.assertRaisesRegex(runner.Refusal, "lock busy"):
                with runner.shared_locks(self.spec["lock_paths"]):
                    self.fail("concurrent owner admitted")
        self.assertEqual(lock.stat().st_ino, inode)
        alias = lock.with_name("alias.lock")
        alias.symlink_to(lock)
        with self.assertRaises((OSError, runner.Refusal)):
            with runner.shared_locks([str(alias)]):
                self.fail("symlink lock admitted")

    def test_limits_uid_mapping_and_event_deltas_fail_closed(self):
        runner.validate_limits(self.limits)
        runner.validate_uid_mapping("0 1000 1\n1 100000 65536\n", 1000)
        runner.check_events(self.events, self.events)
        for key, value in [("memory.max", "max"), ("memory.swap.max", "1"),
                           ("pids.max", "max"), ("cpu.max", "10000 100000")]:
            with self.subTest(key=key), self.assertRaisesRegex(runner.Refusal, key):
                runner.validate_limits(dict(self.limits, **{key: value}))
        with self.assertRaisesRegex(runner.Refusal, "UID mapping"):
            runner.validate_uid_mapping("0 0 4294967295\n", 1000)
        with self.assertRaisesRegex(runner.Refusal, "memory.high"):
            runner.check_events(self.events, dict(self.events, **{"memory.high": 1}))

    def test_resource_thresholds_and_cpu_mode_do_not_require_gpu(self):
        runner.check_resources(self.sample, "cpu", admission=True)
        gpu = dict(self.sample, cpu_celsius=90, gpu_free_mib=4096, gpu_celsius=85)
        runner.check_resources(gpu, "gpu", admission=True)
        for key, value, message in [("memory_available", 16 * runner.GIB - 1, "MemAvailable"),
                                    ("cpu_celsius", 91, "CPU temperature"),
                                    ("gpu_free_mib", 4095, "VRAM"),
                                    ("gpu_celsius", 86, "GPU temperature")]:
            with self.subTest(key=key), self.assertRaisesRegex(runner.Refusal, message):
                runner.check_resources(dict(gpu, **{key: value}), "gpu", admission=False)

    def test_created_container_limits_are_checked_before_native_start(self):
        data = self.inspected(False)
        data["State"]["Status"] = "created"
        runner.validate_created(data)
        data["HostConfig"]["MemorySwap"] = -1
        with self.assertRaisesRegex(runner.Refusal, "created container limits"):
            runner.validate_created(data)

    def test_effective_cgroup_rejects_unauthorized_10000_pid_limit(self):
        limits = dict(self.limits, **{"pids.max": "10000"})
        with self.assertRaises(runner.Refusal) as failure:
            runner.validate_limits(limits)
        self.assertEqual(str(failure.exception), "invalid actual pids.max")

    def test_docker_config_rejects_unauthorized_10000_pid_limit(self):
        data = self.inspected(False)
        data["State"]["Status"] = "created"
        data["HostConfig"]["PidsLimit"] = 10000
        with self.assertRaises(runner.Refusal) as failure:
            runner.validate_created(data)
        self.assertEqual(str(failure.exception), "incorrect created container limits")

    def test_ownership_mismatch_never_sends_stop_or_kill(self):
        self.intent()
        inspected = self.inspected(running=True)
        inspected["Labels"] = dict(inspected["Labels"], **{"io.drysua.training.campaign": "someone-else"})
        with patch.object(runner, "verify_daemon"), patch.object(runner, "docker_checked", return_value=json.dumps(inspected)) as docker:
            with self.assertRaisesRegex(runner.Refusal, "ownership"):
                runner.stop_owned_container(self.spec, self.container_id)
            self.assertEqual(docker.call_count, 1)
            self.assertEqual(docker.call_args.args[1][0], "inspect")

    def test_stop_uses_full_owned_id_and_confirms_stopped_state(self):
        self.intent()
        outputs = [json.dumps(self.inspected(True)), "stopped", json.dumps(self.inspected(False))]
        with patch.object(runner, "verify_daemon"), patch.object(runner, "docker_checked", side_effect=outputs) as docker:
            state = runner.stop_owned_container(self.spec, self.container_id)
        self.assertFalse(state["Running"])
        self.assertEqual(docker.call_args_list[1].args[1], ["stop", "--time=5", self.container_id])

    def test_missing_receipt_or_running_container_cannot_be_success(self):
        result = {"returncode": 0, "cleanup_confirmed": False, "container_state": None,
                  "verified_limits": None}
        with self.assertRaisesRegex(runner.Refusal, "unconfirmed"):
            runner.validate_success(result)
        result.update(cleanup_confirmed=True, container_state=self.inspected(True)["State"],
                      verified_limits=self.limits)
        with self.assertRaisesRegex(runner.Refusal, "still running"):
            runner.validate_success(result)

    def test_recovery_refuses_missing_receipt_before_inspecting_docker(self):
        runner.write_json(Path(self.spec["job_directory"]) / "intent.json", runner.make_intent(self.spec))
        with patch.object(runner, "docker_checked") as docker:
            with self.assertRaisesRegex(runner.Refusal, "receipt missing"):
                runner.inspect_owned_container(self.spec, self.container_id)
            docker.assert_not_called()

    def test_fifo_spec_cannot_block_bounded_reader(self):
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(runner.Refusal, "regular file"):
            runner.read_small(fifo)

    def test_missing_final_evidence_turns_zero_exit_into_persisted_failure(self):
        result = {"returncode": 0, "cleanup_confirmed": True,
                  "container_state": self.inspected(False)["State"], "verified_limits": None, "error": None}
        finished = runner.finish_result(self.spec, result)
        self.assertEqual(finished["returncode"], 1)
        self.assertIn("limits.json", finished["error"])

    def test_event_just_before_native_exit_is_saved_and_refuses_success(self):
        baseline = {"limits": self.limits, "events": self.events}
        changed = {"limits": self.limits, "events": dict(self.events, **{"pids.max": 1})}
        child = Mock(poll=Mock(return_value=0))
        def read(path, _limit=65536):
            if str(path).endswith("uid_map"):
                return b"0 1000 1\n"
            if str(path).endswith("cpu.stat"):
                return b"usage_usec 100\nuser_usec 60\nsystem_usec 40\n"
            return b"1"

        with patch.object(runner.os, "getuid", return_value=0), \
                patch.object(runner, "read_small", side_effect=read), \
                patch.object(runner, "read_cgroup", side_effect=[baseline, baseline, changed]), \
                patch.object(runner, "received_locks", return_value=contextlib.nullcontext([])), \
                patch.object(runner.subprocess, "Popen", return_value=child), \
                patch.object(runner, "stop_session"), patch.object(runner, "write_json") as write, \
                patch.object(runner, "open", side_effect=lambda *_args, **_kwargs: (self.root / "usage").open("wb")):
            with self.assertRaisesRegex(runner.Refusal, "pids.max"):
                runner.inside(self.spec, 1000, runner.Signals())
        self.assertEqual(write.call_args.args[1]["events"]["pids.max"], 1)

    def test_preflight_resource_failure_writes_result_without_creating_container(self):
        with patch.object(runner, "verify_daemon"), \
                patch.object(runner, "host_sample", side_effect=runner.Refusal("CPU temperature above 90 C")), \
                patch.object(runner, "docker_checked") as docker:
            result = runner.run_job(self.spec)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("CPU temperature", result["error"])
        docker.assert_not_called()
        self.assertTrue((Path(self.spec["job_directory"]) / "result.json").exists())

    def test_contended_duplicate_cannot_write_into_active_job_evidence(self):
        with runner.shared_locks(self.spec["lock_paths"]), patch.object(runner, "docker_checked") as docker:
            result = runner.run_job(self.spec)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("lock busy", result["error"])
        self.assertEqual(list(Path(self.spec["job_directory"]).iterdir()), [])
        docker.assert_not_called()

    def test_existing_job_intent_refuses_without_overwriting_completion(self):
        self.intent()
        path = Path(self.spec["job_directory"]) / "result.json"
        runner.write_json(path, {"original": True})
        with patch.object(runner, "docker_checked") as docker:
            result = runner.run_job(self.spec)
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(json.loads(path.read_text()), {"original": True})
        docker.assert_not_called()

    def test_daemon_must_report_rootless_v2_before_any_container_creation(self):
        for security, cgroup in [([], "2"), (["name=rootless"], "1")]:
            with self.subTest(cgroup=cgroup), patch.object(runner.os, "getuid", return_value=1000), \
                    patch.object(runner, "docker_checked", return_value=json.dumps({"security": security, "cgroup": cgroup})):
                with self.assertRaisesRegex(runner.Refusal, "rootless cgroup-v2"):
                    runner.verify_daemon(self.spec)

    def test_cleanup_commands_never_outlive_remaining_deadline(self):
        with patch.object(runner.time, "monotonic", return_value=100):
            self.assertLessEqual(runner.remaining_timeout(102, 8), 2)
            with self.assertRaisesRegex(runner.Refusal, "cleanup deadline"):
                runner.remaining_timeout(100, 5)

    def test_transferred_flock_survives_host_descriptor_close_and_long_socket_path(self):
        directory = self.root / ("x" * 90)
        directory.mkdir()
        ready, release = threading.Event(), threading.Event()
        errors = []
        with runner.shared_locks(self.spec["lock_paths"]) as descriptors:
            lease = runner.LockTransfer(directory / "runner-locks.sock", descriptors)
            self.addCleanup(lease.close)
            def receive():
                try:
                    with runner.received_locks(self.spec, socket_path=lease.address):
                        ready.set()
                        release.wait(timeout=1)
                except Exception as error:
                    errors.append(error)
                    ready.set()
            thread = threading.Thread(target=receive)
            thread.start()
            self.addCleanup(release.set)
            with selectors.DefaultSelector() as selector:
                selector.register(lease.socket, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=1), "receiver did not connect")
            lease.send()
            self.assertTrue(ready.wait(timeout=1))
        try:
            with self.assertRaisesRegex(runner.Refusal, "lock busy"):
                with runner.shared_locks(self.spec["lock_paths"]):
                    self.fail("shared lease released before native owner")
        finally:
            release.set()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        with runner.shared_locks(self.spec["lock_paths"]):
            pass

    def test_signal_and_deadline_refuse_continuation_without_native_execution(self):
        signals = runner.Signals()
        runner.check_deadline(signals, 10, now=9)
        with self.assertRaisesRegex(runner.Refusal, "deadline"):
            runner.check_deadline(signals, 10, now=10)
        signals.request(15, None)
        with self.assertRaisesRegex(runner.Refusal, "shutdown"):
            runner.check_deadline(signals, 10, now=9)

    def test_bounded_logs_preserve_existing_evidence_and_reject_short_writes(self):
        path = self.root / "evidence.json"
        runner.write_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            runner.write_json(path, {"replacement": True})
        self.assertEqual(json.loads(path.read_text()), {"original": True})
        with self.assertRaisesRegex(runner.Refusal, "log limit"):
            runner.append_bytes(Mock(), b"xx", runner.PAYLOAD_LIMIT - 1, runner.PAYLOAD_LIMIT)
        with self.assertRaisesRegex(runner.Refusal, "short write"):
            runner.append_bytes(Mock(write=lambda data: len(data) - 1), b"xx", 0, 10)

    def test_receipt_is_durable_before_start_and_result_requires_native_limit_evidence(self):
        def execute(spec, container_id, signals, deadline, lease, summary=None):
            job = Path(spec["job_directory"])
            receipt = json.loads((job / "receipt.json").read_text())
            self.assertEqual(receipt["container_id"], container_id)
            self.assertTrue((job / "intent.json").exists())
            runner.write_json(job / "limits.json", {"limits": self.limits, "events": self.events})
            runner.write_json(job / "limits-final.json", {"limits": self.limits, "events": self.events})
            lease.sent = True
            return 0, {"samples": 1, "last": self.sample}

        state = self.inspected(False)["State"]
        created = self.inspected(False)
        created["State"]["Status"] = "created"
        with patch.object(runner, "verify_daemon"), patch.object(runner, "host_sample", return_value=self.sample), \
                patch.object(runner, "check_disk"), patch.object(runner, "docker_checked", return_value=self.container_id), \
                patch.object(runner, "inspect_owned_container", return_value=created), \
                patch.object(runner, "execute_attached", side_effect=execute), \
                patch.object(runner, "cleanup_owned", return_value=state):
            result = runner.run_job(self.spec)
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["verified_limits"], self.limits)

    def test_extended_inside_budget_applies_to_child_and_records_baseline_before_spawn(self):
        clock, baseline = [100.0], {"limits": self.limits, "events": self.events}
        def read(path, _limit=65536):
            if str(path).endswith("uid_map"):
                return b"0 1000 1\n"
            if str(path).endswith("cpu.stat"):
                return b"usage_usec 100\nuser_usec 60\nsystem_usec 40\n"
            return b"1"
        def launch(*_args, **_kwargs):
            self.assertEqual(usage.call_count, 1, "CPU baseline must precede payload")
            return Mock(poll=Mock(side_effect=[None, 0]))

        with patch.object(runner.os, "getuid", return_value=0), \
                patch.object(runner, "read_small", side_effect=read), \
                patch.object(runner, "read_cgroup", return_value=baseline), \
                patch.object(runner, "received_locks", return_value=contextlib.nullcontext([])), \
                patch.object(runner, "cgroup_usage", wraps=runner.cgroup_usage) as usage, \
                patch.object(runner.subprocess, "Popen", side_effect=launch), \
                patch.object(runner.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(runner.time, "sleep", side_effect=lambda _: clock.__setitem__(0, 374.5)), \
                patch.object(runner, "stop_session"), patch.object(runner, "write_json"), \
                patch.object(runner, "open", side_effect=lambda *_a, **_k: (self.root / "usage").open("wb")):
            self.assertEqual(runner.inside(dict(self.spec, invocation_seconds=275), 1000, runner.Signals()), 0)

    def test_failed_job_retains_resource_summary_and_selected_timeout_receipt(self):
        self.spec["invocation_seconds"] = 275
        def fail(spec, container_id, signals, deadline, lease, summary=None):
            receipt = json.loads((Path(spec["job_directory"]) / "receipt.json").read_text())
            self.assertEqual(receipt["timeouts"]["runner_seconds"], 300)
            self.assertEqual(deadline, 400)
            summary.update(samples=2, gpu_utilization_mean_percent=77)
            raise runner.Refusal("invocation deadline reached")

        created = self.inspected(False)
        created["State"]["Status"] = "created"
        with patch.object(runner, "verify_daemon"), patch.object(runner, "host_sample", return_value=dict(self.sample)), \
                patch.object(runner, "check_disk"), patch.object(runner.time, "monotonic", return_value=100), \
                patch.object(runner, "docker_checked", return_value=self.container_id), \
                patch.object(runner, "inspect_owned_container", return_value=created), \
                patch.object(runner, "execute_attached", side_effect=fail), \
                patch.object(runner, "cleanup_owned", return_value=self.inspected(False)["State"]):
            result = runner.run_job(self.spec)
        self.assertEqual(result["resources"]["gpu_utilization_mean_percent"], 77)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("deadline", result["error"])

    def test_optional_invocation_seconds_preserves_legacy_intent_and_rejects_before_work(self):
        original = runner.make_intent(self.spec)
        runner.validate_spec(self.spec)
        self.assertNotIn("invocation_seconds", self.spec)
        self.assertEqual(runner.make_intent(self.spec), original)
        self.assertEqual(runner.validate_spec(dict(self.spec, invocation_seconds=275))["invocation_seconds"], 275)
        for value in (True, 0, 276, 235.0, "235", None):
            with self.subTest(value=value), patch.object(runner, "shared_locks") as locks:
                with self.assertRaisesRegex(ValueError, "invocation_seconds"):
                    runner.run_job(dict(self.spec, invocation_seconds=value))
                locks.assert_not_called()


class TimeoutTelemetryTests(unittest.TestCase):
    def test_timeout_budgets_have_one_validated_source_and_keep_default(self):
        for payload, capture, total, reserve in ((1, 6, 26, 56), (235, 240, 260, 290), (275, 280, 300, 330)):
            with self.subTest(payload=payload):
                value = runner.resolve_timeouts(payload)
                self.assertEqual((value.payload_seconds, value.capture_seconds, value.runner_seconds,
                                  value.controller_reserve_seconds), (payload, capture, total, reserve))
        self.assertEqual(runner.resolve_timeouts().payload_seconds, 235)
        for value in (False, True, None, 0, -1, 276, 10**30, 235.0, float("nan"), float("inf"), "235"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invocation_seconds"):
                runner.resolve_timeouts(value)

    def test_host_cpu_ticks_exclude_guest_double_count_and_discover_core_count(self):
        data = b"cpu 10 0 5 85 0 0 0 0 999 999\ncpu0 5 0 2 43 0 0 0 0\ncpu1 5 0 3 42 0 0 0 0\nintr 50\n"
        self.assertEqual(runner.parse_cpu_ticks(data), {"busy": 15, "total": 100, "count": 2})
        before = {"cpu_ticks": {"busy": 15, "total": 100, "count": 2}}
        after = {"cpu_ticks": {"busy": 35, "total": 200, "count": 2}}
        self.assertEqual(runner.host_cpu_percent(before, after), 40.0)
        self.assertIsNone(runner.host_cpu_percent(None, after))
        self.assertIsNone(runner.host_cpu_percent(after, before))
        self.assertIsNone(runner.host_cpu_percent(before, {"cpu_ticks": {"busy": 35, "total": 200, "count": 3}}))

    def test_cpu_counter_parser_rejects_duplicate_excess_and_invalid_counters(self):
        header = b"cpu 1 0 0 1 0 0 0 0\n"
        line = b"cpu0 1 0 0 1 0 0 0 0\n"
        for data in (header + line + line, header + b"cpu0 -1 0 0 1 0 0 0 0\n",
                     header + b"cpu0 18446744073709551616 0 0 1 0 0 0 0\n",
                     header + b"".join(f"cpu{i} 1 0 0 1 0 0 0 0\n".encode() for i in range(257))):
            with self.assertRaisesRegex(runner.Refusal, "CPU|counter"):
                runner.parse_cpu_ticks(data)

    def test_cpu_reader_does_not_read_unrelated_unbounded_interrupt_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stat"
            path.write_bytes(b"cpu 1 0 0 1 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0\nintr " + b"0 " * 50000)
            self.assertEqual(runner.read_cpu_ticks(path)["count"], 1)

    def test_gpu_optional_utilization_and_power_are_missing_not_fabricated_zero(self):
        uuid = "GPU-11111111-1111-1111-1111-111111111111"
        record = runner.parse_gpu_sample(f"0, {uuid}, 4096, 85, 100, 40, 501.25", uuid)
        self.assertEqual(record["gpu_utilization_percent"], 100)
        self.assertEqual(record["gpu_memory_utilization_percent"], 40)
        self.assertEqual(record["gpu_power_watts"], 501.25)
        missing = runner.parse_gpu_sample(f"0, {uuid}, 4096, 85, N/A, [N/A], N/A", uuid)
        self.assertIsNone(missing["gpu_utilization_percent"])
        for suffix in ("101, 20, 100", "nan, 20, 100", "20, 20, inf"):
            with self.assertRaisesRegex(runner.Refusal, "GPU optional telemetry"):
                runner.parse_gpu_sample(f"0, {uuid}, 4096, 85, {suffix}", uuid)
        with self.assertRaisesRegex(runner.Refusal, "mandatory GPU"):
            runner.parse_gpu_sample(f"0, {uuid}, N/A, 85, 10, 20, 100", uuid)

    def test_host_gpu_sampling_uses_one_bounded_query_with_timestamp(self):
        uuid = "GPU-11111111-1111-1111-1111-111111111111"
        files = {"/proc/meminfo": b"MemAvailable: 33554432 kB\n",
                 "/proc/stat": b"cpu 1 0 0 1 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0\n"}
        with patch.object(runner, "read_small", side_effect=lambda path, *_: files[str(path)]), \
                patch.object(runner, "read_cpu_ticks", return_value=runner.parse_cpu_ticks(files["/proc/stat"])), \
                patch.object(Path, "exists", return_value=False), patch.object(runner.time, "monotonic", return_value=10), \
                patch.object(runner, "bounded_command", return_value=f"0, {uuid}, 5000, 40, 75, 30, 200") as query:
            sample = runner.host_sample({"mode": "gpu", "gpu_uuid": uuid})
        query.assert_called_once()
        self.assertEqual(query.call_args.kwargs, {"timeout": 1, "limit": 4096})
        self.assertIn("utilization.gpu", " ".join(query.call_args.args[0]))
        self.assertEqual(sample["monotonic"], 10)
        self.assertEqual(sample["cpu_count"], 1)

    def test_resource_summary_reports_cpu_core_percent_and_gpu_sample_mean(self):
        summary = {"samples": 0}
        first = {"cpu_ticks": {"busy": 0, "total": 100, "count": 8}, "gpu_utilization_percent": None}
        second = {"cpu_ticks": {"busy": 50, "total": 200, "count": 8}, "gpu_utilization_percent": 100}
        third = {"cpu_ticks": {"busy": 100, "total": 300, "count": 8}, "gpu_utilization_percent": 0}
        runner.update_resource_summary(summary, first)
        self.assertIsNone(summary["host_cpu_percent"])
        self.assertIsNone(summary["gpu_utilization_mean_percent"])
        runner.update_resource_summary(summary, second)
        runner.update_resource_summary(summary, third)
        self.assertEqual(summary["host_cpu_percent"], 400)
        self.assertEqual(summary["host_cpu_machine_percent"], 50)
        self.assertEqual(summary["gpu_utilization_mean_percent"], 50)
        self.assertEqual(summary["gpu_utilization_samples"], 2)

    def test_cgroup_usage_records_bounded_cpu_counters_and_memory_peak(self):
        values = {"cpu.stat": b"usage_usec 2000000\nuser_usec 1500000\nsystem_usec 500000\nnr_periods 0\n",
                  "memory.current": b"1024", "memory.peak": b"2048", "pids.current": b"5"}
        with patch.object(runner, "read_small", side_effect=lambda path, *_: values[Path(path).name]), \
                patch.object(runner.time, "monotonic", return_value=12):
            usage = runner.cgroup_usage(10)
        self.assertEqual(usage["cpu_stat"]["usage_usec"], 2000000)
        self.assertEqual(usage["memory_peak"], 2048)
        first = dict(usage, monotonic=10, cpu_stat=dict(usage["cpu_stat"], usage_usec=0))
        summary = runner.cgroup_summary(first, usage)
        self.assertEqual(summary["cpu_percent"], 100)
        self.assertEqual(summary["memory_peak"], 2048)
        self.assertIsNone(runner.cgroup_summary(usage, usage)["cpu_percent"])


if __name__ == "__main__":
    unittest.main()
