"""Pure in-memory controller regressions: no files, subprocesses, or sleeps."""
from contextlib import ExitStack
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train as controller


class InvocationTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(schema=1, trainer="bin/trainer", inspector="bin/inspector",
                           total_updates=2, invocation_updates=1, history_every=20,
                           training_args=["--games", "40", "--parallel", "20"],
                           workspace_root="/workspace", mode="cpu", docker_context="rootless",
                           image=None, gpu_uuid=None, lock_paths=["/workspace/heavy.lock"])
        self.current = dict(campaign_id="a" * 32, accepted_invocation=0, accepted_updates=0,
                            pending_invocation=None, receipt_sha256=None, last_error=None,
                            phase="prepared", schema=controller.SCHEMA)
        self.directory = Path("/workspace/campaign")
        self.enterContext(patch.object(controller, "private_path", side_effect=Path))
        self.enterContext(patch.object(Path, "is_dir", return_value=True))
        self.enterContext(patch.object(controller.state.subprocess, "Popen",
                                       side_effect=AssertionError("real subprocess forbidden")))

    def test_config_defaults_and_integer_boundaries(self):
        self.assertEqual(controller.validate_config(self.config)["invocation_seconds"], 235)
        for seconds in (1, 235, 275):
            with self.subTest(seconds=seconds):
                value = controller.validate_config(dict(self.config, invocation_seconds=seconds))
                self.assertEqual(value["invocation_seconds"], seconds)
        for seconds in (True, False, 0, 276, 235.0, "235", None):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "invocation_seconds"):
                controller.validate_config(dict(self.config, invocation_seconds=seconds))

    def test_invalid_timeout_rejected_before_job_directory_creation(self):
        manifest = {"config": dict(self.config, invocation_seconds=276)}
        with patch.object(Path, "mkdir") as mkdir, \
                patch.object(controller, "fsync_directory"), \
                patch.object(controller, "write_exclusive"), \
                patch.object(controller, "atomic_status"):
            with self.assertRaisesRegex(ValueError, "invocation_seconds"):
                controller.make_job(self.directory, self.current, manifest)
        mkdir.assert_not_called()

    def test_create_freezes_normalized_default_timeout(self):
        with patch.object(controller, "read_json", return_value=self.config), \
                patch.object(controller, "collect_inputs", return_value=[]), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "mkdir"), \
                patch.object(controller, "fsync_directory"), \
                patch.object(controller, "freeze_inputs", return_value={}), \
                patch.object(controller, "write_exclusive") as write, \
                patch.object(controller, "atomic_status"), \
                patch.object(controller, "status"):
            controller.create(Path("/workspace/config.json"), self.directory)
        payload = next(call.args[1] for call in write.call_args_list
                       if call.args[0].name == "manifest.json")
        self.assertEqual(controller.decode_json(payload)["config"]["invocation_seconds"], 235)
        self.assertNotIn("invocation_seconds", self.config)

    def test_legacy_result_hash_is_unchanged_by_validation(self):
        result = dict(schema="drysua-training-runner/v1", returncode=0,
                      container_id="b" * 64,
                      container_state=dict(Running=False, OOMKilled=False, ExitCode=0))
        before = controller.digest(controller.encode(result))
        with patch.object(controller, "read_json", return_value=result):
            validated = controller.validate_result(self.directory / "result.json")
        self.assertEqual(controller.digest(controller.encode(validated)), before)
        self.assertNotIn("invocation_seconds", validated)

    def test_result_cannot_claim_a_different_timeout_than_frozen_job(self):
        result = dict(schema="drysua-training-runner/v1", returncode=0, container_id="b" * 64,
                      container_state=dict(Running=False, OOMKilled=False, ExitCode=0),
                      timeouts=asdict(controller.resolve_timeouts(275)))
        with patch.object(controller, "read_json", return_value=result):
            with self.assertRaisesRegex(ValueError, "runner timeout evidence differs"):
                controller.validate_result(self.directory / "result.json", controller.resolve_timeouts(235))
            accepted = controller.validate_result(self.directory / "result.json", controller.resolve_timeouts(275))
        self.assertEqual(accepted, result)
        result["timeouts"] = dict(asdict(controller.resolve_timeouts(1)), payload_seconds=True)
        with patch.object(controller, "read_json", return_value=result):
            with self.assertRaisesRegex(ValueError, "runner timeout evidence"):
                controller.validate_result(self.directory / "result.json", controller.resolve_timeouts(1))

    def test_spec_preserves_legacy_absence_and_propagates_explicit_timeout(self):
        for seconds in (None, 235, 275):
            config = dict(self.config)
            if seconds is not None:
                config["invocation_seconds"] = seconds
            before = copy.deepcopy(config)
            spec, expected = controller.job_spec(self.directory, self.current, {"config": config})
            self.assertEqual(spec.get("invocation_seconds"), seconds)
            self.assertEqual("invocation_seconds" in spec, seconds is not None)
            self.assertEqual(config, before)
            self.assertEqual(expected, 1)
            self.assertEqual(spec["command"][2:6], ["--games", "40", "--parallel", "20"])

    def test_make_job_writes_explicit_timeout_without_changing_workload(self):
        manifest = {"config": dict(self.config, invocation_seconds=275)}
        with patch.object(Path, "mkdir"), \
                patch.object(controller, "fsync_directory"), \
                patch.object(controller, "write_exclusive") as write, \
                patch.object(controller, "atomic_status"):
            job, _ = controller.make_job(self.directory, self.current, manifest)
        self.assertEqual(write.call_args.args[0], job / "job.json")
        spec = controller.decode_json(write.call_args.args[1])
        self.assertEqual(spec["invocation_seconds"], 275)
        self.assertEqual(spec["command"][2:6], ["--games", "40", "--parallel", "20"])

    def test_legacy_status_keeps_original_manifest_and_hash_without_writes(self):
        files = dict.fromkeys(("frozen/controller.py", "frozen/train_state.py",
                               "bin/trainer", "bin/inspector"), {})
        manifest = dict(schema=controller.SCHEMA, campaign_id="a" * 32,
                        config=self.config, files=files)
        payload = controller.encode(manifest)
        current = dict(self.current, manifest_sha256=controller.digest(payload))
        with patch.object(Path, "stat", return_value=Mock(st_uid=controller.os.getuid(), st_mode=0o40700)), \
                patch.object(controller, "read_json", return_value=current), \
                patch.object(controller, "read_bytes", return_value=payload), \
                patch.object(controller, "verify_frozen"), \
                patch.object(controller, "validate_progress"), \
                patch.object(controller, "write_exclusive") as write, \
                patch.object(controller, "atomic_status") as atomic:
            result = controller.status(self.directory)
        self.assertEqual(result["manifest_sha256"], controller.digest(payload))
        self.assertEqual(controller.encode(manifest), payload)
        write.assert_not_called()
        atomic.assert_not_called()

    def run_loop(self, seconds, remaining, failure=False):
        config = dict(self.config)
        if seconds is not None:
            config["invocation_seconds"] = seconds
        with ExitStack() as stack:
            for name, result in (("requested", False), ("verify_frozen", None),
                                 ("campaign_disk_usage", 0), ("atomic_status", None)):
                stack.enter_context(patch.object(controller, name, return_value=result))
            stack.enter_context(patch.object(controller, "monotonic", return_value=100))
            make = stack.enter_context(patch.object(controller, "make_job", return_value=(self.directory, 1)))
            launch = stack.enter_context(patch.object(controller.state, "wait_runner"))
            accept = stack.enter_context(patch.object(controller, "accept_job"))
            launch.side_effect = ValueError("runner failure") if failure else None
            accept.side_effect = lambda *args: self.current.update(accepted_updates=2)
            call = lambda: controller.execute_loop(self.directory, self.current, {"config": config},
                                                    {}, Mock(is_set=Mock(return_value=False)), 100 + remaining)
            if failure:
                with self.assertRaisesRegex(ValueError, "runner failure"):
                    call()
            else:
                call()
        self.assertEqual(config["training_args"], ["--games", "40", "--parallel", "20"])
        return make, launch, accept

    def test_dynamic_reserve_pauses_before_creating_job(self):
        for seconds, remaining in ((None, 289), (1, 55), (275, 329)):
            with self.subTest(seconds=seconds):
                make, launch, _ = self.run_loop(seconds, remaining)
                make.assert_not_called()
                launch.assert_not_called()
                self.assertEqual(self.current["phase"], "paused")

    def test_exact_reserve_admits_with_resolved_wait_budgets(self):
        for seconds, runner, reserve in ((None, 260, 290), (1, 26, 56), (275, 300, 330)):
            with self.subTest(seconds=seconds):
                self.current["accepted_updates"] = 0
                _, launch, accept = self.run_loop(seconds, reserve)
                self.assertEqual(launch.call_args.args[1], runner)
                self.assertEqual(launch.call_args.args[4], reserve)
                accept.assert_called_once()

    def test_failure_never_retries_or_reduces_games_or_parallel(self):
        make, launch, accept = self.run_loop(275, 330, failure=True)
        make.assert_called_once()
        launch.assert_called_once()
        accept.assert_not_called()
        self.assertEqual(self.current["accepted_updates"], 0)


class RunnerCleanupTests(unittest.TestCase):
    def exercise_cleanup(self, elapsed, stop, graceful_timeout=False, hard_timeout=False):
        state = controller.state
        clock = [100.0]
        child = Mock(pid=123, returncode=None)
        child.poll.return_value = None
        waits = []

        def wait(timeout):
            waits.append(timeout)
            if (len(waits) == 1 and graceful_timeout) or (len(waits) == 2 and hard_timeout):
                clock[0] += timeout
                raise state.subprocess.TimeoutExpired("mock runner", timeout)
            return 0

        def stop_requested():
            if launch.called:
                clock[0] = 100 + elapsed
                return stop
            return False

        child.wait.side_effect = wait
        with patch.object(state.subprocess, "Popen", return_value=child) as launch, \
                patch.object(state, "process_identity", return_value={"pid": 123}), \
                patch.object(state.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(state.threading, "Event", side_effect=AssertionError("sleep forbidden")):
            message = "runner cleanup deadline exceeded" if hard_timeout else (
                "immediate stop requested" if stop else "runner deadline exceeded")
            with self.assertRaisesRegex(ValueError, message):
                state.wait_runner(["mock"], 260, stop_requested, Mock(), 290)
        child.terminate.assert_called_once_with()
        self.assertLessEqual(clock[0], 390)
        return child, waits

    def test_early_stop_allows_cleanup_until_original_deadline(self):
        child, waits = self.exercise_cleanup(10, True)
        self.assertEqual(waits, [275])
        child.kill.assert_not_called()

    def test_late_timeout_does_not_restart_payload_budget(self):
        child, waits = self.exercise_cleanup(260, False, graceful_timeout=True)
        self.assertEqual(waits, [25, 5])
        child.kill.assert_called_once_with()

    def test_overdue_cleanup_has_only_remaining_reserve(self):
        _, waits = self.exercise_cleanup(288, False, graceful_timeout=True)
        self.assertEqual(waits, [0, 2])

    def test_hard_reap_timeout_is_bounded_and_descriptive(self):
        _, waits = self.exercise_cleanup(260, False, graceful_timeout=True, hard_timeout=True)
        self.assertEqual(waits, [25, 5])

    def test_inspector_and_detach_termination_still_use_five_seconds_each(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [controller.state.subprocess.TimeoutExpired("mock", 5), 0]
        controller.state.terminate_child(child)
        self.assertEqual([call.kwargs["timeout"] for call in child.wait.call_args_list], [5, 5])

    def test_stop_before_launch_never_spawns_child(self):
        with patch.object(controller.state.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "immediate stop requested before runner launch"):
                controller.state.wait_runner(["mock"], 260, lambda: True, Mock(), 290)
        launch.assert_not_called()

    def test_successful_runner_is_not_signalled_or_waited_again(self):
        child = Mock(pid=123, returncode=0)
        child.poll.return_value = 0
        state = controller.state
        with patch.object(state.subprocess, "Popen", return_value=child), \
                patch.object(state, "process_identity", return_value={"pid": 123}), \
                patch.object(state.time, "monotonic", return_value=100):
            state.wait_runner(["mock"], 300, lambda: False, Mock(), 330)
        child.terminate.assert_not_called()
        child.wait.assert_not_called()

    def test_invalid_reserve_is_rejected_before_launch(self):
        for timeout, reserve in ((0, 290), (260, 260), (260, True), (260.0, 290)):
            with self.subTest(timeout=timeout, reserve=reserve), \
                    patch.object(controller.state.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "integer budgets with reap headroom"):
                    controller.state.wait_runner(["mock"], timeout, lambda: False, Mock(), reserve)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
