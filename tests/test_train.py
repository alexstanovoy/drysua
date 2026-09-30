"""Deterministic campaign contracts; no trainer or inspector is executed."""
import contextlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("training_controller", SCRIPT)
sys.path.insert(0, str(SCRIPT.parent))
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)
import train_runner


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trainer = self.root / "trainer"
        self.trainer.write_bytes(b"not an executable: must never run")
        self.config = self.root / "config.json"
        self.campaign = self.root / "campaign"
        self.write_config()

    def write_config(self, **overrides):
        value = {"schema": 1, "trainer": str(self.trainer), "total_updates": 100}
        value.update(overrides)
        self.config.write_text(json.dumps(value))

    def create(self):
        return controller.create(self.config, self.campaign)

    def test_create_freezes_inputs_and_status_does_not_execute_them(self):
        self.create()
        self.trainer.write_bytes(b"changed source")
        status = controller.status(self.campaign)
        self.assertEqual(status["accepted_updates"], 0)
        self.assertEqual(status["phase"], "prepared")
        self.assertEqual((self.campaign / "bin" / "trainer").read_bytes(),
                         b"not an executable: must never run")

    def test_existing_campaign_is_never_modified(self):
        self.create()
        before = (self.campaign / "status.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()
        self.assertEqual((self.campaign / "status.json").read_bytes(), before)

    def test_separate_inspector_and_weight_tree_are_frozen(self):
        inspector = self.root / "inspector"
        inspector.write_bytes(b"inspector bytes")
        weights = self.root / "weights"
        weights.mkdir()
        (weights / "drysua.weights.safetensors").write_bytes(b"model bytes")
        (weights / "checkpoint.meta").write_bytes(b"must not copy optimizer or progress")
        self.write_config(inspector="inspector", initial_weights="weights")
        self.create()
        manifest = json.loads((self.campaign / "manifest.json").read_text())
        self.assertEqual((self.campaign / "bin/inspector").read_bytes(), b"inspector bytes")
        self.assertIn("inputs/initial_weights/drysua.weights.safetensors", manifest["files"])
        self.assertFalse((self.campaign / "inputs/initial_weights/checkpoint.meta").exists())

    def test_symlink_inside_frozen_tree_is_rejected(self):
        self.create()
        artifact = self.campaign / "bin/trainer"
        artifact.unlink()
        artifact.symlink_to(self.trainer)
        with self.assertRaisesRegex(ValueError, "symlink"):
            controller.status(self.campaign)

    def test_unknown_status_schema_is_rejected(self):
        self.create()
        path = self.campaign / "status.json"
        value = json.loads(path.read_text())
        value["schema"] = "future/v2"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "unsupported or corrupt status"):
            controller.status(self.campaign)

    def test_invalid_configuration_is_rejected_before_creation(self):
        cases = [dict(schema=2), dict(total_updates=True), dict(total_updates=10001),
                 dict(invocation_updates=17), dict(max_seconds=0), dict(history_every=0),
                 dict(unknown="value"), dict(training_args=["--updates=10"]),
                 dict(training_args=["--checkpoint-directory", "/tmp"]),
                 dict(training_args=["--metrics-directory=/tmp"]),
                 dict(training_args=["--unreviewed-option"]),
                 dict(training_args=["x" * 4097])]
        for case in cases:
            with self.subTest(case=case):
                self.write_config(**case)
                with self.assertRaises(ValueError):
                    self.create()
                self.assertFalse(self.campaign.exists())

    def test_duplicate_json_keys_are_rejected(self):
        self.config.write_text('{"schema":1,"schema":1}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.create()

    def test_nonfinite_json_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            controller.decode_json(b'{"value":1e9999}')

    def test_symlink_input_and_campaign_parent_are_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.trainer)
        self.write_config(trainer=str(link))
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.create()
        self.write_config()
        directory = self.root / "directory"
        directory.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            controller.create(self.config, directory / "campaign")

    def test_frozen_corruption_fails_closed(self):
        self.create()
        artifact = self.campaign / "bin" / "trainer"
        artifact.chmod(0o600)
        artifact.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            controller.status(self.campaign)

    def test_status_rejects_progress_without_verified_receipt(self):
        self.create()
        path = self.campaign / "status.json"
        value = json.loads(path.read_text())
        value["accepted_updates"] = 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "unsupported or corrupt status"):
            controller.status(self.campaign)

    def test_manifest_path_escape_is_rejected(self):
        self.create()
        path = self.campaign / "manifest.json"
        path.chmod(0o600)
        value = json.loads(path.read_text())
        value["files"]["../trainer"] = value["files"].pop("bin/trainer")
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "manifest integrity"):
            controller.status(self.campaign)

    def test_adoption_never_modifies_campaign(self):
        self.create()
        before = (self.campaign / "status.json").read_bytes()
        for operation in ("adopt",):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    controller.main([operation, str(self.campaign)])
        self.assertEqual((self.campaign / "status.json").read_bytes(), before)

    def test_invocation_budget_clips_at_history_and_total_boundaries(self):
        self.assertEqual(controller.invocation_budget(19, 100, 16, 20), 1)
        self.assertEqual(controller.invocation_budget(98, 100, 16, 20), 2)
        self.assertEqual(controller.invocation_budget(100, 100, 2, 20), 0)
        with self.assertRaisesRegex(ValueError, "accepted updates"):
            controller.invocation_budget(101, 100, 2, 20)


NATIVE_CONTRACT = {
    "schema": "drysua-checkpoint-inspection/v1", "kind": "contract",
    "model": {"version": 25, "hash": "1234567890abcdef", "parameters": 1812983},
    "enabled_features": "builtin,cuda",
    "capabilities": {"inspection": True, "annealed_history": True, "read_only": True,
                     "strict_build_features": True, "controller_run_kind": "train-annealed"},
    "limits": {"max_json_bytes": 4194304, "max_snapshots": 10000, "max_files": 10004,
               "manifest_bytes": 65536, "training_tensor_bytes": 21821332,
               "runtime_tensor_bytes": 7268316, "snapshot_bytes": 4096,
               "max_samples": 46520, "max_games": 40},
    "schemas": {"action": {"version": 5, "hash": "1" * 16},
                "feature": {"version": 22, "hash": "2" * 16},
                "reward": {"version": 7, "hash": "3" * 16},
                "ppo": {"version": 40, "hash": "4" * 16}, "rules_audit_version": 32},
    "checkpoint": {"version": 18, "hash": "5" * 16},
    "numeric_semantics": {"ppo_floats": "IEEE-754 binary32, JSON numbers widened exactly to binary64",
                          "adaptive_rate_units": 1000000,
                          "schema_hash": "lowercase hexadecimal FNV-1a u64 (16 digits)",
                          "file_hash": "lowercase SHA-256 (64 digits)"},
}

INSPECTOR = '''#!{python}
import json, sys
from pathlib import Path
if sys.argv[1:] == ["checkpoint-inspect", "--help"]:
    print("checkpoint-inspect --checkpoint-directory DIR")
    sys.exit(0)
if sys.argv[1:] == ["checkpoint-inspect", "--contract"]:
    print(CONTRACT_JSON)
    sys.exit(0)
assert sys.argv[1:3] == ["checkpoint-inspect", "--checkpoint-directory"]
print((Path(sys.argv[3]) / "inspection.json").read_text())
'''

RUNNER = '''import hashlib, json, struct, sys
from pathlib import Path
scenario = SCENARIO
# update, start_update, previous_awards, clean: generation = index, one transition each.
ADAPTIVE_TRANSITIONS = [(1, 1, 1, "true"), (4, 4, 0, "false"), (5, 5, 2, "false")]
assert sys.argv[1:3] == ["run", "--spec"]
spec = json.loads(Path(sys.argv[3]).read_text())
assert spec["version"] == 1
assert set(spec) == {"version", "workspace_root", "campaign_directory", "job_directory",
                     "command", "mode", "docker_context", "image", "gpu_uuid",
                      "lock_paths", "campaign_id", "invocation_id", "invocation_seconds"}
job = Path(spec["job_directory"])
command = spec["command"]
checkpoint = Path(command[command.index("--checkpoint-directory") + 1])
count = int(command[command.index("--invocation-updates") + 1])
previous = 0
if "--resume" in command:
    previous = json.loads((checkpoint / "inspection.json").read_text())["progress"]["updates"]
updates = previous + count + (1 if scenario == "wrong_updates" else 0)
data = str(updates).encode()
tensor_bytes = data + b"tensor"
tensor_name = "checkpoint." + hashlib.sha256(tensor_bytes).hexdigest() + ".safetensors"
artifacts = {"manifest_sha256": "checkpoint.meta", "tensor_sha256": tensor_name,
             "runtime_sha256": "drysua.weights.safetensors"}
identity = {"scope_sha256": "a" * 64}
files = []
for field, name in artifacts.items():
    contents = tensor_bytes if field == "tensor_sha256" else data + name.encode()
    (checkpoint / name).write_bytes(contents)
    identity[field] = hashlib.sha256(contents).hexdigest()
    files.append({"path": name, "size": len(contents), "sha256": identity[field]})
if scenario == "changed_scope" and previous: identity["scope_sha256"] = "b" * 64
inspection = {"schema": "drysua-checkpoint-inspection/v1", "kind": "train-annealed",
              "identity": identity, "model": {"version": 25, "hash": "1234567890abcdef", "parameters": 1812983},
              "checkpoint": {"version": 18, "hash": "5" * 16},
              "progress": {"updates": updates, "optimizer_steps": updates * 3,
                           "rollout_samples": updates * 40, "games": updates * 2,
                           "policy_version": updates, "scheduler_step": updates, "curriculum_stage": 0,
                           "best_evaluation": 0.0, "rng_states": [{"name": "actor", "state": 10, "draws": updates}],
                           "shuffle_rng": {"state": 20, "draws": updates}, "league_references": []},
              "run": {"git_commit": "a" * 40, "simulator_commit": "b" * 40,
                      "enabled_features": "builtin,cuda", "command_line": "train-annealed --games 2",
                      "run_seed": 9001, "map": 2, "hero": 11, "device": {"kind": "cpu", "ordinal": None},
                      "batch_size": 2, "rules_audit_version": 32},
              "ppo": {"schema_version": 40,
                      "schema_hash": "4" * 16, "decision_interval_ticks": 24, "rollout_decisions": 1163,
                      "environments": 2, "epochs": 1, "minibatch": 256, "clip_epsilon": 0.2,
                      "value_coefficient": 0.5, "entropy_coefficient": 0.01, "learning_rate": 0.0003,
                      "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
                      "gradient_clip": 0.5, "gamma_tick": 1.0, "gae_lambda": 1.0, "target_kl": 0.01,
                      "f32_bits": {}}, "adaptive": None,
              "history": {"kind": "fixed", "verified": True, "snapshot_count": 0},
              "runtime_status": "matched", "recovery_required": False,
              "sources": {"manifest": "checkpoint.meta", "tensor": tensor_name,
                          "runtime": "drysua.weights.safetensors", "canonical_tensor_matches": False},
              "files": files,
              "runtime_matches_model": True}
for name in ("clip_epsilon", "value_coefficient", "entropy_coefficient", "learning_rate", "adam_beta1",
             "adam_beta2", "adam_epsilon", "gradient_clip", "gamma_tick", "gae_lambda", "target_kl"):
    encoded = struct.pack("<f", inspection["ppo"][name])
    inspection["ppo"]["f32_bits"][name] = struct.unpack("<I", encoded)[0]
    inspection["ppo"][name] = struct.unpack("<f", encoded)[0]
if scenario == "wrong_runtime": inspection["runtime_matches_model"] = False
if scenario == "adaptive":
    inspection["adaptive"] = {
        "config": {"extension_units": 750000, "poor_rate_units": 200000, "poor_updates": 1,
                   "success_rate_units": 800000, "success_updates": 2},
        "limits": {"base_updates": 2, "total_updates": 5, "zero_updates": 1},
        "snapshot_count": 0, "snapshot_hash": "0" * 64,
        "state": {"extension_awards": 0,
                  "generation": sum(1 for entry in ADAPTIVE_TRANSITIONS if entry[0] <= updates),
                  "poor_streak": 0, "success_streak": 0,
                  "start_update": max([entry[1] for entry in ADAPTIVE_TRANSITIONS
                                       if entry[0] <= updates], default=0),
                  "updates_in_generation": updates - max([entry[1] for entry in ADAPTIVE_TRANSITIONS
                                                          if entry[0] <= updates], default=0)}}
(checkpoint / "inspection.json").write_text(json.dumps(inspection))
episodes = ["episode: stream=%d map=2 opponent=Policy tick=%d outcome=%s actor_decisions=10 retained=2 terminal_sample=true"
            % (index, 1000 + index, "Win" if index % 2 == 0 else "Loss") for index in range(count * 2)]
if scenario == "short_payload":
    episodes = episodes[:-1]
if scenario == "adaptive":
    for index, (update_at, start_at, awards, clean) in enumerate(ADAPTIVE_TRANSITIONS, start=1):
        if previous < update_at <= updates:
            episodes.append(
                "level=INFO event=adaptive_environment_transition update=%d previous_generation=%d "
                "generation=%d start_update=%d previous_awards=%d clean=%s"
                % (update_at, index - 1, index, start_at, awards, clean))
(job / "payload.log").write_text("\\n".join(episodes) + "\\n")
result = {"schema": "drysua-training-runner/v1", "returncode": 0,
          "container_state": {"Running": scenario == "running", "OOMKilled": scenario == "oom",
                              "ExitCode": 0,
                              "StartedAt": "2026-01-01T00:00:00.000000000Z",
                              "FinishedAt": "2026-01-01T00:00:30.000000000Z"}, "container_id": "b" * 64}
if scenario != "missing_result": (job / "result.json").write_text(json.dumps(result))
if scenario == "pause":
    campaign = Path(spec["campaign_directory"])
    owner = json.loads((campaign / "owner.json").read_text())
    (campaign / ("pause-" + owner["token"] + ".json")).write_text(json.dumps({"token": owner["token"]}))
'''


def runner_fixture(scenario):
    helpers = ("from dataclasses import dataclass\n"
               f"DEFAULT_INVOCATION_SECONDS = {train_runner.DEFAULT_INVOCATION_SECONDS!r}\n"
               f"MAX_INVOCATION_SECONDS = {train_runner.MAX_INVOCATION_SECONDS!r}\n"
               + inspect.getsource(train_runner.InvocationTimeouts) + "\n"
               + inspect.getsource(train_runner.resolve_timeouts))
    return helpers + "\nif __name__ == '__main__':\n" + "\n".join(
        "    " + line for line in RUNNER.replace("SCENARIO", repr(scenario)).splitlines()) + "\n"


class LifecycleTests(unittest.TestCase):
    write_config = CampaignTests.write_config
    create = CampaignTests.create

    def setUp(self):
        CampaignTests.setUp(self)
        self.trainer.write_text(INSPECTOR.format(python=sys.executable).replace(
            "CONTRACT_JSON", repr(json.dumps(NATIVE_CONTRACT))))
        self.trainer.chmod(0o700)
        self.runner = self.root / "train_runner.py"
        self.runner.write_text(runner_fixture("success"))
        self.runner_patch = patch.object(controller, "RUNNER_SOURCE", self.runner)
        self.runner_patch.start()
        self.addCleanup(self.runner_patch.stop)
        self.lock = self.root / "heavy.lock"
        self.lock.touch()
        self.write_config(total_updates=5, invocation_updates=2, history_every=3, workspace_root=str(self.root),
                          image="fixture@sha256:" + "c" * 64, lock_paths=[str(self.lock)])

    def scenario(self, value):
        self.runner.write_text(runner_fixture(value))

    def configure(self, **updates):
        value = json.loads(self.config.read_text())
        value.update(updates)
        self.config.write_text(json.dumps(value))

    def test_generated_deployment_spec_is_accepted_by_actual_runner_load_spec(self):
        native = importlib.util.spec_from_file_location("actual_train_runner", SCRIPT.with_name("train_runner.py"))
        runner = importlib.util.module_from_spec(native)
        sys.modules[native.name] = runner
        self.addCleanup(sys.modules.pop, native.name)
        native.loader.exec_module(runner)
        runtime = self.root / "drysua.weights.safetensors"
        runtime.write_bytes(b"inert runtime fixture")
        self.trainer.write_bytes(b"\x7fELFinert header fixture, never executed")
        deployment = controller.read_json(SCRIPT.parents[1] / "docs/training_controller.example.json")
        self.configure(total_updates=200, invocation_updates=1, initial_weights=str(runtime),
                       opponent_weights=str(runtime), mode="gpu",
                       gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                       training_args=deployment["training_args"])
        current = self.create()
        manifest = controller.read_json(self.campaign / "manifest.json")
        job = controller.job_path(self.campaign, 1)
        job.mkdir(mode=0o700)
        (job / "checkpoint").mkdir(mode=0o700)
        spec, expected = controller.job_spec(self.campaign, current, manifest)
        controller.write_exclusive(job / "job.json", controller.encode(spec))
        self.assertEqual(runner.load_spec(job / "job.json"), spec)
        self.assertEqual(expected, 1)
        command = spec["command"]
        self.assertEqual(command[command.index("--opponent") + 1], "weights")
        self.assertEqual(command[command.index("--opponent-weights") + 1], str(self.campaign / "inputs/opponent"))
        self.assertEqual(command[command.index("--initial-weights") + 1], str(self.campaign / "inputs/initial_weights"))
        self.assertEqual(command[command.index("--device-ordinal") + 1], "0")
        self.assertNotIn("--resume", command)

    def test_initial_and_opponent_inputs_freeze_only_runtime_not_checkpoint_state(self):
        source = self.root / "completed"
        source.mkdir()
        (source / "drysua.weights.safetensors").write_bytes(b"runtime")
        (source / "checkpoint.meta").write_bytes(b"old optimizer state")
        (source / "unrelated-link").symlink_to(self.root)
        self.configure(initial_weights=str(source), opponent_weights=str(source))
        self.create()
        for name in ("initial_weights", "opponent"):
            directory = self.campaign / "inputs" / name
            self.assertEqual([path.name for path in directory.iterdir()], ["drysua.weights.safetensors"])
            self.assertEqual((directory / "drysua.weights.safetensors").read_bytes(), b"runtime")

    def test_opponent_override_arguments_are_rejected(self):
        for arguments in (["--opponent", "teacher"], ["--opponent=weights"],
                          ["--opponent-weights", "/unowned"]):
            with self.subTest(arguments=arguments):
                self.configure(training_args=arguments)
                with self.assertRaisesRegex(ValueError, "controller-owned or unreviewed"):
                    self.create()
                self.assertFalse(self.campaign.exists())

    def test_environment_scale_arguments_are_allowed_and_reach_the_command(self):
        self.configure(total_updates=1, training_args=[
            "--environment-scale-start", "0", "--environment-scale-end", "2"])
        current = self.create()
        manifest = controller.read_json(self.campaign / "manifest.json")
        job = controller.job_path(self.campaign, 1)
        job.mkdir(mode=0o700)
        (job / "checkpoint").mkdir(mode=0o700)
        spec, expected = controller.job_spec(self.campaign, current, manifest)
        command = spec["command"]
        self.assertEqual(expected, 1)
        self.assertEqual(command[command.index("--environment-scale-start") + 1], "0")
        self.assertEqual(command[command.index("--environment-scale-end") + 1], "2")

    def test_selected_immutable_tensor_source_does_not_require_canonical_alias(self):
        self.configure(total_updates=1)
        self.create()
        controller.run_campaign(self.campaign)
        receipt = controller.read_json(self.campaign / "invocations/000001/accepted.json")
        source = receipt["inspection"]["sources"]["tensor"]
        self.assertNotEqual(source, "checkpoint.safetensors")
        self.assertFalse((self.campaign / "invocations/000001/checkpoint/checkpoint.safetensors").exists())
        self.assertEqual(receipt["inspection"]["identity"]["tensor_sha256"],
                         receipt["checkpoint_files"][source]["sha256"])

    def test_unverified_history_or_native_recovery_required_cannot_be_accepted(self):
        self.configure(total_updates=1)
        self.create()
        controller.run_campaign(self.campaign)
        checkpoint = self.campaign / "invocations/000001/checkpoint"
        value = controller.read_json(checkpoint / "inspection.json")
        for replacement, message in (({"history": {"kind": "fixed", "verified": False, "snapshot_count": 0}}, "history"),
                                     ({"recovery_required": True}, "recovery"),
                                     ({"kind": "train-full"}, "kind"),
                                     ({"runtime_status": "missing"}, "runtime")):
            with self.subTest(replacement=replacement), patch.object(controller.state, "capture", return_value=
                    controller.encode(dict(value, **replacement))):
                with self.assertRaisesRegex(ValueError, message):
                    controller.inspect_checkpoint(self.campaign, checkpoint, 1)

    def test_checkpoint_inventory_supports_native_history_beyond_frozen_input_limit(self):
        directory = self.root / "history"
        directory.mkdir()
        for number in range(129):
            (directory / f"snapshot-{number}").write_bytes(b"fixture")
        inventory = controller.checkpoint_inventory(directory)
        self.assertEqual(len(inventory), 129)
        controller.verify_inventory(directory, inventory, exact=True)

    def test_successive_updates_change_artifact_hashes_but_preserve_scope(self):
        self.create()
        controller.run_campaign(self.campaign)
        receipts = [json.loads(path.read_text()) for path in
                    sorted(self.campaign.glob("invocations/*/accepted.json"))]
        identities = [receipt["inspection"]["identity"] for receipt in receipts]
        self.assertEqual(len(identities), 3)
        self.assertEqual({identity["scope_sha256"] for identity in identities}, {"a" * 64})
        for field in ("manifest_sha256", "tensor_sha256", "runtime_sha256"):
            with self.subTest(field=field):
                self.assertEqual(len({identity[field] for identity in identities}), 3)

    def test_changed_native_scope_rejects_next_update_without_losing_accepted_progress(self):
        self.scenario("changed_scope")
        self.create()
        with self.assertRaisesRegex(ValueError, "scope changed"):
            controller.run_campaign(self.campaign)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 2)

    def test_native_artifact_identity_must_match_inspected_file_hash(self):
        self.configure(total_updates=1)
        self.create()
        controller.run_campaign(self.campaign)
        checkpoint = self.campaign / "invocations/000001/checkpoint"
        value = json.loads((checkpoint / "inspection.json").read_text())
        for field in ("manifest_sha256", "tensor_sha256", "runtime_sha256"):
            modified = dict(value, identity=dict(value["identity"], **{field: "f" * 64}))
            with self.subTest(field=field), patch.object(controller.state, "capture", return_value=
                    json.dumps(modified).encode()):
                with self.assertRaisesRegex(ValueError, "artifact identity mismatch"):
                    controller.inspect_checkpoint(self.campaign, checkpoint, 1)

    def test_model_run_and_ppo_must_remain_stable(self):
        self.configure(total_updates=1)
        self.create()
        controller.run_campaign(self.campaign)
        checkpoint = self.campaign / "invocations/000001/checkpoint"
        previous = json.loads((checkpoint / "inspection.json").read_text())
        for field in ("model", "run", "ppo"):
            value = dict(previous, **{field: {"changed": True}})
            with self.subTest(field=field), patch.object(controller.state, "capture", return_value=
                    json.dumps(value).encode()):
                with self.assertRaisesRegex(ValueError, f"native checkpoint {field} changed"):
                    controller.inspect_checkpoint(self.campaign, checkpoint, 1, previous)

    def test_session_deadline_is_shared_by_jobs_and_reset_only_on_explicit_resume(self):
        self.configure(max_seconds=600)
        self.create()
        clock = Mock(return_value=0.0)
        original = controller.state.wait_runner
        launches = []

        def advance_clock(command, timeout, stop_requested, record_child, controller_reserve_seconds):
            self.assertEqual(timeout, 260)
            launches.append(command)
            original(command, timeout, stop_requested, record_child, controller_reserve_seconds)
            clock.return_value += 260

        with patch.object(controller, "monotonic", clock, create=True), \
                patch.object(controller.state, "wait_runner", side_effect=advance_clock):
            current = controller.run_campaign(self.campaign)
            self.assertEqual((current["phase"], current["accepted_updates"]), ("paused", 3))
            self.assertIn("session deadline", current["last_error"])
            self.assertEqual(len(launches), 2)
            clock.return_value = 1000
            current = controller.run_campaign(self.campaign, resume=True)
        self.assertEqual((current["phase"], current["accepted_updates"]), ("completed", 5))

    def test_insufficient_session_reserve_prevents_first_job(self):
        self.configure(max_seconds=260)
        self.create()
        with patch.object(controller, "monotonic", return_value=0.0, create=True), \
                patch.object(controller.state, "wait_runner") as launch:
            current = controller.run_campaign(self.campaign)
        self.assertEqual(current["phase"], "paused")
        self.assertIn("session deadline", current["last_error"])
        launch.assert_not_called()
        self.assertEqual(list((self.campaign / "invocations").iterdir()), [])

    def test_preflight_time_consumes_session_deadline(self):
        self.configure(max_seconds=300)
        self.create()
        clock = Mock(return_value=0.0)
        original = controller.preflight

        def slow_preflight(*arguments, **keywords):
            result = original(*arguments, **keywords)
            clock.return_value = 40.0
            return result

        with patch.object(controller, "monotonic", clock, create=True), \
                patch.object(controller, "preflight", side_effect=slow_preflight), \
                patch.object(controller.state, "wait_runner") as launch:
            current = controller.run_campaign(self.campaign)
        self.assertEqual(current["phase"], "paused")
        launch.assert_not_called()

    def test_deadline_is_rechecked_after_checkpoint_preparation_before_launch(self):
        self.configure(max_seconds=300)
        self.create()
        clock = Mock(return_value=0.0)
        original = controller.make_job

        def slow_preparation(*arguments):
            result = original(*arguments)
            clock.return_value = 20.0
            return result

        with patch.object(controller, "monotonic", clock), \
                patch.object(controller, "make_job", side_effect=slow_preparation), \
                patch.object(controller.state, "wait_runner") as launch:
            with self.assertRaisesRegex(ValueError, "session deadline reserve exhausted during job preparation"):
                controller.run_campaign(self.campaign)
        launch.assert_not_called()
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_inspector_timeout_is_clipped_to_remaining_session_budget(self):
        self.configure(max_seconds=600)
        self.create()
        clock = Mock(return_value=0.0)
        original = controller.state.wait_runner

        def consume_budget(*arguments):
            original(*arguments)
            clock.return_value = 595.0

        with patch.object(controller, "monotonic", clock), \
                patch.object(controller.state, "wait_runner", side_effect=consume_budget), \
                patch.object(controller.state, "capture", wraps=controller.state.capture) as inspector:
            current = controller.run_campaign(self.campaign)
        self.assertEqual(inspector.call_args.kwargs["timeout"], 5.0)
        self.assertEqual((current["phase"], current["accepted_updates"]), ("paused", 2))

    def test_disk_admission_accounts_retained_bytes_and_reserves_next_job(self):
        self.create()
        with patch.object(controller, "campaign_disk_usage", return_value=99 * 1024**3, create=True), \
                patch.object(controller.state, "wait_runner") as launch:
            current = controller.run_campaign(self.campaign)
        self.assertEqual(current["phase"], "paused")
        self.assertIn("campaign disk reserve", current["last_error"])
        launch.assert_not_called()
        self.assertEqual(list((self.campaign / "invocations").iterdir()), [])

    def test_disk_inventory_counts_checkpoints_logs_and_frozen_inputs(self):
        disk = self.root / "disk"
        disk.mkdir()
        for name in ("invocations/000001/checkpoint", "invocations/000001/logs", "inputs"):
            directory = disk / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "payload").write_bytes(b"x" * 16384)
        self.assertGreaterEqual(controller.campaign_disk_usage(disk), 3 * 16384)
        with patch.object(controller, "MAX_CAMPAIGN_BYTES", 3 * 16384 - 1):
            with self.assertRaisesRegex(ValueError, "campaign disk limit"):
                controller.campaign_disk_usage(disk)

    def test_disk_inventory_rejects_symlinks_and_excess_entries(self):
        disk = self.root / "disk"
        disk.mkdir()
        (disk / "escape").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            controller.campaign_disk_usage(disk)
        (disk / "escape").unlink()
        for index in range(4):
            (disk / str(index)).touch()
        with patch.object(controller, "MAX_CAMPAIGN_ENTRIES", 3):
            with self.assertRaisesRegex(ValueError, "disk inventory entry limit"):
                controller.campaign_disk_usage(disk)

    def test_disk_inventory_charges_root_directory_even_when_empty(self):
        disk = self.root / "disk"
        disk.mkdir()
        with patch.object(controller, "MAX_CAMPAIGN_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "campaign disk limit"):
                controller.campaign_disk_usage(disk)

    def test_disk_admission_is_rechecked_after_retaining_a_checkpoint(self):
        self.create()
        original = controller.campaign_disk_usage

        def usage(directory, **keywords):
            if directory == self.campaign:
                current = json.loads((directory / "status.json").read_text())
                if current["accepted_updates"]:
                    return 99 * 1024**3
            return original(directory, **keywords)

        with patch.object(controller, "campaign_disk_usage", side_effect=usage):
            current = controller.run_campaign(self.campaign)
        self.assertEqual((current["phase"], current["accepted_updates"]), ("paused", 2))
        self.assertIn("campaign disk reserve", current["last_error"])
        self.assertEqual(len(list((self.campaign / "invocations").iterdir())), 1)

    def test_active_job_disk_overrun_terminates_only_owned_child(self):
        self.create()
        original_usage = controller.campaign_disk_usage
        original_wait = controller.state.wait_runner
        child = Mock(pid=os.getpid(), returncode=None)
        child.poll.return_value = None
        job_samples = []

        def usage(directory, **keywords):
            if "entry_limit" in keywords:
                job_samples.append(directory)
                return 0 if len(job_samples) == 1 else controller.MAX_CAMPAIGN_BYTES
            return original_usage(directory, **keywords)

        def wait(*arguments):
            with patch.object(controller.state.subprocess, "Popen", return_value=child):
                return original_wait(*arguments)

        with patch.object(controller, "campaign_disk_usage", side_effect=usage), \
                patch.object(controller.state, "wait_runner", side_effect=wait):
            with self.assertRaisesRegex(ValueError, "campaign disk limit reached by active invocation"):
                controller.run_campaign(self.campaign)
        child.terminate.assert_called_once_with()
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_help_only_inspector_is_explicitly_unsupported_before_runner(self):
        self.trainer.write_text(f"#!{sys.executable}\nimport sys\n"
                                "if '--help' in sys.argv:\n"
                                "    print('checkpoint-inspect --checkpoint-directory DIR')\n"
                                "    sys.exit(0)\nraise SystemExit(2)\n")
        self.create()
        with patch.object(controller.state, "wait_runner") as launch:
            with self.assertRaisesRegex(ValueError, "inspector --contract unsupported"):
                controller.run_campaign(self.campaign)
        launch.assert_not_called()

    def test_unknown_inspector_contract_is_rejected_before_runner(self):
        self.create()
        with patch.object(controller.state, "capture", return_value=b'{"schema":"future/v2"}'), \
                patch.object(controller.state, "wait_runner") as launch:
            with self.assertRaisesRegex(ValueError, "unsupported native inspector contract"):
                controller.run_campaign(self.campaign)
        launch.assert_not_called()

    def test_inspector_preflight_requests_contract_not_help(self):
        self.configure(total_updates=1)
        self.create()
        with patch.object(controller.state, "capture", wraps=controller.state.capture) as inspector:
            controller.run_campaign(self.campaign)
        self.assertEqual(inspector.call_args_list[0].args[0][1:], ["checkpoint-inspect", "--contract"])

    def test_run_accepts_exact_native_progress_and_immutable_receipts(self):
        self.create()
        result = controller.run_campaign(self.campaign)
        self.assertEqual(result["phase"], "completed")
        self.assertEqual(result["accepted_updates"], 5)
        receipts = [json.loads(path.read_text()) for path in
                    sorted(self.campaign.glob("invocations/*/accepted.json"))]
        self.assertEqual([entry["inspection"]["progress"]["updates"] for entry in receipts], [2, 3, 5])
        self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in
                            self.campaign.glob("invocations/*/accepted.json")))
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 5)

    def test_bad_runner_or_checkpoint_never_advances_and_never_retries(self):
        for scenario in ("wrong_updates", "wrong_runtime", "running", "oom", "missing_result"):
            with self.subTest(scenario=scenario):
                self.campaign = self.root / scenario
                self.scenario(scenario)
                self.create()
                with self.assertRaises(ValueError):
                    controller.run_campaign(self.campaign)
                current = controller.status(self.campaign)
                self.assertEqual(current["accepted_updates"], 0)
                self.assertEqual(current["phase"], "failed")
                self.assertEqual(len(list((self.campaign / "invocations").iterdir())), 1)
                with self.assertRaisesRegex(ValueError, "recover"):
                    controller.run_campaign(self.campaign)

    def test_inspector_missing_contract_fails_before_runner(self):
        self.trainer.write_text(f"#!{sys.executable}\nraise SystemExit(2)\n")
        self.create()
        with self.assertRaisesRegex(ValueError, "inspector"):
            controller.run_campaign(self.campaign)
        self.assertEqual(list((self.campaign / "invocations").iterdir()), [])

    def test_pause_at_boundary_and_explicit_resume(self):
        self.scenario("pause")
        self.create()
        result = controller.run_campaign(self.campaign)
        self.assertEqual((result["phase"], result["accepted_updates"]), ("paused", 2))
        with self.assertRaisesRegex(ValueError, "resume"):
            controller.run_campaign(self.campaign)
        result = controller.run_campaign(self.campaign, resume=True)
        self.assertEqual((result["phase"], result["accepted_updates"]), ("paused", 3))

    def test_stale_and_live_owners_are_not_implicitly_stolen(self):
        self.create()
        owner = controller.state.process_identity(os.getpid())
        owner.update(token="d" * 32, child=None)
        (self.campaign / "owner.json").write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "owner"):
            controller.run_campaign(self.campaign)
        with self.assertRaisesRegex(ValueError, "live owner"):
            controller.recover(self.campaign, confirm=True)
        owner["pid"] = 2147483647
        (self.campaign / "owner.json").write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "recover"):
            controller.run_campaign(self.campaign)
        with self.assertRaisesRegex(ValueError, "explicit"):
            controller.recover(self.campaign)
        self.assertEqual(controller.recover(self.campaign, confirm=True)["phase"], "paused")

    def test_control_refuses_live_pid_without_controller_lock(self):
        self.create()
        owner = dict(controller.state.process_identity(os.getpid()), token="d" * 32, child=None)
        (self.campaign / "owner.json").write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "no live owner"):
            controller.request_control(self.campaign, "stop")

    def test_boundary_pause_cleans_only_its_own_control_requests(self):
        self.scenario("pause")
        self.create()
        controller.run_campaign(self.campaign)
        self.assertEqual(list(self.campaign.glob("pause-*.json")), [])

    def test_detach_returns_only_after_durable_owner_handshake(self):
        self.scenario("pause")
        self.create()
        children = []
        with patch.object(controller.state, "reap_detached", side_effect=children.append):
            result = controller.detach(self.campaign)
        self.assertEqual(result["phase"], "started")
        # The bounded wait is process synchronization, not a timing assertion.
        self.assertEqual(children[0].wait(timeout=5), 0)
        self.assertEqual(controller.status(self.campaign)["phase"], "paused")

    def test_receipt_corruption_is_detected(self):
        self.create()
        controller.run_campaign(self.campaign)
        path = self.campaign / "invocations/000001/accepted.json"
        path.chmod(0o600)
        value = json.loads(path.read_text())
        value["expected_updates"] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "receipt"):
            controller.status(self.campaign)

    def test_changed_frozen_runner_is_never_launched_again(self):
        self.create()
        original = controller.state.wait_runner
        launches = []

        def mutate_after_run(*arguments):
            launches.append(arguments[0])
            original(*arguments)
            path = self.campaign / "frozen/train_runner.py"
            path.chmod(0o600)
            path.write_text("raise SystemExit(0)\n")

        with patch.object(controller.state, "wait_runner", side_effect=mutate_after_run):
            with self.assertRaisesRegex(ValueError, "frozen artifact integrity"):
                controller.run_campaign(self.campaign)
        self.assertEqual(len(launches), 1)

    def test_bounded_inspector_output_is_rejected(self):
        self.trainer.write_text(f"#!{sys.executable}\nprint('x' * (5 * 1024 * 1024))\n")
        self.create()
        with self.assertRaisesRegex(ValueError, "output limit"):
            controller.run_campaign(self.campaign)
        self.assertEqual(list((self.campaign / "invocations").iterdir()), [])

    def interrupted_after_receipt(self):
        self.create()
        original = controller.atomic_status

        def interrupt(directory, value):
            if value["accepted_updates"]:
                raise KeyboardInterrupt("simulated process crash after receipt fsync")
            return original(directory, value)

        with patch.object(controller, "atomic_status", side_effect=interrupt):
            with self.assertRaisesRegex(KeyboardInterrupt, "simulated process crash"):
                controller.run_campaign(self.campaign)
        owner_path = self.campaign / "owner.json"
        owner = json.loads(owner_path.read_text())
        owner["pid"] = 2147483647
        owner_path.write_text(json.dumps(owner))

    def test_recover_accepts_durable_receipt_once_without_retraining(self):
        self.interrupted_after_receipt()
        receipt = self.campaign / "invocations/000001/accepted.json"
        before = receipt.read_bytes()
        recovered = controller.recover(self.campaign, confirm=True)
        self.assertEqual((recovered["phase"], recovered["accepted_updates"]), ("paused", 2))
        self.assertEqual(receipt.read_bytes(), before)
        self.assertEqual(len(list((self.campaign / "invocations").iterdir())), 1)
        result = controller.run_campaign(self.campaign, resume=True)
        self.assertEqual(result["accepted_updates"], 5)

    def test_recover_refuses_busy_shared_lock(self):
        self.interrupted_after_receipt()
        with controller.state.locks([self.lock]):
            with self.assertRaisesRegex(ValueError, "lock busy"):
                controller.recover(self.campaign, confirm=True)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_recover_refuses_pending_job_without_recorded_child_identity(self):
        self.interrupted_after_receipt()
        path = self.campaign / "owner.json"
        owner = json.loads(path.read_text())
        owner["child"] = None
        path.write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "child identity"):
            controller.recover(self.campaign, confirm=True)

    def test_recover_refuses_live_child_or_running_container(self):
        self.interrupted_after_receipt()
        owner_path = self.campaign / "owner.json"
        owner = json.loads(owner_path.read_text())
        original_child = owner["child"]
        owner["child"] = controller.state.process_identity(os.getpid())
        owner_path.write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, "live owner or owned child"):
            controller.recover(self.campaign, confirm=True)
        owner["child"] = original_child
        owner_path.write_text(json.dumps(owner))
        result_path = self.campaign / "invocations/000001/result.json"
        result = json.loads(result_path.read_text())
        result["container_state"]["Running"] = True
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "container stop is unconfirmed"):
            controller.recover(self.campaign, confirm=True)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_detach_failure_reports_no_success_handshake(self):
        self.trainer.write_text(f"#!{sys.executable}\nraise SystemExit(2)\n")
        self.create()
        with self.assertRaisesRegex(ValueError, "failed before durable ownership handshake"):
            controller.detach(self.campaign)
        self.assertEqual(controller.status(self.campaign)["phase"], "prepared")

    def test_inspector_unknown_schema_and_duplicate_keys_fail_closed(self):
        self.create()
        for payload, message in ((b'{"schema":"future"}', "unsupported native inspector schema"),
                                 (b'{"schema":1,"schema":1}', "duplicate JSON key")):
            with self.subTest(payload=payload), patch.object(controller.state, "capture", return_value=payload):
                with self.assertRaisesRegex(ValueError, message):
                    controller.inspect_checkpoint(self.campaign, self.campaign, 1)

    def test_invalid_native_identity_hash_is_rejected(self):
        self.create()
        controller.run_campaign(self.campaign)
        checkpoint = self.campaign / "invocations/000003/checkpoint"
        value = json.loads((checkpoint / "inspection.json").read_text())
        value["identity"]["tensor_sha256"] = "not a hash"
        with patch.object(controller.state, "capture", return_value=json.dumps(value).encode()):
            with self.assertRaisesRegex(ValueError, "native identity hash"):
                controller.inspect_checkpoint(self.campaign, checkpoint, 5)

    def test_active_owner_lock_cannot_be_taken_by_another_controller(self):
        self.create()
        with controller.state.locks([self.campaign / "owner.lock"]):
            with self.assertRaisesRegex(ValueError, "live owner or lock busy"):
                controller.run_campaign(self.campaign)
        self.assertEqual(controller.status(self.campaign)["phase"], "prepared")

    def test_resume_inspects_accepted_checkpoint_before_another_runner(self):
        self.scenario("pause")
        self.create()
        controller.run_campaign(self.campaign)
        directory = self.campaign / "invocations/000001/checkpoint"
        checkpoint = directory / json.loads((directory / "inspection.json").read_text())["sources"]["tensor"]
        checkpoint.chmod(0o600)
        checkpoint.write_bytes(b"modified")
        with patch.object(controller.state, "wait_runner") as launch:
            with self.assertRaisesRegex(ValueError, "checkpoint file integrity"):
                controller.run_campaign(self.campaign, resume=True)
        launch.assert_not_called()

    def test_recover_rejects_modified_spec_before_checkpoint_acceptance(self):
        self.interrupted_after_receipt()
        job = self.campaign / "invocations/000001"
        (job / "accepted.json").unlink()
        spec_path = job / "job.json"
        spec_path.chmod(0o600)
        spec = json.loads(spec_path.read_text())
        spec["command"][0] = "/unowned/binary"
        spec_path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "spec"):
            controller.recover(self.campaign, confirm=True)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_stop_request_is_consumed_only_by_the_owner(self):
        self.create()
        original = controller.state.wait_runner

        def request_then_run(command, timeout, stop_requested, record_child, controller_reserve_seconds):
            response = controller.request_control(self.campaign, "stop")
            self.assertEqual(response["operation"], "stop")
            return original(command, timeout, stop_requested, record_child, controller_reserve_seconds)

        with patch.object(controller.state, "wait_runner", side_effect=request_then_run):
            with self.assertRaisesRegex(ValueError, "immediate stop"):
                controller.run_campaign(self.campaign)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 0)

    def test_immediate_stop_signals_only_the_popen_child(self):
        child = Mock(pid=os.getpid(), returncode=None)
        child.poll.return_value = None
        with patch.object(controller.state.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(ValueError, "immediate stop"):
                controller.state.wait_runner(["fake"], 10, Mock(side_effect=[False, True]), lambda _: None, 40)
        child.terminate.assert_called_once_with()
        child.kill.assert_not_called()

    def test_runner_deadline_stops_owned_child_without_waiting_for_wall_clock(self):
        child = Mock(pid=os.getpid(), returncode=None)
        child.poll.return_value = None
        with patch.object(controller.state.subprocess, "Popen", return_value=child), \
                patch.object(controller.state.time, "monotonic", side_effect=[0, 11, 11]):
            with self.assertRaisesRegex(ValueError, "runner deadline exceeded"):
                controller.state.wait_runner(["fake"], 10, lambda: False, lambda _: None, 40)
        child.terminate.assert_called_once_with()

    def test_handshake_cannot_override_standard_streams(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "handshake descriptor"):
            controller.run_campaign(self.campaign, ready_fd=0)
        self.assertEqual(controller.status(self.campaign)["phase"], "prepared")

    def test_stop_before_spawn_never_starts_runner(self):
        with patch.object(controller.state.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "immediate stop"):
                controller.state.wait_runner(["fake"], 10, lambda: True, lambda _: None, 40)
        launch.assert_not_called()

    def test_unknown_control_operation_is_rejected(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "control operation"):
            controller.request_control(self.campaign, "../escape")

    def test_status_detects_unknown_checkpoint_files(self):
        self.create()
        controller.run_campaign(self.campaign)
        (self.campaign / "invocations/000003/checkpoint/unexpected.bin").write_bytes(b"untrusted")
        with self.assertRaisesRegex(ValueError, "checkpoint.*inventory"):
            controller.status(self.campaign)

    def test_status_cannot_roll_back_accepted_receipts(self):
        self.create()
        controller.run_campaign(self.campaign)
        path = self.campaign / "status.json"
        current = json.loads(path.read_text())
        current.update(phase="prepared", accepted_updates=0, accepted_invocation=0, receipt_sha256=None)
        path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, "unrecorded invocation"):
            controller.status(self.campaign)

    def test_inspector_timeout_is_bounded_without_sleep(self):
        with patch.object(controller.state.time, "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(ValueError, "inspector timeout"):
                controller.state.capture([sys.executable, "-c", "pass"], timeout=10)

    def test_native_path_escape_and_counter_rollback_are_rejected(self):
        self.create()
        controller.run_campaign(self.campaign)
        job = self.campaign / "invocations/000003"
        value = json.loads((job / "checkpoint/inspection.json").read_text())
        previous = json.loads((self.campaign / "invocations/000002/accepted.json").read_text())["inspection"]
        for field, replacement, message in (
                ("files", [{"path": "../result.json", "size": 1, "sha256": "a" * 64}], "path escape"),
                ("progress", dict(value["progress"], optimizer_steps=0), "rolled back"),
                ("identity", dict(value["identity"], scope_sha256="b" * 64), "scope changed")):
            with self.subTest(field=field), patch.object(controller.state, "capture", return_value=
                    json.dumps(dict(value, **{field: replacement})).encode()):
                with self.assertRaisesRegex(ValueError, message):
                    controller.inspect_checkpoint(self.campaign, job / "checkpoint", 5, previous)


class ReportTests(unittest.TestCase):
    setUp = LifecycleTests.setUp
    scenario = LifecycleTests.scenario
    configure = LifecycleTests.configure
    write_config = CampaignTests.write_config
    create = CampaignTests.create

    def accepted(self, number):
        return controller.read_json(self.campaign / "invocations" / f"{number:06d}" / "accepted.json")

    def run_two_updates(self):
        # The manifest freezes total_updates, so the profile is written first.
        self.configure(total_updates=2, invocation_updates=1)
        self.create()
        controller.run_campaign(self.campaign)

    def test_report_aggregates_accepted_invocations_and_ignores_pending(self):
        self.run_two_updates()
        status = controller.read_json(self.campaign / "status.json")
        status["phase"] = "running"
        status["pending_invocation"] = 3
        (self.campaign / "status.json").write_text(json.dumps(status))
        pending = self.campaign / "invocations" / "000003"
        pending.mkdir(mode=0o700)
        (pending / "payload.log").write_text(
            "".join("episode: stream=0 map=2 outcome=Win\n" for _ in range(20)))
        value = controller.report(self.campaign)
        self.assertEqual(value["schema"], "drysua-training-report/v1")
        self.assertEqual((value["accepted_updates"], value["accepted_invocations"]), (2, 2))
        self.assertEqual((value["total_updates"], value["remaining_updates"]), (2, 0))
        self.assertEqual(value["games_per_update"], 2)
        self.assertEqual(value["overall"], {"games": 4, "wins": 2, "losses": 2, "draws": 0,
                                            "win_rate": 0.5})
        self.assertEqual(value["per_update"], [
            {"update": 1, "wins": 1, "losses": 1, "draws": 0},
            {"update": 2, "wins": 1, "losses": 1, "draws": 0}])
        self.assertEqual([block["updates"] for block in value["blocks"]], [2])
        self.assertEqual(value["blocks"][0]["first_update"], 1)
        self.assertIsNone(value["recent"]["last10"])
        self.assertIsNone(value["recent"]["last20"])
        self.assertEqual(value["adaptive"], None)

    def test_report_rejects_an_incomplete_invocation_payload(self):
        self.run_two_updates()
        payload = self.campaign / "invocations" / "000002" / "payload.log"
        payload.write_text("episode: stream=0 map=2 outcome=Win\n")
        with self.assertRaisesRegex(ValueError, "terminal episodes, expected 2"):
            controller.report(self.campaign)

    def test_report_rejects_an_unknown_outcome_and_an_invalid_block(self):
        self.run_two_updates()
        payload = self.campaign / "invocations" / "000001" / "payload.log"
        payload.write_text("episode: stream=0 map=2 outcome=Draw\nepisode: stream=1 map=2 outcome=Error\n")
        with self.assertRaisesRegex(ValueError, "unsupported episode outcome"):
            controller.report(self.campaign)
        payload.write_text("episode: stream=0 map=2 outcome=Win\nepisode: stream=1 map=2 outcome=Loss\n")
        for block in (0, 10001):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, "report block"):
                controller.report(self.campaign, block=block)

    def test_report_json_line_matches_the_documented_schema(self):
        self.run_two_updates()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(controller.main(["report", str(self.campaign), "--json"]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["schema"], "drysua-training-report/v1")
        self.assertEqual(set(value), {"schema", "campaign_id", "phase", "accepted_updates",
                                      "total_updates", "remaining_updates", "accepted_invocations",
                                      "games_per_update", "games", "overall", "blocks", "recent",
                                      "per_update", "adaptive", "timing",
                                      "environments", "extensions"})
        # The fixture inspection has no adaptive state, so the section is n/a.
        self.assertIsNone(value["environments"])
        self.assertIsNone(value["extensions"])
        self.assertIn("environments n/a (fixed schedule)", controller.format_report(value))
        self.assertEqual(value["timing"], {"timed_invocations": 2, "seconds_per_update": 30.0,
                                           "eta_seconds": 0})

    def test_report_on_a_prepared_campaign_reports_no_progress(self):
        self.create()
        empty = controller.report(self.campaign)
        self.assertEqual(empty["accepted_updates"], 0)
        self.assertIsNone(empty["games_per_update"])
        self.assertEqual(empty["per_update"], [])
        self.assertEqual(empty["blocks"], [])
        self.assertEqual(empty["overall"], {"games": 0, "wins": 0, "losses": 0, "draws": 0,
                                            "win_rate": None})
        self.assertIsNone(empty["timing"]["seconds_per_update"])
        self.assertIsNone(empty["timing"]["eta_seconds"])
        self.assertIn("timing n/a", controller.format_report(empty))

    def test_report_is_read_only_for_accepted_progress(self):
        self.run_two_updates()
        receipts = [(self.campaign / "invocations" / f"{number:06d}" / "accepted.json").read_bytes()
                    for number in (1, 2)]
        status = (self.campaign / "status.json").read_bytes()
        controller.report(self.campaign)
        controller.report(self.campaign, block=1)
        self.assertEqual([(self.campaign / "invocations" / f"{number:06d}" / "accepted.json").read_bytes()
                          for number in (1, 2)], receipts)
        self.assertEqual((self.campaign / "status.json").read_bytes(), status)
        self.assertEqual(controller.status(self.campaign)["accepted_updates"], 2)

    def test_report_block_of_one_lists_every_update(self):
        self.run_two_updates()
        value = controller.report(self.campaign, block=1)
        self.assertEqual([block["updates"] for block in value["blocks"]], [1, 1])
        self.assertEqual([block["first_update"] for block in value["blocks"]], [1, 2])
        self.assertEqual(value["recent"]["last10"], None)
        text = controller.format_report(value)
        self.assertIn("block 001-001 1-1-0 win=50.0%", text)
        self.assertIn("timing 30.0s/update (2/2 timed) eta=0h00m", text)


    def run_adaptive(self, total_updates):
        self.scenario("adaptive")
        self.configure(total_updates=total_updates, invocation_updates=1)
        self.create()
        controller.run_campaign(self.campaign)

    def test_report_aggregates_environment_transitions_and_extensions(self):
        self.run_adaptive(5)
        value = controller.report(self.campaign)
        self.assertEqual(value["accepted_updates"], 5)
        self.assertEqual(value["environments"], {
            "total": 3, "completed": 3, "current": 0, "skipped_early": [5],
            "clean_truncated": [1], "spent": {"min": 1, "max": 3, "mean": 1.67}})
        self.assertEqual(value["extensions"], {"environments": 2, "awards_total": 3,
                                               "extra_updates_total": 1})
        text = controller.format_report(value)
        self.assertIn("environments total=3 completed=3 current=0 skipped_early=1 "
                      "clean_truncated=1 avg_spent=1.67 min=1 max=3", text)
        self.assertIn("skipped early at updates=5", text)
        self.assertIn("clean truncated at updates=1", text)
        self.assertIn("extensions environments=2 awards=3 extra_updates=1", text)

    def test_report_reports_the_running_generation(self):
        self.run_adaptive(2)
        value = controller.report(self.campaign)
        self.assertEqual(value["accepted_updates"], 2)
        self.assertEqual(value["environments"], {
            "total": 2, "completed": 1, "current": 1, "skipped_early": [],
            "clean_truncated": [1], "spent": {"min": 1, "max": 1, "mean": 1.0}})
        self.assertEqual(value["extensions"], {"environments": 1, "awards_total": 1,
                                               "extra_updates_total": 0})

    def test_report_rejects_a_transition_count_that_disagrees_with_the_generation(self):
        self.run_adaptive(2)
        # Dropping the transition line leaves generation 1 without its transition.
        payload = self.campaign / "invocations" / "000001" / "payload.log"
        payload.write_text("".join(line + "\n" for line in payload.read_text().splitlines()
                                   if "adaptive_environment_transition" not in line))
        with self.assertRaisesRegex(ValueError, "transitions but the checkpoint reports generation"):
            controller.report(self.campaign)


def transitions(*rows):
    """Fixture transition records from (update, start_update, previous_awards, clean)."""
    return [{"update": row[0], "previous_generation": index, "generation": index + 1,
             "start_update": row[1], "previous_awards": row[2], "clean": row[3]}
            for index, row in enumerate(rows)]


class ReportAggregationTests(unittest.TestCase):
    def test_episode_outcomes_require_the_exact_episode_count(self):
        payload = (b"episode: stream=0 map=2 outcome=Win\n"
                   b"level=INFO event=map2_episode_reward outcome=Win reward_total=1\n"
                   b"episode: stream=1 map=2 outcome=Draw\n")
        self.assertEqual(controller.state.episode_outcomes(payload, 2), ["Win", "Draw"])
        with self.assertRaisesRegex(ValueError, "records 2 terminal episodes, expected 3"):
            controller.state.episode_outcomes(payload, 3)
        with self.assertRaisesRegex(ValueError, "more than 1 terminal episodes"):
            controller.state.episode_outcomes(payload, 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            controller.state.episode_outcomes(payload, 0)

    def test_episode_outcomes_reject_unknown_outcomes_and_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "unsupported episode outcome: Timeout"):
            controller.state.episode_outcomes(b"episode: stream=0 map=2 outcome=Timeout\n", 1)
        with self.assertRaisesRegex(ValueError, "without an outcome field"):
            controller.state.episode_outcomes(b"episode: stream=0 map=2 tick=4\n", 1)

    def test_blocks_allow_a_shorter_last_block_and_recent_needs_enough_updates(self):
        outcomes = [["Win"] * 2, ["Loss"] * 2, ["Win", "Loss"], ["Draw", "Draw"], ["Win", "Win"]]
        updates = controller.state.update_records(outcomes, 2)
        self.assertEqual([record["update"] for record in updates], [1, 2, 3, 4, 5])
        blocks = controller.state.block_records(updates, 2)
        self.assertEqual([block["updates"] for block in blocks], [2, 2, 1])
        self.assertEqual([block["first_update"] for block in blocks], [1, 3, 5])
        self.assertEqual(blocks[-1], {"games": 2, "wins": 2, "losses": 0, "draws": 0,
                                      "win_rate": 1.0, "first_update": 5, "updates": 1})
        self.assertEqual(controller.state.add_wins(updates), {"games": 10, "wins": 5, "losses": 3, "draws": 2,
                                                   "win_rate": 0.5})
        self.assertIsNone(controller.state.recent_record(updates, 6))
        self.assertEqual(controller.state.recent_record(updates, 5)["wins"], 5)
        with self.assertRaisesRegex(ValueError, "whole number of updates"):
            controller.state.update_records([["Win", "Loss", "Win"]], 2)

    def test_games_per_update_prefers_environments_and_falls_back_to_the_run_scope(self):
        self.assertEqual(controller.state.games_per_update({"ppo": {"environments": 8}, "run": {}}), 8)
        self.assertEqual(controller.state.games_per_update(
            {"ppo": {}, "run": {"command_line": "train-annealed --games 40 --generation-games 160"}}), 40)
        for inspection in ({"ppo": {"environments": 0}, "run": {"command_line": "--games 0"}},
                           {"ppo": {}, "run": {"command_line": "--generation-games 160"}},
                           {"ppo": {}, "run": {}}):
            with self.subTest(inspection=inspection), self.assertRaisesRegex(ValueError, "neither"):
                controller.state.games_per_update(inspection)

    def test_mean_seconds_uses_available_timings_only(self):
        self.assertEqual(controller.state.mean_seconds([1.5, None, 2.5]), 2.0)
        self.assertIsNone(controller.state.mean_seconds([]))
        self.assertIsNone(controller.state.mean_seconds([None, 0.0, 90000.0, -1.0]))

    def test_environment_transitions_parse_known_lines_and_ignore_other_records(self):
        payload = (b"episode: stream=0 map=2 outcome=Win\n"
                   b"level=INFO event=adaptive_environment_transition update=2 previous_generation=0 "
                   b"generation=1 start_update=2 previous_awards=1 clean=false dropped_logs=0\n"
                   b"level=INFO event=map2_episode_reward outcome=Win reward_total=1\n")
        self.assertEqual(controller.state.environment_transitions(payload), [
            {"update": 2, "previous_generation": 0, "generation": 1, "start_update": 2,
             "previous_awards": 1, "clean": False}])
        self.assertEqual(controller.state.environment_transitions(b""), [])
        invalid = [
            (b"event=adaptive_environment_transition update=2 previous_generation=0 generation=1 "
             b"start_update=2 previous_awards=1\n", "missing fields"),
            (b"event=adaptive_environment_transition update=x previous_generation=0 generation=1 "
             b"start_update=x previous_awards=0 clean=false\n", "non-negative integer"),
            (b"event=adaptive_environment_transition update=2 previous_generation=0 generation=1 "
             b"start_update=3 previous_awards=0 clean=false\n", "must match"),
            (b"event=adaptive_environment_transition update=2 previous_generation=0 generation=2 "
             b"start_update=2 previous_awards=0 clean=false\n", "must follow"),
            (b"event=adaptive_environment_transition update=2 previous_generation=0 generation=1 "
             b"start_update=2 previous_awards=0 clean=maybe\n", "true or false"),
            (b"event=adaptive_environment_transition update=2 previous_generation=0 generation=1 "
             b"start_update=2 previous_awards=0 clean=false update=3\n", "duplicate"),
        ]
        for payload_bytes, message in invalid:
            with self.subTest(payload=payload_bytes), self.assertRaisesRegex(ValueError, message):
                controller.state.environment_transitions(payload_bytes)

    def test_environment_summary_counts_early_skips_clean_truncations_and_extra_updates(self):
        summary = controller.state.environment_summary(
            transitions((1, 1, 0, True), (4, 4, 0, False), (5, 5, 0, False)), 3, 5, 2, 0)
        self.assertEqual(summary["environments"], {
            "total": 3, "completed": 3, "current": 0, "skipped_early": [5],
            "clean_truncated": [1], "spent": {"min": 1, "max": 3, "mean": 1.67}})
        self.assertEqual(summary["extensions"], {"environments": 0, "awards_total": 0,
                                                 "extra_updates_total": 1})

    def test_environment_summary_extension_awards_floor_whole_extra_updates(self):
        # One 0.75 award grants no whole extra update even though it is counted.
        one = controller.state.environment_summary(transitions((4, 4, 1, False)), 1, 8, 4, 0)
        self.assertEqual((one["extensions"]["environments"], one["extensions"]["awards_total"],
                          one["extensions"]["extra_updates_total"]), (1, 1, 0))
        # Two awards reach 1.5 credit; the played span shows the extra update.
        two = controller.state.environment_summary(transitions((4, 4, 2, False)), 1, 9, 4, 0)
        self.assertEqual((two["extensions"]["environments"], two["extensions"]["awards_total"],
                          two["extensions"]["extra_updates_total"]), (1, 2, 1))
        self.assertEqual(two["environments"]["spent"], {"min": 4, "max": 5, "mean": 4.5})

    def test_environment_summary_reports_the_current_generation_and_rejects_mismatch(self):
        running = controller.state.environment_summary(transitions((4, 4, 2, False)), 1, 6, 4, 1)
        self.assertEqual(running["environments"]["current"], 1)
        self.assertEqual(running["environments"]["spent"], {"min": 2, "max": 4, "mean": 3.0})
        # The completed generation carries two awards and the running one another.
        self.assertEqual(running["extensions"], {"environments": 2, "awards_total": 3,
                                                 "extra_updates_total": 0})
        with self.assertRaisesRegex(ValueError, "transitions but the checkpoint reports generation"):
            controller.state.environment_summary(transitions((2, 2, 0, False)), 2, 2, 2, 0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            controller.state.environment_summary(
                transitions((4, 4, 0, False), (5, 4, 0, False)), 2, 5, 2, 0)
        with self.assertRaisesRegex(ValueError, "base updates must be a positive integer"):
            controller.state.environment_summary([], 0, 0, 0, 0)

    def test_environment_summary_is_empty_without_transitions(self):
        summary = controller.state.environment_summary([], 0, 0, 4, 0)
        self.assertEqual(summary["environments"], {
            "total": 0, "completed": 0, "current": 0, "skipped_early": [], "clean_truncated": [],
            "spent": {"min": None, "max": None, "mean": None}})
        self.assertEqual(summary["extensions"], {"environments": 0, "awards_total": 0,
                                                 "extra_updates_total": 0})


if __name__ == "__main__":
    unittest.main()