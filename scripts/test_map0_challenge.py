"""Fixed Teacher identity and held-out Map0 challenge contracts."""

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import map0_challenge as challenge
from release_build import digest


class ChallengeTests(unittest.TestCase):
    def test_run_keeps_development_schedule_roles_and_full_map_caps_with_four_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            path = root / "baseline.json"
            path.write_text(json.dumps(manifest))
            binary = root / "input-neural"
            binary.write_bytes(b"neural runtime")
            (root / "drysua.weights.safetensors").write_bytes(b"weights")
            output = root / "run"
            output.mkdir()
            args = argparse.Namespace(candidate_policy="neural", candidate_binary=binary,
                                      candidate_weights=root, candidate_metadata="test provenance",
                                      baseline_manifest=path, challenge_final=False, challenge_pairs=2,
                                      challenge_workers=4)
            calls = []

            def execute(directory, server, bots, seed, config):
                calls.append((server, bots, seed, config))
                return dict(result="timeout", errors=[])

            with patch.dict(challenge.CONFIG, baseline_manifest_sha256=digest(path)), \
                    patch("release_crossplay.execute_game", side_effect=execute), patch("builtins.print"):
                result = challenge.run(args, Path(__file__).resolve().parents[1], output)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(report["cohort"], "development")
            self.assertFalse(report["gate"]["qualified_80_percent"])
            self.assertEqual({(game["seed"], game["side"]) for game in report["games"]},
                             set(challenge.schedule(False, 2)))
            for server, bots, _, config in calls:
                self.assertEqual(server, output / "server")
                self.assertEqual(config["map"], 0)
                self.assertEqual(config["tick_limit"], 108900)
                self.assertEqual(config["process_timeout_seconds"], 180)
                teacher = next(bot for bot in bots if bot["policy"] == "teacher")
                self.assertEqual(teacher, dict(binary=output / "baseline-teacher", policy="teacher"))

    def test_prepare_snapshots_fixed_teacher_not_candidate_and_binds_epoch_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            path = root / "baseline.json"
            path.write_text(json.dumps(manifest))
            candidate = root / "neural-input"
            candidate.write_bytes(b"not the baseline binary")
            (root / "drysua.weights.safetensors").write_bytes(b"runtime weights")
            output = root / "run"
            output.mkdir()
            args = argparse.Namespace(baseline_manifest=path, candidate_binary=candidate,
                                      candidate_weights=root, candidate_metadata="test source")
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch for challenge epoch"):
                challenge.prepare(args, output)
            with patch.dict(challenge.CONFIG, baseline_manifest_sha256=digest(path)):
                bots, _, _, _ = challenge.prepare(args, output)
            self.assertEqual(bots[0]["policy"], "neural")
            self.assertEqual(bots[1]["policy"], "teacher")
            self.assertNotIn("weights", bots[1])
            self.assertEqual(bots[1]["binary"].read_bytes(), b"binary")
            self.assertNotEqual(bots[0]["binary"].read_bytes(), bots[1]["binary"].read_bytes())

    def test_epoch_cannot_silently_weaken_full_map_cap(self):
        args = argparse.Namespace(candidate_policy="neural", candidate_weights=Path("weights"),
                                  baseline_manifest=Path("manifest"), challenge_final=True,
                                  challenge_pairs=None, challenge_workers=4)
        with patch.dict(challenge.CONFIG, tick_limit=1000):
            with self.assertRaisesRegex(ValueError, "epoch configuration contract mismatch"):
                challenge.validate_arguments(args)

    def test_final_schedule_is_fifty_predeclared_pairs_and_disjoint_from_development(self):
        final = challenge.schedule(True, None)
        development = challenge.schedule(False, 10)
        self.assertEqual([(game[0], game[1]) for game in final],
                         [(seed, side) for seed in range(9880000, 9880050)
                          for side in ("Radiant", "Dire")])
        self.assertFalse({seed for seed, _ in final} & {seed for seed, _ in development})
        for pairs in (0, 11, 50):
            with self.assertRaisesRegex(ValueError, "development pairs must be 1..10"):
                challenge.schedule(False, pairs)
        with self.assertRaisesRegex(ValueError, "final forbids --challenge-pairs"):
            challenge.schedule(True, 10)

    def test_eighty_of_all_hundred_required_and_errors_never_give_free_wins(self):
        records = [dict(seed=seed, side=side, result="win" if index < 80 else "timeout", errors=[])
                   for index, (seed, side) in enumerate(challenge.schedule(True, None))]
        self.assertTrue(challenge.evaluate(records, True, None)["qualified_80_percent"])
        records[0]["result"] = "timeout"
        self.assertFalse(challenge.evaluate(records, True, None)["qualified_80_percent"])
        records[0].update(result="win", errors=["baseline process failure"])
        self.assertFalse(challenge.evaluate(records, True, None)["eligible"])
        self.assertFalse(challenge.evaluate(records[:-1], True, None)["eligible"])
        self.assertFalse(challenge.evaluate(records + [records[1]], True, None)["eligible"])

    def test_development_cannot_qualify_even_with_all_wins(self):
        records = [dict(seed=seed, side=side, result="win", errors=[])
                   for seed, side in challenge.schedule(False, 3)]
        report = challenge.evaluate(records, False, 3)
        self.assertEqual(report["cohort"], "development")
        self.assertEqual(report["games"], 6)
        self.assertFalse(report["qualified_80_percent"])

    def test_challenge_requires_pure_neural_and_explicit_fixed_baseline(self):
        args = argparse.Namespace(candidate_policy="neural", candidate_weights=Path("weights"),
                                  baseline_manifest=None, challenge_final=False, challenge_pairs=10,
                                  challenge_workers=4)
        with self.assertRaisesRegex(ValueError, "requires --baseline-manifest"):
            challenge.validate_arguments(args)
        args.baseline_manifest = Path("baseline.json")
        challenge.validate_arguments(args)
        for policy in ("hybrid", "teacher", "tactical"):
            args.candidate_policy = policy
            with self.assertRaisesRegex(ValueError, "requires explicit neural candidate"):
                challenge.validate_arguments(args)

    def test_manifest_binds_binary_simulator_source_and_patch_and_rejects_wrong_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            path = root / "baseline.json"
            path.write_text(json.dumps(manifest))
            loaded = challenge.load_baseline(path)
            self.assertEqual(loaded["policy"], "teacher")
            for key, value in (("map", 1), ("role", "candidate"), ("policy", "neural"),
                               ("weights", "fallback.safetensors")):
                invalid = dict(manifest, **{key: value})
                path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "frozen Map0 weights-free Teacher"):
                    challenge.load_baseline(path)
            for asset in ("binary", "simulator", "source_manifest", "patch"):
                invalid = copy.deepcopy(manifest)
                invalid[asset]["sha256"] = "0" * 64
                path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    challenge.load_baseline(path)

    def manifest(self, root):
        manifest = dict(schema_version=1, baseline_id=challenge.BASELINE_ID, role="baseline",
                        policy="teacher", weights=None, map=0, epoch=challenge.EPOCH,
                        provenance=dict(parent_manifest_sha256="a" * 64, drysua_head="b" * 40))
        for key in ("binary", "simulator", "source_manifest", "patch"):
            path = root / key
            path.write_bytes(key.encode())
            manifest[key] = dict(path=key, sha256=digest(path))
        manifest["simulator"]["commit"] = challenge.SIMULATOR
        return manifest


if __name__ == "__main__":
    unittest.main()
