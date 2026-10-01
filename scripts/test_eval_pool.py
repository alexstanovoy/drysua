"""Evaluation statistics on crafted tables and the pool driver against a fake `drysua eval`."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import eval_pool
import eval_stats as stats

FAKE_EVAL = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fake_eval.py"


class StatisticsTests(unittest.TestCase):
    def test_wilson_interval_matches_published_values(self):
        for wins, games, expected in ((0, 10, (0.0, 0.2775)), (5, 10, (0.2366, 0.7634)),
                                      (10, 10, (0.7225, 1.0))):
            with self.subTest(wins=wins, games=games):
                low, high = stats.wilson_interval(wins, games)
                self.assertAlmostEqual(low, expected[0], places=4)
                self.assertAlmostEqual(high, expected[1], places=4)

    def test_mcnemar_exact_on_crafted_discordant_table(self):
        # b=10, c=2: p = 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 158 / 4096.
        self.assertAlmostEqual(stats.mcnemar_exact(10, 2), 158 / 4096, places=12)
        self.assertAlmostEqual(stats.mcnemar_exact(2, 10), 158 / 4096, places=12)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)

    def test_two_player_rating_has_the_closed_form_with_one_virtual_draw(self):
        # 30 of 40 plus a virtual draw: p = 30.5 / 41, information 41 p (1 - p).
        fitted = stats.bradley_terry({("b", "a"): (30, 40)}, anchor="a")
        probability = 30.5 / 41
        self.assertEqual(fitted["a"], (0.0, 0.0))
        self.assertAlmostEqual(fitted["b"][0], 400 * math.log10(probability / (1 - probability)), places=6)
        self.assertAlmostEqual(fitted["b"][1], 400 / math.log(10) / math.sqrt(41 * probability * (1 - probability)),
                               places=6)

    def test_ratings_recover_a_transitive_table_and_stay_finite_for_perfect_records(self):
        truth = {"a": 0.0, "b": 100.0, "c": 250.0}
        table = {(first, second): (1000 * stats.elo_to_score(truth[first] - truth[second]), 1000)
                 for first, second in (("b", "a"), ("c", "a"), ("c", "b"))}
        fitted = stats.bradley_terry(table, anchor="a", prior_games=0)
        for player, elo in truth.items():
            self.assertAlmostEqual(fitted[player][0], elo, places=4)
        self.assertLess(fitted["b"][1], fitted["c"][1] + 1e-9)
        perfect = stats.bradley_terry({("b", "a"): (10, 10)}, anchor="a")
        self.assertTrue(math.isfinite(perfect["b"][0]) and perfect["b"][0] > 400)

    def test_gsprt_log_likelihood_ratio_and_decisions_on_a_crafted_stream(self):
        # Units 1, 0, 1, 0, ...: mean 0.5, variance 0.25; LLR = n * 0.05 * (1 - 0.05) / 0.5 = 0.095 n.
        for units, decision in ((16, None), (32, "H1")):
            with self.subTest(units=units):
                result = stats.gsprt([1.0, 0.0] * (units // 2), 0.0, 0.05)
                self.assertAlmostEqual(result["llr"], 0.095 * units, places=12)
                self.assertAlmostEqual(result["upper"], math.log(0.95 / 0.05), places=12)
                self.assertEqual(result["decision"], decision)
        self.assertIsNone(stats.gsprt([0.0] * 15, 0.0, 0.05)["decision"])
        self.assertEqual(stats.gsprt([0.0] * 16, 0.0, 0.05)["decision"], "H0")


class DriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "drysua"
        self.binary.write_text(f"#!{sys.executable}\n" + FAKE_EVAL.read_text())
        self.binary.chmod(0o700)
        self.pool = self.root / "pool.json"
        self.pool.write_text(json.dumps({"schema": "drysua-eval-pool/v1", "opponents": [
            {"name": "teacher", "player": "teacher", "role": "train"},
            {"name": "harass", "player": "harass-push", "role": "held-out"}]}))
        for name, strength in (("weak", "0"), ("strong", "1")):
            (self.root / name).mkdir()
            (self.root / name / "drysua.weights.safetensors").write_text(strength)

    def run_pool(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            eval_pool.main(["run", "--drysua", str(self.binary), "--pool", str(self.pool),
                            "--store", str(self.root / "store"), "--chunk-seeds", "4", *extra])
        return json.loads(output.getvalue().splitlines()[-1])

    def test_sprt_stops_once_decided_and_a_rerun_replays_nothing(self):
        arguments = ("--candidate", f"weights:{self.root / 'strong'}", "--baseline", f"weights:{self.root / 'weak'}",
                     "--seeds", "1:400")
        first = self.run_pool(*arguments)
        self.assertEqual(first["sprt"]["decision"], "H1")
        self.assertLess(first["sprt"]["units"], 2 * 400)
        self.assertEqual(first["new_runs"], len(list((self.root / "store").glob("*.jsonl"))))
        self.assertEqual(self.run_pool(*arguments)["new_runs"], 0)
        same = self.run_pool("--candidate", f"weights:{self.root / 'weak'}", "--name", "again",
                             "--baseline", f"weights:{self.root / 'weak'}", "--seeds", "1:400")
        self.assertEqual((same["sprt"]["decision"], same["new_runs"]), ("H0", 0))

    def test_report_rates_players_and_separates_held_out_and_worst_case(self):
        self.run_pool("--candidate", f"weights:{self.root / 'strong'}", "--seeds", "1:8")
        value = eval_pool.report(eval_pool.Store(self.root / "store"))
        [candidate] = value["candidates"].values()
        by_name = {value["names"][key]: result for key, result in candidate["by_opponent"].items()}
        self.assertEqual(candidate["held_out"], by_name["harass"])
        self.assertEqual(candidate["train"], by_name["teacher"])
        self.assertEqual(candidate["worst"]["win_rate"], min(result["win_rate"] for result in by_name.values()))
        self.assertEqual(candidate["all"]["games"], 8 * 2 * 2)
        self.assertEqual(candidate["leads"]["2m"]["games"], 32)
        self.assertNotIn("5m", candidate["leads"])
        own = candidate["economy"]["own"]
        self.assertEqual((own["games"], own["items_bought"]), (32, {"salve": 4, "tango": 1}))
        self.assertAlmostEqual(own["places"]["lane"], 8000 / 9000)
        self.assertIsNone(own["minutes"]["5m"]["net_worth"])
        self.assertIn("bought/game salve 4.00", eval_pool.format_report(value))
        ratings = {value["names"][key]: rating for key, rating in value["ratings"].items()}
        self.assertTrue(ratings["teacher"]["anchor"])
        self.assertGreater(ratings["strong"]["elo"], ratings["harass"]["elo"])

    def test_a_replay_with_a_different_outcome_is_rejected(self):
        self.run_pool("--candidate", "teacher", "--seeds", "1:1")
        [stored] = (self.root / "store").glob("*.jsonl")
        lines = stored.read_text().splitlines()
        game = json.loads(lines[1])
        game["outcome"] = "loss" if game["outcome"] == "win" else "win"
        (self.root / "store" / "zz-copy.jsonl").write_text("\n".join([lines[0], json.dumps(game), *lines[2:]]) + "\n")
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            eval_pool.Store(self.root / "store")


if __name__ == "__main__":
    unittest.main()
