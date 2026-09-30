"""Evaluation comparison statistics and pairing; no trainer or simulator is executed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from eval_compare import main, mcnemar_exact, wilson_interval


def game(seed, side, outcome, end_reason="tower"):
    return {"schema": "drysua-eval/v1", "seed": seed, "side": side, "outcome": outcome,
            "end_reason": end_reason, "ticks": 9000}


def summary(games, opponent="teacher"):
    return {"schema": "drysua-eval/v1", "summary": True, "games": games, "opponent": opponent,
            "greedy": False, "seeds": "1:2"}


class StatisticsTests(unittest.TestCase):
    def test_wilson_interval_matches_published_values(self):
        for wins, games, expected in ((0, 10, (0.0, 0.2775)), (5, 10, (0.2366, 0.7634)),
                                      (10, 10, (0.7225, 1.0))):
            with self.subTest(wins=wins, games=games):
                low, high = wilson_interval(wins, games)
                self.assertAlmostEqual(low, expected[0], places=4)
                self.assertAlmostEqual(high, expected[1], places=4)

    def test_wilson_interval_rejects_more_wins_than_games(self):
        with self.assertRaisesRegex(ValueError, "wins 3 outside 0..2"):
            wilson_interval(3, 2)

    def test_mcnemar_exact_on_crafted_discordant_table(self):
        # b=10, c=2: p = 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 158 / 4096.
        self.assertAlmostEqual(mcnemar_exact(10, 2), 158 / 4096, places=12)
        self.assertAlmostEqual(mcnemar_exact(2, 10), 158 / 4096, places=12)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, games, opponent="teacher"):
        path = Path(self.directory.name) / f"{name}.jsonl"
        lines = [json.dumps(record) for record in games + [summary(len(games), opponent)]]
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def run_main(self, *paths):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(list(paths)), 0)
        return output.getvalue()

    def test_pairs_only_shared_seed_side_games(self):
        first = self.write("first", [game(1, "radiant", "win"), game(1, "dire", "win"),
                                     game(2, "radiant", "loss")])
        second = self.write("second", [game(1, "radiant", "loss"), game(1, "dire", "win"),
                                       game(3, "radiant", "win")])
        output = self.run_main(first, second)
        self.assertIn("first vs second: n=2 paired, win difference +50.0 pts, "
                      "discordant 1/0, McNemar exact p=1", output)
        self.assertIn("first: vs teacher (sampled, seeds 1:2) win 66.7% ", output)
        self.assertIn("loss by tower: 100.0% of losses", output)

    def test_different_opponents_are_not_paired(self):
        first = self.write("first", [game(1, "radiant", "win")])
        second = self.write("second", [game(1, "radiant", "loss")], opponent="weights:u100")
        self.assertIn("first vs second: skipped, different opponents",
                      self.run_main(first, second))

    def test_duplicate_game_is_rejected(self):
        path = self.write("duplicate", [game(1, "radiant", "win"), game(1, "radiant", "loss")])
        with self.assertRaisesRegex(ValueError, r"duplicate game \(1, 'radiant'\)"):
            main([path])


if __name__ == "__main__":
    unittest.main()
