#!/usr/bin/env python3
"""Compare `drysua eval` JSONL results: win rates with Wilson 95% CIs and paired McNemar tests.

Usage: eval_compare.py RESULT.jsonl [RESULT.jsonl ...]

Pairs are games with the same (seed, side). McNemar compares only results that
faced the same opponent, since a different opponent makes the pairing meaningless.
"""

import json
import math
from pathlib import Path
import sys


SCHEMA = "drysua-eval/v1"
WILSON_Z = 1.959963984540054
MAX_FILE_BYTES = 64 * 1024 * 1024
END_REASONS = ("tower", "deaths", "timecap", "draw")


def wilson_interval(wins, games):
    """Wilson score interval at 95% for `wins` successes out of `games`."""
    if not 0 <= wins <= games:
        raise ValueError(f"wins {wins} outside 0..{games}")
    if games == 0:
        return 0.0, 1.0
    proportion = wins / games
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / games
    center = (proportion + z2 / (2 * games)) / denominator
    half = WILSON_Z / denominator * math.sqrt(
        proportion * (1 - proportion) / games + z2 / (4 * games * games))
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(only_first, only_second):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    if only_first < 0 or only_second < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def load(path):
    """Games keyed by (seed, side), plus the summary line."""
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
    games, summary = {}, None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        record = json.loads(line)
        if record.get("schema") != SCHEMA:
            raise ValueError(f"{path}:{number}: schema is not {SCHEMA}")
        if record.get("summary"):
            if summary is not None:
                raise ValueError(f"{path}:{number}: second summary line")
            summary = record
            continue
        key = (record["seed"], record["side"])
        if key in games:
            raise ValueError(f"{path}:{number}: duplicate game {key}")
        games[key] = record
    if summary is None:
        raise ValueError(f"{path}: missing summary line")
    if summary["games"] != len(games):
        raise ValueError(f"{path}: summary counts {summary['games']} games, found {len(games)}")
    return games, summary


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else float("nan")


def describe(name, games, summary):
    """Human-readable lines for one result file."""
    outcomes = [game["outcome"] for game in games.values()]
    wins, losses = outcomes.count("win"), outcomes.count("loss")
    low, high = wilson_interval(wins, len(games))
    lines = [f"{name}: vs {summary['opponent']} ({'greedy' if summary['greedy'] else 'sampled'}, "
             f"seeds {summary['seeds']}) win {percent(wins, len(games)):.1f}% "
             f"[{100 * low:.1f}, {100 * high:.1f}] n={len(games)} "
             f"W/L/D {wins}/{losses}/{len(games) - wins - losses}"]
    sides = []
    for side in ("radiant", "dire"):
        played = [game for game in games.values() if game["side"] == side]
        won = sum(game["outcome"] == "win" for game in played)
        sides.append(f"{side} {percent(won, len(played)):.1f}% ({won}/{len(played)})")
    lines.append("  sides: " + ", ".join(sides))
    for outcome in ("win", "loss", "draw"):
        matching = [game for game in games.values() if game["outcome"] == outcome]
        reasons = ", ".join(
            f"{reason} {sum(game['end_reason'] == reason for game in matching)}"
            for reason in END_REASONS)
        lines.append(f"  {outcome} by: {reasons}")
    tower_losses = sum(game["end_reason"] == "tower" for game in games.values()
                       if game["outcome"] == "loss")
    lines.append(f"  loss by tower: {percent(tower_losses, losses):.1f}% of losses")
    return lines


def compare(first, second):
    """One McNemar line for two results on their shared (seed, side) games."""
    (first_name, first_games, first_summary) = first
    (second_name, second_games, second_summary) = second
    if first_summary["opponent"] != second_summary["opponent"]:
        return f"{first_name} vs {second_name}: skipped, different opponents"
    shared = sorted(first_games.keys() & second_games.keys())
    if not shared:
        return f"{first_name} vs {second_name}: skipped, no shared (seed, side) games"
    only_first = sum(first_games[key]["outcome"] == "win" and second_games[key]["outcome"] != "win"
                     for key in shared)
    only_second = sum(second_games[key]["outcome"] == "win" and first_games[key]["outcome"] != "win"
                      for key in shared)
    difference = percent(only_first - only_second, len(shared))
    return (f"{first_name} vs {second_name}: n={len(shared)} paired, win difference "
            f"{difference:+.1f} pts, discordant {only_first}/{only_second}, "
            f"McNemar exact p={mcnemar_exact(only_first, only_second):.4g}")


def main(arguments):
    if not arguments:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    results = [(Path(path).stem, *load(path)) for path in arguments]
    for name, games, summary in results:
        print("\n".join(describe(name, games, summary)))
    for index, first in enumerate(results):
        for second in results[index + 1:]:
            print(compare(first, second))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
