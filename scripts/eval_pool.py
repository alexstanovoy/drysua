#!/usr/bin/env python3
"""Frozen pool evaluation: the single source of truth for "is the bot stronger".

`drysua eval` plays one candidate against every pool opponent on both sides of
each seed. This tool runs it in seed chunks into an append-only store (one JSONL
file per invocation, never rewritten), never replays a game the store already
holds, stops early on a sequential test, and reports ratings and robustness.

  eval_pool.py run --drysua BIN --candidate PLAYER --pool POOL --store DIR [--seeds S:N]
      [--name NAME] [--chunk-seeds N] [--baseline PLAYER --baseline-name NAME --delta D]
  eval_pool.py report --store DIR [--candidate NAME ...] [--anchor NAME] [--json]
  eval_pool.py compare --store DIR FIRST SECOND

PLAYER is a rule policy (`teacher`, `harass-push`), `weights:<dir>` or
`average:<dir>,<dir>,...`. Games are comparable only within one context: the same
`drysua` binary and sampling mode.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import eval_stats as stats

EVAL_SCHEMA = "drysua-eval/v3"
POOL_SCHEMA = "drysua-eval-pool/v1"
RUNTIME_FILE = "drysua.weights.safetensors"
NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}")
SCRIPT = re.compile(r"[a-z][a-z-]{0,31}")
SIDES = ("radiant", "dire")
OUTCOMES = ("win", "loss", "draw")
END_REASONS = ("tower", "deaths", "timecap", "draw")
LEAD_FIELDS = ("xp", "gold", "deaths", "tower_bp", "hp_bp", "last_hits", "denies", "net_worth")
MINUTES = ("2m", "3m", "5m", "10m")
ECONOMY_FIELDS = ("last_hits", "denies", "gold_earned", "net_worth")
PLACES = ("dead", "fountain", "base", "lane", "enemy_base")
# Display names of bota item ids (bota-client catalog); others print as their id.
ITEM_NAMES = {0: "boots", 1: "clarity", 2: "salve", 3: "branch", 5: "quell", 7: "tango", 8: "tp", 9: "circlet",
              11: "slippers", 12: "mantle", 13: "belt", 19: "gloves", 29: "treads", 32: "bracer", 33: "wraith",
              34: "null", 35: "stick", 36: "wand", 42: "mango"}
MAX_STORE_FILES = 10000
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_STORE_BYTES = 1024 ** 3
MAX_POOL_BYTES = 64 * 1024
MAX_POOL = 16
MAX_SEEDS = 10000
MAX_MEMBERS = 16
SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def file_sha256(path, limit=1024 ** 3):
    digest, total = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"{path}: larger than {limit} bytes")
            digest.update(chunk)
    return digest.hexdigest()


def player_key(player):
    """The key `drysua eval` writes for `player`; checked against every run header."""
    kind, _, rest = player.partition(":")
    if kind == "weights" and rest:
        return "weights:" + file_sha256(Path(rest) / RUNTIME_FILE)
    if kind == "average" and rest:
        members = rest.split(",")
        if not 2 <= len(members) <= MAX_MEMBERS or not all(members):
            raise ValueError(f"average needs 2..{MAX_MEMBERS} directories: {player!r}")
        hashes = ",".join(file_sha256(Path(member) / RUNTIME_FILE) for member in members)
        return "average:" + hashlib.sha256(hashes.encode()).hexdigest()
    if not rest and SCRIPT.fullmatch(kind):
        return "script:" + kind
    raise ValueError(f"player must be a rule policy, weights:<dir> or average:<dir>,...: {player!r}")


def absolute_player(player, base):
    """`player` with every directory made absolute against `base`."""
    kind, _, rest = player.partition(":")
    if kind == "weights":
        return f"weights:{(base / rest).absolute()}"
    if kind == "average":
        return "average:" + ",".join(str((base / member).absolute()) for member in rest.split(","))
    return player


def validate_name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"name must be 1..64 of [A-Za-z0-9._-] not starting with '.': {name!r}")
    return name


def parse_pool(value, base):
    """Validated pool entries `{name, player (absolute), role, key}` from a pool document."""
    if not isinstance(value, dict) or set(value) != {"schema", "opponents"} or value["schema"] != POOL_SCHEMA:
        raise ValueError(f"pool must hold exactly schema {POOL_SCHEMA!r} and opponents")
    opponents = value["opponents"]
    if not isinstance(opponents, list) or not 1 <= len(opponents) <= MAX_POOL:
        raise ValueError(f"opponents must list 1..{MAX_POOL} entries")
    entries = []
    for opponent in opponents:
        if not isinstance(opponent, dict) or set(opponent) != {"name", "player", "role"}:
            raise ValueError("every opponent holds exactly name, player and role")
        name = validate_name(opponent["name"])
        if any(entry["name"] == name for entry in entries):
            raise ValueError(f"duplicate opponent name {name!r}")
        if opponent["role"] not in {"train", "held-out"}:
            raise ValueError(f"role must be train or held-out: {opponent['role']!r}")
        player = absolute_player(opponent["player"], base)
        entries.append({"name": name, "player": player, "role": opponent["role"], "key": player_key(player)})
    return entries


def read_pool(path):
    path = Path(path)
    if path.stat().st_size > MAX_POOL_BYTES:
        raise ValueError(f"{path}: larger than {MAX_POOL_BYTES} bytes")
    return parse_pool(json.loads(path.read_text()), path.absolute().parent)


def context_of(binary, greedy):
    return f"exe:{file_sha256(binary)}/{'greedy' if greedy else 'sampled'}"


class Store:
    """Every game of every result file in one directory, indexed by context and players."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.headers = []
        self.games = {}
        files = sorted(self.directory.glob("*.jsonl"))
        if len(files) > MAX_STORE_FILES:
            raise ValueError(f"{self.directory}: more than {MAX_STORE_FILES} result files")
        total = 0
        for path in files:
            total += path.stat().st_size
            if total > MAX_STORE_BYTES:
                raise ValueError(f"{self.directory}: results exceed {MAX_STORE_BYTES} bytes")
            self.add(path)

    def add(self, path):
        """Indexes one result file; a replayed game must match the stored one exactly."""
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
        lines = path.read_text().splitlines()
        header = json.loads(lines[0]) if lines else {}
        if header.get("schema") != EVAL_SCHEMA or header.get("kind") != "header":
            raise ValueError(f"{path}: first line is not a {EVAL_SCHEMA} header")
        opponents = {entry["name"]: entry["key"] for entry in header["pool"]}
        first, count = header["seeds"]["first"], header["seeds"]["count"]
        seen = set()
        for number, line in enumerate(lines[1:], 2):
            game = json.loads(line)
            if game.get("schema") != EVAL_SCHEMA or game.get("kind") != "game":
                raise ValueError(f"{path}:{number}: not a {EVAL_SCHEMA} game line")
            if game["opponent"] not in opponents or not first <= game["seed"] < first + count:
                raise ValueError(f"{path}:{number}: game outside the header's pool or seeds")
            key = (header["context"], header["candidate"]["key"], opponents[game["opponent"]],
                   game["seed"], game["side"])
            if key in seen:
                raise ValueError(f"{path}:{number}: duplicate game {key[2:]}")
            seen.add(key)
            stored = self.games.setdefault(key, game)
            if {**stored, "opponent": None} != {**game, "opponent": None}:
                raise ValueError(f"{path}:{number}: replay of {key[1:]} differs; evaluation is not deterministic")
        if len(seen) != count * 2 * len(opponents):
            raise ValueError(f"{path}: {len(seen)} games, header plans {count * 2 * len(opponents)}")
        self.headers.append(header)

    def latest_context(self):
        return self.headers[-1]["context"] if self.headers else None

    def has(self, context, candidate, opponent, seed):
        return all((context, candidate, opponent, seed, side) in self.games for side in SIDES)

    def names(self, context):
        """Latest display name and pool role of every player key in `context`."""
        names, roles = {}, {}
        for header in self.headers:
            if header["context"] != context:
                continue
            names[header["candidate"]["key"]] = header["candidate"]["name"]
            for entry in header["pool"]:
                names.setdefault(entry["key"], entry["name"])
                roles[entry["key"]] = entry["role"]
        return names, roles

    def results(self, context):
        """`{candidate: {opponent: [game, ...]}}` of one context, in seed order."""
        grouped = {}
        for (game_context, candidate, opponent, seed, side), game in sorted(self.games.items()):
            if game_context == context:
                grouped.setdefault(candidate, {}).setdefault(opponent, []).append(game)
        return grouped


def run_chunk(options, store, context, candidate, pool, first, count):
    """Plays the chunk's missing (opponent, seed) games of `candidate` into a new store file."""
    missing = [entry for entry in pool
               if not all(store.has(context, candidate["key"], entry["key"], seed)
                          for seed in range(first, first + count))]
    if not missing:
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = store.directory / f"{stamp}-{candidate['name']}-{first}-{count}.jsonl"
    pool_file = store.directory / f".pool-{os.getpid()}-{stamp}.json"
    pool_file.write_text(json.dumps({"schema": POOL_SCHEMA, "opponents": [
        {key: entry[key] for key in ("name", "player", "role")} for entry in missing]}))
    try:
        command = [str(options.drysua), "eval", "--candidate", candidate["player"], "--name", candidate["name"],
                   "--pool", str(pool_file), "--seeds", f"{first}:{count}", "--output", str(output),
                   "--device", options.device, "--parallel", str(options.parallel)]
        subprocess.run(command + (["--greedy"] if options.greedy else []), check=True, stdout=subprocess.DEVNULL)
    finally:
        pool_file.unlink()
    store.add(output)
    header = store.headers[-1]
    expected = [(entry["name"], entry["key"]) for entry in missing]
    if (header["context"], header["candidate"]["key"]) != (context, candidate["key"]) or \
            [(entry["name"], entry["key"]) for entry in header["pool"]] != expected:
        raise ValueError(f"{output}: header identities differ from this tool's; binary and tool disagree")
    return True


def pair_scores(games):
    """`{seed: pair score in [0, 2]}` for seeds with both sides played."""
    by_seed = {}
    for game in games:
        by_seed.setdefault(game["seed"], []).append(SCORE[game["outcome"]])
    return {seed: sum(scores) for seed, scores in by_seed.items() if len(scores) == 2}


def paired_units(results, baseline, candidate, pool, seeds):
    """(candidate - baseline) pair score / 2 per (seed, opponent), in plan order."""
    units = []
    first = {opponent["key"]: pair_scores(results.get(candidate, {}).get(opponent["key"], [])) for opponent in pool}
    second = {opponent["key"]: pair_scores(results.get(baseline, {}).get(opponent["key"], [])) for opponent in pool}
    for seed in seeds:
        for opponent in pool:
            key = opponent["key"]
            if seed in first[key] and seed in second[key]:
                units.append((first[key][seed] - second[key][seed]) / 2)
    return units


def evaluate(options, store, candidate, pool, first, count, baseline=None):
    """Fixed-n, or with `baseline` a GSPRT of H0 mean 0 vs H1 mean `delta`, stopping when decided."""
    context = context_of(options.drysua, options.greedy)
    played, test = 0, None
    for start in range(first, first + count, options.chunk_seeds):
        size = min(options.chunk_seeds, first + count - start)
        for player in ([baseline] if baseline else []) + [candidate]:
            played += run_chunk(options, store, context, player, pool, start, size)
        if baseline is None:
            continue
        units = paired_units(store.results(context), baseline["key"], candidate["key"], pool,
                             range(first, start + size))
        test = stats.gsprt(units, 0.0, options.delta, options.alpha, options.beta)
        print(f"sprt units={test['units']} mean={test['mean']:+.4f} llr={test['llr']:.3f} "
              f"bounds=[{test['lower']:.3f}, {test['upper']:.3f}] decision={test['decision']}", file=sys.stderr)
        if test["decision"]:
            break
    return {"context": context, "new_runs": played, "sprt": test}


def summarize(games, roles):
    """Robustness metrics of one candidate: `games` maps opponent key to its games."""
    by_opponent = {opponent: record(played) for opponent, played in games.items()}
    rated = [(value["win_rate"], opponent) for opponent, value in by_opponent.items() if value["games"]]
    worst = min(rated) if rated else None
    pooled = {role: record([game for opponent, played in games.items() if roles.get(opponent) == role
                            for game in played]) for role in ("train", "held-out")}
    everything = [game for played in games.values() for game in played]
    return {"by_opponent": by_opponent, "worst": None if worst is None else dict(by_opponent[worst[1]],
                                                                              opponent=worst[1]),
            "train": pooled["train"], "held_out": pooled["held-out"], "all": record(everything),
            "end_reasons": {outcome: {reason: sum(game["outcome"] == outcome and game["end_reason"] == reason
                                                  for game in everything) for reason in END_REASONS}
                            for outcome in OUTCOMES},
            "raze_hero_hit_rate": {hero: raze_rate(everything, hero) for hero in ("own", "enemy")},
            "leads": leads(everything),
            "economy": {hero: economy(everything, hero) for hero in ("own", "enemy")}}


def record(games):
    wins = sum(game["outcome"] == "win" for game in games)
    draws = sum(game["outcome"] == "draw" for game in games)
    score = wins + draws / 2
    low, high = stats.wilson_interval(score, len(games))
    sides = {side: {"games": sum(game["side"] == side for game in games),
                    "wins": sum(game["side"] == side and game["outcome"] == "win" for game in games)}
             for side in SIDES}
    return {"games": len(games), "wins": wins, "draws": draws, "losses": len(games) - wins - draws,
            "win_rate": score / len(games) if games else None, "ci95": [low, high], "sides": sides}


def raze_rate(games, hero):
    casts = sum(game[hero]["casts"][name] for game in games for name in ("raze_near", "raze_mid", "raze_far"))
    hits = sum(game[hero]["raze_hero_hits"] for game in games)
    return {"razes": casts, "hero_hits": hits, "rate": hits / casts if casts else None}


def leads(games):
    """Mean own-minus-enemy standing per milestone, and the win rate when ahead in XP there."""
    result = {}
    for minute in MINUTES:
        reached = [game["leads"][minute] | {"outcome": game["outcome"]} for game in games
                   if game.get("leads", {}).get(minute)]
        if not reached:
            continue
        ahead = [lead for lead in reached if lead["xp"] > 0]
        result[minute] = {"games": len(reached),
                          **{field: sum(lead.get(field, 0) for lead in reached) / len(reached)
                             for field in LEAD_FIELDS},
                          "ahead_xp_games": len(ahead),
                          "ahead_xp_win_rate": sum(lead["outcome"] == "win" for lead in ahead) / len(ahead)
                          if ahead else None}
    return result


def economy(games, hero):
    """Mean farming, spending and places of one hero; games from before the economy fields are skipped."""
    games = [game[hero] for game in games if "economy" in game[hero]]
    if not games:
        return None
    mean = lambda values: sum(values) / len(values) if values else None
    minutes = {minute: {field: mean([game["minutes"][minute][field] for game in games if game["minutes"].get(minute)])
                        for field in ECONOMY_FIELDS} for minute in MINUTES}
    per_item = lambda kind: {ITEM_NAMES.get(int(item), item): sum(game["spending"][kind].get(item, 0)
                                                                  for game in games) / len(games)
                             for item in sorted({item for game in games for item in game["spending"][kind]}, key=int)}
    placed = sum(sum(game["spending"]["places"].values()) for game in games)
    return {"games": len(games), "end": {field: mean([game["economy"][field] for game in games])
                                         for field in ECONOMY_FIELDS},
            "minutes": minutes, "items_bought": per_item("items_bought"),
            "consumables_used": per_item("consumables_used"),
            "places": {place: sum(game["spending"]["places"][place] for game in games) / placed if placed else None
                       for place in PLACES}}


def economy_lines(value):
    lines = []
    for hero, summary in value.items():
        if not summary:
            continue
        end = summary["end"]
        lines.append(f"  {hero} economy (n={summary['games']}): end " + " ".join(
            f"{field} {end[field]:.1f}" for field in ECONOMY_FIELDS) + "; " + " ".join(
            f"{minute} lh {at['last_hits']:.1f} nw {at['net_worth']:.0f}"
            for minute, at in summary["minutes"].items() if at["net_worth"] is not None))
        lines.append(f"    bought/game " + (", ".join(f"{item} {count:.2f}" for item, count in
                                                     summary["items_bought"].items()) or "nothing") +
                     "; used/game " + (", ".join(f"{item} {count:.2f}" for item, count in
                                                 summary["consumables_used"].items()) or "nothing"))
        lines.append("    places " + " ".join(f"{place} {percent(share)}" for place, share in summary["places"].items()))
    return lines


def ratings(results, anchor, names):
    """Bradley-Terry Elo of every player of one context; `anchor` is a name or key."""
    table = {}
    for candidate, opponents in results.items():
        for opponent, games in opponents.items():
            if candidate == opponent:
                continue
            score = sum(SCORE[game["outcome"]] for game in games)
            previous = table.get((candidate, opponent), (0.0, 0))
            table[(candidate, opponent)] = (previous[0] + score, previous[1] + len(games))
    if not table:
        return {}
    keys = {key for pair in table for key in pair}
    by_name = {name: key for key, name in names.items()}
    anchor_key = by_name.get(anchor, anchor)
    if anchor_key not in keys:
        anchor_key = "script:teacher" if "script:teacher" in keys else sorted(keys)[0]
    fitted = stats.bradley_terry(table, anchor_key)
    return {key: {"elo": elo, "se": error, "anchor": key == anchor_key} for key, (elo, error) in fitted.items()}


def report(store, context=None, anchor="teacher"):
    """Per-candidate metrics and ratings of one context (default: the latest run's)."""
    context = context or store.latest_context()
    names, roles = store.names(context)
    results = store.results(context)
    rated = ratings(results, anchor, names)
    candidates = {}
    for header in store.headers:
        key = header["candidate"]["key"]
        if header["context"] == context and key in results:
            candidates[key] = dict(name=names[key], rating=rated.get(key),
                                   **summarize(results[key], roles))
    return {"context": context, "names": names, "roles": roles, "ratings": rated, "candidates": candidates}


def compare(store, first, second, context=None):
    """Fixed-n paired comparison of two candidates on the (opponent, seed, side) games both played."""
    context = context or store.latest_context()
    names, _ = store.names(context)
    by_name = {name: key for key, name in names.items()}
    first, second = by_name.get(first, first), by_name.get(second, second)
    results = store.results(context)
    shared = sorted(set(results.get(first, {})) & set(results.get(second, {})))
    pool = [{"key": opponent} for opponent in shared]
    seeds = sorted({game["seed"] for opponent in shared for game in results[first][opponent]})
    units = paired_units(results, second, first, pool, seeds)
    outcomes = {}
    for candidate in (first, second):
        for opponent in shared:
            for game in results[candidate][opponent]:
                outcomes.setdefault((opponent, game["seed"], game["side"]), {})[candidate] = game["outcome"]
    both = [value for value in outcomes.values() if len(value) == 2]
    only_first = sum(value[first] == "win" and value[second] != "win" for value in both)
    only_second = sum(value[second] == "win" and value[first] != "win" for value in both)
    mean, low, high = stats.mean_interval(units)
    return {"first": names.get(first, first), "second": names.get(second, second),
            "opponents": [names.get(key, key) for key in shared], "units": len(units),
            "pair_score_difference": mean, "ci95": [low, high], "games": len(both),
            "discordant": [only_first, only_second], "mcnemar_p": stats.mcnemar_exact(only_first, only_second)}


def percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def record_text(value):
    low, high = value["ci95"]
    sides = " ".join(f"{side} {value['sides'][side]['wins']}/{value['sides'][side]['games']}" for side in SIDES)
    return (f"{percent(value['win_rate'])} [{100 * low:.1f}, {100 * high:.1f}] "
            f"W/L/D {value['wins']}/{value['losses']}/{value['draws']} ({sides})")


def format_report(value):
    names, roles = value["names"], value["roles"]
    lines = [f"context {value['context']}"]
    for key, rating in sorted(value["ratings"].items(), key=lambda item: -item[1]["elo"]):
        anchor = " (anchor)" if rating["anchor"] else ""
        lines.append(f"elo {names.get(key, key):<24} {rating['elo']:+7.1f} ± {1.96 * rating['se']:.1f}{anchor}")
    for candidate in value["candidates"].values():
        lines.append(f"{candidate['name']}: all {record_text(candidate['all'])}")
        for opponent, result in candidate["by_opponent"].items():
            lines.append(f"  vs {names.get(opponent, opponent)} ({roles.get(opponent, 'n/a')}) {record_text(result)}")
        if candidate["worst"]:
            lines.append(f"  worst case vs {names.get(candidate['worst']['opponent'])}: "
                         f"{record_text(candidate['worst'])}")
        for role in ("train", "held_out"):
            if candidate[role]["games"]:
                lines.append(f"  {role} {record_text(candidate[role])}")
        for outcome, reasons in candidate["end_reasons"].items():
            lines.append(f"  {outcome} by " + ", ".join(f"{reason} {count}" for reason, count in reasons.items()))
        razes = candidate["raze_hero_hit_rate"]
        lines.append("  raze hero hits " + ", ".join(
            f"{hero} {razes[hero]['hero_hits']}/{razes[hero]['razes']} ({percent(razes[hero]['rate'])})"
            for hero in ("own", "enemy")))
        for minute, lead in candidate["leads"].items():
            lines.append(f"  lead at {minute} (n={lead['games']}): " +
                         " ".join(f"{field} {lead[field]:+.1f}" for field in LEAD_FIELDS) +
                         f"; win {percent(lead['ahead_xp_win_rate'])} when ahead in XP (n={lead['ahead_xp_games']})")
        lines.extend(economy_lines(candidate["economy"]))
    return "\n".join(lines)


def seed_range(text):
    match = re.fullmatch(r"(\d+):(\d+)", text)
    if not match or not 1 <= int(match.group(2)) <= MAX_SEEDS:
        raise ValueError(f"seeds must be <start>:<count> with 1..{MAX_SEEDS} seeds")
    return int(match.group(1)), int(match.group(2))


def add_run_arguments(parser):
    """Execution options shared with `train.py eval`."""
    parser.add_argument("--seeds", type=seed_range, default=(1000000, 100), help="default 1000000:100")
    parser.add_argument("--chunk-seeds", type=int, default=16, help="seeds per drysua invocation (default 16)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--parallel", type=int, default=16, help="worlds per pipeline group")
    parser.add_argument("--greedy", action="store_true")


def candidate_entry(player, name):
    player = absolute_player(player, Path.cwd())
    if name is None:
        kind, _, rest = player.partition(":")
        name = Path(rest).name if kind == "weights" else kind if not rest else None
    return {"name": validate_name(name), "player": player, "key": player_key(player)}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="operation", required=True)
    run = commands.add_parser("run", help="evaluate a candidate against a pool (fixed-n or SPRT)")
    run.add_argument("--drysua", type=Path, required=True, help="drysua binary built with builtin")
    run.add_argument("--candidate", required=True)
    run.add_argument("--name")
    run.add_argument("--pool", type=Path, required=True)
    run.add_argument("--baseline", help="SPRT: the player the candidate must beat by --delta")
    run.add_argument("--baseline-name")
    run.add_argument("--delta", type=float, default=0.05, help="SPRT H1: mean pair-score gain (default 0.05)")
    run.add_argument("--alpha", type=float, default=0.05)
    run.add_argument("--beta", type=float, default=0.05)
    add_run_arguments(run)
    for name in ("run", "report", "compare"):
        command = run if name == "run" else commands.add_parser(name)
        command.add_argument("--store", type=Path, required=True, help="result directory (created if missing)")
        command.add_argument("--context", help="default: the latest run's")
    report_command = commands.choices["report"]
    report_command.add_argument("--anchor", default="teacher", help="player rated 0 Elo (name or key)")
    report_command.add_argument("--json", action="store_true", dest="as_json")
    commands.choices["compare"].add_argument("first")
    commands.choices["compare"].add_argument("second")
    options = parser.parse_args(arguments)
    return execute(options)


def execute(options):
    if options.operation == "run":
        if not 1 <= options.chunk_seeds <= MAX_SEEDS or not 1 <= options.parallel <= 64:
            raise ValueError("chunk seeds must be 1..10000 and parallel 1..64")
        options.store.mkdir(mode=0o700, parents=True, exist_ok=True)
        options.drysua = options.drysua.resolve()
        store, pool = Store(options.store), read_pool(options.pool)
        candidate = candidate_entry(options.candidate, options.name)
        baseline = None if options.baseline is None else candidate_entry(options.baseline, options.baseline_name)
        result = evaluate(options, store, candidate, pool, *options.seeds, baseline=baseline)
        print(format_report(report(store, result["context"])))
        print(json.dumps({key: result[key] for key in ("context", "new_runs", "sprt")}, sort_keys=True))
        return 0
    store = Store(options.store)
    if options.operation == "compare":
        print(json.dumps(compare(store, options.first, options.second, options.context), sort_keys=True))
        return 0
    value = report(store, options.context, options.anchor)
    print(json.dumps(value, sort_keys=True) if options.as_json else format_report(value))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"eval_pool: {error}", file=sys.stderr)
        sys.exit(2)
