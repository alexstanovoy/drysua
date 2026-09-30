"""Fake `drysua eval`: the real output contract with outcomes from player strengths.

A weights player's strength is the number in its weights file (the fake trainer
writes the update there); `teacher` is 0 and `harass-push` 1. The candidate wins
when its strength plus a deterministic per-(seed, side, opponent) noise in [-2, 2]
beats the opponent's.
"""
import hashlib
import json
from pathlib import Path
import sys

RUNTIME_FILE = "drysua.weights.safetensors"
SCRIPTS = {"teacher": 0.0, "harass-push": 1.0}


def option(arguments, name):
    return arguments[arguments.index(name) + 1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def player(spec):
    kind, _, rest = spec.partition(":")
    if kind == "weights":
        data = (Path(rest) / RUNTIME_FILE).read_bytes()
        return "weights:" + sha256(data), float(data)
    if kind == "average":
        members = [(Path(member) / RUNTIME_FILE).read_bytes() for member in rest.split(",")]
        key = "average:" + sha256(",".join(sha256(data) for data in members).encode())
        return key, sum(float(data) for data in members) / len(members)
    return "script:" + kind, SCRIPTS[kind]


def game(candidate, opponent, seed, side):
    noise = int(sha256(f"{opponent[0]}/{seed}/{side}".encode())[:8], 16) / 0xFFFFFFFF * 4 - 2
    won = candidate[1] + noise > opponent[1]
    hero = {"kills": 1, "deaths": 0, "level": 6, "xp": 1000, "tower_hp": 0.5,
            "casts": {"raze_near": 2, "raze_mid": 1, "raze_far": 1, "requiem": 0},
            "raze_hero_hits": 1, "raze_hits": 2, "raze_modes": {"none": 4, "entity": 0, "point": 0}}
    lead = {"xp": 50 if won else -50, "gold": 10, "deaths": 0, "tower_bp": 0, "hp_bp": 100}
    return {"side": side, "outcome": "win" if won else "loss", "end_reason": "tower", "ticks": 9000,
            "own": hero, "enemy": hero, "leads": {"2m": lead, "3m": lead, "5m": None}}


def main(arguments):
    assert arguments[0] == "eval"
    candidate = player(option(arguments, "--candidate"))
    pool = json.loads(Path(option(arguments, "--pool")).read_text())["opponents"]
    first, count = map(int, option(arguments, "--seeds").split(":"))
    mode = "greedy" if "--greedy" in arguments else "sampled"
    header = {"schema": "drysua-eval/v3", "kind": "header",
              "context": f"exe:{sha256(Path(__file__).read_bytes())}/{mode}", "greedy": mode == "greedy",
              "seeds": {"first": first, "count": count},
              "candidate": {"name": option(arguments, "--name"), "player": option(arguments, "--candidate"),
                            "key": candidate[0]},
              "pool": [dict(entry, key=player(entry["player"])[0]) for entry in pool]}
    lines = [header]
    for seed in range(first, first + count):
        for entry in pool:
            for side in ("radiant", "dire"):
                line = game(candidate, player(entry["player"]), seed, side)
                lines.append(dict(line, schema="drysua-eval/v3", kind="game", opponent=entry["name"], seed=seed))
    with open(option(arguments, "--output"), "x") as output:
        output.write("".join(json.dumps(line) + "\n" for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
