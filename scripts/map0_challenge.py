"""Separate fixed-Teacher Map0 challenge; never the historical Map1 release gate."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil

from release_build import SIMULATOR, digest, snapshot_weights

CONFIG_PATH = Path(__file__).with_suffix(".json")
CONFIG = json.loads(CONFIG_PATH.read_text())
CONFIG_SHA256 = digest(CONFIG_PATH)
EPOCH = CONFIG["epoch"]
BASELINE_ID = CONFIG["baseline_id"]
SIDES = ("Radiant", "Dire")


def schedule(final, pairs):
    if final:
        if pairs is not None:
            raise ValueError("final forbids --challenge-pairs")
        return [(seed, side) for seed in range(9880000, 9880050) for side in SIDES]
    pairs = 10 if pairs is None else pairs
    if type(pairs) is not int or not 1 <= pairs <= 10:
        raise ValueError("development pairs must be 1..10")
    return [(seed, side) for seed in range(9870000, 9870000 + pairs) for side in SIDES]


def evaluate(records, final, pairs):
    expected = set(schedule(final, pairs))
    seen, errors = set(), []
    wins = 0
    for game in records:
        key = (game.get("seed"), game.get("side"))
        if key not in expected or key in seen:
            errors.append("unexpected or duplicate seed/side")
        seen.add(key)
        if game.get("result") not in ("win", "loss", "draw", "timeout") or game.get("errors") != []:
            errors.append(f"ineligible game {key}: {game.get('errors')}")
        elif game["result"] == "win":
            wins += 1
    if seen != expected:
        errors.append("missing scheduled games")
    eligible = not errors
    return dict(cohort="final" if final else "development", games=len(expected), wins=wins,
                eligible=eligible, qualified_80_percent=final and eligible and wins >= 80,
                errors=errors)


def validate_arguments(args):
    if (CONFIG["map"], CONFIG["tick_limit"], CONFIG["process_timeout_seconds"], CONFIG["max_workers"],
        CONFIG["cpu_threads"], CONFIG["final_seed_start"], CONFIG["final_pairs"], CONFIG["final_required_wins"],
        CONFIG["development_seed_start"], CONFIG["development_pairs_max"], CONFIG["candidate_policy"],
        CONFIG["baseline_policy"]) != (0, 108900, 180, 4, 1, 9880000, 50, 80, 9870000, 10, "neural", "teacher"):
        raise ValueError("challenge epoch configuration contract mismatch")
    if args.candidate_policy != "neural" or args.candidate_weights is None:
        raise ValueError("Map0 challenge requires explicit neural candidate and weights")
    if args.baseline_manifest is None:
        raise ValueError("Map0 challenge requires --baseline-manifest")
    if not 1 <= args.challenge_workers <= 4:
        raise ValueError("challenge workers must be 1..4")
    schedule(args.challenge_final, args.challenge_pairs)


def checked_asset(root, asset):
    relative = Path(asset["path"])
    if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError("baseline asset path must remain below its manifest")
    path = root / relative
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError("baseline asset must not be a symlink")
    if not path.is_file() or not 1 <= path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("baseline asset must be a bounded regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) or digest(path) != asset["sha256"]:
        raise ValueError(f"baseline asset SHA256 mismatch: {relative}")
    return path


def load_baseline(path, expected=None):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("baseline manifest must be a bounded regular file")
    with path.open("rb") as stream:
        data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError("baseline manifest exceeds byte limit")
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("baseline manifest SHA256 mismatch for challenge epoch")
    manifest = json.loads(data)
    if (manifest.get("schema_version"), manifest.get("baseline_id"), manifest.get("epoch"),
        manifest.get("role"), manifest.get("policy"), manifest.get("map"), manifest.get("weights", "missing")) != (
            1, BASELINE_ID, EPOCH, "baseline", "teacher", 0, None):
        raise ValueError("expected frozen Map0 weights-free Teacher baseline for this epoch")
    provenance = manifest["provenance"]
    if not re.fullmatch(r"[0-9a-f]{64}", provenance["parent_manifest_sha256"]) or not re.fullmatch(
            r"[0-9a-f]{40}", provenance["drysua_head"]):
        raise ValueError("baseline source provenance is missing or invalid")
    if manifest["simulator"]["commit"] != SIMULATOR:
        raise ValueError("baseline simulator identity mismatch")
    for key in ("binary", "simulator", "source_manifest", "patch"):
        checked_asset(path.parent, manifest[key])
    return manifest


def runtime_digest(source):
    if source.is_symlink() or not source.is_file() or not 1 <= source.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("runtime must be a bounded regular non-symlink file")
    return digest(source)


def snapshot_runtime(source, target, expected):
    if runtime_digest(source) != expected:
        raise ValueError("runtime SHA256 mismatch")
    shutil.copyfile(source, target)
    target.chmod(0o500)
    if digest(source) != expected or digest(target) != expected:
        raise ValueError("runtime changed during snapshot")


def prepare(args, output):
    manifest = load_baseline(args.baseline_manifest, CONFIG["baseline_manifest_sha256"])
    (output / "baseline.json").write_text(json.dumps(manifest, indent=2) + "\n")
    identities = [(args.baseline_manifest, CONFIG["baseline_manifest_sha256"]),
                  (CONFIG_PATH, CONFIG_SHA256)]
    for key in ("binary", "simulator"):
        source = checked_asset(args.baseline_manifest.parent, manifest[key])
        target = output / ("baseline-teacher" if key == "binary" else "server")
        snapshot_runtime(source, target, manifest[key]["sha256"])
        identities.append((target, manifest[key]["sha256"]))
    candidate = output / "candidate"
    candidate_sha = runtime_digest(args.candidate_binary)
    snapshot_runtime(args.candidate_binary, candidate, candidate_sha)
    weights = snapshot_weights(args.candidate_weights, output / "candidate-weights", policy="neural")
    identities.extend([(candidate, candidate_sha), (Path(weights["snapshot"]), weights["sha256"])])
    bots = [dict(binary=candidate, policy="neural", weights=output / "candidate-weights"),
            dict(binary=output / "baseline-teacher", policy="teacher")]
    metadata = dict(binary_sha256=candidate_sha, weights=weights, source=args.candidate_metadata)
    return bots, identities, manifest, metadata


def run(args, root, output):
    from release_crossplay import execute_game
    validate_arguments(args)
    bots, identities, manifest, candidate = prepare(args, output)
    harness = output / "harness"
    harness.mkdir()
    for name in ("release_crossplay.py", "release_build.py", "release_wire.py", "map0_challenge.py", "map0_challenge.json"):
        shutil.copyfile(root / "scripts" / name, harness / name)
    games = schedule(args.challenge_final, args.challenge_pairs)
    report = dict(evaluation="fixed-teacher-map0", epoch=EPOCH, config=CONFIG, config_sha256=CONFIG_SHA256,
                  cohort="final" if args.challenge_final else "development",
                  baseline_manifest=str(args.baseline_manifest), baseline=manifest,
                  candidate=candidate, schedule=[dict(seed=seed, side=side) for seed, side in games],
                  games=[], gate=dict(eligible=False, qualified_80_percent=False, errors=["incomplete run"]))
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def game(job):
        seed, side = job
        directory = output / f"baseline-map0-teacher-{seed}-{side}"
        directory.mkdir()
        ordered = bots if side == "Radiant" else list(reversed(bots))
        result = execute_game(directory, output / "server", ordered, seed, CONFIG)
        if side == "Dire" and result["result"] in ("win", "loss"):
            result["result"] = "loss" if result["result"] == "win" else "win"
        result.update(seed=seed, side=side, opponent=BASELINE_ID)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    # Submit only four jobs at a time, rather than an unbounded executor queue.
    with ThreadPoolExecutor(max_workers=args.challenge_workers) as workers:
        for start in range(0, len(games), args.challenge_workers):
            futures = [workers.submit(game, job) for job in games[start:start + args.challenge_workers]]
            for future in futures:
                result = future.result()
                report["games"].append(result)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Map0 {result['seed']} {result['side']}: {result['result']} {result['errors']}", flush=True)
    report["gate"] = evaluate(report["games"], args.challenge_final, args.challenge_pairs)
    if any(digest(path) != expected for path, expected in identities):
        report["gate"].update(eligible=False, qualified_80_percent=False)
        report["gate"]["errors"].append("runtime or weights changed during challenge")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2), flush=True)
    return 0 if report["gate"]["eligible"] and (not args.challenge_final or report["gate"]["qualified_80_percent"]) else 1
