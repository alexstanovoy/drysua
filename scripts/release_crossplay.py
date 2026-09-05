"""Reproducible, fail-closed per-release TCP evaluation. Python stdlib only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from release_build import prepare, read_registry
from release_wire import Relay

SIDES = ("Radiant", "Dire")
OUTCOMES = ("win", "loss", "draw", "timeout", "error")
SUMMARY = re.compile(r"played (\d+) ticks as Some\((Radiant|Dire)\); winner "
                     r"(None|Some\((Radiant|Dire|Neutral)\)); "
                     r"\d+ decisions, \d+ orders, (\d+) rejected orders\n?")


def evaluate_gate(records, opponents, seeds):
    expected = {(tag, seed, side) for tag in opponents for seed in seeds for side in SIDES}
    seen = set()
    errors = []
    totals = {tag: dict(games=len(seeds) * 2, wins=0) for tag in opponents}
    if not opponents or len(seeds) < 10 or len(set(seeds)) != len(seeds):
        errors.append("at least one opponent and ten unique seeds required")
    for record in records:
        key = (record.get("opponent"), record.get("seed"), record.get("side"))
        if key not in expected or key in seen:
            errors.append(f"unexpected or duplicate game: {key}")
            continue
        seen.add(key)
        if record.get("result") not in OUTCOMES or "errors" not in record:
            errors.append(f"unknown or missing result: {key}")
        elif record["errors"] or record["result"] == "error":
            errors.append(f"invalid game: {key}: {record['errors']}")
        elif record["result"] == "win":
            totals[key[0]]["wins"] += 1
    if seen != expected:
        errors.append("missing scheduled games")
    for total in totals.values():
        total["passed"] = total["wins"] * 2 > total["games"]
    return dict(passed=not errors and all(t["passed"] for t in totals.values()),
                opponents=totals, errors=errors)


def validate_game(clients, server_exit, timed_out, tick_limit):
    errors = []
    winners = []
    for index, client in enumerate(clients):
        wire = client["wire"]
        errors.extend(wire["errors"])
        if wire["slot"] != index:
            errors.append("identity mismatch")
        if wire["rejected"]:
            errors.append("order rejection")
        if client["exit"] not in (0, None) and not client.get("killed", False):
            errors.append("client process failure")
        match = SUMMARY.fullmatch(client["stdout"])
        if not match:
            if not timed_out or client["exit"] == 0:
                errors.append("missing or invalid outcome")
            continue
        ticks, side, _, winner, rejected = match.groups()
        if side != SIDES[index]:
            errors.append("identity mismatch")
        if int(rejected):
            errors.append("order rejection")
        if winner != wire["winner"]:
            errors.append("MatchOver mismatch")
        if winner is None and int(ticks) != tick_limit:
            errors.append("missing MatchOver before tick limit")
        winners.append(winner)
    if len(clients) != 2:
        errors.append("missing client")
    if server_exit not in (0, None):
        errors.append("server process failure")
    if timed_out:
        return "timeout", errors
    if len(winners) != 2 or winners[0] != winners[1]:
        errors.append("inconsistent winners")
    if errors:
        return "error", errors
    if winners[0] is None:
        return "timeout", []
    if winners[0] == "Neutral":
        return "draw", []
    return ("win" if winners[0] == "Radiant" else "loss"), []


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def launch(command, path, processes, handles):
    handle = path.open("wb")
    handles.append(handle)
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               cwd=path.parent, start_new_session=True)
    processes.append(process)
    return process


def server_port(path, process, deadline):
    # File polling avoids buffered pipe readline hangs and pipe-capacity deadlocks.
    for _ in range(1000):
        if path.stat().st_size > 65536:
            raise ValueError("server startup output limit exceeded")
        text = path.read_text()
        match = re.search(r"^bota-server listening on 0\.0\.0\.0:(\d+)$", text, re.M)
        if match:
            return int(match[1])
        if process.poll() is not None:
            raise ValueError("server exited before listening")
        if time.monotonic() >= deadline:
            raise TimeoutError("server startup timeout")
        time.sleep(0.01)
    raise TimeoutError("server startup polling limit exceeded")


def collect(processes, paths, relays, killed):
    clients = []
    for index, (process, path) in enumerate(zip(processes[1:], paths)):
        text = path.read_text() if path.stat().st_size <= 65536 else "output limit exceeded"
        clients.append(dict(exit=process.returncode, stdout=text,
                            killed=process.pid in killed, wire=relays[index].observed))
    return clients


def check_resources(directory, paths):
    if any(path.stat().st_size > 65536 for path in paths):
        raise ValueError("client output limit exceeded")
    replay = directory / "match.brp"
    if replay.exists() and replay.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("replay size limit exceeded")


def execute_game(directory, server, bots, seed, registry):
    processes, handles, relays, commands, paths = [], [], [], [], []
    killed, errors = set(), []
    timed_out = False
    deadline = time.monotonic() + registry["process_timeout_seconds"]
    command = [str(server), "--port", "0", "--mode", "lockstep", "--players", "2",
               "--map", str(registry["map"]), "--seed", str(seed), "--ack-timeout-ticks",
               str(registry["process_timeout_seconds"] * 30 + 30),
               "--replay", str(directory / "match.brp")]
    commands.append(command)
    try:
        host = launch(command, directory / "server.log", processes, handles)
        port = server_port(directory / "server.log", host, min(deadline, time.monotonic() + 10))
        for index, bot in enumerate(bots):
            relay = Relay(port, 10, registry["tick_limit"])
            relays.append(relay)
            command = [str(bot), "play", "--policy", "teacher", "--addr", relay.address,
                       "--name", f"release-seat-{index}", "--limit", str(registry["tick_limit"])]
            commands.append(command)
            paths.append(directory / f"client-{index}.log")
            launch(command, paths[-1], processes, handles)
            if not relay.welcomed.wait(timeout=10) or relay.observed["slot"] != index:
                raise ValueError("identity mismatch: Welcome did not confirm scheduled seat")
        for _ in range(registry["process_timeout_seconds"] * 100 + 100):
            if all(process.poll() is not None for process in processes[1:]):
                break
            check_resources(directory, paths)
            if time.monotonic() >= deadline:
                raise TimeoutError("game wall timeout")
            time.sleep(0.01)
        else:
            raise TimeoutError("game polling limit exceeded")
        # A tick-limited server has no MatchOver and must be stopped by the runner.
        if all(relay.observed["winner"] is not None for relay in relays):
            host.wait(timeout=max(0.01, deadline - time.monotonic()))
    except (TimeoutError, subprocess.TimeoutExpired):
        timed_out = True
    except (OSError, ValueError) as error:
        errors.append(str(error))
    finally:
        for process in processes:
            if process.poll() is None:
                killed.add(process.pid)
                os.killpg(process.pid, 9)
            process.wait(timeout=5)
        for relay in relays:
            relay.close()
        for handle in handles:
            handle.close()
    clients = collect(processes, paths, relays, killed)
    server_exit = processes[0].returncode if processes and processes[0].pid not in killed else None
    result, validation = validate_game(clients, server_exit, timed_out, registry["tick_limit"])
    return dict(result=result, errors=errors + validation, clients=clients, commands=commands,
                cwd=str(directory), server_exit=server_exit, killed_pids=sorted(killed),
                wall_timeout=timed_out,
                wall_seconds=time.monotonic() - deadline + registry["process_timeout_seconds"])


def run(args, root, output):
    registry = read_registry(root / "releases.json", root, args.bota_repository)
    candidate = output / "candidate"
    source_hash = digest(args.candidate_binary)
    shutil.copy2(args.candidate_binary, candidate)
    candidate.chmod(0o500)
    if digest(candidate) != source_hash or digest(args.candidate_binary) != source_hash:
        raise ValueError("candidate changed while being snapshotted; retry after build completes")
    harness = output / "harness"
    harness.mkdir()
    for name in ("release_build.py", "release_crossplay.py", "release_wire.py"):
        shutil.copy2(root / "scripts" / name, harness / name)
    report = dict(registry=registry, candidate=dict(source=str(args.candidate_binary),
                  sha256=source_hash, command_metadata=args.candidate_metadata), games=[],
                  invocation=sys.argv, python=sys.version, platform=platform.platform(),
                  gate=dict(passed=False, errors=["incomplete run"]))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    server, opponents = prepare(root, args.bota_repository, output, registry)
    report["server_sha256"] = digest(server)
    report["opponent_sha256"] = {tag: digest(binary) for tag, binary in opponents.items()}
    for tag, binary in opponents.items():
        for seed in registry["seeds"]:
            for side in SIDES:
                directory = output / f"{tag}-{seed}-{side}"
                directory.mkdir()
                bots = [candidate, binary] if side == "Radiant" else [binary, candidate]
                game = execute_game(directory, server, bots, seed, registry)
                if side == "Dire" and game["result"] in ("win", "loss"):
                    game["result"] = "loss" if game["result"] == "win" else "win"
                game.update(opponent=tag, seed=seed, side=side)
                report["games"].append(game)
                (directory / "result.json").write_text(json.dumps(game, indent=2) + "\n")
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"{tag} seed={seed} side={side}: {game['result']} {game['errors']}", flush=True)
    report["gate"] = evaluate_gate(report["games"], list(opponents), registry["seeds"])
    identities = [(candidate, source_hash), (server, report["server_sha256"])]
    identities.extend((binary, report["opponent_sha256"][tag]) for tag, binary in opponents.items())
    if any(digest(binary) != expected for binary, expected in identities):
        report["gate"]["passed"] = False
        report["gate"]["errors"].append("binary identity changed during run")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2), flush=True)
    return 0 if report["gate"]["passed"] else 1


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--candidate-metadata", required=True,
                        help="Exact build command and source/commit/dirty-state description")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bota-repository", type=Path, default=root.parent / "bota")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", args.run_name):
        parser.error("run-name must be a safe basename of at most 80 characters")
    args.candidate_binary = args.candidate_binary.resolve(strict=True)
    args.bota_repository = args.bota_repository.resolve(strict=True)
    output = root / "artifacts" / "temp" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    try:
        return run(args, root, output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        (output / "failure.json").write_text(json.dumps(dict(passed=False, error=str(error))) + "\n")
        print(f"release evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
