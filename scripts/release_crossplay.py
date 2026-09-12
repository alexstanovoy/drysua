"""Reproducible, fail-closed per-release TCP evaluation. Python stdlib only."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from release_build import (digest, prepare, read_registry,
                           snapshot_weights, weights_name)
from release_wire import CURRENT_SIMULATOR, HISTORICAL_SIMULATOR, Relay

SIDES = ("Radiant", "Dire")
OUTCOMES = ("win", "loss", "draw", "timeout", "error")
SUMMARY = re.compile(r"played (\d+) ticks as Some\((Radiant|Dire)\); winner "
                     r"(None|Some\((Radiant|Dire|Neutral)\)); "
                     r"\d+ decisions, \d+ orders, (\d+) rejected orders\n?")
CLIENT_OUTPUT_LIMIT = 1024 * 1024
TELEMETRY_LINE_LIMIT = 4096
MAP2_TICK_LIMIT = 27000 + 900
POLICY_FIELD = r"policy=(?:teacher|hybrid|neural|tactical)"
RECEIVE_FIELD = r"receive_wait_scope=(?:socket_read|wire_hear_including_decode|mixed_socket_read_and_wire_hear|unavailable)"
SEAT_FIELDS = rf"slot=(?P<slot>[01]) {POLICY_FIELD} mode=(?:lockstep|realtime)"
HISTOGRAM_FIELDS = "".join(
    rf" {name}_count=\d+ {name}_total_ns=\d+ {name}_p50_upper_ns=(?:\d+|unknown)"
    rf" {name}_p95_upper_ns=(?:\d+|unknown) {name}_max_ns=\d+"
    for name in ("compute", "receive_wait", "decision", "order_send", "ack_send"))
ASYNC_LOG_FIELDS = r"(?: dropped_logs=(?P<dropped_logs>0|[1-9][0-9]{0,19}))?"
TELEMETRY = tuple(re.compile(pattern + ASYNC_LOG_FIELDS) for pattern in (
    rf"level=INFO event=live_performance_start {SEAT_FIELDS} tick_rate=30 "
    rf"report_every=\d+ debug_every=\d+ debug_limit=\d+ {RECEIVE_FIELD} "
    r"compute_scope=internal_elapsed_excluding_receive_and_send",
    r"level=(?:INFO|WARN) event=live_performance scope=(?:window|total) "
    r"reason=(?:periodic|match_over|limit) updates=\d+ progress_ticks=\d+ elapsed_ns=\d+ "
    r"updates_per_second=(?:\d+\.\d{3}|unknown) realtime_factor=(?:\d+\.\d{3}|unknown) "
    r"tick_rate=30 budget_ns=\d+ compute_overruns=\d+ service_overruns=\d+ percentiles=log2_upper_bounds"
    rf"{HISTOGRAM_FIELDS} pending_update=(?:true|false) saturated=false {SEAT_FIELDS} "
    rf"{RECEIVE_FIELD} timing_valid=true",
    rf"level=DEBUG event=live_decision slot=(?P<slot>[01]) {POLICY_FIELD} tick=\d+ "
    r"decision_ns=(?:\d+|unknown) order_sent=(?:true|false) order_send_ns=(?:\d+|unknown) "
    r"ack_send_ns=(?:\d+|unknown)",
    r'level=INFO event=live_performance_config_defaulted reason="DRYSUA_PERF_DEBUG_EVERY must be '
    r'(?:Unicode digits|an integer in 0\.\.=4294967295)"',
))
assert TELEMETRY_LINE_LIMIT < CLIENT_OUTPUT_LIMIT
assert MAP2_TICK_LIMIT == 27900


def outcome_summary(text, slot):
    """Find exactly one summary amid known, bounded telemetry; retain the original log."""
    assert slot in (0, 1)
    if len(text) > CLIENT_OUTPUT_LIMIT or len(text.encode("utf-8")) > CLIENT_OUTPUT_LIMIT:
        raise ValueError("client output limit exceeded")
    summary = None
    for line in text.splitlines():
        if len(line) > TELEMETRY_LINE_LIMIT:
            raise ValueError("client output line limit exceeded")
        match = SUMMARY.fullmatch(line)
        if match:
            if summary is not None:
                raise ValueError("duplicate client outcome")
            summary = match
            continue
        records = [pattern.fullmatch(line) for pattern in TELEMETRY]
        record = next((record for record in records if record is not None), None)
        if record is None:
            raise ValueError("invalid client telemetry or unexpected output")
        if record.groupdict().get("slot") is not None and int(record["slot"]) != slot:
            raise ValueError("invalid client telemetry or unexpected output")
        if record["dropped_logs"] is not None and int(record["dropped_logs"]) > 2**64 - 1:
            raise ValueError("invalid client dropped_logs counter")
    return summary


def current_map2_registry(server_sha256, process_timeout_seconds=180):
    """Live Map2 settings for execute_game; not a historical release/challenge gate.

    Bots must carry simulator_commit and sha256 attestations bound to their binaries.
    The caller is responsible for trusted build provenance; bota has no wire fingerprint.
    """
    if not isinstance(server_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", server_sha256):
        raise ValueError("current Map2 requires server SHA256")
    if type(process_timeout_seconds) is not int or not 1 <= process_timeout_seconds <= 600:
        raise ValueError("process timeout must be 1..600 seconds")
    return dict(map=2, tick_limit=MAP2_TICK_LIMIT, simulator_commit=CURRENT_SIMULATOR,
                server_sha256=server_sha256, process_timeout_seconds=process_timeout_seconds,
                wire_byte_limit=2 * 1024**3, record_replay=False)


def validate_runtime_contract(server, bots, registry):
    """Select a pinned decoder and reject verifiable cross-protocol binary mismatches."""
    map_id = registry["map"]
    if map_id not in (0, 1, 2):
        raise ValueError("unsupported evaluator map")
    expected = CURRENT_SIMULATOR if map_id == 2 else HISTORICAL_SIMULATOR
    simulator = registry.get("simulator_commit", HISTORICAL_SIMULATOR)
    if simulator != expected:
        raise ValueError("simulator/map contract mismatch")
    if map_id == 2:
        if type(registry["tick_limit"]) is not int or not 1 <= registry["tick_limit"] <= MAP2_TICK_LIMIT:
            raise ValueError("Map2 tick limit must be 1..27900 including pregame")
        if len(bots) != 2:
            raise ValueError("current Map2 requires two runtime attestations")
        expected_hash = registry.get("server_sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("current Map2 requires server SHA256")
        if digest(server) != expected_hash:
            raise ValueError("server runtime SHA256 mismatch")
    for bot in bots:
        if bot.get("simulator_commit", expected) != expected:
            raise ValueError("bot simulator contract mismatch")
        if map_id == 2:
            identity = bot.get("sha256")
            if (bot.get("simulator_commit") != expected or not isinstance(identity, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", identity)):
                raise ValueError("current Map2 requires digest-bound runtime attestation")
            if digest(bot["binary"]) != identity:
                raise ValueError("bot runtime SHA256 mismatch")
    return simulator


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
    if len(clients) != 2:
        return ("timeout" if timed_out else "error"), ["missing client"]
    errors = []
    winners = []
    for index, client in enumerate(clients):
        wire = client["wire"]
        errors.extend(wire["errors"])
        errors.extend(client.get("output_errors", []))
        if wire["slot"] != index:
            errors.append("identity mismatch")
        if wire["rejected"]:
            errors.append("order rejection")
        if client["exit"] not in (0, None) and not client.get("killed", False):
            errors.append("client process failure")
        try:
            match = outcome_summary(client["stdout"], index)
        except ValueError as error:
            errors.append(str(error))
            match = None
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
        if winner is None and not (wire.get("last_snapshot") == tick_limit
                                   and wire.get("cap_ack") and wire.get("cap_events")):
            errors.append("unverified cap boundary")
        if winner is not None and not (int(ticks) == wire.get("duration") == wire.get("last_snapshot")):
            errors.append("terminal tick mismatch")
        winners.append(winner)
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


def add_candidate_arguments(parser):
    parser.add_argument("--candidate-policy", choices=("teacher", "hybrid", "neural", "tactical"), default="teacher")
    parser.add_argument("--candidate-weights", type=Path,
                        help="Directory containing drysua.weights.safetensors (hybrid/neural) or drysua.tactical.bin (tactical)")


def validate_candidate_arguments(args):
    if args.candidate_policy in ("hybrid", "neural", "tactical") and args.candidate_weights is None:
        raise ValueError(f"{args.candidate_policy} requires --candidate-weights")
    if args.candidate_policy == "teacher" and args.candidate_weights is not None:
        raise ValueError("teacher forbids --candidate-weights")


def bot_command(bot, address, index, tick_limit):
    policy = bot["policy"]
    if policy not in ("teacher", "hybrid", "neural", "tactical") or ("weights" in bot) != (policy != "teacher"):
        raise ValueError("bot policy/weights contract mismatch")
    assert index in (0, 1)
    assert 1 <= tick_limit <= 108900
    command = [str(bot["binary"]), "play", "--policy", policy, "--addr", address,
               "--name", f"release-seat-{index}", "--limit", str(tick_limit)]
    if policy != "teacher":
        command.extend(["--weights-directory", str(bot["weights"])])
    return command


def bot_tick_limit(registry):
    """CLI safety bound for a validated registry; native Map2 waits for terminal frames."""
    tick_limit = registry["tick_limit"]
    assert type(tick_limit) is int
    assert 1 <= tick_limit <= 108900
    if (registry["map"], registry.get("simulator_commit"), tick_limit) == (
            2, CURRENT_SIMULATOR, MAP2_TICK_LIMIT):
        # The CLI exits on the limit Snapshot, before reading its Events/MatchOver.
        return tick_limit + 1
    return tick_limit


def launch(command, path, processes, handles, environment=None):
    handle = path.open("wb")
    handles.append(handle)
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               cwd=path.parent, start_new_session=True, env=environment)
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


def read_client_output(path):
    """Read at most one MiB of UTF-8 client output, detecting growth with a sentinel byte."""
    with path.open("rb") as stream:
        data = stream.read(CLIENT_OUTPUT_LIMIT + 1)
    if len(data) > CLIENT_OUTPUT_LIMIT:
        raise ValueError("client output limit exceeded")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("client output is not UTF-8") from error


def collect(processes, paths, relays, killed):
    clients = []
    for index, (process, path) in enumerate(zip(processes[1:], paths)):
        text, errors = "", []
        try:
            text = read_client_output(path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
        clients.append(dict(exit=process.returncode, stdout=text,
                            killed=process.pid in killed, wire=relays[index].observed,
                            output_errors=errors))
    return clients


def check_resources(directory, paths):
    if any(path.stat().st_size > CLIENT_OUTPUT_LIMIT for path in paths):
        raise ValueError("client output limit exceeded")
    replay = directory / "match.brp"
    if replay.exists() and replay.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("replay size limit exceeded")


def execute_game(directory, server, bots, seed, registry):
    simulator_commit = validate_runtime_contract(server, bots, registry)
    processes, handles, relays, commands, paths = [], [], [], [], []
    killed, errors = set(), []
    timed_out = False
    deadline = time.monotonic() + registry["process_timeout_seconds"]
    environment = None
    if registry.get("cpu_threads") == 1:
        environment = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(server), "--port", "0", "--mode", "lockstep", "--players", "2",
               "--map", str(registry["map"]), "--seed", str(seed), "--ack-timeout-ticks",
               str(registry["process_timeout_seconds"] * 30 + 30)]
    if registry.get("record_replay", True):
        command.extend(["--replay", str(directory / "match.brp")])
    commands.append(command)
    try:
        host = launch(command, directory / "server.log", processes, handles, environment)
        port = server_port(directory / "server.log", host, min(deadline, time.monotonic() + 10))
        for index, bot in enumerate(bots):
            relay = Relay(port, 10, registry["tick_limit"], expected_map=registry["map"],
                          expected_seed=seed,
                          byte_limit=registry.get("wire_byte_limit", 512 * 1024 * 1024),
                          simulator_commit=simulator_commit)
            relays.append(relay)
            command = bot_command(bot, relay.address, index, bot_tick_limit(registry))
            commands.append(command)
            paths.append(directory / f"client-{index}.log")
            launch(command, paths[-1], processes, handles, environment)
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
        await_cap_events(relays, min(deadline, time.monotonic() + 2))
        # A tick-limited server has no MatchOver and must be stopped by the runner.
        if all(relay.observed["winner"] is not None for relay in relays):
            host.wait(timeout=max(0.01, deadline - time.monotonic()))
    except (TimeoutError, subprocess.TimeoutExpired):
        timed_out = True
    except (OSError, ValueError) as error:
        errors.append(str(error))
    finally:
        stop_game(processes, relays, handles, killed, timed_out)
    clients = collect(processes, paths, relays, killed)
    server_exit = processes[0].returncode if processes and processes[0].pid not in killed else None
    result, validation = validate_game(clients, server_exit, timed_out, registry["tick_limit"])
    return dict(result=result, errors=errors + validation, clients=clients, commands=commands,
                cwd=str(directory), server_exit=server_exit, killed_pids=sorted(killed),
                wall_timeout=timed_out,
                wall_seconds=time.monotonic() - deadline + registry["process_timeout_seconds"])


def stop_game(processes, relays, handles, killed, timed_out):
    assert len(processes) <= 3
    assert len(relays) <= 2
    # Host teardown must not turn still-running peers into apparent independent failures.
    killed.update(process.pid for process in processes if process.poll() is None)
    if timed_out:
        for relay in relays:
            relay.stop.set()
    for process in processes:
        if process.pid in killed:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
    drain_deadline = time.monotonic() + 2
    for relay in relays:
        if not timed_out:
            relay.finish(timeout=max(0, min(2, drain_deadline - time.monotonic())))
        relay.close()
    for handle in handles:
        handle.close()


def await_cap_events(relays, deadline):
    for _ in range(201):
        if all(relay.observed.get("last_snapshot") != relay.tick_limit
               or relay.observed.get("winner") is not None
               or relay.observed.get("cap_ack") and relay.observed.get("cap_events") for relay in relays):
            return
        if time.monotonic() >= deadline:
            raise ValueError("cap Events not drained before shutdown")
        time.sleep(0.01)
    raise ValueError("cap Events drain polling limit exceeded")


def run(args, root, output):
    if getattr(args, "teacher_challenge_map0", False):
        from map0_challenge import run as run_challenge
        return run_challenge(args, root, output)
    validate_candidate_arguments(args)
    registry = read_registry(root / "releases.json", root, args.bota_repository)
    candidate = output / "candidate"
    source_hash = digest(args.candidate_binary)
    shutil.copy2(args.candidate_binary, candidate)
    candidate.chmod(0o500)
    if digest(candidate) != source_hash or digest(args.candidate_binary) != source_hash:
        raise ValueError("candidate changed while being snapshotted; retry after build completes")
    candidate_bot = dict(binary=candidate, policy=args.candidate_policy)
    if args.candidate_policy in ("hybrid", "neural", "tactical"):
        candidate_bot["weights"] = output / "candidate-weights"
        candidate_bot["weights_metadata"] = snapshot_weights(
            args.candidate_weights, candidate_bot["weights"], policy=args.candidate_policy)
    harness = output / "harness"
    harness.mkdir()
    for name in ("release_build.py", "release_crossplay.py", "release_wire.py"):
        shutil.copy2(root / "scripts" / name, harness / name)
    report = dict(evaluation="historical-map1",
                  registry=registry, candidate=dict(source=str(args.candidate_binary),
                  sha256=source_hash, policy=args.candidate_policy,
                  weights=candidate_bot.get("weights_metadata"),
                  command_metadata=args.candidate_metadata), games=[],
                  invocation=sys.argv, python=sys.version, platform=platform.platform(),
                  gate=dict(passed=False, errors=["incomplete run"]))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    server, opponents = prepare(root, args.bota_repository, output, registry)
    report["evaluation_map"] = registry["map"]
    report["server_sha256"] = digest(server)
    report["opponent_sha256"] = {tag: digest(bot["binary"]) for tag, bot in opponents.items()}
    report["opponent_policies"] = {tag: dict(policy=bot["policy"], weights=bot.get("weights_metadata"))
                                   for tag, bot in opponents.items()}
    for tag, bot in opponents.items():
        for seed in registry["seeds"]:
            for side in SIDES:
                directory = output / f"{tag}-{seed}-{side}"
                directory.mkdir()
                bots = [candidate_bot, bot] if side == "Radiant" else [bot, candidate_bot]
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
    identities.extend((bot["binary"], report["opponent_sha256"][tag]) for tag, bot in opponents.items())
    identities.extend((bot["weights"] / weights_name(bot["policy"]), bot["weights_metadata"]["sha256"])
                      for bot in [candidate_bot, *opponents.values()] if "weights" in bot)
    if any(digest(binary) != expected for binary, expected in identities):
        report["gate"]["passed"] = False
        report["gate"]["errors"].append("binary or weights identity changed during run")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2), flush=True)
    return 0 if report["gate"]["passed"] else 1


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_candidate_arguments(parser)
    parser.add_argument("--teacher-challenge-map0", action="store_true",
                        help="Separate pure Neural challenge against an explicitly frozen Teacher; not the Map1 release gate")
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--challenge-final", action="store_true",
                        help="Fixed 50 paired held-out seeds; never use for optimizer or stage selection")
    parser.add_argument("--challenge-pairs", type=int, help="Development only: 1..10 pairs (default 10)")
    parser.add_argument("--challenge-workers", type=int, default=4)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--candidate-metadata", required=True,
                        help="Exact build command and source/commit/dirty-state description")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bota-repository", type=Path, default=root.parent / "bota")
    args = parser.parse_args()
    try:
        validate_candidate_arguments(args)
        if args.teacher_challenge_map0:
            from map0_challenge import validate_arguments
            validate_arguments(args)
        elif args.baseline_manifest or args.challenge_final or args.challenge_pairs is not None or args.challenge_workers != 4:
            raise ValueError("challenge options require --teacher-challenge-map0")
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", args.run_name):
        parser.error("run-name must be a safe basename of at most 80 characters")
    args.candidate_binary = args.candidate_binary.resolve(strict=True)
    args.bota_repository = args.bota_repository.resolve(strict=True)
    if args.baseline_manifest is not None:
        args.baseline_manifest = args.baseline_manifest.absolute()
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
