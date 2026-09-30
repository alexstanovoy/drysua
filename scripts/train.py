#!/usr/bin/env python3
"""Training campaigns: one long-lived rootless container per session, verified by a frozen inspector."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
from time import monotonic
import uuid

import train_io as io
import train_report as report
import train_session as session_backend

SCHEMA = "drysua-training-campaign/v2"
RECEIPT_SCHEMA = "drysua-training-session/v1"
INSPECTION_SCHEMA = "drysua-checkpoint-inspection/v1"
MAX_FILE = 512 * 1024 * 1024
MAX_FROZEN = 1024 * 1024 * 1024
MAX_SESSIONS = 10000
MAX_CHECKPOINT_FILES = 10004
RUNTIME_FILE = "drysua.weights.safetensors"
CHECKPOINT_MANIFESTS = ("checkpoint.meta", "checkpoint.meta.previous")
SOURCES = ("train.py", "train_dashboard.html", "train_io.py", "train_report.py", "train_session.py")
SOURCE_DIRECTORY = Path(__file__).absolute().parent
REPOSITORY = SOURCE_DIRECTORY.parent
PHASES = {"prepared", "running", "paused", "failed", "completed"}
STATUS_FIELDS = {"schema", "campaign_id", "manifest_sha256", "phase", "updates", "sessions",
                 "scope_sha256", "last_error"}
CAMPAIGN_DIRECTORIES = ("bin", "inputs", "frozen", "checkpoint", "history", "cuda-cache", "sessions")
# An allowlist also blocks future output options unknown to this controller.
ARGUMENTS = frozenset({
    "--samples-per-update", "--slots", "--lanes", "--simulation-threads", "--simulation-groups", "--pin-threads",
    "--generation-updates",
    "--seed", "--training-microbatch",
    "--balanced-minibatches", "--environment-schedule",
    "--environment-success-updates", "--environment-success-rate",
    "--environment-poor-updates", "--environment-poor-rate", "--environment-extension",
    "--zero-updates", "--learning-rate", "--epochs", "--minibatch", "--gae-lambda",
    "--entropy-coefficient", "--environment-scale-start", "--environment-scale-end",
})
CONFIG_FIELDS = {"schema", "trainer", "inspector", "initial_weights", "opponent_weights", "total_updates",
                 "history_every", "checkpoint_seconds", "max_seconds", "stop_seconds",
                 "training_args", "mode", "docker_context",
                 "image", "gpu_uuid", "lock_paths", "memory_gib", "cuda_directory"}


def validate_arguments(arguments):
    if not isinstance(arguments, list) or len(arguments) > 128:
        raise ValueError("training_args must be a list of at most 128 strings")
    previous_option = False
    for argument in arguments:
        if not isinstance(argument, str) or not 1 <= len(argument) <= 4096:
            raise ValueError("training_args strings must contain 1..4096 characters")
        if any(ord(character) < 32 for character in argument):
            raise ValueError("training_args cannot contain control characters")
        if argument.startswith("--"):
            if argument.split("=", 1)[0] not in ARGUMENTS:
                raise ValueError(f"controller-owned or unreviewed argument: {argument}")
            previous_option = "=" not in argument
        elif argument.startswith("-") or not previous_option:
            raise ValueError(f"unexpected training argument: {argument}")
        else:
            previous_option = False
    return arguments


def validate_config(value):
    if not isinstance(value, dict) or set(value) - CONFIG_FIELDS:
        raise ValueError(f"unknown configuration fields: {sorted(set(value) - CONFIG_FIELDS)}")
    if type(value.get("schema")) is not int or value["schema"] != 2:
        raise ValueError("unsupported configuration schema; expected 2")
    result = {"schema": 2}
    for name, default, lower, upper in (("total_updates", None, 1, 10000), ("history_every", 20, 1, 10000),
                                        ("checkpoint_seconds", 600, 60, 86400),
                                        ("max_seconds", 86400, 1, 604800), ("stop_seconds", 300, 5, 3600),
                                        ("memory_gib", 12, 1, 48)):
        result[name] = io.bounded_integer(value.get(name, default), name, lower, upper)
    result["training_args"] = validate_arguments(value.get("training_args", []))
    for name in ("trainer", "inspector", "initial_weights", "opponent_weights"):
        entry = value.get(name, value.get("trainer") if name == "inspector" else None)
        if entry is None and name in {"initial_weights", "opponent_weights"}:
            continue
        if not isinstance(entry, str) or not 1 <= len(entry) <= 4096:
            raise ValueError(f"{name} must be a nonempty path")
        result[name] = entry
    result.update(validate_execution(value))
    return result


def validate_execution(value):
    mode = value.get("mode", "cpu")
    context = value.get("docker_context", "rootless")
    image, gpu = value.get("image"), value.get("gpu_uuid")
    cuda = value.get("cuda_directory", "/usr/local/cuda-13.3")
    if mode not in {"cpu", "gpu"}:
        raise ValueError("mode must be cpu or gpu")
    if not isinstance(context, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", context):
        raise ValueError("invalid docker_context")
    if not isinstance(image, str) or not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]{0,200}@sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be pinned by sha256 digest")
    if (mode == "cpu" and gpu is not None) or (mode == "gpu" and (not isinstance(gpu, str) or not re.fullmatch(
            r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu))):
        raise ValueError("gpu_uuid must be a full GPU UUID in gpu mode, and null in cpu mode")
    if not isinstance(cuda, str) or not re.fullmatch(r"/usr/local/[A-Za-z0-9_.-]{1,64}", cuda):
        raise ValueError("cuda_directory must be a directory directly under /usr/local")
    paths = value.get("lock_paths", [str(REPOSITORY / "heavy.lock")])
    if not isinstance(paths, list) or len(paths) > 8 or not all(isinstance(path, str) for path in paths):
        raise ValueError("lock_paths must contain at most 8 paths")
    normalized = [str(io.private_path(path)) for path in paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate lock_paths")
    return dict(mode=mode, docker_context=context, image=image, gpu_uuid=gpu, lock_paths=normalized,
                cuda_directory=cuda)


def collect_inputs(config, base):
    files = [(f"frozen/{name}", SOURCE_DIRECTORY / name) for name in SOURCES]
    for name in ("trainer", "inspector"):
        files.append((f"bin/{name}", io.private_path(base / config[name])))
    for key, target in (("initial_weights", "initial_weights"), ("opponent_weights", "opponent")):
        if key not in config:
            continue
        source = io.private_path(base / config[key])
        if source.is_dir():
            source = io.private_path(source / RUNTIME_FILE)
        if not source.is_file():
            raise ValueError(f"{key} must select a regular runtime weights file")
        files.append((f"inputs/{target}/{RUNTIME_FILE}", source))
    return sorted(files)


def freeze_inputs(directory, inputs):
    records, total = {}, 0
    for relative, source in inputs:
        data = io.read_bytes(source, MAX_FILE)
        total += len(data)
        if total > MAX_FROZEN:
            raise ValueError("frozen inputs exceed 1 GiB")
        if relative == "bin/trainer" and not data.startswith(b"\x7fELF"):
            raise ValueError("trainer must be a native ELF executable, never a script")
        target = directory / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        io.write_exclusive(target, data, 0o500 if relative.startswith("bin/") else 0o400)
        records[relative] = {"size": len(data), "sha256": io.digest(data)}
    for relative in {str(Path(relative).parent) for relative in records}:
        io.fsync_directory(directory / relative)
    return records


def create(config_path, campaign_path):
    config_path, directory = io.private_path(config_path), io.private_path(campaign_path)
    config = validate_config(io.read_json(config_path))
    inputs = collect_inputs(config, config_path.parent)
    if directory.exists():
        raise ValueError("campaign already exists; never overwrite or adopt implicitly")
    if not directory.parent.is_dir():
        raise ValueError("campaign parent must already exist")
    directory.mkdir(mode=0o700)
    io.fsync_directory(directory.parent)
    for name in CAMPAIGN_DIRECTORIES:
        (directory / name).mkdir(mode=0o700)
    io.write_exclusive(directory / "owner.lock", b"", 0o600)
    records = freeze_inputs(directory, inputs)
    frozen = dict(config, trainer="bin/trainer", inspector="bin/inspector")
    for key, target in (("initial_weights", "inputs/initial_weights"), ("opponent_weights", "inputs/opponent")):
        if key in frozen:
            frozen[key] = target
    manifest = {"schema": SCHEMA, "campaign_id": uuid.uuid4().hex, "config": frozen, "files": records}
    payload = io.encode(manifest)
    io.write_exclusive(directory / "manifest.json", payload)
    io.atomic_json(directory / "status.json", {
        "schema": SCHEMA, "campaign_id": manifest["campaign_id"], "manifest_sha256": io.digest(payload),
        "phase": "prepared", "updates": 0, "sessions": 0, "scope_sha256": None, "last_error": None})
    return status(directory)


def load_campaign(campaign_path):
    """Validated status and manifest of an owned, private campaign directory."""
    directory = io.private_path(campaign_path)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("campaign must be owned by this user with mode 0700")
    current = io.read_json(directory / "status.json")
    if (set(current) != STATUS_FIELDS or current["schema"] != SCHEMA or current["phase"] not in PHASES):
        raise ValueError("unsupported or corrupt campaign status (schema 2 required)")
    payload = io.read_bytes(directory / "manifest.json", io.MAX_JSON)
    if io.digest(payload) != current["manifest_sha256"]:
        raise ValueError("manifest integrity mismatch")
    manifest = io.decode_json(payload)
    if set(manifest) != {"schema", "campaign_id", "config", "files"} or manifest["schema"] != SCHEMA:
        raise ValueError("unsupported manifest")
    if manifest["campaign_id"] != current["campaign_id"] or not re.fullmatch(r"[0-9a-f]{32}", current["campaign_id"]):
        raise ValueError("campaign identity mismatch")
    config = validate_config(manifest["config"])
    if config["trainer"] != "bin/trainer" or config["inspector"] != "bin/inspector":
        raise ValueError("manifest references an unfrozen executable")
    io.bounded_integer(current["updates"], "updates", 0, config["total_updates"])
    io.bounded_integer(current["sessions"], "sessions", 0, MAX_SESSIONS)
    if current["phase"] == "completed" and current["updates"] != config["total_updates"]:
        raise ValueError("corrupt status: completed below the update target")
    return directory, current, manifest


def verify_frozen(directory, manifest):
    expected = {f"frozen/{name}" for name in SOURCES} | {"bin/trainer", "bin/inspector"}
    optional = {f"inputs/initial_weights/{RUNTIME_FILE}", f"inputs/opponent/{RUNTIME_FILE}"}
    if not expected <= manifest["files"].keys() <= expected | optional:
        raise ValueError("unexpected frozen file inventory")
    for relative, record in manifest["files"].items():
        data = io.read_bytes(directory / relative, MAX_FILE)
        if record != {"size": len(data), "sha256": io.digest(data)} or (directory / relative).stat().st_mode & 0o222:
            raise ValueError(f"frozen artifact integrity mismatch: {relative}")


def status(campaign_path, inspect=False):
    directory, current, manifest = load_campaign(campaign_path)
    verify_frozen(directory, manifest)
    owner = owner_record(directory)
    result = dict(current, total_updates=manifest["config"]["total_updates"],
                  owner_live=bool(owner and io.is_live(owner)),
                  container_id=owner["container_id"] if owner else None)
    if inspect:
        try:
            inspection = inspect_checkpoint(directory)
            result["checkpoint"] = None if inspection is None else summary(inspection)
        except ValueError as error:
            result["checkpoint_error"] = str(error)
    return result


def committed(checkpoint):
    return any((checkpoint / name).exists() for name in CHECKPOINT_MANIFESTS)


def inspector_json(directory, *arguments):
    payload = io.capture([str(directory / "bin/inspector"), "checkpoint-inspect", *arguments],
                         timeout=30, limit=io.MAX_JSON)
    value = io.decode_json(payload)
    if value.get("schema") != INSPECTION_SCHEMA:
        raise ValueError("unsupported native inspector schema")
    return value


def validate_contract(value):
    capabilities = value.get("capabilities", {})
    if (value.get("kind") != "contract" or not isinstance(capabilities, dict) or
            any(capabilities.get(key) is not True for key in ("inspection", "annealed_history", "read_only")) or
            capabilities.get("controller_run_kind") != "train-annealed"):
        raise ValueError("native inspector contract lacks read-only annealed inspection")
    model = value.get("model", {})
    if not isinstance(model, dict) or not re.fullmatch(r"[0-9a-f]{16}", str(model.get("hash", ""))):
        raise ValueError("invalid native inspector model contract")


def inspect_checkpoint(directory):
    """Native projection of the committed checkpoint, or None before the first commit."""
    checkpoint = directory / "checkpoint"
    if not committed(checkpoint):
        return None
    value = inspector_json(directory, "--checkpoint-directory", str(checkpoint))
    required = {"identity", "model", "progress", "run", "ppo", "adaptive", "files", "kind", "history",
                "runtime_status", "runtime_matches_model", "recovery_required"}
    if not required <= value.keys() or value["kind"] != "train-annealed":
        raise ValueError("unsupported native inspection kind or fields")
    history = value["history"]
    if not isinstance(history, dict) or history.get("verified") is not True:
        raise ValueError("native annealed history is not verified")
    identity = value["identity"]
    if not isinstance(identity, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("scope_sha256"))):
        raise ValueError("invalid native checkpoint identity")
    progress = value["progress"]
    for name in ("updates", "optimizer_steps", "rollout_samples", "games"):
        io.bounded_integer(progress.get(name) if isinstance(progress, dict) else None, f"native {name}", 0, 10**15)
    if not isinstance(value["files"], list) or not 1 <= len(value["files"]) <= MAX_CHECKPOINT_FILES:
        raise ValueError("invalid native checkpoint file inventory")
    return value


def summary(inspection):
    """The small, durable part of one inspection kept in session receipts."""
    progress = inspection["progress"]
    return {"updates": progress["updates"], "optimizer_steps": progress["optimizer_steps"],
            "rollout_samples": progress["rollout_samples"], "games": progress["games"],
            "identity": inspection["identity"], "runtime_status": inspection["runtime_status"],
            "runtime_matches_model": inspection["runtime_matches_model"],
            "recovery_required": inspection["recovery_required"], "adaptive": inspection["adaptive"],
            "scope": {name: io.digest(io.encode(inspection[name])) for name in ("model", "run", "ppo")}}


def accept_progress(current, inspection, start_updates):
    """Updates status from a verified inspection; scope never changes and progress never rolls back."""
    if inspection is None:
        if start_updates:
            raise ValueError("committed checkpoint disappeared")
        return None
    record = summary(inspection)
    scope = io.digest(io.encode([record["identity"]["scope_sha256"], record["scope"]]))
    if current["scope_sha256"] not in (None, scope):
        raise ValueError("native checkpoint scope changed")
    if record["updates"] < start_updates:
        raise ValueError(f"native progress rolled back: {record['updates']} < {start_updates}")
    current.update(updates=record["updates"], scope_sha256=scope)
    return record


def owner_record(directory):
    path = directory / "owner.json"
    if not path.exists():
        return None
    value = io.read_json(path)
    if (set(value) != {"pid", "start", "uid", "boot", "token", "session", "container_id"} or
            not re.fullmatch(r"[0-9a-f]{32}", str(value.get("token")))):
        raise ValueError("invalid owner record")
    return value


def request_control(campaign_path, operation):
    directory, _, _ = load_campaign(campaign_path)
    owner = owner_record(directory)
    if owner is None or not io.is_live(owner) or not io.is_locked(directory / "owner.lock"):
        raise ValueError("no live controller owns this campaign; use recover for a stale owner")
    path = directory / f"{operation}-{owner['token']}.json"
    if not path.exists():
        io.write_exclusive(path, io.encode({"token": owner["token"]}), 0o600)
    return {"phase": "requested", "operation": operation, "session": owner["session"]}


def requested(directory, owner, operation):
    return (directory / f"{operation}-{owner['token']}.json").exists()


def release_owner(directory, owner):
    for operation in ("pause", "stop"):
        path = directory / f"{operation}-{owner['token']}.json"
        if path.exists():
            path.unlink()
    (directory / "owner.json").unlink()
    io.fsync_directory(directory)


def trainer_command(directory, config, resume):
    """Native argv; every path and lifecycle flag is controller-owned."""
    command = [str(directory / "bin/trainer"), "train-annealed", *config["training_args"],
               "--updates", str(config["total_updates"]),
               "--checkpoint-directory", str(directory / "checkpoint"),
               "--history-directory", str(directory / "history"), "--history-every", str(config["history_every"]),
               "--checkpoint-interval-seconds", str(config["checkpoint_seconds"]),
               "--device", "cuda" if config["mode"] == "gpu" else "cpu"]
    if config["mode"] == "gpu":
        command += ["--device-ordinal", "0"]
    if resume:
        command.append("--resume")
    elif "initial_weights" in config:
        command += ["--initial-weights", str(directory / config["initial_weights"])]
    if "opponent_weights" in config:
        command += ["--opponent", f"weights:{directory / config['opponent_weights']}:1"]
    return command


def preflight(directory, current, manifest):
    """Everything that must hold before a container exists; returns the GPU device index."""
    config = manifest["config"]
    verify_frozen(directory, manifest)
    for name in SOURCES:
        if io.digest(io.read_bytes(SOURCE_DIRECTORY / name, MAX_FILE)) != manifest["files"][f"frozen/{name}"]["sha256"]:
            raise ValueError("controller source differs from the frozen snapshot; run the campaign's frozen/train.py")
    validate_contract(inspector_json(directory, "--contract"))
    inspection = inspect_checkpoint(directory)
    if current["updates"]:
        if inspection is None or inspection["progress"]["updates"] != current["updates"]:
            raise ValueError("committed checkpoint differs from the accepted status; run recover")
        accept_progress(dict(current), inspection, current["updates"])
    elif inspection is not None:
        raise ValueError("checkpoint holds unaccepted progress; run recover")
    else:
        # A session killed before its first commit leaves only disposable files.
        for entry in (directory / "checkpoint").iterdir():
            shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
    session_backend.verify_daemon(config)
    health = session_backend.host_sample(config, directory)
    violation = session_backend.health_violation(health, admission=True)
    if violation:
        raise ValueError(f"admission refused: {violation}")
    return health.get("index")


class ControllerSignals:
    """First SIGINT/SIGTERM asks for a graceful stop, a second one for an immediate kill."""

    def __init__(self):
        self.count, self.previous = 0, {}

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self.receive)
        return self

    def __exit__(self, *_):
        for number, handler in self.previous.items():
            signal.signal(number, handler)

    def receive(self, *_):
        self.count += 1


def session_control(directory, owner, signals, deadline):
    def control():
        if signals.count > 1 or requested(directory, owner, "stop"):
            return ("stop", "stop requested")
        if signals.count == 1 or requested(directory, owner, "pause"):
            return ("pause", "pause requested")
        if monotonic() >= deadline:
            return ("pause", "session deadline reached")
        return None
    return control


class Handshake:
    """One-shot message to a detaching parent: `ready` once the container runs, or the error."""

    def __init__(self, descriptor=None):
        if descriptor is not None and (descriptor < 3 or not stat.S_ISFIFO(os.fstat(descriptor).st_mode)):
            raise ValueError("handshake descriptor must be an inherited pipe")
        self.descriptor = descriptor

    def finish(self, message):
        if self.descriptor is None:
            return
        try:
            os.write(self.descriptor, message[:4096])
        finally:
            os.close(self.descriptor)
            self.descriptor = None


def run_campaign(campaign_path, resume=False, handshake=None):
    started = monotonic()
    handshake = handshake or Handshake()
    directory = io.private_path(campaign_path)
    with io.locks([directory / "owner.lock"]):
        directory, current, manifest = load_campaign(directory)
        if owner_record(directory) is not None:
            raise ValueError("a previous session was not closed; run recover")
        allowed = {"paused", "failed"} if resume else {"prepared"}
        if current["phase"] not in allowed:
            raise ValueError(f"{'resume' if resume else 'run'} requires phase {sorted(allowed)}, "
                             f"campaign is {current['phase']}")
        config = manifest["config"]
        with io.locks(config["lock_paths"]):
            gpu_index = preflight(directory, current, manifest)
            number = io.bounded_integer(current["sessions"] + 1, "session", 1, MAX_SESSIONS)
            session_directory = directory / "sessions" / f"{number:04d}"
            session_directory.mkdir(mode=0o700)
            owner = dict(io.process_identity(os.getpid()), token=uuid.uuid4().hex, session=number, container_id=None)
            io.write_exclusive(directory / "owner.json", io.encode(owner), 0o600)
            start_updates = current["updates"]
            current.update(phase="running", sessions=number, last_error=None)
            io.atomic_json(directory / "status.json", current)
            try:
                with ControllerSignals() as signals:
                    command = trainer_command(directory, config, resume=start_updates > 0)
                    owner["container_id"] = session_backend.create_container(
                        directory, config, current["campaign_id"], number, command, gpu_index)
                    io.atomic_json(directory / "owner.json", owner)
                    session_backend.verify_declared_limits(config, session_backend.inspect_owned(
                        config, owner["container_id"], current["campaign_id"], number))
                    handshake.finish(b"ready\n")
                    outcome = session_backend.Supervisor(
                        config, owner["container_id"], current["campaign_id"], number, session_directory,
                        session_control(directory, owner, signals, started + config["max_seconds"])).run(started)
                return finish_session(directory, current, config, owner, start_updates, outcome, started)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                abandon_session(directory, current, config, owner, start_updates, error)
                raise ValueError(str(error)) from error


def finish_session(directory, current, config, owner, start_updates, outcome, started):
    record = accept_progress(current, inspect_checkpoint(directory), start_updates)
    error = None
    if not outcome["verified_limits"]:
        error = "actual container cgroup limits were never verified"
    elif outcome["request"] is None and outcome["exit_code"] != 0:
        error = f"trainer exited with code {outcome['exit_code']} (oom_killed={outcome['oom_killed']})"
    elif outcome["request"] == "pause" and outcome["exit_code"] != 0:
        error = f"trainer failed during a graceful stop with code {outcome['exit_code']}"
    elif outcome["request"] is None and current["updates"] < config["total_updates"]:
        error = "trainer exited before the update target without a stop request"
    if current["updates"] == config["total_updates"] and error is None:
        phase = "completed"
    else:
        phase = "failed" if error else "paused"
    user_reasons = {None, "pause requested", "stop requested"}
    note = None if outcome["reason"] in user_reasons else outcome["reason"]
    if record is not None and (record["recovery_required"] or not record["runtime_matches_model"]):
        # A kill during a save leaves the previous manifest or runtime export; resume repairs both.
        note = "; ".join(filter(None, [note, "checkpoint was selected from its recovery copy"]))
    current.update(phase=phase, last_error=error or note)
    write_receipt(directory, current, owner, start_updates, record, dict(outcome, error=error), started)
    io.atomic_json(directory / "status.json", current)
    release_owner(directory, owner)
    return current


def write_receipt(directory, current, owner, start_updates, record, outcome, started):
    receipt = {"schema": RECEIPT_SCHEMA, "campaign_id": current["campaign_id"], "session": owner["session"],
               "container_id": owner["container_id"], "phase": current["phase"], "start_updates": start_updates,
               "end_updates": current["updates"], "wall_seconds": round(monotonic() - started, 3),
               "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "outcome": outcome, "checkpoint": record}
    path = directory / "sessions" / f"{owner['session']:04d}" / "receipt.json"
    io.write_exclusive(path, io.encode(receipt))


def abandon_session(directory, current, config, owner, start_updates, error):
    """Best-effort close of a failed session; a leftover container keeps the owner for recover."""
    current.update(phase="failed", last_error=str(error)[:1024])
    try:
        if owner["container_id"]:
            session_backend.remove_orphan(config, owner["container_id"], current["campaign_id"], owner["session"])
        record = accept_progress(current, inspect_checkpoint(directory), start_updates)
        write_receipt(directory, current, owner, start_updates, record,
                      {"request": None, "reason": "controller error", "error": str(error)[:1024]}, monotonic())
        release_owner(directory, owner)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as cleanup:
        current["last_error"] = f"{error}; cleanup failed, run recover: {cleanup}"[:1024]
    io.atomic_json(directory / "status.json", current)


def recover(campaign_path, confirm=False):
    """Closes a session whose controller died: removes its container and accepts the committed checkpoint."""
    if not confirm:
        raise ValueError("recover requires explicit --confirm-offline")
    directory = io.private_path(campaign_path)
    with io.locks([directory / "owner.lock"]):
        directory, current, manifest = load_campaign(directory)
        owner = owner_record(directory)
        if owner is not None and io.is_live(owner):
            raise ValueError("recover refuses a live controller")
        config = manifest["config"]
        with io.locks(config["lock_paths"]):
            if owner is not None and owner["container_id"]:
                session_backend.remove_orphan(config, owner["container_id"], current["campaign_id"], owner["session"])
            start_updates = current["updates"]
            record = accept_progress(current, inspect_checkpoint(directory), start_updates)
            current.update(phase="completed" if current["updates"] == config["total_updates"] else "paused",
                           last_error=None)
            if owner is not None:
                receipt = directory / "sessions" / f"{owner['session']:04d}" / "receipt.json"
                if not receipt.exists():
                    write_receipt(directory, current, owner, start_updates, record,
                                  {"request": None, "reason": "recovered after controller loss"}, monotonic())
                release_owner(directory, owner)
            io.atomic_json(directory / "status.json", current)
            return current


def detach(campaign_path, resume=False):
    """Starts the frozen controller in the background once it owns the campaign."""
    directory, _, _ = load_campaign(campaign_path)
    reader, writer = os.pipe()
    try:
        command = [sys.executable, "-B", str(directory / "frozen/train.py"), "resume" if resume else "run",
                   str(directory), "--ready-fd", str(writer)]
        child = subprocess.Popen(command, pass_fds=(writer,), start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=io.environment())
        os.close(writer)
        writer = None
        ready, _, _ = select.select([reader], [], [], 120)
        message = os.read(reader, 4096) if ready else b""
        if message != b"ready\n":
            io.terminate(child)
            detail = message.decode("utf-8", "replace").strip() or "no handshake within 120 s"
            raise ValueError(f"detached controller did not start: {detail}")
        return {"phase": "started", "pid": child.pid}
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("create", help="freeze a new, never-run campaign")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("campaign", type=Path)
    report.add_arguments(commands.add_parser("report", help="read-only statistics and HTML dashboard"))
    for name in ("status", "run", "resume", "pause", "stop", "recover"):
        command = commands.add_parser(name)
        command.add_argument("campaign", type=Path)
        if name == "status":
            command.add_argument("--inspect", action="store_true", help="also run the native inspector")
        if name in {"run", "resume"}:
            command.add_argument("--detach", action="store_true")
            command.add_argument("--ready-fd", type=int, help=argparse.SUPPRESS)
        if name == "recover":
            command.add_argument("--confirm-offline", action="store_true")
    options = parser.parse_args(arguments)
    if options.operation == "report":
        return report.main(options)
    if options.operation == "create":
        result = create(options.config, options.campaign)
    elif options.operation == "status":
        result = status(options.campaign, inspect=options.inspect)
    elif options.operation in {"run", "resume"}:
        resume = options.operation == "resume"
        if options.detach and options.ready_fd is not None:
            raise ValueError("detach cannot override the handshake descriptor")
        if options.detach:
            result = detach(options.campaign, resume)
        else:
            handshake = Handshake(options.ready_fd)
            try:
                result = run_campaign(options.campaign, resume, handshake)
            except (OSError, ValueError, TypeError, KeyError) as error:
                handshake.finish(f"error: {error}".encode())
                raise
    elif options.operation in {"pause", "stop"}:
        result = request_control(options.campaign, options.operation)
    else:
        result = recover(options.campaign, confirm=options.confirm_offline)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"training controller: {error}", file=sys.stderr)
        sys.exit(2)
