#!/usr/bin/env python3
"""Tracked, bounded training through a frozen runner and native inspector."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import sys
from time import monotonic
import uuid

import train_state as state
from train_runner import DEFAULT_INVOCATION_SECONDS, resolve_timeouts


MAX_JSON = 4 * 1024 * 1024
MAX_FILE = 512 * 1024 * 1024
MAX_FILES = 128
MAX_CHECKPOINT_FILES = 10004
MAX_TOTAL = 1024 * 1024 * 1024
MAX_CAMPAIGN_BYTES = 100 * MAX_TOTAL
MAX_JOB_ENTRIES = MAX_CHECKPOINT_FILES + 128
MAX_CAMPAIGN_ENTRIES = 2560256
PERSISTENCE_RESERVE_BYTES = 16 * 1024 * 1024
JOB_DISK_RESERVE_BYTES = 2 * MAX_TOTAL + 64 * 1024 * 1024
INSPECTION_SCHEMA = "drysua-checkpoint-inspection/v1"
RUNTIME_FILE = "drysua.weights.safetensors"
ARTIFACT_IDENTITIES = {"manifest_sha256": "manifest", "tensor_sha256": "tensor", "runtime_sha256": "runtime"}
SCHEMA = "drysua-training-campaign/v1"
RUNNER_SOURCE = Path(__file__).absolute().with_name("train_runner.py")
WORKSPACE_ROOT = Path(__file__).absolute().parents[1]
# An allowlist also blocks future output options unknown to this controller.
ARGUMENTS = frozenset({
    "--games", "--parallel", "--generation-games", "--seed", "--map",
    "--host-math-workers", "--training-microbatch", "--actor-pipeline-groups",
    "--balanced-minibatches", "--reuse-actor-values", "--environment-schedule",
    "--environment-success-updates", "--environment-success-rate",
    "--environment-poor-updates", "--environment-poor-rate", "--environment-extension",
    "--zero-updates", "--learning-rate", "--epochs", "--minibatch", "--gae-lambda",
    "--entropy-coefficient", "--opponent-inference",
})


def private_path(value):
    path = Path(os.path.abspath(value))
    if ".." in Path(value).parts:
        raise ValueError("parent traversal is forbidden")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink is forbidden: {part}")
    return path


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_bytes(path, limit):
    path = private_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path}")
        value = source.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"file exceeds size limit: {path}")
    return value


def read_json(path):
    return decode_json(read_bytes(path, MAX_JSON))


def decode_json(data):
    try:
        value = json.loads(data, object_pairs_hook=unique_pairs,
                           parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def bounded_integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return value


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
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    allowed = {"schema", "trainer", "inspector", "initial_weights", "opponent_weights", "total_updates",
               "invocation_updates", "invocation_seconds", "max_seconds", "history_every", "training_args",
               "workspace_root", "mode", "docker_context", "image", "gpu_uuid", "lock_paths"}
    if set(value) - allowed:
        raise ValueError("unknown configuration fields")
    if type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("unsupported configuration schema")
    result = dict(value)
    result["invocation_seconds"] = resolve_timeouts(
        value.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS)).payload_seconds
    for name, default, maximum in (("total_updates", None, 10000),
                                   ("invocation_updates", 1, 16),
                                   ("max_seconds", 18000, 86400),
                                   ("history_every", 20, 10000)):
        result[name] = bounded_integer(value.get(name, default), name, 1, maximum)
    result["training_args"] = validate_arguments(value.get("training_args", []))
    for name in ("trainer", "inspector", "initial_weights", "opponent_weights"):
        entry = value.get(name, value.get("trainer") if name == "inspector" else None)
        if entry is None and name in {"initial_weights", "opponent_weights"} and name not in value:
            continue
        if not isinstance(entry, str) or not entry or len(entry) > 4096:
            raise ValueError(f"{name} must be a nonempty path")
        result[name] = entry
    result.update(validate_execution_config(value))
    return result


def validate_execution_config(value):
    workspace = private_path(value.get("workspace_root", WORKSPACE_ROOT))
    if not workspace.is_dir():
        raise ValueError("workspace_root must be an existing directory")
    mode = value.get("mode", "cpu")
    context = value.get("docker_context", "rootless")
    image = value.get("image")
    gpu = value.get("gpu_uuid")
    if mode not in {"cpu", "gpu"}:
        raise ValueError("mode must be cpu or gpu")
    if not isinstance(context, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", context):
        raise ValueError("invalid docker_context")
    if image is not None and (not isinstance(image, str) or not re.fullmatch(
            r"[A-Za-z0-9_./:-]{1,256}@sha256:[0-9a-f]{64}", image)):
        raise ValueError("image must be pinned by sha256 digest")
    if ((mode == "cpu" and gpu is not None) or (mode == "gpu" and
            (not isinstance(gpu, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]{1,80}", gpu)))):
        raise ValueError("gpu_uuid must select one GPU in gpu mode, and be null in cpu mode")
    paths = value.get("lock_paths", [str(workspace / "heavy.lock")])
    if not isinstance(paths, list) or not 1 <= len(paths) <= 8:
        raise ValueError("lock_paths must contain 1..8 paths")
    normalized = [str(private_path(path)) for path in paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate lock_paths")
    return dict(workspace_root=str(workspace), mode=mode, docker_context=context,
                image=image, gpu_uuid=gpu, lock_paths=normalized)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path, data, mode=0o400):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(data)
        target.flush()
        os.fchmod(target.fileno(), mode)
        os.fsync(target.fileno())
    fsync_directory(path.parent)


def atomic_status(directory, value):
    atomic_json(directory / "status.json", value)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    write_exclusive(temporary, encode(value), 0o600)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def collect_inputs(config, base):
    files = [("frozen/controller.py", Path(__file__).absolute()),
             ("frozen/train_state.py", Path(state.__file__).absolute())]
    if RUNNER_SOURCE.exists():
        files.append(("frozen/train_runner.py", private_path(RUNNER_SOURCE)))
    for name in ("trainer", "inspector"):
        files.append((f"bin/{name}", private_path(base / config[name])))
    for key, target in (("initial_weights", "initial_weights"), ("opponent_weights", "opponent")):
        if key not in config:
            continue
        source = private_path(base / config[key])
        if source.is_dir():
            source = private_path(source / RUNTIME_FILE)
        if not source.is_file():
            raise ValueError(f"{key} must select a regular runtime weights file")
        files.append((f"inputs/{target}/{RUNTIME_FILE}", source))
    return sorted(files)


def freeze_inputs(directory, inputs):
    records = {}
    total = 0
    for relative, source in inputs:
        data = read_bytes(source, MAX_FILE)
        total += len(data)
        if total > MAX_TOTAL:
            raise ValueError("frozen inputs exceed total size limit")
        target = directory / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_exclusive(target, data, 0o500 if relative in
                        {"bin/trainer", "bin/inspector"} else 0o400)
        records[relative] = {"size": len(data), "sha256": digest(data)}
    directories = {directory}
    for relative in records:
        parent = (directory / relative).parent
        while parent != directory:
            directories.add(parent)
            parent = parent.parent
    for parent in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(parent)
    return records


def create(config_path, campaign_path):
    config_path = private_path(config_path)
    directory = private_path(campaign_path)
    config = validate_config(read_json(config_path))
    inputs = collect_inputs(config, config_path.parent)
    if directory.exists():
        raise ValueError("campaign already exists; never overwrite or adopt implicitly")
    if not directory.parent.is_dir():
        raise ValueError("campaign parent must already exist")
    # Exclusive creation claims the path. Interrupted preparation is never reused.
    directory.mkdir(mode=0o700)
    fsync_directory(directory.parent)
    for name in ("bin", "inputs", "invocations"):
        (directory / name).mkdir(mode=0o700)
    write_exclusive(directory / "owner.lock", b"", 0o600)
    records = freeze_inputs(directory, inputs)
    frozen_config = dict(config, trainer="bin/trainer", inspector="bin/inspector")
    if "initial_weights" in frozen_config:
        frozen_config["initial_weights"] = "inputs/initial_weights"
    if "opponent_weights" in frozen_config:
        frozen_config["opponent_weights"] = "inputs/opponent"
    manifest = {"schema": SCHEMA, "campaign_id": uuid.uuid4().hex,
                "config": frozen_config, "files": records}
    payload = encode(manifest)
    write_exclusive(directory / "manifest.json", payload)
    atomic_status(directory, {"schema": SCHEMA, "campaign_id": manifest["campaign_id"],
                             "manifest_sha256": digest(payload), "phase": "prepared",
                             "accepted_updates": 0, "accepted_invocation": 0,
                             "receipt_sha256": None, "pending_invocation": None,
                             "last_error": None})
    return status(directory)


def validate_manifest(manifest):
    if set(manifest) != {"schema", "campaign_id", "config", "files"}:
        raise ValueError("unsupported manifest fields")
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported manifest schema")
    identity = manifest["campaign_id"]
    if not isinstance(identity, str) or len(identity) != 32:
        raise ValueError("invalid campaign identity")
    config = validate_config(manifest["config"])
    files = manifest["files"]
    if not isinstance(files, dict) or not 3 <= len(files) <= MAX_FILES:
        raise ValueError("invalid frozen file inventory")
    required = {"frozen/controller.py", "frozen/train_state.py", "bin/trainer", "bin/inspector"}
    if not required <= files.keys():
        raise ValueError("missing frozen artifact")
    for relative in files:
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or str(path) != relative or
                (relative not in required and relative != "frozen/train_runner.py" and
                 relative not in {f"inputs/initial_weights/{RUNTIME_FILE}", f"inputs/opponent/{RUNTIME_FILE}"})):
            raise ValueError("unsafe frozen artifact path")
    if config["trainer"] != "bin/trainer" or config["inspector"] != "bin/inspector":
        raise ValueError("manifest references an unfrozen executable")
    if "initial_weights" in config and config["initial_weights"] != "inputs/initial_weights":
        raise ValueError("manifest references unfrozen input weights")
    if "opponent_weights" in config and config["opponent_weights"] != "inputs/opponent":
        raise ValueError("manifest references unfrozen opponent weights")


def status(campaign_path):
    directory = private_path(campaign_path)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("campaign must be owned by this user with mode 0700")
    current = read_json(directory / "status.json")
    required = {"schema", "campaign_id", "manifest_sha256", "phase", "accepted_updates",
                "accepted_invocation", "receipt_sha256", "pending_invocation", "last_error"}
    if (set(current) != required or current["schema"] != SCHEMA or
            current["phase"] not in {"prepared", "running", "paused", "failed", "completed"}):
        raise ValueError("unsupported or corrupt status")
    payload = read_bytes(directory / "manifest.json", MAX_JSON)
    if digest(payload) != current["manifest_sha256"]:
        raise ValueError("manifest integrity mismatch")
    manifest = decode_json(payload)
    validate_manifest(manifest)
    if current["campaign_id"] != manifest["campaign_id"]:
        raise ValueError("campaign identity mismatch")
    verify_frozen(directory, manifest)
    validate_progress(directory, current, manifest)
    return current


def verify_frozen(directory, manifest):
    total = 0
    for relative, record in manifest["files"].items():
        data = read_bytes(directory / relative, MAX_FILE)
        total += len(data)
        if (total > MAX_TOTAL or record != {"size": len(data), "sha256": digest(data)} or
                (directory / relative).stat().st_mode & 0o222):
            raise ValueError(f"frozen artifact integrity mismatch: {relative}")


def invocation_budget(accepted, total, maximum, history):
    bounded_integer(total, "total_updates", 1, 10000)
    bounded_integer(accepted, "accepted updates", 0, total)
    bounded_integer(maximum, "invocation_updates", 1, 16)
    bounded_integer(history, "history_every", 1, 10000)
    return min(maximum, total - accepted, history - accepted % history)


def job_path(directory, number):
    bounded_integer(number, "invocation number", 1, 10000)
    return private_path(directory / "invocations" / f"{number:06d}")


def validate_progress(directory, current, manifest):
    config = manifest["config"]
    count = bounded_integer(current["accepted_invocation"], "accepted invocation", 0, 10000)
    bounded_integer(current["accepted_updates"], "accepted updates", 0, config["total_updates"])
    previous_hash, updates = None, 0
    for number in range(1, count + 1):
        path = job_path(directory, number) / "accepted.json"
        payload = read_bytes(path, MAX_JSON)
        receipt = decode_json(payload)
        expected = updates + invocation_budget(updates, config["total_updates"],
                                               config["invocation_updates"], config["history_every"])
        if (receipt.get("schema") != "drysua-training-invocation/v1" or
                receipt.get("campaign_id") != current["campaign_id"] or
                receipt.get("invocation_id") != f"{number:06d}" or
                receipt.get("previous_receipt_sha256") != previous_hash or
                receipt.get("start_updates") != updates or receipt.get("expected_updates") != expected or
                receipt.get("inspection", {}).get("progress", {}).get("updates") != expected):
            raise ValueError("invocation receipt chain mismatch")
        if (receipt["spec_sha256"] != digest(read_bytes(path.parent / "job.json", MAX_JSON)) or
                receipt["result_sha256"] != digest(encode(validate_result(path.parent / "result.json")))):
            raise ValueError("invocation receipt evidence mismatch")
        previous_hash, updates = digest(payload), expected
    if updates != current["accepted_updates"] or previous_hash != current["receipt_sha256"]:
        raise ValueError("unsupported or corrupt status: receipt progress mismatch")
    pending = current["pending_invocation"]
    if pending is not None and (type(pending) is not int or pending != count + 1 or pending > 10000):
        raise ValueError("unsupported pending invocation")
    if current["phase"] == "completed" and updates != config["total_updates"]:
        raise ValueError("unsupported or corrupt status: incomplete completion")
    if ((current["phase"] in {"prepared", "paused", "completed"} and pending is not None) or
            (current["phase"] == "prepared" and count != 0)):
        raise ValueError("unsupported or corrupt status phase")
    validate_job_inventory(directory, count, pending)
    if count:
        verify_inventory(job_path(directory, count) / "checkpoint", receipt["checkpoint_files"], exact=True)


def validate_job_inventory(directory, count, pending):
    seen = set()
    with os.scandir(private_path(directory / "invocations")) as jobs:
        for index, entry in enumerate(jobs):
            if (index >= 10000 or entry.is_symlink() or not entry.is_dir(follow_symlinks=False) or
                    not re.fullmatch(r"[0-9]{6}", entry.name)):
                raise ValueError("invalid invocation directory inventory")
            number = int(entry.name)
            if number < 1 or (number > count and number != pending):
                raise ValueError("unrecorded invocation; explicit offline investigation required")
            seen.add(number)
    if seen != set(range(1, count + 1)) | ({pending} if pending else set()):
        raise ValueError("missing invocation directory")


def relative_file(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096:
        raise ValueError("invalid checkpoint file path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError("checkpoint path escape")
    return path


def verify_inventory(directory, inventory, exact=False):
    if not isinstance(inventory, dict) or not 1 <= len(inventory) <= MAX_CHECKPOINT_FILES:
        raise ValueError("invalid checkpoint file inventory")
    total = 0
    for relative, record in inventory.items():
        if (not isinstance(record, dict) or set(record) != {"size", "sha256"} or
                type(record["size"]) is not int or not 0 <= record["size"] <= MAX_FILE or
                not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError("invalid checkpoint file inventory record")
        data = read_bytes(directory / relative_file(relative), MAX_FILE)
        total += len(data)
        if total > MAX_TOTAL or record != {"size": len(data), "sha256": digest(data)}:
            raise ValueError(f"checkpoint file integrity mismatch: {relative}")
    if exact and checkpoint_inventory(directory) != inventory:
        raise ValueError("checkpoint file inventory mismatch")


def checkpoint_inventory(directory):
    pending, inventory, entries_count, total = [directory], {}, 0, 0
    while pending:
        with os.scandir(private_path(pending.pop())) as entries:
            for entry in entries:
                entries_count += 1
                if entries_count > MAX_CHECKPOINT_FILES + 128 or entry.is_symlink():
                    raise ValueError("checkpoint contains symlinks or too many entries")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                data = read_bytes(path, MAX_FILE)
                total += len(data)
                if total > MAX_TOTAL:
                    raise ValueError("checkpoint exceeds total size limit")
                inventory[str(path.relative_to(directory))] = {"size": len(data), "sha256": digest(data)}
                if len(inventory) > MAX_CHECKPOINT_FILES:
                    raise ValueError("checkpoint file count exceeds native limit")
    if not inventory:
        raise ValueError("empty checkpoint")
    return inventory


def inspect_checkpoint(directory, checkpoint, expected, previous=None, deadline=None):
    payload = state.capture([str(directory / "bin/inspector"), "checkpoint-inspect",
                             "--checkpoint-directory", str(checkpoint)],
                            timeout=min(10, session_remaining(deadline)), limit=MAX_JSON)
    value = decode_json(payload)
    required = {"schema", "identity", "model", "progress", "run", "ppo", "adaptive", "files",
                "runtime_matches_model", "kind", "checkpoint", "history", "runtime_status", "recovery_required", "sources"}
    if not required <= value.keys() or value["schema"] != INSPECTION_SCHEMA:
        raise ValueError("unsupported native inspector schema or fields")
    validate_native_state(value)
    if value["runtime_matches_model"] is not True or value["runtime_status"] != "matched":
        raise ValueError("native runtime does not match model")
    validate_identity(value["identity"])
    if value["adaptive"] is not None and not isinstance(value["adaptive"], dict):
        raise ValueError("invalid native adaptive state")
    if previous is not None and value["identity"]["scope_sha256"] != previous["identity"]["scope_sha256"]:
        raise ValueError("native checkpoint scope changed")
    for name in ("model", "run", "ppo"):
        if not isinstance(value[name], dict) or not value[name]:
            raise ValueError(f"invalid native {name}")
        if previous is not None and value[name] != previous[name]:
            raise ValueError(f"native checkpoint {name} changed")
    progress = value["progress"]
    counters = {"updates", "optimizer_steps", "rollout_samples", "games"}
    if not isinstance(progress, dict) or not counters <= progress.keys():
        raise ValueError("invalid native progress")
    for name in counters:
        counter = progress[name]
        bounded_integer(counter, f"native {name}", 0, 10**15)
        if previous is not None and counter < previous["progress"][name]:
            raise ValueError(f"native {name} rolled back")
    if progress["updates"] != expected:
        raise ValueError(f"native update advancement mismatch: expected {expected}, got {progress['updates']}")
    if not isinstance(value["files"], list) or not 1 <= len(value["files"]) <= MAX_CHECKPOINT_FILES:
        raise ValueError("invalid native files")
    inventory = {}
    for record in value["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("invalid native file record")
        name = str(relative_file(record["path"]))
        if name in inventory:
            raise ValueError("duplicate native file record")
        inventory[name] = {key: record[key] for key in ("size", "sha256")}
    verify_inventory(checkpoint, inventory)
    for field, role in ARTIFACT_IDENTITIES.items():
        name = str(relative_file(value["sources"][role]))
        if name not in inventory or value["identity"][field] != inventory[name]["sha256"]:
            raise ValueError(f"native artifact identity mismatch: {field}")
    session_remaining(deadline)
    return value


def validate_native_state(value):
    if value["kind"] != "train-annealed":
        raise ValueError("unsupported native inspection kind")
    if value["recovery_required"] is not False:
        raise ValueError("native checkpoint requires explicit recovery; automatic fallback is forbidden")
    history = value["history"]
    if (not isinstance(history, dict) or history.get("verified") is not True or
            history.get("kind") not in {"fixed", "adaptive"}):
        raise ValueError("native annealed history is not verified")
    bounded_integer(history.get("snapshot_count"), "native history snapshot count", 0, 10000)
    sources = value["sources"]
    if (not isinstance(sources, dict) or not {"manifest", "tensor", "runtime", "canonical_tensor_matches"} <= sources.keys()
            or type(sources["canonical_tensor_matches"]) is not bool):
        raise ValueError("invalid native selected sources")
    for role in ("manifest", "tensor", "runtime"):
        relative_file(sources[role])


def validate_contract(value):
    if value.get("schema") != INSPECTION_SCHEMA or value.get("kind") != "contract":
        raise ValueError("unsupported native inspector contract")
    capabilities = value.get("capabilities", {})
    if (not isinstance(capabilities, dict) or any(capabilities.get(key) is not True for key in
            ("inspection", "annealed_history", "read_only", "strict_build_features")) or
            capabilities.get("controller_run_kind") != "train-annealed"):
        raise ValueError("native inspector contract lacks required read-only annealed capabilities")
    limits = value.get("limits", {})
    for name, expected in (("max_json_bytes", MAX_JSON), ("max_snapshots", 10000), ("max_files", MAX_CHECKPOINT_FILES)):
        if type(limits.get(name)) is not int or limits[name] != expected:
            raise ValueError(f"unsupported native inspector contract limit: {name}")
    model = value.get("model", {})
    if (not isinstance(model, dict) or not isinstance(model.get("hash"), str) or
            not re.fullmatch(r"[0-9a-f]{16}", model["hash"])):
        raise ValueError("invalid native inspector model contract")
    bounded_integer(model.get("version"), "native model version", 1, 2**32 - 1)
    bounded_integer(model.get("parameters"), "native model parameters", 1, 2**32 - 1)


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != ARTIFACT_IDENTITIES.keys() | {"scope_sha256"}:
        raise ValueError("invalid native identity")
    for value in identity.values():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid native identity hash")


def session_remaining(deadline):
    if deadline is None:
        return math.inf
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ValueError("session deadline exceeded; explicit recovery may be required")
    return remaining


def campaign_disk_usage(directory, entry_limit=None, deadline=None):
    """Charge apparent or allocated bytes, whichever is larger; never follow links."""
    directory = private_path(directory)
    limit = MAX_CAMPAIGN_ENTRIES if entry_limit is None else entry_limit
    metadata = directory.stat()
    usage = max(4096, metadata.st_size, metadata.st_blocks * 512)
    if usage > MAX_CAMPAIGN_BYTES:
        raise ValueError("campaign disk limit exceeds 100 GiB")
    pending, count = [directory], 0
    while pending:
        session_remaining(deadline)
        with os.scandir(private_path(pending.pop())) as entries:
            for entry in entries:
                count += 1
                if count > limit:
                    raise ValueError("campaign disk inventory entry limit exceeded")
                if count % 256 == 0:
                    session_remaining(deadline)
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError("campaign disk inventory contains a symlink")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(Path(entry.path))
                elif not stat.S_ISREG(metadata.st_mode) and not (
                        stat.S_ISSOCK(metadata.st_mode) and entry.name == "runner-locks.sock"):
                    raise ValueError("campaign disk inventory contains an unsupported file type")
                usage += max(4096, metadata.st_size, metadata.st_blocks * 512)
                if usage > MAX_CAMPAIGN_BYTES:
                    raise ValueError("campaign disk limit exceeds 100 GiB")
    return usage


def validate_result(path, expected_timeouts=None):
    value = read_json(path)
    container = value.get("container_state")
    if value.get("schema") != "drysua-training-runner/v1":
        raise ValueError("unsupported runner result schema")
    if (type(value.get("returncode")) is not int or value["returncode"] != 0 or
            not isinstance(container, dict) or container.get("Running") is not False or
            container.get("OOMKilled") is not False or type(container.get("ExitCode")) is not int or
            container["ExitCode"] != 0 or not isinstance(value.get("container_id"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", value["container_id"])):
        raise ValueError("runner result is unsuccessful or container stop is unconfirmed")
    if value.get("cleanup_confirmed", True) is not True:
        raise ValueError("runner cleanup is unconfirmed")
    if "timeouts" in value:
        recorded = value["timeouts"]
        if not isinstance(recorded, dict) or any(type(number) is not int for number in recorded.values()):
            raise ValueError("invalid runner timeout evidence")
        resolved = resolve_timeouts(recorded.get("payload_seconds"))
        if recorded != asdict(resolved) or (expected_timeouts is not None and recorded != asdict(expected_timeouts)):
            raise ValueError("runner timeout evidence differs from frozen job")
    return value


def previous_receipt(directory, current):
    if not current["accepted_invocation"]:
        return None
    return read_json(job_path(directory, current["accepted_invocation"]) / "accepted.json")


def preflight(directory, current, manifest, deadline=None):
    config = manifest["config"]
    if "frozen/train_runner.py" not in manifest["files"]:
        raise ValueError("no frozen runner; create a new campaign with the runner installed")
    if config["image"] is None:
        raise ValueError("an immutable runner image digest is required")
    if not directory.is_relative_to(Path(config["workspace_root"])):
        raise ValueError("campaign must be inside workspace_root")
    for lock in config["lock_paths"]:
        if not private_path(lock).is_file():
            raise ValueError(f"required shared lock does not exist: {lock}")
    for relative, source in (("frozen/controller.py", Path(__file__)),
                             ("frozen/train_state.py", Path(state.__file__))):
        if digest(read_bytes(source, MAX_FILE)) != manifest["files"][relative]["sha256"]:
            raise ValueError("controller source changed; invoke the frozen controller")
    try:
        payload = state.capture([str(directory / "bin/inspector"), "checkpoint-inspect", "--contract"],
                                timeout=min(10, session_remaining(deadline)), limit=MAX_JSON)
    except (OSError, ValueError) as error:
        raise ValueError(f"native inspector --contract unsupported or unavailable: {error}") from error
    validate_contract(decode_json(payload))
    previous = previous_receipt(directory, current)
    if previous:
        checkpoint = job_path(directory, current["accepted_invocation"]) / "checkpoint"
        inspected = inspect_checkpoint(directory, checkpoint, current["accepted_updates"],
                                       previous["inspection"], deadline)
        if inspected != previous["inspection"]:
            raise ValueError("accepted native checkpoint changed")


def make_job(directory, current, manifest):
    spec, expected = job_spec(directory, current, manifest)
    number = current["accepted_invocation"] + 1
    job = job_path(directory, number)
    job.mkdir(mode=0o700)
    fsync_directory(job.parent)
    checkpoint = job / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    previous = previous_receipt(directory, current)
    if previous:
        source = job_path(directory, number - 1) / "checkpoint"
        verify_inventory(source, previous["checkpoint_files"], exact=True)
        for relative in previous["checkpoint_files"]:
            target = checkpoint / relative_file(relative)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_exclusive(target, read_bytes(source / relative, MAX_FILE), 0o600)
    write_exclusive(job / "job.json", encode(spec))
    current["pending_invocation"] = number
    atomic_status(directory, current)
    return job, expected


def job_spec(directory, current, manifest):
    config = manifest["config"]
    timeouts = resolve_timeouts(config.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    number = current["accepted_invocation"] + 1
    job = job_path(directory, number)
    checkpoint = job / "checkpoint"
    budget = invocation_budget(current["accepted_updates"], config["total_updates"],
                               config["invocation_updates"], config["history_every"])
    command = [str(directory / "bin/trainer"), "train-annealed", *config["training_args"],
               "--updates", str(config["total_updates"]), "--invocation-updates", str(budget),
               "--checkpoint-directory", str(checkpoint), "--device",
               "cpu" if config["mode"] == "cpu" else "cuda"]
    if current["accepted_invocation"]:
        command.append("--resume")
    elif "initial_weights" in config:
        command.extend(["--initial-weights", str(directory / config["initial_weights"])])
    if "opponent_weights" in config:
        command.extend(["--opponent", "weights", "--opponent-weights", str(directory / config["opponent_weights"])])
    if config["mode"] == "gpu":
        command.extend(["--device-ordinal", "0"])
    spec = {key: config[key] for key in
            ("workspace_root", "mode", "docker_context", "image", "gpu_uuid", "lock_paths")}
    spec.update(version=1, campaign_directory=str(directory), job_directory=str(job),
                command=command, campaign_id=current["campaign_id"], invocation_id=f"{number:06d}")
    if "invocation_seconds" in config:
        spec["invocation_seconds"] = timeouts.payload_seconds
    return spec, current["accepted_updates"] + budget


def accept_job(directory, current, job, expected, deadline=None):
    if campaign_disk_usage(directory, deadline=deadline) + PERSISTENCE_RESERVE_BYTES > MAX_CAMPAIGN_BYTES:
        raise ValueError("campaign disk limit: insufficient receipt reserve")
    payload = read_bytes(directory / "manifest.json", MAX_JSON)
    if digest(payload) != current["manifest_sha256"]:
        raise ValueError("manifest integrity mismatch")
    manifest = decode_json(payload)
    verify_frozen(directory, manifest)
    spec, expected_from_spec = job_spec(directory, current, manifest)
    if read_json(job / "job.json") != spec or expected != expected_from_spec:
        raise ValueError("invocation spec differs from frozen campaign intent")
    result = validate_result(job / "result.json", resolve_timeouts(
        manifest["config"].get("invocation_seconds", DEFAULT_INVOCATION_SECONDS)))
    previous = previous_receipt(directory, current)
    inspection = inspect_checkpoint(directory, job / "checkpoint", expected,
                                    previous["inspection"] if previous else None, deadline)
    receipt = {"schema": "drysua-training-invocation/v1", "campaign_id": current["campaign_id"],
               "invocation_id": job.name, "start_updates": current["accepted_updates"],
               "expected_updates": expected, "previous_receipt_sha256": current["receipt_sha256"],
               "inspection": inspection, "checkpoint_files": checkpoint_inventory(job / "checkpoint"),
               "result_sha256": digest(encode(result)),
               "spec_sha256": digest(read_bytes(job / "job.json", MAX_JSON))}
    payload = encode(receipt)
    session_remaining(deadline)
    if len(payload) > MAX_JSON:
        raise ValueError("invocation receipt exceeds size limit")
    path = job / "accepted.json"
    if path.exists():
        if read_bytes(path, MAX_JSON) != payload:
            raise ValueError("existing immutable invocation receipt mismatch")
    else:
        for relative in receipt["checkpoint_files"]:
            artifact = private_path(job / "checkpoint" / relative)
            artifact.chmod(0o400)
            descriptor = os.open(artifact, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(job / "checkpoint")
        session_remaining(deadline)
        write_exclusive(path, payload)
    current.update(accepted_updates=expected, accepted_invocation=int(job.name),
                   receipt_sha256=digest(payload), pending_invocation=None, last_error=None)
    atomic_status(directory, current)


def owner_record(directory):
    path = private_path(directory / "owner.json")
    if not path.exists():
        return None
    value = read_json(path)
    if set(value) != {"pid", "start", "uid", "boot", "token", "child"} or not re.fullmatch(
            r"[0-9a-f]{32}", value.get("token", "")):
        raise ValueError("invalid owner record")
    state.is_live(value)
    if value["child"] is not None:
        state.is_live(value["child"])
    return value


def request_control(directory, operation):
    if operation not in {"pause", "stop"}:
        raise ValueError("unknown control operation")
    directory = private_path(directory)
    status(directory)
    owner = owner_record(directory)
    if (owner is None or not state.is_live(owner) or
            not state.is_locked(private_path(directory / "owner.lock"))):
        raise ValueError("no live owner; explicit recover is required")
    path = private_path(directory / f"{operation}-{owner['token']}.json")
    if not path.exists():
        write_exclusive(path, encode({"token": owner["token"]}))
    elif read_json(path) != {"token": owner["token"]}:
        raise ValueError("invalid existing control request")
    return {"phase": "requested", "operation": operation, "owner_token": owner["token"]}


def requested(directory, owner, operation):
    path = private_path(directory / f"{operation}-{owner['token']}.json")
    if not path.exists():
        return False
    if read_json(path) != {"token": owner["token"]}:
        raise ValueError("invalid control request")
    return True


def release_owner(directory, owner):
    for operation in ("pause", "stop"):
        if requested(directory, owner, operation):
            (directory / f"{operation}-{owner['token']}.json").unlink()
    (directory / "owner.json").unlink()
    fsync_directory(directory)


def execute_loop(directory, current, manifest, owner, signalled, deadline):
    total = manifest["config"]["total_updates"]
    timeouts = resolve_timeouts(manifest["config"].get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    for _ in range(10000):
        if current["accepted_updates"] == total:
            current["phase"] = "completed"
            break
        if signalled.is_set() or requested(directory, owner, "stop") or requested(directory, owner, "pause"):
            current["phase"] = "paused"
            break
        if deadline - monotonic() < timeouts.controller_reserve_seconds:
            current.update(phase="paused", last_error="session deadline reserve insufficient for another invocation")
            break
        verify_frozen(directory, manifest)
        disk_used = campaign_disk_usage(directory, deadline=deadline)
        if disk_used + JOB_DISK_RESERVE_BYTES > MAX_CAMPAIGN_BYTES:
            current.update(phase="paused", last_error="campaign disk reserve insufficient for another invocation")
            break
        if deadline - monotonic() < timeouts.controller_reserve_seconds:
            current.update(phase="paused", last_error="session deadline reserve consumed by admission checks")
            break
        job, expected = make_job(directory, current, manifest)
        if deadline - monotonic() < timeouts.controller_reserve_seconds:
            raise ValueError("session deadline reserve exhausted during job preparation; no runner launched")

        def record_child(identity):
            owner["child"] = identity
            atomic_json(directory / "owner.json", owner)

        def stop_requested():
            if signalled.is_set() or requested(directory, owner, "stop"):
                return True
            if session_remaining(deadline) <= 10:
                raise ValueError("session deadline shutdown reserve exhausted")
            job_used = campaign_disk_usage(job, entry_limit=MAX_JOB_ENTRIES, deadline=deadline)
            if disk_used + job_used + PERSISTENCE_RESERVE_BYTES > MAX_CAMPAIGN_BYTES:
                raise ValueError("campaign disk limit reached by active invocation")
            return False

        state.wait_runner([sys.executable, str(directory / "frozen/train_runner.py"),
                           "run", "--spec", str(job / "job.json")],
                          timeouts.runner_seconds, stop_requested, record_child,
                          timeouts.controller_reserve_seconds)
        accept_job(directory, current, job, expected, deadline)
        if current["accepted_updates"] == total:
            current["phase"] = "completed"
            break
    else:
        raise ValueError("invocation count bound exceeded")
    atomic_status(directory, current)
    return current


def run_campaign(campaign_path, resume=False, ready_fd=None):
    started = monotonic()
    if ready_fd is not None and (ready_fd < 3 or not stat.S_ISFIFO(os.fstat(ready_fd).st_mode)):
        raise ValueError("handshake descriptor must be an inherited pipe")
    directory = private_path(campaign_path)
    with state.locks([private_path(directory / "owner.lock")]):
        current = status(directory)
        if owner_record(directory) is not None:
            raise ValueError("existing owner; explicit recover is required")
        if current["phase"] != ("paused" if resume else "prepared"):
            raise ValueError("run requires prepared state; paused requires resume, failed requires recover")
        manifest = read_json(directory / "manifest.json")
        deadline = started + manifest["config"]["max_seconds"]
        preflight(directory, current, manifest, deadline)
        owner = dict(state.process_identity(os.getpid()), token=uuid.uuid4().hex, child=None)
        write_exclusive(directory / "owner.json", encode(owner), 0o600)
        current.update(phase="running", last_error=None)
        atomic_status(directory, current)
        try:
            with state.stop_signals() as signalled:
                if ready_fd is not None:
                    os.write(ready_fd, b"ready\n")
                    os.close(ready_fd)
                return execute_loop(directory, current, manifest, owner, signalled, deadline)
        except (OSError, ValueError, TypeError, KeyError) as error:
            current.update(phase="failed", last_error=str(error)[:1024])
            atomic_status(directory, current)
            raise ValueError(str(error)) from error
        finally:
            # Failed invocations retain their owner record for explicit recovery.
            if current["phase"] in {"paused", "completed"}:
                release_owner(directory, owner)


def recover(campaign_path, confirm=False):
    if not confirm:
        raise ValueError("recover requires explicit --confirm-offline")
    directory = private_path(campaign_path)
    with state.locks([private_path(directory / "owner.lock")]):
        current = status(directory)
        manifest = read_json(directory / "manifest.json")
        owner = owner_record(directory)
        if owner and (state.is_live(owner) or (owner["child"] and state.is_live(owner["child"]))):
            raise ValueError("recover refuses a live owner or owned child")
        paths = [private_path(path) for path in manifest["config"]["lock_paths"]]
        with state.locks(paths):
            pending = current["pending_invocation"]
            if pending is not None:
                if owner is None or owner["child"] is None:
                    raise ValueError("pending invocation has no recorded child identity; offline investigation required")
                preflight(directory, current, manifest)
                config = manifest["config"]
                expected = current["accepted_updates"] + invocation_budget(
                    current["accepted_updates"], config["total_updates"],
                    config["invocation_updates"], config["history_every"])
                accept_job(directory, current, job_path(directory, pending), expected)
            current.update(phase="completed" if current["accepted_updates"] ==
                           manifest["config"]["total_updates"] else "paused", last_error=None)
            atomic_status(directory, current)
            if owner:
                release_owner(directory, owner)
            return current


def detach(campaign_path, resume=False):
    directory = private_path(campaign_path)
    status(directory)
    reader, writer = os.pipe()
    child = None
    try:
        command = [sys.executable, str(directory / "frozen/controller.py"),
                   "resume" if resume else "run", str(directory), "--ready-fd", str(writer)]
        child = subprocess.Popen(command, pass_fds=(writer,), start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, env=state.environment())
        os.close(writer)
        writer = None
        ready, _, _ = select.select([reader], [], [], 30)
        if not ready or os.read(reader, 128) != b"ready\n":
            state.terminate_child(child)
            raise ValueError("detached controller failed before durable ownership handshake; inspect status")
        state.reap_detached(child)
        return {"phase": "started", "pid": child.pid}
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("create", "fresh"):
        command = commands.add_parser(name, help="freeze a new, never-run campaign")
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("campaign", type=Path)
    for name in ("status", "run", "pause", "resume", "recover", "adopt", "stop"):
        command = commands.add_parser(name)
        command.add_argument("campaign", type=Path)
        if name in {"run", "resume"}:
            command.add_argument("--detach", action="store_true")
            command.add_argument("--ready-fd", type=int, help=argparse.SUPPRESS)
        if name == "recover":
            command.add_argument("--confirm-offline", action="store_true")
        if name == "stop":
            command.add_argument("--immediate", required=True, action="store_true")
    options = parser.parse_args(arguments)
    if options.operation in {"create", "fresh"}:
        result = create(options.config, options.campaign)
    elif options.operation == "status":
        result = status(options.campaign)
    elif options.operation in {"run", "resume"}:
        if options.detach and options.ready_fd is not None:
            raise ValueError("detach cannot override handshake descriptor")
        if options.detach:
            result = detach(options.campaign, resume=options.operation == "resume")
        else:
            result = run_campaign(options.campaign, resume=options.operation == "resume",
                                  ready_fd=options.ready_fd)
    elif options.operation in {"pause", "stop"}:
        result = request_control(options.campaign, options.operation)
    elif options.operation == "recover":
        result = recover(options.campaign, confirm=options.confirm_offline)
    else:
        raise ValueError("adopt is not supported; no process or state was changed")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"training controller: {error}", file=sys.stderr)
        sys.exit(2)
