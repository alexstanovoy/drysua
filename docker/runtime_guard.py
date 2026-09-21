"""Docker-only, single-lineage admission and one-update supervision."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

GIB = 1024 ** 3
DISK_FLOOR = 100 * GIB
MAX_INVOCATIONS = 1000
PAYLOAD_LIMIT = 16 * 1024 ** 2
WRAPPER_LIMIT = 20 * 1024 ** 2
CGROUP = Path("/sys/fs/cgroup")
DATA = Path("/run-data")
BINARY = "/usr/local/bin/drysua"
STOP_REQUESTED = False
WRAPPER_BYTES = 0
LAST_SAMPLE = 0.0


class Refusal(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def read_small(path, limit=65536):
    with open(path, "rb") as source:
        data = source.read(limit + 1)
    require(len(data) <= limit, f"oversized input: {path}")
    return data


def record(event, **fields):
    global WRAPPER_BYTES
    line = json.dumps({"event": event, **fields}, sort_keys=True) + "\n"
    WRAPPER_BYTES += len(line.encode())
    require(WRAPPER_BYTES <= WRAPPER_LIMIT, "wrapper log limit exceeded")
    print(line, end="", flush=True)


def bounded_integer(name, default, minimum, maximum):
    value = os.environ.get(name, str(default))
    require(re.fullmatch(r"[0-9]{1,10}", value), f"invalid {name}")
    value = int(value)
    require(minimum <= value <= maximum, f"{name} outside {minimum}..{maximum}")
    return value


def run_config():
    config = {
        "updates": bounded_integer("TRAIN_UPDATES", 1000, 1, MAX_INVOCATIONS),
        "games": bounded_integer("TRAIN_GAMES", 40, 28, 40),
        "parallel": bounded_integer("TRAIN_PARALLEL", 40, 1, 40),
        "generation-games": bounded_integer("TRAIN_GENERATION_GAMES", 200, 2, 100000),
        "zero-updates": bounded_integer("TRAIN_ZERO_UPDATES", 200, 0, 1000),
        "seed": bounded_integer("TRAIN_SEED", 9001, 0, 4294967295),
        "seconds": bounded_integer("INVOCATION_SECONDS", 300, 10, 300),
    }
    require(config["games"] % 2 == 0, "games must be even")
    require(config["games"] % config["parallel"] == 0, "parallel must divide games")
    require(config["generation-games"] % config["parallel"] == 0,
            "parallel must divide generation games")
    require(config["generation-games"] % config["games"] == 0,
            "games must divide generation games")
    require(config["zero-updates"] <= config["updates"], "zero updates exceed target")
    return config


def native_command(config, resume):
    command = [BINARY, "train-annealed"]
    for name in ("updates", "games", "parallel", "generation-games", "zero-updates", "seed"):
        command.extend([f"--{name}", str(config[name])])
    command.extend(["--opponent", "teacher", "--epochs", "4", "--minibatch", "2048",
                    "--device", "cuda", "--device-ordinal", "0",
                    "--invocation-updates", "1", "--checkpoint-seconds", "300",
                    "--checkpoint-directory", str(DATA / "checkpoint"),
                    "--metrics-directory", str(DATA / "metrics")])
    command.extend(["--resume"] if resume else ["--initial-weights", "/input/weights"])
    return command


@contextlib.contextmanager
def shared_locks(paths):
    descriptors = []
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            linked = os.stat(path, follow_symlinks=False)
            require(stat.S_ISREG(opened.st_mode), f"not a regular lock: {path}")
            require((opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino),
                    f"lock inode changed: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Refusal(f"heavy lock busy: {path}; existing owner untouched") from error
        yield tuple(descriptors)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def validate_limits(limits):
    for name, expected in [("memory.max", str(12 * GIB)), ("memory.swap.max", "0"),
                           ("pids.max", "1024")]:
        require(limits.get(name) == expected, f"invalid effective {name}: expected {expected}")
    require(re.fullmatch(r"max [1-9][0-9]*", limits.get("cpu.max", "")),
            "invalid effective cpu.max: CPU quota forbidden")


def cgroup_events():
    require(read_small("/proc/self/cgroup", 4096).strip() == b"0::/",
            "private cgroup-v2 namespace required")
    limits = {name: read_small(CGROUP / name, 128).decode().strip()
              for name in ("memory.max", "memory.swap.max", "pids.max", "cpu.max")}
    validate_limits(limits)
    events = {}
    for subsystem in ("memory", "pids"):
        text = read_small(CGROUP / f"{subsystem}.events", 4096).decode()
        for line in text.splitlines():
            key, value = line.split()
            require(value.isdecimal(), "invalid cgroup event counter")
            events[f"{subsystem}.{key}"] = int(value)
    require({"memory.high", "memory.max", "memory.oom", "memory.oom_kill", "pids.max"}
            <= events.keys(), "missing cgroup event keys")
    return events


def validate_events(previous, current):
    require(previous.keys() == current.keys(), "cgroup event keys changed")
    for name, value in current.items():
        require(value == previous[name], f"cgroup event changed: {name}")


def validate_sample(memory, free, cpu, gpu, admission):
    require(memory >= (24 if admission else 16) * GIB, "host MemAvailable below threshold")
    require(free is not None and gpu is not None, "GPU telemetry unavailable")
    require(free >= 4096, "GPU VRAM free below 4 GiB")
    require(cpu is None or cpu <= 90, "CPU temperature above 90 C")
    require(gpu <= 85, "GPU temperature above 85 C")


def cpu_temperature():
    directory = Path("/host/cpu-hwmon")
    if read_small(directory / "name", 128).strip() not in (b"k10temp", b"coretemp"):
        return None
    temperatures = []
    for index in range(1, 65):
        path = directory / f"temp{index}_input"
        if path.exists():
            value = int(read_small(path, 128)) / 1000
            require(-20 <= value <= 150, "CPU sensor value invalid")
            temperatures.append(value)
    return max(temperatures, default=None)


def telemetry_output_limit():
    resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))


def stop_child(child, grace=4):
    # Only a session created by this controller, inside its private PID namespace.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=0.5)


def gpu_sample(uuid):
    with tempfile.TemporaryFile(dir="/tmp") as output:
        child = subprocess.Popen(
            ["nvidia-smi", "--id=" + uuid,
             "--query-gpu=uuid,memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=output, stderr=output, start_new_session=True,
            preexec_fn=telemetry_output_limit)
        try:
            child.wait(timeout=0.75)
            require(child.returncode == 0, "GPU telemetry command failed")
            output.seek(0)
            data = output.read(4097)
            require(len(data) <= 4096, "GPU telemetry output exceeded limit")
            fields = data.decode().strip().split(",")
            require(len(fields) == 3 and fields[0].strip() == uuid,
                    "GPU telemetry identity mismatch")
            free, temperature = int(fields[1]), int(fields[2])
            require(0 <= free <= 1024 * 1024 and 0 <= temperature <= 150,
                    "GPU telemetry values invalid")
            return free, temperature
        finally:
            stop_child(child, grace=0)


def sample(uuid, baseline, admission=False):
    global LAST_SAMPLE
    time.sleep(max(0, LAST_SAMPLE + 1 - time.monotonic()))
    require(not STOP_REQUESTED, "shutdown requested before resource sample")
    LAST_SAMPLE = time.monotonic()
    validate_events(baseline, cgroup_events())
    lines = read_small("/host/meminfo").decode().splitlines()
    available = [line.split() for line in lines if line.startswith("MemAvailable:")]
    require(len(available) == 1 and available[0][2:] == ["kB"], "host MemAvailable missing")
    memory = int(available[0][1]) * 1024
    cpu = cpu_temperature()
    free, gpu = gpu_sample(uuid)
    validate_sample(memory, free, cpu, gpu, admission)
    Path("/tmp/guard-alive").write_text(str(time.monotonic()))
    return cpu


def checkpoint_update(data):
    require(80 <= len(data) <= 65536, "checkpoint metadata size invalid")
    linked = [(13, 0x9D251E7C6725BB43), (5, 0x93EA35FD652475A7),
              (22, 0x92723C71B52788B0), (24, 0xA799157E02A95F4E),
              (38, 0xEAC28E84438D6A75), (7, 0x64F4C97D0C52D062)]
    header = b"DRYCKP18" + b"".join(struct.pack("<IQ", *item) for item in linked)
    require(data[:80] == header, "unsupported checkpoint metadata header")
    offset = 80
    for _ in range(4):
        require(offset + 2 <= len(data), "truncated checkpoint string length")
        size = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        require(1 <= size <= 4096 and offset + size <= len(data), "invalid checkpoint string")
        text = data[offset:offset + size].decode("utf-8")
        require(all(ord(character) >= 32 for character in text), "checkpoint control character")
        offset += size
    require(offset + 26 + 24 <= len(data), "truncated checkpoint progress")
    require(data[offset + 25] == 0, "checkpoint mastery unsupported by annealed guard")
    update, policy, scheduler = struct.unpack_from("<QQQ", data, offset + 26)
    require(update == policy == scheduler and update <= 1000, "checkpoint counters inconsistent")
    return update


def private_directory(path):
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid(),
            f"directory must be owned by container UID: {path}")
    require(metadata.st_mode & 0o077 == 0, f"directory must be private (0700): {path}")


def validate_fresh_metrics(directory):
    with os.scandir(directory) as children:
        for index, child in enumerate(children):
            require(index == 0 and child.name == ".metrics.writer.lock",
                    "fresh metrics must contain only the precreated writer lock")
            metadata = child.stat(follow_symlinks=False)
            require(stat.S_ISREG(metadata.st_mode) and metadata.st_size == 0,
                    "fresh metrics writer lock must be an empty regular file")


def file_sha256(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        require(stat.S_ISREG(metadata.st_mode), "hash input must be a regular file")
        require(0 < metadata.st_size <= 256 * 1024 ** 2, "hash input exceeds 256 MiB or is empty")
        digest = hashlib.sha256()
        for _ in range(257):
            chunk = source.read(1024 ** 2)
            if not chunk:
                require(source.tell() == metadata.st_size, "hash input changed size")
                return digest.hexdigest()
            digest.update(chunk)
        raise Refusal("hash input grew beyond 256 MiB")


def verify_initial_weights(path, expected):
    require(file_sha256(path) == expected, "accepted initial weights hash differs")


def accept_run(config, uuid):
    private_directory(DATA)
    for name in ("checkpoint", "metrics", "logs"):
        private_directory(DATA / name)
    receipt = json.loads(read_small("/input/provenance/acceptance.json", 16384))
    image = json.loads(read_small("/opt/drysua/provenance.json", 16384))
    for key in ("source_sha256", "binary_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", image[key]) and receipt[key] == image[key],
                f"accepted {key} differs from source-built image; no provenance relabel")
    require(receipt["gpu_uuid"] == uuid, "accepted GPU identity mismatch")
    require(receipt["config"] == config, "accepted run configuration mismatch")
    require(receipt["mode"] in ("fresh", "resume"), "acceptance mode must be fresh or resume")
    if receipt["mode"] == "fresh":
        require(not any((DATA / "checkpoint").iterdir()), "fresh checkpoint must be empty")
        validate_fresh_metrics(DATA / "metrics")
        require(receipt["starting_update"] == 0, "fresh starting update must be zero")
        verify_initial_weights(Path("/input/weights/drysua.weights.safetensors"),
                               receipt["initial_weights_sha256"])
        return 0, False
    data = read_small(DATA / "checkpoint/checkpoint.meta")
    require(hashlib.sha256(data).hexdigest() == receipt["checkpoint_meta_sha256"],
            "accepted checkpoint metadata hash differs; review boundary before restart")
    update = checkpoint_update(data)
    require(update == receipt["starting_update"] < config["updates"],
            "accepted starting update mismatch or target already complete")
    return update, True


def append_payload(output, data, written):
    require(written + len(data) <= PAYLOAD_LIMIT, "payload log limit exceeded")
    require(output.write(data) == len(data), "short payload log write")
    return written + len(data)


def run_invocation(config, update, resume, locks, uuid, baseline):
    require(0 <= update < config["updates"] <= MAX_INVOCATIONS,
            "invocation log update outside 1..1000 target")
    logs = DATA / "logs"
    filesystem = os.statvfs(logs)
    available = filesystem.f_bavail * filesystem.f_frsize
    required = DISK_FLOOR + (config["updates"] - update) * PAYLOAD_LIMIT
    require(available >= required,
            f"disk available below 100 GiB plus remaining payload budget: {available} < {required}")
    # Fixed names and exclusive creation retain failure evidence across manual restarts.
    with open(logs / f"payload-{update + 1:04d}.log", "xb", buffering=0) as output:
        record("invocation_started", before_update=update, payload_log=output.name)
        return supervise(native_command(config, resume), locks, output, 0, config, uuid, baseline)


def supervise(command, locks, output, written, config, uuid, baseline):
    require(not STOP_REQUESTED, "shutdown requested before payload")
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True, pass_fds=locks)
    try:
        os.set_blocking(child.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            next_sample = started + 1
            # Eight seconds reserve includes sampling, the GPU probe and child cleanup.
            deadline = started + config["seconds"] - 8
            for _ in range(20000):
                now = time.monotonic()
                require(not STOP_REQUESTED, "shutdown requested")
                require(now < deadline, "heavy invocation deadline reached")
                if now >= next_sample:
                    sample(uuid, baseline)
                    next_sample = time.monotonic() + 1
                for key, _ in selector.select(timeout=min(0.1, max(0, deadline - now))):
                    data = os.read(key.fd, 65536)
                    if data:
                        written = append_payload(output, data, written)
                    else:
                        require(child.wait(timeout=0.5) == 0,
                                f"native invocation failed; inspect {output.name}; no retry")
                        return written
            raise Refusal("payload iteration bound exceeded")
    finally:
        stop_child(child)
        child.stdout.close()
        output.flush()


def request_stop(_signal, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def train():
    require(os.getuid() != 0, "non-root runtime UID required")
    require(os.getpid() != 1, "Docker init required to reap owned descendants")
    os.umask(0o077)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = run_config()
    uuid = os.environ.get("GPU_UUID", "")
    require(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid), "explicit GPU UUID required")
    with shared_locks(["/locks/workspace", "/locks/global"]) as locks:
        baseline = cgroup_events()
        require(all(value == 0 for value in baseline.values()), "nonzero initial cgroup events")
        cpu = sample(uuid, baseline, admission=True)
        record("admitted", cpu_sensor="unavailable" if cpu is None else "available",
               memory_max=12 * GIB, gpu_uuid=uuid)
        update, resume = accept_run(config, uuid)
        for _ in range(config["updates"] - update):
            require(not STOP_REQUESTED, "shutdown requested before invocation")
            sample(uuid, baseline, admission=True)
            written = run_invocation(config, update, resume, locks, uuid, baseline)
            sample(uuid, baseline)
            after = checkpoint_update(read_small(DATA / "checkpoint/checkpoint.meta"))
            require(after == update + 1, "native success without exactly one committed update")
            update, resume = after, True
            record("invocation_committed", update=update, payload_bytes=written)
        record("target_completed", update=update)


def main():
    try:
        if sys.argv[1:] == ["--healthcheck"]:
            age = time.monotonic() - float(read_small("/tmp/guard-alive", 64))
            require(0 <= age <= 10, "guard heartbeat stale")
        else:
            require(not sys.argv[1:], "guard takes bounded environment, not arbitrary commands")
            train()
        return 0
    except (OSError, ValueError, KeyError, Refusal, subprocess.SubprocessError) as error:
        print(json.dumps({"event": "refused", "reason": str(error)[:2048]}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
