"""Rootless, resource-bounded execution of one controller-owned native invocation."""

import argparse
import array
import contextlib
from dataclasses import asdict, dataclass
import fcntl
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time

GIB = 1024 ** 3
PAYLOAD_LIMIT = 16 * 1024 ** 2
CONTROL_LIMIT = 65536
HISTORY_LIMIT = 512 * 1024
DEFAULT_INVOCATION_SECONDS = 235
MAX_INVOCATION_SECONDS = 275
PIDS_LIMIT = 1024
CPU_LIMIT = 256
U64_MAX = 2**64 - 1
CID = re.compile(r"[0-9a-f]{64}")
REQUIRED = {"version", "workspace_root", "campaign_directory", "job_directory", "command",
            "mode", "docker_context", "image", "gpu_uuid", "lock_paths", "campaign_id", "invocation_id"}
INSPECT = ('{"Id":{{json .Id}},"Name":{{json .Name}},"Labels":{{json .Config.Labels}},'
           '"ImageReference":{{json .Config.Image}},"State":{{json .State}},"HostConfig":{{json .HostConfig}}}')
assert CONTROL_LIMIT < HISTORY_LIMIT < PAYLOAD_LIMIT


class Refusal(RuntimeError):
    pass


@dataclass(frozen=True)
class InvocationTimeouts:
    payload_seconds: int
    capture_seconds: int
    runner_seconds: int
    controller_reserve_seconds: int


def resolve_timeouts(invocation_seconds=DEFAULT_INVOCATION_SECONDS):
    if type(invocation_seconds) is not int or not 1 <= invocation_seconds <= MAX_INVOCATION_SECONDS:
        raise ValueError(f"invocation_seconds must be an integer in 1..{MAX_INVOCATION_SECONDS}")
    return InvocationTimeouts(invocation_seconds, invocation_seconds + 5,
                              invocation_seconds + 25, invocation_seconds + 25 + 30)


class Signals:
    def __init__(self):
        self.stopping = False

    def request(self, _number, _frame):
        self.stopping = True

    @contextlib.contextmanager
    def installed(self):
        previous = {number: signal.signal(number, self.request) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def require(condition, message):
    if not condition:
        raise Refusal(message)


def read_small(path, limit=CONTROL_LIMIT):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        require(stat.S_ISREG(os.fstat(source.fileno()).st_mode), f"input must be a regular file: {path}")
        data = source.read(limit + 1)
    require(len(data) <= limit, f"oversized input: {path}")
    return data


def write_json(path, value):
    data = json.dumps(value, sort_keys=True).encode() + b"\n"
    require(len(data) <= CONTROL_LIMIT, "control JSON exceeds 64 KiB")
    with open(path, "xb", buffering=0) as output:
        require(output.write(data) == len(data), "short write of control JSON")
        os.fsync(output.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_path(value):
    require(isinstance(value, str) and 0 < len(value.encode()) <= 4096, "invalid path")
    require(not any(character in value for character in ("\x00", "\n", "\r", ",")), "unsafe path")
    path = Path(value)
    require(path.is_absolute() and str(path.resolve(strict=True)) == value, f"path must be canonical: {value}")
    return path


def private_directory(path):
    metadata = path.stat()
    require(stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid(), f"directory ownership: {path}")
    require(metadata.st_mode & 0o077 == 0, f"directory must be private: {path}")


def validate_spec(spec):
    require(isinstance(spec, dict) and REQUIRED <= spec.keys() <= REQUIRED | {"cuda_directory", "invocation_seconds"}, "invalid spec fields")
    resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    require(type(spec["version"]) is int and spec["version"] == 1, "unsupported spec version")
    for name in ("campaign_id", "invocation_id"):
        require(isinstance(spec[name], str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", spec[name]), f"invalid {name}")
    require(spec["docker_context"] == "rootless", "docker_context must be rootless")
    require(isinstance(spec["image"], str) and re.fullmatch(r"[a-z0-9][a-z0-9./:_-]{0,200}@sha256:[0-9a-f]{64}", spec["image"]), "image requires pinned digest")
    require(spec["mode"] in ("cpu", "gpu"), "mode must be cpu or gpu")
    if spec["mode"] == "gpu":
        require(isinstance(spec["gpu_uuid"], str) and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", spec["gpu_uuid"]), "explicit GPU UUID required")
    workspace = canonical_path(spec["workspace_root"])
    campaign = canonical_path(spec["campaign_directory"])
    job = canonical_path(spec["job_directory"])
    require(campaign.is_relative_to(workspace) and campaign != workspace, "campaign must be below workspace")
    require(job.is_relative_to(campaign / "invocations") and job != campaign / "invocations", "job must be below campaign/invocations")
    for path in (campaign, job):
        private_directory(path)
    for name in ("bin", "inputs"):
        require(canonical_path(str(campaign / name)).is_dir(), f"missing campaign {name}")
    command = spec["command"]
    require(isinstance(command, list) and 2 <= len(command) <= 128, "invalid command argument count")
    require(all(isinstance(arg, str) and 0 < len(arg.encode()) <= 4096 and all(ord(c) >= 32 for c in arg) for arg in command), "invalid command argument")
    require(sum(len(arg.encode()) + 1 for arg in command) <= 32768, "command arguments exceed 32 KiB")
    require(Path(command[0]).is_relative_to(campaign / "bin"), "binary must be under campaign/bin")
    binary = canonical_path(command[0])
    metadata = binary.stat()
    require(binary.is_relative_to(campaign / "bin") and stat.S_ISREG(metadata.st_mode), "binary must be regular and under campaign/bin")
    require(metadata.st_uid == os.getuid() and metadata.st_mode & 0o111 and not metadata.st_mode & 0o022, "untrusted binary permissions")
    with binary.open("rb") as source:
        require(source.read(4) == b"\x7fELF", "binary must be native ELF, not a shell")
    require(command[1] == "train-annealed", "binary operation must be train-annealed")
    for flag, expected in (("--invocation-updates", "1"), ("--device", "cuda" if spec["mode"] == "gpu" else "cpu")):
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command) and command[command.index(flag) + 1] == expected, f"required {flag} {expected}")
    if "--device-ordinal" in command:
        index = command.index("--device-ordinal")
        require(command.count("--device-ordinal") == 1 and index + 1 < len(command) and command[index + 1] == "0", "only GPU0 ordinal is supported")
    locks = spec["lock_paths"]
    require(isinstance(locks, list) and 1 <= len(locks) <= 8 and all(isinstance(path, str) for path in locks) and len(set(locks)) == len(locks), "invalid lock_paths")
    for path in locks:
        require(canonical_path(path).is_file(), "lock must be existing regular file")
    if "cuda_directory" in spec:
        cuda = canonical_path(spec["cuda_directory"])
        require(cuda.is_relative_to("/usr/local") and cuda.is_dir(), "cuda_directory must be under /usr/local")
    return spec


def load_spec(path):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "duplicate JSON fields")
        return result
    return validate_spec(json.loads(read_small(path), object_pairs_hook=unique))


def make_intent(spec):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "drysua-training-runner-intent/v1",
            "name": f"drysua-train-{spec['campaign_id'][:20]}-{digest[:20]}",
            "labels": {"io.drysua.training.campaign": spec["campaign_id"],
                       "io.drysua.training.invocation": spec["invocation_id"],
                       "io.drysua.training.spec": digest}}


@contextlib.contextmanager
def shared_locks(paths):
    descriptors, inodes = [], set()
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            descriptors.append(descriptor)
            opened, current = os.fstat(descriptor), os.stat(path, follow_symlinks=False)
            inode = opened.st_dev, opened.st_ino
            require(stat.S_ISREG(opened.st_mode) and inode == (current.st_dev, current.st_ino) and inode not in inodes, "lock inode mismatch or duplicate")
            inodes.add(inode)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Refusal(f"lock busy: {path}; owner untouched") from error
        yield descriptors
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class LockTransfer:
    """Share existing open-file descriptions; never reacquire or explicitly unlock."""
    def __init__(self, path, descriptors):
        self.descriptors, self.sent = descriptors, False
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.address = f"/proc/self/fd/{self.directory}/{path.name}"
        self.socket.bind(self.address)
        self.socket.listen(1)
        self.socket.setblocking(False)

    def send(self):
        connection, _ = self.socket.accept()
        with connection:
            connection.settimeout(1)
            _, user, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            require(user == os.getuid() and not self.sent, "lock transfer peer mismatch")
            connection.sendmsg([b"L"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", self.descriptors))])
            self.sent = True
            require(connection.recv(1) == b"A", "lock transfer acknowledgement missing")

    def close(self):
        self.socket.close()
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None


@contextlib.contextmanager
def received_locks(spec, socket_path="/out/runner-locks.sock"):
    descriptors = []
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(3)
            connection.connect(socket_path)
            data, ancillary, flags, _ = connection.recvmsg(1, socket.CMSG_SPACE(8 * array.array("i").itemsize))
            for level, kind, payload in ancillary:
                require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS, "invalid lock transfer")
                received = array.array("i")
                received.frombytes(payload)
                descriptors.extend(received)
            require(data == b"L" and not flags & socket.MSG_CTRUNC and len(descriptors) == len(spec["lock_paths"]), "incomplete lock transfer")
            for descriptor, path in zip(descriptors, spec["lock_paths"]):
                opened, expected = os.fstat(descriptor), os.stat(path)
                require((opened.st_dev, opened.st_ino) == (expected.st_dev, expected.st_ino), "transferred lock inode mismatch")
            connection.sendall(b"A")
        yield descriptors
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def stop_session(child, grace=4):
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, number)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace if number == signal.SIGTERM else 0.5)
        except subprocess.TimeoutExpired:
            if number == signal.SIGKILL:
                raise Refusal("owned subprocess did not exit after kill")


def bounded_command(command, timeout=5, limit=CONTROL_LIMIT):
    with tempfile.TemporaryFile() as output:
        limiter = functools.partial(resource.setrlimit, resource.RLIMIT_FSIZE, (limit, limit))
        child = subprocess.Popen(command, stdout=output, stderr=output, start_new_session=True, preexec_fn=limiter)
        try:
            code = child.wait(timeout=timeout)
            output.seek(0)
            data = output.read(limit + 1)
            require(len(data) <= limit, "control command output limit")
            require(code == 0, f"control command failed ({code}): {data[:2048].decode(errors='replace')}")
            return data.decode()
        finally:
            stop_session(child, grace=0)


def docker_checked(spec, arguments, timeout=5, limit=CONTROL_LIMIT):
    return bounded_command(["docker", "--context", spec["docker_context"], *arguments], timeout=timeout, limit=limit)


def verify_daemon(spec, timeout=5):
    require(os.getuid() != 0, "host runner must be unprivileged")
    data = json.loads(docker_checked(spec, ["info", "--format", '{"security":{{json .SecurityOptions}},"cgroup":{{json .CgroupVersion}}}'], timeout=timeout))
    require("name=rootless" in data["security"] and data["cgroup"] == "2", "rootless cgroup-v2 daemon required")


def inspect_owned_container(spec, container_id):
    return inspect_container(spec, container_id, require_receipt=True)


def inspect_container(spec, container_id, require_receipt, timeout=5):
    require(isinstance(container_id, str) and CID.fullmatch(container_id), "full container ID required")
    intent = json.loads(read_small(Path(spec["job_directory"]) / "intent.json"))
    require(intent == make_intent(spec), "job intent ownership mismatch")
    receipt = Path(spec["job_directory"]) / "receipt.json"
    require(not require_receipt or receipt.exists(), "container receipt missing; recovery refused")
    if receipt.exists():
        recorded = json.loads(read_small(receipt))
        require(recorded["container_id"] == container_id and recorded["intent"] == intent, "container receipt ownership mismatch")
        if "timeouts" in recorded:
            require(isinstance(recorded["timeouts"], dict) and all(type(value) is int for value in recorded["timeouts"].values()), "invalid container receipt timeout types")
            require(recorded["timeouts"] == asdict(resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))), "container receipt timeout mismatch")
    data = json.loads(docker_checked(spec, ["inspect", "--format", INSPECT, container_id], timeout=timeout))
    require(data["Id"] == container_id and data["Name"] == "/" + intent["name"], "container identity ownership mismatch")
    require(data["ImageReference"] == spec["image"] and all(data["Labels"].get(key) == value for key, value in intent["labels"].items()), "container label/image ownership mismatch")
    state = data["State"]
    require(type(state["Running"]) is bool and type(state["OOMKilled"]) is bool and type(state["ExitCode"]) is int, "invalid container state")
    return data


def stop_owned_container(spec, container_id):
    return stop_container(spec, container_id, time.monotonic() + 25, require_receipt=True)


def remaining_timeout(deadline, maximum):
    remaining = deadline - time.monotonic() - 0.6
    require(remaining > 0, "cleanup deadline exhausted; do not assume stopped")
    return min(remaining, maximum)


def stop_container(spec, container_id, deadline, require_receipt):
    verify_daemon(spec, timeout=remaining_timeout(deadline, 5))
    state = inspect_container(spec, container_id, require_receipt, timeout=remaining_timeout(deadline, 5))["State"]
    if not state["Running"]:
        return state
    for action, timeout in ((["stop", "--time=5", container_id], 8), (["kill", container_id], 5)):
        try:
            docker_checked(spec, action, timeout=remaining_timeout(deadline, timeout))
        except (OSError, Refusal, subprocess.SubprocessError):
            pass
        state = inspect_container(spec, container_id, require_receipt, timeout=remaining_timeout(deadline, 5))["State"]
        if not state["Running"]:
            return state
    raise Refusal("owned container still running after bounded stop/kill")


def cleanup_owned(spec, container_id, deadline):
    try:
        return stop_container(spec, container_id, deadline, require_receipt=False)
    except (OSError, Refusal, KeyError, ValueError, subprocess.SubprocessError) as error:
        failure = str(error)
    # Native timeout and transferred flock descriptions remain independent of the host CLI.
    budget = resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    time.sleep(max(0, min(budget.runner_seconds, deadline - time.monotonic() - 6)))
    try:
        state = inspect_container(spec, container_id, require_receipt=False, timeout=remaining_timeout(deadline, 5))["State"]
        require(not state["Running"], "owned container still running at cleanup deadline")
        return state
    except (OSError, Refusal, KeyError, ValueError, subprocess.SubprocessError) as error:
        raise Refusal(f"CRITICAL cleanup unconfirmed: {failure}; {error}; halt campaign") from error


def create_command(spec, intent):
    campaign, job = Path(spec["campaign_directory"]), Path(spec["job_directory"])
    command = ["create", "--name", intent["name"], "--pull=never", "--user=0:0", "--network=none",
               "--cgroupns=private", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
               "--memory=12g", "--memory-swap=12g", f"--pids-limit={PIDS_LIMIT}", "--restart=no", "--stop-timeout=5",
               "--log-driver=json-file", "--log-opt=max-size=16m", "--log-opt=max-file=1",
               "--tmpfs=/tmp:rw,nosuid,nodev,size=512m", "--workdir", spec["workspace_root"]]
    for key, value in intent["labels"].items():
        command += ["--label", f"{key}={value}"]
    workspace = str(Path(spec["workspace_root"]).parent)
    mounts = [(path, path, True) for path in ("/usr", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/alternatives", workspace)]
    mounts += [(str(campaign), str(campaign), False), (str(campaign / "bin"), str(campaign / "bin"), True),
               (str(campaign / "inputs"), str(campaign / "inputs"), True), (str(job), "/out", False)]
    mounts += [(path, path, True) for path in spec["lock_paths"]]
    for name in ("runner.py", "runner-spec.json"):
        mounts += [(str(job / name), str(job / name), True), (str(job / name), "/out/" + name, True)]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    cuda = spec.get("cuda_directory", "/usr/local/cuda-13.3")
    environment = {"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "CUDA_CACHE_PATH": "/tmp/cuda-cache",
                   "LD_LIBRARY_PATH": cuda + "/lib64", "CUDA_VISIBLE_DEVICES": "0" if spec["mode"] == "gpu" else ""}
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    if spec["mode"] == "gpu":
        for device in ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"):
            command += ["--device", f"/dev/{device}:/dev/{device}"]
    return command + ["--entrypoint=/usr/libexec/docker/docker-init", spec["image"], "--", "/usr/bin/python3",
                      "-B", "/out/runner.py", "inside", "--spec", "/out/runner-spec.json", "--host-uid", str(os.getuid())]


def check_resources(sample, mode, admission=False):
    require(sample["memory_available"] >= (24 if admission else 16) * GIB, "host MemAvailable below threshold")
    cpu = sample["cpu_celsius"]
    require(cpu is None or math.isfinite(cpu) and -20 <= cpu <= 90, "CPU temperature invalid or above 90 C")
    if mode == "gpu":
        require(sample["gpu_free_mib"] is not None and sample["gpu_free_mib"] >= 4096, "GPU VRAM below 4 GiB or unavailable")
        require(sample["gpu_celsius"] is not None and 0 <= sample["gpu_celsius"] <= 85, "GPU temperature invalid or above 85 C")


def unsigned_counter(value):
    require(re.fullmatch(r"[0-9]{1,20}", value) and int(value) <= U64_MAX, "invalid unsigned telemetry counter")
    return int(value)


def parse_cpu_ticks(data):
    require(len(data) <= CONTROL_LIMIT, "CPU telemetry input too large")
    aggregate, processors = None, set()
    for line in data.decode().splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            continue
        require(9 <= len(fields) <= 11, "invalid CPU tick fields")
        values = [unsigned_counter(value) for value in fields[1:]]
        if fields[0] == "cpu":
            require(aggregate is None, "duplicate CPU aggregate")
            # Guest counters are already included in user/nice, not extra CPU time.
            aggregate = {"busy": sum(values[index] for index in (0, 1, 2, 5, 6, 7)), "total": sum(values[:8])}
        else:
            require(re.fullmatch(r"cpu[0-9]{1,4}", fields[0]) and fields[0] not in processors, "invalid or duplicate CPU identity")
            processors.add(fields[0])
            require(len(processors) <= CPU_LIMIT, "CPU count exceeds 256")
    require(aggregate is not None and aggregate["total"] > 0 and processors, "CPU counters unavailable")
    return dict(aggregate, count=len(processors))


def read_cpu_ticks(path="/proc/stat"):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        require(stat.S_ISREG(os.fstat(source.fileno()).st_mode), "CPU source must be a regular file")
        lines = []
        for _ in range(CPU_LIMIT + 2):
            line = source.readline(1024)
            # Linux lists CPU rows first; unrelated IRQ arrays can be much larger.
            if not line.startswith(b"cpu"):
                break
            require(line.endswith(b"\n"), "CPU telemetry row exceeds bound")
            lines.append(line)
    return parse_cpu_ticks(b"".join(lines))


def host_cpu_percent(before, after):
    first, last = (before or {}).get("cpu_ticks"), after.get("cpu_ticks")
    if not first or not last or first["count"] != last["count"]:
        return None
    total, busy = last["total"] - first["total"], last["busy"] - first["busy"]
    return 100 * last["count"] * busy / total if total > 0 and 0 <= busy <= total else None


def parse_gpu_sample(text, uuid):
    fields = [field.strip() for field in text.strip().split(",")]
    require(len(fields) == 7 and fields[0] == "0" and fields[1] == uuid, "selected GPU0 UUID mismatch")
    require(all(re.fullmatch(r"[0-9]{1,7}", value) for value in fields[2:4]), "mandatory GPU free memory/temperature unavailable")
    sample = {"gpu_free_mib": int(fields[2]), "gpu_celsius": int(fields[3])}
    for name, value, ceiling in zip(("gpu_utilization_percent", "gpu_memory_utilization_percent", "gpu_power_watts"), fields[4:], (100, 100, 10000)):
        if value.upper() in ("N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"):
            sample[name] = None
            continue
        try:
            number = float(value)
        except ValueError as error:
            raise Refusal("invalid GPU optional telemetry") from error
        require(math.isfinite(number) and 0 <= number <= ceiling, "invalid GPU optional telemetry range")
        sample[name] = number
    return sample


def update_resource_summary(summary, sample):
    previous = summary.get("last")
    sample["host_cpu_percent"] = host_cpu_percent(previous, sample)
    if previous and sample["host_cpu_percent"] is None:
        summary["cpu_interval_unavailable"] = True
    summary.setdefault("first", sample)
    summary.update(samples=summary["samples"] + 1, last=sample)
    percent = None if summary.get("cpu_interval_unavailable") else host_cpu_percent(summary["first"], sample)
    count = sample.get("cpu_ticks", {}).get("count")
    summary["host_cpu_percent"] = percent
    summary["host_cpu_machine_percent"] = percent / count if percent is not None and count else None
    number = summary.get("gpu_utilization_samples", 0)
    mean = summary.get("gpu_utilization_mean_percent")
    value = sample.get("gpu_utilization_percent")
    if value is not None:
        mean = ((mean or 0) * number + value) / (number + 1)
        number += 1
    summary.update(gpu_utilization_samples=number, gpu_utilization_mean_percent=mean)


def host_sample(spec, admission=False):
    timestamp = time.monotonic()
    lines = read_small("/proc/meminfo").decode().splitlines()
    available = [line.split() for line in lines if line.startswith("MemAvailable:")]
    require(len(available) == 1 and available[0][2:] == ["kB"], "invalid host MemAvailable")
    temperatures = []
    for index in range(64):
        directory = Path(f"/sys/class/hwmon/hwmon{index}")
        if not directory.exists() or read_small(directory / "name", 128).strip() not in (b"k10temp", b"coretemp"):
            continue
        for sensor in range(1, 65):
            path = directory / f"temp{sensor}_input"
            if path.exists():
                temperatures.append(int(read_small(path, 128)) / 1000)
    ticks = read_cpu_ticks()
    sample = {"monotonic": timestamp, "phase": "admission" if admission else "payload",
              "cpu_ticks": ticks, "cpu_count": ticks["count"],
              "memory_available": int(available[0][1]) * 1024, "cpu_celsius": max(temperatures, default=None),
              "gpu_free_mib": None, "gpu_celsius": None, "gpu_utilization_percent": None,
              "gpu_memory_utilization_percent": None, "gpu_power_watts": None}
    if spec["mode"] == "gpu":
        data = bounded_command(["nvidia-smi", "--id=" + spec["gpu_uuid"], "--query-gpu=index,uuid,memory.free,temperature.gpu,utilization.gpu,utilization.memory,power.draw", "--format=csv,noheader,nounits"], timeout=1, limit=4096)
        sample.update(parse_gpu_sample(data, spec["gpu_uuid"]))
    check_resources(sample, spec["mode"], admission)
    return sample


def check_disk(job):
    filesystem = os.statvfs(job)
    require(filesystem.f_bavail * filesystem.f_frsize >= 100 * GIB + PAYLOAD_LIMIT, "disk below 100 GiB plus job log budget")


def validate_created(data):
    require(data["State"]["Status"] == "created" and not data["State"]["Running"], "container must not be started before admission")
    expected = {"Memory": 12 * GIB, "MemorySwap": 12 * GIB, "PidsLimit": PIDS_LIMIT,
                "CpuQuota": 0, "NanoCpus": 0, "CpusetCpus": ""}
    require(all(data["HostConfig"].get(key) == value for key, value in expected.items()), "incorrect created container limits")


def validate_limits(limits):
    for name, value in (("memory.max", str(12 * GIB)), ("memory.swap.max", "0"), ("pids.max", str(PIDS_LIMIT))):
        require(limits.get(name) == value, f"invalid actual {name}")
    require(re.fullmatch(r"max [1-9][0-9]*", limits.get("cpu.max", "")), "CPU quota forbidden: cpu.max")


def validate_uid_mapping(text, host_uid):
    require(host_uid > 0 and text.splitlines()[0].split() == ["0", str(host_uid), "1"], "rootless UID mapping mismatch")


def read_cgroup():
    require(read_small("/proc/self/cgroup", 4096).strip() == b"0::/", "private cgroup v2 required")
    root = Path("/sys/fs/cgroup")
    limits = {name: read_small(root / name, 128).decode().strip() for name in ("memory.max", "memory.swap.max", "pids.max", "cpu.max")}
    validate_limits(limits)
    events = {}
    for subsystem in ("memory", "pids"):
        for line in read_small(root / f"{subsystem}.events", 4096).decode().splitlines():
            name, value = line.split()
            events[f"{subsystem}.{name}"] = int(value)
    require({"memory.max", "memory.high", "memory.oom", "memory.oom_kill", "pids.max"} <= events.keys(), "missing cgroup event counters")
    return {"limits": limits, "events": events}


def cgroup_usage(started, phase="payload"):
    root, counters = Path("/sys/fs/cgroup"), {}
    lines = read_small(root / "cpu.stat", 4096).decode().splitlines()
    require(len(lines) <= 32, "too many cpu.stat counters")
    for line in lines:
        fields = line.split()
        require(len(fields) == 2 and fields[0] not in counters, "invalid cpu.stat fields")
        counters[fields[0]] = unsigned_counter(fields[1])
    require({"usage_usec", "user_usec", "system_usec"} <= counters.keys(), "cpu.stat required counters missing")
    timestamp = time.monotonic()
    usage = {"time": timestamp, "monotonic": timestamp, "elapsed_seconds": timestamp - started, "phase": phase,
             "cpu_stat": {key: counters[key] for key in ("usage_usec", "user_usec", "system_usec")}}
    for name in ("memory.current", "memory.peak", "pids.current"):
        usage[name.replace(".", "_")] = unsigned_counter(read_small(root / name, 128).decode().strip())
    return usage


def cgroup_summary(first, last):
    elapsed = last["monotonic"] - first["monotonic"] if first else 0
    delta = last["cpu_stat"]["usage_usec"] - first["cpu_stat"]["usage_usec"] if first else -1
    return {"cpu_percent": 100 * delta / (elapsed * 1000000) if elapsed > 0 and delta >= 0 else None,
            "elapsed_seconds": elapsed, "memory_peak": last["memory_peak"], "last": last}


def check_events(previous, current):
    require(previous.keys() == current.keys(), "cgroup event keys changed")
    for name, value in current.items():
        require(value == previous[name], f"cgroup event changed: {name}")


def check_deadline(signals, deadline, now=None):
    require(not signals.stopping, "shutdown requested")
    require((time.monotonic() if now is None else now) < deadline, "invocation deadline reached")


def append_bytes(output, data, written, limit):
    require(written + len(data) <= limit, "log limit exceeded")
    require(output.write(data) == len(data), "short write of log")
    return written + len(data)


def inside(spec, host_uid, signals):
    budget = resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    require(os.getuid() == 0, "container UID zero required for rootless mapping")
    validate_uid_mapping(read_small("/proc/self/uid_map", 4096).decode(), host_uid)
    baseline = read_cgroup()
    require(all(value == 0 for value in baseline["events"].values()), "initial resource events nonzero")
    write_json(Path("/out/limits.json"), baseline)
    with received_locks(spec) as descriptors:
        started, current = time.monotonic(), baseline
        deadline = started + budget.payload_seconds
        first_usage = cgroup_usage(started, phase="baseline")
        check_deadline(signals, deadline)
        child = subprocess.Popen(spec["command"], start_new_session=True, pass_fds=descriptors)
        try:
            with open("/out/cgroup-usage.jsonl", "xb", buffering=0) as output:
                written = 0
                for _ in range(budget.payload_seconds + 1):
                    check_deadline(signals, deadline)
                    current = read_cgroup()
                    check_events(baseline["events"], current["events"])
                    usage = {**current, **cgroup_usage(started)}
                    written = append_bytes(output, json.dumps(usage).encode() + b"\n", written, HISTORY_LIMIT)
                    code = child.poll()
                    if code is not None:
                        return code
                    time.sleep(max(0, min(1, deadline - time.monotonic())))
                raise Refusal("inside iteration limit")
        finally:
            stop_session(child)
            current = read_cgroup()
            final_usage = cgroup_usage(started, phase="finalization")
            write_json(Path("/out/limits-final.json"), {**current, "usage_summary": cgroup_summary(first_usage, final_usage)})
            check_events(baseline["events"], current["events"])


def execute_attached(spec, container_id, signals, deadline, lease, summary=None):
    budget = resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    job = Path(spec["job_directory"])
    summary = summary if summary is not None else {}
    summary.setdefault("samples", 0)
    started = time.monotonic()
    child = subprocess.Popen(["docker", "--context", spec["docker_context"], "start", "--attach", container_id],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    cleanup_reserve = budget.runner_seconds - budget.capture_seconds
    deadline, next_sample = min(deadline - cleanup_reserve, started + budget.capture_seconds), started + 1
    try:
        os.set_blocking(child.stdout.fileno(), False)
        with open(job / "payload.log", "xb", buffering=0) as output, open(job / "resources.jsonl", "xb", buffering=0) as history, selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, "payload")
            selector.register(lease.socket, selectors.EVENT_READ, "locks")
            written = history_written = 0
            for _ in range(budget.capture_seconds * 256):
                check_deadline(signals, deadline)
                if time.monotonic() >= next_sample:
                    sample = host_sample(spec)
                    sample["elapsed_seconds"] = sample["monotonic"] - started
                    update_resource_summary(summary, sample)
                    history_written = append_bytes(history, json.dumps(sample).encode() + b"\n", history_written, HISTORY_LIMIT)
                    next_sample = time.monotonic() + 1
                for key, _ in selector.select(timeout=0.1):
                    if key.data == "locks":
                        lease.send()
                        selector.unregister(lease.socket)
                    else:
                        data = os.read(key.fd, 65536)
                        if not data:
                            return child.wait(timeout=2), summary
                        written = append_bytes(output, data, written, PAYLOAD_LIMIT)
            raise Refusal("host capture iteration limit")
    finally:
        stop_session(child, grace=0)
        child.stdout.close()


def validate_success(result):
    require(result["cleanup_confirmed"] and result["container_state"] is not None, "container cleanup unconfirmed")
    state = result["container_state"]
    require(not state["Running"], "container still running")
    require(not state["OOMKilled"] and state["ExitCode"] == 0, "container failed or OOM-killed")
    require(result["verified_limits"] is not None, "missing native cgroup evidence")
    validate_limits(result["verified_limits"])


def finish_result(spec, result):
    job = Path(spec["job_directory"])
    try:
        initial = json.loads(read_small(job / "limits.json"))
        final = json.loads(read_small(job / "limits-final.json"))
        validate_limits(initial["limits"])
        validate_limits(final["limits"])
        check_events(initial["events"], final["events"])
        result["verified_limits"] = final["limits"]
        if "usage_summary" in final:
            summary = final["usage_summary"]
            resources = result.setdefault("resources", {})
            cores = resources.get("admission", {}).get("cpu_count")
            summary["cpu_machine_percent"] = summary["cpu_percent"] / cores if summary["cpu_percent"] is not None and cores else None
            resources["cgroup"] = summary
        if result["returncode"] == 0:
            validate_success(result)
    except (OSError, Refusal, KeyError, ValueError) as error:
        result.update(returncode=1, error=((result.get("error") or "") + "; " + str(error))[:2048])
    write_json(job / "result.json", result)
    return result


def run_job(spec, signals=None):
    spec, signals = validate_spec(spec), signals or Signals()
    budget = resolve_timeouts(spec.get("invocation_seconds", DEFAULT_INVOCATION_SECONDS))
    job, intent, started = Path(spec["job_directory"]), make_intent(spec), time.monotonic()
    deadline = started + budget.runner_seconds
    result = {"schema": "drysua-training-runner/v1", "returncode": 1, "container_id": None,
              "container_state": None, "cleanup_confirmed": False, "verified_limits": None,
              "resources": {}, "timeouts": asdict(budget), "error": None}
    owns_job = False
    try:
        with shared_locks(spec["lock_paths"]) as descriptors:
            write_json(job / "intent.json", intent)
            owns_job = True
            verify_daemon(spec)
            result["resources"]["admission"] = host_sample(spec, admission=True)
            result["resources"]["admission"]["elapsed_seconds"] = time.monotonic() - started
            check_disk(job)
            write_json(job / "runner-spec.json", spec)
            source = read_small(Path(__file__))
            with open(job / "runner.py", "xb", buffering=0) as output:
                require(output.write(source) == len(source), "short write of runner snapshot")
                os.fsync(output.fileno())
            lease = LockTransfer(job / "runner-locks.sock", descriptors)
            try:
                check_deadline(signals, started + budget.capture_seconds)
                container_id = docker_checked(spec, create_command(spec, intent), limit=4096).strip()
                require(CID.fullmatch(container_id), "Docker create did not return full container ID")
                result["container_id"] = container_id
                write_json(job / "receipt.json", {"schema": "drysua-training-runner-receipt/v1", "container_id": container_id,
                                                 "intent": intent, "runner_sha256": hashlib.sha256(source).hexdigest(),
                                                 "timeouts": asdict(budget)})
                validate_created(inspect_owned_container(spec, container_id))
                check_deadline(signals, started + budget.capture_seconds)
                code, summary = execute_attached(spec, container_id, signals, deadline, lease, summary=result["resources"])
                result.update(returncode=code, resources={**result["resources"], **summary})
            except (Exception, KeyboardInterrupt) as error:
                result.update(returncode=1, error=str(error)[:2048])
            finally:
                if result["container_id"]:
                    try:
                        result["container_state"] = cleanup_owned(spec, result["container_id"], deadline)
                        result["cleanup_confirmed"] = True
                    except (Exception, KeyboardInterrupt) as error:
                        result.update(returncode=1, error=str(error)[:2048])
                result["lock_transfer_sent"] = lease.sent
                lease.close()
            return finish_result(spec, result)
    except (Exception, KeyboardInterrupt) as error:
        result.update(returncode=1, error=str(error)[:2048])
        return finish_result(spec, result) if owns_job else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "inside"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--host-uid", type=int)
    arguments = parser.parse_args()
    os.umask(0o077)
    with Signals().installed() as signals:
        try:
            spec = load_spec(arguments.spec)
            if arguments.operation == "inside":
                require(arguments.host_uid is not None, "inside requires host UID")
                return inside(spec, arguments.host_uid, signals)
            require(arguments.host_uid is None, "host UID is not a run option")
            result = run_job(spec, signals)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["returncode"] == 0 else 1
        except (OSError, Refusal, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(json.dumps({"schema": "drysua-training-runner/v1", "returncode": 1,
                              "cleanup_confirmed": False, "error": str(error)[:2048]}))
            return 1


if __name__ == "__main__":
    sys.exit(main())
