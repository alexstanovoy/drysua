"""One rootless, resource-bounded container per training session.

The trainer runs every remaining update in one process, so CUDA JIT and
container start are paid once per session. The controller owns the container
by full ID and labels, verifies its actual cgroup limits from the host, streams
its output into a bounded log, samples host health, and stops it gracefully
(SIGTERM, one update and a checkpoint) or immediately (SIGKILL). The committed
checkpoint on disk stays valid under both.
"""
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

import train_io as io

GIB = 1024 ** 3
PIDS_LIMIT = 1024
LOG_LIMIT = GIB
RESOURCE_LOG_LIMIT = 16 * 1024 * 1024
CUDA_CACHE_BYTES = GIB
SAMPLE_SECONDS = 5
POLL_SECONDS = 0.25
ADMISSION_MEMORY = 24 * GIB
OPERATING_MEMORY = 16 * GIB
ADMISSION_DISK = 16 * GIB
OPERATING_DISK = 4 * GIB
GPU_FREE_MIB = 4096
CPU_CELSIUS = 90
GPU_CELSIUS = 85
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
DOCKER_INIT = "/usr/libexec/docker/docker-init"
HOST_MOUNTS = ("/usr", "/etc/ld.so.cache", "/etc/alternatives")
GPU_DEVICES = ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools")
LABEL_CAMPAIGN = "io.drysua.training.campaign"
LABEL_SESSION = "io.drysua.training.session"


def docker(config, *arguments, timeout=10, limit=65536):
    command = ["docker", "--context", config["docker_context"], *arguments]
    return io.capture(command, timeout=timeout, limit=limit).decode()


def verify_daemon(config):
    data = json.loads(docker(config, "info", "--format",
                             '{"security":{{json .SecurityOptions}},"cgroup":{{json .CgroupVersion}}}'))
    if "name=rootless" not in data["security"] or data["cgroup"] != "2":
        raise ValueError("a rootless cgroup v2 Docker daemon is required")
    digests = json.loads(docker(config, "image", "inspect", "--format", "{{json .RepoDigests}}", config["image"]))
    if not isinstance(digests, list) or config["image"].split("@", 1)[1] not in {
            entry.split("@", 1)[-1] for entry in digests}:
        raise ValueError("pinned image digest is not present locally; pulling is never automatic")


def gpu_sample(config):
    """Free VRAM, temperature and utilization of the pinned GPU, plus its device index."""
    output = io.capture(["nvidia-smi", "--id=" + config["gpu_uuid"],
                         "--query-gpu=index,uuid,memory.free,memory.used,temperature.gpu,utilization.gpu",
                         "--format=csv,noheader,nounits"], timeout=5, limit=4096).decode()
    fields = [field.strip() for field in output.strip().split(",")]
    if len(fields) != 6 or fields[1] != config["gpu_uuid"]:
        raise ValueError("pinned GPU UUID is not visible to nvidia-smi")
    values = {}
    names = ("index", "gpu_free_mib", "gpu_used_mib", "gpu_celsius", "gpu_percent")
    for name, value in zip(names, fields[:1] + fields[2:]):
        values[name] = int(value) if value.isdigit() else None
    if values["index"] is None or values["gpu_free_mib"] is None or values["gpu_celsius"] is None:
        raise ValueError("mandatory GPU index, free memory or temperature unavailable")
    return values


def host_sample(config, directory):
    """Host health: available memory, free disk, hottest CPU sensor and the pinned GPU."""
    lines = io.read_kernel("/proc/meminfo", 65536).decode().splitlines()
    available = [line.split() for line in lines if line.startswith("MemAvailable:")]
    if len(available) != 1 or available[0][2:] != ["kB"]:
        raise ValueError("invalid host MemAvailable")
    filesystem = os.statvfs(directory)
    sample = {"memory_available": int(available[0][1]) * 1024,
              "disk_free": filesystem.f_bavail * filesystem.f_frsize,
              "cpu_celsius": cpu_celsius()}
    if config["mode"] == "gpu":
        sample.update(gpu_sample(config))
    return sample


def cpu_celsius():
    temperatures = []
    for index in range(64):
        directory = Path(f"/sys/class/hwmon/hwmon{index}")
        if not directory.exists() or io.read_kernel(directory / "name", 128).strip() not in (b"k10temp", b"coretemp"):
            continue
        for sensor in range(1, 65):
            path = directory / f"temp{sensor}_input"
            if path.exists():
                temperatures.append(int(io.read_kernel(path, 128)) / 1000)
    return max(temperatures, default=None)


def health_violation(sample, admission=False):
    """Reason the host cannot (keep) running the trainer, or None."""
    memory = ADMISSION_MEMORY if admission else OPERATING_MEMORY
    if sample["memory_available"] < memory:
        return f"host MemAvailable below {memory // GIB} GiB"
    disk = ADMISSION_DISK if admission else OPERATING_DISK
    if sample["disk_free"] < disk:
        return f"free disk below {disk // GIB} GiB"
    if sample["cpu_celsius"] is not None and sample["cpu_celsius"] > CPU_CELSIUS:
        return f"CPU above {CPU_CELSIUS} C"
    if "gpu_celsius" in sample and sample["gpu_celsius"] > GPU_CELSIUS:
        return f"GPU above {GPU_CELSIUS} C"
    if "gpu_free_mib" in sample and sample["gpu_free_mib"] < GPU_FREE_MIB:
        return f"GPU free memory below {GPU_FREE_MIB} MiB"
    return None


def container_name(campaign_id, session):
    return f"drysua-train-{campaign_id[:16]}-s{session:04d}"


def create_argv(directory, config, campaign_id, session, command, gpu_index):
    """`docker create` arguments; the trainer is the entrypoint's only child, no shell."""
    memory = config["memory_gib"] * GIB
    argv = ["create", "--name", container_name(campaign_id, session), "--pull=never", "--user=0:0",
            "--network=none", "--cgroupns=private", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", f"--memory={memory}", f"--memory-swap={memory}",
            f"--pids-limit={PIDS_LIMIT}", "--restart=no", f"--stop-timeout={config['stop_seconds']}",
            "--log-driver=json-file", "--log-opt=max-size=16m", "--log-opt=max-file=1",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=512m", "--workdir=/tmp",
            "--label", f"{LABEL_CAMPAIGN}={campaign_id}", "--label", f"{LABEL_SESSION}={session}"]
    mounts = [(path, True) for path in HOST_MOUNTS]
    mounts += [(str(directory / name), name in ("bin", "inputs"))
               for name in ("bin", "inputs", "checkpoint", "history", "cuda-cache")]
    for path, readonly in mounts:
        argv += ["--mount", f"type=bind,src={path},dst={path}" + (",readonly" if readonly else "")]
    environment = {"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "LD_LIBRARY_PATH": config["cuda_directory"] + "/lib64",
                   "CUDA_CACHE_PATH": str(directory / "cuda-cache"),
                   "CUDA_CACHE_MAXSIZE": str(CUDA_CACHE_BYTES),
                   "CUDA_VISIBLE_DEVICES": config["gpu_uuid"] if config["mode"] == "gpu" else ""}
    for key, value in environment.items():
        argv += ["--env", f"{key}={value}"]
    if config["mode"] == "gpu":
        for device in (f"nvidia{gpu_index}", *GPU_DEVICES):
            argv += ["--device", f"/dev/{device}:/dev/{device}"]
    return argv + [f"--entrypoint={DOCKER_INIT}", config["image"], "--", *command]


def create_container(directory, config, campaign_id, session, command, gpu_index):
    container_id = docker(config, *create_argv(directory, config, campaign_id, session, command, gpu_index),
                          timeout=30).strip()
    if not CONTAINER_ID.fullmatch(container_id):
        raise ValueError("docker create did not return a full container ID")
    return container_id


def inspect_owned(config, container_id, campaign_id, session):
    """State and limits of a container proven to be this session's by full ID and labels."""
    data = json.loads(docker(config, "inspect", "--format",
                             '{"Id":{{json .Id}},"Labels":{{json .Config.Labels}},"Image":{{json .Config.Image}},'
                             '"State":{{json .State}},"HostConfig":{{json .HostConfig}}}', container_id))
    labels = data["Labels"] or {}
    if (data["Id"] != container_id or data["Image"] != config["image"] or
            labels.get(LABEL_CAMPAIGN) != campaign_id or labels.get(LABEL_SESSION) != str(session)):
        raise ValueError("container identity does not match this campaign session")
    return data


def verify_declared_limits(config, data):
    memory = config["memory_gib"] * GIB
    expected = {"Memory": memory, "MemorySwap": memory, "PidsLimit": PIDS_LIMIT, "ReadonlyRootfs": True,
                "NetworkMode": "none", "CapDrop": ["ALL"], "Privileged": False}
    for key, value in expected.items():
        if data["HostConfig"].get(key) != value:
            raise ValueError(f"created container has unexpected {key}")


def read_cgroup(pid):
    """Actual limits, events and usage of the cgroup that owns `pid`, read from the host."""
    line = io.read_kernel(f"/proc/{pid}/cgroup", 4096).decode().strip()
    if not line.startswith("0::/") or "\n" in line or ".." in line:
        raise ValueError("container process is not in a single cgroup v2 hierarchy")
    root = Path("/sys/fs/cgroup" + line[3:])
    values = {name: io.read_kernel(root / name, 4096).decode().strip()
              for name in ("memory.max", "memory.swap.max", "pids.max", "cpu.max",
                           "memory.current", "memory.peak", "pids.current")}
    for name in ("memory.events", "pids.events"):
        for entry in io.read_kernel(root / name, 4096).decode().splitlines():
            key, value = entry.split()
            values[f"{name}.{key}"] = int(value)
    for entry in io.read_kernel(root / "cpu.stat", 4096).decode().splitlines()[:32]:
        key, value = entry.split()
        if key == "usage_usec":
            values["cpu_usage_usec"] = int(value)
    values["uid_map"] = io.read_kernel(f"/proc/{pid}/uid_map", 4096).decode().split()[:3]
    return values


def verify_actual_limits(config, values):
    expected = {"memory.max": str(config["memory_gib"] * GIB), "memory.swap.max": "0", "pids.max": str(PIDS_LIMIT)}
    for name, value in expected.items():
        if values.get(name) != value:
            raise ValueError(f"container cgroup {name} is {values.get(name)!r}, expected {value}")
    if not values.get("cpu.max", "").startswith("max "):
        raise ValueError("container cgroup has a CPU quota")
    if values.get("uid_map") != ["0", str(os.getuid()), "1"]:
        raise ValueError("container root is not mapped to the invoking user")


def limit_events(values):
    return {key: value for key, value in values.items()
            if key in ("memory.events.oom", "memory.events.oom_kill", "memory.events.max", "pids.events.max")}


def signal_container(config, container_id, name):
    """Best effort: the container may already have exited, which is what we want."""
    try:
        docker(config, "kill", f"--signal={name}", container_id)
    except ValueError:
        pass


class Supervisor:
    """Streams one started container into a bounded log and applies stop requests.

    `control()` returns None, `("pause", reason)` for a graceful stop after the
    in-flight update and its checkpoint, or `("stop", reason)` for an immediate kill. A graceful
    stop escalates to a kill after `stop_seconds`.
    """

    def __init__(self, config, container_id, campaign_id, session, session_directory, control):
        self.config, self.container_id = config, container_id
        self.campaign_id, self.session = campaign_id, session
        self.session_directory, self.control = session_directory, control
        self.request, self.reason = None, None
        self.kill_at, self.cgroup_pid, self.events = None, None, None
        self.log_bytes, self.resource_bytes = 0, 0

    def run(self, started):
        hard_deadline = started + self.config["max_seconds"] + self.config["stop_seconds"] + 120
        attach = subprocess.Popen(["docker", "--context", self.config["docker_context"], "start", "--attach",
                                   self.container_id], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=io.environment(), start_new_session=True)
        try:
            os.set_blocking(attach.stdout.fileno(), False)
            with (open(self.session_directory / "payload.log", "xb", buffering=0) as log,
                  open(self.session_directory / "resources.jsonl", "xb", buffering=0) as resources,
                  selectors.DefaultSelector() as selector):
                selector.register(attach.stdout, selectors.EVENT_READ)
                next_sample = time.monotonic()
                while True:
                    now = time.monotonic()
                    if now > hard_deadline:
                        self.stop("stop", "controller hard deadline")
                        break
                    if not self.pump(selector, attach, log):
                        break
                    if self.cgroup_pid is None and self.request != "stop":
                        self.guard(self.verify_started)
                    if now >= next_sample:
                        self.sample(resources, now - started)
                        next_sample = now + SAMPLE_SECONDS
                    self.apply(now)
        finally:
            if attach.poll() is None:
                signal_container(self.config, self.container_id, "KILL")
            io.terminate(attach, grace=10)
            attach.stdout.close()
        return self.finish()

    def pump(self, selector, attach, log):
        """Copies available output; False once the attached container has exited."""
        for key, _ in selector.select(timeout=POLL_SECONDS):
            data = os.read(key.fd, 65536)
            if not data:
                attach.wait(timeout=30)
                return False
            if self.log_bytes + len(data) > LOG_LIMIT:
                self.stop("pause", "session log limit reached")
                continue
            log.write(data)
            self.log_bytes += len(data)
        return True

    def guard(self, action):
        try:
            action()
        except FileNotFoundError:
            pass  # The container exited between inspection and reading its cgroup.
        except (OSError, ValueError, KeyError) as error:
            self.stop("stop", f"container verification failed: {error}")

    def sample(self, resources, elapsed):
        record = {"elapsed_seconds": round(elapsed, 3)}

        def measure():
            if self.cgroup_pid is not None:
                values = read_cgroup(self.cgroup_pid)
                record.update(cpu_usage_usec=values["cpu_usage_usec"], memory_current=int(values["memory.current"]),
                              memory_peak=int(values["memory.peak"]), pids_current=int(values["pids.current"]))
                if limit_events(values) != self.events:
                    self.stop("pause", f"container cgroup limit event: {limit_events(values)}")
            health = host_sample(self.config, self.session_directory)
            record.update(health)
            violation = health_violation(health)
            if violation:
                self.stop("pause", violation)

        self.guard(measure)
        data = json.dumps(record, sort_keys=True).encode() + b"\n"
        if self.resource_bytes + len(data) <= RESOURCE_LOG_LIMIT:
            resources.write(data)
            self.resource_bytes += len(data)

    def verify_started(self):
        data = inspect_owned(self.config, self.container_id, self.campaign_id, self.session)
        pid = data["State"].get("Pid")
        if not data["State"].get("Running") or type(pid) is not int or pid < 1:
            return
        values = read_cgroup(pid)
        verify_actual_limits(self.config, values)
        self.cgroup_pid, self.events = pid, limit_events(values)

    def stop(self, request, reason):
        """Records the strongest request; the first reason of that strength wins."""
        if self.request == "stop" or self.request == request:
            return
        self.request, self.reason = request, reason
        if request == "stop":
            signal_container(self.config, self.container_id, "KILL")
        else:
            signal_container(self.config, self.container_id, "TERM")
            self.kill_at = time.monotonic() + self.config["stop_seconds"]

    def apply(self, now):
        requested = self.control()
        if requested is not None:
            self.stop(*requested)
        if self.kill_at is not None and now >= self.kill_at and self.request != "stop":
            self.stop("stop", f"graceful stop exceeded {self.config['stop_seconds']} s: {self.reason}")

    def finish(self):
        """Final owned-container state; the container is removed once stopped."""
        for _ in range(3):
            state = inspect_owned(self.config, self.container_id, self.campaign_id, self.session)["State"]
            if not state["Running"]:
                break
            signal_container(self.config, self.container_id, "KILL")
            time.sleep(1)
        else:
            raise ValueError("owned container is still running after kill; run recover")
        docker(self.config, "rm", self.container_id, timeout=30)
        return {"request": self.request, "reason": self.reason, "exit_code": state["ExitCode"],
                "oom_killed": state["OOMKilled"], "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"), "verified_limits": self.cgroup_pid is not None}


def remove_orphan(config, container_id, campaign_id, session):
    """Kills and removes a recorded session container left by a dead controller."""
    try:
        state = inspect_owned(config, container_id, campaign_id, session)["State"]
    except ValueError as error:
        if "No such" in str(error):
            return None
        raise
    if state["Running"]:
        signal_container(config, container_id, "KILL")
        time.sleep(1)
        state = inspect_owned(config, container_id, campaign_id, session)["State"]
        if state["Running"]:
            raise ValueError("orphaned session container is still running after kill")
    docker(config, "rm", container_id, timeout=30)
    return state
