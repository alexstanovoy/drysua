"""Rootless-docker stand-in for controller contracts: the attached container is a fake trainer.

State lives beside this script (the controller passes a minimal environment).
`scenario.json` there steers the fake trainer: update_seconds, pause_at, stop_at,
crash_at, signal_controller_at, controller_pid, repository, campaign.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

STATE = Path(__file__).resolve().parent / "state"


def load(name, default):
    path = STATE / name
    return json.loads(path.read_text()) if path.exists() else default


def save(name, value):
    temporary = STATE / f".{name}.{os.getpid()}"
    temporary.write_text(json.dumps(value))
    os.replace(temporary, STATE / name)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"


def create(arguments):
    labels, memory, index = {}, None, 0
    while arguments[index] != "--":
        argument = arguments[index]
        if argument == "--label":
            key, value = arguments[index + 1].split("=", 1)
            labels[key] = value
            index += 1
        elif argument.startswith("--memory="):
            memory = int(argument.split("=", 1)[1])
        index += 1
    identifier = uuid.uuid4().hex + uuid.uuid4().hex
    containers = load("containers.json", {})
    containers[identifier] = {"labels": labels, "image": arguments[index - 1], "memory": memory,
                              "command": arguments[index + 1:], "create": arguments[:index], "pid": 0,
                              "running": False, "exit": None}
    save("containers.json", containers)
    creates = load("creates.json", [])
    save("creates.json", creates + [arguments])
    print(identifier)


def inspect(identifier):
    containers = load("containers.json", {})
    if identifier not in containers:
        print(f"Error: No such object: {identifier}", file=sys.stderr)
        return 1
    record = containers[identifier]
    if record["running"] and not alive(record["pid"]):
        record.update(running=False, exit=record["exit"] if record["exit"] is not None else 137)
    print(json.dumps({"Id": identifier, "Labels": record["labels"], "Image": record["image"],
                      "State": {"Running": record["running"], "Pid": record["pid"] if record["running"] else 0,
                                "ExitCode": record["exit"] or 0, "OOMKilled": False, "Status": "running",
                                "StartedAt": "2026-09-30T00:00:00Z", "FinishedAt": "2026-09-30T00:00:01Z"},
                      "HostConfig": {"Memory": record["memory"], "MemorySwap": record["memory"], "PidsLimit": 1024,
                                     "ReadonlyRootfs": True, "NetworkMode": "none", "CapDrop": ["ALL"],
                                     "Privileged": False}}))
    return 0


def main(arguments):
    arguments = arguments[2:] if arguments[:1] == ["--context"] else arguments
    operation, rest = arguments[0], arguments[1:]
    if operation == "info":
        print(json.dumps({"security": ["name=seccomp,profile=builtin", "name=rootless"], "cgroup": "2"}))
    elif operation == "image":
        print(json.dumps(["docker.io/library/fake@" + rest[-1].split("@", 1)[1]]))
    elif operation == "create":
        create(rest)
    elif operation == "inspect":
        return inspect(rest[-1])
    elif operation == "start":
        return start(rest[-1])
    elif operation == "kill":
        record = load("containers.json", {}).get(rest[-1])
        if record is None or not record["running"] or not alive(record["pid"]):
            print("Error: container is not running", file=sys.stderr)
            return 1
        os.kill(record["pid"], getattr(signal, "SIG" + rest[0].split("=", 1)[1]))
    elif operation == "rm":
        containers = load("containers.json", {})
        containers.pop(rest[-1])
        save("containers.json", containers)
    return 0


def start(identifier):
    containers = load("containers.json", {})
    containers[identifier].update(running=True, pid=os.getpid())
    save("containers.json", containers)
    code = FakeTrainer(containers[identifier]["command"], load("scenario.json", {})).run()
    containers = load("containers.json", {})
    containers[identifier].update(running=False, exit=code)
    save("containers.json", containers)
    return code


class FakeTrainer:
    """Mirrors the native lifecycle: fresh-vs-resume checks, a progress line per update, a durable checkpoint
    every `checkpoint_every` updates (the stand-in for the wall-clock interval), on stop and at the end."""

    def __init__(self, command, scenario):
        self.scenario, self.stopping = scenario, False
        self.command = command[1:]
        self.checkpoint = Path(self.option("--checkpoint-directory"))
        self.history = Path(self.option("--history-directory"))
        self.total, self.every = int(self.option("--updates")), int(self.option("--history-every"))
        self.checkpoint_every = scenario.get("checkpoint_every", 2)
        signal.signal(signal.SIGTERM, self.request_stop)

    def option(self, name):
        return self.command[self.command.index(name) + 1]

    def request_stop(self, *_):
        self.stopping = True

    def run(self):
        meta = self.checkpoint / "checkpoint.meta"
        if "--resume" in self.command and not meta.exists():
            print("checkpoint directory has no committed checkpoint to resume", flush=True)
            return 2
        if "--resume" not in self.command and any(self.checkpoint.iterdir()):
            print("fresh checkpoint directory must be empty", flush=True)
            return 2
        updates = self.durable = json.loads(meta.read_text())["updates"] if meta.exists() else 0
        # Training starts once the controller has read the container cgroup, as a real
        # trainer is still initializing CUDA then; a fake one would otherwise win the race.
        deadline = time.monotonic() + 30
        while not (STATE / f"cgroup-read-{os.getpid()}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        print(f"annealed: updates={self.total} games=2 parallel=2 generation_games=2 zero_updates=0 seed=1 "
              "opponent=Teacher", flush=True)
        while updates < self.total:
            update = updates + 1
            if not self.play(update):
                return 3
            self.progress(update)
            updates = update
            if self.stopping or update == self.total or update % self.checkpoint_every == 0:
                self.commit(update)
            if self.stopping:
                print(f"level=INFO event=training_stopped reason=signal completed_updates={update}", flush=True)
                break
        return 0

    def play(self, update):
        for outcome, opponent in (("Win", "Teacher"), ("Loss", "Teacher")):
            print(f"episode: stream=0 map=2 opponent={opponent} tick={1000 * update} outcome={outcome} "
                  f"actions=[3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] raw_return=0.5", flush=True)
        time.sleep(self.scenario.get("update_seconds", 0.01))
        if update == self.scenario.get("crash_at"):
            return False
        for key, operation in (("pause_at", "pause"), ("stop_at", "stop")):
            if update == self.scenario.get(key):
                subprocess.run([sys.executable, str(Path(self.scenario["repository"]) / "scripts/train.py"),
                                operation, self.scenario["campaign"]], check=True, stdout=subprocess.DEVNULL)
                self.wait_for_signal()
        if update == self.scenario.get("signal_controller_at"):
            os.kill(self.scenario["controller_pid"], signal.SIGTERM)
            self.wait_for_signal()
        return True

    def wait_for_signal(self):
        # An immediate stop kills this process here, in the middle of the update.
        deadline = time.monotonic() + 30
        while not self.stopping and time.monotonic() < deadline:
            time.sleep(0.01)

    def progress(self, update):
        print(f"level=INFO event=training_update_timing update_index={update - 1} elapsed_ns=2000000000 "
              f"collection_ns=1500000000 optimization_ns=500000000 samples=100", flush=True)
        print(f"progress: update {update}, samples {100 * update}, optimizer step {update}, "
              f"policy loss -0.01, value loss 0.02, entropy 0.5, KL 0.001, KL stop false", flush=True)

    def commit(self, update):
        temporary = self.checkpoint / "checkpoint.meta.tmp"
        temporary.write_text(json.dumps({"updates": update}))
        os.replace(temporary, self.checkpoint / "checkpoint.meta")
        if update // self.every > self.durable // self.every or update == self.total:
            milestone = self.history / f"u{update:04d}"
            milestone.mkdir()
            (milestone / "drysua.weights.safetensors").write_text(str(update))
        self.durable = update
        print(f"checkpoint: update {update}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
