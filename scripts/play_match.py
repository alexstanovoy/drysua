"""Linux local-match supervisor used by the root play.sh launcher."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import select
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO

from play_admission import Admission
from play_weights import read_runtime_metadata


ARTIFACT_LIMIT = 256 * 1024 * 1024
BUILD_TIMEOUT = 1200
LOG_LIMIT = 16 * 1024 * 1024
MAX_CHILDREN = 5
READ_CHUNK = 65536
READINESS_LIMIT = 4096
READINESS_TIMEOUT = 10
REPLAY_LIMIT = 2 * 1024**3
REVIEW_DIRECTORY = Path("drysua/artifacts/temp/human-review-20260909/current")
REVIEW_BINARY_SHA = "64ba25ebb10e6beabc26ff667a3e3bddeb40dbf391110d4f0e31db6478500a2b"
REVIEW_WEIGHTS_SHA = "6348fe57a128ebd521dba68da0949ceb7378d3e0d6b7d6a7ce9a5fb6446547ab"
TERM_GRACE = 2
assert READINESS_LIMIT < READ_CHUNK
assert READ_CHUNK <= LOG_LIMIT
assert LOG_LIMIT < REPLAY_LIMIT
assert 0 < TERM_GRACE <= 3


def main(arguments=None):
    arguments = parse_arguments(arguments)
    root = Path(__file__).resolve().parents[2]
    supervisor, previous, mask = None, {}, None
    status = 1
    try:
        _, arguments.weights_directory = current_paths(root, arguments.weights_directory)
        preflight(root, arguments.no_build)
        if arguments.no_build:
            release_executables(root)
        mask = os.umask(0o077)
        parent = root / "drysua"
        for part in ("artifacts", "temp"):
            parent = parent / part
            if parent.is_symlink():
                raise RuntimeError(f"temporary artifact directory must not be a symlink: {parent}")
            parent.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="play-", dir=parent))
        supervisor = Supervisor(directory)
        previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[number] = signal.signal(number, supervisor.request_stop)
        print(f"play: logs: {directory}", flush=True)
        supervisor.run(root, arguments)
        status = 0
    except Exception as error:
        if supervisor is None or not supervisor.stop_status:
            print(f"play: {error}", file=sys.stderr)
    finally:
        if supervisor is not None:
            try:
                supervisor.close()
            except Exception as error:
                print(f"play: cleanup failed: {error}", file=sys.stderr)
                status = 1
            status = supervisor.stop_status or status
            if status:
                print(f"play: stopped (status {status}); logs: {supervisor.directory}", file=sys.stderr)
        for number, handler in previous.items():
            signal.signal(number, handler)
        if mask is not None:
            os.umask(mask)
    return status


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(
        prog="play.sh", description="Current Map2 pure Neural play; compatible explicit weights required (no default model).")
    parser.add_argument("--port", type=lambda value: integer(value, 65535), default=4455,
                        help="Server port, or 0 for an assigned port (default: 4455)")
    parser.add_argument("--seed", type=lambda value: integer(value, 2**64 - 1), default=9000001,
                        help="Match seed (default: 9000001)")
    parser.add_argument("--no-build", action="store_true",
                        help="Use existing current bota server/client and drysua release binaries")
    parser.add_argument("--human-side", choices=("radiant", "dire"),
                        help="Human side (default: radiant, or opposite --bot-side)")
    parser.add_argument("--bot-side", choices=("radiant", "dire"),
                        help="Neural bot side (default: dire, or opposite --human-side)")
    parser.add_argument("--weights-directory", type=Path,
                        help="Directory with compatible current F15/M17 runtime weights; no trained Map2 default yet")
    result = parser.parse_args(arguments)
    opposite = {"radiant": "dire", "dire": "radiant"}
    if result.human_side is None:
        result.human_side = opposite[result.bot_side] if result.bot_side else "radiant"
    if result.bot_side is None:
        result.bot_side = opposite[result.human_side]
    if result.human_side == result.bot_side:
        parser.error("--human-side and --bot-side must be opposite")
    return result


def current_paths(root, weights_directory):
    if weights_directory is None:
        raise RuntimeError("legacy F12/M14 human-review weights are incompatible with current Map2; "
                           "provide --weights-directory with compatible F15/M17 runtime weights; "
                           "no Map2 model has been trained or promoted; no Teacher fallback")
    weights = weights_directory.resolve()
    read_runtime_metadata(weights)
    return root / "drysua/target/release/drysua", weights


def release_executables(root):
    binaries = [root / "bota/target/release/bota-server", root / "bota/target/release/bota-client",
                root / "drysua/target/release/drysua"]
    assert len(binaries) == 3
    for executable in binaries:
        if executable.is_symlink() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"release executable missing: {executable}; rerun without --no-build")
    assert all(executable.is_absolute() for executable in binaries)
    return binaries


def review_paths(root, weights_directory):
    """Historical archive utility only; never called by the current launcher."""
    review = root / REVIEW_DIRECTORY
    binary = review / "drysua"
    binary_digest = artifact_digest(binary)
    if binary_digest != REVIEW_BINARY_SHA:
        raise RuntimeError(f"pinned review binary SHA-256 mismatch: {binary}; never rebuild this copy")
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"pinned review binary is not executable: {binary}")
    manifest = review / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size > READ_CHUNK:
        raise RuntimeError(f"pinned review manifest missing, non-regular or exceeds 64 KiB: {manifest}")
    try:
        with manifest.open("rb") as stream:
            data = stream.read(READ_CHUNK + 1)
        if len(data) > READ_CHUNK or not isinstance(json.loads(data), dict):
            raise ValueError("expected a bounded JSON object")
    except (ValueError, OSError) as error:
        raise RuntimeError(f"invalid review manifest {manifest}: {error}") from error
    weights = weights_directory.resolve() if weights_directory is not None else review / "weights"
    weights_digest = artifact_digest(weights / "drysua.weights.safetensors")
    if weights_directory is None and weights_digest != REVIEW_WEIGHTS_SHA:
        raise RuntimeError(f"pinned review weights SHA-256 mismatch: {weights}; no Teacher fallback")
    selection = "explicit weights override" if weights_directory is not None else "corrected-ppo-001/u4"
    print(f"play: human-review Neural {selection}; weights: {weights}; SHA-256 {weights_digest}", flush=True)
    print(f"play: frozen review executable: {binary}; SHA-256 {binary_digest}", flush=True)
    return binary, weights


def artifact_digest(path):
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= ARTIFACT_LIMIT:
        raise RuntimeError(f"review artifact missing, non-regular or outside 1..{ARTIFACT_LIMIT} bytes: {path}; "
                           "populate the immutable review copy; no build or Teacher fallback")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for _ in range(ARTIFACT_LIMIT // READ_CHUNK + 1):
            data = stream.read(READ_CHUNK)
            if not data:
                assert 0 < size <= ARTIFACT_LIMIT
                return digest.hexdigest()
            size += len(data)
            if size > ARTIFACT_LIMIT:
                break
            digest.update(data)
    raise RuntimeError(f"review artifact grew beyond {ARTIFACT_LIMIT} bytes while hashing: {path}")


def integer(value, maximum):
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a decimal integer in 0..{maximum}") from error
    if not 0 <= result <= maximum:
        raise argparse.ArgumentTypeError(f"expected a decimal integer in 0..{maximum}")
    return result


def preflight(root, no_build):
    if sys.platform != "linux":
        raise RuntimeError("this launcher requires Linux process groups")
    if not os.environ.get("DISPLAY", "").strip():
        raise RuntimeError("DISPLAY is required: run from an authorized X11/Xwayland desktop terminal; "
                           "a headless or WAYLAND_DISPLAY-only session cannot run bota-client")
    for repository in ("bota", "drysua"):
        manifest = root / repository / "Cargo.toml"
        if not manifest.is_file():
            raise RuntimeError(f"source manifest missing: {manifest}")
    if not no_build and shutil.which("cargo") is None:
        raise RuntimeError("cargo is required to build; install Rust or use --no-build with release binaries")


def ready_port(data, requested):
    line, newline, _ = data.partition(b"\n")
    if len(line) > READINESS_LIMIT:
        raise ValueError(f"server readiness exceeds {READINESS_LIMIT} bytes")
    if not newline:
        return None
    match = re.fullmatch(rb"bota-server listening on 0\.0\.0\.0:([0-9]{1,5})", line)
    port = int(match[1]) if match else 0
    if not 1 <= port <= 65535 or requested not in (0, port):
        raise ValueError("invalid server readiness line or unexpected port")
    return port


@dataclass
class Child:
    name: str
    process: subprocess.Popen
    log: BinaryIO
    size: int = 0
    banner: bytearray = field(default_factory=bytearray)
    port: int | None = None
    tail: bytes = b""

    def exit_status(self):
        # Keep leaders unreaped so their process-group IDs cannot be reused before cleanup.
        assert self.process.pid > 1
        assert self.process.returncode is None
        result = os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if result is None:
            return None
        return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


class Supervisor:
    def __init__(self, directory):
        assert directory.is_dir()
        self.directory = directory
        self.children = []
        self.selector = selectors.DefaultSelector()
        self.stop_status = 0
        self.requested_port = 0
        self.admission = None

    def request_stop(self, number, _frame):
        # A handler must not interrupt Popen before the new child has been registered.
        if not self.stop_status:
            self.stop_status = 128 + number

    def check_stop(self):
        if self.stop_status:
            raise InterruptedError("shutdown requested")

    def spawn(self, name, command, directory, environment=None):
        self.check_stop()
        assert len(self.children) < MAX_CHILDREN
        assert not any(child.name == name for child in self.children)
        log = (self.directory / f"{name}.log").open("xb", buffering=0)
        try:
            process = subprocess.Popen(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        except BaseException:
            log.close()
            raise
        child = Child(name, process, log)
        self.children.append(child)
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, (child, stream is process.stdout))
        print(f"play: started {name} (pid {process.pid})", flush=True)
        return child

    def pump(self, timeout, final=False):
        if self.admission is not None and not final and self.admission.pump():
            timeout = 0
        events = self.selector.select(timeout)
        assert len(events) <= MAX_CHILDREN * 2
        for key, _ in events:
            child, stdout = key.data
            data = os.read(key.fileobj.fileno(), READ_CHUNK)
            if not data:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                if not final and child.name == "server" and stdout and child.port is None:
                    raise RuntimeError("server stdout closed before readiness")
                continue
            kept = data[:LOG_LIMIT - child.size]
            if child.log.write(kept) != len(kept):
                raise OSError(f"short write to {child.name} log")
            child.size += len(kept)
            assert child.size <= LOG_LIMIT
            if len(kept) != len(data):
                raise RuntimeError(f"{child.name} log limit exceeded ({LOG_LIMIT} bytes)")
            if child.name == "client" and not stdout:
                if b"bota-client: " in child.tail + data:
                    raise RuntimeError("bota-client reported an error; see client.log")
                child.tail = (child.tail + data)[-32:]
            if final:
                continue
            if child.name == "server" and stdout and child.port is None:
                child.banner.extend(data[:READINESS_LIMIT + 1 - len(child.banner)])
                child.port = ready_port(child.banner, self.requested_port)
        if self.admission is not None and not final:
            self.admission.pump()
        return len(events)

    def wait_build(self, child):
        deadline = time.monotonic() + BUILD_TIMEOUT
        while True:
            self.check_stop()
            self.pump(0.1)
            status = child.exit_status()
            if status is not None:
                if status:
                    raise RuntimeError(f"{child.name} exited with status {status}; see {child.name}.log")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{child.name} build exceeded {BUILD_TIMEOUT} seconds")

    def wait_ready(self, server, requested):
        self.requested_port = requested
        deadline = time.monotonic() + READINESS_TIMEOUT
        while True:
            self.check_stop()
            if server.exit_status() is not None:
                raise RuntimeError("server exited before readiness; see server.log")
            if server.port is not None:
                return server.port
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"server readiness timed out after {READINESS_TIMEOUT} seconds")
            self.pump(min(0.1, remaining))

    def run(self, root, arguments):
        binary, weights = current_paths(root, arguments.weights_directory)
        print(f"play: current Map2 pure Neural F15/M17; weights: {weights}; executable: {binary}; "
              "metadata preflight only, Rust validates tensors before joining", flush=True)
        if not arguments.no_build:
            command = ["cargo", "build", "--release", "--locked", "--quiet",
                       "--manifest-path", str(root / "bota/Cargo.toml"), "-p", "bota-server",
                       "-p", "bota-client", "--bin", "bota-server", "--bin", "bota-client"]
            environment = dict(os.environ, CARGO_TARGET_DIR=str(root / "bota/target"))
            self.wait_build(self.spawn("build-bota", command, root, environment))
            command = ["cargo", "build", "--release", "--locked", "--quiet", "--bin", "drysua", "--no-default-features"]
            environment = dict(os.environ, CARGO_TARGET_DIR=str(root / "drysua/target"))
            self.wait_build(self.spawn("build-drysua", command, root / "drysua", environment))
        binaries = release_executables(root)
        assert binary == binaries[2]
        server = self.spawn("server", [str(binaries[0]), "--port", str(arguments.port), "--mode", "realtime",
                            "--players", "2", "--map", "2", "--seed", str(arguments.seed),
                            "--replay", str(self.directory / "match.brp")], root)
        resource.prlimit(server.process.pid, resource.RLIMIT_FSIZE, (REPLAY_LIMIT, REPLAY_LIMIT))
        port = self.wait_ready(server, arguments.port)
        self.admission = Admission(port, arguments.human_side)
        client = self.spawn("client", [str(binaries[1]), "--addr", self.admission.addresses["human"],
                                      "--name", "human"], root)
        bot = self.spawn("bot", [str(binary), "--addr", self.admission.addresses["bot"], "--name", "drysua",
                                 "--policy", "neural", "--weights-directory", str(weights)], root)
        print(f"play: requested human {arguments.human_side} / Neural bot {arguments.bot_side}; "
              "verifying server Welcome seats. Choose a hero (1/2/3), then R to ready. Ctrl+C stops all.",
              flush=True)
        self.wait_game(server, bot, client)

    def wait_game(self, server, bot, client):
        # The results window controls lifetime; successful server/bot exits do not close it.
        disconnected = {}
        while True:
            self.check_stop()
            for child in (server, bot, client):
                status = child.exit_status()
                if status not in (None, 0):
                    raise RuntimeError(f"{child.name} exited with status {status}; see {child.name}.log")
            if client.exit_status() == 0:
                return
            self.pump(0.01)
            for relay in self.admission.relays:
                if relay.endpoints and relay.endpoints[0].eof:
                    deadline = disconnected.setdefault(relay.role, time.monotonic() + TERM_GRACE)
                    child = client if relay.role == "human" else bot
                    if time.monotonic() >= deadline and child.exit_status() is None:
                        raise RuntimeError(f"{relay.role} disconnected but process did not exit; see {child.name}.log")
                    if relay.role == "human":
                        continue
                if not relay.welcomed and relay.endpoints and any(peer.eof for peer in relay.endpoints):
                    raise RuntimeError(f"{relay.role} disconnected before verified Welcome")
            if not self.admission.welcomed and any(child.exit_status() == 0 for child in (server, bot)):
                raise RuntimeError("server or bot exited before verified Welcome; see server.log and bot.log")

    def signal_groups(self, number):
        errors = []
        for child in self.children:
            try:
                os.killpg(child.process.pid, number)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"{child.name}: {error}")
        return errors

    def close(self):
        errors = []
        if self.admission is not None:
            try:
                self.admission.close()
            except OSError as error:
                errors.append(f"closing admission relays: {error}")
            self.admission = None
        errors.extend(self.signal_groups(signal.SIGTERM))
        try:
            deadline = time.monotonic() + TERM_GRACE
            for _ in range(40):
                if all(child.exit_status() is not None for child in self.children):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                select.select([], [], [], min(0.05, remaining))
        finally:
            errors.extend(self.signal_groups(signal.SIGKILL))
            for child in self.children:
                try:
                    child.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    errors.append(f"{child.name} did not exit after SIGKILL")
            try:
                for _ in range(MAX_CHILDREN * 2 * (LOG_LIMIT // READ_CHUNK + 1)):
                    if not self.pump(0, final=True):
                        break
            except (OSError, RuntimeError) as error:
                errors.append(f"draining logs: {error}")
            finally:
                self.selector.close()
                for child in self.children:
                    for stream in (child.process.stdout, child.process.stderr, child.log):
                        try:
                            stream.close()
                        except OSError as error:
                            errors.append(f"closing {child.name} log/pipe: {error}")
                self.children.clear()
        if errors:
            raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    sys.exit(main())
