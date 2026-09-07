"""Linux local-match supervisor used by the root play.sh launcher."""

import argparse
from dataclasses import dataclass, field
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


BUILD_TIMEOUT = 1200
LOG_LIMIT = 16 * 1024 * 1024
MAX_CHILDREN = 5
READ_CHUNK = 65536
READINESS_LIMIT = 4096
READINESS_TIMEOUT = 10
REPLAY_LIMIT = 512 * 1024 * 1024
TERM_GRACE = 2
assert READINESS_LIMIT < READ_CHUNK
assert READ_CHUNK <= LOG_LIMIT
assert LOG_LIMIT < REPLAY_LIMIT
assert 0 < TERM_GRACE <= 3


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="play.sh", description="Build and play bota against drysua.")
    parser.add_argument("--port", type=lambda value: integer(value, 65535), default=4455,
                        help="Server port, or 0 for an assigned port (default: 4455)")
    parser.add_argument("--seed", type=lambda value: integer(value, 2**64 - 1), default=9000001,
                        help="Match seed (default: 9000001)")
    parser.add_argument("--no-build", action="store_true", help="Use existing per-repository release binaries")
    arguments = parser.parse_args(arguments)
    root = Path(__file__).resolve().parents[2]
    supervisor, previous, mask = None, {}, None
    status = 1
    try:
        preflight(root, arguments.no_build)
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
        if not arguments.no_build:
            for repository in ("bota", "drysua"):
                command = ["cargo", "build", "--release", "--locked", "--quiet"]
                if repository == "drysua":
                    command.append("--no-default-features")
                command += ["--manifest-path", str(root / repository / "Cargo.toml")]
                command += (["-p", "bota-server", "-p", "bota-client", "--bin", "bota-server",
                             "--bin", "bota-client"] if repository == "bota" else ["--bin", "drysua"])
                environment = dict(os.environ, CARGO_TARGET_DIR=str(root / repository / "target"))
                self.wait_build(self.spawn("build-" + repository, command, root, environment))
        binaries = [root / "bota/target/release/bota-server", root / "bota/target/release/bota-client",
                    root / "drysua/target/release/drysua"]
        for binary in binaries:
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(f"release executable missing: {binary}; rerun without --no-build")
        server = self.spawn("server", [str(binaries[0]), "--port", str(arguments.port), "--mode", "realtime",
                            "--players", "2", "--map", "0", "--seed", str(arguments.seed),
                            "--replay", str(self.directory / "match.brp")], root)
        resource.prlimit(server.process.pid, resource.RLIMIT_FSIZE, (REPLAY_LIMIT, REPLAY_LIMIT))
        port = self.wait_ready(server, arguments.port)
        client = self.spawn("client", [str(binaries[1]), "--addr", f"127.0.0.1:{port}", "--name", "human"], root)
        bot = self.spawn("bot", [str(binaries[2]), "--addr", f"127.0.0.1:{port}", "--name", "drysua"], root)
        print("play: choose a hero (1/2/3), then R to ready; sides follow connection order. Ctrl+C stops all.",
              flush=True)
        # The results window controls lifetime; successful server/bot exits do not close it.
        while True:
            self.check_stop()
            self.pump(0.1)
            for child in (server, bot, client):
                status = child.exit_status()
                if status not in (None, 0):
                    raise RuntimeError(f"{child.name} exited with status {status}; see {child.name}.log")
            if client.exit_status() == 0:
                return

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
        errors = self.signal_groups(signal.SIGTERM)
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
