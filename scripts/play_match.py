"""Linux local-match supervisor used by the scripts/play.sh launcher."""

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

from play_admission import Admission
from play_reward import start_observers, pump_observers, finish_observers, print_interval
from play_pacing import ACK_TIMEOUT_TICKS


BUILD_TIMEOUT = 1200
RULE_POLICIES = ("teacher", "harass-push")
LOG_LIMIT = 16 * 1024 * 1024
MAX_CHILDREN = 8
READ_CHUNK = 65536
RUNTIME_FILE = "drysua.weights.safetensors"
READINESS_LIMIT = 4096
READINESS_TIMEOUT = 10
REPLAY_LIMIT = 2 * 1024**3
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
        _, arguments.weights_directory = opponent_paths(root, arguments)
        preflight(root, arguments.no_build)
        if arguments.no_build:
            release_executables(root, require_fresh=True)
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
        if supervisor is not None and not isinstance(error, InterruptedError):
            for pipe, _, _ in supervisor.reward_pipes:
                try:
                    pipe.fail(f"launcher interrupted observation: {error}")
                except OSError as report_error:
                    print(f"play: cannot save invalid observation marker: {report_error}", file=sys.stderr)
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
        prog="play.sh", description="Map2 human play against a weightless rule policy (Teacher, HarassPush) "
                                    "or pure Neural with compatible explicit weights.")
    parser.add_argument("--opponent", choices=("neural",) + RULE_POLICIES, default="neural")
    parser.add_argument("--watch", choices=RULE_POLICIES,
                        help="Replace the human with this rule policy and open bota-client as a spectator")
    parser.add_argument("--reward-report", action="store_true",
                        help="Score both original streams using paced native Lockstep at 30 Hz; slow clients slow simulation")
    parser.add_argument("--reward-interval", type=lambda value: reward_interval(value), default=300,
                        help="Reward timeline interval in ticks, 30..27900 (default: 300)")
    parser.add_argument("--port", type=lambda value: integer(value, 65535), default=4455,
                        help="Server port, or 0 for an assigned port (default: 4455)")
    parser.add_argument("--seed", type=lambda value: integer(value, 2**64 - 1), default=9000001,
                        help="Match seed (default: 9000001)")
    parser.add_argument("--no-build", action="store_true",
                        help="Use existing release binaries, refusing stale bota ones; never build "
                             "(default: incremental release builds of both workspaces)")
    parser.add_argument("--human-side", choices=("radiant", "dire"),
                        help="Human side (default: radiant, or opposite --bot-side)")
    parser.add_argument("--bot-side", choices=("radiant", "dire"),
                        help="Opponent side (default: dire, or opposite --human-side)")
    parser.add_argument("--weights-directory", type=Path,
                        help="Runtime weights exported by train-annealed; required for Neural, forbidden for Teacher")
    result = parser.parse_args(arguments)
    if result.opponent in RULE_POLICIES and result.weights_directory is not None:
        parser.error(f"--opponent {result.opponent} forbids --weights-directory")
    opposite = {"radiant": "dire", "dire": "radiant"}
    if result.human_side is None:
        result.human_side = opposite[result.bot_side] if result.bot_side else "radiant"
    if result.bot_side is None:
        result.bot_side = opposite[result.human_side]
    if result.human_side == result.bot_side:
        parser.error("--human-side and --bot-side must be opposite")
    return result


def reward_interval(value):
    result = integer(value, 27900)
    if result < 30:
        raise argparse.ArgumentTypeError("reward interval must be in 30..27900 ticks")
    return result


def opponent_paths(root, arguments):
    if arguments.opponent in RULE_POLICIES:
        if arguments.weights_directory is not None:
            raise RuntimeError(f"--opponent {arguments.opponent} forbids --weights-directory")
        return root / "drysua/target/release/drysua", None
    return current_paths(root, arguments.weights_directory)


def bot_command(binary, address, policy, weights, name="drysua"):
    command = [str(binary), "--addr", address, "--name", name, "--policy", policy]
    if policy == "neural":
        assert weights is not None
        command.extend(["--weights-directory", str(weights)])
    else:
        assert policy in RULE_POLICIES and weights is None
    return command


def transport_arguments(arguments):
    if arguments.reward_report:
        return ["--mode", "lockstep", "--ack-timeout-ticks", str(ACK_TIMEOUT_TICKS)]
    return ["--mode", "realtime"]


def current_paths(root, weights_directory):
    if weights_directory is None:
        raise RuntimeError("Neural play requires --weights-directory with runtime weights exported by "
                           "train-annealed; no default model and no Teacher fallback")
    weights = weights_directory.resolve()
    runtime = weights / RUNTIME_FILE
    if runtime.is_symlink() or not runtime.is_file():
        # The Rust loader validates the tensor contract; this only fails before any build or child.
        raise RuntimeError(f"runtime weights preflight failed: {runtime} is not a regular file")
    return root / "drysua/target/release/drysua", weights


def release_paths(root):
    binaries = [root / "bota/target/release/bota-server", root / "bota/target/release/bota-client",
                root / "drysua/target/release/drysua"]
    assert len(binaries) == 3
    assert all(executable.is_absolute() for executable in binaries)
    return binaries


def executable_ready(executable):
    return not executable.is_symlink() and executable.is_file() and os.access(executable, os.X_OK)


def release_executables(root, require_fresh=False):
    binaries = release_paths(root)
    for executable in binaries:
        if not executable_ready(executable):
            raise RuntimeError(f"release executable missing: {executable}; rerun without --no-build")
    if require_fresh:
        reject_stale_bota(root, binaries[:2])
    return binaries


def reject_stale_bota(root, executables):
    # Binaries older than the bota HEAD commit predate its wire protocol; without git there is nothing to compare.
    try:
        result = subprocess.run(["git", "-C", str(root / "bota"), "log", "-1", "--format=%ct"],
                                capture_output=True, text=True, timeout=10, check=True)
        committed = int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return
    for executable in executables:
        if executable.stat().st_mtime < committed:
            raise RuntimeError(f"stale bota binary (older than the bota HEAD commit): {executable}; "
                               "drop --no-build to rebuild so bota and drysua match")


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


@dataclass(eq=False)
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
        self.reward_pipes = []
        self.reward_overview = False

    def request_stop(self, number, _frame):
        # A handler must not interrupt Popen before the new child has been registered.
        if not self.stop_status:
            self.stop_status = 128 + number

    def check_stop(self):
        if self.stop_status:
            raise InterruptedError("shutdown requested")

    def spawn(self, name, command, directory, environment=None, input_stream=None):
        self.check_stop()
        assert len(self.children) < MAX_CHILDREN
        assert not any(child.name == name for child in self.children)
        log = (self.directory / f"{name}.log").open("xb", buffering=0)
        try:
            process = subprocess.Popen(command, cwd=directory, env=environment,
                                       stdin=input_stream if input_stream is not None else subprocess.DEVNULL,
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
        if self.admission is not None and not final and self.admission.pacer is not None:
            timeout = self.admission.pacer.wait_timeout(timeout)
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
            if child.name.startswith("reward-") and stdout:
                print_interval(child, data)
            if child.name == "server" and stdout and child.port is None:
                child.banner.extend(data[:READINESS_LIMIT + 1 - len(child.banner)])
                child.port = ready_port(child.banner, self.requested_port)
        if self.admission is not None and not final:
            self.admission.pump()
            pump_observers(self)
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
        binary, weights = opponent_paths(root, arguments)
        label = "pure Neural" if arguments.opponent == "neural" else f"explicit {arguments.opponent} (no model)"
        print(f"play: current Map2 {label}; weights: {weights}; executable: {binary}", flush=True)
        if not arguments.no_build:
            command = ["cargo", "build", "--release", "--locked", "--quiet",
                       "--manifest-path", str(root / "bota/Cargo.toml"), "-p", "bota-server",
                       "-p", "bota-client", "--bin", "bota-server", "--bin", "bota-client"]
            environment = dict(os.environ, CARGO_TARGET_DIR=str(root / "bota/target"))
            self.wait_build(self.spawn("build-bota", command, root, environment))
            command = ["cargo", "build", "--release", "--locked", "--quiet",
                       "--bin", "drysua", "--no-default-features"]
            environment = dict(os.environ, CARGO_TARGET_DIR=str(root / "drysua/target"))
            self.wait_build(self.spawn("build-drysua", command, root / "drysua", environment))
        binaries = release_executables(root)
        assert binary == binaries[2]
        server = self.spawn("server", [str(binaries[0]), "--port", str(arguments.port), *transport_arguments(arguments),
                            "--players", "2", "--map", "2", "--seed", str(arguments.seed),
                            "--replay", str(self.directory / "match.brp")], root)
        resource.prlimit(server.process.pid, resource.RLIMIT_FSIZE, (REPLAY_LIMIT, REPLAY_LIMIT))
        port = self.wait_ready(server, arguments.port)
        self.admission = Admission(port, arguments.human_side, mode=int(arguments.reward_report),
                                   paced=arguments.reward_report, watch=arguments.watch is not None)
        if arguments.reward_report:
            print("play: reward-report uses native LOCKSTEP paced at 30 ticks/s by delaying original ACKs; "
                  "slow clients slow simulation, not snapshot delivery. No generated ACKs or orders.", flush=True)
            start_observers(self, binary, root, arguments)
        if arguments.watch is None:
            client = self.spawn("client", [str(binaries[1]), "--addr", self.admission.addresses["human"],
                                          "--name", "human"], root)
            seat = client
        else:
            seat = self.spawn("watched", bot_command(binary, self.admission.addresses["watched"],
                                                     arguments.watch, None, name="watched"), root)
            client = self.spawn("client", [str(binaries[1]), "--addr", f"127.0.0.1:{port}", "--spectate",
                                          "--name", "spectator"], root)
        bot = self.spawn("bot", bot_command(binary, self.admission.addresses["bot"], arguments.opponent,
                                            weights), root)
        if arguments.watch is None:
            print(f"play: requested human {arguments.human_side} / {arguments.opponent} bot {arguments.bot_side}; "
                  "verifying server Welcome seats. Choose a hero (1/2/3), then R to ready. Ctrl+C stops all.",
                  flush=True)
        else:
            print(f"play: watching {arguments.watch} {arguments.human_side} / {arguments.opponent} "
                  f"{arguments.bot_side} from a spectator client. Ctrl+C stops all.", flush=True)
        self.wait_game(server, bot, client, seat)

    def wait_game(self, server, bot, client, seat):
        # The results window controls lifetime; successful server/bot exits do not close it.
        disconnected = {}
        while True:
            self.check_stop()
            for child in dict.fromkeys((server, bot, client, seat)):
                status = child.exit_status()
                if status not in (None, 0):
                    raise RuntimeError(f"{child.name} exited with status {status}; see {child.name}.log")
            if client.exit_status() == 0:
                return
            self.pump(0.01)
            for relay in self.admission.relays:
                if relay.endpoints and relay.endpoints[0].eof:
                    deadline = disconnected.setdefault(relay.role, time.monotonic() + TERM_GRACE)
                    child = bot if relay.role == "bot" else seat
                    if time.monotonic() >= deadline and child.exit_status() is None:
                        raise RuntimeError(f"{relay.role} disconnected but process did not exit; see {child.name}.log")
                    if relay.role != "bot":
                        continue
                if not relay.welcomed and relay.endpoints and any(peer.eof for peer in relay.endpoints):
                    raise RuntimeError(f"{relay.role} disconnected before verified Welcome")
            if not self.admission.welcomed and any(child.exit_status() == 0 for child in (server, bot, seat)
                                                   if child is not client):
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
        try:
            finish_observers(self)
        except Exception as error:
            errors.append(f"finalizing reward observation: {error}")
        finally:
            for pipe, _, _ in self.reward_pipes:
                pipe.close()
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
