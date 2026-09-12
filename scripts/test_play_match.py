"""Local launcher regression tests; fixtures stay below artifacts/temp."""

import ctypes
import contextlib
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import resource
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from release_crossplay import outcome_summary, read_client_output


ROOT = Path(__file__).resolve().parents[2]
TEMPORARY = ROOT / "drysua/artifacts/temp"
REVIEW = Path("drysua/artifacts/temp/human-review-20260909/current")
WEIGHTS_SHA = "6348fe57a128ebd521dba68da0949ceb7378d3e0d6b7d6a7ce9a5fb6446547ab"
BINARY_SHA = "64ba25ebb10e6beabc26ff667a3e3bddeb40dbf391110d4f0e31db6478500a2b"
CURRENT_METADATA = {
    "action_schema_hash": "10658390830565586343",
    "feature_schema_hash": "1861607613534772372",
    "model_schema_hash": "13592057279889489276",
    "ppo_schema_version": "30",
    "ppo_schema_hash": "16275284022255703821",
    "ppo_rules_audit_version": "25",
    "map2_reward_schema_version": "1",
    "map2_reward_schema_hash": "798798703797057220",
}


def rust_descriptor(module, name):
    source = (ROOT / f"drysua/src/{module}.rs").read_text()
    match = re.search(rf"pub const {name}: &str = concat!\((.*?)\n\);", source, re.S)
    assert match is not None, name
    strings = re.findall(r'"(?:[^"\\]|\\.)*"', match[1])
    assert strings, name
    return "".join(json.loads(value) for value in strings)


def metadata_fixture():
    return dict(CURRENT_METADATA, map2_reward_schema_descriptor=rust_descriptor(
        "map2_reward", "MAP2_REWARD_SCHEMA_DESCRIPTOR"))


def header_fixture(metadata):
    # Header-only test data is not a model; only mock executables consume these fixtures.
    header = json.dumps({"__metadata__": metadata}).encode()
    return struct.pack("<Q", len(header)) + header


def native_cap_transport_error(error):
    cause = error.__cause__
    if isinstance(cause, (BrokenPipeError, ConnectionResetError)):
        return True
    return (isinstance(cause, ValueError)
            and str(cause) == "connection reset before verified MatchOver"
            and isinstance(cause.__cause__, ConnectionResetError))


FIXTURE = r'''
import json, os, select, signal, socket, struct, subprocess, sys
from pathlib import Path

role = Path(sys.argv[0]).name
depth = 0
if sys.argv[1:2] == ["descendant"]:
    role, depth = sys.argv[2], int(sys.argv[3])
    if depth == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
elif role == "cargo":
    role = "build-" + Path(os.environ["CARGO_TARGET_DIR"]).parent.name
control = socket.socket(socket.AF_UNIX)
control.settimeout(20)
control.connect("\0" + os.environ["PLAY_TEST_CONTROL"])
def report(**values):
    control.sendall(json.dumps(values).encode() + b"\n")
def descendant(name, depth):
    subprocess.Popen([sys.executable, "-B", __file__, "descendant", name, str(depth)])
report(role=role, pid=os.getpid(), arguments=sys.argv[1:],
       target=os.environ.get("CARGO_TARGET_DIR"), cwd=os.getcwd(), executable=sys.argv[0])
if depth:
    descendant(role + "-grandchild", 0)
if role.startswith("build-") and os.environ.get("PLAY_TEST_BUILD") not in ("hold", role):
    sys.exit(0)
listener = None
prefix = False
peers, buffers, seats = [], {}, {}
read_closed = set()
held = None
def frame(payload):
    return struct.pack("<I", len(payload)) + payload
def hello(peer):
    name = b"human" if role == "bota-client" else b"drysua"
    peer.sendall(frame(bytes([0, 0 if role == "bota-client" else 1, len(name)]) + name))
if role in ("bota-client", "drysua"):
    host, port = sys.argv[sys.argv.index("--addr") + 1].rsplit(":", 1)
    peer = socket.create_connection((host, int(port)), 5)
    peers.append(peer)
    buffers[peer] = bytearray()
    if os.environ.get("PLAY_TEST_HOLD_HELLO") != role:
        hello(peer)
    report(connected=True)
for _ in range(4096):
    readers = [control] + [peer for peer in peers if peer not in read_closed] + ([listener] if listener else [])
    readable, _, _ = select.select(readers, [], [], 20)
    if not readable:
        raise RuntimeError("fixture control deadline exceeded")
    if listener in readable:
        peer, _ = listener.accept()
        if len(peers) >= 2:
            raise RuntimeError("unexpected TCP connection beyond the two participants")
        peers.append(peer)
        buffers[peer] = bytearray()
    for peer in tuple(peers):
        if peer not in readable:
            continue
        data = peer.recv(65536)
        if not data:
            if role == "bota-server" and peer not in seats:
                raise RuntimeError("unexpected TCP readiness probe")
            if role == "bota-client":
                read_closed.add(peer)
                report(server_eof=True)
            else:
                peers.remove(peer)
                peer.close()
            continue
        buffers[peer].extend(data)
        for _ in range(13108):
            buffer = buffers[peer]
            if len(buffer) < 4:
                break
            length = struct.unpack_from("<I", buffer)[0]
            assert 0 < length <= 4 * 1024 * 1024
            if len(buffer) < length + 4:
                break
            payload = bytes(buffer[4:4 + length])
            del buffer[:4 + length]
            if role == "bota-server" and payload[0] == 0:
                slot = len(seats)
                seats[peer] = slot
                report(hello=payload[3:].decode(), slot=slot)
                actual = int(os.environ.get("PLAY_TEST_WRONG_SLOT", slot))
                if slot == 1:
                    actual = int(os.environ.get("PLAY_TEST_WRONG_SECOND_SLOT", actual))
                welcome = frame(bytes([0, slot + 1, 1, actual, 30, 0]))
                if os.environ.get("PLAY_TEST_HOLD_WELCOME") and slot == 0:
                    held = peer, welcome
                else:
                    peer.sendall(welcome)
            elif role != "bota-server" and payload[0] == 0:
                report(slot=payload[3])
                peer.sendall(frame(b"\x01\x02") + frame(b"\x02\x01"))
            elif role == "bota-client" and payload[0] == 7:
                report(match_over=True)
    if control not in readable:
        continue
    command = control.recv(1)
    if command in (b"", b"0"):
        sys.exit(0)
    if command in (b"r", b"R"):
        if listener is None:
            listener = socket.socket()
            listener.bind(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])))
            listener.listen(2)
        port = listener.getsockname()[1]
        if command == b"r":
            os.write(1, b"bota-server listening on ")
            prefix = True
        else:
            os.write(1, ("" if prefix else "bota-server listening on ").encode()
                     + f"0.0.0.0:{port}\n".encode())
        report(port=port)
    elif command == b"7":
        print(role + " fixture failure", file=sys.stderr, flush=True)
        sys.exit(7)
    elif command == b"e":
        print("bota-client: connection refused", file=sys.stderr, flush=True)
        sys.exit(0)
    elif command == b"s":
        print("bota-server listening on 0.0.0.0:12345", file=sys.stderr, flush=True)
        sys.exit(7)
    elif command == b"d":
        descendant(role + "-child", 1)
    elif command == b"x":
        os.write(1, b"x" * 4097)
    elif command == b"f":
        for _ in range(257):
            os.write(1, b"x" * 65536)
    elif command == b"i":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        report(ignoring=True)
    elif command == b"q":
        os.close(1)
    elif command == b"h":
        hello(peers[0])
    elif command == b"?":
        report(admitted=len(seats))
    elif command == b"w":
        peer, welcome = held
        peer.sendall(welcome[:-1])
        held = peer, welcome[-1:]
    elif command == b"W":
        peer, welcome = held
        peer.sendall(welcome)
    elif command == b"m":
        for peer in peers:
            peer.sendall(frame(bytes([7, 0, 1, 2, 0] + [0] * 8 + [1] + [0] * 8)))
        report(terminal=True)
    elif command == b"o":
        peers[0].sendall(frame(b"\x03\x01\x00\x00\x00"))
        report(late_order=True)
else:
    raise RuntimeError("fixture command limit exceeded")
'''


NATIVE_BINARY = Path(os.environ.get("PLAY_TEST_REVIEW_BINARY", ROOT / REVIEW / "drysua"))
NATIVE_WEIGHTS = Path(os.environ.get("PLAY_TEST_REVIEW_WEIGHTS", ROOT / REVIEW / "weights"))
HISTORICAL_BASELINE = ROOT / "drysua/artifacts/temp/map0-baseline-observationfix-4096"
HISTORICAL_SERVER_SHA = "24a8efccb285308810678c7e3a8717b57814c9d8fecef386ca923cdb7c04e97c"
NATIVE_CLIENT = r'''
import socket, struct, sys
sys.path.insert(0, sys.argv[5])
from release_wire import varint
address, slot, mode, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def integer(value):
    output = bytearray()
    for _ in range(10):
        output.append((value & 127) | (128 if value > 127 else 0))
        value >>= 7
        if not value:
            return bytes(output)
    raise ValueError("integer overflow")
def send(payload):
    connection.sendall(struct.pack("<I", len(payload)) + payload)
def receive(count):
    output = bytearray()
    for _ in range(count + 1):
        if len(output) == count:
            return bytes(output)
        chunk = connection.recv(min(65536, count - len(output)))
        if not chunk:
            raise RuntimeError("headless client unexpected EOF")
        output.extend(chunk)
    raise RuntimeError("headless read bound exceeded")
host, port = address.rsplit(":", 1)
with socket.create_connection((host, int(port)), 5) as connection:
    connection.settimeout(10)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send(b"\x00\x00\x05human")
    welcomed, snapshot = False, 0
    for _ in range(limit * 10 + 1000):
        length = struct.unpack("<I", receive(4))[0]
        assert 0 < length <= 4 * 1024 * 1024
        payload = receive(length)
        kind, offset = varint(payload, 0)
        if kind == 0:
            fields = []
            for _ in range(5):
                value, offset = varint(payload, offset)
                fields.append(value)
            assert not welcomed
            assert fields[1:] == [1, slot, 30, mode], fields
            assert offset == len(payload)
            welcomed = True
            print(f"headless human Welcome slot {slot}", flush=True)
            send(b"\x01\x02")
            send(b"\x02\x01")
        elif kind == 3:
            snapshot, offset = varint(payload, offset)
            present, offset = varint(payload, offset)
            viewer, _ = varint(payload, offset)
            assert welcomed
            assert present == 1
            assert viewer == slot, (viewer, slot)
        elif kind == 4:
            tick, _ = varint(payload, offset)
            assert tick == snapshot
            if tick >= limit:
                print(f"headless human snapshot team {slot}; completed {limit} ticks", flush=True)
                break
            if mode == 1:
                send(b"\x04" + integer(tick))
        elif kind == 5:
            raise RuntimeError("headless client order rejected")
    else:
        raise RuntimeError("headless client frame bound exceeded")
'''


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMPORARY.mkdir(parents=True, exist_ok=True)
        # Adopt killed grandchildren so tests never leave zombies with PID 1.
        cls.libc = ctypes.CDLL(None, use_errno=True)
        cls.previous_subreaper = ctypes.c_int()
        assert cls.libc.prctl(37, ctypes.byref(cls.previous_subreaper), 0, 0, 0) == 0
        assert cls.libc.prctl(36, 1, 0, 0, 0) == 0

    @classmethod
    def tearDownClass(cls):
        assert cls.libc.prctl(36, cls.previous_subreaper.value, 0, 0, 0) == 0

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="play-test-", dir=TEMPORARY)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "workspace with spaces"
        self.root.mkdir()
        for repository in ("bota", "drysua"):
            directory = self.root / repository
            (directory / "target/release").mkdir(parents=True)
            (directory / "Cargo.toml").write_text("[workspace]\n")
        (self.root / "drysua/scripts").mkdir()
        shutil.copy(ROOT / "play.sh", self.root / "play.sh")
        shutil.copy(ROOT / "drysua/scripts/play_match.py", self.root / "drysua/scripts")
        for name in ("play_admission.py", "play_weights.py", "release_wire.py"):
            source = ROOT / "drysua/scripts" / name
            if source.exists():
                shutil.copy(source, self.root / "drysua/scripts")
        self.tools = self.root / "tools"
        self.tools.mkdir()
        for binary in (self.tools / "cargo", self.root / "bota/target/release/bota-server",
                       self.root / "bota/target/release/bota-client",
                       self.root / "drysua/target/release/drysua"):
            binary.write_text(f"#!{sys.executable} -B\n" + FIXTURE)
            binary.chmod(0o700)
        review = self.root / REVIEW
        (review / "weights").mkdir(parents=True)
        shutil.copy(self.root / "drysua/target/release/drysua", review / "drysua")
        (review / "weights/drysua.weights.safetensors").write_bytes(b"fixture weights")
        (review / "manifest.json").write_text('{"purpose": "human-review fixture"}\n')
        launcher = self.root / "drysua/scripts/play_match.py"
        launcher.write_text(launcher.read_text().replace(
            BINARY_SHA, hashlib.sha256((review / "drysua").read_bytes()).hexdigest()).replace(
            WEIGHTS_SHA, hashlib.sha256(b"fixture weights").hexdigest()))
        self.weights = self.root / "current metadata fixture"
        self.weights.mkdir()
        (self.weights / "drysua.weights.safetensors").write_bytes(header_fixture(metadata_fixture()))
        self.listener = socket.socket(socket.AF_UNIX)
        self.addCleanup(self.listener.close)
        address = "play-test-" + Path(self.temporary.name).name
        self.listener.bind("\0" + address)
        self.listener.listen(16)
        self.listener.settimeout(5)
        self.environment = dict(os.environ, DISPLAY=":fixture", PLAY_TEST_CONTROL=address,
                                PATH=str(self.tools) + os.pathsep + os.environ["PATH"],
                                CARGO_TARGET_DIR=str(self.root / "wrong inherited target"))
        self.children = {}
        self.process = None
        self.addCleanup(self.clean_processes)

    def launch(self, *arguments, weights=True, **environment):
        if self.process is not None:
            self.clean_processes()
        self.environment.update(environment)
        if weights and "--weights-directory" not in arguments:
            arguments = ("--weights-directory", str(self.weights), *arguments)
        self.process = subprocess.Popen([str(self.root / "play.sh"), *arguments],
                                        cwd=self.temporary.name, env=self.environment,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=True)

    def child(self, name):
        for _ in range(12):
            if name in self.children:
                return self.children[name]
            connection, _ = self.listener.accept()
            connection.settimeout(5)
            stream = connection.makefile("rb")
            record = json.loads(stream.readline(8192))
            record.update(connection=connection, stream=stream,
                          pidfd=os.pidfd_open(record["pid"]))
            self.children[record["role"]] = record
        self.fail("fixture connection limit exceeded")

    def send(self, name, command):
        self.child(name)["connection"].sendall(command)

    def record(self, name):
        return json.loads(self.child(name)["stream"].readline(8192))

    def game(self, *arguments):
        self.launch("--port", "0", *arguments)
        self.send("bota-server", b"r")
        first = json.loads(self.child("bota-server")["stream"].readline(8192))
        self.send("bota-server", b"R")
        self.assertEqual(json.loads(self.child("bota-server")["stream"].readline(8192)), first)
        for name in ("bota-client", "drysua"):
            self.assertEqual(self.record(name), {"connected": True})
        for _ in range(2):
            self.assertIn("hello", self.record("bota-server"))
        for name in ("bota-client", "drysua"):
            self.child(name)["slot"] = self.record(name)["slot"]
        return first["port"]

    def finish(self, status):
        output, error = self.process.communicate(timeout=8)
        self.assertEqual(self.process.returncode, status, error.decode())
        if status:
            self.assertIn(b"logs:", error + output)
        return (output + error).decode()

    def assert_children_stopped(self):
        for child in self.children.values():
            self.assertTrue(select.select([child["pidfd"]], [], [], 3)[0], child["role"])

    def clean_processes(self):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate(timeout=5)
        for child in self.children.values():
            if not select.select([child["pidfd"]], [], [], 0)[0]:
                signal.pidfd_send_signal(child["pidfd"], signal.SIGKILL)
            select.select([child["pidfd"]], [], [], 3)
            try:
                os.waitpid(child["pid"], os.WNOHANG)
            except ChildProcessError:
                pass
            os.close(child["pidfd"])
            child["stream"].close()
            child["connection"].close()
        self.children.clear()

    def test_headless_fails_before_building_or_creating_logs(self):
        self.launch(DISPLAY="")
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 1, output.decode())
        self.assertIn(b"DISPLAY", error)
        self.assertFalse(list((self.root / "drysua/artifacts/temp").glob("play-*")))

    def test_help_is_available_without_display(self):
        self.launch("--help", DISPLAY="")
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 0, error.decode())
        self.assertIn(b"--no-build", output)

    def test_explicit_weights_build_current_cpu_and_bota_and_launch_map2_pure_neural(self):
        port = self.game()
        build = self.child("build-bota")
        self.assertEqual(build["target"], str(self.root / "bota/target"))
        self.assertEqual(build["arguments"], ["build", "--release", "--locked", "--quiet",
                         "--manifest-path", str(self.root / "bota/Cargo.toml"), "-p", "bota-server",
                         "-p", "bota-client", "--bin", "bota-server", "--bin", "bota-client"])
        build = self.child("build-drysua")
        self.assertEqual(build["target"], str(self.root / "drysua/target"))
        self.assertEqual(build["cwd"], str(self.root / "drysua"))
        self.assertEqual(build["arguments"], ["build", "--release", "--locked", "--quiet",
                                            "--bin", "drysua", "--no-default-features"])
        self.assertEqual(len(self.children), 5)
        self.assertEqual(self.child("drysua")["executable"], str(self.root / "drysua/target/release/drysua"))
        bot = self.child("drysua")["arguments"]
        self.assertEqual(bot[2:], ["--name", "drysua", "--policy", "neural", "--weights-directory",
                                  str(self.weights)])
        self.assertEqual(self.child("bota-client")["arguments"][2:], ["--name", "human"])
        self.assertNotEqual(bot[1], f"127.0.0.1:{port}")
        self.assertNotEqual(bot[1], self.child("bota-client")["arguments"][1])
        self.assertEqual(self.child("bota-client")["slot"], 0)
        self.assertEqual(self.child("drysua")["slot"], 1)
        server = self.child("bota-server")["arguments"]
        self.assertEqual(server[:10], ["--port", "0", "--mode", "realtime", "--players", "2",
                                      "--map", "2", "--seed", "9000001"])
        self.assertEqual(Path(server[-1]).parent.parent, self.root / "drysua/artifacts/temp")
        self.assertEqual(resource.prlimit(self.child("bota-server")["pid"], resource.RLIMIT_FSIZE),
                         (2 * 1024**3, 2 * 1024**3))
        self.send("bota-client", b"0")
        self.finish(0)
        self.assert_children_stopped()

    def test_no_build_ignores_inherited_target_and_game_completion_keeps_results(self):
        self.game("--no-build")
        self.assertNotIn("build-bota", self.children)
        self.assertNotIn("build-drysua", self.children)
        self.send("bota-server", b"m")
        self.assertEqual(self.record("bota-server"), {"terminal": True})
        self.assertEqual(self.record("bota-client"), {"match_over": True})
        self.send("bota-server", b"0")
        self.send("drysua", b"0")
        for name in ("bota-server", "drysua"):
            self.assertTrue(select.select([self.child(name)["pidfd"]], [], [], 3)[0])
        self.assertEqual(self.record("bota-client"), {"server_eof": True})
        self.send("bota-client", b"o")
        self.assertEqual(self.record("bota-client"), {"late_order": True})
        self.assertIsNone(self.process.poll())
        self.send("bota-client", b"0")
        self.finish(0)

    def test_interrupt_during_build_kills_children_and_term_ignoring_grandchildren(self):
        self.launch(PLAY_TEST_BUILD="hold")
        self.send("build-bota", b"d")
        self.child("build-bota-child")
        self.child("build-bota-child-grandchild")
        self.process.send_signal(signal.SIGINT)
        self.finish(130)
        self.assert_children_stopped()

    def test_interrupt_during_game_kills_exited_bots_descendants_but_not_unowned_process(self):
        self.game("--no-build")
        unrelated = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(1)"],
                                     stdin=subprocess.PIPE, start_new_session=True)
        try:
            self.send("drysua", b"d")
            self.child("drysua-child")
            self.child("drysua-child-grandchild")
            self.send("drysua", b"0")
            self.assertTrue(select.select([self.child("drysua")["pidfd"]], [], [], 3)[0])
            self.process.send_signal(signal.SIGINT)
            self.finish(130)
            self.assert_children_stopped()
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.communicate(b"q", timeout=5)

    def test_term_during_readiness_stops_server(self):
        self.launch("--no-build")
        self.child("bota-server")
        self.process.send_signal(signal.SIGTERM)
        self.finish(143)
        self.assert_children_stopped()

    def test_term_escalates_when_build_ignores_term(self):
        self.launch(PLAY_TEST_BUILD="hold")
        self.send("build-bota", b"i")
        self.assertEqual(json.loads(self.child("build-bota")["stream"].readline(8192)),
                         {"ignoring": True})
        self.process.send_signal(signal.SIGTERM)
        self.finish(143)
        self.assert_children_stopped()

    def test_build_failure_stops_before_server_and_reports_log(self):
        self.launch(PLAY_TEST_BUILD="hold")
        self.send("build-bota", b"7")
        self.assertIn("build-bota exited", self.finish(1))
        self.assert_children_stopped()

    def test_current_bot_build_failure_never_starts_server_or_uses_review_binary(self):
        self.launch(PLAY_TEST_BUILD="build-drysua")
        self.send("build-drysua", b"7")
        self.assertIn("build-drysua exited with status 7", self.finish(1))
        self.assertNotIn("bota-server", self.children)
        self.assert_children_stopped()

    def test_stderr_banner_does_not_hide_server_startup_failure(self):
        self.launch("--no-build")
        self.send("bota-server", b"s")
        self.assertIn("server", self.finish(1))
        self.assert_children_stopped()

    def test_oversized_readiness_fails_promptly(self):
        self.launch("--no-build")
        self.send("bota-server", b"x")
        self.assertIn("4096", self.finish(1))
        self.assert_children_stopped()

    def test_stdout_eof_before_readiness_fails_even_when_server_stays_alive(self):
        self.launch("--no-build")
        self.send("bota-server", b"q")
        self.assertIn("stdout closed before readiness", self.finish(1))
        self.assert_children_stopped()

    def test_runtime_component_crashes_stop_the_match_and_retain_error_logs(self):
        for name in ("bota-client", "drysua", "bota-server"):
            with self.subTest(component=name):
                if self.process is not None:
                    self.clean_processes()
                    self.children.clear()
                self.game("--no-build")
                self.send(name, b"7")
                output = self.finish(1)
                if name == "bota-server":
                    self.assertRegex(output, "exited|server disconnected before verified MatchOver")
                else:
                    self.assertIn("exited", output)
                self.assert_children_stopped()
                logs = list((self.root / "drysua/artifacts/temp").glob("play-*/*.log"))
                self.assertTrue(any(b"fixture failure" in path.read_bytes() for path in logs))

    def test_client_reported_error_is_failure_even_with_zero_exit_status(self):
        self.game("--no-build")
        self.send("bota-client", b"e")
        self.assertIn("bota-client", self.finish(1))

    def test_log_flood_stops_match_without_exceeding_file_limit(self):
        self.game("--no-build")
        self.send("drysua", b"f")
        self.assertIn("log limit", self.finish(1))
        self.assert_children_stopped()
        for path in (self.root / "drysua/artifacts/temp").glob("play-*/*.log"):
            self.assertLessEqual(path.stat().st_size, 16 * 1024 * 1024)

    def test_missing_binary_in_no_build_mode_has_actionable_error(self):
        for relative in ("bota/target/release/bota-client", "bota/target/release/bota-server",
                         "drysua/target/release/drysua"):
            target = self.root / relative
            target.chmod(0o600)
            try:
                self.launch("--no-build")
                output = self.preflight_failure("release executable missing")
                self.assertIn(str(target), output)
                self.assertIn("rerun without --no-build", output)
            finally:
                target.chmod(0o700)

    def test_invalid_port_and_seed_are_rejected_before_preflight(self):
        for arguments in (("--port", "-1"), ("--port", "65536"),
                          ("--seed", str(2**64)), ("--seed", "not-a-seed")):
            with self.subTest(arguments=arguments):
                self.launch(*arguments, DISPLAY="")
                _, error = self.process.communicate(timeout=5)
                self.assertEqual(self.process.returncode, 2)
                self.assertIn(b"expected a decimal integer in 0..", error)

    def test_source_and_cargo_preflight_fail_before_artifacts(self):
        for target, message in ((self.tools / "cargo", b"cargo is required"),
                                (self.root / "bota/Cargo.toml", b"source manifest missing")):
            with self.subTest(target=target), patch("play_match.shutil.which", return_value=None):
                module = importlib.import_module("play_match")
                if target.name != "cargo":
                    target.unlink()
                with patch.dict(os.environ, DISPLAY=":fixture"):
                    with self.assertRaisesRegex(RuntimeError, message.decode()):
                        module.preflight(self.root, False)
        self.assertFalse(list((self.root / "drysua/artifacts/temp").glob("play-*")))

    def test_faster_bot_waits_for_human_complete_welcome_before_server_admission(self):
        self.launch("--no-build", "--port", "0", PLAY_TEST_HOLD_HELLO="bota-client",
                    PLAY_TEST_HOLD_WELCOME="1")
        self.send("bota-server", b"R")
        self.record("bota-server")
        for name in ("bota-client", "drysua"):
            self.assertEqual(self.record(name), {"connected": True})
        self.send("bota-server", b"?")
        self.assertEqual(self.record("bota-server"), {"admitted": 0})
        self.send("bota-client", b"h")
        self.assertEqual(self.record("bota-server"), {"hello": "human", "slot": 0})
        self.send("bota-server", b"w?")
        self.assertEqual(self.record("bota-server"), {"admitted": 1})
        self.send("bota-server", b"W")
        self.assertEqual(self.record("bota-client"), {"slot": 0})
        self.assertEqual(self.record("bota-server"), {"hello": "drysua", "slot": 1})
        self.assertEqual(self.record("drysua"), {"slot": 1})
        self.send("bota-client", b"0")
        self.assertIn("verified", self.finish(0))
        self.assert_children_stopped()

    def test_either_side_option_derives_the_opposite_and_both_explicit_sides_work(self):
        for arguments, human_slot in ((("--human-side", "dire"), 1), (("--bot-side", "radiant"), 1),
                                      (("--human-side", "radiant"), 0), (("--bot-side", "dire"), 0),
                                      (("--human-side", "dire", "--bot-side", "radiant"), 1),
                                      (("--human-side", "radiant", "--bot-side", "dire"), 0)):
            with self.subTest(arguments=arguments):
                if self.process is not None:
                    self.clean_processes()
                    self.children.clear()
                self.game("--no-build", *arguments)
                self.assertEqual(self.child("bota-client")["slot"], human_slot)
                self.assertEqual(self.child("drysua")["slot"], 1 - human_slot)
                self.send("bota-client", b"0")
                self.finish(0)
                self.assert_children_stopped()

    def test_same_sides_are_rejected_before_display_or_children(self):
        for side in ("radiant", "dire"):
            self.launch("--human-side", side, "--bot-side", side, DISPLAY="")
            _, error = self.process.communicate(timeout=5)
            self.assertEqual(self.process.returncode, 2)
            self.assertIn(b"must be opposite", error)

    def test_wrong_welcome_slot_fails_and_cleans_both_clients_and_relays(self):
        self.launch("--no-build", "--port", "0", PLAY_TEST_WRONG_SLOT="1")
        self.send("bota-server", b"R")
        self.record("bota-server")
        for name in ("bota-client", "drysua"):
            self.child(name)
        self.assertIn("expected slot 0", self.finish(1))
        self.assert_children_stopped()
        self.assert_relay_ports_closed()

    def test_second_welcome_is_also_checked_after_first_slot_is_verified(self):
        self.launch("--no-build", "--port", "0", PLAY_TEST_WRONG_SECOND_SLOT="0")
        self.send("bota-server", b"R")
        self.record("bota-server")
        for name in ("bota-client", "drysua"):
            self.child(name)
        output = self.finish(1)
        self.assertIn("verified human Radiant: Welcome slot 0", output)
        self.assertIn("expected slot 1, got 0", output)
        self.assert_children_stopped()
        self.assert_relay_ports_closed()

    def assert_relay_ports_closed(self):
        for name in ("bota-client", "drysua"):
            address = self.child(name)["arguments"][1]
            host, port = address.rsplit(":", 1)
            with socket.socket() as probe:
                probe.settimeout(1)
                self.assertNotEqual(probe.connect_ex((host, int(port))), 0, address)

    def test_term_during_hello_barrier_cleans_listeners_and_owned_descendants(self):
        self.launch("--no-build", "--port", "0", PLAY_TEST_HOLD_HELLO="bota-client")
        self.send("bota-server", b"R")
        self.record("bota-server")
        for name in ("bota-client", "drysua"):
            self.assertEqual(self.record(name), {"connected": True})
        self.send("drysua", b"d")
        self.child("drysua-child")
        self.child("drysua-child-grandchild")
        self.process.send_signal(signal.SIGTERM)
        self.finish(143)
        self.assert_children_stopped()
        self.assert_relay_ports_closed()

    def test_historical_review_utility_still_rejects_tampered_archive(self):
        module = importlib.import_module("play_match")
        review = self.root / REVIEW
        with patch.object(module, "REVIEW_BINARY_SHA", hashlib.sha256((review / "drysua").read_bytes()).hexdigest()), \
                patch.object(module, "REVIEW_WEIGHTS_SHA", hashlib.sha256(b"fixture weights").hexdigest()):
            self.check_tampered_archive(module)

    def check_tampered_archive(self, module):
        for relative in ("weights/drysua.weights.safetensors", "drysua", "manifest.json"):
            target = self.root / REVIEW / relative
            original = target.read_bytes()
            with self.subTest(file=relative):
                try:
                    target.write_bytes(b"corrupt")
                    with self.assertRaisesRegex(RuntimeError, "review"):
                        module.review_paths(self.root, None)
                finally:
                    target.write_bytes(original)

    def test_missing_review_copy_does_not_affect_explicit_current_selection(self):
        (self.root / REVIEW / "drysua").unlink()
        self.game("--no-build")
        self.send("bota-client", b"0")
        self.finish(0)

    def preflight_failure(self, message):
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 1, output.decode())
        self.assertIn(message, error.decode())
        self.assertNotIn(b"logs:", output + error)
        self.assertFalse(list((self.root / "drysua/artifacts/temp").glob("play-*")))
        self.assertFalse(select.select([self.listener], [], [], 0)[0], "unexpected child started")
        return error.decode()

    def test_no_arguments_fail_closed_before_display_build_logs_or_legacy_selection(self):
        self.launch(weights=False, DISPLAY="")
        error = self.preflight_failure("legacy F12/M14 human-review weights are incompatible")
        self.assertIn("--weights-directory", error)
        self.assertIn("F15/M17", error)
        self.assertIn("no Map2 model has been trained or promoted", error)

    def test_missing_weights_checked_before_missing_executables_and_display(self):
        (self.weights / "drysua.weights.safetensors").unlink()
        (self.root / "drysua/target/release/drysua").unlink()
        self.launch("--no-build", DISPLAY="")
        self.preflight_failure("runtime weights")

    def test_old_metadata_fails_before_build_without_rewriting_or_teacher_fallback(self):
        target = self.weights / "drysua.weights.safetensors"
        old = {"action_schema_hash": "1755359086494840931", "feature_schema_hash": "1577122233561586211",
               "model_schema_hash": "7970187849195607202", "ppo_schema_version": "27",
               "ppo_schema_hash": "9274275648898675046", "ppo_rules_audit_version": "22"}
        data = header_fixture(old)
        target.write_bytes(data)
        self.launch()
        self.preflight_failure("incompatible runtime weights metadata")
        self.assertEqual(target.read_bytes(), data)

    def test_client_closing_during_admission_cleans_queued_bot_and_listeners(self):
        self.launch("--no-build", "--port", "0", PLAY_TEST_HOLD_HELLO="bota-client")
        self.send("bota-server", b"R")
        self.record("bota-server")
        for name in ("bota-client", "drysua"):
            self.assertEqual(self.record(name), {"connected": True})
        self.send("bota-client", b"0")
        self.finish(0)
        self.assert_children_stopped()
        self.assert_relay_ports_closed()

    def test_relative_weights_select_current_binary_and_explicit_neural_policy(self):
        weights = self.root / "experimental weights"
        weights.mkdir()
        (weights / "drysua.weights.safetensors").write_bytes(header_fixture(metadata_fixture()))
        self.game("--no-build", "--weights-directory", str(weights.relative_to(self.temporary.name)))
        arguments = self.child("drysua")["arguments"]
        self.assertEqual(arguments[arguments.index("--policy") + 1], "neural")
        self.assertEqual(arguments[-1], str(weights))
        self.assertEqual(self.child("drysua")["executable"], str(self.root / "drysua/target/release/drysua"))
        self.send("bota-client", b"0")
        self.assertIn("current Map2", self.finish(0))


class RuntimeWeightsTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("play_weights")
        TEMPORARY.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="play-header-", dir=TEMPORARY)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "drysua.weights.safetensors"

    def test_current_metadata_is_accepted_without_claiming_tensor_validation(self):
        metadata = metadata_fixture()
        self.path.write_bytes(header_fixture(metadata))

        actual = self.module.read_runtime_metadata(self.directory)

        self.assertEqual(actual, metadata)
        self.assertEqual(self.module.CURRENT_METADATA, CURRENT_METADATA)

    def test_old_a4_f14_m16_nine_key_tuple_is_rejected_without_relabelling(self):
        metadata = dict(metadata_fixture(), action_schema_hash="281345351372519059",
                        feature_schema_hash="16612223928593971806",
                        model_schema_hash="16105106472474017042", ppo_schema_version="29",
                        ppo_schema_hash="6915425029811947603", ppo_rules_audit_version="24")
        original = header_fixture(metadata)
        self.path.write_bytes(original)

        with self.assertRaisesRegex(RuntimeError, "expected exact nine-key F15/M17, A5, PPO30/rules25"):
            self.module.read_runtime_metadata(self.directory)

        self.assertEqual(self.path.read_bytes(), original)

    def test_every_metadata_key_is_required_exactly_once_and_string_typed(self):
        metadata = metadata_fixture()
        for key in metadata:
            for replacement in (None, "wrong", 1, {}, True):
                changed = dict(metadata)
                del changed[key]
                if replacement is not None:
                    changed[key] = replacement
                self.path.write_bytes(header_fixture(changed))
                with self.subTest(key=key, replacement=replacement), \
                        self.assertRaisesRegex(RuntimeError, "incompatible runtime weights metadata.*F15/M17"):
                    self.module.read_runtime_metadata(self.directory)

    def test_extra_metadata_and_self_consistent_wrong_reward_hash_are_rejected(self):
        for metadata in (dict(metadata_fixture(), unexpected="extra"),
                         dict(metadata_fixture(), map2_reward_schema_descriptor="wrong",
                              map2_reward_schema_hash=str(self.module.fnv1a(b"wrong")))):
            self.path.write_bytes(header_fixture(metadata))
            with self.assertRaisesRegex(RuntimeError, "incompatible runtime weights metadata"):
                self.module.read_runtime_metadata(self.directory)

    def test_malformed_header_json_never_escapes_as_unbounded_or_ambiguous_input(self):
        valid = json.dumps({"__metadata__": metadata_fixture()}).encode()
        headers = (b"[]", b"{}", b"null", b"\xff", b"{", b" " + valid,
                   valid.replace(b'"30"', b"NaN"),
                   valid.replace(b'"30"', b'"30", "ppo_schema_version": "30"'),
                   valid[:-1] + b', "__metadata__": {}}',
                   b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}")
        for header in headers:
            self.path.write_bytes(struct.pack("<Q", len(header)) + header)
            with self.subTest(header=header[:40]), self.assertRaisesRegex(RuntimeError, "runtime weights"):
                self.module.read_runtime_metadata(self.directory)

    def test_truncated_and_out_of_bounds_headers_fail_before_body_read(self):
        for data in (b"", b"\x00" * 7, struct.pack("<Q", 0), struct.pack("<Q", 65537),
                     struct.pack("<Q", 2**64 - 1), struct.pack("<Q", 8) + b"{}"):
            self.path.write_bytes(data)
            with self.subTest(data=data), self.assertRaisesRegex(RuntimeError, "runtime weights"):
                self.module.read_runtime_metadata(self.directory)

    def test_exact_header_and_file_limits_are_accepted_but_one_extra_byte_is_not(self):
        header = json.dumps({"__metadata__": metadata_fixture()}).encode().ljust(65536, b" ")
        self.path.write_bytes(struct.pack("<Q", len(header)) + header)
        with self.path.open("r+b") as stream:
            stream.truncate(256 * 1024**2)

        self.assertEqual(self.module.read_runtime_metadata(self.directory), metadata_fixture())

        with self.path.open("r+b") as stream:
            stream.truncate(256 * 1024**2 + 1)
        with self.assertRaisesRegex(RuntimeError, "runtime weights.*268435456"):
            self.module.read_runtime_metadata(self.directory)

    def test_symlink_directory_fifo_and_missing_files_fail_without_blocking(self):
        target = self.directory / "target"
        target.write_bytes(header_fixture(metadata_fixture()))
        self.path.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "runtime weights"):
            self.module.read_runtime_metadata(self.directory)
        self.path.unlink()
        self.path.mkdir()
        with self.assertRaisesRegex(RuntimeError, "runtime weights"):
            self.module.read_runtime_metadata(self.directory)
        self.path.rmdir()
        os.mkfifo(self.path)
        with self.assertRaisesRegex(RuntimeError, "runtime weights"):
            self.module.read_runtime_metadata(self.directory)
        self.path.unlink()
        with self.assertRaisesRegex(RuntimeError, "runtime weights"):
            self.module.read_runtime_metadata(self.directory)

    def test_current_tuple_matches_rust_descriptors_linked_hashes_versions_and_nine_keys(self):
        identities = {}
        reward = rust_descriptor("map2_reward", "MAP2_REWARD_SCHEMA_DESCRIPTOR").encode()

        def independent_hash(data):
            value = 0xcbf29ce484222325
            for byte in data:
                value = ((value ^ byte) * 0x100000001b3) & (2**64 - 1)
            return value

        contracts = (("map2_reward", 1, ()), ("action", 5, ()),
                     ("feature", 15, ("action", "map2_reward")),
                     ("model", 17, ("action", "feature", "map2_reward")),
                     ("ppo", 30, ("action", "feature", "model", "map2_reward")))
        for name, version, links in contracts:
            source = (ROOT / f"drysua/src/{name}.rs").read_text()
            self.assertRegex(source, rf"pub const {name.upper()}_SCHEMA_VERSION: u32 = {version};")
            data = rust_descriptor(name, name.upper() + "_SCHEMA_DESCRIPTOR").encode()
            if links:
                data += b"".join(struct.pack("<IQ", *identities[link]) for link in links) + reward
            digest = independent_hash(data)
            self.assertEqual(str(digest), self.module.CURRENT_METADATA[name + "_schema_hash"], name)
            identities[name] = version, digest
        self.assertEqual(self.module.fnv1a(reward), identities["map2_reward"][1])
        source = (ROOT / "drysua/src/ppo.rs").read_text()
        self.assertIn("pub const PPO_RULES_AUDIT_VERSION: u32 = 25;", source)
        source = (ROOT / "drysua/src/checkpoint.rs").read_text()
        metadata = source.split("fn runtime_tensor_metadata()", 1)[1].split("\n}", 1)[0]
        self.assertEqual(set(re.findall(r'"([a-z0-9_]+)"', metadata)), set(metadata_fixture()))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("play_admission")

    def relay(self):
        relay = self.module.SeatRelay(12345, 0, "human", 0)
        self.addCleanup(relay.close)
        relay.endpoints = [self.module.Endpoint(Mock()), self.module.Endpoint(Mock())]
        return relay

    def frame(self, payload):
        return struct.pack("<I", len(payload)) + payload

    def test_fragmented_and_coalesced_client_frames_are_forwarded_without_rewriting(self):
        relay = self.relay()
        wire = self.frame(b"\x00\x00\x05human") + self.frame(b"\x01\x02") + self.frame(b"\x02\x01")
        relay.endpoints[0].incoming.extend(wire[:2])

        relay.forward_frames(0)

        self.assertFalse(relay.hello)
        self.assertEqual(relay.endpoints[1].outgoing, b"")
        relay.endpoints[0].incoming.extend(wire[2:])
        relay.forward_frames(0)
        self.assertTrue(relay.hello)
        self.assertEqual(relay.endpoints[1].outgoing, wire)
        self.assertEqual(relay.endpoints[0].incoming, b"")

    def test_fragmented_welcome_does_not_open_barrier_until_final_byte(self):
        relay = self.relay()
        relay.hello = True
        wire = self.frame(bytes([0, 1, 1, 0, 30, 0]))
        relay.endpoints[1].incoming.extend(wire[:-1])

        relay.forward_frames(1)

        self.assertFalse(relay.welcomed)
        self.assertEqual(relay.endpoints[0].outgoing, b"")
        relay.endpoints[1].incoming.extend(wire[-1:])
        with contextlib.redirect_stdout(io.StringIO()):
            relay.forward_frames(1)
        self.assertTrue(relay.welcomed)
        self.assertEqual(relay.endpoints[0].outgoing, wire)

    def test_lobby_broadcast_to_ungreeted_connection_does_not_open_welcome_barrier(self):
        relay = self.relay()
        lobby = self.frame(bytes([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))
        relay.endpoints[1].incoming.extend(lobby)

        relay.forward_frames(1)

        self.assertFalse(relay.welcomed)
        self.assertEqual(relay.endpoints[0].outgoing, lobby)
        relay.observe(b"\x00\x00\x05human", 0)
        with contextlib.redirect_stdout(io.StringIO()):
            relay.observe(bytes([0, 1, 1, 0, 30, 0]), 1)
        self.assertTrue(relay.welcomed)

    def test_zero_and_oversized_frames_are_rejected_from_header_alone(self):
        for welcomed, length in ((False, 0), (False, 65), (True, 0),
                                 (True, self.module.FRAME_LIMIT + 1)):
            relay = self.relay()
            relay.hello = relay.welcomed = welcomed
            relay.endpoints[1].incoming.extend(struct.pack("<I", length))
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "invalid relay frame length"):
                relay.forward_frames(1)

    def test_maximum_frame_is_forwarded_and_full_queue_applies_backpressure(self):
        relay = self.relay()
        relay.hello = relay.welcomed = True
        wire = self.frame(b"\x01" + b"\x00" * (self.module.FRAME_LIMIT - 1))
        relay.endpoints[1].incoming.extend(wire)

        relay.forward_frames(1)

        self.assertEqual(relay.endpoints[0].outgoing, wire)
        self.assertEqual(relay.endpoints[1].incoming, b"")
        self.assertEqual(relay.read_budget(1), 0)
        relay.endpoints[0].outgoing.clear()
        self.assertEqual(relay.read_budget(1), self.module.READ_CHUNK)

    def test_duplicate_handshakes_and_snapshot_side_mismatches_fail(self):
        relay = self.relay()
        relay.hello = relay.welcomed = True
        for index, payload, message in ((0, b"\x00\x00\x05human", "duplicate handshake"),
                                        (1, bytes([0, 1, 1, 0, 30, 0]), "duplicate handshake"),
                                        (1, bytes([3, 1, 1, 1]), "snapshot side")):
            with self.subTest(index=index, message=message), self.assertRaisesRegex(ValueError, message):
                relay.observe(payload, index)

    def test_truncated_frame_at_eof_fails(self):
        relay = self.relay()
        relay.endpoints[0].incoming.extend(b"\x01")
        relay.endpoints[0].connection.recv.return_value = b""
        with self.assertRaisesRegex(ValueError, "truncated relay frame at EOF"):
            relay.receive(0)

    def test_wire_byte_limit_stops_receiving_before_buffering(self):
        relay = self.relay()
        relay.endpoints[0].received = self.module.CLIENT_BYTE_LIMIT
        relay.endpoints[0].connection.recv.return_value = b"\x01"
        with self.assertRaisesRegex(ValueError, "relay wire byte limit exceeded"):
            relay.receive(0)
        self.assertEqual(relay.endpoints[0].incoming, b"")

    def test_frame_count_limit_stops_before_forwarding(self):
        relay = self.relay()
        relay.hello = True
        relay.endpoints[0].frames = self.module.FRAME_COUNT_LIMIT
        relay.endpoints[0].incoming.extend(self.frame(b"\x01\x02"))
        with self.assertRaisesRegex(ValueError, "relay frame count limit exceeded"):
            relay.forward_frames(0)
        self.assertEqual(relay.endpoints[1].outgoing, b"")

    def test_partial_writes_preserve_unsent_bytes_and_do_not_exceed_chunk_bound(self):
        endpoint = self.module.Endpoint(Mock(), outgoing=bytearray(b"abc"))
        endpoint.connection.send.return_value = 1

        endpoint.write()

        endpoint.connection.send.assert_called_once_with(b"abc")
        self.assertEqual(endpoint.outgoing, b"bc")

    def test_partial_frame_and_blocked_write_deadlines_do_not_require_sleep(self):
        for field, message in (("incoming", "incomplete frame timed out"),
                               ("outgoing", "blocked write timed out")):
            relay = self.relay()
            relay.welcomed = True
            getattr(relay.endpoints[0], field).extend(b"x")
            with patch.object(self.module.time, "monotonic", return_value=31):
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, message):
                    relay.check_deadlines()

    def test_hello_requires_expected_player_or_bot_identity_and_bounded_name(self):
        self.module.verify_hello(b"\x00\x00\x05human", "human")
        self.module.verify_hello(b"\x00\x01\x06drysua", "bot")
        for payload in (b"\x01\x00\x05human", b"\x00\x01\x05human",
                        b"\x00\x02\x05human", b"\x00\x00\x05huma",
                        b"\x00\x00\x05humanX", b"\x80" * 10):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Hello"):
                self.module.verify_hello(payload, "human")

    def test_welcome_requires_exact_slot_tick_rate_mode_and_complete_payload(self):
        self.module.verify_welcome(bytes([0, 1, 1, 0, 30, 0]), 0, 0)
        self.module.verify_welcome(bytes([0, 2, 1, 1, 30, 1]), 1, 1)
        for payload in (bytes([0, 1, 0, 0, 30, 0]), bytes([0, 1, 1, 0, 30, 1]),
                        bytes([0, 1, 1, 0, 29, 0]), bytes([0, 1, 1, 0, 30]),
                        bytes([0, 1, 1, 0, 30, 0, 0]), b"\x80" * 10):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Welcome"):
                self.module.verify_welcome(payload, 0, 0)
        with self.assertRaisesRegex(ValueError, "expected slot 0.*got 1"):
            self.module.verify_welcome(bytes([0, 1, 1, 1, 30, 0]), 0, 0)

    def test_handshake_deadline_is_bounded_without_sleep_and_close_releases_listeners(self):
        with patch.object(self.module.time, "monotonic", return_value=100):
            admission = self.module.Admission(12345, "radiant")
        addresses = tuple(admission.addresses.values())
        try:
            with patch.object(self.module.time, "monotonic", return_value=131):
                with self.assertRaisesRegex(RuntimeError, "Hello/Welcome.*30"):
                    admission.pump()
        finally:
            admission.close()
        for address in addresses:
            host, port = address.rsplit(":", 1)
            with socket.socket() as connection:
                self.assertNotEqual(connection.connect_ex((host, int(port))), 0)


class TerminalLifecycleTests(unittest.TestCase):
    MATCH_OVER = bytes([7, 0, 1, 2, 0] + [0] * 8 + [1] + [0] * 8)
    ORDER = b"\x05\x00\x00\x00\x03\x01\x00\x00\x00"

    def setUp(self):
        self.wire = importlib.import_module("play_admission")
        self.launcher = importlib.import_module("play_match")

    def mock_relay(self):
        relay = self.wire.SeatRelay(12345, 0, "human", 0)
        self.addCleanup(relay.close)
        relay.hello = relay.welcomed = True
        relay.endpoints = [self.wire.Endpoint(Mock()), self.wire.Endpoint(Mock())]
        return relay

    def socket_relay(self):
        relay = self.mock_relay()
        peers = []
        for index in range(2):
            connection, peer = socket.socketpair()
            self.addCleanup(peer.close)
            connection.setblocking(False)
            peer.setblocking(False)
            relay.endpoints[index] = self.wire.Endpoint(connection)
            peers.append(peer)
        return relay, *peers

    def frame(self, payload):
        return struct.pack("<I", len(payload)) + payload

    def drain_to_eof(self, relay, human):
        received = bytearray()
        for _ in range(64):
            relay.pump()
            try:
                data = human.recv(4096)
            except BlockingIOError:
                continue
            if not data:
                return bytes(received)
            received.extend(data)
            self.assertLessEqual(len(received), 4096)
        self.fail("terminal relay did not deliver EOF within 64 nonblocking turns")

    def test_final_frame_drains_before_eof_despite_queued_order_to_closed_server(self):
        relay, human, server = self.socket_relay()
        terminal = self.frame(self.MATCH_OVER)
        relay.endpoints[1].outgoing.extend(self.ORDER)
        server.sendall(terminal)
        server.close()

        received = self.drain_to_eof(relay, human)

        self.assertEqual(received, terminal)
        self.assertFalse(relay.endpoints[0].eof, "the GUI's write socket remains open")
        self.assertEqual(relay.endpoints[1].outgoing, b"")
        human.sendall(self.ORDER)
        for _ in range(4):
            relay.pump()
        self.assertEqual(relay.endpoints[1].outgoing, b"")

    def test_epipe_during_fragmented_match_over_waits_for_verified_terminal_frame(self):
        relay, human, server = self.socket_relay()
        terminal = self.frame(self.MATCH_OVER)
        relay.endpoints[1].outgoing.extend(self.ORDER)
        server.sendall(terminal[:-1])
        server.shutdown(socket.SHUT_RD)

        relay.pump()

        self.assertFalse(relay.match_over)
        self.assertEqual(relay.endpoints[0].outgoing, b"")
        server.sendall(terminal[-1:])
        server.shutdown(socket.SHUT_WR)
        self.assertEqual(self.drain_to_eof(relay, human), terminal)

    def test_upstream_write_failure_without_terminal_frame_has_bounded_deadline(self):
        relay = self.mock_relay()
        server = relay.endpoints[1]
        server.outgoing.extend(self.ORDER)
        server.connection.send.side_effect = BrokenPipeError(errno.EPIPE, "injected late order")
        with patch.object(self.wire.select, "select", return_value=([], [server.connection], [])), \
                patch.object(self.wire.time, "monotonic", return_value=100):
            relay.pump()

        with patch.object(self.wire.time, "monotonic", return_value=131):
            with self.assertRaisesRegex(RuntimeError, "write failed before verified MatchOver.*30"):
                relay.check_deadlines()

    def test_server_eof_before_match_over_is_not_success_while_client_is_connected(self):
        relay = self.mock_relay()
        server = relay.endpoints[1]
        server.connection.recv.return_value = b""
        with patch.object(self.wire.select, "select", return_value=([server.connection], [], [])):
            with self.assertRaisesRegex(ValueError, "server disconnected before verified MatchOver"):
                relay.pump()

    def test_reset_before_verified_match_over_is_not_silently_converted_to_eof(self):
        for index in (0, 1):
            relay = self.mock_relay()
            relay.endpoints[index].connection.recv.side_effect = ConnectionResetError(errno.ECONNRESET, "injected")
            with self.subTest(endpoint=index), self.assertRaisesRegex(ValueError, "reset before verified MatchOver"):
                relay.receive(index)

    def test_server_reset_after_verified_match_over_preserves_pending_final_frame(self):
        relay = self.mock_relay()
        terminal = self.frame(self.MATCH_OVER)
        relay.endpoints[1].incoming.extend(terminal)
        relay.forward_frames(1)
        relay.endpoints[1].connection.recv.side_effect = ConnectionResetError(errno.ECONNRESET, "terminal reset")

        relay.receive(1)

        self.assertTrue(relay.endpoints[1].eof)
        self.assertEqual(relay.endpoints[0].outgoing, terminal)

    def test_eof_and_reset_never_hide_truncated_frames_even_after_match_over(self):
        for index in (0, 1):
            for reset in (False, True):
                relay = self.mock_relay()
                relay.observe(self.MATCH_OVER, 1)
                source = relay.endpoints[index]
                source.incoming.extend(b"\x01")
                if reset:
                    source.connection.recv.side_effect = ConnectionResetError(errno.ECONNRESET, "injected")
                else:
                    source.connection.recv.return_value = b""
                with self.subTest(endpoint=index, reset=reset), self.assertRaisesRegex(ValueError, "truncated"):
                    relay.receive(index)

    def test_invalid_or_duplicate_match_over_cannot_authorize_terminal_cleanup(self):
        malformed = (b"\x07", self.MATCH_OVER[:-1], self.MATCH_OVER + b"\x00",
                     bytes([7, 3]) + self.MATCH_OVER[2:], bytes([7, 0, 0]) + self.MATCH_OVER[3:],
                     self.MATCH_OVER[:3] + b"\x01" + self.MATCH_OVER[4:],
                     self.MATCH_OVER[:13] + b"\x00" + self.MATCH_OVER[14:],
                     self.MATCH_OVER[:2] + b"\x81\x00" + self.MATCH_OVER[3:],
                     self.MATCH_OVER[:5] + b"\x80\x80\x04" + self.MATCH_OVER[6:],
                     self.MATCH_OVER[:10] + b"\x80\x80\x80\x80\x10" + self.MATCH_OVER[11:])
        for payload in malformed:
            relay = self.mock_relay()
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "MatchOver"):
                relay.observe(payload, 1)
            self.assertFalse(relay.match_over)
        relay = self.mock_relay()
        relay.observe(self.MATCH_OVER, 1)
        with self.assertRaisesRegex(ValueError, "after MatchOver"):
            relay.observe(self.MATCH_OVER, 1)

    def test_downstream_epipe_cannot_claim_success_before_final_frame_reaches_client(self):
        relay = self.mock_relay()
        relay.endpoints[1].incoming.extend(self.frame(self.MATCH_OVER))
        relay.forward_frames(1)
        client = relay.endpoints[0]
        client.connection.send.side_effect = BrokenPipeError(errno.EPIPE, "undelivered final frame")
        with patch.object(self.wire.select, "select", return_value=([], [client.connection], [])):
            with self.assertRaisesRegex(BrokenPipeError, "undelivered final frame"):
                relay.pump()

    def test_client_reset_cannot_discard_undelivered_final_frame(self):
        relay = self.mock_relay()
        relay.endpoints[1].incoming.extend(self.frame(self.MATCH_OVER))
        relay.forward_frames(1)
        relay.endpoints[0].connection.recv.side_effect = ConnectionResetError(errno.ECONNRESET, "injected")

        with self.assertRaisesRegex(ValueError, "client reset before final-frame drain"):
            relay.receive(0)

    def test_normal_server_and_bot_exits_retain_gui_after_final_frame_and_late_order(self):
        relay, human, server_peer = self.socket_relay()
        terminal = self.frame(self.MATCH_OVER)
        relay.endpoints[1].outgoing.extend(self.ORDER)
        server_peer.sendall(terminal)
        server_peer.close()
        with tempfile.TemporaryDirectory(prefix="play-terminal-", dir=TEMPORARY) as temporary:
            supervisor = self.launcher.Supervisor(Path(temporary))
            self.addCleanup(supervisor.close)
            supervisor.admission = SimpleNamespace(relays=[relay], welcomed=True, pump=relay.pump, close=relay.close)
            server = SimpleNamespace(name="server", exit_status=lambda: 0)
            bot = SimpleNamespace(name="bot", exit_status=lambda: 0)
            client = SimpleNamespace(name="client", exit_status=lambda: None)
            received, turns = bytearray(), 0

            def check_gui():
                nonlocal turns
                turns += 1
                if turns > 16:
                    raise RuntimeError("GUI results still active")
                try:
                    data = human.recv(4096)
                except BlockingIOError:
                    return
                received.extend(data)
                if not data:
                    human.sendall(self.ORDER)

            with patch.object(supervisor.selector, "select", return_value=[]), \
                    patch.object(supervisor, "check_stop", side_effect=check_gui), \
                    patch.object(self.wire.time, "monotonic", side_effect=lambda: 100 + turns):
                with self.assertRaisesRegex(RuntimeError, "GUI results still active"):
                    supervisor.wait_game(server, bot, client)
            self.assertEqual(received, terminal)
            self.assertFalse(relay.endpoints[0].eof)

    def test_ready_relay_frame_does_not_wait_twice_on_idle_log_selector(self):
        relay, human, server = self.socket_relay()
        human.sendall(self.ORDER)
        waits = []
        with tempfile.TemporaryDirectory(prefix="play-poll-", dir=TEMPORARY) as temporary:
            supervisor = self.launcher.Supervisor(Path(temporary))
            self.addCleanup(supervisor.close)
            supervisor.admission = SimpleNamespace(pump=relay.pump, close=relay.close)
            with patch.object(supervisor.selector, "select", side_effect=lambda timeout: waits.append(timeout) or []):
                for _ in range(2):
                    supervisor.pump(0.01)
                    try:
                        received = server.recv(4096)
                        break
                    except BlockingIOError:
                        continue
                else:
                    self.fail("ready frame was not forwarded within two supervisor turns")
            self.assertEqual(received, self.ORDER)
            self.assertEqual(sum(waits), 0, f"already-ready frame paid log-only waits: {waits}")

    def test_idle_relay_keeps_log_selector_timeout_instead_of_busy_polling(self):
        with tempfile.TemporaryDirectory(prefix="play-poll-", dir=TEMPORARY) as temporary:
            supervisor = self.launcher.Supervisor(Path(temporary))
            self.addCleanup(supervisor.close)
            supervisor.admission = SimpleNamespace(pump=Mock(return_value=False), close=Mock())
            with patch.object(supervisor.selector, "select", return_value=[]) as selector:
                supervisor.pump(0.01)
            selector.assert_called_once_with(0.01)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        TEMPORARY.mkdir(parents=True, exist_ok=True)
        self.module = importlib.import_module("play_match")

    def test_review_pins_and_replay_limit_match_human_review_contract(self):
        self.assertEqual(self.module.REVIEW_DIRECTORY, REVIEW)
        self.assertEqual(self.module.REVIEW_BINARY_SHA, BINARY_SHA)
        self.assertEqual(self.module.REVIEW_WEIGHTS_SHA, WEIGHTS_SHA)
        self.assertEqual(self.module.REPLAY_LIMIT, 2 * 1024**3)

    def test_no_arguments_select_human_radiant_bot_dire_and_no_weights_override(self):
        arguments = self.module.parse_arguments([])
        self.assertEqual(arguments.human_side, "radiant")
        self.assertEqual(arguments.bot_side, "dire")
        self.assertIsNone(arguments.weights_directory)
        self.assertEqual(arguments.port, 4455)
        self.assertEqual(arguments.seed, 9000001)

    def test_readiness_requires_complete_exact_line_and_valid_matching_port(self):
        parse = self.module.ready_port
        self.assertIsNone(parse(b"bota-server listening on 0.0.0.0:12", 0))
        self.assertEqual(parse(b"bota-server listening on 0.0.0.0:12\n", 0), 12)
        self.assertEqual(parse(b"bota-server listening on 0.0.0.0:12\n", 12), 12)
        for data in (b"wrong\n", b"bota-server listening on 0.0.0.0:0\n",
                     b"bota-server listening on 0.0.0.0:65536\n",
                     b"bota-server listening on 0.0.0.0:13\n"):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "readiness"):
                parse(data, 12)
        self.assertIsNone(parse(b"x" * 4096, 0))
        with self.assertRaisesRegex(ValueError, "4096"):
            parse(b"x" * 4097, 0)

    def test_successful_server_and_bot_exits_do_not_end_runtime_loop(self):
        with tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            server = SimpleNamespace(name="server", exit_status=lambda: 0)
            client = SimpleNamespace(name="client", exit_status=lambda: None)
            bot = SimpleNamespace(name="bot", exit_status=lambda: 0)
            supervisor.admission = SimpleNamespace(relays=[], welcomed=True, close=lambda: None)
            try:
                with patch.object(supervisor, "pump", side_effect=[None, RuntimeError("GUI still active")]):
                    with self.assertRaisesRegex(RuntimeError, "GUI still active"):
                        supervisor.wait_game(server, bot, client)
            finally:
                supervisor.close()

    def test_signal_during_spawn_is_recorded_and_global_failure_still_cleans_child(self):
        with tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            original = subprocess.Popen

            def interrupted_spawn(*arguments, **keywords):
                process = original(*arguments, **keywords)
                supervisor.request_stop(signal.SIGINT, None)
                return process

            try:
                with patch.object(self.module.subprocess, "Popen", side_effect=interrupted_spawn):
                    child = supervisor.spawn("build-bota", [sys.executable, "-c",
                                             "import signal; signal.pause()"], ROOT)
                self.assertEqual(supervisor.stop_status, 130)
                self.assertEqual(len(supervisor.children), 1)
                raise RuntimeError("injected supervisor failure")
            except RuntimeError as error:
                self.assertEqual(str(error), "injected supervisor failure")
            finally:
                supervisor.close()
            self.assertIsNotNone(child.process.returncode)

    def test_readiness_deadline_uses_monotonic_time_without_sleep(self):
        with tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            try:
                child = supervisor.spawn("server", [sys.executable, "-c",
                                         "import signal; signal.pause()"], ROOT)
                with patch.object(self.module.time, "monotonic", side_effect=[100, 111]):
                    with self.assertRaisesRegex(RuntimeError, "readiness.*10"):
                        supervisor.wait_ready(child, 0)
            finally:
                supervisor.close()

    def test_fragmented_client_error_is_detected_across_more_than_two_reads(self):
        with tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            try:
                child = supervisor.spawn("client", [sys.executable, "-c",
                                         "import signal; signal.pause()"], ROOT)
                key = SimpleNamespace(fileobj=child.process.stderr, data=(child, False))
                with patch.object(supervisor.selector, "select", return_value=[(key, 1)]), \
                        patch.object(self.module.os, "read", side_effect=[b"bota-", b"cli", b"ent: failure"]):
                    supervisor.pump(0)
                    supervisor.pump(0)
                    with self.assertRaisesRegex(RuntimeError, "bota-client reported an error"):
                        supervisor.pump(0)
            finally:
                supervisor.close()

    def test_final_drain_reports_errors_instead_of_turning_late_failures_into_success(self):
        for data, limit, message in ((b"bota-client: late failure", 128, "bota-client"),
                                     (b"x" * 65, 64, "log limit")):
            with self.subTest(message=message), \
                    tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary:
                supervisor = self.module.Supervisor(Path(temporary))
                try:
                    child = supervisor.spawn("client", [sys.executable, "-c",
                                             "import signal; signal.pause()"], ROOT)
                    key = SimpleNamespace(fileobj=child.process.stderr, data=(child, False))
                    with patch.object(supervisor.selector, "select", return_value=[(key, 1)]), \
                            patch.object(self.module.os, "read", return_value=data), \
                            patch.object(self.module, "LOG_LIMIT", limit):
                        with self.assertRaisesRegex(RuntimeError, message):
                            supervisor.pump(0, final=True)
                finally:
                    supervisor.close()

    def test_main_global_failure_cleans_child_with_inherited_sigchld_ignore(self):
        children = []

        def failure(supervisor, root, _arguments):
            children.append(supervisor.spawn("build-bota", [sys.executable, "-c",
                                            "import signal; signal.pause()"], root))
            raise RuntimeError("injected global failure")

        previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        mask = os.umask(0o077)
        try:
            with tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY) as temporary, \
                    patch.dict(os.environ, DISPLAY=":fixture"), \
                    patch.object(self.module, "current_paths", return_value=(Path("unused"), Path("unused"))), \
                    patch.object(self.module, "release_executables"), \
                    patch.object(self.module.tempfile, "mkdtemp", return_value=temporary), \
                    patch.object(self.module.Supervisor, "run", failure), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(self.module.main(["--no-build"]), 1)
                self.assertIn("injected global failure", errors.getvalue())
                self.assertIn("logs:", errors.getvalue())
                self.assertNotIn("cleanup failed", errors.getvalue())
                self.assertIsNotNone(children[0].process.returncode)
        finally:
            os.umask(mask)
            signal.signal(signal.SIGCHLD, previous)

    @unittest.skipUnless((ROOT / "bota/target/release/bota-server").is_file(),
                         "optional smoke requires an existing release server; never builds it")
    def test_existing_release_server_readiness_works_without_connecting_a_probe(self):
        with tempfile.TemporaryDirectory(prefix="play-native-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            try:
                server = supervisor.spawn("server", [str(ROOT / "bota/target/release/bota-server"),
                                           "--port", "0", "--mode", "realtime", "--players", "2",
                                           "--map", "2", "--seed", "9000001"], ROOT)
                port = supervisor.wait_ready(server, 0)
                self.assertGreater(port, 0)
                self.assertLessEqual(port, 65535)
                self.assertIsNone(server.exit_status())
                self.assertIn(f"bota-server listening on 0.0.0.0:{port}\n".encode(),
                              (Path(temporary) / "server.log").read_bytes())
            finally:
                supervisor.close()
            self.assertIn(server.process.returncode, (-signal.SIGTERM, -signal.SIGKILL))

    def test_native_neural_review_and_player_verify_both_sides_in_realtime_and_lockstep(self):
        server = self.historical_server()
        if not NATIVE_BINARY.is_file() or not (NATIVE_WEIGHTS / "drysua.weights.safetensors").is_file():
            self.skipTest("historical smoke needs frozen review bot and weights; never builds")
        self.assertEqual(self.module.artifact_digest(NATIVE_BINARY), BINARY_SHA)
        self.assertEqual(self.module.artifact_digest(NATIVE_WEIGHTS / "drysua.weights.safetensors"), WEIGHTS_SHA)
        for mode, limit in ((0, 30), (1, 1000)):
            for human_slot in (0, 1):
                with self.subTest(mode=mode, human_slot=human_slot):
                    self.native_match(mode, limit, human_slot, server, NATIVE_BINARY, "neural", 0, NATIVE_WEIGHTS)

    def historical_server(self):
        manifest = HISTORICAL_BASELINE / "baseline.json"
        if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size > 65536:
            self.skipTest("historical smoke needs bounded pinned baseline.json; root server is never a fallback")
        try:
            with manifest.open("rb") as stream:
                contents = stream.read(65537)
            if len(contents) > 65536:
                raise ValueError("baseline.json exceeds 64 KiB")
            data = json.loads(contents)
            if not isinstance(data, dict) or not isinstance(data.get("simulator"), dict):
                raise ValueError("baseline.json lacks simulator object")
        except (ValueError, OSError) as error:
            self.skipTest(f"invalid historical baseline.json: {error}; root server is never a fallback")
        simulator = data.get("simulator", {})
        if (simulator.get("path") != "runtime/bota-server"
                or simulator.get("sha256") != HISTORICAL_SERVER_SHA
                or simulator.get("commit") != "18db0f62d9a2b94e755c43fd29a959db204cc20b"):
            self.skipTest("historical server manifest pin mismatch; root server is never a fallback")
        server = HISTORICAL_BASELINE / simulator["path"]
        try:
            digest = self.module.artifact_digest(server)
        except RuntimeError as error:
            self.skipTest(f"historical server unavailable: {error}; root server is never a fallback")
        if digest != HISTORICAL_SERVER_SHA or not os.access(server, os.X_OK):
            self.skipTest("historical server SHA/executable mismatch; root server is never a fallback")
        return server

    def test_historical_smoke_never_accepts_a_manifest_redirect_to_current_root_server(self):
        with tempfile.TemporaryDirectory(prefix="play-archive-", dir=TEMPORARY) as temporary:
            manifest = {"simulator": {"path": str(ROOT / "bota/target/release/bota-server"),
                                     "sha256": HISTORICAL_SERVER_SHA,
                                     "commit": "18db0f62d9a2b94e755c43fd29a959db204cc20b"}}
            (Path(temporary) / "baseline.json").write_text(json.dumps(manifest))
            with patch.dict(globals(), HISTORICAL_BASELINE=Path(temporary)):
                with self.assertRaisesRegex(unittest.SkipTest, "root server is never a fallback"):
                    self.historical_server()

    def test_current_native_map2_teacher_protocol_with_explicit_build_attestation(self):
        # An existing target path alone does not establish that the concurrent rebase build finished.
        bot = ROOT / "drysua/target/release/drysua"
        server = ROOT / "bota/target/release/bota-server"
        for role, binary in (("BOT", bot), ("SERVER", server)):
            expected = os.environ.get(f"PLAY_TEST_CURRENT_{role}_SHA256", "")
            if not expected:
                self.skipTest("current native Map2 smoke requires PLAY_TEST_CURRENT_BOT_SHA256 and "
                              "PLAY_TEST_CURRENT_SERVER_SHA256 from verified current builds; never uses stale targets")
            self.assertRegex(expected, r"^[0-9a-f]{64}$")
            self.assertNotIn(expected, (BINARY_SHA, HISTORICAL_SERVER_SHA))
            self.assertEqual(self.module.artifact_digest(binary), expected)
            self.assertTrue(os.access(binary, os.X_OK))
        for mode, limit in ((0, 30), (1, 1000)):
            for human_slot in (0, 1):
                with self.subTest(mode=mode, human_slot=human_slot):
                    self.native_match(mode, limit, human_slot, server, bot, "teacher", 2)

    def native_match(self, mode, limit, human_slot, server_binary, bot_binary, policy, map_id, weights=None):
        with tempfile.TemporaryDirectory(prefix="play-native-review-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            children, statuses, failure = [], None, None
            previous = {}
            try:
                previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    previous[number] = signal.signal(number, supervisor.request_stop)
                server = supervisor.spawn("server", [str(server_binary),
                                           "--port", "0", "--mode", ("realtime", "lockstep")[mode], "--players", "2",
                                           "--map", str(map_id), "--seed", "9000001",
                                           "--ack-timeout-ticks", "900"], ROOT)
                children.append(server)
                port = supervisor.wait_ready(server, 0)
                admission = self.module.Admission(port, ("radiant", "dire")[human_slot], mode)
                supervisor.admission = admission
                command = [str(bot_binary), "--addr", admission.addresses["bot"],
                           "--name", "drysua", "--policy", policy, "--limit", str(limit)]
                if weights is not None:
                    command += ["--weights-directory", str(weights)]
                children.append(supervisor.spawn("bot", command, ROOT))
                children.append(supervisor.spawn("client", [sys.executable, "-B", "-c", NATIVE_CLIENT,
                                                admission.addresses["human"], str(human_slot), str(mode), str(limit),
                                                str(ROOT / "drysua/scripts")], ROOT))
                statuses = self.wait_native_peers(supervisor, children)
            except Exception as error:
                failure = error
            finally:
                try:
                    supervisor.close()
                finally:
                    for number, handler in previous.items():
                        signal.signal(number, handler)
            logs = {}
            for child in children:
                logs[child.name] = read_client_output(Path(temporary) / f"{child.name}.log")
            diagnostic = "\n".join(f"{name}:\n{text}" for name, text in logs.items())
            self.assertIsNone(failure, f"{failure}\n{diagnostic}")
            self.assertIn(statuses[0], (None, 0), diagnostic)
            self.assertEqual(statuses[1:], [0, 0], diagnostic)
            self.assertTrue(admission.welcomed, diagnostic)
            self.assertIn(f"headless human Welcome slot {human_slot}", logs["client"], diagnostic)
            self.assertIn(f"headless human snapshot team {human_slot}; completed {limit} ticks",
                          logs["client"], diagnostic)
            bot_team = ("Dire", "Radiant")[human_slot]
            self.assertRegex(logs["bot"], rf"(?m)^played {limit} ticks as Some\({bot_team}\); winner None; "
                             r"[0-9]+ decisions, [0-9]+ orders, 0 rejected orders$", diagnostic)
            self.assertIsNotNone(outcome_summary(logs["bot"], 1 - human_slot), diagnostic)
            print(f"play native verified: map={map_id} mode={mode} human_slot={human_slot} "
                  f"{policy}_bot={bot_team} ticks={limit}")

    def test_native_cap_transport_errors_require_known_socket_errors(self):
        reset = ConnectionResetError("peer closed at its explicit cap")
        wrapped = ValueError("connection reset before verified MatchOver")
        wrapped.__cause__ = reset
        truncated = ValueError("truncated relay frame at reset")
        truncated.__cause__ = reset
        for cause, expected in ((BrokenPipeError(), True), (reset, True), (wrapped, True),
                                (truncated, False), (ValueError(str(wrapped)), False),
                                (RuntimeError("unrelated failure"), False), (None, False)):
            with self.subTest(cause=cause):
                error = RuntimeError("relay error")
                error.__cause__ = cause
                self.assertEqual(native_cap_transport_error(error), expected)

    def wait_native_peers(self, supervisor, children):
        deadline = self.module.time.monotonic() + 30
        for _ in range(60000):
            supervisor.check_stop()
            statuses = [child.exit_status() for child in children]
            if all(status is not None for status in statuses[1:]) or any(status not in (None, 0) for status in statuses):
                return statuses
            try:
                supervisor.pump(0.001)
            except RuntimeError as error:
                if not native_cap_transport_error(error):
                    raise
                # Tick-cap clients can close with unread Events and reset TCP. Both must exit,
                # and exact-cap summaries and zero exits remain mandatory in native_match.
                for child in children[1:]:
                    descriptor = os.pidfd_open(child.process.pid)
                    try:
                        remaining = max(0, min(3, deadline - self.module.time.monotonic()))
                        if not select.select([descriptor], [], [], remaining)[0]:
                            raise RuntimeError("native peer did not exit at explicit tick cap") from error
                    finally:
                        os.close(descriptor)
                return [child.exit_status() for child in children]
            if self.module.time.monotonic() >= deadline:
                self.fail("native side smoke exceeded its 30-second deadline")
        self.fail("native side supervision iteration limit exceeded")


if __name__ == "__main__":
    unittest.main()
