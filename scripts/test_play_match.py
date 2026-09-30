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


ROOT = Path(__file__).resolve().parents[2]
TEMPORARY = ROOT / "drysua/artifacts/temp"
# Bounded native client-output assertions moved here from the archived historical
# release evaluator when that harness left the tracked tree.
SUMMARY = re.compile(r"played (\d+) ticks as Some\((Radiant|Dire)\); winner "
                     r"(None|Some\((Radiant|Dire|Neutral)\)); "
                     r"\d+ decisions, \d+ orders, (\d+) rejected orders\n?")
CLIENT_OUTPUT_LIMIT = 1024 * 1024
TELEMETRY_LINE_LIMIT = 4096
POLICY_FIELD = r"policy=(?:teacher|neural)"
RECEIVE_FIELD = (r"receive_wait_scope=(?:socket_read|wire_hear_including_decode"
                 r"|mixed_socket_read_and_wire_hear|unavailable)")
SEAT_FIELDS = rf"slot=(?P<slot>[01]) {POLICY_FIELD} mode=(?:lockstep|realtime)"
HISTOGRAM_FIELDS = "".join(
    rf" {name}_count=\d+ {name}_total_ns=\d+ {name}_p50_upper_ns=(?:\d+|unknown)"
    rf" {name}_p95_upper_ns=(?:\d+|unknown) {name}_max_ns=\d+"
    for name in ("compute", "receive_wait", "decision", "order_send", "ack_send"))
ASYNC_LOG_FIELDS = r"(?: dropped_logs=(?P<dropped_logs>0|[1-9][0-9]{0,19}))?"
TELEMETRY = tuple(re.compile(pattern + ASYNC_LOG_FIELDS) for pattern in (
    rf"level=INFO event=live_performance_start {SEAT_FIELDS} tick_rate=30 "
    rf"report_every=\d+ debug_every=\d+ debug_limit=\d+ {RECEIVE_FIELD} "
    r"compute_scope=internal_elapsed_excluding_receive_and_send",
    r"level=(?:INFO|WARN) event=live_performance scope=(?:window|total) "
    r"reason=(?:periodic|match_over|limit) updates=\d+ progress_ticks=\d+ elapsed_ns=\d+ "
    r"updates_per_second=(?:\d+\.\d{3}|unknown) realtime_factor=(?:\d+\.\d{3}|unknown) "
    r"tick_rate=30 budget_ns=\d+ compute_overruns=\d+ service_overruns=\d+ percentiles=log2_upper_bounds"
    rf"{HISTOGRAM_FIELDS} pending_update=(?:true|false) saturated=false {SEAT_FIELDS} "
    rf"{RECEIVE_FIELD} timing_valid=true",
    rf"level=DEBUG event=live_decision slot=(?P<slot>[01]) {POLICY_FIELD} tick=\d+ "
    r"decision_ns=(?:\d+|unknown) order_sent=(?:true|false) order_send_ns=(?:\d+|unknown) "
    r"ack_send_ns=(?:\d+|unknown)",
    r'level=INFO event=live_performance_config_defaulted reason="DRYSUA_PERF_DEBUG_EVERY must be '
    r'(?:Unicode digits|an integer in 0\.\.=4294967295)"',
))
assert TELEMETRY_LINE_LIMIT < CLIENT_OUTPUT_LIMIT


def outcome_summary(text, slot):
    """Find exactly one summary amid known, bounded telemetry; retain the original log."""
    assert slot in (0, 1)
    if len(text) > CLIENT_OUTPUT_LIMIT or len(text.encode("utf-8")) > CLIENT_OUTPUT_LIMIT:
        raise ValueError("client output limit exceeded")
    summary = None
    for line in text.splitlines():
        if len(line) > TELEMETRY_LINE_LIMIT:
            raise ValueError("client output line limit exceeded")
        match = SUMMARY.fullmatch(line)
        if match:
            if summary is not None:
                raise ValueError("duplicate client outcome")
            summary = match
            continue
        records = [pattern.fullmatch(line) for pattern in TELEMETRY]
        record = next((record for record in records if record is not None), None)
        if record is None:
            raise ValueError("invalid client telemetry or unexpected output")
        if record.groupdict().get("slot") is not None and int(record["slot"]) != slot:
            raise ValueError("invalid client telemetry or unexpected output")
        if record["dropped_logs"] is not None and int(record["dropped_logs"]) > 2**64 - 1:
            raise ValueError("invalid client dropped_logs counter")
    return summary


def read_client_output(path):
    """Read at most one MiB of UTF-8 client output, detecting growth with a sentinel byte."""
    with path.open("rb") as stream:
        data = stream.read(CLIENT_OUTPUT_LIMIT + 1)
    if len(data) > CLIENT_OUTPUT_LIMIT:
        raise ValueError("client output limit exceeded")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("client output is not UTF-8") from error


def weights_fixture():
    # Launchers never parse the tensor file; only mock executables consume it.
    return b"runtime weights fixture"


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
selected, ready = set(), set()
start_requested = False
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
    if start_requested and len(ready) == 2:
        for peer, slot in seats.items():
            peer.sendall(frame(bytes([3, 1, 1, slot])))
        start_requested = False
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
            elif role != "bota-server" and payload[0] == 3:
                report(snapshot=payload.hex())
                peer.sendall(frame(b"\x03\x01\x00\x00\x00"))
            elif role == "bota-server" and payload[0] == 3:
                report(order=payload.hex(), slot=seats[peer])
            elif role == "bota-server" and payload[0] == 1:
                assert payload == b"\x01\x02"
                assert peer in seats
                selected.add(peer)
            elif role == "bota-server" and payload[0] == 2:
                assert payload == b"\x02\x01"
                assert peer in selected
                ready.add(peer)
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
    elif command == b"g":
        start_requested = True
    elif command == b"o":
        peers[0].sendall(frame(b"\x03\x01\x00\x00\x00"))
        report(late_order=True)
else:
    raise RuntimeError("fixture command limit exceeded")
'''


NATIVE_CLIENT = r'''
import socket, struct, sys
sys.path.insert(0, sys.argv[5])
from play_admission import varint
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
        for name in ("play.sh", "play_match.py", "play_admission.py",
                     "play_reward.py", "play_pacing.py"):
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
        self.weights = self.root / "current metadata fixture"
        self.weights.mkdir()
        (self.weights / "drysua.weights.safetensors").write_bytes(weights_fixture())
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
        self.process = subprocess.Popen([str(self.root / "drysua/scripts/play.sh"), *arguments],
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

    def game(self, *arguments, **options):
        self.launch("--port", "0", *arguments, **options)
        self.send("bota-server", b"r")
        first = self.record("bota-server")
        self.send("bota-server", b"R")
        self.assertEqual(self.record("bota-server"), first)
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

    def test_explicit_build_flag_rebuilds_both_workspaces_and_launches_map2_pure_neural(self):
        port = self.game("--build")
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
        self.send("bota-server", b"g")
        for name, slot in (("bota-client", 0), ("drysua", 1)):
            self.assertEqual(self.record(name), {"snapshot": bytes([3, 1, 1, slot]).hex()})
        orders = [self.record("bota-server") for _ in range(2)]
        self.assertEqual(sorted(orders, key=lambda record: record["slot"]),
                         [{"order": "0301000000", "slot": slot} for slot in (0, 1)])
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

    def test_default_uses_existing_release_binaries_without_building(self):
        self.game()
        self.assertNotIn("build-bota", self.children)
        self.assertNotIn("build-drysua", self.children)
        self.send("bota-client", b"0")
        self.finish(0)
        self.assert_children_stopped()

    def test_default_builds_only_missing_release_workspace(self):
        (self.root / "drysua/target/release/drysua").unlink()
        self.launch()
        self.child("build-drysua")
        self.assertNotIn("build-bota", self.children)
        # The mock cargo never materializes binaries; a still-missing bot must fail
        # closed before the server starts, never fall back to a stale executable.
        self.assertIn("release executable missing", self.finish(1))
        self.assert_children_stopped()

    def test_interrupt_during_build_kills_children_and_term_ignoring_grandchildren(self):
        self.launch("--build", PLAY_TEST_BUILD="hold")
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
        self.launch("--build", PLAY_TEST_BUILD="hold")
        self.send("build-bota", b"i")
        self.assertEqual(json.loads(self.child("build-bota")["stream"].readline(8192)),
                         {"ignoring": True})
        self.process.send_signal(signal.SIGTERM)
        self.finish(143)
        self.assert_children_stopped()

    def test_build_failures_stop_before_server_without_fallback(self):
        for name in ("build-bota", "build-drysua"):
            with self.subTest(component=name):
                self.launch("--build", PLAY_TEST_BUILD=name)
                self.send(name, b"7")
                self.assertIn(f"{name} exited with status 7", self.finish(1))
                self.assertNotIn("bota-server", self.children)
                self.assert_children_stopped()

    def test_invalid_readiness_stops_server_and_reports_reason(self):
        for command, message in ((b"s", "server"),
                                 (b"x", "4096"), (b"q", "stdout closed before readiness")):
            with self.subTest(command=command):
                self.launch("--no-build")
                self.send("bota-server", command)
                self.assertIn(message, self.finish(1))
                self.assert_children_stopped()

    def test_runtime_component_crashes_stop_the_match_and_retain_error_logs(self):
        for name in ("bota-client", "drysua", "bota-server"):
            with self.subTest(component=name):
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
        module = importlib.import_module("play_match")
        with patch("play_match.shutil.which", return_value=None), patch.dict(os.environ, DISPLAY=":fixture"):
            # Existing release binaries make cargo unnecessary without --build.
            self.assertIsNone(module.preflight(self.root, False, False))
            with self.assertRaisesRegex(RuntimeError, "cargo is required"):
                module.preflight(self.root, False, True)
        for target, message in ((self.tools / "cargo", b"cargo is required"),
                                (self.root / "bota/Cargo.toml", b"source manifest missing")):
            with self.subTest(target=target), patch("play_match.shutil.which", return_value=None):
                if target.name != "cargo":
                    target.unlink()
                with patch.dict(os.environ, DISPLAY=":fixture"):
                    with self.assertRaisesRegex(RuntimeError, message.decode()):
                        module.preflight(self.root, False, True)
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

    def test_side_options_and_weightless_teacher_preserve_admission_and_cleanup(self):
        for arguments, human_slot in ((("--human-side", "dire"), 1), (("--bot-side", "radiant"), 1),
                                      (("--human-side", "radiant"), 0), (("--bot-side", "dire"), 0),
                                      (("--human-side", "dire", "--bot-side", "radiant"), 1),
                                      (("--human-side", "radiant", "--bot-side", "dire"), 0),
                                      (("--human-side", "radiant", "--opponent", "teacher"), 0),
                                      (("--human-side", "dire", "--opponent", "teacher"), 1)):
            with self.subTest(arguments=arguments):
                teacher = "teacher" in arguments
                self.game("--no-build", *arguments, weights=not teacher)
                self.assertEqual(self.child("bota-client")["slot"], human_slot)
                self.assertEqual(self.child("drysua")["slot"], 1 - human_slot)
                if teacher:
                    bot = self.child("drysua")["arguments"]
                    self.assertEqual(bot[-2:], ["--policy", "teacher"])
                    self.assertNotIn("--weights-directory", bot)
                self.send("bota-client", b"0")
                self.finish(0)
                self.assert_children_stopped()

    def test_same_sides_are_rejected_before_display_or_children(self):
        for side in ("radiant", "dire"):
            self.launch("--human-side", side, "--bot-side", side, DISPLAY="")
            _, error = self.process.communicate(timeout=5)
            self.assertEqual(self.process.returncode, 2)
            self.assertIn(b"must be opposite", error)

    def test_either_wrong_welcome_slot_fails_and_cleans_clients_and_relays(self):
        for slot in (0, 1):
            with self.subTest(slot=slot):
                self.launch("--no-build", "--port", "0", PLAY_TEST_WRONG_SLOT="0" if slot else "1",
                            PLAY_TEST_WRONG_SECOND_SLOT="0")
                self.send("bota-server", b"R")
                self.record("bota-server")
                for name in ("bota-client", "drysua"):
                    self.child(name)
                output = self.finish(1)
                self.assertIn(f"expected slot {slot}, got {1 - slot}", output)
                if slot:
                    self.assertIn("verified human Radiant: Welcome slot 0", output)
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

    def preflight_failure(self, message):
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 1, output.decode())
        self.assertIn(message, error.decode())
        self.assertNotIn(b"logs:", output + error)
        self.assertFalse(list((self.root / "drysua/artifacts/temp").glob("play-*")))
        self.assertFalse(select.select([self.listener], [], [], 0)[0], "unexpected child started")
        return error.decode()

    def test_no_arguments_fail_closed_before_display_build_logs_or_default_model(self):
        self.launch(weights=False, DISPLAY="")
        error = self.preflight_failure("Neural play requires --weights-directory")
        self.assertIn("no default model and no Teacher fallback", error)

    def test_missing_weights_checked_before_missing_executables_and_display(self):
        (self.weights / "drysua.weights.safetensors").unlink()
        (self.root / "drysua/target/release/drysua").unlink()
        self.launch("--no-build", DISPLAY="")
        self.preflight_failure("runtime weights")

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
        (weights / "drysua.weights.safetensors").write_bytes(weights_fixture())
        self.game("--no-build", "--weights-directory", str(weights.relative_to(self.temporary.name)))
        arguments = self.child("drysua")["arguments"]
        self.assertEqual(arguments[arguments.index("--policy") + 1], "neural")
        self.assertEqual(arguments[-1], str(weights))
        self.assertEqual(self.child("drysua")["executable"], str(self.root / "drysua/target/release/drysua"))
        self.send("bota-client", b"0")
        self.assertIn("current Map2", self.finish(0))


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

    def test_ready_relay_frame_does_not_wait_twice_on_idle_log_selector(self):
        relay, human, server = self.socket_relay()
        human.sendall(self.ORDER)
        waits = []
        with tempfile.TemporaryDirectory(prefix="play-poll-", dir=TEMPORARY) as temporary:
            supervisor = self.launcher.Supervisor(Path(temporary))
            self.addCleanup(supervisor.close)
            supervisor.admission = SimpleNamespace(pacer=None, pump=relay.pump, close=relay.close)
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
            supervisor.admission = SimpleNamespace(pacer=None, pump=Mock(return_value=False), close=Mock())
            with patch.object(supervisor.selector, "select", return_value=[]) as selector:
                supervisor.pump(0.01)
            selector.assert_called_once_with(0.01)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        TEMPORARY.mkdir(parents=True, exist_ok=True)
        self.module = importlib.import_module("play_match")

    def supervisor(self):
        temporary = tempfile.TemporaryDirectory(prefix="play-unit-", dir=TEMPORARY)
        self.addCleanup(temporary.cleanup)
        supervisor = self.module.Supervisor(Path(temporary.name))
        self.addCleanup(supervisor.close)
        return supervisor

    def test_replay_limit_matches_bounded_local_budget(self):
        self.assertEqual(self.module.REPLAY_LIMIT, 2 * 1024**3)

    def test_no_arguments_select_human_radiant_bot_dire_and_no_weights_override(self):
        arguments = self.module.parse_arguments([])
        self.assertEqual(arguments.human_side, "radiant")
        self.assertEqual(arguments.bot_side, "dire")
        self.assertIsNone(arguments.weights_directory)
        self.assertEqual(arguments.port, 4455)
        self.assertEqual(arguments.seed, 9000001)
        self.assertFalse(arguments.build)
        self.assertFalse(arguments.no_build)

    def test_build_and_no_build_flags_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.module.parse_arguments(["--build", "--no-build"])
        self.assertEqual(raised.exception.code, 2)

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

    def test_signal_during_spawn_is_recorded_and_cleanup_stops_registered_child(self):
        supervisor = self.supervisor()
        original = subprocess.Popen

        def interrupted_spawn(*arguments, **keywords):
            process = original(*arguments, **keywords)
            supervisor.request_stop(signal.SIGINT, None)
            return process

        with patch.object(self.module.subprocess, "Popen", side_effect=interrupted_spawn):
            child = supervisor.spawn("build-bota", [sys.executable, "-c",
                                     "import signal; signal.pause()"], ROOT)
        self.assertEqual(supervisor.stop_status, 130)
        self.assertEqual(len(supervisor.children), 1)
        self.doCleanups()
        self.assertIsNotNone(child.process.returncode)

    def test_readiness_deadline_uses_monotonic_time_without_sleep(self):
        supervisor = self.supervisor()
        child = supervisor.spawn("server", [sys.executable, "-c", "import signal; signal.pause()"], ROOT)
        with patch.object(self.module.time, "monotonic", side_effect=[100, 111]):
            with self.assertRaisesRegex(RuntimeError, "readiness.*10"):
                supervisor.wait_ready(child, 0)

    def test_fragmented_and_final_drain_errors_cannot_become_success(self):
        cases = ((False, [b"bota-", b"cli", b"ent: failure"], 128, "bota-client reported an error"),
                 (True, [b"bota-client: late failure"], 128, "bota-client reported an error"),
                 (True, [b"x" * 65], 64, "log limit"))
        for final, chunks, limit, message in cases:
            with self.subTest(final=final, message=message):
                supervisor = self.supervisor()
                child = supervisor.spawn("client", [sys.executable, "-c", "import signal; signal.pause()"], ROOT)
                key = SimpleNamespace(fileobj=child.process.stderr, data=(child, False))
                with patch.object(supervisor.selector, "select", return_value=[(key, 1)]), \
                        patch.object(self.module.os, "read", side_effect=chunks), \
                        patch.object(self.module, "LOG_LIMIT", limit):
                    for _ in chunks[:-1]:
                        supervisor.pump(0, final=final)
                    with self.assertRaisesRegex(RuntimeError, message):
                        supervisor.pump(0, final=final)
                self.doCleanups()

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
            self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), expected)
            self.assertTrue(os.access(binary, os.X_OK))
        for mode, limit in ((0, 30), (1, 1000)):
            for human_slot in (0, 1):
                with self.subTest(mode=mode, human_slot=human_slot):
                    self.native_match(mode, limit, human_slot, server, bot, "teacher", 2)

    def native_match(self, mode, limit, human_slot, server_binary, bot_binary, policy, map_id):
        with tempfile.TemporaryDirectory(prefix="play-native-map2-", dir=TEMPORARY) as temporary:
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
