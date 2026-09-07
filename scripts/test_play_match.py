"""Local launcher regression tests; fixtures stay below artifacts/temp."""

import ctypes
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
TEMPORARY = ROOT / "drysua/artifacts/temp"
FIXTURE = r'''
import json, os, select, signal, socket, subprocess, sys
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
       target=os.environ.get("CARGO_TARGET_DIR"), cwd=os.getcwd())
if depth:
    descendant(role + "-grandchild", 0)
if role.startswith("build-") and os.environ.get("PLAY_TEST_BUILD") != "hold":
    sys.exit(0)
listener = None
prefix = False
for _ in range(32):
    readable, _, _ = select.select([control] + ([listener] if listener else []), [], [], 20)
    if not readable:
        raise RuntimeError("fixture control deadline exceeded")
    if listener in readable:
        raise RuntimeError("unexpected TCP readiness probe")
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
else:
    raise RuntimeError("fixture command limit exceeded")
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
        self.tools = self.root / "tools"
        self.tools.mkdir()
        for binary in (self.tools / "cargo", self.root / "bota/target/release/bota-server",
                       self.root / "bota/target/release/bota-client",
                       self.root / "drysua/target/release/drysua"):
            binary.write_text(f"#!{sys.executable} -B\n" + FIXTURE)
            binary.chmod(0o700)
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

    def launch(self, *arguments, **environment):
        self.environment.update(environment)
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

    def game(self, *arguments):
        self.launch("--port", "0", *arguments)
        self.send("bota-server", b"r")
        first = json.loads(self.child("bota-server")["stream"].readline(8192))
        self.send("bota-server", b"R")
        self.assertEqual(json.loads(self.child("bota-server")["stream"].readline(8192)), first)
        self.child("bota-client")
        self.child("drysua")
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

    def test_headless_fails_before_building_or_creating_logs(self):
        self.launch(DISPLAY="")
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 1, output.decode())
        self.assertIn(b"DISPLAY", error)
        self.assertFalse((self.root / "drysua/artifacts").exists())

    def test_help_is_available_without_display(self):
        self.launch("--help", DISPLAY="")
        output, error = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 0, error.decode())
        self.assertIn(b"--no-build", output)

    def test_builds_exact_release_targets_then_uses_real_banner_port_and_default_bot(self):
        port = self.game()
        for repository in ("bota", "drysua"):
            build = self.child("build-" + repository)
            self.assertEqual(build["target"], str(self.root / repository / "target"))
            expected = ["build", "--release", "--locked", "--quiet"]
            if repository == "drysua":
                expected.append("--no-default-features")
            expected += ["--manifest-path", str(self.root / repository / "Cargo.toml")]
            expected += (["-p", "bota-server", "-p", "bota-client", "--bin", "bota-server",
                          "--bin", "bota-client"] if repository == "bota" else ["--bin", "drysua"])
            self.assertEqual(build["arguments"], expected)
        self.assertEqual(self.child("drysua")["arguments"],
                         ["--addr", f"127.0.0.1:{port}", "--name", "drysua"])
        self.assertEqual(self.child("bota-client")["arguments"],
                         ["--addr", f"127.0.0.1:{port}", "--name", "human"])
        server = self.child("bota-server")["arguments"]
        self.assertEqual(server[:10], ["--port", "0", "--mode", "realtime", "--players", "2",
                                      "--map", "0", "--seed", "9000001"])
        self.assertEqual(Path(server[-1]).parent.parent, self.root / "drysua/artifacts/temp")
        self.send("bota-client", b"0")
        self.finish(0)
        self.assert_children_stopped()

    def test_no_build_ignores_inherited_target_and_game_completion_keeps_results(self):
        self.game("--no-build")
        self.assertNotIn("build-bota", self.children)
        self.send("bota-server", b"0")
        self.send("drysua", b"0")
        for name in ("bota-server", "drysua"):
            self.assertTrue(select.select([self.child(name)["pidfd"]], [], [], 3)[0])
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
                self.assertIn("exited", self.finish(1))
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
        (self.root / "drysua/target/release/drysua").unlink()
        self.launch("--no-build")
        self.assertIn("build", self.finish(1))

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
        self.assertFalse((self.root / "drysua/artifacts").exists())


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        TEMPORARY.mkdir(parents=True, exist_ok=True)
        self.module = importlib.import_module("play_match")

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
            server = SimpleNamespace(process=SimpleNamespace(pid=123), exit_status=lambda: 0)
            client = SimpleNamespace(exit_status=lambda: None)
            bot = SimpleNamespace(exit_status=lambda: 0)
            try:
                with patch.object(supervisor, "spawn", side_effect=[server, client, bot]), \
                        patch.object(supervisor, "wait_ready", return_value=4455), \
                        patch.object(self.module.resource, "prlimit"), \
                        patch.object(Path, "is_file", return_value=True), \
                        patch.object(self.module.os, "access", return_value=True), \
                        patch.object(supervisor, "pump", side_effect=[None, RuntimeError("GUI still active")]):
                    with self.assertRaisesRegex(RuntimeError, "GUI still active"):
                        supervisor.run(ROOT, SimpleNamespace(no_build=True, port=4455, seed=9000001))
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
                                           "--map", "0", "--seed", "9000001"], ROOT)
                port = supervisor.wait_ready(server, 0)
                self.assertGreater(port, 0)
                self.assertLessEqual(port, 65535)
                self.assertIsNone(server.exit_status())
                self.assertIn(f"bota-server listening on 0.0.0.0:{port}\n".encode(),
                              (Path(temporary) / "server.log").read_bytes())
            finally:
                supervisor.close()
            self.assertIn(server.process.returncode, (-signal.SIGTERM, -signal.SIGKILL))

    @unittest.skipUnless(all((ROOT / path).is_file() for path in (
        "bota/target/release/bota-server", "drysua/target/release/drysua")),
        "optional native integration requires both release binaries; build them before running")
    def test_native_default_deployment_bots_reach_1000_ticks_without_rejections(self):
        """Uses real TCP and fresh release binaries, with no builds or graphical client."""
        with tempfile.TemporaryDirectory(prefix="play-native-default-", dir=TEMPORARY) as temporary:
            supervisor = self.module.Supervisor(Path(temporary))
            children, statuses, failure = [], None, None
            previous = {}
            try:
                previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    previous[number] = signal.signal(number, supervisor.request_stop)
                server = supervisor.spawn("server", [str(ROOT / "bota/target/release/bota-server"),
                                           "--port", "0", "--mode", "lockstep", "--players", "2",
                                           "--map", "0", "--seed", "9000001",
                                           "--ack-timeout-ticks", "900"], ROOT)
                children.append(server)
                port = supervisor.wait_ready(server, 0)
                for index in range(2):
                    name = f"bot-{index}"
                    children.append(supervisor.spawn(name, [str(ROOT / "drysua/target/release/drysua"),
                                               "--addr", f"127.0.0.1:{port}", "--name", name,
                                               "--limit", "1000"], ROOT))
                deadline = self.module.time.monotonic() + 30
                for _ in range(3000):
                    supervisor.check_stop()
                    supervisor.pump(0.01)
                    statuses = [child.exit_status() for child in children]
                    if all(status is not None for status in statuses):
                        break
                    if any(status not in (None, 0) for status in statuses):
                        break
                    if self.module.time.monotonic() >= deadline:
                        self.fail("native lockstep match exceeded its 30-second deadline")
                else:
                    self.fail("native lockstep supervision iteration limit exceeded")
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
                with (Path(temporary) / f"{child.name}.log").open(errors="replace") as log:
                    logs[child.name] = log.read(4096)
            diagnostic = ("Fresh release builds are required; rebuild bota-server and drysua from the "
                          "current working trees before this test. No policy/weights override is supplied.\n"
                          + "\n".join(f"{name}:\n{text}" for name, text in logs.items()))
            self.assertIsNone(failure, f"{failure}\n{diagnostic}")
            self.assertEqual(statuses, [0, 0, 0], diagnostic)
            teams = set()
            for name in ("bot-0", "bot-1"):
                self.assertRegex(logs[name], r"(?m)^deployment: repository-selected [^\r\n]+$", diagnostic)
                summary = self.assert_native_summary(logs[name], diagnostic)
                teams.add(summary)
            self.assertEqual(teams, {"Radiant", "Dire"}, diagnostic)

    def assert_native_summary(self, log, diagnostic):
        pattern = (r"(?m)^played 1000 ticks as Some\((Radiant|Dire)\); winner None; "
                   r"[0-9]+ decisions, [0-9]+ orders, 0 rejected orders$")
        self.assertRegex(log, pattern, diagnostic)
        return "Radiant" if "played 1000 ticks as Some(Radiant)" in log else "Dire"


if __name__ == "__main__":
    unittest.main()
