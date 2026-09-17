"""Bounded passive reward pipes; observation failures do not take over player input."""

import json
import socket
import time

from play_admission import QUEUE_LIMIT, READ_CHUNK, PROGRESS_TIMEOUT


class RewardPipe:
    limit = QUEUE_LIMIT

    def __init__(self, connection, prefix, role):
        self.connection, self.prefix, self.role = connection, prefix, role
        connection.setblocking(False)
        self.pending = bytearray()
        self.failure = None
        self.persistence_error = None
        self.ending = self.closed = False
        self.progress = time.monotonic()
        self.failure_path = prefix.with_suffix(".invalid.json")

    def offer(self, frame):
        if self.failure or self.closed:
            return
        self.pump()
        if self.failure:
            return
        if len(self.pending) + len(frame) > self.limit:
            self.fail("observer queue limit exceeded; copied stream INCOMPLETE")
            return
        if not self.pending:
            self.progress = time.monotonic()
        self.pending.extend(frame)
        self.pump()

    def pump(self):
        if self.closed:
            return
        if self.pending:
            try:
                count = self.connection.send(self.pending[:READ_CHUNK])
                if not count:
                    raise OSError("zero write progress")
                del self.pending[:count]
                self.progress = time.monotonic()
            except BlockingIOError:
                if time.monotonic() - self.progress >= PROGRESS_TIMEOUT:
                    self.fail("observer write timeout; copied stream INCOMPLETE")
            except OSError as error:
                self.fail(f"observer write failed: {error}; copied stream INCOMPLETE")
        if self.ending and not self.pending:
            self.close()
        assert len(self.pending) <= self.limit

    def finish(self):
        self.ending = True
        self.pump()

    def fail(self, reason):
        if self.failure:
            return
        self.failure = reason
        self.pending.clear()
        self.close()
        report = {"complete": False, "valid": False, "role": self.role, "error": reason}
        print(f"play: {self.role} reward INVALID/INCOMPLETE: {reason}", flush=True)
        try:
            with self.failure_path.open("x") as output:
                json.dump(report, output)
        except OSError as error:
            self.note_persistence_error(error)

    def note_persistence_error(self, error):
        self.persistence_error = str(error)[:1024]
        print(f"play: {self.role} reward persistence failed: {self.persistence_error}; "
              "observation remains INVALID in memory; do not trust an uncorrected report file", flush=True)

    def close(self):
        if not self.closed:
            self.connection.close()
            self.closed = True


def start_observers(supervisor, binary, root, arguments):
    for relay in supervisor.admission.relays:
        prefix = supervisor.directory / f"reward-{relay.role}"
        source, target = socket.socketpair()
        try:
            child = supervisor.spawn(f"reward-{relay.role}", [str(binary), "reward-observer",
                "--output", str(prefix.with_suffix(".json")), "--interval-ticks",
                str(arguments.reward_interval)], root, input_stream=target)
        except BaseException:
            source.close()
            raise
        finally:
            target.close()
        label = "human" if relay.role == "human" else arguments.opponent
        pipe = RewardPipe(source, prefix, label)
        supervisor.reward_pipes.append((pipe, child, relay))
        relay.observer = pipe.offer
    print(f"play: passive per-seat reward reports: {supervisor.directory}/reward-*.json", flush=True)


def pump_observers(supervisor):
    for pipe, child, relay in supervisor.reward_pipes:
        status = child.exit_status()
        if status not in (None, 0) and not pipe.failure:
            pipe.fail("observer rejected stream; see observer log and partial JSON")
        elif status == 0 and not pipe.ending and not pipe.failure:
            pipe.fail("observer exited before copied stream completion")
        if relay.match_over or (relay.endpoints and relay.endpoints[1].eof):
            pipe.finish()
        pipe.pump()
    if (supervisor.reward_pipes and not supervisor.reward_overview
            and all(child.exit_status() is not None for _, child, _ in supervisor.reward_pipes)):
        print_overview(supervisor.reward_pipes)
        supervisor.reward_overview = True


def finish_observers(supervisor):
    if not supervisor.reward_pipes:
        return
    for pipe, _, _ in supervisor.reward_pipes:
        pipe.finish()
    deadline = time.monotonic() + 2
    for _ in range(200):
        if all(child.exit_status() is not None for _, child, _ in supervisor.reward_pipes):
            break
        if time.monotonic() >= deadline:
            break
        for pipe, _, _ in supervisor.reward_pipes:
            pipe.pump()
        supervisor.pump(0.01, final=True)
    for pipe, child, _ in supervisor.reward_pipes:
        if child.exit_status() is None:
            pipe.fail("observer finalization timeout; report INCOMPLETE")
        elif child.exit_status() != 0 and not pipe.failure:
            pipe.fail("observer rejected stream; see observer log and partial JSON")
        pipe.close()
    if not supervisor.reward_overview or any(pipe.failure for pipe, _, _ in supervisor.reward_pipes):
        print_overview(supervisor.reward_pipes)
        supervisor.reward_overview = True


def print_overview(pipes):
    reports = []
    for pipe, _, _ in pipes:
        path = pipe.prefix.with_suffix(".json")
        try:
            with path.open("rb") as stream:
                data = stream.read(65537)
            if len(data) > 65536:
                raise ValueError("final report exceeds 64 KiB")
            report = json.loads(data)
            if not isinstance(report, dict):
                raise ValueError("expected report object")
        except (OSError, ValueError) as error:
            print(f"play: {pipe.role} reward INCOMPLETE: {error}", flush=True)
            continue
        valid = report.get("valid") is True and not pipe.failure
        if pipe.failure:
            report.setdefault("observer_error", report.get("error"))
            report.update(valid=False, complete=False, error=pipe.failure)
            try:
                with path.open("w") as stream:
                    json.dump(report, stream)
            except OSError as error:
                pipe.note_persistence_error(error)
        label = "COMPLETE" if valid else "INVALID/INCOMPLETE"
        print(f"play: {pipe.role} {report.get('team')} reward {label}; "
              f"outcome={report.get('outcome')} ticks={report.get('ticks')} report={path}", flush=True)
        reports.append((pipe.role, report))
    if reports:
        print("reward category".ljust(26) + " ".join(role.rjust(16) for role, _ in reports), flush=True)
        names = list(reports[0][1].get("components", {}))[:17] + ["total_without_terminal", "total"]
        for name in names:
            values = [report.get("components", {}).get(name, report.get(name, 0)) for _, report in reports]
            print(name.ljust(26) + " ".join(f"{value:16.9f}" for value in values), flush=True)


def print_interval(child, data):
    child.tail += data
    if len(child.tail) > 65536:
        raise RuntimeError("observer stdout line exceeds 64 KiB")
    for _ in range(64):
        line, separator, remaining = child.tail.partition(b"\n")
        if not separator:
            break
        child.tail = remaining
        try:
            report = json.loads(line)
        except ValueError as error:
            raise RuntimeError("observer emitted malformed JSON") from error
        if not isinstance(report, dict):
            raise RuntimeError("observer emitted JSON other than an object")
        if report.get("kind") == "interval":
            print(f"play: {child.name} tick={report['ticks']} running reward={report['total']:.9f} "
                  "(partial, no terminal yet; category deltas in JSONL)", flush=True)
