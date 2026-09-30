"""Linux ownership and bounded child I/O; never discovers or signals containers."""
from contextlib import contextmanager
import fcntl
import io
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import threading
import time


class LockBusy(ValueError):
    pass


def environment():
    result = {key: os.environ[key] for key in
              ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR") if key in os.environ}
    result["CUDA_VISIBLE_DEVICES"] = ""
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def process_identity(pid):
    if type(pid) is not int or pid < 1:
        raise ValueError("invalid owner PID")
    path = Path(f"/proc/{pid}")
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": fields[19], "uid": path.stat().st_uid,
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        return None


def is_live(identity):
    if not isinstance(identity, dict) or not {"pid", "start", "uid", "boot"} <= identity.keys():
        raise ValueError("invalid owner identity")
    current = process_identity(identity["pid"])
    return current is not None and all(current[key] == identity[key] for key in current)


@contextmanager
def locks(paths):
    descriptors = []
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            descriptors.append(descriptor)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"lock is not a regular file: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LockBusy(f"live owner or lock busy: {path}") from error
        yield
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def is_locked(path):
    try:
        with locks([path]):
            return False
    except LockBusy:
        return True


def terminate_child(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def reap_detached(child):
    # Keep the Popen alive and reap it for library callers; CLI exit reparents it.
    threading.Thread(target=child.wait, name="training-controller-reaper", daemon=True).start()


def capture(command, timeout=10, limit=1024 * 1024):
    """Drain both pipes with one aggregate memory bound, including stderr."""
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=environment(), close_fds=True)
    output = [bytearray(), bytearray()]
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, 0)
            selector.register(child.stderr, selectors.EVENT_READ, 1)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("inspector timeout")
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    output[key.data].extend(data)
                    if sum(map(len, output)) > limit:
                        raise ValueError("inspector output limit exceeded")
        try:
            code = child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise ValueError("inspector timeout") from error
        if code != 0:
            message = bytes(output[1][:512]).decode("utf-8", "replace")
            raise ValueError(f"inspector failed with exit {code}: {message}")
        return bytes(output[0])
    finally:
        terminate_child(child)
        child.stdout.close()
        child.stderr.close()


@contextmanager
def stop_signals():
    requested = threading.Event()
    previous = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, lambda *_: requested.set())
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def terminate_runner(child, deadline):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        # The runner may need its original deadline to finish owned-container cleanup.
        child.wait(timeout=max(0, deadline - time.monotonic() - 5))
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise ValueError("runner cleanup deadline exceeded; recovery required") from error


def wait_runner(command, timeout, stop_requested, record_child, controller_reserve_seconds):
    if (type(timeout) is not int or type(controller_reserve_seconds) is not int or
            timeout <= 0 or controller_reserve_seconds - timeout < 5):
        raise ValueError("runner timeout and controller reserve must be positive integer budgets with reap headroom")
    if stop_requested():
        raise ValueError("immediate stop requested before runner launch; recovery required")
    started = time.monotonic()
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=environment(), close_fds=True)
    deadline = started + timeout
    try:
        identity = process_identity(child.pid)
        if identity is None:
            raise ValueError("runner exited before its ownership could be recorded")
        record_child(identity)
        while child.poll() is None:
            if stop_requested():
                raise ValueError("immediate stop requested; recovery required")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("runner deadline exceeded; recovery required")
            threading.Event().wait(min(remaining, 0.05))
        if child.returncode != 0:
            raise ValueError(f"runner exited with {child.returncode}; recovery required")
    finally:
        # Only this Popen child is signalled. Container cleanup belongs to the runner.
        terminate_runner(child, started + controller_reserve_seconds)


EPISODE_OUTCOMES = ("Win", "Loss", "Draw")
MAX_REPORT_UPDATES = 10000


def episode_outcomes(payload, expected):
    """Terminal Map2 outcomes of one bounded native payload log, in file order.

    Exactly `expected` episodes must be present: a short or over-long log must
    fail loudly instead of feeding zeros into a win rate.
    """
    if type(expected) is not int or expected < 1:
        raise ValueError("expected episode count must be a positive integer")
    outcomes = []
    for line in io.BytesIO(payload):
        if not line.startswith(b"episode: "):
            continue
        outcome = None
        for field in line.split():
            if field.startswith(b"outcome="):
                outcome = field[len(b"outcome="):].decode("ascii", "replace")
                break
        if outcome is None:
            raise ValueError("episode line without an outcome field")
        if outcome not in EPISODE_OUTCOMES:
            raise ValueError(f"unsupported episode outcome: {outcome}")
        outcomes.append(outcome)
        if len(outcomes) > expected:
            raise ValueError(f"payload log records more than {expected} terminal episodes")
    if len(outcomes) != expected:
        raise ValueError(f"payload log records {len(outcomes)} terminal episodes, expected {expected}")
    return outcomes


def win_record(outcomes):
    """Wins, losses, draws and win rate for one bounded outcome batch."""
    if len(outcomes) > MAX_REPORT_UPDATES * 64:
        raise ValueError("outcome batch exceeds the report bound")
    wins = outcomes.count("Win")
    losses = outcomes.count("Loss")
    draws = outcomes.count("Draw")
    if wins + losses + draws != len(outcomes):
        raise ValueError("unsupported outcome in report batch")
    return {"games": len(outcomes), "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(wins / len(outcomes), 4) if outcomes else None}


def add_wins(records):
    """Sums win records without re-scanning their episodes."""
    wins = sum(record["wins"] for record in records)
    losses = sum(record["losses"] for record in records)
    draws = sum(record["draws"] for record in records)
    games = wins + losses + draws
    return {"games": games, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(wins / games, 4) if games else None}


def update_records(invocation_outcomes, games_per_update):
    """One record per accepted update; each invocation covers whole updates."""
    if type(games_per_update) is not int or games_per_update < 1:
        raise ValueError("games per update must be a positive integer")
    updates = []
    for outcomes in invocation_outcomes:
        if len(outcomes) % games_per_update:
            raise ValueError("invocation outcome count is not a whole number of updates")
        for start in range(0, len(outcomes), games_per_update):
            record = win_record(outcomes[start:start + games_per_update])
            record["update"] = len(updates) + 1
            updates.append(record)
    if len(updates) > MAX_REPORT_UPDATES:
        raise ValueError("report exceeds the bounded update count")
    return updates


def block_records(updates, block):
    """Fixed-size update blocks; only the last block may be shorter."""
    if type(block) is not int or not 1 <= block <= MAX_REPORT_UPDATES:
        raise ValueError("report block must be within 1..=10000")
    records = []
    for start in range(0, len(updates), block):
        chunk = updates[start:start + block]
        record = add_wins(chunk)
        record["first_update"] = chunk[0]["update"]
        record["updates"] = len(chunk)
        records.append(record)
    return records


def recent_record(updates, count):
    """Win record for the last `count` updates, or None when there are fewer."""
    return add_wins(updates[-count:]) if len(updates) >= count else None


def mean_seconds(values):
    """Mean of the bounded positive seconds that are actually available."""
    usable = [value for value in values if type(value) is float and 0 < value < 86400]
    return round(sum(usable) / len(usable), 3) if usable else None


def games_per_update(inspection):
    """Games per update from the accepted inspection, with a run-scope fallback."""
    environments = inspection.get("ppo", {}).get("environments")
    if type(environments) is int and environments > 0:
        return environments
    command = inspection.get("run", {}).get("command_line")
    if isinstance(command, str):
        fields = command.split()
        for index, field in enumerate(fields[:-1]):
            if field == "--games":
                value = fields[index + 1]
                if value.isdigit() and int(value) > 0:
                    return int(value)
    raise ValueError("accepted inspection reports neither ppo environments nor a --games run option")


TRANSITION_EVENT = b"event=adaptive_environment_transition "
TRANSITION_FIELDS = ("update", "previous_generation", "generation", "start_update",
                     "previous_awards", "clean")
MAX_ENVIRONMENT_TRANSITIONS = 10000


def transition_fields(line):
    """Exact named fields of one transition line; duplicates are corruption."""
    fields = {}
    for token in line.split():
        name, separator, value = token.partition(b"=")
        if not separator:
            continue
        name = name.decode("ascii", "replace")
        if name not in TRANSITION_FIELDS:
            continue
        if name in fields:
            raise ValueError(f"duplicate environment transition field: {name}")
        fields[name] = value.decode("ascii", "replace")
    if set(fields) != set(TRANSITION_FIELDS):
        missing = sorted(set(TRANSITION_FIELDS) - set(fields))
        raise ValueError(f"environment transition line is missing fields: {missing}")
    return fields


def environment_transitions(payload):
    """Ordered adaptive environment transitions from one bounded payload log.

    A generation change resets the running generation, so the recorded
    `start_update` must equal the transition update.
    """
    transitions = []
    for line in io.BytesIO(payload):
        if TRANSITION_EVENT not in line:
            continue
        fields = transition_fields(line)
        record = {}
        for name in TRANSITION_FIELDS[:-1]:
            value = fields[name]
            if not value.isdigit():
                raise ValueError(f"environment transition {name} must be a non-negative integer")
            record[name] = int(value)
        clean = fields["clean"]
        if clean not in ("true", "false"):
            raise ValueError("environment transition clean must be true or false")
        record["clean"] = clean == "true"
        if record["update"] < 1 or record["update"] != record["start_update"]:
            raise ValueError("environment transition update and start_update must match")
        if record["generation"] != record["previous_generation"] + 1:
            raise ValueError("environment transition generation must follow the previous one")
        transitions.append(record)
        if len(transitions) > MAX_ENVIRONMENT_TRANSITIONS:
            raise ValueError("payload log exceeds the bounded environment transition count")
    return transitions


def environment_summary(transitions, generation, accepted_updates, base_updates, current_awards):
    """Environment-transition and extension totals for one adaptive campaign.

    `transitions` are the parsed transition records, `generation` is the
    checkpoint's current generation, and `current_awards` is that generation's
    award count. Played updates of a generation come from the transition starts,
    which is what separates an early success transition from a clean boundary.
    """
    for name, value in (("generation", generation), ("accepted updates", accepted_updates),
                        ("environment extension awards", current_awards)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if type(base_updates) is not int or base_updates < 1:
        raise ValueError("environment base updates must be a positive integer")
    if len(transitions) != generation:
        raise ValueError(f"payload logs record {len(transitions)} environment transitions "
                         f"but the checkpoint reports generation {generation}")
    starts = [record["start_update"] for record in transitions]
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("environment transition starts are not strictly increasing")
    played, previous = [], 0
    for start in starts:
        played.append(start - previous)
        previous = start
    current_start = starts[-1] if starts else 0
    current = 1 if accepted_updates > current_start else 0
    if current:
        played.append(accepted_updates - current_start)
    skipped = [record["update"] for record, spent in zip(transitions, played)
               if spent < base_updates and not record["clean"]]
    truncated = [record["update"] for record, spent in zip(transitions, played)
                 if spent < base_updates and record["clean"]]
    awards = [record["previous_awards"] for record in transitions]
    if current:
        awards.append(current_awards)
    return {
        "environments": {
            "total": len(played),
            "completed": len(transitions),
            "current": current,
            "skipped_early": skipped,
            "clean_truncated": truncated,
            "spent": {"min": min(played, default=None), "max": max(played, default=None),
                      "mean": round(sum(played) / len(played), 2) if played else None},
        },
        "extensions": {
            "environments": sum(1 for award in awards if award > 0),
            "awards_total": sum(awards),
            "extra_updates_total": sum(max(0, spent - base_updates) for spent in played),
        },
    }
