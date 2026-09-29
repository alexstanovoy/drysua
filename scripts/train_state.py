"""Linux ownership and bounded child I/O; never discovers or signals containers."""
from contextlib import contextmanager
import fcntl
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
