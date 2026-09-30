"""Bounded file, JSON, lock, process and subprocess helpers of the training controller."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time

MAX_JSON = 4 * 1024 * 1024


class LockBusy(ValueError):
    pass


def private_path(value):
    """Absolute path without parent traversal or symlinked components."""
    path = Path(os.path.abspath(value))
    if ".." in Path(value).parts:
        raise ValueError("parent traversal is forbidden")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink is forbidden: {part}")
    return path


def read_bytes(path, limit):
    path = private_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path}")
        value = source.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"file exceeds size limit: {path}")
    return value


def read_kernel(path, limit):
    """One bounded read of a /proc or /sys pseudo-file, whose paths legitimately traverse symlinks."""
    if not str(path).startswith(("/proc/", "/sys/")):
        raise ValueError(f"not a kernel pseudo-file: {path}")
    with open(path, "rb", buffering=0) as source:
        value = source.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"kernel file exceeds size limit: {path}")
    return value


def read_json(path, limit=MAX_JSON):
    return decode_json(read_bytes(path, limit))


def decode_json(data):
    def unique_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    try:
        value = json.loads(data, object_pairs_hook=unique_pairs,
                           parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded_integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return value


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path, data, mode=0o400):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(data)
        target.flush()
        os.fchmod(target.fileno(), mode)
        os.fsync(target.fileno())
    fsync_directory(path.parent)


def atomic_json(path, value, mode=0o600):
    temporary = path.with_name(path.name + ".pending")
    if temporary.exists():
        temporary.unlink()
    write_exclusive(temporary, encode(value), mode)
    os.replace(temporary, path)
    fsync_directory(path.parent)


@contextmanager
def locks(paths):
    """Exclusive nonblocking flocks on existing regular files, held for the block."""
    descriptors = []
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            descriptors.append(descriptor)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"lock is not a regular file: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LockBusy(f"lock busy: {path}") from error
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


def process_identity(pid):
    """PID plus kernel start tick and boot ID, so a reused PID never matches."""
    if type(pid) is not int or pid < 1:
        raise ValueError("invalid process ID")
    path = Path(f"/proc/{pid}")
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": fields[19], "uid": path.stat().st_uid,
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_live(identity):
    if not isinstance(identity, dict) or not {"pid", "start", "uid", "boot"} <= identity.keys():
        raise ValueError("invalid process identity")
    current = process_identity(identity["pid"])
    return current is not None and all(current[key] == identity[key] for key in current)


def environment():
    """Minimal child environment: no inherited secrets, no accidental GPU use."""
    result = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR") if key in os.environ}
    result["CUDA_VISIBLE_DEVICES"] = ""
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def terminate(child, grace=5):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=grace)


def capture(command, timeout=10, limit=1024 * 1024):
    """Stdout of one command; stdout and stderr share one memory bound and one deadline."""
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
                    raise ValueError(f"command timeout: {command[0]}")
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    output[key.data].extend(data)
                    if sum(map(len, output)) > limit:
                        raise ValueError(f"command output limit exceeded: {command[0]}")
        try:
            code = child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"command timeout: {command[0]}") from error
        if code != 0:
            message = bytes(output[1][:512]).decode("utf-8", "replace").strip()
            raise ValueError(f"{Path(command[0]).name} failed with exit {code}: {message}")
        return bytes(output[0])
    finally:
        terminate(child)
        child.stdout.close()
        child.stderr.close()
