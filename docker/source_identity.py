"""Bounded inventory of the allowlisted image sources, not a claimed Git revision."""

import hashlib
import json
import os
from pathlib import Path
import stat
import sys


def inventory(root):
    entries = []
    total = 0
    directories = [root]
    for _ in range(4096):
        if not directories:
            break
        directory = directories.pop()
        with os.scandir(directory) as children:
            for index, child in enumerate(children):
                if index >= 4096 or len(entries) + len(directories) >= 4096:
                    raise ValueError("source inventory entry limit exceeded")
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError("source inventory refuses symlink")
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(Path(child.path))
                    continue
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4 * 1024 ** 2:
                    raise ValueError("source input must be a regular file <=4 MiB")
                total += metadata.st_size
                if total > 64 * 1024 ** 2:
                    raise ValueError("source inventory exceeds 64 MiB")
                data = Path(child.path).read_bytes()
                if len(data) != metadata.st_size:
                    raise ValueError("source changed during inventory")
                entries.append([str(Path(child.path).relative_to(root)),
                                stat.S_IMODE(metadata.st_mode), hashlib.sha256(data).hexdigest()])
    if directories or not entries:
        raise ValueError("source inventory incomplete or empty")
    return sorted(entries)


def main():
    root = Path("/src")
    entries = inventory(root)
    encoded = json.dumps(entries, separators=(",", ":")).encode()
    source = hashlib.sha256(encoded).hexdigest()
    simulator = json.dumps([entry for entry in entries if entry[0].startswith("bota/")],
                           separators=(",", ":")).encode()
    destination = Path("/opt/drysua")
    destination.mkdir(exist_ok=True)
    (destination / "sources.json").write_bytes(encoded)
    (destination / "source.sha256").write_text(source + "\n")
    (destination / "source.env").write_text(
        f"export DRYSUA_GIT_COMMIT=source-sha256-{source}\n"
        f"export BOTA_GIT_COMMIT=source-sha256-{hashlib.sha256(simulator).hexdigest()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
