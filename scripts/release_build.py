"""Build immutable git archives with the original sibling path dependencies."""

import json
import os
import re
import subprocess
import tarfile

SIMULATOR = "18db0f62d9a2b94e755c43fd29a959db204cc20b"


def git(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments],
                                   text=True, timeout=30).strip()


def read_registry(path, repository, simulator_repository):
    registry = json.loads(path.read_text())
    if registry["schema_version"] != 1 or registry["simulator_commit"] != SIMULATOR:
        raise ValueError("unsupported registry schema or simulator identity")
    if registry["map"] != 1 or registry["policy"] != "teacher":
        raise ValueError("unsupported map or policy")
    if registry["gate"] != "candidate_wins * 2 > all_scheduled_games_per_opponent":
        raise ValueError("unsupported release gate")
    seeds = registry["seeds"]
    if not 10 <= len(seeds) <= 100 or len(set(seeds)) != len(seeds):
        raise ValueError("registry requires 10..100 unique seeds")
    if any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("invalid seed")
    if not 1 <= registry["tick_limit"] <= 100000:
        raise ValueError("tick limit must be 1..100000")
    if not 1 <= registry["process_timeout_seconds"] <= 600:
        raise ValueError("process timeout must be 1..600 seconds")
    releases = registry["releases"]
    if not 1 <= len(releases) <= 100:
        raise ValueError("registry requires 1..100 releases")
    tags = [release["tag"] for release in releases]
    actual = git(repository, "tag", "--list", "v*").splitlines()
    if len(set(tags)) != len(tags) or set(tags) != set(actual):
        raise ValueError("release tags and registry differ; no historical release may be skipped")
    for release in releases:
        tag = release["tag"]
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("invalid release tag")
        if git(repository, "cat-file", "-t", f"refs/tags/{tag}") != "tag":
            raise ValueError(f"release {tag} must be an annotated tag")
        if git(repository, "rev-parse", f"refs/tags/{tag}^{{commit}}") != release["commit"]:
            raise ValueError(f"release {tag} commit identity mismatch")
        if (release["simulator_commit"], release["map"], release["policy"]) != (SIMULATOR, 1, "teacher"):
            raise ValueError(f"release {tag} has unsupported simulator/map/policy")
    if git(simulator_repository, "rev-parse", f"{SIMULATOR}^{{commit}}") != SIMULATOR:
        raise ValueError("simulator commit identity mismatch")
    return registry


def archive(repository, commit, destination, log_directory):
    destination.mkdir(parents=True)
    archive_path = log_directory / f"{destination.name}-{commit}.tar"
    with archive_path.open("wb") as stream:
        subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", commit],
                       stdout=stream, check=True, timeout=60)
    with tarfile.open(archive_path) as source:
        members = source.getmembers()
        if len(members) > 10000 or sum(member.size for member in members) > 256 * 1024 * 1024:
            raise ValueError("source archive exceeds limits")
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError("source archive contains a link or special file")
        source.extractall(destination, filter="data")


def build(source, target, arguments, output, name):
    command = ["cargo", "build", "--release", "--locked", "--quiet", *arguments]
    environment = os.environ.copy()
    # Do not inherit a concurrent agent's target or compiler-instrumentation settings.
    for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        environment.pop(key, None)
    environment["CARGO_TARGET_DIR"] = str(target)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    metadata = dict(command=command, cwd=str(source), target=str(target),
                    rustc=subprocess.check_output(["rustc", "-Vv"], cwd=source, text=True, timeout=60),
                    cargo=subprocess.check_output(["cargo", "-V"], cwd=source, text=True, timeout=60))
    (output / f"{name}-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / f"{name}-build.log").open("wb") as log:
        subprocess.run(command, cwd=source, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1200)


def prepare(repository, simulator_repository, output, registry):
    source = output / "sources"
    archive(simulator_repository, SIMULATOR, source / "bota", output)
    build(source / "bota", output / "target-server", ["-p", "bota-server"], output, "server")
    opponents = {}
    for release in registry["releases"]:
        tag = release["tag"]
        # Every archived bot sees ../bota, never the mutable checkout.
        archive(repository, release["commit"], source / tag, output)
        target = output / f"target-{tag}"
        build(source / tag, target, ["--bin", "drysua", "--no-default-features"], output, tag)
        opponents[tag] = target / "release" / "drysua"
    return output / "target-server" / "release" / "bota-server", opponents
