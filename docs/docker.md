# Docker deployment: prepared, not runtime-qualified

Docker daemon access remains denied. No image build, container start or migration
is established by native Rust/Python qualification. Training is OFF; existing
monitoring stays in place. Do not bypass socket permissions, escalate privileges,
control host services, or replay historical launchers. Future deployment requires
authorized daemon access, bounded builder/runtime qualification and an explicit
data/port handoff. This document does not authorize execution now.

## Stack and security

[Root Compose](../compose.yml) extends [monitoring Compose](../monitoring/compose.yml)
and imports its two secrets; do not duplicate services or replace this with an
unaudited include/merge. Default startup selects monitoring only; training requires
the explicit `training` profile. Root restart policy is **no** for every service.

| Service | Network/listener | Hard memory / PIDs |
|---|---|---|
| training | none | 12 GiB / 1024 |
| metrics | host, 127.0.0.1:9464 | 256 MiB / 32 |
| prometheus | host, 127.0.0.1:9090 | 512 MiB / 128 |
| grafana | host, 127.0.0.1:3000 | 512 MiB / 128 |

Non-root UID/GID (default 1000), read-only root, dropped capabilities,
no-new-privileges, no extra swap, bounded tmpfs/logs; no Docker socket, privileged
mode, host PID namespace, CPU quota, affinity or global thread limit. Native services
use init and one explicit NVIDIA UUID (`compute,utility`); learner ordinal is zero.
The CUDA-linked exporter needs matching libraries/device access but starts no learner.
Host networking must mean the local Linux host: Desktop, remote daemons and rootless
namespace translation are unqualified. Never widen loopback listeners. Healthchecks
are observational; HTTP 200 alone does not establish healthy persisted metrics.

## Configuration and accepted data

Use [.env.example](../.env.example) as a variable schema, not deployable values.
Keep real configuration in ignored `docker/local.env`; never log its expanded secrets.

| Variables | Contract |
|---|---|
| `DRYSUA_IMAGE`, `CUDA_COMPUTE_CAP`, `GPU_UUID` | Approved local image, selected GPU capability and exact UUID; no opportunistic discovery |
| `DRYSUA_UID/GID`, `MONITORING_UID/GID` | Match owned data and readable secret files; no root chown/init service |
| `TRAINING_DIRECTORY` | New private accepted copy, never live campaign state |
| `INITIAL_WEIGHTS_DIRECTORY`, `PROVENANCE_DIRECTORY` | Immutable weights and truthful image/source evidence plus acceptance.json |
| `GLOBAL_HEAVY_LOCK` | Original shared lock inode, not a copied/new substitute |
| `CPU_HWMON_DIRECTORY` | Resolved host k10temp/coretemp directory, not a broken relocated symlink |
| `TRAIN_UPDATES/GAMES/PARALLEL/GENERATION_GAMES/ZERO_UPDATES/SEED` | Defaults 1000/40/40/200/200/9001; receipt must agree |
| `INVOCATION_SECONDS` | Default 300, including telemetry/shutdown reserve |

Monitoring data, secret-file and port variables are defined in
[monitoring operations](monitoring.md). All bind sources must already exist;
Compose never creates or repairs them. Keep checkpoint and metrics directories
separate siblings. Preserve all randomization generations and model+Adam+RNG together.

```text
TRAINING_DIRECTORY/              0700, owned by DRYSUA_UID
  checkpoint/                   accepted copy or empty fresh directory
  metrics/                      matching journal or fresh coverage
    .metrics.writer.lock        precreated regular 0600 file
  logs/                         empty when fresh; retain prior evidence on resume
PROVENANCE_DIRECTORY/
  acceptance.json               reviewed receipt; mounted read-only
INITIAL_WEIGHTS_DIRECTORY/       approved immutable runtime; mounted read-only
```

Exporter mounts metrics read-only except the same writer-lock file for its liveness
probe. Fresh metrics may contain only that empty lock. See [journal recovery](metrics_schema.md).
A resume receipt has this shape; placeholders and update 180 are **not authorization**:

```json
{
  "source_sha256": "ACTUAL_IMAGE_SOURCE_SHA256",
  "binary_sha256": "ACTUAL_IMAGE_BINARY_SHA256",
  "gpu_uuid": "ACTUAL_GPU_UUID",
  "mode": "resume",
  "starting_update": 180,
  "checkpoint_meta_sha256": "ACCEPTED_COPY_MANIFEST_SHA256",
  "config": {"updates": 1000, "games": 40, "parallel": 40,
    "generation-games": 200, "zero-updates": 200, "seed": 9001, "seconds": 300}
}
```

Fresh receipts use mode `fresh`, update 0 and `initial_weights_sha256` over the
complete runtime file (regular, <=256 MiB). No concurrent input writer is permitted.
The source inventory hash is not a Git revision. A fresh image cannot impersonate
the frozen B40 binary; annealed has no provenance migration operation. Scope mismatch
must fail, never be bypassed by editing metadata or switching to initial weights.
Accepted U428 paths/hashes and future resume requirements are in the local
`artifacts/deslop-20260922/RESUME.md`; its preserved frozen runtime is a separate lineage.

## Guard contract

[runtime_guard.py](../docker/runtime_guard.py) accepts no arbitrary shell payload.
It holds nonblocking exclusive flocks, in order, on existing repository `heavy.lock`
and `artifacts/temp/map2-learning-20260911/credit-095-002-resume-001-journal/heavy.lock`.
Descriptors stay open/inherited across invocations; missing/contended locks refuse
admission. Read-only Linux bind mounts support flock; copied locks do not coordinate.

Before CUDA initialization, verify private cgroup v2: memory.max 12884901888,
swap.max 0, pids.max 1024, unlimited cpu.max and initially zero resource events.
Admission requires host MemAvailable >=24 GiB and selected GPU free VRAM >=4 GiB;
operating floors are 16 GiB RAM/4 GiB VRAM, ceilings CPU 90 C/GPU 85 C. Sample no
faster than ~1 Hz and recheck limits/events each time. Missing CPU temperature is
reported unavailable; missing GPU/malformed telemetry or changed events fails closed.
Validate that read-only proc/hwmon mounts really expose host values. UUID GPU queries
have a 0.75-second wait and 4096-byte output limit. Each invocation has <=300 seconds,
including eight seconds reserved for telemetry/shutdown; TERM/KILL targets only the
owned session. Docker stop grace is five seconds; kernel-stalled I/O is not bounded
by a userspace watchdog. No automatic retry, SIGSTOP or checkpoint-triggered kill.

Each child gets 16 MiB and a unique `logs/payload-NNNN.log`, never overwritten:
at most 1000 files / 15.625 GiB for target <=1000. Before every child require
100 GiB + remaining updates ×16 MiB user-available log-filesystem space. This is
headroom, not a reservation. Wrapper output is capped at 20 MiB; native Docker logs
retain 20 MiB ×1. Cap, short-write, deadline, resource or native failure stops the
controller. A failed-name collision requires review, a new receipt and new private
logs directory—not deletion/rotation of evidence or blind restart.

## Native invocation and future commands

Native invocation-limit/stop-resume contracts passed the 2026-09-22 Rust suite.
`--invocation-updates 1` adds one completed update after restore, clamps to total
target, stays outside canonical scope, and forces checkpoint/runtime/metrics
finalization before return. Completed native resumes do no work; the guard refuses
initial admission at target. Its v13 scalar progress check requires exactly before+1;
native tensor/Adam/RNG validation remains authoritative. Guard supports annealed
large-batch even games 28..40, not standard v12 or unknown formats. Its fixed payload is:

```text
train-annealed --updates 1000 --games 40 --parallel 40 --generation-games 200
  --zero-updates 200 --opponent teacher --epochs 4 --minibatch 2048 --seed 9001
  --device cuda --device-ordinal 0 --invocation-updates 1 --checkpoint-seconds 300
  --checkpoint-directory /run-data/checkpoint --metrics-directory /run-data/metrics --resume
```

Daemon-independent configuration checks (examples contain no live data):

```sh
docker compose --env-file .env.example config --quiet
docker compose --env-file .env.example --profile training config --quiet
```

Only after explicit qualification, build from the repository root with parent context
containing sibling bota. [Dockerfile](../Dockerfile) pins toolchain/CUDA images and
[its allowlist](../Dockerfile.dockerignore) excludes artifacts, targets, secrets and
Git history. BuildKit itself needs verified exclusion, memory/PID/swap/log/deadline
bounds and worker cleanup; the runtime guard does **not** contain build steps.

```sh
docker build --platform linux/amd64 --file Dockerfile \
  --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:?Set selected GPU capability}" \
  --tag "${DRYSUA_IMAGE:?Set approved image tag}" ..
docker compose --env-file docker/local.env up -d --pull never --no-build metrics prometheus grafana
docker compose --env-file docker/local.env --profile training up -d --pull never --no-build training
```

Export build arguments explicitly; `docker build` does not load Compose env files.
Do not use `compose build`: x-source-build is documentation, not a service build.
Qualify matching CUDA 13.3 runtime/PTX driver support and ELF closure, GCC/G++15 with
NVCC_CCBIN=/usr/bin/g++-15, resolved package versions and actual image/source/binary
hashes. Image pins do not prove signatures, reproducible APT resolution or runtime ABI.
Offline guard fixtures use `python3 -B -m unittest discover -s docker -p 'test_*.py'`;
container deadline/cleanup, mount/UID/healthcheck behavior, strict one-update resume
and copied monitoring history still need Docker qualification before any cutover.
