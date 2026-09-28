# Docker deployment: rootless probes passed, production stack pending

Rootless Docker access is now available (2026-09-27); the earlier socket-access
blocker is resolved for the `rootless` context. Contained toolchain, test and actual
CUDA kernel/learning-diagnostic workloads passed as described below. This does not qualify the production Compose
stack or establish a training migration. Existing monitoring stays in place; no
full training is authorized. Do not escalate privileges or control host services.
Production deployment still requires runtime qualification and a data/port handoff.

## Current rootless experiment qualification

Ignored runner/evidence: `artifacts/deslop-20260922/learning-experiments/` (see
`QUALIFICATION.md`). `docker_run.py` explicitly selects context `rootless`, socket
`unix:///run/user/1000/docker.sock`, Docker 29.8/cgroup v2. The daemon's `systemd`
cgroup driver does not require this runner to call a service manager or D-Bus.
Verified container UID 0 maps to unprivileged host UID 1000, not host root.

Official Ubuntu 26.04 pin/local image ID:
`sha256:61ebaa5cc23ca45450db85eac015435199ec569e28ec222ea13f2aed2110b8a6`.
Host toolchain/loader/libraries are read-only mounts; workspace and Cargo/target
caches retain their absolute paths and write access. Artifacts remain read-only
except the unique job output at `/out`. This is host-dependent experiment packaging,
not a reproducible production image. No image build was performed; Cargo compilation
and tests used this container with the mounted host toolchain.

The parent holds both original heavy locks plus a probe lock until owned-container
exit. Actual verified cgroup values: hard memory 12 GiB, swap 0, PIDs 1024, unlimited
CPU; zero resource events. Host telemetry monitoring, a 240-second watchdog,
235-second payload timeout plus five-second kill grace, 16-MiB logs, unique names,
no retries, network none, dropped caps and no-new-privileges apply. No socket mount.
Only one heavy job runs at a time, after source editing is coordinated. Test temporary
files use the job's `/out` directory rather than raising the 512-MiB `/tmp` limit.

`cpu-toolchain-02` passed glibc 2.43, Rust/Cargo 1.98, g++ 15.3, NVCC 13.3 and
offline Cargo metadata probes. `gpu-query-02` passed GPU0/UUID query: RTX 5090,
driver 610.57.04, capability 12.0. No NVIDIA runtime/CDI specs were installed;
explicit device-node passthrough plus host libraries worked. Subsequent integration
passed bota workspace debug/release tests, drysua all-feature/no-default checks and
real Candle CUDA backward/readback. Frozen U428 40-game collection plus isolated
CPU readout fits also executed successfully; U428 parameters stayed bit-identical.
The unsupported head prototypes were then archived and removed from compilation;
the retained diagnostic is a census with no head fitting or new policy architecture.
This qualifies the adapter for the measured workloads, not learning improvement or
production resume. See `artifacts/deslop-20260922/learning-experiments/REPORT.md`.
Resource-failure/timeout cleanup paths were not deliberately fault-injected. No
production training, native fallback, Compose migration or service restart occurred.

Run only a separately approved short workload, with a new label each time:

```sh
python3 -B artifacts/deslop-20260922/learning-experiments/docker_run.py cpu metadata-NEXT cargo metadata --offline --no-deps --format-version 1
python3 -B artifacts/deslop-20260922/learning-experiments/docker_run.py gpu gpu-query-NEXT nvidia-smi --id=GPU-1cd1082a-71fa-610e-1d1a-86a3328c6b6e --query-gpu=uuid,memory.free,temperature.gpu --format=csv,noheader,nounits
```

Replace `NEXT` with a unique lowercase/digit label. The runner accepts direct command
arguments after the label and preserves per-job commands, inspect, logs and telemetry.

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
| `RUST_TARGET_CPU` | Docker build argument, default `native`; use `x86-64` for baseline CPU ISA |
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

## Rust CPU code generation

Both `drysua/.cargo/config.toml` and `bota/.cargo/config.toml` set
`build.rustflags = ["-C", "target-cpu=native"]`. Cargo invoked from either project
root or its descendants inherits that project's config. Invocation from their
parent with `--manifest-path` does **not** discover the child config; pass
`RUSTFLAGS='-C target-cpu=native'` explicitly there. Without `--target`, these flags
apply to all Rust crates Cargo compiles, including bota path dependencies, build
scripts and proc macros; the toolchain's prebuilt standard library is not rebuilt.

Docker keeps `WORKDIR /src` and explicitly sets `RUSTFLAGS` from build argument
`RUST_TARGET_CPU` (default `native`), rather than relying on child config discovery.
For baseline-ISA builds, override locally with `RUSTFLAGS='-C target-cpu=x86-64'`
on the Cargo command, or pass Docker `--build-arg RUST_TARGET_CPU=x86-64`.
`RUSTFLAGS` replaces config rustflags; unset `CARGO_ENCODED_RUSTFLAGS` if present
because it takes precedence. Preserve any other required flags in the override.

`native` means the **builder's visible CPU**, including inside Docker, not the GPU
or the eventual runtime host. The intended host is AMD Ryzen 9 9950X3D (Zen 5),
8 online cores with SMT off, Rust 1.98; this config changes no CPU/thread limits.
CUDA 13.3 / `CUDA_COMPUTE_CAP=120` is separate; no CFLAGS, NVCC or fast-math changes.
The target triple stays `x86_64-unknown-linux-gnu`, never `native`. Run the resulting
binary only on compatible CPU ISA/OS support; baseline ISA does not remove ELF,
system-library or CUDA ABI requirements. Native kernels may change timings and
floating-point bits even without fast-math; requalify rather than assume identical
results or performance. No native rebuild or profiling is established by this edit.

The Docker allowlist admits both configs, so `sources.json` and `source.sha256`
include them. `provenance.json` also records `rust_target_cpu` and exact `rustflags`;
profiling receipts must retain effective flags and builder CPU/ISA. These describe
a new source build, not a replacement for frozen binary/checkpoint hashes or scope.

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
  --build-arg RUST_TARGET_CPU="${RUST_TARGET_CPU:-native}" \
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
