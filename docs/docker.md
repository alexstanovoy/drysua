# Docker deployment source — not an executed migration

## Status and non-interference boundary (2026-09-21)

The root `compose.yml` describes **training + stable native metrics exporter +
Prometheus + Grafana**. It does not replace, adopt, pause, signal, or stop a running
host process. The existing B40 campaign and monitoring remain the active deployment.
No current metrics path, port, secret, checkpoint, TSDB or Grafana database was changed.

**Blocked:** Docker CLI 29.8 / Compose 5.5.1 are available, but daemon access is
denied at `/var/run/docker.sock`; passwordless administrative access is unavailable.
This work does not change socket permissions, groups, daemon configuration or host
services, and does not use privilege escalation. No images were pulled/built, no
containers started, and no tests run while training is active. A working authorized
Docker context is required, not a command that bypasses these restrictions.

This is a replacement deployment backend, not a technique for moving a live CUDA
process into a container. A future training migration must start **from a verified
copy at an accepted native checkpoint boundary**. It must not run a second learner
while the existing controller owns the campaign. Existing service retirement and
release of ports require a separate approved handoff; they are not authorized here.
No host service-manager commands or legacy artifact runner imports are needed by
the new Docker runtime.

## Services and ownership contract

| Service | Start selection | Network / listener | Memory / PIDs | Restart |
| --- | --- | --- | --- | --- |
| `training` | Explicit `training` profile only | `none` | hard 12 GiB / 1024 | **no** |
| `metrics` | Default | host / `127.0.0.1:9464` | 256 MiB / 32 | **no** |
| `prometheus` | Default | host / `127.0.0.1:9090` | 512 MiB / 128 | **no** |
| `grafana` | Default | host / `127.0.0.1:3000` | 512 MiB / 128 | **no** |

All run non-root (UID/GID 1000 by default, configurable), read-only root filesystems,
all capabilities dropped, no-new-privileges, no extra swap, bounded tmpfs and Docker
logs. There is no Docker socket, privileged container, host PID namespace, CPU quota,
affinity or global thread limit. Training has an init process to reap descendants,
and requests one explicit NVIDIA UUID with `compute,utility`. The CUDA-visible
ordinal is zero, matching the recorded B40 scope. The exporter uses the same image
and device to resolve its CUDA-linked ELF and collect GPU telemetry; `metrics-serve`
does not start a learner.

Host networking is needed for the existing loopback-only metrics contract. It must
be the **local Linux host's** network namespace. Remote daemons, Docker Desktop and
rootless network-namespace translation have not been qualified. No `ports:` mapping
is used. Localhost-only services are still accessible to other local users; this is
not an authenticated external metrics endpoint. Do not widen listeners.

Root Compose **extends** `monitoring/compose.yml` for the two monitoring images,
command lines, query/retention limits, plugin restrictions, provisioning and data
copy checks. The monitoring owner retains `monitoring/**` and `docs/monitoring.md`.
Root overrides restart to `no`, stop grace to 5 seconds and adds healthchecks; it
does not fork dashboards, rules or datasource config. Integration variables are:

- `PROMETHEUS_DATA_DIRECTORY`, `GRAFANA_DATA_DIRECTORY`: verified private copies,
  never a second writer into live databases. The shared configuration refuses empty
  history/database directories rather than silently losing the existing dashboard.
- `GRAFANA_ADMIN_PASSWORD_FILE`, `GRAFANA_SECRET_KEY_FILE`: both original secrets
  or preserved copies; bind-mounted as files only into Grafana. Preserve the database,
  password **and encryption key** together. Do not print them or replace them.
- `MONITORING_UID/GID` and `DRYSUA_UID/GID`: match actual host ownership. No root
  init/chown service is supplied. File-backed Compose secrets retain host permissions;
  a YAML secret `uid` is not a substitute for readable host files.
- `PROMETHEUS_PORT`, `GRAFANA_PORT`: default original ports. Alternative monitoring
  UI ports require the monitoring owner's config contract; the metrics endpoint is
  fixed at 9464 and cannot coexist with the current listener.
- `monitoring/prometheus/prometheus.yml` must scrape `127.0.0.1:9464`; Grafana's
  provisioned datasource uses `DRYSUA_PROMETHEUS_URL` from the extended service.

Healthchecks are observational, not restart controllers. Exporter HTTP 200 proves
liveness, **not** that persisted training state is healthy; inspect native
`metrics_available` / `metrics_state_healthy` gauges too. Availability of the pinned
Prometheus/Grafana `wget` executables and health routes still needs image-runtime
verification. No `depends_on: service_healthy` blocks collection of unavailable-state
metrics or hides the original service startup error.

## Source packaging, ABI and image pins

Build context is the parent containing **both** `drysua/` and sibling `bota/`:
Dockerfile location `drysua/Dockerfile`, context `..` from this repository.
`Dockerfile.dockerignore` starts with `**` and admits only Rust source/manifests,
lockfiles, the Dockerfile and its two runtime/packaging Python files. Artifacts,
targets, Git history, secrets, arbitrary parent-directory contents and training data
are not transferred. Do not use a different Dockerfile without an equivalent
Dockerfile-specific allowlist. No model is baked into the image.

Pins are linux/amd64 child manifests, except the monitoring owner's multi-platform
indexes. Registry index/manifest/config JSON hashes and publication metadata were
checked without downloading image layers:

| Purpose / exact tag | Pinned SHA256 | Published / pushed |
| --- | --- | --- |
| `rust:1.98.0-slim-bookworm` toolchain donor | `af0579d28b9a7ec5251aaafcb0c0a23dcde5c97065112aae0cc3abeda42d5394` | 2026-08-25; Rust release 2026-08-20 |
| `nvidia/cuda:13.3.1-devel-ubuntu26.04` | `0ee41c7eac41d2579b268be60db1012ad23b0d4f3222b76566128fe28881a8f8` | 2026-07-28; CUDA release 2026-06-29 |
| `nvidia/cuda:13.3.1-runtime-ubuntu26.04` | `b9321b748007329ae6a63261eb041612d18b802e23a485717cfa3584d640dd57` | 2026-07-28 |
| `prom/prometheus:v3.14.0` | `5ce7540c3c00ef4ab0c9d2c995c6a5b9c421f44b4a115d97a2c7af3b1c21cbb0` | release 2026-08-18 |
| `grafana/grafana:13.2.2` | `ac461fb352abc50da10a51c7d02462e9c05488f11f53f14b3ad79a8145f638a0` | 2026-09-15, security-patch exception |

Sources: Docker Hub Registry V2 and tag metadata for these exact tags;
[Rust distribution manifest](https://static.rust-lang.org/dist/channel-rust-1.98.0.toml),
[NVIDIA release metadata](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json),
[CUDA compiler compatibility](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-installation-guide-linux/index.html).
These are verified metadata, **not** publisher-signature or startup/ABI tests.

The Rust donor contributes the actual toolchain, not rustup shims that might fetch
components. Build uses Rust 1.98.0, `builtin,cuda`, the locked Cargo dependency graph,
`gcc-15` / `g++-15`, and **`NVCC_CCBIN=/usr/bin/g++-15` inside the build payload**.
`CUDA_COMPUTE_CAP` must be supplied for the explicitly selected GPU; do not discover
a random device during build. Build and runtime use the same Ubuntu 26.04 baseline.

Read-only ELF inspection of the existing `artifacts/runtimes-current/drysua-training`
found GLIBC_2.39 and direct dependencies on `libcuda.so.1`, `libcurand.so.10`,
`libcublas.so.13`, `libcudart.so.13`, libstdc++, libgcc and libc. An old Ubuntu image
or CUDA 12 image is not an adequate frozen-runtime assumption. This is not proof
of the complete dependency closure of every frozen binary. Ubuntu 26.04's published
baseline is glibc 2.43; a fresh build can acquire newer symbol requirements. Keep
the matching CUDA runtime, not driver stubs. The host driver/NVIDIA toolkit must
support the actual CUDA 13.3 PTX; the generic CUDA 13 minor-compatibility floor
alone is not proof of PTX JIT support.

**Package reproducibility note:** APT installs `g++-15`, Python and runtime C++
libraries from the trusted distribution repositories configured in the pinned base.
Record the resolved package versions in the approved build window; image pins alone
do not imply bit-reproducible package installs. A maintained dated snapshot is an
optional reproducibility improvement, not an additional deployment permission gate.
Do not pin old vulnerable packages for reproducibility; include applicable security
updates. The installed toolchain and runtime ELF closure still need qualification.
No frozen-only runtime image is supplied with an unproven ABI.

`source_identity.py` records a bounded, sorted path/mode/content inventory of the
allowlisted payload. Its canonical JSON hash becomes `source-sha256-...` in the
native provenance fields; simulator identity has its own inventory hash. This is
**not a Git revision**, clean or dirty. `/opt/drysua/` includes `sources.json`, the
source hash and `provenance.json` with the actual resulting binary SHA256. Package
resolution can change that binary hash even for identical source. Preserve the
final immutable image ID/digest, compiler/package versions and build arguments
alongside these records.

Historical B40 freezer records, read as provenance rather than recomputed here:

- binary: `8dfccb68a60dad4ecd0e7d86f35db519500986dea2a125903a55c914686201f6`;
- source inventory: `f2a76573616808c4038bdb6548021a167905c886a3936bc117a085e2da988b03`;
- source archive: `2eeb5ec667bfd84c15d929cc0eea31a70ae9247893dd93bec884189879e40a38`.

Those values must never be substituted for the hashes of this new source image.
Keep the original frozen archive/inventory/diff and boundary evidence read-only
under the selected provenance directory.

## Native invocation contract — implemented in source, qualification pending

The core owner has implemented `--invocation-updates` as
`Option<NonZeroU64>` in the Rust CLI/config, with a relative invocation boundary and
five tests in `src/tests/annealed_invocation.rs`. The source integration is present;
those tests have **not been run**, and no rebuilt binary or Docker runtime has been
qualified. No Rust files are changed by this deployment work. The guard's command
now matches the source CLI, not an assumed flag in the existing frozen binary:

```text
train-annealed --updates 1000 --games 40 --parallel 40
  --generation-games 200 --zero-updates 200 --opponent teacher
  --epochs 4 --minibatch 2048 --seed 9001 --device cuda --device-ordinal 0
  --invocation-updates 1 --checkpoint-seconds 300
  --checkpoint-directory /run-data/checkpoint --metrics-directory /run-data/metrics
  --resume
```

`--invocation-updates 1` means **one additional completed update after restoration**:

1. The limit is excluded from canonical run scope. The full N1000 annealing target,
   generation schedule, explicit parallel count, PPO settings and RNG are unchanged.
2. Before successful return, the invocation boundary forces native checkpoint, runtime export and
   metrics finalization, even if the 300-second cadence is not due. Do not collect
   the next update. A log record alone is not a durable boundary.
3. CLI/config bounds reject zero and out-of-range/overflowing limits. The relative
   boundary clamps to the total target; an already-complete native resume does no
   further work. This guard refuses initial admission at/above target.
4. The five native fixtures cover CLI/library bounds, unchanged scope/schedule,
   fresh forced checkpoint/export and resumed model/Adam/RNG equivalence, target
   clamping and the completed-run no-op. These are written checks, not passed tests.

The guard supplies the implemented flag and still refuses an older unsupported
binary rather than falling back. Build and qualify the current source first.
After a successful child exit it checks bounded `checkpoint.meta` v13 progress
for exactly `before + 1`; this scalar check is **not** a second tensor validator.
The native loader retains full tensor SHA, model, Adam, RNG and snapshot checks.
The small guard supports the annealed large-batch schema only (even games 28–40),
not standard-profile checkpoint v12. Unknown schemas fail closed.

**Strict old-B40 resume is separately blocked:** the fresh image has a different
truthful source identity. Annealed resume currently has no explicit provenance
migration operation. A scope failure must remain a failure. Do not spoof an old
revision, relabel checkpoint scope, edit metadata or switch to `--initial-weights`.
Adding the invocation flag does not itself solve this provenance boundary. Until
an explicit audited native migration exists, this image supports only a genuinely
new lineage or strict resumes of its own source-compatible lineage; the old B40
campaign remains on its existing runtime.

## Runtime guard and data layout

`docker/runtime_guard.py` is a stdlib-only, single-purpose supervisor, not a host
service framework. No arbitrary payload command or shell environment is accepted.
The training profile is intentionally absent from a default `up`.

It takes nonblocking exclusive `flock` on the **existing inodes**, in order:

1. repository `heavy.lock`;
2. `artifacts/temp/map2-learning-20260911/credit-095-002-resume-001-journal/heavy.lock`.

Read-only bind mounts still support Linux exclusive flock. Both descriptors are
retained for the campaign and inherited by each child, and only closed, never
explicitly unlocked. Missing files or lock contention refuse admission, without
touching the owner. Do not replace these locks with named volumes or copied files.
`GLOBAL_HEAVY_LOCK` may point to the same real shared lock in a relocated workspace,
not an unrelated new lock. Docker CLI cannot pass host flock descriptors through
the daemon; acquiring inside the container avoids that false protection.

Before any learner/CUDA initialization, the guard requires its **actual** private
cgroup-v2 namespace with `memory.max=12884901888`, `memory.swap.max=0`,
`pids.max=1024`, and unlimited `cpu.max`. Admission needs host MemAvailable ≥24 GiB
and selected GPU free VRAM ≥4 GiB. Operating limits are MemAvailable ≥16 GiB,
free VRAM ≥4 GiB, CPU ≤90°C and GPU ≤85°C. Missing CPU temperature is explicitly
recorded as unavailable; malformed telemetry, missing GPU telemetry or any changed
memory/PID event counter refuse/terminate **only the owned container payload**.
Initial event counters must be zero. Cgroup limits are rechecked on each sample.

Sampling is approximately 1 Hz, never faster, and continues across invocation
boundaries. Only `/proc/meminfo` and one **resolved** CPU hwmon directory are bound
read-only for the guard. Set `CPU_HWMON_DIRECTORY` to the real `k10temp`/`coretemp`
directory, not a broken relocated `/sys/class/hwmon` symlink. The exporter gets
host `/proc/stat` and `/proc/meminfo` at the exact native collector paths. No full
host `/proc` or host root mount is needed. Verify these actually expose the host
values on the authorized daemon; a path name alone is not proof.

GPU queries specify the UUID, have a 0.75-second wait and 4096-byte file limit.
Each heavy child has at most a 300-second budget including an eight-second reserve
for telemetry/shutdown. Owned-session TERM then KILL cleanup fits within the Docker
5-second stop grace under normal scheduling. Kernel hangs/uninterruptible I/O are
not made impossible by a userspace watchdog; validate daemon/PID-namespace cleanup
in an isolated qualification window. No SIGSTOP, checkpoint-triggered termination,
checkpoint archiving or automatic retry is used. The lightweight container can
last through 1000 successful one-update invocations, holding the global locks.

Payload output has a **16 MiB cap per bounded native invocation**, with a reset
byte counter and a unique exclusive-created `logs/payload-NNNN.log` for its target
update: `payload-0001.log` for U0→U1, through `payload-1000.log` for U999→U1000.
The fixed namespace and target ≤1000 allow at most **1000 files / 15.625 GiB** per
approved logs directory. Normal ~20 KB/update episode audit logs therefore do not
consume a campaign-wide 16 MiB limit. Existing files, including failed attempts,
are never overwritten or rotated away; a duplicate name refuses before spawning.

Before **each** invocation, user-available space on the logs filesystem must be at
least **100 GiB + (target − completed updates) × 16 MiB**: 115.625 GiB at U0/N1000.
This admission floor budgets all remaining logs and leaves checkpoint/data headroom;
it is not a filesystem reservation or a bound on other writers or checkpoint size.
Cap/short-write/deadline/resource/native failures exit nonzero with no next invocation.
Wrapper output retains its **20 MiB total per controller lifetime** bound and Docker
`json-file` retention of 20 MiB × 1 for native services. No automatic archival cache
is created. Native checkpoint tensor/manifest publication remains unchanged.

Prepare **new** owned private paths only, outside active campaign state:

```text
TRAINING_DIRECTORY/                 0700, DRYSUA_UID owner
  checkpoint/                       0700, native checkpoint copy or empty fresh directory
  metrics/                          0700, matching native metrics copy
    .metrics.writer.lock            existing regular 0600 file; same file for exporter
  logs/                             0700, empty for fresh run; keep prior logs on resume
    payload-0001.log ...             exclusive per-update evidence, at most 1000 files
INITIAL_WEIGHTS_DIRECTORY/           immutable approved weights; mounted read-only
PROVENANCE_DIRECTORY/                frozen evidence; mounted read-only
  acceptance.json                   explicit reviewed admission receipt
```

For a new lineage, the metrics directory may contain only the precreated empty
`.metrics.writer.lock`; the native writer creates state. The exporter sees the whole
metrics directory read-only and only that one lock file read/write for its liveness
probe. Checkpoint and metrics are siblings, never nested. All four bind sources are
required; none is auto-created by Compose. No automatic permission repair occurs.

Example **receipt shape**, not accepted values (do not use placeholder hashes):

```json
{
  "source_sha256": "ACTUAL_IMAGE_SOURCE_SHA256",
  "binary_sha256": "ACTUAL_IMAGE_BINARY_SHA256",
  "gpu_uuid": "ACTUAL_GPU_UUID",
  "mode": "resume",
  "starting_update": 180,
  "checkpoint_meta_sha256": "SHA256_OF_ACCEPTED_COPY_CHECKPOINT_META",
  "config": {
    "updates": 1000, "games": 40, "parallel": 40,
    "generation-games": 200, "zero-updates": 200,
    "seed": 9001, "seconds": 300
  }
}
```

U180 above illustrates the field, not authorization to resume a particular B40
checkpoint. A fresh receipt uses `mode: fresh`, update 0 and
`initial_weights_sha256` for the complete `drysua.weights.safetensors` bytes.
The guard verifies the selected weights (regular file, bounded to 256 MiB), so a
different compatible model cannot silently satisfy the same receipt. Frozen inputs
must not have concurrent host writers. Resume receipts cannot bypass native scope
checks; preserve matching metrics and all randomization generations.

Manual restarts require review and a **new** receipt for the inspected committed
boundary. Keep prior per-update logs; a resumed next update uses its own unused name.
If a failed update's name already exists, it cannot be retried in that logs directory.
Preserve the entire failed attempt; a separately approved retry needs a new private
logs directory and its own disk budget, not deletion/rotation of evidence. No automatic
retry directory is created. Do not blindly repeat `up`/`restart` after a resource
failure. No service has implicit startup after a daemon reboot.

## Safe validation now; deferred qualification and handoff

The following is daemon-independent and does not read secret contents or create
containers. `.env.example` contains deliberately non-deployable path/UUID examples:

```sh
docker compose --env-file .env.example config --quiet
docker compose --env-file .env.example --profile training config --quiet
```

An actual config belongs in ignored `docker/local.env`. Do not run `up` with the
example, and do not dump expanded real configuration into public diagnostics.
`x-source-build` documents the source build inputs but is **not** a service build
stanza: default `up` cannot unexpectedly compile/pull if the local image is absent.

The source image needs an **explicit manual build**, not `docker compose build`.
Only in the approved exclusive build window, after builder containment below is
established, the build command from the `drysua/` repository root is:

```sh
docker build --platform linux/amd64 --file Dockerfile \
  --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:?Set the selected GPU compute capability}" \
  --tag "${DRYSUA_IMAGE:?Set the approved local source-image tag}" ..
```

This command is documentation, **not executed or authorized during live training**.
Export those two approved shell values explicitly; `docker build` does not load
Compose's env file. Use the same local tag in `docker/local.env`, and record the
resulting immutable image ID/digest and provenance. Both native services use it.

Completed lightweight checks: Compose parsing with all four services; default
service selection contains only `metrics`, `prometheus`, `grafana`; normalized
resource/security/restart settings and extended monitoring mount paths; AST syntax
for the four Python files without importing them; function-length bounds; ignored
local environment path; whitespace checks. Compose 5 serializes memory byte values
as JSON strings, so normalized checks convert them to integers. None of these
checks establishes runtime safety or executes a unit test.

Deferred test/qualification plan — **not executed during live B40**:

1. Offline focused guard fixtures (written before runtime logic):
   `python3 -B -m unittest discover -s docker -p 'test_*.py'`.
   They cover cap boundaries, event deltas, lock contention, bounded checkpoint
   scalar parsing, command/scope preservation, input/config refusal, source identity,
   shutdown admission, per-invocation log-budget resets, disk-floor boundaries,
   duplicate/failure-evidence preservation and log overflow/short writes. No GPU or
   host service needed.
2. Core owner qualifies the implemented native invocation contract and its five
   unexecuted tests separately. No Cargo checks,
   builds or tests are added to the active session by this deployment work.
3. Arrange authorized daemon access without socket chmod/group bypasses. Establish
   local host networking, private cgroup v2, NVIDIA toolkit, exact UUID/compute
   capability and driver compatibility. Preserve currently active services.
4. Qualify the source-image build in an exclusive heavy-work
   window. **The runtime guard does not guard BuildKit RUN steps.** A bare Docker
   build is not an approved heavy launcher: Docker client timeout or a host flock
   alone does not contain daemon-side build processes. The builder needs verified
   memory/PID/swap limits, host telemetry, shared exclusion, bounded logs/deadline
   and cleanup of only its own build workers before any large compile/download.
   This builder admission remains unresolved; no unsafe host launcher is supplied.
5. Inspect source/binary/image identities and runtime `ldd`/ELF dependencies. Check
   read-only mounts and UID permissions, actual cgroup limits, host RAM/CPU telemetry,
   bounded UUID query, and healthcheck binaries. An isolated owned-container fixture
   must verify deadline, descendant cleanup, resource failure and **no retries**.
6. Use a new non-production lineage/copy to demonstrate graceful one-update saves,
   strict resume through the next update (model/Adam/RNG preserved), durable metrics,
   no lost progress, and scope-mismatch refusal. Resolve native provenance migration
   explicitly before even considering an old B40 copy.
7. Monitoring owner verifies the TSDB and SQLite/data copies, secret preservation,
   Prometheus config/rules and Grafana datasource/provisioning with the pinned images.
   Never copy a live SQLite/WAL/TSDB directory arbitrarily and call it consistent.
8. Only at a separately accepted handoff, after the old owner naturally releases
   the campaign and the original ports are made available by an authorized owner,
   launch the Docker services from verified copies. Explicit future commands are:

   ```sh
   docker compose --env-file docker/local.env up -d --pull never --no-build metrics prometheus grafana
   docker compose --env-file docker/local.env --profile training up -d --pull never --no-build training
   ```

   These are **not instructions to run now**. Starting the first command today
   would conflict with live listeners. Starting the second cannot complete the
   pending native/runtime/provenance/access qualification on its own.
9. Verify localhost listeners, effective cgroups/events, endpoint state, Prometheus
   target and real historical samples, authenticated dashboard/Explore/save behavior,
   and copy persistence. Keep bounded evidence; do not claim browser rendering from
   HTTP checks alone. Do not remove volumes/data or stop training to test monitoring.

The final deployment decision remains blocked by daemon access, builder containment,
native/runtime qualification, strict provenance compatibility and the agreed handoff.
Prepared source/configuration and successful Compose parsing are not a migration.
