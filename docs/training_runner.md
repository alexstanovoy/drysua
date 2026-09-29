# Rootless single-invocation runner contract

Entrypoint: `python3 -B scripts/train_runner.py run --spec /absolute/JOB.json`.
This is the backend for the separately owned `scripts/train.py` controller, not a
campaign planner or checkpoint parser. It imports no artifact runners. Existing
running campaigns/frozen helpers are not migrated or modified.

## Version 1 input

The JSON object contains exactly these required fields:

```json
{
  "version": 1,
  "workspace_root": "/absolute/bots/drysua",
  "campaign_directory": "/absolute/bots/drysua/temp/new-campaign",
  "job_directory": "/absolute/bots/drysua/temp/new-campaign/invocations/000001",
  "command": ["/absolute/bots/drysua/temp/new-campaign/bin/drysua", "train-annealed", "--invocation-updates", "1", "--device", "cuda"],
  "mode": "gpu",
  "docker_context": "rootless",
  "image": "ubuntu:26.04@sha256:61ebaa5cc23ca45450db85eac015435199ec569e28ec222ea13f2aed2110b8a6",
  "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
  "lock_paths": ["/absolute/existing/workspace.lock", "/absolute/existing/global.lock"],
  "campaign_id": "new-campaign",
  "invocation_id": "000001"
}
```

The command above is a schema illustration, not a complete native training argv.
The controller supplies all native settings and checkpoint/input paths. No shell,
arbitrary environment, automatic retry, model metadata offsets or provenance relabel.
`workspace_root` is the repository containing `scripts/train_runner.py`; its parent
is mounted read-only at the same absolute path. Campaign `bin/` and `inputs/` must
exist and are mounted read-only over the writable campaign. Jobs must be under
`campaign_directory/invocations/`. Paths must be absolute, canonical, non-symlink;
campaign/job are private and owned by the invoking host user. The binary must be an
owned executable regular ELF under `bin/`, not group/world-writable. IDs use at most
48 lowercase letters, digits or hyphens. Specs are bounded to 64 KiB, 128 argv entries,
4096 bytes per argument, 32 KiB total argv; native operation is `train-annealed` with
one `--invocation-updates 1`. CPU mode uses `--device cpu` and may set `gpu_uuid: null`.
GPU mode uses `--device cuda`, an explicit physical GPU0 UUID and ordinal zero.

Optional `cuda_directory` defaults to `/usr/local/cuda-13.3`; when provided it must
be an existing canonical directory under `/usr/local/`. No Cargo/toolchain caches
are mounted. The runtime is host-dependent, not a standalone production image.

Optional `invocation_seconds` is the **payload** deadline: default **235**, exact
integer **1..275** (booleans, fractional/nonfinite values and strings are rejected).
The one shared `resolve_timeouts` helper returns an immutable `InvocationTimeouts`:

| Budget | Formula | Default | Maximum |
| --- | --- | --- | --- |
| Native payload | P | 235 s | 275 s |
| Host capture, including child shutdown reserve | P + 5 | 240 s | 280 s |
| Entire runner/control cleanup | P + 25 | 260 s | **300 s** |
| Controller admission/cleanup/acceptance reserve | P + 25 + 30 | 290 s | 330 s |

The final row reserves lightweight controller time; it does not extend native or
runner execution. Values above 275 require an explicitly approved change to the
300-second safety envelope, not a hidden larger host timer. The default has **not**
been raised. Timeout is failure requiring operator review: never reduce games,
parallel worlds, minibatches or retry with a larger timeout automatically. This
option does not change the native `--updates` target or annealing/model scope.

New controller manifests normalize the default and propagate it to each frozen
job spec. Legacy version-1 manifests/specs without the field mean 235 without
inserting it into their stored objects, labels or hashes. Read-only status remains
compatible; old frozen controller/runner/config files are never edited. New receipts
and results include the resolved `timeouts` object, checked against the job when
present. Native-runtime qualification of the new deadlines remains pending.

## Ownership and outputs

Only existing regular shared flock files are accepted; acquire them in supplied
order, nonblocking, without creating/replacing/unlinking them. The controller must
not hold these same flocks while waiting for the runner. One job per process. A
contended lock or existing job intent refuses without writing into that job, so a
duplicate launcher cannot overwrite or preempt the active owner's result.

The host retains the locks throughout the invocation and cleanup. Before starting
native code, a private `runner-locks.sock` transfers **the same open-file descriptions**
via `SCM_RIGHTS` to the inside supervisor; the native child inherits them too. No
second flock acquisition or explicit unlock occurs. Thus a host crash/unconfirmed
Docker cleanup cannot release exclusion while an admitted native child still runs.
Socket peer UID, descriptor count and file identities are checked; a missing handoff
prevents native startup. A directory-FD socket path avoids Unix pathname-length limits.

Before Docker create, the runner writes exclusive immutable intent/spec/runner
snapshots. `receipt.json` records the 64-hex container ID before start. Container
labels bind campaign, invocation and canonical spec SHA256. The created container's
full ID, labels, image, stopped state and declared memory/PID/CPU limits are verified
before attach/start; actual cgroups are then checked before native execution.
Recovery APIs are
`load_spec(path)`, `inspect_owned_container(spec, container_id)` and
`stop_owned_container(spec, container_id)`. Both ownership APIs validate full ID and
labels against the job's persisted intent; stop never uses a name or host PID.
Missing/mismatched ownership evidence refuses recovery rather than guessing.
Inspection returns `{Id, Name, Labels, ImageReference, State, HostConfig}`; stop returns
the confirmed stopped `State`. Both take the validated dictionary from `load_spec`.
Only the current creating process may clean up its known ID if receipt persistence
failed before start; imported recovery never bypasses a missing receipt.

`result.json` uses schema `drysua-training-runner/v1`, including `returncode`,
`container_id`, `container_state`, `cleanup_confirmed`, `verified_limits`, resource
summary, `lock_transfer_sent` when a socket was established, and `error`. Stdout is
one small JSON result, not mirrored payload output. Preflight failure is persisted
only after acquiring the locks and exclusively claiming this job with its intent.
Other evidence: `payload.log`, `resources.jsonl`, `cgroup-usage.jsonl`, `limits.json`,
`limits-final.json`, `intent.json`, `receipt.json`, `runner-spec.json`, `runner.py`.
Existing evidence is never overwritten; the spec and runner snapshot are additionally
mounted read-only through both campaign and `/out` paths. The controller must accept success only
when returncode is zero, cleanup is confirmed, actual limit evidence exists, the
container is stopped and not OOM-killed. It separately validates native checkpoint
progress and model/Adam/RNG compatibility via the native inspector.

## Safety and qualification

Rootless daemon security and UID0-to-host-UID mapping are verified, not assumed.
Limits: private cgroup v2, hard 12 GiB, swap zero, **fixed PIDs 1024**, no CPU quota/affinity,
network none, read-only root, dropped capabilities, no-new-privileges. No pull,
build, Docker socket mount, host service command or privilege fallback. Read-only
host libraries and explicit GPU0 device nodes support the existing qualified host
ABI. CPU mode does not query or expose GPU devices.

Admission requires host RAM >=24 GiB; operating floor 16 GiB; CPU <=90 C. GPU mode
also requires VRAM >=4 GiB and GPU <=85 C, matching the configured UUID. Missing CPU
temperature is recorded as unavailable, not zero. Host/GPU commands and telemetry
are bounded, sampling at most 1 Hz. Disk admission is 100 GiB plus this job's 16 MiB
payload budget; the controller reserves remaining campaign log budget separately.
Inside cgroup limits/events are checked before native launch and during execution.
Native/host/controller deadlines and finite loop bounds use the resolved budgets
above. Payload cap 16 MiB, control result/command outputs 64 KiB, each resource
history **512 KiB**. RAM/swap and temperature/admission thresholds are unchanged.
The approved PID ceiling is **1024**, checked from one source constant in Docker
creation, Docker metadata and actual `pids.max`; 10000 is rejected. The campaign's
10000-update bound is not a PID allowance. An unauthorized source-only increase to
10000 was reverted before runtime qualification; it was not applied to a running
campaign. No frozen runner's PID limit or timeout is rewritten. Timeout configuration
grants no permission to raise resource caps automatically.

## Utilization telemetry (not a performance guarantee)

`resources.jsonl` records monotonic time, elapsed capture time and phase; admission
telemetry is retained in `result.resources.admission`. CPU rows are extracted from
the beginning of `/proc/stat` without reading its unrelated potentially huge IRQ
array. Up to **256 discovered logical CPU IDs** and unsigned 64-bit counters are
validated; compact aggregate busy/total ticks and `cpu_count` are recorded, not an
unbounded per-core map on every row. Guest ticks are not double-counted. Aggregate
host utilization includes **other host processes**, not just the learner.

One UUID-checked `nvidia-smi` call per sample (<=1 Hz, 1-second timeout, 4096-byte
output cap) records free VRAM, temperature, GPU utilization, memory-controller
activity and optional power watts. GPU utilization describes the **whole device**;
memory utilization is activity, not VRAM occupancy. Optional `N/A` is `null`, never
zero; malformed/nonfinite values fail validation. Free VRAM/temperature remain
mandatory GPU safety measurements. CPU mode does not query NVIDIA.

`cgroup-usage.jsonl` additionally records bounded `cpu_stat` counters (`usage_usec`,
`user_usec`, `system_usec`), `memory_current`, **`memory_peak`**, and `pids_current`.
The CPU accounting baseline is taken **before** native spawn. Final usage/summary
is saved in `limits-final.json` after child shutdown and copied into
`result.resources.cgroup`; this counts the whole container, including its small
supervisor. Memory peak is the actual kernel cgroup peak, not the last RSS sample.

Summary `host_cpu_percent` / cgroup `cpu_percent` use **100% = one logical CPU**.
`host_cpu_machine_percent` divides by the discovered logical CPU count; cgroup
`cpu_machine_percent` uses that count captured at admission. Host CPU summaries
use first/last counter deltas, cgroup summaries use CPU microseconds / elapsed
monotonic time. Missing intervals, counter resets or changed CPU topology are not
fabricated idle time. GPU summary `gpu_utilization_mean_percent` is the arithmetic
mean of available polls, with its observation count; no observations means absent
or `null`, not zero. Collected summaries survive timeout/resource/capture failure
alongside the bounded raw history.

These measurements enable utilization diagnosis but cannot certify **useful** 100%
CPU/GPU utilization. Busy hardware is not proof of training throughput; increasing
a timer cannot fix native batching, synchronization or numerical-work inefficiency.

Cleanup control calls consume a shared deadline, with at most one owned stop and one
owned kill attempt. If Docker cannot confirm the result, wait within the bounded
window for the independent native deadline, then inspect once more. The result stays
nonzero with `cleanup_confirmed: false`, a critical error and **no automatic retry**.
Transferred descriptors remain owned by the inside supervisor/native descendants
until actual exit, even if the host closes its duplicate at the end of this window.
The controller must halt the campaign on an unknown state and require operator
reconciliation; timeout is not proof of termination. Kernel-stalled I/O is not made
bounded by a userspace watchdog.

This source preparation does not execute Docker/native training. Only fast stdlib
fixtures with mocked subprocess/telemetry are authorized while the live frozen campaign
continues. Production resource/timeout/recovery behavior needs a separate approved
runtime qualification, including the cross-namespace descriptor handoff; no existing
live helper is changed by this work. The previous ignored runner's Docker qualification
does not qualify this new source automatically.

Source checks include focused stdlib runner/telemetry and controller timeout tests,
using temporary fixtures, deterministic clocks and mocked Docker/native commands.
They exercise the 275/276 boundary, legacy hash preservation, failure without workload
reduction, cleanup headroom and telemetry missing/reset/counter boundaries. The local Unix-socket
fixture verifies shared flock ownership survives closing the host descriptor; it
does not substitute for rootless cross-namespace qualification. Python AST,
whitespace and the 70-line function bound also passed. No Docker, CUDA, native
training, Cargo or full test-suite command was executed for this change.
