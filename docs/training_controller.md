# Tracked training controller

`scripts/train.py` is a standard-library-only Linux campaign controller. Training
always goes through a frozen external runner; checkpoint acceptance always goes
through a frozen native inspector. Python never decodes native checkpoint bytes
and never treats log messages as accepted progress.

**Do not modify `temp/side-actors-teacher-100-20260929-0738`.** Completion of the old
campaign does not transfer ownership to this controller. Use the owner's separately
frozen M25U100 runtime as input to a new campaign; do not adopt or migrate the old
checkpoint. Development verification uses inert fixtures, not old campaign files.

## Commands

Use an existing trusted parent and a new campaign directory beneath the configured
workspace. Edit the placeholder paths/image in the example first:

```sh
python3 scripts/train.py create --config docs/training_controller.example.json temp/my-new-campaign
python3 scripts/train.py status temp/my-new-campaign
python3 scripts/train.py run temp/my-new-campaign
# Alternatively, detach after a durable ownership/preflight handshake:
python3 scripts/train.py run temp/my-new-campaign --detach
python3 scripts/train.py pause temp/my-new-campaign
python3 scripts/train.py resume temp/my-new-campaign --detach
# Explicitly interrupt only the runner child owned by the active controller:
python3 scripts/train.py stop temp/my-new-campaign --immediate
# Offline, explicitly gated reconciliation; never launches training:
python3 scripts/train.py recover temp/my-new-campaign --confirm-offline
```

`fresh` aliases `create`; neither deletes an existing directory or starts training.
`run` requires prepared state, `resume` requires paused state. A failed or stale
campaign requires explicit recovery. No operation retries failed training.
`adopt` is deliberately unsupported and makes no changes.

Detached startup returns `phase: started` only after preflight, exclusive campaign
ownership, and a durable running status. This is **not** a completion receipt; use
`status`. A failed startup or 30-second handshake timeout is an error. Detached
stdout/stderr are discarded; errors after ownership appear in status and runner
evidence remains in the invocation directory. Importing the controller activates
no learner backend; only an explicit `run` or `resume` can launch the runner.

## Configuration, schema 1

Required: `schema: 1`, `trainer`, `total_updates` (1–10000). `inspector` defaults
to `trainer`. Optional `initial_weights` and `opponent_weights` each select a runtime
weights file or a directory containing `drysua.weights.safetensors`. Only that
runtime file is frozen: never optimizer/progress metadata, tensor generations, or
history from the source directory. Relative paths are relative to the config;
`..` traversal is forbidden. Paths must not contain symlinks.

| Field | Default | Bounds / contract |
| --- | --- | --- |
| `invocation_updates` | 1 | controller range 1–16; the deployed runner requires exactly 1 |
| `invocation_seconds` | 235 | payload timeout; exact integer 1–275, never boolean |
| `max_seconds` | 18000 | 1–86400, total monotonic deadline for one run/resume session |
| `history_every` | 20 | 1–10000, checkpoint milestone spacing |
| `training_args` | `[]` | ≤128 strings, each 1–4096 characters |
| `workspace_root` | source repository root | existing absolute directory |
| `mode` | `cpu` | `cpu` or `gpu` |
| `docker_context` | `rootless` | explicit context name |
| `image` | null | pinned `name@sha256:<64 lowercase hex>` required to run |
| `gpu_uuid` | null | explicit `GPU-...` required for GPU; null for CPU |
| `lock_paths` | `[workspace_root/heavy.lock]` | 1–8 distinct existing shared lock files to run |

Unknown fields, duplicate JSON keys, nonfinite numbers, booleans used as integers,
and unreviewed trainer flags are rejected. `training_args` uses an explicit
allowlist in `train.py`. All output paths, opponent selection/weights, device selection, total/invocation
updates, resume, initial weights, checkpoint flags, provenance migration, short
options, and unknown flags remain controller-owned or forbidden. No shell is used.

Omitting `--seed` from `training_args` is supported: the first invocation of a
fresh campaign draws an unpredictable seed, prints it, and records it in the run
scope, and every later `--resume` invocation adopts that recorded seed. Add
`--seed <n>` explicitly only when a campaign must be reproducible from its
config.

The command is `train-annealed`, with a fixed lifetime `--updates` target and a
clipped `--invocation-updates` budget. For example, target 5, budget 2, history 3
produces accepted updates 2, 3, 5. Native trainer semantics validate the remaining
training parameters; the controller does not invent alternate PPO rules.

## Frozen inputs and persistence

Campaigns are exclusively created mode 0700 with:

* `bin/trainer`, `bin/inspector`: executable, read-only copies.
* `inputs/initial_weights/`: optional read-only copied input tree.
* `inputs/opponent/drysua.weights.safetensors`: optional frozen opponent runtime.
  Both weight inputs contain only one runtime file. When an opponent is configured,
  the controller supplies `--opponent weights --opponent-weights <frozen-directory>`.
* `frozen/controller.py`, `train_state.py`, `train_runner.py`: source snapshots.
  Runner source is frozen if installed at creation. Missing runner source blocks
  execution; it is never attached to an existing manifest implicitly.
* `manifest.json`: read-only normalized config, campaign ID, and size/SHA-256
  inventory. Status pins its hash.
* `owner.lock`: persistent-inode advisory lock. `owner.json` records PID, Linux
  start tick, boot ID, UID, ownership token, and the actual child identity.
* `invocations/NNNNNN/job.json`, runner evidence, `checkpoint/`, and `accepted.json`.

Every invocation gets a separate checkpoint directory. Resumption copies only
verified previous checkpoint contents; earlier checkpoints are not overwritten.
All invocation checkpoints are retained, including intermediate ones. History
milestones clip budgets but do not currently prune disk usage.

Accepted receipts are exclusive, read-only, bounded JSON files chained by SHA-256.
They include exact start/expected updates, native inspection, complete checkpoint
inventory, and spec/result hashes. The receipt is fsynced **before** atomically
updating status. Status validates receipt chains and the latest checkpoint,
rejecting corruption, unknown files, unrecorded invocations, and rollback. Receipt
acceptance requires both runner success/stopped-container evidence and exact
native update advancement. Runtime/model mismatch and counter rollback fail.

File writes and parent directories are fsynced; status and owner updates use
exclusive temporary files plus atomic rename. Interrupted file replacement or
preparation is not silently repaired. Frozen-source changes require invoking the
campaign's frozen controller, not silently switching implementations.

Limits: JSON/inspector output 4 MiB, inspector timeout 10 seconds, each artifact
512 MiB, each frozen tree/checkpoint 1 GiB, ≤128 frozen files, ≤10004 checkpoint
files (plus at most 128 directory entries), ≤10000
invocations. Reads are bounded but temporary buffers may overlap (up to roughly
1 GiB memory for maximum-size artifacts).
Inspector output includes stderr in its size bound.

## Session deadline and cumulative disk budget

`max_seconds` is **not** renewed per invocation. A session starts at entry to
`run` or explicit `resume`, including status verification, preflight, checkpoint
copying, runner execution, and inspection. Only a new explicit run/resume session
gets a new deadline. The frozen runner's shared `resolve_timeouts` helper is the
sole source of timeout arithmetic: for payload `P = invocation_seconds`, capture
gets `P + 5`, the runner gets `P + 25` (at most **300 seconds**), and the controller
reserves `P + 55` seconds. Defaults remain **235 / 240 / 260 / 290 seconds**.
Before staging another job, the controller requires that resolved reserve remaining,
including 30 seconds beyond the runner for shutdown, inspection, and persistence. With less
time it saves paused state and an explanatory `last_error`, without creating or
launching another job. Inspector timeouts are clipped to the remaining session
budget, never enlarged beyond ten seconds.

The reserve is rechecked after staging and during execution. If unexpectedly
slow staging consumes it, no runner is launched; the prepared, unexecuted pending
job is retained as failed evidence for offline investigation rather than silently
deleted/retried. Deadline checks also guard inspection and receipt acceptance.
Filesystem syscalls themselves cannot be preempted by this synchronous controller;
the deadline is not a hard real-time guarantee for a blocked kernel I/O operation.

The controller enforces a **100 GiB cumulative campaign accounting limit**,
independent of the runner's disk admission checks. A bounded no-follow inventory
includes every retained checkpoint, log, runner snapshot, frozen input, control
file, and directory beneath the campaign. Each entry is charged the larger of
its apparent size, allocated blocks, or 4 KiB, so sparse files, hard links, and
many tiny files cannot understate the accounting. Symlinks and unsupported file
types are refused; the runner's `runner-locks.sock` is counted without opening it.
The inventory is limited to 2,560,256 entries campaign-wide and 10,132 per active job.

Admission reserves **2 GiB + 64 MiB** before creating a job: checkpoint copy and
output headroom, logs, and control evidence. Insufficient room pauses the session
without launching work. During runner polling, bounded job-only scans check new
growth against the already counted retained campaign; a breach terminates only
the owned runner child and prevents acceptance. Full accounting is repeated
before accepting a checkpoint, preserving 16 MiB for receipts/status updates.
No automatic pruning or retries occur. This is conservative accounting and
admission/monitoring, not a filesystem quota: external writers or an out-of-contract
runner can transiently exceed the threshold between samples. A hard filesystem
write ceiling additionally requires a filesystem quota.

## Pause, immediate stop, and recovery

Pause writes an owner-token-scoped request. The current invocation must finish,
pass runner validation, and pass native inspection before paused state is saved.
Requests from older ownership tokens do not control a new owner.

Immediate stop writes a separate explicit request. Only the active controller
consumes it and signals its own `Popen` runner child, never a PID from a status
file, a process group, another campaign, or a discovered container. SIGINT/SIGTERM
to the controller take the same owned-stop path. The runner is responsible for
owned-container cleanup. After signalling that child, the controller permits
cleanup until five seconds before the **original runner start plus controller
reserve**, then escalates only that child and uses the final five seconds for a
bounded reap. A late timeout never restarts the payload or cleanup budget. This
allows early SIGTERM cleanup to reach the runner's original deadline rather than
killing its Python helper five seconds after the signal. Runtime plus cleanup
waits stay within `P + 55`; lack of confirmed cleanup prevents acceptance.
Inspector and detached-startup termination retain their separate 5/5-second waits.

Recovery acquires the campaign lock and all shared resource locks without
replacing any lock inode. It rejects a live recorded owner/child. A pending
invocation must have a recorded, no-longer-live child identity, a successful
stopped/non-OOM runner result, and an exactly
matching spec and native checkpoint. A receipt fsynced before a status-write
crash is reconciled once without rewriting it or retraining. Recovery never
reduces accepted progress and leaves the campaign paused (or completed).

There is no container-discovery/cleanup protocol in the specified runner API.
Recovery therefore relies on the runner's durable stopped-container result and
exclusive shared-lock ownership; the runner must hold/transfer those locks for
the entire lifetime of native work. Missing result, running/OOM container state,
unsuccessful runner, unknown live ownership, partial update count, or unrecorded
invocation remains a refusal, not an automatic discard/retry. In particular, an
immediate stop without successful final evidence may require manual offline
investigation; this controller does not manufacture a recovery point.

## Integration contract and current mismatches

Runner command: `python <frozen>/train_runner.py run --spec JOB.json`.
The spec has these required keys: integer `version: 1`, `workspace_root`,
`campaign_directory`, `job_directory`, `command` (list), `mode`, `docker_context`,
`image`, `gpu_uuid`, `lock_paths`, `campaign_id`, `invocation_id`.
New campaigns also include `invocation_seconds`: creation normalizes an omitted
config value to 235 and propagates it to every job. Older manifests/specs without
the field retain their original shape and hashes, with an effective default of
235. Read-only status never inserts the field into old manifests, specs, or
results. Failed invocations never retry or reduce `--games`/`--parallel` (including
no automatic games-40-to-8 fallback).
`result.json` requires schema `drysua-training-runner/v1`, integer zero
`returncode`, full 64-hex `container_id`, and `container_state` with exactly typed
`Running: false`, `OOMKilled: false`, integer `ExitCode: 0`. Additional runner
evidence is permitted; explicit `cleanup_confirmed: false` is rejected.

Inspector preflight runs `checkpoint-inspect --contract` before any runner call.
It uses the actual native contract in [checkpoint_inspection.md](checkpoint_inspection.md)
and `src/checkpoint_inspection_json.rs`: `schema: drysua-checkpoint-inspection/v1`,
`kind: contract`, build model/features, capabilities, native limits, schemas,
profiles and numeric semantics. Required capabilities are inspection, verified
annealed history, read-only operation, strict build features, and controller kind
`train-annealed`. Limits must advertise 4 MiB JSON, 10000 snapshots and 10004 files.
An absent command fails explicitly; there is **no help-text fallback**.

Checkpoint inspection invokes `checkpoint-inspect --checkpoint-directory DIR`.
Output follows the actual native projection, including these fields:

* `schema: "drysua-checkpoint-inspection/v1"`
* Nonempty objects `model`, `run`, `ppo`, which must remain identical across
  accepted checkpoints.
* `identity`: exactly four 64-character lowercase SHA-256 strings. **Only
  `scope_sha256` is compared for stability between updates.** Artifact identities
  `manifest_sha256`, `tensor_sha256`, and `runtime_sha256` are expected to vary as
  training advances, not compared to their previous values. Each is cross-checked
  against the independently verified inventory entry selected by `sources.manifest`,
  `sources.tensor`, and `sources.runtime`. The authoritative tensor may be
  `checkpoint.<sha256>.safetensors`; no canonical alias is assumed or required.
  The native inspector owns computation of scope; Python never parses its binary encoding.
* `kind` must be `train-annealed`; `checkpoint` identifies the native schema.
* `progress` includes `updates`, `optimizer_steps`, `rollout_samples`, `games`,
  plus policy/scheduler/curriculum/evaluation, RNG/shuffle, league and mastery fields.
  The four training counters are bounded and cannot roll back. The complete native
  projection, including RNG evidence, is retained in the immutable receipt.
* `adaptive`: object or null; may evolve between invocations.
* `files`: 1–10004 records, each exactly `{path, size, sha256}`, with safe relative
  paths and independently verified bytes.
* `history` must be verified fixed/adaptive annealed history, with ≤10000 snapshots.
* `runtime_status: matched`, `runtime_matches_model: true`, and
  `recovery_required: false` are all required. Diagnostic fallbacks are not adopted.

Controller specs now use the actual runner's **`version: 1`**. The deployment
configuration uses **one update per invocation**, rootless Docker, a pinned image,
and a full GPU UUID. The GPU command explicitly selects CUDA ordinal zero inside
the runner's single-GPU view. A unit test passes a generated GPU/opponent spec
through the actual `train_runner.load_spec` with an inert ELF-header fixture;
it never launches Docker or the trainer. Source-protocol compatibility is tested;
native build/runtime verification remains the authorized operator's responsibility.

Use a trusted parent directory not writable by other users. Frozen permissions
and hashes detect accidental modification, not a malicious same-user process
rewriting both evidence and status. No Linux filesystem immutability flag or
signature is claimed. Ancestor symlink checks do not provide an `openat2`-style
defense against hostile concurrent ancestor-directory replacement.

## Launch the fresh 200-update frozen-M25U100-opponent campaign

`docs/training_controller.example.json` is the deployment template for:
**M40, B20, G1, microbatch 256, actor-value reuse, CUDA0, seed 9001,
generation-games 160, zero-updates 40, total 200**. It uses adaptive environment
scheduling and a maximum 86400-second session. Each invocation advances one update.

Before use, the authorized operator must replace the new trainer path, both
runtime-weight paths, the immutable image digest, and the physical GPU UUID.
Both runtime-weight paths must select the same separately frozen M25U100 runtime.
The new binary must support that runtime for initialization and frozen-opponent
loading and expose the native inspection API; the inspector defaults to that
exact binary. No binary is built or run by editing/preparing this template.

The learner starts from U100 **parameters only**, with fresh Adam, RNG, and campaign
progress. The target is 200 fresh updates (not a resume from progress 100). The
opponent remains the separately frozen U100 runtime throughout. No prior manifest,
optimizer, generation history, or old campaign state is copied.

After editing the template, use a new directory name:

```sh
python3 scripts/train.py create --config docs/training_controller.example.json temp/m25-u100-opponent-u200-new
python3 scripts/train.py run temp/m25-u100-opponent-u200-new --detach
python3 scripts/train.py status temp/m25-u100-opponent-u200-new
```

These are operator launch instructions, not commands executed during development.
Do not reuse the old U100 campaign path. Creation refuses existing destinations.

## Lightweight verification

Fast, pure in-memory timeout checks (mocked processes and clocks; no fixtures or
subprocesses launched):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p test_train_timeouts.py -q
```

The broader suite below launches toy Python fixtures and is **not** part of a
pure-mock-only verification run:

```sh
python3 -m unittest discover -s tests -p test_train.py -q
```

Fixtures are tiny Python executables simulating only the runner and native
inspection JSON. Tests cover foreground and detached startup, clipped progress,
pause/resume, owned stop, recovery after receipt/status interruption, busy locks,
live owners/children, corruption/path escapes, stale ownership, unknown schemas,
changing artifact identities with stable scope, cumulative session deadlines,
disk accounting/admission/overrun handling, contract preflight, output/time limits,
and failed results without retries. Fake clocks, small files, and mocked resource
sizes exercise boundaries without large allocations or waits. No real trainer,
Docker, GPU, Rust build, or active campaign is used.
