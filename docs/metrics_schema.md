# Prometheus telemetry contract

Status: verified and deployed on **2026-09-21**, including the stable exporter,
Prometheus, Grafana and resumed B40 training. All heavy checks used the
unchanged original guard, with evidence under
`artifacts/temp/prometheus-20260921/runs/`. Follow
[experiment-safety.md](experiment-safety.md) for further execution. This document
owns the binary/schema contract; deployment configuration lives separately.

Verification results:

- All-target/all-feature Clippy passed with `-D warnings -D clippy::all`.
- The telemetry verification had **1380 passed, 15 ignored, zero failed** in
  `cargo test --all-targets --all-features --quiet`. The final release, including
  four feared-action regressions, passed **1384 tests, 15 ignored, zero failed**;
  all 13 Criterion test-mode collection cases also succeeded.
- `cargo test --all-targets --no-default-features --quiet`: **1018 passed,
  3 ignored, zero failed**. The no-default-feature standalone binary built.
- The isolated `metrics_enabled_trainers_resume_with_identical_checkpoints`
  test also passed separately. It executes shortened CPU paths for both trainers,
  compares enabled/resumed versus disabled/uninterrupted checkpoint manifests
  (including model/Adam tensor SHA and RNG metadata), and checks persisted
  sample/optimizer/outcome counters, histogram counts, and coverage.
- Real standalone HTTP/resource smoke passed: eight observed CPU series, memory
  and GPU-0 samples, missing-state availability without invented outcomes, finite
  exposition values, 404/405 handling, non-loopback rejection, and SIGTERM exit 0.
  No CUDA learner or production training was started for this HTTP check.
- `cargo fmt`, `cargo machete`, and `git diff --check` passed. No guard resource
  violation was recorded. No frozen artifacts or production checkpoints changed.

Additional checks/limitations: an extra **no-default-feature Clippy with warnings
denied** fails on existing non-builtin dead-code/unused-import warnings in
checkpoint/randomization/training-outcome and old action-test code, not the new
telemetry modules. Those unrelated warnings were not changed. The ordinary
no-default build/tests pass with warnings. Native promtool lives in the private
monitoring runtime rather than PATH: configuration, all 13 recording-rule fixtures,
and actual exporter exposition passed. Prometheus ingestion and Grafana provisioning
were verified through their APIs; see [monitoring.md](monitoring.md).

Production resumed from accepted U135 with the fixed feared-action gates and
`--metrics-directory`. U136 completed all 40 games with zero rejected orders.
Successive checkpoint manifests, sample/Adam counts and persistent outcomes were
checked across restarts: outcomes advanced by 40 without reset or double counting.
Outcome and duration coverage begins after U135; earlier outcomes were not
backfilled. The frozen release and execution evidence are under
`artifacts/temp/prometheus-training-20260921/` and the campaign's `attempt-008/`.

## Commands and ownership

Both `train-full` and `train-annealed` accept:

- `--metrics-directory DIR`: existing private, shared directory for one campaign's
  telemetry. It must be separate from, and not nested with, checkpoint directories.
  Keep this directory constant when invocation/checkpoint/output directories move.
  Only one training writer may hold it. The directory must not be a symlink.
- `--metrics-listen 127.0.0.1:9464`: optional direct, in-process endpoint. Without
  `--metrics-directory`, this is process-local telemetry, suitable for a long-lived
  trainer, **not** a durable segmented-campaign exporter.

Either flag enables Prometheus mode. Neither flag retains legacy behavior. Both
flags may be supplied together. The smoke `train`, gameplay, and RewardObserver
commands are unchanged. Training still requires `builtin`; the standalone exporter
does not require `builtin` or initialize a model, learner, CUDA context, or simulator.

For restarting training processes, use the **same binary** as a stable exporter:

```text
drysua metrics-serve --metrics-directory DIR --metrics-listen 127.0.0.1:9464
```

The exporter requires `--metrics-directory`; its default listener is
`127.0.0.1:9464`. Only literal loopback socket addresses are accepted, including
IPv6 loopback. There is no public-bind override, authentication, dashboard, or
HTTP control API. Exact `GET /metrics` returns Prometheus text format 0.0.4.
Queries, other routes/methods, bodies, and buffered pipelining are rejected;
connections close after one response. SIGINT/SIGTERM gracefully stop the Unix
standalone exporter, close clients, and clean up any resource-collection child.
Direct training listeners shut down with their CLI guard on normal/error return.

Unix signal handling uses the **already locked** `libc 0.2.189` package as a direct,
Unix-only dependency with default features disabled: std has no signal installation
API. No new package or NVML FFI/dependency stack was added. Persistent atomic
replacement/directory fsync and standalone signal shutdown currently require Unix;
other platforms retain training/direct-listener compilation. Host/GPU sampling is
Linux-only; unsupported platforms expose resource availability zero, not samples.

## Durable counters, scope, and coverage

All training progress, outcomes, PPO gauges, and update/stage histograms below
describe **committed checkpoints**, not speculative collections. Absolute sample
and optimizer-step counters come from strict training restoration, never from
adding invocation totals to previously restored totals. Outcomes use the existing
session `SmokeCounters`: per-update deltas are staged once, and invocation counters
start at zero on every resume while durable observed totals are retained.

The bounded transaction is:

1. Complete an update; stage outcome deltas, typed timer measurements, and PPO data.
2. Serialize the existing checkpoint, without changing its bytes or format. Hash
   its exact manifest (which already includes the tensor SHA-256, RNG, and progress).
3. Atomically write and fsync `metrics.pending` **before** checkpoint file writes.
4. Commit and fsync the existing checkpoint manifest.
5. Atomically publish and fsync `metrics.state`, then remove/fsync the pending record.

Resume re-establishes the checkpoint-directory durability barrier after validating
the restored manifest and before telemetry recovery can publish it. Normal removal
of a pending file racing an exporter read is retried within the fixed read bound.

State consists of two fixed-size, versioned, checksummed 858-byte records at most,
fixed crash-temporary filenames, and `.metrics.writer.lock`. Reads are limited to
16 KiB. There is no append-only event log or unbounded update queue. The journal is
independent of the checkpoint format. No metrics field influences policy seeds,
RNG, modifiers, rewards, optimizer choices, scheduling, or checkpoint cadence.

Scope is a SHA-256 of the existing canonical checkpoint run/config encoding plus
linked schema identity and a metrics domain separator. This includes real seeds,
provenance, opponent identity, parallelism, and PPO settings; it does **not** include
logging/metrics flags or output/checkpoint paths. `train-full`'s target is a mutable
gauge, as in its existing checkpoint contract; annealed schedule targets retain
their existing scope rules. Fresh training cannot reuse existing state, even with
the same seed/configuration. Resumed state also matches the exact checkpoint
manifest identity and its absolute update/sample/optimizer progress.

| Resume situation | Behavior |
|---|---|
| Actual checkpoint matches committed state | Restore observed totals, without adding them again. |
| Actual checkpoint matches prepared state | Publish that complete snapshot once; repeated resume is idempotent. |
| Checkpoint remains at committed state; a newer preparation exists | Discard the speculative preparation; resumed training may replay the uncommitted work. |
| Checkpoint is older than state, neither committed nor prepared, or scope/identity differs | Fail with a specific error; never guess/reset historical totals. |
| Corrupt/oversized state, or missing committed state with pending present | Fail; exporter marks unavailable and omits outcome/progress samples. |
| Neither record exists at first opt-in | Establish explicit new coverage at the actual restored update, with zero **observed** outcomes. |

`drysua_training_metrics_start_update = N` means observed outcome/duration coverage
starts **after the checkpoint containing N completed updates**. On first resume,
historical game results before N are unknown—not inferred from games-per-update,
weights, logs, or wins in a previous process. Samples/optimizer steps can still be
restored absolutely because the checkpoint contains them. Deleting both records is
indistinguishable from first opt-in and starts a new, explicitly reported coverage
window; retain the directory and do not delete/reset it to repair an error.
Historical win-history migration is a separate explicit future operation.

Existing metrics scope cannot silently cross a Git provenance migration. Use a new
metrics directory/new coverage window for that change. First opt-in alongside an
already-validated `train-full --migrate-provenance` can rebind the same-update
checkpoint identity only while coverage has no observed updates/outcomes/durations.
It cannot rewrite previously observed history.

The stable exporter only reads committed state, never promotes pending data. An
abandoned pending record (no active writer) makes metrics unavailable until training
resume reconciles it against the actual checkpoint. Missing/corrupt state is an
availability-zero response, not an invented all-zero history. Resource metrics
remain independently available. A valid historical record remains available while
training is inactive.

**Failure policy:** fail the training invocation on a telemetry transaction error,
with a specific diagnostic. Preparation failure prevents that checkpoint write;
publication failure can occur after a durable checkpoint and leaves recovery state.
Do not assume an I/O error rolled back a rename. Timer-hook errors and unexpected
direct-server worker failure latch unhealthy state and fail the next control-plane
operation/final guard; server failure is reported immediately rather than waiting
for shutdown. No success notification
is manufactured after a failed telemetry commit. Neither checkpoints nor telemetry
should be manually deleted as recovery. Exporter read errors are logged on state
transitions and exposed through health/availability gauges.

Persistent counters survive a process ending before its final scrape: the stable
endpoint can expose the final committed totals later. **Process-local counters plus
`rate()` alone do not provide this guarantee.** Absolute totals are authoritative;
`rate`/`increase` need sampled endpoints and cannot recover intervals before the
first Prometheus sample, even with persistence. Keep exporter target identity stable
and scope/coverage changes visible in dashboards.

## Training metric families

All names in this table have the `drysua_training_` prefix. No run IDs, seeds,
paths, process IDs, or entity IDs appear as labels. Absent measurements are omitted.

| Suffix | Type | Labels / meaning |
|---|---|---|
| `updates_completed` | gauge | Absolute durable update count, including restored updates. |
| `updates_target` | gauge | Configured total target; may change under existing train-full rules. |
| `samples_total` | counter | Absolute durable rollout samples, restored from checkpoint. |
| `optimizer_steps_total` | counter | Absolute durable Adam steps, restored from checkpoint. |
| `games_total` | counter | `outcome="win\|loss\|draw\|time_cap"`; committed observed counts since coverage began. |
| `last_update_games` | gauge | Same four outcome labels; last committed **update**, not sum across checkpoint interval. |
| `update_duration_seconds` | histogram | Successful committed updates since coverage began. |
| `stage_duration_seconds` | histogram | `stage="rollout_initialization\|collection\|batch_preparation\|optimization\|finalization"`. |
| `scope_duration_seconds` | histogram | Direct listener only, invocation-local; `scope="session_initialization\|checkpoint_capture_save_runtime_export\|resume_runtime_export"`. |
| `policy_loss`, `value_loss`, `entropy`, `approximate_kl` | gauge | Last committed PPO report; omitted before an observed report. KL uses the existing report's rejected KL when stopped for KL. |
| `generation` | gauge | Annealed only; actual zero-based generation used by the last committed batch. |
| `environment_scale_ratio` | gauge | Annealed only; effective scale in `[0,1]`; zero inside the no-modifier window, including a partially applied cached generation. |
| `parallel_worlds` | gauge | Configured simultaneous worlds. |
| `games_per_update` | gauge | Configured full-episode games; zero denotes reset-window mode. |
| `active` | gauge | Direct: training guard exists. Stable exporter: training writer lock is held. Not proof of learner progress. |
| `last_heartbeat_timestamp_seconds` | gauge | Trainer-supplied Unix seconds at initialization/committed snapshot (or target refresh); **not** exporter scrape time. May be old during a long update. Omitted if unavailable. |
| `metrics_start_update` | gauge | Explicit outcome/duration coverage boundary described above. |
| `metrics_available` | gauge | 1 iff a healthy committed training snapshot can be exposed. |
| `metrics_state_healthy` | gauge | State validation/reconciliation/I/O health; 0 when unavailable or abandoned pending. |

The `|` separators above enumerate literal allowed values, not combined label
values. Win, loss, draw, and time cap are mutually exclusive. Failed/incomplete
collection and infrastructure errors are never game outcomes. Histograms have
cumulative buckets at `0.001, 0.01, 0.1, 1, 5, 15, 60, 300, 900, 3600` seconds, plus
`le="+Inf"`, `_count`, and `_sum`. Each family has one `# HELP` and `# TYPE` declaration;
all sample values and sums are finite. Update/stage histograms persist across
segments; scope/checkpoint histograms intentionally do not and are absent from the
file-only exporter. No KL-stop counter is currently exported.

`metrics_available=0` suppresses outcome, progress, loss, generation and timing
families rather than publishing plausible zero values. Active/health/availability
remain present; a known heartbeat may still be exposed. Use availability and
coverage when interpreting cumulative game totals; compare update progress over
time to detect a stuck trainer, not writer-lock ownership alone.

## Resource metric families

All are gauges, collected **only in the metrics server**, not trainer hot paths:

- `drysua_host_cpu_utilization_ratio{cpu="0".."255"}`: non-idle delta of Linux
  `/proc/stat`'s first eight modes, with idle+iowait treated as idle. Guest fields
  are already included in user/nice and are not double-counted. Initial samples,
  counter regressions, zero deltas, missing/hotplugged CPUs, and invalid samples
  are omitted, not reported as idle CPUs.
- `drysua_host_memory_available_bytes`, `drysua_host_memory_total_bytes`:
  bounded `/proc/meminfo` reads; KiB converted to bytes.
- `drysua_gpu_utilization_ratio{gpu="0".."7"}`,
  `drysua_gpu_memory_used_bytes{gpu=...}`,
  `drysua_gpu_memory_total_bytes{gpu=...}`,
  `drysua_gpu_temperature_celsius{gpu=...}`: optional `nvidia-smi`, no shell.
  Missing command/GPU, unsupported values, malformed output, and timeout omit
  samples. Unsupported index/count sets fail closed rather than truncate secretly.
- `drysua_host_cpu_collection_available`,
  `drysua_host_memory_collection_available`,
  `drysua_gpu_collection_available`: 0/1 collection validity, independent of
  training-state availability. CPU starts at zero availability until a valid delta.

## Bounds and cadence

One server event loop services eight fixed client slots. Requests are at most
4,096 bytes/64 headers; read and write phases each have an absolute two-second
deadline. Responses are bounded to 128 KiB and shared by clients; training and
resource exposition each have a 48 KiB bound. Idle polling is 20 ms. There is no
per-client thread, unbounded client/task queue, or per-decision registry lock.

Training-state reads/rendering and resource collection occur at least five seconds
apart **after the preceding collection finishes**, independent of scrape frequency.
Responses are cached. Proc files are bounded to 64 KiB and CPU indices to 256.
GPU discovery and sampling share a one-second deadline, use at most one owned child
at a time, limit stdout to 4,096 bytes per invocation, and support at most eight
GPUs. Timeout triggers kill/reap cleanup. std cannot impose a hard deadline on a
kernel-stalled file open, process creation, or process reaping; this is not a hard
real-time service. No shell, reader-thread queue, or GPU library is loaded.

## Text-log compatibility exception

Prometheus mode suppresses per-episode `event=map2_episode_reward` dumps, collection metric summaries,
periodic training update/scope timing lines, mastery metric lines, generation
metric lines, and aggregate training reward dumps. Startup, errors, warnings,
checkpoint notifications, and shutdown remain.

The exact existing `episode:` records (one per completed game, including ticks,
retained samples, actions, and all existing fields), `checkpoint:`, and final
`training complete:` / `annealed training complete:` records are retained as
**guarded-runner control/audit protocol**, even though they carry numerical fields.
In the B40 campaign the controller requires 40 episode events per completed update.
These lifecycle/game events are logs, not graph telemetry; they are not the metrics source. No controller
rewrite, RewardObserver removal, or gameplay-log suppression is part of this change.
