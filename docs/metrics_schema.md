# Prometheus telemetry contract

Native HTTP, journal recovery and metrics-enabled trainer resume are qualified;
current U428's 858-byte state and checkpoint binding passed read-only verification.
Strict all-feature and no-default Clippy/tests passed in the final refactor.
See `artifacts/deslop-20260922/REPORT.md` for exact executed results, not the retired
test-count diary (`git show 2ddb68b:docs/metrics_schema.md`). Docker runtime remains
unqualified; [deployment/security](docker.md), [database/credential operations](monitoring.md)
and [execution safety](experiment-safety.md) are separate contracts.

## CLI and endpoint

Both `train-full` and `train-annealed` accept `--metrics-directory DIR` and optional
`--metrics-listen 127.0.0.1:9464`; either enables Prometheus mode, both may coexist.
DIR must be existing, private, non-symlink, separate from/not nested with checkpoints,
and shared consistently across campaign segments with one writer. Listen-only is
process-local, not durable. Neither flag means legacy logging. Gameplay, smoke train
and RewardObserver behavior is unchanged. Training requires builtin; the exporter
requires no builtin, model, learner, CUDA context or simulator initialization:

```text
drysua metrics-serve --metrics-directory DIR --metrics-listen 127.0.0.1:9464
```

DIR is required; the listener defaults to 127.0.0.1:9464. Only literal loopback IPv4/
IPv6 addresses are accepted. Exact GET /metrics returns text format 0.0.4; queries,
other routes/methods, bodies and buffered pipelining are rejected. One response per
connection; no authentication/control API. Unix SIGINT/SIGTERM closes clients and
owned collection children; direct listeners close with the training guard. Persistent
atomic replacement/directory fsync and standalone signal shutdown require Unix.
Resources are Linux-only; unsupported platforms report availability zero.

## Durable publication and recovery

Training samples, optimizer steps, outcomes, PPO gauges and update/stage histograms
describe **committed checkpoints**, not speculative collections. Absolute counters
come from restored progress; invocation outcome counters contribute staged deltas
once, never re-add previously committed totals. The transaction is:

1. Stage a completed update and hash the exact existing checkpoint manifest bytes.
2. Write/fsync metrics.pending before checkpoint writes; commit/fsync the manifest.
3. Replace/fsync metrics.state, then remove/fsync pending. Restore reestablishes the
   checkpoint directory durability barrier before journal recovery publishes anything.

At most two checksummed/versioned 858-byte records, fixed temporary names and
.metrics.writer.lock; reads are <=16 KiB. No unbounded journal/queue and no telemetry
input to policy, RNG, rewards, scheduling, optimizer or checkpoint cadence.
Scope hashes canonical run/config/schema identity, including provenance/opponent/
parallelism, excluding metrics flags and paths. Train-full target remains mutable;
annealed target keeps its schedule scope. Fresh runs cannot reuse existing state.

| Actual checkpoint / journal condition | Recovery |
|---|---|
| Matches committed | Restore totals without adding again |
| Matches pending | Publish once; repeated recovery is idempotent |
| Matches committed with newer pending | Discard speculative preparation |
| Ahead/behind incompatible state, wrong scope/identity, corrupt/oversized record, or pending without committed | Fail; never guess/reset history |
| Neither record at first opt-in | Start explicit zero-observation coverage at restored update |

Coverage N begins **after** checkpoint N. Earlier outcomes are unknown; samples/steps
remain absolute. Deleting both records silently starts a new coverage window, so
never use deletion as repair. Existing coverage cannot cross provenance migration:
use a new metrics directory. First opt-in with validated train-full migration may
rebind same-update identity only while observed outcomes/updates/durations are empty.
Exporter never promotes pending; abandoned pending makes training metrics unavailable
until resume reconciles. Valid historical state stays available without an active writer.

Telemetry transaction errors fail the invocation. Preparation failure prevents the
checkpoint write; publication failure may follow a durable rename—do not assume
rollback. Timer/direct-worker failure latches unhealthy state and fails control-plane
operations/finalization; worker failure is reported immediately. Exporter logs read
health transitions. Resources remain independently available. Durable totals survive
missed final scrapes, but rate/increase cannot reconstruct pre-scrape intervals.

## Training families (`drysua_training_` prefix)

No run/seed/path/PID/entity labels. Missing measurements are omitted, never fabricated.

| Suffix | Type | Meaning / fixed labels |
|---|---|---|
| updates_completed, updates_target | gauge | Absolute committed updates / configured total target |
| samples_total, optimizer_steps_total | counter | Absolute restored samples / Adam steps |
| games_total | counter | Observed committed outcomes since coverage; outcome="win\|loss\|draw\|time_cap" |
| last_update_games | gauge | Same labels; last update, not entire checkpoint interval |
| update_duration_seconds | histogram | Successful committed updates |
| stage_duration_seconds | histogram | stage="rollout_initialization\|collection\|batch_preparation\|optimization\|finalization" |
| scope_duration_seconds | histogram | Direct listener, invocation-local: scope="session_initialization\|checkpoint_capture_save_runtime_export\|resume_runtime_export" |
| policy_loss, value_loss, entropy, approximate_kl | gauge | Latest committed PPO report; rejected KL when stopped for KL |
| generation, environment_scale_ratio | gauge | Annealed generation and effective [0,1] scale, including cached partial generation |
| parallel_worlds, games_per_update | gauge | Simultaneous worlds / full games; zero games denotes reset-window |
| active | gauge | Direct guard exists / stable writer lock held, not progress |
| last_heartbeat_timestamp_seconds | gauge | Trainer timestamp at initialization/commit/target refresh, never scrape time |
| metrics_start_update | gauge | Explicit outcome/duration coverage boundary |
| metrics_available, metrics_state_healthy | gauge | Exposable healthy committed snapshot / validation-reconciliation-I/O health |

Pipe separators enumerate literal labels. Outcomes are mutually exclusive; incomplete
work/infrastructure failures are not game results. Histograms have cumulative bounds
0.001, 0.01, 0.1, 1, 5, 15, 60, 300, 900, 3600 seconds, +Inf, count and sum; one HELP/
TYPE declaration and finite values. Update/stage histograms persist; scope histograms
are absent from file-only export. No KL-stop counter. Unavailable state suppresses
training value/timing families, keeping active/health/availability and known heartbeat.

## Resource families and bounds

Collected only by the server, describing whole host/device, not individual bots:

- drysua_host_cpu_utilization_ratio{cpu="0".."255"}: first eight /proc/stat modes,
  idle+iowait treated as idle, guest not double-counted. Initial/reset/zero-delta/
  missing-hotplug/invalid samples are omitted rather than reported as idle.
- drysua_host_memory_available_bytes and drysua_host_memory_total_bytes: bounded
  /proc/meminfo reads, KiB converted to bytes.
- drysua_gpu_utilization_ratio, drysua_gpu_memory_used_bytes,
  drysua_gpu_memory_total_bytes, drysua_gpu_temperature_celsius: gpu="0".."7".
  Optional nvidia-smi without shell; missing/unsupported/malformed/timeout samples
  are omitted, and unsupported index/count sets fail closed.
- drysua_host_cpu_collection_available, drysua_host_memory_collection_available,
  drysua_gpu_collection_available: independent 0/1 validity; CPU needs a valid delta.

Eight client slots, <=4096-byte requests/64 headers, absolute 2s read/write deadlines,
128 KiB shared response, 48 KiB per training/resource exposition, 20ms idle polling.
No per-client thread or hot-path registry lock. Cached collection starts >=5s after
the preceding completion, independent of scrapes. Proc files <=64 KiB; GPU discovery/
sampling share 1s, one owned child, <=4096 stdout bytes per invocation and <=8 GPUs.
Timeout kills/reaps; kernel-stalled open/spawn/reap cannot have hard real-time guarantees.

## Log compatibility

Prometheus suppresses redundant reward/collection/timing/mastery/generation metric
dumps, not startup/errors/warnings/shutdown. Exact episode:, checkpoint:, training
complete: and annealed training complete: control/audit records remain; B40 requires
40 completed-game episode records per update. These logs are not the graph data source.
