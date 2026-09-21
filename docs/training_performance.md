# Training throughput experiments

Production now batches actor sampling and greedy warmup on the learner device.
Actor collection and optimization remain sequential: each rollout binds the
current model identity, which the direct trainer validates before optimization.
Production does not publish CPU snapshots or send rollouts through loopback channels.
Each update derives bounded
per-environment RNG streams from the checkpointed master RNG.

The rollout limit is 2048 decisions per environment, with an unchanged global
limit of 32768 transitions. Environments remain disposable at update boundaries,
so resume requires no simulator-state serialization.

## Measurements

RTX 5090, F32, one PPO epoch; elapsed times include command startup/build overhead.
All runs are retained under `artifacts/temp/`.

| Experiment | Environments | Rollout | Updates | Samples | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| v33 initialized baseline | 4 | 256 | 8 | 8192 | 211.3 |
| Fresh scalar actor | 4 | 2048 | 8 | 65536 | 665.7 |
| Fresh CPU batched actor | 4 | 2048 | 8 | 65536 | 480.8 |
| Fresh CPU batched actor | 16 | 2048 | 2 | 65536 | 351.5 |
| Fresh CPU batched actor and warmup | 16 | 2048 | 2 | 65536 | 253.1 |
| Fresh CUDA batched actor and warmup | 16 | 2048 | 2 | 65536 | 79.0 |

These are exploratory end-to-end runs, not controlled same-trajectory speedups.
Initial weights, RNG scheduling, legality masks and optimization work differ
between some rows. The last run applied only two optimizer steps: KL early
stopping prevented the remaining minibatches. Its roughly 830 retained samples/s
must not be extrapolated as a four-epoch release-training guarantee.

The former launcher used four environments and eight decisions per update.
Across the full phase/baseline cycle it discarded 30600 warmup decisions while
retaining only 256 transitions. Longer rollouts amortize this work; batching
removes repeated feature-trunk inference and device transfer overhead.

Before release training, verify throughput and gameplay retention from an accepted
behavioral anchor with the actual epoch/minibatch settings. Passing the throughput
experiment does not qualify its random-policy weights for deployment.

## Bounded collection benchmark (criterion)

`cargo bench -p drysua --features builtin --bench training` measures the
production complete-episode collector over a fixed decision window, without the
optimizer, checkpoints or full-episode wall time:

- `collection_serial/single-thread` is one environment advanced on the calling
  thread through the same prepare/sample/step/flush functions the worker pool
  uses; the single-thread reference.
- `collection_parallel/environments/{2,4,6,8,16,26}` is the production worker
  pool over the same window, one case per even environment count up to the
  invariant ceiling.
- `collection_pipelined/groups_{2,4}/{8,16,26}` is the same worker pool with
  `--pipeline-groups` collection groups, the opt-in mode below.
- `DRYSUA_BENCH_PHASES_ONLY=1 cargo bench ...` prints one untimed per-phase
  window per default/pipelined case
  (`phase-accounting: groups=... prepare_ns=... advance_ns=... forward_ns=...
  barrier_ns=... apply_ns=... flush_wait_ns=... evaluator_ns=...`);
  `DRYSUA_BENCH_PHASES=1` adds the same table to a sweep. Timed cases never
  carry instrumentation.

Fixed configuration: seed `10141700`, update `0`, `320` untimed warmup decision
rounds (the measured window starts at tick `961`, just after the 900-tick
pregame), `64` measured decision rounds (`192` ticks), mastery-v1 at its fresh
Weak stage (the only schedule valid across the full environment range). Each
criterion case sets throughput to the decisions in its window, so
ns/decision, decisions/s and speedup/efficiency against the serial case read
directly from its report. Setup (worlds, model, warmup) is outside the timed
section; the library asserts the exact window tick count, stream count and an
empty terminal set on every run. `cargo bench` builds this target under
`[profile.bench]` (release inheritance, opt-level 3); the `[profile.dev]` and
`[profile.test]` opt-level 2 settings do not apply to it.

```sh
# Smoke: one untimed pass per case, about a minute of wall time.
cargo bench -p drysua --features builtin --bench training -- --test

# Bounded measurement: 10 samples per case (~40-65 s serial..E8, ~95-135 s E16/E26).
cargo bench -p drysua --features builtin --bench training -- \
    --sample-size 10 --warm-up-time 1 --measurement-time 5

# One case only, e.g. the invariant ceiling.
cargo bench -p drysua --features builtin --bench training -- \
    --exact collection_parallel/environments/26 \
    --sample-size 10 --warm-up-time 1 --measurement-time 5

# CUDA learner forward is opt-in; collection stays CPU and is the scaling story.
DRYSUA_BENCH_DEVICE=cuda cargo bench --features builtin,cuda --bench training -- ...
```

Measured 2026-09-18 on 8 physical cores / 8 threads, one guarded job at a time
(criterion point estimate; lower / upper are the interval edges):

| E | wall/sample, ms | ns/decision | decisions/s | speedup vs 1 | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 176.8 / 177.8 / 178.9 | 2762656 / 2778906 / 2796094 | 360 | 1.00x | 100.0% |
| 2 | 234.0 / 235.6 / 237.0 | 1828281 / 1840391 / 1851875 | 543 | 1.51x | 75.5% |
| 4 | 349.5 / 351.0 / 352.7 | 1365117 / 1371211 / 1377656 | 729 | 2.03x | 50.7% |
| 6 | 451.3 / 452.5 / 453.6 | 1175286 / 1178281 / 1181354 | 849 | 2.36x | 39.3% |
| 8 | 564.6 / 566.6 / 569.2 | 1102695 / 1106660 / 1111719 | 904 | 2.51x | 31.4% |
| 16 | 1006.8 / 1010.5 / 1014.1 | 983203 / 986816 / 990332 | 1013 | 2.82x | 17.6% |
| 26 | 1582.7 / 1588.0 / 1592.7 | 951142 / 954327 / 957151 | 1048 | 2.91x | 11.2% |

The curve plateaus at 2.91x: in the default one-group collector the batched
learner forward and the sampler stay on one thread, so environment-parallel
stepping and encoding cannot scale past that serial fraction. The absolute
ns/decision is one early-game seed and window; treat the shape (speedup and
efficiency) as the portable result and re-measure before quoting absolutes for
release training.

## Pipelined collection groups (`--pipeline-groups`)

`train-full --pipeline-groups N` breaks the single per-decision global barrier:
the complete-episode environments are split into `N` fixed, disjoint **collection
groups** (`N` in `{1, 2, 4}`; default `1` is the historical collector). Each
group owns contiguous whole environment pairs — the mirrored policy seats of one
seed pair never split across groups, and group sizes differ by at most one pair.
A group runs its own bounded worker pool, its own dedicated flush evaluator, its
own batched forward/sampling and its own decision barrier on its own orchestrator
thread, so one group's forward or reply application overlaps another group's
world stepping and encoding. A finished stream leaves its group's active set
exactly as in the default collector.

The update boundary stays **global**: an update still collects exactly
`environments` complete episodes, derives mastery from all `environment`
outcomes, and runs one PPO batch/optimizer step. Group results are merged in
fixed group order, never completion order, so repeated runs with the same
seed/config produce identical episodes and identical policy/Adam tensors.

Two differences are contractual, not incidental:

1. **Batch composition changes.** Group forwards are smaller GEMMs, so the
   sampled choices and the merged rollout order differ from the one-group
   collector. Cross-mode byte comparisons do not apply; within a mode the
   contract is deterministic byte-for-byte.
2. **The mode is part of the checkpoint run scope.** A grouped run appends
   `--pipeline-groups N` to the canonical command line in `checkpoint.meta`, so
   strict resume rejects a scope mismatch in both directions: an old or default
   checkpoint cannot silently continue as a grouped run, and a grouped
   checkpoint cannot silently continue as a default run. `--migrate-provenance`
   also rejects the mismatch; it can only rebind the Git commit of an otherwise
   identical scope. No checkpoint schema version changes and no artifact is
   relabeled.

The grouping applies only to complete-episode collection; grouped window-mode
configurations are rejected. `--pipeline-groups 3` is rejected (only `1`, `2`,
`4` are defined) and a group count above the number of environment pairs is
rejected, so every group owns at least one paired batch.

Recommended configuration is chosen by measurement on the target host; the
bounded criterion sweep and the real-update A/B under `artifacts/temp/` are the
reference (criterion: `collection_pipelined/groups_{2,4}/{8,16,26}`).

### Measured 2026-09-18 (8 cores, one guarded job at a time)

Criterion fixed window, seed `10141700`, 320 warmup + 64 measured rounds, CPU
learner (point estimates):

| E | default dec/s | groups=2 dec/s | groups=4 dec/s | G4 vs default | G4 vs 1 thread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 867 | 1199 | 1727 | 1.99x | 4.86x |
| 16 | 992 | 1511 | 2299 | 2.32x | 6.46x |
| 26 | 1045 | 1545 | 2264 | 2.17x | 6.36x |

Real training A/B with the CUDA learner (canonical update command, 2 updates,
two interleaved repeats per cell, games/hour from update wall):

| E | groups=1 | groups=2 | groups=4 |
| ---: | ---: | ---: | ---: |
| 8 | 752 | 606 (0.81x) | 395 (0.52x) |
| 16 | 1042 | 878 (0.84x) | 607 (0.58x) |
| 26 | 1207 | 1045 (0.87x) | 777 (0.64x) |

The same binary with the CPU learner at E16 moved one update from 191.4 s to
101.4 s (collection 155.8 s to 66.1 s): 301 to 568 games/hour end-to-end.

Read the device split plainly: on a CPU learner the batched forward and sampler
are the dominant serial fraction, so overlapping group forwards with other
groups' stepping collapses the barrier (phase accounting in the session report
shows forward work rising in total but the wall falling). On the CUDA learner
the forward is already cheap, and concurrent group forwards serialize on the
device with per-call overhead that grows as the batch shrinks, so groups lose
there. Use groups for CPU-learner runs; keep the default one-group contract for
CUDA-learner runs until the CUDA sampler path is measured independently.
Within a mode, repeats are byte-identical; only the interleaving of concurrent
episode log lines differs between groups.

## Accelerator-transfer experiment

The B40 campaign keeps using its original frozen binary. The changes below were
verified during a user-authorized pause but **were not deployed: the end-to-end
gain was within measurement variation**. All tests and A/B runs used exclusive
guarded execution under `experiment-safety.md`, then training resumed immediately.

The current learner reads each named gradient separately after every backward
pass. The prepared path packs same-device F32 accelerator gradients in descriptor
order and performs one host read instead of up to 62. Missing spans remain positive
zero. CPU gradients keep their direct path. Concatenation only copies data:
microbatch weighting, accumulation order, Adam arithmetic and serial global-norm
reduction are unchanged. Packed storage is at most 6.49 MiB; noncontiguous inputs
can require another 6.49 MiB of temporary device data, excluding backend metadata.

PPO also reuses its existing pre-candidate parameter snapshot as Adam input,
removing a second export of the same 1,700,020 parameters under the same exclusive
lock. The old moments, candidate KL check and exact rollback remain unchanged.
This removes 6,800,080 logical transfer bytes per attempted Adam candidate.

The annealed loop now uses the existing `training_update_timing` record with
`mode=annealed`. It separates rollout initialization, collection, batch preparation,
optimization and finalization. Collection includes per-batch world creation and
generation snapshot handling; optimization includes shuffled minibatch preparation
and transfers, not just GPU execution. Checkpoint save/export remains a separate
scope. Durations never enter game state, RNG or optimization decisions.

Existing B40 segment 0038 took 74.92 seconds including its wrapper; after its
cgroup task count fell from 50 to 9, about 23 seconds remained. This is consistent
with a serial learner tail, not proof of its exact duration or GPU utilization.
Its 20,815 samples and four epochs imply 1,304 gradient microbatches: packing can
reduce their gradient host-read calls from at most 80,848 to 1,304. Device copies
have a cost too, so these operation counts are not a wall-time prediction.

### Verification contract

- Run Clippy, the full Rust suite, formatting and dependency checks serially under
  the existing guard. Run the ignored CUDA tests in `model::transfer_tests` too.
- Compare old/new gradient bits from the same backward pass; compare parameters,
  both Adam moments, RNG, applied steps and rejection/rollback behavior.
- Compare complete B40 updates from copies of the same committed checkpoint and
  generation, with identical samples, epochs and KL decisions. Any build-scope
  rebinding must be explicit on the copies. Do not touch the campaign artifacts.
- Report collection, preparation, optimization and total wall time separately.
  Fewer optimizer steps or different trajectories do not establish acceleration.

### Measured 2026-09-21: no meaningful end-to-end improvement

Clippy, 1,135 ordinary Rust tests, the two explicitly selected ignored CUDA
transfer/PPO tests, formatting and dependency checks passed. Four full B40 runs
started from independent copies of the same committed U92 checkpoint and trained
U93 with unchanged N1000/M40/B40/K200/S200 settings. Candidate copies changed only
the explicitly recorded build provenance; benchmark work was not promoted into
the production campaign.

| Execution order | Binary | End-to-end capture seconds |
| --- | --- | ---: |
| 1 | Original | 68.483 |
| 2 | Candidate | 69.714 |
| 3 | Candidate | 67.694 |
| 4 | Original | 69.662 |

Original mean: 69.073 s. Candidate mean: 68.704 s, only 0.53% lower and below the
5% admission threshold. Each run collected the same 17,897 samples, applied 36
Adam steps, and produced identical 40 episode records, parameters, both moments,
RNG/progress and snapshots. Runtime exports matched; manifests matched after
normalizing only the authorized build identity. Correctness passed, but the
source-level reduction in host-read calls did not establish a useful speedup.

Candidate-only collection/optimization times were 48.944/20.349 s and
47.232/20.040 s. The original was not rebuilt to add timers, so there is no
measured original-versus-candidate learner-only speedup. Do not attribute the
remaining approximately 20-second learner phase to transfers without profiling.

The production campaign resumed from its real U92 on the original binary
(`cb812d81…`), not from an A/B output. Evidence is in
`artifacts/temp/ppo-transfer-check-20260921/` and
`artifacts/temp/annealed-teacher-20260920/attempt-007/`.

### More cores

B40 already has 40 persistent world workers. Moving to 16 cores / 32 hardware
threads can accelerate world stepping and preparation without changing batching.
It does not remove serial host work or device synchronization; 32 SMT threads
are not 32 independent physical cores. These transfer changes introduce no fixed
CPU count, affinity, quota or global library-thread setting. Further host-gradient
or Adam parallelism should partition independent parameter coordinates while
preserving each coordinate's arithmetic order and the serial norm. It is deferred
until phase measurements establish its value; overlapping smaller CUDA collection
groups previously made end-to-end training slower, as recorded above.
