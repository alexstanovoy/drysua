# Opt-in concurrency prototypes

## Idle-host verification — 2026-09-22

The production campaign was stopped for this verification. All-target/all-feature
Clippy, the final **1404-test** library suite (19 ignored), both binary targets,
all **13 collection smoke cases**, and the separately selected CUDA actor/folding
tests passed. Formatting, dependency and diff checks also passed. No resource
events or leftover probe cgroups occurred; production was not resumed.

The test harness now accepts bounded `DRYSUA_PROBE_ROUNDS`, `DRYSUA_PROBE_EPOCHS`
and `DRYSUA_PROBE_MINIBATCH` overrides. Defaults remain 40/1/128. With 1024 rounds,
four epochs and minibatch 2048, warmup plus ABBA comparisons used 5081 samples and
12 Adam steps per trial. Parameters, both moments, RNG, report bits and work matched.

| Mode | End-to-end time change versus its reference |
| --- | ---: |
| No-change control | -0.86% |
| A: early Continue | +3.28% |
| B: minibatch prefetch | +0.01% |
| C: two workers | -1.80% |
| C: four workers | -3.07% |
| C: eight workers | -2.49% |
| A+B+C: two workers | -0.98% |

Negative means faster. The best candidate, **C with four workers**, was also checked
on forty complete games in a release test binary: **68.010→66.529 seconds** overall
(-2.18%), **22.013→20.817 seconds** in PPO (-5.43%). Collection was effectively
unchanged. The 19411 samples, 40 Adam steps, forty episode records, parameters,
moments, RNG and report bits matched exactly. This full comparison is one
reference-first pair, not a statistically established production percentage.

**Recommendation:** leave A/B and the combination disabled. C/four workers is a
modest candidate for a later authorized deployment, not automatic activation or
proof of scaling to sixteen/thirty-two cores. Evidence and exact values:
`artifacts/temp/concurrency-idle-20260922/{REPORT.md,results.csv,jobs/}`.

## Earlier background verification

Implementation and short CPU/CUDA correctness checks completed through the authorized
background runner on 2026-09-21. The library suite passed 1,402 tests with 19 ignored;
the two targeted ignored CUDA parity tests passed, and every CPU/CUDA base/A/B/C/all
short probe passed. **This is short-workload correctness evidence, not a demonstrated
production speedup or permission to enable the flags on main.** The original aggregate
all-target invocation remains recorded as a deadline abort; after owner review its
remaining smoke case passed separately without raising any cap or repeating the suite.
Main training, frozen binaries and real checkpoints were not changed.

## Initial verification wave and mandatory stop

Evidence is under `artifacts/temp/concurrency-probes-20260921/jobs/`:

| Job | Outcome | Guard wall seconds |
| --- | --- | ---: |
| `concurrency-fmt-001` | `cargo fmt` completed | 1.104 |
| `concurrency-clippy-001` | Compile failure: cast/comparison in the memory-bound assertion parsed as generics | 4.216 |
| `concurrency-clippy-002` | Clippy failure: nested Continue-reply return type triggered `type_complexity` | 6.438 |
| `concurrency-clippy-003` | Required all-target/all-feature Clippy with denied warnings passed | 5.210 |
| `concurrency-tests-all-001` | Library passed; all-target command aborted at payload deadline | 235.120 |

The two specific fixes were parentheses around the `as u64` cast and a `ValuedReply`
type alias. No arithmetic, scheduling or production-work change was made to fix them.
The full library result was **1,402 passed / 0 failed / 19 ignored in 35.64 seconds**,
including the nonignored concurrency and invocation-limit tests. Both binary test
targets completed with zero tests. Twelve collection smoke cases printed `Success`;
`collection_pipelined/groups_4/26` started but did not report completion before the
235-second payload deadline (240 seconds includes five seconds for cleanup).

This was a **failed/incomplete all-target invocation**, not an all-target pass. The
guard removed only the probe cgroup and cleared its ACTIVE record. Per the owner's
stop-on-resource/deadline-abort rule, there was no retry, CUDA run, timing probe,
additional formatter run or `cargo machete` after the abort. `git diff --check`
passed. The owner subsequently reviewed that deadline-only abort and authorized the
isolated continuation below. Original failed-job evidence was not overwritten.

Probe peak memory was 8,277,495,808 bytes, with zero recorded probe memory/pids
limit events. The last resource sample recorded main U380 accepted/U381 running,
age 38.52 seconds, no main memory/pids events, CPU 67 C, GPU 44 C, available RAM
50,180,046,848 bytes and free VRAM 32,525,778,944 bytes. These are final samples,
not asserted extrema. The runner enforced nice 19, memory high 10/max 12 GiB,
swap 0, pids 512 and the documented main-health/temperature/free-memory limits.

Read-only weights were pinned once from accepted `attempt-008/history/update-0376`.
Their before/after SHA256 is unchanged:
`d52047764c4a7b43d6ef911f871724d91daee02f7961caa28c2d64698f2f3b19`.
They were not used in the initial wave, and were then loaded read-only by every
continuation probe. The same hash was verified again after the runs.

## Authorized isolated continuation: actual results

- `concurrency-smoke-g4e26-001`: passed in 95.037 s with explicit `--test --exact
  collection_pipelined/groups_4/26`. Exactly that Criterion case printed `Testing`
  and `Success`; the benchmark's existing unconditional setup checks still run.
  This was not statistical benchmarking. All thirteen smoke cases now have a
  successful result across the two jobs; the original invocation remains aborted.
- `concurrency-fmt-002`, `concurrency-fmt-003`: passed. `concurrency-machete-001`:
  passed, no unused dependencies. Final `git diff --check`: passed.
- `concurrency-lib-build-001`: prebuilt the library tests under the CPU budget.
- `concurrency-cuda-actor-001`: 1 passed, 0 failed, 1,420 filtered out; test body
  1.23 s. Covers mixed actions and late-error drain/RNG protection.
- `concurrency-cuda-fold-001`: 1 passed, 0 failed, 1,420 filtered out; body 0.46 s.
  Covers exact optimizer/report parity and candidate rejection.
- The ten `concurrency-probe-{cpu,gpu}-{base,a,b,c,all}-001` pairs passed state/RNG/
  work checks. A test-only verification gap was then closed: the probe now checks
  every latest-report floating field by `to_bits`, plus all report counters.
  No production behavior or workload changed. `concurrency-clippy-004` passed
  all-target/all-feature Clippy with warnings denied after that addition.
- All ten fresh `...-002` probe pairs passed the stronger report checks. Each
  invocation reported 1 passed, 0 failed, 1,420 filtered out. No resource abort
  occurred in this authorized continuation. No worker-count search was performed.

The earlier single-pair table below is sweep `002`, measured invocation seconds **reference →
candidate**. These times include session initialization and checkpointing, but not
Cargo build or post-run hash verification. Reference is always first, so cold/warm
effects are not randomized away. Exact nanoseconds, collection/PPO phases and job
IDs are retained in `artifacts/temp/concurrency-probes-20260921/results-002.csv`.

| Mode | CPU seconds | CUDA seconds | State/RNG/report/work parity |
| --- | ---: | ---: | --- |
| base | 3.716 → 3.780 | 1.291 → 1.075 | Passed |
| A | 2.750 → 2.685 | 1.314 → 1.131 | Passed |
| B | 2.778 → 2.684 | 1.202 → 1.038 | Passed |
| C | 3.897 → 4.003 | 1.415 → 1.178 | Passed |
| all | 4.077 → 3.917 | 1.543 → 1.274 | Passed |

Every reference/candidate used U376 weights, seed 9001, forty worlds, forty decision
rounds, one epoch, effective minibatch 128 and unchanged 64-row microbatches. Each
performed 1,600 decisions, 4,800 ticks, retained 161 samples and applied two Adam
steps. Continue was 1,394/1,600 = **87.125%**. C/all requested and resolved two fold
workers; other modes used one. CPU state hash was `2374791058f24cd2`, CUDA
`e56a68857d4a1800`, constant across modes within each backend. Full parameter/moment
bit comparisons, actor/shuffle RNG state, progress and report checks—not merely
hash equality—passed. Cross-backend bit identity is not claimed.

**Do not attribute the apparent wall reductions to the flags.** The unchanged CUDA
base control improved 16.76% between its first and second invocation. In C, even
collection (which C does not change) fell from 1,110.06 to 997.69 ms, whereas PPO
changed only 115.51 to 113.29 ms. Warmup, main-training phase and contention confound
these tiny fixed-order pairs. This is neither full-game/end-to-end throughput
qualification nor evidence for 16-core/32-thread scaling. Keep modes opt-in.

## Final test-only warmup + ABBA check

`DRYSUA_PROBE_BALANCED=1` adds one untimed baseline warmup followed by four fresh
trials: **A(reference), B(candidate), B(candidate), A(reference)**. The arm names
are independent of the design names A/B/C. Missing/`0` retains the old single pair;
other flag values fail with an explicit error. Each trial has its own new output
directory and calls the fresh initialization path with the same U376 weights,
seed, dimensions and zeroed Adam/new RNG. No learned state carries between trials.
The warmup is excluded from means but included in correctness comparisons.

Only `src/tests/training_concurrency.rs` changed for this final methodology check.
Existing collection/optimization timing lines are enclosed by labelled trial
markers; four invocation times and their ABBA means are printed separately. No
production execution or telemetry implementation changed and no framework was added.

All five `concurrency-balanced-{base,a,b,c,all}-001` CUDA jobs passed on their first
runs: 1 test passed each, 0 failed, 1,421 filtered out. Every trial had the same
initial fingerprint `cb368d1139c8d202`, final state hash `e56a68857d4a1800`, full
parameter/moment/RNG/report parity and unchanged work: 1,600 decisions, 161 samples,
two optimizer steps and 4,800 ticks. C/all resolved two workers. The new pure
flag/order test passed; formatting, all-target/all-feature Clippy with denied
warnings, machete and diff checks passed. The prior 1,402 library-test result was
not repeated or relabelled as a new full-suite run.

Means below exclude warmup. Positive wall change means the candidate was slower.
Exact four-trial nanoseconds are in
`artifacts/temp/concurrency-probes-20260921/balanced-results.csv`.

| Design | Wall mean ref → candidate, ms | Change | Collection mean ref → candidate, ms | PPO mean ref → candidate, ms |
| --- | ---: | ---: | ---: | ---: |
| base/null | 1084.53 → 1092.36 | +0.72% | 930.00 → 936.73 | 100.98 → 105.40 |
| A | 1205.23 → 1218.90 | +1.13% | 1035.76 → 1047.82 | 105.56 → 113.42 |
| B | 1209.06 → 1222.78 | +1.13% | 1034.59 → 1029.09 | 112.69 → 121.77 |
| C | 1116.05 → 1077.35 | -3.47% | 951.24 → 920.28 | 105.08 → 99.56 |
| all | 1073.50 → 1092.77 | +1.79% | 922.68 → 939.10 | 99.85 → 99.81 |

The null quartet ranged 1083.70–1096.61 ms, a 12.90 ms spread. Its 0.72% arm gap is
an observed null result, **not a confidence interval or a noise bound for other
jobs**. A/B/all show no positive wall signal in this quartet. C is only a weak
positive hint: its unchanged collection phase also fell 3.25%, and its reference
endpoints drifted from 1197.11 to 1035.00 ms (13.54%). ABBA reduces simple order bias
but does not remove changing foreground contention or nonlinear drift. No robust
winner or production speedup is established. No more tuning or optimization loops
were run, and nothing was enabled or deployed on main.

GPU guard durations were 6.47–7.54 seconds, below the unchanged 30-second total cap;
largest GPU probe cgroup peak was 1,454,362,624 bytes. All nine jobs in this final
methodology wave passed their guards, removed their cgroups and recorded no
resource or main-health violation. No CPU performance retimings, full-game runs,
full-suite reruns or statistical benchmark were performed.

## Configuration and scope

`train-annealed` accepts independent, combinable options:

| Option | Default | Implemented work |
| --- | --- | --- |
| `--actor-overlap continue-v1` | `off` | Advance resolved Continue worlds while the unchanged full actor batch finishes decoding |
| `--learner-prefetch` | absent | Materialize the next **effective minibatch** on one CPU worker |
| `--host-math-workers N` | `1`, serial | At most 32 persistent-per-minibatch coordinate workers for gradient validation, scaling and accumulation |

Nondefaults are appended to `CheckpointRun.command_line`; strict resume cannot
silently add/remove/change a mode. `TrainingExecutionOptions` is outside
`PpoConfig` and the checkpoint codec. The annealed initializer reapplies it after
both fresh construction and restoration. Schemas and tensor dimensions are
unchanged. No next-update gameplay or policy lag is introduced.

## A: early Continue

Only after every base value/kind row validates does the private sampler dispatch
Continue choices. Targets and statistics use the existing code and raw logits;
kind sampling still consumes sixteen draws. Every subsequent conditioned stage
still evaluates every original row and prefix, including an early terminal world.
No full ActionSpace copy or second Continue-frame clone is needed.

One bounded request/reply slot belongs to each world, at most forty. A validated
Continue job preserves local decision history and the existing no-order transport
semantics, then uses the shared ordinary advance/reward/retention code. Teacher
and Weak are supported internally; the annealed CLI admits Teacher only for A,
rejecting a weights opponent before execution.

Workers return CPU state and prepared next frames. They never execute Candle or
flush inference. Once all actor choices validate and staged RNGs commit, the
remaining worlds are dispatched, replies applied in original stream order, and
retained values evaluated on the GPU owner with the original batch-one shape.
On late sampler failure pending replies drain, scoped workers join, and the whole
uncommitted update fails. Advanced private worlds are not reusable after that
failure; no checkpoint is written for the failed update.

## B: effective-minibatch prefetch, not 64-row host staging

The full host-packing/target/prefix split is deliberately deferred. This prototype
overlaps **ragged expansion and target unpacking for the next effective minibatch**
with current forward/backward/Adam/candidate-KL work. It does not claim to overlap
all feature conditioning or uploads, and does not claim the research's 9 MiB
two-microbatch payload bound.

One scoped CPU worker persists through all epochs of the update. The consumer
submits only one next job after receiving the current buffer, so two materialized
buffers cover current, filling and ready ownership. Each holds at most
`config.minibatch` rows (normally 2,048, maximum 8,192), not 64. The complete current
minibatch is materialized before its first model operation. Errors are observed
only when that logical minibatch is consumed; future errors are discarded on KL
stop. Both channels close before joining on cancellation/error.

The compiled `size_of::<PpoPreparedSample>()` was **67,488 bytes**. In this probe,
two full-capacity 128-row buffers allow 17,276,928 bytes (the actual 128+33 row
payload totals 10,865,568 bytes). Two 2,048-row buffers allow 276,430,848 bytes;
two maximum 8,192-row buffers allow 1,105,723,392 bytes. These are dense sample
payloads, excluding allocator overhead and index/header storage.

The conservative additional allocation is
`MODEL_MAX_BATCH * size_of::<PpoPreparedSample>()`, two index buffers and a 2 MiB
worker stack. `PPO_PREFETCH_STORAGE_PEAK_BYTES` adds this to the unchanged default
annealed storage ledger: **5,974,129,696 bytes**, with a compile-time bound below
8 GiB. This ledger excludes
native worlds, allocator overhead and Candle/driver allocations; it is not a claim
that whole-process memory was measured or that resource monitoring can be removed.
The qualified background runner's memory-high 10 GiB / hard 12 GiB limits were
unchanged. Continuation probe cgroups peaked at 1,905,262,592 bytes including a CPU
test rebuild; CUDA probe maximum was 1,260,871,680 bytes. All continuation jobs
reported clean memory/pids events and removed their owned cgroups.

## C: gradient folding only; Adam remains serial

Requested workers resolve locally to the minimum of the request, available CPUs
and the 32,768-coordinate minimum partition budget; one selects the old serial
path. Tests can exercise 8/16/32 partitions on small synthetic vectors without
extrapolating hardware scaling. Workers persist across an effective minibatch,
using 256 KiB stacks and bounded channels, never spawning per microbatch.

Downloaded gradients are copied into reusable disjoint worker buffers, then the
contiguous result is dropped before the next GPU readback. Two result sets plus
the accumulator hold at most 20,400,240 bytes. This extra CPU copy is a cost to
measure, not a free speedup. Combined B ledger, folding buffers and at most 32
worker stacks have a compile-time bound below 8 GiB, with the same exclusions.

Each coordinate retains validation, f32 multiplication, stored f32 addition and
final division order. Errors are selected by original validation/scale/add phase,
then global coordinate, not worker arrival. Previous fold/report errors outrank
later microbatch errors. All folds drain before Adam. Global F64 norm, Adam
corrections, scalar Adam, parameter import and both rollback layers are unchanged.
Parallel Adam coordinates are **deferred**, not implemented.

## Focused commands and completed probe entrypoints

The nonignored tests below were included in the completed library suite. CUDA
parity and all per-mode probes were executed individually after explicit owner
reauthorization. Any future run still requires the approved background runner;
do not run these commands directly beside training.

```text
cargo test --lib --all-features --quiet continue_overlap_tests
cargo test --lib --all-features --quiet host_folding::tests
cargo test --lib --all-features --quiet model::transfer_tests::concurrency_tests
cargo test --lib --all-features --quiet ppo_arena::annealed::tests::concurrency_tests -- --skip concurrency_probe
```

Explicit ignored CUDA parity tests:

```text
ppo_arena::episode::continue_overlap_tests::cuda_continue_overlap_preserves_mixed_actor_rng_orders_rewards_and_retained_rows
model::transfer_tests::concurrency_tests::cuda_folded_ppo_preserves_parameters_moments_reports_and_rejection
```

Use `cargo test --lib --all-features --quiet NAME -- --ignored --exact --nocapture`.
The CPU/CUDA short probe entrypoints are:

```text
ppo_arena::annealed::tests::concurrency_tests::concurrency_probe_cpu
ppo_arena::annealed::tests::concurrency_tests::concurrency_probe_cuda
```

For each individual probe set `DRYSUA_PROBE_MODE=base`, `a`, `b`, `c`, or `all`.
Set `DRYSUA_PROBE_BALANCED=1` for the final warmup+ABBA method above; otherwise the
original single-pair behavior remains available.
`DRYSUA_PROBE_WORKERS` defaults to 2; accepts 1..32. Optional
`DRYSUA_PROBE_WEIGHTS` reads a compatible accepted runtime directory without
writing it. Each probe runs a reference and candidate, forty worlds for forty
decision rounds, one epoch, effective minibatch 128. This gives multiple
64-row microbatches and a subsequent effective minibatch without full games.
It prints elapsed time, work counters, Continue counts, worker configuration and
state hashes, and compares parameter/moment bits, checkpoint progress and report
bits/counters. All
checkpoints go to newly created probe-owned temporary directories, removed after
success. There are no time assertions. GPU jobs in the stronger sweep completed
in 3.212–4.340 guard seconds, below the 30-second total budget; CPU probe bodies
completed in 5.57–8.17 seconds (the first CPU job also rebuilt the test binary).
