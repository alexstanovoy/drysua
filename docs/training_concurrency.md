# CPU/GPU overlap candidates

Research baseline at `ccd69e1`, 2026-09-21. The original investigation used only
source inspection and bounded reads of existing telemetry/logs. Opt-in prototypes
now exist; their authorized background verification compiled successfully and
completed the CPU library suite (1,402 passed, 0 failed, 19 ignored). The aggregate
all-target command hit its wall budget, was stopped, and after owner review the
last smoke case passed separately. Two targeted CUDA tests and all five short
CPU/CUDA modes then passed exact state/RNG/report checks using pinned U376 weights.
The unchanged CUDA control improved 16.76% in the original single pair. A final
test-only warmup+ABBA pass reduced the null arm gap to 0.72%; A/B/all were 1.1–1.8%
slower in their quartets, while C was 3.47% faster but also had substantial drift
in its unchanged collection phase and reference endpoints. All balanced trials
passed exact state/RNG/report checks. These remain short contended signals, not
a clean A/B or an established production speedup; no further tuning followed.
**No production or 16-core/32-thread performance qualification is claimed.** The
active frozen B40 campaign was not stopped or reconfigured. See
[prototype status and evidence](training_concurrency_prototypes.md) for the exact
implemented scope (effective-minibatch prefetch; serial Adam), fixes and limits.

## Evidence and objective

During inspection, the 15-minute phase means were 51.94 seconds for collection,
22.80 seconds for PPO and 6.56 milliseconds for final rollout preparation. The
collection timer includes world construction, preparation, inference and stepping;
optimization includes host minibatch preparation as well as GPU/Adam work. These
numbers locate expensive phases, not their internal critical paths.

Accepted updates 294–296 contain 120 games and 474,174 actor decisions. Of those,
410,832 (86.64%) were Continue, from the exact `episode:` audit records. Continue
keeps existing orders running; advancing those worlds still performs Teacher,
unit, reward and observation work. It is not an empty simulation step.

Optimize equal-work update wall time, not utilization alone. Independent work can
replace a CPU-then-GPU sum with overlap approaching the slower resource's time,
but dependencies, pipeline fill/drain and the final surviving game remain. Never
add dummy games, stale-policy samples or redundant GPU work merely to reach 100%.

## First candidate: advance resolved Continue worlds during decoding

Current `model::selection_batch_locked` runs four decoder stages. The collector
waits for the entire `sample_batch` result before dispatching any world advances
in `episode::collect_with_workers`.

Continue is resolved after `initialize_sampling_rows`:

- Its value and action kind are known and validated.
- `BehavioralTarget` activates only the kind head. Reuse its existing construction
  and `SampledPathLogits::statistics`, including inactive-head arithmetic; use raw
  logits for statistics, never Gumbel-perturbed logits.
- All 16 kind draws, including masked entries, remain consumed. Subsequent unit
  and slot selection skip Continue, and final decode uses the cached kind without
  drawing again. There are no additional RNG decisions to speculate about.

Proposed private collector interface:

1. Complete base sampling for every current row before any early dispatch.
2. Finalize Continue choices through the existing validation/statistics path.
3. Submit those worlds' ordinary CPU advance/prepare jobs immediately.
4. Execute every original conditioned GPU stage for every original row, including
   Continue rows, with unchanged prefixes and tensor shapes.
5. On full sampling success, commit staged RNGs and dispatch only remaining worlds.
6. Receive and apply all results in the original stream order. Construct the next
   active batch only at the original boundary.

```text
GPU: full-batch trunk/base | full-batch remaining decoder stages | next boundary
CPU:                      | Continue advance and prepare        |
CPU:                                                           | other advances
```

Bound one outstanding advance per world and at most 40 pending results. An early
terminal world must remain in this round's GPU tensors; remove it only next round.
Do not clone/reconstruct decision inputs gratuitously: `ActionSpace` is not Clone,
and sampling still borrows it. Use shared immutable prepared inputs or a validated
Continue-specific job while preserving local decision/order bookkeeping.

Start with the current scripted Teacher opponent. A neural opponent's request
performs another model call and is not CPU-only. Defer retained-value flush GPU
calls until the actor forward has completed, retaining their existing batch-1
shape and terminal zero bootstrap. Workers may prepare/queue those requests, but
the first version must not introduce uncontrolled extra GPU submitters.

### Failure contract and proof

Later decoder/backend failure can occur after private worlds have advanced. Do
not pretend this is the old all-choices-before-any-advance contract. The proposed
experimental mode must abort the whole uncommitted update, drain/join work and
discard its worlds, pending samples and RNG state without checkpointing. It must
never continue with those mutated worlds. Preserve ordered error reporting where
possible; encode the new execution contract explicitly before enabling it.

Successful-run bit identity is a target, not yet evidence: compare mixed-action
B40 choices, statistics, RNG, decoded orders, rewards, retained sample order,
outcomes and final policy/Adam state. Inject a late decoder error after early
dispatch and verify the previous checkpoint is untouched. The 86.64% eligible
decision fraction is not a predicted speedup: only remaining decoder time is
hideable, and late non-Continue worlds may still determine the barrier.

## Second candidate: prepare the next learner chunk while GPU computes

Keep one CPU preparation worker across an update's epochs and gradient/KL passes,
with exactly two buffer permits of at most 64 rows. It can expand ragged samples,
unpack targets, condition/pack features and prepare prefix indices, masks, labels,
old probabilities, advantages and returns from immutable rollout data.

The existing GPU-owner thread retains every Candle operation, parameter import,
embedding lookup, forward/backward and candidate KL calculation. Keep shuffle/RNG,
exact microbatch partitions and effective minibatch boundaries ordered. Do not
cache embeddings or activations across Adam steps or shuffle the next epoch early.
Preserve whole-minibatch input validation before its first GPU work; surface
prefetch errors at their original logical stage. On KL stop/error, discard future
buffers and close channels before joining; retain existing rollback behavior.

Source-derived packed payload is 17,453 F32 feature values plus 738 bytes of target/
statistics arrays and 28 bytes of prefix arrays per row. Two 64-row payloads total
9,033,984 bytes (8.62 MiB), plus bounded metadata. Account separately for expansion
scratch `64 * size_of::<PpoPreparedSample>()` and any retained intermediate copies.
Two CPU buffers alone do not make pageable CUDA uploads asynchronous.

## Third candidate: overlap host gradient folding with the next backward

Within an effective minibatch, parameters stay fixed. CPU workers can validate,
scale and accumulate downloaded gradient n while the GPU processes n+1. At most
two gradient results plus one accumulator occupy 20,400,240 bytes, excluding
existing temporaries/snapshots. Drain all results before Adam.

Partition independent parameter coordinates, preserving each coordinate's f32
multiply/add/divide order. Keep the global norm's serial f64 reduction and Adam
correction calculation unchanged; publish only a complete replacement. Resolve
errors by original phase/index, not worker arrival. Use persistent bounded workers,
not a new thread per microbatch, and retain a serial fallback. Cache/memory bandwidth
may saturate long before 32 workers; more threads are not inherently faster.

## Why not just split into groups or play the next update now?

Existing `collect_groups_bounded` already overlaps one group's CPU work with
another group's inference. A queue/broker around the same calls is not a new
optimization. Earlier CUDA grouping regressed throughput (E16 G2: 0.84x). A staged
20+20 wavefront with packed per-stage readbacks is a possible later experiment, but
smaller GEMMs can change logits, trajectories, retained counts and optimizer work.
It requires a separate scoped mode, not a silent scheduling change.

During PPO, the next update's learner decisions require the updated policy, even
at pregame tick 1. Playing those games with old weights introduces policy lag.
`pipeline.rs` permits a bounded one-generation lag for its Standard profile, but
annealed B40 does not use that contract. This alternative is a versioned algorithm
choice whose learning quality needs evaluation, not a transparent optimization.
Only policy-independent native world setup can be prefetched safely; keep tracker
lineage allocation, generation-file publication, counters and actor RNG ordered.

## Future 16-core / 32-thread host and measurement gate

World count and worker count should be separate. Keep B40 inference membership and
40 RNG streams fixed while evaluating a bounded local world pool at different
worker counts. One world may have only one job in flight; completion order must not
control inference batch formation or rollout insertion. SMT32 is not 32 physical
cores. No global thread caps, affinity, CPU quotas or host tuning are required.

First implement/prove Continue-only early dispatch and the two-buffer learner
separately; benchmark each in an authorized exclusive window on checkpoint copies.
Measure total/collection/optimization wall time, useful CPU/GPU overlap, active
worlds, decisions/ticks, samples and applied optimizer steps. No current run should
be stopped or changed merely to prepare this work. Investigate the CPU-fold pool
only if learner substage measurements justify it.
