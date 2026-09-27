# Training performance

Actor sampling and greedy warmup batch on the learner device; collection and PPO
remain sequential at update boundaries. Rollouts bind model identity, derive
bounded per-environment RNG streams, and discard worlds between updates. Standard
capacity is 32,768 transitions; expanded M40/B40 uses the distinct
[annealed budget](annealed-sequential-budget.md), not a relaxed standard limit.

## Current decisions

- Keep one collection group on CUDA. Historical E16 groups=2/4 achieved only
  0.84x/0.58x default end-to-end throughput despite favorable CPU-window results.
- `train-full --pipeline-groups 1|2|4` is supported for complete episodes only,
  with at least one environment pair per group. Groups own disjoint whole pairs;
  ordered merge and the update boundary remain global. Smaller GEMMs change
  trajectories, so only within-mode repeatability is byte-identical. Nondefault
  groups enter scope; strict resume and provenance migration reject mode changes.
- Packed accelerator gradient readback and reuse of the pre-candidate parameter
  snapshot preserve arithmetic/rollback. The historical equal-work U93 ABBA
  comparison (17,897 samples, 36 Adam steps) was 69.073 vs 68.704 s mean: a 0.53%
  difference below the 5% admission threshold, not a useful deployed speedup.
- [C4 folding](training_concurrency_prototypes.md) is opt-in; A/B overlap is retired.
  Timing records separate collection, preparation, optimization and checkpointing;
  collection includes world setup, and PPO timing is not GPU-kernel time alone.

## Bounded collection benchmark

[benches/training.rs](../benches/training.rs) measures the production collector,
not optimizer/checkpoint/full-episode wall time. Fixed seed 10141700, update 0,
320 untimed warmup rounds, 64 measured rounds starting at tick 961. Construction
is outside the timed section. Cases cover serial E1, parallel E2/4/6/8/16/26 and
groups 2/4 at E8/16/26. Bench profile inherits release optimization, not test O2.
All heavy payloads require [execution safety](experiment-safety.md); commands below
are payload examples, not permission to launch an unbounded sweep:

```sh
cargo bench --features builtin --bench training -- --test
cargo bench --features builtin --bench training -- \
  --exact collection_parallel/environments/26 \
  --sample-size 10 --warm-up-time 1 --measurement-time 5
DRYSUA_BENCH_DEVICE=cuda cargo bench --features builtin,cuda --bench training -- \
  --exact collection_parallel/environments/8 \
  --sample-size 10 --warm-up-time 1 --measurement-time 5
```

`DRYSUA_BENCH_PHASES_ONLY=1` prints untimed phase windows; `DRYSUA_BENCH_PHASES=1`
adds them to a sweep. Timed cases have no phase instrumentation. Smoke is not a
statistical benchmark and remains relatively slow. Preserve identical weights,
seeds, episodes, samples, epochs and applied steps when comparing throughput.
E2/E8/E16 CUDA identity goldens are embedded in `src/tests/train_full_identity.rs`
and passed qualification; missing TEMP JSON no longer silently skips them.
Old exploratory tables and transfer diaries remain at
`git show 2ddb68b:docs/training_performance.md`; removed TEMP runs are not dependencies.
