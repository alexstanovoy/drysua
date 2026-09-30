# Training performance

Actor sampling and greedy warmup batch on the learner device; collection and PPO
remain sequential at update boundaries. Rollouts bind model identity, derive
bounded per-environment RNG streams, and discard worlds between updates. Standard
capacity is 32,768 transitions; expanded M40/B40 uses the distinct
[annealed budget](annealed-sequential-budget.md), not a relaxed standard limit.

## Current decisions

- The `train-annealed` CLI defaults to M40/B20/G2, microbatch256 and actor-value
  reuse. Parallel20 is independent of CPU count; CUDA still requires explicit
  `--device cuda`. This is distinct from the `train-full` group experiments below.
  [Profile and legacy-resume settings](training_microbatch.md) remain scope-bound.
- Keep one collection group for `train-full` and library defaults. Historical E16 groups=2/4 achieved only
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
