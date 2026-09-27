# Host folding: supported scope and evidence

`train-annealed --host-math-workers N` accepts 1..32; **1 is the serial default**.
Workers resolve against local CPUs and the minimum coordinate partition budget.
Bounded workers preserve per-coordinate arithmetic and error order; all folds
drain before serial norm/Adam. No global thread limit, affinity or CPU quota is set.
Nondefault worker counts enter checkpoint scope. Strict resume rejects a changed
scope. Retired `--actor-overlap`/`--learner-prefetch` flags are rejected; their old
artifacts remain readable as weights, but continuing those runs needs their original code.

## Decision evidence

Idle-host comparisons on 2026-09-22 found A 3.28% slower and B unchanged (+0.01%).
C4 remains a modest candidate, not automatic deployment or proof of larger-host scaling.

| Full release workload | Serial seconds | C4 seconds |
|---|---:|---:|
| Historical pre-refactor pair | 68.010335326 | 66.528735626 |
| Qualified refactor pair | 68.621567864 | 66.778648342 |

Both use U376, seed 9001, forty full games/worlds, four epochs, minibatch 2048,
microbatch 64: 19,411 samples, 40 Adam steps, 465,818 ticks and state hash
`eb52a187d338a4e8`. Current serial/C4 parameter, moment, RNG and report bits match.
These are historical versus current pairs, **not contemporaneous A/B** or a
statistically established speedup. Earlier short contended comparisons were inconclusive.
Preserved CSVs are under `artifacts/temp/concurrency-{idle,probes}-2026092*/`;
final qualification is in `artifacts/deslop-20260922/REPORT.md` and `verification/`.
The retired designs and original job ledger remain in Git at `2ddb68b`.

## Probe inputs

Run only through the [authorized bounded runner](experiment-safety.md), never beside
a learner without its explicit resource authorization. CPU/CUDA probe names are
`ppo_arena::annealed::tests::concurrency_tests::concurrency_probe_{cpu,cuda}`.
Use the appropriate feature tuple, including `builtin,cuda` for CUDA, and payload:

```text
cargo test --release --lib --no-default-features --features builtin,cuda --quiet \
  ppo_arena::annealed::tests::concurrency_tests::concurrency_probe_cuda \
  -- --ignored --exact --nocapture
```

`DRYSUA_PROBE_MODE=base|c` defaults to `c`; `DRYSUA_PROBE_WORKERS` defaults to 4
in this probe, not the production CLI. `DRYSUA_PROBE_WEIGHTS` is a read-only runtime directory.
ROUNDS/EPOCHS/MINIBATCH variables with the same prefix default to 40/1/128;
the full comparison sets 9300/4/2048. `DRYSUA_PROBE_BALANCED=1` selects an untimed
warmup plus ABBA; absent/0 selects a pair. Each trial uses fresh owned output and
new optimizer/RNG state. Keep work counters and identity checks with every timing.
