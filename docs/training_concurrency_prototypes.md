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
