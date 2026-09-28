# Fixed two-shape actor graph lab

This is `cfg(test)` only. Production has no graph flag, workspace, owner binding,
or graph cache guard. No dependency, CUDA stream, or device-sharing factory is added.
Run every ignored graph test in its own fresh, exclusively owned bounded process.

## Invariants

- One irreversible process admission is shared by the original encoder probe and
  the actor scope, including the eager control. Admission precedes warmup and is
  never returned after success or failure. Both actor slots use the admitted model's
  existing DeviceId; this does not require a process-global device singleton.
  The admission owner retains one clone of that existing device for exact identity
  checks. Per-slot atomic budgets reject a third guarded warmup or a second capture.
- The model must be created and previously used on the current owner thread.
  Admission publishes ownership under its exclusive parameter lock; model locks
  check ownership before and after acquisition to reject already-waiting callers.
- One owner workspace has exactly two physical slots, `[main_batch, 1]`, where
  `main_batch` is 20 or 40. It owns fixed input/output Vars, input views, and the
  parameter storage references. No lazy, adaptive, or all-batch-size graph cache
  is permitted. There is no recapture.
- Both modes perform four encoder warmups: two for each fixed shape. Eager-control
  warmups run outside cache guards and capture nothing. Graph mode captures exactly
  two graphs, even if the one-row shape is never subsequently requested. Other
  actor batch sizes retain their original eager GEMMs. Public actor limit64,
  decoder dispatch, masks, log probabilities and staged RNG commit stay unchanged.
- Dynamic staging/uploads, decoder heads, parameter imports and PPO are outside
  cache guards. Only `sample_batch` can dispatch through the graph; public training
  and all PPO forwards remain autograd-preserving eager paths.
- Replay and all decoder consumers synchronize while the sample's parameter read
  lock is held, before committing RNGs. Tensor aliases never escape in choices.
- Scope completion synchronizes, destroys captured graphs and checks context errors,
  drops fixed buffers, and checks completion again **before PPO**. Backend/capture
  failures terminate the owned test process; there is no retry or eager recovery.

Each fixed shape has at most 137 metadata entries of at most 12 `usize`s, with
eight shared F32 scalar entries across the two shapes. On a 64-bit target the bound
is `137 * 2 * 12 * 8 + 8 * 4 = 26,336` logical device bytes. This is **not** total
VRAM/RSS: driver allocations, contexts, modules, events, table headers, fixed tensor
buffers and graph storage remain subject to the existing runtime limits. Admission
and the two shapes are fixed for the lifetime of the process.

`actor-graph-slot` reports each slot's fixed input/output payload. With the current
encoder layout this is 132,276 bytes per row: 5,423,316 bytes for `[40, 1]`, or
2,777,796 bytes for `[20, 1]`, summed across the two slots. These numbers exclude
staging, transient activations, graph allocation nodes and backend workspaces;
they are not total peak VRAM estimates. Parameter storage is shared, not copied.

The original one-batch encoder probe remains a one-capture experiment. It shares
the same process admission and must never run in the actor gate's process.

## Earlier result and the new hypothesis

The user reported that the previous single-shape B40 full-update gate passed exact
checkpoint bytes, pre-PPO traces, actor RNGs and reports: eager **54.441 s**, graph
**51.115 s**, approximately **1.065x**. It replayed B40 for **1,297 of 6,107** actor
calls, with **4,810** eager fallbacks. These are supplied earlier measurements,
not new results from the two-shape implementation.

Capturing the singleton tail, and separately testing B20/G2, may increase useful
coverage. That is a hypothesis, not evidence of additional speedup. Compare the
histograms and full-update timings from new matched runs; the extra setup cost is
included and may outweigh replay savings. B40/G1 and B20/G2 are different normal
execution scopes, so neither cross-layout bit equality nor equal call histories
is promised.

## Verification (new changes not executed by the source owner)

Run through the integrator's authorized exclusive runner, not alongside training.

First run the sampler parity/bounds test alone:

```text
model::cuda_graph_probe::actor_tests::cuda_actor_graph_preserves_samples_rng_tails_weights_and_owner_bounds
--exact --ignored --nocapture
```

It covers changing frames and in-place weights, sampled actions/targets/frames,
value/logprob/entropy bits, exact RNGs, a 39-row eager tail, actual graph replay
counts, invalid inputs, wrong model/device/thread, prior foreign-thread use,
checked retirement and rejection of a second admission. The original encoder
probe still has its original filter and must also run in a separate process.

Run the B20/singleton sampler regression separately as well:

```text
model::cuda_graph_probe::actor_tests::cuda_actor_graph_twenty_and_singleton_preserve_samples_rng_and_tails
--exact --ignored --nocapture
```

Both sampler cases alternate their main and singleton slots, verify an uncaptured
tail eagerly, check both slots after changing weights, and reject rewarming or
recapturing either slot. The B20 case's eager tail has 19 rows.

Use the owner's prepared test executable with CUDA and builtin features. Every
ignored graph filter gets a separate fresh process. Next run the full-update gate
**twice in separate processes for each layout**, changing only
`DRYSUA_PROBE_ACTOR_GRAPH` from `0` to `1` within a pair:

```text
ppo_arena::annealed::tests::graph_full_update::actor_graph_full_update_gate_cuda
--exact --ignored --nocapture

DRYSUA_PROBE_ACTOR_GRAPH=0                 # Required; use 1 in the graph process.
DRYSUA_PROBE_WEIGHTS=<fixed CREDITu10 directory>
DRYSUA_PROBE_MODE=base                    # Enables existing G1 actor-count log lines.
DRYSUA_PROBE_ACTOR_BATCH=40
DRYSUA_PROBE_ACTOR_GROUPS=1
```

For B20/G2 use exactly:

```text
DRYSUA_PROBE_ACTOR_BATCH=20
DRYSUA_PROBE_ACTOR_GROUPS=2
```

The batch parser accepts only `20` or `40`, defaulting to `40`; the groups parser
accepts only `1` or `2`, defaulting to `1`. The gate admits only `(40, 1)` and
`(20, 2)`, validating before directory/model setup. `(20, 1)` would require two
sequential collection scopes and is unsupported; `(40, 2)` exceeds the 40-world
workload. Setting only one of the two controls for B20/G2 is rejected.

Both layouts retain M40, Teacher, reuse=1, microbatch256, epochs4, effective
minibatch2048, seed9001 and the full production episode ceiling. There is no shortened
rounds control. B20/G2 runs one two-group collection scope with 40 total streams;
the retained actor trace remains 40-wide, not 20-wide. Teacher plus reuse is
mandatory: the ordinary reuse=0 collector has a CUDA flush-evaluation thread and
is intentionally not admitted. Do not use the old multi-trial `concurrency_probe_cuda`.

Graph/eager selection never changes run scope: it is an exact optimization. Batch
and group choices use their existing canonical `--parallel` and
`--actor-pipeline-groups` scope fields. Default B40/G1 retains its previous scope.

The gate records `graph_stats`: `main_batch`, the two `shapes`, 65-entry
`calls_by_batch` and `hits_by_batch` arrays, `captures`, `setup_ns` and `retirement_ns`.
These are the single retired-scope statistics published after checked teardown
before PPO, not a growing collection of workspaces. Total calls are bounded by
`2 * MAP2_ACTOR_DECISIONS`; index zero must be zero, hits cannot exceed calls and
can occur only at the main shape or singleton. Eager mode requires zero captures
and hits. Graph mode requires two captures and positive main-shape hits; singleton
hits may be zero. No timing threshold is an assertion.

The total elapsed time includes initialization, all four warmups, any captures,
collection, checked retirement, PPO and checkpoint completion. Compare the
`graph-actor-trace` lines from both logs; G1 also emits `concurrency-actor` counts. The gate rejects
anything other than 40 games, 40 applied Adam steps and four complete passes
without KL rejection. Fewer optimized samples or steps cannot qualify as speedup.

Each run retains a uniquely owned artifact directory printed in its JSON result,
including a SHA-256 of the complete checkpoint tensor file (not the old probe's
restored-model hash). CUDA-scoped artifacts must not be restored into a CPU model.
Then compare those artifacts read-only, on CPU:

```text
ppo_arena::annealed::tests::graph_full_update::actor_graph_full_update_artifacts_match_exactly
--exact --ignored --nocapture

DRYSUA_GRAPH_EAGER_DIRECTORY=<eager directory for the chosen layout>
DRYSUA_GRAPH_CANDIDATE_DIRECTORY=<graph directory for the same layout>
```

The read-only comparison checks pre-PPO per-stream action/request trace hashes,
exact final actor RNGs and decision/retention counts, reports, scope, generation
history, artifact digests and validated complete checkpoint bytes containing
model/Adam/RNG state, without creating a CUDA device or attempting device migration.
It additionally requires identical `main_batch`, `shapes` and `calls_by_batch`
between eager and graph within the same layout. It deliberately does **not** require
equal `hits_by_batch`, `captures`, `setup_ns` or `retirement_ns` between modes.
Old records without the new histograms cannot qualify this two-shape comparison.

No new two-shape speedup or production qualification is claimed until the new
gates and the repository's full checks pass under the authorized runner.
