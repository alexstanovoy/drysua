# Frozen neural-opponent batching

`train-annealed --opponent weights --opponent-weights DIRECTORY` now defaults to
`--opponent-inference batched`. The library's
`TrainingExecutionOptions::neural_opponent_batching` still defaults to `false`.
Use `--opponent-inference scalar --actor-pipeline-groups 1` for the legacy
opponent path. Other settings must also match when resuming an existing run.

Teacher settings remain unchanged: both inference options are benignly ignored
by the CLI, the library flag stays false, and no inference scope marker is added.
There is no annealed Weak-opponent CLI variant; non-neural collectors retain
their existing default behavior. Explicitly enabling the library flag with a
non-weights annealed opponent is rejected rather than silently ignored.

Only batched weights runs append ` --opponent-inference batched` to execution
scope. Scalar mode emits no marker and preserves the old scope bytes. Switching
between scalar and batched inference is a strict resume mismatch in either
direction; this is not an automatic checkpoint migration.

G1 supports either weights mode. G2/G4 support Teacher or batched weights, not
scalar weights. Admission precedes model/world loading, retains the existing
`B * G <= 64` active-world limit, and requires whole waves per update. Neural
admission separately accounts for prepared opponent frames/choices within the
unchanged 12 GiB bound, in addition to the pipeline memory check.
The new fixed payload charge is bounded below 64 MiB for 64 worlds; dynamic
action-space backing allocations remain in the existing non-rollout reserve.
This is not a whole-process RSS proof, and does not relax the runtime guard.

In batched mode both policies and learner bootstrap fallbacks execute on the
collection owner. Workers only prepare observations, apply the prepared decisions
in seat order, and advance the original interval. There is one opponent
`sample_batch` call per nonempty group round instead of one scalar call per world.
The current Teacher-only graph probe does not admit this mode.

Opponent inference remains stochastic `sample_batch` sampling with per-world
RNG ownership and sampling cadence, never greedy `choose_batch`. Moving from
scalar B1 inference to active-row batches can change logits and actions through
floating-point grouping. Different action branches can also change RNG draw
counts. Scalar-versus-batched bit identity is therefore not promised; exact
checkpoint/resume replay is required within the same mode and scope.

The frozen opponent is not optimized. Collection, including a late interrupted
wave, must not commit learner model/Adam/RNG state until the update succeeds.
Regression coverage lives in `src/tests/neural_opponent_scope.rs`; adding these
tests does not authorize executing them or touching a live training process.
