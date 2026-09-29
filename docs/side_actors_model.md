# Feature-gated side actors

`side-actors` selects model25 at compile time. It is off by default. The normal
build retains the model24 descriptor/hash, 1,700,020 parameters and 62 named
parameter tensors. The side build has 1,812,983 parameters and 86 tensors.

## Shared state and routing

There is one `PolicyModel`, encoder, trunk, value head and set of four conditional
embeddings. Only the twelve actor linears are duplicated. The original names and
all first 62 parameter positions remain unchanged as shared/Radiant state; 24
new `dire.*` weight/bias tensors follow them. Their Vars are independently allocated.
Fresh construction preserves the complete legacy initialization draw prefix.

Routing reads only the already-observed `global_feature::SIDE_RADIANT` and
`SIDE_DIRE`. Exactly `(1,0)` or `(0,1)` is accepted; either sign of zero represents
zero, but a subnormal/fractional value does not. Missing/ambiguous side data fails
before tensor evaluation and sampling RNG consumption. `FeatureFrame::new()` is
not changed. Existing team-canonical geometry is not changed.

Both actor branches use the original full batch shape. A U8 `where_cond` selects
each row; pointer queries are selected before the existing pointer multiply/sum/
scale. No row compaction, padding, per-side optimizer or side-specific loss scaling
is introduced. The encoder's three-field output is unchanged, and routing stays
outside encoder graph capture.

Entirely unused inference head families remain skipped. Every exercised family
checks both raw branches, including unselected rows. A test-only forced-eager
forward therefore rejects an overflowing family it deliberately computes, while
normal inference can still skip that family.

Training retains its 13 public selected outputs and privately retains the 24 raw
actor outputs for validation. One bounded CUDA readback checks shared value, all
24 raw actor tensors, and both selected pointer-score tensors before loss/backward.
The maximum logical payload is `256 * 823 * 4 = 842,752` bytes; packing/backend
temporary allocations are additional. Validation aliases are detached; the actual
learning graph is not. CPU validation avoids a packing allocation.

PPO's value-input detach remains unchanged. Actor gradients reach the common trunk
and conditional embeddings. Opposite actor-head derivatives are zero for rows
belonging to the other side. This does not promise frozen parameters when Adam has
nonzero historical moments, or model24/model25 training-trajectory bit equality.

## Initializer boundary

The model module exposes read-only `LEGACY_MODEL_SCHEMA_VERSION`,
`LEGACY_MODEL_SCHEMA_DESCRIPTOR`, `LEGACY_MODEL_SCHEMA_HASH`, and
`LEGACY_MODEL_PARAMETER_COUNT`, plus the feature-gated crate function:

```rust
expand_m24_side_actor_parameters(source: &[f32]) -> Result<Vec<f32>, ModelError>
```

It checks source length/finiteness, preserves all 1,700,020 source float bits,
appends `[1586113..1590225]`, then `[1591169..1700020]`. It does not authenticate
files. The checkpoint coordinator must first verify the pinned source's full SHA,
metadata, dtype and shape, and must create fresh optimizer/RNG/progress state.
No existing checkpoint is a model25 resume merely because it has compatible
shared tensor shapes. Neither old independent expert's backbone is merged here.

## Verification handoff

Source-owner tests and builds have not been executed. Use the authorized runner:

```text
CPU feature tests: model::side_actor_tests
CUDA contract: model::side_actor_tests::cuda_side_actors_preserve_routing_gradients_and_ppo_rollback
CUDA test arguments: --exact --ignored --nocapture
```

The tests cover exact expansion/layout, malformed sides, mixed/permuted native
sampling through one encoder pass, independent head gradients, shared embeddings,
full-batch masked pointer references, unused-family skipping, unselected overflow,
global evaluation row indices, scalar value/RNG rejection and coupled PPO rollback.
Run the normal model24 suite separately without `side-actors`; `--all-features`
now selects model25 and does not replace that compatibility check.
