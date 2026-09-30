# Side actors (model25)

Model25 is the only policy model: 1,812,983 parameters in 86 named tensors.

## Shared state and routing

There is one `PolicyModel`, encoder, trunk, value head and set of four conditional
embeddings. Only the twelve actor linears are duplicated. The original names and
first 62 parameter positions are shared/Radiant state; 24 `dire.*` weight/bias
tensors follow them. Their Vars are independently allocated. Fresh construction
draws the Dire heads after the Radiant parameter order.

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
nonzero historical moments.

## Tests

```text
CPU tests: model::side_actor_tests
CUDA contract: model::side_actor_tests::cuda_side_actors_preserve_routing_gradients_and_ppo_rollback
CUDA test arguments: --exact --ignored --nocapture
```

The tests cover layout, malformed sides, mixed/permuted native sampling through
one encoder pass, independent head gradients, shared embeddings, full-batch masked
pointer references, unused-family skipping, unselected overflow, global evaluation
row indices, scalar value/RNG rejection and coupled PPO rollback.
