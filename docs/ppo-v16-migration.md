# Action v3 / PPO v16 compatibility cut

This integrates the action-owner Buy change, pregame cadence, and their dependent
contracts. It does not change the PPO architecture, optimizer/reward algorithms,
DAgger algorithm, or the independent Tactical network. Simulator source remains
`bota` commit `18db0f6`.

| Contract | Version | FNV-1a descriptor hash | Rules audit |
| --- | ---: | ---: | ---: |
| Action | 3 | `1755359086494840931` | — |
| Feature | 8 | `10322490384647633864` | — |
| Policy model | 8 | `3097714014199697774` | — |
| PPO | 16 | `11450737853127354910` | 15 |
| League | 17 | `14981756892451872554` | 15 |
| Checkpoint wire layout, unchanged | 2 | `4581258024746721724` | — |

Rules audit 14 was already assigned to the v15 reward repair; this cut uses 15,
not a reused audit number. The PPO descriptor preserves that normalized reward
contract, detached critic, LR `3e-6`, and post-step rollback semantics.

## Why feature/model versions change without new shapes

Buy legality now requires positive missing leaves, sufficient total missing cost,
and leaf capacity, rather than the root's full purchase price. When the server's
root validation would reject the full-price order, a legal structured Buy decodes
to the first missing leaf. The shop/action indices and network head widths stay
the same, but the meaning of a legal Buy changes.

Encoded item legality depends on ActionSpace masks. Feature v8 therefore explicitly
binds action v3; model v8 explicitly binds action v3 and feature v8. Compile-time
assertions freeze these dependencies. All feature dimensions and the policy
model's **1,684,724 parameters** remain unchanged. This is a semantic compatibility
cut, not an architecture change or a weight conversion.

The PPO arena/deployment descriptor records tick-complete decisions with default
cadence `1 + 3n`, including pregame, instead of suppressing decisions until the
horn. Its Teacher economy binding records the custom-bota once-only plan: Wraith
Band, Tango, Boots, optional Stick, Gloves, Belt. These are policy semantics, not
additional PPO heads/features or a claim about standard Dota item economics.

Tactical remains a **separate v2 artifact**: 24 inputs, 8 hidden units, 4 controls,
236 parameters. It does not use PPO feature/model/checkpoint hashes. Its combat
features, controls, action/economy descriptor integration and fresh training are
owned separately; this migration does not edit `tactical.rs` or Teacher helpers.

## Strict compatibility decision

There is **no legacy runtime whitelist**. The former exact v13 tuple is invalid
because action v2 semantics differ from v3. Tensor shape equality is insufficient.

- Current runtime metadata must equal the complete six-key map: current action,
  feature, model and PPO hashes, PPO version 16, and rules audit 15.
- Action-v2/PPO v13, v14 and v15 runtime weights all reject. Missing/extra fields,
  independently stale fields, and mixed tuples reject before model mutation.
- Training `load` and `load_compatible` require every current version/hash.
  `--migrate-provenance` also calls the strict loader first; it can change Git
  provenance within a compatible run, never schemas, action semantics or rewards.
- No file is relabelled or rewritten on rejection. Names, shape, dtype, finite
  validation, bounded reads and training SHA-256 checks are unchanged.

The exact checkpoint error is `CheckpointError::SchemaMismatch`, displayed as
`checkpoint schema does not match this build`. The training orchestration wrapper
displays `PPO model error: checkpoint schema does not match this build`.

Historical anchor `e63f0bdb478ebb8b`, probe files and tagged release artifacts stay
immutable. Run historical artifacts only through their matching tagged binaries
and independent historical source trees. Do not import raw legacy tensors to
bypass the loader, patch old metadata, or mix current behavior into a historical
tag build. Fresh Tactical v2 training and the historical-tag cross-play release
gate remain main-owner operations, not part of this schema integration.

## Regression proof

Tests were changed before compatibility logic. The initial build failed the old
action-v2 compile-time assertions. After updating only those assertions, the new
v13 runtime rejection test failed because the whitelist still admitted it. The
provenance-migration regression also showed v15 passing schema validation before
the PPO bump. Neither reproduction ran a training update.

`src/tests/checkpoint.rs` covers:

- Current runtime serialization/load and complete metadata equality.
- Rejection of v13/v14/v15 runtime tuples with valid tensor payloads, preserving
  model parameters, policy identity, and the input file bytes.
- Wrong/mixed current metadata rejected before even malformed tensor validation.
- Current-schema tensor name/shape/dtype/nonfinite validation.
- Independent old action/feature/model bindings and old PPO version/hash pairs
  rejected on training load; prior rules audits rejected on capture.
- v13/v14/v15 rejected by the actual provenance-migration entry point, with no
  training callback or manifest rewrite.
- An ignored local-artifact proof that the real old BC anchor now rejects, as do
  the preserved conservative u1/u8 training probes.

The old anchor-dependent critic tests are no longer a way to load legacy weights.
CPU and CUDA critic regressions instead use the same seeded fresh current model
and controlled 16-row batch, LR `3e-4`, zero advantages/entropy, and value targets
`old_value + 10`. They assert an applied step, lower value MSE, unchanged non-value
tensors, and zero sampled KL. Existing nonzero-Adam/uneven-microbatch rollback and
candidate-error regressions remain intact.

The seeded Teacher corpus regression was rebaselined after reproducing the count
change under the new behavior: 4,385 samples, Continue counts 1,512/315/315 across
the three splits. Existing per-action caps, split coverage, action-diversity and
collapse assertions were preserved; no collector or DAgger algorithm was changed.

```sh
cargo test --release --all-features --quiet tests::checkpoint -- --include-ignored
cargo test --release --all-features --quiet ppo_critic_only_step -- --include-ignored --nocapture
cargo clippy --release --all-targets --all-features -- -D warnings -D clippy::all
cargo test --release --all-targets --all-features --quiet
```

Verification on the integration checkout:

- Release all-feature binaries build successfully; the schema assertion blockers
  are gone. Release Clippy with warnings denied passes.
- Checkpoint suite including local legacy-artifact rejection: **21 passed**.
- Model suite including all four CUDA regressions: **61 passed**. Controlled
  critic MSE is `100 -> 98.8259048461914` on both CPU and CUDA, with sampled KL zero.
- Full release suite: **554 passed, 3 failed, 7 ignored**. The remaining failures
  are separately owned Tactical/pregame tests, not waived by this migration:
  - `fixed_v004_on_both_seats_has_identical_orders_and_opposite_candidate_results`
    still tries to read a v1 artifact with the v2 reader (`Schema`).
  - `pregame_v004_parameters_reach_lane_before_first_creep_meet_with_tcp_builtin_parity`
    still expects 172 rather than 236 parameters.
  - `pregame_teacher_and_tactical_reach_lane_before_first_creep_meet_with_tcp_builtin_parity`
    has Dire `creep_meet_in_lane_reach = Some(false)`, not `Some(true)`.
- Formatting, unused-dependency audit and diff whitespace checks pass.

The full release gate is **not green** until those owners resolve the remaining
failures. A schema bypass or weakened lane-arrival assertion is not a fix.

No dependencies, debug builds, training runs, artifact publication or commits are
part of this integration. Prior v14/v15 migration documents are historical records,
not current runtime-compatibility promises.
