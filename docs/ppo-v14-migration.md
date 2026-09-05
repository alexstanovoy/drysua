# PPO v14 training migration

**Superseded by [PPO v15 / league v16](ppo-v15-migration.md).** This document
records the historical v14 contract, not the current reward or resume semantics.
v14 training checkpoints cannot resume under v15.

This integrates the learner changes recorded in
`artifacts/temp/neural-v004-transfer-evidence.md`; it is not a release or promotion.

| Contract | Version | FNV-1a descriptor hash | Rules audit |
| --- | ---: | ---: | ---: |
| PPO | 14 | `15610409340106916160` | 13 |
| League | 15 | `17988894822320626017` | 13 |
| Checkpoint wire layout (unchanged) | 2 | `4581258024746721724` | — |

The learner descriptor now binds PPO-only detached critic input, default Adam LR
`3e-6`, and pre/post-step KL rejection. Candidate overshoot or evaluation error
restores exact parameters, Adam moments, step and policy revision before releasing
the exclusive parameter lock. Applied reports use post-step sampled rollout-policy
KL, weighted over the complete effective minibatch. This does not bound cumulative
drift from the BC anchor. Reward, GAE, model architecture and inference are unchanged.
The rules-audit bump covers learner semantics, not a simulator reward change.

## Compatibility boundary

Training `load`, `load_compatible` and resume remain current-schema only. There is
no training migration or optimizer conversion. A prior PPO version or hash fails
with `CheckpointError::SchemaMismatch`: `checkpoint schema does not match this build`.

Runtime reads allow exactly the current metadata map or this frozen prior map:

| Metadata key | Exact prior value |
| --- | --- |
| `action_schema_hash` | `1018254919734743331` |
| `feature_schema_hash` | `13875648161437731669` |
| `model_schema_hash` | `10644717168650027237` |
| `ppo_schema_version` | `13` |
| `ppo_schema_hash` | `11103744726312279053` |
| `ppo_rules_audit_version` | `12` |

This is a runtime-only compatibility audit, not a quality/promotion whitelist of
all weights bearing that tuple. It preserves accepted BC anchor `e63f0bdb478ebb8b`.
Exact metadata equality rejects extra/missing fields and mixed tuples. Tensor name,
shape, dtype and finite validation still precede model mutation. Compile-time hash
assertions require re-auditing this exception if inference schemas change.

A separate inference metadata format would require another artifact migration;
the minimal whitelist avoids that churn and never rewrites input metadata. New
runtime saves still emit current metadata. No general old-version bypass exists.

## Verification and probe disposition

Regression tests were added first: old training acceptance failed the rejection
test before the bump; the exact prior runtime fixture then failed after the bump
and before the whitelist. Release checkpoint tests cover current roundtrip,
prior runtime import with byte-identical metadata, wrong/older/mixed tuples,
invalid tensors, prior training version/hash, and prior rules-audit capture.

Read-only local-artifact proof:

```sh
cargo test --release --all-features --quiet accepted_anchor_runtime_loads_but_pre_migration_probe_training_rejects -- --include-ignored
cargo test --release --all-features --quiet bc_anchor_critic_only_step -- --include-ignored --nocapture
```

The first test checks the real anchor's exact prior metadata and fingerprint,
then rejects both conservative u1/u8 training checkpoints. The second runs the
existing CPU/CUDA fixed-observation critic regressions using that anchor; no
artifact is modified. These are tests, not new training runs.

Verified after integration: full release all-target/all-feature suite **407 passed,
8 ignored, zero failed**; checkpoint suite including the local proof **19 passed**;
accepted-anchor CPU/CUDA critic tests **2 passed** separately. Release Clippy with
warnings denied, `cargo fmt`, `cargo machete` and `git diff --check` passed. No
dependencies were added.

Preserve pre-migration probe files and their evidence under `artifacts/temp`.
They retain stale training metadata and must not be resumed, relabelled, published,
promoted or used as post-migration training evidence. No probes were rerun for this
migration, and neither probe replaces the accepted anchor. Scripts and release
packaging remain separately owned; this document describes the Rust reader contract.
