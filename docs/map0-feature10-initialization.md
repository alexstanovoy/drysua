# Full-map capacity, hull reach, and explicit M10 training initialization

## Capacity and feature facts

`MAX_TRACKED_ENTITIES` and `MAX_PROJECTILES` are now **4096**, independent of the
model's unchanged **96 unit / 32 projectile** rows. The observed 520-unit failure
was reproduced before changing code. The protocol at simulator commit
`18db0f62d9a2b94e755c43fd29a959db204cc20b` accepts a **4 MiB payload** plus a four-byte
prefix; it defines no 512-unit/projectile population maximum. Full Map0 waves,
neutral units, buildings, wards and projectile lifetimes make 512 inadequate.
4096 provides nearly eightfold headroom over the observed 520-unit incident.
It is a fixed application resource limit, not a protocol maximum or proof that an
indefinitely accumulating world can never exceed it. Entry 4097 is rejected with
the specific field/count/limit error before tracker mutation.

Representative serialized Snapshot frames (including prefix), verified without
truncation and against the unchanged protocol byte bound:

| Visible units | Projectiles | Frame bytes |
| ---: | ---: | ---: |
| 520 | 39 | 29,716 |
| 2048 | 2048 | 143,411 |
| 4096 | 4096 | 286,771 |

A separate mixed-catalog full-capacity test includes lane/neutral creeps, all
structure kinds, wards and periodically populated hero abilities/items/effects.
Its 4096-unit/4096-projectile frame is **295,045 bytes**.
These are **representative views**, not a claim that arbitrary maximal/fuzzed
nested lists at every unit can all fit one 4 MiB frame. The byte cap still applies
independently of list-count bounds; no codec or simulator changes were made.

Projectile histories remain fixed-size **boxed arrays**, including each of the
16 rollback states. No growable/unbounded history vector or increased model
padding was added. Tests exercise 4096 projectiles across the complete journal
on the normal release test-thread stack. The inline encoder remains below 64 KiB.
Worst-case duplicate/history scans are finite O(4096²), with at most 4096×32
token insert comparisons. Retained unit/provenance memory also scales with the
larger cap; this is not a constant-cost or universal real-time guarantee.

`OWN_IN_ATTACK_RANGE` and `UNIT_IN_ATTACK_RANGE` now compare center distance against
**that attacker's range plus both hull radii**, using the simulator's saturating
Fixed sum and inclusive squared-distance comparison. No attack-continuation
leeway is added. Tests cover one raw fixed-point unit below/on/above both distinct
reach boundaries for Radiant and Dire, creeps and every structure kind. All
normalizers, other numeric features, parameter shapes/order and 1,684,724 F32
parameter count are unchanged. No Teacher, Tactical or neural-training algorithm
was modified.

## Current semantic ABI

| Schema | Version | Hash |
| --- | ---: | ---: |
| Action | 3 | `1755359086494840931` |
| Feature | 10 | `15519817897416174399` |
| Model | 11 | `18229126264156367519` |
| PPO | 19 | `3810026640568905163` |
| League | 20 | `290030435976226060` |

Checkpoint envelope v2 and PPO/league rules audit 15 are unchanged. Model/PPO/league
changes are schema-only. **M10/F9 runtime weights and checkpoint resumes are not
compatible**, because attack-reach input meaning changed, even where dimensions
match. Normal runtime and resume loaders remain strict and reject before model
mutation. Old observation samples must be recollected/re-encoded from valid source
observations, not relabeled as current features. The old sparse feature golden
still passing does not establish general old-schema compatibility.

## Trainer-owner handoff: explicit initialization, not resume

```rust
let (model, source_sha256) = TrainingArtifact::initialize_selected_m10_for_training(
    source_directory,
    new_model_seed,
    device,
)?;
// Construct new BC/PPO optimizer and new RNG/progress/corpus state here.
```

The trainer must expose a **separate explicit initialization flag** to invoke this
method, never redirect runtime loading or checkpoint resume through it. The API
returns a newly constructed model rather than accepting an existing learner, so
old optimizer ownership cannot be transferred. A regression claims a new optimizer
and verifies step zero. Record the returned source SHA, original source path and
old tuple as initialization provenance alongside the new run's current tuple,
seeds and fresh optimizer configuration. Do not describe initialization as
continued optimizer training or mark the original artifact current-compatible.

Only this exact source is allowed:

`artifacts/temp/pure-neural-v10-dagger-002/sources/drysua/artifacts/temp/iteration-002/stage-0-epoch-008/drysua.weights.safetensors`

SHA-256 **`b3802642b34487d66fc3f0fe526e7f8b2a84df542793ac336b58ea046b8f8b53`**.
All six metadata fields must match, with no missing or additional fields:

| Field | Exact old value |
| --- | --- |
| `action_schema_hash` | `1755359086494840931` |
| `feature_schema_hash` | `9669721049329356661` |
| `model_schema_hash` | `720439888929233033` |
| `ppo_schema_version` | `18` |
| `ppo_schema_hash` | `6877503070358232325` |
| `ppo_rules_audit_version` | `15` |

Bounded reads, exact tensor name `model.parameters`, F32 dtype, full element
count/shape, finite values and the full file SHA are mandatory. No metadata is
rewritten. No arbitrary M10 checkpoint, partial legacy tuple, or generic
ignore-schema path is accepted. Compile-time current-version/hash/count pins
require re-auditing this one-way initializer after any subsequent ABI change.
The selected real fixture was loaded through initialization and rejected through
normal runtime loading; parameter bits and original file bytes were checked.

The trainer's existing `requires finalized model v10` guard remains its owner's
integration responsibility. Update it to the exact current tuple, add the explicit
initialization option, and create fresh Adam and current-feature samples. This
patch does not change `neural_training.rs`, its CLI, or any launcher/default.

## Separate Teacher-only baseline patch and action parity

`artifacts/temp/map0-bound4096/baseline-observation-only.patch` changes only the two
tracker limits and two feature compile-time bound assertions on a **copy** of
`artifacts/temp/map0-baseline-observationfix/sources`. It contains no hull-reach
correction or neural schema/loader changes. Existing history arrays are already
boxed and follow the new count constant. Teacher's live path observes that history
but does not consume encoded numeric feature rows when selecting actions.

**Restriction: this derivative is only a weights-free `play --policy teacher`
reference, not a neural-capable schema migration.** Do not use its unchanged old
neural metadata to export/load/train neural models. Keep the original sealed
baseline untouched and give the derivative new source/binary identities.

Both independently built versions were run against the original raw seat-visible
seed **9470001** streams, through all **46,760** snapshots and ACKs per side.
Complete `(tick, ClientMsg::Order)` byte-stream hashes match before/after:

- Radiant, 1206 orders: `7b31169750a48db2807e86bf0d1f379504551b49e22c44cf63d9f144b6f872ce`.
- Dire, 1158 orders: `72ae6d5ff0640e5c140984c514eb8c8973db6b53b2461c6c7bd5f28de75b28a7`.

Both report 15,587 decisions, zero rejections and the same terminal outcome/tick.
The source audit finds exactly `tracker.rs`/`feature.rs` changed; Teacher SHA stays
`29431d2381272f1c15ed63a798a240134d974fda4fee99d24444f65908129f34`.
This is complete live-controller action parity on recorded seat inputs, not a new
TCP match or universal parity when old history would have evicted units.

Evidence, before-source copies, patch, separately built baseline and parity JSON
are under `artifacts/temp/map0-bound4096`. All checks are release/locked/offline;
no dependencies, debug builds, training runs, commits or tags.
