# Release cross-play

`releases.json` is the machine-readable historical release registry and evaluation
contract. The initial weights-free Teacher release is the **annotated** `v0.0.1`
tag at `2cd104c8b8f0c5d1bed9988dfad4ddf4defd23f6`. Simulator source is pinned to
`18db0f62d9a2b94e755c43fd29a959db204cc20b`, Map1, TCP lockstep.

## Mandatory gate

The candidate must win **strictly more than 50% of all scheduled games against
each past release independently**. Ten fixed seeds, each played on both sides,
give 20 games per opponent: **10/20 fails; 11/20 passes**. Wins against one release
cannot compensate for failure against another. Draws and tick/wall timeouts stay
in the denominator and are never wins. Order rejections, unexpected process
failures, identity mismatches, missing/duplicate games, unknown outcomes, and
inconsistent terminal messages invalidate the evaluation regardless of win count.

All `v*` tags must appear in the registry, must be annotated, and must peel to
their recorded commits. The runner never creates or moves a tag. A future release
is evaluated against the existing registry **before** its tag is created. After
approval, a maintainer adds the annotated tag and its tag/commit/policy/simulator/map
entry together as part of the release procedure. The initial release is the
bootstrap; it has no predecessor to defeat.

## Run a changed candidate

Requirements: Python 3.12+ (stdlib only), Git, and the Rust toolchains and Cargo
dependencies specified by the archived repositories. Run from `drysua`:

```sh
# Wait for the candidate's source changes to be complete before building it.
cargo build --release --locked --quiet --bin drysua --no-default-features \
  --target-dir artifacts/temp/candidate-build

PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_crossplay.py \
  --candidate-binary artifacts/temp/candidate-build/release/drysua \
  --candidate-metadata 'cargo build --release --locked --quiet --bin drysua --no-default-features --target-dir artifacts/temp/candidate-build; SOURCE COMMIT + DESCRIPTION OF UNCOMMITTED CHANGES' \
  --run-name candidate-crossplay-001
```

Replace the metadata placeholder with the actual provenance. Alternatively point
`--candidate-binary` at the other agent's completed build. The runner snapshots and
hashes that explicit binary; it does not rebuild the mutable candidate checkout.
Its archived historical builds and dedicated Cargo target directories cannot race
with another agent editing Teacher or building a different candidate. Use a fresh
run name each time; existing runs are never overwritten. `--bota-repository` can
identify another local Git repository containing the pinned simulator commit.

### Explicit neural candidate (full gate)

The runner defaults to `--candidate-policy teacher` for backward compatibility.
For a neural Map1 candidate, **both** `--candidate-policy hybrid` and
`--candidate-weights DIRECTORY` are required. Supplying weights with Teacher is an
error, not an ignored option; Hybrid without weights is also an error. This is
independent of the bot CLI's own default policy. Every seat receives an explicit
policy, and only Hybrid seats receive `--weights-directory`.

After the candidate source/schema compatibility work is complete, run from `drysua`:

```sh
cargo build --release --locked --quiet --bin drysua --no-default-features \
  --target-dir artifacts/temp/neural-release-build

PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_crossplay.py \
  --candidate-binary artifacts/temp/neural-release-build/release/drysua \
  --candidate-policy hybrid \
  --candidate-weights artifacts/temp/neural-v004-default-e8 \
  --candidate-metadata "cargo build --release --locked --quiet --bin drysua --no-default-features --target-dir artifacts/temp/neural-release-build; source=$(git rev-parse HEAD); dirty=$(git status --porcelain); accepted BC source=artifacts/temp/neural-v004-default-e8" \
  --run-name neural-v004-default-e8-crossplay-001
```

This is the full **60-game** gate against the currently registered `v0.0.1`,
`v0.0.2`, and `v0.0.3` Teachers, not a shortened smoke. A fresh run name is required.
Build only after edits have settled; the metadata command records provenance but
does not lock the working tree. The runner never trains or modifies source weights.

The accepted BC source directory supplies **`drysua.weights.safetensors`**, the
deployment-only SafeTensors artifact, not a training checkpoint or a directory of
per-layer files. Its sole tensor is `model.parameters`, a finite F32 vector of
the runtime's exact parameter count. The SafeTensors `__metadata__` object contains
decimal-string values for `action_schema_hash`, `feature_schema_hash`,
`model_schema_hash`, `ppo_schema_hash`, `ppo_schema_version`, and
`ppo_rules_audit_version`. The binary validates the exact metadata, tensor name,
shape, dtype, and finite values. Metadata must match the current runtime contract
or its exact audited v13 inference-compatible tuple (see `ppo-v15-migration.md`).
Old training resumes remain rejected. The runner never rewrites metadata to make
old artifacts load.

Before building opponents, the runner copies only that canonical file to
`<run>/candidate-weights/drysua.weights.safetensors` and makes it read-only.
Missing, empty, symlink, and over-256-MiB files are rejected. Source/copy hashes
are checked around the copy; `report.json` binds the source path, snapshot path,
SHA-256, and explicit policy to the binary SHA-256. Games use only the snapshot,
so subsequent training output cannot change the evaluated weights. Final identity
checks cover all binaries and weight snapshots. No `.previous`, temporary file,
or other checkpoint fallback is copied. Schema/load failure remains a failed run;
there is **no retry as Teacher**. The current Hybrid implementation selects neural
inference on Map1 (its Map0 Teacher routing is not used by this registry). As with
binary provenance generally, the harness trusts the supplied executable to honor
its CLI; wire observations do not prove its internal inference implementation.

### Future tagged Hybrid opponents

Historical Teacher entries stay unchanged and weights-free. Registry schema 1
also accepts a release entry with `"policy": "hybrid"` and:

```json
"weights": {
  "path": "artifacts/v0.0.4",
  "sha256": "<64 lowercase hexadecimal characters>"
}
```

The path must be exactly `artifacts/<that entry's tag>`. The canonical deployment
file must be committed **in the registered tagged source** at that path. The
runner extracts it from the immutable tag archive, verifies the registered SHA-256,
and snapshots it under `<run>/weights-<tag>`. It never takes historical weights
from the mutable checkout. Teacher entries with a `weights` field, missing Hybrid
weights, traversal/absolute/other-version paths, and malformed hashes fail closed.
The top-level registry `policy: teacher` remains the legacy registry contract,
not an override of explicit per-release or candidate policy. No registry entries
or tags are added by this implementation.

### Compact Tactical deployment

The canonical Tactical artifact is **`drysua.tactical.bin`**. Trainers must write
`TacticalPolicy::to_bytes()` to that filename; it is not a SafeTensors file and
does not use the Hybrid/PPO schema. The file starts with the exact UTF-8
`TACTICAL_SCHEMA_DESCRIPTOR`, followed by 172 little-endian F32 parameters in
`W1[8,16], b1[8], W2[4,8], b2[4]` order. The live loader reads at most
`TACTICAL_FILE_BYTES + 1` bytes, then validates exact length, schema, and finite
parameters in `[-4, 4]` through `TacticalPolicy::from_bytes`. Missing files, links,
directories, extra/truncated bytes, and invalid values are errors before connecting.
There is no default-file or Teacher/Hybrid fallback.

```sh
cargo build --release --locked --quiet --bin drysua --no-default-features \
  --target-dir artifacts/temp/tactical-live-target

artifacts/temp/tactical-live-target/release/drysua play \
  --policy tactical --weights-directory artifacts/temp/tactical-default \
  --addr 127.0.0.1:4455 --limit 30000
```

`--weights-directory` is explicitly required for Tactical, including implicit
play without the `play` subcommand. Hybrid retains its existing `.` default and
Teacher remains weights-free. The library APIs are
`play_tactical(address, name, limit, weights_directory)` and
`play_tactical_on(wire, seated, limit, &TacticalPolicy)`; the filename is exported
as `TACTICAL_FILE_NAME`. Tactical uses the existing live ACK, snapshot/event,
decision cadence, order persistence, readiness, and rejection state machine,
calling `Teacher::decide_tactical` on both maps. It never constructs `PolicyModel`
or initializes Candle. FeatureEncoder observation is unchanged for now, including
Hybrid's Map0 path. `TacticalPolicy::default()` selects Teacher behavior; it is a
parity baseline, not evidence of learned strength.

The release runner accepts `--candidate-policy tactical` with required
`--candidate-weights DIRECTORY`, snapshots **only** `drysua.tactical.bin`, and
binds/checks its SHA-256 exactly as for Hybrid. The runner bounds Tactical files
to 1..16,384 bytes; the executing binary enforces its exact schema/length. If both
artifact types are present, the explicit policy selects exactly one and the other
is not copied. Historical Teachers still have no weight arguments. Future
`"policy": "tactical"` registry entries use the same `weights.path: artifacts/<tag>`
and lowercase `weights.sha256` contract described above, but hash the canonical
Tactical file from that immutable tagged archive.

After trained weights are ready, the **unchanged full release gate** command is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/release_crossplay.py \
  --candidate-binary artifacts/temp/tactical-live-target/release/drysua \
  --candidate-policy tactical \
  --candidate-weights artifacts/temp/TACTICAL_TRAINED_OUTPUT \
  --candidate-metadata 'EXACT BUILD COMMAND, SOURCE COMMIT, DIRTY PATCH AND TRAINING PROVENANCE' \
  --run-name tactical-trained-crossplay-001
```

Replace the output/provenance placeholders and use a new run name. A two-sided
baseline TCP smoke is not this 60-game gate and cannot approve a release.

Default Tactical parity evidence is retained at
`artifacts/temp/tactical-default-tcp-001/evidence.md`. At seed 9,200,003 versus
the archived `v0.0.3` Teacher, both TCP matches ended with Dire winning at tick
8,890 (candidate Radiant loss, candidate Dire win), without rejections, errors,
or timeouts. Builtin inference matched every accepted order/application tick and
the complete terminal stats against each TCP replay. The default artifact is
`artifacts/temp/tactical-default/drysua.tactical.bin`; this is not trained weight
evidence and no full Tactical gate has been run.

Exit codes: `0` approved, `1` failed gate, `2` setup/build/registry failure. An
interrupted or incomplete report is not approval. No selected-opponent, early-win,
pooled-score, or reduced-seed release mode is provided.

## What actually runs

* `git archive` extracts immutable source under `artifacts/temp/<run>/sources`.
  Each release directory is a sibling of `bota`, so its original `../bota/crates/*`
  path dependencies resolve against the **pinned** archive, not the working tree.
* The simulator and every tagged bot are separately built with `--release --locked`.
  Historical Teachers run their own tagged `play --policy teacher` implementations;
  they are never approximated by the candidate's Teacher. Tagged Hybrid releases
  use their own source and SHA-bound archived weights. Unsupported future
  policy/map/protocol contracts fail closed and require an explicit runner update.
* Two loopback TCP relays connect actual bot processes to the actual server.
  The first bot's wire `Welcome` must confirm slot zero before the second launches.
  Snapshot viewers and both CLI summaries confirm the planned Radiant/Dire seats.
* The pinned postcard protocol's framed `MatchOver` winner and final seat records
  are observed directly from **both server connections**, then checked against
  both bot summaries. Reported wins alone cannot manufacture victory. Wire
  `OrderRejected` is counted independently of each bot's rejection summary.
* Games are limited to 30,000 ticks and 90 wall seconds. Server ACK timeout exceeds
  the game wall deadline, preventing load-induced realtime-style tick advancement.
  Relay tick/frame/byte limits, log/replay size checks, startup/build deadlines,
  and process-group cleanup bound work. Terminal EOF uses a half-close and drains
  final client ACKs so unread ACKs do not reset the connection carrying MatchOver.

These are trusted locally built binaries, **not a sandbox for hostile executables**.
Wall timeouts under host load still count; do not run release evaluation on an
overloaded machine. Fixed seeds and lockstep reproduce gameplay, not wall timings,
ephemeral ports, or platform-independent binary hashes. Cargo may download locked
dependencies into its normal cache; source/build/run artifacts and build temporary
files are confined to `artifacts/temp`. No worktrees or changes to `bota` are needed.

## Evidence and tests

Each run retains `report.json`, a candidate snapshot, harness source snapshot,
archive tarballs, build command/toolchain metadata and logs, and binary SHA-256s.
Each game retains `result.json`, exact server/client commands, wire observations,
exit status/cleanup information, client/server logs, and the server's `match.brp`.
The final report includes the complete registry and per-opponent gate decisions.
Setup errors additionally produce `failure.json`; partial reports stay failed.
Hybrid runs additionally retain canonical weight snapshots and policy/weight
provenance for the candidate and every opponent.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts -p 'test_release_crossplay.py'
```

Tests were written before the implementation and cover the strict boundary,
per-opponent gate, denominator, missing/unknown/duplicate results, rejections,
process/seat/terminal identity failures, protocol framing and limits, and tag
identity. Regression tests also cover two issues discovered during real runs:
the replay file does not exist until match start, and full TCP close with unread
final ACKs can reset a bot before it consumes MatchOver. Failed exploratory runs
are retained rather than relabeled as successful evidence.

Hybrid support verification: **31 script tests passed**, including CLI ambiguity,
both policy/seat assignments, artifact size/type boundaries, SHA mismatch/copy
races, snapshot isolation, final weight-tamper invalidation, and immutable tagged
Hybrid weight paths. The unchanged live registry also validated all three Teacher
tags. No neural smoke or full gate was run during this script-only change while
candidate schema compatibility was being updated; these tests do not establish
neural gameplay strength or artifact compatibility. No Rust checks were needed.

Baseline evidence: `artifacts/temp/release-baseline-selfplay-v5/report.json`.
Both candidate and historical opponent were separately built from the registered
`v0.0.1` source against the pinned simulator. All 20 games had genuine terminal
winners, no rejections or validation errors, and exactly **10 wins / 10 losses**:
the gate correctly **failed**. This validates the pipeline, not a changed Teacher.

The original script suite passed 20 tests. Additional Rust checks ran only inside the
archived `v0.0.1` source, avoiding changes to another agent's mutable Rust files:
all-target/all-feature Clippy with warnings denied, `cargo fmt`, and `cargo machete`
passed. The debug all-feature test run exceeded the 120-second command timeout;
the release all-target/all-feature suite completed with 388 passed and 5 ignored.
