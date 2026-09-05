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

Exit codes: `0` approved, `1` failed gate, `2` setup/build/registry failure. An
interrupted or incomplete report is not approval. No selected-opponent, early-win,
pooled-score, or reduced-seed release mode is provided.

## What actually runs

* `git archive` extracts immutable source under `artifacts/temp/<run>/sources`.
  Each release directory is a sibling of `bota`, so its original `../bota/crates/*`
  path dependencies resolve against the **pinned** archive, not the working tree.
* The simulator and every tagged bot are separately built with `--release --locked`.
  Historical bots run their own tagged `play --policy teacher` implementations;
  they are never approximated by the candidate's Teacher. Unsupported future
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

Baseline evidence: `artifacts/temp/release-baseline-selfplay-v5/report.json`.
Both candidate and historical opponent were separately built from the registered
`v0.0.1` source against the pinned simulator. All 20 games had genuine terminal
winners, no rejections or validation errors, and exactly **10 wins / 10 losses**:
the gate correctly **failed**. This validates the pipeline, not a changed Teacher.

The script suite passed 20 tests. Additional Rust checks ran only inside the
archived `v0.0.1` source, avoiding changes to another agent's mutable Rust files:
all-target/all-feature Clippy with warnings denied, `cargo fmt`, and `cargo machete`
passed. The debug all-feature test run exceeded the 120-second command timeout;
the release all-target/all-feature suite completed with 388 passed and 5 ignored.
