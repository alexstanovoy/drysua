# Human-feedback laning preview — not a promoted release

The reported problems were actionable: late arrival, excessive homeward movement,
an unconditional Wand/Ring opening, no deliberate creep pull, and fragile raze
execution. This preview addresses those mechanisms. It has **not** passed the
historical release gate and is not v0.0.5. v0.0.4 remains the registered release.

## Try the corrected controller

From the parent `bots` directory, with the compatible Map1 server already running:

```sh
cargo run --release --locked --quiet --no-default-features \
  --manifest-path drysua/Cargo.toml --bin drysua -- \
  play --policy teacher --addr 127.0.0.1:4455 --name drysua-preview
```

This is the **updated deterministic controller**, not a successful neural successor.
No weights are needed. Old v0.0.4 Tactical weights require their matching tagged
runtime; they must not be passed to the new 236-parameter Tactical v2 parser.

## Changed mechanics

- Decisions start at tick 1 and continue every three ticks, including pregame.
  Snapshot/Events completion and live/builtin order parity remain enforced.
- Wraith Band + Tango spend 595 of 600 starting gold. No mandatory Ring or Wand.
  Stick requires confirmed visible enemy casting; boot components build Treads.
- Regeneration does not overwrite existing Mending; emergency Stick use can spend
  one useful charge. Component purchases respect missing cost and decode to a
  server-legal leaf order when the composite validator requires full-price gold.
- Neural Recover is a short 200-unit lane backoff, not a repeated fountain trip.
  Retreat/aim/pull goals have bounded lifetime and stall handling. Targeted
  regeneration and own razes preserve server-body bookkeeping and its deadline;
  obsolete movement is explicitly stopped. Real emergency retreat remains.
- Raze selection uses predicted hit margin. Turning is followed by Stop before
  casting; farming razes use the same execution path. Ready lethal opportunities
  can avoid an otherwise over-conservative health-threshold retreat.
- Farm can attack-click a visible hero and pull creeps toward an allied ranged
  creep. Hold/cooldown accounting is bounded; an imminent last hit takes priority.
- Tactical v2 exposes 24 observations, including facing, raze margin, motion,
  nearby creeps, aggro wait and navigation progress, to a 24→8→4 neural selector.
  These are action capabilities and constraints, not a complete scripted lane plan.

## Final human-style proxy evidence

The same three scripted opponents, two seeds and both sides ran against archived
v0.0.4 and the final Teacher preview. Scripts are not calibrated human players.
Arrival uses a 1500-unit lane-center proxy: **1019–1031 → 122–134 ticks**;
horn is 900. Separate TCP/builtin tests use a stricter 600-unit arrival boundary.

Ten identical games reach the full measurement phase in both runs:

| Metric | v0.0.4 | Preview |
| --- | ---: | ---: |
| Visible-creep XP-range occupancy | 67.04% | 72.21% |
| Healthy behind-tower posture | 6.51% | 2.67% |
| Hero damage dealt | 10345 | 23874 |
| Hero damage taken | 3391 | 9841 |
| Last hits | 145 | 130 |
| Denies | 16 | 33 |
| Candidate deaths / rejected orders | 0 / 0 | 0 / 0 |

Both preview Radiant–RazeFarm games ended in wins at tick 2024, before the phase;
they are excluded from the table. Other games capped without an adjudicated
result. More pressure is not unconditionally better: incoming damage increased
and last hits remain worse. No human-strength or raze-accuracy percentage is claimed.
Actual spell damage and creep retarget/pull are separately simulator-tested.

Full evidence: `artifacts/temp/laning-teacher-preview-final-001/EVIDENCE.md`.
Frozen source/build evidence: `artifacts/temp/laning-teacher-preview-build-001/`.

## Learning and release disposition

Two builtin searches evaluated 6656 development games in 15.14 minutes. Their
selected policy scored 22/32 against the equally prepared new Teacher on fresh
confirmation, but failed historical TCP comparisons: 18/20 vs v0.0.1, 18/20 vs
v0.0.2, 6/20 vs v0.0.3 and 10/20 vs v0.0.4. It does not replace the preview default.

A corrected-execution default also failed historical comparison (19/20, 19/20,
6/20, 8/20). A further search against actual v0.0.3/v0.0.4 processes completed
1248 games including founder validation, but the selected weights scored only
9/16 and 7/16 on fresh validation. No new release tag was created or gate weakened.
These older cohorts use earlier source snapshots; their outcomes are not silently
reassigned to the final optional-Stick correction. All historical comparisons
include an asymmetric legal pregame advantage, so they cannot establish human skill.

The historical gate still applies to promotion. The proxy suite makes the user's
reported lane behavior observable alongside it, rather than treating bot wins as
the only definition of quality. See `laning-evaluation.md` for metric limitations.
