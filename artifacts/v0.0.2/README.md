# drysua v0.0.2

Weights-free Teacher for the pinned Bota Map1 simulator. Launch with the same
server/client commands as v0.0.1 and explicitly select `play --policy teacher`.
No GPU or training checkpoint is required.

## Changes

- Ready combat spells can interrupt persistent attacks instead of being starved.
- Hero razes no longer require an additional 100-mana reserve beyond their cost.
- Razes can damage visible enemy towers under the pinned simulator rules.
- `releases.json` and `scripts/release_crossplay.py` enforce a strict majority
  against every past release separately, using actual tagged bot processes.

## Evaluation

The candidate won 12/20 TCP games against v0.0.1: 6/10 from each side,
8 losses, no draws/timeouts, zero rejected orders or validation errors.
Ten seeds (9000001–9000010) were each played on both sides. This passes the
required strict-majority gate; it is not a statistical significance claim.
An earlier strategy attempt scored 8/20 on the same seeds, so these are reused
development/evaluation seeds, not held-out evidence.

Local complete evidence: `artifacts/temp/teacher-v002-attempt2/report.json`.
Evaluated development binary SHA-256:
`bc24e7e5307799d9a9b543a149e9f0eae7b961804639827876da64dfdd2e3245`.
Source release identity is recorded by the annotated tag and release registry.

## Rules caveat

Simulator commit: `18db0f62d9a2b94e755c43fd29a959db204cc20b`.
Its Shadowraze damages buildings, unlike real Dota 2. This version deliberately
uses that existing rule; the simulator was not changed to favor the candidate.
This is a Bota Map1 improvement, not evidence of Dota-faithful competitive strength.
Correcting that simulator discrepancy requires an explicit evaluation-contract
change and fresh comparisons, not silently changing historical results.
