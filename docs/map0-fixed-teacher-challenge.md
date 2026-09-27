# Fixed-Teacher Map0 challenge (historical)

Archived 2026-09-18. The Map0 challenge runner, configuration, and tests are
available in git history at `41bc295`; the complete former procedure is in this
page at `2ddb68b`. They are not current commands or a Map2 qualification gate.
The frozen baseline copies were deleted in the repo-slim audit. The obsolete
launcher smoke against that baseline has also been retired; current launcher
tests do not substitute a current server for the historical simulator.

## Epoch provenance

- Epoch: `map0-fixed-teacher-4096-v1`.
- Baseline manifest SHA-256:
  `e9c279bfe82b63a1c17ab2f3e1bbebb7b6936a0c256241251e1dfb4a992fbe8b`.
- Teacher binary SHA-256:
  `dcd3baf950a5dbf9018a40bb7be37b4413b06a18380d333987aaf8c29b3dbc3f`.
- Simulator SHA-256:
  `24a8efccb285308810678c7e3a8717b57814c9d8fecef386ca923cdb7c04e97c`.
- Observation-only patch SHA-256:
  `ecd8fc673d464fed74407f517228714fb5efdc82da4abda890173204d1792cd1`.

The derivative changed two tracker bounds and two feature assertions from 512
to 4096, not Teacher strategy, simulator, token rows, or neural schemas. The
frozen Teacher ran without weights. Restoring the challenge requires recovering
and verifying that exact provenance, not relabeling a new baseline.

## Historical interpretation

Development used at most ten paired seeds starting at 9870000. The final cohort
was seeds 9880000–9880049 on both sides: exactly 100 games, requiring 80 wins.
Final seeds were excluded from optimizer input and development selection.
Draws and timeouts remained non-wins; identity, protocol, or integrity errors
invalidated qualification. Each game had a 108900-tick cap and 180-second wall
watchdog. Preparing the harness did not qualify a candidate.

See [release history](release-crossplay.md) for the separate Map1 gate and
[local play](local-play.md) for supported current launcher usage.
