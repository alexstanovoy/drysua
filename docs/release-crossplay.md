# Release cross-play (historical)

Archived 2026-09-18. The Map1 TCP release runner and Map0 challenge framework
were removed from the tracked tree. Their source is in git history at `41bc295`;
the full former procedure is in this page at `2ddb68b`. Old build/evaluation
commands and temporary report paths are not live instructions. No external
archive is required to recover the tracked source or this documentation.

There is currently no in-tree Map2 checkpoint qualification command. Launcher
smokes and identity checks do not establish gameplay strength or replace a
release gate. For supported operation, see [local play](local-play.md),
[reward observation](human-reward-play.md), and
[neural deployment](neural-deployment.md).

## Registry and provenance

`releases.json` remains the machine-readable historical registry. Preserve its
v0.0.1–v0.0.4 identities, simulator pins, policies, seeds, and artifact hashes.
The initial weights-free Teacher is annotated tag `v0.0.1`, commit
`2cd104c8b8f0c5d1bed9988dfad4ddf4defd23f6`, pinned to simulator
`18db0f62d9a2b94e755c43fd29a959db204cc20b`, Map1, TCP lockstep.
v0.0.1–v0.0.3 were Teachers; v0.0.4 used trained Tactical weights.

Historical opponents used their own immutable tagged source and simulator,
never the candidate's Teacher implementation. Weight-bearing releases bound
the canonical artifact under `artifacts/<tag>` in that tagged source to its
registered SHA-256. Hybrid used `drysua.weights.safetensors`; Tactical used
`drysua.tactical.bin`. Historical artifacts require their historical runtimes:
metadata rewriting or fallback to Teacher was not permitted. Candidate reports
bound explicit policy, source/build provenance, binary hash, and immutable
weight snapshots, with identity checks before and after evaluation.

## Historical gate and evidence

Each opponent required strictly more than half of all twenty scheduled games:
ten fixed seeds on both sides, so 10/20 failed and 11/20 passed. Scores could
not be pooled across opponents. Draws and tick/wall timeouts stayed in the
denominator. Missing or duplicate games, rejections, process failures, identity
mismatches, and inconsistent terminal outcomes invalidated a run. Each game
was bounded to 30000 ticks and 90 wall seconds. Both wire streams and bot
summaries had to agree; incomplete reports were never approval.

The v0.0.1 self-play validation produced ten wins and ten losses and correctly
failed. Default Tactical parity was not learned-strength evidence. Trained
v0.0.4 subsequently passed the sixty-game gate against its three predecessors.
The historical registry defines eighty games for a candidate against all four.
See the tagged source and the prior version of this page for original evidence
descriptions; deleted temporary reports are not presented as available files.

`scripts/release_wire.py` retains bounded framing and pinned postcard decoding
for current and historical contracts, including terminal/cap handling. Its
focused tests retain malformed/truncated input coverage and registry-pin checks;
the live admission relay also imports its integer decoder. Removing the old
evaluation framework does not remove these wire compatibility contracts.
