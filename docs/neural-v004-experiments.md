# Neural v0.0.4 investigation — not release-approved

No candidate demonstrated dominance. Keep v0.0.3 as the playable release.
Neural evaluation uses explicit Hybrid deployment: learned Map1 decisions plus
the existing safety/objective shield. Map0 Teacher results are not neural gains.

## Behavioral cloning

Eight-epoch CUDA/F32 pretraining on the v0.0.3 Teacher took 357.087 seconds.
It produced fingerprint `e63f0bdb478ebb8b`, passing the existing BC acceptance
checks, with held-out kind/full agreement 67.67%/64.53%.
Artifact: `artifacts/temp/neural-v004-default-e8/drysua.weights.safetensors`.
SHA-256: `6f1a90035f4cfa96e22dea396d22078de0cc3b951422c841b0be7cf4d3d2062e`.

The full weight-bound TCP release gate subsequently **failed**:

| Opponent | Wins | Losses |
| --- | ---: | ---: |
| v0.0.1 | 8 | 12 |
| v0.0.2 | 6 | 14 |
| v0.0.3 | 0 | 20 |

All 60 games terminated with zero rejections or validation errors. Gameplay
wall time was 23m09s; longest game 37.40s. Full report and source provenance:
`artifacts/temp/neural-anchor-crossplay-001/` and its `-provenance/` sibling.
Passing BC acceptance is therefore not evidence of historical superiority.

Increasing BC epochs and mixing active Teacher DAgger with expanded selection
did not produce an accepted stronger anchor. Those pretraining code experiments
were reverted. Evidence: `artifacts/temp/neural-v004-combat-evidence.md`.

## PPO continuation

Original PPO damaged the BC policy rapidly. A critic-only reproduction showed
substantial actor drift through the shared trunk; this is permitted by standard
shared actor–critic PPO, not a likelihood implementation bug. Conservative
critic isolation and lower LR are explicit design choices, not proven strength
improvements. Post-step KL rollback and reward repairs are described separately
in `ppo-v15-migration.md`.

The 30-minute bounded v15 experiment completed 25 durable updates, 819200
transitions, and 1600 optimizer steps. Final value loss was 0.268300 and sampled
post-step KL 0.000578. No reported KL stops or rejections. Intermediate snapshots
were retained; the requested 32-update target was not reached before timeout.

On four paired Map1 dev games per opponent:

| Candidate | Active Teacher W/L/T | Weak W/L/T |
| --- | --- | --- |
| Original BC | 0/0/4 | 4/0/0 |
| PPO u2 | 0/3/1 | 3/0/1 |
| PPO u13 | 0/1/3 | 3/0/1 |
| PPO u25 | 0/2/2 | 4/0/0 |

PPO u25 passes the reused-dev Weak/quality check, not release approval; it is
worse than BC on recorded Teacher outcomes and slower against Weak. It was not
submitted to historical promotion. Do not replace the original BC anchor with it.
Artifact: `artifacts/temp/neural-v015-e16-u32-run/u000025`, fingerprint
`b8756dde10169063`. Full evidence: `artifacts/temp/neural-v015-evidence.md`.

## Remaining problem

Neither more optimizer steps, lower value error nor small local KL establishes
better gameplay. Short discount/GAE horizons, imitation coverage and cumulative
actor drift remain hypotheses to test, not diagnosed causes with proven fixes.
No new release tag or accepted neural release weights were created.
