# drysua v0.0.3

Weights-free Teacher for Bota Map1. Use the v0.0.1 server/client instructions and
`play --policy teacher`; no GPU or weights are needed.

## Improvements

- Combat build: Magic Wand, Ring of Regen, Sage's Mask, Wraith Band,
  Power Treads, Chainmail. Avoid filling active slots with unused consumables.
- Retreat at 40% health rather than 25%, preserving existing pressure checks.
- Stay in the fountain's 1200-unit recovery radius until both HP and mana reach
  95%. Channel preservation still takes precedence over recovery and retreat.

## Cross-play evidence

| Opponent | Wins | Losses | Win rate |
| --- | ---: | ---: | ---: |
| v0.0.1 | 20 | 0 | 100% |
| v0.0.2 | 18 | 2 | 90% |

Ten registered seeds (9000001–9000010), both sides, actual separately built tagged
bots and TCP server. All games terminal; zero rejections or validation errors.
These results exceed the 80% target against each opponent. One strategy was
evaluated and a second complete run reproduced the scores. A separate invocation
timed out during archived builds before any games and remains failed/incomplete.

In the first run, 35 of 38 wins ended at the enemy's second hero death with both
towers standing. No ablation was run: the contribution of each change is unknown.
Existing tower-raze behavior remains, including the simulator/Dota discrepancy
documented in v0.0.2. Simulator commit is unchanged:
`18db0f62d9a2b94e755c43fd29a959db204cc20b`.

These are reused release seeds, not held-out evidence or a human-strength claim.
Local evidence: `artifacts/temp/teacher-v003-final-complete/report.json` and
`artifacts/temp/v003-evidence.md`. Release identity is the annotated tag and
`releases.json`; the final post-commit gate summary is recorded separately.
