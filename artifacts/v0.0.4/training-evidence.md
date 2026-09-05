# Tactical neural population search, 2026-09-05

## Status

The trainer works, but neither bounded search met the requested 80% development
target, much less an 80% confidence lower bound. No external release gate, commit,
or tag was performed. Do not describe these weights as release-approved.

Best standalone policy directory:
`artifacts/temp/tactical-search-001/selection-017-1/`.
Canonical payload: `drysua.tactical.bin`.
SHA-256: `bfa19eb4ec2f11fd852f6f6fd6284be84aa0c0dc1b597a52fc00352d64ed5bd1`.

## Implementation ownership

- New `src/tactical_training.rs` and `src/tests/tactical_training.rs`.
- New `src/bin/train_tactical.rs`.
- Builtin-gated module/re-export additions in `src/lib.rs`.
- Explicit builtin-required binary entry in `Cargo.toml`.
- No edits to Teacher, tactical policy, live seat/CLI, simulator, or release scripts.

Training uses `Teacher::decide_tactical`, no FeatureEncoder, no tensor model, no
PPO, no hidden simulator state and no new dependency. Every tick's Snapshot and
Events are observed; decisions occur after pregame at ticks 901, 904, ... .
Orders pass through decoding, persistence and sent bookkeeping; any rejection
fails the run and retains previously committed archives. A same-tick terminal at
the deployment tick cap is a timeout, not a win.

Each generation evaluates the same paired development cohort for every candidate.
Fitness compares terminal wins first, then fewer timeouts, then capped public
death/farm differentials. Scores from different cohorts cannot be compared.
The population retains a Teacher-equivalent anchor, uses fixed nonzero default
hidden features, searches output weights first, then all 172 parameters.
Mutation scales are 0.1, 0.2, 0.35, 0.5; hidden perturbations are scaled by 0.2
after generation four. Mutation candidates are mirrored where population slots
permit; the initial population additionally contains constant macro founders.

Four workers, at most 24 candidates, 20 generations, 16 paired seeds per cohort,
30,000 ticks, 5,120 scheduled games and 2,700 wall seconds per invocation are hard
bounds. Actual runs each used a 1,200-second internal cap and a 21-minute external
watchdog. Candidate/seed/seat ordering does not depend on worker completion order.

## Results

| Round/cohort | Seed start | Games | Wins | Losses | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 1 selection | 9200100 | 32 | 24 | 6 | 2 |
| Round 1 confirmation | 9200200 | 32 | 20 | 12 | 0 |
| Round 2 selection | 9200600 | 32 | 19 | 11 | 2 |
| Round 2 confirmation | 9200700 | 32 | 23 | 5 | 4 |

Round 1: 16 candidates, 20 generations, four rotating paired training seeds from
9200000 through 9200079; 3,328 games; 447.805 wall seconds.
Round 2: same bounds, initialized from the best round-one payload; rotating
training seeds 9200500 through 9200579; 3,296 games; 456.477 wall seconds.
Round 2 retained its initial policy, i.e. the round-one best. No improvements
were selected on the second round's fixed cohort. Confirmation was not used to
choose weights. Repeatedly used selection cohorts are development selection data,
not unbiased held-out evidence.

The paired-sweep Wilson diagnostic deliberately treats a seed pair, not its two
correlated seats, as the Bernoulli unit. It is conservative for average game win
rate, approximate, and not confidence-calibrated after adaptive selection. None
of these results reaches the requested lower-bound threshold.

Logs: `tactical-search-001.log`, `tactical-search-002.log` in this directory.
Each run retains `config.json`, per-generation population results, immutable
policy/report archive directories, a `best.txt` pointer, confirmation and summary.
The small artifact auditor `tactical_search_summary.py` validates payload hashes,
cohort completeness and development-only seeds, and reports seat/outcome details.

## Diagnostic findings / next intervention

The founder probe, eight games each on 9200400..9200403, produced:
Teacher 4/4/0 W/L/T; Fight 6/2/0; Recover 2/5/1; Farm 5/2/1.
All 32 games took 3.605 seconds. These tiny cohorts are exploratory only.

The best policy is active: 615 overrides / 4041 sampled decisions on round-one
selection, 655 / 4219 on its confirmation. The diagnostic compares against a
cloned Teacher on the same candidate-visible trajectory every 32 decisions; it
does not compare against a counterfactual independent match or feed extra inputs
to the policy.

All six first-selection losses, and eleven of twelve first-confirmation losses,
occurred before a second own hero death. Map1 therefore ended through objective
loss in those games. Combat/death differential improved, but tower/wave retention
remains a problem. First-selection Radiant/Dire wins were 15/16 and 9/16; first
confirmation was 13/16 and 7/16. Ask the tactical owner to inspect objective and
side-dependent macro behavior, rather than treating this as dead-network
exploration or spending another unchanged search round.

## Verification

Tests were added before implementation; the initial trainer test run failed to
compile on the missing APIs. A later tick-boundary test specifically reproduced
acceptance of a forged terminal-at-cap report, then the validator was fixed.

15 trainer tests cover terminal-first ranking, timeout handling, cohort isolation,
missing/duplicate/rejected games, development/bound constraints, full-match
Teacher/default action parity on both seats, nondefault live Wire/trainer parity,
deterministic worker-count-independent tiny search, head-only mutation boundaries,
atomic archives and serialization, deadline preservation and paired diagnostics.

Latest complete checks:
- Release all-target/all-feature Clippy with warnings and clippy::all denied.
- Release all-target/all-feature tests: 473 passed, 8 ignored.
- Owned files formatted; cargo fmt ran with skip_children to avoid concurrent
  ownership collisions in other modules.
- cargo machete: no unused dependencies.
- git diff --check: clean.

Only optimized release builds/tests were run. Search round one preceded the
additional public fitness tick-boundary validation; its actual simulator outcomes
were already handled correctly at the boundary. Round two used the validated
implementation. Both policy action paths are the same as deployment.
