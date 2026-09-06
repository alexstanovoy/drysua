# Tactical policy search

Historical experiment note: the commands and v0.0.4 artifacts below refer to
Tactical v1 (172 parameters). The current laning preview uses Tactical v3
(236 parameters) and rejects v1/v2 weights. Reproduce the old run using its matching
tagged source; current `--initial-policy` and `--opponent-policy` inputs must use
the current schema. See `laning-preview.md` for the newer experiments and limits.

The first trained release is v0.0.4, not the rejected full-policy PPO experiments.
It uses a 16→8→4 F32 neural selector (172 parameters) over Teacher/Fight/Recover/Farm.
Economy, legal action construction and safety remain deterministic Teacher logic.
Population-based parameter search optimizes full-match results; this is not PPO.

## Three-hour experiment outcome

The v1 search and continuation evaluated 6624 development games in about 15.1
minutes of optimization wall time. The selected weights passed actual historical
TCP cross-play: 19/20 versus v0.0.1, 18/20 versus v0.0.2, 14/20 versus v0.0.3.
The clean committed build reproduced every outcome. Artifact, SHA-256 and source
identity are recorded in `artifacts/v0.0.4/` and `releases.json`.

The apparent objective-aware v2 improvement failed fresh confirmations (16/32
twice). Its code was reverted; artifacts and source snapshots remain temporary.
Do not load those v2 weights as v1.

A successor was trained against both Teacher and frozen v0.0.4, with worst-opponent
win rate ranked before total wins. It evaluated 3328 games in 474.212 seconds.
Selection scored 21/32 versus Teacher and 20/32 versus v0.0.4; fresh confirmation
fell to 18/32 and 14/32. It is **not v0.0.5**, was not release-gated, and does not
replace v0.0.4. Evidence: `artifacts/temp/tactical-v005-mixed-001/evidence.json`.

In total, these five search runs evaluated 16608 development games. Elapsed
engineering time also included implementation, tests, builds, reviews and TCP
gates; it was not three hours of continuous GPU training. No GPU is needed here.

## Reproduce mixed-opponent training

From `drysua`, using a new output directory name:

```sh
cargo build --release --locked --quiet --no-default-features --features builtin --bin train_tactical
timeout --signal=TERM --kill-after=10s 12m target/release/train_tactical \
  --population 16 --generations 16 --pairs 2 --workers 4 \
  --seed 9203000 --selection-seed 9203100 --confirmation-seed 9203200 \
  --selection-pairs 16 --tick-limit 30000 --wall-seconds 540 \
  --initial-policy artifacts/v0.0.4/drysua.tactical.bin \
  --opponent-policy artifacts/v0.0.4/drysua.tactical.bin \
  --output-directory artifacts/temp/tactical-mixed-new-run
```

Do not merely add `--opponent-policy` to otherwise-default settings: doubling
the default schedule exceeds the deliberate 5120-game run cap. Match deadlines
are cooperative; retain the external watchdog. Only complete same-cohort
comparisons may replace the archive. Confirmation and release approval are
separate from adaptive selection; repeated seeds are not independent evidence.

For the next release, run the full **80-game** historical gate, including v0.0.4.
Every prior release must be beaten strictly more than 50% of scheduled games;
selection scores and pooled wins cannot substitute for that requirement.
