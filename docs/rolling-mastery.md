# Rolling training mastery (current reward6)

The mastery implementation introduced with reward4 is retained. The subsequent
[reward5 rebalance](reward-rebalance.md) changes coefficients, Draw reward and two
appended accounting inputs, not mastery windows, stages or threshold semantics.

## Selection and configuration

The default `--opponent-schedule teacher` and the deterministic
`weak-warmup-v1` selection rules are unchanged. New **opt-in** `mastery-v1`
trains against the existing Weak opponent, then the existing Teacher. Historical
v0.0.1..v0.0.4 compatibility adaptations remain external evaluation opponents.
No new Weak strategy, Teacher override, policy input, critic, GAE, BC, mask or
architecture change is part of mastery.

Example configuration, after explicitly obtaining compatible M22 runtime weights:

```sh
drysua train-full --updates 2 --checkpoint-directory /new/empty/run \
  --initial-weights /compatible/m21/weights --complete-episodes \
  --opponent-schedule mastery-v1 --mastery-window 50 --mastery-win-percent 80 \
  --opponent-win-percent weak=90 --opponent-win-percent teacher=80 \
  --environments 6 --rollout 1163 --epochs 1 --minibatch 512 \
  --gamma-per-tick 1 --checkpoint-seconds 1
```

- `--mastery-window`: positive integer1..1024, default50.
- `--mastery-win-percent`: integer1..100, global default80.
- Repeated `--opponent-win-percent opponent=integer`: at most one each for
  `weak` and `teacher`, overriding the global default for that opponent.
  Unknown opponents, duplicates, noninteger/out-of-range values and malformed
  assignments fail with explicit errors. Per-override text is bounded to32 bytes.
- Mastery-specific options with another schedule are rejected, not ignored.
  Nondefault schedules require full episodes, with existing E2/E4/E6 bounds.
- The canonical checkpoint command records window and resolved Weak/Teacher
  thresholds in that order. Override argument order does not change run identity.
  An unused global default overridden for both opponents is not a different
  effective configuration. No floating-point win-rate comparison is used.

## Exact criterion and ordering

Only completed games from the CURRENT training opponent enter its window:
Win=true, Loss/Draw/completed-task TimeCap=false. Infrastructure errors have no
game-outcome variant and abort the current transaction; they are never fake losses.

The predicate is `window_full && 100 * recent_wins >= percent * window_size`.
With defaults,49 wins from49 games are insufficient;39/50 fails and40/50 passes.
At capacity the oldest flag is evicted. This is neither an all-time average nor
a consecutive-win streak. With window7/80%, six wins are required, not five.

An entire batch uses the same opponent/stage. Its completed games are ordered by
native terminal tick, then stable stream index (not worker completion order).
Only after the WHOLE successful batch is optimized is the updated window installed
and the gate applied once. No opponent changes midbatch, even if an intermediate
prefix would have passed. E6/window50 first checks a full window at54 completed
games and uses the last50 of those54. Paired sides and seed derivation are preserved.

Weak→Teacher clears the window and stage-game count. Teacher→Completed retains
the full qualifying Teacher window/count as evidence, forces a coherent checkpoint
and ends normally even if `--updates` allows more work. A restored Completed
checkpoint does not collect/replay games or optimize again.

`--updates` remains an explicit cumulative maximum. If the gate is not reached,
the run stops at its update budget without claiming mastery; later strict resume
continues the same window. Existing wall/resource limits still apply. A large
update budget is not permission for an unprotected unbounded job; use the guarded
segmentation described in [experiment-safety.md](experiment-safety.md).

Mastery completion is a TRAINING criterion, not release/promotion qualification.
There is no new separate20-game greedy evaluation gate.

## Coherent persistence

The codec introduced in checkpoint8 (now linked as checkpoint9) stores typed `CheckpointRun.mastery_config` and
`CheckpointProgress.mastery`. Nonmastery runs store None for both. The generic
reserved `curriculum_stage` is not repurposed as an implicit mastery counter.

The run stores window and effective percentages. State stores Weak/Teacher/Completed,
the current stage's completed-game count and the ENTIRE ordered recent win-flag
window, oldest first. A logical ring wrap is serialized without relying on its
in-memory allocation/head offset. No RNG field is abused and no history is rebuilt
from logs. Staged state is installed only after successful PPO optimization; any
collector/optimizer error leaves the previous committed window/model/Adam/RNG intact.

Validation rejects invalid presence/stage/flag bytes, window/count bounds, mismatched
config/state, already-qualifying active windows, insufficient Completed windows and
stage-game counters inconsistent with completed updates and E2/E4/E6 batch sizes.
The stage count is bounded by `MAX_TRAINING_COUNTER`; queue size never exceeds1024.
The existing immutable generation + canonical tensor + manifest-last/fsync protocol
commits parameters, both Adam moments/step, shuffle/actor RNG, progress and mastery
together. Stage transitions force checkpointing independently of normal cadence.

Strict load/restore checks run scope, including typed configuration and canonical
command, before tensor reading or model/optimizer mutation. Changed schedule,
window or effective threshold cannot resume; Git-only provenance migration cannot
bypass those comparisons. No automatic retry or configuration migration is added.
Telemetry `training_mastery_progress` reports stage, stage_games, recent_games,
recent_wins, configured window/percentage and Completed status after successful updates.

## Reward and semantic migration

Actual Map2 outcomes retain their labels, but terminal reward is now:

| Outcome | Terminal reward | Mastery win flag |
|---|---:|---|
| Win | +0.2 | true |
| Loss | -0.2 | false |
| Draw, including native cap Draw | 0 | false |
| Completed learner-task TimeCap | -0.2 | false |
| Infrastructure failure/resource abort | no reward/game | not recorded |

Reward6 terminal values and unchanged reward5 dense/opening rules are in
[reward-rebalance.md](reward-rebalance.md). Normal zero-initial-potential starts
have dense bounds[-1.0488,+0.445]. General primed baselines have bounds[-1.4488,+0.845].
Neither has guaranteed winner-return dominance with the new terminal gaps0.2/0.4.
Legacy non-Map2 reward profiles are not silently reinterpreted.

| Contract | Version | Hash |
|---|---:|---:|
| Action | 5 | 10658390830565586343 |
| Feature | 20 | 9233114641639769206 |
| Model | 22 | 4891874295003631291 |
| PPO | 35 | 13569352384922890857 |
| League | 35 | 7630384836954837061 |
| Checkpoint | 10 | 2382613649322819763 |
| Map2 reward | 6 | 1084583101075978392 |

Rules audit30, imitation audit20. Dimensions are global92/unit84,
62 named tensors and1,700,020 F32 parameters. Mastery state is NOT a neural feature.

## Explicit old-parameter initialization only

Ordinary loaders reject old M19/reward3/checkpoint7, M20/reward4/checkpoint8 and M21/reward5/checkpoint9.
Do not edit old metadata to claim current compatibility. Existing frozen artifacts,
descriptors, weights and reports remain historical; no gameplay/reward equivalence
or qualification transfers.

`TrainingArtifact::initialize_selected_m19_for_nonwin_reward(directory, seed, device)`
accepts only original M19/u162 runtime SHA256
`9d0b88128bb4a74d636e0774ab53eeed2e2aea306c3ab0186a5698f41f92afea`, with exact frozen
F17/M19/PPO32/rules27/reward3 metadata and descriptor, names/F32 shape and finite values.
It preserves all old parameter bits and pads the two appended global rows with positive zero;
it does not restore old Adam/progress/mastery/RNG. Record the returned provenance
description with any subsequent new run and use fresh state. This is not resume,
not a metadata-only relabel and not evidence that the old critic matches new targets.

Existing explicit M14/M16/M17 pinned parameter initializers retain their frozen
original source contracts and now export the current target metadata when used.
No additional M18/M20 or pilot-weight pin is added. Reward6 provides an explicit
pinned M21/u300 parameter-only initializer; it resets optimizer/mastery/RNG and starts
a new Weak stage, not an old checkpoint resume. See [reward-rebalance.md](reward-rebalance.md).
Manual Neural play requires compatible weights; the launcher never manufactures
them or falls back to Teacher.
