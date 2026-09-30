# Continuous collection

`train-annealed` keeps every world busy and overlaps collection with learning.

## Shape

```
slots (--slots, default 64)   one live game each; a finished game is replaced in the same job
lanes (--lanes, default 2)    one thread, CUDA stream and actor weight replica per lane;
                              lane l owns slots l, l+lanes, ...; at most 64 slots per lane
simulation pool               --simulation-threads workers (default: available cores) that
                              step any lane's slots; never changes results
learner                       the session thread; trains update u while lanes collect u+1
```

A lane runs rounds: one batched inference over its slots (self-play opponent rows
share the call; each frozen snapshot gets its own call on the lane's replica), then
one advance job per slot on the pool. Lanes never wait for each other inside an
update. A lane's share of update `u` ends after the first round in which it has
closed `--samples-per-update / --lanes` retained intervals (default 8,000 in total),
so an update holds at least the target and at most two more intervals per slot.

## Determinism

Every decision depends on round indices and seeds, never on thread timing:

- Game `n` of slot `s` has seeds, seat (`(s + n) % 2`) and opponent drawn from
  `(seed, s, n)`; its spawn modifiers come from the update it started in.
- A lane's rounds, batch composition and part boundaries are its own sequence.
- Update `u` is collected with the actor weights of update `u - 1`
  (`PIPELINE_STALENESS = 1`); lanes switch weights exactly at their part boundary
  and block only if the learner is late.
- Samples, episodes and snapshots are merged in lane, round and slot order.

The simulation thread count is excluded from the run scope; a test checks that one
and four threads produce identical checkpoints.

## Resume

Checkpoint `u` stores, besides model, Adam and shuffle RNG: the actor weights that
collect `u`, its spawn modifiers, and every slot's in-flight game as its plan, the
actions both seats have taken, both RNG states and the behaviour statistics of the
open retained interval. A resumed run replays each game's logged actions (in
parallel on the pool) to the exact state it had at the boundary and continues;
the replay verifies the plan and rejects any divergence. Stop/resume therefore
equals an uninterrupted run bit for bit, including games against neural opponents.

## Opponents

`--opponent` is repeatable and forms a per-game mixture: `teacher[:w]`,
`self[:w]` (the lane's current actor weights) and `weights:<dir>:<w>` (frozen
snapshots, fingerprinted in the run scope). The default is `teacher:1`.
`draw_opponent` in `src/ppo_arena/slot.rs` is the single pluggable schedule;
adaptive schedules may only use reports of updates every lane has finished.
Episode logs carry `slot=`, `game=` and `opponent=`.

## What changed numerically

- PPO is 1-stale: the importance ratio uses the stored behaviour log-probability,
  and batches may mix samples from actor versions `u-2..u`; the trainer rejects
  anything older (`PPO_MAX_STALENESS = 2`).
- Games no longer align with updates: a game's early and late intervals may land in
  different updates, and GAE truncates at the update boundary with the stored
  bootstrap value. Update reports count games that finished during the update.
- Inference batches are a lane's slots (plus self-play rows), so GEMM shapes and
  therefore sampled trajectories differ from the old waves.
- Adaptive environment transitions apply two updates later than before.
