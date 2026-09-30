# Continuous collection

`train-annealed` keeps every world busy and overlaps collection with learning.

## Shape

```
slots (--slots)               one live game each; a finished game is replaced in the same job;
                              default 16 per available core (max 256)
lanes (--lanes)               one thread, CUDA stream and actor weight replica per lane;
                              lane l owns slots l, l+lanes, ...; at most 64 slots per lane;
                              default two per simulation group, more if the slots need them
simulation groups             --simulation-groups (default: last-level cache domains, i.e.
                              CCDs) split consecutive lanes and --simulation-threads workers
                              (default: available cores) into pools, so a lane's round never
                              waits on another CCD; --pin-threads pins each group to its
                              domain (opt-in). Neither changes results
learner                       the session thread; trains update u while lanes collect u+1
```

On resume, omitted `--slots`/`--lanes` adopt the recorded values, so a run survives
a change of core count.

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

## Disk writes

A checkpoint is written every `--checkpoint-interval-seconds` (default 600, at
least 60), on a graceful stop and at the invocation's last update: the tensor file,
the runtime weights and the manifest, each synced once, then one directory sync;
generation snapshots drawn since the previous checkpoint are written just before
it. History milestones are exported only at checkpoints. Logs go through one
buffered stream flushed every two seconds and at exit. A crash loses at most one
interval, which the resume recomputes bit-exactly.

## Opponents

`--opponent` is repeatable and forms a per-game mixture: `teacher[:w]`,
`harass-push[:w]` (the HarassPush rule policy in `src/scripted/`), `self[:w]` (the
lane's current actor weights) and `weights:<dir>:<w>` (frozen snapshots,
fingerprinted in the run scope). The default is `teacher:1`.
`draw_opponent` in `src/ppo_arena/slot.rs` is the single pluggable schedule;
adaptive schedules may only use reports of updates every lane has finished.
Episode logs carry `slot=`, `game=` and `opponent=`.

## Learner

The learner is device resident (`src/model/device_learner.rs`). An update's
samples are packed into encoder rows and uploaded once, in 256-row chunks, into
preallocated device columns. Each Adam step gathers its minibatch on the device,
runs `--training-microbatch` rows (256/512/1024/2048, default 512) per
forward/backward, sums the microbatch gradients on the device and applies a
clipped f32 Adam there. Per step the host reads back the loss sums with a
finiteness probe, the gradient norm, the new Adam moments and the candidate KL;
a rejected candidate is restored from device copies of the parameters. The loss
definitions live only in `src/model/ppo_objective.rs`.

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
- The learner computes each microbatch's losses divided by the whole minibatch
  size and adds gradients, so `--training-microbatch` only regroups float sums.
  Adam now runs in f32 on the device (it used f64 temporaries on the host) and the
  gradient norm sums per-tensor f32 squares in f64.
