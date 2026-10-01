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
                              CCDs, that the slots and lanes can feed) split consecutive lanes
                              and --simulation-threads workers (default: physical cores; SMT
                              siblings would slow the lanes and learner) into pools, so a lane's round never
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
closed `--samples-per-update / --lanes` retained intervals (default 24,000 in total),
so an update holds at least the target and at most two more intervals per slot.
A retained interval begins at the first decision, at every non-Continue decision, at
a Continue that finds the open interval eight decisions long, and at any other
Continue with probability 1/8 (a draw from the game's seed and the decision index);
it closes at the next retained decision (bootstrapped from that decision's value) or
at the game's end. A sample's weight is the inverse of its decision's retention
probability (8 for a drawn Continue, else 1), and the learner scales its normalized
advantage by the weight over the update's mean weight, so the policy gradient over
retained samples estimates the one over every decision. Retention that depends on the
action without this weight biases the gradient: any offset of the advantages (a
critic that over- or underestimates a side or a phase) then moves the probability of
Continue: in the phase-2 runs of 2026-10-01 the side the critic overrated (Dire) drifted
into idling and lost by deaths.
Lanes compute behaviour statistics for every policy row.

## Determinism

Every decision depends on round indices and seeds, never on thread timing:

- Game `n` of slot `s` has seeds, seat (`(s + n) % 2`) and opponent drawn from
  `(seed, s, n)` and the mixture of the update it started in, as are its spawn
  modifiers.
- A lane's rounds, batch composition and part boundaries are its own sequence.
- Update `u` is collected with the actor weights of update `u - 1`
  (`PIPELINE_STALENESS = 1`); lanes switch weights exactly at their part boundary
  and block only if the learner is late.
- Samples, episodes and snapshots are merged in lane, round and slot order.

The simulation thread count is excluded from the run scope; a test checks that one
and four threads produce identical checkpoints.

## Resume

Checkpoint `u` stores, besides model, Adam and shuffle RNG: the actor weights that
collect `u`, its spawn modifiers and opponent mixture, the PFSP outcome window, the
league snapshots it wrote, and every slot's in-flight game as its plan, the
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
fingerprinted in the run scope) and `league:<w>`: each of the `--league-size`
(default 4) latest snapshots of the learner, taken every `--league-every` (default
20) updates. The default is `teacher:1`.

Every published update gets its own mixture (`src/ppo_arena/opponents.rs`). With
`--opponent-schedule pfsp` (the default) each configured weight is multiplied by
`(1 - p)^2`, where `p` is the Laplace-smoothed score (win 1, draw 1/2) of the last
100 games against that opponent; `fixed` keeps the weights. Weights are exact
integers computed after update `u` from the games of updates up to `u` and apply
to update `u + 2`, so they are a pure function of the run. After each update the
trainer logs `event=opponent_pool update=… scope=<opponent> games= wins= win_rate=
probability=`; the dashboard charts win rate and probability per opponent, with
league snapshots (`u0040`) folded into one `league` series.

League snapshots live in memory. A checkpoint writes each snapshot the next two
publications or any in-flight game still need once, as runtime weights under
`checkpoint/league/u<update>/` with its fingerprint in the collection state, and
deletes the ones it no longer records after the commit; its own update's snapshot
is the checkpoint model. Episode logs carry `slot=`, `game=` and `opponent=`.

## Learner

A preparer thread (`src/ppo_arena/collector.rs`) takes each update's lane parts
and builds its batch (rollout, GAE, report, snapshots) while the learner still
trains the previous update; the learner logs the parts' games and takes the
batches in update order.

The learner is device resident (`src/model/device_learner.rs`). An update's
samples are packed into encoder rows and uploaded once, in 256-row chunks, into
preallocated device columns. Each Adam step gathers its minibatch on the device,
runs `--training-microbatch` rows (256/512/1024/2048, default 2048) per
forward/backward, sums the microbatch gradients on the device and applies a
clipped f32 Adam there. The Adam moments stay on the device (read back only
for checkpoints). Per step the host reads back the loss sums with a finiteness
probe, the gradient norm, one moment/parameter finiteness check and the
candidate KL; a rejected candidate is restored from device copies of the
parameters and the previous moment tensors. `--target-kl` is the threshold of
both guards; `--kl-guard early-stop` (recorded in the run scope; default
`post-step`) skips that candidate pass and rollback: an update stops at the
first minibatch whose KL before its step, from the gradient's own forward pass,
exceeds `--target-kl`, and taken steps are kept. Imitation and critic warm-up
change only the loss, so they work the same under either guard. The
loss definitions live only in `src/model/ppo_objective.rs`. With
`--side-networks separate` each network trains on its own side's rows of every
minibatch ([model](model.md#side-networks)).

Device memory: a CUDA trainer reserves a fixed VRAM budget at startup
(`--vram-budget-mib`, default from `vram_budget_estimate`: the sum of every
stream's own high-water mark, because streams do not share freed memory: the
learner's optimizer state, staged update at rollout capacity and one Adam step
at the full microbatch, each lane's largest sampling call, and the replicas the
configuration keeps (per lane the actor, frozen weights opponents and, with a
league, three times its size: current, still-played and spare milestones; each
in its side-network layout, a separate model holding 1.89 times the parameters),
plus 5% for block rounding; per-row costs are measured by
`vram_budget_constants_bound_measured_peaks`). Production (24k samples, 256
slots, microbatch 2048, league 4) reserves 15,008 MiB against a measured
300-update peak of 13,122 MiB. The budget becomes the device's
allocation pool, capped, filled up front and never released; streams reuse
only their own freed blocks, so each lane and the learner settle into a steady
state and the process footprint is the budget plus the CUDA context from the
first update on. An allocation beyond it fails instead of growing. Each update
logs `event=device_memory` with the budget, reserved, live and peak live MiB.
Lanes reload retired league replicas in place and share one CUDA handle per
lane.

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
