# drysua architecture

drysua is a private Rust + Candle bot for the deterministic simulator `../bota`. It
plays Shadow Fiend (`HeroId(2)`) on Map2: 1v1 mid only, 900 pregame ticks plus 15
minutes at 30 ticks/s, native cap 27,900 ticks. A side loses when it loses any tower
or when its hero dies for the second time; reaching the cap is a draw.

The crate depends on the sibling checkout `../bota` (`bota-proto`, and `bota-server`
with the `builtin` feature). Features: `builtin` (in-process arena, training, eval)
and `cuda` (Candle CUDA backend; build with `NVCC_CCBIN=/usr/bin/g++-15` under gcc 16).
Production `train-annealed` binaries need `DRYSUA_GIT_COMMIT` and `BOTA_GIT_COMMIT`
at compile time; they are recorded in every checkpoint.

## Commands

| Command | Purpose |
| --- | --- |
| `drysua [play]` | Join a TCP server and play one match; Teacher by default, Neural with `--weights-directory`, `--policy harass-push` for HarassPush |
| `drysua train-annealed` | PPO training against a per-game opponent mixture ([training](training.md)) |
| `drysua eval` | Frozen pool evaluation, both sides of each seed ([training](training.md#evaluation)) |
| `drysua duel` | Rule policy against rule policy on paired seeds in the builtin arena ([local play](local-play.md)) |
| `drysua checkpoint-inspect` | Read-only checkpoint JSON ([checkpoint inspection](checkpoint_inspection.md)) |
| `drysua reward-observer` | Score copied frames of a human game ([human reward play](human-reward-play.md)) |

`scripts/train.py` runs training campaigns; `scripts/play.sh` (or the workspace
`play.sh`) launches a local human game ([local play](local-play.md)).

## Data flow

```text
bota-server World (arena.rs, builtin)   or   TCP Link (link.rs, wire.rs)
  -> the seat's own ServerMsg stream only
  -> StateTracker (tracker.rs): fogged-unit memory, projectiles, history, Map2Reward
  -> FeatureFrame + ActionSpace legality masks (feature.rs, action.rs)
  -> PolicyModel (model.rs)   or   Teacher (teacher.rs)
  -> StructuredAction -> RazeAim macro (raze_aim.rs) -> order bookkeeping (persistence.rs)
  -> wire Order, or nothing for Continue (seat.rs live, ppo_arena in training)
```

Decisions happen at ticks 1, 4, 7, ... (`MAP2_DECISION_INTERVAL_TICKS = 3`). A seat
sends at most one order per tick; `Continue` sends nothing and keeps the current
server-side order. The model, Teacher, reward and features never see `World`, only
the seat's `ServerMsg` stream. See [model and actions](model.md).

## Policy isolation

These rules keep the policy from winning through simulator leaks rather than play:

- Policy, reward and Teacher get only the seat's own messages and terminal result.
- `match_id`, seeds and numeric `EntityId`s are not features; ids are only memory keys.
- Invisible handles are never target candidates, even where the server would accept them.
- `OrderRejected` reasons are telemetry, never observations (no server oracle).
- Fogless replays and spectator views are never training data.
- Builtin and TCP paths must produce identical orders; evaluation is side-paired and
  its seeds are separate from training seeds.

## Training loop

`train-annealed` is one long-lived process (details in [training](training.md) and
[continuous collection](continuous_collection.md)):

- `--slots` worlds (default 64) each always hold a live game against an opponent
  drawn per game from `--opponent` (`teacher`, `harass-push`, their styled variants
  `teacher-styled` and `harass-push-styled` that draw a new style per game, `self`, frozen
  `weights:`, `league` snapshots of the learner), weighted by PFSP per update.
- `--lanes` inference threads (default 2) batch their slots through a weight replica;
  a shared pool steps the simulations.
- The first decision, every non-Continue decision and each Continue that finds the
  open interval `MAP2_CONTINUE_STRIDE = 8` decisions long begin a PPO sample (about
  0.4 per decision); an update is due at `--samples-per-update` samples (default
  24,000, about sixteen games).
- The learner trains update `u` while lanes collect `u + 1` with the weights of
  `u - 1` (1-stale PPO; the ratio uses stored behaviour log-probabilities).
- PPO defaults: Adam lr 1e-5, 4 epochs, minibatch 2,048, clip 0.2, value coefficient
  0.5, entropy 0.004, gradient clip 0.5, λ 0.99979 per tick (0.995 per 24 ticks, a
  160 s horizon), γ = 1, target KL 0.02 (step rollback and early stop).
- Domain randomization draws spawn modifiers per environment generation; the
  adaptive schedule extends or ends generations by win rate, and the last
  `--zero-updates` are unmodified.
- Every update commits a checkpoint (model, Adam, RNG, in-flight slot games); resume
  is bit-identical to an uninterrupted run. `--history-every` exports runtime weights.

Reward: [reward.md](reward.md) (version 8, outcome plus potential-based shaping).

## Versions

Every contract carries a version and hash: action schema 8, feature schema 26, model
schema 27 (2,004,663 parameters), checkpoint format 20 (collection state v2), reward 8. Backward
compatibility is not kept: a change bumps the version and old artifacts are
rejected; old bots are played from their git commit. `--initial-weights` is the one
exception: it warm-starts from any runtime weights, reusing every tensor whose name and
shape match and initializing the rest.

## Source map

| Path | Contents |
| --- | --- |
| `src/cli.rs` | Command line |
| `src/link.rs`, `src/wire.rs`, `src/seat.rs` | TCP framing and the live seat loop |
| `src/arena.rs` | In-process bota-server match (`builtin`) |
| `src/tracker.rs` | Per-seat state tracking |
| `src/feature.rs` | Feature encoder |
| `src/action.rs`, `src/raze_aim.rs`, `src/persistence.rs`, `src/readiness.rs` | Action space, masks, aimed razes, order and item bookkeeping |
| `src/teacher.rs`, `src/teacher_economy.rs` | Scripted Teacher controller |
| `src/scripted/` | Rule-policy seat (`ScriptedPolicy`), HarassPush, shared tactics, progress watchdog, `duel` runner |
| `src/model.rs`, `src/model/` | Policy/value network, sampling, side actors |
| `src/ppo.rs`, `src/ppo/` | PPO, GAE, Adam |
| `src/ppo_arena.rs`, `src/ppo_arena/` | Collection (slots, lanes, pool), annealed session, eval |
| `src/randomization.rs`, `src/adaptive_*.rs` | Domain randomization and adaptive schedule |
| `src/map2_reward.rs`, `src/map2_reward/`, `src/reward_observer.rs` | Reward and its passive observer |
| `src/checkpoint*.rs`, `src/training_history.rs` | Checkpoint format, inspection, milestone weights |
| `src/telemetry/` | Buffered logs and timing |
| `scripts/train*.py` | Campaign controller, report and dashboard |
| `scripts/eval_pool.py`, `eval_stats.py` | Pool evaluation store, ratings, sequential test and reports |
| `scripts/play*.py`, `scripts/play.sh` | Local human play and reward reports |

## Open problems

- **Plateau against Teacher.** Frozen paired evaluation (E0, `drysua eval`) puts the
  M25 lineage (u100, u200v1, u200v2) at about 31–34% against Teacher with no
  forgetting between checkpoints. Training barely moves the policy: KL per update
  is about 1e-4 against a 0.02 target.
- **Credit assignment.** The critic is a 256×256 MLP trained with the trunk and λ is
  per tick (160 s horizon); GAE truncates at update boundaries (about 375 samples,
  115 s of game per slot per update) and bootstraps there from the stored value, so
  nothing is cut off without a bootstrap. `ppo_update` logs two explained variances:
  against the λ-returns the critic trains on, and (`explained_variance_mc`) against
  the Monte Carlo return of the samples whose game ended in the batch. The outcome is
  mostly unpredictable from a state: on 550 frozen u200v2 self-play games a
  win-probability model fitted on 380 games explains only 0.11–0.13 of the held-out
  return variance (0.06 with 100 games; opponent-private inputs add ≤ 0.01), and an
  update sees about 18 outcomes, so an explained variance near 0.1 is the ceiling,
  not a bug. The hand potential predicts the winner at chance in the first two game
  minutes (AUC 0.47–0.57) where a logistic model over the global features reaches
  0.64–0.73, so early decisions get little outcome credit from shaping or the critic.
  With lr 1e-5 the policy moves about 0.009 KL per update; against the old, freezing
  Teacher that changed the win rate by +3 points (p = 0.38, 400 paired games).
- **Aiming.** Razes fire along the hero's facing and bota has no face order. Action
  schema 7 makes a raze one decision (`Cast` untargeted, at an entity, or toward a
  point candidate such as a cluster landing or a fog guess) expanded by `RazeAim`,
  instead of a turn and a cast that were rarely retained together.
- **Teacher weaknesses** the policy has not found: it retreats at 40% HP without
  hysteresis, never pushes into tower range while the enemy hero is within 1,200,
  spends its mana on creep razes. HarassPush (`src/scripted/harass_push.rs`)
  wins 115 of 200 `drysua duel` games against it (seeds 1–100, both sides); it won 83
  while razes still damaged towers and both scripts razed them. Before
  Teacher abandoned unreachable walks and Tango trees (`src/scripted/progress.rs`) it
  froze in most of those games and HarassPush won 195; E0 and every other number
  measured against Teacher before that fix are against the freezing Teacher.

In flight: reward 8 and aimed razes in campaigns; HarassPush as an opponent and imitation target to seed the push strategy;
collection and learner performance (the learner is host-bound).
