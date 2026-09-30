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
| `drysua [play]` | Join a TCP server and play one match; Teacher by default, Neural with `--weights-directory` |
| `drysua train-annealed` | PPO training against a per-game opponent mixture ([training](training.md)) |
| `drysua eval` | Frozen-weights paired evaluation ([training](training.md#evaluation)) |
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
  drawn per game from `--opponent` (`teacher`, `self`, frozen `weights:`).
- `--lanes` inference threads (default 2) batch their slots through a weight replica;
  a shared pool steps the simulations.
- One decision in `MAP2_RETENTION_STRIDE = 8` is retained as a PPO sample (about one
  every 24 ticks); an update is due at `--samples-per-update` samples (default 8,000).
- The learner trains update `u` while lanes collect `u + 1` with the weights of
  `u - 1` (1-stale PPO; the ratio uses stored behaviour log-probabilities).
- PPO defaults: Adam lr 3e-6, 4 epochs, minibatch 2,048, clip 0.2, value coefficient
  0.5, entropy 0.01, gradient clip 0.5, λ 0.98 per retained sample, γ = 1, target
  KL 0.02 (early stop).
- Domain randomization draws spawn modifiers per environment generation; the
  adaptive schedule extends or ends generations by win rate, and the last
  `--zero-updates` are unmodified.
- Every update commits a checkpoint (model, Adam, RNG, in-flight slot games); resume
  is bit-identical to an uninterrupted run. `--history-every` exports runtime weights.

Reward: [reward.md](reward.md) (version 8, outcome plus potential-based shaping).

## Versions

Every contract carries a version and hash: action schema 6, feature schema 23, model
schema 25 (1,812,983 parameters), checkpoint format 20, reward 8. Backward
compatibility is not kept: a change bumps the version and old artifacts are
rejected; old bots are played from their git commit. `--initial-weights` is the one
exception: it warm-starts from any runtime weights with the current parameter layout.

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
| `src/model.rs`, `src/model/` | Policy/value network, sampling, side actors |
| `src/ppo.rs`, `src/ppo/` | PPO, GAE, Adam |
| `src/ppo_arena.rs`, `src/ppo_arena/` | Collection (slots, lanes, pool), annealed session, eval |
| `src/randomization.rs`, `src/adaptive_*.rs` | Domain randomization and adaptive schedule |
| `src/map2_reward.rs`, `src/map2_reward/`, `src/reward_observer.rs` | Reward and its passive observer |
| `src/checkpoint*.rs`, `src/training_history.rs` | Checkpoint format, inspection, milestone weights |
| `src/telemetry/` | Buffered logs and timing |
| `scripts/train*.py` | Campaign controller, report and dashboard |
| `scripts/eval_compare.py` | Paired comparison of `eval` results |
| `scripts/play*.py`, `scripts/play.sh` | Local human play and reward reports |

## Open problems

- **Plateau against Teacher.** Frozen paired evaluation (E0, `drysua eval`) puts the
  M25 lineage (u100, u200v1, u200v2) at about 31–34% against Teacher with no
  forgetting between checkpoints. Training barely moves the policy: KL per update
  is about 1e-4 against a 0.02 target.
- **Credit assignment.** The critic is one `Linear(256, 1)` on a detached trunk, and
  λ = 0.98 per retained sample reaches back only about 40 s. Reward 8 makes every
  return equal the outcome and moves credit earlier through shaping.
- **Aiming.** Razes fire along the hero's facing and bota has no face order. Action
  schema 6 makes an aimed raze one decision (`Cast` at an entity) expanded by
  `RazeAim`, instead of a turn and a cast that were rarely retained together.
- **Teacher weaknesses** the policy has not found: it retreats at 40% HP without
  hysteresis and never pushes into tower range while the enemy hero is within 1,200.

In flight: reward 8 and aimed razes in campaigns; critic capacity and trunk
gradient, λ, retention of non-Continue decisions and a PFSP opponent mixture; a
scripted HarassPush opponent that exploits Teacher's retreat to seed the push
strategy; collection and learner performance (the learner is host-bound).
