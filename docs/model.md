# Model and actions

## Observations

`FeatureEncoder` (`src/feature.rs`, schema 26) turns one `StateTracker` view into a
fixed-size `FeatureFrame`. Coordinates are team-canonical; every value is finite,
lies in [-1, 1] and has an explicit presence flag where it can be unknown. The
index modules (`global_feature`, `unit_feature`, ...) name every column.

| Part | Size |
| --- | --- |
| Global | 81, including the reward potential inputs ([reward](reward.md#policy-inputs)) |
| History | 7 global samples × 12 at ages 480, 240, 120, 60, 30, 15, 0 ticks; 16 policy-history samples × 16 (one-hot kind) |
| Map | 87 |
| Units | 96 current + 32 remembered + 2 own tokens × 109 |
| Tokens | abilities 56, items 102 (own bag, stash, courier, shop, then the nearest enemy hero's bag: 94 tokens), point candidates 60 (64 tokens), projectiles 41 (32), loot 75 (16) |

Scales follow the fight rather than the map: distances are Euclidean in world
units, both near (saturating at 2,000, so the 200/450/700 raze reaches and 525
attack range keep resolution) and logarithmic; health and mana carry a fraction,
an absolute value and a log; cooldowns a near (10 s) and a log value; categories
(unit kind, action kind, point source, ability and item ids, aim, slots) are
one-hot. Derived inputs, all from the seat's own messages and the public rules
(`src/feature/combat.rs`):

- per unit, from the own hero's side: its hit and hits to kill both ways after
  armor, range margins, whether the hero stands in its acquisition range, last hit
  or deny now, the next raze's damage (magic resist and held stacks) and razes to
  kill, whether each raze reach strikes it along the current facing, and whether
  it attacked the own hero or side in the last 2 s;
- per hero (own or enemy, visible or remembered): raze and Requiem cooldowns and
  levels, whether it can pay a raze, and for a hostile hero whether each of its
  razes would strike the own hero along its facing;
- global: seconds to kill the visible enemy hero and to be killed (ready razes
  stacking, then attacks), creeps acquiring or attacking the own hero, creep
  balance near it, creeps under either tower, tower range and aggro, ready and
  affordable razes.

There is no recurrent state beyond this history.

## Actions

The action is an autoregressive tuple (`src/action.rs`, schema 7):

```text
kind (16) -> controlled unit (hero, courier) -> ability/item/source slot
          -> target mode (None, Entity, Point) -> entity pointer (96) or point pointer (64)
```

Kinds, append-only: Continue, Stop, MovePoint, FollowUnit, Hold, AttackMovePoint,
AttackUnit, Cast, Use, PutPoint, PutUnit, Take, Buy, Sell, Swap, Learn. `Continue`
sends nothing and keeps the current order. `ActionSpace` builds legality masks before
sampling (ownership, visibility, range, mana, cooldown, charges, inventory, shop
range, gold, skill points, channel, courier errand); only legal choices are sampled.

Point candidates (at most 64): first up to 48 general ones — 8 directions at
200/600/1,200, building landing points, the 8 nearest trees, fountains and towers,
predicted hero and creep positions — then up to 16 raze-only ones (`src/action/raze_points.rs`)
while the own hero lives: straight ahead along its facing, per raze reach the landing
that strikes the most visible hostile units (then heroes; the middle of the widest
run of 128 scanned headings), the last-seen and extrapolated positions of up to two
enemy heroes out of sight for at most 150 ticks, and 8 blind headings halfway between
the tactical directions. Raze-only points are legal only as raze targets, so movement,
items and Teacher see exactly the general candidates. Every point token carries, per
raze reach, the visible hostile non-hero units and enemy heroes the landing along the
heading toward it would strike, and fog guesses carry their sighting age.

### Aimed razes

A Shadowraze has no target on the wire: it lands 200/450/700 units along the
caster's facing with radius 250, strikes only hostile units its side sees at that
tick, and bota has no order that only turns. The policy picks the target mode of a
ready raze:

- **None** fires along the current facing at once.
- **Entity** tracks a visible enemy or neutral within the raze's reach ± 250:
  `RazeAim` (`src/raze_aim.rs`) walks 48 units toward its predicted position and casts
  once the landing predicted for the next tick covers it.
- **Point** takes any candidate but the caster's own position as a heading (the reach
  stays fixed): the macro fixes the heading toward the point at the decision, walks
  32 units along it (one walk node; longer walks route through node centres and
  leave the facing off the heading) and casts once the landing is within 25 units
  of the landing along the heading. This expresses area razes at cluster landings and
  blind razes at fog guesses; a blind raze strikes only if the hero is back in sight
  when it lands.

Continue advances the macro; any other decision replaces it; it aborts after 15 ticks
or when the raze is not ready (and for Entity when the target is lost or leaves the
reach window). Teacher razes only as Entity. Episode logs count each seat's raze
decisions per mode (`raze_mode_*`) and razes that struck any hostile unit
(`raze_hits`) besides enemy heroes (`raze_hero_hits`).

## Network

`PolicyModel` (`src/model.rs`, schema 27) has 2,004,663 f32 parameters in 88 named
tensors:

- Unit encoder 109 → 64 → 128 → 128, shared by all unit tokens; token encoders
  → 64 → 64 per token family; pooled per group (units by kind; items as own and
  shop rows, then the enemy bag) into the trunk input (2,812).
- Trunk 2,812 → 512 → 256 → 256.
- Value head 256 → 256 (ReLU) → 1, its read-out multiplied by a fixed 16 (a
  critic-only learning-rate multiplier under Adam). The value loss trains the shared
  trunk and encoders too.
- Kind head from the trunk; conditional heads (unit, ability, item, swap, learn,
  shop, loot, target mode, put mode, entity and point pointer queries) read a
  336-wide context: trunk + kind (32) + unit (32) + slot (16) embeddings. Pointer
  queries are dot products with the encoded candidates.

### Side actors

Radiant and Dire have independent actor heads: the kind head and the eleven
conditional linears are duplicated (24 `dire.*` tensors after the 64 shared ones);
encoder, trunk, value head and embeddings are shared. Routing reads the observed
`SIDE_RADIANT`/`SIDE_DIRE` global features; anything but exactly one set fails
before evaluation. Both branches run on the full batch and a per-row select picks
the side, so gradients from one side's rows reach only its own heads plus the shared
parameters.

CUDA contract test (ignored by default):
`cargo test --release --features builtin,cuda model::side_actor_tests::cuda_side_actors_preserve_routing_gradients_and_ppo_rollback -- --exact --ignored`.

## Runtime weights and neural play

Training writes `drysua.weights.safetensors` (one named tensor per parameter plus
schema metadata) into the checkpoint directory and each `history/u<update>/`. Play it with:

```sh
cargo run --release --bin drysua -- play --policy neural \
  --weights-directory /absolute/path/to/weights --addr 127.0.0.1:4455
```

Neural play has no Teacher fallback: the model chooses every action, including
pregame buys and skill points. Weights whose action, feature, model, PPO or reward
identity differs from this build fail to load before connecting; play such weights
from the commit that produced them. `play`, `eval` and frozen training opponents all
use this strict loader (metadata, names and shapes). `--initial-weights` accepts other
linked schemas: it reuses each tensor with the same name and shape, keeps the seeded
initialization for the rest and logs both in `event=initial_weights_loaded`.

The zero-argument default (`src/default_deployment.rs`) is Teacher. A neural default
must name a directory below `artifacts/`.
