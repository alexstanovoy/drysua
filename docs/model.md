# Model and actions

## Observations

`FeatureEncoder` (`src/feature.rs`, schema 23) turns one `StateTracker` view into a
fixed-size `FeatureFrame`. Coordinates are team-canonical; every value is finite,
bounded and has an explicit presence flag where it can be unknown.

| Part | Size |
| --- | --- |
| Global | 92 features; 73–80 are the reward potential inputs ([reward](reward.md#policy-inputs)), 81–91 reserved zeros |
| History | 7 global samples × 24 at ages 480, 240, 120, 60, 30, 15, 0 ticks; 16 policy-history samples × 4 |
| Map | 96 |
| Units | 96 current + 32 remembered + 2 own tokens × 84 |
| Tokens | abilities 24, items 28, point candidates 32 (48 tokens), projectiles 20 (32), loot 16 (16) |

Enemy ability cooldowns are not encoded, and there is no recurrent state beyond this
history.

## Actions

The action is an autoregressive tuple (`src/action.rs`, schema 6):

```text
kind (16) -> controlled unit (hero, courier) -> ability/item/source slot
          -> target mode (None, Entity, Point) -> entity pointer (96) or point pointer (48)
```

Kinds, append-only: Continue, Stop, MovePoint, FollowUnit, Hold, AttackMovePoint,
AttackUnit, Cast, Use, PutPoint, PutUnit, Take, Buy, Sell, Swap, Learn. `Continue`
sends nothing and keeps the current order. `ActionSpace` builds legality masks before
sampling (ownership, visibility, range, mana, cooldown, charges, inventory, shop
range, gold, skill points, channel, courier errand); only legal choices are sampled.

Point candidates: 8 directions at 200/600/1,200, fountains, towers, predicted hero
and creep positions, static and planted trees, and building landing points.

### Aimed razes

A Shadowraze has no target on the wire: it lands 200/450/700 units along the
caster's facing with radius 250, and bota has no order that only turns. A raze is
therefore legal only as `Cast` with an entity target: a visible enemy or neutral
whose distance is within the raze's reach ± 250 while the ability is ready.
`RazeAim` (`src/raze_aim.rs`) expands that one decision into a macro: it walks
48 units toward the target's predicted position to turn, then casts once the landing
predicted for the next tick covers it. Continue advances the macro; any other
decision replaces it; it aborts after 15 ticks or when the target is lost, out of
the reach window or the raze is not ready. Teacher uses the same macro.

## Network

`PolicyModel` (`src/model.rs`, schema 25) has 1,812,983 f32 parameters in 86 named
tensors:

- Unit encoder 84 → 64 → 128 → 128, shared by all unit tokens; token encoders
  → 64 → 64 per token family; pooled per group into the trunk input (2,596).
- Trunk 2,596 → 512 → 256 → 256.
- Value head `Linear(256, 1)`. PPO feeds it a detached trunk, so value loss trains
  only this head.
- Kind head from the trunk; conditional heads (unit, ability, item, swap, learn,
  shop, loot, target mode, put mode, entity and point pointer queries) read a
  336-wide context: trunk + kind (32) + unit (32) + slot (16) embeddings. Pointer
  queries are dot products with the encoded candidates.

### Side actors

Radiant and Dire have independent actor heads: the kind head and the eleven
conditional linears are duplicated (24 `dire.*` tensors after the 62 shared ones);
encoder, trunk, value head and embeddings are shared. Routing reads the observed
`SIDE_RADIANT`/`SIDE_DIRE` global features; anything but exactly one set fails
before evaluation. Both branches run on the full batch and a per-row select picks
the side, so gradients from one side's rows reach only its own heads plus the shared
parameters.

CUDA contract test (ignored by default):
`cargo test --release --features builtin,cuda model::side_actor_tests::cuda_side_actors_preserve_routing_gradients_and_ppo_rollback -- --exact --ignored`.

## Runtime weights and neural play

Training writes `drysua.weights.safetensors` (parameters plus schema metadata) into
the checkpoint directory and each `history/u<update>/`. Play it with:

```sh
cargo run --release --bin drysua -- play --policy neural \
  --weights-directory /absolute/path/to/weights --addr 127.0.0.1:4455
```

Neural play has no Teacher fallback: the model chooses every action, including
pregame buys and skill points. Weights whose action, feature, model, PPO or reward
identity differs from this build fail to load before connecting; play such weights
from the commit that produced them. `play`, `eval` and frozen training opponents all
use this strict loader; only `--initial-weights` accepts other linked schemas with
the same parameter layout.

The zero-argument default (`src/default_deployment.rs`) is Teacher. A neural default
must name a directory below `artifacts/`.
