# Map2 reward (version 8)

The objective is to win Map2, and to win early. Under Map2 rules a side loses when it
loses any tower or when its hero dies for the second time. A game that reaches the
native cap of 27,900 ticks is a draw. The reward is `Map2Reward` in `src/map2_reward.rs`.
Each seat computes it from its own `ServerMsg` stream: `MatchInfo`, the fogged
`WorldView`, visible `Events` and `MatchOver`. It never reads `World`.

## Definition

```text
r_t = Φ(s_t) − Φ(s_{t−1})                     every completed tick (γ = 1)
r_T = Φ(s_T) − Φ(s_{T−1}) − Φ(s_T) + R(end)   final tick: Φ(terminal) = 0

R(Win)  = +1 + 0.5 · min(1, (27900 − T) / 27000)
R(Loss) = −1
R(Draw) = R(TimeCap) = −0.5   (loss-ish: a stalemate must not be safe)

Φ = 0.75 · (own weakest tower HP fraction − enemy weakest tower HP fraction)
  + 0.40 · (enemy hero deaths − own hero deaths)          deaths capped at 2
  + 0.20 · (own hero HP fraction − enemy hero HP fraction)
  + 0.20 · clamp((own XP − enemy XP) / 1000, −1, 1)
```

- **Towers.** Buildings are always visible to both sides, so every tower is observed.
  A known tower that disappears has been destroyed and counts as 0.
- **Deaths.** These are public scoreboard deaths.
- **Hero HP.** Each hero's HP is its last seen HP fraction. A dead hero counts as 1,
  its full next life. A kill therefore turns the HP lead built by harassing into a
  death lead.
- **XP.** This is public cumulative scoreboard XP.

Constants are `MAP2_REWARD_*` in `src/map2_reward.rs`. `MAP2_REWARD_VERSION` is
recorded in checkpoints, runtime weights and reward reports.

## Rationale

1. **Only the outcome defines what is optimal.** Shaping is potential-based
   (Ng, Harada and Russell 1999) with γ = 1 and `Φ(terminal) = 0`, so over an
   episode the shaping terms sum to `−Φ(s_0) = 0` and every return equals `R(end)`.
   No dense term can be farmed; shaping only moves credit earlier than GAE's
   roughly 40-second horizon would.
2. **Φ tracks the decisive quantities.** It approximates `E[outcome | s]` in terminal
   units. The weights come from a least-squares fit of the outcome on these features
   (240 U200v2-vs-Teacher and 400 Teacher-mirror games), set conservatively: the
   weakest-tower lead predicts best (0.8–1.0), XP predicts in the Teacher matchup,
   deaths and HP in the mirror. By the rules one death is half a loss. Assertions
   keep `health < deaths`, `xp < deaths` and `|Φ| < 2`, so shaping never outweighs
   the gap between a win and a loss.
3. **Win fast.** The bonus falls linearly from +0.5 after pregame to 0 at the cap;
   winning five minutes earlier is worth about +0.17.
4. **Reward 7 was removed.** Its ten diminishing channels (gold, XP, damage, creeps,
   towers, mana) and its lane, pregame, position and stagnation terms were not
   potential-based, were net-negative in wins and losses, and charged for harassing
   and pushing. Its tower term averaged 11 towers, so the deciding tower paid
   +0.027. The raw measurements remain as diagnostic counters.

## Logs and reports

- Per episode, `level=INFO event=map2_episode_reward` carries:
  - `reward_version`, `reward_ticks` and `reward_total`;
  - `reward_<component>` for `towers`, `deaths`, `health`, `xp`, `closure`,
    `terminal` and `fast_win`;
  - the counters `own_deaths`, `enemy_deaths`, `own_xp_gained`, `enemy_xp_gained`,
    `hero_damage_dealt`, `hero_damage_taken`, `tower_damage_taken`,
    `other_damage_taken` and `structure_damage_dealt`.
- `reward_towers + reward_deaths + reward_health + reward_xp` is the potential
  progress made before the end. It is positive when the seat was ahead.
  `reward_closure` returns it.
- `event=map2_training_reward` carries the same fields summed per checkpoint or
  invocation.
- `drysua reward-observer` (`play.sh --reward-report`) writes the same components and
  counters for each seat of a human game ([human reward play](human-reward-play.md)).

## Reward 9: learned potential (`--potential learned`)

Reward 9 keeps the terminal ±1 and the fast-win bonus and replaces the hand
potential with `Φ9(s) = P(win | s) − ½` (`src/ppo_arena/win_model.rs`). P is a
logistic model over 17 features of both seats — the clock; the XP, level, last-hit,
deny, bounty-gold and hand-potential leads; own and enemy deaths, weakest-tower HP,
hero HP, mana and respawn — plus each lead times the clock. It may read what the
policy seat cannot see because it only shapes training games: nothing reaches the
policy's input, the runtime weights, `play` or evaluation.

- Each finished training game contributes a sample every 30 s of game clock and its
  score (win 1, draw or cap ½, loss 0; unchanged by the draw reward). Every `--win-model-every` updates (10) the
  learner refits on the last `--win-model-games` games (1024) with the seat-swapped
  copy of every sample, 12 ridge-regularized Newton steps in f64: a pure function of
  the run.
- Games keep the model version they started with, so their shaping telescopes to
  `Φ9(end) − Φ9(start)` with `Φ9 = 0` at the end. A refit after update `u` shapes the
  games starting in update `u + 1` on. Until the window holds
  `--win-model-min-games` games (256) new games use the hand potential.
- The window, the live model versions and every in-flight game's version are part
  of the collection checkpoint, so resume stays bit-exact.
- `reward_learned` is the learned shaping; the hand components are zero in a
  learned game. Each refit logs `event=win_model` with the held-out AUC of the
  learned and the hand potential by game minute on the games since the previous
  refit; the dashboard charts both.

## Policy inputs

The global `MAP2_*` features carry the potential's inputs (own and enemy
weakest-tower HP, deaths over 2, last-seen hero HP, the clamped XP lead) and Φ
itself, so a linear critic can subtract the shaping.

## Offline check

Builtin games scored per seat by `Map2Reward` (harness not committed).

**U200v2 (greedy) against Teacher**, 240 games, 50 W / 189 L / 1 D. Per-episode
means ± SD of the neural seat; "dense" is `towers + deaths + health + xp` before
closure (reward 7's shaping was −0.065 in wins and −0.105 in losses):

| Component | Win | Loss |
|---|---:|---:|
| towers | +0.43 ± 0.22 | −0.56 ± 0.21 |
| deaths | −0.10 ± 0.22 | −0.12 ± 0.28 |
| health | +0.05 ± 0.07 | −0.02 ± 0.07 |
| xp | +0.12 ± 0.10 | −0.13 ± 0.07 |
| dense | **+0.51 ± 0.33** | **−0.82 ± 0.31** |
| terminal + fast_win | +1.23 | −1.00 |

AUC of Φ for predicting a win: 0.50 at 1–2 minutes (greedy openings are identical),
0.60 at 3, 0.68 at 5, 0.73 at 6.

**Teacher mirror**, 400 games, Radiant won 330. Dense progress +0.36 in wins and
−0.35 in losses, carried by deaths; AUC of Φ 0.83 at 1–2 minutes.

The deaths term does not separate the U200v2 matchup: that policy dies about as often
in wins as in losses and is decided by the tower. It stays because the rules make
deaths decisive; revisit once policies kill reliably.
