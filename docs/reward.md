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
R(Draw) = R(TimeCap) = 0

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
   (Ng, Harada and Russell 1999) with γ = 1, and the potential is returned at the
   terminal (`closure = −Φ`).
   - Over any episode the shaping terms sum to `−Φ(s_0)`, and `Φ(s_0) = 0` at a
     native start.
   - So every episode's return is exactly `R(end)`, and no dense term can be
     farmed.
   - Shaping only moves credit earlier. The terminal no longer has to reach a
     decision made minutes before through GAE's roughly 40-second λ horizon.
2. **The potential tracks the decisive quantities only.**
   - Φ approximates `E[outcome | s]` in the units of the terminal.
   - The four terms and their signs follow the Map2 rules.
   - The weights are set conservatively from a least-squares fit of the outcome on
     these features, using 240 offline U200v2-vs-Teacher games and 400 Teacher-mirror games
     (`temp/` harness, not committed).
   - The weakest-tower lead predicts the outcome best, with a coefficient of about
     0.8–1.0. The XP lead predicts it in the Teacher matchup (0.6–1.3), and deaths
     and HP predict it in the mirror.
   - In the rules, one death is half of a loss.
   - Two orderings are enforced by assertions: `health < deaths` (a kill is worth
     more than the damage that landed it) and `xp < deaths`.
   - `|Φ| < 2` always holds, so shaping never outweighs the gap between a win and a
     loss.
3. **Win fast.** The bonus falls linearly from +0.5 after pregame to 0 at the cap.
   Winning five minutes earlier is worth about +0.17, which equals about 8 points
   of win probability. The outcome still dominates, since a win is worth at least
   2 more than a loss.
4. **What was removed and why.** Reward 7's ten diminishing channels are gone:
   gold, XP, hero damage, damage taken from heroes, creeps, towers and other sources,
   and mana. So are its lane, pregame, opening-position, fountain-wait and stagnation
   terms. They were not potential-based, so they changed the objective. They were
   net-negative in both wins and losses, and they charged for harassing and for
   pushing, which are the two ways to win. The tower potential averaged over 11
   towers, so the tower that decides the game paid only +0.027. The raw measurements
   survive as diagnostics.

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
  counters for each seat of a human-vs-bot game. See
  [human-reward-play.md](human-reward-play.md).

## Policy inputs

Global features 73–80 carry the potential's inputs:
- own and enemy weakest-tower HP;
- deaths divided by 2;
- own and enemy last-seen hero HP;
- the clamped XP lead;
- Φ itself.

A linear critic can then subtract the shaping. Features 81–91 are reserved zeros,
which keeps the model width.

## Offline sanity check

Games were played with the uncommitted harness in the worktree's `temp/`. The harness
runs builtin arena games, and each seat is scored by its own `Map2Reward`.

**Neural against Teacher.** There were 240 games of greedy U200v2 against Teacher,
with seats alternating. The neural seat went 50 W, 189 L and 1 D. Its reward-state
inputs are reinterpreted by version 8, so the win rate is below the live 35%.

Per-episode returns of the neural seat (mean ± SD). "Dense" is the sum of
`towers`, `deaths`, `health` and `xp` before closure. For comparison, reward 7's
shaping measured −0.065 in wins and −0.105 in losses.

| Component | Win | Loss |
|---|---:|---:|
| towers | +0.43 ± 0.22 | −0.56 ± 0.21 |
| deaths | −0.10 ± 0.22 | −0.12 ± 0.28 |
| health | +0.05 ± 0.07 | −0.02 ± 0.07 |
| xp | +0.12 ± 0.10 | −0.13 ± 0.07 |
| dense | **+0.51 ± 0.33** | **−0.82 ± 0.31** |
| terminal + fast_win | +1.23 | −1.00 |

The AUC of Φ for predicting a win depends on game time:
- **0.50 at 1–2 minutes.** Greedy openings are identical across seeds.
- **0.60 at 3 minutes.**
- **0.68 at 5 minutes.**
- **0.73 at 6 minutes.**

**Teacher mirror.** There were 400 Teacher-against-Teacher games, and Radiant won
330 of them. Mean dense progress was +0.36 in wins and −0.35 in losses, carried by
`deaths` (±0.37), since the mirror is decided by kills. The AUC of Φ was 0.83 at
1 and 2 minutes.

**The deaths term does not separate the U200v2 matchup.** That policy dies about as
often in its wins as in its losses, and it wins or loses by the tower. The deaths
term is kept because the rules make it decisive. It is a candidate to revisit once
policies can kill reliably.
