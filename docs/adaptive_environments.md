# Annealed environment schedules

`train-annealed` defaults to `--environment-schedule adaptive`. The default does
not depend on `--resume`: resuming a legacy fixed-schedule checkpoint requires
**explicit `--environment-schedule fixed`**, together with the original training
and execution arguments, including values that used to be defaults. There is no
automatic fallback or migration from fixed to adaptive.
The runtime's strict run-scope comparison rejects incompatible schedules before
training state or generation history is changed.

## Fresh adaptive runs

For an already existing, empty checkpoint directory:

```sh
drysua train-annealed --updates 200 --games 40 --parallel 20 \
  --generation-games 160 --checkpoint-directory artifacts/my-new-run \
  --environment-schedule adaptive \
  --environment-success-updates 2 --environment-success-rate .8 \
  --environment-poor-updates 1 --environment-poor-rate .2 \
  --environment-extension .75
```

This is M40/B20 with a base budget of four updates per environment. The schedule,
all five environment tuning flags, `--games 40`, and `--parallel 20` may be omitted;
these are their defaults. `--generation-games` remains required, with no new default.

The five `train-annealed` execution defaults are:

| Flag | CLI default | Meaning |
| --- | --- | --- |
| `--games` | `40` | Games per PPO update (M). |
| `--parallel` | `20` | Worlds per actor group (B), independent of CPU count. |
| `--actor-pipeline-groups` | `2` | Actor groups per wave (G); the default wave has 40 live worlds. |
| `--training-microbatch` | `256` | Tensor rows per PPO gradient / candidate-KL pass, not the effective Adam minibatch. |
| `--reuse-actor-values` | `true` | Reuse next-actor bootstrap values. Bare `--reuse-actor-values` and `--reuse-actor-values=true` enable it; `--reuse-actor-values=false` disables it. |

These are CLI defaults, not changes to the library defaults. The learner backend
still defaults to **CPU**; add **`--device cuda`** to select the measured CUDA
profile. The execution defaults do not imply a backend switch or a measured CPU
speedup. PPO rewards and optimizer defaults, host-math workers (`1`), balanced
minibatches (`false`), and the adaptive schedule defaults below are unchanged.

There is no automatic CPU-count or greatest-common-divisor choice for `--parallel`.
When overriding dimensions, B must divide both game counts, and B times G must
divide M and be at most 64. For example, M40/B10/G4 is valid; reducing M to 8 while
leaving B20 is not. G2/G4 require a Teacher opponent; a weights opponent requires
explicit `--actor-pipeline-groups 1`. Wide profiles above M40 require explicit
`--training-microbatch 64` under the unchanged memory admission limit. See
[training microbatches](training_microbatch.md) for numerical scope and library defaults.

The adaptive environment tuning defaults remain:

| Flag | Default | Meaning / bounds |
| --- | --- | --- |
| `--environment-success-updates N` | `2` | Consecutive successful updates; `1..=MAX_TRAINING_COUNTER`. |
| `--environment-success-rate P` | `0.8` | Inclusive per-update win-rate threshold, `[0, 1]`. |
| `--environment-poor-updates M` | `1` | Consecutive poor updates before extension awards; `1..=MAX_TRAINING_COUNTER`. |
| `--environment-poor-rate Q` | `0.2` | Inclusive per-update poor win-rate threshold, `[0, 1]`. |
| `--environment-extension X` | `0.75` | Exact extra-update credit per award, `0..=MAX_TRAINING_COUNTER`. |

Adaptive generations start with `--generation-games / --games` updates. Both
dimensions must be positive, and generation games must be a **whole multiple**
of games per update. The base update budget may exceed the total run budget,
but must fit `MAX_TRAINING_COUNTER`. Total updates must be positive and fit the
same bound; `--zero-updates` may cover none or all of the run, but not exceed
total updates. Its existing default remains one fifth of total updates, rounded
up. Existing global game, sample, optimizer, and parallel-world bounds still
apply independently.

Rates and credit use exact millionths, never binary floating point. Accepted
forms include `.8`, `0.8`, and `0.800000`; all have the canonical representation
`0.8` in adaptive scope flags. Use unsigned plain decimal notation with at most
six fractional digits. Signs (including `+` and `-0`), exponents, whitespace,
`NaN`, infinity, and trailing decimal points are rejected. Extensions `1`,
`1.25`, and values larger than the initial generation budget are allowed: there
is no local cap at one or at the base budget. The full exact value, including
its fraction, must fit the global bound.

After each successful PPO update, the scheduler uses that update's terminal wins
divided by `--games`. Draws and task timeouts are non-wins. A partial collection
or failed PPO update cannot advance the persisted controller. Production requires
all games to have complete, unrejected native outcomes.

Each of the last `N` updates in the **current environment** must independently
meet `win_rate >= P`; their average is not used. With defaults, `.7, .9` does not
qualify, while `.8, .8` advances unconditionally, regardless of accumulated extra
budget. Incomplete windows do not trigger. Success takes priority if the success
and poor thresholds overlap.

Each of the last `M` updates must independently meet `win_rate <= Q` to award
`X` extra-update credit. Windows overlap: a continuing poor streak awards credit
after every update once its first complete window exists, including during extra
updates. Credit is accumulated exactly; only the accumulated value is floored
when calculating `base_updates + floor(credit)`. The budget is checked **after**
the possible extension. For base four and `X=.75`, poor/poor/mid/mid/mid lasts
exactly five updates: two awards produce1.5 credit, hence one whole extra update.
Ten `.1` awards similarly produce exactly one extra update.

Every environment transition resets both streaks and all local credit. The global
`--updates` remains a hard total budget, and at `updates - zero_updates` the
controller forces a fresh clean environment even when extension credit remains.
The clean tail therefore cannot be delayed by poor results. The final update does
not create an unused next environment. These controls never change reward shaping.

## Environment scale ramp

`train-annealed` controls how far the domain randomization spreads over the run
with two exact decimals:

```sh
# Default: full variance at the first update, zero across the clean tail.
drysua train-annealed --updates 200 --generation-games 32 ...
# Rise from a clean start to twice full variance.
drysua train-annealed --updates 200 --generation-games 32 \
  --environment-scale-start 0 --environment-scale-end 2 ...
# Hold one constant scale for the whole run (equal endpoints).
drysua train-annealed --updates 200 --generation-games 32 \
  --environment-scale-start 0.5 --environment-scale-end 0.5 ...
```

* `--environment-scale-start` defaults to `1` and `--environment-scale-end` to
  `0`; both accept `0..=10` (ten times full variance) written as plain unsigned
  decimals with at most six fractional digits. The loop works in basis points, so
  an endpoint truncates to a hundredth of full variance.
* The ramp is `scale_bp = (start * (10000 - root) + end * root) / 10000` with
  integer floor division and `root = isqrt(update * 10^8 / (updates - zero_updates))`.
  The default endpoints reproduce the historical `10000 - root` bit for bit.
* The clean tail (`updates - zero_updates .. updates`) is always zero whatever the
  endpoints are, and a zero window covering every update stays all zero.
* Only the ramp endpoints are configurable. The eleven randomized variable ranges
  and their clamps in `VARIABLES` are model constants: a scale above one widens
  the sampling sigmas, and every sampled delta still clamps to its variable's
  lower and upper bound, so the saturation is the variable range, not the scale.
* A non-default ramp is recorded in the run scope as
  `--environment-scale-start <decimal> --environment-scale-end <decimal>`, in that
  fixed order after the adaptive environment suffix. The default ramp records
  nothing, so existing checkpoints and their scope bytes are unchanged, and a
  resume compares the recorded ramp byte for byte. Generation snapshots are
  recomputed from the seed and the schedule, so a resume under a substituted ramp
  stops before the first game.
* The maintained controller accepts both flags through `training_args` in
  `scripts/train.py`.

## Run seed

`train-annealed` has no fixed seed default:

- `--seed <n>` is an explicit override and is recorded in the run scope and the
  checkpoint as `run_seed`.
- A **fresh** run without `--seed` draws an unpredictable `u64` when settings are
  built, prints the resolved value, and records it in the run scope. On Unix this
  reads eight bytes from `/dev/urandom`; the non-Unix fallback mixes the wall
  clock with the process id through splitmix64 and is best effort. No dependency
  is added either way.
- A **resume** without `--seed` adopts the recorded scope seed, so an interrupted
  run replays identical streams. An unreadable run scope is a clear error, never a
  silent fallback.
- A resume with an explicit `--seed` that differs from the recorded one still
  fails the run-scope check with the `--seed: recorded X, requested Y` message.

`train-full` is unchanged and keeps its fixed default seed.

## Legacy fixed resume

Retain all of the checkpoint's original arguments and add the fixed selector.
For a CPU run originally using M8/B2, one actor group, microbatch 64, no actor-value
reuse, one host-math worker, and unbalanced minibatches:

```sh
drysua train-annealed --updates 20 --games 8 --parallel 2 \
  --generation-games 12 --checkpoint-directory artifacts/my-legacy-run \
  --actor-pipeline-groups 1 --training-microbatch 64 \
  --reuse-actor-values=false --host-math-workers 1 --device cpu \
  --resume --environment-schedule fixed
```

Omit `--balanced-minibatches` only when it was originally disabled, as here. Keep
the original device, seed, zero-update budget, PPO options, opponent, and any
nondefault execution flags too. An old M8/B8 run needs explicit `--parallel 8`,
not the B2 in this example. `--resume` does not restore old CLI defaults: selecting
fixed alone does not restore M8/B8/G1/micro64/reuse=false. G1, micro64, and reuse=false
still have their historical scope encoding (no execution suffix for those values).

Fixed mode retains legacy generation boundaries on the global game counter.
For example, 12 generation games and 8 games per update are allowed when the
existing parallel divisor is valid (2 here). Fixed mode rejects **every
explicit adaptive tuning flag**, even if its value equals the adaptive default;
omit all five, rather than relying on them being ignored.

The fixed selector preserves historical command-scope encoding; adaptive
configuration is appended canonically by the runtime. Equivalent decimal
spellings therefore do not create distinct adaptive scopes. An omitted schedule
on a legacy resume still selects adaptive and must not silently continue that
fixed checkpoint. An adaptive checkpoint also requires its original execution
arguments on resume, with compatible build provenance; changing CLI defaults does
not migrate either kind of run.

## Checkpoint and generation recovery

Adaptive checkpoints use capacity-specific manifest identities15/16/17. Fixed
checkpoints retain their exact12/13/14 identities and byte layout. Runtime weights,
model/action/feature/PPO tensor contracts are unchanged. The adaptive fixed-size
metadata block stores typed configuration, controller counters and a committed
generation-prefix SHA-256 together with model parameters, Adam and RNG state.

Adaptive generation snapshots use `adaptive-v3`, recording actual `start_update`
and explicit `end_game_bound`/`applied_games_bound`, not a guessed fixed end. Each
canonical snapshot extends the rolling SHA-256 committed by the checkpoint.
Resume verifies strictly ordered starts and every byte of the committed prefix
before collection. Missing, altered or mismatched state is rejected, not repaired.
Later orphan snapshots from an uncommitted update are ignored during prefix
verification and must match exactly when that update is replayed.

Outcome fixtures used by short native regression tests are test-only, explicitly
bound into their test scope, and unavailable through the production CLI. Separate
full-native CUDA resume coverage uses real terminal outcomes without those fixtures.
