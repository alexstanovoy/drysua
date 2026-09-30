# Training and evaluation

`scripts/train.py` runs a training campaign as a sequence of **sessions**. Each
session is one rootless Docker container running one long-lived
`drysua train-annealed` process that trains every remaining update, so CUDA JIT and
container start are paid once per session (the JIT cache persists across sessions).
The controller is standard-library Python and never decodes checkpoint bytes:
progress is read only through the frozen native `checkpoint-inspect`
([checkpoint inspection](checkpoint_inspection.md)). Collection itself is described
in [continuous collection](continuous_collection.md).

## Commands

```sh
python3 scripts/train.py create --config docs/training_controller.example.json temp/my-campaign
python3 scripts/train.py run temp/my-campaign            # foreground session
python3 scripts/train.py run temp/my-campaign --detach   # returns once the container runs
python3 scripts/train.py status temp/my-campaign [--inspect]
python3 scripts/train.py pause temp/my-campaign          # commit the in-flight update, then stop
python3 scripts/train.py stop temp/my-campaign           # kill now; keep the last committed update
python3 scripts/train.py resume temp/my-campaign [--detach]
python3 scripts/train.py recover temp/my-campaign --confirm-offline
python3 scripts/train.py report temp/my-campaign [--json] [--block N]
python3 scripts/train.py report temp/my-campaign --html temp/my-campaign.html [--refresh 60 --follow]
```

Phases: `prepared` -> `running` -> `paused` | `failed` | `completed`. `run` starts a
prepared campaign, `resume` continues a paused or failed one. Nothing retries
automatically.

## Sessions

`run`/`resume` hold the campaign's `owner.lock` and every `lock_paths` file
(exclusive, nonblocking; a held lock is a refusal) for the whole session, then:

1. **Preflight.** Frozen inputs match the manifest, the running controller sources
   equal the frozen snapshot, the inspector advertises its contract, the committed
   checkpoint matches `status.json`, the daemon is rootless with cgroup v2, the
   pinned image is present locally (never pulled), and the host has
   MemAvailable ≥ 24 GiB, free disk ≥ 16 GiB, CPU ≤ 90 °C; in GPU mode the pinned
   GPU UUID has ≥ 4 GiB free VRAM and ≤ 85 °C.
2. **Container.** Read-only root, no network, all capabilities dropped,
   `no-new-privileges`, private cgroup namespace, `memory_gib` hard limit with zero
   swap, 1024 PIDs, no CPU quota, 16 MiB json-file log. The trainer runs under
   `docker-init` with a controller-built argv. Host `/usr` and the linker cache are
   mounted read-only for the CUDA driver ABI; `bin/` and `inputs/` are read-only;
   only `checkpoint/`, `history/` and `cuda-cache/` are writable. In GPU mode only the
   pinned GPU's device node is passed.
3. **Limits.** After start the controller reads the container's actual cgroup files
   (`memory.max`, `memory.swap.max`, `pids.max`, `cpu.max`) and the UID mapping; a
   mismatch kills the container and fails the session.
4. **Monitoring.** Output streams into `sessions/NNNN/payload.log` (at most 1 GiB,
   then a pause). Every 5 s cgroup and host health go to `resources.jsonl` (at most
   16 MiB). A cgroup limit event, MemAvailable < 16 GiB, free disk < 4 GiB,
   CPU > 90 °C, GPU > 85 °C or free VRAM < 4 GiB requests a pause with that reason.
5. **Exit.** The container is removed, the committed checkpoint is inspected, and
   `sessions/NNNN/receipt.json` records start/end updates, stop request and reason,
   exit code, verified limits and an inspection summary.

A limit violation is not a reason to raise the limit and retry: read the reason and
shrink the job.

## Stopping and crash safety

The trainer commits a checkpoint (model, Adam, RNG, in-flight slot games and
progress; the manifest is written last with file and directory fsync) only every
`checkpoint_seconds` of wall time, on a graceful stop and at the last update of the
invocation, to keep SSD writes low. A crash loses up to one interval of updates,
which the resume recomputes bit-exactly. `status` `updates` is the last durable
checkpoint. The trainer logs `progress: update N, ...` (the statistics) after every
update and `checkpoint: update N` after every durable commit.

- `pause`, the session deadline (`max_seconds`), a health violation, or the first
  SIGINT/SIGTERM to the controller send SIGTERM to the trainer. It finishes the
  in-flight update, checkpoints it, logs `event=training_stopped` and exits 0; after
  `stop_seconds` it is killed.
- `stop` or a second controller signal kills the container at once.
- A trainer crash fails the session; the committed checkpoint is still verified
  and accepted.
- If the controller dies, `run`/`resume` refuse until `recover --confirm-offline`,
  which requires the recorded controller to be dead, kills and removes the recorded
  container (matched by full ID and labels), accepts the committed checkpoint and
  leaves the campaign paused.
- A kill during a checkpoint save leaves the previous commit selected, possibly
  with the newer runtime export (`recovery_required`); the next commit replaces it.

`pause`/`stop` write an owner-token-scoped request; they fail when no live
controller owns the campaign.

## Configuration (schema 2)

| Field | Default | Contract |
| --- | --- | --- |
| `schema` | required | `2` |
| `trainer` | required | `drysua` ELF built with `builtin` (and `cuda` for GPU) |
| `inspector` | `trainer` | binary providing `checkpoint-inspect` |
| `initial_weights` | none | runtime weights with the current parameter layout; fresh start only |
| `opponent_weights` | none | frozen weights opponent (`--opponent weights:...:1`); Teacher otherwise |
| `total_updates` | required | 1..10000 |
| `history_every` | 20 | milestone spacing for `history/uNNNN/` runtime weights; exported only at checkpoints |
| `checkpoint_seconds` | 600 | 60..86400, wall time between checkpoints (`--checkpoint-interval-seconds`) |
| `max_seconds` | 86400 | 1..604800, session deadline, then a graceful pause |
| `stop_seconds` | 300 | 5..3600, graceful stop budget (one update plus a checkpoint) before a kill |
| `memory_gib` | 12 | 1..48, container memory limit, swap 0 |
| `training_args` | `[]` | allowlisted `train-annealed` options only |
| `mode` | `cpu` | `cpu` or `gpu` |
| `gpu_uuid` | null | full `GPU-...` UUID in gpu mode |
| `image` | required | `name@sha256:<digest>`, present locally |
| `docker_context` | `rootless` | Docker CLI context |
| `cuda_directory` | `/usr/local/cuda-13.3` | directory directly under `/usr/local` |
| `lock_paths` | `[<repository>/heavy.lock]` | 0..8 existing files held exclusively per session |

Unknown fields, duplicate keys, nonfinite numbers and non-allowlisted trainer flags
are rejected. The controller owns `--updates`, `--checkpoint-directory`,
`--history-directory`, `--history-every`, `--checkpoint-interval-seconds`, `--device`, `--device-ordinal`,
`--resume`, `--initial-weights` and `--opponent`. At a checkpoint of update N the
trainer exports `history/uNNNN` when a multiple of `history_every` was crossed since
the previous checkpoint, and always at the final update, so directory names need not
be multiples of `history_every`. Without `--seed` the first session
draws a random seed, recorded in the run scope and adopted by later sessions.

A campaign directory (mode 0700) holds `manifest.json`, `status.json`,
`owner.lock`, `owner.json` while a session is open, read-only `bin/` and `inputs/`,
`frozen/` (the controller sources that created it; `--detach` runs this snapshot and
a foreground run refuses changed sources), `checkpoint/`, `history/`, `cuda-cache/`
(1 GiB) and `sessions/NNNN/`.

## Environment schedule

`train-annealed` randomizes spawn modifiers (11 variables in `VARIABLES`,
`src/randomization.rs`) per environment generation; the last `--zero-updates`
(default one fifth of `--updates`, rounded up) are unmodified. `--generation-updates`
(required) is the base length of a generation.

The default `--environment-schedule adaptive` ends or extends a generation by the
per-update win rate (terminal wins over games finished during the update; an update
with no finished game breaks both streaks):

| Flag | Default | Effect |
| --- | --- | --- |
| `--environment-success-updates N` | 2 | N consecutive updates each with win rate ≥ P start a new environment |
| `--environment-success-rate P` | 0.8 | |
| `--environment-poor-updates M` | 1 | M consecutive updates each with win rate ≤ Q award X extra updates |
| `--environment-poor-rate Q` | 0.2 | |
| `--environment-extension X` | 0.75 | credit accumulates exactly; the generation lasts `base + floor(credit)` |

Success wins if both thresholds match. A transition resets both streaks and the
credit; `--updates` stays a hard budget and the clean tail always starts on time.
Rates are exact decimals with at most six fractional digits (no sign, exponent or
whitespace). Collection runs one update ahead of the learner, so a transition decided
after update `u` applies to games that start while `u + 2` is collected.
`--environment-schedule fixed` starts a generation every `--generation-updates`
updates and rejects the adaptive flags.

`--environment-scale-start` (default 1) and `--environment-scale-end` (default 0),
each `0..=10` times full variance, set the modifier spread: `scale_bp = (start ·
(10000 − root) + end · root) / 10000` with `root = isqrt(update · 10⁸ / (updates −
zero_updates))`, truncated to basis points. The clean tail is always zero. Sampled
deltas still clamp to each variable's range.

## Run scope and resume

The checkpoint records the run scope: seed, schedule, opponents, collection and
optimizer settings, execution modes and the binary's commit hashes. A resume must
repeat them; any difference is rejected before training state or generation
history changes. `--seed` is optional: a fresh run draws one from `/dev/urandom`
and prints it, a resume adopts the recorded one. `--simulation-threads` is not in
the scope and never changes results. Adaptive generation snapshots
(`adaptive-v3`) extend a SHA-256 chain committed by the checkpoint and are
verified before collection.

## Reports and dashboard

`report` is read-only and accepts a campaign directory, a directory of logs or one
log file, gzip-compressed or not (e.g. `artifacts/history/<campaign>/payload.log.gz`).
It parses:

- `episode: key=value ...` as one finished episode;
- `level=<LEVEL> event=<name> key=value ...` as an event (`scope=` becomes part of the
  series name, `*_ns` fields are shown in seconds), e.g. `ppo_update`,
  `episode_summary`, `map2_episode_reward`;
- `progress: update N, ...` as the statistics of update N (shown as
  `checkpoint.<field>` series) and `checkpoint: update N` as its durability marker. A
  process restart (`### ` separator, new file, `annealed: updates=` header) drops the
  unattributed tail and every update after the last durable checkpoint, which the
  trainer replays and logs again.

The text report shows W-L-D blocks, recent windows, per-opponent records,
environment transitions and timing with an ETA; `--json` prints schema
`drysua-training-report/v2`. `--html FILE` writes one self-contained page (no
network): outcomes per update, rolling win rate with a Wilson 95% band per opponent,
action-kind shares, phase timing and samples/s, PPO statistics (losses, entropy,
KL, clip fraction, explained variance), environment generation, reward components,
and every other numeric field as a small multiple, so new log fields appear without
code changes. `--refresh S` adds an auto-reload; `--follow` regenerates the file
every `S` seconds while the campaign runs.

## Evaluation

`drysua eval` plays frozen runtime weights once per side of every seed, with no
optimizer, rollout or checkpoint:

```sh
drysua eval --weights artifacts/weights/u200v2-v18 --opponent teacher \
  --seeds 1000000:200 --device cuda --output temp/eval-u200v2.jsonl
python3 scripts/eval_compare.py temp/eval-u200v2.jsonl temp/eval-other.jsonl
```

`--opponent` is `teacher`, `harass-push` or `weights:<dir>`; `--greedy` takes the legal argmax
instead of sampling; `--parallel` (default 16) and `--actor-pipeline-groups`
(default 2) change only speed, never results. Each game's arena seed and actor RNG
depend only on `(seed, seat)`, so results are identical across batch shapes and
CPU/CUDA. The output (`drysua-eval/v1`, never overwritten) has one line per game
(outcome, end reason, ticks, kills, deaths, level, XP, weakest tower HP, casts per
ability, raze hero hits for both heroes) and a summary with a Wilson 95% interval,
side split and end reasons. `eval_compare.py` prints win rates and exact McNemar
tests on shared `(seed, side)` games of results that faced the same opponent.
Detecting +10 points at 35% needs about 370 games per arm; +5 points about 1,500.

## Machine rules

- Heavy commands (cargo builds and tests, Python suites, trainer and eval runs) pass
  `bench.gate` and run under the shared `bench.lock`:
  `flock bench.gate true && flock -s bench.lock <command>`. Benchmarks hold the gate
  and the exclusive lock (`flock bench.gate flock bench.lock <command>`) for at most
  about 20 minutes. Timings taken alongside training or other heavy jobs measure
  contention, not the change; do not benchmark while a campaign runs.
- Memory, PIDs, VRAM, temperature and disk limits for training are the controller's
  (above); do not run training outside it.
- SSD: keep scratch output under the worktree's gitignored `temp/`, delete finished
  runs and copied checkpoints, and do not add per-update files; weight history is
  written only at checkpoints that cross a `history_every` milestone.
- Never pass `-j1`, `--test-threads=1` or hard-coded thread counts; defaults derive
  from `available_parallelism`.

## Verification

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
```

`tests/test_train.py` drives the real controller against a fake Docker CLI whose
container is a fake trainer (fresh and resume checks, one commit per update,
graceful SIGTERM) and a fake inspector; only host cgroup and health probes are
replaced. `tests/test_train_report.py` parses fixture logs of every layout.
