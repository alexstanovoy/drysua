# Training campaigns

`scripts/train.py` runs a training campaign as a sequence of **sessions**. Each
session is one rootless Docker container running one long-lived
`drysua train-annealed` process that trains every remaining update. CUDA JIT and
container start are paid once per session (the JIT cache even persists across
sessions), not once per update. The controller is standard-library Python and
never decodes checkpoint bytes: progress is accepted only through the frozen
native `checkpoint-inspect`.

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

Phases: `prepared` -> `running` -> `paused` | `failed` | `completed`. `run`
starts a prepared campaign, `resume` continues a paused or failed one. Nothing
retries automatically: a failed session stays failed until someone resumes it.

## Sessions

`run`/`resume` hold the campaign's `owner.lock` and every configured
`lock_paths` file (exclusive, nonblocking) for the whole session, then:

1. Preflight: frozen inputs match the manifest, the running controller sources
   equal the frozen snapshot, the inspector advertises its contract, the
   committed checkpoint matches `status.json` (updates and scope), the daemon is
   rootless with cgroup v2, the pinned image is present locally (never pulled),
   and host admission holds (MemAvailable >= 24 GiB, free disk >= 16 GiB,
   CPU <= 90 C; in GPU mode the pinned UUID is visible with >= 4 GiB free VRAM
   and <= 85 C).
2. `docker create` with a read-only root, no network, all capabilities
   dropped, `no-new-privileges`, private cgroup namespace, `memory_gib` hard
   memory limit with zero swap, 1024 PIDs, no CPU quota, a 16 MiB bounded
   json-file log, and labels binding campaign and session. The trainer runs
   directly under `docker-init` (no shell) with a controller-built argv. Host
   `/usr` and the linker cache are mounted read-only for the host CUDA/driver
   ABI; `bin/` and `inputs/` are read-only; only `checkpoint/`, `history/`
   and `cuda-cache/` are writable. In GPU mode only the pinned GPU's device
   node is passed and `CUDA_VISIBLE_DEVICES` is its UUID.
3. The declared limits are checked with `docker inspect` before start. After
   start the controller reads the container's **actual** cgroup files from the
   host (`memory.max`, `memory.swap.max`, `pids.max`, `cpu.max`) and the
   rootless UID mapping; a mismatch kills the container and fails the session.
4. The container output streams into `sessions/NNNN/payload.log` (at most
   1 GiB; reaching it requests a pause). Every 5 s the controller records
   cgroup CPU/memory/PIDs and host health in `resources.jsonl` (at most
   16 MiB); a cgroup limit event, MemAvailable < 16 GiB, free disk < 4 GiB,
   CPU > 90 C, GPU > 85 C or free VRAM < 4 GiB requests a pause with that
   reason.
5. At exit the container is removed, the committed checkpoint is inspected,
   and one `sessions/NNNN/receipt.json` records start/end updates, the stop
   request and reason, the exit code, whether limits were verified, and a
   small inspection summary.

## Stopping and crash safety

The trainer checkpoints after **every** update (model, Adam, RNG and progress
in one manifest committed last, with file and directory fsync), so the
checkpoint on disk is always the last committed update.

* `pause`, the session deadline (`max_seconds`), a health violation, or the
  first SIGINT/SIGTERM to the controller send SIGTERM to the trainer. It
  finishes and commits the in-flight update, logs
  `event=training_stopped`, and exits 0. After `stop_seconds` the controller
  kills it.
* `stop` or a second controller signal kills the container at once; the
  in-flight update is lost, the last committed one is kept.
* A trainer crash marks the session failed with the exit code; the committed
  checkpoint is still verified and accepted.
* If the controller itself dies, the owner record stays behind and `run`/
  `resume` refuse. `recover --confirm-offline` requires the recorded
  controller to be dead, kills and removes the recorded container (only after
  matching its full ID and labels), accepts the committed checkpoint and leaves
  the campaign paused. Output written while no controller was attached is not
  in `payload.log`.
* Killing during a checkpoint save can leave the previous manifest copy
  selected (`recovery_required`) or a stale runtime export. The inspection is
  accepted, `last_error` says so, and the next resume repairs both.

`pause`/`stop` write an owner-token-scoped request that only the live owner
consumes; they fail when no live controller owns the campaign.

## Configuration, schema 2

| Field | Default | Contract |
| --- | --- | --- |
| `schema` | required | `2` |
| `trainer` | required | native ELF `drysua` built with `builtin` (and `cuda` for GPU) |
| `inspector` | `trainer` | binary providing `checkpoint-inspect` |
| `initial_weights` | none | runtime weights file or directory; fresh start only |
| `opponent_weights` | none | frozen runtime weights opponent (Teacher otherwise) |
| `total_updates` | required | 1..10000 |
| `history_every` | 20 | milestone spacing for `history/uNNNN/` runtime weights |
| `max_seconds` | 86400 | 1..604800, one session's deadline, then a graceful pause |
| `stop_seconds` | 300 | 5..3600, graceful stop budget before a kill |
| `memory_gib` | 12 | 1..48, container hard memory limit, swap 0 |
| `training_args` | `[]` | allowlisted `train-annealed` options only |
| `mode` | `cpu` | `cpu` or `gpu` |
| `gpu_uuid` | null | full `GPU-...` UUID in gpu mode |
| `image` | required | `name@sha256:<digest>`, present locally |
| `docker_context` | `rootless` | Docker CLI context |
| `cuda_directory` | `/usr/local/cuda-13.3` | directory directly under `/usr/local` |
| `lock_paths` | `[<repository>/heavy.lock]` | 0..8 existing files held exclusively per session |

Unknown fields, duplicate JSON keys, nonfinite numbers, booleans as integers
and non-allowlisted trainer flags are rejected. The controller owns
`--updates`, `--checkpoint-directory`, `--history-directory`,
`--history-every`, `--device`, `--device-ordinal`, `--resume`,
`--initial-weights` and `--opponent*`. Without `--seed` the first session draws
a random seed recorded in the run scope and later sessions adopt it.

A campaign directory (mode 0700) holds `manifest.json` (read-only config and
frozen-file inventory), `status.json`, `owner.lock`, `owner.json` while a
session is open, read-only `bin/trainer`, `bin/inspector`,
`inputs/*/drysua.weights.safetensors`, `frozen/` (the controller sources that
created it; `--detach` runs this snapshot, and a foreground run refuses changed
sources), `checkpoint/`, `history/uNNNN/drysua.weights.safetensors`,
`cuda-cache/` (`CUDA_CACHE_MAXSIZE` 1 GiB) and `sessions/NNNN/`.

## Reports and the dashboard

`report` is read-only and works on a campaign directory, the old per-invocation
layout (`invocations/*/payload.log`), a directory of logs, or one log file,
gzip-compressed or not, e.g. `artifacts/history/<campaign>/payload.log.gz`.
Lines are parsed generically:

* `episode: key=value ...` is one terminal episode (outcome, opponent, actions,
  and any numeric or low-cardinality string field);
* `level=<LEVEL> event=<name> key=value ...` is an event; a `scope=` value
  becomes part of the series name and `*_ns` fields are shown in seconds;
* `checkpoint: update N, key value, ...` commits update N: everything seen
  since the previous commit belongs to it. A process restart (`### ` separator,
  new file, `annealed: updates=` header) drops the uncommitted tail, just as the
  trainer does.

The text report shows W-L-D blocks, recent windows, per-opponent records,
per-update wins, environment transitions and timing with an ETA; `--json`
prints schema `drysua-training-report/v2`.

`--html FILE` writes one self-contained page (inline CSS, JS and data; no
network): win/draw/loss per update, rolling win rate with a Wilson 95% band
(per opponent when several appear), action-kind shares, update phase timing
and samples/s, episode length and every other episode field, PPO statistics
(losses, entropy, KL, and any `clip_fraction`/`explained_variance`/
`grad_norm` field of any event), environment generation, mean reward
components, and every remaining numeric series as a small multiple, so fields
added later appear without code changes. It follows the OS light/dark theme
with a toggle and has a table of block outcomes and latest values.
`--refresh S` adds an auto-reload; with `--follow` the command regenerates the
file every `S` seconds while the campaign is running (writes are atomic).

## Verification

```sh
python3 -m unittest discover -s tests -p 'test_train*.py'
```

`tests/test_train.py` drives the real controller in-process against a fake
Docker CLI whose container is a fake trainer (fresh/resume checks, one commit
per update, graceful SIGTERM) and a fake inspector; only the host cgroup and
health probes are replaced. `tests/test_train_report.py` parses fixture logs of
every layout. The real container path is qualified by a smoke run: a tiny
campaign paused and resumed in rootless Docker on the GPU.
