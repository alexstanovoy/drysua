# Local play: current Map2 Teacher or Neural integration

## Human vs Teacher with exact current reward

```sh
/home/alexstanovoy/Workspace/bots/play.sh --opponent teacher --reward-report \
  --human-side radiant --seed 9000001 --port 0 --no-build
```

Explicit Teacher needs no weights; Teacher plus `--weights-directory` is rejected.
Reward-report mode uses native **Lockstep paced at at most 30 Hz**, not coalescing
Realtime. The two passive observers reuse production Map2Reward and save per-seat
JSON plus bounded interval deltas under the announced play-run directory. A slower
renderer slows simulation. See [human-reward-play.md](human-reward-play.md) for
components, completeness flags, public raw paid-gold nets and verified native examples.
The GUI is for the human to launch from a desktop; agent verification was headless.

## Default Neural mode still fails closed without explicit weights

The current reward7/F22/M24 terminal and victory-time change and explicit migration are documented in
[reward-rebalance.md](reward-rebalance.md). Win has terminal reward+0.2, Draw0,
Loss and completed-task TimeCap-0.2. Dense terms are unchanged. Historical
M19/M20/M21 models are not current-compatible without explicit initialization.

Integration checks against the earlier bota rebase are recorded in
[`artifacts/temp/bota-rebase-integration-20260910/RESULTS.md`](../artifacts/temp/bota-rebase-integration-20260910/RESULTS.md).
The current wire rebase to bota `78427bb` (new `Missed` event, `Cheat` orders and
`NoCheats` rejection, attack-time/bound/collision view fields) is recorded in
[`artifacts/temp/bota-port-20260914/REPORT.md`](../artifacts/temp/bota-port-20260914/REPORT.md).
These are compatibility checks, not a trained-model release or win-rate result.

From an authorized Linux X11/Xwayland desktop terminal:

```sh
# Requires an externally supplied compatible model; none is selected by default.
/home/alexstanovoy/Workspace/bots/play.sh \
  --weights-directory "/absolute/path/to/compatible M24 runtime weights"
# Human Dire, pure Neural bot Radiant. Either side option derives the other.
/home/alexstanovoy/Workspace/bots/play.sh \
  --weights-directory "/absolute/path/to/compatible M24 runtime weights" \
  --human-side dire --port 0 --seed 9000001 --no-build
```

**No-argument play is not ready and fails closed.** No trained Map2 model has been
supplied or promoted for this launcher. The old F12/M14 human-review checkpoint
is incompatible with the current server/runtime. Supply `--weights-directory`
containing compatible **A5/F22/M24/PPO37/rules32/reward7** `drysua.weights.safetensors`; do not relabel or
copy old metadata to make a file pass. The launcher never creates, migrates,
initializes, trains, promotes, or substitutes weights.

With explicit weights, the default sides remain human Radiant / pure Neural bot
Dire. Both side options can be explicit; equal sides are rejected before startup.
With the default `--opponent neural`, the bot receives **`--policy neural`**, never
Hybrid or a Teacher fallback. Non-report play retains native Realtime.
The core binary's repository-selected default remains **Teacher**, unchanged in
`src/default_deployment.rs`; that is a separate CLI default, not launcher behavior.

The script works from other working directories and with spaces in the checkout
path. Relative weights directories resolve against the caller's working directory.
`--help` needs no display, weights, or children. Normal play checks weights first,
then `DISPLAY`, before building or creating logs: `WAYLAND_DISPLAY` alone cannot run
the X11 client. A nonempty `DISPLAY` does not establish X authorization or prove
OpenGL works. Python 3.10+ is required; Cargo/Rust is needed unless `--no-build`
is used. No packages are downloaded or installed by the Python launcher.

## Current weights compatibility preflight

`scripts/play_weights.py` performs standard-library, metadata-only preflight.
It opens the weights file read-only with no final symlink following and nonblocking
open (so a FIFO cannot hang startup), requires a regular file of at most **256 MiB**,
and reads only the eight-byte Safetensors length prefix and at most **64 KiB** of
header. Invalid/truncated JSON, duplicate keys, non-string metadata, missing or
extra metadata keys, and incompatible identities fail before children or logs.

Exactly nine metadata keys are required; numeric values are decimal strings:

| Key | Required value |
| --- | --- |
| `action_schema_hash` | `10658390830565586343` (A5) |
| `feature_schema_hash` | `10552563335950731440` (F22) |
| `model_schema_hash` | `12076707506725412686` (M24) |
| `ppo_schema_version` | `37` |
| `ppo_schema_hash` | `12793043235719775693` |
| `ppo_rules_audit_version` | `32` |
| `map2_reward_schema_version` | `7` |
| `map2_reward_schema_hash` | `7274660837025042530` |
| `map2_reward_schema_descriptor` | Full current descriptor matching that FNV-1a hash |

M24 retains A5 navigation/action legality, wait/refund and progress-debt rules.
Reward7 retains reward5 event/potential coefficients and one first-wave positioning
cost, not a perpetual location penalty. F22 preserves all92 global positions including
Tower remaining90/opening pending91: global92, unit84,62 named tensors,
1,700,020 F32 parameters. See the exact coefficient/bound table in reward-rebalance.md.

Inactive ticks, including death, add one debt tick up to2700. Useful activity
refreshes a30-tick lease including the current tick; active ticks repay3 debt.
At2700, a0.02 base cost latches until debt reaches zero; subsequent inactive
capped ticks cost0.000002. No stagnation refund is paid. A positive-timer own
effect3 qualifies directly even with full pools or lingering outside a fountain.
Any own purchase still fully refunds the open v2 fountain wait, but grants only
a partial generic activity lease, not a generic debt reset. No strategy mask,
Teacher override or extra anti-abuse condition is added.

M17/M18/M19/M20/M21 runtime and old checkpoint resumes are incompatible; metadata is not
silently relabelled. No M18 weights were generated or added to the source
whitelist. No-argument play still fails with the missing-model prompt, and
never substitutes root M17 weights or a Teacher.

Explicit M17-to-current-M24 **initialization only** is available through
`TrainingArtifact::initialize_selected_m17_for_map2_wait(directory, seed, device)`.
The additional exact M19/u162 parameter-only initializer is documented in
[reward-rebalance.md](reward-rebalance.md); it does not resume or relabel the old model.
Only these full-file SHA256 sources and their exact original nine-key reward1
metadata are accepted:

- Initial: `05e78663dd45ac23ad6c0242a69d8b8e45c163f7861c59154e3f3fd81de9ab1f`
- Recovery004: `107e19e61794c457ce8ec6adc2b3e66ccce08ee5b6544cb2005964f3e7dfedc8`

Seven positive-zero rows are inserted at row85 of `trunk.0.weight` (2589x512 to
2596x512); every source parameter bit is retained. This creates fresh model and
optimizer/progress ancestry, not runtime/resume compatibility, equivalent
gameplay, or a qualified release. The launcher never calls this API.

An explicitly selected M16 source may be used only through
`TrainingArtifact::initialize_selected_m16_for_map2_navigation(directory, seed, device)`.
The exact permitted full-file SHA-256 values are:

- Initial: `1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4`
- Advantage: `fdd4d3f2a85a64d9b4d162f1607a94a4aaba8a8e3136e51dab2152181cdf63b7`

The API also checks the complete old nine-key tuple, tensor contract and finite
values before constructing a model. Source reward1 metadata remains frozen;
the same five wait/progress-input rows are now zero-padded. All old parameter bits are
copied; no optimizer,
progress, RNG history, samples, league history or qualification is imported.
Retain its `INITIALIZATION_ONLY` provenance with `GAMEPLAY_EQUIVALENCE=false` and
create fresh training state. Do not replace historical metadata. The existing
explicit M14 padding initializer remains initialization-only into the current
contract, not a way to reproduce old gameplay under new masks.

The ignored, manual utility
`tests::navigation_initialization_utility::initialize_pinned_m16_navigation_artifact_only`
can write and reload a **new** runtime/checkpoint pair without training. It requires
`DRYSUA_SELECTED_M16_SOURCE`, `DRYSUA_NAVIGATION_INITIALIZATION_OUTPUT` (nonexistent,
outside the source directory), `DRYSUA_NAVIGATION_INITIALIZATION_SEED`,
`DRYSUA_INITIALIZATION_GIT_COMMIT`, and `DRYSUA_INITIALIZATION_SIMULATOR_COMMIT`.
Run it only under the [resource guard](experiment-safety.md), with all needed
environment assignments **inside** the guarded command. CUDA-enabled checks here
use `env NVCC_CCBIN=/usr/bin/g++-15 cargo ...`; no unsupported-compiler override.
The utility is not run by ordinary tests and is not used by the play launcher.

Tests recompute the schema hashes from the Rust descriptors and linked identities,
and check versions and the Rust loader's nine-key set. FNV is a compatibility
check, **not artifact authentication or qualification**. Preflight neither hashes
the tensor body nor proves tensor validity. The current Rust loader remains
authoritative for exact metadata/descriptor equality, tensor names, dimensions,
layout, dtype, and finite values **before the bot connects or sends Hello**.
A metadata-valid but invalid tensor file can therefore start build/server/client
children, then fail in the bot; cleanup still runs. Use a trusted weights directory
and do not replace files during launch.

## Build and launch contract

The canonical tracked entry point is `drysua/scripts/play.sh`; the workspace-root
`play.sh` remains a convenience shim that execs the same `play_match.py`. With
`root` the workspace directory containing both repositories, explicit weights select
`$root/drysua/target/release/drysua`, not a pinned historical executable.

By default the launcher builds **only missing** release binaries: no cargo
invocation runs when all three executables exist, and only the affected workspace is
built when one is missing or not executable. `--build` forces the two release
builds below before launching; `--no-build` never builds and checks **all three**
existing release executables after weights preflight and before logs. `--build` and
`--no-build` are mutually exclusive.

```sh
# bota workspace: only for --build or a missing bota binary
CARGO_TARGET_DIR="$root/bota/target" \
cargo build --release --locked --quiet \
  --manifest-path "$root/bota/Cargo.toml" \
  -p bota-server -p bota-client --bin bota-server --bin bota-client

# Working directory: "$root/drysua"; only for --build or a missing bot binary
CARGO_TARGET_DIR="$root/drysua/target" \
cargo build --release --locked --quiet --bin drysua --no-default-features
```

Each build has a 20-minute deadline. Inherited `CARGO_TARGET_DIR` does not redirect
build or launch paths. There are no debug, builtin simulator, CUDA, training, or
full-workspace bot builds. No historical frozen executable is overwritten. Cargo's
normal release build can need its existing registry/cache. An existing target path
alone does not prove freshness or compatibility: the default path deliberately
reuses existing binaries and never recompiles; pass `--build` after source changes,
or `--no-build` to forbid builds explicitly.

The protocol/slot mapping is for bota commit
`78427bb80eb716f851cb039ade33e2964bbf3c11`. Keep the server/client and current drysua
builds together. This includes current unit effects (Guarded 13, Inspired 14,
Shadowraze 15), the `Healed` event's separate health/mana fields and the `Missed`
event inserted after `Damaged`; the old wire, rules-audit30 weights and checkpoints
are not interchangeable with this protocol. Cheat orders exist in the schema but
drysua starts no match with cheats on and never issues one.
The bot command includes `--addr <bot-relay> --name drysua --policy neural
--weights-directory <absolute-current-weights>`.

The server **always** uses `--mode realtime --players 2 --map 2`, default port
`4455`, and seed `9000001`. Map2's native cap is **27,900 ticks**, including 900
pregame ticks and 15 minutes of gameplay at 30 Hz; simultaneous outcomes and the
time cap can produce a neutral draw. No launcher Map0 or legacy fallback exists.
Readiness requires the exact stdout listening line within ten
seconds, with a 4096-byte buffer. Port `0` is taken from that line, not guessed or
reserved with a race-prone bind-and-close. **No TCP health probe enters the lobby.**

## Side guarantee: accepted Welcome, not launch order

The server assigns the first free seat when it accepts a participant's `Hello`.
Launching the GUI first, accepting its TCP socket first, or sleeping does **not**
reserve Radiant. Upstream has no side CLI option, and the launcher changes neither
the public protocol nor game rules.

`scripts/play_admission.py` supplies two separate ephemeral **127.0.0.1** relays.
Both processes may start immediately, but the slot-1 relay does not even connect
to the game server until the slot-0 relay has validated the expected Player/Bot
`Hello` identity and a complete server `Welcome(slot=0, tick_rate=30, realtime)`.
The faster second participant can queue a connection/Hello only at its own relay.
Its actual `Welcome` must then contain slot 1. At the pinned server, slot 0 is
Radiant and slot 1 is Dire. Snapshot viewer teams are also checked against these
verified identities. Mismatches, duplicate handshakes, malformed framing, and
handshake timeouts stop the match, rather than reporting success on the wrong side.
The pinned server can broadcast a lobby update to an accepted but ungreeted
socket; that bounded update is forwarded but never opens the Welcome barrier.

Look for both console confirmations, for example:

```text
play: verified human Radiant: Welcome slot 0
play: verified bot Dire: Welcome slot 1
```

No hero/ready input is synthesized by the relay. In the GUI choose `1` (Sylla),
`2` (Pudge), or `3` (Shadow Fiend), then press `R` to ready. Sylla is initially
selected. Drysua selects Shadow Fiend and readies itself. Hero selection is not
part of the 30-second Hello/Welcome deadline.

All relay listeners and outbound connections are loopback-only. **The unmodified
bota server itself binds all IPv4 interfaces.** Use a trusted host/network or a
firewall; the protocol has no authentication. Direct outsiders can cause startup
to fail by taking a seat, but the launcher will not silently accept swapped sides.
Loopback is not isolation from hostile same-host processes.

## Bounded resources, shutdown, and diagnostics

The standard-library relays run in the supervisor thread, with nonblocking I/O,
backpressure, and no worker threads or child relay processes. Ready relay work is
serviced before waiting on child logs, with a nonblocking log poll when that work
progresses. Idle polling remains at 10 ms; no thread/event-loop rewrite is involved.
Bounds per relay:

- 30 seconds for Hello/Welcome, incomplete frames, or stalled writes; four hours
  for the whole local session, including an unattended lobby/results window.
- 64-byte handshake payloads; subsequent payloads at most 4 MiB, matching bota.
  Each reassembly/output buffer is at most 4 MiB + its four-byte length prefix.
- At most 1,000,000 frames in each direction, 2 GiB server traffic and 64 MiB
  participant traffic. Before terminal completion, messages are forwarded unchanged.

Ctrl+C exits `130`, TERM `143`, and HUP `129`, closing relay listeners/connections
and stopping only owned process groups. Closing the GUI also cleans up the
remaining match. Nonzero exits, startup errors, protocol/side failures, log
overflow, and `bota-client:` error diagnostics fail the run. Successful server/bot
exits after verified `MatchOver` leave the GUI results visible until it closes or
the session bound expires.
A disconnected participant that does not exit gets a two-second exit grace, then
the match fails and is cleaned up.

The relay validates the complete `MatchOver` payload (winner, integer bounds,
positive duration, both seat identities, and absence of trailing bytes). Only then
does it discard late client orders/ACKs that the finished server cannot accept.
Final server frames still drain to the GUI before its read side receives EOF;
the GUI keeps its write socket and results window alive. An upstream EPIPE/reset
while that final frame is still arriving stops further upstream writes and allows
at most 30 seconds to obtain verified `MatchOver`, not unconditional success.
Premature server EOF while the client is connected, premature receive resets,
corrupt/truncated frames, and failure to deliver the final frame to the client
remain errors. A server reset after a verified terminal frame cannot erase the
already buffered results.

Each child has a new session/process group. Cleanup sends TERM, allows up to two
seconds for leaders to exit, then KILLs all owned groups and reaps direct children.
Exited leaders remain unreaped until signalling finishes, preventing process-group
ID reuse. This also kills owned descendants of an already exited bot or compiler;
it does not signal existing servers or unrelated processes. SIGKILL of the
supervisor cannot be handled; descendants deliberately starting a different
session are outside process-group cleanup.

The printed private `drysua/artifacts/temp/play-*` directory retains combined
stdout/stderr child logs (`build-bota.log` and `build-drysua.log` when built,
`server.log`, `client.log`, `bot.log`) and `match.brp`. Each log is capped at
**16 MiB**; overflow stops the run.
The replay has a **2 GiB** `RLIMIT_FSIZE` cap, raised from the inadequate 512 MiB
limit. This permits approximately 75 KiB per tick over a 27,900-tick Map2 game,
before header overhead. It is a bounded local-play budget, not
a guarantee for every possible full-map trajectory: reaching it stops the server,
not disk growth. Interrupted/size-limited games can leave incomplete replays. No
full-length interactive replay size was measured by the headless launcher tests.
Run directories are retained for diagnosis and may be removed after use.

## Historical human-review archive and replay viewing

The old **corrected-ppo-001/u4 F12/M14/PPO27/rules22** archive is historical,
not current Map2 evidence. The never-called `review_paths` utility and its pinned
constants were removed from `scripts/play_match.py` on 2026-09-18; the archive and
its pins below remain evidence-only, and no launcher code references them. The
regression suite still hashes the archived review binary and weights directly when
its native smokes run. Archive locations and pins remain unchanged:

```text
drysua/artifacts/temp/human-review-20260909/current/
  drysua
  weights/drysua.weights.safetensors
  manifest.json
```

| Archived input | SHA-256 |
| --- | --- |
| Weights | `6348fe57a128ebd521dba68da0949ceb7378d3e0d6b7d6a7ce9a5fb6446547ab` |
| Frozen executable | `64ba25ebb10e6beabc26ff667a3e3bddeb40dbf391110d4f0e31db6478500a2b` |

Do not run the frozen bot against the current root server. The historical native
test requires the SHA-pinned historical server referenced by
`artifacts/temp/map0-baseline-observationfix-4096/baseline.json`; it skips if that
copy/pin is unavailable and never substitutes the root target server.

The two complete review replays use that archived Neural checkpoint. Their
original replay/viewer files are untouched by this integration. Run either command
from an authorized graphical terminal:

```sh
R=/home/alexstanovoy/Workspace/bots/drysua/artifacts/temp/human-review-20260909/replays
"$R/bota-client" --replay "$R/neural_vs_neural.brp"
"$R/bota-client" --replay "$R/teacher_radiant_vs_neural_dire.brp"
```

The bundled viewer was replaced with the streaming fix on 2026-09-09. Restart an
already open viewer to use it. Both files render a world, minimap, and HUD in the
actual Xwayland/OpenGL window, including with this bundled executable. Their BRP
bytes are unchanged. The original recording report/manifest remain historical;
`viewer_update.json` identifies the updated client, and `SHA256SUMS` verifies the
current bundle. Original viewer/checksums, source, test logs, and window captures
are retained in `artifacts/temp/human-review-20260909/viewer-fix/`.

Space pauses, period steps one tick while paused, +/- changes speed, the mouse
wheel zooms, and Escape closes. The initial -0:30 clock is the recorded pregame.

## Verification and GUI caveat

```sh
bash -n /home/alexstanovoy/Workspace/bots/drysua/scripts/play.sh
python3 -B -m unittest discover \
  -s /home/alexstanovoy/Workspace/bots/drysua/scripts -p test_play_match.py -q
```

The suite uses realistic framed-TCP mock clients/server and real subprocess groups
below `artifacts/temp`, with no builds or graphical window. It reproduces a faster
bot racing a delayed human Hello and a fragmented Welcome, checks both side
selections and explicit current Neural arguments, and tests failures, bounded I/O,
signals, descendants, and relay cleanup. No readiness probe is accepted by the mock server.
Socketpair/mock tests also cover fragmented terminal drain, queued/late orders,
EPIPE/reset handling, retained GUI write sockets, and selector waits without sleeps.

When the review copy and SHA-pinned historical server are available, archive smokes
run a real-protocol headless **Player** against the exact frozen Neural bot on both
sides: 30 ticks in realtime and 1000 ticks in lockstep. Both participants' Welcome
and snapshot identities are checked, as are zero rejected bot orders. If the
copy is not yet prepared, only the tests (not the launcher) can explicitly select
the original frozen inputs via `PLAY_TEST_REVIEW_BINARY` and
`PLAY_TEST_REVIEW_WEIGHTS`; the same two pinned hashes are still mandatory.
These short native smokes intentionally cancel at their tick cap, rather than
reach `MatchOver`. Post-cap EPIPE is accepted only by the test harness after both
clients exit successfully; exact tick/side summaries are still required. Production
terminal handling is verified by the deterministic tests, not relaxed for a cap.

An additional current Map2 **Teacher-only protocol smoke** uses the existing root
release bot/server on both sides (30 realtime / 1000 lockstep ticks), never a GUI
or a generated Neural checkpoint. To enable it, supply
`PLAY_TEST_CURRENT_BOT_SHA256` and `PLAY_TEST_CURRENT_SERVER_SHA256` from independently
verified current builds when running the test command. Both hashes must match the
target files and must not be the known frozen pins. Without that build attestation
the test explicitly skips rather than trusting potentially stale targets. This is
a protocol smoke, not a launcher Teacher fallback or proof of a trained Neural
artifact. Current pure-Neural TCP behavior is separately covered by Rust's bounded
31-tick test. Neither test proves full-game model quality.

TCP smokes prove admission and policy/runtime compatibility, not rendering, human
input, or full-game quality. Separate Xwayland GUI smokes verified rendering of
both complete replay files with the updated viewer; they did not play an entire
interactive human match. Perform the actual human review on a desktop.
