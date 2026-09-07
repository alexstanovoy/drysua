# Play locally against drysua

From a Linux X11/Xwayland desktop terminal:

```sh
/home/alexstanovoy/Workspace/bots/play.sh
# Optional: an OS-assigned port and an explicit reproducible match seed.
/home/alexstanovoy/Workspace/bots/play.sh --port 0 --seed 9000001
# Skip compilation only when all three release binaries are already current.
/home/alexstanovoy/Workspace/bots/play.sh --no-build
```

The script works from another directory and with spaces in the checkout path.
`--help` works without a display. `DISPLAY` is required before any build starts;
the client uses X11, so `WAYLAND_DISPLAY` alone is insufficient. A nonempty display
variable does not guarantee X authorization or a working OpenGL context. Python
3.10+ and, unless using `--no-build`, Cargo/Rust must already be installed.

The small Bash entry point replaces itself with `drysua/scripts/play_match.py`.
The helper uses only Python's standard library to supervise subprocess sessions,
drain pipes without startup sleeps, and clean up descendants on signals. No
additional Python packages, shell process-management utilities, or Cargo
configuration changes are needed.

## Build and launch contract

By default every invocation builds the current working trees, including local
uncommitted changes. With `root` set to the directory containing `play.sh`, the
commands are:

```sh
CARGO_TARGET_DIR="$root/bota/target" \
cargo build --release --locked --quiet \
  --manifest-path "$root/bota/Cargo.toml" \
  -p bota-server -p bota-client --bin bota-server --bin bota-client

CARGO_TARGET_DIR="$root/drysua/target" \
cargo build --release --locked --quiet --no-default-features \
  --manifest-path "$root/drysua/Cargo.toml" --bin drysua
```

Each build has a 20-minute deadline. The two explicit `CARGO_TARGET_DIR` values
override any inherited value for these build subprocesses only. Executables are
always taken from the corresponding `target/release` directories, including with
`--no-build`; an inherited target directory does not redirect execution. There
are no debug, full-workspace, CUDA, or training builds.

The server uses `--mode realtime --players 2 --map 0`, with port `4455` and seed
`9000001` by default. Startup waits for its exact stdout listening line, with a
10-second deadline and a 4096-byte readiness buffer. No TCP health probe connects
to the lobby. With port `0`, both clients receive the port in that line. The
server currently binds all IPv4 interfaces, even though clients use loopback;
use an appropriately trusted host/network or firewall.

The GUI and bot launch automatically after readiness; no terminal confirmation
is required. **Sides follow server connection order, not launch order.** There
is no promise that the human is Radiant. In the GUI, press `1` for Sylla, `2` for
Pudge, or `3` for Shadow Fiend, then `R` to become ready. Sylla is initially
selected. Drysua picks Shadow Fiend and readies itself.

The bot command contains only `--addr` and `--name`. The selected default
deployment belongs to drysua's CLI (currently the Teacher preview), not this
launcher. There is no hardcoded policy, weights path, artifact selection, or
fallback in the shell/helper. Keep both repositories' release binaries current;
`--no-build` does not verify protocol or deployment compatibility.

## Run only the selected bot

With a server already listening at `127.0.0.1:4455`, run from `drysua`:

```sh
cargo run --release --bin drysua
```

No operation, policy, or artifact argument is required. Both implicit play and
`play` use `DEFAULT_DEPLOYMENT` in `src/default_deployment.rs`. The current choice
is the human-tested Teacher preview; this is a maintained play selection, not a
claim that the preview passed historical promotion. The CLI logs the selection.

When a stronger deployment is selected, update that one configuration and its
evidence. A future `Tactical` or `Hybrid` selection must specify its compatible
repository-relative artifact directory, such as `artifacts/v0.0.5`. The runtime
resolves it from the compiled repository path, not the caller's working directory.
Teacher must remain weights-free. Invalid policy/weights combinations are checked
at compile time for the configured default. There is no newest-file/tag guessing
and no fallback to another model when the selected artifact cannot be loaded.

Explicit experimental overrides remain available: `--policy teacher`, or
`--policy tactical --weights-directory ...`, or Hybrid with explicit weights.
For compatibility, supplying `--weights-directory` without `--policy` still
selects Hybrid; it does not silently ignore that explicit artifact request.

## Shutdown and diagnostics

Ctrl+C stops the launcher and its owned process groups, including Cargo/compiler
descendants during builds, and exits `130`. TERM exits `143`; HUP exits `129`.
Closing the GUI also stops the remaining match processes. Nonzero component
exits, startup failures, log overflow, and the GUI's `bota-client:` error diagnostic
fail the launcher. Successful server/bot exits leave the GUI results visible
until the window closes or the launcher is interrupted.

Each child gets a new session/process group. Cleanup sends TERM, allows up to two
seconds for leaders to exit, then sends KILL to every owned group and waits for
the direct children. Exited leaders remain unreaped until group signalling is
complete, preventing process-group ID reuse. No existing server or unrelated
process is signalled. SIGKILL of the supervisor cannot be handled, and descendants
that deliberately create a different session are outside process-group cleanup.

The printed private `drysua/artifacts/temp/play-*` directory retains combined
stdout/stderr logs (`build-bota.log`, `build-drysua.log`, `server.log`, `client.log`,
`bot.log`) and `match.brp`. Each child log is capped at 16 MiB; overflow stops the
run rather than consuming unbounded disk. The server inherits a 512 MiB file-size
limit for its replay. Interrupted games may leave incomplete replays. Run
directories are retained for diagnosis and can be removed manually when no longer
needed; builds still write to their usual per-repository target directories.

## Launcher tests

```sh
bash -n /home/alexstanovoy/Workspace/bots/play.sh
python3 -B -m unittest discover \
  -s /home/alexstanovoy/Workspace/bots/drysua/scripts -p test_play_match.py -q
```

Tests use only the standard library and fake Cargo/game executables below
`drysua/artifacts/temp`, without compiling Rust or opening a graphical window.
They exercise actual subprocess groups, bounded pipe readiness, process exits,
signals, descendant cleanup, argument contracts, and per-repository target paths.
When existing release binaries are available, native smokes check the server's
piped stdout readiness and two default-deployment bots over real TCP to tick 1000.
The latter requires a fresh bot build; it does not test the GUI/OpenGL window.
