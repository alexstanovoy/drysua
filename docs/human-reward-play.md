# Human vs Teacher with current RL reward

Current scoring is reward6: Win+0.2, Loss-0.2, Draw0, with all reward5 dense terms
unchanged. Earlier human/native reports below used reward5 and remain historical;
their saved numbers and profile identities are never rewritten as reward6.

**Verified with the current native server in headless fixtures on both sides.**
The graphical client was not launched by the agent; launch from your desktop.
Actual checks, binary/source identities, full and partial native reports are recorded
in [REPORT.md](../artifacts/temp/human-reward-20260914/REPORT.md). Verification ran
serially after the explicit stopped-training ownership handoff; no production training was resumed.

After verification, from the workspace root in an authorized desktop terminal:

```sh
./play.sh --opponent teacher --reward-report --human-side radiant --seed 9000001 --port 0 --no-build
```

Use `--human-side dire` for the other seat. `--bot-side` still selects the opposite
human seat; specifying both requires opposite sides. Teacher is explicitly selected
with `--policy teacher`, needs no model, and rejects `--weights-directory` rather
than ignoring it. Without `--opponent`, the launcher still requires compatible
explicit F20/M22 weights and runs **pure Neural**, never a Teacher fallback.

**`--reward-report` uses native Lockstep, paced at a maximum of 30 ticks per second,
not native Realtime.** Both ordinary clients still create their own ACKs. The shared
relay clock delays those original frames; it never creates or edits ACKs or orders.
A slower renderer/client slows the simulation instead of skipping observations.
Non-report Neural and Teacher play retain their previous native Realtime default.

`--reward-interval 300` controls diagnostic intervals in ticks (default 300, ten
simulation seconds; wall time can be longer; allowed 30..27900). The ordinary native client retains all human
orders, hero selection and readiness controls. No observer sends an ACK, order,
suggestion, or third-client handshake. A graphical desktop is required for ordinary
play; verification fixtures can run headless without launching the GUI.

## What is measured

Native Realtime outboxes can coalesce an old Snapshot while retaining its Events.
Such a stream is insufficient for exact reward reconstruction. Paced Lockstep was
selected to prevent that loss during normal diagnostic play, without modifying bota
or manufacturing missing observations. The Rust observer's gap rejection remains
strict. A failure or timeout does not weaken the complete-stream contract.

The existing two admission relays copy each complete original **server-to-seat**
native length-framed message to independent Rust observers. Each scorer derives its
own identity from Welcome and MatchStart/MatchInfo, validates the matching viewer,
and calls production `Map2Reward` directly on every Snapshot and matching Events.
This is the accounting implementation used by `StateTracker`, not a Python reward
approximation, order-journal heuristic, spectator projection, or model inference.
The human and opponent observations never enter the same scorer.

Tick 1 is required as the initial complete baseline. Subsequent ticks must be
contiguous; empty Events are required and consumed too. Completed-tick reward
intervals are drained without resetting budgets, potential baselines, refundable
fountain charges, or opening/progress state. Diagnostics sum those production f64
components; alternative accumulation groupings can differ at floating-point roundoff.

All 17 production components appear in the report, including net earned gold and
net public XP, hero dealt/taken, creep/tower/other taken, mana, tower/lane potential,
pregame hint, opening positioning, fountain wait/refund, stagnation base/rate and
terminal. Raw production observation counts are included, with progress reason
flags combined by bitwise OR rather than summed.

The public raw paid-gold net is
`raw_counts.own_gold_earned - raw_counts.enemy_gold_earned`; public raw XP net is
`raw_counts.own_xp_gained - raw_counts.enemy_xp_gained`. These are seat-visible
measurements, not omniscient totals. Gold comes from observed paid bounties, not
current wallet, passive income, purchases, sales or end-stat net worth. The signed
`components.gold` and `components.experience` rewards subtract independently
diminishing own/enemy channels; they are not a constant times either raw net.

Only an authoritative MatchOver after the complete final pair closes the lane
potential through `Map2Reward::finish`. Its result is relative to that scorer's
actual assigned team: Win +0.2, Loss -0.2, native Draw 0. TaskTimeCap would be -0.2 in
production, but this passive observer **never invents a task deadline** and never
calls `finish(TimeCap)` on timeout, EOF or manual close.

## Files and console

The launcher announces its existing private `drysua/artifacts/temp/play-*` directory:

* `reward-human.json` / `reward-bot.json`: final JSON for each original seat, with
  profile version/hash, slot/team, completed tick, pending-pair status, outcome,
  completeness/validity, all components, raw counts, total and total without terminal.
* `reward-human.jsonl` / `reward-bot.jsonl`: bounded periodic cumulative component
  values and interval deltas, readable while playing.
* `reward-*.invalid.json`: supervisor failure marker. **Its presence invalidates that
  observation**, including an otherwise complete native JSON file. The supervisor
  also marks a readable final JSON invalid when displaying it. Repeated display
  preserves the original Rust `observer_error` separately from the supervisor error.
  If marker creation or final-file rewriting fails, the in-memory invalid state and
  console warning remain authoritative: persistence failure is logged separately,
  forwarding continues, and an uncorrected file must not be treated as valid.
* `reward-*.log`: bounded observer stdout/stderr.

Periodic running totals are printed to the launcher's terminal. The final category
table labels the human and Teacher separately, and prints each actual team and
outcome. It appears when both observers exit, without requiring the results window
to be closed. Partial totals are prefix measurements, not full episode scores and
not comparable to complete training episodes.

The existing launcher's native replay default and 2 GiB limit are unchanged. The
reward diagnostic adds no replay/BRP stream and saves no full snapshots by default.

## Bounds and failure behavior

* The shared pacing deadline starts on the first original Snapshot 1. Each next
  tick gets one new 1/30-second interval from its actual first Snapshot; late clients
  accumulate no catch-up credit. Timer-aware nonblocking polling releases original
  ACKs only after both seat streams contain that tick's complete Snapshot/Events.
* ACK enum/tick integers are canonical and width-bounded to native u32. Before-start,
  stale, future, duplicate, or noncontiguous ACKs abort the diagnostic rather than
  reaching native `acked = max(acked, tick)` and enabling fast-forward.
* A held ACK and all bytes after it remain in the existing bounded FIFO, with no
  reordering. Orders before it can leave immediately; later orders wait behind it.
  Large queued suffixes drain in bounded batches, not an unbounded catch-up loop.
  A complete held frame at EOF is distinct from a truncated frame.
* Native Lockstep has a finite timeout, not an absolute barrier. Diagnostic servers
  use `--ack-timeout-ticks 900` (about 30 seconds). The relay aborts a stalled tick
  after 10 seconds, before that fallback in normal scheduling. A suspended/failed
  supervisor is not a guarantee of complete play; strict tick/ACK checks and invalid
  reports remain necessary. This is not an invented reward TimeCap.

* Native payload limit: shared `bota_proto::MAX_PAYLOAD_LEN` (4 MiB); canonical native
  decoding rejects extra/trailing payload bytes. Rust accepts at most one million
  messages and 2 GiB input; the relay already enforces the same stream bounds.
* Each nonblocking socketpair writer has at most one maximum framed payload queued
  in Python, plus the bounded OS socket buffer. No observer write blocks the UI.
  Writes make at most 64 KiB progress per pump. A queue overflow/write error or
  30-second blocked write invalidates observation loudly; gameplay may continue.
* Missing, duplicate, gapped, mismatched or truncated pairs do not produce a valid
  score. Manual close/EOF preserves only completed-prefix value with no new terminal.
* Timeline cap: 1 MiB per seat. Very short requested intervals may exhaust this cap;
  that is an explicit diagnostic failure, not permission to silently drop history.
  Default 300-tick intervals are intended to fit a full native episode.
* Observer logs retain the launcher's 16 MiB cap. Its log/relay hard failures still
  abort the supervised game rather than weakening existing safety guarantees.
* At shutdown copied input gets a bounded two-second drain/finalization opportunity,
  followed by the existing own-process-group cleanup. The child registry cap is
  seven: two retained completed builders plus server, bot, client, and two observers;
  actual simultaneously live gameplay children are at most five.

The native `drysua reward-observer --output PATH --interval-ticks 300` subcommand
reads copied frames on stdin. The launcher owns its I/O/lifecycle deadlines; it is
not a server client. Protocol compatibility must be established by building against
the current bota source and recording provenance. Historical wire fixtures and frozen
training/review binaries are not interchangeable with the current Healed amount+mana
protocol, and this diagnostic does not add a protocol version or compatibility shim.

## Verification scope

The [handoff](../artifacts/temp/human-reward-20260914/HANDOFF.md) preserves the earlier
unverified implementation stages. [REPORT.md](../artifacts/temp/human-reward-20260914/REPORT.md)
records the subsequent executed checks and test-contract failures corrected during verification.

Verification has three distinct layers: fake-clock pacing/byte-identity tests;
real **paced** 60-tick manual cuts on both human sides; and at most two complete
headless native Teacher-vs-passive-Player **test fixtures**, one per side. Only the
complete fixtures disable pacing to avoid fifteen wall-clock minutes. That bypass
is test-code-only: normal `--reward-report` has no unpaced option. Complete fixtures
must produce genuine native MatchOver and valid final reports for both seats; unit
transcripts and incomplete cuts alone are not that evidence. Fixture outcomes are
not human performance, training, or policy evaluations. Both complete fixtures
produced genuine native ends with valid human-seat and Teacher-seat reports.
