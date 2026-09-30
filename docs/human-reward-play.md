# Human play with reward reports

`--reward-report` scores both seats of a human game with the production
`Map2Reward` ([reward](reward.md)). From the workspace root, on a graphical desktop:

```sh
./play.sh --opponent teacher --reward-report --human-side radiant --seed 9000001 --port 0 --no-build
```

`--human-side dire` plays the other seat (`--bot-side` selects the opposite; both
must differ). `--opponent teacher` needs no weights and rejects
`--weights-directory`; without it the bot is pure Neural and requires explicit
runtime weights ([local play](local-play.md)). `--reward-interval` sets the
periodic report interval in ticks (default 300, allowed 30..27900).

## Pacing

Reward-report games run native **Lockstep paced at 30 ticks/s**, not Realtime:
Realtime outboxes can coalesce a Snapshot while keeping its Events, which makes
exact reward reconstruction impossible. The relay delays the clients' own ACKs until
both seat streams hold the tick's complete Snapshot and Events; it never creates or
edits ACKs or orders. A slow client slows the simulation instead of dropping
observations. Other play keeps native Realtime.

## What is measured

Each admission relay copies every server-to-seat frame to its own
`drysua reward-observer` process, which validates the seat identity and calls
`Map2Reward` on every Snapshot and its Events, exactly as `StateTracker` does in
training. The two seats never share a scorer. Tick 1 is the baseline; later ticks
must be contiguous. Reports carry all seven components (`towers`, `deaths`,
`health`, `xp`, `closure`, `terminal`, `fast_win`) and the seat-visible counters.

Only a native `MatchOver` after the complete final tick calls `Map2Reward::finish`
(Win +1 plus the fast-win bonus, Loss −1, Draw 0). A timeout, EOF or manual close
never invents a terminal: the report keeps the completed prefix without closure,
and partial totals are not comparable with training episodes.

## Files

The launcher announces its `drysua/artifacts/temp/play-*` directory:

- `reward-human.json`, `reward-bot.json`: final per-seat report (seat, team,
  completed tick, outcome, completeness, components, counters, totals).
- `reward-human.jsonl`, `reward-bot.jsonl`: periodic cumulative values and deltas,
  readable during play.
- `reward-*.invalid.json`: failure marker. **Its presence invalidates that seat's
  report**, even a complete-looking JSON.
- `reward-*.log`: observer output.

Running totals print to the launcher's terminal; the final table labels each seat
with its team and outcome.

## Bounds and failures

- Pacing: each tick gets one 1/30 s slot from its first Snapshot, with no catch-up
  credit. Stale, future, duplicate or noncontiguous ACKs abort the diagnostic.
- Native Lockstep runs with `--ack-timeout-ticks 900`; the relay aborts a tick
  stalled for 10 s before that fallback.
- The observer accepts at most 1,000,000 messages, 2 GiB of input and a 1 MiB
  timeline per seat; payloads are bounded by `bota_proto::MAX_PAYLOAD_LEN` (4 MiB).
- Observer writes are nonblocking with bounded queues; a queue overflow, write error
  or 30 s blocked write invalidates the observation loudly while the game continues.
- Missing, duplicate, gapped or truncated frames never produce a valid score.
- Observer logs share the launcher's 16 MiB cap.

`drysua reward-observer --output PATH --interval-ticks 300` reads copied frames on
stdin; the launcher owns its lifecycle. It must be built against the same bota
source as the server.
