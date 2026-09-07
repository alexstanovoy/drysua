# Fixed-Teacher Map0 challenge

This is **not** the historical Map1 release gate and does not promote a deployment
default. No candidate is qualified by preparing this harness.

## Frozen opponent

Manifest (repository-relative):

```text
artifacts/temp/map0-baseline-observationfix-4096/baseline.json
```

- Epoch: `map0-fixed-teacher-4096-v1`.
- Manifest SHA256: `e9c279bfe82b63a1c17ab2f3e1bbebb7b6936a0c256241251e1dfb4a992fbe8b`.
- Teacher binary SHA256: `dcd3baf950a5dbf9018a40bb7be37b4413b06a18380d333987aaf8c29b3dbc3f`.
- Simulator SHA256: `24a8efccb285308810678c7e3a8717b57814c9d8fecef386ca923cdb7c04e97c`.

The derivative was built **release-only** from all 310 source files in the sealed
`map0-baseline-observationfix` source manifest. Its only source difference is the
four-line observation bound patch from
`artifacts/temp/map0-bound4096/baseline-observation-only.patch`, SHA256
`ecd8fc673d464fed74407f517228714fb5efdc82da4abda890173204d1792cd1`.
Two tracker bounds and two associated feature assertions change from 512 to 4096.
Teacher strategy, the simulator, token rows, and neural schemas are not modified.
The simulator runtime is copied byte-for-byte from the sealed reference.

The frozen binary is always invoked with `play --policy teacher`, **without
weights**. It is never substituted with the candidate binary's Teacher. The
epoch configuration pins the manifest SHA; loading also checks map, role, policy,
source provenance, and runtime/source-manifest/patch hashes. Run-local immutable
binary and weight snapshots are rechecked after evaluation.

Verification artifacts:

- `artifacts/temp/map0-baseline-observationfix-4096/parity.json`: seed `9470001`,
  complete recorded send-stream hashes and outcomes match the old 512 reference.
- `artifacts/temp/map0-fixed-runtime-verification/108900/result.json`: real TCP,
  accepted-order stream and both terminal summaries equal the old reference;
  Radiant wins at tick `46760`, zero rejected orders, approximately 21.6 seconds.
- The same verification directory has cap-8 and cap-32 TCP regressions, both clean
  timeouts, not wins or protocol errors.
- `cap-lifecycle-final-v2/summary.json` adds twelve real TCP cap/EOF regressions
  at caps 7, 10, and 32, run with four parallel workers; all are clean timeouts.
- The old sealed snapshot's full `SHA256SUMS` check passes unchanged.

## Development and final cohorts

`scripts/map0_challenge.json` predeclares the epoch and limits:

| Cohort | Seeds | Games | Qualification |
|---|---|---:|---|
| Development | `9870000` onward, at most ten pairs | 2–20 | Never qualifies |
| Final | `9880000–9880049`, every seed on both sides | Exactly 100 | At least 80 wins |

**Final seeds must never enter optimizer input, curriculum/stage selection, or
development evaluation.** Freeze the candidate binary and weights before the
final cohort. Neither cohort was run for a candidate during harness preparation.
Freshness is an operational constraint: the harness cannot audit external
training processes. Changing the baseline or final cohort requires a new epoch,
not relabeling an existing report.

Every draw, tick-cap timeout, or wall timeout remains a non-win in the full
100-game denominator. Missing/duplicate games and any protocol, baseline,
candidate, identity, or runtime-integrity errors make the run ineligible. A
baseline failure never gives the candidate a free win. CLI claims alone cannot
establish a winner: summaries must match observed terminal frames and tick/seat
identities. `--challenge-final` rejects `--challenge-pairs` overrides.

From the repository root, with `CANDIDATE`, `WEIGHTS`, and `PROVENANCE` set to the
exact candidate binary, runtime-weight directory, and build/source description:

```sh
python3 -B scripts/release_crossplay.py \
  --teacher-challenge-map0 \
  --baseline-manifest "$PWD/artifacts/temp/map0-baseline-observationfix-4096/baseline.json" \
  --candidate-policy neural --candidate-binary "$CANDIDATE" \
  --candidate-weights "$WEIGHTS" --candidate-metadata "$PROVENANCE" \
  --challenge-pairs 10 --challenge-workers 4 --run-name map0-fixed-dev-001
```

Only when a strong, fixed candidate is ready, replace `--challenge-pairs 10` with
`--challenge-final` and use a fresh run name. The final flag schedules all 100
games; it cannot be reduced to a development sample.

## Bounds and cap lifecycle

Every Map0 game has tick cap `108900` and a 180-second wall watchdog. At most four
games run concurrently, with at most four queued jobs and one Rayon/OpenMP/BLAS
thread per process. Each relay has a 2 GiB cumulative server-wire bound, 64 MiB
client-wire bound, a 4 MiB frame bound, and tick-derived frame/iteration limits.
Client logs remain bounded to 64 KiB. Map0 challenge runs do not record full
replays, avoiding hundreds of gigabytes across the final cohort; reports retain
the exact runtimes, weights, commands, seeds, roles, and proxy observations.

The relay withholds **only the verified cap ACK**, leaving the upstream seat
connected when the client exits. This prevents cap+1 advancement instead of
relaxing the wire hard maximum. It drains cap Events before supervisor shutdown;
both correct peer identities and verified cap boundaries are required for a
clean cap timeout. A server snapshot beyond the cap is still an error.
The wall watchdog marks interrupted peers before tearing down the host, so its
own shutdown does not manufacture baseline errors. Previously observed errors
remain ineligible; a wall timeout can never become a win.

The historical Map1 registry, opponents, seeds, strict-majority release gate, and
20 games per opponent are unchanged. Its shared relay receives the same cap/ACK
lifecycle fix; there is no timeout-to-win normalization.
