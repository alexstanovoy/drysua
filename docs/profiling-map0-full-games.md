# Map0 full-game performance profile

Profiling only: no policy, simulator, precision, or optimizer changes were applied.
All measurements use release builds and the pinned simulator `18db0f62d9a2b94e755c43fd29a959db204cc20b`.

## Workloads and correctness

The policy is the retained trained u10 reference transferred to M12 with zero
new input weights, not random initialization. Its artifact SHA-256 is
`adbecb8293548ad1602b24b046b9a6f032fc62798d46102029a1f3448d764bdd`.

- CUDA F32 experience collection: six complete Map0 games, three paired seeds,
  production stochastic sampling and randomized one-in-eight retention. 111809
  actions, 335418 simulator ticks, 13976 retained transitions; two wins/four losses,
  no timeouts or rejections. No optimizer update was run.
- Greedy CPU gameplay: seed 9891000, both seats, actual wins/losses at ticks 47758
  and 51828. This is builtin gameplay, not a TCP-overhead measurement.
- Independent simulator replay: a fresh recorded neural-versus-frozen-Teacher game
  at seed 9891100, Radiant wins at tick 70891. All recorded snapshots/events and
  per-tick world hashes match instrumented and uninstrumented reconstruction.

Application timer-on/off action, value/log-probability and request hashes match.
The CPU greedy path produces no log-probability; none was added for profiling.
Both application modes retain common trace-hashing overhead; they are not pristine
production binaries. Runs advance as fast as possible, without realtime sleeping.
Build/model initialization are separate from decision-loop timing. A shared lock
serialized measurements. One failed Nsight attempt left a short unlocked tail;
that capture was discarded. Ordinary measurements preceded that interval, and
the successful capture held the lock through process completion.

## Where experience-collection time goes

Disjoint scopes. CUDA values average two ordinary instrumented complete batches;
nested measurements below must not be added to this table.

| Component | CUDA E6 seconds | CUDA E6 share | CPU greedy share |
| --- | ---: | ---: | ---: |
| Arena stepping, including projection | 94.347 | **39.27%** | 10.04% |
| ActionSpace construction | 40.284 | **16.77%** | 5.77% |
| FeatureEncoder observation/history | 27.211 | **11.32%** | 3.84% |
| Feature encoding, including validation | 9.867 | 4.11% | 1.22% |
| StateTracker Snapshot + Events | 6.790 | 2.83% | 0.86% |
| Neural sampling/choice and bootstrap | 57.066 | **23.75%** | **77.66%** |
| Other work/instrumentation | 4.706 | 1.96% | 0.61% |

Timer-off CUDA collection took **241.711 s**, or **462.57 raw actions/s** and
**57.82 retained transitions/s**. Timed repeats were 242.413 and 238.130 seconds.
CPU greedy took 199.100 seconds without timers, 202.358 with timers. CPU and CUDA
use different policies (greedy/sampled), batch sizes and trajectories; this is a
bottleneck comparison, not an isolated backend-speedup experiment.

Full games reverse the earlier short-prefix conclusion. CUDA collection falls
from about 592 actions/s early to **262 actions/s late**, while Arena's share
grows from 29.90% to **50.92%**. Late NN-visible unit counts average 130, peaking
at 291. These are not total world-entity counts. CPU games ended before this
profile's late phase; no late CPU rate is claimed.

## Inside the simulator

The independent recorded game's plain World loop took **18.866 s median**;
phase instrumentation added **2.49%**. The following shares are of its own World
phase sum, not fractions of the different CUDA training trajectory above.

| World phase | Seconds | World share |
| --- | ---: | ---: |
| Walking/steering/pathfinding | 5.880 | **30.47%** |
| Target selection | 4.345 | **22.52%** |
| Visibility | 3.421 | **17.73%** |
| Neutral AI | 2.272 | **11.78%** |
| Collision separation | 2.023 | **10.48%** |
| All remaining phases | 1.355 | 7.02% |

The deeper counter run has 8.85% overhead and is reported separately. It found:

- `blocked_by_bodies`: **11.584 million calls / 2.091 billion entity visits**.
- `best_valid_in_range`: **6.662 million calls / 1.479 billion entity visits**.
- `anything_hostile_near`: **5.926 million calls / 1.189 billion entity visits**.
- Visibility: **2.881 billion viewer-target pairs** before filtering.
- Collision: **1.385 billion pairs**.

World cost rises from **147 to 491 microseconds/tick** as the match progresses.
Walking rises from **23 to 194 microseconds/tick**. Repeated broad entity scans and
crowded steering are the important scaling issue. A* itself took only 262 ms over
1039 calls. Collision gathering/allocation took 46.5 ms versus 1.941 s in its pair
loop: recycling that Vec alone is not a major acceleration.

The server-shaped output benchmark separately measured projections at 0.860 s,
three snapshot encodings at 2.188 s, and an additional replay encoding at 1.072 s.
Those encodings are not costs of builtin Arena training; sockets/disk writes were
excluded from that benchmark.

## GPU overhead

Bounded Nsight captures cover 100 actual actor rounds each, not full-game GPU
utilization. Midgame batch6 runs approximately **76 kernels/action**; late batch3
runs **125 kernels/action**. Actual kernel execution is about **115 / 189 us per
action**. There are **62 / 103 H-to-D copies per action** respectively.

For the midgame 600-action capture, device kernels total 68.9 ms, versus 84.8 ms
of host launch API time and 108.6 ms of host allocation/free/event APIs. These
overlapping profiler domains are not additive wall-time fractions. The finding
is many tiny operations and host bookkeeping, not saturated bulk PCIe or GEMM.

## Cheap experiments to try next

These are opportunities and scope ceilings, **not measured optimization gains**.

1. **Avoid redundant singleton value evaluation.** It occupies **6.75%** of E6
   collection, before associated preparation. All 13970 nonterminal bootstrap
   states recur at the next actor call, whose batch output already has a value.
   Values differ by at most 5.96e-8 in 8436 cases, so a general replacement needs
   frame-equivalence and numerical/GAE tests, not a claim of bitwise equality.
   The existing lambda=1 MC path ignores intermediate next-values altogether;
   only a final truncation bootstrap is needed. A lazy representation must retain
   that value and must not be consumed accidentally by a lambda<1 trainer.
   Eliminating just the measured singleton-model scope caps gain at **1.072x**.

2. **Skip encoder history on permanently Teacher-only opponent seats.** Keep all
   tracker, event, readiness and Teacher state work. Do not skip expert-label or
   neural/future-neural seats. Both-seat observation is 11.32%; the exact removable
   Teacher share was not isolated, so it is not valid to assume half the cost.

3. **Cache immutable passability geometry.** Reconstruction costs **8.92%** of
   E6 collection (ceiling **1.098x**). Cache terrain and integer tree footprints,
   not a final passability result: visibility-dependent felling, planting,
   overlapping blockers and structures must still be handled exactly.

4. **Cache `phased` in the collision body snapshot.** In
   `bota-server/src/game/systems/ground.rs:190–215`, the same immutable property is
   queried inside the pair loop. This is a small, low-risk experiment with replay
   parity tests. Its saving is not measured; it cannot eliminate the whole pair loop.

5. **Then address repeated spatial scans.** Hoist per-query seeker facts, check
   duplicated hostility work, and consider compact visibility/body inputs before
   a spatial index. An index has higher potential but must preserve deterministic
   ordering and update as bodies move sequentially. A whole-tick position cache
   is unsafe for steering. Halving all walking would yield **1.18x World-only**
   throughput on the measured replay, not a demonstrated training speedup.

Do not start with wholesale validation redesign: all finite scans cost only
**0.41%** of E6 time (ceiling 1.0042x). Do not claim 2x from these measurements.

## Raw evidence and reproduction

- `artifacts/temp/map0-full-profile-001/REPORT.md`: application tables, phases,
  exact commands, frozen source and weight identities, timer toggles and Nsight.
- `artifacts/temp/map0-simulator-profile-001/REPORT.md` and `TABLES.md`: fresh
  recording, replay verification, disjoint engine phases and nested query counts.
- Replay SHA-256:
  `b5d8e8ed8d4a41cdb14e810d596c80608c9d2aed6a980a2f329609710e3c96f3`.

From `drysua`, replay the plain World benchmark:

```sh
flock artifacts/temp/map0-full-profile-001/measurement.lock \
  artifacts/temp/map0-simulator-profile-001/probe-plain \
  --replay artifacts/temp/map0-simulator-profile-001/match.brp --benchmark
```

The application report records its frozen release probe commands. Builds are
excluded from benchmark timing. No root privileges, perf sysctl changes, new
dependencies, reduced precision, training updates or production optimizations
were used for this investigation.
