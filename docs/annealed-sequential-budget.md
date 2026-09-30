# Sequential annealed games and sample budget

`train-annealed` admits an even **M = 2..=40** games per PPO update, with
**B = 1..=64** worlds per actor group. B divides both M and K (generation games),
and B times the actor-group count G must divide M and not exceed 64 live worlds.
The CLI defaults to **M40/B20/G2, microbatch256, actor-value reuse enabled**;
B is fixed independently of available CPU cores. The learner backend remains CPU;
pass `--device cuda` for the measured CUDA profile. See
[execution settings](training_microbatch.md) for the complete default profile.

With explicit G1 and a fixed environment schedule, M40/B8/K200 runs five sequential
eight-world batches per update and one generation spans five updates. Under the
default adaptive schedule, K/M is an initial budget, not a fixed duration. K remains
a required CLI argument: K160 gives base4 with M40. Collection does not split PPO
updates or reduce episode decisions, retention, or matches.

There is one PPO sample budget: an update reserves M × 1,163 retained samples,
at most 46,520. Strict resume rejects changed M, B, K, PPO configuration, and
other run scope.

Before constructing worlds or loading weights, preflight bounds cumulative
samples, actor seed draws, optimizer steps, and shuffle draws. Each actor's
per-episode draw limit is checked at compile time. Shuffle accounting is
`updates × epochs × (M × 1,163 - 1)`: M already contributes to the sample
count and is not multiplied again. For 1,000 M40 updates and four epochs this
is 186,076,000 shuffle draws, within the existing counter limit. Shortened
test episodes do not weaken production preflight.

## Memory

Feature arenas reserve fallibly with capped geometric growth. Every arena's
maximum row offset fits `u32`. The arena bound includes all row capacities and
the largest possible moving-reallocation overlap (4,537,188,640 bytes). A separate
compile-time bound adds both compact preparation vectors, shuffle indices and one
fully materialized 8,192-sample minibatch, and requires the total below 6 GiB.
This excludes allocator overhead, model/optimizer tensors and native worlds:
it is not a whole-process memory guarantee. The unchanged
[resource guard](experiment-safety.md) is mandatory.

## Scoped actor bootstrap reuse

`train-annealed --reuse-actor-values` reuses the next actor batch's value for a
retained transition instead of running a separate batch-one flush evaluator.
It is **on by default for the train-annealed CLI**, and remains off in library
execution defaults. `--reuse-actor-values=false` disables it; bare or `=true` enables
it. Actor batch membership, action sampling, reward
accounting, and retained-row order stay unchanged; terminal bootstraps stay zero.
Bounded test windows use a final batch-one fallback without another actor draw.

Batch-one and actor-batch GEMMs can differ in floating-point bits. Reuse therefore
may change GAE and subsequent PPO parameters even when collection trajectories
match. The flag is recorded in canonical checkpoint scope, and changing it on
resume is rejected. It is an annealed-collector option, not a public trainer or
standard `train`/`train-full` option. Old checkpoints still require their original
reuse setting explicitly; performance qualification does not authorize a new
production training launch or implicit resume migration.

## Regression tests

Boundary and compatibility tests precede their implementation. Heavy checks
must execute under the currently authorized guard; neither these tests nor this
extension authorize a production training launch.

Current contracts in `src/tests/annealed_capacity.rs`, included by `src/tests/annealed.rs`:

- `m40_sequential_and_b40_updates_resume_model_optimizer_rng_and_generation_bytes`
- `expanded_jobs_enforce_capacity_and_partition_boundaries`
- `shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one`
- `completed_episode_capacity_rejects_overflow_atomically`
- `partitions_admit_forty_games_and_reject_overflow`

The parameterized simulator integration test uses only the existing private
16-decision harness, one epoch, and minibatch 80. It compares six uninterrupted
updates with a run stopped after update five, interrupted after 24 games of
update six, then resumed. It checks checkpoint-file digests, parameters, Adam
moments, RNG state, progress, generation snapshots, and unchanged committed
files after interruption or rejected M/B/K changes. Synthetic completed-stream
tests separately exercise terminal bookkeeping that these short windows do
not reach. The earlier exhaustive side-balance matrix was retired during test
consolidation; it is not an additional current coverage claim.

`ppo_capacity.rs` additionally fills all 46,520 retained slots, checks max+1,
stream limits and public trainer isolation. `feature_capacity.rs` checks capped
allocations and overflow without allocating a dense maximum-size rollout.
`checkpoint_capacity.rs` checks the codec, foreign-identity rejection, update-three
counters (139,560 accepted; 139,561 rejected), and exact runtime compatibility.
The retired optional local-initializer test is no longer part of this suite.
Pinned artifact compatibility and CUDA identity probes require the separately
authorized resource-bounded integration run.
