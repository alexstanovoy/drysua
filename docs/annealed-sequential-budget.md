# Sequential annealed games and sample budgets

`train-annealed` admits an even **M = 2..=80** games per PPO update, with
**B = 1..=64** worlds per actor group. B divides both M and K (generation games),
and B times the actor-group count G must divide M and not exceed 64 live worlds.
The CLI defaults to **M40/B20/G2, microbatch256, actor-value reuse enabled**;
B is fixed independently of available CPU cores. The learner backend remains CPU;
pass `--device cuda` for the measured CUDA profile. See
[execution settings](training_microbatch.md) for the complete default and legacy profiles.

With explicit G1 and a fixed environment schedule, M40/B8/K200 runs five sequential
eight-world batches per update and one generation spans five updates. Under the
default adaptive schedule, K/M is an initial budget, not a fixed duration. K remains
a required CLI argument: K160 gives base4 with M40. Collection does not split PPO
updates or reduce episode decisions, retention, or matches.

The CLI chooses `PpoSampleBudget::Standard` for M <= 26, `Annealed` for M28..40,
and the closed `WideAnnealed` profile for M42..80.
Library callers must set the canonical profile explicitly: validation rejects
either mismatch. Expanded updates reserve M × 1,163 retained samples, up to
93,040 in the wide profile (the old profile remains capped at 46,520).
Completed-outcome storage has 80 slots independently of the unchanged
26-world standard train-full limit. Explicit `--parallel 40 --actor-pipeline-groups 1`
collects M40 in one batch. Existing runs must pass their original M/B/G/microbatch/
reuse settings on resume; the new CLI defaults do not migrate old checkpoints.

Only expanded canonical run scopes add `--sample-budget annealed-v1`; existing
M <= 26 command-line bytes are unchanged. This is a generated scope marker,
not a user-selectable CLI override: `--games` determines the profile. Strict
resume still rejects changed M, B, K, PPO configuration, and other run scope.
Wide scopes use `--sample-budget wide-annealed-v1`; all M<=40 scope bytes stay
unchanged for an unchanged resolved execution profile. Candidate M48/B48, M64/B64
and M80/B40 explicitly select G1 and microbatch64; they can use K240, K320 and K400
for a base budget of five updates. Fixed mode makes that duration exact; adaptive
mode can shorten or extend it. M80/B64 is not divisible and is rejected rather
than truncating or inventing a partial batch.

Before constructing worlds or loading weights, preflight bounds cumulative
samples, actor seed draws, optimizer steps, and shuffle draws. Each actor's
per-episode draw limit is checked at compile time. Shuffle accounting is
`updates × epochs × (M × 1,163 - 1)`: M already contributes to the sample
count and is not multiplied again. For 1,000 M40 updates and four epochs this
is 186,076,000 shuffle draws, within the existing counter limit. Shortened
test episodes do not weaken production preflight.

## Compatibility and memory

Wide capacity has explicit PPO39/fixed-checkpoint14 identities, linked to unchanged
PPO38/fixed-checkpoint13. Adaptive state adds the separate checkpoint15/16/17 profiles;
see [environment recovery](adaptive_environments.md#checkpoint-and-generation-recovery).
All three exact runtime tuples are accepted; tensor shapes
and inference semantics are unchanged. Checkpoints keep the 64-byte config
encoding and validate cumulative samples against U*M*1163. No old initializer
is rewritten, and no profile transition is accepted as an implicit resume.

The wide admission ledger is 12,557,484,608 bytes: 9,074,377,280 for capped arena
storage plus worst reallocation overlap, 93,040*8,192 for both compact vectors
and shuffle, 8,192*70,000 for a materialized minibatch, and a 2 GiB non-rollout
reserve. Compile-time size checks enforce both row ceilings and the 12 GiB
total, retaining the original M40 <6 GiB assertion. The reserve is an admission
assumption, not a source-proven whole-process RSS bound; native/model storage
and optional SIL/RND state share it. The unchanged 12 GiB runtime guard remains
mandatory, and actual memory qualification is deferred to the authorized runner.

Standard PPO37 and checkpoint12 descriptors, hashes and encodings are unchanged.
Expanded training uses the explicitly linked PPO38 / checkpoint13 identities.
The parallel40 extension preserves these exact identities and descriptors:
their original `parallel_worlds1to26` / `concurrent_world_limit_unchanged` text
records the initial implementation ceiling, not this build's execution capacity.
This is a build-scoped execution-capability extension, not a tensor, PPO algorithm
or checkpoint-format change. B is already recorded in the strict run scope;
changing B or build provenance still requires an explicit operational migration.
Old binaries continue to reject B40; no artifact is relabeled.
The checkpoint header selects the capacity profile without changing the 64-byte
PPO configuration encoding. Expanded committed sample counts are bounded by
`updates × configured_games × 1,163`, not by the old 32,768-sample ceiling.
Mastery and the actor/learner pipeline remain standard-only.

The original two runtime metadata tuples remain accepted: tensors, feature,
action and inference semantics are identical. Existing PPO37 initializers load
without conversion or rewriting; expanded runtime exports honestly identify
PPO38. Older binaries cannot read these new exports/checkpoints. No migration
is required for existing initializers or standard artifacts; training resume
never crosses profiles and still requires exact provenance and run scope.

Annealed feature arenas reserve fallibly with capped geometric growth; the
standard allocation path is unchanged. Every arena's maximum row offset fits
`u32`. The arena bound includes all row capacities and the largest possible
moving-reallocation overlap (4,537,188,640 bytes). A separate compile-time
bound adds both compact preparation vectors, shuffle indices and one fully
materialized 8,192-sample minibatch, and requires the total below 6 GiB.
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
- `completed_episode_capacity_rejects_overflow_and_duplicate_merge_atomically`
- `wide_capacity_m64_b64_and_m80_b40_collect_one_ppo_step_and_resume_exactly`
- `wide_capacity_profiles_and_partitions_admit_candidates_and_reject_overflow`

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
stream/per-episode limits, public trainer and pipeline isolation, and unchanged
standard identity. `feature_capacity.rs` checks capped allocations and overflow
without allocating a dense maximum-size rollout. `checkpoint_capacity.rs`
checks both codecs, mixed-profile rejection, update-three
counters (139,560 accepted; 139,561 rejected), and exact runtime compatibility.
The retired optional local-initializer test is no longer part of this suite.
Pinned artifact compatibility and CUDA identity probes require the separately
authorized resource-bounded integration run.
