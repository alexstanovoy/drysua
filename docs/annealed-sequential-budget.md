# Sequential annealed games and sample budgets

`train-annealed` admits an even **M = 2..=40** games per PPO update, with
**B = 1..=40** concurrent worlds. B must still divide both M and K (games per
generation). M40/B8/K200 runs five sequential eight-world batches per update;
one generation spans five updates. Collection does not split PPO updates or
reduce episode decisions, retention, or matches.

The CLI chooses `PpoSampleBudget::Standard` for M <= 26 and `Annealed` above 26.
Library callers must set the canonical profile explicitly: validation rejects
either mismatch. Expanded updates reserve M × 1,163 retained samples, up to
46,520. Completed-outcome storage has 40 slots independently of the unchanged
26-world standard train-full limit. Explicit `--parallel 40` collects M40 in one
batch. Automatic parallel selection retains its previous 26-core ceiling, so
existing invocations without an explicit parallel setting do not change.

Only expanded canonical run scopes add `--sample-budget annealed-v1`; existing
M <= 26 command-line bytes are unchanged. This is a generated scope marker,
not a user-selectable CLI override: `--games` determines the profile. Strict
resume still rejects changed M, B, K, PPO configuration, and other run scope.

Before constructing worlds or loading weights, preflight bounds cumulative
samples, actor seed draws, optimizer steps, and shuffle draws. Each actor's
per-episode draw limit is checked at compile time. Shuffle accounting is
`updates × epochs × (M × 1,163 - 1)`: M already contributes to the sample
count and is not multiplied again. For 1,000 M40 updates and four epochs this
is 186,076,000 shuffle draws, within the existing counter limit. Shortened
test episodes do not weaken production preflight.

## Compatibility and memory

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

Runtime readers accept exactly either audited metadata tuple: tensors, feature,
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

## Regression tests

Boundary and compatibility tests precede their implementation. Heavy checks
must execute serially under the original guard; neither these tests nor this
extension authorize a production training launch.

Current contracts in `src/tests/annealed_capacity.rs`, included by `src/tests/annealed.rs`:

- `m40_sequential_and_b40_updates_resume_model_optimizer_rng_and_generation_bytes`
- `expanded_jobs_enforce_capacity_and_partition_boundaries`
- `shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one`
- `completed_episode_capacity_rejects_overflow_and_duplicate_merge_atomically`

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
