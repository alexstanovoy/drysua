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

New tests in `src/tests/annealed_capacity.rs`, included by `src/tests/annealed.rs`:

- `m40_b8_k200_settings_validate_without_reducing_the_rollout`
- `completed_episodes_merge_all_forty_streams_in_tick_then_stream_order`
- `completed_episodes_accept_stream39_and_reject_stream40_and_duplicates`
- `games40_is_valid_but_games41_and_games42_are_rejected`
- `parallel_worlds26_is_valid_but_parallel_worlds41_is_rejected`
- `parallel40_accepts_m40_k200_and_rejects_maximum_plus_one`
- `parallel40_m40_k200_shortened_updates_resume_byte_identically`
- `expanded_games_keep_batch_and_generation_divisibility_requirements`
- `library_settings_reject_noncanonical_sample_profiles_and_dimensions`
- `cli_selects_expanded_profile_only_above26_and_legacy_scope_bytes_are_unchanged`
- `m40_four_epoch_thousand_update_budget_uses_one_shuffle_per_update`
- `shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one`
- `counter_budget_accepts_exact_limit_and_rejects_limit_plus_one_or_overflow`
- `m40_b8_k200_six_updates_resume_and_mid_update_replay_are_byte_identical`

The single new simulator integration test uses only the existing private
16-decision harness, one epoch, and minibatch 80. It compares six uninterrupted
updates with a run stopped after update five, interrupted after 24 games of
update six, then resumed. It checks checkpoint-file digests, parameters, Adam
moments, RNG state, progress, generation snapshots, and unchanged committed
files after interruption or rejected M/B/K changes. Synthetic completed-stream
tests separately exercise terminal bookkeeping that these short windows do
not reach. The existing side-balance test now covers M28 and M40 as well.

`ppo_capacity.rs` additionally fills all 46,520 retained slots, checks max+1,
stream/per-episode limits, public trainer and pipeline isolation, and unchanged
standard identity. `feature_capacity.rs` checks capped allocations and overflow
without allocating a dense maximum-size rollout. `checkpoint_capacity.rs`
checks both codecs, legacy byte identity, mixed-profile rejection, update-three
counters (139,560 accepted; 139,561 rejected), and exact runtime compatibility.
Its ignored local-initializer test requires an explicit
`DRYSUA_INITIALIZER_TEST_DIRECTORY`, loads the existing runtime on CPU/CUDA and
verifies every initializer file's SHA256 is unchanged.
