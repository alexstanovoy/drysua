# Scoped PPO tensor microbatches

`train-annealed --training-microbatch 64|128|256` selects the tensor rows in each
PPO gradient and post-step candidate-KL forward. The **train-annealed CLI default
is 256**; the **library `TrainingExecutionOptions::default()` remains 64**.
Explicit `--training-microbatch 64` retains the prior partition. This is separate
from the effective Adam minibatch (normally 2048). Actor sampling, inference
chunking and public `training_forward` stay at 64.

128 and 256 are distinct numerical modes: larger GEMMs and different F32 gradient
and KL reduction groupings can change bits, KL stopping and later learning. Values
above the historical/library 64 are appended to `CheckpointRun.command_line`,
including the CLI default 256 even when its flag was omitted. Strict resume rejects
changing the resolved mode. The PPO config codec, model/tensor/action
schemas, epoch count, effective minibatch and optimizer hyperparameters are not
changed. No cross-mode equality or automatic migration is claimed. Annealed restore
strips only collector flags, preserving the training microbatch and host-math mode.

The internal ceiling and candidate old-logprob stack buffer are 256. Current chunks
and final tails retain all rows. Frame checks, all thirteen output finite checks,
NLL masks/labels, serial F64 norm and complete effective-minibatch candidate/rollback
logic remain enabled. Within one mode, exact replay is required. Public limit and
library-default-64 reference tests remain separate from within-mode 128/256 tests.

## CLI profile and resume

Fresh `train-annealed` runs default to M40/B20/G2, training microbatch 256, and
actor-value reuse enabled: `--games 40 --parallel 20 --actor-pipeline-groups 2
--training-microbatch 256 --reuse-actor-values=true`. Parallel width is fixed at 20,
not chosen from the host's CPU count. Bare `--reuse-actor-values` also means true;
use `--reuse-actor-values=false` to disable it. The backend is still **CPU** by
default; **`--device cuda` is required** to select the measured CUDA profile.
This default change does not establish a speedup on CPU or another workload.
Library execution defaults remain G1/micro64/reuse=false; PPO, adaptive schedule,
host-math worker, balanced-minibatch, model, and inference defaults are unchanged.

For an existing empty checkpoint directory, this selects the fresh CUDA profile
with a base environment budget of four updates (160 / 40):

```sh
drysua train-annealed --updates 200 --games 40 --parallel 20 \
  --generation-games 160 --device cuda \
  --checkpoint-directory artifacts/my-new-run
```

`--generation-games` is still required. Adaptive generations require a positive
whole multiple of M; 32 is invalid with the default M40. The `--zero-updates`
default is unchanged: one fifth of total updates rounded up (40 here). G2/G4
require a Teacher opponent and whole waves of at most 64 live worlds; an explicit
M40/B10/G4 is valid. Weights opponents require `--actor-pipeline-groups 1`.

Resume requires the checkpoint's **original execution settings**, not the new CLI
defaults. For a legacy fixed CPU run originally using M8/B8/G1/micro64/reuse=false,
one host-math worker, and unbalanced minibatches:

```sh
drysua train-annealed --updates 200 --games 8 --parallel 8 \
  --generation-games 32 --actor-pipeline-groups 1 --training-microbatch 64 \
  --reuse-actor-values=false --host-math-workers 1 --device cpu \
  --environment-schedule fixed --resume \
  --checkpoint-directory artifacts/my-legacy-run
```

Keep its original seed, zero-update budget, optimizer options, opponent, and any
other execution overrides and compatible build provenance too. Omit
`--balanced-minibatches` only if originally
disabled. The fixed selector alone cannot restore the old execution profile.
Explicit micro64/G1/reuse=false retain the historical suffix-free scope encoding;
they do not rewrite the checkpoint. Adaptive resumes likewise retain their
original execution settings and adaptive tuning. See
[environment schedules](adaptive_environments.md) for the scheduler contract.

## Memory admission

The fixed-model source estimate includes encoder forward/backward copies and four
full-width pooling intermediates, multiplied by a four-live-copy allowance plus
two dense feature frames. A compile-time assertion checks this against a conservative
**16 MiB per tensor row** reservation. This is an admission envelope, not a measured
allocator/VRAM peak or a proof about all backend workspaces.

With the current dimensions the estimate is 523,152 float elements per row before
the four-live-copy multiplier. Even replacing each feature-frame size with the
existing 70,000-byte prepared-sample ceiling gives 8,510,432 bytes per row, below
the 16 MiB reservation. This arithmetic is an admission bound, not a measured
allocator or backend-workspace guarantee. Compiler and runtime checks must
accompany changes to the bound.

For standard/M40 profiles, admission adds that reservation (2 GiB at 128; 4 GiB at
256) and a 2 GiB non-graph reserve to the existing M40 rollout/storage ledger. For
the maximum-wide profile it adds only the extra rows above the existing 64-row
allowance to the pre-existing wide payload bound. The maximum-wide ledger is already
near 12 GiB, so larger microbatches are conservatively rejected for that profile.
Wide profiles (M48/M64/M80, for example) must explicitly select microbatch 64
instead of the new CLI default 256. To retain a legacy wide profile, also specify
its original M/B, `--actor-pipeline-groups 1`, and `--reuse-actor-values=false`.
Library-default or explicit 64 keeps existing admission behavior. The ceiling is
still **12 GiB**; runtime RAM/VRAM guards remain authoritative and must not be raised. Admission is
checked before checkpoint-directory or opponent setup and again when installing
the trainer's execution options.

The existing M40 storage assertion is strictly below 6 GiB, so adding 2 GiB and
the maximum 4 GiB graph reservation stays below 12 GiB. The current wide ledger
would reach 13,631,226,432 bytes at 128 or 15,778,710,080 bytes at 256, both above
the unchanged 12,884,901,888-byte ceiling.

## Verification protocol

Run these commands only when the authorized exclusive runner is available, never
alongside a training campaign. Source-only edits do not authorize concurrent builds,
tests, or probes.

Focused CPU filters:

```sh
cargo test --lib --all-features --quiet model::transfer_tests::microbatch_tests
cargo test --lib --all-features --quiet ppo_arena::annealed::tests::training_microbatch_scope
```

Ignored CUDA test (only through the authorized exclusive runner):

```sh
cargo test --lib --all-features --quiet model::transfer_tests::microbatch_tests::cuda_training_microbatch_modes_preserve_within_mode_state_and_uneven_tails -- --ignored --exact --nocapture
```

The CUDA cases cover 65/129/257/2049 rows and the configured 64/128/256 partition,
including candidate rejection and injected post-step evaluation failure. CPU tests
cover default-64 bit identity, exact same-mode updates, bounds/finite/target checks,
scope mismatch and restored execution options. These checks verify contracts;
throughput claims additionally require matched full-work measurements.

The integrator must also run the repository-wide gates in that runner:

```sh
cargo clippy --all-targets --all-features -- -D warnings -D clippy::all
cargo test --all-targets --all-features --quiet
cargo fmt
cargo machete
```

The existing ignored `concurrency_probe_cuda` accepts
`DRYSUA_PROBE_TRAINING_MICROBATCH=64|128|256` (default 64). Every fresh trial within
one invocation uses the same numerical mode, so existing within-invocation bit
checks remain meaningful. Cross-mode final weights/reports may differ. For the
planned full40 CREDIT-u10 comparison keep the identical pinned initial weights,
seed, collector settings, epochs=4 and effective minibatch=2048. Require the same
pre-update actor trace/retained rows and **40 applied Adam steps** on both sides;
a different KL stop or fewer steps is not a same-work speedup. The theoretical
2048-row tensor-pass count is 32/16/8 for 64/128/256, not a measured 2x speedup.
