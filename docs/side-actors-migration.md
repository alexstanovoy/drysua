# M25 side-actor initialization (opt-in)

`side-actors` is default off. Without it the M24 descriptors, schema hashes and
compiled-feature string are unchanged. M25 has 1,812,983 parameters / 86 tensors;
checkpoint codec versions remain 12–17, with architecture distinguished by linked
hashes and compiled features. Runtime loading and training resume remain strict.

The only accepted M24 initialization source is the full runtime file
`artifacts/temp/annealed-teacher-20260920/attempt-008/history/update-0428/drysua.weights.safetensors`,
SHA-256 `895e66a79186570ce8f653eadfad59eace8c664f16c5c524806af4936ac9f21f`.
The path is a locator, not an authentication mechanism: the bounded full bytes
must match the pin. Exact PPO38/rules32/feature22/action5/reward7 metadata and one
finite F32 `model.parameters[1700020]` tensor are also required. The frozen PPO38
hash links the legacy M24 model identity, never M25's identity.

## Fresh-session integration

With the feature enabled:

```rust,ignore
TrainingArtifact::initialize_selected_m24_u428_for_side_actors(
    directory: &Path, seed: u64, device: PolicyDevice,
) -> Result<(PolicyModel, String), CheckpointError>
```

This validates and expands on the host before creating the model/device, returning
the model and initialization provenance. Persist that provenance with the new run.
It delegates expansion to `crate::expand_m24_side_actor_parameters`; the first
1,700,020 parameter bits remain unchanged and 24 Dire actor tensors are appended
as copies of the Radiant actor. No optimizer or checkpoint manifest is read.

For an already-created fresh model, available with either feature configuration:

```rust,ignore
TrainingArtifact::load_initial_weights(
    model: &PolicyModel, directory: &Path,
) -> Result<(), CheckpointError>
```

Feature-on tries strict current metadata first; only exact SHA-recognized U428
bytes may use the legacy expansion path. Other errors retain the strict decoder's
specific error. Validation finishes before importing any model parameters.
Feature-off delegates to the existing strict runtime loader.

The caller must create a **fresh** trainer/Adam, zero training counters, new seeded
RNG streams, and fresh scheduler, curriculum, mastery and league state. Neither API
resets an existing trainer. Do not route resume, frozen historical opponents or
normal runtime loading through this initializer. M21 initialization explicitly
fails with side-actors before reading a path or creating a device.

`TrainingSession::initialize` uses the crate-private `initialize_from_weights`
factory only for fresh runs with `--initial-weights`. In an M25 build it validates
and expands before backend creation, then constructs the fresh trainer and RNG
streams. Current M25 weights still use strict current decoding. The M24 build
keeps its original fresh-model/runtime-load behavior. Resume never uses this path.

Build the architecture explicitly with `--features builtin,cuda,side-actors`;
the feature does not select a device, so CUDA execution also requires the ordinary
`--device cuda` argument. No new architecture CLI switch or implicit old-checkpoint
conversion is provided.

No old checkpoint, Adam or RNG data is read; no artifact is rewritten. In particular U124 and the
196-of-200 ensemble artifacts must remain untouched. No scored-game equivalence
or qualification is claimed.

Source regression tests are in `src/tests/side_actor_initialization.rs` and
`src/tests/side_actor_session.rs`, including a separately ignored actual-U428 gate.
They check failed imports before device creation, unchanged source files, and zero
Adam moments, counters and RNG draw counts in the fresh session. Integration
qualification additionally exercises the actual pinned U428 initializer, CPU/CUDA
native-side raw-head parity, mixed-side gradients and rollback, and a full-native
M4/B2/G2 micro256/reuse two-update resume where both actor heads change.

M24 and M25 suites are run separately. M24-specific autonomous ensemble/blend/
sharpen research tools, RND/memory screens, and their old identity goldens are not
silently generalized to M25. Feature-off retains their original coverage. M25 has
its own linked identity goldens, while shared checkpoint/body and gameplay contracts
remain tested with both architectures. This qualification is not a policy-strength
evaluation or authorization to mutate old campaigns.
