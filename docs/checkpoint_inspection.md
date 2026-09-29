# Native checkpoint inspection v1

```text
drysua checkpoint-inspect --checkpoint-directory DIR
drysua checkpoint-inspect --contract
```

Exactly one input mode is required. There are no resume, repair, device-selection,
training, or output-file options. Success writes one UTF-8 JSON document followed
by a newline to stdout. Failure writes no inspection JSON; the normal CLI prints
the error on stderr and exits nonzero. Output, including its newline, is bounded
to 4 MiB. No model/device is constructed and no optimizer operation occurs.

The command never creates directories, locks, metrics, snapshots or checkpoint
files, and never repairs exports, prunes generations or syncs directory state.
It does not authorize stopping or modifying an active training process.

## Build and platform contract

Both modes return `schema: "drysua-checkpoint-inspection/v1"`.
`--contract` works without filesystem access and returns:

- `kind: "contract"`;
- `model: {version, hash, parameters}` from this build;
- `enabled_features`, with the existing exact feature-string semantics;
- `capabilities`: `inspection` (Linux), `annealed_history` (`builtin` compiled),
  `read_only: true`, `strict_build_features: true`, and
  `controller_run_kind: "train-annealed"`;
- `limits`: `max_json_bytes=4194304`, `max_snapshots=10000`, `max_files=10004`,
  native manifest/training/runtime byte limits, and `snapshot_bytes=4096`;
- `schemas`: action, feature and reward version/hash pairs plus rules audit version;
- `profiles`: three objects, each containing `sample_budget`, native `ppo`
  version/hash, `fixed_checkpoint` version/hash, `adaptive_checkpoint` version/hash,
  `max_samples` and `max_games`;
- `numeric_semantics`, documenting float, hash and adaptive-unit representations.

Schema hashes are **16-digit lowercase hexadecimal strings**, not JSON floating
numbers. SHA-256 hashes are 64-digit lowercase hexadecimal strings. Sample budget
objects are `{name, code}` with this JSON protocol's stable mapping:
`standard/0`, `annealed-v1/1`, `wide-annealed-v1/2`. These codes do not expose binary
manifest offsets or imply an on-disk enum representation.

Use a binary with the checkpoint's exact enabled features and model/schema
identities. Model24/model25 and incompatible profiles are not migrated or guessed.
A CUDA-enabled inspection binary does not construct a CUDA device. Removing its
CUDA build feature is not a substitute for matching checkpoint provenance.

Directory inspection currently requires Linux descriptor-anchored no-follow I/O.
Other platforms receive an explicit unsupported error; `--contract` advertises
that limitation. Without `builtin`, generic metadata inspection remains available,
but an annealed history request fails explicitly—even for an empty history.

## Inspection document

`kind` is the exact recognized operation `train-annealed`, `train-full`, or `train`;
unrecognized operations are `other`. Controllers for annealed campaigns must reject
other kinds rather than treating absent history as verified.

Fields:

- `identity`: `manifest_sha256`, authoritative `tensor_sha256`, nullable
  `runtime_sha256`, and `scope_sha256`.
- `model`: `{version, hash, parameters}`.
- `checkpoint`: native `{version, hash}`, including the adaptive identity when present.
- `progress`: `updates`, `optimizer_steps`, `rollout_samples`, `games`,
  `policy_version`, `scheduler_step`, `curriculum_stage`, `best_evaluation`,
  `rng_states: [{name,state,draws}]`, `shuffle_rng: {state,draws}`,
  `league_references`, and `mastery_present`.
  `games` is exactly `updates * games_per_update` for verified annealed scope and
  **null** for other operations; no generic episode count is invented.
- `run`: `git_commit`, `simulator_commit`, `enabled_features`, `command_line`,
  `run_seed`, `map`, `hero`, `device: {kind,ordinal}`, `batch_size`,
  `rules_audit_version`, and `mastery_config_present`. CPU ordinal is null.
- `ppo`: `sample_budget`, `schema_version`, `schema_hash`,
  `decision_interval_ticks`, `rollout_decisions`, `environments`, `epochs`,
  `minibatch`, and all eleven floating hyperparameters: `clip_epsilon`,
  `value_coefficient`, `entropy_coefficient`, `learning_rate`, `adam_beta1`,
  `adam_beta2`, `adam_epsilon`, `gradient_clip`, `gamma_tick`, `gae_lambda`,
  `target_kl`. JSON floats are the stored F32 values widened exactly to F64;
  `ppo.f32_bits` additionally maps these eleven names to their exact U32 bits.
- `adaptive`: null, or `{config,limits,state,snapshot_count,snapshot_hash}`.
  Config fields: `success_updates`, `success_rate_units`, `poor_updates`,
  `poor_rate_units`, `extension_units`; rates use integer millionths.
  Limits: `base_updates`, `total_updates`, `zero_updates`.
  State: `generation`, `start_update`, `updates_in_generation`, `success_streak`,
  `poor_streak`, `extension_awards`.
- `history`: `{kind,verified,snapshot_count}`. Kind is `fixed`, `adaptive`, or
  `unsupported`; unsupported history always has `verified=false`.
- `runtime_status`: `matched`, `missing`, or `mismatch`;
  `runtime_matches_model` is true only for `matched`.
- `recovery_required`: true for a previous-manifest/tensor fallback or an unmatched
  runtime export. This is a diagnostic, not permission to recover automatically.
- `sources`: actual relative `manifest` and `tensor` paths, nullable observed
  `runtime` path, and `canonical_tensor_matches`.
- `files`: ordered `{path,size,sha256}` entries for the selected manifest,
  authoritative tensor, matching canonical tensor alias when separate, matching
  runtime export, and committed generation snapshots. Paths are restricted relative
  filenames or `domain-randomization/<filename>`; never absolute or traversing.

`scope_sha256` hashes the domain `drysua-checkpoint-inspection-scope/v1\0`, then the
native checkpoint identity, linked schemas, encoded run and encoded PPO config.
It excludes progress, optimizer step and tensor values. Python must use this field,
not reconstruct the native encoding or parse manifest offsets.

## Validation and crash recovery

The reader uses existing native manifest/tensor decoders and full artifact
validation. This enforces current identities, tensor names/shapes/dtype/checksum,
finite parameters and moments, optimizer/RNG bounds, and control-plane consistency.
It intentionally does not call the pathname-reopening loader or restore a model:
inspection reads are anchored to owned Linux directory descriptors.

The immutable `checkpoint.<tensor-sha>.safetensors` file named by the manifest is preferred. A canonical
tensor alias is listed only when its bytes hash to that same commitment. Native
`.previous` fallbacks are reported under their actual names when the primary is
absent; an unsafe existing primary is an error, not absence.

Runtime export is not authoritative checkpoint state. A missing export or a valid
older export returns false with `missing`/`mismatch`; its observed SHA is retained
when present. A runtime metadata/schema mismatch is also classified `mismatch`.
An export claiming the current metadata but having malformed tensors/nonfinite
values is an error. Symlinks, nonregular files and oversized files are errors in
all cases. Unmatched runtime files are **excluded** from the adoption inventory.
Normal job acceptance must require `runtime_matches_model=true`; only an explicit
controller recovery policy may proceed without that export. Inspection never writes
a replacement export.

Fixed annealed scope reads exactly one canonical `--updates`, `--games`,
`--generation-games`, and `--zero-updates`. Missing/duplicate/noncanonical counters
are rejected. Only the committed prefix is read, using bounded no-follow reads and
the existing native draw/canonical-render functions. Adaptive inspection verifies
the existing native snapshot contract and binds the inventoried bytes to the stored
rolling SHA-256. Pending or orphan snapshots are not inventoried or materialized.
The 10,000-generation cap is checked **before** snapshot loops or tensor loading.

Manifest bytes and selected manifest path are checked before/after the complete
operation, even if payload inspection fails. Directory replacements and detected
changes return `checkpoint changed during inspection`; there is no retry loop.
This is not a lease on future filesystem state. An adopting controller must verify
every copied file against the returned SHA and re-inspect the completed owned copy.

## Deferred verification

No builds or tests were run by the source owner while training was active.
After natural completion, use the owner's authorized runner for the full repository
checks plus filters `checkpoint::inspection::tests` and
`cli::checkpoint_inspection_tests`, with and without `builtin` and with the relevant
model feature. Tests use owned fixtures/native serializers, not live checkpoints
or model/device construction.
