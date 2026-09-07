# Pure neural deployment

The maintained zero-argument launcher still selects `DEFAULT_DEPLOYMENT` (currently
Teacher). Adding Neural does not qualify a model or change the selected default.

```sh
cargo run --release --bin drysua -- play --policy neural \
  --weights-directory /absolute/path/to/weights --addr 127.0.0.1:4455
```

The directory must contain `drysua.weights.safetensors` written for the current
model architecture and runtime metadata. Explicit Neural requires a weights path;
loading failure is fatal to this invocation, before connecting, with no fallback.
Do not relabel an older artifact's metadata to make it load. Generate/export new
weights through `TrainingArtifact::save_runtime_weights` and retain their identity.

Public Rust API:

```text
drysua::play_neural(address: &str, name: &str, limit: Option<u32>,
                    weights_directory: &Path) -> io::Result<Outcome>
drysua::play_neural_on(wire: &mut impl Wire, seated: Seated, limit: Option<u32>,
                       model: &PolicyModel) -> io::Result<Outcome>
```

Both maps use `PolicyModel::choose`, with tensor operations remaining in model.rs.
No Teacher is constructed or invoked: no economy, pregame buy/learn, emergency
retreat, channel protection, or Map0 fallback. The model may itself select legal
buy/learn actions or interrupt a channel. ActionSpace legality masks, item
readiness, body tracking, rejection rollback, order deduplication, bounded message
processing, Snapshot/Events completion, and decisions at ticks 1, 4, 7, … remain
shared with the historical modes. Hybrid, Tactical, and Teacher retain their
existing behavior.

After independent model qualification, `DefaultDeployment` accepts
`PlayPolicy::Neural` with `Some("artifacts/<qualified-directory>")`; compile-time
checks require weights and resolution rejects paths outside the artifact tree.
No default promotion is part of this change.

## Evaluation identities

Historical release crossplay remains Map1, with both seats for each registry seed
(20 games per opponent for the current ten seeds):

```sh
python3 scripts/release_crossplay.py \
  --candidate-binary "$PWD/target/release/drysua" \
  --candidate-policy neural --candidate-weights /absolute/path/to/weights \
  --candidate-metadata 'EXACT BUILD COMMAND AND SOURCE/DIRTY STATE' \
  --run-name neural-map1
```

The **separate** `--teacher-challenge-map0` now requires an explicit frozen
`--baseline-manifest`; it never uses the candidate binary's Teacher. See
[the fixed-Teacher challenge](map0-fixed-teacher-challenge.md) for the pinned
4096-bound runtime, development cohort, and predeclared 100-game final cohort.
This challenge does not load or modify the historical Map1 registry and cannot
substitute for that release gate.

Future registry releases may use `policy: "neural"` and the existing weights
contract: `path: "artifacts/<tag>"` plus a lowercase SHA256 of
`drysua.weights.safetensors`. Evaluation snapshots the bounded, non-symlink file
and verifies binary and weight identities after all games.

The automated real-TCP smoke test serializes a fresh untrained model (seed 70010)
with current metadata, plays short matches on both maps, and deletes the temporary
artifact. This verifies deployment plumbing only, not strength, qualification,
or a release-gate pass. No human-strength claim is made.
