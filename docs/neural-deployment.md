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
shared with Teacher, which retains its existing behavior.

After independent model qualification, `DefaultDeployment` accepts
`PlayPolicy::Neural` with `Some("artifacts/<qualified-directory>")`; compile-time
checks require weights and resolution rejects paths outside the artifact tree.
No default promotion is part of this change.
