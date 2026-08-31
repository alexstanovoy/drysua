# drysua

Private Shadow Fiend competition bot for the sibling `../bota` simulator. The development
plan and architecture decisions live in `docs/plan.md`.

## Boundaries

- Policy, teacher, reward, datasets, and metrics use only seat-specific protocol data and
  bounded local history.
- The optional `builtin` feature may run `bota-server` in process, but model inputs never
  read `World`, server components, RNG state, match seeds, or numeric entity IDs.
- All Candle operations live in `src/model.rs`.
- New dependencies require discussion. Every dependency disables default features.
- Keep all queues, buffers, histories, batches, and loops explicitly bounded.
- `lib.rs` and test `mod.rs` files contain only module declarations and re-exports.
- Tests live under `src/tests/`; no root integration-test directory.

## Checks

```text
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --release --all-targets --all-features
cargo test --all-targets --no-default-features
cargo test --release --all-targets --no-default-features
cargo machete
```
