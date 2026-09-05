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

Use targeted release tests during development and one release suite before committing.
Do not run debug tests or repeat feature matrices without a specific reason. Use default
Cargo build and test parallelism; fix shared-state races instead of serializing the suite.

```text
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --all-targets --all-features --quiet
cargo machete
```

## Simplicity

- Reconsider optional machinery, duplicate paths, and designs based on weak assumptions
  as soon as they are noticed. Delete or simplify when evidence supports it; leave uncertain
  cases unchanged rather than replacing them with another speculative abstraction.
- Preserve correctness checks, seat-data boundaries, and reproducibility when simplifying.
- Keep all experiment artifacts under `artifacts/temp/`; release artifacts use
  `artifacts/vX.Y.Z/`. Stop and report before long release training.
