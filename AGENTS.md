# Engineering and regression contracts

- Keep control flow explicit, work bounded, and state ownership transactional.
  Preserve checkpoint model, optimizer, and RNG recovery together.
- Reproduce a behavior change with a failing regression test before fixing it.
  Prefer end-to-end contracts for HTTP, training, stop/resume, and persistence.
- Do not add tautological tests or tests that mirror private helper implementations.
  Parameterize repeated scenarios rather than copying setup and assertions.
- Keep focused unit tests for codec corruption, ordering, overflow, RNG, numerical
  boundaries, and real bug regressions that end-to-end tests cannot isolate.
- Replace useful coverage before deleting tests. Record removal categories and
  rationale; a line-count target never justifies losing a behavior contract.
- Measure identical workloads before claiming performance improvements. Keep
  checkpoint compatibility, bounded resources, and security validation intact.
- Do not start production training or change services during verification. Use
  the owner's authorized exclusive resource-bounded runner for heavy commands.
