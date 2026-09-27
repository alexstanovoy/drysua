# Training concurrency decisions

Keep serial execution as the default. Retain opt-in C/host gradient folding;
retire A/early Continue and B/minibatch prefetch because measured full-update
benefit was absent. [Current options, evidence and probes](training_concurrency_prototypes.md)
replace the historical implementation diary, available at
`git show 2ddb68b:docs/training_concurrency.md`.

Optimize equal-work update time, not GPU utilization. Preserve batch membership,
RNG draws, ordered rollout insertion, parameter/Adam transactions and bounded
worker ownership. Smaller collection groups change trajectories; next-update
gameplay with old weights introduces policy lag. Neither is a transparent speedup.
See [collection measurements](training_performance.md) and
[execution safety](experiment-safety.md) before proposing another experiment.
