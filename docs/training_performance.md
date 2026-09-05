# Training throughput experiments

Production now batches actor sampling and greedy warmup on the learner device.
Actor collection and optimization remain sequential: each rollout binds the
current model identity, which the direct trainer validates before optimization.
Production does not publish CPU snapshots or send rollouts through loopback channels.
Each update derives bounded
per-environment RNG streams from the checkpointed master RNG.

The rollout limit is 2048 decisions per environment, with an unchanged global
limit of 32768 transitions. Environments remain disposable at update boundaries,
so resume requires no simulator-state serialization.

## Measurements

RTX 5090, F32, one PPO epoch; elapsed times include command startup/build overhead.
All runs are retained under `artifacts/temp/`.

| Experiment | Environments | Rollout | Updates | Samples | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| v33 initialized baseline | 4 | 256 | 8 | 8192 | 211.3 |
| Fresh scalar actor | 4 | 2048 | 8 | 65536 | 665.7 |
| Fresh CPU batched actor | 4 | 2048 | 8 | 65536 | 480.8 |
| Fresh CPU batched actor | 16 | 2048 | 2 | 65536 | 351.5 |
| Fresh CPU batched actor and warmup | 16 | 2048 | 2 | 65536 | 253.1 |
| Fresh CUDA batched actor and warmup | 16 | 2048 | 2 | 65536 | 79.0 |

These are exploratory end-to-end runs, not controlled same-trajectory speedups.
Initial weights, RNG scheduling, legality masks and optimization work differ
between some rows. The last run applied only two optimizer steps: KL early
stopping prevented the remaining minibatches. Its roughly 830 retained samples/s
must not be extrapolated as a four-epoch release-training guarantee.

The former launcher used four environments and eight decisions per update.
Across the full phase/baseline cycle it discarded 30600 warmup decisions while
retaining only 256 transitions. Longer rollouts amortize this work; batching
removes repeated feature-trunk inference and device transfer overhead.

Before release training, verify throughput and gameplay retention from an accepted
behavioral anchor with the actual epoch/minibatch settings. Passing the throughput
experiment does not qualify its random-policy weights for deployment.
