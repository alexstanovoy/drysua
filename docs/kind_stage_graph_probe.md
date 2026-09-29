# Bounded kind-prefix stage replay experiment

This source-only prototype targets **host CUDA submissions between sampling
barriers**, not a few scalar copies. It is compiled only for tests with CUDA and
`side-actors`, has no CLI enablement, and cannot be installed in production training.
No new dependency, unsafe code, parameter layout, schema hash or checkpoint format
is introduced. The active frozen trainer is unaffected.

## Implemented boundary

`KindStage` owns one B20 stage for one admitted model/device/owner thread:

1. Refresh fixed trunk, U32 kind-index and U8 side-mask buffers outside cache guards.
2. Replay shared kind-embedding gather, original context concatenation, both
   Controlled/Learn actor-head pairs, side selection and packed-output copies.
3. Read one 480-element F32 buffer, validate required families, then use the
   existing CPU decoder/Gumbel/RNG functions.

The pure raw-pair and kind-context operations are shared with eager inference;
there is no matrix decomposition, row compaction, padding or changed GEMM shape.
The prototype's full-sampling reference wrapper replaces only the kind-prefix
stage; encoder, base, unit and slot stages remain eager.

The packed layout is part-major:

| Block | F32 range |
|---|---|
| Controlled Radiant | 0..40 |
| Controlled Dire | 40..80 |
| Learn Radiant | 80..200 |
| Learn Dire | 200..320 |
| Controlled selected | 320..360 |
| Learn selected | 360..480 |

CPU logical-needed bits come from the same predicate as eager inference. If
neither family is needed, there is no replay or readback. A required family checks
both raw branches over the full batch, in eager order. A completely unused family
is ignored even if speculative GPU calculation produced Inf/NaN. No blanket finite
scan is applied to the packed buffer. Controlled sampling remains at its existing
point; Learn draws remain deferred to final decoding.

This preserves the intended numerical/error contract on valid backend execution.
Computing an unused family can still encounter a backend/allocation error that the
eager skipped path would avoid. Such failures terminate this isolated experiment;
the probe is not a claim of general production recovery equivalence.

## Bounds and ownership

- One irreversible process admission, shared with the existing encoder experiments.
- A distinct `KindPrefixB20` purpose; an encoder admission does not authorize it.
- One model, one DeviceId, one shape, two guarded warmups, one capture, at most
  128 graph replays. No registry, shape discovery, eviction or recapture.
- Stable parameter storage is pinned, and in-place weight updates remain visible.
- The workspace is thread-bound and returns only owned host results.
- Replay/readback complete under the caller's model read lock. Checked finish
  destroys the graph before its buffers; forgetting finish retires the test process.
- Capture/backend errors terminate the owned process, with no retry or eager recovery.

The audited stage introduces at most these five metadata keys:

```text
[20,1]
[20,2,2,1,0,1]
[20,6,6,1,0,1]
[20,2,1,0,2,1,2,1]
[20,6,1,0,6,1,6,1]
```

That is 30 `usize`s, **240 logical device bytes** on 64-bit, plus key-vector
contents and metadata overhead. Changing indices/masks never enter the cache as
host data. Expected host-data cache entries: zero.

Fixed buffers total **22,500 bytes**: trunk 20,480; indices 80; side mask 20;
packed output 1,920. Listed capture-local tensor outputs add approximately
38,400 bytes. These are **not total VRAM bounds**: CUDA context/module resources,
cuBLAS workspace, graph allocation nodes, allocator granularity and events are
excluded. Use the existing exclusive process resource limits; measure actual
resident and peak memory before admitting more graphs or models.

## Deferred verification

No compilation, tests or GPU measurements were run by the source owner. When an
exclusive GPU window is authorized, run this filter alone in a fresh process with
CUDA, `side-actors`, and preferably `builtin` enabled:

```text
model::cuda_graph_probe::kind_stage::tests::cuda_kind_prefix_graph_preserves_dynamic_inputs_weights_errors_and_rng
--exact --ignored --nocapture
```

It compares eager/graph bits with changing prefixes, masks, trunk data and weights;
checks skipped versus required overflowing heads and input bounds; and, with
`builtin`, checks native sampled actions, statistics, frames, targets, policy
identity and exact RNG states. Three CPU-only tests check input and packed-output
contracts without capture.

`kind-stage-measure` reports 20 eager and 20 replayed prefix stages. The eager
reference keeps its trunk and routing mask resident; replay timing includes its
input copies and packed readback. Warmup/capture setup is reported separately.
CUDA event intervals include enqueue gaps; they are not isolated kernel time.
These numbers qualify **this stage only**, not an update or the entire pipeline.

## Whole-pipeline next gate

The four-stage endpoint is encoder+value/kind, kind-prefix, unit-prefix and
slot-prefix, retaining the current CPU sampling barriers. This experiment tests
the dynamic GPU-prefix input and logical finite-validation mechanism needed for
that endpoint. It does not yet reduce an entire actor step to four launches.

Before any broader implementation:

1. Establish the integrated NN-opponent-batching eager baseline on the exact same
   learner/opponent weights, seeds, batch sizes, traces and retained rows.
2. Qualify this stage's exactness and measure actual launch, metadata-transfer,
   allocation and readback counts, plus memory. Do not infer counts from time alone.
3. Audit/cache-budget each additional tensor-only stage separately. Do not multiply
   the old encoder bound and call that a proof for different operations.
4. Only then consider an explicitly bounded two-model context. Fixed owner/device
   bindings and a hard capture-job budget are required; no blind 1..64 shape bank.
5. Compare complete update wall time, identical Adam work, CPU simulation time,
   GPU busy-time distribution and queue/barrier stalls, with setup included.

Neither 100% hardware utilization nor an end-to-end speedup is claimed. Remaining
CPU simulation, sampling dependencies and small-batch GEMMs must be measured rather
than hidden behind a successful microbenchmark.
