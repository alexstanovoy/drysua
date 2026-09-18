//! Bounded Map2 complete-episode collection benchmark.
//!
//! Measures the production collector's per-decision work — world step, feature
//! encode, actor sampling and retained-interval value evaluation — as a
//! single-threaded baseline (one environment, serial path), as a sweep over
//! environment counts through the production worker pool, and as a pipelined
//! sweep over fixed collection groups, so per-core scaling and the barrier
//! collapse are visible directly.
//!
//! Every measured window is a fixed number of decision rounds over fresh,
//! identically seeded worlds. Construction (worlds, model, warmup) runs outside
//! the timed section; the slice is single-use, so each criterion iteration
//! measures the same work from the same initial state.
//!
//! One extra untimed phased window per case prints per-phase wall accounting
//! (`step`/`encode`/`forward`/`apply`/`barrier`/`wait` on stderr) for the
//! analysis script; criterion's timed cases never enable instrumentation.
//! `DRYSUA_BENCH_PHASES_ONLY=1` runs just the phase table;
//! `DRYSUA_BENCH_PHASES=1` adds it to a normal sweep.
//!
//! The library target is built by `cargo bench` under `[profile.bench]`, which
//! inherits the release profile (opt-level 3); drysua's `[profile.dev]` and
//! `[profile.test]` opt-level 2 overrides do not apply to this target.
//!
//! ```text
//! # Smoke: one untimed pass per case, about 30 s of wall time.
//! cargo bench -p drysua --features builtin --bench training -- --test
//!
//! # Full measurement with criterion's default 5 s target per case.
//! cargo bench -p drysua --features builtin --bench training
//!
//! # Bounded measurement for the guarded resource window.
//! cargo bench -p drysua --features builtin --bench training -- \
//!     --sample-size 10 --warm-up-time 1 --measurement-time 5
//!
//! # One case only, e.g. a pipelined ceiling.
//! cargo bench -p drysua --features builtin --bench training -- \
//!     --exact "collection_pipelined/groups_2/26" \
//!     --sample-size 10 --warm-up-time 1 --measurement-time 5
//!
//! # CUDA learner forward, opt-in and separate from the per-core story.
//! DRYSUA_BENCH_DEVICE=cuda cargo bench --features builtin,cuda --bench training
//! ```

use std::hint::black_box;
use std::time::Instant;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use drysua::{
    FEATURE_SCHEMA_VERSION, MAP2_ACTOR_DECISIONS, MAP2_DECISION_INTERVAL_TICKS,
    MAP2_RETENTION_STRIDE, MAX_TRAINING_ENVIRONMENTS, MODEL_PARAMETER_COUNT, MODEL_SCHEMA_VERSION,
    PolicyDevice, TrainingCollectionSlice, TrainingCollectionSliceConfig,
};

/// Fixed master seed for every measured window; changing it invalidates the table.
const BENCH_SEED: u64 = 10_141_700;

/// Fixed update index seeding the opponent schedule and retention phases.
const BENCH_UPDATE: u64 = 0;

/// Measured decision rounds per window; a multiple of the retention stride so
/// each stream closes exactly the same number of retained intervals.
const BENCH_ROUNDS: usize = 64;

/// Untimed decision rounds before the measured window. The 900-tick Map2
/// pregame ends at round 300, so 320 opens the measured window just after the
/// horn at tick 961, where creeps and economy have started.
const BENCH_WARMUP_ROUNDS: usize = 320;

/// Environment counts under test: one is the serial baseline, even counts from
/// two to the training maximum use the production worker pool.
const PARALLEL_ENVIRONMENTS: [usize; 6] = [2, 4, 6, 8, 16, 26];

/// Environment counts of the pipelined sweep, the range the owner asked for.
const PIPELINED_ENVIRONMENTS: [usize; 3] = [8, 16, 26];

/// Fixed collection group counts of the pipelined sweep, versus one global barrier.
const PIPELINED_GROUPS: [usize; 2] = [2, 4];

// Compile-time guards pin the benchmark's interpretation of the library
// constants; drift fails the bench build instead of a timed run.
const _: () = assert!(FEATURE_SCHEMA_VERSION == 22);
const _: () = assert!(MODEL_SCHEMA_VERSION == 24);
const _: () = assert!(MODEL_PARAMETER_COUNT == 1_700_020);
const _: () = assert!(MAP2_ACTOR_DECISIONS == 9_300);
const _: () = assert!(MAP2_RETENTION_STRIDE == 8);
const _: () = assert!(MAX_TRAINING_ENVIRONMENTS == 26);
const _: () = assert!(BENCH_ROUNDS > 0);
const _: () = assert!(BENCH_ROUNDS.is_multiple_of(MAP2_RETENTION_STRIDE));
const _: () = assert!(BENCH_WARMUP_ROUNDS + BENCH_ROUNDS <= MAP2_ACTOR_DECISIONS);
const _: () = assert!(BENCH_ROUNDS / MAP2_RETENTION_STRIDE <= drysua::MAP2_RETAINED_DECISIONS);
const _: () = assert!(PARALLEL_ENVIRONMENTS[PARALLEL_ENVIRONMENTS.len() - 1] == 26);
const _: () = {
    let mut index = 0;
    while index < PARALLEL_ENVIRONMENTS.len() {
        assert!(PARALLEL_ENVIRONMENTS[index] > 1);
        assert!(PARALLEL_ENVIRONMENTS[index].is_multiple_of(2));
        index += 1;
    }
};
const _: () = assert!(PIPELINED_GROUPS[PIPELINED_GROUPS.len() - 1] == 4);
const _: () = {
    let mut index = 0;
    while index < PIPELINED_GROUPS.len() {
        assert!(matches!(PIPELINED_GROUPS[index], 2 | 4));
        index += 1;
    }
};
const _: () = {
    let mut index = 0;
    while index < PIPELINED_ENVIRONMENTS.len() {
        assert!(PIPELINED_ENVIRONMENTS[index].is_multiple_of(2));
        assert!(PIPELINED_ENVIRONMENTS[index] / 2 >= PIPELINED_GROUPS[PIPELINED_GROUPS.len() - 1]);
        index += 1;
    }
};

criterion_group!(benches, collection);
criterion_main!(benches);

fn collection(c: &mut Criterion) {
    let device = bench_device();
    // The phase table is its own bounded run: `DRYSUA_BENCH_PHASES_ONLY=1`
    // skips criterion entirely, while `DRYSUA_BENCH_PHASES=1` adds it to a
    // normal run outside every timed case.
    let phases_only = std::env::var("DRYSUA_BENCH_PHASES_ONLY").is_ok();
    if phases_only {
        phase_accounting(device);
    }
    if !phases_only {
        serial_baseline(c, device);
        parallel_sweep(c, device);
        pipelined_sweep(c, device);
    }
    if !phases_only && std::env::var("DRYSUA_BENCH_PHASES").is_ok() {
        phase_accounting(device);
    }
}

/// One environment on one thread: the per-decision reference for scaling.
fn serial_baseline(c: &mut Criterion, device: PolicyDevice) {
    verify_window(1, 1, device);
    let mut group = c.benchmark_group("collection_serial");
    group.throughput(Throughput::Elements(BENCH_ROUNDS as u64));
    group.bench_function("single-thread", |bencher| {
        bencher.iter_batched_ref(
            || slice(1, 1, device),
            |slice| black_box(slice.run(BENCH_ROUNDS).expect("serial collection window")),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// The production worker pool over the offered environment counts.
fn parallel_sweep(c: &mut Criterion, device: PolicyDevice) {
    let mut group = c.benchmark_group("collection_parallel");
    for environments in PARALLEL_ENVIRONMENTS {
        verify_window(environments, 1, device);
        let decisions = (BENCH_ROUNDS * environments) as u64;
        group.throughput(Throughput::Elements(decisions));
        group.bench_with_input(
            BenchmarkId::new("environments", environments),
            &environments,
            |bencher, &environments| {
                bencher.iter_batched_ref(
                    || slice(environments, 1, device),
                    |slice| black_box(slice.run(BENCH_ROUNDS).expect("parallel collection window")),
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// Fixed collection groups over the environment counts the owner cares about.
fn pipelined_sweep(c: &mut Criterion, device: PolicyDevice) {
    let mut group = c.benchmark_group("collection_pipelined");
    for groups in PIPELINED_GROUPS {
        for environments in PIPELINED_ENVIRONMENTS {
            verify_window(environments, groups, device);
            let decisions = (BENCH_ROUNDS * environments) as u64;
            group.throughput(Throughput::Elements(decisions));
            group.bench_with_input(
                BenchmarkId::new(format!("groups_{groups}"), environments),
                &environments,
                |bencher, &environments| {
                    bencher.iter_batched_ref(
                        || slice(environments, groups, device),
                        |slice| {
                            black_box(
                                slice
                                    .run(BENCH_ROUNDS)
                                    .expect("pipelined collection window"),
                            )
                        },
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
    group.finish();
}

/// Prints one phased untimed window per default/pipelined case.
///
/// The phased run is instrumentation, not a timed criterion case: the analysis
/// reads the printed line to attribute the window to encode, world step,
/// forward, apply, barrier and evaluator waits. It also prints the measured
/// wall so phase sums can be compared against the same window's total.
fn phase_accounting(device: PolicyDevice) {
    for groups in [1, PIPELINED_GROUPS[0], PIPELINED_GROUPS[1]] {
        for environments in PIPELINED_ENVIRONMENTS {
            if environments / 2 < groups {
                continue;
            }
            let mut slice = slice(environments, groups, device);
            let started = Instant::now();
            let (report, phases) = slice.run_phased(BENCH_ROUNDS).expect("phased window");
            let wall_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            assert_window(&report, environments, groups);
            eprintln!(
                "phase-accounting: groups={groups} environments={environments} decisions={} wall_ns={wall_ns} prepare_ns={} advance_ns={} forward_ns={} barrier_ns={} apply_ns={} flush_wait_ns={} evaluator_ns={}",
                report.decisions,
                phases.prepare_ns,
                phases.advance_ns,
                phases.forward_ns,
                phases.barrier_ns,
                phases.apply_ns,
                phases.flush_wait_ns,
                phases.evaluator_ns,
            );
        }
    }
}

/// Runs one untimed window per case and asserts the fixed work identity, so a
/// drifted setup fails before any measurement is recorded.
fn verify_window(environments: usize, groups: usize, device: PolicyDevice) {
    let mut slice = slice(environments, groups, device);
    let report = slice.run(BENCH_ROUNDS).expect("verification window");
    assert_window(&report, environments, groups);
}

fn assert_window(
    report: &drysua::TrainingCollectionSliceReport,
    environments: usize,
    groups: usize,
) {
    assert!(environments <= MAX_TRAINING_ENVIRONMENTS);
    assert!((groups == 1) || (groups <= environments / 2));
    assert_eq!(report.environments, environments);
    assert_eq!(report.warmup_rounds, BENCH_WARMUP_ROUNDS);
    assert_eq!(report.rounds, BENCH_ROUNDS);
    assert_eq!(report.decisions, (BENCH_ROUNDS * environments) as u64);
    assert_eq!(
        report.ticks,
        report.decisions * u64::from(MAP2_DECISION_INTERVAL_TICKS)
    );
    assert_eq!(report.start_tick, expected_start_tick());
    assert_eq!(
        report.end_tick,
        report.start_tick + BENCH_ROUNDS as u32 * MAP2_DECISION_INTERVAL_TICKS
    );
    assert!(report.retained_samples > 0);
}

fn slice(environments: usize, groups: usize, device: PolicyDevice) -> TrainingCollectionSlice {
    TrainingCollectionSlice::new(
        TrainingCollectionSliceConfig {
            seed: BENCH_SEED,
            environments,
            update: BENCH_UPDATE,
            warmup_rounds: BENCH_WARMUP_ROUNDS,
            pipeline_groups: groups,
        },
        device,
    )
    .expect("bounded collection slice")
}

/// The first snapshot tick is one; each round advances three ticks.
const fn expected_start_tick() -> u32 {
    1 + BENCH_WARMUP_ROUNDS as u32 * MAP2_DECISION_INTERVAL_TICKS
}

/// `cpu` by default; `cuda` requires a build with the `cuda` feature.
fn bench_device() -> PolicyDevice {
    match std::env::var("DRYSUA_BENCH_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => PolicyDevice::Cpu,
        Ok("cuda") => cuda_device(),
        Ok(other) => panic!("unsupported DRYSUA_BENCH_DEVICE {other:?}; expected cpu or cuda"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_device() -> PolicyDevice {
    PolicyDevice::Cuda { ordinal: 0 }
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> PolicyDevice {
    panic!("DRYSUA_BENCH_DEVICE=cuda requires a build with the cuda feature");
}
