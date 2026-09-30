use std::time::Duration;

use super::*;

const STAGES: [TrainingStage; 5] = [
    TrainingStage::RolloutInitialization,
    TrainingStage::Collection,
    TrainingStage::BatchPreparation,
    TrainingStage::Optimization,
    TrainingStage::Finalization,
];

const INVALID_CASES: [&str; 10] = [
    "unfinished",
    "missing",
    "counter",
    "wall",
    "twice",
    "after",
    "backwards",
    "skipped",
    "stage_overflow",
    "total_overflow",
];

#[test]
fn only_valid_completed_updates_forward_exact_durations_and_counters() {
    for mode in [TrainingUpdateMode::Annealed] {
        for (nanos, outcome) in [
            ([999_999_999, 2, 3, 4, 5], TrainingTimingOutcome::Complete),
            ([0; 5], TrainingTimingOutcome::Complete),
            ([0; 5], TrainingTimingOutcome::Error),
            ([0; 5], TrainingTimingOutcome::Incomplete),
            ([0; 5], TrainingTimingOutcome::Complete.on_scope_exit(true)),
        ] {
            let durations = nanos.map(Duration::from_nanos);
            let elapsed = durations.into_iter().sum();
            let mut timing = TrainingUpdateTiming::new(u64::MAX, mode, 0);
            for (stage, duration) in STAGES.into_iter().zip(durations) {
                timing.record(stage, duration);
            }
            timing.set_samples(0);
            let steps = if nanos == [0; 5] { 0 } else { u64::MAX };
            timing.set_optimizer_step(steps);
            timing.finish(elapsed, outcome);
            let mut forwarded = None;
            timing.forward_metrics(|index, elapsed, stages, valid| {
                forwarded = Some((index, elapsed, stages, valid));
            });
            assert_eq!(
                forwarded,
                (outcome == TrainingTimingOutcome::Complete).then_some((
                    u64::MAX,
                    elapsed,
                    durations.map(Some),
                    true
                ))
            );
            let text = timing.to_string();
            assert!(text.contains(&format!("elapsed_ns={}", elapsed.as_nanos())));
            assert!(text.contains(&format!("samples=0 optimizer_steps={steps}")));
            assert!(text.ends_with("timing_valid=true"));
        }
    }
}

#[test]
fn invalid_or_unfinished_updates_never_publish_metrics() {
    for case in INVALID_CASES {
        let mut timing = TrainingUpdateTiming::new(7, TrainingUpdateMode::Annealed, 100);
        let overflow = case.ends_with("overflow");
        for (index, stage) in STAGES.into_iter().enumerate() {
            if case == "missing" && index == 4 {
                break;
            }
            let duration = if overflow && index == 0 {
                Duration::MAX
            } else {
                Duration::from_nanos(1)
            };
            timing.record(stage, duration);
            if index == 1 && matches!(case, "backwards" | "skipped") {
                let invalid = if case == "backwards" {
                    STAGES[0]
                } else {
                    STAGES[4]
                };
                timing.record(invalid, Duration::from_nanos(100));
            }
            if case == "stage_overflow" && index == 0 {
                timing.record(stage, Duration::from_nanos(1));
            }
        }
        if case == "counter" {
            timing.set_optimizer_step(99);
        }
        let elapsed = if overflow {
            Duration::MAX
        } else {
            Duration::from_nanos(if case == "wall" { 4 } else { 5 })
        };
        if case != "unfinished" {
            timing.finish(elapsed, TrainingTimingOutcome::Complete);
        }
        if case == "twice" {
            timing.finish(Duration::ZERO, TrainingTimingOutcome::Complete);
        }
        if case == "after" {
            timing.record(STAGES[4], Duration::from_nanos(100));
        }
        timing.forward_metrics(|_, _, _, _| panic!("forwarded {case}"));
        let text = timing.to_string();
        assert!(text.ends_with("timing_valid=false"), "{case}: {text}");
        if overflow {
            assert!(text.contains(&format!("measured_ns={}", Duration::MAX.as_nanos())));
            assert!(text.contains("duration_overflow=true"));
            assert!(text.contains("collection_ns=1"));
        }
        if matches!(case, "backwards" | "skipped" | "after" | "twice") {
            assert!(
                text.contains("elapsed_ns=5 measured_ns=5"),
                "{case}: {text}"
            );
        }
        if case == "counter" {
            assert!(text.contains("optimizer_steps=unknown"));
        }
        if case == "missing" {
            assert!(
                text.contains("finalization_ns=unknown samples=unknown optimizer_steps=unknown")
            );
        }
    }
}
