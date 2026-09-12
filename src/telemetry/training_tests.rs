use std::cell::Cell;
use std::io::{self, Cursor, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::telemetry::{
    TrainingScopeTiming, TrainingStage, TrainingTimingOutcome, TrainingTimingScope,
    TrainingUpdateMode, TrainingUpdateTimer, TrainingUpdateTiming, time_training_checkpoint,
    time_training_scope, write_training_timing,
};

const STAGES: [TrainingStage; 5] = [
    TrainingStage::RolloutInitialization,
    TrainingStage::Collection,
    TrainingStage::BatchPreparation,
    TrainingStage::Optimization,
    TrainingStage::Finalization,
];

#[test]
fn complete_update_formats_exact_measured_stages_and_applied_optimizer_delta() {
    let mut timing = TrainingUpdateTiming::new(7, TrainingUpdateMode::CompleteEpisodes, 100);
    for (stage, milliseconds) in STAGES.into_iter().zip(1..=5) {
        timing.record(stage, Duration::from_millis(milliseconds));
    }
    timing.set_samples(32);
    timing.set_optimizer_step(103);

    timing.finish(Duration::from_millis(16), TrainingTimingOutcome::Complete);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=7 mode=complete_episodes ",
            "outcome=complete elapsed_ns=16000000 measured_ns=15000000 ",
            "rollout_initialization_ns=1000000 collection_ns=2000000 ",
            "batch_preparation_ns=3000000 optimization_ns=4000000 finalization_ns=5000000 ",
            "samples=32 optimizer_steps=3 last_stage=finalization ",
            "duration_overflow=false timing_valid=true",
        )
    );
}

#[test]
fn reset_window_zero_durations_and_counters_are_measured_not_unknown() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 9);
    for stage in STAGES {
        timing.record(stage, Duration::ZERO);
    }
    timing.set_samples(0);
    timing.set_optimizer_step(9);

    timing.finish(Duration::ZERO, TrainingTimingOutcome::Complete);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=0 mode=reset_window ",
            "outcome=complete elapsed_ns=0 measured_ns=0 ",
            "rollout_initialization_ns=0 collection_ns=0 batch_preparation_ns=0 ",
            "optimization_ns=0 finalization_ns=0 samples=0 optimizer_steps=0 ",
            "last_stage=finalization duration_overflow=false timing_valid=true",
        )
    );
}

#[test]
fn nanosecond_boundary_preserves_integer_precision_and_stage_order() {
    let mut timing = TrainingUpdateTiming::new(1, TrainingUpdateMode::ResetWindow, 0);
    timing.record(
        TrainingStage::RolloutInitialization,
        Duration::new(0, 999_999_999),
    );
    timing.record(TrainingStage::Collection, Duration::from_nanos(2));
    for stage in &STAGES[2..] {
        timing.record(*stage, Duration::ZERO);
    }

    timing.finish(Duration::new(1, 1), TrainingTimingOutcome::Complete);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=1 mode=reset_window ",
            "outcome=complete elapsed_ns=1000000001 measured_ns=1000000001 ",
            "rollout_initialization_ns=999999999 collection_ns=2 batch_preparation_ns=0 ",
            "optimization_ns=0 finalization_ns=0 samples=unknown optimizer_steps=unknown ",
            "last_stage=finalization duration_overflow=false timing_valid=true",
        )
    );
}

#[test]
fn incomplete_collection_keeps_unreached_stages_and_counts_unknown() {
    let mut timing = TrainingUpdateTiming::new(2, TrainingUpdateMode::CompleteEpisodes, 4);
    timing.record(
        TrainingStage::RolloutInitialization,
        Duration::from_nanos(3),
    );
    timing.record(TrainingStage::Collection, Duration::from_nanos(7));

    timing.finish(Duration::from_nanos(11), TrainingTimingOutcome::Incomplete);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=2 mode=complete_episodes ",
            "outcome=incomplete elapsed_ns=11 measured_ns=10 ",
            "rollout_initialization_ns=3 collection_ns=7 batch_preparation_ns=unknown ",
            "optimization_ns=unknown finalization_ns=unknown samples=unknown ",
            "optimizer_steps=unknown last_stage=collection ",
            "duration_overflow=false timing_valid=true",
        )
    );
}

#[test]
fn optimizer_error_reports_observed_applied_steps_without_claiming_finalization() {
    let mut timing = TrainingUpdateTiming::new(3, TrainingUpdateMode::ResetWindow, 20);
    for stage in &STAGES[..4] {
        timing.record(*stage, Duration::from_nanos(1));
    }
    timing.set_samples(8);
    timing.set_optimizer_step(22);

    timing.finish(Duration::from_nanos(4), TrainingTimingOutcome::Error);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=3 mode=reset_window ",
            "outcome=error elapsed_ns=4 measured_ns=4 rollout_initialization_ns=1 ",
            "collection_ns=1 batch_preparation_ns=1 optimization_ns=1 ",
            "finalization_ns=unknown samples=8 optimizer_steps=2 last_stage=optimization ",
            "duration_overflow=false timing_valid=true",
        )
    );
}

#[test]
fn incomplete_optimization_does_not_invent_an_optimizer_step_delta() {
    let mut timing = TrainingUpdateTiming::new(3, TrainingUpdateMode::ResetWindow, 20);
    for stage in &STAGES[..4] {
        timing.record(*stage, Duration::ZERO);
    }

    timing.finish(Duration::ZERO, TrainingTimingOutcome::Incomplete);
    let output = timing.to_string();

    assert!(output.contains("outcome=incomplete"));
    assert!(output.contains("optimizer_steps=unknown last_stage=optimization"));
}

#[test]
fn optimizer_counter_regression_is_unknown_and_marks_timing_invalid() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 4);
    timing.record(TrainingStage::RolloutInitialization, Duration::ZERO);

    timing.set_optimizer_step(3);
    timing.finish(Duration::ZERO, TrainingTimingOutcome::Error);
    let output = timing.to_string();

    assert!(output.contains("optimizer_steps=unknown"));
    assert!(output.ends_with("duration_overflow=false timing_valid=false"));
}

#[test]
fn maximum_update_index_and_optimizer_delta_do_not_wrap() {
    let mut timing = TrainingUpdateTiming::new(u64::MAX, TrainingUpdateMode::ResetWindow, 0);
    for stage in STAGES {
        timing.record(stage, Duration::ZERO);
    }

    timing.set_optimizer_step(u64::MAX);
    timing.finish(Duration::ZERO, TrainingTimingOutcome::Complete);
    let output = timing.to_string();

    assert!(output.contains("update_index=18446744073709551615"));
    assert!(output.contains("optimizer_steps=18446744073709551615"));
    assert!(output.ends_with("timing_valid=true"));
}

#[test]
fn same_stage_duration_overflow_saturates_and_is_explicit() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
    timing.record(TrainingStage::RolloutInitialization, Duration::MAX);

    timing.record(
        TrainingStage::RolloutInitialization,
        Duration::from_nanos(1),
    );
    timing.finish(Duration::MAX, TrainingTimingOutcome::Incomplete);

    assert_eq!(
        timing.to_string(),
        concat!(
            "level=INFO event=training_update_timing update_index=0 mode=reset_window ",
            "outcome=incomplete elapsed_ns=18446744073709551615999999999 ",
            "measured_ns=18446744073709551615999999999 ",
            "rollout_initialization_ns=18446744073709551615999999999 ",
            "collection_ns=unknown batch_preparation_ns=unknown optimization_ns=unknown ",
            "finalization_ns=unknown samples=unknown optimizer_steps=unknown ",
            "last_stage=rollout_initialization duration_overflow=true timing_valid=false",
        )
    );
}

#[test]
fn total_duration_overflow_does_not_corrupt_individual_stages() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
    timing.record(TrainingStage::RolloutInitialization, Duration::MAX);

    timing.record(TrainingStage::Collection, Duration::from_nanos(1));
    timing.finish(Duration::MAX, TrainingTimingOutcome::Incomplete);
    let output = timing.to_string();

    assert!(output.contains("measured_ns=18446744073709551615999999999"));
    assert!(output.contains("collection_ns=1 batch_preparation_ns=unknown"));
    assert!(output.ends_with("duration_overflow=true timing_valid=false"));
}

#[test]
fn measured_duration_exceeding_wall_time_is_marked_invalid() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
    timing.record(
        TrainingStage::RolloutInitialization,
        Duration::from_nanos(2),
    );

    timing.finish(Duration::from_nanos(1), TrainingTimingOutcome::Incomplete);
    let output = timing.to_string();

    assert!(output.contains("elapsed_ns=1 measured_ns=2"));
    assert!(output.ends_with("duration_overflow=false timing_valid=false"));
}

#[test]
fn skipped_or_backwards_stage_does_not_fabricate_a_duration() {
    for invalid_stage in [
        TrainingStage::RolloutInitialization,
        TrainingStage::Finalization,
    ] {
        let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
        timing.record(
            TrainingStage::RolloutInitialization,
            Duration::from_nanos(1),
        );
        timing.record(TrainingStage::Collection, Duration::from_nanos(2));

        timing.record(invalid_stage, Duration::from_nanos(100));
        timing.finish(Duration::from_nanos(103), TrainingTimingOutcome::Incomplete);
        let output = timing.to_string();

        assert!(output.contains("measured_ns=3 rollout_initialization_ns=1"));
        assert!(output.contains("finalization_ns=unknown"));
        assert!(output.ends_with("timing_valid=false"));
    }
}

#[test]
fn completed_operation_with_missing_measurements_is_marked_invalid() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
    timing.record(TrainingStage::RolloutInitialization, Duration::ZERO);

    timing.finish(Duration::ZERO, TrainingTimingOutcome::Complete);
    let output = timing.to_string();

    assert!(output.contains("outcome=complete"));
    assert!(output.contains("collection_ns=unknown"));
    assert!(output.ends_with("timing_valid=false"));
}

#[test]
fn stage_recorded_after_finish_cannot_change_measured_durations() {
    let mut timing = TrainingUpdateTiming::new(0, TrainingUpdateMode::ResetWindow, 0);
    timing.record(TrainingStage::RolloutInitialization, Duration::ZERO);
    timing.finish(Duration::ZERO, TrainingTimingOutcome::Incomplete);

    timing.record(TrainingStage::Collection, Duration::from_nanos(1));
    let output = timing.to_string();

    assert!(output.contains("elapsed_ns=0 measured_ns=0"));
    assert!(output.contains("collection_ns=unknown"));
    assert!(output.ends_with("timing_valid=false"));
}

#[test]
fn initialization_checkpoint_and_resume_export_scopes_have_accurate_labels() {
    let scopes = [
        (
            TrainingTimingScope::SessionInitialization,
            "session_initialization",
        ),
        (
            TrainingTimingScope::CheckpointCaptureSaveRuntimeExport,
            "checkpoint_capture_save_runtime_export",
        ),
        (
            TrainingTimingScope::ResumeRuntimeExport,
            "resume_runtime_export",
        ),
    ];

    for (scope, label) in scopes {
        let timing = TrainingScopeTiming::new(
            scope,
            Some(5),
            Duration::from_nanos(123),
            TrainingTimingOutcome::Complete,
        );

        assert_eq!(
            timing.to_string(),
            format!(
                "level=INFO event=training_scope_timing scope={label} completed_updates=5 \
                 outcome=complete elapsed_ns=123"
            )
        );
    }
}

#[test]
fn initialization_without_restored_counter_and_scope_errors_are_explicit() {
    let outcomes = [
        (TrainingTimingOutcome::Complete, "complete"),
        (TrainingTimingOutcome::Error, "error"),
        (TrainingTimingOutcome::Incomplete, "incomplete"),
    ];

    for (outcome, label) in outcomes {
        let timing = TrainingScopeTiming::new(
            TrainingTimingScope::SessionInitialization,
            None,
            Duration::ZERO,
            outcome,
        );

        assert_eq!(
            timing.to_string(),
            format!(
                "level=INFO event=training_scope_timing scope=session_initialization \
                 completed_updates=unknown outcome={label} elapsed_ns=0"
            )
        );
    }
}

#[test]
fn scope_timer_preserves_success_value_and_invokes_operation_once() {
    let calls = Cell::new(0);

    let result = time_training_scope(TrainingTimingScope::SessionInitialization, None, || {
        calls.set(calls.get() + 1);
        Ok::<_, &'static str>(37)
    });

    assert_eq!(result, Ok(37));
    assert_eq!(calls.get(), 1);
}

#[test]
fn unwinding_after_success_or_error_is_reported_as_incomplete() {
    for outcome in [
        TrainingTimingOutcome::Complete,
        TrainingTimingOutcome::Error,
        TrainingTimingOutcome::Incomplete,
    ] {
        let observed = outcome.on_scope_exit(true);

        assert_eq!(observed, TrainingTimingOutcome::Incomplete);
    }
}

#[test]
fn ordinary_scope_exit_preserves_the_recorded_outcome() {
    for outcome in [
        TrainingTimingOutcome::Complete,
        TrainingTimingOutcome::Error,
        TrainingTimingOutcome::Incomplete,
    ] {
        let observed = outcome.on_scope_exit(false);

        assert_eq!(observed, outcome);
    }
}

#[test]
fn scope_timer_preserves_original_error_and_invokes_operation_once() {
    let calls = Cell::new(0);

    let result = time_training_scope(
        TrainingTimingScope::CheckpointCaptureSaveRuntimeExport,
        Some(8),
        || {
            calls.set(calls.get() + 1);
            Err::<(), _>("original checkpoint capture failure")
        },
    );

    assert_eq!(result, Err("original checkpoint capture failure"));
    assert_eq!(calls.get(), 1);
}

#[test]
fn checkpoint_timer_preserves_success_value_and_invokes_operation_once() {
    let calls = Cell::new(0);

    let result = time_training_checkpoint(0, || {
        calls.set(calls.get() + 1);
        Ok::<_, &'static str>(37)
    });

    assert_eq!(result, Ok(37));
    assert_eq!(calls.get(), 1);
}

#[test]
fn checkpoint_timer_preserves_original_error_and_invokes_operation_once() {
    let calls = Cell::new(0);

    let result = time_training_checkpoint(u64::MAX, || {
        calls.set(calls.get() + 1);
        Err::<(), _>("original checkpoint runtime export failure")
    });

    assert_eq!(result, Err("original checkpoint runtime export failure"));
    assert_eq!(calls.get(), 1);
}

#[test]
fn update_timer_preserves_original_collection_error() {
    let mut timer = TrainingUpdateTimer::new(0, TrainingUpdateMode::ResetWindow, 0);
    timer.enter(TrainingStage::Collection);
    timer.set_samples(3);

    let result = timer.observe_result(Err::<(), _>("original collection failure"));

    assert_eq!(result, Err("original collection failure"));
}

#[test]
fn update_timer_preserves_original_optimizer_error_after_observing_step_delta() {
    let mut timer = TrainingUpdateTimer::new(0, TrainingUpdateMode::CompleteEpisodes, 3);
    for stage in &STAGES[1..4] {
        timer.enter(*stage);
    }
    timer.set_optimizer_step(5);

    let result = timer.observe_result(Err::<(), _>("original optimizer failure"));

    assert_eq!(result, Err("original optimizer failure"));
}

#[test]
fn timing_writer_emits_one_complete_line_without_disabling_logging() {
    let disabled = AtomicBool::new(false);
    let mut writer = Cursor::new([0_u8; 512]);
    let timing = TrainingScopeTiming::new(
        TrainingTimingScope::ResumeRuntimeExport,
        Some(8),
        Duration::from_nanos(9),
        TrainingTimingOutcome::Complete,
    );

    write_training_timing(&mut writer, &disabled, &timing);

    assert_eq!(
        &writer.get_ref()[..writer.position() as usize],
        b"level=INFO event=training_scope_timing scope=resume_runtime_export \
          completed_updates=8 outcome=complete elapsed_ns=9\n",
    );
    assert!(!disabled.load(Ordering::Relaxed));
}

#[test]
fn timing_writer_disables_after_write_error_and_does_not_retry() {
    let disabled = AtomicBool::new(false);
    let mut writer = BrokenWriter { calls: 0 };

    write_training_timing(&mut writer, &disabled, &"first record");
    write_training_timing(&mut writer, &disabled, &"second record");

    assert!(disabled.load(Ordering::Relaxed));
    assert_eq!(writer.calls, 1);
}

#[test]
fn timing_writer_handles_partial_line_failure_without_panicking() {
    let disabled = AtomicBool::new(false);
    let mut writer = Cursor::new([0_u8; 4]);

    write_training_timing(&mut writer, &disabled, &"long record");
    write_training_timing(&mut writer, &disabled, &"not retried");

    assert!(disabled.load(Ordering::Relaxed));
    assert_eq!(writer.position(), 4);
    assert_eq!(writer.into_inner(), *b"long");
}

struct BrokenWriter {
    calls: usize,
}

impl Write for BrokenWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "timing sink is closed",
        ))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
