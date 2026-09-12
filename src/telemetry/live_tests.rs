use std::io::Cursor;
use std::time::Duration;

use super::*;

fn sample(compute: Duration, receive_wait: Duration) -> UpdateTiming {
    UpdateTiming {
        compute,
        receive_wait,
        decision: None,
        order_send: None,
        ack_send: None,
        saturated: false,
    }
}

#[test]
fn histogram_empty_zero_and_power_of_two_boundaries_are_explicit() {
    let mut histogram = LatencyHistogram::default();
    assert_eq!(histogram.percentile_upper(50), None);
    for nanos in [0, 1, 2, 3, 4, 5] {
        histogram.record(Duration::from_nanos(nanos));
    }
    assert_eq!(histogram.count(), 6);
    assert_eq!(
        histogram.percentile_upper(50),
        Some(Duration::from_nanos(2))
    );
    assert_eq!(
        histogram.percentile_upper(95),
        Some(Duration::from_nanos(5))
    );
    assert_eq!(histogram.max(), Duration::from_nanos(5));
    assert_eq!(histogram.total(), Duration::from_nanos(15));
}

#[test]
fn histogram_p95_is_nearest_rank_upper_bound_not_an_exact_percentile() {
    let mut histogram = LatencyHistogram::default();
    for _ in 0..19 {
        histogram.record(Duration::from_nanos(5));
    }
    histogram.record(Duration::from_nanos(100));
    assert_eq!(
        histogram.percentile_upper(50),
        Some(Duration::from_nanos(8))
    );
    assert_eq!(
        histogram.percentile_upper(95),
        Some(Duration::from_nanos(8))
    );
    assert_eq!(
        histogram.percentile_upper(100),
        Some(Duration::from_nanos(100))
    );
}

#[test]
fn histogram_duration_overflow_saturates_without_wrapping_or_losing_maximum() {
    let mut histogram = LatencyHistogram::default();
    histogram.record(Duration::MAX);
    histogram.record(Duration::from_nanos(1));
    assert_eq!(histogram.max(), Duration::MAX);
    assert_eq!(histogram.total(), Duration::MAX);
    assert_eq!(histogram.percentile_upper(95), Some(Duration::MAX));
    assert!(histogram.saturated());
}

#[test]
#[should_panic(expected = "percentile must be in 1..=100")]
fn histogram_rejects_zero_percentile() {
    LatencyHistogram::default().percentile_upper(0);
}

#[test]
fn live_configuration_rejects_zero_tickrate_and_unbounded_log_intervals() {
    assert_eq!(
        LiveTelemetry::new(0, LiveConfig::default()).err(),
        Some("performance tick rate must be positive")
    );
    assert_eq!(
        LiveConfig::new(0, 0).err(),
        Some("performance report interval must be in 1..=30000")
    );
    assert_eq!(
        LiveConfig::new(30_001, 0).err(),
        Some("performance report interval must be in 1..=30000")
    );
    assert_eq!(
        LiveConfig::from_debug_value(Some("-1")).err(),
        Some("DRYSUA_PERF_DEBUG_EVERY must be an integer in 0..=4294967295")
    );
    assert_eq!(
        LiveConfig::from_debug_value(Some("4294967296")).err(),
        Some("DRYSUA_PERF_DEBUG_EVERY must be an integer in 0..=4294967295")
    );
    assert_eq!(
        LiveConfig::from_debug_value(Some("4294967295"))
            .unwrap()
            .debug_every,
        u32::MAX
    );
    assert_eq!(LiveConfig::default().report_every, 300);
    assert_eq!(LiveConfig::default().debug_every, 0);
}

#[test]
fn receive_wait_does_not_trigger_compute_or_service_budget_warning() {
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::new(2, 0).unwrap()).unwrap();
    telemetry.start(Duration::ZERO);
    assert!(
        !telemetry
            .record(
                1,
                Duration::from_secs(1),
                sample(Duration::from_millis(1), Duration::from_millis(999))
            )
            .unwrap()
    );
    assert!(
        telemetry
            .record(
                2,
                Duration::from_secs(2),
                sample(Duration::from_millis(1), Duration::from_millis(999))
            )
            .unwrap()
    );
    let text = telemetry
        .report(ReportScope::Window, FinishReason::Periodic, false)
        .to_string();
    assert!(text.starts_with("level=INFO event=live_performance scope=window reason=periodic "));
    assert!(text.contains("updates=2 progress_ticks=1 elapsed_ns=2000000000 updates_per_second=1.000 realtime_factor=0.016"));
    assert!(text.contains("budget_ns=33333333 compute_overruns=0 service_overruns=0"));
    assert!(text.contains("compute_count=2 compute_total_ns=2000000"));
    assert!(text.contains("receive_wait_count=2 receive_wait_total_ns=1998000000"));
    assert!(text.contains("percentiles=log2_upper_bounds"));
}

#[test]
fn budget_boundary_is_strict_and_send_stalls_are_not_mislabeled_compute() {
    let mut telemetry = LiveTelemetry::new(10, LiveConfig::default()).unwrap();
    telemetry.start(Duration::ZERO);
    let mut update = sample(Duration::from_millis(100), Duration::ZERO);
    telemetry
        .record(1, Duration::from_millis(100), update)
        .unwrap();
    update.ack_send = Some(Duration::from_nanos(1));
    telemetry
        .record(2, Duration::from_millis(201), update)
        .unwrap();
    let text = telemetry
        .report(ReportScope::Total, FinishReason::MatchOver, false)
        .to_string();
    assert!(text.starts_with("level=WARN event=live_performance"));
    assert!(text.contains("compute_overruns=0 service_overruns=1"));
    assert!(text.contains("ack_send_count=1 ack_send_total_ns=1"));
}

#[test]
fn reporting_reset_keeps_lifetime_totals_and_next_window_tick_baseline() {
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::new(1, 0).unwrap()).unwrap();
    telemetry.start(Duration::ZERO);
    telemetry
        .record(
            400,
            Duration::from_secs(1),
            sample(Duration::ZERO, Duration::ZERO),
        )
        .unwrap();
    telemetry.reset_window();
    telemetry
        .record(
            403,
            Duration::from_secs(2),
            sample(Duration::ZERO, Duration::ZERO),
        )
        .unwrap();
    let window = telemetry
        .report(ReportScope::Window, FinishReason::Periodic, false)
        .to_string();
    let total = telemetry
        .report(ReportScope::Total, FinishReason::MatchOver, false)
        .to_string();
    assert!(window.contains("updates=1 progress_ticks=3 elapsed_ns=1000000000 updates_per_second=1.000 realtime_factor=0.100"));
    assert!(total.contains("updates=2 progress_ticks=3 elapsed_ns=2000000000"));
}

#[test]
fn empty_final_report_does_not_invent_rates_percentiles_or_completed_updates() {
    let telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    let text = telemetry
        .report(ReportScope::Total, FinishReason::Error, true)
        .to_string();
    assert!(text.contains(
        "updates=0 progress_ticks=0 elapsed_ns=0 updates_per_second=unknown realtime_factor=unknown"
    ));
    assert!(
        text.contains("compute_p50_upper_ns=unknown compute_p95_upper_ns=unknown compute_max_ns=0")
    );
    assert!(text.ends_with("pending_update=true saturated=false"));
    assert!(text.len() < 4096);
}

#[test]
fn regressed_ticks_and_clock_are_rejected_without_mutating_aggregates() {
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    telemetry.start(Duration::ZERO);
    telemetry
        .record(
            4,
            Duration::from_secs(1),
            sample(Duration::ZERO, Duration::ZERO),
        )
        .unwrap();
    assert_eq!(
        telemetry.record(
            4,
            Duration::from_secs(2),
            sample(Duration::ZERO, Duration::ZERO)
        ),
        Err("performance completed tick must increase")
    );
    assert_eq!(
        telemetry.record(5, Duration::ZERO, sample(Duration::ZERO, Duration::ZERO)),
        Err("performance clock must not regress")
    );
    assert!(
        telemetry
            .report(ReportScope::Total, FinishReason::Error, false)
            .to_string()
            .contains("updates=1 progress_ticks=0 elapsed_ns=1000000000")
    );
}

#[test]
fn debug_decision_sampling_is_opt_in_and_has_a_fixed_session_cap() {
    let mut disabled = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    assert!(!disabled.debug_due());
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::new(300, 2).unwrap()).unwrap();
    let emitted = (0..1000).filter(|_| telemetry.debug_due()).count();
    assert_eq!(emitted, DEBUG_RECORD_LIMIT as usize);
    const { assert!(DEBUG_RECORD_LIMIT <= 32) };
}

#[test]
fn telemetry_writer_disables_after_io_failure_without_propagating_to_gameplay() {
    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 4]));
    output.emit(&"first long line");
    output.emit(&"second line");
    assert!(output.failed());
    assert_eq!(output.into_inner().into_inner(), *b"firs");
}

#[test]
fn update_handler_time_excludes_sends_and_decision_is_a_compute_subset() {
    let mut timing = sample(Duration::from_nanos(2), Duration::from_nanos(50));
    timing.order_send = Some(Duration::from_nanos(3));
    timing.ack_send = Some(Duration::from_nanos(5));
    timing.finish_handler(Duration::from_nanos(18), Some(Duration::from_nanos(7)));
    assert_eq!(timing.compute, Duration::from_nanos(12));
    assert_eq!(timing.decision, Some(Duration::from_nanos(4)));
    assert_eq!(timing.receive_wait, Duration::from_nanos(50));
    assert!(!timing.saturated);
}

#[test]
fn impossible_nested_timings_and_pending_duration_overflow_are_flagged() {
    let mut timing = sample(Duration::MAX, Duration::ZERO);
    timing.merge(sample(Duration::from_nanos(1), Duration::ZERO));
    assert_eq!(timing.compute, Duration::MAX);
    assert!(timing.saturated);
    let mut nested = sample(Duration::ZERO, Duration::ZERO);
    nested.ack_send = Some(Duration::from_secs(1));
    nested.finish_handler(Duration::ZERO, None);
    assert!(nested.saturated);
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    telemetry.record(1, Duration::from_secs(1), nested).unwrap();
    assert!(
        telemetry
            .report(ReportScope::Total, FinishReason::Error, false)
            .to_string()
            .ends_with("saturated=true")
    );
}

#[test]
fn backward_handler_span_without_sends_is_flagged_even_if_completed_clock_increases() {
    let mut timing = sample(Duration::ZERO, Duration::ZERO);
    timing.finish_handler_span(Duration::from_nanos(20), Duration::from_nanos(19), None);
    assert!(timing.saturated);
    assert_eq!(timing.compute, Duration::ZERO);
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    telemetry.record(1, Duration::from_secs(1), timing).unwrap();
    assert!(
        !telemetry
            .report(ReportScope::Total, FinishReason::MatchOver, false)
            .is_valid()
    );
}

#[test]
fn live_aggregate_output_is_exact_for_injected_durations() {
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    for tick in 1..=2 {
        telemetry
            .record(
                tick,
                Duration::from_secs(u64::from(tick)),
                sample(Duration::from_millis(1), Duration::from_millis(999)),
            )
            .unwrap();
    }
    let report = telemetry.report(ReportScope::Total, FinishReason::MatchOver, false);
    assert_eq!(
        report.to_string(),
        concat!(
            "level=INFO event=live_performance scope=total reason=match_over ",
            "updates=2 progress_ticks=1 elapsed_ns=2000000000 updates_per_second=1.000 realtime_factor=0.016 ",
            "tick_rate=30 budget_ns=33333333 compute_overruns=0 service_overruns=0 percentiles=log2_upper_bounds ",
            "compute_count=2 compute_total_ns=2000000 compute_p50_upper_ns=1000000 compute_p95_upper_ns=1000000 compute_max_ns=1000000 ",
            "receive_wait_count=2 receive_wait_total_ns=1998000000 receive_wait_p50_upper_ns=999000000 receive_wait_p95_upper_ns=999000000 receive_wait_max_ns=999000000 ",
            "decision_count=0 decision_total_ns=0 decision_p50_upper_ns=unknown decision_p95_upper_ns=unknown decision_max_ns=0 ",
            "order_send_count=0 order_send_total_ns=0 order_send_p50_upper_ns=unknown order_send_p95_upper_ns=unknown order_send_max_ns=0 ",
            "ack_send_count=0 ack_send_total_ns=0 ack_send_p50_upper_ns=unknown ack_send_p95_upper_ns=unknown ack_send_max_ns=0 ",
            "pending_update=false saturated=false",
        )
    );
}

#[test]
fn decision_outside_handler_span_or_longer_than_internal_work_is_flagged() {
    let mut timing = sample(Duration::ZERO, Duration::ZERO);
    timing.finish_handler_span(
        Duration::from_nanos(10),
        Duration::from_nanos(20),
        Some((Duration::from_nanos(9), Duration::from_nanos(15))),
    );
    assert!(timing.saturated);
    let mut longer = sample(Duration::ZERO, Duration::ZERO);
    longer.finish_handler(Duration::from_nanos(1), Some(Duration::from_nanos(2)));
    assert!(longer.saturated);
}
