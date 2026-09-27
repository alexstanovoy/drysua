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
fn live_percentiles_are_upper_bounds_and_histogram_overflow_invalidates_reports() {
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    for tick in 1..=20 {
        let nanos = if tick == 20 { 100 } else { 5 };
        telemetry
            .record(
                tick,
                Duration::ZERO,
                sample(Duration::from_nanos(nanos), Duration::ZERO),
            )
            .unwrap();
    }
    let report = telemetry
        .report(ReportScope::Total, FinishReason::MatchOver, false)
        .to_string();
    assert!(report.contains(
        "compute_total_ns=195 compute_p50_upper_ns=8 compute_p95_upper_ns=8 compute_max_ns=100"
    ));
    for tick in 21..=22 {
        telemetry
            .record(tick, Duration::ZERO, sample(Duration::MAX, Duration::ZERO))
            .unwrap();
    }
    let report = telemetry.report(ReportScope::Total, FinishReason::Error, false);
    assert!(!report.is_valid());
    assert!(
        report
            .to_string()
            .contains(&format!("compute_total_ns={}", Duration::MAX.as_nanos()))
    );
}

#[test]
fn live_windows_exclude_wait_preserve_totals_and_reject_regressions_transactionally() {
    let mut telemetry = LiveTelemetry::new(10, LiveConfig::new(1, 0).unwrap()).unwrap();
    telemetry.start(Duration::ZERO);
    for tick in 1..=2 {
        let mut timing = sample(Duration::from_millis(100), Duration::from_secs(1));
        timing.ack_send = (tick == 2).then(|| Duration::from_nanos(1));
        let now = Duration::from_secs(u64::from(tick));
        assert!(telemetry.record(tick, now, timing).unwrap());
        let report = telemetry
            .report(ReportScope::Window, FinishReason::Periodic, false)
            .to_string();
        assert!(report.contains("updates=1"));
        assert!(report.contains("compute_overruns=0"));
        assert!(report.contains(if tick == 1 {
            "service_overruns=0"
        } else {
            "service_overruns=1"
        }));
        assert!(report.contains("receive_wait_count=1 receive_wait_total_ns=1000000000"));
        assert!(report.starts_with(if tick == 1 {
            "level=INFO"
        } else {
            "level=WARN"
        }));
        telemetry.reset_window();
    }
    let before = telemetry
        .report(ReportScope::Total, FinishReason::Error, false)
        .to_string();
    for (tick, now, error) in [
        (
            2,
            Duration::from_secs(3),
            "performance completed tick must increase",
        ),
        (3, Duration::ZERO, "performance clock must not regress"),
    ] {
        assert_eq!(
            telemetry.record(tick, now, sample(Duration::ZERO, Duration::ZERO)),
            Err(error)
        );
        assert_eq!(
            telemetry
                .report(ReportScope::Total, FinishReason::Error, false)
                .to_string(),
            before
        );
    }
    assert!(before.contains("updates=2 progress_ticks=1 elapsed_ns=2000000000"));
    assert!(before.contains("updates_per_second=1.000 realtime_factor=0.050"));
}

#[test]
fn malformed_nested_timings_and_overflow_invalidate_final_reports() {
    let nanos = Duration::from_nanos;
    for case in 0..5 {
        let mut timing = sample(Duration::ZERO, Duration::ZERO);
        match case {
            0 => {
                timing.compute = Duration::MAX;
                timing.merge(sample(nanos(1), Duration::ZERO));
                assert_eq!(timing.compute, Duration::MAX);
            }
            1 => {
                timing.ack_send = Some(nanos(1));
                timing.finish_handler(Duration::ZERO, None);
            }
            2 => timing.finish_handler_span(nanos(20), nanos(19), None),
            3 => timing.finish_handler_span(nanos(10), nanos(20), Some((nanos(9), nanos(15)))),
            _ => timing.finish_handler(nanos(1), Some(nanos(2))),
        }
        let mut telemetry = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
        telemetry.record(1, Duration::from_secs(1), timing).unwrap();
        let report = telemetry.report(ReportScope::Total, FinishReason::Error, false);
        assert!(!report.is_valid(), "case {case}");
        assert!(report.to_string().ends_with("saturated=true"));
    }
    let mut timing = sample(nanos(2), nanos(50));
    timing.order_send = Some(nanos(3));
    timing.ack_send = Some(nanos(5));
    timing.finish_handler(nanos(18), Some(nanos(7)));
    assert_eq!(timing.compute, nanos(12));
    assert_eq!(timing.decision, Some(nanos(4)));
    assert_eq!(timing.receive_wait, nanos(50));
    assert!(!timing.saturated);
}

#[test]
fn diagnostic_output_is_opt_in_bounded_and_disables_failed_writes() {
    let mut disabled = LiveTelemetry::new(30, LiveConfig::default()).unwrap();
    assert!(!disabled.debug_due());
    let mut telemetry = LiveTelemetry::new(30, LiveConfig::new(300, 2).unwrap()).unwrap();
    assert_eq!(
        (0..1000).filter(|_| telemetry.debug_due()).count(),
        DEBUG_RECORD_LIMIT as usize
    );
    const { assert!(DEBUG_RECORD_LIMIT <= 32) };
    let mut output = PerformanceOutput::new(Cursor::new([0_u8; 4]));
    output.emit(&"first long line");
    output.emit(&"second line");
    assert!(output.failed());
    assert_eq!(output.into_inner().into_inner(), *b"firs");
}
