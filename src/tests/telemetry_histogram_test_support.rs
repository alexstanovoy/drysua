use super::*;

#[test]
fn counter_at_fixed_bound_drops_new_sample_and_marks_saturation() {
    let mut histogram = LatencyHistogram {
        count: u64::MAX,
        ..LatencyHistogram::default()
    };
    histogram.buckets[0] = u64::MAX;
    histogram.record(Duration::from_secs(1));
    assert_eq!(histogram.count(), u64::MAX);
    assert_eq!(histogram.total(), Duration::ZERO);
    assert_eq!(histogram.percentile_upper(95), Some(Duration::ZERO));
    assert!(histogram.saturated());
}
