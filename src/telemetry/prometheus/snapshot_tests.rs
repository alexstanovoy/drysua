use super::*;

#[test]
fn duration_boundaries_are_inclusive_but_the_next_nanosecond_is_not() {
    for (index, seconds) in BUCKETS.into_iter().enumerate() {
        for above in [false, true] {
            let mut histogram = DurationHistogram::default();
            let duration =
                Duration::from_secs_f64(seconds) + Duration::from_nanos(u64::from(above));
            histogram.observe(duration).unwrap();
            let first = index + usize::from(above);
            assert!(histogram.buckets[..first].iter().all(|count| *count == 0));
            assert!(histogram.buckets[first..].iter().all(|count| *count == 1));
            assert_eq!(histogram.count, 1);
            assert_eq!(histogram.sum_seconds, duration.as_secs_f64());
        }
    }
}

#[test]
fn exact_integer_limit_is_accepted_but_overflow_is_transactional() {
    let mut histogram = DurationHistogram {
        buckets: [MAX_COUNTER - 1; 10],
        count: MAX_COUNTER - 1,
        sum_seconds: 0.0,
    };
    histogram.observe(Duration::ZERO).unwrap();
    assert_eq!(histogram.count, MAX_COUNTER);
    assert_eq!(histogram.buckets, [MAX_COUNTER; 10]);
    let before = histogram;
    assert_eq!(
        histogram.observe(Duration::ZERO).unwrap_err().to_string(),
        "metrics duration count exceeds 9007199254740991"
    );
    assert_eq!(histogram, before);
}

#[test]
fn invalid_histograms_are_neither_mutated_nor_repaired_by_observation() {
    for (buckets, count, sum_seconds, message) in [
        (
            [1; 10],
            1,
            f64::INFINITY,
            "metrics duration sum must be finite and nonnegative",
        ),
        (
            [2; 10],
            1,
            1.0,
            "metrics duration buckets must be cumulative and at most count",
        ),
        (
            [0; 10],
            0,
            1.0,
            "metrics empty duration histogram must have zero sum",
        ),
    ] {
        let mut histogram = DurationHistogram {
            buckets,
            count,
            sum_seconds,
        };
        let before = histogram;
        assert_eq!(
            histogram.observe(Duration::ZERO).unwrap_err().to_string(),
            message
        );
        assert_eq!(histogram, before);
    }
}
