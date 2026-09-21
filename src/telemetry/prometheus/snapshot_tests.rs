use super::*;

fn snapshot() -> TrainingSnapshot {
    TrainingSnapshot {
        completed_updates: 2,
        updates_target: 10,
        parallel: 1,
        ..TrainingSnapshot::default()
    }
}

#[test]
fn duration_observation_uses_inclusive_cumulative_buckets() {
    let mut histogram = DurationHistogram::default();
    histogram.observe(Duration::ZERO).unwrap();
    histogram.observe(Duration::from_millis(1)).unwrap();
    histogram.observe(Duration::from_secs(1)).unwrap();
    histogram.observe(Duration::from_secs(3601)).unwrap();
    assert_eq!(histogram.buckets, [2, 2, 2, 3, 3, 3, 3, 3, 3, 3]);
    assert_eq!(histogram.count, 4);
    assert_eq!(histogram.sum_seconds, 3602.001);
}

#[test]
fn every_duration_boundary_is_inclusive() {
    for (index, seconds) in BUCKETS.into_iter().enumerate() {
        let mut histogram = DurationHistogram::default();
        histogram.observe(Duration::from_secs_f64(seconds)).unwrap();
        assert!(histogram.buckets[..index].iter().all(|count| *count == 0));
        assert!(histogram.buckets[index..].iter().all(|count| *count == 1));
        assert_eq!(histogram.count, 1);
        assert_eq!(histogram.sum_seconds, seconds);
    }
}

#[test]
fn duration_above_largest_bucket_increments_only_count_and_sum() {
    let mut histogram = DurationHistogram::default();
    histogram.observe(Duration::MAX).unwrap();
    assert_eq!(histogram.buckets, [0; 10]);
    assert_eq!(histogram.count, 1);
    assert!(histogram.sum_seconds.is_finite());
}

#[test]
fn duration_one_nanosecond_above_boundary_excludes_that_bucket() {
    let mut histogram = DurationHistogram::default();
    histogram.observe(Duration::from_nanos(1_000_001)).unwrap();
    assert_eq!(histogram.buckets, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(histogram.count, 1);
}

#[test]
fn duration_observation_rejects_invalid_sum_without_mutation() {
    let mut histogram = DurationHistogram {
        buckets: [1; 10],
        count: 1,
        sum_seconds: f64::INFINITY,
    };
    let before = histogram;
    assert_eq!(
        histogram.observe(Duration::ZERO).unwrap_err().to_string(),
        "metrics duration sum must be finite and nonnegative"
    );
    assert_eq!(histogram, before);
}

#[test]
fn duration_count_limit_failure_leaves_histogram_unchanged() {
    let mut histogram = DurationHistogram {
        buckets: [MAX_COUNTER; 10],
        count: MAX_COUNTER,
        sum_seconds: 1.0,
    };
    let before = histogram;
    let error = histogram.observe(Duration::ZERO).unwrap_err();
    assert_eq!(
        error.to_string(),
        "metrics duration count exceeds 9007199254740991"
    );
    assert_eq!(histogram, before);
}

#[test]
fn invalid_histogram_is_not_repaired_by_observation() {
    let mut histogram = DurationHistogram {
        buckets: [2; 10],
        count: 1,
        sum_seconds: 1.0,
    };
    let before = histogram;
    let error = histogram.observe(Duration::from_secs(1)).unwrap_err();
    assert_eq!(
        error.to_string(),
        "metrics duration buckets must be cumulative and at most count"
    );
    assert_eq!(histogram, before);
}

#[test]
fn snapshot_accepts_counter_and_configuration_boundaries() {
    let mut value = snapshot();
    value.start_update = 1_000_000;
    value.completed_updates = 1_000_000;
    value.updates_target = 1_000_000;
    value.samples = MAX_COUNTER;
    value.optimizer_steps = MAX_COUNTER;
    value.games = [MAX_COUNTER; 4];
    value.last_update_games = value.games;
    value.parallel = 40;
    value.games_per_update = 40;
    value.generation = Some(u64::MAX);
    value.scale_bp = Some(10_000);
    value.losses = Some([-1.0, 0.0, 1.0, f64::MAX]);
    value.heartbeat = u64::MAX;
    value.durations = [DurationHistogram {
        buckets: [MAX_COUNTER; 10],
        count: MAX_COUNTER,
        sum_seconds: f64::MAX,
    }; 6];
    value.validate().unwrap();
}

#[test]
fn snapshot_accepts_window_mode_and_absent_optional_metrics() {
    let value = snapshot();
    value.validate().unwrap();
    assert_eq!(value.games_per_update, 0);
    assert_eq!(value.generation, None);
    assert_eq!(value.scale_bp, None);
    assert_eq!(value.losses, None);
}

#[test]
fn snapshot_rejects_invalid_update_ranges() {
    for (start, completed, target) in [(3, 2, 10), (0, 11, 10), (0, 0, 1_000_001)] {
        let mut value = snapshot();
        value.start_update = start;
        value.completed_updates = completed;
        value.updates_target = target;
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            "metrics updates must satisfy start <= completed <= target <= 1000000"
        );
    }
}

#[test]
fn snapshot_rejects_inexact_integer_counters() {
    for index in 0..6 {
        let mut value = snapshot();
        match index {
            0 => value.samples = MAX_COUNTER + 1,
            1 => value.optimizer_steps = MAX_COUNTER + 1,
            _ => value.games[index - 2] = MAX_COUNTER + 1,
        }
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            "metrics counter exceeds 9007199254740991"
        );
    }
}

#[test]
fn snapshot_rejects_invalid_parallel_and_game_limits() {
    for parallel in [0, 41, u64::MAX] {
        let mut value = snapshot();
        value.parallel = parallel;
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            "metrics parallel must be in 1..=40"
        );
    }
    let mut value = snapshot();
    value.games_per_update = 41;
    assert_eq!(
        value.validate().unwrap_err().to_string(),
        "metrics games_per_update must be at most 40"
    );
}

#[test]
fn snapshot_rejects_last_update_outcomes_above_cumulative_outcomes() {
    for index in 0..4 {
        let mut value = snapshot();
        value.last_update_games[index] = 1;
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            "metrics last-update games exceed cumulative games"
        );
    }
}

#[test]
fn snapshot_requires_generation_and_scale_together() {
    for (generation, scale) in [(Some(1), None), (None, Some(0))] {
        let mut value = snapshot();
        value.generation = generation;
        value.scale_bp = scale;
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            "metrics generation and scale must be present together"
        );
    }
    let mut value = snapshot();
    value.generation = Some(1);
    value.scale_bp = Some(10_001);
    assert_eq!(
        value.validate().unwrap_err().to_string(),
        "metrics scale must be at most 10000 basis points"
    );
}

#[test]
fn snapshot_rejects_nonfinite_losses_in_every_slot() {
    for index in 0..4 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut value = snapshot();
            let mut losses = [0.0; 4];
            losses[index] = invalid;
            value.losses = Some(losses);
            assert_eq!(
                value.validate().unwrap_err().to_string(),
                "metrics losses must be finite"
            );
        }
    }
}

#[test]
fn snapshot_rejects_nonfinite_or_negative_duration_sums_in_every_stage() {
    for index in 0..6 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut value = snapshot();
            value.durations[index].sum_seconds = invalid;
            assert_eq!(
                value.validate().unwrap_err().to_string(),
                "metrics duration sum must be finite and nonnegative"
            );
        }
    }
}

#[test]
fn snapshot_rejects_decreasing_buckets_excess_counts_and_empty_nonzero_sums() {
    let mut value = snapshot();
    value.durations[0].count = 1;
    value.durations[0].buckets[0] = 1;
    assert_eq!(
        value.validate().unwrap_err().to_string(),
        "metrics duration buckets must be cumulative and at most count"
    );
    value.durations[0] = DurationHistogram {
        count: MAX_COUNTER + 1,
        ..DurationHistogram::default()
    };
    assert_eq!(
        value.validate().unwrap_err().to_string(),
        "metrics duration count exceeds 9007199254740991"
    );
    value.durations[0] = DurationHistogram {
        sum_seconds: 1.0,
        ..DurationHistogram::default()
    };
    assert_eq!(
        value.validate().unwrap_err().to_string(),
        "metrics empty duration histogram must have zero sum"
    );
}

#[test]
fn duration_count_can_reach_the_exact_integer_limit_without_saturating() {
    let mut histogram = DurationHistogram {
        buckets: [MAX_COUNTER - 1; 10],
        count: MAX_COUNTER - 1,
        sum_seconds: 0.0,
    };
    histogram.observe(Duration::ZERO).unwrap();
    assert_eq!(histogram.count, MAX_COUNTER);
    assert_eq!(histogram.buckets, [MAX_COUNTER; 10]);
    assert_eq!(histogram.sum_seconds, 0.0);
}
