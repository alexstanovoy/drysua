use super::*;
use std::time::Duration;

const FAMILIES: [(&str, &str, usize); 22] = [
    ("drysua_training_updates_completed", "gauge", 1),
    ("drysua_training_updates_target", "gauge", 1),
    ("drysua_training_samples_total", "counter", 1),
    ("drysua_training_optimizer_steps_total", "counter", 1),
    ("drysua_training_games_total", "counter", 4),
    ("drysua_training_last_update_games", "gauge", 4),
    ("drysua_training_update_duration_seconds", "histogram", 13),
    ("drysua_training_stage_duration_seconds", "histogram", 65),
    ("drysua_training_scope_duration_seconds", "histogram", 39),
    ("drysua_training_policy_loss", "gauge", 1),
    ("drysua_training_value_loss", "gauge", 1),
    ("drysua_training_entropy", "gauge", 1),
    ("drysua_training_approximate_kl", "gauge", 1),
    ("drysua_training_generation", "gauge", 1),
    ("drysua_training_environment_scale_ratio", "gauge", 1),
    ("drysua_training_parallel_worlds", "gauge", 1),
    ("drysua_training_games_per_update", "gauge", 1),
    ("drysua_training_active", "gauge", 1),
    (
        "drysua_training_last_heartbeat_timestamp_seconds",
        "gauge",
        1,
    ),
    ("drysua_training_metrics_start_update", "gauge", 1),
    ("drysua_training_metrics_available", "gauge", 1),
    ("drysua_training_metrics_state_healthy", "gauge", 1),
];

fn snapshot() -> TrainingSnapshot {
    TrainingSnapshot {
        scope: [b's'; 32],
        checkpoint: [b'c'; 32],
        completed_updates: 103,
        updates_target: 200,
        samples: 456_789,
        optimizer_steps: 987,
        games: [11, 22, 33, 44],
        last_update_games: [1, 2, 3, 4],
        start_update: 100,
        parallel: 8,
        games_per_update: 40,
        generation: Some(0),
        scale_bp: Some(2500),
        losses: Some([-0.5, 0.25, 0.75, 0.125]),
        heartbeat: 111,
        ..TrainingSnapshot::default()
    }
}

fn samples(output: &str) -> Vec<&str> {
    assert!(output.len() <= 48 * 1024);
    assert!(output.ends_with('\n'));
    output
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect()
}

fn sample_value<'a>(output: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key} ");
    let values: Vec<_> = samples(output)
        .into_iter()
        .filter_map(|line| line.strip_prefix(&prefix))
        .collect();
    assert_eq!(values.len(), 1, "sample {key}");
    values[0]
}

fn assert_invalid(result: io::Result<String>, message: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), message);
}

#[test]
fn complete_exposition_has_exact_families_types_and_sample_cardinality() {
    let value = snapshot();
    let scopes = [DurationHistogram::default(); 3];

    let output = render(Some(&value), true, true, 123, Some(&scopes)).unwrap();

    for prefix in ["# HELP ", "# TYPE "] {
        assert_eq!(
            output
                .lines()
                .filter(|line| line.starts_with(prefix))
                .count(),
            22
        );
    }
    assert_eq!(samples(&output).len(), 142);
    for (name, kind, expected) in FAMILIES {
        let help = format!("# HELP {name} ");
        let declaration = format!("# TYPE {name} {kind}");
        assert_eq!(
            output
                .lines()
                .filter(|line| line.starts_with(&help))
                .count(),
            1
        );
        assert_eq!(
            output.lines().filter(|line| *line == declaration).count(),
            1
        );
        let count = samples(&output)
            .into_iter()
            .filter(|line| {
                let key = line.split(['{', ' ']).next().unwrap();
                key == name
                    || (kind == "histogram"
                        && key
                            .strip_prefix(name)
                            .is_some_and(|suffix| matches!(suffix, "_bucket" | "_count" | "_sum")))
            })
            .count();
        assert_eq!(count, expected, "family {name}");
    }
}

#[test]
fn progress_and_counters_keep_absolute_restored_values_and_coverage_baseline() {
    let value = snapshot();

    let output = render(Some(&value), false, true, 0, None).unwrap();

    for (name, expected) in [
        ("updates_completed", "103"),
        ("updates_target", "200"),
        ("samples_total", "456789"),
        ("optimizer_steps_total", "987"),
        ("metrics_start_update", "100"),
        ("parallel_worlds", "8"),
        ("games_per_update", "40"),
    ] {
        assert_eq!(
            sample_value(&output, &format!("drysua_training_{name}")),
            expected
        );
    }
}

#[test]
fn all_four_committed_outcomes_and_last_update_outcomes_are_distinct() {
    let value = snapshot();

    let output = render(Some(&value), true, true, 0, None).unwrap();

    for (outcome, total, last) in [
        ("win", "11", "1"),
        ("loss", "22", "2"),
        ("draw", "33", "3"),
        ("time_cap", "44", "4"),
    ] {
        let total_key = format!("drysua_training_games_total{{outcome=\"{outcome}\"}}");
        let last_key = format!("drysua_training_last_update_games{{outcome=\"{outcome}\"}}");
        assert_eq!(sample_value(&output, &total_key), total);
        assert_eq!(sample_value(&output, &last_key), last);
    }
}

#[test]
fn unavailable_history_exposes_only_status_and_optional_supplied_heartbeat() {
    let value = snapshot();
    let scopes = [DurationHistogram::default(); 3];
    for history in [None, Some(&value)] {
        for healthy in [false, true] {
            if history.is_some() && healthy {
                continue;
            }
            for active in [false, true] {
                for heartbeat in [0, 123] {
                    let output =
                        render(history, active, healthy, heartbeat, Some(&scopes)).unwrap();
                    let mut expected = vec![
                        format!("drysua_training_active {}", u8::from(active)),
                        "drysua_training_metrics_available 0".to_owned(),
                        format!(
                            "drysua_training_metrics_state_healthy {}",
                            u8::from(healthy)
                        ),
                    ];
                    if heartbeat > 0 {
                        expected.push(format!(
                            "drysua_training_last_heartbeat_timestamp_seconds {heartbeat}"
                        ));
                    }
                    expected.sort_unstable();
                    let mut actual = samples(&output);
                    actual.sort_unstable();
                    assert_eq!(actual, expected);
                    assert_eq!(output.lines().count(), expected.len() * 3);
                    assert!(!output.contains("games_total"));
                    assert!(!output.contains("duration_seconds"));
                }
            }
        }
    }
}

#[test]
fn healthy_snapshot_remains_available_while_training_is_inactive() {
    let value = snapshot();
    for active in [false, true] {
        let output = render(Some(&value), active, true, 0, None).unwrap();

        assert_eq!(
            sample_value(&output, "drysua_training_metrics_available"),
            "1"
        );
        assert_eq!(
            sample_value(&output, "drysua_training_metrics_state_healthy"),
            "1"
        );
        assert_eq!(
            sample_value(&output, "drysua_training_active"),
            u8::from(active).to_string()
        );
    }
}

#[test]
fn heartbeat_is_supplied_by_caller_and_zero_omits_the_entire_family() {
    let value = snapshot();
    let name = "drysua_training_last_heartbeat_timestamp_seconds";
    for heartbeat in [1, 456, u64::MAX] {
        let output = render(Some(&value), true, true, heartbeat, None).unwrap();

        assert_eq!(sample_value(&output, name), heartbeat.to_string());
    }
    let output = render(Some(&value), true, true, 0, None).unwrap();
    assert!(!output.contains(name));
}

#[test]
fn window_mode_exports_zero_games_per_update_without_omitting_configuration() {
    let mut value = snapshot();
    value.games_per_update = 0;

    let output = render(Some(&value), true, true, 0, None).unwrap();

    assert_eq!(
        sample_value(&output, "drysua_training_games_per_update"),
        "0"
    );
    assert_eq!(
        sample_value(&output, "drysua_training_parallel_worlds"),
        "8"
    );
}

#[test]
fn optional_losses_map_to_the_exact_gauges_without_renaming() {
    let value = snapshot();

    let output = render(Some(&value), true, true, 0, None).unwrap();

    for (name, expected) in [
        ("policy_loss", "-0.5"),
        ("value_loss", "0.25"),
        ("entropy", "0.75"),
        ("approximate_kl", "0.125"),
    ] {
        assert_eq!(
            sample_value(&output, &format!("drysua_training_{name}")),
            expected
        );
    }
}

#[test]
fn absent_optional_metrics_omit_samples_and_metadata_including_scope_histogram() {
    let mut value = snapshot();
    value.losses = None;
    value.generation = None;
    value.scale_bp = None;

    let output = render(Some(&value), true, true, 0, None).unwrap();

    for name in [
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "generation",
        "environment_scale_ratio",
        "scope_duration_seconds",
    ] {
        assert!(!output.contains(&format!("drysua_training_{name}")));
    }
}

#[test]
fn generation_is_zero_based_and_scale_basis_points_become_a_ratio() {
    for (generation, scale, ratio) in [
        (0, 0, "0"),
        (7, 1, "0.0001"),
        (9, 2500, "0.25"),
        (10, 10_000, "1"),
    ] {
        let mut value = snapshot();
        value.generation = Some(generation);
        value.scale_bp = Some(scale);

        let output = render(Some(&value), true, true, 0, None).unwrap();

        assert_eq!(
            sample_value(&output, "drysua_training_generation"),
            generation.to_string()
        );
        assert_eq!(
            sample_value(&output, "drysua_training_environment_scale_ratio"),
            ratio
        );
    }
}

#[test]
fn update_histogram_exports_each_inclusive_boundary_and_cumulative_infinity() {
    let mut value = snapshot();
    for seconds in BUCKETS {
        value.durations[0]
            .observe(Duration::from_secs_f64(seconds))
            .unwrap();
    }
    value.durations[0]
        .observe(Duration::from_secs(3601))
        .unwrap();

    let output = render(Some(&value), true, true, 0, None).unwrap();

    let bounds = [
        "0.001", "0.01", "0.1", "1", "5", "15", "60", "300", "900", "3600", "+Inf",
    ];
    let mut previous = 0;
    for (index, bound) in bounds.into_iter().enumerate() {
        let key = format!("drysua_training_update_duration_seconds_bucket{{le=\"{bound}\"}}");
        let count: u64 = sample_value(&output, &key).parse().unwrap();
        assert_eq!(count, (index + 1) as u64);
        assert!(count >= previous);
        previous = count;
    }
    assert_eq!(
        sample_value(&output, "drysua_training_update_duration_seconds_count"),
        "11"
    );
    assert_eq!(
        sample_value(&output, "drysua_training_update_duration_seconds_sum"),
        value.durations[0].sum_seconds.to_string()
    );
}

#[test]
fn duration_just_above_boundary_and_empty_histograms_keep_correct_buckets() {
    let mut value = snapshot();
    value.durations[0]
        .observe(Duration::from_nanos(1_000_001))
        .unwrap();

    let output = render(Some(&value), true, true, 0, None).unwrap();

    for (bound, expected) in [("0.001", "0"), ("0.01", "1"), ("+Inf", "1")] {
        let key = format!("drysua_training_update_duration_seconds_bucket{{le=\"{bound}\"}}");
        assert_eq!(sample_value(&output, &key), expected);
    }
    for suffix in ["count", "sum"] {
        let key =
            format!("drysua_training_stage_duration_seconds_{suffix}{{stage=\"collection\"}}");
        assert_eq!(sample_value(&output, &key), "0");
    }
}

#[test]
fn stage_and_invocation_scope_histograms_keep_their_exact_array_mapping() {
    let mut value = snapshot();
    for (index, histogram) in value.durations.iter_mut().enumerate() {
        histogram
            .observe(Duration::from_secs((index + 1) as u64))
            .unwrap();
    }
    let mut scopes = [DurationHistogram::default(); 3];
    for (index, histogram) in scopes.iter_mut().enumerate() {
        histogram
            .observe(Duration::from_secs((index + 10) as u64))
            .unwrap();
    }

    let output = render(Some(&value), true, true, 0, Some(&scopes)).unwrap();

    for (label, name, seconds) in [
        ("stage", "rollout_initialization", 2),
        ("stage", "collection", 3),
        ("stage", "batch_preparation", 4),
        ("stage", "optimization", 5),
        ("stage", "finalization", 6),
        ("scope", "session_initialization", 10),
        ("scope", "checkpoint_capture_save_runtime_export", 11),
        ("scope", "resume_runtime_export", 12),
    ] {
        let base = format!("drysua_training_{label}_duration_seconds");
        let sum_key = format!("{base}_sum{{{label}=\"{name}\"}}");
        let count_key = format!("{base}_count{{{label}=\"{name}\"}}");
        let infinity_key = format!("{base}_bucket{{{label}=\"{name}\",le=\"+Inf\"}}");
        assert_eq!(sample_value(&output, &sum_key), seconds.to_string());
        assert_eq!(sample_value(&output, &count_key), "1");
        assert_eq!(sample_value(&output, &infinity_key), "1");
    }
}

#[test]
fn every_nonfinite_loss_is_rejected_before_any_exposition_is_returned() {
    for index in 0..4 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut value = snapshot();
            let mut losses = [0.0; 4];
            losses[index] = invalid;
            value.losses = Some(losses);

            assert_invalid(
                render(Some(&value), true, true, 0, None),
                "metrics losses must be finite",
            );
        }
    }
}

#[test]
fn nonfinite_or_negative_duration_sums_are_rejected_for_every_training_histogram() {
    for index in 0..6 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut value = snapshot();
            value.durations[index].sum_seconds = invalid;

            assert_invalid(
                render(Some(&value), true, true, 0, None),
                "metrics duration sum must be finite and nonnegative",
            );
        }
    }
}

#[test]
fn invalid_snapshot_is_rejected_even_when_history_is_marked_unhealthy() {
    let mut value = snapshot();
    value.losses = Some([f64::NAN; 4]);

    assert_invalid(
        render(Some(&value), false, false, 0, None),
        "metrics losses must be finite",
    );
}

#[test]
fn invalid_snapshot_histogram_buckets_are_not_rendered_or_repaired() {
    let mut value = snapshot();
    value.durations[0].buckets[0] = 1;

    assert_invalid(
        render(Some(&value), true, true, 0, None),
        "metrics duration buckets must be cumulative and at most count",
    );
}

#[test]
fn invocation_scope_histograms_reject_nonfinite_or_negative_sums() {
    let value = snapshot();
    for index in 0..3 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut scopes = [DurationHistogram::default(); 3];
            scopes[index].sum_seconds = invalid;
            assert_invalid(
                render(Some(&value), true, true, 0, Some(&scopes)),
                "metrics scope duration sum must be finite and nonnegative",
            );
        }
    }
}

#[test]
fn invocation_scope_histograms_reject_inexact_counts_and_empty_nonzero_sums() {
    let value = snapshot();
    for index in 0..3 {
        for (histogram, message) in [
            (
                DurationHistogram {
                    count: 1_u64 << 53,
                    ..DurationHistogram::default()
                },
                "metrics scope duration count exceeds 9007199254740991",
            ),
            (
                DurationHistogram {
                    sum_seconds: 1.0,
                    ..DurationHistogram::default()
                },
                "metrics empty scope duration histogram must have zero sum",
            ),
        ] {
            let mut scopes = [DurationHistogram::default(); 3];
            scopes[index] = histogram;

            assert_invalid(render(Some(&value), true, true, 0, Some(&scopes)), message);
        }
    }
}

#[test]
fn invocation_scope_histograms_reject_excess_or_decreasing_buckets() {
    let value = snapshot();
    for index in 0..3 {
        for histogram in [
            DurationHistogram {
                buckets: [1; 10],
                ..DurationHistogram::default()
            },
            DurationHistogram {
                buckets: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                count: 1,
                sum_seconds: 1.0,
            },
        ] {
            let mut scopes = [DurationHistogram::default(); 3];
            scopes[index] = histogram;

            assert_invalid(
                render(Some(&value), true, true, 0, Some(&scopes)),
                "metrics scope duration buckets must be cumulative and at most count",
            );
        }
    }
}

#[test]
fn identifiers_never_affect_output_and_labels_need_no_external_escaping() {
    let original = snapshot();
    let mut changed = original.clone();
    changed.scope = [b'"'; 32];
    changed.checkpoint = [b'\n'; 32];
    let scopes = [DurationHistogram::default(); 3];

    let output = render(Some(&changed), true, true, 0, Some(&scopes)).unwrap();

    assert_eq!(
        output,
        render(Some(&original), true, true, 0, Some(&scopes)).unwrap()
    );
    assert!(!output.contains('\\'));
    for forbidden in [
        "run_id",
        "scope_id",
        "checkpoint=",
        "path=",
        "seed=",
        "drysua_host_",
        "drysua_gpu_",
    ] {
        assert!(!output.contains(forbidden));
    }
    for line in samples(&output) {
        if let Some((_, labels)) = line.split_once('{') {
            let (labels, _) = labels.split_once('}').unwrap();
            for label in labels.split(',') {
                let (key, quoted) = label.split_once('=').unwrap();
                assert!(matches!(key, "outcome" | "stage" | "scope" | "le"));
                let value = quoted.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
                assert!(
                    value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"_.+".contains(&byte))
                );
            }
        }
    }
}

#[test]
fn largest_integer_counters_and_extreme_finite_floats_fit_within_48_kib() {
    let maximum = (1_u64 << 53) - 1;
    let mut value = snapshot();
    value.start_update = 1_000_000;
    value.completed_updates = 1_000_000;
    value.updates_target = 1_000_000;
    value.samples = maximum;
    value.optimizer_steps = maximum;
    value.games = [maximum; 4];
    value.last_update_games = [maximum; 4];
    value.parallel = 40;
    value.generation = Some(u64::MAX);
    value.scale_bp = Some(10_000);
    value.losses = Some([f64::MIN, f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)]);
    for sum_seconds in [f64::MAX, f64::from_bits(1)] {
        let histogram = DurationHistogram {
            buckets: [maximum; 10],
            count: maximum,
            sum_seconds,
        };
        value.durations = [histogram; 6];
        let scopes = [histogram; 3];

        let output = render(Some(&value), true, true, u64::MAX, Some(&scopes)).unwrap();

        assert!(output.len() <= 48 * 1024);
        assert_eq!(samples(&output).len(), 142);
        for line in samples(&output) {
            let (_, number) = line.rsplit_once(' ').unwrap();
            assert!(number.parse::<f64>().unwrap().is_finite());
        }
    }
}

#[test]
fn bounded_writer_accepts_exact_limit_and_rejects_overflow_without_appending() {
    let mut output = Exposition {
        text: String::new(),
    };
    let prefix = "x".repeat(48 * 1024 - 1);
    output.write_str(&prefix).unwrap();

    assert!(output.write_str("yz").is_err());
    assert_eq!(output.text, prefix);
    output.write_str("z").unwrap();
    assert_eq!(output.text.len(), 48 * 1024);
    output.write_str("").unwrap();
    assert!(output.write_str("x").is_err());
    assert_eq!(output.text.len(), 48 * 1024);
}
