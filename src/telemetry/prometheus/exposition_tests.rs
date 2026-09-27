use super::*;
use std::time::Duration;

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

fn sample_value<'a>(output: &'a str, key: &str) -> &'a str {
    assert!(output.len() <= 48 * 1024);
    assert!(output.ends_with('\n'));
    let prefix = format!("{key} ");
    let mut values = output.lines().filter_map(|line| line.strip_prefix(&prefix));
    let value = values.next().expect("required sample");
    assert!(values.next().is_none(), "duplicate sample {key}");
    value
}

#[test]
fn unavailable_history_omits_counters_and_optional_metrics_are_not_fabricated() {
    let mut value = snapshot();
    value.losses = None;
    value.generation = None;
    value.scale_bp = None;
    for healthy in [false, true] {
        let output = render(Some(&value), false, healthy, 0, None).unwrap();
        assert_eq!(
            sample_value(&output, "drysua_training_metrics_available"),
            u8::from(healthy).to_string()
        );
        assert_eq!(output.contains("games_total"), healthy);
        for absent in [
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "generation",
            "environment_scale_ratio",
            "scope_duration_seconds",
            "last_heartbeat_timestamp_seconds",
        ] {
            assert!(!output.contains(absent));
        }
    }
}

#[test]
fn histogram_wire_names_boundaries_and_stage_mapping_are_stable() {
    let mut value = snapshot();
    for seconds in BUCKETS {
        value.durations[0]
            .observe(Duration::from_secs_f64(seconds))
            .unwrap();
    }
    value.durations[0]
        .observe(Duration::from_secs(3601))
        .unwrap();
    for (index, histogram) in value.durations[1..].iter_mut().enumerate() {
        histogram
            .observe(Duration::from_secs((index + 2) as u64))
            .unwrap();
    }
    let output = render(Some(&value), true, true, 0, None).unwrap();
    for (index, bound) in [
        "0.001", "0.01", "0.1", "1", "5", "15", "60", "300", "900", "3600", "+Inf",
    ]
    .into_iter()
    .enumerate()
    {
        let key = format!("drysua_training_update_duration_seconds_bucket{{le=\"{bound}\"}}");
        assert_eq!(sample_value(&output, &key), (index + 1).to_string());
    }
    for (index, stage) in [
        "rollout_initialization",
        "collection",
        "batch_preparation",
        "optimization",
        "finalization",
    ]
    .into_iter()
    .enumerate()
    {
        let key = format!("drysua_training_stage_duration_seconds_sum{{stage=\"{stage}\"}}");
        assert_eq!(sample_value(&output, &key), (index + 2).to_string());
    }
    assert!(output.contains("# TYPE drysua_training_update_duration_seconds histogram\n"));
    assert_eq!(
        sample_value(&output, "drysua_training_update_duration_seconds_count"),
        "11"
    );
}

#[test]
fn extreme_finite_values_remain_bounded_and_identifiers_never_become_labels() {
    let mut value = snapshot();
    let maximum = (1_u64 << 53) - 1;
    value.samples = maximum;
    value.optimizer_steps = maximum;
    value.games = [maximum; 4];
    value.losses = Some([f64::MIN, f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)]);
    let histogram = DurationHistogram {
        buckets: [maximum; 10],
        count: maximum,
        sum_seconds: f64::MAX,
    };
    value.durations = [histogram; 6];
    let scopes = [histogram; 3];
    let output = render(Some(&value), true, true, u64::MAX, Some(&scopes)).unwrap();
    value.scope = [b'"'; 32];
    value.checkpoint = [b'\n'; 32];
    assert_eq!(
        output,
        render(Some(&value), true, true, u64::MAX, Some(&scopes)).unwrap()
    );
    assert!(output.len() <= 48 * 1024);
    let samples: Vec<_> = output
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(samples.len(), 142);
    for line in samples {
        assert!(
            line.rsplit_once(' ')
                .unwrap()
                .1
                .parse::<f64>()
                .unwrap()
                .is_finite()
        );
        if let Some((_, labels)) = line.split_once('{') {
            for label in labels.split_once('}').unwrap().0.split(',') {
                assert!(matches!(
                    label.split_once('=').unwrap().0,
                    "outcome" | "stage" | "scope" | "le"
                ));
            }
        }
    }
}
