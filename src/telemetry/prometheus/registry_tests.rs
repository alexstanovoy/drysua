#![cfg(unix)]

use super::*;

type InvalidObservation = (fn(&mut UpdateObservation), &'static str);

#[test]
fn http_publication_replay_and_resume_preserve_only_checkpointed_observations() {
    use super::super::state::{has_pending, read_snapshot};
    for crash_before_publish in [false, true] {
        let directory = Directory::new();
        prepare_invocation(&directory, crash_before_publish);
        let mut checkpoint = baseline();
        checkpoint.completed_updates = 42;
        checkpoint.start_update = 42;
        checkpoint.samples = 4200;
        checkpoint.optimizer_steps = 84;
        checkpoint.checkpoint = [3; 32];
        checkpoint.updates_target = 50;
        for _ in 0..2 {
            let mut resumed = directory.registry();
            resumed.begin(checkpoint.clone(), true).unwrap();
            let persisted = read_snapshot(&directory.0).unwrap();
            assert_eq!(persisted.updates_target, 50);
            assert_eq!(persisted.durations.map(|value| value.count), [2; 6]);
            assert!(!has_pending(&directory.0).unwrap());
            assert_exposition(&directory);
        }
        let mut resumed = directory.registry();
        resumed.begin(checkpoint, true).unwrap();
        resumed.observe(update(43, [1, 37, 0, 2])).unwrap();
        resumed.prepare([1; 32], 43, [4; 32], 124).unwrap();
        resumed.commit(43, [4; 32]).unwrap();
        let persisted = read_snapshot(&directory.0).unwrap();
        assert_eq!(persisted.games, [6, 108, 2, 4]);
        assert_eq!(persisted.last_update_games, [1, 37, 0, 2]);
        assert_eq!(persisted.samples, 4300);
        assert_eq!(persisted.optimizer_steps, 86);
        drop(resumed);
        let body = super::super::DirectoryReader::new(directory.0.clone())
            .scrape_for_test("GET /metrics HTTP/1.0\r\n\r\n");
        assert!(body.contains("drysua_training_active 0\n"));
    }
}

fn prepare_invocation(directory: &Directory, crash_before_publish: bool) {
    let mut first = directory.registry();
    first.begin(baseline(), true).unwrap();
    for (completed, games) in [(41, [3, 35, 1, 1]), (42, [5, 71, 2, 2])] {
        if completed == 42 {
            reject_invalid_observations(&mut first);
        }
        first.observe(update(completed, games)).unwrap();
        first
            .timing(
                completed - 1,
                Duration::from_secs(5),
                [Some(Duration::from_secs(1)); 5],
                true,
            )
            .unwrap();
    }
    for (result, message) in [
        (
            first.observe(update(42, [5, 71, 2, 2])),
            "metrics update must follow the previous completed update",
        ),
        (
            first.timing(41, Duration::ZERO, [None; 5], true),
            "metrics update timing is duplicated or out of order",
        ),
        (
            first.prepare([9; 32], 42, [3; 32], 123),
            "metrics checkpoint scope or update does not match staging",
        ),
    ] {
        assert_eq!(result.unwrap_err().to_string(), message);
    }
    first.prepare([1; 32], 42, [3; 32], 123).unwrap();
    first.prepare([1; 32], 42, [3; 32], 123).unwrap();
    let body = super::super::DirectoryReader::new(directory.0.clone())
        .scrape_for_test("GET /metrics HTTP/1.0\r\n\r\n");
    assert!(body.contains("drysua_training_updates_completed 40\n"));
    for outcome in ["win", "loss", "draw", "time_cap"] {
        assert!(body.contains(&format!(
            "drysua_training_games_total{{outcome=\"{outcome}\"}} 0\n"
        )));
    }
    if !crash_before_publish {
        first.commit(42, [3; 32]).unwrap();
        first.commit(42, [3; 32]).unwrap();
        assert_exposition(directory);
    }
}

fn baseline() -> TrainingSnapshot {
    TrainingSnapshot {
        scope: [1; 32],
        checkpoint: [2; 32],
        completed_updates: 40,
        updates_target: 100,
        samples: 4000,
        optimizer_steps: 80,
        start_update: 40,
        parallel: 8,
        games_per_update: 40,
        generation: Some(0),
        scale_bp: Some(2500),
        ..TrainingSnapshot::default()
    }
}

fn update(completed_updates: u64, games: [u64; 4]) -> UpdateObservation {
    UpdateObservation {
        completed_updates,
        samples: completed_updates * 100,
        optimizer_steps: completed_updates * 2,
        games,
        losses: [0.1, 0.2, 0.3, 0.01],
    }
}

fn reject_invalid_observations(registry: &mut Registry) {
    let cases: [InvalidObservation; 4] = [
        (
            |value| value.games[0] = 2,
            "metrics invocation outcome counters regressed",
        ),
        (
            |value| value.losses[0] = f64::NAN,
            "metrics losses must be finite",
        ),
        (
            |value| value.samples = 1,
            "metrics absolute sample or optimizer counters regressed",
        ),
        (
            |value| value.optimizer_steps = 1,
            "metrics absolute sample or optimizer counters regressed",
        ),
    ];
    for (invalidate, message) in cases {
        let mut observation = update(42, [5, 71, 2, 2]);
        invalidate(&mut observation);
        assert_eq!(
            registry.observe(observation).unwrap_err().to_string(),
            message
        );
    }
}

fn assert_exposition(directory: &Directory) {
    let output = super::super::DirectoryReader::new(directory.0.clone())
        .scrape_for_test("GET /metrics HTTP/1.1\r\nhOsT: [::1]:9100\r\nContent-Length: 0\r\n\r\n");
    let sample = |name: &str, value: &str| {
        let prefix = format!("drysua_training_{name} ");
        assert_eq!(
            output
                .lines()
                .filter_map(|line| line.strip_prefix(&prefix))
                .collect::<Vec<_>>(),
            [value],
            "{name}"
        );
    };
    for (name, value) in [
        ("updates_completed", "42"),
        ("samples_total", "4200"),
        ("optimizer_steps_total", "84"),
        ("metrics_start_update", "40"),
        ("policy_loss", "0.1"),
        ("value_loss", "0.2"),
        ("entropy", "0.3"),
        ("approximate_kl", "0.01"),
        ("generation", "0"),
        ("environment_scale_ratio", "0.25"),
        ("active", "1"),
        ("metrics_available", "1"),
    ] {
        sample(name, value);
    }
    for (outcome, total, last) in [
        ("win", "5", "2"),
        ("loss", "71", "36"),
        ("draw", "2", "1"),
        ("time_cap", "2", "1"),
    ] {
        sample(&format!("games_total{{outcome=\"{outcome}\"}}"), total);
        sample(&format!("last_update_games{{outcome=\"{outcome}\"}}"), last);
    }
}

struct Directory(std::path::PathBuf);

impl Directory {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "drysua-registry-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn registry(&self) -> Registry {
        Registry::new(Some(MetricsStore::open(&self.0).unwrap()))
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
