use super::*;

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

fn registry() -> Registry {
    let mut registry = Registry::new(None);
    registry.begin(baseline(), true).expect("first coverage");
    registry
}

#[test]
fn completed_update_outcomes_are_invisible_until_checkpoint_commit() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    assert_eq!(registry.committed.as_ref().unwrap().games, [0; 4]);
    registry.prepare([1; 32], 41, [3; 32], 123).unwrap();
    assert_eq!(registry.committed.as_ref().unwrap().completed_updates, 40);
    registry.commit(41, [3; 32]).unwrap();
    assert_eq!(registry.committed.as_ref().unwrap().games, [3, 35, 1, 1]);
}

#[test]
fn invocation_cumulative_outcomes_are_differenced_once_per_update() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    registry.observe(update(42, [5, 71, 2, 2])).unwrap();
    registry.prepare([1; 32], 42, [3; 32], 123).unwrap();
    registry.commit(42, [3; 32]).unwrap();
    let committed = registry.committed.as_ref().unwrap();
    assert_eq!(committed.games, [5, 71, 2, 2]);
    assert_eq!(committed.last_update_games, [2, 36, 1, 1]);
}

#[test]
fn resumed_invocation_zero_counters_do_not_subtract_previous_history() {
    let mut resumed = baseline();
    resumed.start_update = 10;
    resumed.games = [50, 1000, 20, 130];
    let mut registry = Registry::new(None);
    registry.begin(resumed, true).unwrap();
    registry.observe(update(41, [1, 37, 0, 2])).unwrap();
    registry.prepare([1; 32], 41, [3; 32], 123).unwrap();
    registry.commit(41, [3; 32]).unwrap();
    assert_eq!(
        registry.committed.as_ref().unwrap().games,
        [51, 1037, 20, 132]
    );
}

#[test]
fn duplicate_update_is_rejected_without_double_counting() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    let before = registry.staged.clone();
    let error = registry.observe(update(41, [3, 35, 1, 1])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "metrics update must follow the previous completed update"
    );
    assert_eq!(registry.staged, before);
}

#[test]
fn failed_or_reordered_update_counters_leave_staging_unchanged() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    let before = registry.staged.clone();
    let error = registry.observe(update(42, [2, 71, 2, 2])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "metrics invocation outcome counters regressed"
    );
    assert_eq!(registry.staged, before);
}

#[test]
fn nonfinite_loss_is_rejected_without_publishing_update() {
    let mut registry = registry();
    let mut observation = update(41, [3, 35, 1, 1]);
    observation.losses[0] = f64::NAN;
    assert_eq!(
        registry.observe(observation).unwrap_err().to_string(),
        "metrics losses must be finite"
    );
    assert_eq!(registry.staged.as_ref().unwrap().completed_updates, 40);
}

#[test]
fn invalid_or_duplicate_timing_never_increments_histograms() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    let stages = [Some(Duration::from_secs(1)); 5];
    registry
        .timing(40, Duration::from_secs(5), stages, false)
        .unwrap();
    assert_eq!(registry.staged.as_ref().unwrap().durations[0].count, 0);
    registry
        .timing(40, Duration::from_secs(5), stages, true)
        .unwrap();
    assert_eq!(
        registry
            .timing(40, Duration::from_secs(5), stages, true)
            .unwrap_err()
            .to_string(),
        "metrics update timing is duplicated or out of order"
    );
    assert_eq!(registry.staged.as_ref().unwrap().durations[0].count, 1);
}

#[test]
fn committed_histograms_survive_idempotent_checkpoint_commit() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    registry
        .timing(
            40,
            Duration::from_secs(5),
            [Some(Duration::from_secs(1)); 5],
            true,
        )
        .unwrap();
    registry.prepare([1; 32], 41, [3; 32], 123).unwrap();
    registry.commit(41, [3; 32]).unwrap();
    registry.commit(41, [3; 32]).unwrap();
    assert_eq!(registry.committed.as_ref().unwrap().durations[0].count, 1);
}

#[test]
fn wrong_checkpoint_scope_or_commit_identity_cannot_publish() {
    let mut registry = registry();
    registry.observe(update(41, [3, 35, 1, 1])).unwrap();
    assert_eq!(
        registry
            .prepare([9; 32], 41, [3; 32], 123)
            .unwrap_err()
            .to_string(),
        "metrics checkpoint scope or update does not match staging"
    );
    registry.prepare([1; 32], 41, [3; 32], 123).unwrap();
    assert_eq!(
        registry.commit(41, [9; 32]).unwrap_err().to_string(),
        "metrics commit does not match a prepared snapshot"
    );
    assert_eq!(registry.committed.as_ref().unwrap().games, [0; 4]);
}

#[test]
fn update_progress_counter_regression_is_rejected() {
    let mut registry = registry();
    let mut observation = update(41, [3, 35, 1, 1]);
    observation.samples = 1;
    assert_eq!(
        registry.observe(observation).unwrap_err().to_string(),
        "metrics absolute sample or optimizer counters regressed"
    );
    assert_eq!(registry.staged.as_ref().unwrap().samples, 4000);
}

#[test]
fn update_target_is_available_before_first_new_checkpoint() {
    let mut registry = Registry::new(None);
    let mut current = baseline();
    current.updates_target = 200;
    registry.begin(current, true).unwrap();
    assert_eq!(registry.committed.as_ref().unwrap().updates_target, 200);
    assert_eq!(registry.committed.as_ref().unwrap().games, [0; 4]);
}

#[cfg(unix)]
struct Directory(std::path::PathBuf);

#[cfg(unix)]
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

#[cfg(unix)]
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn persisted_registry_resume_restarts_session_counters_and_accepts_lower_target() {
    let directory = Directory::new();
    let mut first = directory.registry();
    first.begin(baseline(), true).unwrap();
    first.observe(update(41, [3, 35, 1, 1])).unwrap();
    first.prepare([1; 32], 41, [3; 32], 123).unwrap();
    first.commit(41, [3; 32]).unwrap();
    drop(first);
    let mut restored = baseline();
    restored.completed_updates = 41;
    restored.start_update = 41;
    restored.samples = 4100;
    restored.optimizer_steps = 82;
    restored.checkpoint = [3; 32];
    restored.updates_target = 50;
    let mut second = directory.registry();
    second.begin(restored, true).unwrap();
    assert_eq!(second.committed.as_ref().unwrap().updates_target, 50);
    second.observe(update(42, [2, 36, 1, 1])).unwrap();
    second.prepare([1; 32], 42, [4; 32], 124).unwrap();
    second.commit(42, [4; 32]).unwrap();
    let snapshot = super::super::state::read_snapshot(&directory.0).unwrap();
    assert_eq!(snapshot.games, [5, 71, 2, 2]);
    assert_eq!(snapshot.last_update_games, [2, 36, 1, 1]);
    assert_eq!(snapshot.start_update, 40);
    assert_eq!(snapshot.samples, 4200);
    assert_eq!(snapshot.optimizer_steps, 84);
}

#[cfg(unix)]
#[test]
fn registry_crash_after_checkpoint_before_metrics_publish_recovers_once() {
    let directory = Directory::new();
    let mut first = directory.registry();
    first.begin(baseline(), true).unwrap();
    first.observe(update(41, [3, 35, 1, 1])).unwrap();
    first.prepare([1; 32], 41, [3; 32], 123).unwrap();
    drop(first);
    let mut actual_checkpoint = baseline();
    actual_checkpoint.completed_updates = 41;
    actual_checkpoint.start_update = 41;
    actual_checkpoint.samples = 4100;
    actual_checkpoint.optimizer_steps = 82;
    actual_checkpoint.checkpoint = [3; 32];
    for _ in 0..2 {
        let mut resumed = directory.registry();
        resumed.begin(actual_checkpoint.clone(), true).unwrap();
        assert_eq!(resumed.committed.as_ref().unwrap().games, [3, 35, 1, 1]);
        assert_eq!(resumed.committed.as_ref().unwrap().start_update, 40);
    }
    assert!(!super::super::state::has_pending(&directory.0).unwrap());
}
