use super::*;

#[test]
fn absent_options_do_not_enable_or_initialize_global_metrics() {
    let guard = MetricsOptions::default().start().unwrap();
    assert!(!guard.installed);
    assert!(guard.server.is_none());
    guard.finish().unwrap();
}

#[test]
fn training_listener_rejects_nonloopback_before_creating_state() {
    let options = MetricsOptions {
        metrics_listen: Some("0.0.0.0:9464".parse().unwrap()),
        metrics_directory: Some(PathBuf::from("/does/not/exist")),
    };
    assert_eq!(
        options.start().err().unwrap().to_string(),
        "metrics listener must bind a loopback address"
    );
}

#[test]
fn shared_checkpoint_and_metrics_directory_is_rejected_before_writes() {
    let directory = std::env::temp_dir();
    let options = MetricsOptions {
        metrics_directory: Some(directory.clone()),
        metrics_listen: None,
    };
    assert_eq!(
        options
            .validate_checkpoint_directory(&directory)
            .unwrap_err()
            .to_string(),
        "metrics and checkpoint directories must be separate, non-nested directories"
    );
}

#[test]
fn missing_exporter_state_reports_unavailable_without_fabricating_games() {
    let rendered =
        DirectoryReader::new(std::env::temp_dir().join("drysua-metrics-no-such-directory"))
            .render()
            .unwrap();
    assert!(rendered.contains("drysua_training_metrics_available 0\n"));
    assert!(rendered.contains("drysua_training_metrics_state_healthy 0\n"));
    assert!(!rendered.contains("drysua_training_games_total{"));
}

#[test]
fn active_writer_does_not_fabricate_a_trainer_heartbeat() {
    let snapshot = snapshot::TrainingSnapshot {
        parallel: 1,
        updates_target: 1,
        heartbeat: 123,
        ..snapshot::TrainingSnapshot::default()
    };
    let rendered =
        exposition::render(Some(&snapshot), true, true, snapshot.heartbeat, None).unwrap();
    assert!(rendered.contains("drysua_training_active 1\n"));
    assert!(rendered.contains("drysua_training_last_heartbeat_timestamp_seconds 123\n"));
}
