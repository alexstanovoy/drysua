//! Long-lived session contracts: signal stop and milestone runtime history.

use std::num::NonZeroU64;

use super::*;

const CHILD: &str = "DRYSUA_SIGNAL_STOP_CHILD";

/// The stop flag is process-global, so the scenario runs in a child test process.
#[test]
fn signal_stop_commits_each_in_flight_update_and_history_keeps_milestones() {
    if std::env::var_os(CHILD).is_some() {
        signal_stop_scenario();
        return;
    }
    let module = module_path!().split_once("::").expect("crate path").1;
    let name =
        format!("{module}::signal_stop_commits_each_in_flight_update_and_history_keeps_milestones");
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", &name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .expect("child test process");
    assert!(
        output.status.success(),
        "child scenario failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

fn signal_stop_scenario() {
    let checkpoint = test_directory("signal-stop-checkpoint");
    let history = test_directory("signal-stop-history");
    let mut config = settings(0x5eed, 3);
    // A day-long interval: every commit below comes from the stop or the final update.
    config.checkpoint_cadence =
        crate::TrainingCheckpointCadence::WallTime(std::time::Duration::from_secs(86_400));
    config.history = Some(crate::RuntimeHistory {
        directory: history.clone(),
        every: NonZeroU64::new(2).expect("nonzero"),
    });
    crate::training_signals::install().expect("stop handlers");
    // SAFETY: The installed handler only stores an atomic flag.
    assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
    assert!(crate::training_signals::stop_requested());
    for (update, resume) in [(1, false), (2, true), (3, true)] {
        let commits = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = commits.clone();
        let report = run_annealed_job_harnessed(
            config.clone(),
            harness(),
            PolicyDevice::Cpu,
            &checkpoint,
            resume,
            None,
            move |committed| recorder.lock().unwrap().push(committed.completed_updates),
        )
        .expect("stopped session");
        assert_eq!(
            report.completed_updates, update,
            "one committed update per session"
        );
        assert_eq!(
            *commits.lock().unwrap(),
            [update],
            "the stop is the session's only checkpoint"
        );
        let artifact = TrainingArtifact::load(&checkpoint).expect("committed checkpoint");
        assert_eq!(artifact.progress().global_update, update);
    }
    assert_eq!(history_entries(&history), ["u0002", "u0003"]);
    let milestone = PolicyModel::fresh(0).expect("milestone model");
    TrainingArtifact::load_runtime_weights(&milestone, &history.join("u0003"))
        .expect("final milestone weights");
    let committed = PolicyModel::fresh(0).expect("committed model");
    TrainingArtifact::load_runtime_weights(&committed, &checkpoint).expect("runtime weights");
    assert_eq!(
        milestone.export_parameters().expect("milestone parameters"),
        committed.export_parameters().expect("committed parameters")
    );
    let report = run(config, &checkpoint, true).expect("completed resume");
    assert_eq!(report.completed_updates, 3);
    assert_eq!(history_entries(&history), ["u0002", "u0003"]);
    for directory in [checkpoint, history] {
        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}

fn history_entries(directory: &std::path::Path) -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(directory)
        .expect("history directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}
