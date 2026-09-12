use super::*;

#[test]
fn diagnostic_export_preserves_current_parameters_and_failed_gate_provenance() {
    let root = std::env::temp_dir().join(format!("skill-transfer-export-{}", std::process::id()));
    assert_eq!(root.parent(), Some(std::env::temp_dir().as_path()));
    assert!(!root.exists());
    std::fs::create_dir(&root).unwrap();
    let model = PolicyModel::fresh_on(10091899, PolicyDevice::Cpu).unwrap();
    let provenance = format!(
        "source_init_sha256={SHA}\ntraining_data_sha256={}\n",
        "0".repeat(64)
    );
    let target = save_diagnostic(&model, &root, &provenance).unwrap();
    let manifest = std::fs::read_to_string(target.join("MANIFEST.txt")).unwrap();
    assert!(manifest.contains("diagnostic_only=true\nlocal_gate_passed=false\nqualified=false\n"));
    assert!(manifest.contains("export_reason=authorized_transfer_diagnosis_of_failed_local_gate"));
    assert!(manifest.contains("model_schema=17\nfeature_schema=15\naction_schema=5\n"));
    assert!(manifest.contains(&format!(
        "weights_sha256={}",
        file_hash(&target.join("drysua.weights.safetensors"))
    )));
    let reload = PolicyModel::fresh_on(10091898, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&reload, &target).unwrap();
    assert_eq!(
        model.export_parameters().unwrap(),
        reload.export_parameters().unwrap()
    );
    let error = save_diagnostic(&model, &root, &provenance).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn training_fingerprint_is_stable_but_detects_tensor_label_and_row_order_changes() {
    let spec = specs(false, &[0])[0];
    let mut rows = collect(spec);
    let expected = data_hash(&rows);
    assert_eq!(expected, data_hash(&collect(spec)));
    let before = rows[0].frame.global[0];
    rows[0].frame.global[0] = before + 1.0;
    assert_ne!(expected, data_hash(&rows));
    rows[0].frame.global[0] = before;
    rows.swap(0, 1);
    assert_ne!(expected, data_hash(&rows));
    rows.swap(0, 1);
    rows[0].target.kind.selected = ActionKind::Continue.index();
    assert_ne!(expected, data_hash(&rows));
}

#[test]
fn reproduction_comparison_ignores_only_wall_times_and_detects_changed_outcomes() {
    let original = "initial_fixed_loss=1 final_fixed_loss=0.5\nfitted spec=X complete=false\nstep=8 batch_loss=1 fixed_probe_loss=0.5 fit_seconds=1\n";
    let timed = original.replace("fit_seconds=1", "fit_seconds=2");
    assert!(metric_differences(original, &timed).is_empty());
    let changed = original.replace("complete=false", "complete=true");
    assert_eq!(
        metric_differences(original, &changed),
        [
            "fitted spec=X complete=true",
            "missing:fitted spec=X complete=false"
        ]
    );
}

#[test]
fn locked_fixture_and_fit_sources_are_unchanged_except_test_wiring() {
    verify_locked_sources();
}
