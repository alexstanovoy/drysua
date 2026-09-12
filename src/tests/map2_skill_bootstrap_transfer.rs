use super::*;
use std::io::Write as _;

#[path = "map2_skill_bootstrap_transfer_tests.rs"]
mod tests;

const ORIGINAL_RESULTS_SHA: &str =
    "2d2f9f51809900dab34ead125e3524cae5a85b0d4c3b15a83715a56008bfa8a4";
const FIXTURE_SHA: &str = "1f3929a03d109a5c9a4c6cccaff9ffe2c5fccd91565b96a59bf5ee4f69fb75b8";
const FIT_SHA: &str = "cc940ff5f7d9be6f63eb99e421ce3d97e7d9e2e87f17953fd4929589b515bc0a";

#[test]
#[ignore = "Authorized exact locked fit reproduction for diagnostic-only DEV transfer, not a local gate retry."]
fn reproduce_locked_fit_for_diagnostic_transfer() {
    // Historical gameplay reproduction must fail closed under a different legal-action set.
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 16);
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 14);
    let started = Instant::now();
    verify_locked_sources();
    let parent =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/map2-learning-20260911");
    let output = parent.join("skill-bootstrap-001-transfer");
    assert!(output.join("PLAN.md").exists());
    assert!(!output.join("REPRODUCTION.txt").exists());
    assert!(!output.join("diagnostic-weights").exists());
    let original_path = parent.join("skill-bootstrap-001/RESULTS.txt");
    assert_eq!(file_hash(&original_path), ORIGINAL_RESULTS_SHA);
    let original = std::fs::read_to_string(&original_path).unwrap();
    let source =
        std::env::var("DRYSUA_SKILL_BOOTSTRAP_INITIAL").expect("explicit unchanged initial path");
    let source = Path::new(&source);
    assert_eq!(file_hash(&source.join("drysua.weights.safetensors")), SHA);
    let model = PolicyModel::fresh_on(10091899, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, source).unwrap();
    let train = dataset(&specs(false, &[0, 2]), started);
    let held = specs(true, &[1]);
    let validation = dataset(&held, started);
    assert_eq!(train.len(), 340);
    assert_eq!(validation.len(), 170);
    assert_eq!(held.len(), 24);
    let mut report = provenance(&train, &validation);
    let baseline = agreement(&model, &validation);
    writeln!(report, "baseline_agreement={baseline:?}").unwrap();
    let before = free_running(&model, &held, &mut report, "initial", started);
    let (steps, initial_loss, final_loss) =
        fit_model(&model, &train, &validation, started, &mut report);
    assert_eq!(steps, 256, "incomplete locked reproduction: no export");
    let final_agreement = agreement(&model, &validation);
    writeln!(report, "final_agreement={final_agreement:?}\ninitial_fixed_loss={initial_loss} final_fixed_loss={final_loss}").unwrap();
    let after = free_running(&model, &held, &mut report, "fitted", started);
    assert_eq!(before.len(), 24);
    assert_eq!(after.len(), 24);
    connectivity(&model, &mut report);
    let differences = metric_differences(&original, &report);
    writeln!(report, "recorded_metrics_match={}\nmetric_differences={differences:?}\ninitial_completed={}/24\nfitted_completed={}/24\noptimizer_steps={steps}\nwall_seconds={:.3}", differences.is_empty(), before.iter().filter(|pass| **pass).count(), after.iter().filter(|pass| **pass).count(), started.elapsed().as_secs_f64()).unwrap();
    report.push_str("original_parameter_hash=unavailable_original_failed_gate_did_not_export\noriginal_training_data_hash=unavailable_first_fingerprint_recorded_in_reproduction\nlocal_gate_passed=false\nqualified=false\n");
    assert!(started.elapsed() < Duration::from_secs(300));
    assert_eq!(file_hash(&original_path), ORIGINAL_RESULTS_SHA);
    assert_eq!(file_hash(&source.join("drysua.weights.safetensors")), SHA);
    let target =
        save_diagnostic(&model, &output, &report).expect("new explicitly diagnostic artifact");
    write_new(&output.join("REPRODUCTION.txt"), &report).unwrap();
    eprintln!(
        "{report}\ndiagnostic_weights={}\nweights_sha256={}",
        target.display(),
        file_hash(&target.join("drysua.weights.safetensors"))
    );
}

fn verify_locked_sources() {
    let fixture = include_str!("map2_skill_bootstrap.rs").replace("pub(super) ", "");
    assert_eq!(hash_bytes(fixture.as_bytes()), FIXTURE_SHA);
    let fit = include_str!("map2_skill_bootstrap_fit.rs").replacen(
        "#[path = \"map2_skill_bootstrap_transfer.rs\"]\nmod transfer;\n",
        "",
        1,
    );
    assert_eq!(hash_bytes(fit.as_bytes()), FIT_SHA);
}

fn provenance(train: &[Row], validation: &[Row]) -> String {
    format!(
        "diagnostic_only=true\nexport_reason=authorized_transfer_diagnosis_of_failed_local_gate\norigin=validated fixture trajectories\nsource_init_sha256={SHA}\noriginal_results_sha256={ORIGINAL_RESULTS_SHA}\nfixture_source_sha256={FIXTURE_SHA}\nlocked_fit_source_sha256={FIT_SHA}\ntraining_data_sha256={}\nvalidation_data_sha256={}\ndata_hash_format=ordered_spec_tick_F14_tensor_F32_LE_bits_checked_BehavioralTarget_Debug_v1\ntrain_rows=340\nvalidation_rows=170\ntrain_variants=0,2\nvalidation_variants=1\ntrain_cases=48\nvalidation_cases=24\ntraining_namespace=10091700..10091899\nvalidation_namespace=10092700..10092899\nseed_model_and_sampling=10091899\noptimizer=Adam\nlearning_rate=0.0003\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8\ngradient_clip=0.5\nbatch=32\nsampling=uniform_rows_with_replacement\nupdates=256\nprecision=unchanged_F32\ndevice=CPU\n",
        data_hash(train),
        data_hash(validation)
    )
}

fn data_hash(rows: &[Row]) -> String {
    assert!(rows.len() <= 2048);
    let mut digest = Sha256::new();
    digest.update(b"ordered-fixture-F14-targets-v1");
    for row in rows {
        digest.update(format!(
            "{:?}:{}:{:?}:{:?}",
            row.spec,
            row.space.tick(),
            row.action,
            row.target
        ));
        let frame = &row.frame;
        let values = frame
            .global
            .iter()
            .chain(frame.history.iter().flatten())
            .chain(frame.policy_history.iter().flatten())
            .chain(frame.units.iter().flatten())
            .chain(frame.own_units.iter().flatten())
            .chain(frame.remembered_units.iter().flatten())
            .chain(frame.points.iter().flatten())
            .chain(frame.abilities.iter().flatten())
            .chain(frame.items.iter().flatten())
            .chain(frame.projectiles.iter().flatten())
            .chain(frame.loot.iter().flatten())
            .chain(frame.map.iter());
        for value in values {
            assert!(value.is_finite());
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn metric_differences(original: &str, reproduced: &str) -> Vec<String> {
    let prefixes = [
        "baseline_agreement=",
        "preview_agreement=",
        "final_agreement=",
        "initial_fixed_loss=",
        "effect15_connected_weights=",
        "initial spec=",
        "fitted spec=",
        "step=",
    ];
    let metrics = |text: &str| {
        text.lines()
            .filter(|line| prefixes.iter().any(|prefix| line.starts_with(prefix)))
            .map(|line| line.split(" fit_seconds=").next().unwrap().to_string())
            .collect::<Vec<_>>()
    };
    let expected = metrics(original);
    let actual = metrics(reproduced);
    assert!(expected.len() <= 64);
    assert!(actual.len() <= 64);
    let mut differences: Vec<_> = actual
        .iter()
        .filter(|line| !expected.contains(line))
        .cloned()
        .collect();
    differences.extend(
        expected
            .iter()
            .filter(|line| !actual.contains(line))
            .map(|line| format!("missing:{line}")),
    );
    differences
}

fn save_diagnostic(
    model: &PolicyModel,
    output: &Path,
    provenance: &str,
) -> std::io::Result<std::path::PathBuf> {
    assert!(provenance.len() < 128 * 1024);
    let target = output.join("diagnostic-weights");
    std::fs::create_dir(&target)?;
    TrainingArtifact::save_runtime_weights(model, &target).map_err(std::io::Error::other)?;
    let weights = target.join("drysua.weights.safetensors");
    let reload =
        PolicyModel::fresh_on(10091898, PolicyDevice::Cpu).map_err(std::io::Error::other)?;
    TrainingArtifact::load_runtime_weights(&reload, &target).map_err(std::io::Error::other)?;
    assert_eq!(
        model.export_parameters().unwrap(),
        reload.export_parameters().unwrap()
    );
    let manifest = format!(
        "diagnostic_only=true\nlocal_gate_passed=false\nqualified=false\nexport_reason=authorized_transfer_diagnosis_of_failed_local_gate\nweights_sha256={}\nmodel_schema={}\nfeature_schema={}\naction_schema={}\n{provenance}",
        file_hash(&weights),
        crate::MODEL_SCHEMA_VERSION,
        crate::FEATURE_SCHEMA_VERSION,
        crate::ACTION_SCHEMA_VERSION
    );
    write_new(&target.join("MANIFEST.txt"), &manifest)?;
    let mut permissions = std::fs::metadata(&weights)?.permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(weights, permissions)?;
    Ok(target)
}

fn write_new(path: &Path, text: &str) -> std::io::Result<()> {
    assert!(text.len() < 128 * 1024);
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}
fn hash_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn file_hash(path: &Path) -> String {
    assert!(std::fs::metadata(path).unwrap().len() < 16 * 1024 * 1024);
    hash_bytes(&std::fs::read(path).unwrap())
}
