use super::*;

const BASELINE: &str = "e94e39a2979607c4ed38c3314fc0cbb1c5d4c125";

fn sources() -> Vec<(&'static str, String)> {
    let mut paths = vec![
        "src/tests/map2_recovery_004.rs",
        "src/tests/map2_recovery_004_audit.rs",
        "src/tests/map2_recovery_004_tests.rs",
        "src/tests/map2_recovery_004_data.rs",
        "src/tests/map2_recovery_004_fit.rs",
        "src/tests/map2_recovery_004_eval.rs",
        "src/tests/map2_recovery_003.rs",
        "src/tests/map2_recovery_003_train.rs",
        "src/tests/map2_recovery_003_goal.rs",
        "src/tests/map2_recovery_002.rs",
        "src/tests/map2_recovery_002_trip.rs",
        "src/tests/map2_recovery_002_fit.rs",
        "src/tests/map2_advantage.rs",
        "src/tests/map2_advantage_rank.rs",
        "src/model.rs",
        "src/model_advantage_fit.rs",
        "src/model_transfer_fit.rs",
        "artifacts/temp/map2-gameplay-fix-20260912/recovery-004/PLAN.md",
        "artifacts/temp/map2-gameplay-fix-20260912/recovery-004/run_guard.py",
    ];
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            (
                path,
                file_hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join(path)),
            )
        })
        .collect()
}

fn freeze_sources(sources: &[(&str, String)]) {
    assert!(sources.len() <= 32);
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(head.status.success());
    assert_eq!(String::from_utf8(head.stdout).unwrap().trim(), BASELINE);
    let diff = std::process::Command::new("git")
        .args(["diff", "--binary", BASELINE])
        .output()
        .unwrap();
    assert!(diff.status.success());
    write_new(
        &root4().join("PREFIT_DIFF.patch"),
        &String::from_utf8(diff.stdout).unwrap(),
    );
    for (name, hash) in sources {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        let target = root4().join("frozen-source").join(name);
        assert!(!target.exists());
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(&source, &target).unwrap();
        assert_eq!(file_hash(&target), *hash);
        let mut permissions = std::fs::metadata(&target).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&target, permissions).unwrap();
    }
}

#[test]
#[ignore = "One recovery004 collection/fresh-optimizer fit; predeclared config, never resume."]
fn collect_freeze_and_fit_once() {
    let started = Instant::now();
    assert!(root4().join("PROOF.txt").exists());
    assert!(!root4().join("weights").exists());
    let mut report = String::new();
    let dataset = data::collect(&mut report);
    let hashes = dataset.hashes();
    let source_hashes = sources();
    freeze_sources(&source_hashes);
    writeln!(report, "rows={} data_hashes={hashes:?} source_hashes={source_hashes:?}\nbaseline={BASELINE}\nstart_sha={INITIAL_SHA}\nreference_sha={PARENT_SHA}\ncollection_seconds={:.3}\nfrozen_before_optimizer=true", dataset.count(), started.elapsed().as_secs_f64()).unwrap();
    write_new(&root4().join("DATASET.txt"), &report);
    let model = initial();
    let mut fit_report = String::new();
    let steps = optimize(&model, &dataset, started, &mut fit_report);
    assert_eq!(hashes, dataset.hashes());
    assert_eq!(source_hashes, sources());
    writeln!(
        fit_report,
        "steps={steps} total_seconds={:.3}",
        started.elapsed().as_secs_f64()
    )
    .unwrap();
    write_new(&root4().join("FIT.txt"), &fit_report);
    assert_eq!(steps, 512, "incomplete bounded fit: stop, do not resume");
    let target = root4().join("weights");
    std::fs::create_dir(&target).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &target).unwrap();
    let hash = file_hash(&target.join("drysua.weights.safetensors"));
    write_new(
        &target.join("MANIFEST.txt"),
        &format!(
            "diagnostic_only=true\nqualified=false\ncandidate_pending_outcomes=true\nweights_sha256={hash}\nbaseline={BASELINE}\nstart_sha={INITIAL_SHA}\nreference_sha={PARENT_SHA}\nmodel=17 feature=15 action=5 ppo=30 rules=25\nfresh_Adam_lr.0001_beta.9_.999_eps1e-8_clip.5_steps512_batch16_CPU_rng10102999\nordinary8_journey4_counterfactual4_context_balanced=true\ndata_hashes={hashes:?}\ndataset_sha={}\nsource_hashes={source_hashes:?}\n",
            file_hash(&root4().join("DATASET.txt"))
        ),
    );
    eprintln!("{fit_report}\nrows={} weights_sha={hash}", dataset.count());
}

fn optimize(
    model: &PolicyModel,
    dataset: &data::Dataset,
    started: Instant,
    report: &mut String,
) -> u64 {
    let mut optimizer = model
        .claim_optimizer(AdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .unwrap();
    let mut random = PpoRng::new(10102999);
    let fit_start = Instant::now();
    for step in 1..=512 {
        if started.elapsed() >= Duration::from_secs(265) {
            break;
        }
        let batch = dataset.batch(step, &mut random);
        let update = model
            .train_checked_action_sets(&batch, &mut optimizer)
            .unwrap();
        if step == 1 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} sampled_loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_start.elapsed().as_secs_f64()
            )
            .unwrap();
        }
    }
    assert!(optimizer.step() <= 512);
    optimizer.step()
}

pub(super) fn fitted() -> PolicyModel {
    let model = PolicyModel::fresh_on(10102999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &root4().join("weights")).unwrap();
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    model
}
