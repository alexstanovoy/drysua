use super::*;
use safetensors::tensor::{Dtype, TensorView, serialize};

#[test]
#[ignore = "Explicit authorized u162 to M21 initialization only; requires new output and frozen provenance under resource guard."]
fn initialize_authorized_u162_m21_artifact_only() {
    let source = std::path::PathBuf::from(std::env::var_os("DRYSUA_M21_SOURCE").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("DRYSUA_M21_OUTPUT").unwrap());
    assert!(!output.exists());
    assert!(!output.starts_with(source.canonicalize().unwrap()));
    let seed = 9_142_100;
    let bytes = std::fs::read(source.join("drysua.weights.safetensors")).unwrap();
    let (model, provenance) = TrainingArtifact::initialize_selected_m19_for_nonwin_reward(
        &source,
        seed,
        PolicyDevice::Cpu,
    )
    .unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor("model.parameters").unwrap();
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    assert!(remainder.is_empty());
    let parameters: Vec<f32> = chunks
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    assert_m19_padding(&parameters, &model);
    let config = crate::PpoConfig {
        gamma_tick: 1.0,
        ..crate::PpoConfig::default()
    };
    let trainer = crate::PpoTrainer::new(&model, config, seed).unwrap();
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.updates(), 0);
    let run = authorized_initialization_run(config, seed, provenance.description());
    let progress = crate::CheckpointProgress {
        mastery: None,
        global_update: 0,
        policy_version: 0,
        scheduler_step: 0,
        curriculum_stage: 0,
        rollout_samples: 0,
        best_evaluation: None,
        rng_states: vec![],
        league_references: vec![],
    };
    let artifact =
        TrainingArtifact::capture(&model, &trainer, run.clone(), progress.clone()).unwrap();
    std::fs::create_dir(&output).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &output).unwrap();
    assert_eq!(
        artifact.save(&output).unwrap(),
        crate::CheckpointSaveOutcome::Committed
    );
    let reload = PolicyModel::fresh(1).unwrap();
    TrainingArtifact::load_runtime_weights(&reload, &output).unwrap();
    assert_m19_padding(&parameters, &reload);
    let restored = TrainingArtifact::load_compatible(&output, &run).unwrap();
    assert_eq!(restored.progress(), &progress);
    let state = restored.restore(&reload, &run).unwrap();
    assert_eq!(state.trainer().optimizer_step(), 0);
    assert_m19_padding(&parameters, &reload);
    assert_eq!(
        bytes,
        std::fs::read(source.join("drysua.weights.safetensors")).unwrap()
    );
    println!("{} output={}", provenance.description(), output.display());
}

fn authorized_initialization_run(
    config: crate::PpoConfig,
    seed: u64,
    command_line: String,
) -> crate::CheckpointRun {
    assert_eq!(seed, 9_142_100);
    assert!(!command_line.is_empty());
    crate::CheckpointRun {
        mastery_config: None,
        git_commit: std::env::var("DRYSUA_GIT_COMMIT").unwrap(),
        simulator_commit: std::env::var("BOTA_GIT_COMMIT").unwrap(),
        enabled_features: crate::compiled_features(),
        command_line,
        run_seed: seed,
        map: crate::MAP2_ID,
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config.minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

#[test]
fn rebalance_rejects_exact_unpinned_m20_reward4_and_checkpoint8_without_relabelling() {
    let directory = std::env::temp_dir().join(format!("reward5-old-m20-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut metadata = source_metadata();
    for (name, value) in [
        ("feature_schema_hash", "5307034649837880808"),
        ("model_schema_hash", "17593713929660069669"),
        ("ppo_schema_version", "33"),
        ("ppo_schema_hash", "3388911021249010403"),
        ("ppo_rules_audit_version", "28"),
        ("map2_reward_schema_version", "4"),
        ("map2_reward_schema_hash", "14419233923370975736"),
    ] {
        metadata.insert(name.to_owned(), value.to_owned());
    }
    let descriptor = super::super::legacy_reward::reward_v4_descriptor();
    let hash = descriptor
        .bytes()
        .fold(0xcbf29ce484222325u64, |value, byte| {
            (value ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    assert_eq!(hash, 14_419_233_923_370_975_736);
    metadata.insert("map2_reward_schema_descriptor".to_owned(), descriptor);
    let data = vec![0u8; M19_PARAMETERS * 4];
    let tensor = TensorView::new(Dtype::F32, vec![M19_PARAMETERS], &data).unwrap();
    let bytes = serialize([("model.parameters", tensor)], Some(metadata)).unwrap();
    std::fs::write(directory.join("drysua.weights.safetensors"), &bytes).unwrap();
    let model = PolicyModel::fresh(9150000).unwrap();
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &directory),
        Err(CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        TrainingArtifact::initialize_selected_m19_for_nonwin_reward(
            &directory,
            9150001,
            PolicyDevice::Cpu
        )
        .err(),
        Some(CheckpointError::SchemaMismatch)
    );
    let mut header = b"DRYCKP18".to_vec();
    header.extend(8u32.to_le_bytes());
    header.extend(14_188_134_017_800_521_262u64.to_le_bytes());
    std::fs::write(directory.join("checkpoint.meta"), &header).unwrap();
    assert_eq!(
        TrainingArtifact::load(&directory).expect_err("old checkpoint"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(
        std::fs::read(directory.join("drysua.weights.safetensors")).unwrap(),
        bytes
    );
    assert_eq!(
        std::fs::read(directory.join("checkpoint.meta")).unwrap(),
        header
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn mastery_current_checkpoint_rejects_original_checkpoint7_before_tensor_access() {
    let path = std::env::temp_dir().join(format!("drysua-masteryv7-{}", std::process::id()));
    std::fs::create_dir(&path).expect("exclusive temporary directory");
    let mut bytes = b"DRYCKP18".to_vec();
    bytes.extend(7u32.to_le_bytes());
    bytes.extend(14_752_195_835_173_705_073u64.to_le_bytes());
    std::fs::write(path.join("checkpoint.meta"), &bytes).expect("old manifest");
    assert_eq!(
        TrainingArtifact::load(&path).expect_err("old schema"),
        CheckpointError::SchemaMismatch
    );
    assert_eq!(
        std::fs::read(path.join("checkpoint.meta")).expect("unchanged"),
        bytes
    );
    assert!(!path.join("checkpoint.safetensors").exists());
    std::fs::remove_dir_all(path).expect("cleanup");
}

#[test]
fn mastery_m19_initialization_freezes_reward3_and_only_accepts_the_u162_pin() {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in super::super::legacy_reward::MAP2_REWARD_V3_DESCRIPTOR.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    assert_eq!(hash, 11_643_768_462_079_275_437);
    let provenance = Map2NonwinInitializationProvenance::from_source_sha256(SOURCE).expect("pin");
    assert_eq!(provenance.source_sha256(), SOURCE);
    assert!(provenance.description().contains("mastery_window=fresh"));
    assert!(
        provenance
            .description()
            .contains("reward_equivalence=false")
    );
    assert_eq!(
        Map2NonwinInitializationProvenance::from_source_sha256([0; 32]),
        Err(CheckpointError::TensorContract(
            "selected M19 nonwin initialization source SHA-256"
        ))
    );
}

#[test]
fn mastery_m19_parameter_import_preserves_bits_and_fresh_optimizer_ownership() {
    let mut parameters: Vec<_> = (0..M19_PARAMETERS)
        .map(|index| f32::from_bits(0x3e00_0000 + index as u32))
        .collect();
    parameters[0] = -0.0;
    parameters[1] = -0.75;
    let model = initialize_model(&parameters, 9140001, PolicyDevice::Cpu).expect("model");
    assert_m19_padding(&parameters, &model);
    let trainer = crate::PpoTrainer::new(&model, crate::PpoConfig::default(), 9140002)
        .expect("fresh optimizer");
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.updates(), 0);
    let snapshot = trainer.checkpoint_snapshot(&model).expect("snapshot");
    assert!(
        snapshot
            .adam
            .moments()
            .0
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(
        snapshot
            .adam
            .moments()
            .1
            .iter()
            .all(|value| value.to_bits() == 0)
    );
}

#[test]
fn mastery_m19_current_loading_and_unpinned_initialization_fail_without_mutation() {
    let path = std::env::temp_dir().join(format!("drysua-m19-nonwin-{}", std::process::id()));
    std::fs::create_dir(&path).expect("exclusive temporary directory");
    let data = vec![0u8; M19_PARAMETERS * 4];
    let tensor = TensorView::new(Dtype::F32, vec![M19_PARAMETERS], &data).expect("tensor");
    let bytes = serialize([("model.parameters", tensor)], Some(source_metadata()))
        .expect("old source fixture");
    std::fs::write(path.join("drysua.weights.safetensors"), &bytes).expect("fixture");
    let model = PolicyModel::fresh(9140003).expect("model");
    let before = model.export_parameters().expect("before");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &path),
        Err(CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        TrainingArtifact::initialize_selected_m19_for_nonwin_reward(
            &path,
            9140004,
            PolicyDevice::Cpu
        )
        .err(),
        Some(CheckpointError::TensorContract(
            "selected M19 nonwin initialization source SHA-256"
        ))
    );
    assert_eq!(model.export_parameters().expect("unchanged"), before);
    assert_eq!(
        std::fs::read(path.join("drysua.weights.safetensors")).expect("source unchanged"),
        bytes
    );
    std::fs::remove_dir_all(path).expect("cleanup");
}

fn assert_m19_padding(source: &[f32], model: &PolicyModel) {
    let target = model.export_parameters().expect("target");
    let (mut before, mut after, mut zeros) = (0, 0, 0);
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        if name == "trunk.0.weight" {
            assert_eq!(shape, [2596, 512]);
            let prefix = 90 * 512;
            let inserted = 2 * 512;
            assert!(
                target[after + prefix..after + prefix + inserted]
                    .iter()
                    .all(|value| value.to_bits() == 0)
            );
            for index in 0..count - inserted {
                let offset = index + if index >= prefix { inserted } else { 0 };
                assert_eq!(
                    source[before + index].to_bits(),
                    target[after + offset].to_bits()
                );
            }
            before += count - inserted;
            zeros += inserted;
        } else {
            for index in 0..count {
                assert_eq!(
                    source[before + index].to_bits(),
                    target[after + index].to_bits()
                );
            }
            before += count;
        }
        after += count;
    }
    assert_eq!(before, M19_PARAMETERS);
    assert_eq!(after, crate::MODEL_PARAMETER_COUNT);
    assert_eq!(zeros, 1024);
}
