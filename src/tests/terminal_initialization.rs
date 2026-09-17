use super::*;
use safetensors::tensor::{Dtype, TensorView, serialize};

#[test]
fn terminal02_m21_source_metadata_and_reward5_descriptor_are_frozen() {
    let hash = reward_v5::DESCRIPTOR
        .bytes()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    assert_eq!(hash, 10775256611790261869);
    assert_eq!(
        source_metadata()["model_schema_hash"],
        "13521186719558157260"
    );
    assert_ne!(
        source_metadata()["map2_reward_schema_descriptor"],
        crate::MAP2_REWARD_SCHEMA_DESCRIPTOR
    );
}

#[test]
fn terminal02_parameter_import_preserves_signed_zero_and_all_bits_with_fresh_optimizer() {
    let mut source = vec![0.25; crate::MODEL_PARAMETER_COUNT];
    source[0] = -0.0;
    source[1] = -0.5;
    let model = initialize_parameters(&source, 9142200, PolicyDevice::Cpu).unwrap();
    assert_bits(&source, &model);
    let trainer = crate::PpoTrainer::new(&model, crate::PpoConfig::default(), 9142200).unwrap();
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn terminal02_initializer_rejects_wrong_pin_metadata_and_nonfinite_without_source_mutation() {
    let path = std::env::temp_dir().join(format!("terminal02-source-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let mut data = vec![0u8; crate::MODEL_PARAMETER_COUNT * 4];
    let mut metadata = source_metadata();
    for mode in 0..3 {
        if mode == 1 {
            metadata.insert("ppo_rules_audit_version".to_owned(), "30".to_owned());
        }
        if mode == 2 {
            metadata = source_metadata();
            data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        let tensor =
            TensorView::new(Dtype::F32, vec![crate::MODEL_PARAMETER_COUNT], &data).unwrap();
        let bytes = serialize([("model.parameters", tensor)], Some(metadata.clone())).unwrap();
        std::fs::write(path.join("drysua.weights.safetensors"), &bytes).unwrap();
        let error = TrainingArtifact::initialize_selected_m21_for_terminal_reward(
            &path,
            1,
            PolicyDevice::Cpu,
        )
        .err()
        .unwrap();
        match mode {
            0 => assert_eq!(
                error,
                CheckpointError::TensorContract(
                    "selected M21/u300 terminal initialization source SHA-256"
                )
            ),
            1 => assert_eq!(error, CheckpointError::SchemaMismatch),
            _ => assert_eq!(
                error,
                CheckpointError::NonFiniteTensor {
                    name: "model.parameters",
                    index: 0
                }
            ),
        }
        assert_eq!(
            std::fs::read(path.join("drysua.weights.safetensors")).unwrap(),
            bytes
        );
    }
    std::fs::remove_dir_all(path).unwrap();
}

fn assert_bits(source: &[f32], model: &PolicyModel) {
    let target = model.export_parameters().unwrap();
    assert_eq!(source.len(), crate::MODEL_PARAMETER_COUNT);
    assert_eq!(source.len(), target.len());
    assert!(
        source
            .iter()
            .zip(target)
            .all(|(source, target)| source.to_bits() == target.to_bits())
    );
}

#[test]
fn terminal02_ordinary_runtime_and_checkpoint_loads_reject_original_m21() {
    let path = std::env::temp_dir().join(format!("terminal02-runtime-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let model = PolicyModel::fresh(9142201).unwrap();
    let source = model.export_parameters().unwrap();
    let data = vec![0u8; crate::MODEL_PARAMETER_COUNT * 4];
    let tensor = TensorView::new(Dtype::F32, vec![crate::MODEL_PARAMETER_COUNT], &data).unwrap();
    let bytes = serialize([("model.parameters", tensor)], Some(source_metadata())).unwrap();
    std::fs::write(path.join("drysua.weights.safetensors"), &bytes).unwrap();
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &path),
        Err(CheckpointError::SchemaMismatch)
    );
    assert_bits(&source, &model);
    let mut header = b"DRYCKP18".to_vec();
    header.extend(9u32.to_le_bytes());
    header.extend(11471594345688682053u64.to_le_bytes());
    std::fs::write(path.join("checkpoint.meta"), &header).unwrap();
    assert_eq!(
        TrainingArtifact::load(&path).err(),
        Some(CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        std::fs::read(path.join("drysua.weights.safetensors")).unwrap(),
        bytes
    );
    assert_eq!(std::fs::read(path.join("checkpoint.meta")).unwrap(), header);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
#[ignore = "Authorized parameter-only M21/u300 initialization into a NEW directory under resource guard; no training."]
fn initialize_authorized_u300_terminal02_artifact_only() {
    let source = std::path::PathBuf::from(std::env::var_os("DRYSUA_TERMINAL_SOURCE").unwrap())
        .canonicalize()
        .unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("DRYSUA_TERMINAL_OUTPUT").unwrap());
    assert!(!output.exists());
    assert!(!output.starts_with(&source));
    let bytes = std::fs::read(source.join("drysua.weights.safetensors")).unwrap();
    let (model, provenance) = TrainingArtifact::initialize_selected_m21_for_terminal_reward(
        &source,
        9142200,
        PolicyDevice::Cpu,
    )
    .unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor("model.parameters").unwrap();
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    assert!(remainder.is_empty());
    let values: Vec<f32> = chunks
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    assert_bits(&values, &model);
    let config = crate::PpoConfig {
        gamma_tick: 1.0,
        ..crate::PpoConfig::default()
    };
    let trainer = crate::PpoTrainer::new(&model, config, 9142200).unwrap();
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer_step(), 0);
    let run = initialization_run(config, provenance.clone());
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
    assert_bits(&values, &reload);
    let restored = TrainingArtifact::load_compatible(&output, &run).unwrap();
    assert_eq!(restored.progress(), &progress);
    let state = restored.restore(&reload, &run).unwrap();
    assert_eq!(state.trainer().optimizer_step(), 0);
    assert_bits(&values, &reload);
    assert_eq!(
        bytes,
        std::fs::read(source.join("drysua.weights.safetensors")).unwrap()
    );
    println!("{provenance} output={}", output.display());
}

fn initialization_run(config: crate::PpoConfig, command_line: String) -> crate::CheckpointRun {
    assert!(!command_line.is_empty());
    assert_eq!(config.gamma_tick, 1.0);
    crate::CheckpointRun {
        mastery_config: None,
        git_commit: std::env::var("DRYSUA_GIT_COMMIT").unwrap(),
        simulator_commit: std::env::var("BOTA_GIT_COMMIT").unwrap(),
        enabled_features: crate::compiled_features(),
        command_line,
        run_seed: 9142200,
        map: crate::MAP2_ID,
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config.minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}
