use super::*;

#[test]
fn fresh_session_imports_only_initial_parameters_and_resets_all_training_state() {
    let directory = crate::ppo::test_directory("side-actors-fresh-session");
    let source_directory = directory.join("source");
    let target_directory = directory.join("new-checkpoint");
    std::fs::create_dir(&source_directory).unwrap();
    std::fs::create_dir(&target_directory).unwrap();
    let source = PolicyModel::fresh(428).unwrap();
    let expected = source.export_parameters().unwrap();
    TrainingArtifact::save_runtime_weights(&source, &source_directory).unwrap();
    std::fs::write(
        source_directory.join("checkpoint.meta"),
        b"old state must not be read",
    )
    .unwrap();
    std::fs::write(
        source_directory.join("checkpoint.safetensors"),
        b"old Adam must not be read",
    )
    .unwrap();
    let config = PpoConfig::default();
    let session = TrainingSession::initialize(
        PolicyDevice::Cpu,
        &target_directory,
        false,
        Some(&source_directory),
        config,
        initialization_run(config),
    )
    .unwrap();
    let actual = session.model.export_parameters().unwrap();
    assert_eq!(actual.len(), crate::MODEL_PARAMETER_COUNT);
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );
    assert_eq!(session.trainer.optimizer_step(), 0);
    assert_eq!(session.trainer.updates(), 0);
    assert_eq!(session.completed_updates, 0);
    assert_eq!(session.rollout_samples, 0);
    assert!(session.collection.is_none());
    assert_eq!(session.trainer.rng_checkpoint(), (9001 ^ 0x51a9, 0));
    assert!(session.adaptive_environment.is_none());
    let snapshot = session.trainer.checkpoint_snapshot(&session.model).unwrap();
    let (first, second) = snapshot.adam.moments().unwrap();
    assert!(first.iter().all(|value| value.to_bits() == 0));
    assert!(second.iter().all(|value| value.to_bits() == 0));
    assert_eq!(
        std::fs::read(source_directory.join("checkpoint.meta")).unwrap(),
        b"old state must not be read"
    );
    assert_eq!(
        std::fs::read(source_directory.join("checkpoint.safetensors")).unwrap(),
        b"old Adam must not be read"
    );
    assert_eq!(std::fs::read_dir(target_directory).unwrap().count(), 0);
}

fn initialization_run(config: PpoConfig) -> CheckpointRun {
    CheckpointRun {
        git_commit: "side-actor-test".into(),
        simulator_commit: "side-actor-simulator".into(),
        enabled_features: crate::compiled_features(),
        command_line: "side-actor-initialization-test".into(),
        run_seed: 9001,
        map: MapId(2),
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config.minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}
