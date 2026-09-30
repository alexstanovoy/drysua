use crate::ppo::test_directory;
use std::fs;

use bota_proto::{MapId, Team};
use safetensors::tensor::{Dtype, TensorView, serialize};

use super::feature::{encode, tracker_with_view, world_view};
use super::map2_checkpoint::{Directory, runtime_bytes};
use crate::{
    ActionSpace, CheckpointDevice, CheckpointError, CheckpointProgress, CheckpointRun,
    LocalPolicyState, PolicyModel, PpoConfig, PpoOutcome, PpoRng, PpoRollout, PpoTrainer,
    RngCheckpoint, TrainingArtifact,
};

#[path = "checkpoint_training_contract.rs"]
mod checkpoint_training_contract;

#[test]
fn capture_requires_map2_scope_rules_and_reward_discount() {
    let rules = crate::PPO_RULES_AUDIT_VERSION;
    for (map, gamma_tick, rules, expected) in [
        (2, 1.0, rules, None),
        (0, 1.0, rules, Some("hero or map scope")),
        (1, 1.0, rules, Some("hero or map scope")),
        (3, 1.0, rules, Some("hero or map scope")),
        (2, 0.99, rules, Some("Map2 reward discount")),
        (2, 1.0, rules - 1, Some("rules audit or batch size")),
    ] {
        let model = PolicyModel::fresh(191).expect("model");
        let trainer = PpoTrainer::new(
            &model,
            PpoConfig {
                gamma_tick,
                ..checkpoint_config()
            },
            192,
        )
        .expect("trainer");
        let run = CheckpointRun {
            map: MapId(map),
            rules_audit_version: rules,
            ..run_metadata()
        };
        let result = TrainingArtifact::capture(
            &model,
            &trainer,
            run,
            progress_metadata(0),
            crate::checkpoint::collection_fixture(&model),
        );
        if let Some(field) = expected {
            let error = result.expect_err("invalid scope");
            assert_eq!(error, CheckpointError::InvalidManifest(field));
            assert_eq!(
                error.to_string(),
                format!("checkpoint manifest has invalid {field}")
            );
        } else {
            result.expect("valid Map2 capture");
        }
    }
}

#[test]
fn runtime_requires_every_current_identity_field_without_mutation() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(18_102).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 1).expect("owner");
    let identity = model.policy_identity().expect("identity");
    let parameters = model.export_parameters().expect("parameters");
    let current = current_runtime_metadata();
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("save");
    let bytes = fs::read(directory.0.join("drysua.weights.safetensors")).expect("runtime");
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
    assert_eq!(metadata.metadata().as_ref(), Some(&current));
    for key in current.keys().map(String::as_str).chain(["unexpected"]) {
        for replacement in [None, Some("wrong")] {
            if key == "unexpected" && replacement.is_none() {
                continue;
            }
            let mut metadata = current.clone();
            metadata.remove(key);
            if let Some(value) = replacement {
                metadata.insert(key.to_owned(), value.to_owned());
            }
            let bytes = runtime_bytes(&parameters, metadata);
            let path = directory.0.join("drysua.weights.safetensors");
            fs::write(&path, &bytes).expect("fixture");
            let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
                .expect_err("exact schema required");
            assert_eq!(error, CheckpointError::SchemaMismatch, "{key}");
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
            assert_eq!(model.policy_identity().expect("identity"), identity);
            assert_eq!(model.export_parameters().expect("parameters"), parameters);
            assert_eq!(fs::read(path).expect("unchanged file"), bytes);
            TrainingArtifact::capture(
                &model,
                &trainer,
                run_metadata(),
                progress_metadata(0),
                crate::checkpoint::collection_fixture(&model),
            )
            .expect("optimizer binding preserved");
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn warm_start_reuses_named_tensors_reinitializes_the_rest_and_refuses_unrelated_files() {
    let directory = Directory::new();
    let source = PolicyModel::fresh(18_111).expect("source");
    let parameters = source.export_parameters().expect("parameters");
    let schema = source.parameter_schema().expect("schema");
    // An older model: another critic layout under older schema metadata.
    let data: Vec<u8> = parameters.iter().flat_map(|value| value.to_le_bytes()).collect();
    let legacy_value = vec![0; 257 * 4];
    let mut tensors = vec![
        ("value.weight".to_owned(), TensorView::new(Dtype::F32, vec![256, 1], &legacy_value[..1024])),
        ("value.bias".to_owned(), TensorView::new(Dtype::F32, vec![1], &legacy_value[1024..])),
    ];
    let mut offset = 0;
    for (name, shape) in &schema {
        let size = shape.iter().product::<usize>() * 4;
        if !name.starts_with("value.") {
            let view = TensorView::new(Dtype::F32, shape.clone(), &data[offset..offset + size]);
            tensors.push(((*name).to_owned(), view));
        }
        offset += size;
    }
    let tensors = tensors
        .into_iter()
        .map(|(name, view)| (name, view.expect("tensor")));
    let mut older = current_runtime_metadata();
    older.insert("model_schema_hash".to_owned(), "1".to_owned());
    let path = directory.0.join("drysua.weights.safetensors");
    fs::write(&path, serialize(tensors, Some(older)).expect("older model")).expect("fixture");

    let model =
        TrainingArtifact::initialize_from_weights(&directory.0, 5, crate::PolicyDevice::Cpu)
            .expect("warm start across schemas");

    let imported = model.export_parameters().expect("imported");
    let fresh = PolicyModel::fresh(5).expect("fresh").export_parameters().expect("fresh");
    let mut offset = 0;
    for (name, shape) in &schema {
        let range = offset..offset + shape.iter().product::<usize>();
        let expected = if name.starts_with("value.") { &fresh } else { &parameters };
        assert_eq!(imported[range.clone()], expected[range.clone()], "{name}");
        offset = range.end;
    }
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &directory.0),
        Err(CheckpointError::SchemaMismatch),
        "play and eval stay strict"
    );
    fs::write(&path, runtime_bytes(&parameters[1..], current_runtime_metadata()))
        .expect("flat fixture");
    assert_eq!(
        TrainingArtifact::initialize_from_weights(&directory.0, 5, crate::PolicyDevice::Cpu).err(),
        Some(CheckpointError::TensorContract("names")),
        "a file sharing no tensor is no warm start"
    );
}

#[test]
fn resume_rejects_each_linked_schema_version_and_hash_before_tensor_io() {
    let directory = Directory::new();
    let artifact = fresh_artifact();
    artifact.save(&directory.0).expect("save");
    let path = directory.0.join("checkpoint.meta");
    let original = fs::read(&path).expect("manifest");
    for entry in fs::read_dir(&directory.0).expect("directory") {
        let path = entry.expect("entry").path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            fs::remove_file(path).expect("remove tensors to prove validation ordering");
        }
    }
    for offset in [8, 12, 20, 24, 32, 36, 44, 48, 56, 60] {
        let mut bytes = original.clone();
        bytes[offset] ^= 1;
        fs::write(&path, &bytes).expect("alter schema");
        for result in [
            TrainingArtifact::load(&directory.0),
            TrainingArtifact::load_compatible(&directory.0, artifact.run()),
        ] {
            let error = result.expect_err("schema before missing tensors");
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
        assert_eq!(fs::read(&path).expect("unchanged manifest"), bytes);
    }
}

#[test]
fn strict_checkpoint_restores_adam_rng_and_identical_next_update() {
    let directory = Directory::new();
    let source = PolicyModel::fresh(18_001).expect("source");
    let mut trainer = PpoTrainer::new(&source, checkpoint_config(), 18_002).expect("trainer");
    advance_trainer(&source, &mut trainer);
    TrainingArtifact::capture(
        &source,
        &trainer,
        run_metadata(),
        progress_metadata(1),
        crate::checkpoint::collection_fixture(&source),
    )
    .expect("capture")
    .save(&directory.0)
    .expect("save");
    let loaded = TrainingArtifact::load_compatible(&directory.0, &run_metadata()).expect("load");
    let target = PolicyModel::fresh(18_003).expect("target");
    let mut state = loaded.restore(&target, &run_metadata()).expect("restore");
    assert_eq!(loaded.run(), &run_metadata());
    assert_eq!(state.progress(), &progress_metadata(1));
    assert_eq!(state.trainer().config(), trainer.config());
    assert_eq!(state.trainer().updates(), trainer.updates());
    assert_snapshot_equal(&source, &trainer, &target, state.trainer());
    let (source_batch, source_actions) = checkpoint_batch(&source, 18_005);
    let (target_batch, target_actions) = checkpoint_batch(&target, 18_005);
    trainer
        .train_update(&source, &source_batch)
        .expect("next source update");
    state
        .trainer_mut()
        .train_update(&target, &target_batch)
        .expect("next restored update");
    assert_eq!(source_actions, target_actions);
    assert_snapshot_equal(&source, &trainer, &target, state.trainer());
    assert!(fs::read_dir(&directory.0).expect("directory").all(|entry| {
        !entry
            .expect("entry")
            .file_name()
            .to_string_lossy()
            .contains(".tmp")
    }));
}

fn assert_snapshot_equal(
    source: &PolicyModel,
    trainer: &PpoTrainer,
    target: &PolicyModel,
    restored: &PpoTrainer,
) {
    let source_rng = trainer.rng_checkpoint();
    let target_rng = restored.rng_checkpoint();
    let source = trainer
        .checkpoint_snapshot(source)
        .expect("source snapshot");
    let target = restored
        .checkpoint_snapshot(target)
        .expect("restored snapshot");
    assert_eq!(source.parameters, target.parameters);
    assert_eq!(source.adam.moments(), target.adam.moments());
    assert_eq!(trainer.optimizer_step(), restored.optimizer_step());
    assert_eq!(trainer.rng_checkpoint(), source_rng);
    assert_eq!(restored.rng_checkpoint(), target_rng);
    assert_eq!(trainer.rng_checkpoint(), restored.rng_checkpoint());
}

#[test]
fn checkpoint_storage_rejects_corrupt_payload_and_truncated_manifest() {
    let directory = Directory::new();
    fresh_artifact().save(&directory.0).expect("save");
    let manifest = directory.0.join("checkpoint.meta");
    let files = fs::read_dir(&directory.0)
        .expect("directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("name")
        })
        .collect::<std::collections::BTreeSet<_>>();
    let generation = files
        .iter()
        .find(|name| name.starts_with("checkpoint.") && name.ends_with(".safetensors"))
        .expect("immutable generation");
    assert_eq!(
        files,
        [
            generation.as_str(),
            "checkpoint.meta",
            "drysua.weights.safetensors"
        ]
        .map(str::to_owned)
        .into(),
        "one commit leaves exactly its three files"
    );
    let generation = directory.0.join(generation);
    let mut bytes = fs::read(&generation).expect("tensor bytes");
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&generation, bytes).expect("corrupt generation");
    let error = TrainingArtifact::load(&directory.0).expect_err("hash mismatch");
    assert_eq!(error, CheckpointError::TensorHashMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint tensor SHA-256 does not match manifest"
    );
    fs::write(manifest, b"DRYSUA").expect("truncate manifest");
    let error = TrainingArtifact::load(&directory.0).expect_err("truncated manifest");
    assert_eq!(error, CheckpointError::ManifestTruncated);
    assert_eq!(error.to_string(), "checkpoint manifest is truncated");
}

#[test]
fn restore_rejects_optimizer_ownership_actor_and_scope_without_mutation() {
    let artifact = fresh_artifact();
    let target = PolicyModel::fresh(18_052).expect("target");
    let owner = PpoTrainer::new(&target, checkpoint_config(), 18_053).expect("owner");
    let before = target.export_parameters().expect("before");
    let error = artifact
        .restore(&target, &run_metadata())
        .err()
        .expect("owned optimizer");
    assert_eq!(
        error.to_string(),
        "checkpoint model restore failed: model already has a behavioral optimizer owner"
    );
    assert_eq!(target.export_parameters().expect("after"), before);
    drop(owner);
    let mut expected = run_metadata();
    expected.simulator_commit = "different".to_owned();
    let error = artifact
        .restore(&target, &expected)
        .err()
        .expect("wrong scope");
    assert_eq!(
        error,
        CheckpointError::InvalidManifest("compatibility scope")
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[cfg(unix)]
#[test]
fn runtime_loader_rejects_symlink_artifact() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(18_090).expect("model");
    let target = directory.0.join("outside.safetensors");
    fs::write(&target, b"outside").expect("outside file");
    std::os::unix::fs::symlink(&target, directory.0.join("drysua.weights.safetensors"))
        .expect("symlink");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("symlink"),
        CheckpointError::InvalidManifest("artifact file type")
    );
    assert_eq!(fs::read(target).expect("unchanged target"), b"outside");
}

fn fresh_artifact() -> TrainingArtifact {
    let model = PolicyModel::fresh(18_000).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 1).expect("trainer");
    TrainingArtifact::capture(
        &model,
        &trainer,
        run_metadata(),
        progress_metadata(0),
        crate::checkpoint::collection_fixture(&model),
    )
    .expect("capture")
}

fn current_runtime_metadata() -> std::collections::HashMap<String, String> {
    [
        ("action_schema_hash", crate::ACTION_SCHEMA_HASH.to_string()),
        (
            "feature_schema_hash",
            crate::FEATURE_SCHEMA_HASH.to_string(),
        ),
        ("model_schema_hash", crate::MODEL_SCHEMA_HASH.to_string()),
        ("ppo_schema_version", crate::PPO_SCHEMA_VERSION.to_string()),
        ("ppo_schema_hash", crate::PPO_SCHEMA_HASH.to_string()),
        (
            "ppo_rules_audit_version",
            crate::PPO_RULES_AUDIT_VERSION.to_string(),
        ),
        (
            "map2_reward_version",
            crate::MAP2_REWARD_VERSION.to_string(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

fn checkpoint_config() -> PpoConfig {
    PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        samples_per_update: 4,
        epochs: 1,
        minibatch: 4,
        ..PpoConfig::default()
    }
}

fn run_metadata() -> CheckpointRun {
    CheckpointRun {
        git_commit: "28e196f".to_owned(),
        simulator_commit: "b575129".to_owned(),
        enabled_features: crate::compiled_features(),
        command_line: "drysua train --device cpu".to_owned(),
        run_seed: 18_000,
        map: MapId(2),
        hero: crate::SHADOW_FIEND,
        device: CheckpointDevice::Cpu,
        batch_size: 4,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn progress_metadata(global_update: u64) -> CheckpointProgress {
    CheckpointProgress {
        adaptive_environment: None,
        global_update,
        policy_version: global_update,
        scheduler_step: 3,
        curriculum_stage: 2,
        rollout_samples: global_update * 4,
        best_evaluation: Some(0.75),
        rng_states: vec![
            RngCheckpoint::new("actor", 11, 12).expect("actor RNG"),
            RngCheckpoint::new("learner", 13, 14).expect("learner RNG"),
        ],
        league_references: vec![101, 102],
    }
}

fn advance_trainer(model: &PolicyModel, trainer: &mut PpoTrainer) {
    let (batch, _) = checkpoint_batch(model, 18_004);
    trainer.train_update(model, &batch).expect("update");
}

fn checkpoint_batch(
    model: &PolicyModel,
    seed: u64,
) -> (crate::PpoBatch, Vec<crate::StructuredAction>) {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let mut rng = PpoRng::new(seed);
    let mut rollout = PpoRollout::new(4).expect("rollout");
    let mut actions = Vec::with_capacity(4);
    for stream in 0..4 {
        let choice = model.sample(&frame, &space, &mut rng).expect("choice");
        actions.push(choice.action());
        rollout
            .push(
                choice
                    .finish(
                        0,
                        PpoOutcome {
                            stream,
                            decision: 0,
                            ticks: 3,
                            next_value: 0.0,
                            reward: stream as f32,
                            terminal: true,
                        },
                    )
                    .expect("transition"),
            )
            .expect("push");
    }
    (rollout.finish(checkpoint_config()).expect("batch"), actions)
}
