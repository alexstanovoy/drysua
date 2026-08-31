use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::{MapId, Team};
use safetensors::tensor::{Dtype, TensorView, serialize};

use super::feature::{encode, tracker_with_view, world_view};
use crate::{
    ActionSpace, CheckpointDevice, CheckpointError, CheckpointProgress, CheckpointRun,
    LocalPolicyState, PolicyDevice, PolicyModel, PpoConfig, PpoOutcome, PpoRng, PpoRollout,
    PpoTrainer, RngCheckpoint, TrainingArtifact,
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);

#[test]
fn checkpoint_schema_hash_is_stable() {
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 1);
    assert_eq!(crate::CHECKPOINT_SCHEMA_HASH, 2_133_011_134_179_236_231);
}

#[test]
fn strict_training_artifact_roundtrips_model_optimizer_and_run_state() {
    let directory = test_directory("roundtrip");
    let source = PolicyModel::fresh(18_001).expect("source model");
    let mut trainer = PpoTrainer::new(&source, checkpoint_config(), 18_002).expect("trainer");
    advance_trainer(&source, &mut trainer);
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(1))
            .expect("capture");

    artifact.save(&directory).expect("atomic save");
    let loaded =
        TrainingArtifact::load_compatible(&directory, &run_metadata()).expect("strict load");
    let restored = PolicyModel::fresh(18_003).expect("restored model");
    let mut restored_state = loaded
        .restore(&restored, &run_metadata())
        .expect("restore state");
    let restored_trainer = restored_state.trainer();

    assert_eq!(loaded.run(), &run_metadata());
    assert_eq!(loaded.progress(), &progress_metadata(1));
    assert_eq!(restored.device(), PolicyDevice::Cpu);
    assert_eq!(
        restored.export_parameters().expect("restored parameters"),
        source.export_parameters().expect("source parameters")
    );
    assert_eq!(restored_trainer.config(), trainer.config());
    assert_eq!(restored_trainer.optimizer_step(), trainer.optimizer_step());
    assert_eq!(restored_trainer.updates(), trainer.updates());
    assert_eq!(restored_trainer.rng_checkpoint(), trainer.rng_checkpoint());
    assert_eq!(
        restored_state
            .pipeline(4, 1, &restored)
            .expect("restored pipeline")
            .version()
            .expect("pipeline version")
            .get(),
        1
    );
    let (source_batch, source_actions) = checkpoint_batch(&source, 18_005);
    let (restored_batch, restored_actions) = checkpoint_batch(&restored, 18_005);
    trainer
        .train_update(&source, &source_batch)
        .expect("continued source update");
    restored_state
        .trainer_mut()
        .train_update(&restored, &restored_batch)
        .expect("continued restored update");
    assert_eq!(source_actions, restored_actions);
    assert_eq!(
        source.export_parameters().expect("continued source"),
        restored.export_parameters().expect("continued restored")
    );
    assert_eq!(
        trainer.rng_checkpoint(),
        restored_state.trainer().rng_checkpoint()
    );
    assert_eq!(
        trainer.optimizer_step(),
        restored_state.trainer().optimizer_step()
    );
    assert!(!directory.join("checkpoint.safetensors.tmp").exists());
    assert!(!directory.join("checkpoint.meta.tmp").exists());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn tensor_corruption_is_rejected_by_sha256_before_deserialization() {
    let directory = test_directory("tensor-corruption");
    let model = PolicyModel::fresh(18_010).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_011).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    let tensor_path = fs::read_dir(&directory)
        .expect("artifact directory")
        .map(|entry| entry.expect("entry").path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("checkpoint.")
                        && name.ends_with(".safetensors")
                        && name != "checkpoint.safetensors"
                })
        })
        .expect("immutable tensor generation");
    let mut bytes = fs::read(&tensor_path).expect("tensor bytes");
    let index = bytes.len() - 1;
    bytes[index] ^= 0x01;
    fs::write(&tensor_path, bytes).expect("corrupt tensor");

    let error = TrainingArtifact::load(&directory).expect_err("hash mismatch");

    assert_eq!(error, CheckpointError::TensorHashMismatch);
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn truncated_manifest_is_rejected_with_exact_error() {
    let directory = test_directory("manifest-truncated");
    let model = PolicyModel::fresh(18_020).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_021).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("capture")
        .save(&directory)
        .expect("save");
    fs::write(directory.join("checkpoint.meta"), b"DRYSUA").expect("truncate manifest");

    let error = TrainingArtifact::load(&directory).expect_err("truncated manifest");

    assert_eq!(error.to_string(), "checkpoint manifest is truncated");
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn runtime_weights_reject_unknown_or_missing_tensor_contract() {
    let directory = test_directory("runtime");
    let model = PolicyModel::fresh(18_030).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("runtime save");
    let restored = PolicyModel::fresh(18_031).expect("restored");

    TrainingArtifact::load_runtime_weights(&restored, &directory).expect("runtime load");

    assert_eq!(
        restored.export_parameters().expect("restored parameters"),
        model.export_parameters().expect("source parameters")
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn runtime_weights_reject_unknown_tensor_name_before_model_mutation() {
    let directory = test_directory("runtime-name");
    let model = PolicyModel::fresh(18_040).expect("model");
    let before = model.export_parameters().expect("before");
    let data = 0.0f32.to_le_bytes();
    let view = TensorView::new(Dtype::F32, vec![1], &data).expect("view");
    let metadata = std::collections::HashMap::from([
        (
            "action_schema_hash".to_owned(),
            crate::ACTION_SCHEMA_HASH.to_string(),
        ),
        (
            "feature_schema_hash".to_owned(),
            crate::FEATURE_SCHEMA_HASH.to_string(),
        ),
        (
            "model_schema_hash".to_owned(),
            crate::MODEL_SCHEMA_HASH.to_string(),
        ),
    ]);
    let bytes = serialize([("unknown", view)], Some(metadata)).expect("malformed runtime file");
    fs::write(directory.join("drysua.weights.safetensors"), bytes).expect("write malformed");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory)
        .expect_err("unknown tensor name");

    assert_eq!(error, CheckpointError::TensorContract("names"));
    assert_eq!(model.export_parameters().expect("after"), before);
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn restore_rejects_owned_optimizer_without_parameter_mutation() {
    let source = PolicyModel::fresh(18_050).expect("source");
    let source_trainer =
        PpoTrainer::new(&source, checkpoint_config(), 18_051).expect("source trainer");
    let artifact = TrainingArtifact::capture(
        &source,
        &source_trainer,
        run_metadata(),
        progress_metadata(0),
    )
    .expect("artifact");
    let target = PolicyModel::fresh(18_052).expect("target");
    let _owner = PpoTrainer::new(&target, checkpoint_config(), 18_053).expect("existing owner");
    let before = target.export_parameters().expect("before");

    let Err(error) = artifact.restore(&target, &run_metadata()) else {
        panic!("owned optimizer was replaced");
    };

    assert_eq!(
        error.to_string(),
        "checkpoint model restore failed: model already has a behavioral optimizer owner"
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn restore_rejects_immutable_actor_model_without_parameter_mutation() {
    let source = PolicyModel::fresh(18_060).expect("source");
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18_061).expect("trainer");
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
            .expect("artifact");
    let learner = PolicyModel::fresh(18_062).expect("learner");
    let mut pipeline = crate::ActorLearnerPipeline::new(4, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let before = lease.policy().export_parameters().expect("before");

    let Err(error) = artifact.restore(lease.policy(), &run_metadata()) else {
        panic!("actor model accepted a training checkpoint");
    };

    assert_eq!(
        error.to_string(),
        "checkpoint model restore failed: immutable actor model cannot own an optimizer"
    );
    assert_eq!(lease.policy().export_parameters().expect("after"), before);
}

#[test]
fn restore_rejects_different_compatibility_scope_before_mutation() {
    let source = PolicyModel::fresh(18_070).expect("source");
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 18_071).expect("trainer");
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run_metadata(), progress_metadata(0))
            .expect("artifact");
    let target = PolicyModel::fresh(18_072).expect("target");
    let before = target.export_parameters().expect("before");
    let mut expected = run_metadata();
    expected.simulator_commit = "different".to_owned();

    let Err(error) = artifact.restore(&target, &expected) else {
        panic!("incompatible simulator commit was accepted");
    };

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("compatibility scope")
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn manifest_backup_recovers_checkpoint_after_interrupted_replacement() {
    let directory = test_directory("manifest-recovery");
    let model = PolicyModel::fresh(18_080).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_081).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("artifact")
        .save(&directory)
        .expect("save");
    fs::rename(
        directory.join("checkpoint.meta"),
        directory.join("checkpoint.meta.previous"),
    )
    .expect("simulate crash after backup rename");

    let recovered = TrainingArtifact::load(&directory).expect("recover previous manifest");

    assert_eq!(recovered.run(), &run_metadata());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn documented_two_file_artifact_loads_without_immutable_generation_copy() {
    let directory = test_directory("two-file-copy");
    let model = PolicyModel::fresh(18_085).expect("model");
    let trainer = PpoTrainer::new(&model, checkpoint_config(), 18_086).expect("trainer");
    TrainingArtifact::capture(&model, &trainer, run_metadata(), progress_metadata(0))
        .expect("artifact")
        .save(&directory)
        .expect("save");
    for entry in fs::read_dir(&directory).expect("directory") {
        let path = entry.expect("entry").path();
        let is_generation = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("checkpoint.")
                    && name.ends_with(".safetensors")
                    && name != "checkpoint.safetensors"
            });
        if is_generation {
            fs::remove_file(path).expect("remove generation");
        }
    }

    let loaded = TrainingArtifact::load(&directory).expect("canonical tensor fallback");

    assert_eq!(loaded.run(), &run_metadata());
    fs::remove_dir_all(directory).expect("remove test directory");
}

#[cfg(unix)]
#[test]
fn runtime_loader_rejects_symlink_artifact() {
    use std::os::unix::fs::symlink;

    let directory = test_directory("runtime-symlink");
    let model = PolicyModel::fresh(18_090).expect("model");
    let target = directory.join("outside.safetensors");
    fs::write(&target, b"outside").expect("outside file");
    symlink(&target, directory.join("drysua.weights.safetensors")).expect("artifact symlink");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("symlink artifact");

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("artifact file type")
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}

fn checkpoint_config() -> PpoConfig {
    PpoConfig {
        rollout_decisions: 2,
        environments: 2,
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
        map: MapId(1),
        hero: crate::SHADOW_FIEND,
        device: CheckpointDevice::Cpu,
        batch_size: 4,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn progress_metadata(global_update: u64) -> CheckpointProgress {
    CheckpointProgress {
        global_update,
        policy_version: global_update,
        scheduler_step: 3,
        curriculum_stage: 2,
        rollout_samples: 1_024,
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
    let policy = model.policy_identity().expect("policy");
    let mut rollout = PpoRollout::new(4, policy).expect("rollout");
    let mut actions = Vec::with_capacity(4);
    for stream in 0..4 {
        let choice = model.sample(&frame, &space, &mut rng).expect("choice");
        actions.push(choice.action());
        rollout
            .push(
                choice
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: stream as f32,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    let batch = rollout.finish(checkpoint_config()).expect("batch");
    (batch, actions)
}

fn test_directory(name: &str) -> PathBuf {
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-checkpoint-{name}-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("remove stale test directory");
    }
    fs::create_dir(&directory).expect("create test directory");
    directory
}
