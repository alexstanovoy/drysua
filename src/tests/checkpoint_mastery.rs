use super::*;
use crate::{MasteryConfig, MasteryProgress, MasteryStage, TrainingGameOutcome as Outcome};

#[test]
fn mastery_checkpoint_roundtrips_wrapped_window_adam_rng_and_identical_next_update() {
    let directory = test_directory("mastery-ring");
    let source = PolicyModel::fresh(9140100).expect("source");
    let mut trainer = PpoTrainer::new(&source, checkpoint_config(), 9140101).expect("trainer");
    let config = MasteryConfig::new(3, 100, &[]).expect("mastery config");
    let mut mastery = MasteryProgress::default();
    for results in [[Outcome::Win, Outcome::Loss], [Outcome::Win, Outcome::Win]] {
        advance_trainer(&source, &mut trainer);
        mastery.record_batch(config, &results).expect("batch");
    }
    assert_eq!(
        mastery.recent().iter().copied().collect::<Vec<_>>(),
        [false, true, true]
    );
    let mut run = run_metadata();
    run.mastery_config = Some(config);
    let mut progress = progress_metadata(2);
    progress.rollout_samples = 8;
    progress.scheduler_step = 2;
    progress.curriculum_stage = 0;
    progress.mastery = Some(mastery.clone());
    let artifact = TrainingArtifact::capture(&source, &trainer, run.clone(), progress.clone())
        .expect("capture");
    artifact.save(&directory).expect("save");
    let loaded = TrainingArtifact::load_compatible(&directory, &run).expect("load");
    let target = PolicyModel::fresh(9140102).expect("target");
    let mut restored = loaded.restore(&target, &run).expect("restore");
    assert_eq!(restored.progress(), &progress);
    assert_eq!(
        restored.trainer().rng_checkpoint(),
        trainer.rng_checkpoint()
    );
    assert_eq!(
        restored.trainer().optimizer_step(),
        trainer.optimizer_step()
    );
    assert!(trainer.optimizer_step() > 0);
    let before = trainer
        .checkpoint_snapshot(&source)
        .expect("before snapshot");
    let after = restored
        .trainer()
        .checkpoint_snapshot(&target)
        .expect("restored snapshot");
    assert_eq!(before.adam.moments(), after.adam.moments());
    let mut resumed_window = restored.progress().mastery.clone().expect("window");
    advance_trainer(&source, &mut trainer);
    advance_trainer(&target, restored.trainer_mut());
    mastery
        .record_batch(config, &[Outcome::Win, Outcome::Win])
        .expect("next results");
    resumed_window
        .record_batch(config, &[Outcome::Win, Outcome::Win])
        .expect("resumed results");
    assert_eq!(mastery, resumed_window);
    assert_eq!(mastery.stage(), MasteryStage::Teacher);
    assert_eq!(mastery.games(), 0);
    assert_promoted_checkpoint(&source, &trainer, &run, &mastery);
    assert_eq!(
        source.export_parameters().expect("source"),
        target.export_parameters().expect("target")
    );
    assert_eq!(
        trainer.optimizer_step(),
        restored.trainer().optimizer_step()
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

fn assert_promoted_checkpoint(
    model: &PolicyModel,
    trainer: &PpoTrainer,
    run: &CheckpointRun,
    mastery: &MasteryProgress,
) {
    let directory = test_directory("mastery-after-promotion");
    let mut progress = progress_metadata(trainer.updates());
    progress.mastery = Some(mastery.clone());
    progress.rollout_samples = trainer.updates() * 4;
    let artifact = TrainingArtifact::capture(model, trainer, run.clone(), progress.clone())
        .expect("promoted capture");
    artifact.save(&directory).expect("promoted checkpoint");
    let loaded = TrainingArtifact::load_compatible(&directory, run).expect("promoted load");
    let target = PolicyModel::fresh(9140104).expect("target");
    let restored = loaded.restore(&target, run).expect("promoted restore");
    assert_eq!(restored.progress(), &progress);
    assert_eq!(
        restored.trainer().rng_checkpoint(),
        trainer.rng_checkpoint()
    );
    assert_eq!(
        restored.trainer().optimizer_step(),
        trainer.optimizer_step()
    );
    assert_eq!(
        target.export_parameters().expect("target"),
        model.export_parameters().expect("source")
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn mastery_checkpoint_config_mismatch_precedes_tensor_io_and_parameter_mutation() {
    let directory = test_directory("mastery-mismatch");
    let source = PolicyModel::fresh(9140110).expect("source");
    let trainer = PpoTrainer::new(&source, checkpoint_config(), 9140111).expect("trainer");
    let config = MasteryConfig::default();
    let mut run = run_metadata();
    run.mastery_config = Some(config);
    let mut progress = progress_metadata(0);
    progress.mastery = Some(MasteryProgress::default());
    let artifact =
        TrainingArtifact::capture(&source, &trainer, run.clone(), progress).expect("capture");
    artifact.save(&directory).expect("save");
    let target = PolicyModel::fresh(9140112).expect("target");
    let before = target.export_parameters().expect("before");
    for changed in [
        None,
        Some(MasteryConfig::new(51, 80, &[]).expect("window")),
        Some(MasteryConfig::new(50, 81, &[]).expect("percent")),
    ] {
        let mut expected = run.clone();
        expected.mastery_config = changed;
        assert_eq!(
            TrainingArtifact::load_compatible(&directory, &expected).expect_err("mismatch"),
            CheckpointError::InvalidManifest("compatibility scope")
        );
        assert_eq!(
            artifact.restore(&target, &expected).err(),
            Some(CheckpointError::InvalidManifest("compatibility scope"))
        );
        assert_eq!(target.export_parameters().expect("unchanged"), before);
    }
    let _owner = target
        .claim_optimizer(checkpoint_config().adam())
        .expect("failed restore did not claim optimizer");
    let mut changed = run;
    changed.mastery_config = None;
    for entry in fs::read_dir(&directory).expect("files") {
        let path = entry.expect("entry").path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            fs::write(path, b"broken tensor").expect("corrupt tensor fixture");
        }
    }
    assert_eq!(
        TrainingArtifact::load_compatible(&directory, &changed).expect_err("scope before tensors"),
        CheckpointError::InvalidManifest("compatibility scope")
    );
    fs::remove_dir_all(directory).expect("cleanup");
}
