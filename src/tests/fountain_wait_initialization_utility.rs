use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::fountain_wait_initialization::{M17_PARAMETERS, assert_wait_padding};
use super::map2_checkpoint::progress;
use super::map2_model_initialization::assert_fresh_state;
use crate::{CheckpointRun, PolicyDevice, TrainingArtifact};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "INITIALIZATION ONLY under resource guard: explicit DRYSUA_SELECTED_M17_SOURCE, absent DRYSUA_WAIT_INITIALIZATION_OUTPUT, DRYSUA_WAIT_INITIALIZATION_SEED and frozen git/simulator provenance. No training or gameplay."]
fn initialize_pinned_m17_wait_artifact_only() {
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M17_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("source directory");
    let output = crate::tests::support::new_output("DRYSUA_WAIT_INITIALIZATION_OUTPUT", &source);
    let seed = std::env::var("DRYSUA_WAIT_INITIALIZATION_SEED")
        .expect("explicit seed")
        .parse::<u64>()
        .expect("u64 seed");
    let (model, provenance) =
        TrainingArtifact::initialize_selected_m17_for_map2_wait(&source, seed, PolicyDevice::Cpu)
            .expect("exact selected source only");
    let source_bytes = fs::read(source.join("drysua.weights.safetensors")).expect("source");
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&source_bytes)),
        provenance.source_sha256()
    );
    let source_values = old_parameters(&source_bytes);
    let target = model.export_parameters().expect("new values");
    assert_wait_padding(&model, &source_values, &target);
    let config = crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        ..crate::PpoConfig::default()
    };
    let trainer = crate::PpoTrainer::new(&model, config, seed).expect("fresh trainer");
    assert_fresh_state(&model, &trainer);
    let run = crate::tests::support::initialization_run(
        seed,
        config.minibatch,
        format!(
            "tests::fountain_wait_initialization_utility::initialize_pinned_m17_wait_artifact_only --exact --ignored; {}",
            provenance.description()
        ),
        &crate::tests::support::INITIALIZATION_PROVENANCE,
    );
    let artifact = TrainingArtifact::capture(&model, &trainer, run.clone(), progress())
        .expect("fresh artifact");
    fs::create_dir(&output).expect("never overwrite output");
    TrainingArtifact::save_runtime_weights(&model, &output).expect("new runtime");
    assert_eq!(
        artifact.save(&output).expect("checkpoint"),
        crate::CheckpointSaveOutcome::Committed
    );
    verify_output(&output, &run, &target);
    assert_eq!(
        source_bytes,
        fs::read(source.join("drysua.weights.safetensors")).expect("source unchanged")
    );
    let mut record = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("INITIALIZATION.txt"))
        .expect("new audit record");
    writeln!(record, "{}\nsource={}\nparameters={}\nglobal92_unit84\ntrunk.0.weight=2589x512_to2596x512_zero_rows85..92\nsource_unchanged=true\ntraining_updates=0\ngameplay_runs=0\ngit_commit={}\nsimulator_commit={}",
        provenance.description(), source.display(), target.len(), run.git_commit, run.simulator_commit).expect("audit");
    record.sync_all().expect("durable record");
    fs::File::open(&output)
        .expect("directory")
        .sync_all()
        .expect("durable directory");
    eprintln!("{} output={}", provenance.description(), output.display());
}

fn old_parameters(bytes: &[u8]) -> Vec<f32> {
    let tensors = safetensors::SafeTensors::deserialize(bytes).expect("source tensors");
    let tensor = tensors.tensor("model.parameters").expect("parameters");
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    assert_eq!(tensor.shape(), [M17_PARAMETERS]);
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    assert!(remainder.is_empty());
    chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn verify_output(output: &Path, run: &CheckpointRun, target: &[f32]) {
    crate::tests::support::verify_initialized_output(output, run, target);
}
