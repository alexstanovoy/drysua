use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::map2_checkpoint::progress;
use super::map2_model_initialization::assert_fresh_state;
use crate::{CheckpointRun, PolicyDevice, PolicyModel, TrainingArtifact};

#[test]
#[ignore = "INITIALIZATION ONLY: explicit DRYSUA_SELECTED_M16_SOURCE, new DRYSUA_NAVIGATION_INITIALIZATION_OUTPUT, DRYSUA_NAVIGATION_INITIALIZATION_SEED and git/simulator provenance; no training or gameplay"]
fn initialize_pinned_m16_navigation_artifact_only() {
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M16_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("existing source");
    let output =
        crate::tests::support::new_output("DRYSUA_NAVIGATION_INITIALIZATION_OUTPUT", &source);
    let seed = std::env::var("DRYSUA_NAVIGATION_INITIALIZATION_SEED")
        .expect("explicit fresh seed")
        .parse::<u64>()
        .expect("u64 seed");
    let (model, provenance) = TrainingArtifact::initialize_selected_m16_for_map2_navigation(
        &source,
        seed,
        PolicyDevice::Cpu,
    )
    .expect("exact selected M16 source, initialization only");
    let bytes =
        fs::read(source.join("drysua.weights.safetensors")).expect("immutable selected source");
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&bytes)),
        provenance.source_sha256()
    );
    let parameters = model.export_parameters().expect("copied parameters");
    assert_source_bits(&model, &bytes, &parameters);
    let rejected = PolicyModel::fresh(seed).expect("runtime rejection target");
    let identity = rejected.policy_identity().expect("identity");
    assert_eq!(
        TrainingArtifact::load_runtime_weights(&rejected, &source),
        Err(crate::CheckpointError::SchemaMismatch)
    );
    assert_eq!(
        rejected.policy_identity().expect("unchanged identity"),
        identity
    );
    let config = crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        ..crate::PpoConfig::default()
    };
    let trainer =
        crate::PpoTrainer::new(&model, config, seed).expect("fresh optimizer and shuffle RNG");
    assert_fresh_state(&model, &trainer);
    let run = crate::tests::support::initialization_run(
        seed,
        config.minibatch,
        format!(
            "tests::navigation_initialization_utility::initialize_pinned_m16_navigation_artifact_only --exact --ignored; {}",
            provenance.description()
        ),
        &crate::tests::support::INITIALIZATION_PROVENANCE,
    );
    let artifact = TrainingArtifact::capture(&model, &trainer, run.clone(), progress())
        .expect("fresh progress");
    fs::create_dir(&output).expect("new output only");
    TrainingArtifact::save_runtime_weights(&model, &output).expect("new current runtime");
    assert_eq!(
        artifact.save(&output).expect("new checkpoint"),
        crate::CheckpointSaveOutcome::Committed
    );
    verify_output(&output, &run, &parameters);
    assert_eq!(
        fs::read(source.join("drysua.weights.safetensors")).expect("source unchanged"),
        bytes
    );
    write_provenance(&output, &source, &run, &provenance.description());
    eprintln!("{} output={}", provenance.description(), output.display());
}

fn assert_source_bits(model: &PolicyModel, bytes: &[u8], parameters: &[f32]) {
    let tensors = safetensors::SafeTensors::deserialize(bytes).expect("pinned tensors");
    let tensor = tensors
        .tensor("model.parameters")
        .expect("source parameters");
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    assert_eq!(tensor.shape(), [1_696_436]);
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    assert!(remainder.is_empty());
    let source: Vec<_> = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    super::fountain_wait_initialization::assert_wait_padding(model, &source, parameters);
}

fn verify_output(output: &Path, run: &CheckpointRun, parameters: &[f32]) {
    crate::tests::support::verify_initialized_output(output, run, parameters);
}

fn write_provenance(output: &Path, source: &Path, run: &CheckpointRun, provenance: &str) {
    let bytes = fs::read(output.join("drysua.weights.safetensors")).expect("new runtime");
    let digest = crate::tests::support::sha256_hex(&bytes);
    assert_eq!(digest.len(), 64);
    let text = format!(
        "{provenance}\nsource={}\noutput_sha256={digest}\ngit_commit={}\nsimulator_commit={}\nseed={}\nenabled_features={}\ntraining_updates=0\ngameplay_runs=0\nsource_unchanged=true\nparameters=1700020\nglobal_features=92\nunit_features=84\nold_parameter_bits_preserved=true\nnew_zero_trunk_rows=85..92\nruntime_and_checkpoint_reload_exact=true\n",
        source.display(),
        run.git_commit,
        run.simulator_commit,
        run.run_seed,
        run.enabled_features
    );
    assert!(text.len() < 16 * 1024);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("INITIALIZATION.txt"))
        .expect("new provenance record");
    file.write_all(text.as_bytes()).expect("provenance");
    file.sync_all().expect("durable provenance");
    fs::File::open(output)
        .expect("output directory")
        .sync_all()
        .expect("durable directory");
}
