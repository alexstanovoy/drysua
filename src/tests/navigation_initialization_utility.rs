use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::map2_checkpoint::progress;
use super::map2_model_initialization::{assert_bits, assert_fresh_state};
use crate::{CheckpointRun, PolicyDevice, PolicyModel, TrainingArtifact};

#[test]
#[ignore = "INITIALIZATION ONLY: explicit DRYSUA_SELECTED_M16_SOURCE, new DRYSUA_NAVIGATION_INITIALIZATION_OUTPUT, DRYSUA_NAVIGATION_INITIALIZATION_SEED and git/simulator provenance; no training or gameplay"]
fn initialize_pinned_m16_navigation_artifact_only() {
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M16_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("existing source");
    let output = new_output(&source);
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
    assert_source_bits(&bytes, &parameters);
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
    let run = initialization_run(config, seed, &provenance.description());
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

fn new_output(source: &Path) -> PathBuf {
    let requested = PathBuf::from(
        std::env::var_os("DRYSUA_NAVIGATION_INITIALIZATION_OUTPUT").expect("explicit new output"),
    );
    let parent = requested
        .parent()
        .expect("parent")
        .canonicalize()
        .expect("existing parent");
    assert!(
        !parent.starts_with(source),
        "no writes inside the historical source"
    );
    let output = parent.join(requested.file_name().expect("directory name"));
    assert!(
        fs::symlink_metadata(&output)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
        "never overwrite or follow an existing output"
    );
    output
}

fn assert_source_bits(bytes: &[u8], parameters: &[f32]) {
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
    assert_bits(parameters, &source);
}

fn initialization_run(config: crate::PpoConfig, seed: u64, provenance: &str) -> CheckpointRun {
    CheckpointRun {
        git_commit: std::env::var("DRYSUA_INITIALIZATION_GIT_COMMIT")
            .expect("actual source revision"),
        simulator_commit: std::env::var("DRYSUA_INITIALIZATION_SIMULATOR_COMMIT")
            .expect("actual simulator revision"),
        enabled_features: crate::compiled_features(),
        command_line: format!(
            "tests::navigation_initialization_utility::initialize_pinned_m16_navigation_artifact_only --exact --ignored; {provenance}"
        ),
        run_seed: seed,
        map: bota_proto::MapId(2),
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config.minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn verify_output(output: &Path, run: &CheckpointRun, parameters: &[f32]) {
    let runtime = PolicyModel::fresh(1).expect("new runtime target");
    TrainingArtifact::load_runtime_weights(&runtime, output)
        .expect("strict current runtime reload");
    assert_bits(
        &runtime.export_parameters().expect("runtime bits"),
        parameters,
    );
    let checkpoint =
        TrainingArtifact::load_compatible(output, run).expect("strict current checkpoint reload");
    assert_eq!(checkpoint.progress(), &progress());
    let restored = PolicyModel::fresh(2).expect("new checkpoint target");
    let state = checkpoint
        .restore(&restored, run)
        .expect("restore new checkpoint only");
    assert_fresh_state(&restored, state.trainer());
    assert_bits(
        &restored.export_parameters().expect("checkpoint bits"),
        parameters,
    );
}

fn write_provenance(output: &Path, source: &Path, run: &CheckpointRun, provenance: &str) {
    let bytes = fs::read(output.join("drysua.weights.safetensors")).expect("new runtime");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest.len(), 64);
    let text = format!(
        "{provenance}\nsource={}\noutput_sha256={digest}\ngit_commit={}\nsimulator_commit={}\nseed={}\nenabled_features={}\ntraining_updates=0\ngameplay_runs=0\nsource_unchanged=true\nparameters=1696436\nglobal_features=85\nunit_features=84\nall_62_named_tensors_bit_identical=true\nruntime_and_checkpoint_reload_exact=true\n",
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
