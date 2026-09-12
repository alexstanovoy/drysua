use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::map2_checkpoint::{Directory, m14_metadata, progress, runtime_bytes};
use super::map2_model_initialization::{assert_bits, assert_fresh_state, assert_padding};
use crate::{CheckpointError, PolicyDevice, PolicyModel, TrainingArtifact};

const INITIALIZATION_SEED: u64 = 10_091_500;

#[test]
#[ignore = "INITIALIZATION ONLY: explicit DRYSUA_SELECTED_M14_SOURCE, new DRYSUA_MAP2_INITIALIZATION_OUTPUT, DRYSUA_INITIALIZATION_GIT_COMMIT and DRYSUA_INITIALIZATION_SIMULATOR_COMMIT; no training or gameplay"]
#[allow(
    clippy::assertions_on_constants,
    reason = "The ignored utility must reject explicit debug execution without breaking debug test compilation"
)]
fn map2_initialize_pinned_m14_local_artifact_only() {
    assert!(
        !cfg!(debug_assertions),
        "run this explicit utility in release only"
    );
    let source =
        PathBuf::from(std::env::var_os("DRYSUA_SELECTED_M14_SOURCE").expect("explicit source"))
            .canonicalize()
            .expect("existing source");
    let output = new_output(&source);
    let bytes = fs::read(source.join("drysua.weights.safetensors")).expect("source bytes");
    let source_sha256: [u8; 32] = Sha256::digest(&bytes).into();
    assert_legacy_rejected(&source);

    let (model, provenance) = TrainingArtifact::initialize_selected_m14_for_map2(
        &source,
        INITIALIZATION_SEED,
        PolicyDevice::Cpu,
    )
    .expect("pinned initialization only");

    assert_eq!(provenance.source_sha256, source_sha256);
    let parameters = model.export_parameters().expect("initialized parameters");
    let old = source_parameters(&bytes);
    assert_padding(&model, &old, &parameters);
    assert_paired_digest_rejected(&old, provenance.source_ppo_schema_version);
    let config = crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        ..crate::PpoConfig::default()
    };
    let trainer =
        crate::PpoTrainer::new(&model, config, INITIALIZATION_SEED).expect("fresh trainer");
    assert_fresh_state(&model, &trainer);
    let run = initialization_run(config, &provenance.description());
    let artifact = TrainingArtifact::capture(&model, &trainer, run.clone(), progress())
        .expect("fresh capture");
    fs::create_dir(&output).expect("new output, never replace prior artifacts");
    TrainingArtifact::save_runtime_weights(&model, &output).expect("current runtime");
    assert_eq!(
        artifact.save(&output).expect("current checkpoint"),
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
    let output = PathBuf::from(
        std::env::var_os("DRYSUA_MAP2_INITIALIZATION_OUTPUT").expect("explicit output"),
    );
    assert!(!output.exists(), "never overwrite an output");
    let parent = output
        .parent()
        .expect("parent")
        .canonicalize()
        .expect("existing parent");
    assert!(
        !parent.starts_with(source),
        "do not create anything inside the historical source"
    );
    let output = parent.join(output.file_name().expect("new directory name"));
    assert!(
        fs::symlink_metadata(&output)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    );
    output
}

fn source_parameters(bytes: &[u8]) -> Vec<f32> {
    let tensors = safetensors::SafeTensors::deserialize(bytes).expect("source tensors");
    let tensor = tensors.tensor("model.parameters").expect("parameters");
    assert_eq!(tensor.shape(), [1_689_076]);
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    assert!(remainder.is_empty());
    chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn assert_legacy_rejected(source: &Path) {
    let model = PolicyModel::fresh(1).expect("runtime target");
    let identity = model.policy_identity().expect("identity");
    let error = TrainingArtifact::load_runtime_weights(&model, source).expect_err("no old runtime");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
    let error = TrainingArtifact::load(source).expect_err("no old resume");
    assert_eq!(error, CheckpointError::SchemaMismatch);
}

fn assert_paired_digest_rejected(parameters: &[f32], version: u32) {
    let directory = Directory::new();
    let other = if version == 27 { 26 } else { 27 };
    fs::write(
        directory.0.join("drysua.weights.safetensors"),
        runtime_bytes(parameters, m14_metadata(other)),
    )
    .expect("wrong source tuple");
    let error =
        TrainingArtifact::initialize_selected_m14_for_map2(&directory.0, 1, PolicyDevice::Cpu)
            .err()
            .expect("a digest must stay paired with its tuple");
    assert_eq!(
        error,
        CheckpointError::TensorContract("selected M14 Map2 initialization source SHA-256")
    );
    assert_eq!(
        error.to_string(),
        "checkpoint tensor contract has invalid selected M14 Map2 initialization source SHA-256"
    );
}

fn initialization_run(config: crate::PpoConfig, provenance: &str) -> crate::CheckpointRun {
    crate::CheckpointRun {
        git_commit: std::env::var("DRYSUA_INITIALIZATION_GIT_COMMIT")
            .expect("actual source revision"),
        simulator_commit: std::env::var("DRYSUA_INITIALIZATION_SIMULATOR_COMMIT")
            .expect("actual simulator revision"),
        enabled_features: crate::compiled_features(),
        command_line: format!(
            "tests::map2_initialization_utility::map2_initialize_pinned_m14_local_artifact_only --exact --ignored; {provenance}"
        ),
        run_seed: INITIALIZATION_SEED,
        map: bota_proto::MapId(2),
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size: config.minibatch,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

fn verify_output(output: &Path, run: &crate::CheckpointRun, parameters: &[f32]) {
    let runtime = PolicyModel::fresh(2).expect("runtime");
    TrainingArtifact::load_runtime_weights(&runtime, output)
        .expect("strict current runtime reload");
    assert_bits(
        &runtime.export_parameters().expect("runtime bits"),
        parameters,
    );
    let checkpoint = TrainingArtifact::load_compatible(output, run).expect("new checkpoint reload");
    assert_eq!(checkpoint.progress(), &progress());
    let restored = PolicyModel::fresh(3).expect("checkpoint target");
    let state = checkpoint
        .restore(&restored, run)
        .expect("new current checkpoint restore");
    assert_fresh_state(&restored, state.trainer());
    assert_bits(
        &restored.export_parameters().expect("restored bits"),
        parameters,
    );
}

fn write_provenance(output: &Path, source: &Path, run: &crate::CheckpointRun, provenance: &str) {
    use std::io::Write;
    let bytes = fs::read(output.join("drysua.weights.safetensors")).expect("new runtime bytes");
    let output_hash: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let text = format!(
        "{provenance}\nsource={}\noutput_sha256={output_hash}\ngit_commit={}\nsimulator_commit={}\nenabled_features={}\nseed={INITIALIZATION_SEED}\ntraining_updates=0\ngameplay_runs=0\noptimizer_moments=positive_zero\nprogress=all_zero_no_evaluation_rng_history_or_league\nsource_unchanged=true\nnamed_tensors=62\nold_parameters=1689076\nnew_parameters=1696436\nunit.0.weight=73x64_to_84x64_insert_rows73..84\ntrunk.0.weight=2576x512_to_2589x512_insert_rows72..85\nall_other_tensors_bit_identical=true\nruntime_and_checkpoint_reload_exact=true\nfeature_hash={}\naction_hash={}\nmodel_hash={}\nppo_hash={}\nleague_hash={}\ncheckpoint_hash={}\nreward_descriptor={}\n",
        source.display(),
        run.git_commit,
        run.simulator_commit,
        run.enabled_features,
        crate::FEATURE_SCHEMA_HASH,
        crate::ACTION_SCHEMA_HASH,
        crate::MODEL_SCHEMA_HASH,
        crate::PPO_SCHEMA_HASH,
        crate::LEAGUE_SCHEMA_HASH,
        crate::CHECKPOINT_SCHEMA_HASH,
        crate::MAP2_REWARD_SCHEMA_DESCRIPTOR,
    );
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("INITIALIZATION.txt"))
        .expect("new explicit provenance");
    file.write_all(text.as_bytes()).expect("write provenance");
    file.sync_all().expect("durable provenance");
    fs::File::open(output)
        .expect("directory")
        .sync_all()
        .expect("durable directory");
}
