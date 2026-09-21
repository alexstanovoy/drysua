use super::*;

#[test]
#[ignore = "read-only check of the locally selected initializer; requires DRYSUA_INITIALIZER_TEST_DIRECTORY"]
fn selected_legacy_initializer_loads_without_rewriting_files() {
    let directory = std::env::var_os("DRYSUA_INITIALIZER_TEST_DIRECTORY")
        .map(PathBuf::from)
        .expect("explicit initializer fixture path");
    let before = initializer_file_digests(&directory);
    let bytes = fs::read(directory.join(RUNTIME_TENSOR_FILE)).expect("old runtime");
    let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
    assert_eq!(metadata.metadata(), &Some(legacy_runtime_metadata()));
    let parameters = decode_runtime_tensor(&bytes).expect("legacy decoder");
    let model = PolicyModel::fresh(40_008).expect("CPU model");
    TrainingArtifact::load_runtime_weights(&model, &directory).expect("selected initializer loads");
    assert_eq!(
        model.export_parameters().expect("loaded parameters"),
        parameters
    );
    #[cfg(feature = "cuda")]
    {
        let cuda =
            PolicyModel::fresh_on(40_008, PolicyDevice::Cuda { ordinal: 0 }).expect("CUDA model");
        TrainingArtifact::load_runtime_weights(&cuda, &directory)
            .expect("selected initializer loads on CUDA");
        assert_eq!(
            cuda.export_parameters().expect("CUDA parameters"),
            parameters
        );
    }
    assert_eq!(initializer_file_digests(&directory), before);
}

fn initializer_file_digests(directory: &Path) -> Vec<(std::ffi::OsString, Vec<u8>)> {
    let mut hashes = Vec::new();
    for entry in fs::read_dir(directory).expect("initializer directory") {
        let entry = entry.expect("entry");
        assert!(entry.file_type().expect("type").is_file());
        assert!(hashes.len() < 16);
        let bytes = fs::read(entry.path()).expect("file");
        hashes.push((entry.file_name(), Sha256::digest(bytes).to_vec()));
    }
    hashes.sort();
    hashes
}

// Recorded from artifacts/runtimes-current/initial-next-run/checkpoint.meta.
const LEGACY_CHECKPOINT_HASH: u64 = 0x496a_ad2e_adbb_e586;
const LEGACY_PPO_HASH: u64 = 0xb18a_050a_dd4a_85cd;
const LEGACY_SCHEMAS: [(u32, u64); 5] = [
    (5, 0x93ea_35fd_6524_75a7),
    (22, 0x9272_3c71_b527_88b0),
    (24, 0xa799_157e_02a9_5f4e),
    (37, LEGACY_PPO_HASH),
    (7, 0x64f4_c97d_0c52_d062),
];

#[test]
fn standard_checkpoint_identity_matches_the_recorded_legacy_header() {
    assert_eq!(CHECKPOINT_SCHEMA_VERSION, 12);
    assert_eq!(CHECKPOINT_SCHEMA_HASH, LEGACY_CHECKPOINT_HASH);
    assert_eq!(LINKED_SCHEMAS, LEGACY_SCHEMAS);
    assert_eq!(PpoSampleBudget::Standard.schema_version(), 37);
    assert_eq!(PpoSampleBudget::Standard.schema_hash(), LEGACY_PPO_HASH);
}

#[test]
fn standard_manifest_encoding_preserves_every_legacy_field_byte() {
    let artifact = manifest_artifact(PpoSampleBudget::Standard, 3, 131_072);
    let expected = legacy_manifest(&artifact);

    let actual = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let decoded = decode_manifest(&expected).expect("legacy decode");

    assert_eq!(actual, expected);
    assert_eq!(decoded.config, artifact.config);
    assert_eq!(decoded.progress, artifact.progress);
}

#[test]
fn annealed_manifest_roundtrips_profile_and_reuses_the_64_byte_config() {
    let artifact = manifest_artifact(PpoSampleBudget::Annealed, 3, 139_560);
    let mut writer = ManifestWriter::default();

    encode_config(&mut writer, artifact.config).expect("config");
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    let decoded = decode_manifest(&encoded).expect("annealed decode");

    assert_eq!(writer.bytes.len(), 64);
    assert_eq!(&encoded[8..12], &13u32.to_le_bytes());
    assert_ne!(&encoded[12..20], &LEGACY_CHECKPOINT_HASH.to_le_bytes());
    assert_eq!(&encoded[56..60], &38u32.to_le_bytes());
    assert_eq!(decoded.config, artifact.config);
    assert_eq!(decoded.progress, artifact.progress);
    assert_eq!(decoded.tensor_hash, artifact.tensor_hash);
    assert_eq!(decoded.shuffle, artifact.shuffle);
}

#[test]
fn manifest_rejects_mixed_checkpoint_and_linked_ppo_identities() {
    let standard = encoded_manifest(PpoSampleBudget::Standard, 0, 0);
    let annealed = encoded_manifest(PpoSampleBudget::Annealed, 0, 0);

    for (source, other) in [(&standard, &annealed), (&annealed, &standard)] {
        for range in [8..12, 12..20, 8..20, 56..60, 60..68, 56..68] {
            let mut mixed = source.clone();
            mixed[range.clone()].copy_from_slice(&other[range]);
            let error = decode_manifest(&mixed).expect_err("mixed schema");
            assert_eq!(error, CheckpointError::SchemaMismatch);
            assert_eq!(
                error.to_string(),
                "checkpoint schema does not match this build"
            );
        }
    }
}

#[test]
fn downgraded_annealed_manifest_rejects_oversized_standard_config() {
    let standard = encoded_manifest(PpoSampleBudget::Standard, 0, 0);
    let mut annealed = encoded_manifest(PpoSampleBudget::Annealed, 0, 0);
    annealed[..80].copy_from_slice(&standard[..80]);

    let error = decode_manifest(&annealed).expect_err("downgraded capacity");

    assert_eq!(error, CheckpointError::InvalidManifest("PPO config"));
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid PPO config"
    );
}

#[test]
fn annealed_manifest_rejects_invalid_dimensions_and_nonfinite_hyperparameters() {
    let original = encoded_manifest(PpoSampleBudget::Annealed, 0, 0);
    let config_offset = original.len() - 128;

    for (offset, value) in [
        (0, 0u32),
        (4, 1_162),
        (8, 41),
        (8, 42),
        (12, 17),
        (16, 8_193),
        (32, f32::NAN.to_bits()),
        (52, 0.0f32.to_bits()),
    ] {
        let mut encoded = original.clone();
        let start = config_offset + offset;
        encoded[start..start + 4].copy_from_slice(&value.to_le_bytes());
        let error = decode_manifest(&encoded).expect_err("invalid config");
        assert_eq!(error, CheckpointError::InvalidManifest("PPO config"));
    }
}

#[test]
fn profile_sample_counters_accept_the_boundary_and_reject_one_more() {
    for (budget, updates, maximum) in [
        (PpoSampleBudget::Standard, 0, 32_768),
        (PpoSampleBudget::Standard, 3, 131_072),
        (PpoSampleBudget::Annealed, 0, 0),
        (PpoSampleBudget::Annealed, 1, 46_520),
        (PpoSampleBudget::Annealed, 2, 93_040),
        (PpoSampleBudget::Annealed, 3, 139_560),
    ] {
        let accepted = encoded_manifest(budget, updates, maximum);
        let rejected = encoded_manifest(budget, updates, maximum + 1);

        let decoded = decode_manifest(&accepted).expect("counter boundary");
        let error = decode_manifest(&rejected).expect_err("counter overflow");

        assert_eq!(decoded.progress.rollout_samples, maximum);
        assert_eq!(
            error,
            CheckpointError::InvalidManifest("rollout sample counter")
        );
        assert_eq!(
            error.to_string(),
            "checkpoint manifest has invalid rollout sample counter"
        );
    }
}

#[test]
fn annealed_sample_counter_uses_configured_games_not_the_profile_ceiling() {
    let mut artifact = manifest_artifact(PpoSampleBudget::Annealed, 3, 6_978);
    artifact.config.environments = 2;
    let accepted = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");
    artifact.progress.rollout_samples += 1;
    let rejected = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");

    assert_eq!(
        decode_manifest(&accepted)
            .expect("two games")
            .progress
            .rollout_samples,
        6_978
    );
    assert_eq!(
        decode_manifest(&rejected).expect_err("two-game counter overflow"),
        CheckpointError::InvalidManifest("rollout sample counter")
    );
}

#[test]
fn annealed_manifest_rejects_mastery_even_when_its_game_count_fits() {
    let mut artifact = manifest_artifact(PpoSampleBudget::Annealed, 0, 0);
    artifact.config.environments = 2;
    artifact.run.mastery_config = Some(crate::MasteryConfig::default());
    artifact.progress.mastery = Some(crate::MasteryProgress::default());
    let encoded = encode_manifest(&artifact, artifact.tensor_hash).expect("manifest");

    let error = decode_manifest(&encoded).expect_err("annealed mastery");

    assert_eq!(error, CheckpointError::InvalidManifest("annealed mastery"));
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid annealed mastery"
    );
}

#[test]
fn profile_manifests_reject_truncation_and_trailing_bytes() {
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let mut encoded = encoded_manifest(budget, 0, 0);
        let truncated = decode_manifest(&encoded[..encoded.len() - 1]);
        encoded.push(0);
        let trailing = decode_manifest(&encoded);

        assert_eq!(
            truncated.expect_err("truncated"),
            CheckpointError::ManifestTruncated
        );
        assert_eq!(
            trailing.expect_err("trailing"),
            CheckpointError::ManifestTrailingBytes
        );
    }
}

#[test]
fn public_load_rejects_invalid_annealed_config_before_tensor_io() {
    let directory = test_directory("invalid-config");
    let mut encoded = encoded_manifest(PpoSampleBudget::Annealed, 0, 0);
    let start = encoded.len() - 128 + 8;
    encoded[start..start + 4].copy_from_slice(&42u32.to_le_bytes());
    fs::write(directory.join(CHECKPOINT_META_FILE), encoded).expect("manifest fixture");

    let error = TrainingArtifact::load(&directory).expect_err("config before missing tensors");

    assert_eq!(error, CheckpointError::InvalidManifest("PPO config"));
    fs::remove_dir_all(directory).expect("remove fixture");
}

#[test]
fn annealed_public_capture_save_load_restore_preserves_profile_and_state() {
    let directory = test_directory("roundtrip");
    let fixture = manifest_artifact(PpoSampleBudget::Annealed, 0, 0);
    let source = PolicyModel::fresh(40_008).expect("source model");
    let trainer = PpoTrainer::new(&source, fixture.config, 91).expect("trainer");
    let mut artifact = TrainingArtifact::capture(&source, &trainer, fixture.run, fixture.progress)
        .expect("capture");
    artifact.trainer_updates = 3;
    artifact.progress = manifest_artifact(PpoSampleBudget::Annealed, 3, 139_560).progress;

    artifact.save(&directory).expect("save");
    let loaded = TrainingArtifact::load_compatible(&directory, artifact.run()).expect("load");
    let target = PolicyModel::fresh(40_009).expect("target model");
    let restored = loaded.restore(&target, artifact.run()).expect("restore");

    assert_eq!(loaded.config(), trainer.config());
    assert_eq!(restored.trainer().config(), trainer.config());
    assert_eq!(restored.trainer().updates(), 3);
    assert_eq!(
        restored.trainer().optimizer_step(),
        trainer.optimizer_step()
    );
    assert_eq!(
        restored.trainer().rng_checkpoint(),
        trainer.rng_checkpoint()
    );
    assert_eq!(restored.progress(), artifact.progress());
    assert_eq!(
        target.export_parameters().expect("restored parameters"),
        artifact.parameters
    );
    assert_eq!(loaded.parameters, artifact.parameters);
    assert_eq!(
        loaded.optimizer.first_moment,
        artifact.optimizer.first_moment
    );
    assert_eq!(
        loaded.optimizer.second_moment,
        artifact.optimizer.second_moment
    );
    let error = restored
        .pipeline(1, 1, &target)
        .err()
        .expect("standard-only pipeline");
    assert_eq!(
        error,
        CheckpointError::InvalidManifest("annealed actor-learner pipeline")
    );
    fs::remove_dir_all(directory).expect("remove fixture");
}

#[test]
fn annealed_public_capture_rejects_samples_before_the_first_committed_update() {
    let fixture = manifest_artifact(PpoSampleBudget::Annealed, 0, 1);
    let model = PolicyModel::fresh(40_013).expect("model");
    let trainer = PpoTrainer::new(&model, fixture.config, 93).expect("trainer");

    let error = TrainingArtifact::capture(&model, &trainer, fixture.run, fixture.progress)
        .expect_err("uncommitted samples");

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("rollout sample counter")
    );
    assert_eq!(
        error.to_string(),
        "checkpoint manifest has invalid rollout sample counter"
    );
}

#[test]
fn annealed_public_restore_rejects_excess_samples_before_parameter_mutation() {
    let fixture = manifest_artifact(PpoSampleBudget::Annealed, 0, 0);
    let source = PolicyModel::fresh(40_014).expect("source model");
    let trainer = PpoTrainer::new(&source, fixture.config, 94).expect("trainer");
    let mut artifact = TrainingArtifact::capture(&source, &trainer, fixture.run, fixture.progress)
        .expect("capture");
    artifact.trainer_updates = 3;
    artifact.progress = manifest_artifact(PpoSampleBudget::Annealed, 3, 139_561).progress;
    let target = PolicyModel::fresh(40_015).expect("target model");
    let before = target.export_parameters().expect("before");

    let error = artifact
        .restore(&target, artifact.run())
        .err()
        .expect("excess committed samples");

    assert_eq!(
        error,
        CheckpointError::InvalidManifest("rollout sample counter")
    );
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn annealed_public_capture_and_restore_reject_mastery_without_mutation() {
    let fixture = manifest_artifact(PpoSampleBudget::Annealed, 0, 0);
    let source = PolicyModel::fresh(40_010).expect("source model");
    let trainer = PpoTrainer::new(&source, fixture.config, 92).expect("trainer");
    let mut artifact = TrainingArtifact::capture(
        &source,
        &trainer,
        fixture.run.clone(),
        fixture.progress.clone(),
    )
    .expect("valid capture");
    artifact.run.mastery_config = Some(crate::MasteryConfig::default());
    artifact.progress.mastery = Some(crate::MasteryProgress::default());

    let capture_error = TrainingArtifact::capture(
        &source,
        &trainer,
        artifact.run.clone(),
        artifact.progress.clone(),
    )
    .expect_err("capture mastery");
    let target = PolicyModel::fresh(40_011).expect("target model");
    let before = target.export_parameters().expect("before");
    let restore_error = artifact
        .restore(&target, artifact.run())
        .err()
        .expect("restore mastery");

    assert_eq!(
        capture_error,
        CheckpointError::InvalidManifest("annealed mastery")
    );
    assert_eq!(restore_error, capture_error);
    assert_eq!(target.export_parameters().expect("after"), before);
}

#[test]
fn standard_runtime_encoding_preserves_the_legacy_canonical_bytes() {
    let parameters = [0.0, 1.0, -0.0];
    let metadata: std::collections::BTreeMap<_, _> =
        legacy_runtime_metadata().into_iter().collect();
    let metadata = serde_json::to_string(&metadata).expect("legacy metadata JSON");
    let mut header = format!(
        "{{\"__metadata__\":{metadata},\"model.parameters\":{{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]}}}}"
    )
    .into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut expected = (header.len() as u64).to_le_bytes().to_vec();
    expected.extend(header);
    expected.extend(encode_f32(&parameters));

    let actual = serialize_runtime_tensor(&parameters, PpoSampleBudget::Standard).expect("runtime");

    assert_eq!(actual, expected);
}

#[test]
fn runtime_accepts_exact_profile_tuples_and_legacy_unordered_metadata() {
    let parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    let legacy = runtime_fixture(&parameters, legacy_runtime_metadata());

    assert_eq!(
        decode_runtime_tensor(&legacy).expect("legacy metadata"),
        parameters
    );
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let bytes = serialize_runtime_tensor(&parameters, budget).expect("runtime");
        let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
        let metadata = metadata.metadata().as_ref().expect("metadata map");
        assert_eq!(metadata.len(), 9);
        assert_eq!(
            metadata["ppo_schema_version"],
            budget.schema_version().to_string()
        );
        assert_eq!(
            metadata["ppo_schema_hash"],
            budget.schema_hash().to_string()
        );
        assert_eq!(
            decode_runtime_tensor(&bytes).expect("profile tuple"),
            parameters
        );
    }
}

#[test]
fn runtime_rejects_mixed_profiles_and_changed_noncapacity_identity_fields() {
    let parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    let mut annealed = legacy_runtime_metadata();
    annealed.insert(
        "ppo_schema_version".to_owned(),
        PpoSampleBudget::Annealed.schema_version().to_string(),
    );
    annealed.insert(
        "ppo_schema_hash".to_owned(),
        PpoSampleBudget::Annealed.schema_hash().to_string(),
    );
    for (key, value) in [
        ("ppo_schema_version", "37".to_owned()),
        ("ppo_schema_hash", LEGACY_PPO_HASH.to_string()),
        ("action_schema_hash", "0".to_owned()),
        ("feature_schema_hash", "0".to_owned()),
        ("model_schema_hash", "0".to_owned()),
        ("ppo_rules_audit_version", "0".to_owned()),
        ("map2_reward_schema_version", "0".to_owned()),
        ("map2_reward_schema_hash", "0".to_owned()),
        ("map2_reward_schema_descriptor", "unknown".to_owned()),
        ("unknown_profile", "annealed".to_owned()),
    ] {
        let mut metadata = annealed.clone();
        metadata.insert(key.to_owned(), value);
        let bytes = runtime_fixture(&parameters, metadata);
        assert_eq!(
            decode_runtime_tensor(&bytes).expect_err("mixed tuple"),
            CheckpointError::SchemaMismatch
        );
    }
}

#[test]
fn annealed_runtime_identity_does_not_bypass_tensor_shape_or_finiteness() {
    let short = serialize_runtime_tensor(&[0.0], PpoSampleBudget::Annealed).expect("short fixture");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    parameters[0] = f32::NAN;
    let nonfinite =
        serialize_runtime_tensor(&parameters, PpoSampleBudget::Annealed).expect("NaN fixture");

    assert_eq!(
        decode_runtime_tensor(&short).expect_err("shape"),
        CheckpointError::TensorContract("dtype or shape")
    );
    assert_eq!(
        decode_runtime_tensor(&nonfinite).expect_err("NaN"),
        CheckpointError::NonFiniteTensor {
            name: "model.parameters",
            index: 0
        }
    );
}

#[test]
fn public_runtime_export_selects_profile_and_rejects_mixed_tuple_without_mutation() {
    let directory = test_directory("runtime");
    let model = PolicyModel::fresh(40_012).expect("model");
    let parameters = model.export_parameters().expect("parameters");
    let path = directory.join(RUNTIME_TENSOR_FILE);
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("legacy export");
    let legacy = fs::read(&path).expect("legacy bytes");
    TrainingArtifact::save_runtime_weights_with_budget(
        &model,
        &directory,
        PpoSampleBudget::Standard,
    )
    .expect("explicit standard");
    assert_eq!(fs::read(&path).expect("standard bytes"), legacy);

    TrainingArtifact::save_runtime_weights_with_budget(
        &model,
        &directory,
        PpoSampleBudget::Annealed,
    )
    .expect("annealed export");
    let annealed = fs::read(&path).expect("annealed bytes");
    assert_ne!(annealed, legacy);
    TrainingArtifact::load_runtime_weights(&model, &directory).expect("annealed import");
    assert_eq!(
        model.export_parameters().expect("imported parameters"),
        parameters
    );

    let mut mixed = legacy_runtime_metadata();
    mixed.insert(
        "ppo_schema_version".to_owned(),
        PpoSampleBudget::Annealed.schema_version().to_string(),
    );
    fs::write(&path, runtime_fixture(&parameters, mixed)).expect("mixed fixture");
    let before = model.policy_identity().expect("before identity");
    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory).expect_err("mixed import");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(model.policy_identity().expect("after identity"), before);
    assert_eq!(
        model.export_parameters().expect("after parameters"),
        parameters
    );
    fs::remove_dir_all(directory).expect("remove fixture");
}

fn manifest_artifact(budget: PpoSampleBudget, updates: u64, samples: u64) -> TrainingArtifact {
    TrainingArtifact {
        run: CheckpointRun {
            mastery_config: None,
            git_commit: "capacity-drysua".to_owned(),
            simulator_commit: "capacity-simulator".to_owned(),
            enabled_features: compiled_features(),
            command_line: "train-annealed capacity-fixture".to_owned(),
            run_seed: 40_008,
            map: MapId(2),
            hero: SHADOW_FIEND,
            device: CheckpointDevice::Cpu,
            batch_size: 512,
            rules_audit_version: PPO_RULES_AUDIT_VERSION,
        },
        progress: CheckpointProgress {
            mastery: None,
            global_update: updates,
            policy_version: updates,
            scheduler_step: updates,
            curriculum_stage: 0,
            rollout_samples: samples,
            best_evaluation: None,
            rng_states: vec![RngCheckpoint::new("ppo_actor_sampling", 8, 40).expect("RNG")],
            league_references: Vec::new(),
        },
        config: PpoConfig {
            sample_budget: budget,
            environments: if budget == PpoSampleBudget::Annealed {
                40
            } else {
                2
            },
            rollout_decisions: 1_163,
            decision_interval_ticks: 3,
            epochs: 1,
            minibatch: 512,
            gamma_tick: 1.0,
            ..PpoConfig::default()
        },
        trainer_updates: updates,
        shuffle: (17, 19),
        parameters: Vec::new(),
        optimizer: CheckpointOptimizer {
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            step: 0,
        },
        tensor_hash: [42; 32],
    }
}

fn encoded_manifest(budget: PpoSampleBudget, updates: u64, samples: u64) -> Vec<u8> {
    let artifact = manifest_artifact(budget, updates, samples);
    encode_manifest(&artifact, artifact.tensor_hash).expect("manifest fixture")
}

fn legacy_manifest(artifact: &TrainingArtifact) -> Vec<u8> {
    assert_eq!(artifact.config.sample_budget, PpoSampleBudget::Standard);
    assert_eq!(
        artifact.config,
        manifest_artifact(PpoSampleBudget::Standard, 0, 0).config
    );
    let mut writer = ManifestWriter::default();
    writer.bytes.extend(b"DRYCKP18");
    writer.u32(12);
    writer.u64(LEGACY_CHECKPOINT_HASH);
    for (version, hash) in LEGACY_SCHEMAS {
        writer.u32(version);
        writer.u64(hash);
    }
    encode_run(&mut writer, &artifact.run).expect("legacy run");
    encode_progress(&mut writer, &artifact.progress).expect("legacy progress");
    for dimension in [3, 1_163, 2, 1, 512] {
        writer.u32(dimension);
    }
    for value in [
        0.2, 0.5, 0.01, 3.0e-6, 0.9, 0.999, 1.0e-5, 0.5, 1.0, 0.98, 0.02,
    ] {
        writer.f32(value);
    }
    writer.u64(artifact.trainer_updates);
    writer.u64(artifact.optimizer.step);
    writer.u64(artifact.shuffle.0);
    writer.u64(artifact.shuffle.1);
    writer.bytes.extend(artifact.tensor_hash);
    writer.bytes
}

fn legacy_runtime_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", LEGACY_SCHEMAS[0].1.to_string()),
        ("feature_schema_hash", LEGACY_SCHEMAS[1].1.to_string()),
        ("model_schema_hash", LEGACY_SCHEMAS[2].1.to_string()),
        ("ppo_schema_version", "37".to_owned()),
        ("ppo_schema_hash", LEGACY_PPO_HASH.to_string()),
        ("ppo_rules_audit_version", "32".to_owned()),
        ("map2_reward_schema_version", "7".to_owned()),
        ("map2_reward_schema_hash", LEGACY_SCHEMAS[4].1.to_string()),
        (
            "map2_reward_schema_descriptor",
            MAP2_REWARD_SCHEMA_DESCRIPTOR.to_owned(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

fn runtime_fixture(parameters: &[f32], metadata: HashMap<String, String>) -> Vec<u8> {
    let bytes = encode_f32(parameters);
    let view = TensorView::new(Dtype::F32, vec![parameters.len()], &bytes).expect("view");
    serialize([("model.parameters".to_owned(), view)], Some(metadata)).expect("runtime fixture")
}

fn test_directory(name: &str) -> PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-checkpoint-capacity-{name}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&directory).expect("create fixture directory");
    directory
}
