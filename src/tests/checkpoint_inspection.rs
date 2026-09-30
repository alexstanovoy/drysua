use super::*;
use serde_json::Value;

#[test]
fn inspection_contract_exposes_current_model_limits_and_build_capabilities() {
    let contract = parse_json(&checkpoint_inspection_contract().expect("contract"));
    assert_eq!(contract["kind"], "contract");
    assert_eq!(contract["schema"], "drysua-checkpoint-inspection/v1");
    assert_eq!(contract["model"]["version"], MODEL_SCHEMA_VERSION);
    assert_eq!(
        contract["model"]["hash"],
        format!("{MODEL_SCHEMA_HASH:016x}")
    );
    assert_eq!(contract["model"]["parameters"], MODEL_PARAMETER_COUNT);
    assert_eq!(contract["enabled_features"], compiled_features());
    assert_eq!(
        contract["capabilities"]["inspection"],
        cfg!(target_os = "linux")
    );
    assert_eq!(
        contract["capabilities"]["annealed_history"],
        cfg!(feature = "builtin")
    );
    assert_eq!(contract["limits"]["max_json_bytes"], 4_194_304);
    assert_eq!(contract["limits"]["max_snapshots"], 10_000);
    assert_eq!(contract["limits"]["max_files"], 10_004);
    assert_eq!(contract["limits"]["max_slots"], crate::PPO_MAX_SLOTS);
    assert_eq!(
        contract["checkpoint"],
        serde_json::json!({"version": CHECKPOINT_SCHEMA_VERSION,
            "hash": format!("{CHECKPOINT_SCHEMA_HASH:016x}")})
    );
}

fn parse_json(bytes: &[u8]) -> Value {
    assert!(bytes.len() <= 4 * 1024 * 1024);
    assert_eq!(bytes.last(), Some(&b'\n'));
    serde_json::from_slice(bytes).expect("inspection JSON")
}

#[cfg(target_os = "linux")]
mod files {
    use super::*;
    use serde_json::json;
    use std::collections::{BTreeMap, BTreeSet};
    use std::os::unix::fs::symlink;

    const FIXED: &str = "train-annealed --updates 8 --generation-updates 2 --zero-updates 0";

    #[test]
    fn generic_inspection_is_read_only_and_never_restores_the_recorded_cuda_device() {
        let mut artifact = fixture_artifact("other-command --updates 8", 0);
        let fixture = Fixture::new(&mut artifact);
        let report = inspect(&fixture).expect("inspection without a device");
        assert_eq!(report["schema"], "drysua-checkpoint-inspection/v1");
        assert_eq!(report["kind"], "other");
        assert_eq!(
            report["run"]["device"],
            json!({"kind": "cuda", "ordinal": u32::MAX})
        );
        assert_current_identity(&fixture, &artifact, &report);
        assert_eq!(report["progress"]["updates"], 0);
        assert_eq!(report["progress"]["optimizer_steps"], 0);
        assert_eq!(report["progress"]["rollout_samples"], 0);
        assert!(report["progress"]["games"].is_null());
        assert!(report["adaptive"].is_null());
        assert_eq!(report["history"]["kind"], "unsupported");
        assert_eq!(report["history"]["verified"], false);
        assert_eq!(report["runtime_status"], "matched");
        assert_eq!(report["runtime_matches_model"], true);
        assert_inventory(&fixture, &report);
        assert_eq!(report["files"].as_array().unwrap().len(), 3);
        for name in [CHECKPOINT_META_FILE, RUNTIME_TENSOR_FILE] {
            assert!(listed(&report, name), "matched inventory contains {name}");
        }
    }

    #[test]
    fn runtime_missing_lagging_and_old_schema_are_classified_without_mutation() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let path = fixture.0.join(RUNTIME_TENSOR_FILE);
        fs::remove_file(&path).expect("own missing export");
        let missing = inspect(&fixture).expect("runtime optional");
        assert_eq!(missing["runtime_status"], "missing");
        assert_eq!(missing["runtime_matches_model"], false);
        assert!(missing["identity"]["runtime_sha256"].is_null());
        assert!(!listed(&missing, RUNTIME_TENSOR_FILE));
        for value in [1.0, 0.0] {
            let mut lagging = artifact.parameters.clone();
            lagging[0] = value;
            fs::write(&path, serialize_runtime_tensor(&lagging).unwrap()).unwrap();
            assert_runtime_mismatch(&fixture, &path);
        }
        let old =
            serialize_named_tensors(&[("model.parameters", &artifact.parameters)], None).unwrap();
        fs::write(&path, old).expect("runtime without current metadata");
        assert_runtime_mismatch(&fixture, &path);
    }

    #[test]
    fn malformed_runtime_is_an_error_not_a_lagging_export() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let mut schema = current_parameter_schema().unwrap();
        schema[0].1.reverse();
        fs::write(
            fixture.0.join(RUNTIME_TENSOR_FILE),
            runtime::serialize(&schema, &artifact.parameters).unwrap(),
        )
        .unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("wrong runtime shape"),
            CheckpointError::TensorContract("dtype or shape")
        );
    }

    #[test]
    fn immutable_payload_is_listed_and_checksummed() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let report = inspect(&fixture).expect("named tensor");
        let name = tensor_generation_path(&fixture.0, artifact.tensor_hash);
        assert!(listed(&report, name.file_name().unwrap().to_str().unwrap()));
        assert_inventory(&fixture, &report);
        let mut corrupt = fs::read(&name).unwrap();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(name, corrupt).unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("named checksum corruption"),
            CheckpointError::TensorHashMismatch
        );
    }

    #[test]
    fn unknown_checkpoint_identity_is_rejected_without_migration() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let mut writer = ManifestWriter::default();
        writer.bytes.extend(CHECKPOINT_MAGIC);
        writer.u32(u32::MAX);
        writer.u64(0);
        fs::write(fixture.0.join(CHECKPOINT_META_FILE), writer.bytes).unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("unknown tuple"),
            CheckpointError::SchemaMismatch
        );
    }

    #[test]
    fn unsafe_or_oversized_runtime_is_rejected_before_runtime_classification() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let path = fixture.0.join(RUNTIME_TENSOR_FILE);
        fs::remove_file(&path).unwrap();
        symlink(CHECKPOINT_META_FILE, &path).unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("runtime symlink"),
            CheckpointError::InvalidManifest("inspection requires a non-symlink regular file")
        );
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("runtime directory"),
            CheckpointError::InvalidManifest("inspection requires a non-symlink regular file")
        );
        fs::remove_dir(&path).unwrap();
        File::create(&path)
            .unwrap()
            .set_len(MAX_RUNTIME_TENSOR_BYTES + 1)
            .unwrap();
        assert_eq!(
            inspect(&fixture).expect_err("runtime size"),
            CheckpointError::InvalidManifest("inspection file size")
        );
    }

    #[test]
    fn manifest_change_after_inventory_is_rejected_deterministically() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        artifact.run.git_commit = "next-inspection-fixture".to_owned();
        let changed = encode_manifest(&artifact, artifact.tensor_hash).unwrap();
        let mut before = inventory(&fixture.0);
        before.remove(Path::new(CHECKPOINT_META_FILE));
        let error = inspect_with_hook(&fixture.0, || {
            let pending = fixture.0.join("inspection-next.meta");
            fs::write(&pending, &changed).unwrap();
            fs::rename(pending, fixture.0.join(CHECKPOINT_META_FILE)).unwrap();
        })
        .expect_err("manifest changed at hook");
        assert_eq!(
            error,
            CheckpointError::InvalidManifest("checkpoint changed during inspection")
        );
        assert_eq!(
            fs::read(fixture.0.join(CHECKPOINT_META_FILE)).unwrap(),
            changed
        );
        let mut after = inventory(&fixture.0);
        after.remove(Path::new(CHECKPOINT_META_FILE));
        assert_eq!(
            after, before,
            "only the hook's manifest replacement may change files"
        );
    }

    #[test]
    fn checkpoint_directory_rejects_symlinked_ancestors_and_parent_components() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let direct = fixture.0.join("checkpoint-link");
        symlink(&fixture.0, &direct).unwrap();
        let ancestor = fixture.0.join("parent-link");
        symlink(fixture.0.parent().unwrap(), &ancestor).unwrap();
        let through_ancestor = ancestor.join(fixture.0.file_name().unwrap());
        let child = fixture.0.join("child");
        fs::create_dir(&child).unwrap();
        for (path, message) in [
            (direct, "inspection requires non-symlink directories"),
            (
                through_ancestor,
                "inspection requires non-symlink directories",
            ),
            (child.join(".."), "inspection path component"),
        ] {
            assert_eq!(
                inspect_path(&fixture, &path).expect_err("unsafe checkpoint path"),
                CheckpointError::InvalidManifest(message)
            );
        }
    }

    #[test]
    fn replacing_checkpoint_root_after_inventory_is_detected_and_fixture_is_restored() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let holding = Fixture(crate::ppo::test_directory("inspection-root-holding"));
        let swap = RootSwap {
            original: &fixture.0,
            saved: holding.0.join("original"),
        };
        let before = inventory(&fixture.0);
        let result = inspect_with_hook(&fixture.0, || {
            fs::rename(&fixture.0, &swap.saved).expect("move owned checkpoint root");
            fs::create_dir(&fixture.0).expect("empty replacement root");
        });
        drop(swap);
        assert_eq!(
            result.expect_err("root identity changed"),
            CheckpointError::InvalidManifest("checkpoint changed during inspection")
        );
        assert_eq!(inventory(&fixture.0), before);
        assert!(fs::read_dir(&holding.0).unwrap().next().is_none());
    }

    #[test]
    fn symlinked_manifest_or_payload_is_rejected() {
        let mut artifact = fixture_artifact("other-command", 0);
        let fixture = Fixture::new(&mut artifact);
        let generation = tensor_generation_path(Path::new(""), artifact.tensor_hash);
        for name in [Path::new(CHECKPOINT_META_FILE), generation.as_path()] {
            let primary = fixture.0.join(name);
            let aside = fixture.0.join("aside");
            fs::rename(&primary, &aside).unwrap();
            symlink("aside", &primary).unwrap();
            let result = inspect(&fixture);
            fs::remove_file(&primary).unwrap();
            fs::rename(&aside, &primary).unwrap();
            assert_eq!(
                result.expect_err("unsafe primary"),
                CheckpointError::InvalidManifest("inspection requires a non-symlink regular file")
            );
        }
    }

    #[test]
    fn scope_hash_ignores_progress_and_tensor_updates_but_binds_optimizer_configuration() {
        let mut artifact = fixture_artifact("other-command --updates 8", 0);
        let fixture = Fixture::new(&mut artifact);
        let initial = inspect(&fixture).expect("initial identity");
        artifact.progress.global_update = 1;
        artifact.trainer_updates = 1;
        fixture.manifest(&artifact);
        let progressed = inspect(&fixture).expect("progress identity");
        assert_eq!(
            progressed["identity"]["scope_sha256"],
            initial["identity"]["scope_sha256"]
        );
        assert_ne!(
            progressed["identity"]["manifest_sha256"],
            initial["identity"]["manifest_sha256"]
        );
        assert_eq!(
            progressed["identity"]["tensor_sha256"],
            initial["identity"]["tensor_sha256"]
        );
        assert_eq!(progressed["progress"]["updates"], 1);
        artifact.parameters[1] = 0.25;
        fixture.publish(&mut artifact);
        let updated = inspect(&fixture).expect("new tensor identity");
        assert_eq!(
            updated["identity"]["scope_sha256"],
            initial["identity"]["scope_sha256"]
        );
        assert_ne!(
            updated["identity"]["tensor_sha256"],
            initial["identity"]["tensor_sha256"]
        );
        artifact.config.learning_rate = 3.25e-6;
        fixture.manifest(&artifact);
        let configured = inspect(&fixture).expect("new config identity");
        assert_ne!(
            configured["identity"]["scope_sha256"],
            updated["identity"]["scope_sha256"]
        );
        assert_eq!(
            configured["identity"]["tensor_sha256"],
            updated["identity"]["tensor_sha256"]
        );
        assert_ppo_projection(&configured, artifact.config);
    }

    #[cfg(feature = "builtin")]
    #[test]
    fn history_rejects_symlinked_directories_and_committed_leaves_but_ignores_orphans() {
        for adaptive in [false, true] {
            let (fixture, committed_name, orphan_name) = history_fixture(adaptive);
            let directory = fixture.0.join("domain-randomization");
            let saved = fixture.0.join("history-original");
            fs::rename(&directory, &saved).unwrap();
            symlink("history-original", &directory).unwrap();
            let before = inventory(&saved);
            let result = inspect(&fixture);
            assert_eq!(inventory(&saved), before);
            fs::remove_file(&directory).unwrap();
            fs::rename(&saved, &directory).unwrap();
            assert_eq!(
                result.expect_err("history symlink"),
                CheckpointError::InvalidManifest("inspection requires non-symlink directories")
            );
            let committed = directory.join(committed_name);
            let backup = directory.join("committed-copy.json");
            fs::rename(&committed, &backup).unwrap();
            symlink("committed-copy.json", &committed).unwrap();
            let result = inspect(&fixture);
            fs::remove_file(&committed).unwrap();
            fs::rename(backup, &committed).unwrap();
            assert_eq!(
                result.expect_err("committed snapshot symlink"),
                CheckpointError::InvalidManifest("inspection requires a non-symlink regular file")
            );
            let orphan = directory.join(orphan_name);
            if fs::symlink_metadata(&orphan).is_ok() {
                fs::remove_file(&orphan).unwrap();
            }
            symlink("missing-orphan-target", &orphan).unwrap();
            let report = inspect(&fixture).expect("uncommitted symlink is outside the prefix");
            assert_eq!(report["history"]["verified"], true);
            assert_eq!(report["history"]["snapshot_count"], 1);
            assert!(!listed(
                &report,
                &format!("domain-randomization/{orphan_name}")
            ));
        }
    }

    #[cfg(feature = "builtin")]
    #[test]
    fn fixed_history_verifies_only_the_committed_prefix_and_accepts_zero_updates() {
        let mut artifact = fixture_artifact(FIXED, 0);
        let fixture = Fixture::new(&mut artifact);
        let empty = inspect(&fixture).expect("no committed snapshots");
        assert_eq!(empty["history"]["snapshot_count"], 0);
        assert!(!fixture.0.join("domain-randomization").exists());
        let directory = fixture.0.join("domain-randomization");
        let schedule = crate::randomization::AnnealSchedule {
            updates: 8,
            zero_updates: 0,
            scale: crate::randomization::AnnealScale::FULL,
        };
        for generation in 0..2 {
            let draw =
                crate::randomization::draw_generation(9001, generation, 2, 1, schedule).unwrap();
            crate::randomization::write_generation_snapshots(
                &directory,
                std::slice::from_ref(&draw),
            )
            .unwrap();
        }
        fs::write(
            directory.join("generation-000000000002.json"),
            b"uncommitted orphan",
        )
        .unwrap();
        artifact.progress.global_update = 3;
        artifact.trainer_updates = 3;
        fixture.manifest(&artifact);
        let report = inspect(&fixture).expect("two committed generations");
        assert_eq!(
            report["history"],
            json!({"kind": "fixed", "verified": true, "snapshot_count": 2})
        );
        assert_eq!(report["progress"]["games"], 3);
        assert!(listed(
            &report,
            "domain-randomization/generation-000000000001.json"
        ));
        assert!(!listed(
            &report,
            "domain-randomization/generation-000000000002.json"
        ));
        assert_inventory(&fixture, &report);
        fs::write(directory.join("generation-000000000001.json"), b"tampered").unwrap();
        let error = inspect(&fixture).expect_err("committed fixed snapshot must match");
        assert!(error.to_string().contains("snapshot"), "{error}");
    }

    #[cfg(feature = "builtin")]
    #[test]
    fn adaptive_pending_generation_ignores_orphan_but_authenticates_committed_chain() {
        let mut artifact = fixture_artifact(FIXED, 0);
        let fixture = Fixture::new(&mut artifact);
        let mut checkpoint = adaptive_checkpoint(2, 8);
        artifact
            .run
            .command_line
            .push_str(&checkpoint.config.scope_suffix());
        artifact.progress.adaptive_environment = Some(checkpoint);
        fixture.manifest(&artifact);
        let empty = inspect(&fixture).expect("empty adaptive history");
        assert_eq!(empty["history"]["snapshot_count"], 0);
        let directory = fixture.0.join("domain-randomization");
        write_adaptive_prefix(&directory, &mut checkpoint);
        assert_eq!(checkpoint.state.generation, 1);
        assert_eq!(checkpoint.state.updates_in_generation, 0);
        artifact.progress.global_update = 2;
        artifact.trainer_updates = 2;
        artifact.progress.adaptive_environment = Some(checkpoint);
        fixture.manifest(&artifact);
        let report = inspect(&fixture).expect("committed adaptive prefix");
        assert_eq!(
            report["history"],
            json!({"kind": "adaptive", "verified": true, "snapshot_count": 1})
        );
        assert_eq!(
            report["adaptive"]["snapshot_hash"],
            hex(&checkpoint.snapshot_hash)
        );
        assert_eq!(report["adaptive"]["state"]["generation"], 1);
        assert!(!listed(
            &report,
            "domain-randomization/adaptive-generation-0000000000000001.json"
        ));
        fs::write(
            directory.join("adaptive-generation-0000000000000000.json"),
            b"tampered",
        )
        .unwrap();
        let error = inspect(&fixture).expect_err("committed chain cannot be ignored");
        assert!(error.to_string().contains("snapshot"), "{error}");
    }

    #[cfg(feature = "builtin")]
    #[test]
    fn inspection_caps_history_before_attempting_any_snapshot_read() {
        let command = "train-annealed --updates 10002 --generation-updates 1 --zero-updates 0";
        for adaptive in [false, true] {
            let mut artifact = fixture_artifact(command, 10_001);
            if adaptive {
                let mut checkpoint = adaptive_checkpoint(1, 10_002);
                checkpoint.state.generation = 10_001;
                checkpoint.state.start_update = 10_001;
                checkpoint.snapshot_count = 10_001;
                checkpoint.snapshot_hash = [7; 32];
                artifact
                    .run
                    .command_line
                    .push_str(&checkpoint.config.scope_suffix());
                artifact.progress.adaptive_environment = Some(checkpoint);
            }
            let fixture = Fixture::new(&mut artifact);
            assert_eq!(
                inspect(&fixture).expect_err("cap before nonexistent history"),
                CheckpointError::InvalidManifest("inspection snapshot count exceeds 10000")
            );
        }
    }

    #[cfg(feature = "builtin")]
    #[test]
    fn fixed_scope_rejects_duplicate_required_counters() {
        let mut artifact = fixture_artifact(FIXED, 0);
        let fixture = Fixture::new(&mut artifact);
        for suffix in [
            " --updates 8",
            " --generation-updates 2",
            " --zero-updates 0",
        ] {
            artifact.run.command_line = format!("{FIXED}{suffix}");
            fixture.manifest(&artifact);
            let error = inspect(&fixture).expect_err("duplicate scope counter");
            assert!(
                error.to_string().contains("inspection scope counter"),
                "{error}"
            );
        }
    }

    #[cfg(not(feature = "builtin"))]
    #[test]
    fn annealed_history_requires_builtin_even_for_an_empty_prefix() {
        let mut artifact = fixture_artifact(FIXED, 0);
        let fixture = Fixture::new(&mut artifact);
        assert_eq!(
            inspect(&fixture).expect_err("unsupported history"),
            CheckpointError::InvalidManifest(
                "annealed history inspection requires builtin feature"
            )
        );
    }

    fn fixture_artifact(command: &str, updates: u64) -> TrainingArtifact {
        let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
        parameters[0] = -0.0;
        TrainingArtifact {
            run: CheckpointRun {
                git_commit: "inspection-fixture".to_owned(),
                simulator_commit: "inspection-simulator".to_owned(),
                enabled_features: compiled_features(),
                command_line: command.to_owned(),
                run_seed: 9001,
                map: bota_proto::MapId(2),
                hero: SHADOW_FIEND,
                device: CheckpointDevice::Cuda { ordinal: u32::MAX },
                batch_size: 2,
                rules_audit_version: PPO_RULES_AUDIT_VERSION,
            },
            progress: CheckpointProgress {
                adaptive_environment: None,
                global_update: updates,
                policy_version: 0,
                scheduler_step: 0,
                curriculum_stage: 0,
                rollout_samples: 0,
                best_evaluation: None,
                rng_states: Vec::new(),
                league_references: Vec::new(),
            },
            config: PpoConfig {
                gamma_tick: 1.0,
                samples_per_update: 2 * crate::MAP2_RETAINED_DECISIONS,
                minibatch: 2,
                epochs: 1,
                ..PpoConfig::default()
            },
            trainer_updates: updates,
            shuffle: (9001, 0),
            parameters,
            optimizer: CheckpointOptimizer {
                first_moment: vec![0.0; MODEL_PARAMETER_COUNT],
                second_moment: vec![0.0; MODEL_PARAMETER_COUNT],
                step: 0,
            },
            collection: crate::CollectionCheckpoint {
                actor: vec![0.0; MODEL_PARAMETER_COUNT],
                state: vec![1],
            },
            tensor_hash: [0; 32],
        }
    }

    struct Fixture(PathBuf);

    struct RootSwap<'a> {
        original: &'a Path,
        saved: PathBuf,
    }

    impl Drop for RootSwap<'_> {
        fn drop(&mut self) {
            if self.saved.exists() {
                // Never recursively delete an unexpected replacement tree.
                if self.original.exists() {
                    fs::remove_dir(self.original).expect("remove own empty replacement root");
                }
                fs::rename(&self.saved, self.original).expect("restore owned checkpoint root");
            }
        }
    }

    impl Fixture {
        fn new(artifact: &mut TrainingArtifact) -> Self {
            let fixture = Self(crate::ppo::test_directory("checkpoint-inspection"));
            fixture.publish(artifact);
            fixture
        }

        fn publish(&self, artifact: &mut TrainingArtifact) {
            let tensors = serialize_training_tensors(artifact).expect("native tensors");
            artifact.tensor_hash = sha256(&tensors);
            fs::write(
                tensor_generation_path(&self.0, artifact.tensor_hash),
                &tensors,
            )
            .unwrap();
            let runtime = serialize_runtime_tensor(&artifact.parameters).unwrap();
            fs::write(self.0.join(RUNTIME_TENSOR_FILE), runtime).unwrap();
            self.manifest(artifact);
        }

        fn manifest(&self, artifact: &TrainingArtifact) {
            let bytes = encode_manifest(artifact, artifact.tensor_hash).expect("native manifest");
            fs::write(self.0.join(CHECKPOINT_META_FILE), bytes).expect("own manifest");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove own inspection fixture");
        }
    }

    fn inspect(fixture: &Fixture) -> Result<Value, CheckpointError> {
        inspect_path(fixture, &fixture.0)
    }

    fn inspect_path(fixture: &Fixture, path: &Path) -> Result<Value, CheckpointError> {
        let before = inventory(&fixture.0);
        let result = checkpoint_inspect(path);
        assert_eq!(
            inventory(&fixture.0),
            before,
            "inspection must not write or create files"
        );
        result.map(|bytes| parse_json(&bytes))
    }

    fn inventory(root: &Path) -> BTreeMap<PathBuf, (u64, String)> {
        let mut files = BTreeMap::new();
        for relative in ["", "domain-randomization"] {
            let directory = root.join(relative);
            if !fs::symlink_metadata(&directory).is_ok_and(|metadata| metadata.is_dir()) {
                continue;
            }
            for (index, entry) in fs::read_dir(directory).unwrap().enumerate() {
                assert!(index < 16, "only bounded owned fixtures are inventoried");
                let path = entry.unwrap().path();
                let metadata = fs::symlink_metadata(&path).unwrap();
                let hash = if metadata.file_type().is_symlink() {
                    format!("symlink:{}", fs::read_link(&path).unwrap().display())
                } else if metadata.is_file() {
                    assert!(metadata.len() <= MAX_TRAINING_TENSOR_BYTES + 1);
                    file_hash(&path)
                } else {
                    "directory".to_owned()
                };
                files.insert(
                    path.strip_prefix(root).unwrap().to_owned(),
                    (metadata.len(), hash),
                );
            }
        }
        files
    }

    fn assert_inventory(fixture: &Fixture, report: &Value) {
        let files = report["files"].as_array().expect("file inventory");
        assert!(files.len() <= 10_004);
        let mut names = BTreeSet::new();
        for file in files {
            let name = file["path"].as_str().expect("relative path");
            assert!(!name.is_empty());
            assert!(names.insert(name), "duplicate inventory path: {name}");
            assert!(
                Path::new(name)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
            );
            let path = fixture.0.join(name);
            assert_eq!(file["size"], fs::metadata(&path).unwrap().len());
            assert_eq!(file["sha256"], file_hash(&path));
            assert_hex(&file["sha256"], 64);
        }
    }

    fn assert_current_identity(fixture: &Fixture, artifact: &TrainingArtifact, report: &Value) {
        assert_eq!(report["model"]["version"], MODEL_SCHEMA_VERSION);
        assert_eq!(report["model"]["parameters"], MODEL_PARAMETER_COUNT);
        assert_eq!(report["model"]["hash"], format!("{MODEL_SCHEMA_HASH:016x}"));
        assert_eq!(report["ppo"]["schema_version"], PPO_SCHEMA_VERSION);
        assert_eq!(
            report["ppo"]["schema_hash"],
            format!("{PPO_SCHEMA_HASH:016x}")
        );
        assert_eq!(
            report["identity"]["manifest_sha256"],
            file_hash(&fixture.0.join(CHECKPOINT_META_FILE))
        );
        assert_eq!(
            report["identity"]["tensor_sha256"],
            hex(&artifact.tensor_hash)
        );
        assert_eq!(
            report["identity"]["runtime_sha256"],
            file_hash(&fixture.0.join(RUNTIME_TENSOR_FILE))
        );
        assert_hex(&report["identity"]["scope_sha256"], 64);
        assert_ppo_projection(report, artifact.config);
    }

    fn assert_ppo_projection(report: &Value, config: PpoConfig) {
        for (name, expected) in [
            (
                "decision_interval_ticks",
                u64::from(config.decision_interval_ticks),
            ),
            ("samples_per_update", config.samples_per_update as u64),
            ("epochs", config.epochs as u64),
            ("minibatch", config.minibatch as u64),
        ] {
            assert_eq!(report["ppo"][name], expected, "PPO {name}");
        }
        for (name, expected) in [
            ("clip_epsilon", config.clip_epsilon),
            ("value_coefficient", config.value_coefficient),
            ("entropy_coefficient", config.entropy_coefficient),
            ("learning_rate", config.learning_rate),
            ("adam_beta1", config.adam_beta1),
            ("adam_beta2", config.adam_beta2),
            ("adam_epsilon", config.adam_epsilon),
            ("gradient_clip", config.gradient_clip),
            ("gamma_tick", config.gamma_tick),
            ("gae_lambda", config.gae_lambda),
            ("target_kl", config.target_kl),
        ] {
            assert_eq!(
                report["ppo"]["f32_bits"][name],
                expected.to_bits(),
                "PPO bits {name}"
            );
            let number = report["ppo"][name].as_f64().expect("numeric PPO field");
            assert_eq!(
                (number as f32).to_bits(),
                expected.to_bits(),
                "PPO number {name}"
            );
        }
    }

    fn assert_runtime_mismatch(fixture: &Fixture, path: &Path) {
        let report = inspect(fixture).expect("lagging runtime is not a restore");
        assert_eq!(report["runtime_status"], "mismatch");
        assert_eq!(report["runtime_matches_model"], false);
        assert_eq!(report["identity"]["runtime_sha256"], file_hash(path));
        assert!(!listed(&report, RUNTIME_TENSOR_FILE));
    }

    fn listed(report: &Value, path: &str) -> bool {
        report["files"]
            .as_array()
            .expect("files")
            .iter()
            .any(|file| file["path"] == path)
    }

    fn file_hash(path: &Path) -> String {
        hex(&sha256(&fs::read(path).expect("bounded owned fixture")))
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn assert_hex(value: &Value, length: usize) {
        let value = value.as_str().expect("hex string");
        assert_eq!(value.len(), length);
        assert!(
            value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[cfg(feature = "builtin")]
    fn adaptive_checkpoint(base_updates: u64, total_updates: u64) -> AdaptiveEnvironmentCheckpoint {
        AdaptiveEnvironmentCheckpoint {
            config: crate::AdaptiveEnvironmentConfig::default(),
            limits: crate::AdaptiveEnvironmentLimits {
                base_updates,
                total_updates,
                zero_updates: 0,
            },
            state: crate::AdaptiveEnvironmentState::default(),
            snapshot_count: 0,
            snapshot_hash: [0; 32],
        }
    }

    #[cfg(feature = "builtin")]
    fn history_fixture(adaptive: bool) -> (Fixture, &'static str, &'static str) {
        let mut artifact = fixture_artifact(FIXED, if adaptive { 2 } else { 1 });
        let fixture = Fixture::new(&mut artifact);
        let directory = fixture.0.join("domain-randomization");
        let names = if adaptive {
            let mut checkpoint = adaptive_checkpoint(2, 8);
            write_adaptive_prefix(&directory, &mut checkpoint);
            artifact
                .run
                .command_line
                .push_str(&checkpoint.config.scope_suffix());
            artifact.progress.adaptive_environment = Some(checkpoint);
            (
                "adaptive-generation-0000000000000000.json",
                "adaptive-generation-0000000000000001.json",
            )
        } else {
            let schedule = crate::randomization::AnnealSchedule {
                updates: 8,
                zero_updates: 0,
                scale: crate::randomization::AnnealScale::FULL,
            };
            let draw = crate::randomization::draw_generation(9001, 0, 2, 1, schedule).unwrap();
            crate::randomization::write_generation_snapshots(
                &directory,
                std::slice::from_ref(&draw),
            )
            .unwrap();
            (
                "generation-000000000000.json",
                "generation-000000000001.json",
            )
        };
        fixture.manifest(&artifact);
        (fixture, names.0, names.1)
    }

    #[cfg(feature = "builtin")]
    fn write_adaptive_prefix(directory: &Path, checkpoint: &mut AdaptiveEnvironmentCheckpoint) {
        crate::adaptive_randomization::draw_adaptive_generation(
            directory,
            9001,
            1,
            checkpoint,
            crate::randomization::AnnealScale::FULL,
        )
        .expect("own committed generation");
        for update in 1..=2 {
            checkpoint.state = checkpoint
                .state
                .observe(checkpoint.config, checkpoint.limits, update, 2, 2)
                .expect("pending next generation");
        }
        let mut orphan = *checkpoint;
        crate::adaptive_randomization::draw_adaptive_generation(
            directory,
            9001,
            1,
            &mut orphan,
            crate::randomization::AnnealScale::FULL,
        )
        .expect("own uncommitted generation");
        fs::write(
            directory.join("adaptive-generation-0000000000000001.json"),
            b"uncommitted orphan",
        )
        .expect("orphan must not be validated");
    }
}
