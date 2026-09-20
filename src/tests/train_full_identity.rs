//! Default `train-full` identity: the canonical run scope and, on a CUDA
//! host, the recorded campaign digests.

/// The canonical default-config identity settings (E2) recorded in
/// `artifacts/temp/pipeline-groups-20260917/ab-results.jsonl`.
fn canonical_settings() -> crate::TrainingJobConfig {
    crate::cli::training_settings_for_test(&[
        "--complete-episodes",
        "--environments",
        "2",
        "--rollout",
        "1163",
        "--epochs",
        "1",
        "--minibatch",
        "512",
        "--map",
        "2",
        "--seed",
        "10141700",
        "--gamma-per-tick",
        "1",
        "--gae-lambda",
        ".95",
        "--learning-rate",
        ".00003",
        "--entropy-coefficient",
        ".001",
        "--opponent-schedule",
        "mastery-v1",
        "--mastery-window",
        "50",
        "--mastery-win-percent",
        "80",
    ])
    .expect("canonical train-full settings")
}

#[test]
fn canonical_train_full_scope_is_pinned() {
    let settings = canonical_settings();
    let run = crate::ppo_arena::training_checkpoint_run(
        &settings,
        crate::PolicyDevice::Cpu,
        settings.ppo,
    )
    .expect("canonical run scope");
    assert_eq!(
        run.command_line,
        "train-full --environments 2 --rollout 1163 --epochs 1 --minibatch 512 --seed 10141700 --map 2 --device cpu --complete-episodes --opponent-schedule mastery-v1 --mastery-window 50 --opponent-win-percent weak=80 --opponent-win-percent teacher=80"
    );
    assert_eq!(run.run_seed, 10_141_700);
    assert_eq!(run.map, bota_proto::MapId(2));
    assert_eq!(run.hero, crate::SHADOW_FIEND);
    assert_eq!(run.batch_size, 512);
    assert_eq!(run.rules_audit_version, crate::PPO_RULES_AUDIT_VERSION);
    assert!(run.mastery_config.is_some());
    assert_eq!(run.enabled_features, crate::compiled_features());
}

/// The campaign baseline file of the recorded identity slices.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn baseline_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/pipeline-groups-20260917/ab-results.jsonl")
}

/// Parses one digest out of a complete recorded identity entry.
///
/// `Err` is any parse failure once the file was read; only an absent file is a
/// skip, so a reformatted baseline cannot silently pass the replays.
fn parse_recorded_digest(text: &str, tag: &str, field: &str) -> Result<String, String> {
    let mut found = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("baseline line {} is malformed JSON: {error}", index + 1))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("baseline line {} is not an object", index + 1))?;
        let record_tag = object
            .get("tag")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("baseline line {} has no string tag", index + 1))?;
        if record_tag != tag {
            continue;
        }
        if found.is_some() {
            return Err(format!("baseline tag {tag} was duplicated"));
        }
        if object.get("exit").and_then(serde_json::Value::as_i64) != Some(0) {
            return Err(format!("baseline entry for {tag} was not successful"));
        }
        found = Some((
            parse_recorded_hash(object, tag, "weights")?,
            parse_recorded_hash(object, tag, "optimizer")?,
        ));
    }
    let (weights, optimizer) = found.ok_or_else(|| format!("baseline tag {tag} was not found"))?;
    match field {
        "weights" => Ok(weights),
        "optimizer" => Ok(optimizer),
        _ => Err(format!("baseline field {field} for {tag} is unsupported")),
    }
}

/// Parses one lowercase sha256 field from a recorded identity entry.
fn parse_recorded_hash(
    object: &serde_json::Map<String, serde_json::Value>,
    tag: &str,
    field: &str,
) -> Result<String, String> {
    let digest = object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("baseline field {field} for {tag} was not a string"))?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "baseline field {field} for {tag} was not a lowercase sha256"
        ));
    }
    Ok(digest.to_owned())
}

/// Reads one recorded digest from a chosen baseline path.
fn recorded_digest_at(
    path: &std::path::Path,
    tag: &str,
    field: &str,
) -> Result<Option<String>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("baseline read failed: {error}")),
    };
    parse_recorded_digest(&text, tag, field).map(Some)
}

/// Reads one recorded digest, distinguishing an absent baseline from a broken one.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn recorded_digest(tag: &str, field: &str) -> Result<Option<String>, String> {
    recorded_digest_at(&baseline_path(), tag, field)
}

#[test]
fn recorded_digest_skips_only_an_absent_baseline() {
    let absent =
        std::env::temp_dir().join(format!("drysua-absent-baseline-{}", std::process::id()));
    let _ = std::fs::remove_file(&absent);
    assert_eq!(recorded_digest_at(&absent, "e02g1u1", "weights"), Ok(None));
}

#[test]
fn recorded_digest_rejects_malformed_identity_records() {
    const WEIGHTS: &str = "fd8bf5bd21968263be8f413f6113b3600eb2a37de216ea97d33fbde38889ee98";
    const OPTIMIZER: &str = "0e31970182db99fafc6a5400cf00332bb27505926aa480063ba0a472c05bdce0";
    let line = format!(
        "{{\"exit\": 0, \"optimizer\": \"{OPTIMIZER}\", \"tag\": \"e02g1u1\", \"weights\": \"{WEIGHTS}\"}}"
    );
    assert_eq!(
        parse_recorded_digest(&line, "e02g1u1", "weights"),
        Ok(WEIGHTS.to_owned())
    );
    assert!(parse_recorded_digest(&line[..line.len() - 1], "e02g1u1", "weights").is_err());
    assert!(parse_recorded_digest(&line.replace(WEIGHTS, "abc"), "e02g1u1", "weights").is_err());
    assert!(
        parse_recorded_digest(
            &line.replace("\"optimizer\"", "\"missing\""),
            "e02g1u1",
            "weights"
        )
        .is_err()
    );
    assert!(parse_recorded_digest(&format!("{line}\n{line}"), "e02g1u1", "weights").is_err());
    let malformed = format!(
        "{{not-json \"optimizer\": \"{OPTIMIZER}\", \"tag\": \"e02g1u1\", \"weights\": \"{WEIGHTS}\"}}"
    );
    assert!(parse_recorded_digest(&malformed, "e02g1u1", "weights").is_err());
    let failed = line.replace("\"exit\": 0", "\"exit\": 1");
    assert!(parse_recorded_digest(&failed, "e02g1u1", "weights").is_err());
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn sha256_f32(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

/// Replays one recorded identity slice and compares its digests.
///
/// The recorded baselines were produced on a CUDA device, so the slice and the
/// digest check need one too. The test is ignored by default and reads the
/// recorded digests from the campaign artifacts.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn replay_recorded_identity(environments: usize, tag: &str) {
    use sha2::{Digest, Sha256};

    let recorded_weights = match recorded_digest(tag, "weights") {
        Ok(Some(digest)) => digest,
        Ok(None) => {
            eprintln!("recorded identity baselines are absent; nothing to replay");
            return;
        }
        Err(error) => panic!("recorded identity baseline is malformed: {error}"),
    };
    let recorded_optimizer = match recorded_digest(tag, "optimizer") {
        Ok(Some(digest)) => digest,
        Ok(None) => unreachable!("the file was present for the weights digest"),
        Err(error) => panic!("recorded identity baseline is malformed: {error}"),
    };
    let mut settings = canonical_settings();
    settings.ppo.environments = environments;
    let directory =
        std::env::temp_dir().join(format!("drysua-identity-{tag}-{}", std::process::id()));
    if directory.exists() {
        std::fs::remove_dir_all(&directory).expect("remove stale identity directory");
    }
    std::fs::create_dir(&directory).expect("create identity directory");
    let device = crate::PolicyDevice::Cuda { ordinal: 0 };
    crate::run_training_job_on_with_initial_weights(
        settings,
        device,
        &directory,
        false,
        None,
        |_| {},
    )
    .expect("recorded identity slice");
    let artifact = crate::TrainingArtifact::load(&directory).expect("identity artifact");
    let model = crate::PolicyModel::fresh_on(1, device).expect("identity model");
    let state = artifact
        .restore(&model, artifact.run())
        .expect("identity restore");
    let snapshot = state
        .trainer()
        .checkpoint_snapshot(&model)
        .expect("identity snapshot");
    let (first, second) = snapshot.adam.moments();
    assert_eq!(
        sha256_f32(&snapshot.parameters),
        recorded_weights,
        "weights digest for {tag}"
    );
    let mut hasher = Sha256::new();
    for values in [first, second, &snapshot.parameters] {
        for value in values {
            hasher.update(value.to_le_bytes());
        }
    }
    let optimizer = hasher.finalize();
    assert_eq!(
        optimizer
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        recorded_optimizer,
        "optimizer digest for {tag}"
    );
    std::fs::remove_dir_all(directory).expect("remove identity directory");
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "replays the recorded CUDA identity slice; run with --ignored"]
fn recorded_identity_slice_e2_replays_byte_for_byte() {
    replay_recorded_identity(2, "e02g1u1");
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "replays the recorded CUDA identity slice; run with --ignored"]
fn recorded_identity_slice_e8_replays_byte_for_byte() {
    replay_recorded_identity(8, "e08g1u1");
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "replays the recorded CUDA identity slice; run with --ignored"]
fn recorded_identity_slice_e16_replays_byte_for_byte() {
    replay_recorded_identity(16, "e16g1u1");
}
