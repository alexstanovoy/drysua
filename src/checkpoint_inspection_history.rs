use super::*;

#[cfg(feature = "builtin")]
use crate::randomization::{MAX_SNAPSHOT_BYTES, RANDOMIZATION_DIRECTORY as HISTORY_DIRECTORY};

/// Generations count updates: one virtual game per update.
const GAMES_PER_UPDATE: u64 = 1;

pub(super) struct HistoryPlan {
    pub(super) kind: &'static str,
    pub(super) games: Option<u64>,
    pub(super) snapshot_count: u64,
    updates: u64,
    generation_games: u64,
    zero_updates: u64,
    /// Scale ramp endpoints in basis points; parsed only with the builtin feature.
    #[cfg(feature = "builtin")]
    scale_start_bp: i32,
    #[cfg(feature = "builtin")]
    scale_end_bp: i32,
}

pub(super) fn plan(artifact: &TrainingArtifact) -> Result<HistoryPlan, CheckpointError> {
    let command = &artifact.run.command_line;
    let kind = match command.split(' ').next() {
        Some("train-annealed") => "train-annealed",
        _ => "other",
    };
    let mut plan = HistoryPlan {
        kind,
        games: None,
        snapshot_count: 0,
        updates: 0,
        generation_games: 0,
        zero_updates: 0,
        #[cfg(feature = "builtin")]
        scale_start_bp: 0,
        #[cfg(feature = "builtin")]
        scale_end_bp: 0,
    };
    if kind != "train-annealed" {
        return Ok(plan);
    }
    if !cfg!(feature = "builtin") {
        return Err(CheckpointError::InvalidManifest(
            "annealed history inspection requires builtin feature",
        ));
    }
    // `decode_manifest` has already validated the scope, including the scale flags.
    #[cfg(feature = "builtin")]
    {
        let (_, scale) = crate::randomization::split_scale_scope(command);
        plan.scale_start_bp = scale.start_bp;
        plan.scale_end_bp = scale.end_bp;
    }
    plan.updates = counter(command, "--updates")?;
    plan.generation_games = counter(command, "--generation-updates")?;
    plan.zero_updates = counter(command, "--zero-updates")?;
    if plan.updates == 0
        || plan.generation_games == 0
        || plan.zero_updates > plan.updates
        || artifact.progress.global_update > plan.updates
    {
        return Err(CheckpointError::InvalidManifest(
            "inspection annealed scope counters",
        ));
    }
    let games = artifact.progress.global_update * GAMES_PER_UPDATE;
    plan.games = Some(games);
    plan.snapshot_count = artifact.progress.adaptive_environment.map_or_else(
        || games.div_ceil(plan.generation_games),
        |checkpoint| checkpoint.snapshot_count,
    );
    if plan.snapshot_count > MAX_SNAPSHOTS {
        return Err(CheckpointError::InvalidManifest(
            "inspection snapshot count exceeds 10000",
        ));
    }
    Ok(plan)
}

fn counter(command: &str, flag: &str) -> Result<u64, CheckpointError> {
    assert!(command.len() <= MAX_TEXT_BYTES);
    assert!(flag.starts_with("--"));
    let mut tokens = command.split(' ');
    let mut found = None;
    for _ in 0..=MAX_TEXT_BYTES {
        let Some(token) = tokens.next() else {
            return found.ok_or(CheckpointError::InvalidManifest("inspection scope counter"));
        };
        if token
            .strip_prefix(flag)
            .is_some_and(|rest| rest.starts_with('='))
        {
            return Err(CheckpointError::InvalidManifest("inspection scope counter"));
        }
        if token != flag {
            continue;
        }
        let value = tokens
            .next()
            .ok_or(CheckpointError::InvalidManifest("inspection scope counter"))?;
        if found.is_some()
            || value.is_empty()
            || value.len() > 20
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || (value.len() > 1 && value.starts_with('0'))
        {
            return Err(CheckpointError::InvalidManifest("inspection scope counter"));
        }
        let value = value
            .parse::<u64>()
            .ok()
            .filter(|value| *value <= MAX_TRAINING_COUNTER)
            .ok_or(CheckpointError::InvalidManifest("inspection scope counter"))?;
        found = Some(value);
    }
    Err(CheckpointError::InvalidManifest("inspection scope counter"))
}

pub(super) fn verify(
    root: &Directory,
    artifact: &TrainingArtifact,
    plan: &HistoryPlan,
    files: &mut Vec<Value>,
) -> Result<Value, CheckpointError> {
    if plan.kind != "train-annealed" {
        return Ok(json!({"kind": "unsupported", "verified": false, "snapshot_count": 0}));
    }
    #[cfg(feature = "builtin")]
    {
        verify_annealed(root, artifact, plan, files)
    }
    #[cfg(not(feature = "builtin"))]
    {
        let _ = (root, artifact, files);
        Err(CheckpointError::InvalidManifest(
            "annealed history inspection requires builtin feature",
        ))
    }
}

#[cfg(feature = "builtin")]
fn verify_annealed(
    root: &Directory,
    artifact: &TrainingArtifact,
    plan: &HistoryPlan,
    files: &mut Vec<Value>,
) -> Result<Value, CheckpointError> {
    assert!(plan.snapshot_count <= MAX_SNAPSHOTS);
    let child = root.child(HISTORY_DIRECTORY)?;
    if child.is_none() && plan.snapshot_count != 0 {
        return Err(CheckpointError::InvalidManifest(
            "inspection committed snapshots are missing",
        ));
    }
    let adaptive = artifact.progress.adaptive_environment.as_ref();
    let mut hash = [0u8; 32];
    if let Some(directory) = &child {
        for generation in 0..plan.snapshot_count {
            let name = if adaptive.is_some() {
                format!("adaptive-generation-{generation:016}.json")
            } else {
                format!("generation-{generation:012}.json")
            };
            let bytes = directory.read(&name, MAX_SNAPSHOT_BYTES)?.ok_or(
                CheckpointError::InvalidManifest("inspection committed snapshot is missing"),
            )?;
            if adaptive.is_none() {
                verify_fixed(&bytes, artifact.run.run_seed, generation, plan)?;
            } else {
                let mut next = Sha256::new();
                next.update(hash);
                next.update((bytes.len() as u64).to_le_bytes());
                next.update(&bytes);
                hash = next.finalize().into();
            }
            add_file(files, &format!("{HISTORY_DIRECTORY}/{name}"), &bytes)?;
        }
    }
    if let Some(checkpoint) = adaptive {
        if hash != checkpoint.snapshot_hash {
            return Err(CheckpointError::InvalidManifest(
                "inspection adaptive snapshot hash",
            ));
        }
        let directory = child.as_ref().map_or_else(
            || root.anchored_path().join(HISTORY_DIRECTORY),
            Directory::anchored_path,
        );
        crate::adaptive_randomization::verify_adaptive_snapshots(
            &directory,
            artifact.run.run_seed,
            GAMES_PER_UPDATE,
            checkpoint,
            crate::randomization::AnnealScale {
                start_bp: plan.scale_start_bp,
                end_bp: plan.scale_end_bp,
            },
        )
        .map_err(|error| CheckpointError::Io(format!("snapshot verification: {error}")))?;
    }
    if let Some(directory) = &child {
        root.check_child(HISTORY_DIRECTORY, directory)?;
    }
    Ok(
        json!({"kind": if adaptive.is_some() { "adaptive" } else { "fixed" }, "verified": true,
        "snapshot_count": plan.snapshot_count}),
    )
}

#[cfg(feature = "builtin")]
fn verify_fixed(
    bytes: &[u8],
    seed: u64,
    generation: u64,
    plan: &HistoryPlan,
) -> Result<(), CheckpointError> {
    let schedule = crate::randomization::AnnealSchedule {
        updates: plan.updates,
        zero_updates: plan.zero_updates,
        scale: crate::randomization::AnnealScale {
            start_bp: plan.scale_start_bp,
            end_bp: plan.scale_end_bp,
        },
    };
    let draw = crate::randomization::draw_generation(
        seed,
        generation,
        plan.generation_games,
        GAMES_PER_UPDATE,
        schedule,
    )
    .map_err(|error| CheckpointError::Io(format!("snapshot draw verification: {error}")))?;
    if bytes != crate::randomization::generation_json(&draw).as_bytes() {
        return Err(CheckpointError::InvalidManifest(
            "inspection fixed snapshot mismatch",
        ));
    }
    Ok(())
}
