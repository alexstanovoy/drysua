//! Per-generation randomization snapshots for the adaptive schedule, committed to the checkpoint
//! by count and rolling SHA-256. A snapshot records the generation's actual start update; its end
//! is decided later by the controller, so end and applied-game fields are only bounds.

#[cfg(test)]
#[path = "adaptive_randomization_tests.rs"]
mod tests;

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::checkpoint::AdaptiveEnvironmentCheckpoint;
use crate::durability::RegularFileError;
use crate::randomization::MAX_SNAPSHOT_BYTES;
use crate::randomization::{AnnealSchedule, GenerationDraw, VARIABLES, draw_generation_at_start};
use crate::{MAX_TRAINING_COUNTER, PpoError};

const SNAPSHOT_PREFIX: &str =
    "{\"schema\":\"drysua-domain-randomization/adaptive-v3\",\"start_update\":";
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);
const _: () = assert!(MAX_SNAPSHOT_BYTES < usize::MAX as u64);

/// Verifies the first `snapshot_count` snapshots against the checkpoint; orphan files beyond that
/// prefix are ignored. Call on resume before drawing or collecting any games. Performs no writes.
pub(crate) fn verify_adaptive_snapshots(
    directory: &Path,
    seed: u64,
    games_per_update: u64,
    checkpoint: &AdaptiveEnvironmentCheckpoint,
    scale: crate::randomization::AnnealScale,
) -> Result<(), PpoError> {
    validate_checkpoint(checkpoint, games_per_update)?;
    if !validate_directory(directory)? && checkpoint.snapshot_count != 0 {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot is missing",
        ));
    }
    let mut hash = [0; 32];
    let mut previous_start = None;
    for generation in 0..checkpoint.snapshot_count {
        let stored = read_snapshot(&generation_path(directory, generation))?;
        let start_update = snapshot_start(&stored)?;
        if previous_start.map_or(start_update != 0, |previous| start_update <= previous) {
            return Err(PpoError::InvalidConfig(
                "adaptive randomization snapshot starts are not strictly increasing from zero",
            ));
        }
        let draw = draw_generation_at_start(
            seed,
            generation,
            start_update,
            games_per_update,
            schedule(checkpoint, scale),
        )?;
        compare_snapshot(&stored, &adaptive_generation_json(&draw))?;
        hash = append_hash(hash, stored.as_bytes());
        previous_start = Some(start_update);
    }
    if checkpoint.snapshot_count > checkpoint.state.generation {
        if previous_start != Some(checkpoint.state.start_update) {
            return Err(PpoError::InvalidConfig(
                "adaptive randomization active start mismatch",
            ));
        }
    } else if previous_start.is_some_and(|start| start >= checkpoint.state.start_update) {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization pending start must follow snapshots",
        ));
    }
    if hash != checkpoint.snapshot_hash {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot hash mismatch",
        ));
    }
    debug_assert!(checkpoint.snapshot_count <= MAX_TRAINING_COUNTER);
    debug_assert_eq!(previous_start.is_none(), checkpoint.snapshot_count == 0);
    Ok(())
}

/// Draws the current generation and, if it is not yet committed, publishes its snapshot and
/// advances `snapshot_count` and `snapshot_hash` (the only checkpoint fields changed). Callers cache
/// the draw per generation. A later collection or PPO failure may leave an orphan snapshot, which a
/// replay must reproduce byte for byte.
pub(crate) fn draw_adaptive_generation(
    directory: &Path,
    seed: u64,
    games_per_update: u64,
    checkpoint: &mut AdaptiveEnvironmentCheckpoint,
    scale: crate::randomization::AnnealScale,
) -> Result<GenerationDraw, PpoError> {
    validate_checkpoint(checkpoint, games_per_update)?;
    let directory_exists = validate_directory(directory)?;
    if !directory_exists && checkpoint.snapshot_count != 0 {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot is missing",
        ));
    }
    let draw = draw_generation_at_start(
        seed,
        checkpoint.state.generation,
        checkpoint.state.start_update,
        games_per_update,
        schedule(checkpoint, scale),
    )?;
    let path = generation_path(directory, draw.generation);
    let canonical = adaptive_generation_json(&draw);
    if checkpoint.snapshot_count > draw.generation {
        compare_snapshot(&read_snapshot(&path)?, &canonical)?;
        return Ok(draw);
    }
    let next_count = checkpoint
        .snapshot_count
        .checked_add(1)
        .ok_or(PpoError::CounterOverflow)?;
    let next_hash = append_hash(checkpoint.snapshot_hash, canonical.as_bytes());
    write_snapshot(directory, &path, &canonical, directory_exists)?;
    checkpoint.snapshot_count = next_count;
    checkpoint.snapshot_hash = next_hash;
    debug_assert_eq!(checkpoint.snapshot_count, draw.generation + 1);
    debug_assert_eq!(checkpoint.state.start_update, draw.start_update);
    Ok(draw)
}

fn validate_checkpoint(
    checkpoint: &AdaptiveEnvironmentCheckpoint,
    games_per_update: u64,
) -> Result<(), PpoError> {
    if !(1..=MAX_TRAINING_COUNTER).contains(&games_per_update) {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization games per update must be in 1..=MAX_TRAINING_COUNTER",
        ));
    }
    let global_update = checkpoint
        .state
        .start_update
        .checked_add(checkpoint.state.updates_in_generation)
        .ok_or(PpoError::CounterOverflow)?;
    checkpoint
        .state
        .validate(checkpoint.config, checkpoint.limits, global_update)?;
    checkpoint
        .limits
        .total_updates
        .checked_mul(games_per_update)
        .ok_or(PpoError::CounterOverflow)?;
    let generation = checkpoint.state.generation;
    let count = checkpoint.snapshot_count;
    if count > MAX_TRAINING_COUNTER
        || (count != generation && count != generation + 1)
        || (count == generation
            && checkpoint.state.updates_in_generation != 0
            && checkpoint.current_generation_collected())
    {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot count is invalid",
        ));
    }
    if (count == 0) != (checkpoint.snapshot_hash == [0; 32]) {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot hash mismatch",
        ));
    }
    debug_assert!(generation < checkpoint.limits.total_updates);
    debug_assert!(count <= checkpoint.limits.total_updates);
    Ok(())
}

fn schedule(
    checkpoint: &AdaptiveEnvironmentCheckpoint,
    scale: crate::randomization::AnnealScale,
) -> AnnealSchedule {
    AnnealSchedule {
        updates: checkpoint.limits.total_updates,
        zero_updates: checkpoint.limits.zero_updates,
        scale,
    }
}

fn generation_path(directory: &Path, generation: u64) -> PathBuf {
    directory.join(format!("adaptive-generation-{generation:016}.json"))
}

fn snapshot_start(stored: &str) -> Result<u64, PpoError> {
    let start = stored
        .strip_prefix(SNAPSHOT_PREFIX)
        .and_then(|rest| rest.split_once(','))
        .map(|(start, _)| start)
        .filter(|start| !start.is_empty() && start.len() <= 20)
        .filter(|start| start.bytes().all(|byte| byte.is_ascii_digit()))
        .filter(|start| start.len() == 1 || !start.starts_with('0'))
        .and_then(|start| start.parse::<u64>().ok())
        .filter(|start| *start < MAX_TRAINING_COUNTER)
        .ok_or(PpoError::InvalidConfig(
            "adaptive randomization snapshot start is invalid",
        ))?;
    debug_assert!(stored.len() <= MAX_SNAPSHOT_BYTES as usize);
    debug_assert!(start < MAX_TRAINING_COUNTER);
    Ok(start)
}

fn adaptive_generation_json(draw: &GenerationDraw) -> String {
    let mut body = format!(
        "{SNAPSHOT_PREFIX}{},\"generation\":{},\"start_game\":{},\"end_game_bound\":{},\"scale_bp\":{},\"applied_games_bound\":{},\"deltas\":{{",
        draw.start_update,
        draw.generation,
        draw.start_game,
        draw.end_game,
        draw.scale_bp,
        draw.applied_games,
    );
    for (index, variable) in VARIABLES.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        body.push('"');
        body.push_str(variable.name);
        body.push_str("\":");
        body.push_str(&draw.deltas[index].to_string());
    }
    body.push_str("},\"spec\":{");
    for (index, variable) in VARIABLES.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        body.push('"');
        body.push_str(variable.name);
        body.push_str("\":");
        body.push_str(&(variable.nominal + draw.deltas[index]).to_string());
    }
    body.push_str("}}\n");
    assert!(body.len() <= MAX_SNAPSHOT_BYTES as usize);
    assert!(draw.spec.is_bounded());
    body
}

fn append_hash(previous: [u8; 32], canonical: &[u8]) -> [u8; 32] {
    assert!(!canonical.is_empty());
    assert!(canonical.len() <= MAX_SNAPSHOT_BYTES as usize);
    let mut hash = Sha256::new();
    hash.update(previous);
    hash.update((canonical.len() as u64).to_le_bytes());
    hash.update(canonical);
    hash.finalize().into()
}

fn compare_snapshot(stored: &str, canonical: &str) -> Result<(), PpoError> {
    if stored != canonical {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot mismatch",
        ));
    }
    Ok(())
}

fn validate_directory(directory: &Path) -> Result<bool, PpoError> {
    crate::durability::real_directory_exists(directory).map_err(|error| match error {
        RegularFileError::Io(error) => io_error("directory metadata", error),
        _ => PpoError::InvalidConfig("adaptive randomization directory must be a real directory"),
    })
}

fn read_snapshot(path: &Path) -> Result<String, PpoError> {
    let bytes = crate::durability::read_regular_file(path, MAX_SNAPSHOT_BYTES).map_err(
        |error| match error {
            RegularFileError::Missing => {
                PpoError::InvalidConfig("adaptive randomization snapshot is missing")
            }
            RegularFileError::NotRegular => {
                PpoError::InvalidConfig("adaptive randomization snapshot must be a regular file")
            }
            RegularFileError::Oversized => {
                PpoError::InvalidConfig("adaptive randomization snapshot is oversized")
            }
            RegularFileError::Changed => {
                PpoError::InvalidConfig("adaptive randomization snapshot length changed")
            }
            RegularFileError::Io(error) => io_error("snapshot read", error),
        },
    )?;
    String::from_utf8(bytes)
        .map_err(|_| PpoError::InvalidConfig("adaptive randomization snapshot mismatch"))
}

fn write_snapshot(
    directory: &Path,
    path: &Path,
    canonical: &str,
    directory_exists: bool,
) -> Result<(), PpoError> {
    match fs::symlink_metadata(path) {
        Ok(_) => {
            compare_snapshot(&read_snapshot(path)?, canonical)?;
            // A prior process may have stopped between publication and directory sync.
            return sync_snapshot_directory(directory);
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("snapshot metadata", error)),
    }
    if !directory_exists {
        fs::create_dir(directory).map_err(|error| io_error("directory create", error))?;
    }
    let result = write_new_snapshot(directory, path, canonical);
    if result.is_err() && !directory_exists {
        fs::remove_dir(directory).map_err(|error| io_error("directory rollback", error))?;
    }
    result
}

fn write_new_snapshot(directory: &Path, path: &Path, canonical: &str) -> Result<(), PpoError> {
    let sequence = NEXT_TEMPORARY
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |sequence| {
            sequence.checked_add(1)
        })
        .map_err(|_| PpoError::CounterOverflow)?;
    let temporary = path.with_extension(format!("json.tmp-{}-{sequence}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| io_error("snapshot create", error))?;
    let mut published = false;
    let result = (|| {
        file.write_all(canonical.as_bytes())
            .map_err(|error| io_error("snapshot write", error))?;
        file.sync_all()
            .map_err(|error| io_error("snapshot sync", error))?;
        drop(file);
        // Publication must not replace a racing writer or a pre-existing orphan.
        match publish_snapshot(&temporary, path) {
            Ok(()) => {
                published = true;
                Ok(())
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                compare_snapshot(&read_snapshot(path)?, canonical)
            }
            Err(error) => Err(io_error("snapshot publish", error)),
        }
    })();
    let cleanup = remove_snapshot(&temporary);
    let result = result
        .and(cleanup)
        .and_then(|()| sync_snapshot_directory(directory));
    if result.is_err() && published {
        remove_snapshot(path)?;
        sync_directory(directory)?;
    }
    result
}

fn remove_snapshot(path: &Path) -> Result<(), PpoError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error("snapshot rollback or temporary cleanup", error)),
    }
}

fn sync_snapshot_directory(directory: &Path) -> Result<(), PpoError> {
    sync_directory(directory)?;
    // The new snapshot directory's entry must also survive a crash.
    let parent = directory
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    sync_directory(parent)
}

#[cfg(not(windows))]
fn publish_snapshot(source: &Path, target: &Path) -> std::io::Result<()> {
    fs::hard_link(source, target)
}

#[cfg(windows)]
fn publish_snapshot(source: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW};

    let source: Vec<_> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let target: Vec<_> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: Both paths are NUL-terminated UTF-16 and live through the call.
    if unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn sync_directory(directory: &Path) -> Result<(), PpoError> {
    crate::durability::sync_directory(directory).map_err(|error| io_error("directory sync", error))
}

fn io_error(operation: &str, error: std::io::Error) -> PpoError {
    PpoError::Model(format!("adaptive randomization {operation}: {error}"))
}
