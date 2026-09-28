//! Adaptive snapshots authenticate actual starts, without predicting environment ends.

#[cfg(test)]
#[path = "adaptive_randomization_tests.rs"]
mod tests;

#[cfg(not(windows))]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::checkpoint::AdaptiveEnvironmentCheckpoint;
use crate::randomization::{AnnealSchedule, GenerationDraw, VARIABLES, draw_generation_at_start};
use crate::{MAX_TRAINING_COUNTER, PpoError};

const MAX_SNAPSHOT_BYTES: u64 = 4096;
const SNAPSHOT_PREFIX: &str =
    "{\"schema\":\"drysua-domain-randomization/adaptive-v3\",\"start_update\":";
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);
const _: () = assert!(MAX_SNAPSHOT_BYTES < usize::MAX as u64);

/// Reads only the committed prefix; later orphan files are deliberately ignored.
/// Call on resume before drawing or collecting any games. No filesystem writes occur.
pub(crate) fn verify_adaptive_snapshots(
    directory: &Path,
    seed: u64,
    games_per_update: u64,
    checkpoint: &AdaptiveEnvironmentCheckpoint,
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
            schedule(checkpoint),
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

/// Materializes one immutable draw after resume verification, then commits its hash.
/// Cache the result for the generation. Only snapshot_count/hash change; collection
/// or PPO failure may leave a canonical orphan, which replay must match exactly.
/// GenerationDraw end/applied-game fields are bounds, never actual adaptive ends.
pub(crate) fn draw_adaptive_generation(
    directory: &Path,
    seed: u64,
    games_per_update: u64,
    checkpoint: &mut AdaptiveEnvironmentCheckpoint,
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
        schedule(checkpoint),
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
        || (count == generation && checkpoint.state.updates_in_generation != 0)
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

fn schedule(checkpoint: &AdaptiveEnvironmentCheckpoint) -> AnnealSchedule {
    AnnealSchedule {
        updates: checkpoint.limits.total_updates,
        zero_updates: checkpoint.limits.zero_updates,
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
    match fs::symlink_metadata(directory) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(PpoError::InvalidConfig(
            "adaptive randomization directory must be a real directory",
        )),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("directory metadata", error)),
    }
}

fn read_snapshot(path: &Path) -> Result<String, PpoError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            PpoError::InvalidConfig("adaptive randomization snapshot is missing")
        } else {
            io_error("snapshot metadata", error)
        }
    })?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot must be a regular file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options
        .open(path)
        .map_err(|error| io_error("snapshot open", error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("snapshot metadata", error))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot must be a regular file",
        ));
    }
    if metadata.len() > MAX_SNAPSHOT_BYTES {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot is oversized",
        ));
    }
    let mut bytes = Vec::with_capacity(MAX_SNAPSHOT_BYTES as usize + 1);
    file.take(MAX_SNAPSHOT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("snapshot read", error))?;
    if bytes.len() > MAX_SNAPSHOT_BYTES as usize {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot is oversized",
        ));
    }
    if bytes.len() as u64 != metadata.len() {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization snapshot length changed",
        ));
    }
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

#[cfg(not(windows))]
fn sync_directory(directory: &Path) -> Result<(), PpoError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|error| io_error("directory sync", error))
}

#[cfg(windows)]
fn sync_directory(_: &Path) -> Result<(), PpoError> {
    // Snapshot publication uses MOVEFILE_WRITE_THROUGH, like checkpoint commits.
    Ok(())
}

fn io_error(operation: &str, error: std::io::Error) -> PpoError {
    PpoError::Model(format!("adaptive randomization {operation}: {error}"))
}
