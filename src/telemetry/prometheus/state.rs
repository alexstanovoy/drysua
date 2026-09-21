use std::fs::{self, File, Metadata, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
#[cfg(any(feature = "builtin", test))]
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::snapshot::{DurationHistogram, TrainingSnapshot};

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

const LOCK_FILE: &str = ".metrics.writer.lock";
const STATE_FILE: &str = "metrics.state";
const PENDING_FILE: &str = "metrics.pending";
const STATE_TEMP: &str = ".metrics.state.tmp";
const PENDING_TEMP: &str = ".metrics.pending.tmp";
const MAGIC: &[u8; 8] = b"DRYMET01";
const VERSION: u32 = 1;
const MAX_STATE_BYTES: usize = 16 * 1024;
const PAYLOAD_BYTES: usize = 826;
const RECORD_BYTES: usize = PAYLOAD_BYTES + 32;
const _: () = assert!(RECORD_BYTES <= MAX_STATE_BYTES);
const _: () = assert!(PAYLOAD_BYTES == 12 + 64 + 32 + 64 + 24 + 13 + 33 + 6 * 96 + 8);

/// One externally serialized writer in an owned private directory, separate from checkpoints.
/// The parent supplies checkpoint identities; this journal never opens checkpoint files.
pub(super) struct MetricsStore {
    directory: PathBuf,
    _writer: File,
}

impl MetricsStore {
    #[cfg(any(feature = "builtin", test))]
    pub(super) fn open(directory: &Path) -> io::Result<Self> {
        validate_directory(directory)?;
        require_atomic_replacement()?;
        let writer = open_lock(directory, true)?;
        // Exporter liveness probes briefly hold this lock; retry only during startup.
        lock_writer_with_retry(|| writer.try_lock(), std::thread::sleep)?;
        verify_open_file(&directory.join(LOCK_FILE), &writer)?;
        for name in [STATE_FILE, PENDING_FILE, STATE_TEMP, PENDING_TEMP] {
            regular_metadata(&directory.join(name))?;
        }
        for name in [STATE_TEMP, PENDING_TEMP] {
            remove_regular(directory, name)?;
        }
        sync_directory(directory)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            _writer: writer,
        })
    }

    /// Establishes explicit first coverage, or reconciles the actual resumed checkpoint.
    /// Missing both records cannot be distinguished from first opt-in: coverage resets.
    /// Returned target/heartbeat are committed values; the parent may prepare new values.
    #[cfg(any(feature = "builtin", test))]
    pub(super) fn restore(
        &self,
        baseline: TrainingSnapshot,
        resume: bool,
    ) -> io::Result<TrainingSnapshot> {
        baseline.validate()?;
        let state = read_optional(&self.directory, STATE_FILE)?;
        let pending = read_optional(&self.directory, PENDING_FILE)?;
        let Some(state) = state else {
            if pending.is_some() {
                return Err(invalid(
                    "metrics committed state is missing while pending exists",
                ));
            }
            validate_first_coverage(&baseline)?;
            self.replace(STATE_FILE, STATE_TEMP, &baseline)?;
            return Ok(baseline);
        };
        if !resume {
            return Err(invalid("metrics state already exists for a fresh run"));
        }
        same_scope(&state, &baseline)?;
        if state.completed_updates > baseline.completed_updates {
            return Err(invalid(
                "metrics committed state is ahead of the checkpoint",
            ));
        }
        if let Some(pending) = pending {
            validate_transition(&state, &pending)?;
            if same_checkpoint(&pending, &baseline) {
                return self.publish(&pending);
            }
            if same_checkpoint(&state, &baseline) {
                remove_regular(&self.directory, PENDING_FILE)?;
                sync_directory(&self.directory)?;
                return Ok(state);
            }
        }
        if state.completed_updates == baseline.completed_updates
            && !same_checkpoint(&state, &baseline)
        {
            return Err(invalid(
                "metrics committed checkpoint identity or progress does not match",
            ));
        }
        if !same_checkpoint(&state, &baseline) {
            return Err(invalid(
                "metrics checkpoint is neither committed nor pending",
            ));
        }
        Ok(state)
    }

    /// Must finish successfully before the parent commits the corresponding checkpoint.
    /// Resolve any pending record before preparing a different candidate.
    /// Same-update target/heartbeat changes are allowed. Only empty first coverage may
    /// rebind a checkpoint identity, allowing opt-in migration without rewriting outcomes.
    pub(super) fn prepare(&self, snapshot: &TrainingSnapshot) -> io::Result<()> {
        let state = self.committed()?;
        validate_transition(&state, snapshot)?;
        if let Some(pending) = read_optional(&self.directory, PENDING_FILE)? {
            validate_transition(&state, &pending)?;
            if pending != *snapshot {
                return Err(invalid(
                    "metrics pending snapshot must be resolved before another prepare",
                ));
            }
        }
        self.replace(PENDING_FILE, PENDING_TEMP, snapshot)
    }

    /// Called only after the parent's exact checkpoint manifest is durable.
    /// An I/O error may leave a visible replacement: reopen/reconcile, never assume rollback.
    pub(super) fn commit(&self, update: u64, checkpoint: [u8; 32]) -> io::Result<TrainingSnapshot> {
        let state = self.committed()?;
        let Some(pending) = read_optional(&self.directory, PENDING_FILE)? else {
            if state.completed_updates != update || state.checkpoint != checkpoint {
                return Err(invalid("metrics commit does not match a prepared snapshot"));
            }
            sync_directory(&self.directory)?;
            return Ok(state);
        };
        validate_transition(&state, &pending)?;
        if pending.completed_updates != update || pending.checkpoint != checkpoint {
            return Err(invalid("metrics commit does not match a prepared snapshot"));
        }
        self.publish(&pending)
    }

    fn committed(&self) -> io::Result<TrainingSnapshot> {
        read_optional(&self.directory, STATE_FILE)?
            .ok_or_else(|| invalid("metrics committed state is missing"))
    }

    fn publish(&self, snapshot: &TrainingSnapshot) -> io::Result<TrainingSnapshot> {
        self.replace(STATE_FILE, STATE_TEMP, snapshot)?;
        remove_regular(&self.directory, PENDING_FILE)?;
        sync_directory(&self.directory)?;
        Ok(snapshot.clone())
    }

    fn replace(&self, name: &str, temporary: &str, snapshot: &TrainingSnapshot) -> io::Result<()> {
        assert!(matches!(
            (name, temporary),
            (STATE_FILE, STATE_TEMP) | (PENDING_FILE, PENDING_TEMP)
        ));
        let bytes = encode_snapshot(snapshot)?;
        assert_eq!(bytes.len(), RECORD_BYTES);
        validate_directory(&self.directory)?;
        let target = self.directory.join(name);
        let temporary = self.directory.join(temporary);
        regular_metadata(&target)?;
        regular_metadata(&temporary)?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        private_creation(&mut options);
        // Fixed crash remnants are deliberately left for locked recovery in open().
        let mut file = options.open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        verify_open_file(&temporary, &file)?;
        regular_metadata(&target)?;
        fs::rename(&temporary, &target)?;
        sync_directory(&self.directory)
    }
}

/// Exporters never consume pending metrics, even if the checkpoint might have committed.
pub(super) fn read_snapshot(directory: &Path) -> io::Result<TrainingSnapshot> {
    read_optional(directory, STATE_FILE)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "metrics committed state is missing",
        )
    })
}

/// Reports a fully validated pending transaction without publishing its metrics.
pub(super) fn has_pending(directory: &Path) -> io::Result<bool> {
    read_optional(directory, PENDING_FILE).map(|snapshot| snapshot.is_some())
}

/// Observes the existing advisory lock without creating files or updating timestamps.
pub(super) fn writer_active(directory: &Path) -> io::Result<bool> {
    validate_directory(directory)?;
    let file = open_lock(directory, false)?;
    match file.try_lock() {
        Ok(()) => {
            verify_open_file(&directory.join(LOCK_FILE), &file)?;
            file.unlock()?;
            Ok(false)
        }
        Err(TryLockError::WouldBlock) => Ok(true),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

#[cfg(any(feature = "builtin", test))]
fn validate_first_coverage(snapshot: &TrainingSnapshot) -> io::Result<()> {
    if !has_empty_coverage(snapshot) {
        return Err(invalid(
            "metrics first coverage must start at the checkpoint with zero outcomes and durations",
        ));
    }
    Ok(())
}

fn has_empty_coverage(snapshot: &TrainingSnapshot) -> bool {
    snapshot.start_update == snapshot.completed_updates
        && snapshot.games == [0; 4]
        && snapshot.last_update_games == [0; 4]
        && snapshot.durations == [DurationHistogram::default(); 6]
}

fn same_scope(base: &TrainingSnapshot, next: &TrainingSnapshot) -> io::Result<()> {
    if base.scope != next.scope {
        return Err(invalid("metrics scope does not match the run"));
    }
    Ok(())
}

#[cfg(any(feature = "builtin", test))]
fn same_checkpoint(left: &TrainingSnapshot, right: &TrainingSnapshot) -> bool {
    left.completed_updates == right.completed_updates
        && left.checkpoint == right.checkpoint
        && left.samples == right.samples
        && left.optimizer_steps == right.optimizer_steps
}

fn validate_transition(base: &TrainingSnapshot, next: &TrainingSnapshot) -> io::Result<()> {
    base.validate()?;
    next.validate()?;
    same_scope(base, next)?;
    if base.start_update != next.start_update
        || base.parallel != next.parallel
        || base.games_per_update != next.games_per_update
    {
        return Err(invalid(
            "metrics coverage or configuration changed within the run",
        ));
    }
    if counters_regress(base, next) {
        return Err(invalid("metrics cumulative counters must not regress"));
    }
    if base.completed_updates == next.completed_updates {
        let mut comparable = next.clone();
        comparable.updates_target = base.updates_target;
        comparable.heartbeat = base.heartbeat;
        if has_empty_coverage(base) {
            comparable.checkpoint = base.checkpoint;
        }
        if comparable != *base {
            return Err(invalid(
                "metrics same-update snapshot changes committed metrics",
            ));
        }
    } else if base.checkpoint == next.checkpoint {
        return Err(invalid(
            "metrics advancing update requires a new checkpoint identity",
        ));
    }
    assert!(next.completed_updates >= base.completed_updates);
    assert_eq!(next.start_update, base.start_update);
    Ok(())
}

fn counters_regress(base: &TrainingSnapshot, next: &TrainingSnapshot) -> bool {
    base.completed_updates > next.completed_updates
        || base.samples > next.samples
        || base.optimizer_steps > next.optimizer_steps
        || base
            .games
            .iter()
            .zip(next.games)
            .any(|(before, after)| *before > after)
        || base.generation > next.generation
        || base
            .durations
            .iter()
            .zip(&next.durations)
            .any(|(before, after)| {
                before.count > after.count
                    || before.sum_seconds > after.sum_seconds
                    || before
                        .buckets
                        .iter()
                        .zip(after.buckets)
                        .any(|(left, right)| *left > right)
            })
}

fn validate_directory(directory: &Path) -> io::Result<()> {
    // A trailing slash or dot must not make lstat follow the final symlink.
    let normalized: PathBuf = directory.components().collect();
    match fs::symlink_metadata(normalized) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(invalid(
            "metrics directory must be an existing non-symlink directory",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Err(invalid(
            "metrics directory must be an existing non-symlink directory",
        )),
        Err(error) => Err(error),
    }
}

fn regular_metadata(path: &Path) -> io::Result<Option<Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            Ok(Some(metadata))
        }
        Ok(_) => Err(invalid_path(path)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(any(feature = "builtin", test))]
fn lock_writer_with_retry(
    mut try_lock: impl FnMut() -> Result<(), TryLockError>,
    mut pause: impl FnMut(Duration),
) -> io::Result<()> {
    for attempt in 0..4 {
        match try_lock() {
            Ok(()) => return Ok(()),
            Err(TryLockError::WouldBlock) => {
                if attempt < 3 {
                    pause(Duration::from_millis(10));
                }
            }
            Err(TryLockError::Error(error)) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::WouldBlock,
        "metrics writer is already active",
    ))
}

fn open_lock(directory: &Path, create: bool) -> io::Result<File> {
    let path = directory.join(LOCK_FILE);
    regular_metadata(&path)?;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .truncate(false);
    private_creation(&mut options);
    let file = options.open(&path)?;
    verify_open_file(&path, &file)?;
    Ok(file)
}

fn verify_open_file(path: &Path, file: &File) -> io::Result<()> {
    let actual = file.metadata()?;
    let expected =
        regular_metadata(path)?.ok_or_else(|| invalid("metrics file disappeared while opening"))?;
    if !actual.is_file() || !same_file(&actual, &expected) {
        return Err(invalid("metrics file changed while opening"));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(_left: &Metadata, _right: &Metadata) -> bool {
    true
}

fn private_creation(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = options;
}

fn require_atomic_replacement() -> io::Result<()> {
    if cfg!(unix) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "metrics atomic replacement and directory sync require Unix",
        ))
    }
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    require_atomic_replacement()?;
    validate_directory(directory)?;
    File::open(directory)?.sync_all()
}

fn remove_regular(directory: &Path, name: &str) -> io::Result<()> {
    validate_directory(directory)?;
    let path = directory.join(name);
    if regular_metadata(&path)?.is_some() {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn read_optional(directory: &Path, name: &str) -> io::Result<Option<TrainingSnapshot>> {
    read_optional_with_open(directory, name, |path| File::open(path))
}

fn read_optional_with_open(
    directory: &Path,
    name: &str,
    mut open: impl FnMut(&Path) -> io::Result<File>,
) -> io::Result<Option<TrainingSnapshot>> {
    assert!(matches!(name, STATE_FILE | PENDING_FILE));
    validate_directory(directory)?;
    let path = directory.join(name);
    // Atomic replacement and pending removal can race an exporter open.
    // Retry only these bounded identity races.
    for _ in 0..3 {
        let Some(expected) = regular_metadata(&path)? else {
            return Ok(None);
        };
        let file = match open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        let actual = file.metadata()?;
        if !actual.is_file() {
            return Err(invalid_path(&path));
        }
        if !same_file(&expected, &actual) {
            continue;
        }
        if actual.len() > MAX_STATE_BYTES as u64 {
            return Err(invalid("metrics snapshot exceeds 16384 bytes"));
        }
        let mut bytes = [0; MAX_STATE_BYTES];
        let mut reader = file.take(MAX_STATE_BYTES as u64);
        let mut length = 0;
        for _ in 0..MAX_STATE_BYTES {
            let read = reader.read(&mut bytes[length..])?;
            if read == 0 {
                break;
            }
            length += read;
        }
        assert!(length <= MAX_STATE_BYTES);
        if reader.get_ref().metadata()?.len() > MAX_STATE_BYTES as u64 {
            return Err(invalid("metrics snapshot exceeds 16384 bytes"));
        }
        return decode_snapshot(&bytes[..length]).map(Some);
    }
    Err(invalid("metrics file changed while opening"))
}

// Version one has fixed-width little-endian fields in TrainingSnapshot order.
// The generation/scale pair shares one presence byte; absent payloads are all zero.
// Losses have one presence byte; floats are IEEE-754 with canonical positive zero.
// SHA-256 covers all 826 payload bytes, including magic and version.
fn encode_snapshot(snapshot: &TrainingSnapshot) -> io::Result<Vec<u8>> {
    snapshot.validate()?;
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&snapshot.scope);
    bytes.extend_from_slice(&snapshot.checkpoint);
    for value in [
        snapshot.completed_updates,
        snapshot.updates_target,
        snapshot.samples,
        snapshot.optimizer_steps,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in snapshot.games.into_iter().chain(snapshot.last_update_games) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [
        snapshot.start_update,
        snapshot.parallel,
        snapshot.games_per_update,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.push(u8::from(snapshot.generation.is_some()));
    bytes.extend_from_slice(&snapshot.generation.unwrap_or(0).to_le_bytes());
    bytes.extend_from_slice(&snapshot.scale_bp.unwrap_or(0).to_le_bytes());
    bytes.push(u8::from(snapshot.losses.is_some()));
    for value in snapshot.losses.unwrap_or([0.0; 4]) {
        bytes.extend_from_slice(&canonical_float(value));
    }
    for histogram in snapshot.durations {
        for value in histogram.buckets {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&histogram.count.to_le_bytes());
        bytes.extend_from_slice(&canonical_float(histogram.sum_seconds));
    }
    bytes.extend_from_slice(&snapshot.heartbeat.to_le_bytes());
    assert_eq!(bytes.len(), PAYLOAD_BYTES);
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    assert_eq!(bytes.len(), RECORD_BYTES);
    Ok(bytes)
}

fn canonical_float(value: f64) -> [u8; 8] {
    assert!(value.is_finite());
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    assert_ne!(bits, 1_u64 << 63);
    bits.to_le_bytes()
}

fn decode_snapshot(bytes: &[u8]) -> io::Result<TrainingSnapshot> {
    validate_encoding(bytes)?;
    let mut reader = Decoder {
        remaining: &bytes[12..PAYLOAD_BYTES],
    };
    let mut snapshot = TrainingSnapshot {
        scope: reader.take()?,
        checkpoint: reader.take()?,
        completed_updates: reader.integer()?,
        updates_target: reader.integer()?,
        samples: reader.integer()?,
        optimizer_steps: reader.integer()?,
        games: reader.counters()?,
        last_update_games: reader.counters()?,
        start_update: reader.integer()?,
        parallel: reader.integer()?,
        games_per_update: reader.integer()?,
        ..TrainingSnapshot::default()
    };
    (snapshot.generation, snapshot.scale_bp) = reader.generation_scale()?;
    snapshot.losses = reader.losses()?;
    for histogram in &mut snapshot.durations {
        *histogram = DurationHistogram {
            buckets: reader.counters()?,
            count: reader.integer()?,
            sum_seconds: reader.float()?,
        };
    }
    snapshot.heartbeat = reader.integer()?;
    assert!(reader.remaining.is_empty());
    snapshot.validate()?;
    assert_eq!(bytes.len(), RECORD_BYTES);
    Ok(snapshot)
}

fn validate_encoding(bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_STATE_BYTES {
        return Err(invalid("metrics snapshot exceeds 16384 bytes"));
    }
    if bytes.len() != RECORD_BYTES {
        return Err(invalid("metrics snapshot has invalid length"));
    }
    if &bytes[..8] != MAGIC || bytes[8..12] != VERSION.to_le_bytes() {
        return Err(invalid("metrics snapshot format or version is unsupported"));
    }
    let checksum = Sha256::digest(&bytes[..PAYLOAD_BYTES]);
    if bytes[PAYLOAD_BYTES..] != checksum[..] {
        return Err(invalid("metrics snapshot checksum mismatch"));
    }
    Ok(())
}

struct Decoder<'a> {
    remaining: &'a [u8],
}

impl Decoder<'_> {
    fn take<const LENGTH: usize>(&mut self) -> io::Result<[u8; LENGTH]> {
        let Some((value, remaining)) = self.remaining.split_at_checked(LENGTH) else {
            return Err(invalid("metrics snapshot has invalid length"));
        };
        let mut bytes = [0; LENGTH];
        bytes.copy_from_slice(value);
        self.remaining = remaining;
        Ok(bytes)
    }

    fn integer(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn counters<const LENGTH: usize>(&mut self) -> io::Result<[u64; LENGTH]> {
        assert!(LENGTH <= 10);
        let mut values = [0; LENGTH];
        for value in &mut values {
            *value = self.integer()?;
        }
        Ok(values)
    }

    fn float(&mut self) -> io::Result<f64> {
        let bits = self.integer()?;
        if bits == (1_u64 << 63) {
            return Err(invalid(
                "metrics snapshot floating-point zero is not canonical",
            ));
        }
        Ok(f64::from_bits(bits))
    }

    fn generation_scale(&mut self) -> io::Result<(Option<u64>, Option<u32>)> {
        let tag = self.take::<1>()?[0];
        let generation = self.integer()?;
        let scale = u32::from_le_bytes(self.take()?);
        match tag {
            0 if generation == 0 && scale == 0 => Ok((None, None)),
            1 => Ok((Some(generation), Some(scale))),
            _ => Err(invalid("metrics snapshot optional value is not canonical")),
        }
    }

    fn losses(&mut self) -> io::Result<Option<[f64; 4]>> {
        let tag = self.take::<1>()?[0];
        let mut values = [0.0; 4];
        for value in &mut values {
            *value = self.float()?;
        }
        match tag {
            0 if values.iter().all(|value| value.to_bits() == 0) => Ok(None),
            1 => Ok(Some(values)),
            _ => Err(invalid("metrics snapshot optional value is not canonical")),
        }
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn invalid_path(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "metrics path must be a non-symlink regular file: {}",
            path.file_name().unwrap_or_default().to_string_lossy()
        ),
    )
}
