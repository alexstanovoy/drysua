//! Read-only inspection anchored to owned directory descriptors, never pathname parents.
use crate::CheckpointError;
use std::path::Path;
#[cfg(any(target_os = "linux", feature = "builtin"))]
use std::path::PathBuf;

#[cfg(target_os = "linux")]
use std::{
    ffi::OsStr,
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::Component,
};

#[derive(Debug)]
pub(super) struct Directory {
    #[cfg(target_os = "linux")]
    file: File,
}

#[cfg(target_os = "linux")]
const COMPONENT: &str = "inspection path component";
#[cfg(target_os = "linux")]
const DIRECTORIES: &str = "inspection requires non-symlink directories";
#[cfg(target_os = "linux")]
const REGULAR: &str = "inspection requires a non-symlink regular file";
#[cfg(target_os = "linux")]
const CHANGED: &str = "checkpoint changed during inspection";
#[cfg(target_os = "linux")]
const SIZE: &str = "inspection file size";
#[cfg(target_os = "linux")]
const MAXIMUM_FILE_BYTES: u64 = crate::MODEL_PARAMETER_COUNT as u64 * 12 + 64 * 1024;

#[cfg(target_os = "linux")]
impl Directory {
    pub(super) fn open(path: &Path) -> Result<Self, CheckpointError> {
        Self::walk(path, false)
    }

    fn walk(path: &Path, checking: bool) -> Result<Self, CheckpointError> {
        let length = path.as_os_str().as_encoded_bytes().len();
        if length == 0 || length > 4096 || path.components().count() > 256 {
            return Err(CheckpointError::InvalidManifest(
                "inspection directory path bound",
            ));
        }
        for component in path.components() {
            match component {
                Component::Normal(name) => validate_component(name)?,
                Component::RootDir | Component::CurDir => {}
                _ => return Err(CheckpointError::InvalidManifest(COMPONENT)),
            }
        }
        let start = if path.is_absolute() {
            Path::new("/")
        } else {
            Path::new(".")
        };
        let file =
            open_file(start, true).map_err(|error| open_error(error, DIRECTORIES, checking))?;
        let mut current = Self { file };
        for component in path.components() {
            if let Component::Normal(name) = component {
                current = match current.child_component(name) {
                    Ok(Some(child)) => child,
                    Ok(None) if checking => return Err(CheckpointError::InvalidManifest(CHANGED)),
                    Ok(None) => {
                        return Err(io::Error::new(
                            io::ErrorKind::NotFound,
                            "inspection directory component not found",
                        )
                        .into());
                    }
                    Err(error) if checking => return Err(directory_changed(error)),
                    Err(error) => return Err(error),
                };
            }
        }
        Ok(current)
    }

    #[cfg(feature = "builtin")]
    pub(super) fn child(&self, name: &str) -> Result<Option<Self>, CheckpointError> {
        self.child_component(OsStr::new(name))
    }

    fn child_component(&self, name: &OsStr) -> Result<Option<Self>, CheckpointError> {
        validate_component(name)?;
        let path = self.anchored_path().join(name);
        let Some(before) = initial_metadata(&path)? else {
            return Ok(None);
        };
        if !before.file_type().is_dir() {
            return Err(CheckpointError::InvalidManifest(DIRECTORIES));
        }
        let file = open_file(&path, true).map_err(|error| open_error(error, DIRECTORIES, true))?;
        let opened = file.metadata().map_err(changed_path_error)?;
        let after = fs::symlink_metadata(&path).map_err(changed_path_error)?;
        if !opened.is_dir()
            || !after.is_dir()
            || !same_identity(&before, &opened)
            || !same_identity(&opened, &after)
        {
            return Err(CheckpointError::InvalidManifest(CHANGED));
        }
        Ok(Some(Self { file }))
    }

    pub(super) fn read(
        &self,
        name: &str,
        maximum: u64,
    ) -> Result<Option<Vec<u8>>, CheckpointError> {
        validate_component(OsStr::new(name))?;
        if maximum == 0 || maximum > MAXIMUM_FILE_BYTES {
            return Err(CheckpointError::InvalidManifest(SIZE));
        }
        let limit = maximum
            .checked_add(1)
            .ok_or(CheckpointError::InvalidManifest(SIZE))?;
        let capacity =
            usize::try_from(limit).map_err(|_| CheckpointError::InvalidManifest(SIZE))?;
        let path = self.anchored_path().join(name);
        let Some(before) = initial_metadata(&path)? else {
            return Ok(None);
        };
        if !before.file_type().is_file() {
            return Err(CheckpointError::InvalidManifest(REGULAR));
        }
        if before.len() == 0 || before.len() > maximum {
            return Err(CheckpointError::InvalidManifest(SIZE));
        }
        let file = open_file(&path, false).map_err(|error| open_error(error, REGULAR, true))?;
        let opened = file.metadata().map_err(changed_path_error)?;
        if !opened.is_file() || !same_file(&before, &opened) {
            return Err(CheckpointError::InvalidManifest(CHANGED));
        }
        let bytes = read_limited(&file, limit, capacity)?;
        let after = file.metadata().map_err(changed_path_error)?;
        let current = fs::symlink_metadata(&path).map_err(changed_path_error)?;
        if !current.is_file()
            || !same_file(&opened, &after)
            || !same_file(&after, &current)
            || bytes.len() as u64 != opened.len()
        {
            return Err(CheckpointError::InvalidManifest(CHANGED));
        }
        if bytes.is_empty() || bytes.len() as u64 > maximum {
            return Err(CheckpointError::InvalidManifest(SIZE));
        }
        Ok(Some(bytes))
    }

    /// The final dot lets existing validators inspect the directory, not the procfs symlink.
    /// Keep this directory alive until every consumer of the returned path has finished.
    pub(super) fn anchored_path(&self) -> PathBuf {
        PathBuf::from(format!("/proc/self/fd/{}/.", self.file.as_raw_fd()))
    }

    pub(super) fn check_path(&self, path: &Path) -> Result<(), CheckpointError> {
        let current = Self::walk(path, true)?;
        self.check_identity(&current)
    }

    #[cfg(feature = "builtin")]
    pub(super) fn check_child(&self, name: &str, child: &Self) -> Result<(), CheckpointError> {
        let current = self
            .child(name)
            .map_err(directory_changed)?
            .ok_or(CheckpointError::InvalidManifest(CHANGED))?;
        child.check_identity(&current)
    }

    fn check_identity(&self, current: &Self) -> Result<(), CheckpointError> {
        if !same_identity(&self.file.metadata()?, &current.file.metadata()?) {
            return Err(CheckpointError::InvalidManifest(CHANGED));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn validate_component(name: &OsStr) -> Result<(), CheckpointError> {
    let bytes = name.as_encoded_bytes();
    let mut components = Path::new(name).components();
    if bytes.is_empty()
        || bytes.len() > 255
        || bytes.contains(&0)
        || !matches!(components.next(), Some(Component::Normal(value)) if value == name)
        || components.next().is_some()
    {
        return Err(CheckpointError::InvalidManifest(COMPONENT));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_file(path: &Path, directory: bool) -> io::Result<File> {
    let flags = libc::O_NOFOLLOW | libc::O_NONBLOCK | if directory { libc::O_DIRECTORY } else { 0 };
    OpenOptions::new().read(true).custom_flags(flags).open(path)
}

#[cfg(target_os = "linux")]
fn initial_metadata(path: &Path) -> Result<Option<Metadata>, CheckpointError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "linux")]
fn open_error(error: io::Error, unsafe_kind: &'static str, checking: bool) -> CheckpointError {
    let unsafe_path = matches!(
        error.raw_os_error(),
        Some(libc::ELOOP) | Some(libc::ENOTDIR)
    );
    if checking && (unsafe_path || error.kind() == io::ErrorKind::NotFound) {
        CheckpointError::InvalidManifest(CHANGED)
    } else if unsafe_path {
        CheckpointError::InvalidManifest(unsafe_kind)
    } else {
        error.into()
    }
}

#[cfg(target_os = "linux")]
fn changed_path_error(error: io::Error) -> CheckpointError {
    open_error(error, CHANGED, true)
}

#[cfg(target_os = "linux")]
fn directory_changed(error: CheckpointError) -> CheckpointError {
    match error {
        CheckpointError::InvalidManifest(DIRECTORIES) => CheckpointError::InvalidManifest(CHANGED),
        error => error,
    }
}

#[cfg(target_os = "linux")]
fn same_identity(before: &Metadata, after: &Metadata) -> bool {
    before.dev() == after.dev() && before.ino() == after.ino()
}

#[cfg(target_os = "linux")]
fn same_file(before: &Metadata, after: &Metadata) -> bool {
    same_identity(before, after)
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

#[cfg(target_os = "linux")]
fn read_limited(file: &File, limit: u64, capacity: usize) -> Result<Vec<u8>, CheckpointError> {
    assert_eq!(capacity as u64, limit);
    assert!(limit <= MAXIMUM_FILE_BYTES + 1);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| CheckpointError::Io(error.to_string()))?;
    bytes.resize(capacity, 0);
    let mut reader = file.take(limit);
    let mut length = 0;
    while length < capacity {
        let count = reader
            .read(&mut bytes[length..])
            .map_err(changed_path_error)?;
        if count == 0 {
            break;
        }
        length += count;
    }
    bytes.truncate(length);
    Ok(bytes)
}

#[cfg(not(target_os = "linux"))]
impl Directory {
    pub(super) fn open(_: &Path) -> Result<Self, CheckpointError> {
        Err(unsupported())
    }

    #[cfg(feature = "builtin")]
    pub(super) fn child(&self, _: &str) -> Result<Option<Self>, CheckpointError> {
        Err(unsupported())
    }

    pub(super) fn read(&self, _: &str, _: u64) -> Result<Option<Vec<u8>>, CheckpointError> {
        Err(unsupported())
    }

    #[cfg(feature = "builtin")]
    pub(super) fn anchored_path(&self) -> PathBuf {
        PathBuf::new()
    }

    pub(super) fn check_path(&self, _: &Path) -> Result<(), CheckpointError> {
        Err(unsupported())
    }

    #[cfg(feature = "builtin")]
    pub(super) fn check_child(&self, _: &str, _: &Self) -> Result<(), CheckpointError> {
        Err(unsupported())
    }
}

#[cfg(not(target_os = "linux"))]
fn unsupported() -> CheckpointError {
    CheckpointError::InvalidManifest("checkpoint inspection requires Linux directory handles")
}
