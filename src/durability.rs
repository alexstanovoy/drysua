//! Crash-durable directory updates and symlink-refusing bounded reads.

use std::path::Path;

/// Flushes a directory's entries, so a created, renamed or removed child survives a crash.
/// Windows cannot open directories; its commits rename with `MOVEFILE_WRITE_THROUGH` instead.
pub(crate) fn sync_directory(directory: &Path) -> std::io::Result<()> {
    #[cfg(not(windows))]
    std::fs::File::open(directory)?.sync_all()?;
    #[cfg(windows)]
    let _ = directory;
    Ok(())
}

#[cfg(any(feature = "builtin", test))]
/// Why a bounded read of a regular file was refused.
#[derive(Debug)]
pub(crate) enum RegularFileError {
    /// The path does not exist.
    Missing,
    /// The path is a symlink, directory or other non-regular entry.
    NotRegular,
    /// The file is larger than the bound.
    Oversized,
    /// The file changed length while it was read.
    Changed,
    /// Any other I/O failure.
    Io(std::io::Error),
}

#[cfg(any(feature = "builtin", test))]
/// Reads a regular file of at most `limit` bytes without following a symlink at
/// its last component. The opened handle is checked again, so a link swapped in
/// between the check and the open is refused too.
pub(crate) fn read_regular_file(path: &Path, limit: u64) -> Result<Vec<u8>, RegularFileError> {
    use std::io::Read;

    let metadata = std::fs::symlink_metadata(path).map_err(missing_or_io)?;
    if !metadata.is_file() {
        return Err(RegularFileError::NotRegular);
    }
    let mut options = std::fs::OpenOptions::new();
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
    let file = options.open(path).map_err(RegularFileError::Io)?;
    let metadata = file.metadata().map_err(RegularFileError::Io)?;
    if !metadata.is_file() {
        return Err(RegularFileError::NotRegular);
    }
    if metadata.len() > limit {
        return Err(RegularFileError::Oversized);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(RegularFileError::Io)?;
    if bytes.len() as u64 > limit {
        return Err(RegularFileError::Oversized);
    }
    if bytes.len() as u64 != metadata.len() {
        return Err(RegularFileError::Changed);
    }
    Ok(bytes)
}

#[cfg(any(feature = "builtin", test))]
/// Whether `directory` exists as a real directory; a symlink or any other entry
/// there is `NotRegular`.
pub(crate) fn real_directory_exists(directory: &Path) -> Result<bool, RegularFileError> {
    match std::fs::symlink_metadata(directory) {
        Ok(metadata) if metadata.is_dir() => Ok(true),
        Ok(_) => Err(RegularFileError::NotRegular),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(RegularFileError::Io(error)),
    }
}

#[cfg(any(feature = "builtin", test))]
/// Whether anything, a dangling symlink included, occupies `path`.
pub(crate) fn entry_exists(path: &Path) -> std::io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(any(feature = "builtin", test))]
fn missing_or_io(error: std::io::Error) -> RegularFileError {
    if error.kind() == std::io::ErrorKind::NotFound {
        RegularFileError::Missing
    } else {
        RegularFileError::Io(error)
    }
}
