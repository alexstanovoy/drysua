//! Crash-durable directory updates.

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
