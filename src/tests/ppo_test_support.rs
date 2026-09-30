use super::*;

/// A private test directory that is removed with its contents when dropped,
/// so a panicking test cannot leak it.
pub(crate) struct TestDirectory(std::path::PathBuf);

impl std::ops::Deref for TestDirectory {
    type Target = std::path::Path;

    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for TestDirectory {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let result = std::fs::remove_dir_all(&self.0);
        if let Err(error) = result
            && error.kind() != std::io::ErrorKind::NotFound
            && !std::thread::panicking()
        {
            panic!("remove test directory {}: {error}", self.0.display());
        }
    }
}

pub(crate) fn test_directory(name: &str) -> TestDirectory {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-test-{name}-{}-{sequence}",
        std::process::id()
    ));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&directory)
        .expect("unique private test directory");
    TestDirectory(directory)
}

#[cfg(test)]
pub(crate) fn open_unit_bounds_for_test() -> (f64, f64) {
    (
        open_unit_from_bits(0),
        open_unit_from_bits((1u64 << 52) - 1),
    )
}

impl PpoBatch {
    #[cfg(test)]
    pub(crate) fn corrupt_materialization_frame_for_test(&mut self, index: usize) {
        self.samples[index]
            .transition
            .frame
            .corrupt_unit_offset_for_test();
    }

    #[cfg(test)]
    pub(crate) fn replace_advantage_for_test(&mut self, index: usize, value: f32) -> f32 {
        std::mem::replace(&mut self.samples[index].advantage, value)
    }
}
