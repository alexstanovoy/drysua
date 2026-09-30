use super::*;

pub(crate) fn test_directory(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-learning-{name}-{}-{sequence}",
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
    directory
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
    pub(crate) fn reject_minibatch_for_test(&mut self, index: usize) {
        self.samples[index].transition.old_log_probability = -5.0;
    }

    #[cfg(test)]
    pub(crate) fn replace_advantage_for_test(&mut self, index: usize, value: f32) -> f32 {
        std::mem::replace(&mut self.samples[index].advantage, value)
    }
}
