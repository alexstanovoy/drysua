use super::*;
use crate::PolicyIdentity;

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

/// Observed outcome that closes one sampled decision into a rollout transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoOutcome {
    pub stream: usize,
    pub decision: u32,
    pub ticks: u32,
    pub next_value: f32,
    pub reward: f32,
    pub terminal: bool,
}

impl PpoPolicyChoice {
    pub(crate) const fn action(&self) -> StructuredAction {
        self.action
    }

    pub(crate) const fn policy(&self) -> PolicyIdentity {
        self.policy
    }

    #[cfg(feature = "builtin")]
    pub(crate) const fn log_probability(&self) -> f32 {
        self.log_probability
    }

    pub(crate) const fn value(&self) -> f32 {
        self.value
    }

    pub(crate) fn finish(
        self,
        behaviour: u64,
        outcome: PpoOutcome,
    ) -> Result<PpoTransition, PpoError> {
        let transition = PpoTransition {
            frame: self.frame,
            target: self.target,
            shadow: None,
            action: self.action,
            behaviour,
            stream: outcome.stream,
            decision: outcome.decision,
            ticks: outcome.ticks,
            old_log_probability: self.log_probability,
            old_value: self.value,
            next_value: outcome.next_value,
            reward: outcome.reward,
            terminal: outcome.terminal,
        };
        validate_transition(&transition)?;
        Ok(transition)
    }
}

impl PpoPreparedSample {
    pub(crate) const fn return_value(&self) -> f32 {
        self.return_value
    }
}

impl PpoRng {
    pub(crate) const fn draws(&self) -> u64 {
        self.draws
    }
}

/// Sampled action and old-policy statistics returned by the model.
#[derive(Clone, Debug)]
pub struct PpoPolicyChoice {
    pub(crate) frame: FeatureFrame,
    pub(crate) target: BehavioralTarget,
    pub(crate) action: StructuredAction,
    pub(crate) policy: PolicyIdentity,
    pub(crate) log_probability: f32,
    pub(crate) entropy: f32,
    pub(crate) value: f32,
}
