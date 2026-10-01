//! Milestone runtime-weight snapshots of a long-lived training run.
//!
//! Each milestone directory `u<update>` holds only deployment weights, never
//! optimizer or progress state, so snapshots can seed opponent pools and
//! evaluation without becoming resumable checkpoints.

use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use crate::{PolicyModel, PpoError, TrainingArtifact};

/// Where and how often runtime weights are exported; excluded from the run scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeHistory {
    /// Existing directory that receives one `u<update>` directory per milestone.
    pub directory: PathBuf,
    /// Milestone spacing in completed updates; the final update is always one.
    pub every: NonZeroU64,
}

impl RuntimeHistory {
    /// Exports the weights of checkpoint `update` when the run crossed a
    /// milestone since the `previous` checkpoint, or finished.
    ///
    /// Exports happen only at checkpoints, so a milestone directory carries the
    /// checkpoint's update, the first one at or after the milestone. The
    /// snapshot is staged in a hidden sibling and renamed into place, so an
    /// existing `u<update>` directory is always complete and is never rewritten.
    pub(crate) fn export(
        &self,
        model: &PolicyModel,
        update: u64,
        previous: u64,
        total_updates: u64,
    ) -> Result<(), PpoError> {
        let every = self.every.get();
        let milestone = update / every > previous / every || update == total_updates;
        if update == 0 || !milestone {
            return Ok(());
        }
        let target = self.directory.join(format!("u{update:04}"));
        if target.is_dir() {
            return Ok(());
        }
        let staging = self.directory.join(format!(".u{update:04}.partial"));
        if staging.exists() {
            std::fs::remove_dir_all(&staging).map_err(|error| history_error(&staging, error))?;
        }
        std::fs::create_dir(&staging).map_err(|error| history_error(&staging, error))?;
        TrainingArtifact::save_runtime_weights(model, &staging)
            .map_err(|error| PpoError::Model(format!("runtime history export: {error}")))?;
        std::fs::rename(&staging, &target).map_err(|error| history_error(&target, error))?;
        crate::durability::sync_directory(&self.directory)
            .map_err(|error| history_error(&self.directory, error))?;
        crate::telemetry::log_line!(
            "level=INFO event=runtime_history_export update={update} path={}",
            target.display()
        );
        Ok(())
    }
}

fn history_error(path: &Path, error: std::io::Error) -> PpoError {
    PpoError::Model(format!("runtime history {}: {error}", path.display()))
}
