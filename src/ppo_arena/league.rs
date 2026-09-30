//! League snapshots of the learner: kept in memory, persisted only by checkpoints.
//!
//! A snapshot is taken every `--league-every` updates. The session keeps the
//! ones the next publications can draw and the ones in-flight games still play.
//! A checkpoint writes each kept snapshot once, as runtime weights under
//! `league/u<update>/`, records their fingerprints in its collection state and,
//! after the commit, deletes snapshot directories it no longer records. The
//! snapshot of the checkpoint's own update is its model and is never copied.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::lane::LeagueWeights;
use crate::model::parameter_fingerprint_of;
use crate::{PpoError, TrainingArtifact};

const LEAGUE_DIRECTORY: &str = "league";

#[derive(Clone, Debug)]
struct Snapshot {
    parameters: Arc<Vec<f32>>,
    fingerprint: u64,
}

/// Snapshots by the completed updates they were taken after.
#[derive(Debug, Default)]
pub(super) struct LeagueStore {
    snapshots: BTreeMap<u64, Snapshot>,
}

impl LeagueStore {
    pub(super) fn insert(&mut self, update: u64, parameters: Arc<Vec<f32>>) {
        let fingerprint = parameter_fingerprint_of(&parameters);
        self.snapshots.insert(
            update,
            Snapshot {
                parameters,
                fingerprint,
            },
        );
    }

    /// The weights of `updates`, every one of which must be kept.
    pub(super) fn weights(
        &self,
        updates: impl IntoIterator<Item = u64>,
    ) -> Result<LeagueWeights, PpoError> {
        let mut weights: LeagueWeights = Vec::new();
        for update in updates {
            if weights.iter().any(|(known, _)| *known == update) {
                continue;
            }
            let snapshot = self
                .snapshots
                .get(&update)
                .ok_or(PpoError::InvalidTransition("league snapshot is not kept"))?;
            weights.push((update, Arc::clone(&snapshot.parameters)));
        }
        Ok(weights)
    }

    pub(super) fn retain(&mut self, keep: impl Fn(u64) -> bool) {
        self.snapshots.retain(|update, _| keep(*update));
    }

    /// Kept snapshots a checkpoint of `completed` updates must write, with fingerprints.
    pub(super) fn manifest(&self, completed: u64) -> Vec<(u64, u64)> {
        self.snapshots
            .iter()
            .filter(|(update, _)| **update != completed)
            .map(|(update, snapshot)| (*update, snapshot.fingerprint))
            .collect()
    }

    /// Writes every manifest snapshot the checkpoint directory lacks.
    pub(super) fn persist(
        &self,
        checkpoint: &Path,
        manifest: &[(u64, u64)],
    ) -> Result<(), PpoError> {
        let root = checkpoint.join(LEAGUE_DIRECTORY);
        if !manifest.is_empty() {
            std::fs::create_dir_all(&root).map_err(|error| league_error(&root, &error))?;
        }
        for (update, _) in manifest {
            let target = snapshot_directory(checkpoint, *update);
            if target.is_dir() {
                continue;
            }
            let staging = root.join(format!(".u{update:04}.partial"));
            if staging.exists() {
                std::fs::remove_dir_all(&staging)
                    .map_err(|error| league_error(&staging, &error))?;
            }
            std::fs::create_dir(&staging).map_err(|error| league_error(&staging, &error))?;
            let snapshot = &self.snapshots[update];
            TrainingArtifact::save_runtime_parameters(&snapshot.parameters, &staging)
                .map_err(|error| PpoError::Model(format!("league snapshot {update}: {error}")))?;
            std::fs::rename(&staging, &target).map_err(|error| league_error(&target, &error))?;
            #[cfg(unix)]
            std::fs::File::open(&root)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| league_error(&root, &error))?;
        }
        Ok(())
    }

    /// Deletes snapshot directories a committed manifest no longer records.
    pub(super) fn prune(checkpoint: &Path, manifest: &[(u64, u64)]) -> Result<(), PpoError> {
        let root = checkpoint.join(LEAGUE_DIRECTORY);
        let Ok(entries) = std::fs::read_dir(&root) else {
            return Ok(());
        };
        for entry in entries {
            let path = entry.map_err(|error| league_error(&root, &error))?.path();
            let recorded = manifest
                .iter()
                .any(|(update, _)| path == snapshot_directory(checkpoint, *update));
            if !recorded {
                std::fs::remove_dir_all(&path).map_err(|error| league_error(&path, &error))?;
            }
        }
        Ok(())
    }

    /// The snapshots a checkpoint recorded, verified against their fingerprints.
    pub(super) fn restore(checkpoint: &Path, manifest: &[(u64, u64)]) -> Result<Self, PpoError> {
        let mut store = Self::default();
        for &(update, fingerprint) in manifest {
            let directory = snapshot_directory(checkpoint, update);
            let parameters = TrainingArtifact::load_runtime_parameters(&directory)
                .map_err(|error| PpoError::Model(format!("league snapshot {update}: {error}")))?;
            store.insert(update, Arc::new(parameters));
            if store.snapshots[&update].fingerprint != fingerprint {
                return Err(PpoError::InvalidConfig("league snapshot fingerprint"));
            }
        }
        Ok(store)
    }
}

fn snapshot_directory(checkpoint: &Path, update: u64) -> PathBuf {
    checkpoint
        .join(LEAGUE_DIRECTORY)
        .join(format!("u{update:04}"))
}

fn league_error(path: &Path, error: &std::io::Error) -> PpoError {
    PpoError::Model(format!("league snapshot {}: {error}", path.display()))
}
