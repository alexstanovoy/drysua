//! Shared harness code for the pinned-initialization and fixture tests.
//!
//! Every helper here preserves the exact checks each call site performed before
//! consolidation; only the duplication was removed.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use bota_proto::{EntityId, Order, ServerMsg};
use sha2::{Digest, Sha256};

use super::map2_checkpoint::progress;
use super::map2_model_initialization::{assert_bits, assert_fresh_state};
use crate::{CheckpointRun, PolicyModel, TrainingArtifact, Wire};

/// Git and simulator provenance variable names for one test family.
pub(crate) struct ProvenanceVariables {
    pub(crate) source: &'static str,
    pub(crate) simulator: &'static str,
}

/// Pinned initialization utilities read their frozen revisions from these variables.
pub(crate) const INITIALIZATION_PROVENANCE: ProvenanceVariables = ProvenanceVariables {
    source: "DRYSUA_INITIALIZATION_GIT_COMMIT",
    simulator: "DRYSUA_INITIALIZATION_SIMULATOR_COMMIT",
};

/// The training-job utilities read the compiled command-line embedded revisions.
pub(crate) const TRAINING_PROVENANCE: ProvenanceVariables = ProvenanceVariables {
    source: "DRYSUA_GIT_COMMIT",
    simulator: "BOTA_GIT_COMMIT",
};

/// In-memory match connection that records orders and acknowledgements.
pub(crate) struct RecordingWire {
    pub(crate) messages: VecDeque<ServerMsg>,
    pub(crate) acknowledgements: Vec<u32>,
    pub(crate) orders: Vec<(Option<EntityId>, Order)>,
}

impl Wire for RecordingWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        Ok(self.messages.pop_front())
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        self.orders.push((unit, order));
        u32::try_from(self.orders.len())
            .map_err(|_| std::io::Error::other("mock order sequence overflow"))
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        self.acknowledgements.push(tick);
        Ok(())
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Resolves a new output directory from an explicit environment variable.
pub(crate) fn new_output(variable: &str, source: &Path) -> PathBuf {
    let requested =
        PathBuf::from(std::env::var_os(variable).unwrap_or_else(|| panic!("explicit {variable}")));
    assert!(!requested.exists(), "never overwrite an output");
    let parent = requested
        .parent()
        .expect("parent")
        .canonicalize()
        .expect("existing parent");
    assert!(
        !parent.starts_with(source),
        "do not create anything inside the historical source"
    );
    let output = parent.join(requested.file_name().expect("new directory name"));
    assert!(
        fs::symlink_metadata(&output)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
        "never overwrite or follow an existing output"
    );
    output
}

/// Builds the strict run identity shared by every pinned source utility.
pub(crate) fn initialization_run(
    seed: u64,
    batch_size: usize,
    command_line: String,
    provenance: &ProvenanceVariables,
) -> CheckpointRun {
    assert!(!command_line.is_empty());
    assert_ne!(provenance.source, provenance.simulator);
    CheckpointRun {
        mastery_config: None,
        git_commit: std::env::var(provenance.source).expect("frozen source revision"),
        simulator_commit: std::env::var(provenance.simulator).expect("frozen simulator revision"),
        enabled_features: crate::compiled_features(),
        command_line,
        run_seed: seed,
        map: crate::MAP2_ID,
        hero: crate::SHADOW_FIEND,
        device: crate::CheckpointDevice::Cpu,
        batch_size,
        rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
    }
}

/// Asserts the current frozen schema versions and rule audits in one place.
pub(crate) fn assert_frozen_schema_versions() {
    assert_eq!(crate::ACTION_SCHEMA_VERSION, 5);
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 22);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 24);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 37);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 32);
}

/// Reloads runtime weights and the checkpoint and proves both restore the same bits.
pub(crate) fn verify_initialized_output(output: &Path, run: &CheckpointRun, parameters: &[f32]) {
    let runtime = PolicyModel::fresh(1).expect("new runtime target");
    TrainingArtifact::load_runtime_weights(&runtime, output)
        .expect("strict current runtime reload");
    assert_bits(
        &runtime.export_parameters().expect("runtime bits"),
        parameters,
    );
    let checkpoint = TrainingArtifact::load_compatible(output, run).expect("new checkpoint reload");
    assert_eq!(checkpoint.progress(), &progress());
    let restored = PolicyModel::fresh(2).expect("new checkpoint target");
    let state = checkpoint
        .restore(&restored, run)
        .expect("new current checkpoint restore");
    assert_fresh_state(&restored, state.trainer());
    assert_bits(
        &restored.export_parameters().expect("restored bits"),
        parameters,
    );
}
