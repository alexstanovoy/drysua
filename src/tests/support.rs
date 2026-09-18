//! Shared harness code for the pinned-initialization and fixture tests.
//!
//! Every helper here preserves the exact checks each call site performed before
//! consolidation; only the duplication was removed.

use std::collections::VecDeque;

use bota_proto::{EntityId, Order, ServerMsg};

use crate::{CheckpointRun, PolicyModel, Wire};

/// Git and simulator provenance variable names for one test family.
pub(crate) struct ProvenanceVariables {
    pub(crate) source: &'static str,
    pub(crate) simulator: &'static str,
}

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

/// Bit-exact F32 comparison used by pinned-initializer reload checks.
pub(crate) fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(left, right)| left.to_bits() == right.to_bits())
    );
}

/// Fresh-optimizer state check used by pinned-initializer reload checks.
pub(crate) fn assert_fresh_state(model: &PolicyModel, trainer: &crate::PpoTrainer) {
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.rng_checkpoint().1, 0);
    let snapshot = trainer
        .checkpoint_snapshot(model)
        .expect("bound fresh optimizer");
    for moments in [snapshot.adam.moments().0, snapshot.adam.moments().1] {
        assert_eq!(moments.len(), crate::MODEL_PARAMETER_COUNT);
        assert!(moments.iter().all(|value| value.to_bits() == 0));
    }
}
