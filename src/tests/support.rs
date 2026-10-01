//! Shared test harness: an in-memory match connection and bit-exact model checks.

use std::collections::VecDeque;

use bota_proto::{EntityId, Order, ServerMsg};

use crate::{PolicyModel, Wire};

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
    let (first, second) = snapshot.adam.moments().expect("fresh moments");
    for moments in [first, second] {
        assert_eq!(moments.len(), crate::MODEL_PARAMETER_COUNT);
        assert!(moments.iter().all(|value| value.to_bits() == 0));
    }
}
