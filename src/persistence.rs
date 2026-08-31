use std::error::Error;
use std::fmt;

use bota_proto::{EntityId, Order};

use crate::IssuedOrder;

/// Sequence-link or persistence state error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceError {
    SequenceNotIncreasing { incoming: u32, previous: u32 },
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceNotIncreasing { incoming, previous } => write!(
                formatter,
                "order sequence {incoming} must be greater than last sent sequence {previous}"
            ),
        }
    }
}

impl Error for PersistenceError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SentBodyOrder {
    sequence: u32,
    issued: IssuedOrder,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BodyRollback {
    sequence: u32,
    previous: Option<SentBodyOrder>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BodyOrderState {
    current: Option<SentBodyOrder>,
    rollback: Option<BodyRollback>,
}

/// Constant-size suppression state for persistent body orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrderPersistence {
    bodies: [BodyOrderState; 2],
    last_sequence: Option<u32>,
}

impl OrderPersistence {
    /// Returns no order for Continue and suppresses an exact active body repeat.
    pub fn should_send(&self, issued: Option<IssuedOrder>) -> Option<IssuedOrder> {
        let issued = issued?;
        if is_persistent_body_order(issued.order)
            && self
                .body_state(issued.unit)
                .current
                .is_some_and(|active| active.issued == issued)
        {
            return None;
        }
        Some(issued)
    }

    /// Records one order after it was sent with a strictly increasing sequence.
    ///
    /// One-shot body work clears persistence without becoming suppressible.
    pub fn record_sent(
        &mut self,
        sequence: u32,
        issued: IssuedOrder,
    ) -> Result<(), PersistenceError> {
        if let Some(previous) = self.last_sequence
            && sequence <= previous
        {
            return Err(PersistenceError::SequenceNotIncreasing {
                incoming: sequence,
                previous,
            });
        }
        self.last_sequence = Some(sequence);
        if is_persistent_body_order(issued.order) {
            self.body_state_mut(issued.unit).transition(
                sequence,
                issued.unit,
                Some(SentBodyOrder { sequence, issued }),
            );
        } else if interrupts_persistent_body(issued.order) {
            self.body_state_mut(issued.unit)
                .transition(sequence, issued.unit, None);
        }
        Ok(())
    }

    /// Restores the preceding order only when a body's newest order was rejected.
    pub fn observe_rejection(&mut self, sequence: u32) -> bool {
        self.bodies.iter_mut().any(|body| body.reject(sequence))
    }

    /// Clears both body orders after external observation proves they are no longer active.
    pub fn clear_body(&mut self) {
        self.bodies = [BodyOrderState::default(); 2];
    }

    /// Greatest sequence among the currently suppressed body orders.
    pub const fn active_body_sequence(&self) -> Option<u32> {
        let hero = match self.bodies[0].current {
            Some(active) => Some(active.sequence),
            None => None,
        };
        let courier = match self.bodies[1].current {
            Some(active) => Some(active.sequence),
            None => None,
        };
        match (hero, courier) {
            (Some(hero), Some(courier)) => Some(if hero > courier { hero } else { courier }),
            (Some(hero), None) => Some(hero),
            (None, Some(courier)) => Some(courier),
            (None, None) => None,
        }
    }

    /// Sequence of the currently suppressed order for one controlled body.
    pub fn active_body_sequence_for(&self, unit: Option<EntityId>) -> Option<u32> {
        self.active_body_for(unit).map(|(sequence, _)| sequence)
    }

    /// Currently suppressed order for one controlled body.
    pub fn active_body_order_for(&self, unit: Option<EntityId>) -> Option<IssuedOrder> {
        self.active_body_for(unit).map(|(_, issued)| issued)
    }

    /// Sequence and order currently suppressed for one controlled body.
    pub fn active_body_for(&self, unit: Option<EntityId>) -> Option<(u32, IssuedOrder)> {
        self.body_state(unit)
            .current
            .filter(|active| active.issued.unit == unit)
            .map(|active| (active.sequence, active.issued))
    }

    /// Last sequence recorded as sent, including non-body orders.
    pub const fn last_sequence(&self) -> Option<u32> {
        self.last_sequence
    }

    fn body_state(&self, unit: Option<EntityId>) -> &BodyOrderState {
        &self.bodies[body_index(unit)]
    }

    fn body_state_mut(&mut self, unit: Option<EntityId>) -> &mut BodyOrderState {
        &mut self.bodies[body_index(unit)]
    }
}

impl BodyOrderState {
    fn transition(
        &mut self,
        sequence: u32,
        unit: Option<EntityId>,
        current: Option<SentBodyOrder>,
    ) {
        if self
            .current
            .is_some_and(|active| active.issued.unit != unit)
        {
            *self = Self::default();
        }
        self.rollback = Some(BodyRollback {
            sequence,
            previous: self.current,
        });
        self.current = current;
    }

    fn reject(&mut self, sequence: u32) -> bool {
        let Some(rollback) = self
            .rollback
            .filter(|rollback| rollback.sequence == sequence)
        else {
            return false;
        };
        self.current = rollback.previous;
        self.rollback = None;
        true
    }
}

const fn is_persistent_body_order(order: Order) -> bool {
    matches!(order, Order::Move { .. } | Order::Attack { .. })
}

const fn interrupts_persistent_body(order: Order) -> bool {
    matches!(
        order,
        Order::Cast { .. } | Order::Use { .. } | Order::Put { .. } | Order::Take { .. }
    )
}

const fn body_index(unit: Option<EntityId>) -> usize {
    if unit.is_some() { 1 } else { 0 }
}
