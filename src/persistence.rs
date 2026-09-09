use std::error::Error;
use std::fmt;

#[cfg(feature = "builtin")]
#[path = "training_order_bookkeeping.rs"]
pub(crate) mod training;

use bota_proto::{AbilityId, Aim, EntityId, Order, StatusFlags, Target, UnitView};

use crate::{
    ActionKind, ActivePolicyOrder, ActivePolicyTarget, IssuedOrder, LocalPolicyError,
    LocalPolicyState, SHADOW_FIEND, StateTracker,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActiveOrderUpdate {
    Preserve,
    Replace(Option<ActionKind>),
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
        self.record_sent_with_interruption(
            sequence,
            issued,
            interrupts_persistent_body(issued.order),
        )
    }

    /// Records a candidate request; true preserves hero active state and pending rollback.
    pub(crate) fn record_neural_sent(
        &mut self,
        sequence: u32,
        issued: IssuedOrder,
        tracker: &StateTracker,
    ) -> Result<bool, PersistenceError> {
        let preserves = preserves_neural_body(tracker, issued);
        let interrupts = interrupts_persistent_body(issued.order) && !preserves;
        self.record_sent_with_interruption(sequence, issued, interrupts)?;
        Ok(preserves)
    }

    fn record_sent_with_interruption(
        &mut self,
        sequence: u32,
        issued: IssuedOrder,
        interrupts: bool,
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
        } else if interrupts {
            self.body_state_mut(issued.unit)
                .transition(sequence, issued.unit, None);
        }
        Ok(())
    }

    /// Reconciles candidate directives and bounded rollback after a complete snapshot/event tick.
    pub(crate) fn reconcile_neural_snapshot(
        &mut self,
        tracker: &StateTracker,
        local: &mut LocalPolicyState,
        pending: &mut Option<(u32, Option<ActivePolicyOrder>)>,
    ) -> Result<(), LocalPolicyError> {
        let tick = tracker
            .current()
            .expect("candidate reconciliation requires a snapshot")
            .tick;
        let mut bodies = self.bodies;
        assert_eq!(bodies.len(), 2);
        for body in &mut bodies {
            body.current = reconcile_sent_body(body.current, tracker);
            if let Some(rollback) = &mut body.rollback {
                rollback.previous = reconcile_sent_body(rollback.previous, tracker);
            }
        }
        let active = reconcile_active_order(local.active_order(), tracker);
        let previous =
            pending.map(|(sequence, active)| (sequence, reconcile_active_order(active, tracker)));
        if active != local.active_order() {
            local.restore_active_order(tick, active)?;
        }
        self.bodies = bodies;
        *pending = previous;
        assert_eq!(local.active_order(), active);
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

    /// Clears one controlled body's order after external lifecycle evidence invalidates it.
    pub fn clear_body_for(&mut self, unit: Option<EntityId>) {
        let index = body_index(unit);
        assert!(index < self.bodies.len());
        self.bodies[index] = BodyOrderState::default();
        assert!(self.active_body_sequence_for(unit).is_none());
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

/// Records both ledgers; true preserves the candidate hero's active state and rollback.
pub(crate) fn record_sent_for_policy(
    legacy: &mut OrderPersistence,
    neural: &mut Option<OrderPersistence>,
    sequence: u32,
    issued: IssuedOrder,
    tracker: &StateTracker,
) -> Result<bool, PersistenceError> {
    record_sent_for_ledgers(legacy, neural.as_mut(), sequence, issued, tracker)
}

fn record_sent_for_ledgers(
    legacy: &mut OrderPersistence,
    neural: Option<&mut OrderPersistence>,
    sequence: u32,
    issued: IssuedOrder,
    tracker: &StateTracker,
) -> Result<bool, PersistenceError> {
    let preserves = if let Some(neural) = neural {
        assert_eq!(legacy.last_sequence(), neural.last_sequence());
        neural.record_neural_sent(sequence, issued, tracker)?
    } else {
        false
    };
    legacy.record_sent(sequence, issued)?;
    assert_eq!(legacy.last_sequence(), Some(sequence));
    Ok(preserves)
}

fn preserves_neural_body(tracker: &StateTracker, issued: IssuedOrder) -> bool {
    if issued.unit.is_some() {
        return false;
    }
    let Order::Cast {
        slot,
        target: Target::None,
    } = issued.order
    else {
        return false;
    };
    let Some(hero) = tracker
        .own_hero()
        .filter(|hero| hero.hero == Some(SHADOW_FIEND))
    else {
        return false;
    };
    hero.abilities
        .get(usize::from(slot.0))
        .is_some_and(|ability| {
            ability.aim == Aim::Own
                && !ability.passive
                && matches!(
                    ability.id,
                    AbilityId(13) | AbilityId(14) | AbilityId(15) | AbilityId(16)
                )
        })
}

fn reconcile_sent_body(
    active: Option<SentBodyOrder>,
    tracker: &StateTracker,
) -> Option<SentBodyOrder> {
    let mut active = active?;
    assert!(is_persistent_body_order(active.issued.order));
    active.issued = reconcile_body_directive(active.issued, tracker)?;
    assert!(is_persistent_body_order(active.issued.order));
    Some(active)
}

fn reconcile_body_directive(
    mut issued: IssuedOrder,
    tracker: &StateTracker,
) -> Option<IssuedOrder> {
    if !controlled_body_alive(tracker, issued.unit) {
        return None;
    }
    let target = match issued.order {
        Order::Attack {
            target: Target::Unit(target),
        }
        | Order::Move {
            target: Target::Unit(target),
        } => target,
        _ => return Some(issued),
    };
    // Candidate truncation cannot prove a visibility loss. The tracker owns the
    // complete validated seat snapshot, sorted by full-generation handle.
    let current = tracker.current().expect("candidate snapshot");
    if current
        .units
        .binary_search_by_key(&target, |unit| unit.id)
        .ok()
        .is_some_and(|index| unit_alive(&current.units[index]))
    {
        return Some(issued);
    }
    let observed = tracker.entity(target)?;
    // Gaps and same-tick deaths can hide a newer server last-seen position.
    if observed.last_seen_tick.checked_add(1) != Some(current.tick)
        || observed
            .last_death
            .is_some_and(|death| death.tick >= observed.last_seen_tick)
    {
        return None;
    }
    let position = observed.unit.pos;
    issued.order = match issued.order {
        Order::Attack { .. } => Order::Attack {
            target: Target::Pos(position),
        },
        Order::Move { .. } => Order::Move {
            target: Target::Pos(position),
        },
        _ => unreachable!("checked unit-target directive"),
    };
    Some(issued)
}

fn reconcile_active_order(
    active: Option<ActivePolicyOrder>,
    tracker: &StateTracker,
) -> Option<ActivePolicyOrder> {
    let mut active = active?;
    if !controlled_body_alive(tracker, None) {
        return None;
    }
    let target = match active.target {
        ActivePolicyTarget::Unit(target) => Target::Unit(target),
        _ => return Some(active),
    };
    let order = match active.kind {
        ActionKind::AttackUnit => Order::Attack { target },
        ActionKind::FollowUnit => Order::Move { target },
        _ => return None,
    };
    let corrected = reconcile_body_directive(IssuedOrder { unit: None, order }, tracker)?;
    if corrected.order == order {
        return Some(active);
    }
    let (kind, position) = match corrected.order {
        Order::Attack {
            target: Target::Pos(position),
        } => (ActionKind::AttackMovePoint, position),
        Order::Move {
            target: Target::Pos(position),
        } => (ActionKind::MovePoint, position),
        _ => unreachable!("reconciled point directive"),
    };
    active.kind = kind;
    active.target = ActivePolicyTarget::Point(position);
    active.started_tick = tracker.current().expect("candidate snapshot").tick;
    Some(active)
}

fn controlled_body_alive(tracker: &StateTracker, unit: Option<EntityId>) -> bool {
    match unit {
        None => tracker.own_hero().is_some_and(unit_alive),
        Some(id) => tracker
            .own_courier()
            .is_some_and(|courier| courier.id == id && unit_alive(courier)),
    }
}

fn unit_alive(unit: &UnitView) -> bool {
    unit.hp > 0 && unit.statuses.bits & StatusFlags::DEAD == 0
}

pub(crate) fn active_order_update_for_sent(
    persistence: &OrderPersistence,
    unit: Option<EntityId>,
    sent_sequence: u32,
    sent_kind: ActionKind,
) -> ActiveOrderUpdate {
    assert_eq!(persistence.last_sequence(), Some(sent_sequence));
    if unit.is_some() {
        return ActiveOrderUpdate::Preserve;
    }
    let Some(active_sequence) = persistence.active_body_sequence_for(unit) else {
        assert!(persistence.active_body_order_for(unit).is_none());
        return ActiveOrderUpdate::Replace(None);
    };
    assert!(active_sequence <= sent_sequence);
    if active_sequence == sent_sequence {
        return ActiveOrderUpdate::Replace(Some(sent_kind));
    }
    ActiveOrderUpdate::Preserve
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
