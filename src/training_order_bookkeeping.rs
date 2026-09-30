use super::{OrderPersistence, PersistenceError, record_sent_for_ledgers};
use crate::{ActivePolicyOrder, IssuedOrder, LocalPolicyError, LocalPolicyState, StateTracker};
use bota_proto::EntityId;

/// Training-only execution role; the candidate follows actual sent requests, not
/// unissued labels or proof of execution.
#[allow(
    clippy::large_enum_variant,
    reason = "one inline value per seat; boxing would allocate on the decision path"
)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PolicyOrderBookkeeping {
    #[default]
    Legacy,
    Candidate(OrderPersistence),
}

impl PolicyOrderBookkeeping {
    pub(crate) fn enable_candidate(
        &mut self,
        legacy: &OrderPersistence,
    ) -> Result<(), &'static str> {
        match self {
            Self::Candidate(_) => Ok(()),
            Self::Legacy => {
                self.validate_start(legacy)?;
                *self = Self::Candidate(*legacy);
                Ok(())
            }
        }
    }

    fn validate_start(&self, legacy: &OrderPersistence) -> Result<(), &'static str> {
        assert!(matches!(self, Self::Legacy));
        if legacy.last_sequence().is_some() {
            return Err("neural bookkeeping must start before the first actual request");
        }
        assert!(legacy.active_body_sequence().is_none());
        Ok(())
    }

    pub(crate) const fn is_candidate(&self) -> bool {
        matches!(self, Self::Candidate(_))
    }

    pub(crate) fn transport<'a>(&'a self, legacy: &'a OrderPersistence) -> &'a OrderPersistence {
        if self.is_candidate() {
            self.effective(legacy)
        } else {
            legacy
        }
    }

    pub(crate) fn effective<'a>(&'a self, legacy: &'a OrderPersistence) -> &'a OrderPersistence {
        match self {
            Self::Candidate(neural) => neural,
            Self::Legacy => legacy,
        }
    }

    fn neural_mut(&mut self) -> Option<&mut OrderPersistence> {
        match self {
            Self::Candidate(neural) => Some(neural),
            Self::Legacy => None,
        }
    }

    pub(crate) fn record_sent(
        &mut self,
        legacy: &mut OrderPersistence,
        sequence: u32,
        issued: IssuedOrder,
        tracker: &StateTracker,
    ) -> Result<bool, PersistenceError> {
        record_sent_for_ledgers(legacy, self.neural_mut(), sequence, issued, tracker)
    }

    pub(crate) fn reconcile(
        &mut self,
        tracker: &StateTracker,
        local: &mut LocalPolicyState,
        pending: &mut Option<(u32, Option<ActivePolicyOrder>)>,
    ) -> Result<(), LocalPolicyError> {
        if let Some(neural) = self.neural_mut() {
            neural.reconcile_neural_snapshot(tracker, local, pending)?;
        }
        Ok(())
    }

    pub(crate) fn observe_rejection(&mut self, sequence: u32) {
        if let Some(neural) = self.neural_mut() {
            neural.observe_rejection(sequence);
        }
    }

    pub(crate) fn clear_body_for(&mut self, unit: Option<EntityId>) {
        if let Some(neural) = self.neural_mut() {
            neural.clear_body_for(unit);
        }
    }
}
