use super::{OrderPersistence, PersistenceError, record_sent_for_ledgers};
use crate::{ActivePolicyOrder, IssuedOrder, LocalPolicyError, LocalPolicyState, StateTracker};
use bota_proto::EntityId;

/// Training-only order ledger selection. `SentRequests` uses the ledger of sent requests
/// alone; `Neural` adds a neural ledger, copied from it before the first request and updated
/// from requests actually sent, not from unissued action labels or confirmed execution.
#[allow(
    clippy::large_enum_variant,
    reason = "one inline value per seat; boxing would allocate on the decision path"
)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PolicyOrderBookkeeping {
    #[default]
    SentRequests,
    Neural(OrderPersistence),
}

impl PolicyOrderBookkeeping {
    pub(crate) fn enable_neural(
        &mut self,
        requests: &OrderPersistence,
    ) -> Result<(), &'static str> {
        match self {
            Self::Neural(_) => Ok(()),
            Self::SentRequests => {
                self.validate_start(requests)?;
                *self = Self::Neural(*requests);
                Ok(())
            }
        }
    }

    fn validate_start(&self, requests: &OrderPersistence) -> Result<(), &'static str> {
        assert!(matches!(self, Self::SentRequests));
        if requests.last_sequence().is_some() {
            return Err("neural bookkeeping must start before the first actual request");
        }
        assert!(requests.active_body_sequence().is_none());
        Ok(())
    }

    pub(crate) fn effective<'a>(&'a self, requests: &'a OrderPersistence) -> &'a OrderPersistence {
        match self {
            Self::Neural(neural) => neural,
            Self::SentRequests => requests,
        }
    }

    fn neural_mut(&mut self) -> Option<&mut OrderPersistence> {
        match self {
            Self::Neural(neural) => Some(neural),
            Self::SentRequests => None,
        }
    }

    pub(crate) fn record_sent(
        &mut self,
        requests: &mut OrderPersistence,
        sequence: u32,
        issued: IssuedOrder,
        tracker: &StateTracker,
    ) -> Result<bool, PersistenceError> {
        record_sent_for_ledgers(requests, self.neural_mut(), sequence, issued, tracker)
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
