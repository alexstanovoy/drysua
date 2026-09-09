use super::{OrderPersistence, PersistenceError, record_sent_for_ledgers};
use crate::{ActivePolicyOrder, IssuedOrder, LocalPolicyError, LocalPolicyState, StateTracker};
use bota_proto::EntityId;

/// Training-only execution role. Observer changes model-facing bookkeeping, never transport.
/// Both neural roles follow actual sent requests, not unissued labels or proof of execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PolicyOrderBookkeeping {
    #[default]
    Legacy,
    Candidate(OrderPersistence),
    Observer(OrderPersistence),
}

impl PolicyOrderBookkeeping {
    pub(crate) fn for_new_trajectory(&self) -> Self {
        match self {
            Self::Legacy => Self::Legacy,
            Self::Candidate(_) => Self::Candidate(OrderPersistence::default()),
            Self::Observer(_) => Self::Observer(OrderPersistence::default()),
        }
    }

    pub(crate) fn enable_candidate(
        &mut self,
        legacy: &OrderPersistence,
    ) -> Result<(), &'static str> {
        match self {
            Self::Candidate(_) => Ok(()),
            Self::Observer(_) => Err("neural observer cannot become a candidate"),
            Self::Legacy => {
                self.validate_start(legacy)?;
                *self = Self::Candidate(*legacy);
                Ok(())
            }
        }
    }

    pub(crate) fn enable_observer(
        &mut self,
        legacy: &OrderPersistence,
    ) -> Result<(), &'static str> {
        if matches!(self, Self::Legacy) {
            self.validate_start(legacy)?;
            *self = Self::Observer(*legacy);
        }
        // Prediction and label collection must never downgrade a candidate's execution role.
        Ok(())
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

    pub(crate) const fn is_neural(&self) -> bool {
        !matches!(self, Self::Legacy)
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
            Self::Candidate(neural) | Self::Observer(neural) => neural,
            Self::Legacy => legacy,
        }
    }

    fn neural_mut(&mut self) -> Option<&mut OrderPersistence> {
        match self {
            Self::Candidate(neural) | Self::Observer(neural) => Some(neural),
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

    pub(crate) fn clear_body(&mut self) {
        if let Some(neural) = self.neural_mut() {
            neural.clear_body();
        }
    }
}
