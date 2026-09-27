use super::*;

#[cfg(test)]
pub(crate) struct PpoOrderContractProbe(ArenaSeatPolicy);

#[cfg(test)]
impl PpoOrderContractProbe {
    pub(crate) fn new(side: usize, messages: &[ServerMsg]) -> Self {
        Self(setup_seat(side, messages).expect("PPO order-contract seat"))
    }

    pub(crate) fn observe(&mut self, messages: &[ServerMsg]) {
        assert!(
            observe_messages(&mut self.0, messages)
                .expect("PPO probe observations")
                .is_none()
        );
    }

    pub(crate) fn decide(
        &mut self,
        model: &PolicyModel,
        candidate: bool,
    ) -> (FeatureFrame, Option<Request>, Option<ActivePolicyOrder>) {
        let (frame, space) = if candidate {
            prepare_neural_seat_policy_sample(&mut self.0)
        } else {
            prepare_seat_policy_sample(&mut self.0)
        }
        .expect("PPO probe input");
        let action = model
            .choose(&frame, &space)
            .expect("PPO probe choice")
            .action;
        let request = if candidate {
            // Only the learner's request transport is tested; these placeholder
            // statistics never enter a rollout, reward calculation or optimizer.
            let choice = PpoPolicyChoice {
                frame: frame.clone(),
                action,
                target: crate::BehavioralTarget::from_action(&frame, &space, action)
                    .expect("probe target"),
                policy: model.policy_identity().expect("probe policy"),
                log_probability: 0.0,
                entropy: 0.0,
                value: 0.0,
            };
            policy_request_in_space(&mut self.0, &choice, &space).expect("PPO learner transport")
        } else {
            neural_policy_request_in_space(&mut self.0, action, &space)
                .expect("legacy probe request")
                .1
        };
        (frame, request, self.0.local.active_order())
    }

    pub(crate) fn legacy_persistence(&self) -> OrderPersistence {
        self.0.persistence
    }
}

#[cfg(test)]
pub(super) fn observe_messages(
    seat: &mut ArenaSeatPolicy,
    messages: &[ServerMsg],
) -> Result<Option<Team>, PpoError> {
    observe_messages_owned(seat, messages.to_vec())
}
