use super::{
    ActionSpace, FeatureFrame, OpponentRuntime, PolicyModel, PpoError, PpoPolicyChoice, PpoRng,
    TrainingEnvironment, prepare_neural_seat_policy_sample, text_error,
};
use std::sync::Arc;

pub(super) struct PreparedOpponent {
    pub(super) model: Arc<PolicyModel>,
    pub(super) frame: FeatureFrame,
    pub(super) space: ActionSpace,
    pub(super) random: PpoRng,
}

pub(super) struct OpponentChoice {
    pub(super) model: Arc<PolicyModel>,
    pub(super) choice: PpoPolicyChoice,
    pub(super) space: ActionSpace,
    pub(super) before: PpoRng,
    pub(super) after: PpoRng,
}

pub(super) struct OpponentBatch {
    model: Arc<PolicyModel>,
    policy: crate::PolicyIdentity,
}

pub(super) fn prepare(environment: &mut TrainingEnvironment) -> Result<PreparedOpponent, PpoError> {
    if environment.seats.len() != 2 || environment.policy_seat > 1 {
        return Err(PpoError::InvalidTransition(
            "prepared opponent requires two valid seats",
        ));
    }
    let OpponentRuntime::Policy { model, rng } = &environment.opponent else {
        return Err(PpoError::InvalidTransition(
            "prepared opponent requires neural policy",
        ));
    };
    let (frame, space) =
        prepare_neural_seat_policy_sample(&mut environment.seats[1 - environment.policy_seat])?;
    Ok(PreparedOpponent {
        model: Arc::clone(model),
        frame,
        space,
        random: rng.clone(),
    })
}

impl OpponentBatch {
    pub(super) fn new(
        learner: &PolicyModel,
        environments: &[TrainingEnvironment],
    ) -> Result<Self, PpoError> {
        validate_count(environments.len())?;
        let OpponentRuntime::Policy { model, .. } = &environments[0].opponent else {
            return Err(PpoError::InvalidTransition(
                "opponent batch requires neural policy",
            ));
        };
        let policy = model.policy_identity().map_err(text_error)?;
        if policy.lineage() == learner.policy_identity().map_err(text_error)?.lineage() {
            return Err(PpoError::InvalidTransition(
                "opponent batch shares learner lineage",
            ));
        }
        for environment in environments {
            if environment.seats.len() != 2 || environment.policy_seat > 1 {
                return Err(PpoError::InvalidTransition(
                    "opponent batch requires two valid seats",
                ));
            }
            let OpponentRuntime::Policy { model: other, .. } = &environment.opponent else {
                return Err(PpoError::InvalidTransition(
                    "opponent batch requires neural policy",
                ));
            };
            if !Arc::ptr_eq(model, other) {
                return Err(PpoError::InvalidTransition(
                    "opponent batch model allocation mismatch",
                ));
            }
            if other.policy_identity().map_err(text_error)? != policy {
                return Err(PpoError::InvalidTransition(
                    "opponent batch policy identity changed",
                ));
            }
        }
        Ok(Self {
            model: Arc::clone(model),
            policy,
        })
    }

    pub(super) fn sample(
        &self,
        inputs: Vec<PreparedOpponent>,
    ) -> Result<Vec<OpponentChoice>, PpoError> {
        self.validate_inputs(&inputs)?;
        let mut frames = Vec::with_capacity(inputs.len());
        let mut spaces = Vec::with_capacity(inputs.len());
        let mut before = Vec::with_capacity(inputs.len());
        for input in inputs {
            frames.push(input.frame);
            spaces.push(input.space);
            before.push(input.random);
        }
        let mut random = before.clone();
        #[cfg(test)]
        BATCH_COUNTS.with(|counts| {
            let (calls, rows) = counts.get();
            counts.set((
                calls.checked_add(1).expect("bounded calls"),
                rows.checked_add(frames.len()).expect("bounded rows"),
            ));
        });
        let choices = self
            .model
            .sample_batch(&frames, &spaces, &mut random)
            .map_err(text_error)?;
        assert_eq!(choices.len(), frames.len());
        if choices.iter().any(|choice| choice.policy() != self.policy) {
            return Err(PpoError::InvalidTransition(
                "opponent batch policy identity changed",
            ));
        }
        Ok(choices
            .into_iter()
            .zip(spaces)
            .zip(before)
            .zip(random)
            .map(|(((choice, space), before), after)| OpponentChoice {
                model: Arc::clone(&self.model),
                choice,
                space,
                before,
                after,
            })
            .collect())
    }

    fn validate_inputs(&self, inputs: &[PreparedOpponent]) -> Result<(), PpoError> {
        validate_count(inputs.len())?;
        if self.model.policy_identity().map_err(text_error)? != self.policy {
            return Err(PpoError::InvalidTransition(
                "opponent batch policy identity changed",
            ));
        }
        for input in inputs {
            if !Arc::ptr_eq(&self.model, &input.model) {
                return Err(PpoError::InvalidTransition(
                    "prepared opponent model allocation mismatch",
                ));
            }
            if !input.frame.is_finite() {
                return Err(PpoError::InvalidTransition(
                    "prepared opponent frame is not finite",
                ));
            }
            if !input.frame.matches_action_space(&input.space) {
                return Err(PpoError::InvalidTransition(
                    "prepared opponent action space mismatch",
                ));
            }
        }
        Ok(())
    }
}

fn validate_count(count: usize) -> Result<(), PpoError> {
    if !(1..=64).contains(&count) {
        return Err(PpoError::InvalidTransition(
            "opponent batch row count must be 1..=64",
        ));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    static BATCH_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
pub(super) fn take_batch_counts() -> (usize, usize) {
    BATCH_COUNTS.with(|counts| counts.replace((0, 0)))
}
