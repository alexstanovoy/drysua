use super::*;

#[derive(Clone, Debug)]
pub(crate) struct CheckedActionSet {
    samples: Vec<CheckedBehavioralSample>,
    actions: Vec<StructuredAction>,
}

impl CheckedActionSet {
    pub(crate) fn new(
        frame: FeatureFrame,
        space: &ActionSpace,
        actions: &[StructuredAction],
    ) -> Result<Self, ModelError> {
        if actions.is_empty() || actions.len() > 4 {
            return Err(ModelError::InvalidModelState("advantage action-set size"));
        }
        let mut samples = Vec::with_capacity(actions.len());
        for (index, action) in actions.iter().enumerate() {
            if actions[..index].contains(action) {
                return Err(ModelError::InvalidModelState("duplicate advantage action"));
            }
            samples.push(
                CheckedBehavioralSample::new(frame.clone(), space, *action)
                    .map_err(|error| ModelError::Backend(error.to_string()))?,
            );
        }
        Ok(Self {
            samples,
            actions: actions.to_vec(),
        })
    }
    pub(crate) fn frame(&self) -> &FeatureFrame {
        self.samples[0].frame()
    }
    pub(crate) fn actions(&self) -> &[StructuredAction] {
        &self.actions
    }
}

impl PolicyModel {
    pub(crate) fn train_checked_action_sets(
        &self,
        sets: &[&CheckedActionSet],
        optimizer: &mut AdamState,
    ) -> Result<ModelUpdateReport, ModelError> {
        if sets.is_empty() || sets.len() > 16 {
            return Err(ModelError::InvalidModelState("advantage state batch size"));
        }
        let samples: Vec<_> = sets.iter().flat_map(|set| set.samples.iter()).collect();
        let (frames, prefixes) = transfer_fit::checked_inputs(&samples)?;
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(optimizer.binding)?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let (row_losses, active_head_counts) = transfer_fit::checked_row_losses(&output, &samples)?;
        let mut losses = Vec::with_capacity(sets.len());
        let mut offset = 0;
        for set in sets {
            let count = set.samples.len();
            losses.push(
                row_losses
                    .narrow(0, offset, count)?
                    .neg()?
                    .log_sum_exp(0)?
                    .neg()?,
            );
            offset += count;
        }
        assert_eq!(offset, samples.len());
        let loss = Tensor::stack(&losses, 0)?.mean_all()?;
        let average_loss = f64::from(loss.to_scalar::<f32>()?);
        if !average_loss.is_finite() {
            return Err(ModelError::NonFiniteLoss);
        }
        let gradients = collect_host_gradients(self.backward_named_locked(&loss)?)?;
        let step = optimizer.step();
        let diagnostics = self.apply_adam_locked(optimizer, &gradients)?;
        assert_eq!(optimizer.step(), step + 1);
        assert_eq!(optimizer.policy_identity(), self.policy_identity_locked());
        Ok(ModelUpdateReport {
            average_loss,
            active_head_counts,
            unclipped_norm: diagnostics.unclipped_norm,
            applied_scale: diagnostics.applied_scale,
            sample_count: sets.len(),
            optimizer_step: optimizer.step(),
        })
    }
}
