use super::*;

impl PolicyModel {
    pub(crate) fn fit_value_probe(
        &self,
        frames: &[FeatureFrame],
        targets: &[f32],
        optimizer: &mut AdamState,
        train_representation: bool,
    ) -> Result<f64, ModelError> {
        if frames.is_empty() || frames.len() > MODEL_TRAINING_BATCH || frames.len() != targets.len() {
            return Err(ModelError::InvalidModelState("value-probe batch shape"));
        }
        if targets.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return Err(ModelError::InvalidModelState("value-probe target range"));
        }
        validate_batch(frames)?;
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(optimizer.binding)?;
        let state = self.forward_frames(frames)?;
        let input = if train_representation { state.trunk } else { state.trunk.detach() };
        let value = self.value.forward(&input)?;
        let targets = Tensor::from_slice(targets, (frames.len(), 1), self.tensor_device())?;
        let loss = value.sub(&targets)?.sqr()?.mean_all()?;
        let measured = loss.to_scalar::<f32>()?;
        if !measured.is_finite() { return Err(ModelError::NonFiniteLoss); }
        let gradients = collect_host_gradients(self.backward_named_locked(&loss)?)?;
        let step = optimizer.step();
        self.apply_adam_locked(optimizer, &gradients)?;
        assert_eq!(optimizer.step(), step + 1);
        assert_eq!(optimizer.policy_identity(), self.policy_identity_locked());
        Ok(f64::from(measured))
    }
}
