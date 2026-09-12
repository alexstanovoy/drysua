use super::*;

#[derive(Clone, Debug)]
pub(crate) struct CheckedBehavioralSample {
    frame: FeatureFrame,
    target: BehavioralTarget,
}

impl CheckedBehavioralSample {
    pub(crate) fn new(
        frame: FeatureFrame,
        space: &ActionSpace,
        action: StructuredAction,
    ) -> Result<Self, crate::ImitationError> {
        let target = BehavioralTarget::from_action(&frame, space, action)?;
        target.validate()?;
        assert!(frame.matches_action_space(space));
        Ok(Self { frame, target })
    }
    pub(crate) fn frame(&self) -> &FeatureFrame {
        &self.frame
    }
    pub(crate) fn target(&self) -> &BehavioralTarget {
        &self.target
    }
}

impl PolicyModel {
    pub(crate) fn train_checked_behavioral_batch(
        &self,
        samples: &[&CheckedBehavioralSample],
        optimizer: &mut AdamState,
    ) -> Result<ModelUpdateReport, ModelError> {
        let (frames, prefixes) = checked_inputs(samples)?;
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(optimizer.binding)?;
        validate_adam_parts(
            optimizer.config,
            &optimizer.first_moment,
            &optimizer.second_moment,
            optimizer.step,
            MODEL_PARAMETER_COUNT,
        )?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let (loss, active_head_counts) = checked_loss(&output, samples)?;
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
            sample_count: samples.len(),
            optimizer_step: optimizer.step(),
        })
    }

    pub(crate) fn checked_behavioral_predictions(
        &self,
        samples: &[&CheckedBehavioralSample],
    ) -> Result<Vec<BehavioralPrediction>, ModelError> {
        let (frames, prefixes) = checked_inputs(samples)?;
        let _guard = self.read_parameter_lock()?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        validate_training_tensors_finite(&output)?;
        let logits = BehavioralHostLogits::from_tensors(&output)?;
        samples
            .iter()
            .enumerate()
            .map(|(index, sample)| logits.predict(index, &sample.target))
            .collect()
    }
}

pub(super) fn checked_inputs(
    samples: &[&CheckedBehavioralSample],
) -> Result<(Vec<FeatureFrame>, Vec<TrainingPrefix>), ModelError> {
    if samples.is_empty() || samples.len() > MODEL_TRAINING_BATCH {
        return Err(ModelError::BehavioralExampleCount {
            count: samples.len(),
            maximum: MODEL_TRAINING_BATCH,
        });
    }
    for sample in samples {
        sample
            .target
            .validate()
            .map_err(|error| ModelError::Backend(error.to_string()))?;
    }
    let frames: Vec<_> = samples.iter().map(|sample| sample.frame.clone()).collect();
    let prefixes: Vec<_> = samples
        .iter()
        .map(|sample| sample.target.prefix())
        .collect();
    validate_training_batch(&frames, &prefixes)?;
    Ok((frames, prefixes))
}

fn checked_loss(
    output: &PolicyTensorTensors,
    samples: &[&CheckedBehavioralSample],
) -> Result<(Tensor, [usize; MODEL_BEHAVIORAL_HEADS]), ModelError> {
    let (loss, counts) = checked_row_losses(output, samples)?;
    Ok((loss.mean_all()?, counts))
}

pub(super) fn checked_row_losses(
    output: &PolicyTensorTensors,
    samples: &[&CheckedBehavioralSample],
) -> Result<(Tensor, [usize; MODEL_BEHAVIORAL_HEADS]), ModelError> {
    let mut counts = [0; MODEL_BEHAVIORAL_HEADS];
    macro_rules! head {
        ($field:ident, $index:expr) => {{
            let mut masks = Vec::new();
            let mut labels = Vec::new();
            let mut active = Vec::new();
            for sample in samples {
                let target = &sample.target.$field;
                counts[$index] += usize::from(target.active);
                append_tensor_target(
                    target,
                    stringify!($field),
                    &mut masks,
                    &mut labels,
                    &mut active,
                )?;
            }
            masked_loss_from_parts(
                &output.$field,
                samples.len(),
                output.$field.dim(1)?,
                masks,
                labels,
                active,
            )?
        }};
    }
    let mut loss = head!(kind, 0);
    macro_rules! add {
        ($field:ident, $index:expr) => {
            loss = (loss + head!($field, $index))?;
        };
    }
    add!(controlled, 1);
    add!(ability, 2);
    add!(item, 3);
    add!(swap, 4);
    add!(learn, 5);
    add!(shop, 6);
    add!(loot, 7);
    add!(target_mode, 8);
    add!(put_mode, 9);
    add!(entity_pointer, 10);
    add!(point_pointer, 11);
    Ok((loss, counts))
}
