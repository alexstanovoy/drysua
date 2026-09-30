//! The device-resident PPO learner.
//!
//! An update's samples are staged on the learner device once. Each Adam step
//! gathers its minibatch there, accumulates gradients over microbatches on the
//! device and runs Adam on the device; the host reads back only the loss
//! statistics, the gradient norm and the new moments. A candidate whose KL
//! exceeds the target is rolled back from device copies of the parameters.

use super::ppo_objective::{
    HostTargets, ObjectiveTargets, candidate_kl_sum, ppo_loss, report_from_sums,
};
use super::rows::{ENCODER_PARTS, PART_LENGTHS};
use super::*;

/// Largest microbatch the device learner runs in one forward and backward pass.
pub const MODEL_PPO_MAX_MICROBATCH: usize = 2_048;

/// One update's samples on the learner device, row aligned across every tensor.
pub(crate) struct StagedPpoBatch {
    parts: Vec<Tensor>,
    sides: Tensor,
    prefixes: PrefixUpload,
    targets: ObjectiveTargets,
    rows: usize,
}

/// Rows uploaded per host-to-device copy while staging; bounds the host buffer.
const STAGING_CHUNK_ROWS: usize = 256;

/// A [`StagedPpoBatch`] being filled: encoder parts stream to preallocated device
/// columns in chunks, while the small per-row targets stay on the host until done.
pub(crate) struct PpoStaging {
    parts: Vec<Tensor>,
    chunk: Vec<Vec<f32>>,
    chunk_rows: usize,
    sides: Vec<u8>,
    prefixes: Vec<TrainingPrefix>,
    targets: HostTargets,
    row: EncoderRow,
    capacity: usize,
}

impl PpoStaging {
    /// Appends one prepared sample: its packed encoder row, side, prefix and targets.
    pub(crate) fn push(&mut self, sample: &PpoPreparedSample) -> Result<(), ModelError> {
        if self.sides.len() >= self.capacity {
            return Err(ModelError::InvalidModelState("PPO staging capacity"));
        }
        let index = self.sides.len();
        self.row
            .pack(&sample.transition.frame)
            .map_err(|error| match error {
                ModelError::NonFiniteFrame { .. } => ModelError::NonFiniteFrame { index },
                error => error,
            })?;
        self.targets.push(sample)?;
        for (part, values) in self.chunk.iter_mut().enumerate() {
            values.extend_from_slice(self.row.part_values(part));
        }
        self.sides.push(u8::from(self.row.radiant()));
        self.prefixes.push(sample.transition.target.prefix());
        self.chunk_rows += 1;
        if self.chunk_rows == STAGING_CHUNK_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ModelError> {
        let rows = self.chunk_rows;
        if rows == 0 {
            return Ok(());
        }
        let offset = self.sides.len() - rows;
        for ((part, values), length) in self.parts.iter().zip(&mut self.chunk).zip(PART_LENGTHS) {
            let chunk = Tensor::from_slice(values, (rows, length), part.device())?;
            part.slice_set(&chunk, 0, offset)?;
            values.clear();
        }
        self.chunk_rows = 0;
        Ok(())
    }
}

/// Updated parameters and the new first and second moments of one Adam step.
type AdamUpdate = (Vec<Tensor>, Vec<f32>, Vec<f32>);

/// Upper bound on the rows one update stages.
pub(crate) const MODEL_MAX_STAGED_ROWS: usize = crate::PPO_MAX_SAMPLES;

/// One gathered microbatch, ready for the training forward.
pub(super) struct StagedInputs {
    pub(super) encoder: EncoderInputs,
    pub(super) sides: Tensor,
    pub(super) prefixes: PrefixUpload,
    pub(super) targets: ObjectiveTargets,
}

impl StagedPpoBatch {
    pub(super) fn gather(&self, indices: &Tensor) -> Result<StagedInputs, ModelError> {
        let rows = indices.elem_count();
        let parts = self
            .parts
            .iter()
            .map(|part| Ok(part.index_select(indices, 0)?))
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(StagedInputs {
            encoder: EncoderInputs::from_parts(parts, rows)?,
            sides: self.sides.index_select(indices, 0)?,
            prefixes: self.prefixes.gather(indices)?,
            targets: self.targets.gather(indices)?,
        })
    }
}

impl PolicyModel {
    /// Empty staging for at most `capacity` rows, allocated on the learner device.
    pub(crate) fn ppo_staging(&self, capacity: usize) -> Result<PpoStaging, ModelError> {
        if capacity == 0 || capacity > MODEL_MAX_STAGED_ROWS {
            return Err(ModelError::InvalidModelState("PPO staging capacity"));
        }
        let device = self.tensor_device();
        Ok(PpoStaging {
            parts: PART_LENGTHS
                .iter()
                .map(|&length| Ok(Tensor::zeros((capacity, length), DType::F32, device)?))
                .collect::<Result<_, ModelError>>()?,
            chunk: PART_LENGTHS
                .iter()
                .map(|length| Vec::with_capacity(STAGING_CHUNK_ROWS * length))
                .collect(),
            chunk_rows: 0,
            sides: Vec::with_capacity(capacity),
            prefixes: Vec::with_capacity(capacity),
            targets: HostTargets::with_capacity(capacity),
            row: EncoderRow::new(),
            capacity,
        })
    }

    /// Finishes staging: uploads the last chunk and the per-row targets.
    pub(crate) fn stage_ppo_batch(
        &self,
        mut staging: PpoStaging,
    ) -> Result<StagedPpoBatch, ModelError> {
        staging.flush()?;
        let rows = staging.sides.len();
        if rows == 0 {
            return Err(ModelError::EmptyTrainingBatch);
        }
        let device = self.tensor_device();
        assert_eq!(staging.parts.len(), ENCODER_PARTS);
        Ok(StagedPpoBatch {
            parts: staging
                .parts
                .iter()
                .map(|part| Ok(part.narrow(0, 0, rows)?))
                .collect::<Result<_, ModelError>>()?,
            sides: Tensor::from_vec(staging.sides, (rows, 1), device)?,
            prefixes: PrefixUpload::new(&staging.prefixes, device)?,
            targets: staging.targets.upload(device)?,
            rows,
        })
    }

    #[cfg(test)]
    /// Stages prepared samples; the convenience path of tests and small updates.
    pub(crate) fn stage_ppo_examples(
        &self,
        examples: &[&PpoPreparedSample],
    ) -> Result<StagedPpoBatch, ModelError> {
        let mut staging = self.ppo_staging(examples.len().max(1))?;
        for sample in examples {
            staging.push(sample)?;
        }
        self.stage_ppo_batch(staging)
    }

    #[cfg(test)]
    pub(crate) fn ppo_update(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_update_with_microbatch(
            examples,
            adam,
            config,
            crate::training_execution::DEFAULT_TRAINING_MICROBATCH,
            #[cfg(test)]
            PpoTestFaults::default(),
        )
    }

    #[cfg(test)]
    pub(super) fn ppo_update_with_microbatch(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch: usize,
        #[cfg(test)] faults: PpoTestFaults,
    ) -> Result<PpoMinibatchReport, ModelError> {
        if examples.is_empty() || examples.len() > MODEL_MAX_BATCH {
            return Err(ModelError::InvalidModelState("PPO minibatch count"));
        }
        let staged = self.stage_ppo_examples(examples)?;
        let indices = (0..examples.len()).collect::<Vec<_>>();
        self.ppo_step_staged(
            &staged,
            &indices,
            adam,
            config,
            microbatch,
            #[cfg(test)]
            faults,
        )
    }

    /// One Adam step on the staged rows at `indices`, guarded by the KL target.
    pub(crate) fn ppo_update_staged(
        &self,
        staged: &StagedPpoBatch,
        indices: &[usize],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch: usize,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_step_staged(
            staged,
            indices,
            adam,
            config,
            microbatch,
            #[cfg(test)]
            PpoTestFaults::default(),
        )
    }

    fn ppo_step_staged(
        &self,
        staged: &StagedPpoBatch,
        indices: &[usize],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch: usize,
        #[cfg(test)] faults: PpoTestFaults,
    ) -> Result<PpoMinibatchReport, ModelError> {
        if indices.is_empty() || indices.len() > MODEL_MAX_BATCH {
            return Err(ModelError::InvalidModelState("PPO minibatch count"));
        }
        if !(1..=MODEL_PPO_MAX_MICROBATCH).contains(&microbatch) {
            return Err(ModelError::InvalidModelState("PPO microbatch size"));
        }
        if indices.iter().any(|&index| index >= staged.rows) {
            return Err(ModelError::InvalidModelState("PPO staged row index"));
        }
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(adam.binding)?;
        let rows = indices
            .iter()
            .map(|&index| index as u32)
            .collect::<Vec<_>>();
        let rows = Tensor::from_vec(rows, indices.len(), self.tensor_device())?;
        let (gradients, mut report) = self.staged_gradients(staged, &rows, config, microbatch)?;
        if report.approximate_kl > f64::from(config.target_kl) {
            return Ok(report);
        }
        let original = self.device_parameter_copies()?;
        let original_adam = adam.clone();
        let diagnostics = self.apply_adam_device(adam, &gradients, &original)?;
        let candidate_kl = self.staged_candidate_kl(staged, &rows, microbatch);
        #[cfg(test)]
        let candidate_kl = if faults.candidate_evaluation {
            Err(ModelError::Backend(format!(
                "injected PPO candidate evaluation failure after {} rows",
                indices.len().min(microbatch)
            )))
        } else {
            candidate_kl
        };
        if !candidate_kl
            .as_ref()
            .is_ok_and(|kl| *kl <= f64::from(config.target_kl))
        {
            self.rollback_device_candidate(
                &original,
                original_adam,
                adam,
                &candidate_kl,
                #[cfg(test)]
                faults.rollback_import,
            )?;
            report.approximate_kl = candidate_kl?;
            return Ok(report);
        }
        report.approximate_kl = candidate_kl?;
        report.gradient_norm = diagnostics.unclipped_norm;
        report.applied_scale = diagnostics.applied_scale;
        report.applied = true;
        Ok(report)
    }

    /// Minibatch gradients summed on the device and the minibatch report.
    fn staged_gradients(
        &self,
        staged: &StagedPpoBatch,
        rows: &Tensor,
        config: PpoConfig,
        microbatch: usize,
    ) -> Result<(Vec<Option<Tensor>>, PpoMinibatchReport), ModelError> {
        let total = rows.elem_count();
        let parameters = self.parameters();
        let mut gradients: Vec<Option<Tensor>> = vec![None; parameters.len()];
        let mut sums: Option<Tensor> = None;
        for start in (0..total).step_by(microbatch) {
            let length = microbatch.min(total - start);
            let inputs = staged.gather(&rows.narrow(0, start, length)?)?;
            let output = self.training_forward_inputs(&inputs, false)?;
            let probe = side_actors::training_finite_probe(&output)?;
            let terms = ppo_loss(&output, &inputs.targets, config, total)?;
            let store = terms.loss.backward()?;
            for (total, parameter) in gradients.iter_mut().zip(&parameters) {
                if let Some(gradient) = store.get(parameter.value.as_tensor()) {
                    *total = Some(match total.take() {
                        None => gradient.clone(),
                        Some(sum) => (sum + gradient)?,
                    });
                }
            }
            let terms = Tensor::cat(&[terms.sums, probe.reshape(1)?], 0)?;
            sums = Some(match sums {
                None => terms,
                Some(sum) => (sum + terms)?,
            });
        }
        let sums = sums
            .ok_or(ModelError::EmptyTrainingBatch)?
            .to_vec1::<f32>()?;
        let (probe, sums) = sums.split_last().expect("probe");
        if !probe.is_finite() {
            return Err(self.locate_nonfinite_output(staged, rows, microbatch));
        }
        Ok((gradients, report_from_sums(sums, total)?))
    }

    fn staged_candidate_kl(
        &self,
        staged: &StagedPpoBatch,
        rows: &Tensor,
        microbatch: usize,
    ) -> Result<f64, ModelError> {
        let total = rows.elem_count();
        let mut sum: Option<Tensor> = None;
        for start in (0..total).step_by(microbatch) {
            let length = microbatch.min(total - start);
            let inputs = staged.gather(&rows.narrow(0, start, length)?)?;
            let output = self.training_forward_inputs(&inputs, true)?;
            let probe = side_actors::training_finite_probe(&output)?;
            let kl = candidate_kl_sum(&output, &inputs.targets)?.detach();
            let terms = Tensor::stack(&[kl, probe], 0)?;
            sum = Some(match sum {
                None => terms,
                Some(sum) => (sum + terms)?,
            });
        }
        let sums = sum
            .ok_or(ModelError::EmptyTrainingBatch)?
            .to_vec1::<f32>()?;
        if !sums[1].is_finite() {
            return Err(self.locate_nonfinite_output(staged, rows, microbatch));
        }
        let kl = f64::from(sums[0]) / total as f64;
        if !kl.is_finite() {
            return Err(ModelError::NonFiniteLoss);
        }
        Ok(kl)
    }

    /// The first non-finite training output, found by reading every microbatch
    /// back; runs only after the device probe has already failed.
    fn locate_nonfinite_output(
        &self,
        staged: &StagedPpoBatch,
        rows: &Tensor,
        microbatch: usize,
    ) -> ModelError {
        let total = rows.elem_count();
        for start in (0..total).step_by(microbatch) {
            let located = rows
                .narrow(0, start, microbatch.min(total - start))
                .map_err(ModelError::from)
                .and_then(|rows| staged.gather(&rows))
                .and_then(|inputs| self.training_forward_inputs(&inputs, true))
                .and_then(|output| validate_training_tensors_finite(&output));
            if let Err(error) = located {
                return error;
            }
        }
        ModelError::NonFiniteLoss
    }

    /// Device copies of every parameter, the rollback point of one Adam step.
    fn device_parameter_copies(&self) -> Result<Vec<Tensor>, ModelError> {
        self.parameters()
            .iter()
            .map(|parameter| Ok(parameter.value.as_tensor().copy()?.detach()))
            .collect()
    }

    fn rollback_device_candidate(
        &self,
        original: &[Tensor],
        original_adam: AdamState,
        adam: &mut AdamState,
        candidate_kl: &Result<f64, ModelError>,
        #[cfg(test)] inject_failure: bool,
    ) -> Result<(), ModelError> {
        assert_eq!(adam.binding.lineage, original_adam.binding.lineage);
        assert_eq!(adam.step(), original_adam.step() + 1);
        #[cfg(not(test))]
        let fail_after = None;
        #[cfg(test)]
        let fail_after = inject_failure.then_some(0);
        let parameters = self.parameters();
        apply_parameter_tensors(&parameters, original, original, fail_after).map_err(
            |rollback| {
                let cause = match candidate_kl {
                    Ok(kl) => format!("sampled KL {kl} exceeds target"),
                    Err(error) => error.to_string(),
                };
                ModelError::Backend(format!(
                    "PPO candidate rejected ({cause}); parameter rollback failed ({rollback})"
                ))
            },
        )?;
        self.parameter_revision
            .store(original_adam.binding.policy.revision, Ordering::Relaxed);
        *adam = original_adam;
        Ok(())
    }

    /// Clipped Adam on the device; the host keeps the committed moments.
    fn apply_adam_device(
        &self,
        adam: &mut AdamState,
        gradients: &[Option<Tensor>],
        original: &[Tensor],
    ) -> Result<AdamDiagnostics, ModelError> {
        assert_eq!(adam.binding.policy, self.policy_identity_locked());
        let next = self.next_policy_identity_locked()?;
        let step = adam
            .step
            .checked_add(1)
            .filter(|value| *value <= MODEL_MAX_OPTIMIZER_STEP)
            .ok_or(ModelError::OptimizerStepOverflow)?;
        let norm = gradient_norm_device(gradients)?;
        let (scale, corrections) = adam_step_factors(adam.config, norm, step);
        let (updated, first, second) =
            self.adam_update_parameters(adam, gradients, scale, corrections)?;
        apply_parameter_tensors(&self.parameters(), &updated, original, None)?;
        adam.first_moment = first;
        adam.second_moment = second;
        adam.step = step;
        adam.binding.policy = next;
        self.parameter_revision
            .store(next.revision, Ordering::Relaxed);
        Ok(AdamDiagnostics {
            unclipped_norm: norm,
            applied_scale: scale,
        })
    }

    /// Every parameter's Adam update and the new moments, checked finite.
    fn adam_update_parameters(
        &self,
        adam: &AdamState,
        gradients: &[Option<Tensor>],
        scale: f64,
        corrections: (f64, f64),
    ) -> Result<AdamUpdate, ModelError> {
        let config = adam.config;
        let device = self.tensor_device();
        let first = Tensor::from_slice(&adam.first_moment, MODEL_PARAMETER_COUNT, device)?;
        let second = Tensor::from_slice(&adam.second_moment, MODEL_PARAMETER_COUNT, device)?;
        let parameters = self.parameters();
        let mut updated = Vec::with_capacity(parameters.len());
        let mut moments = (
            Vec::with_capacity(parameters.len()),
            Vec::with_capacity(parameters.len()),
        );
        let mut offset = 0;
        for (parameter, gradient) in parameters.iter().zip(gradients) {
            let value = parameter.value.as_tensor();
            let count = value.elem_count();
            let gradient = match gradient {
                Some(gradient) => gradient.affine(scale, 0.0)?,
                None => value.zeros_like()?,
            };
            let moments_in = (
                first.narrow(0, offset, count)?.reshape(value.shape())?,
                second.narrow(0, offset, count)?.reshape(value.shape())?,
            );
            let (value, m, v) =
                adam_tensor_step(value, &gradient, moments_in, config, corrections)?;
            updated.push(value);
            moments.0.push(m.flatten_all()?);
            moments.1.push(v.flatten_all()?);
            offset += count;
        }
        assert_eq!(offset, MODEL_PARAMETER_COUNT);
        let first = Tensor::cat(&moments.0, 0)?.to_vec1::<f32>()?;
        let second = Tensor::cat(&moments.1, 0)?.to_vec1::<f32>()?;
        if let Some(index) = first
            .iter()
            .zip(&second)
            .position(|(first, second)| !first.is_finite() || !second.is_finite() || *second < 0.0)
        {
            return Err(ModelError::NonFiniteOptimizerUpdate { index });
        }
        let check = updated
            .iter()
            .map(|tensor| tensor.sum_all())
            .collect::<Result<Vec<_>, _>>()?;
        if !Tensor::stack(&check, 0)?
            .sum_all()?
            .to_scalar::<f32>()?
            .is_finite()
        {
            return Err(ModelError::NonFiniteOptimizerUpdate { index: 0 });
        }
        Ok((updated, first, second))
    }
}

/// One Adam update of a parameter tensor from its already clipped gradient.
pub(super) fn adam_tensor_step(
    value: &Tensor,
    gradient: &Tensor,
    (m, v): (Tensor, Tensor),
    config: AdamConfig,
    (first_correction, second_correction): (f64, f64),
) -> Result<(Tensor, Tensor, Tensor), ModelError> {
    let beta1 = f64::from(config.beta1);
    let beta2 = f64::from(config.beta2);
    let m = (m.affine(beta1, 0.0)? + gradient.affine(1.0 - beta1, 0.0)?)?;
    let v = (v.affine(beta2, 0.0)? + gradient.sqr()?.affine(1.0 - beta2, 0.0)?)?;
    let denominator = v
        .affine(1.0 / second_correction, 0.0)?
        .sqrt()?
        .affine(1.0, f64::from(config.epsilon))?;
    let delta = m
        .affine(f64::from(config.learning_rate) / first_correction, 0.0)?
        .div(&denominator)?;
    Ok(((value - delta)?.detach(), m, v))
}

/// The clip scale for a gradient norm and Adam's bias corrections at `step`.
pub(super) fn adam_step_factors(config: AdamConfig, norm: f64, step: u64) -> (f64, (f64, f64)) {
    let scale = if norm > f64::from(config.gradient_clip) {
        f64::from(config.gradient_clip) / norm
    } else {
        1.0
    };
    let corrections = (
        1.0 - f64::from(config.beta1).powi(step as i32),
        1.0 - f64::from(config.beta2).powi(step as i32),
    );
    (scale, corrections)
}

/// Global L2 norm of the summed gradients, read back once.
pub(super) fn gradient_norm_device(gradients: &[Option<Tensor>]) -> Result<f64, ModelError> {
    let squares = gradients
        .iter()
        .flatten()
        .map(|gradient| gradient.sqr()?.sum_all())
        .collect::<Result<Vec<_>, _>>()?;
    let squared = if squares.is_empty() {
        0.0
    } else {
        Tensor::stack(&squares, 0)?
            .to_dtype(DType::F64)?
            .sum_all()?
            .to_scalar::<f64>()?
    };
    let norm = squared.sqrt();
    if !norm.is_finite() {
        return Err(ModelError::NonFiniteOptimizerNorm);
    }
    Ok(norm)
}
