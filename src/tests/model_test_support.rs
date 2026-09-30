use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdamStepTestResult {
    pub parameters: Vec<f32>,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    pub unclipped_norm: f64,
    pub applied_scale: f64,
}

impl PolicyModel {
    pub(crate) fn import_parameters_with_failure(
        &self,
        values: &[f32],
        fail_after: usize,
    ) -> Result<(), ModelError> {
        self.import_parameters_inner(values, Some(fail_after))
    }

    pub(crate) fn ppo_update_with_microbatch_for_test(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        microbatch_size: usize,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_update_with_microbatch(
            examples,
            adam,
            config,
            microbatch_size,
            PpoTestFaults::default(),
        )
    }

    pub(crate) fn ppo_update_with_faults_for_test(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        rollback_import: bool,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.ppo_update_with_microbatch(
            examples,
            adam,
            config,
            MODEL_TRAINING_BATCH,
            PpoTestFaults {
                candidate_evaluation: true,
                rollback_import,
            },
        )
    }

    /// Makes every item head output NaN.
    pub(crate) fn poison_item_head_for_test(&self) -> Result<(), ModelError> {
        let poison = Tensor::full(f32::NAN, MODEL_ITEM_HEAD, self.tensor_device())?;
        Ok(self.item_head.bias.set(&poison)?)
    }

    /// Behaviour-action log-probabilities and the objective report at the current weights.
    pub(crate) fn ppo_likelihood_for_test(
        &self,
        examples: &[&PpoPreparedSample],
    ) -> Result<(Vec<f32>, PpoMinibatchReport), ModelError> {
        let staged = self.stage_ppo_examples(examples)?;
        let _guard = self.read_parameter_lock()?;
        let rows = Tensor::arange(0u32, examples.len() as u32, self.tensor_device())?;
        let inputs = staged.gather(&rows)?;
        let output = self.training_forward_inputs(&inputs, false)?;
        validate_training_tensors_finite(&output)?;
        let log_probability = ppo_objective::log_probability(&output, &inputs.targets)?;
        let terms = ppo_objective::ppo_loss(
            &output,
            &inputs.targets,
            PpoConfig::default(),
            examples.len(),
        )?;
        let report = ppo_objective::report_from_sums(&terms.sums.to_vec1()?, examples.len())?;
        Ok((log_probability.to_vec1()?, report))
    }
}

pub(crate) struct PpoEntropyProbe {
    pub entropy: Vec<f32>,
    pub gradients: Vec<Vec<f32>>,
    pub log_normalizer: Vec<f32>,
}

pub(crate) fn masked_ppo_entropy_for_test(
    logits: &[[f32; MODEL_KIND_HEAD]],
    examples: &[&PpoPreparedSample],
    device: PolicyDevice,
) -> Result<PpoEntropyProbe, ModelError> {
    assert!(!examples.is_empty());
    assert!(examples.len() <= MODEL_TRAINING_BATCH);
    let device = device.candle()?;
    let tensor = Tensor::from_vec(
        logits.iter().flatten().copied().collect::<Vec<_>>(),
        (logits.len(), MODEL_KIND_HEAD),
        &device,
    )?;
    let variable = Var::from_tensor(&tensor)?;
    let mut targets = ppo_objective::HostTargets::with_capacity(examples.len());
    for sample in examples {
        targets.push(sample)?;
    }
    let targets = targets.upload(&device)?;
    let (masks, active) = targets.head_for_test(0);
    let (entropy, log_normalizer) =
        ppo_objective::masked_head_entropy(variable.as_tensor(), masks, active)?;
    let gradients = entropy.sum_all()?.backward()?;
    let gradient = gradients
        .get(variable.as_tensor())
        .ok_or(ModelError::InvalidModelState("PPO entropy gradient"))?;
    Ok(PpoEntropyProbe {
        entropy: entropy.to_vec1()?,
        gradients: gradient.to_vec2()?,
        log_normalizer: log_normalizer.flatten_all()?.to_vec1()?,
    })
}

/// One step of the device Adam on a flat parameter vector, at `step` completed steps.
pub(crate) fn adam_step_for_test(
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    step: u64,
    config: AdamConfig,
) -> Result<AdamStepTestResult, ModelError> {
    let device = Device::Cpu;
    let tensor = |values: &[f32]| Tensor::from_slice(values, values.len(), &device);
    let gradient = tensor(gradients)?;
    let norm = device_learner::gradient_norm_device(&[Some(gradient.clone())])?;
    let (scale, corrections) = device_learner::adam_step_factors(config, norm, step + 1);
    let (parameters, first, second) = device_learner::adam_tensor_step(
        &tensor(parameters)?,
        &gradient.affine(scale, 0.0)?,
        (tensor(first_moment)?, tensor(second_moment)?),
        config,
        corrections,
    )?;
    Ok(AdamStepTestResult {
        parameters: parameters.to_vec1()?,
        first_moment: first.to_vec1()?,
        second_moment: second.to_vec1()?,
        unclipped_norm: norm,
        applied_scale: scale,
    })
}
