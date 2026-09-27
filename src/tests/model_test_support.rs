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

    pub(crate) fn ppo_likelihood_for_test(
        &self,
        examples: &[&PpoPreparedSample],
    ) -> Result<(Vec<f32>, PpoMinibatchReport), ModelError> {
        let frames = examples
            .iter()
            .map(|sample| sample.transition.frame.clone())
            .collect::<Vec<_>>();
        let prefixes = examples
            .iter()
            .map(|sample| sample.transition.target.prefix())
            .collect::<Vec<_>>();
        validate_training_batch(&frames, &prefixes)?;
        let _guard = self.read_parameter_lock()?;
        let output = self.training_forward_locked(&frames, &prefixes)?;
        let log_probability = ppo_negative_log_probability(&output, examples)?.neg()?;
        let (_, report) = ppo_loss(&output, examples, PpoConfig::default())?;
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
    let output =
        masked_ppo_head_entropy_tensors(variable.as_tensor(), examples, |target| &target.kind)?;
    let gradients = output.entropy.sum_all()?.backward()?;
    let gradient = gradients
        .get(variable.as_tensor())
        .ok_or(ModelError::InvalidModelState("PPO entropy gradient"))?;
    Ok(PpoEntropyProbe {
        entropy: output.entropy.to_vec1()?,
        gradients: gradient.to_vec2()?,
        log_normalizer: output.log_normalizer.flatten_all()?.to_vec1()?,
    })
}

pub(crate) fn adam_step_for_test(
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    step: u64,
    config: AdamConfig,
) -> Result<AdamStepTestResult, ModelError> {
    let update = compute_adam_step(
        parameters,
        gradients,
        first_moment,
        second_moment,
        step,
        config,
    )?;
    Ok(AdamStepTestResult {
        parameters: update.parameters,
        first_moment: update.first_moment,
        second_moment: update.second_moment,
        unclipped_norm: update.unclipped_norm,
        applied_scale: update.applied_scale,
    })
}
