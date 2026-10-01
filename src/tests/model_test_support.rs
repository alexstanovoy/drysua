use super::*;
use crate::PpoPolicyChoice;

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

    #[cfg(feature = "builtin")]
    pub(crate) fn ppo_update_early_stop_for_test(
        &self,
        examples: &[&PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
    ) -> Result<PpoMinibatchReport, ModelError> {
        let staged = self.stage_ppo_examples(examples)?;
        let indices = (0..examples.len()).collect::<Vec<_>>();
        self.ppo_update_staged(
            &staged,
            &indices,
            adam,
            (config, crate::UpdateObjective::default()),
            (2, crate::KlGuard::EarlyStop),
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
        let output = self.training_forward_inputs(&inputs)?;
        validate_training_tensors_finite(&output)?;
        let log_probability = ppo_objective::log_probability(&output, &inputs.targets)?;
        let terms = ppo_objective::ppo_loss(
            &output,
            None,
            &inputs.targets,
            (PpoConfig::default(), crate::UpdateObjective::default()),
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
    let mut targets = ppo_objective::HostTargets::with_capacity(examples.len(), false);
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

impl PolicyModel {
    /// Samples at most [`MODEL_SAMPLING_BATCH`] rows with one transactional RNG per row.
    pub fn sample_batch(
        &self,
        frames: &[FeatureFrame],
        action_spaces: &[ActionSpace],
        rngs: &mut [PpoRng],
    ) -> Result<Vec<PpoPolicyChoice>, ModelError> {
        validate_policy_batch(frames, action_spaces)?;
        validate_sampling_rng_count(frames.len(), rngs.len())?;
        let rows = packed_rows(frames)?;
        let rows = rows.iter().collect::<Vec<_>>();
        let spaces = action_spaces.iter().collect::<Vec<_>>();
        let statistics = vec![true; frames.len()];
        let _guard = self.read_parameter_lock()?;
        let policy = self.policy_identity_locked();
        let sampled = self.sample_rows_locked(&rows, &spaces, rngs, &statistics)?;
        sampled
            .into_iter()
            .zip(frames)
            .map(|(row, frame)| {
                let statistics = row
                    .statistics
                    .ok_or(ModelError::InvalidModelState("sampled row statistics"))?;
                Ok(PpoPolicyChoice {
                    frame: frame.clone(),
                    target: statistics.target,
                    action: row.action,
                    policy,
                    log_probability: statistics.log_probability,
                    entropy: statistics.entropy,
                    value: row.value,
                })
            })
            .collect()
    }

    /// Exact number of scalar F32 parameters.
    pub const fn parameter_count(&self) -> usize {
        MODEL_PARAMETER_COUNT
    }

    /// Evaluates one frame without mutating model state.
    pub fn evaluate(&self, frame: &FeatureFrame) -> Result<PolicyOutput, ModelError> {
        let mut outputs = self.evaluate_batch(std::slice::from_ref(frame))?;
        outputs
            .pop()
            .ok_or(ModelError::InvalidModelState("single-frame output"))
    }

    /// Evaluates a bounded nonempty batch in input order.
    pub fn evaluate_batch(&self, frames: &[FeatureFrame]) -> Result<Vec<PolicyOutput>, ModelError> {
        validate_batch(frames)?;
        let _guard = self.read_parameter_lock()?;
        let mut output = Vec::with_capacity(frames.len());
        for (chunk_index, chunk) in frames.chunks(MODEL_EVALUATION_MICROBATCH).enumerate() {
            let offset = chunk_index * MODEL_EVALUATION_MICROBATCH;
            output.extend(self.evaluate_chunk(chunk, offset)?);
        }
        Ok(output)
    }

    fn evaluate_chunk(
        &self,
        frames: &[FeatureFrame],
        batch_offset: usize,
    ) -> Result<Vec<PolicyOutput>, ModelError> {
        let routing = ActorRouting::new(frames, self.tensor_device(), false)?;
        let state = self.forward_frames(frames)?;
        let base = self.base_logits(&state, &routing, batch_offset)?;
        collect_outputs(base.value, base.kind, batch_offset)
    }

    /// Samples one legal autoregressive action and records exact old-policy statistics.
    pub fn sample(
        &self,
        frame: &FeatureFrame,
        space: &ActionSpace,
        rng: &mut PpoRng,
    ) -> Result<PpoPolicyChoice, ModelError> {
        if !frame.matches_action_space(space) {
            return Err(ModelError::FrameActionSpaceMismatch);
        }
        validate_batch(std::slice::from_ref(frame))?;
        let _guard = self.read_parameter_lock()?;
        let routing = ActorRouting::new(std::slice::from_ref(frame), self.tensor_device(), false)?;
        let state = self.forward_frames(std::slice::from_ref(frame))?;
        let value = self
            .value
            .forward(&state.trunk)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        validate_value_rows(std::slice::from_ref(&value), 0)?;
        let mut source = ModelDecoder {
            model: self,
            state,
            routing,
            rng: Some(rng),
            observed: Some(SampledPathLogits::default()),
        };
        let action = decode_from_source(space, &mut source)?;
        let target = BehavioralTarget::from_action(frame, space, action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let (log_probability, entropy) = source
            .observed
            .as_ref()
            .ok_or(ModelError::InvalidModelState("sampled path logits"))?
            .statistics(&target)?;
        Ok(PpoPolicyChoice {
            frame: frame.clone(),
            target,
            action,
            policy: self.policy_identity_locked(),
            log_probability,
            entropy,
            value,
        })
    }

    pub fn action_statistics(
        &self,
        frame: &FeatureFrame,
        space: &ActionSpace,
        action: StructuredAction,
    ) -> Result<(f32, f32, f32), ModelError> {
        if !frame.matches_action_space(space) {
            return Err(ModelError::FrameActionSpaceMismatch);
        }
        let target = BehavioralTarget::from_action(frame, space, action)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let _guard = self.read_parameter_lock()?;
        let statistics = self.policy_path_statistics_locked(frame, &target)?;
        Ok((
            statistics.log_probability,
            statistics.entropy,
            statistics.value,
        ))
    }

    fn policy_path_statistics_locked(
        &self,
        frame: &FeatureFrame,
        target: &BehavioralTarget,
    ) -> Result<PolicyPathStatistics, ModelError> {
        let output = self.training_forward_locked(
            std::slice::from_ref(frame),
            std::slice::from_ref(&target.prefix()),
        )?;
        validate_training_tensors_finite(&output)?;
        let value = output.value.flatten_all()?.to_vec1::<f32>()?[0];
        let logits = BehavioralHostLogits::from_tensors(&output)?;
        let (log_probability, entropy) = logits.statistics(0, target)?;
        Ok(PolicyPathStatistics {
            log_probability,
            entropy,
            value,
        })
    }

    /// Evaluates every trainable head for bounded teacher-selected prefixes.
    pub fn training_forward<'model>(
        &'model self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorOutput<'model>, ModelError> {
        validate_training_batch(frames, prefixes)?;
        let guard = self.read_parameter_lock()?;
        let tensors = self.training_forward_locked(frames, prefixes)?;
        validate_training_tensors_finite(&tensors)?;
        Ok(PolicyTensorOutput {
            model_identity: std::ptr::from_ref(self).addr(),
            tensors,
            _parameter_guard: guard,
        })
    }

    /// Every head over one shared trunk; the value loss trains the trunk too.
    pub(super) fn training_forward_locked(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorTensors, ModelError> {
        let routing = ActorRouting::new(frames, self.tensor_device(), true)?;
        let state = self.forward_frames(frames)?;
        let prefixes = PrefixUpload::new(prefixes, self.tensor_device())?;
        self.training_heads(state, routing, &prefixes)
    }

    /// Backpropagates a scalar loss tied to one live guarded training output.
    pub fn backward_named(
        &self,
        output: &PolicyTensorOutput<'_>,
        loss: &Tensor,
    ) -> Result<Vec<NamedPolicyGradient>, ModelError> {
        if output.model_identity != std::ptr::from_ref(self).addr() {
            return Err(ModelError::InvalidModelState(
                "training output of another model",
            ));
        }
        self.backward_named_locked(loss)
    }

    pub(super) fn backward_named_locked(
        &self,
        loss: &Tensor,
    ) -> Result<Vec<NamedPolicyGradient>, ModelError> {
        let gradients = loss.backward()?;
        Ok(self
            .parameters()
            .into_iter()
            .map(|parameter| NamedPolicyGradient {
                name: parameter.name,
                parameter_shape: parameter.value.dims().to_vec(),
                gradient: gradients.get(parameter.value.as_tensor()).cloned(),
            })
            .collect())
    }
}

/// Public value and append-only action-kind logits for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyOutput {
    /// Unbounded scalar state-value prediction.
    pub value: f32,
    /// Logits in append-only [`ActionKind`] order.
    pub kind_logits: [f32; MODEL_KIND_HEAD],
}

impl PolicyOutput {
    /// Whether the value and every action-kind logit are finite.
    pub fn is_finite(&self) -> bool {
        self.value.is_finite() && self.kind_logits.iter().all(|value| value.is_finite())
    }
}

/// Autograd-preserving output holding one complete parameter read session.
pub struct PolicyTensorOutput<'model> {
    model_identity: usize,
    pub(super) tensors: PolicyTensorTensors,
    _parameter_guard: RwLockReadGuard<'model, ()>,
}

impl fmt::Debug for PolicyTensorOutput<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolicyTensorOutput")
            .field("shapes", &self.shapes())
            .finish_non_exhaustive()
    }
}

/// Exact tensor dimensions returned by [`PolicyModel::training_forward`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyTensorShapes {
    /// State-value tensor dimensions.
    pub value: Vec<usize>,
    /// Action-kind tensor dimensions.
    pub kind: Vec<usize>,
    /// Controlled-unit tensor dimensions.
    pub controlled: Vec<usize>,
    /// Ability-slot tensor dimensions.
    pub ability: Vec<usize>,
    /// Item-slot tensor dimensions.
    pub item: Vec<usize>,
    /// Swap-destination tensor dimensions.
    pub swap: Vec<usize>,
    /// Learn-slot tensor dimensions.
    pub learn: Vec<usize>,
    /// Shop tensor dimensions.
    pub shop: Vec<usize>,
    /// Loot tensor dimensions.
    pub loot: Vec<usize>,
    /// Target-mode tensor dimensions.
    pub target_mode: Vec<usize>,
    /// Put-mode tensor dimensions.
    pub put_mode: Vec<usize>,
    /// Entity-pointer tensor dimensions.
    pub entity_pointer: Vec<usize>,
    /// Point-pointer tensor dimensions.
    pub point_pointer: Vec<usize>,
}

impl PolicyTensorOutput<'_> {
    /// Shape `[batch, 1]` state values.
    pub const fn value(&self) -> &Tensor {
        &self.tensors.value
    }

    /// Shape `[batch, 16]` action-kind logits.
    pub const fn kind(&self) -> &Tensor {
        &self.tensors.kind
    }

    /// Shape `[batch, 2]` controlled-unit logits.
    pub const fn controlled(&self) -> &Tensor {
        &self.tensors.controlled
    }

    /// Shape `[batch, 8]` ability-slot logits.
    pub const fn ability(&self) -> &Tensor {
        &self.tensors.ability
    }

    /// Shape `[batch, 15]` item or source-slot logits.
    pub const fn item(&self) -> &Tensor {
        &self.tensors.item
    }

    /// Shape `[batch, 15]` swap-destination logits.
    pub const fn swap(&self) -> &Tensor {
        &self.tensors.swap
    }

    /// Shape `[batch, 6]` learn-slot logits.
    pub const fn learn(&self) -> &Tensor {
        &self.tensors.learn
    }

    /// Shape `[batch, 64]` shop logits.
    pub const fn shop(&self) -> &Tensor {
        &self.tensors.shop
    }

    /// Shape `[batch, 16]` loot logits.
    pub const fn loot(&self) -> &Tensor {
        &self.tensors.loot
    }

    /// Shape `[batch, 3]` None, Entity, and Point mode logits.
    pub const fn target_mode(&self) -> &Tensor {
        &self.tensors.target_mode
    }

    /// Shape `[batch, 2]` Underfoot and Point mode logits.
    pub const fn put_mode(&self) -> &Tensor {
        &self.tensors.put_mode
    }

    /// Shape `[batch, 96]` current-unit pointer logits.
    pub const fn entity_pointer(&self) -> &Tensor {
        &self.tensors.entity_pointer
    }

    /// Shape `[batch, 64]` point-candidate pointer logits.
    pub const fn point_pointer(&self) -> &Tensor {
        &self.tensors.point_pointer
    }

    /// Returns every head shape without converting tensor values to host storage.
    pub fn shapes(&self) -> PolicyTensorShapes {
        PolicyTensorShapes {
            value: self.value().dims().to_vec(),
            kind: self.kind().dims().to_vec(),
            controlled: self.controlled().dims().to_vec(),
            ability: self.ability().dims().to_vec(),
            item: self.item().dims().to_vec(),
            swap: self.swap().dims().to_vec(),
            learn: self.learn().dims().to_vec(),
            shop: self.shop().dims().to_vec(),
            loot: self.loot().dims().to_vec(),
            target_mode: self.target_mode().dims().to_vec(),
            put_mode: self.put_mode().dims().to_vec(),
            entity_pointer: self.entity_pointer().dims().to_vec(),
            point_pointer: self.point_pointer().dims().to_vec(),
        }
    }

    /// Checks every tensor value while preserving the existing autograd graph.
    pub fn validate_finite(&self) -> Result<(), ModelError> {
        side_actors::validate_training(&self.tensors)
    }

    /// Sums all heads into one scalar graph-connected probe loss.
    pub fn sum_all_heads(&self) -> Result<Tensor, ModelError> {
        sum_training_tensors(&self.tensors)
    }
}

/// One gradient in stable parameter export order.
pub struct NamedPolicyGradient {
    pub(crate) name: &'static str,
    pub(crate) parameter_shape: Vec<usize>,
    pub(crate) gradient: Option<Tensor>,
}

impl NamedPolicyGradient {
    /// Stable parameter name covered by the model schema descriptor.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Parameter dimensions expected by an optimizer update.
    pub fn parameter_shape(&self) -> &[usize] {
        &self.parameter_shape
    }

    /// Read-only gradient tensor, absent when the loss did not use this parameter.
    pub const fn gradient(&self) -> Option<&Tensor> {
        self.gradient.as_ref()
    }

    /// Gradient dimensions, absent when the parameter was outside the loss graph.
    pub fn gradient_shape(&self) -> Option<&[usize]> {
        self.gradient.as_ref().map(Tensor::dims)
    }
}

impl fmt::Debug for NamedPolicyGradient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NamedPolicyGradient")
            .field("name", &self.name)
            .field("parameter_shape", &self.parameter_shape)
            .field("gradient_shape", &self.gradient_shape())
            .finish()
    }
}

struct PolicyPathStatistics {
    log_probability: f32,
    entropy: f32,
    value: f32,
}

struct BehavioralHostLogits {
    kind: Vec<Vec<f32>>,
    controlled: Vec<Vec<f32>>,
    ability: Vec<Vec<f32>>,
    item: Vec<Vec<f32>>,
    swap: Vec<Vec<f32>>,
    learn: Vec<Vec<f32>>,
    shop: Vec<Vec<f32>>,
    loot: Vec<Vec<f32>>,
    target_mode: Vec<Vec<f32>>,
    put_mode: Vec<Vec<f32>>,
    entity_pointer: Vec<Vec<f32>>,
    point_pointer: Vec<Vec<f32>>,
}

impl BehavioralHostLogits {
    fn from_tensors(output: &PolicyTensorTensors) -> Result<Self, ModelError> {
        Ok(Self {
            kind: output.kind.to_vec2()?,
            controlled: output.controlled.to_vec2()?,
            ability: output.ability.to_vec2()?,
            item: output.item.to_vec2()?,
            swap: output.swap.to_vec2()?,
            learn: output.learn.to_vec2()?,
            shop: output.shop.to_vec2()?,
            loot: output.loot.to_vec2()?,
            target_mode: output.target_mode.to_vec2()?,
            put_mode: output.put_mode.to_vec2()?,
            entity_pointer: output.entity_pointer.to_vec2()?,
            point_pointer: output.point_pointer.to_vec2()?,
        })
    }

    fn statistics(
        &self,
        index: usize,
        target: &BehavioralTarget,
    ) -> Result<(f32, f32), ModelError> {
        macro_rules! add_head {
            ($logp:ident, $entropy:ident, $values:ident, $field:ident) => {
                let (head_logp, head_entropy) =
                    host_head_statistics(self.row(&self.$values, index)?, &target.$field)?;
                $logp += head_logp;
                $entropy += head_entropy;
            };
        }
        let (mut log_probability, mut entropy) =
            host_head_statistics(self.row(&self.kind, index)?, &target.kind)?;
        add_head!(log_probability, entropy, controlled, controlled);
        add_head!(log_probability, entropy, ability, ability);
        add_head!(log_probability, entropy, item, item);
        add_head!(log_probability, entropy, swap, swap);
        add_head!(log_probability, entropy, learn, learn);
        add_head!(log_probability, entropy, shop, shop);
        add_head!(log_probability, entropy, loot, loot);
        add_head!(log_probability, entropy, target_mode, target_mode);
        add_head!(log_probability, entropy, put_mode, put_mode);
        add_head!(log_probability, entropy, entity_pointer, entity_pointer);
        add_head!(log_probability, entropy, point_pointer, point_pointer);
        if !log_probability.is_finite() || !entropy.is_finite() || entropy < 0.0 {
            return Err(ModelError::InvalidModelState("policy path statistics"));
        }
        Ok((log_probability, entropy))
    }

    fn row<'a>(&self, values: &'a [Vec<f32>], index: usize) -> Result<&'a [f32], ModelError> {
        values
            .get(index)
            .map(Vec::as_slice)
            .ok_or(ModelError::InvalidModelState(
                "behavioral output batch shape",
            ))
    }
}

fn collect_outputs(
    values: Vec<f32>,
    kinds: Vec<Vec<f32>>,
    batch_offset: usize,
) -> Result<Vec<PolicyOutput>, ModelError> {
    if values.len() != kinds.len() {
        return Err(ModelError::InvalidModelState("batch output shape"));
    }
    let mut output = Vec::with_capacity(values.len());
    for (batch, (value, logits)) in values.into_iter().zip(kinds).enumerate() {
        if !value.is_finite() {
            return Err(ModelError::NonFiniteOutput {
                field: "value",
                batch: batch_offset + batch,
                index: 0,
            });
        }
        if let Some((index, _)) = logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ModelError::NonFiniteOutput {
                field: "kind",
                batch: batch_offset + batch,
                index,
            });
        }
        let kind_logits = logits
            .try_into()
            .map_err(|_| ModelError::InvalidModelState("kind head shape"))?;
        output.push(PolicyOutput { value, kind_logits });
    }
    Ok(output)
}

/// Frames per host evaluation graph.
pub(crate) const MODEL_EVALUATION_MICROBATCH: usize = 64;

fn validate_training_batch(
    frames: &[FeatureFrame],
    prefixes: &[TrainingPrefix],
) -> Result<(), ModelError> {
    if frames.is_empty() {
        return Err(ModelError::EmptyTrainingBatch);
    }
    if frames.len() > MODEL_TRAINING_BATCH {
        return Err(ModelError::BatchTooLarge {
            count: frames.len(),
            maximum: MODEL_TRAINING_BATCH,
        });
    }
    assert_eq!(prefixes.len(), frames.len());
    if let Some(index) = frames.iter().position(|frame| !frame.is_finite()) {
        return Err(ModelError::NonFiniteFrame { index });
    }
    side_actors::validate_sides(frames)?;
    Ok(())
}

fn sum_training_tensors(output: &PolicyTensorTensors) -> Result<Tensor, ModelError> {
    let tensors = [
        &output.kind,
        &output.controlled,
        &output.ability,
        &output.item,
        &output.swap,
        &output.learn,
        &output.shop,
        &output.loot,
        &output.target_mode,
        &output.put_mode,
        &output.entity_pointer,
        &output.point_pointer,
    ];
    let mut loss = output.value.sum_all()?;
    for tensor in tensors {
        loss = (loss + tensor.sum_all()?)?;
    }
    Ok(loss)
}
