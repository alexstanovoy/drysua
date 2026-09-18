use super::*;

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaskedCrossEntropyTestResult {
    pub loss: f32,
    pub gradients: Vec<f32>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdamStepTestResult {
    pub parameters: Vec<f32>,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    pub unclipped_norm: f64,
    pub applied_scale: f64,
}

#[cfg(test)]
pub(crate) struct PolicyTensorSnapshot {
    pub controlled: Vec<f32>,
    pub ability: Vec<f32>,
    pub target_mode: Vec<f32>,
    pub entity_pointer: Vec<f32>,
    pub point_pointer: Vec<f32>,
}

impl PolicyModel {
    #[cfg(test)]
    pub(crate) fn claim_adam_for_test(&self, config: AdamConfig) -> Result<AdamState, ModelError> {
        self.claim_optimizer(config)
    }

    #[cfg(test)]
    pub(crate) fn import_parameters_with_failure(
        &self,
        values: &[f32],
        fail_after: usize,
    ) -> Result<(), ModelError> {
        self.import_parameters_inner(values, Some(fail_after))
    }

    #[cfg(test)]
    pub(crate) fn restore_snapshot_with_failure(
        &self,
        snapshot: &ModelAdamSnapshot,
        adam: &mut AdamState,
        expected: OptimizerBinding,
    ) -> Result<OptimizerBinding, ModelError> {
        self.restore_snapshot_inner(snapshot, adam, expected, Some(0))
    }

    #[cfg(test)]
    pub(crate) fn behavioral_update_with_barrier(
        &self,
        examples: &[&ImitationSample],
        adam: &mut AdamState,
        entered: &std::sync::Barrier,
        release: &std::sync::Barrier,
    ) -> Result<ModelUpdateReport, ModelError> {
        validate_behavioral_examples(examples)?;
        let _guard = self.write_parameter_lock()?;
        entered.wait();
        release.wait();
        self.behavioral_update_locked(examples, adam)
    }

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn training_snapshot(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<PolicyTensorSnapshot, ModelError> {
        let output = self.training_forward(frames, prefixes)?;
        Ok(PolicyTensorSnapshot {
            controlled: output.controlled().flatten_all()?.to_vec1()?,
            ability: output.ability().flatten_all()?.to_vec1()?,
            target_mode: output.target_mode().flatten_all()?.to_vec1()?,
            entity_pointer: output.entity_pointer().flatten_all()?.to_vec1()?,
            point_pointer: output.point_pointer().flatten_all()?.to_vec1()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn gradient_probe(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
    ) -> Result<Vec<(&'static str, bool)>, ModelError> {
        let output = self.training_forward(frames, prefixes)?;
        let loss = output.sum_all_heads()?;
        Ok(self
            .backward_named(&output, &loss)?
            .into_iter()
            .map(|gradient| (gradient.name, gradient.gradient.is_some()))
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn behavioral_loss_for_test(
        &self,
        examples: &[&ImitationSample],
    ) -> Result<f64, ModelError> {
        assert!(!examples.is_empty());
        assert!(examples.len() <= MODEL_TRAINING_BATCH);
        let _guard = self.read_parameter_lock()?;
        let result = self.behavioral_microbatch_locked(examples)?;
        assert!(result.gradients.iter().all(|value| value.is_finite()));
        Ok(result.loss_sum / examples.len() as f64)
    }

    #[cfg(test)]
    pub(crate) fn activation_rms_for_test(
        &self,
        frames: &[FeatureFrame],
    ) -> Result<[f32; 3], ModelError> {
        validate_batch(frames)?;
        let _guard = self.read_parameter_lock()?;
        let state = self.forward_frames(frames)?;
        let mut norms = [0.0; 3];
        for (norm, tensor) in
            norms
                .iter_mut()
                .zip([&state.current_units, &state.points, &state.trunk])
        {
            *norm = tensor.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        }
        Ok(norms)
    }

    #[cfg(test)]
    pub(crate) fn parameter_write_available_for_test(&self) -> bool {
        self.parameter_lock.try_write().is_ok()
    }
}

#[cfg(test)]
pub(crate) struct PpoEntropyProbe {
    pub entropy: Vec<f32>,
    pub gradients: Vec<Vec<f32>>,
    pub log_normalizer: Vec<f32>,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn masked_cross_entropy_for_test(
    logits: &[f32],
    mask: &[bool],
    selected: usize,
    active: bool,
) -> Result<MaskedCrossEntropyTestResult, ModelError> {
    if logits.len() != mask.len() || logits.is_empty() {
        return Err(ModelError::SelectionShape {
            logits: logits.len(),
            mask: mask.len(),
        });
    }
    let variable = Var::from_tensor(&Tensor::from_slice(
        logits,
        (1, logits.len()),
        &Device::Cpu,
    )?)?;
    let target = HeadTarget::<1> {
        active,
        mask: [active],
        selected: 0,
    };
    let dynamic = DynamicTestTarget {
        active: target.active,
        mask,
        selected,
    };
    let loss = dynamic_masked_loss(variable.as_tensor(), dynamic)?;
    let loss_value = loss.to_scalar::<f32>()?;
    let gradients = loss.backward()?;
    let gradient = match gradients.get(variable.as_tensor()) {
        Some(gradient) => gradient.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0.0; logits.len()],
    };
    Ok(MaskedCrossEntropyTestResult {
        loss: loss_value,
        gradients: gradient,
    })
}

#[cfg(test)]
struct DynamicTestTarget<'a> {
    active: bool,
    mask: &'a [bool],
    selected: usize,
}

#[cfg(test)]
fn dynamic_masked_loss(
    logits: &Tensor,
    target: DynamicTestTarget<'_>,
) -> Result<Tensor, ModelError> {
    if target.active && !target.mask.get(target.selected).copied().unwrap_or(false) {
        return Err(ModelError::BehavioralTarget {
            head: "test",
            label: target.selected,
        });
    }
    let mask = if target.active {
        target.mask.iter().copied().map(u8::from).collect()
    } else {
        std::iter::once(1)
            .chain(std::iter::repeat_n(0, target.mask.len() - 1))
            .collect()
    };
    let selected = if target.active { target.selected } else { 0 };
    Ok(masked_loss_from_parts(
        logits,
        1,
        target.mask.len(),
        mask,
        vec![selected as u32],
        vec![if target.active { 1.0 } else { 0.0 }],
    )?
    .sum_all()?)
}

#[cfg(test)]
fn test_mask_tensors(masks: &[Vec<bool>], tokens: usize) -> Result<Vec<Tensor>, ModelError> {
    masks
        .iter()
        .map(|mask| {
            let values = mask
                .iter()
                .map(|value| *value as u8 as f32)
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(values, (1, tokens, 1), &Device::Cpu)?)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn pool_groups_for_test(
    encoded: &[Vec<f32>],
    masks: &[Vec<bool>],
) -> Result<Vec<f32>, ModelError> {
    let tokens = encoded.len();
    let width = encoded.first().map_or(0, Vec::len);
    if tokens == 0 || width == 0 || encoded.iter().any(|row| row.len() != width) {
        return Err(ModelError::InvalidModelState("pool test shape"));
    }
    if masks.iter().any(|mask| mask.len() != tokens) {
        return Err(ModelError::InvalidModelState("pool test mask shape"));
    }
    let values = encoded.iter().flatten().copied().collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (1, tokens, width), &Device::Cpu)?;
    let masks = test_mask_tensors(masks, tokens)?;
    Ok(pool_groups(&tensor, &masks, 1, tokens, width)?
        .flatten_all()?
        .to_vec1()?)
}

#[cfg(test)]
pub(crate) fn pool_max_gradient_for_test(
    encoded: &[Vec<f32>],
    masks: &[Vec<bool>],
) -> Result<Vec<f32>, ModelError> {
    if masks.len() != 1 {
        return Err(ModelError::InvalidModelState("pool gradient group count"));
    }
    let tokens = encoded.len();
    let width = encoded.first().map_or(0, Vec::len);
    if tokens == 0 || width == 0 || encoded.iter().any(|row| row.len() != width) {
        return Err(ModelError::InvalidModelState("pool gradient shape"));
    }
    let values = encoded.iter().flatten().copied().collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values, (1, tokens, width), &Device::Cpu)?;
    let variable = Var::from_tensor(&tensor)?;
    let masks = test_mask_tensors(masks, tokens)?;
    let pooled = pool_groups(variable.as_tensor(), &masks, 1, tokens, width)?;
    let loss = pooled.narrow(1, width, width)?.sum_all()?;
    let gradients = loss.backward()?;
    let gradient = gradients
        .get(variable.as_tensor())
        .ok_or(ModelError::InvalidModelState("pool gradient"))?;
    Ok(gradient.flatten_all()?.to_vec1()?)
}

/// Fixed scripted logits used to verify the real legality decoder.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct DecoderLogits {
    pub kind: [f32; MODEL_KIND_HEAD],
    pub controlled: [f32; MODEL_UNIT_HEAD],
    pub ability: [f32; MODEL_ABILITY_HEAD],
    pub item: [f32; MODEL_ITEM_HEAD],
    pub swap: [f32; MODEL_SWAP_HEAD],
    pub learn: [f32; MODEL_LEARN_HEAD],
    pub shop: [f32; MODEL_SHOP_HEAD],
    pub loot: [f32; MODEL_LOOT_HEAD],
    pub target_mode: [f32; TARGET_MODE_HEAD],
    pub put_mode: [f32; PUT_MODE_HEAD],
    pub entity: [f32; MODEL_ENTITY_POINTER_HEAD],
    pub point: [f32; MODEL_POINT_POINTER_HEAD],
}

#[cfg(test)]
impl DecoderLogits {
    pub(crate) fn favor(kind: ActionKind) -> Self {
        let mut output = Self::default();
        output.kind[kind.index()] = 1.0;
        output
    }
}

#[cfg(test)]
impl Default for DecoderLogits {
    fn default() -> Self {
        Self {
            kind: [0.0; MODEL_KIND_HEAD],
            controlled: [0.0; MODEL_UNIT_HEAD],
            ability: [0.0; MODEL_ABILITY_HEAD],
            item: [0.0; MODEL_ITEM_HEAD],
            swap: [0.0; MODEL_SWAP_HEAD],
            learn: [0.0; MODEL_LEARN_HEAD],
            shop: [0.0; MODEL_SHOP_HEAD],
            loot: [0.0; MODEL_LOOT_HEAD],
            target_mode: [0.0; TARGET_MODE_HEAD],
            put_mode: [0.0; PUT_MODE_HEAD],
            entity: [0.0; MODEL_ENTITY_POINTER_HEAD],
            point: [0.0; MODEL_POINT_POINTER_HEAD],
        }
    }
}

#[cfg(test)]
pub(crate) fn decode_with_logits(
    space: &ActionSpace,
    logits: &DecoderLogits,
) -> Result<StructuredAction, ModelError> {
    let mut source = ScriptedDecoder(logits);
    decode_from_source(space, &mut source)
}

#[cfg(test)]
pub(crate) fn select_target_for_test(
    modes: [f32; 3],
    entities: &[f32],
    points: &[f32],
) -> Result<ActionTarget, ModelError> {
    let entity_mask = vec![true; entities.len()];
    let point_mask = vec![true; points.len()];
    match select_target_mode(&modes, true, &entity_mask, &point_mask)? {
        0 => Ok(ActionTarget::None),
        1 => Ok(ActionTarget::Entity(EntityIndex(masked_argmax(
            entities,
            &entity_mask,
        )?))),
        2 => Ok(ActionTarget::Point(PointIndex(masked_argmax(
            points,
            &point_mask,
        )?))),
        _ => Err(ModelError::InvalidModelState("target mode")),
    }
}

#[cfg(test)]
struct ScriptedDecoder<'a>(&'a DecoderLogits);

#[cfg(test)]
impl DecoderSource for ScriptedDecoder<'_> {
    fn kind(&mut self) -> Result<[f32; 16], ModelError> {
        Ok(self.0.kind)
    }
    fn controlled(&mut self, _: ActionKind) -> Result<[f32; 2], ModelError> {
        Ok(self.0.controlled)
    }
    fn ability(
        &mut self,
        _: ActionKind,
        _: Option<ControlledUnit>,
    ) -> Result<[f32; 8], ModelError> {
        Ok(self.0.ability)
    }
    fn item(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 15], ModelError> {
        Ok(self.0.item)
    }
    fn swap(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 15], ModelError> {
        Ok(self.0.swap)
    }
    fn learn(&mut self, _: ActionKind) -> Result<[f32; 6], ModelError> {
        Ok(self.0.learn)
    }
    fn shop(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 64], ModelError> {
        Ok(self.0.shop)
    }
    fn loot(&mut self, _: ActionKind, _: ControlledUnit) -> Result<[f32; 16], ModelError> {
        Ok(self.0.loot)
    }
    fn target_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: SlotSelection,
    ) -> Result<[f32; 3], ModelError> {
        Ok(self.0.target_mode)
    }
    fn put_mode(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: usize,
    ) -> Result<[f32; 2], ModelError> {
        Ok(self.0.put_mode)
    }
    fn entity(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: Option<SlotSelection>,
    ) -> Result<[f32; 96], ModelError> {
        Ok(self.0.entity)
    }
    fn point(
        &mut self,
        _: ActionKind,
        _: ControlledUnit,
        _: Option<SlotSelection>,
    ) -> Result<[f32; 48], ModelError> {
        Ok(self.0.point)
    }
}
