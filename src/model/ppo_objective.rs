//! The PPO objective: clipped policy surrogate, value regression, entropy bonus
//! and the candidate KL guard. Every learner path computes its losses here, from
//! targets that already live on the learner device.

use super::*;

/// Heads in [`PolicyTensorTensors`] order with their widths.
const HEADS: [(&str, usize); 12] = [
    ("kind", MODEL_KIND_HEAD),
    ("controlled", MODEL_UNIT_HEAD),
    ("ability", MODEL_ABILITY_HEAD),
    ("item", MODEL_ITEM_HEAD),
    ("swap", MODEL_SWAP_HEAD),
    ("learn", MODEL_LEARN_HEAD),
    ("shop", MODEL_SHOP_HEAD),
    ("loot", MODEL_LOOT_HEAD),
    ("target mode", TARGET_MODE_HEAD),
    ("put mode", PUT_MODE_HEAD),
    ("entity pointer", MODEL_ENTITY_POINTER_HEAD),
    ("point pointer", MODEL_POINT_POINTER_HEAD),
];

/// One head's legal classes, selected label and activity for a batch of rows.
///
/// An inactive head carries a singleton mask, so its normalizer stays finite.
pub(super) struct HeadTargets {
    masks: Tensor,
    labels: Tensor,
    active: Tensor,
}

/// Everything the objective needs besides the model outputs, row aligned.
pub(super) struct ObjectiveTargets {
    heads: Vec<HeadTargets>,
    old_log_probability: Tensor,
    advantages: Tensor,
    returns: Tensor,
    rows: usize,
}

/// Host columns of [`ObjectiveTargets`] before one upload per tensor.
#[derive(Default)]
pub(super) struct HostTargets {
    masks: [Vec<u8>; 12],
    labels: [Vec<u32>; 12],
    active: [Vec<f32>; 12],
    old_log_probability: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

impl HostTargets {
    pub(super) fn with_capacity(rows: usize) -> Self {
        let mut host = Self::default();
        for (index, (_, width)) in HEADS.iter().enumerate() {
            host.masks[index].reserve_exact(rows * width);
            host.labels[index].reserve_exact(rows);
            host.active[index].reserve_exact(rows);
        }
        host.old_log_probability.reserve_exact(rows);
        host.advantages.reserve_exact(rows);
        host.returns.reserve_exact(rows);
        host
    }

    /// Appends one prepared sample, rejecting a label outside its legal mask.
    pub(super) fn push(&mut self, sample: &PpoPreparedSample) -> Result<(), ModelError> {
        let target = &sample.transition.target;
        self.head(0, &target.kind)?;
        self.head(1, &target.controlled)?;
        self.head(2, &target.ability)?;
        self.head(3, &target.item)?;
        self.head(4, &target.swap)?;
        self.head(5, &target.learn)?;
        self.head(6, &target.shop)?;
        self.head(7, &target.loot)?;
        self.head(8, &target.target_mode)?;
        self.head(9, &target.put_mode)?;
        self.head(10, &target.entity_pointer)?;
        self.head(11, &target.point_pointer)?;
        self.old_log_probability
            .push(sample.transition.old_log_probability);
        self.advantages.push(sample.advantage);
        self.returns.push(sample.return_value);
        Ok(())
    }

    fn head<const WIDTH: usize>(
        &mut self,
        index: usize,
        target: &HeadTarget<WIDTH>,
    ) -> Result<(), ModelError> {
        assert_eq!(HEADS[index].1, WIDTH);
        if target.active && !target.mask.get(target.selected).copied().unwrap_or(false) {
            return Err(ModelError::BehavioralTarget {
                head: HEADS[index].0,
                label: target.selected,
            });
        }
        let masks = &mut self.masks[index];
        if target.active {
            masks.extend(target.mask.map(u8::from));
            self.labels[index].push(target.selected as u32);
        } else {
            masks.push(1);
            masks.resize(masks.len() + WIDTH - 1, 0);
            self.labels[index].push(0);
        }
        self.active[index].push(f32::from(target.active));
        Ok(())
    }

    pub(super) fn upload(self, device: &Device) -> Result<ObjectiveTargets, ModelError> {
        let rows = self.returns.len();
        assert!(rows > 0);
        let mut heads = Vec::with_capacity(HEADS.len());
        for (((masks, labels), active), (_, width)) in self
            .masks
            .into_iter()
            .zip(self.labels)
            .zip(self.active)
            .zip(HEADS)
        {
            heads.push(HeadTargets {
                masks: Tensor::from_vec(masks, (rows, width), device)?,
                labels: Tensor::from_vec(labels, (rows, 1), device)?,
                active: Tensor::from_vec(active, rows, device)?,
            });
        }
        Ok(ObjectiveTargets {
            heads,
            old_log_probability: Tensor::from_vec(self.old_log_probability, rows, device)?,
            advantages: Tensor::from_vec(self.advantages, rows, device)?,
            returns: Tensor::from_vec(self.returns, rows, device)?,
            rows,
        })
    }
}

impl ObjectiveTargets {
    /// One head's legal mask and activity.
    #[cfg(test)]
    pub(super) fn head_for_test(&self, index: usize) -> (&Tensor, &Tensor) {
        (&self.heads[index].masks, &self.heads[index].active)
    }

    /// The rows at `indices`, gathered on the device.
    pub(super) fn gather(&self, indices: &Tensor) -> Result<Self, ModelError> {
        let heads = self
            .heads
            .iter()
            .map(|head| {
                Ok(HeadTargets {
                    masks: head.masks.index_select(indices, 0)?,
                    labels: head.labels.index_select(indices, 0)?,
                    active: head.active.index_select(indices, 0)?,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            heads,
            old_log_probability: self.old_log_probability.index_select(indices, 0)?,
            advantages: self.advantages.index_select(indices, 0)?,
            returns: self.returns.index_select(indices, 0)?,
            rows: indices.elem_count(),
        })
    }
}

fn head_logits(output: &PolicyTensorTensors) -> [&Tensor; 12] {
    [
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
    ]
}

/// Log-probability of the behaviour action under the current outputs.
pub(super) fn log_probability(
    output: &PolicyTensorTensors,
    targets: &ObjectiveTargets,
) -> Result<Tensor, ModelError> {
    let mut negative: Option<Tensor> = None;
    for ((logits, head), (_, width)) in head_logits(output)
        .into_iter()
        .zip(&targets.heads)
        .zip(HEADS)
    {
        if logits.dims() != [targets.rows, width] {
            return Err(ModelError::InvalidModelState("PPO head shape"));
        }
        let negative_infinity = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?;
        let legal = head.masks.where_cond(logits, &negative_infinity)?;
        let legal = legal.broadcast_sub(&legal.max_keepdim(1)?.detach())?;
        let selected = legal.gather(&head.labels, 1)?.squeeze(1)?;
        let loss = (legal.log_sum_exp(1)? - selected)?.mul(&head.active)?;
        negative = Some(match negative {
            None => loss,
            Some(total) => (total + loss)?,
        });
    }
    Ok(negative.expect("twelve heads").neg()?)
}

/// Masked entropy of one head; zero where the head is inactive.
pub(super) fn masked_head_entropy(
    logits: &Tensor,
    masks: &Tensor,
    active: &Tensor,
) -> Result<(Tensor, Tensor), ModelError> {
    let device = logits.device();
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), device)?;
    let legal = masks.where_cond(logits, &negative)?;
    let log_normalizer = legal.log_sum_exp(1)?.unsqueeze(1)?;
    let log_probability = legal.broadcast_sub(&log_normalizer)?;
    let zeros = Tensor::zeros(logits.shape(), DType::F32, device)?;
    let safe_log_probability = masks.where_cond(&log_probability, &zeros)?;
    let probability = masks.where_cond(&safe_log_probability.exp()?, &zeros)?;
    let entropy = probability
        .mul(&safe_log_probability)?
        .sum(1)?
        .neg()?
        .mul(active)?;
    Ok((entropy, log_normalizer))
}

fn entropy(output: &PolicyTensorTensors, targets: &ObjectiveTargets) -> Result<Tensor, ModelError> {
    let mut total: Option<Tensor> = None;
    for (logits, head) in head_logits(output).into_iter().zip(&targets.heads) {
        let (entropy, _) = masked_head_entropy(logits, &head.masks, &head.active)?;
        total = Some(match total {
            None => entropy,
            Some(total) => (total + entropy)?,
        });
    }
    Ok(total.expect("twelve heads"))
}

/// Per-row statistics of one objective evaluation, before any readback.
pub(super) struct ObjectiveTerms {
    /// Scalar training loss.
    pub(super) loss: Tensor,
    /// `[policy loss, value loss, entropy, approximate KL, clipped rows]` summed over rows.
    pub(super) sums: Tensor,
}

/// The clipped surrogate, critic regression and entropy bonus of one microbatch,
/// each averaged over the `minibatch_rows` rows of its effective minibatch so
/// that the gradients of the minibatch's microbatches simply add up.
pub(super) fn ppo_loss(
    output: &PolicyTensorTensors,
    targets: &ObjectiveTargets,
    config: PpoConfig,
    minibatch_rows: usize,
) -> Result<ObjectiveTerms, ModelError> {
    assert!(minibatch_rows >= targets.rows);
    let log_ratio = (log_probability(output, targets)? - &targets.old_log_probability)?;
    let ratio = log_ratio.exp()?;
    let clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)?;
    let surrogate = ratio
        .mul(&targets.advantages)?
        .minimum(&clipped_ratio.mul(&targets.advantages)?)?;
    let values = output.value.squeeze(1)?;
    let squared_error = (&values - &targets.returns)?.sqr()?;
    let entropy = entropy(output, targets)?;
    let rows = minibatch_rows as f64;
    let policy_loss = surrogate.sum_all()?.affine(-1.0 / rows, 0.0)?;
    let value_loss = squared_error.sum_all()?.affine(1.0 / rows, 0.0)?;
    let mean_entropy = entropy.sum_all()?.affine(1.0 / rows, 0.0)?;
    let loss = (&policy_loss + &value_loss.affine(f64::from(config.value_coefficient), 0.0)?)?;
    let loss = (&loss - &mean_entropy.affine(f64::from(config.entropy_coefficient), 0.0)?)?;
    let kl = (&ratio.affine(1.0, -1.0)? - &log_ratio)?.detach();
    let outside = (ratio
        .lt(1.0 - f64::from(config.clip_epsilon))?
        .to_dtype(DType::F32)?
        + ratio
            .gt(1.0 + f64::from(config.clip_epsilon))?
            .to_dtype(DType::F32)?)?;
    let sums = Tensor::stack(
        &[
            surrogate.detach().sum_all()?.neg()?,
            squared_error.detach().sum_all()?,
            entropy.detach().sum_all()?,
            kl.sum_all()?,
            outside.sum_all()?,
        ],
        0,
    )?;
    Ok(ObjectiveTerms { loss, sums })
}

/// Summed approximate KL of the current outputs against the behaviour policy.
pub(super) fn candidate_kl_sum(
    output: &PolicyTensorTensors,
    targets: &ObjectiveTargets,
) -> Result<Tensor, ModelError> {
    let log_ratio = (log_probability(output, targets)? - &targets.old_log_probability)?;
    let ratio = log_ratio.exp()?;
    Ok((&ratio.affine(1.0, -1.0)? - &log_ratio)?.sum_all()?)
}

/// Turns summed terms of `rows` rows into the per-row minibatch report.
pub(super) fn report_from_sums(
    sums: &[f32],
    rows: usize,
) -> Result<PpoMinibatchReport, ModelError> {
    let [policy, value, entropy, kl, clipped] = sums else {
        return Err(ModelError::InvalidModelState("PPO objective sums"));
    };
    if sums.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteLoss);
    }
    let rows_f = rows as f64;
    Ok(PpoMinibatchReport {
        policy_loss: f64::from(*policy) / rows_f,
        value_loss: f64::from(*value) / rows_f,
        entropy: f64::from(*entropy) / rows_f,
        approximate_kl: f64::from(*kl) / rows_f,
        clip_fraction: f64::from(*clipped) / rows_f,
        gradient_norm: 0.0,
        applied_scale: 0.0,
        samples: rows,
        applied: false,
    })
}
