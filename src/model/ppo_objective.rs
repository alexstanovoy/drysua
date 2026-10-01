//! The PPO objective: clipped policy surrogate, value regression, entropy bonus,
//! the fading imitation term and the candidate KL guard. Every learner path
//! computes its losses here, from targets that already live on the learner device.

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
    /// Shadow labels, staged only for an update that imitates; unlabeled rows
    /// have every head inactive.
    shadow: Option<Vec<HeadTargets>>,
    old_log_probability: Tensor,
    advantages: Tensor,
    returns: Tensor,
    /// One on dire rows, zero on radiant rows.
    dire: Tensor,
    rows: usize,
}

/// Host columns of one label per row over the twelve heads.
#[derive(Default)]
struct HostHeads {
    masks: [Vec<u8>; 12],
    labels: [Vec<u32>; 12],
    active: [Vec<f32>; 12],
}

/// Host columns of [`ObjectiveTargets`] before one upload per tensor.
#[derive(Default)]
pub(super) struct HostTargets {
    heads: HostHeads,
    shadow: Option<HostHeads>,
    old_log_probability: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    dire: Vec<f32>,
}

impl HostTargets {
    pub(super) fn with_capacity(rows: usize, imitation: bool) -> Self {
        let mut host = Self {
            heads: HostHeads::with_capacity(rows),
            shadow: imitation.then(|| HostHeads::with_capacity(rows)),
            ..Self::default()
        };
        host.old_log_probability.reserve_exact(rows);
        host.advantages.reserve_exact(rows);
        host.returns.reserve_exact(rows);
        host.dire.reserve_exact(rows);
        host
    }

    /// Appends one prepared sample of the given side, rejecting a label outside its legal mask.
    pub(super) fn push(
        &mut self,
        sample: &PpoPreparedSample,
        radiant: bool,
    ) -> Result<(), ModelError> {
        self.heads.push(&sample.transition.target)?;
        if let Some(shadow) = &mut self.shadow {
            match &sample.transition.shadow {
                Some(label) => shadow.push(label)?,
                None => shadow.push_unlabeled(),
            }
        }
        self.old_log_probability
            .push(sample.transition.old_log_probability);
        self.advantages.push(sample.advantage);
        self.returns.push(sample.return_value);
        self.dire.push(if radiant { 0.0 } else { 1.0 });
        Ok(())
    }

    pub(super) fn upload(self, device: &Device) -> Result<ObjectiveTargets, ModelError> {
        let rows = self.returns.len();
        assert!(rows > 0);
        Ok(ObjectiveTargets {
            heads: self.heads.upload(rows, device)?,
            shadow: self
                .shadow
                .map(|shadow| shadow.upload(rows, device))
                .transpose()?,
            old_log_probability: Tensor::from_vec(self.old_log_probability, rows, device)?,
            advantages: Tensor::from_vec(self.advantages, rows, device)?,
            returns: Tensor::from_vec(self.returns, rows, device)?,
            dire: Tensor::from_vec(self.dire, rows, device)?,
            rows,
        })
    }
}

impl HostHeads {
    fn with_capacity(rows: usize) -> Self {
        let mut host = Self::default();
        for (index, (_, width)) in HEADS.iter().enumerate() {
            host.masks[index].reserve_exact(rows * width);
            host.labels[index].reserve_exact(rows);
            host.active[index].reserve_exact(rows);
        }
        host
    }

    fn push(&mut self, target: &ActionHeadTargets) -> Result<(), ModelError> {
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
        Ok(())
    }

    /// A row without a label: every head inactive.
    fn push_unlabeled(&mut self) {
        for (index, (_, width)) in HEADS.iter().enumerate() {
            let masks = &mut self.masks[index];
            masks.push(1);
            masks.resize(masks.len() + width - 1, 0);
            self.labels[index].push(0);
            self.active[index].push(0.0);
        }
    }

    fn head<const WIDTH: usize>(
        &mut self,
        index: usize,
        target: &HeadTarget<WIDTH>,
    ) -> Result<(), ModelError> {
        assert_eq!(HEADS[index].1, WIDTH);
        if target.active && !target.mask.get(target.selected).copied().unwrap_or(false) {
            return Err(ModelError::ActionHeadTargets {
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

    fn upload(self, rows: usize, device: &Device) -> Result<Vec<HeadTargets>, ModelError> {
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
        Ok(heads)
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
        let gather = |heads: &[HeadTargets]| {
            heads
                .iter()
                .map(|head| {
                    Ok(HeadTargets {
                        masks: head.masks.index_select(indices, 0)?,
                        labels: head.labels.index_select(indices, 0)?,
                        active: head.active.index_select(indices, 0)?,
                    })
                })
                .collect::<Result<Vec<_>, ModelError>>()
        };
        Ok(Self {
            heads: gather(&self.heads)?,
            shadow: self.shadow.as_deref().map(gather).transpose()?,
            old_log_probability: self.old_log_probability.index_select(indices, 0)?,
            advantages: self.advantages.index_select(indices, 0)?,
            returns: self.returns.index_select(indices, 0)?,
            dire: self.dire.index_select(indices, 0)?,
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
        let (_, loss) = head_cross_entropy(logits, head)?;
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

/// Per-row masked entropy of every head, in [`HEADS`] order.
fn head_entropies(
    output: &PolicyTensorTensors,
    targets: &ObjectiveTargets,
) -> Result<Vec<Tensor>, ModelError> {
    head_logits(output)
        .into_iter()
        .zip(&targets.heads)
        .map(|(logits, head)| Ok(masked_head_entropy(logits, &head.masks, &head.active)?.0))
        .collect()
}

/// Loss and row-summed statistics of one objective evaluation, before any readback.
pub(super) struct ObjectiveTerms {
    /// Scalar training loss.
    pub(super) loss: Tensor,
    /// `[policy loss, value loss, entropy, approximate KL, clipped rows]`, the
    /// [`IMITATION_SUMS`] imitation statistics and the [`SIDE_SUMS`] side
    /// statistics, summed over rows.
    pub(super) sums: Tensor,
}

/// Imitation statistics per evaluation: cross entropy, labeled rows, rows agreeing
/// on every labeled head, then per head the agreeing and the labeled rows.
const IMITATION_SUMS: usize = 3 + 2 * HEADS.len();

/// Side statistics per evaluation, over all rows and then over dire rows:
/// rows, policy loss, value loss, entropy, approximate KL from the gradient's
/// own forward, clipped rows, imitation cross entropy and labeled rows; then the
/// entropy of every head over all rows and over dire rows.
const SIDE_SUMS: usize = 2 * (crate::SIDE_QUANTITIES + HEADS.len());

/// The clipped surrogate, critic regression, entropy bonus and imitation term of
/// one microbatch, each averaged over the `minibatch_rows` rows of its effective
/// minibatch so that the gradients of the minibatch's microbatches simply add up.
///
/// `shadow` holds the heads conditioned on the shadow labels' own prefixes and
/// is present exactly when the update imitates. A critic-only update trains the
/// value regression alone; its training forward detaches the trunk under the
/// value head, so no gradient reaches the trunk.
pub(super) fn ppo_loss(
    output: &PolicyTensorTensors,
    shadow: Option<&PolicyTensorTensors>,
    targets: &ObjectiveTargets,
    (config, objective): (PpoConfig, crate::UpdateObjective),
    minibatch_rows: usize,
) -> Result<ObjectiveTerms, ModelError> {
    assert!(minibatch_rows >= targets.rows);
    assert_eq!(shadow.is_some(), crate::ppo::imitates(objective));
    let log_ratio = (log_probability(output, targets)? - &targets.old_log_probability)?;
    let ratio = log_ratio.exp()?;
    let clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)?;
    let surrogate = ratio
        .mul(&targets.advantages)?
        .minimum(&clipped_ratio.mul(&targets.advantages)?)?;
    let values = output.value.squeeze(1)?;
    let squared_error = (&values - &targets.returns)?.sqr()?;
    let heads = head_entropies(output, targets)?;
    let entropy = heads
        .iter()
        .skip(1)
        .try_fold(heads[0].clone(), |total, head| total + head)?;
    let rows = minibatch_rows as f64;
    let policy_loss = surrogate.sum_all()?.affine(-1.0 / rows, 0.0)?;
    let value_loss = squared_error.sum_all()?.affine(1.0 / rows, 0.0)?;
    let mean_entropy = entropy.sum_all()?.affine(1.0 / rows, 0.0)?;
    let critic = value_loss.affine(f64::from(config.value_coefficient), 0.0)?;
    let mut loss = if objective.critic_only {
        critic
    } else {
        let loss = (&policy_loss + &critic)?;
        (&loss - &mean_entropy.affine(f64::from(config.entropy_coefficient), 0.0)?)?
    };
    let zeros = Tensor::zeros(targets.rows, DType::F32, output.value.device())?;
    let (imitation, imitation_rows) = match shadow {
        Some(shadow) => {
            let heads = targets
                .shadow
                .as_deref()
                .ok_or(ModelError::InvalidModelState(
                    "imitation without shadow labels",
                ))?;
            let (cross_entropy, sums) = imitation_terms(shadow, heads)?;
            let mean = cross_entropy.sum_all()?.affine(1.0 / rows, 0.0)?;
            loss = (&loss + &mean.affine(f64::from(objective.imitation), 0.0)?)?;
            (sums, [cross_entropy.detach(), heads[0].active.clone()])
        }
        None => (
            Tensor::zeros(IMITATION_SUMS, DType::F32, output.value.device())?,
            [zeros.clone(), zeros],
        ),
    };
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
    let quantities = [
        Tensor::ones(targets.rows, DType::F32, output.value.device())?,
        surrogate.detach().neg()?,
        squared_error.detach(),
        entropy.detach(),
        kl,
        outside,
    ]
    .into_iter()
    .chain(imitation_rows)
    .chain(heads.iter().map(Tensor::detach))
    .collect::<Vec<_>>();
    let mut side = Vec::with_capacity(SIDE_SUMS);
    for mask in [None, Some(&targets.dire)] {
        for quantity in &quantities {
            side.push(match mask {
                None => quantity.sum_all()?,
                Some(dire) => quantity.mul(dire)?.sum_all()?,
            });
        }
    }
    assert_eq!(side.len(), SIDE_SUMS);
    let sums = Tensor::cat(&[sums, imitation, Tensor::stack(&side, 0)?], 0)?;
    Ok(ObjectiveTerms { loss, sums })
}

/// One head's legal logits (shifted by their detached row maximum; illegal
/// classes are -inf) and the per-row cross entropy to its label, zero where the
/// head is inactive.
fn head_cross_entropy(logits: &Tensor, head: &HeadTargets) -> Result<(Tensor, Tensor), ModelError> {
    let negative_infinity = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?;
    let legal = head.masks.where_cond(logits, &negative_infinity)?;
    let legal = legal.broadcast_sub(&legal.max_keepdim(1)?.detach())?;
    let selected = legal.gather(&head.labels, 1)?.squeeze(1)?;
    let loss = (legal.log_sum_exp(1)? - selected)?.mul(&head.active)?;
    Ok((legal, loss))
}

/// Per-row cross entropy to the shadow labels over every head a label defines,
/// and the [`IMITATION_SUMS`] statistics of the legal argmax agreeing with them.
fn imitation_terms(
    shadow: &PolicyTensorTensors,
    heads: &[HeadTargets],
) -> Result<(Tensor, Tensor), ModelError> {
    assert_eq!(heads.len(), HEADS.len());
    let mut cross_entropy: Option<Tensor> = None;
    let mut misses: Option<Tensor> = None;
    let mut agreements = Vec::with_capacity(HEADS.len());
    let mut labels = Vec::with_capacity(HEADS.len());
    for ((logits, head), (_, width)) in head_logits(shadow).into_iter().zip(heads).zip(HEADS) {
        if logits.dim(1)? != width {
            return Err(ModelError::InvalidModelState("imitation head shape"));
        }
        let (legal, loss) = head_cross_entropy(logits, head)?;
        let agree = legal
            .detach()
            .argmax(1)?
            .eq(&head.labels.squeeze(1)?)?
            .to_dtype(DType::F32)?
            .mul(&head.active)?;
        let miss = (&head.active - &agree)?;
        agreements.push(agree.sum_all()?);
        labels.push(head.active.sum_all()?);
        cross_entropy = Some(match cross_entropy {
            None => loss,
            Some(total) => (total + loss)?,
        });
        misses = Some(match misses {
            None => miss,
            Some(total) => (total + miss)?,
        });
    }
    let cross_entropy = cross_entropy.expect("twelve heads");
    // Every label activates the kind head.
    let labeled = &heads[0].active;
    let whole = misses
        .expect("twelve heads")
        .eq(0.0)?
        .to_dtype(DType::F32)?
        .mul(labeled)?;
    let mut sums = vec![
        cross_entropy.detach().sum_all()?,
        labeled.sum_all()?,
        whole.sum_all()?,
    ];
    sums.extend(agreements);
    sums.extend(labels);
    assert_eq!(sums.len(), IMITATION_SUMS);
    Ok((cross_entropy, Tensor::stack(&sums, 0)?))
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

/// Turns summed terms of `rows` rows into the per-row minibatch report; the
/// imitation statistics stay sums.
pub(super) fn report_from_sums(
    sums: &[f32],
    rows: usize,
) -> Result<PpoMinibatchReport, ModelError> {
    if sums.len() != 5 + IMITATION_SUMS + SIDE_SUMS {
        return Err(ModelError::InvalidModelState("PPO objective sums"));
    }
    if sums.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteLoss);
    }
    let (ppo, imitation) = sums.split_at(5);
    let (imitation, side) = imitation.split_at(IMITATION_SUMS);
    let [policy, value, entropy, kl, clipped] = ppo else {
        unreachable!("five PPO sums");
    };
    let heads = HEADS.len();
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
        imitation: crate::ImitationReport {
            cross_entropy: f64::from(imitation[0]),
            labeled: f64::from(imitation[1]),
            action_agreements: f64::from(imitation[2]),
            head_agreements: std::array::from_fn(|head| f64::from(imitation[3 + head])),
            head_labels: std::array::from_fn(|head| f64::from(imitation[3 + heads + head])),
        },
        sides: crate::SideReport::from_sums(side),
    })
}
