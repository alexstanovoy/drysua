use super::*;

struct ImitationGradient {
    gradients: Vec<f32>,
    negative_log_probability: Vec<f32>,
    loss: f64,
}

impl PolicyModel {
    pub(crate) fn self_imitation_update(
        &self,
        samples: &[PpoPreparedSample],
        adam: &mut AdamState,
        config: PpoConfig,
        coefficient: f32,
    ) -> Result<PpoMinibatchReport, ModelError> {
        self.imitation_step(samples, adam, config.target_kl, coefficient, false)
    }

    fn imitation_step(
        &self,
        samples: &[PpoPreparedSample],
        adam: &mut AdamState,
        target_kl: f32,
        coefficient: f32,
        inject_failure: bool,
    ) -> Result<PpoMinibatchReport, ModelError> {
        validate_replay(samples, coefficient)?;
        let _guard = self.write_parameter_lock()?;
        self.validate_optimizer_binding_locked(adam.binding)?;
        let mut report = PpoMinibatchReport {
            samples: samples.len(),
            ..Default::default()
        };
        if coefficient == 0.0 {
            return Ok(report);
        }
        let frames = samples
            .iter()
            .map(|sample| sample.transition.frame.clone())
            .collect::<Vec<_>>();
        let prefixes = samples
            .iter()
            .map(|sample| sample.transition.target.prefix())
            .collect::<Vec<_>>();
        validate_training_batch(&frames, &prefixes)?;
        let references = samples.iter().collect::<Vec<_>>();
        let Some(gradient) =
            self.imitation_gradients(&frames, &prefixes, &references, coefficient)?
        else {
            return Ok(report);
        };
        report.policy_loss = gradient.loss;
        let original = self.export_parameters_locked()?;
        let original_adam = adam.clone();
        let diagnostics = self.apply_adam_locked(adam, &gradient.gradients, &original)?;
        let candidate_kl = (|| {
            self.restore_imitation_value_head(&original, &original_adam, adam)?;
            if inject_failure {
                return Err(ModelError::Backend(
                    "injected self-imitation candidate evaluation failure".to_owned(),
                ));
            }
            self.imitation_candidate_kl(
                &frames,
                &prefixes,
                &references,
                &gradient.negative_log_probability,
            )
        })();
        if !candidate_kl
            .as_ref()
            .is_ok_and(|kl| *kl <= f64::from(target_kl))
        {
            self.rollback_ppo_candidate_locked(
                &original,
                original_adam,
                adam,
                &candidate_kl,
                false,
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

    fn imitation_gradients(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
        samples: &[&PpoPreparedSample],
        coefficient: f32,
    ) -> Result<Option<ImitationGradient>, ModelError> {
        let output = self.training_forward_value_gradient_locked(frames, prefixes, false)?;
        validate_training_tensors_finite(&output)?;
        // Host readback detaches the baseline and caps subtraction before converting to f32.
        let weights = samples
            .iter()
            .zip(output.value.flatten_all()?.to_vec1::<f32>()?)
            .map(|(sample, value)| {
                (f64::from(sample.return_value) - f64::from(value)).clamp(0.0, 1.0) as f32
            })
            .collect::<Vec<_>>();
        if weights.iter().all(|weight| *weight == 0.0) {
            return Ok(None);
        }
        let negative_log_probability = ppo_negative_log_probability(&output, samples)?;
        let before = negative_log_probability.to_vec1::<f32>()?;
        let weights = Tensor::from_vec(weights, samples.len(), self.tensor_device())?;
        let loss = negative_log_probability
            .mul(&weights)?
            .mean_all()?
            .affine(f64::from(coefficient), 0.0)?;
        let value = f64::from(loss.to_scalar::<f32>()?);
        if !value.is_finite() {
            return Err(ModelError::InvalidModelState("self-imitation loss"));
        }
        let gradients = collect_host_gradients(self.backward_named_locked(&loss)?)?;
        Ok(Some(ImitationGradient {
            gradients,
            negative_log_probability: before,
            loss: value,
        }))
    }

    fn imitation_candidate_kl(
        &self,
        frames: &[FeatureFrame],
        prefixes: &[TrainingPrefix],
        samples: &[&PpoPreparedSample],
        before: &[f32],
    ) -> Result<f64, ModelError> {
        assert_eq!(samples.len(), before.len());
        assert!(!samples.is_empty());
        let output = self.training_forward_value_gradient_locked(frames, prefixes, false)?;
        validate_training_tensors_finite(&output)?;
        let after = ppo_negative_log_probability(&output, samples)?.to_vec1::<f32>()?;
        let kl = before
            .iter()
            .zip(after)
            .map(|(old, new)| {
                let log_ratio = f64::from(*old) - f64::from(new);
                log_ratio.exp_m1() - log_ratio
            })
            .sum::<f64>()
            / samples.len() as f64;
        if !kl.is_finite() {
            return Err(ModelError::InvalidModelState("self-imitation candidate KL"));
        }
        Ok(kl)
    }

    fn restore_imitation_value_head(
        &self,
        original: &[f32],
        original_adam: &AdamState,
        adam: &mut AdamState,
    ) -> Result<(), ModelError> {
        assert_eq!(original.len(), MODEL_PARAMETER_COUNT);
        assert_eq!(adam.step(), original_adam.step() + 1);
        let mut offset = 0;
        for parameter in self.parameters() {
            let end = offset + parameter.value.elem_count();
            if matches!(parameter.name, "value.weight" | "value.bias") {
                // Zero gradients still move dense Adam coordinates with existing momentum.
                parameter.value.set(&Tensor::from_vec(
                    original[offset..end].to_vec(),
                    parameter.value.shape().clone(),
                    self.tensor_device(),
                )?)?;
                adam.first_moment[offset..end]
                    .copy_from_slice(&original_adam.first_moment[offset..end]);
                adam.second_moment[offset..end]
                    .copy_from_slice(&original_adam.second_moment[offset..end]);
            }
            offset = end;
        }
        Ok(())
    }
}

fn validate_replay(samples: &[PpoPreparedSample], coefficient: f32) -> Result<(), ModelError> {
    if samples.is_empty() || samples.len() > MODEL_TRAINING_BATCH {
        return Err(ModelError::InvalidModelState("self-imitation sample count"));
    }
    if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
        return Err(ModelError::InvalidModelState("self-imitation coefficient"));
    }
    for sample in samples {
        let transition = &sample.transition;
        if !sample.return_value.is_finite() || !transition.frame.is_finite() {
            return Err(ModelError::InvalidModelState(
                "self-imitation return or frame",
            ));
        }
        let target = &transition.target;
        target
            .validate()
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        if target
            .reconstruct_action()
            .map_err(|error| ModelError::Backend(error.to_string()))?
            != transition.action
        {
            return Err(ModelError::InvalidModelState(
                "self-imitation action target",
            ));
        }
    }
    Ok(())
}

#[test]
fn self_imitation_positive_return_improves_action_and_zero_weight_is_exact_noop() {
    let model = PolicyModel::fresh(9_428).expect("model");
    let samples = replay_samples(&model);
    let mut trainer = crate::PpoTrainer::new(&model, PpoConfig::default(), 42).expect("trainer");
    let value_bias = model
        .parameters()
        .iter()
        .take_while(|parameter| parameter.name != "value.bias")
        .map(|parameter| parameter.value.elem_count())
        .sum::<usize>();
    let mut seeded = trainer
        .checkpoint_snapshot(&model)
        .expect("seed critic momentum");
    seeded.adam.first_moment[value_bias] = 0.125;
    seeded.adam.second_moment[value_bias] = 0.125;
    trainer = crate::PpoTrainer::restore_checkpoint(
        trainer.config(),
        seeded.adam,
        trainer.rng_checkpoint(),
        0,
    )
    .expect("existing critic momentum");
    let references = samples.iter().collect::<Vec<_>>();
    let before = model
        .ppo_likelihood_for_test(&references)
        .expect("before")
        .0;
    let random = trainer.rng_checkpoint();
    let disabled = trainer
        .checkpoint_snapshot(&model)
        .expect("disabled snapshot");
    let report = trainer
        .self_imitation_update(&model, &samples, 1.0)
        .expect("imitation");
    assert!(report.applied);
    assert!((report.policy_loss + f64::from(before[0])).abs() < 1.0e-6);
    assert!(report.approximate_kl <= f64::from(trainer.config().target_kl));
    assert!(model.ppo_likelihood_for_test(&references).expect("after").0[0] > before[0]);
    assert_eq!(trainer.optimizer_step(), 1);
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.rng_checkpoint(), random);
    let snapshot = trainer.checkpoint_snapshot(&model).expect("snapshot");
    assert_eq!(
        snapshot.parameters[value_bias],
        disabled.parameters[value_bias]
    );
    assert_eq!(snapshot.adam.first_moment[value_bias], 0.125);
    assert_eq!(snapshot.adam.second_moment[value_bias], 0.125);
    assert_imitation_noop(&model, &mut trainer, &samples);
}

fn assert_imitation_noop(
    model: &PolicyModel,
    trainer: &mut crate::PpoTrainer,
    samples: &[PpoPreparedSample],
) {
    let snapshot = trainer.checkpoint_snapshot(model).expect("snapshot");
    let random = trainer.rng_checkpoint();
    for (coefficient, return_value) in [(0.0, 10.0), (1.0, -f32::MAX)] {
        let mut samples = samples.to_vec();
        for sample in &mut samples {
            sample.return_value = return_value;
        }
        let report = trainer
            .self_imitation_update(model, &samples, coefficient)
            .expect("noop");
        assert!(!report.applied);
        assert_eq!(
            trainer.checkpoint_snapshot(model).expect("unchanged"),
            snapshot
        );
        assert_eq!(trainer.rng_checkpoint(), random);
    }
}

#[test]
fn self_imitation_kl_rejection_restores_parameters_moments_step_and_revision() {
    let model = PolicyModel::fresh(9_428).expect("model");
    let samples = replay_samples(&model);
    let mut trainer = crate::PpoTrainer::new(&model, PpoConfig::default(), 42).expect("trainer");
    assert!(
        trainer
            .self_imitation_update(&model, &samples, 1.0)
            .expect("seed moments")
            .applied
    );
    let snapshot = trainer.checkpoint_snapshot(&model).expect("before");
    let random = trainer.rng_checkpoint();
    let config = PpoConfig {
        target_kl: f32::MIN_POSITIVE,
        ..PpoConfig::default()
    };
    let mut trainer =
        crate::PpoTrainer::restore_checkpoint(config, snapshot.adam.clone(), random, 0)
            .expect("restrictive guard");
    let report = trainer
        .self_imitation_update(&model, &samples, 1.0)
        .expect("rejected");
    assert!(!report.applied);
    assert!(report.approximate_kl > f64::from(config.target_kl));
    assert_eq!(
        trainer.checkpoint_snapshot(&model).expect("restored"),
        snapshot
    );
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.updates(), 0);
    let mut adam = snapshot.adam.clone();
    let error = model
        .imitation_step(&samples, &mut adam, config.target_kl, 1.0, true)
        .expect_err("candidate evaluation failure");
    assert_eq!(
        error.to_string(),
        "model tensor operation failed: injected self-imitation candidate evaluation failure"
    );
    assert_eq!(
        model.coherent_snapshot(&adam).expect("error rollback"),
        snapshot
    );
}

fn replay_samples(model: &PolicyModel) -> Vec<PpoPreparedSample> {
    let tracker = replay_tracker();
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut frame = FeatureFrame::new();
    let mut encoder = crate::FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    encoder
        .encode(
            &tracker,
            &space,
            &crate::ItemReadiness::new(),
            &crate::LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("frame");
    let mut target =
        BehavioralTarget::from_action(&frame, &space, StructuredAction::Continue).expect("target");
    // Match the existing transfer fixture's two-choice masked learner probe.
    target.kind.mask[ActionKind::Stop.index()] = true;
    vec![PpoPreparedSample {
        transition: crate::PpoTransition {
            frame,
            target,
            action: StructuredAction::Continue,
            policy: model.policy_identity().expect("identity"),
            stream: 0,
            decision: 0,
            ticks: 3,
            old_log_probability: -10.0,
            old_value: 0.0,
            next_value: 0.0,
            reward: 0.0,
            terminal: true,
        },
        advantage: 0.0,
        return_value: 10.0,
    }]
}

fn replay_tracker() -> crate::StateTracker {
    use bota_proto::{MapId, MatchInfo, Pick, SlotId, Team, TickMode};
    let info = MatchInfo {
        match_id: 1,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 0,
        trees: Vec::new(),
        terrain_cells: 1,
        terrain_rle: vec![(1, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![Pick {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: crate::SHADOW_FIEND,
        }],
        shop: Vec::new(),
    };
    let mut tracker = crate::StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&bota_proto::WorldView {
            tick: 1,
            viewer: Some(Team::Radiant),
            units: Vec::new(),
            projectiles: Vec::new(),
            players: vec![bota_proto::PlayerView {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: crate::SHADOW_FIEND,
                unit: None,
                level: 1,
                xp: 0,
                gold: Some(0),
                stash: Some(vec![None; 6]),
                kit: None,
                kills: 0,
                deaths: 0,
                assists: 0,
                last_hits: 0,
                denies: 0,
                respawn_left: 1,
            }],
            felled_trees: Vec::new(),
            planted_trees: Vec::new(),
            loot: Vec::new(),
        })
        .expect("snapshot");
    tracker
}
