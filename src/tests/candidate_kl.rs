use super::*;
use std::cell::Cell;

thread_local! { static FULL_LOSS: Cell<bool> = const { Cell::new(false) }; }

pub(in crate::model) fn evaluate(
    output: &PolicyTensorTensors,
    examples: &[&PpoPreparedSample],
    config: PpoConfig,
) -> Result<f64, ModelError> {
    if FULL_LOSS.get() {
        ppo_loss(output, examples, config).map(|(_, report)| report.approximate_kl)
    } else {
        ppo_candidate_kl(output, examples)
    }
}

fn with_full_loss<T>(operation: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FULL_LOSS.set(self.0);
        }
    }
    let _reset = Reset(FULL_LOSS.replace(true));
    operation()
}

#[test]
fn candidate_kl_matches_full_loss_bits_at_microbatch_boundaries() {
    assert_chunk_parity(PolicyDevice::Cpu);
}

#[test]
fn candidate_kl_preserves_bad_label_and_unused_head_errors() {
    assert_candidate_errors(PolicyDevice::Cpu);
}

#[test]
fn candidate_kl_preserves_accepted_rejected_and_failed_update_state() {
    assert_update_parity(PolicyDevice::Cpu);
}

#[test]
fn candidate_kl_preserves_trainer_reports_shuffle_and_optimizer_bits() {
    for target_kl in [1000.0, 1.0e-12] {
        let model = PolicyModel::fresh(9101).expect("model");
        let reference = PolicyModel::fresh(9101).expect("reference");
        let config = PpoConfig {
            environments: 1,
            rollout_decisions: 3,
            minibatch: 2,
            epochs: 2,
            target_kl,
            ..transfer_ppo_config()
        };
        let mut actual = crate::PpoTrainer::new(&model, config, 19).expect("trainer");
        let mut expected = crate::PpoTrainer::new(&reference, config, 19).expect("trainer");
        let source = trainer_batch(&reference, config);
        let target = trainer_batch(&model, config);
        let report = actual
            .train_update(&model, &target)
            .expect("KL-only update");
        let original = with_full_loss(|| expected.train_update(&reference, &source))
            .expect("full loss update");
        assert_eq!(report, original);
        for (a, b) in [
            (report.policy_loss, original.policy_loss),
            (report.value_loss, original.value_loss),
            (report.entropy, original.entropy),
            (report.approximate_kl, original.approximate_kl),
            (report.rejected_kl, original.rejected_kl),
            (report.clip_fraction, original.clip_fraction),
            (report.gradient_norm, original.gradient_norm),
            (report.applied_scale, original.applied_scale),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(actual.rng_checkpoint(), expected.rng_checkpoint());
        assert_snapshot_bits(
            &actual.checkpoint_snapshot(&model).expect("snapshot"),
            &expected.checkpoint_snapshot(&reference).expect("snapshot"),
        );
        assert_eq!(
            model.policy_identity().expect("identity").revision(),
            reference.policy_identity().expect("identity").revision()
        );
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires an authorized CUDA runner"]
fn cuda_candidate_kl_matches_full_loss_and_transaction_bits() {
    let device = PolicyDevice::Cuda { ordinal: 0 };
    assert_chunk_parity(device);
    assert_candidate_errors(device);
    assert_update_parity(device);
}

fn candidate(model: &PolicyModel, samples: &[PpoPreparedSample]) -> Result<f64, ModelError> {
    let examples = samples.iter().collect::<Vec<_>>();
    let _guard = model.read_parameter_lock()?;
    let config = transfer_ppo_config();
    model.ppo_candidate_kl_locked(&examples, config, MODEL_TRAINING_BATCH, false)
}

fn assert_chunk_parity(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let base = transfer_ppo_samples(&model);
    for count in [1, 64, 65, 81] {
        let mut samples = (0..count)
            .map(|index| base[index % base.len()].clone())
            .collect::<Vec<_>>();
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.transition.target.kind.mask[ActionKind::Stop.index()] = index % 3 != 0;
            sample.transition.target.kind.mask[ActionKind::Hold.index()] = index % 3 == 2;
            sample.transition.target.validate().expect("legal target");
            assert!(!sample.transition.target.item.active);
            sample.transition.old_log_probability = -0.25 * (index % 7) as f32;
        }
        let expected = with_full_loss(|| candidate(&model, &samples)).expect("full loss KL");
        let actual = candidate(&model, &samples).expect("KL-only");
        assert_eq!(actual.to_bits(), expected.to_bits(), "rows={count}");
    }
}

fn assert_candidate_errors(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let mut samples = transfer_ppo_samples(&model);
    samples.truncate(1);
    samples[0].transition.target.kind.selected = MODEL_KIND_HEAD;
    assert_candidate_error(
        &model,
        &samples,
        "model behavioral target label 16 is illegal for head kind",
    );
    samples[0].transition.target.kind.selected = 0;
    let nonfinite =
        Tensor::full(f32::NAN, MODEL_ITEM_HEAD, model.tensor_device()).expect("nonfinite head");
    model
        .item_head
        .bias
        .set(&nonfinite)
        .expect("inject unused head");
    assert!(!samples[0].transition.target.item.active);
    let message = "model radiant.item output at batch 0 index 0 is non-finite";
    assert_candidate_error(&model, &samples, message);
}

fn assert_candidate_error(model: &PolicyModel, samples: &[PpoPreparedSample], message: &str) {
    for result in [
        candidate(model, samples),
        with_full_loss(|| candidate(model, samples)),
    ] {
        assert_eq!(
            result.expect_err("candidate validation").to_string(),
            message
        );
    }
}

fn assert_update_parity(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let reference = PolicyModel::fresh_on(9101, device).expect("reference");
    let config = transfer_ppo_config();
    let mut actual = model.claim_optimizer(config.adam()).expect("Adam");
    let mut expected = reference.claim_optimizer(config.adam()).expect("Adam");
    let base = transfer_ppo_samples(&model);
    let mut samples = (0..65)
        .map(|index| base[index % base.len()].clone())
        .collect::<Vec<_>>();
    for scenario in 0..3 {
        refresh(&model, &mut samples);
        let examples = samples.iter().collect::<Vec<_>>();
        let before = model.coherent_snapshot(&actual).expect("before");
        let identity = model.policy_identity().expect("identity");
        let reference_identity = reference.policy_identity().expect("reference identity");
        let target_kl = if scenario == 1 {
            1.0e-12
        } else {
            config.target_kl
        };
        let config = PpoConfig {
            target_kl,
            ..config
        };
        let faults = PpoTestFaults {
            candidate_evaluation: scenario == 2,
            rollback_import: false,
        };
        let result = model.ppo_update_with_microbatch(&examples, &mut actual, config, 64, faults);
        let original = with_full_loss(|| {
            reference.ppo_update_with_microbatch(&examples, &mut expected, config, 64, faults)
        });
        if scenario == 2 {
            let message = "model tensor operation failed: injected PPO candidate evaluation failure after 64 rows";
            assert_eq!(result.expect_err("candidate error").to_string(), message);
            assert_eq!(original.expect_err("full-loss error").to_string(), message);
        } else {
            let report = result.expect("update");
            assert_eq!(report.applied, scenario == 0);
            assert_report_bits(report, original.expect("full-loss update"));
        }
        assert_state_bits(&model, &actual, &reference, &expected);
        if scenario != 0 {
            assert_snapshot_bits(
                &model.coherent_snapshot(&actual).expect("rollback"),
                &before,
            );
            assert_eq!(actual.binding, before.adam.binding);
            assert_eq!(model.policy_identity().expect("identity"), identity);
            assert_eq!(
                reference.policy_identity().expect("identity"),
                reference_identity
            );
        }
    }
    assert_eq!(actual.step(), 1);
    assert!(actual.first_moment.iter().any(|value| *value != 0.0));
}

fn refresh(model: &PolicyModel, samples: &mut [PpoPreparedSample]) {
    for chunk in samples.chunks_mut(MODEL_TRAINING_BATCH) {
        let examples = chunk.iter().collect::<Vec<_>>();
        let (values, _) = model
            .ppo_likelihood_for_test(&examples)
            .expect("likelihood");
        for (sample, value) in chunk.iter_mut().zip(values) {
            sample.transition.old_log_probability = value;
        }
    }
}

fn trainer_batch(model: &PolicyModel, config: PpoConfig) -> crate::PpoBatch {
    let mut samples = transfer_ppo_samples(model);
    refresh_old_probabilities(model, &mut samples);
    let mut rollout =
        crate::PpoRollout::new(3, model.policy_identity().expect("identity")).expect("rollout");
    for (index, mut sample) in samples.into_iter().enumerate() {
        sample.transition.reward = if index == 0 { 1.0 } else { -0.5 };
        rollout.push(sample.transition).expect("transition");
    }
    rollout.finish(config).expect("batch")
}
