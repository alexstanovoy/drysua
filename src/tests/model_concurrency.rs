use super::*;

#[test]
fn folded_ppo_preserves_parameters_moments_reports_and_rejection() {
    assert_folded_ppo(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires authorized CUDA runner"]
fn cuda_folded_ppo_preserves_parameters_moments_reports_and_rejection() {
    assert_folded_ppo(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_folded_ppo(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let reference = PolicyModel::fresh_on(9101, device).expect("reference");
    let config = transfer_ppo_config();
    let mut actual_adam = model.claim_optimizer(config.adam()).expect("Adam");
    let mut expected_adam = reference
        .claim_optimizer(config.adam())
        .expect("reference Adam");
    let mut samples = transfer_ppo_samples(&model);
    for target_kl in [1000.0, 1.0e-12] {
        refresh_old_probabilities(&model, &mut samples);
        let examples = samples.iter().collect::<Vec<_>>();
        let config = PpoConfig {
            target_kl,
            ..config
        };
        let expected = reference
            .ppo_update_with_microbatch(
                &examples,
                &mut expected_adam,
                config,
                2,
                PpoTestFaults::default(),
            )
            .expect("reference update");
        let actual = model
            .ppo_update_with_microbatch_and_workers(
                &examples,
                &mut actual_adam,
                config,
                2,
                2,
                PpoTestFaults::default(),
            )
            .expect("folded update");
        assert_report_bits(actual, expected);
        assert_state_bits(&model, &actual_adam, &reference, &expected_adam);
    }
}

#[test]
fn prefetched_updates_preserve_shuffle_parameters_and_optimizer_state() {
    let config = PpoConfig {
        environments: 1,
        rollout_decisions: 3,
        minibatch: 2,
        epochs: 2,
        target_kl: 1000.0,
        ..PpoConfig::default()
    };
    let model = PolicyModel::fresh(9101).expect("model");
    let reference = PolicyModel::fresh(9101).expect("reference");
    let mut actual = crate::PpoTrainer::new(&model, config, 19).expect("trainer");
    let mut expected = crate::PpoTrainer::new(&reference, config, 19).expect("reference trainer");
    actual
        .set_execution(crate::TrainingExecutionOptions {
            learner_prefetch: true,
            host_math_workers: 2,
            ..Default::default()
        })
        .expect("options");
    let first = learner_batch(&model, config);
    let second = learner_batch(&reference, config);
    let source = expected.train_update(&reference, &second).expect("serial");
    let target = actual.train_update(&model, &first).expect("prefetched");
    assert_eq!(source, target);
    assert_eq!(actual.rng_checkpoint(), expected.rng_checkpoint());
    assert_snapshot_bits(
        &actual.checkpoint_snapshot(&model).expect("actual snapshot"),
        &expected
            .checkpoint_snapshot(&reference)
            .expect("expected snapshot"),
    );
}

fn learner_batch(model: &PolicyModel, config: PpoConfig) -> crate::PpoBatch {
    let mut samples = transfer_ppo_samples(model);
    refresh_old_probabilities(model, &mut samples);
    let mut rollout =
        crate::PpoRollout::new(3, model.policy_identity().expect("identity")).expect("rollout");
    for sample in samples {
        rollout.push(sample.transition).expect("transition");
    }
    rollout.finish(config).expect("batch")
}

#[test]
fn prefetch_ignores_unused_error_on_kl_stop_and_rolls_back_consumed_error() {
    for reject in [false, true] {
        let config = PpoConfig {
            environments: 1,
            rollout_decisions: 3,
            minibatch: 2,
            epochs: 2,
            target_kl: if reject { 0.02 } else { 1000.0 },
            ..PpoConfig::default()
        };
        let model = PolicyModel::fresh(9101).expect("model");
        let reference = PolicyModel::fresh(9101).expect("reference");
        let mut actual = crate::PpoTrainer::new(&model, config, 19).expect("trainer");
        let mut expected = crate::PpoTrainer::new(&reference, config, 19).expect("trainer");
        actual
            .set_execution(crate::TrainingExecutionOptions {
                learner_prefetch: true,
                ..Default::default()
            })
            .expect("options");
        let mut first = learner_batch(&model, config);
        let mut second = learner_batch(&reference, config);
        let mut order = vec![0, 1, 2];
        PpoRng::new(19)
            .shuffle(&mut order)
            .expect("reference order");
        for batch in [&mut first, &mut second] {
            batch.corrupt_prefetch_frame_for_test(order[2]);
            if reject {
                batch.reject_prefetch_minibatch_for_test(order[0]);
            }
        }
        let source = expected.train_update(&reference, &second);
        let target = actual.train_update(&model, &first);
        if reject {
            let report = target.expect("unused prefetch error is discarded");
            assert!(report.stopped_for_kl);
            assert_eq!(report, source.expect("serial KL stop"));
        } else {
            let message = "invalid PPO transition: ragged feature range is invalid";
            assert_eq!(
                source
                    .expect_err("serial materialization error")
                    .to_string(),
                message
            );
            assert_eq!(
                target
                    .expect_err("prefetch materialization error")
                    .to_string(),
                message
            );
            assert_eq!(actual.rng_checkpoint(), PpoRng::new(19).checkpoint());
        }
        assert_eq!(actual.rng_checkpoint(), expected.rng_checkpoint());
        assert_snapshot_bits(
            &actual.checkpoint_snapshot(&model).expect("snapshot"),
            &expected.checkpoint_snapshot(&reference).expect("snapshot"),
        );
    }
}

#[test]
fn previous_fold_error_outranks_a_later_microbatch_frame_error() {
    let model = PolicyModel::fresh(9101).expect("model");
    let reference = PolicyModel::fresh(9101).expect("reference");
    let config = transfer_ppo_config();
    let mut first = model.claim_optimizer(config.adam()).expect("Adam");
    let mut second = reference.claim_optimizer(config.adam()).expect("Adam");
    let mut samples = transfer_ppo_samples(&model);
    samples[0].advantage = f32::NAN;
    samples[2].transition.frame.global[0] = f32::NAN;
    let examples = samples.iter().collect::<Vec<_>>();
    let expected = reference
        .ppo_update_with_microbatch(&examples, &mut second, config, 2, PpoTestFaults::default())
        .expect_err("serial gradient error");
    let actual = model
        .ppo_update_with_microbatch_and_workers(
            &examples,
            &mut first,
            config,
            2,
            2,
            PpoTestFaults::default(),
        )
        .expect_err("fold error before later frame error");
    assert!(matches!(expected, ModelError::NonFiniteGradient { .. }));
    assert_eq!(actual.to_string(), expected.to_string());
    assert_state_bits(&model, &first, &reference, &second);
}
