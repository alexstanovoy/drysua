use super::*;

#[test]
fn training_microbatch_default_retains_exact_legacy_64_update() {
    let model = PolicyModel::fresh(9101).expect("model");
    let reference = PolicyModel::fresh(9101).expect("reference");
    let config = transfer_ppo_config();
    let mut actual = model.claim_optimizer(config.adam()).expect("Adam");
    let mut expected = reference.claim_optimizer(config.adam()).expect("Adam");
    let samples = samples(&model, 65, 64);
    let examples = samples.iter().collect::<Vec<_>>();
    let report = model
        .ppo_update_with_execution(
            &examples,
            &mut actual,
            config,
            crate::TrainingExecutionOptions::default(),
        )
        .expect("default");
    let legacy = reference
        .ppo_update(&examples, &mut expected, config)
        .expect("legacy 64");
    assert_report_bits(report, legacy);
    assert_state_bits(&model, &actual, &reference, &expected);
}

#[test]
fn larger_training_microbatches_replay_exactly_and_restore_rejected_candidates() {
    for microbatch in [128, 256] {
        assert_mode(PolicyDevice::Cpu, microbatch, microbatch + 1);
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires the owner's bounded CUDA runner"]
fn cuda_training_microbatch_modes_preserve_within_mode_state_and_uneven_tails() {
    for (microbatch, rows) in [(64, 65), (128, 129), (256, 257), (256, 2049)] {
        assert_mode(PolicyDevice::Cuda { ordinal: 0 }, microbatch, rows);
    }
}

#[test]
fn internal_training_bound_is_256_but_public_forward_and_actor_stay_64() {
    let model = PolicyModel::fresh(9101).expect("model");
    let frames = vec![FeatureFrame::new(); 257];
    let prefixes = vec![TrainingPrefix::new(ActionKind::Continue, None, None); 257];
    assert_eq!(
        validate_ppo_training_batch(&frames, &prefixes)
            .expect_err("257 bound")
            .to_string(),
        "model training batch count 257 exceeds maximum 256"
    );
    assert_eq!(
        model
            .training_forward(&frames[..65], &prefixes[..65])
            .expect_err("public 64")
            .to_string(),
        "model training batch count 65 exceeds maximum 64"
    );
    assert_eq!(MODEL_TRAINING_BATCH, 64);
    assert_eq!(MODEL_EVALUATION_MICROBATCH, 64);
    assert_eq!(MODEL_PPO_MAX_MICROBATCH, 256);
}

#[test]
fn internal_training_checks_finite_frames_and_all_unused_output_heads() {
    let model = PolicyModel::fresh(9101).expect("model");
    let mut samples = samples(&model, 129, 128);
    samples[128].transition.frame.global[0] = f32::NAN;
    let examples = samples.iter().collect::<Vec<_>>();
    let error = model
        .ppo_microbatch_locked(&examples, transfer_ppo_config())
        .err()
        .expect("finite frame");
    assert_eq!(
        error.to_string(),
        "model frame 128 contains a non-finite value"
    );
    samples[128].transition.frame.global[0] = 0.0;
    samples[0].transition.target.kind.selected = MODEL_KIND_HEAD;
    let examples = samples.iter().collect::<Vec<_>>();
    assert_eq!(
        model
            .ppo_candidate_kl_locked(&examples, transfer_ppo_config(), 256, false)
            .expect_err("NLL target remains checked")
            .to_string(),
        "model behavioral target label 16 is illegal for head kind"
    );
    samples[0].transition.target.kind.selected = 0;
    let bad =
        Tensor::full(f32::NAN, MODEL_ITEM_HEAD, model.tensor_device()).expect("bad unused head");
    model.item_head.bias.set(&bad).expect("inject");
    let examples = samples.iter().collect::<Vec<_>>();
    let error = model
        .ppo_candidate_kl_locked(&examples, transfer_ppo_config(), 128, false)
        .expect_err("all 13 outputs checked");
    assert_eq!(
        error.to_string(),
        "model item output at batch 0 index 0 is non-finite"
    );
}

fn samples(model: &PolicyModel, count: usize, microbatch: usize) -> Vec<PpoPreparedSample> {
    assert!((1..=2049).contains(&count));
    assert!((1..=MODEL_PPO_MAX_MICROBATCH).contains(&microbatch));
    let base = transfer_ppo_samples(model);
    let mut samples = (0..count)
        .map(|index| base[index % base.len()].clone())
        .collect::<Vec<_>>();
    for chunk in samples.chunks_mut(microbatch) {
        let frames = chunk
            .iter()
            .map(|sample| sample.transition.frame.clone())
            .collect::<Vec<_>>();
        let prefixes = chunk
            .iter()
            .map(|sample| sample.transition.target.prefix())
            .collect::<Vec<_>>();
        let output = model
            .training_forward_locked(&frames, &prefixes)
            .expect("within-mode forward");
        let examples = chunk.iter().collect::<Vec<_>>();
        let values = ppo_negative_log_probability(&output, &examples)
            .expect("NLL")
            .neg()
            .expect("log probability")
            .to_vec1::<f32>()
            .expect("values");
        assert_eq!(values.len(), chunk.len());
        for (sample, value) in chunk.iter_mut().zip(values) {
            sample.transition.old_log_probability = value;
        }
    }
    samples
}

fn assert_mode(device: PolicyDevice, microbatch: usize, rows: usize) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let reference = PolicyModel::fresh_on(9101, device).expect("reference");
    let config = transfer_ppo_config();
    let mut actual = model.claim_optimizer(config.adam()).expect("Adam");
    let mut expected = reference.claim_optimizer(config.adam()).expect("Adam");
    for scenario in 0..3 {
        let mut samples = samples(&model, rows, microbatch);
        // Warm nonzero critic moments without saturating the actor before its rejection trial.
        if scenario == 0 {
            for sample in &mut samples {
                sample.advantage = 0.0;
            }
        }
        let examples = samples.iter().collect::<Vec<_>>();
        let before = model.coherent_snapshot(&actual).expect("before");
        let result = mode_step(&model, &mut actual, &examples, microbatch, scenario, false);
        let replay = mode_step(
            &reference,
            &mut expected,
            &examples,
            microbatch,
            scenario,
            true,
        );
        if scenario == 2 {
            let message = format!(
                "model tensor operation failed: injected PPO candidate evaluation failure after {microbatch} rows"
            );
            assert_eq!(result.expect_err("candidate failure").to_string(), message);
            assert_eq!(replay.expect_err("reference failure").to_string(), message);
        } else {
            let report = result.expect("selected mode");
            assert_report_bits(report, replay.expect("explicit mode replay"));
            assert_eq!(report.samples, rows);
            assert_eq!(report.applied, scenario == 0);
        }
        assert_state_bits(&model, &actual, &reference, &expected);
        if scenario != 0 {
            assert_snapshot_bits(
                &model.coherent_snapshot(&actual).expect("after rollback"),
                &before,
            );
            assert_eq!(actual.binding, before.adam.binding);
        }
    }
    assert_eq!(actual.step(), 1);
    assert!(actual.first_moment.iter().any(|value| *value != 0.0));
}

fn mode_step(
    model: &PolicyModel,
    adam: &mut AdamState,
    examples: &[&PpoPreparedSample],
    microbatch: usize,
    scenario: usize,
    reference: bool,
) -> Result<PpoMinibatchReport, ModelError> {
    let config = PpoConfig {
        target_kl: if scenario == 1 { 1.0e-12 } else { 1000.0 },
        entropy_coefficient: if scenario == 0 { 0.0 } else { 0.01 },
        ..transfer_ppo_config()
    };
    if scenario == 2 || reference {
        let faults = PpoTestFaults {
            candidate_evaluation: scenario == 2,
            rollback_import: false,
        };
        model.ppo_update_with_microbatch(examples, adam, config, microbatch, faults)
    } else {
        model.ppo_update_with_execution(
            examples,
            adam,
            config,
            crate::TrainingExecutionOptions {
                training_microbatch: microbatch,
                ..Default::default()
            },
        )
    }
}
