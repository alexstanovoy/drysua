#![allow(
    clippy::float_arithmetic,
    reason = "model-output tolerance checks use f32 arithmetic"
)]

use std::collections::BTreeSet;
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::thread;

use bota_proto::{Aim, SlotId, Team, UnitKind};

use super::feature::{encode, reverse_entity_ids_and_generations, tracker_with_view, world_view};
use crate::model::{
    DecoderLogits, adam_step_for_test, decode_with_logits, masked_argmax,
    masked_cross_entropy_for_test, masked_ppo_entropy_for_test, pool_groups_for_test,
    pool_max_gradient_for_test, select_target_for_test, unit_group, validate_batch_count,
};
use crate::{
    ActionKind, ActionSpace, ActionTarget, AdamConfig, ControlledUnit, FeatureFrame, HeadTarget,
    ImitationSample, LocalPolicyState, MODEL_ABILITY_HEAD, MODEL_ENTITY_POINTER_HEAD,
    MODEL_EVALUATION_MICROBATCH, MODEL_ITEM_HEAD, MODEL_KIND_HEAD, MODEL_LEARN_HEAD,
    MODEL_LOOT_HEAD, MODEL_MAX_BATCH, MODEL_PARAMETER_COUNT, MODEL_POINT_POINTER_HEAD,
    MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, MODEL_SHOP_HEAD, MODEL_SWAP_HEAD,
    MODEL_TRAINING_BATCH, MODEL_UNIT_HEAD, ModelError, PolicyDevice, PolicyModel, PpoOutcome,
    PpoPreparedSample, PpoRng, PutPointTarget, SampleIdentity, SeedNamespace, StructuredAction,
    TrainingAbilitySlot, TrainingItemSlot, TrainingPrefix, TrainingSlot, unit_feature,
};

#[path = "model_input_adapter.rs"]
mod model_input_adapter;

#[test]
fn fresh_policy_fixed_corpus_has_unsaturated_heads() {
    assert_fresh_policy_conditioning(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_fresh_policy_fixed_corpus_has_unsaturated_heads() {
    assert_fresh_policy_conditioning(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_fresh_policy_conditioning(device: PolicyDevice) {
    let prefixes = mixed_training_prefixes();
    let mut frames = vec![populated_frame(); prefixes.len()];
    frames[1] = FeatureFrame::new();
    frames[2].items[0][crate::item_feature::SLOT_TOKEN] = 64.0;
    frames[2].items[0][crate::item_feature::ITEM_TOKEN] = 65_536.0;
    frames[2].abilities[0][crate::ability_feature::ID_TOKEN] = 65_547.0;
    let mut conditioned = true;
    for seed in [1, 503, 9_101] {
        let model = PolicyModel::fresh_on(seed, device).expect("model");
        let norms = model.activation_rms_for_test(&frames).expect("norms");
        eprintln!("seed={seed} unit_point_trunk_rms={norms:?}");
        assert!(
            norms
                .iter()
                .all(|norm| norm.is_finite() && *norm > 0.0 && *norm < 1.0)
        );
        let output = model.training_forward(&frames, &prefixes).expect("forward");
        assert_pointer_scale_and_gradient(output.kind().device());
        for (name, tensor) in [
            ("kind", output.kind()),
            ("controlled", output.controlled()),
            ("ability", output.ability()),
            ("item", output.item()),
            ("swap", output.swap()),
            ("learn", output.learn()),
            ("shop", output.shop()),
            ("loot", output.loot()),
            ("target_mode", output.target_mode()),
            ("put_mode", output.put_mode()),
            ("entity", output.entity_pointer()),
            ("point", output.point_pointer()),
        ] {
            for row in tensor.to_vec2::<f32>().expect("logits") {
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let minimum = row.iter().copied().fold(f32::INFINITY, f32::min);
                let sum = row.iter().map(|value| (value - maximum).exp()).sum::<f32>();
                let entropy = row
                    .iter()
                    .map(|value| {
                        let log_probability = value - maximum - sum.ln();
                        -log_probability.exp() * log_probability
                    })
                    .sum::<f32>();
                eprintln!(
                    "seed={seed} head={name} spread={} entropy={entropy} uniform={}",
                    maximum - minimum,
                    (row.len() as f32).ln()
                );
                conditioned &= maximum - minimum < 2.0;
                conditioned &= entropy > 0.95 * (row.len() as f32).ln();
            }
        }
    }
    assert!(conditioned, "fresh policy heads must start near uniform");
}

#[test]
fn full_model_overfits_fixed_batch_and_learns_unseen_observations() {
    assert_fixed_batch_learning(PolicyDevice::Cpu);
}

fn assert_pointer_scale_and_gradient(device: &candle_core::Device) {
    for width in [64, 128] {
        let tokens = candle_core::Tensor::ones((1, 2, width), candle_core::DType::F32, device)
            .expect("tokens");
        let query =
            candle_core::Var::ones((1, 1, width), candle_core::DType::F32, device).expect("query");
        let scores = crate::model::scaled_pointer_dot(&tokens, query.as_tensor()).expect("pointer");
        for score in scores
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("scores")
        {
            assert!((score - (width as f32).sqrt()).abs() < 1.0e-6);
        }
        let gradients = scores.sum_all().expect("sum").backward().expect("backward");
        let gradient = gradients.get(query.as_tensor()).expect("query gradient");
        for value in gradient
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("gradient")
        {
            assert!((value - 2.0 / (width as f32).sqrt()).abs() < 1.0e-6);
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_full_model_overfits_fixed_batch_and_learns_unseen_observations() {
    assert_fixed_batch_learning(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_fixed_batch_learning(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(503, device).expect("model");
    let training = conditioning_learning_corpus(false);
    let held_out = conditioning_learning_corpus(true);
    let training = training.iter().collect::<Vec<_>>();
    let held_out = held_out.iter().collect::<Vec<_>>();
    let before = model
        .behavioral_loss_for_test(&training)
        .expect("initial loss");
    let held_before = model
        .behavioral_loss_for_test(&held_out)
        .expect("held initial loss");
    let mut adam = model
        .claim_adam_for_test(AdamConfig {
            learning_rate: 0.003,
            ..AdamConfig::default()
        })
        .expect("Adam");
    assert_ne!(
        training[0].frame().global()[20],
        training[1].frame().global()[20]
    );
    for _ in 0..192 {
        let report = model.behavioral_update(&training, &mut adam).expect("step");
        assert!(report.unclipped_norm.is_finite());
        assert!(report.average_loss.is_finite());
    }
    let after = model
        .behavioral_loss_for_test(&training)
        .expect("final loss and gradients");
    let held_after = model
        .behavioral_loss_for_test(&held_out)
        .expect("held final loss");
    eprintln!(
        "fixed_bc seed=503 steps=192 train={before}->{after} held={held_before}->{held_after}"
    );
    assert!(after < 0.05, "training loss {before} -> {after}");
    assert!(held_after < 0.1, "held loss {held_before} -> {held_after}");
    assert_actor_snapshot_sampling_parity(&model);
    assert!(
        model
            .export_parameters()
            .expect("weights")
            .iter()
            .all(|value| value.is_finite())
    );
    for (prediction, sample) in model
        .behavioral_predictions(&held_out)
        .expect("predictions")
        .iter()
        .zip(held_out)
    {
        assert_eq!(prediction.kind, sample.target().kind.selected);
    }
}

fn conditioning_learning_corpus(held_out: bool) -> Vec<ImitationSample> {
    [ActionKind::Cast, ActionKind::Swap]
        .into_iter()
        .enumerate()
        .map(|(index, kind)| {
            let tick = if held_out { 11 } else { 10 };
            let mut view = world_view(Team::Radiant, tick);
            let player = view
                .players
                .iter_mut()
                .find(|player| player.slot == SlotId(0))
                .expect("player");
            player.gold = Some(if index == 0 {
                100 + i32::from(held_out)
            } else {
                99_900 - i32::from(held_out)
            });
            let tracker = tracker_with_view(Team::Radiant, view);
            let space = ActionSpace::from_tracker(&tracker).expect("space");
            let frame = encode(&tracker, &LocalPolicyState::new(0));
            let mut logits = DecoderLogits::favor(kind);
            logits.target_mode[2] = 100.0;
            let action = decode_with_logits(&space, &logits).expect("label");
            assert_eq!(action.kind(), kind);
            let namespace = if held_out {
                SeedNamespace::Promotion
            } else {
                SeedNamespace::Training
            };
            let identity = SampleIdentity::from_frame(namespace, index as u64 + 1, 1, tick, &frame)
                .expect("identity");
            ImitationSample::teacher(frame, &space, action, identity).expect("sample")
        })
        .collect()
}

#[test]
fn input_conditioning_preserves_zero_signed_scalars_and_semantic_id_distinctions() {
    let mut rows = [0.0, 0.0, -0.75, 1.0, 1.0, 0.5, 64.0, 65_536.0, 1.0];
    crate::model::condition_rows(&mut rows, 3, &[(0, 64.0)], Some((1, 65_536.0)));
    assert_eq!(&rows[..3], &[0.0, 0.0, -0.75]);
    assert_eq!(rows[3], 1.0 / 64.0);
    assert!(rows[4] > 0.06);
    assert!(rows[4] < 0.07);
    assert_eq!(rows[5], 0.5);
    assert_eq!(&rows[6..], &[1.0, 1.0, 1.0]);
}

#[test]
fn output_head_gain_preserves_initializer_draw_order_and_hidden_parameters() {
    let model = PolicyModel::fresh(503).expect("model");
    let parameters = model.export_parameters().expect("parameters");
    let mut state = 503u64;
    let mut offset = 0;
    let mut decoder = false;
    for (name, shape) in model.parameter_schema().expect("schema") {
        decoder |= name == "value.weight";
        let count = shape.iter().product::<usize>();
        for &parameter in &parameters[offset..offset + count] {
            if name.ends_with(".bias") {
                assert_eq!(parameter, 0.0);
                continue;
            }
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let symmetric = (value >> 40) as f32 / ((1u32 << 24) - 1) as f32 * 2.0 - 1.0;
            let scale = if name.contains("_embedding") {
                (3.0 / shape[1] as f32).sqrt()
            } else {
                (6.0 / shape[0] as f32).sqrt() * if decoder { 0.01 } else { 1.0 }
            };
            assert_eq!(parameter, symmetric * scale, "{name}");
        }
        offset += count;
    }
    assert_eq!(offset, MODEL_PARAMETER_COUNT);
}

#[test]
fn ppo_inactive_entropy_normalizers_and_gradients_are_finite() {
    assert_ppo_entropy(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_ppo_inactive_entropy_normalizers_and_gradients_are_finite() {
    assert_ppo_entropy(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_ppo_entropy(device: PolicyDevice) {
    let (samples, logits) = ppo_entropy_inputs();
    let references = samples.iter().collect::<Vec<_>>();

    let probe = masked_ppo_entropy_for_test(&logits, &references, device).expect("entropy probe");
    let inactive = masked_ppo_entropy_for_test(&logits[..1], &references[..1], device)
        .expect("all-inactive batch");

    assert!(
        probe.log_normalizer.iter().all(|value| value.is_finite()),
        "{:?}",
        probe.log_normalizer
    );
    assert!(probe.entropy.iter().all(|value| value.is_finite()));
    assert!(
        probe
            .gradients
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert_eq!(probe.entropy[0], 0.0);
    assert_eq!(probe.gradients[0], vec![0.0; MODEL_KIND_HEAD]);
    assert_eq!(inactive.log_normalizer, probe.log_normalizer[..1]);
    assert_eq!(inactive.entropy, probe.entropy[..1]);
    assert_eq!(inactive.gradients, probe.gradients[..1]);
    assert_eq!(probe.entropy[1], 0.0);
    assert_eq!(probe.gradients[1], vec![0.0; MODEL_KIND_HEAD]);
    assert_active_ppo_entropy(&probe);
}

fn ppo_entropy_inputs() -> (Vec<PpoPreparedSample>, [[f32; MODEL_KIND_HEAD]; 4]) {
    let model = PolicyModel::fresh(9_101).expect("fixture model");
    let mut samples = sampled_ppo_examples(&model, 4, false);
    let mut logits = [[1.0e30; MODEL_KIND_HEAD]; 4];
    for (index, sample) in samples.iter_mut().enumerate() {
        sample.transition.target.kind = HeadTarget {
            active: index != 0,
            mask: std::array::from_fn(|column| match index {
                1 => column == 1,
                2 => column < 2,
                3 => column < 3,
                _ => false,
            }),
            selected: usize::from(index == 1),
        };
    }
    logits[0][0] = -1.0e30;
    logits[1][1] = -9.0;
    logits[2][0] = 0.0;
    logits[2][1] = 1.0;
    logits[3][..3].fill(-1000.0);
    (samples, logits)
}

fn assert_active_ppo_entropy(probe: &crate::model::PpoEntropyProbe) {
    let probability = 1.0f64 / (1.0 + 1.0f64.exp());
    let entropy = -probability * probability.ln() - (1.0 - probability) * (1.0 - probability).ln();
    assert!((f64::from(probe.entropy[2]) - entropy).abs() < 1.0e-6);
    assert!((f64::from(probe.gradients[2][0]) - probability * (1.0 - probability)).abs() < 1.0e-6);
    assert!((f64::from(probe.gradients[2][1]) + probability * (1.0 - probability)).abs() < 1.0e-6);
    assert_eq!(&probe.gradients[2][2..], &[0.0; MODEL_KIND_HEAD - 2]);
    assert!((probe.entropy[3] - 3.0f32.ln()).abs() < 1.0e-4);
    assert!(
        probe.gradients[3][..3]
            .iter()
            .all(|gradient| gradient.abs() < 1.0e-4)
    );
    assert_eq!(&probe.gradients[3][3..], &[0.0; MODEL_KIND_HEAD - 3]);
}

#[test]
fn ppo_entropy_rejects_mismatched_logit_batch_shape() {
    let (samples, logits) = ppo_entropy_inputs();
    let references = samples.iter().collect::<Vec<_>>();

    let error = masked_ppo_entropy_for_test(&logits[..1], &references, PolicyDevice::Cpu)
        .err()
        .expect("batch shape error");

    assert_eq!(
        error.to_string(),
        "model produced invalid PPO entropy head shape"
    );
}

#[test]
fn ppo_sampled_actor_log_probability_matches_tensor_likelihood_before_update() {
    assert_ppo_likelihood(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_ppo_sampled_actor_log_probability_matches_tensor_likelihood_before_update() {
    assert_ppo_likelihood(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_ppo_likelihood(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9_101, device).expect("model");
    assert_actor_snapshot_sampling_parity(&model);
    let identity = model.policy_identity().expect("identity");
    for batched in [false, true] {
        let samples = sampled_ppo_examples(&model, MODEL_TRAINING_BATCH, batched);
        assert!(
            samples
                .iter()
                .any(|sample| sample.transition.target.point_pointer.active)
        );
        assert!(
            samples
                .iter()
                .any(|sample| sample.transition.target.entity_pointer.active)
        );
        assert!(
            samples
                .iter()
                .any(|sample| sample.transition.target.item.active
                    || sample.transition.target.ability.active)
        );
        for batch_size in [1, 16, MODEL_TRAINING_BATCH] {
            for microbatch in samples.chunks(batch_size) {
                let references = microbatch.iter().collect::<Vec<_>>();

                let (likelihood, report) = model
                    .ppo_likelihood_for_test(&references)
                    .expect("tensor PPO likelihood");

                for (current, sample) in likelihood.iter().zip(microbatch) {
                    let old = sample.transition.old_log_probability;
                    assert!(
                        (current - old).abs() < 1.0e-4,
                        "batch={batch_size}, batched={batched}, action={:?}: {old} -> {current}",
                        sample.action()
                    );
                    assert!(((current - old).exp() - 1.0).abs() < 1.0e-4);
                }
                assert!(report.approximate_kl.abs() < 1.0e-6, "{report:?}");
                assert_eq!(report.clip_fraction, 0.0);
                assert!(report.entropy.is_finite());
            }
        }
    }
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
}

#[test]
fn ppo_critic_only_step_preserves_actor_and_trains_value() {
    assert_critic_transfer(&PolicyModel::fresh(9_101).expect("model"));
}

fn assert_actor_snapshot_sampling_parity(model: &PolicyModel) {
    let actor = model.actor_snapshot().expect("CPU actor snapshot");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let frames = vec![frame; 16];
    let spaces = (0..16)
        .map(|_| ActionSpace::from_tracker(&tracker).expect("space"))
        .collect::<Vec<_>>();
    let mut batch_random = (0..16)
        .map(|index| PpoRng::new(17 + index * 97))
        .collect::<Vec<_>>();
    let mut actor_random = batch_random.clone();
    let batch = model
        .sample_batch(&frames, &spaces, &mut batch_random)
        .expect("learner batch");
    let mut samples = Vec::with_capacity(16);
    for (index, choice) in batch.iter().enumerate() {
        let sampled = actor
            .sample(&frames[index], &spaces[index], &mut actor_random[index])
            .expect("actor sample");
        assert_eq!(sampled.action(), choice.action());
        assert_eq!(sampled.policy(), choice.policy());
        assert!((sampled.log_probability() - choice.log_probability()).abs() < 1.0e-4);
        samples.push(PpoPreparedSample {
            return_value: sampled.value(),
            advantage: 1.0,
            transition: sampled
                .finish(PpoOutcome {
                    stream: index,
                    decision: 0,
                    ticks: 3,
                    next_value: 0.0,
                    reward: 0.0,
                    terminal: true,
                })
                .expect("transition"),
        });
    }
    assert_eq!(actor_random, batch_random);
    let references = samples.iter().collect::<Vec<_>>();
    let (likelihood, _) = model
        .ppo_likelihood_for_test(&references)
        .expect("learner likelihood");
    for (value, sample) in likelihood.iter().zip(samples) {
        assert!((value - sample.transition.old_log_probability).abs() < 1.0e-4);
    }
}

#[test]
fn ppo_overshooting_step_restores_parameters_moments_and_identity() {
    let model = PolicyModel::fresh(9_101).expect("model");
    let samples = sampled_ppo_examples(&model, 16, true);
    let references = samples.iter().collect::<Vec<_>>();
    let config = crate::PpoConfig {
        learning_rate: 0.01,
        ..crate::PpoConfig::default()
    };
    let mut adam = model.claim_adam_for_test(config.adam()).expect("optimizer");
    let before = model.coherent_snapshot(&adam).expect("snapshot");

    let report = model
        .ppo_update(&references, &mut adam, config)
        .expect("guarded update");

    assert!(!report.applied);
    assert!(report.approximate_kl > f64::from(config.target_kl));
    assert_eq!(model.coherent_snapshot(&adam).expect("snapshot"), before);
}

#[test]
fn ppo_uneven_microbatch_overshoot_restores_nonzero_optimizer_state() {
    for count in [65, 81] {
        let (model, mut adam, samples, config) = ppo_rollback_fixture(count);
        let references = samples.iter().collect::<Vec<_>>();
        let before = model.coherent_snapshot(&adam).expect("snapshot");
        let identity = model.policy_identity().expect("identity");

        let report = model
            .ppo_update(&references, &mut adam, config)
            .expect("guarded update");

        assert!(!report.applied);
        assert_eq!(report.samples, count);
        assert!(report.approximate_kl > f64::from(config.target_kl));
        let expected_kl = uneven_candidate_weighted_kl(count);
        assert!((report.approximate_kl - expected_kl).abs() < 1.0e-6);
        assert_eq!(model.coherent_snapshot(&adam).expect("snapshot"), before);
        assert_eq!(model.policy_identity().expect("identity"), identity);
    }
}

fn uneven_candidate_weighted_kl(count: usize) -> f64 {
    let (model, mut adam, samples, config) = ppo_rollback_fixture(count);
    let references = samples.iter().collect::<Vec<_>>();
    let report = model
        .ppo_update(
            &references,
            &mut adam,
            crate::PpoConfig {
                target_kl: 1.0e10,
                ..config
            },
        )
        .expect("retain identical candidate for independent likelihood measurement");
    assert!(report.applied);
    let mut weighted_kl = 0.0;
    let mut chunk_kls = Vec::new();
    for chunk in references.chunks(MODEL_TRAINING_BATCH) {
        let (_, report) = model
            .ppo_likelihood_for_test(chunk)
            .expect("candidate likelihood");
        weighted_kl += report.approximate_kl * chunk.len() as f64;
        chunk_kls.push(report.approximate_kl);
    }
    let expected = weighted_kl / count as f64;
    let unweighted = chunk_kls.iter().sum::<f64>() / chunk_kls.len() as f64;
    assert!(
        (expected - unweighted).abs() > 1.0e-6,
        "fixture must distinguish row weighting from chunk weighting"
    );
    expected
}

#[test]
fn ppo_candidate_evaluation_error_after_first_microbatch_restores_exact_state() {
    let (model, mut adam, samples, config) = ppo_rollback_fixture(81);
    let references = samples.iter().collect::<Vec<_>>();
    let before = model.coherent_snapshot(&adam).expect("snapshot");
    let identity = model.policy_identity().expect("identity");

    let error = model
        .ppo_update_with_faults_for_test(&references, &mut adam, config, false)
        .expect_err("candidate evaluation failure");

    assert_eq!(
        error.to_string(),
        "model tensor operation failed: injected PPO candidate evaluation failure after 64 rows"
    );
    assert_eq!(model.coherent_snapshot(&adam).expect("snapshot"), before);
    assert_eq!(model.policy_identity().expect("identity"), identity);
}

#[test]
fn ppo_rollback_import_failure_preserves_candidate_evaluation_error_context() {
    let (model, mut adam, samples, config) = ppo_rollback_fixture(65);
    let references = samples.iter().collect::<Vec<_>>();

    let error = model
        .ppo_update_with_faults_for_test(&references, &mut adam, config, true)
        .expect_err("rollback import failure");

    assert_eq!(
        error.to_string(),
        "model tensor operation failed: PPO candidate rejected (model tensor operation failed: injected PPO candidate evaluation failure after 64 rows); parameter rollback failed (model injected parameter replacement failure after tensor 0)"
    );
}

fn ppo_rollback_fixture(
    count: usize,
) -> (
    PolicyModel,
    crate::AdamState,
    Vec<PpoPreparedSample>,
    crate::PpoConfig,
) {
    assert!([65, 81].contains(&count));
    let model = PolicyModel::fresh(9_101).expect("model");
    let base = sampled_ppo_examples(&model, MODEL_TRAINING_BATCH, true);
    let samples = (0..count)
        .map(|index| base[index % base.len()].clone())
        .collect::<Vec<_>>();
    let config = crate::PpoConfig {
        learning_rate: 0.01,
        ..crate::PpoConfig::default()
    };
    let mut adam = model.claim_adam_for_test(config.adam()).expect("optimizer");
    let mut warmup = samples.clone();
    for sample in &mut warmup {
        sample.advantage = 0.0;
        sample.return_value = sample.transition.old_value + 1.0;
    }
    let references = warmup.iter().collect::<Vec<_>>();
    let report = model
        .ppo_update(
            &references,
            &mut adam,
            crate::PpoConfig {
                entropy_coefficient: 0.0,
                ..config
            },
        )
        .expect("value-only fixture step");
    assert!(report.applied);
    assert_eq!(adam.step(), 1);
    assert!(adam.moments().0.iter().any(|value| *value != 0.0));
    (model, adam, samples, config)
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_ppo_critic_only_step_preserves_actor_and_trains_value() {
    let model = PolicyModel::fresh_on(9_101, PolicyDevice::Cuda { ordinal: 0 }).expect("model");
    assert_critic_transfer(&model);
}

fn assert_critic_transfer(model: &PolicyModel) {
    let mut samples = sampled_ppo_examples(model, 16, true);
    for sample in &mut samples {
        sample.advantage = 0.0;
        sample.return_value = sample.transition.old_value + 10.0;
    }
    let references = samples.iter().collect::<Vec<_>>();
    let config = crate::PpoConfig {
        learning_rate: 3.0e-4,
        entropy_coefficient: 0.0,
        ..crate::PpoConfig::default()
    };
    let mut adam = model.claim_adam_for_test(config.adam()).expect("optimizer");
    let before = model.export_parameters().expect("parameters");
    let report = model
        .ppo_update(&references, &mut adam, config)
        .expect("critic update");
    let (_, after) = model
        .ppo_likelihood_for_test(&references)
        .expect("likelihood");
    eprintln!(
        "critic transfer {:?}: gradient={} post_kl={} value_loss={} -> {}",
        model.device(),
        report.gradient_norm,
        after.approximate_kl,
        report.value_loss,
        after.value_loss
    );
    assert!(report.applied);
    assert!(after.value_loss < report.value_loss);
    let parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("shapes") {
        let end = offset + shape.iter().product::<usize>();
        if !name.starts_with("value.") {
            assert!(
                before[offset..end] == parameters[offset..end],
                "critic changed {name}"
            );
        }
        offset = end;
    }
    assert!(after.approximate_kl.abs() < 1.0e-6);
}

fn sampled_ppo_examples(
    model: &PolicyModel,
    count: usize,
    batched: bool,
) -> Vec<PpoPreparedSample> {
    assert!(count > 0);
    assert!(count <= MODEL_TRAINING_BATCH);
    let mut frames = Vec::with_capacity(count);
    let mut spaces = Vec::with_capacity(count);
    let mut random = Vec::with_capacity(count);
    for index in 0..count {
        let team = if index.is_multiple_of(2) {
            Team::Radiant
        } else {
            Team::Dire
        };
        let tracker = tracker_with_view(team, world_view(team, 10 + index as u32));
        spaces.push(ActionSpace::from_tracker(&tracker).expect("space"));
        let mut frame = encode(&tracker, &LocalPolicyState::new(0));
        frame.global[63] = index as f32 / MODEL_TRAINING_BATCH as f32;
        frames.push(frame);
        random.push(PpoRng::new(17 + index as u64 * 97));
    }
    let choices = if batched {
        model
            .sample_batch(&frames, &spaces, &mut random)
            .expect("batch samples")
    } else {
        frames
            .iter()
            .zip(&spaces)
            .zip(&mut random)
            .map(|((frame, space), random)| {
                model.sample(frame, space, random).expect("scalar sample")
            })
            .collect()
    };
    choices
        .into_iter()
        .enumerate()
        .map(|(index, choice)| {
            assert!(spaces[index].allows(choice.action()));
            PpoPreparedSample {
                return_value: choice.value(),
                advantage: 1.0,
                transition: choice
                    .finish(PpoOutcome {
                        stream: index,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: 0.0,
                        terminal: true,
                    })
                    .expect("transition"),
            }
        })
        .collect()
}

#[test]
fn unit_kind_tokens_enter_their_semantic_pool() {
    let groups = [
        (1.0, 0),
        (2.0, 1),
        (3.0, 1),
        (4.0, 1),
        (5.0, 1),
        (6.0, 3),
        (7.0, 2),
        (8.0, 2),
        (9.0, 2),
        (10.0, 2),
        (11.0, 4),
        (12.0, 4),
    ];
    for (token, group) in groups {
        assert_eq!(unit_group(token), Some(group), "token {token}");
    }
    assert_eq!(unit_group(0.0), None);
    assert_eq!(unit_group(13.0), None);
}

#[test]
fn behavioral_update_holds_one_exclusive_guard_against_parameter_import() {
    let model = Arc::new(PolicyModel::fresh(501).expect("model"));
    let sample = model_sample();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let update_model = Arc::clone(&model);
    let update_entered = Arc::clone(&entered);
    let update_release = Arc::clone(&release);
    let update = thread::spawn(move || {
        let mut adam = update_model
            .claim_adam_for_test(AdamConfig::default())
            .expect("Adam");
        update_model
            .behavioral_update_with_barrier(&[&sample], &mut adam, &update_entered, &update_release)
            .expect("update");
        adam
    });
    entered.wait();
    let replacement = vec![0.25; MODEL_PARAMETER_COUNT];
    let import_model = Arc::clone(&model);
    let (complete_tx, complete_rx) = mpsc::sync_channel(1);
    let importer = thread::spawn(move || {
        import_model
            .import_parameters(&replacement)
            .expect("import");
        complete_tx.send(()).expect("complete");
    });
    assert!(!model.parameter_write_available_for_test());
    assert!(complete_rx.try_recv().is_err());
    release.wait();
    update.join().expect("update thread");
    complete_rx.recv().expect("serialized import");
    importer.join().expect("import thread");
    assert_eq!(
        model.export_parameters().expect("parameters"),
        vec![0.25; MODEL_PARAMETER_COUNT]
    );
}

#[test]
fn second_adam_and_stale_adam_after_raw_import_fail_without_mutation() {
    let model = PolicyModel::fresh(506).expect("model");
    let initial_identity = model.policy_identity().expect("initial identity");
    let sample = model_sample();
    let mut first = model
        .claim_adam_for_test(AdamConfig::default())
        .expect("first Adam");
    let mut second = first.clone();
    model
        .behavioral_update(&[&sample], &mut first)
        .expect("first update");
    let identity_after_update = model.policy_identity().expect("identity after update");
    assert_eq!(identity_after_update.lineage(), initial_identity.lineage());
    assert_eq!(
        identity_after_update.revision(),
        initial_identity.revision() + 1
    );
    let before_second = model.export_parameters().expect("before second");
    let identity_before_second = model.policy_identity().expect("before second identity");
    let second_before = second.clone();
    assert_eq!(
        model
            .behavioral_update(&[&sample], &mut second)
            .unwrap_err()
            .to_string(),
        "model optimizer owner or parameter revision does not match"
    );
    assert_eq!(
        model.export_parameters().expect("after second"),
        before_second
    );
    assert_eq!(
        model.policy_identity().expect("after second identity"),
        identity_before_second
    );
    assert_eq!(second, second_before);

    let mut invalid = before_second.clone();
    invalid[0] = f32::NAN;
    assert!(model.import_parameters(&invalid).is_err());
    model
        .behavioral_update(&[&sample], &mut first)
        .expect("owner survives failed import");

    let identity_before_import = model.policy_identity().expect("identity before import");
    let current = model.export_parameters().expect("current parameters");
    model
        .import_parameters(&current)
        .expect("raw parameter import");
    assert_ne!(
        model.policy_identity().expect("identity after import"),
        identity_before_import
    );
    assert_eq!(
        model.policy_identity().expect("import revision").revision(),
        identity_before_import.revision() + 1
    );
    let before_stale = model.export_parameters().expect("before stale");
    let identity_before_stale = model.policy_identity().expect("before stale identity");
    let first_before = first.clone();
    assert!(model.behavioral_update(&[&sample], &mut first).is_err());
    assert_eq!(
        model.export_parameters().expect("after stale"),
        before_stale
    );
    assert_eq!(
        model.policy_identity().expect("after stale identity"),
        identity_before_stale
    );
    assert_eq!(first, first_before);
}

#[test]
fn behavioral_update_rejects_nonfinite_in_inactive_illegal_head_atomically() {
    let model = PolicyModel::fresh(502).expect("model");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    set_named_parameter_range(&model, &mut parameters, "kind_embedding.weight", 0, 32, 1.0);
    for input in 256..288 {
        set_named_parameter_range(
            &model,
            &mut parameters,
            "shop_head.weight",
            input * 64,
            64,
            f32::MAX,
        );
    }
    model
        .import_parameters(&parameters)
        .expect("finite parameters");
    let before = model.export_parameters().expect("before");
    let sample = model_sample();
    assert_eq!(
        model
            .training_forward(
                std::slice::from_ref(sample.frame()),
                &[sample.target().prefix()],
            )
            .unwrap_err()
            .to_string(),
        "model shop output at batch 0 index 0 is non-finite"
    );
    let mut adam = model
        .claim_adam_for_test(AdamConfig::default())
        .expect("Adam");
    let adam_before = adam.clone();
    assert_eq!(
        model
            .behavioral_update(&[&sample], &mut adam)
            .unwrap_err()
            .to_string(),
        "model shop output at batch 0 index 0 is non-finite"
    );
    assert_eq!(model.export_parameters().expect("after"), before);
    assert_eq!(adam, adam_before);
}

#[test]
fn behavioral_update_rejects_held_out_samples_before_mutation() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Promotion, 3, 1, 10, &frame).expect("identity");
    let sample = ImitationSample::teacher(frame, &space, StructuredAction::Continue, identity)
        .expect("held-out sample");
    let model = PolicyModel::fresh(505).expect("model");
    let before = model.export_parameters().expect("parameters");
    let mut adam = model
        .claim_adam_for_test(AdamConfig::default())
        .expect("Adam");
    let adam_before = adam.clone();
    assert_eq!(
        model
            .behavioral_update(&[&sample], &mut adam)
            .unwrap_err()
            .to_string(),
        "model behavioral training example 0 is not Train"
    );
    assert_eq!(model.export_parameters().expect("after"), before);
    assert_eq!(adam, adam_before);
}

#[test]
fn effective_batch_above_microbatch_boundary_matches_identical_single_example_update() {
    let large_model = PolicyModel::fresh(503).expect("large model");
    let single_model = PolicyModel::fresh(504).expect("single model");
    let zero = vec![0.0; MODEL_PARAMETER_COUNT];
    large_model.import_parameters(&zero).expect("large zero");
    single_model.import_parameters(&zero).expect("single zero");
    let sample = model_sample();
    let examples = vec![&sample; MODEL_TRAINING_BATCH + 1];
    let mut large_adam = large_model
        .claim_adam_for_test(AdamConfig::default())
        .expect("large Adam");
    let mut single_adam = single_model
        .claim_adam_for_test(AdamConfig::default())
        .expect("single Adam");
    let large = large_model
        .behavioral_update(&examples, &mut large_adam)
        .expect("large update");
    let single = single_model
        .behavioral_update(&[&sample], &mut single_adam)
        .expect("single update");
    assert!((large.average_loss - single.average_loss).abs() <= 1.0e-6);
    let legal_kind_count = sample
        .target()
        .kind
        .mask
        .iter()
        .filter(|allowed| **allowed)
        .count();
    assert!((large.average_loss - (legal_kind_count as f64).ln()).abs() <= 1.0e-6);
    assert_eq!(large.active_head_counts[0], MODEL_TRAINING_BATCH + 1);
    assert!(
        large.active_head_counts[1..]
            .iter()
            .all(|count| *count == 0)
    );
    for (left, right) in large_model
        .export_parameters()
        .expect("large parameters")
        .iter()
        .zip(single_model.export_parameters().expect("single parameters"))
    {
        assert!((*left - right).abs() <= 1.0e-6);
    }
}

#[test]
fn masked_cross_entropy_is_invariant_to_representable_common_offsets() {
    let baseline = masked_cross_entropy_for_test(&[0.0, 1.0, 2.0], &[true, true, false], 1, true)
        .expect("baseline");
    for offset in [-1_000_000.0, 1_000_000.0] {
        let shifted = masked_cross_entropy_for_test(
            &[offset, offset + 1.0, 1.0e30],
            &[true, true, false],
            1,
            true,
        )
        .expect("shifted");
        assert!(
            (shifted.loss - baseline.loss).abs() < 1.0e-6,
            "{shifted:?} != {baseline:?}"
        );
        for (left, right) in shifted.gradients.iter().zip(&baseline.gradients) {
            assert!((left - right).abs() < 1.0e-6);
        }
    }
}

#[test]
fn masked_cross_entropy_matches_reference_and_excludes_illegal_or_inactive_logits() {
    let positive = masked_cross_entropy_for_test(&[0.0, 1.0, 2.0], &[true, true, false], 1, true)
        .expect("positive");
    assert!((positive.loss - 0.313_261_66).abs() <= 1.0e-6);
    assert_eq!(positive.gradients[2], 0.0);

    let illegal_high =
        masked_cross_entropy_for_test(&[0.0, 1.0, 1.0e30], &[true, true, false], 1, true)
            .expect("illegal high");
    assert_eq!(illegal_high, positive);
    let one =
        masked_cross_entropy_for_test(&[9.0, -9.0], &[false, true], 1, true).expect("one choice");
    assert_eq!(one.loss, 0.0);
    assert_eq!(one.gradients, vec![0.0, 0.0]);
    let inactive = masked_cross_entropy_for_test(&[1.0e30, -1.0e30], &[false, false], 0, false)
        .expect("inactive");
    assert_eq!(inactive.loss, 0.0);
    assert_eq!(inactive.gradients, vec![0.0, 0.0]);
    assert_eq!(
        masked_cross_entropy_for_test(&[0.0, 1.0], &[true, false], 1, true)
            .unwrap_err()
            .to_string(),
        "model behavioral target label 1 is illegal for head test"
    );
}

#[test]
fn adam_matches_scalar_vector_reference_and_clips_only_above_boundary() {
    let config = AdamConfig::default();
    let scalar = adam_step_for_test(&[1.0], &[0.1], &[0.0], &[0.0], 0, config).expect("scalar");
    assert!((scalar.parameters[0] - 0.999).abs() <= 1.0e-7);
    assert!((scalar.first_moment[0] - 0.01).abs() <= 1.0e-7);
    assert!((scalar.second_moment[0] - 0.000_01).abs() <= 1.0e-9);

    for gradient in [0.25, 0.5] {
        let update =
            adam_step_for_test(&[1.0], &[gradient], &[0.0], &[0.0], 0, config).expect("unclipped");
        assert_eq!(update.applied_scale, 1.0);
    }
    let clipped = adam_step_for_test(&[1.0], &[1.0], &[0.0], &[0.0], 0, config).expect("clipped");
    assert_eq!(clipped.unclipped_norm, 1.0);
    assert_eq!(clipped.applied_scale, 0.5);
}

#[test]
fn adam_later_step_nonzero_moments_matches_multidimensional_clipped_reference() {
    let config = AdamConfig::default();
    let update = adam_step_for_test(
        &[1.0, -2.0],
        &[3.0, 4.0],
        &[0.1, -0.2],
        &[0.01, 0.04],
        4,
        config,
    )
    .expect("update");
    assert!((update.unclipped_norm - 5.0).abs() <= f64::EPSILON);
    assert!((update.applied_scale - 0.1).abs() <= f64::EPSILON);
    let clipped = [0.3f32, 0.4f32];
    for index in 0..2 {
        let first = config.beta1 * [0.1, -0.2][index] + (1.0 - config.beta1) * clipped[index];
        let second = config.beta2 * [0.01, 0.04][index]
            + (1.0 - config.beta2) * clipped[index] * clipped[index];
        let first_hat = f64::from(first) / (1.0 - f64::from(config.beta1).powi(5));
        let second_hat = f64::from(second) / (1.0 - f64::from(config.beta2).powi(5));
        let expected = [1.0f64, -2.0][index]
            - f64::from(config.learning_rate) * first_hat
                / (second_hat.sqrt() + f64::from(config.epsilon));
        assert!((update.parameters[index] - expected as f32).abs() <= 1.0e-7);
        assert!((update.first_moment[index] - first).abs() <= 1.0e-7);
        assert!((update.second_moment[index] - second).abs() <= 1.0e-7);
    }
}

#[test]
fn adam_rejects_invalid_config_nonfinite_and_extreme_updates_without_partial_state() {
    let invalid = AdamConfig {
        learning_rate: 0.0,
        ..AdamConfig::default()
    };
    assert!(adam_step_for_test(&[1.0], &[0.1], &[0.0], &[0.0], 0, invalid).is_err());
    assert!(
        adam_step_for_test(
            &[1.0],
            &[f32::NAN],
            &[0.0],
            &[0.0],
            0,
            AdamConfig::default()
        )
        .is_err()
    );
    let extreme = AdamConfig {
        learning_rate: f32::MAX,
        ..AdamConfig::default()
    };
    assert!(adam_step_for_test(&[f32::MAX], &[-f32::MAX], &[0.0], &[0.0], 0, extreme).is_err());
}

#[test]
fn model_schema_and_head_dimensions_are_stable() {
    assert_eq!(MODEL_SCHEMA_VERSION, 14);
    assert!(
        crate::MODEL_SCHEMA_DESCRIPTOR
            .contains("action_schema_version=3;action_schema_hash=1755359086494840931;")
    );
    assert_eq!(MODEL_SCHEMA_HASH, 7_970_187_849_195_607_202);
    assert_eq!(MODEL_KIND_HEAD, 16);
    assert_eq!(MODEL_UNIT_HEAD, 2);
    assert_eq!(MODEL_ABILITY_HEAD, 8);
    assert_eq!(MODEL_ITEM_HEAD, 15);
    assert_eq!(MODEL_SWAP_HEAD, 15);
    assert_eq!(MODEL_LEARN_HEAD, 6);
    assert_eq!(MODEL_SHOP_HEAD, 64);
    assert_eq!(MODEL_LOOT_HEAD, 16);
    assert_eq!(MODEL_ENTITY_POINTER_HEAD, 96);
    assert_eq!(MODEL_POINT_POINTER_HEAD, 48);
    assert_eq!(MODEL_EVALUATION_MICROBATCH, 64);
    assert_eq!(MODEL_TRAINING_BATCH, 64);
}

#[test]
fn model_parameter_count_and_f32_size_are_bounded() {
    let model = PolicyModel::fresh(7).expect("model");
    let count = model.parameter_count();
    let schema = model.parameter_schema().expect("schema");

    assert_eq!(count, MODEL_PARAMETER_COUNT);
    assert!((1_000_000..=3_000_000).contains(&count));
    assert!((4 * 1_048_576..=12 * 1_048_576).contains(&(count * size_of::<f32>())));
    assert_eq!(schema.len(), 62);
    assert_eq!(schema.first(), Some(&("unit.0.weight", vec![73, 64])));
    assert_eq!(schema.last(), Some(&("point_query.bias", vec![64])));
    assert_eq!(
        schema
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>(),
        count
    );
    assert_eq!(
        schema
            .iter()
            .map(|(name, _)| *name)
            .collect::<BTreeSet<_>>()
            .len(),
        schema.len()
    );
}

#[test]
fn typed_pooling_uses_exact_mean_max_and_isolates_other_groups() {
    let pooled = pool_groups_for_test(
        &[vec![3.0, 5.0], vec![100.0, 200.0], vec![7.0, 11.0]],
        &[vec![true, false, true], vec![false, true, false]],
    )
    .expect("pool");

    assert_eq!(
        pooled,
        vec![5.0, 8.0, 7.0, 11.0, 100.0, 200.0, 100.0, 200.0]
    );
    let isolated = pool_groups_for_test(
        &[vec![3.0, 5.0], vec![9_000.0, 8_000.0], vec![7.0, 11.0]],
        &[vec![true, false, true], vec![false, true, false]],
    )
    .expect("isolated pool");
    assert_eq!(&isolated[..4], &pooled[..4]);
    let empty =
        pool_groups_for_test(&[vec![9_000.0, 8_000.0]], &[vec![false]]).expect("empty pool");
    assert_eq!(empty, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn tied_max_pool_gradient_selects_only_the_lowest_token() {
    let gradients = pool_max_gradient_for_test(
        &[vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        &[vec![true, true, true]],
    )
    .expect("gradients");

    assert_eq!(gradients, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(gradients.iter().sum::<f32>(), 2.0);
}

#[test]
fn feature_frame_accessors_are_read_only_views_of_every_tensor_group() {
    let frame = populated_frame();

    assert_eq!(frame.global(), &frame.global);
    assert_eq!(frame.history(), &frame.history);
    assert_eq!(frame.policy_history(), &frame.policy_history);
    assert_eq!(frame.units(), &frame.units);
    assert_eq!(frame.own_units(), &frame.own_units);
    assert_eq!(frame.remembered_units(), &frame.remembered_units);
    assert_eq!(frame.points(), &frame.points);
    assert_eq!(frame.abilities(), &frame.abilities);
    assert_eq!(frame.items(), &frame.items);
    assert_eq!(frame.projectiles(), &frame.projectiles);
    assert_eq!(frame.loot(), &frame.loot);
    assert_eq!(frame.map(), &frame.map);
}

#[test]
fn initialization_is_seed_deterministic_for_parameters_and_outputs() {
    let first = PolicyModel::fresh(91).expect("first");
    let second = PolicyModel::fresh(91).expect("second");
    let different = PolicyModel::fresh(92).expect("different");
    let frame = populated_frame();

    assert_eq!(
        first.export_parameters().expect("first parameters"),
        second.export_parameters().expect("second parameters")
    );
    assert_ne!(
        first.export_parameters().expect("first parameters"),
        different.export_parameters().expect("different parameters")
    );
    assert_eq!(
        first.evaluate(&frame).expect("first output"),
        second.evaluate(&frame).expect("second output")
    );
    assert_ne!(
        first.evaluate(&frame).expect("first output"),
        different.evaluate(&frame).expect("different output")
    );
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_model_runs_forward_backward_and_parameter_export_on_selected_device() {
    let cpu = PolicyModel::fresh(17_002).expect("CPU model");
    let gpu = PolicyModel::fresh_on(17_002, PolicyDevice::Cuda { ordinal: 0 }).expect("CUDA model");
    let frame = populated_frame();

    assert_eq!(gpu.device(), PolicyDevice::Cuda { ordinal: 0 });
    assert_eq!(
        cpu.export_parameters().expect("CPU parameters"),
        gpu.export_parameters().expect("CUDA parameters")
    );
    assert_outputs_close(
        &cpu.evaluate(&frame).expect("CPU output"),
        &gpu.evaluate(&frame).expect("CUDA output"),
        1.0e-4,
    );

    let sample = model_sample();
    let mut adam = gpu
        .claim_adam_for_test(AdamConfig::default())
        .expect("CUDA Adam");
    let report = gpu
        .behavioral_update(&[&sample], &mut adam)
        .expect("CUDA update");
    assert_eq!(report.sample_count, 1);
    assert_eq!(report.optimizer_step, 1);
    assert!(
        gpu.export_parameters()
            .expect("updated CUDA parameters")
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn single_and_batch_evaluation_are_equivalent_and_finite() {
    let model = PolicyModel::fresh(1).expect("model");
    let first = populated_frame();
    let mut second = first.clone();
    second.global[0] += 0.125;

    let one = model.evaluate(&first).expect("single");
    let batch = model.evaluate_batch(&[first, second]).expect("batch");

    assert_eq!(batch.len(), 2);
    assert_outputs_close(&one, &batch[0], 1.0e-5);
    assert!(batch.iter().all(|output| output.is_finite()));
}

#[test]
fn batch_boundaries_fail_before_evaluation_with_exact_errors() {
    let model = PolicyModel::fresh(1).expect("model");

    assert_eq!(
        model.evaluate_batch(&[]).unwrap_err().to_string(),
        "model batch must contain at least one frame"
    );
    assert_eq!(
        validate_batch_count(MODEL_MAX_BATCH + 1)
            .unwrap_err()
            .to_string(),
        format!(
            "model batch count {} exceeds maximum {MODEL_MAX_BATCH}",
            MODEL_MAX_BATCH + 1
        )
    );
}

#[test]
fn evaluation_is_equivalent_across_the_microbatch_boundary() {
    let model = PolicyModel::fresh(101).expect("model");
    let frames = vec![populated_frame(); MODEL_EVALUATION_MICROBATCH + 1];

    let chunked = model.evaluate_batch(&frames).expect("chunked");
    let first = model
        .evaluate_batch(&frames[..MODEL_EVALUATION_MICROBATCH])
        .expect("first chunk");
    let last = model
        .evaluate(&frames[MODEL_EVALUATION_MICROBATCH])
        .expect("last chunk");

    assert_eq!(&chunked[..MODEL_EVALUATION_MICROBATCH], first);
    assert_eq!(chunked[MODEL_EVALUATION_MICROBATCH], last);
}

#[test]
fn training_batch_limit_and_prefix_count_have_exact_errors() {
    let model = PolicyModel::fresh(102).expect("model");
    let frame = FeatureFrame::new();
    let prefix = TrainingPrefix::new(ActionKind::Continue, None, None);

    assert_eq!(
        model.training_forward(&[], &[]).unwrap_err().to_string(),
        "model training batch must contain at least one frame"
    );
    assert_eq!(
        model
            .training_forward(std::slice::from_ref(&frame), &[])
            .unwrap_err()
            .to_string(),
        "model training prefix count 0 differs from frame count 1"
    );
    assert_eq!(
        crate::model::validate_training_batch_count(MODEL_TRAINING_BATCH + 1)
            .unwrap_err()
            .to_string(),
        format!(
            "model training batch count {} exceeds maximum {MODEL_TRAINING_BATCH}",
            MODEL_TRAINING_BATCH + 1
        )
    );
    assert!(
        model
            .training_forward(std::slice::from_ref(&frame), &[prefix])
            .is_ok()
    );
}

#[test]
fn absent_token_garbage_is_masked_before_and_after_encoders() {
    let model = PolicyModel::fresh(2).expect("model");
    let clean = FeatureFrame::new();
    let mut garbage = clean.clone();
    garbage.units[7][1..].fill(900.0);
    garbage.abilities[5][1..].fill(-700.0);
    garbage.items[12][1..].fill(500.0);
    garbage.points[9][1..].fill(300.0);
    garbage.projectiles[4][1..].fill(-200.0);
    garbage.loot[3][1..].fill(100.0);

    assert_eq!(
        model.evaluate(&clean).expect("clean"),
        model.evaluate(&garbage).expect("garbage")
    );
    assert!(model.evaluate(&clean).expect("empty").is_finite());
}

#[test]
fn deepsets_is_invariant_to_permutations_inside_typed_groups() {
    let model = PolicyModel::fresh(3).expect("model");
    let mut first = FeatureFrame::new();
    first.units[0][unit_feature::TOKEN_PRESENT] = 1.0;
    first.units[0][unit_feature::KIND_TOKEN] = 2.0;
    first.units[0][30] = 0.2;
    first.units[1][unit_feature::TOKEN_PRESENT] = 1.0;
    first.units[1][unit_feature::KIND_TOKEN] = 4.0;
    first.units[1][30] = 0.8;
    let mut second = first.clone();
    second.units.swap(0, 1);

    let left = model.evaluate(&first).expect("left");
    let right = model.evaluate(&second).expect("right");
    assert_outputs_close(&left, &right, 1.0e-6);
}

#[test]
fn each_encoded_group_can_influence_the_output() {
    let model = PolicyModel::fresh(4).expect("model");
    let baseline = model.evaluate(&FeatureFrame::new()).expect("baseline");
    let mutations: [fn(&mut FeatureFrame); 10] = [
        |f| f.global[1] = 1.0,
        |f| f.history[0][0] = 1.0,
        |f| f.policy_history[0][0] = 1.0,
        |f| {
            f.units[0][0] = 1.0;
            f.units[0][5] = 1.0;
        },
        |f| {
            f.own_units[0][0] = 1.0;
            f.own_units[0][5] = 1.0;
        },
        |f| {
            f.points[0][0] = 1.0;
            f.points[0][2] = 1.0;
        },
        |f| {
            f.abilities[0][0] = 1.0;
            f.abilities[0][5] = 1.0;
        },
        |f| {
            f.items[0][0] = 1.0;
            f.items[0][4] = 1.0;
        },
        |f| {
            f.projectiles[0][0] = 1.0;
            f.projectiles[0][6] = 1.0;
        },
        |f| {
            f.loot[0][0] = 1.0;
            f.loot[0][1] = 1.0;
        },
    ];
    for mutate in mutations {
        let mut frame = FeatureFrame::new();
        mutate(&mut frame);
        assert_ne!(model.evaluate(&frame).expect("changed"), baseline);
    }
}

#[test]
fn every_typed_unit_pool_can_influence_the_output() {
    let model = PolicyModel::fresh(41).expect("model");
    let baseline = model.evaluate(&FeatureFrame::new()).expect("baseline");
    for kind in [1.0, 2.0, 8.0, 6.0, 11.0] {
        let mut frame = FeatureFrame::new();
        frame.units[0][unit_feature::TOKEN_PRESENT] = 1.0;
        frame.units[0][unit_feature::KIND_TOKEN] = kind;
        frame.units[0][unit_feature::HP_RATIO] = 0.5;
        assert_ne!(model.evaluate(&frame).expect("typed group"), baseline);
    }
    let mut remembered = FeatureFrame::new();
    remembered.remembered_units[0][unit_feature::TOKEN_PRESENT] = 1.0;
    remembered.remembered_units[0][unit_feature::KIND_TOKEN] = 1.0;
    remembered.remembered_units[0][unit_feature::REMEMBERED] = 1.0;
    assert_ne!(model.evaluate(&remembered).expect("remembered"), baseline);
}

#[test]
fn masked_argmax_excludes_illegal_scores_and_has_exact_errors() {
    assert_eq!(
        masked_argmax(&[1.0, 99.0, 1.0], &[true, false, true]).expect("choice"),
        0
    );
    assert_eq!(
        masked_argmax(&[], &[]).unwrap_err().to_string(),
        "model selection mask is empty"
    );
    assert_eq!(
        masked_argmax(&[1.0], &[true, false])
            .unwrap_err()
            .to_string(),
        "model selection logits length 1 differs from mask length 2"
    );
    assert_eq!(
        masked_argmax(&[1.0], &[false]).unwrap_err().to_string(),
        "model selection has no legal continuation"
    );
    assert_eq!(
        masked_argmax(&[1.0, f32::NAN], &[true, false])
            .unwrap_err()
            .to_string(),
        "model selection logit 1 is non-finite"
    );
    assert_eq!(
        masked_argmax(&[f32::INFINITY], &[true])
            .unwrap_err()
            .to_string(),
        "model selection logit 0 is non-finite"
    );
}

#[test]
fn parameter_import_validation_is_atomic() {
    let model = PolicyModel::fresh(5).expect("model");
    let original = model.export_parameters().expect("original");
    let identity = model.policy_identity().expect("identity");
    let short = &original[..original.len() - 1];
    assert_eq!(
        model.import_parameters(short).unwrap_err().to_string(),
        format!(
            "model parameter length {} differs from expected {}",
            short.len(),
            original.len()
        )
    );
    assert_eq!(model.export_parameters().expect("after length"), original);
    assert_eq!(
        model.policy_identity().expect("after length identity"),
        identity
    );
    let mut nonfinite = original.clone();
    nonfinite[10] = f32::NAN;
    assert_eq!(
        model.import_parameters(&nonfinite).unwrap_err(),
        ModelError::NonFiniteParameter { index: 10 }
    );
    assert_eq!(
        model.export_parameters().expect("after nonfinite"),
        original
    );
    assert_eq!(
        model.policy_identity().expect("after nonfinite identity"),
        identity
    );
}

#[test]
fn injected_middle_import_failure_deeply_restores_every_parameter() {
    let model = PolicyModel::fresh(113).expect("model");
    let original = model.export_parameters().expect("original");
    let identity = model.policy_identity().expect("identity");
    let replacement = vec![0.375; MODEL_PARAMETER_COUNT];

    let error = model
        .import_parameters_with_failure(&replacement, 30)
        .expect_err("injected failure");

    assert_eq!(
        error.to_string(),
        "model injected parameter replacement failure after tensor 30"
    );
    assert_eq!(model.export_parameters().expect("restored"), original);
    assert_eq!(
        model.policy_identity().expect("restored identity"),
        identity
    );
}

#[test]
fn finite_extreme_parameters_fail_inference_without_nonfinite_output() {
    let model = PolicyModel::fresh(103).expect("model");
    let parameters = vec![f32::MAX; MODEL_PARAMETER_COUNT];
    model.import_parameters(&parameters).expect("finite import");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(
        model.evaluate(&frame).unwrap_err().to_string(),
        "model value output at batch 0 index 0 is non-finite"
    );
    assert_eq!(
        model.choose(&frame, &space).unwrap_err().to_string(),
        "model value output at batch 0 index 0 is non-finite"
    );
}

#[test]
fn finite_parameters_that_overflow_a_conditional_head_fail_choose_precisely() {
    let model = PolicyModel::fresh(111).expect("model");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    set_named_parameter_value(
        &model,
        &mut parameters,
        "kind.bias",
        ActionKind::Stop.index(),
        1.0,
    );
    set_named_parameter_range(
        &model,
        &mut parameters,
        "kind_embedding.weight",
        ActionKind::Stop.index() * 32,
        32,
        1.0,
    );
    for input in 256..288 {
        set_named_parameter_value(
            &model,
            &mut parameters,
            "controlled.weight",
            input * 2,
            f32::MAX,
        );
    }
    model.import_parameters(&parameters).expect("finite import");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(
        model.choose(&frame, &space).unwrap_err().to_string(),
        "model controlled output at batch 0 index 0 is non-finite"
    );
}

#[test]
fn finite_parameters_that_overflow_a_pointer_fail_choose_precisely() {
    let model = PolicyModel::fresh(112).expect("model");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    set_named_parameter_value(
        &model,
        &mut parameters,
        "kind.bias",
        ActionKind::MovePoint.index(),
        1.0,
    );
    set_named_parameter_range(&model, &mut parameters, "point.0.bias", 0, 64, 1.0);
    set_named_parameter_range(&model, &mut parameters, "point.1.bias", 0, 64, f32::MAX);
    set_named_parameter_range(&model, &mut parameters, "point_query.bias", 0, 64, f32::MAX);
    model.import_parameters(&parameters).expect("finite import");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(
        model.choose(&frame, &space).unwrap_err().to_string(),
        "model point pointer output at batch 0 index 0 is non-finite"
    );
}

#[test]
fn parameter_replacement_is_atomic_for_concurrent_export_readers() {
    let model = Arc::new(PolicyModel::fresh(104).expect("model"));
    let old = model.export_parameters().expect("old");
    let new = vec![0.125; MODEL_PARAMETER_COUNT];
    let barrier = Arc::new(Barrier::new(3));
    let writer = spawn_parameter_writer(
        Arc::clone(&model),
        Arc::clone(&barrier),
        old.clone(),
        new.clone(),
    );
    let reader = spawn_parameter_reader(
        Arc::clone(&model),
        Arc::clone(&barrier),
        old.clone(),
        new.clone(),
    );
    barrier.wait();

    writer.join().expect("writer");
    reader.join().expect("reader");
    assert_eq!(model.export_parameters().expect("final"), new);
}

#[test]
fn live_training_output_blocks_import_until_dropped() {
    let model = Arc::new(PolicyModel::fresh(114).expect("model"));
    let frame = populated_frame();
    let prefix = TrainingPrefix::new(ActionKind::Continue, None, None);
    let output = model
        .training_forward(std::slice::from_ref(&frame), &[prefix])
        .expect("training output");
    let replacement = vec![0.25; MODEL_PARAMETER_COUNT];
    let barrier = Arc::new(Barrier::new(2));
    let (complete_tx, complete_rx) = mpsc::sync_channel(1);
    let writer_model = Arc::clone(&model);
    let writer_barrier = Arc::clone(&barrier);
    let writer = thread::spawn(move || {
        writer_barrier.wait();
        writer_model
            .import_parameters(&replacement)
            .expect("replacement");
        complete_tx.send(()).expect("completion");
    });
    barrier.wait();

    assert!(!model.parameter_write_available_for_test());
    assert!(complete_rx.try_recv().is_err());
    drop(output);
    complete_rx.recv().expect("unblocked completion");
    writer.join().expect("writer");
}

#[test]
fn choose_returns_a_legal_decodable_action() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let choice = PolicyModel::fresh(6)
        .expect("model")
        .choose(&frame, &space)
        .expect("choice");

    assert!(space.allows(choice.action));
    assert!(space.decode(choice.action).is_ok());
    assert!(choice.value.is_finite());
}

#[test]
fn choose_remains_legal_when_hero_is_dead_and_courier_is_live() {
    let mut view = world_view(Team::Radiant, 10);
    let hero_id = view
        .players
        .iter_mut()
        .find(|player| player.slot == SlotId(0))
        .expect("player")
        .unit
        .take()
        .expect("hero id");
    view.units.retain(|unit| unit.id != hero_id);
    let tracker = tracker_with_view(Team::Radiant, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let choice = PolicyModel::fresh(61)
        .expect("model")
        .choose(&frame, &space)
        .expect("choice");

    assert!(space.allows(choice.action));
    assert!(space.decode(choice.action).is_ok());
}

#[test]
fn output_is_invariant_to_entity_id_remapping() {
    let first_view = world_view(Team::Radiant, 10);
    let mut second_view = first_view.clone();
    reverse_entity_ids_and_generations(&mut second_view, 50_000, 70);
    let first_tracker = tracker_with_view(Team::Radiant, first_view);
    let second_tracker = tracker_with_view(Team::Radiant, second_view);
    let first = encode(&first_tracker, &LocalPolicyState::new(0));
    let second = encode(&second_tracker, &LocalPolicyState::new(0));
    let model = PolicyModel::fresh(62).expect("model");

    assert_eq!(first, second);
    assert_eq!(
        model.evaluate(&first).expect("first"),
        model.evaluate(&second).expect("second")
    );
}

#[test]
fn choose_rejects_synthetic_lineage_and_stale_frame_provenance() {
    let model = PolicyModel::fresh(105).expect("model");
    let first_view = world_view(Team::Radiant, 10);
    let mut first_tracker = tracker_with_view(Team::Radiant, first_view.clone());
    let first_space = ActionSpace::from_tracker(&first_tracker).expect("first space");
    let frame = encode(&first_tracker, &LocalPolicyState::new(0));
    let other_tracker = tracker_with_view(Team::Radiant, first_view);
    let other_space = ActionSpace::from_tracker(&other_tracker).expect("other space");

    assert_eq!(
        model
            .choose(&FeatureFrame::new(), &first_space)
            .unwrap_err()
            .to_string(),
        "model feature frame does not belong to the supplied action space"
    );
    assert_eq!(
        model.choose(&frame, &other_space).unwrap_err().to_string(),
        "model feature frame does not belong to the supplied action space"
    );
    first_tracker
        .observe_events(1, &[])
        .expect("same-snapshot state revision");
    let revised_space = ActionSpace::from_tracker(&first_tracker).expect("revised space");
    assert_eq!(
        model
            .choose(&frame, &revised_space)
            .unwrap_err()
            .to_string(),
        "model feature frame does not belong to the supplied action space"
    );
    first_tracker
        .observe_snapshot(&world_view(Team::Radiant, 11))
        .expect("next snapshot");
    let stale_space = ActionSpace::from_tracker(&first_tracker).expect("stale space");
    assert_eq!(
        model.choose(&frame, &stale_space).unwrap_err().to_string(),
        "model feature frame does not belong to the supplied action space"
    );
}

#[test]
fn scripted_decoder_covers_every_legal_family() {
    let mut view = world_view(Team::Radiant, 10);
    view.units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(SlotId(0)))
        .expect("hero")
        .abilities[0]
        .can_level = true;
    view.players
        .iter_mut()
        .find(|player| player.slot == SlotId(0))
        .expect("player")
        .stash = Some(vec![None; 6]);
    let tracker = tracker_with_view(Team::Radiant, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut covered = [false; ActionKind::COUNT];
    for kind in ActionKind::ALL {
        assert!(
            space.kind_mask().allows(kind),
            "fixture must allow {kind:?}"
        );
        let logits = DecoderLogits::favor(kind);
        let action = decode_with_logits(&space, &logits).expect("scripted action");
        assert_eq!(action.kind(), kind);
        assert!(space.allows(action));
        assert!(space.decode(action).is_ok());
        covered[kind.index()] = true;
    }
    assert!(covered.into_iter().all(|value| value));
}

#[test]
fn scripted_decoder_selects_cast_use_and_safe_put_modes() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut cast = DecoderLogits::favor(ActionKind::Cast);
    cast.ability[0] = 100.0;
    cast.target_mode[2] = 100.0;
    let cast = decode_with_logits(&space, &cast).expect("cast");
    assert!(matches!(
        cast,
        StructuredAction::Cast {
            target: ActionTarget::Point(_),
            ..
        }
    ));

    let mut use_action = DecoderLogits::favor(ActionKind::Use);
    use_action.item[0] = 100.0;
    use_action.target_mode[0] = 100.0;
    let use_action = decode_with_logits(&space, &use_action).expect("use");
    assert!(matches!(
        use_action,
        StructuredAction::Use {
            target: ActionTarget::None,
            ..
        }
    ));

    let mut put = DecoderLogits::favor(ActionKind::PutPoint);
    put.item[0] = 100.0;
    put.put_mode[0] = 100.0;
    let put = decode_with_logits(&space, &put).expect("put");
    assert!(matches!(
        put,
        StructuredAction::PutPoint {
            target: PutPointTarget::Underfoot,
            ..
        }
    ));

    let mut unsafe_put_point = DecoderLogits::favor(ActionKind::PutPoint);
    unsafe_put_point.item[0] = 100.0;
    unsafe_put_point.put_mode[1] = 100.0;
    let put_point = decode_with_logits(&space, &unsafe_put_point).expect("masked put point");
    assert!(matches!(
        put_point,
        StructuredAction::PutPoint {
            target: PutPointTarget::Underfoot,
            ..
        }
    ));
}

#[test]
fn pointer_offsets_cannot_override_selected_put_target_mode() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut underfoot = DecoderLogits::favor(ActionKind::PutPoint);
    underfoot.item[0] = 100.0;
    underfoot.put_mode = [1.0, 0.0];
    underfoot.point.fill(1.0e30);

    let action = decode_with_logits(&space, &underfoot).expect("underfoot");
    assert!(matches!(
        action,
        StructuredAction::PutPoint {
            target: PutPointTarget::Underfoot,
            ..
        }
    ));
}

#[test]
fn pointer_offsets_cannot_override_selected_cast_or_use_target_mode() {
    let entity = [1.0e30, 2.0e30];
    let point = [3.0e30, 4.0e30];
    let none = select_target_for_test([1.0, 0.0, 0.0], &entity, &point).expect("none");
    assert_eq!(none, ActionTarget::None);

    let entity_target =
        select_target_for_test([0.0, 2.0, 1.0], &[3.0, 4.0], &point).expect("entity");
    assert_eq!(entity_target, ActionTarget::Entity(crate::EntityIndex(1)));

    let point_target =
        select_target_for_test([0.0, 1.0, 2.0], &entity, &[3.0, 4.0]).expect("point");
    assert_eq!(point_target, ActionTarget::Point(crate::PointIndex(1)));
}

#[test]
fn scripted_decoder_compares_none_entity_and_point_targets() {
    let point_space = action_space_with_aims(Aim::Point, Aim::Point);
    let mut cast_point = DecoderLogits::favor(ActionKind::Cast);
    cast_point.target_mode[2] = 100.0;
    assert!(matches!(
        decode_with_logits(&point_space, &cast_point).expect("cast point"),
        StructuredAction::Cast {
            target: ActionTarget::Point(_),
            ..
        }
    ));
    let mut use_point = DecoderLogits::favor(ActionKind::Use);
    use_point.target_mode[2] = 100.0;
    assert!(matches!(
        decode_with_logits(&point_space, &use_point).expect("use point"),
        StructuredAction::Use {
            target: ActionTarget::Point(_),
            ..
        }
    ));

    let entity_space = action_space_with_aims(Aim::Unit, Aim::Unit);
    let mut cast_entity = DecoderLogits::favor(ActionKind::Cast);
    cast_entity.target_mode[1] = 100.0;
    assert!(matches!(
        decode_with_logits(&entity_space, &cast_entity).expect("cast entity"),
        StructuredAction::Cast {
            target: ActionTarget::Entity(_),
            ..
        }
    ));
    let mut use_entity = DecoderLogits::favor(ActionKind::Use);
    use_entity.target_mode[1] = 100.0;
    assert!(matches!(
        decode_with_logits(&entity_space, &use_entity).expect("use entity"),
        StructuredAction::Use {
            target: ActionTarget::Entity(_),
            ..
        }
    ));

    let none_space = action_space_with_aims(Aim::Own, Aim::Own);
    assert!(matches!(
        decode_with_logits(&none_space, &DecoderLogits::favor(ActionKind::Cast))
            .expect("cast none"),
        StructuredAction::Cast {
            target: ActionTarget::None,
            ..
        }
    ));
    assert!(matches!(
        decode_with_logits(&none_space, &DecoderLogits::favor(ActionKind::Use)).expect("use none"),
        StructuredAction::Use {
            target: ActionTarget::None,
            ..
        }
    ));
}

#[test]
fn decoder_pointer_indices_align_with_action_space_bounds() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut follow = DecoderLogits::favor(ActionKind::FollowUnit);
    follow.entity.fill(-1.0);
    let legal = space
        .follow_entity_mask(ControlledUnit::Hero)
        .iter()
        .rposition(|allowed| *allowed)
        .expect("entity");
    follow.entity[legal] = 10.0;
    let action = decode_with_logits(&space, &follow).expect("follow");
    assert!(matches!(action, StructuredAction::FollowUnit { target, .. } if target.0 == legal));

    let mut movement = DecoderLogits::favor(ActionKind::MovePoint);
    movement.point.fill(-1.0);
    let legal = space
        .move_point_mask(ControlledUnit::Hero)
        .iter()
        .rposition(|allowed| *allowed)
        .expect("point");
    movement.point[legal] = 10.0;
    let action = decode_with_logits(&space, &movement).expect("move");
    assert!(matches!(action, StructuredAction::MovePoint { point, .. } if point.0 == legal));
}

#[test]
fn real_tensor_forward_has_all_finite_exact_head_shapes() {
    let model = PolicyModel::fresh(106).expect("model");
    let frames = vec![populated_frame(), FeatureFrame::new()];
    let prefixes = mixed_training_prefixes();
    let output = model
        .training_forward(&frames, &prefixes[..2])
        .expect("forward");
    let shapes = output.shapes();

    assert_eq!(shapes.value, vec![2, 1]);
    assert_eq!(shapes.kind, vec![2, 16]);
    assert_eq!(shapes.controlled, vec![2, 2]);
    assert_eq!(shapes.ability, vec![2, 8]);
    assert_eq!(shapes.item, vec![2, 15]);
    assert_eq!(shapes.swap, vec![2, 15]);
    assert_eq!(shapes.learn, vec![2, 6]);
    assert_eq!(shapes.shop, vec![2, 64]);
    assert_eq!(shapes.loot, vec![2, 16]);
    assert_eq!(shapes.target_mode, vec![2, 3]);
    assert_eq!(shapes.put_mode, vec![2, 2]);
    assert_eq!(shapes.entity_pointer, vec![2, 96]);
    assert_eq!(shapes.point_pointer, vec![2, 48]);
    output.validate_finite().expect("finite tensors");
}

#[test]
fn real_kind_unit_and_slot_embeddings_change_downstream_heads() {
    let model = PolicyModel::fresh(107).expect("model");
    let frame = populated_frame();
    let prefixes = [
        TrainingPrefix::new(ActionKind::Continue, None, None),
        TrainingPrefix::new(ActionKind::Stop, None, None),
        TrainingPrefix::new(ActionKind::Cast, Some(ControlledUnit::Hero), None),
        TrainingPrefix::new(ActionKind::Cast, Some(ControlledUnit::Courier), None),
        TrainingPrefix::new(
            ActionKind::Cast,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Ability(
                TrainingAbilitySlot::new(1).expect("first slot"),
            )),
        ),
        TrainingPrefix::new(
            ActionKind::Cast,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Ability(
                TrainingAbilitySlot::new(3).expect("second slot"),
            )),
        ),
        TrainingPrefix::new(
            ActionKind::Use,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Item(
                TrainingItemSlot::new(1).expect("first item slot"),
            )),
        ),
        TrainingPrefix::new(
            ActionKind::Use,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Item(
                TrainingItemSlot::new(4).expect("second item slot"),
            )),
        ),
    ];
    let snapshots = prefixes
        .iter()
        .map(|prefix| {
            model
                .training_snapshot(std::slice::from_ref(&frame), &[*prefix])
                .expect("snapshot")
        })
        .collect::<Vec<_>>();

    assert_ne!(snapshots[0].controlled, snapshots[1].controlled);
    assert_ne!(snapshots[2].ability, snapshots[3].ability);
    assert_ne!(snapshots[4].target_mode, snapshots[5].target_mode);
    assert_ne!(snapshots[6].target_mode, snapshots[7].target_mode);
}

#[test]
fn swapping_token_rows_swaps_real_pointer_logits() {
    let model = PolicyModel::fresh(108).expect("model");
    let first = populated_frame();
    let mut units = first.clone();
    units.units.swap(0, 2);
    let prefix = TrainingPrefix::new(ActionKind::FollowUnit, Some(ControlledUnit::Hero), None);
    let left = model
        .training_snapshot(std::slice::from_ref(&first), &[prefix])
        .expect("left");
    let right = model.training_snapshot(&[units], &[prefix]).expect("right");
    assert_eq!(left.entity_pointer[0], right.entity_pointer[2]);
    assert_eq!(left.entity_pointer[2], right.entity_pointer[0]);

    let mut points = first.clone();
    points.points.swap(0, 1);
    let point_prefix = TrainingPrefix::new(ActionKind::MovePoint, Some(ControlledUnit::Hero), None);
    let left = model
        .training_snapshot(&[first], &[point_prefix])
        .expect("left points");
    let right = model
        .training_snapshot(&[points], &[point_prefix])
        .expect("right points");
    assert_eq!(left.point_pointer[0], right.point_pointer[1]);
    assert_eq!(left.point_pointer[1], right.point_pointer[0]);
}

#[test]
fn illegal_high_real_kind_logit_is_excluded() {
    let model = PolicyModel::fresh(109).expect("model");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    set_named_parameter_value(
        &model,
        &mut parameters,
        "kind.bias",
        ActionKind::Continue.index(),
        1.0,
    );
    set_named_parameter_value(
        &model,
        &mut parameters,
        "kind.bias",
        ActionKind::Learn.index(),
        100.0,
    );
    model.import_parameters(&parameters).expect("parameters");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    let choice = model.choose(&frame, &space).expect("choice");
    assert_eq!(choice.action, StructuredAction::Continue);
}

#[test]
fn mixed_prefix_gradient_probe_reaches_all_named_parameters() {
    let model = PolicyModel::fresh(110).expect("model");
    let frame = populated_frame();
    let prefixes = mixed_training_prefixes();
    let frames = vec![frame; prefixes.len()];

    let gradients = model.gradient_probe(&frames, &prefixes).expect("gradients");
    assert_eq!(gradients.len(), 62);
    assert!(gradients.iter().all(|(_, present)| *present));
}

#[test]
fn guarded_named_backward_returns_stable_names_shapes_and_gradients() {
    let model = PolicyModel::fresh(115).expect("model");
    let prefixes = mixed_training_prefixes();
    let frames = vec![populated_frame(); prefixes.len()];
    let output = model
        .training_forward(&frames, &prefixes)
        .expect("training output");
    let loss = output.sum_all_heads().expect("loss");

    let gradients = model
        .backward_named(&output, &loss)
        .expect("named gradients");
    let schema = model.parameter_schema().expect("schema");

    assert_eq!(gradients.len(), 62);
    for (gradient, (name, shape)) in gradients.iter().zip(schema) {
        assert_eq!(gradient.name(), name);
        assert_eq!(gradient.parameter_shape(), shape);
        assert_eq!(gradient.gradient_shape(), Some(shape.as_slice()));
    }
}

#[test]
fn named_backward_rejects_output_from_another_model() {
    let first = PolicyModel::fresh(116).expect("first");
    let second = PolicyModel::fresh(117).expect("second");
    let frame = populated_frame();
    let prefix = TrainingPrefix::new(ActionKind::Continue, None, None);
    let output = first
        .training_forward(std::slice::from_ref(&frame), &[prefix])
        .expect("output");
    let loss = output.sum_all_heads().expect("loss");

    assert_eq!(
        second
            .backward_named(&output, &loss)
            .unwrap_err()
            .to_string(),
        "model training output belongs to a different policy model"
    );
}

fn populated_frame() -> FeatureFrame {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    encode(&tracker, &LocalPolicyState::new(0))
}

fn model_sample() -> ImitationSample {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Training, 1, 1, 10, &frame).expect("identity");
    ImitationSample::teacher(frame, &space, StructuredAction::Continue, identity).expect("sample")
}

fn action_space_with_aims(ability_aim: Aim, item_aim: Aim) -> ActionSpace {
    let mut view = world_view(Team::Radiant, 10);
    let hero = view
        .units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(SlotId(0)))
        .expect("hero");
    hero.abilities[0].aim = ability_aim;
    hero.abilities[0].range = 1_200;
    hero.items[0].as_mut().expect("item").aim = Some(item_aim);
    hero.items[0].as_mut().expect("item").range = 1_200;
    let tracker = tracker_with_view(Team::Radiant, view);
    ActionSpace::from_tracker(&tracker).expect("space")
}

fn mixed_training_prefixes() -> Vec<TrainingPrefix> {
    vec![
        TrainingPrefix::new(ActionKind::Continue, None, None),
        TrainingPrefix::new(ActionKind::AttackUnit, Some(ControlledUnit::Hero), None),
        TrainingPrefix::new(
            ActionKind::Cast,
            Some(ControlledUnit::Courier),
            Some(TrainingSlot::Ability(
                TrainingAbilitySlot::new(3).expect("ability slot"),
            )),
        ),
        TrainingPrefix::new(
            ActionKind::Use,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Item(
                TrainingItemSlot::new(4).expect("item slot"),
            )),
        ),
    ]
}

fn set_named_parameter_value(
    model: &PolicyModel,
    parameters: &mut [f32],
    name: &str,
    index: usize,
    value: f32,
) {
    let offset = named_parameter_offset(model, name);
    parameters[offset + index] = value;
}

fn set_named_parameter_range(
    model: &PolicyModel,
    parameters: &mut [f32],
    name: &str,
    index: usize,
    count: usize,
    value: f32,
) {
    let offset = named_parameter_offset(model, name);
    parameters[offset + index..offset + index + count].fill(value);
}

fn named_parameter_offset(model: &PolicyModel, name: &str) -> usize {
    let mut offset = 0usize;
    for (parameter_name, shape) in model.parameter_schema().expect("schema") {
        if parameter_name == name {
            return offset;
        }
        offset += shape.iter().product::<usize>();
    }
    panic!("missing parameter {name}");
}

fn spawn_parameter_writer(
    model: Arc<PolicyModel>,
    barrier: Arc<Barrier>,
    old: Vec<f32>,
    new: Vec<f32>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        barrier.wait();
        for index in 0..8 {
            let parameters = if index % 2 == 0 { &new } else { &old };
            model.import_parameters(parameters).expect("replace");
        }
        model.import_parameters(&new).expect("final replace");
    })
}

fn spawn_parameter_reader(
    model: Arc<PolicyModel>,
    barrier: Arc<Barrier>,
    old: Vec<f32>,
    new: Vec<f32>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        barrier.wait();
        for _ in 0..16 {
            let observed = model.export_parameters().expect("read");
            assert!(observed == old || observed == new);
        }
    })
}

fn assert_outputs_close(left: &crate::PolicyOutput, right: &crate::PolicyOutput, tolerance: f32) {
    assert!((left.value - right.value).abs() <= tolerance);
    for (left, right) in left.kind_logits.iter().zip(right.kind_logits) {
        assert!((*left - right).abs() <= tolerance);
    }
}
