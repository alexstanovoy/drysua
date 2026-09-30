#![allow(
    clippy::float_arithmetic,
    reason = "independent numerical regression references"
)]

use bota_proto::Team;

use super::feature::{encode, tracker_with_view, world_view};
use crate::model::{adam_step_for_test, masked_ppo_entropy_for_test};
use crate::{
    ActionKind, ActionSpace, AdamConfig, ControlledUnit, FeatureFrame, HeadTarget,
    LocalPolicyState, MODEL_EVALUATION_MICROBATCH, MODEL_KIND_HEAD, MODEL_PARAMETER_COUNT,
    MODEL_TRAINING_BATCH, PolicyDevice, PolicyModel, PpoConfig, PpoPreparedSample,
    TrainingAbilitySlot, TrainingItemSlot, TrainingPrefix, TrainingSlot,
};

#[test]
fn model_roundtrip_preserves_predictions_and_microbatch_shapes() {
    let source = PolicyModel::fresh(101).expect("source");
    let target = PolicyModel::fresh(102).expect("target");
    let parameters = source.export_parameters().expect("parameters");
    assert_eq!(parameters.len(), MODEL_PARAMETER_COUNT);
    assert!((4 * 1_048_576..=12 * 1_048_576).contains(&(parameters.len() * size_of::<f32>())));
    assert_eq!(
        parameters,
        PolicyModel::fresh(101)
            .expect("same seed")
            .export_parameters()
            .expect("replayed parameters")
    );
    assert_ne!(
        parameters,
        target.export_parameters().expect("different seed")
    );
    target.import_parameters(&parameters).expect("load");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frames = vec![encode(&tracker, &LocalPolicyState::new(0)); MODEL_EVALUATION_MICROBATCH + 1];
    let expected = source.evaluate_batch(&frames).expect("source predictions");
    assert_eq!(
        target.evaluate_batch(&frames).expect("loaded predictions"),
        expected
    );
    assert_eq!(expected.len(), frames.len());
    assert!(expected.iter().all(|output| output.is_finite()));
    assert_eq!(
        expected[MODEL_EVALUATION_MICROBATCH],
        source.evaluate(&frames[0]).expect("scalar")
    );
    assert_training_shapes_and_backward(&source, &frames[..4]);
}

fn assert_training_shapes_and_backward(source: &PolicyModel, frames: &[FeatureFrame]) {
    assert_eq!(frames.len(), 4);
    let prefixes = [
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
    ];
    let output = source
        .training_forward(frames, &prefixes)
        .expect("training");
    for (tensor, width) in [
        (output.value(), 1),
        (output.kind(), 16),
        (output.controlled(), 2),
        (output.ability(), 8),
        (output.item(), 15),
        (output.swap(), 15),
        (output.learn(), 6),
        (output.shop(), 64),
        (output.loot(), 16),
        (output.target_mode(), 3),
        (output.put_mode(), 2),
        (output.entity_pointer(), 96),
        (output.point_pointer(), 64),
    ] {
        assert_eq!(tensor.dims(), &[prefixes.len(), width]);
    }
    output.validate_finite().expect("finite heads");
    let loss = output.sum_all_heads().expect("loss");
    let gradients = source.backward_named(&output, &loss).expect("backward");
    let schema = source.parameter_schema().expect("schema");
    assert_eq!(gradients.len(), schema.len());
    for (gradient, (name, shape)) in gradients.iter().zip(schema) {
        assert_eq!(gradient.name(), name);
        assert_eq!(gradient.parameter_shape(), shape);
        match gradient.gradient_shape() {
            Some(actual) => assert_eq!(actual, shape.as_slice()),
            None => assert!(
                name.starts_with("dire."),
                "Radiant fixture must differentiate {name}"
            ),
        }
        if name.starts_with("dire.")
            && let Some(tensor) = gradient.gradient()
        {
            let values = tensor
                .flatten_all()
                .expect("flat inactive gradient")
                .to_vec1::<f32>()
                .expect("inactive gradient");
            assert!(values.iter().all(|value| *value == 0.0), "inactive {name}");
        }
    }
}

#[test]
fn parameter_import_rejects_invalid_and_partial_replacements_atomically() {
    let model = PolicyModel::fresh(113).expect("model");
    let original = model.export_parameters().expect("original");
    let identity = model.policy_identity().expect("identity");
    let mut nonfinite = original.clone();
    nonfinite[10] = f32::NAN;
    for (values, failure, message) in [
        (
            original[..original.len() - 1].to_vec(),
            None,
            format!(
                "model parameter length {} differs from expected {}",
                original.len() - 1,
                original.len()
            ),
        ),
        (
            nonfinite,
            None,
            "model parameter 10 is non-finite".to_owned(),
        ),
        (
            vec![0.375; MODEL_PARAMETER_COUNT],
            Some(30),
            "model injected parameter replacement failure after tensor 30".to_owned(),
        ),
    ] {
        let error = match failure {
            Some(index) => model.import_parameters_with_failure(&values, index),
            None => model.import_parameters(&values),
        }
        .expect_err("invalid import");
        assert_eq!(error.to_string(), message);
        assert_eq!(model.export_parameters().expect("restored"), original);
        assert_eq!(
            model.policy_identity().expect("restored identity"),
            identity
        );
    }
}

#[test]
fn invalid_frames_and_extreme_parameters_fail_without_nonfinite_predictions() {
    let model = PolicyModel::fresh(103).expect("model");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    assert_eq!(
        model.evaluate_batch(&[]).expect_err("empty").to_string(),
        "model batch must contain at least one frame"
    );
    assert_eq!(
        model
            .training_forward(&[], &[])
            .expect_err("empty training")
            .to_string(),
        "model training batch must contain at least one frame"
    );
    assert_eq!(
        model
            .training_forward(std::slice::from_ref(&frame), &[])
            .expect_err("prefix count")
            .to_string(),
        "model training prefix count 0 differs from frame count 1"
    );
    assert_eq!(
        model
            .choose(&test_frame(), &space)
            .expect_err("unbound frame")
            .to_string(),
        "model feature frame does not belong to the supplied action space"
    );
    model
        .import_parameters(&vec![f32::MAX; MODEL_PARAMETER_COUNT])
        .expect("finite import");
    assert_eq!(
        model.evaluate(&frame).expect_err("overflow").to_string(),
        "model value output at batch 0 index 0 is non-finite"
    );
    assert_eq!(
        model
            .choose(&frame, &space)
            .expect_err("overflow choice")
            .to_string(),
        "model value output at batch 0 index 0 is non-finite"
    );
}

#[test]
fn absent_tokens_cannot_inject_garbage_into_predictions() {
    let model = PolicyModel::fresh(2).expect("model");
    let clean = test_frame();
    let mut garbage = clean.clone();
    garbage.units[7][1..].fill(900.0);
    garbage.abilities[5][1..].fill(-700.0);
    garbage.items[12][1..].fill(500.0);
    garbage.points[9][1..].fill(300.0);
    garbage.projectiles[4][1..].fill(-200.0);
    garbage.loot[3][1..].fill(100.0);
    let expected = model.evaluate(&clean).expect("empty");
    assert!(expected.is_finite());
    assert_eq!(model.evaluate(&garbage).expect("masked garbage"), expected);
}

#[test]
fn fresh_heads_are_unsaturated_and_pointer_scaling_has_the_expected_gradient() {
    let model = PolicyModel::fresh(503).expect("model");
    let frames = [test_frame()];
    let prefixes = [TrainingPrefix::new(ActionKind::Continue, None, None)];
    let output = model.training_forward(&frames, &prefixes).expect("forward");
    for head in [
        output.kind(),
        output.controlled(),
        output.ability(),
        output.item(),
        output.swap(),
        output.learn(),
        output.shop(),
        output.loot(),
        output.target_mode(),
        output.put_mode(),
        output.entity_pointer(),
        output.point_pointer(),
    ] {
        let row = head.to_vec2::<f32>().expect("logits").remove(0);
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let minimum = row.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(maximum - minimum < 2.0);
        assert!(row.iter().all(|value| value.is_finite()));
    }
    for width in [64, 128] {
        let tokens = candle_core::Tensor::ones(
            (1, 2, width),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .expect("tokens");
        let query = candle_core::Var::ones(
            (1, 1, width),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .expect("query");
        let scores = crate::model::scaled_pointer_dot(&tokens, query.as_tensor()).expect("pointer");
        for value in scores
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("scores")
        {
            assert!((value - (width as f32).sqrt()).abs() < 1.0e-6);
        }
        let gradients = scores.sum_all().expect("sum").backward().expect("backward");
        let gradient = gradients.get(query.as_tensor()).expect("query gradient");
        for value in gradient
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("gradient")
        {
            assert!((value - 2.0 / (width as f32).sqrt()).abs() < 1.0e-6);
        }
    }
}

#[test]
fn masked_entropy_is_finite_for_inactive_singleton_and_extreme_logits() {
    assert_entropy(PolicyDevice::Cpu);
}

fn assert_entropy(device: PolicyDevice) {
    let model = PolicyModel::fresh(9_101).expect("fixture");
    let mut samples = sampled_examples(&model, 4);
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
    let references = samples.iter().collect::<Vec<_>>();
    let probe = masked_ppo_entropy_for_test(&logits, &references, device).expect("entropy");
    assert!(probe.log_normalizer.iter().all(|value| value.is_finite()));
    assert!(
        probe
            .gradients
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    for index in 0..2 {
        assert_eq!(probe.entropy[index], 0.0);
        assert_eq!(probe.gradients[index], vec![0.0; MODEL_KIND_HEAD]);
    }
    let probability = 1.0f64 / (1.0 + 1.0f64.exp());
    let entropy = -probability * probability.ln() - (1.0 - probability) * (1.0 - probability).ln();
    assert!((f64::from(probe.entropy[2]) - entropy).abs() < 1.0e-6);
    for (index, sign) in [(0, 1.0), (1, -1.0)] {
        assert!(
            (f64::from(probe.gradients[2][index]) - sign * probability * (1.0 - probability)).abs()
                < 1.0e-6
        );
    }
    assert_eq!(&probe.gradients[2][2..], &[0.0; MODEL_KIND_HEAD - 2]);
    assert!((probe.entropy[3] - 3.0f32.ln()).abs() < 1.0e-4);
    assert!(probe.gradients[3].iter().all(|value| value.abs() < 1.0e-4));
    let inactive = masked_ppo_entropy_for_test(&logits[..1], &references[..1], device)
        .expect("inactive batch");
    assert_eq!(inactive.entropy, probe.entropy[..1]);
    assert_eq!(inactive.gradients, probe.gradients[..1]);
}

#[test]
fn adam_clipping_and_nonzero_moments_match_an_independent_scalar_reference() {
    let config = AdamConfig::default();
    for gradient in [0.25, 0.5, 1.0] {
        let result =
            adam_step_for_test(&[1.0], &[gradient], &[0.0], &[0.0], 0, config).expect("step");
        assert_eq!(result.applied_scale, if gradient > 0.5 { 0.5 } else { 1.0 });
    }
    let result = adam_step_for_test(
        &[1.0, -2.0],
        &[3.0, 4.0],
        &[0.1, -0.2],
        &[0.01, 0.04],
        4,
        config,
    )
    .expect("step");
    assert!((result.unclipped_norm - 5.0).abs() <= f64::EPSILON);
    assert!((result.applied_scale - 0.1).abs() <= f64::EPSILON);
    for (index, clipped) in [0.3f32, 0.4].into_iter().enumerate() {
        let first = config.beta1 * [0.1, -0.2][index] + (1.0 - config.beta1) * clipped;
        let second = config.beta2 * [0.01, 0.04][index] + (1.0 - config.beta2) * clipped * clipped;
        let first_hat = f64::from(first) / (1.0 - f64::from(config.beta1).powi(5));
        let second_hat = f64::from(second) / (1.0 - f64::from(config.beta2).powi(5));
        let expected = [1.0, -2.0][index]
            - f64::from(config.learning_rate) * first_hat
                / (second_hat.sqrt() + f64::from(config.epsilon));
        assert!((result.parameters[index] - expected as f32).abs() <= 1.0e-7);
        assert!((result.first_moment[index] - first).abs() <= 1.0e-7);
        assert!((result.second_moment[index] - second).abs() <= 1.0e-7);
    }
}

#[test]
fn critic_update_reduces_value_loss_without_changing_actor_parameters() {
    assert_critic(PolicyDevice::Cpu);
}

fn assert_critic(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9_101, device).expect("model");
    let mut samples = sampled_examples(&model, 16);
    for sample in &mut samples {
        sample.advantage = 0.0;
        sample.return_value = sample.transition.old_value + 10.0;
    }
    let references = samples.iter().collect::<Vec<_>>();
    let config = PpoConfig {
        learning_rate: 3.0e-4,
        entropy_coefficient: 0.0,
        ..PpoConfig::default()
    };
    let mut adam = model.claim_optimizer(config.adam()).expect("optimizer");
    let before = model.export_parameters().expect("before");
    let report = model
        .ppo_update(&references, &mut adam, config)
        .expect("critic update");
    let (_, after) = model
        .ppo_likelihood_for_test(&references)
        .expect("likelihood");
    assert!(report.applied);
    assert!(after.value_loss < report.value_loss);
    assert!(after.approximate_kl.abs() < 1.0e-6);
    let parameters = model.export_parameters().expect("after");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let end = offset + shape.iter().product::<usize>();
        if !name.starts_with("value.") {
            assert_eq!(
                &before[offset..end],
                &parameters[offset..end],
                "critic changed {name}"
            );
        }
        offset = end;
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires an authorized CUDA runner"]
fn cuda_entropy_and_critic_preserve_numerical_contracts() {
    assert_entropy(PolicyDevice::Cuda { ordinal: 0 });
    assert_critic(PolicyDevice::Cuda { ordinal: 0 });
}

#[test]
fn rejected_uneven_updates_restore_parameters_moments_and_identity() {
    for count in [65, 81] {
        let (model, mut adam, samples, config) = rollback_fixture(count);
        let references = samples.iter().collect::<Vec<_>>();
        let before = model.coherent_snapshot(&adam).expect("before");
        let identity = model.policy_identity().expect("identity");
        let report = model
            .ppo_update(&references, &mut adam, config)
            .expect("KL guard");
        assert!(!report.applied);
        assert_eq!(report.samples, count);
        assert!(report.approximate_kl > f64::from(config.target_kl));
        assert!((report.approximate_kl - candidate_weighted_kl(count)).abs() < 1.0e-6);
        assert_eq!(model.coherent_snapshot(&adam).expect("restored"), before);
        assert_eq!(
            model.policy_identity().expect("restored identity"),
            identity
        );
        let error = model
            .ppo_update_with_faults_for_test(&references, &mut adam, config, false)
            .expect_err("candidate failure");
        assert_eq!(
            error.to_string(),
            "model tensor operation failed: injected PPO candidate evaluation failure after 64 rows"
        );
        assert_eq!(
            model.coherent_snapshot(&adam).expect("error rollback"),
            before
        );
        assert_eq!(model.policy_identity().expect("error identity"), identity);
    }
}

#[test]
fn rollback_failure_retains_the_original_candidate_error() {
    let (model, mut adam, samples, config) = rollback_fixture(65);
    let references = samples.iter().collect::<Vec<_>>();
    let error = model
        .ppo_update_with_faults_for_test(&references, &mut adam, config, true)
        .expect_err("rollback failure");
    assert_eq!(
        error.to_string(),
        "model tensor operation failed: PPO candidate rejected (model tensor operation failed: injected PPO candidate evaluation failure after 64 rows); parameter rollback failed (model injected parameter replacement failure after tensor 0)"
    );
}

fn rollback_fixture(
    count: usize,
) -> (
    PolicyModel,
    crate::AdamState,
    Vec<PpoPreparedSample>,
    PpoConfig,
) {
    assert!([65, 81].contains(&count));
    let model = PolicyModel::fresh(9_101).expect("model");
    let base = sampled_examples(&model, MODEL_TRAINING_BATCH);
    let samples = (0..count)
        .map(|index| base[index % base.len()].clone())
        .collect::<Vec<_>>();
    let config = PpoConfig {
        learning_rate: 0.01,
        ..PpoConfig::default()
    };
    let mut adam = model.claim_optimizer(config.adam()).expect("optimizer");
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
            PpoConfig {
                entropy_coefficient: 0.0,
                ..config
            },
        )
        .expect("warm moments");
    assert!(report.applied);
    assert_eq!(adam.step(), 1);
    assert!(adam.moments().0.iter().any(|value| *value != 0.0));
    (model, adam, samples, config)
}

fn candidate_weighted_kl(count: usize) -> f64 {
    let (model, mut adam, samples, config) = rollback_fixture(count);
    let references = samples.iter().collect::<Vec<_>>();
    let report = model
        .ppo_update(
            &references,
            &mut adam,
            PpoConfig {
                target_kl: 1.0e10,
                ..config
            },
        )
        .expect("retain candidate");
    assert!(report.applied);
    let mut weighted = 0.0;
    let mut chunks = Vec::new();
    for chunk in references.chunks(MODEL_TRAINING_BATCH) {
        let (_, report) = model
            .ppo_likelihood_for_test(chunk)
            .expect("candidate likelihood");
        weighted += report.approximate_kl * chunk.len() as f64;
        chunks.push(report.approximate_kl);
    }
    let expected = weighted / count as f64;
    let unweighted = chunks.iter().sum::<f64>() / chunks.len() as f64;
    assert!(
        (expected - unweighted).abs() > 1.0e-6,
        "fixture distinguishes row weighting"
    );
    expected
}

fn sampled_examples(model: &PolicyModel, count: usize) -> Vec<PpoPreparedSample> {
    let (frames, spaces, mut random) = super::ppo::sampling_inputs(count);
    model
        .sample_batch(&frames, &spaces, &mut random)
        .expect("samples")
        .into_iter()
        .enumerate()
        .map(|(index, choice)| {
            assert!(spaces[index].allows(choice.action()));
            super::ppo::prepared_choice(index, choice)
        })
        .collect()
}

fn test_frame() -> FeatureFrame {
    let mut frame = FeatureFrame::new();
    frame.global[crate::global_feature::SIDE_RADIANT] = 1.0;
    frame
}
