#![allow(
    clippy::float_arithmetic,
    reason = "independent numerical regression references"
)]

use bota_proto::Team;

use super::feature::{encode, tracker_with_view, world_view};
use crate::model::{adam_step_for_test, masked_ppo_entropy_for_test};
use crate::{
    ActionKind, ActionSpace, AdamConfig, ControlledUnit, FeatureFrame, HeadTarget,
    LocalPolicyState, MODEL_KIND_HEAD, MODEL_PARAMETER_COUNT, MODEL_TRAINING_BATCH, PolicyDevice,
    PolicyModel, PpoConfig, PpoPreparedSample, TrainingAbilitySlot, TrainingItemSlot,
    TrainingPrefix, TrainingSlot,
};

#[test]
fn parameter_roundtrip_reproduces_seeded_weights_and_greedy_predictions() {
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
    let (frames, spaces, _) = super::ppo::sampling_inputs(MODEL_TRAINING_BATCH);
    let expected = source.choose_batch(&frames, &spaces).expect("source");
    let actual = target.choose_batch(&frames, &spaces).expect("loaded");
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual.action, expected.action, "row {index}");
        assert_eq!(
            actual.value.to_bits(),
            expected.value.to_bits(),
            "row {index}"
        );
    }
}

#[test]
fn fresh_heads_start_unsaturated_and_radiant_losses_reach_only_radiant_parameters() {
    let model = PolicyModel::fresh(503).expect("model");
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frames = vec![encode(&tracker, &LocalPolicyState::new(0)); 4];
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
    let output = model
        .training_forward(&frames, &prefixes)
        .expect("training");
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
        for row in head.to_vec2::<f32>().expect("logits") {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let minimum = row.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(maximum - minimum < 2.0, "{:?}", head.dims());
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
    assert_radiant_gradient_routing(&model, &output);
}

/// Every Radiant parameter receives a gradient of its shape; Dire ones none.
fn assert_radiant_gradient_routing(model: &PolicyModel, output: &crate::PolicyTensorOutput<'_>) {
    let loss = output.sum_all_heads().expect("loss");
    let gradients = model.backward_named(output, &loss).expect("backward");
    let schema = model.parameter_schema().expect("schema");
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
        model.choose_batch(&[], &[]).expect_err("empty").to_string(),
        "model batch must contain at least one frame"
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
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let clean = encode(&tracker, &LocalPolicyState::new(0));
    // Encoded frames always carry every ability token.
    let mut garbage = clean.clone();
    fill_absent_token("units", &mut garbage.units[..], 900.0);
    fill_absent_token("items", &mut garbage.items[..], 500.0);
    fill_absent_token("points", &mut garbage.points[..], 300.0);
    fill_absent_token("projectiles", &mut garbage.projectiles[..], -200.0);
    fill_absent_token("loot", &mut garbage.loot[..], 100.0);
    let expected = model.choose(&clean, &space).expect("clean");
    let actual = model.choose(&garbage, &space).expect("masked garbage");
    assert_eq!(actual.action, expected.action);
    assert_eq!(actual.value.to_bits(), expected.value.to_bits());
}

/// Fills every feature but the presence flag of the last absent token.
fn fill_absent_token<const FEATURES: usize>(
    table: &str,
    tokens: &mut [[f32; FEATURES]],
    value: f32,
) {
    let token = tokens
        .iter_mut()
        .rev()
        .find(|token| token[0] == 0.0)
        .unwrap_or_else(|| panic!("an absent {table} token"));
    token[1..].fill(value);
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
fn critic_update_reduces_value_loss_and_trains_the_trunk_but_no_actor_head() {
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
    let mut trunk_changed = false;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let end = offset + shape.iter().product::<usize>();
        let changed = before[offset..end] != parameters[offset..end];
        let encoder = [
            "unit.",
            "ability.",
            "item.",
            "point.",
            "projectile.",
            "loot.",
            "trunk.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix));
        trunk_changed |= name.starts_with("trunk.") && changed;
        if !encoder && !name.starts_with("value.") {
            assert!(!changed, "critic changed actor parameter {name}");
        }
        offset = end;
    }
    assert!(trunk_changed, "the value loss trains the shared trunk");
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "needs a CUDA device"]
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

#[test]
fn staged_update_rejects_invalid_rows_and_any_nonfinite_output_without_mutation() {
    let (model, mut adam, samples, config) = rollback_fixture(65);
    let before = model.coherent_snapshot(&adam).expect("before");
    let update = |samples: &[PpoPreparedSample], adam: &mut crate::AdamState| {
        let references = samples.iter().collect::<Vec<_>>();
        model
            .ppo_update(&references, adam, config)
            .expect_err("invalid update")
            .to_string()
    };
    let mut invalid = samples.clone();
    invalid[64].transition.frame.global[0] = f32::NAN;
    assert_eq!(
        update(&invalid, &mut adam),
        "model frame 64 contains a non-finite value"
    );
    let mut invalid = samples.clone();
    invalid[64].transition.target.kind.selected = MODEL_KIND_HEAD;
    assert_eq!(
        update(&invalid, &mut adam),
        "model behavioral target label 16 is illegal for head kind"
    );
    assert_eq!(model.coherent_snapshot(&adam).expect("unchanged"), before);
    model.poison_item_head_for_test().expect("poison");
    assert_eq!(
        update(&samples, &mut adam),
        "model radiant.item output at batch 0 index 0 is non-finite"
    );
    assert_eq!(adam.step(), before.adam.step());
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
    // A zero read-out keeps the critic warm-up below away from the shared trunk,
    // so only the candidate actor step can move the policy.
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let end = offset + shape.iter().product::<usize>();
        if name == "value.1.weight" {
            parameters[offset..end].fill(0.0);
        }
        offset = end;
    }
    model.import_parameters(&parameters).expect("zero read-out");
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
    assert!(adam.moments().unwrap().0.iter().any(|value| *value != 0.0));
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
