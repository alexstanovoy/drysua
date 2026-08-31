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
    masked_cross_entropy_for_test, pool_groups_for_test, pool_max_gradient_for_test,
    select_target_for_test, validate_batch_count,
};
use crate::{
    ActionKind, ActionSpace, ActionTarget, AdamConfig, ControlledUnit, FeatureFrame,
    ImitationSample, LocalPolicyState, MODEL_ABILITY_HEAD, MODEL_ENTITY_POINTER_HEAD,
    MODEL_EVALUATION_MICROBATCH, MODEL_ITEM_HEAD, MODEL_KIND_HEAD, MODEL_LEARN_HEAD,
    MODEL_LOOT_HEAD, MODEL_MAX_BATCH, MODEL_PARAMETER_COUNT, MODEL_POINT_POINTER_HEAD,
    MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, MODEL_SHOP_HEAD, MODEL_SWAP_HEAD,
    MODEL_TRAINING_BATCH, MODEL_UNIT_HEAD, ModelError, PolicyModel, PutPointTarget, SampleIdentity,
    SeedNamespace, StructuredAction, TrainingAbilitySlot, TrainingItemSlot, TrainingPrefix,
    TrainingSlot, unit_feature,
};

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
    assert_eq!(MODEL_SCHEMA_VERSION, 2);
    assert_eq!(MODEL_SCHEMA_HASH, 15_888_684_091_468_496_519);
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
    assert_eq!(schema.first(), Some(&("unit.0.weight", vec![69, 64])));
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
fn scripted_decoder_selects_cast_and_use_target_modes_and_put_modes() {
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

    let mut put_point = DecoderLogits::favor(ActionKind::PutPoint);
    put_point.item[0] = 100.0;
    put_point.put_mode[1] = 100.0;
    let put_point = decode_with_logits(&space, &put_point).expect("put point");
    assert!(matches!(
        put_point,
        StructuredAction::PutPoint {
            target: PutPointTarget::Point(_),
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
