#![allow(
    clippy::float_arithmetic,
    reason = "behavioral-training reference checks use floating-point arithmetic"
)]

use bota_proto::{Aim, MapId, SlotId, Team, UnitKind};

use super::feature::{encode, tracker_with_view, world_view};
use crate::model::{DecoderLogits, decode_with_logits};
use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, ActionKind, ActionSpace, AdamConfig,
    BehavioralTarget, BehavioralTrainer, DaggerStatistics, EarlyStoppingConfig, FeatureFrame,
    ImitationPool, ImitationSample, ImitationSide, ImitationSplit, ItemReadiness,
    LearnerMatchOutcome, LearnerTeacherResult, LocalPolicyState, MAX_IMITATION_SAMPLES,
    MAX_SEED_NAMESPACE, MAX_TRAINING_COUNTER, MODEL_PARAMETER_COUNT, OfflineEvaluation,
    OrderPersistence, PairedGameplayReport, PairedSeedResult, PolicyModel, PromotionGateInput,
    RolloutAudit, SampleIdentity, SeedNamespace, SeedNamespaces, ShuffleState, StructuredAction,
    Teacher, TeacherCoverage, TrainingCheckpoint, TrainingScope,
};

#[test]
fn action_and_training_schema_identities_are_stable() {
    assert_eq!(ACTION_SCHEMA_VERSION, 1);
    assert_eq!(ACTION_SCHEMA_HASH, 17_797_499_074_169_920_257);
    assert_eq!(crate::IMITATION_OPTIMIZER_VERSION, 2);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 1);
}

#[test]
fn target_reconstructs_every_family_with_exact_active_path() {
    let (frame, space) = complete_fixture_for(ImitationSide::Radiant);
    let mut covered = [false; ActionKind::COUNT];
    for kind in ActionKind::ALL {
        let action = decode_with_logits(&space, &DecoderLogits::favor(kind)).expect("action");
        let target = BehavioralTarget::from_action(&frame, &space, action)
            .unwrap_or_else(|error| panic!("{kind:?}: {error:?}"));
        assert_eq!(target.reconstruct_action().expect("reconstructed"), action);
        assert_eq!(target.active_head_count(), expected_active_heads(action));
        assert!(target.validate().is_ok());
        covered[kind.index()] = true;
    }
    assert!(covered.into_iter().all(|covered| covered));
}

#[test]
fn target_modes_are_all_reconstructed_without_skipped_cases() {
    for (aim, mode) in [(Aim::Own, 0), (Aim::Unit, 1), (Aim::Point, 2)] {
        for kind in [ActionKind::Cast, ActionKind::Use] {
            let (frame, space) = fixture_with_aim(aim, kind);
            let mut logits = DecoderLogits::favor(kind);
            logits.target_mode[mode] = 10.0;
            let action = decode_with_logits(&space, &logits).expect("targeted action");
            let target = BehavioralTarget::from_action(&frame, &space, action).expect("target");
            assert_eq!(target.target_mode.selected, mode);
            assert_eq!(target.reconstruct_action().expect("reconstructed"), action);
        }
    }
    let (frame, space) = complete_fixture_for(ImitationSide::Radiant);
    for mode in 0..2 {
        let mut logits = DecoderLogits::favor(ActionKind::PutPoint);
        logits.put_mode[mode] = 10.0;
        let action = decode_with_logits(&space, &logits).expect("put action");
        let target = BehavioralTarget::from_action(&frame, &space, action).expect("target");
        assert_eq!(target.put_mode.selected, mode);
        assert_eq!(target.reconstruct_action().expect("reconstructed"), action);
    }
}

#[test]
fn padded_target_mask_rejects_oversize_input() {
    assert_eq!(
        crate::imitation::padded_mask_for_test::<2>(&[true, false, true])
            .unwrap_err()
            .to_string(),
        "imitation mask width 3 exceeds head width 2"
    );
}

#[test]
fn sample_rejects_stale_provenance_and_dagger_outside_train() {
    let (frame, space) = complete_fixture_for(ImitationSide::Radiant);
    let (_, other_space) = complete_fixture_for(ImitationSide::Radiant);
    let train_identity = identity(&frame, ImitationSplit::Train, 1);
    assert_eq!(
        ImitationSample::teacher(
            frame.clone(),
            &other_space,
            StructuredAction::Continue,
            train_identity
        )
        .unwrap_err()
        .to_string(),
        "imitation feature frame does not belong to the supplied action space"
    );
    let validation = identity(&frame, ImitationSplit::Validation, 2);
    assert_eq!(
        ImitationSample::dagger(
            frame,
            &space,
            StructuredAction::Continue,
            StructuredAction::Continue,
            validation,
        )
        .unwrap_err()
        .to_string(),
        "imitation DAgger sample must belong to Train"
    );
}

#[test]
fn side_is_derived_from_actual_frame_and_metrics_are_authentic() {
    let radiant = sample(
        ImitationSplit::HeldOut,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    let dire = sample(
        ImitationSplit::HeldOut,
        ImitationSide::Dire,
        ActionKind::Continue,
        2,
    );
    assert_eq!(radiant.side(), ImitationSide::Radiant);
    assert_eq!(dire.side(), ImitationSide::Dire);
    let mut pool = pool(2, 100);
    pool.push(radiant).expect("radiant");
    pool.push(dire).expect("dire");
    let mut coverage = TeacherCoverage::new();
    record_held_out_coverage(&mut coverage, &pool);
    let model = zero_model(1);
    let evaluation =
        OfflineEvaluation::evaluate_held_out(&model, &pool, coverage).expect("evaluation");
    assert_eq!(
        evaluation.candidate(),
        model.policy_identity().expect("identity")
    );
    assert_eq!(evaluation.metrics().radiant.samples, 1);
    assert_eq!(evaluation.metrics().dire.samples, 1);
    assert_eq!(evaluation.metrics().radiant.teacher_coverage(), Some(1.0));
    assert_eq!(evaluation.metrics().dire.teacher_coverage(), Some(1.0));
}

#[test]
fn seed_and_trajectory_metadata_never_change_model_inputs_or_update() {
    let (frame, space) = complete_fixture_for(ImitationSide::Radiant);
    let first = ImitationSample::teacher(
        frame.clone(),
        &space,
        StructuredAction::Continue,
        SampleIdentity::from_frame(SeedNamespace::Training, 1, 10, 10, &frame)
            .expect("first identity"),
    )
    .expect("first sample");
    let second = ImitationSample::teacher(
        frame.clone(),
        &space,
        StructuredAction::Continue,
        SampleIdentity::from_frame(SeedNamespace::Training, 999, 20, 10, &frame)
            .expect("second identity"),
    )
    .expect("second sample");
    assert_eq!(first.frame(), second.frame());
    assert_eq!(first.target(), second.target());
    let first_model = zero_model(20);
    let second_model = zero_model(21);
    let mut first_adam = first_model
        .claim_adam_for_test(AdamConfig::default())
        .expect("Adam");
    let mut second_adam = second_model
        .claim_adam_for_test(AdamConfig::default())
        .expect("Adam");
    assert_eq!(
        first_model
            .behavioral_update(&[&first], &mut first_adam)
            .expect("first update"),
        second_model
            .behavioral_update(&[&second], &mut second_adam)
            .expect("second update")
    );
    assert_eq!(
        first_model.export_parameters().expect("first parameters"),
        second_model.export_parameters().expect("second parameters")
    );
}

#[test]
fn held_out_api_rejects_train_input_instead_of_filtering_it() {
    let train = sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    let held_out = sample(
        ImitationSplit::HeldOut,
        ImitationSide::Dire,
        ActionKind::Continue,
        2,
    );
    let mut pool = pool(2, 111);
    pool.push(train).expect("train");
    pool.push(held_out).expect("held out");
    let mut coverage = TeacherCoverage::new();
    coverage.record_represented().expect("coverage");
    assert_eq!(
        OfflineEvaluation::evaluate_held_out_samples(
            &zero_model(2),
            &pool,
            &[pool.get(0).expect("train")],
            coverage,
        )
        .unwrap_err()
        .to_string(),
        "imitation held-out evaluation received non-HeldOut or DAgger data"
    );
}

#[test]
fn held_out_evaluation_rejects_unbound_tautological_coverage_counts() {
    let held_out = sample(
        ImitationSplit::HeldOut,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    let mut pool = pool(1, 114);
    pool.push(held_out).expect("held out");
    let mut coverage = TeacherCoverage::new();
    coverage.record_represented().expect("unbound count");
    assert_eq!(
        OfflineEvaluation::evaluate_held_out(&zero_model(3), &pool, coverage)
            .unwrap_err()
            .to_string(),
        "imitation teacher coverage counts are inconsistent"
    );
}

#[test]
fn held_out_evaluation_rejects_same_identity_from_outside_the_pool() {
    let held_out = sample(
        ImitationSplit::HeldOut,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    let mut pool = pool(1, 115);
    pool.push(held_out).expect("held out");
    let cloned = pool.get(0).expect("held out reference").clone();
    let mut coverage = TeacherCoverage::new();
    coverage.record_represented_for(&cloned).expect("coverage");
    assert_eq!(
        OfflineEvaluation::evaluate_held_out_samples(&zero_model(15), &pool, &[&cloned], coverage,)
            .unwrap_err()
            .to_string(),
        "imitation held-out evaluation received a sample outside the pool"
    );
}

#[test]
fn pool_enforces_seed_identity_revision_and_protected_eviction() {
    assert!(ImitationPool::new(MAX_IMITATION_SAMPLES + 1, 1, seeds(), scope(),).is_err());
    let mut pool = pool(2, 101);
    let validation = sample(
        ImitationSplit::Validation,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    let duplicate = validation.clone();
    pool.push(validation).expect("validation");
    let revision = pool.provenance();
    assert_eq!(
        pool.push(duplicate).unwrap_err().to_string(),
        "imitation sample identity is duplicated"
    );
    assert_eq!(pool.provenance(), revision);
    let (frame, space) = complete_fixture_for(ImitationSide::Radiant);
    let wrong_identity =
        SampleIdentity::from_frame(SeedNamespace::Training, 99, 50, 10, &frame).expect("identity");
    let wrong_seed =
        ImitationSample::teacher(frame, &space, StructuredAction::Continue, wrong_identity)
            .expect("sample");
    assert_eq!(
        pool.push(wrong_seed).unwrap_err().to_string(),
        "imitation training seed 99 is absent from its seed namespace"
    );
    pool.push(sample(
        ImitationSplit::HeldOut,
        ImitationSide::Dire,
        ActionKind::Continue,
        2,
    ))
    .expect("held out");
    assert_eq!(
        pool.push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            3,
        ))
        .unwrap_err()
        .to_string(),
        "imitation pool is full and has no evictable Train sample"
    );
}

#[test]
fn pool_evicts_only_oldest_train_and_filters_training_order() {
    let mut pool = pool(3, 102);
    pool.push(sample(
        ImitationSplit::Validation,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    ))
    .expect("validation");
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Stop,
        2,
    ))
    .expect("train");
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Dire,
        ActionKind::Hold,
        3,
    ))
    .expect("train");
    let evicted = pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            4,
        ))
        .expect("push")
        .expect("evicted");
    assert_eq!(evicted.teacher_action().kind(), ActionKind::Stop);
    assert_eq!(
        pool.get(0).expect("protected").split(),
        ImitationSplit::Validation
    );
    let mut shuffle = ShuffleState::new(7);
    assert_eq!(pool.training_order(&mut shuffle).expect("order").len(), 2);
}

#[test]
fn pool_remembers_bounded_identity_history_after_train_eviction() {
    let mut pool = pool(1, 113);
    let first = sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    );
    pool.push(first.clone()).expect("first");
    let second = sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        2,
    );
    pool.push(second.clone()).expect("second");
    assert_eq!(
        pool.push(first).unwrap_err().to_string(),
        "imitation sample identity is duplicated"
    );
    pool.clear_identity_history_for_test();
    assert_eq!(
        pool.push(second).unwrap_err().to_string(),
        "imitation sample identity is duplicated"
    );
}

#[test]
fn trainer_rebinds_after_same_lineage_pool_revision_without_resetting_state() {
    let mut imitation_pool = pool(2, 116);
    imitation_pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            1,
        ))
        .expect("first sample");
    let model = PolicyModel::fresh(16).expect("model");
    let mut trainer = make_trainer(2, 17, &model, 2, &imitation_pool);
    trainer
        .train_epoch(&model, &imitation_pool)
        .expect("initial epoch");
    trainer.observe_gameplay(1.0, &model).expect("best state");
    let old_checkpoint =
        TrainingCheckpoint::capture(&model, &trainer, &imitation_pool).expect("old checkpoint");
    let counters = trainer.counters();
    let adam = trainer.adam().clone();
    imitation_pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            2,
        ))
        .expect("dagger revision");
    assert_eq!(
        trainer
            .train_epoch(&model, &imitation_pool)
            .unwrap_err()
            .to_string(),
        "imitation trainer pool lineage, revision, scope, or seeds do not match"
    );
    trainer
        .rebind_pool(&imitation_pool)
        .expect("same lineage rebind");
    assert_eq!(trainer.counters(), counters);
    assert_eq!(trainer.adam(), &adam);
    let before_old_restore =
        TrainingCheckpoint::capture(&model, &trainer, &imitation_pool).expect("before old restore");
    assert!(
        old_checkpoint
            .restore(&model, &mut trainer, &imitation_pool)
            .is_err()
    );
    assert_eq!(
        TrainingCheckpoint::capture(&model, &trainer, &imitation_pool).expect("after old restore"),
        before_old_restore
    );
    let report = trainer
        .train_epoch(&model, &imitation_pool)
        .expect("rebound epoch");
    assert_eq!(report.order.len(), 2);
    trainer.restore_best(&model).expect("best after rebind");
    assert_eq!(trainer.counters(), counters);
    TrainingCheckpoint::capture(&model, &trainer, &imitation_pool)
        .expect("best state keeps current pool binding");

    let mut duplicate_lineage = pool(2, 116);
    duplicate_lineage
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            3,
        ))
        .expect("duplicate-lineage sample");
    let before_rejected_rebind = TrainingCheckpoint::capture(&model, &trainer, &imitation_pool)
        .expect("before rejected rebind");
    assert!(trainer.rebind_pool(&duplicate_lineage).is_err());
    let other = pool(2, 117);
    assert!(trainer.rebind_pool(&other).is_err());
    assert_eq!(
        TrainingCheckpoint::capture(&model, &trainer, &imitation_pool)
            .expect("after rejected rebind"),
        before_rejected_rebind
    );
}

#[test]
fn second_trainer_owner_is_rejected_and_raw_import_invalidates_first_owner() {
    let mut imitation_pool = pool(1, 118);
    imitation_pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            1,
        ))
        .expect("sample");
    let model = PolicyModel::fresh(18).expect("model");
    let mut first = make_trainer(1, 1, &model, 2, &imitation_pool);
    let identity = model.policy_identity().expect("identity");
    let parameters_before_second = model.export_parameters().expect("before second owner");
    assert!(
        BehavioralTrainer::new(
            1,
            2,
            AdamConfig::default(),
            EarlyStoppingConfig {
                minimum_improvement: 0.0,
                patience: 2,
            },
            &model,
            &imitation_pool,
        )
        .is_err()
    );
    assert_eq!(
        model.policy_identity().expect("after second owner"),
        identity
    );
    assert_eq!(
        model.export_parameters().expect("after second owner"),
        parameters_before_second
    );
    let parameters = model.export_parameters().expect("parameters");
    model.import_parameters(&parameters).expect("raw import");
    let before = model.export_parameters().expect("before stale trainer");
    let adam_before = first.adam().clone();
    let counters_before = first.counters();
    assert!(first.train_epoch(&model, &imitation_pool).is_err());
    assert_eq!(
        model.export_parameters().expect("after stale trainer"),
        before
    );
    assert_eq!(first.adam(), &adam_before);
    assert_eq!(first.counters(), counters_before);
    make_trainer(1, 3, &model, 2, &imitation_pool);
}

#[test]
fn paired_gameplay_aggregates_only_structural_per_seed_results() {
    let model = PolicyModel::fresh(19).expect("model");
    let identity = model.policy_identity().expect("identity");
    let first = PairedSeedResult::new(
        100,
        LearnerTeacherResult::new(LearnerMatchOutcome::Win, 3.0, 2.0).expect("radiant"),
        LearnerTeacherResult::new(LearnerMatchOutcome::Loss, 1.0, 2.0).expect("dire"),
    );
    let second = PairedSeedResult::new(
        101,
        LearnerTeacherResult::new(LearnerMatchOutcome::Draw, 2.0, 2.0).expect("radiant"),
        LearnerTeacherResult::new(LearnerMatchOutcome::Win, 4.0, 3.0).expect("dire"),
    );
    let report = PairedGameplayReport::new(identity, vec![first, second]).expect("report");
    assert_eq!(report.paired_seeds(), &[100, 101]);
    assert_eq!(report.radiant().games(), 2);
    assert_eq!(report.radiant().wins(), 1);
    assert_eq!(report.dire().wins(), 1);
    assert!(LearnerTeacherResult::new(LearnerMatchOutcome::Win, f64::NAN, 0.0).is_err());
    assert!(PairedGameplayReport::new(identity, Vec::new()).is_err());
    assert!(PairedGameplayReport::new(identity, vec![first, first]).is_err());
}

#[test]
fn seed_namespaces_are_sorted_bounded_and_pairwise_disjoint() {
    let namespaces = SeedNamespaces::new(vec![3, 1], vec![4, 2], vec![6, 5]).expect("seeds");
    assert_eq!(namespaces.training(), &[1, 3]);
    assert_eq!(namespaces.validation(), &[2, 4]);
    assert_eq!(namespaces.promotion(), &[5, 6]);
    assert!(SeedNamespaces::new(vec![1], vec![1], vec![]).is_err());
    assert!(SeedNamespaces::new(vec![0; MAX_SEED_NAMESPACE + 1], vec![], vec![]).is_err());
}

#[test]
fn teacher_coverage_keeps_failed_attempt_in_denominator() {
    let mut coverage = TeacherCoverage::new();
    let failed = coverage.collect::<(), _>(|| {
        Err(crate::ImitationError::ActionNotAllowed {
            role: "teacher",
            kind: ActionKind::Continue,
        })
    });
    assert!(failed.is_err());
    coverage.record_represented().expect("represented");
    assert_eq!(coverage.attempted(), 2);
    assert_eq!(coverage.represented(), 1);
    assert_eq!(coverage.failed(), 1);
    assert_eq!(coverage.ratio(), Some(0.5));
}

#[test]
fn identity_bound_teacher_collector_records_target_failure_without_admitting_it() {
    let (frame, _) = complete_fixture_for(ImitationSide::Radiant);
    let failed_identity =
        SampleIdentity::from_frame(SeedNamespace::Promotion, 3, 7, 10, &frame).expect("identity");
    let mut coverage = TeacherCoverage::new();
    let result = coverage.collect_for_test(failed_identity, || {
        Err::<(), _>(crate::ImitationError::ActionNotAllowed {
            role: "teacher",
            kind: ActionKind::Cast,
        })
    });
    assert!(result.is_err());
    assert_eq!(coverage.attempted(), 1);
    assert_eq!(coverage.represented(), 0);
    assert_eq!(coverage.failed(), 1);
    assert!(
        coverage
            .collect_for_test(failed_identity, || Err::<(), _>(()))
            .is_err()
    );
    assert_eq!(coverage.attempted(), 1);
    assert_eq!(coverage.failed(), 1);
}

#[test]
fn concrete_teacher_collector_counts_decision_and_target_failures() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Promotion, 3, 12, 10, &frame).expect("identity");
    let mut coverage = TeacherCoverage::new();
    let mut teacher = Teacher::new();
    let persistence = OrderPersistence::default();
    let readiness = ItemReadiness::new();
    coverage
        .collect_teacher_sample(
            identity,
            frame,
            &mut teacher,
            &tracker,
            &persistence,
            &readiness,
        )
        .expect("teacher sample");
    assert_eq!(coverage.attempted(), 1);
    assert_eq!(coverage.represented(), 1);

    let (mismatched_frame, _) = complete_fixture_for(ImitationSide::Dire);
    let mismatched_identity =
        SampleIdentity::from_frame(SeedNamespace::Promotion, 3, 13, 10, &mismatched_frame)
            .expect("mismatched identity");
    assert!(
        coverage
            .collect_teacher_sample(
                mismatched_identity,
                mismatched_frame,
                &mut teacher,
                &tracker,
                &persistence,
                &readiness,
            )
            .is_err()
    );
    assert_eq!(coverage.attempted(), 2);
    assert_eq!(coverage.represented(), 1);
    assert_eq!(coverage.failed(), 1);
}

#[test]
fn teacher_coverage_cannot_be_attached_to_a_different_sample_with_same_metadata() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Promotion, 3, 14, 10, &frame).expect("identity");
    let mut coverage = TeacherCoverage::new();
    let represented = coverage
        .collect_teacher_sample(
            identity,
            frame,
            &mut Teacher::new(),
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("represented sample");
    let (alternate_frame, alternate_space) = complete_fixture_for(ImitationSide::Radiant);
    let alternate = ImitationSample::teacher(
        alternate_frame,
        &alternate_space,
        StructuredAction::Continue,
        identity,
    )
    .expect("alternate sample");
    let mut alternate_pool = pool(1, 120);
    alternate_pool.push(alternate).expect("alternate pool");
    assert!(
        OfflineEvaluation::evaluate_held_out(&zero_model(22), &alternate_pool, coverage.clone(),)
            .is_err()
    );
    let mut represented_pool = pool(1, 121);
    represented_pool
        .push(represented)
        .expect("represented pool");
    OfflineEvaluation::evaluate_held_out(&zero_model(23), &represented_pool, coverage)
        .expect("exact represented sample");
}

#[test]
fn robust_promotion_gate_requires_real_counts_both_sides_and_audits() {
    let model = zero_model(11);
    let input = passing_gate(&model);
    let identity = model.policy_identity().expect("identity");
    let result = input.evaluate(&model).expect("gate");
    assert!(result.passed);
    assert_eq!(result.radiant_win_rate, 0.5);
    assert_eq!(result.dire_win_rate, 0.5);
    let too_small = PromotionGateInput {
        rollout: RolloutAudit::new(identity, 0, 999, true, true).expect("small rollout"),
        ..input.clone()
    };
    assert!(too_small.evaluate(&model).is_err());
    let exact_rejection = PromotionGateInput {
        rollout: RolloutAudit::new(identity, 1, 1_000, true, true).expect("exact rollout"),
        ..input.clone()
    };
    assert!(!exact_rejection.evaluate(&model).expect("boundary").passed);
    let failed_audit = PromotionGateInput {
        rollout: RolloutAudit::new(identity, 0, 1_000, true, false).expect("failed audit"),
        ..input
    };
    assert!(!failed_audit.evaluate(&model).expect("audit").passed);
    assert!(
        !gate_with_mismatches(&model, 0)
            .evaluate(&model)
            .expect("trivial corpus")
            .passed
    );
    let one_side = PromotionGateInput {
        held_out: one_side_held_out(&model),
        ..passing_gate(&model)
    };
    assert!(!one_side.evaluate(&model).expect("one side").passed);
}

#[test]
fn promotion_agreement_and_real_coverage_boundaries_are_exact() {
    let model = zero_model(12);
    assert!(
        gate_with_mismatches(&model, 5)
            .evaluate(&model)
            .expect("95 percent")
            .passed
    );
    assert!(
        !gate_with_mismatches(&model, 6)
            .evaluate(&model)
            .expect("94 percent")
            .passed
    );
    let mut input = passing_gate(&model);
    input.held_out = held_out_with_failed_coverage(&model);
    assert!(!input.evaluate(&model).expect("failed coverage").passed);
    let identity = model.policy_identity().expect("identity");
    let below_rejection = PromotionGateInput {
        rollout: RolloutAudit::new(identity, 1, 1_001, true, true).expect("rollout"),
        held_out: passing_gate(&model).held_out,
        ..input
    };
    assert!(
        below_rejection
            .evaluate(&model)
            .expect("below rejection threshold")
            .passed
    );
}

#[test]
fn paired_gameplay_rejects_missing_promotion_seed_and_zero_score() {
    let model = zero_model(13);
    let identity = model.policy_identity().expect("identity");
    let mut missing = passing_gate(&model);
    missing.gameplay = PairedGameplayReport::new(identity, paired_results(100..109, 0.2, 0.2))
        .expect("missing seed report");
    assert_eq!(
        missing.evaluate(&model).unwrap_err().to_string(),
        "imitation paired gameplay promotion seed namespace is invalid"
    );
    let mut zero_score = passing_gate(&model);
    zero_score.gameplay = PairedGameplayReport::new(identity, paired_results(100..110, 0.0, 0.0))
        .expect("zero score report");
    assert!(!zero_score.evaluate(&model).expect("zero score").passed);
}

#[test]
fn promotion_rejects_different_and_stale_policy_evidence() {
    let model = zero_model(20);
    let other = zero_model(21);
    let input = passing_gate(&model);
    assert_eq!(
        input.evaluate(&other).unwrap_err().to_string(),
        "imitation promotion evidence policy identity does not match candidate"
    );
    let mut mixed = input.clone();
    mixed.rollout = RolloutAudit::new(
        other.policy_identity().expect("other identity"),
        0,
        1_000,
        true,
        true,
    )
    .expect("mixed rollout");
    assert!(mixed.evaluate(&model).is_err());
    let mut mixed_gameplay = input.clone();
    mixed_gameplay.gameplay = PairedGameplayReport::new(
        other.policy_identity().expect("other identity"),
        paired_results(100..110, 0.2, 0.2),
    )
    .expect("mixed gameplay");
    assert!(mixed_gameplay.evaluate(&model).is_err());

    let mut training_pool = pool(1, 119);
    training_pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            1,
        ))
        .expect("training sample");
    let mut trainer = make_trainer(1, 4, &model, 2, &training_pool);
    trainer
        .train_epoch(&model, &training_pool)
        .expect("model update");
    assert!(input.evaluate(&model).is_err());
    let refreshed = passing_gate(&model);
    let parameters = model.export_parameters().expect("parameters");
    model.import_parameters(&parameters).expect("raw import");
    assert_eq!(
        refreshed.evaluate(&model).unwrap_err().to_string(),
        "imitation promotion evidence policy identity does not match candidate"
    );
}

#[test]
fn later_batch_failure_rolls_back_model_adam_counters_shuffle_and_early_state() {
    let mut pool = pool(3, 103);
    for ordinal in 0..3 {
        pool.push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            ordinal,
        ))
        .expect("sample");
    }
    let model = PolicyModel::fresh(4).expect("model");
    let mut trainer = make_trainer(2, 9, &model, 2, &pool);
    let before = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("before");
    assert!(
        trainer
            .train_epoch_with_failure(&model, &pool, 1, false)
            .is_err()
    );
    let after = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("after");
    assert_checkpoint_state_eq_after_reinstall(&after, &before);
}

#[test]
fn rollback_failure_returns_combined_exact_error() {
    let mut pool = pool(2, 104);
    for ordinal in 0..2 {
        pool.push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            ordinal,
        ))
        .expect("sample");
    }
    let model = PolicyModel::fresh(5).expect("model");
    let mut trainer = make_trainer(1, 10, &model, 2, &pool);
    let error = trainer
        .train_epoch_with_failure(&model, &pool, 1, true)
        .unwrap_err()
        .to_string();
    assert!(error.contains("imitation epoch failed"));
    assert!(error.contains("rollback failed"));
}

#[test]
fn shuffle_overflow_and_pool_revision_mismatch_are_failure_atomic() {
    let mut pool = pool(3, 105);
    for ordinal in 0..3 {
        pool.push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            ordinal,
        ))
        .expect("sample");
    }
    let model = PolicyModel::fresh(6).expect("model");
    let mut trainer = make_trainer(2, 11, &model, 2, &pool);
    trainer.set_shuffle_draws_for_test(MAX_TRAINING_COUNTER);
    let before = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("before");
    assert!(trainer.train_epoch(&model, &pool).is_err());
    let after = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("after");
    assert_checkpoint_state_eq_after_reinstall(&after, &before);
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Dire,
        ActionKind::Continue,
        99,
    ))
    .expect("revision");
    assert_eq!(
        trainer.train_epoch(&model, &pool).unwrap_err().to_string(),
        "imitation trainer pool lineage, revision, scope, or seeds do not match"
    );
}

#[test]
fn epoch_and_global_update_counter_overflow_rolls_back_exactly() {
    let mut pool = pool(1, 112);
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    ))
    .expect("sample");
    let model = PolicyModel::fresh(60).expect("model");
    let mut trainer = make_trainer(1, 1, &model, 2, &pool);
    trainer
        .set_counters_for_test(crate::TrainerCounters {
            epoch: MAX_TRAINING_COUNTER,
            global_update: MAX_TRAINING_COUNTER,
        })
        .expect("counters");
    let before = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("before");
    assert!(trainer.train_epoch(&model, &pool).is_err());
    let after = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("after");
    assert_checkpoint_state_eq_after_reinstall(&after, &before);
}

#[test]
fn early_stopping_is_monotonic_sticky_and_restores_complete_state() {
    let mut pool = pool(1, 106);
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    ))
    .expect("sample");
    let model = PolicyModel::fresh(7).expect("model");
    let mut trainer = make_trainer(1, 12, &model, 1, &pool);
    trainer.train_epoch(&model, &pool).expect("epoch one");
    trainer.observe_gameplay(2.0, &model).expect("best");
    let best_checkpoint = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("best");
    let mut invalid_best = best_checkpoint.clone();
    invalid_best
        .best_second_moment
        .as_mut()
        .expect("best moment")[0] = f32::NAN;
    assert_checkpoint_rejected_unchanged(&invalid_best, &pool);
    let mut invalid_best_owner = best_checkpoint.clone();
    invalid_best_owner.best_optimizer_lineage = Some(0);
    assert_checkpoint_rejected_unchanged(&invalid_best_owner, &pool);
    let mut future_best = best_checkpoint.clone();
    future_best.best_adam_step = Some(2);
    future_best.best_counters = Some(crate::TrainerCounters {
        epoch: 1,
        global_update: 2,
    });
    assert_checkpoint_rejected_unchanged(&future_best, &pool);
    assert!(trainer.observe_gameplay(1.0, &model).is_err());
    trainer.train_epoch(&model, &pool).expect("epoch two");
    assert!(trainer.observe_gameplay(1.0, &model).expect("stop"));
    let stopped = TrainingCheckpoint::capture(&model, &trainer, &pool).expect("stopped");
    let mut invalid_history = stopped.clone();
    invalid_history.last_evaluation_epoch = invalid_history.best_epoch;
    assert_checkpoint_rejected_unchanged(&invalid_history, &pool);
    let mut future_shuffle = best_checkpoint.clone();
    future_shuffle.best_shuffle_draws = Some(future_shuffle.shuffle_draws + 1);
    assert_checkpoint_rejected_unchanged(&future_shuffle, &pool);
    assert!(trainer.observe_gameplay(3.0, &model).expect("sticky"));
    assert_eq!(
        TrainingCheckpoint::capture(&model, &trainer, &pool).expect("same"),
        stopped
    );
    trainer.restore_best(&model).expect("restore");
    assert_eq!(trainer.counters().epoch, 1);
    assert_eq!(trainer.adam().step(), 1);
    assert_eq!(
        trainer.model_identity(),
        model.policy_identity().expect("restored identity")
    );
    assert_eq!(trainer.adam().policy_identity(), trainer.model_identity());
    let baseline_model = PolicyModel::fresh(70).expect("baseline model");
    let mut baseline = make_trainer(1, 1, &baseline_model, 1, &pool);
    best_checkpoint
        .restore(&baseline_model, &mut baseline, &pool)
        .expect("baseline restore");
    let restored_report = trainer
        .train_epoch(&model, &pool)
        .expect("restored continuation");
    let baseline_report = baseline
        .train_epoch(&baseline_model, &pool)
        .expect("baseline continuation");
    assert_eq!(restored_report, baseline_report);
    assert_adam_values_eq(trainer.adam(), baseline.adam());
    assert_eq!(
        model.export_parameters().expect("restored parameters"),
        baseline_model
            .export_parameters()
            .expect("baseline parameters")
    );
}

#[test]
fn checkpoint_resume_and_best_restore_continue_byte_exactly() {
    let mut pool = pool(2, 107);
    for ordinal in 0..2 {
        pool.push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            ordinal,
        ))
        .expect("sample");
    }
    let model = zero_model(8);
    let mut source_trainer = make_trainer(2, 13, &model, 3, &pool);
    source_trainer.train_epoch(&model, &pool).expect("warmup");
    source_trainer.observe_gameplay(1.0, &model).expect("best");
    let checkpoint =
        TrainingCheckpoint::capture(&model, &source_trainer, &pool).expect("checkpoint");
    let expected = source_trainer.train_epoch(&model, &pool).expect("expected");
    let restored_model = PolicyModel::fresh(999).expect("restored model");
    let mut restored = make_trainer(1, 1, &restored_model, 3, &pool);
    let identity_before_restore = restored_model.policy_identity().expect("before restore");
    checkpoint
        .restore(&restored_model, &mut restored, &pool)
        .expect("restore");
    let identity_after_restore = restored_model.policy_identity().expect("after restore");
    assert_eq!(
        identity_after_restore.lineage(),
        identity_before_restore.lineage()
    );
    assert!(identity_after_restore.revision() > identity_before_restore.revision());
    assert_ne!(identity_after_restore, checkpoint.model_identity);
    assert_eq!(restored.model_identity(), identity_after_restore);
    assert_eq!(restored.adam().policy_identity(), identity_after_restore);
    let actual = restored
        .train_epoch(&restored_model, &pool)
        .expect("actual");
    assert_eq!(actual, expected);
    assert_adam_values_eq(restored.adam(), source_trainer.adam());
    assert_eq!(
        restored_model
            .export_parameters()
            .expect("actual parameters"),
        model.export_parameters().expect("expected parameters")
    );
}

#[test]
fn checkpoint_matrix_rejects_schema_nan_moment_counter_and_pool_without_mutation() {
    let mut source_pool = pool(1, 108);
    source_pool
        .push(sample(
            ImitationSplit::Train,
            ImitationSide::Radiant,
            ActionKind::Continue,
            1,
        ))
        .expect("sample");
    let source = PolicyModel::fresh(9).expect("source");
    let source_trainer = make_trainer(1, 14, &source, 2, &source_pool);
    let checkpoint =
        TrainingCheckpoint::capture(&source, &source_trainer, &source_pool).expect("checkpoint");
    for mutation in 0..12 {
        let mut invalid = checkpoint.clone();
        match mutation {
            0 => invalid.action_schema_hash ^= 1,
            1 => invalid.parameters[0] = f32::NAN,
            2 => invalid.first_moment[0] = f32::NAN,
            3 => {
                invalid.first_moment.pop();
            }
            4 => invalid.global_update = 1,
            5 => invalid.second_moment[0] = f32::NAN,
            6 => invalid.last_evaluation_epoch = Some(0),
            7 => invalid.stopped = true,
            8 => invalid.model_schema_version ^= 1,
            9 => invalid.feature_schema_version ^= 1,
            10 => invalid.optimizer_version ^= 1,
            _ => invalid.optimizer_lineage = 0,
        }
        assert_checkpoint_rejected_unchanged(&invalid, &source_pool);
    }
    let other = pool(1, 109);
    assert_checkpoint_rejected_unchanged(&checkpoint, &other);
}

#[test]
fn tiny_repeated_bc_batch_reduces_loss_and_tracks_dagger_counts() {
    let mut pool = pool(2, 110);
    pool.push(sample(
        ImitationSplit::Train,
        ImitationSide::Radiant,
        ActionKind::Continue,
        1,
    ))
    .expect("sample");
    let (frame, space) = complete_fixture_for(ImitationSide::Dire);
    let identity = identity(&frame, ImitationSplit::Train, 2);
    pool.push(
        ImitationSample::dagger(
            frame,
            &space,
            StructuredAction::Stop {
                unit: crate::ControlledUnit::Hero,
            },
            StructuredAction::Continue,
            identity,
        )
        .expect("dagger"),
    )
    .expect("push");
    assert_eq!(
        pool.statistics(),
        DaggerStatistics {
            teacher: 1,
            dagger: 1,
            disagreements: 1
        }
    );
    let model = zero_model(10);
    let mut trainer = make_trainer(2, 15, &model, 3, &pool);
    let first = trainer.train_epoch(&model, &pool).expect("first");
    let second = trainer.train_epoch(&model, &pool).expect("second");
    assert!(second.average_loss < first.average_loss);
}

fn passing_gate(model: &PolicyModel) -> PromotionGateInput {
    gate_with_mismatches(model, 5)
}

fn gate_with_mismatches(model: &PolicyModel, mismatches: u64) -> PromotionGateInput {
    let mut pool = gate_pool(100, 200);
    for ordinal in 0..100 {
        let side = if ordinal % 2 == 0 {
            ImitationSide::Radiant
        } else {
            ImitationSide::Dire
        };
        let kind = if ordinal < mismatches {
            ActionKind::Stop
        } else {
            ActionKind::Continue
        };
        pool.push(gate_sample(side, kind, ordinal))
            .expect("held-out sample");
    }
    let mut coverage = TeacherCoverage::new();
    record_held_out_coverage(&mut coverage, &pool);
    let held_out = OfflineEvaluation::evaluate_held_out(model, &pool, coverage).expect("held out");
    let identity = model.policy_identity().expect("identity");
    PromotionGateInput {
        held_out,
        rollout: RolloutAudit::new(identity, 0, 1_000, true, true).expect("rollout"),
        gameplay: PairedGameplayReport::new(identity, paired_results(100..110, 0.2, 0.2))
            .expect("gameplay"),
    }
}

fn held_out_with_failed_coverage(model: &PolicyModel) -> crate::HeldOutEvaluation {
    let mut pool = gate_pool(99, 201);
    for ordinal in 0..99 {
        let side = if ordinal % 2 == 0 {
            ImitationSide::Radiant
        } else {
            ImitationSide::Dire
        };
        pool.push(gate_sample(side, ActionKind::Continue, ordinal))
            .expect("held-out sample");
    }
    let mut coverage = TeacherCoverage::new();
    record_held_out_coverage(&mut coverage, &pool);
    let failed_identity = SampleIdentity::from_frame(
        SeedNamespace::Promotion,
        100,
        999,
        10,
        pool.get(0).expect("held-out sample").frame(),
    )
    .expect("failed identity");
    coverage.record_failed_for(failed_identity).expect("failed");
    let evaluation = OfflineEvaluation::evaluate_held_out(model, &pool, coverage)
        .expect("typed held out with failed attempt");
    assert_eq!(evaluation.metrics().overall.teacher_coverage(), Some(0.99));
    evaluation
}

fn one_side_held_out(model: &PolicyModel) -> crate::HeldOutEvaluation {
    let mut pool = gate_pool(100, 202);
    for ordinal in 0..100 {
        pool.push(gate_sample(
            ImitationSide::Radiant,
            ActionKind::Continue,
            ordinal,
        ))
        .expect("held-out sample");
    }
    let mut coverage = TeacherCoverage::new();
    record_held_out_coverage(&mut coverage, &pool);
    OfflineEvaluation::evaluate_held_out(model, &pool, coverage).expect("held out")
}

fn paired_results(
    seeds: impl Iterator<Item = u64>,
    learner_score: f64,
    teacher_score: f64,
) -> Vec<PairedSeedResult> {
    seeds
        .enumerate()
        .map(|(index, seed)| {
            let outcome = match index {
                0..=4 => LearnerMatchOutcome::Win,
                5..=8 => LearnerMatchOutcome::Loss,
                _ => LearnerMatchOutcome::Draw,
            };
            let result = LearnerTeacherResult::new(outcome, learner_score, teacher_score)
                .expect("paired result");
            PairedSeedResult::new(seed, result, result)
        })
        .collect()
}

fn assert_checkpoint_rejected_unchanged(checkpoint: &TrainingCheckpoint, pool: &ImitationPool) {
    let model = PolicyModel::fresh(300).expect("model");
    let mut trainer = make_trainer(1, 1, &model, 2, pool);
    let before = TrainingCheckpoint::capture(&model, &trainer, pool).expect("before");
    assert!(checkpoint.restore(&model, &mut trainer, pool).is_err());
    assert_eq!(
        TrainingCheckpoint::capture(&model, &trainer, pool).expect("after"),
        before
    );
}

fn assert_checkpoint_state_eq_after_reinstall(
    after: &TrainingCheckpoint,
    before: &TrainingCheckpoint,
) {
    assert_eq!(
        after.model_identity.lineage(),
        before.model_identity.lineage()
    );
    assert!(after.model_identity.revision() > before.model_identity.revision());
    let mut normalized = after.clone();
    normalized.model_identity = before.model_identity;
    assert_eq!(&normalized, before);
}

fn assert_adam_values_eq(left: &crate::AdamState, right: &crate::AdamState) {
    assert_eq!(left.config(), right.config());
    assert_eq!(left.step(), right.step());
    assert_eq!(left.moments(), right.moments());
}

fn make_trainer(
    batch: usize,
    shuffle: u64,
    model: &PolicyModel,
    patience: u32,
    pool: &ImitationPool,
) -> BehavioralTrainer {
    BehavioralTrainer::new(
        batch,
        shuffle,
        AdamConfig::default(),
        EarlyStoppingConfig {
            minimum_improvement: 0.0,
            patience,
        },
        model,
        pool,
    )
    .expect("trainer")
}

fn pool(capacity: usize, lineage: u64) -> ImitationPool {
    ImitationPool::new(capacity, lineage, seeds(), scope()).expect("pool")
}

fn gate_pool(capacity: usize, lineage: u64) -> ImitationPool {
    let seeds =
        SeedNamespaces::new(vec![1], vec![2], (100..110).collect()).expect("gate seed namespaces");
    ImitationPool::new(capacity, lineage, seeds, scope()).expect("gate pool")
}

fn gate_sample(side: ImitationSide, kind: ActionKind, ordinal: u64) -> ImitationSample {
    let (frame, space) = complete_fixture_for(side);
    let action = decode_with_logits(&space, &DecoderLogits::favor(kind)).expect("action");
    let identity = SampleIdentity::from_frame(
        SeedNamespace::Promotion,
        100 + ordinal % 10,
        ordinal,
        10,
        &frame,
    )
    .expect("gate identity");
    ImitationSample::teacher(frame, &space, action, identity).expect("gate sample")
}

fn record_held_out_coverage(coverage: &mut TeacherCoverage, pool: &ImitationPool) {
    for index in 0..pool.len() {
        if let Some(sample) = pool.get(index)
            && sample.split() == ImitationSplit::HeldOut
        {
            coverage
                .record_represented_for(sample)
                .expect("held-out coverage");
        }
    }
}

fn seeds() -> SeedNamespaces {
    SeedNamespaces::new(vec![1], vec![2], vec![3]).expect("seed namespaces")
}

fn scope() -> TrainingScope {
    TrainingScope::new(MapId(0), crate::IMITATION_RULES_AUDIT_VERSION).expect("scope")
}

fn sample(
    split: ImitationSplit,
    side: ImitationSide,
    kind: ActionKind,
    ordinal: u64,
) -> ImitationSample {
    let (frame, space) = complete_fixture_for(side);
    let action = decode_with_logits(&space, &DecoderLogits::favor(kind)).expect("action");
    let identity = identity(&frame, split, ordinal);
    ImitationSample::teacher(frame, &space, action, identity).expect("sample")
}

fn identity(frame: &FeatureFrame, split: ImitationSplit, trajectory: u64) -> SampleIdentity {
    let (namespace, seed) = match split {
        ImitationSplit::Train => (SeedNamespace::Training, 1),
        ImitationSplit::Validation => (SeedNamespace::Validation, 2),
        ImitationSplit::HeldOut => (SeedNamespace::Promotion, 3),
    };
    SampleIdentity::from_frame(namespace, seed, trajectory, 10, frame).expect("identity")
}

fn zero_model(seed: u64) -> PolicyModel {
    let model = PolicyModel::fresh(seed).expect("model");
    model
        .import_parameters(&vec![0.0; MODEL_PARAMETER_COUNT])
        .expect("zero parameters");
    model
}

fn complete_fixture_for(side: ImitationSide) -> (FeatureFrame, ActionSpace) {
    let team = match side {
        ImitationSide::Radiant => Team::Radiant,
        ImitationSide::Dire => Team::Dire,
    };
    let mut view = world_view(team, 10);
    let own_slot = if team == Team::Radiant {
        SlotId(0)
    } else {
        SlotId(1)
    };
    view.units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(own_slot))
        .expect("hero")
        .abilities[0]
        .can_level = true;
    view.players
        .iter_mut()
        .find(|player| player.slot == own_slot)
        .expect("player")
        .stash = Some(vec![None; 6]);
    let tracker = tracker_with_view(team, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    (frame, space)
}

fn fixture_with_aim(aim: Aim, kind: ActionKind) -> (FeatureFrame, ActionSpace) {
    let mut view = world_view(Team::Radiant, 10);
    let hero = view
        .units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(SlotId(0)))
        .expect("hero");
    if kind == ActionKind::Cast {
        hero.abilities[0].aim = aim;
        hero.abilities[0].range = 1_200;
    } else {
        hero.items[0].as_mut().expect("item").aim = Some(aim);
        hero.items[0].as_mut().expect("item").range = 1_200;
    }
    let tracker = tracker_with_view(Team::Radiant, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    (frame, space)
}

fn expected_active_heads(action: StructuredAction) -> usize {
    match action {
        StructuredAction::Continue => 1,
        StructuredAction::Stop { .. } | StructuredAction::Hold { .. } => 2,
        StructuredAction::MovePoint { .. }
        | StructuredAction::FollowUnit { .. }
        | StructuredAction::AttackMovePoint { .. }
        | StructuredAction::AttackUnit { .. }
        | StructuredAction::Take { .. }
        | StructuredAction::Buy { .. }
        | StructuredAction::Sell { .. } => 3,
        StructuredAction::Cast { target, .. } | StructuredAction::Use { target, .. } => {
            4 + usize::from(!matches!(target, crate::ActionTarget::None))
        }
        StructuredAction::PutPoint { target, .. } => {
            4 + usize::from(matches!(target, crate::PutPointTarget::Point(_)))
        }
        StructuredAction::PutUnit { .. } | StructuredAction::Swap { .. } => 4,
        StructuredAction::Learn { .. } => 2,
    }
}
