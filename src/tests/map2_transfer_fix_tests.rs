use super::*;
use crate::model::CheckedBehavioralSample;

fn checked_opening_sample(environment: &mut Opening) -> CheckedBehavioralSample {
    let side = environment.side;
    let (frame, space) = prepare_neural_seat_policy_sample(&mut environment.seats[side]).unwrap();
    CheckedBehavioralSample::new(
        frame,
        &space,
        StructuredAction::Learn {
            slot: bota_proto::AbilitySlot(0),
        },
    )
    .unwrap()
}

#[test]
fn model_owned_fit_checks_provenance_empty_batches_and_optimizer_ownership() {
    let mut environment = opening(10091900, 0);
    let sample = checked_opening_sample(&mut environment);
    let (frame, _) = prepare_neural_seat_policy_sample(&mut environment.seats[0]).unwrap();
    environment.advance(None, None);
    let (_, space) = prepare_neural_seat_policy_sample(&mut environment.seats[0]).unwrap();
    assert_eq!(
        CheckedBehavioralSample::new(frame, &space, StructuredAction::Continue)
            .unwrap_err()
            .to_string(),
        "imitation feature frame does not belong to the supplied action space"
    );
    let model = PolicyModel::fresh_on(10092098, PolicyDevice::Cpu).unwrap();
    let other = PolicyModel::fresh_on(10092099, PolicyDevice::Cpu).unwrap();
    let mut optimizer = model.claim_optimizer(AdamConfig::default()).unwrap();
    assert_eq!(
        model
            .train_checked_behavioral_batch(&[], &mut optimizer)
            .unwrap_err(),
        crate::ModelError::BehavioralExampleCount {
            count: 0,
            maximum: 64
        }
    );
    let before = other.export_parameters().unwrap();
    assert_eq!(
        other
            .train_checked_behavioral_batch(&[&sample], &mut optimizer)
            .unwrap_err(),
        crate::ModelError::OptimizerOwnershipMismatch
    );
    assert_eq!(before, other.export_parameters().unwrap());
    let updated = model
        .train_checked_behavioral_batch(&[&sample], &mut optimizer)
        .unwrap();
    assert_eq!(updated.optimizer_step, 1);
    assert_eq!(updated.active_head_counts[0], 1);
    assert_eq!(updated.active_head_counts[5], 1);
    assert!(updated.average_loss.is_finite());
}

#[test]
fn joint_datasets_split_whole_variants_and_seeds_without_dev_or_final() {
    let training = data::new_specs(true);
    let held = data::new_specs(false);
    assert_eq!(training.len(), 48);
    assert_eq!(held.len(), 48);
    for spec in training {
        assert!((10091900..=10092099).contains(&spec.seed));
        assert!(
            held.iter()
                .all(|other| other.seed != spec.seed && other.variant != spec.variant)
        );
    }
    for spec in held {
        assert!((10092900..=10093099).contains(&spec.seed));
    }
}

#[test]
fn every_declared_skill_trajectory_has_a_validated_native_outcome() {
    for training in [true, false] {
        let rows = data::collect_skills(training);
        assert!(!rows.is_empty());
        assert!(rows.len() <= 1024);
        for row in rows {
            assert!(row.space.allows(row.action));
            assert!(row.sample.target().validate().is_ok());
        }
    }
}

#[test]
fn ordinary_teacher_anchors_include_useful_movement_and_no_idle_continue() {
    for side in 0..2 {
        let rows = data::collect_anchor(10092080, side);
        assert!(
            rows.iter()
                .all(|row| row.origin == "unchanged_teacher_native_opening")
        );
        assert!(rows.iter().any(|row| matches!(
            row.action.kind(),
            ActionKind::MovePoint | ActionKind::AttackMovePoint
        )));
        assert!(
            rows.iter()
                .any(|row| row.action.kind() == ActionKind::Continue)
        );
        assert!(rows.iter().all(|row| row.action.kind() != ActionKind::Cast));
    }
}
