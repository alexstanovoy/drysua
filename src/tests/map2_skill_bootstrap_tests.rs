use super::*;

#[test]
fn native_scripts_validate_outcomes_and_identifier_free_targets_on_both_sides() {
    for spec in specs(false, &[0, 2]).into_iter().chain(specs(true, &[1])) {
        let rows = collect(spec);
        assert!(!rows.is_empty());
        assert!(rows.len() <= 20);
        for row in rows {
            assert_eq!(row.spec, spec);
            assert!(row.space.allows(row.action));
            assert!(row.target.validate().is_ok());
            assert_eq!(row.target.kind.selected, row.action.kind().index());
            assert!(row.frame.matches_action_space(&row.space));
        }
    }
}

#[test]
fn splits_keep_whole_fixture_seeds_and_parameters_disjoint() {
    let train = specs(false, &[0, 2]);
    let validation = specs(true, &[1]);
    assert_eq!(validation.len(), KINDS.len() * 2);
    assert!(validation.len() >= 16);
    for held in validation {
        assert!((10092700..=10092899).contains(&held.seed));
        assert!(train.iter().all(|row| row.seed != held.seed));
        assert!(train.iter().all(|row| row.variant != held.variant));
    }
}

#[test]
fn targets_reject_illegal_casts_stale_provenance_and_fog_entity_pointers() {
    let mut environment = environment(Spec {
        kind: Kind::NoMana,
        side: 0,
        variant: 0,
        seed: 10091799,
    });
    let (frame, space) = prepare_neural_seat_policy_sample(&mut environment.seats[0]).unwrap();
    let error = BehavioralTarget::from_action(&frame, &space, cast(0)).unwrap_err();
    assert_eq!(
        error.to_string(),
        "imitation teacher action Cast is not allowed by the supplied action space"
    );
    environment.advance(None);
    let (_, next) = prepare_neural_seat_policy_sample(&mut environment.seats[0]).unwrap();
    let error =
        BehavioralTarget::from_action(&frame, &next, StructuredAction::Continue).unwrap_err();
    assert_eq!(
        error.to_string(),
        "imitation feature frame does not belong to the supplied action space"
    );
    let mut empty = environment_for(Kind::Empty, 0);
    let (frame, space) = prepare_neural_seat_policy_sample(&mut empty.seats[0]).unwrap();
    assert!(
        !space.entity_candidates().iter().any(
            |unit| unit.kind == UnitKind::Hero && unit.relation == crate::EntityRelation::Enemy
        )
    );
    let error = BehavioralTarget::from_action(
        &frame,
        &space,
        StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target: crate::EntityIndex(space.entity_candidates().len()),
        },
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "imitation teacher action AttackUnit is not allowed by the supplied action space"
    );
}

#[test]
fn masked_loss_checks_selected_class_and_inactive_head_boundaries() {
    let logits =
        candle_core::Tensor::from_vec(vec![0f32, 2.0], (1, 2), &candle_core::Device::Cpu).unwrap();
    let valid = crate::HeadTarget {
        active: true,
        mask: [true, false],
        selected: 0,
    };
    assert_eq!(
        fit::head_loss(&logits, &[valid])
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        0.0
    );
    let illegal = crate::HeadTarget {
        selected: 1,
        ..valid
    };
    assert!(fit::head_loss(&logits, &[illegal]).is_err());
}

fn environment_for(kind: Kind, side: usize) -> Environment {
    environment(Spec {
        kind,
        side,
        variant: 0,
        seed: 10091798,
    })
}
