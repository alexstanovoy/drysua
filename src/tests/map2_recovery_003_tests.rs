use super::*;

#[test]
fn exact_and_near_ties_keep_reference_but_strict_gain_is_allowed() {
    let reference = use_mango();
    let wait = StructuredAction::Continue;
    for gain in [0.0, TIE / 2.0] {
        assert_eq!(
            conservative_targets(reference, &[(wait, 1.0 + gain), (reference, 1.0)]),
            (vec![reference], false)
        );
    }
    assert_eq!(
        conservative_targets(reference, &[(wait, 1.1), (reference, 1.0)]),
        (vec![wait], true)
    );
}

#[test]
fn actual_mango_after_attack_targets_parent_use_and_keeps_suffix() {
    let frozen = parent();
    let failed = start_model();
    let physics = Physics {
        seed: 10099498,
        ..mango_physics()
    };
    let mut state = game(physics);
    let (frame, space) = state.prepare();
    let attack = failed.choose(&frame, &space).unwrap().action;
    let prefix = Prefix {
        physics,
        actions: vec![attack],
    };
    let (row, reference, improved) = target_row(&prefix, &frozen);
    assert_eq!(reference, use_mango());
    assert!(!improved);
    assert_eq!(row.set.actions(), &[use_mango()]);
    let estimated = neural_estimate(&prefix, reference, &frozen);
    assert!(estimated.score > 0.9);
    assert!(estimated.metrics.mango > 0);
    assert!(estimated.metrics.damage > 0);
}

#[test]
fn all_unseparated_values_retain_reference_instead_of_a_deferral_set() {
    let action = cast(1);
    let values = [
        (StructuredAction::Continue, 0.0),
        (action, 0.0),
        (cast(2), 0.0),
    ];
    assert_eq!(conservative_targets(action, &values), (vec![action], false));
}

#[test]
fn two_native_legal_recovery_endpoints_satisfy_same_user_objective() {
    let (exact, exact_endpoint) = goal::controlled_endpoint(false);
    let (early, early_endpoint) = goal::controlled_endpoint(true);
    assert!(exact.success());
    assert!(early.success());
    assert_ne!(exact_endpoint, early_endpoint);
    assert!(exact.exact.arrival.is_some());
    assert!(early.exact.arrival.is_none());
}

#[test]
fn partial_hp_or_mana_turnback_never_counts_as_full_recovery() {
    let setup = trip::Setup {
        seed: 10100091,
        side: 0,
        place: trip::Place::Barracks,
        threat: trip::Threat::None,
        variant: 0,
    };
    let state = trip::create(setup);
    let mut hero = state.seats[0].tracker.own_hero().unwrap().clone();
    for missing_hp in [true, false] {
        let mut objective = goal::Outcome::new(&state);
        hero.pos = Vec2::from_ints(1888, 2400);
        hero.hp = hero.max_hp - i32::from(missing_hp);
        hero.mana = hero.max_mana - i32::from(!missing_hp);
        objective.observe_pools(1, &hero);
        hero.pos = Vec2::from_ints(4000, 4200);
        objective.observe_pools(2, &hero);
        assert!(objective.entered.is_some());
        assert!(objective.full.is_none());
        assert!(!objective.success());
    }
}

#[test]
fn conservative_samples_reject_stale_frames_and_nonobserved_targets() {
    let mut state = game(mango_physics());
    let (frame, _) = state.prepare();
    state.action(StructuredAction::Continue);
    let (_, space) = state.prepare();
    assert_eq!(
        CheckedActionSet::new(frame, &space, &[use_mango()]).unwrap_err(),
        crate::ModelError::Backend(
            "imitation feature frame does not belong to the supplied action space".into()
        )
    );
    let (frame, space) = state.prepare();
    let invalid = StructuredAction::AttackUnit {
        unit: ControlledUnit::Hero,
        target: EntityIndex(space.entity_candidates().len()),
    };
    assert_eq!(
        CheckedActionSet::new(frame, &space, &[invalid]).unwrap_err(),
        crate::ModelError::Backend(
            "imitation teacher action AttackUnit is not allowed by the supplied action space"
                .into()
        )
    );
}
