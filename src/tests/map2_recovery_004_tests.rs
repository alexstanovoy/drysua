use super::*;

fn setup(side: usize) -> trip::Setup {
    trip::Setup {
        seed: 10102090 + side as u64,
        side,
        place: trip::Place::Barracks,
        threat: trip::Threat::None,
        variant: 0,
    }
}

fn native_full_state(side: usize) -> Game {
    let mut state = trip::create(setup(side));
    for _ in 0..400 {
        if full_inside(&state) {
            return state;
        }
        let (_, space) = state.prepare();
        let action = trip::reference(&state, &space);
        trip::act(&mut state, action, false, |_| {});
    }
    panic!("native reference did not refill within1200ticks");
}

#[test]
#[ignore = "Historical M17/reward1 saved-parent return-support probe; requires original schema/weights, never M18 metadata relabelling."]
fn zero_or_losing_reference_ties_abstain_but_mango_conversion_is_supported() {
    let reference = parent();
    for (_, mut physics) in combat_setups(true)
        .into_iter()
        .filter(|(name, _)| *name == "empty")
        .take(2)
    {
        physics.seed += 2000;
        let prefix = Prefix {
            physics,
            actions: vec![],
        };
        let support = reference_support(&prefix, &reference);
        assert!(!support.productive);
        assert!(
            !retain_reference(&support),
            "unproductive reference tie must not become a positive label: {support:?}"
        );
    }
    let prefix = Prefix {
        physics: Physics {
            seed: 10102498,
            ..mango_physics()
        },
        actions: vec![],
    };
    let support = reference_support(&prefix, &reference);
    assert!(support.productive);
    assert!(retain_reference(&support));
}

#[test]
fn full_resources_inside_distinct_native_endpoints_target_departure_before_exact_landing() {
    for side in 0..2 {
        let mut state = native_full_state(side);
        let (_, space) = state.prepare();
        let (_, landing) = trip::fountain(&space);
        for _ in 0..2 {
            let (_, space) = state.prepare();
            let hero = state.seats[side].tracker.own_hero().unwrap();
            assert_ne!(hero.pos, landing);
            assert!(full_inside(&state));
            let action = data_reference(&state, &space);
            assert_eq!(
                action.kind(),
                ActionKind::MovePoint,
                "full native region must have a departure target, not exact-landing Continue"
            );
            assert!(space.allows(action));
            trip::act(&mut state, StructuredAction::Continue, false, |_| {});
        }
    }
}

#[test]
fn partial_native_refill_preserves_active_recovery_continue() {
    let mut state = trip::create(setup(0));
    for _ in 0..400 {
        let (_, space) = state.prepare();
        let hero = state.seats[0].tracker.own_hero().unwrap();
        let inside = hero.pos.within(
            fountain_center(&state),
            Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
        );
        if inside && !full_inside(&state) {
            assert!(state.seats[0].local.active_order().is_some());
            assert_eq!(data_reference(&state, &space), StructuredAction::Continue);
            return;
        }
        let action = data_reference(&state, &space);
        trip::act(&mut state, action, false, |_| {});
    }
    panic!("partial native refill frame not observed");
}

#[test]
fn repeated_departure_and_close_finishing_complete_under_data_reference() {
    for side in 0..2 {
        for finish in [false, true] {
            let mut settings = setup(side);
            if finish {
                settings.place = trip::Place::Lane;
                settings.threat = trip::Threat::Finish;
            }
            let mut state = trip::create(settings);
            let start = state.arena.tick();
            let (_, space) = state.prepare();
            let (_, landing) = trip::fountain(&space);
            let mut outcome = goal::Outcome::new(&state);
            for _ in 0..600 {
                let (_, space) = state.prepare();
                let action = data_reference(&state, &space);
                assert!(space.allows(action));
                trip::act(&mut state, action, finish, |state| {
                    outcome.observe(state, start, landing)
                });
                if goal::valid(settings, &outcome) {
                    break;
                }
            }
            assert!(!outcome.exact.died);
            assert!(
                goal::valid(settings, &outcome),
                "data-reference native suffix failed: {settings:?} {outcome:?}"
            );
        }
    }
}

#[test]
fn singleton_gradient_rejects_empty_batches_without_model_changes() {
    let model = PolicyModel::fresh(10_102_490).expect("synthetic model; empty-batch validation is weight-independent");
    let identity = model.policy_identity().unwrap();
    let error = model.checked_singleton_gradient(&[]).unwrap_err();
    assert!(matches!(
        error,
        crate::ModelError::InvalidModelState("singleton gradient batch")
    ));
    assert_eq!(error.to_string(), "model produced invalid singleton gradient batch");
    assert_eq!(identity, model.policy_identity().unwrap());
}

#[test]
fn delayed_full_fountain_history_still_targets_and_completes_departure() {
    for side in 0..2 {
        let mut state = native_full_state(side);
        for _ in 0..100 {
            trip::act(&mut state, StructuredAction::Continue, false, |_| {});
        }
        let start = state.arena.tick();
        let (_, space) = state.prepare();
        let (_, landing) = trip::fountain(&space);
        let mut outcome = goal::Outcome::new(&state);
        outcome.observe(&state, start, landing);
        assert_eq!(data_reference(&state, &space).kind(), ActionKind::MovePoint);
        for _ in 0..400 {
            let (_, space) = state.prepare();
            let action = data_reference(&state, &space);
            trip::act(&mut state, action, false, |state| {
                outcome.observe(state, start, landing)
            });
            if outcome.success() {
                break;
            }
        }
        assert!(outcome.success());
    }
}

#[test]
fn validated_journeys_include_paired_refill_departure_and_both_sides_finishes() {
    let groups = data::journey_rows(&mut String::new());
    assert!(groups.iter().all(|rows| !rows.is_empty()));
    for side in 0..2 {
        let departures = &groups[side * 4 + 1];
        assert!(
            departures
                .iter()
                .all(|row| row.set.actions()[0].kind() == ActionKind::MovePoint)
        );
        assert!(
            departures
                .iter()
                .any(|row| row.identity.contains("delay_decisions100"))
        );
        assert!(
            groups[side * 4]
                .iter()
                .any(|row| row.set.actions()[0] == StructuredAction::Continue)
        );
        assert!(
            groups[side * 4 + 3]
                .iter()
                .any(|row| row.set.actions()[0].kind() == ActionKind::AttackUnit)
        );
    }
}
