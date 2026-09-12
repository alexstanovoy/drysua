use super::*;

#[test]
fn frozen_neural_prefix_suffix_matches_direct_without_reset_or_double_drain() {
    let frozen = parent();
    let identity = frozen.policy_identity().unwrap();
    let prefix = Prefix {
        physics: mango_physics(),
        actions: vec![StructuredAction::Continue],
    };
    let estimate = neural_estimate(&prefix, use_mango(), &frozen);
    let mut direct = game(prefix.physics);
    direct.action(StructuredAction::Continue);
    let before = direct.total.score;
    direct.action(use_mango());
    for _ in 1..HORIZON / 3 {
        if direct.terminal {
            break;
        }
        let (frame, space) = direct.prepare();
        direct.action(frozen.choose(&frame, &space).unwrap().action);
    }
    assert_eq!(estimate.score, direct.total.score - before);
    assert_eq!(estimate.metrics.mana, direct.total.mana);
    assert!(estimate.metrics.mango > 0);
    assert!(estimate.metrics.damage > 0);
    assert!(estimate.score > 0.9);
    assert_eq!(frozen.policy_identity().unwrap(), identity);
}

#[test]
fn skipped_loss_still_visits_restored_mana_and_neural_cast_state() {
    let frozen = parent();
    let mut prefix = Prefix {
        physics: mango_physics(),
        actions: vec![],
    };
    let (row, action) = bootstrap_visit(&mut prefix, &frozen, false);
    assert!(row.is_none());
    assert_eq!(prefix.actions.len(), 1);
    assert_eq!(action, use_mango());
    let mut restored = rank::replay(&prefix);
    let (frame, space) = restored.prepare();
    assert!(restored.seats[0].tracker.own_hero().unwrap().mana >= 100);
    assert_eq!(
        frozen.choose(&frame, &space).unwrap().action.kind(),
        ActionKind::Cast
    );
    bootstrap_visit(&mut prefix, &frozen, true);
    let converted = rank::replay(&prefix);
    assert!(converted.total.damage > 0);
    assert!(converted.total.score > 0.9);
}

#[test]
fn mango_timing_ties_are_accepted_as_equivalent_neural_returns() {
    let frozen = parent();
    let prefix = Prefix {
        physics: mango_physics(),
        actions: vec![],
    };
    let immediate = neural_estimate(&prefix, use_mango(), &frozen);
    let delayed = neural_estimate(&prefix, StructuredAction::Continue, &frozen);
    assert!((immediate.score - delayed.score).abs() <= 1e-4);
    assert!(immediate.metrics.mango > 0);
    assert!(delayed.metrics.mango > 0);
    assert!(immediate.score > 0.9);
    assert!(delayed.score > 0.9);
}

#[test]
fn recovery_reference_flips_for_visible_finish_not_for_coverage_metadata() {
    let none = trip::Setup {
        seed: 10098000,
        side: 0,
        place: trip::Place::Barracks,
        threat: trip::Threat::None,
        variant: 0,
    };
    let finish = trip::Setup {
        threat: trip::Threat::Finish,
        ..none
    };
    let mut recovering = trip::create(none);
    let (_, space) = recovering.prepare();
    assert_eq!(
        trip::reference(&recovering, &space).kind(),
        ActionKind::MovePoint
    );
    let mut finishing = trip::create(finish);
    let (_, space) = finishing.prepare();
    assert_eq!(
        recovering.seats[0].tracker.own_hero().unwrap().hp,
        finishing.seats[0].tracker.own_hero().unwrap().hp
    );
    assert_eq!(
        trip::reference(&finishing, &space).kind(),
        ActionKind::AttackUnit
    );
}

#[test]
fn no_threat_reference_reaches_restores_and_departs_both_sides_and_places() {
    for setup in trip::setups(true, &[0])
        .into_iter()
        .filter(|setup| setup.threat == trip::Threat::None)
    {
        let (trace, _) = trip::run(setup, trip::Controller::Reference);
        assert!(trace.success(), "{setup:?} {trace:?}");
        assert!(trace.arrival.unwrap() <= 900);
        assert!(trace.departure.unwrap() <= 1200);
    }
}

#[test]
fn finish_reference_keeps_attack_intent_without_repeating_orders() {
    let setup = trip::Setup {
        seed: 10098096,
        side: 0,
        place: trip::Place::Lane,
        threat: trip::Threat::Finish,
        variant: 0,
    };
    let mut state = trip::create(setup);
    let (_, space) = state.prepare();
    let attack = trip::reference(&state, &space);
    assert_eq!(attack.kind(), ActionKind::AttackUnit);
    trip::act(&mut state, attack, true, |_| {});
    let (_, space) = state.prepare();
    assert_eq!(trip::reference(&state, &space), StructuredAction::Continue);
    let (trace, metrics) = trip::run(setup, trip::Controller::Reference);
    assert!(trace.finish);
    assert!(metrics.damage > 0);
    assert!(metrics.gold > 0);
}
