use super::*;
use crate::{ControlledUnit, StructuredAction};
use bota_proto::{EntityId, Fixed, StatusFlags, Vec2};

fn target_fixture(side: usize) -> (Vec<ServerMsg>, ArenaSeatPolicy) {
    let mut initial = super::super::super::training_order_contract::transcript(side);
    for message in &mut initial {
        if let ServerMsg::Snapshot { view } = message {
            let hero = view.players[side].unit.expect("own hero");
            let target = view.players[1 - side].unit.expect("target");
            for unit in &mut view.units {
                if unit.id == hero {
                    unit.attack_damage = 40;
                    unit.attack_range = Fixed::from_int(500);
                }
                if unit.id == target {
                    unit.hp = 40;
                    unit.max_hp = 100;
                    unit.attack_range = Fixed::ZERO;
                }
            }
        }
    }
    let mut seat = setup_seat(side, &initial).expect("projected fixture");
    let (_, space) = prepare_neural_observer_sample(&mut seat).expect("Observer from start");
    let target = super::super::super::training_order_contract::attack_action(&seat.tracker, &space);
    send_action(&mut seat, target);
    assert!(seat.local.active_order().is_some());
    (initial, seat)
}

fn send_action(seat: &mut ArenaSeatPolicy, action: StructuredAction) {
    let (_, space) = prepare_neural_observer_sample(seat).expect("pre-send space");
    assert!(space.allows(action));
    seat.local
        .note_decision(space.tick(), action.kind())
        .expect("actual selected kind");
    assert!(
        issue_request(
            seat,
            space.decode(action).expect("decode"),
            &space,
            action.kind(),
            true
        )
        .expect("actual send")
        .is_some()
    );
}

fn sample_summary(seat: &mut ArenaSeatPolicy) -> ([f32; 5], FeatureFrame, ActionSpace) {
    let (frame, space) = prepare_neural_observer_sample(seat).expect("complete Observer frame");
    let context = TargetContext::capture(seat);
    let values = summary(context, &space, &frame);
    assert_eq!(frame.global[59..64], [0.0; 5]);
    (values, frame, space)
}

#[test]
fn active_target_summary_copies_current_unit_facts_and_correct_range_direction() {
    for side in 0..2 {
        let (_, mut seat) = target_fixture(side);
        let (values, frame, space) = sample_summary(&mut seat);
        let ActivePolicyTarget::Unit(target) = seat.local.active_order().expect("prior").target
        else {
            panic!("unit directive")
        };
        let token = &frame.units[space.entity_index(target).expect("visible target").0];
        assert_eq!(token[unit_feature::ATTACKS_TO_KILL_PRESENT], 1.0);
        assert_eq!(token[unit_feature::OWN_IN_ATTACK_RANGE], 1.0);
        assert_eq!(token[unit_feature::UNIT_IN_ATTACK_RANGE], 0.0);
        assert_eq!(values, [1.0, 0.4, 0.0, 1.0, 0.0]);
    }
}

#[test]
fn active_target_summary_keeps_legitimate_zero_hp_distinct_from_missing_facts() {
    let (_, mut seat) = target_fixture(0);
    let (_, mut frame, space) = sample_summary(&mut seat);
    let context = TargetContext::capture(&seat);
    let ActivePolicyTarget::Unit(target) = context.prior.expect("prior").target else {
        panic!("unit")
    };
    let index = space.entity_index(target).expect("current row").0;
    frame.units[index][unit_feature::HP_RATIO] = 0.0;
    assert_eq!(summary(context, &space, &frame), [1.0, 0.0, 0.0, 1.0, 0.0]);
    frame.units[index][unit_feature::HP_PRESENT] = 0.0;
    frame.units[index][unit_feature::ATTACKS_TO_KILL_PRESENT] = 0.0;
    assert_eq!(
        summary(
            TargetContext {
                own_hero_current: false,
                ..context
            },
            &space,
            &frame
        ),
        [1.0, -1.0, -1.0, -1.0, 0.0]
    );
}

#[test]
fn active_target_summary_preserves_prior_attack_through_sent_own_cast_without_inventing_success() {
    let (initial, mut seat) = target_fixture(0);
    let prior = sample_summary(&mut seat).0;
    send_action(
        &mut seat,
        super::super::super::training_order_contract::cast_action(),
    );
    let messages = super::super::super::training_order_contract::packet(&initial, 2, |view| {
        let target = view.players[1].unit.expect("target");
        view.units
            .iter_mut()
            .find(|unit| unit.id == target)
            .expect("visible target")
            .statuses
            .bits |= StatusFlags::INVULNERABLE;
    });
    observe_messages(&mut seat, &messages).expect("current observation without cast event");
    let (current, frame, _) = sample_summary(&mut seat);
    assert_eq!(current[..4], prior[..4]);
    assert_eq!(current[4], 1.0);
    assert_eq!(
        frame.abilities[0][crate::ability_feature::LAST_CAST_PRESENT],
        0.0
    );
}

#[test]
fn active_target_summary_absent_for_nonunit_missing_and_noncurrent_encoded_rows() {
    let (_, mut seat) = target_fixture(0);
    let (_, frame, space) = sample_summary(&mut seat);
    let context = TargetContext::capture(&seat);
    let prior = context.prior.expect("prior");
    for target in [
        ActivePolicyTarget::None,
        ActivePolicyTarget::Point(Vec2::from_ints(10, 20)),
        ActivePolicyTarget::Unit(EntityId {
            idx: 999999,
            generation: 12,
        }),
    ] {
        assert_eq!(
            summary(
                TargetContext {
                    prior: Some(ActivePolicyOrder { target, ..prior }),
                    ..context
                },
                &space,
                &frame
            ),
            [0.0; 5]
        );
    }
    let ActivePolicyTarget::Unit(target) = prior.target else {
        panic!("unit")
    };
    let index = space.entity_index(target).expect("row").0;
    for (field, value) in [
        (unit_feature::TOKEN_PRESENT, 0.0),
        (unit_feature::OBSERVATION_PRESENT, 0.0),
        (unit_feature::VISIBLE, 0.0),
        (unit_feature::REMEMBERED, 1.0),
        (unit_feature::AGE, 0.1),
    ] {
        let mut invalid = frame.clone();
        invalid.units[index][field] = value;
        assert_eq!(summary(context, &space, &invalid), [0.0; 5]);
    }
}

#[test]
fn active_target_summary_clears_for_fog_point_fallback_and_stays_absent_on_reappearance() {
    let (initial, mut seat) = target_fixture(0);
    let (before, _, _) = sample_summary(&mut seat);
    let prior = TargetContext::capture(&seat);
    let target = super::super::super::training_order_contract::snapshot(&initial).players[1]
        .unit
        .expect("target");
    let fog = super::super::super::training_order_contract::packet(&initial, 2, |view| {
        view.units.retain(|unit| unit.id != target)
    });
    observe_messages(&mut seat, &fog).expect("complete invisible observation");
    let (absent, frame, space) = sample_summary(&mut seat);
    assert_eq!(before[0], 1.0);
    assert_eq!(absent, [0.0; 5]);
    assert_eq!(summary(prior, &space, &frame), [0.0; 5]);
    assert!(matches!(
        seat.local.active_order().expect("fallback").target,
        ActivePolicyTarget::Point(_)
    ));
    observe_messages(
        &mut seat,
        &super::super::super::training_order_contract::packet(&initial, 3, |_| {}),
    )
    .expect("reappearance");
    assert_eq!(sample_summary(&mut seat).0, [0.0; 5]);
}

#[test]
fn active_target_summary_never_binds_old_handle_to_a_replacement_generation() {
    let (initial, mut seat) = target_fixture(0);
    let context = TargetContext::capture(&seat);
    let ActivePolicyTarget::Unit(target) = context.prior.expect("prior").target else {
        panic!("unit")
    };
    let replacement = EntityId {
        generation: target.generation + 1,
        ..target
    };
    let packet = super::super::super::training_order_contract::packet(&initial, 2, |view| {
        view.players[1].unit = Some(replacement);
        view.units
            .iter_mut()
            .find(|unit| unit.id == target)
            .expect("target")
            .id = replacement;
        view.units.sort_by_key(|unit| unit.id);
    });
    observe_messages(&mut seat, &packet).expect("replacement generation");
    let (_, frame, space) = sample_summary(&mut seat);
    assert!(space.entity_index(replacement).is_some());
    assert!(space.entity_index(target).is_none());
    assert_eq!(summary(context, &space, &frame), [0.0; 5]);
    assert_eq!(
        summary(TargetContext::capture(&seat), &space, &frame),
        [0.0; 5]
    );
}

#[test]
fn active_target_summary_is_absent_when_current_visible_target_is_outside_cap96() {
    let (initial, mut seat) = target_fixture(0);
    let context = TargetContext::capture(&seat);
    let ActivePolicyTarget::Unit(target) = context.prior.expect("prior").target else {
        panic!("unit")
    };
    let crowded = super::super::super::training_order_contract::packet(&initial, 2, |view| {
        let own = view
            .units
            .iter()
            .find(|unit| Some(unit.id) == view.players[0].unit)
            .expect("hero")
            .pos;
        let template = view
            .units
            .iter()
            .find(|unit| unit.id == target)
            .expect("target")
            .clone();
        for index in 0..100 {
            let mut extra = template.clone();
            extra.id = EntityId {
                idx: 1000 + index,
                generation: 19,
            };
            extra.owner = None;
            extra.pos = Vec2 {
                x: own.x + Fixed::from_int(50 + index as i32),
                y: own.y,
            };
            view.units.push(extra);
        }
        view.units.sort_by_key(|unit| unit.id);
    });
    observe_messages(&mut seat, &crowded).expect("crowded current snapshot");
    let (values, _, space) = sample_summary(&mut seat);
    assert_eq!(space.entity_candidates().len(), 96);
    assert!(
        seat.tracker
            .current()
            .expect("snapshot")
            .units
            .iter()
            .any(|unit| unit.id == target)
    );
    assert!(space.entity_index(target).is_none());
    assert_eq!(values, [0.0; 5]);
}

#[test]
fn active_target_summary_uses_frozen_prior_not_the_current_selected_target() {
    let (_, mut seat) = target_fixture(0);
    let (before, frame, space) = sample_summary(&mut seat);
    let context = TargetContext::capture(&seat);
    let ActivePolicyTarget::Unit(target) = context.prior.expect("prior").target else {
        panic!("unit")
    };
    let current = space
        .entity_candidates()
        .iter()
        .enumerate()
        .find_map(|(index, candidate)| {
            let action = StructuredAction::FollowUnit {
                unit: ControlledUnit::Hero,
                target: crate::EntityIndex(index),
            };
            (candidate.id() != target && space.allows(action)).then_some(action)
        })
        .expect("different legal current target");
    send_action(&mut seat, current);
    assert_ne!(
        seat.local.active_order().expect("new directive").target,
        context.prior.expect("old directive").target
    );
    assert_eq!(summary(context, &space, &frame), before);
    assert_ne!(sample_summary(&mut seat).0, before);
}

#[test]
fn active_target_summary_is_invariant_under_consistent_opaque_handle_renaming() {
    fn renamed(id: EntityId) -> EntityId {
        assert!(id.idx < 30000);
        assert!(id.generation < 1000);
        EntityId {
            idx: 30000 - id.idx,
            generation: id.generation + 17,
        }
    }
    for side in 0..2 {
        let (mut initial, mut original) = target_fixture(side);
        let expected = sample_summary(&mut original).0;
        for message in &mut initial {
            if let ServerMsg::Snapshot { view } = message {
                assert!(view.projectiles.is_empty());
                assert!(view.loot.is_empty());
                for unit in &mut view.units {
                    unit.id = renamed(unit.id);
                }
                for player in &mut view.players {
                    player.unit = player.unit.map(renamed);
                }
                view.units.sort_by_key(|unit| unit.id);
            }
        }
        let mut remapped = setup_seat(side, &initial).expect("remapped seat");
        let (_, space) = prepare_neural_observer_sample(&mut remapped).expect("Observer");
        let action =
            super::super::super::training_order_contract::attack_action(&remapped.tracker, &space);
        send_action(&mut remapped, action);
        assert_eq!(sample_summary(&mut remapped).0, expected);
    }
}

#[test]
fn active_target_summary_annotation_preserves_identity_targets_and_equal_initial_outputs() {
    let mut rows = Vec::with_capacity(4);
    let mut spaces = Vec::with_capacity(4);
    for index in 0..4 {
        let (_, mut seat) = target_fixture(index % 2);
        let (mut values, frame, space) = sample_summary(&mut seat);
        if index >= 2 {
            values[1..4].fill(-1.0);
        }
        let identity = SampleIdentity::from_frame(
            SeedNamespace::Training,
            10089000,
            index as u64,
            space.tick(),
            &frame,
        )
        .expect("identity");
        let raw = ImitationSample::teacher(frame, &space, StructuredAction::Continue, identity)
            .expect("label");
        let mut row = ConditioningRow::new(index, raw, &space);
        row.alternate = annotate(&row.raw, &space, values);
        assert_eq!(row.raw.identity(), row.alternate.identity());
        assert_eq!(row.raw.target(), row.alternate.target());
        rows.push(row);
        spaces.push(space);
    }
    let models: [_; 2] = std::array::from_fn(|_| PolicyModel::fresh(10089200).expect("model"));
    let parameters = zero_probe_input_weights(&models[0]);
    for model in &models {
        model
            .import_parameters(&parameters)
            .expect("matched zero incoming rows");
    }
    conditioning_equivalence(
        &models[0],
        &models[1],
        &rows,
        &spaces,
        true,
        "active-target-unit-initial",
    );
}

#[test]
fn active_target_summary_replay_annotation_preserves_legacy_teacher_request_and_state() {
    for side in 0..2 {
        let (initial, observed) = target_fixture(side);
        let mut reference = setup_seat(side, &initial).expect("legacy reference");
        let space =
            ActionSpace::from_tracker_with_readiness(&reference.tracker, &reference.readiness)
                .expect("legacy space");
        let action =
            super::super::super::training_order_contract::attack_action(&reference.tracker, &space);
        reference
            .local
            .note_decision(space.tick(), action.kind())
            .expect("same prior actual attack");
        issue_request(
            &mut reference,
            space.decode(action).expect("decode"),
            &space,
            action.kind(),
            true,
        )
        .expect("legacy send");
        let mut game = history_games().remove(1 - side);
        game.environment.seats[side] = observed;
        let (_, annotation_space) =
            prepare_neural_observer_sample(&mut game.environment.seats[side])
                .expect("exact post-prefix pre-label space");
        let before = TargetContext::capture(&game.environment.seats[side]);
        let expected =
            teacher_request(&mut reference).expect("original Teacher decision and transport");
        let mut requests = [None, None];
        requests[side] = expected;
        let (samples, captured) =
            super::super::replay::replay_teacher_observation(&mut game, &requests);
        assert_eq!(captured.prior, before.prior);
        let values = summary(captured, &annotation_space, samples[0].frame());
        assert_eq!(values[0], 1.0);
        let augmented = annotate(&samples[0], &annotation_space, values);
        assert_eq!(augmented.identity(), samples[0].identity());
        assert_eq!(augmented.target(), samples[0].target());
        let observed = &game.environment.seats[side];
        assert_eq!(reference.teacher, observed.teacher);
        assert_eq!(reference.persistence, observed.persistence);
        assert_eq!(reference.readiness, observed.readiness);
        assert_eq!(reference.sequence, observed.sequence);
        assert!(!reference.order_bookkeeping.is_neural());
        assert!(observed.order_bookkeeping.is_neural());
        assert!(!observed.order_bookkeeping.is_candidate());
    }
}

#[test]
fn active_target_summary_clears_when_own_hero_lifecycle_clears_the_prior_directive() {
    let (initial, mut seat) = target_fixture(0);
    assert_eq!(sample_summary(&mut seat).0[0], 1.0);
    let hero = super::super::super::training_order_contract::snapshot(&initial).players[0]
        .unit
        .expect("hero");
    let messages = super::super::super::training_order_contract::packet(&initial, 2, |view| {
        view.players[0].unit = None;
        view.units.retain(|unit| unit.id != hero);
    });
    observe_messages(&mut seat, &messages).expect("complete hero disappearance");
    assert!(seat.local.active_order().is_none());
    assert_eq!(sample_summary(&mut seat).0, [0.0; 5]);
}

#[test]
fn active_target_summary_copies_the_existing_normalized_attack_estimate_without_new_heuristics() {
    for (hp, max_hp, expected) in [(80, 100, 1.0 / 99.0), (40000, 50000, 1.0)] {
        let (initial, mut seat) = target_fixture(0);
        let packet = super::super::super::training_order_contract::packet(&initial, 2, |view| {
            let target = view.players[1].unit.expect("target");
            let unit = view
                .units
                .iter_mut()
                .find(|unit| unit.id == target)
                .expect("current target");
            unit.hp = hp;
            unit.max_hp = max_hp;
            unit.statuses.bits |= StatusFlags::INVULNERABLE;
        });
        observe_messages(&mut seat, &packet).expect("observed HP change");
        let (values, _, _) = sample_summary(&mut seat);
        assert_eq!(values[2], expected);
        assert_eq!(
            values[4], 1.0,
            "invulnerability is separate from the crude count estimate"
        );
    }
}

#[test]
fn active_target_summary_does_not_add_a_strategy_distinction_between_attack_and_follow() {
    let (_, mut seat) = target_fixture(0);
    let (values, frame, space) = sample_summary(&mut seat);
    let context = TargetContext::capture(&seat);
    let mut prior = context.prior.expect("prior");
    prior.kind = ActionKind::FollowUnit;
    assert_eq!(
        summary(
            TargetContext {
                prior: Some(prior),
                ..context
            },
            &space,
            &frame
        ),
        values
    );
    assert_eq!(values[0], 1.0);
}
