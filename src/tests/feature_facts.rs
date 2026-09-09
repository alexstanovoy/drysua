use super::*;

const FLAGS: [u16; 2] = [StatusFlags::INVULNERABLE, StatusFlags::CHANNELLING];

fn flag_frame(team: Team, flag: u16, observed: bool) -> FeatureFrame {
    let mut view = world_view(team, 100);
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY)
        .expect("enemy")
        .statuses
        .bits |= flag;
    if !observed {
        view.units.retain(|unit| unit.id != ENEMY);
    }
    encode(&tracker_with_view(team, view), &LocalPolicyState::new(0))
}

#[test]
fn every_missing_visible_status_changes_unit_neural_input_without_shifting_old_columns() {
    let base = flag_frame(Team::Radiant, 0, true);
    for (offset, flag) in FLAGS.into_iter().enumerate() {
        let changed = flag_frame(Team::Radiant, flag, true);
        assert_ne!(base.units, changed.units, "missing visible status {flag}");
        for (before, after) in base.units.iter().zip(&changed.units) {
            assert_eq!(&before[..69], &after[..69]);
            if before != after {
                assert_eq!(after[69 + offset], 1.0);
                assert_eq!(before[69 + offset], 0.0);
            }
        }
    }
}

#[test]
fn never_observed_status_changes_do_not_enter_any_neural_input() {
    let base = flag_frame(Team::Radiant, 0, false);
    for flag in FLAGS {
        assert_eq!(base, flag_frame(Team::Radiant, flag, false));
    }
    assert!(base.is_finite());
}

#[test]
fn own_hero_and_courier_visible_flags_reach_both_current_and_fixed_body_tokens() {
    for (body, id) in [HERO, COURIER].into_iter().enumerate() {
        for (offset, flag) in FLAGS.into_iter().enumerate() {
            let mut view = world_view(Team::Radiant, 100);
            view.units
                .iter_mut()
                .find(|unit| unit.id == id)
                .expect("own body")
                .statuses
                .bits = flag;
            let frame = encode(
                &tracker_with_view(Team::Radiant, view),
                &LocalPolicyState::new(0),
            );
            assert_eq!(frame.units[body][69 + offset], 1.0);
            assert_eq!(frame.own_units[body][69 + offset], 1.0);
        }
    }
}

#[test]
fn observed_status_facts_and_facing_vector_are_team_canonical() {
    for flag in FLAGS {
        let radiant = flag_frame(Team::Radiant, flag, true);
        let dire = flag_frame(Team::Dire, flag, true);
        assert_eq!(radiant.units, dire.units);
        assert_eq!(radiant.own_units, dire.own_units);
    }
    let frame = flag_frame(Team::Radiant, 0, true);
    const {
        assert!(UNIT_FEATURES >= 73, "facing vector must be appended");
    }
    let hero = &frame.own_units[0];
    assert!((hero[71] * hero[71] + hero[72] * hero[72] - 1.0).abs() < 1e-6);
}

fn ancient_frame(team: Team, include: bool) -> FeatureFrame {
    let mut view = world_view(team, 100);
    if include {
        view.units.push(building(
            entity(70, 1),
            team,
            UnitKind::Ancient,
            canonical_position(team, Vec2::from_ints(800, 900)),
        ));
        view.units.push(building(
            entity(71, 1),
            opposing(team),
            UnitKind::Ancient,
            canonical_position(team, Vec2::from_ints(7300, 7400)),
        ));
    }
    view.units.sort_by_key(|unit| unit.id);
    encode(&tracker_with_view(team, view), &LocalPolicyState::new(0))
}

#[test]
fn available_ancient_geometry_is_appended_canonical_and_missing_is_zero() {
    const {
        assert!(GLOBAL_FEATURES >= 72, "Ancient facts must be appended");
    }
    let radiant = ancient_frame(Team::Radiant, true);
    let dire = ancient_frame(Team::Dire, true);
    let missing = ancient_frame(Team::Radiant, false);
    assert_eq!(&radiant.global[64..], &dire.global[64..]);
    assert!(missing.global[64..].iter().all(|value| *value == 0.0));
    assert_eq!(radiant.global[64], 1.0);
    assert_eq!(radiant.global[68], 1.0);
    assert!(radiant.global[65] < 0.0);
    assert!(radiant.global[69] > 0.0);
    assert_eq!(
        &radiant.global[64..68],
        &[1.0, -1200.0 / 8192.0, -1100.0 / 8192.0, 1200.0 / 8192.0]
    );
    assert_eq!(
        &radiant.global[68..72],
        &[1.0, 5300.0 / 8192.0, 5400.0 / 8192.0, 5400.0 / 8192.0]
    );
}

#[test]
fn ambiguous_ancient_locations_do_not_select_a_goal_or_leak_a_priority() {
    let mut view = world_view(Team::Radiant, 100);
    view.units.push(building(
        entity(70, 1),
        Team::Dire,
        UnitKind::Ancient,
        Vec2::from_ints(7000, 7100),
    ));
    view.units.push(building(
        entity(71, 1),
        Team::Dire,
        UnitKind::Ancient,
        Vec2::from_ints(7200, 7300),
    ));
    let frame = encode(
        &tracker_with_view(Team::Radiant, view),
        &LocalPolicyState::new(0),
    );
    assert!(frame.global[68..72].iter().all(|value| *value == 0.0));
    assert!(frame.is_finite());
}

#[test]
fn ancient_relative_geometry_is_missing_without_live_hero_even_if_courier_exists() {
    let mut view = world_view(Team::Radiant, 100);
    let hero = view
        .units
        .iter()
        .find(|unit| unit.id == HERO)
        .expect("hero")
        .clone();
    view.units.retain(|unit| unit.id != HERO);
    view.players[0].unit = None;
    view.players[0].kit = Some(bota_proto::Kit {
        abilities: hero.abilities,
        items: hero.items,
    });
    view.units.push(building(
        entity(70, 1),
        Team::Radiant,
        UnitKind::Ancient,
        Vec2::from_ints(800, 900),
    ));
    let frame = encode(
        &tracker_with_view(Team::Radiant, view),
        &LocalPolicyState::new(0),
    );
    assert_eq!(frame.own_units[1][unit_feature::TOKEN_PRESENT], 1.0);
    assert!(frame.global[64..].iter().all(|value| *value == 0.0));
}

#[test]
fn remembered_statuses_are_last_observed_facts_not_current_hidden_claims() {
    for flag in FLAGS {
        let mut view = world_view(Team::Radiant, 100);
        view.units
            .iter_mut()
            .find(|unit| unit.id == ENEMY)
            .expect("enemy")
            .statuses
            .bits = flag;
        let mut tracker = tracker_with_view(Team::Radiant, view.clone());
        view.tick = 101;
        view.units.retain(|unit| unit.id != ENEMY);
        tracker.observe_snapshot(&view).expect("hidden snapshot");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        let token = frame
            .remembered_units
            .iter()
            .find(|token| {
                token[unit_feature::KIND_TOKEN] == 1.0 && token[unit_feature::TOKEN_PRESENT] == 1.0
            })
            .expect("remembered enemy");
        assert_eq!(token[unit_feature::VISIBLE], 0.0);
        assert_eq!(token[unit_feature::REMEMBERED], 1.0);
        assert!(token[unit_feature::AGE] > 0.0);
        let column = if flag == StatusFlags::INVULNERABLE {
            69
        } else {
            70
        };
        assert_eq!(token[column], 1.0);
    }
}

#[test]
fn facing_vector_is_continuous_at_angle_wrap_and_absent_tokens_stay_zero() {
    let mut view = world_view(Team::Radiant, 100);
    view.units
        .iter_mut()
        .find(|unit| unit.id == HERO)
        .expect("hero")
        .facing
        .brads = 0;
    let zero = encode(
        &tracker_with_view(Team::Radiant, view.clone()),
        &LocalPolicyState::new(0),
    );
    view.units
        .iter_mut()
        .find(|unit| unit.id == HERO)
        .expect("hero")
        .facing
        .brads = u16::MAX;
    let wrapped = encode(
        &tracker_with_view(Team::Radiant, view),
        &LocalPolicyState::new(0),
    );
    assert_eq!(zero.own_units[0][71], 1.0);
    assert_eq!(zero.own_units[0][72], 0.0);
    assert!((zero.own_units[0][71] - wrapped.own_units[0][71]).abs() < 1e-4);
    assert!((zero.own_units[0][72] - wrapped.own_units[0][72]).abs() < 1e-4);
    for row in zero.units.iter().filter(|row| row[0] == 0.0) {
        assert!(row[69..].iter().all(|value| *value == 0.0));
    }
}

#[test]
fn protected_visible_structure_status_is_an_input_fact_not_a_teacher_or_mask_rule() {
    let view = world_view(Team::Radiant, 100);
    let base_tracker = tracker_with_view(Team::Radiant, view.clone());
    let base_space = ActionSpace::from_tracker(&base_tracker).expect("base space");
    let mut protected = view;
    protected
        .units
        .iter_mut()
        .find(|unit| unit.id == entity(33, 1))
        .expect("enemy tower")
        .statuses
        .bits |= StatusFlags::INVULNERABLE;
    let tracker = tracker_with_view(Team::Radiant, protected);
    let space = ActionSpace::from_tracker(&tracker).expect("protected space");
    let index = space
        .entity_candidates()
        .iter()
        .position(|unit| unit.id() == entity(33, 1))
        .expect("tower token");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let base = encode(&base_tracker, &LocalPolicyState::new(0));
    assert_eq!(base.units[index][69], 0.0);
    assert_eq!(frame.units[index][69], 1.0);
    assert_eq!(
        space.attack_entity_mask(crate::ControlledUnit::Hero),
        base_space.attack_entity_mask(crate::ControlledUnit::Hero)
    );
    assert_eq!(&frame.units[index][..69], &base.units[index][..69]);
    eprintln!(
        "unchanged_range_inputs attack_500={} vision_1800={}",
        frame.own_units[0][unit_feature::ATTACK_RANGE],
        frame.own_units[0][unit_feature::VISION]
    );
}
