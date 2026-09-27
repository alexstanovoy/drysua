use super::*;

const FLAGS: [u16; 2] = [StatusFlags::INVULNERABLE, StatusFlags::CHANNELLING];

#[test]
fn visible_status_facts_are_canonical_for_heroes_couriers_enemies_and_protected_structures() {
    for id in [HERO, COURIER, ENEMY, entity(33, 1)] {
        for (offset, flag) in FLAGS.into_iter().enumerate() {
            let mut canonical = None;
            for team in [Team::Radiant, Team::Dire] {
                let baseline = tracker_with_view(team, status_view(team, id, 0));
                let tracker = tracker_with_view(team, status_view(team, id, flag));
                let space = ActionSpace::from_tracker(&tracker).expect("status space");
                let index = space.entity_index(id).expect("status target").0;
                let frame = encode(&tracker, &LocalPolicyState::new(0));
                let base = encode(&baseline, &LocalPolicyState::new(0));
                let expected = if offset == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
                assert_eq!(&frame.units[index][69..71], &expected);
                assert_eq!(&base.units[index][69..71], &[0.0; 2]);
                if let Some(body) = [HERO, COURIER].iter().position(|body| *body == id) {
                    assert_eq!(&frame.own_units[body][69..71], &expected);
                }
                if let Some((units, own_units)) = &canonical {
                    assert_eq!(&frame.units, units);
                    assert_eq!(&frame.own_units, own_units);
                } else {
                    canonical = Some((frame.units, frame.own_units));
                }
            }
        }
    }
}

#[test]
fn hidden_status_claims_are_absent_but_remembered_statuses_keep_last_observed_facts() {
    let mut hidden = status_view(Team::Radiant, ENEMY, 0);
    hidden.units.retain(|unit| unit.id != ENEMY);
    let baseline = encoded_frame(Team::Radiant, hidden);
    assert!(baseline.is_finite());
    for (offset, flag) in FLAGS.into_iter().enumerate() {
        let mut view = status_view(Team::Radiant, ENEMY, flag);
        let mut tracker = tracker_with_view(Team::Radiant, view.clone());
        view.units.retain(|unit| unit.id != ENEMY);
        assert_eq!(encoded_frame(Team::Radiant, view.clone()), baseline);
        view.tick = 101;
        tracker.observe_snapshot(&view).expect("hidden snapshot");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        let row = frame
            .remembered_units
            .iter()
            .find(|row| {
                row[unit_feature::KIND_TOKEN] == 1.0 && row[unit_feature::TOKEN_PRESENT] == 1.0
            })
            .expect("remembered enemy");
        assert_eq!(
            [
                row[unit_feature::VISIBLE],
                row[unit_feature::REMEMBERED],
                row[69 + offset]
            ],
            [0.0, 1.0, 1.0]
        );
        assert!(row[unit_feature::AGE] > 0.0);
    }
}

#[test]
fn ancient_geometry_goldens_cover_both_sides_missing_ambiguous_and_absent_hero() {
    let own = [1.0, -1200.0 / 8192.0, -1100.0 / 8192.0, 1200.0 / 8192.0];
    let enemy = [1.0, 5300.0 / 8192.0, 5400.0 / 8192.0, 5400.0 / 8192.0];
    for team in [Team::Radiant, Team::Dire] {
        for (scenario, expected) in [
            ("available", [own, enemy]),
            ("missing", [[0.0; 4]; 2]),
            ("ambiguous", [[0.0; 4]; 2]),
            ("absent hero", [[0.0; 4]; 2]),
        ] {
            let mut view = world_view(team, 100);
            let locations: &[(bool, i32, i32)] = match scenario {
                "available" => &[(true, 800, 900), (false, 7300, 7400)],
                "ambiguous" => &[(false, 7000, 7100), (false, 7200, 7300)],
                "absent hero" => &[(true, 800, 900)],
                "missing" => &[],
                _ => unreachable!("bounded geometry scenarios"),
            };
            for (index, &(allied, x, y)) in locations.iter().enumerate() {
                view.units.push(building(
                    entity(70 + index as u32, 1),
                    if allied { team } else { opposing(team) },
                    UnitKind::Ancient,
                    canonical_position(team, Vec2::from_ints(x, y)),
                ));
            }
            if scenario == "absent hero" {
                remove_hero_body(&mut view, 0);
            }
            let frame = encoded_frame(team, view);
            assert_eq!(
                &frame.global[64..72],
                expected.as_flattened(),
                "{team:?} {scenario}"
            );
        }
    }
}

#[test]
fn facing_vector_goldens_are_continuous_at_the_angle_wrap_boundary() {
    for (angle, tolerance) in [(0, 0.0), (u16::MAX, 1e-4)] {
        let frame = boundary_frame(Team::Radiant, Fixed::from_int(2_000).raw, angle);
        for (actual, expected) in frame.own_units[0][71..73].iter().zip([1.0, 0.0]) {
            if angle == 0 {
                assert_eq!(*actual, expected);
            } else {
                assert!((actual - expected).abs() < tolerance, "angle={angle}");
            }
        }
    }
}

fn status_view(team: Team, id: EntityId, flag: u16) -> WorldView {
    let mut view = world_view(team, 100);
    let index = unit_index(&view, id);
    view.units[index].statuses.bits = flag;
    view
}
