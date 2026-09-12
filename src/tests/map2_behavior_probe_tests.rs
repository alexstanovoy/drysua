use super::*;

#[test]
fn fixtures_are_native_map2_complete_deterministic_seat_streams() {
    for family in FAMILIES {
        for side in 0..2 {
            let mut source = fixture(family, side);
            let mut target = fixture(family, side);
            assert_eq!(source.seats[side].tracker.metadata().map, MapId(2));
            assert_eq!(source.view(), target.view());
            assert!(source.view().tick > 0);
            for _ in 0..3 {
                source.advance(None);
                target.advance(None);
                assert_eq!(source.view(), target.view());
            }
            assert_eq!(source.metrics.ticks, 3);
        }
    }
}

#[test]
fn seat_features_expose_effect15_and_mango42_but_not_exogenous_goals() {
    for side in 0..2 {
        let mut chain = fixture(Family::RazeChain, side);
        let (frame, space) =
            prepare_neural_seat_policy_sample(&mut chain.seats[side]).expect("seat frame");
        let row = space
            .entity_candidates()
            .iter()
            .position(|unit| {
                unit.kind == UnitKind::Hero && unit.relation == crate::EntityRelation::Enemy
            })
            .expect("visible enemy row");
        assert_eq!(
            frame.units()[row][crate::unit_feature::RAZE_EFFECT_PRESENT],
            1.0
        );
        assert_eq!(
            frame.units()[row][crate::unit_feature::RAZE_STACKS],
            2.0 / 255.0
        );
        chain.goal = Vec2::from_ints(42, 42);
        let (other_goal, _) =
            prepare_neural_seat_policy_sample(&mut chain.seats[side]).expect("same seat frame");
        assert_eq!(frame, other_goal);
        let mut absent = fixture_with(Family::RazeChain, side, |world| {
            world
                .statuses
                .remove(world.seats[1 - side].unit.expect("enemy"));
        });
        let (absent, _) = prepare_neural_seat_policy_sample(&mut absent.seats[side])
            .expect("effect-free seat frame");
        assert_eq!(
            absent.units()[row][crate::unit_feature::RAZE_EFFECT_PRESENT],
            0.0
        );
        let mut mango = fixture(Family::MangoUse, side);
        let (frame, _) =
            prepare_neural_seat_policy_sample(&mut mango.seats[side]).expect("Mango seat frame");
        assert_eq!(frame.items()[0][crate::item_feature::ITEM_TOKEN], 43.0);
        assert_eq!(frame.items()[0][crate::item_feature::LEGAL], 1.0);
    }
}

#[test]
fn confirmed_raze_requires_event_and_pool_or_cooldown_evidence_not_learning() {
    for side in 0..2 {
        let mut case = fixture(Family::RazeChain, side);
        let before = case.hero().clone();
        let event = EventKind::AbilityCast {
            caster: before.id,
            ability: before.abilities[0].id,
        };
        assert!(!confirmed_raze(&before, &before, &event));
        let mut after = before.clone();
        after.abilities[0].level += 1;
        assert!(!confirmed_raze(&before, &after, &event));
        after.abilities[0].cooldown_left = 300;
        assert!(confirmed_raze(&before, &after, &event));
        assert!(!confirmed_raze(
            &before,
            &after,
            &EventKind::ItemBought {
                slot: SlotId(side as u8),
                item: ItemId(42)
            }
        ));
        let action = cast(ControlledUnit::Hero, 1);
        case.send_control(action);
        assert_eq!(case.metrics.casts, 1);
        assert!(case.metrics.magic_hero_damage > 0);
        assert_eq!(case.metrics.max_stacks, 3);
    }
}

#[test]
fn known_wire_three_reaches_hit_and_stack_but_empty_cast_has_no_damage() {
    for side in 0..2 {
        let mut chain = fixture(Family::RazeChain, side);
        for slot in 0..3 {
            chain.send_control(cast(ControlledUnit::Hero, slot));
        }
        assert_eq!(chain.metrics.casts, 3);
        assert_eq!(chain.metrics.hero_hit_casts, 3);
        assert_eq!(chain.metrics.max_stacks, 5);
        let mut empty = fixture(Family::RazeEmpty, side);
        empty.send_control(cast(ControlledUnit::Hero, 1));
        assert_eq!(empty.metrics.casts, 1);
        assert_eq!(empty.metrics.magic_hero_damage, 0);
        assert_eq!(empty.metrics.no_damage_unknown, 1);
    }
}

#[test]
fn mango_buy_fetch_and_use_are_reachable_without_policy_success_claims() {
    for side in 0..2 {
        let mut buying = fixture(Family::MangoBuy, side);
        let action = mango_buy(&buying.space()).expect("Mango42 buy represented");
        buying.send_control(action);
        assert_eq!(buying.metrics.mango_bought, 1);
        assert_eq!(buying.stash_mango(), 1);
        let mut delivery = fixture(Family::MangoDelivery, side);
        let courier = delivery.seats[side].tracker.own_courier().expect("courier");
        let slot = courier
            .abilities
            .iter()
            .position(|ability| ability.id == bota_server::game::ability::TAKE_STASH)
            .expect("native Take Stash slot");
        delivery.send_control(cast(ControlledUnit::Courier, slot as u8));
        for _ in 0..599 {
            delivery.advance(None);
        }
        assert_eq!(delivery.stash_mango(), 0);
        assert_eq!(delivery.held_mango(), 1);
        let mut using = fixture(Family::MangoUse, side);
        using.send_control(mango_use());
        assert_eq!(using.metrics.mango_used, 1);
        assert_eq!(using.metrics.manual_mana, 100);
        assert_eq!(using.held_mango(), 0);
        assert!(!using.space().allows(mango_use()));
    }
}

#[test]
fn missing_mango_and_insufficient_mana_are_not_reachable_casts() {
    for side in 0..2 {
        let case = fixture(Family::MangoBuy, side);
        assert!(!case.space().allows(mango_use()));
        assert!(!case.space().allows(cast(ControlledUnit::Hero, 0)));
        assert!(mango_buy(&case.space()).is_some());
    }
}

#[test]
fn native_learn_events_are_not_confirmed_casts_and_mana_boundary_is_exact() {
    for side in 0..2 {
        let mut learning = fixture(Family::RazeEmpty, side);
        learning.send_control(StructuredAction::Learn {
            slot: AbilitySlot(0),
        });
        assert!(learning.metrics.cast_events_without_confirmation > 0);
        assert_eq!(learning.metrics.casts, 0);
        for mana in [74, 75] {
            let mut case = fixture_with(Family::RazeEmpty, side, |world| {
                let hero = world.seats[side].unit.expect("hero");
                world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(mana);
            });
            assert_eq!(
                case.space().allows(cast(ControlledUnit::Hero, 0)),
                mana == 75
            );
            if mana == 75 {
                case.send_control(cast(ControlledUnit::Hero, 0));
                assert_eq!(case.metrics.casts, 1);
            }
        }
    }
}

#[test]
fn native_neutral_minute_spawn_adds_a_stack_only_when_box_is_clear() {
    for clear in [false, true] {
        let _case = fixture_with(Family::NeutralIdle, 0, |world| {
            let camp = world.map.camps[2].pos;
            let hero = world.seats[0].unit.expect("hero");
            world.transform.get_mut(hero).expect("position").pos = camp + Vec2::from_ints(0, 600);
            if clear {
                let beasts: Vec<_> = world
                    .entities
                    .iter()
                    .filter(|entity| {
                        world
                            .camp_home
                            .get(*entity)
                            .is_some_and(|home| home.camp == 2)
                    })
                    .collect();
                assert!(beasts.len() <= 8);
                for beast in beasts {
                    world.transform.get_mut(beast).expect("beast").pos += Vec2::from_ints(600, 0);
                }
            }
            let count = |world: &World| {
                world
                    .entities
                    .iter()
                    .filter(|entity| {
                        world
                            .camp_home
                            .get(*entity)
                            .is_some_and(|home| home.camp == 2)
                    })
                    .count()
            };
            let before = count(world);
            assert!(before > 0);
            world.tick = rules::FIRST_NEUTRAL_TICK + rules::NEUTRAL_SPAWN_PERIOD_TICKS;
            world.fill_camps();
            assert_eq!(
                count(world) > before,
                clear,
                "separate world ground truth: native minute spawn, box_clear={clear}"
            );
        });
    }
}

#[test]
fn barracks_move_control_makes_progress_but_idle_does_not() {
    for side in 0..2 {
        let mut moving = fixture(Family::RecoveryBarracks, side);
        let mut idle = fixture(Family::RecoveryBarracks, side);
        let action = goal_move(&moving.space(), moving.goal).expect("home move represented");
        moving.send_control(action);
        idle.advance(None);
        for _ in 0..299 {
            moving.advance(None);
            idle.advance(None);
        }
        assert!(
            moving.metrics.progress() > 100.0,
            "side={side} {:?}",
            moving.metrics
        );
        assert_eq!(idle.metrics.progress(), 0.0);
        assert_eq!(idle.metrics.stationary_ticks, 300);
    }
}

#[test]
fn path_metrics_distinguish_stationary_progress_and_turnback() {
    let origin = Vec2::from_ints(0, 0);
    let goal = Vec2::from_ints(100, 0);
    let mut metric = Metrics::new(origin, goal);
    metric.movement(origin, Vec2::from_ints(10, 0), goal);
    metric.movement(Vec2::from_ints(10, 0), Vec2::from_ints(5, 0), goal);
    metric.movement(Vec2::from_ints(5, 0), Vec2::from_ints(5, 0), goal);
    assert_eq!(metric.path, 15.0);
    assert_eq!(metric.progress(), 5.0);
    assert_eq!(metric.reversals, 1);
    assert_eq!(metric.away_ticks, 1);
    assert_eq!(metric.stationary_ticks, 1);
}

#[test]
fn lane_and_neutral_idle_damage_is_seat_observed_and_map2_accounted() {
    for family in [Family::LaneIdle, Family::NeutralIdle] {
        for side in 0..2 {
            let mut case = fixture(family, side);
            for _ in 0..150 {
                case.advance(None);
            }
            assert!(case.metrics.creep_damage > 0, "{family:?}/{side}");
            assert_eq!(case.metrics.creep_damage, case.metrics.reward_creep_damage);
            assert_eq!(case.metrics.unknown_incoming, 0);
        }
    }
}
