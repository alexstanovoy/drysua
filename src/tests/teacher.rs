use bota_proto::{
    AbilityId, AbilitySlot, AbilityView, Aim, Angle, Attribute, Attributes, EffectId, EffectView,
    EntityId, Fixed, HeroId, ItemId, ItemSlot, ItemView, MapId, MatchInfo, Order, Pick, PlayerView,
    ShopEntry, SlotId, StatusFlags, Target, Team, TickMode, UnitKind, UnitView, Vec2, WorldView,
};

use crate::{
    ActionError, ActionTarget, ControlledUnit, IssuedOrder, ItemReadiness, OrderPersistence,
    SHADOW_FIEND, StateTracker, StructuredAction, Teacher,
};

const HERO_ID: EntityId = entity(1, 1);
const COURIER_ID: EntityId = entity(2, 1);
const ENEMY_HERO_ID: EntityId = entity(3, 1);
const CREEP_ID: EntityId = entity(4, 1);

#[test]
fn teacher_learns_requiem_before_other_legal_skills() {
    let mut view = base_view();
    own_hero_mut(&mut view).abilities[5].can_level = true;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(
        action,
        StructuredAction::Learn {
            slot: AbilitySlot(5)
        }
    );
    assert!(space.allows(action));
    assert!(space.decode(action).expect("learn decodes").is_some());
}

#[test]
fn teacher_buys_wraith_band_before_opening_tango() {
    let mut view = base_view();
    view.players[0].gold = Some(600);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(
        matches!(action, StructuredAction::Buy { unit: ControlledUnit::Hero, item } if space.shop_candidates()[item.0].item == ItemId(33))
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_buys_opening_and_treads_components_without_unearned_stick_or_duplicates() {
    let plan = [33, 7, 0, 19, 13];
    for count in 0..=plan.len() {
        let mut view = base_view();
        view.players[0].gold = Some(5_000);
        for (slot, id) in plan.iter().copied().take(count).enumerate() {
            own_hero_mut(&mut view).items[slot] = Some(item(ItemId(id), None, 0, None));
        }
        let tracker = tracker(view);

        let (action, space) = decide(&tracker);

        if let Some(expected) = plan.get(count) {
            let StructuredAction::Buy { item, .. } = action else {
                panic!("missing planned purchase after {count} items");
            };
            assert_eq!(space.shop_candidates()[item.0].item, ItemId(*expected));
        } else {
            assert!(!matches!(action, StructuredAction::Buy { .. }));
        }
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_does_not_skip_unaffordable_wraith_band_for_tango() {
    let mut view = base_view();
    view.players[0].gold = Some(504);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(!matches!(action, StructuredAction::Buy { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_rolls_back_a_rejected_purchase_by_sequence() {
    let mut view = base_view();
    view.players[0].gold = Some(600);
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let persistence = OrderPersistence::default();
    let readiness = ItemReadiness::new();
    let (first, first_space) = teacher
        .decide(&tracker, &persistence, &readiness)
        .expect("first purchase");
    let issued = first_space
        .decode(first)
        .expect("purchase decodes")
        .expect("purchase sends");
    teacher.note_sent(9, issued, first_space.tick());

    assert!(teacher.note_rejected(9));
    let (retried, space) = teacher
        .decide(&tracker, &persistence, &readiness)
        .expect("retried purchase");

    assert!(
        matches!(retried, StructuredAction::Buy { item, .. } if space.shop_candidates()[item.0].item == ItemId(33))
    );
    assert!(space.allows(retried));
}

#[test]
fn teacher_uses_wand_before_retreating_at_low_health() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.hp = 300;
    hero.items[0] = Some(item(ItemId(36), Some(Aim::Own), 0, Some(10)));
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(
        action,
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::None,
        }
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_uses_a_single_stick_charge_before_an_emergency_retreat() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.hp = 300;
    hero.items[0] = Some(item(ItemId(35), Some(Aim::Own), 0, Some(1)));

    let (action, space) = decide(&tracker(view));

    assert_eq!(
        action,
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::None,
        }
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_uses_salve_above_old_retreat_threshold_even_near_enemy_creeps() {
    let mut view = base_view();
    own_hero_mut(&mut view).hp = 700;
    own_hero_mut(&mut view).items[0] = Some(item(ItemId(2), Some(Aim::Unit), 250, Some(1)));
    view.units.push(unit(
        CREEP_ID,
        UnitKind::CreepRanged,
        Team::Dire,
        3_600,
        3_000,
    ));
    sort_units(&mut view);

    let (action, space) = decide(&tracker(view));

    assert_eq!(
        action,
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::Entity(space.entity_index(HERO_ID).expect("hero")),
        }
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_does_not_overwrite_mending_with_a_second_tango() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.hp = 600;
    hero.items[0] = Some(item(ItemId(7), Some(Aim::Tree), 165, Some(2)));
    hero.effects.push(EffectView {
        id: EffectId(1),
        ticks_left: Some(470),
        stacks: None,
    });

    let (action, space) = decide(&tracker(view));

    assert!(!matches!(action, StructuredAction::Use { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_does_not_rebuy_consumed_opening_tango_after_its_purchase_is_recorded() {
    let mut view = base_view();
    view.players[0].gold = Some(500);
    own_hero_mut(&mut view).items[0] = Some(item(ItemId(33), None, 0, None));
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    teacher.note_sent(
        1,
        IssuedOrder {
            unit: None,
            order: Order::Buy { item: ItemId(7) },
        },
        1,
    );

    let (action, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("post-consumption purchase");

    assert_eq!(
        space.decode(action).expect("decode").expect("buy").order,
        Order::Buy { item: ItemId(0) }
    );
    assert!(teacher.note_rejected(1));
    let (retry, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("rejected Tango is retryable");
    assert_eq!(
        space.decode(retry).expect("decode").expect("buy").order,
        Order::Buy { item: ItemId(7) }
    );
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_does_not_mark_wraith_bought_when_its_wire_order_only_buys_a_recipe() {
    let mut info = match_info();
    info.shop = bota_server::game::shop_entries();
    let mut view = base_view();
    view.players[0].gold = Some(210);
    own_hero_mut(&mut view).items[0] = Some(item(ItemId(9), None, 0, None));
    own_hero_mut(&mut view).items[1] = Some(item(ItemId(11), None, 0, None));
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    let mut teacher = Teacher::new();
    let persistence = OrderPersistence::default();
    let readiness = ItemReadiness::new();
    let (action, space) = teacher
        .decide(&tracker, &persistence, &readiness)
        .expect("upgrade");
    let issued = space.decode(action).expect("decode").expect("buy");
    assert_eq!(issued.order, Order::Buy { item: ItemId(39) });
    teacher.note_sent(1, issued, space.tick());

    let (retry, space) = teacher
        .decide(&tracker, &persistence, &readiness)
        .expect("unassembled root");

    assert_eq!(space.decode(retry).expect("decode"), Some(issued));
    assert!(
        !teacher.note_rejected(1),
        "recipe 39 is not a completed plan entry"
    );
    let (retry, space) = teacher
        .decide(&tracker, &persistence, &readiness)
        .expect("retry recipe");
    assert_eq!(space.decode(retry).expect("decode"), Some(issued));
}

#[test]
fn teacher_buys_stick_only_after_observing_an_enemy_cast_nearby() {
    for cast_observed in [false, true] {
        let mut view = tactical_view(500, 1_000);
        view.players[0].gold = Some(0);
        for (slot, id) in [33, 7, 0].into_iter().enumerate() {
            own_hero_mut(&mut view).items[slot] = Some(item(ItemId(id), None, 0, None));
        }
        let enemy = view
            .units
            .iter_mut()
            .find(|unit| unit.id == ENEMY_HERO_ID)
            .expect("enemy");
        enemy.abilities = shadow_fiend_abilities();
        let mut tracker = tracker(view.clone());
        let mut teacher = Teacher::new();
        let persistence = OrderPersistence::default();
        let readiness = ItemReadiness::new();
        teacher
            .decide(&tracker, &persistence, &readiness)
            .expect("prior cooldown");
        view.tick = 4;
        view.players[0].gold = Some(450);
        view.units
            .iter_mut()
            .find(|unit| unit.id == ENEMY_HERO_ID)
            .expect("enemy")
            .abilities[0]
            .cooldown_left = 290;
        tracker
            .observe_snapshot(&view)
            .expect("cooldown transition");
        if cast_observed {
            tracker
                .observe_events(
                    4,
                    &[bota_proto::EventKind::AbilityCast {
                        caster: ENEMY_HERO_ID,
                        ability: AbilityId(13),
                    }],
                )
                .expect("enemy cast");
        }

        let (action, space) = teacher
            .decide(&tracker, &persistence, &readiness)
            .expect("purchase");

        let expected = if cast_observed {
            ItemId(35)
        } else {
            ItemId(19)
        };
        assert_eq!(
            space.decode(action).expect("decode").expect("buy").order,
            Order::Buy { item: expected }
        );
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_quelling_last_hits_include_neutral_creeps_and_lane_denies() {
    for (kind, team) in [
        (UnitKind::CreepRanged, Team::Dire),
        (UnitKind::CreepRanged, Team::Radiant),
        (UnitKind::CreepNeutral, Team::Neutral),
        (UnitKind::CreepNeutral, Team::Dire),
    ] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.attack_damage = 45;
        hero.items[0] = Some(item(ItemId(5), Some(Aim::Tree), 350, None));
        let mut creep = unit(CREEP_ID, kind, team, 3_300, 3_000);
        creep.hp = 56;
        creep.max_hp = 300;
        creep.armor = Fixed::from_int(2);
        view.units.push(creep);
        sort_units(&mut view);

        let (action, space) = decide(&tracker(view));

        assert_eq!(
            space.decode(action).expect("decode").expect("attack").order,
            Order::Attack {
                target: Target::Unit(CREEP_ID)
            }
        );
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_quelling_estimate_excludes_backpack_muted_blades_and_surviving_creeps() {
    for (slot, mute, health) in [(6, 0, 50), (0, 1, 50), (0, 0, 57)] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.attack_damage = 45;
        let mut blade = item(ItemId(5), Some(Aim::Tree), 350, None);
        blade.mute_left = mute;
        hero.items[slot] = Some(blade);
        let mut creep = unit(CREEP_ID, UnitKind::CreepRanged, Team::Dire, 3_300, 3_000);
        creep.hp = health;
        creep.max_hp = 300;
        creep.armor = Fixed::from_int(2);
        view.units.push(creep);
        sort_units(&mut view);

        let (action, space) = decide(&tracker(view));

        assert!(!matches!(action, StructuredAction::AttackUnit { .. }));
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_retreats_toward_own_fountain_at_critical_health() {
    let mut view = base_view();
    own_hero_mut(&mut view).hp = 200;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    let StructuredAction::MovePoint { point, .. } = action else {
        panic!("critical health must select a retreat move");
    };
    assert!(
        space.point_candidates()[point.0]
            .position
            .distance_squared(Vec2::from_ints(1_000, 1_000))
            < Vec2::from_ints(3_000, 3_000).distance_squared(Vec2::from_ints(1_000, 1_000))
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_retreats_before_entering_raze_burst_lethal_health() {
    for hp in [350, 400, 401] {
        let mut view = base_view();
        own_hero_mut(&mut view).hp = hp;
        let tracker = tracker(view);

        let (action, space) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::MovePoint { .. }),
            hp <= 400
        );
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_finishes_fountain_recovery_instead_of_leaving_half_full() {
    for (hp, mana) in [(600, 250), (949, 500), (1_000, 474)] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.pos = Vec2::from_ints(2_200, 1_000);
        hero.hp = hp;
        hero.mana = mana;
        let tracker = tracker(view);

        let (action, space) = decide(&tracker);

        assert_eq!(
            action,
            StructuredAction::Hold {
                unit: ControlledUnit::Hero
            }
        );
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_does_not_wait_for_fountain_recovery_outside_its_radius() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.pos = Vec2::from_ints(2_201, 1_000);
    hero.hp = 600;
    hero.mana = 250;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_leaves_fountain_when_both_pools_are_recovered() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.pos = Vec2::from_ints(1_500, 1_500);
    hero.hp = 950;
    hero.mana = 475;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_safety_action_retreats_from_an_uncovered_enemy_tower() {
    let mut view = base_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_500, 6_000);
    let tracker = tracker(view);
    let space = crate::ActionSpace::from_tracker(&tracker).expect("action space");

    let action = Teacher::new()
        .safety_action(&tracker, &space)
        .expect("tower retreat");

    let StructuredAction::MovePoint { point, .. } = action else {
        panic!("uncovered tower danger must force a retreat");
    };
    assert!(
        space.point_candidates()[point.0]
            .position
            .distance_squared(Vec2::from_ints(1_000, 1_000))
            < Vec2::from_ints(5_500, 6_000).distance_squared(Vec2::from_ints(1_000, 1_000))
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_deployment_action_attacks_a_safe_in_range_enemy_structure() {
    let mut view = base_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_500, 6_000);
    view.units.push(unit(
        CREEP_ID,
        UnitKind::CreepMelee,
        Team::Radiant,
        5_700,
        6_000,
    ));
    sort_units(&mut view);
    let tracker = tracker(view);
    let space = crate::ActionSpace::from_tracker(&tracker).expect("action space");

    let action = Teacher::new()
        .deployment_action(&tracker, &space)
        .expect("safe structure attack");

    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("in-range enemy structure must be attacked");
    };
    assert_eq!(space.entity_candidates()[target.0].kind, UnitKind::Tower);
    assert!(space.allows(action));
}

#[test]
fn teacher_attacks_an_enemy_creep_killable_at_projectile_landing() {
    let mut view = base_view();
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    creep.hp = 40;
    creep.max_hp = 550;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("killable enemy creep must be attacked");
    };
    assert_eq!(space.entity_candidates()[target.0].unit().id, CREEP_ID);
    assert!(space.allows(action));
}

#[test]
fn teacher_preserves_a_seat_visible_channel_instead_of_issuing_a_body_order() {
    let mut view = base_view();
    own_hero_mut(&mut view).statuses.bits = StatusFlags::CHANNELLING;
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    creep.hp = 40;
    creep.max_hp = 550;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
}

#[test]
fn teacher_preserves_a_channel_before_low_health_and_mana_fountain_recovery() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.pos = Vec2::from_ints(1_500, 1_500);
    hero.hp = 350;
    hero.mana = 100;
    hero.statuses.bits = StatusFlags::CHANNELLING;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
    assert_eq!(
        Teacher::new().safety_action(&tracker, &space),
        Some(StructuredAction::Continue)
    );
}

#[test]
fn teacher_preserves_a_channel_before_forty_percent_retreat_outside_fountain() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.hp = 400;
    hero.statuses.bits = StatusFlags::CHANNELLING;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
    assert_eq!(
        Teacher::new().safety_action(&tracker, &space),
        Some(StructuredAction::Continue)
    );
}

#[test]
fn teacher_razes_an_enemy_tower_from_outside_its_attack_range() {
    for distance in [900, 951] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.pos = Vec2::from_ints(6_000 - distance, 6_000);
        hero.mana = 75;
        let tracker = tracker(view);

        let (action, space) = decide_stopped(&tracker);

        if distance == 900 {
            assert_eq!(action, cast(ControlledUnit::Hero, 2));
        } else {
            assert!(!matches!(action, StructuredAction::Cast { .. }));
        }
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_does_not_raze_a_tower_behind_it_or_on_cooldown() {
    for facing in [0, 32_768] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.pos = Vec2::from_ints(5_100, 6_000);
        hero.mana = 75;
        hero.facing.brads = facing;
        if facing == 0 {
            hero.abilities[2].cooldown_left = 1;
        }
        let tracker = tracker(view);

        let (action, space) = decide(&tracker);

        assert!(!matches!(action, StructuredAction::Cast { .. }));
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_ready_hero_raze_stops_a_persistent_attack_before_casting() {
    let mut view = base_view();
    own_hero_mut(&mut view).mana = 500;
    view.units.push(unit(
        ENEMY_HERO_ID,
        UnitKind::Hero,
        Team::Dire,
        3_400,
        3_000,
    ));
    view.players[1].unit = Some(ENEMY_HERO_ID);
    sort_units(&mut view);
    let mut tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID),
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("attack recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("combat decision");

    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
    assert!(space.allows(action));
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::None,
        },
        space.tick(),
    );
    let mut stopped = tracker.current().expect("snapshot").clone();
    stopped.tick += 3;
    tracker.observe_snapshot(&stopped).expect("stop applied");
    let (action, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("stable cast");
    assert_eq!(action, cast(ControlledUnit::Hero, 1));
}

#[test]
fn teacher_hero_raze_uses_last_available_mana_but_respects_cooldowns() {
    for mana in [74, 75] {
        let mut view = base_view();
        let hero = own_hero_mut(&mut view);
        hero.mana = mana;
        hero.abilities[0].cooldown_left = 30;
        view.units.push(unit(
            ENEMY_HERO_ID,
            UnitKind::Hero,
            Team::Dire,
            3_400,
            3_000,
        ));
        view.players[1].unit = Some(ENEMY_HERO_ID);
        sort_units(&mut view);
        let tracker = tracker(view);

        let (action, space) = decide_stopped(&tracker);

        if mana == 75 {
            assert_eq!(action, cast(ControlledUnit::Hero, 1));
        } else {
            assert!(matches!(action, StructuredAction::AttackUnit { .. }));
        }
        assert!(space.allows(action));
    }
}

#[test]
fn teacher_casts_only_the_raze_whose_facing_circle_contains_enemy_hero() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.mana = 500;
    hero.max_mana = 500;
    hero.abilities[0].cooldown_left = 1;
    let mut enemy = unit(ENEMY_HERO_ID, UnitKind::Hero, Team::Dire, 3_460, 3_000);
    enemy.hero = Some(SHADOW_FIEND);
    view.units.push(enemy);
    view.players[1].unit = Some(ENEMY_HERO_ID);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide_stopped(&tracker);

    assert_eq!(action, cast(ControlledUnit::Hero, 1));
    assert!(space.allows(action));
}

#[test]
fn teacher_denies_only_an_allied_creep_below_half_health() {
    let mut view = base_view();
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Radiant, 3_300, 3_000);
    creep.hp = 40;
    creep.max_hp = 100;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("killable allied creep below half health must be denied");
    };
    assert_eq!(space.entity_candidates()[target.0].unit().id, CREEP_ID);
    assert!(space.allows(action));
}

#[test]
fn teacher_casts_requiem_only_with_enough_visible_soul_stacks() {
    let mut view = base_view();
    let hero = own_hero_mut(&mut view);
    hero.mana = 500;
    hero.max_mana = 500;
    hero.effects.push(EffectView {
        id: EffectId(11),
        ticks_left: None,
        stacks: Some(12),
    });
    let mut enemy = unit(ENEMY_HERO_ID, UnitKind::Hero, Team::Dire, 2_700, 3_000);
    enemy.hero = Some(SHADOW_FIEND);
    enemy.hp = 300;
    view.units.push(enemy);
    view.players[1].unit = Some(ENEMY_HERO_ID);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, cast(ControlledUnit::Hero, 5));
    assert!(space.allows(action));
}

#[test]
fn teacher_asks_courier_to_take_stash_and_relies_on_automatic_delivery() {
    let mut view = base_view();
    view.players[0].stash.as_mut().expect("own stash")[0] =
        Some(item(ItemId(7), Some(Aim::Tree), 165, Some(3)));
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, cast(ControlledUnit::Courier, 0));
    assert!(space.allows(action));
}

#[test]
fn teacher_delivers_a_full_courier_before_collecting_more_stash_items() {
    let mut view = base_view();
    view.players[0].stash.as_mut().expect("own stash")[0] =
        Some(item(ItemId(7), Some(Aim::Tree), 165, Some(3)));
    for slot in &mut own_courier_mut(&mut view).items {
        *slot = Some(item(ItemId(1), Some(Aim::Unit), 600, Some(1)));
    }
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, cast(ControlledUnit::Courier, 3));
    assert!(space.allows(action));
}

#[test]
fn teacher_defends_a_threatened_courier_then_resumes_delivery() {
    let mut view = base_view();
    own_courier_mut(&mut view).items[0] = Some(item(ItemId(1), Some(Aim::Unit), 600, Some(1)));
    let enemy = unit(
        entity(20, 1),
        UnitKind::CreepRanged,
        Team::Dire,
        1_300,
        1_100,
    );
    view.units.push(enemy);
    sort_units(&mut view);
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();

    let (defense, defense_space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("courier defense");
    assert_eq!(defense, cast(ControlledUnit::Courier, 4));
    let issued = defense_space
        .decode(defense)
        .expect("defense decodes")
        .expect("defense sends");
    teacher.note_sent(6, issued, defense_space.tick());
    view.tick = 2;
    own_courier_mut(&mut view).abilities[4].cooldown_left = 10;
    own_courier_mut(&mut view).abilities[2].cooldown_left = 10;
    tracker
        .observe_snapshot(&view)
        .expect("defense cooldown snapshot");

    let (resumed, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("delivery resumes");

    assert_eq!(resumed, cast(ControlledUnit::Courier, 3));
    assert!(space.allows(resumed));
}

#[test]
fn teacher_starts_a_second_courier_trip_after_the_first_one_completed() {
    let mut first = base_view();
    first.tick = 2;
    own_courier_mut(&mut first).pos = Vec2::from_ints(1_000, 1_000);
    let mut tracker = tracker(first.clone());
    let mut teacher = Teacher::new();
    teacher.note_sent(
        4,
        IssuedOrder {
            unit: Some(COURIER_ID),
            order: Order::Cast {
                slot: AbilitySlot(0),
                target: Target::None,
            },
        },
        1,
    );

    teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("idle snapshot clears completed errand");
    first.tick = 3;
    first.players[0].stash.as_mut().expect("own stash")[0] =
        Some(item(ItemId(7), Some(Aim::Tree), 165, Some(3)));
    tracker
        .observe_snapshot(&first)
        .expect("new stash snapshot");

    let (action, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("second courier trip");

    assert_eq!(action, cast(ControlledUnit::Courier, 0));
    assert!(space.allows(action));
}

#[test]
fn teacher_recollects_stash_items_returned_by_a_diverted_delivery() {
    let mut view = base_view();
    view.tick = 2;
    own_courier_mut(&mut view).pos = Vec2::from_ints(1_000, 1_000);
    view.players[0].stash.as_mut().expect("own stash")[0] =
        Some(item(ItemId(7), Some(Aim::Tree), 165, Some(3)));
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    teacher.note_sent(
        5,
        IssuedOrder {
            unit: Some(COURIER_ID),
            order: Order::Cast {
                slot: AbilitySlot(3),
                target: Target::None,
            },
        },
        1,
    );

    let (action, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("returned stash trip");

    assert_eq!(action, cast(ControlledUnit::Courier, 0));
    assert!(space.allows(action));
}

#[test]
fn teacher_continues_an_attack_through_the_remaining_attack_interval() {
    let mut view = base_view();
    view.tick = 45;
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    creep.hp = 40;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(CREEP_ID),
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("attack recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("teacher decision");

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
}

#[test]
fn teacher_releases_an_old_out_of_range_attack_after_its_windup_commitment() {
    let mut view = base_view();
    view.tick = 200;
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_600, 3_000);
    creep.hp = 40;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(CREEP_ID),
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("attack recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("teacher decision");

    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_preempts_a_persistent_hold_to_last_hit_a_killable_creep() {
    let mut view = base_view();
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    creep.hp = 40;
    creep.max_hp = 550;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::None,
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("hold recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("teacher decision");

    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("killable enemy creep must preempt a persistent hold");
    };
    assert_eq!(space.entity_candidates()[target.0].unit().id, CREEP_ID);
    assert!(space.allows(action));
}

#[test]
fn teacher_releases_a_persistent_hold_to_resume_objective_progress() {
    let tracker = tracker(base_view());
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::None,
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("hold recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("teacher decision");

    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_spends_a_skill_point_without_cancelling_a_useful_move() {
    let mut view = base_view();
    own_hero_mut(&mut view).abilities[5].can_level = true;
    let tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Move {
            target: Target::Pos(Vec2::from_ints(4_000, 3_000)),
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(7, issued).expect("move recorded");
    let mut teacher = Teacher::new();
    teacher.note_sent(7, issued, 1);

    let (action, space) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("teacher decision");

    assert_eq!(
        action,
        StructuredAction::Learn {
            slot: AbilitySlot(5)
        }
    );
    assert!(space.allows(action));
    assert_eq!(persistence.active_body_order_for(None), Some(issued));
}

#[test]
fn teacher_drops_teleport_continuation_after_a_stun_interrupts_it() {
    let mut view = base_view();
    own_hero_mut(&mut view).items[0] = Some(item(ItemId(8), Some(Aim::Point), 1_200, Some(1)));
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    teacher.note_sent(
        2,
        IssuedOrder {
            unit: None,
            order: Order::Use {
                slot: ItemSlot(0),
                target: Target::Pos(Vec2::from_ints(2_000, 2_000)),
            },
        },
        1,
    );
    view.tick = 2;
    own_hero_mut(&mut view).statuses.bits = StatusFlags::STUNNED;
    tracker.observe_snapshot(&view).expect("stunned snapshot");
    teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("stun invalidates teleport note");
    view.tick = 3;
    own_hero_mut(&mut view).statuses.bits = 0;
    tracker.observe_snapshot(&view).expect("recovered snapshot");

    let (action, space) = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("post-stun decision");

    assert_ne!(action, StructuredAction::Continue);
    assert!(space.allows(action));
}

#[test]
fn teacher_attack_moves_toward_a_safe_visible_objective_without_a_wave() {
    let tracker = tracker(base_view());

    let (action, space) = decide(&tracker);

    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn teacher_follows_an_allied_wave_into_enemy_tower_range() {
    let mut view = base_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_300, 6_000);
    view.units.push(unit(
        CREEP_ID,
        UnitKind::CreepMelee,
        Team::Radiant,
        5_700,
        6_000,
    ));
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    let StructuredAction::AttackMovePoint { point, .. } = action else {
        panic!("allied tower cover must permit an objective-directed move");
    };
    let destination = space.point_candidates()[point.0].position;
    assert!(destination.within(Vec2::from_ints(6_000, 6_000), Fixed::from_int(750)));
    assert!(space.allows(action));
}

#[test]
fn teacher_falls_back_to_continue_without_a_live_hero_or_courier_work() {
    let mut view = base_view();
    view.players[0].unit = None;
    view.units.retain(|unit| unit.id != HERO_ID);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(action, StructuredAction::Continue);
    assert_eq!(space.decode(action).expect("continue decodes"), None);
}

#[test]
fn teacher_requires_a_snapshot_with_exact_action_error() {
    let tracker = StateTracker::new(SlotId(0), &match_info()).expect("empty tracker");
    let mut teacher = Teacher::new();

    let error = teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .err()
        .expect("snapshot is required");

    assert_eq!(error, ActionError::SnapshotRequired);
    assert_eq!(
        error.to_string(),
        "action space requires a validated snapshot"
    );
}

#[test]
fn tactical_default_and_zero_weights_match_teacher_actions_and_memory() {
    let policies = [
        crate::TacticalPolicy::default(),
        crate::TacticalPolicy::from_parameters(&[0.0; crate::TACTICAL_PARAMETERS])
            .expect("zero policy"),
    ];
    for policy in policies {
        for scenario in 0..9 {
            let mut view = tactical_view(500, 100);
            match scenario {
                0 => own_hero_mut(&mut view).mana = 500,
                1 => own_hero_mut(&mut view).hp = 300,
                2 => own_hero_mut(&mut view).statuses.bits = StatusFlags::CHANNELLING,
                3 => view.players[0].gold = Some(600),
                4 => own_hero_mut(&mut view).abilities[5].can_level = true,
                5 => own_hero_mut(&mut view).pos = Vec2::from_ints(1_500, 1_500),
                6 => own_hero_mut(&mut view).hp = 0,
                7 => view.units.retain(|unit| unit.id != ENEMY_HERO_ID),
                _ => {}
            }
            let tracker = tracker(view);
            let mut baseline = Teacher::new();
            let mut tactical = baseline.clone();
            let mut persistence = OrderPersistence::default();
            for sequence in 1..=3 {
                let (expected, expected_space) = baseline
                    .decide(&tracker, &persistence, &ItemReadiness::new())
                    .expect("baseline");
                let (actual, space) = tactical
                    .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
                    .expect("tactical");
                assert_eq!(actual, expected, "scenario {scenario}");
                assert_eq!(space.decode(actual), expected_space.decode(expected));
                assert_eq!(baseline, tactical);
                assert!(space.allows(actual));
                if let Some(issued) = space.decode(actual).expect("decode") {
                    baseline.note_sent(sequence, issued, space.tick());
                    tactical.note_sent(sequence, issued, space.tick());
                    persistence.record_sent(sequence, issued).expect("record");
                }
            }
        }
    }
}

#[test]
fn tactical_fight_pursues_a_burst_kill_outside_current_attack_range() {
    let tracker = tracker(tactical_view(800, 100));
    let policy = tactical_policy(crate::TacticalMode::Fight);

    let (baseline, _) = decide(&tracker);
    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &policy,
        )
        .expect("fight");

    assert_ne!(action, baseline);
    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("pursuit must target hero");
    };
    assert_eq!(space.entity_candidates()[target.0].id(), ENEMY_HERO_ID);
    assert!(space.allows(action));
}

#[test]
fn tactical_fight_does_not_pursue_nonlethal_or_distant_targets() {
    for (distance, hp) in [(800, 1_000), (1_201, 100)] {
        let tracker = tracker(tactical_view(distance, hp));

        let (baseline, _) = decide(&tracker);
        let (action, space) = Teacher::new()
            .decide_tactical(
                &tracker,
                &OrderPersistence::default(),
                &ItemReadiness::new(),
                &tactical_policy(crate::TacticalMode::Fight),
            )
            .expect("bounded fight");

        assert_eq!(action, baseline);
        assert!(space.allows(action));
    }
}

#[test]
fn tactical_neural_health_feature_changes_real_combat_decisions() {
    let mut parameters = [0.0; crate::TACTICAL_PARAMETERS];
    parameters[0] = 1.0;
    parameters[(crate::TACTICAL_FEATURES + 2) * crate::TACTICAL_HIDDEN] = 1.0;
    parameters[crate::TACTICAL_OUTPUT_BIAS_OFFSET] = 0.8;
    let policy = crate::TacticalPolicy::from_parameters(&parameters).expect("health neuron");
    for hp in [600, 1_000] {
        let mut view = tactical_view(800, 100);
        own_hero_mut(&mut view).hp = hp;
        let tracker = tracker(view);

        let (baseline, _) = decide(&tracker);
        let (action, space) = Teacher::new()
            .decide_tactical(
                &tracker,
                &OrderPersistence::default(),
                &ItemReadiness::new(),
                &policy,
            )
            .expect("conditional combat");

        assert_eq!(action == baseline, hp == 600);
        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            hp == 1_000
        );
        assert!(space.allows(action));
    }
}

#[test]
fn tactical_fight_requires_ready_mana_affordable_burst_for_pursuit() {
    for (mana, cooldown) in [(74, 0), (75, 1), (75, 0)] {
        let mut view = tactical_view(1_000, 200);
        let hero = own_hero_mut(&mut view);
        hero.mana = mana;
        for ability in &mut hero.abilities[..3] {
            ability.cooldown_left = cooldown;
        }
        let tracker = tracker(view);

        let (baseline, _) = decide(&tracker);
        let (action, space) = Teacher::new()
            .decide_tactical(
                &tracker,
                &OrderPersistence::default(),
                &ItemReadiness::new(),
                &tactical_policy(crate::TacticalMode::Fight),
            )
            .expect("burst pursuit");

        let burst_available = mana == 75 && cooldown == 0;
        assert_eq!(action != baseline, burst_available);
        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            burst_available
        );
        assert!(space.allows(action));
    }
}

#[test]
fn tactical_fight_does_not_chase_a_kill_into_enemy_tower_cover() {
    let mut view = tactical_view(800, 100);
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_100, 6_000);
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy")
        .pos = Vec2::from_ints(5_900, 6_000);
    let tracker = tracker(view);

    let (baseline, _) = decide(&tracker);
    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("tower guard");

    assert_eq!(action, baseline);
    assert!(!matches!(action, StructuredAction::AttackUnit { .. }));
    assert!(space.allows(action));
}

#[test]
fn tactical_fight_does_not_cross_tower_range_between_two_safe_endpoints() {
    let mut view = tactical_view(800, 100);
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_300, 5_700);
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy")
        .pos = Vec2::from_ints(6_400, 5_300);
    let tracker = tracker(view);

    let (baseline, _) = decide(&tracker);
    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("corridor guard");

    assert_eq!(action, baseline);
    assert!(!matches!(action, StructuredAction::AttackUnit { .. }));
    assert!(space.allows(action));
}

#[test]
fn tactical_fight_preserves_a_valuable_ready_requiem() {
    let mut view = tactical_view(-300, 300);
    let hero = own_hero_mut(&mut view);
    hero.mana = 500;
    hero.effects.push(EffectView {
        id: EffectId(11),
        ticks_left: None,
        stacks: Some(12),
    });
    let tracker = tracker(view);

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("fight ultimate");

    assert_eq!(action, cast(ControlledUnit::Hero, 5));
    assert!(space.allows(action));
}

#[test]
fn tactical_recover_disengages_above_teachers_emergency_threshold() {
    let mut view = tactical_view(500, 800);
    own_hero_mut(&mut view).hp = 600;
    let tracker = tracker(view);

    let (baseline, _) = decide(&tracker);
    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Recover),
        )
        .expect("recover");

    assert_ne!(action, baseline);
    let StructuredAction::MovePoint { point, .. } = action else {
        panic!("recovery must move");
    };
    assert!(
        space.point_candidates()[point.0]
            .position
            .distance_squared(Vec2::from_ints(1_000, 1_000))
            < Vec2::from_ints(3_000, 3_000).distance_squared(Vec2::from_ints(1_000, 1_000))
    );
    assert!(space.allows(action));
}

#[test]
fn tactical_farm_takes_last_hit_instead_of_spending_mana_on_harassment() {
    let mut view = tactical_view(400, 1_000);
    own_hero_mut(&mut view).mana = 500;
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    creep.hp = 40;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (baseline, _) = decide(&tracker);
    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Farm),
        )
        .expect("farm");

    assert_eq!(
        baseline,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
    let StructuredAction::AttackUnit { target, .. } = action else {
        panic!("last hit must attack creep");
    };
    assert_eq!(space.entity_candidates()[target.0].id(), CREEP_ID);
    assert!(space.allows(action));
}

#[test]
fn tactical_modes_preserve_channels_sustain_economy_and_emergency_retreat() {
    for mode in [
        crate::TacticalMode::Fight,
        crate::TacticalMode::Recover,
        crate::TacticalMode::Farm,
    ] {
        for scenario in 0..8 {
            let mut view = tactical_view(500, 100);
            match scenario {
                0 => own_hero_mut(&mut view).statuses.bits = StatusFlags::CHANNELLING,
                1 => own_hero_mut(&mut view).hp = 400,
                2 => view.players[0].gold = Some(600),
                3 => own_hero_mut(&mut view).abilities[5].can_level = true,
                4 => {
                    own_hero_mut(&mut view).hp = 300;
                    own_hero_mut(&mut view).items[0] =
                        Some(item(ItemId(36), Some(Aim::Own), 0, Some(10)));
                }
                5 => {
                    view.players[0].stash.as_mut().expect("stash")[0] =
                        Some(item(ItemId(7), Some(Aim::Tree), 165, Some(3)))
                }
                6 => own_hero_mut(&mut view).pos = Vec2::from_ints(5_500, 6_000),
                _ => own_hero_mut(&mut view).pos = Vec2::from_ints(1_500, 1_500),
            }
            let tracker = tracker(view);

            let (baseline, _) = decide(&tracker);
            let (action, space) = Teacher::new()
                .decide_tactical(
                    &tracker,
                    &OrderPersistence::default(),
                    &ItemReadiness::new(),
                    &tactical_policy(mode),
                )
                .expect("protected work");

            assert_eq!(action, baseline, "scenario {scenario}, mode {mode:?}");
            assert!(space.allows(action));
        }
    }
}

#[test]
fn tactical_pursuit_is_suppressed_when_its_body_order_is_already_active() {
    let tracker = tracker(tactical_view(800, 100));
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Fight);
    let (first, space) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("first");
    let issued = space.decode(first).expect("decode").expect("pursuit");
    persistence.record_sent(1, issued).expect("record");
    teacher.note_sent(1, issued, space.tick());

    let (action, space) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("repeat");

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
    assert_eq!(persistence.active_body_order_for(None), Some(issued));
}

#[test]
fn tactical_fight_preserves_an_active_creep_attack_in_windup_leeway() {
    let mut view = tactical_view(800, 100);
    view.units.push(unit(
        CREEP_ID,
        UnitKind::CreepMelee,
        Team::Dire,
        3_600,
        3_000,
    ));
    sort_units(&mut view);
    let tracker = tracker(view);
    let issued = IssuedOrder {
        unit: None,
        order: Order::Attack {
            target: Target::Unit(CREEP_ID),
        },
    };
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(1, issued).expect("record");
    let mut teacher = Teacher::new();
    teacher.note_sent(1, issued, 1);

    let (action, space) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("persistent attack");

    assert_eq!(action, StructuredAction::Continue);
    assert!(space.allows(action));
}

#[test]
fn tactical_ignores_remembered_enemy_health_when_enemy_is_no_longer_visible() {
    let mut current = base_view();
    current.tick = 2;
    let mut trackers = [
        tracker(tactical_view(800, 100)),
        tracker(tactical_view(800, 1_000)),
    ];
    for tracker in &mut trackers {
        tracker.observe_snapshot(&current).expect("hidden enemy");
    }
    let policy = tactical_policy(crate::TacticalMode::Fight);

    let (first, first_space) = Teacher::new()
        .decide_tactical(
            &trackers[0],
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &policy,
        )
        .expect("first history");
    let (second, second_space) = Teacher::new()
        .decide_tactical(
            &trackers[1],
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &policy,
        )
        .expect("other history");

    assert_eq!(first_space.decode(first), second_space.decode(second));
    assert!(first_space.entity_index(ENEMY_HERO_ID).is_none());
}

#[test]
fn tactical_pursuit_releases_body_order_when_target_leaves_visibility() {
    let mut tracker = tracker(tactical_view(800, 100));
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Fight);
    let (first, space) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("pursuit");
    let issued = space.decode(first).expect("decode").expect("body order");
    persistence.record_sent(1, issued).expect("record");
    teacher.note_sent(1, issued, space.tick());
    let mut hidden = base_view();
    hidden.tick = 2;
    tracker.observe_snapshot(&hidden).expect("visibility lost");

    let (action, space) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("release pursuit");

    assert_ne!(action, StructuredAction::Continue);
    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
    assert!(space.allows(action));
}

#[test]
fn tactical_mutated_policies_produce_only_legal_actions_under_disabling_statuses() {
    for seed in 0..8 {
        let policy = crate::TacticalPolicy::default()
            .mutated(seed, 1.0)
            .expect("mutation");
        for status in [
            0,
            StatusFlags::STUNNED,
            StatusFlags::ROOTED,
            StatusFlags::DISARMED,
            StatusFlags::SILENCED,
            StatusFlags::CHANNELLING,
        ] {
            let mut view = tactical_view(500, 200);
            let hero = own_hero_mut(&mut view);
            hero.hp = 600;
            hero.mana = 200;
            hero.statuses.bits = status;
            let tracker = tracker(view);

            let (action, space) = Teacher::new()
                .decide_tactical(
                    &tracker,
                    &OrderPersistence::default(),
                    &ItemReadiness::new(),
                    &policy,
                )
                .expect("masked decision");

            assert!(space.allows(action), "seed {seed}, status {status}");
            assert!(space.decode(action).is_ok(), "seed {seed}, status {status}");
        }
    }
}

#[test]
fn combat_v2_recover_uses_one_short_step_instead_of_a_fountain_trip() {
    let view = tactical_view(500, 1_000);
    let origin = Vec2::from_ints(3_000, 3_000);
    let tracker = tracker(view);

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Recover),
        )
        .expect("backoff");

    let StructuredAction::MovePoint { point, .. } = action else {
        panic!("short backoff")
    };
    let destination = space.point_candidates()[point.0].position;
    assert!(origin.within(destination, Fixed::from_int(210)));
    assert!(
        destination.distance_squared(Vec2::from_ints(3_500, 3_000))
            > origin.distance_squared(Vec2::from_ints(3_500, 3_000))
    );
}

#[test]
fn combat_v2_low_mana_does_not_abandon_a_lane_with_a_distant_enemy() {
    let tracker = tracker(tactical_view(1_100, 1_000));

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Recover),
        )
        .expect("safe low mana");

    assert!(!matches!(action, StructuredAction::MovePoint { .. }));
    assert!(space.allows(action));
}

#[test]
fn combat_v2_recover_stops_underlying_movement_when_health_recovers_and_enemy_disappears() {
    let mut view = tactical_view(500, 1_000);
    own_hero_mut(&mut view).hp = 600;
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Recover);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick = 4;
    own_hero_mut(&mut view).hp = 1_000;
    own_hero_mut(&mut view).mana = 500;
    view.units.retain(|unit| unit.id != ENEMY_HERO_ID);
    tracker
        .observe_snapshot(&view)
        .expect("recovered without threat");

    let (action, _) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("cancel backoff");

    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
}

#[test]
fn combat_v2_backoff_keeps_a_fixed_goal_instead_of_moving_it_each_decision() {
    let mut view = tactical_view(500, 1_000);
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Recover);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    let active = persistence.active_body_order_for(None);
    view.tick = 4;
    own_hero_mut(&mut view).pos.x -= Fixed::from_int(10);
    tracker.observe_snapshot(&view).expect("walking");

    let (action, _) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("fixed goal");

    assert_eq!(action, StructuredAction::Continue);
    assert_eq!(persistence.active_body_order_for(None), active);
}

#[test]
fn combat_v2_raze_selects_the_centered_circle_over_an_earlier_slot_edge() {
    for (distance, slot) in [(450, 1), (700, 2)] {
        let mut view = tactical_view(distance, 1_000);
        view.tick = 4;
        own_hero_mut(&mut view).mana = 500;
        let tracker = tracker(view);
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        record_combat_order(
            &mut teacher,
            &mut persistence,
            Order::Move {
                target: Target::None,
            },
            1,
        );

        let (action, _) = teacher
            .decide_tactical(
                &tracker,
                &persistence,
                &ItemReadiness::new(),
                &tactical_policy(crate::TacticalMode::Fight),
            )
            .expect("centered raze");

        assert_eq!(action, cast(ControlledUnit::Hero, slot));
    }
}

#[test]
fn combat_v2_raze_stops_a_conflicting_move_before_cast_application_can_turn_it() {
    let mut view = tactical_view(700, 1_000);
    view.tick = 4;
    own_hero_mut(&mut view).mana = 500;
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::Pos(Vec2::from_ints(2_000, 3_000)),
        },
        1,
    );

    let (action, _) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("stabilize facing");

    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
}

#[test]
fn combat_v2_fight_turns_then_stops_then_casts_at_a_hero_behind_it() {
    let mut view = tactical_view(700, 1_000);
    own_hero_mut(&mut view).mana = 500;
    own_hero_mut(&mut view).facing.brads = 32_768;
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Fight);

    let first = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert!(matches!(
        first,
        Order::Move {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    ));
    view.tick = 7;
    own_hero_mut(&mut view).facing.brads = 0;
    tracker.observe_snapshot(&view).expect("turned");
    let second = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        second,
        Order::Move {
            target: Target::None
        }
    );
    view.tick = 10;
    tracker.observe_snapshot(&view).expect("stopped");
    let third = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        third,
        Order::Cast {
            slot: AbilitySlot(2),
            target: Target::None
        }
    );
}

#[test]
fn combat_v2_fight_selects_a_farther_killable_victim_over_a_healthy_decoy() {
    let mut view = tactical_view(800, 50);
    view.units.push(unit(
        entity(30, 1),
        UnitKind::Hero,
        Team::Dire,
        3_700,
        3_100,
    ));
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("victim selection");

    assert_eq!(
        space
            .decode(action)
            .expect("decode")
            .expect("pursuit")
            .order,
        Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
}

#[test]
fn combat_v2_safe_spell_kill_is_not_blocked_by_the_forty_percent_floor() {
    let mut view = tactical_view(700, 60);
    view.tick = 4;
    own_hero_mut(&mut view).hp = 350;
    own_hero_mut(&mut view).mana = 500;
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::None,
        },
        1,
    );

    let (action, _) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("safe kill");

    assert_eq!(action, cast(ControlledUnit::Hero, 2));
}

#[test]
fn combat_v2_farm_attack_clicks_then_pulls_toward_the_allied_ranged_creep() {
    let mut view = aggro_view();
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);

    let first = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        first,
        Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
    view.tick += 3;
    tracker
        .observe_snapshot(&view)
        .expect("aggro click applied");
    let second = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    let Order::Move {
        target: Target::Pos(destination),
    } = second
    else {
        panic!("pull movement")
    };
    assert!(destination.within(Vec2::from_ints(3_000, 3_000), Fixed::from_int(210)));
    assert!(
        destination.distance_squared(Vec2::from_ints(2_800, 3_000))
            < Vec2::from_ints(3_000, 3_000).distance_squared(Vec2::from_ints(2_800, 3_000))
    );
}

#[test]
fn combat_v2_farm_does_not_trade_a_guaranteed_last_hit_for_an_aggro_click() {
    let mut view = aggro_view();
    view.units
        .iter_mut()
        .find(|unit| unit.id == CREEP_ID)
        .expect("creep")
        .hp = 40;
    let tracker = tracker(view);

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Farm),
        )
        .expect("last hit");

    assert_eq!(
        space.decode(action).expect("decode").expect("attack").order,
        Order::Attack {
            target: Target::Unit(CREEP_ID)
        }
    );
}

#[test]
fn combat_v2_backoff_stops_when_no_progress_is_observed_within_its_bound() {
    let mut view = tactical_view(500, 1_000);
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Recover);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick = 19;
    tracker.observe_snapshot(&view).expect("blocked navigation");

    let order = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);

    assert_eq!(
        order,
        Order::Move {
            target: Target::None
        }
    );
}

#[test]
fn combat_v2_rejected_aggro_click_does_not_start_a_cooldown_or_pull_sequence() {
    let tracker = tracker(aggro_view());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);
    let first = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    let sequence = persistence.last_sequence().expect("sent");
    assert!(teacher.note_rejected(sequence));
    assert!(persistence.observe_rejection(sequence));

    let retry = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);

    assert_eq!(first, retry);
    assert_eq!(
        retry,
        Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
}

#[test]
fn combat_v2_aggro_movement_yields_to_a_new_guaranteed_last_hit() {
    let mut view = aggro_view();
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick += 3;
    view.units
        .iter_mut()
        .find(|unit| unit.id == CREEP_ID)
        .expect("creep")
        .hp = 40;
    tracker.observe_snapshot(&view).expect("last hit appears");
    view.tick += 1;
    tracker
        .observe_snapshot(&view)
        .expect("last hit health is stable");

    let order = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);

    assert_eq!(
        order,
        Order::Attack {
            target: Target::Unit(CREEP_ID)
        }
    );
}

#[test]
fn combat_v2_aggro_pull_stops_at_hold_expiry_and_does_not_reclick_before_cooldown() {
    let mut view = aggro_view();
    let start = view.tick;
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick += 3;
    tracker.observe_snapshot(&view).expect("clicked");
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick = start + 70;
    own_hero_mut(&mut view).pos.x -= Fixed::from_int(50);
    tracker.observe_snapshot(&view).expect("hold expires");

    let stop = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        stop,
        Order::Move {
            target: Target::None
        }
    );
    view.tick += 3;
    tracker.observe_snapshot(&view).expect("cooldown remains");
    let (action, space) = teacher
        .decide_tactical(&tracker, &persistence, &ItemReadiness::new(), &policy)
        .expect("no new click");
    assert_ne!(
        space
            .decode(action)
            .expect("decode")
            .map(|issued| issued.order),
        Some(Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID)
        })
    );
}

#[test]
fn combat_v2_self_sustain_does_not_hide_an_obsolete_underlying_retreat() {
    let mut view = tactical_view(500, 1_000);
    own_hero_mut(&mut view).hp = 300;
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Recover);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Use {
            slot: ItemSlot(0),
            target: Target::None,
        },
        2,
    );
    assert!(persistence.active_body_order_for(None).is_none());
    view.tick = 4;
    own_hero_mut(&mut view).hp = 1_000;
    own_hero_mut(&mut view).mana = 500;
    view.units.retain(|unit| unit.id != ENEMY_HERO_ID);
    tracker.observe_snapshot(&view).expect("recovered");

    let stop = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);

    assert_eq!(
        stop,
        Order::Move {
            target: Target::None
        }
    );
}

#[test]
fn combat_v2_fight_can_cast_at_a_faster_enemy_without_needing_to_chase() {
    let mut view = tactical_view(700, 1_000);
    view.tick = 4;
    own_hero_mut(&mut view).mana = 500;
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy")
        .move_speed = Fixed::from_int(310);
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::None,
        },
        1,
    );

    let (action, _) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Fight),
        )
        .expect("raze without chase");

    assert_eq!(action, cast(ControlledUnit::Hero, 2));
}

#[test]
fn combat_v2_teacher_deliberately_aims_when_a_nonlethal_raze_trade_is_in_reach() {
    let mut view = tactical_view(700, 1_000);
    own_hero_mut(&mut view).mana = 500;
    own_hero_mut(&mut view).facing.brads = 32_768;
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert_eq!(
        space.decode(action).expect("decode").expect("aim").order,
        Order::Move {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
}

#[cfg(feature = "builtin")]
#[test]
fn combat_v2_builtin_aggro_click_retargets_creep_without_a_hit_and_move_pulls_it() {
    use bota_server::game::{Command, wire_id};
    let (mut world, config, heroes, enemy_creep) = combat_world();
    let mut tracker = StateTracker::new(SlotId(0), &config.info()).expect("tracker");
    tracker
        .observe_snapshot(&world.view(Team::Radiant))
        .expect("visible world");
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);
    let origin = world.transform.get(enemy_creep).expect("creep").pos;

    let click = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        click,
        Order::Attack {
            target: Target::Unit(wire_id(heroes[1]))
        }
    );
    assert_eq!(world.validate_order(SlotId(0), None, &click), Ok(()));
    let events = world.advance(&[Command {
        slot: SlotId(0),
        unit: None,
        order: click,
    }]);
    assert_eq!(world.target_of(enemy_creep), Some(heroes[0]));
    assert!(!events.iter().any(|event| matches!(event.kind,
        bota_proto::EventKind::Damaged { source: Some(source), .. } if source == wire_id(heroes[0]))));
    for _ in 0..2 {
        world.advance(&[]);
    }
    tracker
        .observe_snapshot(&world.view(Team::Radiant))
        .expect("click applied");
    let movement = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert!(matches!(
        movement,
        Order::Move {
            target: Target::Pos(_)
        }
    ));
    world.advance(&[Command {
        slot: SlotId(0),
        unit: None,
        order: movement,
    }]);
    for _ in 0..27 {
        world.advance(&[]);
    }

    let after = world.transform.get(enemy_creep).expect("creep alive").pos;
    assert!(after.x < origin.x);
    assert_eq!(world.target_of(enemy_creep), Some(heroes[0]));
}

#[cfg(feature = "builtin")]
#[test]
fn combat_v2_builtin_fight_turn_stop_cast_hits_without_resuming_navigation() {
    use bota_server::game::{Command, UnitOrder};
    let (mut world, config, heroes, _) = combat_world();
    world.transform.get_mut(heroes[1]).expect("enemy").pos = Vec2::from_ints(9_300, 8_900);
    world
        .transform
        .get_mut(heroes[0])
        .expect("hero")
        .facing
        .brads = 32_768;
    world.settle();
    let mut tracker = StateTracker::new(SlotId(0), &config.info()).expect("tracker");
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Fight);
    let mut orders = Vec::new();
    let mut damage = 0;

    for tick in 0..24 {
        tracker
            .observe_snapshot(&world.view(Team::Radiant))
            .expect("snapshot");
        let mut commands = Vec::new();
        if tick % 3 == 0
            && let Some(order) =
                optional_combat_order(&mut teacher, &tracker, &mut persistence, &policy)
        {
            assert_eq!(world.validate_order(SlotId(0), None, &order), Ok(()));
            orders.push(order);
            commands.push(Command {
                slot: SlotId(0),
                unit: None,
                order,
            });
        }
        let events = world.advance(&commands);
        damage += combat_magical_damage(&events, heroes);
        if damage > 0 {
            break;
        }
    }

    assert!(
        matches!(
            orders.first(),
            Some(Order::Move {
                target: Target::Unit(_)
            })
        ),
        "{orders:?}"
    );
    assert!(
        orders.windows(2).any(|pair| matches!(
            pair,
            [
                Order::Move {
                    target: Target::None
                },
                Order::Cast { .. }
            ]
        )),
        "{orders:?}"
    );
    assert_eq!(damage, 67);
    assert_eq!(
        world.orders.get(heroes[0]).expect("body").current,
        UnitOrder::Stand
    );
}

#[cfg(feature = "builtin")]
fn combat_magical_damage(
    events: &[bota_server::game::Event],
    heroes: [bota_server::game::Entity; 2],
) -> i32 {
    events
        .iter()
        .filter_map(|event| match event.kind {
            bota_proto::EventKind::Damaged {
                source: Some(source),
                target,
                amount,
                kind: bota_proto::DamageKind::Magical,
                ..
            } if source == bota_server::game::wire_id(heroes[0])
                && target == bota_server::game::wire_id(heroes[1]) =>
            {
                Some(amount)
            }
            _ => None,
        })
        .sum()
}

#[cfg(feature = "builtin")]
fn combat_world() -> (
    bota_server::game::World,
    bota_server::game::MatchConfig,
    [bota_server::game::Entity; 2],
    bota_server::game::Entity,
) {
    use bota_server::game::{Command, MatchConfig, UnitOrder, World};
    let config = MatchConfig {
        match_id: 90_001,
        master_key: [17; 32],
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: SHADOW_FIEND,
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: SHADOW_FIEND,
            },
        ],
        map: MapId(1),
        tick_rate: 30,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 150,
    };
    let mut world = World::for_match(&config, config.rng());
    world.advance(&[Command {
        slot: SlotId(0),
        unit: None,
        order: Order::Learn {
            slot: AbilitySlot(0),
        },
    }]);
    let heroes = [
        world.seats[0].unit.expect("hero"),
        world.seats[1].unit.expect("enemy"),
    ];
    world.tick = 1_201;
    for (index, hero) in heroes.into_iter().enumerate() {
        world.transform.get_mut(hero).expect("hero position").pos =
            Vec2::from_ints(8_600 - index as i32 * 100, 8_900);
        world.set_order(hero, UnitOrder::Stand);
        world.statuses.remove(hero);
        world.seats[index].gold = 0;
    }
    let enemy_creep =
        combat_world_creep(&mut world, Team::Dire, Vec2::from_ints(9_050, 8_900), false);
    combat_world_creep(
        &mut world,
        Team::Radiant,
        Vec2::from_ints(8_400, 8_900),
        true,
    );
    let ally = combat_world_creep(
        &mut world,
        Team::Radiant,
        Vec2::from_ints(9_100, 9_000),
        false,
    );
    world.settle();
    world.set_target(enemy_creep, ally);
    (world, config, heroes, enemy_creep)
}

#[cfg(feature = "builtin")]
fn combat_world_creep(
    world: &mut bota_server::game::World,
    team: Team,
    position: Vec2,
    ranged: bool,
) -> bota_server::game::Entity {
    use bota_server::game::{LaneAi, MELEE_CREEP, RANGED_CREEP};
    let entity = world.spawn_unit(
        if ranged { &RANGED_CREEP } else { &MELEE_CREEP },
        team,
        position,
    );
    world.march.remove(entity);
    world.lane_ai.insert(
        entity,
        LaneAi {
            anchor: None,
            last_seen: None,
            keep_until: 0,
            roused_by: None,
            roused_at_own: false,
            chase_until: 0,
        },
    );
    entity
}

#[test]
fn combat_v2_farm_can_end_hero_harassment_after_the_bounded_attack_commitment() {
    for (tick, expected) in [
        (4, StructuredAction::Continue),
        (
            45,
            StructuredAction::Stop {
                unit: ControlledUnit::Hero,
            },
        ),
    ] {
        let mut view = tactical_view(500, 1_000);
        view.tick = tick;
        let tracker = tracker(view);
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        record_combat_order(
            &mut teacher,
            &mut persistence,
            Order::Attack {
                target: Target::Unit(ENEMY_HERO_ID),
            },
            1,
        );

        let (action, _) = teacher
            .decide_tactical(
                &tracker,
                &persistence,
                &ItemReadiness::new(),
                &tactical_policy(crate::TacticalMode::Farm),
            )
            .expect("farm can leave a trade");

        assert_eq!(action, expected, "tick {tick}");
    }
}

#[test]
fn combat_v2_prediction_caps_velocity_and_clamps_at_the_public_map_boundary() {
    let mut view = tactical_view(4_000, 1_000);
    let mut tracker = tracker(view.clone());
    view.tick = 2;
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy")
        .pos
        .x = Fixed::from_int(8_190);
    tracker
        .observe_snapshot(&view)
        .expect("fast observed displacement");
    let enemy = tracker
        .current()
        .expect("snapshot")
        .units
        .iter()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy");

    let predicted = crate::teacher::predicted_position(&tracker, enemy);

    assert!(predicted.x.raw < Fixed::from_int(8_192).raw);
    assert!(enemy.pos.within(predicted, Fixed::from_int(10)));
}

#[test]
fn combat_followup_direct_safety_and_deployment_record_retreat_and_cancel_it() {
    for deployment in [false, true] {
        let mut view = base_view();
        own_hero_mut(&mut view).hp = 300;
        let mut tracker = tracker(view.clone());
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        let space = crate::ActionSpace::from_tracker(&tracker).expect("space");
        let action = if deployment {
            teacher.deployment_action(&tracker, &space)
        } else {
            teacher.safety_action(&tracker, &space)
        }
        .expect("retreat");
        let issued = space.decode(action).expect("decode").expect("move");
        assert!(matches!(
            issued.order,
            Order::Move {
                target: Target::Pos(_)
            }
        ));
        record_combat_order(&mut teacher, &mut persistence, issued.order, space.tick());
        view.tick += 3;
        own_hero_mut(&mut view).hp = 1_000;
        own_hero_mut(&mut view).mana = 500;
        tracker.observe_snapshot(&view).expect("recovered");
        let space = crate::ActionSpace::from_tracker(&tracker).expect("space");

        let action = if deployment {
            teacher.deployment_action(&tracker, &space)
        } else {
            teacher.safety_action(&tracker, &space)
        };

        assert_eq!(
            action,
            Some(StructuredAction::Stop {
                unit: ControlledUnit::Hero
            })
        );
    }
}

#[test]
fn combat_followup_targeted_self_sustain_and_raze_preserve_retreat_expiry() {
    for item_id in [1, 2, 7, 0] {
        let mut view = base_view();
        own_hero_mut(&mut view).hp = 300;
        own_hero_mut(&mut view).mana = 500;
        if item_id != 0 {
            let mut held = item(
                ItemId(item_id),
                Some(if item_id == 7 { Aim::Tree } else { Aim::Unit }),
                600,
                Some(1),
            );
            held.cooldown_left = 10;
            own_hero_mut(&mut view).items[0] = Some(held);
        }
        let mut tracker = tracker(view.clone());
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        let policy = tactical_policy(crate::TacticalMode::Teacher);
        send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
        let order = preserving_order(item_id, HERO_ID, Vec2::from_ints(3_100, 3_000));
        record_combat_order(&mut teacher, &mut persistence, order, 10);
        view.tick = 91;
        own_hero_mut(&mut view).pos.x -= Fixed::from_int(50);
        tracker.observe_snapshot(&view).expect("original deadline");

        let (action, _) = teacher
            .decide(&tracker, &persistence, &ItemReadiness::new())
            .expect("deadline cancellation");

        assert_eq!(
            action,
            StructuredAction::Stop {
                unit: ControlledUnit::Hero
            },
            "item {item_id}"
        );
    }
}

#[test]
fn combat_followup_targeted_sustain_rejection_restores_the_retreat_plan() {
    let mut view = base_view();
    own_hero_mut(&mut view).hp = 300;
    let mut held = item(ItemId(2), Some(Aim::Unit), 600, Some(1));
    held.cooldown_left = 10;
    own_hero_mut(&mut view).items[0] = Some(held);
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    send_combat_decision(
        &mut teacher,
        &tracker,
        &mut persistence,
        &tactical_policy(crate::TacticalMode::Teacher),
    );
    record_combat_order(
        &mut teacher,
        &mut persistence,
        preserving_order(2, HERO_ID, Vec2::ZERO),
        2,
    );
    assert!(teacher.note_rejected(2));
    assert!(persistence.observe_rejection(2));
    view.tick = 4;
    own_hero_mut(&mut view).hp = 1_000;
    own_hero_mut(&mut view).mana = 500;
    tracker.observe_snapshot(&view).expect("recovered");

    let (action, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("cancel restored plan");

    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
}

#[test]
fn combat_followup_farm_stops_conflicting_movement_then_casts_the_centered_raze() {
    let mut view = farm_raze_view();
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::Pos(Vec2::from_ints(2_000, 3_000)),
        },
        0,
    );
    let policy = tactical_policy(crate::TacticalMode::Farm);

    let stop = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        stop,
        Order::Move {
            target: Target::None
        }
    );
    view.tick += 3;
    tracker.observe_snapshot(&view).expect("stopped");
    let cast = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        cast,
        Order::Cast {
            slot: AbilitySlot(1),
            target: Target::None
        }
    );
}

#[test]
fn combat_followup_farm_aims_at_a_killable_creep_behind_it_before_casting() {
    let mut view = farm_raze_view();
    own_hero_mut(&mut view).facing.brads = 32_768;
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);

    let turn = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        turn,
        Order::Move {
            target: Target::Unit(CREEP_ID)
        }
    );
    view.tick += 6;
    own_hero_mut(&mut view).facing.brads = 0;
    tracker.observe_snapshot(&view).expect("turned");
    let stop = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        stop,
        Order::Move {
            target: Target::None
        }
    );
    view.tick += 3;
    tracker.observe_snapshot(&view).expect("stopped");
    let cast = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    assert_eq!(
        cast,
        Order::Cast {
            slot: AbilitySlot(1),
            target: Target::None
        }
    );
}

#[cfg(feature = "builtin")]
#[test]
fn combat_followup_server_targeted_consumables_and_raze_keep_move_until_stale_stop() {
    for item_id in [1, 2, 7, 0] {
        verify_server_preserving_order(item_id);
    }
}

#[cfg(feature = "builtin")]
fn verify_server_preserving_order(item_id: u16) {
    use bota_server::game::{UnitOrder, wire_id};
    let (mut world, config, heroes, _) = combat_world();
    let tree = world.transform.get(heroes[0]).expect("hero").pos + Vec2::from_ints(0, 100);
    prepare_preserving_item(&mut world, heroes[0], item_id, tree);
    world.health.get_mut(heroes[0]).expect("health").hp = Fixed::from_int(150);
    let mut tracker = StateTracker::new(SlotId(0), &config.info()).expect("tracker");
    tracker
        .observe_snapshot(&world.view(Team::Radiant))
        .expect("snapshot");
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let movement = send_combat_decision(
        &mut teacher,
        &tracker,
        &mut persistence,
        &tactical_policy(crate::TacticalMode::Teacher),
    );
    assert!(matches!(
        movement,
        Order::Move {
            target: Target::Pos(_)
        }
    ));
    advance_combat_order(&mut world, movement);
    let body = world.orders.get(heroes[0]).expect("body").current;
    if item_id != 0 {
        world.inventory.get_mut(heroes[0]).expect("inventory").slots[0]
            .as_mut()
            .expect("item")
            .cooldown = 0;
    }
    let order = preserving_order(item_id, wire_id(heroes[0]), tree);
    assert_eq!(world.validate_order(SlotId(0), None, &order), Ok(()));
    record_combat_order(&mut teacher, &mut persistence, order, world.tick);
    advance_combat_order(&mut world, order);
    assert_eq!(world.orders.get(heroes[0]).expect("body").current, body);
    assert_preserving_effect(&world, heroes[0], item_id);
    world.fill_pools(heroes[0]);
    tracker
        .observe_snapshot(&world.view(Team::Radiant))
        .expect("recovered");
    let space = crate::ActionSpace::from_tracker(&tracker).expect("space");

    let stop = teacher
        .safety_action(&tracker, &space)
        .expect("cancel underlying navigation");
    assert_eq!(
        stop,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
    let order = space
        .decode(stop)
        .expect("decode")
        .expect("Stop order")
        .order;
    advance_combat_order(&mut world, order);
    assert_eq!(
        world.orders.get(heroes[0]).expect("body").current,
        UnitOrder::Stand
    );
}

#[cfg(feature = "builtin")]
fn advance_combat_order(world: &mut bota_server::game::World, order: Order) {
    assert_eq!(world.validate_order(SlotId(0), None, &order), Ok(()));
    world.advance(&[bota_server::game::Command {
        slot: SlotId(0),
        unit: None,
        order,
    }]);
}

#[cfg(feature = "builtin")]
fn assert_preserving_effect(
    world: &bota_server::game::World,
    hero: bota_server::game::Entity,
    item_id: u16,
) {
    if item_id == 0 {
        assert!(world.abilities.get(hero).expect("abilities").slots[2].cooldown > 0);
    } else {
        let effect = EffectId(if item_id == 1 { 2 } else { 1 });
        assert!(
            world
                .view(Team::Radiant)
                .units
                .iter()
                .find(|unit| unit.id == bota_server::game::wire_id(hero))
                .expect("hero")
                .effects
                .iter()
                .any(|held| held.id == effect)
        );
    }
}

#[cfg(feature = "builtin")]
fn prepare_preserving_item(
    world: &mut bota_server::game::World,
    hero: bota_server::game::Entity,
    item_id: u16,
    tree: Vec2,
) {
    if item_id != 0 {
        world.inventory.get_mut(hero).expect("inventory").slots[0] =
            Some(bota_server::game::ItemStack {
                id: ItemId(item_id),
                charges: 1,
                cooldown: 20,
                mute: 0,
                mode: None,
                bought_tick: world.tick,
                touched: false,
                owner: SlotId(0),
                for_sale: false,
            });
    }
    if item_id == 7 {
        world.trees.plant(tree, world.tick + 300);
        world.lay_passability();
        world.lay_sight_block();
    }
}

#[cfg(feature = "builtin")]
#[test]
fn combat_followup_server_farm_raze_last_hits_after_stopping_or_turning_without_resuming_move() {
    for facing in [0, 32_768] {
        verify_server_farm_raze(facing);
    }
}

#[cfg(feature = "builtin")]
fn verify_server_farm_raze(facing: u16) {
    use bota_server::game::UnitOrder;
    let (mut world, config, heroes, creep) = combat_world();
    prepare_farm_cast(&mut world, heroes[0], creep, facing);
    let mut tracker = StateTracker::new(SlotId(0), &config.info()).expect("tracker");
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::Pos(Vec2::from_ints(8_000, 8_900)),
        },
        world.tick - 1,
    );
    let (orders, damage) = drive_farm_cast(
        &mut world,
        &mut tracker,
        &mut teacher,
        &mut persistence,
        [heroes[0], creep],
    );
    assert!(
        orders.windows(2).any(|pair| matches!(
            pair,
            [
                Order::Move {
                    target: Target::None
                },
                Order::Cast { .. }
            ]
        )),
        "{orders:?}"
    );
    assert!(
        (80..=90).contains(&damage),
        "damage={damage}, orders={orders:?}"
    );
    assert_eq!(world.seats[0].last_hits, 1);
    assert_eq!(
        world.orders.get(heroes[0]).expect("body").current,
        UnitOrder::Stand
    );
}

#[cfg(feature = "builtin")]
fn prepare_farm_cast(
    world: &mut bota_server::game::World,
    hero: bota_server::game::Entity,
    creep: bota_server::game::Entity,
    facing: u16,
) {
    use bota_server::game::UnitOrder;
    let creeps = world
        .entities
        .iter()
        .filter(|entity| {
            world
                .kind
                .get(*entity)
                .is_some_and(|kind| matches!(kind, UnitKind::CreepMelee | UnitKind::CreepRanged))
        })
        .collect::<Vec<_>>();
    for entity in creeps {
        world.set_order(entity, UnitOrder::Stand);
    }
    world.health.get_mut(creep).expect("creep hp").hp = Fixed::from_int(80);
    world
        .transform
        .get_mut(hero)
        .expect("hero facing")
        .facing
        .brads = facing;
    world.set_order(
        hero,
        UnitOrder::Move {
            pos: Vec2::from_ints(8_000, 8_900),
        },
    );
}

#[cfg(feature = "builtin")]
fn drive_farm_cast(
    world: &mut bota_server::game::World,
    tracker: &mut StateTracker,
    teacher: &mut Teacher,
    persistence: &mut OrderPersistence,
    targets: [bota_server::game::Entity; 2],
) -> (Vec<Order>, i32) {
    let policy = tactical_policy(crate::TacticalMode::Farm);
    let mut orders = Vec::with_capacity(8);
    let mut damage = 0;
    for tick in 0..24 {
        tracker
            .observe_snapshot(&world.view(Team::Radiant))
            .expect("snapshot");
        let mut commands = Vec::with_capacity(1);
        if tick % 3 == 0
            && let Some(order) = optional_combat_order(teacher, tracker, persistence, &policy)
        {
            assert_eq!(world.validate_order(SlotId(0), None, &order), Ok(()));
            orders.push(order);
            commands.push(bota_server::game::Command {
                slot: SlotId(0),
                unit: None,
                order,
            });
        }
        damage += combat_magical_damage(&world.advance(&commands), targets);
        if damage > 0 {
            break;
        }
    }
    (orders, damage)
}

#[test]
fn combat_followup_farm_can_raze_tower_covered_creeps_without_approaching_them() {
    let mut view = farm_raze_view();
    view.tick = 4;
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_100, 6_000);
    view.units
        .iter_mut()
        .find(|unit| unit.id == CREEP_ID)
        .expect("creep")
        .pos = Vec2::from_ints(5_850, 6_000);
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy")
        .pos = Vec2::from_ints(5_100, 6_900);
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::None,
        },
        1,
    );

    let (action, _) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Farm),
        )
        .expect("no chase needed");

    assert_eq!(action, cast(ControlledUnit::Hero, 2));
}

fn preserving_order(item_id: u16, hero: EntityId, tree: Vec2) -> Order {
    match item_id {
        0 => Order::Cast {
            slot: AbilitySlot(2),
            target: Target::None,
        },
        7 => Order::Use {
            slot: ItemSlot(0),
            target: Target::Pos(tree),
        },
        _ => Order::Use {
            slot: ItemSlot(0),
            target: Target::Unit(hero),
        },
    }
}

fn farm_raze_view() -> WorldView {
    let mut view = tactical_view(1_100, 1_000);
    own_hero_mut(&mut view).mana = 500;
    let mut creep = unit(CREEP_ID, UnitKind::CreepRanged, Team::Dire, 3_450, 3_000);
    creep.hp = 80;
    view.units.push(creep);
    sort_units(&mut view);
    view
}

fn aggro_view() -> WorldView {
    let mut view = tactical_view(800, 1_000);
    view.tick = 1_201;
    let mut melee = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_300, 3_000);
    melee.attack_range = Fixed::from_int(100);
    view.units.push(melee);
    view.units.push(unit(
        entity(31, 1),
        UnitKind::CreepRanged,
        Team::Radiant,
        2_800,
        3_000,
    ));
    sort_units(&mut view);
    view
}

fn send_combat_decision(
    teacher: &mut Teacher,
    tracker: &StateTracker,
    persistence: &mut OrderPersistence,
    policy: &crate::TacticalPolicy,
) -> Order {
    optional_combat_order(teacher, tracker, persistence, policy).expect("new order")
}

fn optional_combat_order(
    teacher: &mut Teacher,
    tracker: &StateTracker,
    persistence: &mut OrderPersistence,
    policy: &crate::TacticalPolicy,
) -> Option<Order> {
    let (action, space) = teacher
        .decide_tactical(tracker, persistence, &ItemReadiness::new(), policy)
        .expect("combat decision");
    let issued = space.decode(action).expect("decode")?;
    let sequence = persistence.last_sequence().unwrap_or(0) + 1;
    persistence.record_sent(sequence, issued).expect("record");
    teacher.note_sent(sequence, issued, space.tick());
    Some(issued.order)
}

fn record_combat_order(
    teacher: &mut Teacher,
    persistence: &mut OrderPersistence,
    order: Order,
    tick: u32,
) {
    let issued = IssuedOrder { unit: None, order };
    let sequence = persistence.last_sequence().unwrap_or(0) + 1;
    persistence.record_sent(sequence, issued).expect("record");
    teacher.note_sent(sequence, issued, tick);
}

fn tactical_policy(mode: crate::TacticalMode) -> crate::TacticalPolicy {
    let mut parameters = [0.0; crate::TACTICAL_PARAMETERS];
    parameters[crate::TACTICAL_OUTPUT_BIAS_OFFSET + mode.index()] = 1.0;
    crate::TacticalPolicy::from_parameters(&parameters).expect("forced macro")
}

fn decide_stopped(tracker: &StateTracker) -> (StructuredAction, crate::ActionSpace) {
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let tick = tracker.current().expect("snapshot").tick.saturating_sub(1);
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::None,
        },
        tick,
    );
    teacher
        .decide(tracker, &persistence, &ItemReadiness::new())
        .expect("stopped teacher")
}

#[test]
fn finish_one_auto_selects_the_hero_before_farm_at_fifteen_percent_health() {
    let mut view = one_auto_view();
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_200, 3_000);
    creep.hp = 10;
    creep.attack_damage = 0;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);

    let (action, space) = Teacher::new()
        .decide_tactical(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Farm),
        )
        .expect("one-auto finish");

    assert_eq!(
        space.decode(action).expect("decode").expect("attack").order,
        Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
}

#[test]
fn finish_one_auto_declines_leeway_healing_disables_and_lethal_replies() {
    for scenario in 0..7 {
        let mut view = one_auto_view();
        match scenario {
            0 => enemy_hero_mut(&mut view).pos.x = Fixed::from_int(3_549),
            1 => enemy_hero_mut(&mut view).hp = 37,
            2 => own_hero_mut(&mut view).hp = 37,
            3 => own_hero_mut(&mut view).statuses.bits = StatusFlags::STUNNED,
            4 => own_hero_mut(&mut view).statuses.bits = StatusFlags::DOT,
            5 => {
                let enemy = enemy_hero_mut(&mut view);
                enemy.mana = 75;
                enemy.abilities = vec![ability(13, 2, 75, false)];
            }
            _ => own_hero_mut(&mut view).attack_interval = 255,
        }
        let tracker = tracker(view);

        let (action, space) = decide(&tracker);

        assert!(
            !matches!(action, StructuredAction::AttackUnit { .. }),
            "scenario {scenario}"
        );
        assert!(space.allows(action));
    }
}

#[test]
fn finish_raze_budget_allows_a_nonlethal_known_raze_but_rejects_a_lethal_one() {
    for (level, cost, expected) in [(1, 75, true), (2, 80, false)] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = cost;
        enemy.abilities = vec![ability(13, level, cost, false)];
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_raze_budget_counts_only_mana_feasible_casts_from_three_learned_razes() {
    for (mana, expected) in [(75, true), (150, false)] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = mana;
        enemy.abilities = [13, 14, 15].map(|id| ability(id, 1, 75, false)).to_vec();
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_raze_budget_chooses_the_most_damaging_affordable_subset_not_slot_order() {
    for ids in [[13, 14, 15], [15, 14, 13]] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = 80;
        enemy.abilities = ids
            .map(|id| {
                ability(
                    id,
                    if id == 14 { 2 } else { 1 },
                    if id == 14 { 80 } else { 75 },
                    false,
                )
            })
            .to_vec();
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert!(matches!(action, StructuredAction::MovePoint { .. }));
    }
}

#[test]
fn finish_raze_budget_ignores_cooldowns_beyond_the_deadline_but_not_at_it() {
    for (cooldown, expected) in [(44, false), (45, true)] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = 80;
        let mut raze = ability(13, 2, 80, false);
        raze.cooldown_left = cooldown;
        enemy.abilities = vec![raze];
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_raze_budget_excludes_impossible_geometry_but_ignores_facing() {
    for (reach_id, expected) in [(13, false), (15, true)] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.pos = Vec2::from_ints(3_080, 3_000);
        enemy.move_speed = Fixed::ZERO;
        enemy.facing.brads = 0;
        enemy.mana = 80;
        enemy.abilities = vec![ability(reach_id, 2, 80, false)];
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_raze_budget_reserves_natural_mana_regeneration_and_visible_restore_uncertainty() {
    for (mana, restore, expected) in [(75, false, true), (78, false, false), (0, true, true)] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = mana;
        enemy.abilities = vec![ability(13, 2, 80, false)];
        if restore {
            enemy.effects.push(EffectView {
                id: EffectId(2),
                ticks_left: Some(300),
                stacks: None,
            });
        }
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_raze_budget_keeps_the_unknown_active_and_requiem_veto() {
    for id in [16, 999] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = 150;
        enemy.abilities = vec![ability(id, 1, 150, false)];
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert!(matches!(action, StructuredAction::MovePoint { .. }));
    }
}

fn finish_raze_view() -> WorldView {
    let mut view = one_auto_view();
    let hero = own_hero_mut(&mut view);
    hero.hp = 150;
    hero.max_hp = 1_000;
    let enemy = enemy_hero_mut(&mut view);
    enemy.max_mana = 291;
    view
}

#[test]
fn finish_restoration_empty_stick_and_wand_cannot_fund_a_raze() {
    for id in [35, 36] {
        let tracker = tracker(finish_item_view(id, 0, 0));

        let (action, _) = decide(&tracker);

        assert!(matches!(action, StructuredAction::AttackUnit { .. }));
    }
}

#[test]
fn finish_restoration_one_wand_charge_adds_only_fifteen_mana() {
    for (mana, expected) in [(0, true), (61, true), (62, false)] {
        let mut view = finish_item_view(36, 1, 0);
        enemy_hero_mut(&mut view).mana = mana;
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_restoration_uses_only_inventory_items_ready_within_the_window() {
    for (slot, cooldown, mute, funded) in [
        (0, 0, 0, true),
        (5, 44, 0, true),
        (0, 45, 0, false),
        (0, 0, 44, true),
        (0, 0, 45, false),
        (6, 0, 0, false),
        (8, 0, 0, false),
    ] {
        let mut view = finish_item_view(36, 1, slot);
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = 62;
        let held = enemy.items[slot].as_mut().expect("wand");
        held.cooldown_left = cooldown;
        held.mute_left = mute;
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            !funded,
            "slot={slot} cooldown={cooldown} mute={mute}"
        );
    }
}

#[test]
fn finish_restoration_wand_healing_invalidates_one_hit_even_without_enemy_spells() {
    for (hp, expected) in [(5, true), (15, true), (16, false), (20, false)] {
        let mut view = finish_item_view(36, 1, 0);
        let enemy = enemy_hero_mut(&mut view);
        enemy.hp = hp;
        enemy.abilities.clear();
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_restoration_unavailable_wand_does_not_inflate_enemy_health() {
    for (charges, slot, cooldown) in [(0, 0, 0), (1, 6, 0), (1, 0, 45)] {
        let mut view = finish_item_view(36, charges, slot);
        let enemy = enemy_hero_mut(&mut view);
        enemy.hp = 20;
        enemy.abilities.clear();
        enemy.items[slot].as_mut().expect("wand").cooldown_left = cooldown;
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert!(matches!(action, StructuredAction::AttackUnit { .. }));
    }
}

#[test]
fn finish_restoration_treads_mana_increase_is_finite_and_respects_current_mode() {
    for (mana, mode, expected) in [
        (0, Attribute::Strength, true),
        (30, Attribute::Strength, false),
        (30, Attribute::Intelligence, true),
    ] {
        let mut view = finish_item_view(29, 0, 0);
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = mana;
        enemy.items[0].as_mut().expect("treads").mode = Some(mode);
        enemy.abilities = [13, 14, 15].map(|id| ability(id, 1, 75, false)).to_vec();
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_restoration_clarity_is_duration_bounded_and_does_not_stack_with_its_item() {
    for (mana, duration, consumable, expected) in [
        (0, 300, false, true),
        (67, 300, false, true),
        (68, 300, false, false),
        (75, 5, false, true),
        (76, 5, false, false),
        (60, 300, true, true),
    ] {
        let mut view = finish_raze_view();
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = mana;
        enemy.abilities = vec![ability(13, 2, 80, false)];
        enemy.effects.push(EffectView {
            id: EffectId(2),
            ticks_left: Some(duration),
            stacks: None,
        });
        if consumable {
            enemy.items = vec![Some(item(ItemId(1), Some(Aim::Unit), 250, Some(1)))];
        }
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_restoration_clarity_item_requires_a_charge_and_a_ready_inventory_slot() {
    for (charges, slot, cooldown, expected) in [
        (1, 0, 0, false),
        (0, 0, 0, true),
        (1, 6, 0, true),
        (1, 0, 45, true),
    ] {
        let mut view = finish_item_view(1, charges, slot);
        let enemy = enemy_hero_mut(&mut view);
        enemy.mana = 68;
        enemy.items[slot].as_mut().expect("clarity").cooldown_left = cooldown;
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

#[test]
fn finish_restoration_reserves_future_charges_only_for_a_visible_ready_allied_caster() {
    for (mana, cooldown, expected) in [(0, 0, true), (75, 0, false), (75, 45, true)] {
        let mut view = finish_item_view(36, 0, 0);
        let mut caster = unit(entity(30, 1), UnitKind::Hero, Team::Radiant, 3_200, 3_100);
        caster.mana = mana;
        let mut raze = ability(13, 1, 75, false);
        raze.cooldown_left = cooldown;
        caster.abilities = vec![raze];
        view.units.push(caster);
        sort_units(&mut view);
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            expected
        );
    }
}

fn finish_item_view(id: u16, charges: u8, slot: usize) -> WorldView {
    assert!(slot < 9);
    let mut view = finish_raze_view();
    let enemy = enemy_hero_mut(&mut view);
    enemy.hp = 5;
    enemy.mana = 0;
    enemy.abilities = vec![ability(13, 2, 80, false)];
    enemy.items = vec![None; 9];
    enemy.items[slot] = Some(item(ItemId(id), Some(Aim::Own), 0, Some(charges)));
    view
}

#[test]
fn finish_one_auto_expires_without_renewal_and_aborts_when_normal_range_is_lost() {
    for elapsed in [3, 60, 63] {
        let mut view = one_auto_view();
        let mut tracker = tracker(view.clone());
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        let policy = tactical_policy(crate::TacticalMode::Farm);
        let first = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
        assert_eq!(
            first,
            Order::Attack {
                target: Target::Unit(ENEMY_HERO_ID)
            }
        );
        view.tick += elapsed;
        if elapsed == 3 {
            enemy_hero_mut(&mut view).pos.x = Fixed::from_int(3_549);
        }
        tracker
            .observe_snapshot(&view)
            .expect("expired or out of reach");

        let stop = send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
        assert!(matches!(stop, Order::Move { .. }));
        view.tick += 3;
        tracker.observe_snapshot(&view).expect("next decision");
        let (next, _) = teacher
            .decide(&tracker, &persistence, &ItemReadiness::new())
            .expect("no renewal");
        assert!(!matches!(next, StructuredAction::AttackUnit { .. }));
    }
}

#[test]
fn finish_one_auto_adopts_an_existing_attack_without_resetting_its_deadline() {
    let mut view = one_auto_view();
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Attack {
            target: Target::Unit(ENEMY_HERO_ID),
        },
        view.tick,
    );

    let (first, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("adopt");
    assert_eq!(first, StructuredAction::Continue);
    view.tick += 63;
    tracker.observe_snapshot(&view).expect("deadline");
    let (next, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("expire");
    assert!(matches!(
        next,
        StructuredAction::Stop { .. } | StructuredAction::MovePoint { .. }
    ));
}

#[test]
fn finish_one_auto_safety_adoption_is_bounded_and_rejected_orders_roll_back_the_attempt() {
    for rejected in [false, true] {
        let mut view = one_auto_view();
        let mut tracker = tracker(view.clone());
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        record_combat_order(
            &mut teacher,
            &mut persistence,
            Order::Attack {
                target: Target::Unit(ENEMY_HERO_ID),
            },
            view.tick,
        );
        let space = crate::ActionSpace::from_tracker(&tracker).expect("space");
        let action = teacher
            .safety_action(&tracker, &space)
            .expect("adopt finish");
        assert_eq!(
            space.decode(action).expect("decode").expect("attack").order,
            Order::Attack {
                target: Target::Unit(ENEMY_HERO_ID)
            }
        );
        if rejected {
            assert!(teacher.note_rejected(1));
            assert!(persistence.observe_rejection(1));
        }
        view.tick += 63;
        tracker.observe_snapshot(&view).expect("expired");

        let (action, _) = teacher
            .decide(&tracker, &persistence, &ItemReadiness::new())
            .expect("after deadline");

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            rejected
        );
        if !rejected {
            assert!(matches!(
                action,
                StructuredAction::Stop { .. } | StructuredAction::MovePoint { .. }
            ));
        }
    }
}

#[test]
fn finish_one_auto_vetoes_visible_hostile_flight_even_with_no_visible_launcher() {
    let mut view = one_auto_view();
    view.projectiles.push(bota_proto::ProjectileView {
        id: entity(90, 1),
        pos: Vec2::from_ints(3_100, 3_000),
        facing: Angle { brads: 32_768 },
        team: Team::Dire,
        ability: None,
    });
    let tracker = tracker(view);

    let (action, _) = decide(&tracker);

    assert!(matches!(action, StructuredAction::MovePoint { .. }));
}

#[test]
fn finish_one_auto_does_not_restart_by_switching_victims_while_still_low_health() {
    let mut view = one_auto_view();
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let policy = tactical_policy(crate::TacticalMode::Farm);
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick += 63;
    tracker.observe_snapshot(&view).expect("expired");
    send_combat_decision(&mut teacher, &tracker, &mut persistence, &policy);
    view.tick += 3;
    enemy_hero_mut(&mut view).id = entity(30, 1);
    view.players[1].unit = Some(entity(30, 1));
    sort_units(&mut view);
    tracker.observe_snapshot(&view).expect("different victim");

    let (action, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("no second trial");

    assert!(!matches!(action, StructuredAction::AttackUnit { .. }));
}

#[test]
fn tower_guard_new_navigation_does_not_cross_toward_a_hero_even_with_a_wave() {
    let mut view = base_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_000, 6_000);
    view.units
        .retain(|unit| !matches!(unit.kind, UnitKind::Ancient));
    view.units.push(unit(
        entity(31, 1),
        UnitKind::CreepRanged,
        Team::Radiant,
        6_700,
        6_000,
    ));
    view.units.push(unit(
        ENEMY_HERO_ID,
        UnitKind::Hero,
        Team::Dire,
        6_400,
        6_000,
    ));
    let tracker = tracker({
        sort_units(&mut view);
        view
    });

    let (action, space) = decide(&tracker);

    let issued = space.decode(action).expect("decode").expect("navigation");
    if let Order::Move {
        target: Target::Pos(point),
    }
    | Order::Attack {
        target: Target::Pos(point),
    } = issued.order
    {
        assert!(
            point.x <= Fixed::from_int(5_200),
            "must not cut through tower to a wave: {point:?}"
        );
    } else {
        assert!(matches!(
            action,
            StructuredAction::Stop { .. } | StructuredAction::Hold { .. }
        ));
    }
}

#[test]
fn tower_guard_rechecks_an_aim_follow_when_the_target_enters_tower_cover() {
    let mut view = tactical_view(-500, 500);
    let hero = own_hero_mut(&mut view);
    hero.pos = Vec2::from_ints(5_100, 6_000);
    hero.mana = 75;
    enemy_hero_mut(&mut view).pos = Vec2::from_ints(4_600, 6_000);
    let mut tracker = tracker(view.clone());
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    let aim = send_combat_decision(
        &mut teacher,
        &tracker,
        &mut persistence,
        &tactical_policy(crate::TacticalMode::Fight),
    );
    assert_eq!(
        aim,
        Order::Move {
            target: Target::Unit(ENEMY_HERO_ID)
        }
    );
    view.tick += 3;
    enemy_hero_mut(&mut view).pos = Vec2::from_ints(5_850, 6_000);
    tracker.observe_snapshot(&view).expect("target at tower");

    let (action, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("revalidate aim");

    assert_eq!(
        action,
        StructuredAction::Stop {
            unit: ControlledUnit::Hero
        }
    );
}

#[test]
fn tower_guard_cancels_retained_attack_follow_move_and_attack_move_across_the_corridor() {
    for order_kind in 0usize..4 {
        let mut view = tactical_view(1_100, 100);
        own_hero_mut(&mut view).pos = Vec2::from_ints(5_300, 5_700);
        enemy_hero_mut(&mut view).pos = Vec2::from_ints(6_400, 5_300);
        view.tick = 4;
        let target = if order_kind < 2 {
            Target::Unit(ENEMY_HERO_ID)
        } else {
            Target::Pos(Vec2::from_ints(6_400, 5_300))
        };
        let order = if order_kind.is_multiple_of(2) {
            Order::Attack { target }
        } else {
            Order::Move { target }
        };
        let tracker = tracker(view);
        let mut teacher = Teacher::new();
        let mut persistence = OrderPersistence::default();
        record_combat_order(&mut teacher, &mut persistence, order, 1);
        let space = crate::ActionSpace::from_tracker(&tracker).expect("space");

        let action = teacher
            .safety_action(&tracker, &space)
            .expect("unsafe retained order");

        assert_eq!(
            action,
            StructuredAction::Stop {
                unit: ControlledUnit::Hero
            },
            "order {order:?}"
        );
    }
}

#[test]
fn tower_guard_allows_an_in_range_shot_from_outside_but_not_creep_covered_hero_aggro() {
    for own_x in [5_100, 5_250, 5_500] {
        let mut view = one_auto_view();
        own_hero_mut(&mut view).pos = Vec2::from_ints(own_x, 6_000);
        enemy_hero_mut(&mut view).pos = Vec2::from_ints(own_x + 240, 6_000);
        view.units.push(unit(
            CREEP_ID,
            UnitKind::CreepMelee,
            Team::Radiant,
            5_700,
            6_000,
        ));
        sort_units(&mut view);
        let tracker = tracker(view);

        let (action, _) = decide(&tracker);

        assert_eq!(
            matches!(action, StructuredAction::AttackUnit { .. }),
            own_x == 5_100
        );
    }
}

#[test]
fn tower_guard_does_not_block_a_retained_escape_from_inside_range() {
    let mut view = base_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_500, 6_000);
    view.tick = 4;
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Move {
            target: Target::Pos(Vec2::from_ints(5_000, 6_000)),
        },
        1,
    );

    let (action, _) = teacher
        .decide(&tracker, &persistence, &ItemReadiness::new())
        .expect("escape");

    assert!(!matches!(
        action,
        StructuredAction::Stop { .. } | StructuredAction::AttackUnit { .. }
    ));
}

#[test]
fn finish_one_auto_budgets_a_possible_tower_windup_beyond_acquisition_range() {
    let mut view = one_auto_view();
    own_hero_mut(&mut view).pos = Vec2::from_ints(5_190, 6_000);
    enemy_hero_mut(&mut view).pos = Vec2::from_ints(5_430, 6_000);
    let tower = view
        .units
        .iter_mut()
        .find(|unit| unit.id == entity(12, 1))
        .expect("tower");
    tower.radius = Fixed::from_int(40);
    tower.attack_range = Fixed::from_int(700);
    tower.attack_damage = 110;
    tower.attack_interval = 29;
    tower.move_speed = Fixed::ZERO;
    let tracker = tracker(view);

    let (action, _) = decide(&tracker);

    assert!(matches!(action, StructuredAction::MovePoint { .. }));
}

#[test]
fn attack_protection_keeps_the_full_fifteen_tick_windup_at_high_attack_speed() {
    let mut view = tactical_view(800, 100);
    view.tick = 16;
    own_hero_mut(&mut view).attack_speed = 300;
    let mut creep = unit(CREEP_ID, UnitKind::CreepMelee, Team::Dire, 3_600, 3_000);
    creep.hp = 1_000;
    view.units.push(creep);
    sort_units(&mut view);
    let tracker = tracker(view);
    let mut teacher = Teacher::new();
    let mut persistence = OrderPersistence::default();
    record_combat_order(
        &mut teacher,
        &mut persistence,
        Order::Attack {
            target: Target::Unit(CREEP_ID),
        },
        1,
    );

    let (action, _) = teacher
        .decide_tactical(
            &tracker,
            &persistence,
            &ItemReadiness::new(),
            &tactical_policy(crate::TacticalMode::Farm),
        )
        .expect("windup");

    assert_eq!(action, StructuredAction::Continue);
}

fn one_auto_view() -> WorldView {
    let mut view = tactical_view(240, 20);
    for body in [
        own_hero_mut as fn(&mut WorldView) -> &mut UnitView,
        enemy_hero_mut,
    ] {
        let hero = body(&mut view);
        hero.max_hp = 516;
        hero.attack_damage = 45;
        hero.attack_interval = 42;
        hero.attack_speed = 120;
        hero.armor = Fixed::from_ratio(20, 6);
        hero.magic_resist = Fixed::from_ratio(25, 100);
    }
    own_hero_mut(&mut view).hp = 80;
    view
}

fn enemy_hero_mut(view: &mut WorldView) -> &mut UnitView {
    view.units
        .iter_mut()
        .find(|unit| unit.id == ENEMY_HERO_ID)
        .expect("enemy hero")
}

fn tactical_view(distance: i32, hp: i32) -> WorldView {
    let mut view = base_view();
    let mut enemy = unit(
        ENEMY_HERO_ID,
        UnitKind::Hero,
        Team::Dire,
        3_000 + distance,
        3_000,
    );
    enemy.hp = hp;
    view.units.push(enemy);
    view.players[1].unit = Some(ENEMY_HERO_ID);
    sort_units(&mut view);
    view
}

fn decide(tracker: &StateTracker) -> (StructuredAction, crate::ActionSpace) {
    Teacher::new()
        .decide(tracker, &OrderPersistence::default(), &ItemReadiness::new())
        .expect("teacher decision")
}

fn tracker(view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

fn match_info() -> MatchInfo {
    MatchInfo {
        match_id: 1,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 900,
        trees: vec![Vec2::from_ints(3_100, 3_000)],
        terrain_cells: 128,
        terrain_rle: vec![(16_384, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: SHADOW_FIEND,
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: SHADOW_FIEND,
            },
        ],
        shop: [
            (0, 500),
            (1, 50),
            (2, 110),
            (7, 90),
            (8, 100),
            (13, 450),
            (19, 450),
            (24, 550),
            (25, 175),
            (26, 175),
            (29, 1_400),
            (33, 505),
            (35, 200),
            (36, 450),
        ]
        .into_iter()
        .map(|(id, cost)| ShopEntry {
            id: ItemId(id),
            cost,
            components: Vec::new(),
        })
        .collect(),
    }
}

fn base_view() -> WorldView {
    let mut hero = unit(HERO_ID, UnitKind::Hero, Team::Radiant, 3_000, 3_000);
    hero.hero = Some(SHADOW_FIEND);
    hero.owner = Some(SlotId(0));
    hero.mana = 0;
    hero.max_mana = 500;
    hero.attack_damage = 60;
    hero.abilities = shadow_fiend_abilities();
    hero.items = vec![None; 9];
    let mut courier = unit(COURIER_ID, UnitKind::Courier, Team::Radiant, 1_200, 1_100);
    courier.owner = Some(SlotId(0));
    courier.attack_damage = 0;
    courier.abilities = courier_abilities();
    courier.items = vec![None; 6];
    let mut view = WorldView {
        tick: 1,
        viewer: Some(Team::Radiant),
        units: vec![
            hero,
            courier,
            unit(
                entity(10, 1),
                UnitKind::Fountain,
                Team::Radiant,
                1_000,
                1_000,
            ),
            unit(entity(11, 1), UnitKind::Tower, Team::Radiant, 2_000, 2_000),
            unit(entity(12, 1), UnitKind::Tower, Team::Dire, 6_000, 6_000),
            unit(entity(13, 1), UnitKind::Ancient, Team::Dire, 7_000, 7_000),
        ],
        projectiles: Vec::new(),
        players: vec![own_player(), enemy_player()],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    };
    sort_units(&mut view);
    view
}

fn shadow_fiend_abilities() -> Vec<AbilityView> {
    vec![
        ability(13, 1, 75, false),
        ability(14, 1, 75, false),
        ability(15, 1, 75, false),
        ability(17, 1, 0, true),
        ability(18, 1, 0, true),
        ability(16, 1, 150, false),
    ]
}

fn courier_abilities() -> Vec<AbilityView> {
    [10, 9, 8, 11, 12]
        .into_iter()
        .map(|id| ability(id, 1, 0, false))
        .collect()
}

fn ability(id: u16, level: u8, mana_cost: i32, passive: bool) -> AbilityView {
    AbilityView {
        id: AbilityId(id),
        level,
        max_level: if id == 16 { 3 } else { 4 },
        cooldown_left: 0,
        mana_cost,
        range: 0,
        aim: Aim::Own,
        passive,
        on: false,
        can_level: false,
    }
}

fn own_player() -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team: Team::Radiant,
        hero: SHADOW_FIEND,
        unit: Some(HERO_ID),
        level: 6,
        xp: 0,
        gold: Some(0),
        stash: Some(vec![None; 6]),
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

fn enemy_player() -> PlayerView {
    PlayerView {
        slot: SlotId(1),
        team: Team::Dire,
        hero: SHADOW_FIEND,
        unit: None,
        level: 1,
        xp: 0,
        gold: None,
        stash: None,
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

fn unit(id: EntityId, kind: UnitKind, team: Team, x: i32, y: i32) -> UnitView {
    UnitView {
        id,
        kind,
        team,
        pos: Vec2::from_ints(x, y),
        facing: Angle { brads: 0 },
        hp: 1_000,
        max_hp: 1_000,
        mana: 0,
        max_mana: 0,
        move_speed: Fixed::from_int(300),
        attack_damage: 50,
        attack_range: Fixed::from_int(500),
        attack_interval: 51,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(24),
        vision_radius: Fixed::from_int(1_800),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::all(20),
        primary: Some(Attribute::Agility),
        hero: (kind == UnitKind::Hero).then_some(HeroId(2)),
        owner: None,
        level: 0,
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}

fn item(id: ItemId, aim: Option<Aim>, range: i32, charges: Option<u8>) -> ItemView {
    ItemView {
        id,
        charges,
        cooldown_left: 0,
        mute_left: 0,
        mode: None,
        mana_cost: 0,
        range,
        aim,
        for_sale: false,
    }
}

fn cast(unit: ControlledUnit, slot: u8) -> StructuredAction {
    StructuredAction::Cast {
        unit,
        slot: AbilitySlot(slot),
        target: ActionTarget::None,
    }
}

fn own_hero_mut(view: &mut WorldView) -> &mut UnitView {
    view.units
        .iter_mut()
        .find(|unit| unit.id == HERO_ID)
        .expect("own hero")
}

fn own_courier_mut(view: &mut WorldView) -> &mut UnitView {
    view.units
        .iter_mut()
        .find(|unit| unit.id == COURIER_ID)
        .expect("own courier")
}

fn sort_units(view: &mut WorldView) {
    view.units.sort_by_key(|unit| unit.id);
}

const fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}
