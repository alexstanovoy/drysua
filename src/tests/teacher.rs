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
fn teacher_buys_wand_before_unused_consumables() {
    let mut view = base_view();
    view.players[0].gold = Some(600);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(
        matches!(action, StructuredAction::Buy { unit: ControlledUnit::Hero, item } if space.shop_candidates()[item.0].item == ItemId(36))
    );
    assert!(space.allows(action));
}

#[test]
fn teacher_builds_six_active_combat_items_in_order_without_duplicates() {
    let plan = [36, 25, 26, 33, 29, 24];
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
fn teacher_does_not_skip_unaffordable_wand_for_unused_consumables() {
    let mut view = base_view();
    view.players[0].gold = Some(449);
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
        matches!(retried, StructuredAction::Buy { item, .. } if space.shop_candidates()[item.0].item == ItemId(36))
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

        let (action, space) = decide(&tracker);

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
fn teacher_ready_hero_raze_preempts_a_persistent_attack() {
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
    let tracker = tracker(view);
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

    assert_eq!(action, cast(ControlledUnit::Hero, 0));
    assert!(space.allows(action));
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

        let (action, space) = decide(&tracker);

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

    let (action, space) = decide(&tracker);

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
fn teacher_continues_an_old_attack_inside_server_windup_leeway() {
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

    assert_eq!(action, StructuredAction::Continue);
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

    assert!(matches!(baseline, StructuredAction::Cast { .. }));
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
    assert!(matches!(action, StructuredAction::AttackMovePoint { .. }));
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

fn tactical_policy(mode: crate::TacticalMode) -> crate::TacticalPolicy {
    let mut parameters = [0.0; crate::TACTICAL_PARAMETERS];
    parameters[crate::TACTICAL_OUTPUT_BIAS_OFFSET + mode.index()] = 1.0;
    crate::TacticalPolicy::from_parameters(&parameters).expect("forced macro")
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
            (1, 50),
            (2, 110),
            (7, 90),
            (8, 100),
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
