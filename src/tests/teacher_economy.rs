use bota_proto::{
    AbilityId, EffectId, EffectView, EntityId, EventKind, Fixed, ItemId, ItemSlot, Order,
    ServerMsg, SlotId, Team, UnitKind, Vec2, WorldView,
};

use crate::teacher_economy::{
    ECONOMY_PLAN, EconomyObservation, attack_damage_against, select_purchase, select_sustain,
};
use crate::{
    ActionSpace, Arena, ArenaConfig, ControlledUnit, ItemReadiness, OrderPersistence, StateTracker,
    StructuredAction, Teacher,
};

#[test]
fn stick_ignores_learn_event_while_existing_cooldown_decreases_from_300_to_297() {
    let (mut teacher, mut tracker, mut view, enemy) = stick_history(Some(300));
    view.units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy")
        .abilities[0]
        .level = 2;
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 5, 297, Some(AbilityId(13)));

    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(19));
}

#[test]
fn stick_accepts_matching_cast_when_observed_cooldown_rises_from_zero_to_300() {
    let (mut teacher, mut tracker, mut view, enemy) = stick_history(Some(0));
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 4, 300, Some(AbilityId(13)));

    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(35));
}

#[test]
fn stick_rejects_first_sight_cooling_even_with_an_ability_cast_event() {
    let (mut teacher, mut tracker, mut view, enemy) = stick_history(None);
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 4, 300, Some(AbilityId(13)));

    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(19));
}

#[test]
fn stick_keeps_confirmed_cast_for_300_ticks_without_refreshing_on_later_learn() {
    let (mut teacher, mut tracker, mut view, enemy) = stick_history(Some(0));
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 4, 300, Some(AbilityId(13)));
    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(35));
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 7, 297, Some(AbilityId(13)));
    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(35));
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 304, 0, None);
    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(35));
    advance_enemy_cooldown(&mut tracker, &mut view, enemy, 305, 0, None);
    assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(19));
}

#[test]
fn stick_requires_a_matching_event_not_just_a_cooldown_increase() {
    for event in [None, Some(AbilityId(14))] {
        let (mut teacher, mut tracker, mut view, enemy) = stick_history(Some(0));
        advance_enemy_cooldown(&mut tracker, &mut view, enemy, 4, 300, event);

        assert_eq!(teacher_purchase(&mut teacher, &tracker), ItemId(19));
    }
}

fn stick_history(prior: Option<u32>) -> (Teacher, StateTracker, WorldView, EntityId) {
    let (_, info, mut view) = fixture();
    for (slot, item) in [33, 7, 0].into_iter().enumerate() {
        equip(&mut view, slot, item, None);
    }
    for ability in &mut hero_mut(&mut view).abilities {
        ability.can_level = false;
    }
    view.players[0].gold = Some(0);
    let mut enemy = own_hero(&view).clone();
    enemy.id = EntityId {
        idx: 1_000,
        generation: 1,
    };
    enemy.team = Team::Dire;
    enemy.owner = Some(SlotId(1));
    enemy.pos.x += Fixed::from_int(500);
    enemy.abilities[0].level = 1;
    enemy.abilities[0].cooldown_left = prior.unwrap_or(0);
    let id = enemy.id;
    let mut teacher = Teacher::new();
    let mut tracker = track(&info, view.clone());
    if prior.is_some() {
        view.tick = 2;
        view.units.push(enemy.clone());
        view.units.sort_by_key(|unit| unit.id);
        tracker.observe_snapshot(&view).expect("prior cooldown");
    }
    teacher
        .decide(
            &tracker,
            &OrderPersistence::default(),
            &ItemReadiness::new(),
        )
        .expect("baseline decision");
    if prior.is_none() {
        view.units.push(enemy);
        view.units.sort_by_key(|unit| unit.id);
    }
    (teacher, tracker, view, id)
}

fn advance_enemy_cooldown(
    tracker: &mut StateTracker,
    view: &mut WorldView,
    enemy: EntityId,
    tick: u32,
    cooldown: u32,
    event: Option<AbilityId>,
) {
    view.tick = tick;
    view.players[0].gold = Some(450);
    view.units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy")
        .abilities[0]
        .cooldown_left = cooldown;
    tracker.observe_snapshot(view).expect("current cooldown");
    let events: Vec<_> = event
        .into_iter()
        .map(|ability| EventKind::AbilityCast {
            caster: enemy,
            ability,
        })
        .collect();
    tracker
        .observe_events(tick, &events)
        .expect("complete event batch");
}

fn teacher_purchase(teacher: &mut Teacher, tracker: &StateTracker) -> ItemId {
    let (action, space) = teacher
        .decide(tracker, &OrderPersistence::default(), &ItemReadiness::new())
        .expect("purchase decision");
    let Order::Buy { item } = space.decode(action).expect("decode").expect("order").order else {
        panic!("optional Stick or next component must be selected");
    };
    item
}

#[test]
fn opening_buys_wraith_then_tango_for_595_without_mandatory_ring_or_wand() {
    let mut arena = fixture();
    let mut bought = [false; ECONOMY_PLAN.len()];
    for (sequence, expected) in [(1, ItemId(33)), (2, ItemId(7))] {
        let tracker = track(&arena.1, arena.2.clone());
        let space = ActionSpace::from_tracker(&tracker).expect("opening space");
        let action = select_purchase(&tracker, &space, &bought, &EconomyObservation::new())
            .expect("opening purchase");
        let issued = space.decode(action).expect("decode").expect("buy");
        assert_eq!(issued.order, Order::Buy { item: expected });
        bought[sequence - 1] = true;
        let step = arena
            .0
            .step(&[
                Some(crate::Request {
                    seq: sequence as u32,
                    unit: issued.unit,
                    order: issued.order,
                }),
                None,
            ])
            .expect("buy step");
        assert!(
            !step.messages[0]
                .iter()
                .any(|message| matches!(message, ServerMsg::OrderRejected { .. }))
        );
        arena.2 = snapshot(&step.messages[0]);
    }
    assert_eq!(arena.2.players[0].gold, Some(5));
    assert!(!ECONOMY_PLAN.contains(&ItemId(25)));
    assert!(!ECONOMY_PLAN.contains(&ItemId(36)));
}

#[test]
fn opening_waits_for_wraith_budget_and_does_not_buy_tango_first() {
    let (_, info, mut view) = fixture();
    view.players[0].gold = Some(504);
    let tracker = track(&info, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");

    assert_eq!(
        select_purchase(
            &tracker,
            &space,
            &[false; ECONOMY_PLAN.len()],
            &EconomyObservation::new()
        ),
        None
    );
    assert!(space.buy_mask(ControlledUnit::Hero)[7]);
}

#[test]
fn consumed_opening_tango_is_not_rebought_and_boots_are_next() {
    let (_, info, view) = fixture();
    let tracker = track(&info, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut bought = [false; ECONOMY_PLAN.len()];
    bought[..2].fill(true);

    let action =
        select_purchase(&tracker, &space, &bought, &EconomyObservation::new()).expect("boots");
    assert_eq!(
        space.decode(action).expect("decode").expect("buy").order,
        Order::Buy { item: ItemId(0) }
    );
}

#[test]
fn optional_stick_requires_a_recent_visible_enemy_cast_not_a_skill_learn_event() {
    for (cast, cooldown, distance, expected) in [
        (false, 0, 500, ItemId(19)),
        (true, 0, 500, ItemId(19)),
        (true, 290, 1_200, ItemId(35)),
        (true, 290, 1_201, ItemId(19)),
    ] {
        let (_, info, mut view) = fixture();
        let hero = own_hero(&view).id;
        let mut enemy = own_hero(&view).clone();
        enemy.id = EntityId {
            idx: 1_000,
            generation: 1,
        };
        enemy.team = Team::Dire;
        enemy.owner = Some(SlotId(1));
        enemy.pos = own_hero(&view).pos + Vec2::from_ints(distance, 0);
        enemy.abilities[0].level = 1;
        enemy.abilities[0].cooldown_left = 0;
        let caster = enemy.id;
        view.units.push(enemy);
        view.units.sort_by_key(|unit| unit.id);
        let mut tracker = track(&info, view.clone());
        let mut observation = EconomyObservation::new();
        observation.observe(&tracker);
        view.tick = 4;
        view.units
            .iter_mut()
            .find(|unit| unit.id == caster)
            .expect("enemy")
            .abilities[0]
            .cooldown_left = cooldown;
        tracker
            .observe_snapshot(&view)
            .expect("cooldown transition");
        if cast {
            tracker
                .observe_events(
                    4,
                    &[EventKind::AbilityCast {
                        caster,
                        ability: AbilityId(13),
                    }],
                )
                .expect("cast");
        }
        let mut bought = [false; ECONOMY_PLAN.len()];
        bought[..3].fill(true);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        observation.observe(&tracker);
        let action = select_purchase(&tracker, &space, &bought, &observation).expect("follow-up");
        assert_eq!(
            space.decode(action).expect("decode").expect("buy").order,
            Order::Buy { item: expected }
        );
        assert_eq!(tracker.own_hero().expect("hero").id, hero);
    }
}

#[test]
fn treads_and_wand_satisfy_consumed_components_without_duplicate_purchases() {
    let (_, info, mut view) = fixture();
    equip(&mut view, 0, 29, None);
    equip(&mut view, 1, 36, Some(0));
    let tracker = track(&info, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut bought = [false; ECONOMY_PLAN.len()];
    bought[..2].fill(true);

    assert_eq!(
        select_purchase(&tracker, &space, &bought, &EconomyObservation::new()),
        None
    );
}

#[test]
fn tango_does_not_overwrite_tango_or_salve_mending() {
    for active in [false, true] {
        let (_, mut info, mut view) = fixture();
        let position = own_hero(&view).pos;
        info.trees = vec![position + Vec2::from_ints(100, 0)];
        hero_mut(&mut view).hp -= 115;
        equip(&mut view, 0, 7, Some(3));
        if active {
            hero_mut(&mut view).effects.push(EffectView {
                id: EffectId(1),
                ticks_left: Some(300),
                stacks: None,
            });
        }
        let tracker = track(&info, view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        let action = select_sustain(&tracker, &space, false);
        assert_eq!(action.is_some(), !active);
        assert!(action.is_none_or(|action| space.allows(action)));
    }
}

#[test]
fn tango_waits_for_missing_health_and_a_tree_inside_165() {
    for (missing, distance, uses) in [(114, 100, false), (115, 165, true), (115, 166, false)] {
        let (_, mut info, mut view) = fixture();
        info.trees = vec![own_hero(&view).pos + Vec2::from_ints(distance, 0)];
        hero_mut(&mut view).hp -= missing;
        equip(&mut view, 0, 7, Some(3));
        let tracker = track(&info, view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        let action = select_sustain(&tracker, &space, false);
        assert_eq!(action.is_some(), uses);
        assert!(action.is_none_or(|action| space.allows(action)));
    }
}

#[test]
fn salve_and_clarity_ignore_creep_damage_but_wait_after_hero_or_tower_damage() {
    for item in [1, 2] {
        for (kind, uses) in [
            (UnitKind::CreepRanged, true),
            (UnitKind::Hero, false),
            (UnitKind::Tower, false),
        ] {
            let (_, info, mut view) = fixture();
            hero_mut(&mut view).hp -= 300;
            hero_mut(&mut view).mana -= 150;
            equip(&mut view, 0, item, Some(1));
            let source = add_source(&mut view, kind, 800);
            let target = own_hero(&view).id;
            let mut tracker = track(&info, view);
            tracker
                .observe_events(1, &[damage(source, target)])
                .expect("damage");
            let space = ActionSpace::from_tracker(&tracker).expect("space");

            let action = select_sustain(&tracker, &space, false);
            assert_eq!(action.is_some(), uses, "item {item}, source {kind:?}");
            assert!(action.is_none_or(|action| space.allows(action)));
        }
    }
}

#[test]
fn later_creep_damage_does_not_hide_a_recent_hero_hit_from_salve() {
    let (_, info, mut view) = fixture();
    hero_mut(&mut view).hp -= 300;
    equip(&mut view, 0, 2, Some(1));
    let hero_source = add_source(&mut view, UnitKind::Hero, 800);
    let creep_source = add_source(&mut view, UnitKind::CreepRanged, 900);
    let target = own_hero(&view).id;
    let mut tracker = track(&info, view);
    tracker
        .observe_events(
            1,
            &[damage(hero_source, target), damage(creep_source, target)],
        )
        .expect("damage");
    let space = ActionSpace::from_tracker(&tracker).expect("space");

    assert_eq!(select_sustain(&tracker, &space, false), None);
}

#[test]
fn salve_is_available_above_old_retreat_threshold_and_creeps_do_not_block_it() {
    let (_, info, mut view) = fixture();
    hero_mut(&mut view).max_hp = 1_000;
    hero_mut(&mut view).hp = 700;
    equip(&mut view, 0, 2, Some(1));
    add_source(&mut view, UnitKind::CreepRanged, 300);
    let tracker = track(&info, view);
    let space = ActionSpace::from_tracker(&tracker).expect("space");

    let action = select_sustain(&tracker, &space, false).expect("salve");
    assert!(space.allows(action));
    assert!(matches!(
        action,
        StructuredAction::Use {
            slot: ItemSlot(0),
            ..
        }
    ));
}

#[test]
fn health_drinks_and_clarity_wait_for_their_own_active_effect_to_finish() {
    for (item, effect, uses) in [(2, 1, false), (1, 2, false), (1, 1, true)] {
        let (_, info, mut view) = fixture();
        hero_mut(&mut view).hp -= 300;
        hero_mut(&mut view).mana -= 150;
        equip(&mut view, 0, item, Some(1));
        hero_mut(&mut view).effects.push(EffectView {
            id: EffectId(effect),
            ticks_left: Some(1),
            stacks: None,
        });
        let tracker = track(&info, view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        assert_eq!(select_sustain(&tracker, &space, false).is_some(), uses);
    }
}

#[test]
fn salve_resumes_after_90_ticks_without_hero_damage_and_rejects_unknown_attackers() {
    for (age, known, uses) in [(90, true, false), (91, true, true), (0, false, false)] {
        let (_, info, mut view) = fixture();
        hero_mut(&mut view).hp -= 300;
        equip(&mut view, 0, 2, Some(1));
        let source = if known {
            add_source(&mut view, UnitKind::Hero, 800)
        } else {
            EntityId {
                idx: 3_000,
                generation: 1,
            }
        };
        let target = own_hero(&view).id;
        let mut tracker = track(&info, view.clone());
        tracker
            .observe_events(1, &[damage(source, target)])
            .expect("damage");
        if age > 0 {
            view.tick += age;
            tracker.observe_snapshot(&view).expect("later snapshot");
        }
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        assert_eq!(select_sustain(&tracker, &space, false).is_some(), uses);
    }
}

#[test]
fn salve_respects_visible_hero_buffer_and_full_tower_attack_reach() {
    for (kind, distance, uses) in [
        (UnitKind::Hero, 700, false),
        (UnitKind::Hero, 701, true),
        (UnitKind::Tower, 748, false),
        (UnitKind::Tower, 749, true),
    ] {
        let (_, info, mut view) = fixture();
        hero_mut(&mut view).hp -= 300;
        equip(&mut view, 0, 2, Some(1));
        let source = add_source(&mut view, kind, distance);
        if kind == UnitKind::Tower {
            view.units
                .iter_mut()
                .find(|unit| unit.id == source)
                .expect("tower")
                .attack_range = Fixed::from_int(700);
        }
        let tracker = track(&info, view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        assert_eq!(select_sustain(&tracker, &space, false).is_some(), uses);
    }
}

#[test]
fn one_charge_stick_or_wand_is_used_in_emergency_but_not_wasted_at_full_pools() {
    for item in [35, 36] {
        for (charges, missing, emergency, uses) in [
            (0, 300, true, false),
            (1, 300, true, true),
            (1, 0, true, false),
            (1, 100, false, false),
        ] {
            let (_, info, mut view) = fixture();
            hero_mut(&mut view).hp -= missing;
            equip(&mut view, 0, item, Some(charges));
            let tracker = track(&info, view);
            let space = ActionSpace::from_tracker(&tracker).expect("space");

            let action = select_sustain(&tracker, &space, emergency);
            assert_eq!(action.is_some(), uses);
            assert!(action.is_none_or(|action| space.allows(action)));
        }
    }
}

#[test]
fn sustain_respects_cooldown_mute_backpack_and_channel_masks() {
    for scenario in 0..4 {
        let (_, info, mut view) = fixture();
        hero_mut(&mut view).hp -= 300;
        equip(&mut view, if scenario == 2 { 6 } else { 0 }, 36, Some(10));
        match scenario {
            0 => {
                hero_mut(&mut view).items[0]
                    .as_mut()
                    .expect("wand")
                    .cooldown_left = 1
            }
            1 => {
                hero_mut(&mut view).items[0]
                    .as_mut()
                    .expect("wand")
                    .mute_left = 1
            }
            2 => {}
            _ => hero_mut(&mut view).statuses.bits = bota_proto::StatusFlags::CHANNELLING,
        }
        let tracker = track(&info, view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");

        assert_eq!(select_sustain(&tracker, &space, true), None);
    }
}

#[test]
fn quelling_adds_18_for_every_creep_kind_including_denies_but_not_buildings_or_heroes() {
    let (_, _, mut view) = fixture();
    equip(&mut view, 0, 5, None);
    let source = own_hero(&view);
    for kind in [
        UnitKind::CreepMelee,
        UnitKind::CreepFlagbearer,
        UnitKind::CreepRanged,
        UnitKind::CreepSiege,
        UnitKind::CreepNeutral,
        UnitKind::Hero,
        UnitKind::Tower,
        UnitKind::Ancient,
        UnitKind::Barracks,
        UnitKind::Courier,
    ] {
        let mut target = source.clone();
        target.kind = kind;
        let creep = bota_server::game::is_creep(kind);
        for team in [Team::Radiant, Team::Dire, Team::Neutral] {
            target.team = team;
            assert_eq!(
                attack_damage_against(source, &target),
                source.attack_damage + if creep { 18 } else { 0 }
            );
        }
    }
}

#[test]
fn quelling_stacks_only_unmuted_active_blades_and_ignores_cooldown() {
    let (_, _, mut view) = fixture();
    for slot in [0, 1, 2, 6] {
        equip(&mut view, slot, 5, None);
    }
    hero_mut(&mut view).items[0]
        .as_mut()
        .expect("blade")
        .cooldown_left = 120;
    hero_mut(&mut view).items[2]
        .as_mut()
        .expect("blade")
        .mute_left = 1;
    let source = own_hero(&view);
    let mut target = source.clone();
    target.kind = UnitKind::CreepRanged;

    assert_eq!(
        attack_damage_against(source, &target),
        source.attack_damage + 36
    );
    assert_eq!(source.attack_damage, 45);
}

#[test]
fn quelling_prediction_matches_catalog_bonus_and_saturates_extreme_damage() {
    let (_, _, mut view) = fixture();
    equip(&mut view, 0, 5, None);
    let mut target = own_hero(&view).clone();
    target.kind = UnitKind::CreepRanged;
    let bonus = bota_server::game::item_def(ItemId(5))
        .expect("Quelling")
        .carried
        .damage_to_creeps;
    assert_eq!(bonus, 18);
    let source = hero_mut(&mut view);
    assert_eq!(
        attack_damage_against(source, &target),
        source.attack_damage + bonus
    );

    source.attack_damage = i32::MAX;
    assert_eq!(attack_damage_against(source, &target), i32::MAX);
}

fn fixture() -> (Arena, bota_proto::MatchInfo, WorldView) {
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: bota_proto::MapId(1),
        seed: 92_009_003,
    })
    .expect("arena");
    let info = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info.clone()),
            _ => None,
        })
        .expect("match info");
    let view = snapshot(&start.messages[0]);
    assert_eq!(view.tick, 1);
    (arena, info, view)
}

fn snapshot(messages: &[ServerMsg]) -> WorldView {
    messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("snapshot")
}

fn track(info: &bota_proto::MatchInfo, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

fn own_hero(view: &WorldView) -> &bota_proto::UnitView {
    view.units
        .iter()
        .find(|unit| unit.id == view.players[0].unit.expect("hero"))
        .expect("own hero")
}

fn hero_mut(view: &mut WorldView) -> &mut bota_proto::UnitView {
    let id = view.players[0].unit.expect("hero");
    view.units
        .iter_mut()
        .find(|unit| unit.id == id)
        .expect("own hero")
}

fn equip(view: &mut WorldView, slot: usize, id: u16, charges: Option<u8>) {
    let def = bota_server::game::item_def(ItemId(id)).expect("catalog item");
    hero_mut(view).items[slot] = Some(bota_proto::ItemView {
        id: ItemId(id),
        charges,
        cooldown_left: 0,
        mute_left: 0,
        mode: def.mode,
        mana_cost: def.mana_cost,
        range: def.active.map_or(0, bota_server::game::item_range),
        aim: def.active.map(bota_server::game::item_aim),
        for_sale: false,
    });
}

fn add_source(view: &mut WorldView, kind: UnitKind, distance: i32) -> EntityId {
    let mut source = own_hero(view).clone();
    source.id = EntityId {
        idx: 1_000 + view.units.len() as u32,
        generation: 1,
    };
    source.kind = kind;
    source.team = Team::Dire;
    source.owner = None;
    source.pos.x += Fixed::from_int(distance);
    source.attack_range = Fixed::from_int(500);
    source.radius = Fixed::from_int(24);
    source.abilities.clear();
    let id = source.id;
    view.units.push(source);
    view.units.sort_by_key(|unit| unit.id);
    id
}

fn damage(source: EntityId, target: EntityId) -> EventKind {
    EventKind::Damaged {
        source: Some(source),
        target,
        amount: 10,
        kind: bota_proto::DamageKind::Physical,
        crit: false,
    }
}
