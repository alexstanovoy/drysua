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
fn teacher_buys_starting_sustain_before_equipment() {
    let mut view = base_view();
    view.players[0].gold = Some(600);
    let tracker = tracker(view);

    let (action, space) = decide(&tracker);

    assert!(
        matches!(action, StructuredAction::Buy { unit: ControlledUnit::Hero, item } if space.shop_candidates()[item.0].item == ItemId(7))
    );
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
        matches!(retried, StructuredAction::Buy { item, .. } if space.shop_candidates()[item.0].item == ItemId(7))
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
