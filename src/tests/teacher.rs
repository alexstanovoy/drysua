use bota_proto::{
    AbilityId, AbilitySlot, AbilityView, Aim, EntityId, Fixed, MapId, MatchInfo, Order, SlotId,
    StatusFlags, Target, Team, UnitKind, UnitView, Vec2, WorldView,
};

use super::fixtures;
use crate::{
    ActionSpace, IssuedOrder, ItemReadiness, OrderPersistence, StateTracker, StructuredAction,
    Teacher,
};

const HERO: EntityId = EntityId {
    idx: 1,
    generation: 1,
};
const ENEMY: EntityId = EntityId {
    idx: 3,
    generation: 1,
};

#[test]
fn public_decision_requires_snapshot_and_preserves_channels() {
    let mut tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    assert_eq!(
        Teacher::new()
            .decide(
                &tracker,
                &OrderPersistence::default(),
                &ItemReadiness::new()
            )
            .err()
            .expect("snapshot required")
            .to_string(),
        "action space requires a validated snapshot"
    );
    let mut view = base_view();
    body(&mut view, HERO).hp = 100;
    body(&mut view, HERO).statuses.bits = StatusFlags::CHANNELLING;
    tracker.observe_snapshot(&view).expect("channel snapshot");
    let (action, space) = decide(&mut Teacher::new(), &tracker);
    assert_eq!(action, StructuredAction::Continue);
    assert_eq!(space.decode(action).expect("continue"), None);
}

#[test]
fn public_cast_contract_requires_mana_and_cooldown_readiness() {
    for (mana, cooldown, casts) in [(75, 0, true), (74, 0, false), (75, 1, false)] {
        let mut view = base_view();
        let hero = body(&mut view, HERO);
        hero.pos = Vec2::from_ints(5_100, 6_000);
        hero.mana = mana;
        hero.abilities[2].cooldown_left = cooldown;
        let mut teacher = Teacher::new();
        teacher.note_sent(
            1,
            IssuedOrder {
                unit: None,
                order: Order::Move {
                    target: Target::None,
                },
            },
            0,
        );
        let (action, space) = decide(&mut teacher, &track(view));
        assert_eq!(matches!(action, StructuredAction::Cast { .. }), casts);
        if casts {
            assert_eq!(
                wire(&space, action),
                Order::Cast {
                    slot: AbilitySlot(2),
                    target: Target::None
                }
            );
        }
    }
}

#[test]
fn tower_safety_cancels_retained_chase_but_allows_escape() {
    for escape in [false, true] {
        let mut view = base_view();
        body(&mut view, HERO).pos = Vec2::from_ints(5_300, 5_700);
        view.tick = 4;
        let target = if escape {
            Vec2::from_ints(4_500, 5_000)
        } else {
            Vec2::from_ints(6_400, 5_300)
        };
        let mut teacher = Teacher::new();
        teacher.note_sent(
            1,
            IssuedOrder {
                unit: None,
                order: Order::Move {
                    target: Target::Pos(target),
                },
            },
            1,
        );
        let tracker = track(view);
        let space = ActionSpace::from_tracker(&tracker).expect("space");
        let action = teacher.safety_action(&tracker, &space);
        if escape {
            assert!(action.is_none());
        } else {
            assert_eq!(
                wire(&space, action.expect("cancel chase")),
                Order::Move {
                    target: Target::None
                }
            );
        }
    }
}

#[test]
fn respawn_generation_does_not_inherit_an_expired_finish_attempt() {
    let mut view = base_view();
    let mut enemy = unit(ENEMY, UnitKind::Hero, Team::Dire, 3_240, 3_000);
    enemy.hp = 20;
    view.units.push(enemy);
    view.units.sort_by_key(|unit| unit.id);
    view.players[1].unit = Some(ENEMY);
    for id in [HERO, ENEMY] {
        let hero = body(&mut view, id);
        hero.max_hp = 516;
        hero.attack_damage = 45;
        hero.attack_time = 1400;
        hero.attack_speed = 120;
        hero.armor = Fixed::from_ratio(20, 6);
        hero.magic_resist = Fixed::from_ratio(25, 100);
    }
    body(&mut view, HERO).hp = 80;
    let mut tracker = track(view.clone());
    let mut teacher = Teacher::new();
    let (action, space) = decide(&mut teacher, &tracker);
    assert_eq!(
        wire(&space, action),
        Order::Attack {
            target: Target::Unit(ENEMY)
        }
    );
    teacher.note_sent(1, space.decode(action).expect("decode").expect("finish"), 1);
    let mut dead = view.clone();
    dead.tick = 64;
    dead.players[0].unit = None;
    dead.players[0].deaths = 1;
    dead.players[0].respawn_left = 1;
    dead.units.retain(|unit| unit.id != HERO);
    tracker.observe_snapshot(&dead).expect("death");
    assert_eq!(decide(&mut teacher, &tracker).0, StructuredAction::Continue);
    let replacement = EntityId {
        generation: 2,
        ..HERO
    };
    body(&mut view, HERO).id = replacement;
    view.players[0].unit = Some(replacement);
    view.players[0].deaths = 1;
    view.tick = 65;
    view.units.sort_by_key(|unit| unit.id);
    tracker.observe_snapshot(&view).expect("respawn");
    let (action, space) = decide(&mut teacher, &tracker);
    assert_eq!(
        wire(&space, action),
        Order::Attack {
            target: Target::Unit(ENEMY)
        }
    );
}

fn decide(teacher: &mut Teacher, tracker: &StateTracker) -> (StructuredAction, ActionSpace) {
    let (action, space) = teacher
        .decide(tracker, &OrderPersistence::default(), &ItemReadiness::new())
        .expect("decision");
    assert!(space.allows(action), "{action:?}");
    assert!(space.decode(action).is_ok());
    (action, space)
}

fn wire(space: &ActionSpace, action: StructuredAction) -> Order {
    assert!(space.allows(action));
    space
        .decode(action)
        .expect("decode")
        .expect("wire order")
        .order
}

fn track(mut view: WorldView) -> StateTracker {
    view.units.sort_by_key(|unit| unit.id);
    let mut tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(1, MapId(0), fixtures::two_seat_picks(Team::Radiant))
        .pregame_ticks(900)
        .terrain_cells(128)
        .terrain_rle(vec![(16_384, 0x80)])
        .build()
}

fn base_view() -> WorldView {
    let mut hero = unit(HERO, UnitKind::Hero, Team::Radiant, 3_000, 3_000);
    hero.owner = Some(SlotId(0));
    hero.max_mana = 500;
    hero.attack_damage = 60;
    hero.abilities = [13, 14, 15, 17, 18, 16]
        .map(|id| AbilityView {
            id: AbilityId(id),
            level: 1,
            max_level: if id == 16 { 3 } else { 4 },
            cooldown_left: 0,
            mana_cost: if id == 16 { 150 } else { 75 },
            range: 0,
            aim: Aim::Own,
            passive: matches!(id, 17 | 18),
            on: false,
            can_level: false,
        })
        .to_vec();
    hero.items = vec![None; 9];
    let mut units = vec![hero];
    for (idx, kind, team, coordinate) in [
        (10, UnitKind::Fountain, Team::Radiant, 1_000),
        (12, UnitKind::Tower, Team::Dire, 6_000),
        (13, UnitKind::Ancient, Team::Dire, 7_000),
    ] {
        units.push(unit(
            EntityId { idx, generation: 1 },
            kind,
            team,
            coordinate,
            coordinate,
        ));
    }
    WorldView {
        tick: 1,
        viewer: Some(Team::Radiant),
        players: players(),
        units,
        projectiles: Vec::new(),
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn players() -> Vec<bota_proto::PlayerView> {
    [true, false]
        .map(|own| bota_proto::PlayerView {
            slot: SlotId(u8::from(!own)),
            team: if own { Team::Radiant } else { Team::Dire },
            hero: crate::SHADOW_FIEND,
            unit: own.then_some(HERO),
            level: if own { 6 } else { 1 },
            xp: 0,
            gold: own.then_some(0),
            stash: own.then(|| vec![None; 6]),
            kit: None,
            kills: 0,
            deaths: 0,
            assists: 0,
            last_hits: 0,
            denies: 0,
            respawn_left: 0,
        })
        .to_vec()
}

fn unit(id: EntityId, kind: UnitKind, team: Team, x: i32, y: i32) -> UnitView {
    fixtures::UnitFixture {
        id,
        kind,
        team,
        pos: Vec2::from_ints(x, y),
        mana: 0,
        attack_damage: 50,
        attack_time: 1700,
        attributes: bota_proto::Attributes::all(20),
        primary: Some(bota_proto::Attribute::Agility),
        hero: (kind == UnitKind::Hero).then_some(crate::SHADOW_FIEND),
        owner: None,
        level: 0,
    }
    .build()
}

fn body(view: &mut WorldView, id: EntityId) -> &mut UnitView {
    view.units
        .iter_mut()
        .find(|unit| unit.id == id)
        .expect("fixture body")
}
