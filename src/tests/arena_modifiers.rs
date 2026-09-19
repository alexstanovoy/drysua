//! Trusted spawn modifiers on the builtin arena: applied at spawn to the
//! selected categories, timed or match-long, and off by default.

use bota_proto::{Cheat, ModifierSpec, Target, Team, UnitKind, Vec2};
use bota_server::game::{
    Entity, MELEE_CREEP, ModifierDuration, SpawnCategory, SpawnModifier, SpawnSelector, SpawnTarget,
};

use super::*;

fn config(seed: u64) -> ArenaConfig {
    ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed,
    }
}

fn hero_entity(arena: &Arena, team: Team) -> Entity {
    arena
        .world
        .seats
        .iter()
        .find(|seat| seat.team == team)
        .expect("a seat per team")
        .unit
        .expect("a standing hero")
}

fn hero_max_hp(arena: &Arena, team: Team) -> i32 {
    let unit = hero_entity(arena, team);
    arena
        .world
        .stats
        .get(unit)
        .expect("settled stats")
        .max_hp
        .to_int()
}

fn first_spec(arena: &Arena, unit: Entity) -> ModifierSpec {
    arena
        .world
        .applied
        .get(unit)
        .expect("applied")
        .iter()
        .next()
        .expect("one entry")
        .spec
}

fn hero_rule(spec: ModifierSpec, duration: ModifierDuration) -> SpawnModifier {
    SpawnModifier {
        select: SpawnSelector {
            team: None,
            targets: vec![SpawnTarget::Category(SpawnCategory::Hero)],
        },
        spec,
        duration,
    }
}

fn max_hp_rule(scale: i32) -> SpawnModifier {
    hero_rule(
        ModifierSpec {
            max_hp: scale,
            ..ModifierSpec::NOMINAL
        },
        ModifierDuration::MatchLong,
    )
}

#[test]
fn an_arena_without_modifiers_keeps_cheats_off() {
    let (arena, _start) = Arena::new(config(31)).expect("default arena");
    assert!(!arena.world.cheats);
    assert!(arena.world.spawn_modifiers.is_empty());
    assert!(arena.world.applied.is_empty());
}

#[test]
fn a_rule_lands_on_both_heroes_at_full_scaled_health() {
    let (plain, _start) = Arena::new(config(33)).expect("plain arena");
    let baseline = hero_max_hp(&plain, Team::Radiant);
    let rule = max_hp_rule(12_000);
    let (arena, _start) =
        Arena::new_with_spawn_modifiers(config(33), vec![rule.clone()]).expect("modified arena");
    assert!(!arena.world.cheats, "trusted setup opens no cheat gate");
    for team in [Team::Radiant, Team::Dire] {
        let unit = hero_entity(&arena, team);
        assert_eq!(first_spec(&arena, unit), rule.spec);
        let stats = arena.world.stats.get(unit).expect("settled stats");
        assert!(stats.max_hp.to_int() > baseline, "the maximum is scaled");
        let health = arena.world.health.get(unit).expect("a pool");
        assert_eq!(
            health.hp, stats.max_hp,
            "a spawn stands full at the scaled maximum"
        );
    }
}

#[test]
fn a_timed_rule_honours_its_ticks() {
    let rule = |ticks| {
        hero_rule(
            ModifierSpec {
                max_hp: 12_000,
                ..ModifierSpec::NOMINAL
            },
            ModifierDuration::Ticks(ticks),
        )
    };
    let left = |arena: &Arena, unit: Entity| {
        arena
            .world
            .applied
            .get(unit)
            .map(|applied| applied.iter().next().expect("one entry").ticks_left)
    };

    let (mut short, _start) =
        Arena::new_with_spawn_modifiers(config(34), vec![rule(3)]).expect("short arena");
    let unit = hero_entity(&short, Team::Radiant);
    // The spawn application counts its first tick during the first advance.
    assert_eq!(left(&short, unit), Some(Some(2)));
    short.step(&[None, None]).expect("tick");
    assert_eq!(left(&short, unit), Some(Some(1)));
    short.step(&[None, None]).expect("tick");
    assert_eq!(left(&short, unit), None, "the rule lifts after its ticks");

    let (long, _start) =
        Arena::new_with_spawn_modifiers(config(34), vec![rule(50)]).expect("long arena");
    let unit = hero_entity(&long, Team::Radiant);
    assert_eq!(
        left(&long, unit),
        Some(Some(49)),
        "the requested ticks are kept"
    );
}

#[test]
fn a_structure_and_a_lane_creep_carry_the_max_hp_rule() {
    let rule = SpawnModifier {
        select: SpawnSelector {
            team: None,
            targets: vec![
                SpawnTarget::Category(SpawnCategory::Hero),
                SpawnTarget::Category(SpawnCategory::LaneCreep),
                SpawnTarget::Category(SpawnCategory::NeutralCreep),
                SpawnTarget::Category(SpawnCategory::Structure),
            ],
        },
        spec: ModifierSpec {
            max_hp: 12_000,
            ..ModifierSpec::NOMINAL
        },
        duration: ModifierDuration::MatchLong,
    };
    let (mut arena, _start) =
        Arena::new_with_spawn_modifiers(config(35), vec![rule.clone()]).expect("wide arena");
    let tower = arena
        .world
        .entities
        .iter()
        .find(|entity| arena.world.kind.get(*entity) == Some(&UnitKind::Tower))
        .expect("a tower stands at tick one");
    assert_eq!(first_spec(&arena, tower), rule.spec);
    let creep = arena.world.spawn_creep(
        &MELEE_CREEP,
        Team::Radiant,
        Vec2::from_ints(9_216, 9_216),
        0,
        0,
    );
    arena.world.settle();
    assert_eq!(first_spec(&arena, creep), rule.spec);
    let stats = arena.world.stats.get(creep).expect("settled stats");
    assert!(stats.max_hp > bota_proto::Fixed::ZERO);
}

#[test]
fn a_respawned_hero_carries_the_rule_and_stands_full() {
    let rule = max_hp_rule(12_000);
    let (mut arena, _start) =
        Arena::new_with_spawn_modifiers(config(36), vec![rule.clone()]).expect("arena");
    let old = hero_entity(&arena, Team::Radiant);
    let seat = arena
        .world
        .seats
        .iter_mut()
        .find(|seat| seat.team == Team::Radiant)
        .expect("a seat");
    seat.unit = None;
    seat.respawn_left = 0;
    assert!(arena.world.despawn(old));
    arena.step(&[None, None]).expect("respawn tick");
    let new = hero_entity(&arena, Team::Radiant);
    assert_ne!(new, old, "a new body stands");
    assert_eq!(first_spec(&arena, new), rule.spec);
    let stats = arena.world.stats.get(new).expect("settled stats");
    let health = arena.world.health.get(new).expect("a pool");
    assert_eq!(health.hp, stats.max_hp, "a respawn stands full");
}

#[test]
fn an_out_of_bounds_or_zero_lifetime_rule_is_refused() {
    let unbounded = max_hp_rule(ModifierSpec::MAX_SCALE + 1);
    let error = Arena::new_with_spawn_modifiers(config(37), vec![unbounded])
        .err()
        .expect("out of bounds");
    assert!(
        matches!(&error, ArenaError::Config { message } if message.contains("out of bounds")),
        "unexpected error: {error}"
    );
    let zero = hero_rule(ModifierSpec::NOMINAL, ModifierDuration::Ticks(0));
    let error = Arena::new_with_spawn_modifiers(config(37), vec![zero])
        .err()
        .expect("zero lifetime");
    assert!(
        matches!(&error, ArenaError::Config { message } if message.contains("duration")),
        "unexpected error: {error}"
    );
}

#[test]
fn cheat_orders_are_refused_in_training_worlds() {
    let (mut arena, _start) = Arena::new(config(38)).expect("default arena");
    let request = Request {
        seq: 1,
        unit: None,
        order: Order::Cheat {
            cheat: Cheat::ApplyModifier {
                target: Target::None,
                spec: ModifierSpec {
                    max_hp: 11_000,
                    ..ModifierSpec::NOMINAL
                },
                ticks: 10,
            },
        },
    };
    let step = arena
        .step(&[Some(request), None])
        .expect("rejected order tick");
    assert!(step.messages[0].iter().any(|message| matches!(
        message,
        ServerMsg::OrderRejected {
            reason: bota_proto::RejectReason::NoCheats,
            ..
        }
    )));
    assert!(arena.world.applied.is_empty());
}
