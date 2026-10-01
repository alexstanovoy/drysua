//! Trusted spawn modifiers on the builtin arena: applied at spawn to the
//! selected categories, timed or match-long, and off by default.

use bota_proto::{Cheat, MAX_MODIFIER_TICKS, ModifierSpec, Target, Team, UnitKind, Vec2};
use bota_server::game::{
    Entity, MAX_SPAWN_MODIFIERS, MELEE_CREEP, ModifierDuration, SpawnCategory, SpawnModifier,
    SpawnSelector, SpawnTarget,
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
fn default_arena_keeps_cheats_off_and_rejects_cheat_orders() {
    let (mut arena, _start) = Arena::new(config(38)).expect("default arena");
    assert!(!arena.world.cheats);
    assert!(arena.world.spawn_modifiers.is_empty());
    assert!(arena.world.applied.is_empty());
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

#[test]
fn a_rule_lands_on_every_selected_spawn_and_respawn_at_full_scaled_health() {
    let (plain, _start) = Arena::new(config(35)).expect("plain arena");
    let baseline = hero_max_hp(&plain, Team::Radiant);
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
        ..max_hp_rule(12_000)
    };
    let (mut arena, _start) =
        Arena::new_with_spawn_modifiers(config(35), vec![rule.clone()]).expect("wide arena");
    assert!(!arena.world.cheats, "trusted setup opens no cheat gate");
    for team in [Team::Radiant, Team::Dire] {
        let unit = hero_entity(&arena, team);
        assert!(
            hero_max_hp(&arena, team) > baseline,
            "{team:?} maximum is scaled"
        );
        assert_full_with_rule(&arena, unit, &rule, &format!("{team:?} spawn"));
    }
    let tower = arena
        .world
        .entities
        .iter()
        .find(|entity| arena.world.kind.get(*entity) == Some(&UnitKind::Tower))
        .expect("a tower stands at tick one");
    assert_eq!(first_spec(&arena, tower), rule.spec, "tower");
    let creep = arena.world.spawn_creep(
        &MELEE_CREEP,
        Team::Radiant,
        Vec2::from_ints(9_216, 9_216),
        0,
        0,
    );
    arena.world.settle();
    assert_full_with_rule(&arena, creep, &rule, "lane creep");
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
    assert_full_with_rule(&arena, new, &rule, "respawn");
}

fn assert_full_with_rule(arena: &Arena, unit: Entity, rule: &SpawnModifier, case: &str) {
    assert_eq!(first_spec(arena, unit), rule.spec, "{case}");
    let stats = arena.world.stats.get(unit).expect("settled stats");
    let health = arena.world.health.get(unit).expect("a pool");
    assert!(stats.max_hp > bota_proto::Fixed::ZERO, "{case}");
    assert_eq!(
        health.hp, stats.max_hp,
        "{case} stands full at the scaled maximum"
    );
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
fn out_of_bounds_zero_or_overlong_lifetime_and_too_many_rules_are_refused() {
    let invalid = max_hp_rule(ModifierSpec::MAX_SCALE + 1);
    for (case, rules, expected) in [
        ("out of bounds", vec![invalid.clone()], "out of bounds"),
        (
            "zero lifetime",
            vec![hero_rule(ModifierSpec::NOMINAL, ModifierDuration::Ticks(0))],
            "duration",
        ),
        (
            "lifetime past the ceiling",
            vec![hero_rule(
                ModifierSpec::NOMINAL,
                ModifierDuration::Ticks(MAX_MODIFIER_TICKS + 1),
            )],
            "duration",
        ),
        (
            "too many rules, refused before walking them",
            vec![invalid; MAX_SPAWN_MODIFIERS + 1],
            "65 spawn modifiers exceed the 64 a match may carry",
        ),
    ] {
        let error = Arena::new_with_spawn_modifiers(config(37), rules)
            .err()
            .expect(case);
        assert!(
            matches!(&error, ArenaError::Config { message } if message.contains(expected)),
            "{case}: unexpected error: {error}"
        );
    }
}
