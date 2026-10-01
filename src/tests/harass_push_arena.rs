//! HarassPush through the production request path against the native simulator.

use super::map2_tests::configured_environment;
use super::*;
use crate::ppo_arena::issue_request;
use crate::{ScriptKind, ScriptedPolicy};
use bota_proto::{Fixed, Team, UnitKind, Vec2};
use bota_server::game::{Modifier, ModifierKind, Modifiers, UnitOrder, World};

/// Decisions to walk home; the path from the base start bends before it nears the fountain.
const RETREAT_DECISIONS: usize = 40;
const MID: Vec2 = Vec2::from_ints(9_216, 9_216);
/// Distance from the own fountain toward mid where the hero starts.
const BASE_DISTANCE: i64 = 2_500;

/// Regression: HarassPush counted a Tango's Mending as a Healing Salve's 400 health,
/// so a few health points above its retreat share it turned from home back to the lane.
#[test]
fn harass_push_counts_a_tango_at_its_own_rate_and_keeps_walking_home() {
    let mut environment = configured_environment(1_200, 0, OpponentRuntime::Idle, |world| {
        let hero = world.seats[0].unit.expect("own hero");
        let enemy = world.seats[1].unit.expect("enemy hero");
        // In the own base on the way to mid: home and the lane post lie on opposite sides.
        let fountain = fountain_of(world, Team::Radiant);
        let (dx, dy) = (
            i64::from(MID.x.to_int() - fountain.x.to_int()),
            i64::from(MID.y.to_int() - fountain.y.to_int()),
        );
        let span = (dx * dx + dy * dy).isqrt();
        let step = |delta: i64| i32::try_from(delta * BASE_DISTANCE / span).expect("on the map");
        world.transform.get_mut(hero).expect("own position").pos = Vec2::from_ints(
            fountain.x.to_int() + step(dx),
            fountain.y.to_int() + step(dy),
        );
        world.transform.get_mut(enemy).expect("enemy position").pos =
            Vec2::from_ints(17_000, 17_000);
        world.set_order(hero, UnitOrder::Stand);
        world.set_order(enemy, UnitOrder::Stand);
        // No fountain regeneration from the spawn: the Tango is the only heal.
        world.modifiers.insert(hero, Modifiers::default());
        world.health.get_mut(hero).expect("health").hp = Fixed::from_int(150);
        // A Tango eaten from a grown tree: 115 health over 480 ticks.
        world.put_modifier(
            hero,
            Modifier {
                kind: ModifierKind::Mending {
                    per_tick: 115 * 100 / 480,
                    breaks: false,
                },
                source: Some(hero),
                ticks_left: Some(480),
            },
        );
    });
    let hero = environment.seats[0].tracker.own_hero().expect("own hero");
    assert!(
        hero.hp * 100 <= hero.max_hp * 30 && (hero.hp + 10) * 100 > hero.max_hp * 30,
        "the hero starts at its retreat share and the Tango lifts it past"
    );
    environment.seats[0].script = ScriptedPolicy::new(ScriptKind::HarassPush);
    let fountain = radiant_fountain(&environment);
    let start = own_distance(&environment, fountain);
    run_decisions(&mut environment, RETREAT_DECISIONS);
    let hero = environment.seats[0].tracker.own_hero().expect("own hero");
    assert!(
        hero.hp * 100 > hero.max_hp * 30,
        "the Tango lifted the hero past its retreat share"
    );
    let end = own_distance(&environment, fountain);
    assert!(
        end < start,
        "still walking home: {start} -> {end} units from the fountain"
    );
}

/// Decides and plays `decisions` rule-policy decisions of seat 0, three ticks apart.
fn run_decisions(environment: &mut TrainingEnvironment, decisions: usize) {
    for _ in 0..decisions {
        let seat = &mut environment.seats[0];
        let (action, space) = seat
            .script
            .decide(&seat.tracker, &seat.persistence, &seat.readiness)
            .expect("decision");
        seat.local
            .note_decision(space.tick(), action.kind())
            .expect("decision history");
        let issued = space.decode(action).expect("decode");
        let request = issue_request(seat, issued, &space, action.kind(), true).expect("request");
        advance_interval(environment, vec![request, None], 3).expect("ticks");
    }
}

fn radiant_fountain(environment: &TrainingEnvironment) -> Vec2 {
    fountain_of(environment.arena.world_for_test(), Team::Radiant)
}

fn fountain_of(world: &World, team: Team) -> Vec2 {
    world
        .entities
        .iter()
        .find(|entity| {
            world.kind.get(*entity) == Some(&UnitKind::Fountain)
                && world.team.get(*entity) == Some(&team)
        })
        .and_then(|entity| world.transform.get(entity))
        .expect("fountain")
        .pos
}

fn own_distance(environment: &TrainingEnvironment, to: Vec2) -> i64 {
    let hero = environment.seats[0].tracker.own_hero().expect("own hero");
    hero.pos.distance_squared(to).isqrt() / i64::from(Fixed::ONE.raw)
}
