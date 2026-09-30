//! Aimed razes through the production request path against the native simulator.

use super::map2_tests::configured_environment;
use super::*;
use crate::ppo_arena::game_summary::GameSummary;
use crate::ppo_arena::{build_environment, issue_request, terminal_outcome};
use crate::raze_aim::{SHADOWRAZE_RADIUS, facing_towards, raze_center};
use crate::{
    ActionTarget, ControlledUnit, PointIndex, PointSource, ScriptKind, ScriptedPolicy,
    StructuredAction, Teacher,
};
use bota_proto::{AbilitySlot, Fixed, MapId, Order, Target, Team, UnitKind, Vec2};
use bota_server::game::{Entity, MELEE_CREEP, UnitOrder, World, wire_id};

const ORIGIN: Vec2 = Vec2::from_ints(8_900, 9_216);
const CROSSING_TARGET_OFFSET: i32 = 3_000;
/// One aim decision plus three Continue decisions cover a full reversal.
const DECISION_LIMIT: usize = 4;

#[derive(Clone, Copy, Debug)]
enum Motion {
    Standing,
    Crossing,
}

#[test]
fn one_aimed_raze_decision_hits_a_standing_or_crossing_hero_behind_or_beside_at_every_reach() {
    for (slot, reach) in [(0u8, 200), (1, 450), (2, 700)] {
        for facing in [16_384u16, 32_768] {
            for motion in [Motion::Standing, Motion::Crossing] {
                let decisions = decisions_until_hit(slot, reach, facing, motion);
                assert!(
                    decisions <= DECISION_LIMIT,
                    "slot {slot} facing {facing} {motion:?}: {decisions} decisions"
                );
            }
        }
    }
}

/// Plays the aimed raze and then Continue until the raze lands, and returns the decision count.
fn decisions_until_hit(slot: u8, reach: i32, facing: u16, motion: Motion) -> usize {
    let mut environment = aim_environment(reach, facing, motion);
    let (_, space) =
        prepare_neural_seat_policy_sample(&mut environment.seats[0]).expect("aim space");
    let tracker = &environment.seats[0].tracker;
    let hero = tracker.own_hero().expect("hero");
    let enemy = tracker.current().expect("view").players[1]
        .unit
        .expect("enemy hero");
    let enemy_view = space
        .entity_candidates()
        .iter()
        .find(|candidate| candidate.id() == enemy)
        .expect("visible enemy")
        .unit();
    assert!(
        !crate::raze_aim::raze_contains(tracker, hero, enemy_view, hero.facing.brads, reach),
        "the untargeted raze must miss, so the old schema needed a turn decision first"
    );
    let aimed = StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot),
        target: ActionTarget::Entity(space.entity_index(enemy).expect("enemy candidate")),
    };
    assert!(space.allows(aimed));
    let mut action = aimed;
    let mut space = space;
    for decision in 1..=DECISION_LIMIT + 1 {
        let (_, request) =
            neural_policy_request_in_space(&mut environment.seats[0], action, &space)
                .expect("aim transport");
        advance_interval(&mut environment, vec![request, None], 3).expect("aim ticks");
        reject_production_rejection(&environment, "aimed raze").expect("no rejection");
        let own = &GameSummary::capture(&environment, None, 3).json()["own"];
        if own["raze_hero_hits"] == 1 {
            let casts = &own["casts"];
            assert_eq!(
                casts["raze_near"].as_u64().unwrap()
                    + casts["raze_mid"].as_u64().unwrap()
                    + casts["raze_far"].as_u64().unwrap(),
                1
            );
            return decision;
        }
        action = StructuredAction::Continue;
        space = prepare_neural_seat_policy_sample(&mut environment.seats[0])
            .expect("continue space")
            .1;
    }
    DECISION_LIMIT + 1
}

/// Shadow Fiend at `ORIGIN` looking `facing` away from the enemy hero east of it at `reach`.
fn aim_environment(reach: i32, facing: u16, motion: Motion) -> TrainingEnvironment {
    raze_environment(ORIGIN, facing, |world, _, enemy| {
        let enemy_position = Vec2::from_ints(ORIGIN.x.to_int() + reach, ORIGIN.y.to_int());
        world.transform.get_mut(enemy).expect("enemy position").pos = enemy_position;
        if let Motion::Crossing = motion {
            world.set_order(
                enemy,
                UnitOrder::Move {
                    pos: Vec2::from_ints(
                        enemy_position.x.to_int(),
                        enemy_position.y.to_int() + CROSSING_TARGET_OFFSET,
                    ),
                },
            );
        }
    })
}

/// Hero-raze accuracy of Teacher's own FollowUnit, Stop, Cast aim against the same
/// Teacher handing hero razes to the aim macro, one of each per game, sides alternating.
#[test]
#[ignore = "measurement over about a hundred full Map2 games; run with --ignored --nocapture"]
fn teacher_hero_raze_accuracy_native_aim_versus_aim_macro() {
    let games: u64 = std::env::var("DRYSUA_AIM_GAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100);
    let workers = 4u64;
    let totals = std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|worker| {
                scope.spawn(move || {
                    let mut totals = [[0u64; 4]; 2];
                    for game in (worker..games).step_by(workers as usize) {
                        let macro_seat = (game % 2) as usize;
                        let seats = teacher_game(71_000 + game, macro_seat);
                        for (seat, counts) in seats.iter().enumerate() {
                            let variant = usize::from(seat == macro_seat);
                            for (total, count) in totals[variant].iter_mut().zip(counts) {
                                *total += count;
                            }
                        }
                    }
                    totals
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("measurement worker"))
            .fold([[0u64; 4]; 2], |mut sum, part| {
                for (variant, counts) in sum.iter_mut().zip(part) {
                    for (total, count) in variant.iter_mut().zip(counts) {
                        *total += count;
                    }
                }
                sum
            })
    });
    for (label, [hits, hero_casts, casts, wins]) in ["native", "macro"].into_iter().zip(totals) {
        eprintln!(
            "event=teacher_raze_accuracy aim={label} games={games} wins={wins} raze_casts={casts} hero_aimed_casts={hero_casts} raze_hero_hits={hits} hits_per_raze={:.3} hits_per_hero_aimed_cast={:.3}",
            hits as f64 / casts.max(1) as f64,
            hits as f64 / hero_casts.max(1) as f64,
        );
    }
}

/// Per seat: raze hero hits, hero-aimed raze casts, all raze casts, wins.
fn teacher_game(seed: u64, macro_seat: usize) -> [[u64; 4]; 2] {
    let mut environment =
        build_environment(seed, MapId(2), 0, OpponentRuntime::Teacher, Vec::new())
            .expect("teacher arena");
    environment.seats[macro_seat].script =
        crate::ScriptedPolicy::Teacher(Box::new(Teacher::with_macro_hero_aim()));
    let mut hero_aimed = [false; 2];
    let mut hero_casts = [0u64; 2];
    let mut winner = None;
    let mut ticks = 0u32;
    while winner.is_none() && ticks < TICK_CAP {
        let mut requests = Vec::with_capacity(2);
        for (index, seat) in environment.seats.iter_mut().enumerate() {
            let (action, space) = seat
                .script
                .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                .expect("teacher decision");
            seat.local
                .note_decision(space.tick(), action.kind())
                .expect("decision history");
            match action {
                StructuredAction::Cast {
                    target: ActionTarget::Entity(target),
                    ..
                } => hero_aimed[index] = space.entity_candidates()[target.0].kind == UnitKind::Hero,
                StructuredAction::Continue => {}
                _ => hero_aimed[index] = false,
            }
            let issued = space.decode(action).expect("teacher decode");
            let request =
                issue_request(seat, issued, &space, action.kind(), true).expect("teacher request");
            if hero_aimed[index]
                && let Some(Request {
                    order:
                        Order::Cast {
                            slot,
                            target: Target::None,
                        },
                    ..
                }) = request
                && slot.0 < 3
            {
                hero_casts[index] += 1;
            }
            requests.push(request);
        }
        let advanced = advance_interval(&mut environment, requests, 3.min(TICK_CAP - ticks))
            .expect("teacher ticks");
        ticks += advanced.ticks;
        winner = advanced.winner;
    }
    let summary =
        GameSummary::capture(&environment, terminal_outcome(&environment, winner), ticks).json();
    std::array::from_fn(|seat| {
        let hero = &summary[if seat == 0 { "own" } else { "enemy" }];
        let casts = &hero["casts"];
        let razes = ["raze_near", "raze_mid", "raze_far"]
            .iter()
            .map(|label| casts[*label].as_u64().expect("raze casts"))
            .sum();
        let team = environment.seats[seat].tracker.team();
        [
            hero["raze_hero_hits"].as_u64().expect("hits"),
            hero_casts[seat],
            razes,
            u64::from(winner == Some(team)),
        ]
    })
}

/// A far Point raze at the best far landing strikes a creep cluster standing
/// past the far reach, which no raze along the current facing touches.
#[test]
fn a_point_raze_at_the_best_far_landing_strikes_a_creep_cluster_at_the_edge_of_range() {
    let mut creeps = Vec::new();
    let mut environment = raze_environment(ORIGIN, 32_768, |world, _, enemy| {
        world.transform.get_mut(enemy).expect("enemy position").pos =
            Vec2::from_ints(ORIGIN.x.to_int() - 2_500, ORIGIN.y.to_int());
        for (x, y) in [(850, 150), (850, -150), (900, 0)] {
            let at = Vec2::from_ints(ORIGIN.x.to_int() + x, ORIGIN.y.to_int() + y);
            creeps.push(world.spawn_creep(&MELEE_CREEP, Team::Dire, at, 0, 0));
        }
    });
    let (_, space) =
        prepare_neural_seat_policy_sample(&mut environment.seats[0]).expect("raze space");
    let far = 2;
    let (facing, _) = raze_point(&space, |source| source == PointSource::RazeFacing);
    assert_eq!(
        facing.raze_coverage[far].units, 0,
        "the untargeted raze misses"
    );
    let (cluster, index) = raze_point(&space, |source| {
        source == PointSource::RazeCluster { reach: 700 }
    });
    assert_eq!(cluster.raze_coverage[far].units, 3);
    let before = hit_points(&environment, &creeps);
    let decisions = raze_until_cast(&mut environment, space, 2, ActionTarget::Point(index));
    assert!(decisions <= DECISION_LIMIT, "{decisions} decisions");
    let after = hit_points(&environment, &creeps);
    let struck = before
        .iter()
        .zip(&after)
        .filter(|(was, is)| is < was)
        .count();
    assert!(struck >= 2, "struck {struck} of {before:?} -> {after:?}");
}

/// A hero walking straight behind an opaque wall leaves the razer's sight. A
/// raze strikes only what the caster's side sees, so the blind far Point raze
/// at the extrapolated position is timed to land as the hero steps back into
/// sight, where a raze toward its last sighting lands far behind it.
#[test]
fn a_blind_point_raze_at_the_extrapolated_fogged_hero_strikes_it_as_it_reappears() {
    let mut environment = raze_environment(ORIGIN, 32_768, |world, _, enemy| {
        world.transform.get_mut(enemy).expect("enemy position").pos = Vec2::from_ints(9_550, 8_920);
        world.set_order(
            enemy,
            UnitOrder::Move {
                pos: Vec2::from_ints(9_550, 12_000),
            },
        );
        // A planted tree line between them hides part of the walk.
        for x in (9_280..9_344).step_by(16) {
            for y in (9_088..9_280).step_by(16) {
                world.trees.plant(Vec2::from_ints(x, y), u32::MAX);
            }
        }
        world.lay_sight_block();
    });
    let enemy = enemy_hero(&environment);
    let mut hidden = 0;
    while hidden < 30 {
        advance_interval(&mut environment, vec![None, None], 3).expect("walk ticks");
        let seen = environment.seats[0]
            .tracker
            .current()
            .expect("view")
            .units
            .iter()
            .any(|unit| unit.id == wire_id(enemy));
        hidden = if seen { 0 } else { hidden + 3 };
        assert!(environment.arena.tick() < 200, "the hero never left sight");
    }
    let (_, space) =
        prepare_neural_seat_policy_sample(&mut environment.seats[0]).expect("blind space");
    let (last_seen, _) = raze_point(&space, |source| {
        matches!(source, PointSource::LastSeenHero { .. })
    });
    let (guess, index) = raze_point(&space, |source| {
        matches!(source, PointSource::ExtrapolatedHero { .. })
    });
    let before = hit_points(&environment, &[enemy]);
    let decisions = raze_until_cast(&mut environment, space, 2, ActionTarget::Point(index));
    assert!(decisions <= DECISION_LIMIT, "{decisions} decisions");
    assert!(
        hit_points(&environment, &[enemy])[0] < before[0],
        "the blind raze hit"
    );
    let own = environment.seats[0]
        .tracker
        .own_hero()
        .expect("own hero")
        .pos;
    let landing = |point: Vec2| raze_center(own, facing_towards(own, point), 700);
    let struck = unit_position(&environment, enemy);
    let radius = Fixed::from_int(SHADOWRAZE_RADIUS);
    assert!(landing(guess.position).within(struck, radius));
    assert!(!landing(last_seen.position).within(struck, radius));
}

/// An untargeted raze fires along the current facing on the decision's own tick.
#[test]
fn a_none_raze_fires_on_the_decision_tick() {
    let mut environment = aim_environment(450, 0, Motion::Standing);
    let (_, space) =
        prepare_neural_seat_policy_sample(&mut environment.seats[0]).expect("raze space");
    let action = StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(1),
        target: ActionTarget::None,
    };
    let (_, request) = neural_policy_request_in_space(&mut environment.seats[0], action, &space)
        .expect("none transport");
    assert!(matches!(
        request,
        Some(Request {
            order: Order::Cast {
                slot: AbilitySlot(1),
                target: Target::None
            },
            ..
        })
    ));
    advance_interval(&mut environment, vec![request, None], 1).expect("cast tick");
    reject_production_rejection(&environment, "none raze").expect("no rejection");
    let own = &GameSummary::capture(&environment, None, 1).json()["own"];
    assert_eq!(own["casts"]["raze_mid"], 1);
    assert_eq!(own["raze_hero_hits"], 1);
}

/// Decides the raze, then Continue until a raze is cast; returns the decision count.
fn raze_until_cast(
    environment: &mut TrainingEnvironment,
    mut space: ActionSpace,
    slot: u8,
    target: ActionTarget,
) -> usize {
    let mut action = StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot),
        target,
    };
    for decision in 1..=DECISION_LIMIT + 1 {
        let (_, request) =
            neural_policy_request_in_space(&mut environment.seats[0], action, &space)
                .expect("raze transport");
        advance_interval(environment, vec![request, None], 3).expect("raze ticks");
        reject_production_rejection(environment, "point raze").expect("no rejection");
        let casts = &GameSummary::capture(environment, None, 3).json()["own"]["casts"];
        if ["raze_near", "raze_mid", "raze_far"]
            .iter()
            .any(|label| casts[*label] != 0)
        {
            return decision;
        }
        action = StructuredAction::Continue;
        space = prepare_neural_seat_policy_sample(&mut environment.seats[0])
            .expect("continue space")
            .1;
    }
    DECISION_LIMIT + 1
}

fn raze_point(
    space: &ActionSpace,
    source: impl Fn(PointSource) -> bool,
) -> (crate::PointCandidate, PointIndex) {
    let (index, point) = space
        .point_candidates()
        .iter()
        .enumerate()
        .find(|(_, point)| source(point.source))
        .expect("raze point candidate");
    (*point, PointIndex(index))
}

fn enemy_hero(environment: &TrainingEnvironment) -> Entity {
    environment.arena.world_for_test().seats[1]
        .unit
        .expect("enemy hero")
}

fn unit_position(environment: &TrainingEnvironment, unit: Entity) -> Vec2 {
    environment
        .arena
        .world_for_test()
        .transform
        .get(unit)
        .expect("position")
        .pos
}

fn hit_points(environment: &TrainingEnvironment, units: &[Entity]) -> Vec<Fixed> {
    let world = environment.arena.world_for_test();
    units
        .iter()
        .map(|unit| world.health.get(*unit).expect("health").hp)
        .collect()
}

/// Shadow Fiend at `own` looking `facing` with every raze learned and mana for one.
fn raze_environment(
    own: Vec2,
    facing: u16,
    configure: impl FnOnce(&mut World, Entity, Entity),
) -> TrainingEnvironment {
    configured_environment(30, 0, OpponentRuntime::Idle, |world| {
        let hero = world.seats[0].unit.expect("own hero");
        let enemy = world.seats[1].unit.expect("enemy hero");
        let at = world.transform.get_mut(hero).expect("own position");
        at.pos = own;
        at.facing.brads = facing;
        world.set_order(hero, UnitOrder::Stand);
        world.set_order(enemy, UnitOrder::Stand);
        let book = world.abilities.get_mut(hero).expect("abilities");
        for raze in &mut book.slots[..3] {
            raze.level = 1;
        }
        world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(150);
        configure(world, hero, enemy);
    })
}

/// Decisions each rule policy gets beside the lone enemy tower.
const TOWER_DECISIONS: usize = 10;

/// A raze never targets a structure: with only the enemy tower under the far raze and the
/// own wave tanking it, neither rule policy razes, no raze target mode is legal on the
/// tower, and no raze point candidate counts it.
#[test]
fn no_rule_policy_or_raze_candidate_aims_a_raze_at_a_lone_enemy_tower() {
    for kind in [ScriptKind::Teacher, ScriptKind::HarassPush] {
        let mut environment = lone_tower_environment();
        environment.seats[0].script = ScriptedPolicy::new(kind);
        for decision in 0..TOWER_DECISIONS {
            let seat = &mut environment.seats[0];
            let (action, space) = seat
                .script
                .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                .expect("decision");
            let tower = space
                .entity_candidates()
                .iter()
                .position(|candidate| candidate.kind == UnitKind::Tower)
                .expect("visible enemy tower");
            for slot in 0..3 {
                let at_tower = StructuredAction::Cast {
                    unit: ControlledUnit::Hero,
                    slot: AbilitySlot(slot),
                    target: ActionTarget::Entity(crate::EntityIndex(tower)),
                };
                assert!(!space.allows(at_tower), "{kind:?} {decision}: slot {slot}");
            }
            assert!(
                space.point_candidates().iter().all(|point| {
                    point.source != PointSource::RazeCluster { reach: 700 }
                        && point.raze_coverage.iter().all(|covered| covered.units == 0)
                }),
                "{kind:?} {decision}: a raze candidate counts the tower"
            );
            assert!(
                !matches!(action, StructuredAction::Cast { unit: ControlledUnit::Hero, slot, .. } if slot.0 < 3),
                "{kind:?} {decision}: razed beside the tower: {action:?}"
            );
            seat.local
                .note_decision(space.tick(), action.kind())
                .expect("decision history");
            let issued = space.decode(action).expect("decode");
            let request =
                issue_request(seat, issued, &space, action.kind(), true).expect("request");
            advance_interval(&mut environment, vec![request, None], 3).expect("ticks");
        }
    }
}

/// Shadow Fiend 900 from the Dire tower nearest mid, facing it with mana for every raze,
/// two own melee creeps tanking the tower, and the enemy hero at the far corner.
fn lone_tower_environment() -> TrainingEnvironment {
    let mid = Vec2::from_ints(9_216, 9_216);
    raze_environment(mid, 0, |world, hero, enemy| {
        let tower = world
            .entities
            .iter()
            .filter(|entity| {
                world.kind.get(*entity) == Some(&UnitKind::Tower)
                    && world.team.get(*entity) == Some(&Team::Dire)
            })
            .min_by_key(|entity| {
                let at = world.transform.get(*entity).expect("tower position").pos;
                (at.distance_squared(mid), *entity)
            })
            .expect("Dire tower");
        let at = world.transform.get(tower).expect("tower position").pos;
        let toward_mid = |distance: f64| {
            let (dx, dy) = (
                f64::from(mid.x.to_int() - at.x.to_int()),
                f64::from(mid.y.to_int() - at.y.to_int()),
            );
            let scale = distance / dx.hypot(dy);
            Vec2::from_ints(
                at.x.to_int() + (dx * scale).round() as i32,
                at.y.to_int() + (dy * scale).round() as i32,
            )
        };
        let own = toward_mid(900.0);
        let placed = world.transform.get_mut(hero).expect("own position");
        placed.pos = own;
        placed.facing.brads = facing_towards(own, at);
        world.mana.get_mut(hero).expect("mana").mana = Fixed::from_int(400);
        for offset in [-60, 60] {
            let near = toward_mid(450.0);
            let near = Vec2::from_ints(near.x.to_int() + offset, near.y.to_int() - offset);
            world.spawn_creep(&MELEE_CREEP, Team::Radiant, near, 0, 0);
        }
        world.transform.get_mut(enemy).expect("enemy position").pos =
            Vec2::from_ints(17_000, 17_000);
    })
}
