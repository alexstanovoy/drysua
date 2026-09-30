//! Aimed razes through the production request path against the native simulator.

use super::map2_tests::configured_environment;
use super::*;
use crate::ppo_arena::game_summary::GameSummary;
use crate::ppo_arena::{build_environment, issue_request, terminal_outcome};
use crate::{ActionTarget, ControlledUnit, StructuredAction, Teacher};
use bota_proto::{AbilitySlot, Fixed, MapId, Order, Target, UnitKind, Vec2};
use bota_server::game::UnitOrder;

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
    configured_environment(30, 0, OpponentSpec::Idle, |world| {
        let own = world.seats[0].unit.expect("own hero");
        let enemy = world.seats[1].unit.expect("enemy hero");
        let enemy_position = Vec2::from_ints(ORIGIN.x.to_int() + reach, ORIGIN.y.to_int());
        let at = world.transform.get_mut(own).expect("own position");
        at.pos = ORIGIN;
        at.facing.brads = facing;
        world.transform.get_mut(enemy).expect("enemy position").pos = enemy_position;
        world.set_order(own, UnitOrder::Stand);
        world.set_order(
            enemy,
            match motion {
                Motion::Standing => UnitOrder::Stand,
                Motion::Crossing => UnitOrder::Move {
                    pos: Vec2::from_ints(
                        enemy_position.x.to_int(),
                        enemy_position.y.to_int() + CROSSING_TARGET_OFFSET,
                    ),
                },
            },
        );
        let book = world.abilities.get_mut(own).expect("abilities");
        for raze in &mut book.slots[..3] {
            raze.level = 1;
        }
        world.mana.get_mut(own).expect("mana").mana = Fixed::from_int(150);
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
    let mut environment = build_environment(
        seed,
        seed ^ 0x5eed,
        MapId(2),
        0,
        OpponentSpec::Teacher,
        Vec::new(),
    )
    .expect("teacher arena");
    environment.seats[macro_seat].teacher = Teacher::with_macro_hero_aim();
    let mut hero_aimed = [false; 2];
    let mut hero_casts = [0u64; 2];
    let mut winner = None;
    let mut ticks = 0u32;
    while winner.is_none() && ticks < TICK_CAP {
        let mut requests = Vec::with_capacity(2);
        for (index, seat) in environment.seats.iter_mut().enumerate() {
            let (action, space) = seat
                .teacher
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
