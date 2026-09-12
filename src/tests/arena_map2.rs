use bota_proto::{
    AbilitySlot, DamageKind, EventKind, Fixed, FrameReader, HeroId, ItemId, ItemSlot, MapId, Order,
    Pick, RejectReason, ServerMsg, SlotId, Target, Team, TickMode, UnitKind, Vec2, encode_frame,
};
use bota_server::game::{
    Command, Entity, Event, EventVisibility, ITEM_MANGO, ItemStack, Level, MatchConfig, UnitOrder,
    World, rules, wave_plan, wire_id,
};

use super::{Arena, ArenaConfig, ArenaError, ArenaStep, Request};

const MAP: MapId = crate::MAP2_ID;
const SEED: u64 = 0x0807_0605_0403_0201;
const TICK_CAP: u32 = crate::MAP2_TICK_CAP;

#[test]
fn map2_native_start_and_next_tick_match_bota_for_both_seats() {
    let (mut arena, mut world) = configured_pair(2, |_| {});

    let step = step_pair(&mut arena, &mut world, &[None, None]);

    assert_live(&step, &world);
    assert_eq!(world.tick, 2);
    assert_eq!(world.map.id, MAP);
}

#[test]
fn map2_wrong_request_counts_leave_the_world_unchanged() {
    let (mut arena, mut world) = configured_pair(2, |_| {});
    let before = arena.world.hash();
    for count in [0, 1, 3, 10] {
        let error = arena
            .step(&vec![None; count])
            .expect_err("wrong request count");

        assert_eq!(
            error,
            ArenaError::RequestCount {
                expected: 2,
                got: count
            }
        );
        assert_eq!(
            error.to_string(),
            format!("arena request count must equal seat count 2, got {count}")
        );
        assert_eq!(arena.world.hash(), before);
        assert_eq!(arena.tick(), 1);
    }
    let step = step_pair(&mut arena, &mut world, &[None, None]);
    assert_live(&step, &world);
}

#[test]
fn map2_first_hero_death_and_respawn_keep_both_native_streams_live() {
    for seat in 0..2 {
        let (mut arena, mut world) = configured_pair(2, |world| {
            place_hero_witnesses(world);
            lethal_hit(world, world.seats[seat].unit.expect("first life"));
        });
        let victim = world.seats[seat].unit.expect("queued victim");

        let step = step_pair(&mut arena, &mut world, &[None, None]);

        assert_live(&step, &world);
        assert_event_once(&step, &death(victim));
        assert_eq!(world.seats[seat].deaths, 1);
        assert_eq!(world.seats[seat].unit, None);
        arena.configure_for_test(|world| world.seats[seat].respawn_left = 1);
        world.seats[seat].respawn_left = 1;
        let step = step_pair(&mut arena, &mut world, &[None, None]);
        assert_live(&step, &world);
        assert_ne!(world.seats[seat].unit.expect("respawned hero"), victim);
    }
}

#[test]
fn map2_second_hero_death_sends_final_death_events_before_match_over_to_both_seats() {
    for (seat, winner) in [(0, Team::Dire), (1, Team::Radiant)] {
        let (mut arena, mut world) = configured_pair(2, |world| {
            place_hero_witnesses(world);
            world.seats[seat].deaths = 1;
            lethal_hit(world, world.seats[seat].unit.expect("second life"));
        });
        let victim = world.seats[seat].unit.expect("queued victim");

        let step = step_pair(&mut arena, &mut world, &[None, None]);

        assert_event_once(&step, &death(victim));
        assert_eq!(world.seats[seat].deaths, 2);
        assert_eq!(world.seats[seat].unit, None);
        assert_terminal(&step, &world, winner);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_two_team_deaths_across_different_seats_end_every_seat_stream() {
    for (seat, winner) in [(0, Team::Dire), (1, Team::Radiant)] {
        let (mut arena, mut world) = configured_pair(4, |world| {
            place_hero_witnesses(world);
            world.seats[seat + 2].deaths = 1;
            lethal_hit(
                world,
                world.seats[seat].unit.expect("teammate's first life"),
            );
        });
        let victim = world.seats[seat].unit.expect("queued victim");

        let step = step_pair(&mut arena, &mut world, &[None; 4]);

        assert_event_once(&step, &death(victim));
        assert_eq!(world.seats[seat].deaths, 1);
        assert_eq!(world.seats[seat + 2].deaths, 1);
        assert_terminal(&step, &world, winner);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_first_tower_loss_on_every_lane_sends_public_reason_before_match_over() {
    for (loser, winner) in [(Team::Radiant, Team::Dire), (Team::Dire, Team::Radiant)] {
        for lane in [rules::LANE_MID, rules::LANE_TOP, rules::LANE_BOT] {
            let (mut arena, mut world) = configured_pair(2, |world| {
                lethal_hit(world, tower(world, loser, lane));
            });
            let victim = tower(&world, loser, lane);

            let step = step_pair(&mut arena, &mut world, &[None, None]);

            assert_event_once(&step, &destroyed(victim, loser));
            assert!(world.seats.iter().all(|seat| seat.deaths == 0));
            assert_terminal(&step, &world, winner);
            assert_frozen(&mut arena, &mut world);
        }
    }
}

#[test]
fn map2_simultaneous_second_hero_deaths_draw_after_both_events_in_either_hit_order() {
    for order in [[0, 1], [1, 0]] {
        let (mut arena, mut world) = configured_pair(2, |world| {
            place_hero_witnesses(world);
            for seat in order {
                world.seats[seat].deaths = 1;
                lethal_hit(world, world.seats[seat].unit.expect("second life"));
            }
        });
        let victims = order.map(|seat| world.seats[seat].unit.expect("queued victim"));

        let step = step_pair(&mut arena, &mut world, &[None, None]);

        for victim in victims {
            assert_event_once(&step, &death(victim));
        }
        assert!(world.seats.iter().all(|seat| seat.deaths == 2));
        assert_terminal(&step, &world, Team::Neutral);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_simultaneous_tower_losses_draw_after_both_public_events_in_either_hit_order() {
    for order in [[Team::Radiant, Team::Dire], [Team::Dire, Team::Radiant]] {
        let (mut arena, mut world) = configured_pair(2, |world| {
            for team in order {
                lethal_hit(world, tower(world, team, rules::LANE_TOP));
            }
        });
        let reasons = order.map(|team| destroyed(tower(&world, team, rules::LANE_TOP), team));

        let step = step_pair(&mut arena, &mut world, &[None, None]);

        for reason in reasons {
            assert_event_once(&step, &reason);
        }
        assert_terminal(&step, &world, Team::Neutral);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_opposing_tower_and_second_hero_losses_draw_after_final_events_in_either_order() {
    for (tower_team, seat) in [(Team::Radiant, 1), (Team::Dire, 0)] {
        for reverse in [false, true] {
            let (mut arena, mut world) = configured_pair(2, |world| {
                place_hero_witnesses(world);
                world.seats[seat].deaths = 1;
                let mut victims = [
                    tower(world, tower_team, rules::LANE_BOT),
                    world.seats[seat].unit.expect("second life"),
                ];
                if reverse {
                    victims.reverse();
                }
                for victim in victims {
                    lethal_hit(world, victim);
                }
            });
            let building = tower(&world, tower_team, rules::LANE_BOT);
            let hero = world.seats[seat].unit.expect("queued victim");

            let step = step_pair(&mut arena, &mut world, &[None, None]);

            assert_event_once(&step, &destroyed(building, tower_team));
            assert_event_once(&step, &death(hero));
            assert_eq!(world.seats[seat].deaths, 2);
            assert_terminal(&step, &world, Team::Neutral);
            assert_frozen(&mut arena, &mut world);
        }
    }
}

#[test]
fn map2_simultaneous_first_hero_deaths_do_not_emit_match_over() {
    let (mut arena, mut world) = configured_pair(2, |world| {
        for seat in 0..2 {
            lethal_hit(world, world.seats[seat].unit.expect("first life"));
        }
    });

    let step = step_pair(&mut arena, &mut world, &[None, None]);

    assert!(world.seats.iter().all(|seat| seat.deaths == 1));
    assert_live(&step, &world);
}

#[test]
fn map2_courier_death_spends_no_hero_life_and_does_not_emit_match_over() {
    let (mut arena, mut world) = configured_pair(2, |world| {
        lethal_hit(world, world.seats[0].courier.expect("courier"));
    });

    let step = step_pair(&mut arena, &mut world, &[None, None]);

    assert!(world.seats.iter().all(|seat| seat.deaths == 0));
    assert_live(&step, &world);
}

#[test]
fn map2_cap_draw_follows_the_final_event_batch_but_never_arrives_early() {
    let (mut arena, mut world) = configured_pair(2, |world| world.tick = TICK_CAP - 2);

    let before_cap = step_pair(&mut arena, &mut world, &[None, None]);

    assert_live(&before_cap, &world);
    assert_eq!(world.tick, TICK_CAP - 1);
    let final_tick = step_pair(&mut arena, &mut world, &[None, None]);
    assert_eq!(world.tick, TICK_CAP);
    assert_eq!(
        TICK_CAP,
        rules::PREGAME_TICKS + bota_server::game::MAP2_GAME_TICKS
    );
    assert_eq!(TICK_CAP, bota_server::game::MAP2_TICK_CAP);
    assert_terminal(&final_tick, &world, Team::Neutral);
    for messages in &final_tick.messages {
        assert!(matches!(&messages[1], ServerMsg::Events { events, .. } if events.is_empty()));
    }
    assert_frozen(&mut arena, &mut world);
}

#[test]
fn map2_tower_loss_before_cap_wins_but_on_cap_draws_after_the_destruction_event() {
    for (loser, winner) in [(Team::Radiant, Team::Dire), (Team::Dire, Team::Radiant)] {
        for (tick, outcome) in [(TICK_CAP - 2, winner), (TICK_CAP - 1, Team::Neutral)] {
            let (mut arena, mut world) = configured_pair(2, |world| {
                world.tick = tick;
                lethal_hit(world, tower(world, loser, rules::LANE_BOT));
            });
            let victim = tower(&world, loser, rules::LANE_BOT);

            let step = step_pair(&mut arena, &mut world, &[None, None]);

            assert_event_once(&step, &destroyed(victim, loser));
            assert_eq!(world.tick, tick + 1);
            assert_terminal(&step, &world, outcome);
            assert_frozen(&mut arena, &mut world);
        }
    }
}

#[test]
fn map2_second_hero_loss_before_cap_wins_but_on_cap_draws_after_death_accounting() {
    for (seat, winner) in [(0, Team::Dire), (1, Team::Radiant)] {
        for (tick, outcome) in [(TICK_CAP - 2, winner), (TICK_CAP - 1, Team::Neutral)] {
            let (mut arena, mut world) = configured_pair(2, |world| {
                place_hero_witnesses(world);
                world.tick = tick;
                world.seats[seat].deaths = 1;
                lethal_hit(world, world.seats[seat].unit.expect("second life"));
            });
            let victim = world.seats[seat].unit.expect("queued victim");

            let step = step_pair(&mut arena, &mut world, &[None, None]);

            assert_event_once(&step, &death(victim));
            assert_eq!(world.tick, tick + 1);
            assert_eq!(world.seats[seat].deaths, 2);
            assert_terminal(&step, &world, outcome);
            assert_frozen(&mut arena, &mut world);
        }
    }
}

#[test]
fn map2_cap_simultaneous_second_hero_losses_still_account_for_both_final_deaths() {
    for order in [[0, 1], [1, 0]] {
        let (mut arena, mut world) = configured_pair(2, |world| {
            place_hero_witnesses(world);
            world.tick = TICK_CAP - 1;
            for seat in order {
                world.seats[seat].deaths = 1;
                lethal_hit(world, world.seats[seat].unit.expect("second life"));
            }
        });
        let victims = order.map(|seat| world.seats[seat].unit.expect("queued victim"));

        let step = step_pair(&mut arena, &mut world, &[None, None]);

        for victim in victims {
            assert_event_once(&step, &death(victim));
        }
        assert_eq!(world.tick, TICK_CAP);
        assert!(world.seats.iter().all(|seat| seat.deaths == 2));
        assert_terminal(&step, &world, Team::Neutral);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_cap_runs_purchases_gold_and_waves_before_drawing_even_with_both_towers_lost() {
    let (mut arena, mut world) = configured_pair(2, |world| {
        world.tick = TICK_CAP - 1;
        for team in [Team::Radiant, Team::Dire] {
            lethal_hit(world, tower(world, team, rules::LANE_TOP));
        }
    });
    let reasons = [Team::Radiant, Team::Dire]
        .map(|team| destroyed(tower(&world, team, rules::LANE_TOP), team));
    let buy = request(Order::Buy {
        item: ItemId(ITEM_MANGO),
    });

    let step = step_pair(&mut arena, &mut world, &[Some(buy); 2]);

    assert_terminal(&step, &world, Team::Neutral);
    for reason in reasons {
        assert_event_once(&step, &reason);
    }
    for (index, messages) in step.messages.iter().enumerate() {
        let ServerMsg::Snapshot { view } = &messages[0] else {
            panic!("final snapshot")
        };
        let ServerMsg::Events { events, .. } = &messages[1] else {
            panic!("final events")
        };
        let slot = SlotId(u8::try_from(index).expect("bounded seat index"));
        let purchases = events
            .iter()
            .filter(|event| matches!(event, EventKind::ItemBought { .. }))
            .collect::<Vec<_>>();
        assert_eq!(
            purchases,
            [&EventKind::ItemBought {
                slot,
                item: ItemId(ITEM_MANGO)
            }]
        );
        assert_eq!(
            view.players[index].gold,
            Some(rules::STARTING_GOLD - 65 + 1)
        );
        let hero = view
            .units
            .iter()
            .find(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(slot))
            .expect("buyer");
        assert_eq!(hero.items[0].expect("bought Mango").id, ItemId(ITEM_MANGO));
        assert_eq!(hero.items[0].expect("one charge").charges, Some(1));
    }
    let plan = wave_plan(bota_server::game::wave_at(TICK_CAP).expect("cap wave"));
    let creeps = world
        .entities
        .iter()
        .filter(|entity| world.march.get(*entity).is_some())
        .collect::<Vec<_>>();
    assert_eq!(
        creeps.len(),
        2 * (plan.melee + plan.ranged + plan.siege) as usize
    );
    assert!(creeps.iter().all(|entity| {
        world
            .lane
            .get(*entity)
            .is_some_and(|lane| lane.0 == rules::LANE_MID)
    }));
    assert_eq!(world.tick, TICK_CAP);
    assert_frozen(&mut arena, &mut world);
}

#[test]
fn map2_final_tick_rejections_precede_snapshot_events_and_match_over_only_for_the_issuer() {
    for seat in 0..2 {
        let (mut arena, mut world) = configured_pair(2, |world| world.tick = TICK_CAP - 1);
        let mut requests = [None; 2];
        requests[seat] = Some(request(Order::Buy {
            item: ItemId(u16::MAX),
        }));

        let step = step_pair(&mut arena, &mut world, &requests);

        assert_eq!(
            step.messages[seat][0],
            ServerMsg::OrderRejected {
                seq: 17,
                reason: RejectReason::UnknownItem
            }
        );
        assert_eq!(step.messages[seat].len(), 4);
        let mut without_rejection = step;
        without_rejection.messages[seat].remove(0);
        assert_terminal(&without_rejection, &world, Team::Neutral);
        assert_frozen(&mut arena, &mut world);
    }
}

#[test]
fn map2_mango_purchase_and_use_restore_enough_mana_for_a_previously_unaffordable_raze() {
    let (mut arena, mut world) = configured_pair(2, |world| {
        for seat in 0..2 {
            let hero = world.seats[seat].unit.expect("hero");
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.abilities.get_mut(hero).expect("abilities").slots[0].level = 1;
        }
    });
    let buy = request(Order::Buy {
        item: ItemId(ITEM_MANGO),
    });
    let cast = request(Order::Cast {
        slot: AbilitySlot(0),
        target: Target::None,
    });
    let use_mango = request(Order::Use {
        slot: ItemSlot(0),
        target: Target::None,
    });

    let bought = step_pair(&mut arena, &mut world, &[Some(buy); 2]);

    assert_live(&bought, &world);
    for seat in &world.seats {
        assert_eq!(
            world.validate_order(seat.slot, None, &cast.order),
            Err(RejectReason::NotEnoughMana)
        );
    }
    let used = step_pair(&mut arena, &mut world, &[Some(use_mango); 2]);
    assert_live(&used, &world);
    for (seat, stream) in world.seats.iter().zip(&used.messages) {
        let hero = seat.unit.expect("hero");
        let ServerMsg::Events { events, .. } = &stream[1] else {
            panic!("native Events")
        };
        assert!(events.contains(&EventKind::Healed {
            source: Some(wire_id(hero)),
            target: wire_id(hero),
            amount: 0,
            mana: 100,
        }));
        assert!(world.mana.get(hero).expect("restored mana").mana >= Fixed::from_int(100));
        assert_eq!(world.inventory.get(hero).expect("inventory").slots[0], None);
        assert_eq!(world.validate_order(seat.slot, None, &cast.order), Ok(()));
    }
    let cast = step_pair(&mut arena, &mut world, &[Some(cast); 2]);
    assert_live(&cast, &world);
}

#[test]
fn map2_cap_delivers_healed_mana_after_final_snapshot_and_before_native_draw() {
    let (mut arena, mut world) = configured_pair(2, |world| {
        world.tick = TICK_CAP - 1;
        for index in 0..2 {
            let hero = world.seats[index].unit.expect("hero");
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.inventory.get_mut(hero).expect("inventory").slots[0] =
                ItemStack::bought(ItemId(ITEM_MANGO), world.seats[index].slot, world.tick);
        }
    });
    let used = step_pair(
        &mut arena,
        &mut world,
        &[Some(request(Order::Use {
            slot: ItemSlot(0),
            target: Target::None,
        })); 2],
    );
    assert_terminal(&used, &world, Team::Neutral);
    for (stream, seat) in used.messages.iter().zip(&world.seats) {
        let hero = wire_id(seat.unit.expect("hero"));
        let ServerMsg::Snapshot { view } = &stream[0] else {
            panic!("Snapshot first")
        };
        assert!(
            view.units
                .iter()
                .find(|unit| unit.id == hero)
                .expect("hero")
                .mana
                >= 100
        );
        let ServerMsg::Events { events, .. } = &stream[1] else {
            panic!("Events second")
        };
        assert!(events.contains(&EventKind::Healed {
            source: Some(hero),
            target: hero,
            amount: 0,
            mana: 100
        }));
    }
    assert_frozen(&mut arena, &mut world);
}

#[test]
fn map2_identical_integer_mana_snapshots_can_hide_different_mango_readiness() {
    let (mut full_arena, mut full_world) =
        configured_pair(2, |world| prepare_mango(world, Fixed::ZERO));
    let (mut ready_arena, mut ready_world) =
        configured_pair(2, |world| prepare_mango(world, Fixed::EPSILON));
    let use_mango = request(Order::Use {
        slot: ItemSlot(0),
        target: Target::None,
    });
    for team in [Team::Radiant, Team::Dire] {
        assert_eq!(full_world.view(team), ready_world.view(team));
    }

    let full = step_pair(&mut full_arena, &mut full_world, &[Some(use_mango); 2]);
    let ready = step_pair(&mut ready_arena, &mut ready_world, &[Some(use_mango); 2]);

    assert_live(&ready, &ready_world);
    for (index, messages) in full.messages.iter().enumerate() {
        assert_eq!(messages.len(), 3);
        assert_eq!(
            messages[0],
            ServerMsg::OrderRejected {
                seq: 17,
                reason: RejectReason::NotReady
            }
        );
        let hero = full_world.seats[index].unit.expect("hero");
        let held = full_world
            .inventory
            .get(hero)
            .expect("full inventory")
            .slots[0]
            .expect("preserved charge");
        assert_eq!(held.charges, 1);
        assert_eq!(
            ready_world
                .inventory
                .get(hero)
                .expect("used inventory")
                .slots[0],
            None
        );
        assert_eq!(ready_world.mana.get(hero), full_world.mana.get(hero));
    }
}

#[test]
fn map1_second_hero_death_and_first_tower_loss_no_longer_end_the_demo() {
    for hero_loss in [true, false] {
        let (mut arena, _) = Arena::new(ArenaConfig {
            seats: 2,
            map: MapId(1),
            seed: SEED,
        })
        .expect("honest historical map metadata");
        let mut victim = None;
        arena.configure_for_test(|world| {
            let target = if hero_loss {
                world.seats[0].deaths = 1;
                world.seats[0].unit.expect("second life")
            } else {
                tower(world, Team::Radiant, rules::LANE_MID)
            };
            victim = Some(target);
            lethal_hit(world, target);
        });
        let step = arena
            .step(&[None, None])
            .expect("loss tick is live on Map1");
        assert_live(&step, &arena.world);
        assert!(!arena.world.alive(victim.expect("victim")));
        if hero_loss {
            assert_eq!(arena.world.seats[0].deaths, 2);
        } else {
            assert_event_once(&step, &destroyed(victim.expect("tower"), Team::Radiant));
        }
        assert!(arena.step(&[None, None]).is_ok());
    }
}

fn prepare_mango(world: &mut World, deficit: Fixed) {
    assert!(deficit == Fixed::ZERO || deficit == Fixed::EPSILON);
    for seat in 0..2 {
        let hero = world.seats[seat].unit.expect("hero");
        world.level.insert(hero, Level(2));
        world.seats[seat].level = 2;
        world.inventory.get_mut(hero).expect("inventory").slots[0] =
            ItemStack::bought(ItemId(ITEM_MANGO), world.seats[seat].slot, world.tick);
    }
    world.settle();
    for seat in 0..2 {
        let hero = world.seats[seat].unit.expect("hero");
        world.fill_pools(hero);
        let pool = &mut world.mana.get_mut(hero).expect("mana").mana;
        assert!(*pool > Fixed::from_int(pool.to_int()) + Fixed::EPSILON);
        *pool -= deficit;
    }
}

fn configured_pair(seats: u8, configure: impl Fn(&mut World)) -> (Arena, World) {
    assert!((2..=10).contains(&seats));
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats,
        map: MAP,
        seed: SEED,
    })
    .expect("Map2 arena");
    let config = MatchConfig {
        match_id: SEED,
        master_key: [
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
        picks: (0..seats)
            .map(|index| Pick {
                slot: SlotId(index),
                team: if index.is_multiple_of(2) {
                    Team::Radiant
                } else {
                    Team::Dire
                },
                hero: HeroId(2),
            })
            .collect(),
        map: MAP,
        tick_rate: 30,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 150,
    };
    let mut world = World::for_match(&config, config.rng());
    let events = world.advance(&[]);
    let mut expected = native_tick(&world, &events);
    for stream in &mut expected {
        stream.insert(
            0,
            ServerMsg::MatchStart {
                info: config.info(),
            },
        );
    }
    assert_native_streams(&start.messages, &expected);
    assert_eq!(world.hash(), arena.world.hash());
    arena.configure_for_test(&configure);
    configure(&mut world);
    world.settle();
    world.lay_passability();
    assert_eq!(world.victor(), None);
    assert_eq!(world.hash(), arena.world.hash());
    (arena, world)
}

fn step_pair(arena: &mut Arena, world: &mut World, requests: &[Option<Request>]) -> ArenaStep {
    assert_eq!(requests.len(), world.seats.len());
    let mut commands = Vec::new();
    let mut rejected = vec![None; requests.len()];
    for (index, request) in requests.iter().enumerate() {
        let Some(request) = request else { continue };
        let slot = world.seats[index].slot;
        match world.validate_order(slot, request.unit, &request.order) {
            Ok(()) => commands.push(Command {
                slot,
                unit: request.unit,
                order: request.order,
            }),
            Err(reason) => {
                rejected[index] = Some(ServerMsg::OrderRejected {
                    seq: request.seq,
                    reason,
                })
            }
        }
    }

    let events = world.advance(&commands);
    let step = arena.step(requests).expect("not yet terminal");

    let mut expected = native_tick(world, &events);
    for (stream, rejection) in expected.iter_mut().zip(rejected) {
        if let Some(rejection) = rejection {
            stream.insert(0, rejection);
        }
    }
    assert_native_streams(&step.messages, &expected);
    assert_eq!(arena.world.victor(), world.victor());
    assert_eq!(arena.world.hash(), world.hash());
    assert_eq!(arena.world.view_full(), world.view_full());
    assert_eq!(arena.world.match_stats(), world.match_stats());
    step
}

fn native_tick(world: &World, events: &[Event]) -> Vec<Vec<ServerMsg>> {
    assert!((2..=10).contains(&world.seats.len()));
    world
        .seats
        .iter()
        .map(|seat| {
            let visible = events
                .iter()
                .filter(|event| {
                    event.visible_to == EventVisibility::Everyone
                        || event.visible_to == EventVisibility::OneTeam(seat.team)
                })
                .map(|event| event.kind.clone())
                .collect();
            let mut stream = vec![
                ServerMsg::Snapshot {
                    view: world.view(seat.team),
                },
                ServerMsg::Events {
                    tick: world.tick,
                    events: visible,
                },
            ];
            if let Some(winner) = world.victor() {
                stream.push(ServerMsg::MatchOver {
                    winner,
                    stats: world.match_stats(),
                });
            }
            assert!((2..=3).contains(&stream.len()));
            stream
        })
        .collect()
}

fn assert_native_streams(actual: &[Vec<ServerMsg>], expected: &[Vec<ServerMsg>]) {
    assert_eq!(actual.len(), expected.len());
    assert!((2..=10).contains(&actual.len()));
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        assert!((2..=4).contains(&actual.len()));
        let mut arena_bytes = Vec::new();
        let mut world_bytes = Vec::new();
        for (actual, expected) in actual.iter().zip(expected) {
            encode_frame(actual, &mut arena_bytes).expect("Arena native frame");
            encode_frame(expected, &mut world_bytes).expect("bota native frame");
        }
        assert_eq!(arena_bytes, world_bytes);
        let mut reader = FrameReader::new();
        reader.push(&arena_bytes[..3]);
        assert_eq!(
            reader.next_message::<ServerMsg>().expect("partial prefix"),
            None
        );
        reader.push(&arena_bytes[3..]);
        for message in expected {
            assert_eq!(
                reader.next_message::<ServerMsg>().expect("native decode"),
                Some(message.clone())
            );
        }
        assert_eq!(
            reader.next_message::<ServerMsg>().expect("end of stream"),
            None
        );
        assert_eq!(reader.buffered(), 0);
    }
}

fn assert_live(step: &ArenaStep, world: &World) {
    assert_eq!(world.victor(), None);
    assert_eq!(step.messages.len(), world.seats.len());
    for messages in &step.messages {
        let [ServerMsg::Snapshot { view }, ServerMsg::Events { tick, .. }] = messages.as_slice()
        else {
            panic!("live tick must be exactly Snapshot, Events");
        };
        assert_eq!(view.tick, world.tick);
        assert_eq!(*tick, world.tick);
    }
}

fn assert_terminal(step: &ArenaStep, world: &World, expected: Team) {
    assert_eq!(world.victor(), Some(expected));
    assert_eq!(step.messages.len(), world.seats.len());
    for (messages, seat) in step.messages.iter().zip(&world.seats) {
        let [
            ServerMsg::Snapshot { view },
            ServerMsg::Events { tick, .. },
            ServerMsg::MatchOver { winner, stats },
        ] = messages.as_slice()
        else {
            panic!("terminal tick must be exactly Snapshot, Events, MatchOver");
        };
        assert_eq!(view.viewer, Some(seat.team));
        assert_eq!(*tick, world.tick);
        assert_eq!(view.tick, world.tick);
        assert_eq!(*winner, expected);
        assert_eq!(stats.duration, world.tick);
        assert_eq!(*stats, world.match_stats());
        for (player, stats) in view.players.iter().zip(&stats.slots) {
            assert_eq!(player.slot, stats.slot);
            assert_eq!(player.deaths, stats.deaths);
        }
    }
}

fn assert_event_once(step: &ArenaStep, expected: &EventKind) {
    assert!((2..=10).contains(&step.messages.len()));
    for messages in &step.messages {
        let ServerMsg::Events { events, .. } = &messages[1] else {
            panic!("events immediately after snapshot")
        };
        assert_eq!(
            events.iter().filter(|event| *event == expected).count(),
            1,
            "final event {expected:?}"
        );
    }
}

fn assert_frozen(arena: &mut Arena, world: &mut World) {
    assert!(world.victor().is_some());
    let hash = world.hash();
    let view = world.view_full();
    let stats = world.match_stats();
    let buy = request(Order::Buy {
        item: ItemId(ITEM_MANGO),
    });
    let invalid = request(Order::Buy {
        item: ItemId(u16::MAX),
    });
    for requests in [
        vec![None; arena.seat_count()],
        vec![Some(buy); arena.seat_count()],
        vec![Some(invalid); arena.seat_count()],
        Vec::new(),
    ] {
        let error = arena
            .step(&requests)
            .expect_err("no step or validation after MatchOver");
        assert_eq!(error, ArenaError::MatchOver);
        assert_eq!(error.to_string(), "arena cannot step after MatchOver");
        assert_eq!(arena.world.hash(), hash);
        assert_eq!(arena.world.view_full(), view);
        assert_eq!(arena.world.match_stats(), stats);
    }
    assert!(
        world
            .advance(&[Command {
                slot: SlotId(0),
                unit: None,
                order: buy.order
            }])
            .is_empty()
    );
    assert!(world.step().is_empty());
    assert_eq!(world.hash(), hash);
    assert_eq!(world.view_full(), view);
    assert_eq!(world.match_stats(), stats);
}

fn place_hero_witnesses(world: &mut World) {
    assert!((2..=10).contains(&world.seats.len()));
    for index in 0..world.seats.len() {
        for unit in [world.seats[index].unit, world.seats[index].courier] {
            let unit = unit.expect("living hero or courier");
            world.transform.get_mut(unit).expect("position").pos = Vec2::from_ints(9_216, 9_216);
            world.set_order(unit, UnitOrder::Stand);
            assert!(world.alive(unit));
        }
    }
}

fn tower(world: &World, team: Team, lane: u8) -> Entity {
    assert_ne!(team, Team::Neutral);
    assert!(lane < 3);
    world
        .entities
        .iter()
        .find(|entity| {
            world.kind.get(*entity) == Some(&UnitKind::Tower)
                && world.team.get(*entity) == Some(&team)
                && world.lane.get(*entity).is_some_and(|value| value.0 == lane)
                && world.tier.get(*entity).is_some_and(|value| value.0 == 1)
        })
        .expect("tier-one tower on requested lane")
}

fn lethal_hit(world: &mut World, victim: Entity) {
    assert!(world.alive(victim));
    assert!(
        !world
            .stats
            .get(victim)
            .expect("settled victim")
            .invulnerable
    );
    world.push_hit(None, victim, 30_000, DamageKind::Pure);
}

fn death(victim: Entity) -> EventKind {
    EventKind::Died {
        unit: wire_id(victim),
        killer: None,
        denied: false,
        gold: 0,
    }
}

fn destroyed(victim: Entity, team: Team) -> EventKind {
    EventKind::StructureDestroyed {
        unit: wire_id(victim),
        team,
    }
}

fn request(order: Order) -> Request {
    Request {
        seq: 17,
        unit: None,
        order,
    }
}
