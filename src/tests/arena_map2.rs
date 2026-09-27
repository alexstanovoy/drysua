use bota_proto::{
    DamageKind, Fixed, ItemId, ItemSlot, MapId, Order, RejectReason, ServerMsg, SlotId, Target,
    Team, Vec2,
};
use bota_server::game::{ITEM_MANGO, ItemStack, Level, UnitOrder};

use super::{Arena, ArenaConfig, ArenaError, ArenaStep, Request};

#[test]
fn native_two_life_episode_respawns_then_freezes_win_loss_or_draw() {
    for victims in [&[0][..], &[1][..], &[0, 1][..]] {
        let (mut arena, _) = new_arena();
        for life in 0..2 {
            arena.configure_for_test(|world| {
                for seat in 0..2 {
                    let hero = world.seats[seat].unit.unwrap();
                    world.transform.get_mut(hero).unwrap().pos = Vec2::from_ints(9216, 9216);
                    world.set_order(hero, UnitOrder::Stand);
                }
                for &seat in victims {
                    world.push_hit(
                        None,
                        world.seats[seat].unit.unwrap(),
                        30_000,
                        DamageKind::Pure,
                    );
                }
            });
            let step = arena.step(&[None; 2]).unwrap();
            let winner = match victims {
                [0] => Team::Dire,
                [1] => Team::Radiant,
                _ => Team::Neutral,
            };
            assert_stream(&step, arena.tick(), (life == 1).then_some(winner));
            for &seat in victims {
                assert_eq!(arena.world.seats[seat].deaths, life + 1);
            }
            if life == 0 {
                arena.configure_for_test(|world| {
                    for &seat in victims {
                        world.seats[seat].respawn_left = 1;
                    }
                });
                let respawn = arena.step(&[None; 2]).unwrap();
                assert_stream(&respawn, arena.tick(), None);
                assert!(
                    victims
                        .iter()
                        .all(|&seat| arena.world.seats[seat].unit.is_some())
                );
            }
        }
        assert_frozen(&mut arena);
    }
}

#[test]
fn cap_tick_executes_private_purchase_then_draws_even_with_lethal_damage() {
    let (mut arena, _) = new_arena();
    arena.configure_for_test(|world| {
        world.tick = crate::MAP2_TICK_CAP - 1;
        world.seats[1].deaths = 1;
        world.push_hit(None, world.seats[1].unit.unwrap(), 30_000, DamageKind::Pure);
    });
    let step = arena
        .step(&[
            Some(request(Order::Buy {
                item: ItemId(ITEM_MANGO),
            })),
            None,
        ])
        .unwrap();
    assert_stream(&step, crate::MAP2_TICK_CAP, Some(Team::Neutral));
    let ServerMsg::Snapshot { view } = &step.messages[0][0] else {
        unreachable!()
    };
    assert_eq!(
        view.players[0].gold,
        Some(bota_server::game::rules::STARTING_GOLD - 65 + 1)
    );
    let ServerMsg::Events { events, .. } = &step.messages[0][1] else {
        unreachable!()
    };
    assert!(events.contains(&bota_proto::EventKind::ItemBought {
        slot: SlotId(0),
        item: ItemId(ITEM_MANGO)
    }));
    let ServerMsg::Events { events, .. } = &step.messages[1][1] else {
        unreachable!()
    };
    assert!(!events.iter().any(|event| matches!(
        event,
        bota_proto::EventKind::ItemBought {
            slot: SlotId(0),
            ..
        }
    )));
    assert_frozen(&mut arena);
}

#[test]
fn integer_mana_projection_does_not_determine_raw_mango_readiness() {
    let mut views = Vec::new();
    for deficit in [Fixed::ZERO, Fixed::from_ratio(1, 10)] {
        let (mut arena, _) = new_arena();
        arena.configure_for_test(|world| {
            for seat in 0..2 {
                let hero = world.seats[seat].unit.unwrap();
                world.level.insert(hero, Level(2));
                world.seats[seat].level = 2;
                world.inventory.get_mut(hero).unwrap().slots[0] = ItemStack::bought(
                    ItemId(bota_server::game::ITEM_MANGO),
                    SlotId(seat as u8),
                    world.tick,
                );
            }
        });
        // Settling would refill the fractional deficit before admission, hiding the regression.
        for seat in 0..2 {
            let hero = arena.world.seats[seat].unit.unwrap();
            arena.world.transform.get_mut(hero).unwrap().pos =
                Vec2::from_ints(9216 + seat as i32 * 200, 9216);
            let maximum = arena.world.stats.get(hero).unwrap().max_mana;
            let mana = &mut arena.world.mana.get_mut(hero).unwrap().mana;
            *mana -= deficit;
            assert_eq!(mana.to_int(), maximum.to_int());
        }
        views.push([
            arena.world.view(Team::Radiant),
            arena.world.view(Team::Dire),
        ]);
        let step = arena
            .step(
                &[Some(request(Order::Use {
                    slot: ItemSlot(0),
                    target: Target::None,
                })); 2],
            )
            .unwrap();
        for stream in &step.messages {
            if deficit == Fixed::ZERO {
                assert_eq!(
                    stream[0],
                    ServerMsg::OrderRejected {
                        seq: 17,
                        reason: RejectReason::NotReady
                    }
                );
            } else {
                assert!(matches!(stream[0], ServerMsg::Snapshot { .. }));
            }
        }
        // Regeneration precedes deferred use, so admission can still preserve the charge.
        for seat in &arena.world.seats {
            assert_eq!(
                arena.world.inventory.get(seat.unit.unwrap()).unwrap().slots[0]
                    .unwrap()
                    .charges,
                1
            );
        }
    }
    assert_eq!(views[0], views[1]);
}

fn assert_stream(step: &ArenaStep, tick: u32, expected: Option<Team>) {
    assert_eq!(step.messages.len(), 2);
    for stream in &step.messages {
        assert_eq!(stream.len(), 2 + usize::from(expected.is_some()));
        let ServerMsg::Snapshot { view } = &stream[0] else {
            panic!("Snapshot first")
        };
        let ServerMsg::Events {
            tick: event_tick, ..
        } = &stream[1]
        else {
            panic!("Events second")
        };
        assert_eq!(view.tick, tick);
        assert_eq!(*event_tick, tick);
        if let Some(expected) = expected {
            let ServerMsg::MatchOver { winner, stats } = &stream[2] else {
                panic!("MatchOver last")
            };
            assert_eq!(*winner, expected);
            assert_eq!(stats.duration, tick);
            for (player, stats) in view.players.iter().zip(&stats.slots) {
                assert_eq!(player.deaths, stats.deaths);
            }
        }
    }
}

fn assert_frozen(arena: &mut Arena) {
    let hash = arena.world.hash();
    for requests in [vec![None; 2], vec![]] {
        let error = arena.step(&requests).unwrap_err();
        assert_eq!(error, ArenaError::MatchOver);
        assert_eq!(error.to_string(), "arena cannot step after MatchOver");
        assert_eq!(arena.world.hash(), hash);
    }
}

fn new_arena() -> (Arena, crate::ArenaStart) {
    Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 0x0807_0605_0403_0201,
    })
    .unwrap()
}

fn request(order: Order) -> Request {
    Request {
        seq: 17,
        unit: None,
        order,
    }
}
