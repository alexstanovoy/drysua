use bota_proto::{ItemId, MapId, Order, RejectReason, ServerMsg, SlotId, Team, UnitKind};

use crate::{Arena, ArenaConfig, ArenaError, Request};

#[test]
fn arena_admits_all_registered_maps_and_every_supported_seat_count() {
    for map in [MapId(0), MapId(1), MapId(2)] {
        for seats in 2..=10 {
            let (arena, start) = Arena::new(ArenaConfig {
                seats,
                map,
                seed: 7,
            })
            .expect("registered map and supported seat count");

            assert_eq!(arena.seat_count(), usize::from(seats));
            assert_eq!(start.messages.len(), usize::from(seats));
            for (index, messages) in start.messages.iter().enumerate() {
                let [
                    ServerMsg::MatchStart { info },
                    ServerMsg::Snapshot { view },
                    ServerMsg::Events { tick, .. },
                ] = messages.as_slice()
                else {
                    panic!("start must contain exactly MatchStart, Snapshot, Events");
                };
                let slot = SlotId(u8::try_from(index).expect("bounded seat index"));
                let team = if index.is_multiple_of(2) {
                    Team::Radiant
                } else {
                    Team::Dire
                };
                assert_eq!(info.map, map);
                assert_eq!(info.match_id, 7);
                assert_eq!(info.picks.len(), usize::from(seats));
                assert_eq!(info.picks[index].slot, slot);
                assert_eq!(info.picks[index].team, team);
                assert_eq!(info.picks[index].hero, crate::SHADOW_FIEND);
                assert_eq!(view.viewer, Some(team));
                assert!(has_owned_hero(view, slot));
                assert_eq!(view.tick, 1);
                assert_eq!(*tick, 1);
            }
        }
    }
}

#[test]
fn arena_rejects_unknown_maps_instead_of_silently_running_map0() {
    for map in [MapId(3), MapId(u16::MAX)] {
        let error = Arena::new(ArenaConfig {
            seats: 2,
            map,
            seed: 7,
        })
        .err()
        .expect("unknown map must fail admission");

        assert_eq!(error, ArenaError::UnsupportedMap { got: map });
        assert_eq!(
            error.to_string(),
            format!(
                "arena map must be 0 (Dota), 1 (demo), or 2 (mid-only Dota), got {}",
                map.0
            )
        );
    }
}

#[test]
fn arena_rejects_invalid_seats_before_validating_the_map() {
    for map in [MapId(0), MapId(1), MapId(2), MapId(3)] {
        for seats in [0, 1, 11, u8::MAX] {
            let error = Arena::new(ArenaConfig {
                seats,
                map,
                seed: 7,
            })
            .err()
            .expect("invalid seat count must fail admission");

            assert_eq!(error, ArenaError::SeatCount { got: seats });
            assert_eq!(
                error.to_string(),
                format!("arena seat count must be between 2 and 10, got {seats}")
            );
        }
    }
}

#[test]
fn arena_starts_with_match_start_then_snapshot_tick_one() {
    let (arena, start) = new_arena(11);

    assert_eq!(arena.tick(), 1);
    for messages in start.messages {
        assert!(matches!(messages[0], ServerMsg::MatchStart { .. }));
        assert!(matches!(
            messages[1],
            ServerMsg::Snapshot { ref view } if view.tick == 1
        ));
        assert!(matches!(messages[2], ServerMsg::Events { tick: 1, .. }));
    }
}

#[test]
fn arena_emits_an_event_batch_as_the_end_of_every_visible_tick() {
    let (mut arena, _) = new_arena(17);

    let step = arena.step(&[None, None]).expect("arena step");

    for messages in step.messages {
        assert!(matches!(messages[0], ServerMsg::Snapshot { .. }));
        assert!(matches!(messages[1], ServerMsg::Events { tick: 2, .. }));
    }
}

#[test]
fn arenas_with_same_seed_produce_identical_streams() {
    let (mut first, first_start) = new_arena(29);
    let (mut second, second_start) = new_arena(29);

    assert_eq!(first_start, second_start);
    for _ in 0..3 {
        let first_step = first.step(&[None, None]).expect("first arena steps");
        let second_step = second.step(&[None, None]).expect("second arena steps");
        assert_eq!(first_step, second_step);
    }
}

#[test]
fn arena_projects_each_snapshot_through_its_team_fog() {
    let (_, start) = new_arena(41);
    let radiant = snapshot(&start.messages[0]);
    let dire = snapshot(&start.messages[1]);

    assert_eq!(radiant.viewer, Some(Team::Radiant));
    assert_eq!(dire.viewer, Some(Team::Dire));
    assert!(has_owned_hero(radiant, SlotId(0)));
    assert!(has_owned_hero(dire, SlotId(1)));
    assert!(!has_owned_hero(radiant, SlotId(1)));
    assert!(!has_owned_hero(dire, SlotId(0)));
    assert!(radiant.players[0].gold.is_some());
    assert!(radiant.players[1].gold.is_none());
    assert!(dire.players[0].gold.is_none());
    assert!(dire.players[1].gold.is_some());
}

#[test]
fn arena_places_rejection_before_next_snapshot() {
    let (mut arena, _) = new_arena(53);
    let request = Request {
        seq: 17,
        unit: None,
        order: Order::Buy {
            item: ItemId(u16::MAX),
        },
    };

    let step = arena
        .step(&[Some(request), None])
        .expect("rejected request still advances");

    assert_eq!(
        step.messages[0][0],
        ServerMsg::OrderRejected {
            seq: 17,
            reason: RejectReason::UnknownItem,
        }
    );
    assert!(matches!(step.messages[0][1], ServerMsg::Snapshot { .. }));
    assert!(matches!(step.messages[1][0], ServerMsg::Snapshot { .. }));
}

#[test]
fn arena_rejects_invalid_seat_counts_with_exact_messages() {
    let too_few = Arena::new(ArenaConfig {
        seats: 1,
        map: MapId(1),
        seed: 1,
    })
    .err()
    .expect("one seat must fail");
    let too_many = Arena::new(ArenaConfig {
        seats: 11,
        map: MapId(1),
        seed: 1,
    })
    .err()
    .expect("eleven seats must fail");

    assert_eq!(
        too_few.to_string(),
        "arena seat count must be between 2 and 10, got 1"
    );
    assert_eq!(
        too_many.to_string(),
        "arena seat count must be between 2 and 10, got 11"
    );
}

#[test]
fn arena_rejects_wrong_request_count_with_exact_message() {
    let (mut arena, _) = new_arena(67);

    let error = arena
        .step(&[None])
        .expect_err("one request entry for two seats must fail");

    assert_eq!(
        error.to_string(),
        "arena request count must equal seat count 2, got 1"
    );
}

fn new_arena(seed: u64) -> (Arena, crate::ArenaStart) {
    Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed,
    })
    .expect("valid arena")
}

fn snapshot(messages: &[ServerMsg]) -> &bota_proto::WorldView {
    messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .expect("seat stream has a snapshot")
}

fn has_owned_hero(view: &bota_proto::WorldView, slot: SlotId) -> bool {
    view.units
        .iter()
        .any(|unit| unit.kind == UnitKind::Hero && unit.owner == Some(slot))
}
