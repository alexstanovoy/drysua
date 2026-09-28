#[path = "action_execution.rs"]
mod action_execution;

#[path = "arena_feared.rs"]
mod feared_tests;

use crate::{Arena, ArenaConfig, ArenaError, Request};
use bota_proto::{ItemId, MapId, Order, RejectReason, ServerMsg, SlotId, Team, UnitKind};

#[test]
fn registered_maps_admit_every_seat_count_with_native_metadata_and_team_fog() {
    for map in [MapId(0), MapId(1), MapId(2)] {
        for seats in 2..=10 {
            let (arena, start) = Arena::new(ArenaConfig {
                seats,
                map,
                seed: 7,
            })
            .unwrap();
            assert_eq!(arena.tick(), 1);
            assert_eq!(arena.seat_count(), usize::from(seats));
            assert_eq!(start.messages.len(), usize::from(seats));
            for (index, messages) in start.messages.iter().enumerate() {
                let [
                    ServerMsg::MatchStart { info },
                    ServerMsg::Snapshot { view },
                    ServerMsg::Events { tick, .. },
                ] = messages.as_slice()
                else {
                    panic!("start must be exactly MatchStart, Snapshot, Events")
                };
                let slot = SlotId(u8::try_from(index).unwrap());
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
                for player in &view.players {
                    assert_eq!(player.gold.is_some(), player.team == team);
                    if player.team != team {
                        assert!(!has_owned_hero(view, player.slot));
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_seats_precede_map_validation_and_all_errors_are_descriptive() {
    for map in [MapId(0), MapId(1), MapId(2), MapId(3), MapId(u16::MAX)] {
        for seats in [0, 1, 2, 11, u8::MAX] {
            let result = Arena::new(ArenaConfig {
                seats,
                map,
                seed: 7,
            });
            if seats != 2 {
                let error = result.err().expect("invalid seats");
                assert_eq!(error, ArenaError::SeatCount { got: seats });
                assert_eq!(
                    error.to_string(),
                    format!("arena seat count must be between 2 and 10, got {seats}")
                );
            } else if map.0 > 2 {
                let error = result.err().expect("unsupported map");
                assert_eq!(error, ArenaError::UnsupportedMap { got: map });
                assert_eq!(
                    error.to_string(),
                    format!(
                        "arena map must be 0 (Dota), 1 (demo), or 2 (mid-only Dota), got {}",
                        map.0
                    )
                );
            } else {
                assert!(result.is_ok());
            }
        }
    }
}

#[test]
fn seeded_streams_are_deterministic_and_private_rejections_precede_complete_ticks() {
    for map in [MapId(0), MapId(1), MapId(2)] {
        let config = ArenaConfig {
            seats: 2,
            map,
            seed: 29,
        };
        let (mut first, start) = Arena::new(config).unwrap();
        let (mut second, repeated) = Arena::new(ArenaConfig {
            seats: 2,
            map,
            seed: 29,
        })
        .unwrap();
        assert_eq!(start, repeated);
        for tick in 2..=4 {
            let mut requests = [None; 2];
            if tick == 3 {
                requests[0] = Some(Request {
                    seq: 17,
                    unit: None,
                    order: Order::Buy {
                        item: ItemId(u16::MAX),
                    },
                });
            }
            let step = first.step(&requests).unwrap();
            assert_eq!(step, second.step(&requests).unwrap());
            for (seat, stream) in step.messages.iter().enumerate() {
                let stream = if tick == 3 && seat == 0 {
                    assert_eq!(
                        stream[0],
                        ServerMsg::OrderRejected {
                            seq: 17,
                            reason: RejectReason::UnknownItem
                        }
                    );
                    &stream[1..]
                } else {
                    stream.as_slice()
                };
                let [
                    ServerMsg::Snapshot { view },
                    ServerMsg::Events {
                        tick: event_tick, ..
                    },
                ] = stream
                else {
                    panic!("live tick must be exactly Snapshot, Events")
                };
                assert_eq!(view.tick, tick);
                assert_eq!(*event_tick, tick);
            }
        }
        let error = first.step(&[None]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "arena request count must equal seat count 2, got 1"
        );
        assert_eq!(first.tick(), 4);
        assert_eq!(
            first.step(&[None; 2]).unwrap(),
            second.step(&[None; 2]).unwrap()
        );
    }
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
