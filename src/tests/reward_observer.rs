use bota_proto::{MatchStats, PlayerId, ServerMsg, SlotId, SlotStats, Team, TickMode};

use super::map2_reward::{match_info, snapshot};
use crate::reward_observer::{BoundedInput, Observer, consume, read_message};

#[cfg(feature = "builtin")]
#[test]
fn native_arena_frames_close_observers_with_seat_relative_wins_losses_and_draws() {
    use crate::Map2RewardEnd;
    for loser in [Some(Team::Radiant), Some(Team::Dire), None] {
        let streams = tower_ending_streams(loser);
        for (index, team) in [Team::Radiant, Team::Dire].into_iter().enumerate() {
            let mut wire = Vec::new();
            bota_proto::encode_frame(&welcome(index as u8), &mut wire).unwrap();
            for message in &streams[index] {
                bota_proto::encode_frame(message, &mut wire).unwrap();
            }
            let mut observer = Observer::new();
            consume(&mut Fragmented(&wire), &mut observer, 30, &mut Vec::new()).unwrap();
            let (end, terminal) = match loser {
                None => (Map2RewardEnd::Draw, -0.5),
                Some(loser) if loser == team => (Map2RewardEnd::Loss, -1.0),
                _ => (Map2RewardEnd::Win, 1.0),
            };
            let case = format!("loser {loser:?}, observer {team:?}");
            assert_eq!(observer.last_interval.end, Some(end), "{case}");
            assert_eq!(observer.last_interval.terminal, terminal, "{case}");
            assert!(
                observer
                    .report(None)
                    .contains("\"complete\":true,\"valid\":true"),
                "{case}"
            );
            assert_eq!(
                observer.observe(empty_events()).unwrap_err().to_string(),
                "server message after MatchOver",
                "{case}"
            );
        }
    }
}

/// Both seats' complete native Map2 streams: 33 ticks with a mango purchase on the last, then
/// a tick that destroys the tier-one lane-zero tower of `loser`, or both teams' for a draw.
#[cfg(feature = "builtin")]
fn tower_ending_streams(loser: Option<Team>) -> Vec<Vec<ServerMsg>> {
    use crate::{Arena, ArenaConfig};
    use bota_proto::{DamageKind, ItemId, MapId, UnitKind};
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 41,
    })
    .unwrap();
    for tick in 2..=33 {
        let purchase = (tick == 33).then_some(crate::Request {
            seq: tick,
            unit: None,
            order: bota_proto::Order::Buy {
                item: ItemId(bota_server::game::ITEM_MANGO),
            },
        });
        let step = arena.step(&[purchase; 2]).unwrap();
        for (stream, messages) in start.messages.iter_mut().zip(step.messages) {
            stream.extend(messages);
        }
    }
    arena.configure_for_test(|world| {
        let towers: Vec<_> = world
            .entities
            .iter()
            .filter(|entity| {
                world.kind.get(*entity) == Some(&UnitKind::Tower)
                    && world.tier.get(*entity).is_some_and(|tier| tier.0 == 1)
                    && world.lane.get(*entity).is_some_and(|lane| lane.0 == 0)
                    && loser.is_none_or(|team| world.team.get(*entity) == Some(&team))
            })
            .collect();
        assert_eq!(towers.len(), if loser.is_some() { 1 } else { 2 });
        for tower in towers {
            world.push_hit(None, tower, 30_000, DamageKind::Pure);
        }
    });
    let final_tick = arena.step(&[None; 2]).unwrap();
    for (stream, messages) in start.messages.iter_mut().zip(final_tick.messages) {
        stream.extend(messages);
    }
    start.messages
}

#[test]
fn invalid_stream_transitions_never_invent_terminal_reward_and_allow_pair_completion() {
    for (scenario, expected) in [
        (0, "first Snapshot must be tick 1; missing initial baseline"),
        (1, "Snapshot viewer differs from assigned participant team"),
        (2, "Map2 reward: Snapshot arrived before pending Events"),
        (3, "Map2 reward: Events without pending Snapshot"),
        (
            4,
            "MatchOver without complete final Snapshot/Events duration",
        ),
        (5, "MatchOver must contain both participant statistics"),
    ] {
        let mut observer = started(if scenario == 1 { 1 } else { 0 });
        if scenario >= 2 {
            observer
                .observe(ServerMsg::Snapshot { view: snapshot(1) })
                .unwrap();
            if scenario == 3 || scenario == 5 {
                observer.observe(empty_events()).unwrap();
            }
        }
        let invalid = match scenario {
            0 | 2 => ServerMsg::Snapshot { view: snapshot(2) },
            1 => ServerMsg::Snapshot { view: snapshot(1) },
            3 => empty_events(),
            _ => ServerMsg::MatchOver {
                winner: Team::Radiant,
                stats: if scenario == 5 {
                    MatchStats {
                        duration: 1,
                        slots: vec![],
                    }
                } else {
                    statistics(1)
                },
            },
        };
        assert_eq!(observer.observe(invalid).unwrap_err().to_string(), expected);
        assert_eq!(observer.last_interval.terminal, 0.0);
        if scenario == 2 || scenario == 4 {
            observer
                .observe(ServerMsg::OrderRejected {
                    seq: 1,
                    reason: bota_proto::RejectReason::NotEnoughMana,
                })
                .unwrap();
            observer.observe(empty_events()).unwrap();
            assert_eq!(observer.last_interval.ticks, 0);
        }
    }
}

#[test]
fn eof_at_empty_pending_or_complete_prefix_is_invalid_not_timecap_or_win() {
    for pairs in 0..3 {
        let mut observer = started(0);
        let mut wire = Vec::new();
        if pairs > 0 {
            bota_proto::encode_frame(&ServerMsg::Snapshot { view: snapshot(1) }, &mut wire)
                .unwrap();
        }
        if pairs == 2 {
            bota_proto::encode_frame(&empty_events(), &mut wire).unwrap();
        }
        let error = consume(&mut wire.as_slice(), &mut observer, 30, &mut Vec::new()).unwrap_err();
        assert_eq!(error.to_string(), "EOF before MatchOver");
        let report = observer.report(Some(&error.to_string()));
        assert!(report.contains("\"complete\":false,\"valid\":false"));
        assert!(report.contains("\"outcome\":null"));
        assert!(report.contains("\"terminal\":0"));
        assert!(!report.contains("TimeCap"));
        assert_eq!(observer.last_interval.terminal, 0.0);
    }
}

#[test]
fn frame_reader_accepts_coalescing_and_fragmentation_but_rejects_noncanonical_boundaries() {
    let message = empty_events();
    let frame = bota_proto::encode_frame_to_vec(&message).unwrap();
    let mut frames = frame.clone();
    frames.extend_from_slice(&frame);
    let mut input = Fragmented(&frames);
    for _ in 0..2 {
        assert_eq!(read_message(&mut input).unwrap(), Some(message.clone()));
    }
    assert_eq!(read_message(&mut input).unwrap(), None);
    for length in [1, 3, frame.len() - 1] {
        let expected = if length < 4 {
            "truncated observer frame prefix"
        } else {
            "truncated observer frame payload"
        };
        assert_eq!(
            read_message(&mut &frame[..length]).unwrap_err().to_string(),
            expected
        );
    }
    for length in [0, (bota_proto::MAX_PAYLOAD_LEN + 1) as u32, u32::MAX] {
        assert_eq!(
            read_message(&mut length.to_le_bytes().as_slice())
                .unwrap_err()
                .to_string(),
            "observer frame length outside 1..=MAX_PAYLOAD_LEN"
        );
    }
    let mut trailing = frame;
    let length = u32::from_le_bytes(trailing[..4].try_into().unwrap()) + 1;
    trailing[..4].copy_from_slice(&length.to_le_bytes());
    trailing.push(0);
    assert_eq!(
        read_message(&mut trailing.as_slice())
            .unwrap_err()
            .to_string(),
        "noncanonical or trailing observer payload"
    );
}

#[test]
fn input_cap_is_an_error_not_synthetic_eof_and_teacher_rejects_weights() {
    use std::io::Read;
    let mut input = BoundedInput::new(&b"abc"[..], 2);
    let mut first = [0; 2];
    input.read_exact(&mut first).unwrap();
    assert_eq!(&first, b"ab");
    assert_eq!(
        input.read(&mut [0]).unwrap_err().to_string(),
        "observer byte limit exceeded"
    );
    let error = crate::cli::play_policy_for_test([
        "drysua",
        "--policy",
        "teacher",
        "--weights-directory",
        ".",
    ])
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("rule policies forbid --weights-directory")
    );
}

fn welcome(slot: u8) -> ServerMsg {
    ServerMsg::Welcome {
        player_id: PlayerId(1),
        slot: Some(SlotId(slot)),
        tick_rate: 30,
        mode: TickMode::Lockstep,
    }
}

fn started(slot: u8) -> Observer {
    let mut observer = Observer::new();
    observer.observe(welcome(slot)).unwrap();
    observer
        .observe(ServerMsg::MatchStart { info: match_info() })
        .unwrap();
    observer
}

fn empty_events() -> ServerMsg {
    ServerMsg::Events {
        tick: 1,
        events: vec![],
    }
}

fn statistics(duration: u32) -> MatchStats {
    MatchStats {
        duration,
        slots: (0..2)
            .map(|slot| SlotStats {
                slot: SlotId(slot),
                kills: 0,
                deaths: 0,
                assists: 0,
                last_hits: 0,
                denies: 0,
                net_worth: 0,
                hero_damage: 0,
                structure_damage: 0,
            })
            .collect(),
    }
}

struct Fragmented<'a>(&'a [u8]);

impl std::io::Read for Fragmented<'_> {
    fn read(&mut self, target: &mut [u8]) -> std::io::Result<usize> {
        let count = usize::from(!self.0.is_empty() && !target.is_empty());
        target[..count].copy_from_slice(&self.0[..count]);
        self.0 = &self.0[count..];
        Ok(count)
    }
}
