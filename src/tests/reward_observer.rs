use bota_proto::{
    EventKind, ItemId, MatchStats, PlayerId, ServerMsg, SlotId, SlotStats, Team, TickMode, Vec2,
};

use super::map2_reward::{damage, match_info, own_hero, snapshot};
use crate::reward_observer::{BoundedInput, Observer, consume, read_message};
use crate::{Map2Reward, Map2RewardEnd};

fn started(slot: u8) -> Observer {
    let mut observer = Observer::new();
    observer
        .observe(ServerMsg::Welcome {
            player_id: PlayerId(1),
            slot: Some(SlotId(slot)),
            tick_rate: 30,
            mode: TickMode::Lockstep,
        })
        .unwrap();
    observer
        .observe(ServerMsg::MatchStart { info: match_info() })
        .unwrap();
    observer
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

#[test]
fn copied_seat_components_equal_production_including_native_draw_closure() {
    let mut observer = started(0);
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    for tick in 1..=3 {
        let view = snapshot(tick);
        let events = if tick == 2 {
            vec![damage(1, 2, 67)]
        } else {
            vec![]
        };
        reward.observe_snapshot(&view).unwrap();
        reward.observe_events(tick, &events).unwrap();
        observer.observe(ServerMsg::Snapshot { view }).unwrap();
        observer
            .observe(ServerMsg::Events { tick, events })
            .unwrap();
        assert_eq!(observer.last_interval, reward.take_interval().unwrap());
    }
    observer
        .observe(ServerMsg::MatchOver {
            winner: Team::Neutral,
            stats: statistics(3),
        })
        .unwrap();
    assert_eq!(
        observer.last_interval,
        reward.finish(Map2RewardEnd::Draw).unwrap()
    );
    assert_eq!(observer.last_interval.terminal, 0.0);
}

#[test]
fn lost_pair_gap_and_wrong_viewer_are_rejected_without_terminal() {
    let mut observer = started(0);
    observer
        .observe(ServerMsg::Snapshot { view: snapshot(1) })
        .unwrap();
    let error = observer
        .observe(ServerMsg::Snapshot { view: snapshot(2) })
        .unwrap_err();
    assert!(error.to_string().contains("pending Events"));
    assert_eq!(observer.last_interval.terminal, 0.0);
    let mut observer = started(1);
    assert!(
        observer
            .observe(ServerMsg::Snapshot { view: snapshot(1) })
            .unwrap_err()
            .to_string()
            .contains("viewer")
    );
}

#[test]
fn framed_reader_handles_coalescing_and_rejects_truncation() {
    let message = ServerMsg::Events {
        tick: 1,
        events: vec![],
    };
    let frame = bota_proto::encode_frame_to_vec(&message).unwrap();
    let mut frames = frame.clone();
    frames.extend_from_slice(&frame);
    let mut input = frames.as_slice();
    assert_eq!(read_message(&mut input).unwrap(), Some(message.clone()));
    assert_eq!(read_message(&mut input).unwrap(), Some(message));
    assert_eq!(read_message(&mut input).unwrap(), None);
    assert!(
        read_message(&mut &frame[..frame.len() - 1])
            .unwrap_err()
            .to_string()
            .contains("truncated observer frame")
    );
}

#[test]
fn teacher_cli_rejects_weights_rather_than_ignoring_them() {
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
            .contains("teacher forbids --weights-directory")
    );
}

#[test]
fn winner_is_relative_to_welcome_seat_for_both_human_sides() {
    for (slot, team) in [(0, Team::Radiant), (1, Team::Dire)] {
        let mut observer = started(slot);
        let mut view = snapshot(1);
        view.viewer = Some(team);
        for player in &mut view.players {
            player.gold = (player.slot == SlotId(slot)).then_some(100);
            player.stash = (player.slot == SlotId(slot)).then(|| vec![None; 6]);
        }
        observer.observe(ServerMsg::Snapshot { view }).unwrap();
        observer
            .observe(ServerMsg::Events {
                tick: 1,
                events: vec![],
            })
            .unwrap();
        observer
            .observe(ServerMsg::MatchOver {
                winner: team,
                stats: statistics(1),
            })
            .unwrap();
        assert_eq!(observer.last_interval.end, Some(Map2RewardEnd::Win));
        assert_eq!(observer.last_interval.terminal, 0.2);
    }
}

#[test]
fn missing_first_snapshot_and_duplicate_events_are_invalid() {
    let mut observer = started(0);
    assert!(
        observer
            .observe(ServerMsg::Snapshot { view: snapshot(2) })
            .unwrap_err()
            .to_string()
            .contains("missing initial baseline")
    );
    observer
        .observe(ServerMsg::Snapshot { view: snapshot(1) })
        .unwrap();
    observer
        .observe(ServerMsg::Events {
            tick: 1,
            events: vec![],
        })
        .unwrap();
    assert!(
        observer
            .observe(ServerMsg::Events {
                tick: 1,
                events: vec![]
            })
            .unwrap_err()
            .to_string()
            .contains("Events without pending Snapshot")
    );
    assert_eq!(observer.last_interval.terminal, 0.0);
}

#[test]
fn opening_fee_and_refund_survive_interval_drains_identically_to_production() {
    let mut observer = started(0);
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    for tick in 1..=901 {
        let mut view = snapshot(tick);
        own_hero(&mut view).pos = Vec2::from_ints(-2000, 0);
        let events = if tick == 35 {
            vec![EventKind::ItemBought {
                slot: SlotId(0),
                item: ItemId(42),
            }]
        } else {
            vec![]
        };
        reward.observe_snapshot(&view).unwrap();
        reward.observe_events(tick, &events).unwrap();
        observer.observe(ServerMsg::Snapshot { view }).unwrap();
        observer
            .observe(ServerMsg::Events { tick, events })
            .unwrap();
        let expected = reward.take_interval().unwrap();
        assert_eq!(observer.last_interval, expected, "tick={tick}");
        if tick == 35 {
            assert!(expected.fountain_wait_refund > 0.0);
        }
        if tick == 900 {
            assert_eq!(expected.opening_position, -0.1);
        }
    }
}

#[test]
fn delayed_empty_events_complete_pair_but_early_match_over_does_not() {
    let mut observer = started(0);
    observer
        .observe(ServerMsg::Snapshot { view: snapshot(1) })
        .unwrap();
    observer
        .observe(ServerMsg::OrderRejected {
            seq: 1,
            reason: bota_proto::RejectReason::NotEnoughMana,
        })
        .unwrap();
    let error = observer
        .observe(ServerMsg::MatchOver {
            winner: Team::Radiant,
            stats: statistics(1),
        })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("without complete final Snapshot/Events")
    );
    assert_eq!(observer.last_interval.terminal, 0.0);
    observer
        .observe(ServerMsg::Events {
            tick: 1,
            events: vec![],
        })
        .unwrap();
    assert_eq!(observer.last_interval.ticks, 0);
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

#[test]
fn native_frames_decode_when_every_byte_arrives_separately() {
    let message = ServerMsg::Events {
        tick: 42,
        events: vec![],
    };
    let frame = bota_proto::encode_frame_to_vec(&message).unwrap();
    assert_eq!(
        read_message(&mut Fragmented(&frame)).unwrap(),
        Some(message)
    );
}

#[test]
fn trailing_payload_bytes_and_oversized_lengths_are_rejected() {
    let mut frame = bota_proto::encode_frame_to_vec(&ServerMsg::Events {
        tick: 1,
        events: vec![],
    })
    .unwrap();
    let length = u32::from_le_bytes(frame[..4].try_into().unwrap()) + 1;
    frame[..4].copy_from_slice(&length.to_le_bytes());
    frame.push(0);
    assert!(read_message(&mut frame.as_slice()).is_err());
    let frame = ((bota_proto::MAX_PAYLOAD_LEN + 1) as u32).to_le_bytes();
    assert!(
        read_message(&mut frame.as_slice())
            .unwrap_err()
            .to_string()
            .contains("observer frame length outside")
    );
}

#[test]
fn missing_terminal_seat_statistics_is_invalid_not_a_win() {
    let mut observer = started(0);
    observer
        .observe(ServerMsg::Snapshot { view: snapshot(1) })
        .unwrap();
    observer
        .observe(ServerMsg::Events {
            tick: 1,
            events: vec![],
        })
        .unwrap();
    let error = observer
        .observe(ServerMsg::MatchOver {
            winner: Team::Radiant,
            stats: MatchStats {
                duration: 1,
                slots: vec![],
            },
        })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("MatchOver must contain both participant statistics")
    );
    assert_eq!(observer.last_interval.terminal, 0.0);
}

#[test]
fn eof_after_complete_tick_keeps_prefix_value_without_invented_terminal() {
    let mut observer = started(0);
    let mut wire = Vec::new();
    bota_proto::encode_frame(&ServerMsg::Snapshot { view: snapshot(1) }, &mut wire).unwrap();
    bota_proto::encode_frame(
        &ServerMsg::Events {
            tick: 1,
            events: vec![],
        },
        &mut wire,
    )
    .unwrap();
    let error = consume(&mut wire.as_slice(), &mut observer, 30, &mut Vec::new()).unwrap_err();
    assert_eq!(error.to_string(), "EOF before MatchOver");
    let report = observer.report(Some(&error.to_string()));
    assert!(report.contains("\"complete\":false,\"valid\":false"));
    assert!(report.contains("\"outcome\":null"));
    assert!(report.contains("\"terminal\":0"));
}

#[test]
fn empty_stream_and_missing_events_never_finish_as_time_cap() {
    for pending in [false, true] {
        let mut observer = started(0);
        if pending {
            observer
                .observe(ServerMsg::Snapshot { view: snapshot(1) })
                .unwrap();
        }
        let error = consume(&mut &[][..], &mut observer, 30, &mut Vec::new()).unwrap_err();
        assert_eq!(error.to_string(), "EOF before MatchOver");
        assert!(
            !observer
                .report(Some(&error.to_string()))
                .contains("TimeCap")
        );
        assert_eq!(observer.last_interval.terminal, 0.0);
    }
}

#[test]
fn input_byte_cap_is_an_error_not_a_synthetic_clean_eof() {
    use std::io::Read;
    let mut input = BoundedInput::new(&b"abc"[..], 2);
    let mut first = [0; 2];
    input.read_exact(&mut first).unwrap();
    assert_eq!(&first, b"ab");
    assert_eq!(
        input.read(&mut [0]).unwrap_err().to_string(),
        "observer byte limit exceeded"
    );
}
