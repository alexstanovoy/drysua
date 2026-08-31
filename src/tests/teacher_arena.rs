use bota_proto::{MapId, ServerMsg, SlotId, Vec2};

use crate::{
    Arena, ArenaConfig, ArenaStart, ItemReadiness, OrderPersistence, Request, StateTracker, Teacher,
};

const MATCH_TICKS: usize = 1_800;

struct SeatPolicy {
    teacher: Teacher,
    tracker: StateTracker,
    persistence: OrderPersistence,
    readiness: ItemReadiness,
    sequence: u32,
}

#[derive(Default)]
struct GateCounts {
    decisions: u64,
    rejections: u64,
    requests: u64,
    suppressions: u64,
}

#[test]
fn two_teachers_cover_and_decode_every_decision_with_low_rejection_rate_on_both_maps() {
    for map in [MapId(0), MapId(1)] {
        run_teacher_match(map);
    }
}

fn run_teacher_match(map: MapId) {
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map,
        seed: 8_271 + u64::from(map.0),
    })
    .expect("teacher arena starts");
    let mut seats = setup_seats(start);
    let initial_hero_positions: Vec<Option<Vec2>> = seats
        .iter()
        .map(|seat| seat.tracker.own_hero().map(|hero| hero.pos))
        .collect();
    let mut counts = GateCounts::default();

    for _ in 0..MATCH_TICKS {
        let requests: Vec<Option<Request>> = seats
            .iter_mut()
            .map(|seat| decide_request(seat, &mut counts))
            .collect();
        let step = arena.step(&requests).expect("teacher arena advances");
        let finished = step
            .messages
            .iter()
            .flatten()
            .any(|message| matches!(message, ServerMsg::MatchOver { .. }));
        for (seat, messages) in seats.iter_mut().zip(step.messages) {
            observe_messages(seat, &messages, &mut counts);
        }
        if finished {
            break;
        }
    }

    assert!(counts.decisions > 0, "teacher must make decisions");
    assert!(counts.requests > 0, "teacher must exercise wire requests");
    assert_eq!(
        counts.suppressions, 0,
        "teacher must return Continue instead of a suppressed wire order"
    );
    for (seat, initial) in seats.iter().zip(initial_hero_positions) {
        let current = seat.tracker.own_hero().map(|hero| hero.pos);
        assert_ne!(current, initial, "teacher orders must move each hero");
    }
    assert!(
        counts.rejections.saturating_mul(1_000) < counts.requests,
        "teacher rejection rate must be below 0.1%: {}/{}",
        counts.rejections,
        counts.requests
    );
}

fn setup_seats(start: ArenaStart) -> Vec<SeatPolicy> {
    start
        .messages
        .into_iter()
        .enumerate()
        .map(|(index, messages)| {
            let info = messages.iter().find_map(|message| match message {
                ServerMsg::MatchStart { info } => Some(info),
                _ => None,
            });
            let snapshot = messages.iter().find_map(|message| match message {
                ServerMsg::Snapshot { view } => Some(view),
                _ => None,
            });
            let mut tracker = StateTracker::new(
                SlotId(u8::try_from(index).expect("seat index fits")),
                info.expect("match info"),
            )
            .expect("seat tracker");
            tracker
                .observe_snapshot(snapshot.expect("initial snapshot"))
                .expect("initial snapshot is valid");
            SeatPolicy {
                teacher: Teacher::new(),
                tracker,
                persistence: OrderPersistence::default(),
                readiness: ItemReadiness::new(),
                sequence: 0,
            }
        })
        .collect()
}

fn decide_request(seat: &mut SeatPolicy, counts: &mut GateCounts) -> Option<Request> {
    counts.decisions = counts.decisions.checked_add(1).expect("decisions bounded");
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .expect("teacher decision");
    assert!(
        space.allows(action),
        "teacher action must have mask coverage"
    );
    let decoded = space.decode(action).expect("teacher action must decode")?;
    let Some(issued) = seat.persistence.should_send(Some(decoded)) else {
        counts.suppressions = counts
            .suppressions
            .checked_add(1)
            .expect("suppressions bounded");
        return None;
    };
    seat.sequence = seat
        .sequence
        .checked_add(1)
        .expect("sequence stays bounded");
    seat.persistence
        .record_sent(seat.sequence, issued)
        .expect("sequence increases");
    seat.readiness.note_sent(seat.sequence, issued, &space);
    seat.teacher.note_sent(seat.sequence, issued, space.tick());
    counts.requests = counts.requests.checked_add(1).expect("requests bounded");
    Some(Request {
        seq: seat.sequence,
        unit: issued.unit,
        order: issued.order,
    })
}

fn observe_messages(seat: &mut SeatPolicy, messages: &[ServerMsg], counts: &mut GateCounts) {
    for message in messages {
        match message {
            ServerMsg::OrderRejected { seq, .. } => {
                seat.persistence.observe_rejection(*seq);
                seat.readiness.note_rejected(*seq);
                seat.teacher.note_rejected(*seq);
                counts.rejections = counts
                    .rejections
                    .checked_add(1)
                    .expect("rejections bounded");
            }
            ServerMsg::Snapshot { view } => seat
                .tracker
                .observe_snapshot(view)
                .expect("teacher snapshot is valid"),
            ServerMsg::Events { tick, events } => {
                seat.tracker
                    .observe_events(*tick, events)
                    .expect("teacher events are valid");
            }
            ServerMsg::MatchStart { .. }
            | ServerMsg::Welcome { .. }
            | ServerMsg::LobbyState { .. }
            | ServerMsg::Orders { .. }
            | ServerMsg::ParticipantLeft { .. }
            | ServerMsg::MatchOver { .. } => {}
        }
    }
}
