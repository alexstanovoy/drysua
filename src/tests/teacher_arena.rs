use bota_proto::{MapId, ServerMsg, SlotId, Team, Vec2};

use crate::{
    ActionKind, Arena, ArenaConfig, ArenaStart, CheckpointEvaluationBaseline,
    CheckpointEvaluationOutcome, ItemReadiness, OrderPersistence, Request, StateTracker, Teacher,
};

const MATCH_TICKS: usize = 1_800;
const TEACHER_WEAK_DECISIONS: usize = 3_072;
const TEACHER_WEAK_SEED: u64 = 90_001;

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
fn teachers_buy_wraith_and_tango_for_595_during_first_pregame_decisions_on_both_maps() {
    for map in [MapId(0), MapId(1)] {
        let (mut arena, start) = Arena::new(ArenaConfig {
            seats: 2,
            map,
            seed: 92_009_004,
        })
        .expect("opening arena");
        let mut seats = setup_seats(start);
        let mut counts = GateCounts::default();
        let mut purchases = [[None; 2]; 2];
        let mut purchase_counts = [0; 2];
        for tick in 1u32..=9 {
            assert_eq!(arena.tick(), tick);
            let mut requests = [None; 2];
            if (tick - 1).is_multiple_of(3) {
                for (index, seat) in seats.iter_mut().enumerate() {
                    requests[index] = decide_request(seat, &mut counts);
                    if let Some(Request {
                        order: bota_proto::Order::Buy { item },
                        ..
                    }) = requests[index]
                    {
                        assert!(purchase_counts[index] < 2);
                        purchases[index][purchase_counts[index]] = Some(item);
                        purchase_counts[index] += 1;
                    }
                }
            }
            let step = arena.step(&requests).expect("opening tick");
            for (seat, messages) in seats.iter_mut().zip(step.messages) {
                observe_messages(seat, &messages, &mut counts);
            }
        }

        assert_eq!(counts.rejections, 0);
        assert_eq!(counts.suppressions, 0);
        for (seat, purchased) in seats.iter().zip(purchases) {
            assert_eq!(
                purchased,
                [Some(bota_proto::ItemId(33)), Some(bota_proto::ItemId(7))]
            );
            assert!(arena.tick() < seat.tracker.metadata().pregame_ticks);
            assert_eq!(seat.tracker.own_player().expect("player").gold, Some(5));
            let items = &seat.tracker.own_hero().expect("hero").items;
            assert_eq!(
                items
                    .iter()
                    .flatten()
                    .map(|item| item.id)
                    .collect::<Vec<_>>(),
                [bota_proto::ItemId(33), bota_proto::ItemId(7)]
            );
        }
    }
}

#[test]
fn two_teachers_cover_and_decode_every_decision_with_low_rejection_rate_on_both_maps() {
    for map in [MapId(0), MapId(1)] {
        run_teacher_match(map);
    }
}

#[test]
fn teacher_against_weak_makes_gameplay_progress_on_both_maps_and_sides() {
    let games =
        crate::evaluate_teacher_against_weak_for_test(TEACHER_WEAK_SEED, TEACHER_WEAK_DECISIONS)
            .expect("teacher-versus-weak evaluation");
    let expected = [
        (MapId(0), Team::Radiant),
        (MapId(0), Team::Dire),
        (MapId(1), Team::Radiant),
        (MapId(1), Team::Dire),
    ];
    let mut total_denies = 0u64;

    assert_eq!(games.len(), 4);
    for (game, (map, team)) in games.into_iter().zip(expected) {
        println!(
            "map={} team={:?} outcome={:?} decisions={} ticks={} orders={} rejected={} actions={:?} kills={} deaths={} last_hits={} denies={} structures={}",
            game.map.0,
            game.candidate_team,
            game.outcome,
            game.decisions,
            game.elapsed_ticks,
            game.wire_orders,
            game.rejected_orders,
            game.action_counts,
            game.final_summary.allied.kills,
            game.final_summary.allied.deaths,
            game.final_summary.allied.last_hits,
            game.final_summary.allied.denies,
            game.final_summary.enemy_structures_destroyed,
        );
        assert_eq!((game.map, game.candidate_team), (map, team));
        assert_eq!(game.baseline, CheckpointEvaluationBaseline::Weak);
        assert_eq!(game.seed, TEACHER_WEAK_SEED);
        assert!((1..=TEACHER_WEAK_DECISIONS as u32).contains(&game.decisions));
        assert!(game.elapsed_ticks <= game.decisions * 3);
        if game.outcome == CheckpointEvaluationOutcome::Timeout {
            assert_eq!(game.decisions, TEACHER_WEAK_DECISIONS as u32);
            assert_eq!(game.elapsed_ticks, game.decisions * 3);
        }
        assert_eq!(game.rejected_orders, 0);
        assert_eq!(game.baseline_wire_orders, 0);
        assert_eq!(game.baseline_rejected_orders, 0);
        assert_eq!(game.action_counts.iter().sum::<u32>(), game.decisions);
        assert!(
            game.action_counts[ActionKind::AttackUnit.index()] > 0,
            "Teacher must target a unit on map {} as {:?}: {:?}",
            game.map.0,
            game.candidate_team,
            game.action_counts,
        );
        assert!(
            game.final_summary.allied.last_hits > 0,
            "Teacher must last-hit on map {} as {:?}",
            game.map.0,
            game.candidate_team,
        );
        if game.map == MapId(1) {
            assert_eq!(game.outcome, CheckpointEvaluationOutcome::Win);
            assert!(game.final_summary.enemy_structures_destroyed > 0);
        }
        total_denies = total_denies
            .checked_add(game.final_summary.allied.denies)
            .expect("four games have bounded denies");
    }
    assert!(
        total_denies > 0,
        "Teacher must demonstrate deny supervision"
    );
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
    let mut moved = vec![false; seats.len()];

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
        for (index, seat) in seats.iter().enumerate() {
            if let Some(hero) = seat.tracker.own_hero() {
                moved[index] |= Some(hero.pos) != initial_hero_positions[index];
            }
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
    // A hero can move, die, and respawn at its initial position before the final snapshot.
    assert!(
        moved.into_iter().all(|moved| moved),
        "teacher orders must move each hero"
    );
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
