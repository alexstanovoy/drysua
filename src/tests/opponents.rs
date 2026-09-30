//! PFSP weights and league membership at their numerical boundaries.

use super::*;

#[test]
fn league_members_are_the_latest_exported_milestones_oldest_first() {
    let league = League {
        weight: 1,
        size: 3,
        every: 20,
    };
    assert_eq!(league.members(0), Vec::<u64>::new(), "nothing exported yet");
    assert_eq!(league.members(1), [0]);
    assert_eq!(league.members(20), [0]);
    assert_eq!(league.members(21), [0, 20]);
    assert_eq!(league.members(81), [40, 60, 80]);
}

#[test]
fn pfsp_prefers_opponents_the_learner_loses_to_and_never_drops_one() {
    let mut window = OutcomeWindow::default();
    for _ in 0..PFSP_WINDOW + 10 {
        window
            .record(OpponentKind::Teacher, Some(PpoTerminalOutcome::Loss))
            .expect("record");
        window
            .record(OpponentKind::SelfPlay, Some(PpoTerminalOutcome::Win))
            .expect("record");
    }
    window
        .record(OpponentKind::Snapshot(0), Some(PpoTerminalOutcome::Draw))
        .expect("record");
    assert_eq!(
        window.tally(OpponentKind::Teacher),
        (100, 0),
        "window of 100"
    );
    let entries = [
        (OpponentKind::Teacher, 1_000_000),
        (OpponentKind::SelfPlay, 1_000_000),
        (OpponentKind::Snapshot(0), 1_000_000),
        (OpponentKind::League(0), 1_000_000),
    ];
    let mixture = schedule_mixture(&entries, &window, OpponentSchedule::Pfsp).expect("mixture");
    let weights: Vec<u64> = mixture
        .entries()
        .iter()
        .map(|(_, weight)| *weight)
        .collect();
    // (101/102)^2 after 100 losses, (1/102)^2 after 100 wins, (1/2)^2 after one draw or none.
    assert_eq!(weights, [980_488, 96, 250_000, 250_000]);
    let fixed = schedule_mixture(&entries, &window, OpponentSchedule::Fixed).expect("fixed");
    assert_eq!(fixed.total(), 4_000_000);
}
