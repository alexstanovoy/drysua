//! Retained intervals on a real slot: which decisions they cover, what they
//! earn and how they close, independently of the policy's retention phase.
#![allow(
    clippy::float_arithmetic,
    reason = "Interval rewards are compared against summed per-decision rewards"
)]

use super::*;

fn schedule(decision_cap: usize) -> GameSchedule {
    GameSchedule {
        seed: 9_971_001,
        mixture: OpponentMixture::new(vec![(OpponentKind::Teacher, 1)]).expect("mixture"),
        decision_cap,
        config: PpoConfig {
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
    }
}

struct Played {
    actions: Vec<StructuredAction>,
    random: PpoRng,
    rewards: Vec<f64>,
    samples: Vec<PpoTransition>,
    finished: Option<FinishedGame>,
}

/// Plays `decisions` decisions of slot zero's first game with a fixed phase.
fn play(model: &PolicyModel, phase: usize, decisions: usize, cap: usize) -> Played {
    let schedule = schedule(cap);
    let plan = GamePlan::new(&schedule, 0, 0, 0, ModifierSpec::NOMINAL).expect("plan");
    let mut slot = Slot::start(plan, &schedule).expect("slot");
    slot.stream = EpisodeStream::new(phase);
    let mut played = Played {
        actions: Vec::new(),
        random: PpoRng::new(0),
        rewards: Vec::new(),
        samples: Vec::new(),
        finished: None,
    };
    for _ in 0..decisions {
        let sampled = model
            .sample_rows(
                &[&slot.policy.row],
                &[&slot.policy.space],
                std::slice::from_mut(&mut slot.actor_random),
                &[slot.stream.begins_interval()],
            )
            .expect("sample")
            .pop()
            .expect("one row");
        if slot.stream.awaits_value() {
            played
                .samples
                .push(slot.stream.flush(Some(sampled.value)).expect("flush"));
        }
        played.actions.push(sampled.action);
        let before = slot.stream.raw_return();
        let decision = Decision {
            action: sampled.action,
            retained: sampled
                .statistics
                .map(|statistics| Box::new((statistics, sampled.value, 0))),
            opponent: None,
        };
        let next = NextGame {
            update: 0,
            spec: ModifierSpec::NOMINAL,
        };
        played.random = slot.actor_random.clone();
        let (next, finished) = slot.advance(decision, &schedule, next).expect("advance");
        if let Some(finished) = finished {
            played.rewards.push(finished.record.total_reward() - before);
            played.samples.extend(finished.terminal.clone());
            played.finished = Some(finished);
            break;
        }
        played.rewards.push(next.stream.raw_return() - before);
        slot = next;
    }
    played
}

#[test]
fn retention_phase_changes_neither_actions_nor_rewards_and_intervals_sum_their_rewards() {
    let model = PolicyModel::fresh(23_077).expect("model");
    let reference = play(&model, 0, 20, crate::MAP2_ACTOR_DECISIONS);
    for phase in 0..RETENTION_STRIDE {
        let played = play(&model, phase, 20, crate::MAP2_ACTOR_DECISIONS);
        assert_eq!(played.actions, reference.actions, "phase {phase}");
        assert_eq!(played.random, reference.random, "phase {phase}");
        assert_eq!(played.rewards, reference.rewards, "phase {phase}");
        // Decisions phase..phase+8 closed when decision phase+8 was sampled.
        let closed = (19 - phase) / RETENTION_STRIDE;
        assert_eq!(played.samples.len(), closed, "phase {phase}");
        for (index, sample) in played.samples.iter().enumerate() {
            let start = phase + index * RETENTION_STRIDE;
            let reward: f64 = played.rewards[start..start + RETENTION_STRIDE].iter().sum();
            assert_eq!(sample.ticks, 24);
            assert!(!sample.terminal);
            assert_eq!(sample.action, played.actions[start]);
            assert!((f64::from(sample.reward) - reward).abs() < 1e-6);
        }
    }
}

#[test]
fn a_game_ending_inside_an_interval_closes_it_terminal_without_a_bootstrap() {
    let model = PolicyModel::fresh(23_078).expect("model");
    let played = play(&model, 2, 64, 5);
    let finished = played.finished.expect("decision cap ends the game");
    let terminal = finished.terminal.expect("open interval at the end");
    assert!(terminal.terminal);
    assert_eq!(terminal.next_value, 0.0);
    assert_eq!(terminal.ticks, 9);
    assert_eq!(terminal.action, played.actions[2]);
    let reward: f64 = played.rewards[2..5].iter().sum();
    assert!((f64::from(terminal.reward) - reward).abs() < 1e-6);
    let mut report = CollectionReport::default();
    finished.record.accumulate(&mut report).expect("report");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.elapsed_ticks, 15);
    let unsampled = play(&model, 7, 64, 5);
    assert!(unsampled.samples.is_empty(), "no fabricated sample");
    assert!(unsampled.finished.expect("ended").terminal.is_none());
}
