//! Retained intervals on a real slot: which decisions they cover, what they
//! earn and how they close.
#![allow(
    clippy::float_arithmetic,
    reason = "Interval rewards are compared against summed per-decision rewards"
)]

use super::*;

fn schedule(decision_cap: usize) -> GameSchedule {
    GameSchedule {
        seed: 9_971_001,
        decision_cap,
        config: PpoConfig {
            gamma_tick: MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
        shadow: None,
        potentials: None,
    }
}

fn next_game() -> NextGame {
    NextGame {
        update: 0,
        spec: ModifierSpec::NOMINAL,
        mixture: std::sync::Arc::new(
            OpponentMixture::new(vec![(OpponentKind::Teacher, 1)]).expect("mixture"),
        ),
        potential: 0,
    }
}

struct Played {
    actions: Vec<StructuredAction>,
    rewards: Vec<f64>,
    samples: Vec<PpoTransition>,
    finished: Option<FinishedGame>,
}

/// Plays `decisions` decisions of slot zero's first game like a lane does,
/// replacing each sampled action with `choose(decision, sampled)`.
fn play(
    model: &PolicyModel,
    decisions: usize,
    cap: usize,
    choose: impl Fn(usize, StructuredAction) -> StructuredAction,
) -> Played {
    let schedule = schedule(cap);
    let plan = GamePlan::new(&schedule, 0, 0, &next_game()).expect("plan");
    let mut slot = Slot::start(plan, &schedule).expect("slot");
    let mut played = Played {
        actions: Vec::new(),
        rewards: Vec::new(),
        samples: Vec::new(),
        finished: None,
    };
    for index in 0..decisions {
        let sampled = model
            .sample_rows(
                &[&slot.policy.row],
                &[&slot.policy.space],
                std::slice::from_mut(&mut slot.actor_random),
                &[true],
            )
            .expect("sample")
            .pop()
            .expect("one row");
        let action = choose(index, sampled.action);
        let statistics = if action == sampled.action {
            sampled.statistics.expect("statistics")
        } else {
            SampledStatistics {
                target: ActionHeadTargets::from_sampled_action(&slot.policy.space, action)
                    .expect("target"),
                log_probability: -1.0,
            }
        };
        if slot.stream.closes_interval(action.kind()) {
            played
                .samples
                .push(slot.stream.flush(Some(sampled.value)).expect("flush"));
        }
        played.actions.push(action);
        let before = slot.stream.raw_return();
        let decision = Decision {
            action,
            retained: slot
                .stream
                .retains(action.kind())
                .then(|| Box::new((statistics, sampled.value, 0))),
            opponent: None,
        };
        let next = next_game();
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

/// The sampled action at decisions 3 and 20, Continue everywhere else.
fn two_orders(index: usize, sampled: StructuredAction) -> StructuredAction {
    if [3, 20].contains(&index) {
        sampled
    } else {
        StructuredAction::Continue
    }
}

#[test]
fn intervals_begin_at_the_first_every_order_and_each_eighth_continue_and_sum_their_rewards() {
    let model = PolicyModel::fresh(23_077).expect("model");
    let played = play(&model, 30, crate::MAP2_ACTOR_DECISIONS, two_orders);
    assert_ne!(played.actions[3], StructuredAction::Continue);
    assert_ne!(played.actions[20], StructuredAction::Continue);
    // Starts: 0 first, 3 order, 11 and 19 eighth Continue, 20 order, 28 eighth Continue.
    let intervals = [(0, 3), (3, 11), (11, 19), (19, 20), (20, 28)];
    assert_eq!(played.samples.len(), intervals.len());
    for (sample, (start, end)) in played.samples.iter().zip(intervals) {
        let reward: f64 = played.rewards[start..end].iter().sum();
        assert_eq!(sample.ticks, 3 * (end - start) as u32);
        assert!(!sample.terminal);
        assert_eq!(sample.action, played.actions[start]);
        assert!((f64::from(sample.reward) - reward).abs() < 1e-6);
    }
}

#[test]
fn a_game_ending_inside_an_interval_closes_it_terminal_without_a_bootstrap() {
    let model = PolicyModel::fresh(23_078).expect("model");
    let played = play(&model, 64, 5, |_, _| StructuredAction::Continue);
    let finished = played.finished.expect("decision cap ends the game");
    let terminal = finished.terminal.expect("open interval at the end");
    assert!(played.samples.len() == 1, "only the terminal interval");
    assert!(terminal.terminal);
    assert_eq!(terminal.next_value, 0.0);
    assert_eq!(terminal.ticks, 15);
    assert_eq!(terminal.action, played.actions[0]);
    let reward: f64 = played.rewards.iter().sum();
    assert!((f64::from(terminal.reward) - reward).abs() < 1e-6);
    let mut report = CollectionReport::default();
    finished.record.accumulate(&mut report).expect("report");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.elapsed_ticks, 15);
}
