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

/// `(start, end)` decisions of each sample, in order, from their elapsed ticks.
fn intervals(samples: &[PpoTransition]) -> Vec<(usize, usize)> {
    let mut start = 0;
    samples
        .iter()
        .map(|sample| {
            assert_eq!(sample.ticks % 3, 0);
            let end = start + sample.ticks as usize / 3;
            let interval = (start, end);
            start = end;
            interval
        })
        .collect()
}

#[test]
fn intervals_begin_at_every_order_or_retained_continue_and_sum_their_rewards() {
    let model = PolicyModel::fresh(23_077).expect("model");
    let played = play(&model, 60, crate::MAP2_ACTOR_DECISIONS, two_orders);
    assert_ne!(played.actions[3], StructuredAction::Continue);
    assert_ne!(played.actions[20], StructuredAction::Continue);
    let intervals = intervals(&played.samples);
    assert!(intervals.len() >= 5);
    for (sample, &(start, end)) in played.samples.iter().zip(&intervals) {
        assert!(end - start <= crate::MAP2_CONTINUE_STRIDE);
        let reward: f64 = played.rewards[start..end].iter().sum();
        assert!(!sample.terminal);
        assert_eq!(sample.action, played.actions[start]);
        assert!((f64::from(sample.reward) - reward).abs() < 1e-6);
        let forced = start == 0
            || sample.action != StructuredAction::Continue
            || intervals
                .iter()
                .any(|&(before, end)| end == start && end - before == crate::MAP2_CONTINUE_STRIDE);
        let weight = if forced {
            1.0
        } else {
            crate::MAP2_CONTINUE_STRIDE as f32
        };
        assert_eq!(sample.weight, weight, "interval {start}..{end}");
    }
    for order in [3, 20] {
        assert!(intervals.iter().any(|&(start, _)| start == order));
    }
}

/// Reproduces the retention bias: Continue decisions were retained only once an
/// interval was eight decisions long, so between frequent orders none trained,
/// and any offset of the advantages moved Continue's probability. Weighted by
/// their inverse retention probability, retained Continue samples must count
/// every Continue decision.
#[test]
fn retained_continue_weights_count_every_continue_decision() {
    let model = PolicyModel::fresh(23_079).expect("model");
    let played = play(
        &model,
        2_400,
        crate::MAP2_ACTOR_DECISIONS,
        |index, sampled| {
            if index % 4 == 0 {
                sampled
            } else {
                StructuredAction::Continue
            }
        },
    );
    let continues = |action: &StructuredAction| *action == StructuredAction::Continue;
    // Decisions of the closed intervals; the open one has no sample yet.
    let decided = intervals(&played.samples).last().map_or(0, |&(_, end)| end);
    let actions = &played.actions[..decided];
    let continue_decisions = actions.iter().filter(|action| continues(action)).count();
    assert!(
        continue_decisions > 800,
        "{continue_decisions} Continue decisions"
    );
    let weighted: f64 = played
        .samples
        .iter()
        .filter(|sample| continues(&sample.action))
        .map(|sample| f64::from(sample.weight))
        .sum();
    let ratio = weighted / continue_decisions as f64;
    assert!(
        (ratio - 1.0).abs() < 0.2,
        "weighted Continue samples per decision {ratio}"
    );
    let orders = actions.iter().filter(|action| !continues(action)).count();
    let order_samples = played
        .samples
        .iter()
        .filter(|sample| !continues(&sample.action));
    assert!(order_samples.clone().all(|sample| sample.weight == 1.0));
    assert_eq!(order_samples.count(), orders);
}

#[test]
fn a_game_ending_inside_an_interval_closes_it_terminal_without_a_bootstrap() {
    let model = PolicyModel::fresh(23_078).expect("model");
    let played = play(&model, 64, 5, |_, _| StructuredAction::Continue);
    let finished = played.finished.expect("decision cap ends the game");
    let terminal = finished.terminal.clone().expect("open interval at the end");
    assert!(terminal.terminal);
    assert_eq!(terminal.next_value, 0.0);
    let (last, earlier) = played.samples.split_last().expect("samples");
    assert!(last.terminal);
    assert_eq!((last.ticks, last.reward), (terminal.ticks, terminal.reward));
    assert!(earlier.iter().all(|sample| !sample.terminal));
    let intervals = intervals(&played.samples);
    assert_eq!(intervals.last().map(|&(_, end)| end), Some(5));
    for (sample, &(start, end)) in played.samples.iter().zip(&intervals) {
        let reward: f64 = played.rewards[start..end].iter().sum();
        assert!((f64::from(sample.reward) - reward).abs() < 1e-6);
    }
    let mut report = CollectionReport::default();
    finished.record.accumulate(&mut report).expect("report");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.elapsed_ticks, 15);
}
