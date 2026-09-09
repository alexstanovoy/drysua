#![allow(
    clippy::float_arithmetic,
    reason = "Read-only policy diagnostics use F32 features"
)]

use super::*;
use crate::{StructuredAction, global_feature, unit_feature};

#[derive(Default)]
struct BehaviorCounters {
    kinds: [u32; ActionKind::COUNT],
    sent: [u32; ActionKind::COUNT],
    idle_continue: u32,
    continued: u32,
    maximum_continue: u32,
}

impl BehaviorCounters {
    fn observe(&mut self, frame: &FeatureFrame, action: StructuredAction, sent: bool) {
        assert!(frame.is_finite());
        assert!(self.kinds.iter().sum::<u32>() < 36300);
        let kind = action.kind();
        self.kinds[kind.index()] += 1;
        self.sent[kind.index()] += u32::from(sent);
        self.continued = if kind == ActionKind::Continue {
            self.continued + 1
        } else {
            0
        };
        self.maximum_continue = self.maximum_continue.max(self.continued);
        self.idle_continue += u32::from(
            kind == ActionKind::Continue
                && frame.global()[global_feature::OWN_ALIVE] == 1.0
                && frame.global()[global_feature::ACTIVE_ORDER_PRESENT] == 0.0,
        );
    }
}

#[test]
fn continued_active_order_is_not_counted_as_idle_or_sent() {
    let mut frame = FeatureFrame::new();
    frame.global[global_feature::OWN_ALIVE] = 1.0;
    frame.global[global_feature::ACTIVE_ORDER_PRESENT] = 1.0;
    let mut counters = BehaviorCounters::default();
    counters.observe(&frame, StructuredAction::Continue, false);
    assert_eq!(counters.idle_continue, 0);
    assert_eq!(counters.sent.iter().sum::<u32>(), 0);
    assert_eq!(counters.maximum_continue, 1);
}

#[test]
fn continued_without_order_counts_only_living_hero() {
    let mut frame = FeatureFrame::new();
    let mut counters = BehaviorCounters::default();
    counters.observe(&frame, StructuredAction::Continue, false);
    frame.global[global_feature::OWN_ALIVE] = 1.0;
    counters.observe(&frame, StructuredAction::Continue, false);
    counters.observe(
        &frame,
        StructuredAction::Stop {
            unit: crate::ControlledUnit::Hero,
        },
        true,
    );
    assert_eq!(counters.idle_continue, 1);
    assert_eq!(counters.continued, 0);
    assert_eq!(counters.maximum_continue, 2);
}

#[test]
fn diagnostic_greedy_requests_match_existing_neural_collector() {
    let model = PolicyModel::fresh(10082000).expect("model");
    for side in 0..2 {
        let mut source = build_environment(10082000, 10, MapId(0), side, 0, OpponentSpec::Teacher)
            .expect("source");
        let mut target = build_environment(10082000, 10, MapId(0), side, 0, OpponentSpec::Teacher)
            .expect("target");
        for _ in 0..16 {
            let (frame, space) = prepare_policy_sample(&mut source).expect("frame");
            let action = model
                .choose_batch(&[frame], std::slice::from_ref(&space))
                .expect("batch")[0]
                .action;
            let (_, request) =
                neural_policy_request_in_space(&mut source.seats[side], action, &space)
                    .expect("candidate");
            let requests = requests_with_candidate(&mut source, request).expect("requests");
            let (expected, kind) =
                requests_for_neural_greedy_decision(&mut target, &model).expect("reference");
            assert_eq!(kind, action.kind());
            assert_eq!(requests, expected);
            advance_interval(&mut source, requests, 3).expect("source step");
            advance_interval(&mut target, expected, 3).expect("target step");
            assert_eq!(
                prepare_policy_sample(&mut source).expect("source").0,
                prepare_policy_sample(&mut target).expect("target").0
            );
        }
    }
}

#[test]
#[ignore = "Bounded retained-weight gameplay diagnosis; not a release gate"]
fn diagnose_retained_neural_gameplay() {
    let weights = std::env::var("DRYSUA_DIAG_WEIGHTS").expect("explicit diagnostic weights");
    let seed = std::env::var("DRYSUA_DIAG_SEED")
        .unwrap_or_else(|_| "9870000".into())
        .parse::<u64>()
        .expect("seed");
    assert!((9870000..=9870008).contains(&seed));
    let limit = std::env::var("DRYSUA_DIAG_LIMIT")
        .unwrap_or_else(|_| "108900".into())
        .parse::<u32>()
        .expect("tick cap");
    assert!((4..=108900).contains(&limit));
    let mode = std::env::var("DRYSUA_DIAG_MODE").unwrap_or_else(|_| "greedy".into());
    assert!(matches!(mode.as_str(), "greedy" | "sampled"));
    let interval = std::env::var("DRYSUA_DIAG_INTERVAL")
        .unwrap_or_else(|_| "3".into())
        .parse::<u32>()
        .expect("decision interval");
    assert!(matches!(interval, 3 | 6 | 12));
    let device = PolicyDevice::Cpu;
    let cuda = std::env::var("DRYSUA_DIAG_DEVICE").as_deref() == Ok("cuda");
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = if cuda {
        PolicyDevice::Cuda { ordinal: 0 }
    } else {
        device
    };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    assert!(!cuda, "CUDA diagnostic requested without CUDA build");
    let model = PolicyModel::fresh_on(10082001, device).expect("model");
    TrainingArtifact::load_runtime_weights(&model, Path::new(&weights)).expect("weights");
    eprintln!(
        "diagnosis weights={weights} mode={mode} device={device:?} seed={seed} limit={limit} interval={interval} opponent=current_in_process_teacher qualification=false"
    );
    diagnose_games(&model, seed, limit, mode == "sampled", interval);
}

fn diagnose_games(model: &PolicyModel, seed: u64, limit: u32, sampled: bool, interval: u32) {
    assert!(matches!(interval, 3 | 6 | 12));
    let mut arenas = diagnostic_arenas(seed);
    let mut random: Vec<_> = (0..4).map(|index| PpoRng::new(10082003 + index)).collect();
    let mut done = [false; 4];
    let mut counters: Vec<_> = (0..4).map(|_| BehaviorCounters::default()).collect();
    let started = Instant::now();
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(1200),
            "diagnostic wall limit"
        );
        let active: Vec<_> = (0..4).filter(|&index| !done[index]).collect();
        if active.is_empty() {
            break;
        }
        let prepared = parallel::ordered_active(
            &mut arenas,
            active.iter().map(|&index| (index, ())).collect(),
            |_, arena, ()| prepare_policy_sample(arena),
        )
        .expect("prepare");
        let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let actions = diagnostic_actions(model, &frames, &spaces, &active, &mut random, sampled);
        let mut jobs = Vec::with_capacity(active.len());
        for (((&index, frame), space), action) in
            active.iter().zip(&frames).zip(&spaces).zip(actions)
        {
            let arena = &mut arenas[index];
            let tick = arena.arena.tick();
            let side = arena.policy_seat;
            let (_, request) =
                neural_policy_request_in_space(&mut arena.seats[side], action, space)
                    .expect("candidate");
            counters[index].observe(frame, action, request.is_some());
            if tick <= 73 || (tick - 1).is_multiple_of(3000) {
                print_behavior(
                    arena,
                    frame,
                    &counters[index],
                    action,
                    seed + index as u64 / 2,
                );
            }
            let requests = requests_with_candidate(arena, request).expect("opponent");
            jobs.push((index, (requests, interval.min(limit - tick))));
        }
        let completed =
            parallel::ordered_active(&mut arenas, jobs, |_, arena, (requests, ticks)| {
                advance_diagnostic_interval(arena, requests, ticks)
            })
            .expect("advance");
        for (&index, result) in active.iter().zip(completed) {
            reject_production_rejection(&arenas[index], "diagnosis").expect("no rejections");
            done[index] = result.winner.is_some() || arenas[index].arena.tick() >= limit;
            if done[index] {
                print_terminal(
                    &arenas[index],
                    &counters[index],
                    result.winner,
                    seed + index as u64 / 2,
                );
            }
        }
    }
    assert!(done.iter().all(|value| *value));
    eprintln!("diagnosis_seconds={:.3}", started.elapsed().as_secs_f64());
}

fn diagnostic_arenas(seed: u64) -> Vec<TrainingEnvironment> {
    assert!((9870000..=9870008).contains(&seed));
    let arenas = (0..4)
        .map(|index| {
            build_environment(
                seed + index / 2,
                10082002,
                MapId(0),
                index as usize % 2,
                0,
                OpponentSpec::Teacher,
            )
            .expect("arena")
        })
        .collect::<Vec<_>>();
    assert_eq!(arenas.len(), 4);
    arenas
}

fn diagnostic_actions(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    active: &[usize],
    random: &mut [PpoRng],
    sampled: bool,
) -> Vec<StructuredAction> {
    assert_eq!(frames.len(), spaces.len());
    assert_eq!(frames.len(), active.len());
    if !sampled {
        return model
            .choose_batch(frames, spaces)
            .expect("greedy")
            .into_iter()
            .map(|choice| choice.action)
            .collect();
    }
    let mut selected: Vec<_> = active.iter().map(|&index| random[index].clone()).collect();
    let actions = model
        .sample_batch(frames, spaces, &mut selected)
        .expect("sample");
    for (&index, next) in active.iter().zip(selected) {
        random[index] = next;
    }
    actions.into_iter().map(|choice| choice.action()).collect()
}

fn print_behavior(
    arena: &TrainingEnvironment,
    frame: &FeatureFrame,
    counters: &BehaviorCounters,
    action: StructuredAction,
    seed: u64,
) {
    let seat = &arena.seats[arena.policy_seat];
    let hero = &frame.own_units()[0];
    assert!(frame.is_finite());
    assert_eq!(
        seat.tracker.current().expect("snapshot").tick,
        arena.arena.tick()
    );
    eprintln!(
        "behavior seed={seed} side={} tick={} action={action:?} active={:?} hp={:.4} mana={:.4} x={:.4} y={:.4} kinds={:?} sent={:?} idle_continue={} max_continue={} summary={:?}",
        arena.policy_seat,
        arena.arena.tick(),
        seat.local.active_order(),
        hero[unit_feature::HP_RATIO],
        hero[unit_feature::MANA_RATIO],
        hero[unit_feature::POSITION_X],
        hero[unit_feature::POSITION_Y],
        counters.kinds,
        counters.sent,
        counters.idle_continue,
        counters.maximum_continue,
        seat.tracker.latest_summary()
    );
}

fn print_terminal(
    arena: &TrainingEnvironment,
    counters: &BehaviorCounters,
    winner: Option<Team>,
    seed: u64,
) {
    assert!(arena.arena.tick() <= 108900);
    assert_eq!(arena.seats[arena.policy_seat].rejections, 0);
    eprintln!(
        "diagnostic_result seed={seed} side={} tick={} outcome={:?} kinds={:?} sent={:?} idle_continue={} max_continue={} summary={:?}",
        arena.policy_seat,
        arena.arena.tick(),
        checkpoint_evaluation_outcome(arena.seats[arena.policy_seat].tracker.team(), winner),
        counters.kinds,
        counters.sent,
        counters.idle_continue,
        counters.maximum_continue,
        arena.seats[arena.policy_seat].tracker.latest_summary()
    );
}

#[test]
fn slower_candidate_keeps_teacher_decisions_at_three_ticks() {
    for ticks in [3, 6, 12] {
        let mut source =
            build_environment(10082000, 10, MapId(0), 0, 0, OpponentSpec::Teacher).expect("source");
        let mut target =
            build_environment(10082000, 10, MapId(0), 0, 0, OpponentSpec::Teacher).expect("target");
        let requests = requests_with_candidate(&mut source, None).expect("first requests");
        advance_diagnostic_interval(&mut source, requests, ticks).expect("long step");
        for _ in 0..ticks / 3 {
            let requests = requests_with_candidate(&mut target, None).expect("teacher requests");
            advance_interval(&mut target, requests, 3).expect("reference step");
        }
        assert_eq!(
            source.seats[1].sequence, target.seats[1].sequence,
            "teacher cadence ticks={ticks}"
        );
        assert_eq!(
            source.seats[1].tracker.latest_summary(),
            target.seats[1].tracker.latest_summary()
        );
        assert_eq!(source.arena.tick(), target.arena.tick());
    }
}

fn advance_diagnostic_interval(
    arena: &mut TrainingEnvironment,
    mut requests: Vec<Option<Request>>,
    ticks: u32,
) -> Result<ArenaAdvance, PpoError> {
    assert!((1..=12).contains(&ticks));
    assert_eq!(requests.len(), 2);
    let mut elapsed = 0;
    for _ in 0..4 {
        let step = advance_interval(arena, requests, 3.min(ticks - elapsed))?;
        elapsed += step.ticks;
        if step.winner.is_some() || elapsed == ticks {
            return Ok(ArenaAdvance {
                winner: step.winner,
                ticks: elapsed,
            });
        }
        requests = requests_with_candidate(arena, None)?;
    }
    Err(PpoError::InvalidTransition(
        "diagnostic interval exceeds four teacher decisions",
    ))
}
