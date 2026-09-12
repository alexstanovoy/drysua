#![allow(
    clippy::float_arithmetic,
    reason = "Test-only bounded opening measurements."
)]
use super::*;
use crate::StructuredAction;
use bota_proto::Vec2;
use std::fmt::Write as _;

#[path = "map2_transfer_fix_data.rs"]
pub(super) mod data;
#[path = "map2_transfer_fix_tests.rs"]
mod tests;
#[path = "map2_transfer_fix_train.rs"]
mod train;

const OPENING_TICKS: u32 = 1800;
const ROOT: &str = "artifacts/temp/map2-gameplay-fix-20260912/transfer-001";

pub(super) struct Opening {
    arena: Arena,
    seats: Vec<ArenaSeatPolicy>,
    side: usize,
    initial_position: Vec2,
    pub(super) stats: OpeningStats,
}

#[derive(Debug, Default)]
pub(super) struct OpeningStats {
    tick: u32,
    decisions: u32,
    kinds: [u32; ActionKind::COUNT],
    sent: u32,
    suppressed: u32,
    continue_idle: u32,
    continue_active: u32,
    casts: u32,
    hero_damage: u64,
    creep_damage: u64,
    mana_spent: u64,
    pregame_progress: f64,
    progress: f64,
    path: f64,
    stationary: u32,
    trace: Vec<String>,
}

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ROOT)
}

fn opening(seed: u64, side: usize) -> Opening {
    assert!(side < 2);
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed,
    })
    .unwrap();
    let seats = setup_seats(start).expect("complete native ordinary baseline");
    let initial_position = seats[side].tracker.own_hero().unwrap().pos;
    assert_eq!(arena.tick(), 1);
    Opening {
        arena,
        seats,
        side,
        initial_position,
        stats: OpeningStats::default(),
    }
}

fn distance(source: Vec2, target: Vec2) -> f64 {
    let x = (f64::from(source.x.raw) - f64::from(target.x.raw)) / 65536.0;
    let y = (f64::from(source.y.raw) - f64::from(target.y.raw)) / 65536.0;
    x.hypot(y)
}

pub(super) fn progress_gate(stats: &OpeningStats) -> bool {
    stats.tick == OPENING_TICKS + 1 && stats.pregame_progress >= 1000.0 && stats.progress >= 5000.0
}

impl Opening {
    fn advance(&mut self, candidate: Option<Request>, opponent: Option<Request>) {
        let prior = self.seats[self.side].tracker.own_hero().cloned();
        let mut requests = [None, None];
        requests[self.side] = candidate;
        requests[1 - self.side] = opponent;
        let step = self.arena.step(&requests).expect("bounded opening tick");
        for (index, (seat, messages)) in self.seats.iter_mut().zip(step.messages).enumerate() {
            if index == self.side {
                count_casts(&mut self.stats, prior.as_ref(), &messages);
            }
            assert!(
                observe_messages(seat, &messages)
                    .expect("contiguous full native pair")
                    .is_none()
            );
            let reward = seat.tracker.take_map2_reward_interval().unwrap();
            if index == self.side {
                self.stats.hero_damage += reward.observations.hero_damage_dealt;
                self.stats.creep_damage += reward.observations.creep_damage_taken;
                self.stats.mana_spent += reward.observations.mana_spent;
            }
        }
        self.stats.tick = self.arena.tick();
        if let Some(hero) = self.seats[self.side].tracker.own_hero() {
            if let Some(prior) = prior.filter(|prior| prior.id == hero.id) {
                self.stats.path += distance(prior.pos, hero.pos);
                self.stats.stationary += u32::from(prior.pos == hero.pos);
            }
            let mid = Vec2::from_ints(9216, 9216);
            self.stats.progress = distance(self.initial_position, mid) - distance(hero.pos, mid);
            if self.stats.tick == 901 {
                self.stats.pregame_progress = self.stats.progress;
            }
        }
    }
}

fn count_casts(
    stats: &mut OpeningStats,
    before: Option<&bota_proto::UnitView>,
    messages: &[ServerMsg],
) {
    let Some(before) = before else {
        return;
    };
    let next = messages.iter().find_map(|message| match message {
        ServerMsg::Snapshot { view } => view.units.iter().find(|unit| unit.id == before.id),
        _ => None,
    });
    let Some(next) = next else {
        return;
    };
    for message in messages {
        if let ServerMsg::Events { events, .. } = message {
            for event in events {
                if let EventKind::AbilityCast { caster, ability } = event
                    && *caster == before.id
                    && (13..=15).contains(&ability.0)
                {
                    let slot = (ability.0 - 13) as usize;
                    stats.casts += u32::from(
                        next.abilities[slot].cooldown_left > before.abilities[slot].cooldown_left,
                    );
                }
            }
        }
    }
}

pub(super) fn run_opening(
    model: &PolicyModel,
    comparison: Option<&PolicyModel>,
    seed: u64,
    side: usize,
) -> Opening {
    let mut environment = opening(seed, side);
    for decision in 0..OPENING_TICKS / 3 {
        let (frame, space) =
            prepare_neural_seat_policy_sample(&mut environment.seats[side]).unwrap();
        let action = model.choose(&frame, &space).unwrap().action;
        let other = comparison.map(|model| model.choose(&frame, &space).unwrap().action);
        let active = environment.seats[side].local.active_order();
        let decoded = space.decode(action).unwrap();
        let request = neural_policy_request_in_space(&mut environment.seats[side], action, &space)
            .unwrap()
            .1;
        let stats = &mut environment.stats;
        stats.decisions += 1;
        stats.kinds[action.kind().index()] += 1;
        stats.sent += u32::from(request.is_some());
        stats.suppressed += u32::from(decoded.is_some() && request.is_none());
        stats.continue_idle += u32::from(action == StructuredAction::Continue && active.is_none());
        stats.continue_active +=
            u32::from(action == StructuredAction::Continue && active.is_some());
        if stats.trace.len() < 128 && (decision < 32 || request.is_some() || decision % 30 == 0) {
            let margin = kind_margin(model, &frame, &space, action.kind());
            let other_margin = comparison
                .zip(other)
                .map(|(model, action)| kind_margin(model, &frame, &space, action.kind()));
            let reason = if request.is_some() {
                "sent"
            } else if decoded.is_none() {
                "Continue_no_request"
            } else {
                "deduplicated"
            };
            stats.trace.push(format!("tick={} selected={action:?} same_frame_other={other:?} margin={margin:?} other_margin={other_margin:?} decoded={decoded:?} wire={request:?} reason={reason} active={active:?} hero={:?}", space.tick(), environment.seats[side].tracker.own_hero().map(|hero| (hero.pos, hero.hp, hero.mana))));
        }
        let opponent = teacher_request(&mut environment.seats[1 - side]).unwrap();
        for tick in 0..3 {
            environment.advance(
                if tick == 0 { request } else { None },
                if tick == 0 { opponent } else { None },
            );
        }
    }
    assert_eq!(environment.seats[side].rejections, 0);
    environment
}

fn kind_margin(
    model: &PolicyModel,
    frame: &FeatureFrame,
    space: &ActionSpace,
    selected: ActionKind,
) -> (f32, f32) {
    let logits = model.evaluate(frame).unwrap().kind_logits;
    let rival = logits
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected.index() && space.kind_mask().as_array()[*index])
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    (logits[selected.index()], logits[selected.index()] - rival)
}

fn load_model(path: &Path) -> PolicyModel {
    let model = PolicyModel::fresh_on(10092099, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, path).expect("strict current M16 weights");
    assert_eq!(model.device(), PolicyDevice::Cpu);
    model
}

#[test]
#[ignore = "Guarded causal regression: failed BC must fail actual ordinary opening progress, not just order count."]
fn ordinary_opening_progress_regression() {
    let old = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/map2-learning-20260911");
    let initial = load_model(&old.join("initial"));
    let subject = std::env::var("MAP2_TRANSFER_FIX_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| old.join("skill-bootstrap-001-transfer/diagnostic-weights"));
    let subject = load_model(&subject);
    let tag = std::env::var("MAP2_TRANSFER_FIX_TAG").unwrap_or_else(|_| "opening-red".into());
    assert!(tag == "opening-red" || tag == "opening-green");
    let mut report = String::new();
    let mut passed = true;
    for side in 0..2 {
        let seed = 10091900 + side as u64;
        if tag == "opening-red" {
            let original = run_opening(&initial, Some(&subject), seed, side);
            writeln!(
                report,
                "INITIAL side={side} seed={seed} {:?}",
                original.stats
            )
            .unwrap();
        }
        let observed = run_opening(&subject, Some(&initial), seed, side);
        passed &= progress_gate(&observed.stats);
        writeln!(
            report,
            "SUBJECT side={side} seed={seed} gate={} {:?}",
            progress_gate(&observed.stats),
            observed.stats
        )
        .unwrap();
    }
    let path = root().join(format!("{tag}.txt"));
    assert!(!path.exists());
    std::fs::write(path, &report).unwrap();
    eprintln!("{report}");
    assert!(
        passed,
        "ordinary opening regression: must make useful lane progress by 901/1801, not merely send orders"
    );
}

#[test]
fn opening_gate_rejects_stationary_orders_and_accepts_progress_without_many_orders() {
    let mut stats = OpeningStats {
        tick: 1801,
        sent: 100,
        ..OpeningStats::default()
    };
    assert!(!progress_gate(&stats));
    stats.sent = 1;
    stats.pregame_progress = 1000.0;
    stats.progress = 5000.0;
    assert!(progress_gate(&stats));
    stats.pregame_progress = 999.99;
    assert!(!progress_gate(&stats));
}

#[test]
fn distance_retains_subunit_precision_and_is_symmetric() {
    let source = Vec2::from_ints(0, 0);
    let target = Vec2 {
        x: bota_proto::Fixed::from_ratio(1, 2),
        y: bota_proto::Fixed::ZERO,
    };
    assert_eq!(distance(source, target), 0.5);
    assert_eq!(distance(source, target), distance(target, source));
}
