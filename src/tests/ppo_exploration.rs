#![allow(
    clippy::float_arithmetic,
    reason = "Bounded test-only policy entropy measurements."
)]
use super::*;
use crate::{BehavioralTarget, ControlledUnit, PolicyChoice, StructuredAction};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

const DECISIONS: usize = 600;
const ROOT: &str = "artifacts/temp/map2-gameplay-fix-20260912/ppo-exploration-001";
const PARENT_SHA: &str = "107e19e61794c457ce8ec6adc2b3e66ccce08ee5b6544cb2005964f3e7dfedc8";

struct Measurement {
    choice: PolicyChoice,
    sample: PpoPolicyChoice,
    probability: f64,
    kind_entropy: f64,
    conditional_entropy: f64,
    conditional_heads: usize,
}

struct Observation {
    side: usize,
    decision: usize,
    frame: FeatureFrame,
    space: ActionSpace,
    measured: Measurement,
}

#[derive(Debug)]
struct Summary {
    frames: usize,
    confident: usize,
    kind_entropy: f64,
    path_entropy: f64,
    conditional_entropy: f64,
    conditional_heads: usize,
    value_min: f64,
    value_max: f64,
    value_outside: usize,
    greedy_heads: [usize; 12],
}

#[test]
#[ignore = "Authorized bounded ordinary-frame probe and conditional initialization; no optimizer."]
fn ordinary_probe_and_conditional_initialization() {
    let parent_path = root().join("parent");
    assert_eq!(
        file_hash(&parent_path.join("drysua.weights.safetensors")),
        PARENT_SHA
    );
    let parent = load_model(&parent_path);
    let identity = parent.policy_identity().unwrap();
    let mut report = String::new();
    let mut observations = Vec::with_capacity(2 * DECISIONS);
    let mut before = [Summary::new(), Summary::new()];
    for (side, summary) in before.iter_mut().enumerate() {
        collect_side(&parent, side, summary, &mut observations, &mut report);
    }
    assert_eq!(observations.len(), 1200);
    write_new("PARENT.tsv", &report);
    let confident = before.iter().all(Summary::confident);
    let mut selected = "parent";
    if confident {
        let softened = prepare_softened(&parent);
        let mut after = [Summary::new(), Summary::new()];
        let mut parity = String::new();
        for observation in &observations {
            check_parity(
                &softened,
                observation,
                &mut after[observation.side],
                &mut parity,
            );
        }
        for side in 0..2 {
            assert!(after[side].kind_entropy > before[side].kind_entropy);
            assert!(after[side].path_entropy > before[side].path_entropy);
            writeln!(parity, "SOFTENED side={side} {:?}", after[side]).unwrap();
        }
        write_new("PARITY.tsv", &parity);
        let directory = root().join("softened");
        std::fs::create_dir(&directory).unwrap();
        TrainingArtifact::save_runtime_weights(&softened, &directory).unwrap();
        selected = "softened";
    }
    assert_eq!(parent.policy_identity().unwrap(), identity);
    assert_eq!(
        file_hash(&parent_path.join("drysua.weights.safetensors")),
        PARENT_SHA
    );
    record_decision(&before, confident, selected);
}

fn record_decision(before: &[Summary; 2], confident: bool, selected: &str) {
    let selected_hash = file_hash(&root().join(selected).join("drysua.weights.safetensors"));
    let mut decision = format!(
        "{{\"hypothesis_verified\":{confident},\"selected\":\"{selected}\",\"sha256\":\"{selected_hash}\",\"parity_frames\":{},\"critic_changed\":false,\"sides\":[",
        if confident { 1200 } else { 0 }
    );
    for (side, summary) in before.iter().enumerate() {
        if side != 0 {
            decision.push(',');
        }
        write!(decision, "{{\"side\":{side},\"frames\":{},\"confident\":{},\"kind_entropy_mean\":{},\"path_entropy_mean\":{},\"conditional_entropy_mean\":{},\"conditional_heads\":{},\"value_min\":{},\"value_max\":{},\"value_outside\":{}}}", summary.frames, summary.confident, summary.kind_entropy / summary.frames as f64, summary.path_entropy / summary.frames as f64, summary.conditional_mean(), summary.conditional_heads, summary.value_min, summary.value_max, summary.value_outside).unwrap();
    }
    decision.push_str("]}\n");
    write_new("DECISION.json", &decision);
    eprintln!("{decision}");
}

impl Summary {
    fn new() -> Self {
        Self {
            frames: 0,
            confident: 0,
            kind_entropy: 0.0,
            path_entropy: 0.0,
            conditional_entropy: 0.0,
            conditional_heads: 0,
            value_min: f64::INFINITY,
            value_max: f64::NEG_INFINITY,
            value_outside: 0,
            greedy_heads: [0; 12],
        }
    }

    fn add(&mut self, measured: &Measurement, frame: &FeatureFrame, space: &ActionSpace) {
        assert!(self.frames < DECISIONS);
        let value = f64::from(measured.choice.value);
        assert!(value.is_finite());
        self.frames += 1;
        self.confident += usize::from(measured.probability > 0.98);
        self.kind_entropy += measured.kind_entropy;
        self.path_entropy += f64::from(measured.sample.entropy());
        self.conditional_entropy += measured.conditional_entropy;
        self.conditional_heads += measured.conditional_heads;
        self.value_min = self.value_min.min(value);
        self.value_max = self.value_max.max(value);
        self.value_outside += usize::from(!(-1.4..=1.4).contains(&value));
        let target = BehavioralTarget::from_action(frame, space, measured.choice.action).unwrap();
        for (count, active) in self.greedy_heads.iter_mut().zip(active_heads(&target)) {
            *count += usize::from(active);
        }
    }

    fn conditional_mean(&self) -> f64 {
        self.conditional_entropy / self.conditional_heads.max(1) as f64
    }

    fn confident(&self) -> bool {
        confidence_gate(
            self.frames,
            self.confident,
            self.path_entropy / self.frames as f64,
            self.conditional_mean(),
            self.conditional_heads,
        )
    }
}

fn confidence_gate(
    frames: usize,
    confident: usize,
    path: f64,
    conditional: f64,
    heads: usize,
) -> bool {
    assert!(confident <= frames && frames <= DECISIONS);
    assert!(path.is_finite() && conditional.is_finite());
    frames == DECISIONS
        && confident * 5 > frames * 4
        && path < 0.25
        && conditional < 0.20
        && heads >= 16
}

fn kind_statistics(logits: &[f32; 16], mask: &[bool; 16]) -> (f64, f64) {
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(mask.contains(&true));
    let maximum = logits
        .iter()
        .zip(mask)
        .filter(|(_, allowed)| **allowed)
        .map(|(value, _)| f64::from(*value))
        .fold(f64::NEG_INFINITY, f64::max);
    let masses: [f64; 16] = std::array::from_fn(|index| {
        if mask[index] {
            (f64::from(logits[index]) - maximum).exp()
        } else {
            0.0
        }
    });
    let total: f64 = masses.iter().sum();
    let entropy = masses
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let probability = value / total;
            -probability * probability.ln()
        })
        .sum();
    (1.0 / total, entropy)
}

fn active_heads(target: &BehavioralTarget) -> [bool; 12] {
    [
        target.kind.active,
        target.controlled.active,
        target.ability.active,
        target.item.active,
        target.swap.active,
        target.learn.active,
        target.shop.active,
        target.loot.active,
        target.target_mode.active,
        target.put_mode.active,
        target.entity_pointer.active,
        target.point_pointer.active,
    ]
}

fn conditional_heads(target: &BehavioralTarget) -> usize {
    macro_rules! count {
        ($field:ident) => {
            usize::from(
                target.$field.active
                    && target.$field.mask.iter().filter(|value| **value).count() > 1,
            )
        };
    }
    count!(controlled)
        + count!(ability)
        + count!(item)
        + count!(swap)
        + count!(learn)
        + count!(shop)
        + count!(loot)
        + count!(target_mode)
        + count!(put_mode)
        + count!(entity_pointer)
        + count!(point_pointer)
}

fn measure(
    model: &PolicyModel,
    frame: &FeatureFrame,
    space: &ActionSpace,
    seed: u64,
) -> Measurement {
    let choice = model.choose(frame, space).unwrap();
    let output = model.evaluate(frame).unwrap();
    let sample = model.sample(frame, space, &mut PpoRng::new(seed)).unwrap();
    assert_eq!(choice.value.to_bits(), output.value.to_bits());
    assert_eq!(choice.value.to_bits(), sample.value().to_bits());
    let (probability, kind_entropy) =
        kind_statistics(&output.kind_logits, space.kind_mask().as_array());
    let conditional = f64::from(sample.entropy()) - kind_entropy;
    assert!(
        conditional >= -1e-3,
        "kind/path entropy precision disagreement"
    );
    Measurement {
        choice,
        probability,
        kind_entropy,
        conditional_entropy: conditional.max(0.0),
        conditional_heads: conditional_heads(&sample.target),
        sample,
    }
}

fn collect_side(
    parent: &PolicyModel,
    side: usize,
    summary: &mut Summary,
    observations: &mut Vec<Observation>,
    report: &mut String,
) {
    assert!(side < 2);
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 10104010 + side as u64,
    })
    .unwrap();
    let mut seats = setup_seats(start).unwrap();
    let mut reward = 0.0;
    let mut sent = 0;
    let mut active_continue = 0;
    let mut idle_continue = 0;
    for decision in 0..DECISIONS {
        let (frame, space) = prepare_neural_seat_policy_sample(&mut seats[side]).unwrap();
        let measured = measure(
            parent,
            &frame,
            &space,
            10104200 + side as u64 * 1000 + decision as u64,
        );
        let active = seats[side].local.active_order();
        active_continue +=
            usize::from(measured.choice.action == StructuredAction::Continue && active.is_some());
        idle_continue +=
            usize::from(measured.choice.action == StructuredAction::Continue && active.is_none());
        summary.add(&measured, &frame, &space);
        log_measurement(report, "parent", side, &space, &measured, active);
        let request =
            neural_policy_request_in_space(&mut seats[side], measured.choice.action, &space)
                .unwrap()
                .1;
        sent += usize::from(request.is_some());
        let opponent = teacher_request(&mut seats[1 - side]).unwrap();
        reward += advance_probe(&mut arena, &mut seats, side, request, opponent);
        assert!(observations.len() < 2 * DECISIONS);
        observations.push(Observation {
            side,
            decision,
            frame,
            space,
            measured,
        });
    }
    assert_eq!(arena.tick(), 1800);
    writeln!(report, "SUMMARY side={side} seed={} sent={sent} active_continue={active_continue} idle_continue={idle_continue} observed_prefix_reward={reward:.17} {summary:?}", 10104010 + side as u64).unwrap();
}

fn advance_probe(
    arena: &mut Arena,
    seats: &mut [ArenaSeatPolicy],
    side: usize,
    request: Option<Request>,
    opponent: Option<Request>,
) -> f64 {
    assert_eq!(seats.len(), 2);
    assert!(side < 2);
    let mut reward = 0.0;
    for tick in 0..3 {
        if arena.tick() >= 1800 {
            break;
        }
        let mut orders = [None, None];
        if tick == 0 {
            orders[side] = request;
            orders[1 - side] = opponent;
        }
        let step = arena.step(&orders).unwrap();
        for (index, (seat, messages)) in seats.iter_mut().zip(step.messages).enumerate() {
            assert!(observe_messages(seat, &messages).unwrap().is_none());
            assert_eq!(
                seat.rejections, 0,
                "native rejection: {:?}",
                seat.last_rejection
            );
            let interval = seat.tracker.take_map2_reward_interval().unwrap();
            if index == side {
                reward += interval.total;
            }
        }
    }
    reward
}

fn log_measurement(
    report: &mut String,
    variant: &str,
    side: usize,
    space: &ActionSpace,
    measured: &Measurement,
    active: Option<ActivePolicyOrder>,
) {
    assert!(report.len() < 4 * 1024 * 1024);
    let mask = space.kind_mask().as_array();
    let bits = mask
        .iter()
        .enumerate()
        .fold(0_u32, |bits, (index, active)| {
            bits | (u32::from(*active) << index)
        });
    let legal = mask.iter().filter(|value| **value).count();
    writeln!(report, "ROW\t{variant}\t{side}\t{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}\t{legal}\t{bits}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{:?}",
        space.tick(), measured.choice.value, measured.probability, measured.kind_entropy, measured.sample.entropy(),
        measured.conditional_entropy, measured.conditional_heads,
        space.move_point_mask(ControlledUnit::Hero).iter().filter(|value| **value).count(),
        space.attack_move_point_mask(ControlledUnit::Hero).iter().filter(|value| **value).count(),
        space.attack_entity_mask(ControlledUnit::Hero).iter().filter(|value| **value).count(),
        space.ability_slot_mask(ControlledUnit::Hero).iter().filter(|value| **value).count(),
        measured.choice.action, measured.sample.action(), active).unwrap();
}

fn prepare_softened(parent: &PolicyModel) -> PolicyModel {
    let original = parent.export_parameters().unwrap();
    let softened = parent.softened_actor_parameters().unwrap();
    let mut manifest = String::new();
    for range in &softened.ranges {
        writeln!(
            manifest,
            "name={} shape={:?} begin={} end={} selected={} factor={}",
            range.name,
            range.shape,
            range.begin,
            range.end,
            range.selected,
            if range.selected { 0.5 } else { 1.0 }
        )
        .unwrap();
        for (source, target) in original[range.begin..range.end]
            .iter()
            .zip(&softened.parameters[range.begin..range.end])
        {
            let expected = if range.selected {
                source * 0.5
            } else {
                *source
            };
            assert_eq!(target.to_bits(), expected.to_bits());
        }
    }
    let model = PolicyModel::fresh_on(10104200, PolicyDevice::Cpu).unwrap();
    assert_eq!(
        model.parameter_schema().unwrap(),
        parent.parameter_schema().unwrap()
    );
    model.import_parameters(&softened.parameters).unwrap();
    assert_ne!(
        model.policy_identity().unwrap(),
        parent.policy_identity().unwrap()
    );
    assert_eq!(parent.export_parameters().unwrap(), original);
    write_new("PARAMETER_SELECTION.txt", &manifest);
    model
}

fn check_parity(
    model: &PolicyModel,
    observed: &Observation,
    summary: &mut Summary,
    report: &mut String,
) {
    let after = measure(
        model,
        &observed.frame,
        &observed.space,
        10104200 + observed.side as u64 * 1000 + observed.decision as u64,
    );
    assert_eq!(after.choice.action, observed.measured.choice.action);
    assert_eq!(
        after.choice.value.to_bits(),
        observed.measured.choice.value.to_bits()
    );
    let prior = BehavioralTarget::from_action(
        &observed.frame,
        &observed.space,
        observed.measured.choice.action,
    )
    .unwrap();
    let next = BehavioralTarget::from_action(&observed.frame, &observed.space, after.choice.action)
        .unwrap();
    assert_eq!(
        prior, next,
        "all active head selections, masks and conditional pointers"
    );
    summary.add(&after, &observed.frame, &observed.space);
    log_measurement(
        report,
        "softened",
        observed.side,
        &observed.space,
        &after,
        None,
    );
}

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ROOT)
}

fn load_model(path: &Path) -> PolicyModel {
    let model = PolicyModel::fresh_on(10104200, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, path).unwrap();
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 24);
    assert_eq!(model.device(), PolicyDevice::Cpu);
    model
}

fn file_hash(path: &Path) -> String {
    assert!(path.is_file());
    assert!(std::fs::metadata(path).unwrap().len() <= 8 * 1024 * 1024);
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_new(name: &str, text: &str) {
    use std::io::Write;
    assert!(text.len() < 4 * 1024 * 1024);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root().join(name))
        .unwrap();
    output.write_all(text.as_bytes()).unwrap();
    output.sync_all().unwrap();
}

#[test]
fn confidence_thresholds_are_strict_and_require_conditional_coverage() {
    assert!(confidence_gate(600, 481, 0.24, 0.19, 16));
    assert!(!confidence_gate(600, 480, 0.24, 0.19, 16));
    assert!(!confidence_gate(600, 481, 0.25, 0.19, 16));
    assert!(!confidence_gate(600, 481, 0.24, 0.20, 16));
    assert!(!confidence_gate(600, 481, 0.24, 0.19, 15));
}

#[test]
fn masked_kind_entropy_excludes_illegal_logits_and_handles_singletons() {
    let mut logits = [1000.0; 16];
    logits[0] = 0.0;
    logits[1] = 0.0;
    let mut mask = [false; 16];
    mask[0] = true;
    let (probability, entropy) = kind_statistics(&logits, &mask);
    assert_eq!(probability, 1.0);
    assert_eq!(entropy, 0.0);
    mask[1] = true;
    let (probability, entropy) = kind_statistics(&logits, &mask);
    assert_eq!(probability, 0.5);
    assert!((entropy - std::f64::consts::LN_2).abs() < 1e-12);
}
