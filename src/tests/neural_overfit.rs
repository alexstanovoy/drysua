use super::*;
use crate::{BehavioralTarget, HeadTarget};

const KINDS: [ActionKind; 8] = [
    ActionKind::Buy,
    ActionKind::Learn,
    ActionKind::MovePoint,
    ActionKind::FollowUnit,
    ActionKind::AttackMovePoint,
    ActionKind::AttackUnit,
    ActionKind::Cast,
    ActionKind::Use,
];

struct RealRows {
    reservoir: Reservoir,
    spaces: Vec<ActionSpace>,
}

impl RealRows {
    fn new(seed: u64) -> Self {
        Self {
            reservoir: Reservoir::new(64, seed),
            spaces: Vec::with_capacity(64),
        }
    }

    fn consider(
        &mut self,
        seat: &mut NeuralSeat,
        space: &ActionSpace,
        action: StructuredAction,
        seed: u64,
        namespace: SeedNamespace,
    ) -> Result<Option<usize>> {
        if !KINDS.contains(&action.kind()) {
            return Ok(None);
        }
        self.reservoir
            .consider(seat, space, action, seed, namespace, None)?;
        for (index, sample) in self.reservoir.samples.iter().enumerate() {
            if sample.identity().tick() == space.tick()
                && sample.frame().matches_action_space(space)
            {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn collect(seed: u64, namespace: SeedNamespace, deadline: Instant) -> Result<Self> {
        let mut rows = Self::new(seed);
        let (mut arena, start) = Arena::new(ArenaConfig {
            map: MapId(0),
            seats: 2,
            seed,
        })?;
        let mut seats = [
            NeuralSeat::new(0, &start.messages[0])?,
            NeuralSeat::new(1, &start.messages[1])?,
        ];
        let mut experts = [Some(Teacher::new()), Some(Teacher::new())];
        for _ in 1..18000 {
            if Instant::now() >= deadline {
                return Err("tiny collection deadline exhausted".into());
            }
            let mut requests = [None, None];
            if (arena.tick() - 1).is_multiple_of(3) {
                for (index, seat) in seats.iter_mut().enumerate() {
                    let expert = experts[index].as_mut().expect("expert");
                    let (action, space) =
                        expert.decide(&seat.tracker, &seat.persistence, &seat.readiness)?;
                    let selected = rows.consider(seat, &space, action, seed, namespace)?;
                    if let Some(issued) = seat.issue(action, &space)? {
                        expert.note_sent(seat.sequence, issued, space.tick());
                        requests[index] = Some(Request {
                            seq: seat.sequence,
                            unit: issued.unit,
                            order: issued.order,
                        });
                    }
                    if let Some(selected) = selected {
                        let spaces = &mut rows.spaces;
                        if selected == spaces.len() {
                            spaces.push(space);
                        } else {
                            spaces[selected] = space;
                        }
                    }
                }
            }
            let step = arena.step(&requests)?;
            if observe_game_tick(&mut seats, &mut experts, &step.messages, seed, &mut None)?
                .is_some()
            {
                break;
            }
        }
        eprintln!(
            "collection seed={seed} ticks={} seen={:?}",
            arena.tick(),
            rows.reservoir.counts
        );
        Ok(rows)
    }

    fn pairs(&self) -> Vec<(&ImitationSample, &ActionSpace)> {
        assert_eq!(self.reservoir.samples.len(), self.spaces.len());
        let pairs: Vec<_> = self.reservoir.samples.iter().zip(&self.spaces).collect();
        assert!(pairs.len() <= 64);
        assert!(
            pairs
                .iter()
                .all(|(sample, space)| sample.frame().matches_action_space(space))
        );
        pairs
    }
}

#[test]
#[ignore = "bounded real-game collector integration test"]
fn real_tiny_collector_preserves_frames_masks_and_body_labels() {
    let rows = RealRows::collect(
        9_873_100,
        SeedNamespace::Training,
        Instant::now() + Duration::from_secs(60),
    )
    .expect("real collector");
    let pairs = rows.pairs();
    assert!(!pairs.is_empty());
    let mut body_counts = [[0; 3]; ActionKind::COUNT];
    for (sample, space) in pairs {
        assert_ne!(sample.teacher_action().kind(), ActionKind::Continue);
        assert_eq!(
            *sample.target(),
            BehavioralTarget::from_action(sample.frame(), space, sample.teacher_action())
                .expect("exact preserved target")
        );
        let body = &sample.target().controlled;
        body_counts[sample.teacher_action().kind().index()]
            [if body.active { body.selected } else { 2 }] += 1;
    }
    eprintln!("collector_body_counts=[hero,courier,no_body] {body_counts:?}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "bounded real-data CUDA diagnostic; no match gate or promotion"]
fn real_tiny_batch_gpu_overfit_diagnostic() {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(600);
    let train =
        RealRows::collect(9_873_100, SeedNamespace::Training, deadline).expect("train rows");
    let held =
        RealRows::collect(9_873_101, SeedNamespace::Validation, deadline).expect("held rows");
    eprintln!(
        "model_schema={} model_hash={} feature_schema={} feature_hash={} dtype=F32 seed=9101",
        crate::MODEL_SCHEMA_VERSION,
        crate::MODEL_SCHEMA_HASH,
        crate::FEATURE_SCHEMA_VERSION,
        crate::FEATURE_SCHEMA_HASH
    );
    write_rows(&train, &held);
    audit_collisions(&train);
    let seeds =
        SeedNamespaces::new(vec![9_873_100], vec![9_873_101], vec![9_873_102]).expect("seeds");
    let mut pool = ImitationPool::new(
        128,
        9101,
        seeds,
        TrainingScope::new(MapId(0), IMITATION_RULES_AUDIT_VERSION).expect("scope"),
    )
    .expect("pool");
    for (sample, _) in train.pairs() {
        assert!(pool.push(sample.clone()).expect("push").is_none());
    }
    let model = PolicyModel::fresh_on(9101, PolicyDevice::Cuda { ordinal: 0 }).expect("CUDA F32");
    let mut trainer = BehavioralTrainer::new(
        64,
        9102,
        AdamConfig {
            learning_rate: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            gradient_clip: 0.5,
        },
        &model,
        &pool,
    )
    .expect("trainer");
    report(&model, &train, "train", 0);
    for update in 1..=2048 {
        assert!(
            Instant::now() < deadline,
            "bounded overfit deadline exhausted"
        );
        let epoch = trainer.train_epoch(&model, &pool).expect("update");
        if [256, 1024, 2048].contains(&update) {
            eprintln!(
                "update={update} loss={} elapsed={:.3}",
                epoch.average_loss,
                started.elapsed().as_secs_f64()
            );
            report(&model, &train, "train", update);
            report(&model, &held, "near_held", update);
        }
    }
    assert_eq!(trainer.counters().global_update, 2048);
}

#[cfg(feature = "cuda")]
fn write_rows(train: &RealRows, held: &RealRows) {
    let mut artifact = File::create("/tmp/opencode/neural-overfit-rows.txt").expect("row artifact");
    for (split, rows) in [("train", train), ("near_held", held)] {
        assert!(rows.pairs().len() <= 64);
        for (sample, space) in rows.pairs() {
            assert_eq!(
                *sample.target(),
                BehavioralTarget::from_action(sample.frame(), space, sample.teacher_action())
                    .expect("preserved label and mask")
            );
            writeln!(artifact, "split={split} {sample:?}").expect("write real row");
        }
    }
}

#[cfg(feature = "cuda")]
fn audit_collisions(rows: &RealRows) {
    let pairs = rows.pairs();
    let mut identical = 0;
    let mut conflicts = 0;
    for (index, (sample, _)) in pairs.iter().enumerate() {
        for (other, _) in &pairs[..index] {
            if sample.frame() == other.frame() {
                identical += 1;
                if sample.teacher_action() != other.teacher_action() {
                    conflicts += 1;
                }
            }
        }
    }
    eprintln!("raw_input_collision_pairs={identical} conflicting_label_pairs={conflicts}");
    assert!(pairs.len() <= 64);
}

#[cfg(feature = "cuda")]
fn report(model: &PolicyModel, rows: &RealRows, split: &str, update: usize) {
    let pairs = rows.pairs();
    let frames: Vec<_> = pairs
        .iter()
        .map(|(sample, _)| sample.frame().clone())
        .collect();
    let choices: Vec<_> = pairs
        .iter()
        .map(|(sample, space)| {
            model
                .choose(sample.frame(), space)
                .expect("autoregressive choice")
        })
        .collect();
    let prefixes: Vec<_> = pairs
        .iter()
        .map(|(sample, _)| sample.target().prefix())
        .collect();
    let output = model
        .training_forward(&frames, &prefixes)
        .expect("teacher forced");
    let mut counts = [[0; 3]; ActionKind::COUNT];
    let mut heads = std::collections::BTreeMap::<&str, [f64; 5]>::new();
    for (index, ((sample, space), choice)) in pairs.iter().zip(choices).enumerate() {
        let count = &mut counts[sample.teacher_action().kind().index()];
        count[0] += 1;
        count[1] += usize::from(choice.action.kind() == sample.teacher_action().kind());
        count[2] += usize::from(choice.action == sample.teacher_action());
        let predicted = BehavioralTarget::from_action(sample.frame(), space, choice.action)
            .expect("chosen target");
        let target = sample.target();
        let mut prefix = true;
        macro_rules! head {
            ($name:ident) => {{
                let logits = output.$name().to_vec2::<f32>().expect("head");
                let metric = heads.entry(stringify!($name)).or_default();
                record_head(
                    &logits[index],
                    &target.$name,
                    &predicted.$name,
                    &mut prefix,
                    metric,
                );
            }};
        }
        head!(kind);
        head!(controlled);
        head!(ability);
        head!(item);
        head!(swap);
        head!(learn);
        head!(shop);
        head!(loot);
        head!(target_mode);
        head!(put_mode);
        head!(entity_pointer);
        head!(point_pointer);
    }
    eprintln!(
        "split={split} update={update} per_kind=[total,kind_correct,autoregressive_exact] {counts:?}"
    );
    eprintln!(
        "split={split} update={update} heads=[active,teacher_ce_sum,teacher_correct,actual_prefix_count,actual_prefix_correct] {heads:?}"
    );
}

fn record_head<const WIDTH: usize>(
    logits: &[f32],
    target: &HeadTarget<WIDTH>,
    predicted: &HeadTarget<WIDTH>,
    prefix: &mut bool,
    metric: &mut [f64; 5],
) {
    assert_eq!(logits.len(), WIDTH);
    if !target.active {
        return;
    }
    assert!(target.is_selected_legal());
    let best = (0..WIDTH)
        .filter(|&index| target.mask[index])
        .max_by(|&left, &right| {
            logits[left]
                .total_cmp(&logits[right])
                .then(right.cmp(&left))
        })
        .expect("legal");
    let maximum = f64::from(logits[best]);
    let normalizer: f64 = (0..WIDTH)
        .filter(|&index| target.mask[index])
        .map(|index| (f64::from(logits[index]) - maximum).exp())
        .sum();
    metric[0] += 1.0;
    metric[1] += normalizer.ln() + maximum - f64::from(logits[target.selected]);
    metric[2] += f64::from(best == target.selected);
    if *prefix {
        metric[3] += 1.0;
        metric[4] += f64::from(predicted.active && predicted.selected == target.selected);
    }
    *prefix &= predicted.active && predicted.selected == target.selected;
}

#[test]
fn teacher_correct_pointer_does_not_count_when_actual_prefix_is_wrong() {
    let target = HeadTarget {
        active: true,
        mask: [true, true],
        selected: 1,
    };
    let mut metric = [0.0; 5];
    let mut prefix = false;
    record_head(&[0.0, 2.0], &target, &target, &mut prefix, &mut metric);
    assert_eq!(metric[2], 1.0);
    assert_eq!(metric[3], 0.0);
    assert_eq!(metric[4], 0.0);
    assert!(!prefix);
}

#[test]
fn masked_head_loss_ignores_illegal_logits_and_counts_correct_prefix() {
    let target = HeadTarget {
        active: true,
        mask: [true, false],
        selected: 0,
    };
    let mut metric = [0.0; 5];
    let mut prefix = true;
    record_head(&[0.0, 100.0], &target, &target, &mut prefix, &mut metric);
    assert_eq!(metric, [1.0, 0.0, 1.0, 1.0, 1.0]);
    assert!(prefix);
}

#[test]
fn wrong_body_branch_breaks_actual_prefix_before_slot_and_pointer() {
    let target = HeadTarget {
        active: true,
        mask: [true, true],
        selected: 0,
    };
    let predicted = HeadTarget {
        selected: 1,
        ..target
    };
    let mut metric = [0.0; 5];
    let mut prefix = true;
    record_head(&[2.0, 0.0], &target, &predicted, &mut prefix, &mut metric);
    assert_eq!(metric[2], 1.0);
    assert_eq!(metric[3], 1.0);
    assert_eq!(metric[4], 0.0);
    assert!(!prefix);
}
