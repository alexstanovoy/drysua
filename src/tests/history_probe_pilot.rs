use super::*;
use crate::{BehavioralPrediction, BehavioralTarget, HeadTarget, ImitationSplit};
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Write};

#[path = "active_target_probe.rs"]
mod active_target;
#[path = "history_probe_replay.rs"]
mod replay;
#[path = "history_probe_pilot_tests.rs"]
mod tests;

const INITIAL: &str = "artifacts/temp/neural-reset-20260908/order-contract-m14/initial";
const INITIAL_SHA256: &str = "ce17f6f1e3668837776b366fdcab59c250a4d6bb6de3ba3615dc3e7f23a53c4b";
const VARIANTS: [&str; 2] = ["baseline", "enriched"];

const CONDITIONING_GLOBALS: [usize; 6] = [0, 33, 35, 52, 53, 54];
const CONDITIONING_UPDATES: usize = 680;
const CONDITIONING_BATCH: usize = 64;
const CONDITIONING_PRESENTATIONS: usize = CONDITIONING_UPDATES * CONDITIONING_BATCH;
const CONDITIONING_VARIANTS: [(&str, bool, bool); 4] = [
    ("A-natural-raw", false, false),
    ("B-balanced-raw", true, false),
    ("C-natural-rescaled-folded", false, true),
    ("D-balanced-rescaled-folded", true, true),
];

struct ConditioningRow {
    index: usize,
    raw: ImitationSample,
    alternate: ImitationSample,
    target_summary: [f32; 5],
}

impl ConditioningRow {
    fn new(index: usize, raw: ImitationSample, space: &ActionSpace) -> Self {
        assert!(index < 8 * 1488);
        assert!(raw.frame().matches_action_space(space));
        let scaled = ImitationSample::teacher(
            conditioning_frame(raw.frame()),
            space,
            raw.teacher_action(),
            raw.identity(),
        )
        .expect("same-label rescaled example");
        assert_eq!(raw.target(), scaled.target());
        assert_eq!(raw.identity(), scaled.identity());
        Self {
            index,
            raw,
            alternate: scaled,
            target_summary: [0.0; 5],
        }
    }

    fn sample(&self, scaled: bool) -> &ImitationSample {
        assert_eq!(self.raw.identity(), self.alternate.identity());
        assert_eq!(self.raw.target(), self.alternate.target());
        if scaled { &self.alternate } else { &self.raw }
    }
}

#[test]
#[ignore = "Predeclared matched sampling/rescaling 2x2; one recorded replay, no gameplay evaluation"]
fn probe_matched_kind_conditioning() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root.join("artifacts/temp/neural-reset-20260908/kind-conditioning-001");
    let original = root.join("artifacts/temp/neural-reset-20260908/training-contract-bc");
    assert!(
        output.join("experiment-manifest.json").is_file(),
        "freeze source/binary first"
    );
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 12);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 14);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 27);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 22);
    assert_eq!(
        file_sha256(&root.join(INITIAL).join("drysua.weights.safetensors")),
        INITIAL_SHA256
    );
    let (rows, spaces) = replay::reconstruct_conditioning_rows(&original, &output);
    let started = Instant::now();
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let parent = PolicyModel::fresh_on(10089200, device).expect("INITIAL parent");
    TrainingArtifact::load_runtime_weights(&parent, &root.join(INITIAL))
        .expect("M14 INITIAL runtime");
    let parameters = zero_probe_input_weights(&parent);
    parent
        .import_parameters(&parameters)
        .expect("matched zero reserved rows");
    assert_eq!(
        parameter_sha256(&parameters),
        "c23b5357787f67ab2d64af4282f2147841ba61635673ecdf809dfa75957c99be"
    );
    conditioning_train_variants(parent, rows, spaces, device, output, started);
    eprintln!(
        "conditioning_complete fit_seconds={:.3} variants=4 updates_per_variant=680 samples_per_variant=43520 gameplay_evaluated=false",
        started.elapsed().as_secs_f64()
    );
}

fn conditioning_train_variants(
    parent: PolicyModel,
    rows: Vec<ConditioningRow>,
    spaces: Vec<ActionSpace>,
    device: PolicyDevice,
    output: std::path::PathBuf,
    started: Instant,
) {
    assert_eq!(rows.len(), spaces.len());
    let parameters = parent.export_parameters().expect("zero-row INITIAL");
    let references: Vec<_> = rows.iter().map(|row| &row.raw).collect();
    let schedules = [
        conditioning_schedule(&references, false),
        conditioning_schedule(&references, true),
    ];
    conditioning_schedule_audit(&rows, &schedules, &output);
    let mut predictions = new_writer(&output.join("predictions.tsv"));
    writeln!(
        predictions,
        "variant\tupdate\tindex\tpredicted\tkind_correct\tfull_correct"
    )
    .expect("header");
    for (name, balanced, scaled) in CONDITIONING_VARIANTS {
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "total conditioning fit budget"
        );
        let model = PolicyModel::fresh_on(10089200, device).expect("independent learner");
        let initial = if scaled {
            conditioning_parameters(&parent, 1.0 / 64.0)
        } else {
            parameters.clone()
        };
        model
            .import_parameters(&initial)
            .expect("fresh matched initialization");
        conditioning_equivalence(
            &parent,
            &model,
            &rows,
            &spaces,
            scaled,
            &format!("initial-{name}"),
        );
        conditioning_metrics(&model, &rows, scaled, name, 0, &mut predictions);
        conditioning_fit(
            &model,
            &rows,
            scaled,
            &schedules[usize::from(balanced)],
            name,
            started,
        );
        conditioning_metrics(&model, &rows, scaled, name, 680, &mut predictions);
        conditioning_export(&model, &rows, &spaces, scaled, name, device, &output);
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "total conditioning fit budget"
        );
    }
    predictions.flush().expect("all fixed endpoint predictions");
}

fn conditioning_fit(
    model: &PolicyModel,
    rows: &[ConditioningRow],
    scaled: bool,
    schedule: &[usize],
    name: &str,
    started: Instant,
) {
    assert_eq!(schedule.len(), CONDITIONING_PRESENTATIONS);
    assert_eq!(CONDITIONING_BATCH, crate::MODEL_TRAINING_BATCH);
    let mut adam = model
        .claim_optimizer(AdamConfig {
            learning_rate: 3e-5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .expect("fresh Adam");
    assert_eq!(adam.step(), 0);
    let mut loss_sum = 0.0;
    let mut clipped = 0;
    let mut clip_scale_sum = 0.0f64;
    let (batches, remainder) = schedule.as_chunks::<CONDITIONING_BATCH>();
    assert!(remainder.is_empty());
    for (index, batch) in batches.iter().enumerate() {
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "total conditioning fit budget"
        );
        let samples: Vec<_> = batch
            .iter()
            .map(|&index| rows[index].sample(scaled))
            .collect();
        let report = model
            .behavioral_update(&samples, &mut adam)
            .expect("bounded training-only update");
        assert_eq!(report.sample_count, 64);
        assert_eq!(report.optimizer_step, (index + 1) as u64);
        loss_sum += report.average_loss;
        clipped += usize::from(report.applied_scale < 1.0);
        clip_scale_sum += report.applied_scale;
    }
    assert_eq!(adam.step(), 680);
    eprintln!(
        "conditioning_fit variant={name} updates={} samples={} mean_training_loss={} clipped_updates={clipped} mean_clip_scale={}",
        adam.step(),
        schedule.len(),
        loss_sum / 680.0,
        clip_scale_sum / 680.0
    );
}

fn conditioning_schedule_audit(
    rows: &[ConditioningRow],
    schedules: &[Vec<usize>; 2],
    output: &Path,
) {
    assert_eq!(rows.len(), 10783);
    let mut writer = new_writer(&output.join("schedules.tsv"));
    writeln!(writer, "sampling\tpresentation\tindex").expect("schedule header");
    for (balanced, schedule) in schedules.iter().enumerate() {
        let mut counts = [[0usize; ActionKind::COUNT]; 2];
        let mut seen = vec![false; rows.len()];
        let mut digest = Sha256::new();
        for (presentation, &index) in schedule.iter().enumerate() {
            let sample = &rows[index].raw;
            assert_eq!(sample.split(), ImitationSplit::Train);
            counts[usize::from(sample.side() == ImitationSide::Dire)]
                [sample.teacher_action().kind().index()] += 1;
            seen[index] = true;
            digest.update((index as u64).to_le_bytes());
            writeln!(writer, "{balanced}\t{presentation}\t{index}").expect("exact schedule");
        }
        eprintln!(
            "conditioning_schedule balanced={balanced} samples={} unique={} radiant={:?} dire={:?} index_sha256={}",
            schedule.len(),
            seen.iter().filter(|value| **value).count(),
            counts[0],
            counts[1],
            digest_hex(digest.finalize().into())
        );
    }
    writer.flush().expect("schedules");
}

#[derive(Default)]
struct ConditioningMetrics {
    confusion: [[usize; ActionKind::COUNT]; ActionKind::COUNT],
    full: [usize; ActionKind::COUNT],
}

impl ConditioningMetrics {
    fn record(
        &mut self,
        sample: &ImitationSample,
        prediction: BehavioralPrediction,
    ) -> (bool, bool) {
        let kind = sample.teacher_action().kind().index();
        assert!(prediction.kind < ActionKind::COUNT);
        assert!(self.confusion[kind][prediction.kind] < 8 * 1488);
        let full = prediction == target_prediction(sample.target());
        self.confusion[kind][prediction.kind] += 1;
        self.full[kind] += usize::from(full);
        (prediction.kind == kind, full)
    }
}

fn conditioning_metrics(
    model: &PolicyModel,
    rows: &[ConditioningRow],
    scaled: bool,
    name: &str,
    update: u64,
    writer: &mut impl Write,
) {
    assert!([0, 680].contains(&update));
    assert!(rows.len() <= 8 * 1488);
    let mut metrics: [[ConditioningMetrics; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| ConditioningMetrics::default()));
    for batch in rows.chunks(64) {
        let samples: Vec<_> = batch.iter().map(|row| row.sample(scaled)).collect();
        let predictions = model
            .behavioral_predictions(&samples)
            .expect("bounded predictions");
        for (row, prediction) in batch.iter().zip(predictions) {
            let sample = row.sample(scaled);
            let split = match sample.split() {
                ImitationSplit::Train => 0,
                ImitationSplit::Validation => 1,
                ImitationSplit::HeldOut => panic!("no promotion data"),
            };
            let side = usize::from(sample.side() == ImitationSide::Dire) + 1;
            metrics[split][0].record(sample, prediction);
            let (kind, full) = metrics[split][side].record(sample, prediction);
            writeln!(
                writer,
                "{name}\t{update}\t{}\t{}\t{}\t{}",
                row.index,
                prediction.kind,
                usize::from(kind),
                usize::from(full)
            )
            .expect("prediction evidence");
        }
    }
    for (split, sides) in ["Train", "Validation"].into_iter().zip(metrics) {
        for (side, values) in ["Overall", "Radiant", "Dire"].into_iter().zip(sides) {
            eprintln!(
                "conditioning_metrics variant={name} update={update} split={split} side={side} confusion={:?} full={:?}",
                values.confusion, values.full
            );
        }
    }
}

fn conditioning_export(
    model: &PolicyModel,
    rows: &[ConditioningRow],
    spaces: &[ActionSpace],
    scaled: bool,
    name: &str,
    device: PolicyDevice,
    output: &Path,
) {
    assert!(
        CONDITIONING_VARIANTS
            .iter()
            .any(|variant| variant.0 == name)
    );
    assert!(
        rows.iter()
            .all(|row| row.raw.frame().global[59..64] == [0.0; 5])
    );
    let folded = PolicyModel::fresh_on(10089502, device).expect("raw-F12 folded model");
    let parameters = if scaled {
        conditioning_parameters(model, 64.0)
    } else {
        model.export_parameters().expect("raw parameters")
    };
    folded
        .import_parameters(&parameters)
        .expect("fold scale OUT into raw-input model");
    conditioning_equivalence(
        &folded,
        model,
        rows,
        spaces,
        scaled,
        &format!("fold-{name}"),
    );
    let directory = output.join(name);
    std::fs::create_dir(&directory).expect("new lawful terminal artifact");
    TrainingArtifact::save_runtime_weights(&folded, &directory).expect("ordinary raw F12 export");
    let restored = PolicyModel::fresh_on(10089503, device).expect("independent runtime reload");
    TrainingArtifact::load_runtime_weights(&restored, &directory)
        .expect("current runtime metadata");
    assert_eq!(
        parameter_sha256(&restored.export_parameters().expect("reloaded parameters")),
        parameter_sha256(&parameters)
    );
    let weights = file_sha256(&directory.join("drysua.weights.safetensors"));
    std::fs::write(directory.join("PROVENANCE.txt"), format!("variant={name}\nparent=INITIAL_M14_initialization\nparent_sha256={INITIAL_SHA256}\nweights_sha256={weights}\nparameter_sha256={}\nsource_manifest_sha256={}\ncohort_rows_sha256={}\nschedules_sha256={}\nupdates=680\neffective_batch=64\nsample_presentations=43520\noptimizer=fresh_Adam_lr3e-5_beta1.9_beta2.999_epsilon1e-8_clip.5\nrescaled_training={scaled}\nscale=64\nselected_globals=0,33,35,52,53,54\nruntime_input=unchanged_raw_F12\nfold_equivalence_passed=true\ninitial_reserved59..63=zero\nenriched_history=false\nqualified=false\nautomatic_promotion=false\ngameplay_evaluated=false\n", parameter_sha256(&parameters), file_sha256(&output.join("manifest.json")), file_sha256(&output.join("rows.tsv")), file_sha256(&output.join("schedules.tsv")))).expect("training provenance");
    eprintln!(
        "conditioning_export variant={name} weights_sha256={weights} raw_F12=true fold_equivalence_passed=true"
    );
}

fn conditioning_equivalence(
    raw: &PolicyModel,
    candidate: &PolicyModel,
    rows: &[ConditioningRow],
    spaces: &[ActionSpace],
    scaled: bool,
    label: &str,
) {
    assert!(!rows.is_empty());
    assert!(rows.len() <= 8 * 1488);
    assert_eq!(rows.len(), spaces.len());
    let mut head_error = 0.0f32;
    for (batch, spaces) in rows.chunks(64).zip(spaces.chunks(64)) {
        let raw_frames: Vec<_> = batch.iter().map(|row| row.raw.frame().clone()).collect();
        let candidate_frames: Vec<_> = batch
            .iter()
            .map(|row| row.sample(scaled).frame().clone())
            .collect();
        let prefixes: Vec<_> = batch.iter().map(|row| row.raw.target().prefix()).collect();
        head_error = head_error.max(conditioning_head_error(
            raw,
            candidate,
            &raw_frames,
            &candidate_frames,
            &prefixes,
        ));
        let expected = raw
            .choose_batch(&raw_frames, spaces)
            .expect("raw greedy decoder");
        let actual = candidate
            .choose_batch(&candidate_frames, spaces)
            .expect("training-function greedy decoder");
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(
                expected.action, actual.action,
                "fold/initial greedy action mismatch"
            );
            conditioning_close(expected.value, actual.value).expect("greedy value equivalence");
        }
        let expected: Vec<_> = batch.iter().map(|row| &row.raw).collect();
        let actual: Vec<_> = batch.iter().map(|row| row.sample(scaled)).collect();
        assert!(
            raw.behavioral_predictions(&expected).expect("raw labels")
                == candidate
                    .behavioral_predictions(&actual)
                    .expect("scaled labels"),
            "fold/initial teacher-forced prediction mismatch"
        );
    }
    let (statistic_error, statistics) =
        conditioning_statistics_error(raw, candidate, rows, spaces, scaled);
    eprintln!(
        "conditioning_equivalence label={label} frames={} heads=13 max_head_absolute_error={head_error} greedy_action_matches={} teacher_forced_predictions_match=true statistics_rows={statistics} max_statistics_absolute_error={statistic_error} abs_tolerance=2e-5 rel_tolerance=2e-5",
        rows.len(),
        rows.len()
    );
}

fn conditioning_head_error(
    raw: &PolicyModel,
    candidate: &PolicyModel,
    raw_frames: &[FeatureFrame],
    candidate_frames: &[FeatureFrame],
    prefixes: &[crate::TrainingPrefix],
) -> f32 {
    assert!(raw_frames.len() <= 64);
    assert_eq!(raw_frames.len(), candidate_frames.len());
    let expected = raw
        .training_forward(raw_frames, prefixes)
        .expect("raw all-head function");
    let actual = candidate
        .training_forward(candidate_frames, prefixes)
        .expect("scaled all-head function");
    let mut maximum = 0.0f32;
    for (expected, actual) in conditioning_heads(&expected)
        .into_iter()
        .zip(conditioning_heads(&actual))
    {
        assert_eq!(expected.dims(), actual.dims());
        let expected = expected
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host logits");
        let actual = actual
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host logits");
        for (source, target) in expected.iter().zip(actual) {
            maximum = maximum.max(
                conditioning_close(*source, target)
                    .expect("fold/initial all-head numerical equivalence"),
            );
        }
    }
    maximum
}

fn conditioning_heads<'output>(
    output: &'output crate::PolicyTensorOutput<'_>,
) -> [&'output candle_core::Tensor; 13] {
    [
        output.value(),
        output.kind(),
        output.controlled(),
        output.ability(),
        output.item(),
        output.swap(),
        output.learn(),
        output.shop(),
        output.loot(),
        output.target_mode(),
        output.put_mode(),
        output.entity_pointer(),
        output.point_pointer(),
    ]
}

fn conditioning_close(source: f32, target: f32) -> Result<f32, &'static str> {
    if !source.is_finite() || !target.is_finite() {
        return Err("non-finite conditioning equivalence value");
    }
    let difference = (source - target).abs();
    if difference > 2e-5 + 2e-5 * source.abs().max(target.abs()) {
        return Err("conditioning equivalence tolerance exceeded");
    }
    Ok(difference)
}

fn conditioning_statistics_error(
    raw: &PolicyModel,
    candidate: &PolicyModel,
    rows: &[ConditioningRow],
    spaces: &[ActionSpace],
    scaled: bool,
) -> (f32, usize) {
    assert!(rows.len() <= 8 * 1488);
    let mut counts = [[[0usize; ActionKind::COUNT]; 2]; 2];
    let mut maximum = 0.0f32;
    let mut checked = 0;
    assert_eq!(rows.len(), spaces.len());
    for (row, space) in rows.iter().zip(spaces) {
        let sample = &row.raw;
        let split = usize::from(sample.split() == ImitationSplit::Validation);
        let side = usize::from(sample.side() == ImitationSide::Dire);
        let count = &mut counts[split][side][sample.teacher_action().kind().index()];
        if *count == 4 {
            continue;
        }
        *count += 1;
        let source = raw
            .action_statistics(sample.frame(), space, sample.teacher_action())
            .expect("raw path logprob/entropy/value");
        let target = candidate
            .action_statistics(row.sample(scaled).frame(), space, sample.teacher_action())
            .expect("training path logprob/entropy/value");
        for (source, target) in [source.0, source.1, source.2]
            .into_iter()
            .zip([target.0, target.1, target.2])
        {
            maximum = maximum.max(
                conditioning_close(source, target)
                    .expect("fold/initial path statistics equivalence"),
            );
        }
        checked += 1;
    }
    assert!(checked <= 4 * 2 * 2 * ActionKind::COUNT);
    (maximum, checked)
}

fn conditioning_frame(frame: &FeatureFrame) -> FeatureFrame {
    assert!(frame.is_finite());
    assert_eq!(frame.global.len(), 72);
    assert_eq!(frame.global[59..64], [0.0; 5], "no enriched history");
    let mut scaled = frame.clone();
    for index in CONDITIONING_GLOBALS {
        scaled.global[index] *= 64.0;
    }
    assert!(scaled.is_finite(), "rescaling must not overflow");
    scaled
}

fn conditioning_parameters(model: &PolicyModel, factor: f32) -> Vec<f32> {
    assert_eq!(model.parameter_count(), crate::MODEL_PARAMETER_COUNT);
    assert!(
        [64.0, 1.0 / 64.0].contains(&factor),
        "conditioning weight factor must be 64 or 1/64"
    );
    let mut values = model.export_parameters().expect("parameters");
    let offset = conditioning_trunk_offset(model);
    for row in CONDITIONING_GLOBALS {
        for value in &mut values[offset + row * 512..offset + (row + 1) * 512] {
            *value *= factor;
        }
    }
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn conditioning_trunk_offset(model: &PolicyModel) -> usize {
    use crate::global_feature as global;
    assert_eq!(
        CONDITIONING_GLOBALS,
        [
            global::TICK,
            global::ACTIVE_ORDER_AGE,
            global::TICKS_SINCE_DECISION,
            global::ACTIVE_TARGET_RELATIVE_X,
            global::ACTIVE_TARGET_RELATIVE_Y,
            global::ACTIVE_TARGET_DISTANCE
        ]
    );
    let mut offset = 0;
    let mut trunk = None;
    for (name, shape) in model.parameter_schema().expect("audited parameter layout") {
        if name == "trunk.0.weight" {
            assert_eq!(shape, [2576, 512]);
            trunk = Some(offset);
        }
        offset += shape.iter().product::<usize>();
    }
    assert_eq!(offset, crate::MODEL_PARAMETER_COUNT);
    trunk.expect("scalar-first input/output trunk weight")
}

fn conditioning_schedule(samples: &[&ImitationSample], balanced: bool) -> Vec<usize> {
    assert!(!samples.is_empty());
    assert!(samples.len() <= 8 * 1488);
    let mut groups: [_; ActionKind::COUNT] = std::array::from_fn(|_| ConditioningCycle::default());
    for (index, sample) in samples.iter().enumerate() {
        if sample.split() == ImitationSplit::Train {
            groups[if balanced {
                sample.teacher_action().kind().index()
            } else {
                0
            }]
            .values
            .push(index);
        }
    }
    let mut kinds = ConditioningCycle {
        values: (0..ActionKind::COUNT)
            .filter(|&kind| !groups[kind].values.is_empty())
            .collect(),
        cursor: 0,
    };
    assert!(
        !kinds.values.is_empty(),
        "conditioning requires training rows"
    );
    let mut random = PpoRng::new(10089501);
    let schedule: Vec<_> = (0..CONDITIONING_PRESENTATIONS)
        .map(|_| {
            let kind = kinds.next(&mut random);
            groups[kind].next(&mut random)
        })
        .collect();
    assert_eq!(schedule.len(), 43520);
    assert!(
        schedule
            .iter()
            .all(|&index| samples[index].split() == ImitationSplit::Train)
    );
    schedule
}

#[derive(Default)]
struct ConditioningCycle {
    values: Vec<usize>,
    cursor: usize,
}

impl ConditioningCycle {
    fn next(&mut self, random: &mut PpoRng) -> usize {
        assert!(!self.values.is_empty());
        assert!(self.values.len() <= 8 * 1488);
        if self.cursor == 0 {
            for index in (1..self.values.len()).rev() {
                let other = random.below((index + 1) as u64).expect("bounded shuffle") as usize;
                self.values.swap(index, other);
            }
        }
        let value = self.values[self.cursor];
        self.cursor = (self.cursor + 1) % self.values.len();
        value
    }
}

#[test]
#[ignore = "Predeclared eight-game matched BC/history pilot; only baseline F12 weights exported"]
fn probe_matched_history_after_training_contract() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 12);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 14);
    assert_eq!(crate::PPO_SCHEMA_VERSION, 27);
    assert_eq!(crate::PPO_RULES_AUDIT_VERSION, 22);
    assert_eq!(IMITATION_RULES_AUDIT_VERSION, 13);
    let output =
        std::env::var_os("DRYSUA_MATCHED_HISTORY_OUTPUT").expect("explicit NEW run directory");
    let output = Path::new(&output);
    assert!(
        output.join("manifest.json").is_file(),
        "freeze binary and source before running"
    );
    assert!(!output.join("baseline-epoch004").exists());
    assert!(!output.join("baseline-epoch008").exists());
    let initial = Path::new(env!("CARGO_MANIFEST_DIR")).join(INITIAL);
    assert_eq!(
        file_sha256(&initial.join("drysua.weights.safetensors")),
        INITIAL_SHA256
    );
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let actor = PolicyModel::fresh_on(10089200, device).expect("M14 initialization opponent");
    TrainingArtifact::load_runtime_weights(&actor, &initial).expect("audited INITIAL runtime load");
    let frozen_parameters = actor
        .export_parameters()
        .expect("immutable opponent parameters");
    eprintln!(
        "history_plan opponent=M14_initialization_INITIAL current_candidate=true teacher_observer=true parent_sha256={INITIAL_SHA256} no_historical_M12_identity=true"
    );
    let (pools, coverage) = collect_history_pools(&actor, Some(output));
    assert_eq!(
        actor.export_parameters().expect("opponent unchanged"),
        frozen_parameters
    );
    let started = Instant::now();
    let parameters = zero_probe_input_weights(&actor);
    let models: [_; 2] = std::array::from_fn(|_| {
        let model = PolicyModel::fresh_on(10089200, device).expect("fresh learner");
        model
            .import_parameters(&parameters)
            .expect("matched zero-row INITIAL");
        model
    });
    eprintln!(
        "history_initial parameter_sha256={} zero_incoming_rows=59..63 both_models=true frozen_opponent_unchanged=true",
        parameter_sha256(&parameters)
    );
    write_retained_rows(&pools, output);
    assert_initial_outputs(&models, &pools);
    fit_matched_models(&models, &pools, &coverage, output, output, started);
    assert_eq!(
        file_sha256(&initial.join("drysua.weights.safetensors")),
        INITIAL_SHA256
    );
    eprintln!(
        "history_complete fit_seconds={:.3} enriched_exported=false automatic_promotion=false",
        started.elapsed().as_secs_f64()
    );
}

pub(super) fn open_transcripts(games: &mut [HistoryGame], output: &Path) {
    assert_eq!(games.len(), 8);
    assert!(output.is_dir());
    for game in games {
        let side = 1 - game.environment.policy_seat;
        let path = output.join(format!("trajectory-{}-side{side}.tsv", game.seed));
        let mut writer = new_writer(&path);
        writeln!(
            writer,
            "tick\tteacher_action\tradiant_request\tdire_request\tprevious_sent_features"
        )
        .expect("transcript header");
        game.transcript = Some(writer);
    }
}

pub(super) fn record_history_round(
    game: &mut HistoryGame,
    requests: &[Option<Request>],
    samples: &[ImitationSample; 2],
) {
    assert_eq!(requests.len(), 2);
    assert!(game.counts.iter().sum::<u32>() < 36300);
    assert_eq!(samples[0].identity(), samples[1].identity());
    game.counts[samples[0].teacher_action().kind().index()] += 1;
    if let Some(writer) = &mut game.transcript {
        let line = format!(
            "{}\t{:?}\t{:?}\t{:?}\t{:?}\n",
            samples[0].identity().tick(),
            samples[0].teacher_action(),
            requests[0],
            requests[1],
            &samples[1].frame().global[59..64]
        );
        assert!(line.len() <= 2048);
        writer
            .write_all(line.as_bytes())
            .expect("bounded trajectory transcript");
    }
}

pub(super) fn finish_transcript(game: &mut HistoryGame) {
    assert!(game.finished);
    assert!(game.counts.iter().sum::<u32>() <= 36300);
    if let Some(mut writer) = game.transcript.take() {
        writer.flush().expect("complete transcript");
    }
}

pub(super) fn assert_matched_pools(pools: &[ImitationPool; 2]) {
    assert!(!pools[0].is_empty());
    assert_eq!(pools[0].len(), pools[1].len());
    assert!(pools[0].len() <= 8 * 1488);
    for index in 0..pools[0].len() {
        let baseline = pools[0].get(index).expect("baseline row");
        let enriched = pools[1].get(index).expect("enriched row");
        assert_eq!(baseline.identity(), enriched.identity());
        assert_eq!(baseline.target(), enriched.target());
        assert_eq!(baseline.teacher_action(), enriched.teacher_action());
        assert_eq!(baseline.split(), enriched.split());
        assert_eq!(&baseline.frame().global[59..64], &[0.0; 5]);
        let mut stripped = enriched.frame().clone();
        stripped.global[59..64].fill(0.0);
        assert_eq!(
            baseline.frame(),
            &stripped,
            "only the five reserved channels may differ"
        );
    }
}

fn write_retained_rows(pools: &[ImitationPool; 2], output: &Path) {
    assert_matched_pools(pools);
    let mut writer = new_writer(&output.join("rows.tsv"));
    writeln!(writer, "index\tseed\ttick\tside\tsplit\tkind\taction\tprevious_sent_features\tbaseline_frame_sha256\tenriched_frame_sha256").expect("row header");
    for index in 0..pools[0].len() {
        let baseline = pools[0].get(index).expect("row");
        let enriched = pools[1].get(index).expect("paired row");
        let identity = baseline.identity();
        assert_eq!(identity.trajectory(), 0);
        assert_eq!(identity.map(), MapId(0));
        writeln!(
            writer,
            "{index}\t{}\t{}\t{:?}\t{:?}\t{}\t{:?}\t{:?}\t{}\t{}",
            identity.seed(),
            identity.tick(),
            identity.side(),
            baseline.split(),
            baseline.teacher_action().kind().index(),
            baseline.teacher_action(),
            &enriched.frame().global[59..64],
            frame_sha256(baseline.frame()),
            frame_sha256(enriched.frame())
        )
        .expect("retained identity and tensor audit");
    }
    writer.flush().expect("complete retained rows");
}

fn assert_initial_outputs(models: &[PolicyModel; 2], pools: &[ImitationPool; 2]) {
    assert_matched_pools(pools);
    assert_eq!(
        models[0].export_parameters().expect("baseline"),
        models[1].export_parameters().expect("enriched")
    );
    let indices: Vec<_> = (0..pools[0].len()).collect();
    for batch in indices.chunks(64) {
        let frames: [Vec<_>; 2] = std::array::from_fn(|variant| {
            batch
                .iter()
                .map(|&index| {
                    pools[variant]
                        .get(index)
                        .expect("paired sample")
                        .frame()
                        .clone()
                })
                .collect()
        });
        assert_eq!(
            models[0]
                .evaluate_batch(&frames[0])
                .expect("baseline outputs"),
            models[1]
                .evaluate_batch(&frames[1])
                .expect("enriched outputs")
        );
    }
    let samples: [Vec<_>; 2] = std::array::from_fn(|variant| {
        (0..pools[variant].len())
            .map(|index| pools[variant].get(index).expect("sample"))
            .collect()
    });
    assert_eq!(
        history_predictions(&models[0], &samples[0]),
        history_predictions(&models[1], &samples[1])
    );
    eprintln!(
        "history_equality rows={} all_initial_outputs=true conditional_predictions=true sample_identities=true targets=true nonreserved_frames=true",
        pools[0].len()
    );
}

fn fit_matched_models(
    models: &[PolicyModel; 2],
    pools: &[ImitationPool; 2],
    coverage: &[TeacherCoverage; 2],
    output: &Path,
    export_root: &Path,
    started: Instant,
) {
    let mut trainers = matched_trainers(models, pools);
    let mut predictions = new_writer(&output.join("predictions.tsv"));
    writeln!(
        predictions,
        "variant\tepoch\tindex\tkind_prediction\tkind_correct\tfull_correct"
    )
    .expect("prediction header");
    for epoch in 0..=8 {
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "combined history fit budget"
        );
        if epoch > 0 {
            let reports: [_; 2] = std::array::from_fn(|index| {
                trainers[index]
                    .train_epoch(&models[index], &pools[index])
                    .expect("matched epoch")
            });
            assert_eq!(
                reports[0].order, reports[1].order,
                "identical shuffle order"
            );
            for (index, report) in reports.iter().enumerate() {
                eprintln!(
                    "history_train variant={} epoch={epoch} samples={} updates={} loss={} order_sha256={}",
                    VARIANTS[index],
                    report.order.len(),
                    trainers[index].counters().global_update,
                    report.average_loss,
                    order_sha256(&report.order)
                );
            }
        }
        for index in 0..2 {
            report_metrics(
                &models[index],
                &pools[index],
                &coverage[index],
                VARIANTS[index],
                epoch,
                &mut predictions,
            );
        }
        if epoch == 4 || epoch == 8 {
            export_baseline(&models[0], &pools[0], epoch, output, export_root);
        }
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "combined history fit budget"
        );
    }
    predictions.flush().expect("all matched predictions");
    assert_eq!(trainers[0].counters().epoch, 8);
    assert_eq!(trainers[0].counters(), trainers[1].counters());
}

fn matched_trainers(
    models: &[PolicyModel; 2],
    pools: &[ImitationPool; 2],
) -> [BehavioralTrainer; 2] {
    let trainers: [_; 2] = std::array::from_fn(|index| {
        BehavioralTrainer::new(
            64,
            10089151,
            AdamConfig {
                learning_rate: 3e-5,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                gradient_clip: 0.5,
            },
            &models[index],
            &pools[index],
        )
        .expect("fresh matched Adam")
    });
    assert_eq!(trainers[0].counters().global_update, 0);
    assert_eq!(trainers[1].counters().global_update, 0);
    trainers
}

#[derive(Default)]
struct HistoryMetrics {
    total: usize,
    kind: usize,
    full: usize,
    noncontinue: usize,
    noncontinue_kind: usize,
    noncontinue_full: usize,
    labels: [usize; ActionKind::COUNT],
    predicted: [usize; ActionKind::COUNT],
    kind_by_label: [usize; ActionKind::COUNT],
    full_by_label: [usize; ActionKind::COUNT],
}

impl HistoryMetrics {
    fn record(
        &mut self,
        sample: &ImitationSample,
        prediction: BehavioralPrediction,
    ) -> (bool, bool) {
        let label = sample.teacher_action().kind().index();
        assert!(prediction.kind < ActionKind::COUNT);
        assert!(self.total < 8 * 1488);
        let kind = prediction.kind == label;
        let full = prediction == target_prediction(sample.target());
        self.total += 1;
        self.kind += usize::from(kind);
        self.full += usize::from(full);
        self.labels[label] += 1;
        self.predicted[prediction.kind] += 1;
        self.kind_by_label[label] += usize::from(kind);
        self.full_by_label[label] += usize::from(full);
        if label != ActionKind::Continue.index() {
            self.noncontinue += 1;
            self.noncontinue_kind += usize::from(kind);
            self.noncontinue_full += usize::from(full);
        }
        (kind, full)
    }
}

fn target_prediction(target: &BehavioralTarget) -> BehavioralPrediction {
    fn selected<const WIDTH: usize>(head: &HeadTarget<WIDTH>) -> Option<usize> {
        assert!(head.selected < WIDTH);
        assert!(!head.active || head.is_selected_legal());
        head.active.then_some(head.selected)
    }
    assert!(target.kind.active);
    assert!(target.kind.is_selected_legal());
    BehavioralPrediction {
        kind: target.kind.selected,
        controlled: selected(&target.controlled),
        ability: selected(&target.ability),
        item: selected(&target.item),
        swap: selected(&target.swap),
        learn: selected(&target.learn),
        shop: selected(&target.shop),
        loot: selected(&target.loot),
        target_mode: selected(&target.target_mode),
        put_mode: selected(&target.put_mode),
        entity_pointer: selected(&target.entity_pointer),
        point_pointer: selected(&target.point_pointer),
    }
}

fn report_metrics(
    model: &PolicyModel,
    pool: &ImitationPool,
    coverage: &TeacherCoverage,
    variant: &str,
    epoch: u64,
    writer: &mut impl Write,
) {
    assert!(VARIANTS.contains(&variant));
    assert!(epoch <= 8);
    let samples: Vec<_> = (0..pool.len())
        .map(|index| pool.get(index).expect("sample"))
        .collect();
    let predictions = history_predictions(model, &samples);
    assert_eq!(samples.len(), predictions.len());
    let mut metrics: [[HistoryMetrics; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| HistoryMetrics::default()));
    for (index, (sample, prediction)) in samples.iter().zip(predictions).enumerate() {
        let split = match sample.split() {
            ImitationSplit::Train => 0,
            ImitationSplit::Validation => 1,
            ImitationSplit::HeldOut => panic!("no promotion samples"),
        };
        let side = usize::from(sample.side() == ImitationSide::Dire);
        let (kind, full) = metrics[split][side].record(sample, prediction);
        writeln!(
            writer,
            "{variant}\t{epoch}\t{index}\t{}\t{}\t{}",
            prediction.kind,
            u8::from(kind),
            u8::from(full)
        )
        .expect("row prediction");
    }
    let audited = OfflineEvaluation::evaluate_validation(model, pool, coverage.clone())
        .expect("production validation crosscheck");
    for (side, aggregate) in [&audited.metrics().radiant, &audited.metrics().dire]
        .into_iter()
        .enumerate()
    {
        assert_eq!(metrics[1][side].kind, aggregate.kind.matching);
        assert_eq!(metrics[1][side].full, aggregate.full.matching);
        assert_eq!(metrics[1][side].labels, aggregate.action_distribution);
    }
    for (split, sides) in ["Train", "Validation"].into_iter().zip(metrics) {
        for (side, values) in ["Radiant", "Dire"].into_iter().zip(sides) {
            eprintln!(
                "history_metrics variant={variant} epoch={epoch} split={split} side={side} total={} kind={} full={} noncontinue={} noncontinue_kind={} noncontinue_full={} labels={:?} predicted={:?} kind_by_label={:?} full_by_label={:?}",
                values.total,
                values.kind,
                values.full,
                values.noncontinue,
                values.noncontinue_kind,
                values.noncontinue_full,
                values.labels,
                values.predicted,
                values.kind_by_label,
                values.full_by_label
            );
        }
    }
}

fn history_predictions(
    model: &PolicyModel,
    samples: &[&ImitationSample],
) -> Vec<BehavioralPrediction> {
    assert!(!samples.is_empty());
    assert!(samples.len() <= 8 * 1488);
    let mut predictions = Vec::with_capacity(samples.len());
    for batch in samples.chunks(crate::MODEL_MAX_BATCH) {
        predictions.extend(
            model
                .behavioral_predictions(batch)
                .expect("bounded teacher-forced predictions"),
        );
    }
    assert_eq!(predictions.len(), samples.len());
    predictions
}

fn validate_baseline_pool(pool: &ImitationPool) -> Result<(), &'static str> {
    assert!(!pool.is_empty());
    assert!(pool.len() <= 8 * 1488);
    for index in 0..pool.len() {
        if pool.get(index).expect("baseline sample").frame().global[59..64] != [0.0; 5] {
            return Err("enriched history cannot be exported under F12");
        }
    }
    Ok(())
}

fn export_baseline(
    model: &PolicyModel,
    pool: &ImitationPool,
    epoch: u64,
    output: &Path,
    export_root: &Path,
) {
    assert!([4, 8].contains(&epoch));
    validate_baseline_pool(pool).expect("baseline-only F12 export");
    let parameters = model.export_parameters().expect("trained baseline");
    assert_eq!(
        parameters,
        zero_probe_input_weights(model),
        "baseline reserved input rows must remain zero"
    );
    let directory = export_root.join(format!("baseline-epoch{epoch:03}"));
    std::fs::create_dir(&directory).expect("NEW baseline export directory");
    TrainingArtifact::save_runtime_weights(model, &directory)
        .expect("legal current runtime export");
    let restored = PolicyModel::fresh(10089202).expect("independent runtime reload");
    TrainingArtifact::load_runtime_weights(&restored, &directory)
        .expect("F12/M14/PPO27 runtime reload");
    assert_eq!(
        parameter_sha256(&restored.export_parameters().expect("reloaded bits")),
        parameter_sha256(&parameters)
    );
    let weights = file_sha256(&directory.join("drysua.weights.safetensors"));
    let provenance = format!(
        "experiment=matched_BC_history_training_contract\nvariant=baseline_only\nparent=INITIAL_M14_initialization\nparent_sha256={INITIAL_SHA256}\nweights_sha256={weights}\nparameter_sha256={}\nmanifest_sha256={}\nrows_sha256={}\nepoch={epoch}\noptimizer=fresh_BC_Adam_batch64_shuffle10089151_lr3e-5_beta1.9_beta2.999_epsilon1e-8_clip.5\nincoming_rows59..63=zero_for_both_initial_models_and_baseline_export\ntrain_seeds=10089000,10089001_bothsides\nvalidation_seeds=10089002,10089003_bothsides\nobserver_from_start=true\nopponent=M14_initialization_INITIAL_current_Candidate\nopening=32\nwinning_tail=1200\nrare_per_noncontinue_kind=16\nqualified=false\nautomatic_promotion=false\nfull_game_evaluation=not_run\nenriched_exported=false\ntraining_resume_checkpoint=not_exported\n",
        parameter_sha256(&parameters),
        file_sha256(&output.join("manifest.json")),
        file_sha256(&output.join("rows.tsv"))
    );
    std::fs::write(directory.join("PROVENANCE.txt"), provenance).expect("baseline provenance");
    eprintln!(
        "history_export epoch={epoch} directory={} weights_sha256={weights} qualified=false",
        directory.display()
    );
}

fn new_writer(path: &Path) -> BufWriter<std::fs::File> {
    assert!(path.parent().expect("parent").is_dir());
    assert!(!path.exists(), "never overwrite probe evidence");
    BufWriter::new(std::fs::File::create_new(path).expect("new probe evidence"))
}

fn file_sha256(path: &Path) -> String {
    assert!(path.is_file());
    assert!(path.metadata().expect("file metadata").len() < 64 * 1024 * 1024);
    digest_hex(Sha256::digest(std::fs::read(path).expect("hash input")).into())
}

fn parameter_sha256(parameters: &[f32]) -> String {
    assert_eq!(parameters.len(), crate::MODEL_PARAMETER_COUNT);
    assert!(parameters.iter().all(|value| value.is_finite()));
    let mut digest = Sha256::new();
    for value in parameters {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest_hex(digest.finalize().into())
}

fn frame_sha256(frame: &FeatureFrame) -> String {
    assert!(frame.is_finite());
    assert_eq!(frame.global.len(), crate::GLOBAL_FEATURES);
    let mut digest = Sha256::new();
    for values in [
        frame.global.as_slice(),
        frame.history.as_flattened(),
        frame.policy_history.as_flattened(),
        frame.units.as_flattened(),
        frame.own_units.as_flattened(),
        frame.remembered_units.as_flattened(),
        frame.points.as_flattened(),
        frame.abilities.as_flattened(),
        frame.items.as_flattened(),
        frame.projectiles.as_flattened(),
        frame.loot.as_flattened(),
        frame.map.as_slice(),
    ] {
        for value in values {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest_hex(digest.finalize().into())
}

fn order_sha256(order: &[usize]) -> String {
    assert!(!order.is_empty());
    assert!(order.len() <= 4 * 1488);
    let mut digest = Sha256::new();
    for index in order {
        digest.update((*index as u64).to_le_bytes());
    }
    digest_hex(digest.finalize().into())
}

fn digest_hex(digest: [u8; 32]) -> String {
    let text: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(text.len(), 64);
    assert!(text.bytes().all(|byte| byte.is_ascii_hexdigit()));
    text
}
