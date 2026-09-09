use super::*;
use crate::{ActivePolicyOrder, ActivePolicyTarget, unit_feature};

#[path = "active_target_probe_tests.rs"]
mod tests;

const VARIANTS: [&str; 2] = ["baseline", "target-summary"];

#[test]
#[ignore = "Predeclared prior-active-target factual summary; no weight exports or gameplay"]
fn probe_active_target_summary() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root.join("artifacts/temp/neural-reset-20260908/active-target-probe-001");
    let original = root.join("artifacts/temp/neural-reset-20260908/training-contract-bc");
    assert!(
        output.join("experiment-manifest.json").is_file(),
        "freeze before execution"
    );
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 12);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 14);
    assert_eq!(IMITATION_RULES_AUDIT_VERSION, 13);
    assert_eq!(
        file_sha256(&root.join(INITIAL).join("drysua.weights.safetensors")),
        INITIAL_SHA256
    );
    let initialization = Instant::now();
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let parent = PolicyModel::fresh_on(10089200, device).expect("unchanged M14 INITIAL");
    TrainingArtifact::load_runtime_weights(&parent, &root.join(INITIAL))
        .expect("audited runtime initializer");
    let parameters = zero_probe_input_weights(&parent);
    assert_eq!(
        parameter_sha256(&parameters),
        "c23b5357787f67ab2d64af4282f2147841ba61635673ecdf809dfa75957c99be"
    );
    let initialization_time = initialization.elapsed();
    let (mut rows, spaces) = replay::reconstruct_conditioning_rows(&original, &output);
    let started = Instant::now()
        .checked_sub(initialization_time)
        .expect("include initialization budget");
    annotate_rows(&mut rows, &spaces, &output);
    let references: Vec<_> = rows.iter().map(|row| &row.raw).collect();
    let schedule = conditioning_schedule(&references, false);
    audit_natural_schedule(&rows, &schedule, &output);
    let models: [_; 2] = std::array::from_fn(|_| {
        let model = PolicyModel::fresh_on(10089200, device).expect("independent learner");
        model
            .import_parameters(&parameters)
            .expect("same zero-row initialization");
        model
    });
    conditioning_equivalence(
        &models[0],
        &models[1],
        &rows,
        &spaces,
        true,
        "active-target-initial",
    );
    fit_variants(&models, &rows, &schedule, &output, started);
    assert!(
        started.elapsed() < Duration::from_secs(180),
        "total active-target fit budget"
    );
    eprintln!(
        "active_target_complete fit_seconds={:.3} variants=2 updates_per_variant=680 samples_per_variant=43520 weights_exported=false gameplay_evaluated=false",
        started.elapsed().as_secs_f64()
    );
}

fn fit_variants(
    models: &[PolicyModel; 2],
    rows: &[ConditioningRow],
    schedule: &[usize],
    output: &Path,
    started: Instant,
) {
    assert_eq!(rows.len(), 10783);
    assert_eq!(schedule.len(), CONDITIONING_PRESENTATIONS);
    let mut predictions = new_writer(&output.join("predictions.tsv"));
    writeln!(
        predictions,
        "variant\tupdate\tindex\tpredicted\tkind_correct\tfull_correct"
    )
    .expect("prediction header");
    for (index, name) in VARIANTS.into_iter().enumerate() {
        conditioning_metrics(&models[index], rows, index == 1, name, 0, &mut predictions);
        conditioning_fit(&models[index], rows, index == 1, schedule, name, started);
        conditioning_metrics(
            &models[index],
            rows,
            index == 1,
            name,
            680,
            &mut predictions,
        );
        eprintln!(
            "active_target_parameters variant={name} sha256={} weights_exported=false",
            parameter_sha256(
                &models[index]
                    .export_parameters()
                    .expect("diagnostic fingerprint only")
            )
        );
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "total active-target fit budget"
        );
    }
    predictions.flush().expect("all predictions");
}

fn annotate_rows(rows: &mut [ConditioningRow], spaces: &[ActionSpace], output: &Path) {
    assert_eq!(rows.len(), spaces.len());
    assert_eq!(rows.len(), 10783);
    let mut writer = new_writer(&output.join("target-summary.tsv"));
    writeln!(writer, "index\tprior_kind_token\tprior_unit_target\tprior_visible\ttarget_kind_token\tpresent\thp_ratio\tattacks_estimate\tin_own_range\tinvulnerable\traw_frame_sha256\taugmented_frame_sha256").expect("factual audit header");
    for (row, space) in rows.iter_mut().zip(spaces) {
        row.alternate = annotate(&row.raw, space, row.target_summary);
        let frame = row.raw.frame();
        let values = row.target_summary;
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row.index,
            frame.global[crate::global_feature::ACTIVE_ORDER_KIND],
            frame.global[crate::global_feature::ACTIVE_TARGET_UNIT],
            frame.global[crate::global_feature::ACTIVE_TARGET_VISIBLE],
            frame.global[crate::global_feature::ACTIVE_TARGET_KIND_TOKEN],
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            frame_sha256(frame),
            frame_sha256(row.alternate.frame())
        )
        .expect("factual row audit");
    }
    writer.flush().expect("target summary audit");
}

fn annotate(raw: &ImitationSample, space: &ActionSpace, values: [f32; 5]) -> ImitationSample {
    assert_eq!(raw.frame().global[59..64], [0.0; 5]);
    assert!(values.iter().all(|value| (-1.0..=1.0).contains(value)));
    let mut frame = raw.frame().clone();
    frame.global[59..64].copy_from_slice(&values);
    let augmented = ImitationSample::teacher(frame, space, raw.teacher_action(), raw.identity())
        .expect("same identity and label, factual prior summary only");
    assert_eq!(raw.identity(), augmented.identity());
    assert_eq!(raw.target(), augmented.target());
    let mut stripped = augmented.frame().clone();
    stripped.global[59..64].fill(0.0);
    assert_eq!(raw.frame(), &stripped, "no other feature changes");
    augmented
}

fn audit_natural_schedule(rows: &[ConditioningRow], schedule: &[usize], output: &Path) {
    assert_eq!(schedule.len(), 43520);
    assert_eq!(rows.len(), 10783);
    let mut writer = new_writer(&output.join("schedule.tsv"));
    writeln!(writer, "presentation\tindex").expect("schedule header");
    let mut digest = Sha256::new();
    let mut counts = [[0usize; ActionKind::COUNT]; 2];
    for (presentation, &index) in schedule.iter().enumerate() {
        let row = &rows[index].raw;
        assert_eq!(row.split(), ImitationSplit::Train);
        counts[usize::from(row.side() == ImitationSide::Dire)]
            [row.teacher_action().kind().index()] += 1;
        digest.update((index as u64).to_le_bytes());
        writeln!(writer, "{presentation}\t{index}").expect("fixed natural presentation");
    }
    let hash = digest_hex(digest.finalize().into());
    assert_eq!(
        hash,
        "3b2fcffdc00aa9abe43e5b5de2c0e02c23de35f1e30fd38c5ef954a35c1a3625"
    );
    writer.flush().expect("schedule audit");
    eprintln!(
        "active_target_schedule samples={} radiant={:?} dire={:?} index_sha256={hash} balancing=false",
        schedule.len(),
        counts[0],
        counts[1]
    );
}

#[derive(Clone, Copy)]
pub(super) struct TargetContext {
    prior: Option<ActivePolicyOrder>,
    own_hero_current: bool,
}

impl TargetContext {
    pub(super) fn capture(seat: &ArenaSeatPolicy) -> Self {
        assert!(seat.order_bookkeeping.is_neural());
        assert!(!seat.order_bookkeeping.is_candidate());
        Self {
            prior: seat.local.active_order(),
            own_hero_current: seat.tracker.own_hero().is_some(),
        }
    }
}

pub(super) fn summary(
    context: TargetContext,
    space: &ActionSpace,
    frame: &FeatureFrame,
) -> [f32; 5] {
    assert!(frame.matches_action_space(space));
    assert!(frame.is_finite());
    let Some(prior) = context.prior else {
        return [0.0; 5];
    };
    assert!(prior.started_tick <= space.tick());
    let ActivePolicyTarget::Unit(target) = prior.target else {
        return [0.0; 5];
    };
    let Some(index) = space.entity_index(target) else {
        return [0.0; 5];
    };
    let token = &frame.units[index.0];
    if token[unit_feature::TOKEN_PRESENT] != 1.0
        || token[unit_feature::OBSERVATION_PRESENT] != 1.0
        || token[unit_feature::VISIBLE] != 1.0
        || token[unit_feature::REMEMBERED] != 0.0
        || token[unit_feature::AGE] != 0.0
    {
        return [0.0; 5];
    }
    let result = [
        1.0,
        if token[unit_feature::HP_PRESENT] == 1.0 {
            token[unit_feature::HP_RATIO]
        } else {
            -1.0
        },
        if context.own_hero_current && token[unit_feature::ATTACKS_TO_KILL_PRESENT] == 1.0 {
            token[unit_feature::ATTACKS_TO_KILL]
        } else {
            -1.0
        },
        if context.own_hero_current && token[unit_feature::ORIGIN_PRESENT] == 1.0 {
            token[unit_feature::OWN_IN_ATTACK_RANGE]
        } else {
            -1.0
        },
        token[unit_feature::INVULNERABLE],
    ];
    assert!(result.iter().all(|value| (-1.0..=1.0).contains(value)));
    assert!([0.0, 1.0].contains(&result[4]));
    result
}
