use super::*;
use data::Row;
use sha2::{Digest, Sha256};

const INITIAL_SHA: &str = "1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4";

#[test]
#[ignore = "One guarded fresh-initial joint rehearsal configuration; no runtime overrides."]
fn joint_rehearsal_fit() {
    let started = Instant::now();
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/map2-learning-20260911/initial");
    assert_eq!(
        file_hash(&source.join("drysua.weights.safetensors")),
        INITIAL_SHA
    );
    let output = root().join("joint-weights");
    assert!(!output.exists());
    let skills = data::collect_skills(true);
    let anchors = data::collect_anchors(true);
    assert!(skills.len() + anchors.len() <= 4096);
    assert!(!skills.is_empty());
    assert!(!anchors.is_empty());
    let mut report = format!(
        "origin_skills=validated_skill_trajectory\norigin_anchors=unchanged_teacher_native_opening\ninitial_sha256={INITIAL_SHA}\ntrain_skill_rows={}\ntrain_anchor_rows={}\nskill_data_sha256={}\nanchor_data_sha256={}\nconstruction_seconds={:.3}\n",
        skills.len(),
        anchors.len(),
        row_hash(&skills),
        row_hash(&anchors),
        started.elapsed().as_secs_f64()
    );
    describe_rows(&skills, &mut report);
    describe_rows(&anchors, &mut report);
    let model = load_model(&source);
    let steps = fit_joint(&model, &skills, &anchors, started, &mut report);
    writeln!(report, "optimizer_steps={steps} total_seconds={:.3}\nconfiguration=batch64_half_anchor_half_skill_lr1e-4_beta.9_.999_eps1e-8_clip.5_seed10092099_CPU\ndiagnostic_only=true\nqualified=false", started.elapsed().as_secs_f64()).unwrap();
    write_new(&root().join("FIT.txt"), &report);
    assert_eq!(
        steps, 512,
        "guarded work budget incomplete; no export or retry"
    );
    std::fs::create_dir(&output).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &output).unwrap();
    let weights_hash = file_hash(&output.join("drysua.weights.safetensors"));
    let manifest = format!(
        "diagnostic_only=true\nqualified=false\nopening_gate=pending\nweights_sha256={weights_hash}\nmodel_schema={}\nfeature_schema={}\n{report}",
        crate::MODEL_SCHEMA_VERSION,
        crate::FEATURE_SCHEMA_VERSION
    );
    write_new(&output.join("MANIFEST.txt"), &manifest);
    eprintln!("{manifest}");
    assert_eq!(
        file_hash(&source.join("drysua.weights.safetensors")),
        INITIAL_SHA
    );
}

fn fit_joint(
    model: &PolicyModel,
    skills: &[Row],
    anchors: &[Row],
    started: Instant,
    report: &mut String,
) -> u64 {
    let mut optimizer = model
        .claim_optimizer(AdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .unwrap();
    let mut random = PpoRng::new(10092099);
    let fit_started = Instant::now();
    for step in 1..=512 {
        if started.elapsed() > Duration::from_secs(250) {
            break;
        }
        let mut batch = Vec::with_capacity(64);
        for _ in 0..32 {
            batch.push(&anchors[random.below(anchors.len() as u64).unwrap() as usize].sample);
        }
        for _ in 0..32 {
            batch.push(&skills[random.below(skills.len() as u64).unwrap() as usize].sample);
        }
        let update = model
            .train_checked_behavioral_batch(&batch, &mut optimizer)
            .unwrap();
        assert_eq!(update.optimizer_step, step);
        if step == 1 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} loss={} active_heads={:?} fit_seconds={:.3}",
                update.average_loss,
                update.active_head_counts,
                fit_started.elapsed().as_secs_f64()
            )
            .unwrap();
            eprintln!(
                "joint step={step} loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_started.elapsed().as_secs_f64()
            );
        }
    }
    optimizer.step()
}

#[derive(Debug, Default)]
struct Agreement {
    rows: [u32; 2],
    full: [u32; 2],
    kinds: [u32; 2],
    heads: [u32; 2],
    conditional: [u32; 2],
    continues: [u32; 2],
    continue_rows: [u32; 2],
    utility: [u32; 2],
    utility_rows: [u32; 2],
}

fn agreement(model: &PolicyModel, rows: &[Row]) -> Agreement {
    let mut report = Agreement::default();
    for chunk in rows.chunks(64) {
        let predictions = model
            .checked_behavioral_predictions(
                &chunk.iter().map(|row| &row.sample).collect::<Vec<_>>(),
            )
            .unwrap();
        for (row, prediction) in chunk.iter().zip(predictions) {
            let action = model.choose(row.sample.frame(), &row.space).unwrap().action;
            let side = row.side;
            report.rows[side] += 1;
            report.full[side] += u32::from(action == row.action);
            report.kinds[side] += u32::from(action.kind() == row.action.kind());
            if row.action == StructuredAction::Continue {
                report.continue_rows[side] += 1;
                report.continues[side] += u32::from(action == row.action);
            } else {
                report.utility_rows[side] += 1;
                report.utility[side] += u32::from(action == row.action);
            }
            let target = row.sample.target();
            macro_rules! head { ($($name:ident),*) => { $(
                if target.$name.active { report.heads[side] += 1; report.conditional[side] += u32::from(prediction.$name == Some(target.$name.selected)); }
            )* }; }
            head!(
                controlled,
                ability,
                item,
                swap,
                learn,
                shop,
                loot,
                target_mode,
                put_mode,
                entity_pointer,
                point_pointer
            );
        }
    }
    report
}

#[test]
#[ignore = "Guarded fixed-model opening and local regression gates; no model update or tuning."]
fn joint_rehearsal_validation_gates() {
    let started = Instant::now();
    let model = load_model(&root().join("joint-weights"));
    let old = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/map2-learning-20260911");
    let initial = load_model(&old.join("initial"));
    let failed = load_model(&old.join("skill-bootstrap-001-transfer/diagnostic-weights"));
    let rows = data::collect_skills(false);
    let anchors = data::collect_anchors(false);
    assert!(rows.len() + anchors.len() <= 2048);
    let regression = data::old_regression_rows();
    assert_eq!(regression.len(), 170);
    let mut report = format!(
        "new_held_skill_rows={} new_held_anchor_rows={} held_skill_sha={} held_anchor_sha={}\n",
        rows.len(),
        anchors.len(),
        row_hash(&rows),
        row_hash(&anchors)
    );
    let old_baseline = agreement(&failed, &regression);
    let old_joint = agreement(&model, &regression);
    writeln!(report, "old_diagnostic_baseline={old_baseline:?}\nold_diagnostic_joint={old_joint:?}\nnew_held_initial={:?}\nnew_held_joint={:?}\nnew_anchor_initial={:?}\nnew_anchor_joint={:?}", agreement(&initial, &rows), agreement(&model, &rows), agreement(&initial, &anchors), agreement(&model, &anchors)).unwrap();
    let mut opening_pass = true;
    for (seed, side) in data::opening_specs(false) {
        let mut observed = run_opening(&model, Some(&initial), seed, side);
        opening_pass &= progress_gate(&observed.stats);
        let trace = std::mem::take(&mut observed.stats.trace);
        writeln!(
            report,
            "new_opening seed={seed} side={side} passed={} {:?}",
            progress_gate(&observed.stats),
            observed.stats
        )
        .unwrap();
        for row in trace {
            writeln!(report, "{row}").unwrap();
        }
    }
    let old_outcomes = data::old_skill_outcomes(&model, &mut report, "old_diagnostic_joint");
    let new_initial = data::skill_outcomes(&initial, &mut report, "new_initial");
    let new_joint = data::skill_outcomes(&model, &mut report, "new_joint");
    let retention = (0..2).all(|side| {
        preserved(
            old_joint.continues[side],
            old_baseline.continues[side],
            old_joint.continue_rows[side],
        ) && preserved(
            old_joint.utility[side],
            old_baseline.utility[side],
            old_joint.utility_rows[side],
        ) && old_outcomes[side] >= 11
    });
    let permitted = opening_pass && retention;
    writeln!(report, "old_outcomes={old_outcomes:?}/12 new_initial={new_initial:?}/24 new_joint={new_joint:?}/24\nopening_pass={opening_pass}\nretention_pass={retention}\ndev_permitted={permitted}\nqualified=false\nwall_seconds={:.3}", started.elapsed().as_secs_f64()).unwrap();
    write_new(&root().join("GATES.txt"), &report);
    eprintln!("{report}");
}

fn preserved(actual: u32, before: u32, count: u32) -> bool {
    assert!(count > 0);
    assert!(actual <= count);
    f64::from(actual) / f64::from(count) + 0.05 >= f64::from(before) / f64::from(count)
}

#[test]
#[ignore = "Read-only fixed-model residual witnesses on the old diagnostic rows; no fit or full game."]
fn joint_remaining_witnesses() {
    let model = load_model(&root().join("joint-weights"));
    let rows = data::old_regression_rows();
    let mut report =
        String::from("scope=old_diagnostic_scripted_frames_not_new_validation\nmodel_updates=0\n");
    let mut count = 0;
    for row in rows {
        let action = model.choose(row.sample.frame(), &row.space).unwrap().action;
        if action == row.action || count >= 12 {
            continue;
        }
        let facts: Vec<_> = row
            .space
            .entity_candidates()
            .iter()
            .filter(|unit| unit.relation == crate::EntityRelation::Enemy)
            .map(|unit| {
                (
                    unit.kind,
                    unit.position,
                    unit.unit().hp,
                    unit.unit().effects.clone(),
                )
            })
            .collect();
        writeln!(report, "seed={} side={} tick={} validated_label={:?} selected={action:?} decoded={:?} kind_margin={:?} visible_enemy_facts={facts:?}", row.seed, row.side, row.space.tick(), row.action, row.space.decode(action).unwrap(), kind_margin(&model, row.sample.frame(), &row.space, action.kind())).unwrap();
        count += 1;
    }
    assert!(count > 0);
    assert!(count <= 12);
    write_new(&root().join("REMAINING.txt"), &report);
    eprintln!("{report}");
}

fn describe_rows(rows: &[Row], report: &mut String) {
    let mut kinds = [0usize; ActionKind::COUNT];
    for row in rows {
        kinds[row.action.kind().index()] += 1;
    }
    writeln!(
        report,
        "origin={} rows={} kinds={kinds:?}",
        rows[0].origin,
        rows.len()
    )
    .unwrap();
}

fn row_hash(rows: &[Row]) -> String {
    assert!(rows.len() <= 4096);
    let mut hash = Sha256::new();
    for row in rows {
        hash.update(format!(
            "{}:{}:{}:{}:{:?}:{:?}",
            row.origin,
            row.seed,
            row.side,
            row.space.tick(),
            row.action,
            row.sample.target()
        ));
        let frame = row.sample.frame();
        for value in frame
            .global
            .iter()
            .chain(frame.history.iter().flatten())
            .chain(frame.policy_history.iter().flatten())
            .chain(frame.units.iter().flatten())
            .chain(frame.own_units.iter().flatten())
            .chain(frame.remembered_units.iter().flatten())
            .chain(frame.points.iter().flatten())
            .chain(frame.abilities.iter().flatten())
            .chain(frame.items.iter().flatten())
            .chain(frame.projectiles.iter().flatten())
            .chain(frame.loot.iter().flatten())
            .chain(frame.map.iter())
        {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_new(path: &Path, text: &str) {
    use std::io::Write;
    assert!(text.len() < 16 * 1024 * 1024);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(text.as_bytes()).unwrap();
    file.sync_all().unwrap();
}

fn file_hash(path: &Path) -> String {
    assert!(std::fs::metadata(path).unwrap().len() < 16 * 1024 * 1024);
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
