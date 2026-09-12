#[path = "map2_skill_bootstrap_transfer.rs"]
mod transfer;
use super::*;
use crate::{HeadTarget, PolicyTensorOutput};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

const SHA: &str = "1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4";

pub(super) fn head_loss<const WIDTH: usize>(
    logits: &Tensor,
    heads: &[HeadTarget<WIDTH>],
) -> candle_core::Result<Tensor> {
    let mut masks = Vec::new();
    let mut labels = Vec::new();
    let mut active = Vec::new();
    for head in heads {
        if head.active && !head.is_selected_legal() {
            candle_core::bail!("illegal selected fixture head");
        }
        if !head.active && head.mask.contains(&true) {
            candle_core::bail!("inactive fixture head has legal classes");
        }
        masks.extend((0..WIDTH).map(|index| {
            u8::from(if head.active {
                head.mask[index]
            } else {
                index == 0
            })
        }));
        labels.push(if head.active { head.selected as u32 } else { 0 });
        active.push(f32::from(head.active));
    }
    let mask = Tensor::from_vec(masks, (heads.len(), WIDTH), &Device::Cpu)?;
    let negative = Tensor::full(f32::NEG_INFINITY, logits.shape(), &Device::Cpu)?;
    let legal = mask.where_cond(logits, &negative)?;
    let legal = legal.broadcast_sub(&legal.max_keepdim(1)?.detach())?;
    let selected = legal
        .gather(
            &Tensor::from_vec(labels, (heads.len(), 1), &Device::Cpu)?,
            1,
        )?
        .squeeze(1)?;
    (legal.log_sum_exp(1)? - selected)?
        .mul(&Tensor::from_vec(active, heads.len(), &Device::Cpu)?)?
        .mean_all()
}

fn loss(output: &PolicyTensorOutput<'_>, rows: &[&Row]) -> Tensor {
    macro_rules! head {
        ($field:ident) => {
            head_loss(
                output.$field(),
                &rows.iter().map(|row| row.target.$field).collect::<Vec<_>>(),
            )
            .unwrap()
        };
    }
    let mut result = head!(kind);
    macro_rules! add { ($($field:ident),*) => { $(result = (result + head!($field)).unwrap();)* }; }
    add!(
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
    result
}

struct Optimizer {
    first: Vec<f32>,
    second: Vec<f32>,
    step: u64,
}
fn update(model: &PolicyModel, rows: &[&Row], optimizer: &mut Optimizer) -> f32 {
    assert!(!rows.is_empty());
    assert!(rows.len() <= 32);
    for row in rows {
        row.target.validate().unwrap();
        assert!(row.frame.matches_action_space(&row.space));
    }
    let frames: Vec<_> = rows.iter().map(|row| row.frame.clone()).collect();
    let prefixes: Vec<_> = rows.iter().map(|row| row.target.prefix()).collect();
    let output = model.training_forward(&frames, &prefixes).unwrap();
    let loss = loss(&output, rows);
    let value = loss.to_scalar::<f32>().unwrap();
    assert!(value.is_finite());
    let named = model.backward_named(&output, &loss).unwrap();
    let mut gradients = Vec::with_capacity(crate::MODEL_PARAMETER_COUNT);
    for named in named {
        let count = named.parameter_shape().iter().product::<usize>();
        if let Some(gradient) = named.gradient() {
            gradients.extend(gradient.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        } else {
            gradients.resize(gradients.len() + count, 0.0);
        }
    }
    drop(output);
    let parameters = model.export_parameters().unwrap();
    assert_eq!(parameters.len(), gradients.len());
    let next = crate::model::adam_step_for_test(
        &parameters,
        &gradients,
        &optimizer.first,
        &optimizer.second,
        optimizer.step,
        AdamConfig {
            learning_rate: 0.0003,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        },
    )
    .unwrap();
    model.import_parameters(&next.parameters).unwrap();
    optimizer.first = next.first_moment;
    optimizer.second = next.second_moment;
    optimizer.step += 1;
    value
}

fn loss_value(model: &PolicyModel, rows: &[&Row]) -> f32 {
    let frames: Vec<_> = rows.iter().map(|row| row.frame.clone()).collect();
    let prefixes: Vec<_> = rows.iter().map(|row| row.target.prefix()).collect();
    let output = model.training_forward(&frames, &prefixes).unwrap();
    loss(&output, rows).to_scalar::<f32>().unwrap()
}

#[derive(Debug, Default)]
struct Agreement {
    kinds: [u32; 2],
    full: [u32; 2],
    rows: [u32; 2],
    conditional: [u32; 2],
    heads: [u32; 2],
    continues: [u32; 2],
    continue_rows: [u32; 2],
}
fn agreement(model: &PolicyModel, rows: &[Row]) -> Agreement {
    let mut result = Agreement::default();
    for chunk in rows.chunks(32) {
        let frames: Vec<_> = chunk.iter().map(|row| row.frame.clone()).collect();
        let choices: Vec<_> = chunk
            .iter()
            .map(|row| model.choose(&row.frame, &row.space).unwrap())
            .collect();
        for (row, choice) in chunk.iter().zip(choices) {
            let side = row.spec.side;
            result.rows[side] += 1;
            result.kinds[side] += u32::from(choice.action.kind() == row.action.kind());
            result.full[side] += u32::from(choice.action == row.action);
            if row.action == StructuredAction::Continue {
                result.continue_rows[side] += 1;
                result.continues[side] += u32::from(choice.action == row.action);
            }
        }
        let prefixes: Vec<_> = chunk.iter().map(|row| row.target.prefix()).collect();
        let output = model.training_forward(&frames, &prefixes).unwrap();
        macro_rules! check { ($($field:ident),*) => { $(
            let logits = output.$field().to_vec2::<f32>().unwrap();
            for (row, logits) in chunk.iter().zip(logits) {
                let head = &row.target.$field;
                if head.active { result.heads[row.spec.side] += 1;
                    let choice = crate::model::masked_argmax(&logits, &head.mask).unwrap();
                    result.conditional[row.spec.side] += u32::from(choice == head.selected);
                }
            }
        )* }; }
        check!(
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
    result
}

fn free_running(
    model: &PolicyModel,
    held: &[Spec],
    report: &mut String,
    label: &str,
    started: Instant,
) -> Vec<bool> {
    let mut passes = Vec::new();
    for spec in held {
        if started.elapsed() > Duration::from_secs(180) {
            break;
        }
        let mut environment = environment(*spec);
        for _ in 0..TICKS / 3 {
            let (frame, space) =
                prepare_neural_seat_policy_sample(&mut environment.seats[spec.side]).unwrap();
            let choice = model.choose(&frame, &space).unwrap();
            let request = neural_policy_request_in_space(
                &mut environment.seats[spec.side],
                choice.action,
                &space,
            )
            .unwrap()
            .1;
            for tick in 0..3 {
                environment.advance(if tick == 0 { request } else { None });
            }
        }
        let rejection = environment.seats[spec.side].rejections;
        let pass = completed(spec.kind, &environment.result) && rejection == 0;
        writeln!(
            report,
            "{label} spec={spec:?} complete={pass} rejections={rejection} outcome={:?}",
            environment.result
        )
        .unwrap();
        passes.push(pass);
    }
    passes
}

#[test]
#[ignore = "CPU-only validated fixture supervised bootstrap; no full games or qualification."]
fn bounded_validated_fixture_bootstrap() {
    let started = Instant::now();
    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/map2-learning-20260911/skill-bootstrap-001");
    assert!(output.join("DESIGN.md").exists());
    assert!(!output.join("RESULTS.txt").exists());
    let source = std::env::var("DRYSUA_SKILL_BOOTSTRAP_INITIAL").expect("explicit initial path");
    let bytes = std::fs::read(Path::new(&source).join("drysua.weights.safetensors")).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, SHA);
    let model = PolicyModel::fresh_on(10091899, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, Path::new(&source)).unwrap();
    let train_specs = specs(false, &[0, 2]);
    let held = specs(true, &[1]);
    let train = dataset(&train_specs, started);
    let validation = dataset(&held, started);
    assert!(train.len() <= 2048 && validation.len() <= 1024);
    assert_eq!(
        validation
            .iter()
            .map(|row| row.spec.seed)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        held.len()
    );
    let mut report = format!(
        "origin=validated fixture trajectories\nsource_sha256={SHA}\ndevice=CPU\ntrain_cases={} validation_cases={} train_rows={} validation_rows={} construction_seconds={:.3}\n",
        train_specs.len(),
        held.len(),
        train.len(),
        validation.len(),
        started.elapsed().as_secs_f64()
    );
    let baseline = agreement(&model, &validation);
    writeln!(report, "baseline_agreement={baseline:?}").unwrap();
    let before = free_running(&model, &held, &mut report, "initial", started);
    let (steps, initial_loss, final_loss) =
        fit_model(&model, &train, &validation, started, &mut report);
    let final_agreement = agreement(&model, &validation);
    writeln!(report, "final_agreement={final_agreement:?}\ninitial_fixed_loss={initial_loss} final_fixed_loss={final_loss}").unwrap();
    let after = free_running(&model, &held, &mut report, "fitted", started);
    let continuation = (0..2).all(|side| {
        fraction(
            final_agreement.continues[side],
            final_agreement.continue_rows[side],
        ) + 0.05
            >= fraction(baseline.continues[side], baseline.continue_rows[side])
    });
    let passed = after.len() == held.len() && after.iter().all(|pass| *pass) && continuation;
    writeln!(report, "initial_completed={}/{} fitted_completed={}/{} continue_gate={continuation} candidate={passed} optimizer_steps={steps} total_seconds={:.3}", before.iter().filter(|pass| **pass).count(), held.len(), after.iter().filter(|pass| **pass).count(), held.len(), started.elapsed().as_secs_f64()).unwrap();
    connectivity(&model, &mut report);
    if passed {
        export(&model, &output, &report);
    }
    std::fs::write(output.join("RESULTS.txt"), &report).unwrap();
    eprintln!("{report}");
    assert_eq!(
        bytes,
        std::fs::read(Path::new(&source).join("drysua.weights.safetensors")).unwrap()
    );
}

fn dataset(specs: &[Spec], started: Instant) -> Vec<Row> {
    assert!(specs.len() <= 48);
    let mut rows = Vec::new();
    for spec in specs {
        if started.elapsed() >= Duration::from_secs(120) {
            break;
        }
        rows.extend(collect(*spec));
    }
    assert!(rows.len() <= 2048);
    rows
}

fn fit_model(
    model: &PolicyModel,
    train: &[Row],
    validation: &[Row],
    started: Instant,
    report: &mut String,
) -> (u64, f32, f32) {
    let mut optimizer = Optimizer {
        first: vec![0.0; model.parameter_count()],
        second: vec![0.0; model.parameter_count()],
        step: 0,
    };
    let fit_started = Instant::now();
    let mut random = PpoRng::new(10091899);
    let probe: Vec<_> = train.iter().take(32).collect();
    let initial_loss = loss_value(model, &probe);
    for step in 1..=256 {
        if fit_started.elapsed() >= Duration::from_secs(120)
            || started.elapsed() >= Duration::from_secs(150)
        {
            break;
        }
        let batch: Vec<_> = (0..32)
            .map(|_| &train[random.below(train.len() as u64).unwrap() as usize])
            .collect();
        let loss = update(model, &batch, &mut optimizer);
        if step == 8 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} batch_loss={loss} fixed_probe_loss={} fit_seconds={:.3}",
                loss_value(model, &probe),
                fit_started.elapsed().as_secs_f64()
            )
            .unwrap();
        }
        if step == 8 {
            writeln!(
                report,
                "preview_agreement={:?}",
                agreement(model, validation)
            )
            .unwrap();
            eprintln!(
                "bootstrap first-eight fixed_loss={initial_loss}->{}",
                loss_value(model, &probe)
            );
        }
        if step == 8 && loss_value(model, &probe) >= initial_loss {
            report.push_str("STOP=first_eight_updates_did_not_reduce_fixed_training_loss\n");
            break;
        }
    }
    (optimizer.step, initial_loss, loss_value(model, &probe))
}

fn fraction(count: u32, total: u32) -> f64 {
    assert!(total > 0);
    f64::from(count) / f64::from(total)
}
fn connectivity(model: &PolicyModel, report: &mut String) {
    let parameters = model.export_parameters().unwrap();
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().unwrap() {
        if name == "unit.0.weight" {
            let nonzero = parameters[offset + 73 * 64..offset + 76 * 64]
                .iter()
                .filter(|value| **value != 0.0)
                .count();
            writeln!(report, "effect15_connected_weights={nonzero}/192").unwrap();
        }
        offset += shape.iter().product::<usize>();
    }
}
fn export(model: &PolicyModel, root: &Path, report: &str) {
    let target = root.join("candidate");
    assert!(!target.exists());
    std::fs::create_dir(&target).unwrap();
    TrainingArtifact::save_runtime_weights(model, &target).unwrap();
    let bytes = std::fs::read(target.join("drysua.weights.safetensors")).unwrap();
    let digest: String = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    std::fs::write(target.join("MANIFEST.txt"), format!("UNQUALIFIED_LOCAL_CANDIDATE\nfreeze_sha256={digest}\nfeature={} model={} action={}\n{report}", crate::FEATURE_SCHEMA_VERSION, crate::MODEL_SCHEMA_VERSION, crate::ACTION_SCHEMA_VERSION)).unwrap();
}
