use super::*;

#[test]
fn m12_initialization_layout_accepts_exactly_62_ordered_names_and_shapes() {
    let model = PolicyModel::fresh(43).expect("model");
    let schema = model.parameter_schema().expect("schema");

    PolicyModel::validate_m12_parameter_schema(&schema).expect("unchanged M12 parameter layout");

    assert_eq!(schema.len(), 62);
    assert_eq!(
        schema
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>(),
        1_689_076
    );
}

#[test]
fn m12_initialization_layout_rejects_each_changed_name_shape_order_and_count() {
    let model = PolicyModel::fresh(44).expect("model");
    let schema = model.parameter_schema().expect("schema");
    for index in 0..62 {
        let mut changed = schema.clone();
        changed[index].0 = "unapproved";
        assert_m12_layout_error(&changed, "selected M12 parameter name or order");
        let mut changed = schema.clone();
        changed[index].1.insert(0, 1);
        assert_m12_layout_error(&changed, "selected M12 parameter shape");
    }
    let mut reordered = schema.clone();
    reordered.swap(0, 1);
    assert_m12_layout_error(&reordered, "selected M12 parameter name or order");
    assert_m12_layout_error(&schema[..61], "selected M12 parameter tensor count");
    let mut extra = schema.clone();
    extra.push(schema[0].clone());
    assert_m12_layout_error(&extra, "selected M12 parameter tensor count");
}

fn assert_m12_layout_error(schema: &[(&str, Vec<usize>)], field: &'static str) {
    let error = PolicyModel::validate_m12_parameter_schema(schema).expect_err("exact layout");
    let expected = ModelError::InvalidModelState(field);
    assert_eq!(error.to_string(), format!("model produced invalid {field}"));
    assert_eq!(error, expected);
}

#[test]
fn m11_named_input_layout_copies_every_old_parameter_and_zeros_only_appended_inputs() {
    let model = PolicyModel::fresh(41).expect("model");
    let source: Vec<f32> = (0..1_684_724).map(|index| index as f32 + 1.0).collect();
    let target = model
        .widen_m11_input_parameters(&source)
        .expect("audited layout");
    assert_eq!(target.len(), MODEL_PARAMETER_COUNT);
    let mut source_offset = 0;
    let mut target_offset = 0;
    let mut zeros = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        let (insert, added) = match name {
            "unit.0.weight" => (69 * 64, 4 * 64),
            "trunk.0.weight" => (64 * 512, 8 * 512),
            _ => (count, 0),
        };
        for index in 0..count {
            let expected = if (insert..insert + added).contains(&index) {
                zeros += 1;
                0.0
            } else {
                source[source_offset + index - usize::from(index >= insert + added) * added]
            };
            assert_eq!(
                target[target_offset + index],
                expected,
                "{name} index={index}"
            );
        }
        source_offset += count - added;
        target_offset += count;
    }
    assert_eq!(source_offset, source.len());
    assert_eq!(zeros, 4352);
}

#[test]
fn m11_input_adapter_rejects_wrong_length_and_nonfinite_source() {
    let model = PolicyModel::fresh(42).expect("model");
    let error = model
        .widen_m11_input_parameters(&[0.0])
        .expect_err("old width required");
    assert_eq!(
        error,
        ModelError::ParameterLength {
            actual: 1,
            expected: 1_684_724
        }
    );
    let mut source = vec![0.0; 1_684_724];
    source[123] = f32::NAN;
    assert_eq!(
        model
            .widen_m11_input_parameters(&source)
            .expect_err("finite source"),
        ModelError::NonFiniteParameter { index: 123 }
    );
}

fn fact_probe_rows() -> Vec<(ImitationSample, ActionSpace)> {
    (0..4)
        .map(|bits| {
            let mut view = world_view(Team::Radiant, 100);
            let enemy = view
                .units
                .iter_mut()
                .find(|unit| unit.team == Team::Dire && unit.kind == UnitKind::Hero)
                .expect("visible enemy");
            enemy.statuses.bits = if bits & 1 != 0 {
                bota_proto::StatusFlags::INVULNERABLE
            } else {
                0
            } | if bits & 2 != 0 {
                bota_proto::StatusFlags::CHANNELLING
            } else {
                0
            };
            let tracker = tracker_with_view(Team::Radiant, view);
            let space = ActionSpace::from_tracker(&tracker).expect("space");
            let frame = encode(&tracker, &LocalPolicyState::new(0));
            let kind = if bits & 2 != 0 {
                ActionKind::AttackMovePoint
            } else {
                ActionKind::MovePoint
            };
            let mut logits = DecoderLogits::favor(kind);
            logits.point[bits & 1] = 10.0;
            let action = decode_with_logits(&space, &logits).expect("synthetic diagnostic label");
            let identity = SampleIdentity::from_frame(
                SeedNamespace::Training,
                9873300 + bits as u64,
                0,
                100,
                &frame,
            )
            .expect("identity");
            (
                ImitationSample::teacher(frame, &space, action, identity).expect("probe row"),
                space,
            )
        })
        .collect()
}

fn without_status_facts(frame: &FeatureFrame) -> FeatureFrame {
    let mut frame = frame.clone();
    for row in frame
        .units
        .iter_mut()
        .chain(&mut frame.own_units)
        .chain(&mut frame.remembered_units)
    {
        row[unit_feature::INVULNERABLE] = 0.0;
        row[unit_feature::CHANNELLING] = 0.0;
    }
    frame
}

#[test]
fn status_probe_labels_are_aliased_without_new_facts_and_have_identical_legal_masks() {
    let rows = fact_probe_rows();
    let baseline = without_status_facts(rows[0].0.frame());
    for (sample, _) in &rows {
        assert_eq!(without_status_facts(sample.frame()), baseline);
        assert_eq!(sample.target().kind.mask, rows[0].0.target().kind.mask);
        assert_eq!(
            sample.target().controlled.mask,
            rows[0].0.target().controlled.mask
        );
        assert_eq!(
            sample.target().point_pointer.mask,
            rows[0].0.target().point_pointer.mask
        );
    }
    assert_ne!(rows[0].0.teacher_action(), rows[1].0.teacher_action());
    assert_ne!(
        rows[0].0.teacher_action().kind(),
        rows[2].0.teacher_action().kind()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "bounded 512-update CUDA fact-causality probe; synthetic labels, not a strategy"]
fn approved_m11_zero_extended_inputs_learn_status_dependent_kind_and_point_without_alias() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/map0-ppo21-u010-u040-001/baseline-u010");
    let (model, _) = crate::TrainingArtifact::initialize_selected_m11_for_training(
        &directory,
        9873300,
        PolicyDevice::Cuda { ordinal: 0 },
    )
    .expect("approved source");
    let rows = fact_probe_rows();
    let examples: Vec<_> = rows.iter().map(|(sample, _)| sample).collect();
    let mut optimizer = model
        .claim_adam_for_test(AdamConfig {
            learning_rate: 0.001,
            ..AdamConfig::default()
        })
        .expect("fresh optimizer");
    let before = model
        .behavioral_loss_for_test(&examples)
        .expect("initial loss");
    for _ in 0..512 {
        model
            .behavioral_update(&examples, &mut optimizer)
            .expect("bounded update");
    }
    let after = model
        .behavioral_loss_for_test(&examples)
        .expect("final loss");
    let mut correct = 0;
    let mut ablated_correct = 0;
    for (sample, space) in &rows {
        correct += usize::from(
            model.choose(sample.frame(), space).expect("choice").action == sample.teacher_action(),
        );
        ablated_correct += usize::from(
            model
                .choose(&without_status_facts(sample.frame()), space)
                .expect("ablated choice")
                .action
                == sample.teacher_action(),
        );
    }
    eprintln!(
        "fact_probe updates=512 rows=4 loss={before}->{after} actual_exact={correct}/4 ablated_exact={ablated_correct}/4"
    );
    assert_eq!(correct, 4);
    assert!(ablated_correct <= 1);
    assert!(after < 0.02);
    let parameters = model.export_parameters().expect("trained parameters");
    assert!(
        parameters[69 * 64..71 * 64]
            .iter()
            .any(|value| value.abs() > 1e-6)
    );
}
