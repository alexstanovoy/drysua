use super::*;

const M11_PARAMETERS: usize = 1_684_724;
const M12_PARAMETERS: usize = 1_689_076;

#[test]
fn retired_m12_initialization_rejects_exact_legacy_and_current_map2_layouts() {
    let model = PolicyModel::fresh(43).expect("model");
    let current = model.parameter_schema().expect("current schema");
    let mut historical = current.clone();
    for (name, shape) in &mut historical {
        match *name {
            "unit.0.weight" => *shape = vec![73, 64],
            "trunk.0.weight" => *shape = vec![2576, 512],
            _ => {}
        }
    }
    assert_eq!(historical.len(), 62);
    assert_eq!(
        historical
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>(),
        M12_PARAMETERS
    );
    assert_ne!(historical, current);
    assert_retired_m12_layout(&historical);
    assert_retired_m12_layout(&current);
}

#[test]
fn map2_layout_validation_keeps_name_shape_order_count_checks_and_cannot_reenable_m12() {
    let model = PolicyModel::fresh(44).expect("model");
    let schema = model.parameter_schema().expect("schema");
    PolicyModel::validate_map2_parameter_schema(&schema).expect("current Map2 layout");
    for index in 0..62 {
        let mut changed = schema.clone();
        changed[index].0 = "unapproved";
        assert_map2_and_retired_layout_errors(&changed, "Map2 parameter name or order");
        let mut changed = schema.clone();
        changed[index].1.insert(0, 1);
        assert_map2_and_retired_layout_errors(&changed, "Map2 parameter shape");
    }
    let mut reordered = schema.clone();
    reordered.swap(0, 1);
    assert_map2_and_retired_layout_errors(&reordered, "Map2 parameter name or order");
    assert_map2_and_retired_layout_errors(&schema[..61], "Map2 parameter tensor count");
    let mut extra = schema.clone();
    extra.push(schema[0].clone());
    assert_map2_and_retired_layout_errors(&extra, "Map2 parameter tensor count");
}

fn assert_map2_and_retired_layout_errors(schema: &[(&str, Vec<usize>)], field: &'static str) {
    let error =
        PolicyModel::validate_map2_parameter_schema(schema).expect_err("exact current layout");
    let expected = ModelError::InvalidModelState(field);
    assert_eq!(error.to_string(), format!("model produced invalid {field}"));
    assert_eq!(error, expected);
    assert_retired_m12_layout(schema);
}

fn assert_retired_m12_layout(schema: &[(&str, Vec<usize>)]) {
    let error =
        PolicyModel::validate_m12_parameter_schema(schema).expect_err("retired M12 adapter");
    assert_eq!(
        error,
        ModelError::InvalidModelState(
            "M12 initialization retired; use pinned M14 Map2 initialization"
        )
    );
    assert_eq!(
        error.to_string(),
        "model produced invalid M12 initialization retired; use pinned M14 Map2 initialization"
    );
}

#[test]
fn retired_m11_finite_input_rejects_without_copying_or_padding_any_parameter() {
    let model = PolicyModel::fresh(41).expect("model");
    let source: Vec<f32> = (0..M11_PARAMETERS)
        .map(|index| index as f32 + 1.0)
        .collect();
    let identity = model.policy_identity().expect("identity");
    let before = model.export_parameters().expect("parameters");
    assert_ne!(source.len(), model.parameter_count());

    let error = model
        .widen_m11_input_parameters(&source)
        .expect_err("M11 initialization is retired");

    assert_eq!(
        error,
        ModelError::InvalidModelState(
            "M11 initialization retired; use pinned M14 Map2 initialization"
        )
    );
    assert_eq!(
        error.to_string(),
        "model produced invalid M11 initialization retired; use pinned M14 Map2 initialization"
    );
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
    crate::tests::map2_model_initialization::assert_bits(
        &model.export_parameters().expect("unchanged parameters"),
        &before,
    );
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
            expected: M11_PARAMETERS
        }
    );
    assert_eq!(
        error.to_string(),
        "model parameter length 1 differs from expected 1684724"
    );
    let mut source = vec![0.0; M11_PARAMETERS];
    source[123] = f32::NAN;
    let error = model
        .widen_m11_input_parameters(&source)
        .expect_err("finite source");
    assert_eq!(error, ModelError::NonFiniteParameter { index: 123 });
    assert_eq!(error.to_string(), "model parameter 123 is non-finite");
    let error = model
        .widen_m11_input_parameters(&vec![0.0; MODEL_PARAMETER_COUNT])
        .expect_err("M15 count cannot be used for M11");
    assert_eq!(
        error,
        ModelError::ParameterLength {
            actual: MODEL_PARAMETER_COUNT,
            expected: M11_PARAMETERS
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "model parameter length {MODEL_PARAMETER_COUNT} differs from expected {M11_PARAMETERS}"
        )
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
fn status_probe_labels_are_aliased_without_facts_and_per_family_masks_are_identical() {
    let rows = fact_probe_rows();
    let baseline = without_status_facts(rows[0].0.frame());
    for (sample, space) in &rows {
        assert_eq!(without_status_facts(sample.frame()), baseline);
        assert_eq!(sample.target().kind.mask, rows[0].0.target().kind.mask);
        assert_eq!(
            sample.target().controlled.mask,
            rows[0].0.target().controlled.mask
        );
        // A5 Move admits landing cells, unlike AttackMove; compare conditional masks within a family.
        let family = usize::from(sample.teacher_action().kind() == ActionKind::AttackMovePoint) * 2;
        assert_eq!(
            sample.target().point_pointer.mask,
            rows[family].0.target().point_pointer.mask
        );
        assert_eq!(
            space.move_point_mask(ControlledUnit::Hero),
            rows[0].1.move_point_mask(ControlledUnit::Hero)
        );
        assert_eq!(
            space.attack_move_point_mask(ControlledUnit::Hero),
            rows[0].1.attack_move_point_mask(ControlledUnit::Hero)
        );
    }
    assert_ne!(rows[0].0.teacher_action(), rows[1].0.teacher_action());
    assert_ne!(
        rows[0].0.teacher_action().kind(),
        rows[2].0.teacher_action().kind()
    );
}

#[test]
#[ignore = "read-only local M11 source audit; retired initialization must reject before any fact-learning probe"]
fn retired_m11_local_source_cannot_initialize_a_fact_learning_probe() {
    use sha2::{Digest, Sha256};
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/map0-ppo21-u010-u040-001/baseline-u010");
    let path = directory.join("drysua.weights.safetensors");
    let bytes = std::fs::read(&path).expect("historical source");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        digest,
        "5bbb8843fec88f3c6443618de9cabeba44ff9dbb0b7f9a9856c2c8936551e880"
    );
    let error = crate::TrainingArtifact::initialize_selected_m11_for_training(
        &directory,
        9873300,
        PolicyDevice::Cpu,
    )
    .err()
    .expect("M11 source cannot initialize Map2");
    assert_eq!(error, crate::CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(std::fs::read(path).expect("source unchanged"), bytes);
}
