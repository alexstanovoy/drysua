use std::fs;

use super::map2_checkpoint::{Directory, M14_PARAMETERS, m14_metadata, runtime_bytes};
use crate::{CheckpointError, ModelError, PolicyDevice, PolicyModel, TrainingArtifact};

#[test]
fn map2_m14_padding_preserves_every_old_bit_and_only_inserts_positive_zero_rows() {
    let model = PolicyModel::fresh(201).expect("model");
    let mut source: Vec<_> = (0..M14_PARAMETERS)
        .map(|index| f32::from_bits(0x3e00_0000 + index as u32))
        .collect();
    for (index, value) in [
        (0, -0.0),
        (73 * 64 - 1, f32::MIN_POSITIVE),
        (M14_PARAMETERS - 1, f32::MAX),
        (73 * 64, f32::from_bits(1)),
        (58_368 + 72 * 512 - 1, -0.0),
        (58_368 + 72 * 512, f32::from_bits(0x8000_0001)),
        (58_368 + 2576 * 512 - 1, f32::MIN),
    ] {
        source[index] = value;
    }
    let before = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");

    let target = model
        .widen_m14_input_parameters(&source)
        .expect("audited padding");

    assert_padding(&model, &source, &target);
    assert!(target.iter().all(|value| value.is_finite()));
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
    assert_eq!(model.export_parameters().expect("unchanged model"), before);
    model
        .import_parameters(&target)
        .expect("finite padded import");
    assert_bits(&model.export_parameters().expect("imported"), &target);
}

pub(super) fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(left, right)| left.to_bits() == right.to_bits())
    );
}

pub(super) fn assert_padding(model: &PolicyModel, source: &[f32], target: &[f32]) {
    let schema = model.parameter_schema().expect("schema");
    assert_eq!(schema.len(), 62);
    assert_eq!(source.len(), M14_PARAMETERS);
    assert_eq!(target.len(), M14_PARAMETERS + 7_360);
    let mut source_offset = 0;
    let mut target_offset = 0;
    let mut inserted_total = 0;
    for (name, shape) in schema {
        let count = shape.iter().product::<usize>();
        let (prefix, inserted) = match name {
            "unit.0.weight" => {
                assert_eq!(shape, [84, 64]);
                (73 * 64, 11 * 64)
            }
            "trunk.0.weight" => {
                assert_eq!(shape, [2589, 512]);
                (72 * 512, 13 * 512)
            }
            _ => (count, 0),
        };
        let old = &source[source_offset..source_offset + count - inserted];
        let new = &target[target_offset..target_offset + count];
        assert_bits(&new[..prefix], &old[..prefix]);
        assert!(
            new[prefix..prefix + inserted]
                .iter()
                .all(|value| value.to_bits() == 0)
        );
        assert_bits(&new[prefix + inserted..], &old[prefix..]);
        source_offset += old.len();
        target_offset += new.len();
        inserted_total += inserted;
    }
    assert_eq!(source_offset, source.len());
    assert_eq!(target_offset, target.len());
    assert_eq!(inserted_total, 7_360);
}

#[test]
fn map2_m14_padding_rejects_lengths_and_nonfinite_values_before_mutation() {
    let model = PolicyModel::fresh(202).expect("model");
    let identity = model.policy_identity().expect("identity");
    let before = model.export_parameters().expect("parameters");
    for count in [0, M14_PARAMETERS - 1, M14_PARAMETERS + 1] {
        let error = model
            .widen_m14_input_parameters(&vec![0.0; count])
            .expect_err("length");
        assert_eq!(
            error,
            ModelError::ParameterLength {
                actual: count,
                expected: M14_PARAMETERS
            }
        );
        assert_eq!(
            error.to_string(),
            format!("model parameter length {count} differs from expected {M14_PARAMETERS}")
        );
    }
    let mut source = vec![0.0; M14_PARAMETERS];
    for index in [0, 73 * 64, 58_368 + 72 * 512, M14_PARAMETERS - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            source[index] = value;
            let error = model
                .widen_m14_input_parameters(&source)
                .expect_err("finite source");
            assert_eq!(error, ModelError::NonFiniteParameter { index });
            assert_eq!(
                error.to_string(),
                format!("model parameter {index} is non-finite")
            );
        }
        source[index] = 0.0;
    }
    assert_eq!(
        model.policy_identity().expect("unchanged identity"),
        identity
    );
    assert_bits(
        &model.export_parameters().expect("unchanged parameters"),
        &before,
    );
}

#[test]
fn map2_padded_bounded_parameters_produce_finite_outputs_for_all_heads() {
    let model = PolicyModel::fresh(210).expect("model");
    let parameters = model
        .widen_m14_input_parameters(&vec![0.0001; M14_PARAMETERS])
        .expect("bounded finite padding");
    model.import_parameters(&parameters).expect("finite import");
    let mut frame = crate::FeatureFrame::new();
    let appended = frame.global.get_mut(72..85).expect("F14 global append");
    appended[..10].fill(1.0);
    appended[10] = 0.05;
    appended[11] = 0.01;
    appended[12] = 1.0;
    frame.units[0][crate::unit_feature::TOKEN_PRESENT] = 1.0;
    frame.units[0][73..84].fill(1.0);
    let prefix = crate::TrainingPrefix::new(crate::ActionKind::Continue, None, None);

    let output = model
        .training_forward(&[frame], &[prefix])
        .expect("all-head forward");

    output.validate_finite().expect("every head remains finite");
    assert_eq!(output.shapes().value, [1, 1]);
    assert_eq!(output.shapes().entity_pointer, [1, 96]);
}

#[test]
fn map2_named_layout_rejects_every_changed_name_shape_order_and_count() {
    let model = PolicyModel::fresh(203).expect("model");
    let schema = model.parameter_schema().expect("schema");
    PolicyModel::validate_map2_parameter_schema(&schema).expect("audited layout");
    for index in 0..schema.len() {
        let mut changed = schema.clone();
        changed[index].0 = "unknown";
        assert_layout_error(&changed, "Map2 parameter name or order");
        let mut changed = schema.clone();
        changed[index].1[0] += 1;
        assert_layout_error(&changed, "Map2 parameter shape");
    }
    let mut changed = schema.clone();
    changed.swap(0, 1);
    assert_layout_error(&changed, "Map2 parameter name or order");
    assert_layout_error(&schema[..61], "Map2 parameter tensor count");
    changed.push(schema[0].clone());
    assert_layout_error(&changed, "Map2 parameter tensor count");
}

fn assert_layout_error(schema: &[(&str, Vec<usize>)], field: &'static str) {
    let error = PolicyModel::validate_map2_parameter_schema(schema).expect_err("strict layout");
    assert_eq!(error, ModelError::InvalidModelState(field));
    assert_eq!(error.to_string(), format!("model produced invalid {field}"));
}

#[test]
fn map2_initializer_rejects_each_missing_wrong_or_extra_source_metadata_key() {
    let directory = Directory::new();
    let values = vec![0.0; M14_PARAMETERS];
    for version in [26, 27] {
        let metadata = m14_metadata(version);
        for key in metadata.keys() {
            for replacement in [None, Some("wrong")] {
                let mut changed = metadata.clone();
                changed.remove(key);
                if let Some(value) = replacement {
                    changed.insert(key.clone(), value.to_owned());
                }
                assert_initializer_error(
                    &directory,
                    runtime_bytes(&values, changed),
                    CheckpointError::SchemaMismatch,
                );
            }
        }
        let mut extra = metadata;
        extra.insert("map2_relabel".to_owned(), "false".to_owned());
        assert_initializer_error(
            &directory,
            runtime_bytes(&values, extra),
            CheckpointError::SchemaMismatch,
        );
    }
}

#[test]
fn map2_initializer_rejects_unpinned_digest_and_nonfinite_payload_at_boundaries() {
    let directory = Directory::new();
    let mut source = vec![0.0; M14_PARAMETERS];
    for version in [26, 27] {
        assert_initializer_error(
            &directory,
            runtime_bytes(&source, m14_metadata(version)),
            CheckpointError::TensorContract("selected M14 Map2 initialization source SHA-256"),
        );
        for index in [0, M14_PARAMETERS - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                source[index] = value;
                assert_initializer_error(
                    &directory,
                    runtime_bytes(&source, m14_metadata(version)),
                    CheckpointError::NonFiniteTensor {
                        name: "model.parameters",
                        index,
                    },
                );
            }
            source[index] = 0.0;
        }
    }
}

fn assert_initializer_error(directory: &Directory, bytes: Vec<u8>, expected: CheckpointError) {
    let path = directory.0.join("drysua.weights.safetensors");
    fs::write(&path, &bytes).expect("fixture");
    let error =
        TrainingArtifact::initialize_selected_m14_for_map2(&directory.0, 204, PolicyDevice::Cpu)
            .err()
            .expect("initialization rejected");
    let message = match expected {
        CheckpointError::SchemaMismatch => "checkpoint schema does not match this build".to_owned(),
        CheckpointError::TensorContract(field) => {
            format!("checkpoint tensor contract has invalid {field}")
        }
        CheckpointError::NonFiniteTensor { name, index } => {
            format!("checkpoint tensor {name} contains non-finite value at {index}")
        }
        _ => panic!("unsupported initializer error fixture"),
    };
    assert_eq!(error.to_string(), message);
    assert_eq!(error, expected);
    assert_eq!(fs::read(path).expect("source untouched"), bytes);
}

#[test]
fn map2_initializer_rejects_tensor_name_dtype_rank_count_and_extra_tensor() {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let directory = Directory::new();
    let data = vec![0u8; M14_PARAMETERS * 4];
    for (name, dtype, shape, field) in [
        ("wrong", Dtype::F32, vec![M14_PARAMETERS], "names"),
        (
            "model.parameters",
            Dtype::I32,
            vec![M14_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![1, M14_PARAMETERS],
            "dtype or shape",
        ),
        (
            "model.parameters",
            Dtype::F32,
            vec![M14_PARAMETERS - 1],
            "dtype or shape",
        ),
    ] {
        let count = shape.iter().product::<usize>();
        let tensor = TensorView::new(dtype, shape, &data[..count * 4]).expect("tensor");
        let bytes = serialize([(name, tensor)], Some(m14_metadata(27))).expect("fixture");
        assert_initializer_error(&directory, bytes, CheckpointError::TensorContract(field));
    }
    let tensor = TensorView::new(Dtype::F32, vec![M14_PARAMETERS], &data).expect("tensor");
    let extra = TensorView::new(Dtype::F32, vec![1], &data[..4]).expect("extra");
    let bytes = serialize(
        [("model.parameters", tensor), ("extra", extra)],
        Some(m14_metadata(27)),
    )
    .expect("fixture");
    assert_initializer_error(&directory, bytes, CheckpointError::TensorContract("names"));
}

#[test]
fn map2_padded_policy_has_fresh_optimizer_and_round_trips_without_progress() {
    use super::map2_checkpoint::{config, progress, run};
    let directory = Directory::new();
    let model = PolicyModel::fresh(207).expect("model");
    let source = vec![0.125; M14_PARAMETERS];
    let parameters = model.widen_m14_input_parameters(&source).expect("padding");
    model.import_parameters(&parameters).expect("import");
    let trainer = crate::PpoTrainer::new(&model, config(), 208).expect("fresh optimizer");
    assert_fresh_state(&model, &trainer);
    let run = run();
    TrainingArtifact::capture(&model, &trainer, run.clone(), progress())
        .expect("capture")
        .save(&directory.0)
        .expect("save");
    let checkpoint = TrainingArtifact::load_compatible(&directory.0, &run).expect("load");
    assert_eq!(checkpoint.progress(), &progress());
    let restored = PolicyModel::fresh(209).expect("new model");
    let state = checkpoint
        .restore(&restored, &run)
        .expect("restore new checkpoint only");
    assert_fresh_state(&restored, state.trainer());
    assert_bits(
        &restored.export_parameters().expect("restored"),
        &parameters,
    );
    assert_ne!(
        model.policy_identity().expect("identity"),
        restored.policy_identity().expect("identity")
    );
}

pub(super) fn assert_fresh_state(model: &PolicyModel, trainer: &crate::PpoTrainer) {
    assert_eq!(trainer.updates(), 0);
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.rng_checkpoint().1, 0);
    let snapshot = trainer
        .checkpoint_snapshot(model)
        .expect("bound fresh optimizer");
    for moments in [snapshot.adam.moments().0, snapshot.adam.moments().1] {
        assert_eq!(moments.len(), crate::MODEL_PARAMETER_COUNT);
        assert!(moments.iter().all(|value| value.to_bits() == 0));
    }
}
