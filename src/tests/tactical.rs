#![allow(
    clippy::float_arithmetic,
    reason = "tests exercise bounded f32 policy parameters"
)]

use crate::{
    TACTICAL_FEATURES, TACTICAL_HIDDEN, TACTICAL_OUTPUT_BIAS_OFFSET, TACTICAL_PARAMETERS,
    TacticalFeatures, TacticalMode, TacticalPolicy,
};

#[test]
fn tactical_v2_rejects_released_v1_weights_instead_of_reinterpreting_them() {
    let bytes = include_bytes!("../../artifacts/v0.0.4/drysua.tactical.bin");

    let error = TacticalPolicy::from_bytes(bytes).expect_err("v1 is not a v2 policy");

    assert_eq!(
        error.to_string(),
        "tactical file schema does not match this architecture and feature order"
    );
    assert!(crate::TACTICAL_SCHEMA_DESCRIPTOR.starts_with("drysua-tactical/v2;"));
}

#[test]
fn tactical_v2_descriptor_records_macro_and_decision_cadence_semantics() {
    let descriptor = crate::TACTICAL_SCHEMA_DESCRIPTOR;

    assert!(descriptor.contains("Recover=lane_backoff"));
    assert!(descriptor.contains("decision_ticks=1+3n;pregame=enabled"));
    assert!(descriptor.contains("Farm=last_hit_deny_aggro_pull"));
}

#[test]
fn tactical_parameters_reject_wrong_shape_with_specific_error() {
    let error = TacticalPolicy::from_parameters(&[0.0; 1]).expect_err("wrong shape");

    assert_eq!(
        error.to_string(),
        "tactical parameters have 1 values; expected 236"
    );
}

#[test]
fn tactical_parameters_reject_nonfinite_and_out_of_bounds_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 4.001, -4.001] {
        let mut parameters = [0.0; TACTICAL_PARAMETERS];
        parameters[7] = value;

        let error = TacticalPolicy::from_parameters(&parameters).expect_err("invalid weight");

        assert_eq!(
            error.to_string(),
            "tactical parameter 7 must be finite and in [-4, 4]"
        );
    }
}

#[test]
fn tactical_features_reject_nonfinite_and_out_of_bounds_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.001, -1.001] {
        let mut values = [0.0; TACTICAL_FEATURES];
        values[3] = value;

        let error = TacticalFeatures::from_values(values).expect_err("invalid feature");

        assert_eq!(
            error.to_string(),
            "tactical feature 3 must be finite and in [-1, 1]"
        );
    }
}

#[test]
fn tactical_boundary_parameters_and_features_produce_finite_bounded_logits() {
    for weight in [-4.0, 4.0] {
        let policy = TacticalPolicy::from_parameters(&[weight; TACTICAL_PARAMETERS])
            .expect("boundary parameters");
        for value in [-1.0, 1.0] {
            let features = TacticalFeatures::from_values([value; TACTICAL_FEATURES])
                .expect("boundary features");

            let scores = policy.scores(&features);

            assert!(scores.iter().all(|score| score.is_finite()));
            assert!(scores.iter().all(|score| score.abs() <= 36.0));
        }
    }
}

#[test]
fn tactical_default_and_zero_parameters_choose_teacher_for_all_available_modes() {
    let zero = TacticalPolicy::from_parameters(&[0.0; TACTICAL_PARAMETERS]).expect("zeros");
    for policy in [zero, TacticalPolicy::default()] {
        for value in [-1.0, 0.0, 1.0] {
            let features =
                TacticalFeatures::from_values([value; TACTICAL_FEATURES]).expect("features");

            assert_eq!(policy.choose(&features, [true; 4]), TacticalMode::Teacher);
        }
    }
}

#[test]
fn tactical_hidden_neuron_changes_mode_with_observation_not_seed() {
    let mut parameters = [0.0; TACTICAL_PARAMETERS];
    parameters[0] = 1.0;
    parameters[(TACTICAL_FEATURES + 1) * TACTICAL_HIDDEN + TACTICAL_HIDDEN] = 1.0;
    parameters[TACTICAL_OUTPUT_BIAS_OFFSET] = 0.5;
    let policy = TacticalPolicy::from_parameters(&parameters).expect("conditional policy");
    let low = TacticalFeatures::from_values([0.0; TACTICAL_FEATURES]).expect("low");
    let mut high = [0.0; TACTICAL_FEATURES];
    high[0] = 1.0;
    let high = TacticalFeatures::from_values(high).expect("high");

    assert_eq!(policy.choose(&low, [true; 4]), TacticalMode::Teacher);
    assert_eq!(policy.choose(&high, [true; 4]), TacticalMode::Fight);
    assert_eq!(
        policy.choose(&high, [true, false, true, true]),
        TacticalMode::Teacher
    );
}

#[test]
fn tactical_mutations_are_deterministic_bounded_and_do_not_change_parent() {
    let parent = TacticalPolicy::default();
    let original = parent.clone();

    let first = parent.mutated(0, 0.5).expect("mutation");
    let repeated = parent.mutated(0, 0.5).expect("repeated mutation");
    let other = parent.mutated(1, 0.5).expect("different seed");

    assert_eq!(first, repeated);
    assert_ne!(first, other);
    assert_eq!(parent, original);
    assert_eq!(parent.mutated(123, 0.0).expect("zero mutation"), parent);
    assert!(TacticalPolicy::from_parameters(first.parameters()).is_ok());
}

#[test]
fn tactical_mutation_rejects_invalid_scale_with_specific_error() {
    for scale in [-0.001, 1.001, f32::NAN, f32::INFINITY] {
        let error = TacticalPolicy::default()
            .mutated(0, scale)
            .expect_err("invalid scale");

        assert_eq!(
            error.to_string(),
            "tactical mutation scale must be finite and in [0, 1]"
        );
    }
}

#[test]
fn tactical_serialization_round_trips_exact_parameter_bits() {
    let policy = TacticalPolicy::default().mutated(9, 0.25).expect("policy");

    let bytes = policy.to_bytes();
    let restored = TacticalPolicy::from_bytes(&bytes).expect("restored");

    assert_eq!(policy, restored);
    assert_eq!(bytes, restored.to_bytes());
}

#[test]
fn tactical_serialization_rejects_shape_schema_and_invalid_weights() {
    let policy = TacticalPolicy::default();
    let mut bytes = policy.to_bytes();
    bytes.push(0);
    let error = TacticalPolicy::from_bytes(&bytes).expect_err("trailing bytes");
    assert_eq!(
        error.to_string(),
        format!(
            "tactical file has {} bytes; expected {}",
            bytes.len(),
            bytes.len() - 1
        )
    );
    bytes.pop();
    bytes[0] ^= 1;
    assert_eq!(
        TacticalPolicy::from_bytes(&bytes)
            .expect_err("schema")
            .to_string(),
        "tactical file schema does not match this architecture and feature order"
    );
    let mut bytes = policy.to_bytes();
    let offset = bytes.len() - TACTICAL_PARAMETERS * 4;
    bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(
        TacticalPolicy::from_bytes(&bytes)
            .expect_err("nan")
            .to_string(),
        "tactical parameter 0 must be finite and in [-4, 4]"
    );
}
