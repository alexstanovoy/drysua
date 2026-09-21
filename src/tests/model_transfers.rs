use super::*;

#[test]
fn packed_gradients_preserve_descriptor_order_missing_spans_and_float_bits() {
    assert_gradient_parity(&Device::Cpu);
}

#[test]
fn packed_contiguous_gradients_preserve_offset_views_and_count_only_shapes() {
    assert_contiguous_gradient_parity(&Device::Cpu);
}

#[test]
fn actual_backward_gradients_match_legacy_readback_bitwise() {
    assert_backward_gradient_parity(PolicyDevice::Cpu);
}

#[test]
fn packed_single_gradient_preserves_offset_and_strided_views() {
    assert_single_gradient_parity(&Device::Cpu);
}

#[test]
fn scalar_and_zero_sized_gradients_preserve_legacy_bits_and_offsets() {
    assert_scalar_gradient_parity(&Device::Cpu);
}

#[test]
fn zero_sized_gradients_produce_only_positive_zero() {
    assert_zero_sized_gradient_parity(&Device::Cpu);
}

#[test]
fn empty_gradient_descriptors_return_parameter_count_error() {
    let named = Vec::new();
    let expected = legacy_collect_gradients(&named).expect_err("legacy empty descriptors");

    let packed = collect_packed_gradients(&named).expect_err("packed empty descriptors");
    let selected = collect_host_gradients(named).expect_err("selected empty descriptors");

    assert_eq!(
        expected.to_string(),
        "model produced invalid gradient parameter count"
    );
    assert_eq!(packed.to_string(), expected.to_string());
    assert_eq!(selected.to_string(), expected.to_string());
}

#[test]
fn missing_gradients_produce_only_positive_zero_without_tensor_inputs() {
    let named = vec![named_gradient(&[MODEL_PARAMETER_COUNT], None)];

    let packed = collect_packed_gradients(&named).expect("missing gradients");
    let direct = collect_host_gradients(clone_named(&named)).expect("direct gradients");

    assert_eq!(packed.len(), MODEL_PARAMETER_COUNT);
    assert!(packed.iter().all(|value| value.to_bits() == 0));
    assert_bits(&direct, &packed);
}

#[test]
fn gradient_shape_and_total_errors_precede_nonfinite_values() {
    assert_gradient_error_parity(&Device::Cpu);
}

#[test]
fn gradient_shape_and_dtype_errors_precede_an_oversized_total() {
    assert_capacity_error_order(&Device::Cpu);
}

#[test]
fn gradient_collection_rejects_shape_overflow_and_capacity_before_unbounded_allocation() {
    for (shape, message) in [
        (vec![usize::MAX, 2], "model produced invalid gradient shape"),
        (
            vec![MODEL_PARAMETER_COUNT + 1],
            "model produced invalid gradient parameter count",
        ),
    ] {
        let named = vec![named_gradient(&shape, None)];

        let direct = collect_host_gradients(clone_named(&named)).expect_err("bounded direct");
        let packed = collect_packed_gradients(&named).expect_err("bounded packed");

        assert_eq!(direct.to_string(), message);
        assert_eq!(packed.to_string(), message);
    }
    let named = vec![
        named_gradient(&[MODEL_PARAMETER_COUNT], None),
        named_gradient(&[1], None),
    ];
    assert_eq!(
        collect_packed_gradients(&named)
            .expect_err("sum bound")
            .to_string(),
        "model produced invalid gradient parameter count"
    );
    assert_eq!(
        collect_host_gradients(named)
            .expect_err("sum bound")
            .to_string(),
        "model produced invalid gradient parameter count"
    );
}

#[test]
fn gradient_collection_bounds_descriptor_count() {
    let named = (0..63)
        .map(|_| named_gradient(&[1], None))
        .collect::<Vec<_>>();

    let direct = collect_host_gradients(clone_named(&named)).expect_err("descriptor bound");
    let packed = collect_packed_gradients(&named).expect_err("descriptor bound");

    assert_eq!(
        direct.to_string(),
        "model produced invalid gradient parameter count"
    );
    assert_eq!(packed.to_string(), direct.to_string());
}

#[test]
fn oversized_present_gradient_reports_total_before_nonfinite_values() {
    let tensor = Tensor::from_vec(
        vec![f32::NAN; MODEL_PARAMETER_COUNT + 1],
        MODEL_PARAMETER_COUNT + 1,
        &Device::Cpu,
    )
    .expect("oversized gradient");
    let named = vec![named_gradient(&[MODEL_PARAMETER_COUNT + 1], Some(tensor))];
    let expected = legacy_collect_gradients(&named).expect_err("legacy total error");

    let packed = collect_packed_gradients(&named).expect_err("bounded packed total");
    let direct = collect_host_gradients(named).expect_err("bounded direct total");

    assert_eq!(
        expected.to_string(),
        "model produced invalid gradient parameter count"
    );
    assert_eq!(packed.to_string(), expected.to_string());
    assert_eq!(direct.to_string(), expected.to_string());
}

#[test]
fn oversized_gradient_preserves_dtype_error_before_shape_and_total() {
    let tensor = Tensor::from_vec(
        vec![1u32; MODEL_PARAMETER_COUNT + 1],
        MODEL_PARAMETER_COUNT + 1,
        &Device::Cpu,
    )
    .expect("oversized integer gradient");
    for count in [1, MODEL_PARAMETER_COUNT + 1] {
        let named = vec![named_gradient(&[count], Some(tensor.clone()))];
        let expected = legacy_collect_gradients(&named).expect_err("legacy dtype error");

        let actual = collect_host_gradients(named).expect_err("bounded dtype error");

        assert_dtype_error(expected);
        assert_dtype_error(actual);
    }
}

#[test]
fn direct_gradient_dtype_error_precedes_a_later_shape_error() {
    assert_dtype_error_order(&Device::Cpu);
}

#[test]
fn repeated_ppo_updates_match_duplicate_export_reference_bitwise() {
    assert_repeated_ppo_parity(PolicyDevice::Cpu);
}

#[test]
fn ppo_snapshot_reuse_restores_exact_state_after_candidate_rejection() {
    assert_candidate_rollback_parity(PolicyDevice::Cpu, false);
}

#[test]
fn ppo_snapshot_reuse_restores_exact_state_after_candidate_evaluation_error() {
    assert_candidate_rollback_parity(PolicyDevice::Cpu, true);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_packed_gradients_match_legacy_bits_views_and_error_order() {
    let device = Device::new_cuda(0).expect("CUDA device");
    assert_gradient_parity(&device);
    assert_contiguous_gradient_parity(&device);
    assert_single_gradient_parity(&device);
    assert_scalar_gradient_parity(&device);
    assert_zero_sized_gradient_parity(&device);
    assert_gradient_error_parity(&device);
    assert_capacity_error_order(&device);
    assert_dtype_error_order(&device);
    assert_mixed_device_gradient_parity(&device);
    assert_backward_gradient_parity(PolicyDevice::Cuda { ordinal: 0 });
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_ppo_snapshot_reuse_matches_repeated_updates_and_exact_rollback() {
    let device = PolicyDevice::Cuda { ordinal: 0 };
    assert_repeated_ppo_parity(device);
    assert_candidate_rollback_parity(device, false);
    assert_candidate_rollback_parity(device, true);
}

fn assert_gradient_parity(device: &Device) {
    let views = gradient_views(device);
    let mut named = vec![named_gradient(&[2], None)];
    for tensor in views {
        named.push(named_gradient(&[tensor.elem_count()], Some(tensor)));
        named.push(named_gradient(&[1], None));
    }
    pad_missing_gradients(&mut named);
    let expected = legacy_collect_gradients(&named).expect("legacy gradients");

    let packed = collect_packed_gradients(&named).expect("packed gradients");
    let selected = collect_host_gradients(clone_named(&named)).expect("selected path");

    assert_bits(&packed, &expected);
    assert_bits(&selected, &expected);
    assert_eq!(packed[2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(packed[3].to_bits(), 1);
    assert_eq!(packed[0].to_bits(), 0);
}

fn assert_single_gradient_parity(device: &Device) {
    for tensor in gradient_views(device) {
        let mut named = vec![
            named_gradient(&[1], None),
            named_gradient(&[tensor.elem_count()], Some(tensor)),
        ];
        pad_missing_gradients(&mut named);
        let expected = legacy_collect_gradients(&named).expect("legacy singleton");

        let actual = collect_packed_gradients(&named).expect("packed singleton");
        let selected = collect_host_gradients(clone_named(&named)).expect("selected singleton");

        assert_bits(&actual, &expected);
        assert_bits(&selected, &expected);
    }
}

fn assert_scalar_gradient_parity(device: &Device) {
    let scalar = Tensor::new(-0.0f32, device).expect("scalar gradient");
    let empty = scalar
        .reshape(1)
        .expect("scalar vector")
        .narrow(0, 0, 0)
        .expect("empty gradient view");
    let second = Tensor::new(f32::from_bits(1), device).expect("second scalar gradient");
    for second in [None, Some(second)] {
        let mut named = vec![
            named_gradient(&[0, 2], Some(empty.clone())),
            named_gradient(&[], Some(scalar.clone())),
            named_gradient(&[], None),
            named_gradient(&[0], None),
            named_gradient(&[], second),
        ];
        pad_missing_gradients(&mut named);
        let expected = legacy_collect_gradients(&named).expect("legacy scalar gradients");

        let packed = collect_packed_gradients(&named).expect("packed scalar gradients");
        let selected = collect_host_gradients(named).expect("selected scalar gradients");

        assert_bits(&packed, &expected);
        assert_bits(&selected, &expected);
        assert_eq!(packed[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(packed[1].to_bits(), 0);
    }
}

fn assert_zero_sized_gradient_parity(device: &Device) {
    let source = Tensor::from_vec(vec![f32::NAN], 1, device).expect("empty view source");
    let empty = source.narrow(0, 0, 0).expect("empty gradient view");
    let named = vec![
        named_gradient(&[0], Some(empty.clone())),
        named_gradient(&[MODEL_PARAMETER_COUNT], None),
        named_gradient(&[2, 0], Some(empty)),
    ];
    let expected = legacy_collect_gradients(&named).expect("legacy zero-sized gradients");

    let packed = collect_packed_gradients(&named).expect("packed zero-sized gradients");
    let selected = collect_host_gradients(named).expect("selected zero-sized gradients");

    assert_bits(&packed, &expected);
    assert_bits(&selected, &expected);
    assert!(packed.iter().all(|value| value.to_bits() == 0));
}

fn assert_contiguous_gradient_parity(device: &Device) {
    let [offset, prefix, _, _] = gradient_views(device);
    assert!(offset.is_contiguous());
    assert!(prefix.is_contiguous());
    let mut named = vec![
        named_gradient(&[2, 2], Some(offset)),
        named_gradient(&[1], None),
        named_gradient(&[2], Some(prefix)),
    ];
    pad_missing_gradients(&mut named);
    let expected = legacy_collect_gradients(&named).expect("legacy contiguous");

    let actual = collect_packed_gradients(&named).expect("packed contiguous");
    let selected = collect_host_gradients(clone_named(&named)).expect("selected contiguous");

    assert_bits(&actual, &expected);
    assert_bits(&selected, &expected);
}

fn assert_backward_gradient_parity(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9_102, device).expect("model");
    let frames = [FeatureFrame::new()];
    let prefixes = [TrainingPrefix::new(ActionKind::Continue, None, None)];
    let output = model.training_forward(&frames, &prefixes).expect("forward");
    let loss = output.sum_all_heads().expect("loss");
    let named = model
        .backward_named(&output, &loss)
        .expect("single backward");
    assert_eq!(named.len(), 62);
    assert!(named.iter().any(|gradient| gradient.gradient.is_some()));
    let expected = legacy_collect_gradients(&named).expect("legacy backward readback");

    let packed = collect_packed_gradients(&named).expect("packed backward readback");
    let selected = collect_host_gradients(named).expect("selected backward readback");

    assert_bits(&packed, &expected);
    assert_bits(&selected, &expected);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn assert_mixed_device_gradient_parity(device: &Device) {
    let [offset, _, strided, _] = gradient_views(device);
    let empty = offset.narrow(0, 0, 0).expect("empty device view");
    let host = Tensor::from_vec(vec![-0.0f32], 1, &Device::Cpu).expect("host gradient");
    let mut named = vec![
        named_gradient(&[4], Some(offset)),
        named_gradient(&[1], Some(host)),
        named_gradient(&[3], Some(strided)),
        named_gradient(&[0], Some(empty)),
    ];
    pad_missing_gradients(&mut named);
    let expected = legacy_collect_gradients(&named).expect("legacy mixed devices");

    let actual = collect_host_gradients(named).expect("mixed-device fallback");

    assert_bits(&actual, &expected);
}

fn gradient_views(device: &Device) -> [Tensor; 4] {
    let source = Tensor::from_vec(
        vec![19.0f32, -0.0, f32::from_bits(1), 2.0, -3.0, 23.0],
        6,
        device,
    )
    .expect("offset source");
    let matrix = Tensor::from_vec(
        vec![0.0f32, -0.0, 2.0, f32::from_bits(0x8000_0001), 4.0, 5.0],
        (3, 2),
        device,
    )
    .expect("strided source");
    let strided = matrix
        .narrow(1, 1, 1)
        .expect("column")
        .squeeze(1)
        .expect("vector");
    assert!(!strided.is_contiguous());
    let transposed = matrix.t().expect("transpose");
    assert!(!transposed.is_contiguous());
    [
        source.narrow(0, 1, 4).expect("offset view"),
        source
            .narrow(0, 0, 2)
            .expect("prefix with unused backing tail"),
        strided,
        transposed,
    ]
}

fn assert_gradient_error_parity(device: &Device) {
    let nonfinite =
        Tensor::from_vec(vec![f32::NAN, f32::INFINITY], 2, device).expect("nonfinite tensor");
    let finite = Tensor::from_vec(vec![1.0f32], 1, device).expect("finite tensor");
    let mut cases = vec![
        (
            vec![
                named_gradient(&[2], Some(nonfinite.clone())),
                named_gradient(&[2], Some(finite)),
            ],
            "model produced invalid gradient shape",
        ),
        (
            vec![named_gradient(&[2], Some(nonfinite.clone()))],
            "model produced invalid gradient parameter count",
        ),
        (
            vec![
                named_gradient(&[5], None),
                named_gradient(&[2], Some(nonfinite)),
            ],
            "model gradient 5 is non-finite",
        ),
    ];
    pad_missing_gradients(&mut cases[0].0);
    pad_missing_gradients(&mut cases[2].0);
    for (named, message) in cases {
        let expected = legacy_collect_gradients(&named).expect_err("legacy error");

        let packed = collect_packed_gradients(&named).expect_err("packed error");
        let selected = collect_host_gradients(clone_named(&named)).expect_err("selected error");

        assert_eq!(expected.to_string(), message);
        assert_eq!(packed.to_string(), message);
        assert_eq!(selected.to_string(), message);
    }
}

fn assert_capacity_error_order(device: &Device) {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0], 2, device).expect("shape mismatch");
    for extra_missing in [false, true] {
        let mut named = vec![named_gradient(&[MODEL_PARAMETER_COUNT], None)];
        if extra_missing {
            named.push(named_gradient(&[1], None));
        }
        named.push(named_gradient(&[1], Some(tensor.clone())));
        let expected = legacy_collect_gradients(&named).expect_err("legacy shape error");

        let packed = collect_packed_gradients(&named).expect_err("shape before count");
        let selected = collect_host_gradients(named).expect_err("shape before count");

        assert_eq!(
            expected.to_string(),
            "model produced invalid gradient shape"
        );
        assert_eq!(packed.to_string(), expected.to_string());
        assert_eq!(selected.to_string(), expected.to_string());
    }
    let tensor = Tensor::from_vec(vec![1u32], 1, device).expect("integer gradient");
    let named = vec![
        named_gradient(&[MODEL_PARAMETER_COUNT], None),
        named_gradient(&[1], Some(tensor)),
    ];
    let expected = legacy_collect_gradients(&named).expect_err("legacy dtype error");
    let actual = collect_host_gradients(named).expect_err("dtype before count");
    assert_dtype_error(expected);
    assert_dtype_error(actual);
}

fn assert_dtype_error_order(device: &Device) {
    let tensor = Tensor::from_vec(vec![1u32], 1, device).expect("integer tensor");
    let empty = tensor.narrow(0, 0, 0).expect("empty integer view");
    for first in [tensor.clone(), empty] {
        let named = vec![
            named_gradient(&[1], Some(first)),
            named_gradient(&[2], Some(tensor.clone())),
        ];
        let expected = legacy_collect_gradients(&named).expect_err("legacy dtype error");

        let actual = collect_host_gradients(named).expect_err("same dtype error");

        assert_dtype_error(expected);
        assert_dtype_error(actual);
    }
}

fn assert_dtype_error(error: ModelError) {
    assert!(matches!(&error, ModelError::Backend(_)));
    // Candle appends environment-dependent backtraces after the diagnostic line.
    assert_eq!(
        error.to_string().lines().next(),
        Some("model tensor operation failed: unexpected dtype, expected: F32, got: U32")
    );
}

fn named_gradient(shape: &[usize], tensor: Option<Tensor>) -> NamedPolicyGradient {
    assert!(shape.len() <= 2);
    NamedPolicyGradient {
        name: "transfer fixture",
        parameter_shape: shape.to_vec(),
        gradient: tensor,
    }
}

fn pad_missing_gradients(named: &mut Vec<NamedPolicyGradient>) {
    assert!(named.len() < 62);
    let total = named
        .iter()
        .map(|gradient| gradient.parameter_shape.iter().product::<usize>())
        .sum::<usize>();
    assert!(total < MODEL_PARAMETER_COUNT);
    named.push(named_gradient(&[MODEL_PARAMETER_COUNT - total], None));
}

fn clone_named(named: &[NamedPolicyGradient]) -> Vec<NamedPolicyGradient> {
    assert!(named.len() <= 63);
    named
        .iter()
        .map(|gradient| NamedPolicyGradient {
            name: gradient.name,
            parameter_shape: gradient.parameter_shape.clone(),
            gradient: gradient.gradient.clone(),
        })
        .collect()
}

// Keep the old readback and validation order as the differential oracle.
fn legacy_collect_gradients(named: &[NamedPolicyGradient]) -> Result<Vec<f32>, ModelError> {
    assert!(named.len() <= 62);
    let mut output = Vec::with_capacity(MODEL_PARAMETER_COUNT);
    for gradient in named {
        let count = gradient.parameter_shape.iter().product::<usize>();
        if let Some(tensor) = &gradient.gradient {
            let values = tensor.flatten_all()?.to_vec1::<f32>()?;
            if values.len() != count {
                return Err(ModelError::InvalidModelState("gradient shape"));
            }
            assert!(count <= MODEL_PARAMETER_COUNT + 1 - output.len());
            output.extend(values);
        } else {
            assert!(count <= MODEL_PARAMETER_COUNT + 1 - output.len());
            output.resize(output.len() + count, 0.0);
        }
    }
    if output.len() != MODEL_PARAMETER_COUNT {
        return Err(ModelError::InvalidModelState("gradient parameter count"));
    }
    validate_gradients(&output)?;
    Ok(output)
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "float bits at {index}"
        );
    }
}

fn assert_repeated_ppo_parity(device: PolicyDevice) {
    let actual = PolicyModel::fresh_on(9_101, device).expect("actual model");
    let reference = PolicyModel::fresh_on(9_101, device).expect("reference model");
    let config = transfer_ppo_config();
    let mut actual_adam = actual.claim_optimizer(config.adam()).expect("actual Adam");
    let mut reference_adam = reference
        .claim_optimizer(config.adam())
        .expect("reference Adam");
    let mut samples = transfer_ppo_samples(&actual);
    for step in 1..=3 {
        refresh_old_probabilities(&actual, &mut samples);
        let examples = samples.iter().collect::<Vec<_>>();

        let actual_report = actual
            .ppo_update_with_microbatch(
                &examples,
                &mut actual_adam,
                config,
                2,
                PpoTestFaults::default(),
            )
            .expect("actual PPO update");
        let expected_report = legacy_ppo_update(
            &reference,
            &examples,
            &mut reference_adam,
            config,
            PpoTestFaults::default(),
        )
        .expect("reference PPO update");

        assert!(actual_report.applied);
        assert_report_bits(actual_report, expected_report);
        assert_state_bits(&actual, &actual_adam, &reference, &reference_adam);
        assert_eq!(actual_adam.step(), step);
        assert_eq!(actual.policy_identity().expect("identity").revision(), step);
    }
    assert!(actual_adam.moments().0.iter().any(|value| *value != 0.0));
}

fn assert_candidate_rollback_parity(device: PolicyDevice, evaluation_error: bool) {
    let actual = PolicyModel::fresh_on(9_101, device).expect("actual model");
    let reference = PolicyModel::fresh_on(9_101, device).expect("reference model");
    let config = transfer_ppo_config();
    let mut actual_adam = actual.claim_optimizer(config.adam()).expect("actual Adam");
    let mut reference_adam = reference
        .claim_optimizer(config.adam())
        .expect("reference Adam");
    let mut samples = transfer_ppo_samples(&actual);
    warm_ppo_optimizers(
        &actual,
        &mut actual_adam,
        &reference,
        &mut reference_adam,
        config,
        &mut samples,
    );
    refresh_old_probabilities(&actual, &mut samples);
    let before = actual
        .coherent_snapshot(&actual_adam)
        .expect("before rejection");
    let reference_binding = reference_adam.binding;
    assert!(before.adam.first_moment.iter().any(|value| *value != 0.0));
    let config = PpoConfig {
        target_kl: 1.0e-12,
        ..config
    };
    let faults = PpoTestFaults {
        candidate_evaluation: evaluation_error,
        rollback_import: false,
    };
    let examples = samples.iter().collect::<Vec<_>>();

    let result = actual.ppo_update_with_microbatch(&examples, &mut actual_adam, config, 2, faults);
    let expected = legacy_ppo_update(&reference, &examples, &mut reference_adam, config, faults);

    if evaluation_error {
        let message = concat!(
            "model tensor operation failed: injected PPO candidate evaluation failure ",
            "after 2 rows"
        );
        assert_eq!(result.expect_err("candidate error").to_string(), message);
        assert_eq!(expected.expect_err("reference error").to_string(), message);
    } else {
        let report = result.expect("rejected candidate");
        assert!(!report.applied);
        assert!(report.approximate_kl > f64::from(config.target_kl));
        assert_report_bits(report, expected.expect("reference rejection"));
    }
    let after = actual
        .coherent_snapshot(&actual_adam)
        .expect("after rejection");
    assert_snapshot_bits(&after, &before);
    assert_eq!(after.adam.binding, before.adam.binding);
    assert_eq!(reference_adam.binding, reference_binding);
    assert_state_bits(&actual, &actual_adam, &reference, &reference_adam);
}

fn warm_ppo_optimizers(
    actual: &PolicyModel,
    actual_adam: &mut AdamState,
    reference: &PolicyModel,
    reference_adam: &mut AdamState,
    config: PpoConfig,
    samples: &mut [PpoPreparedSample],
) {
    refresh_old_probabilities(actual, samples);
    let examples = samples.iter().collect::<Vec<_>>();
    let actual_report = actual
        .ppo_update_with_microbatch(&examples, actual_adam, config, 2, PpoTestFaults::default())
        .expect("warm actual moments");
    let reference_report = legacy_ppo_update(
        reference,
        &examples,
        reference_adam,
        config,
        PpoTestFaults::default(),
    )
    .expect("warm reference moments");
    assert!(actual_report.applied);
    assert!(reference_report.applied);
}

fn transfer_ppo_config() -> PpoConfig {
    PpoConfig {
        learning_rate: 0.01,
        target_kl: 1.0e3,
        ..PpoConfig::default()
    }
}

fn transfer_ppo_samples(model: &PolicyModel) -> Vec<PpoPreparedSample> {
    let tracker = transfer_tracker();
    let space = ActionSpace::from_tracker(&tracker).expect("action space");
    let mut encoder = crate::FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            &tracker,
            &space,
            &crate::ItemReadiness::new(),
            &crate::LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("frame");
    let mut target = BehavioralTarget::from_action(&frame, &space, StructuredAction::Continue)
        .expect("continue target");
    // A synthetic two-choice learner fixture avoids simulator work and RNG consumption.
    target.kind.mask[ActionKind::Stop.index()] = true;
    target.validate().expect("synthetic target");
    (0..3)
        .map(|stream| PpoPreparedSample {
            transition: crate::PpoTransition {
                frame: frame.clone(),
                target: target.clone(),
                action: StructuredAction::Continue,
                policy: model.policy_identity().expect("policy"),
                stream,
                decision: 0,
                ticks: 3,
                old_log_probability: 0.0,
                old_value: 0.0,
                next_value: 0.0,
                reward: 0.0,
                terminal: true,
            },
            advantage: 1.0,
            return_value: 1.0,
        })
        .collect()
}

fn transfer_tracker() -> crate::StateTracker {
    use bota_proto::{MapId, MatchInfo, Pick, SlotId, Team, TickMode};

    let info = MatchInfo {
        match_id: 1,
        map: MapId(0),
        tick_rate: 30,
        pregame_ticks: 0,
        trees: Vec::new(),
        terrain_cells: 1,
        terrain_rle: vec![(1, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![Pick {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: crate::SHADOW_FIEND,
        }],
        shop: Vec::new(),
    };
    let mut tracker = crate::StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker
        .observe_snapshot(&transfer_world_view())
        .expect("snapshot");
    assert!(tracker.own_hero().is_none());
    assert!(tracker.own_player().is_some());
    tracker
}

fn transfer_world_view() -> bota_proto::WorldView {
    use bota_proto::{PlayerView, SlotId, Team, WorldView};

    WorldView {
        tick: 1,
        viewer: Some(Team::Radiant),
        units: Vec::new(),
        projectiles: Vec::new(),
        players: vec![PlayerView {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: crate::SHADOW_FIEND,
            unit: None,
            level: 1,
            xp: 0,
            gold: Some(0),
            stash: Some(vec![None; 6]),
            kit: None,
            kills: 0,
            deaths: 0,
            assists: 0,
            last_hits: 0,
            denies: 0,
            respawn_left: 1,
        }],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn refresh_old_probabilities(model: &PolicyModel, samples: &mut [PpoPreparedSample]) {
    assert_eq!(samples.len(), 3);
    // Use the same 2+1 partition as the update, not a differently shaped GEMM.
    for chunk in samples.chunks_mut(2) {
        let references = chunk.iter().collect::<Vec<_>>();
        let (probabilities, _) = model
            .ppo_likelihood_for_test(&references)
            .expect("likelihood");
        assert_eq!(probabilities.len(), chunk.len());
        for (sample, probability) in chunk.iter_mut().zip(probabilities) {
            sample.transition.old_log_probability = probability;
        }
    }
}

fn assert_state_bits(
    actual: &PolicyModel,
    actual_adam: &AdamState,
    reference: &PolicyModel,
    reference_adam: &AdamState,
) {
    let actual = actual
        .coherent_snapshot(actual_adam)
        .expect("actual snapshot");
    let reference = reference
        .coherent_snapshot(reference_adam)
        .expect("reference snapshot");
    assert_snapshot_bits(&actual, &reference);
    assert_eq!(
        actual.adam.binding.policy.revision,
        reference.adam.binding.policy.revision
    );
}

fn assert_snapshot_bits(actual: &ModelAdamSnapshot, expected: &ModelAdamSnapshot) {
    assert_bits(&actual.parameters, &expected.parameters);
    assert_bits(&actual.adam.first_moment, &expected.adam.first_moment);
    assert_bits(&actual.adam.second_moment, &expected.adam.second_moment);
    assert_eq!(actual.adam.step, expected.adam.step);
    assert_eq!(actual.adam.config, expected.adam.config);
}

fn assert_report_bits(actual: PpoMinibatchReport, expected: PpoMinibatchReport) {
    for (actual, expected) in [
        (actual.policy_loss, expected.policy_loss),
        (actual.value_loss, expected.value_loss),
        (actual.entropy, expected.entropy),
        (actual.approximate_kl, expected.approximate_kl),
        (actual.clip_fraction, expected.clip_fraction),
        (actual.gradient_norm, expected.gradient_norm),
        (actual.applied_scale, expected.applied_scale),
    ] {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(actual.samples, expected.samples);
    assert_eq!(actual.applied, expected.applied);
}

// Preserve the original candidate transaction, including its second parameter export.
fn legacy_ppo_update(
    model: &PolicyModel,
    examples: &[&PpoPreparedSample],
    adam: &mut AdamState,
    config: PpoConfig,
    faults: PpoTestFaults,
) -> Result<PpoMinibatchReport, ModelError> {
    assert_eq!(examples.len(), 3);
    let _guard = model.write_parameter_lock()?;
    model.validate_optimizer_binding_locked(adam.binding)?;
    let mut gradients = vec![0.0f32; MODEL_PARAMETER_COUNT];
    let mut report = PpoMinibatchReport::default();
    for microbatch in examples.chunks(2) {
        let mut result = model.ppo_microbatch_locked(microbatch, config)?;
        scale_gradients(&mut result.gradients, microbatch.len() as f32)?;
        accumulate_gradients(&mut gradients, &result.gradients)?;
        accumulate_ppo_report(&mut report, result.report)?;
    }
    let divisor = examples.len() as f32;
    for gradient in &mut gradients {
        *gradient /= divisor;
    }
    average_ppo_report(&mut report, examples.len())?;
    if report.approximate_kl > f64::from(config.target_kl) {
        return Ok(report);
    }
    let original = model.export_parameters_locked()?;
    let original_adam = adam.clone();
    let diagnostics = legacy_apply_adam(model, adam, &gradients)?;
    let candidate_kl =
        model.ppo_candidate_kl_locked(examples, config, 2, faults.candidate_evaluation);
    if !candidate_kl
        .as_ref()
        .is_ok_and(|kl| *kl <= f64::from(config.target_kl))
    {
        model.rollback_ppo_candidate_locked(
            &original,
            original_adam,
            adam,
            &candidate_kl,
            faults.rollback_import,
        )?;
        report.approximate_kl = candidate_kl?;
        return Ok(report);
    }
    report.approximate_kl = candidate_kl?;
    report.gradient_norm = diagnostics.unclipped_norm;
    report.applied_scale = diagnostics.applied_scale;
    report.applied = true;
    Ok(report)
}

fn legacy_apply_adam(
    model: &PolicyModel,
    adam: &mut AdamState,
    gradients: &[f32],
) -> Result<AdamDiagnostics, ModelError> {
    let next = model.next_policy_identity_locked()?;
    let parameters = model.export_parameters_locked()?;
    let replacement = compute_adam_step(
        &parameters,
        gradients,
        &adam.first_moment,
        &adam.second_moment,
        adam.step,
        adam.config,
    )?;
    model.import_parameters_locked(&replacement.parameters, None)?;
    adam.first_moment = replacement.first_moment;
    adam.second_moment = replacement.second_moment;
    adam.step = replacement.step;
    adam.binding.policy = next;
    model
        .parameter_revision
        .store(next.revision, Ordering::Relaxed);
    Ok(AdamDiagnostics {
        unclipped_norm: replacement.unclipped_norm,
        applied_scale: replacement.applied_scale,
    })
}
