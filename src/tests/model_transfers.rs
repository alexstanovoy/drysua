use super::*;

#[path = "candidate_kl.rs"]
pub(super) mod candidate_kl_tests;

#[path = "model_concurrency.rs"]
mod concurrency_tests;
#[path = "training_microbatch.rs"]
mod microbatch_tests;

#[test]
fn gradient_readback_preserves_view_bits_and_rejects_unbounded_descriptors() {
    let source = Tensor::from_vec(
        vec![19.0f32, -0.0, 2.0, f32::from_bits(1)],
        (2, 2),
        &Device::Cpu,
    )
    .expect("source");
    let view = source.narrow(1, 1, 1).expect("strided view");
    let named = vec![
        NamedPolicyGradient {
            name: "view",
            parameter_shape: vec![2],
            gradient: Some(view),
        },
        NamedPolicyGradient {
            name: "missing",
            parameter_shape: vec![MODEL_PARAMETER_COUNT - 2],
            gradient: None,
        },
    ];
    let mut expected = vec![0.0; MODEL_PARAMETER_COUNT];
    expected[0] = -0.0;
    expected[1] = f32::from_bits(1);
    assert_bits(
        &collect_packed_gradients(&named).expect("packed"),
        &expected,
    );
    assert_bits(&collect_host_gradients(named).expect("selected"), &expected);
    for (shape, message) in [
        (vec![usize::MAX, 2], "model produced invalid gradient shape"),
        (
            vec![MODEL_PARAMETER_COUNT + 1],
            "model produced invalid gradient parameter count",
        ),
    ] {
        let named = vec![NamedPolicyGradient {
            name: "overflow",
            parameter_shape: shape,
            gradient: None,
        }];
        assert_eq!(
            collect_packed_gradients(&named)
                .expect_err("packed bound")
                .to_string(),
            message
        );
        assert_eq!(
            collect_host_gradients(named)
                .expect_err("host bound")
                .to_string(),
            message
        );
    }
}

#[test]
fn real_backward_readback_matches_tensor_values_bitwise() {
    assert_backward_readback(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires an authorized CUDA runner"]
fn cuda_real_backward_readback_matches_tensor_values_bitwise() {
    assert_backward_readback(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_backward_readback(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9_102, device).expect("model");
    let output = model
        .training_forward(
            &[test_frame()],
            &[TrainingPrefix::new(ActionKind::Continue, None, None)],
        )
        .expect("forward");
    let loss = output.sum_all_heads().expect("loss");
    let named = model.backward_named(&output, &loss).expect("backward");
    let mut expected = Vec::with_capacity(MODEL_PARAMETER_COUNT);
    for gradient in &named {
        let count = gradient.parameter_shape.iter().product::<usize>();
        match &gradient.gradient {
            Some(tensor) => expected.extend(
                tensor
                    .flatten_all()
                    .expect("flat")
                    .to_vec1::<f32>()
                    .expect("values"),
            ),
            None => expected.resize(expected.len() + count, 0.0),
        }
    }
    assert_eq!(expected.len(), MODEL_PARAMETER_COUNT);
    assert_bits(
        &collect_packed_gradients(&named).expect("packed"),
        &expected,
    );
    assert_bits(&collect_host_gradients(named).expect("selected"), &expected);
}

fn test_frame() -> FeatureFrame {
    let mut frame = FeatureFrame::new();
    frame.global[crate::global_feature::SIDE_RADIANT] = 1.0;
    frame
}

fn transfer_ppo_config() -> PpoConfig {
    PpoConfig {
        learning_rate: 0.01,
        target_kl: 1.0e3,
        ..PpoConfig::default()
    }
}

fn transfer_ppo_samples() -> Vec<PpoPreparedSample> {
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
    // A two-choice learner fixture keeps folding parity independent of simulator and RNG work.
    target.kind.mask[ActionKind::Stop.index()] = true;
    target.validate().expect("synthetic target");
    (0..3)
        .map(|stream| PpoPreparedSample {
            transition: crate::PpoTransition {
                frame: frame.clone(),
                target: target.clone(),
                action: StructuredAction::Continue,
                behaviour: 0,
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
    // Preserve the update's 2+1 GEMM partition rather than introducing rounding differences.
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
