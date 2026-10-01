use super::*;

#[test]
fn dire_actor_heads_mirror_radiant_heads_after_the_shared_parameters() {
    let model = PolicyModel::fresh(9001).expect("model");
    let schema = model.parameter_schema().expect("schema");
    assert_eq!(schema.len(), 88);
    let heads = [
        "kind",
        "controlled",
        "ability_head",
        "item_head",
        "swap_head",
        "learn_head",
        "shop_head",
        "loot_head",
        "target_mode",
        "put_mode",
        "entity_query",
        "point_query",
    ];
    for (index, original) in (36..38).chain(42..64).enumerate() {
        let suffix = if index % 2 == 0 { "weight" } else { "bias" };
        assert_eq!(schema[original].0, format!("{}.{suffix}", heads[index / 2]));
        assert_eq!(
            schema[64 + index].0,
            format!("dire.{}.{suffix}", heads[index / 2])
        );
        assert_eq!(schema[64 + index].1, schema[original].1);
    }
    assert_eq!(schema[32].0, "value.0.weight");
    assert_eq!(schema[34].0, "value.1.weight");
    for (entry, name) in schema[38..42].iter().zip([
        "kind_embedding.weight",
        "unit_embedding.weight",
        "ability_embedding.weight",
        "item_embedding.weight",
    ]) {
        assert_eq!(entry.0, name);
    }
}

#[test]
fn side_validation_accepts_signed_zero_and_prioritizes_nonfinite_frames() {
    let model = PolicyModel::fresh(9001).expect("model");
    let prefix = [TrainingPrefix::new(ActionKind::Continue, None, None)];
    for (radiant, dire) in [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (-1.0, 2.0),
        (1.0, f32::from_bits(1)),
    ] {
        let mut frame = side_frame(false);
        frame.global[4] = radiant;
        frame.global[5] = dire;
        take_encoder_forwards_for_test();
        let error = model
            .training_forward(&[frame], &prefix)
            .expect_err("invalid side");
        assert_eq!(
            error,
            ModelError::InvalidSideOneHot {
                index: 0,
                radiant_bits: radiant.to_bits(),
                dire_bits: dire.to_bits(),
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "model frame 0 has invalid side one-hot: radiant={radiant}, dire={dire}; expected (1,0) or (0,1)"
            )
        );
        assert_eq!(take_encoder_forwards_for_test(), 0);
    }
    for dire in [false, true] {
        let mut frame = side_frame(dire);
        frame.global[if dire { 4 } else { 5 }] = -0.0;
        model
            .training_forward(&[frame], &prefix)
            .expect("signed zero side");
    }
    let mut frames = [FeatureFrame::new(), side_frame(true)];
    frames[1].global[0] = f32::NAN;
    assert_eq!(
        model
            .training_forward(&frames, &[prefix[0]; 2])
            .expect_err("finite first"),
        ModelError::NonFiniteFrame { index: 1 }
    );
}

#[test]
fn selected_actor_gradients_reach_shared_trunk_but_not_opposite_actor_or_value() {
    assert_gradients(PolicyDevice::Cpu);
}

#[test]
fn mixed_actor_gradients_match_full_batch_masked_heads_and_pointer_references() {
    assert_mixed_gradients(PolicyDevice::Cpu);
}

#[test]
fn training_rejects_overflow_in_unselected_dire_heads_before_backward() {
    assert_unselected_head_overflow(PolicyDevice::Cpu);
}

#[cfg(feature = "builtin")]
#[test]
fn native_mixed_sides_route_independent_heads_in_one_encoder_pass() {
    assert_routing(PolicyDevice::Cpu);
}

#[cfg(feature = "builtin")]
#[test]
fn inference_skips_unused_item_family_but_rejects_traversed_controlled_overflow() {
    assert_unused_family_skipping(PolicyDevice::Cpu);
}

#[cfg(feature = "builtin")]
#[test]
fn sampling_rejects_nonfinite_shared_value_before_consuming_rng() {
    assert_value_rejection(PolicyDevice::Cpu);
}

#[cfg(feature = "builtin")]
#[test]
fn mixed_side_ppo_accepts_then_restores_both_head_moments_on_rejection_and_error() {
    assert_ppo_rollback(PolicyDevice::Cpu);
}

#[cfg(all(
    feature = "builtin",
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
#[ignore = "needs a CUDA device"]
fn cuda_side_actors_preserve_routing_gradients_and_ppo_rollback() {
    let device = PolicyDevice::Cuda { ordinal: 0 };
    assert_gradients(device);
    assert_mixed_gradients(device);
    assert_unselected_head_overflow(device);
    assert_routing(device);
    assert_unused_family_skipping(device);
    assert_value_rejection(device);
    assert_ppo_rollback(device);
}

fn side_frame(dire: bool) -> FeatureFrame {
    let mut frame = FeatureFrame::new();
    frame.global[4] = f32::from(!dire);
    frame.global[5] = f32::from(dire);
    frame
}

fn edit(model: &PolicyModel, mut update: impl FnMut(&str, &mut [f32])) {
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let end = offset + shape.iter().product::<usize>();
        update(name, &mut parameters[offset..end]);
        offset = end;
    }
    assert_eq!(offset, MODEL_PARAMETER_COUNT);
    model
        .import_parameters(&parameters)
        .expect("finite in-place import");
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits()))
    );
}

fn gradient_values(named: &[NamedPolicyGradient], name: &str) -> Vec<f32> {
    assert_eq!(named.len(), 88);
    let gradient = named
        .iter()
        .find(|entry| entry.name == name)
        .expect("named gradient");
    match &gradient.gradient {
        Some(tensor) => tensor
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("gradient values"),
        None => vec![0.0; gradient.parameter_shape.iter().product()],
    }
}

fn assert_gradients(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9001, device).expect("model");
    edit(&model, |name, values| match name {
        "trunk.2.weight" => values.fill(0.0),
        "trunk.2.bias" => values.fill(2.0),
        "kind.weight" | "dire.kind.weight" => values.fill(0.25),
        _ => {}
    });
    for dire in [false, true] {
        let output = model
            .training_forward(
                &[side_frame(dire)],
                &[TrainingPrefix::new(ActionKind::Continue, None, None)],
            )
            .expect("forward");
        let named = model
            .backward_named(&output, &output.kind().sum_all().expect("kind loss"))
            .expect("backward");
        let (selected, opposite) = if dire {
            ("dire.kind.bias", "kind.bias")
        } else {
            ("kind.bias", "dire.kind.bias")
        };
        assert!(
            gradient_values(&named, selected)
                .iter()
                .all(|value| *value > 0.0)
        );
        assert!(
            gradient_values(&named, &selected.replace("bias", "weight"))
                .iter()
                .any(|value| *value > 0.0)
        );
        assert!(
            gradient_values(&named, opposite)
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            gradient_values(&named, &opposite.replace("bias", "weight"))
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            gradient_values(&named, "value.1.weight")
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            gradient_values(&named, "trunk.2.bias")
                .iter()
                .any(|value| *value > 0.0)
        );
    }
    assert_critic_trains_the_shared_trunk_only(device);
}

fn assert_critic_trains_the_shared_trunk_only(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9002, device).expect("model");
    let _guard = model.read_parameter_lock().expect("critic read guard");
    let output = model
        .training_forward_locked(
            &[side_frame(false), side_frame(true)],
            &[TrainingPrefix::new(ActionKind::Continue, None, None); 2],
        )
        .expect("PPO critic");
    let named = model
        .backward_named_locked(&output.value.sum_all().expect("value loss"))
        .expect("critic backward");
    // Two rows through the fixed read-out scale of sixteen.
    assert_eq!(gradient_values(&named, "value.1.bias"), vec![32.0]);
    assert!(
        gradient_values(&named, "trunk.2.weight")
            .iter()
            .any(|value| *value != 0.0)
    );
    for name in ["kind.bias", "dire.kind.bias"] {
        assert!(
            gradient_values(&named, name)
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}

fn mixed_gradient_model(device: PolicyDevice) -> PolicyModel {
    let model = PolicyModel::fresh_on(9001, device).expect("model");
    edit(&model, |name, values| {
        values.fill(0.0);
        let base = name.strip_prefix("dire.").unwrap_or(name);
        match base {
            "unit.2.bias" | "point.1.bias" => values.fill(1.0),
            "trunk.2.bias" => values.fill(2.0),
            "kind_embedding.weight"
            | "unit_embedding.weight"
            | "ability_embedding.weight"
            | "item_embedding.weight" => values.fill(0.25),
            "item_head.weight" | "entity_query.weight" | "point_query.weight" => {
                values.fill(if name.starts_with("dire.") {
                    0.25
                } else {
                    0.125
                });
            }
            _ => {}
        }
    });
    model
}

fn assert_mixed_gradients(device: PolicyDevice) {
    let model = mixed_gradient_model(device);
    let mut frames = [side_frame(false), side_frame(true)];
    for frame in &mut frames {
        frame.units[0][unit_feature::TOKEN_PRESENT] = 1.0;
        frame.units[0][unit_feature::KIND_START] = 1.0;
        frame.points[0][point_feature::TOKEN_PRESENT] = 1.0;
    }
    let prefixes = [
        TrainingPrefix::new(
            ActionKind::Use,
            Some(ControlledUnit::Hero),
            Some(TrainingSlot::Item(
                TrainingItemSlot::new(2).expect("item slot"),
            )),
        ),
        TrainingPrefix::new(
            ActionKind::Cast,
            Some(ControlledUnit::Courier),
            Some(TrainingSlot::Ability(
                TrainingAbilitySlot::new(3).expect("ability slot"),
            )),
        ),
    ];
    let output = model
        .training_forward(&frames, &prefixes)
        .expect("mixed forward");
    let state = model
        .forward_frames(&frames)
        .expect("full-batch reference tokens");
    for (head, name, selected) in [
        (3, "item_head", output.item()),
        (10, "entity_query", output.entity_pointer()),
        (11, "point_query", output.point_pointer()),
    ] {
        let reference = reference_head(head, &output.tensors.side_raw[head], &state);
        assert_bits(
            &selected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &reference.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let actual = model
            .backward_named(&output, &selected.sum_all().unwrap())
            .expect("selected backward");
        let expected = model
            .backward_named(&output, &reference.sum_all().unwrap())
            .expect("reference backward");
        for gradient in &actual {
            assert_bits(
                &gradient_values(&actual, gradient.name()),
                &gradient_values(&expected, gradient.name()),
            );
        }
        assert_mixed_gradient_activity(&actual, name, head != 3);
    }
}

fn reference_head(head: usize, raw: &[Tensor; 2], state: &ForwardState) -> Tensor {
    let branches = match head {
        3 => raw.clone(),
        10 | 11 => {
            let tokens = if head == 10 {
                &state.current_units
            } else {
                &state.points
            };
            std::array::from_fn(|side| {
                scaled_pointer_dot(tokens, &raw[side].unsqueeze(1).expect("full query"))
                    .expect("full-batch branch pointer scores")
            })
        }
        _ => panic!("unsupported reference head"),
    };
    assert_eq!(branches[0].dims()[0], 2);
    assert_eq!(branches[0].dims(), branches[1].dims());
    let masks = [[1u8, 0], [0u8, 1]];
    let masked: [Tensor; 2] = std::array::from_fn(|side| {
        let branch = &branches[side];
        let mask = Tensor::from_slice(&masks[side], (2, 1), branch.device())
            .expect("side mask")
            .broadcast_as(branch.shape())
            .expect("full mask");
        let zeros = Tensor::zeros(branch.shape(), DType::F32, branch.device()).expect("zeros");
        mask.where_cond(branch, &zeros)
            .expect("independent row mask")
    });
    (&masked[0] + &masked[1]).expect("full-batch reference")
}

fn assert_mixed_gradient_activity(named: &[NamedPolicyGradient], head: &str, slots: bool) {
    for prefix in ["", "dire."] {
        for suffix in ["weight", "bias"] {
            let name = format!("{prefix}{head}.{suffix}");
            assert!(
                gradient_values(named, &name)
                    .iter()
                    .any(|value| *value > 0.0),
                "{name}"
            );
        }
    }
    for name in [
        "kind_embedding.weight",
        "unit_embedding.weight",
        "trunk.2.bias",
    ] {
        assert!(
            gradient_values(named, name)
                .iter()
                .any(|value| *value > 0.0),
            "{name}"
        );
    }
    for name in ["ability_embedding.weight", "item_embedding.weight"] {
        let values = gradient_values(named, name);
        if slots {
            assert!(values.iter().any(|value| *value > 0.0), "{name}");
        } else {
            assert!(values.iter().all(|value| *value == 0.0), "unused {name}");
        }
    }
    assert!(
        gradient_values(named, "value.1.weight")
            .iter()
            .all(|value| *value == 0.0)
    );
}

fn assert_unselected_head_overflow(device: PolicyDevice) {
    for head in ["dire.kind", "dire.entity_query"] {
        let model = PolicyModel::fresh_on(9001, device).expect("model");
        let (weight, bias) = (format!("{head}.weight"), format!("{head}.bias"));
        edit(&model, |name, values| match name {
            "trunk.2.weight" => values.fill(0.0),
            "trunk.2.bias" => values.fill(2.0),
            _ if name == weight || name == bias => values.fill(f32::MAX),
            _ => {}
        });
        let error = model
            .training_forward(
                &[side_frame(false)],
                &[TrainingPrefix::new(ActionKind::Continue, None, None)],
            )
            .expect_err(head);
        assert_eq!(
            error,
            ModelError::NonFiniteOutput {
                field: head,
                batch: 0,
                index: 0,
            },
            "{head}"
        );
    }
}

#[cfg(feature = "builtin")]
fn native_inputs() -> (Vec<FeatureFrame>, Vec<ActionSpace>) {
    use bota_proto::{MapId, ServerMsg, SlotId};
    let (_arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 9001,
    })
    .expect("native arena");
    (0..2)
        .map(|seat| {
            let [
                ServerMsg::MatchStart { info },
                ServerMsg::Snapshot { view },
                ServerMsg::Events { tick, events },
            ] = start.messages[seat].as_slice()
            else {
                panic!("native initial messages")
            };
            let mut tracker = crate::StateTracker::new(SlotId(seat as u8), info).expect("tracker");
            tracker.observe_snapshot(view).expect("snapshot");
            tracker.observe_events(*tick, events).expect("events");
            let space = ActionSpace::from_tracker(&tracker).expect("space");
            assert!(space.kind_mask().as_array()[ActionKind::Continue.index()]);
            assert!(space.kind_mask().as_array()[ActionKind::Stop.index()]);
            let mut encoder = crate::FeatureEncoder::new(&tracker);
            encoder.observe(&tracker).expect("observe");
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
            assert_eq!(frame.global[4], f32::from(seat == 0));
            assert_eq!(frame.global[5], f32::from(seat == 1));
            (frame, space)
        })
        .unzip()
}

#[cfg(feature = "builtin")]
fn routing_model(device: PolicyDevice) -> PolicyModel {
    let model = PolicyModel::fresh_on(9001, device).expect("model");
    edit(&model, |name, values| match name {
        "kind.weight"
        | "dire.kind.weight"
        | "value.weight"
        | "controlled.weight"
        | "dire.controlled.weight" => values.fill(0.0),
        "kind.bias" | "dire.kind.bias" => {
            values.fill(-100.0);
            let kind = if name.starts_with("dire.") {
                ActionKind::Stop
            } else {
                ActionKind::Continue
            };
            values[kind.index()] = 100.0;
        }
        "controlled.bias" | "dire.controlled.bias" => values.copy_from_slice(&[100.0, -100.0]),
        "value.bias" => values.fill(0.25),
        _ => {}
    });
    model
}

#[cfg(feature = "builtin")]
fn assert_routing(device: PolicyDevice) {
    let model = routing_model(device);
    let (frames, mut spaces) = native_inputs();
    let initial = [PpoRng::new(41), PpoRng::new(42)];
    let mut random = initial.clone();
    take_encoder_forwards_for_test();
    let selected = model
        .sample_batch(&frames, &spaces, &mut random)
        .expect("mixed sample");
    assert_eq!(take_encoder_forwards_for_test(), 1);
    let mut reversed = [initial[1].clone(), initial[0].clone()];
    spaces.swap(0, 1);
    let permuted = model
        .sample_batch(
            &[frames[1].clone(), frames[0].clone()],
            &spaces,
            &mut reversed,
        )
        .expect("permutation");
    spaces.swap(0, 1);
    let greedy = model.choose_batch(&frames, &spaces).expect("mixed greedy");
    for index in 0..2 {
        assert_eq!(
            selected[index].action().kind(),
            [ActionKind::Continue, ActionKind::Stop][index]
        );
        assert_eq!(selected[index].action(), permuted[1 - index].action());
        assert_eq!(random[index], reversed[1 - index]);
        let left = &selected[index];
        let right = &permuted[1 - index];
        assert_bits(
            &[left.value, left.log_probability],
            &[right.value, right.log_probability],
        );
        assert_eq!(greedy[index].action, selected[index].action());
        assert_eq!(
            greedy[index].value.to_bits(),
            selected[index].value.to_bits()
        );
        assert_single(&model, &frames[index], &spaces[index], &selected[index]);
    }
    let mut malformed = frames;
    malformed[1].global[4] = 0.0;
    malformed[1].global[5] = 0.0;
    let before = random.clone();
    take_encoder_forwards_for_test();
    assert_eq!(
        model
            .sample_batch(&malformed, &spaces, &mut random)
            .expect_err("side before RNG"),
        ModelError::InvalidSideOneHot {
            index: 1,
            radiant_bits: 0,
            dire_bits: 0
        }
    );
    assert_eq!(random, before);
    assert_eq!(take_encoder_forwards_for_test(), 0);
}

#[cfg(feature = "builtin")]
fn assert_unused_family_skipping(device: PolicyDevice) {
    let model = routing_model(device);
    let (frames, spaces) = native_inputs();
    edit(&model, |name, values| match name {
        "trunk.2.weight" => values.fill(0.0),
        "trunk.2.bias" => values.fill(2.0),
        "item_head.weight" | "item_head.bias" | "dire.item_head.weight" | "dire.item_head.bias" => {
            values.fill(f32::MAX)
        }
        _ => {}
    });
    let mut random = [PpoRng::new(41)];
    let selected = model
        .sample_batch(&frames[..1], &spaces[..1], &mut random)
        .expect("Continue skips overflowing item family");
    assert_eq!(selected[0].action().kind(), ActionKind::Continue);
    edit(&model, |name, values| match name {
        "kind.bias" => {
            values.fill(-100.0);
            values[ActionKind::Stop.index()] = 100.0;
        }
        "controlled.weight" | "controlled.bias" => values.fill(f32::MAX),
        _ => {}
    });
    let before = random.clone();
    assert_eq!(
        model
            .sample_batch(&frames[..1], &spaces[..1], &mut random)
            .expect_err("Stop traverses controlled head"),
        ModelError::NonFiniteOutput {
            field: "radiant.controlled",
            batch: 0,
            index: 0,
        }
    );
    assert_eq!(random, before);
}

#[cfg(feature = "builtin")]
fn assert_value_rejection(device: PolicyDevice) {
    let model = routing_model(device);
    let (frames, spaces) = native_inputs();
    edit(&model, |name, values| match name {
        "trunk.2.weight" => values.fill(0.0),
        "trunk.2.bias" => values.fill(2.0),
        "value.1.weight" => values.fill(f32::MAX),
        _ => {}
    });
    let mut random = [PpoRng::new(19)];
    let before = random.clone();
    let error = model
        .sample_actions(&frames[..1], &spaces[..1], &mut random)
        .expect_err("value must be finite");
    assert_eq!(
        error.to_string(),
        "model value output at batch 0 index 0 is non-finite"
    );
    assert_eq!(random, before);
}

/// The scalar greedy path and the learner agree with one row of a mixed-side batch.
#[cfg(feature = "builtin")]
fn assert_single(
    model: &PolicyModel,
    frame: &FeatureFrame,
    space: &ActionSpace,
    expected: &crate::PpoPolicyChoice,
) {
    let greedy = model.choose(frame, space).expect("single greedy");
    assert_eq!(greedy.action, expected.action());
    let outcome = crate::PpoOutcome {
        stream: 0,
        decision: 0,
        ticks: 3,
        next_value: 0.0,
        reward: 0.0,
        terminal: true,
    };
    let sample = PpoPreparedSample {
        transition: expected.clone().finish(0, outcome).expect("transition"),
        advantage: 0.0,
        return_value: 0.0,
    };
    let (learner, _) = model
        .ppo_likelihood_for_test(&[&sample])
        .expect("learner likelihood");
    for (actual, expected) in [
        (learner[0], expected.log_probability),
        (greedy.value, expected.value),
    ] {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "batch versus singleton shape"
        );
    }
}

#[cfg(feature = "builtin")]
fn ppo_samples() -> Vec<PpoPreparedSample> {
    let (frames, spaces) = native_inputs();
    frames
        .into_iter()
        .zip(spaces)
        .enumerate()
        .map(|(stream, (frame, space))| {
            let target = ActionHeadTargets::from_action(&frame, &space, StructuredAction::Continue)
                .expect("native continue target");
            PpoPreparedSample {
                transition: crate::PpoTransition {
                    frame,
                    target,
                    shadow: None,
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
                    weight: 1.0,
                },
                advantage: 1.0,
                return_value: 1.0,
            }
        })
        .collect()
}

#[cfg(feature = "builtin")]
fn ppo_model(device: PolicyDevice) -> (PolicyModel, PpoConfig) {
    let model = PolicyModel::fresh_on(9001, device).expect("model");
    edit(&model, |name, values| {
        values.fill(0.0);
        if matches!(name, "kind.bias" | "dire.kind.bias") {
            values.fill(-100.0);
            values[0] = 0.0;
            values[1] = 0.0;
        }
    });
    let config = PpoConfig {
        learning_rate: 0.01,
        target_kl: 1.0e3,
        ..PpoConfig::default()
    };
    (model, config)
}

#[cfg(feature = "builtin")]
fn assert_ppo_rollback(device: PolicyDevice) {
    let (model, config) = ppo_model(device);
    let mut adam = model.claim_optimizer(config.adam()).expect("Adam");
    let mut samples = ppo_samples();
    for candidate in [
        Candidate::Accepted,
        Candidate::KlRejected,
        Candidate::Failed,
    ] {
        assert_candidate(&model, &mut adam, &mut samples, config, candidate);
    }
    // Early stopping keeps the step the post-step guard rejected above: only
    // the KL before the step (from the gradient pass) is held to the target.
    let before = model.coherent_snapshot(&adam).expect("before early stop");
    let config = PpoConfig {
        target_kl: 1.0e-12,
        ..config
    };
    let report = model
        .ppo_update_early_stop_for_test(&samples.iter().collect::<Vec<_>>(), &mut adam, config)
        .expect("early-stop step");
    assert!(report.applied);
    assert!(report.approximate_kl <= f64::from(config.target_kl));
    let after = model.coherent_snapshot(&adam).expect("after early stop");
    assert_eq!(after.adam.step, 2);
    assert_ne!(after.parameters, before.parameters);
}

#[cfg(feature = "builtin")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Accepted,
    KlRejected,
    Failed,
}

/// Runs one two-row candidate update after the first accepted one; only an
/// accepted candidate may change parameters or Adam state.
#[cfg(feature = "builtin")]
fn assert_candidate(
    model: &PolicyModel,
    adam: &mut crate::AdamState,
    samples: &mut [PpoPreparedSample],
    config: PpoConfig,
    candidate: Candidate,
) {
    let (probabilities, _) = model
        .ppo_likelihood_for_test(&samples.iter().collect::<Vec<_>>())
        .expect("same two-row old probabilities");
    for (sample, probability) in samples.iter_mut().zip(probabilities) {
        sample.transition.old_log_probability = probability;
    }
    let before = model.coherent_snapshot(adam).expect("before candidate");
    let config = PpoConfig {
        target_kl: if candidate == Candidate::KlRejected {
            1.0e-12
        } else {
            config.target_kl
        },
        ..config
    };
    let result = model.ppo_update_with_microbatch(
        &samples.iter().collect::<Vec<_>>(),
        adam,
        config,
        2,
        PpoTestFaults {
            candidate_evaluation: candidate == Candidate::Failed,
            rollback_import: false,
        },
    );
    if candidate == Candidate::Failed {
        assert_eq!(
            result.expect_err("candidate failure").to_string(),
            "model tensor operation failed: injected PPO candidate evaluation failure after 2 rows"
        );
    } else {
        let report = result.expect("candidate decision");
        assert_eq!(report.applied, candidate == Candidate::Accepted);
        assert!(
            candidate == Candidate::Accepted || report.approximate_kl > f64::from(config.target_kl)
        );
    }
    let after = model.coherent_snapshot(adam).expect("after candidate");
    assert_eq!(after.adam.step, 1, "{candidate:?}");
    assert_both_kind_heads_have_moments(model, &after.adam);
    if candidate != Candidate::Accepted {
        assert_bits(&after.parameters, &before.parameters);
        let (before_first, before_second) = before.adam.moments().expect("before moments");
        let (after_first, after_second) = after.adam.moments().expect("after moments");
        assert_bits(&after_first, &before_first);
        assert_bits(&after_second, &before_second);
        assert_eq!(after.adam.binding, before.adam.binding);
        assert_eq!(after.adam.config, before.adam.config);
    }
}

#[cfg(feature = "builtin")]
fn assert_both_kind_heads_have_moments(model: &PolicyModel, adam: &crate::AdamState) {
    let (first, second) = adam.moments().expect("moments");
    for head in ["kind", "dire.kind"] {
        let mut offset = 0;
        let mut range = None;
        for (name, shape) in model.parameter_schema().expect("schema") {
            let end = offset + shape.iter().product::<usize>();
            if name == format!("{head}.weight") {
                range = Some(offset..end);
            } else if name == format!("{head}.bias") {
                range = range.map(|range: std::ops::Range<usize>| range.start..end);
            }
            offset = end;
        }
        let range = range.expect("kind head parameters");
        for moments in [&first, &second] {
            assert!(
                moments[range.clone()].iter().any(|value| *value != 0.0),
                "{head}"
            );
        }
    }
}

/// Measures the device pool's live peak of each budget component on CUDA
/// (run alone: the pool is per process) and checks the budget constants bound
/// them; prints the measurements for recalibration.
#[cfg(all(feature = "builtin", feature = "cuda"))]
#[test]
#[ignore = "exclusive CUDA: measures the process-wide memory pool"]
fn vram_budget_constants_bound_measured_peaks() {
    use crate::model::vram::current_pool_usage;
    let device = PolicyDevice::Cuda { ordinal: 0 };
    let model = PolicyModel::fresh_on(9001, device).expect("model");
    let (frames, spaces) = native_inputs();
    let base = ppo_samples();
    let config = PpoConfig {
        target_kl: 1.0e3,
        ..PpoConfig::default()
    };
    let mut adam = model.claim_optimizer(config.adam()).expect("Adam");
    for (rows, imitation) in [(256_usize, false), (512, false), (256, true), (512, true)] {
        let samples: Vec<_> = (0..rows)
            .map(|row| {
                let mut sample = base[row % 2].clone();
                if imitation {
                    sample.transition.shadow = Some(sample.transition.target.clone());
                }
                sample
            })
            .collect();
        let (before, _) = current_pool_usage(0, true);
        let mut staging = model
            .ppo_staging(rows, imitation.then_some(0.5))
            .expect("staging");
        for sample in &samples {
            staging.push(sample).expect("push");
        }
        let staged = model.stage_ppo_batch(staging).expect("stage");
        let (after, _) = current_pool_usage(0, false);
        let staged_row = (after - before) / rows as u64;
        let (resting, _) = current_pool_usage(0, true);
        let indices: Vec<_> = (0..rows).collect();
        model
            .ppo_update_staged(
                &staged,
                &indices,
                &mut adam,
                (
                    config,
                    crate::UpdateObjective {
                        imitation: if imitation { 1.0 } else { 0.0 },
                        imitation_balance: 0.5,
                        critic_only: false,
                    },
                ),
                (rows, crate::KlGuard::PostStep),
            )
            .expect("step");
        let (_, high) = current_pool_usage(0, false);
        let learner_row = (high - resting) / rows as u64;
        eprintln!(
            "vram rows={rows} staged_row={staged_row} learner_peak={} learner_row={learner_row}",
            high - resting
        );
        assert!(staged_row <= crate::model::VRAM_STAGED_ROW_BYTES);
        assert!(
            high - resting
                <= crate::model::VRAM_LEARNER_FIXED_BYTES
                    + rows as u64 * crate::model::VRAM_LEARNER_ROW_BYTES
        );
    }
    for rows in [64_usize, 128] {
        let encoded: Vec<_> = (0..rows)
            .map(|row| EncoderRow::from_frame(&frames[row % 2]).expect("row"))
            .collect();
        let references: Vec<_> = encoded.iter().collect();
        let space_refs: Vec<_> = (0..rows).map(|row| &spaces[row % 2]).collect();
        let mut random: Vec<_> = (0..rows).map(|row| PpoRng::new(row as u64)).collect();
        let wanted = vec![true; rows];
        let (resting, _) = current_pool_usage(0, true);
        model
            .sample_rows(&references, &space_refs, &mut random, &wanted)
            .expect("inference");
        let (_, high) = current_pool_usage(0, false);
        eprintln!("vram inference rows={rows} peak={}", high - resting);
        assert!(
            high - resting
                <= crate::model::VRAM_INFERENCE_FIXED_BYTES
                    + rows as u64 * crate::model::VRAM_INFERENCE_ROW_BYTES
        );
    }
}
