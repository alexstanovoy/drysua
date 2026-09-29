use super::*;

#[test]
fn kind_stage_inputs_require_twenty_rows_and_closed_kind_and_side_ranges() {
    assert_eq!((STAGE_BATCH, PACKED_ELEMENTS), (20, 480));
    for (kinds, sides) in [(0, 0), (19, 20), (21, 20), (20, 19), (20, 21)] {
        assert_eq!(
            validate_inputs(&vec![16; kinds], &vec![2; sides]),
            Err(ModelError::InvalidModelState("kind stage input counts"))
        );
    }
    for kind in [0, 15] {
        for side in [0, 1] {
            assert_eq!(validate_inputs(&[kind; 20], &[side; 20]), Ok(()));
        }
    }
    assert_eq!(
        validate_inputs(&[16; 20], &[2; 20]),
        Err(ModelError::InvalidModelState("kind stage kind index"))
    );
    assert_eq!(
        validate_inputs(&[0; 20], &[2; 20]),
        Err(ModelError::InvalidModelState("kind stage side mask"))
    );
}

#[test]
fn kind_stage_unpack_preserves_selected_bits_and_skips_unneeded_nonfinite_families() {
    let mut values = (0..480).map(|index| index as f32).collect::<Vec<_>>();
    values[320] = -0.0;
    values[360] = f32::from_bits(1);
    for needed in [[true, true], [true, false], [false, true], [false, false]] {
        let mut packed = values.clone();
        if !needed[0] {
            packed[..80].fill(f32::NAN);
            packed[320..360].fill(f32::NAN);
        }
        if !needed[1] {
            packed[80..320].fill(f32::NAN);
            packed[360..].fill(f32::NAN);
        }
        let actual = unpack(&packed, needed).expect("logical family extraction");
        assert_part(&actual.controlled, &values[320..360], 2, needed[0]);
        assert_part(&actual.learn, &values[360..480], 6, needed[1]);
    }
    for count in [0, 479, 481] {
        assert_eq!(
            unpack(&vec![f32::NAN; count], [false, false])
                .err()
                .expect("count first"),
            ModelError::InvalidModelState("kind stage output count")
        );
    }
}

#[test]
fn kind_stage_unpack_checks_raw_branch_order_and_selected_coordinates() {
    let cases = [
        (7, "radiant.controlled", 3, 1),
        (47, "dire.controlled", 3, 1),
        (97, "radiant.learn", 2, 5),
        (217, "dire.learn", 2, 5),
    ];
    let mut values = vec![0.0; 480];
    for (offset, _, _, _) in cases {
        values[offset] = f32::NAN;
    }
    for (offset, field, batch, index) in cases {
        assert_eq!(
            unpack(&values, [true, true]).err().expect("raw validation"),
            ModelError::NonFiniteOutput {
                field,
                batch,
                index
            }
        );
        values[offset] = 0.0;
    }
    for (offset, field, batch, index) in [(327, "controlled", 3, 1), (377, "learn", 2, 5)] {
        values[offset] = f32::INFINITY;
        assert_eq!(
            unpack(&values, [true, true])
                .err()
                .expect("selected validation"),
            ModelError::NonFiniteOutput {
                field,
                batch,
                index
            }
        );
        values[offset] = 0.0;
    }
}

#[test]
#[ignore = "single capture in a fresh process; requires the owner's bounded CUDA runner"]
fn cuda_kind_prefix_graph_preserves_dynamic_inputs_weights_errors_and_rng() {
    let model = stage_model();
    let trunk = Tensor::full(2.0f32, (20, 256), model.tensor_device()).expect("fixed trunk");
    let mut stage = KindStage::new(&model).expect("one admitted kind capture");
    for case in 0..6 {
        let kinds = std::array::from_fn::<_, 20, _>(|row| match case {
            1 => ActionKind::Learn.index() as u32,
            2 => {
                [ActionKind::Stop, ActionKind::Learn, ActionKind::Continue][row % 3].index() as u32
            }
            4 => ActionKind::Continue.index() as u32,
            _ => ActionKind::Stop.index() as u32,
        });
        let sides = std::array::from_fn::<_, 20, _>(|row| match case {
            3 => 1,
            5 => 0,
            _ => (row % 2) as u8,
        });
        assert_run(&model, &mut stage, &trunk, &kinds, &sides);
    }
    assert_changed_weights(&model, &mut stage, &trunk);
    measure_prefix_stage(&model, &mut stage, &trunk);
    #[cfg(feature = "builtin")]
    assert_native_sampling(&model, &mut stage);
    assert_unused_learn(&model, &mut stage, &trunk);
    assert!(stage.replays() <= 128);
    stage.finish();
    assert_eq!(
        KindStage::new(&model).err().expect("no second admission"),
        ModelError::InvalidModelState("actor graph admission already consumed")
    );
}

fn stage_model() -> PolicyModel {
    let model = PolicyModel::fresh_on(9001, PolicyDevice::Cuda { ordinal: 0 }).expect("model");
    edit(&model, |name, values| match name {
        "kind.weight" | "dire.kind.weight" => values.fill(0.0),
        "kind.bias" | "dire.kind.bias" => {
            values.fill(-100.0);
            values[ActionKind::Stop.index()] = 100.0;
        }
        "controlled.bias" => values.copy_from_slice(&[4.0, -4.0]),
        "dire.controlled.bias" => values.copy_from_slice(&[2.0, -2.0]),
        _ => {}
    });
    model
}

fn assert_part(actual: &Option<Vec<Vec<f32>>>, expected: &[f32], width: usize, needed: bool) {
    if !needed {
        assert!(actual.is_none());
        return;
    }
    let rows = actual.as_ref().expect("needed family");
    assert_eq!(rows.len(), 20);
    assert!(rows.iter().all(|row| row.len() == width));
    assert!(
        rows.iter()
            .flatten()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits()))
    );
}

fn assert_logits(actual: &SamplingKindLogits, expected: &SamplingKindLogits) {
    for (actual, expected, width) in [
        (&actual.controlled, &expected.controlled, 2),
        (&actual.learn, &expected.learn, 6),
    ] {
        let values = expected
            .as_ref()
            .map(|rows| rows.iter().flatten().copied().collect::<Vec<_>>());
        assert_part(
            actual,
            values.as_deref().unwrap_or(&[]),
            width,
            expected.is_some(),
        );
    }
}

fn assert_run(
    model: &PolicyModel,
    stage: &mut KindStage<'_>,
    trunk: &Tensor,
    kinds: &[u32],
    sides: &[u8],
) -> SamplingKindLogits {
    let _guard = model.read_parameter_lock().expect("stage owner read guard");
    let expected = eager(model, trunk, kinds, sides).expect("eager kind prefix");
    let before = stage.replays();
    let actual = stage
        .run_locked(trunk, kinds, sides)
        .expect("graph kind prefix");
    assert_logits(&actual, &expected);
    let needed = expected.controlled.is_some() || expected.learn.is_some();
    assert_eq!(stage.replays(), before + usize::from(needed));
    actual
}

fn eager(
    model: &PolicyModel,
    trunk: &Tensor,
    kinds: &[u32],
    sides: &[u8],
) -> Result<SamplingKindLogits, ModelError> {
    validate_inputs(kinds, sides)?;
    let frames = sides
        .iter()
        .map(|side| {
            let mut frame = FeatureFrame::new();
            frame.global[crate::global_feature::SIDE_RADIANT] = f32::from(*side);
            frame.global[crate::global_feature::SIDE_DIRE] = f32::from(1 - *side);
            frame
        })
        .collect::<Vec<_>>();
    let prefixes = kinds
        .iter()
        .map(|kind| {
            TrainingPrefix::new(
                ActionKind::from_index(*kind as usize).expect("checked kind"),
                None,
                None,
            )
        })
        .collect::<Vec<_>>();
    let dummy = Tensor::zeros((20, 1), DType::F32, model.tensor_device())?;
    let state = ForwardState {
        trunk: trunk.clone(),
        current_units: dummy.clone(),
        points: dummy,
    };
    let routing = ActorRouting::new(&frames, model.tensor_device(), false)?;
    model.sampling_kind_logits(&state, &prefixes, &routing)
}

fn edit(model: &PolicyModel, mut operation: impl FnMut(&str, &mut [f32])) {
    let mut values = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let end = offset + shape.iter().product::<usize>();
        operation(name, &mut values[offset..end]);
        offset = end;
    }
    assert_eq!(offset, MODEL_PARAMETER_COUNT);
    model
        .import_parameters(&values)
        .expect("finite in-place weights");
}

fn assert_changed_weights(model: &PolicyModel, stage: &mut KindStage<'_>, trunk: &Tensor) {
    let kinds = [ActionKind::Stop.index() as u32; 20];
    let sides = [1; 20];
    let before = assert_run(model, stage, trunk, &kinds, &sides);
    edit(model, |name, values| match name {
        "controlled.weight" | "learn_head.weight" => values.fill(0.125),
        "dire.controlled.weight" | "dire.learn_head.weight" => values.fill(0.25),
        "kind_embedding.weight" => values.fill(0.5),
        _ => {}
    });
    let after = assert_run(model, stage, trunk, &kinds, &sides);
    assert_ne!(
        before.controlled, after.controlled,
        "captured weights must stay live"
    );
    let changed = Tensor::full(3.0f32, (20, 256), model.tensor_device()).expect("changed trunk");
    let dynamic = assert_run(model, stage, &changed, &kinds, &sides);
    assert_ne!(
        after.controlled, dynamic.controlled,
        "fixed input buffer must be refreshed"
    );
    assert_run(
        model,
        stage,
        &changed,
        &[ActionKind::Learn.index() as u32; 20],
        &[0; 20],
    );
}

fn assert_unused_learn(model: &PolicyModel, stage: &mut KindStage<'_>, trunk: &Tensor) {
    edit(model, |name, values| {
        if matches!(
            name,
            "learn_head.weight"
                | "learn_head.bias"
                | "dire.learn_head.weight"
                | "dire.learn_head.bias"
        ) {
            values.fill(f32::MAX);
        }
    });
    assert_run(
        model,
        stage,
        trunk,
        &[ActionKind::Stop.index() as u32; 20],
        &[1; 20],
    );
    let _guard = model.read_parameter_lock().expect("error read guard");
    let before = stage.replays();
    assert_eq!(
        stage
            .run_locked(trunk, &[0; 19], &[1; 20])
            .err()
            .expect("invalid counts"),
        ModelError::InvalidModelState("kind stage input counts")
    );
    assert_eq!(stage.replays(), before);
    let kinds = [ActionKind::Learn.index() as u32; 20];
    let expected = eager(model, trunk, &kinds, &[1; 20])
        .err()
        .expect("eager raw Learn error");
    let actual = stage
        .run_locked(trunk, &kinds, &[1; 20])
        .err()
        .expect("graph raw Learn error");
    assert_eq!(stage.replays(), before + 1);
    assert_eq!(actual, expected);
    assert_eq!(
        actual,
        ModelError::NonFiniteOutput {
            field: "radiant.learn",
            batch: 0,
            index: 0
        }
    );
}

fn measure_prefix_stage(model: &PolicyModel, stage: &mut KindStage<'_>, trunk: &Tensor) {
    let kinds: [u32; 20] = std::array::from_fn(|row| {
        if row % 2 == 0 {
            ActionKind::Stop.index() as u32
        } else {
            ActionKind::Learn.index() as u32
        }
    });
    let sides = [1u8; 20];
    let frames = (0..20)
        .map(|_| {
            let mut frame = FeatureFrame::new();
            frame.global[crate::global_feature::SIDE_RADIANT] = 1.0;
            frame
        })
        .collect::<Vec<_>>();
    let prefixes = kinds
        .iter()
        .map(|kind| {
            TrainingPrefix::new(ActionKind::from_index(*kind as usize).unwrap(), None, None)
        })
        .collect::<Vec<_>>();
    let _guard = model.read_parameter_lock().expect("measurement owner");
    let routing =
        ActorRouting::new(&frames, model.tensor_device(), false).expect("resident routing");
    let dummy = Tensor::zeros((20, 1), DType::F32, model.tensor_device()).unwrap();
    let state = ForwardState {
        trunk: trunk.clone(),
        current_units: dummy.clone(),
        points: dummy,
    };
    let stream = probe_stream(model);
    let eager = measure_kind_calls("eager", &stream, || {
        model.sampling_kind_logits(&state, &prefixes, &routing)
    });
    let replayed = measure_kind_calls("graph_including_input_copies", &stream, || {
        stage.run_locked(trunk, &kinds, &sides)
    });
    assert_logits(&replayed, &eager);
}

fn measure_kind_calls(
    label: &str,
    stream: &Arc<CudaStream>,
    mut operation: impl FnMut() -> Result<SamplingKindLogits, ModelError>,
) -> SamplingKindLogits {
    stream.synchronize().expect("measurement start");
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("start event");
    let wall = Instant::now();
    let mut last = None;
    for _ in 0..20 {
        last = Some(operation().expect("measured stage"));
    }
    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("end event");
    end.synchronize().expect("measurement completion");
    eprintln!(
        "kind-stage-measure mode={label} iterations=20 batch=20 wall_ns={} stream_interval_ms={} scope=kind_prefix_only",
        wall.elapsed().as_nanos(),
        start.elapsed_ms(&end).expect("elapsed events")
    );
    last.expect("twenty stage calls")
}

#[cfg(feature = "builtin")]
fn assert_native_sampling(model: &PolicyModel, stage: &mut KindStage<'_>) {
    let (frame, spaces) = native_inputs();
    for shift in [0.0, 0.25] {
        let mut frames = vec![frame.clone(); 20];
        for frame in &mut frames {
            frame.global[0] += shift;
        }
        let mut actual_rng = (0..20).map(|row| PpoRng::new(91 + row)).collect::<Vec<_>>();
        let mut expected_rng = actual_rng.clone();
        let expected = model
            .sample_batch(&frames, &spaces, &mut expected_rng)
            .expect("eager native sample");
        let before = stage.replays();
        let actual = stage
            .sample_batch(&frames, &spaces, &mut actual_rng)
            .expect("staged native sample");
        assert_eq!(stage.replays(), before + 1);
        assert_eq!(actual.len(), 20);
        assert_eq!(expected.len(), 20);
        assert_eq!(actual_rng, expected_rng);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.action().kind(), ActionKind::Stop);
            assert_eq!(actual.action(), expected.action());
            assert_eq!(actual.frame, expected.frame);
            assert_eq!(actual.target, expected.target);
            assert_eq!(actual.policy(), expected.policy());
            assert_eq!(
                [actual.value, actual.log_probability, actual.entropy].map(f32::to_bits),
                [expected.value, expected.log_probability, expected.entropy].map(f32::to_bits)
            );
        }
    }
}

#[cfg(feature = "builtin")]
fn native_inputs() -> (FeatureFrame, Vec<ActionSpace>) {
    use bota_proto::{MapId, ServerMsg, SlotId};
    let (_arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 9001,
    })
    .expect("native fixture");
    let [
        ServerMsg::MatchStart { info },
        ServerMsg::Snapshot { view },
        ServerMsg::Events { tick, events },
    ] = start.messages[0].as_slice()
    else {
        panic!("native start messages")
    };
    let mut tracker = crate::StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(view).expect("snapshot");
    tracker.observe_events(*tick, events).expect("events");
    let spaces = (0..20)
        .map(|_| ActionSpace::from_tracker(&tracker).expect("space"))
        .collect::<Vec<_>>();
    assert!(spaces[0].kind_mask().as_array()[ActionKind::Stop.index()]);
    let mut encoder = crate::FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            &tracker,
            &spaces[0],
            &crate::ItemReadiness::new(),
            &crate::LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("encoded frame");
    assert_eq!(frame.global[crate::global_feature::SIDE_RADIANT], 1.0);
    assert_eq!(frame.global[crate::global_feature::SIDE_DIRE], 0.0);
    (frame, spaces)
}
