use super::*;

#[test]
fn graph_mode_is_explicit_closed_and_rejects_non_boolean_values() {
    use std::ffi::OsStr;
    assert_eq!(parse_graph_mode(None).expect("absent"), None);
    for (value, enabled) in [("0", false), ("1", true)] {
        assert_eq!(
            parse_graph_mode(Some(OsStr::new(value))).expect("mode"),
            Some(enabled)
        );
    }
    for value in ["", "true", "2", "01", " 1"] {
        assert_eq!(
            parse_graph_mode(Some(OsStr::new(value)))
                .expect_err("closed flag")
                .to_string(),
            "model produced invalid DRYSUA_PROBE_ACTOR_GRAPH must be 0 or 1"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            parse_graph_mode(Some(OsStr::from_bytes(&[255])))
                .expect_err("non-UTF8")
                .to_string(),
            "model produced invalid DRYSUA_PROBE_ACTOR_GRAPH must be 0 or 1"
        );
    }
}

#[cfg(not(feature = "builtin"))]
#[test]
#[ignore = "one admission per process; requires the owner's isolated CUDA runner"]
fn cuda_actor_graph_scope_retires_and_never_returns_its_admission() {
    let model = PolicyModel::fresh_on(9001, PolicyDevice::Cuda { ordinal: 0 }).expect("model");
    with_actor_graph_for_test(&model, true, 40, || {
        assert_eq!(actor::counts_for_test(), Some((0, 0)));
    })
    .expect("checked retirement");
    assert_eq!(actor::counts_for_test(), None);
    let stats: ActorGraphStats = actor_graph_stats_for_test().expect("retired stats");
    assert_eq!(stats.shapes, [40, 1]);
    assert_eq!(stats.captures, 2);
    assert_eq!(
        with_actor_graph_for_test(&model, true, 40, || ())
            .expect_err("consumed")
            .to_string(),
        "model produced invalid actor graph admission already consumed"
    );
    model
        .export_parameters()
        .expect("owner remains usable after retirement");
}

#[test]
fn actor_graph_rejects_unbounded_shapes_before_device_or_admission() {
    let model = PolicyModel::fresh(9).expect("CPU preflight fixture");
    for batch in [0, 1, 19, 21, 39, 41, 64, 65] {
        assert_eq!(
            with_actor_graph_for_test(&model, true, batch, || ())
                .expect_err("closed shape plan")
                .to_string(),
            "model produced invalid actor graph main batch must be 20 or 40"
        );
    }
}

#[cfg(feature = "builtin")]
#[test]
#[ignore = "one admission per process; requires the owner's isolated CUDA runner"]
fn cuda_actor_graph_preserves_samples_rng_tails_weights_and_owner_bounds() {
    assert_graph_mode(40);
}

#[cfg(feature = "builtin")]
#[test]
#[ignore = "one admission per process; requires the owner's isolated CUDA runner"]
fn cuda_actor_graph_twenty_and_singleton_preserve_samples_rng_and_tails() {
    assert_graph_mode(20);
}

#[cfg(feature = "builtin")]
fn assert_graph_mode(main_batch: usize) {
    assert_previous_thread_use_rejected();
    let model = PolicyModel::fresh_on(9001, PolicyDevice::Cuda { ordinal: 0 }).expect("model");
    let (frames, spaces) = sampling_inputs(main_batch);
    let original = model.export_parameters().expect("original weights");
    let baseline = sample_reference(&model, &frames, &spaces);
    let mut changed = frames.clone();
    for frame in &mut changed {
        frame.global[0] += 0.25;
    }
    let changed_input = sample_reference(&model, &changed, &spaces);
    assert_ne!(
        baseline.0[0].value.to_bits(),
        changed_input.0[0].value.to_bits()
    );
    let tail_count = main_batch - 1;
    let tail = sample_reference(&model, &changed[..tail_count], &spaces[..tail_count]);
    let singleton = sample_reference(&model, &changed[..1], &spaces[..1]);
    change_weights(&model);
    let changed_weight = sample_reference(&model, &changed, &spaces);
    let singleton_weight = sample_reference(&model, &changed[..1], &spaces[..1]);
    assert_ne!(
        changed_input.0[0].value.to_bits(),
        changed_weight.0[0].value.to_bits()
    );
    model
        .import_parameters(&original)
        .expect("restore initial weights");
    with_actor_graph_for_test(&model, true, main_batch, || {
        assert_samples(&model, &frames, &spaces, &baseline);
        assert_samples(&model, &changed[..1], &spaces[..1], &singleton);
        assert_samples(&model, &changed, &spaces, &changed_input);
        assert_samples(&model, &changed[..tail_count], &spaces[..tail_count], &tail);
        change_weights(&model);
        assert_samples(&model, &changed[..1], &spaces[..1], &singleton_weight);
        assert_samples(&model, &changed, &spaces, &changed_weight);
        assert_eq!(actor::counts_for_test(), Some((6, 5)));
        assert_capture_budgets(&model, main_batch);
        assert_rejected_owners(&model, &frames, &spaces);
        assert_invalid_inputs(&model, &frames, &spaces);
        assert_eq!(actor::counts_for_test(), Some((6, 5)));
        Ok::<_, ModelError>(())
    })
    .expect("graph scope")
    .expect("samples");
    assert_eq!(actor::counts_for_test(), None);
    assert_eq!(
        with_actor_graph_for_test(&model, true, main_batch, || ())
            .expect_err("irreversible admission")
            .to_string(),
        "model produced invalid actor graph admission already consumed"
    );
    assert_histograms(main_batch);
    assert_samples(&model, &changed, &spaces, &changed_weight);
}

#[cfg(feature = "builtin")]
fn assert_capture_budgets(model: &PolicyModel, main_batch: usize) {
    for batch in [main_batch, 1] {
        assert_eq!(
            actor::authorize_shape(model, batch, true)
                .expect_err("one capture per slot")
                .to_string(),
            "model produced invalid actor graph shape capture already consumed"
        );
        assert_eq!(
            actor::authorize_shape(model, batch, false)
                .expect_err("two warmups per slot")
                .to_string(),
            "model produced invalid actor graph warmup budget exceeded"
        );
    }
    assert_eq!(
        actor::authorize_shape(model, main_batch - 1, false)
            .expect_err("no adaptive cache shapes")
            .to_string(),
        "model produced invalid actor graph shape was not admitted"
    );
}

#[cfg(feature = "builtin")]
fn assert_histograms(main_batch: usize) {
    let stats = actor_graph_stats_for_test().expect("retired stats");
    assert_eq!(stats.main_batch, main_batch);
    assert_eq!(stats.shapes, [main_batch, 1]);
    assert_eq!(stats.captures, 2);
    let mut calls = [0u64; 65];
    calls[main_batch] = 3;
    calls[main_batch - 1] = 1;
    calls[1] = 2;
    let mut hits = calls;
    hits[main_batch - 1] = 0;
    assert_eq!(stats.calls_by_batch, calls);
    assert_eq!(stats.hits_by_batch, hits);
}

#[cfg(feature = "builtin")]
fn assert_previous_thread_use_rejected() {
    let model =
        PolicyModel::fresh_on(19, PolicyDevice::Cuda { ordinal: 0 }).expect("unadmitted model");
    std::thread::scope(|scope| {
        scope
            .spawn(|| model.policy_identity())
            .join()
            .expect("thread")
    })
    .expect("ordinary model is not owner-restricted");
    assert_eq!(
        actor::admit(&model)
            .expect_err("historical PTDS use")
            .to_string(),
        "model produced invalid actor graph model was used by another thread"
    );
}

#[cfg(feature = "builtin")]
fn assert_invalid_inputs(model: &PolicyModel, frames: &[FeatureFrame], spaces: &[ActionSpace]) {
    let mut invalid = frames.to_vec();
    let last = frames.len() - 1;
    invalid[last].global[0] = f32::NAN;
    let mut random = vec![PpoRng::new(31); frames.len()];
    let before = random.clone();
    assert_eq!(
        model
            .sample_batch(&invalid, spaces, &mut random)
            .expect_err("nonfinite frame")
            .to_string(),
        format!("model frame {last} contains a non-finite value")
    );
    assert_eq!(random, before);
    let mut random = vec![PpoRng::new(31); 65];
    let before = random.clone();
    let (oversized_frames, oversized_spaces) = sampling_inputs(65);
    assert_eq!(
        model
            .sample_batch(&oversized_frames, &oversized_spaces, &mut random)
            .expect_err("public actor limit")
            .to_string(),
        "model batch count 65 exceeds maximum 64"
    );
    assert_eq!(random, before);
}

#[cfg(feature = "builtin")]
fn sampling_inputs(count: usize) -> (Vec<FeatureFrame>, Vec<ActionSpace>) {
    assert!((1..=65).contains(&count));
    use bota_proto::{MapId, ServerMsg, SlotId};
    let (_, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 9001,
    })
    .expect("native input fixture");
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
    let space = ActionSpace::from_tracker(&tracker).expect("space");
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
    let spaces = (0..count)
        .map(|_| ActionSpace::from_tracker(&tracker).expect("independent action space"))
        .collect();
    (vec![frame; count], spaces)
}

#[cfg(feature = "builtin")]
fn sample_reference(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
) -> (Vec<PpoPolicyChoice>, Vec<PpoRng>) {
    let mut random = (0..frames.len())
        .map(|row| PpoRng::new(91 + row as u64))
        .collect::<Vec<_>>();
    let choices = model
        .sample_batch(frames, spaces, &mut random)
        .expect("samples");
    (choices, random)
}

#[cfg(feature = "builtin")]
fn assert_samples(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    expected: &(Vec<PpoPolicyChoice>, Vec<PpoRng>),
) {
    let (choices, random) = sample_reference(model, frames, spaces);
    assert_eq!(random, expected.1);
    assert_eq!(choices.len(), expected.0.len());
    for (actual, expected) in choices.iter().zip(&expected.0) {
        assert_eq!(actual.action, expected.action);
        assert_eq!(actual.frame, expected.frame);
        assert_eq!(actual.target, expected.target);
        assert_eq!(
            [actual.value, actual.log_probability, actual.entropy].map(f32::to_bits),
            [expected.value, expected.log_probability, expected.entropy].map(f32::to_bits)
        );
        assert_eq!(
            actual.policy,
            model.policy_identity().expect("current revision")
        );
    }
}

#[cfg(feature = "builtin")]
fn assert_rejected_owners(model: &PolicyModel, frames: &[FeatureFrame], spaces: &[ActionSpace]) {
    let other = PolicyModel::fresh(7).expect("other model");
    let snapshot = model
        .actor_snapshot()
        .expect("same lineage on another device");
    for (other, message) in [
        (&other, "model produced invalid actor graph model mismatch"),
        (
            &snapshot,
            "model produced invalid actor graph device mismatch",
        ),
    ] {
        let mut random = vec![PpoRng::new(3); frames.len()];
        let before = random.clone();
        assert_eq!(
            other
                .sample_batch(frames, spaces, &mut random)
                .expect_err("binding")
                .to_string(),
            message
        );
        assert_eq!(random, before);
    }
    let error = std::thread::scope(|scope| {
        scope
            .spawn(|| model.policy_identity())
            .join()
            .expect("owner check")
    })
    .expect_err("wrong owner");
    assert_eq!(
        error.to_string(),
        "model produced invalid actor graph owner thread mismatch"
    );
}
