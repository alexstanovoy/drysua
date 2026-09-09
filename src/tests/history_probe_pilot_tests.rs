use super::*;

#[test]
fn conditioning_only_rescales_the_six_continuous_globals_without_clipping() {
    let [sample, _] = sample_pair();
    let mut raw = sample.frame().clone();
    for (index, value) in CONDITIONING_GLOBALS
        .into_iter()
        .zip([0.25, 2.0, 0.5, -3.0, 1.5, 4.0])
    {
        raw.global[index] = value;
    }
    let scaled = conditioning_frame(&raw);
    for index in 0..crate::GLOBAL_FEATURES {
        let expected = if CONDITIONING_GLOBALS.contains(&index) {
            raw.global[index] * 64.0
        } else {
            raw.global[index]
        };
        assert_eq!(scaled.global[index], expected);
    }
    let mut stripped = scaled;
    for index in CONDITIONING_GLOBALS {
        stripped.global[index] /= 64.0;
    }
    assert_eq!(stripped, raw);
}

#[test]
fn conditioning_trunk_rows_use_input_output_layout_and_fold_back_exactly() {
    let model = PolicyModel::fresh(10089500).expect("model");
    let original = zero_probe_input_weights(&model);
    model
        .import_parameters(&original)
        .expect("zero reserved rows");
    let scaled = conditioning_parameters(&model, 1.0 / 64.0);
    assert!(scaled != original, "six incoming rows must change");
    let offset = conditioning_trunk_offset(&model);
    for (index, (source, target)) in original.iter().zip(&scaled).enumerate() {
        let selected = (offset..offset + 2576 * 512).contains(&index)
            && CONDITIONING_GLOBALS.contains(&((index - offset) / 512));
        assert_eq!(*target, if selected { *source / 64.0 } else { *source });
    }
    model.import_parameters(&scaled).expect("scaled rows");
    assert!(
        conditioning_parameters(&model, 64.0) == original,
        "fold must restore parameter bits"
    );
}

#[test]
fn conditioning_balanced_schedule_excludes_validation_and_empty_kinds() {
    let samples = conditioning_schedule_fixture();
    let references: Vec<_> = samples.iter().collect();
    let schedule = conditioning_schedule(&references, true);
    let mut counts = [0usize; ActionKind::COUNT];
    assert_eq!(schedule.len(), 43520);
    for index in &schedule {
        assert_eq!(samples[*index].split(), ImitationSplit::Train);
        counts[samples[*index].teacher_action().kind().index()] += 1;
    }
    let active: Vec<_> = counts.into_iter().filter(|count| *count > 0).collect();
    assert_eq!(active.len(), 2);
    assert!(active.iter().max().expect("max") - active.iter().min().expect("min") <= 1);
    assert_eq!(schedule, conditioning_schedule(&references, true));
}

fn conditioning_schedule_fixture() -> Vec<ImitationSample> {
    let mut arena = observed_teacher_game(10089000, 0);
    let (frame, space) = prepare_neural_observer_sample(&mut arena.seats[1]).expect("frame");
    let mut samples = Vec::with_capacity(7);
    for index in 0..7 {
        let (namespace, seed) = if index == 6 {
            (SeedNamespace::Validation, 10089002)
        } else {
            (SeedNamespace::Training, 10089000)
        };
        let identity = SampleIdentity::from_frame(namespace, seed, index, space.tick(), &frame)
            .expect("identity");
        let action = match index {
            5 => crate::StructuredAction::Stop {
                unit: crate::ControlledUnit::Hero,
            },
            6 => crate::StructuredAction::Hold {
                unit: crate::ControlledUnit::Hero,
            },
            _ => crate::StructuredAction::Continue,
        };
        samples.push(
            ImitationSample::teacher(frame.clone(), &space, action, identity).expect("sample"),
        );
    }
    assert_eq!(samples.len(), 7);
    samples
}

#[test]
fn conditioning_initial_and_trained_fold_match_real_all_head_and_decoder_functions() {
    let (rows, spaces) = conditioning_real_fixture();
    let raw = PolicyModel::fresh(10089500).expect("raw model");
    raw.import_parameters(&zero_probe_input_weights(&raw))
        .expect("zero reserved");
    let trained = PolicyModel::fresh(10089500).expect("preconditioned model");
    trained
        .import_parameters(&conditioning_parameters(&raw, 1.0 / 64.0))
        .expect("inverse rows");
    conditioning_equivalence(&raw, &trained, &rows, &spaces, true, "unit-initial");
    let samples: Vec<_> = (0..64)
        .map(|index| &rows[index % rows.len()].alternate)
        .collect();
    let mut adam = trained
        .claim_optimizer(AdamConfig {
            learning_rate: 3e-5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .expect("fresh Adam");
    trained
        .behavioral_update(&samples, &mut adam)
        .expect("one smoke update");
    assert_eq!(adam.step(), 1);
    raw.import_parameters(&conditioning_parameters(&trained, 64.0))
        .expect("fold after training");
    conditioning_equivalence(&raw, &trained, &rows, &spaces, true, "unit-trained-fold");
}

#[test]
fn conditioning_uncompensated_scaling_fails_the_numerical_equivalence_gate() {
    let (rows, _) = conditioning_real_fixture();
    let model = PolicyModel::fresh(10089500).expect("instrument");
    let mut values = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("layout") {
        if name == "trunk.0.weight" {
            values[offset] = 128.0;
        }
        if ["trunk.1.weight", "trunk.2.weight", "value.weight"].contains(&name) {
            values[offset] = 1.0;
        }
        offset += shape.iter().product::<usize>();
    }
    model
        .import_parameters(&values)
        .expect("continuous tick instrument");
    let source = model
        .evaluate(rows[0].raw.frame())
        .expect("raw value")
        .value;
    let target = model
        .evaluate(rows[0].alternate.frame())
        .expect("uncompensated value")
        .value;
    assert_eq!(
        conditioning_close(source, target),
        Err("conditioning equivalence tolerance exceeded")
    );
    assert_eq!(
        conditioning_close(f32::NAN, target),
        Err("non-finite conditioning equivalence value")
    );
}

fn conditioning_real_fixture() -> (Vec<ConditioningRow>, Vec<ActionSpace>) {
    let model = PolicyModel::fresh(10089500).expect("opponent instrument");
    let mut rows = Vec::with_capacity(16);
    let mut spaces = Vec::with_capacity(16);
    for side in 0..2 {
        let mut environment = observed_teacher_game(10089000, side);
        let mut random = PpoRng::new(10089500);
        let mut history = SentOrderHistory::default();
        for _ in 0..8 {
            let expert = &environment.seats[1 - side];
            let teacher_space =
                ActionSpace::from_tracker_with_readiness(&expert.tracker, &expert.readiness)
                    .expect("pre-send space");
            let (frame, space) = prepare_policy_sample(&mut environment).expect("frame");
            let choice = model
                .sample(&frame, &space, &mut random)
                .expect("instrument choice");
            let (requests, [raw, _]) = history_samples(
                &mut environment,
                &choice,
                &space,
                10089000,
                SeedNamespace::Training,
                &mut history,
            )
            .expect("real observer frame");
            rows.push(ConditioningRow::new(rows.len(), raw, &teacher_space));
            spaces.push(teacher_space);
            advance_interval(&mut environment, requests, 3).expect("bounded tick");
        }
    }
    assert_eq!(rows.len(), 16);
    (rows, spaces)
}

#[test]
fn conditioning_confusion_reports_continue_false_positives_and_false_negatives() {
    let samples = conditioning_schedule_fixture();
    let mut metrics = ConditioningMetrics::default();
    let predicted_stop = target_prediction(samples[5].target());
    let predicted_continue = target_prediction(samples[0].target());
    assert_eq!(metrics.record(&samples[0], predicted_stop), (false, false));
    assert_eq!(
        metrics.record(&samples[5], predicted_continue),
        (false, false)
    );
    assert_eq!(metrics.confusion[0][1], 1);
    assert_eq!(metrics.confusion[1][0], 1);
    assert_eq!(metrics.full.iter().sum::<usize>(), 0);
}

#[test]
fn conditioning_natural_schedule_preserves_prior_and_exact_matched_sample_budget() {
    let samples = conditioning_schedule_fixture();
    let references: Vec<_> = samples.iter().collect();
    let schedule = conditioning_schedule(&references, false);
    let balanced = conditioning_schedule(&references, true);
    let mut counts = [0usize; 7];
    for index in &schedule {
        counts[*index] += 1;
    }
    assert_eq!(counts[6], 0);
    assert!(counts[..6].iter().max().expect("max") - counts[..6].iter().min().expect("min") <= 1);
    assert_eq!(schedule.len(), balanced.len());
    assert_eq!(schedule.len() / 64, 680);
    assert!(schedule != balanced);
}

#[test]
#[should_panic(expected = "conditioning requires training rows")]
fn conditioning_schedule_rejects_validation_only_corpora() {
    let samples = conditioning_schedule_fixture();
    conditioning_schedule(&[&samples[6]], true);
}

#[test]
#[should_panic(expected = "conditioning weight factor must be 64 or 1/64")]
fn conditioning_parameter_transform_rejects_unplanned_factors() {
    let model = PolicyModel::fresh(10089500).expect("model");
    conditioning_parameters(&model, 32.0);
}

#[test]
fn matched_contract_prediction_audit_handles_the_model_batch_boundary_without_dropping_rows() {
    let [sample, _] = sample_pair();
    let model = PolicyModel::fresh(10089200).expect("model");
    let samples = vec![&sample; crate::MODEL_MAX_BATCH + 1];
    let predictions = history_predictions(&model, &samples);
    assert_eq!(predictions.len(), samples.len());
    assert_eq!(predictions.first(), predictions.last());
}

#[test]
fn matched_contract_frame_audit_hash_covers_all_72_global_features() {
    let [sample, _] = sample_pair();
    let original = frame_sha256(sample.frame());
    let mut changed = sample.frame().clone();
    changed.global[71] = 1.0;
    assert_ne!(original, frame_sha256(&changed));
    assert_eq!(original.len(), 64);
}

#[test]
fn matched_contract_sha256_encoding_matches_the_known_empty_digest() {
    let digest = digest_hex(Sha256::digest([]).into());
    assert_eq!(
        digest,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(digest_hex([0; 32]), "0".repeat(64));
}

fn sample_pair() -> [ImitationSample; 2] {
    let mut arena = observed_teacher_game(10089000, 0);
    let (frame, space) = prepare_neural_observer_sample(&mut arena.seats[1]).expect("frame");
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Training, 10089000, 0, space.tick(), &frame)
            .expect("identity");
    let baseline = ImitationSample::teacher(
        frame.clone(),
        &space,
        crate::StructuredAction::Continue,
        identity,
    )
    .expect("baseline");
    let mut enriched = frame;
    enriched.global[59..64].copy_from_slice(&[1.0, 0.02, 0.5, 0.5, 0.0625]);
    let enriched = ImitationSample::teacher(
        enriched,
        &space,
        crate::StructuredAction::Continue,
        identity,
    )
    .expect("enriched");
    assert_eq!(baseline.identity(), enriched.identity());
    assert_eq!(baseline.target(), enriched.target());
    [baseline, enriched]
}

#[test]
fn matched_contract_pools_have_identical_rows_and_export_rejects_history() {
    let mut pools = std::array::from_fn(|_| new_history_pool());
    for (pool, sample) in pools.iter_mut().zip(sample_pair()) {
        assert!(pool.push(sample).expect("push").is_none());
    }
    assert_matched_pools(&pools);
    assert_eq!(validate_baseline_pool(&pools[0]), Ok(()));
    assert_eq!(
        validate_baseline_pool(&pools[1]),
        Err("enriched history cannot be exported under F12")
    );
}

#[test]
fn matched_contract_two_zero_row_models_have_equal_initial_outputs() {
    let mut pools = std::array::from_fn(|_| new_history_pool());
    for (pool, sample) in pools.iter_mut().zip(sample_pair()) {
        pool.push(sample).expect("row");
    }
    let models = std::array::from_fn(|_| PolicyModel::fresh(10089200).expect("model"));
    let parameters = zero_probe_input_weights(&models[0]);
    for model in &models {
        model
            .import_parameters(&parameters)
            .expect("equal baseline");
    }
    assert_initial_outputs(&models, &pools);
}

#[test]
fn matched_contract_continue_correctness_never_inflates_noncontinue_metrics() {
    let [sample, _] = sample_pair();
    let mut metrics = HistoryMetrics::default();
    let correct = target_prediction(sample.target());
    assert_eq!(metrics.record(&sample, correct), (true, true));
    let wrong = BehavioralPrediction {
        kind: ActionKind::Stop.index(),
        ..correct
    };
    assert_eq!(metrics.record(&sample, wrong), (false, false));
    assert_eq!((metrics.total, metrics.kind, metrics.full), (2, 1, 1));
    assert_eq!(
        (
            metrics.noncontinue,
            metrics.noncontinue_kind,
            metrics.noncontinue_full
        ),
        (0, 0, 0)
    );
}

#[test]
fn matched_contract_noncontinue_kind_match_does_not_imply_full_match() {
    let mut arena = observed_teacher_game(10089000, 0);
    let (frame, space) = prepare_neural_observer_sample(&mut arena.seats[1]).expect("frame");
    let identity =
        SampleIdentity::from_frame(SeedNamespace::Training, 10089000, 0, space.tick(), &frame)
            .expect("identity");
    let sample = ImitationSample::teacher(
        frame,
        &space,
        crate::StructuredAction::Stop {
            unit: crate::ControlledUnit::Hero,
        },
        identity,
    )
    .expect("Stop sample");
    let mut metrics = HistoryMetrics::default();
    let prediction = BehavioralPrediction {
        controlled: Some(1),
        ..target_prediction(sample.target())
    };
    assert_eq!(metrics.record(&sample, prediction), (true, false));
    assert_eq!(metrics.noncontinue, 1);
    assert_eq!(metrics.noncontinue_kind, 1);
    assert_eq!(metrics.noncontinue_full, 0);
}

#[test]
fn matched_contract_history_age_is_bounded_and_unsent_labels_do_not_change_history() {
    let mut history = SentOrderHistory::default();
    history.sent(
        Some(Request {
            seq: 1,
            unit: None,
            order: Order::Cast {
                slot: AbilitySlot(2),
                target: bota_proto::Target::None,
            },
        }),
        ActionKind::Cast,
        1,
    );
    let previous = history.features(4);
    history.sent(None, ActionKind::Learn, 4);
    assert_eq!(history.features(4), previous);
    assert_eq!(history.features(151)[1], 1.0);
    assert_eq!(history.features(108900)[1], 1.0);
    assert_eq!(previous[4], 3.0 / 16.0);
}

#[test]
fn matched_contract_history_preserves_legacy_teacher_requests_and_uses_only_previous_send() {
    let model = PolicyModel::fresh(10089200).expect("model");
    for side in 0..2 {
        let mut reference =
            build_environment(10089000, 10089110, MapId(0), side, 0, OpponentSpec::Teacher)
                .expect("reference");
        let mut observed = observed_teacher_game(10089000, side);
        let mut source_random = PpoRng::new(10089100);
        let mut target_random = source_random.clone();
        let mut history = SentOrderHistory::default();
        for _ in 0..16 {
            let source_choice = sample_policy(&model, &mut source_random, &mut reference)
                .expect("reference choice");
            let expected = requests_for_decision(&mut reference, &source_choice)
                .expect("legacy Teacher requests");
            let (frame, space) = prepare_policy_sample(&mut observed).expect("Candidate sample");
            let choice = model
                .sample(&frame, &space, &mut target_random)
                .expect("matched choice");
            let previous = history.features(space.tick());
            let (actual, samples) = history_samples(
                &mut observed,
                &choice,
                &space,
                10089000,
                SeedNamespace::Training,
                &mut history,
            )
            .expect("observed labels");
            assert_eq!(samples[1].frame().global[59..64], previous);
            assert_eq!(samples[0].identity(), samples[1].identity());
            assert_eq!(samples[0].target(), samples[1].target());
            assert_eq!(expected, actual);
            assert_eq!(source_random, target_random);
            let expert = 1 - side;
            assert_eq!(
                reference.seats[expert].teacher,
                observed.seats[expert].teacher
            );
            assert_eq!(
                reference.seats[expert].persistence,
                observed.seats[expert].persistence
            );
            assert_eq!(
                reference.seats[expert].readiness,
                observed.seats[expert].readiness
            );
            assert_eq!(
                reference.seats[expert].sequence,
                observed.seats[expert].sequence
            );
            advance_interval(&mut reference, expected, 3).expect("reference ticks");
            advance_interval(&mut observed, actual, 3).expect("observed ticks");
            for index in 0..2 {
                assert_eq!(
                    reference.seats[index].tracker.latest_summary(),
                    observed.seats[index].tracker.latest_summary()
                );
                assert_eq!(
                    reference.seats[index].rejections,
                    observed.seats[index].rejections
                );
            }
        }
    }
}
