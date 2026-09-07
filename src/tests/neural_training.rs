use super::*;
use crate::Wire;

#[test]
fn current_m11_training_configuration_is_accepted() {
    validate_config(&NeuralTrainingConfig::default()).expect("current M11/F10");
}

#[test]
fn current_and_legacy_initialization_flags_are_mutually_exclusive() {
    let config = NeuralTrainingConfig {
        initial_weights: Some(PathBuf::from("current")),
        initialize_selected_m10: Some(PathBuf::from("legacy")),
        ..NeuralTrainingConfig::default()
    };
    assert_eq!(
        validate_config(&config)
            .expect_err("ambiguous initialization")
            .to_string(),
        "choose only one of initial_weights and initialize_selected_m10"
    );
}

#[test]
fn expired_deadline_stops_match_with_specific_error() {
    let error =
        run_game(None, 9860300, 0, 31, Some(Instant::now()), None).expect_err("expired deadline");
    assert_eq!(error.to_string(), "neural match deadline exhausted");
}

#[test]
fn current_weights_initialize_new_owned_model_and_optimizer() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!(
            "neural-current-initialization-{}",
            std::process::id()
        ));
    fs::create_dir(&directory).expect("directory");
    let original = PolicyModel::fresh(91).expect("model");
    TrainingArtifact::save_runtime_weights(&original, &directory).expect("weights");
    let config = NeuralTrainingConfig {
        initial_weights: Some(directory.clone()),
        ..NeuralTrainingConfig::default()
    };
    let (model, provenance) = initialize_model(&config).expect("current initialization");
    assert_eq!(
        model.export_parameters().expect("parameters"),
        original.export_parameters().expect("original")
    );
    assert!(matches!(provenance, Initialization::CurrentWeights { .. }));
    assert_eq!(
        model
            .claim_optimizer(AdamConfig::default())
            .expect("new optimizer")
            .step(),
        0
    );
    let invalid = NeuralTrainingConfig {
        initial_weights: None,
        initialize_selected_m10: Some(directory.clone()),
        ..config
    };
    assert_eq!(
        initialize_model(&invalid)
            .err()
            .expect("not legacy")
            .to_string(),
        "checkpoint schema does not match this build"
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
#[ignore = "explicit 615-decision stock trajectory diagnostic; requires DRYSUA_CURRENT_M11"]
fn stock_615_decisions_report_per_kind_accuracy_and_actual_actions() {
    let directory =
        PathBuf::from(std::env::var_os("DRYSUA_CURRENT_M11").expect("current artifact"));
    let model = PolicyModel::fresh(9860400).expect("model");
    TrainingArtifact::load_runtime_weights(&model, &directory).expect("current schema");
    let (mut arena, start) = Arena::new(ArenaConfig {
        map: MapId(0),
        seats: 2,
        seed: 9860400,
    })
    .expect("arena");
    let mut seats = [
        NeuralSeat::new(0, &start.messages[0]).expect("seat"),
        NeuralSeat::new(1, &start.messages[1]).expect("seat"),
    ];
    let mut experts = [Some(Teacher::new()), Some(Teacher::new())];
    let mut totals = [0u64; ActionKind::COUNT];
    let mut kinds = totals;
    let mut exact = totals;
    let mut predicted = totals;
    for _ in 0..1845 {
        let mut requests = [None, None];
        if (arena.tick() - 1).is_multiple_of(3) {
            for (index, seat) in seats.iter_mut().enumerate() {
                let expert = experts[index].as_mut().expect("expert");
                let (target, space) = expert
                    .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                    .expect("label");
                let (action, _) = seat.neural_choice(&model).expect("unmodified prediction");
                totals[target.kind().index()] += 1;
                kinds[target.kind().index()] += u64::from(action.kind() == target.kind());
                exact[target.kind().index()] += u64::from(action == target);
                predicted[action.kind().index()] += 1;
                if let Some(issued) = seat.issue(target, &space).expect("stock trajectory") {
                    expert.note_sent(seat.sequence, issued, space.tick());
                    requests[index] = Some(Request {
                        seq: seat.sequence,
                        unit: issued.unit,
                        order: issued.order,
                    });
                }
            }
        }
        let step = arena.step(&requests).expect("step");
        observe_game_tick(&mut seats, &mut experts, &step.messages, 9860400, &mut None)
            .expect("observe");
    }
    assert_eq!(totals.iter().sum::<u64>(), 1230);
    assert!(totals[ActionKind::Buy.index()] > 0);
    assert!(totals[ActionKind::Learn.index()] > 0);
    let actual = evaluate_pair(
        &model,
        9860400,
        1846,
        Some(Instant::now() + Duration::from_secs(120)),
    )
    .expect("bounded actual pair");
    println!(
        "stock_decisions_per_seat=615 totals={totals:?} kind_correct={kinds:?} exact_correct={exact:?} predicted={predicted:?} continue_baseline={}/1230 actual_neural_games={actual:?}",
        totals[0]
    );
}

#[test]
fn legacy_initialization_flag_rejects_correct_tuple_with_wrong_sha() {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!("neural-wrong-sha-{}", std::process::id()));
    fs::create_dir(&directory).expect("directory");
    let data = vec![0u8; crate::MODEL_PARAMETER_COUNT * 4];
    let tensor =
        TensorView::new(Dtype::F32, vec![crate::MODEL_PARAMETER_COUNT], &data).expect("tensor");
    let metadata = [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "9669721049329356661"),
        ("model_schema_hash", "720439888929233033"),
        ("ppo_schema_version", "18"),
        ("ppo_schema_hash", "6877503070358232325"),
        ("ppo_rules_audit_version", "15"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect();
    let bytes = serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture");
    fs::write(directory.join("drysua.weights.safetensors"), &bytes).expect("write");
    let config = NeuralTrainingConfig {
        initialize_selected_m10: Some(directory.clone()),
        ..NeuralTrainingConfig::default()
    };
    assert_eq!(
        initialize_model(&config)
            .err()
            .expect("wrong SHA")
            .to_string(),
        "checkpoint tensor contract has invalid selected M10 training source SHA-256"
    );
    assert_eq!(
        fs::read(directory.join("drysua.weights.safetensors")).expect("unchanged"),
        bytes
    );
    fs::remove_dir_all(directory).expect("cleanup");
}

#[path = "neural_overfit.rs"]
mod neural_overfit;

#[test]
fn actual_tcp_neural_orders_match_builtin_neural_orders_without_teacher_override() {
    let seed = 9830000;
    let model = PolicyModel::fresh(23).expect("model");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address").to_string();
    let server = std::thread::spawn(move || {
        bota_server::game_loop::run(
            listener,
            bota_server::game_loop::ServerOpts {
                mode: bota_proto::TickMode::Lockstep,
                tick_rate: 30,
                players: 2,
                replay: None,
                seed,
                map: MapId(0),
                ack_timeout_ticks: 300,
            },
        )
    });
    let (link, seated) =
        crate::Link::join_with_timeout(&address, "neural", Duration::from_secs(10))
            .expect("neural join");
    assert_eq!(seated.slot, SlotId(0));
    let (mut opponent, other) =
        crate::Link::join_with_timeout(&address, "expert", Duration::from_secs(10))
            .expect("expert join");
    let expert = std::thread::spawn(move || crate::play_teacher_on(&mut opponent, other, Some(31)));
    let mut wire = CapturedWire {
        link,
        tick: 0,
        hash: DefaultHasher::new(),
    };
    let outcome = crate::play_neural_on(&mut wire, seated, Some(31), &model).expect("pure TCP");
    let fingerprint = wire.hash.finish();
    drop(wire);
    expert
        .join()
        .expect("expert worker")
        .expect("expert outcome");
    server
        .join()
        .expect("server worker")
        .expect("server outcome");
    let expected = run_game(Some(&model), seed, 0, 31, None, None).expect("builtin");
    assert_eq!(fingerprint, expected.order_fingerprints[0]);
    assert_eq!(outcome.orders, expected.orders[0]);
    assert_eq!(outcome.decisions, expected.decisions[0]);
    assert_eq!(outcome.rejections, expected.rejections[0]);
}

struct CapturedWire {
    link: crate::Link,
    tick: u32,
    hash: DefaultHasher,
}

impl Wire for CapturedWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        let message = self.link.hear()?;
        if let Some(ServerMsg::Snapshot { view }) = &message {
            self.tick = view.tick;
        }
        Ok(message)
    }
    fn order(
        &mut self,
        unit: Option<bota_proto::EntityId>,
        order: bota_proto::Order,
    ) -> std::io::Result<u32> {
        (self.tick, unit, order).hash(&mut self.hash);
        self.link.order(unit, order)
    }
    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        self.link.acknowledge(tick)
    }
}

#[test]
fn stratified_selector_is_bounded_and_reproducible() {
    let mut first = CoverageSelector::new(32, 17, false);
    let mut second = CoverageSelector::new(32, 17, false);
    let selected = (0..1000)
        .map(|index| first.destination(index % 10).expect("draw"))
        .collect::<Vec<_>>();
    let repeated = (0..1000)
        .map(|index| second.destination(index % 10).expect("draw"))
        .collect::<Vec<_>>();
    assert_eq!(selected, repeated);
    assert_eq!(first.seen, 1000);
    assert!(selected.iter().flatten().all(|index| *index < 32));
    assert!(selected[32..].iter().any(Option::is_none));
}

#[test]
fn phase_bins_include_pregame_and_the_end_of_a_full_map0_game() {
    assert_eq!(phase(1), 0);
    assert_eq!(phase(899), 0);
    assert_eq!(phase(900), 1);
    assert_eq!(phase(108900), 4);
}

#[test]
fn train_selection_and_final_namespaces_are_distinct_and_map0_only() {
    let config = NeuralTrainingConfig::default();
    let namespaces = namespaces(&config).expect("namespaces");
    assert!(
        !namespaces
            .training()
            .contains(&config.seed.saturating_add(100))
    );
    assert!(
        !namespaces
            .validation()
            .contains(&config.seed.saturating_add(200))
    );
    assert_eq!(config.tick_limit, 108900);
    assert!(
        validate_config(&NeuralTrainingConfig {
            training_games: 9,
            ..config
        })
        .is_err()
    );
}

#[test]
fn learner_uses_network_choice_in_pregame_even_when_teacher_would_buy() {
    let (_, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(0),
        seed: 9830000,
    })
    .expect("arena");
    let mut seat = NeuralSeat::new(0, &start.messages[0]).expect("seat");
    let model = PolicyModel::fresh(19).expect("model");
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if name == "kind.bias" {
            parameters[offset + ActionKind::Hold.index()] = 20.0;
        }
        offset += shape.iter().product::<usize>();
    }
    model
        .import_parameters(&parameters)
        .expect("forced network choice");
    let (action, space) = seat.neural_choice(&model).expect("pure choice");
    let expert = Teacher::new()
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .expect("expert")
        .0;
    assert_eq!(action.kind(), ActionKind::Hold);
    assert_ne!(action.kind(), expert.kind());
    let issued = seat
        .issue(action, &space)
        .expect("issue")
        .expect("wire order");
    assert!(matches!(
        issued.order,
        bota_proto::Order::Attack {
            target: bota_proto::Target::None
        }
    ));
    assert_eq!(seat.decisions, 1);
}

#[test]
fn expert_collection_keeps_both_seats_and_no_eviction_of_evaluation_samples() {
    let config = NeuralTrainingConfig {
        training_games: 1,
        tick_limit: 31,
        ..NeuralTrainingConfig::default()
    };
    let data = collect_dataset(&config, None).expect("dataset");
    assert_eq!(data.games.len(), 3);
    assert!(data.pool.len() <= 60);
    assert!(!data.pool.is_empty());
    assert!(data.pool.binding().scope.contains_map(MapId(0)));
    assert!(!data.pool.binding().scope.contains_map(MapId(1)));
    assert!(
        data.games
            .iter()
            .all(|game| game.winner.is_none() && game.ticks == 31)
    );
}

#[test]
fn rare_labels_are_all_kept_before_the_kind_budget_fills() {
    let mut selector = CoverageSelector::new(128, 19, true);
    for index in 0..100 {
        assert_eq!(
            selector.destination(index % 2).expect("rare sample"),
            Some(index)
        );
    }
    assert_eq!(selector.cells.len(), 100);
    assert_eq!(selector.kept[0], 50);
    assert_eq!(selector.kept[1], 50);
}

#[test]
fn late_common_samples_cannot_erase_opening_or_seat_phase_coverage() {
    let mut selector = CoverageSelector::new(384, 21, true);
    for cell in 0..10 {
        for _ in 0..80 {
            selector.destination(cell).expect("initial coverage");
        }
    }
    for _ in 0..10000 {
        selector.destination(9).expect("late arrivals");
    }
    assert!(selector.kept[0] >= 64);
    assert!(selector.kept[1] >= 64);
    for count in &selector.kept[2..] {
        assert!(*count >= 4);
    }
    assert_eq!(selector.cells.len(), 384);
}

#[test]
fn stratified_budgets_keep_total_pool_at_9216_and_continue_at_most_one_third() {
    assert_eq!(BASE_KIND_CAPACITIES.iter().sum::<usize>(), 4096);
    assert_eq!(
        CAPACITIES.iter().sum::<usize>() + DAGGER_CAPACITY,
        MAX_IMITATION_SAMPLES
    );
    let config = NeuralTrainingConfig {
        training_games: 1,
        tick_limit: 901,
        ..NeuralTrainingConfig::default()
    };
    let data = collect_dataset(&config, None).expect("stratified dataset");
    for namespace in [
        SeedNamespace::Training,
        SeedNamespace::Validation,
        SeedNamespace::Promotion,
    ] {
        let samples = (0..data.pool.len())
            .filter_map(|index| data.pool.get(index))
            .filter(|sample| sample.identity().namespace() == namespace)
            .collect::<Vec<_>>();
        let continued = samples
            .iter()
            .filter(|sample| sample.teacher_action().kind() == ActionKind::Continue)
            .count();
        assert!(continued * 3 <= samples.len());
        for side in [crate::ImitationSide::Radiant, crate::ImitationSide::Dire] {
            assert!(
                samples.iter().any(|sample| sample.side() == side
                    && sample.teacher_action().kind() == ActionKind::Buy)
            );
            assert!(samples.iter().any(|sample| sample.side() == side
                && sample.teacher_action().kind() == ActionKind::Learn));
        }
    }
}

#[test]
fn selection_ties_use_noncontinue_full_agreement_not_checkpoint_age() {
    let mut weak = OfflineEvaluation::default();
    weak.overall.families[0] = crate::AgreementCount {
        matching: 900,
        total: 900,
    };
    weak.overall.families[2] = crate::AgreementCount {
        matching: 0,
        total: 100,
    };
    let mut stronger = weak.clone();
    stronger.overall.families[2].matching = 75;
    assert!(selection_rank((1, 0, 3, 100), &stronger) > selection_rank((1, 0, 3, 100), &weak));
    assert!(
        selection_rank((2, 0, -64, -100000), &weak) > selection_rank((1, 0, 64, 100000), &stronger)
    );
}

#[test]
fn dagger_labels_never_replace_the_network_order_and_track_only_real_sends() {
    let (_, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(0),
        seed: 9840000,
    })
    .expect("arena");
    let mut seat = NeuralSeat::new(0, &start.messages[0]).expect("seat");
    let space =
        ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness).expect("space");
    let learner = StructuredAction::Hold {
        unit: crate::ControlledUnit::Hero,
    };
    let mut reservoir = Reservoir::new(1024, 31);
    let mut collection = Collection {
        reservoir: &mut reservoir,
        namespace: SeedNamespace::Training,
        labeler: Some(Teacher::new()),
    };
    let mut expected = Teacher::new();
    let expert = expected
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .expect("label")
        .0;
    collect_decision(&mut collection, &mut seat, &space, learner, 9840050, true)
        .expect("DAgger label");
    let issued = seat
        .issue(learner, &space)
        .expect("issue")
        .expect("real send");
    note_labeler_send(&mut collection, seat.sequence, issued, space.tick());
    expected.note_sent(seat.sequence, issued, space.tick());
    assert_eq!(collection.labeler.as_ref().expect("labeler"), &expected);
    let sample = &collection.reservoir.samples[expert.kind().index()][0];
    assert_eq!(sample.source(), crate::ImitationSource::Dagger);
    assert_eq!(sample.learner_action(), Some(learner));
    assert_eq!(sample.teacher_action(), expert);
    assert_eq!(sample.identity().namespace(), SeedNamespace::Training);
    assert!(matches!(
        issued.order,
        bota_proto::Order::Attack {
            target: bota_proto::Target::None
        }
    ));
    assert_ne!(learner.kind(), expert.kind());
}

#[test]
fn dagger_collection_does_not_change_either_actual_game_order_stream() {
    let model = PolicyModel::fresh(41).expect("v10 model");
    let deadline = None;
    let plain = run_game(Some(&model), 9840050, 0, 901, deadline, None).expect("plain neural");
    let mut reservoir = Reservoir::new(1024, 43);
    let labeled = run_game(
        Some(&model),
        9840050,
        0,
        901,
        deadline,
        Some((&mut reservoir, SeedNamespace::Training)),
    )
    .expect("shadow labeled neural");
    assert_eq!(labeled.order_fingerprints, plain.order_fingerprints);
    assert_eq!(labeled.orders, plain.orders);
    assert_eq!(labeled.rejections, plain.rejections);
    let samples = reservoir.into_samples().expect("bounded samples");
    assert!(!samples.is_empty());
    assert!(
        samples
            .iter()
            .all(|sample| sample.source() == crate::ImitationSource::Dagger
                && sample.identity().namespace() == SeedNamespace::Training
                && sample.side() == crate::ImitationSide::Radiant)
    );
}

#[test]
fn dagger_rejects_selection_or_final_namespace_before_playing() {
    let model = PolicyModel::fresh(47).expect("model");
    for namespace in [SeedNamespace::Validation, SeedNamespace::Promotion] {
        let mut reservoir = Reservoir::new(1024, 47);
        let error = run_game(
            Some(&model),
            9840050,
            0,
            31,
            None,
            Some((&mut reservoir, namespace)),
        )
        .expect_err("held-out contamination");
        assert_eq!(
            error.to_string(),
            "DAgger may collect only Training namespace states"
        );
    }
}

#[test]
fn dagger_append_preserves_held_out_binding_and_trains_only_the_expanded_train_set() {
    let config = NeuralTrainingConfig {
        training_games: 1,
        tick_limit: 31,
        ..NeuralTrainingConfig::default()
    };
    let deadline = None;
    let mut data = collect_dataset(&config, deadline).expect("base data");
    let held = (0..data.pool.len())
        .filter_map(|index| data.pool.get(index))
        .filter(|sample| sample.identity().namespace() != SeedNamespace::Training)
        .map(ImitationSample::identity)
        .collect::<Vec<_>>();
    let mut session = TrainingSession::new(&config, &data).expect("bound trainer");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!("neural-dagger-binding-{}", std::process::id()));
    fs::create_dir(&directory).expect("directory");
    append_dagger(&config, &mut data, &session.actor, 0, deadline, &directory).expect("append");
    session
        .trainer
        .rebind_pool(&data.pool)
        .expect("same lineage");
    let epoch = session
        .trainer
        .train_epoch(&session.model, &data.pool)
        .expect("new train rows");
    assert!(epoch.order.iter().all(|index| {
        data.pool
            .get(*index)
            .expect("sample")
            .identity()
            .namespace()
            == SeedNamespace::Training
    }));
    assert!(held.iter().all(|identity| {
        (0..data.pool.len())
            .any(|index| data.pool.get(index).expect("sample").identity() == *identity)
    }));
    assert!(data.pool.statistics().dagger > 0);
    OfflineEvaluation::evaluate_held_out(&session.actor, &data.pool, data.coverage[1].clone())
        .expect("held-out binding unchanged");
    fs::remove_dir_all(directory).expect("cleanup");
}

#[test]
fn corrective_reservoir_reserves_more_buy_and_cast_labels_without_exceeding_2048() {
    let reservoir = Reservoir::dagger(2048, 51);
    assert_eq!(DAGGER_KIND_CAPACITIES.iter().sum::<usize>(), 2048);
    assert_eq!(reservoir.selectors[ActionKind::Buy.index()].capacity, 384);
    assert_eq!(reservoir.selectors[ActionKind::Cast.index()].capacity, 384);
    assert_eq!(
        reservoir
            .selectors
            .iter()
            .map(|selector| selector.capacity)
            .sum::<usize>(),
        2048
    );
}

#[test]
fn parallel_pure_evaluation_matches_sequential_order_streams() {
    let model = PolicyModel::fresh(53).expect("model");
    let deadline = None;
    let parallel = evaluate_pair(&model, 9840100, 301, deadline).expect("parallel pair");
    for (side, game) in parallel.iter().enumerate() {
        let sequential =
            run_game(Some(&model), 9840100, side, 301, deadline, None).expect("sequential");
        assert_eq!(game.order_fingerprints, sequential.order_fingerprints);
        assert_eq!(game.orders, sequential.orders);
        assert_eq!(game.winner, sequential.winner);
        assert_eq!(game.ticks, sequential.ticks);
    }
}
