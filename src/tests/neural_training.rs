use super::*;
use crate::Wire;

#[path = "input_transfer.rs"]
mod input_transfer;

#[test]
fn current_map2_training_configuration_is_accepted() {
    let config = NeuralTrainingConfig::default();
    validate_config(&config).expect("current schema");
    assert_eq!(config.tick_limit, crate::MAP2_TICK_CAP);
}

#[test]
fn current_initialization_rejects_wrong_model_and_feature_hashes() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!(
            "neural-incompatible-current-{}",
            std::process::id()
        ));
    fs::create_dir(&directory).expect("directory");
    let model = PolicyModel::fresh(87).expect("current model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("save");
    let path = directory.join("drysua.weights.safetensors");
    let original = fs::read(&path).expect("weights");
    for hash in [crate::MODEL_SCHEMA_HASH, crate::FEATURE_SCHEMA_HASH] {
        let needle = hash.to_string();
        let mut bytes = original.clone();
        let index = bytes
            .windows(needle.len())
            .position(|window| window == needle.as_bytes())
            .expect("metadata hash");
        bytes[index] = if bytes[index] == b'1' { b'2' } else { b'1' };
        fs::write(&path, bytes).expect("wrong hash");
        let config = NeuralTrainingConfig {
            initial_weights: Some(directory.clone()),
            ..NeuralTrainingConfig::default()
        };
        assert_eq!(
            initialize_model(&config)
                .err()
                .expect("incompatible artifact")
                .to_string(),
            "checkpoint schema does not match this build"
        );
    }
    fs::remove_dir_all(directory).expect("cleanup");
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
        map: MapId(2),
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
    let data = vec![0u8; 1_684_724 * 4];
    let tensor = TensorView::new(Dtype::F32, vec![1_684_724], &data).expect("tensor");
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
                map: MapId(2),
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
fn map2_phase_bins_cover_pregame_and_four_equal_gameplay_intervals() {
    for (tick, expected) in [
        (1, 0),
        (PREGAME_TICKS - 1, 0),
        (PREGAME_TICKS, 1),
        (PREGAME_TICKS + GAMEPLAY_PHASE_TICKS - 1, 1),
        (PREGAME_TICKS + GAMEPLAY_PHASE_TICKS, 2),
        (PREGAME_TICKS + 2 * GAMEPLAY_PHASE_TICKS - 1, 2),
        (PREGAME_TICKS + 2 * GAMEPLAY_PHASE_TICKS, 3),
        (PREGAME_TICKS + 3 * GAMEPLAY_PHASE_TICKS - 1, 3),
        (PREGAME_TICKS + 3 * GAMEPLAY_PHASE_TICKS, 4),
        (MAP2_TICK_CAP, 4),
    ] {
        assert_eq!(phase(tick), expected, "tick={tick}");
    }
}

#[test]
fn map2_train_selection_and_final_namespaces_are_distinct() {
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
    assert_eq!(config.tick_limit, crate::MAP2_TICK_CAP);
    assert!(
        validate_config(&NeuralTrainingConfig {
            training_games: 21,
            ..config
        })
        .is_err()
    );
}

#[test]
fn learner_uses_network_choice_in_pregame_even_when_teacher_would_buy() {
    let (_, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
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
    assert!(data.pool.binding().scope.contains_map(MapId(2)));
    assert!(!data.pool.binding().scope.contains_map(MapId(0)));
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
fn stratified_budgets_keep_total_pool_at_32768_and_continue_at_most_one_third() {
    assert_eq!(CAPACITIES, [16384, 6144, 6144]);
    assert_eq!(DAGGER_CAPACITY, 4096);
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
        map: MapId(2),
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
    let sample = &collection.reservoir.samples[0];
    assert_eq!(sample.source(), crate::ImitationSource::Dagger);
    assert_eq!(sample.learner_action(), Some(learner));
    assert_eq!(sample.teacher_action(), expert);
    assert_eq!(sample.identity().namespace(), SeedNamespace::Training);
    assert_eq!(sample.identity().map(), MapId(2));
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
                && sample.identity().map() == MapId(2)
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
fn broad_configuration_accepts_twenty_matches_and_45_minutes() {
    let config = NeuralTrainingConfig {
        seed: 9950000,
        training_games: 20,
        wall_time: Duration::from_secs(2700),
        ..NeuralTrainingConfig::default()
    };
    validate_config(&config).expect("bounded broad collection");
    assert_eq!(namespaces(&config).expect("seeds").training().len(), 21);
}

#[test]
fn branch_keys_separate_body_and_slot_before_target_classification() {
    use crate::{ActionTarget, ControlledUnit, EntityIndex};
    let action = StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: bota_proto::ItemSlot(0),
        target: ActionTarget::Entity(EntityIndex(1)),
    };
    let (hero, pointer) = BranchKey::new(action, 0);
    let (courier, _) = BranchKey::new(
        StructuredAction::Use {
            unit: ControlledUnit::Courier,
            slot: bota_proto::ItemSlot(0),
            target: pointer,
        },
        0,
    );
    let (other_slot, _) = BranchKey::new(
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: bota_proto::ItemSlot(1),
            target: pointer,
        },
        0,
    );
    assert_ne!(hero, courier);
    assert_ne!(hero, other_slot);
    assert_eq!(hero.slot, 1);
    assert_eq!(courier.body, 1);
}

#[test]
fn expanded_pool_paired_frame_estimate_leaves_room_below_eight_gib() {
    let paired = std::mem::size_of::<ImitationSample>() + std::mem::size_of::<FeatureFrame>();
    let storage = paired * MAX_IMITATION_SAMPLES;
    assert!(storage > 3 * 1024 * 1024 * 1024usize);
    assert!(storage < 5 * 1024 * 1024 * 1024usize);
    let transient = std::mem::size_of::<ImitationSample>() * CAPACITIES[0] * 2;
    assert!(storage + transient < 8 * 1024 * 1024 * 1024usize);
}

#[test]
fn late_rare_branch_displaces_common_rows_without_losing_observed_counts() {
    let common = BranchKey {
        kind: 7,
        body: 0,
        target: 0,
        slot: 1,
        point: 0,
        cell: 2,
    };
    let rare = BranchKey {
        body: 1,
        slot: 3,
        ..common
    };
    let mut selector = BranchSelector::new(64, 23);
    for _ in 0..10000 {
        selector.destination(common).expect("common");
    }
    for _ in 0..20 {
        assert!(selector.destination(rare).expect("rare").is_some());
    }
    for _ in 0..10000 {
        selector.destination(common).expect("later common");
    }
    assert_eq!(selector.counts[&rare].1.len(), 20);
    assert_eq!(selector.counts[&rare].0, 20);
    assert_eq!(selector.counts[&common].0, 20000);
    assert_eq!(
        selector
            .counts
            .values()
            .map(|entry| entry.1.len())
            .sum::<usize>(),
        64
    );
}

#[test]
fn conditional_stratum_limit_rejects_the_next_key_with_a_specific_error() {
    let mut selector = BranchSelector::new(16384, 29);
    for index in 0..8192 {
        selector
            .destination(BranchKey {
                kind: index / 512,
                body: index % 2,
                target: 0,
                slot: index / 2 % 256,
                point: 0,
                cell: 0,
            })
            .expect("bounded stratum");
    }
    let error = selector
        .destination(BranchKey {
            kind: 0,
            body: 2,
            target: 0,
            slot: 0,
            point: 0,
            cell: 0,
        })
        .expect_err("excess stratum");
    assert_eq!(
        error.to_string(),
        "conditional reservoir exceeds 8192 observed strata"
    );
    assert_eq!(selector.counts.len(), 8192);
    assert_eq!(selector.keys.len(), 8192);
}

#[test]
fn branch_reservoir_preserves_rare_courier_and_is_reproducible() {
    let mut first = BranchSelector::new(64, 17);
    let mut second = BranchSelector::new(64, 17);
    let common = BranchKey {
        kind: 2,
        body: 0,
        target: 0,
        slot: 0,
        point: 1,
        cell: 0,
    };
    let rare = BranchKey { body: 1, ..common };
    for index in 0..10000 {
        let key = if index < 20 { rare } else { common };
        assert_eq!(
            first.destination(key).expect("draw"),
            second.destination(key).expect("draw")
        );
    }
    assert_eq!(first.counts[&rare].1.len(), 20);
    assert_eq!(first.keys.len(), 64);
    assert_eq!(first.counts[&common].0, 9980);
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

#[test]
fn map2_neural_training_tick_limits_reject_legacy_caps_and_accept_both_boundaries() {
    for tick_limit in [2, 18_900, MAP2_TICK_CAP] {
        validate_config(&NeuralTrainingConfig {
            tick_limit,
            ..NeuralTrainingConfig::default()
        })
        .expect("bounded Map2 ticks");
    }
    for tick_limit in [0, 1, MAP2_TICK_CAP + 1, 108_900, u32::MAX] {
        let error = validate_config(&NeuralTrainingConfig {
            tick_limit,
            ..NeuralTrainingConfig::default()
        })
        .expect_err("out of Map2 bounds");
        assert_eq!(
            error.to_string(),
            format!(
                "Map2 training requires 1..20 expert games, 1..64 epochs per stage, 0..2 DAgger rounds, 2..{MAP2_TICK_CAP} ticks, <=2700 seconds and nonoverflowing disjoint seeds"
            )
        );
    }
}

#[test]
fn map2_neural_seat_rejects_legacy_maps_with_specific_error_before_tracking() {
    for map in [MapId(0), MapId(1)] {
        let (_, start) = Arena::new(ArenaConfig {
            map,
            seats: 2,
            seed: 9840000,
        })
        .expect("legacy fixture");
        let error = NeuralSeat::new(0, &start.messages[0])
            .err()
            .expect("Map2 only");
        assert_eq!(error.to_string(), "neural training requires Map2");
    }
}

#[test]
fn map2_neural_seat_rejects_missing_match_start_without_panicking() {
    let error = NeuralSeat::new(0, &[]).err().expect("missing start");
    assert_eq!(error.to_string(), "initial MatchStart missing");
}

#[test]
fn map2_order_probe_uses_the_production_constructor_and_keeps_map2_metadata() {
    let (_, start) = Arena::new(ArenaConfig {
        map: crate::MAP2_ID,
        seats: 2,
        seed: 9840000,
    })
    .expect("Map2 arena");
    let model = PolicyModel::fresh(9840000).expect("model");
    let mut probe = NeuralSeatOrderContractProbe::new(0, &start.messages[0]);
    let (frame, _, _) = probe.decide(&model, true);
    assert_eq!(frame.global()[crate::global_feature::MAP_TWO], 1.0);
    assert_eq!(frame.global()[crate::global_feature::MAP_ZERO], 0.0);
}

#[test]
fn map2_actor_budget_includes_pregame_and_ends_at_the_shared_decision_cap() {
    assert_eq!(DECISION_INTERVAL_TICKS, crate::MAP2_DECISION_INTERVAL_TICKS);
    assert_eq!(ACTOR_DECISIONS as usize, crate::MAP2_ACTOR_DECISIONS);
    assert_eq!(ACTOR_DECISIONS * DECISION_INTERVAL_TICKS, MAP2_TICK_CAP);
}

#[test]
fn map2_game_loop_preserves_authoritative_draw_at_the_inclusive_tick_cap() {
    let seed = 9840000;
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        map: MapId(2),
        seats: 2,
        seed,
    })
    .expect("Map2 arena");
    let projected = arena.configure_for_test(|world| world.tick = MAP2_TICK_CAP - 1);
    for (initial, current) in start.messages.iter_mut().zip(projected.messages) {
        initial.truncate(1);
        initial.extend(current);
    }

    let game = run_game_in_arena((arena, start), None, seed, 0, MAP2_TICK_CAP, None, None)
        .expect("only the final cap tick runs");

    assert_eq!(game.ticks, MAP2_TICK_CAP);
    assert_eq!(game.winner, Some(Team::Neutral));
    assert_eq!(game.decisions, [0, 0]);
}
