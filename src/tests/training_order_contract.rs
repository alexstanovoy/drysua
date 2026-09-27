use super::*;

fn constant_model(kind: ActionKind) -> PolicyModel {
    let model = PolicyModel::fresh(10_093_001).expect("instrument");
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if name == "kind.bias" {
            parameters[offset + kind.index()] = 1000.0;
        }
        if name == "controlled.bias" {
            parameters[offset] = 1000.0;
        }
        offset += shape.iter().product::<usize>();
    }
    assert_eq!(offset, parameters.len());
    model
        .import_parameters(&parameters)
        .expect("finite diagnostic policy");
    model
}

#[test]
fn frozen_and_shared_policy_dispatch_match_candidate_requests_and_rng_through_native_ticks() {
    let model = constant_model(ActionKind::AttackUnit);
    for side in 0..2 {
        for shared in [false, true] {
            let policy = PolicySnapshot::capture(&model, 0).expect("snapshot");
            let spec = if shared {
                OpponentSpec::SharedPolicy(Arc::new(policy.instantiate().expect("model")))
            } else {
                OpponentSpec::Policy(policy)
            };
            let mut environment =
                build_environment(10_093_000, 1, MapId(0), 1 - side, 0, spec).expect("environment");
            let (mut arena, start) = Arena::new(ArenaConfig {
                seats: 2,
                map: MapId(0),
                seed: 10_093_000,
            })
            .expect("arena");
            let projected = arena.configure_for_test(|world| {
                for side in 0..2 {
                    let hero = world.seats[side].unit.expect("hero");
                    world.transform.get_mut(hero).expect("position").pos =
                        bota_proto::Vec2::from_ints(8600 + side as i32 * 200, 8900);
                    world
                        .modifiers
                        .insert(hero, bota_server::game::Modifiers::default());
                }
            });
            let mut initial = vec![start.messages[side][0].clone()];
            initial.extend(projected.messages[side].clone());
            let mut candidate = setup_seat(side, &initial).expect("candidate");
            let mut frozen = setup_seat(side, &initial).expect("opponent");
            let mut sampling = PpoRng::new(1);
            for _ in 0..8 {
                let (frame, space) =
                    prepare_neural_seat_policy_sample(&mut candidate).expect("candidate input");
                let choice = model.sample(&frame, &space, &mut sampling).expect("choice");
                let request = policy_request_in_space(&mut candidate, &choice, &space)
                    .expect("candidate dispatch");
                assert_eq!(
                    opponent_request(&mut frozen, &mut environment.opponent)
                        .expect("frozen dispatch"),
                    request
                );
                let OpponentRuntime::Policy { rng, .. } = &environment.opponent else {
                    panic!("neural opponent")
                };
                assert_eq!(rng, &sampling);
                let mut requests = [None, None];
                requests[side] = request;
                let step = arena.step(&requests).expect("native tick");
                for seat in [&mut candidate, &mut frozen] {
                    observe_messages(seat, &step.messages[side]).expect("native observation");
                }
            }
            assert!(candidate.sequence > 0);
            assert_eq!(candidate.local.active_order(), frozen.local.active_order());
        }
    }
}

#[test]
fn observer_collection_preserves_teacher_gameplay_and_restart_roles() {
    let build = || {
        build_environment(10_093_004, 1, MapId(0), 0, 0, OpponentSpec::Teacher)
            .expect("environment")
    };
    let mut observed = build();
    let mut reference = build();
    for seat in &mut observed.seats {
        prepare_neural_observer_sample(seat).expect("observer from start");
    }
    for _ in 0..16 {
        let requests = [&mut observed, &mut reference].map(|environment| {
            environment
                .seats
                .iter_mut()
                .map(|seat| teacher_request(seat).expect("Teacher request"))
                .collect::<Vec<_>>()
        });
        assert_eq!(requests[0], requests[1]);
        for (environment, requests) in [&mut observed, &mut reference].into_iter().zip(requests) {
            assert!(
                advance_interval(environment, requests, 3)
                    .expect("native interval")
                    .winner
                    .is_none()
            );
        }
        for (source, target) in observed.seats.iter_mut().zip(&reference.seats) {
            prepare_neural_observer_sample(source).expect("complete tick");
            assert_eq!(source.tracker.current(), target.tracker.current());
            assert_eq!(source.persistence, target.persistence);
            assert_eq!(source.teacher, target.teacher);
            assert_eq!(source.readiness, target.readiness);
            assert_eq!(source.sequence, target.sequence);
        }
    }
    let before = observed.seats[0].order_bookkeeping;
    assert_eq!(
        prepare_neural_seat_policy_sample(&mut observed.seats[0])
            .err()
            .expect("no hot switch")
            .to_string(),
        "invalid PPO transition: neural observer cannot become a candidate"
    );
    assert_eq!(observed.seats[0].order_bookkeeping, before);
    let seat = &mut reference.seats[1];
    assert!(seat.sequence > 0);
    let before = (seat.persistence, seat.local, seat.pending_active);
    assert_eq!(
        prepare_neural_observer_sample(seat)
            .err()
            .expect("lost prefix")
            .to_string(),
        "invalid PPO transition: neural bookkeeping must start before the first actual request"
    );
    assert_eq!((seat.persistence, seat.local, seat.pending_active), before);
    restart_environment(&mut observed).expect("restart observers");
    for seat in &observed.seats {
        assert!(seat.order_bookkeeping.is_neural());
        assert!(!seat.order_bookkeeping.is_candidate());
        assert_eq!((seat.sequence, seat.pending_active), (0, None));
    }
}

#[test]
fn zero_warmup_initializes_candidate_before_cleanup_and_restart_preserves_roles() {
    let mut environments = vec![
        build_environment(10_093_000, 1, MapId(0), 0, 0, OpponentSpec::Weak).expect("environment"),
    ];
    observe_neural_seat_orders(&mut environments[0].seats[1]).expect("observer before cleanup");
    warmup_training_environments(
        &mut environments,
        &[0],
        &constant_model(ActionKind::Continue),
        3,
    )
    .expect("zero warmup");
    let environment = &mut environments[0];
    assert!(environment.seats[0].sequence > 0);
    prepare_policy_sample(environment).expect("candidate before legacy cleanup");
    observe_neural_seat_orders(&mut environment.seats[1]).expect("observer");
    restart_environment(environment).expect("restart roles");
    assert!(environment.seats[0].order_bookkeeping.is_candidate());
    assert!(environment.seats[1].order_bookkeeping.is_neural());
    assert!(!environment.seats[1].order_bookkeeping.is_candidate());
    for seat in &environment.seats {
        assert_eq!(
            seat.order_bookkeeping
                .effective(&seat.persistence)
                .last_sequence(),
            None
        );
        assert_eq!((seat.sequence, seat.pending_active), (0, None));
    }
}
