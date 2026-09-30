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
                build_environment(10_093_000, 1, MapId(0), 1 - side, spec, Vec::new())
                    .expect("environment");
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
