use super::*;
use crate::{ActivePolicyTarget, ControlledUnit, StructuredAction, global_feature};
use bota_proto::{AbilitySlot, Order, Target, Vec2, WorldView};

pub(crate) fn transcript(side: usize) -> Vec<ServerMsg> {
    assert!(side < 2);
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(0),
        seed: 10_093_000,
    })
    .expect("bounded transcript fixture");
    let projected = arena.configure_for_test(|world| {
        let heroes = [
            world.seats[0].unit.expect("Radiant"),
            world.seats[1].unit.expect("Dire"),
        ];
        let removed: Vec<_> = world
            .entities
            .iter()
            .filter(|entity| {
                !heroes.contains(entity)
                    && world.kind.get(*entity) != Some(&bota_proto::UnitKind::Ancient)
            })
            .collect();
        assert!(removed.len() < 64);
        for entity in removed {
            assert!(world.despawn(entity));
        }
        for index in 0..2 {
            let hero = world.seats[index].unit.expect("hero");
            world.transform.get_mut(hero).expect("position").pos =
                Vec2::from_ints(8600 + index as i32 * 200, 8900);
            world.statuses.remove(hero);
            world.abilities.get_mut(hero).expect("kit").slots[0].level = 1;
        }
    });
    let mut messages = vec![start.messages[side][0].clone()];
    messages.extend(projected.messages[side].clone());
    assert_eq!(messages.len(), 3);
    messages
}

pub(crate) fn snapshot(messages: &[ServerMsg]) -> &WorldView {
    assert!(messages.len() <= 3);
    messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .expect("projected snapshot")
}

pub(crate) fn packet(
    initial: &[ServerMsg],
    tick: u32,
    edit: impl FnOnce(&mut WorldView),
) -> Vec<ServerMsg> {
    assert!(tick > 1);
    assert!(tick <= 32);
    let mut view = snapshot(initial).clone();
    view.tick = tick;
    edit(&mut view);
    vec![
        ServerMsg::Snapshot { view },
        ServerMsg::Events {
            tick,
            events: Vec::new(),
        },
    ]
}

pub(crate) fn constant_model(kind: ActionKind) -> PolicyModel {
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
        .expect("finite constant policy");
    model
}

pub(crate) fn attack_action(tracker: &StateTracker, space: &ActionSpace) -> StructuredAction {
    let target = tracker.current().expect("snapshot").players[1 - usize::from(tracker.slot().0)]
        .unit
        .expect("enemy");
    StructuredAction::AttackUnit {
        unit: ControlledUnit::Hero,
        target: space.entity_index(target).expect("visible enemy"),
    }
}

pub(crate) fn cast_action() -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(0),
        target: crate::ActionTarget::None,
    }
}

fn send(seat: &mut ArenaSeatPolicy, cast: bool) -> Request {
    let space =
        ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness).expect("space");
    let action = if cast {
        cast_action()
    } else {
        attack_action(&seat.tracker, &space)
    };
    seat.local
        .note_decision(space.tick(), action.kind())
        .expect("decision");
    issue_request(
        seat,
        space.decode(action).expect("decode"),
        &space,
        action.kind(),
        true,
    )
    .expect("record actual request")
    .expect("request delivered")
}

#[test]
fn training_contract_current_policy_dispatch_preserves_own_cast_like_candidate() {
    for side in 0..2 {
        for shared in [false, true] {
            let model = constant_model(ActionKind::Continue);
            let policy = PolicySnapshot::capture(&model, 0).expect("current snapshot");
            let spec = if shared {
                OpponentSpec::SharedPolicy(Arc::new(policy.instantiate().expect("model")))
            } else {
                OpponentSpec::Policy(policy)
            };
            let mut environment =
                build_environment(10_093_000, 1, MapId(0), 1 - side, 0, spec).expect("environment");
            let initial = transcript(side);
            environment.seats[side] = setup_seat(side, &initial).expect("opponent projection");
            let mut candidate = setup_seat(side, &initial).expect("candidate projection");
            prepare_neural_seat_policy_sample(&mut candidate).expect("candidate opt-in");
            let mut sampling = PpoRng::new(1);
            paired_dispatch(
                &model,
                &mut sampling,
                &mut candidate,
                &mut environment.seats[side],
                &mut environment.opponent,
            );
            for cast in [false, true] {
                assert_eq!(
                    send(&mut candidate, cast),
                    send(&mut environment.seats[side], cast)
                );
            }
            let messages = packet(&initial, 2, |_| {});
            observe_messages(&mut candidate, &messages).expect("candidate observation");
            observe_messages(&mut environment.seats[side], &messages)
                .expect("opponent observation");
            paired_dispatch(
                &model,
                &mut sampling,
                &mut candidate,
                &mut environment.seats[side],
                &mut environment.opponent,
            );
            assert_eq!(
                environment.seats[side].local.active_order(),
                candidate.local.active_order()
            );
            assert_eq!(
                environment.seats[side]
                    .local
                    .active_order()
                    .expect("preserved attack")
                    .started_tick,
                1
            );
        }
    }
}

fn paired_dispatch(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    candidate: &mut ArenaSeatPolicy,
    frozen: &mut ArenaSeatPolicy,
    opponent: &mut OpponentRuntime,
) {
    let (frame, space) = prepare_neural_seat_policy_sample(candidate).expect("candidate input");
    let (opponent_frame, _) = prepare_seat_policy_sample(frozen).expect("opponent input");
    assert_eq!(frame, opponent_frame, "all tensor features before dispatch");
    let choice = model
        .sample(&frame, &space, sampling)
        .expect("candidate sampling");
    let request = policy_request_in_space(candidate, &choice, &space).expect("candidate dispatch");
    assert_eq!(
        opponent_request(frozen, opponent).expect("frozen dispatch"),
        request
    );
    let OpponentRuntime::Policy { rng, .. } = opponent else {
        panic!("neural runtime")
    };
    assert_eq!(rng, sampling);
    assert_eq!(candidate.persistence, frozen.persistence);
    assert_eq!(candidate.order_bookkeeping, frozen.order_bookkeeping);
}

#[test]
fn training_contract_expert_sample_preserves_sent_cast_attack_without_changing_teacher() {
    for side in 0..2 {
        let initial = transcript(side);
        let mut observer = setup_seat(side, &initial).expect("observer");
        let mut reference = setup_seat(side, &initial).expect("legacy reference");
        expert_sample_and_reference(&mut observer, &mut reference);
        for cast in [false, true] {
            assert_eq!(send(&mut observer, cast), send(&mut reference, cast));
        }
        let messages = packet(&initial, 2, |_| {});
        observe_messages(&mut observer, &messages).expect("observer tick");
        observe_messages(&mut reference, &messages).expect("reference tick");
        let sample = expert_sample_and_reference(&mut observer, &mut reference);
        assert_eq!(
            sample.frame().global()[global_feature::ACTIVE_ORDER_PRESENT],
            1.0
        );
        assert_eq!(
            observer
                .local
                .active_order()
                .expect("sent directive")
                .target,
            ActivePolicyTarget::Unit(snapshot(&initial).players[1 - side].unit.expect("enemy"))
        );
        assert_eq!(observer.persistence.active_body_order_for(None), None);
        assert_eq!(send(&mut observer, false), send(&mut reference, false));
        assert_eq!(observer.teacher, reference.teacher);
    }
}

fn expert_sample_and_reference(
    observer: &mut ArenaSeatPolicy,
    reference: &mut ArenaSeatPolicy,
) -> crate::ImitationSample {
    let (action, _, sample) = pretraining_teacher_action(
        observer,
        SeedNamespace::Training,
        10_093_000,
        0,
        0,
        &[[[0; ActionKind::COUNT]; PRETRAINING_WINDOWS]; 2],
        &mut TeacherCoverage::new(),
    )
    .expect("production expert sample");
    let expected = reference
        .teacher
        .decide(
            &reference.tracker,
            &reference.persistence,
            &reference.readiness,
        )
        .expect("unchanged Teacher")
        .0;
    assert_eq!(action, expected);
    assert_eq!(observer.teacher, reference.teacher);
    assert_eq!(observer.persistence, reference.persistence);
    sample.expect("retained row")
}

#[test]
fn training_contract_current_policy_resends_after_fog_on_both_sides() {
    for side in 0..2 {
        let model = constant_model(ActionKind::AttackUnit);
        let spec = OpponentSpec::Policy(PolicySnapshot::capture(&model, 0).expect("snapshot"));
        let mut opponent = build_opponent(&spec, 42).expect("runtime");
        let OpponentRuntime::Policy { rng, .. } = &opponent else {
            panic!("policy runtime")
        };
        let sampling = rng.clone();
        let initial = transcript(side);
        let mut seat = setup_seat(side, &initial).expect("seat");
        let first = opponent_request(&mut seat, &mut opponent)
            .expect("dispatch")
            .expect("attack");
        let Order::Attack {
            target: Target::Unit(target),
        } = first.order
        else {
            panic!("model-selected attack")
        };
        observe_messages(
            &mut seat,
            &packet(&initial, 2, |view| {
                view.units.retain(|unit| unit.id != target)
            }),
        )
        .expect("fog");
        let fallback = seat.local.active_order();
        assert_eq!(
            fallback.expect("known fallback").kind,
            ActionKind::AttackMovePoint
        );
        observe_messages(&mut seat, &packet(&initial, 3, |_| {})).expect("return");
        assert_eq!(seat.local.active_order(), fallback);
        let OpponentRuntime::Policy { rng, .. } = &mut opponent else {
            panic!("policy runtime")
        };
        *rng = sampling;
        let repeated = opponent_request(&mut seat, &mut opponent)
            .expect("dispatch")
            .expect("real reattack");
        assert_eq!(repeated.order, first.order);
        assert_eq!(repeated.seq, first.seq + 1);
    }
}

#[test]
fn training_contract_league_current_accepted_history_and_restart_keep_candidate_role() {
    let initial =
        PolicySnapshot::capture(&constant_model(ActionKind::Continue), 0).expect("initial");
    let current = PolicySnapshot::capture(&constant_model(ActionKind::Hold), 2).expect("current");
    let historical =
        PolicySnapshot::capture(&constant_model(ActionKind::Stop), 1).expect("history");
    let mut league = League::new(32, initial).expect("league");
    league
        .insert_historical(historical, 0.0, CrossPlayProfile::default())
        .expect("history");
    for (bucket, generation) in [(0, 2), (30, 0), (55, 1)] {
        let opponent = league.select_bucket(bucket, &current).expect("selection");
        assert_eq!(
            opponent
                .snapshot()
                .expect("current-contract weights")
                .generation(),
            generation
        );
        let mut environment = build_environment(
            10_093_000,
            1,
            MapId(0),
            0,
            0,
            opponent_spec(&opponent).expect("spec"),
        )
        .expect("environment");
        assert!(environment.seats[1].order_bookkeeping.is_candidate());
        assert!(!environment.seats[0].order_bookkeeping.is_neural());
        restart_environment(&mut environment).expect("restart");
        assert!(environment.seats[1].order_bookkeeping.is_candidate());
        assert_eq!(environment.seats[1].sequence, 0);
    }
}

#[test]
fn training_contract_observer_fog_and_rejection_keep_teacher_transport_legacy() {
    for side in 0..2 {
        let initial = transcript(side);
        let target = snapshot(&initial).players[1 - side].unit.expect("target");
        let mut observer = setup_seat(side, &initial).expect("observer");
        let mut reference = setup_seat(side, &initial).expect("legacy");
        prepare_neural_observer_sample(&mut observer).expect("observer from start");
        assert_eq!(send(&mut observer, false), send(&mut reference, false));
        for seat in [&mut observer, &mut reference] {
            send_selected(
                seat,
                StructuredAction::Stop {
                    unit: ControlledUnit::Hero,
                },
            )
            .expect("sent Stop");
        }
        let mut messages = packet(&initial, 2, |view| {
            view.units.retain(|unit| unit.id != target)
        });
        for seat in [&mut observer, &mut reference] {
            observe_messages(seat, &messages).expect("fog");
        }
        messages = packet(&initial, 3, |view| {
            view.units.retain(|unit| unit.id != target)
        });
        messages.insert(
            0,
            ServerMsg::OrderRejected {
                seq: 2,
                reason: RejectReason::UnknownTarget,
            },
        );
        for seat in [&mut observer, &mut reference] {
            observe_messages(seat, &messages).expect("rejection");
        }
        let sample = expert_sample_and_reference(&mut observer, &mut reference);
        assert_eq!(
            sample.frame().global()[global_feature::ACTIVE_TARGET_POINT],
            1.0
        );
        assert_eq!(
            observer
                .local
                .active_order()
                .expect("rollback fallback")
                .started_tick,
            2
        );
        for seat in [&mut observer, &mut reference] {
            observe_messages(seat, &packet(&initial, 4, |_| {})).expect("return");
        }
        let (frame, space) = prepare_neural_observer_sample(&mut observer).expect("observer frame");
        assert_eq!(frame.global()[global_feature::ACTIVE_TARGET_UNIT], 0.0);
        let action = attack_action(&observer.tracker, &space);
        let issued = space.decode(action).expect("attack");
        assert_eq!(
            issue_request(&mut observer, issued, &space, action.kind(), true)
                .expect("legacy suppression"),
            None
        );
        assert_eq!(observer.persistence, reference.persistence);
        assert_eq!(observer.teacher, reference.teacher);
        assert_eq!(observer.sequence, reference.sequence);
        assert_eq!(observer.readiness, reference.readiness);
    }
}

#[test]
fn training_contract_observer_hero_death_and_respawn_cannot_restore_old_body_orders() {
    for side in 0..2 {
        let initial = transcript(side);
        let hero = snapshot(&initial).players[side].unit.expect("hero");
        let mut observer = setup_seat(side, &initial).expect("observer");
        let mut reference = setup_seat(side, &initial).expect("legacy");
        prepare_neural_observer_sample(&mut observer).expect("from start");
        for cast in [false, true] {
            assert_eq!(send(&mut observer, cast), send(&mut reference, cast));
        }
        let messages = packet(&initial, 2, |view| {
            view.players[side].unit = None;
            view.units.retain(|unit| unit.id != hero);
        });
        for seat in [&mut observer, &mut reference] {
            observe_messages(seat, &messages).expect("hero death");
        }
        assert_eq!(observer.local.active_order(), None);
        assert_eq!(observer.pending_active, None);
        let mut messages = packet(&initial, 3, |view| {
            let mut replacement = hero;
            replacement.generation += 1;
            view.players[side].unit = Some(replacement);
            view.units
                .iter_mut()
                .find(|unit| unit.id == hero)
                .expect("hero")
                .id = replacement;
        });
        messages.insert(
            0,
            ServerMsg::OrderRejected {
                seq: 1,
                reason: RejectReason::UnknownTarget,
            },
        );
        for seat in [&mut observer, &mut reference] {
            observe_messages(seat, &messages).expect("respawn");
        }
        let sample = expert_sample_and_reference(&mut observer, &mut reference);
        assert_eq!(
            sample.frame().global()[global_feature::ACTIVE_ORDER_PRESENT],
            0.0
        );
        assert_eq!(observer.local.active_order(), None);
        assert_eq!(observer.persistence, reference.persistence);
        assert_eq!(observer.readiness, reference.readiness);
        assert_eq!(observer.sequence, reference.sequence);
    }
}

#[test]
fn training_contract_late_observer_enable_rejects_without_inventing_prefix_state() {
    let initial = transcript(0);
    let mut seat = setup_seat(0, &initial).expect("seat");
    send(&mut seat, false);
    let before = (seat.persistence, seat.local, seat.pending_active);
    let error = prepare_neural_observer_sample(&mut seat)
        .err()
        .expect("lost legacy prefix");
    assert_eq!(
        error.to_string(),
        "invalid PPO transition: neural bookkeeping must start before the first actual request"
    );
    assert_eq!((seat.persistence, seat.local, seat.pending_active), before);
}

#[test]
fn training_contract_zero_warmup_initializes_candidate_before_cleanup_requests() {
    let mut environments = vec![
        build_environment(10_093_000, 1, MapId(0), 0, 0, OpponentSpec::Weak).expect("environment"),
    ];
    let model = constant_model(ActionKind::Continue);
    warmup_training_environments(&mut environments, &[0], &model, 3).expect("zero warmup");
    assert!(
        environments[0].seats[0].sequence > 0,
        "cleanup still sends its original requests"
    );
    prepare_policy_sample(&mut environments[0])
        .expect("candidate must not inherit a legacy-only prefix");
    assert!(environments[0].seats[0].order_bookkeeping.is_candidate());
    assert_eq!(environments[0].seats[0].pending_active, None);
}

#[test]
fn training_contract_restart_preserves_observer_and_candidate_roles_with_empty_ledgers() {
    let mut environment = build_environment(10_093_000, 1, MapId(0), 0, 0, OpponentSpec::Teacher)
        .expect("environment");
    prepare_policy_sample(&mut environment).expect("candidate");
    observe_neural_seat_orders(&mut environment.seats[1]).expect("observer");
    teacher_request(&mut environment.seats[1]).expect("actual Teacher request");
    assert!(environment.seats[1].sequence > 0);
    restart_environment(&mut environment).expect("restart");
    assert!(environment.seats[0].order_bookkeeping.is_candidate());
    assert!(environment.seats[1].order_bookkeeping.is_neural());
    assert!(!environment.seats[1].order_bookkeeping.is_candidate());
    for seat in &environment.seats {
        assert_eq!(seat.sequence, 0);
        assert_eq!(
            seat.order_bookkeeping
                .effective(&seat.persistence)
                .last_sequence(),
            None
        );
        assert_eq!(seat.pending_active, None);
    }
}

#[test]
fn training_contract_observer_uses_same_tick_death_evidence_not_an_invented_point() {
    for side in 0..2 {
        let initial = transcript(side);
        let target = snapshot(&initial).players[1 - side].unit.expect("target");
        let mut seat = setup_seat(side, &initial).expect("observer");
        prepare_neural_observer_sample(&mut seat).expect("from start");
        send(&mut seat, false);
        let mut messages = packet(&initial, 2, |view| {
            view.units.retain(|unit| unit.id != target)
        });
        let ServerMsg::Events { events, .. } = &mut messages[1] else {
            panic!("events")
        };
        events.push(EventKind::Died {
            unit: target,
            killer: None,
            denied: false,
            gold: 0,
        });
        observe_messages(&mut seat, &messages).expect("complete tick");
        let (frame, _) = prepare_neural_observer_sample(&mut seat).expect("F12 frame");
        assert_eq!(frame.global()[global_feature::ACTIVE_ORDER_PRESENT], 0.0);
        assert_eq!(frame.global()[global_feature::ACTIVE_TARGET_POINT], 0.0);
        assert_eq!(seat.local.active_order(), None);
        assert_eq!(
            seat.persistence.active_body_order_for(None),
            Some(crate::IssuedOrder {
                unit: None,
                order: Order::Attack {
                    target: Target::Unit(target)
                }
            }),
            "legacy Teacher ledger is not reconciled"
        );
        assert_eq!(seat.sequence, 1);
    }
}

#[test]
fn training_contract_rejected_own_cast_keeps_observer_attack_without_claiming_cast_success() {
    let initial = transcript(0);
    let mut observer = setup_seat(0, &initial).expect("observer");
    let mut reference = setup_seat(0, &initial).expect("legacy");
    prepare_neural_observer_sample(&mut observer).expect("from start");
    for cast in [false, true] {
        assert_eq!(send(&mut observer, cast), send(&mut reference, cast));
    }
    assert_eq!(
        observer
            .local
            .active_order()
            .expect("sent attack")
            .started_tick,
        1
    );
    let mut messages = packet(&initial, 2, |_| {});
    messages.insert(
        0,
        ServerMsg::OrderRejected {
            seq: 2,
            reason: RejectReason::UnknownTarget,
        },
    );
    for seat in [&mut observer, &mut reference] {
        observe_messages(seat, &messages).expect("cast rejection");
    }
    let sample = expert_sample_and_reference(&mut observer, &mut reference);
    assert_eq!(
        sample.frame().global()[global_feature::ACTIVE_ORDER_PRESENT],
        1.0
    );
    assert_eq!(
        sample.frame().abilities()[0][crate::ability_feature::LAST_CAST_PRESENT],
        0.0
    );
    assert_eq!(observer.rejections, 1);
    assert_eq!(observer.sequence, 2);
    assert_eq!(observer.readiness, reference.readiness);
}

#[test]
fn training_contract_observer_collection_preserves_real_teacher_requests_and_complete_ticks() {
    let mut observed =
        build_environment(10_093_004, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("observed");
    let mut reference =
        build_environment(10_093_004, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("reference");
    for _ in 0..16 {
        let mut requests = Vec::with_capacity(2);
        let mut expected = Vec::with_capacity(2);
        for side in 0..2 {
            let source = &mut observed.seats[side];
            let target = &mut reference.seats[side];
            let (request, reference_request) = labeled_teacher_requests(source, target);
            requests.push(request);
            expected.push(reference_request);
        }
        assert_eq!(requests, expected);
        assert!(
            advance_interval(&mut observed, requests, 3)
                .expect("observed tick")
                .winner
                .is_none()
        );
        assert!(
            advance_interval(&mut reference, expected, 3)
                .expect("reference tick")
                .winner
                .is_none()
        );
        for side in 0..2 {
            assert_eq!(
                observed.seats[side].tracker.current(),
                reference.seats[side].tracker.current()
            );
            assert_eq!(
                observed.seats[side].persistence,
                reference.seats[side].persistence
            );
            assert_eq!(observed.seats[side].teacher, reference.seats[side].teacher);
            assert_eq!(
                observed.seats[side].readiness,
                reference.seats[side].readiness
            );
            assert_eq!(
                observed.seats[side].sequence,
                reference.seats[side].sequence
            );
        }
    }
}

fn labeled_teacher_requests(
    source: &mut ArenaSeatPolicy,
    target: &mut ArenaSeatPolicy,
) -> (Option<Request>, Option<Request>) {
    let sample = expert_sample_and_reference(source, target);
    let action = sample.teacher_action();
    let request = send_selected(source, action);
    let reference = send_selected(target, action);
    assert_eq!(source.teacher, target.teacher);
    assert_eq!(source.persistence, target.persistence);
    (request, reference)
}

fn send_selected(seat: &mut ArenaSeatPolicy, action: StructuredAction) -> Option<Request> {
    let space =
        ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness).expect("space");
    assert!(space.allows(action));
    seat.local
        .note_decision(space.tick(), action.kind())
        .expect("decision");
    issue_request(
        seat,
        space.decode(action).expect("decode"),
        &space,
        action.kind(),
        true,
    )
    .expect("actual dispatch")
}

#[test]
fn training_contract_observer_and_candidate_frames_match_for_the_same_actual_prefix() {
    for side in 0..2 {
        let initial = transcript(side);
        let target = snapshot(&initial).players[1 - side].unit.expect("target");
        let mut observer = setup_seat(side, &initial).expect("observer");
        let mut candidate = setup_seat(side, &initial).expect("candidate");
        prepare_neural_observer_sample(&mut observer).expect("observer from start");
        prepare_neural_seat_policy_sample(&mut candidate).expect("candidate from start");
        for cast in [false, true] {
            assert_eq!(send(&mut observer, cast), send(&mut candidate, cast));
        }
        for tick in 2..=4 {
            let mut messages = packet(&initial, tick, |view| {
                if tick == 2 {
                    view.units.retain(|unit| unit.id != target);
                }
            });
            if tick == 4 {
                messages.insert(
                    0,
                    ServerMsg::OrderRejected {
                        seq: 2,
                        reason: RejectReason::UnknownTarget,
                    },
                );
            }
            for seat in [&mut observer, &mut candidate] {
                observe_messages(seat, &messages).expect("complete tick");
            }
            let (observed, _) =
                prepare_neural_observer_sample(&mut observer).expect("observer input");
            let (expected, _) =
                prepare_neural_seat_policy_sample(&mut candidate).expect("candidate input");
            assert_eq!(
                observed, expected,
                "all F12 tensor fields; no execution-role feature"
            );
            assert_eq!(
                observer.local.active_order(),
                candidate.local.active_order()
            );
            assert_eq!(observer.sequence, 2, "reconciliation never invents a send");
        }
        assert!(!observer.order_bookkeeping.is_candidate());
        let before = observer.order_bookkeeping;
        let error = prepare_neural_seat_policy_sample(&mut observer)
            .err()
            .expect("no controller hot switch");
        assert_eq!(
            error.to_string(),
            "invalid PPO transition: neural observer cannot become a candidate"
        );
        assert_eq!(observer.order_bookkeeping, before);
    }
}
