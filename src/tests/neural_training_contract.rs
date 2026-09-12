use super::*;
use crate::ppo_arena::training_order_contract::{
    attack_action, cast_action, constant_model, packet, snapshot,
};
use crate::{ActivePolicyTarget, global_feature};

fn transcript(side: usize) -> Vec<ServerMsg> {
    assert!(side < 2);
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 10_093_000,
    })
    .expect("Map2 transcript fixture");
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
        for (index, hero) in heroes.into_iter().enumerate() {
            world.transform.get_mut(hero).expect("position").pos =
                bota_proto::Vec2::from_ints(8600 + index as i32 * 200, 8900);
            world.statuses.remove(hero);
            world.abilities.get_mut(hero).expect("kit").slots[0].level = 1;
        }
    });
    let mut messages = vec![start.messages[side][0].clone()];
    messages.extend(projected.messages[side].clone());
    assert_eq!(messages.len(), 3);
    messages
}

#[test]
fn training_contract_prediction_does_not_change_teacher_repeat_delivery() {
    for side in 0..2 {
        let initial = transcript(side);
        let mut predicted = NeuralSeat::new(side as u8, &initial).expect("predicted seat");
        let mut reference = NeuralSeat::new(side as u8, &initial).expect("legacy seat");
        let mut teacher = Teacher::new();
        let mut baseline = Teacher::new();
        let model = constant_model(ActionKind::Continue);
        predicted
            .neural_choice(&model)
            .expect("prediction before any sends");
        for cast in [false, true, false] {
            let source = send(&mut predicted, cast);
            let target = send(&mut reference, cast);
            assert_eq!(
                source, target,
                "asking for a prediction must not change actual delivery"
            );
            if let Some(issued) = source {
                teacher.note_sent(predicted.sequence, issued, 1);
            }
            if let Some(issued) = target {
                baseline.note_sent(reference.sequence, issued, 1);
            }
            assert_eq!(teacher, baseline);
        }
        let messages = packet(&initial, 2, |_| {});
        predicted.observe(&messages).expect("tick");
        reference.observe(&messages).expect("tick");
        assert_eq!(predicted.sequence, 3);
        assert_eq!(predicted.persistence, reference.persistence);
        assert_eq!(
            teacher
                .decide(
                    &predicted.tracker,
                    &predicted.persistence,
                    &predicted.readiness
                )
                .expect("label")
                .0,
            baseline
                .decide(
                    &reference.tracker,
                    &reference.persistence,
                    &reference.readiness
                )
                .expect("baseline label")
                .0
        );
        assert_eq!(teacher, baseline);
        assert_eq!(
            predicted.orders_hash.finish(),
            reference.orders_hash.finish()
        );
    }
}

fn send(seat: &mut NeuralSeat, cast: bool) -> Option<crate::IssuedOrder> {
    let space =
        ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness).expect("space");
    let action = if cast {
        cast_action()
    } else {
        attack_action(&seat.tracker, &space)
    };
    seat.issue(action, &space).expect("actual send")
}

#[test]
fn training_contract_neuralseat_bc_frames_follow_actual_sends_fog_and_rejection() {
    let initial = [transcript(0), transcript(1)];
    let mut seats =
        std::array::from_fn(|side| NeuralSeat::new(side as u8, &initial[side]).expect("seat"));
    prepare_game_seats(&mut seats, None, true).expect("production BC roles from trajectory start");
    for side in 0..2 {
        let seat = &mut seats[side];
        let mut reference = NeuralSeat::new(side as u8, &initial[side]).expect("legacy reference");
        let target = snapshot(&initial[side]).players[1 - side]
            .unit
            .expect("target");
        for cast in [false, true] {
            assert_eq!(send(seat, cast), send(&mut reference, cast));
        }
        let messages = packet(&initial[side], 2, |_| {});
        seat.observe(&messages).expect("observed tick");
        reference.observe(&messages).expect("legacy tick");
        let sample = retained_teacher_sample(seat);
        assert_eq!(sample.identity().map(), MapId(2));
        assert_eq!(
            sample.frame().global()[global_feature::ACTIVE_ORDER_PRESENT],
            1.0
        );
        assert_eq!(seat.local.active_order().expect("attack").started_tick, 1);
        assert_eq!(seat.sequence, 2, "collecting a label is not sending it");
        assert_eq!(
            send(seat, false),
            send(&mut reference, false),
            "actual Teacher repeat is NOT deduplicated by observer"
        );
        let messages = packet(&initial[side], 3, |view| {
            view.units.retain(|unit| unit.id != target)
        });
        seat.observe(&messages).expect("fog");
        reference.observe(&messages).expect("legacy fog");
        assert!(matches!(
            seat.local.active_order().expect("fallback").target,
            ActivePolicyTarget::Point(_)
        ));
        let mut messages = packet(&initial[side], 4, |_| {});
        messages.insert(
            0,
            ServerMsg::OrderRejected {
                seq: 3,
                reason: bota_proto::RejectReason::UnknownTarget,
            },
        );
        seat.observe(&messages).expect("rejected repeat");
        reference
            .observe(&messages)
            .expect("legacy rejected repeat");
        assert!(matches!(
            seat.local
                .active_order()
                .expect("reconciled rollback")
                .target,
            ActivePolicyTarget::Point(_)
        ));
        assert_eq!(seat.persistence, reference.persistence);
        assert_eq!(seat.readiness, reference.readiness);
        assert_eq!(seat.sequence, reference.sequence);
    }
}

fn retained_teacher_sample(seat: &mut NeuralSeat) -> ImitationSample {
    let mut reservoir = Reservoir::new(64, 10_093_003);
    let mut labeler = Teacher::new();
    let (action, space) = labeler
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .expect("label");
    reservoir
        .consider(
            seat,
            &space,
            action,
            10_093_000,
            SeedNamespace::Training,
            None,
        )
        .expect("BC sample");
    assert_eq!(reservoir.samples.len(), 1);
    reservoir.samples.pop().expect("retained row")
}

#[test]
fn training_contract_dagger_roles_keep_candidate_and_labeler_ledgers_separate() {
    let initial = [transcript(0), transcript(1)];
    for side in 0..2 {
        let mut seats = std::array::from_fn(|index| {
            NeuralSeat::new(index as u8, &initial[index]).expect("seat")
        });
        prepare_game_seats(&mut seats, Some(side), true).expect("DAgger roles");
        assert!(seats[side].order_bookkeeping.is_candidate());
        assert!(!seats[1 - side].order_bookkeeping.is_neural());
        for cast in [false, true] {
            assert!(send(&mut seats[side], cast).is_some());
        }
        assert!(
            seats[side]
                .persistence
                .active_body_order_for(None)
                .is_none()
        );
        assert_eq!(
            seats[side]
                .local
                .active_order()
                .expect("preserved attack")
                .started_tick,
            1
        );
        let mut collection = Collection {
            reservoir: &mut Reservoir::new(64, 1),
            namespace: SeedNamespace::Training,
            labeler: Some(Teacher::new()),
        };
        let space =
            ActionSpace::from_tracker_with_readiness(&seats[side].tracker, &seats[side].readiness)
                .expect("space");
        collect_decision(
            &mut collection,
            &mut seats[side],
            &space,
            StructuredAction::Continue,
            10_093_000,
            true,
        )
        .expect("label");
        assert_eq!(
            collection.reservoir.samples[0].source(),
            crate::ImitationSource::Dagger
        );
        assert_eq!(collection.reservoir.samples[0].identity().map(), MapId(2));
        assert_eq!(
            collection.reservoir.samples[0].frame().global()[global_feature::ACTIVE_ORDER_PRESENT],
            1.0
        );
        assert_eq!(seats[side].sequence, 2);
    }
}
