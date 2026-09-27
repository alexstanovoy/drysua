use crate::{IssuedOrder, OrderPersistence};
use bota_proto::{AbilitySlot, EntityId, ItemId, ItemSlot, Order, Target, Vec2};

#[test]
fn body_ledgers_suppress_exact_repeats_and_rollback_only_the_latest_matching_sequence() {
    let mut ledger = OrderPersistence::default();
    let courier = EntityId {
        idx: 2,
        generation: 1,
    };
    assert_eq!(ledger.should_send(None), None);
    for (index, unit) in [None, Some(courier)].into_iter().enumerate() {
        let sequence = index as u32 * 4 + 1;
        let first = IssuedOrder {
            unit,
            order: Order::Attack {
                target: Target::None,
            },
        };
        let next = IssuedOrder {
            unit,
            order: Order::Move {
                target: Target::Pos(Vec2::from_ints(10, 20)),
            },
        };
        ledger.record_sent(sequence, first).expect("first");
        ledger.record_sent(sequence + 1, next).expect("replacement");
        assert_eq!(ledger.should_send(Some(next)), None);
        assert!(!ledger.observe_rejection(sequence));
        assert!(ledger.observe_rejection(sequence + 1));
        assert_eq!(ledger.should_send(Some(first)), None);
        assert_eq!(ledger.should_send(Some(next)), Some(next));
        assert!(!ledger.observe_rejection(sequence));
        ledger
            .record_sent(
                sequence + 2,
                IssuedOrder {
                    unit,
                    order: Order::Buy { item: ItemId(3) },
                },
            )
            .expect("economy");
        assert_eq!(ledger.should_send(Some(first)), None);
    }
    ledger.clear_body_for(None);
    assert_eq!(ledger.active_body_order_for(None), None);
    assert!(ledger.active_body_order_for(Some(courier)).is_some());
    assert_eq!(ledger.last_sequence(), Some(7));
    let replacement = IssuedOrder {
        unit: Some(EntityId {
            generation: 2,
            ..courier
        }),
        order: Order::Cast {
            slot: AbilitySlot(0),
            target: Target::None,
        },
    };
    ledger.record_sent(8, replacement).expect("new generation");
    assert!(ledger.observe_rejection(8));
    assert_eq!(ledger.active_body_order_for(Some(courier)), None);
    assert_eq!(ledger.active_body_order_for(replacement.unit), None);
}

#[test]
fn one_shots_always_send_and_rejection_restores_the_interrupted_body() {
    for order in [
        Order::Cast {
            slot: AbilitySlot(0),
            target: Target::None,
        },
        Order::Use {
            slot: ItemSlot(0),
            target: Target::None,
        },
        Order::Put {
            slot: ItemSlot(0),
            target: Target::Pos(Vec2::from_ints(10, 20)),
        },
        Order::Take {
            target: Target::Unit(EntityId {
                idx: 30,
                generation: 2,
            }),
        },
    ] {
        let mut ledger = OrderPersistence::default();
        let body = IssuedOrder {
            unit: None,
            order: Order::Attack {
                target: Target::None,
            },
        };
        let one_shot = IssuedOrder { unit: None, order };
        ledger.record_sent(9, body).expect("body");
        ledger.record_sent(10, one_shot).expect("one shot");
        assert_eq!(ledger.should_send(Some(one_shot)), Some(one_shot));
        assert_eq!(ledger.active_body_sequence(), None);
        let before = ledger;
        for sequence in [0, 9, 10] {
            assert_eq!(
                ledger
                    .record_sent(sequence, body)
                    .expect_err("nonmonotonic sequence")
                    .to_string(),
                format!("order sequence {sequence} must be greater than last sent sequence 10")
            );
            assert_eq!(ledger, before);
        }
        assert!(ledger.observe_rejection(10));
        assert_eq!(ledger.should_send(Some(body)), None);
    }
}
