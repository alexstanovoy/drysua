use bota_proto::{EntityId, ItemId, Order, Target, Vec2};

use crate::{IssuedOrder, OrderPersistence};

#[test]
fn persistence_sends_none_for_continue_and_suppresses_animation_safe_exact_repeats() {
    let body = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(10, 20)),
    });
    let mut persistence = OrderPersistence::default();

    assert_eq!(persistence.should_send(None), None);
    assert_eq!(persistence.should_send(Some(body)), Some(body));
    persistence.record_sent(1, body).expect("first body");

    assert_eq!(persistence.should_send(Some(body)), None);
    assert_eq!(persistence.active_body_sequence(), Some(1));
}

#[test]
fn persistence_emits_changed_body_and_non_body_does_not_replace_active_body() {
    let first = issued(Order::Attack {
        target: Target::None,
    });
    let changed = issued(Order::Attack {
        target: Target::Unit(EntityId {
            idx: 20,
            generation: 1,
        }),
    });
    let buy = issued(Order::Buy { item: ItemId(3) });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(4, first).expect("first");

    assert_eq!(persistence.should_send(Some(changed)), Some(changed));
    persistence.record_sent(5, changed).expect("changed");
    persistence.record_sent(6, buy).expect("buy");

    assert_eq!(persistence.active_body_sequence(), Some(5));
    assert_eq!(persistence.active_body_order_for(None), Some(changed));
    assert_eq!(persistence.should_send(Some(changed)), None);
    assert_eq!(persistence.last_sequence(), Some(6));
}

#[test]
fn persistence_suppresses_hero_and_courier_orders_independently() {
    let courier = EntityId {
        idx: 2,
        generation: 1,
    };
    let hero_order = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(10, 20)),
    });
    let courier_order = issued_for(
        Some(courier),
        Order::Move {
            target: Target::Pos(Vec2::from_ints(30, 40)),
        },
    );
    let mut persistence = OrderPersistence::default();

    persistence.record_sent(1, hero_order).expect("hero order");
    persistence
        .record_sent(2, courier_order)
        .expect("courier order");

    assert_eq!(persistence.should_send(Some(hero_order)), None);
    assert_eq!(persistence.should_send(Some(courier_order)), None);
    assert_eq!(persistence.active_body_sequence(), Some(2));
    assert_eq!(persistence.active_body_sequence_for(None), Some(1));
    assert_eq!(persistence.active_body_sequence_for(Some(courier)), Some(2));
    assert_eq!(persistence.active_body_order_for(None), Some(hero_order));
    assert_eq!(
        persistence.active_body_order_for(Some(courier)),
        Some(courier_order)
    );
}

#[test]
fn persistence_rejection_restores_previous_order_for_same_body() {
    let first = issued(Order::Attack {
        target: Target::None,
    });
    let replacement = issued(Order::Attack {
        target: Target::Unit(EntityId {
            idx: 20,
            generation: 1,
        }),
    });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(1, first).expect("first order");
    persistence
        .record_sent(2, replacement)
        .expect("replacement order");

    assert!(persistence.observe_rejection(2));

    assert_eq!(persistence.should_send(Some(first)), None);
    assert_eq!(
        persistence.should_send(Some(replacement)),
        Some(replacement)
    );
    assert_eq!(persistence.active_body_sequence(), Some(1));
    assert!(!persistence.observe_rejection(1));
    assert_eq!(persistence.active_body_sequence(), Some(1));
}

#[test]
fn persistence_keeps_only_immediately_previous_body_order() {
    let first = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(10, 10)),
    });
    let second = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(20, 20)),
    });
    let third = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(30, 30)),
    });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(1, first).expect("first order");
    persistence.record_sent(2, second).expect("second order");
    persistence.record_sent(3, third).expect("third order");

    assert!(persistence.observe_rejection(3));

    assert_eq!(persistence.should_send(Some(second)), None);
    assert_eq!(persistence.should_send(Some(first)), Some(first));
    assert!(!persistence.observe_rejection(2));
    assert_eq!(persistence.active_body_sequence(), Some(2));
}

#[test]
fn persistence_rejection_clears_only_the_matching_active_body_sequence() {
    let body = issued(Order::Move {
        target: Target::Pos(Vec2::from_ints(10, 20)),
    });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(9, body).expect("body");

    assert!(!persistence.observe_rejection(8));
    assert_eq!(persistence.should_send(Some(body)), None);
    assert!(persistence.observe_rejection(9));
    assert_eq!(persistence.should_send(Some(body)), Some(body));
}

#[test]
fn persistence_never_suppresses_repeated_one_shot_orders() {
    let target = EntityId {
        idx: 30,
        generation: 2,
    };
    let orders = [
        Order::Cast {
            slot: bota_proto::AbilitySlot(0),
            target: Target::None,
        },
        Order::Use {
            slot: bota_proto::ItemSlot(0),
            target: Target::None,
        },
        Order::Put {
            slot: bota_proto::ItemSlot(0),
            target: Target::Pos(Vec2::from_ints(10, 20)),
        },
        Order::Take {
            target: Target::Unit(target),
        },
    ];
    let mut persistence = OrderPersistence::default();

    for (index, order) in orders.into_iter().enumerate() {
        let issued = issued(order);
        persistence
            .record_sent(u32::try_from(index + 1).expect("sequence fits"), issued)
            .expect("one-shot order");
        assert_eq!(persistence.should_send(Some(issued)), Some(issued));
    }

    assert_eq!(persistence.active_body_sequence(), None);
}

#[test]
fn persistence_one_shot_order_clears_active_body_and_rejection_restores_it() {
    let body = issued(Order::Attack {
        target: Target::None,
    });
    let one_shot = issued(Order::Use {
        slot: bota_proto::ItemSlot(0),
        target: Target::None,
    });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(9, body).expect("body");
    persistence.record_sent(10, one_shot).expect("one shot");

    assert_eq!(persistence.should_send(Some(body)), Some(body));
    assert_eq!(persistence.should_send(Some(one_shot)), Some(one_shot));
    assert_eq!(persistence.active_body_sequence(), None);

    assert!(persistence.observe_rejection(10));

    assert_eq!(persistence.should_send(Some(body)), None);
    assert_eq!(persistence.active_body_sequence(), Some(9));
}

#[test]
fn persistence_rejection_never_restores_a_previous_courier_generation() {
    let old_courier = EntityId {
        idx: 2,
        generation: 1,
    };
    let new_courier = EntityId {
        idx: 2,
        generation: 2,
    };
    let old_move = issued_for(
        Some(old_courier),
        Order::Move {
            target: Target::Pos(Vec2::from_ints(10, 20)),
        },
    );
    let new_cast = issued_for(
        Some(new_courier),
        Order::Cast {
            slot: bota_proto::AbilitySlot(0),
            target: Target::None,
        },
    );
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(1, old_move).expect("old courier");
    persistence.record_sent(2, new_cast).expect("new courier");

    assert!(persistence.observe_rejection(2));

    assert_eq!(persistence.active_body_order_for(Some(old_courier)), None);
    assert_eq!(persistence.active_body_order_for(Some(new_courier)), None);
}

#[test]
fn persistence_rejects_non_monotonic_sequence_with_exact_message() {
    let body = issued(Order::Take {
        target: Target::Unit(EntityId {
            idx: 30,
            generation: 2,
        }),
    });
    let mut persistence = OrderPersistence::default();
    persistence.record_sent(12, body).expect("first");

    let error = persistence
        .record_sent(12, body)
        .expect_err("repeat sequence");

    assert_eq!(
        error.to_string(),
        "order sequence 12 must be greater than last sent sequence 12"
    );
}

fn issued(order: Order) -> IssuedOrder {
    issued_for(None, order)
}

fn issued_for(unit: Option<EntityId>, order: Order) -> IssuedOrder {
    IssuedOrder { unit, order }
}
