use crate::{
    ActionKind, ActionSpace, ActiveOrderUpdate, ActivePolicyOrder, ActivePolicyTarget, Arena,
    ArenaConfig, IssuedOrder, ItemReadiness, LocalPolicyState, OrderPersistence, StateTracker,
    Teacher, active_order_update_for_sent,
};
use bota_proto::{
    AbilityId, AbilitySlot, Aim, EntityId, ItemSlot, MapId, Order, ServerMsg, SlotId, Target,
    UnitKind, Vec2,
};

type PendingActive = Option<(u32, Option<ActivePolicyOrder>)>;

struct Fixture {
    tracker: StateTracker,
    orders: OrderPersistence,
    local: LocalPolicyState,
    pending: PendingActive,
    target: EntityId,
    courier: EntityId,
}

impl Fixture {
    fn new() -> Self {
        let (mut arena, start) = Arena::new(ArenaConfig {
            seats: 2,
            map: MapId(0),
            seed: 10_092_100,
        })
        .expect("arena");
        let configured = arena.configure_for_test(|world| {
            for side in 0..2 {
                let hero = world.seats[side].unit.expect("hero");
                world.transform.get_mut(hero).expect("position").pos =
                    Vec2::from_ints(8600 + side as i32 * 200, 8900);
                world.statuses.remove(hero);
            }
            let hero = world.seats[0].unit.expect("caster");
            world.abilities.get_mut(hero).expect("kit").slots[0].level = 1;
        });
        let ServerMsg::MatchStart { info } = &start.messages[0][0] else {
            panic!("metadata")
        };
        let ServerMsg::Snapshot { view } = &configured.messages[0][0] else {
            panic!("snapshot")
        };
        let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
        tracker.observe_snapshot(view).expect("fixture observation");
        let target = tracker
            .own_player()
            .and_then(|_| view.players[1].unit)
            .expect("enemy handle");
        assert!(
            tracker
                .current()
                .expect("view")
                .units
                .iter()
                .any(|unit| unit.id == target)
        );
        let courier = tracker.own_courier().expect("courier").id;
        Self {
            tracker,
            orders: OrderPersistence::default(),
            local: LocalPolicyState::new(0),
            pending: None,
            target,
            courier,
        }
    }

    fn send(&mut self, sequence: u32, issued: IssuedOrder, kind: ActionKind) {
        let tick = self.tracker.current().expect("snapshot").tick;
        let previous = self.local.active_order();
        self.local.note_decision(tick, kind).expect("decision");
        let preserves = self
            .orders
            .record_neural_sent(sequence, issued, &self.tracker)
            .expect("candidate send");
        let update = if preserves {
            ActiveOrderUpdate::Preserve
        } else {
            active_order_update_for_sent(&self.orders, issued.unit, sequence, kind)
        };
        match update {
            ActiveOrderUpdate::Preserve => {}
            ActiveOrderUpdate::Replace(next) => {
                if let Some(kind) = next {
                    self.local
                        .set_active_order_from_issued(tick, kind, issued)
                        .expect("active");
                } else {
                    self.local.set_active_order(tick, None).expect("inactive");
                }
                self.pending = Some((sequence, previous));
            }
        }
    }

    fn observe(&mut self, edit: impl FnOnce(&mut bota_proto::WorldView)) {
        let mut view = self.tracker.current().expect("view").clone();
        view.tick += 1;
        edit(&mut view);
        view.units.sort_by_key(|unit| unit.id);
        self.tracker
            .observe_snapshot(&view)
            .expect("valid new seat snapshot");
        self.orders
            .reconcile_neural_snapshot(&self.tracker, &mut self.local, &mut self.pending)
            .expect("reconcile");
    }

    fn reject(&mut self, sequence: u32) -> bool {
        let rejected = self.orders.observe_rejection(sequence);
        if let Some((pending, previous)) = self.pending
            && pending == sequence
        {
            self.local
                .restore_active_order(self.tracker.current().expect("tick").tick, previous)
                .expect("rollback");
            self.pending = None;
        }
        rejected
    }

    fn attack(&self) -> IssuedOrder {
        issued(
            None,
            Order::Attack {
                target: Target::Unit(self.target),
            },
        )
    }

    fn hide_target(&mut self) {
        let target = self.target;
        self.observe(|view| view.units.retain(|unit| unit.id != target));
    }
}

#[test]
fn candidate_own_cast_preserves_exact_body_target_start_and_rejection_link() {
    let mut fixture = Fixture::new();
    let attack = fixture.attack();
    fixture.send(1, attack, ActionKind::AttackUnit);
    let original = fixture.local.active_order();
    fixture.observe(|_| {});
    fixture.send(2, own_cast(None, 0), ActionKind::Cast);

    assert_eq!(fixture.orders.active_body_for(None), Some((1, attack)));
    assert_eq!(fixture.local.active_order(), original);
    assert_eq!(fixture.orders.last_sequence(), Some(2));
    assert!(
        !fixture.reject(2),
        "rejected own cast never replaced the body"
    );
    assert_eq!(fixture.local.active_order(), original);
    assert!(
        fixture.reject(1),
        "the bounded preceding attack rejection is still supported"
    );
    assert_eq!(fixture.orders.active_body_for(None), None);
    assert_eq!(fixture.local.active_order(), None);
}

#[test]
fn rejected_interrupt_after_an_own_cast_restores_active_and_body_together() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let original = fixture.local.active_order();
    fixture.send(
        2,
        issued(
            None,
            Order::Use {
                slot: ItemSlot(0),
                target: Target::None,
            },
        ),
        ActionKind::Use,
    );
    fixture.send(3, own_cast(None, 0), ActionKind::Cast);
    assert!(fixture.reject(2));

    assert_eq!(
        fixture.orders.active_body_for(None),
        Some((1, fixture.attack()))
    );
    assert_eq!(
        fixture.local.active_order(),
        original,
        "a preserving cast must not discard the earlier interrupt's pending rollback"
    );
}

#[test]
fn visible_target_outside_cap96_is_not_a_visibility_loss() {
    let mut fixture = Fixture::new();
    let mut creep = fixture.tracker.own_hero().expect("template").clone();
    creep.kind = UnitKind::CreepMelee;
    creep.owner = None;
    creep.hero = None;
    creep.abilities.clear();
    creep.items.clear();
    let target = EntityId {
        idx: 20_000,
        generation: 1,
    };
    fixture.observe(|view| {
        for index in 0..100 {
            let mut nearby = creep.clone();
            nearby.id = EntityId {
                idx: 10_000 + index,
                generation: 1,
            };
            view.units.push(nearby);
        }
        creep.id = target;
        creep.pos = Vec2::from_ints(18_000, 18_000);
        view.units.push(creep);
    });
    let attack = issued(
        None,
        Order::Attack {
            target: Target::Unit(target),
        },
    );
    fixture.send(1, attack, ActionKind::AttackUnit);
    let original = fixture.local.active_order();
    fixture.observe(|_| {});

    let space = ActionSpace::from_tracker(&fixture.tracker).expect("capped space");
    assert_eq!(space.entity_candidates().len(), 96);
    assert_eq!(space.entity_index(target), None);
    assert_eq!(fixture.orders.should_send(Some(attack)), None);
    assert_eq!(fixture.local.active_order(), original);
}

#[test]
fn hidden_target_fallback_is_truthful_and_reappearance_does_not_reset_it() {
    let mut fixture = Fixture::new();
    let target_view = fixture
        .tracker
        .entity(fixture.target)
        .expect("observed target")
        .unit
        .clone();
    let attack = fixture.attack();
    fixture.send(1, attack, ActionKind::AttackUnit);
    fixture.hide_target();
    let fallback = issued(
        None,
        Order::Attack {
            target: Target::Pos(target_view.pos),
        },
    );
    let active = fixture.local.active_order();

    assert_eq!(fixture.orders.active_body_for(None), Some((1, fallback)));
    assert_eq!(
        active,
        Some(ActivePolicyOrder {
            started_tick: 2,
            kind: ActionKind::AttackMovePoint,
            target: ActivePolicyTarget::Point(target_view.pos)
        })
    );
    fixture.observe(|view| view.units.push(target_view));
    assert_eq!(fixture.orders.should_send(Some(attack)), Some(attack));
    assert_eq!(fixture.orders.should_send(Some(fallback)), None);
    assert_eq!(fixture.local.active_order(), active);
    assert_eq!(
        fixture.orders.last_sequence(),
        Some(1),
        "fallback never invents a sent sequence"
    );
}

#[test]
fn rejection_restores_reconciled_fallback_not_a_stale_unit_target() {
    let mut fixture = Fixture::new();
    let attack = fixture.attack();
    let target_position = fixture
        .tracker
        .entity(fixture.target)
        .expect("target")
        .unit
        .pos;
    fixture.send(1, attack, ActionKind::AttackUnit);
    fixture.send(
        2,
        issued(
            None,
            Order::Move {
                target: Target::Pos(Vec2::from_ints(9000, 9000)),
            },
        ),
        ActionKind::MovePoint,
    );
    fixture.hide_target();
    assert!(fixture.reject(2));

    assert_eq!(
        fixture.orders.active_body_for(None),
        Some((
            1,
            issued(
                None,
                Order::Attack {
                    target: Target::Pos(target_position)
                }
            )
        ))
    );
    assert_eq!(
        fixture.local.active_order(),
        Some(ActivePolicyOrder {
            started_tick: 2,
            kind: ActionKind::AttackMovePoint,
            target: ActivePolicyTarget::Point(target_position)
        })
    );
    assert_eq!(fixture.orders.should_send(Some(attack)), Some(attack));
}

#[test]
fn forgotten_target_clears_unknown_state_without_claiming_stop() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let target = fixture.target;
    let mut view = fixture.tracker.current().expect("view").clone();
    view.tick += crate::HISTORY_TICKS + 1;
    view.units.retain(|unit| unit.id != target);
    fixture
        .tracker
        .observe_snapshot(&view)
        .expect("history expires");
    fixture
        .orders
        .reconcile_neural_snapshot(&fixture.tracker, &mut fixture.local, &mut fixture.pending)
        .expect("unknown fallback");

    assert!(fixture.tracker.entity(target).is_none());
    assert_eq!(fixture.local.active_order(), None);
    assert_eq!(fixture.orders.active_body_order_for(None), None);
    assert_eq!(fixture.orders.last_sequence(), Some(1));
}

#[test]
fn missing_snapshot_gap_does_not_invent_the_server_fallback_position() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let mut view = fixture.tracker.current().expect("view").clone();
    view.tick = 4;
    view.units.retain(|unit| unit.id != fixture.target);
    fixture
        .tracker
        .observe_snapshot(&view)
        .expect("valid snapshot gap");
    fixture
        .orders
        .reconcile_neural_snapshot(&fixture.tracker, &mut fixture.local, &mut fixture.pending)
        .expect("unknown fallback");

    assert_eq!(
        fixture.orders.active_body_order_for(None),
        None,
        "the server may have seen a later target position during the missing snapshots"
    );
    assert_eq!(fixture.local.active_order(), None);
    assert_eq!(fixture.orders.last_sequence(), Some(1));
}

#[test]
fn observed_death_without_final_position_does_not_invent_a_fallback() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let mut view = fixture.tracker.current().expect("view").clone();
    view.tick = 2;
    view.units.retain(|unit| unit.id != fixture.target);
    fixture.tracker.observe_snapshot(&view).expect("snapshot");
    fixture
        .tracker
        .observe_events(
            2,
            &[bota_proto::EventKind::Died {
                unit: fixture.target,
                killer: None,
                denied: false,
                gold: 0,
            }],
        )
        .expect("seat-visible death");
    fixture
        .orders
        .reconcile_neural_snapshot(&fixture.tracker, &mut fixture.local, &mut fixture.pending)
        .expect("unknown final position");

    assert_eq!(
        fixture.orders.active_body_order_for(None),
        None,
        "death may follow same-tick movement absent from the final snapshot"
    );
    assert_eq!(fixture.local.active_order(), None);
}

#[test]
fn courier_follow_fallback_does_not_replace_the_hero_active_order() {
    let mut fixture = Fixture::new();
    fixture.send(
        1,
        issued(
            None,
            Order::Attack {
                target: Target::None,
            },
        ),
        ActionKind::Hold,
    );
    let hero_active = fixture.local.active_order();
    let position = fixture
        .tracker
        .entity(fixture.target)
        .expect("target")
        .unit
        .pos;
    let follow = issued(
        Some(fixture.courier),
        Order::Move {
            target: Target::Unit(fixture.target),
        },
    );
    fixture.send(2, follow, ActionKind::FollowUnit);
    fixture.hide_target();

    assert_eq!(
        fixture.orders.active_body_for(Some(fixture.courier)),
        Some((
            2,
            issued(
                Some(fixture.courier),
                Order::Move {
                    target: Target::Pos(position)
                }
            )
        ))
    );
    assert_eq!(fixture.local.active_order(), hero_active);
    assert_eq!(fixture.orders.should_send(Some(follow)), Some(follow));
}

#[test]
fn courier_generation_change_invalidates_both_current_and_bounded_rollback() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let hero_active = fixture.local.active_order();
    let courier = fixture.courier;
    fixture.send(
        2,
        issued(
            Some(courier),
            Order::Move {
                target: Target::Pos(Vec2::from_ints(8000, 8000)),
            },
        ),
        ActionKind::MovePoint,
    );
    fixture.send(
        3,
        issued(
            Some(courier),
            Order::Attack {
                target: Target::None,
            },
        ),
        ActionKind::Hold,
    );
    fixture.observe(|view| {
        view.units
            .iter_mut()
            .find(|unit| unit.id == courier)
            .expect("courier")
            .id
            .generation += 1;
    });
    assert_eq!(fixture.orders.active_body_for(Some(courier)), None);
    fixture.reject(3);
    assert_eq!(fixture.orders.active_body_for(Some(courier)), None);
    assert_eq!(fixture.local.active_order(), hero_active);
}

#[test]
fn hero_death_and_respawn_do_not_restore_orders_from_the_previous_body() {
    let mut fixture = Fixture::new();
    fixture.send(
        1,
        issued(
            None,
            Order::Attack {
                target: Target::None,
            },
        ),
        ActionKind::Hold,
    );
    fixture.send(2, fixture.attack(), ActionKind::AttackUnit);
    let mut hero = fixture.tracker.own_hero().expect("hero").clone();
    fixture.observe(|view| {
        view.players[0].unit = None;
        view.players[0].kit = Some(bota_proto::Kit {
            abilities: hero.abilities.clone(),
            items: hero.items.clone(),
        });
        view.units.retain(|unit| unit.id != hero.id);
    });
    assert_eq!(fixture.orders.active_body_order_for(None), None);
    assert_eq!(fixture.local.active_order(), None);
    hero.id.generation += 1;
    fixture.observe(|view| {
        view.players[0].unit = Some(hero.id);
        view.players[0].kit = None;
        view.units.push(hero);
    });
    fixture.reject(2);
    assert_eq!(fixture.orders.active_body_order_for(None), None);
    assert_eq!(fixture.local.active_order(), None);
}

#[test]
fn replacement_target_generation_is_not_the_old_visible_target() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let old = fixture.target;
    let replacement = EntityId {
        generation: old.generation + 1,
        ..old
    };
    let position = fixture
        .tracker
        .entity(old)
        .expect("old observation")
        .unit
        .pos;
    fixture.observe(|view| {
        view.units
            .iter_mut()
            .find(|unit| unit.id == old)
            .expect("target")
            .id = replacement;
        view.players[1].unit = Some(replacement);
    });
    assert_eq!(
        fixture.orders.active_body_order_for(None),
        Some(issued(
            None,
            Order::Attack {
                target: Target::Pos(position)
            }
        ))
    );
    assert_eq!(
        fixture.local.active_order().expect("known fallback").kind,
        ActionKind::AttackMovePoint
    );
    let attack = issued(
        None,
        Order::Attack {
            target: Target::Unit(replacement),
        },
    );
    assert_eq!(fixture.orders.should_send(Some(attack)), Some(attack));
}

#[test]
fn courier_cast_and_unverified_hero_casts_and_use_remain_interrupting() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    let hero_active = fixture.local.active_order();
    fixture.send(
        2,
        issued(
            Some(fixture.courier),
            Order::Attack {
                target: Target::None,
            },
        ),
        ActionKind::Hold,
    );
    fixture.send(3, own_cast(Some(fixture.courier), 0), ActionKind::Cast);
    assert_eq!(fixture.orders.active_body_for(Some(fixture.courier)), None);
    assert_eq!(fixture.local.active_order(), hero_active);
    fixture.send(
        4,
        issued(
            None,
            Order::Use {
                slot: ItemSlot(0),
                target: Target::None,
            },
        ),
        ActionKind::Use,
    );
    assert_eq!(fixture.orders.active_body_for(None), None);
    assert_eq!(fixture.local.active_order(), None);

    for (ability, aim) in [(AbilityId(1), Aim::Own), (AbilityId(13), Aim::Point)] {
        let mut fixture = Fixture::new();
        fixture.observe(|view| {
            let hero = view
                .units
                .iter_mut()
                .find(|unit| unit.owner == Some(SlotId(0)) && unit.kind == UnitKind::Hero)
                .expect("hero");
            hero.abilities[0].id = ability;
            hero.abilities[0].aim = aim;
        });
        fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
        fixture.send(2, own_cast(None, 0), ActionKind::Cast);
        assert_eq!(fixture.orders.active_body_order_for(None), None);
        assert_eq!(fixture.local.active_order(), None);
    }
}

#[test]
fn reconciliation_chronology_error_is_atomic() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    fixture
        .local
        .note_decision(10, ActionKind::Continue)
        .expect("future local decision");
    let mut view = fixture.tracker.current().expect("view").clone();
    view.tick = 2;
    view.units.retain(|unit| unit.id != fixture.target);
    fixture.tracker.observe_snapshot(&view).expect("snapshot");
    let before = (fixture.orders, fixture.local, fixture.pending);
    let error = fixture
        .orders
        .reconcile_neural_snapshot(&fixture.tracker, &mut fixture.local, &mut fixture.pending)
        .expect_err("regressed local update");

    assert_eq!(
        error.to_string(),
        "local policy tick 2 is older than latest tick 10"
    );
    assert_eq!((fixture.orders, fixture.local, fixture.pending), before);
}

#[test]
fn older_than_one_body_transition_rejections_are_explicit_noops() {
    let mut fixture = Fixture::new();
    fixture.send(1, fixture.attack(), ActionKind::AttackUnit);
    fixture.send(
        2,
        issued(
            None,
            Order::Attack {
                target: Target::None,
            },
        ),
        ActionKind::Hold,
    );
    fixture.send(
        3,
        issued(
            None,
            Order::Move {
                target: Target::None,
            },
        ),
        ActionKind::Stop,
    );
    let before = (fixture.orders, fixture.local, fixture.pending);

    assert!(!fixture.reject(1));
    assert_eq!((fixture.orders, fixture.local, fixture.pending), before);
    assert!(fixture.reject(3));
    assert_eq!(
        fixture.local.active_order().expect("restored Hold").kind,
        ActionKind::Hold
    );
}

#[test]
fn candidate_sequence_error_does_not_mutate_either_candidate_or_legacy_shadow() {
    let fixture = Fixture::new();
    let mut legacy = OrderPersistence::default();
    let mut candidate = Some(OrderPersistence::default());
    crate::record_sent_for_policy(
        &mut legacy,
        &mut candidate,
        5,
        fixture.attack(),
        &fixture.tracker,
    )
    .expect("send");
    let before = (legacy, candidate);
    let error = crate::record_sent_for_policy(
        &mut legacy,
        &mut candidate,
        5,
        own_cast(None, 0),
        &fixture.tracker,
    )
    .expect_err("duplicate sequence");

    assert_eq!(
        error.to_string(),
        "order sequence 5 must be greater than last sent sequence 5"
    );
    assert_eq!((legacy, candidate), before);
}

#[test]
fn teacher_labeler_receives_unchanged_legacy_persistence_when_candidate_reconciles() {
    let mut fixture = Fixture::new();
    let mut legacy = OrderPersistence::default();
    let mut candidate = Some(OrderPersistence::default());
    let mut reference = OrderPersistence::default();
    let mut teacher = Teacher::new();
    let mut baseline = teacher.clone();
    for (sequence, order) in [
        (1, fixture.attack()),
        (2, own_cast(None, 0)),
        (3, fixture.attack()),
    ] {
        crate::record_sent_for_policy(
            &mut legacy,
            &mut candidate,
            sequence,
            order,
            &fixture.tracker,
        )
        .expect("candidate plus legacy shadow");
        reference
            .record_sent(sequence, order)
            .expect("historical reference");
        teacher.note_sent(sequence, order, 1);
        baseline.note_sent(sequence, order, 1);
        assert_eq!(legacy, reference);
        if sequence == 2 {
            assert_eq!(legacy.active_body_order_for(None), None);
            assert_eq!(
                candidate.as_ref().expect("candidate").active_body_for(None),
                Some((1, fixture.attack()))
            );
        }
    }
    fixture.hide_target();
    candidate
        .as_mut()
        .expect("candidate")
        .reconcile_neural_snapshot(&fixture.tracker, &mut fixture.local, &mut fixture.pending)
        .expect("candidate visibility update");
    let readiness = ItemReadiness::new();
    assert_eq!(
        legacy, reference,
        "observation must not reconcile Teacher's legacy ledger"
    );
    assert_eq!(legacy.active_body_order_for(None), Some(fixture.attack()));
    assert_eq!(
        teacher
            .decide(&fixture.tracker, &legacy, &readiness)
            .expect("Teacher labels")
            .0,
        baseline
            .decide(&fixture.tracker, &reference, &readiness)
            .expect("historical labels")
            .0
    );
    assert_eq!(teacher, baseline);
}

fn issued(unit: Option<EntityId>, order: Order) -> IssuedOrder {
    IssuedOrder { unit, order }
}

fn own_cast(unit: Option<EntityId>, slot: u8) -> IssuedOrder {
    issued(
        unit,
        Order::Cast {
            slot: AbilitySlot(slot),
            target: Target::None,
        },
    )
}
