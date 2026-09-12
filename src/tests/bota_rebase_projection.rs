use super::feature::encode;
use crate::{FeatureFrame, LocalPolicyState, StateTracker, unit_feature};
use bota_proto::{EffectId, EventKind, Fixed, MapId, ServerMsg, SlotId, Target};
use bota_server::game::{Status, StatusKind, wire_id};

#[test]
fn rebased_upstream_effect_catalog_and_15_minute_cap_match_inference_contract() {
    assert_eq!(
        bota_server::game::EFFECT_SHADOWRAZE,
        crate::SHADOWRAZE_EFFECT_ID.0
    );
    assert_eq!(
        bota_server::game::rules::AURA_LINGER_TICKS,
        crate::AURA_EFFECT_TICKS
    );
    assert_eq!(bota_server::game::MAP2_TICK_CAP, 27_900);
    assert_eq!(bota_server::game::MAP2_GAME_TICKS, 27_000);
    assert_eq!(bota_server::game::ITEM_MANGO, 42);
}

#[test]
fn rebase_arena_and_wire_frames_match_after_manual_mana_use_with_visible_auras() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 918241,
    })
    .expect("arena");
    let ServerMsg::MatchStart { info } = &start.messages[0][0] else {
        panic!("match start")
    };
    let mut native = [0, 1].map(|slot| StateTracker::new(SlotId(slot), info).expect("tracker"));
    let mut socket =
        [0, 1].map(|slot| StateTracker::new(SlotId(slot), info).expect("wire tracker"));
    let configured = arena.configure_for_test(|world| {
        for slot in 0..2 {
            let body = world.seats[slot].unit.expect("hero");
            let enemy = world.seats[1 - slot].unit.expect("enemy");
            put_visible_effects(world, body, enemy);
            world.inventory.get_mut(body).expect("inventory").slots[0] =
                bota_server::game::ItemStack::bought(
                    bota_proto::ItemId(42),
                    SlotId(slot as u8),
                    world.tick,
                );
            world.mana.get_mut(body).expect("mana").mana = Fixed::from_int(1);
        }
    });
    observe_arena_pair(&mut native, &mut socket, &configured);
    let request = Some(crate::Request {
        seq: 1,
        unit: None,
        order: bota_proto::Order::Use {
            slot: bota_proto::ItemSlot(0),
            target: Target::None,
        },
    });
    let used = arena.step(&[request, request]).expect("manual use");
    observe_arena_pair(&mut native, &mut socket, &used);
    let next = arena.step(&[None, None]).expect("next tick");
    observe_arena_pair(&mut native, &mut socket, &next);
    for seat in &native {
        let frame = encode(seat, &LocalPolicyState::new(0));
        assert_eq!(
            frame.own_units()[0][unit_feature::MANA_RESTORE_REPORT_PRESENT],
            1.0
        );
        assert_eq!(
            frame.own_units()[0][unit_feature::HEALTH_RESTORE_REPORT_PRESENT],
            0.0
        );
        assert!(frame.own_units()[0][unit_feature::GUARDED_TICKS_LEFT] > 0.0);
        assert!(frame.own_units()[0][unit_feature::INSPIRED_TICKS_LEFT] > 0.0);
    }
}

fn observe_arena_pair(
    native: &mut [StateTracker; 2],
    socket: &mut [StateTracker; 2],
    step: &crate::ArenaStep,
) {
    for slot in 0..2 {
        for message in &step.messages[slot] {
            observe(&mut native[slot], message);
            let bytes = bota_proto::encode_frame_to_vec(message).expect("encode");
            let decoded = bota_proto::decode_payload::<ServerMsg>(&bytes[4..]).expect("decode");
            observe(&mut socket[slot], &decoded);
        }
        assert_eq!(
            encode(&native[slot], &LocalPolicyState::new(0)),
            encode(&socket[slot], &LocalPolicyState::new(0))
        );
        assert_eq!(
            native[slot].map2_reward_state(),
            socket[slot].map2_reward_state()
        );
    }
}

#[test]
fn rebase_native_and_wire_snapshot_events_produce_identical_inference_on_both_seats() {
    for slot in 0..2 {
        let (mut world, info) = super::map2_actions::server_world();
        let body = world.seats[slot].unit.expect("hero");
        let target = world.seats[1 - slot].unit.expect("opposing caster");
        let side = world.seats[slot].team;
        assert_eq!(info.map, MapId(2));
        let mut native = StateTracker::new(SlotId(slot as u8), &info).expect("native observer");
        let mut socket = StateTracker::new(SlotId(slot as u8), &info).expect("wire observer");
        for observer in [&mut native, &mut socket] {
            observer
                .observe_snapshot(&world.view(side))
                .expect("baseline");
            observer.observe_events(1, &[]).expect("events");
        }
        put_visible_effects(&mut world, body, target);
        world.tick = 2;
        world.inventory.get_mut(body).expect("bag").slots[0] =
            bota_server::game::ItemStack::bought(bota_proto::ItemId(42), SlotId(slot as u8), 1);
        world.stats.get_mut(body).expect("stats").max_mana = Fixed::from_int(500);
        world.mana.get_mut(body).expect("mana").mana = Fixed::from_int(400);
        let mut events = Vec::new();
        assert!(world.use_item(body, 0, Target::None, &mut events));
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].kind,
            EventKind::Healed {
                source: Some(wire_id(body)),
                target: wire_id(body),
                amount: 0,
                mana: 100
            }
        );
        let report = events.remove(0).kind;
        for tick in 2..=3 {
            world.tick = tick;
            let snapshot = ServerMsg::Snapshot {
                view: world.view(side),
            };
            let events = ServerMsg::Events {
                tick,
                events: if tick == 2 {
                    vec![report.clone()]
                } else {
                    vec![]
                },
            };
            for message in [snapshot, events] {
                observe(&mut native, &message);
                let bytes = bota_proto::encode_frame_to_vec(&message).expect("native wire encode");
                let decoded =
                    bota_proto::decode_payload::<ServerMsg>(&bytes[4..]).expect("wire decode");
                assert_eq!(message, decoded);
                observe(&mut socket, &decoded);
            }
        }
        let frame = encode(&native, &LocalPolicyState::new(0));
        assert_eq!(frame, encode(&socket, &LocalPolicyState::new(0)));
        assert_eq!(frame, encode(&native.clone(), &LocalPolicyState::new(0)));
        assert_projected_features(&frame);
        assert_eq!(native.map2_reward_state(), socket.map2_reward_state());
    }
}

fn put_visible_effects(
    world: &mut bota_server::game::World,
    body: bota_server::game::Entity,
    source: bota_server::game::Entity,
) {
    let mut statuses = world.statuses.remove(body).unwrap_or_default();
    for kind in [
        StatusKind::Guarded {
            armor: 2,
            hp_per_second: 1,
        },
        StatusKind::Inspired { hp_per_second: 2 },
    ] {
        statuses.put(Status {
            kind,
            ticks_left: 15,
        });
    }
    statuses.stack_raze(source);
    world.statuses.insert(body, statuses);
    let view = world.view(world.team.get(body).copied().expect("team"));
    let unit = view
        .units
        .iter()
        .find(|unit| unit.id == wire_id(body))
        .expect("visible hero");
    assert!(unit.effects.iter().any(|effect| effect.id == EffectId(13)));
    assert!(unit.effects.iter().any(|effect| effect.id == EffectId(14)));
    assert!(unit.effects.iter().any(|effect| effect.id == EffectId(15)));
}

fn observe(tracker: &mut StateTracker, message: &ServerMsg) {
    match message {
        ServerMsg::Snapshot { view } => tracker.observe_snapshot(view).expect("snapshot"),
        ServerMsg::Events { tick, events } => {
            tracker.observe_events(*tick, events).expect("events")
        }
        _ => panic!("fixture must contain snapshots/events"),
    }
}

fn assert_projected_features(frame: &FeatureFrame) {
    let row = &frame.own_units()[0];
    assert_eq!(row[unit_feature::GUARDED_TICKS_LEFT], 1.0);
    assert_eq!(row[unit_feature::INSPIRED_TICKS_LEFT], 1.0);
    assert_eq!(row[unit_feature::RAZE_EFFECT_PRESENT], 1.0);
    assert_eq!(row[unit_feature::MANA_RESTORE_REPORT_PRESENT], 1.0);
    assert_eq!(row[unit_feature::HEALTH_RESTORE_REPORT_PRESENT], 0.0);
}
