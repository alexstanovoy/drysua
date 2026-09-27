use super::bota_rebase_inference::heal;
use super::map2_inference::frame;
use crate::{FeatureFrame, StateTracker, unit_feature};
use bota_proto::{Fixed, ItemId, MapId, MatchInfo, ServerMsg, SlotId, Target};
use bota_server::game::{Entity, ItemStack, Modifier, ModifierKind, World, wire_id};

#[test]
fn native_clamped_mana_report_projects_identically_through_wire_on_both_seats() {
    for slot in 0..2 {
        let (mut world, info) = super::map2_actions::server_world();
        let side = world.seats[slot].team;
        let mut pair = observers(&info, slot as u8);
        project(
            &mut pair,
            &[
                ServerMsg::Snapshot {
                    view: world.view(side),
                },
                ServerMsg::Events {
                    tick: 1,
                    events: vec![],
                },
            ],
        );
        let body = configure(&mut world, slot, 400);
        world.stats.get_mut(body).expect("stats").max_mana = Fixed::from_int(500);
        world.tick = 2;
        let mut events = Vec::new();
        assert!(world.use_item(body, 0, Target::None, &mut events));
        let report = heal(Some(wire_id(body)), wire_id(body), 0, 100);
        assert_eq!(
            events.iter().map(|event| &event.kind).collect::<Vec<_>>(),
            [&report]
        );
        for tick in 2..=3 {
            world.tick = tick;
            let reports = if tick == 2 {
                events.iter().map(|event| event.kind.clone()).collect()
            } else {
                vec![]
            };
            let output = project(
                &mut pair,
                &[
                    ServerMsg::Snapshot {
                        view: world.view(side),
                    },
                    ServerMsg::Events {
                        tick,
                        events: reports,
                    },
                ],
            );
            if tick == 3 {
                let row = &output.own_units()[0];
                for column in [
                    unit_feature::RAZE_EFFECT_PRESENT,
                    unit_feature::MANA_RESTORE_REPORT_PRESENT,
                    unit_feature::GUARDED_TICKS_LEFT,
                    unit_feature::INSPIRED_TICKS_LEFT,
                ] {
                    assert_eq!(row[column], 1.0);
                }
                assert_eq!(row[unit_feature::HEALTH_RESTORE_REPORT_PRESENT], 0.0);
            }
        }
    }
}

fn observers(info: &MatchInfo, slot: u8) -> [StateTracker; 2] {
    assert_eq!(info.map, MapId(2));
    assert!(slot < 2);
    let native = StateTracker::new(SlotId(slot), info).expect("native tracker");
    [native.clone(), native]
}

fn project(pair: &mut [StateTracker; 2], messages: &[ServerMsg]) -> FeatureFrame {
    assert_eq!(messages.len(), 2);
    let [native, socket] = pair;
    for message in messages {
        let bytes = bota_proto::encode_frame_to_vec(message).expect("encode");
        let decoded = bota_proto::decode_payload::<ServerMsg>(&bytes[4..]).expect("decode");
        assert_eq!(message, &decoded);
        for (tracker, message) in [(&mut *native, message), (&mut *socket, &decoded)] {
            match message {
                ServerMsg::Snapshot { view } => tracker.observe_snapshot(view).expect("snapshot"),
                ServerMsg::Events { tick, events } => {
                    tracker.observe_events(*tick, events).expect("events")
                }
                _ => panic!("fixture must contain snapshots/events"),
            }
        }
    }
    let output = frame(native);
    assert_eq!(output, frame(socket));
    assert_eq!(native.map2_reward_state(), socket.map2_reward_state());
    let masks = [&*native, &*socket].map(|tracker| {
        *crate::ActionSpace::from_tracker(tracker)
            .expect("space")
            .kind_mask()
    });
    assert_eq!(masks[0], masks[1]);
    output
}

fn configure(world: &mut World, slot: usize, mana: i32) -> Entity {
    assert!(slot < 2);
    assert!(mana >= 0);
    let body = world.seats[slot].unit.expect("hero");
    let source = world.seats[1 - slot].unit.expect("enemy");
    let mut modifiers = world.modifiers.remove(body).unwrap_or_default();
    for kind in [
        ModifierKind::Guarded {
            armor: 2,
            hp_per_second: 1,
        },
        ModifierKind::Inspired { hp_per_second: 2 },
    ] {
        modifiers.put(Modifier {
            kind,
            source: Some(source),
            ticks_left: Some(15),
        });
    }
    modifiers.stack_raze(source);
    world.modifiers.insert(body, modifiers);
    world.inventory.get_mut(body).expect("inventory").slots[0] =
        ItemStack::bought(ItemId(42), SlotId(slot as u8), world.tick);
    world.mana.get_mut(body).expect("mana").mana = Fixed::from_int(mana);
    body
}
