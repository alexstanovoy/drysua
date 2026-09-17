#![allow(
    clippy::float_arithmetic,
    reason = "Bounded reward-state normalization assertions."
)]

use super::feature::{encode, match_info, world_view};
use crate::{ActionSpace, LocalPolicyState, Map2RewardEnd, StateTracker, global_feature};
use bota_proto::{EventKind, ItemId, MapId, SlotId, Team, UnitKind, WorldView};

#[test]
fn wait_feature_columns_stay_fixed_before_the_three_new_progress_fields() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 20);
    assert_eq!(crate::GLOBAL_FEATURES, 92);
    assert_eq!(crate::UNIT_FEATURES, 84);
    assert_eq!(global_feature::MAP2_FOUNTAIN_WAIT_TICKS, 85);
    assert_eq!(global_feature::MAP2_FOUNTAIN_WAIT_REFUNDABLE_COST, 86);
}

#[test]
fn wait_features_normalize_open_period_ticks_and_refundable_cost_without_extra_state() {
    let (mut tracker, mut view) = waiting_tracker();
    advance_to(&mut tracker, &mut view, 31);
    let state = tracker.map2_reward_state().expect("reward");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(state.fountain_wait_ticks, 30);
    assert_eq!(frame.global()[85], 30.0 / 27_900.0);
    assert_eq!(
        frame.global()[86],
        state.fountain_wait_refundable_cost / 0.093
    );
    assert!(frame.is_finite());
}

#[test]
fn wait_features_stay_bounded_at_native_cap() {
    let (mut tracker, mut view) = waiting_tracker();
    advance_to(&mut tracker, &mut view, crate::MAP2_TICK_CAP);
    let state = tracker.map2_reward_state().expect("reward");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(state.fountain_wait_ticks, 27_899);
    assert_eq!(frame.global()[85], 27_899.0 / 27_900.0);
    assert!(frame.global()[86] > 0.0);
    assert!(frame.global()[86] <= 1.0);
}

#[test]
fn any_own_purchase_cancels_wait_even_after_drain_and_resets_both_feature_columns() {
    let (mut tracker, mut view) = waiting_tracker();
    advance_to(&mut tracker, &mut view, 31);
    let prior = tracker.provenance();
    let old_space = ActionSpace::from_tracker(&tracker).expect("space");
    let charged = tracker
        .take_map2_reward_interval()
        .expect("charged interval");
    assert!(prior.matches(&tracker));
    let mut branch = tracker.clone();
    view.tick = 32;
    let events = [EventKind::ItemBought {
        slot: SlotId(0),
        item: ItemId(0),
    }];
    for observer in [&mut tracker, &mut branch] {
        observer.observe_snapshot(&view).expect("snapshot");
        observer
            .observe_events(32, &events)
            .expect("cheap purchase");
    }
    let refund = tracker.take_map2_reward_interval().expect("refund");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(refund.fountain_wait_refund, -charged.fountain_wait);
    assert_eq!(refund.fountain_wait, 0.0);
    assert_eq!(&frame.global()[85..87], &[0.0; 2]);
    assert_eq!(frame, encode(&branch, &LocalPolicyState::new(0)));
    assert_eq!(
        branch.take_map2_reward_interval().expect("clone refund"),
        refund
    );
    assert!(!old_space.matches_tracker(&tracker));
}

#[test]
fn movement_or_resource_break_resets_wait_without_refunding_and_enemy_buy_does_not_cancel() {
    for change in 0..3 {
        let (mut tracker, mut view) = waiting_tracker();
        advance_to(&mut tracker, &mut view, 31);
        tracker.take_map2_reward_interval().expect("drain");
        view.tick = 32;
        match change {
            0 => view.units[0].pos.x.raw += 1,
            1 => view.units[0].mana -= 1,
            _ => view.units[0].hp -= 1,
        }
        tracker.observe_snapshot(&view).expect("snapshot");
        tracker
            .observe_events(
                32,
                &[EventKind::ItemBought {
                    slot: SlotId(1),
                    item: ItemId(0),
                }],
            )
            .expect("enemy buy");
        let interval = tracker.take_map2_reward_interval().expect("drain");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        assert_eq!(&frame.global()[85..87], &[0.0; 2]);
        assert_eq!(interval.fountain_wait_refund, 0.0);
    }
}

#[test]
fn legacy_maps_leave_both_wait_accounting_columns_zero() {
    for map in [MapId(0), MapId(1)] {
        let mut info = match_info(Team::Radiant);
        info.map = map;
        let mut tracker = StateTracker::new(SlotId(0), &info).expect("legacy tracker");
        tracker.observe_snapshot(&waiting_view()).expect("snapshot");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        assert_eq!(&frame.global()[85..87], &[0.0; 2]);
    }
}

#[test]
fn segmented_interval_refunds_preserve_total_components_without_adding_a_terminal_charge() {
    let (mut whole, mut view) = waiting_tracker();
    let mut split = whole.clone();
    let mut pieces = Vec::new();
    for tick in 2..=65 {
        view.tick = tick;
        let events = if tick == 62 {
            vec![EventKind::ItemBought {
                slot: SlotId(0),
                item: ItemId(0),
            }]
        } else {
            vec![]
        };
        for observer in [&mut whole, &mut split] {
            observer.observe_snapshot(&view).expect("snapshot");
            observer.observe_events(tick, &events).expect("events");
        }
        if tick % 7 == 0 {
            pieces.push(split.take_map2_reward_interval().expect("segment"));
        }
    }
    pieces.push(split.finish_map2_reward(Map2RewardEnd::Draw).expect("end"));
    let total = whole
        .finish_map2_reward(Map2RewardEnd::Draw)
        .expect("whole end");
    assert!((pieces.iter().map(|piece| piece.total).sum::<f64>() - total.total).abs() < 1.0e-12);
    assert_eq!(
        pieces
            .iter()
            .map(|piece| piece.observations.fountain_wait_refunds)
            .sum::<u64>(),
        1
    );
    for piece in pieces {
        let sum = piece.gold
            + piece.experience
            + piece.hero_damage
            + piece.hero_damage_taken
            + piece.creep_damage_taken
            + piece.other_damage_taken
            + piece.mana_spent
            + piece.tower_health
            + piece.lane_pressure
            + piece.pregame_movement
            + piece.fountain_wait
            + piece.fountain_wait_refund
            + piece.stagnation_base
            + piece.stagnation_ticks_cost
            + piece.terminal;
        assert_eq!(piece.total, sum);
    }
    assert_eq!(whole.map2_reward_state(), split.map2_reward_state());
}

fn waiting_tracker() -> (StateTracker, WorldView) {
    let mut info = match_info(Team::Radiant);
    info.map = MapId(2);
    info.pregame_ticks = 0;
    let view = waiting_view();
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker.observe_events(1, &[]).expect("events");
    (tracker, view)
}

fn waiting_view() -> WorldView {
    let mut view = world_view(Team::Radiant, 1);
    view.units[0].hp = view.units[0].max_hp;
    view.units[0].mana = view.units[0].max_mana;
    let position = view.units[0].pos;
    view.units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Fountain && unit.team == Team::Radiant)
        .expect("fountain")
        .pos = position;
    view
}

fn advance_to(tracker: &mut StateTracker, view: &mut WorldView, tick: u32) {
    assert!(tick <= crate::MAP2_TICK_CAP);
    assert!(tick > view.tick);
    for next in view.tick + 1..=tick {
        view.tick = next;
        tracker.observe_snapshot(view).expect("snapshot");
        tracker.observe_events(next, &[]).expect("events");
    }
}
