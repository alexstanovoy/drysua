#![allow(
    clippy::float_arithmetic,
    reason = "Bounded accounting-feature assertions."
)]

use super::feature::{encode, match_info, reverse_entity_ids_and_generations, world_view};
use crate::{
    ActionSpace, FeatureEncoder, FeatureFrame, LocalPolicyState, StateTracker, global_feature,
};
use bota_proto::{EffectId, EffectView, EventKind, ItemId, MapId, SlotId, Team, WorldView};

#[test]
fn progress_debt_schema_adds_only_three_accounting_globals_after_wait_inputs() {
    assert_eq!(crate::FEATURE_SCHEMA_VERSION, 22);
    assert_eq!(crate::GLOBAL_FEATURES, 92);
    assert_eq!(crate::UNIT_FEATURES, 84);
    assert_eq!(global_feature::MAP2_STAGNATION_TICKS, 87);
    assert_eq!(global_feature::MAP2_ACTIVITY_TICKS_LEFT, 88);
    assert_eq!(global_feature::MAP2_STAGNATION_BASE_CHARGED, 89);
}

#[test]
fn progress_debt_baseline_is_free_and_idle_debt_is_normalized_at_threshold() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    assert_eq!(&encoded(&tracker).global()[87..90], &[0.0; 3]);
    advance_to(&mut tracker, &mut view, 2700);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[2699.0 / 2700.0, 0.0, 0.0]
    );
    advance_to(&mut tracker, &mut view, 2701);
    let interval = tracker.take_map2_reward_interval().expect("threshold");
    assert_eq!(&encoded(&tracker).global()[87..90], &[1.0, 0.0, 1.0]);
    assert_eq!(interval.stagnation_base, -0.02);
    assert_eq!(interval.stagnation_ticks_cost, 0.0);
    advance_to(&mut tracker, &mut view, 2702);
    let interval = tracker.take_map2_reward_interval().expect("capped cost");
    assert_eq!(interval.stagnation_base, 0.0);
    assert_eq!(interval.stagnation_ticks_cost, -0.000002);
    assert_eq!(&encoded(&tracker).global()[87..90], &[1.0, 0.0, 1.0]);
}

#[test]
fn progress_purchase_refunds_open_fountain_wait_but_only_partially_repays_generic_debt() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    let position = view.units[0].pos;
    view.units
        .iter_mut()
        .find(|unit| unit.kind == bota_proto::UnitKind::Fountain && unit.team == Team::Radiant)
        .expect("fountain")
        .pos = position;
    advance_to(&mut tracker, &mut view, 2701);
    let charged = tracker.take_map2_reward_interval().expect("charged");
    assert!(charged.fountain_wait < 0.0);
    feed(&mut tracker, &mut view, &[purchase()]);
    let refund = tracker.take_map2_reward_interval().expect("purchase");
    let frame = encoded(&tracker);
    assert_eq!(&frame.global()[85..87], &[0.0; 2]);
    assert_eq!(
        &frame.global()[87..90],
        &[2697.0 / 2700.0, 29.0 / 30.0, 1.0]
    );
    assert_eq!(refund.fountain_wait_refund, -charged.fountain_wait);
    assert_eq!(refund.stagnation_base, 0.0);
    assert_eq!(refund.stagnation_ticks_cost, 0.0);
    assert_eq!(refund.observations.stagnation_repaid_ticks, 3);
}

#[test]
fn progress_single_activity_lease_counts_current_tick_and_expires_after_thirty() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    advance_to(&mut tracker, &mut view, 101);
    feed(&mut tracker, &mut view, &[purchase()]);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[97.0 / 2700.0, 29.0 / 30.0, 0.0]
    );
    advance_to(&mut tracker, &mut view, 131);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[10.0 / 2700.0, 0.0, 0.0]
    );
    advance_to(&mut tracker, &mut view, 132);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[11.0 / 2700.0, 0.0, 0.0]
    );
}

#[test]
fn progress_direct_fountain_aura_repays_full_pool_lingering_body_until_latch_rearms() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    advance_to(&mut tracker, &mut view, 2701);
    view.units[0].effects.push(EffectView {
        id: EffectId(3),
        ticks_left: Some(1),
        stacks: None,
    });
    // The body stays full and outside the observed fountain; effect3 alone qualifies.
    advance_to(&mut tracker, &mut view, 3600);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[3.0 / 2700.0, 29.0 / 30.0, 1.0]
    );
    advance_to(&mut tracker, &mut view, 3601);
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[0.0, 29.0 / 30.0, 0.0]
    );
}

#[test]
fn progress_movement_and_death_do_not_reset_inactive_debt() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    advance_to(&mut tracker, &mut view, 11);
    view.units[0].pos.x.raw += 1;
    feed(&mut tracker, &mut view, &[]);
    assert_eq!(
        tracker.map2_reward_state().expect("state").stagnation_ticks,
        11
    );
    let hero = view.players[0].unit.take().expect("hero");
    view.players[0].respawn_left = 100;
    view.players[0].deaths = 1;
    view.units.retain(|unit| unit.id != hero);
    feed(
        &mut tracker,
        &mut view,
        &[EventKind::Died {
            unit: hero,
            killer: None,
            denied: false,
            gold: 0,
        }],
    );
    assert_eq!(
        tracker.map2_reward_state().expect("state").stagnation_ticks,
        12
    );
    assert_eq!(
        &encoded(&tracker).global()[87..90],
        &[12.0 / 2700.0, 0.0, 0.0]
    );
}

#[test]
fn progress_complete_pair_drains_clones_and_provenance_preserve_exact_accounting_state() {
    let (mut tracker, mut view) = fixture(Team::Radiant);
    advance_to(&mut tracker, &mut view, 2701);
    let before = tracker.provenance();
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    tracker.take_map2_reward_interval().expect("drain");
    assert!(before.matches(&tracker));
    let mut branch = tracker.clone();
    view.tick += 1;
    let mut encoder = FeatureEncoder::new(&tracker);
    for observer in [&mut tracker, &mut branch] {
        observer.observe_snapshot(&view).expect("pending snapshot");
    }
    assert_eq!(
        encoder
            .observe(&tracker)
            .expect_err("needs events")
            .to_string(),
        "feature Map2 reward requires complete Snapshot/Events for tick 2702"
    );
    for observer in [&mut tracker, &mut branch] {
        observer
            .observe_events(view.tick, &[purchase()])
            .expect("events");
    }
    encoder.observe(&tracker).expect("completed retry");
    assert!(!space.matches_tracker(&tracker));
    assert_eq!(encoded(&tracker), encoded(&branch));
    assert_eq!(tracker.map2_reward_state(), branch.map2_reward_state());
}

#[test]
fn progress_facts_are_off_map2_zero_and_side_handle_invariant() {
    for map in [MapId(0), MapId(1)] {
        let mut info = match_info(Team::Radiant);
        info.map = map;
        let mut tracker = StateTracker::new(SlotId(0), &info).expect("legacy");
        tracker
            .observe_snapshot(&world_view(Team::Radiant, 1))
            .expect("snapshot");
        assert_eq!(&encoded(&tracker).global()[87..90], &[0.0; 3]);
    }
    let mut frames = Vec::new();
    for side in [Team::Radiant, Team::Dire] {
        for remapped in [false, true] {
            let mut view = idle_view(side);
            if remapped {
                reverse_entity_ids_and_generations(&mut view, 80000, 29);
            }
            let mut tracker = observer(side, &view);
            advance_to(&mut tracker, &mut view, 2701);
            feed(&mut tracker, &mut view, &[purchase()]);
            let mut frame = encoded(&tracker);
            frame.global[4..6].fill(0.0);
            frames.push(frame);
        }
    }
    assert!(frames.windows(2).all(|frames| frames[0] == frames[1]));
}

fn fixture(side: Team) -> (StateTracker, WorldView) {
    let view = idle_view(side);
    (observer(side, &view), view)
}

fn observer(side: Team, view: &WorldView) -> StateTracker {
    let mut info = match_info(side);
    info.map = MapId(2);
    info.pregame_ticks = 0;
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(view).expect("snapshot");
    tracker.observe_events(view.tick, &[]).expect("baseline");
    tracker
}

fn idle_view(side: Team) -> WorldView {
    let mut view = world_view(side, 1);
    view.units[0].hp = view.units[0].max_hp;
    view.units[0].mana = view.units[0].max_mana;
    view.units[0].effects.clear();
    view
}

fn feed(tracker: &mut StateTracker, view: &mut WorldView, events: &[EventKind]) {
    assert!(view.tick < crate::MAP2_TICK_CAP);
    view.tick += 1;
    tracker.observe_snapshot(view).expect("snapshot");
    tracker.observe_events(view.tick, events).expect("events");
}

fn advance_to(tracker: &mut StateTracker, view: &mut WorldView, tick: u32) {
    assert!(tick <= crate::MAP2_TICK_CAP);
    assert!(tick >= view.tick);
    for _ in view.tick..tick {
        feed(tracker, view, &[]);
    }
}

fn purchase() -> EventKind {
    EventKind::ItemBought {
        slot: SlotId(0),
        item: ItemId(0),
    }
}

fn encoded(tracker: &StateTracker) -> FeatureFrame {
    encode(tracker, &LocalPolicyState::new(0))
}
