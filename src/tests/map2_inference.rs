#![allow(clippy::float_arithmetic, reason = "Bounded feature assertions.")]

use super::feature::{encode, match_info, world_view};
use crate::{
    ActionSpace, FeatureEncoder, FeatureFrame, ItemReadiness, LocalPolicyState, Map2RewardEnd,
    StateTracker, unit_feature,
};
use bota_proto::{
    EffectId, EffectView, EntityId, EventKind, MapId, MatchInfo, SlotId, Team, WorldView,
};

#[test]
fn map_metadata_selects_strict_reward_or_independent_legacy_streams() {
    for map in [MapId(0), MapId(1), MapId(2)] {
        let mut info = match_info(Team::Radiant);
        info.map = map;
        let mut tracker = StateTracker::new(SlotId(0), &info).expect("supported map");
        assert_eq!(tracker.metadata().map, map);
        assert_eq!(tracker.shop(), info.shop);
        assert_eq!(tracker.map2_reward_state().is_some(), map == MapId(2));
        if map != MapId(2) {
            tracker.observe_events(99, &[]).expect("events first");
            tracker
                .observe_snapshot(&world_view(Team::Radiant, 1))
                .expect("independent snapshot");
            assert_eq!(&frame(&tracker).global()[72..], &[0.0; 20]);
            tracker
                .observe_snapshot(&world_view(Team::Radiant, 30))
                .expect("sparse snapshot");
            for result in [
                tracker.take_map2_reward_interval(),
                tracker.finish_map2_reward(Map2RewardEnd::Draw),
            ] {
                assert_eq!(
                    result.expect_err("Map2 only").to_string(),
                    "Map2 reward is unavailable outside MapId(2)"
                );
            }
        }
    }
    let mut info = info(Team::Radiant);
    info.picks.pop();
    let error = StateTracker::new(SlotId(0), &info)
        .err()
        .expect("duel required");
    assert_eq!(error.to_string(), "Map2 reward: expected exactly two picks");
    assert_eq!(info.map, MapId(2));
}

#[test]
fn incomplete_pairs_reject_atomically_and_feature_observation_can_retry() {
    let mut tracker = tracker(Team::Radiant, world_view(Team::Radiant, 7));
    let prior = tracker.provenance();
    let gap = world_view(Team::Radiant, 9);
    assert_eq!(
        tracker.observe_snapshot(&gap).expect_err("gap").to_string(),
        "Map2 reward: Snapshot ticks must be contiguous"
    );
    assert!(prior.matches(&tracker));
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 8))
        .expect("next snapshot");
    let pending = tracker.provenance();
    let mut encoder = FeatureEncoder::new(&tracker);
    assert_eq!(
        encoder
            .observe(&tracker)
            .expect_err("pending events")
            .to_string(),
        "feature Map2 reward requires complete Snapshot/Events for tick 8"
    );
    for (result, expected) in [
        (
            tracker.observe_snapshot(&gap),
            "Map2 reward: Snapshot arrived before pending Events",
        ),
        (
            tracker.observe_events(9, &[]),
            "Map2 reward: Events tick does not match pending Snapshot",
        ),
        (
            tracker.take_map2_reward_interval().map(|_| ()),
            "Map2 reward: interval has pending Events",
        ),
        (
            tracker.finish_map2_reward(Map2RewardEnd::Win).map(|_| ()),
            "Map2 reward: interval has pending Events",
        ),
    ] {
        assert_eq!(result.expect_err("incomplete pair").to_string(), expected);
        assert!(pending.matches(&tracker));
    }
    tracker
        .observe_events(8, &[])
        .expect("correct events retry");
    encoder.observe(&tracker).expect("feature retry");
    assert_eq!(tracker.take_map2_reward_interval().expect("drain").ticks, 1);
}

#[test]
fn reward_drain_preserves_frames_but_each_terminal_invalidates_pairings_exactly_once() {
    for (end, terminal) in [
        (Map2RewardEnd::Win, 0.2),
        (Map2RewardEnd::Loss, -0.2),
        (Map2RewardEnd::Draw, 0.0),
        (Map2RewardEnd::TimeCap, -0.2),
    ] {
        let (mut view, mut tracker) = initial(Team::Radiant);
        let visible = frame(&tracker);
        let space = ActionSpace::from_tracker(&tracker).expect("space");
        let mut encoder = FeatureEncoder::new(&tracker);
        encoder.observe(&tracker).expect("observation before drain");
        let prior = tracker.provenance();
        tracker.take_map2_reward_interval().expect("drain");
        assert!(prior.matches(&tracker));
        let mut output = FeatureFrame::new();
        encode_observed(&mut encoder, &tracker, &space, &mut output)
            .expect("drain preserves observation");
        assert_eq!(output, visible);
        let result = tracker.finish_map2_reward(end).expect("finish");
        assert_eq!(result.end, Some(end));
        assert_eq!(result.terminal, terminal);
        assert!(!prior.matches(&tracker));
        let new_space = ActionSpace::from_tracker(&tracker).expect("new space");
        assert!(!visible.matches_action_space(&new_space));
        for (space, expected) in [
            (
                &space,
                "feature action space belongs to a different snapshot",
            ),
            (
                &new_space,
                "feature observation belongs to a different snapshot at tick 1",
            ),
        ] {
            let error = encode_observed(&mut encoder, &tracker, space, &mut output)
                .expect_err("stale pairing");
            assert_eq!(error.to_string(), expected);
            assert_eq!(output, visible);
        }
        view.tick += 1;
        for result in [
            tracker.finish_map2_reward(end).map(|_| ()),
            tracker.observe_snapshot(&view),
        ] {
            assert_eq!(
                result.expect_err("ended").to_string(),
                "Map2 reward: episode already ended"
            );
        }
    }
}

fn encode_observed(
    encoder: &mut FeatureEncoder,
    tracker: &StateTracker,
    space: &ActionSpace,
    output: &mut FeatureFrame,
) -> Result<(), crate::FeatureError> {
    encoder.encode(
        tracker,
        space,
        &ItemReadiness::new(),
        &LocalPolicyState::new(0),
        output,
    )
}

#[test]
fn replacement_and_observed_death_clear_raze_before_its_timer_expires() {
    for died in [false, true] {
        let mut view = world_view(Team::Radiant, 1);
        let own = view.players[0].unit.expect("own");
        let hero_index = view
            .units
            .iter()
            .position(|unit| unit.id == own)
            .expect("hero");
        view.units[hero_index].effects = vec![effect(15, Some(3), Some(240))];
        let mut tracker = tracker(Team::Radiant, view.clone());
        let events = if died {
            view.units.retain(|unit| unit.id != own);
            view.players[0].unit = None;
            view.players[0].deaths = 1;
            view.players[0].respawn_left = 100;
            vec![EventKind::Died {
                unit: own,
                killer: view.players[1].unit,
                denied: false,
                gold: 100,
            }]
        } else {
            let replacement = EntityId {
                generation: own.generation + 1,
                ..own
            };
            view.players[0].unit = Some(replacement);
            view.units[hero_index].id = replacement;
            view.units[hero_index].effects.clear();
            vec![]
        };
        advance(&mut tracker, &mut view, &events);
        let frame = frame(&tracker);
        assert_eq!(&frame.own_units()[0][73..76], &[0.0; 3]);
        assert_eq!(
            frame.own_units()[0][unit_feature::REMEMBERED],
            f32::from(died)
        );
        if !died {
            assert!(tracker.entity(own).is_none());
        }
    }
}

pub(super) fn initial(team: Team) -> (WorldView, StateTracker) {
    let view = world_view(team, 1);
    let tracker = tracker(team, view.clone());
    (view, tracker)
}

pub(super) fn stage_next(tracker: &mut StateTracker, view: &mut WorldView) {
    view.tick = view.tick.checked_add(1).expect("bounded fixture tick");
    tracker.observe_snapshot(view).expect("contiguous snapshot");
}

pub(super) fn advance(tracker: &mut StateTracker, view: &mut WorldView, events: &[EventKind]) {
    stage_next(tracker, view);
    tracker
        .observe_events(view.tick, events)
        .expect("complete events");
}

pub(super) fn frame(tracker: &StateTracker) -> FeatureFrame {
    encode(tracker, &LocalPolicyState::new(0))
}

pub(super) fn info(team: Team) -> MatchInfo {
    let mut info = match_info(team);
    info.map = MapId(2);
    info
}

pub(super) fn tracker(team: Team, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &info(team)).expect("Map2 tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker.observe_events(view.tick, &[]).expect("events");
    tracker
}

pub(super) fn effect(id: u16, stacks: Option<u32>, ticks_left: Option<u32>) -> EffectView {
    EffectView {
        id: EffectId(id),
        stacks,
        ticks_left,
    }
}
