#![allow(clippy::float_arithmetic, reason = "Bounded feature assertions.")]

use bota_proto::{
    Aim, DamageKind, EffectId, EffectView, EntityId, EventKind, ItemId, ItemView, MapId, MatchInfo,
    ShopEntry, SlotId, Team, UnitKind, Vec2, WorldView,
};

use crate::{
    ActionSpace, FeatureEncoder, FeatureFrame, ItemReadiness, LocalPolicyState, Map2Reward,
    Map2RewardEnd, StateTracker, global_feature, unit_feature,
};

use super::feature::{encode, match_info, reverse_entity_ids_and_generations, world_view};

#[test]
fn map2_tracker_preserves_metadata_and_owns_reward_only_for_map_two() {
    for map in [MapId(0), MapId(1), MapId(2)] {
        let mut info = match_info(Team::Radiant);
        info.map = map;
        let tracker = StateTracker::new(SlotId(0), &info).expect("supported map");

        assert_eq!(tracker.metadata().map, map);
        assert_eq!(tracker.map2_reward_state().is_some(), map == MapId(2));
        assert_eq!(tracker.shop(), info.shop);
    }
}

#[test]
fn map2_tracker_rejects_non_duel_metadata_without_changing_it() {
    let mut info = info(Team::Radiant);
    info.picks.pop();

    let error = StateTracker::new(SlotId(0), &info)
        .err()
        .expect("duel required");

    assert_eq!(error.to_string(), "Map2 reward: expected exactly two picks");
    assert_eq!(info.map, MapId(2));
}

#[test]
fn legacy_trackers_keep_independent_diagnostic_streams_and_reject_map2_drains() {
    for map in [MapId(0), MapId(1)] {
        let mut info = match_info(Team::Radiant);
        info.map = map;
        let mut tracker = StateTracker::new(SlotId(0), &info).expect("legacy tracker");
        tracker.observe_events(99, &[]).expect("events first");
        tracker
            .observe_snapshot(&world_view(Team::Radiant, 1))
            .expect("independent snapshot");
        tracker
            .observe_snapshot(&world_view(Team::Radiant, 30))
            .expect("sparse snapshot");

        assert_eq!(
            tracker
                .take_map2_reward_interval()
                .expect_err("Map2 only")
                .to_string(),
            "Map2 reward is unavailable outside MapId(2)"
        );
        assert_eq!(
            tracker
                .finish_map2_reward(Map2RewardEnd::Draw)
                .expect_err("Map2 only")
                .to_string(),
            "Map2 reward is unavailable outside MapId(2)"
        );
    }
}

#[test]
fn map2_tracker_consumes_full_events_before_the_64_event_journal_and_clones_budgets() {
    let info = info(Team::Radiant);
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    let mut reference = Map2Reward::new(SlotId(0), &info).expect("reward");
    let view = world_view(Team::Radiant, 1);
    let events = vec![damage(&view, 1); 100];

    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
        .observe_events(view.tick, &events)
        .expect("all events");
    reference.observe_snapshot(&view).expect("snapshot");
    reference
        .observe_events(view.tick, &events)
        .expect("all events");
    let mut branch = tracker.clone();

    assert_eq!(tracker.recent_events().len(), 64);
    assert_eq!(tracker.map2_reward_state(), Some(reference.state()));
    let expected = reference.take_interval().expect("baseline events");
    assert_eq!(expected.observations.hero_damage_dealt, 100);
    assert_eq!(
        tracker.take_map2_reward_interval().expect("drain"),
        expected
    );
    assert_eq!(
        branch.take_map2_reward_interval().expect("clone drain"),
        expected
    );
    assert_eq!(tracker.map2_reward_state(), branch.map2_reward_state());
    assert_eq!(
        tracker
            .take_map2_reward_interval()
            .expect("repeat drain")
            .total,
        0.0
    );
}

#[test]
fn map2_tracker_rejects_gaps_and_incomplete_pairs_atomically_and_allows_retry() {
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
    assert_eq!(
        tracker
            .observe_snapshot(&gap)
            .expect_err("missing events")
            .to_string(),
        "Map2 reward: Snapshot arrived before pending Events"
    );
    assert_eq!(
        tracker
            .observe_events(9, &[])
            .expect_err("wrong tick")
            .to_string(),
        "Map2 reward: Events tick does not match pending Snapshot"
    );
    assert_eq!(
        tracker
            .take_map2_reward_interval()
            .expect_err("incomplete")
            .to_string(),
        "Map2 reward: interval has pending Events"
    );
    assert_eq!(
        tracker
            .finish_map2_reward(Map2RewardEnd::Win)
            .expect_err("incomplete")
            .to_string(),
        "Map2 reward: interval has pending Events"
    );
    assert!(pending.matches(&tracker));
    tracker
        .observe_events(8, &[])
        .expect("correct events retry");
    assert_eq!(tracker.take_map2_reward_interval().expect("drain").ticks, 1);
}

#[test]
fn map2_tracker_rejected_event_batch_does_not_count_a_valid_prefix() {
    let view = world_view(Team::Radiant, 1);
    let mut tracker = StateTracker::new(SlotId(0), &info(Team::Radiant)).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    let pending = tracker.provenance();
    let events = vec![damage(&view, 5); crate::MAP2_REWARD_MAX_EVENTS + 1];

    assert_eq!(
        tracker
            .observe_events(1, &events)
            .expect_err("bounded batch")
            .to_string(),
        "Map2 reward: event batch has 4097 entries; maximum is 4096"
    );
    assert!(pending.matches(&tracker));
    tracker.observe_events(1, &events[..1]).expect("retry");
    assert_eq!(
        tracker
            .take_map2_reward_interval()
            .expect("drain")
            .observations
            .hero_damage_dealt,
        5
    );
}

#[test]
fn map2_tracker_finish_is_seat_relative_exact_once_and_invalidates_frame_provenance() {
    for (end, terminal) in [
        (Map2RewardEnd::Win, 1.0),
        (Map2RewardEnd::Loss, -1.0),
        (Map2RewardEnd::Draw, 0.0),
        (Map2RewardEnd::TimeCap, 0.0),
    ] {
        let mut tracker = tracker(Team::Radiant, world_view(Team::Radiant, 1));
        let before = tracker.provenance();
        let space = ActionSpace::from_tracker(&tracker).expect("space");
        let mut branch = tracker.clone();

        let result = tracker.finish_map2_reward(end).expect("finish");

        assert_eq!(result.end, Some(end));
        assert_eq!(result.terminal, terminal);
        assert_eq!(
            branch.finish_map2_reward(end).expect("cloned finish"),
            result
        );
        assert!(!before.matches(&tracker));
        assert!(!space.matches_tracker(&tracker));
        assert_eq!(
            tracker
                .finish_map2_reward(end)
                .expect_err("once")
                .to_string(),
            "Map2 reward: episode already ended"
        );
        assert_eq!(
            tracker
                .observe_snapshot(&world_view(Team::Radiant, 2))
                .expect_err("ended")
                .to_string(),
            "Map2 reward: episode already ended"
        );
    }
}

#[test]
fn map2_features_require_a_complete_pair_and_rejected_observation_can_be_retried() {
    let mut tracker = StateTracker::new(SlotId(0), &info(Team::Radiant)).expect("tracker");
    tracker
        .observe_snapshot(&world_view(Team::Radiant, 1))
        .expect("snapshot");
    let mut encoder = FeatureEncoder::new(&tracker);

    assert_eq!(
        encoder
            .observe(&tracker)
            .expect_err("pending events")
            .to_string(),
        "feature Map2 reward requires complete Snapshot/Events for tick 1"
    );
    tracker.observe_events(1, &[]).expect("matching events");
    encoder.observe(&tracker).expect("retry");
}

#[test]
fn map2_features_export_accounting_state_and_drains_preserve_observation_provenance() {
    let view = world_view(Team::Radiant, 1);
    let mut tracker = tracker(Team::Radiant, view.clone());
    let mut next = view.clone();
    next.tick = 2;
    next.players[0].xp += 50;
    next.players[1].xp += 70;
    next.units
        .iter_mut()
        .find(|unit| unit.id == view.players[0].unit.expect("hero"))
        .expect("hero")
        .mana -= 75;
    tracker.observe_snapshot(&next).expect("snapshot");
    tracker
        .observe_events(2, &[damage(&view, 67)])
        .expect("events");
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("observation");
    let state = tracker.map2_reward_state().expect("reward state");
    let before = tracker.provenance();
    tracker
        .take_map2_reward_interval()
        .expect("drain after feature observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            &tracker,
            &space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("encode");

    assert!(before.matches(&tracker));
    assert_eq!(frame.global()[global_feature::MAP_TWO], 1.0);
    assert_eq!(
        &frame.global()[global_feature::MAP2_REWARD_REMAINING_START
            ..global_feature::MAP2_REWARD_REMAINING_START + 9],
        &state.remaining
    );
    assert_eq!(
        frame.global()[global_feature::MAP2_TOWER_POTENTIAL],
        state.tower_potential
    );
    assert_eq!(
        frame.global()[global_feature::MAP2_LANE_POTENTIAL],
        state.lane_potential
    );
    assert_eq!(
        frame.global()[global_feature::MAP2_LANE_OBSERVED],
        f32::from(state.lane_observed)
    );
    assert!(frame.is_finite());
}

#[test]
fn legacy_features_leave_every_map2_global_zero() {
    let tracker = super::feature::tracker_with_view(Team::Radiant, world_view(Team::Radiant, 1));
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(&frame.global()[72..], &[0.0; 13]);
}

#[test]
fn map2_anonymous_raze_pair_is_visible_normalized_and_effect_order_independent() {
    let mut view = world_view(Team::Radiant, 1);
    let enemy = view.players[1].unit.expect("enemy");
    let unit = view
        .units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy");
    unit.effects = vec![
        effect(15, Some(2), Some(240)),
        effect(15, Some(4), Some(120)),
        effect(15, Some(4), Some(119)),
        effect(11, Some(255), None),
    ];
    let tracker = tracker(Team::Radiant, view.clone());
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let row = ActionSpace::from_tracker(&tracker)
        .expect("space")
        .entity_index(enemy)
        .expect("enemy")
        .0;

    assert_eq!(frame.units()[row][unit_feature::RAZE_EFFECT_PRESENT], 1.0);
    assert_eq!(frame.units()[row][unit_feature::RAZE_STACKS], 4.0 / 255.0);
    assert_eq!(frame.units()[row][unit_feature::RAZE_TICKS_LEFT], 0.5);
    view.units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy")
        .effects
        .reverse();
    assert_eq!(
        frame,
        encode(
            &self::tracker(Team::Radiant, view),
            &LocalPolicyState::new(0)
        )
    );
}

#[test]
fn map2_raze_expired_incomplete_other_effect_and_absent_tokens_are_zero() {
    for effects in [
        vec![],
        vec![effect(12, Some(2), Some(200))],
        vec![effect(15, Some(1), Some(0))],
        vec![effect(15, Some(0), Some(240))],
        vec![effect(15, None, Some(240))],
        vec![effect(15, Some(1), None)],
        vec![effect(15, Some(1), Some(241))],
        vec![effect(15, Some(256), Some(240))],
        vec![effect(15, Some(u32::MAX), Some(u32::MAX))],
    ] {
        let mut view = world_view(Team::Radiant, 1);
        view.units[0].effects = effects;
        let frame = encode(&tracker(Team::Radiant, view), &LocalPolicyState::new(0));

        assert_eq!(&frame.own_units()[0][73..76], &[0.0; 3]);
        assert_eq!(&frame.units()[95], &[0.0; crate::UNIT_FEATURES]);
    }
}

#[test]
fn map2_remembered_raze_timer_decays_without_refresh_or_target_pointer_and_expires() {
    let mut view = world_view(Team::Radiant, 1);
    let enemy = view.players[1].unit.expect("enemy");
    view.units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy")
        .effects = vec![effect(15, Some(3), Some(2))];
    let mut tracker = tracker(Team::Radiant, view.clone());
    view.units.retain(|unit| unit.id != enemy);
    for tick in 2..=3 {
        view.tick = tick;
        tracker.observe_snapshot(&view).expect("fog snapshot");
        tracker.observe_events(tick, &[]).expect("events");
        let space = ActionSpace::from_tracker(&tracker).expect("space");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        let token = &frame.remembered_units()[0];

        assert!(space.entity_index(enemy).is_none());
        assert_eq!(token[unit_feature::REMEMBERED], 1.0);
        if tick == 2 {
            assert_eq!(token[unit_feature::RAZE_STACKS], 3.0 / 255.0);
            assert_eq!(token[unit_feature::RAZE_TICKS_LEFT], 1.0 / 240.0);
        } else {
            assert_eq!(&token[73..76], &[0.0; 3]);
        }
    }
}

#[test]
fn map2_effects_and_reward_features_are_side_and_handle_invariant() {
    let mut frames = Vec::new();
    for team in [Team::Radiant, Team::Dire] {
        let mut view = world_view(team, 1);
        view.units[0].effects = vec![effect(15, Some(255), Some(240))];
        let mut remapped = view.clone();
        reverse_entity_ids_and_generations(&mut remapped, 100_000, 57);
        let mut original = tracker(team, view.clone());
        let mut renamed = tracker(team, remapped.clone());
        view.tick = 2;
        remapped.tick = 2;
        original.observe_snapshot(&view).expect("snapshot");
        original
            .observe_events(2, &[damage(&view, 67)])
            .expect("events");
        renamed.observe_snapshot(&remapped).expect("snapshot");
        renamed
            .observe_events(2, &[damage(&remapped, 67)])
            .expect("events");
        let mut frame = encode(&original, &LocalPolicyState::new(0));
        assert_eq!(frame, encode(&renamed, &LocalPolicyState::new(0)));
        frame.global[global_feature::SIDE_RADIANT] = 0.0;
        frame.global[global_feature::SIDE_DIRE] = 0.0;
        frames.push(frame);
    }
    assert_eq!(frames[0], frames[1]);
}

#[test]
fn map2_generation_replacement_does_not_inherit_raze_effects() {
    let mut view = world_view(Team::Radiant, 1);
    let own = view.players[0].unit.expect("own");
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .expect("own")
        .effects = vec![effect(15, Some(3), Some(240))];
    let mut tracker = tracker(Team::Radiant, view.clone());
    let replacement = EntityId {
        generation: own.generation + 1,
        ..own
    };
    view.players[0].unit = Some(replacement);
    let hero = view
        .units
        .iter_mut()
        .find(|unit| unit.id == own)
        .expect("hero");
    hero.id = replacement;
    hero.effects.clear();
    view.tick = 2;
    tracker.observe_snapshot(&view).expect("replacement");
    tracker.observe_events(2, &[]).expect("events");
    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(&frame.own_units()[0][73..76], &[0.0; 3]);
    assert!(tracker.entity(own).is_none());
}

#[test]
fn map2_observed_death_clears_remembered_raze_before_its_old_timer_expires() {
    let mut view = world_view(Team::Radiant, 1);
    let own = view.players[0].unit.expect("own");
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .expect("own")
        .effects = vec![effect(15, Some(3), Some(240))];
    let mut tracker = tracker(Team::Radiant, view.clone());
    view.tick = 2;
    view.units.retain(|unit| unit.id != own);
    view.players[0].unit = None;
    view.players[0].deaths = 1;
    view.players[0].respawn_left = 100;
    let death = EventKind::Died {
        unit: own,
        killer: view.players[1].unit,
        denied: false,
        gold: 100,
    };
    tracker.observe_snapshot(&view).expect("death snapshot");
    tracker.observe_events(2, &[death]).expect("observed death");

    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(frame.own_units()[0][unit_feature::REMEMBERED], 1.0);
    assert_eq!(&frame.own_units()[0][73..76], &[0.0; 3]);
}

#[test]
fn map2_teacher_uses_identical_observed_geometry_without_strategy_changes() {
    for team in [Team::Radiant, Team::Dire] {
        for health in [50, 600, 1_000] {
            let mut view = world_view(team, 1);
            view.units[0].hp = health;
            view.players[0].gold = Some(0);
            let legacy = super::feature::tracker_with_view(team, view.clone());
            let mid = tracker(team, view);
            let mut legacy_teacher = crate::Teacher::new();
            let mut mid_teacher = crate::Teacher::new();
            let persistence = crate::OrderPersistence::default();

            let (legacy_action, legacy_space) = legacy_teacher
                .decide(&legacy, &persistence, &ItemReadiness::new())
                .expect("legacy action");
            let (mid_action, mid_space) = mid_teacher
                .decide(&mid, &persistence, &ItemReadiness::new())
                .expect("Map2 adapter");

            assert_eq!(
                legacy_space.point_candidates(),
                mid_space.point_candidates()
            );
            assert_eq!(
                legacy_space.decode(legacy_action),
                mid_space.decode(mid_action)
            );
            assert_eq!(legacy_teacher, mid_teacher);
            assert_eq!(mid.metadata().map, MapId(2));
        }
    }
}

#[test]
fn map2_nonzero_lane_and_tower_state_remain_visible_through_fog_and_terminal() {
    let mut view = pressure_view();
    let mut tracker = tracker(Team::Radiant, view.clone());
    let first = encode(&tracker, &LocalPolicyState::new(0));
    let state = tracker.map2_reward_state().expect("reward state");
    assert_ne!(state.tower_potential, 0.0);
    assert_ne!(state.lane_potential, 0.0);
    assert_eq!(
        first.global()[global_feature::MAP2_TOWER_POTENTIAL],
        state.tower_potential
    );
    assert_eq!(
        first.global()[global_feature::MAP2_LANE_POTENTIAL],
        state.lane_potential
    );
    assert_eq!(first.global()[global_feature::MAP2_LANE_OBSERVED], 1.0);

    view.tick = 2;
    view.units.retain(|unit| unit.kind != UnitKind::CreepMelee);
    tracker.observe_snapshot(&view).expect("fog snapshot");
    tracker.observe_events(2, &[]).expect("events");
    let fog = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(fog.global()[global_feature::MAP2_LANE_OBSERVED], 0.0);
    assert_eq!(
        fog.global()[global_feature::MAP2_LANE_POTENTIAL],
        state.lane_potential
    );

    let end = tracker
        .finish_map2_reward(Map2RewardEnd::Draw)
        .expect("finish");
    let final_frame = encode(&tracker, &LocalPolicyState::new(0));
    assert_eq!(
        final_frame.global()[global_feature::MAP2_LANE_POTENTIAL],
        0.0
    );
    assert_eq!(
        final_frame.global()[global_feature::MAP2_TOWER_POTENTIAL],
        state.tower_potential
    );
    assert!((end.lane_pressure + f64::from(state.lane_potential)).abs() < 1.0e-9);
}

#[test]
fn map2_same_snapshot_after_finish_cannot_reuse_prior_observation_or_frame_pairing() {
    let mut tracker = tracker(Team::Radiant, pressure_view());
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("before finish");
    let old_space = ActionSpace::from_tracker(&tracker).expect("old space");
    let old_frame = encode(&tracker, &LocalPolicyState::new(0));
    let mut output = old_frame.clone();
    tracker
        .finish_map2_reward(Map2RewardEnd::Draw)
        .expect("finish");
    let new_space = ActionSpace::from_tracker(&tracker).expect("new space");

    let error = encoder
        .encode(
            &tracker,
            &old_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("stale action space");
    assert_eq!(
        error.to_string(),
        "feature action space belongs to a different snapshot"
    );
    assert_eq!(output, old_frame);
    let error = encoder
        .encode(
            &tracker,
            &new_space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut output,
        )
        .expect_err("stale observation");
    assert_eq!(
        error.to_string(),
        "feature observation belongs to a different snapshot at tick 1"
    );
    assert_eq!(output, old_frame);
    assert!(!old_frame.matches_action_space(&new_space));
    assert!(!old_space.matches_tracker(&tracker.clone()));
}

#[test]
fn map2_clone_of_pending_snapshot_keeps_reward_staging_and_independent_future() {
    let view = world_view(Team::Radiant, 1);
    let mut tracker = StateTracker::new(SlotId(0), &info(Team::Radiant)).expect("tracker");
    tracker.observe_snapshot(&view).expect("staged snapshot");
    let mut branch = tracker.clone();

    tracker
        .observe_events(1, &[damage(&view, 67)])
        .expect("original events");
    branch.observe_events(1, &[]).expect("branch events");

    assert_eq!(tracker.current(), branch.current());
    assert_ne!(tracker.map2_reward_state(), branch.map2_reward_state());
    assert_eq!(
        tracker
            .take_map2_reward_interval()
            .expect("original drain")
            .observations
            .hero_damage_dealt,
        67
    );
    assert_eq!(
        branch
            .take_map2_reward_interval()
            .expect("branch drain")
            .observations
            .hero_damage_dealt,
        0
    );
}

#[test]
fn map2_mango_category_charges_and_legality_reach_existing_item_and_loot_tokens() {
    let mut info = info(Team::Radiant);
    info.shop.push(ShopEntry {
        id: ItemId(42),
        cost: 65,
        components: vec![],
    });
    let mut view = world_view(Team::Radiant, 1);
    let mango = ItemView {
        id: ItemId(42),
        charges: Some(3),
        cooldown_left: 0,
        mute_left: 0,
        mode: None,
        mana_cost: 0,
        range: 0,
        aim: Some(Aim::Own),
        for_sale: false,
    };
    view.units[0].items[0] = Some(mango);
    view.players[0].stash.as_mut().expect("stash")[0] = Some(mango);
    view.loot[0].item = mango.id;
    let mut tracker = StateTracker::new(SlotId(0), &info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker.observe_events(1, &[]).expect("events");

    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(frame.items()[0][crate::item_feature::ITEM_TOKEN], 43.0);
    assert_eq!(frame.items()[0][crate::item_feature::CHARGES], 3.0 / 255.0);
    assert_eq!(frame.items()[0][crate::item_feature::LEGAL], 1.0);
    assert_eq!(frame.items()[9][crate::item_feature::ITEM_TOKEN], 43.0);
    assert_eq!(frame.items()[9][crate::item_feature::LEGAL], 0.0);
    assert_eq!(frame.loot()[0][crate::loot_feature::ITEM_TOKEN], 43.0);
}

fn pressure_view() -> WorldView {
    let mut view = world_view(Team::Radiant, 1);
    view.units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::Tower && unit.team == Team::Dire)
        .expect("enemy tower")
        .hp = 100;
    for (idx, team, position) in [(100, Team::Radiant, 5_000), (101, Team::Dire, 6_000)] {
        let mut creep = view.units[0].clone();
        creep.id = EntityId { idx, generation: 1 };
        creep.kind = UnitKind::CreepMelee;
        creep.team = team;
        creep.owner = None;
        creep.hero = None;
        creep.abilities.clear();
        creep.items.clear();
        creep.pos = Vec2::from_ints(position, position);
        view.units.push(creep);
    }
    view.units.sort_by_key(|unit| unit.id);
    view
}

fn info(team: Team) -> MatchInfo {
    let mut info = match_info(team);
    info.map = MapId(2);
    info
}

fn tracker(team: Team, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &info(team)).expect("Map2 tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker.observe_events(view.tick, &[]).expect("events");
    tracker
}

fn damage(view: &WorldView, amount: i32) -> EventKind {
    EventKind::Damaged {
        source: view.players[0].unit,
        target: view.players[1].unit.expect("opposing hero"),
        amount,
        kind: DamageKind::Magical,
        crit: false,
    }
}

fn effect(id: u16, stacks: Option<u32>, ticks_left: Option<u32>) -> EffectView {
    EffectView {
        id: EffectId(id),
        stacks,
        ticks_left,
    }
}
