#![allow(
    clippy::float_arithmetic,
    reason = "Bounded observation-feature assertions."
)]

use super::feature::{encode, match_info, reverse_entity_ids_and_generations, world_view};
use crate::{ActionSpace, FeatureFrame, LocalPolicyState, Map2Reward, StateTracker, unit_feature};
use bota_proto::{EffectId, EffectView, EntityId, EventKind, MapId, SlotId, Team, WorldView};

#[test]
fn rebase_guarded_and_inspired_are_not_shadowraze_even_with_stack_payloads() {
    for id in [13, 14] {
        let mut view = world_view(Team::Radiant, 1);
        view.units[0].effects = vec![effect(id, Some(3), Some(15))];
        let frame = frame(view);

        assert_eq!(&frame.own_units()[0][73..76], &[0.0; 3]);
    }
}

#[test]
fn rebase_effect15_encodes_only_valid_active_raze_pairs() {
    for (ticks, expected) in [(0, 0.0), (1, 1.0 / 240.0), (240, 1.0), (241, 0.0)] {
        let mut view = world_view(Team::Radiant, 1);
        view.units[0].effects = vec![effect(15, Some(255), Some(ticks))];
        let frame = frame(view);

        assert_eq!(
            frame.own_units()[0][unit_feature::RAZE_TICKS_LEFT],
            expected
        );
        assert_eq!(
            frame.own_units()[0][unit_feature::RAZE_EFFECT_PRESENT],
            f32::from(expected > 0.0)
        );
    }
}

#[test]
fn rebase_aura_timers_include_presence_and_ignore_expired_or_malformed_rows() {
    for (ticks, stacks, expected) in [
        (Some(15), None, 1.0),
        (Some(1), None, 1.0 / 15.0),
        (Some(0), None, 0.0),
        (Some(16), None, 0.0),
        (None, None, 0.0),
        (Some(15), Some(1), 0.0),
    ] {
        let mut view = world_view(Team::Radiant, 1);
        view.units[0].effects = vec![effect(13, stacks, ticks), effect(14, stacks, ticks)];
        let frame = frame(view);

        assert_eq!(
            frame.own_units()[0][unit_feature::GUARDED_TICKS_LEFT],
            expected
        );
        assert_eq!(
            frame.own_units()[0][unit_feature::INSPIRED_TICKS_LEFT],
            expected
        );
        assert_eq!(&frame.units()[95][76..], &[0.0; 8]);
    }
}

#[test]
fn rebase_hp_only_mana_only_and_mixed_reports_preserve_the_other_channel() {
    let mut tracker = tracker(world_view(Team::Radiant, 1));
    let view = tracker.current().expect("snapshot").clone();
    let source = view.players[0].unit.expect("source");
    let target = view.players[1].unit.expect("target");
    for (tick, hp, mana, hp_expected, mana_expected) in [
        (2, 20, 0, Some((2, 20)), None),
        (3, 0, 35, Some((2, 20)), Some((3, 35))),
        (4, 11, 12, Some((4, 11)), Some((4, 12))),
        (5, 0, 0, Some((4, 11)), Some((4, 12))),
        (6, 10, 0, Some((6, 10)), Some((4, 12))),
    ] {
        let mut next = view.clone();
        next.tick = tick;
        tracker.observe_snapshot(&next).expect("snapshot");
        tracker
            .observe_events(tick, &[heal(Some(source), target, hp, mana)])
            .expect("report");
        let from = tracker.entity(source).expect("source");
        let into = tracker.entity(target).expect("target");

        assert_eq!(
            from.last_heal_dealt.map(|event| (event.tick, event.amount)),
            hp_expected
        );
        assert_eq!(
            into.last_heal_received
                .map(|event| (event.tick, event.amount)),
            hp_expected
        );
        assert_eq!(
            from.last_mana_restoration_dealt
                .map(|event| (event.tick, event.amount)),
            mana_expected
        );
        assert_eq!(
            into.last_mana_restoration_received
                .map(|event| (event.tick, event.amount)),
            mana_expected
        );
        assert_eq!(
            into.last_heal_received
                .expect("health report retained")
                .counterpart,
            Some(source)
        );
    }
}

#[test]
fn rebase_healing_report_limits_reject_batches_before_valid_prefix_or_reward_mutation() {
    let mut tracker = tracker(world_view(Team::Radiant, 1));
    let mut next = tracker.current().expect("snapshot").clone();
    next.tick = 2;
    let target = next.players[0].unit.expect("hero");
    tracker.observe_snapshot(&next).expect("snapshot");
    let prior = tracker.provenance();
    for (hp, mana, channel) in [
        (-1, 0, "health"),
        (1_000_001, 0, "health"),
        (0, -1, "mana"),
        (0, 1_000_001, "mana"),
    ] {
        let events = [heal(None, target, 5, 6), heal(None, target, hp, mana)];
        let error = tracker
            .observe_events(2, &events)
            .expect_err("invalid restoration");

        assert_eq!(
            error.to_string(),
            format!(
                "Healed {channel} report {value} is outside 0..=1000000",
                value = if channel == "health" { hp } else { mana }
            )
        );
        assert!(prior.matches(&tracker));
    }
    tracker
        .observe_events(2, &[heal(None, target, 1_000_000, 1_000_000)])
        .expect("inclusive limit");
    assert_eq!(
        tracker
            .entity(target)
            .expect("hero")
            .last_mana_restoration_received
            .expect("mana")
            .amount,
        1_000_000
    );
}

#[test]
fn rebase_promised_restoration_does_not_mutate_pools_or_pay_reward_twice() {
    let info = map2_info(Team::Radiant);
    let view = world_view(Team::Radiant, 1);
    let mut reference = Map2Reward::new(SlotId(0), &info).expect("reward");
    let mut tracker = tracker(view.clone());
    reference.observe_snapshot(&view).expect("snapshot");
    reference.observe_events(1, &[]).expect("events");
    for tick in 2..=3 {
        let mut next = view.clone();
        next.tick = tick;
        next.units[0].mana -= 75;
        let report = heal(None, next.players[0].unit.expect("hero"), 300, 100);
        tracker.observe_snapshot(&next).expect("snapshot");
        tracker
            .observe_events(tick, &[report])
            .expect("manual promise");
        reference
            .observe_snapshot(&next)
            .expect("reference snapshot");
        reference
            .observe_events(tick, &[])
            .expect("no healing credit");

        assert_eq!(tracker.current(), Some(&next));
        assert_eq!(tracker.map2_reward_state(), Some(reference.state()));
        assert_eq!(
            tracker.take_map2_reward_interval().expect("drain"),
            reference.take_interval().expect("reference drain")
        );
    }
}

#[test]
fn rebase_received_report_features_are_prior_tick_only_bounded_and_not_actual_pool_deltas() {
    let mut tracker = tracker(world_view(Team::Radiant, 1));
    let own = tracker.own_hero().expect("hero").id;
    let mut next = tracker.current().expect("snapshot").clone();
    next.tick = 2;
    tracker.observe_snapshot(&next).expect("snapshot");
    tracker
        .observe_events(2, &[heal(None, own, 100, 75)])
        .expect("promise");
    assert_eq!(
        &encode(&tracker, &LocalPolicyState::new(0)).own_units()[0][78..],
        &[0.0; 6]
    );
    next.tick = 3;
    tracker.observe_snapshot(&next).expect("snapshot");
    tracker
        .observe_events(3, &[heal(None, own, 0, 999)])
        .expect("later mana report");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let row = &frame.own_units()[0];

    assert_eq!(row[unit_feature::HEALTH_RESTORE_REPORT_PRESENT], 1.0);
    assert_eq!(
        row[unit_feature::HEALTH_RESTORE_REPORT_AMOUNT],
        100.0 / 100_000.0
    );
    assert_eq!(row[unit_feature::HEALTH_RESTORE_REPORT_AGE], 1.0 / 480.0);
    assert_eq!(row[unit_feature::MANA_RESTORE_REPORT_PRESENT], 1.0);
    assert_eq!(
        row[unit_feature::MANA_RESTORE_REPORT_AMOUNT],
        75.0 / 20_000.0
    );
    assert_eq!(row[unit_feature::MANA_RESTORE_REPORT_AGE], 1.0 / 480.0);
    assert_eq!(row[unit_feature::HP_DELTA], 0.0);
    assert_eq!(row[unit_feature::MANA_DELTA], 0.0);
}

#[test]
fn rebase_aura_and_raze_memory_decays_and_expires_without_fog_refresh() {
    let mut view = world_view(Team::Radiant, 1);
    let enemy = view.players[1].unit.expect("enemy");
    view.units
        .iter_mut()
        .find(|unit| unit.id == enemy)
        .expect("enemy")
        .effects = vec![
        effect(13, None, Some(2)),
        effect(14, None, Some(2)),
        effect(15, Some(3), Some(2)),
    ];
    let mut tracker = tracker(view.clone());
    view.units.retain(|unit| unit.id != enemy);
    for tick in 2..=3 {
        view.tick = tick;
        tracker.observe_snapshot(&view).expect("fog");
        tracker.observe_events(tick, &[]).expect("events");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        let token = &frame.remembered_units()[0];
        assert!(
            ActionSpace::from_tracker(&tracker)
                .expect("space")
                .entity_index(enemy)
                .is_none()
        );
        if tick == 2 {
            assert_eq!(token[unit_feature::GUARDED_TICKS_LEFT], 1.0 / 15.0);
            assert_eq!(token[unit_feature::INSPIRED_TICKS_LEFT], 1.0 / 15.0);
            assert_eq!(token[unit_feature::RAZE_TICKS_LEFT], 1.0 / 240.0);
        } else {
            assert_eq!(&token[73..78], &[0.0; 5]);
        }
    }
}

#[test]
fn rebase_aura_report_frames_are_clone_and_full_handle_invariant_on_both_sides() {
    let mut frames = Vec::new();
    for side in [Team::Radiant, Team::Dire] {
        let mut view = world_view(side, 1);
        view.units[0].effects = vec![
            effect(13, None, Some(15)),
            effect(14, None, Some(8)),
            effect(15, Some(2), Some(240)),
        ];
        let mut renamed = view.clone();
        reverse_entity_ids_and_generations(&mut renamed, 10_000, 39);
        let original = reported_frame(view);
        assert_eq!(original, reported_frame(renamed));
        let mut canonical = original;
        canonical.global[4..6].fill(0.0);
        frames.push(canonical);
    }
    assert_eq!(frames[0], frames[1]);
}

#[test]
fn rebase_received_report_age_is_inclusive_at_480_then_absent() {
    let view = world_view(Team::Radiant, 1);
    let own = view.players[0].unit.expect("hero");
    let mut tracker = tracker(view.clone());
    for tick in 2..=483 {
        let mut next = view.clone();
        next.tick = tick;
        tracker
            .observe_snapshot(&next)
            .expect("contiguous snapshot");
        let events = if tick == 2 {
            vec![heal(None, own, 1, 2)]
        } else {
            vec![]
        };
        tracker.observe_events(tick, &events).expect("events");
        if tick >= 482 {
            let frame = encode(&tracker, &LocalPolicyState::new(0));
            let row = &frame.own_units()[0];
            assert_eq!(
                row[unit_feature::HEALTH_RESTORE_REPORT_PRESENT],
                f32::from(tick == 482)
            );
            assert_eq!(
                row[unit_feature::MANA_RESTORE_REPORT_PRESENT],
                f32::from(tick == 482)
            );
            assert_eq!(
                row[unit_feature::MANA_RESTORE_REPORT_AGE],
                f32::from(tick == 482)
            );
        }
    }
}

#[test]
fn rebase_report_features_are_bounded_by_existing_prior_event_journal() {
    let mut view = world_view(Team::Radiant, 1);
    let own = view.players[0].unit.expect("hero");
    let mut tracker = tracker(view.clone());
    view.tick = 2;
    tracker.observe_snapshot(&view).expect("snapshot");
    let mut events = vec![heal(None, own, 12, 34)];
    events.extend((0..crate::MAX_RECENT_EVENTS).map(|_| heal(None, own, 0, 0)));
    tracker.observe_events(2, &events).expect("bounded events");
    view.tick = 3;
    tracker.observe_snapshot(&view).expect("next snapshot");
    tracker.observe_events(3, &[]).expect("events");

    let frame = encode(&tracker, &LocalPolicyState::new(0));

    assert_eq!(&frame.own_units()[0][78..84], &[0.0; 6]);
    assert_eq!(
        tracker
            .entity(own)
            .expect("hero")
            .last_heal_received
            .expect("bookkeeping retained")
            .amount,
        12
    );
    assert_eq!(
        tracker
            .entity(own)
            .expect("hero")
            .last_mana_restoration_received
            .expect("bookkeeping retained")
            .amount,
        34
    );
}

#[test]
fn rebase_aura_and_received_report_semantics_precede_opaque_id_ties() {
    for aura in [false, true] {
        let mut view = world_view(Team::Radiant, 1);
        let mut first = view.units[0].clone();
        first.id = EntityId {
            idx: 100,
            generation: 1,
        };
        first.kind = bota_proto::UnitKind::CreepMelee;
        first.owner = None;
        first.hero = None;
        first.abilities.clear();
        first.items.clear();
        let mut second = first.clone();
        second.id.idx = 101;
        if aura {
            second.effects = vec![effect(13, None, Some(15)), effect(14, None, Some(7))];
        }
        let recipient = second.id;
        view.units.extend([first, second]);
        let mut renamed = view.clone();
        let mut original = tracker(view.clone());
        let second_index = renamed
            .units
            .iter()
            .position(|unit| unit.id == recipient)
            .expect("recipient");
        reverse_entity_ids_and_generations(&mut renamed, 90_000, 45);
        let renamed_recipient = EntityId {
            idx: 90_000 + view.units.len() as u32 - second_index as u32,
            generation: 45 + second_index as u32 % 3,
        };
        let mut mapped = tracker(renamed.clone());
        for tick in 2..=3 {
            view.tick = tick;
            renamed.tick = tick;
            original.observe_snapshot(&view).expect("snapshot");
            mapped.observe_snapshot(&renamed).expect("snapshot");
            original
                .observe_events(tick, &[heal(None, recipient, 0, 33)])
                .expect("events");
            mapped
                .observe_events(tick, &[heal(None, renamed_recipient, 0, 33)])
                .expect("events");
        }
        assert_eq!(
            encode(&original, &LocalPolicyState::new(0)),
            encode(&mapped, &LocalPolicyState::new(0))
        );
    }
}

#[test]
fn rebase_healing_bookkeeping_invalidates_pre_event_provenance_without_overwriting_channels() {
    let mut view = world_view(Team::Radiant, 1);
    let own = view.players[0].unit.expect("hero");
    let mut tracker = tracker(view.clone());
    view.tick = 2;
    tracker.observe_snapshot(&view).expect("snapshot");
    let prior = tracker.provenance();
    let space = ActionSpace::from_tracker(&tracker).expect("pre-event space");

    tracker
        .observe_events(2, &[heal(None, own, 0, 5)])
        .expect("mana report");

    assert!(!prior.matches(&tracker));
    assert!(!space.matches_tracker(&tracker));
    assert!(
        tracker
            .entity(own)
            .expect("hero")
            .last_heal_received
            .is_none()
    );
    assert!(
        tracker
            .entity(own)
            .expect("hero")
            .last_mana_restoration_received
            .is_some()
    );
}

fn reported_frame(mut view: WorldView) -> FeatureFrame {
    let mut original = tracker(view.clone());
    let mut branch = original.clone();
    for tick in 2..=3 {
        view.tick = tick;
        for observer in [&mut original, &mut branch] {
            observer.observe_snapshot(&view).expect("snapshot");
            observer
                .observe_events(
                    tick,
                    &[heal(
                        view.players[1].unit,
                        view.players[0].unit.expect("hero"),
                        21,
                        35,
                    )],
                )
                .expect("reports");
        }
    }
    let output = encode(&original, &LocalPolicyState::new(0));
    assert_eq!(output, encode(&branch, &LocalPolicyState::new(0)));
    assert_eq!(original.entities(), branch.entities());
    output
}

fn frame(view: WorldView) -> FeatureFrame {
    encode(&tracker(view), &LocalPolicyState::new(0))
}

fn tracker(view: WorldView) -> StateTracker {
    let mut tracker =
        StateTracker::new(SlotId(0), &map2_info(view.viewer.expect("seat"))).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker.observe_events(view.tick, &[]).expect("events");
    tracker
}

fn map2_info(side: Team) -> bota_proto::MatchInfo {
    let mut info = match_info(side);
    info.map = MapId(2);
    info
}

fn effect(id: u16, stacks: Option<u32>, ticks_left: Option<u32>) -> EffectView {
    EffectView {
        id: EffectId(id),
        stacks,
        ticks_left,
    }
}

fn heal(source: Option<EntityId>, target: EntityId, amount: i32, mana: i32) -> EventKind {
    EventKind::Healed {
        source,
        target,
        amount,
        mana,
    }
}
