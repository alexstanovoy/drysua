#![allow(
    clippy::float_arithmetic,
    reason = "Bounded observation-feature assertions."
)]

use super::feature::world_view;
use super::map2_inference::{advance, effect, frame, initial, stage_next, tracker};
use crate::{ActionSpace, unit_feature};
use bota_proto::{EntityId, EventKind, Team};

#[test]
fn restoration_channels_preserve_each_other_and_invalidate_pre_event_provenance() {
    let (mut view, mut tracker) = initial(Team::Radiant);
    let source = view.players[0].unit.expect("source");
    let target = view.players[1].unit.expect("target");
    for (hp, mana, hp_expected, mana_expected) in [
        (0, 5, None, Some((2, 5))),
        (20, 0, Some((3, 20)), Some((2, 5))),
        (0, 35, Some((3, 20)), Some((4, 35))),
        (11, 12, Some((5, 11)), Some((5, 12))),
        (0, 0, Some((5, 11)), Some((5, 12))),
        (10, 0, Some((7, 10)), Some((5, 12))),
    ] {
        stage_next(&mut tracker, &mut view);
        let prior = tracker.provenance();
        let space = ActionSpace::from_tracker(&tracker).expect("pre-event space");
        tracker
            .observe_events(view.tick, &[heal(Some(source), target, hp, mana)])
            .expect("report");
        let from = tracker.entity(source).expect("source");
        let into = tracker.entity(target).expect("target");
        assert!(!prior.matches(&tracker));
        assert!(!space.matches_tracker(&tracker));
        assert_eq!(
            [
                from.last_heal_dealt.map(|event| (event.tick, event.amount)),
                into.last_heal_received
                    .map(|event| (event.tick, event.amount)),
                from.last_mana_restoration_dealt
                    .map(|event| (event.tick, event.amount)),
                into.last_mana_restoration_received
                    .map(|event| (event.tick, event.amount)),
            ],
            [hp_expected, hp_expected, mana_expected, mana_expected],
            "tick {}",
            view.tick
        );
        assert_eq!(
            into.last_heal_received.map(|report| report.counterpart),
            hp_expected.map(|_| Some(source)),
        );
    }
}

#[test]
fn invalid_restoration_batches_reject_the_valid_prefix_and_accept_the_inclusive_limit() {
    let (mut view, mut tracker) = initial(Team::Radiant);
    let target = view.players[0].unit.expect("hero");
    stage_next(&mut tracker, &mut view);
    let prior = tracker.provenance();
    for (hp, mana, channel, value) in [
        (-1, 0, "health", -1),
        (1_000_001, 0, "health", 1_000_001),
        (0, -1, "mana", -1),
        (0, 1_000_001, "mana", 1_000_001),
    ] {
        let events = [heal(None, target, 5, 6), heal(None, target, hp, mana)];
        let error = tracker
            .observe_events(view.tick, &events)
            .expect_err("invalid restoration");
        assert_eq!(
            error.to_string(),
            format!("Healed {channel} report {value} is outside 0..=1000000")
        );
        assert!(prior.matches(&tracker));
    }
    tracker
        .observe_events(view.tick, &[heal(None, target, 1_000_000, 1_000_000)])
        .expect("inclusive limit");
    let hero = tracker.entity(target).expect("hero");
    assert_eq!(hero.last_heal_received.expect("health").amount, 1_000_000);
    assert_eq!(
        hero.last_mana_restoration_received.expect("mana").amount,
        1_000_000
    );
}

#[test]
fn prior_tick_restoration_features_expire_or_evict_without_erasing_reports() {
    for evicted in [false, true] {
        let (mut view, mut tracker) = initial(Team::Radiant);
        let own = view.players[0].unit.expect("hero");
        let mut events = vec![heal(None, own, 100, 75)];
        if evicted {
            events.extend((0..crate::MAX_RECENT_EVENTS).map(|_| heal(None, own, 0, 0)));
        }
        advance(&mut tracker, &mut view, &events);
        assert_eq!(&frame(&tracker).own_units()[0][78..84], &[0.0; 6]);
        for tick in 3..=if evicted { 3 } else { 483 } {
            advance(&mut tracker, &mut view, &[]);
            if ![3, 482, 483].contains(&tick) {
                continue;
            }
            let expected = if evicted || tick == 483 {
                [0.0; 6]
            } else {
                let age = (tick - 2) as f32 / 480.0;
                [1.0, 100.0 / 100_000.0, age, 1.0, 75.0 / 20_000.0, age]
            };
            assert_eq!(
                &frame(&tracker).own_units()[0][78..84],
                &expected,
                "evicted={evicted} tick={tick}"
            );
        }
        let hero = tracker
            .entity(own)
            .expect("bookkeeping survives feature expiry/eviction");
        assert_eq!(hero.last_heal_received.expect("health").amount, 100);
        assert_eq!(
            hero.last_mana_restoration_received.expect("mana").amount,
            75
        );
    }
}

#[test]
fn fog_decays_aura_and_raze_without_refresh_or_target_pointer() {
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
    let mut tracker = tracker(Team::Radiant, view.clone());
    view.units.retain(|unit| unit.id != enemy);
    for expected in [
        [1.0, 3.0 / 255.0, 1.0 / 240.0, 1.0 / 15.0, 1.0 / 15.0],
        [0.0; 5],
    ] {
        advance(&mut tracker, &mut view, &[]);
        let output = frame(&tracker);
        let token = &output.remembered_units()[0];
        assert_eq!(token[unit_feature::REMEMBERED], 1.0);
        assert_eq!(&token[73..78], &expected);
        assert!(
            ActionSpace::from_tracker(&tracker)
                .expect("space")
                .entity_index(enemy)
                .is_none()
        );
    }
}

pub(super) fn heal(
    source: Option<EntityId>,
    target: EntityId,
    amount: i32,
    mana: i32,
) -> EventKind {
    EventKind::Healed {
        source,
        target,
        amount,
        mana,
    }
}
