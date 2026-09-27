#![allow(
    clippy::float_arithmetic,
    reason = "Reward tests compare normalized components"
)]

use super::map2_reward::{advance, damage, death, id, initialized, match_info, snapshot, unit};
use crate::{MAP2_REWARD_MAX_EVENTS, MAP2_REWARD_SCHEMA_HASH, Map2Reward};
use bota_proto::{EntityId, EventKind, MapId, SlotId, Team, UnitKind};

#[test]
fn reward_rejects_other_maps_and_preserves_schema_identity() {
    for map in [0, 1, u16::MAX] {
        let mut info = match_info();
        info.map = MapId(map);
        assert_eq!(
            Map2Reward::new(SlotId(0), &info).unwrap_err().to_string(),
            "Map2 reward: expected map 2"
        );
    }
    assert_ne!(MAP2_REWARD_SCHEMA_HASH, 0);
}

#[test]
fn stream_pairing_errors_leave_committed_state_and_pending_retry_intact() {
    let mut reward = initialized();
    let before = reward.state();
    assert_eq!(
        reward
            .observe_snapshot(&snapshot(3))
            .unwrap_err()
            .to_string(),
        "Map2 reward: Snapshot ticks must be contiguous"
    );
    assert_eq!(reward.state(), before);
    reward.observe_snapshot(&snapshot(2)).unwrap();
    for error in [
        reward
            .observe_snapshot(&snapshot(3))
            .unwrap_err()
            .to_string(),
        reward.take_interval().unwrap_err().to_string(),
        reward.observe_events(3, &[]).unwrap_err().to_string(),
    ]
    .into_iter()
    .zip([
        "Map2 reward: Snapshot arrived before pending Events",
        "Map2 reward: interval has pending Events",
        "Map2 reward: Events tick does not match pending Snapshot",
    ]) {
        assert_eq!(error.0, error.1);
    }
    reward
        .observe_events(2, &[damage(1, 2, 20), damage(1, 2, 20)])
        .unwrap();
    assert_eq!(
        reward
            .take_interval()
            .unwrap()
            .observations
            .hero_damage_dealt,
        40
    );
    let before = reward.state();
    assert_eq!(
        reward.observe_events(2, &[]).unwrap_err().to_string(),
        "Map2 reward: Events without pending Snapshot"
    );
    assert_eq!(reward.state(), before);
}

#[test]
fn interval_partitioning_keeps_all_events_budgets_and_empty_drains() {
    let mut split = initialized();
    let mut whole = initialized();
    let mut total = 0.0;
    for tick in 2..=10 {
        let mut events = vec![damage(1, 2, 1); 100];
        events.push(damage(6, 1, 5));
        total += advance(&mut split, &snapshot(tick), &events).total;
        whole.observe_snapshot(&snapshot(tick)).unwrap();
        whole.observe_events(tick, &events).unwrap();
    }
    let result = whole.take_interval().unwrap();
    assert_eq!(result.ticks, 9);
    assert_eq!(result.observations.hero_damage_dealt, 900);
    assert!((result.total - total).abs() < 1.0e-12);
    assert_eq!(split.state(), whole.state());
    let empty = whole.take_interval().unwrap();
    assert_eq!(empty.total, 0.0);
    assert_eq!(empty.ticks, 0);
}

#[test]
fn malformed_event_batches_never_charge_valid_prefix_and_allow_corrected_retry() {
    for (events, expected) in [
        (
            vec![damage(1, 2, -1)],
            "Map2 reward: damage amount outside 0..=1000000".to_owned(),
        ),
        (
            vec![damage(1, 2, 1_000_001)],
            "Map2 reward: damage amount outside 0..=1000000".to_owned(),
        ),
        (
            vec![death(6, 1, 40)],
            "Map2 reward: Died victim is still alive in matching Snapshot".to_owned(),
        ),
        (
            vec![destroyed(4)],
            "Map2 reward: destroyed structure is still alive in matching Snapshot".to_owned(),
        ),
        (
            vec![damage(1, 2, 1); MAP2_REWARD_MAX_EVENTS],
            format!(
                "Map2 reward: event batch has {} entries; maximum is {}",
                MAP2_REWARD_MAX_EVENTS + 1,
                MAP2_REWARD_MAX_EVENTS
            ),
        ),
    ] {
        let mut reward = initialized();
        reward.observe_snapshot(&snapshot(2)).unwrap();
        let before = reward.state();
        let mut batch = vec![damage(1, 2, 10)];
        batch.extend(events);
        assert_eq!(
            reward.observe_events(2, &batch).unwrap_err().to_string(),
            expected
        );
        assert_eq!(reward.state(), before);
        reward.observe_events(2, &[damage(1, 2, 10)]).unwrap();
        assert_eq!(
            reward
                .take_interval()
                .unwrap()
                .observations
                .hero_damage_dealt,
            10
        );
    }
}

#[test]
fn public_identity_cannot_be_reclassified_by_scoreboard_or_destruction_events() {
    for scenario in 0..4 {
        let initial = matches!(scenario, 0 | 2);
        let mut reward = if initial {
            Map2Reward::new(SlotId(0), &match_info()).unwrap()
        } else {
            initialized()
        };
        let mut view = snapshot(if initial { 1 } else { 2 });
        let (event, expected) = match scenario {
            0 => {
                view.units.retain(|unit| unit.id != id(2));
                (
                    Some(destroyed(2)),
                    "Map2 reward: StructureDestroyed names a public hero body",
                )
            }
            1 => {
                let mut creep = unit(9, UnitKind::CreepMelee, Team::Dire, 550);
                creep.hp = 0;
                view.units.push(creep);
                (
                    Some(destroyed(9)),
                    "Map2 reward: StructureDestroyed conflicts with public unit identity",
                )
            }
            2 => {
                view.units[0].kind = UnitKind::CreepMelee;
                (None, "Map2 reward: scoreboard body is not a hero")
            }
            _ => {
                view.players[1].unit = Some(id(6));
                view.units
                    .retain(|unit| unit.id != id(2) && unit.id != id(6));
                (
                    None,
                    "Map2 reward: scoreboard body conflicts with public identity history",
                )
            }
        };
        let before = reward.state();
        let error = if let Some(event) = event {
            reward.observe_snapshot(&view).unwrap();
            reward.observe_events(view.tick, &[event]).unwrap_err()
        } else {
            reward.observe_snapshot(&view).unwrap_err()
        };
        assert_eq!(error.to_string(), expected);
        assert_eq!(reward.state(), before);
        if scenario < 2 {
            reward.observe_events(view.tick, &[]).unwrap();
            assert_eq!(reward.take_interval().unwrap().total, 0.0);
        }
    }
}

#[test]
fn invalid_snapshot_privacy_and_capacity_fail_before_staging() {
    for scenario in 0..3 {
        let mut reward = initialized();
        let mut view = snapshot(2);
        let expected = match scenario {
            0 => {
                view.viewer = None;
                "Map2 reward: Snapshot viewer does not match own team"
            }
            1 => {
                view.players[1].gold = Some(10);
                "Map2 reward: opposing private scoreboard fields are present"
            }
            _ => {
                for index in 9..72 {
                    view.units
                        .push(unit(index, UnitKind::Tower, Team::Dire, 800));
                }
                "Map2 reward: tower history has 65 entries; maximum is 64"
            }
        };
        let before = reward.state();
        assert_eq!(
            reward.observe_snapshot(&view).unwrap_err().to_string(),
            expected
        );
        assert_eq!(reward.state(), before);
        assert_eq!(advance(&mut reward, &snapshot(2), &[]).total, 0.0);
    }
}

#[test]
fn deaths_pay_once_by_full_identity_and_retained_hero_identity_survives_disappearance() {
    for (victim, gold, hero_damage, lane_hits) in [(2, 200, 50, 0), (6, 40, 0, 1)] {
        let mut reward = initialized();
        let mut view = snapshot(2);
        if victim == 2 {
            view.players[1].unit = None;
        }
        view.units.retain(|unit| unit.id != id(victim));
        let result = advance(
            &mut reward,
            &view,
            &[
                damage(1, victim, 50),
                death(victim, 1, gold),
                death(victim, 1, gold),
            ],
        );
        assert_eq!(result.observations.own_gold_earned, gold as u64);
        assert_eq!(result.observations.hero_damage_dealt, hero_damage);
        assert_eq!(result.observations.lane_last_hits, lane_hits);
        assert_eq!(result.observations.duplicate_deaths, 1);
        assert_eq!(
            reward
                .observe_snapshot(&snapshot(3))
                .unwrap_err()
                .to_string(),
            "Map2 reward: dead unit reappeared without a new generation"
        );
        assert_eq!(reward.state().completed_tick, Some(2));
    }
    let mut reward = initialized();
    let replaced = EventKind::Died {
        unit: EntityId {
            idx: 6,
            generation: 2,
        },
        killer: Some(id(1)),
        denied: false,
        gold: 40,
    };
    let result = advance(&mut reward, &snapshot(2), &[damage(1, 99, 500), replaced]);
    assert_eq!(result.total, 0.0);
    assert_eq!(result.observations.unattributed_damage_events, 1);
    assert_eq!(result.observations.unattributed_deaths, 1);
    assert_eq!(result.observations.lane_last_hits, 0);
}

fn destroyed(index: u32) -> EventKind {
    EventKind::StructureDestroyed {
        unit: id(index),
        team: Team::Dire,
    }
}
