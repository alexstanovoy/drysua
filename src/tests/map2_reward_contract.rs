#![allow(
    clippy::float_arithmetic,
    reason = "Reward tests compare normalized components"
)]

use bota_proto::{EntityId, EventKind, MapId, SlotId, Team, UnitKind, Vec2};

use super::map2_reward::{advance, damage, death, id, initialized, match_info, snapshot, unit};
use crate::{MAP2_REWARD_MAX_EVENTS, MAP2_REWARD_SCHEMA_HASH, Map2Reward, Map2RewardEnd};

#[test]
fn map2_reward_rejects_other_maps_and_has_its_own_schema_identity() {
    let mut info = match_info();
    info.map = MapId(0);

    let error = Map2Reward::new(SlotId(0), &info).unwrap_err();

    assert_eq!(error.to_string(), "Map2 reward: expected map 2");
    assert_ne!(MAP2_REWARD_SCHEMA_HASH, 0);
}

#[test]
fn reward_requires_matching_events_before_next_snapshot_or_flush() {
    let mut reward = initialized();
    reward.observe_snapshot(&snapshot(2)).unwrap();

    let snapshot_error = reward.observe_snapshot(&snapshot(3)).unwrap_err();
    let flush_error = reward.take_interval().unwrap_err();
    let event_error = reward.observe_events(3, &[]).unwrap_err();

    assert_eq!(
        snapshot_error.to_string(),
        "Map2 reward: Snapshot arrived before pending Events"
    );
    assert_eq!(
        flush_error.to_string(),
        "Map2 reward: interval has pending Events"
    );
    assert_eq!(
        event_error.to_string(),
        "Map2 reward: Events tick does not match pending Snapshot"
    );
}

#[test]
fn reward_rejects_gapped_ticks_without_changing_committed_state() {
    let mut reward = initialized();
    let before = reward.state();

    let error = reward.observe_snapshot(&snapshot(3)).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: Snapshot ticks must be contiguous"
    );
    assert_eq!(reward.state(), before);
}

#[test]
fn duplicate_events_are_rejected_but_two_identical_hits_in_one_batch_both_count() {
    let mut reward = initialized();
    let events = [damage(1, 2, 20), damage(1, 2, 20)];
    let result = advance(&mut reward, &snapshot(2), &events);
    let before = reward.state();

    let error = reward.observe_events(2, &events).unwrap_err();

    assert_eq!(result.observations.hero_damage_dealt, 40);
    assert_eq!(
        error.to_string(),
        "Map2 reward: Events without pending Snapshot"
    );
    assert_eq!(reward.state(), before);
}

#[test]
fn all_events_before_a_decision_count_even_when_more_than_sixty_four_arrive() {
    let mut reward = initialized();
    for tick in 2..=4 {
        reward.observe_snapshot(&snapshot(tick)).unwrap();
        reward
            .observe_events(tick, &vec![damage(1, 2, 1); 100])
            .unwrap();
    }

    let result = reward.take_interval().unwrap();
    let empty = reward.take_interval().unwrap();

    assert_eq!(result.ticks, 3);
    assert_eq!(result.observations.hero_damage_dealt, 300);
    assert_eq!(empty.total, 0.0);
    assert_eq!(empty.ticks, 0);
}

#[test]
fn interval_partitioning_preserves_rewards_and_budget_state() {
    let mut split = initialized();
    let mut whole = initialized();
    let mut total = 0.0;
    for tick in 2..=10 {
        let events = [damage(1, 2, 10), damage(6, 1, 5)];
        total += advance(&mut split, &snapshot(tick), &events).total;
        whole.observe_snapshot(&snapshot(tick)).unwrap();
        whole.observe_events(tick, &events).unwrap();
    }

    let result = whole.take_interval().unwrap();

    assert!((result.total - total).abs() < 1.0e-12);
    assert_eq!(split.state(), whole.state());
}

#[test]
fn malformed_event_batches_fail_before_charging_any_valid_prefix() {
    let mut reward = initialized();
    reward.observe_snapshot(&snapshot(2)).unwrap();
    let before = reward.state();

    let error = reward
        .observe_events(2, &[damage(1, 2, 10), damage(1, 2, -1)])
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: damage amount outside 0..=1000000"
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

#[test]
fn death_of_a_still_living_snapshot_victim_is_rejected_before_any_credit() {
    let mut reward = initialized();
    reward.observe_snapshot(&snapshot(2)).unwrap();
    let before = reward.state();

    let error = reward
        .observe_events(2, &[damage(1, 2, 10), death(6, 1, 40)])
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: Died victim is still alive in matching Snapshot"
    );
    assert_eq!(reward.state(), before);
    reward.observe_events(2, &[]).unwrap();
    assert_eq!(reward.take_interval().unwrap().total, 0.0);
}

#[test]
fn destroyed_structure_cannot_remain_alive_in_the_matching_snapshot() {
    let mut reward = initialized();
    reward.observe_snapshot(&snapshot(2)).unwrap();
    let event = EventKind::StructureDestroyed {
        unit: id(4),
        team: Team::Dire,
    };

    let error = reward.observe_events(2, &[event]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: destroyed structure is still alive in matching Snapshot"
    );
    assert_eq!(reward.state().completed_tick, Some(1));
}

#[test]
fn initial_hidden_scoreboard_hero_cannot_be_destroyed_as_a_structure() {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    view.units.retain(|unit| unit.id != id(2));
    reward.observe_snapshot(&view).unwrap();
    let event = EventKind::StructureDestroyed {
        unit: id(2),
        team: Team::Dire,
    };

    let error = reward.observe_events(1, &[event]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: StructureDestroyed names a public hero body"
    );
    assert_eq!(reward.state().completed_tick, None);
    reward.observe_events(1, &[]).unwrap();
    assert_eq!(reward.take_interval().unwrap().total, 0.0);
}

#[test]
fn a_newly_seen_creep_cannot_supply_structure_destruction_evidence() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    let mut creep = unit(9, UnitKind::CreepMelee, Team::Dire, 550);
    creep.hp = 0;
    view.units.push(creep);
    reward.observe_snapshot(&view).unwrap();
    let event = EventKind::StructureDestroyed {
        unit: id(9),
        team: Team::Dire,
    };

    let error = reward.observe_events(2, &[event]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: StructureDestroyed conflicts with public unit identity"
    );
    assert_eq!(reward.state().completed_tick, Some(1));
}

#[test]
fn a_dead_full_generation_handle_cannot_reappear_alive() {
    let mut reward = initialized();
    let mut dead = snapshot(2);
    dead.units.retain(|unit| unit.id != id(6));
    advance(&mut reward, &dead, &[death(6, 1, 40)]);

    let error = reward.observe_snapshot(&snapshot(3)).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: dead unit reappeared without a new generation"
    );
    assert_eq!(reward.state().completed_tick, Some(2));
}

#[test]
fn snapshot_capacity_failure_does_not_stick_an_unfinishable_pending_tick() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    for index in 9..72 {
        view.units
            .push(unit(index, UnitKind::Tower, Team::Dire, 800));
    }

    let error = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: tower history has 65 entries; maximum is 64"
    );
    assert_eq!(advance(&mut reward, &snapshot(2), &[]).total, 0.0);
}

#[test]
fn oversized_event_batch_is_descriptive_and_does_not_consume_pending_tick() {
    let mut reward = initialized();
    reward.observe_snapshot(&snapshot(2)).unwrap();
    let events = vec![damage(1, 2, 1); MAP2_REWARD_MAX_EVENTS + 1];

    let error = reward.observe_events(2, &events).unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "Map2 reward: event batch has {} entries; maximum is {}",
            events.len(),
            MAP2_REWARD_MAX_EVENTS
        )
    );
    reward.observe_events(2, &[]).unwrap();
    assert_eq!(reward.take_interval().unwrap().total, 0.0);
}

#[test]
fn dying_hero_identity_is_retained_from_public_scoreboard_not_current_unit_presence() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[1].unit = None;
    view.units.retain(|unit| unit.id != id(2));

    let result = advance(&mut reward, &view, &[damage(1, 2, 50), death(2, 1, 200)]);

    assert_eq!(result.observations.hero_damage_dealt, 50);
    assert_eq!(result.observations.own_gold_earned, 200);
    assert_eq!(result.observations.lane_last_hits, 0);
}

#[test]
fn unknown_victim_and_reused_generation_never_receive_guessed_damage_or_last_hit_credit() {
    let mut reward = initialized();
    let replaced = EntityId {
        idx: 6,
        generation: 2,
    };
    let death = EventKind::Died {
        unit: replaced,
        killer: Some(id(1)),
        denied: false,
        gold: 40,
    };

    let result = advance(&mut reward, &snapshot(2), &[damage(1, 99, 500), death]);

    assert_eq!(result.total, 0.0);
    assert_eq!(result.observations.unattributed_damage_events, 1);
    assert_eq!(result.observations.unattributed_deaths, 1);
    assert_eq!(result.observations.lane_last_hits, 0);
}

#[test]
fn unknown_sources_get_no_invented_hero_credit_and_increment_attribution_gaps_once() {
    let mut reward = initialized();

    let result = advance(
        &mut reward,
        &snapshot(2),
        &[damage(99, 2, 40), damage(99, 100, 20)],
    );

    assert_eq!(result.observations.unattributed_damage_events, 2);
    assert_eq!(result.hero_damage, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn npc_damage_to_enemy_hero_and_friendly_fire_never_earn_hero_damage_credit() {
    let mut reward = initialized();

    let result = advance(
        &mut reward,
        &snapshot(2),
        &[damage(5, 2, 200), damage(1, 5, 100)],
    );

    assert_eq!(result.hero_damage, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn explicit_environmental_damage_is_not_mislabelled_as_an_unknown_unit() {
    let mut reward = initialized();
    let event = EventKind::Damaged {
        source: None,
        target: id(1),
        amount: 20,
        kind: bota_proto::DamageKind::Pure,
        crit: false,
    };

    let result = advance(&mut reward, &snapshot(2), &[event]);

    assert_eq!(result.observations.other_damage_taken, 20);
    assert_eq!(result.observations.unattributed_damage_taken, 0);
}

#[test]
fn the_same_death_never_pays_a_second_bounty_or_last_hit() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(6));
    let events = [death(6, 1, 40), death(6, 1, 40)];

    let result = advance(&mut reward, &view, &events);

    assert_eq!(result.observations.own_gold_earned, 40);
    assert_eq!(result.observations.lane_last_hits, 1);
    assert_eq!(result.observations.duplicate_deaths, 1);
}

#[test]
fn tower_hp_and_wave_potentials_are_symmetric_between_playable_sides() {
    let mut original = initialized();
    let mut info = match_info();
    info.picks[0].team = Team::Dire;
    info.picks[1].team = Team::Radiant;
    let mut reflected = Map2Reward::new(SlotId(0), &info).unwrap();
    let reflect = |mut view: bota_proto::WorldView| {
        view.viewer = Some(Team::Dire);
        for unit in &mut view.units {
            unit.pos.x = bota_proto::Fixed::from_int(1000) - unit.pos.x;
            unit.team = if unit.team == Team::Radiant {
                Team::Dire
            } else {
                Team::Radiant
            };
        }
        view.players[0].team = Team::Dire;
        view.players[1].team = Team::Radiant;
        view
    };
    advance(&mut reflected, &reflect(snapshot(1)), &[]);
    let mut view = snapshot(2);
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(4))
        .unwrap()
        .hp -= 100;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(5))
        .unwrap()
        .pos
        .x += bota_proto::Fixed::from_int(100);

    let left = advance(&mut original, &view, &[damage(1, 2, 50)]);
    let right = advance(&mut reflected, &reflect(view), &[damage(1, 2, 50)]);

    assert!((left.total - right.total).abs() < 1.0e-12);
    assert_eq!(original.state(), reflected.state());
}

#[test]
fn neutral_last_hits_are_distinct_from_lane_and_building_bounties() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units
        .push(unit(9, UnitKind::CreepNeutral, Team::Neutral, 500));
    advance(&mut reward, &view, &[]);
    view.tick = 3;
    view.units
        .retain(|unit| unit.id != id(9) && unit.id != id(4));

    let result = advance(&mut reward, &view, &[death(9, 1, 40), death(4, 1, 100)]);

    assert_eq!(result.observations.neutral_last_hits, 1);
    assert_eq!(result.observations.lane_last_hits, 0);
    assert_eq!(result.observations.own_gold_earned, 140);
}

#[test]
fn spectator_or_unfogged_enemy_gold_is_rejected() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.viewer = None;
    let spectator = reward.observe_snapshot(&view).unwrap_err();
    view.viewer = Some(Team::Radiant);
    view.players[1].gold = Some(10);

    let enemy_gold = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        spectator.to_string(),
        "Map2 reward: Snapshot viewer does not match own team"
    );
    assert_eq!(
        enemy_gold.to_string(),
        "Map2 reward: opposing private scoreboard fields are present"
    );
}

#[test]
fn hidden_scoreboard_body_cannot_reclassify_a_known_creep_as_a_hero() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[1].unit = Some(id(6));
    view.units
        .retain(|unit| unit.id != id(2) && unit.id != id(6));

    let error = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: scoreboard body conflicts with public identity history"
    );
    assert_eq!(reward.state().completed_tick, Some(1));
}

#[test]
fn initial_scoreboard_body_must_be_a_hero_even_without_identity_history() {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    view.units[0].kind = UnitKind::CreepMelee;

    let error = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: scoreboard body is not a hero"
    );
    assert_eq!(reward.state().completed_tick, None);
}

#[test]
fn draw_and_time_cap_are_explicit_distinct_non_wins_and_end_the_accumulator() {
    let mut draw = initialized();
    let mut cap = initialized();

    let drawn = draw.finish(Map2RewardEnd::Draw).unwrap();
    let capped = cap.finish(Map2RewardEnd::TimeCap).unwrap();
    let error = cap.observe_snapshot(&snapshot(2)).unwrap_err();

    assert_eq!(drawn.terminal, 0.0);
    assert_eq!(capped.terminal, 0.0);
    assert_ne!(drawn.end, capped.end);
    assert_eq!(error.to_string(), "Map2 reward: episode already ended");
}

#[test]
fn terminal_closes_lane_potential_without_clipping_away_its_reversal() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(5))
        .unwrap()
        .pos = Vec2::from_ints(900, 0);
    let forward = advance(&mut reward, &view, &[]);

    let terminal = reward.finish(Map2RewardEnd::Draw).unwrap();

    assert!(forward.lane_pressure > 0.0);
    assert!((forward.lane_pressure + terminal.lane_pressure).abs() < 1.0e-12);
}

#[test]
fn repeated_damage_and_distinct_paid_creep_deaths_never_exceed_dense_episode_bound() {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    for index in 9..109 {
        view.units
            .push(unit(index, UnitKind::CreepMelee, Team::Dire, 550));
    }
    advance(&mut reward, &view, &[]);
    let mut total = 0.0;
    for index in 9..109 {
        view.tick += 1;
        view.units.retain(|unit| unit.id != id(index));
        let events = [damage(1, 2, 1_000_000), death(index, 1, 1_000_000)];
        total += advance(&mut reward, &view, &events).total;
    }

    let terminal = reward.finish(Map2RewardEnd::Draw).unwrap();

    assert!(total + terminal.total < 0.4);
    assert!(
        reward
            .state()
            .remaining
            .iter()
            .all(|remaining| *remaining >= 0.0 && *remaining <= 1.0)
    );
}

#[test]
fn numeric_entity_remapping_and_match_id_do_not_change_reward_state_or_breakdown() {
    let mut original = initialized();
    let mut info = match_info();
    info.match_id = u64::MAX;
    let mut remapped = Map2Reward::new(SlotId(0), &info).unwrap();
    let remap = |mut view: bota_proto::WorldView| {
        for unit in &mut view.units {
            unit.id.idx += 100;
        }
        for player in &mut view.players {
            player.unit = player.unit.map(|mut id| {
                id.idx += 100;
                id
            });
        }
        view
    };
    advance(&mut remapped, &remap(snapshot(1)), &[]);

    let left = advance(&mut original, &snapshot(2), &[damage(1, 2, 75)]);
    let right = advance(&mut remapped, &remap(snapshot(2)), &[damage(101, 102, 75)]);

    assert_eq!(left, right);
    assert_eq!(original.state(), remapped.state());
}
