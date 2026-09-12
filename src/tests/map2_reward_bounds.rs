#![allow(
    clippy::float_arithmetic,
    reason = "Reward bound tests sum normalized components"
)]

use bota_proto::{EventKind, SlotId, Team, UnitKind};

use super::map2_reward::{
    advance, damage, death, id, initialized, match_info, own_hero, snapshot, unit,
};
use crate::{
    MAP2_REWARD_DENSE_BOUND, MAP2_REWARD_MAX_EVENTS, MAP2_REWARD_MAX_UNITS, Map2Reward,
    Map2RewardBreakdown, Map2RewardEnd,
};

#[test]
fn whole_episode_dense_return_is_bounded_for_both_signs_and_all_channels() {
    let (positive, gain) = extreme_profile(false, Map2RewardEnd::Draw);
    let (negative, cost) = extreme_profile(true, Map2RewardEnd::Draw);

    assert!(positive.abs() <= MAP2_REWARD_DENSE_BOUND);
    assert!(negative.abs() <= MAP2_REWARD_DENSE_BOUND);
    assert!(gain.gold > 0.0);
    assert!(gain.experience > 0.0);
    assert!(gain.hero_damage > 0.0);
    assert!(cost.gold < 0.0);
    assert!(cost.experience < 0.0);
    assert!(cost.hero_damage_taken < 0.0);
    assert!(cost.creep_damage_taken < 0.0);
    assert!(cost.other_damage_taken < 0.0);
    assert!(cost.mana_spent < 0.0);
}

#[test]
fn adverse_win_outranks_favorable_draw_time_cap_and_loss_across_interval_drains() {
    let (win, _) = extreme_profile(true, Map2RewardEnd::Win);

    for end in [
        Map2RewardEnd::Draw,
        Map2RewardEnd::TimeCap,
        Map2RewardEnd::Loss,
    ] {
        let (other, _) = extreme_profile(false, end);
        assert!(win > other, "win={win}, {end:?}={other}");
    }
    assert!(win >= 1.0 - MAP2_REWARD_DENSE_BOUND);
}

#[test]
fn spending_event_budgets_does_not_clip_a_later_pressure_loop_reversal() {
    let mut reward = initialized();
    advance(&mut reward, &snapshot(2), &[damage(1, 2, 1_000_000)]);
    let mut forward = snapshot(3);
    for unit in &mut forward.units {
        if unit.kind == UnitKind::CreepMelee {
            unit.pos.x += bota_proto::Fixed::from_int(200);
        }
    }
    let before = reward.state();

    let outward = advance(&mut reward, &forward, &[]);
    let returning = advance(&mut reward, &snapshot(4), &[]);

    assert!(outward.lane_pressure > 0.0);
    assert!(returning.lane_pressure < 0.0);
    assert!((outward.total + returning.total).abs() < 1.0e-12);
    assert_eq!(before.remaining, reward.state().remaining);
}

#[test]
fn terminal_keeps_final_tower_health_progress_instead_of_canceling_it() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(4))
        .unwrap()
        .hp = 500;
    let interval = advance(&mut reward, &view, &[]);

    let terminal = reward.finish(Map2RewardEnd::Draw).unwrap();

    assert!((interval.tower_health - 0.025).abs() < 1.0e-12);
    assert_eq!(terminal.tower_health, 0.0);
    assert!((interval.total + terminal.total - 0.025).abs() < 1.0e-12);
}

#[test]
fn exactly_the_event_batch_limit_is_accepted_without_journal_loss() {
    let mut reward = initialized();
    let events = vec![damage(1, 2, 1); MAP2_REWARD_MAX_EVENTS];

    let result = advance(&mut reward, &snapshot(2), &events);

    assert_eq!(
        result.observations.hero_damage_dealt,
        MAP2_REWARD_MAX_EVENTS as u64
    );
    assert!(result.hero_damage > 0.0);
}

#[test]
fn visible_unit_bound_accepts_boundary_and_rejects_excess_before_staging() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    for index in 9..=MAP2_REWARD_MAX_UNITS as u32 {
        view.units
            .push(unit(index, UnitKind::CreepNeutral, Team::Neutral, 500));
    }
    advance(&mut reward, &view, &[]);
    view.tick = 3;
    view.units.push(unit(
        MAP2_REWARD_MAX_UNITS as u32 + 1,
        UnitKind::CreepNeutral,
        Team::Neutral,
        500,
    ));

    let error = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: visible units has 4097 entries; maximum is 4096"
    );
    assert_eq!(reward.state().completed_tick, Some(2));
    assert_eq!(advance(&mut reward, &snapshot(3), &[]).ticks, 1);
}

#[test]
fn identity_capacity_failure_can_retry_a_corrected_snapshot() {
    let mut reward = initialized();
    for (tick, start) in [(2, 9), (3, 4097)] {
        let mut view = snapshot(tick);
        for index in start..start + 4088 {
            view.units
                .push(unit(index, UnitKind::CreepNeutral, Team::Neutral, 500));
        }
        advance(&mut reward, &view, &[]);
    }
    let mut view = snapshot(4);
    for index in 8185..8201 {
        view.units
            .push(unit(index, UnitKind::CreepNeutral, Team::Neutral, 500));
    }

    let error = reward.observe_snapshot(&view).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: identity history has 8200 entries; maximum is 8192"
    );
    assert_eq!(advance(&mut reward, &snapshot(4), &[]).total, 0.0);
}

#[test]
fn last_seen_creep_classification_has_an_explicit_expiry_boundary() {
    let recent = fog_death(481);
    let expired = fog_death(482);

    assert_eq!(recent.observations.lane_last_hits, 1);
    assert_eq!(expired.observations.lane_last_hits, 0);
    assert_eq!(expired.observations.unattributed_deaths, 1);
}

#[test]
fn public_xp_decrease_is_rejected_without_rebasing_or_replenishing_budget() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[0].xp = 100;
    advance(&mut reward, &view, &[]);
    let before = reward.state();

    let error = reward.observe_snapshot(&snapshot(3)).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: public cumulative XP decreased"
    );
    assert_eq!(reward.state(), before);
}

#[test]
fn ability_cast_event_without_observed_mana_spend_is_not_a_mana_cost() {
    let mut reward = initialized();
    let event = EventKind::AbilityCast {
        caster: id(1),
        ability: bota_proto::AbilityId(13),
    };

    let result = advance(&mut reward, &snapshot(2), &[event]);

    assert_eq!(result.mana_spent, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn disappearance_without_death_evidence_never_creates_a_last_hit() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(6));

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.observations.lane_last_hits, 0);
    assert_eq!(result.gold, 0.0);
}

#[test]
fn structured_destruction_before_died_does_not_suppress_the_paid_bounty() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(4));
    let events = [
        EventKind::StructureDestroyed {
            unit: id(4),
            team: Team::Dire,
        },
        death(4, 1, 100),
    ];

    let result = advance(&mut reward, &view, &events);

    assert_eq!(result.observations.own_gold_earned, 100);
    assert!(result.tower_health > 0.0);
    assert_eq!(result.observations.lane_last_hits, 0);
}

#[test]
fn a_single_larger_hit_and_identical_smaller_hits_have_equal_credit() {
    let mut joined = initialized();
    let mut split = initialized();

    let joined = advance(&mut joined, &snapshot(2), &[damage(1, 2, 200)]);
    let split = advance(
        &mut split,
        &snapshot(2),
        &[damage(1, 2, 100), damage(1, 2, 100)],
    );

    assert!((joined.hero_damage - split.hero_damage).abs() < 1.0e-12);
    assert_eq!(
        joined.observations.hero_damage_dealt,
        split.observations.hero_damage_dealt
    );
}

fn fog_death(tick: u32) -> Map2RewardBreakdown {
    assert!((481..=482).contains(&tick));
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(6));
    for next in 2..tick {
        view.tick = next;
        advance(&mut reward, &view, &[]);
    }
    view.tick = tick;
    advance(&mut reward, &view, &[death(6, 1, 40)])
}

fn extreme_profile(adverse: bool, end: Map2RewardEnd) -> (f64, Map2RewardBreakdown) {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    own_hero(&mut view).mana = 1_000_000;
    own_hero(&mut view).max_mana = 1_000_000;
    view.units
        .push(unit(9, UnitKind::CreepMelee, Team::Radiant, 450));
    view.units
        .push(unit(10, UnitKind::CreepMelee, Team::Dire, 550));
    advance(&mut reward, &view, &[]);
    view.tick = 2;
    let victim = if adverse { 5 } else { 6 };
    view.units.retain(|unit| unit.id != id(victim));
    view.players[usize::from(adverse)].xp = 1_000_000_000;
    let tower = if adverse { 3 } else { 4 };
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(tower))
        .unwrap()
        .hp = 0;
    for unit in &mut view.units {
        if unit.kind == UnitKind::CreepMelee {
            unit.pos.x = bota_proto::Fixed::from_int(if adverse { 0 } else { 1000 });
        }
    }
    let events = if adverse {
        own_hero(&mut view).mana = 0;
        vec![
            damage(2, 1, 1_000_000),
            damage(6, 1, 1_000_000),
            damage(4, 1, 1_000_000),
            death(5, 2, 1_000_000),
        ]
    } else {
        vec![damage(1, 2, 1_000_000), death(6, 1, 1_000_000)]
    };
    let first = advance(&mut reward, &view, &events);
    view.tick = 3;
    own_hero(&mut view).mana = 1_000_000;
    for unit in &mut view.units {
        if unit.kind == UnitKind::CreepMelee {
            unit.pos.x =
                bota_proto::Fixed::from_int(if unit.team == Team::Radiant { 450 } else { 550 });
        }
    }
    let second = advance(&mut reward, &view, &[]);
    let terminal = reward.finish(end).unwrap();
    (first.total + second.total + terminal.total, first)
}
