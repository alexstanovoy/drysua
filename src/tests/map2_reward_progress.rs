use bota_proto::{
    EffectId, EffectView, EventKind, Fixed, ItemId, SlotId, Team, UnitKind, Vec2, WorldView,
};

use super::{advance, damage, death, id, match_info, own_hero, snapshot};
use crate::{Map2Reward, Map2RewardEnd};

#[test]
fn progress_schema_declares_version_three_lease_latch_and_asymmetric_bounds() {
    assert_eq!(crate::MAP2_REWARD_SCHEMA_VERSION, 3);
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.starts_with("drysua-map2-reward/v3;"));
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.contains("progress_debt="));
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.contains("progress_flags="));
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.contains("positive_bound.255"));
    assert_ne!(crate::MAP2_REWARD_SCHEMA_HASH, 699_687_995_158_557_285);
}

#[test]
fn stagnation_2699_is_free_2700_charges_only_base_and_2701_charges_rate() {
    for elapsed in [2699, 2700, 2701] {
        let mut view = snapshot(1);
        let mut reward = observer(&view);

        feed_to(&mut reward, &mut view, 1 + elapsed);
        let result = reward.take_interval().unwrap();

        assert_close(
            result.stagnation_base,
            if elapsed >= 2700 { -0.02 } else { 0.0 },
        );
        assert_close(
            result.stagnation_ticks_cost,
            if elapsed > 2700 { -0.000002 } else { 0.0 },
        );
        assert_eq!(reward.state().stagnation_ticks, elapsed.min(2700));
        assert_eq!(reward.state().stagnation_base_charged, elapsed >= 2700);
        assert_eq!(
            result.observations.stagnation_idle_ticks,
            u64::from(elapsed)
        );
        assert_eq!(
            result.observations.stagnation_base_charges,
            u64::from(elapsed >= 2700)
        );
        assert_eq!(
            result.observations.stagnation_charged_ticks,
            u64::from(elapsed.saturating_sub(2699))
        );
    }
}

#[test]
fn useful_progress_on_the_first_threshold_tick_prevents_the_base_charge() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 2700);
    reward.take_interval().unwrap();

    tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.stagnation_base, 0.0);
    assert_eq!(result.stagnation_ticks_cost, 0.0);
    assert_eq!(reward.state().stagnation_ticks, 2696);
    assert_eq!(reward.state().activity_ticks_left, 29);
    assert!(!reward.state().stagnation_base_charged);
    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_HERO_DAMAGE
    );
}

#[test]
fn one_event_grants_exactly_thirty_active_ticks_and_repays_only_ninety_debt_ticks() {
    let (mut reward, mut view) = stalled();

    tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
    assert_eq!(reward.state().activity_ticks_left, 29);
    let target = view.tick + 29;
    feed_to(&mut reward, &mut view, target);
    let active = reward.take_interval().unwrap();

    assert_eq!(reward.state().stagnation_ticks, 2610);
    assert_eq!(reward.state().activity_ticks_left, 0);
    assert!(reward.state().stagnation_base_charged);
    assert_eq!(active.observations.stagnation_active_ticks, 30);
    assert_eq!(active.observations.stagnation_repaid_ticks, 90);
    assert_eq!(active.stagnation_base, 0.0);
    tick(&mut reward, &mut view, &[]);
    assert_eq!(reward.state().stagnation_ticks, 2611);
}

#[test]
fn continuous_thirty_seconds_of_activity_clears_debt_and_only_then_rearms_base() {
    let (mut reward, mut view) = stalled();
    own_hero(&mut view).effects = vec![aura(Some(30))];
    let target = view.tick + 899;
    feed_to(&mut reward, &mut view, target);
    assert_eq!(reward.state().stagnation_ticks, 3);
    assert!(reward.state().stagnation_base_charged);

    tick(&mut reward, &mut view, &[]);
    let repaid = reward.take_interval().unwrap();

    assert_eq!(reward.state().stagnation_ticks, 0);
    assert!(!reward.state().stagnation_base_charged);
    assert_eq!(repaid.observations.stagnation_repaid_ticks, 2700);
    own_hero(&mut view).effects.clear();
    let lease_end = view.tick + 29;
    feed_to(&mut reward, &mut view, lease_end);
    reward.take_interval().unwrap();
    let threshold = view.tick + 2700;
    feed_to(&mut reward, &mut view, threshold);
    let second_bout = reward.take_interval().unwrap();
    assert_close(second_bout.stagnation_base, -0.02);
    assert_eq!(second_bout.stagnation_ticks_cost, 0.0);
    assert_eq!(second_bout.observations.stagnation_base_charges, 1);
}

#[test]
fn intermittent_partial_repayment_never_recharges_the_latched_base() {
    let (mut reward, mut view) = stalled();
    for _ in 0..5 {
        tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
        let after_lease = view.tick + 29;
        feed_to(&mut reward, &mut view, after_lease);
        let back_at_threshold = view.tick + 90;
        feed_to(&mut reward, &mut view, back_at_threshold);

        let result = reward.take_interval().unwrap();

        assert_eq!(result.stagnation_base, 0.0);
        assert_close(result.stagnation_ticks_cost, -0.000002);
        assert_eq!(result.observations.stagnation_base_charges, 0);
        assert!(reward.state().stagnation_base_charged);
        assert_eq!(reward.state().stagnation_ticks, 2700);
    }
}

#[test]
fn all_completed_dead_ticks_accrue_debt_and_respawn_does_not_reset_it() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 101);
    view.players[0].unit = None;
    view.units.retain(|unit| unit.id != id(1));
    feed_to(&mut reward, &mut view, 151);
    assert_eq!(reward.state().stagnation_ticks, 150);
    let mut returned = snapshot(152);
    own_hero(&mut returned).id.generation = 2;
    returned.players[0].unit = Some(own_hero(&mut returned).id);

    advance(&mut reward, &returned, &[]);

    assert_eq!(reward.state().stagnation_ticks, 151);
    assert_eq!(reward.state().activity_ticks_left, 0);
}

#[test]
fn passive_cash_income_cannot_keep_the_progress_clock_alive() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    for _ in 0..2700 {
        view.players[0].gold = view.players[0].gold.map(|gold| gold + 1);
        tick(&mut reward, &mut view, &[]);
    }

    let result = reward.take_interval().unwrap();

    assert_close(result.stagnation_base, -0.02);
    assert_eq!(result.observations.progress_reasons, 0);
    assert_eq!(result.observations.stagnation_active_ticks, 0);
    assert_eq!(result.gold, 0.0);
}

#[test]
fn own_bounty_and_xp_count_even_if_opponent_xp_gain_is_larger() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    view.players[0].xp += 1;
    view.players[1].xp += 100;
    view.units.retain(|unit| unit.id != id(6));

    tick(&mut reward, &mut view, &[death(6, 1, 40)]);
    let result = reward.take_interval().unwrap();

    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_GOLD | crate::MAP2_PROGRESS_XP | crate::MAP2_PROGRESS_CREEP_KILL
    );
    assert_eq!(result.observations.creep_kills, 1);
    assert_eq!(result.observations.lane_last_hits, 1);
    assert!(result.experience < 0.0);
    assert_eq!(reward.state().activity_ticks_left, 29);
}

#[test]
fn zero_bounty_creep_kill_is_activity_without_changing_old_gold_or_last_hit_reward() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    view.units.retain(|unit| unit.id != id(6));

    tick(&mut reward, &mut view, &[death(6, 1, 0)]);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.creep_kills, 1);
    assert_eq!(result.observations.lane_last_hits, 0);
    assert_eq!(result.gold, 0.0);
    assert_eq!(result.total, 0.0);
    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_CREEP_KILL
    );
}

#[test]
fn neutral_zero_bounty_kill_counts_as_activity_but_enemy_denies_and_duplicate_deaths_do_not() {
    let mut view = snapshot(1);
    let neutral = view.units.iter_mut().find(|unit| unit.id == id(6)).unwrap();
    neutral.kind = UnitKind::CreepNeutral;
    neutral.team = Team::Neutral;
    let mut reward = observer(&view);
    view.units.retain(|unit| unit.id != id(6));
    tick(&mut reward, &mut view, &[death(6, 1, 0), death(6, 1, 0)]);
    let killed = reward.take_interval().unwrap();
    assert_eq!(killed.observations.creep_kills, 1);
    assert_eq!(killed.observations.duplicate_deaths, 1);
    assert_eq!(
        killed.observations.progress_reasons,
        crate::MAP2_PROGRESS_CREEP_KILL
    );

    let mut view = snapshot(1);
    let mut reward = observer(&view);
    view.units.retain(|unit| unit.id != id(6));
    let deny = EventKind::Died {
        unit: id(6),
        killer: Some(id(2)),
        denied: true,
        gold: 0,
    };
    tick(&mut reward, &mut view, &[deny]);

    let result = reward.take_interval().unwrap();
    assert_eq!(result.observations.creep_denies, 0);
    assert_eq!(result.observations.progress_reasons, 0);
}

#[test]
fn prolonged_stalls_remain_repayable_and_repayment_never_exceeds_actual_debt() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 10001);
    reward.take_interval().unwrap();
    tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
    assert_eq!(reward.state().stagnation_ticks, 2697);

    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 3);
    reward.take_interval().unwrap();
    tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
    let result = reward.take_interval().unwrap();

    assert_eq!(reward.state().stagnation_ticks, 0);
    assert_eq!(result.observations.stagnation_repaid_ticks, 2);
    assert!(!reward.state().stagnation_base_charged);
}

#[test]
fn own_activity_is_relative_to_the_selected_seat_not_hardcoded_slot_zero() {
    let mut view = snapshot(1);
    view.viewer = Some(Team::Dire);
    view.players[0].gold = None;
    view.players[0].stash = None;
    view.players[1].gold = Some(100);
    view.players[1].stash = Some(vec![None; 6]);
    let mut reward = Map2Reward::new(SlotId(1), &match_info()).unwrap();
    advance(&mut reward, &view, &[]);

    tick(&mut reward, &mut view, &[damage(2, 1, 1), purchase(1)]);
    let result = reward.take_interval().unwrap();

    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_HERO_DAMAGE | crate::MAP2_PROGRESS_PURCHASE
    );
    assert_eq!(reward.state().activity_ticks_left, 29);
}

#[test]
fn own_creep_deny_is_activity_without_an_instant_deny_reward() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    view.units.retain(|unit| unit.id != id(5));
    let deny = EventKind::Died {
        unit: id(5),
        killer: Some(id(1)),
        denied: true,
        gold: 0,
    };

    tick(&mut reward, &mut view, &[deny]);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.creep_denies, 1);
    assert_eq!(result.observations.creep_kills, 0);
    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_CREEP_DENY
    );
    assert_eq!(result.gold, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn own_hits_on_enemy_tower_barracks_and_ancient_count_as_activity() {
    for kind in [UnitKind::Tower, UnitKind::Barracks, UnitKind::Ancient] {
        let mut view = snapshot(1);
        view.units
            .iter_mut()
            .find(|unit| unit.id == id(4))
            .unwrap()
            .kind = kind;
        let mut reward = observer(&view);
        view.units
            .iter_mut()
            .find(|unit| unit.id == id(4))
            .unwrap()
            .hp -= 10;

        tick(&mut reward, &mut view, &[damage(1, 4, 10)]);
        let result = reward.take_interval().unwrap();

        assert_eq!(result.observations.structure_damage_dealt, 10);
        assert_eq!(
            result.observations.progress_reasons,
            crate::MAP2_PROGRESS_STRUCTURE_DAMAGE
        );
        assert_eq!(result.hero_damage, 0.0);
        assert_eq!(result.total, result.tower_health);
    }
}

#[test]
fn npc_structure_hits_fountain_hits_and_creep_scratches_are_not_own_objective_activity() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);

    tick(
        &mut reward,
        &mut view,
        &[damage(5, 4, 10), damage(1, 8, 10), damage(1, 6, 10)],
    );
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.structure_damage_dealt, 0);
    assert_eq!(result.observations.progress_reasons, 0);
    assert_eq!(reward.state().stagnation_ticks, 1);
}

#[test]
fn full_own_fountain_aura_counts_directly_without_actual_regeneration_or_radius_inference() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    own_hero(&mut view).effects = vec![aura(Some(1))];

    tick(&mut reward, &mut view, &[]);
    let result = reward.take_interval().unwrap();

    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_FOUNTAIN_AURA
    );
    assert_eq!(reward.state().activity_ticks_left, 29);
    let hero = own_hero(&mut view);
    assert_eq!(hero.hp, hero.max_hp);
    assert_eq!(hero.mana, hero.max_mana);
}

#[test]
fn expired_absent_or_enemy_fountain_effects_do_not_count_as_own_aura() {
    for expiry in [Some(0), None] {
        let mut view = snapshot(1);
        let mut reward = observer(&view);
        own_hero(&mut view).effects = vec![aura(expiry)];
        view.units
            .iter_mut()
            .find(|unit| unit.id == id(2))
            .unwrap()
            .effects = vec![aura(Some(30))];

        tick(&mut reward, &mut view, &[]);

        assert_eq!(
            reward
                .take_interval()
                .unwrap()
                .observations
                .progress_reasons,
            0
        );
        assert_eq!(reward.state().stagnation_ticks, 1);
    }
}

#[test]
fn ordinary_regeneration_outside_aura_and_empty_casts_are_not_activity() {
    let mut view = snapshot(1);
    own_hero(&mut view).hp -= 100;
    let mut reward = observer(&view);
    own_hero(&mut view).hp += 1;
    own_hero(&mut view).mana -= 75;
    let cast = EventKind::AbilityCast {
        caster: id(1),
        ability: bota_proto::AbilityId(13),
    };

    tick(&mut reward, &mut view, &[cast]);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.progress_reasons, 0);
    assert_eq!(reward.state().stagnation_ticks, 1);
    assert!(result.mana_spent < 0.0);
}

#[test]
fn enemy_progress_and_unattributed_fog_events_do_not_grant_a_lease() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    view.players[1].xp += 10;
    view.units.retain(|unit| unit.id != id(5));
    let events = [
        damage(2, 1, 10),
        damage(99, 2, 10),
        damage(1, 99, 10),
        death(5, 2, 40),
        death(98, 1, 40),
        purchase(1),
    ];

    tick(&mut reward, &mut view, &events);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.progress_reasons, 0);
    assert_eq!(reward.state().stagnation_ticks, 1);
    assert_eq!(reward.state().activity_ticks_left, 0);
}

#[test]
fn unattended_wave_pressure_keeps_old_reward_but_only_near_participation_grants_activity() {
    for (distance, expected) in [(1500, crate::MAP2_PROGRESS_WAVE_PRESSURE), (1501, 0)] {
        let mut view = snapshot(1);
        own_hero(&mut view).pos = Vec2::from_ints(550 + distance, 0);
        let mut reward = observer(&view);
        for unit in &mut view.units {
            if unit.kind == UnitKind::CreepMelee {
                unit.pos.x += Fixed::from_int(100);
            }
        }

        tick(&mut reward, &mut view, &[]);
        let result = reward.take_interval().unwrap();

        assert!(result.lane_pressure > 0.0);
        assert_eq!(result.observations.progress_reasons, expected);
        assert_eq!(reward.state().stagnation_ticks, u32::from(expected == 0));
    }
}

#[test]
fn only_positive_pre_wave_center_progress_grants_movement_activity() {
    let mut info = match_info();
    info.pregame_ticks = 900;
    let mut view = snapshot(898);
    own_hero(&mut view).pos = Vec2::ZERO;
    let mut reward = Map2Reward::new(SlotId(0), &info).unwrap();
    advance(&mut reward, &view, &[]);
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);
    tick(&mut reward, &mut view, &[]);
    let moved = reward.take_interval().unwrap();
    assert_eq!(
        moved.observations.progress_reasons,
        crate::MAP2_PROGRESS_PREGAME_MOVEMENT
    );
    own_hero(&mut view).pos = Vec2::ZERO;

    tick(&mut reward, &mut view, &[]);
    let after_wave = reward.take_interval().unwrap();

    assert_eq!(after_wave.observations.progress_reasons, 0);
    assert_eq!(after_wave.pregame_movement, 0.0);
    assert_eq!(reward.state().activity_ticks_left, 28);
}

#[test]
fn purchase_refunds_only_v2_wait_and_partially_repays_generic_debt() {
    let (mut reward, mut view) = stalled();
    own_hero(&mut view).pos = Vec2::from_ints(-2000, 0);
    tick(&mut reward, &mut view, &[]);
    let charged_at = view.tick + 30;
    feed_to(&mut reward, &mut view, charged_at);
    let charged = reward.take_interval().unwrap();
    assert_close(charged.fountain_wait, -0.0001);

    tick(&mut reward, &mut view, &[purchase(0)]);
    let bought = reward.take_interval().unwrap();

    assert_close(bought.fountain_wait_refund, 0.0001);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
    assert_eq!(reward.state().stagnation_ticks, 2697);
    assert!(reward.state().stagnation_base_charged);
    assert_eq!(bought.stagnation_base, 0.0);
    assert_eq!(bought.stagnation_ticks_cost, 0.0);
    assert_eq!(
        bought.observations.progress_reasons,
        crate::MAP2_PROGRESS_PURCHASE
    );
}

#[test]
fn full_aura_activity_and_v2_full_wait_penalty_operate_independently() {
    let mut view = snapshot(1);
    own_hero(&mut view).pos = Vec2::from_ints(-2000, 0);
    own_hero(&mut view).effects = vec![aura(Some(30))];
    let mut reward = observer(&view);

    feed_to(&mut reward, &mut view, 31);
    let result = reward.take_interval().unwrap();

    assert_close(result.fountain_wait, -0.0001);
    assert_eq!(result.stagnation_base, 0.0);
    assert_eq!(reward.state().stagnation_ticks, 0);
    assert_eq!(result.observations.stagnation_active_ticks, 30);
}

#[test]
fn more_than_sixty_four_hits_and_multiple_reasons_still_grant_only_one_lease() {
    let (mut reward, mut view) = stalled();
    let mut events = vec![damage(1, 2, 1); 100];
    events.push(purchase(0));

    tick(&mut reward, &mut view, &events);
    let result = reward.take_interval().unwrap();

    assert_eq!(result.observations.hero_damage_dealt, 100);
    assert_eq!(
        result.observations.progress_reasons,
        crate::MAP2_PROGRESS_HERO_DAMAGE | crate::MAP2_PROGRESS_PURCHASE
    );
    assert_eq!(reward.state().stagnation_ticks, 2697);
    assert_eq!(reward.state().activity_ticks_left, 29);
    assert_eq!(result.observations.stagnation_repaid_ticks, 3);
}

#[test]
fn interval_retention_and_clone_do_not_turn_one_old_event_into_continuous_activity() {
    let (mut sparse, mut view) = stalled();
    let mut dense = sparse.clone();
    let mut dense_cost = 0.0;
    for offset in 0..100 {
        let events = if offset == 0 {
            vec![damage(1, 2, 1)]
        } else {
            vec![]
        };
        tick(&mut sparse, &mut view, &events);
        let interval = advance(&mut dense, &view, &events);
        dense_cost += interval.stagnation_base + interval.stagnation_ticks_cost;
    }

    let interval = sparse.take_interval().unwrap();

    assert_eq!(sparse.state(), dense.state());
    assert_eq!(sparse.state().stagnation_ticks, 2680);
    assert_eq!(interval.observations.stagnation_active_ticks, 30);
    assert_close(
        dense_cost,
        interval.stagnation_base + interval.stagnation_ticks_cost,
    );
}

#[test]
fn invalid_activity_batch_cannot_partially_repay_or_change_the_base_latch() {
    let (mut reward, mut view) = stalled();
    view.tick += 1;
    reward.observe_snapshot(&view).unwrap();
    let before = reward.state();

    let error = reward
        .observe_events(view.tick, &[purchase(0), damage(1, 2, -1)])
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: damage amount outside 0..=1000000"
    );
    assert_eq!(reward.state(), before);
    reward.observe_events(view.tick, &[]).unwrap();
    let result = reward.take_interval().unwrap();
    assert_close(result.stagnation_ticks_cost, -0.000002);
    assert_eq!(result.observations.progress_reasons, 0);
}

#[test]
fn first_complete_pair_and_finish_never_add_progress_time_or_refund_stagnation() {
    let mut view = snapshot(1000);
    own_hero(&mut view).effects = vec![aura(Some(30))];
    let mut reward = observer(&view);
    assert_eq!(reward.state().stagnation_ticks, 0);
    assert_eq!(reward.state().activity_ticks_left, 0);
    let result = reward.finish(Map2RewardEnd::Draw).unwrap();
    assert_eq!(result.observations.stagnation_active_ticks, 0);
    assert_eq!(result.observations.stagnation_idle_ticks, 0);
    let (mut stalled, _) = stalled();

    let end = stalled.finish(Map2RewardEnd::TimeCap).unwrap();

    assert_eq!(end.stagnation_base, 0.0);
    assert_eq!(end.stagnation_ticks_cost, 0.0);
    assert!(stalled.state().stagnation_base_charged);
    assert_eq!(
        stalled.finish(Map2RewardEnd::Draw).unwrap_err().to_string(),
        "Map2 reward: episode already ended"
    );
}

#[test]
fn a_whole_native_episode_without_activity_has_one_base_and_bounded_unclipped_tick_cost() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, crate::MAP2_TICK_CAP);

    let result = reward.finish(Map2RewardEnd::TimeCap).unwrap();

    assert_close(result.stagnation_base, -0.02);
    assert_close(
        result.stagnation_ticks_cost,
        -0.000002 * f64::from(crate::MAP2_TICK_CAP - 1 - 2700),
    );
    assert_eq!(result.observations.stagnation_base_charges, 1);
    assert_eq!(reward.state().stagnation_ticks, 2700);
    assert!(result.total >= -crate::MAP2_REWARD_DENSE_BOUND);
    assert_close(crate::MAP2_REWARD_STAGNATION_BOUND, 0.2158);
    assert_close(crate::MAP2_REWARD_DENSE_BOUND, 0.7138);
    assert_close(crate::MAP2_REWARD_POSITIVE_BOUND, 0.255);
}

#[test]
fn fastest_full_repayments_and_rearms_stay_within_the_derived_base_count_bound() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    let mut base_count = 0;
    let mut total = 0.0;
    for _ in 1..crate::MAP2_TICK_CAP {
        let clearing = reward.state().stagnation_base_charged;
        let events = if clearing {
            vec![damage(1, 2, 1)]
        } else {
            vec![]
        };
        tick(&mut reward, &mut view, &events);
        let interval = reward.take_interval().unwrap();
        base_count += interval.observations.stagnation_base_charges;
        total += interval.total;
    }
    total += reward.finish(Map2RewardEnd::Draw).unwrap().total;

    assert!(base_count > 1);
    assert!(base_count <= u64::from(crate::MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES));
    assert!(total >= -crate::MAP2_REWARD_DENSE_BOUND);
    assert!(total <= crate::MAP2_REWARD_POSITIVE_BOUND);
}

#[test]
fn full_episode_asymmetric_bounds_keep_stalled_adverse_wins_above_favorable_nonwins() {
    let adverse_win = full_episode(false, Map2RewardEnd::Win);
    assert!(adverse_win >= 1.0 - crate::MAP2_REWARD_DENSE_BOUND);
    for end in [
        Map2RewardEnd::Draw,
        Map2RewardEnd::TimeCap,
        Map2RewardEnd::Loss,
    ] {
        let favorable = full_episode(true, end);
        let terminal = if end == Map2RewardEnd::Loss {
            -1.0
        } else {
            0.0
        };
        assert!(favorable - terminal <= crate::MAP2_REWARD_POSITIVE_BOUND);
        assert!(
            adverse_win > favorable,
            "win={adverse_win} other={favorable}"
        );
    }
    assert_close(
        crate::MAP2_REWARD_DENSE_BOUND + crate::MAP2_REWARD_POSITIVE_BOUND,
        0.9688,
    );
}

fn full_episode(favorable: bool, end: Map2RewardEnd) -> f64 {
    let mut view = snapshot(1);
    own_hero(&mut view).mana = 1_000_000;
    own_hero(&mut view).max_mana = 1_000_000;
    let mut reward = observer(&view);
    let victim = if favorable { 6 } else { 5 };
    view.units.retain(|unit| unit.id != id(victim));
    view.players[usize::from(!favorable)].xp = 1_000_000_000;
    let tower = if favorable { 4 } else { 3 };
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(tower))
        .unwrap()
        .hp = 0;
    let events = if favorable {
        own_hero(&mut view).effects = vec![aura(Some(30))];
        vec![damage(1, 2, 1_000_000), death(6, 1, 1_000_000)]
    } else {
        own_hero(&mut view).mana = 0;
        vec![
            damage(2, 1, 1_000_000),
            damage(6, 1, 1_000_000),
            damage(4, 1, 1_000_000),
            death(5, 2, 1_000_000),
        ]
    };
    tick(&mut reward, &mut view, &events);
    let mut total = reward.take_interval().unwrap().total;
    feed_to(&mut reward, &mut view, crate::MAP2_TICK_CAP);
    let terminal = reward.finish(end).unwrap();
    if !favorable {
        assert!(terminal.stagnation_base < 0.0);
    }
    total += terminal.total;
    assert!(total.is_finite());
    total
}

fn observer(view: &WorldView) -> Map2Reward {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let baseline = advance(&mut reward, view, &[]);
    assert_eq!(baseline.ticks, 0);
    reward
}

fn stalled() -> (Map2Reward, WorldView) {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 2701);
    reward.take_interval().unwrap();
    assert_eq!(reward.state().stagnation_ticks, 2700);
    (reward, view)
}

fn feed_to(reward: &mut Map2Reward, view: &mut WorldView, target: u32) {
    assert!(target >= view.tick);
    assert!(target <= crate::MAP2_TICK_CAP);
    for _ in view.tick..target {
        tick(reward, view, &[]);
    }
}

fn tick(reward: &mut Map2Reward, view: &mut WorldView, events: &[EventKind]) {
    view.tick += 1;
    reward.observe_snapshot(view).unwrap();
    reward.observe_events(view.tick, events).unwrap();
}

fn aura(ticks_left: Option<u32>) -> EffectView {
    EffectView {
        id: EffectId(3),
        ticks_left,
        stacks: None,
    }
}

fn purchase(slot: u8) -> EventKind {
    EventKind::ItemBought {
        slot: SlotId(slot),
        item: ItemId(0),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual={actual}, expected={expected}"
    );
}
