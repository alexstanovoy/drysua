use super::*;

#[test]
fn rebalance_opening_ignores_enemy_neutral_and_dead_creeps_and_has_no_early_clamped_deadline() {
    let (mut reward, mut view) = opening_fixture(Team::Radiant, 9, false, 3000);
    view.tick = 10;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(6))
        .unwrap()
        .pos = Vec2::from_ints(9216, 9216);
    view.units
        .push(unit(9, UnitKind::CreepNeutral, Team::Neutral, 9216));
    view.units.last_mut().unwrap().pos.y = Fixed::from_int(9216);
    set_wave(&mut view, Team::Radiant, true);
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(5))
        .unwrap()
        .statuses
        .bits |= StatusFlags::DEAD;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, 0.0);
    assert!(reward.state().opening_position_pending);
    view.tick += 1;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(5))
        .unwrap()
        .statuses
        .bits &= !StatusFlags::DEAD;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, -0.1);
    let mut info = match_info();
    info.pregame_ticks = crate::MAP2_TICK_CAP;
    let mut reward = Map2Reward::new(SlotId(0), &info).unwrap();
    advance(&mut reward, &snapshot(crate::MAP2_TICK_CAP - 1), &[]);
    assert_eq!(
        advance(&mut reward, &snapshot(crate::MAP2_TICK_CAP), &[]).opening_position,
        0.0
    );
    assert_eq!(
        reward.finish(Map2RewardEnd::Draw).unwrap().opening_position,
        0.0
    );
}

#[test]
fn rebalance_full_cost_raw_boundary_is_exact_and_zero_cost_marks_resolution() {
    for raw_offset in [-1, 0, 1] {
        let (mut reward, mut view) = opening_fixture(Team::Radiant, 9, false, 3000);
        own_hero(&mut view).pos.x.raw += raw_offset;
        view.tick = 10;
        set_wave(&mut view, Team::Radiant, true);
        let cost = advance(&mut reward, &view, &[]).opening_position;
        if raw_offset < 0 {
            assert!(cost > -0.1);
        } else {
            assert_eq!(cost, -0.1);
        }
        assert!(!reward.state().opening_position_pending);
    }
}

#[test]
fn rebalance_new_features_append_without_overwriting_old_budget_and_potential_indices() {
    use crate::global_feature as global;
    assert_eq!(crate::GLOBAL_FEATURES, 92);
    assert_eq!(global::MAP2_REWARD_REMAINING_START, 73);
    assert_eq!(global::MAP2_TOWER_POTENTIAL, 82);
    assert_eq!(global::MAP2_LANE_POTENTIAL, 83);
    assert_eq!(global::MAP2_STAGNATION_BASE_CHARGED, 89);
    assert_eq!(global::MAP2_TOWER_DAMAGE_REMAINING, 90);
    assert_eq!(global::MAP2_OPENING_POSITION_PENDING, 91);
    let mut tracker = crate::StateTracker::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    view.units[6].pos = Vec2::ZERO;
    view.units[7].pos = Vec2::from_ints(1000, 0);
    tracker.observe_snapshot(&view).unwrap();
    tracker.observe_events(1, &[]).unwrap();
    view.tick = 2;
    tracker.observe_snapshot(&view).unwrap();
    tracker.observe_events(2, &[damage(4, 1, 100)]).unwrap();
    let frame = super::super::feature::encode(&tracker, &crate::LocalPolicyState::new(0));
    assert_eq!(&frame.global()[73..82], &[1.0; 9]);
    assert_eq!(
        frame.global()[82],
        tracker.map2_reward_state().unwrap().tower_potential
    );
    assert_eq!(
        frame.global()[83],
        tracker.map2_reward_state().unwrap().lane_potential
    );
    assert_eq!(
        frame.global()[90],
        tracker.map2_reward_state().unwrap().remaining[9]
    );
    assert!(frame.global()[90] < 1.0);
    assert_eq!(frame.global()[91], 1.0);
    tracker.finish_map2_reward(Map2RewardEnd::Draw).unwrap();
    assert!(
        !tracker
            .map2_reward_state()
            .unwrap()
            .opening_position_pending
    );
}

#[test]
fn rebalance_raw_fixed_radius_edges_dead_flags_clone_and_complete_pair_are_preserved() {
    for raw_offset in [-1, 0, 1] {
        let (mut reward, mut view) = opening_fixture(Team::Radiant, 9, false, 1500);
        own_hero(&mut view).pos.x.raw += raw_offset;
        view.tick = 10;
        set_wave(&mut view, Team::Radiant, true);
        reward.observe_snapshot(&view).unwrap();
        assert!(reward.state().opening_position_pending);
        reward.observe_events(10, &[]).unwrap();
        let result = reward.take_interval().unwrap();
        if raw_offset <= 0 {
            assert_eq!(result.opening_position, 0.0);
        } else {
            assert!(result.opening_position < 0.0);
            assert!(result.opening_position > -1.0e-8);
        }
    }
    let (mut reward, mut view) = opening_fixture(Team::Radiant, 9, false, 1500);
    let mut copy = reward.clone();
    view.tick = 10;
    set_wave(&mut view, Team::Radiant, true);
    own_hero(&mut view).statuses.bits |= StatusFlags::DEAD;
    let first = advance(&mut reward, &view, &[]);
    assert_eq!(first.opening_position, -0.1);
    assert_eq!(advance(&mut copy, &view, &[]), first);
    view.tick = 11;
    let new_id = EntityId {
        idx: 1,
        generation: 2,
    };
    own_hero(&mut view).id = new_id;
    own_hero(&mut view).statuses.bits &= !StatusFlags::DEAD;
    view.players[0].unit = Some(new_id);
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, 0.0);
}

#[test]
fn rebalance_general_bounds_do_not_claim_win_dominance_for_nonzero_primed_potentials() {
    close(crate::MAP2_REWARD_EVENT_POSITIVE_BOUND, 0.14);
    close(crate::MAP2_REWARD_EVENT_NEGATIVE_BOUND, 0.335);
    close(crate::MAP2_REWARD_NATIVE_POSITIVE_BOUND, 0.445);
    close(crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND, 1.0488);
    close(crate::MAP2_REWARD_POSITIVE_BOUND, 0.845);
    close(crate::MAP2_REWARD_DENSE_BOUND, 1.4488);
    close(
        0.2 - crate::MAP2_REWARD_NATIVE_POSITIVE_BOUND - crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND,
        -1.2938,
    );
    let win = primed_episode(true, Map2RewardEnd::Win);
    let draw = primed_episode(false, Map2RewardEnd::Draw);
    assert!(
        draw > win,
        "general primed starts have no unconditional dominance claim"
    );
    assert!(win - 0.2 >= -crate::MAP2_REWARD_DENSE_BOUND);
    assert!(draw <= crate::MAP2_REWARD_POSITIVE_BOUND);
}

fn primed_episode(adverse: bool, end: Map2RewardEnd) -> f64 {
    let mut view = snapshot(1000);
    own_hero(&mut view).max_mana = 1_000_000;
    own_hero(&mut view).mana = 1_000_000;
    view.units[if adverse { 3 } else { 2 }].hp = 1;
    for unit in &mut view.units {
        if matches!(unit.kind, UnitKind::CreepMelee) {
            unit.pos.x = Fixed::from_int(if adverse { 3000 } else { -2000 });
        }
    }
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    advance(&mut reward, &view, &[]);
    assert!(reward.state().tower_potential.abs() > 0.29);
    assert!(reward.state().lane_potential.abs() > 0.09);
    view.tick += 1;
    view.units[2].hp = if adverse { 1 } else { 1000 };
    view.units[3].hp = if adverse { 1000 } else { 1 };
    let events = if adverse {
        own_hero(&mut view).mana = 0;
        view.players[1].xp = 1_000_000_000;
        view.units.retain(|unit| unit.id != id(5));
        vec![
            damage(2, 1, 1_000_000),
            damage(6, 1, 1_000_000),
            damage(4, 1, 1_000_000),
            damage(8, 1, 1_000_000),
            death(5, 2, 1_000_000),
        ]
    } else {
        view.players[0].xp = 1_000_000_000;
        view.units.retain(|unit| unit.id != id(6));
        vec![damage(1, 2, 1_000_000), death(6, 1, 1_000_000)]
    };
    advance(&mut reward, &view, &events).total + reward.finish(end).unwrap().total
}

#[test]
fn rebalance_received_hero_creep_and_exclusive_tower_costs_have_exact_scales() {
    let mut hero = initialized();
    let mut creep = initialized();
    let hero_first = advance(&mut hero, &snapshot(2), &[damage(2, 1, 100)]);
    let creep_first = advance(&mut creep, &snapshot(2), &[damage(6, 1, 100)]);
    close(hero_first.hero_damage_taken, -0.0029411764705882353);
    close(creep_first.creep_damage_taken, -0.0058823529411764705);
    close(
        creep_first.creep_damage_taken,
        2.0 * hero_first.hero_damage_taken,
    );
    let hero_second = advance(&mut hero, &snapshot(3), &[damage(2, 1, 100)]);
    let creep_second = advance(&mut creep, &snapshot(3), &[damage(6, 1, 100)]);
    close(
        creep_second.creep_damage_taken,
        2.0 * hero_second.hero_damage_taken,
    );
    for source in [3, 4] {
        let mut tower = initialized();
        let result = advance(&mut tower, &snapshot(2), &[damage(source, 1, 100)]);
        close(result.tower_damage_taken, -0.016666666666666666);
        assert_eq!(result.other_damage_taken, 0.0);
        assert_eq!(result.observations.tower_damage_taken, 100);
        assert_eq!(result.observations.other_damage_taken, 0);
        assert_eq!(tower.state().remaining[7], 1.0);
        assert!(tower.state().remaining[9] < 1.0);
    }
}

#[test]
fn rebalance_neutral_creeps_share_creep_budget_and_unknown_damage_keeps_other_budget() {
    let mut view = snapshot(1);
    view.units
        .push(unit(9, UnitKind::CreepNeutral, Team::Neutral, 500));
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    advance(&mut reward, &view, &[]);
    view.tick = 2;
    let neutral = advance(&mut reward, &view, &[damage(9, 1, 100)]);
    close(neutral.creep_damage_taken, -0.1 * 100.0 / 1700.0);
    view.tick = 3;
    let unknown = advance(&mut reward, &view, &[damage(999, 1, 100)]);
    close(unknown.other_damage_taken, -0.005 * 100.0 / 600.0);
    assert_eq!(unknown.tower_damage_taken, 0.0);
    assert_eq!(reward.state().remaining[9], 1.0);
    assert_eq!(unknown.observations.unattributed_damage_taken, 100);
}

#[test]
fn rebalance_enemy_economy_is_point_zero_two_while_own_budgets_remain_point_zero_three() {
    let mut own = initialized();
    let mut enemy = initialized();
    let mut own_view = snapshot(2);
    own_view.players[0].xp = 300;
    own_view.units.retain(|unit| unit.id != id(6));
    let positive = advance(&mut own, &own_view, &[death(6, 1, 100)]);
    let mut enemy_view = snapshot(2);
    enemy_view.players[1].xp = 300;
    enemy_view.units.retain(|unit| unit.id != id(5));
    let negative = advance(&mut enemy, &enemy_view, &[death(5, 2, 100)]);
    close(positive.gold, 0.03 * 100.0 / 400.0);
    close(negative.gold, -0.02 * 100.0 / 400.0);
    close(positive.experience, 0.03 * 300.0 / 3300.0);
    close(negative.experience, -0.02 * 300.0 / 3300.0);
}

#[test]
fn rebalance_opening_radius_is_exact_once_for_both_teams_and_zero_also_resolves() {
    for team in [Team::Radiant, Team::Dire] {
        for (distance, expected) in [(1500, 0.0), (2250, -0.05), (3000, -0.1), (6000, -0.1)] {
            let (mut reward, mut view) = opening_fixture(team, 9, false, distance);
            assert!(reward.state().opening_position_pending);
            view.tick = 10;
            set_wave(&mut view, team, true);
            let result = advance(&mut reward, &view, &[]);
            close(result.opening_position, expected);
            assert_eq!(result.observations.opening_position_checks, 1);
            assert!(!reward.state().opening_position_pending);
            view.tick += 1;
            own_hero(&mut view).pos = Vec2::ZERO;
            let later = advance(
                &mut reward,
                &view,
                &[EventKind::ItemBought {
                    slot: SlotId(0),
                    item: bota_proto::ItemId(42),
                }],
            );
            assert_eq!(later.opening_position, 0.0);
            assert_eq!(later.observations.opening_position_checks, 0);
        }
    }
}

#[test]
fn rebalance_opening_uses_first_wave_epoch_fallback_and_free_resolved_baselines() {
    let (mut reward, mut view) = opening_fixture(Team::Radiant, 8, true, 3000);
    view.tick = 9;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, 0.0);
    view.tick = 10;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, -0.1);
    let (mut reward, mut view) = opening_fixture(Team::Radiant, 908, false, 3000);
    view.tick = 909;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, 0.0);
    view.tick = 910;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, -0.1);
    for (tick, wave) in [(10, true), (910, false), (911, false)] {
        let (mut reward, mut view) = opening_fixture(Team::Radiant, tick, wave, 3000);
        assert!(!reward.state().opening_position_pending);
        view.tick += 1;
        assert_eq!(advance(&mut reward, &view, &[]).opening_position, 0.0);
    }
}

#[test]
fn rebalance_opening_missing_body_costs_full_once_and_finish_does_not_charge_late() {
    let (mut reward, mut view) = opening_fixture(Team::Radiant, 9, false, 1500);
    view.tick = 10;
    set_wave(&mut view, Team::Radiant, true);
    view.units.retain(|unit| unit.id != id(1));
    view.players[0].unit = None;
    assert_eq!(advance(&mut reward, &view, &[]).opening_position, -0.1);
    let mut clone = reward.clone();
    assert_eq!(clone.take_interval().unwrap().opening_position, 0.0);
    assert!(!clone.state().opening_position_pending);
    let (mut early, _) = opening_fixture(Team::Radiant, 1, false, 6000);
    let terminal = early.finish(Map2RewardEnd::Draw).unwrap();
    assert_eq!(terminal.opening_position, 0.0);
    assert_eq!(terminal.terminal, 0.0);
    assert!(!early.state().opening_position_pending);
    let (mut task, _) = opening_fixture(Team::Radiant, 1, false, 6000);
    assert_eq!(task.finish(Map2RewardEnd::TimeCap).unwrap().terminal, -0.2);
}

fn opening_fixture(team: Team, tick: u32, wave: bool, distance: i32) -> (Map2Reward, WorldView) {
    let mut info = match_info();
    info.pregame_ticks = 10;
    let mut view = snapshot(tick);
    if team == Team::Dire {
        for pick in &mut info.picks {
            pick.team = opposite(pick.team);
        }
        for player in &mut view.players {
            player.team = opposite(player.team);
        }
        for unit in &mut view.units {
            unit.team = opposite(unit.team);
        }
        view.viewer = Some(team);
    }
    own_hero(&mut view).pos = Vec2::from_ints(9216 + distance, 9216);
    set_wave(&mut view, team, wave);
    let mut reward = Map2Reward::new(SlotId(0), &info).unwrap();
    let baseline = advance(&mut reward, &view, &[]);
    assert_eq!(baseline.opening_position, 0.0);
    (reward, view)
}

fn set_wave(view: &mut WorldView, team: Team, near: bool) {
    let creep = view
        .units
        .iter_mut()
        .find(|unit| unit.kind == UnitKind::CreepMelee && unit.team == team)
        .unwrap();
    creep.pos = Vec2::from_ints(if near { 9216 + 1500 } else { 9216 + 1501 }, 9216);
}

fn opposite(team: Team) -> Team {
    match team {
        Team::Radiant => Team::Dire,
        Team::Dire => Team::Radiant,
        Team::Neutral => Team::Neutral,
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}"
    );
}
