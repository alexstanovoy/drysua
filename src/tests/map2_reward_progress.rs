use bota_proto::{EffectId, EffectView, EventKind, ItemId, SlotId, Vec2, WorldView};

use super::{advance, damage, death, id, match_info, own_hero, snapshot};
use crate::{Map2Reward, Map2RewardEnd};

#[test]
fn observed_progress_repays_debt_but_passive_income_and_empty_casts_do_not() {
    let mut view = snapshot(1);
    let mut reward = observer(&view);
    feed_to(&mut reward, &mut view, 101);
    view.players[0].gold = Some(1000);
    own_hero(&mut view).mana -= 75;
    tick(
        &mut reward,
        &mut view,
        &[EventKind::AbilityCast {
            caster: id(1),
            ability: bota_proto::AbilityId(13),
        }],
    );
    let idle = reward.take_interval().unwrap();
    assert_eq!(idle.observations.progress_reasons, 0);
    assert_eq!(reward.state().stagnation_ticks, 101);
    view.units.retain(|unit| unit.id != id(6));
    view.players[0].xp += 60;
    tick(
        &mut reward,
        &mut view,
        &[damage(1, 2, 10), death(6, 1, 40), purchase()],
    );
    let active = reward.take_interval().unwrap();
    assert_eq!(
        active.observations.progress_reasons,
        crate::MAP2_PROGRESS_HERO_DAMAGE
            | crate::MAP2_PROGRESS_CREEP_KILL
            | crate::MAP2_PROGRESS_GOLD
            | crate::MAP2_PROGRESS_XP
            | crate::MAP2_PROGRESS_PURCHASE
    );
    assert_eq!(active.observations.stagnation_repaid_ticks, 3);
    assert_eq!(reward.state().stagnation_ticks, 98);
}

#[test]
fn idle_threshold_and_native_cap_follow_independent_elapsed_time_arithmetic() {
    for (elapsed, base, cost, charges) in [
        (2699, 0.0, 0.0, 0),
        (2700, -0.02, 0.0, 1),
        (2701, -0.02, -0.000002, 2),
        (
            crate::MAP2_TICK_CAP - 1,
            -0.02,
            -0.000002 * f64::from(crate::MAP2_TICK_CAP - 2701),
            u64::from(crate::MAP2_TICK_CAP - 2700),
        ),
    ] {
        let mut view = snapshot(1);
        let mut reward = observer(&view);
        feed_to(&mut reward, &mut view, 1 + elapsed);
        let result = reward.finish(Map2RewardEnd::TimeCap).unwrap();
        assert_close(result.stagnation_base, base);
        assert_close(result.stagnation_ticks_cost, cost);
        assert_eq!(
            result.observations.stagnation_idle_ticks,
            u64::from(elapsed)
        );
        assert_eq!(result.observations.stagnation_charged_ticks, charges);
        assert_eq!(result.terminal, -0.2);
    }
}

#[test]
fn a_single_event_grants_thirty_ticks_repaying_only_actual_debt_before_base_charge() {
    for (idle, debt, latched) in [
        (2, 0, false),
        (2699, 2696, false),
        (2700, 2697, true),
        (10000, 2697, true),
    ] {
        let mut view = snapshot(1);
        let mut reward = observer(&view);
        feed_to(&mut reward, &mut view, 1 + idle);
        reward.take_interval().unwrap();
        tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
        let result = reward.take_interval().unwrap();
        assert_eq!(reward.state().stagnation_ticks, debt);
        assert_eq!(reward.state().activity_ticks_left, 29);
        assert_eq!(reward.state().stagnation_base_charged, latched);
        assert_eq!(
            result.observations.stagnation_repaid_ticks,
            u64::from(idle.min(3))
        );
        assert_eq!(result.stagnation_base, 0.0);
        assert_eq!(result.stagnation_ticks_cost, 0.0);
    }
    let (mut reward, mut view) = stalled();
    tick(&mut reward, &mut view, &[damage(1, 2, 1)]);
    let lease_end = view.tick + 29;
    feed_to(&mut reward, &mut view, lease_end);
    let active = reward.take_interval().unwrap();
    assert_eq!(reward.state().stagnation_ticks, 2610);
    assert_eq!(active.observations.stagnation_active_ticks, 30);
    assert_eq!(active.observations.stagnation_repaid_ticks, 90);
    let threshold = view.tick + 90;
    feed_to(&mut reward, &mut view, threshold);
    let idle = reward.take_interval().unwrap();
    assert_eq!(idle.stagnation_base, 0.0);
    assert_close(idle.stagnation_ticks_cost, -0.000002);
}

#[test]
fn only_complete_repayment_rearms_base_and_aura_is_independent_of_full_wait_cost() {
    let (mut reward, mut view) = stalled();
    own_hero(&mut view).effects = vec![aura()];
    own_hero(&mut view).pos = Vec2::from_ints(-2000, 0);
    let almost = view.tick + 899;
    feed_to(&mut reward, &mut view, almost);
    assert_eq!(reward.state().stagnation_ticks, 3);
    assert!(reward.state().stagnation_base_charged);
    tick(&mut reward, &mut view, &[]);
    let repaid = reward.take_interval().unwrap();
    assert_eq!(reward.state().stagnation_ticks, 0);
    assert!(!reward.state().stagnation_base_charged);
    assert_eq!(repaid.observations.stagnation_repaid_ticks, 2700);
    assert!(repaid.fountain_wait < 0.0);
    own_hero(&mut view).effects.clear();
    let lease_end = view.tick + 29;
    feed_to(&mut reward, &mut view, lease_end);
    reward.take_interval().unwrap();
    let threshold = view.tick + 2700;
    feed_to(&mut reward, &mut view, threshold);
    let next = reward.take_interval().unwrap();
    assert_close(next.stagnation_base, -0.02);
    assert_eq!(next.stagnation_ticks_cost, 0.0);
    assert_eq!(next.observations.stagnation_base_charges, 1);
}

#[test]
fn purchase_refund_and_invalid_batch_do_not_erase_latched_stagnation_costs() {
    let (mut reward, mut view) = stalled();
    own_hero(&mut view).pos = Vec2::from_ints(-2000, 0);
    let charged_at = view.tick + 31;
    feed_to(&mut reward, &mut view, charged_at);
    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
    view.tick += 1;
    reward.observe_snapshot(&view).unwrap();
    let before = reward.state();
    assert_eq!(
        reward
            .observe_events(view.tick, &[purchase(), damage(1, 2, -1)])
            .unwrap_err()
            .to_string(),
        "Map2 reward: damage amount outside 0..=1000000"
    );
    assert_eq!(reward.state(), before);
    reward.observe_events(view.tick, &[purchase()]).unwrap();
    let result = reward.take_interval().unwrap();
    assert_close(result.fountain_wait_refund, 0.0001);
    assert_eq!(reward.state().stagnation_ticks, 2697);
    assert!(reward.state().stagnation_base_charged);
    assert_eq!(result.stagnation_base, 0.0);
    assert_eq!(result.stagnation_ticks_cost, 0.0);
}

#[test]
fn favorable_nonwins_can_outrank_adverse_wins_without_rescaling_dense_budgets() {
    let win = full_episode(false, Map2RewardEnd::Win);
    assert!(win >= 0.2 - crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND);
    for end in [
        Map2RewardEnd::Draw,
        Map2RewardEnd::TimeCap,
        Map2RewardEnd::Loss,
    ] {
        let favorable = full_episode(true, end);
        let terminal = if end == Map2RewardEnd::Draw {
            0.0
        } else {
            -0.2
        };
        assert!(favorable - terminal <= crate::MAP2_REWARD_NATIVE_POSITIVE_BOUND);
        assert!(win < favorable, "win={win}, {end:?}={favorable}");
    }
    assert_close(
        crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND + crate::MAP2_REWARD_NATIVE_POSITIVE_BOUND,
        1.6938,
    );
    assert_close(0.2 - crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND, -0.8488);
}

fn full_episode(favorable: bool, end: Map2RewardEnd) -> f64 {
    let mut view = snapshot(1);
    own_hero(&mut view).mana = 1_000_000;
    own_hero(&mut view).max_mana = 1_000_000;
    let mut reward = observer(&view);
    view.units
        .retain(|unit| unit.id != id(if favorable { 6 } else { 5 }));
    view.players[usize::from(!favorable)].xp = 1_000_000_000;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(if favorable { 4 } else { 3 }))
        .unwrap()
        .hp = 0;
    let events = if favorable {
        own_hero(&mut view).effects = vec![aura()];
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
    let first = reward.take_interval().unwrap().total;
    feed_to(&mut reward, &mut view, crate::MAP2_TICK_CAP);
    let terminal = reward.finish(end).unwrap();
    if !favorable {
        assert!(terminal.stagnation_base < 0.0);
    }
    let total = first + terminal.total;
    assert!(total.is_finite());
    total
}

fn observer(view: &WorldView) -> Map2Reward {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    assert_eq!(advance(&mut reward, view, &[]).ticks, 0);
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

fn aura() -> EffectView {
    EffectView {
        id: EffectId(3),
        ticks_left: Some(30),
        stacks: None,
    }
}

fn purchase() -> EventKind {
    EventKind::ItemBought {
        slot: SlotId(0),
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
