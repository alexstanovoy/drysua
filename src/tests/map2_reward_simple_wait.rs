use bota_proto::{EntityId, EventKind, Fixed, ItemId, SlotId, Team, UnitKind, Vec2, WorldView};

use super::{advance, damage, id, match_info, own_hero, snapshot};
use crate::{Map2Reward, Map2RewardEnd};

#[test]
fn reward_schema_preserves_simple_wait_refund_semantics_with_progress_debt() {
    assert_eq!(crate::MAP2_REWARD_SCHEMA_VERSION, 6);
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.starts_with("drysua-map2-reward/v6;"));
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.contains("fountain_wait="));
    assert!(crate::MAP2_REWARD_SCHEMA_DESCRIPTOR.contains("fountain_purchase="));
    assert_ne!(crate::MAP2_REWARD_SCHEMA_HASH, 798_798_703_797_057_220);
}

#[test]
fn fountain_wait_charges_at_elapsed_30_then_prorates_every_tick() {
    for (elapsed, expected, charged_ticks) in [
        (29, 0.0, 0),
        (30, 0.0001, 1),
        (31, 0.0001 + 0.00005 / 30.0, 2),
        (60, 0.00015, 31),
    ] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);

        feed_to(&mut reward, &mut view, 1 + elapsed);
        let result = reward.take_interval().unwrap();

        assert_close(result.fountain_wait, -expected);
        assert_eq!(reward.state().fountain_wait_ticks, elapsed);
        assert_eq!(
            result.observations.fountain_wait_charged_ticks,
            charged_ticks
        );
        assert_eq!(result.fountain_wait_refund, 0.0);
    }
}

#[test]
fn first_complete_full_snapshot_is_a_free_baseline_at_any_tick() {
    let view = wait_view(1000);

    let reward = observer(&view, 900);

    assert_eq!(reward.state().fountain_wait_ticks, 0);
    assert_eq!(reward.state().fountain_wait_refundable_cost, 0.0);
}

#[test]
fn becoming_full_midstream_starts_a_new_zero_elapsed_grace() {
    let mut view = wait_view(1);
    own_hero(&mut view).hp -= 1;
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 10);
    view.tick = 11;
    own_hero(&mut view).hp += 1;

    advance(&mut reward, &view, &[]);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
    feed_to(&mut reward, &mut view, 40);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
    feed_to(&mut reward, &mut view, 41);

    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
}

#[test]
fn both_full_hp_and_full_mana_are_required_without_near_full_tolerance() {
    for health in [true, false] {
        let mut view = wait_view(1);
        if health {
            own_hero(&mut view).hp -= 1;
        } else {
            own_hero(&mut view).mana -= 1;
        }
        let mut reward = observer(&view, 0);

        feed_to(&mut reward, &mut view, 100);

        assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
        assert_eq!(reward.state().fountain_wait_ticks, 0);
    }
}

#[test]
fn a_full_stationary_channel_has_no_special_wait_exception() {
    let mut view = wait_view(1);
    own_hero(&mut view).statuses.bits |= bota_proto::StatusFlags::CHANNELLING;
    let mut reward = observer(&view, 0);

    feed_to(&mut reward, &mut view, 31);

    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
    assert_eq!(reward.state().fountain_wait_ticks, 30);
}

#[test]
fn a_purchase_after_movement_cannot_refund_the_closed_old_period() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    own_hero(&mut view).pos.x.raw += 1;
    advance(&mut reward, &view, &[]);
    view.tick = 33;

    let purchase = advance(&mut reward, &view, &[purchase(0, 0)]);

    assert_eq!(purchase.fountain_wait_refund, 0.0);
    assert_eq!(purchase.observations.fountain_wait_refunds, 0);
}

#[test]
fn any_actual_movement_resets_the_wait_and_returning_gets_a_new_grace() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
    view.tick = 32;
    own_hero(&mut view).pos.x.raw += 1;

    let moved = advance(&mut reward, &view, &[]);

    assert_eq!(reward.state().fountain_wait_ticks, 0);
    assert_eq!(moved.fountain_wait_refund, 0.0);
    view.tick = 33;
    own_hero(&mut view).pos.x.raw -= 1;
    advance(&mut reward, &view, &[]);
    feed_to(&mut reward, &mut view, 62);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
    feed_to(&mut reward, &mut view, 63);
    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
}

#[test]
fn spending_mana_on_an_empty_raze_is_allowed_to_reset_the_wait_timer() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 36);
    reward.take_interval().unwrap();
    view.tick = 37;
    own_hero(&mut view).mana -= 75;
    let cast = EventKind::AbilityCast {
        caster: id(1),
        ability: bota_proto::AbilityId(13),
    };

    let spent = advance(&mut reward, &view, &[cast]);

    assert_eq!(reward.state().fountain_wait_ticks, 0);
    assert_eq!(reward.state().fountain_wait_refundable_cost, 0.0);
    assert_eq!(spent.fountain_wait_refund, 0.0);
    assert!(spent.mana_spent < 0.0);
    view.tick = 38;
    own_hero(&mut view).mana += 75;
    advance(&mut reward, &view, &[]);
    feed_to(&mut reward, &mut view, 67);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
}

#[test]
fn health_loss_closes_the_refund_period_without_refunding_its_charge() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    own_hero(&mut view).hp -= 1;
    advance(&mut reward, &view, &[]);
    view.tick = 33;

    let unrelated_purchase = advance(&mut reward, &view, &[purchase(0, 0)]);

    assert_eq!(unrelated_purchase.fountain_wait_refund, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
}

#[test]
fn any_confirmed_own_purchase_refunds_the_entire_open_period_after_interval_drain() {
    for item in [0, 1, 42] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);
        feed_to(&mut reward, &mut view, 61);
        let charged = reward.take_interval().unwrap();
        view.tick = 62;

        let bought = advance(&mut reward, &view, &[purchase(0, item)]);

        assert_close(bought.fountain_wait_refund, 0.00015);
        assert_close(charged.fountain_wait + bought.fountain_wait_refund, 0.0);
        assert_eq!(bought.fountain_wait, 0.0);
        assert_eq!(bought.observations.fountain_wait_refunds, 1);
        assert_eq!(reward.state().fountain_wait_ticks, 0);
        assert_eq!(reward.state().fountain_wait_refundable_cost, 0.0);
    }
}

#[test]
fn cheap_already_affordable_purchase_cancels_wait_without_price_or_saving_logic() {
    let mut view = wait_view(1);
    view.players[0].gold = Some(1000);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    view.players[0].gold = Some(999);

    let result = advance(&mut reward, &view, &[purchase(0, 0)]);

    assert_close(result.fountain_wait_refund, 0.0001);
    assert_eq!(result.gold, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
}

#[test]
fn own_purchase_at_the_grace_boundary_cancels_the_tick_before_any_charge() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 30);
    reward.take_interval().unwrap();
    view.tick = 31;

    let result = advance(&mut reward, &view, &[purchase(0, 0)]);

    assert_eq!(result.fountain_wait, 0.0);
    assert_eq!(result.fountain_wait_refund, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
}

#[test]
fn purchase_refunds_before_simultaneous_movement_exit_and_resource_condition_break() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    own_hero(&mut view).pos = Vec2::from_ints(1201, 0);
    own_hero(&mut view).mana -= 1;

    let result = advance(&mut reward, &view, &[purchase(0, 0), purchase(0, 1)]);

    assert_close(result.fountain_wait_refund, 0.0001);
    assert_eq!(result.observations.fountain_wait_refunds, 1);
    assert_eq!(result.fountain_wait, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
}

#[test]
fn enemy_purchase_neither_refunds_nor_restarts_the_wait() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 30);
    reward.take_interval().unwrap();
    view.tick = 31;

    let result = advance(&mut reward, &view, &[purchase(1, 0)]);

    assert_close(result.fountain_wait, -0.0001);
    assert_eq!(result.fountain_wait_refund, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 30);
}

#[test]
fn invalid_batch_cannot_partially_refund_a_purchase_before_its_error() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    reward.observe_snapshot(&view).unwrap();
    let before = reward.state();

    let error = reward
        .observe_events(32, &[purchase(0, 0), damage(1, 2, -1)])
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Map2 reward: damage amount outside 0..=1000000"
    );
    assert_eq!(reward.state(), before);
    reward.observe_events(32, &[purchase(0, 0)]).unwrap();
    assert_close(reward.take_interval().unwrap().fountain_wait_refund, 0.0001);
}

#[test]
fn wait_only_uses_own_observed_fountain_and_includes_the_1200_boundary() {
    for (distance, expected) in [(1200, -0.0001), (1201, 0.0)] {
        let mut view = wait_view(1);
        own_hero(&mut view).pos = Vec2::from_ints(distance, 0);
        let mut reward = observer(&view, 0);

        feed_to(&mut reward, &mut view, 31);

        assert_close(reward.take_interval().unwrap().fountain_wait, expected);
    }
    let mut view = wait_view(1);
    view.units.retain(|unit| unit.id != id(7));
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
}

#[test]
fn death_or_new_body_resets_wait_without_a_refund() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    view.players[0].unit = None;
    view.units.retain(|unit| unit.id != id(1));

    let died = advance(&mut reward, &view, &[]);

    assert_eq!(reward.state().fountain_wait_ticks, 0);
    assert_eq!(died.fountain_wait_refund, 0.0);
    let mut replaced = wait_view(33);
    replace_body(&mut replaced);
    advance(&mut reward, &replaced, &[]);
    feed_to(&mut reward, &mut replaced, 62);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
}

#[test]
fn a_new_full_body_at_the_same_position_still_gets_a_new_grace() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    view.tick = 32;
    replace_body(&mut view);

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.fountain_wait_refund, 0.0);
    assert_eq!(reward.state().fountain_wait_ticks, 0);
}

#[test]
fn finishing_preserves_charges_without_an_extra_charge_or_refund() {
    for drain in [false, true] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);
        feed_to(&mut reward, &mut view, 31);
        let prior = if drain {
            reward.take_interval().unwrap().fountain_wait
        } else {
            0.0
        };

        let result = reward.finish(Map2RewardEnd::Draw).unwrap();

        assert_close(prior + result.fountain_wait, -0.0001);
        assert_eq!(result.fountain_wait_refund, 0.0);
        assert_eq!(
            reward.finish(Map2RewardEnd::Draw).unwrap_err().to_string(),
            "Map2 reward: episode already ended"
        );
    }
}

#[test]
fn cloned_wait_ledger_and_partitioned_intervals_preserve_refund_credit() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    reward.take_interval().unwrap();
    let mut cloned = reward.clone();
    view.tick = 32;

    let left = advance(&mut reward, &view, &[purchase(0, 0)]);
    let right = advance(&mut cloned, &view, &[purchase(0, 0)]);

    assert_eq!(left, right);
    assert_eq!(reward.state(), cloned.state());
}

#[test]
fn prewave_center_approach_rewards_closer_penalizes_backtrack_and_not_standing() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 900);
    view.tick = 2;
    own_hero(&mut view).pos = Vec2::from_ints(4608, 4608);

    let closer = advance(&mut reward, &view, &[]);
    view.tick = 3;
    let standing = advance(&mut reward, &view, &[]);
    view.tick = 4;
    own_hero(&mut view).pos = Vec2::ZERO;
    let backward = advance(&mut reward, &view, &[]);

    assert_close(closer.pregame_movement, 0.0025);
    assert_eq!(standing.pregame_movement, 0.0);
    assert_close(backward.pregame_movement, -0.0025);
}

#[test]
fn center_hint_stops_at_900_without_canceling_899_credit_at_cutoff_or_terminal() {
    let mut view = wait_view(898);
    let mut reward = observer(&view, 900);
    view.tick = 899;
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);
    let before = advance(&mut reward, &view, &[]);
    view.tick = 900;
    own_hero(&mut view).pos = Vec2::ZERO;
    let cutoff = advance(&mut reward, &view, &[]);
    view.tick = 901;
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);
    let after = advance(&mut reward, &view, &[]);

    let terminal = reward.finish(Map2RewardEnd::Draw).unwrap();

    assert_close(before.pregame_movement, 0.005);
    assert_eq!(cutoff.pregame_movement, 0.0);
    assert_eq!(after.pregame_movement, 0.0);
    assert_eq!(terminal.pregame_movement, 0.0);
}

#[test]
fn public_pregame_cutoff_controls_hint_and_not_an_implicit_900_constant() {
    let mut view = wait_view(8);
    let mut reward = observer(&view, 10);
    view.tick = 9;
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);
    assert_close(advance(&mut reward, &view, &[]).pregame_movement, 0.005);
    view.tick = 10;
    own_hero(&mut view).pos = Vec2::ZERO;

    assert_eq!(advance(&mut reward, &view, &[]).pregame_movement, 0.0);
}

#[test]
fn existing_creep_position_potential_still_operates_after_first_wave() {
    let mut view = wait_view(900);
    let mut reward = observer(&view, 900);
    view.tick = 901;
    for unit in &mut view.units {
        if unit.kind == UnitKind::CreepMelee {
            unit.pos.x += Fixed::from_int(100);
        }
    }
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.pregame_movement, 0.0);
    assert!(result.lane_pressure > 0.0);
}

#[test]
fn missing_hero_emits_no_center_movement_and_reappearance_uses_its_observed_position() {
    let mut view = wait_view(1);
    own_hero(&mut view).pos = Vec2::from_ints(9216, 9216);
    let mut reward = observer(&view, 900);
    view.tick = 2;
    view.players[0].unit = None;
    view.units.retain(|unit| unit.id != id(1));
    assert_eq!(advance(&mut reward, &view, &[]).pregame_movement, 0.0);
    let mut returned = wait_view(3);
    replace_body(&mut returned);
    let reappearance = advance(&mut reward, &returned, &[]);
    returned.tick = 4;
    own_hero(&mut returned).pos = Vec2::from_ints(9216, 9216);

    let approach = advance(&mut reward, &returned, &[]);

    assert_close(reappearance.pregame_movement, -0.005);
    assert_close(
        approach.pregame_movement + reappearance.pregame_movement,
        0.0,
    );
}

#[test]
fn sides_have_identical_center_and_wait_semantics_from_reflected_public_inputs() {
    let mut view = wait_view(1);
    let mut reflected = reflect(view.clone());
    let mut left = observer(&view, 900);
    let mut info = match_info();
    info.pregame_ticks = 900;
    info.picks[0].team = Team::Dire;
    info.picks[1].team = Team::Radiant;
    let mut right = Map2Reward::new(SlotId(0), &info).unwrap();
    advance(&mut right, &reflected, &[]);
    feed_to(&mut left, &mut view, 31);
    feed_to(&mut right, &mut reflected, 31);
    assert_eq!(
        left.take_interval().unwrap(),
        right.take_interval().unwrap()
    );
    view.tick = 32;
    own_hero(&mut view).pos = Vec2::from_ints(4608, 4608);

    let moved_left = advance(&mut left, &view, &[]);
    let moved_right = advance(&mut right, &reflect(view), &[]);

    assert_eq!(moved_left, moved_right);
    assert_eq!(left.state(), right.state());
}

#[test]
fn standing_and_many_allowed_resets_remain_bounded_without_wait_clipping() {
    for reset in [false, true] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);
        let mut cost = 0.0;
        for tick in 2..=crate::MAP2_TICK_CAP {
            view.tick = tick;
            if reset {
                own_hero(&mut view).pos.x = Fixed::from_int(((tick / 32) % 2) as i32);
            }
            cost += advance(&mut reward, &view, &[]).fountain_wait;
        }

        assert!(cost.is_finite());
        assert!(cost < 0.0);
        assert!(-cost <= 0.0001 * f64::from(crate::MAP2_TICK_CAP) / 30.0 + 1.0e-12);
        const { assert!(crate::MAP2_REWARD_V2_DENSE_BOUND < 0.5) };
        assert_close(crate::MAP2_REWARD_V2_DENSE_BOUND, 0.498);
    }
}

#[test]
fn public_metadata_and_native_reward_tick_cap_are_validated() {
    for terrain in [0, 513] {
        let mut info = match_info();
        info.terrain_cells = terrain;
        assert_eq!(
            Map2Reward::new(SlotId(0), &info).unwrap_err().to_string(),
            "Map2 reward: terrain axis outside 1..=512"
        );
    }
    let mut info = match_info();
    info.pregame_ticks = crate::MAP2_TICK_CAP + 1;
    assert_eq!(
        Map2Reward::new(SlotId(0), &info).unwrap_err().to_string(),
        "Map2 reward: pregame ticks exceed native Map2 cap"
    );
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    assert_eq!(
        reward
            .observe_snapshot(&wait_view(crate::MAP2_TICK_CAP + 1))
            .unwrap_err()
            .to_string(),
        "Map2 reward: Snapshot tick outside 1..=27900"
    );
}

fn observer(view: &WorldView, pregame: u32) -> Map2Reward {
    let mut info = match_info();
    info.pregame_ticks = pregame;
    info.shop = [0, 1, 42]
        .map(|item| bota_proto::ShopEntry {
            id: ItemId(item),
            cost: 1,
            components: Vec::new(),
        })
        .to_vec();
    let mut reward = Map2Reward::new(SlotId(0), &info).unwrap();
    let baseline = advance(&mut reward, view, &[]);
    assert_eq!(baseline.pregame_movement, 0.0);
    assert_eq!(baseline.fountain_wait, 0.0);
    reward
}

fn wait_view(tick: u32) -> WorldView {
    let mut view = snapshot(tick);
    own_hero(&mut view).pos = Vec2::ZERO;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(7))
        .unwrap()
        .pos = Vec2::ZERO;
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(8))
        .unwrap()
        .pos = Vec2::from_ints(1000, 0);
    view
}

fn feed_to(reward: &mut Map2Reward, view: &mut WorldView, target: u32) {
    assert!(target >= view.tick);
    assert!(target <= crate::MAP2_TICK_CAP);
    for tick in view.tick + 1..=target {
        view.tick = tick;
        reward.observe_snapshot(view).unwrap();
        reward.observe_events(tick, &[]).unwrap();
    }
}

fn purchase(slot: u8, item: u16) -> EventKind {
    EventKind::ItemBought {
        slot: SlotId(slot),
        item: ItemId(item),
    }
}

fn replace_body(view: &mut WorldView) {
    let new = EntityId {
        idx: 1,
        generation: 2,
    };
    view.players[0].unit = Some(new);
    own_hero(view).id = new;
}

fn reflect(mut view: WorldView) -> WorldView {
    view.viewer = Some(Team::Dire);
    for unit in &mut view.units {
        unit.team = if unit.team == Team::Radiant {
            Team::Dire
        } else {
            Team::Radiant
        };
        unit.pos.x = Fixed::from_int(18432) - unit.pos.x;
        unit.pos.y = Fixed::from_int(18432) - unit.pos.y;
    }
    view.players[0].team = Team::Dire;
    view.players[1].team = Team::Radiant;
    view
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}"
    );
}
