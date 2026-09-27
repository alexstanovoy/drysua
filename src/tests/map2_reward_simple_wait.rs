use bota_proto::{EventKind, ItemId, SlotId, Vec2, WorldView};

use super::{advance, id, match_info, own_hero, snapshot};
use crate::{Map2Reward, Map2RewardEnd};

#[test]
fn full_stationary_wait_has_exact_grace_rate_and_no_channel_exception() {
    for channel in [false, true] {
        for (elapsed, cost, charged) in [
            (29, 0.0, 0),
            (30, 0.0001, 1),
            (31, 0.0001 + 0.00005 / 30.0, 2),
            (60, 0.00015, 31),
        ] {
            let mut view = wait_view(1);
            if channel {
                own_hero(&mut view).statuses.bits |= bota_proto::StatusFlags::CHANNELLING;
            }
            let mut reward = observer(&view, 0);
            feed_to(&mut reward, &mut view, 1 + elapsed);
            let result = reward.finish(Map2RewardEnd::Draw).unwrap();
            assert_close(result.fountain_wait, -cost);
            assert_eq!(result.observations.fountain_wait_charged_ticks, charged);
            assert_eq!(result.fountain_wait_refund, 0.0);
        }
    }
}

#[test]
fn wait_requires_exactly_full_resources_and_an_observed_own_fountain_within_1200() {
    for (distance, health, mana, fountain, cost) in [
        (1200, 0, 0, true, -0.0001),
        (1201, 0, 0, true, 0.0),
        (0, 1, 0, true, 0.0),
        (0, 0, 1, true, 0.0),
        (0, 0, 0, false, 0.0),
    ] {
        let mut view = wait_view(1);
        own_hero(&mut view).pos = Vec2::from_ints(distance, 0);
        own_hero(&mut view).hp -= health;
        own_hero(&mut view).mana -= mana;
        if !fountain {
            view.units.retain(|unit| unit.id != id(7));
        }
        let mut reward = observer(&view, 0);
        feed_to(&mut reward, &mut view, 31);
        assert_close(reward.take_interval().unwrap().fountain_wait, cost);
    }
}

#[test]
fn purchase_refunds_only_the_current_wait_once_even_when_leaving_on_purchase_tick() {
    for (elapsed, refund) in [(29, 0.0), (30, 0.0001), (60, 0.00015)] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);
        feed_to(&mut reward, &mut view, elapsed + 1);
        let charged = reward.take_interval().unwrap().fountain_wait;
        view.tick += 1;
        own_hero(&mut view).pos = Vec2::from_ints(1201, 0);
        let bought = advance(&mut reward, &view, &[purchase(), purchase()]);
        assert_close(bought.fountain_wait_refund, refund);
        assert_close(charged + bought.fountain_wait_refund, 0.0);
        view.tick += 1;
        assert_eq!(
            advance(&mut reward, &view, &[purchase()]).fountain_wait_refund,
            0.0
        );
    }
}

#[test]
fn leaving_before_purchase_closes_credit_and_returning_starts_a_new_grace() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 0);
    feed_to(&mut reward, &mut view, 31);
    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
    own_hero(&mut view).pos.x.raw += 1;
    view.tick = 32;
    advance(&mut reward, &view, &[]);
    view.tick = 33;
    assert_eq!(
        advance(&mut reward, &view, &[purchase()]).fountain_wait_refund,
        0.0
    );
    feed_to(&mut reward, &mut view, 62);
    assert_eq!(reward.take_interval().unwrap().fountain_wait, 0.0);
    feed_to(&mut reward, &mut view, 63);
    assert_close(reward.take_interval().unwrap().fountain_wait, -0.0001);
}

#[test]
fn center_approach_is_reversible_until_public_cutoff_and_not_canceled_at_finish() {
    let mut view = wait_view(1);
    let mut reward = observer(&view, 900);
    for (tick, position, expected) in [(2, 4608, 0.0025), (3, 4608, 0.0), (4, 0, -0.0025)] {
        view.tick = tick;
        own_hero(&mut view).pos = Vec2::from_ints(position, position);
        assert_close(advance(&mut reward, &view, &[]).pregame_movement, expected);
    }
    feed_to(&mut reward, &mut view, 898);
    reward.take_interval().unwrap();
    for (tick, position, expected) in [(899, 9216, 0.005), (900, 0, 0.0), (901, 9216, 0.0)] {
        view.tick = tick;
        own_hero(&mut view).pos = Vec2::from_ints(position, position);
        assert_close(advance(&mut reward, &view, &[]).pregame_movement, expected);
    }
    assert_eq!(
        reward.finish(Map2RewardEnd::Draw).unwrap().pregame_movement,
        0.0
    );
}

#[test]
fn whole_episode_wait_cost_remains_bounded_even_with_repeated_grace_resets() {
    for reset in [false, true] {
        let mut view = wait_view(1);
        let mut reward = observer(&view, 0);
        let mut cost = 0.0;
        for tick in 2..=crate::MAP2_TICK_CAP {
            view.tick = tick;
            if reset {
                own_hero(&mut view).pos.x = bota_proto::Fixed::from_int(((tick / 32) % 2) as i32);
            }
            cost += advance(&mut reward, &view, &[]).fountain_wait;
        }
        assert!(cost.is_finite());
        assert!(cost < 0.0);
        assert!(-cost <= 0.0001 * f64::from(crate::MAP2_TICK_CAP) / 30.0 + 1.0e-12);
    }
    assert_close(crate::MAP2_REWARD_V2_DENSE_BOUND, 0.498);
}

#[test]
fn invalid_public_metadata_and_ticks_fail_at_native_boundaries() {
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
    for tick in [0, crate::MAP2_TICK_CAP + 1, u32::MAX] {
        let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
        assert_eq!(
            reward
                .observe_snapshot(&wait_view(tick))
                .unwrap_err()
                .to_string(),
            "Map2 reward: Snapshot tick outside 1..=27900"
        );
    }
}

fn observer(view: &WorldView, pregame: u32) -> Map2Reward {
    let mut info = match_info();
    info.pregame_ticks = pregame;
    info.shop = vec![bota_proto::ShopEntry {
        id: ItemId(42),
        cost: 1,
        components: Vec::new(),
    }];
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

fn purchase() -> EventKind {
    EventKind::ItemBought {
        slot: SlotId(0),
        item: ItemId(42),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}"
    );
}
