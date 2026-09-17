use super::*;

const FULL_TICKS: u32 = 9_000;
const NONE_TICKS: u32 = 21_600;
const SPAN_TICKS: u32 = NONE_TICKS - FULL_TICKS;
const BONUS: f64 = 0.2;

fn win_at(tick: u32) -> Map2RewardBreakdown {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    reward.observe_snapshot(&snapshot(tick)).unwrap();
    reward.observe_events(tick, &[]).unwrap();
    reward.finish(Map2RewardEnd::Win).unwrap()
}

const POWER: u32 = 2;

fn expected_bonus(tick: u32) -> f64 {
    if tick <= FULL_TICKS {
        BONUS
    } else if tick >= NONE_TICKS {
        0.0
    } else {
        BONUS * (f64::from(NONE_TICKS - tick) / f64::from(SPAN_TICKS)).powi(POWER as i32)
    }
}

#[test]
fn victory_time_bonus_is_exact_and_continuous_across_the_whole_window() {
    for tick in [
        8_999_u32, 9_000, 9_001, 12_599, 12_600, 12_601, 15_300, 21_599, 21_600, 21_601,
    ] {
        let result = win_at(tick);
        let expected = expected_bonus(tick);
        assert_eq!(result.victory_time, expected, "tick={tick}");
        assert!((0.0..=0.2).contains(&result.victory_time), "tick={tick}");
    }
    assert_eq!(win_at(FULL_TICKS).victory_time, 0.2);
    assert_eq!(win_at(12_600).victory_time, 0.2 * (5.0 / 7.0f64).powi(2));
    assert_eq!(win_at(15_300).victory_time, 0.05);
    // 21600-18000 = 3600 ticks left, so the fraction is 2/7 and not 1/7.
    assert_eq!(win_at(18_000).victory_time, 0.2 * (2.0 / 7.0f64).powi(2));
    assert_eq!(win_at(NONE_TICKS).victory_time, 0.0);
    assert_eq!(win_at(NONE_TICKS + 1).victory_time, 0.0);
}

#[test]
fn victory_time_bonus_is_monotone_non_increasing_across_the_window() {
    let mut previous = expected_bonus(0);
    let mut tick = 1;
    while tick <= NONE_TICKS + 60 {
        let bonus = win_at(tick).victory_time;
        assert!(bonus <= previous, "tick={tick} {bonus} > {previous}");
        assert!((0.0..=0.2).contains(&bonus), "tick={tick}");
        previous = bonus;
        tick += 30;
    }
}

#[test]
fn victory_time_bonus_is_zero_for_every_non_win_outcome() {
    for end in [
        Map2RewardEnd::Loss,
        Map2RewardEnd::Draw,
        Map2RewardEnd::TimeCap,
    ] {
        let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
        reward.observe_snapshot(&snapshot(8_999)).unwrap();
        reward.observe_events(8_999, &[]).unwrap();
        let result = reward.finish(end).unwrap();
        assert_eq!(result.victory_time, 0.0, "{end:?}");
        assert_eq!(result.terminal, expected_terminal(end), "{end:?}");
    }
}

fn expected_terminal(end: Map2RewardEnd) -> f64 {
    match end {
        Map2RewardEnd::Win => 0.2,
        Map2RewardEnd::Loss | Map2RewardEnd::TimeCap => -0.2,
        Map2RewardEnd::Draw => 0.0,
    }
}

#[test]
fn victory_time_enters_the_sum_and_never_leaves_the_interval_early() {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    reward.observe_snapshot(&snapshot(9_001)).unwrap();
    reward.observe_events(9_001, &[]).unwrap();
    let interval = reward.take_interval().unwrap();
    assert_eq!(interval.victory_time, 0.0);

    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    reward.observe_snapshot(&snapshot(9_001)).unwrap();
    reward.observe_events(9_001, &[]).unwrap();
    let result = reward.finish(Map2RewardEnd::Win).unwrap();
    let sum = result.tower_damage_taken
        + result.opening_position
        + result.gold
        + result.experience
        + result.hero_damage
        + result.hero_damage_taken
        + result.creep_damage_taken
        + result.other_damage_taken
        + result.mana_spent
        + result.tower_health
        + result.lane_pressure
        + result.pregame_movement
        + result.fountain_wait
        + result.fountain_wait_refund
        + result.stagnation_base
        + result.stagnation_ticks_cost
        + result.terminal
        + result.victory_time;
    assert_eq!(result.total, sum);
    assert!(result.total.is_finite());
    assert_eq!(result.total, result.terminal + expected_bonus(9_001));
}
