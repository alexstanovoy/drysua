#![allow(
    clippy::float_arithmetic,
    reason = "Reward tests compare normalized components"
)]

#[path = "map2_reward_progress.rs"]
mod progress;
#[path = "reward_rebalance.rs"]
mod rebalance;
#[path = "map2_reward_simple_wait.rs"]
mod simple_wait;
#[path = "victory_time.rs"]
mod victory_time;

use super::fixtures;
use crate::{Map2Reward, Map2RewardBreakdown, Map2RewardEnd};
use bota_proto::{
    Attributes, DamageKind, EntityId, EventKind, Fixed, HeroId, MapId, MatchInfo, PlayerView,
    SlotId, StatusFlags, Team, UnitKind, UnitView, Vec2, WorldView,
};

#[test]
fn visible_tower_progress_reverses_without_refilling_dense_budgets() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units[3].hp -= 200;
    let progress = advance(&mut reward, &view, &[]);
    assert!(progress.tower_health > 0.0);
    let reversed = advance(&mut reward, &snapshot(3), &[]);
    assert!((progress.total + reversed.total).abs() < 1.0e-12);
    assert_eq!(reward.finish(Map2RewardEnd::Draw).unwrap().total, 0.0);
}

#[test]
fn paid_combat_retains_dense_credit_through_a_lethal_finish() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana -= 75;
    view.players[0].xp += 60;
    view.players[1].unit = None;
    view.units.retain(|unit| unit.id != id(2));
    reward.observe_snapshot(&view).unwrap();
    reward
        .observe_events(2, &[damage(1, 2, 100), damage(6, 1, 30), death(2, 1, 200)])
        .unwrap();
    let result = reward.finish(Map2RewardEnd::Win).unwrap();
    assert_eq!(result.observations.own_gold_earned, 200);
    assert_eq!(result.observations.hero_damage_dealt, 100);
    assert_eq!(result.observations.mana_spent, 75);
    assert!(result.experience > 0.0);
    assert!(result.creep_damage_taken < 0.0);
    assert!(result.mana_spent < 0.0);
    assert!(result.total > result.terminal);
}

#[test]
fn terminal_outcomes_keep_dense_costs_and_only_wins_receive_victory_time() {
    for (end, terminal, victory) in [
        (Map2RewardEnd::Win, 0.2, 0.2),
        (Map2RewardEnd::Loss, -0.2, 0.0),
        (Map2RewardEnd::Draw, 0.0, 0.0),
        (Map2RewardEnd::TimeCap, -0.2, 0.0),
    ] {
        let mut reward = initialized();
        let dense = advance(&mut reward, &snapshot(2), &[damage(4, 1, 100)]);
        let finished = reward.finish(end).unwrap();
        assert_eq!(finished.end, Some(end));
        assert_eq!(finished.terminal, terminal);
        assert_eq!(finished.victory_time, victory);
        assert_eq!(finished.total, terminal + victory);
        assert!((dense.tower_damage_taken + 0.1 * 100.0 / 600.0).abs() < 1.0e-12);
        assert_eq!(
            reward
                .observe_snapshot(&snapshot(3))
                .unwrap_err()
                .to_string(),
            "Map2 reward: episode already ended"
        );
    }
    for (actual, expected) in [
        (crate::MAP2_REWARD_NATIVE_NEGATIVE_BOUND, 1.0488),
        (crate::MAP2_REWARD_NATIVE_POSITIVE_BOUND, 0.645),
        (crate::MAP2_REWARD_DENSE_BOUND, 1.4488),
    ] {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn gold_credit_requires_paid_own_or_enemy_bounty_not_cash_changes_denies_or_npcs() {
    for (victim, killer, gold, denied, own, enemy, hits) in [
        (6, 1, 40, false, 40, 0, 1),
        (5, 2, 40, false, 0, 40, 0),
        (5, 1, 0, true, 0, 0, 0),
        (6, 1, 0, false, 0, 0, 0),
        (6, 5, 0, false, 0, 0, 0),
        (6, 5, 40, false, 0, 0, 0),
    ] {
        let mut reward = initialized();
        let mut view = snapshot(2);
        view.units.retain(|unit| unit.id != id(victim));
        let event = EventKind::Died {
            unit: id(victim),
            killer: Some(id(killer)),
            denied,
            gold,
        };
        let result = advance(&mut reward, &view, &[event]);
        assert_eq!(result.observations.own_gold_earned, own);
        assert_eq!(result.observations.enemy_gold_earned, enemy);
        assert_eq!(result.observations.lane_last_hits, hits);
        assert_eq!(result.observations.duplicate_deaths, 0);
        assert_eq!(result.gold > 0.0, own > 0);
        assert_eq!(result.gold < 0.0, enemy > 0);
    }
    for (cash, purchase) in [(50, true), (1000, false)] {
        let mut reward = initialized();
        let mut view = snapshot(2);
        view.players[0].gold = Some(cash);
        let events = if purchase {
            vec![EventKind::ItemBought {
                slot: SlotId(0),
                item: bota_proto::ItemId(42),
            }]
        } else {
            vec![]
        };
        let result = advance(&mut reward, &view, &events);
        assert_eq!(result.gold, 0.0);
        assert_eq!(result.observations.own_gold_earned, 0);
    }
}

#[test]
fn restoration_does_not_refund_costs_and_capacity_or_body_changes_are_not_spending() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana = 325;
    assert!(advance(&mut reward, &view, &[damage(2, 1, 100)]).total < 0.0);
    let before = reward.state();
    view.tick = 3;
    own_hero(&mut view).mana = 400;
    let healed = EventKind::Healed {
        source: None,
        target: id(1),
        amount: 100,
        mana: 0,
    };
    assert_eq!(advance(&mut reward, &view, &[healed]).total, 0.0);
    assert_eq!(before.remaining, reward.state().remaining);
    for initial in [200, 400] {
        let mut view = snapshot(1);
        own_hero(&mut view).mana = initial;
        let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
        advance(&mut reward, &view, &[]);
        for (tick, mana, maximum, generation) in [
            (2, initial * 3 / 4, 300, 1),
            (3, initial, 400, 1),
            (4, 100, 400, 2),
        ] {
            view.tick = tick;
            let hero = own_hero(&mut view);
            hero.mana = mana;
            hero.max_mana = maximum;
            hero.id.generation = generation;
            view.players[0].unit = Some(EntityId { idx: 1, generation });
            let result = advance(&mut reward, &view, &[]);
            assert_eq!(result.observations.mana_spent, 0);
            assert_eq!(result.mana_spent, 0.0);
        }
    }
}

pub(super) fn initialized() -> Map2Reward {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    assert_eq!(advance(&mut reward, &snapshot(1), &[]).total, 0.0);
    reward
}

pub(super) fn advance(
    reward: &mut Map2Reward,
    view: &WorldView,
    events: &[EventKind],
) -> Map2RewardBreakdown {
    reward.observe_snapshot(view).unwrap();
    reward.observe_events(view.tick, events).unwrap();
    reward.take_interval().unwrap()
}

pub(super) fn id(idx: u32) -> EntityId {
    EntityId { idx, generation: 1 }
}

pub(super) fn damage(source: u32, target: u32, amount: i32) -> EventKind {
    EventKind::Damaged {
        source: Some(id(source)),
        target: id(target),
        amount,
        kind: DamageKind::Magical,
        crit: false,
    }
}

pub(super) fn death(unit: u32, killer: u32, gold: i32) -> EventKind {
    EventKind::Died {
        unit: id(unit),
        killer: Some(id(killer)),
        denied: false,
        gold,
    }
}

pub(super) fn own_hero(view: &mut WorldView) -> &mut UnitView {
    view.units
        .iter_mut()
        .find(|unit| unit.owner == Some(SlotId(0)))
        .unwrap()
}

pub(super) fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(1, MapId(2), fixtures::two_seat_picks(Team::Radiant))
        .terrain_cells(32)
        .terrain_rle(vec![(1_024, 0x80)])
        .build()
}

pub(super) fn snapshot(tick: u32) -> WorldView {
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units: vec![
            unit(1, UnitKind::Hero, Team::Radiant, 400),
            unit(2, UnitKind::Hero, Team::Dire, 600),
            unit(3, UnitKind::Tower, Team::Radiant, 200),
            unit(4, UnitKind::Tower, Team::Dire, 800),
            unit(5, UnitKind::CreepMelee, Team::Radiant, 450),
            unit(6, UnitKind::CreepMelee, Team::Dire, 550),
            unit(7, UnitKind::Fountain, Team::Radiant, -2000),
            unit(8, UnitKind::Fountain, Team::Dire, 3000),
        ],
        projectiles: Vec::new(),
        players: vec![player(0), player(1)],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn player(slot: u8) -> PlayerView {
    PlayerView {
        slot: SlotId(slot),
        team: if slot == 0 { Team::Radiant } else { Team::Dire },
        hero: HeroId(2),
        unit: Some(id(u32::from(slot) + 1)),
        level: 1,
        xp: 0,
        gold: (slot == 0).then_some(100),
        stash: (slot == 0).then(|| vec![None; 6]),
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

pub(super) fn unit(index: u32, kind: UnitKind, team: Team, x: i32) -> UnitView {
    let hero = kind == UnitKind::Hero;
    fixtures::UnitFixture {
        id: id(index),
        kind,
        team,
        pos: Vec2::from_ints(x, 0),
        mana: if hero { 400 } else { 0 },
        attack_damage: 50,
        attack_time: 1000,
        attributes: Attributes::ZERO,
        primary: None,
        hero: hero.then_some(HeroId(2)),
        owner: hero.then_some(SlotId(if team == Team::Radiant { 0 } else { 1 })),
        level: u8::from(hero),
    }
    .build()
}
