#![allow(
    clippy::float_arithmetic,
    reason = "Reward tests compare potential steps"
)]

use super::fixtures;
use crate::{
    MAP2_REWARD_DEATH_WEIGHT, MAP2_REWARD_HEALTH_WEIGHT, MAP2_REWARD_TOWER_WEIGHT, Map2Reward,
    Map2RewardBreakdown, Map2RewardEnd,
};
use bota_proto::{
    Attributes, DamageKind, EntityId, EventKind, HeroId, MapId, MatchInfo, PlayerView, SlotId,
    Team, UnitKind, UnitView, Vec2, WorldView,
};

/// Health is the progress toward the next death: an unseen hero keeps its last seen HP and a
/// dead hero counts as its full next life, so a kill converts the damage lead into a death lead.
#[test]
fn fogged_hero_keeps_last_seen_health_and_a_kill_converts_it_into_a_death_lead() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units[1].hp = 250;
    let hit = advance(&mut reward, &view, &[damage(1, 2, 750)]);
    assert_close(hit.health, MAP2_REWARD_HEALTH_WEIGHT * 0.75);
    let mut fogged = snapshot(3);
    fogged.units.remove(1);
    assert_eq!(advance(&mut reward, &fogged, &[]).total, 0.0);
    let mut killed = snapshot(4);
    killed.units.remove(1);
    killed.players[1].unit = None;
    killed.players[1].deaths = 1;
    let kill = advance(&mut reward, &killed, &[damage(1, 2, 250)]);
    assert_close(kill.health, -MAP2_REWARD_HEALTH_WEIGHT * 0.75);
    assert_close(kill.deaths, MAP2_REWARD_DEATH_WEIGHT);
    assert_eq!(kill.observations.enemy_deaths, 1);
    assert_eq!(kill.observations.hero_damage_dealt, 250);
}

/// Buildings are always visible, so the weakest tower drives the potential and a vanished
/// tower counts as destroyed; the terminal step returns the whole potential.
#[test]
fn weakest_tower_drives_potential_and_finish_returns_it() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units[3].hp = 400;
    let damaged = advance(&mut reward, &view, &[]);
    assert_close(damaged.towers, MAP2_REWARD_TOWER_WEIGHT * 0.6);
    let mut destroyed = snapshot(3);
    destroyed.units.remove(3);
    let fallen = advance(&mut reward, &destroyed, &[]);
    assert_close(fallen.towers, MAP2_REWARD_TOWER_WEIGHT * 0.4);
    let finished = reward.finish(Map2RewardEnd::Win).unwrap();
    assert_close(finished.closure, -MAP2_REWARD_TOWER_WEIGHT);
    assert_eq!(finished.terminal, 1.0);
    assert_close(finished.fast_win, 0.5);
    assert_close(
        finished.total,
        1.0 + finished.fast_win - MAP2_REWARD_TOWER_WEIGHT,
    );
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "{actual} != {expected}"
    );
}

pub(super) fn initialized() -> Map2Reward {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let baseline = advance(&mut reward, &snapshot(1), &[]);
    assert_eq!(baseline, Map2RewardBreakdown::default());
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

pub(super) fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(1, MapId(2), fixtures::two_seat_picks(Team::Radiant))
        .terrain_cells(32)
        .terrain_rle(vec![(1_024, 0x80)])
        .build()
}

/// Own hero 1, enemy hero 2, own tower 3 and enemy tower 4, all at full 1000 HP.
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

fn unit(index: u32, kind: UnitKind, team: Team, x: i32) -> UnitView {
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
