#![allow(
    clippy::float_arithmetic,
    reason = "Reward tests compare normalized components"
)]

use bota_proto::{
    Angle, Attributes, DamageKind, EntityId, EventKind, Fixed, HeroId, MapId, MatchInfo, Pick,
    PlayerView, SlotId, StatusFlags, Team, TickMode, UnitKind, UnitView, Vec2, WorldView,
};

use crate::{Map2Reward, Map2RewardBreakdown, Map2RewardEnd};

#[test]
fn empty_raze_mana_spend_is_worse_than_continue_without_a_cast_request() {
    let mut miss = initialized();
    let mut wait = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana -= 75;

    let miss = advance(&mut miss, &view, &[]);
    let wait = advance(&mut wait, &snapshot(2), &[]);

    assert!(miss.total < wait.total);
    assert_eq!(miss.observations.mana_spent, 75);
    assert_eq!(wait.total, 0.0);
}

#[test]
fn useful_hero_hit_offsets_raze_mana_cost_and_outscores_an_empty_raze() {
    let mut hit = initialized();
    let mut miss = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana -= 75;
    let damage = damage(1, 2, 67);

    let hit = advance(&mut hit, &view, &[damage]);
    let miss = advance(&mut miss, &view, &[]);

    assert!(hit.total > 0.0);
    assert!(hit.total > miss.total);
    assert_eq!(hit.observations.hero_damage_dealt, 67);
}

#[test]
fn creep_scratch_has_no_damage_credit_but_a_last_hit_earns_gold_once() {
    let mut scratch = initialized();
    let mut kill = initialized();
    let mut killed_view = snapshot(2);
    killed_view.units.retain(|unit| unit.id != id(6));

    let scratch = advance(&mut scratch, &snapshot(2), &[damage(1, 6, 20)]);
    let kill = advance(&mut kill, &killed_view, &[death(6, 1, 40)]);

    assert_eq!(scratch.total, 0.0);
    assert!(kill.gold > 0.0);
    assert_eq!(kill.observations.lane_last_hits, 1);
    assert_eq!(kill.observations.own_gold_earned, 40);
    assert_eq!(kill.total, kill.gold);
}

#[test]
fn last_hit_gold_and_xp_can_offset_meaningful_mana_spend() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana -= 75;
    view.players[0].xp += 60;
    view.units.retain(|unit| unit.id != id(6));

    let result = advance(&mut reward, &view, &[death(6, 1, 40)]);

    assert!(result.total > 0.0);
    assert!(result.mana_spent < 0.0);
}

#[test]
fn lethal_finish_outscores_running_despite_modest_creep_damage_taken() {
    let mut finish = initialized();
    let mut run = initialized();
    let mut view = snapshot(2);
    view.players[1].unit = None;
    view.units.retain(|unit| unit.id != id(2));
    own_hero(&mut view).mana -= 75;
    let events = [damage(1, 2, 100), damage(6, 1, 30), death(2, 1, 200)];
    finish.observe_snapshot(&view).unwrap();
    finish.observe_events(2, &events).unwrap();

    let result = finish.finish(Map2RewardEnd::Win).unwrap();
    let run = advance(&mut run, &snapshot(2), &[]);

    assert!(result.total > run.total);
    assert!(result.total > 0.5);
    assert!(result.creep_damage_taken < 0.0);
}

#[test]
fn purchasing_a_consumable_is_not_negative_earned_gold() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[0].gold = Some(50);
    let purchase = EventKind::ItemBought {
        slot: SlotId(0),
        item: bota_proto::ItemId(42),
    };

    let result = advance(&mut reward, &view, &[purchase]);

    assert_eq!(result.gold, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn passive_gold_and_item_sale_cash_do_not_earn_bounty_reward() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[0].gold = Some(1000);

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.observations.own_gold_earned, 0);
    assert_eq!(result.gold, 0.0);
}

#[test]
fn experience_rewards_public_gain_advantage_not_absolute_starting_experience() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.players[0].xp += 60;
    view.players[1].xp += 20;

    let result = advance(&mut reward, &view, &[]);

    assert!(result.experience > 0.0);
    assert_eq!(result.observations.own_xp_gained, 60);
    assert_eq!(result.observations.enemy_xp_gained, 20);
}

#[test]
fn own_received_damage_is_separated_by_hero_creep_and_unknown_source() {
    let mut reward = initialized();
    let events = [damage(2, 1, 40), damage(6, 1, 20), damage(99, 1, 10)];

    let result = advance(&mut reward, &snapshot(2), &events);

    assert!(result.hero_damage_taken < 0.0);
    assert!(result.creep_damage_taken < 0.0);
    assert!(result.other_damage_taken < 0.0);
    assert_eq!(result.observations.hero_damage_taken, 40);
    assert_eq!(result.observations.creep_damage_taken, 20);
    assert_eq!(result.observations.unattributed_damage_taken, 10);
}

#[test]
fn equal_hero_trade_is_not_discouraged_by_a_dominant_received_damage_penalty() {
    let mut reward = initialized();

    let result = advance(
        &mut reward,
        &snapshot(2),
        &[damage(1, 2, 100), damage(2, 1, 100)],
    );

    assert!(result.hero_damage > result.hero_damage_taken.abs());
    assert!(result.total > 0.0);
}

#[test]
fn healing_and_mana_restoration_never_refund_spent_cost_budgets() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana = 325;
    let spent = advance(&mut reward, &view, &[damage(2, 1, 100)]);
    let before = reward.state();
    view.tick = 3;
    own_hero(&mut view).mana = 400;
    let healed = EventKind::Healed {
        source: None,
        target: id(1),
        amount: 100,
        mana: 0,
    };

    let restored = advance(&mut reward, &view, &[healed]);
    let after = reward.state();

    assert!(spent.total < 0.0);
    assert_eq!(restored.total, 0.0);
    assert_eq!(before.remaining, after.remaining);
}

#[test]
fn mana_capacity_reduction_and_respawn_are_not_mana_expenditure() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).mana = 300;
    own_hero(&mut view).max_mana = 300;
    let capacity = advance(&mut reward, &view, &[]);
    view.tick = 3;
    view.players[0].unit = Some(EntityId {
        idx: 1,
        generation: 2,
    });
    own_hero(&mut view).id.generation = 2;
    own_hero(&mut view).mana = 100;

    let respawn = advance(&mut reward, &view, &[]);

    assert_eq!(capacity.observations.mana_spent, 0);
    assert_eq!(respawn.observations.mana_spent, 0);
}

#[test]
fn partially_filled_mana_capacity_changes_do_not_charge_preserved_fraction_as_spend() {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    let mut view = snapshot(1);
    own_hero(&mut view).mana = 200;
    advance(&mut reward, &view, &[]);
    view.tick = 2;
    own_hero(&mut view).mana = 150;
    own_hero(&mut view).max_mana = 300;
    let shrink = advance(&mut reward, &view, &[]);
    view.tick = 3;
    own_hero(&mut view).mana = 200;
    own_hero(&mut view).max_mana = 400;

    let expand = advance(&mut reward, &view, &[]);

    assert_eq!(shrink.mana_spent, 0.0);
    assert_eq!(expand.mana_spent, 0.0);
}

#[test]
fn moving_the_hero_without_moving_creeps_has_no_lane_pressure_reward() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    own_hero(&mut view).pos = Vec2::from_ints(900, 0);

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.lane_pressure, 0.0);
    assert_eq!(result.total, 0.0);
}

#[test]
fn advancing_visible_creeps_gains_small_pressure_and_a_return_loop_is_not_profitable() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    for unit in &mut view.units {
        if matches!(unit.kind, UnitKind::CreepMelee) {
            unit.pos.x += Fixed::from_int(100);
        }
    }
    let forward = advance(&mut reward, &view, &[]);

    let backward = advance(&mut reward, &snapshot(3), &[]);

    assert!(forward.lane_pressure > 0.0);
    assert!(forward.lane_pressure < 0.01);
    assert!((forward.total + backward.total).abs() < 1.0e-12);
}

#[test]
fn disappearing_creeps_hold_pressure_as_unknown_instead_of_earning_fog_progress() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(6));

    let result = advance(&mut reward, &view, &[]);

    assert_eq!(result.lane_pressure, 0.0);
    assert!(!reward.state().lane_observed);
}

#[test]
fn enemy_tower_health_loss_is_positive_and_healing_reverses_the_potential() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units
        .iter_mut()
        .find(|unit| unit.id == id(4))
        .unwrap()
        .hp -= 200;
    let damaged = advance(&mut reward, &view, &[]);

    let healed = advance(&mut reward, &snapshot(3), &[]);

    assert!(damaged.tower_health > 0.0);
    assert!((damaged.total + healed.total).abs() < 1.0e-12);
}

#[test]
fn missing_tower_is_not_destroyed_until_a_visible_destruction_event() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(3));
    let unknown = advance(&mut reward, &view, &[]);
    view.tick = 3;
    let event = EventKind::StructureDestroyed {
        unit: id(3),
        team: Team::Radiant,
    };

    let destroyed = advance(&mut reward, &view, &[event]);

    assert_eq!(unknown.tower_health, 0.0);
    assert!(destroyed.tower_health < 0.0);
}

#[test]
fn terminal_win_and_loss_dominate_opposing_tower_progress() {
    let mut win = initialized();
    let mut loss = initialized();
    let mut won_view = snapshot(2);
    won_view
        .units
        .iter_mut()
        .find(|unit| unit.id == id(3))
        .unwrap()
        .hp = 1;
    win.observe_snapshot(&won_view).unwrap();
    win.observe_events(2, &[damage(2, 1, 1_000_000)]).unwrap();
    loss.observe_snapshot(&snapshot(2)).unwrap();
    loss.observe_events(2, &[damage(1, 2, 1_000_000)]).unwrap();

    let won = win.finish(Map2RewardEnd::Win).unwrap();
    let lost = loss.finish(Map2RewardEnd::Loss).unwrap();

    assert!(won.total > 0.5);
    assert!(lost.total < -0.5);
}

#[test]
fn opponent_bounty_is_negative_without_own_last_hit_credit() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(5));

    let result = advance(&mut reward, &view, &[death(5, 2, 40)]);

    assert!(result.gold < 0.0);
    assert_eq!(result.observations.enemy_gold_earned, 40);
    assert_eq!(result.observations.lane_last_hits, 0);
}

#[test]
fn an_independent_deny_does_not_pay_gold_or_a_last_hit() {
    let mut reward = initialized();
    let mut view = snapshot(2);
    view.units.retain(|unit| unit.id != id(5));
    let deny = EventKind::Died {
        unit: id(5),
        killer: Some(id(1)),
        denied: true,
        gold: 0,
    };

    let result = advance(&mut reward, &view, &[deny]);

    assert_eq!(result.gold, 0.0);
    assert_eq!(result.observations.lane_last_hits, 0);
    assert_eq!(result.observations.duplicate_deaths, 0);
}

#[test]
fn zero_paid_gold_and_npc_killers_do_not_count_as_own_credited_last_hits() {
    for (killer, gold) in [(1, 0), (5, 0), (5, 40)] {
        let mut reward = initialized();
        let mut view = snapshot(2);
        view.units.retain(|unit| unit.id != id(6));

        let result = advance(&mut reward, &view, &[death(6, killer, gold)]);

        assert_eq!(result.gold, 0.0, "killer={killer}, gold={gold}");
        assert_eq!(
            result.observations.lane_last_hits, 0,
            "killer={killer}, gold={gold}"
        );
    }
}

pub(super) fn initialized() -> Map2Reward {
    let mut reward = Map2Reward::new(SlotId(0), &match_info()).unwrap();
    reward.observe_snapshot(&snapshot(1)).unwrap();
    reward.observe_events(1, &[]).unwrap();
    assert_eq!(reward.take_interval().unwrap().total, 0.0);
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
    MatchInfo {
        match_id: 1,
        map: MapId(2),
        tick_rate: 30,
        pregame_ticks: 0,
        trees: Vec::new(),
        terrain_cells: 32,
        terrain_rle: vec![(1024, 0x80)],
        opaque_cells: Vec::new(),
        mode: TickMode::Lockstep,
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: HeroId(2),
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: HeroId(2),
            },
        ],
        shop: Vec::new(),
    }
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
            unit(7, UnitKind::Fountain, Team::Radiant, 0),
            unit(8, UnitKind::Fountain, Team::Dire, 1000),
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
    UnitView {
        id: id(index),
        kind,
        team,
        pos: Vec2::from_ints(x, 0),
        facing: Angle { brads: 0 },
        hp: 1000,
        max_hp: 1000,
        mana: if hero { 400 } else { 0 },
        max_mana: if hero { 400 } else { 0 },
        move_speed: Fixed::from_int(300),
        attack_damage: 50,
        attack_range: Fixed::from_int(500),
        attack_interval: 30,
        attack_speed: 100,
        armor: Fixed::ZERO,
        magic_resist: Fixed::ZERO,
        radius: Fixed::from_int(24),
        vision_radius: Fixed::from_int(1800),
        true_sight_radius: Fixed::ZERO,
        statuses: StatusFlags { bits: 0 },
        attributes: Attributes::ZERO,
        primary: None,
        hero: hero.then_some(HeroId(2)),
        owner: hero.then_some(SlotId(if team == Team::Radiant { 0 } else { 1 })),
        level: u8::from(hero),
        abilities: Vec::new(),
        items: Vec::new(),
        effects: Vec::new(),
    }
}
