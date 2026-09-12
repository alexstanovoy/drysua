#![allow(
    clippy::float_arithmetic,
    reason = "Bounded seat-return diagnostics and supervision."
)]
use super::*;
use crate::model::CheckedActionSet;
use crate::ppo_arena::map2_skill_bootstrap as historical;
use crate::{ActionTarget, ControlledUnit, EntityIndex, PointIndex, ShopIndex, StructuredAction};
use bota_proto::{AbilitySlot, DamageKind, Fixed, ItemId, ItemSlot, UnitKind, Vec2};
use bota_server::game::{ItemStack, Level, UnitOrder, World, rules};
use std::fmt::Write as _;

#[path = "map2_advantage_fit.rs"]
mod fit;
#[path = "map2_advantage_rank.rs"]
mod rank;
#[path = "map2_recovery_002.rs"]
mod recovery_002;
#[path = "map2_advantage_tests.rs"]
mod tests;

const HORIZON: u32 = 90;
const ROOT: &str = "artifacts/temp/map2-gameplay-fix-20260912/advantage-001";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Physics {
    seed: u64,
    side: usize,
    tick: u32,
    level: u8,
    own_hp: i32,
    enemy_hp: i32,
    mana: i32,
    reach: i32,
    facing: u16,
    creep_hp: Option<i32>,
    mango: bool,
    stash: bool,
    gold: i32,
    at_home: bool,
    deaths: u16,
}
#[derive(Clone, Debug)]
struct Prefix {
    physics: Physics,
    actions: Vec<StructuredAction>,
}

struct Game {
    arena: Arena,
    seats: Vec<ArenaSeatPolicy>,
    side: usize,
    total: Metrics,
    terminal: bool,
}
#[derive(Clone, Debug, Default)]
struct Metrics {
    score: f64,
    mana: u64,
    damage: u64,
    received: u64,
    creep_damage: u64,
    gold: u64,
    xp: u64,
    last_hits: u64,
    casts: u32,
    hit_slots: u8,
    stacks: u32,
    mango: u32,
}

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ROOT)
}
fn cast(slot: u8) -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot),
        target: ActionTarget::None,
    }
}
fn use_mango() -> StructuredAction {
    StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(0),
        target: ActionTarget::None,
    }
}

fn game(physics: Physics) -> Game {
    assert!(physics.side < 2);
    assert!(physics.level == 3 || physics.level == 4);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: physics.seed,
    })
    .unwrap();
    let configured = arena.configure_for_test(|world| configure(world, physics));
    for (messages, fresh) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    Game {
        arena,
        seats: setup_seats(start).unwrap(),
        side: physics.side,
        total: Metrics::default(),
        terminal: false,
    }
}

fn configure(world: &mut World, physics: Physics) {
    world.tick = physics.tick;
    for side in 0..2 {
        let hero = world.seats[side].unit.unwrap();
        world.seats[side].level = physics.level;
        world.seats[side].xp = rules::XP_THRESHOLDS[physics.level as usize - 1];
        world.seats[side].gold = if side == physics.side {
            physics.gold
        } else {
            0
        };
        world.seats[side].deaths = physics.deaths;
        world.level.insert(hero, Level(physics.level));
        world.statuses.remove(hero);
        world.set_order(hero, UnitOrder::Stand);
        let mut events = Vec::new();
        assert!(!world.learn(hero, 5, &mut events));
        assert!(world.learn(hero, 0, &mut events));
        assert!(world.learn(hero, 3, &mut events));
        assert!(world.learn(hero, 4, &mut events));
        if physics.level == 4 {
            assert!(world.learn(hero, 0, &mut events));
        }
        assert_eq!(world.points_spent(hero), physics.level);
        assert_eq!(world.abilities.get(hero).unwrap().slots[5].id.0, 16);
        assert_eq!(world.abilities.get(hero).unwrap().slots[5].level, 0);
        assert!(!world.learn(hero, 5, &mut events));
    }
    world.settle();
    for side in 0..2 {
        world.fill_pools(world.seats[side].unit.unwrap());
    }
    let hero = world.seats[physics.side].unit.unwrap();
    let direction = if physics.side == 0 { 1 } else { -1 };
    let position = if physics.at_home {
        world.map.fountains[physics.side] + Vec2::from_ints(900 * direction, 900 * direction)
    } else {
        Vec2::from_ints(9216, 9216)
    };
    world.transform.get_mut(hero).unwrap().pos = position;
    world.transform.get_mut(hero).unwrap().facing.brads = physics.facing;
    world.health.get_mut(hero).unwrap().hp = Fixed::from_int(physics.own_hp);
    world.mana.get_mut(hero).unwrap().mana = Fixed::from_int(physics.mana);
    configure_enemy(world, physics, position);
    let mango = ItemStack::bought(ItemId(42), world.seats[physics.side].slot, world.tick);
    if physics.mango {
        world.inventory.get_mut(hero).unwrap().slots[0] = mango;
    }
    if physics.stash {
        world.seats[physics.side].stash.slots[0] = mango;
    }
}

fn configure_enemy(world: &mut World, physics: Physics, position: Vec2) {
    if physics.enemy_hp == 0 && physics.creep_hp.is_none() {
        return;
    }
    let at = bota_server::game::point_along(
        position,
        position
            + bota_server::game::heading_of(bota_proto::Angle {
                brads: physics.facing,
            }),
        Fixed::from_int(physics.reach),
    );
    if let Some(hp) = physics.creep_hp {
        let creep = world.spawn_unit(
            &bota_server::game::MELEE_CREEP,
            world.seats[1 - physics.side].team,
            at,
        );
        world.settle();
        world.health.get_mut(creep).unwrap().hp = Fixed::from_int(hp);
    } else {
        let enemy = world.seats[1 - physics.side].unit.unwrap();
        world.transform.get_mut(enemy).unwrap().pos = at;
        world.transform.get_mut(enemy).unwrap().facing.brads = physics.facing.wrapping_add(32768);
        world.health.get_mut(enemy).unwrap().hp = Fixed::from_int(physics.enemy_hp);
    }
}

impl Game {
    fn prepare(&mut self) -> (FeatureFrame, ActionSpace) {
        prepare_neural_seat_policy_sample(&mut self.seats[self.side]).unwrap()
    }
    fn action(&mut self, action: StructuredAction) {
        assert!(!self.terminal);
        let (_, space) = self.prepare();
        let request = neural_policy_request_in_space(&mut self.seats[self.side], action, &space)
            .unwrap()
            .1;
        let enemy = teacher_request(&mut self.seats[1 - self.side]).unwrap();
        for offset in 0..3 {
            if self.terminal {
                break;
            }
            self.tick(
                if offset == 0 { request } else { None },
                if offset == 0 { enemy } else { None },
            );
        }
    }
    fn teacher_continuation(&mut self) {
        let request = teacher_request(&mut self.seats[self.side]).unwrap();
        let enemy = teacher_request(&mut self.seats[1 - self.side]).unwrap();
        for offset in 0..3 {
            if self.terminal {
                break;
            }
            self.tick(
                if offset == 0 { request } else { None },
                if offset == 0 { enemy } else { None },
            );
        }
    }
    fn tick(&mut self, own: Option<Request>, enemy: Option<Request>) {
        let before = self.seats[self.side].tracker.own_hero().cloned();
        let opposing_hero =
            self.seats[self.side].tracker.current().unwrap().players[1 - self.side].unit;
        let mut requests = [None, None];
        requests[self.side] = own;
        requests[1 - self.side] = enemy;
        let step = self.arena.step(&requests).unwrap();
        events(
            &mut self.total,
            before.as_ref(),
            opposing_hero,
            &step.messages[self.side],
        );
        for (side, (seat, messages)) in self.seats.iter_mut().zip(step.messages).enumerate() {
            let winner = observe_messages(seat, &messages).unwrap();
            let reward = if let Some(winner) = winner {
                self.terminal = true;
                seat.tracker
                    .finish_map2_reward(if winner == Team::Neutral {
                        Map2RewardEnd::Draw
                    } else if winner == seat.tracker.team() {
                        Map2RewardEnd::Win
                    } else {
                        Map2RewardEnd::Loss
                    })
                    .unwrap()
            } else {
                seat.tracker.take_map2_reward_interval().unwrap()
            };
            if side == self.side {
                self.total.add(reward);
            }
            assert_eq!(
                seat.rejections, 0,
                "counterfactual/deployment order rejection"
            );
        }
    }
}

impl Metrics {
    fn add(&mut self, reward: crate::Map2RewardBreakdown) {
        self.score += reward.total;
        let observed = reward.observations;
        self.mana += observed.mana_spent;
        self.damage += observed.hero_damage_dealt;
        self.received += observed.hero_damage_taken + observed.other_damage_taken;
        self.creep_damage += observed.creep_damage_taken;
        self.gold += observed.own_gold_earned;
        self.xp += observed.own_xp_gained;
        self.last_hits += observed.lane_last_hits + observed.neutral_last_hits;
    }
}

fn events(
    metrics: &mut Metrics,
    before: Option<&bota_proto::UnitView>,
    opposing_hero: Option<bota_proto::EntityId>,
    messages: &[ServerMsg],
) {
    let Some(before) = before else {
        return;
    };
    let view = messages
        .iter()
        .find_map(|message| {
            if let ServerMsg::Snapshot { view } = message {
                Some(view)
            } else {
                None
            }
        })
        .unwrap();
    observe_stacks(metrics, view, before.team);
    for message in messages {
        let ServerMsg::Events { events, .. } = message else {
            continue;
        };
        for event in events {
            observe_event(metrics, before, opposing_hero, view, events, event);
        }
    }
}

fn observe_stacks(metrics: &mut Metrics, view: &bota_proto::WorldView, own_team: Team) {
    for unit in &view.units {
        if unit.kind == UnitKind::Hero && unit.team != own_team {
            metrics.stacks = metrics.stacks.max(
                unit.effects
                    .iter()
                    .filter(|effect| effect.id.0 == 15)
                    .map(|effect| effect.stacks.unwrap_or(0))
                    .max()
                    .unwrap_or(0),
            );
        }
    }
}

fn observe_event(
    metrics: &mut Metrics,
    before: &bota_proto::UnitView,
    opposing_hero: Option<bota_proto::EntityId>,
    view: &bota_proto::WorldView,
    events: &[EventKind],
    event: &EventKind,
) {
    match event {
        EventKind::AbilityCast { caster, ability }
            if *caster == before.id && (13..=15).contains(&ability.0) =>
        {
            let slot = (ability.0 - 13) as usize;
            if view
                .units
                .iter()
                .find(|unit| unit.id == before.id)
                .is_some_and(|next| {
                    next.abilities[slot].cooldown_left > before.abilities[slot].cooldown_left
                })
            {
                metrics.casts += 1;
                if events.iter().any(|event| matches!(event, EventKind::Damaged { source: Some(source), target, kind: DamageKind::Magical, amount, .. } if *source == before.id && *amount > 0 && Some(*target) == opposing_hero)) { metrics.hit_slots |= 1 << slot; }
            }
        }
        EventKind::Healed {
            source: Some(source),
            target,
            mana,
            ..
        } if *source == before.id && *target == before.id && *mana > 0 => {
            let count = |unit: &bota_proto::UnitView| {
                unit.items
                    .iter()
                    .flatten()
                    .filter(|item| item.id == ItemId(42))
                    .map(|item| u32::from(item.charges.unwrap_or(0)))
                    .sum::<u32>()
            };
            if view
                .units
                .iter()
                .find(|unit| unit.id == before.id)
                .is_some_and(|next| count(next) < count(before))
            {
                metrics.mango += 1;
            }
        }
        _ => {}
    }
}

#[test]
#[ignore = "Historical instructional bug reproduction, never used as the corrected label pipeline."]
fn historical_same_frame_label_invariance_red() {
    let spec = historical::Spec {
        kind: historical::Kind::ChainThree,
        side: 0,
        variant: 1,
        seed: 10092726,
    };
    let mut game = historical::environment(spec);
    let (_, space) = prepare_neural_seat_policy_sample(&mut game.seats[0]).unwrap();
    let first = historical::script(&game, &space);
    let request = neural_policy_request_in_space(&mut game.seats[0], first, &space)
        .unwrap()
        .1;
    for tick in 0..3 {
        game.advance(if tick == 0 { request } else { None });
    }
    let (frame, space) = prepare_neural_seat_policy_sample(&mut game.seats[0]).unwrap();
    let chain = historical::script(&game, &space);
    game.spec.kind = historical::Kind::Hit;
    let hit = historical::script(&game, &space);
    assert!(space.allows(chain));
    assert!(space.allows(hit));
    assert!(frame.matches_action_space(&space));
    eprintln!(
        "tick={} same_frame=true chain_label={chain:?} hit_label={hit:?} enemy={:?}",
        space.tick(),
        game.seats[0]
            .tracker
            .current()
            .unwrap()
            .units
            .iter()
            .find(|unit| unit.kind == bota_proto::UnitKind::Hero && unit.owner == Some(SlotId(1)))
    );
    assert_eq!(
        chain, hit,
        "instructional bug: a coverage name must not change the preferred action on an identical frame"
    );
}
