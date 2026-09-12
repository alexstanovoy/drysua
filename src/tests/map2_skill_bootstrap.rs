#![allow(
    clippy::float_arithmetic,
    reason = "Bounded test-only supervised bootstrap measurements."
)]
use super::*;
use crate::{
    ActionTarget, BehavioralTarget, ControlledUnit, EntityIndex, PointIndex, ShopIndex,
    StructuredAction,
};
use bota_proto::{AbilitySlot, DamageKind, Fixed, ItemId, ItemSlot, UnitKind, Vec2};
use bota_server::game::{ItemStack, Level, UnitOrder, World, rules};

#[path = "map2_skill_bootstrap_fit.rs"]
mod fit;
#[path = "map2_skill_bootstrap_tests.rs"]
mod tests;

const KINDS: [Kind; 12] = [
    Kind::Hit,
    Kind::Empty,
    Kind::ChainThree,
    Kind::ChainKill,
    Kind::CreepLastHit,
    Kind::CreepHealthy,
    Kind::NoMana,
    Kind::MangoUse,
    Kind::MangoFull,
    Kind::Supply,
    Kind::Recovery,
    Kind::Finish,
];
const TICKS: u32 = 180;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Hit,
    Empty,
    ChainThree,
    ChainKill,
    CreepLastHit,
    CreepHealthy,
    NoMana,
    MangoUse,
    MangoFull,
    Supply,
    Recovery,
    Finish,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Spec {
    pub(super) kind: Kind,
    pub(super) side: usize,
    pub(super) variant: u32,
    pub(super) seed: u64,
}
pub(super) struct Row {
    pub(super) frame: FeatureFrame,
    pub(super) space: ActionSpace,
    pub(super) target: BehavioralTarget,
    pub(super) action: StructuredAction,
    pub(super) spec: Spec,
}
pub(super) struct Environment {
    pub(super) arena: Arena,
    pub(super) seats: Vec<ArenaSeatPolicy>,
    pub(super) spec: Spec,
    pub(super) goal: Vec2,
    pub(super) start_distance: f64,
    pub(super) result: ResultCounts,
}
#[derive(Clone, Debug, Default)]
pub(super) struct ResultCounts {
    casts: u32,
    slots: u8,
    hero_damage: u32,
    physical: u32,
    hit_slots: u8,
    max_stacks: u32,
    kills: u32,
    creep_paid: u32,
    creep_raze_paid: u32,
    used: u32,
    bought: u32,
    mana_spent: u64,
    progress: f64,
}

pub(super) fn specs(validation: bool, variants: &[u32]) -> Vec<Spec> {
    assert!(variants.len() <= 3);
    let base = if validation { 10092700 } else { 10091700 };
    KINDS
        .iter()
        .enumerate()
        .flat_map(|(index, kind)| {
            variants.iter().flat_map(move |variant| {
                (0..2).map(move |side| Spec {
                    kind: *kind,
                    side,
                    variant: *variant,
                    seed: base + index as u64 * 12 + u64::from(*variant) * 2 + side as u64,
                })
            })
        })
        .collect()
}

pub(super) fn environment(spec: Spec) -> Environment {
    assert!(spec.side < 2);
    assert!(spec.variant <= 3);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: spec.seed,
    })
    .unwrap();
    let mut goal = Vec2::from_ints(0, 0);
    let configured = arena.configure_for_test(|world| {
        configure(world, spec);
        goal = world.map.fountains[spec.side];
    });
    for (messages, fresh) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    let seats = setup_seats(start).expect("configured baseline before tracker");
    let start_distance = distance(seats[spec.side].tracker.own_hero().unwrap().pos, goal);
    Environment {
        arena,
        seats,
        spec,
        goal,
        start_distance,
        result: ResultCounts::default(),
    }
}

pub(super) fn configure(world: &mut World, spec: Spec) {
    assert_eq!(world.map.id, MapId(2));
    assert!(spec.variant < 4);
    world.tick = 1201;
    for side in 0..2 {
        let hero = world.seats[side].unit.unwrap();
        world.seats[side].gold = 0;
        world.seats[side].level = 3;
        world.seats[side].xp = rules::XP_THRESHOLDS[2];
        world.level.insert(hero, Level(3));
        world.statuses.remove(hero);
        world.set_order(hero, UnitOrder::Stand);
        for ability in &mut world.abilities.get_mut(hero).unwrap().slots[..5] {
            ability.level = 1;
        }
    }
    world.settle();
    for side in 0..2 {
        world.fill_pools(world.seats[side].unit.unwrap());
    }
    let hero = world.seats[spec.side].unit.unwrap();
    let direction = if spec.side == 0 { 1 } else { -1 };
    let shift = spec.variant as i32 * 37;
    let position = if spec.kind == Kind::Supply {
        world.map.fountains[spec.side] + Vec2::from_ints((950 + shift) * direction, 950 * direction)
    } else if matches!(spec.kind, Kind::Recovery | Kind::Finish) {
        world.map.barracks[spec.side][0].2
            + Vec2::from_ints(350 * direction, (350 + shift) * direction)
    } else {
        Vec2::from_ints(9216 + shift, 9216 + shift)
    };
    let transform = world.transform.get_mut(hero).unwrap();
    transform.pos = position;
    let angle = if spec.kind == Kind::ChainThree {
        0
    } else {
        (spec.variant as u16) * 2048
    };
    transform.facing.brads = (if spec.side == 0 { 0u16 } else { 32768u16 }).wrapping_add(angle);
    configure_resources(world, spec);
    configure_target(world, spec);
}

fn configure_resources(world: &mut World, spec: Spec) {
    let hero = world.seats[spec.side].unit.unwrap();
    assert!(world.alive(hero));
    assert_eq!(world.tick, 1201);
    if matches!(
        spec.kind,
        Kind::NoMana | Kind::MangoUse | Kind::Supply | Kind::Recovery | Kind::Finish
    ) {
        world.mana.get_mut(hero).unwrap().mana = Fixed::ZERO;
    }
    if matches!(spec.kind, Kind::Recovery | Kind::Finish) {
        world.health.get_mut(hero).unwrap().hp = Fixed::from_int(100);
    }
    if matches!(spec.kind, Kind::MangoUse | Kind::MangoFull) {
        world.inventory.get_mut(hero).unwrap().slots[0] =
            ItemStack::bought(ItemId(42), world.seats[spec.side].slot, world.tick);
    }
    if spec.kind == Kind::Supply {
        world.seats[spec.side].gold = 65;
    }
}

fn configure_target(world: &mut World, spec: Spec) {
    let hero = world.seats[spec.side].unit.unwrap();
    let from = *world.transform.get(hero).unwrap();
    let reach = match spec.kind {
        Kind::Hit => 250 + spec.variant as i32 * 100,
        Kind::ChainKill => 420,
        _ => 450,
    };
    let position = bota_server::game::point_along(
        from.pos,
        from.pos + bota_server::game::heading_of(from.facing),
        Fixed::from_int(reach),
    );
    if matches!(
        spec.kind,
        Kind::Empty | Kind::MangoUse | Kind::MangoFull | Kind::Supply
    ) {
        return;
    }
    let target = if matches!(spec.kind, Kind::CreepLastHit | Kind::CreepHealthy) {
        world.spawn_unit(
            &bota_server::game::MELEE_CREEP,
            world.seats[1 - spec.side].team,
            position,
        )
    } else {
        world.seats[1 - spec.side].unit.unwrap()
    };
    world.transform.get_mut(target).unwrap().pos = position;
    world.settle();
    if let Some(hp) = match spec.kind {
        Kind::ChainKill => Some(120 + spec.variant as i32 * 5),
        Kind::CreepLastHit => Some(30 + spec.variant as i32 * 5),
        Kind::Finish => Some(30),
        _ => None,
    } {
        world.health.get_mut(target).unwrap().hp = Fixed::from_int(hp);
    }
    assert!(world.alive(target));
}

fn cast(slot: u8) -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot),
        target: ActionTarget::None,
    }
}
fn mango() -> StructuredAction {
    StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(0),
        target: ActionTarget::None,
    }
}
pub(super) fn distance(source: Vec2, target: Vec2) -> f64 {
    f64::from((source.x - target.x).to_int()).hypot(f64::from((source.y - target.y).to_int()))
}

pub(super) fn script(environment: &Environment, space: &ActionSpace) -> StructuredAction {
    let kind = environment.spec.kind;
    let tracker = &environment.seats[environment.spec.side].tracker;
    if matches!(kind, Kind::Empty | Kind::MangoFull) {
        return StructuredAction::Continue;
    }
    if matches!(kind, Kind::MangoUse | Kind::Supply) {
        return supply_script(environment, space);
    }
    if kind == Kind::Recovery {
        if environment.arena.tick() > 1201 {
            return StructuredAction::Continue;
        }
        let point = space
            .point_candidates()
            .iter()
            .enumerate()
            .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
            .min_by_key(|(_, point)| point.position.distance_squared(environment.goal))
            .unwrap()
            .0;
        return StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point: PointIndex(point),
        };
    }
    let target = space
        .entity_candidates()
        .iter()
        .enumerate()
        .find(|(_, unit)| {
            unit.relation == crate::EntityRelation::Enemy
                && if matches!(kind, Kind::CreepLastHit | Kind::CreepHealthy) {
                    unit.kind == UnitKind::CreepMelee
                } else {
                    unit.kind == UnitKind::Hero
                }
        });
    let Some((index, target)) = target else {
        return StructuredAction::Continue;
    };
    if matches!(kind, Kind::NoMana | Kind::CreepHealthy | Kind::Finish) {
        return if environment.arena.tick() == 1201 {
            StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(index),
            }
        } else {
            StructuredAction::Continue
        };
    }
    if kind == Kind::Hit && environment.result.hero_damage > 0 {
        return StructuredAction::Continue;
    }
    let hero = tracker.own_hero().unwrap();
    for slot in 0..3 {
        let point = bota_server::game::point_along(
            hero.pos,
            hero.pos + bota_server::game::heading_of(hero.facing),
            Fixed::from_int(rules::RAZE_DISTANCE[slot as usize]),
        );
        if space.allows(cast(slot)) && target.position.within(point, Fixed::from_int(250)) {
            return cast(slot);
        }
    }
    StructuredAction::Continue
}

fn supply_script(environment: &Environment, space: &ActionSpace) -> StructuredAction {
    if space.allows(mango()) {
        return mango();
    }
    if environment.spec.kind == Kind::MangoUse {
        return StructuredAction::Continue;
    }
    if environment.result.bought == 0 {
        let item = space
            .shop_candidates()
            .iter()
            .position(|item| item.item == ItemId(42))
            .unwrap();
        return StructuredAction::Buy {
            unit: ControlledUnit::Hero,
            item: ShopIndex(item),
        };
    }
    let tracker = &environment.seats[environment.spec.side].tracker;
    if tracker
        .own_player()
        .unwrap()
        .stash
        .as_ref()
        .unwrap()
        .iter()
        .any(Option::is_some)
    {
        return StructuredAction::Cast {
            unit: ControlledUnit::Courier,
            slot: AbilitySlot(0),
            target: ActionTarget::None,
        };
    }
    StructuredAction::Continue
}

impl Environment {
    pub(super) fn advance(&mut self, request: Option<Request>) {
        let before = self.seats[self.spec.side]
            .tracker
            .current()
            .unwrap()
            .clone();
        let mut requests = [None, None];
        requests[self.spec.side] = request;
        let step = self.arena.step(&requests).expect("short native step");
        observe_counts(
            &mut self.result,
            &before,
            &step.messages[self.spec.side],
            self.spec.side,
        );
        for (index, (seat, messages)) in self.seats.iter_mut().zip(step.messages).enumerate() {
            assert!(observe_messages(seat, &messages).unwrap().is_none());
            let reward = seat.tracker.take_map2_reward_interval().unwrap();
            if index == self.spec.side {
                self.result.mana_spent += reward.observations.mana_spent;
            }
        }
        if let Some(hero) = self.seats[self.spec.side].tracker.own_hero() {
            self.result.progress = self.start_distance - distance(hero.pos, self.goal);
        }
    }
}

fn observe_counts(
    counts: &mut ResultCounts,
    before: &bota_proto::WorldView,
    messages: &[ServerMsg],
    side: usize,
) {
    let hero = before.players[side].unit.unwrap();
    let enemy = before.players[1 - side].unit;
    let after = messages
        .iter()
        .find_map(|message| {
            if let ServerMsg::Snapshot { view } = message {
                Some(view)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(after.tick, before.tick + 1);
    observe_stacks(counts, after, enemy);
    for message in messages {
        let ServerMsg::Events { events, .. } = message else {
            continue;
        };
        for event in events {
            match *event {
                EventKind::AbilityCast { caster, ability }
                    if caster == hero && (13..=15).contains(&ability.0) =>
                {
                    observe_cast(counts, before, after, events, ability.0 - 13, side);
                }
                EventKind::Damaged {
                    source: Some(source),
                    target,
                    kind,
                    amount,
                    ..
                } if source == hero && amount > 0 => {
                    if kind == DamageKind::Magical && Some(target) == enemy {
                        counts.hero_damage += amount as u32;
                    }
                    if kind == DamageKind::Physical {
                        counts.physical += amount as u32;
                    }
                }
                EventKind::Died {
                    unit,
                    killer: Some(killer),
                    gold,
                    ..
                } if killer == hero => {
                    if Some(unit) == enemy {
                        counts.kills += 1;
                    } else if gold > 0 {
                        counts.creep_paid += 1;
                        if events.iter().any(|event| matches!(event, EventKind::Damaged { source: Some(source), target, kind: DamageKind::Magical, amount, .. } if *source == hero && *target == unit && *amount > 0)) { counts.creep_raze_paid += 1; }
                    }
                }
                EventKind::Healed {
                    source: Some(source),
                    target,
                    mana,
                    ..
                } if source == hero && target == hero && mana > 0 => counts.used += 1,
                EventKind::ItemBought {
                    slot,
                    item: ItemId(42),
                } if slot == SlotId(side as u8) => counts.bought += 1,
                _ => {}
            }
        }
    }
}

fn observe_cast(
    counts: &mut ResultCounts,
    before: &bota_proto::WorldView,
    after: &bota_proto::WorldView,
    events: &[EventKind],
    slot: u16,
    side: usize,
) {
    assert!(slot < 3);
    assert!(side < 2);
    let hero = before.players[side].unit.unwrap();
    let enemy = before.players[1 - side].unit;
    let prior = before.units.iter().find(|unit| unit.id == hero).unwrap();
    let next = after.units.iter().find(|unit| unit.id == hero);
    if next.is_some_and(|next| {
        next.abilities[slot as usize].cooldown_left > prior.abilities[slot as usize].cooldown_left
    }) {
        counts.casts += 1;
        counts.slots |= 1 << slot;
        if events.iter().any(|event| matches!(event, EventKind::Damaged { source: Some(source), target, kind: DamageKind::Magical, amount, .. } if *source == hero && Some(*target) == enemy && *amount > 0)) { counts.hit_slots |= 1 << slot; }
    }
}

fn observe_stacks(
    counts: &mut ResultCounts,
    after: &bota_proto::WorldView,
    enemy: Option<bota_proto::EntityId>,
) {
    for unit in &after.units {
        if Some(unit.id) == enemy {
            for effect in &unit.effects {
                if effect.id.0 == 15 {
                    counts.max_stacks = counts.max_stacks.max(effect.stacks.unwrap_or(0));
                }
            }
        }
    }
}

pub(super) fn completed(kind: Kind, result: &ResultCounts) -> bool {
    match kind {
        Kind::Hit => result.hero_damage > 0,
        Kind::Empty | Kind::MangoFull => result.casts == 0 && result.used == 0,
        Kind::ChainThree => result.hit_slots == 7 && result.max_stacks >= 3,
        Kind::ChainKill => {
            result.kills > 0 && result.casts == 2 && result.hit_slots.count_ones() == 2
        }
        Kind::CreepLastHit => result.creep_raze_paid > 0 && result.casts > 0,
        Kind::CreepHealthy | Kind::NoMana => result.physical > 0 && result.casts == 0,
        Kind::MangoUse => result.used > 0,
        Kind::Supply => result.bought > 0 && result.used > 0,
        Kind::Recovery => result.progress > 100.0,
        Kind::Finish => result.kills > 0,
    }
}

pub(super) fn collect(spec: Spec) -> Vec<Row> {
    let mut environment = environment(spec);
    let mut rows = Vec::new();
    for decision in 0..TICKS / 3 {
        let (frame, space) =
            prepare_neural_seat_policy_sample(&mut environment.seats[spec.side]).unwrap();
        let action = script(&environment, &space);
        let target = BehavioralTarget::from_action(&frame, &space, action)
            .expect("validated fixture target");
        let request =
            neural_policy_request_in_space(&mut environment.seats[spec.side], action, &space)
                .unwrap()
                .1;
        if action != StructuredAction::Continue
            || [1, 2, 5, 15, 30, 50].contains(&decision)
            || decision == 0
        {
            rows.push(Row {
                frame,
                space,
                target,
                action,
                spec,
            });
        }
        for tick in 0..3 {
            environment.advance(if tick == 0 { request } else { None });
        }
    }
    assert!(
        completed(spec.kind, &environment.result),
        "invalid scripted outcome {spec:?}: {:?}",
        environment.result
    );
    assert_eq!(environment.seats[spec.side].rejections, 0);
    assert!(rows.len() <= 20);
    rows
}
