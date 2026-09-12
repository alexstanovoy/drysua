#![allow(
    clippy::float_arithmetic,
    reason = "Test-only bounded geometric measurements."
)]

use super::*;
use crate::{ActionTarget, ControlledUnit, PointIndex, ShopIndex, StructuredAction};
use bota_proto::{
    AbilitySlot, DamageKind, Fixed, ItemId, ItemSlot, UnitKind, UnitView, Vec2, WorldView,
};
use bota_server::game::{ItemStack, Level, Status, StatusKind, Statuses, UnitOrder, World, rules};

#[path = "map2_behavior_probe_run.rs"]
mod run;
#[path = "map2_behavior_probe_tests.rs"]
mod tests;

const FAMILIES: [Family; 8] = [
    Family::RazeEmpty,
    Family::RazeChain,
    Family::MangoBuy,
    Family::MangoDelivery,
    Family::MangoUse,
    Family::RecoveryBarracks,
    Family::LaneIdle,
    Family::NeutralIdle,
];
const TICK_LIMIT: u32 = 600;
const TRACE_LIMIT: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    RazeEmpty,
    RazeChain,
    MangoBuy,
    MangoDelivery,
    MangoUse,
    RecoveryBarracks,
    LaneIdle,
    NeutralIdle,
}

struct Case {
    arena: Arena,
    seats: Vec<ArenaSeatPolicy>,
    side: usize,
    family: Family,
    goal: Vec2,
    metrics: Metrics,
    trace: Vec<String>,
    terminal: bool,
}

fn fixture(family: Family, side: usize) -> Case {
    fixture_with(family, side, |_| {})
}

fn fixture_with(family: Family, side: usize, extra: impl FnOnce(&mut World)) -> Case {
    assert!(side < 2);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 10091600,
    })
    .expect("native Map2 fixture");
    let mut goal = Vec2::from_ints(0, 0);
    let configured = arena.configure_for_test(|world| {
        configure(world, family, side);
        extra(world);
        goal = world.map.fountains[side];
    });
    for (messages, fresh) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    let seats = setup_seats(start).expect("full configured baseline before tracker construction");
    let position = seats[side].tracker.own_hero().expect("hero").pos;
    assert!(arena.tick() > 0);
    Case {
        arena,
        seats,
        side,
        family,
        goal,
        metrics: Metrics::new(position, goal),
        trace: Vec::with_capacity(TRACE_LIMIT),
        terminal: false,
    }
}

fn configure(world: &mut World, family: Family, side: usize) {
    assert_eq!(world.map.id, MapId(2));
    assert!(side < world.seats.len());
    world.tick = 1201;
    for index in 0..2 {
        let hero = world.seats[index].unit.expect("hero");
        world.seats[index].gold = 0;
        world.seats[index].level = 3;
        world.level.insert(hero, Level(3));
        world.statuses.remove(hero);
        world.set_order(hero, UnitOrder::Stand);
        for slot in &mut world.abilities.get_mut(hero).expect("book").slots[..3] {
            slot.level = 1;
        }
    }
    world.settle();
    for index in 0..2 {
        world.fill_pools(world.seats[index].unit.expect("hero"));
    }
    let hero = world.seats[side].unit.expect("actor");
    let direction = if side == 0 { 1 } else { -1 };
    let position = match family {
        Family::RecoveryBarracks => {
            world.map.barracks[side][0].2 + Vec2::from_ints(180 * direction, 180 * direction)
        }
        Family::MangoBuy | Family::MangoDelivery => {
            world.map.fountains[side] + Vec2::from_ints(1100 * direction, 1100 * direction)
        }
        _ => Vec2::from_ints(9216, 9216),
    };
    let transform = world.transform.get_mut(hero).expect("transform");
    transform.pos = position;
    transform.facing.brads = if side == 0 { 0 } else { 32768 };
    configure_resources(world, family, side);
    configure_targets(world, family, side);
}

fn configure_resources(world: &mut World, family: Family, side: usize) {
    assert!(side < 2);
    assert_eq!(world.tick, 1201);
    let hero = world.seats[side].unit.expect("hero");
    if matches!(
        family,
        Family::MangoBuy | Family::MangoDelivery | Family::MangoUse | Family::RecoveryBarracks
    ) {
        world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
    }
    if family == Family::RecoveryBarracks {
        world.health.get_mut(hero).expect("health").hp = Fixed::from_int(100);
    }
    if family == Family::MangoBuy {
        world.seats[side].gold = 65;
    }
    if matches!(family, Family::MangoUse | Family::MangoDelivery) {
        let mango = ItemStack::bought(ItemId(42), world.seats[side].slot, world.tick);
        if family == Family::MangoUse {
            world.inventory.get_mut(hero).expect("bag").slots[0] = mango;
        } else {
            world.seats[side].stash.slots[0] = mango;
        }
    }
}

fn configure_targets(world: &mut World, family: Family, side: usize) {
    assert!(side < 2);
    assert_eq!(world.map.id, MapId(2));
    let hero = world.seats[side].unit.expect("hero");
    let position = world.transform.get(hero).expect("position").pos;
    if family == Family::RazeChain {
        let enemy = world.seats[1 - side].unit.expect("enemy");
        let direction = if side == 0 { 1 } else { -1 };
        world.transform.get_mut(enemy).expect("enemy position").pos =
            position + Vec2::from_ints(450 * direction, 0);
        let mut statuses = Statuses::default();
        statuses.put(Status {
            kind: StatusKind::Shadowraze {
                from: hero,
                stacks: 2,
            },
            ticks_left: 240,
        });
        world.statuses.insert(enemy, statuses);
    }
    if family == Family::LaneIdle {
        for offset in [(-100, 0), (0, 100), (100, 0)] {
            world.spawn_unit(
                &bota_server::game::MELEE_CREEP,
                world.seats[1 - side].team,
                position + Vec2::from_ints(offset.0, offset.1),
            );
        }
    }
    if family == Family::NeutralIdle {
        world.tick = rules::FIRST_NEUTRAL_TICK;
        world.fill_camps();
        let camp = if side == 0 { 2 } else { 22 };
        let position = world.map.camps[camp].pos + Vec2::from_ints(0, 160);
        world
            .transform
            .get_mut(hero)
            .expect("neutral exposure position")
            .pos = position;
    }
}

impl Case {
    fn view(&self) -> &WorldView {
        self.seats[self.side].tracker.current().expect("view")
    }
    fn hero(&self) -> &UnitView {
        self.seats[self.side].tracker.own_hero().expect("hero")
    }
    fn space(&self) -> ActionSpace {
        ActionSpace::from_tracker_with_readiness(
            &self.seats[self.side].tracker,
            &self.seats[self.side].readiness,
        )
        .expect("legal action space")
    }
    fn held_mango(&self) -> u32 {
        mango_count(&self.hero().items)
    }
    fn stash_mango(&self) -> u32 {
        mango_count(
            self.seats[self.side]
                .tracker
                .own_player()
                .expect("player")
                .stash
                .as_ref()
                .expect("stash"),
        )
    }
    fn send_control(&mut self, action: StructuredAction) {
        assert!(!self.terminal);
        let (_, space) =
            prepare_neural_seat_policy_sample(&mut self.seats[self.side]).expect("control frame");
        assert!(
            space.allows(action),
            "known-wire control must be represented: {action:?}"
        );
        let request = neural_policy_request_in_space(&mut self.seats[self.side], action, &space)
            .expect("control transport")
            .1;
        assert!(request.is_some());
        self.advance(request);
    }
    fn advance(&mut self, request: Option<Request>) {
        assert!(!self.terminal);
        assert!(self.metrics.ticks < TICK_LIMIT);
        let before = self.view().clone();
        let mut requests = [None, None];
        requests[self.side] = request;
        let step = self.arena.step(&requests).expect("bounded fixture tick");
        self.metrics
            .observe(&before, &step.messages[self.side], self.side, self.goal);
        for (seat, messages) in self.seats.iter_mut().zip(step.messages) {
            self.terminal |= observe_messages(seat, &messages)
                .expect("complete contiguous seat stream")
                .is_some();
            let interval = seat
                .tracker
                .take_map2_reward_interval()
                .expect("complete reward accounting pair");
            if seat.tracker.slot() == SlotId(self.side as u8) {
                self.metrics.reward_creep_damage += interval.observations.creep_damage_taken;
                self.metrics.mana_spent += interval.observations.mana_spent;
            }
        }
        self.metrics.ticks += 1;
    }
}

fn cast(unit: ControlledUnit, slot: u8) -> StructuredAction {
    assert!(slot < 6);
    StructuredAction::Cast {
        unit,
        slot: AbilitySlot(slot),
        target: ActionTarget::None,
    }
}

fn mango_use() -> StructuredAction {
    StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(0),
        target: ActionTarget::None,
    }
}

fn mango_buy(space: &ActionSpace) -> Option<StructuredAction> {
    let item = space
        .shop_candidates()
        .iter()
        .position(|item| item.item == ItemId(42))?;
    let action = StructuredAction::Buy {
        unit: ControlledUnit::Hero,
        item: ShopIndex(item),
    };
    space.allows(action).then_some(action)
}

fn goal_move(space: &ActionSpace, goal: Vec2) -> Option<StructuredAction> {
    space
        .point_candidates()
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let action = StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point: PointIndex(index),
            };
            space
                .allows(action)
                .then_some((point.position.distance_squared(goal), action))
        })
        .min_by_key(|entry| entry.0)
        .map(|entry| entry.1)
}

fn mango_count(items: &[Option<bota_proto::ItemView>]) -> u32 {
    assert!(items.len() <= 9);
    items
        .iter()
        .flatten()
        .filter(|item| item.id == ItemId(42))
        .map(|item| u32::from(item.charges.unwrap_or(0)))
        .sum()
}

fn confirmed_raze(before: &UnitView, after: &UnitView, event: &EventKind) -> bool {
    assert_eq!(before.id, after.id);
    let EventKind::AbilityCast { caster, ability } = event else {
        return false;
    };
    if *caster != before.id || !(13..=15).contains(&ability.0) {
        return false;
    }
    let Some(index) = before.abilities.iter().position(|slot| slot.id == *ability) else {
        return false;
    };
    let Some(next) = after.abilities.get(index) else {
        return false;
    };
    next.cooldown_left > before.abilities[index].cooldown_left || before.mana - after.mana >= 70
}

fn distance(source: Vec2, target: Vec2) -> f64 {
    let horizontal = f64::from((source.x - target.x).to_int());
    let vertical = f64::from((source.y - target.y).to_int());
    horizontal.hypot(vertical)
}

#[derive(Debug, Default)]
struct Metrics {
    ticks: u32,
    decisions: u32,
    kinds: [u32; ActionKind::COUNT],
    sent: u32,
    suppressed: u32,
    casts: u32,
    cast_events_without_confirmation: u32,
    hero_hit_casts: u32,
    no_damage_unknown: u32,
    visible_enemy_no_damage: u32,
    magic_hero_damage: u64,
    magic_other_damage: u64,
    max_stacks: u32,
    stack_increases: u32,
    raze_opportunities: u32,
    third_stack_opportunities: u32,
    mango_bought: u32,
    mango_used: u32,
    mango_arrivals: u32,
    manual_mana: u64,
    mango_buy_opportunities: u32,
    mango_use_opportunities: u32,
    creep_damage: u64,
    lane_hits: u32,
    neutral_hits: u32,
    unknown_incoming: u64,
    reward_creep_damage: u64,
    mana_spent: u64,
    start_distance: f64,
    end_distance: f64,
    path: f64,
    previous_delta: Option<(i32, i32)>,
    reversals: u32,
    away_ticks: u32,
    stationary_ticks: u32,
}

impl Metrics {
    fn new(position: Vec2, goal: Vec2) -> Self {
        let initial = distance(position, goal);
        Self {
            start_distance: initial,
            end_distance: initial,
            ..Self::default()
        }
    }
    fn progress(&self) -> f64 {
        self.start_distance - self.end_distance
    }
    fn movement(&mut self, before: Vec2, after: Vec2, goal: Vec2) {
        let delta = ((after.x - before.x).to_int(), (after.y - before.y).to_int());
        self.path += distance(before, after);
        self.end_distance = distance(after, goal);
        if delta == (0, 0) {
            self.stationary_ticks += 1;
            return;
        }
        self.away_ticks += u32::from(self.end_distance > distance(before, goal) + 0.5);
        if let Some(previous) = self.previous_delta {
            self.reversals += u32::from(
                i64::from(previous.0) * i64::from(delta.0)
                    + i64::from(previous.1) * i64::from(delta.1)
                    < 0,
            );
        }
        self.previous_delta = Some(delta);
    }
    fn observe(&mut self, before: &WorldView, messages: &[ServerMsg], side: usize, goal: Vec2) {
        assert!(side < 2);
        let after = messages
            .iter()
            .find_map(|message| match message {
                ServerMsg::Snapshot { view } => Some(view),
                _ => None,
            })
            .expect("native snapshot");
        assert_eq!(after.tick, before.tick + 1);
        let events = messages
            .iter()
            .find_map(|message| match message {
                ServerMsg::Events { tick, events } => {
                    assert_eq!(*tick, after.tick);
                    Some(events)
                }
                _ => None,
            })
            .expect("native events");
        let hero = before.units.iter().find(|unit| {
            unit.id
                == before.players[side].unit.unwrap_or(bota_proto::EntityId {
                    idx: u32::MAX,
                    generation: 0,
                })
        });
        let Some(hero) = hero else {
            return;
        };
        let next = after.units.iter().find(|unit| unit.id == hero.id);
        if let Some(next) = next {
            self.movement(hero.pos, next.pos, goal);
            self.mango_arrivals +=
                mango_count(&next.items).saturating_sub(mango_count(&hero.items));
        }
        for event in events {
            self.event(before, after, hero, event);
            if let Some(next) = next {
                if confirmed_raze(hero, next, event) {
                    self.cast_outcome(before, after, hero, events);
                } else if matches!(event, EventKind::AbilityCast { caster, .. } if *caster == hero.id)
                {
                    self.cast_events_without_confirmation += 1;
                }
            }
        }
        for unit in &after.units {
            if unit.kind != UnitKind::Hero || unit.team == hero.team {
                continue;
            }
            let stacks = raze_stacks(unit);
            self.max_stacks = self.max_stacks.max(stacks);
            let prior = before
                .units
                .iter()
                .find(|prior| prior.id == unit.id)
                .map(raze_stacks);
            if prior.is_some_and(|prior| stacks > prior) {
                self.stack_increases += 1;
            }
        }
    }
    fn event(&mut self, before: &WorldView, after: &WorldView, hero: &UnitView, event: &EventKind) {
        match *event {
            EventKind::ItemBought {
                slot,
                item: ItemId(42),
            } if Some(slot) == hero.owner => self.mango_bought += 1,
            EventKind::Healed {
                source: Some(source),
                target,
                mana,
                ..
            } if source == hero.id && target == hero.id && mana > 0 => {
                self.manual_mana += mana as u64;
                if after
                    .units
                    .iter()
                    .find(|unit| unit.id == hero.id)
                    .is_some_and(|next| mango_count(&next.items) < mango_count(&hero.items))
                {
                    self.mango_used += 1;
                }
            }
            EventKind::Damaged {
                source,
                target,
                amount,
                kind,
                ..
            } if amount > 0 => {
                let source_unit = source.and_then(|id| {
                    after
                        .units
                        .iter()
                        .chain(&before.units)
                        .find(|unit| unit.id == id)
                });
                if target == hero.id {
                    match source_unit.map(|unit| unit.kind) {
                        Some(UnitKind::CreepNeutral) => {
                            self.neutral_hits += 1;
                            self.creep_damage += amount as u64;
                        }
                        Some(
                            UnitKind::CreepMelee
                            | UnitKind::CreepFlagbearer
                            | UnitKind::CreepRanged
                            | UnitKind::CreepSiege,
                        ) => {
                            self.lane_hits += 1;
                            self.creep_damage += amount as u64;
                        }
                        None => self.unknown_incoming += amount as u64,
                        _ => {}
                    }
                }
                if source == Some(hero.id) && kind == DamageKind::Magical {
                    if before
                        .players
                        .iter()
                        .any(|player| player.team != hero.team && player.unit == Some(target))
                    {
                        self.magic_hero_damage += amount as u64;
                    } else {
                        self.magic_other_damage += amount as u64;
                    }
                }
            }
            _ => {}
        }
    }
    fn cast_outcome(
        &mut self,
        before: &WorldView,
        after: &WorldView,
        hero: &UnitView,
        events: &[EventKind],
    ) {
        self.casts += 1;
        let hits = events
            .iter()
            .filter_map(|event| match event {
                EventKind::Damaged {
                    source: Some(source),
                    target,
                    kind: DamageKind::Magical,
                    amount,
                    ..
                } if *source == hero.id && *amount > 0 => Some(*target),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(hits.len() <= 4096);
        let enemy = before
            .players
            .iter()
            .find(|player| player.team != hero.team)
            .and_then(|player| player.unit);
        self.hero_hit_casts += u32::from(enemy.is_some_and(|enemy| hits.contains(&enemy)));
        if hits.is_empty() {
            self.no_damage_unknown += 1;
            self.visible_enemy_no_damage += u32::from(enemy.is_some_and(|enemy| {
                before.units.iter().any(|unit| unit.id == enemy)
                    && after.units.iter().any(|unit| unit.id == enemy)
            }));
        }
    }
}

fn raze_stacks(unit: &UnitView) -> u32 {
    unit.effects
        .iter()
        .filter(|effect| {
            effect.id.0 == 15
                && effect
                    .ticks_left
                    .is_some_and(|ticks| ticks > 0 && ticks <= 240)
        })
        .filter_map(|effect| effect.stacks.filter(|stacks| (1..=255).contains(stacks)))
        .max()
        .unwrap_or(0)
}
