use std::{collections::BTreeMap, io};

use bota_proto::{
    AbilityId, AbilitySlot, DamageKind, EntityId, EventKind, Fixed, Order, ServerMsg, SlotId,
    Target, Team, UnitKind, UnitView, Vec2, WorldView,
};

/// Maximum absolute simulation tick, including pregame, for the diagnostic cohort.
pub const LANING_TICKS: u32 = 6300;
const TEAMS: [Team; 2] = [Team::Radiant, Team::Dire];

/// Per-seat observations. Damage comes from Events, never SlotStats damage fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaningMetrics {
    pub hero_damage: u64,
    pub hero_magical_damage: u64,
    pub hero_damage_taken: u64,
    pub hero_physical_hits: u32,
    pub hero_magical_hits: u32,
    pub unit_physical_hits: u32,
    pub confirmed_casts: u32,
    pub first_center_tick: Option<u32>,
    pub phase_ticks: u32,
    pub phase_alive_ticks: u32,
    pub phase_xp_ticks: u32,
    pub healthy_behind_tower_ticks: u32,
    pub last_hits: u16,
    pub denies: u16,
    pub deaths: u16,
    pub rejections: u32,
    pub winner: Option<Team>,
}

/// A single fogged seat stream; feeding a second copy of a tick is an error.
#[derive(Default)]
pub struct LaningObserver {
    slot: u8,
    view: Option<WorldView>,
    pending: bool,
    heroes: BTreeMap<EntityId, u8>,
    resets: Vec<AbilityId>,
    center: Option<Vec2>,
    tower: Option<Vec2>,
    last_damage: Option<u32>,
    pub metrics: LaningMetrics,
}

impl LaningObserver {
    pub fn new(slot: SlotId) -> Self {
        assert!(slot.0 < 2);
        Self {
            slot: slot.0,
            ..Self::default()
        }
    }

    pub fn observe(&mut self, message: &ServerMsg) -> io::Result<()> {
        match message {
            ServerMsg::Snapshot { view } => self.snapshot(view)?,
            ServerMsg::Events { tick, events } => {
                if !self.pending || self.view.as_ref().map(|view| view.tick) != Some(*tick) {
                    return Err(io::Error::other(
                        "laning Events must complete exactly one matching Snapshot",
                    ));
                }
                if events.len() > 4096 {
                    return Err(io::Error::other("laning event batch exceeds 4096"));
                }
                for event in events {
                    self.event(*tick, event);
                }
                self.sample();
                self.pending = false;
            }
            ServerMsg::OrderRejected { .. } => self.metrics.rejections += 1,
            ServerMsg::MatchOver { winner, .. } => self.metrics.winner = Some(*winner),
            _ => {}
        }
        Ok(())
    }

    fn snapshot(&mut self, view: &WorldView) -> io::Result<()> {
        validate_view(view, self.slot)?;
        let previous_tick = self.view.as_ref().map_or(0, |previous| previous.tick);
        if self.pending || previous_tick >= view.tick {
            return Err(io::Error::other(
                "laning Snapshot repeated, regressed, or preceded Events",
            ));
        }
        for player in &view.players {
            if let Some(id) = player.unit {
                self.heroes.insert(id, player.slot.0);
            }
        }
        if self.heroes.len() > 128 {
            return Err(io::Error::other("laning hero identity history exceeds 128"));
        }
        self.resets.clear();
        if let Some(hero) = own_hero(view, self.slot) {
            let previous = self
                .view
                .as_ref()
                .and_then(|view| own_hero(view, self.slot));
            for ability in &hero.abilities {
                let before = previous
                    .and_then(|hero| hero.abilities.iter().find(|held| held.id == ability.id))
                    .map_or(0, |held| held.cooldown_left);
                if !ability.passive && ability.cooldown_left > before {
                    self.resets.push(ability.id);
                }
            }
        }
        if self.tower.is_none() {
            let center = lane_center(view)?;
            self.center = Some(center);
            self.tower = view
                .units
                .iter()
                .filter(|unit| unit.kind == UnitKind::Tower && Some(unit.team) == view.viewer)
                .min_by_key(|unit| unit.pos.distance_squared(center))
                .map(|unit| unit.pos);
        }
        self.view = Some(view.clone());
        self.pending = true;
        Ok(())
    }

    fn event(&mut self, tick: u32, event: &EventKind) {
        match event {
            EventKind::Damaged {
                source,
                target,
                amount,
                kind,
                ..
            } if *amount > 0 => {
                let source = source.and_then(|id| self.heroes.get(&id)).copied();
                let target = self.heroes.get(target).copied();
                if target == Some(self.slot) {
                    self.last_damage = Some(tick);
                    if source.is_some_and(|slot| slot != self.slot) {
                        self.metrics.hero_damage_taken += *amount as u64;
                    }
                }
                if source != Some(self.slot) || target == Some(self.slot) {
                    return;
                }
                self.metrics.unit_physical_hits += u32::from(*kind == DamageKind::Physical);
                if target.is_none() {
                    return;
                }
                self.metrics.hero_damage += *amount as u64;
                self.metrics.hero_physical_hits += u32::from(*kind == DamageKind::Physical);
                if *kind == DamageKind::Magical {
                    self.metrics.hero_magical_damage += *amount as u64;
                    self.metrics.hero_magical_hits += 1;
                }
            }
            EventKind::AbilityCast { caster, ability }
                if self.heroes.get(caster) == Some(&self.slot) =>
            {
                if let Some(index) = self.resets.iter().position(|id| id == ability) {
                    self.resets.swap_remove(index);
                    self.metrics.confirmed_casts += 1;
                }
            }
            _ => {}
        }
    }

    fn sample(&mut self) {
        let view = self.view.as_ref().expect("Events checked its Snapshot");
        let player = &view.players[usize::from(self.slot)];
        self.metrics.last_hits = player.last_hits;
        self.metrics.denies = player.denies;
        self.metrics.deaths = player.deaths;
        let phase = (3000..=6000).contains(&view.tick);
        self.metrics.phase_ticks += u32::from(phase);
        let Some(hero) = own_hero(view, self.slot).filter(|hero| hero.hp > 0) else {
            return;
        };
        if hero.pos.within(
            self.center.expect("Snapshot checked lane geometry"),
            Fixed::from_int(1500),
        ) {
            self.metrics.first_center_tick.get_or_insert(view.tick);
        }
        if !phase {
            return;
        }
        self.metrics.phase_alive_ticks += 1;
        self.metrics.phase_xp_ticks += u32::from(view.units.iter().any(|unit| {
            lane_creep(unit)
                && unit.team != hero.team
                && hero.pos.within(unit.pos, Fixed::from_int(1500))
        }));
        let progress = |pos: Vec2| i64::from(pos.x.raw) + i64::from(pos.y.raw);
        let behind = self.tower.is_some_and(|tower| {
            (progress(tower) - progress(hero.pos)) * if hero.team == Team::Radiant { 1 } else { -1 }
                > i64::from(Fixed::from_int(300).raw)
        });
        let healthy = i64::from(hero.hp) * 100 >= i64::from(hero.max_hp) * 60;
        let quiet = self.last_damage.is_none_or(|tick| view.tick - tick > 90);
        self.metrics.healthy_behind_tower_ticks += u32::from(behind && healthy && quiet);
    }
}

/// Uncalibrated human-behavior probes, not strength-ranked opponents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaningProxy {
    RightClick,
    RazeFarm,
    LanePush,
}

/// Public-view-only script with a 1,4,7,... cadence and bounded attack/cast commitment.
pub struct LaningScript {
    slot: u8,
    proxy: LaningProxy,
    pregame: u32,
    center: Option<Vec2>,
    last: Option<Order>,
    committed_until: u32,
    recovering: bool,
}

impl LaningScript {
    pub fn new(slot: SlotId, proxy: LaningProxy, pregame: u32) -> Self {
        assert!(slot.0 < 2);
        assert!(pregame < LANING_TICKS);
        Self {
            slot: slot.0,
            proxy,
            pregame,
            center: None,
            last: None,
            committed_until: 0,
            recovering: false,
        }
    }

    pub fn decide(&mut self, view: &WorldView) -> io::Result<Option<Order>> {
        validate_view(view, self.slot)?;
        if self.center.is_none() {
            self.center = Some(lane_center(view)?);
        }
        let Some(hero) = own_hero(view, self.slot).filter(|hero| hero.hp > 0) else {
            self.last = None;
            self.committed_until = 0;
            return Ok(None);
        };
        self.recovering = if self.recovering {
            i64::from(hero.hp) * 100 < i64::from(hero.max_hp) * 90
        } else {
            i64::from(hero.hp) * 100 < i64::from(hero.max_hp) * 55
        };
        if !(view.tick - 1).is_multiple_of(3)
            || (!self.recovering && view.tick < self.committed_until)
        {
            return Ok(None);
        }
        let order = self.choose(view, hero);
        if matches!(order, Order::Move { .. } | Order::Attack { .. }) && self.last == Some(order) {
            return Ok(None);
        }
        if matches!(order, Order::Cast { .. } | Order::Attack { .. }) {
            self.committed_until = view.tick + 24;
        }
        if !matches!(order, Order::Learn { .. }) {
            self.last = Some(order);
        }
        Ok(Some(order))
    }

    fn choose(&self, view: &WorldView, hero: &UnitView) -> Order {
        if self.recovering {
            let fountain = view
                .units
                .iter()
                .find(|unit| unit.kind == UnitKind::Fountain && unit.team == hero.team)
                .map_or(hero.pos, |unit| unit.pos);
            return Order::Move {
                target: Target::Pos(fountain),
            };
        }
        if let Some(index) = hero.abilities.iter().position(|ability| ability.can_level) {
            return Order::Learn {
                slot: AbilitySlot(index as u8),
            };
        }
        let center = self.center.expect("decide checked public lane geometry");
        if view.tick < self.pregame {
            return Order::Move {
                target: Target::Pos(center),
            };
        }
        if self.proxy == LaningProxy::RazeFarm
            && let Some(order) = raze(view, hero)
        {
            return order;
        }
        self.attack(view, hero, center)
    }

    fn attack(&self, view: &WorldView, hero: &UnitView, center: Vec2) -> Order {
        let enemy = view
            .units
            .iter()
            .filter(|unit| unit.hp > 0 && unit.team != hero.team && unit.team != Team::Neutral);
        let last_hit = view
            .units
            .iter()
            .filter(|unit| {
                lane_creep(unit)
                    && i64::from(unit.hp) <= i64::from(hero.attack_damage) * 2 / 3
                    && hero
                        .pos
                        .within(unit.pos, hero.attack_range + Fixed::from_int(50))
            })
            .min_by_key(|unit| (unit.hp, unit.id));
        let harass = enemy
            .clone()
            .filter(|unit| {
                unit.kind == UnitKind::Hero
                    && hero.pos.within(unit.pos, Fixed::from_int(1200))
                    && !view.units.iter().any(|tower| {
                        matches!(tower.kind, UnitKind::Tower | UnitKind::Fountain)
                            && tower.team == unit.team
                            && tower
                                .pos
                                .within(unit.pos, tower.attack_range + Fixed::from_int(200))
                    })
            })
            .min_by_key(|unit| (hero.pos.distance_squared(unit.pos), unit.id));
        let push = enemy
            .filter(|unit| lane_creep(unit) && hero.pos.within(unit.pos, Fixed::from_int(1200)))
            .min_by_key(|unit| (hero.pos.distance_squared(unit.pos), unit.id));
        let target = if self.proxy == LaningProxy::LanePush {
            push
        } else {
            last_hit.or(harass)
        };
        if let Some(unit) = target {
            return Order::Attack {
                target: Target::Unit(unit.id),
            };
        }
        let offset = if hero.team == Team::Radiant {
            1200
        } else {
            -1200
        };
        let goal = if self.proxy == LaningProxy::LanePush {
            center + Vec2::from_ints(offset, offset)
        } else {
            center
        };
        Order::Attack {
            target: Target::Pos(goal),
        }
    }
}

fn validate_view(view: &WorldView, slot: u8) -> io::Result<()> {
    if view.viewer != Some(TEAMS[usize::from(slot)])
        || view.players.len() != 2
        || view.players.iter().enumerate().any(|(index, player)| {
            usize::from(player.slot.0) != index
                || player.team != TEAMS[index]
                || (Some(player.team) != view.viewer
                    && (player.gold.is_some() || player.stash.is_some() || player.kit.is_some()))
        })
    {
        return Err(io::Error::other(
            "laning requires the assigned seat's fogged two-player view",
        ));
    }
    if !(1..=LANING_TICKS).contains(&view.tick)
        || view.units.len() > 4096
        || view.units.iter().any(|unit| unit.abilities.len() > 8)
    {
        return Err(io::Error::other(
            "laning view exceeds tick/unit/ability limits",
        ));
    }
    Ok(())
}

fn own_hero(view: &WorldView, slot: u8) -> Option<&UnitView> {
    let id = view.players[usize::from(slot)].unit?;
    view.units
        .iter()
        .find(|unit| unit.id == id && unit.kind == UnitKind::Hero)
}

fn lane_creep(unit: &UnitView) -> bool {
    unit.hp > 0
        && matches!(
            unit.kind,
            UnitKind::CreepMelee
                | UnitKind::CreepRanged
                | UnitKind::CreepFlagbearer
                | UnitKind::CreepSiege
        )
}

fn lane_center(view: &WorldView) -> io::Result<Vec2> {
    let tower = |team| {
        view.units
            .iter()
            .find(|unit| unit.kind == UnitKind::Tower && unit.team == team)
            .map(|unit| unit.pos)
    };
    let (Some(radiant), Some(dire)) = (tower(Team::Radiant), tower(Team::Dire)) else {
        return Err(io::Error::other(
            "laning Map1 requires two initially public lane towers",
        ));
    };
    Ok(Vec2 {
        x: Fixed {
            raw: ((i64::from(radiant.x.raw) + i64::from(dire.x.raw)) / 2) as i32,
        },
        y: Fixed {
            raw: ((i64::from(radiant.y.raw) + i64::from(dire.y.raw)) / 2) as i32,
        },
    })
}

fn raze_center(hero: &UnitView, reach: i32) -> Vec2 {
    let slope = i64::from(hero.facing.brads % 8192);
    let (x, y) = match hero.facing.brads / 8192 {
        0 => (8192, slope),
        1 => (8192 - slope, 8192),
        2 => (-slope, 8192),
        3 => (-8192, 8192 - slope),
        4 => (-8192, -slope),
        5 => (slope - 8192, -8192),
        6 => (slope, -8192),
        _ => (8192, slope - 8192),
    };
    let length = ((x * x + y * y) as u64).isqrt() as i64;
    hero.pos
        + Vec2::from_ints(
            (x * i64::from(reach) / length) as i32,
            (y * i64::from(reach) / length) as i32,
        )
}

fn raze(view: &WorldView, hero: &UnitView) -> Option<Order> {
    for (index, ability) in hero.abilities.iter().enumerate() {
        if !(13..=15).contains(&ability.id.0)
            || ability.level == 0
            || ability.cooldown_left > 0
            || hero.mana < ability.mana_cost
        {
            continue;
        }
        let center = raze_center(hero, ability.range);
        if view.units.iter().any(|unit| {
            unit.hp > 0
                && unit.team != hero.team
                && (unit.kind == UnitKind::Hero || lane_creep(unit))
                && center.within(unit.pos, Fixed::from_int(180))
        }) {
            return Some(Order::Cast {
                slot: AbilitySlot(index as u8),
                target: Target::None,
            });
        }
    }
    None
}
