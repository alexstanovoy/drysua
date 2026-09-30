//! Shadow Fiend rule policy that harasses Teacher off the lane and pushes its tower meanwhile.
//!
//! Evidence from `drysua duel` shaped every rule here. Teacher spends its mana on razes at
//! creeps, towers and heroes, walks home at 40% health with no hysteresis, and never walks into
//! tower range while an enemy hero stands within 1200. HarassPush therefore:
//! - razes only once its facing has settled on the target (a turning hero sweeps 32 degrees a
//!   tick, and a raze fires along the facing at resolution time);
//! - fights when its ready razes force Teacher home or kill it, when Teacher lacks raze mana
//!   (attacking it then, since it can only answer with attacks), or when Teacher sieges our
//!   tower and a raze pair is ready;
//! - otherwise hovers just outside Teacher's far-raze disk, which also denies its tower push;
//! - with Teacher away, hits and razes the tower its wave tanks;
//! - never idles in enemy tower range, backs off when bleeding, and walks home below 30% health.
//!
//! It buys nothing and learns razes first. Every action goes through the drysua action space
//! and every input is seat-visible, so the policy is a behaviour-cloning target.

use bota_proto::{
    AbilityId, AbilitySlot, EffectId, EntityId, Fixed, Order, Target, Team, UnitKind, UnitView,
    Vec2,
};

use crate::raze_aim::{
    SHADOWRAZE_RADIUS, SHADOWRAZES, facing_towards, isqrt, point_along, predicted_position,
    raze_center, raze_reach,
};
use crate::scripted::tactics::{
    best_attack_creep, cast_at, enemy_tower_danger, facing_gap, in_attack_reach, is_lane_creep,
    magical_damage, physical_damage, ratio_at_most, ratio_below, raze_damage, tower_corridor_safe,
};
use crate::{
    ActionError, ActionSpace, ControlledUnit, EntityIndex, EntityRelation, IssuedOrder,
    ItemReadiness, OrderPersistence, PointIndex, StateTracker, StructuredAction,
};

const REQUIEM: AbilityId = AbilityId(16);
const NECROMASTERY: AbilityId = AbilityId(17);
const PRESENCE: AbilityId = AbilityId(18);
const RAZE_STACK_EFFECT: EffectId = EffectId(15);
const RAZE_STACK_DAMAGE: [i32; 4] = [50, 60, 70, 80];
const RAZE_MANA: [i32; 4] = [75, 80, 85, 90];
/// Teacher walks home at or below this health share.
const ENEMY_RETREAT_PERCENT: i32 = 40;
/// Own health that always starts a walk home.
const RETREAT_PERCENT: i32 = 30;
/// Own health that ends a walk home; hysteresis stops lane/fountain oscillation.
const RETURN_PERCENT: i32 = 80;
/// Own health share below which the hero never starts a fight.
const ENGAGE_HEALTH_PERCENT: i32 = 50;
/// Farthest reach of an enemy raze disk plus one decision of approach.
const THREAT_DISTANCE: i32 = 1_000;
/// Just outside Teacher's far-raze disk, still inside the 1200 that stops its tower push.
const HOVER_DISTANCE: i32 = 1_100;
/// Enemy heroes farther than this cannot contest a push.
const ENGAGE_DISTANCE: i32 = 1_600;
/// An enemy hero seen this recently still threatens where it vanished into fog.
const ENEMY_MEMORY_TICKS: u32 = 120;
/// Score improvement below which a walk is not worth an order.
const PROGRESS_SLACK: i32 = 40;
/// Tower attack range (700) plus the tower (144) and hero (24) bounds.
const TOWER_REACH: i32 = 868;
/// Laning posts stay this far from the enemy tower centre, just beyond its reach, so a wave
/// that tanks the tower still keeps the hero close enough to join in.
const POST_TOWER_CLEARANCE: i32 = 950;
/// Extra distance kept outside enemy tower attack range while walking.
const TOWER_MARGIN: i32 = 150;
/// An enemy hero this close to the own tower is razing or pushing it.
const TOWER_DEFENSE_DISTANCE: i32 = 1_300;
/// Distance kept behind the own front creep while laning.
const FRONT_OFFSET: i32 = 250;
/// How far past the lane post the hero may drift while fighting creeps.
const AHEAD_SLACK: i32 = 200;
/// Mana kept for a two-raze trade before spending razes on structures.
const HARASS_MANA_RESERVE: i32 = 160;
/// Facing error (about 5.5 degrees) under which the hero has finished turning toward a target.
const AIM_TOLERANCE_BRADS: u16 = 1_000;
/// Allied creeps that must stand near the enemy tower before the hero hits it.
const TOWER_TANKS: usize = 2;
/// A tower keeps its target while it stays in range, so a focused hero stays out this long.
const TOWER_SHY_TICKS: u32 = 150;

/// Deterministic HarassPush rule policy; see the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HarassPush {
    hero: Option<EntityId>,
    retreating: bool,
    last_hp: i32,
    /// Health lost since the previous decision; creep and tower focus show up here first.
    bleeding: i32,
    tower_shy_until: u32,
}

impl HarassPush {
    /// Creates a policy with empty per-match memory.
    pub const fn new() -> Self {
        Self {
            hero: None,
            retreating: false,
            last_hp: 0,
            bleeding: 0,
            tower_shy_until: 0,
        }
    }

    /// Selects an action and returns the exact action space used to select it.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        let space = ActionSpace::from_tracker_with_readiness(tracker, readiness)?;
        self.observe(tracker, space.tick());
        let active = persistence
            .active_body_for(None)
            .map(|(_, issued)| issued.order);
        let selected = self.select(tracker, &space, active);
        if !space.allows(selected) {
            return Err(ActionError::InvalidSchema(
                "HarassPush selected masked action",
            ));
        }
        let action = if persistence.should_send(space.decode(selected)?).is_none() {
            StructuredAction::Continue
        } else {
            selected
        };
        Ok((action, space))
    }

    /// HarassPush keeps no per-order memory; accepted orders need no bookkeeping.
    pub fn note_sent(&mut self, _sequence: u32, _issued: IssuedOrder, _tick: u32) {}

    /// HarassPush keeps no per-order memory; rejected orders need no rollback.
    pub fn note_rejected(&mut self, _sequence: u32) -> bool {
        false
    }

    fn observe(&mut self, tracker: &StateTracker, tick: u32) {
        let Some(hero) = tracker.own_hero() else {
            return;
        };
        if self.hero != Some(hero.id) {
            *self = Self::new();
            self.hero = Some(hero.id);
            self.last_hp = hero.hp;
        }
        self.bleeding = self.last_hp.saturating_sub(hero.hp).max(0);
        self.last_hp = hero.hp;
        if self.bleeding > 0
            && nearest_enemy_hero(tracker, hero.pos, ENGAGE_DISTANCE).is_none()
            && enemy_tower_danger(tracker, hero.pos, hero.bound)
        {
            self.tower_shy_until = tick.saturating_add(TOWER_SHY_TICKS);
        }
        if ratio_at_most(hero.hp, hero.max_hp, RETREAT_PERCENT) {
            self.retreating = true;
        } else if !ratio_below(hero.hp, hero.max_hp, RETURN_PERCENT) {
            self.retreating = false;
        }
    }

    fn select(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        active: Option<Order>,
    ) -> StructuredAction {
        let Some(hero) = tracker.own_hero() else {
            return StructuredAction::Continue;
        };
        let Some(lane) = Lane::observe(tracker) else {
            return StructuredAction::Continue;
        };
        let turn = Turn {
            tracker,
            space,
            hero,
            lane,
            tower_shy: space.tick() < self.tower_shy_until,
            bleeding: self.bleeding,
            active,
        };
        let enemy = nearest_enemy_hero(tracker, hero.pos, ENGAGE_DISTANCE);
        let fight = enemy.filter(|enemy| turn.wants_fight(enemy));
        let lurking = enemy.or_else(|| remembered_enemy_hero(tracker, hero.pos));
        let action = learn(hero, space)
            .or_else(|| {
                let free = self.retreating || fight.is_some();
                enemy
                    .filter(|_| free)
                    .and_then(|enemy| raze_now(tracker, space, hero, enemy))
            })
            .or_else(|| {
                if self.retreating {
                    return turn.move_towards(turn.lane.own_fountain, false);
                }
                match (fight, lurking) {
                    (Some(enemy), _) => turn.engage(enemy),
                    (None, Some(enemy)) => turn.keep_distance(enemy),
                    (None, None) => turn.push(),
                }
            });
        action.unwrap_or(StructuredAction::Continue)
    }
}

/// Seat-visible mid-lane landmarks.
struct Lane<'a> {
    own_fountain: Vec2,
    enemy_fountain: Vec2,
    own_tower: Option<&'a UnitView>,
    enemy_tower: Option<&'a UnitView>,
    post: Vec2,
}

impl<'a> Lane<'a> {
    fn observe(tracker: &'a StateTracker) -> Option<Self> {
        let view = tracker.current()?;
        let fountain = |own: bool| {
            view.units
                .iter()
                .find(|unit| {
                    unit.kind == UnitKind::Fountain && (unit.team == tracker.team()) == own
                })
                .map(|unit| unit.pos)
        };
        let (own_fountain, enemy_fountain) = (fountain(true)?, fountain(false)?);
        let center = point_along(
            own_fountain,
            enemy_fountain,
            Fixed::from_int(distance(own_fountain, enemy_fountain) / 2),
        );
        // The mid tier-one towers are the living towers nearest the map centre.
        let tower = |own: bool| {
            view.units
                .iter()
                .filter(|unit| {
                    unit.kind == UnitKind::Tower
                        && unit.hp > 0
                        && unit.team != Team::Neutral
                        && (unit.team == tracker.team()) == own
                })
                .min_by_key(|unit| (center.distance_squared(unit.pos), unit.id))
        };
        let own_tower = tower(true);
        // The hero stands just behind the own creep nearest the enemy base.
        let front = view
            .units
            .iter()
            .filter(|unit| unit.team == tracker.team() && is_lane_creep(unit.kind) && unit.hp > 0)
            .min_by_key(|unit| (enemy_fountain.distance_squared(unit.pos), unit.id));
        let post = match (front, own_tower) {
            (Some(creep), _) => point_along(creep.pos, own_fountain, Fixed::from_int(FRONT_OFFSET)),
            (None, Some(tower)) => point_along(tower.pos, own_fountain, Fixed::from_int(400)),
            (None, None) => own_fountain,
        };
        let enemy_tower = tower(false);
        // Never idle inside enemy tower range, even behind a tanking wave.
        let post = match enemy_tower {
            Some(tower) if distance(post, tower.pos) < POST_TOWER_CLEARANCE => point_along(
                tower.pos,
                own_fountain,
                Fixed::from_int(POST_TOWER_CLEARANCE),
            ),
            _ => post,
        };
        Some(Self {
            own_fountain,
            enemy_fountain,
            own_tower,
            enemy_tower,
            post,
        })
    }

    /// Whether `position` stands closer to the enemy base than the lane post allows.
    fn ahead(&self, position: Vec2) -> bool {
        distance(position, self.enemy_fountain) + AHEAD_SLACK
            < distance(self.post, self.enemy_fountain)
    }
}

/// Everything one decision reads, so the tactical helpers share one borrow.
struct Turn<'a> {
    tracker: &'a StateTracker,
    space: &'a ActionSpace,
    hero: &'a UnitView,
    lane: Lane<'a>,
    tower_shy: bool,
    bleeding: i32,
    /// The body order still running from an earlier decision.
    active: Option<Order>,
}

impl Turn<'_> {
    /// Fights away from the enemy tower while healthy when the ready razes force Teacher home
    /// or kill it, when it lacks raze mana, or when it sieges our tower and a raze pair is ready.
    fn wants_fight(&self, enemy: &UnitView) -> bool {
        let (tracker, hero) = (self.tracker, self.hero);
        if enemy_tower_danger(tracker, enemy.pos, hero.bound)
            || enemy_tower_danger(tracker, hero.pos, hero.bound)
            || ratio_below(hero.hp, hero.max_hp, ENGAGE_HEALTH_PERCENT)
        {
            return false;
        }
        let (burst, razes) = own_burst(self.space, hero, enemy);
        let left = enemy.hp.saturating_sub(burst);
        if left <= 0 || ratio_at_most(left, enemy.max_hp, ENEMY_RETREAT_PERCENT) {
            return true;
        }
        let sieging = self.lane.own_tower.is_some_and(|tower| {
            enemy
                .pos
                .within(tower.pos, Fixed::from_int(TOWER_DEFENSE_DISTANCE))
        });
        razes >= 2 && sieging || unarmed(enemy)
    }

    /// Faces and closes on the enemy so the next ready raze lands; an enemy without raze
    /// mana is also attacked, since it can only answer with attacks.
    fn engage(&self, enemy: &UnitView) -> Option<StructuredAction> {
        let follow = self
            .space
            .entity_index(enemy.id)
            .map(|target| {
                if unarmed(enemy) {
                    StructuredAction::AttackUnit {
                        unit: ControlledUnit::Hero,
                        target,
                    }
                } else {
                    StructuredAction::FollowUnit {
                        unit: ControlledUnit::Hero,
                        target,
                    }
                }
            })
            .filter(|action| self.space.allows(*action));
        match follow {
            Some(action) if tower_corridor_safe(self.tracker, self.hero, enemy.pos, false) => {
                Some(action)
            }
            _ => self.keep_distance(enemy),
        }
    }

    /// Stays just outside the enemy's raze reach near the lane post, farming what is safe.
    fn keep_distance(&self, enemy: &UnitView) -> Option<StructuredAction> {
        if distance(self.hero.pos, enemy.pos) > THREAT_DISTANCE
            && self.bleeding == 0
            && let Some(action) = self.farm(Some(enemy))
        {
            return Some(action);
        }
        let post = self.lane.post;
        self.walk(
            |position: Vec2| {
                let gap = distance(position, enemy.pos);
                (HOVER_DISTANCE - gap).max(0) * 4 + distance(position, post) / 2
            },
            false,
        )
    }

    /// With the enemy hero away, hits the tower the wave tanks, clears creeps, or walks to the front.
    fn push(&self) -> Option<StructuredAction> {
        let post = self.lane.post;
        // A hero ahead of its wave, or bleeding, is being focused by creeps or the tower.
        if self.bleeding > 0 || self.lane.ahead(self.hero.pos) || self.tower_shy_here() {
            return self.move_towards(post, false);
        }
        self.lane
            .enemy_tower
            .filter(|_| !self.tower_shy)
            .and_then(|tower| self.hit_tower(tower))
            .or_else(|| self.farm(None))
            .or_else(|| self.move_towards(post, true))
    }

    fn tower_shy_here(&self) -> bool {
        self.tower_shy && enemy_tower_danger(self.tracker, self.hero.pos, self.hero.bound)
    }

    fn hit_tower(&self, tower: &UnitView) -> Option<StructuredAction> {
        let tanks = self
            .tracker
            .current()?
            .units
            .iter()
            .filter(|unit| {
                unit.team == self.tracker.team()
                    && is_lane_creep(unit.kind)
                    && unit.hp > 0
                    && unit
                        .pos
                        .within(tower.pos, Fixed::from_int(700) + tower.bound)
            })
            .count();
        if tanks < TOWER_TANKS {
            return None;
        }
        let (hero, space) = (self.hero, self.space);
        let target = space.entity_index(tower.id)?;
        if hero.mana >= HARASS_MANA_RESERVE
            && let Some((slot, _, _)) = ready_razes(hero, space).find(|(_, reach, _)| {
                raze_center(hero.pos, hero.facing.brads, *reach)
                    .within(tower.pos, Fixed::from_int(SHADOWRAZE_RADIUS))
            })
            && space.allows(cast_at(slot, target))
        {
            return Some(cast_at(slot, target));
        }
        attack(space, target)
    }

    /// Last hits first; with no enemy hero around also hits the weakest creep in reach.
    fn farm(&self, enemy: Option<&UnitView>) -> Option<StructuredAction> {
        let (tracker, space, hero) = (self.tracker, self.space, self.hero);
        let safe = |creep: &UnitView| {
            enemy.is_none_or(|enemy| {
                !creep
                    .pos
                    .within(enemy.pos, Fixed::from_int(THREAT_DISTANCE))
            }) && !enemy_tower_danger(tracker, creep.pos, Fixed::ZERO)
        };
        if let Some(target) = best_attack_creep(tracker, space, EntityRelation::Enemy, false)
            && safe(space.entity_candidates()[target.0].unit())
        {
            return attack(space, target);
        }
        if enemy.is_some() {
            return None;
        }
        // Switching targets mid-swing throws the swing away.
        if let Some(Order::Attack {
            target: Target::Unit(current),
        }) = self.active
            && let Some(index) = space.entity_index(current)
            && let creep = space.entity_candidates()[index.0].unit()
            && is_lane_creep(creep.kind)
            && creep.team != tracker.team()
            && creep.hp > 0
            && in_attack_reach(hero, creep)
            && safe(creep)
        {
            return Some(StructuredAction::Continue);
        }
        let mask = space.attack_entity_mask(ControlledUnit::Hero);
        space
            .entity_candidates()
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                mask.get(*index) == Some(&true)
                    && candidate.relation == EntityRelation::Enemy
                    && is_lane_creep(candidate.kind)
                    && in_attack_reach(hero, candidate.unit())
                    && safe(candidate.unit())
            })
            .min_by_key(|(_, candidate)| (candidate.unit().hp, candidate.unit().id))
            .and_then(|(index, _)| attack(space, EntityIndex(index)))
    }

    /// Walks (or attack-walks) to the safe candidate nearest `goal`, if that makes progress.
    fn move_towards(&self, goal: Vec2, attack: bool) -> Option<StructuredAction> {
        if distance(self.hero.pos, goal) < 60 {
            return None;
        }
        self.walk(|position| distance(position, goal), attack)
    }

    /// Walks to the safe candidate scoring lowest, if that improves on standing still.
    ///
    /// Re-targeting every decision is deliberate: keeping a running walk measurably kept the
    /// hero out of Teacher's raze bait and cut kill wins from 116 to 13 in 200 games.
    fn walk(&self, score: impl Fn(Vec2) -> i32, attack: bool) -> Option<StructuredAction> {
        let (best, point) = self.safe_point(&score)?;
        if best + PROGRESS_SLACK >= score(self.hero.pos) {
            return None;
        }
        let action = if attack {
            StructuredAction::AttackMovePoint {
                unit: ControlledUnit::Hero,
                point,
            }
        } else {
            StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point,
            }
        };
        self.space.allows(action).then_some(action)
    }

    /// Lowest-scoring move candidate that never walks into enemy tower range.
    fn safe_point(&self, score: impl Fn(Vec2) -> i32) -> Option<(i32, PointIndex)> {
        let space = self.space;
        let moves = space.move_point_mask(ControlledUnit::Hero);
        let attacks = space.attack_move_point_mask(ControlledUnit::Hero);
        space
            .point_candidates()
            .iter()
            .enumerate()
            .filter(|(index, point)| {
                moves.get(*index) == Some(&true)
                    && attacks.get(*index) == Some(&true)
                    && self.point_safe(point.position)
            })
            .map(|(index, point)| (score(point.position), PointIndex(index)))
            .min()
    }

    fn point_safe(&self, position: Vec2) -> bool {
        let (tracker, hero) = (self.tracker, self.hero);
        tower_corridor_safe(tracker, hero, position, true)
            && !enemy_tower_danger(
                tracker,
                position,
                hero.bound + Fixed::from_int(TOWER_MARGIN),
            )
    }
}

/// Raze levels first; Requiem, Necromastery and Presence only take otherwise wasted points.
fn learn(hero: &UnitView, space: &ActionSpace) -> Option<StructuredAction> {
    let mask = space.learn_slot_mask();
    for wanted in [SHADOWRAZES[0].0, REQUIEM, NECROMASTERY, PRESENCE] {
        for (index, ability) in hero.abilities.iter().enumerate() {
            let same = ability.id == wanted
                || wanted == SHADOWRAZES[0].0 && raze_reach(ability.id).is_some();
            if same && mask.get(index) == Some(&true) {
                return Some(StructuredAction::Learn {
                    slot: AbilitySlot(index as u8),
                });
            }
        }
    }
    None
}

/// Damage and count of every ready, affordable raze landing in sequence, plus one attack.
fn own_burst(space: &ActionSpace, hero: &UnitView, enemy: &UnitView) -> (i32, usize) {
    let stacks = enemy
        .effects
        .iter()
        .find(|effect| effect.id == RAZE_STACK_EFFECT)
        .and_then(|effect| effect.stacks)
        .unwrap_or(0);
    let (mut mana, mut landed, mut damage, mut razes) = (hero.mana, stacks, 0i32, 0usize);
    for (_, _, level) in ready_razes(hero, space) {
        let index = usize::from(level.clamp(1, 4) - 1);
        if mana < RAZE_MANA[index] {
            break;
        }
        mana -= RAZE_MANA[index];
        let raw = raze_damage(level).saturating_add(
            RAZE_STACK_DAMAGE[index].saturating_mul(i32::try_from(landed).unwrap_or(i32::MAX)),
        );
        damage = damage.saturating_add(magical_damage(raw, enemy.magic_resist));
        landed = landed.saturating_add(1);
        razes += 1;
    }
    let attack = physical_damage(hero.attack_damage.max(0), enemy.armor);
    (damage.saturating_add(attack), razes)
}

/// Whether the enemy lacks the mana for even one raze at its level.
fn unarmed(enemy: &UnitView) -> bool {
    let index = usize::from(enemy.level.saturating_add(1) / 2).clamp(1, 4) - 1;
    enemy.mana < RAZE_MANA[index]
}

/// Casts the ready raze whose disk best contains the enemy's next position.
///
/// A raze fires along the facing at resolution time, and a turning hero sweeps up to
/// 32 degrees a tick, so the facing must already point at the target, not merely cover it.
fn raze_now(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
    enemy: &UnitView,
) -> Option<StructuredAction> {
    let target = predicted_position(tracker, enemy);
    if facing_gap(hero.facing.brads, facing_towards(hero.pos, target)) > AIM_TOLERANCE_BRADS {
        return None;
    }
    let radius = raze_hit_radius(hero, enemy);
    let index = space.entity_index(enemy.id)?;
    ready_razes(hero, space)
        .filter_map(|(slot, reach, level)| {
            let center = raze_center(hero.pos, hero.facing.brads, reach);
            (magical_damage(raze_damage(level), enemy.magic_resist) > 0
                && center.within(target, radius)
                && space.allows(cast_at(slot, index)))
            .then_some((center.distance_squared(target), slot))
        })
        .min()
        .map(|(_, slot)| cast_at(slot, index))
}

fn attack(space: &ActionSpace, target: EntityIndex) -> Option<StructuredAction> {
    let action = StructuredAction::AttackUnit {
        unit: ControlledUnit::Hero,
        target,
    };
    space.allows(action).then_some(action)
}

fn ready_razes<'a>(
    hero: &'a UnitView,
    space: &'a ActionSpace,
) -> impl Iterator<Item = (usize, i32, u8)> + 'a {
    hero.abilities
        .iter()
        .enumerate()
        .filter_map(move |(slot, ability)| {
            let reach = raze_reach(ability.id)?;
            let ready = u8::try_from(slot)
                .is_ok_and(|slot| space.cast_ready(ControlledUnit::Hero, AbilitySlot(slot)));
            (ability.level > 0 && ready).then_some((slot, reach, ability.level))
        })
}

/// Raze disk radius less both bodies' movement before the cast resolves.
fn raze_hit_radius(hero: &UnitView, enemy: &UnitView) -> Fixed {
    let step = |unit: &UnitView| unit.move_speed.raw.max(0) / 30;
    Fixed {
        raw: Fixed::from_int(SHADOWRAZE_RADIUS)
            .raw
            .saturating_sub(step(enemy))
            .saturating_sub(step(hero))
            .max(0),
    }
}

/// The enemy hero last seen near `from` a moment ago; fog does not make it leave.
fn remembered_enemy_hero(tracker: &StateTracker, from: Vec2) -> Option<&UnitView> {
    let tick = tracker.current()?.tick;
    tracker
        .entities()
        .iter()
        .filter(|track| {
            track.unit.kind == UnitKind::Hero
                && track.unit.team != tracker.team()
                && track.unit.team != Team::Neutral
                && !track.visible
                && track
                    .last_death
                    .is_none_or(|death| death.tick < track.last_seen_tick)
                && tick.saturating_sub(track.last_seen_tick) <= ENEMY_MEMORY_TICKS
                && from.within(track.unit.pos, Fixed::from_int(ENGAGE_DISTANCE))
        })
        .min_by_key(|track| (from.distance_squared(track.unit.pos), track.id))
        .map(|track| &track.unit)
}

fn nearest_enemy_hero(tracker: &StateTracker, from: Vec2, within: i32) -> Option<&UnitView> {
    tracker
        .current()?
        .units
        .iter()
        .filter(|unit| {
            unit.kind == UnitKind::Hero
                && unit.team != tracker.team()
                && unit.team != Team::Neutral
                && unit.hp > 0
                && from.within(unit.pos, Fixed::from_int(within))
        })
        .min_by_key(|unit| (from.distance_squared(unit.pos), unit.id))
}

fn distance(from: Vec2, to: Vec2) -> i32 {
    let raw = isqrt(from.distance_squared(to).max(0) as u64);
    (raw / Fixed::ONE.raw as u64).min(i32::MAX as u64) as i32
}

const _: () = assert!(HOVER_DISTANCE > SHADOWRAZES[2].1 + SHADOWRAZE_RADIUS);
const _: () = assert!(HOVER_DISTANCE < 1_200);
const _: () = assert!(POST_TOWER_CLEARANCE > TOWER_REACH);
const _: () = assert!(THREAT_DISTANCE < HOVER_DISTANCE);
const _: () = assert!(RETREAT_PERCENT < ENGAGE_HEALTH_PERCENT);
const _: () = assert!(ENGAGE_HEALTH_PERCENT < RETURN_PERCENT);
