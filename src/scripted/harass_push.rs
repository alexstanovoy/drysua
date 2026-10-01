//! Shadow Fiend rule policy that holds the lane against Teacher and pushes its tower.
//!
//! Teacher never walks into tower range while an enemy hero stands within 1200, so presence
//! is the tower's defense, and Teacher stays in lane down to 40% health. HarassPush therefore:
//! - keeps four Healing Salves and four Clarities, restocked at the home shop, and counts an
//!   active salve as health when deciding to stop walking home;
//! - hovers at 1200 from Teacher, outside its raze reach and inside its push veto, and keeps
//!   a hero that vanished into fog where it was last seen;
//! - fights only for a kill, when Teacher lacks raze mana (attacking it then, since it can
//!   only answer with attacks), or when Teacher sieges our tower and a raze pair is ready;
//! - razes only once its facing has settled on the target (a turning hero sweeps 32 degrees a
//!   tick, and a raze fires along the facing at resolution time);
//! - with Teacher away, hits the tower once one of its own creeps tanks it;
//! - never idles in enemy tower range, backs off when bleeding, and walks home at 30%
//!   health.
//!
//! [`STYLE_KNOBS`] turns these choices into seeded styles for training opponents. Every
//! action goes through the drysua action space and every input is seat-visible, so the policy
//! is a behaviour-cloning target.

use bota_proto::{
    AbilityId, AbilitySlot, EffectId, EntityId, Fixed, ItemId, Order, Target, Team, UnitKind,
    UnitView, Vec2,
};

use crate::raze_aim::{
    SHADOWRAZE_RADIUS, SHADOWRAZES, facing_towards, isqrt, point_along, predicted_position,
    raze_center, raze_reach,
};
use crate::scripted::ScriptKind;
use crate::scripted::progress::GoalProgress;
use crate::scripted::style::{Knob, StyleValues};
use crate::scripted::tactics::{
    best_attack_creep, cast_at, enemy_tower_danger, facing_gap, in_attack_reach, is_lane_creep,
    magical_damage, physical_damage, ratio_at_most, ratio_below, raze_damage, tower_corridor_safe,
};
use crate::teacher_economy::select_sustain;
use crate::{
    ActionError, ActionSpace, ControlledUnit, EntityIndex, EntityRelation, IssuedOrder,
    ItemReadiness, OrderPersistence, PointIndex, ShopIndex, StateTracker, StructuredAction,
};

const REQUIEM: AbilityId = AbilityId(16);
const NECROMASTERY: AbilityId = AbilityId(17);
const PRESENCE: AbilityId = AbilityId(18);
const RAZE_STACK_EFFECT: EffectId = EffectId(15);
const RAZE_STACK_DAMAGE: [i32; 4] = [50, 60, 70, 80];
const RAZE_MANA: [i32; 4] = [75, 80, 85, 90];
/// Own health share below which the hero never starts a fight.
const ENGAGE_HEALTH_PERCENT: i32 = 50;
/// Farthest reach of an enemy raze disk plus one decision of approach.
const THREAT_DISTANCE: i32 = 1_000;
/// Enemy heroes farther than this cannot contest a push.
const ENGAGE_DISTANCE: i32 = 1_600;
const HEALING_SALVE: ItemId = ItemId(2);
const CLARITY: ItemId = ItemId(1);
/// Health a Healing Salve mends over its whole duration, and that duration.
const SALVE_HEAL: i32 = 400;
const SALVE_TICKS: u32 = 300;
const MENDING_EFFECT: EffectId = EffectId(1);
/// Consumables are bought only this deep inside shop range, so they land in the bag.
const SHOP_DISTANCE: i32 = 900;
/// Item slots the bag offers for consumables.
const ACTIVE_ITEM_SLOTS: usize = 6;
/// Score improvement below which a walk is not worth an order.
const PROGRESS_SLACK: i32 = 40;
/// Tower attack range (700) plus the tower (144) and hero (24) bounds.
const TOWER_REACH: i32 = 868;
/// Laning posts stay this far from the enemy tower centre, just beyond its reach, so a wave
/// that tanks the tower still keeps the hero close enough to join in.
const POST_TOWER_CLEARANCE: i32 = 950;
/// Extra distance kept outside enemy tower attack range while walking.
const TOWER_MARGIN: i32 = 150;
/// An enemy hero this close to the own tower is pushing it.
const TOWER_DEFENSE_DISTANCE: i32 = 1_300;
/// Distance kept behind the own front creep while laning.
const FRONT_OFFSET: i32 = 250;
/// How far past the lane post the hero may drift while fighting creeps.
const AHEAD_SLACK: i32 = 200;
/// Facing error (about 5.5 degrees) under which the hero has finished turning toward a target.
const AIM_TOLERANCE_BRADS: u16 = 1_000;
/// A tower keeps its target while it stays in range, so a focused hero stays out this long.
const TOWER_SHY_TICKS: u32 = 150;

/// Own health share that starts a walk home.
const RETREAT_KNOB: Knob = Knob::new("retreat", 30, (5, 60), (10, 40));
/// Own health share, counting an active salve, that ends it; hysteresis stops lane/fountain
/// oscillation.
const RETURN_KNOB: Knob = Knob::new("return", 80, (10, 100), (45, 90));
/// Distance kept from an enemy hero: outside its raze reach and at the edge of the 1200
/// within which Teacher does not walk into our tower's range.
const HOVER_KNOB: Knob = Knob::new("hover", 1_200, (600, 1_600), (900, 1_400));

/// Style knobs after the shared noise knobs; defaults are the canonical HarassPush.
pub(crate) const STYLE_KNOBS: [Knob; 9] = [
    RETREAT_KNOB,
    RETURN_KNOB,
    // Healing Salves and Clarities kept in the bag, restocked at the home shop.
    Knob::new("salves", 4, (0, 6), (0, 5)),
    Knob::new("clarities", 4, (0, 6), (0, 5)),
    HOVER_KNOB,
    // Also fights when the ready razes leave the enemy at or below this health share
    // (Teacher walks home at 40%); 0 fights only for kills, unarmed enemies and tower
    // defense.
    Knob::new("engage", 0, (0, 100), (0, 70)),
    // Own creeps that must tank the enemy tower before the hero hits it; 9 never pushes.
    Knob::new("tanks", 1, (0, 9), (1, 4)),
    // Ticks an enemy hero that vanished into fog still counts where it was last seen.
    Knob::new("memory", 120, (0, 600), (0, 300)),
    // 1 levels razes first; 0 takes Necromastery first.
    Knob::new("razes_first", 1, (0, 1), (0, 1)),
];

/// One game's HarassPush knobs, typed; see [`STYLE_KNOBS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HarassStyle {
    retreat_percent: i32,
    return_percent: i32,
    salves: i32,
    clarities: i32,
    hover_distance: i32,
    engage_percent: i32,
    tower_tanks: usize,
    memory_ticks: u32,
    razes_first: bool,
}

impl HarassStyle {
    /// Reads the HarassPush knobs of one drawn style.
    pub fn from_values(values: &StyleValues) -> Self {
        let get = |name| values.get(ScriptKind::HarassPush, name);
        Self {
            retreat_percent: get("retreat"),
            return_percent: get("return"),
            salves: get("salves"),
            clarities: get("clarities"),
            hover_distance: get("hover"),
            engage_percent: get("engage"),
            tower_tanks: usize::try_from(get("tanks")).expect("knob bounds"),
            memory_ticks: u32::try_from(get("memory")).expect("knob bounds"),
            razes_first: get("razes_first") == 1,
        }
    }
}

impl Default for HarassStyle {
    fn default() -> Self {
        Self::from_values(&StyleValues::canonical(ScriptKind::HarassPush))
    }
}

/// Deterministic HarassPush rule policy; see the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HarassPush {
    style: HarassStyle,
    hero: Option<EntityId>,
    retreating: bool,
    last_hp: i32,
    /// Health lost since the previous decision; creep and tower focus show up here first.
    bleeding: i32,
    tower_shy_until: u32,
}

impl HarassPush {
    pub const fn with_style(style: HarassStyle) -> Self {
        Self {
            style,
            hero: None,
            retreating: false,
            last_hp: 0,
            bleeding: 0,
            tower_shy_until: 0,
        }
    }

    /// Returns the selected action together with the action space it was selected in.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        let space = ActionSpace::from_tracker_with_readiness(tracker, readiness)?;
        let action = self.decide_in(tracker, persistence, &space)?;
        Ok((action, space))
    }

    /// Selects an action in `space`, the space this seat's tracker and readiness build.
    pub fn decide_in(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
    ) -> Result<StructuredAction, ActionError> {
        self.observe(tracker, space.tick());
        let active = persistence
            .active_body_for(None)
            .map(|(_, issued)| issued.order);
        let selected = self.select(tracker, space, active);
        if !space.allows(selected) {
            return Err(ActionError::InvalidSchema(
                "HarassPush selected masked action",
            ));
        }
        Ok(
            if persistence.should_send(space.decode(selected)?).is_none() {
                StructuredAction::Continue
            } else {
                selected
            },
        )
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
            *self = Self::with_style(self.style);
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
        let healed = hero.hp.saturating_add(pending_heal(hero));
        if ratio_at_most(hero.hp, hero.max_hp, self.style.retreat_percent) {
            self.retreating = true;
        } else if !ratio_below(healed, hero.max_hp, self.style.return_percent) {
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
            style: &self.style,
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
        let lurking =
            enemy.or_else(|| remembered_enemy_hero(tracker, hero.pos, self.style.memory_ticks));
        let action = learn(hero, space, self.style.razes_first)
            .or_else(|| turn.restock())
            // Uses sustain items; salves and clarities only when no hit should break the drink.
            .or_else(|| select_sustain(tracker, space, self.retreating, &GoalProgress::new()))
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
    style: &'a HarassStyle,
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
    /// Fights away from enemy towers while healthy when the ready razes kill the enemy or leave
    /// it at or below the `engage` share, when it lacks raze mana, or when it sieges our tower
    /// and a raze pair is ready.
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
        if left <= 0 || ratio_at_most(left, enemy.max_hp, self.style.engage_percent) {
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
                (self.style.hover_distance - gap).max(0) * 4 + distance(position, post) / 2
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

    /// Tops up Healing Salves, then Clarities, while standing in the home shop. Sustain keeps
    /// the hero in lane; without it every trade ends in a walk home while Teacher pushes.
    fn restock(&self) -> Option<StructuredAction> {
        let (hero, space) = (self.hero, self.space);
        if !hero
            .pos
            .within(self.lane.own_fountain, Fixed::from_int(SHOP_DISTANCE))
        {
            return None;
        }
        let mask = space.buy_mask(ControlledUnit::Hero);
        [
            (HEALING_SALVE, self.style.salves),
            (CLARITY, self.style.clarities),
        ]
        .into_iter()
        .filter(|(item, wanted)| carried(hero, *item) < *wanted)
        .find_map(|(item, _)| {
            let index = space
                .shop_candidates()
                .iter()
                .position(|candidate| candidate.item == item)?;
            (mask.get(index) == Some(&true)).then_some(StructuredAction::Buy {
                unit: ControlledUnit::Hero,
                item: ShopIndex(index),
            })
        })
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
        if tanks < self.style.tower_tanks {
            return None;
        }
        let target = self.space.entity_index(tower.id)?;
        attack(self.space, target)
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
    /// Re-targets every decision: keeping a running walk instead cost most kill wins in duels.
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

/// Raze levels first, or Necromastery first; the rest only take otherwise wasted points.
fn learn(hero: &UnitView, space: &ActionSpace, razes_first: bool) -> Option<StructuredAction> {
    let mask = space.learn_slot_mask();
    let order = if razes_first {
        [SHADOWRAZES[0].0, REQUIEM, NECROMASTERY, PRESENCE]
    } else {
        [NECROMASTERY, SHADOWRAZES[0].0, REQUIEM, PRESENCE]
    };
    for wanted in order {
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

/// Whether the enemy lacks the mana for one raze at the highest level its hero level allows.
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

/// Charges of `item` in the bag.
fn carried(hero: &UnitView, item: ItemId) -> i32 {
    hero.items
        .iter()
        .take(ACTIVE_ITEM_SLOTS)
        .flatten()
        .filter(|held| held.id == item)
        .map(|held| i32::from(held.charges.unwrap_or(1)))
        .sum()
}

/// Health an active salve still mends, taking any Mending effect for a salve; the hero counts
/// it as already healed.
fn pending_heal(hero: &UnitView) -> i32 {
    hero.effects
        .iter()
        .filter(|effect| effect.id == MENDING_EFFECT)
        .filter_map(|effect| effect.ticks_left)
        .map(|left| {
            let left = i32::try_from(left.min(SALVE_TICKS)).expect("bounded ticks");
            SALVE_HEAL * left / SALVE_TICKS as i32
        })
        .max()
        .unwrap_or(0)
}

/// The enemy hero last seen near `from` a moment ago; fog does not make it leave.
fn remembered_enemy_hero(tracker: &StateTracker, from: Vec2, memory: u32) -> Option<&UnitView> {
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
                && tick.saturating_sub(track.last_seen_tick) <= memory
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

const _: () = assert!(HOVER_KNOB.default > SHADOWRAZES[2].1 + SHADOWRAZE_RADIUS);
const _: () = assert!(HOVER_KNOB.default <= 1_200);
const _: () = assert!(POST_TOWER_CLEARANCE > TOWER_REACH);
const _: () = assert!(THREAT_DISTANCE < HOVER_KNOB.default);
const _: () = assert!(RETREAT_KNOB.default < ENGAGE_HEALTH_PERCENT);
const _: () = assert!(ENGAGE_HEALTH_PERCENT < RETURN_KNOB.default);
