//! Aimed Shadowraze.
//!
//! A raze takes no aim on the wire: it lands at its fixed reach along the
//! caster's facing. [`RazeAim`] expands one aimed raze decision into a short
//! deterministic macro of legal wire orders: a short walk toward the aim turns
//! the hero, and the cast goes out once the landing predicted for the next tick
//! is on the aim. A unit aim tracks the unit's predicted position until the
//! landing covers it; a point aim fixes the heading toward the point when the
//! decision is made, so the landing centre lands as close to the point as the
//! fixed reach allows.

use bota_proto::{AbilityId, AbilitySlot, EntityId, Fixed, Order, Target, UnitView, Vec2};

use crate::{ActionKind, IssuedOrder, StateTracker};

/// Shadowraze near, mid and far with the distance each lands from the caster.
pub(crate) const SHADOWRAZES: [(AbilityId, i32); 3] = [
    (AbilityId(13), 200),
    (AbilityId(14), 450),
    (AbilityId(15), 700),
];
/// Every hostile unit whose centre lies within this of the landing is struck.
pub(crate) const SHADOWRAZE_RADIUS: i32 = 250;
/// Hero turn rate in brads per tick.
pub(crate) const TURN_RATE_BRADS: u16 = 5_795;
/// Length of the walk that turns the hero. Long enough to finish any turn that
/// reached walking tolerance, short enough that the hero stops by itself.
const TURN_WALK: i32 = 48;
/// An aim that has not fired within this many ticks is abandoned.
const AIM_LIMIT_TICKS: u32 = 15;
/// Length of the walk that turns the hero onto a point aim's heading. One walk
/// node: longer walks route through node centres and leave the facing up to
/// tens of degrees off the heading, while this one ends on it exactly.
const HEADING_WALK: i32 = 32;
/// A point aim fires once the landing centre is this close to the centre along
/// the chosen heading: a tenth of the raze radius.
const HEADING_TOLERANCE: i32 = 25;

const _: () = assert!(TURN_WALK > 0);
const _: () = assert!(AIM_LIMIT_TICKS >= 2 * crate::MAP2_DECISION_INTERVAL_TICKS);
const _: () = assert!(HEADING_WALK > 0);
const _: () = assert!(HEADING_TOLERANCE > 0 && HEADING_TOLERANCE < SHADOWRAZE_RADIUS);

/// Bounded state of one hero's aimed-raze macro.
///
/// Continue decisions keep an active aim running; any other decision replaces it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RazeAim {
    active: Option<AimPlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AimPlan {
    hero: EntityId,
    slot: AbilitySlot,
    reach: i32,
    target: AimTarget,
    started: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AimTarget {
    /// A visible unit, followed until the landing covers its predicted position.
    Unit(EntityId),
    /// A facing in brads, taken toward the chosen point when the decision is made.
    Heading(u16),
}

impl RazeAim {
    /// Translates one decoded decision into the wire order to send now.
    ///
    /// An aimed raze intent (a unit or point cast) starts the macro and a
    /// Continue advances it; an untargeted raze passes through unchanged. Both
    /// report [`ActionKind::Cast`] as the order's family. `active_body` is the
    /// hero's persistent order, which decides how its facing moves next tick.
    pub fn resolve(
        &mut self,
        tracker: &StateTracker,
        decoded: Option<IssuedOrder>,
        kind: ActionKind,
        active_body: Option<IssuedOrder>,
    ) -> Option<(IssuedOrder, ActionKind)> {
        assert_eq!(decoded.is_none(), kind == ActionKind::Continue);
        if let Some(issued) = decoded {
            self.active = aim_plan(tracker, issued);
            if self.active.is_none() {
                return Some((issued, kind));
            }
        }
        let order = self.step(tracker, active_body)?;
        Some((IssuedOrder { unit: None, order }, ActionKind::Cast))
    }

    fn step(&mut self, tracker: &StateTracker, active_body: Option<IssuedOrder>) -> Option<Order> {
        let plan = self.active?;
        let order = aim_order(tracker, plan, active_body);
        if !matches!(order, Some(Order::Move { .. })) {
            self.active = None;
        }
        order
    }
}

/// Whether a hostile unit at `to` lies in the band a raze of `reach` can cover from `from`.
pub(crate) fn within_reach_window(from: Vec2, to: Vec2, reach: i32) -> bool {
    assert!(reach > 0);
    // One unit of slack absorbs the fixed-point rounding of the landing point.
    let outer = i64::from(reach + SHADOWRAZE_RADIUS + 1);
    let inner = i64::from((reach - SHADOWRAZE_RADIUS - 1).max(0));
    let distance = from.distance_squared(to);
    let scale = i64::from(Fixed::ONE.raw);
    distance <= (outer * scale).pow(2) && distance >= (inner * scale).pow(2)
}

fn aim_plan(tracker: &StateTracker, issued: IssuedOrder) -> Option<AimPlan> {
    let Order::Cast { slot, target } = issued.order else {
        return None;
    };
    let hero = tracker.own_hero().filter(|_| issued.unit.is_none())?;
    let reach = raze_reach(hero.abilities.get(usize::from(slot.0))?.id)?;
    let target = match target {
        Target::Unit(unit) => AimTarget::Unit(unit),
        Target::Pos(point) => {
            assert_ne!(point, hero.pos, "a raze point aim needs a heading");
            AimTarget::Heading(facing_towards(hero.pos, point))
        }
        Target::None => return None,
    };
    Some(AimPlan {
        hero: hero.id,
        slot,
        reach,
        target,
        started: tracker.current()?.tick,
    })
}

/// The next macro order: a turning walk, the cast, or nothing once the aim is hopeless.
fn aim_order(
    tracker: &StateTracker,
    plan: AimPlan,
    active_body: Option<IssuedOrder>,
) -> Option<Order> {
    let view = tracker.current()?;
    if view.tick.checked_sub(plan.started)? > AIM_LIMIT_TICKS {
        return None;
    }
    let hero = tracker
        .own_hero()
        .filter(|hero| hero.id == plan.hero && alive(hero))?;
    let ability = hero.abilities.get(usize::from(plan.slot.0))?;
    if raze_reach(ability.id) != Some(plan.reach) || !crate::action::ability_ready(hero, ability) {
        return None;
    }
    let origin = predicted_position(tracker, hero);
    // The aim point the landing must reach, how close it must get, and where to walk.
    let (aim, tolerance, walk_goal) = match plan.target {
        AimTarget::Unit(target) => {
            let target = visible_unit(tracker, target).filter(|unit| alive(unit))?;
            if !within_reach_window(hero.pos, target.pos, plan.reach) {
                return None;
            }
            let aim = predicted_position(tracker, target);
            let walk = point_along(hero.pos, aim, Fixed::from_int(TURN_WALK));
            (aim, raze_radius(target), walk)
        }
        AimTarget::Heading(heading) => (
            raze_center(origin, heading, plan.reach),
            Fixed::from_int(HEADING_TOLERANCE),
            raze_center(hero.pos, heading, HEADING_WALK),
        ),
    };
    // The hero may keep its facing or turn one tick toward its order's goal before
    // the cast goes off; only a landing on the aim either way is taken.
    let covered = [
        hero.facing.brads,
        predicted_facing(tracker, hero, active_body),
    ]
    .into_iter()
    .all(|facing| raze_center(origin, facing, plan.reach).within(aim, tolerance));
    if covered {
        return Some(Order::Cast {
            slot: plan.slot,
            target: Target::None,
        });
    }
    Some(Order::Move {
        target: Target::Pos(walk_goal),
    })
}

/// Facing after one more tick of the hero turning toward its persistent order's goal.
fn predicted_facing(
    tracker: &StateTracker,
    hero: &UnitView,
    active_body: Option<IssuedOrder>,
) -> u16 {
    let goal = active_body
        .filter(|issued| issued.unit.is_none())
        .and_then(|issued| match issued.order {
            Order::Move { target } | Order::Attack { target } => match target {
                Target::Pos(position) => Some(position),
                Target::Unit(unit) => visible_unit(tracker, unit).map(|unit| unit.pos),
                Target::None => None,
            },
            _ => None,
        });
    match goal {
        Some(goal) if goal != hero.pos => turn_towards(
            hero.facing.brads,
            facing_towards(hero.pos, goal),
            TURN_RATE_BRADS,
        ),
        _ => hero.facing.brads,
    }
}

/// One tick of turning, clamped by the rate; an exact reversal turns counter-clockwise.
fn turn_towards(from: u16, to: u16, rate: u16) -> u16 {
    let delta = i32::from(to.wrapping_sub(from) as i16);
    let delta = if delta == -32_768 { 32_768 } else { delta };
    let clamped = delta.clamp(-i32::from(rate), i32::from(rate));
    (i32::from(from) + clamped) as u16
}

fn visible_unit(tracker: &StateTracker, id: EntityId) -> Option<&UnitView> {
    let units = &tracker.current()?.units;
    units
        .binary_search_by_key(&id, |unit| unit.id)
        .ok()
        .map(|index| &units[index])
}

fn alive(unit: &UnitView) -> bool {
    unit.hp > 0
}

pub(crate) fn predicted_position(tracker: &StateTracker, unit: &UnitView) -> Vec2 {
    extrapolated_position(tracker, unit, 1)
}

/// Where `unit` would be after `ticks` more ticks at its last tracked velocity,
/// no faster than its move speed and never off the map.
pub(crate) fn extrapolated_position(tracker: &StateTracker, unit: &UnitView, ticks: u32) -> Vec2 {
    let Some(velocity) = tracker.entity(unit.id).and_then(|entity| entity.velocity) else {
        return unit.pos;
    };
    assert!(velocity.elapsed_ticks > 0);
    let divisor = i64::from(velocity.elapsed_ticks);
    let per_tick = (
        i64::from(velocity.delta.x.raw) / divisor,
        i64::from(velocity.delta.y.raw) / divisor,
    );
    let speed = isqrt(per_tick.0.unsigned_abs().pow(2) + per_tick.1.unsigned_abs().pow(2)) as i64;
    let step = speed
        .min(i64::from(unit.move_speed.raw.max(0) / 30))
        .saturating_mul(i64::from(ticks))
        .min(i64::from(i32::MAX));
    let maximum = crate::tracker::map_maximum_raw(tracker.metadata().terrain_cells);
    assert!(maximum > 0);
    let ticks = i64::from(ticks);
    let axis = |position: i32, delta: i64| -> i32 {
        (i64::from(position) + delta.saturating_mul(ticks)).clamp(0, maximum) as i32
    };
    let target = Vec2 {
        x: Fixed {
            raw: axis(unit.pos.x.raw, per_tick.0),
        },
        y: Fixed {
            raw: axis(unit.pos.y.raw, per_tick.1),
        },
    };
    let available = isqrt(unit.pos.distance_squared(target) as u64).min(i32::MAX as u64) as i64;
    let predicted = point_along(
        unit.pos,
        target,
        Fixed {
            raw: step.min(available) as i32,
        },
    );
    assert!((0..=maximum).contains(&i64::from(predicted.x.raw)));
    assert!((0..=maximum).contains(&i64::from(predicted.y.raw)));
    predicted
}

/// Raze radius shrinks with the target's move speed uncertainty.
pub(crate) fn raze_radius(unit: &UnitView) -> Fixed {
    Fixed {
        raw: Fixed::from_int(SHADOWRAZE_RADIUS)
            .raw
            .saturating_sub(unit.move_speed.raw.max(0) / 30)
            .max(0),
    }
}

pub(crate) fn raze_contains(
    tracker: &StateTracker,
    hero: &UnitView,
    target: &UnitView,
    facing: u16,
    reach: i32,
) -> bool {
    let radius = raze_radius(target);
    raze_center(hero.pos, facing, reach).within(predicted_position(tracker, target), radius)
}

pub(crate) fn raze_reach(id: AbilityId) -> Option<i32> {
    SHADOWRAZES
        .iter()
        .find_map(|(raze, reach)| (*raze == id).then_some(*reach))
}

pub(crate) fn raze_center(position: Vec2, facing: u16, distance: i32) -> Vec2 {
    let ahead = position + heading_of(facing);
    point_along(position, ahead, Fixed::from_int(distance))
}

pub(crate) fn facing_towards(from: Vec2, to: Vec2) -> u16 {
    let dx = i64::from(to.x.raw) - i64::from(from.x.raw);
    let dy = i64::from(to.y.raw) - i64::from(from.y.raw);
    if dx == 0 && dy == 0 {
        return 0;
    }
    let (absolute_x, absolute_y) = (dx.abs(), dy.abs());
    let slope = if absolute_x >= absolute_y {
        (absolute_y << 13) / absolute_x
    } else {
        (absolute_x << 13) / absolute_y
    };
    let octant = match (dx >= 0, dy >= 0, absolute_x >= absolute_y) {
        (true, true, true) => slope,
        (true, true, false) => 16_384 - slope,
        (false, true, false) => 16_384 + slope,
        (false, true, true) => 32_768 - slope,
        (false, false, true) => 32_768 + slope,
        (false, false, false) => 49_152 - slope,
        (true, false, false) => 49_152 + slope,
        (true, false, true) => 65_536 - slope,
    };
    (octant & 0xffff) as u16
}

fn heading_of(facing: u16) -> Vec2 {
    let brads = i32::from(facing);
    let slope = brads % 8_192;
    let (x, y) = match brads / 8_192 {
        0 => (8_192, slope),
        1 => (8_192 - slope, 8_192),
        2 => (-slope, 8_192),
        3 => (-8_192, 8_192 - slope),
        4 => (-8_192, -slope),
        5 => (-(8_192 - slope), -8_192),
        6 => (slope, -8_192),
        _ => (8_192, -(8_192 - slope)),
    };
    Vec2::from_ints(x, y)
}

pub(crate) fn point_along(from: Vec2, towards: Vec2, distance: Fixed) -> Vec2 {
    let x = i64::from(towards.x.raw) - i64::from(from.x.raw);
    let y = i64::from(towards.y.raw) - i64::from(from.y.raw);
    let span = isqrt((x * x + y * y) as u64) as i64;
    if span == 0 {
        return from;
    }
    Vec2 {
        x: Fixed {
            raw: from
                .x
                .raw
                .saturating_add((x * i64::from(distance.raw) / span) as i32),
        },
        y: Fixed {
            raw: from
                .y
                .raw
                .saturating_add((y * i64::from(distance.raw) / span) as i32),
        },
    }
}

pub(crate) fn isqrt(value: u64) -> u64 {
    let mut remainder = value;
    let mut root = 0_u64;
    let mut bit = 1_u64 << 62;
    for _ in 0..32 {
        if remainder >= root.saturating_add(bit) {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}
