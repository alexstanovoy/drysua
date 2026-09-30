//! Seat-visible Shadow Fiend combat estimates shared by the rule policies.

use bota_proto::{AbilitySlot, Fixed, UnitKind, UnitView, Vec2};

use crate::feature::attack_interval_ticks;
use crate::raze_aim::{TURN_RATE_BRADS, facing_towards, isqrt};
use crate::teacher_economy::attack_damage_against;
use crate::{
    ActionSpace, ActionTarget, ControlledUnit, EntityIndex, EntityRelation, StateTracker,
    StructuredAction,
};

const SHADOWRAZE_DAMAGE: [i32; 4] = [90, 160, 230, 300];
const ARMOR_SCALE: i64 = 6;
pub(crate) const ATTACK_POINT_TICKS: u32 = 15;
pub(crate) const ATTACK_PROJECTILE_UNITS_PER_TICK: i32 = 40;
const ATTACK_ANGLE_BRADS: u16 = 2_094;
/// Ticks between two decisions of a rule policy.
pub(crate) const DECISION_TICKS: u32 = 3;
const COLLISION_MARGIN: i32 = DECISION_TICKS as i32 * 4;

pub(crate) fn movement_guard(unit: &UnitView) -> Fixed {
    Fixed {
        raw: (unit.move_speed.raw.max(0) / 30)
            .saturating_mul(DECISION_TICKS as i32)
            .saturating_add(Fixed::from_int(COLLISION_MARGIN + 1).raw),
    }
}

pub(crate) fn tower_corridor_safe(
    tracker: &StateTracker,
    hero: &UnitView,
    target: Vec2,
    navigation: bool,
) -> bool {
    let horizontal = i128::from(target.x.raw) - i128::from(hero.pos.x.raw);
    let vertical = i128::from(target.y.raw) - i128::from(hero.pos.y.raw);
    let squared = horizontal * horizontal + vertical * vertical;
    // Wave-supported structure work remains possible, but never excuses hero pursuit.
    let pushing = navigation
        && allied_creep_near(tracker, target, 750)
        && !enemy_heroes(tracker).any(|enemy| {
            hero.pos.within(enemy.pos, Fixed::from_int(1_200))
                || target.within(enemy.pos, Fixed::from_int(1_200))
        });
    tracker.current().is_some_and(|view| {
        view.units
            .iter()
            .filter(|unit| {
                unit.kind == UnitKind::Tower && unit.team != tracker.team() && unit.hp > 0
            })
            .all(|tower| {
                let radius = Fixed::from_int(700) + hero.bound + tower.bound + movement_guard(hero);
                let outward = (i128::from(hero.pos.x.raw) - i128::from(tower.pos.x.raw))
                    * horizontal
                    + (i128::from(hero.pos.y.raw) - i128::from(tower.pos.y.raw)) * vertical;
                if navigation && hero.pos.within(tower.pos, radius) && squared > 0 && outward >= 0 {
                    return true;
                }
                if pushing && target.within(tower.pos, radius) {
                    return true;
                }
                // Endpoints alone miss a chase corridor that cuts through tower range.
                let projection = ((i128::from(tower.pos.x.raw) - i128::from(hero.pos.x.raw))
                    * horizontal
                    + (i128::from(tower.pos.y.raw) - i128::from(hero.pos.y.raw)) * vertical)
                    .clamp(0, squared);
                let closest = Vec2 {
                    x: Fixed {
                        raw: (i128::from(hero.pos.x.raw) + horizontal * projection / squared.max(1))
                            as i32,
                    },
                    y: Fixed {
                        raw: (i128::from(hero.pos.y.raw) + vertical * projection / squared.max(1))
                            as i32,
                    },
                };
                !closest.within(tower.pos, radius)
            })
    })
}

pub(crate) fn magical_damage(amount: i32, resistance: Fixed) -> i32 {
    let kept = i64::from(
        Fixed::ONE
            .raw
            .saturating_sub(resistance.raw)
            .clamp(0, Fixed::ONE.raw),
    );
    (i64::from(amount) * kept / i64::from(Fixed::ONE.raw)) as i32
}

pub(crate) fn physical_damage(amount: i32, armor: Fixed) -> i32 {
    let armor_raw = i64::from(armor.raw.max(0));
    let whole = i64::from(Fixed::ONE.raw);
    let denominator = 100 * whole + ARMOR_SCALE * armor_raw;
    (i64::from(amount) * 100 * whole / denominator) as i32
}

pub(crate) fn ratio_at_most(value: i32, maximum: i32, percent: i32) -> bool {
    maximum > 0 && i64::from(value) * 100 <= i64::from(maximum) * i64::from(percent)
}

pub(crate) fn ratio_below(value: i32, maximum: i32, percent: i32) -> bool {
    maximum <= 0 || i64::from(value) * 100 < i64::from(maximum) * i64::from(percent)
}

pub(crate) fn best_attack_creep(
    tracker: &StateTracker,
    space: &ActionSpace,
    relation: EntityRelation,
    deny: bool,
) -> Option<EntityIndex> {
    let hero = tracker.own_hero()?;
    let mask = space.attack_entity_mask(ControlledUnit::Hero);
    space
        .entity_candidates()
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let unit = candidate.unit();
            let neutral = !deny
                && relation == EntityRelation::Enemy
                && candidate.kind == UnitKind::CreepNeutral
                && matches!(
                    candidate.relation,
                    EntityRelation::Enemy | EntityRelation::Neutral
                );
            if mask.get(index) != Some(&true)
                || !(neutral || candidate.relation == relation && is_lane_creep(candidate.kind))
                || !in_attack_reach(hero, unit)
                || (deny && unit.hp.saturating_mul(2) >= unit.max_hp)
            {
                return None;
            }
            let landing = attack_landing_ticks(hero, unit);
            let predicted = predicted_hp(tracker, unit, landing);
            let hit = physical_damage(attack_damage_against(hero, unit), unit.armor);
            (predicted > 0 && predicted <= hit).then_some((predicted, index))
        })
        .min()
        .map(|(_, index)| EntityIndex(index))
}

pub(crate) fn predicted_hp(tracker: &StateTracker, unit: &UnitView, ticks: u32) -> i32 {
    let Some(track) = tracker.entity(unit.id) else {
        return unit.hp;
    };
    let elapsed = track
        .last_seen_tick
        .saturating_sub(track.previous_seen_tick);
    if elapsed == 0 || track.hp_delta >= 0 {
        return unit.hp;
    }
    let change = track.hp_delta.saturating_mul(i64::from(ticks)) / i64::from(elapsed);
    i64::from(unit.hp)
        .saturating_add(change)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

pub(crate) fn attack_landing_ticks(hero: &UnitView, target: &UnitView) -> u32 {
    let turn = attack_turn_ticks(hero, target);
    let distance_raw = isqrt(hero.pos.distance_squared(target.pos) as u64);
    let distance = distance_raw.div_ceil(u64::from(Fixed::ONE.raw as u32));
    let travel = distance.div_ceil(ATTACK_PROJECTILE_UNITS_PER_TICK as u64) as u32;
    attack_interval_ticks(hero.attack_time)
        .saturating_add(turn)
        .saturating_add(ATTACK_POINT_TICKS)
        .saturating_add(travel)
}

/// Ticks needed to turn a hero's facing into attack alignment with a target.
pub(crate) fn attack_turn_ticks(hero: &UnitView, target: &UnitView) -> u32 {
    let gap = u32::from(facing_gap(
        hero.facing.brads,
        facing_towards(hero.pos, target.pos),
    ));
    gap.saturating_sub(u32::from(ATTACK_ANGLE_BRADS))
        .div_ceil(u32::from(TURN_RATE_BRADS))
}

pub(crate) fn in_attack_reach(hero: &UnitView, target: &UnitView) -> bool {
    hero.pos
        .within(target.pos, hero.attack_range + hero.bound + target.bound)
}

pub(crate) fn enemy_tower_danger(tracker: &StateTracker, position: Vec2, radius: Fixed) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.kind == UnitKind::Tower
                && unit.team != tracker.team()
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(700) + radius + unit.bound)
        })
    })
}

pub(crate) fn allied_creep_near(tracker: &StateTracker, position: Vec2, radius: i32) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.team == tracker.team()
                && is_lane_creep(unit.kind)
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(radius))
        })
    })
}

pub(crate) fn is_lane_creep(kind: UnitKind) -> bool {
    matches!(
        kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
    )
}

pub(crate) fn facing_gap(one: u16, other: u16) -> u16 {
    let clockwise = one.wrapping_sub(other);
    let counterclockwise = other.wrapping_sub(one);
    clockwise.min(counterclockwise)
}

pub(crate) fn raze_damage(level: u8) -> i32 {
    SHADOWRAZE_DAMAGE[usize::from(level.clamp(1, 4) - 1)]
}

pub(crate) fn enemy_heroes(tracker: &StateTracker) -> impl Iterator<Item = &UnitView> {
    tracker.current().into_iter().flat_map(|view| {
        view.units.iter().filter(|unit| {
            unit.kind == UnitKind::Hero && unit.team != tracker.team() && unit.hp > 0
        })
    })
}

pub(crate) fn own_fountain(tracker: &StateTracker) -> Option<Vec2> {
    tracker.current()?.units.iter().find_map(|unit| {
        (unit.kind == UnitKind::Fountain && unit.team == tracker.team()).then_some(unit.pos)
    })
}

/// An aimed raze; the aim macro casts it at once when the current facing already covers the target.
pub(crate) fn cast_at(slot: usize, target: EntityIndex) -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot as u8),
        target: ActionTarget::Entity(target),
    }
}
