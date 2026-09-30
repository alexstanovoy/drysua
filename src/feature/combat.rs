//! Seat-observable combat arithmetic behind the derived unit and global inputs.
//!
//! Every input here comes from the seat's own snapshot; the constants are the
//! public game rules (`bota-server` `config/rules.rs`), which the wire does not
//! carry: armor and magic-resist mitigation, raze damage and its per-stack
//! bonus, and lane creep acquisition ranges.

#![allow(
    clippy::float_arithmetic,
    reason = "derived policy inputs are bounded f32 values outside the deterministic simulation"
)]

use bota_proto::{AbilityId, AbilityView, StatusFlags, UnitKind, UnitView, Vec2};

use crate::raze_aim::{SHADOWRAZE_RADIUS, SHADOWRAZES};
use crate::tracker::is_structure;

/// Each point of armor adds this much to a base of one hundred.
const ARMOR_SCALE: f32 = 6.0;
/// Raze damage by ability level, before magic resistance.
const RAZE_DAMAGE: [f32; 4] = [90.0, 160.0, 230.0, 300.0];
/// Extra raze damage per prior stack of the same caster, by ability level.
const RAZE_STACK_DAMAGE: [f32; 4] = [50.0, 60.0, 70.0, 80.0];
/// Requiem of Souls.
pub(super) const REQUIEM: AbilityId = AbilityId(16);
/// Lane creeps attack only what comes within these ranges.
const MELEE_ACQUISITION: f32 = 500.0;
const RANGED_ACQUISITION: f32 = 600.0;
const SIEGE_ACQUISITION: f32 = 800.0;
/// An allied lane creep or building may be denied below these fractions of its health.
const DENY_FRACTION: f32 = 0.5;
const DENY_BUILDING_FRACTION: f32 = 0.1;
/// Longest time modelled by the time-to-kill inputs, which divide by it.
pub(super) const MAX_TTK_SECONDS: f32 = 30.0;
const TICKS_PER_SECOND: f32 = 30.0;

/// Health in whole points, never negative.
pub(super) fn health(unit: &UnitView) -> f32 {
    unit.hp.max(0) as f32
}

/// Fraction of physical damage `unit` keeps after its armor.
pub(super) fn physical_kept(unit: &UnitView) -> f32 {
    100.0 / (100.0 + ARMOR_SCALE * unit.armor.to_f32().max(0.0))
}

/// Fraction of magical damage `unit` keeps after its resistance.
pub(super) fn magical_kept(unit: &UnitView) -> f32 {
    (1.0 - unit.magic_resist.to_f32()).clamp(0.0, 1.0)
}

/// Damage one attack of `attacker` deals to `target`.
pub(super) fn hit_damage(attacker: &UnitView, target: &UnitView) -> f32 {
    attacker.attack_damage.max(0) as f32 * physical_kept(target)
}

/// Attacks `attacker` needs to bring `target` down; zero when it cannot hurt it.
pub(super) fn hits_to_kill(attacker: &UnitView, target: &UnitView) -> f32 {
    let hit = hit_damage(attacker, target);
    if hit <= 0.0 {
        return 0.0;
    }
    (health(target) / hit).ceil()
}

/// Seconds between attacks of `unit`.
pub(super) fn attack_seconds(unit: &UnitView) -> f32 {
    (unit.attack_time as f32 / 1000.0).max(1.0 / TICKS_PER_SECOND)
}

/// Health divided by the kept share of each damage kind.
pub(super) fn effective_health(unit: &UnitView) -> [f32; 2] {
    let hp = health(unit);
    [hp / physical_kept(unit), hp / magical_kept(unit).max(0.01)]
}

/// Euclidean distance between two positions in world units.
pub(super) fn distance(left: Vec2, right: Vec2) -> f32 {
    (left.distance_squared(right) as f64).sqrt() as f32 / 65_536.0
}

/// Whether a raze at `reach` along the caster's facing (cosine, sine) strikes a
/// target `offset` world units from the caster. Both are team-canonical, so
/// mirrored situations give identical answers on both sides.
pub(super) fn raze_strikes(offset: (f32, f32), facing: (f32, f32), reach: i32) -> bool {
    let reach = reach as f32;
    let x = offset.0 - reach * facing.0;
    let y = offset.1 - reach * facing.1;
    let radius = SHADOWRAZE_RADIUS as f32;
    x * x + y * y <= radius * radius
}

/// Whether a raze can strike `unit` at all: live, not a structure, not invulnerable.
pub(super) fn razeable(unit: &UnitView) -> bool {
    unit.hp > 0 && !is_structure(unit.kind) && unit.statuses.bits & StatusFlags::INVULNERABLE == 0
}

/// A hero's razes, ultimate and mana as the policy may read them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct HeroKit {
    /// Remaining cooldown ticks of the near, mid and far raze; `None` when unlearned.
    pub razes: [Option<u32>; 3],
    /// Shared raze level, zero while unlearned.
    pub raze_level: u8,
    /// Mana the next raze costs.
    pub raze_mana: i32,
    /// Remaining Requiem cooldown ticks; `None` when unlearned.
    pub requiem: Option<u32>,
    pub requiem_level: u8,
    /// Current mana, whole points.
    pub mana: i32,
}

impl HeroKit {
    /// The kit of a hero last seen `age` ticks ago; cooldowns keep running unseen.
    pub(super) fn of(unit: &UnitView, age: u32) -> Option<Self> {
        if unit.kind != UnitKind::Hero || unit.abilities.is_empty() {
            return None;
        }
        let find = |id: AbilityId| -> Option<&AbilityView> {
            unit.abilities
                .iter()
                .find(|ability| ability.id == id && ability.level > 0)
        };
        let mut kit = Self {
            mana: unit.mana.max(0),
            ..Self::default()
        };
        for (index, (id, _)) in SHADOWRAZES.iter().enumerate() {
            if let Some(ability) = find(*id) {
                kit.razes[index] = Some(ability.cooldown_left.saturating_sub(age));
                kit.raze_level = kit.raze_level.max(ability.level);
                kit.raze_mana = ability.mana_cost.max(0);
            }
        }
        if let Some(ability) = find(REQUIEM) {
            kit.requiem = Some(ability.cooldown_left.saturating_sub(age));
            kit.requiem_level = ability.level;
        }
        Some(kit)
    }

    /// Damage of the next raze on a target already holding `stacks` of this caster.
    pub(super) fn raze_damage(&self, target: &UnitView, stacks: u32) -> f32 {
        let Some(level) = usize::from(self.raze_level).checked_sub(1) else {
            return 0.0;
        };
        let level = level.min(RAZE_DAMAGE.len() - 1);
        (RAZE_DAMAGE[level] + RAZE_STACK_DAMAGE[level] * stacks as f32) * magical_kept(target)
    }

    /// Razes needed to bring `target` down, stacking as they land; `cap` when more.
    pub(super) fn razes_to_kill(&self, target: &UnitView, stacks: u32, cap: u32) -> u32 {
        let mut left = health(target);
        for count in 1..=cap {
            let damage = self.raze_damage(target, stacks + count - 1);
            if damage <= 0.0 {
                return cap;
            }
            left -= damage;
            if left <= 0.0 {
                return count;
            }
        }
        cap
    }

    /// Damage of every raze castable now in a row, stacking, within the mana.
    fn burst(&self, target: &UnitView, stacks: u32) -> f32 {
        let ready = self.razes.iter().filter(|left| **left == Some(0)).count() as u32;
        let affordable = if self.raze_mana > 0 {
            u32::try_from(self.mana / self.raze_mana).unwrap_or(0)
        } else {
            ready
        };
        (0..ready.min(affordable))
            .map(|index| self.raze_damage(target, stacks + index))
            .sum()
    }

    /// Seconds for `attacker` (carrying this kit) to kill `target`: ready razes, then attacks.
    pub(super) fn seconds_to_kill(
        &self,
        attacker: &UnitView,
        target: &UnitView,
        stacks: u32,
    ) -> f32 {
        let left = health(target) - self.burst(target, stacks);
        if left <= 0.0 {
            return 0.0;
        }
        let hit = hit_damage(attacker, target);
        if hit <= 0.0 {
            return MAX_TTK_SECONDS;
        }
        ((left / hit).ceil() * attack_seconds(attacker)).min(MAX_TTK_SECONDS)
    }

    /// Whether this hero can pay for its next raze.
    pub(super) fn can_afford_raze(&self) -> bool {
        self.raze_level > 0 && self.mana >= self.raze_mana
    }
}

/// How far this unit looks for something to attack, if it attacks on its own.
pub(super) fn acquisition(unit: &UnitView) -> Option<f32> {
    match unit.kind {
        UnitKind::CreepMelee | UnitKind::CreepFlagbearer | UnitKind::CreepNeutral => {
            Some(MELEE_ACQUISITION)
        }
        UnitKind::CreepRanged => Some(RANGED_ACQUISITION),
        UnitKind::CreepSiege => Some(SIEGE_ACQUISITION),
        UnitKind::Tower | UnitKind::Fountain => Some(unit.attack_range.to_f32()),
        UnitKind::Hero
        | UnitKind::Ancient
        | UnitKind::Barracks
        | UnitKind::Ward
        | UnitKind::Courier => None,
    }
}

/// Whether one attack of `hero` would kill `unit` now: a last hit, or a deny of an ally.
pub(super) fn killable_now(hero: &UnitView, unit: &UnitView, allied: bool) -> bool {
    if unit.hp <= 0 {
        return false;
    }
    let threshold = if is_structure(unit.kind) {
        DENY_BUILDING_FRACTION
    } else {
        DENY_FRACTION
    };
    let deniable = !allied || health(unit) < threshold * unit.max_hp.max(0) as f32;
    deniable && health(unit) <= hit_damage(hero, unit)
}
