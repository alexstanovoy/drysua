use bota_proto::{EventKind, Fixed};

use super::observation::SnapshotFacts;
use super::{
    MAP2_REWARD_ACTIVITY_LEASE_TICKS, MAP2_REWARD_STAGNATION_BASE_COST,
    MAP2_REWARD_STAGNATION_REPAY_PER_TICK, MAP2_REWARD_STAGNATION_THRESHOLD_TICKS,
    MAP2_REWARD_STAGNATION_TICK_COST, Map2Reward, Map2RewardBreakdown, Map2RewardObservations,
};

pub const MAP2_PROGRESS_XP: u16 = 1 << 0;
pub const MAP2_PROGRESS_GOLD: u16 = 1 << 1;
pub const MAP2_PROGRESS_HERO_DAMAGE: u16 = 1 << 2;
pub const MAP2_PROGRESS_STRUCTURE_DAMAGE: u16 = 1 << 3;
pub const MAP2_PROGRESS_CREEP_KILL: u16 = 1 << 4;
pub const MAP2_PROGRESS_CREEP_DENY: u16 = 1 << 5;
pub const MAP2_PROGRESS_PURCHASE: u16 = 1 << 6;
pub const MAP2_PROGRESS_FOUNTAIN_AURA: u16 = 1 << 7;
pub const MAP2_PROGRESS_PREGAME_MOVEMENT: u16 = 1 << 8;
pub const MAP2_PROGRESS_WAVE_PRESSURE: u16 = 1 << 9;

impl Map2Reward {
    pub(super) fn observe_progress(
        &mut self,
        pending: &SnapshotFacts,
        events: &[EventKind],
        before: Map2RewardBreakdown,
    ) {
        if self.current.is_none() {
            return;
        }
        let mut reasons = counter_reasons(before.observations, self.interval.observations);
        if pending.fountain_aura {
            reasons |= MAP2_PROGRESS_FOUNTAIN_AURA;
        }
        if events.iter().any(|event| {
            matches!(event,
            EventKind::ItemBought { slot, .. } if *slot == self.roles[0].slot)
        }) {
            reasons |= MAP2_PROGRESS_PURCHASE;
        }
        if self.interval.pregame_movement > before.pregame_movement {
            reasons |= MAP2_PROGRESS_PREGAME_MOVEMENT;
        }
        if self.interval.lane_pressure > before.lane_pressure && self.near_own_wave(pending) {
            reasons |= MAP2_PROGRESS_WAVE_PRESSURE;
        }
        self.interval.observations.progress_reasons |= reasons;
        if reasons != 0 {
            self.activity_ticks_left = MAP2_REWARD_ACTIVITY_LEASE_TICKS;
        }
        if self.activity_ticks_left > 0 {
            self.repay_progress_debt();
        } else {
            self.accrue_progress_debt();
        }
        assert!(self.stagnation_ticks <= MAP2_REWARD_STAGNATION_THRESHOLD_TICKS);
        assert!(self.activity_ticks_left < MAP2_REWARD_ACTIVITY_LEASE_TICKS);
    }

    fn repay_progress_debt(&mut self) {
        self.activity_ticks_left -= 1;
        let repaid = self
            .stagnation_ticks
            .min(MAP2_REWARD_STAGNATION_REPAY_PER_TICK);
        self.stagnation_ticks -= repaid;
        self.interval.observations.stagnation_active_ticks += 1;
        self.interval.observations.stagnation_repaid_ticks += u64::from(repaid);
        if self.stagnation_ticks == 0 {
            self.stagnation_base_charged = false;
        }
        assert!(repaid <= MAP2_REWARD_STAGNATION_REPAY_PER_TICK);
        assert!(self.stagnation_ticks <= MAP2_REWARD_STAGNATION_THRESHOLD_TICKS);
    }

    fn accrue_progress_debt(&mut self) {
        self.stagnation_ticks =
            (self.stagnation_ticks + 1).min(MAP2_REWARD_STAGNATION_THRESHOLD_TICKS);
        self.interval.observations.stagnation_idle_ticks += 1;
        if self.stagnation_ticks < MAP2_REWARD_STAGNATION_THRESHOLD_TICKS {
            return;
        }
        self.interval.observations.stagnation_charged_ticks += 1;
        if self.stagnation_base_charged {
            self.interval.stagnation_ticks_cost -= MAP2_REWARD_STAGNATION_TICK_COST;
        } else {
            self.stagnation_base_charged = true;
            self.interval.stagnation_base -= MAP2_REWARD_STAGNATION_BASE_COST;
            self.interval.observations.stagnation_base_charges += 1;
        }
        assert!(self.stagnation_base_charged);
        assert_eq!(
            self.stagnation_ticks,
            MAP2_REWARD_STAGNATION_THRESHOLD_TICKS
        );
    }

    fn near_own_wave(&self, pending: &SnapshotFacts) -> bool {
        let Some(id) = pending.heroes[0] else {
            return false;
        };
        let Ok(index) = pending.units.binary_search_by_key(&id, |unit| unit.id) else {
            return false;
        };
        let hero = &pending.units[index];
        if hero.hp == 0 {
            return false;
        }
        const RADIUS: i128 = 1500_i128 << Fixed::FRAC_BITS;
        pending.units.iter().any(|unit| {
            if unit.hp == 0
                || unit.team != self.roles[0].team
                || !super::events::lane_creep(unit.kind)
            {
                return false;
            }
            let x = i128::from(hero.pos.x.raw) - i128::from(unit.pos.x.raw);
            let y = i128::from(hero.pos.y.raw) - i128::from(unit.pos.y.raw);
            x * x + y * y <= RADIUS * RADIUS
        })
    }
}

fn counter_reasons(before: Map2RewardObservations, after: Map2RewardObservations) -> u16 {
    let mut reasons = 0;
    for (previous, current, bit) in [
        (before.own_xp_gained, after.own_xp_gained, MAP2_PROGRESS_XP),
        (
            before.own_gold_earned,
            after.own_gold_earned,
            MAP2_PROGRESS_GOLD,
        ),
        (
            before.hero_damage_dealt,
            after.hero_damage_dealt,
            MAP2_PROGRESS_HERO_DAMAGE,
        ),
        (
            before.structure_damage_dealt,
            after.structure_damage_dealt,
            MAP2_PROGRESS_STRUCTURE_DAMAGE,
        ),
        (
            before.creep_kills,
            after.creep_kills,
            MAP2_PROGRESS_CREEP_KILL,
        ),
        (
            before.creep_denies,
            after.creep_denies,
            MAP2_PROGRESS_CREEP_DENY,
        ),
    ] {
        assert!(current >= previous);
        if current > previous {
            reasons |= bit;
        }
    }
    reasons
}
