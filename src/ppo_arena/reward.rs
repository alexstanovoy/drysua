#![allow(
    clippy::float_arithmetic,
    reason = "Checked f64 reward telemetry accumulation"
)]

use crate::{Map2RewardBreakdown, Map2RewardObservations, PpoError};

/// Invocation-local full Map2 reward components and raw seat-visible measurements.
/// Tick counters include unretained actor intervals, but exclude setup/warmup baselines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map2TrainingReward {
    pub tower_damage_taken: f64,
    pub opening_position: f64,
    pub ticks: u64,
    pub gold: f64,
    pub experience: f64,
    pub hero_damage: f64,
    pub hero_damage_taken: f64,
    pub creep_damage_taken: f64,
    pub other_damage_taken: f64,
    pub mana_spent: f64,
    pub tower_health: f64,
    pub lane_pressure: f64,
    pub pregame_movement: f64,
    pub fountain_wait: f64,
    pub fountain_wait_refund: f64,
    pub stagnation_base: f64,
    pub stagnation_ticks_cost: f64,
    pub terminal: f64,
    pub victory_time: f64,
    pub total: f64,
    pub observations: Map2RewardObservations,
}

impl Map2TrainingReward {
    pub(crate) fn log(&self, scope: &'static str, updates: u64) {
        assert!(matches!(
            scope,
            "checkpoint" | "invocation" | "ppo_smoke" | "league_smoke"
        ));
        assert!(updates <= 1_000_000);
        if crate::telemetry::prometheus::enabled() {
            return;
        }
        crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default()).emit(
            &format_args!(
                "level=INFO event=map2_training_reward scope={scope} updates={updates} {self}"
            ),
        );
    }

    pub(super) fn record(&mut self, interval: Map2RewardBreakdown) -> Result<(), PpoError> {
        self.merge(Self {
            tower_damage_taken: interval.tower_damage_taken,
            opening_position: interval.opening_position,
            ticks: u64::from(interval.ticks),
            gold: interval.gold,
            experience: interval.experience,
            hero_damage: interval.hero_damage,
            hero_damage_taken: interval.hero_damage_taken,
            creep_damage_taken: interval.creep_damage_taken,
            other_damage_taken: interval.other_damage_taken,
            mana_spent: interval.mana_spent,
            tower_health: interval.tower_health,
            lane_pressure: interval.lane_pressure,
            pregame_movement: interval.pregame_movement,
            fountain_wait: interval.fountain_wait,
            fountain_wait_refund: interval.fountain_wait_refund,
            stagnation_base: interval.stagnation_base,
            stagnation_ticks_cost: interval.stagnation_ticks_cost,
            terminal: interval.terminal,
            victory_time: interval.victory_time,
            total: interval.total,
            observations: interval.observations,
        })
    }

    pub(super) fn merge(&mut self, other: Self) -> Result<(), PpoError> {
        if self.components().iter().any(|value| !value.is_finite())
            || other.components().iter().any(|value| !value.is_finite())
        {
            return Err(PpoError::NonFinite("Map2 reward telemetry"));
        }
        let merged = Self {
            tower_damage_taken: self.tower_damage_taken + other.tower_damage_taken,
            opening_position: self.opening_position + other.opening_position,
            ticks: self
                .ticks
                .checked_add(other.ticks)
                .ok_or(PpoError::CounterOverflow)?,
            gold: self.gold + other.gold,
            experience: self.experience + other.experience,
            hero_damage: self.hero_damage + other.hero_damage,
            hero_damage_taken: self.hero_damage_taken + other.hero_damage_taken,
            creep_damage_taken: self.creep_damage_taken + other.creep_damage_taken,
            other_damage_taken: self.other_damage_taken + other.other_damage_taken,
            mana_spent: self.mana_spent + other.mana_spent,
            tower_health: self.tower_health + other.tower_health,
            lane_pressure: self.lane_pressure + other.lane_pressure,
            pregame_movement: self.pregame_movement + other.pregame_movement,
            fountain_wait: self.fountain_wait + other.fountain_wait,
            fountain_wait_refund: self.fountain_wait_refund + other.fountain_wait_refund,
            stagnation_base: self.stagnation_base + other.stagnation_base,
            stagnation_ticks_cost: self.stagnation_ticks_cost + other.stagnation_ticks_cost,
            terminal: self.terminal + other.terminal,
            victory_time: self.victory_time + other.victory_time,
            total: self.total + other.total,
            observations: merge_observations(self.observations, other.observations)?,
        };
        if merged.components().iter().any(|value| !value.is_finite()) {
            return Err(PpoError::NonFinite("Map2 reward telemetry"));
        }
        assert!(merged.ticks >= self.ticks);
        assert!(merged.ticks >= other.ticks);
        *self = merged;
        Ok(())
    }

    pub(super) fn components(&self) -> [f64; 19] {
        [
            self.tower_damage_taken,
            self.opening_position,
            self.gold,
            self.experience,
            self.hero_damage,
            self.hero_damage_taken,
            self.creep_damage_taken,
            self.other_damage_taken,
            self.mana_spent,
            self.tower_health,
            self.lane_pressure,
            self.pregame_movement,
            self.fountain_wait,
            self.fountain_wait_refund,
            self.stagnation_base,
            self.stagnation_ticks_cost,
            self.terminal,
            self.victory_time,
            self.total,
        ]
    }
}

fn merge_observations(
    left: Map2RewardObservations,
    right: Map2RewardObservations,
) -> Result<Map2RewardObservations, PpoError> {
    let add = |left: u64, right: u64| left.checked_add(right).ok_or(PpoError::CounterOverflow);
    Ok(Map2RewardObservations {
        tower_damage_taken: add(left.tower_damage_taken, right.tower_damage_taken)?,
        opening_position_checks: add(left.opening_position_checks, right.opening_position_checks)?,
        victory_time_ticks: add(left.victory_time_ticks, right.victory_time_ticks)?,
        own_gold_earned: add(left.own_gold_earned, right.own_gold_earned)?,
        enemy_gold_earned: add(left.enemy_gold_earned, right.enemy_gold_earned)?,
        own_xp_gained: add(left.own_xp_gained, right.own_xp_gained)?,
        enemy_xp_gained: add(left.enemy_xp_gained, right.enemy_xp_gained)?,
        hero_damage_dealt: add(left.hero_damage_dealt, right.hero_damage_dealt)?,
        structure_damage_dealt: add(left.structure_damage_dealt, right.structure_damage_dealt)?,
        creep_kills: add(left.creep_kills, right.creep_kills)?,
        creep_denies: add(left.creep_denies, right.creep_denies)?,
        hero_damage_taken: add(left.hero_damage_taken, right.hero_damage_taken)?,
        creep_damage_taken: add(left.creep_damage_taken, right.creep_damage_taken)?,
        other_damage_taken: add(left.other_damage_taken, right.other_damage_taken)?,
        unattributed_damage_taken: add(
            left.unattributed_damage_taken,
            right.unattributed_damage_taken,
        )?,
        mana_spent: add(left.mana_spent, right.mana_spent)?,
        mana_unobserved_ticks: add(left.mana_unobserved_ticks, right.mana_unobserved_ticks)?,
        lane_last_hits: add(left.lane_last_hits, right.lane_last_hits)?,
        neutral_last_hits: add(left.neutral_last_hits, right.neutral_last_hits)?,
        unattributed_damage_events: add(
            left.unattributed_damage_events,
            right.unattributed_damage_events,
        )?,
        unattributed_deaths: add(left.unattributed_deaths, right.unattributed_deaths)?,
        duplicate_deaths: add(left.duplicate_deaths, right.duplicate_deaths)?,
        lane_observed_ticks: add(left.lane_observed_ticks, right.lane_observed_ticks)?,
        fountain_wait_ticks: add(left.fountain_wait_ticks, right.fountain_wait_ticks)?,
        fountain_wait_charged_ticks: add(
            left.fountain_wait_charged_ticks,
            right.fountain_wait_charged_ticks,
        )?,
        fountain_wait_refunds: add(left.fountain_wait_refunds, right.fountain_wait_refunds)?,
        stagnation_active_ticks: add(left.stagnation_active_ticks, right.stagnation_active_ticks)?,
        stagnation_idle_ticks: add(left.stagnation_idle_ticks, right.stagnation_idle_ticks)?,
        stagnation_charged_ticks: add(
            left.stagnation_charged_ticks,
            right.stagnation_charged_ticks,
        )?,
        stagnation_base_charges: add(left.stagnation_base_charges, right.stagnation_base_charges)?,
        stagnation_repaid_ticks: add(left.stagnation_repaid_ticks, right.stagnation_repaid_ticks)?,
        progress_reasons: left.progress_reasons | right.progress_reasons,
    })
}

impl std::fmt::Display for Map2TrainingReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "reward_tower_taken={:.9} reward_opening_position={:.9} tower_damage_taken={} opening_position_checks={} victory_time_ticks={} ",
            self.tower_damage_taken,
            self.opening_position,
            self.observations.tower_damage_taken,
            self.observations.opening_position_checks,
            self.observations.victory_time_ticks
        )?;
        write!(
            formatter,
            "reward_ticks={} reward_total={:.9} reward_gold={:.9} reward_xp={:.9} reward_hero_damage={:.9} reward_hero_taken={:.9} reward_creep_taken={:.9} reward_other_taken={:.9} reward_mana={:.9} reward_towers={:.9} reward_lane={:.9} reward_pregame_movement={:.9} reward_fountain_wait={:.9} reward_fountain_wait_refund={:.9} reward_stagnation_base={:.9} reward_stagnation_ticks_cost={:.9} reward_terminal={:.9} reward_victory_time={:.9}",
            self.ticks,
            self.total,
            self.gold,
            self.experience,
            self.hero_damage,
            self.hero_damage_taken,
            self.creep_damage_taken,
            self.other_damage_taken,
            self.mana_spent,
            self.tower_health,
            self.lane_pressure,
            self.pregame_movement,
            self.fountain_wait,
            self.fountain_wait_refund,
            self.stagnation_base,
            self.stagnation_ticks_cost,
            self.terminal,
            self.victory_time
        )?;
        let raw = self.observations;
        write!(
            formatter,
            " own_gold={} enemy_gold={} own_xp={} enemy_xp={} hero_damage_dealt={} hero_damage_taken={} creep_damage_taken={} other_damage_taken={} unattributed_damage_taken={} mana_spent={} mana_unobserved_ticks={} lane_last_hits={} neutral_last_hits={} unattributed_damage_events={} unattributed_deaths={} duplicate_deaths={} lane_observed_ticks={} fountain_wait_ticks={} fountain_wait_charged_ticks={} fountain_wait_refunds={}",
            raw.own_gold_earned,
            raw.enemy_gold_earned,
            raw.own_xp_gained,
            raw.enemy_xp_gained,
            raw.hero_damage_dealt,
            raw.hero_damage_taken,
            raw.creep_damage_taken,
            raw.other_damage_taken,
            raw.unattributed_damage_taken,
            raw.mana_spent,
            raw.mana_unobserved_ticks,
            raw.lane_last_hits,
            raw.neutral_last_hits,
            raw.unattributed_damage_events,
            raw.unattributed_deaths,
            raw.duplicate_deaths,
            raw.lane_observed_ticks,
            raw.fountain_wait_ticks,
            raw.fountain_wait_charged_ticks,
            raw.fountain_wait_refunds
        )?;
        write!(
            formatter,
            " structure_damage_dealt={} creep_kills={} creep_denies={} stagnation_active_ticks={} stagnation_idle_ticks={} stagnation_charged_ticks={} stagnation_base_charges={} stagnation_repaid_ticks={} progress_reasons=0x{:04x}",
            raw.structure_damage_dealt,
            raw.creep_kills,
            raw.creep_denies,
            raw.stagnation_active_ticks,
            raw.stagnation_idle_ticks,
            raw.stagnation_charged_ticks,
            raw.stagnation_base_charges,
            raw.stagnation_repaid_ticks,
            raw.progress_reasons
        )
    }
}
