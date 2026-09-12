#![allow(
    clippy::float_arithmetic,
    reason = "Checked f64 reward telemetry accumulation"
)]

use crate::{Map2RewardBreakdown, Map2RewardObservations, PpoError};

/// Invocation-local full Map2 reward components and raw seat-visible measurements.
/// Tick counters include unretained actor intervals, but exclude setup/warmup baselines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map2TrainingReward {
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
    pub terminal: f64,
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
        crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default()).emit(
            &format_args!(
                "level=INFO event=map2_training_reward scope={scope} updates={updates} {self}"
            ),
        );
    }

    pub(super) fn record(&mut self, interval: Map2RewardBreakdown) -> Result<(), PpoError> {
        self.merge(Self {
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
            terminal: interval.terminal,
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
            terminal: self.terminal + other.terminal,
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

    pub(super) fn components(&self) -> [f64; 11] {
        [
            self.gold,
            self.experience,
            self.hero_damage,
            self.hero_damage_taken,
            self.creep_damage_taken,
            self.other_damage_taken,
            self.mana_spent,
            self.tower_health,
            self.lane_pressure,
            self.terminal,
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
        own_gold_earned: add(left.own_gold_earned, right.own_gold_earned)?,
        enemy_gold_earned: add(left.enemy_gold_earned, right.enemy_gold_earned)?,
        own_xp_gained: add(left.own_xp_gained, right.own_xp_gained)?,
        enemy_xp_gained: add(left.enemy_xp_gained, right.enemy_xp_gained)?,
        hero_damage_dealt: add(left.hero_damage_dealt, right.hero_damage_dealt)?,
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
    })
}

impl std::fmt::Display for Map2TrainingReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "reward_ticks={} reward_total={:.9} reward_gold={:.9} reward_xp={:.9} reward_hero_damage={:.9} reward_hero_taken={:.9} reward_creep_taken={:.9} reward_other_taken={:.9} reward_mana={:.9} reward_towers={:.9} reward_lane={:.9} reward_terminal={:.9}",
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
            self.terminal
        )?;
        let raw = self.observations;
        write!(
            formatter,
            " own_gold={} enemy_gold={} own_xp={} enemy_xp={} hero_damage_dealt={} hero_damage_taken={} creep_damage_taken={} other_damage_taken={} unattributed_damage_taken={} mana_spent={} mana_unobserved_ticks={} lane_last_hits={} neutral_last_hits={} unattributed_damage_events={} unattributed_deaths={} duplicate_deaths={} lane_observed_ticks={}",
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
            raw.lane_observed_ticks
        )
    }
}
