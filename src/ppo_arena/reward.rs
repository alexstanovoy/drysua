#![allow(
    clippy::float_arithmetic,
    reason = "Checked f64 reward telemetry accumulation"
)]

use crate::{
    MAP2_REWARD_COMPONENTS, MAP2_REWARD_COUNTERS, Map2RewardBreakdown, Map2RewardObservations,
    PpoError,
};

/// Summed Map2 reward components and seat-visible measurements over episodes or invocations.
/// Tick counters include unretained actor intervals, but exclude setup/warmup baselines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map2TrainingReward {
    pub ticks: u64,
    /// Values in [`MAP2_REWARD_COMPONENTS`] order.
    pub components: [f64; MAP2_REWARD_COMPONENTS.len()],
    pub total: f64,
    pub observations: Map2RewardObservations,
}

impl Map2TrainingReward {
    pub(crate) fn log(&self, scope: &'static str, updates: u64) {
        assert!(matches!(scope, "update" | "invocation"));
        assert!(updates <= crate::MAX_TRAINING_COUNTER);
        crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default()).emit(
            &format_args!(
                "level=INFO event=map2_training_reward scope={scope} updates={updates} {self}"
            ),
        );
    }

    pub(super) fn record(&mut self, interval: Map2RewardBreakdown) -> Result<(), PpoError> {
        self.merge(Self {
            ticks: u64::from(interval.ticks),
            components: interval.components(),
            total: interval.total,
            observations: interval.observations,
        })
    }

    pub(super) fn merge(&mut self, other: Self) -> Result<(), PpoError> {
        let components: [f64; MAP2_REWARD_COMPONENTS.len()] =
            std::array::from_fn(|index| self.components[index] + other.components[index]);
        let total = self.total + other.total;
        if !total.is_finite() || components.iter().any(|value| !value.is_finite()) {
            return Err(PpoError::NonFinite("Map2 reward telemetry"));
        }
        *self = Self {
            ticks: self
                .ticks
                .checked_add(other.ticks)
                .ok_or(PpoError::CounterOverflow)?,
            components,
            total,
            observations: self
                .observations
                .checked_add(&other.observations)
                .ok_or(PpoError::CounterOverflow)?,
        };
        Ok(())
    }
}

impl std::fmt::Display for Map2TrainingReward {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "reward_version={} reward_ticks={} reward_total={:.9}",
            crate::MAP2_REWARD_VERSION,
            self.ticks,
            self.total
        )?;
        for (name, value) in MAP2_REWARD_COMPONENTS.iter().zip(self.components) {
            write!(formatter, " reward_{name}={value:.9}")?;
        }
        for (name, value) in MAP2_REWARD_COUNTERS
            .iter()
            .zip(self.observations.counters())
        {
            write!(formatter, " {name}={value}")?;
        }
        Ok(())
    }
}
