//! Fading training aids: imitation of a shadow rule policy and critic warm-up.
//!
//! Both are pure functions of the update index and the run scope, so a resumed
//! run recomputes exactly the terms the uninterrupted run trained with. Neither
//! masks nor overrides an action: imitation only adds a loss term toward the
//! shadow's labels, and a schedule that ends at zero leaves plain PPO.

use super::slot::ShadowLabels;
use crate::{EnvironmentDecimal, MAX_TRAINING_COUNTER, PpoError, ScriptKind, UpdateObjective};

/// Largest imitation coefficient, in millionths.
const MAX_IMITATION_UNITS: u64 = 10 * EnvironmentDecimal::SCALE;

/// A linear imitation coefficient schedule toward one shadow rule policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImitationSchedule {
    /// The rule policy whose decisions on the learner's own states are the labels.
    pub shadow: ScriptKind,
    /// Coefficient at update 0.
    pub start: EnvironmentDecimal,
    /// Coefficient from update `updates` on.
    pub end: EnvironmentDecimal,
    /// Updates over which the coefficient moves linearly from `start` to `end`.
    pub updates: u64,
    /// Power of the inverse imitation-class frequency each label is weighted by:
    /// 0 weighs every label alike, 1 every present class equally.
    pub balance: EnvironmentDecimal,
}

/// The aids of one run; the default trains plain PPO.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingGuidance {
    pub imitation: Option<ImitationSchedule>,
    /// Updates at the start of the run that train only the critic.
    pub critic_warmup_updates: u64,
}

impl TrainingGuidance {
    pub(crate) fn validate(&self) -> Result<(), PpoError> {
        if self.critic_warmup_updates > MAX_TRAINING_COUNTER {
            return Err(PpoError::InvalidConfig("critic warm-up updates"));
        }
        if let Some(schedule) = self.imitation {
            if schedule.updates == 0 || schedule.updates > MAX_TRAINING_COUNTER {
                return Err(PpoError::InvalidConfig("imitation schedule updates"));
            }
            if schedule.start.units() > MAX_IMITATION_UNITS
                || schedule.end.units() > MAX_IMITATION_UNITS
            {
                return Err(PpoError::InvalidConfig("imitation coefficient above 10"));
            }
            if schedule.balance.units() > EnvironmentDecimal::SCALE {
                return Err(PpoError::InvalidConfig("imitation balance above 1"));
            }
        }
        Ok(())
    }

    /// The auxiliary terms of update `update` (completed updates before it).
    pub(crate) fn objective(&self, update: u64) -> UpdateObjective {
        UpdateObjective {
            imitation: self
                .imitation
                .map_or(0.0, |schedule| coefficient(schedule, update)),
            imitation_balance: self
                .imitation
                .map_or(0.0, |schedule| decimal(schedule.balance)),
            critic_only: update < self.critic_warmup_updates,
        }
    }

    /// Which games need shadow labels: after a schedule that fades to zero, games
    /// starting at its end only feed updates that no longer imitate.
    pub(crate) fn shadow_labels(&self) -> Option<ShadowLabels> {
        self.imitation.map(|schedule| ShadowLabels {
            kind: schedule.shadow,
            until_update: if schedule.end.units() == 0 {
                schedule.updates
            } else {
                u64::MAX
            },
        })
    }

    /// Appends the non-default aids to the run scope in a fixed order.
    pub(crate) fn append_scope(&self, command_line: &mut String) {
        if let Some(schedule) = self.imitation {
            command_line.push_str(&format!(
                " --imitation-coefficient {}:{}:{} --imitation-shadow {} --imitation-balance {}",
                schedule.start,
                schedule.end,
                schedule.updates,
                schedule.shadow.label(),
                schedule.balance
            ));
        }
        if self.critic_warmup_updates > 0 {
            command_line.push_str(&format!(
                " --critic-warmup-updates {}",
                self.critic_warmup_updates
            ));
        }
    }
}

/// The coefficient at `update`, interpolated in millionths (truncated), then as `f32`.
#[allow(
    clippy::float_arithmetic,
    reason = "the exact millionths become the loss weight once"
)]
fn coefficient(schedule: ImitationSchedule, update: u64) -> f32 {
    let (start, end) = (
        i128::from(schedule.start.units()),
        i128::from(schedule.end.units()),
    );
    let done = i128::from(update.min(schedule.updates));
    let units = start + (end - start) * done / i128::from(schedule.updates);
    assert!((0..=i128::from(MAX_IMITATION_UNITS)).contains(&units));
    (units as f64 / EnvironmentDecimal::SCALE as f64) as f32
}

#[allow(
    clippy::float_arithmetic,
    reason = "exact millionths become a loss setting once"
)]
fn decimal(value: EnvironmentDecimal) -> f32 {
    (value.units() as f64 / EnvironmentDecimal::SCALE as f64) as f32
}
