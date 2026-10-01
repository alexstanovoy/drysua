#[cfg(test)]
#[path = "adaptive_environment_tests.rs"]
mod tests;

use std::fmt;
use std::str::FromStr;

use crate::{MAX_TRAINING_COUNTER, PpoError};

// Enough for a u64 integer, a decimal point, and six fractional digits.
const MAX_DECIMAL_TEXT_BYTES: usize = 20 + 1 + 6;
const _: () = assert!(MAX_TRAINING_COUNTER > 0);
const _: () = assert!(MAX_TRAINING_COUNTER < u64::MAX);
const _: () =
    assert!(MAX_TRAINING_COUNTER as u128 * EnvironmentDecimal::SCALE as u128 <= u64::MAX as u128);

/// Unsigned fixed-point decimal in millionths; rate and extension bounds are checked by config
/// validation. Text is at most 27 ASCII bytes with up to six fractional digits; signs,
/// whitespace, exponents, and a trailing point are rejected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnvironmentDecimal(u64);

impl EnvironmentDecimal {
    pub const SCALE: u64 = 1_000_000;

    /// Raw millionths; no config bounds are applied.
    pub const fn from_units(units: u64) -> Self {
        Self(units)
    }

    pub const fn units(self) -> u64 {
        self.0
    }
}

impl FromStr for EnvironmentDecimal {
    type Err = PpoError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() > MAX_DECIMAL_TEXT_BYTES {
            return Err(PpoError::InvalidConfig(
                "environment decimal exceeds 27 bytes",
            ));
        }
        let notation_error =
            PpoError::InvalidConfig("environment decimal must use unsigned plain decimal notation");
        let (whole, fraction) = match text.split_once('.') {
            Some((_, "")) => return Err(notation_error),
            Some(parts) => parts,
            None => (text, ""),
        };
        if (whole.is_empty() && fraction.is_empty())
            || !whole.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(notation_error);
        }
        if fraction.len() > 6 {
            return Err(PpoError::InvalidConfig(
                "environment decimal has more than six fractional digits",
            ));
        }
        let overflow = PpoError::InvalidConfig("environment decimal exceeds u64 millionths");
        let whole = if whole.is_empty() {
            0
        } else {
            whole.parse::<u64>().map_err(|_| overflow.clone())?
        };
        let fractional_units = if fraction.is_empty() {
            0
        } else {
            fraction.parse::<u64>().map_err(|_| overflow.clone())?
                * 10_u64.pow(6 - fraction.len() as u32)
        };
        let units = whole
            .checked_mul(Self::SCALE)
            .and_then(|units| units.checked_add(fractional_units))
            .ok_or(overflow)?;
        debug_assert_eq!(units / Self::SCALE, whole);
        debug_assert!(fractional_units < Self::SCALE);
        Ok(Self::from_units(units))
    }
}

impl fmt::Display for EnvironmentDecimal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / Self::SCALE;
        let mut fraction = self.0 % Self::SCALE;
        if fraction == 0 {
            return write!(formatter, "{whole}");
        }
        let mut precision = 6;
        for _ in 0..5 {
            if !fraction.is_multiple_of(10) {
                break;
            }
            fraction /= 10;
            precision -= 1;
        }
        debug_assert!(fraction > 0);
        debug_assert!((1..=6).contains(&precision));
        write!(formatter, "{whole}.{fraction:0precision$}")
    }
}

/// Consecutive per-update thresholds and fractional extra updates per poor-streak award.
/// Thresholds may overlap; a completed success streak takes priority over poor awards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveEnvironmentConfig {
    pub success_updates: u64,
    pub success_rate: EnvironmentDecimal,
    pub poor_updates: u64,
    pub poor_rate: EnvironmentDecimal,
    pub extension: EnvironmentDecimal,
}

impl Default for AdaptiveEnvironmentConfig {
    fn default() -> Self {
        Self {
            success_updates: 2,
            success_rate: EnvironmentDecimal::from_units(800_000),
            poor_updates: 1,
            poor_rate: EnvironmentDecimal::from_units(200_000),
            extension: EnvironmentDecimal::from_units(750_000),
        }
    }
}

impl AdaptiveEnvironmentConfig {
    /// Extension is bounded by its full numeric value, not its truncated integer part.
    pub fn validate(self) -> Result<Self, PpoError> {
        if !(1..=MAX_TRAINING_COUNTER).contains(&self.success_updates) {
            return Err(PpoError::InvalidConfig(
                "environment success updates must be in 1..=MAX_TRAINING_COUNTER",
            ));
        }
        if !(1..=MAX_TRAINING_COUNTER).contains(&self.poor_updates) {
            return Err(PpoError::InvalidConfig(
                "environment poor updates must be in 1..=MAX_TRAINING_COUNTER",
            ));
        }
        if self.success_rate.units() > EnvironmentDecimal::SCALE {
            return Err(PpoError::InvalidConfig(
                "environment success rate must be in [0, 1]",
            ));
        }
        if self.poor_rate.units() > EnvironmentDecimal::SCALE {
            return Err(PpoError::InvalidConfig(
                "environment poor rate must be in [0, 1]",
            ));
        }
        if self.extension.units() > MAX_TRAINING_COUNTER * EnvironmentDecimal::SCALE {
            return Err(PpoError::InvalidConfig(
                "environment extension must not exceed MAX_TRAINING_COUNTER",
            ));
        }
        Ok(self)
    }

    /// Appends the scope flags, including the leading separator space; does not validate.
    pub fn append_scope(&self, scope: &mut String) {
        scope.push_str(&self.scope_suffix());
    }

    pub fn scope_suffix(&self) -> String {
        format!(
            concat!(
                " --environment-schedule adaptive --environment-success-updates {}",
                " --environment-success-rate {} --environment-poor-updates {}",
                " --environment-poor-rate {} --environment-extension {}"
            ),
            self.success_updates,
            self.success_rate,
            self.poor_updates,
            self.poor_rate,
            self.extension
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentSchedule {
    Fixed,
    Adaptive(AdaptiveEnvironmentConfig),
}

/// Update counts: per-generation base budget, run length, and the trailing zero-randomization
/// phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveEnvironmentLimits {
    pub base_updates: u64,
    pub total_updates: u64,
    pub zero_updates: u64,
}

impl AdaptiveEnvironmentLimits {
    /// Base may exceed total; zero updates may cover none or all of the run.
    pub fn validate(self) -> Result<Self, PpoError> {
        if !(1..=MAX_TRAINING_COUNTER).contains(&self.base_updates) {
            return Err(PpoError::InvalidConfig(
                "environment base updates must be in 1..=MAX_TRAINING_COUNTER",
            ));
        }
        if !(1..=MAX_TRAINING_COUNTER).contains(&self.total_updates) {
            return Err(PpoError::InvalidConfig(
                "environment total updates must be in 1..=MAX_TRAINING_COUNTER",
            ));
        }
        if self.zero_updates > self.total_updates {
            return Err(PpoError::InvalidConfig(
                "environment zero updates must not exceed total updates",
            ));
        }
        Ok(self)
    }
}

/// Controller state. Only streak counters are kept; no game history or window averages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveEnvironmentState {
    pub generation: u64,
    pub start_update: u64,
    pub updates_in_generation: u64,
    pub success_streak: u64,
    pub poor_streak: u64,
    pub extension_awards: u64,
}

impl AdaptiveEnvironmentState {
    /// Returns the state after `completed_update`; publish it only after PPO succeeds.
    /// Advancing on a success streak or the zero-phase boundary discards extension credit;
    /// otherwise a poor-streak award is added before the budget-exhaustion check.
    /// The final update keeps its counters and never advances to a next generation.
    pub fn observe(
        &self,
        config: AdaptiveEnvironmentConfig,
        limits: AdaptiveEnvironmentLimits,
        completed_update: u64,
        wins: u64,
        games: u64,
    ) -> Result<Self, PpoError> {
        config.validate()?;
        limits.validate()?;
        if !(1..=limits.total_updates).contains(&completed_update) {
            return Err(PpoError::InvalidTransition(
                "environment completed update must be in 1..=total updates",
            ));
        }
        self.validate(config, limits, completed_update - 1)?;
        if wins > games {
            return Err(PpoError::InvalidTransition(
                "environment wins must not exceed games",
            ));
        }
        let mut next = *self;
        next.updates_in_generation =
            self.updates_in_generation
                .checked_add(1)
                .ok_or(PpoError::InvalidTransition(
                    "environment spent updates addition overflow",
                ))?;
        let scaled_wins = u128::from(wins) * u128::from(EnvironmentDecimal::SCALE);
        // An update that finished no game carries no signal and breaks both streaks.
        next.success_streak = capped_streak(
            self.success_streak,
            config.success_updates,
            games > 0 && scaled_wins >= u128::from(config.success_rate.units()) * u128::from(games),
        );
        next.poor_streak = capped_streak(
            self.poor_streak,
            config.poor_updates,
            games > 0 && scaled_wins <= u128::from(config.poor_rate.units()) * u128::from(games),
        );
        let success = next.success_streak == config.success_updates;
        let boundary = limits.total_updates - limits.zero_updates;
        let force_boundary = completed_update == boundary && self.start_update < boundary;
        if completed_update < limits.total_updates && (success || force_boundary) {
            next = next.advance(completed_update)?;
        } else if !success {
            if next.poor_streak == config.poor_updates {
                next.extension_awards =
                    next.extension_awards
                        .checked_add(1)
                        .ok_or(PpoError::InvalidTransition(
                            "environment extension awards addition overflow",
                        ))?;
            }
            let budget = next.effective_budget(config, limits)?;
            if completed_update < limits.total_updates && next.updates_in_generation >= budget {
                next = next.advance(completed_update)?;
            }
        }
        next.validate(config, limits, completed_update)?;
        Ok(next)
    }

    /// Checks structural controller invariants, not historical provenance of game results.
    /// Completed success streaks and exhausted budgets are retained only at total updates.
    pub fn validate(
        &self,
        config: AdaptiveEnvironmentConfig,
        limits: AdaptiveEnvironmentLimits,
        global_update: u64,
    ) -> Result<(), PpoError> {
        config.validate()?;
        limits.validate()?;
        let accounted_update = self
            .start_update
            .checked_add(self.updates_in_generation)
            .ok_or(PpoError::InvalidTransition(
                "environment global update addition overflow",
            ))?;
        if accounted_update != global_update {
            return Err(PpoError::InvalidTransition(
                "environment start update plus spent updates must equal global update",
            ));
        }
        if global_update > limits.total_updates {
            return Err(PpoError::InvalidTransition(
                "environment global update exceeds total updates",
            ));
        }
        if self.start_update >= limits.total_updates {
            return Err(PpoError::InvalidTransition(
                "environment start update must precede total updates",
            ));
        }
        if self.generation > self.start_update || (self.generation == 0) != (self.start_update == 0)
        {
            return Err(PpoError::InvalidTransition(
                "environment generation and start update are inconsistent",
            ));
        }
        let terminal = global_update == limits.total_updates;
        self.validate_streaks(config, terminal)?;
        let budget = self.effective_budget(config, limits)?;
        if self.updates_in_generation > budget {
            return Err(PpoError::InvalidTransition(
                "environment spent updates exceed effective budget",
            ));
        }
        if !terminal && self.success_streak == config.success_updates {
            return Err(PpoError::InvalidTransition(
                "environment success streak requires advancement before total updates",
            ));
        }
        if !terminal && self.updates_in_generation >= budget {
            return Err(PpoError::InvalidTransition(
                "environment budget is exhausted before total updates",
            ));
        }
        let boundary = limits.total_updates - limits.zero_updates;
        if boundary < limits.total_updates
            && global_update >= boundary
            && self.start_update < boundary
        {
            return Err(PpoError::InvalidTransition(
                "environment state crosses the zero phase boundary without a reset",
            ));
        }
        Ok(())
    }

    /// Base plus the floor of accumulated exact credit, without clamping to total updates.
    /// Checks config, limits, and arithmetic; use `validate` for full state coherence.
    pub fn effective_budget(
        &self,
        config: AdaptiveEnvironmentConfig,
        limits: AdaptiveEnvironmentLimits,
    ) -> Result<u64, PpoError> {
        config.validate()?;
        limits.validate()?;
        if self.extension_awards > MAX_TRAINING_COUNTER {
            return Err(PpoError::InvalidTransition(
                "environment extension awards exceed MAX_TRAINING_COUNTER",
            ));
        }
        let credit = u128::from(self.extension_awards)
            .checked_mul(u128::from(config.extension.units()))
            .ok_or(PpoError::InvalidTransition(
                "environment extension multiplication overflow",
            ))?;
        let budget = u128::from(limits.base_updates)
            .checked_add(credit / u128::from(EnvironmentDecimal::SCALE))
            .ok_or(PpoError::InvalidTransition(
                "environment effective budget addition overflow",
            ))?;
        if budget > u128::from(MAX_TRAINING_COUNTER) {
            return Err(PpoError::InvalidTransition(
                "environment effective budget exceeds MAX_TRAINING_COUNTER",
            ));
        }
        debug_assert!(budget >= u128::from(limits.base_updates));
        debug_assert!(budget <= u128::from(MAX_TRAINING_COUNTER));
        u64::try_from(budget).map_err(|_| {
            PpoError::InvalidTransition("environment effective budget conversion overflow")
        })
    }

    fn validate_streaks(
        &self,
        config: AdaptiveEnvironmentConfig,
        terminal: bool,
    ) -> Result<(), PpoError> {
        debug_assert!(config.success_updates > 0);
        debug_assert!(config.poor_updates > 0);
        if self.success_streak > config.success_updates
            || self.success_streak > self.updates_in_generation
        {
            return Err(PpoError::InvalidTransition(
                "environment success streak exceeds its window or spent updates",
            ));
        }
        if self.poor_streak > config.poor_updates || self.poor_streak > self.updates_in_generation {
            return Err(PpoError::InvalidTransition(
                "environment poor streak exceeds its window or spent updates",
            ));
        }
        if config.success_rate > config.poor_rate && self.success_streak > 0 && self.poor_streak > 0
        {
            return Err(PpoError::InvalidTransition(
                "environment disjoint thresholds cannot both have active streaks",
            ));
        }
        let terminal_success = terminal && self.success_streak == config.success_updates;
        let mut maximum_awards = self
            .updates_in_generation
            .saturating_sub(config.poor_updates - 1);
        if terminal_success && self.poor_streak == config.poor_updates {
            maximum_awards = maximum_awards.saturating_sub(1);
        }
        if self.extension_awards > maximum_awards {
            return Err(PpoError::InvalidTransition(
                "environment extension awards exceed qualifying updates",
            ));
        }
        if self.poor_streak == config.poor_updates
            && self.extension_awards == 0
            && !terminal_success
        {
            return Err(PpoError::InvalidTransition(
                "environment qualifying poor streak is missing an extension award",
            ));
        }
        Ok(())
    }

    fn advance(&self, completed_update: u64) -> Result<Self, PpoError> {
        debug_assert!(completed_update > self.start_update);
        debug_assert!(completed_update <= MAX_TRAINING_COUNTER);
        let generation = self
            .generation
            .checked_add(1)
            .filter(|generation| *generation <= MAX_TRAINING_COUNTER)
            .ok_or(PpoError::InvalidTransition(
                "environment generation addition overflow",
            ))?;
        Ok(Self {
            generation,
            start_update: completed_update,
            ..Self::default()
        })
    }
}

fn capped_streak(current: u64, window: u64, qualifies: bool) -> u64 {
    debug_assert!(window > 0);
    debug_assert!(window <= MAX_TRAINING_COUNTER);
    debug_assert!(current <= window);
    if qualifies {
        (current + 1).min(window)
    } else {
        0
    }
}
