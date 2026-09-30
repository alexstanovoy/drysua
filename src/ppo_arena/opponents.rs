//! Opponent scheduling: the configured pool, this run's league snapshots and
//! prioritized fictitious self-play (PFSP) weights.
//!
//! Every published update gets its own per-game mixture. Its entries are the
//! configured opponents plus, with a league, the `size` latest runtime-history
//! milestones of this run. Under PFSP each entry's configured weight is scaled
//! by `(1 - p)^2`, at least 1/10, where `p` is the Laplace-smoothed score (win 1, draw 1/2)
//! of the learner's last [`PFSP_WINDOW`] games against it. Weights are exact
//! integers computed from logged outcomes of updates every lane has finished,
//! so the schedule is a pure function of the run and resumes identically.
#![allow(
    clippy::float_arithmetic,
    reason = "Only the logged win rates and probabilities are floating point"
)]

use std::collections::VecDeque;

use super::slot::{OpponentKind, OpponentMixture};
use crate::{PpoError, PpoTerminalOutcome};

/// Games per opponent that form its PFSP win rate.
pub(crate) const PFSP_WINDOW: usize = 100;
/// Every opponent keeps at least this fraction (one over it) of its configured
/// weight: an update still plays the styles the learner already beats, so it
/// cannot overfit to the hardest one and forget the rest.
const PFSP_FLOOR_DIVISOR: u64 = 10;
/// Most league milestones playing at once.
pub(crate) const MAX_LEAGUE_SIZE: usize = 16;
/// Most entries of one update's mixture: configured opponents plus the league.
pub(crate) const MAX_MIXTURE_ENTRIES: usize = 16 + MAX_LEAGUE_SIZE;

/// How configured weights become one update's mixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentSchedule {
    /// Configured weights, unchanged.
    Fixed,
    /// Configured weights times `(1 - smoothed win rate)^2`.
    Pfsp,
}

impl OpponentSchedule {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Pfsp => "pfsp",
        }
    }
}

/// This run's own milestone snapshots as opponents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct League {
    /// Configured weight of each member, in mixture units.
    pub(crate) weight: u64,
    pub(crate) size: usize,
    /// Milestone spacing of the runtime history.
    pub(crate) every: u64,
}

impl League {
    /// Milestones playing update `update`, oldest first: the `size` latest at
    /// or below `update - 1`, whose weights exist when `update` is published.
    pub(crate) fn members(&self, update: u64) -> Vec<u64> {
        assert!((1..=MAX_LEAGUE_SIZE).contains(&self.size));
        assert!(self.every > 0);
        let Some(latest) = update.checked_sub(1) else {
            return Vec::new();
        };
        let newest = latest / self.every * self.every;
        let mut members = (0..self.size as u64)
            .map_while(|back| newest.checked_sub(back * self.every))
            .collect::<Vec<_>>();
        members.reverse();
        members
    }
}

/// The learner's recent results against each opponent, oldest first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OutcomeWindow {
    /// Scores in half points: win 2, draw 1, loss or time cap 0.
    pub(crate) entries: Vec<(OpponentKind, VecDeque<u8>)>,
}

impl OutcomeWindow {
    /// Books one finished game, forgetting the oldest beyond the window.
    pub(crate) fn record(
        &mut self,
        opponent: OpponentKind,
        outcome: Option<PpoTerminalOutcome>,
    ) -> Result<(), PpoError> {
        let score = match outcome {
            Some(PpoTerminalOutcome::Win) => 2,
            Some(PpoTerminalOutcome::Draw) => 1,
            Some(PpoTerminalOutcome::Loss) | None => 0,
        };
        let index = match self.entries.iter().position(|(kind, _)| *kind == opponent) {
            Some(index) => index,
            None => {
                if self.entries.len() >= MAX_MIXTURE_ENTRIES + crate::PPO_MAX_SLOTS {
                    return Err(PpoError::InvalidConfig("PFSP window entries"));
                }
                self.entries
                    .push((opponent, VecDeque::with_capacity(PFSP_WINDOW)));
                self.entries.len() - 1
            }
        };
        let scores = &mut self.entries[index].1;
        if scores.len() == PFSP_WINDOW {
            scores.pop_front();
        }
        scores.push_back(score);
        Ok(())
    }

    /// Games and half-point score against `opponent`.
    pub(crate) fn tally(&self, opponent: OpponentKind) -> (u64, u64) {
        self.entries
            .iter()
            .find(|(kind, _)| *kind == opponent)
            .map_or((0, 0), |(_, scores)| {
                (
                    scores.len() as u64,
                    scores.iter().map(|&score| u64::from(score)).sum(),
                )
            })
    }

    /// Forgets opponents that no longer play.
    pub(crate) fn retain(&mut self, playing: impl Fn(OpponentKind) -> bool) {
        self.entries.retain(|(kind, _)| playing(*kind));
    }
}

/// One update's mixture from configured weights and the recent outcomes.
pub(crate) fn schedule_mixture(
    entries: &[(OpponentKind, u64)],
    window: &OutcomeWindow,
    schedule: OpponentSchedule,
) -> Result<OpponentMixture, PpoError> {
    let weighted = entries
        .iter()
        .map(|&(kind, weight)| {
            let weight = match schedule {
                OpponentSchedule::Fixed => weight,
                OpponentSchedule::Pfsp => pfsp_weight(weight, window.tally(kind)),
            };
            (kind, weight)
        })
        .collect();
    OpponentMixture::new(weighted)
}

/// `weight * max((1 - p)^2, 1/10)` with `p = (score / 2 + 1) / (games + 2)`, at least one unit.
fn pfsp_weight(weight: u64, (games, score): (u64, u64)) -> u64 {
    assert!(score <= 2 * games);
    let numerator = u128::from(2 * games + 2 - score);
    let denominator = u128::from(2 * games + 4);
    let scaled = u128::from(weight) * numerator * numerator / (denominator * denominator);
    u64::try_from(scaled)
        .expect("never above the configured weight")
        .max(weight / PFSP_FLOOR_DIVISOR)
        .max(1)
}

/// Logs each entry's recent record and its share of the next mixture.
pub(crate) fn log_pool(update: u64, mixture: &OpponentMixture, window: &OutcomeWindow) {
    for &(kind, weight) in mixture.entries() {
        let (games, score) = window.tally(kind);
        let win_rate = if games == 0 {
            "nan".to_owned()
        } else {
            format!("{:.4}", score as f64 / (2 * games) as f64)
        };
        eprintln!(
            "level=INFO event=opponent_pool update={update} scope={} games={games} score={:.1} win_rate={win_rate} probability={:.4}",
            kind.label(),
            score as f64 / 2.0,
            weight as f64 / mixture.total() as f64,
        );
    }
}

#[cfg(test)]
#[path = "../tests/opponents.rs"]
mod tests;
