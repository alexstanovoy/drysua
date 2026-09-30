#[cfg(feature = "builtin")]
use crate::PpoError;

/// Completed training task result; infrastructure failures have no variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingGameOutcome {
    Win,
    Loss,
    Draw,
    TimeCap,
}

/// At most one completed outcome per update stream, sorted by tick then stream.
/// Sequential annealed batches can finish more games than the concurrent world cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedTrainingEpisodes {
    entries: [Option<(u32, usize, TrainingGameOutcome)>; crate::PPO_MAX_GAMES],
}

impl Default for CompletedTrainingEpisodes {
    fn default() -> Self {
        Self {
            entries: [None; crate::PPO_MAX_GAMES],
        }
    }
}

impl CompletedTrainingEpisodes {
    #[cfg(feature = "builtin")]
    pub(crate) fn record(
        &mut self,
        tick: u32,
        stream: usize,
        outcome: TrainingGameOutcome,
    ) -> Result<(), PpoError> {
        if tick == 0 || tick > crate::MAP2_TICK_CAP || stream >= self.entries.len() {
            return Err(PpoError::InvalidTransition(
                "completed episode tick or stream",
            ));
        }
        if self.entries[stream].is_some() {
            return Err(PpoError::InvalidTransition(
                "duplicate completed episode stream",
            ));
        }
        self.entries[stream] = Some((tick, stream, outcome));
        Ok(())
    }

    pub fn ordered_outcomes(&self) -> Vec<TrainingGameOutcome> {
        let mut entries: Vec<_> = self.entries.iter().flatten().copied().collect();
        assert!(entries.len() <= crate::PPO_MAX_GAMES);
        entries.sort_by_key(|entry| (entry.0, entry.1));
        entries.into_iter().map(|entry| entry.2).collect()
    }
}
