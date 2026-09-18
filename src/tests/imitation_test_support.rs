use super::*;

impl ImitationPool {
    #[cfg(test)]
    pub(crate) fn clear_identity_history_for_test(&mut self) {
        self.seen_identities.clear();
    }
}

#[cfg(test)]
pub(crate) fn padded_mask_for_test<const WIDTH: usize>(
    mask: &[bool],
) -> Result<[bool; WIDTH], ImitationError> {
    padded_mask(mask)
}

impl TeacherCoverage {
    #[cfg(test)]
    pub(crate) fn record_represented(&mut self) -> Result<(), ImitationError> {
        self.ensure_capacity()?;
        self.attempted += 1;
        self.represented += 1;
        Ok(())
    }

    /// Runs one complete teacher-decision/target-construction operation as one attempt.
    #[cfg(test)]
    pub(crate) fn collect<T, E>(
        &mut self,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.ensure_capacity().map_err(CoverageError::Capacity)?;
        let result = operation();
        self.attempted += 1;
        if result.is_ok() {
            self.represented += 1;
        } else {
            self.failed += 1;
        }
        result.map_err(CoverageError::Operation)
    }

    /// Performs one identity-bound teacher attempt while retaining failures.
    #[cfg(test)]
    fn collect_for<T, E>(
        &mut self,
        identity: SampleIdentity,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.ensure_capacity().map_err(CoverageError::Capacity)?;
        self.validate_identity_metadata(identity)
            .map_err(CoverageError::Capacity)?;
        let result = operation();
        self.record_identity_after_capacity(identity, result.is_ok(), None);
        result.map_err(CoverageError::Operation)
    }

    #[cfg(test)]
    pub(crate) fn collect_for_test<T, E>(
        &mut self,
        identity: SampleIdentity,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, CoverageError<E>> {
        self.collect_for(identity, operation)
    }
}

impl BehavioralTrainer {
    #[cfg(test)]
    pub(crate) fn train_epoch_with_failure(
        &mut self,
        model: &PolicyModel,
        pool: &ImitationPool,
        update: usize,
        rollback_failure: bool,
    ) -> Result<TrainingEpochReport, ImitationError> {
        self.train_epoch_atomic(model, pool, Some(update), rollback_failure)
    }

    #[cfg(test)]
    pub(crate) fn orchestration_state_for_test(&self) -> (ShuffleState, PoolBinding) {
        (self.shuffle, self.pool.clone())
    }

    #[cfg(test)]
    pub(crate) fn set_shuffle_draws_for_test(&mut self, draws: u64) {
        self.shuffle.draws = draws;
    }

    #[cfg(test)]
    pub(crate) fn set_counters_for_test(
        &mut self,
        counters: TrainerCounters,
    ) -> Result<(), ImitationError> {
        let (first, second) = self.adam.moments();
        self.adam = AdamState::from_parts(
            self.adam.config(),
            first.to_vec(),
            second.to_vec(),
            counters.global_update,
            self.adam.binding(),
        )?;
        self.counters = counters;
        Ok(())
    }
}
