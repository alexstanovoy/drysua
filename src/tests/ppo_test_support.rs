use super::*;

#[cfg(test)]
pub(crate) fn open_unit_bounds_for_test() -> (f64, f64) {
    (
        open_unit_from_bits(0),
        open_unit_from_bits((1u64 << 52) - 1),
    )
}

impl PpoRollout {
    #[cfg(test)]
    pub(crate) fn ragged_rows_for_test(&self) -> usize {
        self.frames.stored_rows()
    }
}

impl PpoTrainer {
    #[cfg(test)]
    pub(crate) const fn shuffle_draws_for_test(&self) -> u64 {
        self.shuffle.draws()
    }

    #[cfg(test)]
    pub(crate) fn train_pipeline_update_with_barriers_for_test(
        &mut self,
        model: &PolicyModel,
        pipeline: &crate::PipelineBatch,
        entered: &std::sync::Barrier,
        release: &std::sync::Barrier,
    ) -> Result<PpoUpdateReport, PpoError> {
        let generation = self.lock_pipeline_update(model, pipeline)?;
        entered.wait();
        release.wait();
        let report = self.train_accepted_update(model, &pipeline.batch);
        drop(generation);
        report
    }
}

impl PpoBatch {
    #[cfg(test)]
    pub(crate) fn corrupt_prefetch_frame_for_test(&mut self, index: usize) {
        self.samples[index]
            .transition
            .frame
            .corrupt_unit_offset_for_test();
    }

    #[cfg(test)]
    pub(crate) fn reject_prefetch_minibatch_for_test(&mut self, index: usize) {
        self.samples[index].transition.old_log_probability = -5.0;
    }

    #[cfg(test)]
    pub(crate) fn replace_advantage_for_test(&mut self, index: usize, value: f32) -> f32 {
        std::mem::replace(&mut self.samples[index].advantage, value)
    }
}
