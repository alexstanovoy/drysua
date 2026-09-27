use super::*;

#[cfg(feature = "builtin")]
pub(crate) fn training_settings_for_test(seed: u64, updates: u64) -> crate::TrainingJobConfig {
    crate::TrainingJobConfig {
        mastery_config: None,
        opponent_schedule: crate::TrainingOpponentSchedule::Teacher,
        episode_time_cost: 0.0,
        terminal_only: false,
        complete_episodes: false,
        pipeline_groups: 1,
        updates,
        ppo: PpoConfig {
            decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
            environments: 2,
            rollout_decisions: 2,
            epochs: 1,
            minibatch: 2,
            gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
            ..PpoConfig::default()
        },
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed,
        map: bota_proto::MapId(2),
        git_commit: "test-drysua-commit".to_owned(),
        simulator_commit: "test-bota-commit".to_owned(),
    }
}

pub(crate) fn test_directory(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-learning-{name}-{}-{sequence}",
        std::process::id()
    ));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&directory)
        .expect("unique private test directory");
    directory
}

#[cfg(test)]
pub(crate) fn open_unit_bounds_for_test() -> (f64, f64) {
    (
        open_unit_from_bits(0),
        open_unit_from_bits((1u64 << 52) - 1),
    )
}

impl PpoTrainer {
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
    pub(crate) fn corrupt_materialization_frame_for_test(&mut self, index: usize) {
        self.samples[index]
            .transition
            .frame
            .corrupt_unit_offset_for_test();
    }

    #[cfg(test)]
    pub(crate) fn reject_minibatch_for_test(&mut self, index: usize) {
        self.samples[index].transition.old_log_probability = -5.0;
    }

    #[cfg(test)]
    pub(crate) fn replace_advantage_for_test(&mut self, index: usize, value: f32) -> f32 {
        std::mem::replace(&mut self.samples[index].advantage, value)
    }
}
