use super::*;
use crate::{AdaptiveEnvironmentCheckpoint, AdaptiveEnvironmentLimits, EnvironmentSchedule};

pub(super) fn initial_checkpoint(
    settings: &AnnealedJobConfig,
) -> Result<Option<AdaptiveEnvironmentCheckpoint>, PpoError> {
    let EnvironmentSchedule::Adaptive(config) = settings.environment_schedule else {
        return Ok(None);
    };
    if settings.games_per_update == 0
        || !settings
            .games_per_generation
            .is_multiple_of(settings.games_per_update as u64)
    {
        return Err(PpoError::InvalidConfig(
            "adaptive generation must contain whole updates",
        ));
    }
    let limits = AdaptiveEnvironmentLimits {
        base_updates: settings.games_per_generation / settings.games_per_update as u64,
        total_updates: settings.updates,
        zero_updates: settings.zero_updates,
    }
    .validate()?;
    Ok(Some(AdaptiveEnvironmentCheckpoint {
        config: config.validate()?,
        limits,
        state: Default::default(),
        snapshot_count: 0,
        snapshot_hash: [0; 32],
    }))
}

pub(super) fn validate(
    settings: &AnnealedJobConfig,
    _harness: AnnealedHarness,
) -> Result<(), PpoError> {
    initial_checkpoint(settings)?;
    #[cfg(test)]
    if let Some(wins) = _harness.adaptive_wins
        && (!matches!(
            settings.environment_schedule,
            EnvironmentSchedule::Adaptive(_)
        ) || _harness.episode_decisions.is_none()
            || wins.len() > 256
            || wins.len() as u64 != settings.updates
            || wins
                .iter()
                .any(|wins| *wins > settings.games_per_update as u64))
    {
        return Err(PpoError::InvalidConfig("adaptive outcomes fixture scope"));
    }
    Ok(())
}

pub(super) fn append_scope(
    settings: &AnnealedJobConfig,
    _harness: AnnealedHarness,
    command: &mut String,
) {
    #[cfg(test)]
    if let Some(wins) = _harness.adaptive_wins {
        assert!(wins.len() <= 256);
        command.push_str(" --test-adaptive-wins ");
        for (index, wins) in wins.iter().enumerate() {
            if index > 0 {
                command.push(',');
            }
            command.push_str(&wins.to_string());
        }
    }
    if let EnvironmentSchedule::Adaptive(config) = settings.environment_schedule {
        config.append_scope(command);
    }
    // The scale ramp is recorded after the adaptive suffix for both schedules,
    // in a fixed order, and only when it differs from the historical default.
    if settings.scale != crate::randomization::AnnealScale::FULL {
        command.push_str(&settings.scale.scope_suffix());
    }
}

pub(super) fn preflight_resume(
    settings: &AnnealedJobConfig,
    run: &CheckpointRun,
    directory: &Path,
) -> Result<(), PpoError> {
    let (stored, progress) =
        TrainingArtifact::load_resume_metadata(directory).map_err(text_error)?;
    if &stored != run {
        return Err(text_error(crate::CheckpointError::InvalidManifest(
            "compatibility scope",
        )));
    }
    match (initial_checkpoint(settings)?, progress.adaptive_environment) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual))
            if expected.config == actual.config && expected.limits == actual.limits =>
        {
            crate::adaptive_randomization::verify_adaptive_snapshots(
                &directory.join(RANDOMIZATION_DIRECTORY),
                settings.seed,
                settings.games_per_update as u64,
                &actual,
                settings.scale,
            )
        }
        _ => Err(PpoError::InvalidConfig(
            "adaptive environment configuration/state mismatch",
        )),
    }
}

pub(super) fn verified_generation_count(
    settings: &AnnealedJobConfig,
    directory: &Path,
    state: &TrainingSession,
    games: u64,
) -> Result<u64, PpoError> {
    if let Some(checkpoint) = state.adaptive_environment {
        crate::adaptive_randomization::verify_adaptive_snapshots(
            directory,
            settings.seed,
            settings.games_per_update as u64,
            &checkpoint,
            settings.scale,
        )?;
        Ok(checkpoint.snapshot_count)
    } else {
        verify_generation_snapshots(
            directory,
            settings.seed,
            settings.games_per_generation,
            settings.games_per_update as u64,
            anneal_schedule(settings),
            games,
        )
    }
}

impl GenerationCache {
    pub(super) fn draw_for_game(&mut self, game: u64) -> Result<GenerationDraw, PpoError> {
        let Some(checkpoint) = &mut self.adaptive else {
            return self.draw(game / self.games_per_generation);
        };
        if self
            .last
            .as_ref()
            .is_none_or(|draw| draw.generation != checkpoint.state.generation)
        {
            // The adaptive draw ramps with the configured environment scale.
            let scale = self.schedule.scale;
            self.last = Some(crate::adaptive_randomization::draw_adaptive_generation(
                &self.directory,
                self.seed,
                self.games_per_update,
                checkpoint,
                scale,
            )?);
        }
        let draw = self.last.expect("adaptive draw materialized");
        assert!(game >= draw.start_game);
        assert!(game < draw.end_game);
        Ok(draw)
    }

    pub(super) fn next_adaptive(
        &self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        update: u64,
        report: &CollectionReport,
    ) -> Result<Option<AdaptiveEnvironmentCheckpoint>, PpoError> {
        let Some(mut checkpoint) = self.adaptive else {
            return Ok(None);
        };
        let wins = adaptive_wins(settings, harness, update, report)?;
        checkpoint.state = checkpoint.state.observe(
            checkpoint.config,
            checkpoint.limits,
            update.checked_add(1).ok_or(PpoError::CounterOverflow)?,
            wins,
            settings.games_per_update as u64,
        )?;
        Ok(Some(checkpoint))
    }

    pub(super) fn commit_adaptive(
        &mut self,
        next: Option<AdaptiveEnvironmentCheckpoint>,
        update: u64,
    ) {
        if let (Some(previous), Some(next)) = (self.adaptive, next) {
            assert_eq!(
                next.state.start_update + next.state.updates_in_generation,
                update
            );
            if previous.state.generation != next.state.generation {
                eprintln!(
                    "level=INFO event=adaptive_environment_transition update={update} previous_generation={} generation={} start_update={} previous_awards={} clean={}",
                    previous.state.generation,
                    next.state.generation,
                    next.state.start_update,
                    previous.state.extension_awards,
                    update >= next.limits.total_updates - next.limits.zero_updates
                );
            }
        }
        self.adaptive = next;
    }
}

fn adaptive_wins(
    settings: &AnnealedJobConfig,
    _harness: AnnealedHarness,
    _update: u64,
    report: &CollectionReport,
) -> Result<u64, PpoError> {
    #[cfg(test)]
    if let Some(wins) = _harness.adaptive_wins {
        return wins
            .get(_update as usize)
            .copied()
            .ok_or(PpoError::InvalidTransition(
                "adaptive outcomes fixture exhausted",
            ));
    }
    let outcomes = report.completed_episodes.ordered_outcomes();
    let wins = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, crate::TrainingGameOutcome::Win))
        .count();
    if outcomes.len() != settings.games_per_update
        || wins as u64 != report.terminal_wins
        || report.rejected_orders != 0
    {
        return Err(PpoError::InvalidTransition(
            "adaptive outcomes require a complete unrejected update",
        ));
    }
    Ok(report.terminal_wins)
}
