use super::super::{CollectionReport, TrainingSession};
use super::*;
use crate::randomization::verify_generation_snapshots;
use crate::{AdaptiveEnvironmentCheckpoint, AdaptiveEnvironmentLimits, EnvironmentSchedule};

pub(super) fn initial_checkpoint(
    settings: &AnnealedJobConfig,
) -> Result<Option<AdaptiveEnvironmentCheckpoint>, PpoError> {
    let EnvironmentSchedule::Adaptive(config) = settings.environment_schedule else {
        return Ok(None);
    };
    let limits = AdaptiveEnvironmentLimits {
        base_updates: settings.generation_updates,
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
            || wins.iter().any(|wins| *wins > settings.slots as u64))
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
    // The scale ramp follows the adaptive suffix for both schedules and is
    // recorded only when it is not the full scale.
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
                1,
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
) -> Result<u64, PpoError> {
    if let Some(checkpoint) = state.adaptive_environment {
        crate::adaptive_randomization::verify_adaptive_snapshots(
            directory,
            settings.seed,
            1,
            &checkpoint,
            settings.scale,
        )?;
        Ok(checkpoint.snapshot_count)
    } else {
        verify_generation_snapshots(
            directory,
            settings.seed,
            settings.generation_updates,
            1,
            anneal_schedule(settings),
            // Collection has already drawn the generation of every pipelined update.
            state
                .completed_updates
                .checked_add(PIPELINE_STALENESS + 1)
                .ok_or(PpoError::CounterOverflow)?
                .min(settings.updates),
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
        // Collection lags the controller, so an update may pass the draw's nominal end.
        assert!(game >= draw.start_game);
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
        let (wins, games) = adaptive_wins(settings, harness, update, report)?;
        checkpoint.state = checkpoint.state.observe(
            checkpoint.config,
            checkpoint.limits,
            update.checked_add(1).ok_or(PpoError::CounterOverflow)?,
            wins,
            games,
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
                crate::telemetry::log_line!(
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

/// Wins and finished games of one update.
fn adaptive_wins(
    _settings: &AnnealedJobConfig,
    _harness: AnnealedHarness,
    _update: u64,
    report: &CollectionReport,
) -> Result<(u64, u64), PpoError> {
    let games = [
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.episode_timeouts,
    ]
    .into_iter()
    .try_fold(0u64, u64::checked_add)
    .ok_or(PpoError::CounterOverflow)?;
    #[cfg(test)]
    if let Some(wins) = _harness.adaptive_wins {
        let wins = wins
            .get(_update as usize)
            .copied()
            .ok_or(PpoError::InvalidTransition(
                "adaptive outcomes fixture exhausted",
            ))?;
        return Ok((wins, _settings.slots as u64));
    }
    if report.rejected_orders != 0 {
        return Err(PpoError::InvalidTransition(
            "adaptive outcomes require an unrejected update",
        ));
    }
    Ok((report.terminal_wins, games))
}
