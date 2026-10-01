//! One annealed invocation: the learner loop around continuous collection.
//!
//! Update `u` trains on the retained intervals every lane closed while
//! collecting `u`. Lanes collect `u` with the weights of update
//! `u - PIPELINE_STALENESS`, so while the learner trains `u` the lanes already
//! collect `u + 1`. Configurations are published at fixed update indices, never
//! in reaction to timing, and checkpoint `u` records the collection state every
//! lane had when it started collecting `u`.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::collector::{Collector, Prepared};
use super::super::collector_state::CollectorState;
use super::super::lane::{LaneSettings, LaneStart, PartConfig, PartDone};
use super::super::league::LeagueStore;
use super::super::opponents::{OutcomeWindow, log_pool, schedule_mixture};
use super::super::pool::SimPool;
use super::super::slot::{GameSchedule, OpponentKind, OpponentMixture, SlotSnapshot};
use super::super::win_model::{WinModelConfig, WinModels, WinState};
use super::super::{CollectionReport, TrainingSession};
use super::*;
use crate::CollectionCheckpoint;
use crate::telemetry::{TrainingStage, TrainingUpdateMode, TrainingUpdateTimer};

pub(super) struct AnnealedSession {
    state: TrainingSession,
    random_directory: PathBuf,
    generations: u64,
    games: u64,
    /// Collection state the next update starts from, when resuming.
    restored: Option<CollectorState>,
    /// Recent outcomes of every finished update, per opponent.
    outcomes: OutcomeWindow,
    /// League snapshots the next publications or in-flight games need.
    league: LeagueStore,
    /// The learned potential's refit settings, window and models, when the run learns one.
    win: Option<(WinModelConfig, WinState, WinModels)>,
}

impl AnnealedSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn initialize(
        settings: &AnnealedJobConfig,
        device: PolicyDevice,
        directory: &Path,
        resume: bool,
        initial_weights_directory: Option<&Path>,
        config: PpoConfig,
        run: CheckpointRun,
        random_directory: &Path,
    ) -> Result<Self, PpoError> {
        let mut state = TrainingSession::initialize(
            device,
            directory,
            resume,
            initial_weights_directory,
            config,
            run,
        )?;
        let (generations, restored) = if resume {
            std::fs::metadata(random_directory).map_err(|_| {
                PpoError::InvalidConfig("domain randomization snapshots are missing on resume")
            })?;
            let generations =
                adaptive::verified_generation_count(settings, random_directory, &state)?;
            (generations, Some(restored_collection(settings, &state)?))
        } else {
            state.adaptive_environment = adaptive::initial_checkpoint(settings)?;
            (0, None)
        };
        state.trainer.set_execution(settings.execution)?;
        let outcomes = restored
            .as_ref()
            .map_or_else(OutcomeWindow::default, |state| state.outcomes.clone());
        let league = match &restored {
            Some(collection) => restore_league(settings, &state, directory, collection)?,
            None => LeagueStore::default(),
        };
        let win = match settings.potential {
            Some(config) => {
                let window = restored
                    .as_ref()
                    .map_or_else(WinState::default, |state| state.win.clone());
                let models = WinModels::default();
                window.publish(&models)?;
                Some((config, window, models))
            }
            None => None,
        };
        Ok(Self {
            state,
            random_directory: random_directory.to_path_buf(),
            generations,
            games: 0,
            restored,
            outcomes,
            league,
            win,
        })
    }

    pub(super) fn run_updates(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        config: PpoConfig,
        directory: &Path,
        pool: &OpponentPool,
        checkpointed: &mut impl FnMut(TrainingCheckpointReport),
    ) -> Result<(), PpoError> {
        let target = self.invocation_target(settings)?;
        if self.state.completed_updates >= target {
            return Ok(());
        }
        let schedule = GameSchedule {
            seed: settings.seed,
            decision_cap: harness.episode_decisions(),
            config,
            shadow: settings.guidance.shadow_labels(),
            potentials: self.win.as_ref().map(|(_, _, models)| Arc::clone(models)),
        };
        let mut generations = self.generation_cache(settings);
        let lanes = self.lane_plan(settings, pool)?;
        let cpus = crate::ppo_arena::topology::group_cpus(
            settings.simulation_groups,
            settings.pin_threads,
        )?;
        let result = std::thread::scope(|scope| {
            let groups = settings.simulation_groups;
            let mut pools = Vec::with_capacity(groups);
            for (group, cpus) in cpus.into_iter().enumerate() {
                // Threads split as evenly as possible; the first groups take the remainder.
                let threads = settings.simulation_threads / groups
                    + usize::from(group < settings.simulation_threads % groups);
                let pool = SimPool::spawn(
                    scope,
                    (group, threads),
                    settings.slots / groups,
                    &schedule,
                    cpus.as_deref(),
                )?;
                pools.push((pool, cpus));
            }
            let collector = Collector::spawn(
                scope,
                lanes,
                pools,
                &schedule,
                (self.state.completed_updates, settings.slots, settings.ppo),
            )?;
            let mut pipeline = Pipeline {
                collector,
                configs: BTreeMap::new(),
            };
            self.publish_initial(settings, pool, &mut pipeline, &mut generations)?;
            self.update_loop(
                settings,
                (harness, pool),
                directory,
                target,
                (&mut pipeline, &mut generations),
                checkpointed,
            )
        });
        self.generations = generations.counted_through();
        result
    }

    /// The generation draws of this invocation, resuming the recorded snapshots.
    fn generation_cache(&self, settings: &AnnealedJobConfig) -> GenerationCache {
        let mut generations = GenerationCache::new(
            self.random_directory.clone(),
            settings.seed,
            settings.generation_updates,
            anneal_schedule(settings),
            self.generations,
        );
        generations.adaptive = self.state.adaptive_environment;
        generations
    }

    /// The update count this invocation stops at.
    fn invocation_target(&self, settings: &AnnealedJobConfig) -> Result<u64, PpoError> {
        Ok(match settings.invocation_updates {
            Some(limit) => self
                .state
                .completed_updates
                .checked_add(limit.get())
                .ok_or(PpoError::CounterOverflow)?
                .min(settings.updates),
            None => settings.updates,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn update_loop(
        &mut self,
        settings: &AnnealedJobConfig,
        (harness, pool): (AnnealedHarness, &OpponentPool),
        directory: &Path,
        target: u64,
        (pipeline, generations): (&mut Pipeline, &mut GenerationCache),
        checkpointed: &mut impl FnMut(TrainingCheckpointReport),
    ) -> Result<(), PpoError> {
        let started = std::time::Instant::now();
        let mut schedule =
            super::super::TrainingCheckpointSchedule::new(settings.checkpoint_cadence)?;
        let mut committed = self.state.completed_updates;
        while self.state.completed_updates < target {
            self.train_update(settings, (harness, pool), pipeline, generations)?;
            let report = self.state.checkpoint_report(None);
            report.log_progress();
            let completed = self.state.completed_updates;
            let stop = crate::training_signals::stop_requested()
                || harness.stop_after.is_some_and(|stop| completed >= stop);
            if schedule.is_due(completed, started.elapsed()) || completed == target || stop {
                // Snapshots and milestones precede the manifest, so a commit implies them.
                generations.write_pending()?;
                self.export_history(settings, committed)?;
                let league = self.league.manifest(completed);
                self.league.persist(directory, &league)?;
                let durable = crate::telemetry::time_training_checkpoint(completed, || {
                    self.state.save(directory, report)
                })?;
                LeagueStore::prune(directory, &league)?;
                checkpointed(durable);
                schedule.mark_committed(started.elapsed())?;
                committed = completed;
            }
            if stop {
                break;
            }
        }
        Ok(())
    }

    fn export_history(&self, settings: &AnnealedJobConfig, previous: u64) -> Result<(), PpoError> {
        match &settings.history {
            Some(history) => history.export(
                &self.state.model,
                self.state.completed_updates,
                previous,
                settings.updates,
            ),
            None => Ok(()),
        }
    }

    /// Splits slots across lanes and gives each lane its fresh or replayed start.
    fn lane_plan(
        &mut self,
        settings: &AnnealedJobConfig,
        pool: &OpponentPool,
    ) -> Result<Vec<(LaneSettings, LaneStart)>, PpoError> {
        let per_lane = settings.slots / settings.lanes;
        let snapshots = Arc::new(pool.snapshots.clone());
        let mut restored: Vec<Option<SlotSnapshot>> = match self.restored.take() {
            Some(state) => state.slots.into_iter().map(Some).collect(),
            None => Vec::new(),
        };
        let replay = !restored.is_empty();
        let mut lanes = Vec::with_capacity(settings.lanes);
        for lane in 0..settings.lanes {
            let slots: Vec<usize> = (0..per_lane)
                .map(|offset| offset * settings.lanes + lane)
                .collect();
            let start = if replay {
                LaneStart::Replay(
                    slots
                        .iter()
                        .map(|&slot| restored[slot].take())
                        .collect::<Option<Vec<_>>>()
                        .ok_or(PpoError::InvalidConfig("collector snapshot slot layout"))?,
                )
            } else {
                LaneStart::Fresh
            };
            let league = match &start {
                LaneStart::Replay(games) => self.league.weights(league_references(games))?,
                LaneStart::Fresh => Vec::new(),
            };
            let settings = LaneSettings {
                index: lane,
                slots,
                target: settings.ppo.samples_per_update / settings.lanes,
                device: self.state.model.device(),
                snapshots: Arc::clone(&snapshots),
                league,
            };
            lanes.push((settings, start));
        }
        Ok(lanes)
    }

    /// The collection configuration of `update` with its opponent mixture and league.
    fn part_config(
        &self,
        pool: &OpponentPool,
        (update, version, actor): (u64, u64, Arc<Vec<f32>>),
        (spec, mixture, potential): (ModifierSpec, Option<OpponentMixture>, Option<u64>),
    ) -> Result<PartConfig, PpoError> {
        let potential = potential
            .unwrap_or_else(|| self.win.as_ref().map_or(0, |(_, window, _)| window.current));
        let entries = pool.entries_for(update);
        let mixture = match mixture {
            Some(mixture) => mixture,
            None => schedule_mixture(&entries, &self.outcomes, pool.schedule)?,
        };
        let league = self
            .league
            .weights(entries.iter().filter_map(|(kind, _)| match kind {
                OpponentKind::League(milestone) => Some(*milestone),
                _ => None,
            }))?;
        Ok(PartConfig {
            update,
            version,
            actor,
            spec,
            mixture: Arc::new(mixture),
            league,
            potential,
        })
    }

    /// Takes a league snapshot of the just-completed update when one is due.
    fn remember_milestone(&mut self, pool: &OpponentPool, actor: &Arc<Vec<f32>>) {
        let completed = self.state.completed_updates;
        if pool
            .league
            .is_some_and(|league| completed.is_multiple_of(league.every))
        {
            self.league.insert(completed, Arc::clone(actor));
        }
    }

    /// Forgets snapshots that neither the current nor the next publication
    /// draws and no in-flight game plays.
    fn prune_league(&mut self, pool: &OpponentPool, snapshots: &[SlotSnapshot], next: u64) {
        let Some(league) = pool.league else {
            return;
        };
        let mut kept = league.members(next - 1);
        kept.extend(league.members(next));
        kept.extend(league_references(snapshots));
        self.league.retain(|update| kept.contains(&update));
    }

    /// Publishes the configurations of every update collected before the
    /// learner's next result: the current update and the pipelined ones.
    fn publish_initial(
        &mut self,
        settings: &AnnealedJobConfig,
        pool: &OpponentPool,
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        let completed = self.state.completed_updates;
        let current = Arc::new(self.state.model.export_parameters().map_err(text_error)?);
        if completed == 0 {
            self.remember_milestone(pool, &current);
        }
        let first = match &self.state.collection {
            Some(checkpoint) if completed > 0 => {
                let state = CollectorState::decode(&checkpoint.state)?;
                self.part_config(
                    pool,
                    (
                        completed,
                        state.actor_version,
                        Arc::new(checkpoint.actor.clone()),
                    ),
                    (state.spec, Some(state.mixture), Some(state.potential)),
                )?
            }
            _ => {
                let spec = run_spec(settings, generations, completed)?;
                self.part_config(
                    pool,
                    (completed, completed, Arc::clone(&current)),
                    (spec, None, None),
                )?
            }
        };
        pipeline.publish(first)?;
        for offset in 1..=PIPELINE_STALENESS {
            let update = completed + offset;
            let spec = run_spec(settings, generations, update)?;
            let config = self.part_config(
                pool,
                (update, completed, Arc::clone(&current)),
                (spec, None, None),
            )?;
            pipeline.publish(config)?;
        }
        Ok(())
    }

    fn train_update(
        &mut self,
        settings: &AnnealedJobConfig,
        context: (AnnealedHarness, &OpponentPool),
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        let mut timing = TrainingUpdateTimer::new(
            self.state.completed_updates,
            TrainingUpdateMode::Annealed,
            self.state.trainer.optimizer_step(),
        );
        let result = self.train_update_timed(settings, context, pipeline, generations, &mut timing);
        timing.observe_result(result)?;
        timing.complete();
        Ok(())
    }

    fn train_update_timed(
        &mut self,
        settings: &AnnealedJobConfig,
        (harness, pool): (AnnealedHarness, &OpponentPool),
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
        timing: &mut TrainingUpdateTimer,
    ) -> Result<(), PpoError> {
        let update = self.state.completed_updates;
        let config = settings.ppo;
        timing.enter(TrainingStage::Collection);
        let prepared = pipeline.collector.take(update)?;
        timing.enter(TrainingStage::BatchPreparation);
        log_parts(&prepared);
        self.record_outcomes(&prepared.parts, pool, update)?;
        let Prepared {
            samples,
            batch,
            report,
            snapshots,
            ..
        } = prepared;
        timing.set_samples(samples);
        let next_adaptive = generations.next_adaptive(settings, harness, update, &report)?;
        let explained_variance = (batch.explained_variance(), batch.side_statistics());
        let optimizer_step = self.state.trainer.optimizer_step();
        timing.enter(TrainingStage::Optimization);
        let objective = settings.guidance.objective(update);
        let optimized = self
            .state
            .trainer
            .train_update(&self.state.model, &batch, objective);
        timing.set_optimizer_step(self.state.trainer.optimizer_step());
        self.state.latest = optimized?;
        if let Some(usage) = crate::model::vram_usage().map_err(text_error)? {
            crate::telemetry::log_line!(
                "level=INFO event=device_memory update={update} budget_mib={} reserved_mib={} used_mib={} used_high_mib={}",
                usage.budget >> 20,
                usage.reserved >> 20,
                usage.used >> 20,
                usage.used_high >> 20,
            );
        }
        log_ppo_update(
            &self.state.latest,
            explained_variance,
            self.state.trainer.optimizer_step() - optimizer_step,
            samples,
            config.learning_rate,
        );
        timing.enter(TrainingStage::Finalization);
        self.commit_update(samples, &report, next_adaptive, generations)?;
        let completed = self.state.completed_updates;
        let next = completed + PIPELINE_STALENESS;
        let actor = Arc::new(self.state.model.export_parameters().map_err(text_error)?);
        self.remember_milestone(pool, &actor);
        self.prune_league(pool, &snapshots, next);
        let spec = run_spec(settings, generations, next)?;
        self.refit_potential(completed)?;
        let published = self.part_config(pool, (next, completed, actor), (spec, None, None))?;
        log_pool(completed, &published.mixture, &self.outcomes);
        pipeline.publish(published)?;
        self.retain_potentials(&snapshots, pipeline, completed)?;
        let opponents = (&self.outcomes, self.league.manifest(completed));
        let win = self.win.as_ref().map(|(_, window, _)| window);
        self.state.collection = Some(pipeline.checkpoint(completed, snapshots, opponents, win)?);
        Ok(())
    }

    /// Refits the learned potential when due and shares the new model with the lanes.
    fn refit_potential(&mut self, completed: u64) -> Result<(), PpoError> {
        let Some((config, window, models)) = &mut self.win else {
            return Ok(());
        };
        if let Some(line) = window.refit(*config, completed) {
            crate::telemetry::log_line!("{line}");
        }
        window.publish(models)
    }

    /// Forgets learned potentials that no in-flight game, published
    /// configuration or new game uses any more.
    fn retain_potentials(
        &mut self,
        snapshots: &[SlotSnapshot],
        pipeline: &Pipeline,
        completed: u64,
    ) -> Result<(), PpoError> {
        let Some((_, window, models)) = &mut self.win else {
            return Ok(());
        };
        let referenced = |version: u64| {
            snapshots.iter().any(|slot| slot.plan.potential == version)
                || pipeline
                    .configs
                    .range(completed..)
                    .any(|(_, config)| config.potential == version)
        };
        window.retain(referenced);
        let current = window.current;
        models
            .write()
            .map_err(|_| PpoError::InvalidTransition("learned potential registry"))?
            .retain(|&version, _| version == current || referenced(version));
        Ok(())
    }

    /// Books every game the update's parts finished, then forgets opponents
    /// that no longer play from the next publication on.
    fn record_outcomes(
        &mut self,
        parts: &[PartDone],
        pool: &OpponentPool,
        update: u64,
    ) -> Result<(), PpoError> {
        for episode in parts.iter().flat_map(|part| &part.episodes) {
            self.outcomes.record(episode.opponent, episode.outcome)?;
            if let (Some((config, window, _)), Some(game)) = (&mut self.win, &episode.win) {
                window.record(*config, game.clone());
            }
        }
        let playing = pool.entries_for(update + 1 + PIPELINE_STALENESS);
        self.outcomes
            .retain(|kind| playing.iter().any(|(entry, _)| *entry == kind));
        Ok(())
    }

    fn commit_update(
        &mut self,
        samples: usize,
        report: &CollectionReport,
        next_adaptive: Option<crate::AdaptiveEnvironmentCheckpoint>,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        self.state.completed_updates = self.state.trainer.updates();
        self.state.rollout_samples = self
            .state
            .rollout_samples
            .checked_add(samples as u64)
            .ok_or(PpoError::CounterOverflow)?;
        self.state.counters.merge(report)?;
        self.games = [
            report.terminal_wins,
            report.terminal_losses,
            report.terminal_draws,
            report.episode_timeouts,
        ]
        .into_iter()
        .try_fold(self.games, u64::checked_add)
        .ok_or(PpoError::CounterOverflow)?;
        generations.commit_adaptive(next_adaptive, self.state.completed_updates);
        self.state.adaptive_environment = next_adaptive;
        Ok(())
    }

    pub(super) fn report(&self) -> AnnealedJobReport {
        let state = &self.state;
        AnnealedJobReport {
            starting_policy_fingerprint: state.starting_policy_fingerprint,
            completed_updates: state.completed_updates,
            optimizer_step: state.trainer.optimizer_step(),
            rollout_samples: state.rollout_samples,
            games: self.games,
            generations: self.generations,
            map2_reward: state.counters.map2_reward,
            episode_timeouts: state.counters.episode_timeouts,
            terminal_wins: state.counters.terminal_wins,
            terminal_losses: state.counters.terminal_losses,
            terminal_draws: state.counters.terminal_draws,
            elapsed_ticks: state.counters.elapsed_ticks,
            latest: state.latest,
        }
    }
}

/// The collector plus every published configuration a checkpoint may still need.
struct Pipeline {
    collector: Collector,
    configs: BTreeMap<u64, Arc<PartConfig>>,
}

impl Pipeline {
    fn publish(&mut self, config: PartConfig) -> Result<(), PpoError> {
        let config = Arc::new(config);
        self.collector.publish(&config)?;
        let previous = self.configs.insert(config.update, config);
        assert!(previous.is_none());
        assert!(self.configs.len() <= PIPELINE_STALENESS as usize + 2);
        Ok(())
    }

    /// The collection checkpoint of `update`: the lanes' snapshots at its start
    /// and the weights it is collected with.
    fn checkpoint(
        &mut self,
        update: u64,
        snapshots: Vec<SlotSnapshot>,
        (outcomes, league): (&OutcomeWindow, Vec<(u64, u64)>),
        win: Option<&WinState>,
    ) -> Result<CollectionCheckpoint, PpoError> {
        self.configs.retain(|&published, _| published >= update);
        let config = self
            .configs
            .get(&update)
            .ok_or(PpoError::InvalidTransition("collection configuration"))?;
        let state = CollectorState {
            update,
            actor_version: config.version,
            spec: config.spec,
            mixture: config.mixture.as_ref().clone(),
            outcomes: outcomes.clone(),
            league,
            slots: snapshots,
            potential: config.potential,
            win: win.cloned().unwrap_or_default(),
        };
        Ok(CollectionCheckpoint {
            actor: config.actor.as_ref().clone(),
            state: state.encode(),
        })
    }
}

/// Logs the update's finished games and lane parts on the learner thread, so
/// they stay between the previous and this update's progress lines.
fn log_parts(prepared: &Prepared) {
    for (part, samples) in prepared.parts.iter().zip(&prepared.part_samples) {
        for episode in &part.episodes {
            episode.log();
        }
        crate::telemetry::log_line!(
            "level=INFO event=collection_part update={} lane={} rounds={} samples={} games={} inference_s={:.3} simulation_s={:.3}",
            part.update,
            part.lane,
            part.rounds,
            samples,
            part.episodes.len(),
            part.inference.as_secs_f64(),
            part.simulation.as_secs_f64(),
        );
    }
}

/// The spawn modifiers of `update`; collection past the final update only
/// keeps lanes and checkpoints uniform and never draws a generation.
fn run_spec(
    settings: &AnnealedJobConfig,
    generations: &mut GenerationCache,
    update: u64,
) -> Result<ModifierSpec, PpoError> {
    if update < settings.updates {
        generations.spec_for_update(update)
    } else {
        Ok(ModifierSpec::NOMINAL)
    }
}

/// League snapshots in-flight games play.
fn league_references(games: &[SlotSnapshot]) -> Vec<u64> {
    games
        .iter()
        .filter_map(|game| match game.plan.opponent {
            OpponentKind::League(update) => Some(update),
            _ => None,
        })
        .collect()
}

/// The checkpoint's league: its written snapshots plus, when one was due, its own model.
fn restore_league(
    settings: &AnnealedJobConfig,
    state: &TrainingSession,
    directory: &Path,
    collection: &CollectorState,
) -> Result<LeagueStore, PpoError> {
    let mut league = LeagueStore::restore(directory, &collection.league)?;
    let completed = state.completed_updates;
    let league_played = settings
        .opponents
        .iter()
        .any(|(opponent, _)| *opponent == AnnealedOpponent::League);
    if league_played && completed.is_multiple_of(settings.league_every) {
        let parameters = state.model.export_parameters().map_err(text_error)?;
        league.insert(completed, Arc::new(parameters));
    }
    Ok(league)
}

/// Decodes and checks the checkpointed collection state against this run.
fn restored_collection(
    settings: &AnnealedJobConfig,
    state: &TrainingSession,
) -> Result<CollectorState, PpoError> {
    let checkpoint = state.collection.as_ref().ok_or(PpoError::InvalidConfig(
        "training checkpoint collection state",
    ))?;
    let collection = CollectorState::decode(&checkpoint.state)?;
    if collection.update != state.completed_updates
        || collection.actor_version + PIPELINE_STALENESS < collection.update
        || collection.slots.len() != settings.slots
        || collection
            .slots
            .iter()
            .enumerate()
            .any(|(slot, snapshot)| snapshot.plan.slot != slot)
    {
        return Err(PpoError::InvalidConfig(
            "training checkpoint collection state",
        ));
    }
    Ok(collection)
}
