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

use rustc_hash::FxHashMap;

use super::super::collector::Collector;
use super::super::collector_state::CollectorState;
use super::super::lane::{LaneSettings, LaneStart, PartConfig, PartDone};
use super::super::pool::SimPool;
use super::super::slot::{GameSchedule, SlotSnapshot};
use super::super::{CollectionReport, TrainingSession};
use super::*;
use crate::telemetry::{TrainingStage, TrainingUpdateMode, TrainingUpdateTimer};
use crate::{CollectionCheckpoint, PPO_MAX_STREAMS, PpoRollout};

pub(super) struct AnnealedSession {
    state: TrainingSession,
    random_directory: PathBuf,
    generations: u64,
    games: u64,
    /// Collection state the next update starts from, when resuming.
    restored: Option<CollectorState>,
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
        Ok(Self {
            state,
            random_directory: random_directory.to_path_buf(),
            generations,
            games: 0,
            restored,
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
        let target = match settings.invocation_updates {
            Some(limit) => self
                .state
                .completed_updates
                .checked_add(limit.get())
                .ok_or(PpoError::CounterOverflow)?
                .min(settings.updates),
            None => settings.updates,
        };
        // A crash between a committed milestone and its export is repaired here.
        self.export_history(settings)?;
        if self.state.completed_updates >= target {
            return Ok(());
        }
        let schedule = GameSchedule {
            seed: settings.seed,
            mixture: pool.mixture.clone(),
            decision_cap: harness.episode_decisions(),
            config,
        };
        let mut generations = GenerationCache::new(
            self.random_directory.clone(),
            settings.seed,
            settings.generation_updates,
            1,
            anneal_schedule(settings),
            self.generations,
        );
        generations.adaptive = self.state.adaptive_environment;
        let lanes = self.lane_plan(settings, pool)?;
        let result = std::thread::scope(|scope| {
            let simulation = SimPool::spawn(
                scope,
                settings.simulation_threads,
                settings.slots,
                &schedule,
            )?;
            let collector = Collector::spawn(scope, lanes, simulation, &schedule)?;
            let mut pipeline = Pipeline {
                collector,
                configs: BTreeMap::new(),
            };
            self.publish_initial(settings, &mut pipeline, &mut generations)?;
            self.update_loop(
                settings,
                harness,
                directory,
                target,
                (&mut pipeline, &mut generations),
                checkpointed,
            )
        });
        self.generations = generations.counted_through();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn update_loop(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        directory: &Path,
        target: u64,
        (pipeline, generations): (&mut Pipeline, &mut GenerationCache),
        checkpointed: &mut impl FnMut(TrainingCheckpointReport),
    ) -> Result<(), PpoError> {
        let started = std::time::Instant::now();
        let mut schedule =
            super::super::TrainingCheckpointSchedule::new(settings.checkpoint_cadence)?;
        while self.state.completed_updates < target {
            self.train_update(settings, harness, pipeline, generations)?;
            let stop = crate::training_signals::stop_requested();
            let final_update = self.state.completed_updates == target || stop;
            if schedule.is_due(self.state.completed_updates, started.elapsed()) || final_update {
                let report = self.state.checkpoint_report(None);
                let durable = crate::telemetry::time_training_checkpoint(
                    self.state.completed_updates,
                    || self.state.save(directory, report),
                )?;
                checkpointed(durable);
                schedule.mark_committed(started.elapsed())?;
            }
            self.export_history(settings)?;
            if stop
                || harness
                    .stop_after
                    .is_some_and(|stop| self.state.completed_updates >= stop)
            {
                break;
            }
        }
        Ok(())
    }

    fn export_history(&self, settings: &AnnealedJobConfig) -> Result<(), PpoError> {
        match &settings.history {
            Some(history) => history.export(
                &self.state.model,
                self.state.completed_updates,
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
            let settings = LaneSettings {
                index: lane,
                slots,
                target: settings.ppo.samples_per_update / settings.lanes,
                device: self.state.model.device(),
                snapshots: Arc::clone(&snapshots),
            };
            lanes.push((settings, start));
        }
        Ok(lanes)
    }

    /// Publishes the configurations of every update collected before the
    /// learner's next result: the current update and the pipelined ones.
    fn publish_initial(
        &mut self,
        settings: &AnnealedJobConfig,
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        let completed = self.state.completed_updates;
        let current = Arc::new(self.state.model.export_parameters().map_err(text_error)?);
        let first = match &self.state.collection {
            Some(checkpoint) if completed > 0 => {
                let state = CollectorState::decode(&checkpoint.state)?;
                PartConfig {
                    update: completed,
                    version: state.actor_version,
                    actor: Arc::new(checkpoint.actor.clone()),
                    spec: state.spec,
                }
            }
            _ => PartConfig {
                update: completed,
                version: completed,
                actor: Arc::clone(&current),
                spec: run_spec(settings, generations, completed)?,
            },
        };
        pipeline.publish(first)?;
        for offset in 1..=PIPELINE_STALENESS {
            let update = completed + offset;
            pipeline.publish(PartConfig {
                update,
                version: completed,
                actor: Arc::clone(&current),
                spec: run_spec(settings, generations, update)?,
            })?;
        }
        Ok(())
    }

    fn train_update(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
    ) -> Result<(), PpoError> {
        let mut timing = TrainingUpdateTimer::new(
            self.state.completed_updates,
            TrainingUpdateMode::Annealed,
            self.state.trainer.optimizer_step(),
        );
        let result = self.train_update_timed(settings, harness, pipeline, generations, &mut timing);
        timing.observe_result(result)?;
        timing.complete();
        Ok(())
    }

    fn train_update_timed(
        &mut self,
        settings: &AnnealedJobConfig,
        harness: AnnealedHarness,
        pipeline: &mut Pipeline,
        generations: &mut GenerationCache,
        timing: &mut TrainingUpdateTimer,
    ) -> Result<(), PpoError> {
        let update = self.state.completed_updates;
        let config = settings.ppo;
        timing.enter(TrainingStage::Collection);
        let parts = pipeline.collector.take(update)?;
        timing.enter(TrainingStage::BatchPreparation);
        let (rollout, report, snapshots) = assemble(&parts, settings.slots, config)?;
        let samples = rollout.len();
        timing.set_samples(samples);
        let next_adaptive = generations.next_adaptive(settings, harness, update, &report)?;
        let batch = rollout.finish(config)?;
        let explained_variance = batch.explained_variance();
        let optimizer_step = self.state.trainer.optimizer_step();
        timing.enter(TrainingStage::Optimization);
        let optimized = self.state.trainer.train_update(&self.state.model, &batch);
        timing.set_optimizer_step(self.state.trainer.optimizer_step());
        self.state.latest = optimized?;
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
        pipeline.publish(PartConfig {
            update: next,
            version: completed,
            actor,
            spec: run_spec(settings, generations, next)?,
        })?;
        self.state.collection = Some(pipeline.checkpoint(completed, snapshots)?);
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
            slots: snapshots,
        };
        Ok(CollectionCheckpoint {
            actor: config.actor.as_ref().clone(),
            state: state.encode(),
        })
    }
}

/// One update's batch, report and next-update snapshots from its lane parts.
fn assemble(
    parts: &[PartDone],
    slots: usize,
    config: PpoConfig,
) -> Result<(PpoRollout, CollectionReport, Vec<SlotSnapshot>), PpoError> {
    let mut rollout = PpoRollout::new(config.rollout_capacity(slots))?;
    let mut streams: FxHashMap<(usize, u64), usize> = FxHashMap::default();
    let mut report = CollectionReport::default();
    let mut snapshots: Vec<Option<SlotSnapshot>> = (0..slots).map(|_| None).collect();
    for part in parts {
        for sample in &part.samples {
            let next = streams.len();
            let stream = *streams.entry((sample.slot, sample.game)).or_insert(next);
            if stream >= PPO_MAX_STREAMS {
                return Err(PpoError::InvalidConfig(
                    "update exceeds its game stream bound",
                ));
            }
            let mut transition = sample.transition.clone();
            transition.stream = stream;
            rollout.push(transition)?;
        }
        for episode in &part.episodes {
            episode.log();
            episode.accumulate(&mut report)?;
        }
        for snapshot in &part.snapshot {
            snapshots[snapshot.plan.slot] = Some(snapshot.clone());
        }
        eprintln!(
            "level=INFO event=collection_part update={} lane={} rounds={} samples={} games={} inference_s={:.3} simulation_s={:.3}",
            part.update,
            part.lane,
            part.rounds,
            part.samples.len(),
            part.episodes.len(),
            part.inference.as_secs_f64(),
            part.simulation.as_secs_f64(),
        );
    }
    let snapshots = snapshots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(PpoError::InvalidTransition("collection snapshot slots"))?;
    Ok((rollout, report, snapshots))
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
