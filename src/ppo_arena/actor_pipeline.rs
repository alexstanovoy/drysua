//! Fixed independent batches on one inference owner, not smaller actor GEMMs.

use super::super::parallel::StreamWorkers;
use super::*;

#[cfg(test)]
#[path = "../tests/actor_pipeline.rs"]
mod tests;

const MAX_PIPELINE_GROUPS: usize = 4;
const _: () = assert!(MAX_ACTOR_ENVIRONMENTS <= 64);
const MEMORY_LIMIT: u64 = 12 * 1024 * 1024 * 1024;
const NON_ROLLOUT_RESERVE: u64 = 2 * 1024 * 1024 * 1024;
// Charge the new owned inputs, output choices and temporary dense frame copies
// separately; ActionSpace backing allocations still share the guarded non-rollout reserve.
const NEURAL_HOST_ROW_BYTES: usize = std::mem::size_of::<neural_opponent::PreparedOpponent>()
    + std::mem::size_of::<neural_opponent::OpponentChoice>()
    + 2 * std::mem::size_of::<FeatureFrame>();
const _: () = assert!(NEURAL_HOST_ROW_BYTES * 64 < 64 * 1024 * 1024);

pub(in crate::ppo_arena) fn validate_neural_opponent_memory(
    config: PpoConfig,
    live_worlds: usize,
) -> Result<(), PpoError> {
    config.validate()?;
    if !(1..=MAX_ACTOR_ENVIRONMENTS).contains(&live_worlds) || live_worlds > config.environments {
        return Err(PpoError::InvalidConfig(
            "neural opponent active worlds must be within 1..=64",
        ));
    }
    let base = match config.sample_budget {
        crate::PpoSampleBudget::WideAnnealed => crate::PPO_WIDE_ANNEALED_PAYLOAD_BOUND_BYTES,
        _ => crate::PPO_ANNEALED_STORAGE_PEAK_BYTES + NON_ROLLOUT_RESERVE,
    };
    if base + live_worlds as u64 * NEURAL_HOST_ROW_BYTES as u64 > MEMORY_LIMIT {
        return Err(PpoError::InvalidConfig(
            "neural opponent batching exceeds 12 GiB admission budget",
        ));
    }
    Ok(())
}

pub(in crate::ppo_arena) struct ActorGroup {
    pub(in crate::ppo_arena) stream_base: usize,
    pub(in crate::ppo_arena) environments: Vec<TrainingEnvironment>,
    pub(in crate::ppo_arena) streams: Vec<EpisodeStream>,
    pub(in crate::ppo_arena) random: Vec<PpoRng>,
}

pub(in crate::ppo_arena) fn validate_pipeline_memory(
    config: PpoConfig,
    width: usize,
    groups: usize,
) -> Result<(), PpoError> {
    config.validate()?;
    if !matches!(groups, 2 | MAX_PIPELINE_GROUPS) {
        return Err(PpoError::InvalidConfig(
            "actor pipeline group count must be 2 or 4",
        ));
    }
    if width == 0 || width > MAX_ACTOR_ENVIRONMENTS / groups {
        return Err(PpoError::InvalidConfig(
            "actor pipeline active worlds exceed 64",
        ));
    }
    if config.environments < width * groups
        || config.environments > crate::PPO_WIDE_ANNEALED_MAX_GAMES
        || !config.environments.is_multiple_of(width * groups)
        || config.rollout_decisions != RETAINED_PER_EPISODE
    {
        return Err(PpoError::InvalidConfig("actor pipeline rollout dimensions"));
    }
    if pipeline_payload_bytes(config, width) > MEMORY_LIMIT {
        return Err(PpoError::InvalidConfig(
            "actor pipeline memory admission exceeds 12 GiB",
        ));
    }
    Ok(())
}

fn pipeline_payload_bytes(config: PpoConfig, width: usize) -> u64 {
    assert!(config.environments <= crate::PPO_WIDE_ANNEALED_MAX_GAMES);
    assert!(width <= MAX_ACTOR_ENVIRONMENTS);
    let peak = crate::feature::wide_feature_arena_peak_bytes();
    assert_eq!(peak % crate::PPO_WIDE_ANNEALED_MAX_GAMES as u64, 0);
    let samples = config.environments as u64 * RETAINED_PER_EPISODE as u64;
    // Standard arenas can grow past their used-row bound; capped annealed arenas cannot.
    let growth = if config.sample_budget == crate::PpoSampleBudget::Standard {
        2
    } else {
        1
    };
    let main_arena = peak / crate::PPO_WIDE_ANNEALED_MAX_GAMES as u64 * config.environments as u64;
    // All groups share one arena. The existing compact/prepared/shuffle allowance
    // covers compact records plus the temporary usize permutation; scratch is gone
    // before preparation. The non-rollout reserve remains an assumption, not an RSS proof.
    main_arena * growth + samples * 8_192 + NON_ROLLOUT_RESERVE
}

#[allow(clippy::too_many_arguments)]
/// Errors join the worker scopes; the caller must discard the uncommitted update.
pub(in crate::ppo_arena) fn collect_actor_pipeline(
    model: &PolicyModel,
    config: PpoConfig,
    groups: &mut [ActorGroup],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse_actor_values: bool,
) -> Result<(), PpoError> {
    #[cfg(all(
        test,
        feature = "cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    let graph = graph_pipeline_mode(config, groups, rounds, reuse_actor_values)?;
    let mut collect = || {
        collect_with_operation(
            model,
            config,
            groups,
            rounds,
            rollout,
            report,
            reuse_actor_values,
            move |_, world, job| run_stream_job(world, job, config, None, None),
        )?;
        #[cfg(all(
            test,
            feature = "cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if graph.is_some() {
            record_graph_actor_trace(
                groups
                    .iter()
                    .flat_map(|group| group.streams.iter().zip(group.random.iter())),
            )?;
        }
        Ok(())
    };
    #[cfg(all(
        test,
        feature = "cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    if let Some(graph) = graph {
        return crate::model::cuda_graph_probe::with_actor_graph_for_test(
            model, graph, 20, collect,
        )
        .map_err(text_error)?;
    }
    collect()
}

#[allow(clippy::too_many_arguments)]
pub(in crate::ppo_arena) fn collect_actor_pipeline_with_opponent_batching(
    model: &PolicyModel,
    config: PpoConfig,
    groups: &mut [ActorGroup],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse_actor_values: bool,
    neural_opponent_batching: bool,
) -> Result<(), PpoError> {
    if !neural_opponent_batching {
        return collect_actor_pipeline(
            model,
            config,
            groups,
            rounds,
            rollout,
            report,
            reuse_actor_values,
        );
    }
    #[cfg(all(
        test,
        feature = "cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    let _ = graph_pipeline_mode(config, groups, rounds, reuse_actor_values)?;
    collect_with_operation_mode(
        model,
        config,
        groups,
        rounds,
        rollout,
        report,
        reuse_actor_values,
        true,
        move |_, world, job| run_stream_job(world, job, config, None, None),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_single_neural(
    model: &PolicyModel,
    config: PpoConfig,
    stream_base: usize,
    environments: &mut [TrainingEnvironment],
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse: bool,
) -> Result<(), PpoError> {
    validate_neural_opponent_memory(config, environments.len())?;
    if rounds == 0
        || rounds > ACTOR_DECISIONS
        || environments.len() != streams.len()
        || environments.len() != random.len()
        || stream_base
            .checked_add(environments.len())
            .is_none_or(|end| end > config.environments)
    {
        return Err(PpoError::InvalidConfig(
            "neural opponent collection dimensions",
        ));
    }
    #[cfg(all(
        test,
        feature = "cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    let _ = actor_graph_collection_mode(config, stream_base, environments, reuse)?;
    let opponent = neural_opponent::OpponentBatch::new(model, environments)?;
    std::thread::scope(|scope| {
        let mut state = GroupState::spawn_parts(
            scope,
            stream_base,
            environments,
            streams,
            random,
            Some(opponent),
            move |_, world, job| run_stream_job(world, job, config, None, None),
        )?;
        let collected = (|| {
            for round in 0..=rounds {
                if !state.turn(model, round < rounds, reuse, rollout)? {
                    break;
                }
            }
            state.merge_report(report)
        })();
        state.workers.finish()?;
        collected
    })
}

#[cfg(all(
    test,
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn graph_pipeline_mode(
    config: PpoConfig,
    groups: &[ActorGroup],
    rounds: usize,
    reuse: bool,
) -> Result<Option<bool>, PpoError> {
    let mode = crate::model::cuda_graph_probe::parse_graph_mode(
        std::env::var_os("DRYSUA_PROBE_ACTOR_GRAPH").as_deref(),
    )
    .map_err(text_error)?;
    if mode.is_none() {
        return Ok(None);
    }
    validate_groups(config, groups, rounds)?;
    if groups.len() != 2
        || config.environments != 40
        || config.sample_budget != crate::PpoSampleBudget::Annealed
        || groups[0].environments.len() != 20
        || groups[0].stream_base != 0
        || !reuse
        || groups
            .iter()
            .flat_map(|group| &group.environments)
            .any(|world| !matches!(world.opponent, OpponentRuntime::Teacher))
    {
        return Err(PpoError::InvalidConfig(
            "actor graph probe requires one M40 B20 G2 Teacher wave with actor value reuse",
        ));
    }
    Ok(mode)
}

#[allow(clippy::too_many_arguments)]
fn collect_with_operation(
    model: &PolicyModel,
    config: PpoConfig,
    groups: &mut [ActorGroup],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse_actor_values: bool,
    operation: impl Fn(usize, &mut TrainingEnvironment, StreamJob) -> Result<StreamReply, PpoError>
    + Send
    + Sync
    + Copy,
) -> Result<(), PpoError> {
    collect_with_operation_mode(
        model,
        config,
        groups,
        rounds,
        rollout,
        report,
        reuse_actor_values,
        false,
        operation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_with_operation_mode(
    model: &PolicyModel,
    config: PpoConfig,
    groups: &mut [ActorGroup],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse_actor_values: bool,
    neural_opponent_batching: bool,
    operation: impl Fn(usize, &mut TrainingEnvironment, StreamJob) -> Result<StreamReply, PpoError>
    + Send
    + Sync
    + Copy,
) -> Result<(), PpoError> {
    validate_groups_for_inference(config, groups, rounds, neural_opponent_batching)?;
    let width = groups[0].environments.len();
    let group_count = groups.len();
    let opponents = groups
        .iter()
        .map(|group| {
            if neural_opponent_batching {
                neural_opponent::OpponentBatch::new(model, &group.environments).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, PpoError>>()?;
    std::thread::scope(|scope| -> Result<(), PpoError> {
        let mut states = bounded_vec(group_count)?;
        for (group, opponent) in groups.iter_mut().zip(opponents) {
            states.push(GroupState::spawn(scope, group, opponent, operation)?);
        }
        let collected = visit_groups(rounds, group_count, |index, sample| {
            states[index].turn(model, sample, reuse_actor_values, rollout)
        });
        // On failure, close the bounded queues and join every in-flight CPU job.
        // Each stream has at most one reply, so joining never needs another inference.
        for state in states {
            if collected.is_ok() {
                if rounds == ACTOR_DECISIONS {
                    assert!(state.streams.iter().all(|stream| stream.done));
                    assert!(state.streams.iter().all(|stream| stream.choice.is_none()));
                }
                state.merge_report(report)?;
            }
            state.workers.finish()?;
        }
        collected
    })?;
    // Reorder metadata only, including any earlier waves. Global stream/B keys
    // preserve sequential group order before finish performs order-sensitive normalization.
    rollout.canonicalize_actor_order(config, width)
}

#[cfg(test)]
fn validate_groups(
    config: PpoConfig,
    groups: &[ActorGroup],
    rounds: usize,
) -> Result<(), PpoError> {
    validate_groups_for_inference(config, groups, rounds, false)
}

fn validate_groups_for_inference(
    config: PpoConfig,
    groups: &[ActorGroup],
    rounds: usize,
    neural: bool,
) -> Result<(), PpoError> {
    if !matches!(groups.len(), 2 | MAX_PIPELINE_GROUPS) || rounds == 0 || rounds > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("actor pipeline groups or rounds"));
    }
    let width = groups[0].environments.len();
    validate_pipeline_memory(config, width, groups.len())?;
    if neural {
        validate_neural_opponent_memory(config, width * groups.len())?;
    }
    let mut opponent_model = None;
    for (index, group) in groups.iter().enumerate() {
        if group.environments.len() != width
            || group.streams.len() != width
            || group.random.len() != width
            || !group.stream_base.is_multiple_of(width)
            || Some(group.stream_base) != groups[0].stream_base.checked_add(index * width)
            || group
                .stream_base
                .checked_add(width)
                .is_none_or(|end| end > config.environments)
        {
            return Err(PpoError::InvalidConfig("actor pipeline group streams"));
        }
        for world in &group.environments {
            match (&world.opponent, neural) {
                (OpponentRuntime::Policy { model, .. }, true) => {
                    if opponent_model.is_some_and(|expected| !Arc::ptr_eq(expected, model)) {
                        return Err(PpoError::InvalidConfig(
                            "actor pipeline opponent model mismatch",
                        ));
                    }
                    opponent_model = Some(model);
                }
                (OpponentRuntime::Policy { .. }, false) => {
                    return Err(PpoError::InvalidConfig(
                        "actor pipeline requires CPU opponents",
                    ));
                }
                (_, true) => {
                    return Err(PpoError::InvalidConfig(
                        "batched opponent requires neural worlds",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn visit_groups(
    rounds: usize,
    groups: usize,
    mut turn: impl FnMut(usize, bool) -> Result<bool, PpoError>,
) -> Result<(), PpoError> {
    assert!(rounds > 0);
    assert!(rounds <= ACTOR_DECISIONS);
    assert!(matches!(groups, 2 | MAX_PIPELINE_GROUPS));
    for round in 0..=rounds {
        let mut active = false;
        for group in 0..groups {
            active |= turn(group, round < rounds)?;
        }
        if !active {
            break;
        }
    }
    Ok(())
}

fn bounded_vec<T>(count: usize) -> Result<Vec<T>, PpoError> {
    assert!(count > 0);
    assert!(count <= MAX_ACTOR_ENVIRONMENTS);
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| PpoError::Model(format!("actor pipeline allocation: {error}")))?;
    Ok(values)
}

struct GroupState<'scope> {
    workers: StreamWorkers<'scope, TrainingEnvironment, StreamJob, StreamReply>,
    streams: &'scope mut [EpisodeStream],
    random: &'scope mut [PpoRng],
    staged_random: Vec<PpoRng>,
    prepared: Vec<Option<(FeatureFrame, ActionSpace)>>,
    opponent_prepared: Vec<Option<Box<neural_opponent::PreparedOpponent>>>,
    opponent_batch: Option<neural_opponent::OpponentBatch>,
    active: Vec<usize>,
    pending: actor_values::PendingRound,
    terminals: Vec<(usize, CompletedAdvance)>,
    report: PpoSmokeReport,
    stream_base: usize,
    bootstrap: bool,
    in_flight: bool,
}

impl<'scope> GroupState<'scope> {
    fn spawn(
        scope: &'scope std::thread::Scope<'scope, '_>,
        group: &'scope mut ActorGroup,
        opponent: Option<neural_opponent::OpponentBatch>,
        operation: impl Fn(usize, &mut TrainingEnvironment, StreamJob) -> Result<StreamReply, PpoError>
        + Send
        + Sync
        + Copy
        + 'scope,
    ) -> Result<Self, PpoError> {
        Self::spawn_parts(
            scope,
            group.stream_base,
            &mut group.environments,
            &mut group.streams,
            &mut group.random,
            opponent,
            operation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_parts(
        scope: &'scope std::thread::Scope<'scope, '_>,
        stream_base: usize,
        environments: &'scope mut [TrainingEnvironment],
        streams: &'scope mut [EpisodeStream],
        random: &'scope mut [PpoRng],
        opponent: Option<neural_opponent::OpponentBatch>,
        operation: impl Fn(usize, &mut TrainingEnvironment, StreamJob) -> Result<StreamReply, PpoError>
        + Send
        + Sync
        + Copy
        + 'scope,
    ) -> Result<Self, PpoError> {
        let width = environments.len();
        assert_eq!(width, streams.len());
        assert_eq!(width, random.len());
        let name = format!("actor-pipe-{stream_base}");
        let workers =
            StreamWorkers::spawn(scope, environments, &name, move |stream, world, job| {
                operation(stream_base + stream, world, job)
            })?;
        let mut prepared = bounded_vec(width)?;
        prepared.resize_with(width, || None);
        let mut active = bounded_vec(width)?;
        let mut opponent_prepared = bounded_vec(width)?;
        opponent_prepared.resize_with(width, || None);
        active.extend(0..width);
        for &stream in &active {
            workers.submit(
                stream,
                if opponent.is_some() {
                    StreamJob::PrepareNeural
                } else {
                    StreamJob::Prepare
                },
            )?;
        }
        Ok(Self {
            workers,
            streams,
            staged_random: random.to_vec(),
            random,
            prepared,
            opponent_prepared,
            opponent_batch: opponent,
            active,
            pending: actor_values::PendingRound::new(width),
            terminals: bounded_vec(width)?,
            report: PpoSmokeReport::default(),
            stream_base,
            bootstrap: true,
            in_flight: true,
        })
    }

    fn receive(&mut self) -> Result<(), PpoError> {
        assert!(self.in_flight);
        assert!(!self.active.is_empty());
        self.in_flight = false;
        for (stream, reply) in self
            .active
            .iter()
            .copied()
            .zip(self.workers.receive(&self.active)?)
        {
            match reply {
                StreamReply::Prepared(sample)
                    if self.bootstrap && self.opponent_batch.is_none() =>
                {
                    self.prepared[stream] = Some(*sample)
                }
                StreamReply::PreparedNeural(sample, opponent)
                    if self.bootstrap && self.opponent_batch.is_some() =>
                {
                    self.prepared[stream] = Some(*sample);
                    self.opponent_prepared[stream] = Some(opponent);
                }
                StreamReply::Advanced(reply) if !self.bootstrap => {
                    let reply = *reply;
                    assert!(reply.value.is_none(), "single owner has no flush evaluator");
                    self.streams[stream] = reply.state;
                    self.prepared[stream] = reply.prepared.map(|sample| *sample);
                    self.opponent_prepared[stream] = reply.opponent_prepared;
                    assert_eq!(self.prepared[stream].is_none(), self.streams[stream].done);
                    assert_eq!(
                        self.opponent_prepared[stream].is_some(),
                        self.opponent_batch.is_some() && !self.streams[stream].done
                    );
                    if self.streams[stream].done {
                        assert!(self.terminals.len() < self.streams.len());
                        self.terminals.push((stream, reply.completed));
                    }
                    self.pending.push(stream, reply.completed, reply.opponent);
                }
                _ => return Err(PpoError::InvalidConfig("actor pipeline worker reply")),
            }
        }
        self.bootstrap = false;
        Ok(())
    }

    fn turn(
        &mut self,
        model: &PolicyModel,
        sample: bool,
        reuse: bool,
        rollout: &mut PpoRollout,
    ) -> Result<bool, PpoError> {
        if !self.in_flight {
            return Ok(false);
        }
        self.receive()?;
        self.active.clear();
        self.active
            .extend((0..self.streams.len()).filter(|&stream| !self.streams[stream].done));
        if !reuse || !sample || self.active.is_empty() {
            self.pending.finish_fallback(
                model,
                self.streams,
                self.stream_base,
                &self.prepared,
                rollout,
                &mut self.report,
            )?;
        }
        if !sample || self.active.is_empty() {
            return Ok(false);
        }
        let mut frames = bounded_vec(self.active.len())?;
        let mut spaces = bounded_vec(self.active.len())?;
        for &stream in &self.active {
            let (frame, space) = self.prepared[stream]
                .take()
                .ok_or(PpoError::InvalidTransition("actor pipeline prepared frame"))?;
            frames.push(frame);
            spaces.push(space);
        }
        self.staged_random.clone_from_slice(self.random);
        let (choices, spaces) =
            sample_choices(model, &mut self.staged_random, &self.active, frames, spaces)?;
        let mut opponents = self.sample_opponents()?.into_iter().flatten();
        self.pending.finish_sampled(
            self.streams,
            self.stream_base,
            (&self.active, &choices),
            rollout,
            &mut self.report,
        )?;
        self.random.clone_from_slice(&self.staged_random);
        for ((&stream, choice), space) in self.active.iter().zip(choices).zip(spaces) {
            let opponent = opponents.next().map(Box::new);
            assert_eq!(opponent.is_some(), self.opponent_batch.is_some());
            self.workers.submit(
                stream,
                StreamJob::Advance {
                    state: std::mem::take(&mut self.streams[stream]),
                    choice,
                    space,
                    opponent,
                },
            )?;
        }
        assert!(opponents.next().is_none());
        self.in_flight = true;
        Ok(true)
    }

    fn sample_opponents(
        &mut self,
    ) -> Result<Option<Vec<neural_opponent::OpponentChoice>>, PpoError> {
        let Some(batch) = &self.opponent_batch else {
            return Ok(None);
        };
        let mut inputs = bounded_vec(self.active.len())?;
        for &stream in &self.active {
            inputs.push(*self.opponent_prepared[stream].take().ok_or(
                PpoError::InvalidTransition("missing prepared neural opponent"),
            )?);
        }
        let choices = batch.sample(inputs)?;
        assert_eq!(choices.len(), self.active.len());
        Ok(Some(choices))
    }

    fn merge_report(&self, target: &mut PpoSmokeReport) -> Result<(), PpoError> {
        assert!(!self.in_flight);
        assert!(self.terminals.len() <= self.streams.len());
        target.elapsed_ticks = target
            .elapsed_ticks
            .checked_add(self.report.elapsed_ticks)
            .ok_or(PpoError::CounterOverflow)?;
        // Replay individual terminal merges, not grouped floating-point totals.
        for &(stream, completed) in &self.terminals {
            accumulate_episode(
                self.stream_base + stream,
                completed.end_tick,
                &self.streams[stream],
                completed.outcome,
                target,
            )?;
        }
        Ok(())
    }
}
