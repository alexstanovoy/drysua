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
    validate_groups(config, groups, rounds)?;
    let width = groups[0].environments.len();
    let group_count = groups.len();
    std::thread::scope(|scope| -> Result<(), PpoError> {
        let mut states = bounded_vec(group_count)?;
        for group in groups {
            states.push(GroupState::spawn(scope, group, operation)?);
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

fn validate_groups(
    config: PpoConfig,
    groups: &[ActorGroup],
    rounds: usize,
) -> Result<(), PpoError> {
    if !matches!(groups.len(), 2 | MAX_PIPELINE_GROUPS) || rounds == 0 || rounds > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("actor pipeline groups or rounds"));
    }
    let width = groups[0].environments.len();
    validate_pipeline_memory(config, width, groups.len())?;
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
        if group
            .environments
            .iter()
            .any(|world| matches!(world.opponent, OpponentRuntime::Policy { .. }))
        {
            return Err(PpoError::InvalidConfig(
                "actor pipeline requires CPU opponents",
            ));
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
        operation: impl Fn(usize, &mut TrainingEnvironment, StreamJob) -> Result<StreamReply, PpoError>
        + Send
        + Sync
        + Copy
        + 'scope,
    ) -> Result<Self, PpoError> {
        let width = group.environments.len();
        assert_eq!(width, group.streams.len());
        assert_eq!(width, group.random.len());
        let stream_base = group.stream_base;
        let name = format!("actor-pipe-{stream_base}");
        let workers = StreamWorkers::spawn(
            scope,
            &mut group.environments,
            &name,
            move |stream, world, job| operation(stream_base + stream, world, job),
        )?;
        let mut prepared = bounded_vec(width)?;
        prepared.resize_with(width, || None);
        let mut active = bounded_vec(width)?;
        active.extend(0..width);
        for &stream in &active {
            workers.submit(stream, StreamJob::Prepare)?;
        }
        Ok(Self {
            workers,
            streams: &mut group.streams,
            staged_random: group.random.clone(),
            random: &mut group.random,
            prepared,
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
                StreamReply::Prepared(sample) if self.bootstrap => {
                    self.prepared[stream] = Some(*sample)
                }
                StreamReply::Advanced(reply) if !self.bootstrap => {
                    let reply = *reply;
                    assert!(reply.value.is_none(), "single owner has no flush evaluator");
                    self.streams[stream] = reply.state;
                    self.prepared[stream] = reply.prepared.map(|sample| *sample);
                    assert_eq!(self.prepared[stream].is_none(), self.streams[stream].done);
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
        self.pending.finish_sampled(
            self.streams,
            self.stream_base,
            (&self.active, &choices),
            rollout,
            &mut self.report,
        )?;
        self.random.clone_from_slice(&self.staged_random);
        for ((&stream, choice), space) in self.active.iter().zip(choices).zip(spaces) {
            self.workers.submit(
                stream,
                StreamJob::Advance {
                    state: std::mem::take(&mut self.streams[stream]),
                    choice,
                    space,
                },
            )?;
        }
        self.in_flight = true;
        Ok(true)
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
