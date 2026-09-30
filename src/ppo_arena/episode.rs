#![allow(
    clippy::float_arithmetic,
    reason = "Discounted n-step rewards use floating-point arithmetic"
)]

use super::*;

#[cfg(test)]
#[path = "../tests/neural_opponent_collection.rs"]
mod neural_opponent_tests;

#[path = "actor_pipeline.rs"]
mod actor_pipeline;
#[path = "actor_values.rs"]
mod actor_values;

pub(super) use actor_pipeline::{
    ActorGroup, collect_actor_pipeline_with_opponent_batching, validate_neural_opponent_memory,
    validate_pipeline_memory,
};

#[cfg(test)]
#[path = "../tests/actor_value_reuse.rs"]
mod actor_value_reuse_tests;

#[cfg(test)]
#[path = "../tests/map2_collection.rs"]
mod map2_tests;

#[cfg(test)]
#[path = "../tests/episode_collection_parallel.rs"]
mod parallel_tests;

#[cfg(test)]
#[path = "../tests/episode_test_support.rs"]
mod test_support;

#[cfg(test)]
pub(crate) use test_support::*;

pub(super) const TICK_CAP: u32 = crate::MAP2_TICK_CAP;
pub(super) const ACTOR_DECISIONS: usize = crate::MAP2_ACTOR_DECISIONS;
const RETENTION_STRIDE: usize = crate::MAP2_RETENTION_STRIDE;
const RETENTION_DOMAIN: u64 = 0x7265_7465_6e74_696f;
const RETENTION_STREAM_DOMAIN: u64 = 0x7068_6173_655f_726e;
const _: () = assert!(RETENTION_STRIDE.is_power_of_two());
const RETAINED_PER_EPISODE: usize = crate::MAP2_RETAINED_DECISIONS;
const MAX_ACTOR_ENVIRONMENTS: usize = crate::PPO_MAX_PARALLEL_WORLDS;
const _: () = assert!(MAX_ACTOR_ENVIRONMENTS <= crate::MODEL_TRAINING_BATCH);
const _: () = assert!(crate::PPO_MAX_GAMES * RETAINED_PER_EPISODE <= crate::PPO_MAX_SAMPLES);

fn opponent_name(opponent: &OpponentRuntime) -> &'static str {
    match opponent {
        OpponentRuntime::Teacher => "Teacher",
        #[cfg(test)]
        OpponentRuntime::Idle => "Idle",
        OpponentRuntime::Policy { .. } => "Policy",
    }
}

/// One request in a worker's bounded queue.
#[allow(
    clippy::large_enum_variant,
    reason = "Boxing the advance state would add one heap allocation per stream and decision"
)]
enum StreamJob {
    /// Build the policy frame and action space for the next decision.
    Prepare,
    /// Prepare both neural seats; inference remains on the collection owner.
    PrepareNeural,
    /// Apply one chosen action and advance the owned world three ticks.
    Advance {
        state: EpisodeStream,
        choice: PpoPolicyChoice,
        space: ActionSpace,
        opponent: Option<Box<neural_opponent::OpponentChoice>>,
    },
}

/// One ordered reply from a stream worker.
enum StreamReply {
    Prepared(Box<(FeatureFrame, ActionSpace)>),
    PreparedNeural(
        Box<(FeatureFrame, ActionSpace)>,
        Box<neural_opponent::PreparedOpponent>,
    ),
    Advanced(Box<AdvancedReply>),
}

struct AdvancedReply {
    state: EpisodeStream,
    completed: CompletedAdvance,
    /// Reply slot receiving the flush-next value from the dedicated evaluator.
    /// The value is the exact batch-1 evaluation of `frame`, only moved off the
    /// worker's critical path.
    value: Option<FlushValue>,
    opponent: &'static str,
    /// Frame and space for the next decision, built by the worker after the
    /// world step so the next sampling round never waits for a separate
    /// prepare barrier.
    prepared: Option<Box<(FeatureFrame, ActionSpace)>>,
    opponent_prepared: Option<Box<neural_opponent::PreparedOpponent>>,
}

/// One queued retained-interval flush request: the next frame plus the slot
/// receiving its batch-1 value evaluation.
type FlushRequest = (
    FeatureFrame,
    std::sync::mpsc::SyncSender<Result<f32, PpoError>>,
);

/// Receiver half of a queued flush value.
type FlushValue = std::sync::mpsc::Receiver<Result<f32, PpoError>>;

/// Bounded single-thread evaluator for retained-interval next values.
///
/// Retained intervals close on the owning stream worker, where the exact
/// batch-1 evaluation serializes the whole collector behind the slowest
/// worker's single-frame CUDA round trip. This dedicated thread overlaps that
/// same evaluation with the workers' world steps; the collector blocks on the
/// value only when the transition is pushed.
struct FlushEvaluator {
    sender: std::sync::Mutex<Option<std::sync::mpsc::SyncSender<FlushRequest>>>,
}

impl FlushEvaluator {
    fn submit(
        &self,
        frame: FeatureFrame,
        reply: std::sync::mpsc::SyncSender<Result<f32, PpoError>>,
    ) -> Result<(), PpoError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| PpoError::InvalidConfig("flush evaluator lock"))?
            .clone()
            .ok_or(PpoError::InvalidConfig("flush evaluator closed"))?;
        sender
            .send((frame, reply))
            .map_err(|_| PpoError::InvalidConfig("flush evaluator channel"))
    }

    /// Drops the sender so the evaluator loop finishes after the queued work.
    fn shutdown(&self) {
        if let Ok(mut sender) = self.sender.lock() {
            *sender = None;
        }
    }
}

/// Drains queued flush requests until every sender is dropped.
fn flush_evaluator_loop(model: &PolicyModel, receiver: &std::sync::mpsc::Receiver<FlushRequest>) {
    while let Ok((frame, reply)) = receiver.recv() {
        let value = model
            .evaluate_batch(std::slice::from_ref(&frame))
            .map_err(text_error)
            .and_then(|output| {
                output
                    .into_iter()
                    .next()
                    .map(|output| output.value)
                    .ok_or(PpoError::InvalidTransition("episode flush value"))
            });
        // A dropped receiver means the collector no longer needs this value;
        // the evaluator keeps draining so later requests still complete.
        let _ = reply.send(value);
    }
}

/// Closes the evaluator sender on every scope exit, including early errors.
struct FlushThreadGuard<'a>(&'a FlushEvaluator);

impl Drop for FlushThreadGuard<'_> {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

/// Runs one bounded worker pool and applies completions in stream order.
/// Actor-value reuse replaces the default flush evaluator with one pending round;
/// a failed sample/completion aborts the uncommitted update, never retries its worlds.
///
/// All exits join the pool and close the evaluator through an enclosing scope
/// guard, so no path leaks a thread.
#[allow(clippy::too_many_arguments)]
fn collect_with_workers(
    model: &PolicyModel,
    config: PpoConfig,
    stream_base: usize,
    thread_prefix: &str,
    environments: &mut [TrainingEnvironment],
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
    reuse_actor_values: bool,
) -> Result<(), PpoError> {
    assert_eq!(streams.len(), environments.len());
    assert_eq!(random.len(), environments.len());
    // The annealed batch collector admits any world count from one to the ceiling.
    assert!(!environments.is_empty());
    if environments.len() > crate::PPO_MAX_GAMES
        || stream_base
            .checked_add(environments.len())
            .is_none_or(|end| end > config.environments)
    {
        return Err(PpoError::InvalidConfig(
            "episode collection worlds or streams",
        ));
    }
    let (flush_evaluator, flush_receiver) = if reuse_actor_values {
        (None, None)
    } else {
        // One slot per world: every worker has at most one flush in flight.
        let (sender, receiver) = std::sync::mpsc::sync_channel::<FlushRequest>(environments.len());
        (
            Some(FlushEvaluator {
                sender: std::sync::Mutex::new(Some(sender)),
            }),
            Some(receiver),
        )
    };
    std::thread::scope(|scope| -> Result<(), PpoError> {
        if let Some(receiver) = flush_receiver {
            std::thread::Builder::new()
                .name(format!("{thread_prefix}-flush-eval"))
                .spawn_scoped(scope, move || flush_evaluator_loop(model, &receiver))
                .map_err(|error| PpoError::EpisodeWorker {
                    stream: stream_base,
                    cause: error.to_string(),
                })?;
        }
        let _flush_guard = flush_evaluator.as_ref().map(FlushThreadGuard);
        let flush_evaluator_ref = flush_evaluator.as_ref();
        let workers = super::parallel::StreamWorkers::spawn(
            scope,
            environments,
            thread_prefix,
            move |_, environment, job| {
                run_stream_job(environment, job, config, flush_evaluator_ref)
            },
        )?;
        // Bootstrap: every stream builds its first frame before any action.
        let mut prepared = prepare_all_workers(&workers, streams.len())?;
        // Reused across decisions: the active set never changes shape between
        // rounds, only membership.
        let mut active: Vec<usize> = Vec::with_capacity(streams.len());
        let mut pending =
            reuse_actor_values.then(|| actor_values::PendingRound::new(streams.len()));
        let mut staged_random = reuse_actor_values.then(|| random.to_vec());
        for _ in 0..rounds {
            active.clear();
            active.extend((0..streams.len()).filter(|&stream| !streams[stream].done));
            if active.is_empty() {
                break;
            }
            let mut frames = Vec::with_capacity(active.len());
            let mut spaces = Vec::with_capacity(active.len());
            for &stream in &active {
                let sample = prepared[stream]
                    .take()
                    .ok_or(PpoError::InvalidTransition("episode prepared frame"))?;
                frames.push(sample.0);
                spaces.push(sample.1);
            }
            if let Some(staged) = &mut staged_random {
                staged.clone_from_slice(random);
            }
            let sampled = sample_choices(
                model,
                staged_random.as_deref_mut().unwrap_or(&mut *random),
                &active,
                frames,
                spaces,
            );
            let (choices, spaces) = sampled?;
            if let Some(pending) = &mut pending {
                pending.finish_sampled(
                    streams,
                    stream_base,
                    (&active, &choices),
                    rollout,
                    report,
                )?;
                random.clone_from_slice(staged_random.as_ref().expect("staged actor RNGs"));
            }
            let mut samples = choices.into_iter().zip(spaces);
            for &stream in &active {
                let (choice, space) = samples.next().expect("one sample per active stream");
                let state = std::mem::take(&mut streams[stream]);
                workers.submit(
                    stream,
                    StreamJob::Advance {
                        state,
                        choice,
                        space,
                        opponent: None,
                    },
                )?;
            }
            assert!(samples.next().is_none());
            let replies = workers.receive(&active)?;
            assert_eq!(replies.len(), active.len());
            for (stream, reply) in active.iter().copied().zip(replies) {
                let StreamReply::Advanced(reply) = reply else {
                    return Err(PpoError::InvalidConfig("stream worker reply"));
                };
                let mut reply = *reply;
                streams[stream] = std::mem::take(&mut reply.state);
                prepared[stream] = reply.prepared.map(|sample| *sample);
                assert_eq!(prepared[stream].is_none(), streams[stream].done);
                if let Some(pending) = &mut pending {
                    assert!(
                        reply.value.is_none(),
                        "actor reuse never submits a flush forward"
                    );
                    pending.push(stream, reply.completed, reply.opponent);
                    continue;
                }
                let value =
                    match reply.value {
                        Some(receiver) => Some(receiver.recv().map_err(|_| {
                            PpoError::InvalidConfig("episode flush evaluator reply")
                        })??),
                        None => None,
                    };
                finish_advance_from_parts(
                    &mut streams[stream],
                    stream_base + stream,
                    reply.completed,
                    value,
                    reply.opponent,
                    rollout,
                    report,
                )?;
            }
        }
        if let Some(pending) = &mut pending {
            pending.finish_fallback(model, streams, stream_base, &prepared, rollout, report)?;
        }
        workers.finish()
    })
}

fn run_stream_job(
    environment: &mut TrainingEnvironment,
    job: StreamJob,
    config: PpoConfig,
    evaluator: Option<&FlushEvaluator>,
) -> Result<StreamReply, PpoError> {
    if matches!(&job, StreamJob::PrepareNeural) {
        let sample = prepare_policy_sample(environment)?;
        let opponent = neural_opponent::prepare(environment)?;
        return Ok(StreamReply::PreparedNeural(
            Box::new(sample),
            Box::new(opponent),
        ));
    }
    let StreamJob::Advance {
        mut state,
        choice,
        space,
        opponent,
    } = job
    else {
        return prepare_policy_sample(environment)
            .map(|sample| StreamReply::Prepared(Box::new(sample)));
    };
    let batched_opponent = opponent.is_some();
    let completed = advance_cpu_with_opponent(
        environment,
        &mut state,
        choice,
        space,
        config,
        opponent.map(|value| *value),
    )?;
    let (value, prepared) = prepare_after_advance(environment, &state, evaluator)?;
    let opponent_prepared = if batched_opponent && !state.done {
        Some(Box::new(neural_opponent::prepare(environment)?))
    } else {
        None
    };
    Ok(StreamReply::Advanced(Box::new(AdvancedReply {
        state,
        completed,
        value,
        opponent: opponent_name(&environment.opponent),
        prepared,
        opponent_prepared,
    })))
}

type PreparedNext = (Option<FlushValue>, Option<Box<(FeatureFrame, ActionSpace)>>);

fn prepare_after_advance(
    environment: &mut TrainingEnvironment,
    state: &EpisodeStream,
    evaluator: Option<&FlushEvaluator>,
) -> Result<PreparedNext, PpoError> {
    let mut value = None;
    let prepared = if state.done {
        None
    } else {
        let sample = prepare_policy_sample(environment)?;
        if state.should_flush()
            && let Some(evaluator) = evaluator
        {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            evaluator.submit(sample.0.clone(), sender)?;
            value = Some(receiver);
        }
        Some(Box::new(sample))
    };
    assert_eq!(prepared.is_none(), state.done);
    assert!(value.is_none() || state.should_flush());
    Ok((value, prepared))
}

fn retention_phase(seed: u64, update: u64, stream: usize) -> Result<usize, PpoError> {
    assert!(stream < crate::PPO_MAX_GAMES);
    let episode = derive_training_seed(seed, update, RETENTION_DOMAIN);
    let seed = derive_training_seed(episode, stream as u64, RETENTION_STREAM_DOMAIN);
    // A dedicated stream and power-of-two mask avoid actor RNG consumption and modulo bias.
    let mut random = PpoRng::new(seed);
    let phase = (random.next_word()? & (RETENTION_STRIDE as u64 - 1)) as usize;
    assert!(phase < RETENTION_STRIDE);
    Ok(phase)
}

/// One stream whose retention phase is a pure function of the run seed and
/// the global game index.
///
/// The annealed loop gives every game its own stream, so batches sharing one
/// rollout never collide on a retention phase or a stream index.
pub(super) fn game_stream(seed: u64, game: u64) -> Result<EpisodeStream, PpoError> {
    Ok(EpisodeStream {
        retention_phase: retention_phase(seed, game, 0)?,
        ..EpisodeStream::default()
    })
}

/// Runs one annealed batch of episodes over already built environments.
/// Reuse keeps actor batches unchanged but permits different bootstrap bits from
/// their larger GEMMs; the caller must bind this choice to the checkpoint scope.
///
/// `rounds` is the production episode ceiling or a test-only shorter window.
/// A full ceiling asserts every stream reached its terminal; a shorter window
/// stops each stream mid-episode exactly like the bounded benchmark slice.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_batch_with_actor_values(
    model: &PolicyModel,
    config: PpoConfig,
    stream_base: usize,
    thread_prefix: &str,
    environments: &mut [TrainingEnvironment],
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
    reuse_actor_values: bool,
) -> Result<(), PpoError> {
    assert!(!environments.is_empty());
    assert_eq!(streams.len(), environments.len());
    assert_eq!(random.len(), environments.len());
    if rounds == 0 || rounds > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("annealed collection rounds"));
    }
    collect_with_workers(
        model,
        config,
        stream_base,
        thread_prefix,
        environments,
        streams,
        random,
        rounds,
        rollout,
        report,
        reuse_actor_values,
    )?;
    finish_annealed_batch(streams, rounds);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_batch_with_opponent_batching(
    model: &PolicyModel,
    config: PpoConfig,
    stream_base: usize,
    thread_prefix: &str,
    environments: &mut [TrainingEnvironment],
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
    reuse_actor_values: bool,
    neural_opponent_batching: bool,
) -> Result<(), PpoError> {
    if !neural_opponent_batching {
        return collect_batch_with_actor_values(
            model,
            config,
            stream_base,
            thread_prefix,
            environments,
            streams,
            random,
            rounds,
            rollout,
            report,
            reuse_actor_values,
        );
    }
    actor_pipeline::collect_single_neural(
        model,
        config,
        stream_base,
        environments,
        streams,
        random,
        rounds,
        rollout,
        report,
        reuse_actor_values,
    )?;
    finish_annealed_batch(streams, rounds);
    Ok(())
}

fn finish_annealed_batch(streams: &[EpisodeStream], rounds: usize) {
    assert!(!streams.is_empty());
    assert!(streams.len() <= MAX_ACTOR_ENVIRONMENTS);
    if rounds == ACTOR_DECISIONS {
        assert!(
            streams.iter().all(|stream| stream.done),
            "a full annealed batch finishes every episode"
        );
    }
}

/// Bootstrap barrier: every stream prepares its first decision frame.
fn prepare_all_workers<'scope>(
    workers: &super::parallel::StreamWorkers<'scope, TrainingEnvironment, StreamJob, StreamReply>,
    stream_count: usize,
) -> Result<Vec<Option<(FeatureFrame, ActionSpace)>>, PpoError> {
    assert!(stream_count >= 1);
    assert!(stream_count <= MAX_ACTOR_ENVIRONMENTS);
    let active: Vec<usize> = (0..stream_count).collect();
    for &stream in &active {
        workers.submit(stream, StreamJob::Prepare)?;
    }
    let prepared: Vec<StreamReply> = workers.receive(&active)?;
    assert_eq!(prepared.len(), stream_count);
    let mut samples = Vec::with_capacity(stream_count);
    for reply in prepared {
        let StreamReply::Prepared(sample) = reply else {
            return Err(PpoError::InvalidConfig("stream worker reply"));
        };
        samples.push(Some(*sample));
    }
    Ok(samples)
}

/// One batched policy forward and sampled choice set per active stream.
fn sample_choices(
    model: &PolicyModel,
    random: &mut [PpoRng],
    active: &[usize],
    frames: Vec<FeatureFrame>,
    spaces: Vec<ActionSpace>,
) -> Result<(Vec<PpoPolicyChoice>, Vec<ActionSpace>), PpoError> {
    validate_active(random, active)?;
    assert_eq!(frames.len(), active.len());
    assert_eq!(spaces.len(), active.len());
    select_choices(model, random, active, frames, spaces)
}

fn validate_active(random: &[PpoRng], active: &[usize]) -> Result<(), PpoError> {
    assert!(!active.is_empty());
    if active.len() > MAX_ACTOR_ENVIRONMENTS
        || random.len() > MAX_ACTOR_ENVIRONMENTS
        || active.iter().any(|index| *index >= random.len())
        || active.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PpoError::InvalidConfig("episode active streams"));
    }
    Ok(())
}

fn select_choices(
    model: &PolicyModel,
    random: &mut [PpoRng],
    active: &[usize],
    frames: Vec<FeatureFrame>,
    spaces: Vec<ActionSpace>,
) -> Result<(Vec<PpoPolicyChoice>, Vec<ActionSpace>), PpoError> {
    assert_eq!(frames.len(), active.len());
    assert_eq!(spaces.len(), active.len());
    let mut selected_random: Vec<_> = active
        .iter()
        .map(|&stream| random[stream].clone())
        .collect();
    let choices = model
        .sample_batch(&frames, &spaces, &mut selected_random)
        .map_err(text_error)?;
    for (&stream, state) in active.iter().zip(selected_random) {
        random[stream] = state;
    }
    assert_eq!(choices.len(), spaces.len());
    Ok((choices, spaces))
}

#[derive(Default)]
pub(super) struct EpisodeStream {
    #[cfg(test)]
    pub(super) trace: std::collections::hash_map::DefaultHasher,
    #[cfg(test)]
    last_requests: Option<Vec<Option<Request>>>,
    retention_phase: usize,
    map2_reward: Map2TrainingReward,
    elapsed_ticks: u32,
    raw_return: f64,
    discounted_return: f64,
    shaping_return: f64,
    terminal_reward: f32,
    actions: [u32; ActionKind::COUNT],
    choice: Option<PpoPolicyChoice>,
    interval: DiscountedInterval,
    decisions: usize,
    retained: u32,
    done: bool,
    summary: Option<super::game_summary::GameSummary>,
}

impl EpisodeStream {
    fn begins_interval(&self) -> bool {
        assert!(self.retention_phase < RETENTION_STRIDE);
        self.decisions >= self.retention_phase
            && (self.decisions - self.retention_phase).is_multiple_of(RETENTION_STRIDE)
    }

    fn append_retained_reward(
        &mut self,
        reward: f64,
        ticks: u32,
        gamma: f32,
    ) -> Result<(), PpoError> {
        if self.choice.is_some() {
            self.interval.append(reward, ticks, gamma)?;
        } else {
            assert_eq!(self.interval.steps, 0);
        }
        Ok(())
    }

    fn should_flush(&self) -> bool {
        self.choice.is_some() && (self.interval.steps == RETENTION_STRIDE || self.done)
    }
}

#[derive(Default)]
struct DiscountedInterval {
    reward: f64,
    ticks: u32,
    steps: usize,
}

impl DiscountedInterval {
    fn append(&mut self, reward: f64, ticks: u32, gamma: f32) -> Result<(), PpoError> {
        if !reward.is_finite() {
            return Err(PpoError::NonFinite("episode reward"));
        }
        let _ = tick_discount(gamma, ticks)?;
        assert!(self.steps < RETENTION_STRIDE);
        let discount = if self.ticks == 0 {
            1.0
        } else {
            tick_discount(gamma, self.ticks)?
        };
        self.reward += f64::from(discount) * reward;
        self.ticks = self
            .ticks
            .checked_add(ticks)
            .ok_or(PpoError::CounterOverflow)?;
        self.steps += 1;
        assert!(self.reward.is_finite());
        Ok(())
    }

    fn finish(self, stream: usize, decision: u32, terminal: bool, next_value: f32) -> PpoOutcome {
        assert!(self.steps > 0);
        assert!(self.steps <= RETENTION_STRIDE);
        PpoOutcome {
            stream,
            decision,
            ticks: self.ticks,
            reward: self.reward as f32,
            terminal,
            next_value: if terminal { 0.0 } else { next_value },
        }
    }
}

#[cfg(test)]
/// Applies one sampled decision and advances the world through the serial
/// reward and retained-interval bookkeeping shared with the serial slice and
/// the parity tests.
fn advance_stream(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    sample: (usize, PpoPolicyChoice, ActionSpace),
    config: PpoConfig,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
) -> Result<(), PpoError> {
    let (stream, choice, space) = sample;
    let completed = advance_cpu(environment, state, choice, space, config)?;
    finish_advance(
        model,
        environment,
        state,
        stream,
        completed,
        rollout,
        report,
    )
}

#[derive(Clone, Copy)]
struct CompletedAdvance {
    end_tick: u32,
    ticks: u32,
    outcome: Option<PpoTerminalOutcome>,
}

#[cfg(test)]
fn advance_cpu(
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    choice: PpoPolicyChoice,
    space: ActionSpace,
    config: PpoConfig,
) -> Result<CompletedAdvance, PpoError> {
    advance_cpu_with_opponent(environment, state, choice, space, config, None)
}

fn advance_cpu_with_opponent(
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    choice: PpoPolicyChoice,
    space: ActionSpace,
    config: PpoConfig,
    opponent: Option<neural_opponent::OpponentChoice>,
) -> Result<CompletedAdvance, PpoError> {
    assert!(!state.done);
    assert!(state.decisions < ACTOR_DECISIONS);
    if state.begins_interval() {
        assert!(state.choice.is_none());
        assert_eq!(state.interval.steps, 0);
        state.choice = Some(choice.clone());
    }
    let tick = environment.seats[environment.policy_seat]
        .tracker
        .current()
        .ok_or(PpoError::InvalidTransition("episode snapshot"))?
        .tick;
    assert!(tick < TICK_CAP);
    let requests = match opponent {
        Some(opponent) => requests_for_prepared_opponent(environment, &choice, &space, opponent)?,
        None => requests_for_decision_in_space(environment, &choice, &space)?,
    };
    #[cfg(test)]
    {
        use std::hash::Hash;
        choice.action().hash(&mut state.trace);
        format!("{requests:?}").hash(&mut state.trace);
        state.last_requests = Some(requests.clone());
    }
    let advanced = advance_interval(
        environment,
        requests,
        config.decision_interval_ticks.min(TICK_CAP - tick),
    )?;
    reject_production_rejection(environment, "complete episode rollout")?;
    let outcome = terminal_outcome(environment, advanced.winner);
    state.done = outcome.is_some() || tick + advanced.ticks >= TICK_CAP;
    if state.done {
        state.summary = Some(super::game_summary::GameSummary::capture(
            environment,
            outcome,
            tick + advanced.ticks,
        ));
    }
    let reward = observe_reward(
        environment,
        state,
        outcome,
        advanced.ticks,
        config.gamma_tick,
    )?;
    state.actions[choice.action().kind().index()] += 1;
    state.append_retained_reward(reward, advanced.ticks, config.gamma_tick)?;
    state.decisions += 1;
    Ok(CompletedAdvance {
        end_tick: tick + advanced.ticks,
        ticks: advanced.ticks,
        outcome,
    })
}

#[cfg(test)]
/// Serial completion of one advanced decision: retained-interval flush with an
/// in-line next-value evaluation, then terminal episode bookkeeping.
fn finish_advance(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    stream: usize,
    completed: CompletedAdvance,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
) -> Result<(), PpoError> {
    report.elapsed_ticks = report
        .elapsed_ticks
        .checked_add(u64::from(completed.ticks))
        .ok_or(PpoError::CounterOverflow)?;
    if state.should_flush() {
        flush(model, environment, state, stream, state.done, rollout)?;
    }
    if state.done {
        record_episode(
            stream,
            completed.end_tick,
            state,
            completed.outcome,
            report,
            opponent_name(&environment.opponent),
        )?;
    }
    Ok(())
}

/// Worker-thread variant: the next value and opponent name were prepared on
/// the owning stream worker, so no environment access is needed here.
#[allow(clippy::too_many_arguments)]
fn finish_advance_from_parts(
    state: &mut EpisodeStream,
    stream: usize,
    completed: CompletedAdvance,
    value: Option<f32>,
    opponent: &'static str,
    rollout: &mut PpoRollout,
    report: &mut CollectionReport,
) -> Result<(), PpoError> {
    report.elapsed_ticks = report
        .elapsed_ticks
        .checked_add(u64::from(completed.ticks))
        .ok_or(PpoError::CounterOverflow)?;
    if state.should_flush() {
        let terminal = state.done;
        let next_value = if terminal {
            0.0
        } else {
            value.ok_or(PpoError::InvalidTransition("episode flush value"))?
        };
        flush_value(state, stream, terminal, next_value, rollout)?;
    }
    if state.done {
        record_episode(
            stream,
            completed.end_tick,
            state,
            completed.outcome,
            report,
            opponent,
        )?;
    }
    Ok(())
}

fn observe_reward(
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    outcome: Option<PpoTerminalOutcome>,
    ticks: u32,
    gamma: f32,
) -> Result<f64, PpoError> {
    assert!(ticks > 0);
    assert!(state.elapsed_ticks < TICK_CAP);
    assert_eq!(gamma, MAP2_REWARD_GAMMA_TICK);
    let end = map2_reward_end(outcome).or_else(|| state.done.then_some(Map2RewardEnd::TimeCap));
    let reward = take_map2_reward(environment, end, ticks)?;
    state.map2_reward.record(reward)?;
    let emitted = reward.total;
    state.raw_return += emitted;
    state.discounted_return += f64::from(gamma).powi(state.elapsed_ticks as i32) * emitted;
    state.shaping_return += reward.total - reward.terminal;
    state.terminal_reward = reward.terminal as f32;
    state.elapsed_ticks += ticks;
    assert_eq!(state.raw_return, state.discounted_return);
    Ok(emitted)
}

#[cfg(test)]
/// Serial retained-interval flush used by [`advance_stream`]; the worker pool
/// evaluates the same next frame on its dedicated evaluator thread instead.
fn flush(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    stream: usize,
    terminal: bool,
    rollout: &mut PpoRollout,
) -> Result<(), PpoError> {
    let next_value = if terminal {
        0.0
    } else {
        model
            .evaluate_batch(&[encode_next_frame(environment)?])
            .map_err(text_error)?[0]
            .value
    };
    flush_value(state, stream, terminal, next_value, rollout)
}

fn flush_value(
    state: &mut EpisodeStream,
    stream: usize,
    terminal: bool,
    next_value: f32,
    rollout: &mut PpoRollout,
) -> Result<(), PpoError> {
    assert!(state.retained < RETAINED_PER_EPISODE as u32);
    let choice = state
        .choice
        .take()
        .ok_or(PpoError::InvalidTransition("episode retained action"))?;
    assert_eq!(choice.policy(), rollout.policy());
    let outcome =
        std::mem::take(&mut state.interval).finish(stream, state.retained, terminal, next_value);
    rollout.push(choice.finish(outcome)?)?;
    state.retained += 1;
    Ok(())
}

fn record_episode(
    stream: usize,
    tick: u32,
    state: &EpisodeStream,
    outcome: Option<PpoTerminalOutcome>,
    report: &mut CollectionReport,
    opponent: &'static str,
) -> Result<(), PpoError> {
    let label = accumulate_episode(stream, tick, state, outcome, report)?;
    eprintln!(
        "episode: stream={stream} map=2 opponent={opponent} tick={tick} outcome={label} actor_decisions={} retained={} terminal_sample={} raw_return={:.9} discounted_return={:.9} terminal_reward={} shaping_return={:.9} actions={:?} noncontinue={} retention_phase={}",
        state.decisions,
        state.retained,
        state.retained > 0,
        state.raw_return,
        state.discounted_return,
        state.terminal_reward,
        state.shaping_return,
        state.actions,
        state.decisions - state.actions[ActionKind::Continue.index()] as usize,
        state.retention_phase
    );
    let mut output =
        crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default());
    output.emit(&format_args!(
        "level=INFO event=map2_episode_reward stream={stream} tick={tick} outcome={label} opponent={opponent} {}",
        state.map2_reward
    ));
    let summary = state
        .summary
        .as_ref()
        .ok_or(PpoError::InvalidTransition("episode summary"))?;
    output.emit(&format_args!(
        "level=INFO event=episode_summary stream={stream} opponent={opponent} {summary}"
    ));
    Ok(())
}

fn accumulate_episode(
    stream: usize,
    tick: u32,
    state: &EpisodeStream,
    outcome: Option<PpoTerminalOutcome>,
    report: &mut CollectionReport,
) -> Result<&'static str, PpoError> {
    assert!(state.done);
    assert!(tick <= TICK_CAP);
    let (label, counter) = match outcome {
        Some(PpoTerminalOutcome::Win) => ("Win", &mut report.terminal_wins),
        Some(PpoTerminalOutcome::Loss) => ("Loss", &mut report.terminal_losses),
        Some(PpoTerminalOutcome::Draw) => ("Draw", &mut report.terminal_draws),
        None => ("TimeCap", &mut report.episode_timeouts),
    };
    *counter = counter.checked_add(1).ok_or(PpoError::CounterOverflow)?;
    report.map2_reward.merge(state.map2_reward)?;
    report.completed_episodes.record(
        tick,
        stream,
        match outcome {
            Some(PpoTerminalOutcome::Win) => crate::TrainingGameOutcome::Win,
            Some(PpoTerminalOutcome::Loss) => crate::TrainingGameOutcome::Loss,
            Some(PpoTerminalOutcome::Draw) => crate::TrainingGameOutcome::Draw,
            None => crate::TrainingGameOutcome::TimeCap,
        },
    )?;
    Ok(label)
}
