#![allow(
    clippy::float_arithmetic,
    reason = "Discounted n-step rewards use floating-point arithmetic"
)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::*;

#[cfg(test)]
#[path = "../tests/opponent_curriculum.rs"]
mod curriculum_tests;

#[cfg(test)]
#[path = "../tests/mastery_training.rs"]
mod mastery_tests;

#[cfg(test)]
#[path = "../tests/map2_collection.rs"]
mod map2_tests;

#[cfg(test)]
#[path = "../tests/episode_collection_parallel.rs"]
mod parallel_tests;

pub(super) const TICK_CAP: u32 = crate::MAP2_TICK_CAP;
pub(super) const ACTOR_DECISIONS: usize = crate::MAP2_ACTOR_DECISIONS;
const RETENTION_STRIDE: usize = crate::MAP2_RETENTION_STRIDE;
const RETENTION_DOMAIN: u64 = 0x7265_7465_6e74_696f;
const RETENTION_STREAM_DOMAIN: u64 = 0x7068_6173_655f_726e;
const _: () = assert!(RETENTION_STRIDE.is_power_of_two());
const RETAINED_PER_EPISODE: usize = crate::MAP2_RETAINED_DECISIONS;
const MAX_EPISODE_ENVIRONMENTS: usize = super::TRAINING_MAX_ENVIRONMENTS;
const RETAINED_BYTES_PER_ENVIRONMENT: usize = 3
    * RETAINED_PER_EPISODE
    * (std::mem::size_of::<FeatureFrame>() + std::mem::size_of::<crate::BehavioralTarget>());
const _: () = assert!(MAX_EPISODE_ENVIRONMENTS.is_multiple_of(2));
const _: () =
    assert!(MAX_EPISODE_ENVIRONMENTS * RETAINED_BYTES_PER_ENVIRONMENT < 6 * 1024 * 1024 * 1024);
const _: () = assert!(MAX_EPISODE_ENVIRONMENTS * RETAINED_PER_EPISODE <= crate::PPO_MAX_SAMPLES);

/// Map2 comprehensive-reward contract shared by the parallel and serial collectors.
fn validate_reward_contract(settings: &TrainingJobConfig) -> Result<(), PpoError> {
    if (settings.opponent_schedule == crate::TrainingOpponentSchedule::MasteryV1)
        != settings.mastery_config.is_some()
    {
        return Err(PpoError::InvalidConfig(
            "mastery-v1 requires mastery config; other schedules forbid it",
        ));
    }
    if settings.map != MapId(2) {
        return Err(PpoError::InvalidConfig("production training requires Map2"));
    }
    if settings.ppo.gamma_tick != MAP2_REWARD_GAMMA_TICK {
        return Err(PpoError::InvalidConfig(
            "Map2 comprehensive reward requires gamma per tick one",
        ));
    }
    if settings.terminal_only {
        return Err(PpoError::InvalidConfig(
            "Map2 requires comprehensive reward; terminal-only is unsupported",
        ));
    }
    if settings.episode_time_cost != 0.0 {
        return Err(PpoError::InvalidConfig(
            "Map2 comprehensive reward requires zero episode time cost",
        ));
    }
    Ok(())
}

pub(crate) fn validate(settings: &TrainingJobConfig) -> Result<(), PpoError> {
    validate_reward_contract(settings)?;
    validate_pipeline_groups(settings)?;
    if !settings.complete_episodes {
        if settings.opponent_schedule != crate::TrainingOpponentSchedule::Teacher {
            return Err(PpoError::InvalidConfig(
                "opponent curriculum requires complete episodes",
            ));
        }
        return Ok(());
    }
    if !valid_environment_count(settings.ppo.environments)
        || settings.ppo.decision_interval_ticks != crate::MAP2_DECISION_INTERVAL_TICKS
    {
        return Err(PpoError::InvalidConfig(
            "complete episodes require Map2, an even environment count up to the training maximum, and three-tick actions",
        ));
    }
    if settings.ppo.rollout_decisions < RETAINED_PER_EPISODE {
        return Err(PpoError::InvalidConfig(
            "complete episode retained capacity",
        ));
    }
    assert!(settings.ppo.environments * RETAINED_BYTES_PER_ENVIRONMENT < 6 * 1024 * 1024 * 1024);
    assert!(valid_environment_count(settings.ppo.environments));
    settings.ppo.validate()?;
    Ok(())
}

/// Settings of the one-stream serial baseline: the same Map2 reward contract
/// and cadence as production, with only the paired evenness rule lifted, since
/// one stream has no pair to schedule against. `PpoConfig` admits one
/// environment; this predicate exists for the benchmark slice and its tests.
fn validate_serial(settings: &TrainingJobConfig) -> Result<(), PpoError> {
    validate_reward_contract(settings)?;
    validate_pipeline_groups(settings)?;
    if !settings.complete_episodes
        || settings.ppo.environments != 1
        || settings.ppo.decision_interval_ticks != crate::MAP2_DECISION_INTERVAL_TICKS
    {
        return Err(PpoError::InvalidConfig(
            "serial collection requires one complete-episode environment and three-tick actions",
        ));
    }
    if settings.ppo.rollout_decisions < RETAINED_PER_EPISODE {
        return Err(PpoError::InvalidConfig(
            "complete episode retained capacity",
        ));
    }
    settings.ppo.validate()?;
    Ok(())
}

/// Even stream counts up to the training maximum keep pair seeding and the
/// mastery batch math simple; odd counts have no paired policy seat.
pub(super) fn valid_environment_count(count: usize) -> bool {
    (2..=super::TRAINING_MAX_ENVIRONMENTS).contains(&count) && count.is_multiple_of(2)
}

/// Pipeline-group contract: one global barrier by default, or 2/4 fixed groups
/// each running its own paired complete-episode batch concurrently.
///
/// Complete episodes are required because only they define one episode per
/// stream and therefore a global update boundary independent of the groups. A
/// group must own at least one full paired batch, so the count cannot exceed
/// the number of environment pairs.
pub(super) fn validate_pipeline_groups(settings: &TrainingJobConfig) -> Result<(), PpoError> {
    if !matches!(settings.pipeline_groups, 1 | 2 | 4) {
        return Err(PpoError::InvalidConfig("pipeline groups"));
    }
    if settings.pipeline_groups > 1 && !settings.complete_episodes {
        return Err(PpoError::InvalidConfig(
            "pipeline groups require complete episodes",
        ));
    }
    if settings.pipeline_groups > 1 && settings.ppo.environments / 2 < settings.pipeline_groups {
        return Err(PpoError::InvalidConfig(
            "pipeline groups exceed paired environments",
        ));
    }
    Ok(())
}

/// Fixed pair-aligned partition of `environments` streams into `groups` ranges.
///
/// Groups are contiguous by pair: group boundaries never split the mirrored
/// policy seats that one seed pair schedules against each other, and groups
/// differ by at most one pair, so their expected episode work is balanced.
/// The sizes are a pure function of `(environments, groups)`, which makes the
/// partition deterministic and reproducible without any runtime scheduling.
pub(super) fn training_group_ranges(
    environments: usize,
    groups: usize,
) -> Result<Vec<std::ops::Range<usize>>, PpoError> {
    if !matches!(groups, 1 | 2 | 4) {
        return Err(PpoError::InvalidConfig("pipeline groups"));
    }
    if !valid_environment_count(environments) {
        return Err(PpoError::InvalidConfig("pipeline group environments"));
    }
    let pairs = environments / 2;
    if pairs < groups {
        return Err(PpoError::InvalidConfig(
            "pipeline groups exceed paired environments",
        ));
    }
    let base = pairs / groups;
    let extra = pairs % groups;
    let mut ranges = Vec::with_capacity(groups);
    let mut first_pair = 0;
    for group in 0..groups {
        let group_pairs = base + usize::from(group < extra);
        let start = first_pair * 2;
        first_pair += group_pairs;
        ranges.push(start..first_pair * 2);
    }
    assert_eq!(first_pair, pairs);
    assert_eq!(
        ranges.iter().map(std::ops::Range::len).sum::<usize>(),
        environments
    );
    assert!(ranges.iter().all(|range| {
        range.len() >= 2 && range.len().is_multiple_of(2) && valid_environment_count(range.len())
    }));
    Ok(ranges)
}

#[cfg(test)]
fn validate_time_cost(budget: f32) -> Result<(), PpoError> {
    if !budget.is_finite() || !(0.0..=0.25).contains(&budget) {
        return Err(PpoError::InvalidConfig(
            "episode time cost must be finite in [0, 0.25]",
        ));
    }
    Ok(())
}

#[cfg(test)]
fn elapsed_time_cost(budget: f32, elapsed: u32, ticks: u32) -> Result<f64, PpoError> {
    validate_time_cost(budget)?;
    if ticks == 0 {
        return Err(PpoError::InvalidTransition("episode time-cost zero ticks"));
    }
    let end = elapsed
        .checked_add(ticks)
        .ok_or(PpoError::CounterOverflow)?;
    if end >= TICK_CAP {
        return Err(PpoError::InvalidTransition(
            "episode time-cost elapsed tick bound",
        ));
    }
    let before = f64::from(budget) * f64::from(elapsed) / f64::from(TICK_CAP);
    let after = f64::from(budget) * f64::from(end) / f64::from(TICK_CAP);
    assert!(after <= f64::from(budget));
    assert!(after >= before);
    Ok(after - before)
}

pub(super) fn environments(
    settings: &TrainingJobConfig,
    update: u64,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    environments_with_mastery(settings, update, None)
}

pub(super) fn environments_with_mastery(
    settings: &TrainingJobConfig,
    update: u64,
    mastery: Option<&crate::MasteryProgress>,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    validate(settings)?;
    assert!(settings.complete_episodes);
    let mastery_teacher = mastery_teacher_stage(settings, mastery)?;
    let first_pair = update
        .checked_mul((settings.ppo.environments / 2) as u64)
        .ok_or(PpoError::CounterOverflow)?;
    (0..settings.ppo.environments)
        .map(|stream| {
            let pair = first_pair
                .checked_add((stream / 2) as u64)
                .ok_or(PpoError::CounterOverflow)?;
            build_environment(
                derive_training_seed(settings.seed, pair, 0x6172_656e_615f_7365),
                derive_training_seed(settings.seed, pair, 0x6f70_706f_6e65_6e74),
                settings.map,
                stream % 2,
                0,
                match mastery_teacher {
                    Some(true) => OpponentSpec::Teacher,
                    Some(false) => OpponentSpec::Weak,
                    None => scheduled_opponent(settings, update, stream),
                },
            )
        })
        .collect()
}

fn mastery_teacher_stage(
    settings: &TrainingJobConfig,
    mastery: Option<&crate::MasteryProgress>,
) -> Result<Option<bool>, PpoError> {
    match (settings.mastery_config, mastery) {
        (None, None) => Ok(None),
        (Some(config), Some(progress)) => {
            progress.validate(config).map_err(PpoError::InvalidConfig)?;
            if progress.completed() {
                return Err(PpoError::InvalidConfig("mastery already completed"));
            }
            Ok(Some(progress.stage() == crate::MasteryStage::Teacher))
        }
        _ => Err(PpoError::InvalidConfig(
            "mastery configuration/state mismatch",
        )),
    }
}

fn scheduled_opponent(settings: &TrainingJobConfig, update: u64, stream: usize) -> OpponentSpec {
    assert!(settings.complete_episodes);
    assert!(stream < settings.ppo.environments);
    if settings.opponent_schedule.is_teacher(update, stream / 2) {
        OpponentSpec::Teacher
    } else {
        OpponentSpec::Weak
    }
}

/// The stream-zero world of `update`, built exactly as the even-count
/// production collector builds its first pair, for the serial baseline.
///
/// Pair index `update` matches the first pair of an even run at the same
/// update, so the serial slice and stream zero of the two-stream slice start
/// from identical worlds. The mastery argument follows
/// [`environments_with_mastery`]: `Some` for mastery schedules, `None` for the
/// legacy schedules that only admit up to three pairs.
pub(super) fn single_environment(
    settings: &TrainingJobConfig,
    update: u64,
    mastery: Option<&crate::MasteryProgress>,
) -> Result<TrainingEnvironment, PpoError> {
    validate_serial(settings)?;
    let mastery_teacher = mastery_teacher_stage(settings, mastery)?;
    build_environment(
        derive_training_seed(settings.seed, update, 0x6172_656e_615f_7365),
        derive_training_seed(settings.seed, update, 0x6f70_706f_6e65_6e74),
        settings.map,
        0,
        0,
        match mastery_teacher {
            Some(true) => OpponentSpec::Teacher,
            Some(false) => OpponentSpec::Weak,
            None => scheduled_opponent(settings, update, 0),
        },
    )
}

fn opponent_name(opponent: &OpponentRuntime) -> &'static str {
    match opponent {
        OpponentRuntime::Teacher => "Teacher",
        OpponentRuntime::Weak => "Weak",
        OpponentRuntime::Policy { .. } => "Policy",
    }
}

fn collection_opponents(environments: &[TrainingEnvironment]) -> (&'static str, usize, usize) {
    assert!(valid_environment_count(environments.len()));
    let weak = environments
        .iter()
        .filter(|arena| matches!(arena.opponent, OpponentRuntime::Weak))
        .count();
    let teacher = environments
        .iter()
        .filter(|arena| matches!(arena.opponent, OpponentRuntime::Teacher))
        .count();
    assert_eq!(weak + teacher, environments.len());
    let name = if weak == 0 {
        "Teacher"
    } else if teacher == 0 {
        "Weak"
    } else {
        "Mixed"
    };
    (name, weak, teacher)
}

/// Opt-in per-phase wall accounting for collection diagnosis.
///
/// Production always collects without a recorder; the benchmark slice passes
/// one recorder per group and reports the sums. Counters are relaxed atomics
/// because each group owns one recorder and phases only add disjoint
/// durations, so every run observes the same totals regardless of thread
/// interleaving. A zeroed recorder never changes collection results: the
/// instrumented operations are unchanged, only measured.
#[derive(Default)]
pub(super) struct CollectionPhases {
    /// Stream worker: policy sample preparation (feature encode plus orders).
    pub(super) prepare_ns: AtomicU64,
    /// Stream worker: sampled action application and three-tick world advance.
    pub(super) advance_ns: AtomicU64,
    /// Group orchestrator: batched policy forward, autoregressive decode and
    /// sampling for one round.
    pub(super) forward_ns: AtomicU64,
    /// Group orchestrator: waiting for every active stream worker's reply.
    pub(super) barrier_ns: AtomicU64,
    /// Group orchestrator: reply application, rollout push and episode
    /// bookkeeping. Contains `flush_wait_ns`.
    pub(super) apply_ns: AtomicU64,
    /// Group orchestrator: waiting for a retained-interval next value from the
    /// dedicated evaluator; a subset of `apply_ns`.
    pub(super) flush_wait_ns: AtomicU64,
    /// Flush evaluator: the batch-1 next-value forward.
    pub(super) evaluator_ns: AtomicU64,
}

impl CollectionPhases {
    /// Accumulates one optional measured phase into its counter.
    fn record(started: Option<Instant>, counter: Option<&AtomicU64>) {
        if let (Some(started), Some(counter)) = (started, counter) {
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            counter.fetch_add(nanos, Ordering::Relaxed);
        }
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
    /// Apply one chosen action and advance the owned world three ticks.
    Advance {
        state: EpisodeStream,
        choice: PpoPolicyChoice,
        space: ActionSpace,
    },
}

/// One ordered reply from a stream worker.
enum StreamReply {
    Prepared(Box<(FeatureFrame, ActionSpace)>),
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
fn flush_evaluator_loop(
    model: &PolicyModel,
    receiver: &std::sync::mpsc::Receiver<FlushRequest>,
    phases: Option<&CollectionPhases>,
) {
    while let Ok((frame, reply)) = receiver.recv() {
        let started = phases.map(|_| Instant::now());
        let value = model
            .evaluate_batch(std::slice::from_ref(&frame))
            .map_err(model_error)
            .and_then(|output| {
                output
                    .into_iter()
                    .next()
                    .map(|output| output.value)
                    .ok_or(PpoError::InvalidTransition("episode flush value"))
            });
        CollectionPhases::record(started, phases.map(|phases| &phases.evaluator_ns));
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

/// Logs one collection batch before its first decision.
///
/// One group logs its own line, so an operator can attribute every stream
/// range to a group; the default one-group line is byte-identical to the
/// historical single-barrier collector.
fn log_collection(
    environments: &[TrainingEnvironment],
    settings: &TrainingJobConfig,
    update: u64,
    group: Option<(usize, usize)>,
) {
    let config = settings.ppo;
    let retained_bytes_bound = environments.len() * RETAINED_BYTES_PER_ENVIRONMENT;
    let (opponent, weak_environments, teacher_environments) = collection_opponents(environments);
    let prefix = match group {
        None => "collection:".to_owned(),
        Some((index, groups)) => format!("collection: group={}/{}", index + 1, groups),
    };
    eprintln!(
        "{prefix} complete-episodes map=2 reward=map2_full gamma_tick=1 opponent={opponent} actor_ticks={} retention_stride={RETENTION_STRIDE} retention_phase=random_episode_independent_rng tick_cap={TICK_CAP} environments={} retained_bytes_bound={retained_bytes_bound} opponent_schedule={} update_index={update} weak_environments={weak_environments} teacher_environments={teacher_environments}",
        config.decision_interval_ticks,
        environments.len(),
        settings.opponent_schedule.as_str(),
    );
}

/// Runs the worker-pool collector for at most `rounds` decision rounds.
///
/// Production always passes [`ACTOR_DECISIONS`] and `require_complete = true`,
/// which keeps the exact full-episode behaviour: every stream must finish and
/// the batch must contain at least one completed episode. The bounded
/// benchmark slice passes a smaller round count and `false`, so the engine
/// still performs one decision round per active stream per iteration while the
/// caller owns the window and terminal bookkeeping.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_bounded(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environments: &mut [TrainingEnvironment],
    settings: &TrainingJobConfig,
    update: u64,
    rounds: usize,
    require_complete: bool,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    collect_groups_bounded(
        model,
        sampling,
        environments,
        settings,
        update,
        1,
        rounds,
        require_complete,
        None,
        rollout,
        report,
    )
}

/// Runs the worker-pool collector as `groups` fixed independent batches.
///
/// One group is the default global-barrier collector. More groups partition
/// the paired streams by [`training_group_ranges`]; every group runs its own
/// bounded worker pool, decision barrier, batched forward and sampling on its
/// own orchestrator thread, so one group's forward or apply overlaps another
/// group's world stepping. The update boundary stays global: the caller sees
/// exactly the same number of completed episodes and one merged rollout in
/// fixed group order, which keeps the mode deterministic while changing batch
/// composition (a new training contract recorded in the checkpoint run scope).
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_groups_bounded(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environments: &mut [TrainingEnvironment],
    settings: &TrainingJobConfig,
    update: u64,
    groups: usize,
    rounds: usize,
    require_complete: bool,
    phases: Option<&[CollectionPhases]>,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    validate(settings)?;
    assert!(valid_environment_count(environments.len()));
    assert_eq!(
        settings.pipeline_groups, groups,
        "collection group request must match the typed settings"
    );
    assert_eq!(
        settings.ppo.decision_interval_ticks,
        crate::MAP2_DECISION_INTERVAL_TICKS
    );
    if rounds == 0 || rounds > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("collection rounds"));
    }
    let ranges = training_group_ranges(environments.len(), groups)?;
    if phases.is_some_and(|phases| phases.len() != groups) {
        return Err(PpoError::InvalidConfig("collection phase counters"));
    }
    let mut random = actor_stream_rngs(sampling, environments.len())?;
    let mut streams = streams_for_collection(settings, update)?;
    if groups == 1 {
        log_collection(environments, settings, update, None);
        let completed = collect_with_workers(
            model,
            settings.ppo,
            0,
            "ppo",
            environments,
            &mut streams,
            &mut random,
            rounds,
            None,
            phases.map(|phases| &phases[0]),
            rollout,
            report,
        )?;
        assert!(completed, "the one-group collector cannot cancel");
        finish_collection(&streams, environments, require_complete, report)?;
        return if require_complete {
            validate_episode_batch(rollout, report)
        } else {
            Ok(())
        };
    }
    let policy = model.policy_identity().map_err(model_error)?;
    let cancel = AtomicBool::new(false);
    let outcomes = std::thread::scope(|scope| -> Result<Vec<GroupOutcome>, PpoError> {
        let mut handles = Vec::with_capacity(groups);
        let mut environments_rest: &mut [TrainingEnvironment] = environments;
        let mut streams_rest = streams.into_iter();
        let mut random_rest = random.into_iter();
        for (index, range) in ranges.iter().enumerate() {
            let length = range.len();
            let (group_environments, rest) =
                std::mem::take(&mut environments_rest).split_at_mut(length);
            environments_rest = rest;
            let group_streams: Vec<_> = streams_rest.by_ref().take(length).collect();
            let group_random: Vec<_> = random_rest.by_ref().take(length).collect();
            let capacity = length
                .checked_mul(settings.ppo.rollout_decisions)
                .ok_or(PpoError::InvalidConfig("group rollout capacity"))?;
            let group_rollout = PpoRollout::new(capacity, policy)?;
            let group_report = PpoSmokeReport::default();
            let group_phases = phases.map(|phases| &phases[index]);
            let stream_base = range.start;
            let cancel = &cancel;
            let thread_prefix = format!("ppo-g{index}");
            log_collection(group_environments, settings, update, Some((index, groups)));
            let handle = match std::thread::Builder::new()
                .name(format!("ppo-group-{index}"))
                .spawn_scoped(scope, move || {
                    let result = collect_group(
                        model,
                        settings,
                        stream_base,
                        &thread_prefix,
                        group_environments,
                        group_streams,
                        group_random,
                        rounds,
                        require_complete,
                        cancel,
                        group_rollout,
                        group_report,
                        group_phases,
                    );
                    if result.is_err() {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    result
                }) {
                Ok(handle) => handle,
                Err(error) => {
                    cancel.store(true, Ordering::Relaxed);
                    return Err(PpoError::EpisodeWorker {
                        stream: stream_base,
                        cause: error.to_string(),
                    });
                }
            };
            handles.push(handle);
        }
        // Join in group order: erroring groups already cancelled the others,
        // so every join is bounded by one decision round of remaining work.
        let mut outcomes = Vec::with_capacity(groups);
        let mut first_error = None;
        let mut first_panic = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(outcome)) => outcomes.push(outcome),
                Ok(Err(error)) => {
                    first_error.get_or_insert(error);
                }
                Err(payload) => {
                    first_panic.get_or_insert(payload);
                }
            }
        }
        if let Some(payload) = first_panic {
            std::panic::resume_unwind(payload);
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(outcomes)
    })?;
    merge_group_outcomes(outcomes, require_complete, rollout, report)
}

/// One finished or cancelled collection group.
///
/// One outcome exists per group per update, so the boxed payloads trade two
/// cold-path allocations for a small enum moved through join results.
enum GroupOutcome {
    Complete {
        rollout: Box<PpoRollout>,
        report: Box<PpoSmokeReport>,
    },
    Cancelled,
}

/// Runs one group's complete collection on its own orchestrator thread.
#[allow(clippy::too_many_arguments)]
fn collect_group(
    model: &PolicyModel,
    settings: &TrainingJobConfig,
    stream_base: usize,
    thread_prefix: &str,
    environments: &mut [TrainingEnvironment],
    mut streams: Vec<EpisodeStream>,
    mut random: Vec<PpoRng>,
    rounds: usize,
    require_complete: bool,
    cancel: &AtomicBool,
    mut rollout: PpoRollout,
    mut report: PpoSmokeReport,
    phases: Option<&CollectionPhases>,
) -> Result<GroupOutcome, PpoError> {
    assert_eq!(streams.len(), environments.len());
    assert_eq!(random.len(), environments.len());
    let completed = collect_with_workers(
        model,
        settings.ppo,
        stream_base,
        thread_prefix,
        environments,
        &mut streams,
        &mut random,
        rounds,
        Some(cancel),
        phases,
        &mut rollout,
        &mut report,
    )?;
    if !completed {
        return Ok(GroupOutcome::Cancelled);
    }
    finish_collection(&streams, environments, require_complete, &mut report)?;
    Ok(GroupOutcome::Complete {
        rollout: Box::new(rollout),
        report: Box::new(report),
    })
}

/// Merges the fixed-group results into the one global update batch.
///
/// Group order is the partition order, never completion order, so the merged
/// sample sequence and the mastery outcome order are deterministic.
fn merge_group_outcomes(
    outcomes: Vec<GroupOutcome>,
    require_complete: bool,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    for outcome in outcomes {
        match outcome {
            GroupOutcome::Complete {
                rollout: group_rollout,
                report: group_report,
            } => {
                super::merge_actor_report(report, *group_report)?;
                rollout.append(*group_rollout)?;
            }
            GroupOutcome::Cancelled => {
                return Err(PpoError::InvalidConfig("collection group cancelled"));
            }
        }
    }
    if require_complete {
        return validate_episode_batch(rollout, report);
    }
    Ok(())
}

/// Runs one bounded worker pool: dedicated flush evaluator, one persistent
/// stream worker per environment, and the round loop that batches one forward
/// per decision and applies replies in stream order.
///
/// Returns `false` when `cancel` was observed at a round boundary, leaving
/// every worker idle; all exits join the pool and close the evaluator through
/// an enclosing scope guard, so no path leaks a thread.
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
    cancel: Option<&AtomicBool>,
    phases: Option<&CollectionPhases>,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<bool, PpoError> {
    assert_eq!(streams.len(), environments.len());
    assert_eq!(random.len(), environments.len());
    assert!(valid_environment_count(environments.len()));
    let (flush_sender, flush_receiver) =
        std::sync::mpsc::sync_channel::<FlushRequest>(MAX_EPISODE_ENVIRONMENTS);
    let flush_evaluator = FlushEvaluator {
        sender: std::sync::Mutex::new(Some(flush_sender)),
    };
    std::thread::scope(|scope| -> Result<bool, PpoError> {
        std::thread::Builder::new()
            .name(format!("{thread_prefix}-flush-eval"))
            .spawn_scoped(scope, move || {
                flush_evaluator_loop(model, &flush_receiver, phases)
            })
            .map_err(|error| PpoError::EpisodeWorker {
                stream: stream_base,
                cause: error.to_string(),
            })?;
        let _flush_guard = FlushThreadGuard(&flush_evaluator);
        let flush_evaluator_ref = &flush_evaluator;
        let workers = super::parallel::StreamWorkers::spawn(
            scope,
            environments,
            thread_prefix,
            |_, environment, job| {
                match job {
                    StreamJob::Prepare => {
                        let started = phases.map(|_| Instant::now());
                        let sample = prepare_policy_sample(environment);
                        CollectionPhases::record(started, phases.map(|phases| &phases.prepare_ns));
                        sample.map(|sample| StreamReply::Prepared(Box::new(sample)))
                    }
                    StreamJob::Advance {
                        state,
                        choice,
                        space,
                    } => {
                        let mut state = state;
                        let started = phases.map(|_| Instant::now());
                        let advanced = advance_cpu(environment, &mut state, choice, space, config);
                        CollectionPhases::record(started, phases.map(|phases| &phases.advance_ns));
                        let completed = advanced?;
                        // A terminal flush uses a zero next value and never
                        // encodes a frame, exactly like the serial collector.
                        // A non-terminal flush queues the exact next frame for
                        // the dedicated evaluator; the frame is reused as the
                        // next decision's prepared sample.
                        let (value, prepared) = if state.done {
                            (None, None)
                        } else if state.should_flush() {
                            let started = phases.map(|_| Instant::now());
                            let sample = prepare_policy_sample(environment);
                            CollectionPhases::record(
                                started,
                                phases.map(|phases| &phases.prepare_ns),
                            );
                            let sample = sample?;
                            let (value_sender, value_receiver) = std::sync::mpsc::sync_channel(1);
                            flush_evaluator_ref.submit(sample.0.clone(), value_sender)?;
                            (Some(value_receiver), Some(Box::new(sample)))
                        } else {
                            let started = phases.map(|_| Instant::now());
                            let sample = prepare_policy_sample(environment);
                            CollectionPhases::record(
                                started,
                                phases.map(|phases| &phases.prepare_ns),
                            );
                            (None, Some(Box::new(sample?)))
                        };
                        let opponent = opponent_name(&environment.opponent);
                        Ok(StreamReply::Advanced(Box::new(AdvancedReply {
                            state,
                            completed,
                            value,
                            opponent,
                            prepared,
                        })))
                    }
                }
            },
        )?;
        // Bootstrap: every stream builds its first frame before any action.
        let mut prepared = prepare_all_workers(&workers, streams.len())?;
        // Reused across decisions: the active set never changes shape between
        // rounds, only membership.
        let mut active: Vec<usize> = Vec::with_capacity(streams.len());
        let mut completed = true;
        for _ in 0..rounds {
            if cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
                completed = false;
                break;
            }
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
            let started = phases.map(|_| Instant::now());
            let sampled = sample_choices(model, random, &active, frames, spaces);
            CollectionPhases::record(started, phases.map(|phases| &phases.forward_ns));
            let (choices, spaces) = sampled?;
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
                    },
                )?;
            }
            assert!(samples.next().is_none());
            let started = phases.map(|_| Instant::now());
            let replies = workers.receive(&active);
            CollectionPhases::record(started, phases.map(|phases| &phases.barrier_ns));
            let replies = replies?;
            assert_eq!(replies.len(), active.len());
            let apply_started = phases.map(|_| Instant::now());
            for (stream, reply) in active.iter().copied().zip(replies) {
                let StreamReply::Advanced(reply) = reply else {
                    return Err(PpoError::InvalidConfig("stream worker reply"));
                };
                let mut reply = *reply;
                streams[stream] = std::mem::take(&mut reply.state);
                prepared[stream] = reply.prepared.map(|sample| *sample);
                assert_eq!(prepared[stream].is_none(), streams[stream].done);
                let value = match reply.value {
                    Some(receiver) => {
                        let started = phases.map(|_| Instant::now());
                        let received = receiver.recv();
                        CollectionPhases::record(
                            started,
                            phases.map(|phases| &phases.flush_wait_ns),
                        );
                        let value = received.map_err(|_| {
                            PpoError::InvalidConfig("episode flush evaluator reply")
                        })?;
                        Some(value?)
                    }
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
            CollectionPhases::record(apply_started, phases.map(|phases| &phases.apply_ns));
        }
        workers.finish()?;
        Ok(completed)
    })
}

/// Applies the shared completion contract to one finished group.
fn finish_collection(
    streams: &[EpisodeStream],
    environments: &[TrainingEnvironment],
    require_complete: bool,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    if require_complete {
        assert!(streams.iter().all(|stream| stream.done));
        assert!(streams.iter().all(|stream| stream.choice.is_none()));
        report.rejected_orders = environment_rejections(environments)?;
        return Ok(());
    }
    // A bounded slice must never finish or flush a final episode: the caller
    // needs the same number of active streams in every round it measures.
    assert!(
        streams.iter().all(|stream| !stream.done),
        "bounded collection window ended an episode"
    );
    Ok(())
}

/// Runs the one-stream collector serially on the calling thread.
///
/// Each round performs exactly the parallel collector's steady-state work:
/// prepare the policy sample, run the same batched sampler, apply the chosen
/// action, advance the world three ticks, and evaluate a retained-interval
/// value in line every eighth decision. Production requires an even paired
/// count, so this path is reachable only through
/// [`crate::TrainingCollectionSlice`].
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_serial_bounded(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environment: &mut TrainingEnvironment,
    settings: &TrainingJobConfig,
    update: u64,
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    validate_serial(settings)?;
    if rounds == 0 || rounds > ACTOR_DECISIONS {
        return Err(PpoError::InvalidConfig("serial collection rounds"));
    }
    assert_eq!(settings.ppo.environments, 1);
    assert!(!environment.seats.is_empty());
    let mut random = actor_stream_rngs(sampling, 1)?;
    let mut state = EpisodeStream {
        retention_phase: retention_phase(settings.seed, update, 0)?,
        ..EpisodeStream::default()
    };
    for _ in 0..rounds {
        let (frame, space) = prepare_policy_sample(environment)?;
        let (choices, spaces) = select_choices(model, &mut random, &[0], vec![frame], vec![space])?;
        let mut samples = choices.into_iter().zip(spaces);
        let (choice, space) = samples
            .next()
            .ok_or(PpoError::InvalidTransition("serial policy choice"))?;
        assert!(samples.next().is_none());
        advance_stream(
            model,
            environment,
            &mut state,
            (0, choice, space),
            settings.ppo,
            rollout,
            report,
        )?;
        assert!(
            !state.done,
            "bounded serial collection window ended an episode"
        );
    }
    Ok(())
}

fn validate_episode_batch(rollout: &PpoRollout, report: &PpoSmokeReport) -> Result<(), PpoError> {
    if rollout.is_empty() {
        return Err(PpoError::EmptyRollout);
    }
    if [
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.episode_timeouts,
    ]
    .iter()
    .all(|count| *count == 0)
    {
        return Err(PpoError::InvalidTransition(
            "complete episode batch has no completed episodes",
        ));
    }
    Ok(())
}

fn streams_for_collection(
    settings: &TrainingJobConfig,
    update: u64,
) -> Result<Vec<EpisodeStream>, PpoError> {
    validate(settings)?;
    (0..settings.ppo.environments)
        .map(|stream| {
            Ok(EpisodeStream {
                retention_phase: retention_phase(settings.seed, update, stream)?,
                ..EpisodeStream::default()
            })
        })
        .collect()
}

fn retention_phase(seed: u64, update: u64, stream: usize) -> Result<usize, PpoError> {
    assert!(stream < MAX_EPISODE_ENVIRONMENTS);
    let episode = derive_training_seed(seed, update, RETENTION_DOMAIN);
    let seed = derive_training_seed(episode, stream as u64, RETENTION_STREAM_DOMAIN);
    // A dedicated stream and power-of-two mask avoid actor RNG consumption and modulo bias.
    let mut random = PpoRng::new(seed);
    let phase = (random.next_word()? & (RETENTION_STRIDE as u64 - 1)) as usize;
    assert!(phase < RETENTION_STRIDE);
    Ok(phase)
}

/// Bootstrap barrier: every stream prepares its first decision frame.
fn prepare_all_workers<'scope>(
    workers: &super::parallel::StreamWorkers<'scope, TrainingEnvironment, StreamJob, StreamReply>,
    stream_count: usize,
) -> Result<Vec<Option<(FeatureFrame, ActionSpace)>>, PpoError> {
    assert!(stream_count >= 2);
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
    assert!(active.len() <= MAX_EPISODE_ENVIRONMENTS);
    if active.iter().any(|index| *index >= random.len())
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
        .map_err(model_error)?;
    for (&stream, state) in active.iter().zip(selected_random) {
        random[stream] = state;
    }
    assert_eq!(choices.len(), spaces.len());
    Ok((choices, spaces))
}

/// Test-only single-batch sampler: one spawn per call, same choices as the
/// persistent workers because the sampling itself is unchanged.
#[cfg(test)]
fn sample_active(
    model: &PolicyModel,
    random: &mut [PpoRng],
    environments: &mut [TrainingEnvironment],
    active: &[usize],
) -> Result<(Vec<PpoPolicyChoice>, Vec<ActionSpace>), PpoError> {
    validate_active(random, active)?;
    let prepared = super::parallel::ordered_active(
        environments,
        active.iter().map(|&stream| (stream, ())).collect(),
        |_, environment, ()| prepare_policy_sample(environment),
    )?;
    let (frames, spaces) = prepared.into_iter().unzip();
    select_choices(model, random, active, frames, spaces)
}

#[derive(Default)]
struct EpisodeStream {
    #[cfg(test)]
    trace: std::collections::hash_map::DefaultHasher,
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
    report: &mut PpoSmokeReport,
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

struct CompletedAdvance {
    end_tick: u32,
    ticks: u32,
    outcome: Option<PpoTerminalOutcome>,
}

fn advance_cpu(
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    choice: PpoPolicyChoice,
    space: ActionSpace,
    config: PpoConfig,
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
    let requests = requests_for_decision_in_space(environment, &choice, &space)?;
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

/// Serial completion of one advanced decision: retained-interval flush with an
/// in-line next-value evaluation, then terminal episode bookkeeping.
fn finish_advance(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    stream: usize,
    completed: CompletedAdvance,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
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
    report: &mut PpoSmokeReport,
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

#[cfg(test)]
pub(crate) fn assert_retention_actor_parity_for_test(model: &PolicyModel) {
    for phase in 1..RETENTION_STRIDE {
        assert_actor_phase_parity_for_test(model, phase);
    }
}

#[cfg(test)]
fn assert_actor_phase_parity_for_test(model: &PolicyModel, phase: usize) {
    let settings = parity_settings_for_test();
    let mut reference = environments(&settings, 3).expect("reference worlds");
    let mut candidate = environments(&settings, 3).expect("candidate worlds");
    let mut random = actor_stream_rngs(&mut PpoRng::new(9971001), 2).expect("actor RNGs");
    let mut replay = random.clone();
    let mut source = parity_states_for_test(0);
    let mut target = parity_states_for_test(phase);
    let mut retained_rewards = [0.0; 2];
    let mut source_rollout =
        PpoRollout::new(4, model.policy_identity().expect("identity")).expect("source rollout");
    let mut target_rollout = PpoRollout::new(4, source_rollout.policy()).expect("target rollout");
    for decision in 0..16 {
        let (choices, spaces) =
            sample_active(model, &mut random, &mut reference, &[0, 1]).expect("reference choices");
        let (copies, targets) =
            sample_active(model, &mut replay, &mut candidate, &[0, 1]).expect("candidate choices");
        assert_eq!(random, replay);
        for (stream, ((choice, space), (copy, target_space))) in choices
            .into_iter()
            .zip(spaces)
            .zip(copies.into_iter().zip(targets))
            .enumerate()
        {
            assert_eq!(choice.action(), copy.action());
            assert_eq!(choice.log_probability(), copy.log_probability());
            let previous_reward = source[stream].map2_reward.total;
            advance_stream(
                model,
                &mut reference[stream],
                &mut source[stream],
                (stream, choice, space),
                settings.ppo,
                &mut source_rollout,
                &mut PpoSmokeReport::default(),
            )
            .expect("phase zero action");
            advance_stream(
                model,
                &mut candidate[stream],
                &mut target[stream],
                (stream, copy, target_space),
                settings.ppo,
                &mut target_rollout,
                &mut PpoSmokeReport::default(),
            )
            .expect("shifted phase action");
            assert_eq!(source[stream].last_requests, target[stream].last_requests);
            assert_eq!(source[stream].map2_reward, target[stream].map2_reward);
            if (phase..phase + RETENTION_STRIDE).contains(&decision) {
                retained_rewards[stream] += source[stream].map2_reward.total - previous_reward;
            }
            assert_eq!(
                reference[stream].seats[stream].tracker.latest_summary(),
                candidate[stream].seats[stream].tracker.latest_summary()
            );
            if decision < phase {
                assert!(target[stream].choice.is_none());
                assert_eq!(target[stream].interval.steps, 0);
                assert_eq!(target[stream].map2_reward.ticks, (decision as u64 + 1) * 3);
            }
        }
    }
    assert_eq!(source_rollout.len(), 4);
    assert_eq!(target_rollout.len(), 2);
    let batch = target_rollout.finish(settings.ppo).expect("retained batch");
    for index in 0..2 {
        let sample = batch.sample(index).expect("sample").transition;
        assert_eq!(sample.ticks, 24);
        assert!((f64::from(sample.reward) - retained_rewards[sample.stream]).abs() < 1e-8);
    }
}

#[cfg(test)]
fn parity_settings_for_test() -> TrainingJobConfig {
    let settings = crate::cli::training_settings_for_test(&[
        "--complete-episodes",
        "--map",
        "2",
        "--environments",
        "2",
        "--rollout",
        &crate::MAP2_RETAINED_DECISIONS.to_string(),
        "--minibatch",
        "512",
        "--gamma-per-tick",
        "1",
    ])
    .expect("settings");
    assert_eq!(TICK_CAP, crate::MAP2_TICK_CAP);
    assert_eq!(ACTOR_DECISIONS, crate::MAP2_ACTOR_DECISIONS);
    assert_eq!(RETENTION_STRIDE, 8);
    assert_eq!(RETAINED_PER_EPISODE, crate::MAP2_RETAINED_DECISIONS);
    settings
}

#[cfg(test)]
fn parity_states_for_test(retention_phase: usize) -> Vec<EpisodeStream> {
    assert!(retention_phase < RETENTION_STRIDE);
    (0..2)
        .map(|_| EpisodeStream {
            retention_phase,
            ..EpisodeStream::default()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_retention_phase_replay_for_test() {
    let mut settings = parity_settings_for_test();
    settings.ppo.environments = 6;
    validate(&settings).expect("six Map2 streams");
    let mut master = PpoRng::new(9971002);
    let before = master.checkpoint();
    let mut restored = PpoRng::from_checkpoint(before.0, before.1).expect("restore");
    for update in [0, 1, 31, 999999] {
        let first = streams_for_collection(&settings, update).expect("phases");
        let replay = streams_for_collection(&settings, update).expect("replay phases");
        assert_eq!(
            first
                .iter()
                .map(|state| state.retention_phase)
                .collect::<Vec<_>>(),
            replay
                .iter()
                .map(|state| state.retention_phase)
                .collect::<Vec<_>>()
        );
        assert_eq!(master.checkpoint(), before);
    }
    assert_eq!(
        actor_stream_rngs(&mut master, 6).expect("master streams"),
        actor_stream_rngs(&mut restored, 6).expect("restored streams")
    );
    for length in [1, 3, 8, 9, 17, ACTOR_DECISIONS] {
        let mut inclusions = vec![0; length];
        for phase in 0..8 {
            let mut state = EpisodeStream {
                retention_phase: phase,
                ..EpisodeStream::default()
            };
            for (decision, included) in inclusions.iter_mut().enumerate() {
                state.decisions = decision;
                *included += usize::from(state.begins_interval());
            }
        }
        assert!(inclusions.iter().all(|count| *count == 1));
    }
    for phase in 0..8 {
        let count = (ACTOR_DECISIONS - phase).div_ceil(RETENTION_STRIDE);
        assert!(count <= RETAINED_PER_EPISODE);
    }
}

#[cfg(test)]
pub(crate) fn assert_retention_boundaries_for_test(_: &PolicyModel) {
    let model = PolicyModel::fresh(9971003).expect("nonzero value model");
    let settings = parity_settings_for_test();
    let mut arena = environments(&settings, 0).expect("worlds").remove(0);
    let choice = sample_policy(&model, &mut PpoRng::new(9971004), &mut arena).expect("choice");
    for terminal in [true, false] {
        let mut state = EpisodeStream {
            retention_phase: 3,
            ..EpisodeStream::default()
        };
        let mut rollout = PpoRollout::new(1, choice.policy()).expect("rollout");
        for decision in 0..8 {
            if state.begins_interval() {
                state.choice = Some(choice.clone());
            }
            let reward = if decision < 3 { 1000.0 } else { 2.0 };
            state
                .append_retained_reward(
                    reward,
                    if decision == 7 { 2 } else { 3 },
                    settings.ppo.gamma_tick,
                )
                .expect("reward");
            state.decisions += 1;
        }
        state.done = true;
        assert!(state.should_flush());
        let bootstrap = model
            .evaluate_batch(&[encode_next_frame(&mut arena).expect("frame")])
            .expect("value")[0]
            .value;
        assert_ne!(bootstrap, 0.0);
        flush(&model, &mut arena, &mut state, 0, terminal, &mut rollout).expect("partial flush");
        let sample = rollout
            .finish(settings.ppo)
            .expect("batch")
            .sample(0)
            .expect("sample")
            .transition;
        assert_eq!(sample.ticks, 14);
        assert_eq!(sample.terminal, terminal);
        assert_eq!(sample.next_value, if terminal { 0.0 } else { bootstrap });
        assert_eq!(sample.reward, 10.0);
    }
    assert_unsampled_short_episode_for_test(&choice);
}

#[cfg(test)]
fn assert_unsampled_short_episode_for_test(choice: &PpoPolicyChoice) {
    let mut state = EpisodeStream {
        retention_phase: 7,
        ..EpisodeStream::default()
    };
    assert!(!state.begins_interval());
    state
        .append_retained_reward(1.0, 3, 1.0)
        .expect("unretained terminal reward");
    state.map2_reward = Map2TrainingReward {
        ticks: 3,
        terminal: 1.0,
        total: 1.0,
        ..Map2TrainingReward::default()
    };
    state.decisions = 1;
    state.done = true;
    assert!(!state.should_flush());
    assert!(state.choice.is_none());
    assert_eq!(state.interval.steps, 0);
    let mut report = PpoSmokeReport::default();
    record_episode(
        0,
        4,
        &state,
        Some(PpoTerminalOutcome::Win),
        &mut report,
        "Teacher",
    )
    .expect("complete outcome");
    assert_eq!(report.terminal_wins, 1);
    assert_eq!(report.map2_reward, state.map2_reward);
    assert_eq!(state.retained, 0);
    let rollout = PpoRollout::new(1, choice.policy()).expect("empty rollout");
    assert_eq!(
        validate_episode_batch(&rollout, &report),
        Err(PpoError::EmptyRollout)
    );
    assert_eq!(
        validate_episode_batch(&rollout, &report)
            .expect_err("no optimizer input")
            .to_string(),
        "PPO rollout is empty"
    );
}

#[cfg(test)]
pub(crate) fn assert_all_retention_phase_labels_for_test(model: &PolicyModel) {
    let settings = parity_settings_for_test();
    let mut arena = environments(&settings, 0).expect("arena").remove(0);
    let choice = sample_policy(model, &mut PpoRng::new(9971000), &mut arena).expect("sample");
    let mut retained_phase_labels = std::collections::BTreeSet::new();
    for update in 0..64 {
        for mut state in streams_for_collection(&settings, update).expect("production streams") {
            for action_index in 0..16 {
                if state.begins_interval() {
                    retained_phase_labels.insert(action_index % 8);
                    state.choice = Some(choice.clone());
                }
                if state.choice.is_some() {
                    state.interval.append(0.0, 3, 1.0).expect("interval");
                }
                state.decisions += 1;
                if state.interval.steps == 8 {
                    state.choice = None;
                    state.interval = DiscountedInterval::default();
                }
            }
        }
    }
    assert_eq!(retained_phase_labels, (0..8).collect());
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
            .map_err(model_error)?[0]
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
    report: &mut PpoSmokeReport,
    opponent: &'static str,
) -> Result<(), PpoError> {
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
    crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default()).emit(
        &format_args!(
            "level=INFO event=map2_episode_reward stream={stream} tick={tick} outcome={label} opponent={opponent} {}",
            state.map2_reward
        ),
    );
    Ok(())
}

#[cfg(test)]
pub(crate) fn assert_discounted_intervals_for_test() {
    let mut interval = DiscountedInterval::default();
    interval.append(2.0, 2, 0.5).expect("first interval");
    interval.append(4.0, 1, 0.5).expect("second interval");
    let outcome = interval.finish(0, 0, true, 99.0);
    assert_eq!(outcome.reward, 3.0);
    assert_eq!(outcome.ticks, 3);
    assert!(outcome.terminal);
    assert_eq!(outcome.next_value, 0.0);
    let mut partial = DiscountedInterval::default();
    partial.append(-1.0, 1, 0.9).expect("last partial");
    let outcome = partial.finish(0, 1, false, 0.25);
    assert!(!outcome.terminal);
    assert_eq!(outcome.next_value, 0.25);
    assert_eq!(outcome.ticks, 1);
    assert_eq!(outcome.reward, -1.0);
    let mut invalid = DiscountedInterval::default();
    assert_eq!(
        invalid
            .append(f64::NAN, 3, 0.9)
            .expect_err("NaN")
            .to_string(),
        "PPO episode reward is non-finite"
    );
}

#[cfg(test)]
pub(crate) fn assert_reset_loses_terminal_credit_for_test() {
    let delayed_terminal_tick = 18_000;
    let maximum_window_end = 1 + training_warmup_decisions(7) as u32 * 3 + 6 + 2048 * 3;
    assert!(maximum_window_end < delayed_terminal_tick);
    let mut reset_terminals = 0;
    let mut continuous_terminals = 0;
    let mut continuous_tick = 1;
    for _ in 0..4 {
        let reset_tick = 1 + 2048 * 3;
        reset_terminals += usize::from(reset_tick >= delayed_terminal_tick);
        let previous = continuous_tick;
        continuous_tick = (continuous_tick + 2048 * 3).min(TICK_CAP);
        continuous_terminals += usize::from(
            previous < delayed_terminal_tick && continuous_tick >= delayed_terminal_tick,
        );
    }
    assert_eq!(reset_terminals, 0);
    assert_eq!(continuous_terminals, 1);
    assert!(TICK_CAP > delayed_terminal_tick);
    let settings = crate::cli::training_settings_for_test(&[
        "--map",
        "2",
        "--environments",
        "2",
        "--gamma-per-tick",
        "1",
        "--complete-episodes=false",
    ])
    .expect("windows");
    let model = PolicyModel::fresh(9911000).expect("model");
    let mut initial = build_training_environments(&settings, 0, settings.ppo, &model, None)
        .expect("first windows");
    let start_tick = initial[0].seats[0].tracker.current().expect("start").tick;
    advance_interval(&mut initial[0], vec![None, None], 3).expect("progress");
    let progressed_tick = initial[0].seats[0]
        .tracker
        .current()
        .expect("progressed")
        .tick;
    let rebuilt =
        build_training_environments(&settings, 8, settings.ppo, &model, None).expect("new windows");
    assert_eq!(
        rebuilt[0].seats[0].tracker.current().expect("rebuilt").tick,
        start_tick
    );
    assert!(progressed_tick > start_tick);
}

#[cfg(test)]
pub(crate) fn assert_rng_and_sample_provenance_for_test(model: &PolicyModel) {
    let settings = parity_settings_for_test();
    let master = PpoRng::new(9911001);
    let mut first = master.clone();
    let (state, draws) = master.checkpoint();
    let mut second = PpoRng::from_checkpoint(state, draws).expect("restored master");
    let source = initial_interval_for_test(model, &settings, &mut first);
    let target = initial_interval_for_test(model, &settings, &mut second);
    assert_eq!(source, target);
    assert_eq!(first, second);
}

#[cfg(test)]
fn initial_interval_for_test(
    model: &PolicyModel,
    settings: &TrainingJobConfig,
    master: &mut PpoRng,
) -> Vec<(crate::StructuredAction, f32, u32)> {
    let mut arenas = environments(settings, 0).expect("arenas");
    let mut random = actor_stream_rngs(master, 2).expect("random streams");
    let mut streams = [EpisodeStream::default(), EpisodeStream::default()];
    let mut rollout =
        PpoRollout::new(2, model.policy_identity().expect("identity")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    let mut expected = Vec::new();
    for step in 0..RETENTION_STRIDE {
        let (choices, spaces) =
            sample_active(model, &mut random, &mut arenas, &[0, 1]).expect("choices");
        if step == 0 {
            expected = choices
                .iter()
                .map(|choice| (choice.action(), choice.log_probability()))
                .collect();
        }
        for (stream, (choice, space)) in choices.into_iter().zip(spaces).enumerate() {
            advance_stream(
                model,
                &mut arenas[stream],
                &mut streams[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut report,
            )
            .expect("act every tick interval");
        }
    }
    assert_eq!(rollout.len(), 2);
    let batch = rollout.finish(settings.ppo).expect("batch");
    (0..2)
        .map(|index| {
            let sample = batch.sample(index).expect("sample").transition;
            assert_eq!((sample.action, sample.old_log_probability), expected[index]);
            assert_eq!(sample.ticks, 24);
            assert!(!sample.terminal);
            (sample.action, sample.old_log_probability, sample.ticks)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_checkpoint_scope_for_test(mut settings: TrainingJobConfig, directory: &Path) {
    let config = settings.ppo;
    let run = training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("run");
    assert!(run.command_line.ends_with(" --complete-episodes"));
    let session = TrainingSession::initialize(
        &settings,
        PolicyDevice::Cpu,
        directory,
        false,
        None,
        config,
        run.clone(),
    )
    .expect("session");
    session
        .save(directory, session.checkpoint_report(None))
        .expect("checkpoint");
    let restored_model = PolicyModel::fresh(9911002).expect("unowned model");
    let restored = restore_training_session(
        &restored_model,
        directory,
        &run,
        config,
        ResumeProvenance::Strict,
    )
    .expect("restore");
    assert_eq!(restored.trainer.config(), config);
    assert_eq!(restored.sampling, session.sampling);
    for stream in 0..settings.ppo.environments {
        assert_eq!(
            retention_phase(settings.seed, session.completed_updates, stream)
                .expect("original phase"),
            retention_phase(settings.seed, restored.completed_updates, stream)
                .expect("resumed phase")
        );
    }
    settings.complete_episodes = false;
    let windows =
        training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("window run");
    assert_eq!(
        TrainingArtifact::load_compatible(directory, &windows).expect_err("mode mismatch"),
        crate::CheckpointError::InvalidManifest("compatibility scope")
    );
}

#[cfg(test)]
pub(crate) fn assert_pipeline_groups_scope_for_test(settings: TrainingJobConfig, directory: &Path) {
    assert_eq!(settings.pipeline_groups, 2);
    let config = settings.ppo;
    let run = training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("grouped run");
    assert!(run.command_line.ends_with(" --pipeline-groups 2"));
    let session = TrainingSession::initialize(
        &settings,
        PolicyDevice::Cpu,
        directory,
        false,
        None,
        config,
        run.clone(),
    )
    .expect("grouped session");
    session
        .save(directory, session.checkpoint_report(None))
        .expect("grouped checkpoint");
    let restored_model = PolicyModel::fresh(9911003).expect("unowned model");
    restore_training_session(
        &restored_model,
        directory,
        &run,
        config,
        ResumeProvenance::Strict,
    )
    .expect("grouped strict restore");
    let mut default = settings;
    default.pipeline_groups = 1;
    let default_run =
        training_checkpoint_run(&default, PolicyDevice::Cpu, config).expect("default run");
    assert!(!default_run.command_line.contains("--pipeline-groups"));
    assert_eq!(
        TrainingArtifact::load_compatible(directory, &default_run).expect_err("group scope"),
        crate::CheckpointError::InvalidManifest("compatibility scope")
    );
}

#[cfg(test)]
pub(crate) fn assert_time_cost_for_test() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 0.250001] {
        assert_eq!(
            validate_time_cost(invalid)
                .expect_err("historical cost bound")
                .to_string(),
            "invalid PPO config field: episode time cost must be finite in [0, 0.25]"
        );
    }
    let fast = elapsed_time_cost(0.25, 0, 300).expect("pregame cost");
    let slow = elapsed_time_cost(0.25, 0, TICK_CAP / 2).expect("long cost");
    let maximum = elapsed_time_cost(0.25, 0, TICK_CAP - 1).expect("maximum elapsed");
    assert!(fast > 0.0);
    assert!(1.0 - fast > 1.0 - slow);
    assert!(-1.0 - fast > -1.0 - slow);
    assert!(1.0 - maximum >= 0.75);
    assert!(-1.0 - maximum <= -1.0);
    assert!(1.0 - maximum > -1.0 - fast);
    let mut elapsed = 0;
    let mut spent = 0.0;
    for _ in 0..ACTOR_DECISIONS {
        let ticks = 3.min(TICK_CAP - 1 - elapsed);
        spent += elapsed_time_cost(0.25, elapsed, ticks).expect("costed interval");
        elapsed += ticks;
        assert!(spent <= 0.25);
    }
    assert_eq!(elapsed, TICK_CAP - 1);
    assert!((spent - maximum).abs() < 1e-12);
    assert_eq!(elapsed_time_cost(0.0, 0, 3).expect("old profile"), 0.0);
    assert_eq!(
        elapsed_time_cost(0.25, TICK_CAP - 2, 3)
            .expect_err("overrun")
            .to_string(),
        "invalid PPO transition: episode time-cost elapsed tick bound"
    );
    assert_eq!(
        elapsed_time_cost(0.25, 0, 0)
            .expect_err("zero span")
            .to_string(),
        "invalid PPO transition: episode time-cost zero ticks"
    );
}

#[cfg(test)]
pub(crate) fn assert_full_mc_for_test() {
    let model = PolicyModel::fresh(9921000).expect("model");
    let mut arena =
        build_environment(9921001, 9921002, MapId(2), 0, 0, OpponentSpec::Teacher).expect("arena");
    let choice = sample_policy(&model, &mut PpoRng::new(9921003), &mut arena).expect("choice");
    let config = PpoConfig {
        environments: 2,
        rollout_decisions: RETAINED_PER_EPISODE,
        minibatch: 512,
        gae_lambda: 1.0,
        gamma_tick: 1.0,
        ..PpoConfig::default()
    };
    let mut rollout = PpoRollout::new(2 * RETAINED_PER_EPISODE, choice.policy()).expect("rollout");
    for decision in 0..RETAINED_PER_EPISODE {
        let count = (ACTOR_DECISIONS - decision * RETENTION_STRIDE).min(RETENTION_STRIDE);
        let terminal = decision + 1 == RETAINED_PER_EPISODE;
        for stream in 0..2 {
            let mut sample = choice.clone();
            sample.value = if decision.is_multiple_of(2) {
                17.0
            } else {
                -31.0
            };
            let sign = if stream == 0 { 1.0 } else { -1.0 };
            let reward = if terminal { sign } else { 0.0 };
            let ticks = count as u32 * 3 - u32::from(terminal);
            rollout
                .push(
                    sample
                        .finish(PpoOutcome {
                            stream,
                            decision: decision as u32,
                            ticks,
                            reward,
                            terminal,
                            next_value: if terminal { 0.0 } else { 99.0 },
                        })
                        .expect("transition"),
                )
                .expect("push");
        }
    }
    let batch = rollout.finish(config).expect("MC batch");
    assert_eq!(batch.len(), 2 * RETAINED_PER_EPISODE);
    for index in 0..batch.len() {
        let sample = batch.sample(index).expect("sample");
        let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
        let actions_remaining = ACTOR_DECISIONS - 1 - index / 2 * RETENTION_STRIDE;
        let exponent = actions_remaining as i32 * config.decision_interval_ticks as i32;
        let expected = sign * f64::from(config.gamma_tick).powi(exponent);
        assert!((f64::from(sample.return_value()) - expected).abs() < 1.0e-5);
        let terminal = index / 2 + 1 == RETAINED_PER_EPISODE;
        assert_eq!(sample.transition.terminal, terminal);
        let actions = (ACTOR_DECISIONS - index / 2 * RETENTION_STRIDE).min(RETENTION_STRIDE);
        assert_eq!(
            sample.transition.ticks,
            actions as u32 * 3 - u32::from(terminal)
        );
    }
    assert_mc_timeout_for_test(choice, config);
}

#[cfg(test)]
fn assert_mc_timeout_for_test(choice: PpoPolicyChoice, config: PpoConfig) {
    let mut truncated = PpoRollout::new(1, choice.policy()).expect("truncated rollout");
    truncated
        .push(
            choice
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 11,
                    next_value: 0.7,
                    reward: 0.0,
                    terminal: false,
                })
                .expect("timeout"),
        )
        .expect("push");
    let sample = truncated
        .finish(config)
        .expect("timeout MC")
        .sample(0)
        .expect("sample");
    assert!(!sample.transition.terminal);
    assert!(
        (f64::from(sample.return_value())
            - f64::from(config.gamma_tick).powi(11) * f64::from(0.7f32))
        .abs()
            < 1e-5
    );
}

#[cfg(test)]
pub(crate) fn assert_ragged_streams_for_test(model: &PolicyModel) {
    for count in [2, 4, 6] {
        let mut settings = parity_settings_for_test();
        settings.ppo.environments = count;
        validate(&settings).expect("bounded paired batch");
        if count == 2 {
            for update in [0, 1, 17] {
                let arenas = environments(&settings, update).expect("paired E2 seed mapping");
                for (side, arena) in arenas.iter().enumerate() {
                    assert_eq!(arena.policy_seat, side);
                    assert_eq!(
                        arena.next_seed,
                        derive_training_seed(settings.seed, update, 0x6172_656e_615f_7365)
                            .wrapping_add(1u64 << 32)
                    );
                }
            }
        }
        let first = ragged_trial_for_test(model, &settings);
        let second = ragged_trial_for_test(model, &settings);
        assert_eq!(first, second);
    }
}

#[cfg(test)]
fn ragged_trial_for_test(model: &PolicyModel, settings: &TrainingJobConfig) -> Vec<(u32, f32)> {
    let count = settings.ppo.environments;
    let mut arenas = environments(settings, 0).expect("arenas");
    let mut random = actor_stream_rngs(&mut PpoRng::new(9921004), count).expect("streams");
    let mut states: Vec<_> = (0..count).map(|_| EpisodeStream::default()).collect();
    let mut rollout =
        PpoRollout::new(count, model.policy_identity().expect("identity")).expect("rollout");
    let mut first_logprobs = vec![0.0; count];
    for step in 0..8 {
        let active: Vec<_> = (0..count).filter(|&stream| !states[stream].done).collect();
        if active.is_empty() {
            break;
        }
        let before = random.clone();
        let (choices, spaces) =
            sample_active(model, &mut random, &mut arenas, &active).expect("active batch");
        for stream in 0..count {
            if states[stream].done {
                assert_eq!(random[stream], before[stream]);
            }
        }
        for ((stream, choice), space) in active.into_iter().zip(choices).zip(spaces) {
            if step == 0 {
                first_logprobs[stream] = choice.log_probability();
            }
            advance_stream(
                model,
                &mut arenas[stream],
                &mut states[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut PpoSmokeReport::default(),
            )
            .expect("step");
            if step + 1 == 3 + stream % 3 {
                states[stream].done = true;
                flush(
                    model,
                    &mut arenas[stream],
                    &mut states[stream],
                    stream,
                    true,
                    &mut rollout,
                )
                .expect("scripted terminal flush");
            }
        }
    }
    assert!(
        states
            .iter()
            .all(|state| state.done && state.choice.is_none())
    );
    assert_eq!(rollout.len(), count);
    let batch = rollout.finish(settings.ppo).expect("batch");
    (0..count)
        .map(|index| {
            let sample = batch.sample(index).expect("sample").transition;
            assert!(sample.terminal);
            assert_eq!(sample.next_value, 0.0);
            assert_eq!(sample.old_log_probability, first_logprobs[sample.stream]);
            (sample.ticks, sample.old_log_probability)
        })
        .collect()
}
