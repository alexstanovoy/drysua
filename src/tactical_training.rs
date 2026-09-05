#![allow(
    clippy::float_arithmetic,
    reason = "bounded neural parameter search and confidence diagnostics"
)]

use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use bota_proto::{MapId, ServerMsg, SlotId, Team};
use sha2::{Digest, Sha256};

use crate::{
    Arena, ArenaConfig, GlobalSummary, ItemReadiness, OrderPersistence, Request, StateTracker,
    TACTICAL_FEATURES, TACTICAL_HIDDEN, TACTICAL_OUTPUT_BIAS_OFFSET, TACTICAL_PARAMETERS,
    TacticalMode, TacticalPolicy, Teacher,
};

const DEVELOPMENT_START: u64 = 9_200_000;
const DEVELOPMENT_END: u64 = 9_300_000;
const MAX_TICKS: u32 = 30_000;
const MAX_POPULATION: usize = 24;
const MAX_GENERATIONS: u32 = 20;
const MAX_PAIRS: usize = 16;
const MAX_WORKERS: usize = 4;
const MAX_EVALUATED_GAMES: usize = 5_120;
const OUTPUT_OFFSET: usize = (TACTICAL_FEATURES + 1) * TACTICAL_HIDDEN;
const DIAGNOSTIC_STRIDE: u32 = 32;
const _: () = assert!(OUTPUT_OFFSET < TACTICAL_PARAMETERS);
const _: () = assert!(MAX_POPULATION * MAX_PAIRS * 2 <= MAX_EVALUATED_GAMES);

/// Exact standalone tactical artifact, not a PPO checkpoint or SafeTensors model.
pub const TACTICAL_SEARCH_POLICY_FILE: &str = "drysua.tactical.bin";

/// Evaluates one complete deployment-cadence game; `None` selects an unchanged Teacher.
pub fn evaluate_tactical_match(
    policy: Option<&TacticalPolicy>,
    settings: TacticalMatchConfig,
) -> Result<TacticalMatchReport, TacticalSearchError> {
    run_match(policy, settings, Instant::now() + Duration::from_secs(30))
}

/// Runs deterministic elitist population search and atomically archives completed comparisons.
/// The output directory must not exist. Selection and confirmation are development data.
pub fn run_tactical_search(
    settings: &TacticalSearchConfig,
    output: &Path,
    initial: Option<&TacticalPolicy>,
) -> Result<TacticalSearchReport, TacticalSearchError> {
    validate_search(settings)?;
    create_output(output)?;
    write_new(
        &output.join("config.json"),
        search_config_json(settings).as_bytes(),
    )?;
    let mut budget = SearchBudget::new(settings.wall_time);
    let selection = TacticalCohort::new(
        settings.selection_seed,
        settings.selection_pairs,
        settings.tick_limit,
    )?;
    let policy = initial.cloned().unwrap_or_default();
    let evaluated = evaluate_batch(
        std::slice::from_ref(&policy),
        selection,
        settings.workers,
        &mut budget,
    )?
    .remove(0);
    let archive = save_tactical_archive(output, "selected-000", &policy, &evaluated)?;
    let mut state = SearchState {
        center: policy.clone(),
        policy,
        selection: evaluated,
        archive,
        completed_generations: 0,
        stopped_for_deadline: false,
    };
    update_best_pointer(output, &state.archive)?;
    for generation in 0..settings.generations {
        match search_generation(settings, output, generation, &mut state, &mut budget) {
            Ok(()) => state.completed_generations += 1,
            Err(TacticalSearchError::Deadline) => {
                state.stopped_for_deadline = true;
                break;
            }
            Err(error) => return Err(error),
        }
    }
    let confirmation = confirm_selection(settings, output, &state.policy, &mut budget)?;
    let report = TacticalSearchReport {
        policy: state.policy,
        selection: state.selection,
        confirmation,
        archive: state.archive,
        completed_generations: state.completed_generations,
        evaluated_games: budget.completed,
        stopped_for_deadline: state.stopped_for_deadline,
    };
    write_new(
        &output.join("summary.json"),
        search_summary_json(&report).as_bytes(),
    )?;
    Ok(report)
}

/// Evaluates a bounded population on a common paired development cohort, without optimization.
pub fn evaluate_tactical_population(
    policies: &[TacticalPolicy],
    cohort: TacticalCohort,
    workers: usize,
    wall_time: Duration,
) -> Result<Vec<TacticalEvaluation>, TacticalSearchError> {
    if wall_time.is_zero() || wall_time > Duration::from_secs(2_700) {
        return Err(invalid(
            "tactical wall time must be positive and at most 2700 seconds",
        ));
    }
    evaluate_batch(policies, cohort, workers, &mut SearchBudget::new(wall_time))
}

/// Initial Teacher-equivalent and constant Fight, Recover, Farm policies for diagnostics.
pub fn tactical_search_founders() -> [TacticalPolicy; 4] {
    TacticalMode::ALL.map(|mode| {
        let mut parameters = *TacticalPolicy::default().parameters();
        if mode != TacticalMode::Teacher {
            parameters[TACTICAL_OUTPUT_BIAS_OFFSET + mode.index()] = 1.0;
        }
        TacticalPolicy::from_parameters(&parameters).expect("bounded founder biases")
    })
}

/// A checked common-random-number cohort: every seed is played from both seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TacticalCohort {
    first_seed: u64,
    pairs: usize,
    tick_limit: u32,
}

impl TacticalCohort {
    pub fn new(
        first_seed: u64,
        pairs: usize,
        tick_limit: u32,
    ) -> Result<Self, TacticalSearchError> {
        if !(1..=MAX_PAIRS).contains(&pairs) {
            return Err(invalid("tactical paired seeds must be in 1..=16"));
        }
        if !(2..=MAX_TICKS).contains(&tick_limit) {
            return Err(invalid("tactical tick limit must be in 2..=30000"));
        }
        validate_seed_range(first_seed, pairs as u64)?;
        Ok(Self {
            first_seed,
            pairs,
            tick_limit,
        })
    }

    pub const fn first_seed(self) -> u64 {
        self.first_seed
    }
    pub const fn pairs(self) -> usize {
        self.pairs
    }
    pub const fn tick_limit(self) -> u32 {
        self.tick_limit
    }

    fn match_settings(self, index: usize) -> TacticalMatchConfig {
        assert!(index < self.pairs * 2);
        assert!(self.pairs <= MAX_PAIRS);
        TacticalMatchConfig {
            seed: self.first_seed + (index / 2) as u64,
            candidate_seat: (index % 2) as u8,
            tick_limit: self.tick_limit,
        }
    }
}

/// Bounded Map1 settings. Only development seeds are accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TacticalMatchConfig {
    pub seed: u64,
    pub candidate_seat: u8,
    pub tick_limit: u32,
}

/// Only a simulator terminal message can produce Win or Loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TacticalMatchOutcome {
    Win,
    Loss,
    Timeout,
}

/// Seat-visible result and deterministic action-path diagnostics, without wall timings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TacticalMatchReport {
    pub settings: TacticalMatchConfig,
    pub outcome: TacticalMatchOutcome,
    pub ticks: u32,
    pub decisions: [u32; 2],
    pub orders: [u32; 2],
    pub rejections: [u32; 2],
    pub order_fingerprints: [u64; 2],
    pub sampled_decisions: u32,
    pub effective_overrides: u32,
    pub final_summary: GlobalSummary,
}

/// Validated terminal-first fitness. Different cohorts cannot be compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TacticalFitness {
    cohort: TacticalCohort,
    pub wins: u32,
    pub losses: u32,
    pub timeouts: u32,
    pub death_margin: i64,
    pub farm_margin: i64,
    pub paired_sweeps: u32,
}

impl TacticalFitness {
    pub fn from_games(
        cohort: TacticalCohort,
        games: &[TacticalMatchReport],
    ) -> Result<Self, TacticalSearchError> {
        if games.len() != cohort.pairs * 2 {
            return Err(message(format!(
                "tactical cohort has {} games; expected {}",
                games.len(),
                cohort.pairs * 2
            )));
        }
        let mut output = Self {
            cohort,
            wins: 0,
            losses: 0,
            timeouts: 0,
            death_margin: 0,
            farm_margin: 0,
            paired_sweeps: 0,
        };
        for (index, game) in games.iter().enumerate() {
            if game.settings != cohort.match_settings(index) {
                return Err(invalid(
                    "tactical cohort games must match each paired seed and both seats in order",
                ));
            }
            if game.rejections != [0, 0] {
                return Err(invalid(
                    "tactical fitness requires zero rejected orders from both seats",
                ));
            }
            if !(2..=cohort.tick_limit).contains(&game.ticks)
                || (game.outcome == TacticalMatchOutcome::Timeout)
                    != (game.ticks == cohort.tick_limit)
            {
                return Err(invalid(
                    "tactical outcome must agree with the deployment tick-limit boundary",
                ));
            }
            match game.outcome {
                TacticalMatchOutcome::Win => output.wins += 1,
                TacticalMatchOutcome::Loss => output.losses += 1,
                TacticalMatchOutcome::Timeout => output.timeouts += 1,
            }
            let summary = game.final_summary;
            output.death_margin += difference(summary.enemy.deaths, summary.allied.deaths, 2);
            output.farm_margin +=
                difference(summary.allied.last_hits, summary.enemy.last_hits, 500);
            output.farm_margin += difference(summary.allied.denies, summary.enemy.denies, 100);
        }
        output.paired_sweeps = games
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|pair| {
                pair.iter()
                    .all(|game| game.outcome == TacticalMatchOutcome::Win)
            })
            .count() as u32;
        assert_eq!(
            output.wins + output.losses + output.timeouts,
            games.len() as u32
        );
        assert!(output.paired_sweeps <= cohort.pairs as u32);
        Ok(output)
    }

    pub fn compare(&self, other: &Self) -> Result<Ordering, TacticalSearchError> {
        if self.cohort != other.cohort {
            return Err(invalid(
                "tactical fitness comparison requires identical seed, side and tick-limit cohorts",
            ));
        }
        Ok(self.rank().cmp(&other.rank()))
    }

    /// Observed development rate, not a release approval or confidence guarantee.
    pub fn win_percent(&self) -> f64 {
        f64::from(self.wins) * 100.0 / (self.cohort.pairs * 2) as f64
    }

    /// Wilson 95% lower limit for sweeping a seed pair; does not treat seats as independent.
    /// Sweep probability is a conservative surrogate for average game win probability.
    /// Adaptive selection invalidates nominal coverage; this diagnostic is not certification.
    pub fn paired_sweep_lower_percent(&self) -> f64 {
        let count = self.cohort.pairs as f64;
        let proportion = f64::from(self.paired_sweeps) / count;
        let square = 1.96_f64.powi(2);
        let center = proportion + square / (2.0 * count);
        let radius = 1.96
            * (proportion * (1.0 - proportion) / count + square / (4.0 * count * count)).sqrt();
        ((center - radius) / (1.0 + square / count) * 100.0).max(0.0)
    }

    fn rank(&self) -> (u32, Reverse<u32>, i64, i64) {
        (
            self.wins,
            Reverse(self.timeouts),
            self.death_margin,
            self.farm_margin,
        )
    }
}

/// Every game of one complete cohort, retained with its aggregate fitness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TacticalEvaluation {
    pub fitness: TacticalFitness,
    pub games: Vec<TacticalMatchReport>,
}

/// Fixed search bounds and disjoint development cohorts. Four workers are the maximum.
#[derive(Clone, Debug)]
pub struct TacticalSearchConfig {
    pub population: usize,
    pub generations: u32,
    pub pairs: usize,
    pub workers: usize,
    pub seed: u64,
    pub selection_seed: u64,
    pub confirmation_seed: u64,
    pub selection_pairs: usize,
    pub tick_limit: u32,
    pub wall_time: Duration,
}

impl Default for TacticalSearchConfig {
    fn default() -> Self {
        Self {
            population: 16,
            generations: 20,
            pairs: 4,
            workers: 4,
            seed: DEVELOPMENT_START,
            selection_seed: DEVELOPMENT_START + 100,
            confirmation_seed: DEVELOPMENT_START + 200,
            selection_pairs: 16,
            tick_limit: MAX_TICKS,
            wall_time: Duration::from_secs(1_200),
        }
    }
}

/// The selected durable policy; a deadline never promotes a partial comparison.
#[derive(Clone, Debug)]
pub struct TacticalSearchReport {
    pub policy: TacticalPolicy,
    pub selection: TacticalEvaluation,
    pub confirmation: Option<TacticalEvaluation>,
    pub archive: PathBuf,
    pub completed_generations: u32,
    pub evaluated_games: usize,
    pub stopped_for_deadline: bool,
}

/// Specific invalid-input, simulator, filesystem, worker, or wall-budget failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TacticalSearchError {
    Invalid(&'static str),
    Message(String),
    Deadline,
}

impl fmt::Display for TacticalSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(text) => formatter.write_str(text),
            Self::Message(text) => formatter.write_str(text),
            Self::Deadline => formatter.write_str("tactical monotonic wall-time budget exhausted"),
        }
    }
}
impl std::error::Error for TacticalSearchError {}

struct Seat {
    teacher: Teacher,
    tracker: StateTracker,
    persistence: OrderPersistence,
    readiness: ItemReadiness,
    sequence: u32,
    decisions: u32,
    sampled_decisions: u32,
    effective_overrides: u32,
    orders_hash: DefaultHasher,
}

fn run_match(
    policy: Option<&TacticalPolicy>,
    settings: TacticalMatchConfig,
    deadline: Instant,
) -> Result<TacticalMatchReport, TacticalSearchError> {
    TacticalCohort::new(settings.seed, 1, settings.tick_limit)?;
    if settings.candidate_seat > 1 {
        return Err(invalid("tactical candidate seat must be zero or one"));
    }
    check_deadline(deadline)?;
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: settings.seed,
    })
    .map_err(failure)?;
    let mut seats = [
        new_seat(0, &start.messages[0])?,
        new_seat(1, &start.messages[1])?,
    ];
    assert_eq!(arena.tick(), 1);
    assert_eq!(seats[0].tracker.metadata(), seats[1].tracker.metadata());
    let pregame = seats[0].tracker.metadata().pregame_ticks;
    let candidate = usize::from(settings.candidate_seat);
    let mut winner = None;
    for _ in 1..settings.tick_limit {
        if arena.tick().is_multiple_of(256) {
            check_deadline(deadline)?;
        }
        let mut requests = [None, None];
        if arena.tick() > pregame && (arena.tick() - pregame - 1).is_multiple_of(3) {
            for (index, seat) in seats.iter_mut().enumerate() {
                requests[index] =
                    seat_request(seat, if index == candidate { policy } else { None })?;
            }
        }
        let step = arena.step(&requests).map_err(failure)?;
        let radiant = observe_messages(&mut seats[0], &step.messages[0])?;
        let dire = observe_messages(&mut seats[1], &step.messages[1])?;
        if radiant != dire {
            return Err(invalid("tactical arena terminal streams disagree"));
        }
        // Deployment exits at the limit Snapshot, before any same-tick MatchOver.
        if arena.tick() < settings.tick_limit {
            winner = radiant;
        }
        if winner.is_some() {
            break;
        }
    }
    match_report(settings, &seats, arena.tick(), winner)
}

fn match_report(
    settings: TacticalMatchConfig,
    seats: &[Seat; 2],
    ticks: u32,
    winner: Option<Team>,
) -> Result<TacticalMatchReport, TacticalSearchError> {
    assert!(ticks <= settings.tick_limit);
    assert!(settings.candidate_seat < 2);
    let seat = &seats[usize::from(settings.candidate_seat)];
    let outcome = match winner {
        Some(winner) if winner == seat.tracker.team() => TacticalMatchOutcome::Win,
        Some(_) => TacticalMatchOutcome::Loss,
        None => TacticalMatchOutcome::Timeout,
    };
    Ok(TacticalMatchReport {
        settings,
        outcome,
        ticks,
        decisions: seats.each_ref().map(|seat| seat.decisions),
        orders: seats.each_ref().map(|seat| seat.sequence),
        rejections: [0; 2],
        order_fingerprints: seats.each_ref().map(|seat| seat.orders_hash.finish()),
        sampled_decisions: seat.sampled_decisions,
        effective_overrides: seat.effective_overrides,
        final_summary: seat
            .tracker
            .latest_summary()
            .ok_or(invalid("tactical final summary missing"))?,
    })
}

fn new_seat(index: u8, messages: &[ServerMsg]) -> Result<Seat, TacticalSearchError> {
    assert!(index < 2);
    assert!(messages.len() <= 4);
    let info = messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info),
            _ => None,
        })
        .ok_or(invalid("tactical initial MatchStart missing"))?;
    let mut seat = Seat {
        teacher: Teacher::new(),
        tracker: StateTracker::new(SlotId(index), info).map_err(failure)?,
        persistence: OrderPersistence::default(),
        readiness: ItemReadiness::new(),
        sequence: 0,
        decisions: 0,
        sampled_decisions: 0,
        effective_overrides: 0,
        orders_hash: DefaultHasher::new(),
    };
    if observe_messages(&mut seat, messages)?.is_some() {
        return Err(invalid("tactical match ended before first decision"));
    }
    Ok(seat)
}

fn seat_request(
    seat: &mut Seat,
    policy: Option<&TacticalPolicy>,
) -> Result<Option<Request>, TacticalSearchError> {
    assert!(seat.decisions < MAX_TICKS.div_ceil(3));
    assert!(seat.sequence <= seat.decisions);
    let counterfactual = if policy.is_some() && seat.decisions.is_multiple_of(DIAGNOSTIC_STRIDE) {
        Some(
            seat.teacher
                .clone()
                .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                .map_err(failure)?
                .0,
        )
    } else {
        None
    };
    let (action, space) = match policy {
        Some(policy) => {
            seat.teacher
                .decide_tactical(&seat.tracker, &seat.persistence, &seat.readiness, policy)
        }
        None => seat
            .teacher
            .decide(&seat.tracker, &seat.persistence, &seat.readiness),
    }
    .map_err(failure)?;
    seat.decisions += 1;
    if let Some(counterfactual) = counterfactual {
        seat.sampled_decisions += 1;
        seat.effective_overrides += u32::from(counterfactual != action);
    }
    let decoded = space.decode(action).map_err(failure)?;
    let Some(issued) = seat.persistence.should_send(decoded) else {
        return Ok(None);
    };
    seat.sequence += 1;
    seat.persistence
        .record_sent(seat.sequence, issued)
        .map_err(failure)?;
    seat.readiness.note_sent(seat.sequence, issued, &space);
    seat.teacher.note_sent(seat.sequence, issued, space.tick());
    (space.tick(), issued.unit, issued.order).hash(&mut seat.orders_hash);
    Ok(Some(Request {
        seq: seat.sequence,
        unit: issued.unit,
        order: issued.order,
    }))
}

fn observe_messages(
    seat: &mut Seat,
    messages: &[ServerMsg],
) -> Result<Option<Team>, TacticalSearchError> {
    assert!(!messages.is_empty());
    assert!(messages.len() <= 4);
    let mut winner = None;
    for message in messages {
        match message {
            ServerMsg::Snapshot { view } => {
                let previous = seat.tracker.own_hero().map(|hero| hero.id);
                seat.tracker.observe_snapshot(view).map_err(failure)?;
                if previous != seat.tracker.own_hero().map(|hero| hero.id) {
                    seat.persistence.clear_body_for(None);
                }
            }
            ServerMsg::Events { tick, events } => seat
                .tracker
                .observe_events(*tick, events)
                .map_err(failure)?,
            ServerMsg::OrderRejected { seq, reason } => {
                seat.persistence.observe_rejection(*seq);
                seat.readiness.note_rejected(*seq);
                seat.teacher.note_rejected(*seq);
                return Err(message_error(seat, *seq, *reason));
            }
            ServerMsg::MatchOver {
                winner: team,
                stats,
            } => {
                if stats.slots.len() != 2
                    || stats.duration != seat.tracker.current().map_or(0, |view| view.tick)
                {
                    return Err(invalid(
                        "tactical terminal stats do not match the seat stream",
                    ));
                }
                winner = Some(*team);
            }
            ServerMsg::MatchStart { .. } => {}
            _ => return Err(invalid("unexpected message in builtin tactical arena")),
        }
    }
    Ok(winner)
}

fn message_error(
    seat: &Seat,
    sequence: u32,
    reason: bota_proto::RejectReason,
) -> TacticalSearchError {
    message(format!(
        "tactical arena rejected team={:?} sequence={sequence} reason={reason:?}",
        seat.tracker.team()
    ))
}

struct SearchBudget {
    deadline: Instant,
    scheduled: usize,
    completed: usize,
}
impl SearchBudget {
    fn new(wall_time: Duration) -> Self {
        Self {
            deadline: Instant::now() + wall_time,
            scheduled: 0,
            completed: 0,
        }
    }
    fn reserve(&mut self, count: usize) -> Result<(), TacticalSearchError> {
        check_deadline(self.deadline)?;
        if count > MAX_EVALUATED_GAMES.saturating_sub(self.scheduled) {
            return Err(invalid("tactical maximum scheduled game count exceeded"));
        }
        self.scheduled += count;
        assert!(self.completed <= self.scheduled);
        assert!(self.scheduled <= MAX_EVALUATED_GAMES);
        Ok(())
    }
}

fn evaluate_batch(
    policies: &[TacticalPolicy],
    cohort: TacticalCohort,
    workers: usize,
    budget: &mut SearchBudget,
) -> Result<Vec<TacticalEvaluation>, TacticalSearchError> {
    if policies.is_empty() || policies.len() > MAX_POPULATION {
        return Err(invalid("tactical evaluation population must be in 1..=24"));
    }
    validate_workers(workers)?;
    let count = policies.len() * cohort.pairs * 2;
    budget.reserve(count)?;
    let completed = AtomicUsize::new(0);
    let result = parallel_games(policies, cohort, workers, budget.deadline, &completed);
    budget.completed += completed.load(AtomicOrdering::Relaxed);
    let games = result?;
    assert_eq!(games.len(), count);
    assert!(budget.completed <= budget.scheduled);
    games
        .chunks(cohort.pairs * 2)
        .map(|games| evaluation(cohort, games.to_vec()))
        .collect()
}

fn parallel_games(
    policies: &[TacticalPolicy],
    cohort: TacticalCohort,
    workers: usize,
    deadline: Instant,
    completed: &AtomicUsize,
) -> Result<Vec<TacticalMatchReport>, TacticalSearchError> {
    let count = policies.len() * cohort.pairs * 2;
    let mut ordered = vec![None; count];
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                let mut results = Vec::with_capacity(count.div_ceil(workers));
                for index in (worker..count).step_by(workers) {
                    let policy = &policies[index / (cohort.pairs * 2)];
                    let settings = cohort.match_settings(index % (cohort.pairs * 2));
                    let game = run_match(Some(policy), settings, deadline)?;
                    completed.fetch_add(1, AtomicOrdering::Relaxed);
                    results.push((index, game));
                }
                Ok::<_, TacticalSearchError>(results)
            }));
        }
        let mut error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(results)) => {
                    for (index, game) in results {
                        ordered[index] = Some(game);
                    }
                }
                Ok(Err(failure)) => {
                    if error.is_none() {
                        error = Some(failure);
                    }
                }
                Err(_) => error = Some(invalid("tactical match worker panicked")),
            }
        }
        if let Some(error) = error {
            return Err(error);
        }
        ordered
            .into_iter()
            .map(|game| game.ok_or(invalid("tactical worker result missing")))
            .collect()
    })
}

fn evaluation(
    cohort: TacticalCohort,
    games: Vec<TacticalMatchReport>,
) -> Result<TacticalEvaluation, TacticalSearchError> {
    Ok(TacticalEvaluation {
        fitness: TacticalFitness::from_games(cohort, &games)?,
        games,
    })
}

struct SearchState {
    center: TacticalPolicy,
    policy: TacticalPolicy,
    selection: TacticalEvaluation,
    archive: PathBuf,
    completed_generations: u32,
    stopped_for_deadline: bool,
}

fn search_generation(
    settings: &TacticalSearchConfig,
    output: &Path,
    generation: u32,
    state: &mut SearchState,
    budget: &mut SearchBudget,
) -> Result<(), TacticalSearchError> {
    let started = Instant::now();
    let cohort = TacticalCohort::new(
        settings.seed + u64::from(generation) * settings.pairs as u64,
        settings.pairs,
        settings.tick_limit,
    )?;
    let policies = population(
        &state.center,
        settings.population,
        generation,
        settings.seed,
    )?;
    let evaluations = evaluate_batch(&policies, cohort, settings.workers, budget)?;
    let mut ranked = (0..policies.len()).collect::<Vec<_>>();
    ranked.sort_by_key(|index| (Reverse(evaluations[*index].fitness.rank()), *index));
    let best = ranked[0];
    state.center = policies[best].clone();
    let label = format!("generation-{:03}", generation + 1);
    save_tactical_archive(output, &label, &state.center, &evaluations[best])?;
    write_generation(output, &label, &policies, &evaluations)?;
    print_evaluation(&label, &evaluations[best], started.elapsed());
    if generation.is_multiple_of(2) || generation + 1 == settings.generations {
        select_champion(
            settings,
            output,
            generation,
            state,
            [&policies[ranked[0]], &policies[ranked[1]]],
            budget,
        )?;
    }
    Ok(())
}

fn select_champion(
    settings: &TacticalSearchConfig,
    output: &Path,
    generation: u32,
    state: &mut SearchState,
    leaders: [&TacticalPolicy; 2],
    budget: &mut SearchBudget,
) -> Result<(), TacticalSearchError> {
    let mut policies = Vec::with_capacity(2);
    for policy in leaders {
        if policy != &state.policy && !policies.contains(policy) {
            policies.push(policy.clone());
        }
    }
    if policies.is_empty() {
        return Ok(());
    }
    let cohort = TacticalCohort::new(
        settings.selection_seed,
        settings.selection_pairs,
        settings.tick_limit,
    )?;
    let evaluations = evaluate_batch(&policies, cohort, settings.workers, budget)?;
    for (index, (policy, evaluated)) in policies.into_iter().zip(evaluations).enumerate() {
        let label = format!("selection-{:03}-{index}", generation + 1);
        let archive = save_tactical_archive(output, &label, &policy, &evaluated)?;
        print_evaluation(&label, &evaluated, Duration::ZERO);
        if evaluated.fitness.compare(&state.selection.fitness)? == Ordering::Greater {
            state.policy = policy;
            state.selection = evaluated;
            state.archive = archive;
            update_best_pointer(output, &state.archive)?;
        }
    }
    Ok(())
}

fn confirm_selection(
    settings: &TacticalSearchConfig,
    output: &Path,
    policy: &TacticalPolicy,
    budget: &mut SearchBudget,
) -> Result<Option<TacticalEvaluation>, TacticalSearchError> {
    let cohort = TacticalCohort::new(
        settings.confirmation_seed,
        settings.selection_pairs,
        settings.tick_limit,
    )?;
    let result = evaluate_batch(
        std::slice::from_ref(policy),
        cohort,
        settings.workers,
        budget,
    );
    match result {
        Ok(mut evaluations) => {
            let evaluated = evaluations.remove(0);
            save_tactical_archive(output, "confirmation", policy, &evaluated)?;
            print_evaluation("confirmation", &evaluated, Duration::ZERO);
            Ok(Some(evaluated))
        }
        Err(TacticalSearchError::Deadline) => Ok(None),
        Err(error) => Err(error),
    }
}

fn population(
    parent: &TacticalPolicy,
    count: usize,
    generation: u32,
    seed: u64,
) -> Result<Vec<TacticalPolicy>, TacticalSearchError> {
    assert!((4..=MAX_POPULATION).contains(&count));
    assert!(generation < MAX_GENERATIONS);
    let mut output = Vec::with_capacity(count);
    output.push(parent.clone());
    output.push(TacticalPolicy::default());
    if generation == 0 {
        output.extend(
            tactical_search_founders()
                .into_iter()
                .skip(1)
                .take(count - 2),
        );
    }
    let scales = [0.1, 0.2, 0.35, 0.5];
    for pair in 0..count.div_ceil(2) {
        if output.len() == count {
            break;
        }
        let mutation_seed = seed ^ (u64::from(generation) << 32) ^ (pair as u64 * 0x9e37_79b9);
        let noise = TacticalPolicy::from_parameters(&[0.0; TACTICAL_PARAMETERS])
            .map_err(failure)?
            .mutated(mutation_seed, scales[pair % scales.len()])
            .map_err(failure)?;
        for sign in [1.0, -1.0] {
            if output.len() == count {
                break;
            }
            let mut parameters = *parent.parameters();
            for (index, parameter) in parameters.iter_mut().enumerate() {
                let scale = if index >= OUTPUT_OFFSET {
                    1.0
                } else if generation < 4 {
                    0.0
                } else {
                    0.2
                };
                *parameter =
                    (*parameter + noise.parameters()[index] * sign * scale).clamp(-4.0, 4.0);
            }
            output.push(TacticalPolicy::from_parameters(&parameters).map_err(failure)?);
        }
    }
    assert_eq!(output.len(), count);
    Ok(output)
}

fn validate_search(settings: &TacticalSearchConfig) -> Result<(), TacticalSearchError> {
    if !(4..=MAX_POPULATION).contains(&settings.population) {
        return Err(invalid("tactical search population must be in 4..=24"));
    }
    if !(1..=MAX_GENERATIONS).contains(&settings.generations) {
        return Err(invalid("tactical generations must be in 1..=20"));
    }
    validate_workers(settings.workers)?;
    if settings.wall_time.is_zero() || settings.wall_time > Duration::from_secs(2_700) {
        return Err(invalid(
            "tactical wall time must be positive and at most 2700 seconds",
        ));
    }
    TacticalCohort::new(settings.seed, settings.pairs, settings.tick_limit)?;
    TacticalCohort::new(
        settings.selection_seed,
        settings.selection_pairs,
        settings.tick_limit,
    )?;
    TacticalCohort::new(
        settings.confirmation_seed,
        settings.selection_pairs,
        settings.tick_limit,
    )?;
    let training_count = u64::from(settings.generations) * settings.pairs as u64;
    validate_seed_range(settings.seed, training_count)?;
    let ranges = [
        (settings.seed, settings.seed + training_count),
        (
            settings.selection_seed,
            settings.selection_seed + settings.selection_pairs as u64,
        ),
        (
            settings.confirmation_seed,
            settings.confirmation_seed + settings.selection_pairs as u64,
        ),
    ];
    for index in 0..ranges.len() {
        for other in &ranges[index + 1..] {
            if ranges[index].0 < other.1 && other.0 < ranges[index].1 {
                return Err(invalid(
                    "tactical training, selection and confirmation seed ranges must be disjoint",
                ));
            }
        }
    }
    let training = settings.population * settings.generations as usize * settings.pairs * 2;
    let selection_rounds = (settings.generations as usize).div_ceil(2) + 1;
    let maximum = training + (selection_rounds * 2 + 2) * settings.selection_pairs * 2;
    if maximum > MAX_EVALUATED_GAMES {
        return Err(invalid(
            "tactical configuration exceeds 5120 scheduled games",
        ));
    }
    Ok(())
}

fn validate_workers(workers: usize) -> Result<(), TacticalSearchError> {
    if !(1..=MAX_WORKERS).contains(&workers) {
        return Err(invalid("tactical workers must be in 1..=4"));
    }
    Ok(())
}

fn validate_seed_range(seed: u64, count: u64) -> Result<(), TacticalSearchError> {
    if seed < DEVELOPMENT_START
        || seed
            .checked_add(count)
            .is_none_or(|end| end > DEVELOPMENT_END)
    {
        return Err(invalid(
            "tactical seeds must remain in development namespace 9200000..9300000",
        ));
    }
    Ok(())
}

/// Commits policy bytes and a complete seed-bound evaluation as an immutable directory.
pub fn save_tactical_archive(
    root: &Path,
    label: &str,
    policy: &TacticalPolicy,
    evaluated: &TacticalEvaluation,
) -> Result<PathBuf, TacticalSearchError> {
    validate_directory(root)?;
    if label.is_empty()
        || label.len() > 64
        || !label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(invalid(
            "tactical archive label must contain 1..64 ASCII letters, digits or hyphens",
        ));
    }
    if TacticalFitness::from_games(evaluated.fitness.cohort, &evaluated.games)? != evaluated.fitness
    {
        return Err(invalid("tactical archive fitness does not match its games"));
    }
    let target = root.join(label);
    if target.try_exists().map_err(failure)? {
        return Err(message(format!("tactical archive already exists: {label}")));
    }
    let temporary = root.join(format!(".{label}-pending"));
    std::fs::create_dir(&temporary).map_err(failure)?;
    let bytes = policy.to_bytes();
    assert_eq!(
        TacticalPolicy::from_bytes(&bytes).map_err(failure)?,
        *policy
    );
    write_new(&temporary.join(TACTICAL_SEARCH_POLICY_FILE), &bytes)?;
    write_new(
        &temporary.join("report.json"),
        evaluation_json(policy, evaluated).as_bytes(),
    )?;
    sync_directory(&temporary)?;
    std::fs::rename(&temporary, &target).map_err(failure)?;
    sync_directory(root)?;
    Ok(target)
}

fn create_output(output: &Path) -> Result<(), TacticalSearchError> {
    let parent = output.parent().ok_or(invalid(
        "tactical output requires an existing parent directory",
    ))?;
    validate_directory(parent)?;
    std::fs::create_dir(output).map_err(|error| {
        message(format!(
            "create tactical output {}: {error}",
            output.display()
        ))
    })?;
    validate_directory(output)
}

fn validate_directory(path: &Path) -> Result<(), TacticalSearchError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        message(format!(
            "inspect tactical directory {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid(
            "tactical artifact directory must be a real non-symlink directory",
        ));
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), TacticalSearchError> {
    assert!(!bytes.is_empty());
    assert!(bytes.len() <= 1_048_576);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|error| {
        message(format!(
            "create tactical artifact {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            message(format!(
                "commit tactical artifact {}: {error}",
                path.display()
            ))
        })
}

fn sync_directory(path: &Path) -> Result<(), TacticalSearchError> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(failure)?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn update_best_pointer(root: &Path, archive: &Path) -> Result<(), TacticalSearchError> {
    let label = archive
        .file_name()
        .and_then(|label| label.to_str())
        .ok_or(invalid("tactical archive basename missing"))?;
    let temporary = root.join(".best-pending");
    write_new(&temporary, format!("{label}\n").as_bytes())?;
    std::fs::rename(temporary, root.join("best.txt")).map_err(failure)?;
    sync_directory(root)
}

fn policy_hash(policy: &TacticalPolicy) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(policy.to_bytes()) {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 15)]));
    }
    assert_eq!(output.len(), 64);
    output
}

fn evaluation_json(policy: &TacticalPolicy, evaluated: &TacticalEvaluation) -> String {
    let fitness = &evaluated.fitness;
    let mut output = format!(
        "{{\"schema_version\":1,\"policy_sha256\":\"{}\",\"policy_file\":\"{}\",\"opponent\":\"Teacher\",\"map\":1,\"development_only\":true,\"first_seed\":{},\"pairs\":{},\"tick_limit\":{},\"wins\":{},\"losses\":{},\"timeouts\":{},\"death_margin\":{},\"farm_margin\":{},\"win_percent\":{:.6},\"paired_sweeps\":{},\"paired_sweep_wilson95_lower_percent\":{:.6},\"games\":[",
        policy_hash(policy),
        TACTICAL_SEARCH_POLICY_FILE,
        fitness.cohort.first_seed,
        fitness.cohort.pairs,
        fitness.cohort.tick_limit,
        fitness.wins,
        fitness.losses,
        fitness.timeouts,
        fitness.death_margin,
        fitness.farm_margin,
        fitness.win_percent(),
        fitness.paired_sweeps,
        fitness.paired_sweep_lower_percent()
    );
    for (index, game) in evaluated.games.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&format!("{{\"seed\":{},\"candidate_seat\":{},\"outcome\":\"{:?}\",\"ticks\":{},\"decisions\":{:?},\"orders\":{:?},\"rejections\":{:?},\"order_fingerprints\":{:?},\"sampled_decisions\":{},\"effective_overrides\":{},\"kills\":{},\"deaths\":{},\"enemy_deaths\":{},\"last_hits\":{},\"denies\":{}}}",
            game.settings.seed, game.settings.candidate_seat, game.outcome, game.ticks,
            game.decisions, game.orders, game.rejections, game.order_fingerprints,
            game.sampled_decisions, game.effective_overrides, game.final_summary.allied.kills,
            game.final_summary.allied.deaths, game.final_summary.enemy.deaths,
            game.final_summary.allied.last_hits, game.final_summary.allied.denies));
    }
    output.push_str("]}\n");
    output
}

fn write_generation(
    root: &Path,
    label: &str,
    policies: &[TacticalPolicy],
    evaluations: &[TacticalEvaluation],
) -> Result<(), TacticalSearchError> {
    assert_eq!(policies.len(), evaluations.len());
    assert!(policies.len() <= MAX_POPULATION);
    let mut text = String::from("[\n");
    for (index, (policy, evaluated)) in policies.iter().zip(evaluations).enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push_str(&evaluation_json(policy, evaluated));
    }
    text.push_str("]\n");
    write_new(
        &root.join(format!("{label}-population.json")),
        text.as_bytes(),
    )
}

fn search_config_json(settings: &TacticalSearchConfig) -> String {
    format!(
        "{{\"schema_version\":1,\"population\":{},\"generations\":{},\"pairs\":{},\"workers\":{},\"seed\":{},\"selection_seed\":{},\"confirmation_seed\":{},\"selection_pairs\":{},\"tick_limit\":{},\"wall_seconds\":{},\"maximum_games\":{MAX_EVALUATED_GAMES},\"mutation_scales\":[0.1,0.2,0.35,0.5],\"head_only_generations\":4,\"diagnostic_stride\":{DIAGNOSTIC_STRIDE}}}\n",
        settings.population,
        settings.generations,
        settings.pairs,
        settings.workers,
        settings.seed,
        settings.selection_seed,
        settings.confirmation_seed,
        settings.selection_pairs,
        settings.tick_limit,
        settings.wall_time.as_secs()
    )
}

fn search_summary_json(report: &TacticalSearchReport) -> String {
    format!(
        "{{\"schema_version\":1,\"completed_generations\":{},\"evaluated_games\":{},\"stopped_for_deadline\":{},\"policy_sha256\":\"{}\",\"selection_wins\":{},\"selection_games\":{},\"confirmation_wins\":{},\"confirmation_complete\":{}}}\n",
        report.completed_generations,
        report.evaluated_games,
        report.stopped_for_deadline,
        policy_hash(&report.policy),
        report.selection.fitness.wins,
        report.selection.games.len(),
        report
            .confirmation
            .as_ref()
            .map_or(0, |evaluation| evaluation.fitness.wins),
        report.confirmation.is_some()
    )
}

fn print_evaluation(label: &str, evaluated: &TacticalEvaluation, elapsed: Duration) {
    let fitness = &evaluated.fitness;
    let sampled: u32 = evaluated
        .games
        .iter()
        .map(|game| game.sampled_decisions)
        .sum();
    let overrides: u32 = evaluated
        .games
        .iter()
        .map(|game| game.effective_overrides)
        .sum();
    println!(
        "phase={label} seed={} wins={} losses={} timeouts={} games={} death_margin={} farm_margin={} sampled={} overrides={} elapsed_seconds={:.3}",
        fitness.cohort.first_seed,
        fitness.wins,
        fitness.losses,
        fitness.timeouts,
        evaluated.games.len(),
        fitness.death_margin,
        fitness.farm_margin,
        sampled,
        overrides,
        elapsed.as_secs_f64()
    );
}

fn difference(positive: u64, negative: u64, bound: u64) -> i64 {
    assert!(bound <= 500);
    assert!(bound > 0);
    positive.min(bound) as i64 - negative.min(bound) as i64
}
fn check_deadline(deadline: Instant) -> Result<(), TacticalSearchError> {
    if Instant::now() >= deadline {
        return Err(TacticalSearchError::Deadline);
    }
    Ok(())
}
const fn invalid(text: &'static str) -> TacticalSearchError {
    TacticalSearchError::Invalid(text)
}
fn message(text: String) -> TacticalSearchError {
    TacticalSearchError::Message(text)
}
fn failure(error: impl fmt::Display) -> TacticalSearchError {
    message(error.to_string())
}

#[cfg(test)]
#[path = "tests/tactical_training.rs"]
mod tests;
