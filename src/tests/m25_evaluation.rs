//! Bounded greedy evaluation of one M25 side-actor learner.
//!
//! Every matchup plays complete Map2 episodes at the production decision
//! cadence and production tick cap. Greedy means deterministic legal argmax:
//! the learner and a frozen neural opponent both use `PolicyModel::choose_batch`
//! and the scripted Teacher keeps its native decision path. Nothing here
//! samples, runs an optimizer, or writes a checkpoint.
//!
//! One evaluation is split across disjoint bounded jobs. Each job prints a
//! single JSON line so an aggregator can sum offsets without replaying a game.
//!
//! The CUDA-only test drives every helper below. A `side-actors` build without
//! the `cuda` feature has no caller, so dead code is allowed for that
//! configuration only, keeping the CPU validation tests in every build.
#![cfg_attr(
    not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))),
    allow(dead_code)
)]
// One bounded JSON line needs the mean episode length as a decimal.
#![allow(clippy::float_arithmetic)]

use super::{
    OpponentSpec, TrainingEnvironment, advance_interval, build_environment, derive_training_seed,
    neural_policy_request_in_space, prepare_neural_seat_policy_sample, prepare_policy_sample,
    reject_production_rejection, requests_with_candidate, terminal_outcome, text_error,
};
use crate::{
    MAP2_ACTOR_DECISIONS, MAP2_DECISION_INTERVAL_TICKS, MAP2_TICK_CAP, PolicyDevice, PolicyModel,
    PpoError, PpoTerminalOutcome, Request, TrainingArtifact,
};
use bota_proto::MapId;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const EVAL_SCHEMA: &str = "drysua-m25-greedy-eval/v1";
/// Documented seed base for the exclusive bounded evaluation jobs.
const DEFAULT_SEED: u64 = 2026092905;
/// Total paired worlds per matchup; a job covers a disjoint even sub-range.
const MAX_GAMES: usize = 40;
const RUNTIME_FILE: &str = "drysua.weights.safetensors";
const MAX_RUNTIME_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 4096;
const PROBE_PREFIX: &str = "DRYSUA_PROBE_";

#[derive(Clone, Debug, PartialEq, Eq)]
enum EvalOpponent {
    Teacher,
    Weights(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EvalConfig {
    learner: PathBuf,
    opponent: EvalOpponent,
    games: usize,
    offset: usize,
    seed: u64,
}

impl EvalConfig {
    fn parse(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let learner = required_path(&lookup, "DRYSUA_EVAL_LEARNER")?;
        let kind = lookup("DRYSUA_EVAL_OPPONENT")
            .ok_or("DRYSUA_EVAL_OPPONENT must be teacher or weights")?;
        let directory = lookup("DRYSUA_EVAL_OPPONENT_DIR").filter(|value| !value.trim().is_empty());
        let opponent = match kind.trim() {
            "teacher" => {
                if directory.is_some() {
                    return Err("DRYSUA_EVAL_OPPONENT_DIR forbids the teacher opponent".to_owned());
                }
                EvalOpponent::Teacher
            }
            "weights" => EvalOpponent::Weights(PathBuf::from(
                directory.ok_or("DRYSUA_EVAL_OPPONENT_DIR must be set for the weights opponent")?,
            )),
            other => {
                return Err(format!(
                    "DRYSUA_EVAL_OPPONENT must be teacher or weights, got {other:?}"
                ));
            }
        };
        let games = bounded_usize(&lookup, "DRYSUA_EVAL_GAMES", 1, MAX_GAMES)?;
        if !games.is_multiple_of(2) {
            return Err("DRYSUA_EVAL_GAMES must be even for balanced sides".to_owned());
        }
        let offset = bounded_usize(&lookup, "DRYSUA_EVAL_OFFSET", 0, MAX_GAMES - 1)?;
        if !offset.is_multiple_of(2) {
            return Err("DRYSUA_EVAL_OFFSET must be even so sides stay balanced".to_owned());
        }
        if offset + games > MAX_GAMES {
            return Err(format!(
                "DRYSUA_EVAL_OFFSET + DRYSUA_EVAL_GAMES must stay within {MAX_GAMES}"
            ));
        }
        let seed = match lookup("DRYSUA_EVAL_SEED") {
            None => DEFAULT_SEED,
            Some(value) => value
                .trim()
                .parse::<u64>()
                .map_err(|_| "DRYSUA_EVAL_SEED must be an unsigned integer".to_owned())?,
        };
        if seed == 0 {
            return Err("DRYSUA_EVAL_SEED must be positive".to_owned());
        }
        Ok(Self {
            learner,
            opponent,
            games,
            offset,
            seed,
        })
    }

    fn kind(&self) -> &'static str {
        match self.opponent {
            EvalOpponent::Teacher => "teacher",
            EvalOpponent::Weights(_) => "weights",
        }
    }
}

fn required_path(lookup: &impl Fn(&str) -> Option<String>, name: &str) -> Result<PathBuf, String> {
    let value = lookup(name)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} must be set"))?;
    Ok(PathBuf::from(value))
}

fn bounded_usize(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
    minimum: usize,
    maximum: usize,
) -> Result<usize, String> {
    let value = lookup(name)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} must be set"))?;
    let parsed = value
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("{name} must be an unsigned integer"))?;
    if parsed < minimum || parsed > maximum {
        return Err(format!("{name} must be within {minimum}..={maximum}"));
    }
    Ok(parsed)
}

/// The first probe variable that is still set, if any.
///
/// A probe run must never be mistaken for an evaluation, so an inherited
/// `DRYSUA_PROBE_*` variable fails the run closed before any device work.
fn probe_conflict(names: impl Iterator<Item = String>) -> Option<String> {
    names
        .into_iter()
        .find(|name| name.starts_with(PROBE_PREFIX))
}

/// One paired world: both seats of a pair share arena and opponent seeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GamePlan {
    pair: u64,
    seat: usize,
    arena_seed: u64,
    opponent_seed: u64,
}

fn game_plan(config: &EvalConfig) -> Vec<GamePlan> {
    (0..config.games)
        .map(|index| {
            let global = (config.offset + index) as u64;
            let pair = global / 2;
            GamePlan {
                pair,
                seat: (global % 2) as usize,
                arena_seed: derive_training_seed(
                    config.seed,
                    pair,
                    crate::randomization::ARENA_DOMAIN,
                ),
                opponent_seed: derive_training_seed(
                    config.seed,
                    pair,
                    crate::randomization::OPPONENT_DOMAIN,
                ),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GameOutcome {
    seat: usize,
    outcome: PpoTerminalOutcome,
    ticks: u32,
    cap_end: bool,
}

fn load_runtime_model(directory: &Path, device: PolicyDevice) -> Result<PolicyModel, PpoError> {
    let model = PolicyModel::fresh_on(0, device).map_err(text_error)?;
    TrainingArtifact::load_runtime_weights(&model, directory).map_err(text_error)?;
    Ok(model)
}

/// Hashes exactly the runtime tensor the loader consumed, bounded to one file.
fn runtime_sha256(directory: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let path = directory.join(RUNTIME_FILE);
    let file =
        std::fs::File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.len() > MAX_RUNTIME_BYTES {
        return Err(format!("{} exceeds the runtime file bound", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(MAX_RUNTIME_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_RUNTIME_BYTES {
        return Err(format!("{} exceeds the runtime file bound", path.display()));
    }
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Greedy requests for the exact two-seat Map2 world: the learner candidate
/// stays on its own seat and the batched greedy opponent fills the other.
fn greedy_requests(
    environment: &mut TrainingEnvironment,
    mut learner_request: Option<Request>,
    mut opponent_request: Option<Request>,
) -> Result<Vec<Option<Request>>, PpoError> {
    if environment.seats.len() != 2 {
        return Err(PpoError::InvalidTransition("m25 evaluation seat count"));
    }
    let policy_seat = environment.policy_seat;
    let mut requests = Vec::with_capacity(environment.seats.len());
    for index in 0..environment.seats.len() {
        requests.push(if index == policy_seat {
            learner_request.take()
        } else {
            opponent_request.take()
        });
    }
    Ok(requests)
}

fn play_greedy(
    learner: &PolicyModel,
    opponent: Option<&PolicyModel>,
    worlds: &mut [TrainingEnvironment],
) -> Result<Vec<GameOutcome>, PpoError> {
    let mut active: Vec<usize> = (0..worlds.len()).collect();
    let mut outcomes: Vec<Option<GameOutcome>> = vec![None; worlds.len()];
    let mut steps = 0usize;
    while !active.is_empty() {
        if steps >= MAP2_ACTOR_DECISIONS {
            return Err(PpoError::InvalidTransition("m25 evaluation decision bound"));
        }
        steps += 1;
        let mut learner_frames = Vec::with_capacity(active.len());
        let mut learner_spaces = Vec::with_capacity(active.len());
        for &index in &active {
            let (frame, space) = prepare_policy_sample(&mut worlds[index])?;
            learner_frames.push(frame);
            learner_spaces.push(space);
        }
        let learner_choices = learner
            .choose_batch(&learner_frames, &learner_spaces)
            .map_err(text_error)?;
        let mut opponent_frames = Vec::new();
        let mut opponent_spaces = Vec::new();
        let opponent_choices = match opponent {
            Some(model) => {
                for &index in &active {
                    let opposite = 1 - worlds[index].policy_seat;
                    let (frame, space) =
                        prepare_neural_seat_policy_sample(&mut worlds[index].seats[opposite])?;
                    opponent_frames.push(frame);
                    opponent_spaces.push(space);
                }
                Some(
                    model
                        .choose_batch(&opponent_frames, &opponent_spaces)
                        .map_err(text_error)?,
                )
            }
            None => None,
        };
        for (position, &index) in active.iter().enumerate() {
            let environment = &mut worlds[index];
            let policy_seat = environment.policy_seat;
            let (_, learner_request) = neural_policy_request_in_space(
                &mut environment.seats[policy_seat],
                learner_choices[position].action,
                &learner_spaces[position],
            )?;
            let requests = match &opponent_choices {
                Some(choices) => {
                    let (_, opponent_request) = neural_policy_request_in_space(
                        &mut environment.seats[1 - policy_seat],
                        choices[position].action,
                        &opponent_spaces[position],
                    )?;
                    greedy_requests(environment, learner_request, opponent_request)?
                }
                None => requests_with_candidate(environment, learner_request)?,
            };
            let before = environment.arena.tick();
            let remaining = MAP2_TICK_CAP
                .checked_sub(before)
                .ok_or(PpoError::InvalidTransition("m25 evaluation tick cap"))?;
            if remaining == 0 {
                return Err(PpoError::InvalidTransition(
                    "m25 evaluation reached the tick cap without an outcome",
                ));
            }
            let ticks = MAP2_DECISION_INTERVAL_TICKS.min(remaining);
            let advanced = advance_interval(environment, requests, ticks)?;
            reject_production_rejection(environment, "m25 greedy evaluation")?;
            if let Some(outcome) = terminal_outcome(environment, advanced.winner) {
                let end = environment.arena.tick();
                outcomes[index] = Some(GameOutcome {
                    seat: policy_seat,
                    outcome,
                    ticks: end,
                    cap_end: end == MAP2_TICK_CAP,
                });
            }
        }
        active.retain(|&index| outcomes[index].is_none());
    }
    outcomes
        .into_iter()
        .map(|outcome| outcome.ok_or(PpoError::InvalidTransition("m25 evaluation outcome")))
        .collect()
}

fn summarize(
    config: &EvalConfig,
    outcomes: &[GameOutcome],
    learner_sha: &str,
    opponent_sha: Option<&str>,
    seconds: f64,
) -> Result<String, PpoError> {
    if outcomes.len() != config.games {
        return Err(PpoError::InvalidTransition("m25 evaluation game count"));
    }
    let mut sides = [[0u64; 3]; 2];
    let mut cap_endings = 0u64;
    let mut total_ticks = 0u64;
    for outcome in outcomes {
        let slot = match outcome.outcome {
            PpoTerminalOutcome::Win => 0,
            PpoTerminalOutcome::Loss => 1,
            PpoTerminalOutcome::Draw => 2,
        };
        sides[outcome.seat][slot] += 1;
        if outcome.cap_end {
            cap_endings += 1;
        }
        total_ticks += u64::from(outcome.ticks);
    }
    let played = |seat: usize| sides[seat][0] + sides[seat][1] + sides[seat][2];
    if played(0) == 0 || played(0) != played(1) {
        return Err(PpoError::InvalidTransition("m25 evaluation side balance"));
    }
    let side = |seat: usize| {
        format!(
            "{{\"seat\":{seat},\"wins\":{},\"losses\":{},\"draws\":{}}}",
            sides[seat][0], sides[seat][1], sides[seat][2]
        )
    };
    let opponent = match opponent_sha {
        Some(sha) => format!("\"{sha}\""),
        None => "null".to_owned(),
    };
    let wins = sides[0][0] + sides[1][0];
    let losses = sides[0][1] + sides[1][1];
    let draws = sides[0][2] + sides[1][2];
    let mean_ticks = total_ticks as f64 / config.games as f64;
    let json = format!(
        "{{\"schema\":\"{EVAL_SCHEMA}\",\"opponent\":\"{}\",\"seed\":{},\"offset\":{},\"games\":{},\
\"learner_runtime_sha256\":\"{learner_sha}\",\"opponent_runtime_sha256\":{opponent},\
\"sides\":[{},{}],\"wins\":{wins},\"losses\":{losses},\"draws\":{draws},\
\"cap_endings\":{cap_endings},\"mean_ticks\":{mean_ticks:.3},\"seconds\":{seconds:.3}}}",
        config.kind(),
        config.seed,
        config.offset,
        config.games,
        side(0),
        side(1),
    );
    if json.len() > MAX_JSON_BYTES {
        return Err(PpoError::InvalidTransition("m25 evaluation report bound"));
    }
    Ok(json)
}

fn run_evaluation(config: &EvalConfig, device: PolicyDevice) -> Result<String, PpoError> {
    let learner_sha = runtime_sha256(&config.learner).map_err(PpoError::Model)?;
    let learner = load_runtime_model(&config.learner, device)?;
    let (opponent_model, opponent_sha) = match &config.opponent {
        EvalOpponent::Teacher => (None, None),
        EvalOpponent::Weights(directory) => {
            let sha = runtime_sha256(directory).map_err(PpoError::Model)?;
            (
                Some(Arc::new(load_runtime_model(directory, device)?)),
                Some(sha),
            )
        }
    };
    let mut worlds = Vec::with_capacity(config.games);
    for game in game_plan(config) {
        let spec = match &opponent_model {
            Some(model) => OpponentSpec::SharedPolicy(Arc::clone(model)),
            None => OpponentSpec::Teacher,
        };
        worlds.push(build_environment(
            game.arena_seed,
            game.opponent_seed,
            MapId(2),
            game.seat,
            0,
            spec,
        )?);
    }
    let started = Instant::now();
    let outcomes = play_greedy(&learner, opponent_model.as_deref(), &mut worlds)?;
    let seconds = started.elapsed().as_secs_f64();
    for (plan, outcome) in game_plan(config).iter().zip(&outcomes) {
        eprintln!(
            "m25-eval-game opponent={} pair={} seat={} outcome={:?} tick={} cap_end={}",
            config.kind(),
            plan.pair,
            outcome.seat,
            outcome.outcome,
            outcome.ticks,
            outcome.cap_end
        );
    }
    summarize(
        config,
        &outcomes,
        &learner_sha,
        opponent_sha.as_deref(),
        seconds,
    )
}

#[test]
fn evaluation_config_parsing_is_closed_and_bounded() {
    let config = |pairs: &[(&str, &str)]| {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        EvalConfig::parse(|name| map.get(name).cloned())
    };
    let teacher = config(&[
        ("DRYSUA_EVAL_LEARNER", "/learner"),
        ("DRYSUA_EVAL_OPPONENT", "teacher"),
        ("DRYSUA_EVAL_GAMES", "10"),
        ("DRYSUA_EVAL_OFFSET", "30"),
    ])
    .expect("teacher job");
    assert_eq!(teacher.kind(), "teacher");
    assert_eq!(
        (teacher.games, teacher.offset, teacher.seed),
        (10, 30, DEFAULT_SEED)
    );
    let weights = config(&[
        ("DRYSUA_EVAL_LEARNER", "/learner"),
        ("DRYSUA_EVAL_OPPONENT", "weights"),
        ("DRYSUA_EVAL_OPPONENT_DIR", "/opponent"),
        ("DRYSUA_EVAL_GAMES", "40"),
        ("DRYSUA_EVAL_OFFSET", "0"),
        ("DRYSUA_EVAL_SEED", "77"),
    ])
    .expect("weights job");
    assert_eq!(weights.kind(), "weights");
    assert_eq!(weights.seed, 77);
    let base = [
        ("DRYSUA_EVAL_LEARNER", "/learner"),
        ("DRYSUA_EVAL_OPPONENT", "teacher"),
        ("DRYSUA_EVAL_GAMES", "10"),
        ("DRYSUA_EVAL_OFFSET", "0"),
    ];
    let with = |name: &'static str, value: &'static str| {
        let mut pairs = base.to_vec();
        pairs.push((name, value));
        config(&pairs)
    };
    assert!(with("DRYSUA_EVAL_OPPONENT", "greedy").is_err());
    assert!(with("DRYSUA_EVAL_OPPONENT_DIR", "/opponent").is_err());
    assert!(with("DRYSUA_EVAL_SEED", "0").is_err());
    assert!(with("DRYSUA_EVAL_SEED", "-1").is_err());
    for (name, value) in [
        ("DRYSUA_EVAL_GAMES", "0"),
        ("DRYSUA_EVAL_GAMES", "41"),
        ("DRYSUA_EVAL_GAMES", "11"),
        ("DRYSUA_EVAL_GAMES", "ten"),
        ("DRYSUA_EVAL_OFFSET", "1"),
        ("DRYSUA_EVAL_OFFSET", "40"),
        ("DRYSUA_EVAL_OFFSET", "32"),
    ] {
        assert!(with(name, value).is_err(), "{name}={value}");
    }
    assert!(
        config(&[
            ("DRYSUA_EVAL_LEARNER", "/learner"),
            ("DRYSUA_EVAL_OPPONENT", "weights"),
            ("DRYSUA_EVAL_GAMES", "10"),
            ("DRYSUA_EVAL_OFFSET", "0"),
        ])
        .is_err(),
        "weights opponent needs a directory"
    );
    let missing = config(&[("DRYSUA_EVAL_OPPONENT", "teacher")]).expect_err("learner required");
    assert!(missing.contains("DRYSUA_EVAL_LEARNER"), "{missing}");
    assert_eq!(probe_conflict(std::iter::empty()), None);
    assert_eq!(
        probe_conflict(["DRYSUA_EVAL_GAMES".to_owned()].into_iter()),
        None
    );
    assert_eq!(
        probe_conflict(["DRYSUA_PROBE_NN".to_owned(), "DRYSUA_PROBE_MODE".to_owned()].into_iter())
            .as_deref(),
        Some("DRYSUA_PROBE_NN")
    );
}

#[test]
fn evaluation_plan_covers_unique_paired_worlds_with_balanced_sides() {
    let job = |offset: usize| EvalConfig {
        learner: PathBuf::from("/learner"),
        opponent: EvalOpponent::Teacher,
        games: 10,
        offset,
        seed: DEFAULT_SEED,
    };
    let mut pairs = std::collections::BTreeMap::new();
    let mut games = 0usize;
    for offset in [0, 10, 20, 30] {
        let config = job(offset);
        let plan = game_plan(&config);
        assert_eq!(plan.len(), config.games);
        let seats: Vec<usize> = plan.iter().map(|game| game.seat).collect();
        assert_eq!(seats.iter().filter(|seat| **seat == 0).count(), 5);
        assert_eq!(seats.iter().filter(|seat| **seat == 1).count(), 5);
        for game in plan {
            games += 1;
            let entry = pairs.entry(game.pair).or_insert([0usize; 2]);
            entry[game.seat] += 1;
            assert!(game.arena_seed != game.opponent_seed);
        }
    }
    assert_eq!(games, MAX_GAMES);
    assert_eq!(pairs.len(), MAX_GAMES / 2);
    for (pair, seats) in pairs {
        assert_eq!(
            seats,
            [1, 1],
            "pair {pair} must play both seats exactly once"
        );
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "bounded CUDA greedy evaluation; run only through the authorized runner"]
fn m25_greedy_evaluation_cuda() {
    if let Some(name) = probe_conflict(std::env::vars().map(|(name, _)| name)) {
        panic!("probe environment variable must be unset: {name}");
    }
    let config = EvalConfig::parse(|name| std::env::var(name).ok())
        .unwrap_or_else(|error| panic!("evaluation configuration: {error}"));
    let json = run_evaluation(&config, PolicyDevice::Cuda { ordinal: 0 })
        .unwrap_or_else(|error| panic!("m25 greedy evaluation: {error}"));
    println!("{json}");
}
