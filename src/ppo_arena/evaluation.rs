//! Frozen-weights evaluation: no optimizer, no rollout, both sides of every seed.
//!
//! Each game owns its arena seed and its actor RNG streams, both pure functions
//! of `(seed, seat)`. Games are statically assigned to pipeline groups and to
//! batch slots, so one argument set always forms the same inference batches and
//! writes byte-identical output.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use super::game_summary::{EndReason, GameSummary};
use super::{
    OpponentSpec, TrainingEnvironment, advance_interval, build_environment, derive_training_seed,
    neural_policy_request_in_space, prepare_neural_seat_policy_sample, prepare_policy_sample,
    reject_production_rejection, teacher_request, terminal_outcome, text_error,
};
use crate::randomization::{ARENA_DOMAIN, OPPONENT_DOMAIN};
use crate::{
    ActionSpace, FeatureFrame, MAP2_ACTOR_DECISIONS, MAP2_DECISION_INTERVAL_TICKS, MAP2_TICK_CAP,
    PPO_MAX_PARALLEL_WORLDS, PolicyDevice, PolicyModel, PpoError, PpoRng, PpoTerminalOutcome,
    Request, StructuredAction, TrainingArtifact,
};

const SCHEMA: &str = "drysua-eval/v1";
const ACTOR_DOMAIN: u64 = 0x6576_616c_5f61_6374;
/// Bounds one evaluation to a few hours of simulation and a small JSONL file.
pub(crate) const MAX_EVALUATION_SEEDS: u64 = 10_000;
const RUNTIME_FILE: &str = "drysua.weights.safetensors";
const MAX_RUNTIME_BYTES: u64 = 16 * 1024 * 1024;
const WILSON_Z: f64 = 1.959_963_984_540_054;

/// The frozen opponent every candidate game faces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EvaluationOpponent {
    Teacher,
    Weights(PathBuf),
}

/// One validated evaluation request.
#[derive(Clone, Debug)]
pub(crate) struct EvaluationSettings {
    pub candidate: PathBuf,
    pub opponent: EvaluationOpponent,
    pub first_seed: u64,
    pub seeds: u64,
    /// Worlds per pipeline group, which is also the inference batch.
    pub parallel: usize,
    pub groups: usize,
    pub greedy: bool,
    pub device: PolicyDevice,
}

impl EvaluationSettings {
    fn validate(&self) -> Result<(), PpoError> {
        if self.seeds == 0 || self.seeds > MAX_EVALUATION_SEEDS {
            return Err(PpoError::InvalidConfig("evaluation seed count"));
        }
        if self.first_seed.checked_add(self.seeds).is_none() {
            return Err(PpoError::InvalidConfig("evaluation seed range"));
        }
        if !matches!(self.groups, 1 | 2 | 4) {
            return Err(PpoError::InvalidConfig("evaluation pipeline groups"));
        }
        if self.parallel == 0 || self.parallel * self.groups > PPO_MAX_PARALLEL_WORLDS {
            return Err(PpoError::InvalidConfig("evaluation parallel worlds"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedGame {
    seed: u64,
    seat: usize,
}

/// One live world; `prepared` is taken for inference and refilled after each step.
struct LiveWorld {
    game: PlannedGame,
    environment: TrainingEnvironment,
    learner_rng: PpoRng,
    opponent_rng: PpoRng,
    prepared: Option<(Prepared, Option<Prepared>)>,
    decisions: usize,
}

struct Group {
    pending: VecDeque<PlannedGame>,
    live: Vec<LiveWorld>,
    finished: Vec<(PlannedGame, GameSummary)>,
}

/// One chosen action with the action space it was chosen in.
type Chosen = (StructuredAction, ActionSpace);

/// Chosen actions, aligned with a group's live worlds.
struct GroupActions {
    learner: Vec<Chosen>,
    opponent: Option<Vec<Chosen>>,
}

struct Models {
    learner: PolicyModel,
    opponent: Option<Arc<PolicyModel>>,
}

/// Plays every planned game and writes one JSON line per game plus a summary.
pub(crate) fn run_evaluation(settings: &EvaluationSettings, output: &Path) -> Result<(), PpoError> {
    settings.validate()?;
    if std::fs::symlink_metadata(output).is_ok() {
        return Err(PpoError::Model(format!(
            "{} already exists; evaluation never replaces a result",
            output.display()
        )));
    }
    let candidate_sha = weights_sha256(&settings.candidate)?;
    let opponent_sha = match &settings.opponent {
        EvaluationOpponent::Teacher => None,
        EvaluationOpponent::Weights(directory) => Some(weights_sha256(directory)?),
    };
    let models = Models {
        learner: load_model(&settings.candidate, settings.device)?,
        opponent: match &settings.opponent {
            EvaluationOpponent::Teacher => None,
            EvaluationOpponent::Weights(directory) => {
                Some(Arc::new(load_model(directory, settings.device)?))
            }
        },
    };
    let started = std::time::Instant::now();
    let mut results = play_all(settings, &models)?;
    results.sort_by_key(|(game, _)| (game.seed, game.seat));
    let mut lines = Vec::with_capacity(results.len() + 1);
    for (game, summary) in &results {
        let mut line = summary.json();
        line["schema"] = json!(SCHEMA);
        line["seed"] = json!(game.seed);
        lines.push(line);
    }
    let summaries: Vec<GameSummary> = results.iter().map(|(_, summary)| *summary).collect();
    lines.push(summary_json(
        settings,
        &candidate_sha,
        opponent_sha.as_deref(),
        &summaries,
    ));
    write_lines(output, &lines)?;
    eprintln!(
        "level=INFO event=evaluation_complete games={} seconds={:.1} output={}",
        results.len(),
        started.elapsed().as_secs_f64(),
        output.display()
    );
    Ok(())
}

fn load_model(directory: &Path, device: PolicyDevice) -> Result<PolicyModel, PpoError> {
    let model = PolicyModel::fresh_on(0, device).map_err(text_error)?;
    TrainingArtifact::load_runtime_weights(&model, directory).map_err(|error| {
        PpoError::Model(format!(
            "{}: {error}",
            directory.join(RUNTIME_FILE).display()
        ))
    })?;
    Ok(model)
}

fn weights_sha256(directory: &Path) -> Result<String, PpoError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let path = directory.join(RUNTIME_FILE);
    let describe = |error: std::io::Error| PpoError::Model(format!("{}: {error}", path.display()));
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .map_err(describe)?
        .take(MAX_RUNTIME_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(describe)?;
    if bytes.len() as u64 > MAX_RUNTIME_BYTES {
        return Err(PpoError::Model(format!(
            "{} exceeds {MAX_RUNTIME_BYTES} bytes",
            path.display()
        )));
    }
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Game `index` goes to group `index % groups`, so assignment never depends on timing.
fn plan_groups(settings: &EvaluationSettings) -> Vec<VecDeque<PlannedGame>> {
    let mut groups = vec![VecDeque::new(); settings.groups];
    for index in 0..settings.seeds * 2 {
        groups[(index % settings.groups as u64) as usize].push_back(PlannedGame {
            seed: settings.first_seed + index / 2,
            seat: (index % 2) as usize,
        });
    }
    groups
}

/// Infers one group on this thread while the previous group steps on workers.
fn play_all(
    settings: &EvaluationSettings,
    models: &Models,
) -> Result<Vec<(PlannedGame, GameSummary)>, PpoError> {
    let mut groups = Vec::with_capacity(settings.groups);
    for pending in plan_groups(settings) {
        let mut group = Group {
            pending,
            live: Vec::with_capacity(settings.parallel),
            finished: Vec::new(),
        };
        fill(&mut group, settings.parallel, models.opponent.as_ref())?;
        groups.push(Some(group));
    }
    std::thread::scope(|scope| {
        let mut stepping: Vec<Option<std::thread::ScopedJoinHandle<'_, _>>> =
            (0..groups.len()).map(|_| None).collect();
        loop {
            let mut active = false;
            for index in 0..groups.len() {
                let mut group = match stepping[index].take() {
                    Some(handle) => handle
                        .join()
                        .map_err(|_| PpoError::Model("evaluation worker panicked".to_owned()))??,
                    None => groups[index]
                        .take()
                        .ok_or(PpoError::InvalidTransition("evaluation group ownership"))?,
                };
                if group.live.is_empty() {
                    groups[index] = Some(group);
                    continue;
                }
                active = true;
                let actions = infer(&mut group, models, settings.greedy)?;
                let parallel = settings.parallel;
                let opponent = models.opponent.clone();
                stepping[index] = Some(
                    scope.spawn(move || step_group(group, actions, parallel, opponent.as_ref())),
                );
            }
            if !active {
                return Ok::<(), PpoError>(());
            }
        }
    })?;
    let mut results = Vec::new();
    for group in groups.into_iter().flatten() {
        results.extend(group.finished);
    }
    if results.len() as u64 != settings.seeds * 2 {
        return Err(PpoError::InvalidTransition("evaluation game count"));
    }
    Ok(results)
}

/// Starts pending games in plan order until the group holds `parallel` live worlds.
fn fill(
    group: &mut Group,
    parallel: usize,
    opponent: Option<&Arc<PolicyModel>>,
) -> Result<(), PpoError> {
    while group.live.len() < parallel {
        let Some(game) = group.pending.pop_front() else {
            break;
        };
        group.live.push(start_game(game, opponent)?);
    }
    Ok(())
}

fn start_game(
    game: PlannedGame,
    opponent: Option<&Arc<PolicyModel>>,
) -> Result<LiveWorld, PpoError> {
    let spec = match opponent {
        Some(model) => OpponentSpec::SharedPolicy(Arc::clone(model)),
        None => OpponentSpec::Teacher,
    };
    let opponent_seed = derive_training_seed(game.seed, game.seat as u64, OPPONENT_DOMAIN);
    let mut environment = build_environment(
        derive_training_seed(game.seed, 0, ARENA_DOMAIN),
        opponent_seed,
        crate::MAP2_ID,
        game.seat,
        spec,
        Vec::new(),
    )?;
    let prepared = prepare(&mut environment, opponent.is_some())?;
    Ok(LiveWorld {
        game,
        environment,
        learner_rng: PpoRng::new(derive_training_seed(
            game.seed,
            game.seat as u64,
            ACTOR_DOMAIN,
        )),
        opponent_rng: PpoRng::new(opponent_seed),
        prepared: Some(prepared),
        decisions: 0,
    })
}

type Prepared = (FeatureFrame, ActionSpace);

fn prepare(
    environment: &mut TrainingEnvironment,
    neural_opponent: bool,
) -> Result<(Prepared, Option<Prepared>), PpoError> {
    let learner = prepare_policy_sample(environment)?;
    let opponent = if neural_opponent {
        let seat = 1 - environment.policy_seat;
        Some(prepare_neural_seat_policy_sample(
            &mut environment.seats[seat],
        )?)
    } else {
        None
    };
    Ok((learner, opponent))
}

fn infer(group: &mut Group, models: &Models, greedy: bool) -> Result<GroupActions, PpoError> {
    let mut learner = Vec::with_capacity(group.live.len());
    let mut opponent = Vec::with_capacity(group.live.len());
    for world in &mut group.live {
        let (own, other) = world
            .prepared
            .take()
            .ok_or(PpoError::InvalidTransition("evaluation prepared sample"))?;
        learner.push((own, &mut world.learner_rng));
        if let Some(other) = other {
            opponent.push((other, &mut world.opponent_rng));
        }
    }
    let learner = choose(&models.learner, learner, greedy)?;
    let opponent = match &models.opponent {
        Some(model) if opponent.len() == learner.len() => Some(choose(model, opponent, greedy)?),
        None if opponent.is_empty() => None,
        _ => return Err(PpoError::InvalidTransition("evaluation opponent samples")),
    };
    Ok(GroupActions { learner, opponent })
}

/// Samples with one RNG per row, or takes the legal argmax when greedy.
fn choose(
    model: &PolicyModel,
    rows: Vec<(Prepared, &mut PpoRng)>,
    greedy: bool,
) -> Result<Vec<Chosen>, PpoError> {
    let mut frames = Vec::with_capacity(rows.len());
    let mut spaces = Vec::with_capacity(rows.len());
    let mut rngs = Vec::with_capacity(rows.len());
    for ((frame, space), rng) in rows {
        frames.push(frame);
        spaces.push(space);
        rngs.push(rng);
    }
    let actions: Vec<StructuredAction> = if greedy {
        model
            .choose_batch(&frames, &spaces)
            .map_err(text_error)?
            .into_iter()
            .map(|choice| choice.action)
            .collect()
    } else {
        let mut staged: Vec<PpoRng> = rngs.iter().map(|rng| (**rng).clone()).collect();
        let choices = model
            .sample_batch(&frames, &spaces, &mut staged)
            .map_err(text_error)?;
        for (rng, next) in rngs.iter_mut().zip(staged) {
            **rng = next;
        }
        choices.iter().map(|choice| choice.action()).collect()
    };
    Ok(actions.into_iter().zip(spaces).collect())
}

/// Advances every live world one decision, in parallel, then refills finished slots.
fn step_group(
    mut group: Group,
    actions: GroupActions,
    parallel: usize,
    opponent: Option<&Arc<PolicyModel>>,
) -> Result<Group, PpoError> {
    assert_eq!(actions.learner.len(), group.live.len());
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = group.live.len().div_ceil(threads);
    let mut opponent_actions = actions.opponent.map(Vec::into_iter);
    let mut rows: Vec<(&mut LiveWorld, Chosen, Option<Chosen>)> = group
        .live
        .iter_mut()
        .zip(actions.learner)
        .map(|(world, learner)| {
            let opponent = opponent_actions.as_mut().and_then(Iterator::next);
            (world, learner, opponent)
        })
        .collect();
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks_mut(chunk)
            .map(|chunk| {
                scope.spawn(move || -> Result<Vec<Option<GameSummary>>, PpoError> {
                    chunk
                        .iter_mut()
                        .map(|(world, learner, opponent)| {
                            step_world(world, learner, opponent.as_ref())
                        })
                        .collect()
                })
            })
            .collect();
        let mut outcomes = Vec::new();
        for handle in handles {
            outcomes.extend(
                handle
                    .join()
                    .map_err(|_| PpoError::Model("evaluation step panicked".to_owned()))??,
            );
        }
        Ok::<_, PpoError>(outcomes)
    })?;
    drop(rows);
    replace_finished(&mut group, outcomes, parallel, opponent)?;
    Ok(group)
}

fn replace_finished(
    group: &mut Group,
    outcomes: Vec<Option<GameSummary>>,
    parallel: usize,
    opponent: Option<&Arc<PolicyModel>>,
) -> Result<(), PpoError> {
    assert_eq!(outcomes.len(), group.live.len());
    let mut kept = Vec::with_capacity(parallel);
    for (world, outcome) in std::mem::take(&mut group.live).into_iter().zip(outcomes) {
        match outcome {
            Some(summary) => {
                eprintln!(
                    "level=INFO event=evaluation_game seed={} {summary}",
                    world.game.seed
                );
                group.finished.push((world.game, summary));
            }
            None => kept.push(world),
        }
    }
    group.live = kept;
    fill(group, parallel, opponent)
}

fn step_world(
    world: &mut LiveWorld,
    learner: &Chosen,
    opponent: Option<&Chosen>,
) -> Result<Option<GameSummary>, PpoError> {
    if world.decisions >= MAP2_ACTOR_DECISIONS {
        return Err(PpoError::InvalidTransition("evaluation decision bound"));
    }
    world.decisions += 1;
    let environment = &mut world.environment;
    let requests = requests(environment, learner, opponent)?;
    let before = environment.arena.tick();
    let remaining = MAP2_TICK_CAP
        .checked_sub(before)
        .filter(|remaining| *remaining > 0)
        .ok_or(PpoError::InvalidTransition("evaluation tick cap"))?;
    let advanced = advance_interval(
        environment,
        requests,
        MAP2_DECISION_INTERVAL_TICKS.min(remaining),
    )?;
    reject_production_rejection(environment, "evaluation")?;
    let outcome = terminal_outcome(environment, advanced.winner);
    let end = before + advanced.ticks;
    if outcome.is_some() || end >= MAP2_TICK_CAP {
        return Ok(Some(GameSummary::capture(environment, outcome, end)));
    }
    world.prepared = Some(prepare(environment, opponent.is_some())?);
    Ok(None)
}

fn requests(
    environment: &mut TrainingEnvironment,
    learner: &Chosen,
    opponent: Option<&Chosen>,
) -> Result<Vec<Option<Request>>, PpoError> {
    let policy_seat = environment.policy_seat;
    let mut requests = Vec::with_capacity(environment.seats.len());
    for (index, seat) in environment.seats.iter_mut().enumerate() {
        let request = if index == policy_seat {
            neural_policy_request_in_space(seat, learner.0, &learner.1)?.1
        } else {
            match opponent {
                Some((action, space)) => neural_policy_request_in_space(seat, *action, space)?.1,
                None => teacher_request(seat)?,
            }
        };
        requests.push(request);
    }
    Ok(requests)
}

/// Wilson score interval for `wins` successes in `games` trials at 95%.
#[allow(clippy::float_arithmetic, reason = "report statistics only")]
pub(crate) fn wilson_interval(wins: u64, games: u64) -> (f64, f64) {
    assert!(wins <= games);
    if games == 0 {
        return (0.0, 1.0);
    }
    let n = games as f64;
    let p = wins as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = WILSON_Z / denominator * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((center - half).max(0.0), (center + half).min(1.0))
}

#[allow(clippy::float_arithmetic, reason = "report statistics only")]
fn summary_json(
    settings: &EvaluationSettings,
    candidate_sha: &str,
    opponent_sha: Option<&str>,
    games: &[GameSummary],
) -> Value {
    let count = |predicate: &dyn Fn(&GameSummary) -> bool| {
        games.iter().filter(|game| predicate(game)).count() as u64
    };
    let wins = count(&|game| game.outcome == Some(PpoTerminalOutcome::Win));
    let losses = count(&|game| game.outcome == Some(PpoTerminalOutcome::Loss));
    let (low, high) = wilson_interval(wins, games.len() as u64);
    let mut sides = serde_json::Map::new();
    for side in [bota_proto::Team::Radiant, bota_proto::Team::Dire] {
        let played = count(&|game| game.side == side);
        let won = count(&|game| game.side == side && game.outcome == Some(PpoTerminalOutcome::Win));
        sides.insert(
            super::game_summary::side_label(side).to_owned(),
            json!({"games": played, "wins": won}),
        );
    }
    let mut end_reasons = serde_json::Map::new();
    for outcome in ["win", "loss", "draw"] {
        let mut reasons = serde_json::Map::new();
        for reason in [
            EndReason::Tower,
            EndReason::Deaths,
            EndReason::TimeCap,
            EndReason::Draw,
        ] {
            let matched =
                count(&|game| game.outcome_label() == outcome && game.end_reason == reason);
            reasons.insert(reason.label().to_owned(), json!(matched));
        }
        end_reasons.insert(outcome.to_owned(), Value::Object(reasons));
    }
    let opponent = match &settings.opponent {
        EvaluationOpponent::Teacher => "teacher".to_owned(),
        EvaluationOpponent::Weights(directory) => format!("weights:{}", directory.display()),
    };
    json!({
        "schema": SCHEMA,
        "summary": true,
        "candidate": settings.candidate.display().to_string(),
        "candidate_sha256": candidate_sha,
        "opponent": opponent,
        "opponent_sha256": opponent_sha,
        "seeds": format!("{}:{}", settings.first_seed, settings.seeds),
        "greedy": settings.greedy,
        "games": games.len(),
        "wins": wins,
        "losses": losses,
        "draws": games.len() as u64 - wins - losses,
        "win_rate": wins as f64 / games.len() as f64,
        "win_rate_ci95": [low, high],
        "sides": sides,
        "end_reasons": end_reasons,
    })
}

/// Writes a new file and syncs it; never replaces an earlier result.
fn write_lines(output: &Path, lines: &[Value]) -> Result<(), PpoError> {
    let describe =
        |error: std::io::Error| PpoError::Model(format!("{}: {error}", output.display()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(describe)?;
    let mut text = String::new();
    for line in lines {
        text.push_str(&line.to_string());
        text.push('\n');
    }
    file.write_all(text.as_bytes()).map_err(describe)?;
    file.sync_all().map_err(describe)
}

#[cfg(test)]
#[path = "../tests/evaluation.rs"]
mod tests;
