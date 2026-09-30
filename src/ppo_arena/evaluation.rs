//! Frozen evaluation: one candidate against every pool opponent on both sides of
//! every seed; no optimizer, no rollout.
//!
//! Each game owns its arena seed and its actor RNG streams, both pure functions
//! of `(seed, seat)`, so every candidate and every opponent meets the same worlds.
//! Games are statically assigned to pipeline groups and to batch slots, and each
//! policy row samples with its own RNG, so one argument set always writes
//! byte-identical output whatever the batch shape.

use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};

use super::eval_players::{
    Player, PlayerSpec, Policy, PoolEntry, evaluation_context, load_player, validate_name,
};
use super::game_summary::GameSummary;
use super::{
    OpponentRuntime, TrainingEnvironment, advance_interval, build_environment,
    derive_training_seed, neural_policy_request_in_space, prepare_neural_seat_policy_sample,
    prepare_policy_sample, reject_production_rejection, teacher_request, terminal_outcome,
    text_error,
};
use crate::randomization::{ARENA_DOMAIN, OPPONENT_DOMAIN};
use crate::{
    ActionSpace, FeatureFrame, MAP2_ACTOR_DECISIONS, MAP2_DECISION_INTERVAL_TICKS, MAP2_TICK_CAP,
    PPO_MAX_PARALLEL_WORLDS, PolicyDevice, PolicyModel, PpoError, PpoRng, PpoTerminalOutcome,
    Request, ScriptKind, ScriptedPolicy, StructuredAction,
};

const SCHEMA: &str = "drysua-eval/v3";
const ACTOR_DOMAIN: u64 = 0x6576_616c_5f61_6374;
/// Bounds one evaluation to a few hours of simulation and a small JSONL file.
pub(crate) const MAX_EVALUATION_SEEDS: u64 = 10_000;
const MAX_EVALUATION_GAMES: u64 = 20_000;

/// One validated evaluation request.
#[derive(Clone, Debug)]
pub(crate) struct EvaluationSettings {
    pub candidate_name: String,
    pub candidate: PlayerSpec,
    pub pool: Vec<PoolEntry>,
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
        if self.pool.is_empty() || self.pool.len() > super::eval_players::MAX_POOL_ENTRIES {
            return Err(PpoError::InvalidConfig("evaluation pool size"));
        }
        if self.games() > MAX_EVALUATION_GAMES {
            return Err(PpoError::InvalidConfig(
                "evaluation games (seeds x pool x 2)",
            ));
        }
        validate_name(&self.candidate_name).map_err(PpoError::Model)?;
        if !matches!(self.groups, 1 | 2 | 4) {
            return Err(PpoError::InvalidConfig("evaluation pipeline groups"));
        }
        if self.parallel == 0 || self.parallel * self.groups > PPO_MAX_PARALLEL_WORLDS {
            return Err(PpoError::InvalidConfig("evaluation parallel worlds"));
        }
        Ok(())
    }

    fn games(&self) -> u64 {
        self.seeds * 2 * self.pool.len() as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlannedGame {
    seed: u64,
    /// Index into the pool.
    opponent: usize,
    /// The candidate's seat.
    seat: usize,
}

/// Samples of the seats whose policy is neural; rule seats decide while stepping.
type Samples = (Option<Prepared>, Option<Prepared>);

/// One live world; `prepared` is taken for inference and refilled after each step.
struct LiveWorld {
    game: PlannedGame,
    environment: TrainingEnvironment,
    candidate_rng: PpoRng,
    opponent_rng: PpoRng,
    prepared: Option<Samples>,
    decisions: usize,
}

struct Group {
    pending: VecDeque<PlannedGame>,
    live: Vec<LiveWorld>,
    finished: Vec<(PlannedGame, GameSummary)>,
}

/// One chosen action with the action space it was chosen in.
type Chosen = (StructuredAction, ActionSpace);

/// The neural choices of one world: candidate seat, opponent seat.
type WorldActions = (Option<Chosen>, Option<Chosen>);

struct Models {
    candidate: Player,
    pool: Vec<Player>,
}

/// Plays every planned game and writes a header line plus one JSON line per game.
pub(crate) fn run_evaluation(settings: &EvaluationSettings, output: &Path) -> Result<(), PpoError> {
    settings.validate()?;
    if std::fs::symlink_metadata(output).is_ok() {
        return Err(PpoError::Model(format!(
            "{} already exists; evaluation never replaces a result",
            output.display()
        )));
    }
    let context = evaluation_context(settings.greedy)?;
    let models = Models {
        candidate: load_player(&settings.candidate, settings.device)?,
        pool: settings
            .pool
            .iter()
            .map(|entry| load_player(&entry.player, settings.device))
            .collect::<Result<_, _>>()?,
    };
    let started = std::time::Instant::now();
    let mut results = play_all(settings, &models)?;
    results.sort_by_key(|(game, _)| *game);
    let mut lines = Vec::with_capacity(results.len() + 1);
    lines.push(header_json(settings, &models, &context));
    for (game, summary) in &results {
        let mut line = summary.json();
        line["schema"] = json!(SCHEMA);
        line["kind"] = json!("game");
        line["opponent"] = json!(settings.pool[game.opponent].name);
        line["seed"] = json!(game.seed);
        lines.push(line);
    }
    write_lines(output, &lines)?;
    log_results(settings, &results);
    eprintln!(
        "level=INFO event=evaluation_complete games={} seconds={:.1} output={}",
        results.len(),
        started.elapsed().as_secs_f64(),
        output.display()
    );
    Ok(())
}

fn header_json(settings: &EvaluationSettings, models: &Models, context: &str) -> Value {
    json!({
        "schema": SCHEMA,
        "kind": "header",
        "context": context,
        "drysua_commit": option_env!("DRYSUA_GIT_COMMIT"),
        "bota_commit": option_env!("BOTA_GIT_COMMIT"),
        "greedy": settings.greedy,
        "seeds": {"first": settings.first_seed, "count": settings.seeds},
        "candidate": {
            "name": settings.candidate_name,
            "player": settings.candidate.label(),
            "key": models.candidate.key,
        },
        "pool": settings
            .pool
            .iter()
            .zip(&models.pool)
            .map(|(entry, player)| entry.json(&player.key))
            .collect::<Vec<_>>(),
    })
}

/// One `key=value` line per opponent and side for a quick look; statistics
/// belong to `scripts/eval_pool.py`.
fn log_results(settings: &EvaluationSettings, results: &[(PlannedGame, GameSummary)]) {
    for (index, entry) in settings.pool.iter().enumerate() {
        let mut line = format!(
            "level=INFO event=evaluation_opponent opponent={}",
            entry.name
        );
        for side in [bota_proto::Team::Radiant, bota_proto::Team::Dire] {
            let games = results
                .iter()
                .filter(|(game, summary)| game.opponent == index && summary.side == side);
            let (mut wins, mut losses, mut draws) = (0u64, 0u64, 0u64);
            for (_, summary) in games {
                match summary.outcome {
                    Some(PpoTerminalOutcome::Win) => wins += 1,
                    Some(PpoTerminalOutcome::Loss) => losses += 1,
                    Some(PpoTerminalOutcome::Draw) | None => draws += 1,
                }
            }
            let label = super::game_summary::side_label(side);
            line.push_str(&format!(" {label}={wins}-{losses}-{draws}"));
        }
        eprintln!("{line}");
    }
}

/// Game `index` goes to group `index % groups`, so assignment never depends on timing.
fn plan_groups(settings: &EvaluationSettings) -> Vec<VecDeque<PlannedGame>> {
    let mut groups = vec![VecDeque::new(); settings.groups];
    let per_seed = 2 * settings.pool.len() as u64;
    for index in 0..settings.games() {
        groups[(index % settings.groups as u64) as usize].push_back(PlannedGame {
            seed: settings.first_seed + index / per_seed,
            opponent: (index % per_seed / 2) as usize,
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
        fill(&mut group, settings.parallel, models)?;
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
                stepping[index] =
                    Some(scope.spawn(move || step_group(group, actions, parallel, models)));
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
    if results.len() as u64 != settings.games() {
        return Err(PpoError::InvalidTransition("evaluation game count"));
    }
    Ok(results)
}

/// Starts pending games in plan order until the group holds `parallel` live worlds.
fn fill(group: &mut Group, parallel: usize, models: &Models) -> Result<(), PpoError> {
    while group.live.len() < parallel {
        let Some(game) = group.pending.pop_front() else {
            break;
        };
        group.live.push(start_game(game, models)?);
    }
    Ok(())
}

/// The runtime the environment builder expects for the opponent seat.
const fn opponent_runtime(policy: &Policy) -> OpponentRuntime {
    match policy {
        Policy::Neural(_) => OpponentRuntime::Neural,
        Policy::Script(ScriptKind::Teacher) => OpponentRuntime::Teacher,
        Policy::Script(ScriptKind::HarassPush) => OpponentRuntime::HarassPush,
        Policy::Styled(kind) => OpponentRuntime::Styled(*kind),
    }
}

fn start_game(game: PlannedGame, models: &Models) -> Result<LiveWorld, PpoError> {
    let opponent = &models.pool[game.opponent].policy;
    let arena_seed = derive_training_seed(game.seed, 0, ARENA_DOMAIN);
    let mut environment = build_environment(
        arena_seed,
        crate::MAP2_ID,
        game.seat,
        opponent_runtime(opponent),
        Vec::new(),
    )?;
    match models.candidate.policy {
        Policy::Script(kind) => environment.seats[game.seat].script = ScriptedPolicy::new(kind),
        Policy::Styled(kind) => {
            environment.seats[game.seat].script =
                ScriptedPolicy::styled_preset(kind, crate::seat_seed(arena_seed, game.seat));
        }
        Policy::Neural(_) => {}
    }
    let neural = (
        models.candidate.model().is_some(),
        matches!(opponent, Policy::Neural(_)),
    );
    let prepared = prepare(&mut environment, neural)?;
    Ok(LiveWorld {
        game,
        environment,
        candidate_rng: PpoRng::new(derive_training_seed(
            game.seed,
            game.seat as u64,
            ACTOR_DOMAIN,
        )),
        opponent_rng: PpoRng::new(derive_training_seed(
            game.seed,
            game.seat as u64,
            OPPONENT_DOMAIN,
        )),
        prepared: Some(prepared),
        decisions: 0,
    })
}

type Prepared = (FeatureFrame, ActionSpace);

/// Encodes the neural seats, `(candidate, opponent)`.
fn prepare(
    environment: &mut TrainingEnvironment,
    (candidate, opponent): (bool, bool),
) -> Result<Samples, PpoError> {
    let candidate = if candidate {
        Some(prepare_policy_sample(environment)?)
    } else {
        None
    };
    let opponent = if opponent {
        let seat = 1 - environment.policy_seat;
        Some(prepare_neural_seat_policy_sample(
            &mut environment.seats[seat],
        )?)
    } else {
        None
    };
    Ok((candidate, opponent))
}

/// Batches the candidate rows through its model and each opponent model's rows through it.
fn infer(group: &mut Group, models: &Models, greedy: bool) -> Result<Vec<WorldActions>, PpoError> {
    let worlds = group.live.len();
    let mut candidate_rows = Vec::new();
    let mut opponent_rows: Vec<Vec<(usize, Prepared, &mut PpoRng)>> =
        (0..models.pool.len()).map(|_| Vec::new()).collect();
    for (index, world) in group.live.iter_mut().enumerate() {
        let (candidate, opponent) = world
            .prepared
            .take()
            .ok_or(PpoError::InvalidTransition("evaluation prepared sample"))?;
        if let Some(sample) = candidate {
            candidate_rows.push((index, sample, &mut world.candidate_rng));
        }
        if let Some(sample) = opponent {
            opponent_rows[world.game.opponent].push((index, sample, &mut world.opponent_rng));
        }
    }
    let mut actions: Vec<WorldActions> = (0..worlds).map(|_| (None, None)).collect();
    if !candidate_rows.is_empty() {
        let model = models
            .candidate
            .model()
            .ok_or(PpoError::InvalidTransition("evaluation candidate samples"))?;
        for (index, chosen) in choose(model, candidate_rows, greedy)? {
            actions[index].0 = Some(chosen);
        }
    }
    for (player, rows) in models.pool.iter().zip(opponent_rows) {
        if rows.is_empty() {
            continue;
        }
        let model = player
            .model()
            .ok_or(PpoError::InvalidTransition("evaluation opponent samples"))?;
        for (index, chosen) in choose(model, rows, greedy)? {
            actions[index].1 = Some(chosen);
        }
    }
    Ok(actions)
}

/// Samples with one RNG per row, or takes the legal argmax when greedy; keeps row indices.
fn choose(
    model: &PolicyModel,
    rows: Vec<(usize, Prepared, &mut PpoRng)>,
    greedy: bool,
) -> Result<Vec<(usize, Chosen)>, PpoError> {
    let mut indices = Vec::with_capacity(rows.len());
    let mut frames = Vec::with_capacity(rows.len());
    let mut spaces = Vec::with_capacity(rows.len());
    let mut rngs = Vec::with_capacity(rows.len());
    for (index, (frame, space), rng) in rows {
        indices.push(index);
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
    assert_eq!(actions.len(), indices.len());
    Ok(indices
        .into_iter()
        .zip(actions.into_iter().zip(spaces))
        .collect())
}

/// Advances every live world one decision, in parallel, then refills finished slots.
fn step_group(
    mut group: Group,
    actions: Vec<WorldActions>,
    parallel: usize,
    models: &Models,
) -> Result<Group, PpoError> {
    assert_eq!(actions.len(), group.live.len());
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = group.live.len().div_ceil(threads);
    let mut rows: Vec<(&mut LiveWorld, WorldActions)> =
        group.live.iter_mut().zip(actions).collect();
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks_mut(chunk)
            .map(|chunk| {
                scope.spawn(move || -> Result<Vec<Option<GameSummary>>, PpoError> {
                    chunk
                        .iter_mut()
                        .map(|(world, actions)| step_world(world, actions))
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
    replace_finished(&mut group, outcomes, parallel, models)?;
    Ok(group)
}

fn replace_finished(
    group: &mut Group,
    outcomes: Vec<Option<GameSummary>>,
    parallel: usize,
    models: &Models,
) -> Result<(), PpoError> {
    assert_eq!(outcomes.len(), group.live.len());
    let mut kept = Vec::with_capacity(parallel);
    for (world, outcome) in std::mem::take(&mut group.live).into_iter().zip(outcomes) {
        match outcome {
            Some(summary) => {
                eprintln!(
                    "level=INFO event=evaluation_game seed={} opponent_index={} {summary}",
                    world.game.seed, world.game.opponent
                );
                group.finished.push((world.game, summary));
            }
            None => kept.push(world),
        }
    }
    group.live = kept;
    fill(group, parallel, models)
}

fn step_world(
    world: &mut LiveWorld,
    (candidate, opponent): &WorldActions,
) -> Result<Option<GameSummary>, PpoError> {
    if world.decisions >= MAP2_ACTOR_DECISIONS {
        return Err(PpoError::InvalidTransition("evaluation decision bound"));
    }
    world.decisions += 1;
    let environment = &mut world.environment;
    let requests = requests(environment, candidate.as_ref(), opponent.as_ref())?;
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
    world.prepared = Some(prepare(
        environment,
        (candidate.is_some(), opponent.is_some()),
    )?);
    Ok(None)
}

/// Neural seats send their chosen action; rule seats decide now.
fn requests(
    environment: &mut TrainingEnvironment,
    candidate: Option<&Chosen>,
    opponent: Option<&Chosen>,
) -> Result<Vec<Option<Request>>, PpoError> {
    let policy_seat = environment.policy_seat;
    let mut requests = Vec::with_capacity(environment.seats.len());
    for (index, seat) in environment.seats.iter_mut().enumerate() {
        let chosen = if index == policy_seat {
            candidate
        } else {
            opponent
        };
        let request = match chosen {
            Some((action, space)) => neural_policy_request_in_space(seat, *action, space)?.1,
            None => teacher_request(seat)?,
        };
        requests.push(request);
    }
    Ok(requests)
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
