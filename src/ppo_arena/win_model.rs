#![allow(
    clippy::float_arithmetic,
    reason = "The learned potential is a logistic model fitted in f64; reward stays f32-exact per run"
)]

//! Reward 9: the shaping potential `Φ(s) = SCALE · (P(win | s) − ½)` of a
//! logistic win-probability model the run fits on its own recent games.
//!
//! The features read both seats, including what the policy seat cannot see
//! (the opponent's gold, HP, mana and respawn), so the model is strictly
//! training-side: it only shapes the reward of training games and never
//! reaches the policy's input, runtime weights, `play` or evaluation.
//!
//! Every game keeps the model version it started with, so its shaping
//! telescopes to `Φ(end) − Φ(start)` with `Φ(end) = 0` at the terminal; the
//! optimal policy is unchanged. Refits happen every `every` updates from a
//! bounded window of finished games, are deterministic f64 Newton steps over
//! games in collection order, and are checkpointed, so resume stays exact.
//! Until the window holds `min_games` games new games use the hand potential.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};

use super::TrainingEnvironment;
use crate::{MAP2_PREGAME_TICKS, MAP2_TICK_CAP, MAP2_TICK_RATE, PpoError};

/// Raw features; see [`features`] for the layout.
pub(crate) const BASE: usize = 17;
/// Bias, raw features and every non-clock feature times the clock.
pub(crate) const DIM: usize = 1 + BASE + (BASE - 1);
/// Game-clock spacing of the samples a finished game contributes.
pub(crate) const SAMPLE_TICKS: u32 = 30 * MAP2_TICK_RATE;
/// Samples of the longest game: clock 0 through the tick cap.
pub(crate) const MAX_GAME_SAMPLES: usize =
    ((MAP2_TICK_CAP - MAP2_PREGAME_TICKS) / SAMPLE_TICKS) as usize + 1;
pub(crate) const MAX_WINDOW_GAMES: usize = 4096;
/// Model versions one checkpoint may keep alive.
pub(crate) const MAX_MODELS: usize = 16;
/// Potential range in reward units: a sure win is +SCALE/2 above a coin flip.
const SCALE: f64 = 1.0;
const RIDGE: f64 = 1.0;
const NEWTON_STEPS: usize = 12;
/// Antisymmetric leads, negated when the seats swap.
const LEADS: std::ops::Range<usize> = 1..7;
/// Own/enemy pairs, exchanged when the seats swap.
const PAIRS: [(usize, usize); 5] = [(7, 8), (9, 10), (11, 12), (13, 14), (15, 16)];
/// Game minutes whose samples the refit diagnostics score.
const AUC_MINUTES: [usize; 8] = [0, 1, 2, 3, 4, 5, 7, 10];

pub(crate) type WinFeatures = [f32; BASE];

/// How a run with the learned potential refits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WinModelConfig {
    /// Updates between refits.
    pub every: u64,
    /// Most recent finished games the fit uses.
    pub games: usize,
    /// Games the window needs before the first fit replaces the hand potential.
    pub min_games: usize,
}

impl WinModelConfig {
    pub(crate) fn validate(self) -> Result<(), PpoError> {
        if !(1..=crate::MAX_TRAINING_COUNTER).contains(&self.every)
            || !(1..=MAX_WINDOW_GAMES).contains(&self.games)
            || !(1..=self.games).contains(&self.min_games)
        {
            return Err(PpoError::InvalidConfig("learned potential refit"));
        }
        Ok(())
    }
}

/// The policy seat's features, reading both seats' own trackers:
/// clock; leads in XP, level, last hits, denies, bounty gold and the hand
/// potential; then own/enemy deaths, weakest-tower HP, hero HP, mana and respawn.
pub(super) fn features(environment: &TrainingEnvironment) -> WinFeatures {
    let own = &environment.seats[environment.policy_seat];
    let enemy = &environment.seats[1 - environment.policy_seat];
    let tick = own.tracker.current().map_or(0, |view| view.tick);
    let (mine, theirs) = (own.tracker.own_player(), enemy.tracker.own_player());
    let stat = |pick: fn(&bota_proto::PlayerView) -> f32| {
        mine.map_or(0.0, pick) - theirs.map_or(0.0, pick)
    };
    let hero = |seat: &super::ArenaSeatPolicy, mana: bool| {
        seat.tracker.own_hero().map_or(0.0, |hero| {
            let (value, maximum) = if mana {
                (hero.mana, hero.max_mana)
            } else {
                (hero.hp, hero.max_hp)
            };
            value.max(0) as f32 / maximum.max(1) as f32
        })
    };
    let deaths = |view: Option<&bota_proto::PlayerView>| {
        view.map_or(0.0, |view| f32::from(view.deaths)) / 2.0
    };
    let respawn = |view: Option<&bota_proto::PlayerView>| {
        view.map_or(0.0, |view| view.respawn_left as f32) / 900.0
    };
    let hand = own
        .tracker
        .map2_reward_state()
        .map_or(0.0, |state| state.potential);
    [
        tick.saturating_sub(MAP2_PREGAME_TICKS) as f32 / 18_000.0,
        stat(|view| view.xp as f32) / 1_000.0,
        stat(|view| f32::from(view.level)) / 5.0,
        stat(|view| f32::from(view.last_hits)) / 20.0,
        stat(|view| f32::from(view.denies)) / 10.0,
        (own.combat.bounty() - enemy.combat.bounty()) as f32 / 1_000.0,
        hand,
        deaths(mine),
        deaths(theirs),
        super::game_summary::own_tower_fraction(own) as f32,
        super::game_summary::own_tower_fraction(enemy) as f32,
        hero(own, false),
        hero(enemy, false),
        hero(own, true),
        hero(enemy, true),
        respawn(mine),
        respawn(theirs),
    ]
}

/// The same state seen from the other seat.
fn mirror(features: &WinFeatures) -> WinFeatures {
    let mut out = *features;
    for index in LEADS {
        out[index] = -features[index];
    }
    for (own, enemy) in PAIRS {
        out[own] = features[enemy];
        out[enemy] = features[own];
    }
    out
}

fn expand(features: &WinFeatures) -> [f64; DIM] {
    let mut row = [0.0; DIM];
    row[0] = 1.0;
    let clock = f64::from(features[0]);
    for (index, &value) in features.iter().enumerate() {
        row[1 + index] = f64::from(value);
        if index > 0 {
            row[BASE + index] = f64::from(value) * clock;
        }
    }
    row
}

fn logistic(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

/// One fitted model; `version` is the completed updates it was fitted after.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WinModel {
    pub(crate) version: u64,
    weights: [f64; DIM],
}

impl WinModel {
    pub(crate) fn from_weights(version: u64, weights: [f64; DIM]) -> Result<Self, PpoError> {
        if version == 0 || weights.iter().any(|weight| !weight.is_finite()) {
            return Err(PpoError::InvalidTransition("learned potential model"));
        }
        Ok(Self { version, weights })
    }

    pub(crate) const fn weights(&self) -> &[f64; DIM] {
        &self.weights
    }

    pub(crate) fn probability(&self, features: &WinFeatures) -> f64 {
        let row = expand(features);
        logistic(row.iter().zip(&self.weights).map(|(x, w)| x * w).sum())
    }

    pub(crate) fn potential(&self, features: &WinFeatures) -> f64 {
        SCALE * (self.probability(features) - 0.5)
    }
}

/// Model versions games may reference, shared read-only with the lanes;
/// version 0 is the hand potential and never stored.
pub(crate) type WinModels = Arc<RwLock<BTreeMap<u64, Arc<WinModel>>>>;

pub(crate) fn lookup(models: &WinModels, version: u64) -> Result<Option<Arc<WinModel>>, PpoError> {
    if version == 0 {
        return Ok(None);
    }
    let models = models
        .read()
        .map_err(|_| PpoError::InvalidTransition("learned potential registry"))?;
    models
        .get(&version)
        .cloned()
        .map(Some)
        .ok_or(PpoError::InvalidTransition("learned potential version"))
}

/// One finished game's samples and score (0 loss, 1 draw or cap, 2 win).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WinGame {
    pub(crate) samples: Vec<WinFeatures>,
    pub(crate) score: u8,
}

/// The learner-side state: the window, live models and the version new games use.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct WinState {
    pub(crate) games: VecDeque<WinGame>,
    /// Games added since the last fit: the held-out set of its diagnostics.
    pub(crate) fresh: usize,
    pub(crate) current: u64,
    pub(crate) models: Vec<WinModel>,
}

impl WinState {
    pub(crate) fn record(&mut self, config: WinModelConfig, game: WinGame) {
        assert!(game.samples.len() <= MAX_GAME_SAMPLES && game.score <= 2);
        if self.games.len() == config.games {
            self.games.pop_front();
        }
        self.games.push_back(game);
        self.fresh = (self.fresh + 1).min(self.games.len());
    }

    /// Refits after `completed` updates when due; returns the diagnostics line.
    pub(crate) fn refit(&mut self, config: WinModelConfig, completed: u64) -> Option<String> {
        if !completed.is_multiple_of(config.every) || self.games.len() < config.min_games {
            return None;
        }
        let previous = self
            .models
            .iter()
            .find(|model| model.version == self.current);
        let line = diagnostics(&self.games, self.fresh, previous, completed);
        let model = fit(&self.games, completed);
        self.models.push(model);
        self.current = completed;
        self.fresh = 0;
        Some(line)
    }

    /// Keeps the current model and every version a live game or published
    /// configuration still references.
    pub(crate) fn retain(&mut self, referenced: impl Fn(u64) -> bool) {
        let current = self.current;
        self.models
            .retain(|model| model.version == current || referenced(model.version));
        assert!(self.models.len() <= MAX_MODELS);
    }

    pub(crate) fn publish(&self, registry: &WinModels) -> Result<(), PpoError> {
        let mut models = registry
            .write()
            .map_err(|_| PpoError::InvalidTransition("learned potential registry"))?;
        for model in &self.models {
            models
                .entry(model.version)
                .or_insert_with(|| Arc::new(model.clone()));
        }
        Ok(())
    }
}

/// Ridge-regularized logistic regression by Newton steps over every sample and
/// its mirror; sequential f64 sums make it bit-exact for a given window.
fn fit(games: &VecDeque<WinGame>, version: u64) -> WinModel {
    let mut weights = [0.0; DIM];
    for _ in 0..NEWTON_STEPS {
        let mut hessian = [[0.0; DIM]; DIM];
        let mut gradient = [0.0; DIM];
        for game in games {
            let target = f64::from(game.score) / 2.0;
            for sample in &game.samples {
                for (row, target) in [
                    (expand(sample), target),
                    (expand(&mirror(sample)), 1.0 - target),
                ] {
                    let p = logistic(row.iter().zip(&weights).map(|(x, w)| x * w).sum());
                    let curvature = p * (1.0 - p);
                    for i in 0..DIM {
                        gradient[i] += (target - p) * row[i];
                        for j in 0..=i {
                            hessian[i][j] += curvature * row[i] * row[j];
                        }
                    }
                }
            }
        }
        for i in 1..DIM {
            hessian[i][i] += RIDGE;
            gradient[i] -= RIDGE * weights[i];
        }
        hessian[0][0] += 1.0e-9;
        let step = cholesky_solve(&hessian, &gradient);
        for (weight, delta) in weights.iter_mut().zip(step) {
            *weight += delta;
        }
    }
    WinModel { version, weights }
}

/// Solves `H x = g` for a symmetric positive definite `H` given by its lower triangle.
fn cholesky_solve(hessian: &[[f64; DIM]; DIM], gradient: &[f64; DIM]) -> [f64; DIM] {
    let mut lower = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| lower[i][k] * lower[j][k]).sum();
            if i == j {
                lower[i][i] = (hessian[i][i] - sum).max(1.0e-12).sqrt();
            } else {
                lower[i][j] = (hessian[i][j] - sum) / lower[j][j];
            }
        }
    }
    let mut forward = [0.0; DIM];
    for i in 0..DIM {
        let sum: f64 = (0..i).map(|k| lower[i][k] * forward[k]).sum();
        forward[i] = (gradient[i] - sum) / lower[i][i];
    }
    let mut solution = [0.0; DIM];
    for i in (0..DIM).rev() {
        let sum: f64 = (i + 1..DIM).map(|k| lower[k][i] * solution[k]).sum();
        solution[i] = (forward[i] - sum) / lower[i][i];
    }
    solution
}

/// Held-out ranking quality of the model in force and of the hand potential on
/// the games finished since it was fitted, by game minute.
fn diagnostics(
    games: &VecDeque<WinGame>,
    fresh: usize,
    model: Option<&WinModel>,
    completed: u64,
) -> String {
    let held_out: Vec<&WinGame> = games.iter().skip(games.len() - fresh).collect();
    let mut line = format!(
        "level=INFO event=win_model update={completed} version={completed} games={} held_out={}",
        games.len(),
        held_out.len()
    );
    for minute in AUC_MINUTES {
        let index = 2 * minute;
        let scored = |score: &dyn Fn(&WinFeatures) -> f64| {
            let pairs: Vec<(f64, bool)> = held_out
                .iter()
                .filter(|game| game.score != 1)
                .filter_map(|game| game.samples.get(index).map(|x| (score(x), game.score == 2)))
                .collect();
            auc(&pairs)
        };
        let hand = scored(&|x| f64::from(x[6]));
        line.push_str(&format!(" auc_hand_m{minute}={hand:.4}"));
        if let Some(model) = model {
            let learned = scored(&|x| model.probability(x));
            line.push_str(&format!(" auc_learned_m{minute}={learned:.4}"));
        }
    }
    line
}

/// Area under the ROC curve with tied scores ranked by their mean; NaN
/// unless both classes are present.
fn auc(pairs: &[(f64, bool)]) -> f64 {
    let positives = pairs.iter().filter(|(_, win)| *win).count() as f64;
    let negatives = pairs.len() as f64 - positives;
    if positives == 0.0 || negatives == 0.0 {
        return f64::NAN;
    }
    let mut sorted: Vec<(f64, bool)> = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum = 0.0;
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start;
        while end < sorted.len() && sorted[end].0 == sorted[start].0 {
            end += 1;
        }
        let mean_rank = (start + end + 1) as f64 / 2.0;
        rank_sum += mean_rank * sorted[start..end].iter().filter(|(_, win)| *win).count() as f64;
        start = end;
    }
    (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)
}

/// Largest encoded state (see `collector_state`).
pub(crate) const MAX_ENCODED_BYTES: usize =
    16 + MAX_MODELS * (8 + 8 * DIM) + 4 + MAX_WINDOW_GAMES * (2 + MAX_GAME_SAMPLES * BASE * 4);
const _: () = assert!(MAX_ENCODED_BYTES <= crate::checkpoint::MAX_WIN_MODEL_STATE_BYTES);

#[cfg(test)]
#[path = "../tests/win_model.rs"]
mod tests;
