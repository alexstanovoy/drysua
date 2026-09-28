//! Experimental episodic RND: combined reward, gamma=lambda=1, no separate critic.
//! Own-body/resources favor exploration of embodied state, not necessarily winning.
//! No courier, orders, IDs, cooldowns, clocks, reward-accounting or hidden inputs.

use super::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

const INPUTS: usize = 24;
const HIDDEN: usize = 16;
const OUTPUTS: usize = 8;
const SECOND: usize = (INPUTS + 1) * HIDDEN;
const PARAMETERS: usize = SECOND + (HIDDEN + 1) * OUTPUTS;
const FIT_LIMIT: usize = 1024;
const BODY_BYTES: usize = 64 + PARAMETERS * 4;
const STATE_BYTES: usize = BODY_BYTES + 32;
const MAGIC: &[u8; 8] = b"DRND0001";
type Input = [f32; INPUTS];
type Network = [f32; PARAMETERS];
type Scores = Vec<(usize, f64)>;
const _: () = assert!(INPUTS <= 32 && FIT_LIMIT / 32 * 4 == 128);
const _: () = assert!(STATE_BYTES == 2240);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rnd {
    seed: u64,
    target: Network,
    predictor: Network,
    updates: u64,
    sgd_steps: u64,
    error_count: u64,
    error_square_sum: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RndReport {
    pub rows_scored: usize,
    pub error_mean: f64,
    pub bonus_sum: f64,
    pub max_bonus: f32,
    pub predictor_loss_before: f64,
    pub predictor_loss_after: f64,
    pub normalization_rms: f64,
    pub updates: u64,
    pub sgd_steps: u64,
}

impl Rnd {
    pub const SPEC: &'static str = "rnd-v1-own24-16-8-relu-fixed-inputs-frozen-rollout-score-error-rms-floor.001-clip5-bonuscap.0001-reservoir1024-sgd.01-clip1-4epochs-batch32-episodic-gamma1-lambda1";

    pub fn new(seed: u64) -> Self {
        let initialize = |domain| {
            let mut random = PpoRng::new(seed ^ domain);
            std::array::from_fn(|_| {
                ((random.uniform_open().expect("bounded RND initialization") * 2.0 - 1.0) * 0.2)
                    as f32
            })
        };
        Self {
            seed,
            target: initialize(0x524e_4454),
            predictor: initialize(0x524e_4450),
            updates: 0,
            sgd_steps: 0,
            error_count: 0,
            error_square_sum: 0.0,
        }
    }

    /// Call once before `finish`; coefficient zero still fits the predictor.
    /// Neither the PPO optimizer nor rollout action/policy statistics are touched.
    pub fn augment(
        &mut self,
        rollout: &mut PpoRollout,
        coefficient: f32,
    ) -> Result<RndReport, PpoError> {
        if !coefficient.is_finite() || !(0.0..=1.0e-4).contains(&coefficient) {
            return Err(PpoError::InvalidConfig("RND coefficient"));
        }
        if rollout.is_empty() {
            return Err(PpoError::EmptyRollout);
        }
        if rollout.len() > rollout.sample_budget.max_samples() {
            return Err(PpoError::InvalidTransition("RND rollout capacity"));
        }
        let mut draft = self.clone();
        draft.updates = draft
            .updates
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        let (scores, states) = self.score(rollout)?;
        let mut report = RndReport {
            rows_scored: scores.len(),
            ..Default::default()
        };
        draft.error_count = draft
            .error_count
            .checked_add(scores.len() as u64)
            .ok_or(PpoError::CounterOverflow)?;
        draft.error_square_sum += scores.iter().map(|(_, error)| error * error).sum::<f64>();
        report.normalization_rms = (draft.error_square_sum / draft.error_count.max(1) as f64)
            .sqrt()
            .max(0.001);
        report.error_mean =
            scores.iter().map(|(_, error)| error).sum::<f64>() / scores.len().max(1) as f64;
        let mut rewards = Vec::with_capacity(scores.len());
        for (index, error) in scores {
            let bonus = novelty_bonus(error, report.normalization_rms, coefficient);
            let original = rollout.transitions[index].reward;
            let reward = reward_with_bonus(original, bonus);
            if !reward.is_finite() {
                return Err(PpoError::NonFinite("RND augmented reward"));
            }
            rewards.push((index, reward));
            let applied = f64::from(reward) - f64::from(original);
            report.bonus_sum += applied;
            report.max_bonus = report.max_bonus.max(applied as f32);
        }
        (report.predictor_loss_before, report.predictor_loss_after) = draft.fit(&states)?;
        draft.validate()?;
        for (index, reward) in rewards {
            rollout.transitions[index].reward = reward;
        }
        report.updates = draft.updates;
        report.sgd_steps = draft.sgd_steps;
        *self = draft;
        Ok(report)
    }

    fn score(&self, rollout: &PpoRollout) -> Result<(Scores, Vec<Input>), PpoError> {
        let mut next = [None; PPO_MAX_STREAMS];
        let mut scores = Vec::with_capacity(rollout.len());
        let mut states = Vec::with_capacity(FIT_LIMIT);
        let mut random = PpoRng::new(self.seed ^ 0x524e_4453 ^ self.updates.rotate_left(17));
        for (index, row) in rollout.transitions.iter().enumerate().rev() {
            if row.stream >= PPO_MAX_STREAMS || !row.reward.is_finite() {
                return Err(PpoError::InvalidTransition("RND stream or reward"));
            }
            let frame = rollout
                .frames
                .expand(&row.frame)
                .map_err(PpoError::InvalidTransition)?;
            let current = observation(&frame)?;
            let Some(input) = next_observation(&mut next, row.stream, row.terminal, current) else {
                continue;
            };
            let error = self.error(&input);
            if !error.is_finite() {
                return Err(PpoError::NonFinite("RND prediction error"));
            }
            scores.push((index, error));
            if states.len() < FIT_LIMIT {
                states.push(input);
            } else {
                let chosen = (random.next_u64()? % scores.len() as u64) as usize;
                if chosen < FIT_LIMIT {
                    states[chosen] = input;
                }
            }
        }
        Ok((scores, states))
    }

    fn error(&self, input: &Input) -> f64 {
        let (_, expected) = forward(&self.target, input);
        let (_, actual) = forward(&self.predictor, input);
        expected
            .iter()
            .zip(actual)
            .map(|(a, b)| (f64::from(*a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / OUTPUTS as f64
    }

    fn fit(&mut self, states: &[Input]) -> Result<(f64, f64), PpoError> {
        assert!(states.len() <= FIT_LIMIT);
        if states.is_empty() {
            return Ok((0.0, 0.0));
        }
        let before =
            states.iter().map(|input| self.error(input)).sum::<f64>() / states.len() as f64;
        let mut order = (0..states.len()).collect::<Vec<_>>();
        let mut random = PpoRng::new(self.seed ^ 0x524e_4446 ^ self.updates.rotate_left(17));
        for _ in 0..4 {
            random.shuffle(&mut order)?;
            for chunk in order.chunks(32) {
                let mut gradient = [0.0; PARAMETERS];
                for index in chunk {
                    accumulate_gradient(
                        &self.predictor,
                        &states[*index],
                        &forward(&self.target, &states[*index]).1,
                        &mut gradient,
                    );
                }
                for value in &mut gradient {
                    *value /= chunk.len() as f32;
                }
                let norm = gradient
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if !norm.is_finite() {
                    return Err(PpoError::NonFinite("RND gradient"));
                }
                let scale = (0.01 / norm.max(1.0)) as f32;
                for (weight, gradient) in self.predictor.iter_mut().zip(gradient) {
                    *weight -= scale * gradient;
                }
                self.sgd_steps = self
                    .sgd_steps
                    .checked_add(1)
                    .ok_or(PpoError::CounterOverflow)?;
            }
        }
        let after = states.iter().map(|input| self.error(input)).sum::<f64>() / states.len() as f64;
        if !after.is_finite() {
            return Err(PpoError::NonFinite("RND predictor loss"));
        }
        Ok((before, after))
    }

    fn validate(&self) -> Result<(), PpoError> {
        let maximum = PPO_MAX_STREAMS as u64 * PPO_MAX_ROLLOUT_DECISIONS as u64;
        if self.updates > crate::MAX_TRAINING_COUNTER
            || self.sgd_steps > self.updates * 128
            || self.error_count > self.updates * maximum
            || !self.error_square_sum.is_finite()
            || self.error_square_sum < 0.0
            || (self.error_count == 0 && self.error_square_sum != 0.0)
            || self
                .predictor
                .iter()
                .any(|weight| !weight.is_finite() || weight.abs() > 32.0)
        {
            return Err(PpoError::InvalidTransition("RND numeric state"));
        }
        Ok(())
    }

    /// Creates a new sidecar only. The coordinator commits checkpoint/sidecar manifest together.
    pub fn save(&self, path: &Path, completed_update: u64) -> Result<(), PpoError> {
        self.validate()?;
        if completed_update > crate::MAX_TRAINING_COUNTER || completed_update < self.updates {
            return Err(PpoError::InvalidTransition("RND sidecar update"));
        }
        let mut bytes = Vec::with_capacity(STATE_BYTES);
        bytes.extend_from_slice(MAGIC);
        for value in [
            completed_update,
            self.seed,
            self.updates,
            self.sgd_steps,
            self.error_count,
            self.error_square_sum.to_bits(),
            FEATURE_SCHEMA_HASH,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.predictor {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&Sha256::digest(&bytes));
        assert_eq!(bytes.len(), STATE_BYTES);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| PpoError::Model(format!("RND sidecar create: {error}")))?;
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            drop(file);
            std::fs::remove_file(path).map_err(|cleanup| {
                PpoError::Model(format!("RND sidecar write: {error}; cleanup: {cleanup}"))
            })?;
            return Err(PpoError::Model(format!("RND sidecar write: {error}")));
        }
        Ok(())
    }

    pub fn load(path: &Path, expected_update: u64) -> Result<Self, PpoError> {
        let file = std::fs::File::open(path)
            .map_err(|error| PpoError::Model(format!("RND sidecar open: {error}")))?;
        let mut bytes = Vec::with_capacity(STATE_BYTES + 1);
        file.take((STATE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| PpoError::Model(format!("RND sidecar read: {error}")))?;
        if bytes.len() != STATE_BYTES {
            return Err(PpoError::InvalidTransition("RND sidecar length"));
        }
        if Sha256::digest(&bytes[..BODY_BYTES])[..] != bytes[BODY_BYTES..] {
            return Err(PpoError::InvalidTransition("RND sidecar checksum"));
        }
        let word = |offset: usize| {
            u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .expect("fixed RND header"),
            )
        };
        if &bytes[..8] != MAGIC || word(56) != FEATURE_SCHEMA_HASH {
            return Err(PpoError::InvalidTransition("RND sidecar version"));
        }
        if word(8) != expected_update || expected_update > crate::MAX_TRAINING_COUNTER {
            return Err(PpoError::InvalidTransition("RND sidecar update"));
        }
        let mut result = Self::new(word(16));
        result.updates = word(24);
        result.sgd_steps = word(32);
        result.error_count = word(40);
        result.error_square_sum = f64::from_bits(word(48));
        for (weight, bytes) in result
            .predictor
            .iter_mut()
            .zip(bytes[64..BODY_BYTES].as_chunks::<4>().0)
        {
            *weight = f32::from_le_bytes(*bytes);
        }
        result.validate()?;
        if result.updates > expected_update {
            return Err(PpoError::InvalidTransition("RND sidecar update"));
        }
        Ok(result)
    }
}

fn observation(frame: &FeatureFrame) -> Result<Input, PpoError> {
    use crate::{global_feature as g, unit_feature as u};
    let globals = [
        g::OWN_GOLD,
        g::OWN_ASSET_VALUE,
        g::OWN_ALIVE,
        g::OWN_LEVEL,
        g::OWN_XP,
        g::OWN_LAST_HITS,
        g::OWN_DENIES,
    ];
    let hero = [
        u::TOKEN_PRESENT,
        u::POSITION_X,
        u::POSITION_Y,
        u::VELOCITY_PRESENT,
        u::VELOCITY_X,
        u::VELOCITY_Y,
        u::HP_PRESENT,
        u::HP_RATIO,
        u::MANA_PRESENT,
        u::MANA_RATIO,
        u::ATTACK_DAMAGE,
        u::ATTACK_RANGE,
        u::MOVE_SPEED,
        u::ARMOR,
        u::MAGIC_RESISTANCE,
        u::ITEM_SLOT_COUNT,
        u::FREE_ITEM_SLOTS,
    ];
    let input: Input = std::array::from_fn(|index| {
        if index < globals.len() {
            frame.global[globals[index]]
        } else {
            frame.own_units[0][hero[index - globals.len()]]
        }
    });
    if input.iter().any(|value| !value.is_finite()) {
        return Err(PpoError::NonFinite("RND observation"));
    }
    Ok(input.map(|value| value.clamp(-1.0, 1.0)))
}

fn next_observation(
    next: &mut [Option<Input>; PPO_MAX_STREAMS],
    stream: usize,
    terminal: bool,
    current: Input,
) -> Option<Input> {
    assert!(stream < PPO_MAX_STREAMS);
    let result = if terminal { None } else { next[stream] };
    next[stream] = Some(current);
    result
}

fn novelty_bonus(error: f64, rms: f64, coefficient: f32) -> f32 {
    (f64::from(coefficient) * (error / rms).clamp(0.0, 5.0)).min(1.0e-4) as f32
}

fn reward_with_bonus(reward: f32, bonus: f32) -> f32 {
    if bonus == 0.0 {
        return reward;
    }
    let updated = reward + bonus;
    // Round downward if f32 addition would exceed the requested intrinsic budget.
    if f64::from(updated) - f64::from(reward) > f64::from(bonus) {
        updated.next_down()
    } else {
        updated
    }
}

fn forward(weights: &Network, input: &Input) -> ([f32; HIDDEN], [f32; OUTPUTS]) {
    let hidden = std::array::from_fn(|h| {
        let offset = h * (INPUTS + 1);
        (weights[offset + INPUTS]
            + input
                .iter()
                .enumerate()
                .map(|(i, value)| weights[offset + i] * value)
                .sum::<f32>())
        .max(0.0)
    });
    let output = std::array::from_fn(|o| {
        let offset = SECOND + o * (HIDDEN + 1);
        weights[offset + HIDDEN]
            + hidden
                .iter()
                .enumerate()
                .map(|(h, value)| weights[offset + h] * value)
                .sum::<f32>()
    });
    (hidden, output)
}

fn accumulate_gradient(
    weights: &Network,
    input: &Input,
    target: &[f32; OUTPUTS],
    gradient: &mut Network,
) {
    let (hidden, output) = forward(weights, input);
    let mut hidden_gradient = [0.0; HIDDEN];
    for o in 0..OUTPUTS {
        let delta = 2.0 * (output[o] - target[o]) / OUTPUTS as f32;
        let offset = SECOND + o * (HIDDEN + 1);
        gradient[offset + HIDDEN] += delta;
        for h in 0..HIDDEN {
            gradient[offset + h] += delta * hidden[h];
            hidden_gradient[h] += delta * weights[offset + h];
        }
    }
    for h in 0..HIDDEN {
        if hidden[h] <= 0.0 {
            continue;
        }
        let offset = h * (INPUTS + 1);
        gradient[offset + INPUTS] += hidden_gradient[h];
        for i in 0..INPUTS {
            gradient[offset + i] += hidden_gradient[h] * input[i];
        }
    }
}

#[test]
fn rnd_fixed_target_and_repeated_input_predictor_fit() {
    let mut rnd = Rnd::new(428);
    let target = rnd.target;
    let states = vec![[0.5; INPUTS]; 64];
    let (before, after) = rnd.fit(&states).expect("fit");
    assert!(after < before, "before={before}, after={after}");
    assert_eq!(rnd.target, target);
    assert_eq!(rnd.sgd_steps, 8);
}

#[test]
fn rnd_next_state_is_stream_local_terminal_and_missing_tail_receive_no_bonus() {
    let mut next = [None; PPO_MAX_STREAMS];
    assert_eq!(next_observation(&mut next, 0, false, [5.0; INPUTS]), None);
    assert_eq!(next_observation(&mut next, 1, false, [4.0; INPUTS]), None);
    assert_eq!(next_observation(&mut next, 0, true, [3.0; INPUTS]), None);
    assert_eq!(
        next_observation(&mut next, 1, false, [2.0; INPUTS]),
        Some([4.0; INPUTS])
    );
    assert_eq!(
        next_observation(&mut next, 0, false, [1.0; INPUTS]),
        Some([3.0; INPUTS])
    );
    assert_eq!(reward_with_bonus(-0.0, 0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(novelty_bonus(100.0, 0.001, 0.0), 0.0);
    assert!(novelty_bonus(100.0, 0.001, 1.0e-4) <= 1.0e-4);
    for reward in [-1000.0, -0.2, 0.0, 0.2, 1000.0] {
        let applied = f64::from(reward_with_bonus(reward, 1.0e-4)) - f64::from(reward);
        assert!((0.0..=1.0e-4).contains(&applied));
    }
    let mut frame = FeatureFrame::new();
    let input = observation(&frame).expect("input");
    frame.global[crate::global_feature::TICK] = 1.0;
    frame.global[crate::global_feature::ACTIVE_ORDER_AGE] = 1.0;
    frame.own_units[1].fill(1.0);
    assert_eq!(observation(&frame).expect("excluded noise"), input);
    let wide = Rnd {
        updates: 1,
        error_count: 64 * crate::MAP2_RETAINED_DECISIONS as u64,
        error_square_sum: 1.0,
        ..Rnd::new(428)
    };
    wide.validate()
        .expect("normalizer accepts more than forty episodes");
}

#[test]
fn rnd_sidecar_continues_identically_and_rejects_corruption_or_wrong_update() {
    let directory = super::test_directory("rnd-state");
    let path = directory.join("rnd.bin");
    assert!(
        matches!(Rnd::load(&path, 429), Err(PpoError::Model(message)) if message.starts_with("RND sidecar open:"))
    );
    let mut original = Rnd::new(428);
    original.updates = 1;
    original.error_count = 64;
    original.error_square_sum = 0.125;
    let states = vec![[0.25; INPUTS]; 64];
    original.fit(&states).expect("initial fit");
    original.save(&path, 429).expect("save");
    let mut resumed = Rnd::load(&path, 429).expect("load");
    assert_eq!(original, resumed);
    assert!(original.save(&path, 429).is_err());
    assert_eq!(Rnd::load(&path, 429).expect("not overwritten"), original);
    assert_eq!(
        original.fit(&states).expect("continued"),
        resumed.fit(&states).expect("resumed")
    );
    assert_eq!(original, resumed);
    assert_eq!(
        Rnd::load(&path, 430).unwrap_err(),
        PpoError::InvalidTransition("RND sidecar update")
    );
    let mut bytes = std::fs::read(&path).expect("bytes");
    bytes[80] ^= 1;
    std::fs::write(&path, bytes).expect("corrupt");
    assert_eq!(
        Rnd::load(&path, 429).unwrap_err(),
        PpoError::InvalidTransition("RND sidecar checksum")
    );
    let mut bytes = std::fs::read(&path).expect("corrupt bytes");
    bytes[64..68].copy_from_slice(&f32::NAN.to_le_bytes());
    let checksum = Sha256::digest(&bytes[..BODY_BYTES]);
    bytes[BODY_BYTES..].copy_from_slice(&checksum);
    std::fs::write(&path, bytes).expect("nonfinite state");
    assert_eq!(
        Rnd::load(&path, 429).unwrap_err(),
        PpoError::InvalidTransition("RND numeric state")
    );
    std::fs::remove_dir_all(directory).expect("cleanup");
}
