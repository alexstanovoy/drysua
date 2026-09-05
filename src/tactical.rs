#![allow(
    clippy::float_arithmetic,
    reason = "bounded f32 neural inference outside simulation"
)]

use std::{error::Error, fmt};

const PARAMETER_BOUND: f32 = 4.0;
const HIDDEN_BIAS_OFFSET: usize = TACTICAL_FEATURES * TACTICAL_HIDDEN;
const OUTPUT_OFFSET: usize = HIDDEN_BIAS_OFFSET + TACTICAL_HIDDEN;

/// Input width; all inputs are finite and normalized to [-1, 1].
pub const TACTICAL_FEATURES: usize = 16;
/// Width of the single hard-tanh hidden layer.
pub const TACTICAL_HIDDEN: usize = 8;
/// Teacher, Fight, Recover, Farm, in stable logit order.
pub const TACTICAL_MODES: usize = 4;
/// Start of the four output biases in the flat parameter vector.
pub const TACTICAL_OUTPUT_BIAS_OFFSET: usize = OUTPUT_OFFSET + TACTICAL_MODES * TACTICAL_HIDDEN;
/// Row-major W1[8,16], b1[8], W2[4,8], b2[4]; 172 trainable f32 values.
pub const TACTICAL_PARAMETERS: usize = TACTICAL_OUTPUT_BIAS_OFFSET + TACTICAL_MODES;
/// Independent artifact schema, including input semantics, architecture and parameter layout.
/// Payload is exactly 172 little-endian f32 values following these UTF-8 bytes.
pub const TACTICAL_SCHEMA_DESCRIPTOR: &str = concat!(
    "drysua-tactical/v1;f32le;16x8x4;hardtanh;W1,b1,W2,b2;bound4;",
    "Teacher,Fight,Recover,Farm;argmax-first-available;",
    "hp_own,hp_enemy,mana_own,mana_enemy,attack_over_enemy_hp,attack_over_own_hp,",
    "distance_over_1200,ready_razes_over_3,own_burst_over_enemy_hp,enemy_burst_over_own_hp,",
    "own_tower_danger,target_tower_danger,pressure_over_own_hp,allied_wave,last_hit,in_attack_range;",
    "ratios=clamp01;burst=two_attacks_plus_one_best_ready_raze;current_visible_only\n"
);
/// Exact byte count accepted by [`TacticalPolicy::from_bytes`].
pub const TACTICAL_FILE_BYTES: usize = TACTICAL_SCHEMA_DESCRIPTOR.len() + TACTICAL_PARAMETERS * 4;

const _: () = assert!(TACTICAL_PARAMETERS == 172);
const _: () = assert!(TACTICAL_MODES == TacticalMode::ALL.len());

/// A legal macro, not an arbitrary action or a simulation seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TacticalMode {
    Teacher,
    Fight,
    Recover,
    Farm,
}

impl TacticalMode {
    pub const ALL: [Self; TACTICAL_MODES] = [Self::Teacher, Self::Fight, Self::Recover, Self::Farm];

    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Checked feature vector. Its order and scales are part of the artifact schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TacticalFeatures([f32; TACTICAL_FEATURES]);

impl TacticalFeatures {
    pub fn from_values(values: [f32; TACTICAL_FEATURES]) -> Result<Self, TacticalError> {
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() || value.abs() > 1.0 {
                return Err(TacticalError::Feature { index });
            }
        }
        Ok(Self(values))
    }

    pub const fn values(&self) -> &[f32; TACTICAL_FEATURES] {
        &self.0
    }
}

/// Immutable, allocation-free inference; no random or hidden state enters decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct TacticalPolicy {
    parameters: [f32; TACTICAL_PARAMETERS],
}

impl TacticalPolicy {
    /// Imports the exact documented layout, rejecting nonfinite or excessive weights.
    pub fn from_parameters(parameters: &[f32]) -> Result<Self, TacticalError> {
        let mut checked = [0.0; TACTICAL_PARAMETERS];
        if parameters.len() != checked.len() {
            return Err(TacticalError::ParameterCount {
                actual: parameters.len(),
            });
        }
        for (index, value) in parameters.iter().copied().enumerate() {
            if !value.is_finite() || value.abs() > PARAMETER_BOUND {
                return Err(TacticalError::Parameter { index });
            }
            checked[index] = value;
        }
        Ok(Self {
            parameters: checked,
        })
    }

    pub const fn parameters(&self) -> &[f32; TACTICAL_PARAMETERS] {
        &self.parameters
    }

    /// Computes bounded logits in mode order; ties are resolved by [`Self::choose`].
    pub fn scores(&self, features: &TacticalFeatures) -> [f32; TACTICAL_MODES] {
        let mut hidden = [0.0; TACTICAL_HIDDEN];
        for (row, activation) in hidden.iter_mut().enumerate() {
            let mut sum = self.parameters[HIDDEN_BIAS_OFFSET + row];
            for (column, input) in features.0.iter().enumerate() {
                sum += self.parameters[row * TACTICAL_FEATURES + column] * input;
            }
            *activation = sum.clamp(-1.0, 1.0);
        }
        let mut scores = [0.0; TACTICAL_MODES];
        for (row, score) in scores.iter_mut().enumerate() {
            *score = self.parameters[TACTICAL_OUTPUT_BIAS_OFFSET + row];
            for (column, activation) in hidden.iter().enumerate() {
                *score +=
                    self.parameters[OUTPUT_OFFSET + row * TACTICAL_HIDDEN + column] * activation;
            }
        }
        assert!(scores.iter().all(|score| score.is_finite()));
        assert!(scores.iter().all(|score| score.abs() <= 36.0));
        scores
    }

    /// Masks unavailable alternatives before argmax. Teacher is always available.
    pub fn choose(
        &self,
        features: &TacticalFeatures,
        available: [bool; TACTICAL_MODES],
    ) -> TacticalMode {
        let scores = self.scores(features);
        let mut best = TacticalMode::Teacher;
        for mode in TacticalMode::ALL.into_iter().skip(1) {
            if available[mode.index()] && scores[mode.index()] > scores[best.index()] {
                best = mode;
            }
        }
        best
    }

    /// SplitMix64 uniform additive mutation of every weight, clipped to [-4, 4].
    /// Scale must be in [0, 1]; equal parent, seed and scale produce equal children.
    pub fn mutated(&self, seed: u64, scale: f32) -> Result<Self, TacticalError> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return Err(TacticalError::MutationScale);
        }
        let mut parameters = self.parameters;
        let mut state = seed;
        for parameter in &mut parameters {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let noise = (value >> 40) as f32 / 16_777_215.0 * 2.0 - 1.0;
            *parameter = (*parameter + noise * scale).clamp(-PARAMETER_BOUND, PARAMETER_BOUND);
        }
        Self::from_parameters(&parameters)
    }

    /// Encodes only schema and exact weight bits, never a seed or training metadata.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TACTICAL_FILE_BYTES);
        bytes.extend_from_slice(TACTICAL_SCHEMA_DESCRIPTOR.as_bytes());
        for parameter in self.parameters {
            bytes.extend_from_slice(&parameter.to_le_bytes());
        }
        assert_eq!(bytes.len(), TACTICAL_FILE_BYTES);
        bytes
    }

    /// Rejects truncated/oversized files, different schemas and invalid weight values.
    /// File callers should bound reads to [`TACTICAL_FILE_BYTES`] plus one byte.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TacticalError> {
        if bytes.len() != TACTICAL_FILE_BYTES {
            return Err(TacticalError::FileSize {
                actual: bytes.len(),
            });
        }
        let (schema, payload) = bytes.split_at(TACTICAL_SCHEMA_DESCRIPTOR.len());
        if schema != TACTICAL_SCHEMA_DESCRIPTOR.as_bytes() {
            return Err(TacticalError::Schema);
        }
        let mut parameters = [0.0; TACTICAL_PARAMETERS];
        let (chunks, remainder) = payload.as_chunks::<4>();
        assert!(remainder.is_empty());
        assert_eq!(chunks.len(), TACTICAL_PARAMETERS);
        for (parameter, chunk) in parameters.iter_mut().zip(chunks) {
            *parameter = f32::from_le_bytes(*chunk);
        }
        Self::from_parameters(&parameters)
    }
}

impl Default for TacticalPolicy {
    fn default() -> Self {
        let mut parameters = [0.0; TACTICAL_PARAMETERS];
        // A live feature basis lets small output mutations learn without reviving dead neurons.
        for (row, positive, negative) in [(0, 0, 1), (1, 2, 3), (2, 4, 5), (3, 8, 9)] {
            parameters[row * TACTICAL_FEATURES + positive] = 1.0;
            parameters[row * TACTICAL_FEATURES + negative] = -1.0;
        }
        for (row, column) in [(4, 6), (5, 12), (6, 11), (7, 14)] {
            parameters[row * TACTICAL_FEATURES + column] = 1.0;
        }
        // The zero output layer preserves Teacher with a small, evolvable margin.
        parameters[TACTICAL_OUTPUT_BIAS_OFFSET] = 0.125;
        Self { parameters }
    }
}

/// Checked model/feature/serialization boundary error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TacticalError {
    ParameterCount { actual: usize },
    Parameter { index: usize },
    Feature { index: usize },
    MutationScale,
    FileSize { actual: usize },
    Schema,
}

impl fmt::Display for TacticalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParameterCount { actual } => write!(
                formatter,
                "tactical parameters have {actual} values; expected {TACTICAL_PARAMETERS}"
            ),
            Self::Parameter { index } => write!(
                formatter,
                "tactical parameter {index} must be finite and in [-4, 4]"
            ),
            Self::Feature { index } => write!(
                formatter,
                "tactical feature {index} must be finite and in [-1, 1]"
            ),
            Self::MutationScale => {
                formatter.write_str("tactical mutation scale must be finite and in [0, 1]")
            }
            Self::FileSize { actual } => write!(
                formatter,
                "tactical file has {actual} bytes; expected {TACTICAL_FILE_BYTES}"
            ),
            Self::Schema => formatter.write_str(
                "tactical file schema does not match this architecture and feature order",
            ),
        }
    }
}

impl Error for TacticalError {}
