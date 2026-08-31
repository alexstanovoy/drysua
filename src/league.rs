#![allow(
    clippy::float_arithmetic,
    reason = "league ratings and held-out evaluation use floating-point values"
)]

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION,
    MODEL_PARAMETER_COUNT, MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, ModelError, PPO_SCHEMA_HASH,
    PPO_SCHEMA_VERSION, PolicyModel, PpoRng,
};

/// Initial and maximum retained historical policy count.
pub const LEAGUE_MAX_POLICIES: usize = 32;
/// Smallest league that can preserve every retention class.
pub const LEAGUE_MIN_POLICIES: usize = 9;
/// Recent snapshots protected from historical eviction.
pub const LEAGUE_RECENT_POLICIES: usize = 4;
/// Minimum side-paired held-out seeds required for promotion.
pub const LEAGUE_MIN_PROMOTION_PAIRS: usize = 20;
/// Maximum side-paired held-out seeds accepted by one report.
pub const LEAGUE_MAX_PROMOTION_PAIRS: usize = 512;
/// Minimum evaluated actions required for a promotion decision.
pub const LEAGUE_MIN_PROMOTION_ACTIONS: u64 = 1_000;
/// Minimum separate exploit-regression seed pairs required for promotion.
pub const LEAGUE_MIN_EXPLOIT_PAIRS: usize = 2;
/// Minimum candidate actions required in the exploit-regression namespace.
pub const LEAGUE_MIN_EXPLOIT_ACTIONS: u64 = 100;
/// Stage-ten league contract version.
pub const LEAGUE_SCHEMA_VERSION: u32 = 1;
/// Audited simulator rules required by stage-ten league artifacts.
pub const LEAGUE_RULES_AUDIT_VERSION: u32 = 2;
/// Canonical stage-ten frozen-policy, scheduling, retention, and promotion contract.
pub const LEAGUE_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-league/v1;",
    "action_schema_version=1;action_schema_hash=17797499074169920257;",
    "feature_schema_version=4;feature_schema_hash=508444194896722448;",
    "model_schema_version=3;model_schema_hash=6172692684479642043;",
    "ppo_schema_version=1;ppo_schema_hash=18117330041678614078;rules_audit=2;",
    "opponents=current30,accepted25,historical25,teacher15,weak5,frozen_per_rollout;",
    "league=capacity32,minimum9,protect_anchor_accepted_strongest_recent4,evict_nearest_cross_play_profile;",
    "snapshot=immutable_finite_f32_parameters,stable_parameter_fingerprint,generation;",
    "evaluation=held_out_seed_disjoint,paired_radiant_and_dire,min20,max512,horizon_max1024,truncation_draw,min_actions1000,rejections_below0.001;",
    "exploit_audit=separate_seed_namespace,min2_pairs,min100_actions,nonnegative_each_side,rejections_below0.001;",
    "promotion=opaque_paired_evidence_and_exploit_audit,positive_combined_score,nonnegative_each_side,training_reward_excluded;"
);

const LEAGUE_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const LEAGUE_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn league_fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = LEAGUE_FNV_OFFSET;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(LEAGUE_FNV_PRIME);
        index += 1;
    }
    hash
}

/// Stable FNV-1a hash of [`LEAGUE_SCHEMA_DESCRIPTOR`].
pub const LEAGUE_SCHEMA_HASH: u64 = league_fnv1a(LEAGUE_SCHEMA_DESCRIPTOR.as_bytes());

const _: () = assert!(ACTION_SCHEMA_VERSION == 1);
const _: () = assert!(ACTION_SCHEMA_HASH == 17_797_499_074_169_920_257);
const _: () = assert!(FEATURE_SCHEMA_VERSION == 4);
const _: () = assert!(FEATURE_SCHEMA_HASH == 508_444_194_896_722_448);
const _: () = assert!(MODEL_SCHEMA_VERSION == 3);
const _: () = assert!(MODEL_SCHEMA_HASH == 6_172_692_684_479_642_043);
const _: () = assert!(PPO_SCHEMA_VERSION == 1);
const _: () = assert!(PPO_SCHEMA_HASH == 18_117_330_041_678_614_078);

static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

/// Immutable finite F32 weights used by frozen opponents and accepted checkpoints.
#[derive(Clone)]
pub struct PolicySnapshot {
    id: u64,
    fingerprint: u64,
    generation: u64,
    parameters: Arc<[f32]>,
}

impl fmt::Debug for PolicySnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolicySnapshot")
            .field("id", &self.id)
            .field("fingerprint", &self.fingerprint)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl PolicySnapshot {
    /// Captures one coherent model revision into an immutable host snapshot.
    pub fn capture(model: &PolicyModel, generation: u64) -> Result<Self, LeagueError> {
        let parameters = model.export_parameters().map_err(model_error)?;
        Self::from_parameters(parameters, generation)
    }

    fn from_parameters(parameters: Vec<f32>, generation: u64) -> Result<Self, LeagueError> {
        if parameters.len() != MODEL_PARAMETER_COUNT {
            return Err(LeagueError::ParameterCount {
                actual: parameters.len(),
                expected: MODEL_PARAMETER_COUNT,
            });
        }
        if let Some((index, _)) = parameters
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(LeagueError::NonFiniteParameter { index });
        }
        let id = NEXT_SNAPSHOT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| LeagueError::SnapshotIdentityExhausted)?;
        if id == 0 {
            return Err(LeagueError::SnapshotIdentityExhausted);
        }
        let fingerprint = parameter_fingerprint(&parameters);
        Ok(Self {
            id,
            fingerprint,
            generation,
            parameters: parameters.into(),
        })
    }

    /// Materializes a private model instance for one frozen actor.
    pub fn instantiate(&self) -> Result<PolicyModel, LeagueError> {
        let model = PolicyModel::fresh(self.fingerprint ^ self.generation).map_err(model_error)?;
        model
            .import_parameters(&self.parameters)
            .map_err(model_error)?;
        Ok(model)
    }

    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }

    #[cfg(test)]
    pub(crate) fn from_parameters_for_test(
        parameters: Vec<f32>,
        generation: u64,
    ) -> Result<Self, LeagueError> {
        Self::from_parameters(parameters, generation)
    }
}

fn parameter_fingerprint(parameters: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in parameters {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Frozen-opponent family with fixed stage-ten sampling weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeagueOpponentKind {
    CurrentMirror,
    Accepted,
    Historical,
    Teacher,
    Weak,
}

impl LeagueOpponentKind {
    pub const fn index(self) -> usize {
        match self {
            Self::CurrentMirror => 0,
            Self::Accepted => 1,
            Self::Historical => 2,
            Self::Teacher => 3,
            Self::Weak => 4,
        }
    }
}

/// One selected opponent; model-backed kinds carry immutable weights.
#[derive(Clone, Debug)]
pub struct LeagueOpponent {
    kind: LeagueOpponentKind,
    snapshot: Option<PolicySnapshot>,
}

/// Deterministic hidden scheduler for weighted opponent and history selection.
pub struct LeagueSampler {
    rng: PpoRng,
}

impl LeagueSampler {
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: PpoRng::new(seed),
        }
    }

    pub fn sample(
        &mut self,
        league: &League,
        current: &PolicySnapshot,
    ) -> Result<LeagueOpponent, LeagueError> {
        let bucket = self
            .rng
            .below(100)
            .map_err(|error| LeagueError::Rng(error.to_string()))? as u8;
        if !(55..=79).contains(&bucket) {
            return league.select_bucket(bucket, current);
        }
        let history_count = league.historical_count(current.fingerprint);
        let selected = self
            .rng
            .below(history_count as u64)
            .map_err(|error| LeagueError::Rng(error.to_string()))? as usize;
        Ok(LeagueOpponent {
            kind: LeagueOpponentKind::Historical,
            snapshot: Some(
                league
                    .historical_at(selected, current.fingerprint)
                    .snapshot
                    .clone(),
            ),
        })
    }
}

impl LeagueOpponent {
    pub const fn kind(&self) -> LeagueOpponentKind {
        self.kind
    }

    pub const fn snapshot(&self) -> Option<&PolicySnapshot> {
        self.snapshot.as_ref()
    }
}

/// Three bounded cross-play axes used for diversity retention.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrossPlayProfile {
    teacher: f32,
    accepted: f32,
    historical: f32,
}

impl CrossPlayProfile {
    pub fn new(teacher: f32, accepted: f32, historical: f32) -> Result<Self, LeagueError> {
        if [teacher, accepted, historical]
            .iter()
            .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err(LeagueError::InvalidProfile);
        }
        Ok(Self {
            teacher,
            accepted,
            historical,
        })
    }

    fn distance_squared(self, other: Self) -> f32 {
        (self.teacher - other.teacher).powi(2)
            + (self.accepted - other.accepted).powi(2)
            + (self.historical - other.historical).powi(2)
    }
}

/// One retained policy and its finite held-out strength metadata.
#[derive(Clone, Debug)]
pub struct LeagueEntry {
    snapshot: PolicySnapshot,
    score: f64,
    profile: CrossPlayProfile,
    insertion: u64,
}

impl LeagueEntry {
    pub const fn snapshot(&self) -> &PolicySnapshot {
        &self.snapshot
    }
    pub const fn score(&self) -> f64 {
        self.score
    }
    pub const fn generation(&self) -> u64 {
        self.snapshot.generation
    }
    pub const fn profile(&self) -> CrossPlayProfile {
        self.profile
    }
}

/// Candidate result in one held-out match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeagueMatchResult {
    Win,
    Loss,
    Draw,
}

impl LeagueMatchResult {
    #[cfg(any(feature = "builtin", test))]
    const fn score(self) -> i32 {
        match self {
            Self::Win => 1,
            Self::Loss => -1,
            Self::Draw => 0,
        }
    }
}

/// One held-out seed played with candidate sides swapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaguePairedResult {
    pub seed: u64,
    pub candidate_radiant: LeagueMatchResult,
    pub candidate_dire: LeagueMatchResult,
}

/// Validated held-out promotion evidence for exact candidate and accepted weights.
#[derive(Clone, Debug, PartialEq)]
pub struct LeagueEvaluation {
    candidate: u64,
    accepted: u64,
    pairs: Vec<LeaguePairedResult>,
    radiant_score: i32,
    dire_score: i32,
    rejected_actions: u64,
    total_actions: u64,
    profile: CrossPlayProfile,
}

/// Opaque evaluator-produced result from the separate exploit-regression namespace.
#[cfg(any(feature = "builtin", test))]
pub(crate) struct LeagueExploitAudit {
    candidate: u64,
    accepted: u64,
}

#[cfg(any(feature = "builtin", test))]
impl LeagueExploitAudit {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        candidate: u64,
        accepted: u64,
        pairs: &[LeaguePairedResult],
        training_seeds: &[u64],
        promotion_seeds: &[u64],
        rejected_actions: u64,
        total_actions: u64,
    ) -> Result<Self, LeagueError> {
        validate_exploit_counts(pairs, rejected_actions, total_actions)?;
        validate_evaluation_seeds(pairs, training_seeds)?;
        validate_evaluation_seeds(pairs, promotion_seeds)?;
        let radiant_score = pairs
            .iter()
            .map(|pair| pair.candidate_radiant.score())
            .sum::<i32>();
        let dire_score = pairs
            .iter()
            .map(|pair| pair.candidate_dire.score())
            .sum::<i32>();
        if radiant_score < 0 || dire_score < 0 {
            return Err(LeagueError::ExploitRegression);
        }
        Ok(Self {
            candidate,
            accepted,
        })
    }
}

impl LeagueEvaluation {
    #[cfg(any(feature = "builtin", test))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        candidate: u64,
        accepted: u64,
        pairs: Vec<LeaguePairedResult>,
        training_seeds: &[u64],
        rejected_actions: u64,
        total_actions: u64,
        profile: CrossPlayProfile,
        exploit_audit: LeagueExploitAudit,
    ) -> Result<Self, LeagueError> {
        validate_evaluation_counts(&pairs, rejected_actions, total_actions)?;
        if candidate == accepted {
            return Err(LeagueError::CandidateEqualsAccepted);
        }
        if exploit_audit.candidate != candidate || exploit_audit.accepted != accepted {
            return Err(LeagueError::ExploitIdentityMismatch);
        }
        validate_evaluation_seeds(&pairs, training_seeds)?;
        let radiant_score = pairs
            .iter()
            .map(|pair| pair.candidate_radiant.score())
            .sum();
        let dire_score = pairs.iter().map(|pair| pair.candidate_dire.score()).sum();
        Ok(Self {
            candidate,
            accepted,
            pairs,
            radiant_score,
            dire_score,
            rejected_actions,
            total_actions,
            profile,
        })
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    pub fn score(&self) -> f64 {
        f64::from(self.radiant_score + self.dire_score) / (2 * self.pairs.len()) as f64
    }

    #[cfg(any(feature = "builtin", test))]
    fn promotable(&self) -> bool {
        self.radiant_score >= 0 && self.dire_score >= 0 && self.radiant_score + self.dire_score > 0
    }
}

#[cfg(any(feature = "builtin", test))]
fn validate_exploit_counts(
    pairs: &[LeaguePairedResult],
    rejected_actions: u64,
    total_actions: u64,
) -> Result<(), LeagueError> {
    if !(LEAGUE_MIN_EXPLOIT_PAIRS..=64).contains(&pairs.len()) {
        return Err(LeagueError::ExploitPairCount { count: pairs.len() });
    }
    if total_actions < LEAGUE_MIN_EXPLOIT_ACTIONS
        || rejected_actions > total_actions
        || rejected_actions.saturating_mul(1_000) >= total_actions
    {
        return Err(LeagueError::InvalidRejectionRate);
    }
    Ok(())
}

#[cfg(any(feature = "builtin", test))]
fn validate_evaluation_counts(
    pairs: &[LeaguePairedResult],
    rejected_actions: u64,
    total_actions: u64,
) -> Result<(), LeagueError> {
    if !(LEAGUE_MIN_PROMOTION_PAIRS..=LEAGUE_MAX_PROMOTION_PAIRS).contains(&pairs.len()) {
        return Err(LeagueError::PromotionPairCount { count: pairs.len() });
    }
    if total_actions < LEAGUE_MIN_PROMOTION_ACTIONS
        || rejected_actions > total_actions
        || rejected_actions.saturating_mul(1_000) >= total_actions
    {
        return Err(LeagueError::InvalidRejectionRate);
    }
    Ok(())
}

#[cfg(any(feature = "builtin", test))]
fn validate_evaluation_seeds(
    pairs: &[LeaguePairedResult],
    training_seeds: &[u64],
) -> Result<(), LeagueError> {
    if training_seeds.len() > 8_192 {
        return Err(LeagueError::TrainingSeedCapacity);
    }
    let mut previous = None;
    for pair in pairs {
        if previous.is_some_and(|seed| pair.seed <= seed) {
            return Err(LeagueError::PromotionSeedsNotIncreasing);
        }
        if training_seeds.contains(&pair.seed) {
            return Err(LeagueError::EvaluationSeedInTraining { seed: pair.seed });
        }
        previous = Some(pair.seed);
    }
    Ok(())
}

/// Result of applying valid promotion evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaguePromotionDecision {
    Accepted,
    RejectedPerformance,
    Duplicate,
}

/// Bounded historical opponent set with accepted, anchor, strength, recency, and diversity retention.
pub struct League {
    capacity: usize,
    entries: Vec<LeagueEntry>,
    anchor: u64,
    accepted: u64,
    insertion: u64,
}

impl League {
    pub fn new(capacity: usize, initial: PolicySnapshot) -> Result<Self, LeagueError> {
        if !(LEAGUE_MIN_POLICIES..=LEAGUE_MAX_POLICIES).contains(&capacity) {
            return Err(LeagueError::Capacity { capacity });
        }
        let fingerprint = initial.fingerprint;
        Ok(Self {
            capacity,
            entries: vec![LeagueEntry {
                snapshot: initial,
                score: 0.0,
                profile: CrossPlayProfile::default(),
                insertion: 0,
            }],
            anchor: fingerprint,
            accepted: fingerprint,
            insertion: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn accepted(&self) -> &PolicySnapshot {
        &self.entry(self.accepted).snapshot
    }

    pub fn strongest(&self) -> &LeagueEntry {
        self.entries
            .iter()
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| left.insertion.cmp(&right.insertion))
            })
            .expect("league always retains its anchor")
    }

    pub fn iter(&self) -> impl Iterator<Item = &LeagueEntry> {
        self.entries.iter()
    }

    pub fn contains(&self, fingerprint: u64) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.snapshot.fingerprint == fingerprint)
    }

    pub fn select_bucket(
        &self,
        bucket: u8,
        current: &PolicySnapshot,
    ) -> Result<LeagueOpponent, LeagueError> {
        if bucket >= 100 {
            return Err(LeagueError::OpponentBucket { bucket });
        }
        let (kind, snapshot) = match bucket {
            0..=29 => (LeagueOpponentKind::CurrentMirror, Some(current.clone())),
            30..=54 => (LeagueOpponentKind::Accepted, Some(self.accepted().clone())),
            55..=79 => (
                LeagueOpponentKind::Historical,
                Some(self.historical(current.fingerprint).clone()),
            ),
            80..=94 => (LeagueOpponentKind::Teacher, None),
            _ => (LeagueOpponentKind::Weak, None),
        };
        Ok(LeagueOpponent { kind, snapshot })
    }

    #[cfg(any(feature = "builtin", test))]
    pub(crate) fn try_promote(
        &mut self,
        candidate: PolicySnapshot,
        evidence: LeagueEvaluation,
    ) -> Result<LeaguePromotionDecision, LeagueError> {
        if evidence.candidate != candidate.fingerprint || evidence.accepted != self.accepted {
            return Err(LeagueError::EvaluationIdentityMismatch);
        }
        if self.contains(candidate.fingerprint) {
            return Ok(LeaguePromotionDecision::Duplicate);
        }
        let score = evidence.score();
        let profile = evidence.profile;
        let promotable = evidence.promotable();
        self.insert(candidate, score, profile, promotable)?;
        Ok(if promotable {
            LeaguePromotionDecision::Accepted
        } else {
            LeaguePromotionDecision::RejectedPerformance
        })
    }

    pub fn insert_historical(
        &mut self,
        snapshot: PolicySnapshot,
        score: f64,
        profile: CrossPlayProfile,
    ) -> Result<(), LeagueError> {
        self.insert(snapshot, score, profile, false)
    }

    fn insert(
        &mut self,
        snapshot: PolicySnapshot,
        score: f64,
        profile: CrossPlayProfile,
        make_accepted: bool,
    ) -> Result<(), LeagueError> {
        if !score.is_finite() {
            return Err(LeagueError::NonFiniteScore);
        }
        if self.contains(snapshot.fingerprint) {
            return Err(LeagueError::DuplicatePolicy);
        }
        let next_insertion = self
            .insertion
            .checked_add(1)
            .ok_or(LeagueError::InsertionOverflow)?;
        if self.entries.len() == self.capacity {
            let evicted = self.eviction_index(profile)?;
            self.entries.remove(evicted);
        }
        self.insertion = next_insertion;
        if make_accepted {
            self.accepted = snapshot.fingerprint;
        }
        self.entries.push(LeagueEntry {
            snapshot,
            score,
            profile,
            insertion: self.insertion,
        });
        Ok(())
    }

    fn eviction_index(&self, incoming: CrossPlayProfile) -> Result<usize, LeagueError> {
        let strongest = self.strongest().snapshot.fingerprint;
        let diverse = self.most_diverse().snapshot.fingerprint;
        let recent_floor = self
            .insertion
            .saturating_sub(LEAGUE_RECENT_POLICIES as u64 - 1);
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                let fingerprint = entry.snapshot.fingerprint;
                fingerprint != self.anchor
                    && fingerprint != self.accepted
                    && fingerprint != strongest
                    && fingerprint != diverse
                    && entry.insertion < recent_floor
            })
            .min_by(|(left_index, left), (right_index, right)| {
                self.profile_novelty(*left_index, incoming)
                    .total_cmp(&self.profile_novelty(*right_index, incoming))
                    .then_with(|| left.score.total_cmp(&right.score))
                    .then_with(|| left.insertion.cmp(&right.insertion))
            })
            .map(|(index, _)| index)
            .ok_or(LeagueError::NoEvictablePolicy)
    }

    fn profile_novelty(&self, index: usize, incoming: CrossPlayProfile) -> f32 {
        let profile = self.entries[index].profile;
        self.entries
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, entry)| profile.distance_squared(entry.profile))
            .chain(std::iter::once(profile.distance_squared(incoming)))
            .min_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    fn most_diverse(&self) -> &LeagueEntry {
        let accepted = self.entry(self.accepted).profile;
        self.entries
            .iter()
            .max_by(|left, right| {
                left.profile
                    .distance_squared(accepted)
                    .total_cmp(&right.profile.distance_squared(accepted))
                    .then_with(|| left.insertion.cmp(&right.insertion))
            })
            .expect("league always retains its anchor")
    }

    fn historical(&self, current: u64) -> &PolicySnapshot {
        self.historical_at(0, current).snapshot()
    }

    fn historical_count(&self, current: u64) -> usize {
        self.entries
            .iter()
            .filter(|entry| {
                entry.snapshot.fingerprint != self.accepted && entry.snapshot.fingerprint != current
            })
            .count()
            .max(1)
    }

    fn historical_at(&self, index: usize, current: u64) -> &LeagueEntry {
        self.entries
            .iter()
            .filter(|entry| {
                entry.snapshot.fingerprint != self.accepted && entry.snapshot.fingerprint != current
            })
            .nth(index)
            .unwrap_or_else(|| self.entry(self.accepted))
    }

    fn entry(&self, fingerprint: u64) -> &LeagueEntry {
        self.entries
            .iter()
            .find(|entry| entry.snapshot.fingerprint == fingerprint)
            .expect("league fingerprint is retained")
    }

    #[cfg(test)]
    pub(crate) fn insert_evaluated_for_test(
        &mut self,
        snapshot: PolicySnapshot,
        score: f64,
        profile: CrossPlayProfile,
    ) -> Result<(), LeagueError> {
        self.insert(snapshot, score, profile, false)
    }
}

/// Snapshot, league, opponent, or held-out evidence failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LeagueError {
    Capacity { capacity: usize },
    ParameterCount { actual: usize, expected: usize },
    NonFiniteParameter { index: usize },
    SnapshotIdentityExhausted,
    InvalidProfile,
    NonFiniteScore,
    DuplicatePolicy,
    OpponentBucket { bucket: u8 },
    PromotionPairCount { count: usize },
    InvalidRejectionRate,
    ExploitPairCount { count: usize },
    ExploitRegression,
    ExploitIdentityMismatch,
    CandidateEqualsAccepted,
    PromotionSeedsNotIncreasing,
    TrainingSeedCapacity,
    EvaluationSeedInTraining { seed: u64 },
    EvaluationIdentityMismatch,
    InsertionOverflow,
    NoEvictablePolicy,
    Model(String),
    Rng(String),
}

impl fmt::Display for LeagueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { capacity } => write!(
                formatter,
                "league capacity {capacity} is outside {LEAGUE_MIN_POLICIES}..={LEAGUE_MAX_POLICIES}"
            ),
            Self::ParameterCount { actual, expected } => {
                write!(
                    formatter,
                    "snapshot has {actual} parameters, expected {expected}"
                )
            }
            Self::NonFiniteParameter { index } => {
                write!(formatter, "snapshot parameter {index} is non-finite")
            }
            Self::SnapshotIdentityExhausted => {
                formatter.write_str("snapshot identity is exhausted")
            }
            Self::InvalidProfile => formatter.write_str("cross-play profile is invalid"),
            Self::NonFiniteScore => formatter.write_str("league score is non-finite"),
            Self::DuplicatePolicy => formatter.write_str("league policy weights are duplicated"),
            Self::OpponentBucket { bucket } => {
                write!(formatter, "opponent bucket {bucket} is outside 0..100")
            }
            Self::PromotionPairCount { count } => write!(
                formatter,
                "promotion has {count} pairs, expected {LEAGUE_MIN_PROMOTION_PAIRS}..={LEAGUE_MAX_PROMOTION_PAIRS}"
            ),
            Self::InvalidRejectionRate => {
                formatter.write_str("promotion rejection rate is invalid")
            }
            Self::ExploitPairCount { count } => {
                write!(
                    formatter,
                    "exploit audit has {count} pairs, expected 2..=64"
                )
            }
            Self::ExploitRegression => {
                formatter.write_str("exploit audit regresses on at least one side")
            }
            Self::ExploitIdentityMismatch => formatter.write_str("exploit audit identity mismatch"),
            Self::CandidateEqualsAccepted => {
                formatter.write_str("promotion candidate equals accepted weights")
            }
            Self::PromotionSeedsNotIncreasing => {
                formatter.write_str("promotion seeds are not strictly increasing")
            }
            Self::TrainingSeedCapacity => formatter.write_str("training seed capacity exceeded"),
            Self::EvaluationSeedInTraining { seed } => {
                write!(formatter, "promotion seed {seed} occurs in training")
            }
            Self::EvaluationIdentityMismatch => {
                formatter.write_str("promotion evidence identity mismatch")
            }
            Self::InsertionOverflow => formatter.write_str("league insertion counter overflow"),
            Self::NoEvictablePolicy => formatter.write_str("league has no evictable policy"),
            Self::Model(message) => write!(formatter, "league model error: {message}"),
            Self::Rng(message) => write!(formatter, "league RNG error: {message}"),
        }
    }
}

impl Error for LeagueError {}

fn model_error(error: ModelError) -> LeagueError {
    LeagueError::Model(error.to_string())
}
