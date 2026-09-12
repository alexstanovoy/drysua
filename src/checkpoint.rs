use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::{HeroId, MapId};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize};
use sha2::{Digest, Sha256};

#[path = "checkpoint_navigation.rs"]
mod navigation;

pub use navigation::Map2NavigationInitializationProvenance;

use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION,
    MAP2_REWARD_SCHEMA_DESCRIPTOR, MAP2_REWARD_SCHEMA_HASH, MAP2_REWARD_SCHEMA_VERSION,
    MAX_TRAINING_COUNTER, MODEL_MAX_OPTIMIZER_STEP, MODEL_PARAMETER_COUNT, MODEL_SCHEMA_HASH,
    MODEL_SCHEMA_VERSION, PPO_RULES_AUDIT_VERSION, PPO_SCHEMA_HASH, PPO_SCHEMA_VERSION,
    PolicyDevice, PolicyModel, PpoConfig, PpoTrainer, SHADOW_FIEND,
};

const CHECKPOINT_MAGIC: &[u8; 8] = b"DRYCKP18";
/// Version of the strict on-disk tensor and manifest contract.
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 5;
/// Canonical strict checkpoint contract descriptor.
pub const CHECKPOINT_SCHEMA_DESCRIPTOR: &str = "bota-drysua-checkpoint/v5;linked_schemas=action,feature,model,ppo,map2_reward;linked_hash=fnv1a_descriptor_then_ordered_version_le32_hash_le64_then_map2_reward_descriptor_utf8;files=checkpoint.safetensors,checkpoint.meta,drysua.weights.safetensors,immutable_sha256_tensor_generation;tensors=model.parameters,adam.first_moment,adam.second_moment;dtype=f32;runtime_metadata=action_feature_model_ppo_schema_hashes,ppo_schema_version,ppo_rules_audit_version,map2_reward_schema_version,map2_reward_schema_hash,map2_reward_schema_descriptor;load=exact_names_shapes_dtype_finite_schema_sha256,no_legacy_runtime_or_resume,canonical_tensor_fallback;initialization=explicit_two_pinned_m14_sources_candle_named_zero_row_insertion_or_two_pinned_m16_initial1739d280_advantagefdd4d3f2_exact_nine_key_a4_f14_m16_ppo29_rules24_reward1_full_sha_same_shape62_tensor_all_parameter_bits_new_provenance_no_gameplay_equivalence_fresh_optimizer_progress_rng_league;manifest=git_simulator_features_map2_scope_cap27900_including900_pregame_seed_device_batch_command_rules25_progress_rng_curriculum_league;observation=feature15_action5_walkable_building_landing_move_only_effects13_guarded14_inspired15_shadowraze_manual_hp_mana_restoration_reports_not_confirmed_tickregen;save=immutable_generation,canonical_copy,recoverable_manifest_commit_last,file_and_directory_fsync;";
/// FNV-1a of the descriptor, ordered linked identities, and reward descriptor.
pub const CHECKPOINT_SCHEMA_HASH: u64 = crate::model::linked_schema_hash(
    CHECKPOINT_SCHEMA_DESCRIPTOR,
    &[
        (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
        (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
        (PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
        (MAP2_REWARD_SCHEMA_VERSION, MAP2_REWARD_SCHEMA_HASH),
    ],
);
const CHECKPOINT_TENSOR_FILE: &str = "checkpoint.safetensors";
const CHECKPOINT_META_FILE: &str = "checkpoint.meta";
const RUNTIME_TENSOR_FILE: &str = "drysua.weights.safetensors";
const MAX_META_BYTES: u64 = 64 * 1024;
const MAX_TRAINING_TENSOR_BYTES: u64 = MODEL_PARAMETER_COUNT as u64 * 12 + 64 * 1024;
const MAX_RUNTIME_TENSOR_BYTES: u64 = MODEL_PARAMETER_COUNT as u64 * 4 + 16 * 1024;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_RNG_STATES: usize = 32;
const MAX_LEAGUE_REFERENCES: usize = 32;
static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(1);

const M14_PARAMETER_COUNT: usize = 1_689_076;

/// Parameter ancestry of a new, unqualified Map2 policy; never a resume/gameplay identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Map2InitializationProvenance {
    pub source_sha256: [u8; 32],
    pub source_ppo_schema_version: u32,
    pub source_ppo_rules_audit_version: u32,
}

impl Map2InitializationProvenance {
    /// Persist this in the new run's command/provenance record, not the source artifact.
    pub fn description(&self) -> String {
        let digest: String = self
            .source_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        format!(
            "INITIALIZATION_ONLY source_m14_sha256={digest} source_ppo={} source_rules={} target_f={} target_a={} target_m={} target_ppo={} target_rules={} reward_version={} reward_hash={} old_parameter_bits_preserved=true new_weights={}_positive_zero optimizer_progress_rng_league=fresh gameplay_equivalence=false qualification=false",
            self.source_ppo_schema_version,
            self.source_ppo_rules_audit_version,
            FEATURE_SCHEMA_VERSION,
            ACTION_SCHEMA_VERSION,
            MODEL_SCHEMA_VERSION,
            PPO_SCHEMA_VERSION,
            PPO_RULES_AUDIT_VERSION,
            MAP2_REWARD_SCHEMA_VERSION,
            MAP2_REWARD_SCHEMA_HASH,
            MODEL_PARAMETER_COUNT - M14_PARAMETER_COUNT,
        )
    }
}

/// Cargo feature set that strict checkpoint manifests must match.
pub fn compiled_features() -> String {
    let mut features = Vec::with_capacity(3);
    if cfg!(feature = "builtin") {
        features.push("builtin");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    if features.is_empty() {
        return "none".to_owned();
    }
    features.join(",")
}

/// Device and ordinal recorded in portable checkpoint metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointDevice {
    Cpu,
    Cuda { ordinal: u32 },
    Metal { ordinal: u32 },
}

/// Immutable run provenance required for strict artifact compatibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointRun {
    pub git_commit: String,
    pub simulator_commit: String,
    pub enabled_features: String,
    pub command_line: String,
    pub run_seed: u64,
    pub map: MapId,
    pub hero: HeroId,
    pub device: CheckpointDevice,
    pub batch_size: usize,
    pub rules_audit_version: u32,
}

/// One named deterministic RNG stream persisted without hidden simulator RNG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RngCheckpoint {
    name: String,
    state: u64,
    draws: u64,
}

impl RngCheckpoint {
    pub fn new(name: impl Into<String>, state: u64, draws: u64) -> Result<Self, CheckpointError> {
        let checkpoint = Self {
            name: name.into(),
            state,
            draws,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn state(&self) -> u64 {
        self.state
    }

    pub const fn draws(&self) -> u64 {
        self.draws
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        validate_text("RNG name", &self.name)?;
        if self.draws > MAX_TRAINING_COUNTER {
            return Err(CheckpointError::InvalidManifest("RNG draw count"));
        }
        Ok(())
    }
}

/// Scheduler, curriculum, rollout, evaluation, league, and RNG resume state.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointProgress {
    pub global_update: u64,
    pub policy_version: u64,
    pub scheduler_step: u64,
    pub curriculum_stage: u32,
    pub rollout_samples: u64,
    pub best_evaluation: Option<f64>,
    pub rng_states: Vec<RngCheckpoint>,
    pub league_references: Vec<u64>,
}

/// Strict checkpoint I/O, schema, tensor, or resume failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    Io(String),
    ManifestTruncated,
    ManifestMagic,
    ManifestTrailingBytes,
    InvalidManifest(&'static str),
    SchemaMismatch,
    TensorHashMismatch,
    TensorContract(&'static str),
    NonFiniteTensor { name: &'static str, index: usize },
    Backend(String),
    Model(String),
}

/// A checkpoint can be durable even when obsolete-generation cleanup needs retry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointSaveOutcome {
    Committed,
    CommittedWithCleanupError(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "checkpoint I/O failed: {message}"),
            Self::ManifestTruncated => formatter.write_str("checkpoint manifest is truncated"),
            Self::ManifestMagic => formatter.write_str("checkpoint manifest magic is invalid"),
            Self::ManifestTrailingBytes => {
                formatter.write_str("checkpoint manifest has trailing bytes")
            }
            Self::InvalidManifest(field) => {
                write!(formatter, "checkpoint manifest has invalid {field}")
            }
            Self::SchemaMismatch => {
                formatter.write_str("checkpoint schema does not match this build")
            }
            Self::TensorHashMismatch => {
                formatter.write_str("checkpoint tensor SHA-256 does not match manifest")
            }
            Self::TensorContract(field) => {
                write!(formatter, "checkpoint tensor contract has invalid {field}")
            }
            Self::NonFiniteTensor { name, index } => write!(
                formatter,
                "checkpoint tensor {name} contains non-finite value at {index}"
            ),
            Self::Backend(message) => write!(formatter, "checkpoint safetensors failed: {message}"),
            Self::Model(message) => write!(formatter, "checkpoint model restore failed: {message}"),
        }
    }
}

impl Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CheckpointOptimizer {
    pub(crate) first_moment: Vec<f32>,
    pub(crate) second_moment: Vec<f32>,
    pub(crate) step: u64,
}

struct DecodedTensors {
    parameters: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
}

/// Complete model, Adam, PPO, provenance, and control-plane checkpoint.
#[derive(Clone, Debug)]
pub struct TrainingArtifact {
    run: CheckpointRun,
    progress: CheckpointProgress,
    config: PpoConfig,
    trainer_updates: u64,
    shuffle: (u64, u64),
    parameters: Vec<f32>,
    optimizer: CheckpointOptimizer,
    tensor_hash: [u8; 32],
}

/// Restored optimizer plus every control-plane state required by orchestration.
pub struct RestoredTrainingState {
    trainer: PpoTrainer,
    run: CheckpointRun,
    progress: CheckpointProgress,
}

impl RestoredTrainingState {
    pub fn trainer(&self) -> &PpoTrainer {
        &self.trainer
    }

    pub fn trainer_mut(&mut self) -> &mut PpoTrainer {
        &mut self.trainer
    }

    pub fn run(&self) -> &CheckpointRun {
        &self.run
    }

    pub fn progress(&self) -> &CheckpointProgress {
        &self.progress
    }

    pub fn pipeline(
        &self,
        sample_capacity: usize,
        workers: usize,
        model: &PolicyModel,
    ) -> Result<crate::ActorLearnerPipeline, CheckpointError> {
        crate::ActorLearnerPipeline::new_at(
            sample_capacity,
            workers,
            model,
            crate::RolloutVersion::new(self.progress.policy_version),
        )
        .map_err(|error| CheckpointError::Model(error.to_string()))
    }

    pub fn into_parts(self) -> (PpoTrainer, CheckpointRun, CheckpointProgress) {
        (self.trainer, self.run, self.progress)
    }
}

impl TrainingArtifact {
    pub fn capture(
        model: &PolicyModel,
        trainer: &PpoTrainer,
        run: CheckpointRun,
        progress: CheckpointProgress,
    ) -> Result<Self, CheckpointError> {
        validate_run(&run, model, trainer.config())?;
        validate_progress(&progress, trainer.updates())?;
        let snapshot = trainer
            .checkpoint_snapshot(model)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let (first_moment, second_moment) = snapshot.adam.moments();
        let artifact = Self {
            run,
            progress,
            config: trainer.config(),
            trainer_updates: trainer.updates(),
            shuffle: trainer.rng_checkpoint(),
            parameters: snapshot.parameters,
            optimizer: CheckpointOptimizer {
                first_moment: first_moment.to_vec(),
                second_moment: second_moment.to_vec(),
                step: snapshot.adam.step(),
            },
            tensor_hash: [0; 32],
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn run(&self) -> &CheckpointRun {
        &self.run
    }

    pub fn progress(&self) -> &CheckpointProgress {
        &self.progress
    }

    /// Writes tensors and manifest via sibling temporary files, fsync, and rename.
    pub fn save(&self, directory: &Path) -> Result<CheckpointSaveOutcome, CheckpointError> {
        self.validate()?;
        validate_directory(directory)?;
        preflight_tensor_generations(directory)?;
        let tensor_bytes = serialize_training_tensors(self)?;
        let tensor_hash = sha256(&tensor_bytes);
        let manifest_bytes = encode_manifest(self, tensor_hash)?;
        let generation = tensor_generation_path(directory, tensor_hash);
        write_immutable(&generation, &tensor_bytes)?;
        atomic_replace(&directory.join(CHECKPOINT_TENSOR_FILE), &tensor_bytes)?;
        atomic_replace(&directory.join(CHECKPOINT_META_FILE), &manifest_bytes)?;
        sync_directory(directory)?;
        match prune_tensor_generations(directory, tensor_hash) {
            Ok(()) => Ok(CheckpointSaveOutcome::Committed),
            Err(error) => Ok(CheckpointSaveOutcome::CommittedWithCleanupError(
                error.to_string(),
            )),
        }
    }

    /// Loads bounded files, verifies SHA-256, then enforces exact tensor names and shapes.
    pub fn load(directory: &Path) -> Result<Self, CheckpointError> {
        Self::load_inner(directory, None)
    }

    /// Rejects an incompatible run scope before reading the much larger tensor file.
    pub fn load_compatible(
        directory: &Path,
        expected: &CheckpointRun,
    ) -> Result<Self, CheckpointError> {
        Self::load_inner(directory, Some(expected))
    }

    fn load_inner(
        directory: &Path,
        expected: Option<&CheckpointRun>,
    ) -> Result<Self, CheckpointError> {
        validate_directory(directory)?;
        let manifest = read_recoverable(&directory.join(CHECKPOINT_META_FILE), MAX_META_BYTES)?;
        let mut artifact = decode_manifest(&manifest)?;
        if expected.is_some_and(|expected| expected != &artifact.run) {
            return Err(CheckpointError::InvalidManifest("compatibility scope"));
        }
        let generation = tensor_generation_path(directory, artifact.tensor_hash);
        let tensors = if generation.exists() {
            read_bounded(&generation, MAX_TRAINING_TENSOR_BYTES)?
        } else {
            read_recoverable(
                &directory.join(CHECKPOINT_TENSOR_FILE),
                MAX_TRAINING_TENSOR_BYTES,
            )?
        };
        if sha256(&tensors) != artifact.tensor_hash {
            return Err(CheckpointError::TensorHashMismatch);
        }
        let decoded = decode_training_tensors(&tensors)?;
        artifact.parameters = decoded.parameters;
        artifact.optimizer.first_moment = decoded.first_moment;
        artifact.optimizer.second_moment = decoded.second_moment;
        artifact.validate()?;
        Ok(artifact)
    }

    /// Atomically installs parameters and optimizer ownership before rebuilding the trainer.
    pub fn restore(
        &self,
        model: &PolicyModel,
        expected: &CheckpointRun,
    ) -> Result<RestoredTrainingState, CheckpointError> {
        self.validate()?;
        if &self.run != expected {
            return Err(CheckpointError::InvalidManifest("compatibility scope"));
        }
        if !self.run.device.matches(model.device()) {
            return Err(CheckpointError::InvalidManifest("restore device"));
        }
        let adam = model
            .install_training_checkpoint(
                &self.parameters,
                self.config.adam(),
                self.optimizer.first_moment.clone(),
                self.optimizer.second_moment.clone(),
                self.optimizer.step,
            )
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let trainer =
            PpoTrainer::restore_checkpoint(self.config, adam, self.shuffle, self.trainer_updates)
                .map_err(|error| CheckpointError::Model(error.to_string()))?;
        Ok(RestoredTrainingState {
            trainer,
            run: self.run.clone(),
            progress: self.progress.clone(),
        })
    }

    /// Saves the deployment-only policy tensor with strict schema metadata.
    pub fn save_runtime_weights(
        model: &PolicyModel,
        directory: &Path,
    ) -> Result<(), CheckpointError> {
        validate_directory(directory)?;
        let parameters = model
            .export_parameters()
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        validate_tensor_values("model.parameters", &parameters)?;
        let bytes = serialize_runtime_tensor(&parameters)?;
        atomic_replace(&directory.join(RUNTIME_TENSOR_FILE), &bytes)?;
        sync_directory(directory)
    }

    /// Loads only the exact current Map2 identity. Legacy tuples are not runtime compatible.
    pub fn load_runtime_weights(
        model: &PolicyModel,
        directory: &Path,
    ) -> Result<(), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_recoverable(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let parameters = decode_runtime_tensor(&bytes)?;
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))
    }

    /// Retired M10 initializer: validates the legacy source, then rejects Map2 initialization.
    pub fn initialize_selected_m10_for_training(
        directory: &Path,
        _seed: u64,
        _device: PolicyDevice,
    ) -> Result<(PolicyModel, [u8; 32]), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_recoverable(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        decode_selected_m10_training_tensor(&bytes)?;
        Err(CheckpointError::SchemaMismatch)
    }

    /// Retired M11 initializer: validates the legacy source, then rejects Map2 initialization.
    pub fn initialize_selected_m11_for_training(
        directory: &Path,
        _seed: u64,
        _device: PolicyDevice,
    ) -> Result<(PolicyModel, [u8; 32]), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_recoverable(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        decode_selected_m11_training_tensor(&bytes)?;
        Err(CheckpointError::SchemaMismatch)
    }

    /// Retired M12 initializer: validates the legacy source, then rejects Map2 initialization.
    pub fn initialize_selected_m12_for_training(
        directory: &Path,
        _seed: u64,
        _device: PolicyDevice,
    ) -> Result<(PolicyModel, [u8; 32]), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_recoverable(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        decode_selected_m12_training_tensor(&bytes)?;
        Err(CheckpointError::SchemaMismatch)
    }

    /// INITIALIZATION ONLY from corrected M14 u4 (6348fe57...) or initial (ce17f6f1...).
    /// Checks exact paired source SHA-256/metadata, names, F32 count, and finite values.
    /// Candle inserts positive-zero input rows into the audited 62-tensor layout and
    /// preserves every source parameter bit. This creates a new policy, not M14 gameplay.
    /// No optimizer, progress, RNG history, samples, league or qualification is imported.
    /// The caller must persist the returned provenance and create fresh training state.
    /// Source files are read only; this function neither writes artifacts nor trains.
    pub fn initialize_selected_m14_for_map2(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<(PolicyModel, Map2InitializationProvenance), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let (parameters, provenance) = decode_selected_m14_map2_tensor(&bytes)?;
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let parameters = model
            .widen_m14_input_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        assert_eq!(parameters.len(), MODEL_PARAMETER_COUNT);
        assert!(model.device() == device);
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        Ok((model, provenance))
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        validate_run_without_model(&self.run, self.config)?;
        validate_progress(&self.progress, self.trainer_updates)?;
        self.config
            .validate()
            .map_err(|_| CheckpointError::InvalidManifest("PPO config"))?;
        validate_tensor_values("model.parameters", &self.parameters)?;
        validate_tensor_values("adam.first_moment", &self.optimizer.first_moment)?;
        validate_tensor_values("adam.second_moment", &self.optimizer.second_moment)?;
        if self.parameters.len() != MODEL_PARAMETER_COUNT
            || self.optimizer.first_moment.len() != MODEL_PARAMETER_COUNT
            || self.optimizer.second_moment.len() != MODEL_PARAMETER_COUNT
        {
            return Err(CheckpointError::TensorContract("element count"));
        }
        if self.optimizer.step > MODEL_MAX_OPTIMIZER_STEP
            || self.shuffle.1 > MAX_TRAINING_COUNTER
            || self
                .optimizer
                .second_moment
                .iter()
                .any(|value| *value < 0.0)
        {
            return Err(CheckpointError::InvalidManifest("optimizer or RNG state"));
        }
        Ok(())
    }
}

impl CheckpointDevice {
    pub fn from_policy(device: PolicyDevice) -> Result<Self, CheckpointError> {
        match device {
            PolicyDevice::Cpu => Ok(Self::Cpu),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            PolicyDevice::Cuda { ordinal } => Ok(Self::Cuda {
                ordinal: u32::try_from(ordinal)
                    .map_err(|_| CheckpointError::InvalidManifest("device ordinal"))?,
            }),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            PolicyDevice::Metal { ordinal } => Ok(Self::Metal {
                ordinal: u32::try_from(ordinal)
                    .map_err(|_| CheckpointError::InvalidManifest("device ordinal"))?,
            }),
        }
    }

    fn matches(self, device: PolicyDevice) -> bool {
        Self::from_policy(device).is_ok_and(|candidate| candidate == self)
    }
}

fn validate_run(
    run: &CheckpointRun,
    model: &PolicyModel,
    config: PpoConfig,
) -> Result<(), CheckpointError> {
    validate_run_without_model(run, config)?;
    if CheckpointDevice::from_policy(model.device())? != run.device {
        return Err(CheckpointError::InvalidManifest("device"));
    }
    Ok(())
}

fn validate_run_without_model(
    run: &CheckpointRun,
    config: PpoConfig,
) -> Result<(), CheckpointError> {
    for (field, value) in [
        ("git commit", &run.git_commit),
        ("simulator commit", &run.simulator_commit),
        ("enabled features", &run.enabled_features),
        ("command line", &run.command_line),
    ] {
        validate_text(field, value)?;
    }
    if run.hero != SHADOW_FIEND || run.map != MapId(2) {
        return Err(CheckpointError::InvalidManifest("hero or map scope"));
    }
    if run.rules_audit_version != PPO_RULES_AUDIT_VERSION || run.batch_size != config.minibatch {
        return Err(CheckpointError::InvalidManifest(
            "rules audit or batch size",
        ));
    }
    if config.gamma_tick != crate::MAP2_REWARD_GAMMA_TICK {
        return Err(CheckpointError::InvalidManifest("Map2 reward discount"));
    }
    if run.enabled_features != compiled_features() {
        return Err(CheckpointError::InvalidManifest("enabled features"));
    }
    Ok(())
}

fn validate_progress(
    progress: &CheckpointProgress,
    trainer_updates: u64,
) -> Result<(), CheckpointError> {
    if progress.global_update != trainer_updates || trainer_updates > MAX_TRAINING_COUNTER {
        return Err(CheckpointError::InvalidManifest("global update"));
    }
    if progress.scheduler_step > MAX_TRAINING_COUNTER {
        return Err(CheckpointError::InvalidManifest("scheduler step"));
    }
    if progress.policy_version > trainer_updates.saturating_add(1)
        || progress.curriculum_stage > 1_024
    {
        return Err(CheckpointError::InvalidManifest(
            "policy or curriculum counter",
        ));
    }
    let maximum_rollout_samples = trainer_updates
        .saturating_add(1)
        .saturating_mul(crate::PPO_MAX_SAMPLES as u64);
    if progress.rollout_samples > maximum_rollout_samples {
        return Err(CheckpointError::InvalidManifest("rollout sample counter"));
    }
    if progress
        .best_evaluation
        .is_some_and(|value| !value.is_finite())
    {
        return Err(CheckpointError::InvalidManifest("best evaluation"));
    }
    if progress.rng_states.len() > MAX_RNG_STATES
        || progress.league_references.len() > MAX_LEAGUE_REFERENCES
    {
        return Err(CheckpointError::InvalidManifest("bounded collection"));
    }
    let mut names = BTreeSet::new();
    for rng in &progress.rng_states {
        rng.validate()?;
        if !names.insert(rng.name()) {
            return Err(CheckpointError::InvalidManifest("duplicate RNG name"));
        }
    }
    let mut references = BTreeSet::new();
    if progress
        .league_references
        .iter()
        .any(|reference| *reference == 0 || !references.insert(*reference))
    {
        return Err(CheckpointError::InvalidManifest("league references"));
    }
    Ok(())
}

fn validate_text(field: &'static str, value: &str) -> Result<(), CheckpointError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(CheckpointError::InvalidManifest(field));
    }
    Ok(())
}

fn validate_tensor_values(name: &'static str, values: &[f32]) -> Result<(), CheckpointError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CheckpointError::NonFiniteTensor { name, index });
    }
    Ok(())
}

fn serialize_training_tensors(artifact: &TrainingArtifact) -> Result<Vec<u8>, CheckpointError> {
    serialize_named_tensors(
        &[
            ("adam.first_moment", &artifact.optimizer.first_moment),
            ("adam.second_moment", &artifact.optimizer.second_moment),
            ("model.parameters", &artifact.parameters),
        ],
        None,
    )
}

fn serialize_runtime_tensor(parameters: &[f32]) -> Result<Vec<u8>, CheckpointError> {
    serialize_named_tensors(
        &[("model.parameters", parameters)],
        Some(runtime_tensor_metadata()),
    )
}

fn runtime_tensor_metadata() -> HashMap<String, String> {
    HashMap::from([
        (
            "action_schema_hash".to_owned(),
            ACTION_SCHEMA_HASH.to_string(),
        ),
        (
            "feature_schema_hash".to_owned(),
            FEATURE_SCHEMA_HASH.to_string(),
        ),
        (
            "model_schema_hash".to_owned(),
            MODEL_SCHEMA_HASH.to_string(),
        ),
        (
            "ppo_rules_audit_version".to_owned(),
            PPO_RULES_AUDIT_VERSION.to_string(),
        ),
        ("ppo_schema_hash".to_owned(), PPO_SCHEMA_HASH.to_string()),
        (
            "ppo_schema_version".to_owned(),
            PPO_SCHEMA_VERSION.to_string(),
        ),
        (
            "map2_reward_schema_version".to_owned(),
            MAP2_REWARD_SCHEMA_VERSION.to_string(),
        ),
        (
            "map2_reward_schema_hash".to_owned(),
            MAP2_REWARD_SCHEMA_HASH.to_string(),
        ),
        (
            "map2_reward_schema_descriptor".to_owned(),
            MAP2_REWARD_SCHEMA_DESCRIPTOR.to_owned(),
        ),
    ])
}

fn serialize_named_tensors(
    tensors: &[(&str, &[f32])],
    metadata: Option<HashMap<String, String>>,
) -> Result<Vec<u8>, CheckpointError> {
    let bytes = tensors
        .iter()
        .map(|(_, values)| encode_f32(values))
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&bytes)
        .map(|((name, values), bytes)| {
            TensorView::new(Dtype::F32, vec![values.len()], bytes)
                .map(|view| ((*name).to_owned(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    serialize(views, metadata).map_err(|error| CheckpointError::Backend(error.to_string()))
}

fn decode_training_tensors(bytes: &[u8]) -> Result<DecodedTensors, CheckpointError> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(
        &tensors,
        &[
            "adam.first_moment",
            "adam.second_moment",
            "model.parameters",
        ],
    )?;
    Ok(DecodedTensors {
        parameters: decode_tensor(&tensors, "model.parameters")?,
        first_moment: decode_tensor(&tensors, "adam.first_moment")?,
        second_moment: decode_tensor(&tensors, "adam.second_moment")?,
    })
}

fn decode_runtime_tensor(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    let expected = runtime_tensor_metadata();
    if metadata.metadata().as_ref() != Some(&expected) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    decode_tensor(&tensors, "model.parameters")
}

fn selected_m14_map2_metadata(corrected_u4: bool) -> HashMap<String, String> {
    let (version, hash, rules) = if corrected_u4 {
        ("27", "9274275648898675046", "22")
    } else {
        ("26", "4420330489262074980", "21")
    };
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "1577122233561586211"),
        ("model_schema_hash", "7970187849195607202"),
        ("ppo_schema_version", version),
        ("ppo_schema_hash", hash),
        ("ppo_rules_audit_version", rules),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

fn decode_selected_m14_map2_tensor(
    bytes: &[u8],
) -> Result<(Vec<f32>, Map2InitializationProvenance), CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    let (selected, version, rules) =
        if metadata.metadata().as_ref() == Some(&selected_m14_map2_metadata(true)) {
            (
                [
                    0x63, 0x48, 0xfe, 0x57, 0xa1, 0x28, 0xeb, 0xd5, 0x21, 0xdb, 0xa6, 0x8d, 0xa0,
                    0x94, 0x9c, 0xeb, 0x73, 0x78, 0xd3, 0xe0, 0xd6, 0xb7, 0xd6, 0xa7, 0xce, 0x9a,
                    0x5f, 0xb6, 0x44, 0x65, 0x47, 0xab,
                ],
                27,
                22,
            )
        } else if metadata.metadata().as_ref() == Some(&selected_m14_map2_metadata(false)) {
            (
                [
                    0xce, 0x17, 0xf6, 0xf1, 0xe3, 0x66, 0x88, 0x37, 0x77, 0x6b, 0x36, 0x6f, 0xdc,
                    0xab, 0x59, 0xc2, 0x50, 0xa4, 0xd6, 0xbb, 0x6d, 0xe3, 0xba, 0x36, 0x15, 0xdc,
                    0x3e, 0x7f, 0x23, 0xa5, 0x3c, 0x4b,
                ],
                26,
                21,
            )
        } else {
            return Err(CheckpointError::SchemaMismatch);
        };
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let parameters = decode_tensor_count(&tensors, "model.parameters", M14_PARAMETER_COUNT)?;
    let digest = sha256(bytes);
    if digest != selected {
        return Err(CheckpointError::TensorContract(
            "selected M14 Map2 initialization source SHA-256",
        ));
    }
    assert_eq!(parameters.len(), M14_PARAMETER_COUNT);
    assert_eq!(digest, selected);
    Ok((
        parameters,
        Map2InitializationProvenance {
            source_sha256: digest,
            source_ppo_schema_version: version,
            source_ppo_rules_audit_version: rules,
        },
    ))
}

fn selected_m12_training_metadata(skill_alpha050: bool) -> HashMap<String, String> {
    let (version, hash, rules) = if skill_alpha050 {
        ("25", "12302688747093836273", "20")
    } else {
        ("23", "765990392710687046", "18")
    };
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "8078516161541333175"),
        ("model_schema_hash", "17156054387874206897"),
        ("ppo_schema_version", version),
        ("ppo_schema_hash", hash),
        ("ppo_rules_audit_version", rules),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

fn decode_selected_m12_training_tensor(
    bytes: &[u8],
) -> Result<(Vec<f32>, [u8; 32]), CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    let selected = if metadata.metadata().as_ref() == Some(&selected_m12_training_metadata(false)) {
        [
            0xad, 0xbe, 0xcb, 0x82, 0x93, 0x54, 0x8a, 0xd1, 0x60, 0x2b, 0x24, 0xb0, 0x46, 0xb9,
            0xa6, 0xf0, 0x32, 0xfc, 0x62, 0x79, 0x8d, 0x46, 0x10, 0x20, 0x29, 0xa1, 0xf3, 0x44,
            0x8d, 0x76, 0x4b, 0xdd,
        ]
    } else if metadata.metadata().as_ref() == Some(&selected_m12_training_metadata(true)) {
        [
            0xd2, 0x2a, 0x01, 0x1f, 0x82, 0x9c, 0x23, 0x59, 0x3b, 0xbc, 0x6c, 0x03, 0xd2, 0x2c,
            0xba, 0xb9, 0xab, 0x13, 0xbb, 0x79, 0x66, 0x29, 0x59, 0xed, 0xa2, 0xd6, 0xc9, 0x6b,
            0x9d, 0xe8, 0x09, 0x8e,
        ]
    } else {
        return Err(CheckpointError::SchemaMismatch);
    };
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let parameters = decode_tensor_count(&tensors, "model.parameters", M14_PARAMETER_COUNT)?;
    let digest = sha256(bytes);
    if digest != selected {
        return Err(CheckpointError::TensorContract(
            "selected M12 training source SHA-256",
        ));
    }
    Ok((parameters, digest))
}

fn selected_m11_training_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "15519817897416174399"),
        ("model_schema_hash", "18229126264156367519"),
        ("ppo_schema_version", "21"),
        ("ppo_schema_hash", "768751058595344501"),
        ("ppo_rules_audit_version", "17"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

fn decode_selected_m11_training_tensor(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if metadata.metadata().as_ref() != Some(&selected_m11_training_metadata()) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let parameters = decode_tensor_count(&tensors, "model.parameters", 1_684_724)?;
    let selected = [
        0x5b, 0xbb, 0x88, 0x43, 0xfe, 0xc8, 0x8f, 0x3c, 0x64, 0x43, 0x61, 0x8d, 0xe9, 0xca, 0xbe,
        0xba, 0x44, 0xff, 0x9d, 0xbb, 0x0b, 0x7f, 0x9a, 0x98, 0x56, 0xc2, 0xc8, 0x93, 0x65, 0x51,
        0xe8, 0x80,
    ];
    if sha256(bytes) != selected {
        return Err(CheckpointError::TensorContract(
            "selected M11 training source SHA-256",
        ));
    }
    Ok(parameters)
}

fn decode_selected_m10_training_tensor(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    let expected: HashMap<_, _> = [
        ("action_schema_hash", "1755359086494840931"),
        ("feature_schema_hash", "9669721049329356661"),
        ("model_schema_hash", "720439888929233033"),
        ("ppo_schema_version", "18"),
        ("ppo_schema_hash", "6877503070358232325"),
        ("ppo_rules_audit_version", "15"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect();
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if metadata.metadata().as_ref() != Some(&expected) {
        return Err(CheckpointError::SchemaMismatch);
    }
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(&tensors, &["model.parameters"])?;
    let parameters = decode_tensor_count(&tensors, "model.parameters", 1_684_724)?;
    let selected = [
        0xb3, 0x80, 0x26, 0x42, 0xb3, 0x44, 0x87, 0xd6, 0x6f, 0xc3, 0xf0, 0xfe, 0x52, 0x6e, 0x7f,
        0x8b, 0x2a, 0x84, 0xdf, 0x54, 0x27, 0x93, 0xac, 0x33, 0x6b, 0x58, 0xea, 0x04, 0x6b, 0x8f,
        0x8b, 0x53,
    ];
    if sha256(bytes) != selected {
        return Err(CheckpointError::TensorContract(
            "selected M10 training source SHA-256",
        ));
    }
    Ok(parameters)
}

fn validate_names(tensors: &SafeTensors<'_>, expected: &[&str]) -> Result<(), CheckpointError> {
    let mut actual = tensors.names();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(CheckpointError::TensorContract("names"));
    }
    Ok(())
}

fn decode_tensor(
    tensors: &SafeTensors<'_>,
    name: &'static str,
) -> Result<Vec<f32>, CheckpointError> {
    decode_tensor_count(tensors, name, MODEL_PARAMETER_COUNT)
}

fn decode_tensor_count(
    tensors: &SafeTensors<'_>,
    name: &'static str,
    count: usize,
) -> Result<Vec<f32>, CheckpointError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if tensor.dtype() != Dtype::F32 || tensor.shape() != [count] {
        return Err(CheckpointError::TensorContract("dtype or shape"));
    }
    let (chunks, remainder) = tensor.data().as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(CheckpointError::TensorContract("byte alignment"));
    }
    let values = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    validate_tensor_values(name, &values)?;
    Ok(values)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn encode_manifest(
    artifact: &TrainingArtifact,
    tensor_hash: [u8; 32],
) -> Result<Vec<u8>, CheckpointError> {
    let mut writer = ManifestWriter::default();
    writer.bytes.extend(CHECKPOINT_MAGIC);
    writer.u32(CHECKPOINT_SCHEMA_VERSION);
    writer.u64(CHECKPOINT_SCHEMA_HASH);
    encode_schema(&mut writer);
    encode_run(&mut writer, &artifact.run)?;
    encode_progress(&mut writer, &artifact.progress)?;
    encode_config(&mut writer, artifact.config)?;
    writer.u64(artifact.trainer_updates);
    writer.u64(artifact.optimizer.step);
    writer.u64(artifact.shuffle.0);
    writer.u64(artifact.shuffle.1);
    writer.bytes.extend(tensor_hash);
    Ok(writer.bytes)
}

fn decode_manifest(bytes: &[u8]) -> Result<TrainingArtifact, CheckpointError> {
    let mut reader = ManifestReader::new(bytes);
    if reader.take(8)? != CHECKPOINT_MAGIC {
        return Err(CheckpointError::ManifestMagic);
    }
    if reader.u32()? != CHECKPOINT_SCHEMA_VERSION || reader.u64()? != CHECKPOINT_SCHEMA_HASH {
        return Err(CheckpointError::SchemaMismatch);
    }
    decode_schema(&mut reader)?;
    let run = decode_run(&mut reader)?;
    let progress = decode_progress(&mut reader)?;
    let config = decode_config(&mut reader)?;
    let trainer_updates = reader.u64()?;
    let step = reader.u64()?;
    let shuffle = (reader.u64()?, reader.u64()?);
    let tensor_hash = reader.array_32()?;
    reader.finish()?;
    Ok(TrainingArtifact {
        run,
        progress,
        config,
        trainer_updates,
        shuffle,
        parameters: Vec::new(),
        optimizer: CheckpointOptimizer {
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            step,
        },
        tensor_hash,
    })
}

fn encode_schema(writer: &mut ManifestWriter) {
    for (version, hash) in [
        (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
        (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
        (PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
        (MAP2_REWARD_SCHEMA_VERSION, MAP2_REWARD_SCHEMA_HASH),
    ] {
        writer.u32(version);
        writer.u64(hash);
    }
}

fn decode_schema(reader: &mut ManifestReader<'_>) -> Result<(), CheckpointError> {
    for (version, hash) in [
        (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
        (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
        (PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
        (MAP2_REWARD_SCHEMA_VERSION, MAP2_REWARD_SCHEMA_HASH),
    ] {
        if reader.u32()? != version || reader.u64()? != hash {
            return Err(CheckpointError::SchemaMismatch);
        }
    }
    Ok(())
}

fn encode_run(writer: &mut ManifestWriter, run: &CheckpointRun) -> Result<(), CheckpointError> {
    writer.string(&run.git_commit)?;
    writer.string(&run.simulator_commit)?;
    writer.string(&run.enabled_features)?;
    writer.string(&run.command_line)?;
    writer.u64(run.run_seed);
    writer.u16(run.map.0);
    writer.u16(run.hero.0);
    encode_device(writer, run.device);
    writer.u32(
        u32::try_from(run.batch_size)
            .map_err(|_| CheckpointError::InvalidManifest("batch size"))?,
    );
    writer.u32(run.rules_audit_version);
    Ok(())
}

fn decode_run(reader: &mut ManifestReader<'_>) -> Result<CheckpointRun, CheckpointError> {
    Ok(CheckpointRun {
        git_commit: reader.string()?,
        simulator_commit: reader.string()?,
        enabled_features: reader.string()?,
        command_line: reader.string()?,
        run_seed: reader.u64()?,
        map: MapId(reader.u16()?),
        hero: HeroId(reader.u16()?),
        device: decode_device(reader)?,
        batch_size: reader.u32()? as usize,
        rules_audit_version: reader.u32()?,
    })
}

fn encode_device(writer: &mut ManifestWriter, device: CheckpointDevice) {
    match device {
        CheckpointDevice::Cpu => {
            writer.u8(0);
            writer.u32(0);
        }
        CheckpointDevice::Cuda { ordinal } => {
            writer.u8(1);
            writer.u32(ordinal);
        }
        CheckpointDevice::Metal { ordinal } => {
            writer.u8(2);
            writer.u32(ordinal);
        }
    }
}

fn decode_device(reader: &mut ManifestReader<'_>) -> Result<CheckpointDevice, CheckpointError> {
    let kind = reader.u8()?;
    let ordinal = reader.u32()?;
    match (kind, ordinal) {
        (0, 0) => Ok(CheckpointDevice::Cpu),
        (1, ordinal) => Ok(CheckpointDevice::Cuda { ordinal }),
        (2, ordinal) => Ok(CheckpointDevice::Metal { ordinal }),
        _ => Err(CheckpointError::InvalidManifest("device")),
    }
}

fn encode_progress(
    writer: &mut ManifestWriter,
    progress: &CheckpointProgress,
) -> Result<(), CheckpointError> {
    writer.u64(progress.global_update);
    writer.u64(progress.policy_version);
    writer.u64(progress.scheduler_step);
    writer.u32(progress.curriculum_stage);
    writer.u64(progress.rollout_samples);
    writer.option_f64(progress.best_evaluation);
    writer.u8(progress.rng_states.len() as u8);
    for rng in &progress.rng_states {
        writer.string(&rng.name)?;
        writer.u64(rng.state);
        writer.u64(rng.draws);
    }
    writer.u8(progress.league_references.len() as u8);
    for reference in &progress.league_references {
        writer.u64(*reference);
    }
    Ok(())
}

fn decode_progress(reader: &mut ManifestReader<'_>) -> Result<CheckpointProgress, CheckpointError> {
    let global_update = reader.u64()?;
    let policy_version = reader.u64()?;
    let scheduler_step = reader.u64()?;
    let curriculum_stage = reader.u32()?;
    let rollout_samples = reader.u64()?;
    let best_evaluation = reader.option_f64()?;
    let rng_count = reader.bounded_count(MAX_RNG_STATES)?;
    let mut rng_states = Vec::with_capacity(rng_count);
    for _ in 0..rng_count {
        rng_states.push(RngCheckpoint::new(
            reader.string()?,
            reader.u64()?,
            reader.u64()?,
        )?);
    }
    let league_count = reader.bounded_count(MAX_LEAGUE_REFERENCES)?;
    let mut league_references = Vec::with_capacity(league_count);
    for _ in 0..league_count {
        league_references.push(reader.u64()?);
    }
    Ok(CheckpointProgress {
        global_update,
        policy_version,
        scheduler_step,
        curriculum_stage,
        rollout_samples,
        best_evaluation,
        rng_states,
        league_references,
    })
}

fn encode_config(writer: &mut ManifestWriter, config: PpoConfig) -> Result<(), CheckpointError> {
    writer.u32(config.decision_interval_ticks);
    for value in [
        config.rollout_decisions,
        config.environments,
        config.epochs,
        config.minibatch,
    ] {
        writer.u32(
            u32::try_from(value).map_err(|_| CheckpointError::InvalidManifest("PPO dimension"))?,
        );
    }
    for value in config_floats(config) {
        writer.f32(value);
    }
    Ok(())
}

fn decode_config(reader: &mut ManifestReader<'_>) -> Result<PpoConfig, CheckpointError> {
    let decision_interval_ticks = reader.u32()?;
    let rollout_decisions = reader.u32()? as usize;
    let environments = reader.u32()? as usize;
    let epochs = reader.u32()? as usize;
    let minibatch = reader.u32()? as usize;
    let values = std::array::from_fn::<_, 11, _>(|_| reader.f32())
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PpoConfig {
        decision_interval_ticks,
        rollout_decisions,
        environments,
        epochs,
        minibatch,
        clip_epsilon: values[0],
        value_coefficient: values[1],
        entropy_coefficient: values[2],
        learning_rate: values[3],
        adam_beta1: values[4],
        adam_beta2: values[5],
        adam_epsilon: values[6],
        gradient_clip: values[7],
        gamma_tick: values[8],
        gae_lambda: values[9],
        target_kl: values[10],
    })
}

fn config_floats(config: PpoConfig) -> [f32; 11] {
    [
        config.clip_epsilon,
        config.value_coefficient,
        config.entropy_coefficient,
        config.learning_rate,
        config.adam_beta1,
        config.adam_beta2,
        config.adam_epsilon,
        config.gradient_clip,
        config.gamma_tick,
        config.gae_lambda,
        config.target_kl,
    ]
}

#[derive(Default)]
struct ManifestWriter {
    bytes: Vec<u8>,
}

impl ManifestWriter {
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn u16(&mut self, value: u16) {
        self.bytes.extend(value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes.extend(value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend(value.to_le_bytes());
    }
    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }
    fn option_f64(&mut self, value: Option<f64>) {
        self.u8(u8::from(value.is_some()));
        self.u64(value.unwrap_or_default().to_bits());
    }
    fn string(&mut self, value: &str) -> Result<(), CheckpointError> {
        validate_text("text", value)?;
        self.u16(
            u16::try_from(value.len())
                .map_err(|_| CheckpointError::InvalidManifest("text length"))?,
        );
        self.bytes.extend(value.as_bytes());
        Ok(())
    }
}

struct ManifestReader<'data> {
    bytes: &'data [u8],
    offset: usize,
}

impl<'data> ManifestReader<'data> {
    const fn new(bytes: &'data [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, count: usize) -> Result<&'data [u8], CheckpointError> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(CheckpointError::ManifestTruncated)?;
        let output = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(output)
    }
    fn u8(&mut self) -> Result<u8, CheckpointError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, CheckpointError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn f32(&mut self) -> Result<f32, CheckpointError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn array_32(&mut self) -> Result<[u8; 32], CheckpointError> {
        Ok(self.take(32)?.try_into().expect("32 bytes"))
    }
    fn string(&mut self) -> Result<String, CheckpointError> {
        let count = self.u16()? as usize;
        if count == 0 || count > MAX_TEXT_BYTES {
            return Err(CheckpointError::InvalidManifest("text length"));
        }
        String::from_utf8(self.take(count)?.to_vec())
            .map_err(|_| CheckpointError::InvalidManifest("UTF-8 text"))
    }
    fn option_f64(&mut self) -> Result<Option<f64>, CheckpointError> {
        let present = self.u8()?;
        let value = f64::from_bits(self.u64()?);
        match present {
            0 if value == 0.0 => Ok(None),
            1 => Ok(Some(value)),
            _ => Err(CheckpointError::InvalidManifest("optional float")),
        }
    }
    fn bounded_count(&mut self, maximum: usize) -> Result<usize, CheckpointError> {
        let count = self.u8()? as usize;
        if count > maximum {
            return Err(CheckpointError::InvalidManifest("collection count"));
        }
        Ok(count)
    }
    fn finish(self) -> Result<(), CheckpointError> {
        if self.offset != self.bytes.len() {
            return Err(CheckpointError::ManifestTrailingBytes);
        }
        Ok(())
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn validate_directory(directory: &Path) -> Result<(), CheckpointError> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CheckpointError::InvalidManifest("checkpoint directory"));
    }
    Ok(())
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, CheckpointError> {
    let path_metadata = fs::symlink_metadata(path)?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(CheckpointError::InvalidManifest("artifact file type"));
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() == 0 || metadata.len() > maximum {
        return Err(CheckpointError::InvalidManifest("file size"));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| CheckpointError::InvalidManifest("file size"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() != capacity {
        return Err(CheckpointError::InvalidManifest("file length"));
    }
    Ok(bytes)
}

fn read_recoverable(path: &Path, maximum: u64) -> Result<Vec<u8>, CheckpointError> {
    if path.exists() {
        return read_bounded(path, maximum);
    }
    read_bounded(&backup_path(path)?, maximum)
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    if path.exists() {
        let existing = read_bounded(path, bytes.len() as u64)?;
        if existing != bytes {
            return Err(CheckpointError::TensorHashMismatch);
        }
        return Ok(());
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(not(windows))]
fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let temporary = temporary_path(path)?;
    let backup = backup_path(path)?;
    let result = (|| -> Result<(), CheckpointError> {
        write_immutable(&temporary, bytes)?;
        if path.exists() {
            if backup.exists() {
                fs::remove_file(&backup)?;
            }
            fs::rename(path, &backup)?;
            sync_parent(path)?;
        }
        fs::rename(&temporary, path)?;
        sync_parent(path)?;
        if backup.exists() {
            fs::remove_file(&backup)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(windows)]
fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let temporary = temporary_path(path)?;
    let result = (|| -> Result<(), CheckpointError> {
        write_immutable(&temporary, bytes)?;
        windows_replace(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(windows)]
fn windows_replace(source: &Path, target: &Path) -> Result<(), CheckpointError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let target = target
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    // SAFETY: Both UTF-16 paths are NUL-terminated and remain alive for the call.
    if unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), flags) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

fn tensor_generation_path(directory: &Path, hash: [u8; 32]) -> PathBuf {
    let mut encoded = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing into String cannot fail");
    }
    directory.join(format!("checkpoint.{encoded}.safetensors"))
}

fn prune_tensor_generations(directory: &Path, retained: [u8; 32]) -> Result<(), CheckpointError> {
    let retained = tensor_generation_path(directory, retained);
    for (index, entry) in fs::read_dir(directory)?.enumerate() {
        if index >= 128 {
            return Err(CheckpointError::InvalidManifest("artifact file count"));
        }
        let entry = entry?;
        let path = entry.path();
        if path == retained || !is_tensor_generation(&path) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(CheckpointError::InvalidManifest("generation file type"));
        }
        fs::remove_file(path)?;
    }
    Ok(())
}

fn preflight_tensor_generations(directory: &Path) -> Result<(), CheckpointError> {
    for (index, entry) in fs::read_dir(directory)?.enumerate() {
        if index >= 128 {
            return Err(CheckpointError::InvalidManifest("artifact file count"));
        }
        let path = entry?.path();
        if !is_tensor_generation(&path) {
            continue;
        }
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(CheckpointError::InvalidManifest("generation file type"));
        }
    }
    Ok(())
}

fn is_tensor_generation(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("checkpoint."))
        .and_then(|name| name.strip_suffix(".safetensors"))
        .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn backup_path(path: &Path) -> Result<PathBuf, CheckpointError> {
    let name = artifact_name(path)?;
    Ok(path.with_file_name(format!("{name}.previous")))
}

fn temporary_path(path: &Path) -> Result<PathBuf, CheckpointError> {
    let name = artifact_name(path)?;
    let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
    Ok(path.with_file_name(format!("{name}.tmp-{}-{sequence}", std::process::id())))
}

fn artifact_name(path: &Path) -> Result<&str, CheckpointError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or(CheckpointError::InvalidManifest("artifact filename"))
}

fn sync_parent(path: &Path) -> Result<(), CheckpointError> {
    let parent = path
        .parent()
        .ok_or(CheckpointError::InvalidManifest("artifact parent"))?;
    sync_directory(parent)
}

#[cfg(not(windows))]
fn sync_directory(directory: &Path) -> Result<(), CheckpointError> {
    File::open(directory)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn sync_directory(_: &Path) -> Result<(), CheckpointError> {
    // MoveFileExW with MOVEFILE_WRITE_THROUGH makes each Windows replacement durable.
    Ok(())
}
