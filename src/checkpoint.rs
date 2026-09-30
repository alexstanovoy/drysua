use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bota_proto::{HeroId, MapId};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize};
use sha2::{Digest, Sha256};

#[path = "checkpoint_inspection.rs"]
mod inspection;
pub use inspection::{checkpoint_inspect, checkpoint_inspection_contract};

#[path = "checkpoint_adaptive.rs"]
mod adaptive;
#[cfg(test)]
#[path = "tests/checkpoint_adaptive.rs"]
mod adaptive_tests;
#[cfg(test)]
#[path = "tests/checkpoint_capacity.rs"]
mod capacity_tests;
#[path = "runtime_weights.rs"]
mod runtime;
use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION,
    MAP2_REWARD_VERSION, MAX_TRAINING_COUNTER, MODEL_MAX_OPTIMIZER_STEP, MODEL_PARAMETER_COUNT,
    MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, PPO_RULES_AUDIT_VERSION, PPO_SCHEMA_HASH,
    PPO_SCHEMA_VERSION, PolicyDevice, PolicyModel, PpoConfig, PpoTrainer, SHADOW_FIEND,
};

pub use adaptive::AdaptiveEnvironmentCheckpoint;

const CHECKPOINT_MAGIC: &[u8; 8] = b"DRYCKP21";
/// Version of the strict on-disk tensor and manifest contract.
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 21;
/// Canonical strict checkpoint contract descriptor.
pub const CHECKPOINT_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-checkpoint/v21;linked_schemas=action,feature,model,ppo;linked_hash=fnv1a_descriptor_then_ordered_version_le32_hash_le64_then_map2_reward_version_le32;files=checkpoint.meta,drysua.weights.safetensors,immutable_sha256_tensor_generation;",
    "tensors=model.parameters,adam.first_moment,adam.second_moment,actor.parameters_f32,collection.state_u8_bounded;dtype=f32_except_collection_state;runtime=one_named_f32_tensor_per_model_parameter_in_export_order;runtime_metadata=action_feature_model_ppo_schema_hashes,ppo_schema_version,ppo_rules_audit_version,map2_reward_version;load=exact_names_shapes_dtype_finite_schema_sha256;",
    "initialization=runtime_weights_same_name_and_shape_tensors_reused_others_fresh,optimizer_progress_rng=fresh;",
    "manifest=magic_version_hash_linked_schemas_then_git_simulator_features_command_seed_map_hero_device_batch_rules32_then_progress_rng_curriculum_league_then_ppo_config_trainer_updates_optimizer_step_shuffle_rng_tensor_sha256_then_adaptive_presence_u8_and_optional152_byte_block,no_trailing_bytes,max65536;",
    "progress=committed_rollout_samples_le_updates_times_samples_per_update_plus_two_per_max_slot;collection=next_update_actor_version_spec_and_replayable_in_flight_slot_games;",
    "adaptive_block=le64_success_updates_success_rate_millionths_poor_updates_poor_rate_millionths_extension_millionths_base_updates_total_updates_zero_updates_generation_start_update_updates_in_generation_success_streak_poor_streak_extension_awards_snapshot_count_then_snapshot_sha256_raw32;adaptive_scope=train-annealed_only_no_league_exact_config_scope_suffix_last_once;",
    "save=immutable_generation_then_runtime_then_manifest_rename,one_fsync_per_file_then_one_directory_fsync;"
);
/// Ordered linked schema identities captured in every checkpoint manifest.
const LINKED_SCHEMAS: [(u32, u64); 4] = [
    (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
    (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
    (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
    (PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
];

/// FNV-1a of the descriptor, ordered linked identities, and reward version.
pub const CHECKPOINT_SCHEMA_HASH: u64 =
    crate::model::linked_schema_hash(CHECKPOINT_SCHEMA_DESCRIPTOR, &LINKED_SCHEMAS);
const CHECKPOINT_META_FILE: &str = "checkpoint.meta";
const RUNTIME_TENSOR_FILE: &str = "drysua.weights.safetensors";
const MAX_META_BYTES: u64 = 64 * 1024;
/// Largest encoded collection state: every slot with two full action logs.
pub(crate) const MAX_COLLECTION_STATE_BYTES: usize =
    64 + crate::PPO_MAX_SLOTS * (160 + 2 * 4 * crate::MAP2_ACTOR_DECISIONS);
pub(crate) const MAX_TRAINING_TENSOR_BYTES: u64 =
    MODEL_PARAMETER_COUNT as u64 * 16 + MAX_COLLECTION_STATE_BYTES as u64 + 64 * 1024;
const MAX_RUNTIME_TENSOR_BYTES: u64 = MODEL_PARAMETER_COUNT as u64 * 4 + 64 * 1024;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_RNG_STATES: usize = 32;
const MAX_LEAGUE_REFERENCES: usize = 32;
static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(1);

/// Cargo feature set that strict checkpoint manifests must match.
pub fn compiled_features() -> String {
    let mut features = Vec::with_capacity(3);
    if cfg!(feature = "builtin") {
        features.push("builtin");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
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
    /// Typed adaptive controller and snapshot commitment, or None for fixed schedules.
    pub adaptive_environment: Option<AdaptiveEnvironmentCheckpoint>,
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
    collection: CollectionCheckpoint,
}

/// A minimal collection state for checkpoint tests: the model's weights and one byte.
#[cfg(test)]
pub(crate) fn collection_fixture(model: &PolicyModel) -> CollectionCheckpoint {
    CollectionCheckpoint {
        actor: model.export_parameters().expect("fixture parameters"),
        state: vec![1],
    }
}

/// Collection state a resumed run starts from: the weights collecting the next
/// update and the opaque, bounded collector encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionCheckpoint {
    pub(crate) actor: Vec<f32>,
    pub(crate) state: Vec<u8>,
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
    collection: CollectionCheckpoint,
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
        collection: CollectionCheckpoint,
    ) -> Result<Self, CheckpointError> {
        validate_run(&run, model, trainer.config())?;
        validate_progress(&progress, trainer.updates(), trainer.config())?;
        adaptive::validate_scope(&run, &progress)?;
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
            collection,
            tensor_hash: [0; 32],
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// The collection state the next update starts from.
    pub fn collection(&self) -> &CollectionCheckpoint {
        &self.collection
    }

    pub fn run(&self) -> &CheckpointRun {
        &self.run
    }

    pub fn progress(&self) -> &CheckpointProgress {
        &self.progress
    }

    /// The exact optimizer configuration, including its checkpoint capacity profile.
    pub const fn config(&self) -> PpoConfig {
        self.config
    }

    /// Commits one checkpoint generation: the immutable tensor file, the runtime
    /// weights and last the manifest, each synced once, then the directory once.
    pub fn save(&self, directory: &Path) -> Result<CheckpointSaveOutcome, CheckpointError> {
        self.validate()?;
        validate_directory(directory)?;
        preflight_tensor_generations(directory)?;
        let tensor_bytes = serialize_training_tensors(self)?;
        let tensor_hash = sha256(&tensor_bytes);
        let manifest_bytes = encode_manifest(self, tensor_hash)?;
        let runtime_bytes = serialize_runtime_tensor(&self.parameters)?;
        let generation = tensor_generation_path(directory, tensor_hash);
        write_immutable(&generation, &tensor_bytes)?;
        replace_file(&directory.join(RUNTIME_TENSOR_FILE), &runtime_bytes)?;
        replace_file(&directory.join(CHECKPOINT_META_FILE), &manifest_bytes)?;
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

    /// Reads only the run scope of a checkpoint manifest.
    ///
    /// Lets a caller diagnose a scope mismatch by name before the full
    /// compatibility check rejects it.
    #[cfg(feature = "builtin")]
    pub(crate) fn load_run_scope(directory: &Path) -> Result<CheckpointRun, CheckpointError> {
        validate_directory(directory)?;
        let manifest = read_bounded(&directory.join(CHECKPOINT_META_FILE), MAX_META_BYTES)?;
        let artifact = decode_manifest(&manifest)?;
        Ok(artifact.run)
    }

    /// Validated metadata only, for read-only orchestration preflight before acquiring a lock.
    #[cfg(feature = "builtin")]
    pub(crate) fn load_resume_metadata(
        directory: &Path,
    ) -> Result<(CheckpointRun, CheckpointProgress), CheckpointError> {
        validate_directory(directory)?;
        let manifest = read_bounded(&directory.join(CHECKPOINT_META_FILE), MAX_META_BYTES)?;
        let artifact = decode_manifest(&manifest)?;
        Ok((artifact.run, artifact.progress))
    }

    fn load_inner(
        directory: &Path,
        expected: Option<&CheckpointRun>,
    ) -> Result<Self, CheckpointError> {
        validate_directory(directory)?;
        let manifest = read_bounded(&directory.join(CHECKPOINT_META_FILE), MAX_META_BYTES)?;
        let mut artifact = decode_manifest(&manifest)?;
        if expected.is_some_and(|expected| expected != &artifact.run) {
            return Err(CheckpointError::InvalidManifest("compatibility scope"));
        }
        let generation = tensor_generation_path(directory, artifact.tensor_hash);
        let tensors = read_bounded(&generation, MAX_TRAINING_TENSOR_BYTES)?;
        if sha256(&tensors) != artifact.tensor_hash {
            return Err(CheckpointError::TensorHashMismatch);
        }
        let decoded = decode_training_tensors(&tensors)?;
        artifact.parameters = decoded.parameters;
        artifact.optimizer.first_moment = decoded.first_moment;
        artifact.optimizer.second_moment = decoded.second_moment;
        artifact.collection = decoded.collection;
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

    /// Saves deployment weights with the current schema metadata.
    pub fn save_runtime_weights(
        model: &PolicyModel,
        directory: &Path,
    ) -> Result<(), CheckpointError> {
        let parameters = model
            .export_parameters()
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        save_runtime(&parameter_schema(model)?, &parameters, directory)
    }

    /// Constructs a fresh current model, reusing every runtime-weights tensor
    /// whose name and shape it still has.
    ///
    /// Unlike deployment loading, the linked action, feature, reward and PPO
    /// schemas may differ: a warm start only needs parameters. Tensors the file
    /// lacks keep their seeded initialization; the load is logged tensor by tensor.
    #[cfg(feature = "builtin")]
    pub(crate) fn initialize_from_weights(
        directory: &Path,
        seed: u64,
        device: PolicyDevice,
    ) -> Result<PolicyModel, CheckpointError> {
        let model = PolicyModel::fresh_on(seed, device)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let fresh = model
            .export_parameters()
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let warm = runtime::decode_warm_start(&bytes, &parameter_schema(&model)?, &fresh)?;
        model
            .import_parameters(&warm.parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))?;
        let listed = |names: Vec<String>| {
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(",")
            }
        };
        crate::telemetry::log_line!(
            "level=INFO event=initial_weights_loaded path={} reused_tensors={} reused_parameters={} reinitialized_tensors={} reinitialized={} dropped_tensors={} differing_metadata={}",
            directory.display(),
            warm.reused_tensors,
            warm.reused_parameters,
            warm.reinitialized.len(),
            listed(warm.reinitialized.iter().map(|name| (*name).to_owned()).collect()),
            warm.dropped_tensors,
            listed(warm.differing_metadata),
        );
        Ok(model)
    }

    /// Loads runtime weights whose metadata, names and shapes match this build exactly.
    pub fn load_runtime_weights(
        model: &PolicyModel,
        directory: &Path,
    ) -> Result<(), CheckpointError> {
        validate_directory(directory)?;
        let bytes = read_bounded(
            &directory.join(RUNTIME_TENSOR_FILE),
            MAX_RUNTIME_TENSOR_BYTES,
        )?;
        let parameters = runtime::decode_strict(&bytes, &parameter_schema(model)?)?;
        model
            .import_parameters(&parameters)
            .map_err(|error| CheckpointError::Model(error.to_string()))
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        validate_run_without_model(&self.run, self.config)?;
        validate_progress(&self.progress, self.trainer_updates, self.config)?;
        adaptive::validate_scope(&self.run, &self.progress)?;
        validate_tensor_values("model.parameters", &self.parameters)?;
        validate_tensor_values("adam.first_moment", &self.optimizer.first_moment)?;
        validate_tensor_values("adam.second_moment", &self.optimizer.second_moment)?;
        validate_tensor_values("actor.parameters", &self.collection.actor)?;
        if self.collection.actor.len() != MODEL_PARAMETER_COUNT
            || self.collection.state.is_empty()
            || self.collection.state.len() > MAX_COLLECTION_STATE_BYTES
        {
            return Err(CheckpointError::TensorContract("collection state"));
        }
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
    config
        .validate()
        .map_err(|_| CheckpointError::InvalidManifest("PPO config"))?;
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
    config: PpoConfig,
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
    let maximum_rollout_samples = maximum_rollout_samples(config, trainer_updates)?;
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

fn maximum_rollout_samples(config: PpoConfig, updates: u64) -> Result<u64, CheckpointError> {
    updates
        .checked_mul(config.rollout_capacity(crate::PPO_MAX_SLOTS) as u64)
        .ok_or(CheckpointError::InvalidManifest("rollout sample counter"))
}

fn validate_text(field: &'static str, value: &str) -> Result<(), CheckpointError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(CheckpointError::InvalidManifest(field));
    }
    Ok(())
}

fn save_runtime(
    schema: &[(&'static str, Vec<usize>)],
    parameters: &[f32],
    directory: &Path,
) -> Result<(), CheckpointError> {
    validate_directory(directory)?;
    validate_tensor_values("model.parameters", parameters)?;
    let bytes = runtime::serialize(schema, parameters)?;
    replace_file(&directory.join(RUNTIME_TENSOR_FILE), &bytes)?;
    sync_directory(directory)
}

/// Strictly decoded runtime weights, for readers without a live model.
fn decode_runtime_parameters(bytes: &[u8]) -> Result<Vec<f32>, CheckpointError> {
    runtime::decode_strict(bytes, &current_parameter_schema()?)
}

fn current_parameter_schema() -> Result<Vec<(&'static str, Vec<usize>)>, CheckpointError> {
    PolicyModel::fresh(0)
        .and_then(|model| model.parameter_schema())
        .map_err(|error| CheckpointError::Model(error.to_string()))
}

/// Runtime weights bytes of `parameters` in this build's layout.
fn serialize_runtime_tensor(parameters: &[f32]) -> Result<Vec<u8>, CheckpointError> {
    runtime::serialize(&current_parameter_schema()?, parameters)
}

fn parameter_schema(model: &PolicyModel) -> Result<Vec<(&'static str, Vec<usize>)>, CheckpointError> {
    model
        .parameter_schema()
        .map_err(|error| CheckpointError::Model(error.to_string()))
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
            ("actor.parameters", &artifact.collection.actor),
            ("adam.first_moment", &artifact.optimizer.first_moment),
            ("adam.second_moment", &artifact.optimizer.second_moment),
            ("model.parameters", &artifact.parameters),
        ],
        Some(("collection.state", &artifact.collection.state)),
    )
}

/// Appends one JSON string, escaping what the format requires.
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            plain => out.push(plain),
        }
    }
    out.push('"');
}

fn serialize_named_tensors(
    tensors: &[(&str, &[f32])],
    raw: Option<(&str, &[u8])>,
) -> Result<Vec<u8>, CheckpointError> {
    let bytes = tensors
        .iter()
        .map(|(_, values)| encode_f32(values))
        .collect::<Vec<_>>();
    let mut views = tensors
        .iter()
        .zip(&bytes)
        .map(|((name, values), bytes)| {
            TensorView::new(Dtype::F32, vec![values.len()], bytes)
                .map(|view| ((*name).to_owned(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if let Some((name, bytes)) = raw {
        views.push((
            name.to_owned(),
            TensorView::new(Dtype::U8, vec![bytes.len()], bytes)
                .map_err(|error| CheckpointError::Backend(error.to_string()))?,
        ));
    }
    serialize(views, None).map_err(|error| CheckpointError::Backend(error.to_string()))
}

fn decode_training_tensors(bytes: &[u8]) -> Result<DecodedTensors, CheckpointError> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    validate_names(
        &tensors,
        &[
            "actor.parameters",
            "adam.first_moment",
            "adam.second_moment",
            "collection.state",
            "model.parameters",
        ],
    )?;
    let state = tensors
        .tensor("collection.state")
        .map_err(|error| CheckpointError::Backend(error.to_string()))?;
    if state.dtype() != Dtype::U8
        || state.shape().len() != 1
        || state.data().is_empty()
        || state.data().len() > MAX_COLLECTION_STATE_BYTES
    {
        return Err(CheckpointError::TensorContract("collection state"));
    }
    Ok(DecodedTensors {
        parameters: decode_tensor(&tensors, "model.parameters")?,
        first_moment: decode_tensor(&tensors, "adam.first_moment")?,
        second_moment: decode_tensor(&tensors, "adam.second_moment")?,
        collection: CollectionCheckpoint {
            actor: decode_tensor(&tensors, "actor.parameters")?,
            state: state.data().to_vec(),
        },
    })
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
    writer.u8(u8::from(artifact.progress.adaptive_environment.is_some()));
    if let Some(checkpoint) = &artifact.progress.adaptive_environment {
        adaptive::encode(&mut writer, checkpoint);
    }
    if writer.bytes.len() > MAX_META_BYTES as usize {
        return Err(CheckpointError::InvalidManifest("manifest size"));
    }
    Ok(writer.bytes)
}

fn decode_manifest(bytes: &[u8]) -> Result<TrainingArtifact, CheckpointError> {
    if bytes.len() > MAX_META_BYTES as usize {
        return Err(CheckpointError::InvalidManifest("manifest size"));
    }
    let mut reader = ManifestReader::new(bytes);
    if reader.take(8)? != CHECKPOINT_MAGIC {
        return Err(CheckpointError::ManifestMagic);
    }
    if (reader.u32()?, reader.u64()?) != (CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_HASH) {
        return Err(CheckpointError::SchemaMismatch);
    }
    decode_schema(&mut reader)?;
    let run = decode_run(&mut reader)?;
    let mut progress = decode_progress(&mut reader)?;
    let config = decode_config(&mut reader)?;
    validate_run_without_model(&run, config)?;
    let trainer_updates = reader.u64()?;
    validate_progress(&progress, trainer_updates, config)?;
    let step = reader.u64()?;
    let shuffle = (reader.u64()?, reader.u64()?);
    let tensor_hash = reader.array_32()?;
    if reader.flag("adaptive environment presence")? {
        progress.adaptive_environment = Some(adaptive::decode(&mut reader)?);
    }
    reader.finish()?;
    adaptive::validate_scope(&run, &progress)?;
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
        collection: CollectionCheckpoint {
            actor: Vec::new(),
            state: Vec::new(),
        },
        tensor_hash,
    })
}

fn encode_schema(writer: &mut ManifestWriter) {
    for (version, hash) in LINKED_SCHEMAS {
        writer.u32(version);
        writer.u64(hash);
    }
}

fn decode_schema(reader: &mut ManifestReader<'_>) -> Result<(), CheckpointError> {
    for (version, hash) in LINKED_SCHEMAS {
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
    let (kind, ordinal) = match device {
        CheckpointDevice::Cpu => (0, 0),
        CheckpointDevice::Cuda { ordinal } => (1, ordinal),
    };
    writer.u8(kind);
    writer.u32(ordinal);
}

fn decode_device(reader: &mut ManifestReader<'_>) -> Result<CheckpointDevice, CheckpointError> {
    let kind = reader.u8()?;
    let ordinal = reader.u32()?;
    match (kind, ordinal) {
        (0, 0) => Ok(CheckpointDevice::Cpu),
        (1, ordinal) => Ok(CheckpointDevice::Cuda { ordinal }),
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
        adaptive_environment: None,
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
    for value in [config.samples_per_update, config.epochs, config.minibatch] {
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
    let samples_per_update = reader.u32()? as usize;
    let epochs = reader.u32()? as usize;
    let minibatch = reader.u32()? as usize;
    PpoConfig {
        decision_interval_ticks,
        samples_per_update,
        epochs,
        minibatch,
        clip_epsilon: reader.f32()?,
        value_coefficient: reader.f32()?,
        entropy_coefficient: reader.f32()?,
        learning_rate: reader.f32()?,
        adam_beta1: reader.f32()?,
        adam_beta2: reader.f32()?,
        adam_epsilon: reader.f32()?,
        gradient_clip: reader.f32()?,
        gamma_tick: reader.f32()?,
        gae_lambda: reader.f32()?,
        target_kl: reader.f32()?,
    }
    .validate()
    .map_err(|_| CheckpointError::InvalidManifest("PPO config"))
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
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn f32(&mut self) -> Result<f32, CheckpointError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn array_32(&mut self) -> Result<[u8; 32], CheckpointError> {
        self.array()
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], CheckpointError> {
        Ok(self.take(N)?.try_into().expect("exact array length"))
    }
    fn flag(&mut self, field: &'static str) -> Result<bool, CheckpointError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CheckpointError::InvalidManifest(field)),
        }
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

/// Writes a synced sibling temporary file and renames it over `path`; the
/// caller syncs the directory once for every file of one commit.
fn replace_file(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let temporary = temporary_path(path)?;
    let result = write_immutable(&temporary, bytes).and_then(|()| commit_rename(&temporary, path));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Renames a synced file into place; durable after the directory is synced.
#[cfg(not(windows))]
pub(crate) fn commit_rename(source: &Path, target: &Path) -> Result<(), CheckpointError> {
    Ok(fs::rename(source, target)?)
}

/// Renames a synced file into place with write-through.
#[cfg(windows)]
pub(crate) fn commit_rename(source: &Path, target: &Path) -> Result<(), CheckpointError> {
    windows_replace(source, target)
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
    visit_tensor_generations(directory, Some(&retained), |path| {
        fs::remove_file(path).map_err(CheckpointError::from)
    })
}

fn preflight_tensor_generations(directory: &Path) -> Result<(), CheckpointError> {
    visit_tensor_generations(directory, None, |_| Ok(()))
}

fn visit_tensor_generations(
    directory: &Path,
    retained: Option<&Path>,
    mut visit: impl FnMut(&Path) -> Result<(), CheckpointError>,
) -> Result<(), CheckpointError> {
    for (index, entry) in fs::read_dir(directory)?.enumerate() {
        if index >= 128 {
            return Err(CheckpointError::InvalidManifest("artifact file count"));
        }
        let path = entry?.path();
        if retained == Some(path.as_path()) || !is_tensor_generation(&path) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(CheckpointError::InvalidManifest("generation file type"));
        }
        visit(&path)?;
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

#[cfg(not(windows))]
pub(crate) fn sync_directory(directory: &Path) -> Result<(), CheckpointError> {
    File::open(directory)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
pub(crate) fn sync_directory(_: &Path) -> Result<(), CheckpointError> {
    // Every Windows replacement uses MOVEFILE_WRITE_THROUGH; directories cannot be opened.
    Ok(())
}
