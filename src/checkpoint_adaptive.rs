use super::{
    CHECKPOINT_ANNEALED_SCHEMA_HASH, CHECKPOINT_SCHEMA_HASH, CHECKPOINT_WIDE_ANNEALED_SCHEMA_HASH,
    CheckpointError, CheckpointProgress, CheckpointRun, MAX_TEXT_BYTES, ManifestReader,
    ManifestWriter, checkpoint_profile_hash,
};
use crate::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
    EnvironmentDecimal, MAX_TRAINING_COUNTER, PpoConfig, PpoError, PpoSampleBudget,
};

const BLOCK_BYTES: usize = 152;
const SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-checkpoint/adaptive-v1;versions=15_standard_16_annealed_17_wide_annealed;",
    "linked=action_feature_model_capacity_ppo_map2_reward_matching_fixed_checkpoint12or13or14;",
    "base=exact_fixed_manifest_and_tensor_contract_except_checkpoint_identity;runtime_unchanged;",
    "manifest=append_after_tensor_sha256_required152bytes_no_presence_tag_no_padding_no_trailing;",
    "block=le64_success_updates_success_rate_millionths_poor_updates_poor_rate_millionths_extension_millionths_base_updates_total_updates_zero_updates_generation_start_update_updates_in_generation_success_streak_poor_streak_extension_awards_snapshot_count_then_snapshot_sha256_raw32;",
    "scope=train-annealed_only_no_mastery_or_league_exact_config_scope_suffix_last_once;limits=canonical_bounded_updates_zero_updates_generation_games_equals_base_updates_times_ppo_environments_games_equals_ppo_environments;",
    "state=adaptive_environment_validate_config_limits_global_update;count=generation_plus_spent_nonzero;hash_zero_iff_count_zero;hash_owned_by_generation_runtime_no_snapshot_io;",
    "defaults=2_800000_1_200000_750000;metadata_max65536;validate_before_tensor_io_capture_or_restore_mutation;commit=model_adam_rng_and_adaptive_state_manifest_last;"
);
const IDENTITIES: [(u32, u64); 3] = [
    (
        15,
        checkpoint_profile_hash(
            PpoSampleBudget::Standard,
            SCHEMA_DESCRIPTOR,
            (12, CHECKPOINT_SCHEMA_HASH),
        ),
    ),
    (
        16,
        checkpoint_profile_hash(
            PpoSampleBudget::Annealed,
            SCHEMA_DESCRIPTOR,
            (13, CHECKPOINT_ANNEALED_SCHEMA_HASH),
        ),
    ),
    (
        17,
        checkpoint_profile_hash(
            PpoSampleBudget::WideAnnealed,
            SCHEMA_DESCRIPTOR,
            (14, CHECKPOINT_WIDE_ANNEALED_SCHEMA_HASH),
        ),
    ),
];
const _: () = assert!(BLOCK_BYTES == 15 * 8 + 32);
const _: () = assert!(BLOCK_BYTES < super::MAX_META_BYTES as usize);

/// Atomic controller and generation-snapshot commitment, alongside model, Adam, and RNG state.
/// The runtime owns the rolling SHA-256; the checkpoint codec never reads snapshot files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveEnvironmentCheckpoint {
    pub config: AdaptiveEnvironmentConfig,
    pub limits: AdaptiveEnvironmentLimits,
    pub state: AdaptiveEnvironmentState,
    pub snapshot_count: u64,
    pub snapshot_hash: [u8; 32],
}

pub(super) fn validate_scope(
    run: &CheckpointRun,
    progress: &CheckpointProgress,
    ppo: PpoConfig,
) -> Result<(), CheckpointError> {
    super::validate_text("command line", &run.command_line)?;
    // A non-default scale ramp is recorded after the adaptive suffix. Only the
    // canonical rendering is split here, so a partial or malformed
    // `--environment-scale-*` token stays in the prefix and is rejected below.
    // Only the builtin feature carries the randomization module; without it the
    // scale flags cannot be produced, so every such token stays a mismatch.
    #[cfg(feature = "builtin")]
    let (command, _scale) = crate::randomization::split_scale_scope(&run.command_line);
    #[cfg(not(feature = "builtin"))]
    let command = run.command_line.as_str();
    let Some(checkpoint) = progress.adaptive_environment.as_ref() else {
        if has_environment_flags(command) {
            return Err(CheckpointError::InvalidManifest(
                "adaptive environment configuration/state mismatch",
            ));
        }
        return Ok(());
    };
    checkpoint.config.validate().map_err(controller_error)?;
    checkpoint.limits.validate().map_err(controller_error)?;
    if !progress.league_references.is_empty() {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment league",
        ));
    }
    let suffix = checkpoint.config.scope_suffix();
    let prefix = command
        .strip_suffix(&suffix)
        .ok_or(CheckpointError::InvalidManifest(
            "adaptive environment scope suffix",
        ))?;
    if has_environment_flags(prefix) {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment scope suffix",
        ));
    }
    validate_limits_scope(prefix, checkpoint.limits, ppo.environments)?;
    checkpoint
        .state
        .validate(checkpoint.config, checkpoint.limits, progress.global_update)
        .map_err(controller_error)?;
    let expected_count = checkpoint
        .state
        .generation
        .checked_add(u64::from(checkpoint.state.updates_in_generation > 0))
        .ok_or(CheckpointError::InvalidManifest(
            "adaptive environment snapshot count",
        ))?;
    if checkpoint.snapshot_count != expected_count {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment snapshot count",
        ));
    }
    if (checkpoint.snapshot_hash == [0; 32]) != (checkpoint.snapshot_count == 0) {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment snapshot hash",
        ));
    }
    Ok(())
}

pub(super) fn schema_identity(budget: PpoSampleBudget) -> (u32, u64) {
    match budget {
        PpoSampleBudget::Standard => IDENTITIES[0],
        PpoSampleBudget::Annealed => IDENTITIES[1],
        PpoSampleBudget::WideAnnealed => IDENTITIES[2],
    }
}

pub(super) fn encode(writer: &mut ManifestWriter, checkpoint: &AdaptiveEnvironmentCheckpoint) {
    let start = writer.bytes.len();
    for value in [
        checkpoint.config.success_updates,
        checkpoint.config.success_rate.units(),
        checkpoint.config.poor_updates,
        checkpoint.config.poor_rate.units(),
        checkpoint.config.extension.units(),
        checkpoint.limits.base_updates,
        checkpoint.limits.total_updates,
        checkpoint.limits.zero_updates,
        checkpoint.state.generation,
        checkpoint.state.start_update,
        checkpoint.state.updates_in_generation,
        checkpoint.state.success_streak,
        checkpoint.state.poor_streak,
        checkpoint.state.extension_awards,
        checkpoint.snapshot_count,
    ] {
        writer.u64(value);
    }
    writer.bytes.extend_from_slice(&checkpoint.snapshot_hash);
    debug_assert_eq!(writer.bytes.len() - start, BLOCK_BYTES);
    debug_assert_eq!(
        &writer.bytes[writer.bytes.len() - 32..],
        &checkpoint.snapshot_hash
    );
}

pub(super) fn decode(
    reader: &mut ManifestReader<'_>,
) -> Result<AdaptiveEnvironmentCheckpoint, CheckpointError> {
    let mut block = ManifestReader::new(reader.take(BLOCK_BYTES)?);
    let checkpoint = AdaptiveEnvironmentCheckpoint {
        config: AdaptiveEnvironmentConfig {
            success_updates: block.u64()?,
            success_rate: EnvironmentDecimal::from_units(block.u64()?),
            poor_updates: block.u64()?,
            poor_rate: EnvironmentDecimal::from_units(block.u64()?),
            extension: EnvironmentDecimal::from_units(block.u64()?),
        },
        limits: AdaptiveEnvironmentLimits {
            base_updates: block.u64()?,
            total_updates: block.u64()?,
            zero_updates: block.u64()?,
        },
        state: AdaptiveEnvironmentState {
            generation: block.u64()?,
            start_update: block.u64()?,
            updates_in_generation: block.u64()?,
            success_streak: block.u64()?,
            poor_streak: block.u64()?,
            extension_awards: block.u64()?,
        },
        snapshot_count: block.u64()?,
        snapshot_hash: block.array_32()?,
    };
    block.finish()?;
    Ok(checkpoint)
}

fn validate_limits_scope(
    prefix: &str,
    limits: AdaptiveEnvironmentLimits,
    environments: usize,
) -> Result<(), CheckpointError> {
    debug_assert!(prefix.len() <= MAX_TEXT_BYTES);
    debug_assert!(environments > 0);
    if !prefix.starts_with("train-annealed ") {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment command",
        ));
    }
    if prefix
        .split_ascii_whitespace()
        .any(|token| token.starts_with("--mastery") || token.starts_with("--league"))
    {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment mastery or league",
        ));
    }
    let generation_games = limits.base_updates.checked_mul(environments as u64).ok_or(
        CheckpointError::InvalidManifest("adaptive environment generation-games scope"),
    )?;
    for (flag, expected, field) in [
        (
            "--updates",
            limits.total_updates,
            "adaptive environment updates scope",
        ),
        (
            "--zero-updates",
            limits.zero_updates,
            "adaptive environment zero-updates scope",
        ),
        (
            "--generation-games",
            generation_games,
            "adaptive environment generation-games scope",
        ),
        (
            "--games",
            environments as u64,
            "adaptive environment games scope",
        ),
    ] {
        if scope_counter(prefix, flag, field)? != expected {
            return Err(CheckpointError::InvalidManifest(field));
        }
    }
    Ok(())
}

fn scope_counter(command: &str, flag: &str, field: &'static str) -> Result<u64, CheckpointError> {
    debug_assert!(command.len() <= MAX_TEXT_BYTES);
    debug_assert!(flag.starts_with("--"));
    let mut tokens = command.split(' ');
    let mut found = None;
    for _ in 0..=MAX_TEXT_BYTES {
        let Some(token) = tokens.next() else {
            return found.ok_or(CheckpointError::InvalidManifest(field));
        };
        if token
            .strip_prefix(flag)
            .is_some_and(|rest| rest.starts_with('='))
        {
            return Err(CheckpointError::InvalidManifest(field));
        }
        if token != flag {
            continue;
        }
        let value = tokens
            .next()
            .ok_or(CheckpointError::InvalidManifest(field))?;
        if found.is_some()
            || value.is_empty()
            || value.len() > 20
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || (value.len() > 1 && value.starts_with('0'))
        {
            return Err(CheckpointError::InvalidManifest(field));
        }
        let value = value
            .parse::<u64>()
            .map_err(|_| CheckpointError::InvalidManifest(field))?;
        if value > MAX_TRAINING_COUNTER {
            return Err(CheckpointError::InvalidManifest(field));
        }
        found = Some(value);
    }
    Err(CheckpointError::InvalidManifest(field))
}

fn has_environment_flags(command: &str) -> bool {
    debug_assert!(command.len() <= MAX_TEXT_BYTES);
    command
        .split_ascii_whitespace()
        .any(|token| token.starts_with("--environment-"))
}

fn controller_error(error: PpoError) -> CheckpointError {
    match error {
        PpoError::InvalidConfig(field) | PpoError::InvalidTransition(field) => {
            CheckpointError::InvalidManifest(field)
        }
        _ => CheckpointError::InvalidManifest("adaptive environment controller state"),
    }
}
