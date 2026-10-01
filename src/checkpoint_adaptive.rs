use super::{
    CheckpointError, CheckpointProgress, CheckpointRun, MAX_TEXT_BYTES, ManifestReader,
    ManifestWriter,
};
use crate::{
    AdaptiveEnvironmentConfig, AdaptiveEnvironmentLimits, AdaptiveEnvironmentState,
    EnvironmentDecimal, MAX_TRAINING_COUNTER, PpoError,
};

const BLOCK_BYTES: usize = 152;
const _: () = assert!(BLOCK_BYTES == 15 * 8 + 32);
const _: () = assert!(BLOCK_BYTES < super::MAX_META_BYTES as usize);

/// Adaptive controller state plus the snapshot commitment (count and rolling SHA-256), saved
/// atomically with the model, Adam and RNG state. The checkpoint codec never reads snapshot files;
/// `adaptive_randomization` maintains the hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveEnvironmentCheckpoint {
    pub config: AdaptiveEnvironmentConfig,
    pub limits: AdaptiveEnvironmentLimits,
    pub state: AdaptiveEnvironmentState,
    pub snapshot_count: u64,
    pub snapshot_hash: [u8; 32],
}

impl AdaptiveEnvironmentCheckpoint {
    /// Whether collection ever draws the current generation. One opened for the
    /// final update still counts that update but is never drawn or snapshotted.
    pub(crate) fn current_generation_collected(&self) -> bool {
        self.state
            .start_update
            .saturating_add(crate::adaptive_environment::ADAPTIVE_COLLECTION_LAG)
            < self.limits.total_updates
    }
}

pub(super) fn validate_scope(
    run: &CheckpointRun,
    progress: &CheckpointProgress,
) -> Result<(), CheckpointError> {
    super::validate_text("command line", &run.command_line)?;
    // A non-default scale ramp is recorded after the adaptive suffix. Only its exact
    // rendering is split off, so a partial or malformed `--environment-scale-*` token
    // stays in `command` and is rejected below. Without the builtin feature no scale
    // flags can be produced, so any such token is rejected.
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
    validate_limits_scope(prefix, checkpoint.limits)?;
    checkpoint
        .state
        .validate(checkpoint.config, checkpoint.limits, progress.global_update)
        .map_err(controller_error)?;
    let drawn =
        checkpoint.state.updates_in_generation > 0 && checkpoint.current_generation_collected();
    let expected_count = checkpoint
        .state
        .generation
        .checked_add(u64::from(drawn))
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
) -> Result<(), CheckpointError> {
    debug_assert!(prefix.len() <= MAX_TEXT_BYTES);
    if !prefix.starts_with("train-annealed ") {
        return Err(CheckpointError::InvalidManifest(
            "adaptive environment command",
        ));
    }
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
            "--generation-updates",
            limits.base_updates,
            "adaptive environment generation-updates scope",
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
