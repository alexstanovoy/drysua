use super::{CheckpointError, ManifestReader, ManifestWriter};
use crate::{MasteryConfig, MasteryProgress, MasteryStage};

pub(super) fn encode_config(writer: &mut ManifestWriter, config: Option<MasteryConfig>) {
    writer.u8(u8::from(config.is_some()));
    if let Some(config) = config {
        writer.u32(config.window() as u32);
        writer.u8(config.threshold(MasteryStage::Weak));
        writer.u8(config.threshold(MasteryStage::Teacher));
    }
}

pub(super) fn decode_config(
    reader: &mut ManifestReader<'_>,
) -> Result<Option<MasteryConfig>, CheckpointError> {
    match reader.u8()? {
        0 => Ok(None),
        1 => MasteryConfig::from_resolved(reader.u32()? as usize, reader.u8()?, reader.u8()?)
            .map(Some)
            .map_err(CheckpointError::InvalidManifest),
        _ => Err(CheckpointError::InvalidManifest(
            "mastery configuration presence",
        )),
    }
}

pub(super) fn encode_progress(writer: &mut ManifestWriter, progress: Option<&MasteryProgress>) {
    writer.u8(u8::from(progress.is_some()));
    if let Some(progress) = progress {
        writer.u8(progress.stage() as u8);
        writer.u64(progress.games());
        writer.u32(progress.recent().len() as u32);
        for win in progress.recent() {
            writer.u8(u8::from(*win));
        }
    }
}

pub(super) fn decode_progress(
    reader: &mut ManifestReader<'_>,
    config: Option<MasteryConfig>,
) -> Result<Option<MasteryProgress>, CheckpointError> {
    match reader.u8()? {
        0 => return Ok(None),
        1 => {}
        _ => {
            return Err(CheckpointError::InvalidManifest(
                "mastery progress presence",
            ));
        }
    }
    let config = config.ok_or(CheckpointError::InvalidManifest(
        "mastery configuration/state mismatch",
    ))?;
    let stage = match reader.u8()? {
        0 => MasteryStage::Weak,
        1 => MasteryStage::Teacher,
        2 => MasteryStage::Completed,
        _ => return Err(CheckpointError::InvalidManifest("mastery stage")),
    };
    let games = reader.u64()?;
    let count = reader.u32()? as usize;
    if count > config.window() {
        return Err(CheckpointError::InvalidManifest(
            "mastery window exceeds configured capacity",
        ));
    }
    let mut recent = Vec::with_capacity(count);
    for _ in 0..count {
        recent.push(match reader.u8()? {
            0 => false,
            1 => true,
            _ => return Err(CheckpointError::InvalidManifest("mastery outcome flag")),
        });
    }
    MasteryProgress::restore(stage, games, recent, config)
        .map(Some)
        .map_err(CheckpointError::InvalidManifest)
}

pub(super) fn validate_scope(
    config: Option<MasteryConfig>,
    progress: Option<&MasteryProgress>,
    updates: u64,
    environments: usize,
) -> Result<(), CheckpointError> {
    match (config, progress) {
        (None, None) => Ok(()),
        (Some(config), Some(progress)) => {
            progress
                .validate(config)
                .map_err(CheckpointError::InvalidManifest)?;
            if !crate::valid_environment_count(environments) {
                return Err(CheckpointError::InvalidManifest(
                    "mastery environment count",
                ));
            }
            let total = updates.checked_mul(environments as u64).ok_or(
                CheckpointError::InvalidManifest("mastery total game counter"),
            )?;
            let preceding =
                total
                    .checked_sub(progress.games())
                    .ok_or(CheckpointError::InvalidManifest(
                        "mastery game count exceeds updates",
                    ))?;
            let minimum = config.window().div_ceil(environments) * environments;
            if !progress.games().is_multiple_of(environments as u64)
                || (progress.stage() == MasteryStage::Weak && preceding != 0)
                || (progress.stage() != MasteryStage::Weak
                    && (preceding < minimum as u64
                        || !preceding.is_multiple_of(environments as u64)))
            {
                return Err(CheckpointError::InvalidManifest(
                    "mastery stage/game counters",
                ));
            }
            Ok(())
        }
        _ => Err(CheckpointError::InvalidManifest(
            "mastery configuration/state mismatch",
        )),
    }
}

#[cfg(test)]
#[path = "tests/mastery_codec.rs"]
mod tests;
