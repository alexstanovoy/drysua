use super::*;
use crate::TrainingGameOutcome as Outcome;

#[test]
fn mastery_codec_rejects_corrupt_stage_flag_window_and_counter() {
    let config = Some(MasteryConfig::new(3, 80, &[]).expect("config"));
    for (stage, games, flags, expected) in [
        (3, 1, vec![0], "mastery stage"),
        (0, 1, vec![2], "mastery outcome flag"),
        (
            0,
            4,
            vec![0; 4],
            "mastery window exceeds configured capacity",
        ),
        (0, 0, vec![0], "mastery game counter/window mismatch"),
        (
            2,
            2,
            vec![1; 2],
            "completed mastery requires a qualifying Teacher window",
        ),
    ] {
        let mut writer = ManifestWriter::default();
        writer.u8(1);
        writer.u8(stage);
        writer.u64(games);
        writer.u32(flags.len() as u32);
        writer.bytes.extend(flags);
        assert_eq!(
            decode_progress(&mut ManifestReader::new(&writer.bytes), config),
            Err(CheckpointError::InvalidManifest(expected))
        );
    }
    assert_eq!(
        decode_config(&mut ManifestReader::new(&[2])),
        Err(CheckpointError::InvalidManifest(
            "mastery configuration presence"
        ))
    );
    assert_eq!(
        decode_progress(&mut ManifestReader::new(&[2]), config),
        Err(CheckpointError::InvalidManifest(
            "mastery progress presence"
        ))
    );
}

#[test]
fn mastery_codec_roundtrips_partial_teacher_and_completed_windows() {
    let config = MasteryConfig::new(5, 80, &[]).expect("config");
    for (stage, games, flags) in [
        (MasteryStage::Weak, 4, vec![true, false, true, true]),
        (MasteryStage::Teacher, 0, vec![]),
        (
            MasteryStage::Completed,
            12,
            vec![true, true, false, true, true],
        ),
    ] {
        let progress = MasteryProgress::restore(stage, games, flags, config).expect("state");
        let mut writer = ManifestWriter::default();
        encode_progress(&mut writer, Some(&progress));
        let mut reader = ManifestReader::new(&writer.bytes);
        assert_eq!(
            decode_progress(&mut reader, Some(config)).expect("decode"),
            Some(progress)
        );
        reader.finish().expect("all bytes consumed");
    }
}

#[test]
fn mastery_checkpoint_stage_game_counters_must_match_completed_batches() {
    let config = MasteryConfig::new(5, 80, &[]).expect("config");
    let wrong =
        MasteryProgress::restore(MasteryStage::Weak, 4, vec![false; 4], config).expect("state");
    assert_eq!(
        validate_scope(Some(config), Some(&wrong), 1, 2),
        Err(CheckpointError::InvalidManifest(
            "mastery game count exceeds updates"
        ))
    );
    assert_eq!(
        validate_scope(Some(config), Some(&wrong), 3, 2),
        Err(CheckpointError::InvalidManifest(
            "mastery stage/game counters"
        ))
    );
    assert!(validate_scope(Some(config), Some(&wrong), 2, 2).is_ok());
    assert_eq!(
        validate_scope(None, Some(&wrong), 2, 2),
        Err(CheckpointError::InvalidManifest(
            "mastery configuration/state mismatch"
        ))
    );
}

#[test]
fn checkpoint_mastery_scope_accepts_even_counts_up_to_twenty_six() {
    let config = MasteryConfig::new(50, 100, &[]).expect("config");
    for environments in [2usize, 6, 8, 16, 18, 20, 22, 24, 26] {
        let mut progress = MasteryProgress::default();
        progress
            .record_batch(config, &vec![Outcome::Win; environments])
            .expect("batch");
        validate_scope(Some(config), Some(&progress), 1, environments).expect("even scope");
    }
    let mut progress = MasteryProgress::default();
    progress
        .record_batch(config, &[Outcome::Win; 6])
        .expect("batch");
    for environments in [0usize, 7, 28] {
        assert!(validate_scope(Some(config), Some(&progress), 1, environments).is_err());
    }
}
