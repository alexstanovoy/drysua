#[cfg(feature = "builtin")]
use crate::{
    ActionKind, CheckpointEvaluationBaseline, CheckpointEvaluationConfig, TrainingArtifact,
};
use crate::{PolicyModel, PolicySnapshot};

#[cfg(feature = "builtin")]
#[test]
fn runtime_checkpoint_evaluation_is_deterministic_and_covers_the_fixed_matrix() {
    let directory = evaluation_directory();
    let model = PolicyModel::fresh(88_301).expect("evaluation model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("runtime weights");
    let settings = CheckpointEvaluationConfig {
        pairs: 1,
        decisions: 2,
        seed: 88_302,
    };

    let first =
        crate::ppo_arena::evaluate_neural_map_two_checkpoint_cohort(settings, &directory, false)
            .expect("first report");
    let second =
        crate::ppo_arena::evaluate_neural_map_two_checkpoint_cohort(settings, &directory, false)
            .expect("second report");

    assert_eq!(first, second);
    assert_eq!(first.games.len(), 4);
    assert_eq!(
        first.fingerprint,
        PolicySnapshot::capture(&model, 0)
            .expect("snapshot")
            .fingerprint()
    );
    for map in [bota_proto::MapId(2)] {
        for baseline in [
            CheckpointEvaluationBaseline::Teacher,
            CheckpointEvaluationBaseline::Weak,
        ] {
            for team in [bota_proto::Team::Radiant, bota_proto::Team::Dire] {
                assert!(first.games.iter().any(|game| game.map == map
                    && game.baseline == baseline
                    && game.candidate_team == team));
            }
        }
    }
    for game in &first.games {
        assert_eq!(game.decisions, 2);
        assert_eq!(game.action_counts.iter().sum::<u32>(), game.decisions);
        assert!(game.wire_orders <= game.decisions);
        assert!(game.rejected_orders <= game.wire_orders);
        assert!(game.baseline_rejected_orders <= game.baseline_wire_orders);
        assert!(game.elapsed_ticks <= game.decisions * 3);
        assert_eq!(game.action_counts.len(), ActionKind::COUNT);
    }
    let mut collapsed = first.clone();
    for game in &mut collapsed.games {
        game.outcome = crate::CheckpointEvaluationOutcome::Timeout;
        game.action_counts = [0; ActionKind::COUNT];
        game.action_counts[crate::ActionKind::MovePoint.index()] = game.decisions;
    }
    let quality = collapsed.quality();
    assert!(!quality.passed);
    assert_eq!(quality.timeout_games, collapsed.games.len());
    assert_eq!(quality.collapsed_games, collapsed.games.len());

    std::fs::remove_dir_all(directory).expect("remove evaluation directory");
}

#[cfg(feature = "builtin")]
#[test]
fn checkpoint_quality_rejects_a_loss_to_the_weak_baseline() {
    let report = quality_report([
        quality_game(
            CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationOutcome::Loss,
            1,
        ),
        quality_game(
            CheckpointEvaluationBaseline::Teacher,
            crate::CheckpointEvaluationOutcome::Win,
            0,
        ),
    ]);

    let quality = report.quality();

    assert!(!quality.passed);
    assert_eq!(quality.weak_loss_games, 1);
}

#[cfg(feature = "builtin")]
#[test]
fn checkpoint_quality_rejects_a_stalled_weak_baseline_game() {
    let report = quality_report([
        quality_game(
            CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationOutcome::Timeout,
            0,
        ),
        quality_game(
            CheckpointEvaluationBaseline::Teacher,
            crate::CheckpointEvaluationOutcome::Win,
            0,
        ),
    ]);

    let quality = report.quality();

    assert!(!quality.passed);
    assert_eq!(quality.weak_stalled_games, 1);
}

#[cfg(feature = "builtin")]
#[test]
fn checkpoint_quality_requires_an_authoritative_win() {
    let report = quality_report([
        quality_game(
            CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationOutcome::Timeout,
            1,
        ),
        quality_game(
            CheckpointEvaluationBaseline::Teacher,
            crate::CheckpointEvaluationOutcome::Loss,
            0,
        ),
    ]);

    let quality = report.quality();

    assert!(!quality.passed);
    assert_eq!(quality.win_games, 0);
}

#[cfg(feature = "builtin")]
#[test]
fn checkpoint_quality_counts_a_weak_map2_cap_draw_as_stalled_but_not_a_technical_timeout() {
    for tick in [crate::MAP2_TICK_CAP - 1, crate::MAP2_TICK_CAP] {
        let mut drawn = quality_game(
            CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationOutcome::Draw,
            0,
        );
        drawn.map = bota_proto::MapId(2);
        drawn.final_summary.tick = tick;
        let report = quality_report([
            drawn,
            quality_game(
                CheckpointEvaluationBaseline::Teacher,
                crate::CheckpointEvaluationOutcome::Win,
                0,
            ),
        ]);
        let quality = report.quality();
        assert_eq!(
            quality.weak_stalled_games,
            usize::from(tick == crate::MAP2_TICK_CAP)
        );
        assert_eq!(quality.timeout_games, 0);
        assert_eq!(quality.passed, tick < crate::MAP2_TICK_CAP);
    }
}

#[cfg(feature = "builtin")]
fn quality_report(
    games: [crate::CheckpointEvaluationGame; 2],
) -> crate::CheckpointEvaluationReport {
    crate::CheckpointEvaluationReport {
        fingerprint: 1,
        games: games.into(),
    }
}

#[cfg(feature = "builtin")]
fn quality_game(
    baseline: CheckpointEvaluationBaseline,
    outcome: crate::CheckpointEvaluationOutcome,
    structures: u32,
) -> crate::CheckpointEvaluationGame {
    let mut actions = [0u32; ActionKind::COUNT];
    actions[ActionKind::Continue.index()] = 50;
    actions[ActionKind::MovePoint.index()] = 50;
    let summary = crate::GlobalSummary {
        enemy_structures_destroyed: structures,
        ..crate::GlobalSummary::default()
    };
    crate::CheckpointEvaluationGame {
        map: bota_proto::MapId(2),
        baseline,
        seed: 1,
        candidate_team: bota_proto::Team::Radiant,
        outcome,
        decisions: 100,
        wire_orders: 1,
        rejected_orders: 0,
        baseline_wire_orders: usize::from(baseline == CheckpointEvaluationBaseline::Teacher) as u32,
        baseline_rejected_orders: 0,
        elapsed_ticks: 300,
        action_counts: actions,
        final_summary: summary,
    }
}

#[cfg(feature = "builtin")]
fn evaluation_directory() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-evaluation-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        std::fs::remove_dir_all(&directory).expect("remove stale evaluation directory");
    }
    std::fs::create_dir(&directory).expect("create evaluation directory");
    directory
}
