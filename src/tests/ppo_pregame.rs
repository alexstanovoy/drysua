use super::*;

#[test]
fn pregame_checkpoint_evaluation_starts_at_tick_one_like_training_and_deployment() {
    let model = PolicyModel::fresh(9_204_101).expect("model");
    let game = evaluate_checkpoint_game(
        &model,
        CheckpointEvaluationConfig {
            pairs: 1,
            decisions: 3,
            seed: 9_204_101,
        },
        MapId(1),
        CheckpointEvaluationBaseline::Teacher,
        9_204_101,
        0,
    )
    .expect("bounded checkpoint opening");

    assert_eq!(game.final_summary.tick, 10);
    assert_eq!(game.decisions, 3);
    assert_eq!(game.elapsed_ticks, 9);
    assert_eq!(game.rejected_orders, 0);
    assert_eq!(game.baseline_rejected_orders, 0);
    assert_eq!(game.baseline_wire_orders, 3);
}

#[test]
fn pregame_teacher_evaluation_learns_shops_and_moves_before_the_horn() {
    let game = evaluate_teacher_against_weak_game(9_204_101, 3, MapId(1), 1)
        .expect("bounded Teacher opening");

    assert_eq!(game.final_summary.tick, 10);
    assert_eq!(game.wire_orders, 3);
    assert_eq!(game.rejected_orders, 0);
    assert_eq!(game.baseline_wire_orders, 0);
}
