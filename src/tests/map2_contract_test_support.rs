use super::*;

#[test]
fn map2_rebase_duration_cadence_and_retention_boundaries_are_exact() {
    assert_eq!(MAP2_ID, bota_proto::MapId(2));
    assert_eq!(MAP2_PREGAME_TICKS, 900);
    assert_eq!(MAP2_GAME_TICKS, 27_000);
    assert_eq!(MAP2_TICK_CAP, 27_900);
    assert_eq!(MAP2_DECISION_INTERVAL_TICKS, 3);
    assert_eq!(MAP2_ACTOR_DECISIONS, 9_300);
    assert_eq!(MAP2_RETENTION_STRIDE, 8);
    assert_eq!(MAP2_RETAINED_DECISIONS, 1_163);
}
