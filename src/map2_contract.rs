/// Map used by every production training and evaluation entry point.
pub const MAP2_ID: bota_proto::MapId = bota_proto::MapId(2);
/// Simulation ticks per second in the Map2 duration contract.
pub const MAP2_TICK_RATE: u32 = 30;
/// Pregame ticks included in the native Map2 cap.
pub const MAP2_PREGAME_TICKS: u32 = 30 * MAP2_TICK_RATE;
/// Fifteen minutes of gameplay after pregame.
pub const MAP2_GAME_TICKS: u32 = 15 * 60 * MAP2_TICK_RATE;
/// Inclusive native terminal tick, including pregame.
pub const MAP2_TICK_CAP: u32 = MAP2_PREGAME_TICKS + MAP2_GAME_TICKS;
/// Simulation ticks between actor decisions.
pub const MAP2_DECISION_INTERVAL_TICKS: u32 = 3;
/// Maximum decisions from the initial tick-one snapshot through the cap.
pub const MAP2_ACTOR_DECISIONS: usize =
    (MAP2_TICK_CAP - 1).div_ceil(MAP2_DECISION_INTERVAL_TICKS) as usize;
/// One independently phased action retained per this many actor decisions.
pub const MAP2_RETENTION_STRIDE: usize = 8;
/// Per-environment retained capacity sufficient for every retention phase.
pub const MAP2_RETAINED_DECISIONS: usize = MAP2_ACTOR_DECISIONS.div_ceil(MAP2_RETENTION_STRIDE);

const _: () = assert!(MAP2_RETENTION_STRIDE.is_power_of_two());
const _: () = assert!(MAP2_TICK_CAP > MAP2_PREGAME_TICKS);
const _: () = assert!(MAP2_TICK_CAP.is_multiple_of(MAP2_DECISION_INTERVAL_TICKS));
const _: () = assert!((MAP2_RETAINED_DECISIONS - 1) * MAP2_RETENTION_STRIDE < MAP2_ACTOR_DECISIONS);
const _: () = assert!(MAP2_RETAINED_DECISIONS * MAP2_RETENTION_STRIDE >= MAP2_ACTOR_DECISIONS);

#[cfg(feature = "builtin")]
const _: () = {
    assert!(MAP2_ID.0 == bota_server::game::MAP2_ID.0);
    assert!(MAP2_TICK_RATE == bota_server::game::rules::TICKS_PER_SECOND);
    assert!(MAP2_PREGAME_TICKS == bota_server::game::rules::PREGAME_TICKS);
    assert!(MAP2_GAME_TICKS == bota_server::game::MAP2_GAME_TICKS);
    assert!(MAP2_TICK_CAP == bota_server::game::MAP2_TICK_CAP);
};

#[cfg(test)]
mod tests {
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
}
