#[cfg(test)]
#[path = "tests/map2_contract_test_support.rs"]
mod test_support;

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
/// Hero deaths at which a side loses; losing any tower loses as well.
pub const MAP2_DEATH_LIMIT: u16 = 2;
/// Simulation ticks between actor decisions.
pub const MAP2_DECISION_INTERVAL_TICKS: u32 = 3;
/// Maximum decisions from the initial tick-one snapshot through the cap.
pub const MAP2_ACTOR_DECISIONS: usize =
    (MAP2_TICK_CAP - 1).div_ceil(MAP2_DECISION_INTERVAL_TICKS) as usize;
/// Longest retained interval in decisions: every non-Continue decision begins
/// one, and a Continue decision begins one once the open interval is this long.
pub const MAP2_CONTINUE_STRIDE: usize = 8;

const _: () = assert!(MAP2_TICK_CAP > MAP2_PREGAME_TICKS);
const _: () = assert!(MAP2_TICK_CAP.is_multiple_of(MAP2_DECISION_INTERVAL_TICKS));
const _: () = assert!(MAP2_CONTINUE_STRIDE > 1 && MAP2_CONTINUE_STRIDE < MAP2_ACTOR_DECISIONS);

#[cfg(feature = "builtin")]
const _: () = {
    assert!(MAP2_ID.0 == bota_server::game::MAP2_ID.0);
    assert!(MAP2_TICK_RATE == bota_server::game::rules::TICKS_PER_SECOND);
    assert!(MAP2_PREGAME_TICKS == bota_server::game::rules::PREGAME_TICKS);
    assert!(MAP2_GAME_TICKS == bota_server::game::MAP2_GAME_TICKS);
    assert!(MAP2_TICK_CAP == bota_server::game::MAP2_TICK_CAP);
    assert!(MAP2_DEATH_LIMIT == bota_server::game::MAP2_DEATH_LIMIT);
};
