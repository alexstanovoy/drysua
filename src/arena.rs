use core::fmt;

use bota_proto::{EntityId, EventKind, MapId, Order, Pick, ServerMsg, SlotId, Team, TickMode};
use bota_server::game::{Command, Event, EventVisibility, MatchConfig, World};

use crate::SHADOW_FIEND;

/// Smallest supported vectorized match.
pub const MIN_ARENA_SEATS: u8 = 2;
/// Largest supported vectorized match.
pub const MAX_ARENA_SEATS: u8 = 10;
/// Simulation tick rate recorded in builtin match information.
pub const ARENA_TICK_RATE: u16 = 30;

/// Settings used to construct one builtin match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Number of seats in the range 2 through 10.
    pub seats: u8,
    /// Map used by the match.
    pub map: MapId,
    /// Match identity and deterministic random seed.
    pub seed: u64,
}

/// At most one order request from one seat for one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    /// Sequence number returned with a rejection.
    pub seq: u32,
    /// Unit to control, or the seat hero when absent.
    pub unit: Option<EntityId>,
    /// Order to validate and apply.
    pub order: Order,
}

/// Initial message stream for each seat in slot order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaStart {
    /// Per-seat messages, with MatchStart before the first Snapshot.
    pub messages: Vec<Vec<ServerMsg>>,
}

/// One advanced tick of message streams in slot order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaStep {
    /// Per-seat messages for the advanced tick.
    pub messages: Vec<Vec<ServerMsg>>,
}

/// Invalid input or lifecycle use of an arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArenaError {
    /// The configured seat count is outside 2 through 10.
    SeatCount {
        /// Rejected seat count.
        got: u8,
    },
    /// The request slice does not contain one entry per seat.
    RequestCount {
        /// Number of entries required.
        expected: usize,
        /// Number of entries received.
        got: usize,
    },
    /// The match already sent MatchOver.
    MatchOver,
}

impl fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeatCount { got } => {
                write!(
                    formatter,
                    "arena seat count must be between {MIN_ARENA_SEATS} and {MAX_ARENA_SEATS}, got {got}"
                )
            }
            Self::RequestCount { expected, got } => write!(
                formatter,
                "arena request count must equal seat count {expected}, got {got}"
            ),
            Self::MatchOver => formatter.write_str("arena cannot step after MatchOver"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// A synchronous lockstep match for vectorized callers.
pub struct Arena {
    world: World,
    picks: Vec<Pick>,
}

impl Arena {
    /// Creates a match and advances it to its first visible tick.
    pub fn new(settings: ArenaConfig) -> Result<(Self, ArenaStart), ArenaError> {
        validate_seat_count(settings.seats)?;
        let match_config = match_config(settings);
        let mut world = World::for_match(&match_config, match_config.rng());
        let events = world.advance(&[]);
        assert_eq!(world.tick, 1, "the first visible arena tick must be one");
        let arena = Self {
            world,
            picks: match_config.picks.clone(),
        };
        let start = ServerMsg::MatchStart {
            info: match_config.info(),
        };
        let mut messages = (0..arena.picks.len())
            .map(|_| {
                let mut stream = Vec::with_capacity(4);
                stream.push(start.clone());
                stream
            })
            .collect::<Vec<_>>();
        arena.append_tick_messages(&mut messages, &events);
        Ok((arena, ArenaStart { messages }))
    }

    /// Validates one optional request per seat and advances one tick.
    pub fn step(&mut self, requests: &[Option<Request>]) -> Result<ArenaStep, ArenaError> {
        if self.world.victor().is_some() {
            return Err(ArenaError::MatchOver);
        }
        let seats = self.picks.len();
        if requests.len() != seats {
            return Err(ArenaError::RequestCount {
                expected: seats,
                got: requests.len(),
            });
        }
        let mut messages: Vec<Vec<ServerMsg>> = (0..seats).map(|_| Vec::with_capacity(4)).collect();
        let mut commands = Vec::with_capacity(seats);
        for ((stream, pick), request) in messages.iter_mut().zip(&self.picks).zip(requests) {
            let Some(request) = request else {
                continue;
            };
            match self
                .world
                .validate_order(pick.slot, request.unit, &request.order)
            {
                Ok(()) => commands.push(Command {
                    slot: pick.slot,
                    unit: request.unit,
                    order: request.order,
                }),
                Err(reason) => stream.push(ServerMsg::OrderRejected {
                    seq: request.seq,
                    reason,
                }),
            }
        }
        let events = self.world.advance(&commands);
        self.append_tick_messages(&mut messages, &events);
        Ok(ArenaStep { messages })
    }

    /// Number of seats in this match.
    pub fn seat_count(&self) -> usize {
        self.picks.len()
    }

    /// Current simulation tick.
    pub fn tick(&self) -> u32 {
        self.world.tick
    }

    fn append_tick_messages(&self, messages: &mut [Vec<ServerMsg>], events: &[Event]) {
        assert_eq!(messages.len(), self.picks.len());
        for (stream, pick) in messages.iter_mut().zip(&self.picks) {
            stream.push(ServerMsg::Snapshot {
                view: self.world.view(pick.team),
            });
            let visible = visible_events(events, pick.team);
            if !visible.is_empty() {
                stream.push(ServerMsg::Events {
                    tick: self.world.tick,
                    events: visible,
                });
            }
        }
        if let Some(winner) = self.world.victor() {
            let over = ServerMsg::MatchOver {
                winner,
                stats: self.world.match_stats(),
            };
            for stream in messages {
                stream.push(over.clone());
            }
        }
    }
}

fn visible_events(events: &[Event], team: Team) -> Vec<EventKind> {
    events
        .iter()
        .filter(|event| match event.visible_to {
            EventVisibility::Everyone => true,
            EventVisibility::OneTeam(visible_team) => visible_team == team,
        })
        .map(|event| event.kind.clone())
        .collect()
}

fn validate_seat_count(seats: u8) -> Result<(), ArenaError> {
    if !(MIN_ARENA_SEATS..=MAX_ARENA_SEATS).contains(&seats) {
        return Err(ArenaError::SeatCount { got: seats });
    }
    Ok(())
}

fn match_config(settings: ArenaConfig) -> MatchConfig {
    MatchConfig {
        match_id: settings.seed,
        master_key: seed_key(settings.seed),
        picks: (0..settings.seats)
            .map(|index| Pick {
                slot: SlotId(index),
                team: if index.is_multiple_of(2) {
                    Team::Radiant
                } else {
                    Team::Dire
                },
                hero: SHADOW_FIEND,
            })
            .collect(),
        map: settings.map,
        tick_rate: ARENA_TICK_RATE,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 150,
    }
}

fn seed_key(seed: u64) -> [u8; 32] {
    let bytes = seed.to_le_bytes();
    std::array::from_fn(|index| bytes[index % bytes.len()])
}
