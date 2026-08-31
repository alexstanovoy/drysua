use bota_proto::{RejectReason, ServerMsg, SlotId, Team, TickMode};

use crate::{Link, SHADOW_FIEND, Seated, Wire};

/// The result observed by drysua for one match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The seat held by drysua.
    pub slot: Option<SlotId>,
    /// The team of the held seat.
    pub team: Option<Team>,
    /// The winning team, or no team when a tick limit ended play.
    pub winner: Option<Team>,
    /// The latest snapshot tick received.
    pub ticks: u32,
    /// Rejected orders reported by the server.
    pub rejections: u32,
    /// The latest rejection reason, or no reason when none was rejected.
    pub last_rejection: Option<RejectReason>,
}

/// Connects and follows the Continue policy for one match.
pub fn play(address: &str, name: &str, limit: Option<u32>) -> std::io::Result<Outcome> {
    let (mut link, seated) = Link::join(address, name)?;
    play_on(&mut link, seated, limit)
}

/// Follows the Continue policy on an assigned match connection.
pub fn play_on(
    wire: &mut impl Wire,
    seated: Seated,
    limit: Option<u32>,
) -> std::io::Result<Outcome> {
    let mut outcome = Outcome {
        slot: Some(seated.slot),
        ..Outcome::default()
    };
    while let Some(message) = wire.hear()? {
        match message {
            ServerMsg::MatchStart { info } => {
                validate_match_terms(info.tick_rate, info.mode, seated)?;
                outcome.team = Some(validate_pick(&info.picks, seated.slot)?);
            }
            ServerMsg::Snapshot { view } => {
                if outcome.team.is_none() {
                    return Err(std::io::Error::other(
                        "server sent Snapshot before MatchStart",
                    ));
                }
                if view.viewer != outcome.team {
                    return Err(std::io::Error::other(format!(
                        "Snapshot viewer {:?} differs from assigned team {:?}",
                        view.viewer, outcome.team
                    )));
                }
                if view.tick <= outcome.ticks {
                    return Err(std::io::Error::other(format!(
                        "Snapshot tick {} does not follow tick {}",
                        view.tick, outcome.ticks
                    )));
                }
                outcome.ticks = view.tick;
                if seated.mode == TickMode::Lockstep {
                    wire.acknowledge(view.tick)?;
                }
                if limit.is_some_and(|last_tick| view.tick >= last_tick) {
                    return Ok(outcome);
                }
            }
            ServerMsg::OrderRejected { reason, .. } => {
                outcome.rejections = outcome
                    .rejections
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("order rejection count overflowed"))?;
                outcome.last_rejection = Some(reason);
            }
            ServerMsg::MatchOver { winner, .. } => {
                if outcome.team.is_none() {
                    return Err(std::io::Error::other(
                        "server sent MatchOver before MatchStart",
                    ));
                }
                outcome.winner = Some(winner);
                return Ok(outcome);
            }
            ServerMsg::Welcome { .. }
            | ServerMsg::LobbyState { .. }
            | ServerMsg::Events { .. }
            | ServerMsg::Orders { .. }
            | ServerMsg::ParticipantLeft { .. } => {}
        }
    }
    Err(std::io::Error::other(
        "server closed the connection before MatchOver",
    ))
}

fn validate_match_terms(tick_rate: u16, mode: TickMode, seated: Seated) -> std::io::Result<()> {
    if tick_rate != seated.tick_rate {
        return Err(std::io::Error::other(format!(
            "MatchStart tick rate {tick_rate} differs from Welcome tick rate {}",
            seated.tick_rate
        )));
    }
    if mode != seated.mode {
        return Err(std::io::Error::other(format!(
            "MatchStart mode {mode:?} differs from Welcome mode {:?}",
            seated.mode
        )));
    }
    Ok(())
}

fn validate_pick(picks: &[bota_proto::Pick], slot: SlotId) -> std::io::Result<Team> {
    let pick = picks.iter().find(|pick| pick.slot == slot).ok_or_else(|| {
        std::io::Error::other(format!(
            "MatchStart has no hero pick for assigned slot {}",
            slot.0
        ))
    })?;
    if pick.hero != SHADOW_FIEND {
        return Err(std::io::Error::other(format!(
            "assigned slot {} picked HeroId({}), expected Shadow Fiend HeroId({})",
            slot.0, pick.hero.0, SHADOW_FIEND.0
        )));
    }
    Ok(pick.team)
}
