#![allow(
    clippy::float_arithmetic,
    reason = "Diagnostic sums of production reward components"
)]

use std::io::{self, Read, Write};
use std::path::Path;

use bota_proto::{ServerMsg, SlotId, Team, TickMode};

use crate::{Map2Reward, Map2RewardBreakdown, Map2RewardEnd};

const MAX_MESSAGES: usize = 1_000_000;
const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const REPORT_LIMIT: usize = 1024 * 1024;

/// Observes one copied participant stream and writes bounded reward diagnostics.
pub(crate) fn run(output: &Path, interval: u32) -> io::Result<()> {
    assert!((30..=crate::MAP2_TICK_CAP).contains(&interval));
    let mut final_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut timeline = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("jsonl"))?;
    let mut observer = Observer::new();
    let mut input = BoundedInput::new(io::stdin().lock(), MAX_BYTES);
    let result = consume(&mut input, &mut observer, interval, &mut timeline);
    let reason = result.as_ref().err().map(ToString::to_string);
    let report = observer.report(reason.as_deref());
    final_file.write_all(report.as_bytes())?;
    final_file.flush()?;
    println!("{report}");
    result
}

pub(crate) struct BoundedInput<R> {
    input: R,
    remaining: u64,
}

impl<R: Read> BoundedInput<R> {
    pub(crate) fn new(input: R, limit: u64) -> Self {
        assert!(limit > 0);
        assert!(limit <= MAX_BYTES);
        Self {
            input,
            remaining: limit,
        }
    }
}

impl<R: Read> Read for BoundedInput<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return match self.input.read(&mut [0])? {
                0 => Ok(0),
                _ => Err(invalid("observer byte limit exceeded")),
            };
        }
        let limit = output.len().min(self.remaining as usize);
        let count = self.input.read(&mut output[..limit])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

pub(crate) fn consume(
    input: &mut impl Read,
    observer: &mut Observer,
    interval: u32,
    timeline: &mut impl Write,
) -> io::Result<()> {
    let mut next = interval;
    let mut written = 0;
    let mut previous = [0.0; 18];
    for _ in 0..MAX_MESSAGES {
        let Some(message) = read_message(input)? else {
            return if observer.ended {
                Ok(())
            } else {
                Err(invalid("EOF before MatchOver"))
            };
        };
        observer.observe(message)?;
        if observer.tick >= next && !observer.pending {
            let record = observer.timeline(&previous);
            written += record.len() + 1;
            if written > REPORT_LIMIT {
                return Err(invalid("reward timeline exceeds 1 MiB"));
            }
            writeln!(timeline, "{record}")?;
            timeline.flush()?;
            println!("{record}");
            previous = observer.totals;
            next = observer.tick + interval;
        }
    }
    Err(invalid("observer message limit exceeded"))
}

pub(crate) fn read_message(input: &mut impl Read) -> io::Result<Option<ServerMsg>> {
    let mut prefix = [0; bota_proto::LEN_PREFIX];
    match input.read_exact(&mut prefix[..1]) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    input
        .read_exact(&mut prefix[1..])
        .map_err(|_| invalid("truncated observer frame prefix"))?;
    let length = u32::from_le_bytes(prefix) as usize;
    if !(1..=bota_proto::MAX_PAYLOAD_LEN).contains(&length) {
        return Err(invalid("observer frame length outside 1..=MAX_PAYLOAD_LEN"));
    }
    let mut payload = vec![0; length];
    input
        .read_exact(&mut payload)
        .map_err(|_| invalid("truncated observer frame payload"))?;
    let message: ServerMsg = bota_proto::decode_payload(&payload).map_err(io::Error::other)?;
    // The codec accepts a postcard value; require the entire canonical payload as well.
    let canonical = bota_proto::encode_frame_to_vec(&message).map_err(io::Error::other)?;
    if canonical[bota_proto::LEN_PREFIX..] != payload {
        return Err(invalid("noncanonical or trailing observer payload"));
    }
    assert!(length <= bota_proto::MAX_PAYLOAD_LEN);
    Ok(Some(message))
}

pub(crate) struct Observer {
    slot: Option<SlotId>,
    team: Option<Team>,
    mode: Option<TickMode>,
    reward: Option<Map2Reward>,
    tick: u32,
    pending: bool,
    ended: bool,
    outcome: Option<Map2RewardEnd>,
    totals: [f64; 18],
    counts: [u64; 32],
    pub(crate) last_interval: Map2RewardBreakdown,
}

impl Observer {
    pub(crate) fn new() -> Self {
        Self {
            slot: None,
            team: None,
            mode: None,
            reward: None,
            tick: 0,
            pending: false,
            ended: false,
            outcome: None,
            totals: [0.0; 18],
            counts: [0; 32],
            last_interval: Map2RewardBreakdown::default(),
        }
    }

    pub(crate) fn observe(&mut self, message: ServerMsg) -> io::Result<()> {
        if self.ended {
            return Err(invalid("server message after MatchOver"));
        }
        match message {
            ServerMsg::Welcome {
                player_id,
                slot,
                tick_rate,
                mode,
            } => {
                self.welcome(player_id, slot, tick_rate, mode)?;
            }
            ServerMsg::MatchStart { info } => self.start(&info)?,
            ServerMsg::Snapshot { view } => self.snapshot(&view)?,
            ServerMsg::Events { tick, events } => {
                self.reward()?
                    .observe_events(tick, &events)
                    .map_err(io::Error::other)?;
                self.pending = false;
                self.tick = tick;
                let interval = self.reward()?.take_interval().map_err(io::Error::other)?;
                self.add(interval);
            }
            ServerMsg::MatchOver { winner, stats } => {
                if stats.slots.len() != 2
                    || stats.slots[0].slot != SlotId(0)
                    || stats.slots[1].slot != SlotId(1)
                {
                    return Err(invalid(
                        "MatchOver must contain both participant statistics",
                    ));
                }
                self.finish(winner, stats.duration)?;
            }
            ServerMsg::LobbyState { .. } if self.reward.is_none() => {}
            ServerMsg::OrderRejected { .. } if self.reward.is_some() => {}
            ServerMsg::ParticipantLeft { .. } => {}
            _ => return Err(invalid("unexpected message on participant reward stream")),
        }
        Ok(())
    }

    fn welcome(
        &mut self,
        player: bota_proto::PlayerId,
        slot: Option<SlotId>,
        tick_rate: u16,
        mode: TickMode,
    ) -> io::Result<()> {
        if self.slot.is_some()
            || slot.is_none_or(|slot| slot.0 > 1)
            || tick_rate != 30
            || player.0 == 0
        {
            return Err(invalid(
                "invalid or duplicate Welcome; expected Map2 participant at 30 Hz",
            ));
        }
        self.slot = slot;
        self.mode = Some(mode);
        Ok(())
    }

    fn snapshot(&mut self, view: &bota_proto::WorldView) -> io::Result<()> {
        if self.tick == 0 && !self.pending && view.tick != 1 {
            return Err(invalid(
                "first Snapshot must be tick 1; missing initial baseline",
            ));
        }
        if view.viewer != self.team || self.team.is_none() {
            return Err(invalid(
                "Snapshot viewer differs from assigned participant team",
            ));
        }
        self.reward()?
            .observe_snapshot(view)
            .map_err(io::Error::other)?;
        self.pending = true;
        Ok(())
    }

    fn start(&mut self, info: &bota_proto::MatchInfo) -> io::Result<()> {
        let slot = self
            .slot
            .ok_or_else(|| invalid("MatchStart before Welcome"))?;
        if self.reward.is_some() || Some(info.mode) != self.mode || info.tick_rate != 30 {
            return Err(invalid(
                "duplicate MatchStart or inconsistent Welcome terms",
            ));
        }
        if info.picks.len() != 2
            || info.picks[0].slot != SlotId(0)
            || info.picks[1].slot != SlotId(1)
        {
            return Err(invalid(
                "MatchStart must contain participant slots zero and one",
            ));
        }
        let reward = Map2Reward::new(slot, info).map_err(io::Error::other)?;
        self.team = info
            .picks
            .iter()
            .find(|pick| pick.slot == slot)
            .map(|pick| pick.team);
        self.reward = Some(reward);
        assert!(self.team.is_some());
        assert!(self.slot.is_some());
        Ok(())
    }

    fn reward(&mut self) -> io::Result<&mut Map2Reward> {
        self.reward
            .as_mut()
            .ok_or_else(|| invalid("reward observation before MatchStart"))
    }

    fn finish(&mut self, winner: Team, duration: u32) -> io::Result<()> {
        if self.pending || self.tick == 0 || duration != self.tick {
            return Err(invalid(
                "MatchOver without complete final Snapshot/Events duration",
            ));
        }
        let end = if winner == Team::Neutral {
            Map2RewardEnd::Draw
        } else if Some(winner) == self.team {
            Map2RewardEnd::Win
        } else {
            Map2RewardEnd::Loss
        };
        let final_interval = self.reward()?.finish(end).map_err(io::Error::other)?;
        self.add(final_interval);
        self.outcome = Some(end);
        self.ended = true;
        assert!(!self.pending);
        assert!(self.outcome.is_some());
        Ok(())
    }

    fn add(&mut self, interval: Map2RewardBreakdown) {
        for (total, value) in self.totals.iter_mut().zip(components(&interval)) {
            *total += value;
        }
        for (index, (total, value)) in self.counts.iter_mut().zip(counts(&interval)).enumerate() {
            if index == 30 {
                *total |= value;
            } else {
                *total += value;
            }
        }
        self.last_interval = interval;
        assert!(self.totals.iter().all(|value| value.is_finite()));
        assert!(self.tick <= crate::MAP2_TICK_CAP);
    }

    pub(crate) fn report(&self, error: Option<&str>) -> String {
        let complete = self.ended && error.is_none();
        let total: f64 = self.totals.iter().sum();
        format!(
            concat!(
                "{{\"kind\":\"final\",\"profile_version\":{},\"profile_hash\":\"{}\",",
                "\"slot\":{},\"team\":{},\"ticks\":{},\"complete\":{},\"valid\":{},",
                "\"pending_events\":{},\"reward_ticks\":{},",
                "\"outcome\":{},\"error\":{},\"components\":{},\"raw_counts\":{},",
                "\"total\":{},\"total_without_terminal\":{}}}"
            ),
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH,
            self.slot
                .map_or("null".to_owned(), |slot| slot.0.to_string()),
            quoted(self.team.map(|team| format!("{team:?}")).as_deref()),
            self.tick,
            complete,
            complete,
            self.pending,
            self.tick.saturating_sub(1),
            quoted(self.outcome.map(|end| format!("{end:?}")).as_deref()),
            quoted(error),
            fields(&COMPONENTS, &self.totals),
            fields(&COUNTS, &self.counts),
            total,
            total - self.totals[16]
        )
    }

    fn timeline(&self, previous: &[f64; 18]) -> String {
        let delta: [f64; 18] = std::array::from_fn(|index| self.totals[index] - previous[index]);
        format!(
            concat!(
                "{{\"kind\":\"interval\",\"status\":\"partial\",\"profile_version\":{},",
                "\"profile_hash\":\"{}\",\"team\":{},\"ticks\":{},\"components\":{},\"delta\":{},\"total\":{}}}"
            ),
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH,
            quoted(self.team.map(|team| format!("{team:?}")).as_deref()),
            self.tick,
            fields(&COMPONENTS, &self.totals),
            fields(&COMPONENTS, &delta),
            self.totals.iter().sum::<f64>()
        )
    }
}

const COMPONENTS: [&str; 18] = [
    "gold",
    "experience",
    "hero_damage",
    "hero_damage_taken",
    "creep_damage_taken",
    "tower_damage_taken",
    "other_damage_taken",
    "mana_spent",
    "tower_health",
    "lane_pressure",
    "pregame_movement",
    "opening_position",
    "fountain_wait",
    "fountain_wait_refund",
    "stagnation_base",
    "stagnation_ticks_cost",
    "terminal",
    "victory_time",
];

fn components(value: &Map2RewardBreakdown) -> [f64; 18] {
    [
        value.gold,
        value.experience,
        value.hero_damage,
        value.hero_damage_taken,
        value.creep_damage_taken,
        value.tower_damage_taken,
        value.other_damage_taken,
        value.mana_spent,
        value.tower_health,
        value.lane_pressure,
        value.pregame_movement,
        value.opening_position,
        value.fountain_wait,
        value.fountain_wait_refund,
        value.stagnation_base,
        value.stagnation_ticks_cost,
        value.terminal,
        value.victory_time,
    ]
}

const COUNTS: [&str; 32] = [
    "tower_damage_taken",
    "opening_position_checks",
    "victory_time_ticks",
    "own_gold_earned",
    "enemy_gold_earned",
    "own_xp_gained",
    "enemy_xp_gained",
    "hero_damage_dealt",
    "structure_damage_dealt",
    "creep_kills",
    "creep_denies",
    "hero_damage_taken",
    "creep_damage_taken",
    "other_damage_taken",
    "unattributed_damage_taken",
    "mana_spent",
    "mana_unobserved_ticks",
    "lane_last_hits",
    "neutral_last_hits",
    "unattributed_damage_events",
    "unattributed_deaths",
    "duplicate_deaths",
    "lane_observed_ticks",
    "fountain_wait_ticks",
    "fountain_wait_charged_ticks",
    "fountain_wait_refunds",
    "stagnation_active_ticks",
    "stagnation_idle_ticks",
    "stagnation_charged_ticks",
    "stagnation_base_charges",
    "stagnation_repaid_ticks",
    "progress_reasons",
];

fn counts(value: &Map2RewardBreakdown) -> [u64; 32] {
    let value = value.observations;
    [
        value.tower_damage_taken,
        value.opening_position_checks,
        value.victory_time_ticks,
        value.own_gold_earned,
        value.enemy_gold_earned,
        value.own_xp_gained,
        value.enemy_xp_gained,
        value.hero_damage_dealt,
        value.structure_damage_dealt,
        value.creep_kills,
        value.creep_denies,
        value.hero_damage_taken,
        value.creep_damage_taken,
        value.other_damage_taken,
        value.unattributed_damage_taken,
        value.mana_spent,
        value.mana_unobserved_ticks,
        value.lane_last_hits,
        value.neutral_last_hits,
        value.unattributed_damage_events,
        value.unattributed_deaths,
        value.duplicate_deaths,
        value.lane_observed_ticks,
        value.fountain_wait_ticks,
        value.fountain_wait_charged_ticks,
        value.fountain_wait_refunds,
        value.stagnation_active_ticks,
        value.stagnation_idle_ticks,
        value.stagnation_charged_ticks,
        value.stagnation_base_charges,
        value.stagnation_repaid_ticks,
        u64::from(value.progress_reasons),
    ]
}

fn fields<T: std::fmt::Display, const N: usize>(names: &[&str; N], values: &[T; N]) -> String {
    assert!(N <= 64);
    let fields: Vec<_> = names
        .iter()
        .zip(values)
        .map(|(name, value)| format!("\"{name}\":{value}"))
        .collect();
    format!("{{{}}}", fields.join(","))
}

fn quoted(value: Option<&str>) -> String {
    let Some(value) = value else {
        return "null".to_owned();
    };
    let mut result = String::from("\"");
    for character in value.chars().take(1024) {
        match character {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            character if character.is_control() => result.push(' '),
            character => result.push(character),
        }
    }
    result.push('"');
    result
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
