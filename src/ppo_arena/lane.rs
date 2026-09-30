//! One inference lane: a fixed set of slots, its own thread, device stream and
//! actor weight replica.
//!
//! A lane runs rounds: one batched inference over its slots, then one advance
//! job per slot on the shared simulation pool. Lanes never wait for each other
//! inside an update. Every decision depends only on the lane's own round
//! sequence, never on thread timing: an update part ends after the first round
//! whose closed intervals reach the lane's sample target, and the next part's
//! weights are installed before its first round.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use bota_proto::ModifierSpec;

use super::episode::EpisodeRecord;
use super::pool::{Outcome, Reply, SimPool, Work};
use super::slot::{Decision, GamePlan, GameSchedule, NextGame, OpponentKind, Slot, SlotSnapshot};
use super::text_error;
use crate::{EncoderRow, PolicyDevice, PolicyModel, PpoError, PpoRng, PpoTransition};

/// The actor weights and world rules of one update's collection.
pub(super) struct PartConfig {
    pub(super) update: u64,
    /// Completed updates of the weights in `actor`.
    pub(super) version: u64,
    pub(super) actor: Arc<Vec<f32>>,
    pub(super) spec: ModifierSpec,
}

/// One closed interval tagged with the game it belongs to.
pub(super) struct LaneSample {
    pub(super) slot: usize,
    pub(super) game: u64,
    pub(super) transition: PpoTransition,
}

/// One lane's share of one update.
pub(super) struct PartDone {
    pub(super) lane: usize,
    pub(super) update: u64,
    pub(super) samples: Vec<LaneSample>,
    pub(super) episodes: Vec<EpisodeRecord>,
    /// Slot states at the start of the lane's next part.
    pub(super) snapshot: Vec<SlotSnapshot>,
    pub(super) rounds: u64,
    pub(super) inference: Duration,
    pub(super) simulation: Duration,
}

/// Static description of one lane.
pub(super) struct LaneSettings {
    pub(super) index: usize,
    /// Global slot indices, ascending.
    pub(super) slots: Vec<usize>,
    /// Closed intervals that end this lane's part of an update.
    pub(super) target: usize,
    pub(super) device: PolicyDevice,
    /// Frozen opponent pool parameters.
    pub(super) snapshots: Arc<Vec<Arc<Vec<f32>>>>,
}

/// How a lane obtains its first games.
pub(super) enum LaneStart {
    Fresh,
    Replay(Vec<SlotSnapshot>),
}

/// Channels and shared state one lane thread runs with.
pub(super) struct LaneLinks<'a> {
    pub(super) configs: Receiver<Arc<PartConfig>>,
    pub(super) done: SyncSender<Result<PartDone, PpoError>>,
    pub(super) pool: SimPool,
    pub(super) schedule: &'a GameSchedule,
    pub(super) stop: &'a AtomicBool,
}

struct LaneModels {
    actor: PolicyModel,
    version: u64,
    snapshots: Vec<PolicyModel>,
}

impl LaneModels {
    fn new(settings: &LaneSettings, config: &PartConfig) -> Result<Self, PpoError> {
        let load = |parameters: &[f32]| -> Result<PolicyModel, PpoError> {
            let model = PolicyModel::fresh_on(0, settings.device).map_err(text_error)?;
            model.import_parameters(parameters).map_err(text_error)?;
            Ok(model)
        };
        Ok(Self {
            actor: load(&config.actor)?,
            version: config.version,
            snapshots: settings
                .snapshots
                .iter()
                .map(|parameters| load(parameters))
                .collect::<Result<_, _>>()?,
        })
    }

    fn install(&mut self, config: &PartConfig) -> Result<(), PpoError> {
        if config.version != self.version {
            self.actor
                .import_parameters(&config.actor)
                .map_err(text_error)?;
            self.version = config.version;
        }
        Ok(())
    }
}

/// Runs one lane until the session stops it or a part fails.
pub(super) fn run_lane(
    settings: LaneSettings,
    start: LaneStart,
    links: LaneLinks<'_>,
) -> Result<(), PpoError> {
    let result = run_lane_inner(&settings, start, &links);
    if let Err(error) = &result {
        // The session is waiting for parts; hand it the failure instead.
        let _ = links.done.send(Err(error.clone()));
    }
    result
}

fn run_lane_inner(
    settings: &LaneSettings,
    start: LaneStart,
    links: &LaneLinks<'_>,
) -> Result<(), PpoError> {
    let Ok(mut config) = links.configs.recv() else {
        return Ok(());
    };
    let mut models = LaneModels::new(settings, &config)?;
    let (reply, replies) = sync_channel::<Reply>(settings.slots.len());
    let mut slots = start_slots(settings, start, &config, links, &reply, &replies)?;
    let mut part = Part::new(settings.index, config.update);
    loop {
        if links.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let started = Instant::now();
        let decisions = infer(&mut slots, &models, &config, &mut part)?;
        part.inference += started.elapsed();
        let started = Instant::now();
        let next = NextGame {
            update: config.update,
            spec: config.spec,
        };
        for (position, (slot, decision)) in slots.drain(..).zip(decisions).enumerate() {
            links.pool.submit(
                position,
                Work::Advance {
                    slot,
                    decision,
                    next,
                },
                &reply,
            )?;
        }
        slots = receive(&replies, settings.slots.len(), &mut part)?;
        part.simulation += started.elapsed();
        part.rounds += 1;
        if part.samples.len() >= settings.target {
            let snapshot = slots.iter().map(|slot| slot.snapshot()).collect();
            let finished =
                std::mem::replace(&mut part, Part::new(settings.index, config.update + 1));
            if links.done.send(Ok(finished.done(snapshot))).is_err() {
                return Ok(());
            }
            let Ok(next) = links.configs.recv() else {
                return Ok(());
            };
            if next.update != config.update + 1 {
                return Err(PpoError::InvalidTransition("lane part configuration order"));
            }
            models.install(&next)?;
            config = next;
        }
    }
}

/// One lane's accumulating share of the current update.
struct Part {
    lane: usize,
    update: u64,
    samples: Vec<LaneSample>,
    episodes: Vec<EpisodeRecord>,
    rounds: u64,
    inference: Duration,
    simulation: Duration,
}

impl Part {
    fn new(lane: usize, update: u64) -> Self {
        Self {
            lane,
            update,
            samples: Vec::new(),
            episodes: Vec::new(),
            rounds: 0,
            inference: Duration::ZERO,
            simulation: Duration::ZERO,
        }
    }

    fn done(self, snapshot: Vec<SlotSnapshot>) -> PartDone {
        PartDone {
            lane: self.lane,
            update: self.update,
            samples: self.samples,
            episodes: self.episodes,
            snapshot,
            rounds: self.rounds,
            inference: self.inference,
            simulation: self.simulation,
        }
    }
}

fn start_slots(
    settings: &LaneSettings,
    start: LaneStart,
    config: &PartConfig,
    links: &LaneLinks<'_>,
    reply: &SyncSender<Reply>,
    replies: &Receiver<Reply>,
) -> Result<Vec<Box<Slot>>, PpoError> {
    match start {
        LaneStart::Fresh => {
            for (position, &slot) in settings.slots.iter().enumerate() {
                let plan = GamePlan::new(links.schedule, slot, 0, config.update, config.spec)?;
                links.pool.submit(position, Work::Start(plan), reply)?;
            }
        }
        LaneStart::Replay(snapshots) => {
            if snapshots.len() != settings.slots.len()
                || snapshots
                    .iter()
                    .zip(&settings.slots)
                    .any(|(snapshot, &slot)| snapshot.plan.slot != slot)
            {
                return Err(PpoError::InvalidConfig("collector snapshot lane layout"));
            }
            for (position, snapshot) in snapshots.into_iter().enumerate() {
                links
                    .pool
                    .submit(position, Work::Replay(Box::new(snapshot)), reply)?;
            }
        }
    }
    let mut unused = Part::new(settings.index, config.update);
    let slots = receive(replies, settings.slots.len(), &mut unused)?;
    assert!(unused.samples.is_empty());
    assert!(unused.episodes.is_empty());
    Ok(slots)
}

/// Waits for every slot of the round and books finished games in slot order.
fn receive(
    replies: &Receiver<Reply>,
    count: usize,
    part: &mut Part,
) -> Result<Vec<Box<Slot>>, PpoError> {
    let mut outcomes: Vec<Option<Outcome>> = (0..count).map(|_| None).collect();
    let mut failure: Option<(usize, PpoError)> = None;
    for _ in 0..count {
        match replies.recv() {
            Ok(Ok(outcome)) => {
                let position = outcome.position;
                assert!(outcomes[position].is_none());
                outcomes[position] = Some(outcome);
            }
            Ok(Err((position, error))) => {
                if failure.as_ref().is_none_or(|(first, _)| position < *first) {
                    failure = Some((position, error));
                }
            }
            Err(_) => return Err(PpoError::Model("simulation pool stopped".to_owned())),
        }
    }
    if let Some((_, error)) = failure {
        return Err(error);
    }
    let mut slots = Vec::with_capacity(count);
    for outcome in outcomes {
        let outcome = outcome.ok_or(PpoError::InvalidTransition("missing slot reply"))?;
        if let Some(finished) = outcome.finished {
            if let Some(transition) = finished.terminal {
                part.samples.push(LaneSample {
                    slot: finished.slot,
                    game: finished.game,
                    transition,
                });
            }
            part.episodes.push(finished.record);
        }
        slots.push(outcome.slot);
    }
    Ok(slots)
}

/// One row of a batched sampling call and where its result belongs.
struct RowRef {
    position: usize,
    opponent: bool,
}

/// Samples every slot's policy decision and neural opponent, closes intervals
/// that waited for this round's values, and returns one decision per slot.
fn infer(
    slots: &mut [Box<Slot>],
    models: &LaneModels,
    config: &PartConfig,
    part: &mut Part,
) -> Result<Vec<Decision>, PpoError> {
    let mut actions: Vec<Option<crate::SampledRow>> = (0..slots.len()).map(|_| None).collect();
    let mut opponents: Vec<Option<crate::SampledRow>> = (0..slots.len()).map(|_| None).collect();
    let groups = row_groups(slots);
    for (model, rows) in &groups {
        let model = match model {
            None => &models.actor,
            Some(index) => models
                .snapshots
                .get(*index)
                .ok_or(PpoError::InvalidConfig("opponent snapshot index"))?,
        };
        let sampled = sample_group(slots, model, rows)?;
        for (row, sampled) in rows.iter().zip(sampled) {
            let target = if row.opponent {
                &mut opponents[row.position]
            } else {
                &mut actions[row.position]
            };
            *target = Some(sampled);
        }
    }
    let mut decisions = Vec::with_capacity(slots.len());
    for (slot, (action, opponent)) in slots.iter_mut().zip(actions.into_iter().zip(opponents)) {
        let action = action.ok_or(PpoError::InvalidTransition("missing policy row"))?;
        if slot.stream.awaits_value() {
            part.samples.push(LaneSample {
                slot: slot.plan.slot,
                game: slot.plan.game,
                transition: slot.stream.flush(Some(action.value))?,
            });
        }
        let retained = action
            .statistics
            .map(|statistics| Box::new((statistics, action.value, config.version)));
        decisions.push(Decision {
            action: action.action,
            retained,
            opponent: opponent.map(|row| row.action),
        });
    }
    Ok(decisions)
}

/// The sampling calls of one round: the actor's rows (policy and self-play
/// opponents) first, then one group per frozen snapshot in snapshot order.
fn row_groups(slots: &[Box<Slot>]) -> Vec<(Option<usize>, Vec<RowRef>)> {
    let mut groups: Vec<(Option<usize>, Vec<RowRef>)> = vec![(None, Vec::new())];
    for (position, slot) in slots.iter().enumerate() {
        groups[0].1.push(RowRef {
            position,
            opponent: false,
        });
        let group = match slot.plan.opponent {
            OpponentKind::Teacher | OpponentKind::HarassPush => continue,
            OpponentKind::SelfPlay => 0,
            OpponentKind::Snapshot(index) => {
                match groups.iter().position(|(model, _)| *model == Some(index)) {
                    Some(group) => group,
                    None => {
                        groups.push((Some(index), Vec::new()));
                        groups.len() - 1
                    }
                }
            }
        };
        groups[group].1.push(RowRef {
            position,
            opponent: true,
        });
    }
    groups[1..].sort_by_key(|(model, _)| *model);
    groups
}

/// One batched sampling call over rows of `slots`, advancing each row's RNG.
fn sample_group(
    slots: &mut [Box<Slot>],
    model: &PolicyModel,
    rows: &[RowRef],
) -> Result<Vec<crate::SampledRow>, PpoError> {
    let mut random: Vec<PpoRng> = Vec::with_capacity(rows.len());
    let mut statistics = Vec::with_capacity(rows.len());
    let mut encoded: Vec<&EncoderRow> = Vec::with_capacity(rows.len());
    let mut spaces = Vec::with_capacity(rows.len());
    for row in rows {
        let slot = &slots[row.position];
        let (seat, rng) = if row.opponent {
            let seat = slot
                .opponent
                .as_ref()
                .ok_or(PpoError::InvalidTransition("missing opponent row"))?;
            (seat, &slot.opponent_random)
        } else {
            (&slot.policy, &slot.actor_random)
        };
        encoded.push(&seat.row);
        spaces.push(&seat.space);
        random.push(rng.clone());
        statistics.push(!row.opponent && slot.stream.begins_interval());
    }
    let sampled = model
        .sample_rows(&encoded, &spaces, &mut random, &statistics)
        .map_err(text_error)?;
    for (row, random) in rows.iter().zip(random) {
        let slot = &mut slots[row.position];
        if row.opponent {
            slot.opponent_random = random;
        } else {
            slot.actor_random = random;
        }
    }
    Ok(sampled)
}
