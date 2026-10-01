//! Session side of continuous collection: lane threads, part hand-off and batches.
//!
//! The session publishes one [`PartConfig`] per update to every lane. A
//! preparer thread takes each update's parts in lane order and builds its
//! batch while the learner still trains the previous update; the learner
//! takes the prepared updates in order. Lanes run ahead by at most the
//! pipeline depth and the preparer by one batch, so buffers stay bounded.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use rustc_hash::FxHashMap;

use super::CollectionReport;
use super::lane::{LaneLinks, LaneSettings, LaneStart, PartConfig, PartDone, run_lane};
use super::pool::SimPool;
use super::slot::{GameSchedule, SlotSnapshot};
use crate::{PPO_MAX_STREAMS, PpoBatch, PpoConfig, PpoError, PpoRollout, SideNetworks};

/// Parts a lane may queue: its current part plus the pipelined next one.
const LANE_QUEUE: usize = 2;

pub(super) struct Collector {
    configs: Vec<SyncSender<Arc<PartConfig>>>,
    prepared: Receiver<Result<Prepared, PpoError>>,
    stop: Arc<AtomicBool>,
}

/// One update's batch, built off the learner thread, with the lane parts the
/// learner still logs and books (their samples moved into the batch).
pub(super) struct Prepared {
    pub(super) update: u64,
    pub(super) parts: Vec<PartDone>,
    /// Samples each part contributed, in part order.
    pub(super) part_samples: Vec<usize>,
    pub(super) samples: usize,
    pub(super) batch: PpoBatch,
    pub(super) report: CollectionReport,
    /// Every slot's state at the start of the next update, in slot order.
    pub(super) snapshots: Vec<SlotSnapshot>,
}

/// The receiving end of the lanes' parts.
struct Parts {
    lanes: usize,
    done: Receiver<Result<PartDone, PpoError>>,
    pending: Vec<PartDone>,
}

impl Collector {
    /// Spawns one thread per lane inside `scope`, consecutive lanes sharing
    /// one group's pool and, when pinning, its CPUs, plus the preparer of the
    /// batches from `first` on.
    pub(super) fn spawn<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        lanes: Vec<(LaneSettings, LaneStart)>,
        pools: Vec<(SimPool, Option<Vec<usize>>)>,
        schedule: &'scope GameSchedule,
        (first, slots, config, side_networks): (u64, usize, PpoConfig, SideNetworks),
    ) -> Result<Self, PpoError> {
        assert!(!lanes.is_empty() && lanes.len().is_multiple_of(pools.len()));
        let lanes_per_group = lanes.len() / pools.len();
        let stop = Arc::new(AtomicBool::new(false));
        let (done_sender, done) = sync_channel(lanes.len() * LANE_QUEUE);
        let mut configs = Vec::with_capacity(lanes.len());
        for (settings, start) in lanes {
            let (sender, receiver) = sync_channel(LANE_QUEUE);
            configs.push(sender);
            let links_done = done_sender.clone();
            let lane_stop = Arc::clone(&stop);
            let (pool, cpus) = pools[settings.index / lanes_per_group].clone();
            std::thread::Builder::new()
                .name(format!("lane-{}", settings.index))
                .spawn_scoped(scope, move || {
                    if let Some(cpus) = cpus
                        && let Err(error) = super::topology::pin_current_thread(&cpus)
                    {
                        crate::telemetry::log_line!("level=WARN event=pin_failed {error}");
                    }
                    let links = LaneLinks {
                        configs: receiver,
                        done: links_done,
                        pool,
                        schedule,
                        stop: &lane_stop,
                    };
                    run_lane(settings, start, links)
                })
                .map_err(|error| PpoError::Model(format!("lane spawn: {error}")))?;
        }
        let parts = Parts {
            lanes: configs.len(),
            done,
            pending: Vec::new(),
        };
        // Rendezvous: the preparer holds at most one finished batch.
        let (sender, prepared) = sync_channel(0);
        std::thread::Builder::new()
            .name("batch-prepare".to_owned())
            .spawn_scoped(scope, move || {
                prepare_batches(parts, first, (slots, config, side_networks), &sender)
            })
            .map_err(|error| PpoError::Model(format!("preparer spawn: {error}")))?;
        Ok(Self {
            configs,
            prepared,
            stop,
        })
    }

    /// Sends one update's configuration to every lane.
    pub(super) fn publish(&self, config: &Arc<PartConfig>) -> Result<(), PpoError> {
        for sender in &self.configs {
            sender
                .send(Arc::clone(config))
                .map_err(|_| PpoError::Model("collection lane stopped".to_owned()))?;
        }
        Ok(())
    }

    /// The prepared batch of `update`; updates are taken in order.
    pub(super) fn take(&self, update: u64) -> Result<Prepared, PpoError> {
        let prepared = self
            .prepared
            .recv()
            .map_err(|_| PpoError::Model("batch preparer stopped".to_owned()))??;
        if prepared.update != update {
            return Err(PpoError::InvalidTransition("prepared batch order"));
        }
        Ok(prepared)
    }
}

impl Parts {
    /// Every lane's part of `update`, in lane order.
    fn take(&mut self, update: u64) -> Result<Vec<PartDone>, PpoError> {
        let lanes = self.lanes;
        while self
            .pending
            .iter()
            .filter(|part| part.update == update)
            .count()
            < lanes
        {
            let part = self
                .done
                .recv()
                .map_err(|_| PpoError::Model("collection lanes stopped".to_owned()))??;
            if part.update < update || part.update > update + LANE_QUEUE as u64 {
                return Err(PpoError::InvalidTransition("collection part order"));
            }
            self.pending.push(part);
            assert!(self.pending.len() <= lanes * LANE_QUEUE);
        }
        let (mut ready, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|part| part.update == update);
        self.pending = pending;
        ready.sort_by_key(|part| part.lane);
        assert!(
            ready
                .iter()
                .enumerate()
                .all(|(index, part)| part.lane == index)
        );
        Ok(ready)
    }
}

impl Drop for Collector {
    /// Stops the lanes: they leave at their next round or blocked receive.
    /// Dropping the prepared receiver then fails the preparer's next hand-off;
    /// its exit drops the parts receiver, which fails any blocked lane send.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.configs.clear();
    }
}

/// Prepares every update's batch from `first` on, in order, until the
/// learner stops taking them or a part or batch fails.
fn prepare_batches(
    mut parts: Parts,
    first: u64,
    shape: (usize, PpoConfig, SideNetworks),
    sender: &SyncSender<Result<Prepared, PpoError>>,
) {
    for update in first..=u64::MAX {
        let prepared = parts
            .take(update)
            .and_then(|lane_parts| prepare(update, lane_parts, shape));
        let failed = prepared.is_err();
        if sender.send(prepared).is_err() || failed {
            return;
        }
    }
}

/// One update's batch, report and next-update snapshots from its lane parts;
/// the samples move into the rollout (a clone would copy every frame).
fn prepare(
    update: u64,
    mut parts: Vec<PartDone>,
    (slots, config, side_networks): (usize, PpoConfig, SideNetworks),
) -> Result<Prepared, PpoError> {
    let mut rollout = PpoRollout::new(config.rollout_capacity(slots))?;
    let mut streams: FxHashMap<(usize, u64), usize> = FxHashMap::default();
    let mut report = CollectionReport::default();
    let mut snapshots: Vec<Option<SlotSnapshot>> = (0..slots).map(|_| None).collect();
    let mut part_samples = Vec::with_capacity(parts.len());
    for part in &mut parts {
        part_samples.push(part.samples.len());
        for sample in std::mem::take(&mut part.samples) {
            let next = streams.len();
            let stream = *streams.entry((sample.slot, sample.game)).or_insert(next);
            if stream >= PPO_MAX_STREAMS {
                return Err(PpoError::InvalidConfig(
                    "update exceeds its game stream bound",
                ));
            }
            let mut transition = sample.transition;
            transition.stream = stream;
            rollout.push(transition)?;
        }
        for episode in &part.episodes {
            episode.accumulate(&mut report)?;
        }
        for snapshot in &part.snapshot {
            snapshots[snapshot.plan.slot] = Some(snapshot.clone());
        }
    }
    let snapshots = snapshots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(PpoError::InvalidTransition("collection snapshot slots"))?;
    let samples = rollout.len();
    Ok(Prepared {
        update,
        parts,
        part_samples,
        samples,
        batch: rollout.finish(config, side_networks)?,
        report,
        snapshots,
    })
}
