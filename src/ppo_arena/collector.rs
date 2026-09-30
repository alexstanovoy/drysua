//! Session side of continuous collection: lane threads, part hand-off and batches.
//!
//! The session publishes one [`PartConfig`] per update to every lane and takes
//! each update's parts in lane order. Lanes run ahead by at most the pipeline
//! depth, so buffered parts stay bounded.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use super::lane::{LaneLinks, LaneSettings, LaneStart, PartConfig, PartDone, run_lane};
use super::pool::SimPool;
use super::slot::GameSchedule;
use crate::PpoError;

/// Parts a lane may queue: its current part plus the pipelined next one.
const LANE_QUEUE: usize = 2;

pub(super) struct Collector {
    configs: Vec<SyncSender<Arc<PartConfig>>>,
    done: Receiver<Result<PartDone, PpoError>>,
    pending: Vec<PartDone>,
    stop: Arc<AtomicBool>,
}

impl Collector {
    /// Spawns one thread per lane inside `scope`; consecutive lanes share one
    /// group's pool and, when pinning, its CPUs.
    pub(super) fn spawn<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        lanes: Vec<(LaneSettings, LaneStart)>,
        pools: Vec<(SimPool, Option<Vec<usize>>)>,
        schedule: &'scope GameSchedule,
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
        Ok(Self {
            configs,
            done,
            pending: Vec::new(),
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

    /// Every lane's part of `update`, in lane order.
    pub(super) fn take(&mut self, update: u64) -> Result<Vec<PartDone>, PpoError> {
        let lanes = self.configs.len();
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
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.configs.clear();
        while self.done.try_recv().is_ok() {}
    }
}
