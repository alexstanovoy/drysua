use super::*;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

/// The old bound already includes one dense effective minibatch; this adds the second.
pub const PPO_PREFETCH_STORAGE_PEAK_BYTES: u64 = PPO_ANNEALED_STORAGE_PEAK_BYTES
    + MODEL_MAX_BATCH as u64 * std::mem::size_of::<PpoPreparedSample>() as u64
    + (2 * MODEL_MAX_BATCH * std::mem::size_of::<usize>()) as u64
    + 2 * 1024 * 1024;
const _: () = assert!(PPO_PREFETCH_STORAGE_PEAK_BYTES < 8 * 1024 * 1024 * 1024);

type Prepared = Result<Vec<PpoPreparedSample>, PpoError>;

impl PpoTrainer {
    pub(super) fn train_update_prefetched(
        &mut self,
        model: &PolicyModel,
        batch: &PpoBatch,
    ) -> Result<PpoUpdateReport, PpoError> {
        assert!(self.execution.learner_prefetch);
        assert!(self.config.minibatch <= MODEL_MAX_BATCH);
        let maximum_jobs = self.config.epochs * batch.len().div_ceil(self.config.minibatch);
        let (requests, jobs) = sync_channel::<Vec<usize>>(1);
        let (ready, results) = sync_channel::<Prepared>(1);
        let mut report = std::thread::scope(|scope| {
            let worker = std::thread::Builder::new()
                .name("ppo-prefetch".to_owned())
                .stack_size(2 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    for indices in jobs.iter().take(maximum_jobs) {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            batch.materialize(&indices)
                        }))
                        .unwrap_or_else(|_| {
                            Err(PpoError::Model("PPO prefetch worker panicked".to_owned()))
                        });
                        if ready.send(result).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|error| PpoError::Model(format!("PPO prefetch spawn failed: {error}")))?;
            let result = self.prefetched_epochs(model, batch.len(), &requests, &results);
            // Closing both ends releases a worker blocked on either bounded channel.
            drop(requests);
            drop(results);
            let joined = worker.join();
            if result.as_ref().is_ok_and(|report| !report.stopped_for_kl) && joined.is_err() {
                return Err(PpoError::Model("PPO prefetch worker panicked".to_owned()));
            }
            result
        })?;
        self.updates = self
            .updates
            .checked_add(1)
            .ok_or(PpoError::CounterOverflow)?;
        report.update = self.updates;
        Ok(report)
    }

    fn prefetched_epochs(
        &mut self,
        model: &PolicyModel,
        samples: usize,
        requests: &SyncSender<Vec<usize>>,
        results: &Receiver<Prepared>,
    ) -> Result<PpoUpdateReport, PpoError> {
        let mut aggregate = PpoUpdateReport::default();
        let mut order = (0..samples).collect::<Vec<_>>();
        'epochs: for epoch in 0..self.config.epochs {
            self.shuffle.shuffle(&mut order)?;
            let mut chunks = order.chunks(self.config.minibatch);
            let first = chunks.next().ok_or(PpoError::EmptyRollout)?;
            submit(requests, first)?;
            for _ in 0..samples.div_ceil(self.config.minibatch) {
                let prepared = results
                    .recv()
                    .map_err(|_| PpoError::Model("PPO prefetch worker stopped".to_owned()))??;
                let next = chunks.next();
                // Exactly one current owner plus one submitted/ready buffer, never a third job.
                if let Some(indices) = next {
                    submit(requests, indices)?;
                }
                let references = prepared.iter().collect::<Vec<_>>();
                let report = model
                    .ppo_update_with_execution(
                        &references,
                        &mut self.adam,
                        self.config,
                        self.execution,
                    )
                    .map_err(|error| PpoError::Model(error.to_string()))?;
                if !report.applied {
                    aggregate.stopped_for_kl = true;
                    record_kl_rejection(&mut aggregate, report)?;
                    break 'epochs;
                }
                aggregate_minibatch(&mut aggregate, report)?;
                if next.is_none() {
                    break;
                }
            }
            aggregate.epochs_completed = epoch + 1;
        }
        finish_update_report(&mut aggregate, self.adam.step())?;
        Ok(aggregate)
    }
}

fn submit(sender: &SyncSender<Vec<usize>>, indices: &[usize]) -> Result<(), PpoError> {
    assert!(!indices.is_empty());
    assert!(indices.len() <= MODEL_MAX_BATCH);
    sender
        .send(indices.to_vec())
        .map_err(|_| PpoError::Model("PPO prefetch worker stopped".to_owned()))
}
