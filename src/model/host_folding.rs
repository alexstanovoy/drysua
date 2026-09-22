use super::*;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

#[cfg(test)]
#[path = "../tests/host_folding.rs"]
mod tests;

const MIN_COORDINATES: usize = 32_768;
const WORKER_STACK_BYTES: usize = 256 * 1024;
const _: () = assert!(3 * MODEL_PARAMETER_COUNT * std::mem::size_of::<f32>() == 20_400_240);
const _: () = assert!(
    crate::PPO_PREFETCH_STORAGE_PEAK_BYTES
        + ((3 * MODEL_PARAMETER_COUNT * std::mem::size_of::<f32>() + 32 * WORKER_STACK_BYTES)
            as u64)
        < 8 * 1024 * 1024 * 1024
);

pub(super) fn collect(
    model: &PolicyModel,
    examples: &[&PpoPreparedSample],
    config: PpoConfig,
    microbatch: usize,
    requested: usize,
) -> Result<(Vec<f32>, PpoMinibatchReport), ModelError> {
    let workers = resolved_host_math_workers(requested);
    if workers == 1 {
        return collect_serial(model, examples, config, microbatch);
    }
    std::thread::scope(|scope| {
        let mut pool = FoldPool::spawn(
            scope,
            MODEL_PARAMETER_COUNT,
            workers,
            examples.len().div_ceil(microbatch),
        )?;
        let mut report = PpoMinibatchReport::default();
        let mut pending_report = None;
        for chunk in examples.chunks(microbatch) {
            // Only the GPU owner touches Candle. Previous host folds run during this graph.
            let current = model.ppo_microbatch_readback_locked(chunk, config, false);
            if let Some(previous) = pending_report.take() {
                pool.receive()?;
                accumulate_ppo_report(&mut report, previous)?;
            }
            // A delayed previous fold/report error outranks this later graph's failure.
            let current = current?;
            pool.submit(current.gradients, chunk.len() as f32)?;
            pending_report = Some(current.report);
        }
        if let Some(previous) = pending_report {
            pool.receive()?;
            accumulate_ppo_report(&mut report, previous)?;
        }
        Ok((pool.finish()?, report))
    })
}

fn collect_serial(
    model: &PolicyModel,
    examples: &[&PpoPreparedSample],
    config: PpoConfig,
    microbatch: usize,
) -> Result<(Vec<f32>, PpoMinibatchReport), ModelError> {
    let mut gradients = vec![0.0f32; MODEL_PARAMETER_COUNT];
    let mut report = PpoMinibatchReport::default();
    for chunk in examples.chunks(microbatch) {
        let mut result = model.ppo_microbatch_locked(chunk, config)?;
        scale_gradients(&mut result.gradients, chunk.len() as f32)?;
        accumulate_gradients(&mut gradients, &result.gradients)?;
        accumulate_ppo_report(&mut report, result.report)?;
    }
    Ok((gradients, report))
}

fn worker_count(requested: usize, available: usize, coordinates: usize) -> usize {
    assert!((1..=32).contains(&requested));
    assert!(coordinates > 0);
    requested
        .min(available.max(1))
        .min(coordinates.div_ceil(MIN_COORDINATES))
        .max(1)
}

pub(crate) fn resolved_host_math_workers(requested: usize) -> usize {
    let available = if requested == 1 {
        1
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    };
    worker_count(requested, available, MODEL_PARAMETER_COUNT)
}

struct FoldJob {
    gradient: Vec<f32>,
    scale: f32,
}
struct FoldReply {
    gradient: Vec<f32>,
    failure: Option<(u8, ModelError)>,
}

struct FoldPool<'scope> {
    // Field drop order closes both queues before scoped joins on every error exit.
    requests: Vec<SyncSender<FoldJob>>,
    replies: Vec<Receiver<FoldReply>>,
    handles: Vec<std::thread::ScopedJoinHandle<'scope, Vec<f32>>>,
    buffers: Vec<Vec<f32>>,
    ranges: Vec<std::ops::Range<usize>>,
    coordinates: usize,
    pending: bool,
}

impl<'scope> FoldPool<'scope> {
    fn spawn(
        scope: &'scope std::thread::Scope<'scope, '_>,
        coordinates: usize,
        workers: usize,
        maximum_jobs: usize,
    ) -> Result<Self, ModelError> {
        assert!((1..=MODEL_PARAMETER_COUNT).contains(&coordinates));
        assert!((1..=32).contains(&workers));
        assert!(workers <= coordinates);
        assert!((1..=MODEL_MAX_BATCH).contains(&maximum_jobs));
        let mut pool = Self {
            requests: Vec::new(),
            replies: Vec::new(),
            handles: Vec::new(),
            buffers: Vec::new(),
            ranges: Vec::new(),
            coordinates,
            pending: false,
        };
        for worker in 0..workers {
            let range = worker * coordinates / workers..(worker + 1) * coordinates / workers;
            let length = range.len();
            let (sender, jobs) = sync_channel::<FoldJob>(1);
            let (ready, receiver) = sync_channel(1);
            let handle = std::thread::Builder::new()
                .name(format!("ppo-fold-{worker}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    let mut accumulator = vec![0.0f32; length];
                    for mut job in jobs.iter().take(maximum_jobs) {
                        let failure =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                fold_chunk(&mut accumulator, &mut job.gradient, job.scale)
                            }))
                            .unwrap_or_else(|_| {
                                Err((
                                    0,
                                    ModelError::Backend("PPO fold worker panicked".to_owned()),
                                ))
                            })
                            .err();
                        if ready
                            .send(FoldReply {
                                gradient: job.gradient,
                                failure,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    accumulator
                })
                .map_err(|error| ModelError::Backend(format!("PPO fold spawn failed: {error}")))?;
            pool.requests.push(sender);
            pool.replies.push(receiver);
            pool.handles.push(handle);
            pool.buffers.push(Vec::with_capacity(length));
            pool.ranges.push(range);
        }
        Ok(pool)
    }

    fn submit(&mut self, gradient: Vec<f32>, scale: f32) -> Result<(), ModelError> {
        assert!(!self.pending);
        assert_eq!(gradient.len(), self.coordinates);
        for index in 0..self.requests.len() {
            let mut part = std::mem::take(&mut self.buffers[index]);
            part.clear();
            part.extend_from_slice(&gradient[self.ranges[index].clone()]);
            self.requests[index]
                .send(FoldJob {
                    gradient: part,
                    scale,
                })
                .map_err(|_| ModelError::Backend("PPO fold worker stopped".to_owned()))?;
        }
        // Drop the contiguous result before the next GPU readback: two results plus accumulator.
        self.pending = true;
        Ok(())
    }

    fn receive(&mut self) -> Result<(), ModelError> {
        assert!(self.pending);
        assert_eq!(self.replies.len(), self.ranges.len());
        let mut failure: Option<(u8, usize, ModelError)> = None;
        for index in 0..self.replies.len() {
            let reply = self.replies[index]
                .recv()
                .map_err(|_| ModelError::Backend("PPO fold worker stopped".to_owned()))?;
            self.buffers[index] = reply.gradient;
            if let Some((phase, error)) = reply.failure {
                let (coordinate, error) = match error {
                    ModelError::NonFiniteGradient { index: local } => {
                        let coordinate = self.ranges[index].start + local;
                        (
                            coordinate,
                            ModelError::NonFiniteGradient { index: coordinate },
                        )
                    }
                    error => (self.ranges[index].start, error),
                };
                if failure
                    .as_ref()
                    .is_none_or(|old| (phase, coordinate) < (old.0, old.1))
                {
                    failure = Some((phase, coordinate, error));
                }
            }
        }
        self.pending = false;
        failure.map_or(Ok(()), |(_, _, error)| Err(error))
    }

    fn finish(self) -> Result<Vec<f32>, ModelError> {
        assert!(!self.pending);
        drop(self.requests);
        drop(self.buffers);
        let mut output = Vec::with_capacity(self.coordinates);
        for handle in self.handles {
            output.extend(
                handle
                    .join()
                    .map_err(|_| ModelError::Backend("PPO fold worker panicked".to_owned()))?,
            );
        }
        assert_eq!(output.len(), self.coordinates);
        Ok(output)
    }
}

fn fold_chunk(
    accumulator: &mut [f32],
    gradient: &mut [f32],
    scale: f32,
) -> Result<(), (u8, ModelError)> {
    assert_eq!(accumulator.len(), gradient.len());
    assert!(!gradient.is_empty());
    validate_gradients(gradient).map_err(|error| (0, error))?;
    scale_gradients(gradient, scale).map_err(|error| (1, error))?;
    accumulate_gradients(accumulator, gradient).map_err(|error| (2, error))
}
