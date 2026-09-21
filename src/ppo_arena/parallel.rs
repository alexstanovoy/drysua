use crate::PpoError;

#[cfg(test)]
#[path = "../tests/ppo_arena_parallel_test_support.rs"]
mod test_support;

#[cfg(test)]
pub(crate) use test_support::*;

/// Persistent one-thread-per-stream workers for the complete-episode collector.
///
/// Each worker owns one environment for the whole update and is fed one
/// request at a time over a bounded channel, so per-decision work never pays
/// thread spawn/join. Replies are consumed in the submitted stream order, so
/// scheduling cannot change results. A worker that fails mid-update reports
/// the error and keeps serving later requests, letting the caller finish
/// draining in order before returning the first failure.
pub(super) struct StreamWorkers<'scope, T: Send, R: Send, O: Send> {
    senders: Vec<std::sync::mpsc::SyncSender<R>>,
    receivers: Vec<std::sync::mpsc::Receiver<Result<O, PpoError>>>,
    handles: Vec<std::thread::ScopedJoinHandle<'scope, ()>>,
    /// The workers own the worlds, so the type is named only for the contract.
    _worlds: std::marker::PhantomData<fn(T)>,
}

impl<'scope, T: Send + 'scope, R: Send + 'scope, O: Send + 'scope> StreamWorkers<'scope, T, R, O> {
    pub(super) fn spawn(
        scope: &'scope std::thread::Scope<'scope, '_>,
        worlds: &'scope mut [T],
        name_prefix: &str,
        operation: impl Fn(usize, &mut T, R) -> Result<O, PpoError> + Send + Sync + Copy + 'scope,
    ) -> Result<Self, PpoError> {
        if worlds.is_empty() || worlds.len() > crate::PPO_ANNEALED_MAX_GAMES {
            return Err(PpoError::InvalidConfig("stream worker environments"));
        }
        assert!(!name_prefix.is_empty());
        assert!(name_prefix.len() <= 32);
        let mut senders = Vec::with_capacity(worlds.len());
        let mut receivers = Vec::with_capacity(worlds.len());
        let mut handles = Vec::with_capacity(worlds.len());
        for (stream, world) in worlds.iter_mut().enumerate() {
            let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<R>(1);
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<Result<O, PpoError>>(1);
            let handle = std::thread::Builder::new()
                .name(format!("{name_prefix}-env-{stream}"))
                .spawn_scoped(scope, move || {
                    for request in request_rx {
                        if reply_tx.send(operation(stream, world, request)).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|error| PpoError::EpisodeWorker {
                    stream,
                    cause: error.to_string(),
                })?;
            senders.push(request_tx);
            receivers.push(reply_rx);
            handles.push(handle);
        }
        Ok(Self {
            senders,
            receivers,
            handles,
            _worlds: std::marker::PhantomData,
        })
    }

    pub(super) fn submit(&self, stream: usize, request: R) -> Result<(), PpoError> {
        self.senders
            .get(stream)
            .ok_or(PpoError::InvalidConfig("stream worker index"))?
            .send(request)
            .map_err(|_| PpoError::EpisodeWorker {
                stream,
                cause: "stream worker stopped".to_owned(),
            })
    }

    pub(super) fn receive(&self, streams: &[usize]) -> Result<Vec<O>, PpoError> {
        assert!(streams.len() <= self.receivers.len());
        let mut output = Vec::with_capacity(streams.len());
        let mut failure = None;
        for &stream in streams {
            let reply = self
                .receivers
                .get(stream)
                .ok_or(PpoError::InvalidConfig("stream worker index"))?
                .recv()
                .map_err(|_| PpoError::EpisodeWorker {
                    stream,
                    cause: "stream worker stopped".to_owned(),
                })?;
            match reply {
                Ok(value) => output.push(value),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(output), Err)
    }

    pub(super) fn finish(self) -> Result<(), PpoError> {
        drop(self.senders);
        for handle in self.handles {
            let _ = handle.join();
        }
        Ok(())
    }
}
