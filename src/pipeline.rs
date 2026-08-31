use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

use crate::{PPO_MAX_SAMPLES, PolicyModel, PpoBatch, PpoConfig, PpoError, PpoRollout};

/// Number of rollout buffers shared by actors and the learner.
pub const ACTOR_LEARNER_BUFFERS: usize = 2;
/// Maximum CPU workers that can receive immutable actor policies.
pub const ACTOR_LEARNER_MAX_WORKERS: usize = 256;

/// Monotonic policy generation published to CPU actors at rollout boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RolloutVersion(u64);

impl RolloutVersion {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Bounded actor-to-learner handoff failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineError {
    InvalidSampleCapacity {
        capacity: usize,
    },
    InvalidWorkerCount {
        workers: usize,
    },
    WorkerOutOfRange {
        worker: usize,
        workers: usize,
    },
    WorkerAlreadyTaken {
        worker: usize,
    },
    SampleCapacity {
        samples: usize,
        maximum: usize,
    },
    EmptyRollout,
    BuffersExhausted,
    QueueFull,
    QueueEmpty,
    Disconnected,
    PolicyMismatch,
    StaleRollout {
        rollout: RolloutVersion,
        learner: RolloutVersion,
    },
    FutureRollout {
        rollout: RolloutVersion,
        learner: RolloutVersion,
    },
    VersionOverflow,
    LockPoisoned,
    Model(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleCapacity { capacity } => write!(
                formatter,
                "actor-learner sample capacity {capacity} is outside 1..={PPO_MAX_SAMPLES}"
            ),
            Self::InvalidWorkerCount { workers } => write!(
                formatter,
                "actor-learner worker count {workers} is outside 1..={ACTOR_LEARNER_MAX_WORKERS}"
            ),
            Self::WorkerOutOfRange { worker, workers } => write!(
                formatter,
                "actor-learner worker {worker} is outside configured count {workers}"
            ),
            Self::WorkerAlreadyTaken { worker } => {
                write!(formatter, "actor-learner worker {worker} was already taken")
            }
            Self::SampleCapacity { samples, maximum } => write!(
                formatter,
                "actor rollout has {samples} samples, exceeding capacity {maximum}"
            ),
            Self::EmptyRollout => formatter.write_str("actor rollout is empty"),
            Self::BuffersExhausted => {
                formatter.write_str("both actor-learner rollout buffers are owned")
            }
            Self::QueueFull => formatter.write_str("actor-learner ready buffer is full"),
            Self::QueueEmpty => formatter.write_str("actor-learner ready buffer is empty"),
            Self::Disconnected => formatter.write_str("actor-learner pipeline is disconnected"),
            Self::PolicyMismatch => {
                formatter.write_str("actor rollout does not match its immutable policy lease")
            }
            Self::StaleRollout { rollout, learner } => write!(
                formatter,
                "actor rollout version {} is stale for learner version {}",
                rollout.get(),
                learner.get()
            ),
            Self::FutureRollout { rollout, learner } => write!(
                formatter,
                "actor rollout version {} is newer than learner version {}",
                rollout.get(),
                learner.get()
            ),
            Self::VersionOverflow => formatter.write_str("learner policy version is exhausted"),
            Self::LockPoisoned => formatter.write_str("actor policy publication lock is poisoned"),
            Self::Model(message) => write!(formatter, "actor policy snapshot failed: {message}"),
        }
    }
}

impl Error for PipelineError {}

struct PublishedPolicy {
    version: RolloutVersion,
    learner_lineage: std::num::NonZeroU64,
    model: Arc<PolicyModel>,
}

struct SubmittedRollout {
    version: RolloutVersion,
    rollout: Box<PpoRollout>,
    permit: BufferPermit,
}

struct BufferPool {
    sender: SyncSender<()>,
    receiver: Mutex<Receiver<()>>,
}

struct BufferPermit {
    pool: Arc<BufferPool>,
}

impl BufferPool {
    fn acquire(self: &Arc<Self>) -> Result<BufferPermit, PipelineError> {
        let receiver = self
            .receiver
            .lock()
            .map_err(|_| PipelineError::LockPoisoned)?;
        match receiver.try_recv() {
            Ok(()) => Ok(BufferPermit {
                pool: Arc::clone(self),
            }),
            Err(TryRecvError::Empty) => Err(PipelineError::BuffersExhausted),
            Err(TryRecvError::Disconnected) => Err(PipelineError::Disconnected),
        }
    }

    fn acquire_blocking(self: &Arc<Self>) -> Result<BufferPermit, PipelineError> {
        let receiver = self
            .receiver
            .lock()
            .map_err(|_| PipelineError::LockPoisoned)?;
        receiver.recv().map_err(|_| PipelineError::Disconnected)?;
        Ok(BufferPermit {
            pool: Arc::clone(self),
        })
    }
}

impl Drop for BufferPermit {
    fn drop(&mut self) {
        let _ = self.pool.sender.try_send(());
    }
}

/// One bounded CPU worker endpoint. It is intentionally not cloneable.
pub struct ActorRolloutSender {
    sender: SyncSender<SubmittedRollout>,
    published: Arc<RwLock<PublishedPolicy>>,
    buffers: Arc<BufferPool>,
    sample_capacity: usize,
}

impl ActorRolloutSender {
    /// Atomically leases a matching version and immutable CPU policy for one rollout.
    pub fn lease(&self) -> Result<ActorPolicyLease, PipelineError> {
        self.capture_lease()
    }

    /// Waits for one buffer before atomically capturing its policy generation.
    pub fn wait_rollout(&self) -> Result<(ActorPolicyLease, ActorRolloutBuffer), PipelineError> {
        let permit = self.buffers.acquire_blocking()?;
        let lease = self.capture_lease()?;
        let rollout = lease.allocate_rollout(permit)?;
        Ok((lease, rollout))
    }

    fn capture_lease(&self) -> Result<ActorPolicyLease, PipelineError> {
        let published = self
            .published
            .read()
            .map_err(|_| PipelineError::LockPoisoned)?;
        Ok(ActorPolicyLease {
            sender: self.sender.clone(),
            version: published.version,
            model: Arc::clone(&published.model),
            buffers: Arc::clone(&self.buffers),
            sample_capacity: self.sample_capacity,
        })
    }
}

/// Immutable actor policy and generation captured under one publication lock.
pub struct ActorPolicyLease {
    sender: SyncSender<SubmittedRollout>,
    version: RolloutVersion,
    model: Arc<PolicyModel>,
    buffers: Arc<BufferPool>,
    sample_capacity: usize,
}

impl ActorPolicyLease {
    pub const fn version(&self) -> RolloutVersion {
        self.version
    }

    pub fn policy(&self) -> &PolicyModel {
        &self.model
    }

    /// Claims one fixed-capacity buffer bound to the exact immutable actor weights.
    pub fn rollout(&self) -> Result<ActorRolloutBuffer, PipelineError> {
        let permit = self.buffers.acquire()?;
        self.allocate_rollout(permit)
    }

    fn allocate_rollout(&self, permit: BufferPermit) -> Result<ActorRolloutBuffer, PipelineError> {
        let policy = self
            .model
            .policy_identity()
            .map_err(|error| PipelineError::Model(error.to_string()))?;
        let rollout = PpoRollout::new(self.sample_capacity, policy)
            .map_err(|error| PipelineError::Model(error.to_string()))?;
        Ok(ActorRolloutBuffer {
            rollout: Box::new(rollout),
            permit,
        })
    }

    /// Transfers a completed rollout while retaining it when the ready slot is busy.
    pub fn try_submit(self, buffer: ActorRolloutBuffer) -> Result<(), RejectedRollout> {
        let ActorRolloutBuffer { rollout, permit } = buffer;
        let samples = rollout.len();
        if let Err(error) = validate_samples(samples, self.sample_capacity) {
            return Err(RejectedRollout::new(error, rollout, permit));
        }
        let policy = match self.model.policy_identity() {
            Ok(policy) => policy,
            Err(error) => {
                return Err(RejectedRollout::new(
                    PipelineError::Model(error.to_string()),
                    rollout,
                    permit,
                ));
            }
        };
        if rollout.policy() != policy {
            return Err(RejectedRollout::new(
                PipelineError::PolicyMismatch,
                rollout,
                permit,
            ));
        }
        let submitted = SubmittedRollout {
            version: self.version,
            rollout,
            permit,
        };
        match self.sender.try_send(submitted) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(submitted)) => Err(RejectedRollout {
                error: PipelineError::QueueFull,
                rollout: submitted.rollout,
                permit: submitted.permit,
            }),
            Err(TrySendError::Disconnected(submitted)) => Err(RejectedRollout {
                error: PipelineError::Disconnected,
                rollout: submitted.rollout,
                permit: submitted.permit,
            }),
        }
    }
}

/// One of exactly two fixed-capacity actor/learner buffers.
pub struct ActorRolloutBuffer {
    rollout: Box<PpoRollout>,
    permit: BufferPermit,
}

impl Deref for ActorRolloutBuffer {
    type Target = PpoRollout;

    fn deref(&self) -> &Self::Target {
        &self.rollout
    }
}

impl DerefMut for ActorRolloutBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rollout
    }
}

/// Rejected nonblocking submission, retaining ownership for retry or audit.
pub struct RejectedRollout {
    error: PipelineError,
    rollout: Box<PpoRollout>,
    permit: BufferPermit,
}

impl RejectedRollout {
    fn new(error: PipelineError, rollout: Box<PpoRollout>, permit: BufferPermit) -> Self {
        Self {
            error,
            rollout,
            permit,
        }
    }

    pub fn error(&self) -> &PipelineError {
        &self.error
    }

    pub fn into_rollout(self) -> ActorRolloutBuffer {
        ActorRolloutBuffer {
            rollout: self.rollout,
            permit: self.permit,
        }
    }
}

impl fmt::Debug for RejectedRollout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedRollout")
            .field("error", &self.error)
            .field("samples", &self.rollout.len())
            .finish()
    }
}

impl fmt::Display for RejectedRollout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl Error for RejectedRollout {}

/// Rollout accepted by the sole learner after a bounded generation-lag check.
pub struct AcceptedRollout {
    rollout_version: RolloutVersion,
    learner_version: RolloutVersion,
    rollout: PpoRollout,
    permit: BufferPermit,
    published: Arc<RwLock<PublishedPolicy>>,
}

impl AcceptedRollout {
    pub const fn rollout_version(&self) -> RolloutVersion {
        self.rollout_version
    }

    pub const fn learner_version(&self) -> RolloutVersion {
        self.learner_version
    }

    /// Computes GAE and retains trusted generation metadata for the learner.
    pub fn finish(self, config: PpoConfig) -> Result<PipelineBatch, PpoError> {
        Ok(PipelineBatch {
            rollout_version: self.rollout_version,
            learner_version: self.learner_version,
            batch: self.rollout.finish(config)?,
            _permit: self.permit,
            published: self.published,
        })
    }
}

/// PPO batch accepted through the versioned actor/learner boundary.
pub struct PipelineBatch {
    pub(crate) rollout_version: RolloutVersion,
    pub(crate) learner_version: RolloutVersion,
    pub(crate) batch: PpoBatch,
    _permit: BufferPermit,
    published: Arc<RwLock<PublishedPolicy>>,
}

impl PipelineBatch {
    pub const fn rollout_version(&self) -> RolloutVersion {
        self.rollout_version
    }

    pub const fn learner_version(&self) -> RolloutVersion {
        self.learner_version
    }

    pub fn batch(&self) -> &PpoBatch {
        &self.batch
    }

    pub(crate) fn lock_generation(&self) -> Result<PipelineGenerationGuard<'_>, PipelineError> {
        let guard = self
            .published
            .read()
            .map_err(|_| PipelineError::LockPoisoned)?;
        validate_rollout_version(self.rollout_version, guard.version)?;
        Ok(PipelineGenerationGuard(guard))
    }
}

pub(crate) struct PipelineGenerationGuard<'batch>(RwLockReadGuard<'batch, PublishedPolicy>);

impl PipelineGenerationGuard<'_> {
    pub(crate) fn version(&self) -> RolloutVersion {
        self.0.version
    }
}

/// Exactly two rollout owners: one learner checkout and one ready queue slot.
pub struct ActorLearnerPipeline {
    actors: Vec<Option<ActorRolloutSender>>,
    receiver: Receiver<SubmittedRollout>,
    published: Arc<RwLock<PublishedPolicy>>,
    sample_capacity: usize,
}

impl ActorLearnerPipeline {
    /// Captures an initial immutable CPU actor policy and fixed worker endpoints.
    pub fn new(
        sample_capacity: usize,
        workers: usize,
        learner: &PolicyModel,
    ) -> Result<Self, PipelineError> {
        Self::new_at(sample_capacity, workers, learner, RolloutVersion::new(0))
    }

    /// Restores a pipeline at one manifest-validated policy generation.
    pub fn new_at(
        sample_capacity: usize,
        workers: usize,
        learner: &PolicyModel,
        version: RolloutVersion,
    ) -> Result<Self, PipelineError> {
        if !(1..=PPO_MAX_SAMPLES).contains(&sample_capacity) {
            return Err(PipelineError::InvalidSampleCapacity {
                capacity: sample_capacity,
            });
        }
        if !(1..=ACTOR_LEARNER_MAX_WORKERS).contains(&workers) {
            return Err(PipelineError::InvalidWorkerCount { workers });
        }
        let identity = learner
            .policy_identity()
            .map_err(|error| PipelineError::Model(error.to_string()))?;
        let model = Arc::new(
            learner
                .actor_snapshot()
                .map_err(|error| PipelineError::Model(error.to_string()))?,
        );
        let published = Arc::new(RwLock::new(PublishedPolicy {
            version,
            learner_lineage: identity.lineage(),
            model,
        }));
        let (sender, receiver) = sync_channel(ACTOR_LEARNER_BUFFERS);
        let (buffer_sender, buffer_receiver) = sync_channel(ACTOR_LEARNER_BUFFERS);
        for _ in 0..ACTOR_LEARNER_BUFFERS {
            buffer_sender
                .try_send(())
                .map_err(|_| PipelineError::Disconnected)?;
        }
        let buffers = Arc::new(BufferPool {
            sender: buffer_sender,
            receiver: Mutex::new(buffer_receiver),
        });
        let actors = (0..workers)
            .map(|_| {
                Some(ActorRolloutSender {
                    sender: sender.clone(),
                    published: Arc::clone(&published),
                    buffers: Arc::clone(&buffers),
                    sample_capacity,
                })
            })
            .collect();
        Ok(Self {
            actors,
            receiver,
            published,
            sample_capacity,
        })
    }

    /// Transfers one configured worker endpoint exactly once.
    pub fn take_actor(&mut self, worker: usize) -> Result<ActorRolloutSender, PipelineError> {
        let workers = self.actors.len();
        let actor = self
            .actors
            .get_mut(worker)
            .ok_or(PipelineError::WorkerOutOfRange { worker, workers })?;
        actor
            .take()
            .ok_or(PipelineError::WorkerAlreadyTaken { worker })
    }

    pub fn version(&self) -> Result<RolloutVersion, PipelineError> {
        self.published
            .read()
            .map(|published| published.version)
            .map_err(|_| PipelineError::LockPoisoned)
    }

    /// Receives one ready rollout without blocking and rejects lag above one generation.
    pub fn try_accept(&mut self) -> Result<AcceptedRollout, PipelineError> {
        let submitted = match self.receiver.try_recv() {
            Ok(submitted) => submitted,
            Err(TryRecvError::Empty) => return Err(PipelineError::QueueEmpty),
            Err(TryRecvError::Disconnected) => return Err(PipelineError::Disconnected),
        };
        self.accept_submitted(submitted)
    }

    /// Waits without polling for one actor to complete a rollout.
    pub fn accept(&mut self) -> Result<AcceptedRollout, PipelineError> {
        let submitted = self
            .receiver
            .recv()
            .map_err(|_| PipelineError::Disconnected)?;
        self.accept_submitted(submitted)
    }

    fn accept_submitted(
        &self,
        submitted: SubmittedRollout,
    ) -> Result<AcceptedRollout, PipelineError> {
        let learner_version = self.version()?;
        validate_rollout_version(submitted.version, learner_version)?;
        validate_samples(submitted.rollout.len(), self.sample_capacity)?;
        Ok(AcceptedRollout {
            rollout_version: submitted.version,
            learner_version,
            rollout: *submitted.rollout,
            permit: submitted.permit,
            published: Arc::clone(&self.published),
        })
    }

    /// Publishes the next generation and matching immutable CPU weights atomically.
    pub fn publish(&mut self, learner: &PolicyModel) -> Result<RolloutVersion, PipelineError> {
        let mut published = self
            .published
            .write()
            .map_err(|_| PipelineError::LockPoisoned)?;
        let identity = learner
            .policy_identity()
            .map_err(|error| PipelineError::Model(error.to_string()))?;
        let model = Arc::new(
            learner
                .actor_snapshot()
                .map_err(|error| PipelineError::Model(error.to_string()))?,
        );
        if identity.lineage() != published.learner_lineage {
            return Err(PipelineError::PolicyMismatch);
        }
        let version = published
            .version
            .get()
            .checked_add(1)
            .map(RolloutVersion::new)
            .ok_or(PipelineError::VersionOverflow)?;
        published.version = version;
        published.model = model;
        Ok(version)
    }
}

fn validate_samples(samples: usize, maximum: usize) -> Result<(), PipelineError> {
    if samples == 0 {
        return Err(PipelineError::EmptyRollout);
    }
    if samples > maximum {
        return Err(PipelineError::SampleCapacity { samples, maximum });
    }
    Ok(())
}

fn validate_rollout_version(
    rollout: RolloutVersion,
    learner: RolloutVersion,
) -> Result<(), PipelineError> {
    if learner.get().saturating_sub(rollout.get()) > 1 {
        return Err(PipelineError::StaleRollout { rollout, learner });
    }
    if rollout > learner {
        return Err(PipelineError::FutureRollout { rollout, learner });
    }
    Ok(())
}
