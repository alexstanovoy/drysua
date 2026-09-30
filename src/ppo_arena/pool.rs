//! Fixed simulation workers shared by every inference lane.
//!
//! Workers take slot jobs from one bounded queue in any order. A job owns its
//! slot, so which worker runs it never changes the slot's result; replies go to
//! the submitting lane, which orders them by slot position.

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};

use super::slot::{Decision, FinishedGame, GamePlan, GameSchedule, NextGame, Slot, SlotSnapshot};
use crate::PpoError;

/// Work for one slot.
pub(super) enum Work {
    Start(GamePlan),
    Replay(Box<SlotSnapshot>),
    Advance {
        slot: Box<Slot>,
        decision: Decision,
        next: NextGame,
    },
}

/// A slot back from a worker, with the game it finished, if any.
pub(super) struct Outcome {
    pub(super) position: usize,
    pub(super) slot: Box<Slot>,
    pub(super) finished: Option<FinishedGame>,
}

pub(super) type Reply = Result<Outcome, (usize, PpoError)>;

struct Job {
    position: usize,
    work: Work,
    reply: SyncSender<Reply>,
}

/// A handle to the shared workers; they exit once every handle is dropped.
#[derive(Clone)]
pub(super) struct SimPool {
    jobs: SyncSender<Job>,
}

impl SimPool {
    /// Spawns `threads` workers inside `scope`.
    pub(super) fn spawn<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        threads: usize,
        capacity: usize,
        schedule: &'scope GameSchedule,
    ) -> Result<Self, PpoError> {
        if !(1..=256).contains(&threads) || !(1..=super::slot::MAX_SLOTS).contains(&capacity) {
            return Err(PpoError::InvalidConfig(
                "simulation threads or queue capacity",
            ));
        }
        let (jobs, receiver) = sync_channel::<Job>(capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..threads {
            let receiver = Arc::clone(&receiver);
            std::thread::Builder::new()
                .name(format!("sim-{index}"))
                .spawn_scoped(scope, move || work(&receiver, schedule))
                .map_err(|error| PpoError::Model(format!("simulation worker spawn: {error}")))?;
        }
        Ok(Self { jobs })
    }

    pub(super) fn submit(
        &self,
        position: usize,
        work: Work,
        reply: &SyncSender<Reply>,
    ) -> Result<(), PpoError> {
        self.jobs
            .send(Job {
                position,
                work,
                reply: reply.clone(),
            })
            .map_err(|_| PpoError::Model("simulation pool stopped".to_owned()))
    }
}

fn work(receiver: &Mutex<Receiver<Job>>, schedule: &GameSchedule) {
    loop {
        let job = match receiver.lock() {
            Ok(receiver) => receiver.recv(),
            Err(_) => return,
        };
        let Ok(job) = job else {
            return;
        };
        let position = job.position;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(job.work, schedule)))
                .unwrap_or_else(|_| Err(PpoError::Model("simulation job panicked".to_owned())));
        let reply = result.map(|(slot, finished)| Outcome {
            position,
            slot,
            finished,
        });
        // A lane that stopped no longer needs its replies.
        let _ = job.reply.send(reply.map_err(|error| (position, error)));
    }
}

fn run(work: Work, schedule: &GameSchedule) -> Result<(Box<Slot>, Option<FinishedGame>), PpoError> {
    match work {
        Work::Start(plan) => Ok((Slot::start(plan, schedule)?, None)),
        Work::Replay(snapshot) => Ok((Slot::replay(&snapshot, schedule)?, None)),
        Work::Advance {
            slot,
            decision,
            next,
        } => slot.advance(decision, schedule, next),
    }
}
