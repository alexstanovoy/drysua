use std::io;
#[cfg(any(feature = "builtin", test))]
use std::time::Duration;

#[cfg(any(feature = "builtin", test))]
use super::snapshot::DurationHistogram;
use super::snapshot::TrainingSnapshot;
use super::state::MetricsStore;

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;

pub(super) struct Registry {
    pub committed: Option<TrainingSnapshot>,
    staged: Option<TrainingSnapshot>,
    prepared: Option<TrainingSnapshot>,
    #[cfg(any(feature = "builtin", test))]
    session_games: [u64; 4],
    #[cfg(any(feature = "builtin", test))]
    timed_update: Option<u64>,
    #[cfg(any(feature = "builtin", test))]
    pub scopes: [DurationHistogram; 3],
    pub failure: Option<String>,
    store: Option<MetricsStore>,
}

#[cfg(any(feature = "builtin", test))]
pub(super) struct UpdateObservation {
    pub completed_updates: u64,
    pub samples: u64,
    pub optimizer_steps: u64,
    pub games: [u64; 4],
    pub losses: [f64; 4],
}

impl Registry {
    #[cfg(any(feature = "builtin", test))]
    pub(super) fn new(store: Option<MetricsStore>) -> Self {
        Self {
            committed: None,
            staged: None,
            prepared: None,
            session_games: [0; 4],
            timed_update: None,
            scopes: [DurationHistogram::default(); 3],
            failure: None,
            store,
        }
    }

    #[cfg(any(feature = "builtin", test))]
    pub(super) fn begin(&mut self, baseline: TrainingSnapshot, resume: bool) -> io::Result<()> {
        self.check_health()?;
        if self.staged.is_some() {
            return Err(invalid("metrics training session is already initialized"));
        }
        baseline.validate()?;
        let target = baseline.updates_target;
        let heartbeat = baseline.heartbeat;
        let mut snapshot = match &self.store {
            Some(store) => store.restore(baseline, resume)?,
            None => baseline,
        };
        if target < snapshot.completed_updates {
            return Err(invalid("metrics target precedes the restored checkpoint"));
        }
        if snapshot.updates_target != target {
            snapshot.updates_target = target;
            snapshot.heartbeat = heartbeat;
            if let Some(store) = &self.store {
                store.prepare(&snapshot)?;
                snapshot = store.commit(snapshot.completed_updates, snapshot.checkpoint)?;
            }
        }
        let staged = snapshot.clone();
        self.committed = Some(snapshot);
        self.staged = Some(staged);
        assert_eq!(self.session_games, [0; 4]);
        assert!(self.prepared.is_none());
        Ok(())
    }

    #[cfg(any(feature = "builtin", test))]
    pub(super) fn observe(&mut self, observation: UpdateObservation) -> io::Result<()> {
        self.check_health()?;
        let mut next = self.staging()?.clone();
        if next.completed_updates.checked_add(1) != Some(observation.completed_updates) {
            return Err(invalid(
                "metrics update must follow the previous completed update",
            ));
        }
        if observation.samples < next.samples || observation.optimizer_steps < next.optimizer_steps
        {
            return Err(invalid(
                "metrics absolute sample or optimizer counters regressed",
            ));
        }
        for (index, current) in observation.games.into_iter().enumerate() {
            let delta = current
                .checked_sub(self.session_games[index])
                .ok_or_else(|| invalid("metrics invocation outcome counters regressed"))?;
            next.last_update_games[index] = delta;
            next.games[index] = next.games[index]
                .checked_add(delta)
                .ok_or_else(|| invalid("metrics outcome counter overflow"))?;
        }
        next.completed_updates = observation.completed_updates;
        next.samples = observation.samples;
        next.optimizer_steps = observation.optimizer_steps;
        next.losses = Some(observation.losses);
        next.validate()?;
        self.session_games = observation.games;
        self.staged = Some(next);
        Ok(())
    }

    #[cfg(any(feature = "builtin", test))]
    pub(super) fn timing(
        &mut self,
        update: u64,
        elapsed: Duration,
        stages: [Option<Duration>; 5],
        valid: bool,
    ) -> io::Result<()> {
        self.check_health()?;
        if !valid {
            return Ok(());
        }
        let mut next = self.staging()?.clone();
        if update.checked_add(1) != Some(next.completed_updates)
            || self.timed_update == Some(update)
        {
            return Err(invalid(
                "metrics update timing is duplicated or out of order",
            ));
        }
        next.durations[0].observe(elapsed)?;
        for (histogram, duration) in next.durations[1..].iter_mut().zip(stages) {
            if let Some(duration) = duration {
                histogram.observe(duration)?;
            }
        }
        self.staged = Some(next);
        self.timed_update = Some(update);
        Ok(())
    }

    #[cfg(feature = "builtin")]
    pub(super) fn generation(&mut self, generation: u64, scale_bp: u32) -> io::Result<()> {
        self.check_health()?;
        let mut next = self.staging()?.clone();
        next.generation = Some(generation);
        next.scale_bp = Some(scale_bp);
        next.validate()?;
        self.staged = Some(next);
        Ok(())
    }

    pub(super) fn prepare(
        &mut self,
        scope: [u8; 32],
        update: u64,
        checkpoint: [u8; 32],
        heartbeat: u64,
    ) -> io::Result<()> {
        self.check_health()?;
        let mut candidate = self.staging()?.clone();
        if candidate.scope != scope || candidate.completed_updates != update {
            return Err(invalid(
                "metrics checkpoint scope or update does not match staging",
            ));
        }
        candidate.checkpoint = checkpoint;
        candidate.heartbeat = heartbeat;
        candidate.validate()?;
        if let Some(store) = &self.store {
            store.prepare(&candidate)?;
        }
        self.prepared = Some(candidate);
        Ok(())
    }

    pub(super) fn commit(&mut self, update: u64, checkpoint: [u8; 32]) -> io::Result<()> {
        self.check_health()?;
        let candidate = self
            .prepared
            .as_ref()
            .or(self.committed.as_ref())
            .ok_or_else(|| invalid("metrics commit has no snapshot"))?;
        if candidate.completed_updates != update || candidate.checkpoint != checkpoint {
            return Err(invalid("metrics commit does not match a prepared snapshot"));
        }
        let committed = match &self.store {
            Some(store) => store.commit(update, checkpoint)?,
            None => candidate.clone(),
        };
        assert_eq!(committed.completed_updates, update);
        assert_eq!(committed.checkpoint, checkpoint);
        self.staged = Some(committed.clone());
        self.committed = Some(committed);
        self.prepared = None;
        Ok(())
    }

    pub(super) fn check_health(&self) -> io::Result<()> {
        match &self.failure {
            Some(message) => Err(io::Error::other(message.clone())),
            None => Ok(()),
        }
    }

    fn staging(&self) -> io::Result<&TrainingSnapshot> {
        self.staged
            .as_ref()
            .ok_or_else(|| invalid("metrics training session is not initialized"))
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
