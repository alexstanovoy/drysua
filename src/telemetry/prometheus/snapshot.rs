use std::io;
#[cfg(any(feature = "builtin", test))]
use std::time::Duration;

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;

const MAX_COUNTER: u64 = (1_u64 << 53) - 1;
const _: () = assert!(MAX_COUNTER < u64::MAX);

pub(super) const BUCKETS: [f64; 10] =
    [0.001, 0.01, 0.1, 1.0, 5.0, 15.0, 60.0, 300.0, 900.0, 3600.0];

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct TrainingSnapshot {
    pub scope: [u8; 32],
    pub checkpoint: [u8; 32],
    pub completed_updates: u64,
    pub updates_target: u64,
    pub samples: u64,
    pub optimizer_steps: u64,
    pub games: [u64; 4],
    pub last_update_games: [u64; 4],
    pub start_update: u64,
    pub parallel: u64,
    pub games_per_update: u64,
    pub generation: Option<u64>,
    pub scale_bp: Option<u32>,
    pub losses: Option<[f64; 4]>,
    pub durations: [DurationHistogram; 6],
    pub heartbeat: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct DurationHistogram {
    pub buckets: [u64; 10],
    pub count: u64,
    pub sum_seconds: f64,
}

impl TrainingSnapshot {
    pub(super) fn validate(&self) -> io::Result<()> {
        if self.start_update > self.completed_updates
            || self.completed_updates > self.updates_target
            || self.updates_target > 1_000_000
        {
            return Err(invalid(
                "metrics updates must satisfy start <= completed <= target <= 1000000",
            ));
        }
        if self.samples > MAX_COUNTER
            || self.optimizer_steps > MAX_COUNTER
            || self.games.iter().any(|count| *count > MAX_COUNTER)
        {
            return Err(invalid("metrics counter exceeds 9007199254740991"));
        }
        if !(1..=40).contains(&self.parallel) {
            return Err(invalid("metrics parallel must be in 1..=40"));
        }
        if self.games_per_update > 40 {
            return Err(invalid("metrics games_per_update must be at most 40"));
        }
        if self
            .last_update_games
            .iter()
            .zip(self.games)
            .any(|(last, total)| *last > total)
        {
            return Err(invalid("metrics last-update games exceed cumulative games"));
        }
        self.validate_optional_metrics()?;
        for histogram in &self.durations {
            histogram.validate()?;
        }
        assert!(self.start_update <= self.completed_updates);
        assert!(self.completed_updates <= self.updates_target);
        Ok(())
    }

    fn validate_optional_metrics(&self) -> io::Result<()> {
        if self.generation.is_some() != self.scale_bp.is_some() {
            return Err(invalid(
                "metrics generation and scale must be present together",
            ));
        }
        if self.scale_bp.is_some_and(|scale| scale > 10_000) {
            return Err(invalid("metrics scale must be at most 10000 basis points"));
        }
        if self
            .losses
            .is_some_and(|losses| losses.iter().any(|loss| !loss.is_finite()))
        {
            return Err(invalid("metrics losses must be finite"));
        }
        Ok(())
    }
}

impl DurationHistogram {
    /// Records standard double-precision seconds, publishing nothing on failure.
    #[cfg(any(feature = "builtin", test))]
    #[allow(
        clippy::float_arithmetic,
        reason = "Prometheus histogram sums use finite double-precision seconds"
    )]
    pub(super) fn observe(&mut self, duration: Duration) -> io::Result<()> {
        self.validate()?;
        let seconds = duration.as_secs_f64();
        assert!(seconds.is_finite());
        assert!(seconds >= 0.0);
        let mut next = *self;
        next.count = next
            .count
            .checked_add(1)
            .filter(|count| *count <= MAX_COUNTER)
            .ok_or_else(|| invalid("metrics duration count exceeds 9007199254740991"))?;
        next.sum_seconds += seconds;
        for (count, upper) in next.buckets.iter_mut().zip(BUCKETS) {
            if seconds <= upper {
                *count = count
                    .checked_add(1)
                    .filter(|count| *count <= MAX_COUNTER)
                    .ok_or_else(|| invalid("metrics duration count exceeds 9007199254740991"))?;
            }
        }
        next.validate()?;
        assert!(next.count > self.count);
        assert!(next.sum_seconds >= self.sum_seconds);
        *self = next;
        Ok(())
    }

    fn validate(&self) -> io::Result<()> {
        if !self.sum_seconds.is_finite() || self.sum_seconds < 0.0 {
            return Err(invalid(
                "metrics duration sum must be finite and nonnegative",
            ));
        }
        if self.count > MAX_COUNTER {
            return Err(invalid("metrics duration count exceeds 9007199254740991"));
        }
        let mut previous = 0;
        for count in self.buckets {
            if count < previous || count > self.count {
                return Err(invalid(
                    "metrics duration buckets must be cumulative and at most count",
                ));
            }
            previous = count;
        }
        if self.count == 0 && self.sum_seconds != 0.0 {
            return Err(invalid(
                "metrics empty duration histogram must have zero sum",
            ));
        }
        assert!(self.buckets[9] <= self.count);
        assert!(self.sum_seconds.is_finite());
        Ok(())
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
