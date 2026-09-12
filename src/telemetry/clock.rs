use std::time::{Duration, Instant};

pub(crate) trait Clock {
    fn now(&self) -> Duration;
}

pub(crate) struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}
