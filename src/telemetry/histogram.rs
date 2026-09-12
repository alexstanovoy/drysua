use std::fmt;
use std::time::Duration;

const BUCKETS: usize = 66;

#[derive(Clone)]
pub(crate) struct LatencyHistogram {
    buckets: [u64; BUCKETS],
    count: u64,
    total: Duration,
    maximum: Duration,
    saturated: bool,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            buckets: [0; BUCKETS],
            count: 0,
            total: Duration::ZERO,
            maximum: Duration::ZERO,
            saturated: false,
        }
    }
}

impl LatencyHistogram {
    pub(crate) fn record(&mut self, duration: Duration) {
        let Some(count) = self.count.checked_add(1) else {
            self.saturated = true;
            return;
        };
        let bucket = match u64::try_from(duration.as_nanos()) {
            Ok(0) => 0,
            Ok(nanos) => (u64::BITS - (nanos - 1).leading_zeros()) as usize + 1,
            Err(_) => BUCKETS - 1,
        };
        assert!(bucket < BUCKETS);
        assert!(self.buckets[bucket] < count);
        self.buckets[bucket] += 1;
        self.count = count;
        self.maximum = self.maximum.max(duration);
        self.total = self.total.checked_add(duration).unwrap_or_else(|| {
            self.saturated = true;
            Duration::MAX
        });
    }

    pub(crate) fn percentile_upper(&self, percentile: u8) -> Option<Duration> {
        assert!(percentile > 0, "percentile must be in 1..=100");
        assert!(percentile <= 100, "percentile must be in 1..=100");
        if self.count == 0 {
            return None;
        }
        let rank = (u128::from(self.count) * u128::from(percentile)).div_ceil(100);
        let mut cumulative = 0_u128;
        for (index, count) in self.buckets.iter().enumerate() {
            cumulative += u128::from(*count);
            if cumulative >= rank {
                let upper = match index {
                    0 => Duration::ZERO,
                    1..=64 => Duration::from_nanos(1_u64 << (index - 1)),
                    _ => self.maximum,
                };
                return Some(upper.min(self.maximum));
            }
        }
        unreachable!("histogram buckets account for every retained sample");
    }

    pub(crate) const fn count(&self) -> u64 {
        self.count
    }

    pub(crate) const fn total(&self) -> Duration {
        self.total
    }

    pub(crate) const fn max(&self) -> Duration {
        self.maximum
    }

    pub(crate) const fn saturated(&self) -> bool {
        self.saturated
    }

    pub(crate) fn write_fields(&self, output: &mut fmt::Formatter<'_>, name: &str) -> fmt::Result {
        write!(
            output,
            " {name}_count={} {name}_total_ns={} {name}_p50_upper_ns={} {name}_p95_upper_ns={} {name}_max_ns={}",
            self.count(),
            self.total().as_nanos(),
            OptionalDuration(self.percentile_upper(50)),
            OptionalDuration(self.percentile_upper(95)),
            self.max().as_nanos()
        )
    }
}

pub(crate) struct OptionalDuration(pub(crate) Option<Duration>);

impl fmt::Display for OptionalDuration {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(duration) => write!(output, "{}", duration.as_nanos()),
            None => output.write_str("unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_at_fixed_bound_drops_new_sample_and_marks_saturation() {
        let mut histogram = LatencyHistogram {
            count: u64::MAX,
            ..LatencyHistogram::default()
        };
        histogram.buckets[0] = u64::MAX;
        histogram.record(Duration::from_secs(1));
        assert_eq!(histogram.count(), u64::MAX);
        assert_eq!(histogram.total(), Duration::ZERO);
        assert_eq!(histogram.percentile_upper(95), Some(Duration::ZERO));
        assert!(histogram.saturated());
    }
}
