use std::fmt;
use std::time::Duration;

use super::LatencyHistogram;

pub(crate) const DEBUG_RECORD_LIMIT: u32 = 32;
const DEBUG_CONFIG_ERROR: &str = "DRYSUA_PERF_DEBUG_EVERY must be an integer in 0..=4294967295";

#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveConfig {
    pub(crate) report_every: u32,
    pub(crate) debug_every: u32,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            report_every: 300,
            debug_every: 0,
        }
    }
}

impl LiveConfig {
    pub(crate) fn new(report_every: u32, debug_every: u32) -> Result<Self, &'static str> {
        if !(1..=30_000).contains(&report_every) {
            return Err("performance report interval must be in 1..=30000");
        }
        Ok(Self {
            report_every,
            debug_every,
        })
    }

    pub(crate) fn from_debug_value(value: Option<&str>) -> Result<Self, &'static str> {
        let Some(value) = value else {
            return Ok(Self::default());
        };
        if value.is_empty() || value.len() > 10 || !value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(DEBUG_CONFIG_ERROR);
        }
        Self::new(300, value.parse().map_err(|_| DEBUG_CONFIG_ERROR)?)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UpdateTiming {
    /// Internal elapsed work, including decode when socket-read timing is available; not CPU time.
    pub(crate) compute: Duration,
    pub(crate) receive_wait: Duration,
    /// Subset of compute; excludes the order-send call.
    pub(crate) decision: Option<Duration>,
    /// Encoding plus the write call, not delivery or server acceptance latency.
    pub(crate) order_send: Option<Duration>,
    pub(crate) ack_send: Option<Duration>,
    pub(crate) saturated: bool,
}

impl UpdateTiming {
    pub(crate) fn finish_handler_span(
        &mut self,
        started: Duration,
        ended: Duration,
        decision: Option<(Duration, Duration)>,
    ) {
        let elapsed = subtract_duration(ended, started, &mut self.saturated);
        let decision = decision.map(|(begin, end)| {
            self.saturated |= begin < started || end > ended;
            subtract_duration(end, begin, &mut self.saturated)
        });
        self.finish_handler(elapsed, decision);
    }

    pub(crate) fn merge(&mut self, timing: Self) {
        self.saturated |= timing.saturated;
        self.compute = add_duration(self.compute, timing.compute, &mut self.saturated);
        self.receive_wait =
            add_duration(self.receive_wait, timing.receive_wait, &mut self.saturated);
        for (target, source) in [
            (&mut self.decision, timing.decision),
            (&mut self.order_send, timing.order_send),
            (&mut self.ack_send, timing.ack_send),
        ] {
            if let Some(source) = source {
                *target = Some(add_duration(
                    target.unwrap_or_default(),
                    source,
                    &mut self.saturated,
                ));
            }
        }
    }

    pub(crate) fn finish_handler(&mut self, elapsed: Duration, decision: Option<Duration>) {
        let sends = add_duration(
            self.order_send.unwrap_or_default(),
            self.ack_send.unwrap_or_default(),
            &mut self.saturated,
        );
        let compute = subtract_duration(elapsed, sends, &mut self.saturated);
        self.compute = add_duration(self.compute, compute, &mut self.saturated);
        self.decision = decision.map(|duration| {
            subtract_duration(
                duration,
                self.order_send.unwrap_or_default(),
                &mut self.saturated,
            )
        });
        self.saturated |= self.decision.is_some_and(|decision| decision > compute);
    }
}

pub(crate) fn add_duration(left: Duration, right: Duration, saturated: &mut bool) -> Duration {
    left.checked_add(right).unwrap_or_else(|| {
        *saturated = true;
        Duration::MAX
    })
}

pub(crate) fn subtract_duration(left: Duration, right: Duration, saturated: &mut bool) -> Duration {
    left.checked_sub(right).unwrap_or_else(|| {
        *saturated = true;
        Duration::ZERO
    })
}

#[derive(Clone, Default)]
struct LiveStats {
    compute: LatencyHistogram,
    receive_wait: LatencyHistogram,
    decision: LatencyHistogram,
    order_send: LatencyHistogram,
    ack_send: LatencyHistogram,
    compute_overruns: u64,
    service_overruns: u64,
    timing_saturated: bool,
}

impl LiveStats {
    fn record(&mut self, timing: UpdateTiming, budget: Duration) {
        self.timing_saturated |= timing.saturated;
        self.compute.record(timing.compute);
        self.receive_wait.record(timing.receive_wait);
        for (histogram, duration) in [
            (&mut self.decision, timing.decision),
            (&mut self.order_send, timing.order_send),
            (&mut self.ack_send, timing.ack_send),
        ] {
            if let Some(duration) = duration {
                histogram.record(duration);
            }
        }
        if !timing.saturated && timing.compute > budget {
            self.compute_overruns = self.compute_overruns.saturating_add(1);
        }
        let service = timing
            .compute
            .saturating_add(timing.order_send.unwrap_or_default())
            .saturating_add(timing.ack_send.unwrap_or_default());
        if !timing.saturated && service > budget {
            self.service_overruns = self.service_overruns.saturating_add(1);
        }
    }

    fn saturated(&self) -> bool {
        self.timing_saturated
            || [
                &self.compute,
                &self.receive_wait,
                &self.decision,
                &self.order_send,
                &self.ack_send,
            ]
            .iter()
            .any(|histogram| histogram.saturated())
    }
}

pub(crate) struct LiveTelemetry {
    config: LiveConfig,
    tick_rate: u16,
    budget: Duration,
    total: LiveStats,
    window: LiveStats,
    started: Duration,
    window_started: Duration,
    last_completed: Duration,
    first_tick: Option<u32>,
    window_tick: Option<u32>,
    last_tick: Option<u32>,
    debug_decisions: u64,
    debug_records: u32,
}

const _: () = assert!(std::mem::size_of::<LiveTelemetry>() <= 16 * 1024);

impl LiveTelemetry {
    pub(crate) fn new(tick_rate: u16, config: LiveConfig) -> Result<Self, &'static str> {
        if tick_rate == 0 {
            return Err("performance tick rate must be positive");
        }
        LiveConfig::new(config.report_every, config.debug_every)?;
        Ok(Self {
            config,
            tick_rate,
            budget: Duration::from_secs(1) / u32::from(tick_rate),
            total: LiveStats::default(),
            window: LiveStats::default(),
            started: Duration::ZERO,
            window_started: Duration::ZERO,
            last_completed: Duration::ZERO,
            first_tick: None,
            window_tick: None,
            last_tick: None,
            debug_decisions: 0,
            debug_records: 0,
        })
    }

    pub(crate) fn start(&mut self, now: Duration) {
        assert_eq!(self.total.compute.count(), 0);
        assert!(self.last_tick.is_none());
        self.started = now;
        self.window_started = now;
        self.last_completed = now;
    }

    pub(crate) fn record(
        &mut self,
        tick: u32,
        now: Duration,
        timing: UpdateTiming,
    ) -> Result<bool, &'static str> {
        if self.last_tick.is_some_and(|previous| tick <= previous) {
            return Err("performance completed tick must increase");
        }
        if now < self.last_completed {
            return Err("performance clock must not regress");
        }
        self.first_tick.get_or_insert(tick);
        self.window_tick.get_or_insert(tick);
        self.last_tick = Some(tick);
        self.last_completed = now;
        self.total.record(timing, self.budget);
        self.window.record(timing, self.budget);
        Ok(self.window.compute.count() >= u64::from(self.config.report_every))
    }

    pub(crate) fn reset_window(&mut self) {
        self.window = LiveStats::default();
        self.window_started = self.last_completed;
        self.window_tick = self.last_tick;
    }

    pub(crate) fn debug_due(&mut self) -> bool {
        if self.config.debug_every == 0 || self.debug_records == DEBUG_RECORD_LIMIT {
            return false;
        }
        self.debug_decisions = self.debug_decisions.saturating_add(1);
        if self
            .debug_decisions
            .is_multiple_of(u64::from(self.config.debug_every))
        {
            self.debug_records += 1;
            return true;
        }
        false
    }

    pub(crate) fn report(
        &self,
        scope: ReportScope,
        reason: FinishReason,
        pending_update: bool,
    ) -> LiveReport<'_> {
        let (stats, started, first_tick) = match scope {
            ReportScope::Window => (&self.window, self.window_started, self.window_tick),
            ReportScope::Total => (&self.total, self.started, self.first_tick),
        };
        LiveReport {
            stats,
            scope,
            reason,
            pending_update,
            tick_rate: self.tick_rate,
            budget: self.budget,
            elapsed: self.last_completed.saturating_sub(started),
            first_tick,
            last_tick: self.last_tick,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ReportScope {
    Window,
    Total,
}

#[derive(Clone, Copy)]
pub(crate) enum FinishReason {
    Periodic,
    MatchOver,
    Limit,
    Error,
}

pub(crate) struct LiveReport<'a> {
    stats: &'a LiveStats,
    scope: ReportScope,
    reason: FinishReason,
    tick_rate: u16,
    budget: Duration,
    elapsed: Duration,
    first_tick: Option<u32>,
    last_tick: Option<u32>,
    pending_update: bool,
}

impl LiveReport<'_> {
    pub(crate) fn is_valid(&self) -> bool {
        !self.stats.saturated()
    }
}

impl fmt::Display for LiveReport<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats;
        let level = if stats.service_overruns > 0 {
            "WARN"
        } else {
            "INFO"
        };
        let scope = match self.scope {
            ReportScope::Window => "window",
            ReportScope::Total => "total",
        };
        let reason = match self.reason {
            FinishReason::Periodic => "periodic",
            FinishReason::MatchOver => "match_over",
            FinishReason::Limit => "limit",
            FinishReason::Error => "error",
        };
        let progress = self
            .last_tick
            .zip(self.first_tick)
            .map_or(0, |(last, first)| last - first);
        write!(
            output,
            "level={level} event=live_performance scope={scope} reason={reason} updates={} progress_ticks={progress} elapsed_ns={} updates_per_second={} realtime_factor={} tick_rate={} budget_ns={} compute_overruns={} service_overruns={} percentiles=log2_upper_bounds",
            stats.compute.count(),
            self.elapsed.as_nanos(),
            Rate::new(stats.compute.count(), self.elapsed, 1),
            Rate::new(u64::from(progress), self.elapsed, self.tick_rate),
            self.tick_rate,
            self.budget.as_nanos(),
            stats.compute_overruns,
            stats.service_overruns
        )?;
        for (name, histogram) in [
            ("compute", &stats.compute),
            ("receive_wait", &stats.receive_wait),
            ("decision", &stats.decision),
            ("order_send", &stats.order_send),
            ("ack_send", &stats.ack_send),
        ] {
            histogram.write_fields(output, name)?;
        }
        write!(
            output,
            " pending_update={} saturated={}",
            self.pending_update,
            stats.saturated()
        )
    }
}

struct Rate(Option<u128>);

impl Rate {
    fn new(count: u64, elapsed: Duration, tick_rate: u16) -> Self {
        let divisor = elapsed.as_nanos() * u128::from(tick_rate);
        Self((u128::from(count) * 1_000_000_000_000).checked_div(divisor))
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(milli) => write!(output, "{}.{:03}", milli / 1000, milli % 1000),
            None => output.write_str("unknown"),
        }
    }
}
