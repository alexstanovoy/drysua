use std::io::Write;
use std::time::Duration;

use bota_proto::{ServerMsg, TickMode};

use super::{
    DEBUG_RECORD_LIMIT, FinishReason, LiveConfig, LiveTelemetry, OptionalDuration,
    PerformanceOutput, ReportScope, UpdateTiming,
};
use crate::Seated;

#[derive(Clone, Copy)]
pub(crate) enum UpdateBoundary {
    Start,
    Snapshot,
    Complete(u32),
    Other,
}

impl UpdateBoundary {
    pub(crate) fn of(message: &ServerMsg) -> Self {
        match message {
            ServerMsg::MatchStart { .. } => Self::Start,
            ServerMsg::Snapshot { .. } => Self::Snapshot,
            ServerMsg::Events { tick, .. } => Self::Complete(*tick),
            _ => Self::Other,
        }
    }
}

pub(crate) struct LiveMonitor {
    telemetry: LiveTelemetry,
    config: LiveConfig,
    seated: Seated,
    policy: &'static str,
    pending: UpdateTiming,
    pending_snapshot: bool,
    started: bool,
    invalid: bool,
}

impl LiveMonitor {
    pub(crate) fn new(
        seated: Seated,
        policy: &'static str,
        config: LiveConfig,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            telemetry: LiveTelemetry::new(seated.tick_rate, config)?,
            config,
            seated,
            policy,
            pending: UpdateTiming::default(),
            pending_snapshot: false,
            started: false,
            invalid: false,
        })
    }

    pub(crate) fn begin(&mut self, boundary: UpdateBoundary) {
        if matches!(boundary, UpdateBoundary::Snapshot) {
            self.pending_snapshot = true;
        }
    }

    pub(crate) fn observe(
        &mut self,
        boundary: UpdateBoundary,
        now: Duration,
        timing: UpdateTiming,
        receive_scope: &str,
        output: &mut PerformanceOutput<impl Write>,
    ) {
        if matches!(boundary, UpdateBoundary::Start) {
            self.telemetry.start(now);
            self.started = true;
            output.emit(&format_args!("level=INFO event=live_performance_start slot={} policy={} mode={} tick_rate={} report_every={} debug_every={} debug_limit={} receive_wait_scope={receive_scope} compute_scope=internal_elapsed_excluding_receive_and_send",
                self.seated.slot.0, self.policy, self.mode(), self.seated.tick_rate, self.config.report_every, self.config.debug_every, DEBUG_RECORD_LIMIT));
            return;
        }
        if !self.started || self.invalid {
            return;
        }
        self.pending.merge(timing);
        let UpdateBoundary::Complete(tick) = boundary else {
            return;
        };
        let timing = std::mem::take(&mut self.pending);
        self.pending_snapshot = false;
        match self.telemetry.record(tick, now, timing) {
            Ok(due) => {
                if timing.decision.is_some() && self.telemetry.debug_due() {
                    output.emit(&format_args!("level=DEBUG event=live_decision slot={} policy={} tick={tick} decision_ns={} order_sent={} order_send_ns={} ack_send_ns={}",
                        self.seated.slot.0, self.policy, OptionalDuration(timing.decision), timing.order_send.is_some(), OptionalDuration(timing.order_send), OptionalDuration(timing.ack_send)));
                }
                if due {
                    self.emit(
                        ReportScope::Window,
                        FinishReason::Periodic,
                        receive_scope,
                        output,
                    );
                    self.telemetry.reset_window();
                }
            }
            Err(error) => {
                self.invalid = true;
                output.emit(&format_args!(
                    "level=INFO event=live_performance_disabled reason=\"{error}\""
                ));
            }
        }
    }

    pub(crate) fn finish(
        &self,
        reason: FinishReason,
        receive_scope: &str,
        output: &mut PerformanceOutput<impl Write>,
    ) {
        self.emit(ReportScope::Total, reason, receive_scope, output);
    }

    fn emit(
        &self,
        scope: ReportScope,
        reason: FinishReason,
        receive_scope: &str,
        output: &mut PerformanceOutput<impl Write>,
    ) {
        let report = self.telemetry.report(scope, reason, self.pending_snapshot);
        output.emit(&format_args!(
            "{report} slot={} policy={} mode={} receive_wait_scope={receive_scope} timing_valid={}",
            self.seated.slot.0,
            self.policy,
            self.mode(),
            !self.invalid && report.is_valid()
        ));
    }

    fn mode(&self) -> &'static str {
        match self.seated.mode {
            TickMode::Lockstep => "lockstep",
            TickMode::Realtime => "realtime",
        }
    }
}
