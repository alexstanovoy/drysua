use std::fmt::{self, Display, Formatter};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::AsyncLogWriter;

const TRAINING_STAGE_COUNT: usize = 5;
const _: () = assert!(TrainingStage::Finalization as usize + 1 == TRAINING_STAGE_COUNT);
static TRAINING_LOG_DISABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn time_training_scope<T, E>(
    scope: TrainingTimingScope,
    completed_updates: Option<u64>,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let mut timer = TrainingScopeTimer {
        scope,
        completed_updates,
        started: Instant::now(),
        outcome: TrainingTimingOutcome::Incomplete,
    };
    let result = operation();
    timer.outcome = if result.is_ok() {
        TrainingTimingOutcome::Complete
    } else {
        TrainingTimingOutcome::Error
    };
    result
}

pub(crate) fn time_training_checkpoint<T, E>(
    completed_updates: u64,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    time_training_scope(
        TrainingTimingScope::CheckpointCaptureSaveRuntimeExport,
        Some(completed_updates),
        operation,
    )
}

pub(crate) struct TrainingUpdateTimer {
    started: Instant,
    stage_started: Instant,
    stage: TrainingStage,
    timing: TrainingUpdateTiming,
    outcome: TrainingTimingOutcome,
}

impl TrainingUpdateTimer {
    pub(crate) fn new(
        update_index: u64,
        mode: TrainingUpdateMode,
        optimizer_step_before: u64,
    ) -> Self {
        let started = Instant::now();
        Self {
            started,
            stage_started: started,
            stage: TrainingStage::RolloutInitialization,
            timing: TrainingUpdateTiming::new(update_index, mode, optimizer_step_before),
            outcome: TrainingTimingOutcome::Incomplete,
        }
    }

    pub(crate) fn enter(&mut self, stage: TrainingStage) {
        let boundary = Instant::now();
        self.timing.record(
            self.stage,
            boundary.saturating_duration_since(self.stage_started),
        );
        self.stage = stage;
        self.stage_started = boundary;
    }

    pub(crate) fn set_samples(&mut self, samples: usize) {
        self.timing.set_samples(samples);
    }

    pub(crate) fn set_optimizer_step(&mut self, optimizer_step: u64) {
        self.timing.set_optimizer_step(optimizer_step);
    }

    pub(crate) fn observe_result<T, E>(&mut self, result: Result<T, E>) -> Result<T, E> {
        if result.is_err() {
            self.outcome = TrainingTimingOutcome::Error;
        }
        result
    }

    pub(crate) fn complete(&mut self) {
        self.outcome = TrainingTimingOutcome::Complete;
    }
}

impl Drop for TrainingUpdateTimer {
    fn drop(&mut self) {
        let boundary = Instant::now();
        self.timing.record(
            self.stage,
            boundary.saturating_duration_since(self.stage_started),
        );
        self.timing.finish(
            boundary.saturating_duration_since(self.started),
            self.outcome.on_scope_exit(std::thread::panicking()),
        );
        emit_training_timing(&self.timing);
    }
}

/// One bounded update record; supplied durations keep accounting independent of the clock.
pub(crate) struct TrainingUpdateTiming {
    update_index: u64,
    mode: TrainingUpdateMode,
    optimizer_step_before: u64,
    stages: [Option<Duration>; TRAINING_STAGE_COUNT],
    measured: Duration,
    elapsed: Option<Duration>,
    last_stage: Option<TrainingStage>,
    samples: Option<usize>,
    optimizer_steps: Option<u64>,
    outcome: TrainingTimingOutcome,
    duration_overflow: bool,
    timing_valid: bool,
}

impl TrainingUpdateTiming {
    pub(crate) fn new(
        update_index: u64,
        mode: TrainingUpdateMode,
        optimizer_step_before: u64,
    ) -> Self {
        Self {
            update_index,
            mode,
            optimizer_step_before,
            stages: [None; TRAINING_STAGE_COUNT],
            measured: Duration::ZERO,
            elapsed: None,
            last_stage: None,
            samples: None,
            optimizer_steps: None,
            outcome: TrainingTimingOutcome::Incomplete,
            duration_overflow: false,
            timing_valid: true,
        }
    }

    pub(crate) fn record(&mut self, stage: TrainingStage, elapsed: Duration) {
        let index = stage as usize;
        let ordered = match self.last_stage {
            Some(previous) => (previous as usize..=previous as usize + 1).contains(&index),
            None => stage == TrainingStage::RolloutInitialization,
        };
        if self.elapsed.is_some() || !ordered {
            self.timing_valid = false;
            return;
        }
        let duration = self.stages[index].unwrap_or_default().checked_add(elapsed);
        let measured = self.measured.checked_add(elapsed);
        self.stages[index] = Some(duration.unwrap_or(Duration::MAX));
        self.measured = measured.unwrap_or(Duration::MAX);
        self.duration_overflow |= duration.is_none() || measured.is_none();
        self.last_stage = Some(stage);
    }

    pub(crate) fn set_samples(&mut self, samples: usize) {
        self.samples = Some(samples);
    }

    pub(crate) fn set_optimizer_step(&mut self, optimizer_step: u64) {
        self.optimizer_steps = optimizer_step.checked_sub(self.optimizer_step_before);
        self.timing_valid &= self.optimizer_steps.is_some();
    }

    pub(crate) fn finish(&mut self, elapsed: Duration, outcome: TrainingTimingOutcome) {
        if self.elapsed.is_some() {
            self.timing_valid = false;
            return;
        }
        self.elapsed = Some(elapsed);
        self.outcome = outcome;
        self.timing_valid &= !self.duration_overflow && self.measured <= elapsed;
        if outcome == TrainingTimingOutcome::Complete {
            self.timing_valid &= self.last_stage == Some(TrainingStage::Finalization);
        }
    }
}

impl Display for TrainingUpdateTiming {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "level=INFO event=training_update_timing update_index={} mode={} outcome={} \
             elapsed_ns={} measured_ns={}",
            self.update_index,
            self.mode,
            self.outcome,
            TrainingValue(self.elapsed.map(|duration| duration.as_nanos())),
            self.measured.as_nanos(),
        )?;
        for stage in TrainingStage::ALL {
            write!(
                formatter,
                " {stage}_ns={}",
                TrainingValue(self.stages[stage as usize].map(|duration| duration.as_nanos())),
            )?;
        }
        write!(
            formatter,
            " samples={} optimizer_steps={} last_stage={} duration_overflow={} timing_valid={}",
            TrainingValue(self.samples),
            TrainingValue(self.optimizer_steps),
            TrainingValue(self.last_stage),
            self.duration_overflow,
            self.timing_valid && self.elapsed.is_some(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum TrainingStage {
    RolloutInitialization,
    Collection,
    BatchPreparation,
    Optimization,
    Finalization,
}

impl TrainingStage {
    const ALL: [Self; TRAINING_STAGE_COUNT] = [
        Self::RolloutInitialization,
        Self::Collection,
        Self::BatchPreparation,
        Self::Optimization,
        Self::Finalization,
    ];
}

impl Display for TrainingStage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RolloutInitialization => "rollout_initialization",
            Self::Collection => "collection",
            Self::BatchPreparation => "batch_preparation",
            Self::Optimization => "optimization",
            Self::Finalization => "finalization",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainingUpdateMode {
    CompleteEpisodes,
    ResetWindow,
}

impl Display for TrainingUpdateMode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::CompleteEpisodes => "complete_episodes",
            Self::ResetWindow => "reset_window",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainingTimingOutcome {
    Complete,
    Error,
    Incomplete,
}

impl TrainingTimingOutcome {
    pub(crate) fn on_scope_exit(self, panicking: bool) -> Self {
        if panicking { Self::Incomplete } else { self }
    }
}

impl Display for TrainingTimingOutcome {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Complete => "complete",
            Self::Error => "error",
            Self::Incomplete => "incomplete",
        })
    }
}

struct TrainingScopeTimer {
    scope: TrainingTimingScope,
    completed_updates: Option<u64>,
    started: Instant,
    outcome: TrainingTimingOutcome,
}

impl Drop for TrainingScopeTimer {
    fn drop(&mut self) {
        emit_training_timing(&TrainingScopeTiming::new(
            self.scope,
            self.completed_updates,
            self.started.elapsed(),
            self.outcome.on_scope_exit(std::thread::panicking()),
        ));
    }
}

pub(crate) struct TrainingScopeTiming {
    scope: TrainingTimingScope,
    completed_updates: Option<u64>,
    elapsed: Duration,
    outcome: TrainingTimingOutcome,
}

impl TrainingScopeTiming {
    pub(crate) fn new(
        scope: TrainingTimingScope,
        completed_updates: Option<u64>,
        elapsed: Duration,
        outcome: TrainingTimingOutcome,
    ) -> Self {
        Self {
            scope,
            completed_updates,
            elapsed,
            outcome,
        }
    }
}

impl Display for TrainingScopeTiming {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "level=INFO event=training_scope_timing scope={} completed_updates={} \
             outcome={} elapsed_ns={}",
            self.scope,
            TrainingValue(self.completed_updates),
            self.outcome,
            self.elapsed.as_nanos(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainingTimingScope {
    SessionInitialization,
    CheckpointCaptureSaveRuntimeExport,
    ResumeRuntimeExport,
}

impl Display for TrainingTimingScope {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SessionInitialization => "session_initialization",
            Self::CheckpointCaptureSaveRuntimeExport => "checkpoint_capture_save_runtime_export",
            Self::ResumeRuntimeExport => "resume_runtime_export",
        })
    }
}

struct TrainingValue<T>(Option<T>);

impl<T: Display> Display for TrainingValue<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(value) => value.fmt(formatter),
            None => formatter.write_str("unknown"),
        }
    }
}

fn emit_training_timing(timing: &impl Display) {
    if !TRAINING_LOG_DISABLED.load(Ordering::Relaxed) {
        write_training_timing(
            &mut AsyncLogWriter::default(),
            &TRAINING_LOG_DISABLED,
            timing,
        );
    }
}

pub(crate) fn write_training_timing(
    writer: &mut impl Write,
    disabled: &AtomicBool,
    timing: &impl Display,
) {
    if !disabled.load(Ordering::Relaxed) && writeln!(writer, "{timing}").is_err() {
        disabled.store(true, Ordering::Relaxed);
    }
}
