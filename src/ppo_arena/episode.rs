#![allow(
    clippy::float_arithmetic,
    reason = "Discounted n-step rewards use floating-point arithmetic"
)]

//! One game's retained intervals, rewards and terminal report.
//!
//! Every eighth decision (from a per-game phase) begins a retained interval:
//! its behaviour statistics are kept until the interval's eight decisions, or
//! the game, end. The interval then closes into one PPO transition whose
//! reward is the discounted sum of the interval's per-decision rewards.

use super::game_summary::GameSummary;
use super::*;

#[cfg(test)]
#[path = "../tests/map2_collection.rs"]
mod map2_tests;

#[cfg(test)]
#[path = "../tests/raze_aim.rs"]
mod raze_aim_tests;

pub(super) const TICK_CAP: u32 = crate::MAP2_TICK_CAP;
pub(super) const ACTOR_DECISIONS: usize = crate::MAP2_ACTOR_DECISIONS;
pub(super) const RETENTION_STRIDE: usize = crate::MAP2_RETENTION_STRIDE;
const RETAINED_PER_EPISODE: usize = crate::MAP2_RETAINED_DECISIONS;
const _: () = assert!(RETENTION_STRIDE.is_power_of_two());

/// Behaviour statistics of the decision that began the open retained interval.
pub(super) struct RetainedChoice {
    pub(super) frame: FeatureFrame,
    pub(super) target: BehavioralTarget,
    pub(super) action: StructuredAction,
    /// Policy version (completed updates) of the actor weights that sampled it.
    pub(super) behaviour: u64,
    pub(super) log_probability: f32,
    pub(super) value: f32,
}

/// Running state of one game from the policy seat's view.
pub(super) struct EpisodeStream {
    retention_phase: usize,
    map2_reward: Map2TrainingReward,
    elapsed_ticks: u32,
    raw_return: f64,
    discounted_return: f64,
    shaping_return: f64,
    terminal_reward: f32,
    actions: [u32; ActionKind::COUNT],
    choice: Option<RetainedChoice>,
    interval: DiscountedInterval,
    decisions: usize,
    retained: u32,
    done: bool,
}

impl EpisodeStream {
    pub(super) fn new(retention_phase: usize) -> Self {
        assert!(retention_phase < RETENTION_STRIDE);
        Self {
            retention_phase,
            map2_reward: Map2TrainingReward::default(),
            elapsed_ticks: 0,
            raw_return: 0.0,
            discounted_return: 0.0,
            shaping_return: 0.0,
            terminal_reward: 0.0,
            actions: [0; ActionKind::COUNT],
            choice: None,
            interval: DiscountedInterval::default(),
            decisions: 0,
            retained: 0,
            done: false,
        }
    }

    /// Whether the next decision begins a retained interval.
    pub(super) fn begins_interval(&self) -> bool {
        self.decisions >= self.retention_phase
            && (self.decisions - self.retention_phase).is_multiple_of(RETENTION_STRIDE)
    }

    /// A full interval waits for the next decision's value as its bootstrap.
    pub(super) fn awaits_value(&self) -> bool {
        !self.done && self.choice.is_some() && self.interval.steps == RETENTION_STRIDE
    }

    #[cfg(test)]
    pub(super) const fn map2_reward(&self) -> Map2TrainingReward {
        self.map2_reward
    }

    #[cfg(test)]
    pub(super) const fn raw_return(&self) -> f64 {
        self.raw_return
    }

    pub(super) const fn retention_phase(&self) -> usize {
        self.retention_phase
    }

    pub(super) const fn decisions(&self) -> usize {
        self.decisions
    }

    pub(super) const fn done(&self) -> bool {
        self.done
    }

    pub(super) const fn retained_choice(&self) -> Option<&RetainedChoice> {
        self.choice.as_ref()
    }

    /// Opens a retained interval at a decision for which `begins_interval` holds.
    pub(super) fn retain(&mut self, choice: RetainedChoice) {
        assert!(self.begins_interval());
        assert!(self.choice.is_none());
        assert_eq!(self.interval.steps, 0);
        self.choice = Some(choice);
    }

    /// Books one advanced decision: reward, action count and the open interval.
    pub(super) fn record_decision(
        &mut self,
        environment: &mut TrainingEnvironment,
        kind: ActionKind,
        advanced: &CompletedAdvance,
        gamma: f32,
    ) -> Result<(), PpoError> {
        assert!(!self.done);
        assert!(self.decisions < ACTOR_DECISIONS);
        self.done = advanced.done;
        let reward = self.observe_reward(environment, advanced.outcome, advanced.ticks, gamma)?;
        self.actions[kind.index()] += 1;
        if self.choice.is_some() {
            self.interval.append(reward, advanced.ticks, gamma)?;
        } else {
            assert_eq!(self.interval.steps, 0);
        }
        self.decisions += 1;
        Ok(())
    }

    fn observe_reward(
        &mut self,
        environment: &mut TrainingEnvironment,
        outcome: Option<PpoTerminalOutcome>,
        ticks: u32,
        gamma: f32,
    ) -> Result<f64, PpoError> {
        assert!(ticks > 0);
        assert!(self.elapsed_ticks < TICK_CAP);
        assert_eq!(gamma, MAP2_REWARD_GAMMA_TICK);
        let end = map2_reward_end(outcome).or_else(|| self.done.then_some(Map2RewardEnd::TimeCap));
        let reward = take_map2_reward(environment, end, ticks)?;
        self.map2_reward.record(reward)?;
        let emitted = reward.total;
        self.raw_return += emitted;
        self.discounted_return += f64::from(gamma).powi(self.elapsed_ticks as i32) * emitted;
        self.shaping_return += reward.total - reward.terminal;
        self.terminal_reward = reward.terminal as f32;
        self.elapsed_ticks += ticks;
        assert_eq!(self.raw_return, self.discounted_return);
        Ok(emitted)
    }

    /// Closes the open interval: terminal at the game's end, otherwise
    /// bootstrapped from the next decision's value.
    pub(super) fn flush(&mut self, next_value: Option<f32>) -> Result<PpoTransition, PpoError> {
        assert!(self.retained < RETAINED_PER_EPISODE as u32);
        assert_eq!(next_value.is_none(), self.done);
        let choice = self
            .choice
            .take()
            .ok_or(PpoError::InvalidTransition("episode retained action"))?;
        let interval = std::mem::take(&mut self.interval);
        assert!(interval.steps > 0);
        assert!(interval.steps == RETENTION_STRIDE || self.done);
        let transition = PpoTransition {
            frame: choice.frame,
            target: choice.target,
            action: choice.action,
            behaviour: choice.behaviour,
            stream: 0,
            decision: self.retained,
            ticks: interval.ticks,
            old_log_probability: choice.log_probability,
            old_value: choice.value,
            next_value: next_value.unwrap_or(0.0),
            reward: interval.reward as f32,
            terminal: self.done,
        };
        self.retained += 1;
        crate::ppo::validate_transition(&transition)?;
        Ok(transition)
    }

    /// The terminal report of a finished game.
    pub(super) fn record(
        &self,
        slot: usize,
        game: u64,
        advanced: &CompletedAdvance,
        opponent: &str,
        summary: &GameSummary,
    ) -> EpisodeRecord {
        assert!(self.done);
        assert!(self.choice.is_none());
        let label = match advanced.outcome {
            Some(PpoTerminalOutcome::Win) => "Win",
            Some(PpoTerminalOutcome::Loss) => "Loss",
            Some(PpoTerminalOutcome::Draw) => "Draw",
            None => "TimeCap",
        };
        let tick = advanced.end_tick;
        let episode = format!(
            "episode: slot={slot} game={game} map=2 opponent={opponent} tick={tick} outcome={label} actor_decisions={} retained={} terminal_sample={} raw_return={:.9} discounted_return={:.9} terminal_reward={} shaping_return={:.9} actions={:?} noncontinue={} retention_phase={}",
            self.decisions,
            self.retained,
            self.retained > 0,
            self.raw_return,
            self.discounted_return,
            self.terminal_reward,
            self.shaping_return,
            self.actions,
            self.decisions - self.actions[ActionKind::Continue.index()] as usize,
            self.retention_phase
        );
        let reward = format!(
            "level=INFO event=map2_episode_reward slot={slot} game={game} tick={tick} outcome={label} opponent={opponent} {}",
            self.map2_reward
        );
        let summary = format!(
            "level=INFO event=episode_summary slot={slot} game={game} opponent={opponent} {summary}"
        );
        EpisodeRecord {
            outcome: advanced.outcome,
            elapsed_ticks: u64::from(self.elapsed_ticks),
            map2_reward: self.map2_reward,
            lines: [episode, reward, summary],
        }
    }
}

#[derive(Default)]
struct DiscountedInterval {
    reward: f64,
    ticks: u32,
    steps: usize,
}

impl DiscountedInterval {
    fn append(&mut self, reward: f64, ticks: u32, gamma: f32) -> Result<(), PpoError> {
        if !reward.is_finite() {
            return Err(PpoError::NonFinite("episode reward"));
        }
        let _ = tick_discount(gamma, ticks)?;
        assert!(self.steps < RETENTION_STRIDE);
        let discount = if self.ticks == 0 {
            1.0
        } else {
            tick_discount(gamma, self.ticks)?
        };
        self.reward += f64::from(discount) * reward;
        self.ticks = self
            .ticks
            .checked_add(ticks)
            .ok_or(PpoError::CounterOverflow)?;
        self.steps += 1;
        assert!(self.reward.is_finite());
        Ok(())
    }
}

/// One advanced decision as observed by the policy seat.
#[derive(Clone, Copy)]
pub(super) struct CompletedAdvance {
    pub(super) end_tick: u32,
    pub(super) ticks: u32,
    pub(super) outcome: Option<PpoTerminalOutcome>,
    /// Terminal outcome, tick cap or the game's decision cap.
    pub(super) done: bool,
}

/// One finished game: counters for the update report and deferred log lines.
pub(super) struct EpisodeRecord {
    outcome: Option<PpoTerminalOutcome>,
    elapsed_ticks: u64,
    map2_reward: Map2TrainingReward,
    lines: [String; 3],
}

impl EpisodeRecord {
    #[cfg(test)]
    pub(super) const fn total_reward(&self) -> f64 {
        self.map2_reward.total
    }

    /// Adds this game to one update's counters.
    pub(super) fn accumulate(&self, report: &mut CollectionReport) -> Result<(), PpoError> {
        let counter = match self.outcome {
            Some(PpoTerminalOutcome::Win) => &mut report.terminal_wins,
            Some(PpoTerminalOutcome::Loss) => &mut report.terminal_losses,
            Some(PpoTerminalOutcome::Draw) => &mut report.terminal_draws,
            None => &mut report.episode_timeouts,
        };
        *counter = counter.checked_add(1).ok_or(PpoError::CounterOverflow)?;
        report.elapsed_ticks = report
            .elapsed_ticks
            .checked_add(self.elapsed_ticks)
            .ok_or(PpoError::CounterOverflow)?;
        report.map2_reward.merge(self.map2_reward)
    }

    /// Writes the game's `episode:`, reward and summary lines.
    pub(super) fn log(&self) {
        crate::telemetry::log_line!("{}", self.lines[0]);
        let mut output =
            crate::telemetry::PerformanceOutput::new(crate::telemetry::AsyncLogWriter::default());
        output.emit(&format_args!("{}", self.lines[1]));
        output.emit(&format_args!("{}", self.lines[2]));
    }
}
