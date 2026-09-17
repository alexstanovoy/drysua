use std::collections::VecDeque;
use std::str::FromStr;

/// Maximum completed-game flags retained for one current training opponent.
pub const MASTERY_MAX_WINDOW: usize = 1024;
pub const MASTERY_DEFAULT_WINDOW: usize = 50;
pub const MASTERY_DEFAULT_WIN_PERCENT: u8 = 80;
const _: () = assert!(MASTERY_DEFAULT_WINDOW > 0);
const _: () = assert!(MASTERY_DEFAULT_WINDOW <= MASTERY_MAX_WINDOW);
const _: () = assert!(MASTERY_MAX_WINDOW <= u16::MAX as usize);
const _: () = assert!(MASTERY_DEFAULT_WIN_PERCENT > 0);
const _: () = assert!(MASTERY_DEFAULT_WIN_PERCENT <= 100);
const _: () = assert!(MASTERY_MAX_WINDOW * 100 <= u32::MAX as usize);

/// Persisted mastery stage; Completed retains the qualifying Teacher window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum MasteryStage {
    #[default]
    Weak,
    Teacher,
    Completed,
}

/// One validated per-opponent percentage override from the CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpponentWinPercent {
    stage: MasteryStage,
    percent: u8,
}

impl FromStr for OpponentWinPercent {
    type Err = &'static str;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() > 32 {
            return Err("opponent win percent is too long");
        }
        let (opponent, percent) = text
            .split_once('=')
            .ok_or("opponent win percent must be opponent=integer")?;
        let stage = match opponent {
            "weak" => MasteryStage::Weak,
            "teacher" => MasteryStage::Teacher,
            _ => return Err("unknown mastery opponent; expected weak or teacher"),
        };
        if percent.is_empty() || !percent.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("opponent win percent must be an integer in 1..=100");
        }
        let percent = percent
            .parse::<u8>()
            .map_err(|_| "opponent win percent must be an integer in 1..=100")?;
        if !(1..=100).contains(&percent) {
            return Err("opponent win percent must be an integer in 1..=100");
        }
        Ok(Self { stage, percent })
    }
}

/// Effective bounded window and resolved Weak/Teacher win percentages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasteryConfig {
    window: u16,
    percentages: [u8; 2],
}

impl Default for MasteryConfig {
    fn default() -> Self {
        Self {
            window: MASTERY_DEFAULT_WINDOW as u16,
            percentages: [MASTERY_DEFAULT_WIN_PERCENT; 2],
        }
    }
}

impl MasteryConfig {
    pub fn new(
        window: usize,
        percent: u8,
        overrides: &[OpponentWinPercent],
    ) -> Result<Self, &'static str> {
        if !(1..=MASTERY_MAX_WINDOW).contains(&window) {
            return Err("mastery window must be in 1..=1024");
        }
        if !(1..=100).contains(&percent) {
            return Err("mastery win percent must be in 1..=100");
        }
        if overrides.len() > 2 {
            return Err("at most two opponent win percent overrides are allowed");
        }
        let mut percentages = [percent; 2];
        let mut seen = [false; 2];
        for value in overrides {
            let index = value.stage as usize;
            assert!(index < 2);
            assert!((1..=100).contains(&value.percent));
            if seen[index] {
                return Err("duplicate opponent win percent");
            }
            seen[index] = true;
            percentages[index] = value.percent;
        }
        Ok(Self {
            window: window as u16,
            percentages,
        })
    }

    pub(crate) fn from_resolved(
        window: usize,
        weak: u8,
        teacher: u8,
    ) -> Result<Self, &'static str> {
        if !(1..=100).contains(&teacher) {
            return Err("mastery win percent must be in 1..=100");
        }
        Self::new(
            window,
            weak,
            &[OpponentWinPercent {
                stage: MasteryStage::Teacher,
                percent: teacher,
            }],
        )
    }

    pub const fn window(self) -> usize {
        self.window as usize
    }

    pub const fn threshold(self, stage: MasteryStage) -> u8 {
        self.percentages[match stage {
            MasteryStage::Weak => 0,
            MasteryStage::Teacher | MasteryStage::Completed => 1,
        }]
    }

    #[cfg(feature = "builtin")]
    pub(crate) fn canonical_suffix(self) -> String {
        format!(
            " --mastery-window {} --opponent-win-percent weak={} --opponent-win-percent teacher={}",
            self.window, self.percentages[0], self.percentages[1]
        )
    }
}

/// Completed training task result; infrastructure failures have no variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingGameOutcome {
    Win,
    Loss,
    Draw,
    TimeCap,
}

/// Ordered rolling result window of the current opponent, never an all-time average.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MasteryProgress {
    stage: MasteryStage,
    games: u64,
    recent: VecDeque<bool>,
}

impl MasteryProgress {
    pub const fn stage(&self) -> MasteryStage {
        self.stage
    }

    pub fn recent(&self) -> &VecDeque<bool> {
        &self.recent
    }

    pub const fn games(&self) -> u64 {
        self.games
    }

    pub fn wins(&self) -> usize {
        assert!(self.recent.len() <= MASTERY_MAX_WINDOW);
        self.recent.iter().filter(|win| **win).count()
    }

    pub fn completed(&self) -> bool {
        self.stage == MasteryStage::Completed
    }

    pub(crate) fn restore(
        stage: MasteryStage,
        games: u64,
        recent: Vec<bool>,
        config: MasteryConfig,
    ) -> Result<Self, &'static str> {
        if recent.len() > config.window() {
            return Err("mastery window exceeds configured capacity");
        }
        let result = Self {
            stage,
            games,
            recent: recent.into(),
        };
        result.validate(config)?;
        Ok(result)
    }

    pub(crate) fn validate(&self, config: MasteryConfig) -> Result<(), &'static str> {
        if self.games > crate::MAX_TRAINING_COUNTER {
            return Err("mastery game counter exceeds bound");
        }
        if self.recent.len() > config.window() {
            return Err("mastery window exceeds configured capacity");
        }
        if self.completed() && !self.qualifies(config) {
            return Err("completed mastery requires a qualifying Teacher window");
        }
        if self.recent.len() as u64 != self.games.min(config.window() as u64) {
            return Err("mastery game counter/window mismatch");
        }
        if !self.completed() && self.qualifies(config) {
            return Err("active mastery window already qualifies for advancement");
        }
        Ok(())
    }

    fn qualifies(&self, config: MasteryConfig) -> bool {
        assert!(self.recent.len() <= config.window());
        assert!(config.window() <= MASTERY_MAX_WINDOW);
        self.recent.len() == config.window()
            && 100 * self.wins() >= usize::from(config.threshold(self.stage)) * config.window()
    }

    /// Records an already ordered, complete batch and advances at most one stage.
    pub fn record_batch(
        &mut self,
        config: MasteryConfig,
        outcomes: &[TrainingGameOutcome],
    ) -> Result<bool, &'static str> {
        self.validate(config)?;
        if self.completed() {
            return Err("mastery already completed");
        }
        if !(1..=6).contains(&outcomes.len()) {
            return Err("mastery batch must contain 1..=6 completed games");
        }
        let games = self
            .games
            .checked_add(outcomes.len() as u64)
            .filter(|games| *games <= crate::MAX_TRAINING_COUNTER)
            .ok_or("mastery game counter exceeds bound")?;
        for outcome in outcomes {
            if self.recent.len() == config.window() {
                self.recent.pop_front();
            }
            self.recent.push_back(*outcome == TrainingGameOutcome::Win);
        }
        self.games = games;
        let advanced = self.qualifies(config);
        if advanced {
            self.stage = match self.stage {
                MasteryStage::Weak => MasteryStage::Teacher,
                MasteryStage::Teacher => MasteryStage::Completed,
                MasteryStage::Completed => unreachable!("completed stage rejected above"),
            };
            if !self.completed() {
                self.recent.clear();
                self.games = 0;
            }
        }
        self.validate(config)?;
        Ok(advanced)
    }
}
