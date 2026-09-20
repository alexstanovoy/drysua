use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(test)]
#[path = "tests/cli_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;

/// Drysua command line arguments.
#[derive(Parser)]
#[command(version, about = "A Shadow Fiend bot for bota", long_about = None)]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    /// The operation to run. Play is used when absent.
    #[command(subcommand)]
    operation: Option<Operation>,
    #[command(flatten)]
    play: PlayArgs,
}

/// Drysua operations.
#[derive(Subcommand)]
enum Operation {
    /// Passively score copied native participant frames from stdin; never sends orders or ACKs.
    RewardObserver(RewardObserverArgs),
    /// Connect to a server and play one match.
    Play(PlayArgs),
    /// Run a bounded stage-nine PPO actor-to-learner smoke training.
    Train(TrainArgs),
    /// Run resumable PPO training with periodic strict checkpoints.
    TrainFull(TrainFullArgs),
    /// Run the annealed domain-randomization loop with a frozen opponent.
    TrainAnnealed(TrainAnnealedArgs),
}

#[derive(Args)]
struct RewardObserverArgs {
    /// New final JSON file; an adjacent JSONL file contains bounded interval deltas.
    #[arg(long)]
    output: std::path::PathBuf,
    /// Complete ticks between diagnostic intervals.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u32).range(30..=27_900))]
    interval_ticks: u32,
}

/// Options for one server match.
#[derive(Args)]
struct PlayArgs {
    /// Override the repository-selected default with Neural, Hybrid, or Teacher.
    #[arg(long, value_enum)]
    policy: Option<PlayPolicy>,
    /// Server socket address.
    #[arg(long, default_value = "127.0.0.1:4455")]
    addr: String,
    /// Name shown in the lobby.
    #[arg(long, default_value = "drysua")]
    name: String,
    /// Leave after receiving this snapshot tick.
    #[arg(long, value_name = "TICKS")]
    limit: Option<u32>,
    /// Explicit experiment weights; without --policy this selects Hybrid. Defaults need no path.
    #[arg(long, required_if_eq("policy", "neural"))]
    weights_directory: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PlayPolicy {
    Hybrid,
    Neural,
    Teacher,
}

/// Options for a short builtin PPO verification run.
#[derive(Args)]
struct TrainArgs {
    /// PPO updates, bounded to ten for this smoke command.
    #[arg(long, default_value_t = 1)]
    updates: u32,
    /// Independent CPU arenas, bounded to sixteen for desktop headroom.
    #[arg(long, default_value_t = 2)]
    environments: usize,
    /// Decisions collected from each arena per update.
    #[arg(long, default_value_t = 8)]
    rollout: usize,
    /// PPO passes over one rollout.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = 16)]
    minibatch: usize,
    /// Deterministic training seed.
    #[arg(long, default_value_t = 9_001)]
    seed: u64,
    /// Simulator map id, restricted to Map2 (mid-only Dota).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u16).range(2..=2))]
    map: u16,
    /// Learner tensor backend; actors and simulation remain on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA or Metal device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

/// Options for bounded resumable PPO training.
#[derive(Args)]
struct TrainFullArgs {
    /// Separate episode time cost; Map2 comprehensive reward requires zero.
    #[arg(long, default_value_t = 0.0)]
    episode_time_cost: f32,
    /// Legacy terminal-only reward; rejected because Map2 requires comprehensive reward.
    #[arg(long)]
    terminal_only: bool,
    /// Collect paired Map2 episodes; retain one of eight actions. Use =false for windows.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    complete_episodes: bool,
    /// Fixed collection groups (1, 2 or 4), each with its own decision barrier.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=4))]
    pipeline_groups: u8,
    /// Full-episode opponents; nondefault schedules are versioned and checked on resume.
    #[arg(long, value_enum, default_value_t = crate::TrainingOpponentSchedule::Teacher)]
    opponent_schedule: crate::TrainingOpponentSchedule,
    /// Recent completed training games per mastery stage (default 50, at most 1024).
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=1024))]
    mastery_window: Option<u16>,
    /// Default win percentage for all mastery opponents (default 80).
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100))]
    mastery_win_percent: Option<u8>,
    /// Per-opponent mastery override, e.g. weak=90 or teacher=80; no duplicates.
    #[arg(long)]
    opponent_win_percent: Vec<crate::OpponentWinPercent>,
    /// Adam learning rate; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().learning_rate)]
    learning_rate: f32,
    /// Discount per simulator tick; Map2 comprehensive reward requires exactly one.
    #[arg(long, default_value_t = crate::MAP2_REWARD_GAMMA_TICK)]
    gamma_per_tick: f32,
    /// Generalized advantage trace decay in [0, 1]; one uses full discounted Monte Carlo returns.
    #[arg(long, default_value_t = crate::PpoConfig::default().gae_lambda)]
    gae_lambda: f32,
    /// Entropy bonus coefficient; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().entropy_coefficient)]
    entropy_coefficient: f32,
    /// Total PPO update target, including updates restored from a checkpoint.
    #[arg(long)]
    updates: u64,
    /// Independent CPU arenas: even counts up to 26 for complete episodes; at most 16 for windows.
    #[arg(long, default_value_t = 4)]
    environments: usize,
    /// Window decisions or per-episode retained capacity.
    #[arg(long, default_value_t = 2_048,
        help = format!("Window decisions or per-episode retained capacity (at least {}), at most {}", crate::MAP2_RETAINED_DECISIONS, crate::PPO_MAX_ROLLOUT_DECISIONS))]
    rollout: usize,
    /// PPO passes over one rollout.
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = 2_048)]
    minibatch: usize,
    /// Monotonic wall-clock seconds between durable checkpoints.
    #[arg(long, default_value_t = 300)]
    checkpoint_seconds: u64,
    /// Existing empty directory for a fresh run, or checkpoint directory when resuming.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Runtime weights directory loaded before the first update of a fresh run.
    #[arg(long, conflicts_with = "resume")]
    initial_weights: Option<std::path::PathBuf>,
    /// Resume strict model, optimizer, counters, pipeline generation, and actor RNG state.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Rebind an otherwise identical checkpoint to this build's Git provenance once.
    #[arg(long, default_value_t = false, requires = "resume")]
    migrate_provenance: bool,
    /// Deterministic training seed.
    #[arg(long, default_value_t = 9_001)]
    seed: u64,
    /// Simulator map id, restricted to Map2 (mid-only Dota).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u16).range(2..=2))]
    map: u16,
    /// Learner tensor backend; actors and simulation remain on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA or Metal device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LearnerDevice {
    Cpu,
    Cuda,
    Metal,
}

/// Which frozen opponent the annealed run plays against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AnnealedOpponentArg {
    Teacher,
    Weights,
}

/// Options for the annealed domain-randomization loop.
#[derive(Args)]
struct TrainAnnealedArgs {
    /// Total updates; the annealed loop runs one optimizer step per update.
    #[arg(long)]
    updates: u64,
    /// Games per update; always even so sides split exactly.
    #[arg(long, default_value_t = 8)]
    games: usize,
    /// Worlds advanced in parallel per batch; must divide games and generation
    /// games. Defaults to the largest divisor of their gcd within the available
    /// cores and the resolved value is printed, recorded in the run scope, and
    /// compared on resume; pass it explicitly when a run must resume on a host
    /// with a different core count.
    #[arg(long)]
    parallel: Option<usize>,
    /// Games per environment generation, on the global game counter.
    #[arg(long)]
    generation_games: u64,
    /// Final updates with no modifiers; defaults to one fifth of the update budget.
    #[arg(long)]
    zero_updates: Option<u64>,
    /// Frozen opponent: teacher or a strict runtime weights directory.
    #[arg(long, value_enum, default_value_t = AnnealedOpponentArg::Teacher)]
    opponent: AnnealedOpponentArg,
    /// Strict runtime weights directory for a frozen weights opponent.
    #[arg(long)]
    opponent_weights: Option<std::path::PathBuf>,
    /// Adam learning rate; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().learning_rate)]
    learning_rate: f32,
    /// PPO passes over one update's rollout.
    #[arg(long, default_value_t = crate::PpoConfig::default().epochs)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = crate::PpoConfig::default().minibatch)]
    minibatch: usize,
    /// Generalized advantage trace decay in [0, 1].
    #[arg(long, default_value_t = crate::PpoConfig::default().gae_lambda)]
    gae_lambda: f32,
    /// Entropy bonus coefficient; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().entropy_coefficient)]
    entropy_coefficient: f32,
    /// Monotonic wall-clock seconds between durable checkpoints.
    #[arg(long, default_value_t = 300)]
    checkpoint_seconds: u64,
    /// Existing empty directory for a fresh run, or checkpoint directory when resuming.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Runtime weights directory loaded before the first update of a fresh run.
    #[arg(long, conflicts_with = "resume")]
    initial_weights: Option<std::path::PathBuf>,
    /// Resume strict model, optimizer, counters, RNG state and generation snapshots.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Deterministic run seed.
    #[arg(long, default_value_t = 9_001)]
    seed: u64,
    /// Learner tensor backend; actors and simulation remain on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA or Metal device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

/// Parses command line arguments and plays one match.
pub fn run_from_env() -> std::io::Result<()> {
    run(Cli::parse())
}

fn run(arguments: Cli) -> std::io::Result<()> {
    let play = match arguments.operation {
        Some(Operation::RewardObserver(observer)) => {
            return crate::reward_observer::run(&observer.output, observer.interval_ticks);
        }
        Some(Operation::Play(play)) => play,
        Some(Operation::Train(train)) => return run_train(train),
        Some(Operation::TrainFull(train)) => return run_train_full(train),
        Some(Operation::TrainAnnealed(train)) => return run_train_annealed(train),
        None => arguments.play,
    };
    let (policy, weights_directory) = resolve_play_deployment(&play)?;
    if play.policy.is_none() && play.weights_directory.is_none() {
        eprintln!("deployment: repository-selected {policy:?}");
    }
    let outcome = match policy {
        PlayPolicy::Hybrid => crate::play(
            &play.addr,
            &play.name,
            play.limit,
            weights_directory
                .as_deref()
                .unwrap_or(std::path::Path::new(".")),
        )?,
        PlayPolicy::Neural => {
            let directory = weights_directory.as_deref().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "neural requires --weights-directory",
                )
            })?;
            crate::seat::play_neural(&play.addr, &play.name, play.limit, directory)?
        }
        PlayPolicy::Teacher => crate::play_teacher(&play.addr, &play.name, play.limit)?,
    };
    println!(
        "played {} ticks as {:?}; winner {:?}; {} decisions, {} orders, {} rejected orders",
        outcome.ticks,
        outcome.team,
        outcome.winner,
        outcome.decisions,
        outcome.orders,
        outcome.rejections
    );
    Ok(())
}

fn resolve_play_deployment(
    play: &PlayArgs,
) -> std::io::Result<(PlayPolicy, Option<std::path::PathBuf>)> {
    if let Some(policy) = play.policy {
        if policy == PlayPolicy::Teacher && play.weights_directory.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "teacher forbids --weights-directory",
            ));
        }
        return Ok((policy, play.weights_directory.clone()));
    }
    if play.weights_directory.is_some() {
        return Ok((PlayPolicy::Hybrid, play.weights_directory.clone()));
    }
    crate::default_deployment::DEFAULT_DEPLOYMENT.resolve()
}

#[cfg(feature = "builtin")]
fn run_train(arguments: TrainArgs) -> std::io::Result<()> {
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    let report = crate::run_ppo_smoke_on(
        crate::PpoSmokeConfig {
            updates: arguments.updates,
            environments: arguments.environments,
            rollout_decisions: arguments.rollout,
            epochs: arguments.epochs,
            minibatch: arguments.minibatch,
            seed: arguments.seed,
            map: bota_proto::MapId(arguments.map),
        },
        device,
    )
    .map_err(std::io::Error::other)?;
    println!(
        "PPO smoke: {} updates, {} transitions, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}, terminal wins {}, terminal losses {}, terminal draws {}, {} rejected orders, {} arena ticks",
        report.updates,
        report.transitions,
        report.optimizer_step,
        report.final_policy_loss,
        report.final_value_loss,
        report.final_entropy,
        report.final_kl,
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.rejected_orders,
        report.elapsed_ticks,
    );
    report
        .map2_reward
        .log("ppo_smoke", u64::from(report.updates));
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_train_full(arguments: TrainFullArgs) -> std::io::Result<()> {
    arguments.validate_map2_reward()?;
    let settings = arguments.training_settings(
        embedded_commit("DRYSUA_GIT_COMMIT", option_env!("DRYSUA_GIT_COMMIT"))?,
        embedded_commit("BOTA_GIT_COMMIT", option_env!("BOTA_GIT_COMMIT"))?,
    )?;
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    validate_checkpoint_directory(&arguments.checkpoint_directory, arguments.resume)?;
    let report = crate::run_training_job_on_with_initial_weights(
        settings,
        device,
        &arguments.checkpoint_directory,
        arguments.resume,
        arguments.initial_weights.as_deref(),
        report_training_checkpoint,
    )
    .map_err(std::io::Error::other)?;
    println!(
        "training complete: starting fingerprint {:016x}, {} updates, {} samples, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}, session terminal wins {}, session terminal losses {}, session terminal draws {}, session rejected {}, session ticks {}, session episode timeouts {}",
        report.starting_policy_fingerprint,
        report.completed_updates,
        report.rollout_samples,
        report.optimizer_step,
        report.final_policy_loss,
        report.final_value_loss,
        report.final_entropy,
        report.final_kl,
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.rejected_orders,
        report.elapsed_ticks,
        report.episode_timeouts,
    );
    report
        .map2_reward
        .log("invocation", report.completed_updates);
    if report.mastery_completed {
        println!(
            "training mastery complete: rolling training windows qualified through Teacher; not evaluation qualification"
        );
    }
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_train_annealed(arguments: TrainAnnealedArgs) -> std::io::Result<()> {
    let settings = arguments.annealed_settings(
        embedded_commit("DRYSUA_GIT_COMMIT", option_env!("DRYSUA_GIT_COMMIT"))?,
        embedded_commit("BOTA_GIT_COMMIT", option_env!("BOTA_GIT_COMMIT"))?,
    )?;
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    validate_checkpoint_directory(&arguments.checkpoint_directory, arguments.resume)?;
    eprintln!(
        "annealed: updates={} games={} parallel={} generation_games={} zero_updates={} seed={} opponent={:?}",
        settings.updates,
        settings.games_per_update,
        settings.parallel_worlds,
        settings.games_per_generation,
        settings.zero_updates,
        settings.seed,
        settings.opponent,
    );
    let report = crate::run_annealed_job_on_with_initial_weights(
        settings,
        device,
        &arguments.checkpoint_directory,
        arguments.resume,
        arguments.initial_weights.as_deref(),
        report_training_checkpoint,
    )
    .map_err(std::io::Error::other)?;
    println!(
        "annealed training complete: starting fingerprint {:016x}, {} updates, {} samples, {} games, {} generations, optimizer step {}, terminal wins {}, terminal losses {}, terminal draws {}, ticks {}, episode timeouts {}",
        report.starting_policy_fingerprint,
        report.completed_updates,
        report.rollout_samples,
        report.games,
        report.generations,
        report.optimizer_step,
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.elapsed_ticks,
        report.episode_timeouts,
    );
    report
        .map2_reward
        .log("invocation", report.completed_updates);
    Ok(())
}

#[cfg(feature = "builtin")]
fn report_training_checkpoint(checkpoint: crate::TrainingCheckpointReport) {
    println!(
        "checkpoint: update {}, samples {}, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}, KL stop {}, session terminal wins {}, session terminal losses {}, session terminal draws {}, session rejected {}, session ticks {}, session episode timeouts {}",
        checkpoint.completed_updates,
        checkpoint.rollout_samples,
        checkpoint.optimizer_step,
        checkpoint.policy_loss,
        checkpoint.value_loss,
        checkpoint.entropy,
        checkpoint.approximate_kl,
        checkpoint.stopped_for_kl,
        checkpoint.terminal_wins,
        checkpoint.terminal_losses,
        checkpoint.terminal_draws,
        checkpoint.rejected_orders,
        checkpoint.elapsed_ticks,
        checkpoint.episode_timeouts,
    );
    checkpoint
        .map2_reward
        .log("checkpoint", checkpoint.completed_updates);
    if let Some(warning) = checkpoint.cleanup_warning {
        eprintln!("checkpoint cleanup warning: {warning}");
    }
}

#[cfg(feature = "builtin")]
impl TrainFullArgs {
    fn mastery_config(&self) -> std::io::Result<Option<crate::MasteryConfig>> {
        if self.opponent_schedule != crate::TrainingOpponentSchedule::MasteryV1 {
            if self.mastery_window.is_some()
                || self.mastery_win_percent.is_some()
                || !self.opponent_win_percent.is_empty()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "mastery options require --opponent-schedule mastery-v1",
                ));
            }
            return Ok(None);
        }
        crate::MasteryConfig::new(
            self.mastery_window
                .map_or(crate::MASTERY_DEFAULT_WINDOW, usize::from),
            self.mastery_win_percent
                .unwrap_or(crate::MASTERY_DEFAULT_WIN_PERCENT),
            &self.opponent_win_percent,
        )
        .map(Some)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
    }

    fn training_settings(
        &self,
        git_commit: String,
        simulator_commit: String,
    ) -> std::io::Result<crate::TrainingJobConfig> {
        let ppo = crate::PpoConfig {
            decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
            environments: self.environments,
            rollout_decisions: self.rollout,
            epochs: self.epochs,
            minibatch: self.minibatch,
            learning_rate: self.learning_rate,
            gamma_tick: self.gamma_per_tick,
            gae_lambda: self.gae_lambda,
            entropy_coefficient: self.entropy_coefficient,
            ..crate::PpoConfig::default()
        }
        .validate()
        .map_err(std::io::Error::other)?;
        self.validate_map2_reward()?;
        let settings = crate::TrainingJobConfig {
            mastery_config: self.mastery_config()?,
            opponent_schedule: self.opponent_schedule,
            episode_time_cost: self.episode_time_cost,
            terminal_only: self.terminal_only,
            complete_episodes: self.complete_episodes,
            pipeline_groups: usize::from(self.pipeline_groups),
            updates: self.updates,
            ppo,
            checkpoint_cadence: crate::TrainingCheckpointCadence::WallTime(
                std::time::Duration::from_secs(self.checkpoint_seconds),
            ),
            resume_provenance: if self.migrate_provenance {
                crate::ResumeProvenance::MigrateGitCommit
            } else {
                crate::ResumeProvenance::Strict
            },
            seed: self.seed,
            map: bota_proto::MapId(self.map),
            git_commit,
            simulator_commit,
        };
        crate::ppo_arena::episode::validate(&settings).map_err(std::io::Error::other)?;
        Ok(settings)
    }

    fn validate_map2_reward(&self) -> std::io::Result<()> {
        if self.terminal_only {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Map2 comprehensive reward forbids --terminal-only",
            ));
        }
        if self.episode_time_cost != 0.0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Map2 comprehensive reward requires --episode-time-cost 0",
            ));
        }
        if self.gamma_per_tick != crate::MAP2_REWARD_GAMMA_TICK {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Map2 comprehensive reward requires --gamma-per-tick 1",
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "builtin"))]
pub(crate) fn training_settings_for_test(
    overrides: &[&str],
) -> std::io::Result<crate::TrainingJobConfig> {
    let mut arguments = vec![
        "drysua",
        "train-full",
        "--updates",
        "1",
        "--checkpoint-directory",
        ".",
    ];
    arguments.extend_from_slice(overrides);
    let cli = Cli::try_parse_from(arguments).map_err(std::io::Error::other)?;
    let Some(Operation::TrainFull(train)) = cli.operation else {
        unreachable!("train-full arguments");
    };
    train.training_settings(
        "test-drysua-commit".to_owned(),
        "test-bota-commit".to_owned(),
    )
}

#[cfg(feature = "builtin")]
impl TrainAnnealedArgs {
    fn annealed_settings(
        &self,
        git_commit: String,
        simulator_commit: String,
    ) -> std::io::Result<crate::AnnealedJobConfig> {
        if self.updates == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "annealed updates must be positive",
            ));
        }
        if !self.games.is_multiple_of(2) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "annealed games per update must be even so sides split exactly",
            ));
        }
        if self.generation_games == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "annealed generation games must be positive",
            ));
        }
        let opponent = match (self.opponent, &self.opponent_weights) {
            (AnnealedOpponentArg::Teacher, None) => crate::AnnealedOpponent::Teacher,
            (AnnealedOpponentArg::Weights, Some(directory)) => {
                crate::AnnealedOpponent::Weights(directory.clone())
            }
            (AnnealedOpponentArg::Weights, None) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "weights opponent requires --opponent-weights",
                ));
            }
            (AnnealedOpponentArg::Teacher, Some(_)) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "teacher opponent forbids --opponent-weights",
                ));
            }
        };
        let parallel = match self.parallel {
            Some(parallel) => parallel,
            None => default_parallel_worlds(self.games, self.generation_games),
        };
        let zero_updates = self
            .zero_updates
            .unwrap_or_else(|| self.updates.div_ceil(5));
        let ppo = crate::PpoConfig {
            decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
            environments: self.games,
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            epochs: self.epochs,
            minibatch: self.minibatch,
            learning_rate: self.learning_rate,
            gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
            gae_lambda: self.gae_lambda,
            entropy_coefficient: self.entropy_coefficient,
            ..crate::PpoConfig::default()
        };
        Ok(crate::AnnealedJobConfig {
            updates: self.updates,
            games_per_update: self.games,
            parallel_worlds: parallel,
            games_per_generation: self.generation_games,
            zero_updates,
            seed: self.seed,
            opponent,
            ppo,
            checkpoint_cadence: crate::TrainingCheckpointCadence::WallTime(
                std::time::Duration::from_secs(self.checkpoint_seconds),
            ),
            git_commit,
            simulator_commit,
        })
    }
}

/// Largest divisor of `gcd(games, generation_games)` within the core budget.
#[cfg(feature = "builtin")]
fn default_parallel_worlds(games: usize, generation_games: u64) -> usize {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    default_parallel_worlds_for(games, generation_games, cores)
}

/// Largest divisor of `gcd(games, generation_games)` not past `cores`.
#[cfg(feature = "builtin")]
pub(crate) fn default_parallel_worlds_for(
    games: usize,
    generation_games: u64,
    cores: usize,
) -> usize {
    let gcd = greatest_common_divisor(games as u64, generation_games).max(1);
    let cores = cores.clamp(1, crate::MAX_TRAINING_ENVIRONMENTS);
    (1..=cores)
        .rev()
        .find(|candidate| gcd.is_multiple_of(*candidate as u64))
        .unwrap_or(1)
}

#[cfg(feature = "builtin")]
fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

#[cfg(all(test, feature = "builtin"))]
pub(crate) fn annealed_settings_for_test(
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut arguments = vec!["drysua", "train-annealed", "--checkpoint-directory", "."];
    arguments.extend_from_slice(overrides);
    let cli = Cli::try_parse_from(arguments).map_err(std::io::Error::other)?;
    let Some(Operation::TrainAnnealed(train)) = cli.operation else {
        unreachable!("train-annealed arguments");
    };
    train.annealed_settings(
        "test-drysua-commit".to_owned(),
        "test-bota-commit".to_owned(),
    )
}

#[cfg(feature = "builtin")]
fn validate_checkpoint_directory(directory: &std::path::Path, resume: bool) -> std::io::Result<()> {
    if !directory.is_dir() {
        return Err(std::io::Error::other(
            "checkpoint directory must already exist and be a directory",
        ));
    }
    if !resume && directory.read_dir()?.next().is_some() {
        return Err(std::io::Error::other(
            "fresh checkpoint directory must be empty; use --resume for an existing run",
        ));
    }
    Ok(())
}

#[cfg(feature = "builtin")]
fn embedded_commit(name: &str, value: Option<&str>) -> std::io::Result<String> {
    value
        .filter(|commit| !commit.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "production training requires {name} to be set when compiling"
            ))
        })
}

#[cfg(feature = "builtin")]
impl LearnerDevice {
    fn policy_device(self, ordinal: usize) -> std::io::Result<crate::PolicyDevice> {
        match self {
            Self::Cpu => Ok(crate::PolicyDevice::Cpu),
            Self::Cuda => cuda_policy_device(ordinal),
            Self::Metal => metal_policy_device(ordinal),
        }
    }
}

#[cfg(all(
    feature = "builtin",
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_policy_device(ordinal: usize) -> std::io::Result<crate::PolicyDevice> {
    Ok(crate::PolicyDevice::Cuda { ordinal })
}

#[cfg(all(
    feature = "builtin",
    not(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))
))]
fn cuda_policy_device(_: usize) -> std::io::Result<crate::PolicyDevice> {
    Err(std::io::Error::other(
        "CUDA learner requires cargo feature `cuda` on Linux or Windows",
    ))
}

#[cfg(all(feature = "builtin", feature = "metal", target_os = "macos"))]
fn metal_policy_device(ordinal: usize) -> std::io::Result<crate::PolicyDevice> {
    Ok(crate::PolicyDevice::Metal { ordinal })
}

#[cfg(all(feature = "builtin", not(all(feature = "metal", target_os = "macos"))))]
fn metal_policy_device(_: usize) -> std::io::Result<crate::PolicyDevice> {
    Err(std::io::Error::other(
        "Metal learner requires cargo feature `metal` on macOS",
    ))
}

#[cfg(not(feature = "builtin"))]
fn run_train(_: TrainArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "PPO train requires cargo feature `builtin`",
    ))
}

#[cfg(not(feature = "builtin"))]
fn run_train_full(_: TrainFullArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "PPO train requires cargo feature `builtin`",
    ))
}

#[cfg(not(feature = "builtin"))]
fn run_train_annealed(_: TrainAnnealedArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "annealed training requires cargo feature `builtin`",
    ))
}
