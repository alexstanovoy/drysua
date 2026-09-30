use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(test)]
#[path = "tests/cli_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;
#[cfg(test)]
#[path = "tests/checkpoint_inspection_cli.rs"]
mod checkpoint_inspection_tests;

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
#[allow(
    clippy::large_enum_variant,
    reason = "parsed once at startup; boxing would only complicate clap derives"
)]
#[derive(Subcommand)]
enum Operation {
    /// Inspect a validated checkpoint or this build's contract as bounded read-only JSON.
    CheckpointInspect(CheckpointInspectArgs),
    /// Passively score copied native participant frames from stdin; never sends orders or ACKs.
    RewardObserver(RewardObserverArgs),
    /// Connect to a server and play one match.
    Play(PlayArgs),
    /// Run the annealed domain-randomization loop with a frozen opponent.
    TrainAnnealed(TrainAnnealedArgs),
}

#[derive(Args)]
struct CheckpointInspectArgs {
    /// Existing checkpoint directory; never creates, repairs, locks, or changes files.
    #[arg(
        long,
        required_unless_present = "contract",
        conflicts_with = "contract"
    )]
    checkpoint_directory: Option<std::path::PathBuf>,
    /// Describe this build's model and checkpoint identities without reading a checkpoint.
    #[arg(long, conflicts_with = "checkpoint_directory")]
    contract: bool,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LearnerDevice {
    Cpu,
    Cuda,
}

/// Which frozen opponent the annealed run plays against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AnnealedOpponentArg {
    Teacher,
    Weights,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OpponentInferenceArg {
    Batched,
    Scalar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EnvironmentScheduleArg {
    Adaptive,
    Fixed,
}

/// Options for the annealed domain-randomization loop.
#[derive(Args)]
struct TrainAnnealedArgs {
    #[command(flatten)]
    optimizer: OptimizerArgs,
    #[command(flatten)]
    checkpoint: CheckpointArgs,
    /// Local gradient-fold worker ceiling (1 is the historical serial path).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=32))]
    host_math_workers: u8,
    /// PPO tensor microbatch (64/128/256); larger modes change reductions and checkpoint scope.
    #[arg(long, default_value_t = 256, value_parser = crate::training_execution::parse_training_microbatch)]
    training_microbatch: usize,
    /// Spread all rollout rows across nearly equal minibatches; recorded in checkpoint scope.
    #[arg(long)]
    balanced_minibatches: bool,
    /// Reuse next-actor bootstrap values; may change PPO numerics, recorded in checkpoint scope.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    reuse_actor_values: bool,
    /// Actor groups per wave (1/2/4); multiple groups require Teacher or batched weights, at most 64 live worlds, and whole waves per update.
    #[arg(long, default_value_t = 2, value_parser = crate::training_execution::parse_actor_pipeline_groups)]
    actor_pipeline_groups: u8,
    /// Total PPO updates; each may perform multiple Adam minibatch steps.
    #[arg(long)]
    updates: u64,
    /// Additional committed updates this invocation; leaves the total target and annealing unchanged.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<std::num::NonZeroU64>::new().range(1..=crate::MAX_TRAINING_COUNTER))]
    invocation_updates: Option<std::num::NonZeroU64>,
    /// Games per update, even from 2 to 80; above 40 selects the wide annealed budget.
    #[arg(long, default_value_t = 40)]
    games: usize,
    /// Worlds per actor group (1..=64); defaults to 20 independently of CPU count.
    /// Must divide games and generation games; incompatible overrides are rejected.
    #[arg(long, default_value_t = 20)]
    parallel: usize,
    /// Games per generation; adaptive requires a positive whole multiple of --games.
    #[arg(long)]
    generation_games: u64,
    /// Environment transitions; legacy checkpoint resume requires explicit fixed.
    #[arg(long, value_enum, default_value_t = EnvironmentScheduleArg::Adaptive)]
    environment_schedule: EnvironmentScheduleArg,
    /// Consecutive successful updates (adaptive only; default 2).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=crate::MAX_TRAINING_COUNTER))]
    environment_success_updates: Option<u64>,
    /// Inclusive success win-rate threshold in [0, 1] (adaptive only; default 0.8).
    #[arg(long)]
    environment_success_rate: Option<crate::EnvironmentDecimal>,
    /// Consecutive poor updates before extension awards (adaptive only; default 1).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=crate::MAX_TRAINING_COUNTER))]
    environment_poor_updates: Option<u64>,
    /// Inclusive poor win-rate threshold in [0, 1] (adaptive only; default 0.2).
    #[arg(long)]
    environment_poor_rate: Option<crate::EnvironmentDecimal>,
    /// Exact extra-update credit per poor award; may exceed 1 (adaptive only; default 0.75).
    #[arg(long)]
    environment_extension: Option<crate::EnvironmentDecimal>,
    /// Final updates with no modifiers; defaults to one fifth of the update budget.
    #[arg(long)]
    zero_updates: Option<u64>,
    /// Frozen opponent: teacher or a strict runtime weights directory.
    #[arg(long, value_enum, default_value_t = AnnealedOpponentArg::Teacher)]
    opponent: AnnealedOpponentArg,
    /// Strict runtime weights directory for a frozen weights opponent.
    #[arg(long)]
    opponent_weights: Option<std::path::PathBuf>,
    /// Weights-opponent sampling mode; batched changes numerics and checkpoint scope. Ignored for Teacher.
    #[arg(long, value_enum, default_value_t = OpponentInferenceArg::Batched)]
    opponent_inference: OpponentInferenceArg,
    /// Environment scale at the first update in exact decimals; 1 is full variance, at most 10.
    #[arg(long)]
    environment_scale_start: Option<crate::EnvironmentDecimal>,
    /// Environment scale at the clean tail boundary; 0 is no modifiers, at most 10.
    #[arg(long)]
    environment_scale_end: Option<crate::EnvironmentDecimal>,
    /// Run seed; a fresh run without it draws a random seed, and resume adopts the recorded one.
    #[arg(long)]
    seed: Option<u64>,
    /// Learner tensor backend; actors and simulation remain on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

/// Optimizer controls for resumable training.
#[derive(Args)]
struct OptimizerArgs {
    /// Adam learning rate; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().learning_rate)]
    learning_rate: f32,
    /// PPO passes over one rollout.
    #[arg(long, default_value_t = crate::PpoConfig::default().epochs)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = crate::PpoConfig::default().minibatch)]
    minibatch: usize,
    /// Generalized advantage trace decay in [0, 1]; one uses full discounted Monte Carlo returns.
    #[arg(long, default_value_t = crate::PpoConfig::default().gae_lambda)]
    gae_lambda: f32,
    /// Entropy bonus coefficient; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().entropy_coefficient)]
    entropy_coefficient: f32,
}

/// Persistence controls; a run cannot initialize and resume together.
#[derive(Args)]
struct CheckpointArgs {
    /// Monotonic wall-clock seconds between durable checkpoints.
    #[arg(long, default_value_t = 300)]
    checkpoint_seconds: u64,
    /// Existing empty directory for a fresh run, or checkpoint directory when resuming.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Runtime weights directory loaded before the first update of a fresh run.
    #[arg(long, conflicts_with = "resume")]
    initial_weights: Option<std::path::PathBuf>,
    /// Resume strict model, optimizer, counters, RNG state and collection progress.
    #[arg(long, default_value_t = false)]
    resume: bool,
}

/// Parses command line arguments and plays one match.
pub fn run_from_env() -> std::io::Result<()> {
    run(Cli::parse())
}

fn run(arguments: Cli) -> std::io::Result<()> {
    let play = match arguments.operation {
        Some(Operation::CheckpointInspect(inspect)) => return run_checkpoint_inspect(inspect),
        Some(Operation::RewardObserver(observer)) => {
            return crate::reward_observer::run(&observer.output, observer.interval_ticks);
        }
        Some(Operation::Play(play)) => play,
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

fn run_checkpoint_inspect(arguments: CheckpointInspectArgs) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = if arguments.contract {
        crate::checkpoint_inspection_contract()
    } else {
        let directory = arguments.checkpoint_directory.as_deref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "checkpoint-inspect requires --checkpoint-directory or --contract",
            )
        })?;
        crate::checkpoint_inspect(directory)
    }
    .map_err(std::io::Error::other)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&bytes)?;
    stdout.flush()
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
fn run_train_annealed(arguments: TrainAnnealedArgs) -> std::io::Result<()> {
    let settings = arguments.annealed_settings(
        embedded_commit("DRYSUA_GIT_COMMIT", option_env!("DRYSUA_GIT_COMMIT"))?,
        embedded_commit("BOTA_GIT_COMMIT", option_env!("BOTA_GIT_COMMIT"))?,
    )?;
    run_train_annealed_with_settings(arguments, settings)
}

#[cfg(feature = "builtin")]
fn run_train_annealed_with_settings(
    arguments: TrainAnnealedArgs,
    settings: crate::AnnealedJobConfig,
) -> std::io::Result<()> {
    crate::ppo_arena::validate_annealed(&settings, Default::default())
        .map_err(std::io::Error::other)?;
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    validate_checkpoint_directory(
        &arguments.checkpoint.checkpoint_directory,
        arguments.checkpoint.resume,
    )?;
    if arguments.checkpoint.resume {
        crate::ppo_arena::preflight_annealed_resume(
            &settings,
            device,
            &arguments.checkpoint.checkpoint_directory,
        )
        .map_err(std::io::Error::other)?;
    }
    if arguments.seed.is_none() {
        eprintln!(
            "annealed: seed {} resolved from {}; recorded in the run scope",
            settings.seed,
            if arguments.checkpoint.resume {
                "the resumed run scope"
            } else {
                "the operating system random source"
            },
        );
    }
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
        &arguments.checkpoint.checkpoint_directory,
        arguments.checkpoint.resume,
        arguments.checkpoint.initial_weights.as_deref(),
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

#[cfg(any(feature = "builtin", test))]
impl TrainAnnealedArgs {
    fn environment_schedule(&self) -> std::io::Result<crate::EnvironmentSchedule> {
        if self.environment_schedule == EnvironmentScheduleArg::Fixed {
            if self.environment_success_updates.is_some()
                || self.environment_success_rate.is_some()
                || self.environment_poor_updates.is_some()
                || self.environment_poor_rate.is_some()
                || self.environment_extension.is_some()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "environment tuning options require --environment-schedule adaptive",
                ));
            }
            return Ok(crate::EnvironmentSchedule::Fixed);
        }
        let defaults = crate::AdaptiveEnvironmentConfig::default();
        let config = crate::AdaptiveEnvironmentConfig {
            success_updates: self
                .environment_success_updates
                .unwrap_or(defaults.success_updates),
            success_rate: self
                .environment_success_rate
                .unwrap_or(defaults.success_rate),
            poor_updates: self
                .environment_poor_updates
                .unwrap_or(defaults.poor_updates),
            poor_rate: self.environment_poor_rate.unwrap_or(defaults.poor_rate),
            extension: self.environment_extension.unwrap_or(defaults.extension),
        }
        .validate()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
        self.validate_environment_limits()?;
        Ok(crate::EnvironmentSchedule::Adaptive(config))
    }

    fn validate_environment_limits(&self) -> std::io::Result<()> {
        if self.games == 0
            || self.generation_games == 0
            || !self.generation_games.is_multiple_of(self.games as u64)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "adaptive environment generation games must be a positive whole multiple of games per update",
            ));
        }
        crate::AdaptiveEnvironmentLimits {
            base_updates: self.generation_games / self.games as u64,
            total_updates: self.updates,
            zero_updates: self
                .zero_updates
                .unwrap_or_else(|| self.updates.div_ceil(5)),
        }
        .validate()
        .map(|_| ())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
    }
}

#[cfg(feature = "builtin")]
impl TrainAnnealedArgs {
    fn annealed_settings(
        &self,
        git_commit: String,
        simulator_commit: String,
    ) -> std::io::Result<crate::AnnealedJobConfig> {
        let environment_schedule = self.environment_schedule()?;
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
        let opponent = self.frozen_opponent()?;
        let scale = self.environment_scale()?;
        let zero_updates = self
            .zero_updates
            .unwrap_or_else(|| self.updates.div_ceil(5));
        let ppo = self.optimizer.ppo(crate::PpoConfig {
            environments: self.games,
            sample_budget: crate::PpoSampleBudget::for_annealed_games(self.games),
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
            ..crate::PpoConfig::default()
        });
        Ok(crate::AnnealedJobConfig {
            environment_schedule,
            execution: crate::TrainingExecutionOptions {
                actor_pipeline_groups: usize::from(self.actor_pipeline_groups),
                balanced_minibatches: self.balanced_minibatches,
                host_math_workers: usize::from(self.host_math_workers),
                neural_opponent_batching: self.opponent == AnnealedOpponentArg::Weights
                    && self.opponent_inference == OpponentInferenceArg::Batched,
                training_microbatch: self.training_microbatch,
                reuse_actor_values: self.reuse_actor_values,
            },
            updates: self.updates,
            invocation_updates: self.invocation_updates,
            games_per_update: self.games,
            parallel_worlds: self.parallel,
            games_per_generation: self.generation_games,
            zero_updates,
            seed: self.resolved_seed()?,
            scale,
            opponent,
            ppo,
            checkpoint_cadence: self.checkpoint.cadence(),
            git_commit,
            simulator_commit,
        })
    }

    /// The resolved run seed: explicit, adopted from the recorded resume scope,
    /// or freshly random. Adoption is what lets `resume` omit `--seed` without
    /// silently changing every derived stream.
    fn resolved_seed(&self) -> std::io::Result<u64> {
        let Some(seed) = self.seed else {
            if self.checkpoint.resume {
                let run = crate::TrainingArtifact::load_run_scope(
                    &self.checkpoint.checkpoint_directory,
                )
                .map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "train-annealed resume without --seed needs the recorded run scope: {error}"
                        ),
                    )
                })?;
                return Ok(run.run_seed);
            }
            return random_run_seed();
        };
        Ok(seed)
    }

    /// The environment scale ramp in basis points.
    ///
    /// The command line keeps six fractional digits; the loop works in basis
    /// points, so each endpoint truncates to a hundredth of full variance.
    fn environment_scale(&self) -> std::io::Result<crate::randomization::AnnealScale> {
        let start_bp = scale_bp(
            "--environment-scale-start",
            self.environment_scale_start,
            crate::randomization::AnnealScale::FULL.start_bp,
        )?;
        let end_bp = scale_bp(
            "--environment-scale-end",
            self.environment_scale_end,
            crate::randomization::AnnealScale::FULL.end_bp,
        )?;
        crate::randomization::AnnealScale { start_bp, end_bp }
            .validate()
            .map_err(|error| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
            })
    }

    fn frozen_opponent(&self) -> std::io::Result<crate::AnnealedOpponent> {
        match (self.opponent, &self.opponent_weights) {
            (AnnealedOpponentArg::Teacher, None) => Ok(crate::AnnealedOpponent::Teacher),
            (AnnealedOpponentArg::Weights, Some(directory)) => {
                Ok(crate::AnnealedOpponent::Weights(directory.clone()))
            }
            (AnnealedOpponentArg::Weights, None) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "weights opponent requires --opponent-weights",
            )),
            (AnnealedOpponentArg::Teacher, Some(_)) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "teacher opponent forbids --opponent-weights",
            )),
        }
    }
}

/// Fresh runs without an explicit seed draw from the operating system instead
/// of a shared constant, so two campaigns never replay identical trajectories.
#[cfg(feature = "builtin")]
fn random_run_seed() -> std::io::Result<u64> {
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut bytes = [0_u8; 8];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut source| source.read_exact(&mut bytes))
            .map_err(|error| {
                std::io::Error::new(
                    error.kind(),
                    format!("train-annealed random seed needs /dev/urandom: {error}"),
                )
            })?;
        Ok(u64::from_le_bytes(bytes))
    }
    #[cfg(not(unix))]
    {
        // Best effort without a dependency: splitmix64 mixes the wall clock with
        // the process id so two fresh runs on one host do not share a seed.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| {
                std::io::Error::other(format!("train-annealed random seed clock: {error}"))
            })?
            .as_nanos() as u64;
        let mut mixed = nanos ^ (u64::from(std::process::id()) << 32);
        mixed = mixed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Ok(mixed ^ (mixed >> 31))
    }
}

/// One scale endpoint in basis points, bounded to ten times full variance.
#[cfg(feature = "builtin")]
fn scale_bp(
    name: &str,
    value: Option<crate::EnvironmentDecimal>,
    default_bp: i32,
) -> std::io::Result<i32> {
    let Some(value) = value else {
        return Ok(default_bp);
    };
    let units = value.units();
    if units > 10 * crate::EnvironmentDecimal::SCALE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{name} must be within 0 and 10"),
        ));
    }
    Ok((units / 100) as i32)
}

#[cfg(all(test, feature = "builtin"))]
pub(crate) fn annealed_settings_for_test(
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut arguments = vec!["--checkpoint-directory", "."];
    if !requests_seed(overrides) {
        // Unit tests keep the historical explicit seed; only a real run omits it.
        arguments.extend(["--seed", TEST_ANNEALED_SEED]);
    }
    arguments.extend_from_slice(overrides);
    annealed_settings_from_arguments(&arguments)
}

/// Parses fresh train-annealed settings exactly as production does, so a test
/// can observe the random default seed.
#[cfg(all(test, feature = "builtin"))]
pub(crate) fn annealed_settings_for_test_without_seed(
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut arguments = vec!["--checkpoint-directory", "."];
    arguments.extend_from_slice(overrides);
    annealed_settings_from_arguments(&arguments)
}

/// Parses resume settings against one checkpoint directory; the adopted seed
/// comes from the recorded scope unless an override names one.
#[cfg(all(test, feature = "builtin"))]
pub(crate) fn annealed_resume_settings_for_test(
    directory: &std::path::Path,
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let checkpoint = directory.to_str().expect("utf8 test directory");
    let mut arguments = vec!["--checkpoint-directory", checkpoint, "--resume"];
    arguments.extend_from_slice(overrides);
    annealed_settings_from_arguments(&arguments)
}

#[cfg(all(test, feature = "builtin"))]
const TEST_ANNEALED_SEED: &str = "9001";

#[cfg(all(test, feature = "builtin"))]
fn requests_seed(overrides: &[&str]) -> bool {
    overrides
        .iter()
        .any(|argument| *argument == "--seed" || argument.starts_with("--seed="))
}

#[cfg(all(test, feature = "builtin"))]
fn annealed_settings_from_arguments(
    arguments: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut full = vec!["drysua", "train-annealed"];
    full.extend_from_slice(arguments);
    let cli = Cli::try_parse_from(full).map_err(std::io::Error::other)?;
    let Some(Operation::TrainAnnealed(train)) = cli.operation else {
        unreachable!("train-annealed arguments");
    };
    train.annealed_settings(
        "test-drysua-commit".to_owned(),
        "test-bota-commit".to_owned(),
    )
}

#[cfg(feature = "builtin")]
impl OptimizerArgs {
    fn ppo(&self, config: crate::PpoConfig) -> crate::PpoConfig {
        crate::PpoConfig {
            decision_interval_ticks: crate::MAP2_DECISION_INTERVAL_TICKS,
            epochs: self.epochs,
            minibatch: self.minibatch,
            learning_rate: self.learning_rate,
            gae_lambda: self.gae_lambda,
            entropy_coefficient: self.entropy_coefficient,
            ..config
        }
    }
}

#[cfg(feature = "builtin")]
impl CheckpointArgs {
    fn cadence(&self) -> crate::TrainingCheckpointCadence {
        crate::TrainingCheckpointCadence::WallTime(std::time::Duration::from_secs(
            self.checkpoint_seconds,
        ))
    }
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

#[cfg(not(feature = "builtin"))]
fn run_train_annealed(_: TrainAnnealedArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "annealed training requires cargo feature `builtin`",
    ))
}
