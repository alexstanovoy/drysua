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
    /// Evaluate frozen runtime weights on both sides of paired seeds; never trains.
    Eval(EvalArgs),
    /// Play two rule policies against each other on Map2 seeds from both sides.
    Duel(crate::scripted::duel_cli::DuelArgs),
}

/// Options for a frozen pool evaluation.
#[derive(Args)]
struct EvalArgs {
    /// Candidate: a rule policy, `weights:<dir>` or `average:<dir>,<dir>,...`.
    #[arg(long)]
    candidate: String,
    /// Candidate name in the output; defaults to the weights directory or rule name.
    #[arg(long)]
    name: Option<String>,
    /// Opponent pool JSON (`drysua-eval-pool/v1`), see docs/eval_pool.example.json.
    #[arg(long)]
    pool: std::path::PathBuf,
    /// Seed range `<start>:<count>`; every seed is played once per side against every opponent.
    #[arg(long, value_parser = parse_seed_range)]
    seeds: (u64, u64),
    /// Worlds per pipeline group; also the inference batch size.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u16).range(1..=64))]
    parallel: u16,
    /// Pipeline groups (1/2/4): one group infers while the others step.
    #[arg(long, default_value_t = 2, value_parser = crate::training_execution::parse_actor_pipeline_groups)]
    actor_pipeline_groups: u8,
    /// Take the legal argmax instead of sampling from the policy.
    #[arg(long)]
    greedy: bool,
    /// Inference tensor backend; simulation stays on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    /// New JSONL file: a header line, then one line per game.
    #[arg(long)]
    output: std::path::PathBuf,
}

fn parse_seed_range(value: &str) -> Result<(u64, u64), String> {
    let invalid = || format!("seeds must be <start>:<count>, got {value:?}");
    let (start, count) = value.split_once(':').ok_or_else(invalid)?;
    let start = start.parse::<u64>().map_err(|_| invalid())?;
    let count = count.parse::<u64>().map_err(|_| invalid())?;
    if count == 0 || start.checked_add(count).is_none() {
        return Err(format!(
            "seed count must be positive and the range must not overflow, got {value:?}"
        ));
    }
    Ok((start, count))
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
    /// Override the repository-selected default with Neural or a rule policy.
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
    /// Explicit runtime weights; without --policy this selects Neural. Defaults need no path.
    #[arg(long, required_if_eq("policy", "neural"))]
    weights_directory: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PlayPolicy {
    Neural,
    Teacher,
    HarassPush,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LearnerDevice {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OpponentScheduleArg {
    Fixed,
    Pfsp,
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
    /// Rows per device PPO pass (256/512/1024/2048); regroups reductions, recorded in scope.
    #[arg(long, default_value_t = crate::training_execution::DEFAULT_TRAINING_MICROBATCH, value_parser = crate::training_execution::parse_training_microbatch)]
    training_microbatch: usize,
    /// Spread all rollout rows across nearly equal minibatches; recorded in checkpoint scope.
    #[arg(long)]
    balanced_minibatches: bool,
    /// Total PPO updates; each may perform multiple Adam minibatch steps.
    #[arg(long)]
    updates: u64,
    /// Additional committed updates this invocation; leaves the total target and annealing unchanged.
    #[arg(long, value_parser = clap::builder::RangedU64ValueParser::<std::num::NonZeroU64>::new().range(1..=crate::MAX_TRAINING_COUNTER))]
    invocation_updates: Option<std::num::NonZeroU64>,
    /// Retained intervals that make one update due; a multiple of --lanes.
    #[arg(long, default_value_t = crate::PpoConfig::default().samples_per_update)]
    samples_per_update: usize,
    /// Concurrent world slots (1..=256); each always holds a live game. Defaults to
    /// the recorded value on resume, else 16 per available core.
    #[arg(long)]
    slots: Option<usize>,
    /// Inference lanes; each owns a thread, a weight replica and at most 64 slots.
    /// Defaults to the recorded value on resume, else two per simulation group
    /// (more when the slots need them).
    #[arg(long)]
    lanes: Option<usize>,
    /// Simulation worker threads; defaults to the available cores and never changes results.
    #[arg(long)]
    simulation_threads: Option<usize>,
    /// Lane groups with their own simulation workers; defaults to the host's
    /// last-level cache domains (CCDs) and never changes results.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=16))]
    simulation_groups: Option<u8>,
    /// Pin each simulation group's threads to one cache domain (Linux).
    #[arg(long)]
    pin_threads: bool,
    /// Updates per environment generation.
    #[arg(long)]
    generation_updates: u64,
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
    /// Per-game opponent mixture entry, repeatable: `teacher[:weight]`, `harass-push[:weight]`,
    /// `self[:weight]`, `weights:<runtime weights directory>:<weight>` or `league:<weight>`
    /// (each of the latest --league-size learner snapshots); weights are exact decimals.
    #[arg(long = "opponent", default_value = "teacher:1", value_parser = parse_annealed_opponent)]
    opponents: Vec<(crate::AnnealedOpponent, crate::EnvironmentDecimal)>,
    /// How configured opponent weights become each update's mixture: `pfsp` scales
    /// them by (1 - recent win rate)^2, `fixed` keeps them.
    #[arg(long, value_enum, default_value_t = OpponentScheduleArg::Pfsp)]
    opponent_schedule: OpponentScheduleArg,
    /// Learner snapshots a `league` opponent entry plays at once (1..=16).
    #[arg(long, default_value_t = 4)]
    league_size: usize,
    /// Updates between league snapshots; the checkpoint persists the ones in play.
    #[arg(long, default_value_t = 20)]
    league_every: u64,
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
    #[arg(long, default_value_t = crate::PpoConfig::default().gae_lambda_tick)]
    gae_lambda_tick: f32,
    /// Entropy bonus coefficient; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().entropy_coefficient)]
    entropy_coefficient: f32,
    /// Value loss coefficient; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().value_coefficient)]
    value_coefficient: f32,
}

/// Persistence controls; a run cannot initialize and resume together.
#[derive(Args)]
struct CheckpointArgs {
    /// Existing empty directory for a fresh run, or checkpoint directory when resuming.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Runtime weights directory loaded before the first update of a fresh run.
    #[arg(long, conflicts_with = "resume")]
    initial_weights: Option<std::path::PathBuf>,
    /// Resume strict model, optimizer, counters, RNG state and collection progress.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Existing directory receiving runtime weights `u<update>/` at milestones.
    #[arg(long, requires = "history_every")]
    history_directory: Option<std::path::PathBuf>,
    /// Milestone spacing in updates, exported at the first checkpoint past each
    /// milestone; the final update is always exported.
    #[arg(long, requires = "history_directory", value_parser = clap::builder::RangedU64ValueParser::<std::num::NonZeroU64>::new().range(1..=crate::MAX_TRAINING_COUNTER))]
    history_every: Option<std::num::NonZeroU64>,
    /// Wall seconds between checkpoints; a graceful stop and the last update
    /// always checkpoint. A crash loses at most one interval.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(60..=86_400))]
    checkpoint_interval_seconds: u64,
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
        Some(Operation::Eval(evaluation)) => return run_eval(evaluation),
        Some(Operation::Duel(duel)) => return crate::scripted::duel_cli::run(duel),
        None => arguments.play,
    };
    let (policy, weights_directory) = resolve_play_deployment(&play)?;
    if play.policy.is_none() && play.weights_directory.is_none() {
        eprintln!("deployment: repository-selected {policy:?}");
    }
    let outcome = match policy {
        PlayPolicy::Neural => {
            let directory = weights_directory.as_deref().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "neural requires --weights-directory",
                )
            })?;
            crate::seat::play_neural(&play.addr, &play.name, play.limit, directory)?
        }
        PlayPolicy::Teacher => crate::play_script(
            crate::ScriptKind::Teacher,
            &play.addr,
            &play.name,
            play.limit,
        )?,
        PlayPolicy::HarassPush => crate::play_script(
            crate::ScriptKind::HarassPush,
            &play.addr,
            &play.name,
            play.limit,
        )?,
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
        if policy != PlayPolicy::Neural && play.weights_directory.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "rule policies forbid --weights-directory",
            ));
        }
        return Ok((policy, play.weights_directory.clone()));
    }
    if play.weights_directory.is_some() {
        return Ok((PlayPolicy::Neural, play.weights_directory.clone()));
    }
    crate::default_deployment::DEFAULT_DEPLOYMENT.resolve()
}

#[cfg(feature = "builtin")]
fn run_train_annealed(arguments: TrainAnnealedArgs) -> std::io::Result<()> {
    let _flush = crate::telemetry::FlushLogOnDrop;
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
    crate::training_signals::install()?;
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
        crate::telemetry::log_line!(
            "annealed: seed {} resolved from {}; recorded in the run scope",
            settings.seed,
            if arguments.checkpoint.resume {
                "the resumed run scope"
            } else {
                "the operating system random source"
            },
        );
    }
    crate::telemetry::log_line!(
        "annealed: updates={} samples_per_update={} slots={} lanes={} simulation_threads={} simulation_groups={} pin_threads={} generation_updates={} zero_updates={} seed={} opponents={:?}",
        settings.updates,
        settings.ppo.samples_per_update,
        settings.slots,
        settings.lanes,
        settings.simulation_threads,
        settings.simulation_groups,
        settings.pin_threads,
        settings.generation_updates,
        settings.zero_updates,
        settings.seed,
        settings.opponents,
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
    crate::telemetry::log_line!(
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
    if crate::training_signals::stop_requested() {
        crate::telemetry::log_line!(
            "level=INFO event=training_stopped reason=signal completed_updates={}",
            report.completed_updates
        );
    }
    Ok(())
}

#[cfg(feature = "builtin")]
fn report_training_checkpoint(checkpoint: crate::TrainingCheckpointReport) {
    crate::telemetry::log_line!("checkpoint: update {}", checkpoint.completed_updates);
    if let Some(warning) = checkpoint.cleanup_warning {
        crate::telemetry::log_line!("checkpoint cleanup warning: {warning}");
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
        crate::AdaptiveEnvironmentLimits {
            base_updates: self.generation_updates,
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
        if self.generation_updates == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "annealed generation updates must be positive",
            ));
        }
        let scale = self.environment_scale()?;
        let zero_updates = self
            .zero_updates
            .unwrap_or_else(|| self.updates.div_ceil(5));
        let ppo = self.optimizer.ppo(crate::PpoConfig {
            samples_per_update: self.samples_per_update,
            gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
            ..crate::PpoConfig::default()
        });
        let cores = std::thread::available_parallelism()?.get();
        let simulation_threads = self.simulation_threads.unwrap_or(cores);
        let seed = self.resolved_seed()?;
        let (slots, lanes, simulation_groups) =
            self.resolved_collection_shape(cores, simulation_threads)?;
        Ok(crate::AnnealedJobConfig {
            environment_schedule,
            execution: crate::TrainingExecutionOptions {
                balanced_minibatches: self.balanced_minibatches,
                training_microbatch: self.training_microbatch,
            },
            updates: self.updates,
            invocation_updates: self.invocation_updates,
            history: self.checkpoint.history()?,
            slots,
            lanes,
            simulation_threads,
            simulation_groups,
            pin_threads: self.pin_threads,
            generation_updates: self.generation_updates,
            zero_updates,
            seed,
            scale,
            opponents: self.opponents.clone(),
            opponent_schedule: match self.opponent_schedule {
                OpponentScheduleArg::Fixed => crate::OpponentSchedule::Fixed,
                OpponentScheduleArg::Pfsp => crate::OpponentSchedule::Pfsp,
            },
            league_size: self.league_size,
            league_every: self.league_every,
            ppo,
            checkpoint_cadence: crate::TrainingCheckpointCadence::WallTime(
                std::time::Duration::from_secs(self.checkpoint.checkpoint_interval_seconds),
            ),
            git_commit,
            simulator_commit,
        })
    }

    /// Slots, lanes and simulation groups: explicit, adopted from the recorded
    /// resume scope, or sized to the host by [`collection_shape`].
    fn resolved_collection_shape(
        &self,
        cores: usize,
        threads: usize,
    ) -> std::io::Result<(usize, usize, usize)> {
        let recorded = |name| -> std::io::Result<Option<usize>> {
            if !self.checkpoint.resume {
                return Ok(None);
            }
            let run =
                crate::TrainingArtifact::load_run_scope(&self.checkpoint.checkpoint_directory)
                    .map_err(|error| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("train-annealed resume needs the recorded run scope: {error}"),
                        )
                    })?;
            Ok(recorded_option(&run.command_line, name))
        };
        let slots = match self.slots {
            Some(slots) => Some(slots),
            None => recorded("--slots")?,
        };
        let lanes = match self.lanes {
            Some(lanes) => Some(lanes),
            None => recorded("--lanes")?,
        };
        let domains = crate::ppo_arena::topology::cache_domains().len();
        let groups = self.simulation_groups.map(usize::from);
        Ok(collection_shape(
            (slots, lanes, groups),
            cores,
            domains,
            threads,
        ))
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
}

/// Parses one `--opponent` mixture entry; a weights path may itself contain colons.
fn parse_annealed_opponent(
    value: &str,
) -> Result<(crate::AnnealedOpponent, crate::EnvironmentDecimal), String> {
    let weight = |text: &str| {
        text.parse::<crate::EnvironmentDecimal>()
            .map_err(|error| error.to_string())
    };
    let one = crate::EnvironmentDecimal::from_units(crate::EnvironmentDecimal::SCALE);
    match value.split_once(':') {
        None if value == "teacher" => Ok((crate::AnnealedOpponent::Teacher, one)),
        None if value == "self" => Ok((crate::AnnealedOpponent::SelfPlay, one)),
        None if value == "harass-push" => Ok((crate::AnnealedOpponent::HarassPush, one)),
        Some(("teacher", rest)) => Ok((crate::AnnealedOpponent::Teacher, weight(rest)?)),
        Some(("self", rest)) => Ok((crate::AnnealedOpponent::SelfPlay, weight(rest)?)),
        Some(("harass-push", rest)) => {
            Ok((crate::AnnealedOpponent::HarassPush, weight(rest)?))
        }
        Some(("league", rest)) => Ok((crate::AnnealedOpponent::League, weight(rest)?)),
        Some(("weights", rest)) => {
            let (directory, share) = rest
                .rsplit_once(':')
                .ok_or("weights opponents need `weights:<directory>:<weight>`")?;
            if directory.is_empty() {
                return Err("weights opponents need a directory".to_owned());
            }
            Ok((
                crate::AnnealedOpponent::Weights(directory.into()),
                weight(share)?,
            ))
        }
        _ => Err(
            "opponent must be teacher[:weight], harass-push[:weight], self[:weight], weights:<directory>:<weight> or league:<weight>"
                .to_owned(),
        ),
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
    if !arguments.contains(&"--simulation-groups") {
        // Tests must not depend on the host's cache topology.
        full.extend(["--simulation-groups", "1"]);
    }
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
            gae_lambda_tick: self.gae_lambda_tick,
            entropy_coefficient: self.entropy_coefficient,
            value_coefficient: self.value_coefficient,
            ..config
        }
    }
}

#[cfg(feature = "builtin")]
impl CheckpointArgs {
    fn history(&self) -> std::io::Result<Option<crate::RuntimeHistory>> {
        let (Some(directory), Some(every)) = (&self.history_directory, self.history_every) else {
            return Ok(None);
        };
        if !directory.is_dir() {
            return Err(std::io::Error::other(
                "history directory must already exist and be a directory",
            ));
        }
        if directory.starts_with(&self.checkpoint_directory)
            || self.checkpoint_directory.starts_with(directory)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "history and checkpoint directories must not contain each other",
            ));
        }
        Ok(Some(crate::RuntimeHistory {
            directory: directory.clone(),
            every,
        }))
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

#[cfg(feature = "builtin")]
fn run_eval(arguments: EvalArgs) -> std::io::Result<()> {
    use crate::ppo_arena::PlayerSpec;
    let candidate = PlayerSpec::parse(&arguments.candidate).map_err(std::io::Error::other)?;
    let name = match (arguments.name, &candidate) {
        (Some(name), _) => name,
        (None, PlayerSpec::Weights(directory)) => directory
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| std::io::Error::other("weights directory has no name; pass --name"))?,
        (None, PlayerSpec::Script(kind)) => kind.label().to_owned(),
        (None, PlayerSpec::Average(_)) => {
            return Err(std::io::Error::other("an average candidate needs --name"));
        }
    };
    let settings = crate::ppo_arena::EvaluationSettings {
        candidate_name: name,
        candidate,
        pool: crate::ppo_arena::read_pool(&arguments.pool).map_err(std::io::Error::other)?,
        first_seed: arguments.seeds.0,
        seeds: arguments.seeds.1,
        parallel: usize::from(arguments.parallel),
        groups: usize::from(arguments.actor_pipeline_groups),
        greedy: arguments.greedy,
        device: arguments.device.policy_device(arguments.device_ordinal)?,
    };
    crate::ppo_arena::run_evaluation(&settings, &arguments.output).map_err(std::io::Error::other)
}

#[cfg(not(feature = "builtin"))]
fn run_eval(_: EvalArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "eval requires cargo feature `builtin`",
    ))
}

#[cfg(not(feature = "builtin"))]
fn run_train_annealed(_: TrainAnnealedArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "annealed training requires cargo feature `builtin`",
    ))
}

/// Slots, lanes and simulation groups from the requested values and the host:
/// 16 slots per core, a group per cache domain the shape can feed, and the
/// fewest lanes (at least two per group, at most 64 slots each) that split the
/// slots evenly across the groups. Explicit values are kept as given.
#[cfg(feature = "builtin")]
pub(crate) fn collection_shape(
    (slots, lanes, groups): (Option<usize>, Option<usize>, Option<usize>),
    cores: usize,
    domains: usize,
    threads: usize,
) -> (usize, usize, usize) {
    let slots = slots.unwrap_or_else(|| {
        let slots = (16 * cores).clamp(1, crate::PPO_MAX_SLOTS);
        // Explicit lanes cap the default at 64 slots each, split evenly.
        lanes.map_or(slots, |lanes| {
            let lanes = lanes.max(1);
            (slots.min(64 * lanes) / lanes).max(1) * lanes
        })
    });
    let shape = |groups: usize| match lanes {
        Some(lanes) => lanes.is_multiple_of(groups).then_some(lanes),
        None => (slots.div_ceil(64).max(2 * groups)..=slots)
            .find(|lanes| lanes.is_multiple_of(groups) && slots.is_multiple_of(*lanes)),
    };
    let groups = groups.unwrap_or_else(|| {
        (1..=domains.min(threads))
            .rev()
            .find(|&groups| shape(groups).is_some())
            .unwrap_or(1)
    });
    (slots, shape(groups).or(lanes).unwrap_or(1), groups)
}

/// The numeric value of `name` in a recorded command line.
#[cfg(feature = "builtin")]
fn recorded_option(command_line: &str, name: &str) -> Option<usize> {
    let mut tokens = command_line.split_whitespace();
    tokens.find(|token| *token == name)?;
    tokens.next()?.parse().ok()
}
