use clap::{Args, Parser, Subcommand, ValueEnum};

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
    /// Run bounded stage-ten self-play league training and paired evaluation.
    League(LeagueArgs),
    /// Connect to a server and play one match.
    Play(PlayArgs),
    /// Run a bounded stage-nine PPO actor-to-learner smoke training.
    Train(TrainArgs),
    /// Run resumable PPO training with periodic strict checkpoints.
    TrainFull(TrainFullArgs),
}

/// Options for one server match.
#[derive(Args)]
struct PlayArgs {
    /// Server socket address.
    #[arg(long, default_value = "127.0.0.1:4455")]
    addr: String,
    /// Name shown in the lobby.
    #[arg(long, default_value = "drysua")]
    name: String,
    /// Leave after receiving this snapshot tick.
    #[arg(long, value_name = "TICKS")]
    limit: Option<u32>,
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
    /// Simulator map id, zero or one.
    #[arg(long, default_value_t = 1)]
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
    /// Total PPO update target, including updates restored from a checkpoint.
    #[arg(long)]
    updates: u64,
    /// Independent CPU arenas, hard-bounded to sixteen.
    #[arg(long, default_value_t = 4)]
    environments: usize,
    /// Decisions collected from each arena per update, hard-bounded to 64.
    #[arg(long, default_value_t = 8)]
    rollout: usize,
    /// PPO passes over one rollout.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = 32)]
    minibatch: usize,
    /// Monotonic wall-clock seconds between durable checkpoints.
    #[arg(long, default_value_t = 300)]
    checkpoint_seconds: u64,
    /// Existing empty directory for a fresh run, or checkpoint directory when resuming.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Resume strict model, optimizer, counters, pipeline generation, and actor RNG state.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Rebind an otherwise identical checkpoint to this build's Git provenance once.
    #[arg(long, default_value_t = false, requires = "resume")]
    migrate_provenance: bool,
    /// Deterministic training seed.
    #[arg(long, default_value_t = 9_001)]
    seed: u64,
    /// Simulator map id, zero or one.
    #[arg(long, default_value_t = 1)]
    map: u16,
    /// Learner tensor backend; actors and simulation remain on CPU.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA or Metal device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

/// Options for a short builtin self-play league verification run.
#[derive(Args)]
struct LeagueArgs {
    /// PPO updates, bounded to ten for this smoke command.
    #[arg(long, default_value_t = 1)]
    updates: u32,
    /// Independent CPU arenas, bounded to sixteen for desktop headroom.
    #[arg(long, default_value_t = 4)]
    environments: usize,
    /// Decisions collected from each arena per update.
    #[arg(long, default_value_t = 8)]
    rollout: usize,
    /// PPO passes over one rollout.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Effective Adam minibatch.
    #[arg(long, default_value_t = 32)]
    minibatch: usize,
    /// Held-out seeds evaluated once from each side.
    #[arg(long, default_value_t = 2)]
    evaluation_pairs: usize,
    /// Decisions made in each held-out match, bounded to 1024.
    #[arg(long, default_value_t = 8)]
    evaluation_decisions: usize,
    /// Deterministic training seed.
    #[arg(long, default_value_t = 10_001)]
    seed: u64,
    /// Simulator map id, zero or one.
    #[arg(long, default_value_t = 1)]
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

/// Parses command line arguments and plays one match.
pub fn run_from_env() -> std::io::Result<()> {
    run(Cli::parse())
}

fn run(arguments: Cli) -> std::io::Result<()> {
    let play = match arguments.operation {
        Some(Operation::League(league)) => return run_league(league),
        Some(Operation::Play(play)) => play,
        Some(Operation::Train(train)) => return run_train(train),
        Some(Operation::TrainFull(train)) => return run_train_full(train),
        None => arguments.play,
    };
    let outcome = crate::play(&play.addr, &play.name, play.limit)?;
    println!(
        "played {} ticks as {:?}; winner {:?}; {} rejected orders",
        outcome.ticks, outcome.team, outcome.winner, outcome.rejections
    );
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_league(arguments: LeagueArgs) -> std::io::Result<()> {
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    let report = crate::run_league_smoke_on(
        crate::LeagueSmokeConfig {
            updates: arguments.updates,
            environments: arguments.environments,
            rollout_decisions: arguments.rollout,
            epochs: arguments.epochs,
            minibatch: arguments.minibatch,
            evaluation_pairs: arguments.evaluation_pairs,
            evaluation_decisions: arguments.evaluation_decisions,
            seed: arguments.seed,
            map: bota_proto::MapId(arguments.map),
        },
        device,
    )
    .map_err(std::io::Error::other)?;
    println!(
        "league smoke: {} updates, {} transitions, opponents {:?}, {} paired seeds, {} policies, {} promotions, {} rejected evaluation actions",
        report.ppo.updates,
        report.ppo.transitions,
        report.opponent_counts,
        report.paired_evaluations,
        report.league_policies,
        report.promotions,
        report.evaluation_rejections,
    );
    Ok(())
}

#[cfg(not(feature = "builtin"))]
fn run_league(_: LeagueArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "self-play league requires cargo feature `builtin`",
    ))
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
        "PPO smoke: {} updates, {} transitions, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}, {} rejected orders, {} arena ticks",
        report.updates,
        report.transitions,
        report.optimizer_step,
        report.final_policy_loss,
        report.final_value_loss,
        report.final_entropy,
        report.final_kl,
        report.rejected_orders,
        report.elapsed_ticks,
    );
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_train_full(arguments: TrainFullArgs) -> std::io::Result<()> {
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    validate_checkpoint_directory(&arguments.checkpoint_directory, arguments.resume)?;
    let settings = crate::TrainingJobConfig {
        updates: arguments.updates,
        environments: arguments.environments,
        rollout_decisions: arguments.rollout,
        epochs: arguments.epochs,
        minibatch: arguments.minibatch,
        checkpoint_cadence: crate::TrainingCheckpointCadence::WallTime(
            std::time::Duration::from_secs(arguments.checkpoint_seconds),
        ),
        resume_provenance: if arguments.migrate_provenance {
            crate::ResumeProvenance::MigrateGitCommit
        } else {
            crate::ResumeProvenance::Strict
        },
        seed: arguments.seed,
        map: bota_proto::MapId(arguments.map),
        git_commit: embedded_commit("DRYSUA_GIT_COMMIT", option_env!("DRYSUA_GIT_COMMIT"))?,
        simulator_commit: embedded_commit("BOTA_GIT_COMMIT", option_env!("BOTA_GIT_COMMIT"))?,
    };
    let report = crate::run_training_job_on(
        settings,
        device,
        &arguments.checkpoint_directory,
        arguments.resume,
        |checkpoint| {
            println!(
                "checkpoint: update {}, samples {}, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}, KL stop {}",
                checkpoint.completed_updates,
                checkpoint.rollout_samples,
                checkpoint.optimizer_step,
                checkpoint.policy_loss,
                checkpoint.value_loss,
                checkpoint.entropy,
                checkpoint.approximate_kl,
                checkpoint.stopped_for_kl,
            );
            if let Some(warning) = checkpoint.cleanup_warning {
                eprintln!("checkpoint cleanup warning: {warning}");
            }
        },
    )
    .map_err(std::io::Error::other)?;
    println!(
        "training complete: {} updates, {} samples, optimizer step {}, policy loss {:.6}, value loss {:.6}, entropy {:.6}, KL {:.6}",
        report.completed_updates,
        report.rollout_samples,
        report.optimizer_step,
        report.final_policy_loss,
        report.final_value_loss,
        report.final_entropy,
        report.final_kl,
    );
    Ok(())
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

#[cfg(test)]
pub(crate) fn parse_from<I, T>(arguments: I) -> Result<(), clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(arguments).map(|_| ())
}
