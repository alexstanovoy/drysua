use clap::{Args, Parser, Subcommand};

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
    let report = crate::run_league_smoke(crate::LeagueSmokeConfig {
        updates: arguments.updates,
        environments: arguments.environments,
        rollout_decisions: arguments.rollout,
        epochs: arguments.epochs,
        minibatch: arguments.minibatch,
        evaluation_pairs: arguments.evaluation_pairs,
        evaluation_decisions: arguments.evaluation_decisions,
        seed: arguments.seed,
        map: bota_proto::MapId(arguments.map),
    })
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
    let report = crate::run_ppo_smoke(crate::PpoSmokeConfig {
        updates: arguments.updates,
        environments: arguments.environments,
        rollout_decisions: arguments.rollout,
        epochs: arguments.epochs,
        minibatch: arguments.minibatch,
        seed: arguments.seed,
        map: bota_proto::MapId(arguments.map),
    })
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

#[cfg(not(feature = "builtin"))]
fn run_train(_: TrainArgs) -> std::io::Result<()> {
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
