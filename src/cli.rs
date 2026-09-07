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
    /// Evaluate deployment weights across both maps, baselines, and sides.
    Evaluate(EvaluateArgs),
    /// Run bounded stage-ten self-play league training and paired evaluation.
    League(LeagueArgs),
    /// Connect to a server and play one match.
    Play(PlayArgs),
    /// Clone bounded Map1 teacher trajectories and gate the hybrid deployment policy.
    Pretrain(PretrainArgs),
    /// Run a bounded stage-nine PPO actor-to-learner smoke training.
    Train(TrainArgs),
    /// Run resumable PPO training with periodic strict checkpoints.
    TrainFull(TrainFullArgs),
}

/// Options for Map1 teacher pretraining with a fixed two-map deployment gate.
#[derive(Args)]
struct PretrainArgs {
    /// Existing empty output directory.
    #[arg(long)]
    output_directory: std::path::PathBuf,
    /// Complete behavioral passes: 8, 16, 24, or 32.
    #[arg(long, default_value_t = 8)]
    epochs: u32,
    /// Deterministic dataset and optimizer seed.
    #[arg(long, default_value_t = 50_001)]
    seed: u64,
    /// Tensor backend used by behavioral optimization.
    #[arg(long, value_enum, default_value_t = LearnerDevice::Cpu)]
    device: LearnerDevice,
    /// CUDA or Metal device ordinal.
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

/// Options for a fixed deterministic checkpoint evaluation matrix.
#[derive(Args)]
struct EvaluateArgs {
    /// Evaluate greedy Neural on Map0 only, without candidate Teacher overrides.
    #[arg(long)]
    neural_map0: bool,
    /// Omit the secondary weak-opponent cohort from explicit pure Neural evaluation.
    #[arg(long, requires = "neural_map0")]
    teacher_only: bool,
    /// Directory containing drysua.weights.safetensors.
    #[arg(long)]
    checkpoint_directory: std::path::PathBuf,
    /// Paired held-out seeds per map and baseline, bounded to eight.
    #[arg(long, default_value_t = 1)]
    pairs: usize,
    /// Greedy decisions: at most 4096 for Hybrid, 36300 (tick cap 108900) for Neural.
    #[arg(long, default_value_t = 1_024)]
    decisions: usize,
    /// First deterministic held-out seed.
    #[arg(long, default_value_t = 90_001)]
    seed: u64,
}

/// Options for one server match.
#[derive(Args)]
struct PlayArgs {
    /// Override the repository-selected default with Neural, Hybrid, Tactical, or Teacher.
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
    #[arg(long, required_if_eq_any([("policy", "tactical"), ("policy", "neural")]))]
    weights_directory: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PlayPolicy {
    Hybrid,
    Neural,
    Tactical,
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
    /// Use only actual Map0 terminal win/loss rewards; requires complete episodes.
    #[arg(long, requires = "complete_episodes")]
    terminal_only: bool,
    /// Collect two, four, or six complete Map0 matches against Teacher; retain every eighth action.
    #[arg(long)]
    complete_episodes: bool,
    /// Adam learning rate; must be finite and positive.
    #[arg(long, default_value_t = crate::PpoConfig::default().learning_rate)]
    learning_rate: f32,
    /// Discount per simulator tick in [0, 1), not per policy decision.
    #[arg(long, default_value_t = crate::PpoConfig::default().gamma_tick)]
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
    /// Independent CPU arenas, hard-bounded to sixteen.
    #[arg(long, default_value_t = 4)]
    environments: usize,
    /// Window decisions or per-episode retained capacity, hard-bounded to 16384.
    #[arg(long, default_value_t = 2_048)]
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
        Some(Operation::Evaluate(evaluate)) => return run_evaluate(evaluate),
        Some(Operation::League(league)) => return run_league(league),
        Some(Operation::Play(play)) => play,
        Some(Operation::Pretrain(pretrain)) => return run_pretrain(pretrain),
        Some(Operation::Train(train)) => return run_train(train),
        Some(Operation::TrainFull(train)) => return run_train_full(train),
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
        PlayPolicy::Tactical => {
            let directory = weights_directory.as_deref().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "tactical requires --weights-directory",
                )
            })?;
            crate::seat::play_tactical(&play.addr, &play.name, play.limit, directory)?
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
        return Ok((policy, play.weights_directory.clone()));
    }
    if play.weights_directory.is_some() {
        return Ok((PlayPolicy::Hybrid, play.weights_directory.clone()));
    }
    crate::default_deployment::DEFAULT_DEPLOYMENT.resolve()
}

#[cfg(feature = "builtin")]
fn run_pretrain(arguments: PretrainArgs) -> std::io::Result<()> {
    let device = arguments.device.policy_device(arguments.device_ordinal)?;
    validate_checkpoint_directory(&arguments.output_directory, false)?;
    let report = crate::run_behavioral_pretraining_on(
        crate::BehavioralPretrainingConfig {
            epochs: arguments.epochs,
            seed: arguments.seed,
        },
        device,
        &arguments.output_directory,
    )
    .map_err(std::io::Error::other)?;
    println!(
        "pretraining: samples={} validation={} held_out={} optimizer_steps={} loss={:.6} kind_agreement={:.6} full_agreement={:.6} map0_validation_progress={} map1_validation_wins={} gameplay_validation_failures={} fingerprint={:016x} actions={:?}",
        report.training_samples,
        report.validation_samples,
        report.held_out_samples,
        report.optimizer_steps,
        report.final_loss,
        report.held_out_kind_agreement,
        report.held_out_full_agreement,
        report.gameplay_validation_map_zero_progress,
        report.gameplay_validation_map_one_wins,
        report.gameplay_validation_failures,
        report.fingerprint,
        report.action_counts,
    );
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_evaluate(arguments: EvaluateArgs) -> std::io::Result<()> {
    let settings = crate::CheckpointEvaluationConfig {
        pairs: arguments.pairs,
        decisions: arguments.decisions,
        seed: arguments.seed,
    };
    let report = if arguments.neural_map0 {
        crate::ppo_arena::evaluate_neural_map_zero_checkpoint_cohort(
            settings,
            &arguments.checkpoint_directory,
            arguments.teacher_only,
        )
    } else {
        crate::evaluate_runtime_checkpoint(settings, &arguments.checkpoint_directory)
    }
    .map_err(std::io::Error::other)?;
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut timeouts = 0usize;
    for game in &report.games {
        match game.outcome {
            crate::CheckpointEvaluationOutcome::Win
                if game.rejected_orders == 0 && game.baseline_rejected_orders == 0 =>
            {
                wins += 1
            }
            crate::CheckpointEvaluationOutcome::Win => {}
            crate::CheckpointEvaluationOutcome::Loss => losses += 1,
            crate::CheckpointEvaluationOutcome::Timeout => timeouts += 1,
        }
        println!(
            "map={} baseline={:?} team={:?} seed={} outcome={:?} decisions={} orders={} rejected={} baseline_orders={} baseline_rejected={} continue={} actions={:?} level={} gold={} kills={} deaths={} last_hits={} denies={} enemy_structures_destroyed={} elapsed_ticks={}",
            game.map.0,
            game.baseline,
            game.candidate_team,
            game.seed,
            game.outcome,
            game.decisions,
            game.wire_orders,
            game.rejected_orders,
            game.baseline_wire_orders,
            game.baseline_rejected_orders,
            game.action_counts[crate::ActionKind::Continue.index()],
            game.action_counts,
            game.final_summary.own_level,
            game.final_summary.own_gold,
            game.final_summary.allied.kills,
            game.final_summary.allied.deaths,
            game.final_summary.allied.last_hits,
            game.final_summary.allied.denies,
            game.final_summary.enemy_structures_destroyed,
            game.elapsed_ticks,
        );
    }
    let quality = report.quality();
    println!(
        "evaluation: fingerprint={:016x} games={} wins={} losses={} timeouts={} idle={} collapsed={} baseline_failures={} weak_losses={} weak_stalled={} rejected={} quality_pass={}",
        report.fingerprint,
        report.games.len(),
        wins,
        losses,
        timeouts,
        quality.idle_games,
        quality.collapsed_games,
        quality.baseline_failure_games,
        quality.weak_loss_games,
        quality.weak_stalled_games,
        quality.rejected_orders,
        quality.passed,
    );
    if !quality.passed {
        return Err(std::io::Error::other(
            "checkpoint evaluation quality gate failed",
        ));
    }
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

#[cfg(not(feature = "builtin"))]
fn run_evaluate(_: EvaluateArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "checkpoint evaluation requires cargo feature `builtin`",
    ))
}

#[cfg(not(feature = "builtin"))]
fn run_pretrain(_: PretrainArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "teacher pretraining requires cargo feature `builtin`",
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
    Ok(())
}

#[cfg(feature = "builtin")]
fn run_train_full(arguments: TrainFullArgs) -> std::io::Result<()> {
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
        |checkpoint| {
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
            if let Some(warning) = checkpoint.cleanup_warning {
                eprintln!("checkpoint cleanup warning: {warning}");
            }
        },
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
    Ok(())
}

#[cfg(feature = "builtin")]
impl TrainFullArgs {
    fn training_settings(
        &self,
        git_commit: String,
        simulator_commit: String,
    ) -> std::io::Result<crate::TrainingJobConfig> {
        let ppo = crate::PpoConfig {
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
        Ok(crate::TrainingJobConfig {
            terminal_only: self.terminal_only,
            complete_episodes: self.complete_episodes,
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
        })
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

#[cfg(test)]
pub(crate) fn play_policy_for_test<I, T>(arguments: I) -> Result<PlayPolicy, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::try_parse_from(arguments)?;
    let play = match cli.operation {
        Some(Operation::Play(play)) => play,
        None => cli.play,
        _ => panic!("expected play arguments"),
    };
    resolve_play_deployment(&play)
        .map(|(policy, _)| policy)
        .map_err(|error| clap::Error::raw(clap::error::ErrorKind::ValueValidation, error))
}

#[cfg(test)]
pub(crate) fn run_from_for_test<I, T>(arguments: I) -> std::io::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run(Cli::try_parse_from(arguments).map_err(std::io::Error::other)?)
}
