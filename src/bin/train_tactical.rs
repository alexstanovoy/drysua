use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use drysua::{
    TACTICAL_FILE_BYTES, TacticalCohort, TacticalPolicy, TacticalSearchConfig,
    evaluate_tactical_population, evaluate_tactical_population_against, run_tactical_search,
    save_tactical_archive, tactical_search_founders,
};

#[derive(Parser)]
#[command(about = "Bounded development-only neural tactical population search")]
struct Arguments {
    #[arg(long)]
    output_directory: PathBuf,
    #[arg(long, default_value_t = 16)]
    population: usize,
    #[arg(long, default_value_t = 20)]
    generations: u32,
    #[arg(long, default_value_t = 4)]
    pairs: usize,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 9_200_000)]
    seed: u64,
    #[arg(long, default_value_t = 9_200_100)]
    selection_seed: u64,
    #[arg(long, default_value_t = 9_200_200)]
    confirmation_seed: u64,
    #[arg(long, default_value_t = 16)]
    selection_pairs: usize,
    #[arg(long, default_value_t = 30_000)]
    tick_limit: u32,
    #[arg(long, default_value_t = 1_200)]
    wall_seconds: u64,
    /// Validated standalone policy to continue searching, or evaluate with --probe.
    #[arg(long)]
    initial_policy: Option<PathBuf>,
    /// Immutable current-schema opponent artifact; adds its cohort alongside pure Teacher.
    #[arg(long)]
    opponent_policy: Option<PathBuf>,
    /// Evaluate founders (or --initial-policy) without running optimization.
    #[arg(long)]
    probe: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    validate_output(&arguments.output_directory)?;
    let policy = arguments
        .initial_policy
        .as_deref()
        .map(load_policy)
        .transpose()?;
    let settings = TacticalSearchConfig {
        opponent_policy: arguments
            .opponent_policy
            .as_deref()
            .map(load_policy)
            .transpose()?,
        population: arguments.population,
        generations: arguments.generations,
        pairs: arguments.pairs,
        workers: arguments.workers,
        seed: arguments.seed,
        selection_seed: arguments.selection_seed,
        confirmation_seed: arguments.confirmation_seed,
        selection_pairs: arguments.selection_pairs,
        tick_limit: arguments.tick_limit,
        wall_time: Duration::from_secs(arguments.wall_seconds),
    };
    if arguments.probe {
        return probe(&arguments.output_directory, policy, &settings);
    }
    let started = Instant::now();
    let report = run_tactical_search(&settings, &arguments.output_directory, policy.as_ref())?;
    println!(
        "phase=minimum_result selection_minimum_percent={:.3} confirmation_minimum_percent={:?}",
        report.selection.fitness.minimum_win_percent(),
        report
            .confirmation
            .as_ref()
            .map(|result| result.fitness.minimum_win_percent())
    );
    println!(
        "phase=finished generations={} evaluated_games={} deadline={} selection_wins={} selection_games={} win_percent={:.3} paired_sweep_lower_percent={:.3} archive={} elapsed_seconds={:.3}",
        report.completed_generations,
        report.evaluated_games,
        report.stopped_for_deadline,
        report.selection.fitness.wins,
        report.selection.games.len(),
        report.selection.fitness.win_percent(),
        report.selection.fitness.paired_sweep_lower_percent(),
        report.archive.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn probe(
    output: &Path,
    initial: Option<TacticalPolicy>,
    settings: &TacticalSearchConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let policies = initial.map_or_else(
        || tactical_search_founders().to_vec(),
        |policy| vec![policy],
    );
    let cohort = TacticalCohort::new(settings.seed, settings.pairs, settings.tick_limit)?;
    let started = Instant::now();
    let evaluations = match &settings.opponent_policy {
        Some(opponent) => evaluate_tactical_population_against(
            &policies,
            cohort,
            opponent,
            settings.workers,
            settings.wall_time,
        )?,
        None => {
            evaluate_tactical_population(&policies, cohort, settings.workers, settings.wall_time)?
        }
    };
    std::fs::create_dir(output)?;
    for (index, (policy, evaluated)) in policies.iter().zip(&evaluations).enumerate() {
        let label = format!("probe-{index}");
        let archive = save_tactical_archive(output, &label, policy, evaluated)?;
        let overrides: u32 = evaluated
            .games
            .iter()
            .map(|game| game.effective_overrides)
            .sum();
        let sampled: u32 = evaluated
            .games
            .iter()
            .map(|game| game.sampled_decisions)
            .sum();
        println!(
            "phase={label} wins={} losses={} timeouts={} sampled={sampled} overrides={overrides} archive={}",
            evaluated.fitness.wins,
            evaluated.fitness.losses,
            evaluated.fitness.timeouts,
            archive.display()
        );
    }
    println!(
        "phase=probe_finished games={} elapsed_seconds={:.3}",
        probe_game_count(&evaluations),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn probe_game_count(evaluations: &[drysua::TacticalEvaluation]) -> usize {
    assert!(evaluations.len() <= 24);
    evaluations
        .iter()
        .map(|evaluation| evaluation.games.len())
        .sum()
}

fn load_policy(path: &Path) -> Result<TacticalPolicy, Box<dyn std::error::Error>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() != TACTICAL_FILE_BYTES as u64
    {
        return Err("initial tactical policy must be a regular non-symlink file with the exact schema byte count".into());
    }
    let mut bytes = Vec::with_capacity(TACTICAL_FILE_BYTES + 1);
    std::fs::File::open(path)?
        .take(TACTICAL_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(TacticalPolicy::from_bytes(&bytes)?)
}

fn validate_output(output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .canonicalize()?;
    let parent = output
        .parent()
        .ok_or("output requires an existing parent")?
        .canonicalize()?;
    if !parent.starts_with(root) || output.file_name().is_none() {
        return Err("tactical output must be a new directory below artifacts/temp".into());
    }
    if output.try_exists()? {
        return Err("tactical output directory already exists".into());
    }
    Ok(())
}

#[test]
fn probe_game_count_includes_both_mixed_opponent_cohorts() {
    let policy = TacticalPolicy::default();
    let cohort = TacticalCohort::new(9_203_900, 1, 2).expect("cohort");
    let evaluations = evaluate_tactical_population_against(
        std::slice::from_ref(&policy),
        cohort,
        &policy,
        1,
        Duration::from_secs(30),
    )
    .expect("mixed probe");
    assert_eq!(probe_game_count(&evaluations), 4);
    assert_eq!(evaluations.len(), 1);
}
