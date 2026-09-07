use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use drysua::{NeuralTrainingConfig, PolicyDevice, run_neural_training};

#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(about = "Fresh pure-neural Map0 behavioral-cloning diagnostic; never promotes defaults")]
struct Arguments {
    #[arg(long, required = true)]
    map0: bool,
    #[arg(long)]
    output_directory: PathBuf,
    #[arg(long, default_value_t = 9840000)]
    seed: u64,
    #[arg(long, default_value_t = 6)]
    training_games: usize,
    #[arg(long, default_value_t = 64)]
    epochs: u32,
    #[arg(long, default_value_t = 108900)]
    tick_limit: u32,
    #[arg(long, default_value_t = 1800)]
    wall_seconds: u64,
    #[arg(long, value_enum, default_value = "cuda")]
    device: Device,
    #[arg(long, default_value_t = 1)]
    dagger_rounds: usize,
    #[arg(long, default_value_t = 32)]
    dagger_epochs: u32,
    /// Current M11 weights with a new optimizer, never an optimizer resume.
    #[arg(long, conflicts_with = "initialize_selected_m10")]
    initial_weights: Option<PathBuf>,
    /// Explicit approved immutable M10 source initialization into a new M11 model.
    #[arg(long, conflicts_with = "initial_weights")]
    initialize_selected_m10: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Arguments::parse();
    assert!(args.map0);
    println!(
        "feature_schema={} feature_hash={} model_schema={} model_hash={} parameters={} batch=64 dtype=F32",
        drysua::FEATURE_SCHEMA_VERSION,
        drysua::FEATURE_SCHEMA_HASH,
        drysua::MODEL_SCHEMA_VERSION,
        drysua::MODEL_SCHEMA_HASH,
        drysua::MODEL_PARAMETER_COUNT
    );
    let device = match args.device {
        Device::Cpu => PolicyDevice::Cpu,
        #[cfg(feature = "cuda")]
        Device::Cuda => PolicyDevice::Cuda { ordinal: 0 },
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            return Err("--device cuda requires a release build with features builtin,cuda".into());
        }
    };
    run_neural_training(
        &NeuralTrainingConfig {
            seed: args.seed,
            training_games: args.training_games,
            epochs: args.epochs,
            tick_limit: args.tick_limit,
            wall_time: Duration::from_secs(args.wall_seconds),
            device,
            dagger_rounds: args.dagger_rounds,
            dagger_epochs: args.dagger_epochs,
            initial_weights: args.initial_weights,
            initialize_selected_m10: args.initialize_selected_m10,
        },
        &args.output_directory,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_flags_accept_one_source_and_reject_conflicts() {
        for flag in ["--initial-weights", "--initialize-selected-m10"] {
            let arguments = Arguments::try_parse_from([
                "train_neural",
                "--map0",
                "--output-directory",
                "output",
                flag,
                "source",
            ])
            .expect("explicit source");
            assert_ne!(
                arguments.initial_weights.is_some(),
                arguments.initialize_selected_m10.is_some()
            );
        }
        let error = Arguments::try_parse_from([
            "train_neural",
            "--map0",
            "--output-directory",
            "output",
            "--initial-weights",
            "current",
            "--initialize-selected-m10",
            "legacy",
        ])
        .err()
        .expect("conflicting flags");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
        assert!(error.to_string().contains("cannot be used with"));
    }
}
