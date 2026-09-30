use super::*;

#[path = "adaptive_environment_cli.rs"]
mod adaptive_environment_tests;

#[path = "cli_fast_profile.rs"]
mod fast_profile_tests;

#[test]
fn adaptive_environment_cli_accepts_explicit_schedules_and_exact_credit() {
    for schedule in ["fixed", "adaptive"] {
        let mut arguments = vec![
            "drysua",
            "train-annealed",
            "--updates",
            "20",
            "--generation-updates",
            "4",
            "--checkpoint-directory",
            ".",
            "--environment-schedule",
            schedule,
        ];
        if schedule == "adaptive" {
            arguments.extend([
                "--environment-success-updates",
                "2",
                "--environment-success-rate",
                ".8",
                "--environment-poor-updates",
                "1",
                "--environment-poor-rate",
                ".2",
                "--environment-extension",
                "1.25",
            ]);
        }
        Cli::try_parse_from(arguments)
            .unwrap_or_else(|error| panic!("explicit {schedule} schedule must parse: {error}"));
    }
}

#[cfg(feature = "builtin")]
pub(crate) fn fixed_annealed_settings_for_test(
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut arguments = vec!["--environment-schedule", "fixed"];
    arguments.extend_from_slice(overrides);
    annealed_settings_for_test(&arguments)
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
