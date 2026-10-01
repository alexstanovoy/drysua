use super::*;

#[path = "adaptive_environment_cli.rs"]
mod adaptive_environment_tests;

#[path = "cli_fast_profile.rs"]
mod fast_profile_tests;

#[cfg(feature = "builtin")]
pub(crate) fn fixed_annealed_settings_for_test(
    overrides: &[&str],
) -> std::io::Result<crate::AnnealedJobConfig> {
    let mut arguments = vec!["--environment-schedule", "fixed"];
    arguments.extend_from_slice(overrides);
    annealed_settings_for_test(&arguments)
}

pub(crate) fn parse_from<I, T>(arguments: I) -> Result<(), clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(arguments).map(|_| ())
}

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

pub(crate) fn run_from_for_test<I, T>(arguments: I) -> std::io::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run(Cli::try_parse_from(arguments).map_err(std::io::Error::other)?)
}
