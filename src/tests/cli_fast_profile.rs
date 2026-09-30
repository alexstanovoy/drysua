#[cfg(feature = "builtin")]
use super::*;

#[cfg(feature = "builtin")]
fn arguments(extra: &[&str]) -> TrainAnnealedArgs {
    let mut command = vec![
        "drysua",
        "train-annealed",
        "--updates",
        "200",
        "--generation-updates",
        "4",
        "--checkpoint-directory",
        "unused-fast-profile",
    ];
    command.extend_from_slice(extra);
    let cli = Cli::try_parse_from(command).expect("annealed arguments");
    let Some(Operation::TrainAnnealed(arguments)) = cli.operation else {
        panic!("annealed operation");
    };
    arguments
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_resolves_the_continuous_profile_and_validates() {
    let settings = arguments(&["--simulation-threads", "3"])
        .annealed_settings("drysua".into(), "bota".into())
        .unwrap();
    // Host-sized defaults: 16 slots per core in lanes of at most 64 slots per group.
    let cores = std::thread::available_parallelism().unwrap().get();
    assert_eq!(settings.slots, (16 * cores).min(256));
    assert!(settings.lanes >= 2 * settings.simulation_groups);
    assert!(settings.slots.is_multiple_of(settings.lanes) && settings.slots / settings.lanes <= 64);
    assert!(!settings.pin_threads);
    assert_eq!(settings.simulation_threads, 3);
    assert_eq!(settings.ppo.samples_per_update, 24_000);
    assert_eq!(settings.generation_updates, 4);
    assert_eq!(settings.zero_updates, 40);
    assert_eq!(
        settings.opponents,
        [(
            crate::AnnealedOpponent::Teacher,
            crate::EnvironmentDecimal::from_units(1_000_000)
        )]
    );
    assert_eq!(
        settings.environment_schedule,
        crate::EnvironmentSchedule::default()
    );
    crate::ppo_arena::validate_annealed(&settings, Default::default()).unwrap();
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_cli_parses_an_ordered_opponent_mixture_with_colon_paths() {
    let parsed = arguments(&[
        "--opponent",
        "teacher:0.5",
        "--opponent",
        "weights:/runs/a:b/u100:0.25",
        "--opponent",
        "self",
        "--opponent",
        "harass-push:0.25",
    ]);
    let decimal = crate::EnvironmentDecimal::from_units;
    assert_eq!(
        parsed.opponents,
        [
            (crate::AnnealedOpponent::Teacher, decimal(500_000)),
            (
                crate::AnnealedOpponent::Weights("/runs/a:b/u100".into()),
                decimal(250_000)
            ),
            (crate::AnnealedOpponent::SelfPlay, decimal(1_000_000)),
            (crate::AnnealedOpponent::HarassPush, decimal(250_000)),
        ]
    );
    for invalid in [
        "weights:/runs/u100",
        "league:1",
        "teacher:-1",
        "self:0.1234567",
    ] {
        let command = [
            "drysua",
            "train-annealed",
            "--updates",
            "2",
            "--generation-updates",
            "1",
            "--checkpoint-directory",
            "unused",
            "--opponent",
            invalid,
        ];
        assert!(Cli::try_parse_from(command).is_err(), "{invalid}");
    }
}

#[cfg(feature = "builtin")]
#[test]
fn annealed_admission_rejects_lanes_that_do_not_divide_slots_or_hold_too_many() {
    for (slots, lanes, samples, message) in [
        (64, 3, 8_000, "annealed lanes must divide slots"),
        (130, 2, 8_000, "annealed lanes hold at most 64 slots each"),
        (
            64,
            2,
            7_999,
            "annealed samples per update must be a multiple of lanes within 1..=32768",
        ),
        (257, 1, 8_000, "annealed slots must be within 1..=256"),
    ] {
        let settings = arguments(&[
            "--slots",
            &slots.to_string(),
            "--lanes",
            &lanes.to_string(),
            "--samples-per-update",
            &samples.to_string(),
        ])
        .annealed_settings("drysua".into(), "bota".into())
        .unwrap();
        assert_eq!(
            crate::ppo_arena::validate_annealed(&settings, Default::default()),
            Err(crate::PpoError::InvalidConfig(message))
        );
    }
}
