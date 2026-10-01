//! Seed resolution for the annealed loop: explicit, adopted or freshly random.

use super::*;

const PROFILE: &[&str] = &[
    "--updates",
    "2",
    "--generation-updates",
    "1",
    "--environment-schedule",
    "fixed",
    "--slots",
    "2",
    "--samples-per-update",
    "6",
    "--minibatch",
    "3",
    "--epochs",
    "1",
    "--simulation-threads",
    "2",
];

#[test]
fn fresh_annealed_run_without_a_seed_records_the_resolved_seed_in_scope() {
    let options = crate::cli::annealed_settings_for_test_without_seed(PROFILE)
        .expect("fresh settings without an explicit seed");
    let pool = load_opponents(&options).expect("opponents");
    let run = annealed_run(&options, PolicyDevice::Cpu, options.ppo, harness(), &pool)
        .expect("fresh scope");
    assert_eq!(run.run_seed, options.seed);
    assert!(
        run.command_line
            .contains(&format!("--seed {}", options.seed)),
        "scope must record the resolved seed: {}",
        run.command_line
    );
}

#[test]
fn resume_without_a_seed_requires_a_readable_run_scope() {
    let directory = test_directory("annealed-seed-unreadable");
    let error = crate::cli::annealed_resume_settings_for_test(&directory, PROFILE)
        .expect_err("a resume without a recorded scope cannot resolve a seed");
    let text = error.to_string();
    assert!(text.contains("run scope"), "{text}");
    assert!(text.contains("--seed"), "{text}");
}

#[test]
fn resume_adopts_the_recorded_seed_and_rejects_a_different_explicit_one() {
    let directory = test_directory("annealed-seed-resume");
    let options = crate::cli::annealed_settings_for_test_without_seed(PROFILE)
        .expect("fresh settings without an explicit seed");
    run_with(options.clone(), harness(), &directory, false).expect("fresh committed update");
    let before = checkpoint_digests(&directory);
    let resume = |seed: Option<&str>| {
        let mut overrides = PROFILE.to_vec();
        overrides.extend(seed.map(|seed| ["--seed", seed]).into_iter().flatten());
        crate::cli::annealed_resume_settings_for_test(&directory, &overrides)
            .expect("resume settings")
    };
    let adopted = resume(None);
    assert_eq!(adopted.seed, options.seed);
    // Naming the recorded seed resolves the same settings as adopting it.
    assert_eq!(resume(Some(&options.seed.to_string())), adopted);
    // `run_seed` is a typed scope field, so a changed seed reports the same
    // generic compatibility error as any other run-scope change and commits nothing.
    let error = run_with(resume(Some("123456789")), harness(), &directory, true)
        .expect_err("explicit seed must match the recorded scope");
    assert!(error.to_string().contains("compatibility scope"), "{error}");
    assert_eq!(checkpoint_digests(&directory), before);
    let report = run_with(adopted, harness(), &directory, true).expect("adopted seed resumes");
    assert_eq!(report.completed_updates, 2);
}
