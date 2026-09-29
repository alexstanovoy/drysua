//! Seed resolution for the annealed loop: explicit, adopted or freshly random.

use super::*;

const PROFILE: &[&str] = &[
    "--updates",
    "2",
    "--games",
    "2",
    "--parallel",
    "2",
    "--actor-pipeline-groups",
    "1",
    "--training-microbatch",
    "64",
    "--generation-games",
    "2",
    "--environment-schedule",
    "fixed",
];

#[test]
fn explicit_annealed_seed_is_honored() {
    let mut overrides = PROFILE.to_vec();
    overrides.extend(["--seed", "424242"]);
    let settings = crate::cli::annealed_settings_for_test(&overrides).expect("explicit seed");
    assert_eq!(settings.seed, 424242);
}

#[test]
fn fresh_annealed_run_without_a_seed_records_the_resolved_seed_in_scope() {
    let options = crate::cli::annealed_settings_for_test_without_seed(PROFILE)
        .expect("fresh settings without an explicit seed");
    let run = annealed_run(&options, PolicyDevice::Cpu, options.ppo, harness(), None)
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
fn resume_without_a_seed_adopts_the_recorded_scope_seed() {
    let directory = test_directory("annealed-seed-adoption");
    let options = crate::cli::annealed_settings_for_test_without_seed(PROFILE)
        .expect("fresh settings without an explicit seed");
    run_with(options.clone(), harness(), &directory, false).expect("fresh committed update");
    let adopted = crate::cli::annealed_resume_settings_for_test(&directory, PROFILE)
        .expect("resume settings");
    assert_eq!(adopted.seed, options.seed);
    // The adopted seed rebuilds the recorded scope, so the resume proceeds.
    assert_eq!(
        run_with(adopted, harness(), &directory, true)
            .expect("adopted seed resumes")
            .completed_updates,
        2
    );
    std::fs::remove_dir_all(directory).expect("remove own checkpoint");
}

#[test]
fn resume_without_a_seed_requires_a_readable_run_scope() {
    let directory = test_directory("annealed-seed-unreadable");
    let error = crate::cli::annealed_resume_settings_for_test(&directory, PROFILE)
        .expect_err("a resume without a recorded scope cannot resolve a seed");
    let text = error.to_string();
    assert!(text.contains("run scope"), "{text}");
    assert!(text.contains("--seed"), "{text}");
    std::fs::remove_dir_all(directory).expect("remove own directory");
}

#[test]
fn resume_with_a_different_explicit_seed_rejects_the_scope() {
    let directory = test_directory("annealed-seed-conflict");
    let options = crate::cli::annealed_settings_for_test_without_seed(PROFILE)
        .expect("fresh settings without an explicit seed");
    run_with(options.clone(), harness(), &directory, false).expect("fresh committed update");
    let before = checkpoint_digests(&directory);
    let mut conflicting = PROFILE.to_vec();
    conflicting.extend(["--seed", "123456789"]);
    let requested =
        crate::cli::annealed_resume_settings_for_test(&directory, &conflicting).expect("resume");
    // `run_seed` is a typed scope field, so a changed seed reports the same
    // generic compatibility error as any other run-scope change and commits nothing.
    let error = run_with(requested, harness(), &directory, true)
        .expect_err("explicit seed must match the recorded scope");
    assert!(error.to_string().contains("compatibility scope"), "{error}");
    assert_eq!(checkpoint_digests(&directory), before);
    // Naming the recorded seed explicitly is the supported way to resume, which
    // proves the rejection above is about the value and not about passing --seed.
    let mut recorded = PROFILE.to_vec();
    let recorded_seed = options.seed.to_string();
    recorded.extend(["--seed", recorded_seed.as_str()]);
    let stored = crate::cli::annealed_resume_settings_for_test(&directory, &recorded)
        .expect("recorded seed");
    assert_eq!(stored.seed, options.seed);
    assert_eq!(
        run_with(stored, harness(), &directory, true)
            .expect("explicit recorded seed resumes")
            .completed_updates,
        2
    );
    std::fs::remove_dir_all(directory).expect("remove own checkpoint");
}
