//! Counter budgets and run-scope dimensions, proved before and after a real update.

use super::*;

#[test]
fn shuffle_sample_and_optimizer_preflight_accept_max_updates_and_reject_max_plus_one() {
    let capacity = |config: &AnnealedJobConfig| config.ppo.rollout_capacity(config.slots) as u64;
    for (epochs, minibatch, field) in [
        (4, 2, "annealed shuffle RNG counter"),
        (1, 2, "annealed sample counter"),
        (2, 1, "annealed optimizer counter"),
    ] {
        let mut config = settings(9001, 1);
        config.ppo.epochs = epochs;
        config.ppo.minibatch = minibatch;
        let per_update = match field {
            "annealed shuffle RNG counter" => (capacity(&config) - 1) * epochs as u64,
            "annealed sample counter" => capacity(&config),
            _ => capacity(&config) * epochs as u64,
        };
        let limit = if field == "annealed optimizer counter" {
            MODEL_MAX_OPTIMIZER_STEP
        } else {
            MAX_TRAINING_COUNTER
        };
        config.updates = limit / per_update;
        config.zero_updates = 0;
        assert_eq!(
            validate_annealed(&config, harness()).expect("maximum updates"),
            config.ppo
        );
        config.updates += 1;
        assert_eq!(
            validate_annealed(&config, harness()),
            Err(PpoError::InvalidConfig(field))
        );
    }
    assert_eq!(
        validate_counter_budget(u64::MAX, 2, MAX_TRAINING_COUNTER, "annealed sample counter"),
        Err(PpoError::InvalidConfig("annealed sample counter"))
    );
}

#[test]
fn resume_rejects_changed_collection_dimensions_without_committing() {
    let directory = test_directory("capacity-dimensions");
    let config = settings(9002, 2);
    let first = run_with(
        config.clone(),
        AnnealedHarness {
            stop_after: Some(1),
            ..harness()
        },
        &directory,
        false,
    )
    .expect("first update");
    assert_eq!(first.completed_updates, 1);
    let before = checkpoint_digests(&directory);
    type ConfigChange = fn(&mut AnnealedJobConfig);
    let changes: [(ConfigChange, &str); 7] = [
        (
            |config| config.slots = 4,
            "--slots: recorded 2, requested 4",
        ),
        (
            |config| config.lanes = 1,
            "--lanes: recorded 2, requested 1",
        ),
        (
            |config| config.ppo.samples_per_update = 8,
            "--samples-per-update: recorded 6, requested 8",
        ),
        (
            |config| config.generation_updates = 2,
            "--generation-updates: recorded 1, requested 2",
        ),
        (
            |config| config.updates += 1,
            "--updates: recorded 2, requested 3",
        ),
        (|config| config.seed += 1, "compatibility scope"),
        (
            |config| config.ppo.clip_epsilon = 0.1,
            "training checkpoint PPO config",
        ),
    ];
    for (change, message) in changes {
        let mut changed = config.clone();
        change(&mut changed);
        let error = run(changed, &directory, true).expect_err("changed training identity");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(checkpoint_digests(&directory), before);
    }
    // Simulation threads never change results, so they are not part of the scope.
    let mut threads = config;
    threads.simulation_threads = 1;
    assert_eq!(
        run(threads, &directory, true)
            .expect("any thread count resumes")
            .completed_updates,
        2
    );
}
