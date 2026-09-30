use super::*;
use crate::{PPO_ANNEALED_MAX_SAMPLES, PPO_MAX_SAMPLES, PpoError, PpoSampleBudget};

#[test]
fn wide_capacity_enforces_profile_and_memory_boundaries_without_relabeling_m40() {
    let model = PolicyModel::fresh(726).expect("model");
    let policy = model.policy_identity().expect("identity");
    let config = PpoConfig {
        sample_budget: PpoSampleBudget::WideAnnealed,
        environments: 80,
        ..annealed_config()
    };
    assert_eq!(config.validate(), Ok(config));
    for environments in [81, 82] {
        assert_eq!(
            PpoConfig {
                environments,
                ..config
            }
            .validate(),
            Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
        );
    }
    assert_eq!(
        PpoConfig {
            sample_budget: PpoSampleBudget::Annealed,
            ..config
        }
        .validate(),
        Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
    );
    let rollout =
        PpoRollout::with_budget(93_040, policy, config.sample_budget).expect("wide maximum");
    assert!(rollout.is_empty());
    assert_eq!(
        PpoRollout::with_budget(93_041, policy, config.sample_budget)
            .err()
            .expect("overflow")
            .to_string(),
        "PPO wide annealed rollout capacity 93041 is outside 1..=93040"
    );
    let mut rollout = PpoRollout::with_budget(1, policy, config.sample_budget).expect("sparse");
    let mut sample = transition(&model);
    sample.stream = 80;
    assert_eq!(
        rollout.push(sample.clone()),
        Err(PpoError::StreamOutOfRange { stream: 80 })
    );
    sample.stream = 79;
    rollout.push(sample).expect("last valid stream");
    assert_eq!(
        rollout.finish(annealed_config()).err(),
        Some(PpoError::InvalidConfig("rollout sample budget"))
    );
    let bound = crate::PPO_WIDE_ANNEALED_PAYLOAD_BOUND_BYTES;
    assert_eq!(bound, 12_557_484_608);
    assert_eq!(PpoSampleBudget::Annealed.schema_version(), 38);
    assert_eq!(PpoSampleBudget::Annealed.max_samples(), 46_520);
    assert_eq!(PpoSampleBudget::WideAnnealed.schema_version(), 39);
}

#[test]
fn rollout_capacity_bounds_include_annealed_peak_memory_and_reject_overflow() {
    const {
        assert!(
            crate::PPO_ANNEALED_STORAGE_PEAK_BYTES
                > crate::feature::ANNEALED_FEATURE_ARENA_PEAK_BYTES
        );
        assert!(crate::PPO_ANNEALED_STORAGE_PEAK_BYTES < 6 * 1024 * 1024 * 1024);
    }
    let model = PolicyModel::fresh(718).expect("model");
    let policy = model.policy_identity().expect("identity");
    for (budget, maximum) in [
        (PpoSampleBudget::Standard, PPO_MAX_SAMPLES),
        (PpoSampleBudget::Annealed, PPO_ANNEALED_MAX_SAMPLES),
    ] {
        assert!(PpoRollout::with_budget(maximum, policy, budget).is_ok());
        for capacity in [0, maximum + 1] {
            assert!(PpoRollout::with_budget(capacity, policy, budget).is_err());
        }
    }
    let config = PpoConfig {
        environments: PPO_MAX_SAMPLES / crate::PPO_MAX_ROLLOUT_DECISIONS,
        rollout_decisions: crate::PPO_MAX_ROLLOUT_DECISIONS,
        minibatch: 8192,
        ..PpoConfig::default()
    };
    config.validate().expect("maximum standard product");
    assert_eq!(
        PpoConfig {
            environments: config.environments + 1,
            ..config
        }
        .validate(),
        Err(PpoError::InvalidConfig("samples per update"))
    );
}

#[test]
fn annealed_rollout_retains_every_bounded_sample_and_rejects_the_next() {
    let model = PolicyModel::fresh(719).expect("model");
    let mut sample = transition(&model);
    let config = annealed_config();
    let rollout_decisions =
        u32::try_from(config.rollout_decisions).expect("annealed decision bound fits u32");
    let mut rollout = PpoRollout::for_config(config, sample.policy).expect("rollout");
    sample.stream = config.environments;
    assert_eq!(
        rollout.push(sample.clone()),
        Err(PpoError::StreamOutOfRange {
            stream: config.environments
        })
    );
    assert!(rollout.is_empty());
    for stream in 0..config.environments {
        sample.stream = stream;
        for decision in 0..rollout_decisions {
            sample.decision = decision;
            rollout.push(sample.clone()).expect("bounded sample");
        }
        if stream == 0 {
            sample.decision = rollout_decisions;
            assert_eq!(
                rollout.push(sample.clone()),
                Err(PpoError::InvalidTransition("annealed retained decisions"))
            );
        }
    }
    assert_eq!(rollout.len(), PPO_ANNEALED_MAX_SAMPLES);
    assert_eq!(
        rollout.push(sample),
        Err(PpoError::RolloutFull {
            capacity: PPO_ANNEALED_MAX_SAMPLES
        })
    );
    assert_eq!(
        rollout.finish(config).expect("full batch").len(),
        PPO_ANNEALED_MAX_SAMPLES
    );
}

#[test]
fn capacity_mismatch_cannot_relabel_rollouts_or_mutate_trainer_state() {
    for (config, field) in [
        (smoke_config(), "batch sample budget"),
        (
            PpoConfig {
                environments: 2,
                ..annealed_config()
            },
            "annealed batch dimensions",
        ),
    ] {
        let model = PolicyModel::fresh(722).expect("model");
        let sample = transition(&model);
        let mut rollout =
            PpoRollout::with_budget(1, sample.policy, PpoSampleBudget::Annealed).expect("rollout");
        rollout.push(sample).expect("push");
        let batch = rollout
            .finish(annealed_config())
            .expect("underfilled batch");
        let mut trainer = PpoTrainer::new(&model, config, 1).expect("trainer");
        let before = trainer.checkpoint_snapshot(&model).expect("before");
        let random = trainer.rng_checkpoint();
        assert_eq!(
            trainer.train_update(&model, &batch),
            Err(PpoError::InvalidConfig(field))
        );
        assert_eq!(trainer.checkpoint_snapshot(&model).expect("after"), before);
        assert_eq!(trainer.rng_checkpoint(), random);
        assert_eq!(trainer.updates(), 0);
    }
    let model = PolicyModel::fresh(721).expect("model");
    let sample = transition(&model);
    for (budget, config) in [
        (PpoSampleBudget::Standard, annealed_config()),
        (PpoSampleBudget::Annealed, smoke_config()),
    ] {
        let mut rollout = PpoRollout::with_budget(1, sample.policy, budget).expect("rollout");
        rollout.push(sample.clone()).expect("push");
        assert_eq!(
            rollout.finish(config).err(),
            Some(PpoError::InvalidConfig("rollout sample budget"))
        );
    }
}

fn annealed_config() -> PpoConfig {
    PpoConfig {
        sample_budget: PpoSampleBudget::Annealed,
        environments: 40,
        rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
        gamma_tick: 1.0,
        epochs: 1,
        ..PpoConfig::default()
    }
}

fn transition(model: &PolicyModel) -> crate::PpoTransition {
    let (frame, space) = frame_and_space();
    choice(model, &frame, &space, StructuredAction::Continue)
        .finish(PpoOutcome {
            stream: 0,
            decision: 0,
            ticks: 3,
            next_value: 0.0,
            reward: 1.0,
            terminal: true,
        })
        .expect("transition")
}
