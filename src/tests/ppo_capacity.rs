use super::*;
use crate::{PPO_MAX_SAMPLES, PpoError};

#[test]
fn rollout_capacity_bounds_include_peak_memory_and_reject_overflow() {
    const {
        assert!(crate::PPO_STORAGE_PEAK_BYTES > crate::feature::FEATURE_ARENA_PEAK_BYTES);
        assert!(crate::PPO_STORAGE_PEAK_BYTES < 6 * 1024 * 1024 * 1024);
    }
    let model = PolicyModel::fresh(718).expect("model");
    let policy = model.policy_identity().expect("identity");
    assert!(PpoRollout::new(PPO_MAX_SAMPLES, policy).is_ok());
    for capacity in [0, PPO_MAX_SAMPLES + 1] {
        assert_eq!(
            PpoRollout::new(capacity, policy)
                .err()
                .expect("capacity")
                .to_string(),
            format!("PPO rollout capacity {capacity} is outside 1..=46520")
        );
    }
    let config = annealed_config();
    config.validate().expect("maximum games");
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
fn full_rollout_retains_every_bounded_sample_and_rejects_the_next() {
    let model = PolicyModel::fresh(719).expect("model");
    let mut sample = transition(&model);
    let config = annealed_config();
    let rollout_decisions =
        u32::try_from(config.rollout_decisions).expect("decision bound fits u32");
    let mut rollout = PpoRollout::for_config(config, sample.policy).expect("rollout");
    for stream in 0..config.environments {
        sample.stream = stream;
        for decision in 0..rollout_decisions {
            sample.decision = decision;
            rollout.push(sample.clone()).expect("bounded sample");
        }
    }
    assert_eq!(rollout.len(), PPO_MAX_SAMPLES);
    assert_eq!(
        rollout.push(sample),
        Err(PpoError::RolloutFull {
            capacity: PPO_MAX_SAMPLES
        })
    );
    assert_eq!(
        rollout.finish(config).expect("full batch").len(),
        PPO_MAX_SAMPLES
    );
}

#[test]
fn rollouts_outside_configured_streams_are_rejected_at_finish() {
    let model = PolicyModel::fresh(720).expect("model");
    let mut sample = transition(&model);
    let config = PpoConfig {
        environments: 2,
        ..annealed_config()
    };
    let mut rollout = PpoRollout::new(1, sample.policy).expect("rollout");
    sample.stream = config.environments;
    rollout.push(sample).expect("stream checked at finish");
    assert_eq!(
        rollout.finish(config).err(),
        Some(PpoError::InvalidConfig("rollout dimensions"))
    );
}

#[test]
fn dimension_mismatch_cannot_mutate_trainer_state() {
    let model = PolicyModel::fresh(722).expect("model");
    let sample = transition(&model);
    let mut rollout = PpoRollout::new(1, sample.policy).expect("rollout");
    rollout.push(sample).expect("push");
    let batch = rollout
        .finish(annealed_config())
        .expect("underfilled batch");
    let config = PpoConfig {
        environments: 2,
        ..annealed_config()
    };
    let mut trainer = PpoTrainer::new(&model, config, 1).expect("trainer");
    let before = trainer.checkpoint_snapshot(&model).expect("before");
    let random = trainer.rng_checkpoint();
    assert_eq!(
        trainer.train_update(&model, &batch),
        Err(PpoError::InvalidConfig("batch dimensions"))
    );
    assert_eq!(trainer.checkpoint_snapshot(&model).expect("after"), before);
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.updates(), 0);
}

fn annealed_config() -> PpoConfig {
    PpoConfig {
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
