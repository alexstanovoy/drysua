use super::*;
use crate::{PPO_MAX_SAMPLES, PpoError};

#[test]
fn rollout_capacity_bounds_include_peak_memory_and_reject_overflow() {
    const {
        assert!(crate::PPO_STORAGE_PEAK_BYTES > crate::feature::FEATURE_ARENA_PEAK_BYTES);
        assert!(crate::PPO_STORAGE_PEAK_BYTES < 10 * 1024 * 1024 * 1024);
    }
    assert!(PpoRollout::new(PPO_MAX_SAMPLES).is_ok());
    for capacity in [0, PPO_MAX_SAMPLES + 1] {
        assert_eq!(
            PpoRollout::new(capacity)
                .err()
                .expect("capacity")
                .to_string(),
            format!("PPO rollout capacity {capacity} is outside 1..=33280")
        );
    }
    let config = annealed_config();
    config.validate().expect("maximum samples");
    assert_eq!(
        PpoConfig {
            samples_per_update: config.samples_per_update + 1,
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
    let capacity = config.rollout_capacity(crate::PPO_MAX_SLOTS);
    let mut rollout = PpoRollout::new(capacity).expect("rollout");
    for index in 0..capacity {
        sample.stream = index % crate::PPO_MAX_STREAMS;
        sample.decision = (index / crate::PPO_MAX_STREAMS) as u32;
        rollout.push(sample.clone()).expect("bounded sample");
    }
    assert_eq!(rollout.len(), capacity);
    assert_eq!(
        rollout.push(sample),
        Err(PpoError::RolloutFull { capacity })
    );
    assert_eq!(rollout.finish(config).expect("full batch").len(), capacity);
}

#[test]
fn behaviour_policy_from_the_future_is_rejected_without_mutating_trainer_state() {
    let model = PolicyModel::fresh(722).expect("model");
    let mut sample = transition(&model);
    sample.behaviour = 1;
    let mut rollout = PpoRollout::new(1).expect("rollout");
    rollout.push(sample).expect("push");
    let batch = rollout.finish(annealed_config()).expect("batch");
    let mut trainer = PpoTrainer::new(&model, annealed_config(), 1).expect("trainer");
    let before = trainer.checkpoint_snapshot(&model).expect("before");
    let random = trainer.rng_checkpoint();
    assert_eq!(
        trainer.train_update(&model, &batch, crate::UpdateObjective::default()),
        Err(PpoError::PolicyMismatch),
        "behaviour weights from the learner's future"
    );
    assert_eq!(trainer.checkpoint_snapshot(&model).expect("after"), before);
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.updates(), 0);
}

fn annealed_config() -> PpoConfig {
    PpoConfig {
        samples_per_update: crate::PPO_MAX_UPDATE_SAMPLES,
        gamma_tick: 1.0,
        epochs: 1,
        ..PpoConfig::default()
    }
}

fn transition(model: &PolicyModel) -> crate::PpoTransition {
    let (frame, space) = frame_and_space();
    choice(model, &frame, &space, StructuredAction::Continue)
        .finish(
            0,
            PpoOutcome {
                stream: 0,
                decision: 0,
                ticks: 3,
                next_value: 0.0,
                reward: 1.0,
                terminal: true,
            },
        )
        .expect("transition")
}
