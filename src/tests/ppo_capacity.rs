use super::*;
use crate::{PPO_ANNEALED_MAX_SAMPLES, PPO_MAX_SAMPLES, PpoError, PpoSampleBudget};

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

#[test]
fn annealed_rollout_storage_bound_includes_dense_minibatch_and_reallocation() {
    const {
        assert!(
            crate::PPO_ANNEALED_STORAGE_PEAK_BYTES
                > crate::feature::ANNEALED_FEATURE_ARENA_PEAK_BYTES
        );
        assert!(crate::PPO_ANNEALED_STORAGE_PEAK_BYTES < 6 * 1024 * 1024 * 1024);
    }
    println!(
        "annealed rollout storage bound: {} bytes",
        crate::PPO_ANNEALED_STORAGE_PEAK_BYTES
    );
}

#[cfg(feature = "builtin")]
#[test]
fn train_full_rejects_annealed_profile_even_with_two_games() {
    let mut settings = crate::cli::training_settings_for_test(&[
        "--environments",
        "2",
        "--rollout",
        "1163",
        "--minibatch",
        "512",
    ])
    .expect("settings");
    settings.ppo.sample_budget = PpoSampleBudget::Annealed;
    let directory = std::path::Path::new("unused-annealed-profile-rejection");
    let error = crate::run_training_job_on_with_initial_weights(
        settings,
        crate::PolicyDevice::Cpu,
        directory,
        false,
        None,
        |_| {},
    )
    .expect_err("train-full cannot opt into annealed");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: train-full sample budget"
    );
    assert!(!directory.exists());
}

#[test]
fn capacity_profiles_preserve_standard_identity_and_bound_full_episode_dimensions() {
    assert_eq!(PPO_SCHEMA_HASH, 0xb18a_050a_dd4a_85cd);
    assert_eq!(
        PpoConfig::default().sample_budget,
        PpoSampleBudget::Standard
    );
    assert_eq!(PPO_MAX_SAMPLES, 32_768);
    assert_eq!(PPO_ANNEALED_MAX_SAMPLES, 46_520);
    assert_eq!(annealed_config().validate(), Ok(annealed_config()));
    for environments in [0, 1, 39, 41, 42, usize::MAX] {
        assert_eq!(
            PpoConfig {
                environments,
                ..annealed_config()
            }
            .validate(),
            Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
        );
    }
    for rollout_decisions in [1162, 1164] {
        assert_eq!(
            PpoConfig {
                rollout_decisions,
                ..annealed_config()
            }
            .validate(),
            Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
        );
    }
    assert_eq!(
        PpoConfig {
            gamma_tick: 0.99,
            ..annealed_config()
        }
        .validate(),
        Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
    );
    assert_eq!(
        PpoConfig {
            decision_interval_ticks: 4,
            ..annealed_config()
        }
        .validate(),
        Err(PpoError::InvalidConfig("annealed full-episode dimensions"))
    );
    assert_eq!(
        PpoConfig {
            sample_budget: PpoSampleBudget::Standard,
            ..annealed_config()
        }
        .validate(),
        Err(PpoError::InvalidConfig("samples per update"))
    );
}

#[test]
fn rollout_constructors_accept_profile_maximum_and_reject_maximum_plus_one() {
    let model = PolicyModel::fresh(718).expect("model");
    let policy = model.policy_identity().expect("identity");
    assert!(PpoRollout::new(PPO_MAX_SAMPLES, policy).is_ok());
    assert_eq!(
        PpoRollout::new(PPO_MAX_SAMPLES + 1, policy).err(),
        Some(PpoError::Capacity { capacity: 32_769 })
    );
    assert!(PpoRollout::for_config(annealed_config(), policy).is_ok());
    for capacity in [0, PPO_ANNEALED_MAX_SAMPLES + 1] {
        assert_eq!(
            PpoRollout::with_budget(capacity, policy, PpoSampleBudget::Annealed)
                .err()
                .expect("invalid capacity")
                .to_string(),
            format!("PPO annealed rollout capacity {capacity} is outside 1..=46520")
        );
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

#[test]
fn annealed_rollout_retains_all_46520_samples_and_rejects_the_next() {
    let model = PolicyModel::fresh(719).expect("model");
    let mut sample = transition(&model);
    let mut rollout = PpoRollout::for_config(annealed_config(), sample.policy).expect("rollout");
    for stream in 0..40 {
        sample.stream = stream;
        for decision in 0..1163 {
            sample.decision = decision;
            rollout.push(sample.clone()).expect("bounded sample");
        }
    }
    assert_eq!(rollout.len(), 46_520);
    assert_eq!(
        rollout.push(sample),
        Err(PpoError::RolloutFull { capacity: 46_520 })
    );
    let batch = rollout.finish(annealed_config()).expect("full batch");
    assert_eq!(batch.len(), 46_520);
}

#[test]
fn annealed_rollout_rejects_stream_and_per_episode_overflow() {
    let model = PolicyModel::fresh(720).expect("model");
    let mut sample = transition(&model);
    let mut rollout = PpoRollout::for_config(annealed_config(), sample.policy).expect("rollout");
    sample.stream = 40;
    assert_eq!(
        rollout.push(sample.clone()),
        Err(PpoError::StreamOutOfRange { stream: 40 })
    );
    sample.stream = 39;
    for decision in 0..1163 {
        sample.decision = decision;
        rollout.push(sample.clone()).expect("episode sample");
    }
    sample.decision = 1163;
    assert_eq!(
        rollout.push(sample),
        Err(PpoError::InvalidTransition("annealed retained decisions"))
    );
    assert_eq!(
        rollout
            .finish(PpoConfig {
                environments: 2,
                ..annealed_config()
            })
            .err(),
        Some(PpoError::InvalidConfig("annealed rollout dimensions"))
    );
}

#[test]
fn rollout_finish_rejects_capacity_profile_relabeling_in_both_directions() {
    let model = PolicyModel::fresh(721).expect("model");
    let sample = transition(&model);
    for budget in [PpoSampleBudget::Standard, PpoSampleBudget::Annealed] {
        let mut rollout = PpoRollout::with_budget(1, sample.policy, budget).expect("rollout");
        rollout.push(sample.clone()).expect("sample");
        let config = if budget == PpoSampleBudget::Standard {
            annealed_config()
        } else {
            smoke_config()
        };
        assert_eq!(
            rollout.finish(config).err(),
            Some(PpoError::InvalidConfig("rollout sample budget"))
        );
    }
}

#[test]
fn trainer_rejects_foreign_budget_before_mutating_parameters_adam_or_rng() {
    let model = PolicyModel::fresh(722).expect("model");
    let sample = transition(&model);
    let mut rollout =
        PpoRollout::with_budget(1, sample.policy, PpoSampleBudget::Annealed).expect("rollout");
    rollout.push(sample).expect("sample");
    let batch = rollout
        .finish(annealed_config())
        .expect("underfilled batch");
    let mut trainer = PpoTrainer::new(&model, smoke_config(), 1).expect("trainer");
    let parameters = model.export_parameters().expect("parameters");
    let random = trainer.rng_checkpoint();
    assert_eq!(
        trainer.train_update(&model, &batch),
        Err(PpoError::InvalidConfig("batch sample budget"))
    );
    assert_eq!(parameters, model.export_parameters().expect("parameters"));
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.updates(), 0);
}

#[test]
fn annealed_trainer_rejects_batch_with_different_episode_count_before_mutation() {
    let model = PolicyModel::fresh(724).expect("model");
    let sample = transition(&model);
    let mut rollout =
        PpoRollout::with_budget(1, sample.policy, PpoSampleBudget::Annealed).expect("rollout");
    rollout.push(sample).expect("sample");
    let batch = rollout.finish(annealed_config()).expect("batch");
    let mut trainer = PpoTrainer::new(
        &model,
        PpoConfig {
            environments: 2,
            ..annealed_config()
        },
        1,
    )
    .expect("trainer");
    let random = trainer.rng_checkpoint();
    assert_eq!(
        trainer.train_update(&model, &batch),
        Err(PpoError::InvalidConfig("annealed batch dimensions"))
    );
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn pipeline_keeps_standard_capacity_and_rejects_annealed_finish_and_trainer() {
    let model = PolicyModel::fresh(725).expect("model");
    assert!(crate::ActorLearnerPipeline::new(PPO_MAX_SAMPLES, 1, &model).is_ok());
    assert_eq!(
        crate::ActorLearnerPipeline::new(PPO_MAX_SAMPLES + 1, 1, &model)
            .err()
            .expect("capacity rejected")
            .to_string(),
        "actor-learner sample capacity 32769 is outside 1..=32768"
    );
    let mut pipeline = crate::ActorLearnerPipeline::new(1, 1, &model).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    for expanded_finish in [true, false] {
        let lease = actor.lease().expect("lease");
        let mut buffer = lease.rollout().expect("buffer");
        buffer.push(transition(lease.policy())).expect("sample");
        lease.try_submit(buffer).expect("submit");
        let accepted = pipeline.accept().expect("accept");
        if expanded_finish {
            assert_eq!(
                accepted.finish(annealed_config()).err(),
                Some(PpoError::InvalidConfig("rollout sample budget"))
            );
        } else {
            let batch = accepted.finish(smoke_config()).expect("standard finish");
            let mut trainer = PpoTrainer::new(&model, annealed_config(), 1).expect("trainer");
            assert_eq!(
                trainer.train_pipeline_update(&model, &batch),
                Err(PpoError::InvalidConfig("pipeline sample budget"))
            );
            assert_eq!(trainer.optimizer_step(), 0);
            assert_eq!(trainer.rng_checkpoint().1, 0);
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn rollout_append_rejects_foreign_budget_without_mutation() {
    let model = PolicyModel::fresh(723).expect("model");
    let policy = model.policy_identity().expect("identity");
    let mut standard = PpoRollout::new(1, policy).expect("standard");
    let annealed = PpoRollout::with_budget(1, policy, PpoSampleBudget::Annealed).expect("annealed");
    assert_eq!(
        standard.append(annealed),
        Err(PpoError::InvalidConfig("rollout sample budget"))
    );
    assert!(standard.is_empty());
}
