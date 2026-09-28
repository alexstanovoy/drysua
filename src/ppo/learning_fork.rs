#![cfg(test)]

use super::{AdamState, PolicyModel, PpoConfig, PpoError, PpoTrainer};

impl PpoTrainer {
    /// Changes research learning settings without rebinding or modifying the model.
    /// A reset clears only Adam's moments and step; actor-sampling RNG belongs to the caller.
    pub(crate) fn fork_learning(
        &mut self,
        model: &PolicyModel,
        config: PpoConfig,
        reset_adam: bool,
    ) -> Result<(), PpoError> {
        let config = config.validate()?;
        if config.sample_budget != self.config.sample_budget {
            return Err(PpoError::InvalidConfig("learning fork sample budget"));
        }
        if config.environments != self.config.environments {
            return Err(PpoError::InvalidConfig("learning fork environments"));
        }
        if config.decision_interval_ticks != self.config.decision_interval_ticks {
            return Err(PpoError::InvalidConfig("learning fork decision interval"));
        }
        if config.rollout_decisions != self.config.rollout_decisions {
            return Err(PpoError::InvalidConfig("learning fork rollout decisions"));
        }
        if config.adam_beta1.to_bits() != self.config.adam_beta1.to_bits() {
            return Err(PpoError::InvalidConfig("learning fork Adam beta1"));
        }
        if config.adam_beta2.to_bits() != self.config.adam_beta2.to_bits() {
            return Err(PpoError::InvalidConfig("learning fork Adam beta2"));
        }
        if config.adam_epsilon.to_bits() != self.config.adam_epsilon.to_bits() {
            return Err(PpoError::InvalidConfig("learning fork Adam epsilon"));
        }
        self.execution.validate_ppo_memory(config.sample_budget)?;
        let snapshot = self.checkpoint_snapshot(model)?;
        let (first_moment, second_moment) = snapshot.adam.moments();
        let (first_moment, second_moment, step) = if reset_adam {
            (
                vec![0.0; first_moment.len()],
                vec![0.0; second_moment.len()],
                0,
            )
        } else {
            (
                first_moment.to_vec(),
                second_moment.to_vec(),
                snapshot.adam.step(),
            )
        };
        let adam = AdamState::from_parts(
            config.adam(),
            first_moment,
            second_moment,
            step,
            snapshot.adam.binding(),
        )
        .map_err(|error| PpoError::Model(error.to_string()))?;
        debug_assert_eq!(adam.binding(), self.adam.binding());
        debug_assert_eq!(adam.config(), config.adam());
        self.adam = adam;
        self.config = config;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MODEL_PARAMETER_COUNT, PpoRng, PpoSampleBudget, TrainingArtifact};

    type ConfigRejection = (fn(&mut PpoConfig), &'static str);

    #[test]
    fn fork_learning_preserves_model_binding_progress_and_shuffle_with_optional_adam_reset() {
        for reset_adam in [false, true] {
            let (model, mut trainer) = seeded_trainer();
            let before = trainer.checkpoint_snapshot(&model).expect("snapshot");
            let shuffle = trainer.shuffle.clone();
            let execution = trainer.execution;
            let config = changed_config(trainer.config());

            trainer
                .fork_learning(&model, config, reset_adam)
                .expect("fork");

            let after = trainer.checkpoint_snapshot(&model).expect("same binding");
            assert_bits_equal(&after.parameters, &before.parameters);
            assert_eq!(after.adam.binding(), before.adam.binding());
            assert_eq!(
                model.policy_identity().expect("identity"),
                before.adam.policy_identity()
            );
            assert_eq!(trainer.config(), config);
            assert_eq!(after.adam.config(), config.adam());
            assert_eq!(trainer.updates(), 10);
            assert_eq!(trainer.execution, execution);
            assert_eq!(trainer.shuffle, shuffle);
            assert_eq!(trainer.optimizer_step(), if reset_adam { 0 } else { 7 });
            for (actual, original) in [
                (after.adam.moments().0, before.adam.moments().0),
                (after.adam.moments().1, before.adam.moments().1),
            ] {
                assert!(original.iter().any(|value| *value != 0.0));
                if reset_adam {
                    assert!(actual.iter().all(|value| value.to_bits() == 0));
                } else {
                    assert_bits_equal(actual, original);
                }
            }
            assert_shuffle_continues(&mut trainer.shuffle, shuffle);
        }
    }

    #[test]
    fn fork_learning_rejects_invalid_config_execution_and_binding_without_mutation() {
        let (model, mut trainer) = seeded_trainer();
        for reset_adam in [false, true] {
            for (change, field) in rejected_config_changes() {
                let mut config = changed_config(trainer.config());
                change(&mut config);
                assert_rejected(
                    &model,
                    &mut trainer,
                    config,
                    reset_adam,
                    PpoError::InvalidConfig(field),
                );
            }
            let config = changed_config(trainer.config());
            trainer.execution.training_microbatch = 63;
            assert_rejected(
                &model,
                &mut trainer,
                config,
                reset_adam,
                PpoError::InvalidConfig("training microbatch must be 64, 128 or 256"),
            );
            trainer.execution.training_microbatch = 64;
        }
        let other = PolicyModel::fresh(72).expect("unrelated CPU model");
        let config = changed_config(trainer.config());
        for reset_adam in [false, true] {
            assert_rejected(
                &other,
                &mut trainer,
                config,
                reset_adam,
                PpoError::Model(
                    "model optimizer owner or parameter revision does not match".to_owned(),
                ),
            );
        }
        trainer
            .checkpoint_snapshot(&model)
            .expect("original ownership survives");
        model
            .import_parameters(&model.export_parameters().expect("parameters"))
            .expect("invalidate optimizer revision");
        for reset_adam in [false, true] {
            assert_rejected(
                &model,
                &mut trainer,
                config,
                reset_adam,
                PpoError::Model(
                    "model optimizer owner or parameter revision does not match".to_owned(),
                ),
            );
        }
    }

    #[test]
    fn fork_learning_artifact_capture_restore_preserves_changed_config_and_optimizer_bits() {
        let (model, mut trainer) = seeded_trainer();
        let config = changed_config(trainer.config());
        let (run, progress) = checkpoint_metadata(config);
        for reset_adam in [false, true] {
            let target = PolicyModel::fresh(73).expect("restore CPU model");
            trainer
                .fork_learning(&model, config, reset_adam)
                .expect("fork");
            let expected = trainer.checkpoint_snapshot(&model).expect("fork snapshot");

            let artifact =
                TrainingArtifact::capture(&model, &trainer, run.clone(), progress.clone())
                    .expect("capture fork");
            let mut restored = artifact.restore(&target, &run).expect("restore fork");

            assert_eq!(artifact.config(), config);
            assert_eq!(restored.trainer().config(), config);
            assert_eq!(restored.run(), &run);
            assert_eq!(restored.progress(), &progress);
            assert_eq!(restored.trainer().updates(), 10);
            assert_eq!(
                restored.trainer().rng_checkpoint(),
                trainer.rng_checkpoint()
            );
            let actual = restored
                .trainer()
                .checkpoint_snapshot(&target)
                .expect("restored binding");
            assert_eq!(actual.adam.config(), config.adam());
            assert_eq!(actual.adam.step(), if reset_adam { 0 } else { 7 });
            assert_bits_equal(&actual.parameters, &expected.parameters);
            assert_bits_equal(actual.adam.moments().0, expected.adam.moments().0);
            assert_bits_equal(actual.adam.moments().1, expected.adam.moments().1);
            assert_shuffle_continues(&mut restored.trainer_mut().shuffle, trainer.shuffle.clone());
        }
    }

    fn seeded_trainer() -> (PolicyModel, PpoTrainer) {
        let model = PolicyModel::fresh(71).expect("CPU model");
        let mut parameters = model.export_parameters().expect("parameters");
        parameters[0] = -0.0;
        model
            .import_parameters(&parameters)
            .expect("nonzero revision");
        let config = PpoConfig {
            environments: 2,
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            epochs: 1,
            minibatch: 4,
            gamma_tick: 0.75,
            ..PpoConfig::default()
        };
        let mut trainer = PpoTrainer::new(&model, config, 91).expect("trainer");
        // Seed optimizer history directly so these contracts need no autograd or simulator.
        trainer.adam = AdamState::from_parts(
            config.adam(),
            (0..MODEL_PARAMETER_COUNT)
                .map(|index| [-0.0, -0.25, 0.125, 0.5][index % 4])
                .collect(),
            (0..MODEL_PARAMETER_COUNT)
                .map(|index| [0.5, -0.0, 0.25, 0.125][index % 4])
                .collect(),
            7,
            trainer.adam.binding(),
        )
        .expect("nonzero moments");
        trainer.updates = 10;
        trainer
            .shuffle
            .shuffle(&mut [0, 1, 2, 3, 4, 5, 6, 7])
            .expect("consume shuffle");
        trainer
            .set_execution(crate::TrainingExecutionOptions {
                balanced_minibatches: true,
                host_math_workers: 2,
                ..Default::default()
            })
            .expect("execution");
        assert_eq!(trainer.optimizer_step(), 7);
        assert_eq!(trainer.shuffle.draws(), 7);
        (model, trainer)
    }

    fn changed_config(config: PpoConfig) -> PpoConfig {
        PpoConfig {
            learning_rate: 1.0e-5,
            gamma_tick: 1.0,
            gae_lambda: 0.9,
            entropy_coefficient: 0.02,
            epochs: 2,
            minibatch: 8,
            target_kl: 0.03,
            clip_epsilon: 0.15,
            value_coefficient: 0.75,
            gradient_clip: 0.25,
            ..config
        }
    }

    fn rejected_config_changes() -> [ConfigRejection; 12] {
        [
            (
                |config| config.environments = 4,
                "learning fork environments",
            ),
            (
                |config| config.rollout_decisions += 1,
                "learning fork rollout decisions",
            ),
            (
                |config| config.decision_interval_ticks = 4,
                "learning fork decision interval",
            ),
            (
                |config| config.sample_budget = PpoSampleBudget::Annealed,
                "learning fork sample budget",
            ),
            (
                |config| config.sample_budget = PpoSampleBudget::WideAnnealed,
                "learning fork sample budget",
            ),
            (|config| config.adam_beta1 = 0.8, "learning fork Adam beta1"),
            (
                |config| config.adam_beta2 = 0.99,
                "learning fork Adam beta2",
            ),
            (
                |config| config.adam_epsilon = 1.0e-4,
                "learning fork Adam epsilon",
            ),
            (|config| config.learning_rate = f32::NAN, "learning rate"),
            (|config| config.minibatch = 0, "minibatch"),
            (|config| config.gamma_tick = 2.0, "discount"),
            (|config| config.epochs = 17, "epochs"),
        ]
    }

    fn assert_bits_equal(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }

    fn assert_shuffle_continues(actual: &mut PpoRng, mut expected: PpoRng) {
        let mut actual_order = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut expected_order = actual_order;
        actual.shuffle(&mut actual_order).expect("next shuffle");
        expected
            .shuffle(&mut expected_order)
            .expect("reference shuffle");
        assert_eq!(actual_order, expected_order);
        assert_eq!(*actual, expected);
    }

    fn assert_rejected(
        model: &PolicyModel,
        trainer: &mut PpoTrainer,
        config: PpoConfig,
        reset_adam: bool,
        expected: PpoError,
    ) {
        let adam = trainer.adam.clone();
        let previous_config = trainer.config();
        let shuffle = trainer.shuffle.clone();
        let execution = trainer.execution;
        let parameters = model.export_parameters().expect("before parameters");
        let identity = model.policy_identity().expect("before identity");

        let error = trainer
            .fork_learning(model, config, reset_adam)
            .expect_err("reject fork");

        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(error, expected);
        assert_eq!(trainer.config(), previous_config);
        assert_eq!(trainer.adam, adam);
        assert_bits_equal(trainer.adam.moments().0, adam.moments().0);
        assert_bits_equal(trainer.adam.moments().1, adam.moments().1);
        assert_eq!(trainer.shuffle, shuffle);
        assert_eq!(trainer.execution, execution);
        assert_eq!(trainer.updates(), 10);
        assert_eq!(model.policy_identity().expect("after identity"), identity);
        assert_bits_equal(
            &model.export_parameters().expect("after parameters"),
            &parameters,
        );
    }

    fn checkpoint_metadata(config: PpoConfig) -> (crate::CheckpointRun, crate::CheckpointProgress) {
        let run = crate::CheckpointRun {
            mastery_config: None,
            git_commit: "learning-fork-test".to_owned(),
            simulator_commit: "learning-fork-test".to_owned(),
            enabled_features: crate::compiled_features(),
            command_line: "learning-fork CPU capture/restore contract".to_owned(),
            run_seed: 91,
            map: bota_proto::MapId(2),
            hero: crate::SHADOW_FIEND,
            device: crate::CheckpointDevice::Cpu,
            batch_size: config.minibatch,
            rules_audit_version: crate::PPO_RULES_AUDIT_VERSION,
        };
        let progress = crate::CheckpointProgress {
            adaptive_environment: None,
            mastery: None,
            global_update: 10,
            policy_version: 10,
            scheduler_step: 10,
            curriculum_stage: 0,
            rollout_samples: 80,
            best_evaluation: None,
            rng_states: vec![
                crate::RngCheckpoint::new("actor-sampling", 123, 17).expect("external RNG"),
            ],
            league_references: Vec::new(),
        };
        (run, progress)
    }
}
