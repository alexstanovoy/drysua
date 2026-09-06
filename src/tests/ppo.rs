#![allow(
    clippy::float_arithmetic,
    reason = "PPO reference calculations use floating-point arithmetic"
)]

use bota_proto::Team;
#[cfg(feature = "builtin")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::feature::{encode, tracker_with_view, world_view};
#[cfg(feature = "builtin")]
use crate::ActionKind;
use crate::{
    ABILITY_FEATURE_TOKENS, ActionSpace, BehavioralTarget, ControlledUnit, ITEM_FEATURE_TOKENS,
    LOOT_FEATURE_TOKENS, LocalPolicyState, MODEL_TRAINING_BATCH, POINT_FEATURE_TOKENS,
    PPO_MAX_ROLLOUT_DECISIONS, PPO_RULES_AUDIT_VERSION, PPO_SCHEMA_HASH, PPO_SCHEMA_VERSION,
    PPO_SHAPING_BUDGET, PPO_TERMINAL_REWARD, PROJECTILE_FEATURE_TOKENS, PolicyModel, PpoConfig,
    PpoOutcome, PpoPolicyChoice, PpoRng, PpoRollout, PpoTerminalOutcome, PpoTrainer,
    REMEMBERED_UNIT_FEATURE_TOKENS, RewardTracker, StructuredAction, UNIT_FEATURE_TOKENS,
    clipped_surrogate, tick_discount,
};

#[test]
fn ppo_defaults_match_stage_nine_plan() {
    let config = PpoConfig::default();

    assert_eq!(config.decision_interval_ticks, 3);
    assert_eq!(config.rollout_decisions, 256);
    assert_eq!(config.environments, 32);
    assert_eq!(config.epochs, 4);
    assert_eq!(config.minibatch, 2_048);
    assert_eq!(config.clip_epsilon, 0.2);
    assert_eq!(config.value_coefficient, 0.5);
    assert_eq!(config.entropy_coefficient, 0.01);
    assert_eq!(config.gae_lambda, 0.98);
    assert_eq!(config.target_kl, 0.02);
    assert_eq!(PPO_TERMINAL_REWARD, 1.0);
}

#[cfg(feature = "builtin")]
#[test]
#[ignore = "Bounded release evidence for the v0.0.1 Teacher fallback."]
fn teacher_release_evaluation_reports_three_seeds_on_both_sides() {
    for seed in [9_000_001, 9_000_002, 9_000_003] {
        let games = crate::evaluate_teacher_against_weak_for_test(seed, 4_096)
            .expect("bounded Teacher versus Weak evaluation");

        assert_eq!(games.len(), 4);
        for game in games {
            println!("teacher_release {game:?}");
            assert_eq!(game.seed, seed);
            assert!((1..=4_096).contains(&game.decisions));
            assert_eq!(game.action_counts.iter().sum::<u32>(), game.decisions);
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn actor_report_merge_retains_all_terminal_telemetry() {
    let mut aggregate = crate::PpoSmokeReport {
        terminal_wins: 1,
        terminal_losses: 2,
        terminal_draws: 3,
        rejected_orders: 4,
        elapsed_ticks: 5,
        ..crate::PpoSmokeReport::default()
    };
    let actor = crate::PpoSmokeReport {
        terminal_wins: 6,
        terminal_losses: 7,
        terminal_draws: 8,
        rejected_orders: 9,
        elapsed_ticks: 10,
        ..crate::PpoSmokeReport::default()
    };

    crate::merge_actor_report_for_test(&mut aggregate, actor).expect("merge actor report");

    assert_eq!(aggregate.terminal_wins, 7);
    assert_eq!(aggregate.terminal_losses, 9);
    assert_eq!(aggregate.terminal_draws, 11);
    assert_eq!(aggregate.rejected_orders, 13);
    assert_eq!(aggregate.elapsed_ticks, 15);
}

#[cfg(feature = "builtin")]
#[test]
fn actor_report_rejection_delta_does_not_recount_prior_updates() {
    assert_eq!(crate::rejection_delta_for_test(1, 1).expect("no delta"), 0);
    assert_eq!(crate::rejection_delta_for_test(1, 2).expect("one delta"), 1);
    assert_eq!(
        crate::rejection_delta_for_test(2, 1)
            .expect_err("counter regression")
            .to_string(),
        "invalid PPO transition: arena rejection counter regressed"
    );
}

#[test]
fn rollout_compacts_sparse_tokens_and_bit_packs_behavioral_masks_losslessly() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(100).expect("model");
    let policy = model.policy_identity().expect("policy");
    let sampled = choice(&model, &frame, &space, StructuredAction::Continue);
    let target = sampled.target.clone();
    let packed = target.pack();
    let mut rollout = PpoRollout::new(1, policy).expect("rollout");
    rollout
        .push(
            sampled
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 3,
                    next_value: 0.0,
                    reward: 1.0,
                    terminal: true,
                })
                .expect("transition"),
        )
        .expect("push");
    let padded_rows = UNIT_FEATURE_TOKENS
        + REMEMBERED_UNIT_FEATURE_TOKENS
        + POINT_FEATURE_TOKENS
        + ABILITY_FEATURE_TOKENS
        + ITEM_FEATURE_TOKENS
        + PROJECTILE_FEATURE_TOKENS
        + LOOT_FEATURE_TOKENS;

    assert_eq!(packed.unpack(), target);
    assert!(
        std::mem::size_of_val(&packed) < std::mem::size_of::<BehavioralTarget>(),
        "packed={} fixed={}",
        std::mem::size_of_val(&packed),
        std::mem::size_of::<BehavioralTarget>()
    );
    assert!(rollout.ragged_rows_for_test() < padded_rows);
}

#[test]
fn ppo_schema_and_rules_audit_are_stable() {
    assert_eq!(PPO_SCHEMA_VERSION, 16);
    assert_eq!(PPO_RULES_AUDIT_VERSION, 15);
    assert_eq!(PPO_SCHEMA_HASH, 11_450_737_853_127_354_910);
    assert_eq!(PpoConfig::default().learning_rate, 3.0e-6);
}

#[test]
fn ppo_config_rejects_every_unbounded_dimension() {
    assert!(
        PpoConfig {
            environments: 129,
            ..PpoConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        PpoConfig {
            rollout_decisions: PPO_MAX_ROLLOUT_DECISIONS + 1,
            ..PpoConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        PpoConfig {
            epochs: 17,
            ..PpoConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        PpoConfig {
            minibatch: 8_193,
            ..PpoConfig::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn ppo_config_accepts_the_maximum_bounded_production_rollout() {
    let config = PpoConfig {
        environments: 16,
        rollout_decisions: PPO_MAX_ROLLOUT_DECISIONS,
        minibatch: 8_192,
        ..PpoConfig::default()
    };

    config.validate().expect("maximum bounded rollout");
}

#[test]
fn ppo_config_reports_a_sample_product_above_the_global_bound() {
    let error = PpoConfig {
        environments: 17,
        rollout_decisions: PPO_MAX_ROLLOUT_DECISIONS,
        minibatch: 8_192,
        ..PpoConfig::default()
    }
    .validate()
    .expect_err("sample product exceeds the global rollout buffer");

    assert_eq!(
        error.to_string(),
        "invalid PPO config field: samples per update"
    );
}

#[test]
fn discount_uses_elapsed_simulation_ticks() {
    let discount = tick_discount(0.99, 3).expect("discount");

    assert!((discount - 0.970_299).abs() < 1.0e-6);
}

#[test]
fn sampling_uniform_is_open_at_both_integer_boundaries() {
    let (minimum, maximum) = crate::ppo::open_unit_bounds_for_test();

    assert!(minimum > 0.0);
    assert!(maximum < 1.0);
    assert!(minimum.is_finite());
    assert!(maximum.is_finite());
}

#[test]
fn clipped_surrogate_uses_the_worse_boundary_for_each_advantage_sign() {
    assert!((clipped_surrogate(1.5, 2.0, 0.2) - 2.4).abs() < 1.0e-6);
    assert!((clipped_surrogate(0.5, -2.0, 0.2) - -1.6).abs() < 1.0e-6);
    assert!((clipped_surrogate(1.1, 2.0, 0.2) - 2.2).abs() < 1.0e-6);
}

#[test]
fn sampled_action_is_legal_and_statistics_match_exactly() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(91).expect("model");
    let mut rng = PpoRng::new(17);

    let choice = model.sample(&frame, &space, &mut rng).expect("sample");
    let (log_probability, entropy, value) = model
        .action_statistics(&frame, &space, choice.action)
        .expect("statistics");

    assert!(space.allows(choice.action));
    assert_eq!(choice.log_probability, log_probability);
    assert_eq!(choice.entropy, entropy);
    assert_eq!(choice.value, value);
    assert!(choice.log_probability <= 0.0);
    assert!(choice.entropy >= 0.0);
    assert!(rng.draws() > 0);
    assert!(rng.draws() <= crate::PPO_MAX_POLICY_SAMPLE_DRAWS);
}

#[test]
fn batched_sampling_matches_independent_scalar_streams_at_the_actor_batch_limit() {
    let model = PolicyModel::fresh(9_101).expect("model");
    let (frames, spaces) = policy_batch_inputs();
    let mut scalar_rngs = (0..MODEL_TRAINING_BATCH)
        .map(|index| PpoRng::new(17 + index as u64 * 97))
        .collect::<Vec<_>>();
    let mut batch_rngs = scalar_rngs.clone();
    let scalar = frames
        .iter()
        .zip(&spaces)
        .zip(&mut scalar_rngs)
        .map(|((frame, space), rng)| model.sample(frame, space, rng).expect("scalar sample"))
        .collect::<Vec<_>>();

    let batch = model
        .sample_batch(&frames, &spaces, &mut batch_rngs)
        .expect("batch sample");

    assert_eq!(batch.len(), MODEL_TRAINING_BATCH);
    assert!(scalar.iter().any(|choice| matches!(
        choice.action().kind(),
        crate::ActionKind::Cast
            | crate::ActionKind::Use
            | crate::ActionKind::PutPoint
            | crate::ActionKind::PutUnit
            | crate::ActionKind::Swap
    )));
    assert!(
        scalar_rngs
            .iter()
            .map(PpoRng::draws)
            .min()
            .expect("minimum draws")
            < scalar_rngs
                .iter()
                .map(PpoRng::draws)
                .max()
                .expect("maximum draws")
    );
    for (index, (batch, scalar)) in batch.iter().zip(&scalar).enumerate() {
        assert_eq!(batch.action(), scalar.action());
        assert_eq!(batch.policy(), scalar.policy());
        assert!((batch.log_probability() - scalar.log_probability()).abs() <= 1.0e-5);
        assert!((batch.entropy() - scalar.entropy()).abs() <= 1.0e-5);
        assert!((batch.value() - scalar.value()).abs() <= 1.0e-5);
        assert!(spaces[index].allows(batch.action()));
        assert!(spaces[index].decode(batch.action()).is_ok());
    }
    for (batch, scalar) in batch_rngs.iter().zip(&scalar_rngs) {
        assert_eq!(batch.checkpoint(), scalar.checkpoint());
    }
}

#[test]
fn batched_choice_matches_independent_scalar_rows_at_the_actor_batch_limit() {
    let model = PolicyModel::fresh(9_105).expect("model");
    let (frames, spaces) = policy_batch_inputs();
    let identity = model.policy_identity().expect("policy identity");
    let scalar = frames
        .iter()
        .zip(&spaces)
        .map(|(frame, space)| model.choose(frame, space).expect("scalar choice"))
        .collect::<Vec<_>>();

    let batch = model.choose_batch(&frames, &spaces).expect("batch choice");

    assert_eq!(batch.len(), MODEL_TRAINING_BATCH);
    for (index, (batch, scalar)) in batch.iter().zip(&scalar).enumerate() {
        assert_eq!(batch.action, scalar.action);
        assert!((batch.value - scalar.value).abs() <= 1.0e-5);
        assert!(spaces[index].allows(batch.action));
        assert!(spaces[index].decode(batch.action).is_ok());
    }
    assert_eq!(model.policy_identity().expect("stable identity"), identity);
}

#[test]
fn batched_choice_rejects_empty_mismatched_and_stale_inputs() {
    let model = PolicyModel::fresh(9_106).expect("model");
    let (frame, _) = frame_and_space();

    assert_eq!(
        model.choose_batch(&[], &[]).unwrap_err().to_string(),
        "model batch must contain at least one frame"
    );
    assert_eq!(
        model
            .choose_batch(std::slice::from_ref(&frame), &[])
            .unwrap_err()
            .to_string(),
        "model batch action-space count 0 differs from frame count 1"
    );
    let (_, stale_space) = frame_and_space();
    assert_eq!(
        model
            .choose_batch(
                std::slice::from_ref(&frame),
                std::slice::from_ref(&stale_space),
            )
            .unwrap_err()
            .to_string(),
        "model batch frame 0 does not belong to its action space"
    );
}

#[test]
fn batched_sampling_rejects_empty_and_mismatched_counts_without_rng_mutation() {
    let model = PolicyModel::fresh(9_102).expect("model");
    let (frame, space) = frame_and_space();
    let mut rngs = vec![PpoRng::new(21)];
    let before = rngs.clone();

    assert_eq!(
        model
            .sample_batch(&[], &[], &mut [])
            .unwrap_err()
            .to_string(),
        "model batch must contain at least one frame"
    );
    let oversized = vec![frame.clone(); MODEL_TRAINING_BATCH + 1];
    assert_eq!(
        model
            .sample_batch(&oversized, &[], &mut rngs)
            .unwrap_err()
            .to_string(),
        format!(
            "model batch count {} exceeds maximum {MODEL_TRAINING_BATCH}",
            MODEL_TRAINING_BATCH + 1
        )
    );
    assert_eq!(rngs, before);
    assert_eq!(
        model
            .sample_batch(std::slice::from_ref(&frame), &[], &mut rngs)
            .unwrap_err()
            .to_string(),
        "model batch action-space count 0 differs from frame count 1"
    );
    assert_eq!(rngs, before);
    let mut extra_rngs = vec![PpoRng::new(22), PpoRng::new(23)];
    let extra_before = extra_rngs.clone();
    assert_eq!(
        model
            .sample_batch(
                std::slice::from_ref(&frame),
                std::slice::from_ref(&space),
                &mut extra_rngs,
            )
            .unwrap_err()
            .to_string(),
        "model sampling RNG count 2 differs from frame count 1"
    );
    assert_eq!(extra_rngs, extra_before);
}

#[test]
fn batched_sampling_rejects_stale_provenance_without_rng_mutation() {
    let model = PolicyModel::fresh(9_103).expect("model");
    let (frame, _) = frame_and_space();
    let (_, stale_space) = frame_and_space();
    let mut rngs = vec![PpoRng::new(22)];
    let before = rngs.clone();

    assert_eq!(
        model
            .sample_batch(
                std::slice::from_ref(&frame),
                std::slice::from_ref(&stale_space),
                &mut rngs,
            )
            .unwrap_err()
            .to_string(),
        "model batch frame 0 does not belong to its action space"
    );
    assert_eq!(rngs, before);
}

#[test]
fn batched_sampling_rolls_back_rng_after_a_traversed_head_fails() {
    let model = PolicyModel::fresh(9_104).expect("model");
    let mut parameters = vec![0.0; model.parameter_count()];
    let kind = crate::ActionKind::Stop.index();
    set_policy_parameter_range(&model, &mut parameters, "kind.bias", kind..kind + 1, 100.0);
    set_policy_parameter_range(
        &model,
        &mut parameters,
        "kind_embedding.weight",
        kind * 32..(kind + 1) * 32,
        1.0,
    );
    set_policy_parameter_range(
        &model,
        &mut parameters,
        "controlled.weight",
        256 * 2..288 * 2,
        f32::MAX,
    );
    model
        .import_parameters(&parameters)
        .expect("finite parameters");
    let (frame, space) = frame_and_space();
    let mut rngs = vec![PpoRng::new(24)];
    let before = rngs.clone();

    assert_eq!(
        model
            .sample_batch(
                std::slice::from_ref(&frame),
                std::slice::from_ref(&space),
                &mut rngs,
            )
            .unwrap_err()
            .to_string(),
        "model controlled output at batch 0 index 0 is non-finite"
    );
    assert_eq!(rngs, before);
}

#[test]
fn policy_sample_rng_bound_covers_the_longest_decoder_path() {
    let longest_path = 16 + 2 + 15 + 3 + 96;

    assert_eq!(crate::PPO_MAX_POLICY_SAMPLE_DRAWS, longest_path);
}

#[test]
fn policy_ratio_is_one_before_any_update() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(92).expect("model");
    let sampled = choice(&model, &frame, &space, StructuredAction::Continue);

    let current = model
        .action_statistics(&frame, &space, sampled.action)
        .expect("current")
        .0;

    assert_eq!((current - sampled.log_probability).exp(), 1.0);
}

#[test]
fn gae_uses_tick_discount_and_resets_at_terminal_transition() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(93).expect("model");
    let policy = model.policy_identity().expect("policy");
    let mut rollout = PpoRollout::new(2, policy).expect("rollout");
    let mut first = choice(&model, &frame, &space, StructuredAction::Continue);
    first.value = 0.0;
    let mut second = choice(&model, &frame, &space, StructuredAction::Continue);
    second.value = 0.0;
    rollout
        .push(
            first
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 1,
                    next_value: 0.0,
                    reward: 1.0,
                    terminal: false,
                })
                .expect("first"),
        )
        .expect("first push");
    rollout
        .push(
            second
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 1,
                    ticks: 1,
                    next_value: 0.0,
                    reward: 2.0,
                    terminal: true,
                })
                .expect("second"),
        )
        .expect("second push");
    let config = PpoConfig {
        rollout_decisions: 2,
        environments: 1,
        minibatch: 1,
        gamma_tick: 0.9,
        gae_lambda: 0.8,
        ..PpoConfig::default()
    };

    let batch = rollout.finish(config).expect("batch");

    assert!((batch.sample(0).expect("first").return_value() - 2.44).abs() < 1.0e-5);
    assert_eq!(batch.sample(1).expect("second").return_value(), 2.0);
}

#[test]
fn synthetic_bandit_update_increases_rewarded_action_probability() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(101).expect("model");
    let policy = model.policy_identity().expect("policy");
    let before = model
        .action_statistics(&frame, &space, StructuredAction::Continue)
        .expect("before")
        .0;
    let mut rollout = PpoRollout::new(2, policy).expect("rollout");
    rollout
        .push(
            choice(&model, &frame, &space, StructuredAction::Continue)
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 3,
                    next_value: 0.0,
                    reward: 1.0,
                    terminal: true,
                })
                .expect("rewarded transition"),
        )
        .expect("rewarded sample");
    rollout
        .push(
            choice(
                &model,
                &frame,
                &space,
                StructuredAction::Hold {
                    unit: ControlledUnit::Hero,
                },
            )
            .finish(PpoOutcome {
                stream: 1,
                decision: 0,
                ticks: 3,
                next_value: 0.0,
                reward: -1.0,
                terminal: true,
            })
            .expect("penalized transition"),
        )
        .expect("penalized sample");
    let config = smoke_config();
    let batch = rollout.finish(config).expect("batch");
    let mut trainer = PpoTrainer::new(&model, config, 7).expect("trainer");

    let report = trainer.train_update(&model, &batch).expect("PPO update");
    let after = model
        .action_statistics(&frame, &space, StructuredAction::Continue)
        .expect("after")
        .0;

    assert!(after > before, "{before} -> {after}");
    assert_eq!(report.optimizer_step, 1);
    assert_eq!(report.samples_optimized, 2);
}

#[test]
fn direct_trainer_rejects_rollout_one_policy_revision_behind() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(102).expect("model");
    let policy = model.policy_identity().expect("policy");
    let mut rollout = PpoRollout::new(2, policy).expect("rollout");
    for stream in 0..2 {
        rollout
            .push(
                choice(&model, &frame, &space, StructuredAction::Continue)
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: stream as f32,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    let config = smoke_config();
    let batch = rollout.finish(config).expect("batch");
    let parameters = model.export_parameters().expect("parameters");
    model.import_parameters(&parameters).expect("new revision");
    let mut trainer = PpoTrainer::new(&model, config, 5).expect("trainer");
    let before = model.export_parameters().expect("before");
    let error = trainer
        .train_update(&model, &batch)
        .expect_err("direct stale rollout");

    assert_eq!(error.to_string(), "PPO rollout policy identity is stale");
    assert_eq!(model.export_parameters().expect("after"), before);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn rollout_two_policy_revisions_behind_is_rejected_before_optimizer_mutation() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(102).expect("model");
    let policy = model.policy_identity().expect("policy");
    let mut rollout = PpoRollout::new(2, policy).expect("rollout");
    for stream in 0..2 {
        rollout
            .push(
                choice(&model, &frame, &space, StructuredAction::Continue)
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: stream as f32,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    let config = smoke_config();
    let batch = rollout.finish(config).expect("batch");
    let parameters = model.export_parameters().expect("parameters");
    model.import_parameters(&parameters).expect("revision one");
    model.import_parameters(&parameters).expect("revision two");
    let mut trainer = PpoTrainer::new(&model, config, 5).expect("trainer");
    let before = model.export_parameters().expect("before");

    let error = trainer
        .train_update(&model, &batch)
        .expect_err("stale rollout");

    assert_eq!(error.to_string(), "PPO rollout policy identity is stale");
    assert_eq!(model.export_parameters().expect("after"), before);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[test]
fn failed_preupdate_restores_shuffle_and_allows_exact_retry() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(103).expect("model");
    let config = smoke_config();
    let mut batch = bandit_batch(&model, &frame, &space, config);
    let mut trainer = PpoTrainer::new(&model, config, 19).expect("trainer");
    let parameters = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let draws = trainer.shuffle_draws_for_test();
    let advantage = batch.replace_advantage_for_test(0, f32::NAN);

    assert!(trainer.train_update(&model, &batch).is_err());
    assert_eq!(
        model.export_parameters().expect("after failure"),
        parameters
    );
    assert_eq!(model.policy_identity().expect("after identity"), identity);
    assert_eq!(trainer.optimizer_step(), 0);
    assert_eq!(trainer.shuffle_draws_for_test(), draws);

    batch.replace_advantage_for_test(0, advantage);
    let report = trainer.train_update(&model, &batch).expect("retry");
    assert_eq!(report.optimizer_step, 1);
}

#[test]
fn effective_gradient_is_stable_across_microbatch_partitions() {
    let (frame, space) = frame_and_space();
    let first = PolicyModel::fresh(104).expect("first model");
    let second = PolicyModel::fresh(104).expect("second model");
    let config = smoke_config();
    let batch = bandit_batch(&first, &frame, &space, config);
    let samples = (0..65)
        .map(|index| batch.sample(index % 2).expect("sample"))
        .collect::<Vec<_>>();
    let references = samples.iter().collect::<Vec<_>>();
    let mut first_adam = first
        .claim_adam_for_test(config.adam())
        .expect("first Adam");
    let mut second_adam = second
        .claim_adam_for_test(config.adam())
        .expect("second Adam");

    first
        .ppo_update_with_microbatch_for_test(&references, &mut first_adam, config, 64)
        .expect("64-way update");
    second
        .ppo_update_with_microbatch_for_test(&references, &mut second_adam, config, 13)
        .expect("13-way update");

    let first_parameters = first.export_parameters().expect("first parameters");
    let second_parameters = second.export_parameters().expect("second parameters");
    let maximum_difference = first_parameters
        .iter()
        .zip(second_parameters)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(maximum_difference < 2.0e-6, "{maximum_difference}");
}

#[test]
fn reward_shaping_is_bounded_and_terminal_result_dominates() {
    let mut tracker = RewardTracker::default();
    let mut summary = crate::GlobalSummary::default();
    tracker.observe(summary, 1.0, None).expect("baseline");
    let mut shaping = 0.0;
    for step in 1..=200u32 {
        summary.enemy_structures_destroyed = step;
        shaping += tracker.observe(summary, 1.0, None).expect("shaping").total;
    }
    let win = tracker
        .observe(summary, 1.0, Some(PpoTerminalOutcome::Win))
        .expect("win");

    assert!(shaping.abs() <= PPO_SHAPING_BUDGET + 1.0e-5);
    assert_eq!(win.terminal, PPO_TERMINAL_REWARD);
    assert!(win.total >= PPO_TERMINAL_REWARD - 1.0e-5);
    assert!((PPO_TERMINAL_REWARD - PPO_SHAPING_BUDGET - 1.0 / 101.0).abs() < 1.0e-7);
}

#[test]
fn reward_values_experience_not_cash_last_hits_or_denies() {
    let mut economy = RewardTracker::default();
    let baseline = crate::GlobalSummary::default();
    economy.observe(baseline, 1.0, None).expect("baseline");
    let mut improved = baseline;
    improved.own_gold = 100;
    improved.allied.xp = 100;
    let economy_reward = economy.observe(improved, 1.0, None).expect("economy");

    let mut farm = RewardTracker::default();
    farm.observe(baseline, 1.0, None).expect("baseline");
    let mut farmed = baseline;
    farmed.allied.last_hits = 10;
    farmed.allied.denies = 10;
    let farm_reward = farm.observe(farmed, 1.0, None).expect("farm");

    let mut opponent = RewardTracker::default();
    opponent.observe(baseline, 1.0, None).expect("baseline");
    let mut opponent_progress = baseline;
    opponent_progress.enemy.xp = 100;
    let opponent_reward = opponent
        .observe(opponent_progress, 1.0, None)
        .expect("opponent XP");

    assert_eq!(farm_reward.last_hits, 0.0);
    assert_eq!(farm_reward.denies, 0.0);
    assert_eq!(economy_reward.wealth, 0.0);
    assert!(economy_reward.experience > economy_reward.wealth);
    assert!(opponent_reward.experience < 0.0);
    assert!(economy_reward.total > farm_reward.total);
}

#[test]
fn reward_cash_spending_is_neutral() {
    let mut tracker = RewardTracker::default();
    let mut summary = crate::GlobalSummary {
        own_gold: 1_000,
        ..Default::default()
    };
    tracker.observe(summary, 0.99, None).expect("baseline");
    summary.own_gold = 0;

    let reward = tracker.observe(summary, 0.99, None).expect("purchase");

    assert_eq!(reward.wealth, 0.0);
    assert_eq!(reward.total, 0.0);
}

#[test]
fn reward_extreme_public_totals_emit_finite_bounded_components() {
    let mut tracker = RewardTracker::default();
    let mut summary = crate::GlobalSummary::default();
    summary.allied.xp = i64::MIN;
    summary.enemy.xp = i64::MAX;
    summary.allied.kills = u64::MAX;
    tracker.observe(summary, 1.0, None).expect("baseline");
    summary.allied.xp = i64::MAX;
    summary.enemy.xp = i64::MIN;
    summary.allied.kills = 0;
    summary.enemy.kills = u64::MAX;

    let reward = tracker
        .observe(summary, 1.0, Some(PpoTerminalOutcome::Loss))
        .expect("terminal");

    let components = [
        reward.experience,
        reward.combat,
        reward.structures,
        reward.wealth,
    ];
    for value in components {
        assert!(value.is_finite());
        assert!(value.abs() <= PPO_SHAPING_BUDGET);
    }
    assert!(components.iter().map(|value| value.abs()).sum::<f32>() <= PPO_SHAPING_BUDGET + 1.0e-6);
    assert!(reward.total.is_finite());
    assert!(reward.total.abs() <= 1.0 + PPO_SHAPING_BUDGET);
}

#[test]
fn reward_terminal_uses_zero_next_potential_independent_of_final_summary() {
    let previous = crate::GlobalSummary::default();
    let mut rich = previous;
    rich.allied.xp = 1_000;
    rich.enemy_structures_destroyed = 2;
    for outcome in [
        PpoTerminalOutcome::Win,
        PpoTerminalOutcome::Loss,
        PpoTerminalOutcome::Draw,
    ] {
        let mut left = RewardTracker::default();
        let mut right = RewardTracker::default();
        left.observe(previous, 0.99, None).expect("baseline");
        right.observe(previous, 0.99, None).expect("baseline");

        let reward = left
            .observe(previous, 0.99, Some(outcome))
            .expect("terminal");
        let alternate = right.observe(rich, 0.99, Some(outcome)).expect("terminal");

        assert_eq!(reward, alternate);
        assert!(reward.total.abs() <= 1.0);
    }
}

#[test]
fn reward_terminal_removes_previous_potential_at_the_same_normalized_scale() {
    let mut tracker = RewardTracker::default();
    let mut previous = crate::GlobalSummary::default();
    previous.allied.xp = 100;
    tracker.observe(previous, 0.99, None).expect("baseline");

    let reward = tracker
        .observe(previous, 0.99, Some(PpoTerminalOutcome::Win))
        .expect("terminal");

    assert_eq!(reward.terminal, 1.0);
    assert!((reward.experience + 2.0 / 101.0).abs() < 1.0e-7);
    assert!((reward.total - (1.0 - 2.0 / 101.0)).abs() < 1.0e-7);
}

#[test]
fn reward_alternating_potential_exhausts_absolute_budget_without_replenishing() {
    let mut tracker = RewardTracker::default();
    let mut summary = crate::GlobalSummary::default();
    tracker.observe(summary, 1.0, None).expect("baseline");
    let mut expenditure = 0.0;
    for index in 0..64 {
        summary.enemy_structures_destroyed = if index % 2 == 0 { 100 } else { 0 };
        let reward = tracker
            .observe(summary, 1.0, None)
            .expect("alternating shaping");
        expenditure += reward.total.abs();
        assert!(reward.total.is_finite());
        assert!(expenditure <= PPO_SHAPING_BUDGET + 1.0e-6);
        if index > 0 {
            assert_eq!(reward.total, 0.0);
        }
    }
    assert!(expenditure < 1.0);
}

#[test]
fn reward_rejects_invalid_discount_without_consuming_budget_or_previous_state() {
    let mut tracker = RewardTracker::default();
    let baseline = crate::GlobalSummary::default();
    tracker.observe(baseline, 1.0, None).expect("baseline");
    let improved = crate::GlobalSummary {
        enemy_structures_destroyed: 1,
        ..baseline
    };
    for discount in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        let error = tracker
            .observe(improved, discount, None)
            .expect_err("invalid discount");
        assert_eq!(error, crate::PpoError::InvalidDiscount);
        assert_eq!(error.to_string(), "invalid tick discount");
    }
    let reward = tracker.observe(improved, 1.0, None).expect("valid shaping");
    assert_eq!(reward.structures, 5.0 / 101.0);
    assert_eq!(reward.total, reward.structures);
}

fn frame_and_space() -> (crate::FeatureFrame, ActionSpace) {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    (frame, space)
}

fn policy_batch_inputs() -> (Vec<crate::FeatureFrame>, Vec<ActionSpace>) {
    let mut frames = Vec::with_capacity(MODEL_TRAINING_BATCH);
    let mut spaces = Vec::with_capacity(MODEL_TRAINING_BATCH);
    for index in 0..MODEL_TRAINING_BATCH {
        let team = if index.is_multiple_of(2) {
            Team::Radiant
        } else {
            Team::Dire
        };
        let tracker = tracker_with_view(team, world_view(team, 10 + index as u32));
        let space = ActionSpace::from_tracker(&tracker).expect("action space");
        let mut frame = encode(&tracker, &LocalPolicyState::new(0));
        frame.global[63] = index as f32 / MODEL_TRAINING_BATCH as f32;
        frames.push(frame);
        spaces.push(space);
    }
    (frames, spaces)
}

fn set_policy_parameter_range(
    model: &PolicyModel,
    parameters: &mut [f32],
    target: &str,
    range: std::ops::Range<usize>,
    value: f32,
) {
    let mut offset = 0usize;
    for (name, shape) in model.parameter_schema().expect("parameter schema") {
        let count = shape.iter().product::<usize>();
        if name == target {
            assert!(range.start <= range.end);
            assert!(range.end <= count);
            parameters[offset + range.start..offset + range.end].fill(value);
            return;
        }
        offset += count;
    }
    panic!("missing model parameter {target}");
}

fn choice(
    model: &PolicyModel,
    frame: &crate::FeatureFrame,
    space: &ActionSpace,
    action: StructuredAction,
) -> PpoPolicyChoice {
    let (log_probability, entropy, value) = model
        .action_statistics(frame, space, action)
        .expect("action statistics");
    PpoPolicyChoice {
        frame: frame.clone(),
        target: BehavioralTarget::from_action(frame, space, action).expect("target"),
        action,
        policy: model.policy_identity().expect("policy"),
        log_probability,
        entropy,
        value,
    }
}

fn smoke_config() -> PpoConfig {
    PpoConfig {
        rollout_decisions: 1,
        environments: 2,
        epochs: 1,
        minibatch: 2,
        entropy_coefficient: 1.0e-4,
        target_kl: 1.0,
        ..PpoConfig::default()
    }
}

fn bandit_batch(
    model: &PolicyModel,
    frame: &crate::FeatureFrame,
    space: &ActionSpace,
    config: PpoConfig,
) -> crate::PpoBatch {
    let mut rollout =
        PpoRollout::new(2, model.policy_identity().expect("policy")).expect("rollout");
    for (stream, action, reward) in [
        (0, StructuredAction::Continue, 1.0),
        (
            1,
            StructuredAction::Hold {
                unit: ControlledUnit::Hero,
            },
            -1.0,
        ),
    ] {
        rollout
            .push(
                choice(model, frame, space, action)
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    rollout.finish(config).expect("batch")
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_smoke_exercises_real_arena_rollout_and_one_ppo_update() {
    let report = crate::run_ppo_smoke(crate::PpoSmokeConfig {
        updates: 1,
        environments: 1,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 2,
        seed: 77,
        map: bota_proto::MapId(1),
    })
    .expect("smoke PPO");

    assert_eq!(report.updates, 1);
    assert_eq!(report.transitions, 2);
    assert_eq!(report.optimizer_step, 1);
    assert_eq!(report.elapsed_ticks, 6);
    assert!(report.final_policy_loss.is_finite());
    assert!(report.final_value_loss.is_finite());
    assert!(report.final_entropy.is_finite());
    assert!(report.final_kl.is_finite());
}

#[cfg(feature = "builtin")]
#[test]
fn persistent_actor_refreshes_policy_after_waiting_for_a_recycled_buffer() {
    let report = crate::run_ppo_smoke(crate::PpoSmokeConfig {
        updates: 3,
        environments: 2,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 4,
        seed: 3,
        map: bota_proto::MapId(1),
    })
    .expect("three-update pipeline");

    assert_eq!(report.updates, 3);
    assert_eq!(report.transitions, 12);
    assert_eq!(report.optimizer_step, 3);
}

#[cfg(feature = "builtin")]
#[test]
fn wall_checkpoint_schedule_uses_fixed_monotonic_deadlines_without_drift() {
    let cadence = crate::TrainingCheckpointCadence::WallTime(std::time::Duration::from_secs(300));
    let mut schedule = crate::ppo_arena::TrainingCheckpointSchedule::new(cadence)
        .expect("wall checkpoint schedule");

    assert!(!schedule.is_due(1, std::time::Duration::from_secs(299)));
    assert!(schedule.is_due(2, std::time::Duration::from_secs(300)));
    schedule
        .mark_committed(std::time::Duration::from_secs(300))
        .expect("second deadline");
    assert!(!schedule.is_due(3, std::time::Duration::from_secs(599)));
    assert!(schedule.is_due(4, std::time::Duration::from_secs(600)));
    schedule
        .mark_committed(std::time::Duration::from_secs(600))
        .expect("third deadline");
    assert!(!schedule.is_due(5, std::time::Duration::from_secs(899)));
    assert!(schedule.is_due(6, std::time::Duration::from_secs(900)));
    schedule
        .mark_committed(std::time::Duration::from_secs(1_201))
        .expect("deadline after a delayed commit");
    assert!(!schedule.is_due(7, std::time::Duration::from_secs(1_499)));
    assert!(schedule.is_due(8, std::time::Duration::from_secs(1_500)));
}

#[cfg(feature = "builtin")]
#[test]
fn production_training_resume_matches_uninterrupted_parameters_adam_and_rng() {
    let uninterrupted_directory = training_test_directory("production-uninterrupted");
    let resumed_directory = training_test_directory("production-resumed");
    let mut settings = crate::TrainingJobConfig {
        updates: 2,
        environments: 2,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 2,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed: 23_071,
        map: bota_proto::MapId(1),
        git_commit: "test-drysua-commit".to_owned(),
        simulator_commit: "test-bota-commit".to_owned(),
    };

    crate::run_training_job_on(
        settings.clone(),
        crate::PolicyDevice::Cpu,
        &uninterrupted_directory,
        false,
        |_| {},
    )
    .expect("uninterrupted production updates");
    settings.updates = 1;
    crate::run_training_job_on(
        settings.clone(),
        crate::PolicyDevice::Cpu,
        &resumed_directory,
        false,
        |_| {},
    )
    .expect("first production update");
    settings.updates = 2;
    crate::run_training_job_on(
        settings,
        crate::PolicyDevice::Cpu,
        &resumed_directory,
        true,
        |_| {},
    )
    .expect("resumed production update");

    let uninterrupted =
        crate::TrainingArtifact::load(&uninterrupted_directory).expect("uninterrupted artifact");
    let resumed = crate::TrainingArtifact::load(&resumed_directory).expect("resumed artifact");
    assert_eq!(uninterrupted.run(), resumed.run());
    assert_eq!(uninterrupted.progress(), resumed.progress());
    assert_eq!(resumed.progress().global_update, 2);
    assert_eq!(resumed.progress().policy_version, 2);
    assert_eq!(resumed.progress().rollout_samples, 8);
    assert_production_artifact_training_state_equal(&uninterrupted, &resumed);
    std::fs::remove_dir_all(uninterrupted_directory).expect("remove uninterrupted directory");
    std::fs::remove_dir_all(resumed_directory).expect("remove resumed directory");
}

#[cfg(feature = "builtin")]
fn assert_production_artifact_training_state_equal(
    uninterrupted: &crate::TrainingArtifact,
    resumed: &crate::TrainingArtifact,
) {
    let source = PolicyModel::fresh(23_072).expect("source model");
    let target = PolicyModel::fresh(23_073).expect("target model");
    let source_state = uninterrupted
        .restore(&source, uninterrupted.run())
        .expect("source restore");
    let target_state = resumed
        .restore(&target, resumed.run())
        .expect("target restore");
    let source_trainer = source_state.trainer();
    let target_trainer = target_state.trainer();
    let source_snapshot = source_trainer
        .checkpoint_snapshot(&source)
        .expect("source snapshot");
    let target_snapshot = target_trainer
        .checkpoint_snapshot(&target)
        .expect("target snapshot");

    assert_eq!(source_snapshot.parameters, target_snapshot.parameters);
    assert_eq!(
        source_snapshot.adam.moments(),
        target_snapshot.adam.moments()
    );
    assert_eq!(
        source_trainer.optimizer_step(),
        target_trainer.optimizer_step()
    );
    assert_eq!(source_trainer.updates(), target_trainer.updates());
    assert_eq!(
        source_trainer.rng_checkpoint(),
        target_trainer.rng_checkpoint()
    );
    assert!(target_trainer.optimizer_step() > 0);
    assert!(target_trainer.rng_checkpoint().1 > 0);
}

#[cfg(feature = "builtin")]
#[test]
fn training_job_checkpoints_and_resumes_from_the_next_update() {
    let directory = training_test_directory("resume");
    let mut settings = crate::TrainingJobConfig {
        updates: 1,
        environments: 2,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 2,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed: 23_071,
        map: bota_proto::MapId(1),
        git_commit: "test-drysua-commit".to_owned(),
        simulator_commit: "test-bota-commit".to_owned(),
    };

    let first = crate::run_training_job_on(
        settings.clone(),
        crate::PolicyDevice::Cpu,
        &directory,
        false,
        |_| {},
    )
    .expect("first training update");
    assert_eq!(first.completed_updates, 1);
    assert_eq!(first.optimizer_step, 2);
    assert_eq!(
        crate::TrainingArtifact::load(&directory)
            .expect("first checkpoint")
            .progress()
            .global_update,
        1
    );

    settings.updates = 2;
    let resumed = crate::run_training_job_on(
        settings.clone(),
        crate::PolicyDevice::Cpu,
        &directory,
        true,
        |_| {},
    )
    .expect("resumed training update");
    assert_eq!(resumed.completed_updates, 2);
    assert_eq!(resumed.optimizer_step, 4);
    assert_eq!(
        crate::TrainingArtifact::load(&directory)
            .expect("resumed checkpoint")
            .progress()
            .global_update,
        2
    );

    settings.updates = 1;
    settings.git_commit = "test-drysua-next-commit".to_owned();
    settings.resume_provenance = crate::ResumeProvenance::MigrateGitCommit;
    let error = crate::run_training_job_on(
        settings.clone(),
        crate::PolicyDevice::Cpu,
        &directory,
        true,
        |_| {},
    )
    .expect_err("migration target before restored update");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: training update target precedes checkpoint"
    );
    assert_eq!(
        crate::TrainingArtifact::load(&directory)
            .expect("checkpoint after rejected migration")
            .run()
            .git_commit,
        "test-drysua-commit"
    );

    settings.updates = 3;
    let mut incompatible = settings.clone();
    incompatible.simulator_commit = "different-bota-commit".to_owned();
    let error = crate::run_training_job_on(
        incompatible,
        crate::PolicyDevice::Cpu,
        &directory,
        true,
        |_| {},
    )
    .expect_err("migration with a different simulator commit");
    assert_eq!(
        error.to_string(),
        "invalid PPO config field: provenance migration scope"
    );
    let migrated =
        crate::run_training_job_on(settings, crate::PolicyDevice::Cpu, &directory, true, |_| {})
            .expect("migrated training update");
    let migrated_artifact = crate::TrainingArtifact::load(&directory).expect("migrated checkpoint");
    assert_eq!(migrated.completed_updates, 3);
    assert_eq!(migrated_artifact.progress().global_update, 3);
    assert_eq!(
        migrated_artifact.run().git_commit,
        "test-drysua-next-commit"
    );

    std::fs::remove_dir_all(directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
#[test]
fn fresh_training_loads_the_requested_runtime_weights_before_the_first_update() {
    let weights_directory = training_test_directory("initial-weights");
    let checkpoint_directory = training_test_directory("initialized-run");
    let initial_model = PolicyModel::fresh(23_074).expect("initial model");
    crate::TrainingArtifact::save_runtime_weights(&initial_model, &weights_directory)
        .expect("initial runtime weights");
    let initial_fingerprint = crate::PolicySnapshot::capture(&initial_model, 0)
        .expect("initial snapshot")
        .fingerprint();
    let settings = crate::TrainingJobConfig {
        updates: 1,
        environments: 2,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 2,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed: 23_075,
        map: bota_proto::MapId(1),
        git_commit: "test-drysua-commit".to_owned(),
        simulator_commit: "test-bota-commit".to_owned(),
    };

    let report = crate::run_training_job_on_with_initial_weights(
        settings,
        crate::PolicyDevice::Cpu,
        &checkpoint_directory,
        false,
        Some(&weights_directory),
        |_| {},
    )
    .expect("initialized training update");

    assert_eq!(report.starting_policy_fingerprint, initial_fingerprint);
    assert_eq!(report.completed_updates, 1);
    std::fs::remove_dir_all(weights_directory).expect("remove initial weights");
    std::fs::remove_dir_all(checkpoint_directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
#[test]
fn production_training_rejects_an_unpaired_environment_count() {
    let directory = training_test_directory("odd-environments");
    let error = crate::run_training_job_on(
        crate::TrainingJobConfig {
            updates: 1,
            environments: 1,
            rollout_decisions: 2,
            epochs: 1,
            minibatch: 2,
            checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
            resume_provenance: crate::ResumeProvenance::Strict,
            seed: 23_079,
            map: bota_proto::MapId(1),
            git_commit: "test-drysua-commit".to_owned(),
            simulator_commit: "test-bota-commit".to_owned(),
        },
        crate::PolicyDevice::Cpu,
        &directory,
        false,
        |_| {},
    )
    .expect_err("production arenas require complete side pairs");

    assert_eq!(
        error.to_string(),
        "invalid PPO config field: training environments"
    );
    std::fs::remove_dir_all(directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
#[test]
fn resumed_training_rejects_an_initial_weights_directory() {
    let directory = training_test_directory("resume-with-initial");
    let settings = crate::TrainingJobConfig {
        updates: 1,
        environments: 1,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 2,
        checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
        resume_provenance: crate::ResumeProvenance::Strict,
        seed: 23_076,
        map: bota_proto::MapId(1),
        git_commit: "test-drysua-commit".to_owned(),
        simulator_commit: "test-bota-commit".to_owned(),
    };

    let error = crate::run_training_job_on_with_initial_weights(
        settings,
        crate::PolicyDevice::Cpu,
        &directory,
        true,
        Some(&directory),
        |_| {},
    )
    .expect_err("resume must not reload runtime weights");

    assert_eq!(
        error.to_string(),
        "invalid PPO config field: resume initial weights"
    );
    std::fs::remove_dir_all(directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
#[test]
fn training_job_rejects_a_checkpoint_directory_locked_by_another_writer() {
    let directory = training_test_directory("locked");
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(directory.join(".training.lock"))
        .expect("create training lock");
    lock.lock().expect("hold training lock");
    let error = crate::run_training_job_on(
        crate::TrainingJobConfig {
            updates: 1,
            environments: 1,
            rollout_decisions: 2,
            epochs: 1,
            minibatch: 2,
            checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(1),
            resume_provenance: crate::ResumeProvenance::Strict,
            seed: 23_072,
            map: bota_proto::MapId(1),
            git_commit: "test-drysua-commit".to_owned(),
            simulator_commit: "test-bota-commit".to_owned(),
        },
        crate::PolicyDevice::Cpu,
        &directory,
        true,
        |_| {},
    )
    .expect_err("second checkpoint writer");

    assert!(error.to_string().contains("checkpoint directory is locked"));
    drop(lock);
    std::fs::remove_dir_all(directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
#[test]
fn production_warmup_phases_cover_pregame_and_active_match_ticks() {
    assert_eq!(crate::training_warmup_decisions(0), 0);
    assert_eq!(crate::training_warmup_decisions(1), 300);
    assert!(crate::training_warmup_decisions(2) * 3 > 900);
    assert_eq!(crate::training_warmup_decisions(7), 2_400);
    assert_eq!(crate::training_warmup_decisions(8), 0);
}

#[cfg(feature = "builtin")]
#[test]
fn production_warmup_cycle_has_a_fixed_discarded_decision_budget() {
    let decisions = (0..8).map(crate::training_warmup_decisions).sum::<usize>();

    assert_eq!(decisions, 7_650);
    assert_eq!(decisions * 2 * 2, 30_600);
}

#[cfg(feature = "builtin")]
#[test]
fn production_training_alternates_policy_side_for_odd_environment_counts() {
    assert_eq!(crate::training_policy_seat(0), 0);
    assert_eq!(crate::training_policy_seat(1), 1);
    assert_eq!(crate::training_policy_seat(2), 0);
}

#[cfg(feature = "builtin")]
#[test]
fn production_training_covers_every_warmup_phase_on_both_policy_sides() {
    let mut covered = [[false; 2]; 8];
    for stream in 0..16 {
        covered[crate::training_warmup_phase_index(stream)][crate::training_policy_seat(stream)] =
            true;
    }

    assert!(covered.into_iter().flatten().all(|present| present));
    assert_eq!(crate::training_pair_index(0), crate::training_pair_index(1));
    assert_ne!(crate::training_pair_index(1), crate::training_pair_index(2));
}

#[cfg(feature = "builtin")]
#[test]
fn production_training_covers_every_warmup_phase_against_both_baselines() {
    let mut covered = [[false; 8]; 2];
    for stream in 0..32 {
        let pair = crate::training_pair_index(stream);
        let baseline = match crate::training_opponent_baseline_for_test(pair) {
            crate::CheckpointEvaluationBaseline::Weak => 0,
            crate::CheckpointEvaluationBaseline::Teacher => 1,
        };
        covered[baseline][crate::training_warmup_phase_index(stream)] = true;
    }

    assert!(covered.into_iter().flatten().all(|present| present));
}

#[cfg(feature = "builtin")]
#[test]
fn production_warmup_runs_the_frozen_policy_against_the_scheduled_opponent() {
    let model = stop_policy_for_warmup();

    let orders = crate::production_warmup_order_counts_for_test(&model, 16)
        .expect("policy warmup against Weak");

    assert_eq!(orders, [1, 0]);
}

#[cfg(feature = "builtin")]
#[test]
fn batched_production_warmup_runs_both_policy_sides_without_weak_orders() {
    let model = stop_policy_for_warmup();

    let orders = crate::production_batched_warmup_order_counts_for_test(&model, 16)
        .expect("batched policy warmup against Weak");

    assert_eq!(orders, [[1, 0], [0, 1]]);
}

#[cfg(feature = "builtin")]
#[test]
fn production_warmup_cleanup_preserves_server_mirrored_item_timers() {
    assert!(
        crate::production_warmup_cleanup_preserves_readiness_for_test()
            .expect("warmup readiness cleanup")
    );
}

#[cfg(feature = "builtin")]
#[test]
fn production_seed_derivation_accepts_maximum_seed_without_overflow() {
    let arena = crate::derive_training_seed(u64::MAX, 1_000_000, 1);
    let opponent = crate::derive_training_seed(u64::MAX, 1_000_000, 2);

    assert_ne!(arena, opponent);
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_pretraining_collection_is_balanced_bounded_and_diverse() {
    const EXPECTED_MAP_ONE_SAMPLES: u64 = 4_663;
    let (samples, actions, splits) =
        crate::collect_pretraining_summary_for_test(50_001).expect("pretraining collection");
    println!("samples={samples} actions={actions:?} splits={splits:?}");

    assert_eq!(
        u64::try_from(samples).expect("sample count fits"),
        EXPECTED_MAP_ONE_SAMPLES
    );
    assert_eq!(
        actions.into_iter().flatten().sum::<u64>(),
        EXPECTED_MAP_ONE_SAMPLES
    );
    assert_eq!(splits.iter().sum::<usize>(), samples);
    assert!(splits.into_iter().all(|count| count > 0));
    assert!(actions[0].iter().filter(|count| **count != 0).count() >= 2);
    assert!(actions[1][ActionKind::MovePoint.index()] > 0);
    assert!(actions[1][ActionKind::AttackMovePoint.index()] > 0);
    assert!(actions[2][ActionKind::MovePoint.index()] > 0);
    assert!(actions[2][ActionKind::AttackMovePoint.index()] > 0);
    assert!(actions[0][ActionKind::Continue.index()] > actions[0][ActionKind::MovePoint.index()]);
    assert!(
        actions[0][ActionKind::Continue.index()] > actions[0][ActionKind::AttackMovePoint.index()]
    );
    assert_eq!(actions[0][ActionKind::Continue.index()], 1_536);
    assert_eq!(actions[1][ActionKind::Continue.index()], 336);
    assert_eq!(actions[2][ActionKind::Continue.index()], 336);
    for kind in ActionKind::ALL
        .into_iter()
        .filter(|kind| *kind != ActionKind::Continue)
    {
        assert!(actions[0][kind.index()] <= 192);
        assert!(actions[1][kind.index()] <= 64);
        assert!(actions[2][kind.index()] <= 64);
    }
    assert!(
        actions[0].iter().copied().max().expect("action maximum") * 100
            < u64::try_from(splits[0]).expect("training split fits") * 95,
        "teacher actions: {actions:?}"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_epochs_reserve_at_least_one_epoch_for_each_dagger_phase() {
    let valid = crate::BehavioralPretrainingConfig {
        epochs: 8,
        seed: 50_001,
    };
    crate::validate_behavioral_pretraining_for_test(valid).expect("four DAgger phases");

    let error =
        crate::validate_behavioral_pretraining_for_test(crate::BehavioralPretrainingConfig {
            epochs: 6,
            ..valid
        })
        .expect_err("three remaining epochs cannot cover four DAgger phases");

    assert_eq!(
        error.to_string(),
        "invalid PPO config field: pretraining epochs"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_dagger_phases_all_target_the_hard_weak_quality_baseline() {
    let baselines = (0..4)
        .map(crate::pretraining_dagger_baseline_for_test)
        .collect::<Vec<_>>();

    assert_eq!(
        baselines,
        [
            crate::CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationBaseline::Weak,
            crate::CheckpointEvaluationBaseline::Weak,
        ]
    );
}

#[cfg(feature = "builtin")]
#[test]
fn deployment_uses_the_audited_teacher_only_for_map_zero() {
    assert!(crate::deployment_uses_teacher_for_test(bota_proto::MapId(
        0
    )));
    assert!(!crate::deployment_uses_teacher_for_test(bota_proto::MapId(
        1
    )));
    assert!(!crate::deployment_uses_teacher_for_test(bota_proto::MapId(
        2
    )));
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_agreement_accepts_documented_diverse_dataset_boundaries() {
    let metrics = boundary_pretraining_agreement();

    crate::validate_pretraining_agreement_for_test(&metrics)
        .expect("documented agreement boundaries");
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_agreement_rejects_overall_kind_one_below_boundary() {
    let mut metrics = boundary_pretraining_agreement();
    metrics.overall.kind.matching = 59;

    let error = crate::validate_pretraining_agreement_for_test(&metrics)
        .expect_err("overall kind below boundary");

    assert_eq!(
        error.to_string(),
        "PPO model error: pretraining held-out overall agreement failed: kind=59/100, full=55/100"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_agreement_rejects_dire_full_one_below_boundary() {
    let mut metrics = boundary_pretraining_agreement();
    metrics.dire.full.matching = 49;

    let error = crate::validate_pretraining_agreement_for_test(&metrics)
        .expect_err("Dire full agreement below boundary");

    assert_eq!(
        error.to_string(),
        "PPO model error: pretraining held-out dire agreement failed: kind=55/100, full=49/100"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_stage_selection_prefers_exact_agreement_then_kind_agreement() {
    let selected = crate::pretraining_best_stage_for_test(&[(700, 650), (735, 705), (800, 690)]);
    assert_eq!(selected, 1);

    let selected = crate::pretraining_best_stage_for_test(&[(700, 650), (735, 705), (800, 705)]);
    assert_eq!(selected, 2);
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_stage_selection_prioritizes_gameplay_failures_then_wins() {
    let selected = crate::pretraining_gameplay_stage_for_test(&[
        [gameplay(0, 0, 2, 2, 1), gameplay(2, 0, 0, 0, 1)],
        [gameplay(0, 0, 0, 0, 0), gameplay(1, 1, 1, 1, 2)],
        [gameplay(0, 0, 0, 0, 0), gameplay(0, 0, 0, 0, 0)],
    ]);
    assert_eq!(selected, 2);

    let selected = crate::pretraining_gameplay_stage_for_test(&[
        [gameplay(0, 0, 0, 0, 0), gameplay(0, 0, 0, 0, 0)],
        [gameplay(0, 0, 0, 0, 0), gameplay(0, 2, 2, 2, 1)],
    ]);
    assert_eq!(selected, 1);
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_stage_selection_uses_three_fixed_gameplay_validation_seeds() {
    assert_eq!(
        crate::pretraining_gameplay_validation_seeds_for_test(),
        [9_100_001, 9_100_002, 9_100_003]
    );
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_stage_selection_never_trades_a_map_one_failure_for_map_zero_progress() {
    let selected = crate::pretraining_gameplay_stage_for_test(&[
        [gameplay(0, 0, 2, 2, 0), gameplay(1, 1, 1, 1, 0)],
        [gameplay(2, 0, 0, 0, 0), gameplay(0, 0, 0, 0, 2)],
    ]);

    assert_eq!(selected, 1);
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_stage_selection_uses_map_zero_progress_after_map_one_ties() {
    let selected = crate::pretraining_gameplay_stage_for_test(&[
        [gameplay(1, 0, 1, 1, 1), gameplay(0, 2, 2, 2, 1)],
        [gameplay(0, 0, 2, 2, 0), gameplay(0, 2, 2, 2, 1)],
    ]);

    assert_eq!(selected, 1);
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_acceptance_allows_map_zero_timeouts_with_paired_structure_progress() {
    crate::validate_pretraining_gameplay_acceptance_for_test(
        gameplay(0, 0, 2, 2, 1),
        gameplay(0, 2, 2, 2, 1),
    )
    .expect("both map gates");
}

#[cfg(feature = "builtin")]
#[test]
fn pretraining_acceptance_rejects_a_map_zero_stall_on_one_side() {
    let error = crate::validate_pretraining_gameplay_acceptance_for_test(
        gameplay(1, 0, 1, 1, 1),
        gameplay(0, 2, 2, 2, 1),
    )
    .expect_err("one Map0 side stalled");

    assert!(error.to_string().contains("gameplay acceptance failed"));
}

#[cfg(feature = "builtin")]
fn gameplay(
    failures: usize,
    wins: usize,
    progress_games: usize,
    structures: u64,
    deaths: u64,
) -> crate::PretrainingGameplay {
    crate::PretrainingGameplay {
        games: 2,
        failures,
        wins,
        structure_progress_games: progress_games,
        structures,
        deaths,
        rejections: 0,
    }
}

#[cfg(feature = "builtin")]
fn boundary_pretraining_agreement() -> crate::OfflineEvaluation {
    let mut metrics = crate::OfflineEvaluation::default();
    metrics.overall.kind = crate::AgreementCount {
        matching: 60,
        total: 100,
    };
    metrics.overall.full = crate::AgreementCount {
        matching: 55,
        total: 100,
    };
    for side in [&mut metrics.radiant, &mut metrics.dire] {
        side.kind = crate::AgreementCount {
            matching: 55,
            total: 100,
        };
        side.full = crate::AgreementCount {
            matching: 50,
            total: 100,
        };
    }
    metrics
}

#[cfg(feature = "builtin")]
fn stop_policy_for_warmup() -> PolicyModel {
    let model = PolicyModel::fresh(23_077).expect("warmup model");
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0usize;
    for (name, shape) in model.parameter_schema().expect("parameter schema") {
        if name == "kind.bias" {
            parameters[offset + ActionKind::Stop.index()] = 10.0;
            model
                .import_parameters(&parameters)
                .expect("stop policy parameters");
            return model;
        }
        offset += shape.iter().product::<usize>();
    }
    panic!("kind bias parameter is missing");
}

#[cfg(feature = "builtin")]
#[test]
fn training_job_rejects_targets_that_cannot_fit_shuffle_rng_counters() {
    let directory = training_test_directory("counter-bound");
    let error = crate::run_training_job_on(
        crate::TrainingJobConfig {
            updates: 1_000_000,
            environments: 16,
            rollout_decisions: 64,
            epochs: 1,
            minibatch: 32,
            checkpoint_cadence: crate::TrainingCheckpointCadence::Updates(5),
            resume_provenance: crate::ResumeProvenance::Strict,
            seed: 23_073,
            map: bota_proto::MapId(1),
            git_commit: "test-drysua-commit".to_owned(),
            simulator_commit: "test-bota-commit".to_owned(),
        },
        crate::PolicyDevice::Cpu,
        &directory,
        false,
        |_| {},
    )
    .expect_err("uncheckpointable shuffle RNG target");

    assert_eq!(
        error.to_string(),
        "invalid PPO config field: training shuffle RNG counter"
    );
    std::fs::remove_dir_all(directory).expect("remove checkpoint directory");
}

#[cfg(feature = "builtin")]
fn training_test_directory(name: &str) -> std::path::PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "drysua-training-job-{name}-{}-{sequence}",
        std::process::id()
    ));
    if directory.exists() {
        std::fs::remove_dir_all(&directory).expect("remove stale training directory");
    }
    std::fs::create_dir(&directory).expect("create checkpoint directory");
    directory
}
