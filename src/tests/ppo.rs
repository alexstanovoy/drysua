#![allow(
    clippy::float_arithmetic,
    reason = "PPO reference calculations use floating-point arithmetic"
)]

use bota_proto::Team;

use super::feature::{encode, tracker_with_view, world_view};
use crate::{
    ABILITY_FEATURE_TOKENS, ActionSpace, BehavioralTarget, ControlledUnit, ITEM_FEATURE_TOKENS,
    LOOT_FEATURE_TOKENS, LocalPolicyState, POINT_FEATURE_TOKENS, PPO_RULES_AUDIT_VERSION,
    PPO_SCHEMA_HASH, PPO_SCHEMA_VERSION, PPO_SHAPING_BUDGET, PPO_TERMINAL_REWARD,
    PROJECTILE_FEATURE_TOKENS, PolicyModel, PpoConfig, PpoOutcome, PpoPolicyChoice, PpoRng,
    PpoRollout, PpoTerminalOutcome, PpoTrainer, REMEMBERED_UNIT_FEATURE_TOKENS, RewardTracker,
    StructuredAction, UNIT_FEATURE_TOKENS, clipped_surrogate, tick_discount,
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
    assert_eq!(PPO_SCHEMA_VERSION, 3);
    assert_eq!(PPO_RULES_AUDIT_VERSION, 3);
    assert_eq!(PPO_SCHEMA_HASH, 3_564_571_968_222_523_732);
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
            rollout_decisions: 257,
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
}

#[test]
fn reward_values_gold_and_experience_not_last_hits_or_denies() {
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
    assert!(economy_reward.wealth > 0.0);
    assert!(economy_reward.experience > economy_reward.wealth);
    assert!(opponent_reward.experience < 0.0);
    assert!(economy_reward.total > farm_reward.total);
}

fn frame_and_space() -> (crate::FeatureFrame, ActionSpace) {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    (frame, space)
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
        seed: 17_070,
        map: bota_proto::MapId(1),
    })
    .expect("three-update pipeline");

    assert_eq!(report.updates, 3);
    assert_eq!(report.transitions, 12);
    assert_eq!(report.optimizer_step, 3);
}
