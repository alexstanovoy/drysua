#![allow(
    clippy::float_arithmetic,
    reason = "PPO numerical and transaction regressions"
)]

use bota_proto::Team;

use super::feature::{encode, tracker_with_view, world_view};
use crate::{
    ActionSpace, BehavioralTarget, ControlledUnit, LocalPolicyState, MODEL_TRAINING_BATCH,
    PolicyModel, PpoConfig, PpoOutcome, PpoPolicyChoice, PpoRng, PpoRollout, PpoTrainer,
    StructuredAction, clipped_surrogate, tick_discount,
};

#[path = "ppo_capacity.rs"]
mod capacity;

#[test]
fn actor_batch_sampling_preserves_scalar_actions_rng_and_learner_likelihood() {
    let model = PolicyModel::fresh(9_101).expect("model");
    let (frames, spaces, mut random) = sampling_inputs(MODEL_TRAINING_BATCH);
    let mut scalar_random = random.clone();
    let batch = model
        .sample_batch(&frames, &spaces, &mut random)
        .expect("batch");
    let chosen = model.choose_batch(&frames, &spaces).expect("greedy batch");
    assert_eq!(batch.len(), MODEL_TRAINING_BATCH);
    assert_eq!(chosen.len(), batch.len());
    let mut samples = Vec::new();
    for (index, sampled) in batch.into_iter().enumerate() {
        let scalar = model
            .sample(&frames[index], &spaces[index], &mut scalar_random[index])
            .expect("actor");
        assert_eq!(sampled.action(), scalar.action());
        assert_eq!(sampled.policy(), scalar.policy());
        for (actual, expected) in [
            (sampled.log_probability(), scalar.log_probability()),
            (sampled.entropy(), scalar.entropy()),
            (sampled.value(), scalar.value()),
        ] {
            assert!((actual - expected).abs() <= 1.0e-4);
        }
        assert!(spaces[index].decode(sampled.action()).is_ok());
        let greedy = model
            .choose(&frames[index], &spaces[index])
            .expect("scalar greedy");
        assert_eq!(chosen[index].action, greedy.action);
        assert!((chosen[index].value - greedy.value).abs() < 1.0e-5);
        samples.push(prepared_choice(index, sampled));
    }
    assert_eq!(random, scalar_random);
    assert!(
        random.iter().all(
            |random| random.draws() > 0 && random.draws() <= crate::PPO_MAX_POLICY_SAMPLE_DRAWS
        )
    );
    assert_actor_likelihood(&model, &samples);
}

fn assert_actor_likelihood(model: &PolicyModel, samples: &[crate::PpoPreparedSample]) {
    assert!(
        samples
            .iter()
            .any(|sample| sample.transition.target.point_pointer.active)
    );
    assert!(
        samples
            .iter()
            .any(|sample| sample.transition.target.entity_pointer.active)
    );
    let references = samples.iter().collect::<Vec<_>>();
    let (likelihood, report) = model
        .ppo_likelihood_for_test(&references)
        .expect("learner likelihood");
    assert_eq!(likelihood.len(), samples.len());
    for (current, sample) in likelihood.iter().zip(samples) {
        assert!((current - sample.transition.old_log_probability).abs() < 1.0e-4);
    }
    assert!(report.approximate_kl.abs() < 1.0e-6);
    assert_eq!(report.clip_fraction, 0.0);
}

#[test]
fn invalid_sampling_batches_do_not_consume_rng() {
    let model = PolicyModel::fresh(9_102).expect("model");
    let (frame, space) = frame_and_space();
    let (_, stale) = frame_and_space();
    let frames = vec![frame; crate::MODEL_SAMPLING_BATCH + 1];
    let spaces = [space, stale];
    for (count, range, random_count, message) in [
        (
            0,
            0..0,
            0,
            "model batch must contain at least one frame".to_owned(),
        ),
        (
            crate::MODEL_SAMPLING_BATCH + 1,
            0..0,
            1,
            format!(
                "model batch count {} exceeds maximum {}",
                crate::MODEL_SAMPLING_BATCH + 1,
                crate::MODEL_SAMPLING_BATCH
            ),
        ),
        (
            1,
            0..0,
            1,
            "model batch action-space count 0 differs from frame count 1".to_owned(),
        ),
        (
            1,
            0..1,
            2,
            "model sampling RNG count 2 differs from frame count 1".to_owned(),
        ),
        (
            1,
            1..2,
            1,
            "model batch frame 0 does not belong to its action space".to_owned(),
        ),
    ] {
        let mut random = vec![PpoRng::new(22); random_count];
        let before = random.clone();
        let error = model
            .sample_batch(&frames[..count], &spaces[range], &mut random)
            .expect_err("invalid batch");
        assert_eq!(error.to_string(), message);
        assert_eq!(random, before);
    }
}

#[test]
fn failing_a_head_after_sampling_restores_rng() {
    let model = PolicyModel::fresh(9_104).expect("model");
    let mut parameters = vec![0.0; model.parameter_count()];
    let kind = crate::ActionKind::Stop.index();
    set_parameter_range(&model, &mut parameters, "kind.bias", kind..kind + 1, 100.0);
    set_parameter_range(
        &model,
        &mut parameters,
        "kind_embedding.weight",
        kind * 32..(kind + 1) * 32,
        1.0,
    );
    set_parameter_range(
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
    let mut random = [PpoRng::new(24)];
    let before = random.clone();
    assert_eq!(
        model
            .sample_batch(&[frame], &[space], &mut random)
            .expect_err("head overflow")
            .to_string(),
        "model radiant.controlled output at batch 0 index 0 is non-finite"
    );
    assert_eq!(random, before);
}

#[test]
fn trainer_retry_learns_rewarded_action_and_rejects_overly_stale_rollout_transactionally() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(101).expect("model");
    let reference = PolicyModel::fresh(101).expect("reference");
    let config = smoke_config();
    let mut batch = bandit_batch(&model, &frame, &space, config);
    let reference_batch = bandit_batch(&reference, &frame, &space, config);
    let mut trainer = PpoTrainer::new(&model, config, 7).expect("trainer");
    let mut expected = PpoTrainer::new(&reference, config, 7).expect("reference trainer");
    let before = trainer.checkpoint_snapshot(&model).expect("before");
    let identity = model.policy_identity().expect("identity");
    let random = trainer.rng_checkpoint();
    let probability = || {
        model
            .action_statistics(&frame, &space, StructuredAction::Continue)
            .expect("rewarded action statistics")
            .0
    };
    let before_probability = probability();
    let advantage = batch.replace_advantage_for_test(0, f32::NAN);
    let error = trainer
        .train_update(&model, &batch)
        .expect_err("nonfinite advantage");
    assert!(error.to_string().contains("non-finite"), "{error}");
    assert_eq!(
        trainer
            .checkpoint_snapshot(&model)
            .expect("failed snapshot"),
        before
    );
    assert_eq!(trainer.rng_checkpoint(), random);
    assert_eq!(model.policy_identity().expect("failed identity"), identity);
    batch.replace_advantage_for_test(0, advantage);
    let report = trainer.train_update(&model, &batch).expect("retry");
    assert!(probability() > before_probability);
    assert_eq!(report.optimizer_step, 1);
    assert_eq!(report.samples_optimized, 2);
    assert_eq!(
        report,
        expected
            .train_update(&reference, &reference_batch)
            .expect("clean update")
    );
    let actual = trainer
        .checkpoint_snapshot(&model)
        .expect("updated snapshot");
    let expected_state = expected
        .checkpoint_snapshot(&reference)
        .expect("reference snapshot");
    assert_eq!(actual.parameters, expected_state.parameters);
    assert_eq!(actual.adam.moments(), expected_state.adam.moments());
    assert_eq!(trainer.rng_checkpoint(), expected.rng_checkpoint());
    // Behaviour weights one and two updates old are within the pipeline bound.
    for update in [2, 3] {
        let report = trainer
            .train_update(&model, &batch)
            .expect("bounded staleness");
        assert_eq!(report.update, update);
    }
    let actual = trainer.checkpoint_snapshot(&model).expect("third snapshot");
    let random = trainer.rng_checkpoint();
    assert_eq!(
        trainer
            .train_update(&model, &batch)
            .expect_err("stale rollout")
            .to_string(),
        "PPO rollout policy identity is stale"
    );
    assert_eq!(
        trainer.checkpoint_snapshot(&model).expect("stale snapshot"),
        actual
    );
    assert_eq!(trainer.rng_checkpoint(), random);
}

#[test]
fn uneven_microbatch_partitions_produce_the_same_effective_update() {
    let (frame, space) = frame_and_space();
    let first = PolicyModel::fresh(104).expect("first");
    let second = PolicyModel::fresh(104).expect("second");
    let config = smoke_config();
    let batch = bandit_batch(&first, &frame, &space, config);
    let samples = (0..65)
        .map(|index| batch.sample(index % 2).expect("sample"))
        .collect::<Vec<_>>();
    let references = samples.iter().collect::<Vec<_>>();
    let mut first_adam = first
        .claim_optimizer(config.adam())
        .expect("first optimizer");
    let mut second_adam = second
        .claim_optimizer(config.adam())
        .expect("second optimizer");
    first
        .ppo_update_with_microbatch_for_test(&references, &mut first_adam, config, 64)
        .expect("64-way");
    second
        .ppo_update_with_microbatch_for_test(&references, &mut second_adam, config, 13)
        .expect("13-way");
    let difference = first
        .export_parameters()
        .expect("first parameters")
        .iter()
        .zip(second.export_parameters().expect("second parameters"))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(difference < 2.0e-6, "{difference}");
}

#[test]
fn gae_discounts_elapsed_ticks_and_stops_bootstrapping_at_terminal() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(93).expect("model");
    let mut rollout = PpoRollout::new(2).expect("rollout");
    for (decision, reward, terminal) in [(0, 1.0, false), (1, 2.0, true)] {
        let mut sampled = choice(&model, &frame, &space, StructuredAction::Continue);
        sampled.value = 0.0;
        rollout
            .push(
                sampled
                    .finish(
                        0,
                        PpoOutcome {
                            stream: 0,
                            decision,
                            ticks: 1,
                            next_value: 0.0,
                            reward,
                            terminal,
                        },
                    )
                    .expect("transition"),
            )
            .expect("push");
    }
    let batch = rollout
        .finish(PpoConfig {
            samples_per_update: 2,
            minibatch: 1,
            gamma_tick: 0.9,
            gae_lambda_tick: 0.8,
            ..PpoConfig::default()
        })
        .expect("batch");
    assert!((batch.sample(0).expect("first").return_value() - 2.44).abs() < 1.0e-5);
    assert_eq!(batch.sample(1).expect("terminal").return_value(), 2.0);
}

#[test]
fn explained_variance_separates_exact_blind_and_undefined_critics() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(94).expect("model");
    let returns = [1.0f32, -1.0, 0.5, 0.25];
    let batch = |values: [f32; 4], rewards: [f32; 4]| {
        let mut rollout = PpoRollout::new(4).expect("rollout");
        for (stream, (value, reward)) in values.into_iter().zip(rewards).enumerate() {
            let mut sampled = choice(&model, &frame, &space, StructuredAction::Continue);
            sampled.value = value;
            let outcome = PpoOutcome {
                stream,
                decision: 0,
                ticks: 3,
                next_value: 0.0,
                reward,
                terminal: true,
            };
            rollout
                .push(sampled.finish(0, outcome).expect("transition"))
                .expect("push");
        }
        let config = PpoConfig {
            samples_per_update: 4,
            minibatch: 4,
            ..PpoConfig::default()
        };
        rollout.finish(config).expect("batch").explained_variance()
    };
    // Single terminal samples: the lambda and Monte Carlo returns are both the reward.
    let exact = batch(returns, returns);
    assert!((exact.lambda - 1.0).abs() < 1.0e-12);
    assert_eq!(exact.lambda.to_bits(), exact.monte_carlo.to_bits());
    assert!(batch([0.3; 4], returns).monte_carlo.abs() < 1.0e-12);
    assert!(batch([0.0, 1.0, 2.0, 3.0], [0.5; 4]).lambda.is_nan());
}

#[test]
fn sampling_and_surrogate_boundaries_match_independent_references() {
    let (minimum, maximum) = crate::ppo::open_unit_bounds_for_test();
    assert!(minimum > 0.0);
    assert!(maximum < 1.0);
    assert!((tick_discount(0.99, 3).expect("discount") - 0.970_299).abs() < 1.0e-6);
    for (ratio, advantage, expected) in [(1.5, 2.0, 2.4), (0.5, -2.0, -1.6), (1.1, 2.0, 2.2)] {
        assert!((clipped_surrogate(ratio, advantage, 0.2) - expected).abs() < 1.0e-6);
    }
}

#[test]
fn compact_rollout_storage_preserves_frame_target_and_behavior_statistics() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(100).expect("model");
    let sampled = choice(&model, &frame, &space, StructuredAction::Continue);
    let target = sampled.target.clone();
    let packed = target.pack();
    let log_probability = sampled.log_probability;
    let mut rollout = PpoRollout::new(1).expect("rollout");
    rollout
        .push(
            sampled
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
                .expect("transition"),
        )
        .expect("push");
    assert_eq!(packed.unpack(), target);
    assert!(std::mem::size_of_val(&packed) < std::mem::size_of::<BehavioralTarget>());
    let batch = rollout.finish(smoke_config()).expect("batch");
    let sample = batch.sample(0).expect("materialized");
    assert_eq!(sample.transition.frame, frame);
    assert_eq!(sample.transition.target, target);
    assert_eq!(sample.transition.old_log_probability, log_probability);
}

fn frame_and_space() -> (crate::FeatureFrame, ActionSpace) {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = ActionSpace::from_tracker(&tracker).expect("space");
    (encode(&tracker, &LocalPolicyState::new(0)), space)
}

pub(super) fn sampling_inputs(
    count: usize,
) -> (Vec<crate::FeatureFrame>, Vec<ActionSpace>, Vec<PpoRng>) {
    assert!(count > 0);
    assert!(count <= MODEL_TRAINING_BATCH);
    let mut frames = Vec::with_capacity(count);
    let mut spaces = Vec::with_capacity(count);
    let mut random = Vec::with_capacity(count);
    for index in 0..count {
        let team = if index.is_multiple_of(2) {
            Team::Radiant
        } else {
            Team::Dire
        };
        let tracker = tracker_with_view(team, world_view(team, 10 + index as u32));
        spaces.push(ActionSpace::from_tracker(&tracker).expect("space"));
        let mut frame = encode(&tracker, &LocalPolicyState::new(0));
        frame.global[63] = index as f32 / MODEL_TRAINING_BATCH as f32;
        frames.push(frame);
        random.push(PpoRng::new(17 + index as u64 * 97));
    }
    (frames, spaces, random)
}

pub(super) fn prepared_choice(stream: usize, choice: PpoPolicyChoice) -> crate::PpoPreparedSample {
    crate::PpoPreparedSample {
        return_value: choice.value(),
        advantage: 1.0,
        transition: choice
            .finish(
                0,
                PpoOutcome {
                    stream,
                    decision: 0,
                    ticks: 3,
                    next_value: 0.0,
                    reward: 0.0,
                    terminal: true,
                },
            )
            .expect("transition"),
    }
}

fn choice(
    model: &PolicyModel,
    frame: &crate::FeatureFrame,
    space: &ActionSpace,
    action: StructuredAction,
) -> PpoPolicyChoice {
    let (log_probability, entropy, value) = model
        .action_statistics(frame, space, action)
        .expect("statistics");
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
        samples_per_update: 2,
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
    let mut rollout = PpoRollout::new(2).expect("rollout");
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
                    .finish(
                        0,
                        PpoOutcome {
                            stream,
                            decision: 0,
                            ticks: 3,
                            next_value: 0.0,
                            reward,
                            terminal: true,
                        },
                    )
                    .expect("transition"),
            )
            .expect("push");
    }
    rollout.finish(config).expect("batch")
}

fn set_parameter_range(
    model: &PolicyModel,
    parameters: &mut [f32],
    target: &str,
    range: std::ops::Range<usize>,
    value: f32,
) {
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        if name == target {
            assert!(range.start <= range.end);
            assert!(range.end <= count);
            parameters[offset + range.start..offset + range.end].fill(value);
            return;
        }
        offset += count;
    }
    panic!("missing parameter {target}");
}

#[test]
fn full_monte_carlo_returns_ignore_intermediate_bootstraps_but_not_truncation() {
    let (frame, space) = frame_and_space();
    let model = PolicyModel::fresh(9921000).expect("model");
    let sampled = choice(&model, &frame, &space, StructuredAction::Continue);
    let config = PpoConfig {
        samples_per_update: 2 * RETAINED,
        minibatch: 512,
        gae_lambda_tick: 1.0,
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        ..PpoConfig::default()
    };
    // One full-length game retaining every eighth of its decisions.
    const RETAINED: usize = 1_163;
    const STRIDE: usize = 8;
    let retained = RETAINED;
    let mut rollout = PpoRollout::new(2 * retained).expect("rollout");
    for decision in 0..retained {
        let terminal = decision + 1 == retained;
        let ticks = (crate::MAP2_ACTOR_DECISIONS - decision * STRIDE).min(STRIDE) as u32 * 3
            - u32::from(terminal);
        for stream in 0..2 {
            let reward = match (terminal, stream) {
                (true, 0) => 1.0,
                (true, _) => -1.0,
                (false, _) => 0.0,
            };
            let outcome = PpoOutcome {
                stream,
                decision: decision as u32,
                ticks,
                reward,
                terminal,
                next_value: if terminal { 0.0 } else { 99.0 },
            };
            rollout
                .push(sampled.clone().finish(0, outcome).expect("transition"))
                .expect("push");
        }
    }
    let batch = rollout.finish(config).expect("MC batch");
    for index in 0..batch.len() {
        let sample = batch.sample(index).expect("sample");
        let expected = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
        assert!((sample.return_value() - expected).abs() < 1e-5);
        let terminal = index / 2 + 1 == retained;
        assert_eq!(sample.transition.terminal, terminal);
        assert_eq!(sample.transition.ticks, if terminal { 11 } else { 24 });
    }
    let mut truncated = PpoRollout::new(1).expect("truncated");
    let outcome = PpoOutcome {
        stream: 0,
        decision: 0,
        ticks: 11,
        next_value: 0.7,
        reward: 0.0,
        terminal: false,
    };
    truncated
        .push(sampled.finish(0, outcome).expect("timeout"))
        .expect("push");
    let sample = truncated
        .finish(config)
        .expect("timeout MC")
        .sample(0)
        .expect("sample");
    assert!(!sample.transition.terminal);
    assert!((sample.return_value() - 0.7).abs() < 1e-5);
}
