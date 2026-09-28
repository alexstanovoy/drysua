//! Shared external hooks backed by one bounded native actor trial.

use super::*;
use crate::PpoBatch;

pub(super) fn emit_concurrency_probe_counts(streams: &[EpisodeStream], overlap: bool) {
    if std::env::var_os("DRYSUA_PROBE_MODE").is_none() {
        return;
    }
    assert!(streams.len() <= crate::PPO_ANNEALED_MAX_PARALLEL_WORLDS);
    let decisions = streams.iter().map(|stream| stream.decisions).sum::<usize>();
    let continues = streams
        .iter()
        .map(|stream| stream.actions[ActionKind::Continue.index()] as usize)
        .sum::<usize>();
    eprintln!(
        "concurrency-actor overlap={overlap} worlds={} decisions={decisions} continues={continues}",
        streams.len()
    );
}

pub(super) fn sample_active(
    model: &PolicyModel,
    random: &mut [PpoRng],
    environments: &mut [TrainingEnvironment],
    active: &[usize],
) -> Result<(Vec<PpoPolicyChoice>, Vec<ActionSpace>), PpoError> {
    validate_active(random, active)?;
    if environments.len() > MAX_EPISODE_ENVIRONMENTS
        || active.iter().any(|&stream| stream >= environments.len())
    {
        return Err(PpoError::InvalidConfig("parallel episode jobs"));
    }
    let (frames, spaces) = active
        .iter()
        .map(|&stream| prepare_policy_sample(&mut environments[stream]))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    select_choices(model, random, active, frames, spaces)
}

#[test]
fn reference_sampling_rejects_missing_world_without_advancing_rng() {
    let model = PolicyModel::fresh(9971005).expect("model");
    let mut random = [PpoRng::new(7)];
    let before = random.clone();
    let Err(error) = sample_active(&model, &mut random, &mut [], &[0]) else {
        panic!("reference sampling accepted a missing world");
    };
    assert_eq!(error, PpoError::InvalidConfig("parallel episode jobs"));
    assert_eq!(random, before);
}

pub(crate) fn parity_settings_for_test() -> TrainingJobConfig {
    crate::cli::training_settings_for_test(&[
        "--complete-episodes",
        "--map",
        "2",
        "--environments",
        "2",
        "--rollout",
        &crate::MAP2_RETAINED_DECISIONS.to_string(),
        "--minibatch",
        "512",
        "--gamma-per-tick",
        "1",
    ])
    .expect("episode settings")
}

pub(crate) fn assert_batch_parity_for_test(source: &PpoBatch, target: &PpoBatch) {
    assert_eq!(source.len(), target.len());
    assert_ne!(source.len(), 0);
    for index in 0..source.len() {
        let left = source.sample(index).expect("source sample");
        let right = target.sample(index).expect("target sample");
        assert_eq!(left.transition.frame, right.transition.frame);
        assert_eq!(left.transition.target, right.transition.target);
        assert_eq!(left.transition.action, right.transition.action);
        let order =
            |row: &crate::PpoTransition| (row.stream, row.decision, row.ticks, row.terminal);
        assert_eq!(order(&left.transition), order(&right.transition));
        let bits = |sample: &crate::PpoPreparedSample| {
            [
                sample.transition.old_log_probability,
                sample.transition.old_value,
                sample.transition.next_value,
                sample.transition.reward,
                sample.advantage(),
                sample.return_value(),
            ]
            .map(f32::to_bits)
        };
        assert_eq!(bits(&left), bits(&right));
    }
}

pub(crate) fn assert_retention_actor_parity_for_test(model: &PolicyModel) {
    use std::hash::Hasher;
    let source = actor_trial(model, 0);
    assert_eq!(source.batch.len(), 4);
    for phase in 1..RETENTION_STRIDE {
        let target = actor_trial(model, phase);
        assert_eq!(source.random, target.random);
        assert_eq!(target.batch.len(), 2);
        for (source, target) in source.states.iter().zip(&target.states) {
            assert_eq!(source.trace.finish(), target.trace.finish());
            assert_eq!(source.map2_reward, target.map2_reward);
        }
        for index in 0..2 {
            let sample = target
                .batch
                .sample(index)
                .expect("retained interval")
                .transition;
            assert_eq!(sample.ticks, 24);
            assert!((f64::from(sample.reward) - target.rewards[sample.stream]).abs() < 1e-8);
        }
    }
}

struct ActorTrial {
    batch: PpoBatch,
    random: Vec<PpoRng>,
    states: Vec<EpisodeStream>,
    rewards: Vec<f64>,
}

fn actor_trial(model: &PolicyModel, phase: usize) -> ActorTrial {
    assert!(phase < RETENTION_STRIDE);
    let settings = parity_settings_for_test();
    let count = settings.ppo.environments;
    assert_eq!(count, 2);
    let mut arenas = environments(&settings, 3).expect("worlds");
    let mut random = actor_stream_rngs(&mut PpoRng::new(9971001), count).expect("actor RNGs");
    let mut states: Vec<_> = (0..count)
        .map(|_| EpisodeStream {
            retention_phase: phase,
            ..EpisodeStream::default()
        })
        .collect();
    let mut rewards = vec![0.0; count];
    let mut rollout =
        PpoRollout::new(count * 2, model.policy_identity().expect("identity")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    for decision in 0..16 {
        let active: Vec<_> = (0..count).filter(|&stream| !states[stream].done).collect();
        if active.is_empty() {
            break;
        }
        let (choices, spaces) =
            sample_active(model, &mut random, &mut arenas, &active).expect("choices");
        for ((stream, choice), space) in active.into_iter().zip(choices).zip(spaces) {
            let previous = states[stream].map2_reward.total;
            advance_stream(
                model,
                &mut arenas[stream],
                &mut states[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut report,
            )
            .expect("native action");
            if (phase..phase + RETENTION_STRIDE).contains(&decision) {
                rewards[stream] += states[stream].map2_reward.total - previous;
            }
            if decision < phase {
                assert!(states[stream].choice.is_none());
                assert_eq!(states[stream].interval.steps, 0);
            }
        }
    }
    ActorTrial {
        batch: rollout.finish(settings.ppo).expect("batch"),
        random,
        states,
        rewards,
    }
}

pub(crate) fn assert_retention_boundaries_for_test(_: &PolicyModel) {
    let model = PolicyModel::fresh(9971003).expect("nonzero value model");
    let settings = parity_settings_for_test();
    let mut arena = environments(&settings, 0).expect("worlds").remove(0);
    let choice = sample_policy(&model, &mut PpoRng::new(9971004), &mut arena).expect("choice");
    let bootstrap = model
        .evaluate_batch(&[encode_next_frame(&mut arena).expect("frame")])
        .expect("value")[0]
        .value;
    assert_ne!(bootstrap, 0.0);
    for terminal in [true, false] {
        let mut state = EpisodeStream {
            retention_phase: 3,
            ..EpisodeStream::default()
        };
        let mut rollout = PpoRollout::new(1, choice.policy()).expect("rollout");
        state
            .append_retained_reward(1000.0, 9, settings.ppo.gamma_tick)
            .expect("discarded warmup");
        state.choice = Some(choice.clone());
        state
            .append_retained_reward(10.0, 14, settings.ppo.gamma_tick)
            .expect("partial interval");
        state.done = true;
        flush(&model, &mut arena, &mut state, 0, terminal, &mut rollout).expect("partial flush");
        let sample = rollout
            .finish(settings.ppo)
            .expect("batch")
            .sample(0)
            .expect("sample")
            .transition;
        assert_eq!(sample.ticks, 14);
        assert_eq!(sample.terminal, terminal);
        assert_eq!(sample.next_value, if terminal { 0.0 } else { bootstrap });
        assert_eq!(sample.reward, 10.0);
    }
    assert_unsampled_terminal(&choice);
}

fn assert_unsampled_terminal(choice: &PpoPolicyChoice) {
    let mut short = EpisodeStream {
        retention_phase: 7,
        ..EpisodeStream::default()
    };
    short
        .append_retained_reward(1.0, 3, 1.0)
        .expect("unretained reward");
    short.map2_reward = Map2TrainingReward {
        ticks: 3,
        terminal: 1.0,
        total: 1.0,
        ..Map2TrainingReward::default()
    };
    short.decisions = 1;
    short.done = true;
    assert!(!short.should_flush());
    assert!(short.choice.is_none());
    assert_eq!(short.interval.steps, 0);
    let mut report = PpoSmokeReport::default();
    record_episode(
        0,
        4,
        &short,
        Some(PpoTerminalOutcome::Win),
        &mut report,
        "Teacher",
    )
    .expect("outcome");
    assert_eq!(report.terminal_wins, 1);
    assert_eq!(report.map2_reward, short.map2_reward);
    assert_eq!(
        validate_episode_batch(
            &PpoRollout::new(1, choice.policy()).expect("empty"),
            &report
        )
        .expect_err("no fabricated sample")
        .to_string(),
        "PPO rollout is empty"
    );
}

pub(crate) fn assert_reset_loses_terminal_credit_for_test() {
    let settings = TrainingJobConfig {
        complete_episodes: false,
        ..parity_settings_for_test()
    };
    let model = PolicyModel::fresh(9911000).expect("model");
    let mut initial =
        build_training_environments(&settings, 0, settings.ppo, &model, None).expect("windows");
    let start = initial[0].seats[0].tracker.current().expect("start").tick;
    advance_interval(&mut initial[0], vec![None, None], 3).expect("progress");
    let rebuilt =
        build_training_environments(&settings, 8, settings.ppo, &model, None).expect("new windows");
    assert_eq!(
        rebuilt[0].seats[0].tracker.current().expect("rebuilt").tick,
        start
    );
    assert!(
        initial[0].seats[0]
            .tracker
            .current()
            .expect("progressed")
            .tick
            > start
    );
}

pub(crate) fn assert_full_mc_for_test() {
    let model = PolicyModel::fresh(9921000).expect("model");
    let settings = parity_settings_for_test();
    let mut arena = environments(&settings, 0).expect("worlds").remove(0);
    let choice = sample_policy(&model, &mut PpoRng::new(9921003), &mut arena).expect("choice");
    assert_ne!(
        choice.value, 99.0,
        "intermediate bootstraps must not determine MC returns"
    );
    let config = PpoConfig {
        gae_lambda: 1.0,
        ..settings.ppo
    };
    let mut rollout = PpoRollout::new(2 * RETAINED_PER_EPISODE, choice.policy()).expect("rollout");
    for decision in 0..RETAINED_PER_EPISODE {
        let terminal = decision + 1 == RETAINED_PER_EPISODE;
        let ticks = (ACTOR_DECISIONS - decision * RETENTION_STRIDE).min(RETENTION_STRIDE) as u32
            * 3
            - u32::from(terminal);
        for stream in 0..2 {
            let reward = if terminal {
                if stream == 0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            rollout
                .push(
                    choice
                        .clone()
                        .finish(PpoOutcome {
                            stream,
                            decision: decision as u32,
                            ticks,
                            reward,
                            terminal,
                            next_value: if terminal { 0.0 } else { 99.0 },
                        })
                        .expect("transition"),
                )
                .expect("push");
        }
    }
    let batch = rollout.finish(config).expect("MC batch");
    assert_mc_batch_and_timeout(&batch, choice, config);
}

fn assert_mc_batch_and_timeout(batch: &PpoBatch, choice: PpoPolicyChoice, config: PpoConfig) {
    assert_eq!(batch.len(), 2 * RETAINED_PER_EPISODE);
    for index in 0..batch.len() {
        let sample = batch.sample(index).expect("sample");
        let expected = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
        assert!((sample.return_value() - expected).abs() < 1e-5);
        let terminal = index / 2 + 1 == RETAINED_PER_EPISODE;
        assert_eq!(sample.transition.terminal, terminal);
        assert_eq!(sample.transition.ticks, if terminal { 11 } else { 24 });
    }
    let mut truncated = PpoRollout::new(1, choice.policy()).expect("truncated");
    truncated
        .push(
            choice
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 11,
                    next_value: 0.7,
                    reward: 0.0,
                    terminal: false,
                })
                .expect("timeout"),
        )
        .expect("push");
    let sample = truncated
        .finish(config)
        .expect("timeout MC")
        .sample(0)
        .expect("sample");
    assert!(!sample.transition.terminal);
    assert!((sample.return_value() - 0.7).abs() < 1e-5);
}

#[test]
fn wide_capacity_actor_streams_preserve_rng_order_and_reject_overflow_without_draws() {
    let mut master = PpoRng::new(7);
    let mut expected = master.clone();
    let random = actor_stream_rngs(&mut master, 64).expect("sixty-four actor RNGs");
    assert_eq!(random.len(), 64);
    for state in &random {
        assert_eq!(*state, PpoRng::new(expected.next_word().expect("seed")));
    }
    assert_eq!(master, expected);
    assert_eq!(
        validate_active(&random, &(0..64).collect::<Vec<_>>()),
        Ok(())
    );
    assert_eq!(
        validate_active(&random, &[63, 64]),
        Err(PpoError::InvalidConfig("episode active streams"))
    );
    assert_eq!(
        actor_stream_rngs(&mut master, 65),
        Err(PpoError::InvalidConfig("actor RNG streams"))
    );
    assert_eq!(master, expected);
    assert!(valid_environment_count(26));
    assert!(
        !valid_environment_count(40),
        "standard train-full remains bounded at 26"
    );
}
