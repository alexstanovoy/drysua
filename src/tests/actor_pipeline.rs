use std::hash::Hasher;
use std::sync::{
    Barrier,
    atomic::{AtomicUsize, Ordering},
};

use super::*;

type Trial = (
    crate::PpoBatch,
    Vec<PpoRng>,
    Vec<(u64, Map2TrainingReward)>,
    PpoSmokeReport,
);

#[test]
fn actor_pipeline_dispatches_the_other_group_before_waiting_for_advance() {
    for groups in [2, 4] {
        assert_all_groups_dispatched(groups);
    }
}

fn assert_all_groups_dispatched(group_count: usize) {
    let model = PolicyModel::fresh(9952600).expect("model");
    let (config, mut groups) = inputs_grouped(1, group_count);
    let gate = Barrier::new(group_count);
    let owner = std::thread::current().id();
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let mut report = PpoSmokeReport::default();

    collect_with_operation(
        &model,
        config,
        &mut groups,
        9,
        &mut rollout,
        &mut report,
        true,
        |_, world, job| {
            assert_ne!(std::thread::current().id(), owner);
            if matches!(&job, StreamJob::Advance { state, .. } if state.decisions == 0) {
                // No group's first advance can finish until every group has dispatched.
                gate.wait();
            }
            run_stream_job(world, job, config, None, None)
        },
    )
    .expect("overlapped native collection");

    assert_eq!(groups[0].streams[0].decisions, 9);
    assert_eq!(groups[1].streams[0].decisions, 8);
    assert_eq!(report.terminal_draws, 1);
    assert_eq!(rollout.len(), 2);
}

#[test]
fn actor_pipeline_matches_sequential_b20_and_b32_rows_rng_rewards_and_order() {
    assert_parity(PolicyDevice::Cpu);
}

#[test]
fn actor_pipeline_g4_matches_g1_and_g2_at_fixed_b10_and_b16() {
    assert_four_group_parity(PolicyDevice::Cpu);
}

#[test]
fn actor_pipeline_rejects_three_group_waves_before_sampling() {
    let (config, mut groups) = inputs_grouped(1, 4);
    groups.pop();
    let random: Vec<_> = groups.iter().map(|group| group.random.clone()).collect();
    assert_eq!(
        validate_groups(config, &groups, 16),
        Err(PpoError::InvalidConfig("actor pipeline groups or rounds"))
    );
    assert_eq!(
        groups
            .iter()
            .map(|group| group.random.clone())
            .collect::<Vec<_>>(),
        random
    );
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires the authorized exclusive CUDA runner"]
fn cuda_actor_pipeline_matches_sequential_fixed_batches() {
    assert_parity(PolicyDevice::Cuda { ordinal: 0 });
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires the authorized exclusive CUDA runner; no actor graph"]
fn cuda_actor_pipeline_g4_matches_fixed_batch_controls() {
    assert_four_group_parity(PolicyDevice::Cuda { ordinal: 0 });
}

#[test]
fn actor_pipeline_worker_failure_joins_other_group_without_retry() {
    for groups in [2, 4] {
        assert_worker_failure(groups);
    }
}

fn assert_worker_failure(group_count: usize) {
    let model = PolicyModel::fresh(9952600).expect("model");
    let expected_random = two_round_random(&model, group_count);
    let (config, mut groups) = inputs_grouped(1, group_count);
    let completed = AtomicUsize::new(0);
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let mut report = PpoSmokeReport::default();

    let result = collect_with_operation(
        &model,
        config,
        &mut groups,
        9,
        &mut rollout,
        &mut report,
        true,
        |stream, world, job| {
            if matches!(&job, StreamJob::Advance { state, .. } if stream == 0 && state.decisions == 1)
            {
                return Err(PpoError::InvalidTransition("pipeline worker failure"));
            }
            let advancing = matches!(&job, StreamJob::Advance { .. });
            let reply = run_stream_job(world, job, config, None, None)?;
            completed.fetch_add(usize::from(advancing), Ordering::Relaxed);
            Ok(reply)
        },
    );

    assert_eq!(
        result,
        Err(PpoError::InvalidTransition("pipeline worker failure"))
    );
    assert_eq!(completed.load(Ordering::Relaxed), group_count * 2 - 1);
    assert!(rollout.is_empty());
    let random: Vec<_> = groups
        .iter()
        .flat_map(|group| group.random.iter().cloned())
        .collect();
    assert_eq!(
        random, expected_random,
        "failed advances must not retry or sample again"
    );
}

fn two_round_random(model: &PolicyModel, group_count: usize) -> Vec<PpoRng> {
    let (config, mut groups) = inputs_grouped(1, group_count);
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    for group in &mut groups {
        collect_batch_with_actor_values(
            model,
            config,
            group.stream_base,
            "reference",
            &mut group.environments,
            &mut group.streams,
            &mut group.random,
            2,
            &mut rollout,
            &mut report,
            true,
        )
        .expect("reference sampling");
    }
    groups.into_iter().flat_map(|group| group.random).collect()
}

#[test]
fn actor_pipeline_rejects_misaligned_group_buckets_before_sampling() {
    let (mut config, mut groups) = inputs(2);
    config.environments = 8;
    for group in &mut groups {
        group.stream_base += 1;
    }
    let random: Vec<_> = groups.iter().map(|group| group.random.clone()).collect();
    assert_eq!(
        validate_groups(config, &groups, 16),
        Err(PpoError::InvalidConfig("actor pipeline group streams"))
    );
    assert_eq!(
        groups
            .iter()
            .map(|group| group.random.clone())
            .collect::<Vec<_>>(),
        random
    );
    assert!(
        groups
            .iter()
            .all(|group| group.streams.iter().all(|state| state.decisions == 0))
    );
}

#[test]
fn actor_pipeline_memory_gate_keeps_existing_limits_and_rejects_gpu_opponents() {
    for width in [20, 32] {
        let (config, _) = inputs_config(width);
        assert_eq!(validate_pipeline_memory(config, width, 2), Ok(()));
        assert!(pipeline_payload_bytes(config, width) <= 12 * 1024 * 1024 * 1024);
    }
    let (mut config, _) = inputs_config(20);
    config.environments = 80;
    config.sample_budget = crate::PpoSampleBudget::WideAnnealed;
    assert_eq!(validate_pipeline_memory(config, 20, 2), Ok(()));
    assert!(pipeline_payload_bytes(config, 20) <= crate::PPO_WIDE_ANNEALED_PAYLOAD_BOUND_BYTES);
    let (config, _) = inputs_config(40);
    assert_eq!(
        validate_pipeline_memory(config, 40, 2),
        Err(PpoError::InvalidConfig(
            "actor pipeline active worlds exceed 64"
        ))
    );
    let model = Arc::new(PolicyModel::fresh(9952600).expect("model"));
    let (config, mut groups) = inputs(1);
    groups[1].environments[0].opponent = OpponentRuntime::Policy {
        model: Arc::clone(&model),
        rng: PpoRng::new(1),
    };
    let random = groups[0].random.clone();
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let result = collect_actor_pipeline(
        &model,
        config,
        &mut groups,
        1,
        &mut rollout,
        &mut PpoSmokeReport::default(),
        true,
    );
    assert_eq!(
        result,
        Err(PpoError::InvalidConfig(
            "actor pipeline requires CPU opponents"
        ))
    );
    assert_eq!(groups[0].random, random);
    assert!(
        groups
            .iter()
            .all(|group| group.streams.iter().all(|stream| stream.decisions == 0))
    );
}

fn assert_parity(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9952600, device).expect("model");
    for (width, rounds) in [(20, 16), (32, 16), (2, 24)] {
        for reuse in [false, true] {
            let source = trial(&model, width, rounds, reuse, false);
            let target = trial(&model, width, rounds, reuse, true);
            assert_eq!(target.1, source.1);
            assert_eq!(target.2, source.2);
            assert_eq!(target.3, source.3);
            assert_batch_parity_for_test(&source.0, &target.0);
        }
    }
    for reuse in [false, true] {
        let source = multiwave_trial(&model, reuse, false);
        let target = multiwave_trial(&model, reuse, true);
        assert_eq!(target.1, source.1);
        assert_eq!(target.2, source.2);
        assert_eq!(target.3, source.3);
        assert_batch_parity_for_test(&source.0, &target.0);
    }
}

fn assert_four_group_parity(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9952600, device).expect("model");
    for width in [10, 16] {
        for reuse in [false, true] {
            let source = trial_grouped(&model, width, 16, reuse, 1, 4);
            for groups in [2, 4] {
                let target = trial_grouped(&model, width, 16, reuse, groups, 4);
                assert_eq!(target.1, source.1);
                assert_eq!(target.2, source.2);
                assert_eq!(target.3, source.3);
                assert_batch_parity_for_test(&source.0, &target.0);
            }
        }
    }
}

fn multiwave_trial(model: &PolicyModel, reuse: bool, pipeline: bool) -> Trial {
    let (mut config, _) = inputs_config(20);
    config.environments = 80;
    config.sample_budget = crate::PpoSampleBudget::WideAnnealed;
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    let mut master = PpoRng::new(9952701);
    let mut random = Vec::new();
    let mut traces = Vec::new();
    for wave in 0..2 {
        let (_, mut groups) = inputs(20);
        for group in &mut groups {
            group.stream_base += wave * 40;
            group.random = actor_stream_rngs(&mut master, 20).expect("canonical seeds");
        }
        if pipeline {
            collect_actor_pipeline(
                model,
                config,
                &mut groups,
                16,
                &mut rollout,
                &mut report,
                reuse,
            )
            .expect("wave");
        } else {
            for group in &mut groups {
                collect_batch_with_actor_values(
                    model,
                    config,
                    group.stream_base,
                    "serial",
                    &mut group.environments,
                    &mut group.streams,
                    &mut group.random,
                    16,
                    &mut rollout,
                    &mut report,
                    reuse,
                )
                .expect("sequential wave");
            }
        }
        random.extend(groups.iter().flat_map(|group| group.random.iter().cloned()));
        traces.extend(groups.iter().flat_map(|group| {
            group
                .streams
                .iter()
                .map(|state| (state.trace.finish(), state.map2_reward))
        }));
    }
    let batch = rollout.finish(config).expect("canonical multiwave batch");
    let buckets: Vec<_> = (0..batch.len())
        .map(|index| batch.sample(index).expect("row").transition.stream / 20)
        .collect();
    assert!(buckets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(buckets.first(), Some(&0));
    assert_eq!(buckets.last(), Some(&3));
    (batch, random, traces, report)
}

fn trial(model: &PolicyModel, width: usize, rounds: usize, reuse: bool, pipeline: bool) -> Trial {
    trial_grouped(model, width, rounds, reuse, if pipeline { 2 } else { 1 }, 2)
}

fn trial_grouped(
    model: &PolicyModel,
    width: usize,
    rounds: usize,
    reuse: bool,
    wave_groups: usize,
    total_groups: usize,
) -> Trial {
    let (config, mut groups) = inputs_grouped(width, total_groups);
    let mut rollout =
        PpoRollout::for_config(config, model.policy_identity().expect("policy")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    for wave in groups.chunks_mut(wave_groups) {
        if wave_groups > 1 {
            collect_actor_pipeline(
                model,
                config,
                wave,
                rounds,
                &mut rollout,
                &mut report,
                reuse,
            )
            .expect("pipeline");
        } else {
            let group = &mut wave[0];
            collect_batch_with_actor_values(
                model,
                config,
                group.stream_base,
                "serial",
                &mut group.environments,
                &mut group.streams,
                &mut group.random,
                rounds,
                &mut rollout,
                &mut report,
                reuse,
            )
            .expect("sequential");
        }
    }
    let random = groups
        .iter()
        .flat_map(|group| group.random.iter().cloned())
        .collect();
    let traces = groups
        .iter()
        .flat_map(|group| {
            group
                .streams
                .iter()
                .map(|state| (state.trace.finish(), state.map2_reward))
        })
        .collect();
    (
        rollout.finish(config).expect("batch"),
        random,
        traces,
        report,
    )
}

fn inputs_config(width: usize) -> (PpoConfig, usize) {
    let count = width * 2;
    let config = PpoConfig {
        environments: count,
        rollout_decisions: RETAINED_PER_EPISODE,
        sample_budget: crate::PpoSampleBudget::for_annealed_games(count),
        decision_interval_ticks: 3,
        gamma_tick: 1.0,
        ..PpoConfig::default()
    };
    (config, count)
}

fn inputs(width: usize) -> (PpoConfig, Vec<ActorGroup>) {
    inputs_grouped(width, 2)
}

fn inputs_grouped(width: usize, group_count: usize) -> (PpoConfig, Vec<ActorGroup>) {
    assert!(matches!(group_count, 2 | 4));
    let (mut config, _) = inputs_config(width);
    let count = width * group_count;
    config.environments = count;
    config.sample_budget = crate::PpoSampleBudget::for_annealed_games(count);
    assert!(count <= MAX_ACTOR_ENVIRONMENTS);
    let mut master = PpoRng::new(9952601);
    let groups = (0..group_count)
        .map(|group| {
            let stream_base = group * width;
            ActorGroup {
                stream_base,
                environments: (0..width)
                    .map(|stream| {
                        super::super::map2_tests::configured_environment(
                            TICK_CAP - [72, 24, 48][(stream_base + stream) % 3],
                            stream % 2,
                            OpponentSpec::Weak,
                            |_| {},
                        )
                    })
                    .collect(),
                streams: (0..width)
                    .map(|stream| EpisodeStream {
                        retention_phase: (stream_base + stream) % RETENTION_STRIDE,
                        ..EpisodeStream::default()
                    })
                    .collect(),
                random: actor_stream_rngs(&mut master, width).expect("per-group RNG seeds"),
            }
        })
        .collect();
    (config, groups)
}
