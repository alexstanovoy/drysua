use std::hash::Hasher;

use super::*;
use crate::PpoBatch;

#[test]
fn actor_value_reuse_preserves_b40_actions_rng_and_mixed_terminal_sample_order() {
    assert_reuse_collection(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires the authorized exclusive CUDA runner"]
fn cuda_actor_value_reuse_preserves_collection_and_reports_bootstrap_error() {
    assert_reuse_collection(PolicyDevice::Cuda { ordinal: 0 });
}

#[test]
fn actor_value_reuse_short_stop_uses_batch_one_without_extra_actor_draws() {
    let model = PolicyModel::fresh(9952200).expect("model");
    let opponent = Arc::new(PolicyModel::fresh(9952201).expect("frozen opponent"));
    let source = trial(&model, 2, 8, false, Some(&opponent));

    let target = trial(&model, 2, 8, true, Some(&opponent));

    assert_eq!(target.random, source.random);
    assert_eq!(target.traces, source.traces);
    assert_eq!(target.report, source.report);
    assert_eq!(target.batch.len(), 2);
    assert_batch_parity_for_test(&source.batch, &target.batch);
}

#[test]
fn actor_value_reuse_completion_failure_does_not_commit_sampled_rng_or_advance_again() {
    let model = PolicyModel::fresh(9952200).expect("model");
    let config = parity_settings_for_test().ppo;
    let mut worlds: Vec<_> = (0..2)
        .map(|stream| {
            map2_tests::configured_environment(TICK_CAP - 72, stream, OpponentSpec::Idle, |_| {})
        })
        .collect();
    let mut random = actor_stream_rngs(&mut PpoRng::new(9952202), 2).expect("actor streams");
    let mut expected = random.clone();
    sample_active(&model, &mut expected, &mut worlds, &[0, 1]).expect("first-round RNG oracle");
    let mut states: Vec<_> = (0..2).map(|_| EpisodeStream::default()).collect();
    let mut rollout = PpoRollout::for_config(config, model.policy_identity().expect("identity"))
        .expect("rollout");
    let mut report = CollectionReport {
        elapsed_ticks: u64::MAX,
        ..Default::default()
    };

    let result = collect_batch_with_actor_values(
        &model,
        config,
        0,
        "actor-values",
        &mut worlds,
        &mut states,
        &mut random,
        2,
        &mut rollout,
        &mut report,
        true,
    );

    assert_eq!(result, Err(PpoError::CounterOverflow));
    assert_eq!(random, expected);
    assert!(states.iter().all(|state| state.decisions == 1));
    assert!(
        worlds
            .iter()
            .all(|world| world.arena.tick() == TICK_CAP - 69)
    );
    assert!(rollout.is_empty());
}

fn assert_reuse_collection(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9952200, device).expect("model");
    for (count, rounds) in [(40, 16), (4, 24)] {
        let source = trial(&model, count, rounds, false, None);

        let target = trial(&model, count, rounds, true, None);

        assert_eq!(target.random, source.random);
        assert_eq!(target.traces, source.traces);
        assert_eq!(target.report, source.report);
        let terminals = (0..count)
            .filter(|stream| [24, 8, 16][stream % 3] <= rounds)
            .count();
        assert_eq!(target.report.terminal_draws, terminals as u64);
        assert_eq!(target.report.rejected_orders, 0);
        assert_reused_samples(&source.batch, &target.batch, count, device);
    }
}

struct Trial {
    batch: PpoBatch,
    random: Vec<PpoRng>,
    traces: Vec<(u64, [u32; ActionKind::COUNT], usize)>,
    report: CollectionReport,
}

fn trial(
    model: &PolicyModel,
    count: usize,
    rounds: usize,
    reuse: bool,
    opponent: Option<&Arc<PolicyModel>>,
) -> Trial {
    assert!(count <= MAX_ACTOR_ENVIRONMENTS);
    let mut config = parity_settings_for_test().ppo;
    config.environments = count;
    let mut worlds: Vec<_> = (0..count)
        .map(|stream| {
            let opponent = opponent.map_or(OpponentSpec::Idle, |model| {
                OpponentSpec::SharedPolicy(Arc::clone(model))
            });
            map2_tests::configured_environment(
                TICK_CAP - [72, 24, 48][stream % 3],
                stream % 2,
                opponent,
                |_| {},
            )
        })
        .collect();
    let mut states: Vec<_> = (0..count)
        .map(|stream| EpisodeStream {
            retention_phase: stream % RETENTION_STRIDE,
            ..EpisodeStream::default()
        })
        .collect();
    let mut random = actor_stream_rngs(&mut PpoRng::new(9952202), count).expect("actor streams");
    let mut rollout = PpoRollout::for_config(config, model.policy_identity().expect("identity"))
        .expect("rollout");
    let mut report = CollectionReport::default();

    collect_batch_with_actor_values(
        model,
        config,
        0,
        "actor-values",
        &mut worlds,
        &mut states,
        &mut random,
        rounds,
        &mut rollout,
        &mut report,
        reuse,
    )
    .expect("native collection");

    assert_eq!(environment_rejections(&worlds).expect("rejections"), 0);
    Trial {
        batch: rollout.finish(config).expect("retained batch"),
        random,
        traces: states
            .iter()
            .map(|state| (state.trace.finish(), state.actions, state.decisions))
            .collect(),
        report,
    }
}

fn assert_reused_samples(source: &PpoBatch, target: &PpoBatch, count: usize, device: PolicyDevice) {
    assert_eq!(target.len(), source.len());
    let mut previous = vec![None; count];
    let mut linked = 0;
    let mut different = 0;
    let mut maximum_error = 0.0f32;
    let mut largest = None;
    for index in 0..target.len() {
        let source = source.sample(index).expect("reference sample").transition;
        let target = target.sample(index).expect("reuse sample").transition;
        assert_eq!(target.frame, source.frame);
        assert_eq!(target.target, source.target);
        assert_eq!(target.action, source.action);
        assert_eq!(target.policy, source.policy);
        let order =
            |row: &crate::PpoTransition| (row.stream, row.decision, row.ticks, row.terminal);
        assert_eq!(order(&target), order(&source));
        assert_eq!(
            [target.reward, target.old_value, target.old_log_probability].map(f32::to_bits),
            [source.reward, source.old_value, source.old_log_probability].map(f32::to_bits)
        );
        assert!(target.next_value.is_finite());
        assert!(source.next_value.is_finite());
        let error = (target.next_value - source.next_value).abs();
        assert!(
            error <= 1.0e-4 * (1.0 + source.next_value.abs()),
            "bootstrap error {error}"
        );
        different += usize::from(target.next_value.to_bits() != source.next_value.to_bits());
        if error > maximum_error {
            maximum_error = error;
            let row = (target.stream, target.decision);
            largest = Some((row, source.next_value, target.next_value));
        }
        if let Some((decision, value)) = previous[target.stream] {
            assert_eq!(target.decision, decision + 1);
            assert_eq!(target.old_value.to_bits(), value);
            linked += 1;
        }
        if target.terminal {
            assert_eq!(target.next_value.to_bits(), 0);
        }
        previous[target.stream] = Some((target.decision, target.next_value.to_bits()));
    }
    assert!(linked > 0, "no reused actor boundary");
    eprintln!(
        "actor-value-reuse device={device:?} worlds={count} samples={} differing_bootstraps={different} max_absolute_error={maximum_error} largest={largest:?}",
        target.len()
    );
}
