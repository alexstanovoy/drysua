use std::hash::Hasher;

use super::*;

#[test]
fn neural_opponent_collection_matches_same_batch_reference_and_preserves_frozen_weights() {
    assert_reference(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires exclusive authorized CUDA verification"]
fn cuda_neural_opponent_collection_matches_same_batch_reference() {
    assert_reference(PolicyDevice::Cuda { ordinal: 0 });
}

#[test]
fn neural_opponent_forced_continue_preserves_scalar_cadence_and_draws() {
    let learner = PolicyModel::fresh(9952800).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh(9952801).expect("opponent"));
    for model in [&learner, opponent.as_ref()] {
        let mut parameters = vec![0.0; model.parameter_count()];
        let mut offset = 0;
        for (name, shape) in model.parameter_schema().expect("schema") {
            if matches!(name, "kind.bias" | "dire.kind.bias") {
                parameters[offset] = 100.0;
            }
            offset += shape.iter().product::<usize>();
        }
        model
            .import_parameters(&parameters)
            .expect("Continue policy");
    }
    let scalar = trial(&learner, &opponent, Mode::Scalar, false);
    let batched = trial(&learner, &opponent, Mode::Batched, false);
    assert_batch_parity_for_test(&scalar.0, &batched.0);
    assert_eq!(
        (scalar.1, scalar.2, scalar.3),
        (batched.1, batched.2, batched.3)
    );
}

fn assert_reference(device: PolicyDevice) {
    let learner = PolicyModel::fresh_on(9952800, device).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh_on(9952801, device).expect("opponent"));
    let before = opponent.export_parameters().expect("frozen parameters");
    let identity = opponent.policy_identity().expect("frozen identity");
    let source = trial(&learner, &opponent, Mode::Reference, false);
    neural_opponent::take_batch_counts();

    let target = trial(&learner, &opponent, Mode::Batched, false);

    assert_eq!(neural_opponent::take_batch_counts(), (16, 56));
    assert_batch_parity_for_test(&source.0, &target.0);
    assert_eq!(
        (source.1, source.2, source.3),
        (target.1, target.2, target.3)
    );
    assert_eq!(opponent.policy_identity().expect("identity"), identity);
    crate::tests::support::assert_bits(&before, &opponent.export_parameters().expect("weights"));
    let first = trial(&learner, &opponent, Mode::Batched, true);
    let second = trial(&learner, &opponent, Mode::Batched, true);
    assert_batch_parity_for_test(&first.0, &second.0);
    assert_eq!((first.1, first.2, first.3), (second.1, second.2, second.3));
}

#[test]
fn neural_opponent_bad_frame_stale_space_and_changed_policy_do_not_commit_world_rngs() {
    let learner = PolicyModel::fresh(9952800).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh(9952801).expect("opponent"));
    let (_, mut group) = inputs(&opponent, 0, 4);
    let batch = neural_opponent::OpponentBatch::new(&learner, &group.environments).expect("owner");
    let before = opponent_rngs(&group);
    for problem in 0..3 {
        let mut inputs = group
            .environments
            .iter_mut()
            .map(neural_opponent::prepare)
            .collect::<Result<Vec<_>, _>>()
            .expect("prepared opponents");
        let message = match problem {
            0 => {
                inputs[3].frame.global[0] = f32::NAN;
                "prepared opponent frame is not finite"
            }
            1 => {
                let (left, right) = inputs.split_at_mut(1);
                std::mem::swap(&mut left[0].space, &mut right[0].space);
                "prepared opponent action space mismatch"
            }
            _ => {
                opponent
                    .import_parameters(&opponent.export_parameters().expect("export"))
                    .expect("new revision");
                "opponent batch policy identity changed"
            }
        };
        neural_opponent::take_batch_counts();
        assert_eq!(
            batch.sample(inputs).err(),
            Some(PpoError::InvalidTransition(message))
        );
        assert_eq!(neural_opponent::take_batch_counts(), (0, 0));
        assert_eq!(opponent_rngs(&group), before);
    }
}

#[test]
fn neural_opponent_coherent_packet_from_another_world_is_rejected_before_bookkeeping() {
    let learner = PolicyModel::fresh(9952800).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh(9952801).expect("opponent"));
    let (_, mut group) = inputs(&opponent, 0, 2);
    let batch = neural_opponent::OpponentBatch::new(&learner, &group.environments).expect("batch");
    let (frame, space) = prepare_policy_sample(&mut group.environments[0]).expect("learner frame");
    let choice = learner
        .sample(&frame, &space, &mut PpoRng::new(19))
        .expect("learner choice");
    let prepared = group
        .environments
        .iter_mut()
        .map(neural_opponent::prepare)
        .collect::<Result<Vec<_>, _>>()
        .expect("opponent inputs");
    let before = opponent_rngs(&group);
    assert_eq!(
        before[0], before[1],
        "same RNG must not hide foreign lineage"
    );
    let foreign = batch
        .sample(prepared)
        .expect("opponent batch")
        .pop()
        .expect("other world");
    let local = group.environments[0]
        .seats
        .iter()
        .map(|seat| seat.local)
        .collect::<Vec<_>>();

    let error =
        requests_for_prepared_opponent(&mut group.environments[0], &choice, &space, foreign)
            .expect_err("foreign packet");

    assert_eq!(
        error,
        PpoError::InvalidTransition("prepared neural decision seat changed")
    );
    assert_eq!(opponent_rngs(&group), before);
    assert_eq!(
        group.environments[0]
            .seats
            .iter()
            .map(|seat| seat.local)
            .collect::<Vec<_>>(),
        local
    );
    assert!(
        group.environments[0]
            .seats
            .iter()
            .all(|seat| seat.sequence == 0)
    );
}

#[test]
fn neural_opponent_failure_after_learner_sampling_commits_neither_rng_nor_world_advance() {
    let learner = PolicyModel::fresh(9952800).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh(9952801).expect("opponent"));
    let config = config(4);
    let mut groups = vec![inputs(&opponent, 0, 2).1, inputs(&opponent, 2, 2).1];
    let before = groups
        .iter()
        .flat_map(|group| group.random.iter().cloned().chain(opponent_rngs(group)))
        .collect::<Vec<_>>();
    let ticks = groups
        .iter()
        .flat_map(|group| group.environments.iter().map(|world| world.arena.tick()))
        .collect::<Vec<_>>();
    let mut rollout = PpoRollout::for_config(config, learner.policy_identity().expect("policy"))
        .expect("rollout");
    let owner = std::thread::current().id();
    neural_opponent::take_batch_counts();

    let result = actor_pipeline::collect_with_operation_mode(
        &learner,
        config,
        &mut groups,
        16,
        &mut rollout,
        &mut PpoSmokeReport::default(),
        true,
        true,
        |stream, world, job| {
            assert_ne!(std::thread::current().id(), owner);
            let mut reply = run_stream_job(world, job, config, None, None)?;
            if let StreamReply::PreparedNeural(_, opponent) = &mut reply
                && stream == 1
            {
                opponent.frame.global[0] = f32::NAN;
            }
            Ok(reply)
        },
    );

    assert_eq!(
        result,
        Err(PpoError::InvalidTransition(
            "prepared opponent frame is not finite"
        ))
    );
    assert_eq!(neural_opponent::take_batch_counts(), (0, 0));
    assert_eq!(
        groups
            .iter()
            .flat_map(|group| group.random.iter().cloned().chain(opponent_rngs(group)))
            .collect::<Vec<_>>(),
        before
    );
    assert_eq!(
        groups
            .iter()
            .flat_map(|group| group.environments.iter().map(|world| world.arena.tick()))
            .collect::<Vec<_>>(),
        ticks
    );
    assert!(rollout.is_empty());
}

#[test]
fn neural_opponent_pipeline_matches_sequential_groups_and_counts_one_call_per_group_round() {
    let learner = PolicyModel::fresh(9952800).expect("learner");
    let opponent = Arc::new(PolicyModel::fresh(9952801).expect("opponent"));
    for reuse in [false, true] {
        let config = config(8);
        let mut results = Vec::new();
        for pipelined in [false, true] {
            let mut groups = (0..4)
                .map(|index| inputs(&opponent, index * 2, 2).1)
                .collect::<Vec<_>>();
            let mut rollout =
                PpoRollout::for_config(config, learner.policy_identity().expect("policy"))
                    .expect("rollout");
            let mut report = PpoSmokeReport::default();
            neural_opponent::take_batch_counts();
            if pipelined {
                collect_actor_pipeline_with_opponent_batching(
                    &learner,
                    config,
                    &mut groups,
                    16,
                    &mut rollout,
                    &mut report,
                    reuse,
                    true,
                )
                .expect("four owner groups");
            } else {
                for group in &mut groups {
                    collect(
                        &learner,
                        config,
                        group,
                        &mut rollout,
                        &mut report,
                        reuse,
                        true,
                    );
                }
            }
            assert_eq!(neural_opponent::take_batch_counts().0, 64);
            let random = groups
                .iter()
                .flat_map(|group| group.random.iter().cloned().chain(opponent_rngs(group)))
                .collect::<Vec<_>>();
            let trace = groups
                .iter()
                .flat_map(|group| group.streams.iter().map(|state| state.trace.finish()))
                .collect::<Vec<_>>();
            results.push((
                rollout.finish(config).expect("batch"),
                random,
                trace,
                report,
            ));
        }
        assert_batch_parity_for_test(&results[0].0, &results[1].0);
        assert_eq!(
            (&results[0].1, &results[0].2, results[0].3),
            (&results[1].1, &results[1].2, results[1].3)
        );
    }
}

type Trial = (crate::PpoBatch, Vec<PpoRng>, Vec<u64>, PpoSmokeReport);

#[derive(PartialEq)]
enum Mode {
    Reference,
    Scalar,
    Batched,
}

fn trial(learner: &PolicyModel, opponent: &Arc<PolicyModel>, mode: Mode, reuse: bool) -> Trial {
    let (config, mut group) = inputs(opponent, 0, 4);
    let mut rollout = PpoRollout::for_config(config, learner.policy_identity().expect("policy"))
        .expect("rollout");
    let mut report = PpoSmokeReport::default();
    if mode != Mode::Reference {
        collect(
            learner,
            config,
            &mut group,
            &mut rollout,
            &mut report,
            reuse,
            mode == Mode::Batched,
        );
    } else {
        reference(
            learner,
            opponent,
            config,
            &mut group,
            &mut rollout,
            &mut report,
        );
    }
    let mut random = group.random.clone();
    random.extend(opponent_rngs(&group));
    let trace = group
        .streams
        .iter()
        .map(|state| state.trace.finish())
        .collect();
    (
        rollout.finish(config).expect("batch"),
        random,
        trace,
        report,
    )
}

fn collect(
    model: &PolicyModel,
    config: PpoConfig,
    group: &mut ActorGroup,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    reuse: bool,
    neural: bool,
) {
    collect_batch_with_opponent_batching(
        model,
        config,
        group.stream_base,
        "neural-test",
        &mut group.environments,
        &mut group.streams,
        &mut group.random,
        16,
        rollout,
        report,
        reuse,
        neural,
    )
    .expect("owner collection");
}

fn reference(
    model: &PolicyModel,
    opponent: &Arc<PolicyModel>,
    config: PpoConfig,
    group: &mut ActorGroup,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) {
    for _ in 0..16 {
        let active: Vec<_> = (0..group.streams.len())
            .filter(|&stream| !group.streams[stream].done)
            .collect();
        if active.is_empty() {
            break;
        }
        let (frames, spaces): (Vec<_>, Vec<_>) = active
            .iter()
            .map(|&stream| {
                prepare_policy_sample(&mut group.environments[stream]).expect("actor frame")
            })
            .unzip();
        let (choices, spaces) = select_choices(model, &mut group.random, &active, frames, spaces)
            .expect("learner batch");
        let opponents = reference_opponents(opponent, &mut group.environments, &active);
        for (((stream, choice), space), opponent) in
            active.into_iter().zip(choices).zip(spaces).zip(opponents)
        {
            let completed = advance_cpu_with_opponent(
                &mut group.environments[stream],
                &mut group.streams[stream],
                choice,
                space,
                config,
                Some(opponent),
            )
            .expect("native advance");
            finish_advance(
                model,
                &mut group.environments[stream],
                &mut group.streams[stream],
                group.stream_base + stream,
                completed,
                rollout,
                report,
            )
            .expect("reference flush");
        }
    }
}

fn reference_opponents(
    opponent: &Arc<PolicyModel>,
    environments: &mut [TrainingEnvironment],
    active: &[usize],
) -> Vec<neural_opponent::OpponentChoice> {
    let prepared = active
        .iter()
        .map(|&stream| neural_opponent::prepare(&mut environments[stream]).expect("opponent frame"))
        .collect::<Vec<_>>();
    let mut frames = Vec::new();
    let mut opponent_spaces = Vec::new();
    let mut before = Vec::new();
    for sample in prepared {
        frames.push(sample.frame);
        opponent_spaces.push(sample.space);
        before.push(sample.random);
    }
    let mut after = before.clone();
    let opponent_choices = opponent
        .sample_batch(&frames, &opponent_spaces, &mut after)
        .expect("independent same-B oracle");
    opponent_choices
        .into_iter()
        .zip(opponent_spaces)
        .zip(before)
        .zip(after)
        .map(
            |(((choice, space), before), after)| neural_opponent::OpponentChoice {
                model: Arc::clone(opponent),
                choice,
                space,
                before,
                after,
            },
        )
        .collect()
}

fn config(count: usize) -> PpoConfig {
    PpoConfig {
        environments: count,
        ..parity_settings_for_test().ppo
    }
}

fn inputs(opponent: &Arc<PolicyModel>, base: usize, count: usize) -> (PpoConfig, ActorGroup) {
    (
        config((base + count).max(4)),
        ActorGroup {
            stream_base: base,
            environments: (0..count)
                .map(|stream| {
                    map2_tests::configured_environment(
                        TICK_CAP - [72, 24, 48][(base + stream) % 3],
                        stream % 2,
                        OpponentSpec::SharedPolicy(Arc::clone(opponent)),
                        |_| {},
                    )
                })
                .collect(),
            streams: (0..count)
                .map(|stream| EpisodeStream {
                    retention_phase: (base + stream) % RETENTION_STRIDE,
                    ..Default::default()
                })
                .collect(),
            random: (0..count)
                .map(|stream| PpoRng::new(9952810 + (base + stream) as u64))
                .collect(),
        },
    )
}

fn opponent_rngs(group: &ActorGroup) -> Vec<PpoRng> {
    group
        .environments
        .iter()
        .map(|world| match &world.opponent {
            OpponentRuntime::Policy { rng, .. } => rng.clone(),
            _ => panic!("neural opponent"),
        })
        .collect()
}
