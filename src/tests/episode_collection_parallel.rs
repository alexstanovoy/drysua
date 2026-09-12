use super::*;

type CollectionResult = (
    PpoRollout,
    PpoRng,
    PpoSmokeReport,
    Vec<PpoRng>,
    Vec<(u64, Map2TrainingReward)>,
);

#[test]
fn parallel_collection_preserves_actor_rng_frames_outcomes_and_ordering() {
    let model = PolicyModel::fresh(9952000).expect("model");
    compare_collections(&model, 2, 64);
}

#[test]
#[ignore = "Bounded full-game CUDA profiling uses the retained M12 reference artifact"]
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn profile_parallel_full_episode_collection() {
    let model =
        PolicyModel::fresh_on(9952000, PolicyDevice::Cuda { ordinal: 0 }).expect("CUDA model");
    TrainingArtifact::load_runtime_weights(
        &model,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/input-facts-m12-u10-init"),
    )
    .expect("retained reference weights");
    compare_collections(&model, 6, ACTOR_DECISIONS);
}

fn compare_collections(model: &PolicyModel, count: usize, rounds: usize) {
    let mut settings = parity_settings_for_test();
    settings.ppo.environments = count;
    settings.ppo.gae_lambda = 1.0;
    settings.seed = 9952000;
    let serial = comparison_collection(model, &settings, rounds, false);
    let parallel = comparison_collection(model, &settings, rounds, true);
    assert_eq!(serial.1, parallel.1);
    assert_eq!(serial.3, parallel.3);
    assert_eq!(serial.4, parallel.4);
    assert_eq!(serial.2.elapsed_ticks, parallel.2.elapsed_ticks);
    assert_eq!(serial.2.terminal_wins, parallel.2.terminal_wins);
    assert_eq!(serial.2.terminal_losses, parallel.2.terminal_losses);
    assert_eq!(serial.2.terminal_draws, parallel.2.terminal_draws);
    assert_eq!(serial.2.episode_timeouts, parallel.2.episode_timeouts);
    assert_eq!(serial.2.map2_reward, parallel.2.map2_reward);
    assert_eq!(serial.2.rejected_orders, 0);
    assert_eq!(parallel.2.rejected_orders, 0);
    assert!(!serial.0.is_empty());
    assert_eq!(serial.0.len(), parallel.0.len());
    let source = serial.0.finish(settings.ppo).expect("serial samples");
    let target = parallel.0.finish(settings.ppo).expect("parallel samples");
    for index in 0..source.len() {
        let left = source.sample(index).expect("source");
        let right = target.sample(index).expect("target");
        assert_eq!(left.transition.action, right.transition.action);
        assert_eq!(left.transition.frame, right.transition.frame);
        assert_eq!(left.transition.target, right.transition.target);
        assert_eq!(left.transition.reward, right.transition.reward);
        assert_eq!(
            left.transition.old_value.to_bits(),
            right.transition.old_value.to_bits()
        );
        assert_eq!(
            left.transition.next_value.to_bits(),
            right.transition.next_value.to_bits()
        );
        assert_eq!(
            left.transition.old_log_probability.to_bits(),
            right.transition.old_log_probability.to_bits()
        );
        assert_eq!(left.transition.stream, right.transition.stream);
        assert_eq!(left.transition.decision, right.transition.decision);
        assert_eq!(left.transition.ticks, right.transition.ticks);
        assert_eq!(left.transition.terminal, right.transition.terminal);
        assert_eq!(left.return_value.to_bits(), right.return_value.to_bits());
        assert_eq!(left.advantage.to_bits(), right.advantage.to_bits());
    }
}

fn comparison_collection(
    model: &PolicyModel,
    settings: &TrainingJobConfig,
    rounds: usize,
    parallel: bool,
) -> CollectionResult {
    assert!(rounds > 0);
    assert!(rounds <= ACTOR_DECISIONS);
    let mut arenas = environments(settings, 0).expect("worlds");
    let mut master = PpoRng::new(settings.seed);
    let mut random = actor_stream_rngs(&mut master, arenas.len()).expect("actor RNG");
    let mut streams = streams_for_collection(settings, 0).expect("stream state");
    let mut rollout = PpoRollout::new(
        settings.ppo.environments * RETAINED_PER_EPISODE,
        model.policy_identity().expect("identity"),
    )
    .expect("rollout");
    let mut report = PpoSmokeReport::default();
    let started = Instant::now();
    for _ in 0..rounds {
        let active: Vec<_> = (0..streams.len())
            .filter(|&index| !streams[index].done)
            .collect();
        if active.is_empty() {
            break;
        }
        let (choices, spaces) = if parallel {
            sample_active(model, &mut random, &mut arenas, &active).expect("parallel samples")
        } else {
            serial_samples(model, &mut random, &mut arenas, &active)
        };
        if parallel {
            let mut samples = choices.into_iter().zip(spaces);
            let jobs = streams
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| active.contains(index))
                .map(|(index, state)| {
                    let (choice, space) = samples.next().expect("sample");
                    (index, (state, choice, space))
                })
                .collect();
            let completed = super::super::parallel::ordered_active(
                &mut arenas,
                jobs,
                |_, world, (state, choice, space)| {
                    advance_cpu(world, state, choice, space, settings.ppo)
                },
            )
            .expect("CPU jobs");
            for (index, completed) in active.into_iter().zip(completed) {
                finish_advance(
                    model,
                    &mut arenas[index],
                    &mut streams[index],
                    index,
                    completed,
                    &mut rollout,
                    &mut report,
                )
                .expect("finish");
            }
        } else {
            for ((index, choice), space) in active.into_iter().zip(choices).zip(spaces) {
                advance_stream(
                    model,
                    &mut arenas[index],
                    &mut streams[index],
                    (index, choice, space),
                    settings.ppo,
                    &mut rollout,
                    &mut report,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "serial stream={index} decision={} tick={} requests={:?}: {error}",
                        streams[index].decisions,
                        arenas[index].arena.tick(),
                        streams[index].last_requests,
                    );
                });
            }
        }
    }
    println!(
        "parallel={parallel} seconds={:.6} actions={} retained={} ticks={} wins={} losses={}",
        started.elapsed().as_secs_f64(),
        streams.iter().map(|stream| stream.decisions).sum::<usize>(),
        rollout.len(),
        report.elapsed_ticks,
        report.terminal_wins,
        report.terminal_losses
    );
    report.rejected_orders = environment_rejections(&arenas).expect("rejection totals");
    use std::hash::Hasher;
    let traces = streams
        .iter()
        .map(|stream| (stream.trace.finish(), stream.map2_reward))
        .collect();
    (rollout, master, report, random, traces)
}

fn serial_samples(
    model: &PolicyModel,
    random: &mut [PpoRng],
    arenas: &mut [TrainingEnvironment],
    active: &[usize],
) -> (Vec<PpoPolicyChoice>, Vec<ActionSpace>) {
    let mut frames = Vec::new();
    let mut spaces = Vec::new();
    let mut selected: Vec<_> = active.iter().map(|&index| random[index].clone()).collect();
    for &index in active {
        let (frame, space) = prepare_policy_sample(&mut arenas[index]).expect("serial preparation");
        frames.push(frame);
        spaces.push(space);
    }
    let choices = model
        .sample_batch(&frames, &spaces, &mut selected)
        .expect("serial samples");
    for (&index, state) in active.iter().zip(selected) {
        random[index] = state;
    }
    (choices, spaces)
}
