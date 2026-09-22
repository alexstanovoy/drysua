use super::*;

#[test]
fn continue_overlap_preserves_mixed_actor_rng_orders_rewards_and_retained_rows() {
    actor_overlap_parity(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires authorized CUDA runner"]
fn cuda_continue_overlap_preserves_mixed_actor_rng_orders_rewards_and_retained_rows() {
    actor_overlap_parity(PolicyDevice::Cuda { ordinal: 0 });
    assert_late_continue_failure(PolicyDevice::Cuda { ordinal: 0 });
}

#[test]
fn late_continue_failure_drains_advanced_worlds_without_rng_or_rollout_commit() {
    assert_late_continue_failure(PolicyDevice::Cpu);
}

fn assert_late_continue_failure(device: PolicyDevice) {
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        if name == "kind.bias" {
            parameters[offset..offset + count].fill(-1.0e9);
            parameters[offset] = 0.0;
        }
        offset += count;
    }
    model
        .import_parameters(&parameters)
        .expect("Continue policy");
    let settings = parity_settings_for_test();
    let mut worlds = environments(&settings, 3).expect("worlds");
    let mut streams = (0..2)
        .map(|index| game_stream(9001, index).expect("stream"))
        .collect::<Vec<_>>();
    let mut random = actor_stream_rngs(&mut PpoRng::new(9001), 2).expect("RNG");
    let before = random.clone();
    let prepared = worlds
        .iter_mut()
        .map(prepare_policy_sample)
        .collect::<Result<Vec<_>, _>>()
        .expect("prepared inputs");
    let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let mut staged = before.clone();
    let mut dispatched = 0;
    let choices = model
        .sample_batch_continue(
            &frames,
            &spaces,
            &mut staged,
            |_, choice| {
                assert_eq!(choice.action(), crate::StructuredAction::Continue);
                dispatched += 1;
                Ok(())
            },
            false,
        )
        .expect("Continue-only batch");
    assert_eq!(dispatched, 2);
    assert!(choices.iter().all(Option::is_none));
    for (source, target) in before.iter().zip(staged) {
        assert_eq!(target.draws() - source.draws(), 16);
    }
    let mut rollout =
        PpoRollout::for_config(settings.ppo, model.policy_identity().expect("identity"))
            .expect("rollout");
    let mut report = PpoSmokeReport::default();
    let error = continue_overlap::collect(
        &model,
        settings.ppo,
        0,
        &mut worlds,
        &mut streams,
        &mut random,
        1,
        &mut rollout,
        &mut report,
        true,
    )
    .expect_err("late error");
    assert!(
        error
            .to_string()
            .contains("injected Continue overlap late decoder failure")
    );
    assert_eq!(random, before);
    assert!(rollout.is_empty());
    assert_eq!(report.elapsed_ticks, 0);
    for world in &worlds {
        assert_eq!(
            world.seats[world.policy_seat]
                .tracker
                .current()
                .expect("snapshot")
                .tick,
            4
        );
    }
}

fn actor_overlap_parity(device: PolicyDevice) {
    use std::hash::Hasher;
    let model = PolicyModel::fresh_on(9101, device).expect("model");
    let mut settings = parity_settings_for_test();
    settings.ppo.environments = 40;
    settings.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    let worlds = |settings: &TrainingJobConfig| {
        (0..40)
            .map(|stream| {
                super::super::build_environment(
                    settings.seed + stream as u64,
                    19 + stream as u64,
                    settings.map,
                    stream % 2,
                    0,
                    OpponentSpec::Teacher,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("worlds")
    };
    let mut first_worlds = worlds(&settings);
    let mut second_worlds = worlds(&settings);
    let mut first_streams = (0..40)
        .map(|index| game_stream(9001, index).expect("stream"))
        .collect::<Vec<_>>();
    let mut second_streams = (0..40)
        .map(|index| game_stream(9001, index).expect("stream"))
        .collect::<Vec<_>>();
    let mut first_rng = actor_stream_rngs(&mut PpoRng::new(9001), 40).expect("RNG");
    let mut second_rng = first_rng.clone();
    let mut first = PpoRollout::for_config(settings.ppo, model.policy_identity().expect("policy"))
        .expect("rollout");
    let mut second = PpoRollout::for_config(settings.ppo, first.policy()).expect("rollout");
    collect_batch(
        &model,
        settings.ppo,
        0,
        "reference",
        &mut first_worlds,
        &mut first_streams,
        &mut first_rng,
        16,
        &mut first,
        &mut PpoSmokeReport::default(),
    )
    .expect("reference");
    continue_overlap::collect(
        &model,
        settings.ppo,
        0,
        &mut second_worlds,
        &mut second_streams,
        &mut second_rng,
        16,
        &mut second,
        &mut PpoSmokeReport::default(),
        false,
    )
    .expect("overlap");
    assert_eq!(first_rng, second_rng);
    assert!(first_streams.iter().any(|stream| stream.actions[0] > 0));
    assert!(
        first_streams
            .iter()
            .any(|stream| stream.actions[0] < stream.decisions as u32)
    );
    for (a, b) in first_streams.iter().zip(&second_streams) {
        assert_eq!(a.trace.finish(), b.trace.finish());
        assert_eq!(a.map2_reward, b.map2_reward);
        assert_eq!(a.retained, b.retained);
    }
    let first = first.finish(settings.ppo).expect("batch");
    let second = second.finish(settings.ppo).expect("batch");
    assert_eq!(first.len(), second.len());
    for index in 0..first.len() {
        let a = first.sample(index).expect("sample");
        let b = second.sample(index).expect("sample");
        assert_eq!(a.action(), b.action());
        assert_eq!(a.transition.frame, b.transition.frame);
        assert_frame_bits(&a.transition.frame, &b.transition.frame);
        assert_eq!(a.transition.target, b.transition.target);
        assert_eq!(a.transition.stream, b.transition.stream);
        assert_eq!(a.transition.ticks, b.transition.ticks);
        assert_eq!(a.transition.terminal, b.transition.terminal);
        assert_eq!(a.transition.reward.to_bits(), b.transition.reward.to_bits());
        assert_eq!(
            a.transition.old_value.to_bits(),
            b.transition.old_value.to_bits()
        );
        assert_eq!(
            a.transition.next_value.to_bits(),
            b.transition.next_value.to_bits()
        );
        assert_eq!(
            a.transition.old_log_probability.to_bits(),
            b.transition.old_log_probability.to_bits()
        );
    }
}

fn assert_frame_bits(source: &FeatureFrame, target: &FeatureFrame) {
    for (source, target) in [
        (source.global.as_slice(), target.global.as_slice()),
        (source.history.as_flattened(), target.history.as_flattened()),
        (
            source.policy_history.as_flattened(),
            target.policy_history.as_flattened(),
        ),
        (source.units.as_flattened(), target.units.as_flattened()),
        (
            source.own_units.as_flattened(),
            target.own_units.as_flattened(),
        ),
        (
            source.remembered_units.as_flattened(),
            target.remembered_units.as_flattened(),
        ),
        (source.points.as_flattened(), target.points.as_flattened()),
        (
            source.abilities.as_flattened(),
            target.abilities.as_flattened(),
        ),
        (source.items.as_flattened(), target.items.as_flattened()),
        (
            source.projectiles.as_flattened(),
            target.projectiles.as_flattened(),
        ),
        (source.loot.as_flattened(), target.loot.as_flattened()),
        (source.map.as_slice(), target.map.as_slice()),
    ] {
        assert_eq!(source.len(), target.len());
        assert!(
            source
                .iter()
                .zip(target)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}
