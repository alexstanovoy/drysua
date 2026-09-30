use super::*;
use crate::PpoBatch;

type CollectionResult = (PpoBatch, Vec<PpoRng>, Vec<(u64, Map2TrainingReward)>);

#[test]
fn parallel_collection_preserves_actor_rng_frames_outcomes_and_ordering() {
    let model = PolicyModel::fresh(9952000).expect("model");
    for count in [2, 6] {
        let mut settings = parity_settings_for_test();
        settings.ppo.environments = count;
        settings.ppo.gae_lambda = 1.0;
        settings.seed = 9952000;
        let serial = comparison_collection(&model, &settings, false);
        let parallel = comparison_collection(&model, &settings, true);
        assert_eq!(serial.1, parallel.1);
        assert_eq!(serial.2, parallel.2);
        assert_batch_parity_for_test(&serial.0, &parallel.0);
    }
}

#[test]
fn flush_evaluator_preserves_submission_order_and_drains_after_frame_errors() {
    let model = PolicyModel::fresh(9952123).expect("model");
    let settings = parity_settings_for_test();
    let mut arenas = environments(&settings, 0).expect("worlds");
    let mut frames: Vec<FeatureFrame> = arenas
        .iter_mut()
        .map(|arena| prepare_policy_sample(arena).expect("prepared").0)
        .collect();
    let mut expected: Vec<Result<u32, String>> = frames
        .iter()
        .map(|frame| {
            Ok(model
                .evaluate_batch(std::slice::from_ref(frame))
                .expect("direct value")[0]
                .value
                .to_bits())
        })
        .collect();
    let mut bad = frames[0].clone();
    bad.global[0] = f32::NAN;
    frames.insert(0, bad);
    expected.insert(
        0,
        Err("PPO model error: model frame 0 contains a non-finite value".to_owned()),
    );
    let (sender, receiver) = std::sync::mpsc::sync_channel::<FlushRequest>(crate::PPO_MAX_GAMES);
    let evaluator = FlushEvaluator {
        sender: std::sync::Mutex::new(Some(sender)),
    };
    std::thread::scope(|scope| {
        scope.spawn(move || flush_evaluator_loop(&model, &receiver));
        let replies: Vec<_> = frames
            .into_iter()
            .map(|frame| {
                let (value_sender, value_receiver) = std::sync::mpsc::sync_channel(1);
                evaluator.submit(frame, value_sender).expect("submit");
                value_receiver
            })
            .collect();
        for (reply, expected) in replies.into_iter().zip(expected) {
            let value = reply
                .recv()
                .expect("value reply")
                .map(f32::to_bits)
                .map_err(|error| error.to_string());
            assert_eq!(value, expected);
        }
        evaluator.shutdown();
    });
}

fn comparison_collection(
    model: &PolicyModel,
    settings: &ParitySettings,
    parallel: bool,
) -> CollectionResult {
    let mut arenas = environments(settings, 0).expect("worlds");
    let mut random =
        actor_stream_rngs(&mut PpoRng::new(settings.seed), arenas.len()).expect("actor RNG");
    let mut streams = streams_for_collection(settings, 0).expect("stream state");
    let mut rollout = PpoRollout::new(
        settings.ppo.environments * RETAINED_PER_EPISODE,
        model.policy_identity().expect("identity"),
    )
    .expect("rollout");
    let mut report = CollectionReport::default();
    if parallel {
        collect_with_workers(
            model,
            settings.ppo,
            0,
            "ppo",
            &mut arenas,
            &mut streams,
            &mut random,
            64,
            &mut rollout,
            &mut report,
            false,
        )
        .expect("production persistent collector");
    } else {
        for _ in 0..64 {
            let active: Vec<_> = (0..streams.len())
                .filter(|&index| !streams[index].done)
                .collect();
            if active.is_empty() {
                break;
            }
            let (choices, spaces) = sample_active(model, &mut random, &mut arenas, &active)
                .expect("serial reference samples");
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
                .expect("serial reference advance");
            }
        }
    }
    assert_eq!(
        environment_rejections(&arenas).expect("rejection totals"),
        0
    );
    use std::hash::Hasher;
    let traces = streams
        .iter()
        .map(|stream| (stream.trace.finish(), stream.map2_reward))
        .collect();
    (
        rollout.finish(settings.ppo).expect("policy batch"),
        random,
        traces,
    )
}
