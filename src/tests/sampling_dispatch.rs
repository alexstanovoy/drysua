#![allow(
    clippy::float_arithmetic,
    reason = "exact actor-dispatch regression comparisons"
)]

use bota_proto::Aim;

use super::{action, feature};
use crate::model::{take_sampling_dispatches_for_test, with_eager_sampling_for_test};
use crate::{
    ActionKind, ActionSpace, FeatureFrame, LocalPolicyState, PolicyDevice, PolicyModel, PpoRng,
};

#[test]
fn sampling_dispatch_preserves_all_families_mixed_rows_rng_and_statistics() {
    assert_dispatch_parity(PolicyDevice::Cpu);
}

#[test]
fn sampling_dispatch_rejects_traversed_masked_nonfinite_logits_but_not_unused_heads() {
    assert_dispatch_errors(PolicyDevice::Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires the authorized exclusive CUDA runner"]
fn cuda_sampling_dispatch_preserves_eager_bits_and_errors() {
    assert_dispatch_parity(PolicyDevice::Cuda { ordinal: 0 });
    assert_dispatch_errors(PolicyDevice::Cuda { ordinal: 0 });
}

fn assert_dispatch_parity(device: PolicyDevice) {
    let model = uniform_kind_model(device);
    let (frames, spaces) = inputs(64);
    let counts = [0, 1, 2, 2, 1, 2, 2, 5, 5, 4, 3, 2, 2, 2, 3, 1];
    for (index, expected_heads) in counts.into_iter().enumerate() {
        let kind = ActionKind::from_index(index).expect("kind");
        for count in [1, 64] {
            assert_sample_parity(
                &model,
                &frames[..count],
                &spaces[..count],
                &vec![kind; count],
                expected_heads,
            );
        }
    }
    let kinds: Vec<_> = (0..64)
        .map(|index| ActionKind::from_index(index % 16).expect("kind"))
        .collect();
    assert_sample_parity(&model, &frames, &spaces, &kinds, 13);
    for index in 0..16 {
        edit(&model, "kind.bias", |values| {
            values.fill(0.0);
            values[index] = 100.0;
        });
        let actual = model
            .choose_batch(&frames[..1], &spaces[..1])
            .expect("greedy");
        let expected =
            with_eager_sampling_for_test(|| model.choose_batch(&frames[..1], &spaces[..1]))
                .expect("eager greedy");
        assert_eq!(actual[0].action.kind().index(), index);
        assert_eq!(actual[0].action, expected[0].action);
        assert_eq!(actual[0].value.to_bits(), expected[0].value.to_bits());
    }
    #[cfg(feature = "builtin")]
    assert_native_selection(&model);
}

#[cfg(feature = "builtin")]
fn assert_native_selection(model: &PolicyModel) {
    use bota_proto::{MapId, ServerMsg};
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 9952402,
    })
    .expect("native arena");
    let [
        ServerMsg::MatchStart { info },
        ServerMsg::Snapshot { view },
        ServerMsg::Events { tick, events },
    ] = start.messages[0].as_slice()
    else {
        panic!("native initial messages");
    };
    let mut tracker = action::tracker_with_info_and_view(info, view.clone());
    tracker
        .observe_events(*tick, events)
        .expect("native initial events");
    let spaces = [ActionSpace::from_tracker(&tracker).expect("native space")];
    let frames = [feature::encode(&tracker, &LocalPolicyState::new(0))];
    let mut random = [PpoRng::new(17)];
    let mut reference = random.clone();

    let actual = model
        .sample_batch(&frames, &spaces, &mut random)
        .expect("native actor");
    let expected =
        with_eager_sampling_for_test(|| model.sample_batch(&frames, &spaces, &mut reference))
            .expect("native eager actor");

    assert_eq!(random, reference);
    assert_eq!(actual[0].action(), expected[0].action());
    assert_eq!(
        [
            actual[0].value,
            actual[0].log_probability,
            actual[0].entropy
        ]
        .map(f32::to_bits),
        [
            expected[0].value,
            expected[0].log_probability,
            expected[0].entropy
        ]
        .map(f32::to_bits)
    );
    let request = spaces[0]
        .decode(actual[0].action())
        .expect("native order")
        .map(|issued| crate::Request {
            seq: 1,
            unit: issued.unit,
            order: issued.order,
        });
    let step = arena.step(&[request, None]).expect("apply native order");
    assert!(
        !step
            .messages
            .iter()
            .flatten()
            .any(|message| matches!(message, ServerMsg::OrderRejected { .. }))
    );
}

fn assert_sample_parity(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    kinds: &[ActionKind],
    expected_heads: usize,
) {
    let mut actual_rng: Vec<_> = kinds
        .iter()
        .map(|kind| PpoRng::new(seed_for_kind(*kind)))
        .collect();
    let mut expected_rng = actual_rng.clone();
    take_sampling_dispatches_for_test();

    let actual = model
        .sample_batch(frames, spaces, &mut actual_rng)
        .expect("selective actor");
    let dispatched = take_sampling_dispatches_for_test();
    let expected =
        with_eager_sampling_for_test(|| model.sample_batch(frames, spaces, &mut expected_rng))
            .expect("eager actor");

    assert_eq!(dispatched, (expected_heads, expected_heads * frames.len()));
    assert_eq!(take_sampling_dispatches_for_test(), (13, 13 * frames.len()));
    assert_eq!(actual_rng, expected_rng);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual.action().kind(), kinds[index]);
        assert_eq!(actual.action(), expected.action());
        assert_eq!(
            spaces[index].decode(actual.action()),
            spaces[index].decode(expected.action())
        );
        assert_eq!(actual.frame, expected.frame);
        assert_eq!(actual.target, expected.target);
        assert_eq!(actual.policy(), expected.policy());
        assert_eq!(
            [actual.value, actual.log_probability, actual.entropy].map(f32::to_bits),
            [expected.value, expected.log_probability, expected.entropy].map(f32::to_bits)
        );
    }
}

fn assert_dispatch_errors(device: PolicyDevice) {
    let model = uniform_kind_model(device);
    let (frames, spaces) = inputs(65);
    let mut random = vec![PpoRng::new(7); 65];
    let before = random.clone();
    take_sampling_dispatches_for_test();
    assert_eq!(
        model
            .sample_batch(&frames, &spaces, &mut random)
            .expect_err("batch bound")
            .to_string(),
        "model batch count 65 exceeds maximum 64"
    );
    assert_eq!(random, before);
    assert_eq!(take_sampling_dispatches_for_test(), (0, 0));
    edit(&model, "kind_embedding.weight", |values| values.fill(2.0));
    edit(&model, "controlled.weight", |values| {
        for row in 256..288 {
            values[row * 2 + 1] = f32::MAX;
        }
    });
    assert_eq!(
        spaces[0].controlled_unit_mask(ActionKind::Cast).as_array(),
        &[true, false]
    );
    for eager in [false, true] {
        let operation = || {
            let mut random = [PpoRng::new(seed_for_kind(ActionKind::Continue))];
            let before = random.clone();
            let choice = model.sample_batch(&frames[..1], &spaces[..1], &mut random);
            if cfg!(feature = "side-actors") && eager {
                assert_eq!(
                    choice
                        .expect_err("forced eager family is checked")
                        .to_string(),
                    controlled_overflow()
                );
                assert_eq!(random, before);
            } else {
                assert_eq!(
                    choice.expect("skipped overflow")[0].action().kind(),
                    ActionKind::Continue
                );
            }
            let mut random = [PpoRng::new(seed_for_kind(ActionKind::Cast))];
            let before = random.clone();
            let error = model
                .sample_batch(&frames[..1], &spaces[..1], &mut random)
                .expect_err("masked courier logit");
            assert_eq!(error.to_string(), controlled_overflow());
            assert_eq!(random, before);
        };
        if eager {
            with_eager_sampling_for_test(operation);
        } else {
            operation();
        }
    }
    let prefixes = [crate::TrainingPrefix::new(ActionKind::Continue, None, None)];
    let Err(error) = model.training_forward(&frames[..1], &prefixes) else {
        panic!("training must still validate unused conditional heads");
    };
    assert_eq!(error.to_string(), controlled_overflow());
}

fn controlled_overflow() -> &'static str {
    if cfg!(feature = "side-actors") {
        "model radiant.controlled output at batch 0 index 1 is non-finite"
    } else {
        "model controlled output at batch 0 index 1 is non-finite"
    }
}

fn uniform_kind_model(device: PolicyDevice) -> PolicyModel {
    let model = PolicyModel::fresh_on(9952401, device).expect("model");
    for name in ["kind.weight", "kind.bias", "controlled.weight"] {
        edit(&model, name, |values| values.fill(0.0));
    }
    edit(&model, "controlled.bias", |values| {
        values.copy_from_slice(&[100.0, 0.0])
    });
    model
}

fn edit(model: &PolicyModel, target: &str, update: impl FnOnce(&mut [f32])) {
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        if name == target {
            update(&mut parameters[offset..offset + count]);
            model
                .import_parameters(&parameters)
                .expect("finite parameters");
            return;
        }
        offset += count;
    }
    panic!("missing parameter {target}");
}

fn seed_for_kind(kind: ActionKind) -> u64 {
    // Equal kind logits select the largest uniform because Gumbel is monotone.
    for seed in 0..4096 {
        let mut random = PpoRng::new(seed);
        let values: [f64; 16] = std::array::from_fn(|_| random.uniform_open().expect("draw"));
        let selected = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("kind")
            .0;
        if selected == kind.index() {
            return seed;
        }
    }
    panic!("no bounded seed for {kind:?}");
}

fn inputs(count: usize) -> (Vec<FeatureFrame>, Vec<ActionSpace>) {
    assert!((1..=65).contains(&count));
    (0..count)
        .map(|index| {
            let mut view = action::world_view(100 + index as u32);
            let hero = &mut view.units[0];
            for (slot, ability) in hero.abilities.iter_mut().enumerate() {
                ability.cooldown_left = u32::from(slot != index % 3);
            }
            let item = hero.items[0].as_mut().expect("item");
            item.aim = Some([Aim::Own, Aim::Unit, Aim::Point][index % 3]);
            item.range = 1200;
            let tracker = action::tracker_with_info_and_view(&action::match_info(), view);
            let space = ActionSpace::from_tracker(&tracker).expect("space");
            assert!(space.kind_mask().as_array().iter().all(|allowed| *allowed));
            (feature::encode(&tracker, &LocalPolicyState::new(0)), space)
        })
        .unzip()
}
