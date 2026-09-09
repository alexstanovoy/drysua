use super::*;

#[test]
#[ignore = "local immutable u10 artifact; bounded real-observation transfer probe"]
fn approved_u10_real_observation_transfer_preserves_logits_and_actions() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp/map0-ppo21-u010-u040-001/baseline-u010");
    let (model, _) =
        TrainingArtifact::initialize_selected_m11_for_training(&directory, 9101, PolicyDevice::Cpu)
            .expect("approved M11 training initialization");
    let seed = 9873200;
    let (mut arena, start) = Arena::new(ArenaConfig {
        map: MapId(0),
        seats: 2,
        seed,
    })
    .expect("arena");
    let mut seats = [
        NeuralSeat::new(0, &start.messages[0]).expect("seat"),
        NeuralSeat::new(1, &start.messages[1]).expect("seat"),
    ];
    let mut experts = [Some(Teacher::new()), Some(Teacher::new())];
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/input-facts-m11-reference.txt");
    let reference = fs::read_to_string(path).expect("pre-change M11 reference");
    let mut lines = reference.lines();
    let mut maximum_error = 0.0f32;
    let mut observations = 0;
    for tick in 1..=1801 {
        if [1, 31, 121, 301, 901, 1501, 1801].contains(&tick) {
            for seat in &mut seats {
                maximum_error =
                    maximum_error.max(compare_observation(&model, seat, tick, &mut lines));
                observations += 1;
            }
        }
        if tick < 1801 {
            game_tick(&mut arena, &mut seats, &mut experts, None, seed, &mut None)
                .expect("real tick");
        }
    }
    assert_eq!(observations, 14);
    assert!(lines.next().is_none());
    eprintln!(
        "M11 to M12 transfer: 14/14 actions exact, all 13 heads within F32 tolerance; maximum_absolute_error={maximum_error}"
    );
}

fn compare_observation(
    model: &PolicyModel,
    seat: &mut NeuralSeat,
    tick: u32,
    lines: &mut std::str::Lines<'_>,
) -> f32 {
    let space =
        ActionSpace::from_tracker_with_readiness(&seat.tracker, &seat.readiness).expect("space");
    let frame = seat.frame(&space).expect("frame");
    let action = model.choose(&frame, &space).expect("choice").action;
    let target = crate::BehavioralTarget::from_action(&frame, &space, action).expect("target");
    let output = model
        .training_forward(&[frame], &[target.prefix()])
        .expect("forward");
    assert_eq!(lines.next(), Some(format!("{tick} {action:?}").as_str()));
    let mut maximum = 0.0f32;
    for head in [
        output.value(),
        output.kind(),
        output.controlled(),
        output.ability(),
        output.item(),
        output.swap(),
        output.learn(),
        output.shop(),
        output.loot(),
        output.target_mode(),
        output.put_mode(),
        output.entity_pointer(),
        output.point_pointer(),
    ] {
        let actual = head
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("host");
        maximum = maximum.max(compare_reference_logits(
            &actual,
            lines.next().expect("head"),
        ));
    }
    assert!(maximum.is_finite());
    maximum
}

fn compare_reference_logits(actual: &[f32], reference: &str) -> f32 {
    let expected: Vec<f32> = reference
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(", ")
        .map(|value| value.parse().expect("reference float"))
        .collect();
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0f32;
    for (&actual, expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        assert!(
            error <= 1e-4 * (1.0 + expected.abs()),
            "actual={actual} expected={expected}"
        );
        maximum = maximum.max(error);
    }
    maximum
}
