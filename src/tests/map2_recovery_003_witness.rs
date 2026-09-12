use super::*;

fn current() -> PolicyModel {
    let model = PolicyModel::fresh_on(10100999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &root3().join("weights")).unwrap();
    model
}

#[test]
#[ignore = "Fixed-model old Mango closure green plus remaining same-frame liveness witnesses; no fitting."]
fn closure_green_and_remaining_same_frame_choices() {
    let reference = parent();
    let start = start_model();
    let model = current();
    let mut report = String::new();
    let physics = Physics {
        seed: 10099498,
        ..mango_physics()
    };
    let mut state = game(physics);
    let (frame, space) = state.prepare();
    let first = start.choose(&frame, &space).unwrap().action;
    assert_eq!(first.kind(), ActionKind::AttackUnit);
    let prefix = Prefix {
        physics,
        actions: vec![first],
    };
    let mut branch = rank::replay(&prefix);
    let (frame, space) = branch.prepare();
    let fixed_action = reference.choose(&frame, &space).unwrap().action;
    let learned_action = model.choose(&frame, &space).unwrap().action;
    let good = neural_estimate(&prefix, fixed_action, &reference);
    let learned = neural_estimate(&prefix, learned_action, &model);
    assert!(learned.metrics.mango > 0);
    assert!(learned.metrics.damage > 0);
    assert!(learned.score >= good.score - TIE);
    writeln!(report, "old_recorded_mango_after_attack reference={fixed_action:?} learned={learned_action:?} reference_return={} learned_return={} green=true", good.score, learned.score).unwrap();
    fountain_witness(&reference, &start, &model, &mut report);
    let setup = trip::Setup {
        seed: 10101099,
        side: 1,
        place: trip::Place::Lane,
        threat: trip::Threat::Finish,
        variant: 1,
    };
    let mut state = trip::create(setup);
    let (frame, space) = state.prepare();
    writeln!(report, "finish_same_frame setup={setup:?} fixed_reference={:?} start={:?} learned={:?} validated_trip_reference={:?}", reference.choose(&frame, &space).unwrap().action, start.choose(&frame, &space).unwrap().action, model.choose(&frame, &space).unwrap().action, trip::reference(&state, &space)).unwrap();
    write_new(&root3().join("WITNESSES.txt"), &report);
    eprintln!("{report}");
}

fn fountain_witness(
    reference: &PolicyModel,
    start: &PolicyModel,
    model: &PolicyModel,
    report: &mut String,
) {
    let setup = trip::Setup {
        seed: 10101003,
        side: 1,
        place: trip::Place::Barracks,
        threat: trip::Threat::None,
        variant: 1,
    };
    let mut state = trip::create(setup);
    let (_, space) = state.prepare();
    let (_, goal) = trip::fountain(&space);
    for _ in 0..600 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        let hero = state.seats[setup.side].tracker.own_hero().unwrap();
        if hero.pos == goal && hero.hp == hero.max_hp && hero.mana == hero.max_mana {
            writeln!(report, "full_fountain_same_frame tick={} hp={}/{} mana={}/{} fixed_reference={:?} start={:?} learned={:?} validated_trip_reference={:?}", state.arena.tick(), hero.hp, hero.max_hp, hero.mana, hero.max_mana, reference.choose(&frame, &space).unwrap().action, start.choose(&frame, &space).unwrap().action, model.choose(&frame, &space).unwrap().action, trip::reference(&state, &space)).unwrap();
            return;
        }
        trip::act(
            &mut state,
            model.choose(&frame, &space).unwrap().action,
            false,
            |_| {},
        );
    }
    panic!("recorded full-fountain witness was not reproduced");
}
