use super::*;

#[test]
#[ignore = "Read-only postfit opening mechanism; old diagnostic seed, no optimizer or new fit."]
fn remaining_opening_same_frame_and_native_orders() {
    let reference = parent();
    let learned = fit::fitted();
    let identity = learned.policy_identity().unwrap();
    let mut report = String::new();
    let mut state = ordinary(10101980, 0);
    let mut found = false;
    for decision in 0..300 {
        let (frame, space) = state.prepare();
        let expected = reference.choose(&frame, &space).unwrap().action;
        let actual = learned.choose(&frame, &space).unwrap().action;
        if expected != actual {
            writeln!(report, "first_common_reference_prefix_difference decision={decision} tick={} reference={expected:?} learned={actual:?} hero={:?}", space.tick(), state.seats[0].tracker.own_hero().map(|hero| (hero.pos, hero.hp, hero.mana))).unwrap();
            found = true;
            break;
        }
        trip::act(&mut state, expected, true, |_| {});
    }
    assert!(found, "recorded opening difference absent within901ticks");
    for (name, model) in [("reference", &reference), ("learned", &learned)] {
        let mut state = ordinary(10101980, 0);
        let mut sent = 0;
        for _ in 0..300 {
            let (frame, space) = state.prepare();
            let action = model.choose(&frame, &space).unwrap().action;
            if action != StructuredAction::Continue && sent < 16 {
                writeln!(
                    report,
                    "opening actor={name} tick={} action={action:?} active={:?} hero={:?}",
                    space.tick(),
                    state.seats[0].local.active_order(),
                    state.seats[0].tracker.own_hero().map(|hero| hero.pos)
                )
                .unwrap();
                sent += 1;
            }
            trip::act(&mut state, action, true, |_| {});
        }
        writeln!(
            report,
            "opening901 actor={name} tick={} hero={:?}",
            state.arena.tick(),
            state.seats[0].tracker.own_hero().map(|hero| hero.pos)
        )
        .unwrap();
    }
    assert_eq!(identity, learned.policy_identity().unwrap());
    write_new(&root4().join("WITNESSES.txt"), &report);
    eprintln!("{report}");
}
