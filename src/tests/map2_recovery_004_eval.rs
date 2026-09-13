use super::*;

#[test]
#[ignore = "All predeclared recovery004 local outcomes, no DEV/cohorts and no early failure bailout."]
fn all_matched_outcomes() {
    let started = Instant::now();
    let reference = parent();
    let recovery002 = start_model();
    let start = initial();
    let learned = fit::fitted();
    let mut report = String::new();
    let trips = trip_results(&recovery002, &start, &learned, &mut report);
    let combat = combat_results(&reference, &start, &learned, &mut report);
    let ordinary = ordinary_results(&reference, &start, &learned, &mut report);
    old_witnesses(&reference, &start, &learned, &mut report);
    writeln!(report, "trip_gate={trips}\ncombat_gate={combat}\nordinary_gate={ordinary}\nnew_outcome_gate={}\nqualified=false\ndev_games_run=0\nwall_seconds={:.3}", trips && combat && ordinary, started.elapsed().as_secs_f64()).unwrap();
    write_new(&root4().join("OUTCOMES.txt"), &report);
    eprintln!("{report}");
}

fn trip_results(
    recovery002: &PolicyModel,
    start: &PolicyModel,
    learned: &PolicyModel,
    report: &mut String,
) -> bool {
    let mut counts = [[0u32; 2]; 3];
    let mut finishes = [[0u32; 2]; 3];
    let mut viable = [0u32; 2];
    let mut viable_finishes = [0u32; 2];
    for setup in trip_setups4(false) {
        let (control, _) = trip::run(setup, trip::Controller::Reference);
        let valid = trip::valid_demonstration(setup, &control);
        if valid {
            if setup.threat == trip::Threat::Finish {
                viable_finishes[setup.side] += 1;
            } else {
                viable[setup.side] += 1;
            }
        }
        for (index, (name, model)) in [
            ("recovery002", recovery002),
            ("start", start),
            ("learned", learned),
        ]
        .into_iter()
        .enumerate()
        {
            let (mut outcome, metrics) = goal::run(setup, model);
            let pass = goal::valid(setup, &outcome);
            let traces = std::mem::take(&mut outcome.exact.rows);
            writeln!(report, "trip actor={name} setup={setup:?} viable={valid} pass={pass} outcome={outcome:?} metrics={metrics:?}").unwrap();
            for trace in traces.into_iter().take(16) {
                writeln!(report, "{name}_trace {trace}").unwrap();
            }
            if valid && pass {
                if setup.threat == trip::Threat::Finish {
                    finishes[index][setup.side] += 1;
                } else {
                    counts[index][setup.side] += 1;
                }
            }
        }
    }
    writeln!(report, "trip_viable={viable:?} recovery002_start_learned={counts:?} finish_viable={viable_finishes:?} finishes={finishes:?}").unwrap();
    (0..2).all(|side| {
        counts[2][side] >= counts[0][side].max(counts[1][side]).max([4, 5][side])
            && viable_finishes[side] > 0
            && finishes[2][side] == viable_finishes[side]
    })
}

fn closed(prefix: &Prefix, model: &PolicyModel) -> Metrics {
    let mut state = rank::replay(prefix);
    for _ in prefix.actions.len()..60 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        state.action(model.choose(&frame, &space).unwrap().action);
    }
    state.total
}

fn combat_results(
    reference: &PolicyModel,
    start: &PolicyModel,
    learned: &PolicyModel,
    report: &mut String,
) -> bool {
    let mut mango = [0u32; 2];
    let mut closure = [0u32; 2];
    let mut empty = true;
    let mut last_hits = true;
    for (coverage, physics) in combat_cases4(false) {
        let prefix = Prefix {
            physics,
            actions: vec![],
        };
        let before = closed(&prefix, start);
        let after = closed(&prefix, learned);
        let fixed = closed(&prefix, reference);
        if coverage == "empty" {
            empty &= after.mana == 0 && after.casts == 0;
        }
        if coverage == "creep_low" {
            last_hits &= after.last_hits >= before.last_hits.max(1) && after.mana <= before.mana;
        }
        writeln!(report, "combat coverage={coverage} physics={physics:?} reference={fixed:?} start={before:?} learned={after:?}").unwrap();
        if coverage == "mango_combat" {
            mango[physics.side] +=
                u32::from(after.mango > 0 && after.damage > 0 && after.score >= fixed.score - TIE);
            let first =
                data::attack_branch(physics).expect("legal physical Attack-first Mango diagnostic");
            let branch = Prefix {
                physics,
                actions: vec![first],
            };
            let expected = closed(&branch, reference);
            let actual = closed(&branch, learned);
            closure[physics.side] += u32::from(
                actual.mango > 0 && actual.damage > 0 && actual.score >= expected.score - TIE,
            );
            writeln!(
                report,
                "mango_closure seed={} side={} reference={expected:?} learned={actual:?}",
                physics.seed, physics.side
            )
            .unwrap();
        }
    }
    writeln!(
        report,
        "mango={mango:?} attack_closure={closure:?} empty_no_waste={empty} efficient_lh={last_hits}"
    )
    .unwrap();
    mango == [2, 2] && closure == [2, 2] && empty && last_hits
}

fn ordinary_results(
    reference: &PolicyModel,
    start: &PolicyModel,
    learned: &PolicyModel,
    report: &mut String,
) -> bool {
    let mut passed = true;
    for seed in [10103980, 10103981] {
        for side in 0..2 {
            let healthy = ordinary_outcome(reference, seed, side);
            for (name, model) in [("start", start), ("learned", learned)] {
                let result = ordinary_outcome(model, seed, side);
                if name == "learned" {
                    passed &= result.0 + 1.0 >= healthy.0.max(1000.0)
                        && result.1 + 1.0 >= healthy.1.max(5000.0)
                        && (result.2.damage > 0 || result.2.last_hits > 0);
                }
                writeln!(report, "ordinary actor={name} seed={seed} side={side} healthy={healthy:?} observed={result:?}").unwrap();
            }
        }
    }
    passed
}

fn old_witnesses(
    reference: &PolicyModel,
    start: &PolicyModel,
    learned: &PolicyModel,
    report: &mut String,
) {
    for fountain in [true, false] {
        let mut state = audit::witness_state(fountain, start);
        let (frame, space) = state.prepare();
        writeln!(report, "old_same_frame fountain={fountain} tick={} reference={:?} start={:?} learned={:?} data_reference={:?}", space.tick(), reference.choose(&frame, &space).unwrap().action, start.choose(&frame, &space).unwrap().action, learned.choose(&frame, &space).unwrap().action, data_reference(&state, &space)).unwrap();
    }
    let physics = Physics {
        seed: 10099498,
        ..mango_physics()
    };
    let prefix = Prefix {
        physics,
        actions: vec![data::attack_branch(physics).unwrap()],
    };
    writeln!(
        report,
        "old_mango_attack_closure reference={:?} start={:?} learned={:?}",
        closed(&prefix, reference),
        closed(&prefix, start),
        closed(&prefix, learned)
    )
    .unwrap();
}
