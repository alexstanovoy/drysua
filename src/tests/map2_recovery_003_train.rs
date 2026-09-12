use super::*;

fn trip_setups(training: bool) -> Vec<trip::Setup> {
    trip::setups(training, if training { &[0, 2] } else { &[1, 3] })
        .into_iter()
        .map(|mut setup| {
            setup.seed += 2000;
            setup
        })
        .collect()
}

fn combat_cases(training: bool) -> Vec<(&'static str, Physics)> {
    combat_setups(training)
        .into_iter()
        .map(|(coverage, mut physics)| {
            physics.seed += 2000;
            (coverage, physics)
        })
        .collect()
}

fn journeys(report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    for setup in trip_setups(true) {
        let mut state = trip::create(setup);
        let start = state.arena.tick();
        let (_, space) = state.prepare();
        let (_, landing) = trip::fountain(&space);
        let mut trace = trip::Trace::default();
        trace.observe(&state, start, landing);
        let mut trajectory = Vec::new();
        let mut ledger = Vec::new();
        let mut last_send = 0;
        for decision in 0..600 {
            if state.terminal {
                break;
            }
            let (frame, space) = state.prepare();
            let action = trip::reference(&state, &space);
            let close = state.seats[setup.side]
                .tracker
                .own_hero()
                .is_some_and(|hero| trip::distance(hero.pos, landing) <= 64.0);
            let sequence = state.seats[setup.side].sequence;
            trip::act(
                &mut state,
                action,
                setup.threat != trip::Threat::None,
                |state| trace.observe(state, start, landing),
            );
            let sent = sequence != state.seats[setup.side].sequence;
            if sent {
                last_send = decision;
            }
            if decision == 0 || sent || decision <= last_send + 3 || decision % 10 == 0 || close {
                trajectory.push(Row {
                    set: CheckedActionSet::new(frame, &space, &[action]).unwrap(),
                    identity: format!("validated_trip_reference:{setup:?}:tick{}", space.tick()),
                });
            }
            ledger.push(action);
            if trip::valid_demonstration(setup, &trace) {
                break;
            }
        }
        let valid = trip::valid_demonstration(setup, &trace);
        writeln!(
            report,
            "trip_setup={setup:?} valid={valid} rows={} trace={trace:?} ledger={ledger:?}",
            trajectory.len()
        )
        .unwrap();
        if valid {
            rows.extend(trajectory);
        }
    }
    assert!(rows.len() <= 2600);
    rows
}

fn corrected_returns(frozen: &PolicyModel, start: &PolicyModel, report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut improvements = 0;
    let mut ties = 0;
    let mut closures = 0;
    for (coverage, physics) in combat_cases(true) {
        let mut prefix = Prefix {
            physics,
            actions: vec![],
        };
        let mut first_state = game(physics);
        let (frame, space) = first_state.prepare();
        let alternate = start.choose(&frame, &space).unwrap().action;
        let reference_first = frozen.choose(&frame, &space).unwrap().action;
        for decision in 0..60 {
            let mut state = rank::replay(&prefix);
            if state.terminal {
                break;
            }
            let (frame, space) = state.prepare();
            let action = frozen.choose(&frame, &space).unwrap().action;
            if [0, 1, 2, 3, 4, 5, 10, 20, 40].contains(&decision) {
                let (row, reference, improved) = target_row(&prefix, frozen);
                assert_eq!(reference, action);
                improvements += u32::from(improved);
                ties += u32::from(!improved);
                writeln!(report, "coverage={coverage} seed={} decision={decision} improved={improved} reference={reference:?} targets={:?}", physics.seed, row.set.actions()).unwrap();
                rows.push(row);
            }
            prefix.actions.push(action);
        }
        writeln!(
            report,
            "reference_ledger physics={physics:?} actions={:?}",
            prefix.actions
        )
        .unwrap();
        if alternate == reference_first {
            continue;
        }
        let mut branch = Prefix {
            physics,
            actions: vec![alternate],
        };
        for depth in 0..3 {
            if rank::replay(&branch).terminal {
                break;
            }
            let (mut row, action, improved) = target_row(&branch, frozen);
            improvements += u32::from(improved);
            ties += u32::from(!improved);
            closures += 1;
            row.identity.push_str(":starting_policy_branch_closure");
            writeln!(report, "closure physics={physics:?} first={alternate:?} depth={depth} reference={action:?} targets={:?}", row.set.actions()).unwrap();
            rows.push(row);
            branch.actions.push(action);
        }
    }
    writeln!(report, "strict_return_improvements={improvements} verified_reference_ties={ties} closure_rows={closures}").unwrap();
    assert!(rows.len() <= 900);
    rows
}

fn ordinary_data(report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    for seed in [10100980, 10100981] {
        for side in 0..2 {
            rehearsal_game(seed, side, &mut rows, report);
        }
    }
    assert!(rows.len() <= 1024);
    rows
}

fn code_hashes() -> Vec<(&'static str, String)> {
    [
        "src/tests/map2_recovery_003.rs",
        "src/tests/map2_recovery_003_goal.rs",
        "src/tests/map2_recovery_003_train.rs",
        "src/tests/map2_recovery_003_tests.rs",
        "src/tests/map2_recovery_002.rs",
        "src/tests/map2_recovery_002_trip.rs",
        "src/tests/map2_recovery_002_fit.rs",
        "src/tests/map2_advantage.rs",
        "src/tests/map2_advantage_rank.rs",
        "src/model.rs",
        "src/model_advantage_fit.rs",
        "src/model_transfer_fit.rs",
    ]
    .into_iter()
    .map(|name| {
        (
            name,
            file_hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join(name)),
        )
    })
    .collect()
}

#[test]
#[ignore = "One guarded conservative-tie corrective fit from recovery002; fixed advantage-M17 reference."]
fn single_corrective_fit() {
    let started = Instant::now();
    assert!(!root3().join("weights").exists());
    let frozen = parent();
    let reference_identity = frozen.policy_identity().unwrap();
    let start = start_model();
    let mut dataset = String::new();
    let trip = journeys(&mut dataset);
    let combat = corrected_returns(&frozen, &start, &mut dataset);
    let ordinary = ordinary_data(&mut dataset);
    assert!(trip.len() + combat.len() + ordinary.len() <= 4096);
    let hashes = [data_hash(&ordinary), data_hash(&trip), data_hash(&combat)];
    let sources = code_hashes();
    writeln!(dataset, "ordinary_rows={} journey_rows={} counterfactual_rows={} data_hashes={hashes:?}\nsource_hashes={sources:?}\nstart_sha={START_SHA}\nfixed_reference_sha={PARENT_SHA}\nfrozen_before_optimizer=true\ncollection_seconds={:.3}", ordinary.len(), trip.len(), combat.len(), started.elapsed().as_secs_f64()).unwrap();
    write_new(&root3().join("DATASET.txt"), &dataset);
    let model = start_model();
    let mut report = String::new();
    let steps = optimize(&model, [&ordinary, &trip, &combat], started, &mut report);
    assert_eq!(
        hashes,
        [data_hash(&ordinary), data_hash(&trip), data_hash(&combat)]
    );
    assert_eq!(sources, code_hashes());
    assert_eq!(frozen.policy_identity().unwrap(), reference_identity);
    writeln!(
        report,
        "steps={steps} total_seconds={:.3}",
        started.elapsed().as_secs_f64()
    )
    .unwrap();
    write_new(&root3().join("FIT.txt"), &report);
    assert_eq!(steps, 512, "bounded fit incomplete; no resume");
    let target = root3().join("weights");
    std::fs::create_dir(&target).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &target).unwrap();
    let hash = file_hash(&target.join("drysua.weights.safetensors"));
    write_new(
        &target.join("MANIFEST.txt"),
        &format!(
            "diagnostic_only=true\nqualified=false\ncandidate_pending_outcomes=true\nweights_sha256={hash}\nstart_sha256={START_SHA}\nfixed_reference_sha256={PARENT_SHA}\nmodel=17 feature=15 action=5 ppo=30 rules=25\noptimizer=fresh_Adam_lr1e-4_beta.9_.999_eps1e-8_clip.5\nsteps=512 batch=16 ordinary=8 journey=4 counterfactual=4 seed=10100999\ndata_hashes={hashes:?}\ndataset_sha={}\nsource_hashes={sources:?}\n",
            file_hash(&root3().join("DATASET.txt"))
        ),
    );
    eprintln!(
        "rows={} steps={steps} seconds={:.3} weights_sha={hash}",
        ordinary.len() + trip.len() + combat.len(),
        started.elapsed().as_secs_f64()
    );
}

fn optimize(
    model: &PolicyModel,
    groups: [&Vec<Row>; 3],
    started: Instant,
    report: &mut String,
) -> u64 {
    let mut optimizer = model
        .claim_optimizer(AdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .unwrap();
    let mut random = PpoRng::new(10100999);
    let fit_started = Instant::now();
    for step in 1..=512 {
        if started.elapsed() >= Duration::from_secs(265) {
            break;
        }
        let mut batch = Vec::with_capacity(16);
        for (group, count) in groups.iter().zip([8, 4, 4]) {
            assert!(!group.is_empty());
            for _ in 0..count {
                batch.push(&group[random.below(group.len() as u64).unwrap() as usize].set);
            }
        }
        let update = model
            .train_checked_action_sets(&batch, &mut optimizer)
            .unwrap();
        if step == 1 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_started.elapsed().as_secs_f64()
            )
            .unwrap();
            eprintln!(
                "step={step} loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_started.elapsed().as_secs_f64()
            );
        }
    }
    optimizer.step()
}

fn fitted() -> PolicyModel {
    let model = PolicyModel::fresh_on(10100999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &root3().join("weights")).unwrap();
    model
}

fn trip_outcomes(start: &PolicyModel, model: &PolicyModel, report: &mut String) -> bool {
    let mut start_count = [0, 0];
    let mut learned_count = [0, 0];
    let mut viable = [0, 0];
    let mut finishes = [0, 0];
    let mut finish_pass = [0, 0];
    for setup in trip_setups(false) {
        let (reference, _) = trip::run(setup, trip::Controller::Reference);
        let valid = trip::valid_demonstration(setup, &reference);
        for (name, actor) in [("start", start), ("learned", model)] {
            let (mut outcome, metrics) = goal::run(setup, actor);
            let pass = goal::valid(setup, &outcome);
            let traces = std::mem::take(&mut outcome.exact.rows);
            writeln!(report, "trip actor={name} setup={setup:?} viable={valid} pass={pass} outcome={outcome:?} metrics={metrics:?}").unwrap();
            for row in traces.into_iter().take(16) {
                writeln!(report, "{name}_trace {row}").unwrap();
            }
            if valid && setup.threat != trip::Threat::Finish {
                if name == "start" {
                    start_count[setup.side] += u32::from(pass);
                } else {
                    viable[setup.side] += 1;
                    learned_count[setup.side] += u32::from(pass);
                }
            } else if valid && name == "learned" {
                finishes[setup.side] += 1;
                finish_pass[setup.side] += u32::from(pass);
            }
        }
    }
    writeln!(report, "trip_viable={viable:?} start_functional={start_count:?} learned_functional={learned_count:?} finish_viable={finishes:?} finish_passed={finish_pass:?}").unwrap();
    (0..2).all(|side| {
        learned_count[side] >= start_count[side]
            && learned_count[side] >= [4, 5][side]
            && finishes[side] > 0
            && finish_pass[side] == finishes[side]
    })
}

fn closed_from(prefix: &Prefix, model: &PolicyModel) -> Metrics {
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

fn combat_outcomes(
    reference: &PolicyModel,
    start: &PolicyModel,
    model: &PolicyModel,
    report: &mut String,
) -> bool {
    let mut conversion = [0, 0];
    let mut closure = [0, 0];
    let mut empty = true;
    let mut last_hits = true;
    for (coverage, physics) in combat_cases(false) {
        let prefix = Prefix {
            physics,
            actions: vec![],
        };
        let before = closed_from(&prefix, start);
        let after = closed_from(&prefix, model);
        if coverage == "empty" {
            empty &= after.casts == 0 && after.mana == 0;
        }
        if coverage == "creep_low" {
            last_hits &= after.last_hits >= before.last_hits.max(1) && after.mana <= before.mana;
        }
        writeln!(
            report,
            "combat coverage={coverage} physics={physics:?} start={before:?} learned={after:?}"
        )
        .unwrap();
        if coverage == "mango_combat" {
            let expected = closed_from(&prefix, reference);
            conversion[physics.side] += u32::from(
                after.mango > 0 && after.damage > 0 && after.score >= expected.score - TIE,
            );
            let mut state = game(physics);
            let (frame, space) = state.prepare();
            let first = start.choose(&frame, &space).unwrap().action;
            let branch = Prefix {
                physics,
                actions: vec![first],
            };
            let expected = closed_from(&branch, reference);
            let actual = closed_from(&branch, model);
            closure[physics.side] += u32::from(
                actual.mango > 0 && actual.damage > 0 && actual.score >= expected.score - TIE,
            );
            writeln!(report, "mango_closure seed={} side={} first={first:?} reference={expected:?} learned={actual:?}", physics.seed, physics.side).unwrap();
        }
    }
    writeln!(report, "mango_conversion={conversion:?} mango_closure={closure:?} empty_no_waste={empty} efficient_lh={last_hits}").unwrap();
    empty && last_hits && conversion == [2, 2] && closure == [2, 2]
}

#[test]
#[ignore = "Prospective functional outcome and conservative closure gate; no full cohort games."]
fn evaluate_corrective_fit() {
    let started = Instant::now();
    let reference = parent();
    let start = start_model();
    let model = fitted();
    let mut report = String::new();
    let trips = trip_outcomes(&start, &model, &mut report);
    let combat = combat_outcomes(&reference, &start, &model, &mut report);
    let mut openings = true;
    for seed in [10101980, 10101981] {
        for side in 0..2 {
            let healthy = ordinary_outcome(&reference, seed, side);
            for (name, actor) in [("start", &start), ("learned", &model)] {
                let result = ordinary_outcome(actor, seed, side);
                if name == "learned" {
                    openings &= result.0 + 1.0 >= healthy.0.max(1000.0)
                        && result.1 + 1.0 >= healthy.1.max(5000.0);
                }
                writeln!(report, "ordinary actor={name} seed={seed} side={side} healthy={healthy:?} observed={result:?}").unwrap();
            }
        }
    }
    writeln!(report, "functional_recovery_gate={trips}\nconversion_combat_gate={combat}\nhealthy_opening_gate={openings}\nnew_outcome_gate={}\nqualified=false\ndev_games_run=0\nwall_seconds={:.3}", trips && combat && openings, started.elapsed().as_secs_f64()).unwrap();
    write_new(&root3().join("OUTCOMES.txt"), &report);
    eprintln!("{report}");
}
