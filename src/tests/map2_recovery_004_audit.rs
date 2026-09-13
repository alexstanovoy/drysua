use super::*;

pub(super) fn witness_state(fountain: bool, model: &PolicyModel) -> Game {
    let setup = trip::Setup {
        seed: if fountain { 10101003 } else { 10101099 },
        side: 1,
        place: if fountain {
            trip::Place::Barracks
        } else {
            trip::Place::Lane
        },
        threat: if fountain {
            trip::Threat::None
        } else {
            trip::Threat::Finish
        },
        variant: 1,
    };
    let mut state = trip::create(setup);
    for _ in 0..600 {
        let (frame, space) = state.prepare();
        let (_, landing) = trip::fountain(&space);
        let hero = state.seats[1].tracker.own_hero().unwrap();
        if !fountain || hero.pos == landing && hero.hp == hero.max_hp && hero.mana == hero.max_mana
        {
            return state;
        }
        let action = model.choose(&frame, &space).unwrap().action;
        trip::act(&mut state, action, false, |_| {});
    }
    panic!("recorded full-fountain witness not reached within1800ticks");
}

fn gradient_comparison(model: &PolicyModel, source: &Row, witness: &Row, report: &mut String) {
    let (source_loss, source_gradient) = model.checked_singleton_gradient(&[&source.set]).unwrap();
    let (witness_loss, witness_gradient) =
        model.checked_singleton_gradient(&[&witness.set]).unwrap();
    assert_eq!(source_gradient.len(), witness_gradient.len());
    let dot: f64 = source_gradient
        .iter()
        .zip(&witness_gradient)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let cosine = dot / (norm(&source_gradient) * norm(&witness_gradient)).max(1e-30);
    assert!(cosine.is_finite());
    writeln!(report, "source={} target={:?} witness={} target={:?} source_loss={source_loss} witness_loss={witness_loss} gradient_dot={dot} cosine={cosine} source_descent_witness_loss_derivative={} tensor_frames_equal={} exact_collision_claim=false", source.identity, source.set.actions(), witness.identity, witness.set.actions(), -dot, source.set.frame() == witness.set.frame()).unwrap();
}

fn journey_counts(rows: &[Row], report: &mut String) {
    use crate::{global_feature as global, unit_feature as unit};
    let mut full_continue = [0; 2];
    let mut departure = [0; 2];
    let mut finish = [0; 2];
    for row in rows {
        let frame = row.set.frame();
        let side = usize::from(frame.global[global::SIDE_DIRE] > 0.0);
        let full = frame.own_units[0][unit::HP_RATIO] == 1.0
            && frame.own_units[0][unit::MANA_RATIO] == 1.0;
        let action = row.set.actions()[0];
        full_continue[side] += usize::from(full && action == StructuredAction::Continue);
        departure[side] += usize::from(full && action.kind() == ActionKind::MovePoint);
        finish[side] += usize::from(row.identity.contains("threat: Finish"));
    }
    writeln!(report, "journey_rows={} full_pool_continue={full_continue:?} full_pool_move={departure:?} successful_finish_rows={finish:?} uniform_rows_per_batch=4", rows.len()).unwrap();
    assert_eq!(finish, [8, 8]);
    assert_eq!(departure, [8, 8]);
}

#[test]
#[ignore = "Guarded zero-update audit of actual003 rows and recorded witnesses before004 correction."]
fn actual_source_interference_before_fit() {
    let model = initial();
    let identity = model.policy_identity().unwrap();
    let mut report = String::new();
    let journeys = learn::journeys(&mut String::new());
    let hash = data_hash(&journeys);
    assert_eq!(
        hash,
        "420b87f0e5dbe805cce2a1887072f1bd119de9976000004523564f4bc157bc14"
    );
    writeln!(
        report,
        "old003_journey_hash_reproduced={hash} optimizer_updates=0"
    )
    .unwrap();
    journey_counts(&journeys, &mut report);
    let reference = parent();
    for fountain in [true, false] {
        let mut state = witness_state(fountain, &model);
        let (frame, space) = state.prepare();
        let target = trip::reference(&state, &space);
        let witness = Row {
            set: CheckedActionSet::new(frame.clone(), &space, &[target]).unwrap(),
            identity: format!("old_witness_fountain={fountain}:tick{}", space.tick()),
        };
        writeln!(
            report,
            "{} learned={:?} reference={:?} desired={target:?}",
            witness.identity,
            model.choose(&frame, &space).unwrap().action,
            reference.choose(&frame, &space).unwrap().action
        )
        .unwrap();
        let side_rows: Vec<_> = journeys
            .iter()
            .filter(|row| row.set.frame().global[crate::global_feature::SIDE_DIRE] > 0.0)
            .collect();
        for source in side_rows
            .iter()
            .filter(|row| row.set.actions()[0].kind() == target.kind())
            .take(1)
        {
            gradient_comparison(&model, source, &witness, &mut report);
        }
        let mut full_continue: Vec<_> = side_rows
            .iter()
            .filter(|row| {
                row.set.actions()[0] == StructuredAction::Continue
                    && row.set.frame().own_units[0][crate::unit_feature::HP_RATIO] == 1.0
                    && row.set.frame().own_units[0][crate::unit_feature::MANA_RATIO] == 1.0
            })
            .collect();
        full_continue.sort_by(|a, b| a.identity.cmp(&b.identity));
        for source in full_continue.into_iter().take(2) {
            gradient_comparison(&model, source, &witness, &mut report);
        }
        combat_interference(&model, &reference, &witness, &mut report);
    }
    assert_eq!(identity, model.policy_identity().unwrap());
    write_new(&root4().join("PROOF.txt"), &report);
    eprintln!("{report}");
}

fn combat_interference(
    model: &PolicyModel,
    reference: &PolicyModel,
    witness: &Row,
    report: &mut String,
) {
    for (_, mut physics) in combat_setups(true)
        .into_iter()
        .filter(|(name, _)| *name == "empty")
        .take(2)
    {
        physics.seed += 2000;
        let prefix = Prefix {
            physics,
            actions: vec![],
        };
        let (source, action, improved) = target_row(&prefix, reference);
        let estimate = neural_estimate(&prefix, action, reference);
        writeln!(report, "actual003_reference_tie seed={} improved={improved} reference={action:?} return={} metrics={:?}", physics.seed, estimate.score, estimate.metrics).unwrap();
        gradient_comparison(model, &source, witness, report);
    }
}
