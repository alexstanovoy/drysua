use super::*;

#[path = "map2_recovery_003_goal.rs"]
mod goal;
#[path = "map2_recovery_003_train.rs"]
mod learn;
#[path = "map2_recovery_003_tests.rs"]
mod tests;
#[path = "map2_recovery_003_witness.rs"]
mod witness;

const ROOT3: &str = "artifacts/temp/map2-gameplay-fix-20260912/recovery-003";
const START_SHA: &str = "ed55f4dbee06c7e4ea0a3c42953f425e7b5851b0bccb813ca93c5a770ada35d0";
const TIE: f64 = 1e-4;

fn root3() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ROOT3)
}
fn start_model() -> PolicyModel {
    assert_eq!(
        file_hash(&output().join("weights/drysua.weights.safetensors")),
        START_SHA
    );
    load_fitted()
}

fn conservative_targets(
    reference: StructuredAction,
    values: &[(StructuredAction, f64)],
) -> (Vec<StructuredAction>, bool) {
    assert!(values.iter().any(|row| row.0 == reference));
    assert!(values.iter().all(|row| row.1.is_finite()));
    let best = values
        .iter()
        .map(|row| row.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let reference_value = values.iter().find(|row| row.0 == reference).unwrap().1;
    if best - reference_value <= TIE {
        return (vec![reference], false);
    }
    let mut targets: Vec<_> = values
        .iter()
        .filter(|row| best - row.1 <= TIE)
        .map(|row| row.0)
        .collect();
    if targets.len() > 4 {
        targets.truncate(1);
    }
    assert!(!targets.is_empty());
    assert!(!targets.contains(&reference));
    (targets, true)
}

fn target_row(prefix: &Prefix, frozen: &PolicyModel) -> (Row, StructuredAction, bool) {
    let mut state = rank::replay(prefix);
    let (frame, space) = state.prepare();
    let reference = frozen.choose(&frame, &space).unwrap().action;
    let mut actions = rank::candidates(&state, &space);
    if !actions.contains(&reference) {
        actions.push(reference);
    }
    assert!(actions.len() <= 11);
    let values: Vec<_> = actions
        .into_iter()
        .map(|action| (action, neural_estimate(prefix, action, frozen).score))
        .collect();
    let (targets, improved) = conservative_targets(reference, &values);
    let origin = if improved {
        "strict_return_improvement"
    } else {
        "verified_reference_tie"
    };
    let row = Row {
        set: CheckedActionSet::new(frame, &space, &targets).unwrap(),
        identity: format!(
            "{origin}:{:?}:tick{}:scores{values:?}",
            prefix.physics,
            space.tick()
        ),
    };
    (row, reference, improved)
}

#[test]
fn tied_parent_action_is_the_only_supervised_target() {
    let reference = use_mango();
    let values = [(StructuredAction::Continue, 1.0), (reference, 1.0)];
    assert_eq!(
        conservative_targets(reference, &values),
        (vec![reference], false)
    );
}

#[test]
#[ignore = "Recorded recovery002 Mango Attack branch fails to fulfill fixed-reference continuation; expected red."]
fn recorded_mango_deferral_red() {
    let frozen = parent();
    let failed = start_model();
    let physics = Physics {
        seed: 10099498,
        ..mango_physics()
    };
    let mut state = game(physics);
    let (frame, space) = state.prepare();
    let attack = failed.choose(&frame, &space).unwrap().action;
    assert_eq!(attack.kind(), ActionKind::AttackUnit);
    let prefix = Prefix {
        physics,
        actions: vec![attack],
    };
    let mut after_attack = rank::replay(&prefix);
    let (frame, space) = after_attack.prepare();
    let reference = frozen.choose(&frame, &space).unwrap().action;
    let delaying = failed.choose(&frame, &space).unwrap().action;
    assert_eq!(reference, use_mango());
    assert_eq!(delaying, StructuredAction::Continue);
    let good = neural_estimate(&prefix, reference, &frozen);
    let bad = neural_estimate(&prefix, delaying, &failed);
    eprintln!(
        "same_after_attack_frame=true reference={reference:?} learned={delaying:?} reference_return={} learned_return={}",
        good.score, bad.score
    );
    assert!(
        bad.score >= good.score - TIE,
        "accepted first-action tie does not justify perpetual deferral under learned continuation"
    );
}
