use super::*;

#[path = "map2_recovery_002_tests.rs"]
mod tests;
#[path = "map2_recovery_002_fit.rs"]
mod training;
#[path = "map2_recovery_002_trip.rs"]
mod trip;

const OUTPUT: &str = "artifacts/temp/map2-gameplay-fix-20260912/recovery-002";
const PARENT: &str = "artifacts/temp/map2-gameplay-fix-20260912/navigation-run/advantage-m17";
const PARENT_SHA: &str = "8cfd44f1de54666c666d27cc43aafbf0427484cbc6c807dea7cc04c87c1f099b";

fn output() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(OUTPUT)
}
fn parent() -> PolicyModel {
    assert_eq!(
        training::file_hash(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(PARENT)
                .join("drysua.weights.safetensors")
        ),
        PARENT_SHA
    );
    let model = PolicyModel::fresh_on(10098999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(
        &model,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(PARENT),
    )
    .unwrap();
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    model
}
fn mango_physics() -> Physics {
    Physics {
        seed: 10098090,
        side: 0,
        tick: 1201,
        level: 3,
        own_hp: 80,
        enemy_hp: 60,
        mana: 0,
        reach: 450,
        facing: 0,
        creep_hp: None,
        mango: true,
        stash: false,
        gold: 0,
        at_home: false,
        deaths: 1,
    }
}

fn neural_estimate(
    prefix: &Prefix,
    action: StructuredAction,
    frozen: &PolicyModel,
) -> rank::Estimate {
    let mut branch = rank::replay(prefix);
    let before = branch.total.score;
    branch.action(action);
    for _ in 1..HORIZON / 3 {
        if branch.terminal {
            break;
        }
        let (frame, space) = branch.prepare();
        let next = frozen.choose(&frame, &space).unwrap().action;
        branch.action(next);
    }
    rank::Estimate {
        action,
        score: branch.total.score - before,
        metrics: branch.total,
    }
}

fn neural_rank(prefix: &Prefix, frozen: &PolicyModel) -> rank::Ranking {
    let mut state = rank::replay(prefix);
    let (_, space) = state.prepare();
    let estimates: Vec<_> = rank::candidates(&state, &space)
        .into_iter()
        .map(|action| neural_estimate(prefix, action, frozen))
        .collect();
    let value = estimates
        .iter()
        .map(|row| row.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let best = estimates
        .iter()
        .filter(|row| value - row.score <= 1e-4)
        .map(|row| row.action)
        .collect();
    let second = estimates
        .iter()
        .filter(|row| value - row.score > 1e-4)
        .map(|row| row.score)
        .fold(f64::NEG_INFINITY, f64::max);
    rank::Ranking {
        estimates,
        best,
        separation: if second.is_finite() {
            value - second
        } else {
            0.0
        },
    }
}

fn bootstrap_visit(
    prefix: &mut Prefix,
    frozen: &PolicyModel,
    retain: bool,
) -> (Option<(CheckedActionSet, ActionSpace)>, StructuredAction) {
    let mut state = rank::replay(prefix);
    let (frame, space) = state.prepare();
    let action = frozen.choose(&frame, &space).unwrap().action;
    let ranking = retain.then(|| neural_rank(prefix, frozen));
    let sample = ranking
        .filter(|ranking| ranking.best.len() <= 4 && ranking.separation >= 1e-4)
        .map(|ranking| {
            (
                CheckedActionSet::new(frame, &space, &ranking.best).unwrap(),
                space,
            )
        });
    assert!(prefix.actions.len() < 60);
    prefix.actions.push(action);
    (sample, action)
}

#[test]
#[ignore = "Historical suffix/censoring defects under actual M17 parent; expected red, not active fit."]
fn historical_teacher_suffix_and_censoring_red() {
    let frozen = parent();
    let prefix = Prefix {
        physics: mango_physics(),
        actions: vec![],
    };
    let old = rank::estimate(&prefix, use_mango());
    let corrected = neural_estimate(&prefix, use_mango(), &frozen);
    let archived = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/temp/map2-gameplay-fix-20260912/advantage-001/FIT.txt"),
    )
    .unwrap();
    assert!(
        archived.contains("coverage=mango_combat seed=10094096 side=0 decision=0 retained=false")
    );
    assert!(!archived.contains("coverage=mango_combat seed=10094096 side=0 decision=1"));
    eprintln!(
        "old_teacher={old:?} frozen_neural={corrected:?} archived_next_mango_state_censored=true"
    );
    assert_eq!(
        old.score, corrected.score,
        "own-Teacher suffix is not the frozen-neural resource-conversion contract"
    );
}
