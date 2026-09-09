#![allow(
    clippy::float_arithmetic,
    reason = "Bounded neural parameter ablations"
)]

use super::*;
use crate::MODEL_KIND_HEAD;

fn biased_parameters(model: &PolicyModel, bias: &[f32; MODEL_KIND_HEAD]) -> Vec<f32> {
    assert!(
        bias.iter()
            .all(|value| value.is_finite() && value.abs() <= 4.0)
    );
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    let mut applied = false;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if name == "kind.bias" {
            assert_eq!(shape, [16]);
            for (value, addition) in parameters[offset..offset + 16].iter_mut().zip(bias) {
                *value += addition;
            }
            applied = true;
        }
        offset += shape.iter().product::<usize>();
    }
    assert!(applied);
    assert_eq!(offset, parameters.len());
    parameters
}

#[test]
fn batched_bias_probe_matches_actual_parameter_changes_without_mutating_parent() {
    let model = PolicyModel::fresh(10085000).expect("model");
    let original = model.export_parameters().expect("parent parameters");
    let mut biases = [[0.0; MODEL_KIND_HEAD]; 2];
    biases[0][ActionKind::Stop.index()] = 4.0;
    biases[1][ActionKind::MovePoint.index()] = 4.0;
    let mut arenas: Vec<_> = (0..2)
        .map(|side| {
            build_environment(10085000, 1, MapId(0), side, 0, OpponentSpec::Teacher).expect("arena")
        })
        .collect();
    let (frames, spaces): (Vec<_>, Vec<_>) = arenas
        .iter_mut()
        .map(|arena| prepare_policy_sample(arena).expect("frame"))
        .unzip();
    let choices = model
        .choose_kind_bias_probe(&frames, &spaces, &biases)
        .expect("batched probe");
    for index in 0..2 {
        let reference = PolicyModel::fresh(10085001).expect("reference");
        reference
            .import_parameters(&biased_parameters(&model, &biases[index]))
            .expect("changed parameters");
        let expected = reference
            .choose(&frames[index], &spaces[index])
            .expect("reference action");
        assert_eq!(choices[index].action, expected.action);
    }
    assert_eq!(
        model.export_parameters().expect("unmodified parent"),
        original
    );
}

#[test]
fn bias_probe_rejects_wrong_count_and_nonfinite_values() {
    let model = PolicyModel::fresh(10085000).expect("model");
    let mut arena =
        build_environment(10085000, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let (frame, space) = prepare_policy_sample(&mut arena).expect("frame");
    assert_eq!(
        model
            .choose_kind_bias_probe(
                std::slice::from_ref(&frame),
                std::slice::from_ref(&space),
                &[]
            )
            .expect_err("wrong count")
            .to_string(),
        "model produced invalid kind-bias probe count"
    );
    let mut bias = [0.0; MODEL_KIND_HEAD];
    bias[0] = f32::NAN;
    assert_eq!(
        model
            .choose_kind_bias_probe(&[frame], &[space], &[bias])
            .expect_err("nonfinite")
            .to_string(),
        "model produced invalid kind-bias probe finite bound"
    );
}

fn head_grid() -> [[f32; MODEL_KIND_HEAD]; 8] {
    let mut grid = [[0.0; MODEL_KIND_HEAD]; 8];
    grid[1][ActionKind::Stop.index()] = -2.0;
    grid[2][ActionKind::Cast.index()] = -2.0;
    grid[3][ActionKind::Stop.index()] = -2.0;
    grid[3][ActionKind::Cast.index()] = -2.0;
    grid[4][ActionKind::AttackMovePoint.index()] = 2.0;
    grid[5][ActionKind::AttackUnit.index()] = 2.0;
    grid[6][ActionKind::AttackMovePoint.index()] = 2.0;
    grid[6][ActionKind::Stop.index()] = -2.0;
    grid[7][ActionKind::AttackUnit.index()] = 2.0;
    grid[7][ActionKind::Cast.index()] = -2.0;
    grid
}

#[test]
fn head_grid_keeps_a_control_and_eight_distinct_bounded_policies() {
    let grid = head_grid();
    assert_eq!(grid[0], [0.0; MODEL_KIND_HEAD]);
    for (index, row) in grid.iter().enumerate() {
        assert!(!grid[..index].contains(row));
        assert!(
            row.iter()
                .all(|value| value.is_finite() && value.abs() <= 4.0)
        );
    }
}

#[test]
#[ignore = "Bounded training-seed neural-head ablation; not frozen release evaluation"]
fn probe_neural_head_grid_on_training_games() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let parent = std::env::var("DRYSUA_HEAD_PARENT").unwrap_or_else(|_| "initial".into());
    let source = match parent.as_str() {
        "initial" => root.join("artifacts/temp/input-facts-m12-u10-init"),
        "skill" => root.join("artifacts/temp/neural-reset-20260908/finish-skill-bc"),
        _ => panic!("head-grid parent must be initial or skill"),
    };
    let output = root.join(format!(
        "artifacts/temp/neural-reset-20260908/head-grid-{parent}"
    ));
    assert!(!output.exists());
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let model = PolicyModel::fresh_on(10085000, device).expect("parent model");
    TrainingArtifact::load_runtime_weights(&model, &source).expect("source weights");
    let original = model.export_parameters().expect("original weights");
    eprintln!(
        "head_grid parent={parent} seeds=10085000,10085001 sides=both teacher_cadence=3 qualification=false grid={:?}",
        head_grid()
    );
    let mut scores = [[0; 2]; 8];
    for first in [0, 4] {
        play_head_grid(&model, first, &mut scores);
    }
    assert_eq!(
        model.export_parameters().expect("unchanged parent"),
        original
    );
    std::fs::create_dir(&output).expect("grid directory");
    for (index, bias) in head_grid().iter().enumerate() {
        let policy = PolicyModel::fresh(10085001).expect("export model");
        policy
            .import_parameters(&biased_parameters(&model, bias))
            .expect("variant parameters");
        let directory = output.join(format!("candidate-{index:02}"));
        std::fs::create_dir(&directory).expect("candidate directory");
        TrainingArtifact::save_runtime_weights(&policy, &directory).expect("new weights");
        std::fs::write(directory.join("PROVENANCE.txt"), format!("parent={source:?}\nkind_bias_delta={bias:?}\ntraining_seeds=10085000,10085001\ntraining_wins_by_side={:?}\noptimizer=bounded_parameter_ablation\nqualified=false\n", scores[index])).expect("provenance");
    }
    eprintln!("head_grid_scores parent={parent} scores={scores:?}");
}

fn play_head_grid(model: &PolicyModel, first: usize, scores: &mut [[u32; 2]; 8]) {
    assert!(first == 0 || first == 4);
    let mut arenas: Vec<_> = (0..16)
        .map(|index| {
            build_environment(
                10085000 + (index % 4) / 2,
                10085002,
                MapId(0),
                index as usize % 2,
                0,
                OpponentSpec::Teacher,
            )
            .expect("training game")
        })
        .collect();
    let mut done = [false; 16];
    let started = Instant::now();
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(1200),
            "head-grid budget exhausted"
        );
        let active: Vec<_> = (0..16).filter(|&index| !done[index]).collect();
        if active.is_empty() {
            break;
        }
        let results = advance_head_grid(model, &mut arenas, &active, first);
        for (&index, result) in active.iter().zip(results) {
            let arena = &arenas[index];
            reject_production_rejection(arena, "head grid").expect("no rejected orders");
            done[index] = result.winner.is_some() || arena.arena.tick() == 108900;
            if done[index] {
                let candidate = first + index / 4;
                let side = index % 2;
                let outcome =
                    checkpoint_evaluation_outcome(arena.seats[side].tracker.team(), result.winner);
                scores[candidate][side] += u32::from(outcome == CheckpointEvaluationOutcome::Win);
                eprintln!(
                    "head_grid_game candidate={candidate} side={side} seed={} tick={} outcome={outcome:?}",
                    10085000 + (index % 4) / 2,
                    arena.arena.tick()
                );
            }
        }
    }
    assert!(done.iter().all(|done| *done));
    eprintln!(
        "head_grid_group first={first} seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
}

fn advance_head_grid(
    model: &PolicyModel,
    arenas: &mut [TrainingEnvironment],
    active: &[usize],
    first: usize,
) -> Vec<ArenaAdvance> {
    assert!(first == 0 || first == 4);
    assert_eq!(arenas.len(), 16);
    let prepared = parallel::ordered_active(
        arenas,
        active.iter().map(|&index| (index, ())).collect(),
        |_, arena, ()| prepare_policy_sample(arena),
    )
    .expect("prepare");
    let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let grid = head_grid();
    let biases: Vec<_> = active
        .iter()
        .map(|&index| grid[first + index / 4])
        .collect();
    let choices = model
        .choose_kind_bias_probe(&frames, &spaces, &biases)
        .expect("batched model variants");
    let jobs = active
        .iter()
        .zip(choices)
        .zip(spaces)
        .map(|((&index, choice), space)| (index, (choice.action, space)))
        .collect();
    parallel::ordered_active(arenas, jobs, |_, arena, (action, space)| {
        let (_, request) =
            neural_policy_request_in_space(&mut arena.seats[arena.policy_seat], action, &space)?;
        let requests = requests_with_candidate(arena, request)?;
        advance_interval(arena, requests, 3.min(108900 - arena.arena.tick()))
    })
    .expect("advance")
}
