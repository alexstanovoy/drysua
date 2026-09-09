#![allow(clippy::float_arithmetic, reason = "Value-regression diagnostic metrics")]

use super::*;

fn probe_adam(model: &PolicyModel) -> crate::AdamState {
    model.claim_adam_for_test(AdamConfig {
        learning_rate: 1e-3, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, gradient_clip: 0.5,
    }).expect("independent optimizer")
}

#[test]
fn value_probe_can_train_representation_without_changing_a_separate_actor() {
    let actor = PolicyModel::fresh(10087000).expect("actor");
    let original = actor.export_parameters().expect("original");
    let critic = PolicyModel::fresh(10087001).expect("critic");
    critic.import_parameters(&original).expect("same starting parameters");
    let mut arena = build_environment(10087000, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let (frame, _) = prepare_policy_sample(&mut arena).expect("frame");
    let mut optimizer = probe_adam(&critic);
    critic.fit_value_probe(&[frame], &[1.0], &mut optimizer, true).expect("value regression");
    let changed = critic.export_parameters().expect("critic parameters");
    assert!(changed[..1_000_000].iter().zip(&original).any(|(left, right)| left != right));
    assert_eq!(actor.export_parameters().expect("actor unmodified"), original);
    assert_eq!(optimizer.step(), 1);
}

#[test]
fn detached_value_probe_changes_only_the_linear_value_parameters() {
    let critic = PolicyModel::fresh(10087001).expect("critic");
    let original = critic.export_parameters().expect("original");
    let mut arena = build_environment(10087000, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let (frame, _) = prepare_policy_sample(&mut arena).expect("frame");
    let mut optimizer = probe_adam(&critic);
    critic.fit_value_probe(&[frame], &[1.0], &mut optimizer, false).expect("linear regression");
    let changed = critic.export_parameters().expect("critic parameters");
    assert!(changed != original, "value regression must change parameters");
    let mut offset = 0;
    for (name, shape) in critic.parameter_schema().expect("schema") {
        let count = shape.iter().product::<usize>();
        if !name.starts_with("value.") { assert_eq!(changed[offset..offset + count], original[offset..offset + count], "{name}"); }
        offset += count;
    }
    assert_eq!(optimizer.step(), 1);
}

#[test]
fn value_probe_rejects_invalid_targets_without_optimizer_mutation() {
    let critic = PolicyModel::fresh(10087001).expect("critic");
    let mut optimizer = probe_adam(&critic);
    assert_eq!(critic.fit_value_probe(&[], &[], &mut optimizer, true).expect_err("empty batch").to_string(),
        "model produced invalid value-probe batch shape");
    assert_eq!(critic.fit_value_probe(&[FeatureFrame::new()], &[f32::NAN], &mut optimizer, true).expect_err("invalid target").to_string(),
        "model produced invalid value-probe target range");
    assert_eq!(optimizer.step(), 0);
}

struct ValueGame {
    arena: TrainingEnvironment,
    frames: Vec<FeatureFrame>,
    seen: u64,
    retention: PpoRng,
    sampling: PpoRng,
    outcome: Option<CheckpointEvaluationOutcome>,
    seed: u64,
    side: usize,
}

impl ValueGame {
    fn retain(&mut self, frame: &FeatureFrame) {
        assert!(self.seen < 36300);
        assert!(self.frames.len() <= 256);
        self.seen += 1;
        let index = if self.frames.len() < 256 { self.frames.len() as u64 }
            else { self.retention.below(self.seen).expect("retention RNG") };
        if index < 256 {
            if index as usize == self.frames.len() { self.frames.push(frame.clone()); }
            else { self.frames[index as usize] = frame.clone(); }
        }
    }
}

struct ValueRow {
    frame: FeatureFrame,
    target: f32,
    side: usize,
    validation: bool,
}

fn value_games(opponent: OpponentSpec) -> Vec<ValueGame> {
    let games = (0..8).map(|index| {
        let seed = 10087000 + index / 2;
        let side = index as usize % 2;
        ValueGame {
            arena: build_environment(seed, 10087010, MapId(0), side, 0, opponent.clone()).expect("game"),
            frames: Vec::with_capacity(256), seen: 0,
            sampling: PpoRng::new(10087020 + index),
            retention: PpoRng::new(10087040 + index),
            outcome: None, seed, side,
        }
    }).collect::<Vec<_>>();
    assert_eq!(games.len(), 8);
    assert_eq!(games[6].seed, games[7].seed);
    games
}

fn collect_value_rows(model: &PolicyModel, opponent: OpponentSpec) -> Vec<ValueRow> {
    let mut games = value_games(opponent);
    let started = Instant::now();
    for _ in 0..36300 {
        assert!(started.elapsed() < Duration::from_secs(1200), "value data collection budget");
        let active: Vec<_> = (0..games.len()).filter(|&index| games[index].outcome.is_none()).collect();
        if active.is_empty() { break; }
        let prepared = parallel::ordered_active(&mut games,
            active.iter().map(|&index| (index, ())).collect(),
            |_, game, ()| prepare_policy_sample(&mut game.arena)).expect("prepare value data");
        let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let mut random: Vec<_> = active.iter().map(|&index| games[index].sampling.clone()).collect();
        let choices = model.sample_batch(&frames, &spaces, &mut random).expect("frozen actor");
        for (&index, state) in active.iter().zip(random) { games[index].sampling = state; }
        let jobs = active.iter().zip(choices).zip(spaces)
            .map(|((&index, choice), space)| (index, (choice, space))).collect();
        parallel::ordered_active(&mut games, jobs, |_, game, (choice, space)| {
            game.retain(&choice.frame);
            let requests = requests_for_decision_in_space(&mut game.arena, &choice, &space)?;
            let ticks = 3.min(108900 - game.arena.arena.tick());
            let completed = advance_interval(&mut game.arena, requests, ticks)?;
            reject_production_rejection(&game.arena, "value data")?;
            if completed.winner.is_some() || game.arena.arena.tick() == 108900 {
                let team = game.arena.seats[game.side].tracker.team();
                game.outcome = Some(checkpoint_evaluation_outcome(team, completed.winner));
                eprintln!("value_game seed={} side={} ticks={} decisions={} retained={} outcome={:?}",
                    game.seed, game.side, game.arena.arena.tick(), game.seen, game.frames.len(), game.outcome);
            }
            Ok(())
        }).expect("advance value data");
    }
    assert!(games.iter().all(|game| game.outcome.is_some()));
    let rows = games.into_iter().flat_map(|game| {
        let target = f32::from(game.outcome == Some(CheckpointEvaluationOutcome::Win));
        game.frames.into_iter().map(move |frame| ValueRow {
            frame, target, side: game.side, validation: game.seed == 10087003,
        })
    }).collect::<Vec<_>>();
    assert!(rows.len() <= 2048);
    eprintln!("value_collection rows={} seconds={:.3}", rows.len(), started.elapsed().as_secs_f64());
    rows
}

fn value_metrics(model: &PolicyModel, rows: &[ValueRow], validation: bool) -> [f64; 6] {
    let selected: Vec<_> = rows.iter().filter(|row| row.validation == validation).collect();
    assert!(!selected.is_empty());
    assert!(selected.len() <= 2048);
    let frames: Vec<_> = selected.iter().map(|row| row.frame.clone()).collect();
    let predictions = model.evaluate_batch(&frames).expect("value predictions");
    let mut metrics = [0.0; 6];
    for (row, prediction) in selected.into_iter().zip(predictions) {
        let offset = row.side * 3;
        metrics[offset] += f64::from((prediction.value - row.target).powi(2));
        metrics[offset + 1] += f64::from(row.target);
        metrics[offset + 2] += 1.0;
    }
    metrics
}

fn fit_value_variant(model: &PolicyModel, rows: &[ValueRow], train_representation: bool) {
    let mut optimizer = probe_adam(model);
    let train: Vec<_> = rows.iter().filter(|row| !row.validation).collect();
    assert!(train.len() >= 64);
    assert!(train.len() <= 2048);
    let mut order: Vec<_> = (0..train.len()).collect();
    let mut random = PpoRng::new(10087060);
    random.shuffle(&mut order).expect("shuffle");
    let started = Instant::now();
    for step in 0..256 {
        let indices: Vec<_> = (0..64).map(|offset| order[(step * 64 + offset) % order.len()]).collect();
        let frames: Vec<_> = indices.iter().map(|&index| train[index].frame.clone()).collect();
        let targets: Vec<_> = indices.iter().map(|&index| train[index].target).collect();
        model.fit_value_probe(&frames, &targets, &mut optimizer, train_representation).expect("value update");
        if (step + 1) % 64 == 0 {
            eprintln!("value_fit representation={train_representation} step={} train={:?} validation={:?} seconds={:.3}",
                step + 1, value_metrics(model, rows, false), value_metrics(model, rows, true), started.elapsed().as_secs_f64());
        }
        assert!(started.elapsed() < Duration::from_secs(600), "value fit budget");
    }
    assert_eq!(optimizer.step(), 256);
}

#[test]
#[ignore = "Eight-game pilot of critic capacity; not a release or sufficient generalization claim"]
fn probe_value_representation_on_complete_games() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let actor = PolicyModel::fresh_on(10087000, device).expect("actor");
    TrainingArtifact::load_runtime_weights(&actor,
        &root.join("artifacts/temp/neural-reset-20260908/finish-skill-alpha-050")).expect("actor weights");
    let original = actor.export_parameters().expect("actor parameters");
    let opponent = std::env::var("DRYSUA_VALUE_OPPONENT").unwrap_or_else(|_| "teacher".into());
    let opponent_spec = match opponent.as_str() {
        "teacher" => OpponentSpec::Teacher,
        "weak" => OpponentSpec::Weak,
        _ => panic!("value opponent must be teacher or weak"),
    };
    eprintln!("value_opponent={opponent} actor=skill-alpha050 actor_weights_unchanged=true");
    let rows = collect_value_rows(&actor, opponent_spec);
    let mut totals = [[0.0; 2]; 2];
    for row in rows.iter().filter(|row| !row.validation) {
        totals[row.side][0] += f64::from(row.target);
        totals[row.side][1] += 1.0;
    }
    let means = totals.map(|total| total[0] / total[1]);
    let mut baseline = [0.0; 2];
    for row in rows.iter().filter(|row| row.validation) {
        baseline[0] += (means[row.side] - f64::from(row.target)).powi(2);
        baseline[1] += 1.0;
    }
    eprintln!("value_pilot training_side_means={means:?} validation_constant_baseline={baseline:?} metrics_layout=Radiant[squared_error_sum,target_sum,count],Dire[squared_error_sum,target_sum,count] target=win_by108900 pilot_only=true");
    for train_representation in [false, true] {
        let critic = PolicyModel::fresh_on(10087061, device).expect("independent critic");
        critic.import_parameters(&original).expect("same initialization");
        fit_value_variant(&critic, &rows, train_representation);
    }
    assert!(actor.export_parameters().expect("unchanged actor") == original);
}
