#![allow(
    clippy::float_arithmetic,
    reason = "Explicit test-only history features and probe metrics"
)]

use super::tail_curriculum::{TailCorpus, observed_teacher_game, requests_with_expert_label};
use super::*;
use crate::ImitationSample;
use bota_proto::{AbilitySlot, Order};

#[path = "history_probe_pilot.rs"]
mod pilot;

#[derive(Clone, Copy, Debug, Default)]
struct SentOrderHistory {
    last: Option<(u32, ActionKind, bool, Option<u8>)>,
}

impl SentOrderHistory {
    fn features(self, tick: u32) -> [f32; 5] {
        let Some((sent, kind, courier, slot)) = self.last else {
            return [0.0; 5];
        };
        assert!(tick >= sent);
        assert!(slot.is_none_or(|slot| slot < 16));
        [
            1.0,
            (tick - sent).min(150) as f32 / 150.0,
            (kind.index() + 1) as f32 / 16.0,
            if courier { 1.0 } else { 0.5 },
            slot.map_or(0.0, |slot| f32::from(slot + 1) / 16.0),
        ]
    }

    fn sent(&mut self, request: Option<Request>, kind: ActionKind, tick: u32) {
        assert!(tick > 0);
        assert!(self.last.is_none_or(|last| last.0 <= tick));
        if let Some(request) = request {
            let slot = match request.order {
                Order::Cast { slot, .. } | Order::Learn { slot } => Some(slot.0),
                Order::Use { slot, .. } | Order::Put { slot, .. } | Order::Sell { slot } => {
                    Some(slot.0)
                }
                Order::Swap { from, .. } => Some(from.0),
                _ => None,
            };
            self.last = Some((tick, kind, request.unit.is_some(), slot));
        }
    }
}

#[test]
fn sent_history_distinguishes_cast_slots_and_does_not_record_continue_as_a_send() {
    let mut near = SentOrderHistory::default();
    let mut far = SentOrderHistory::default();
    assert_eq!(near.features(1), [0.0; 5]);
    for (history, slot) in [(&mut near, 0), (&mut far, 2)] {
        history.sent(
            Some(Request {
                seq: 1,
                unit: None,
                order: Order::Cast {
                    slot: AbilitySlot(slot),
                    target: bota_proto::Target::None,
                },
            }),
            ActionKind::Cast,
            1,
        );
    }
    assert_eq!(near.features(4)[..4], far.features(4)[..4]);
    assert_ne!(near.features(4)[4], far.features(4)[4]);
    let before = near.features(7);
    near.sent(None, ActionKind::Continue, 4);
    assert_eq!(near.features(7), before);
}

fn history_samples(
    environment: &mut TrainingEnvironment,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
    seed: u64,
    namespace: SeedNamespace,
    history: &mut SentOrderHistory,
) -> Result<(Vec<Option<Request>>, [ImitationSample; 2]), PpoError> {
    let expert = 1 - environment.policy_seat;
    let (_, teacher_space) = prepare_neural_observer_sample(&mut environment.seats[expert])?;
    let previous = history.features(teacher_space.tick());
    let (requests, original) =
        requests_with_expert_label(environment, choice, space, seed, namespace)?;
    let mut frame = original.frame().clone();
    assert_eq!(&frame.global[59..64], &[0.0; 5]);
    frame.global[59..64].copy_from_slice(&previous);
    let enriched = ImitationSample::teacher(
        frame,
        &teacher_space,
        original.teacher_action(),
        original.identity(),
    )
    .map_err(imitation_error)?;
    history.sent(
        requests[expert],
        original.teacher_action().kind(),
        teacher_space.tick(),
    );
    assert_eq!(original.identity(), enriched.identity());
    Ok((requests, [original, enriched]))
}

#[test]
fn history_probe_encodes_previous_send_not_the_current_teacher_label() {
    let model = PolicyModel::fresh(10089000).expect("model");
    let mut arena =
        build_environment(10089000, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let mut random = PpoRng::new(10089100);
    let mut history = SentOrderHistory::default();
    let mut previous = None;
    for _ in 0..4 {
        let (frame, space) = prepare_policy_sample(&mut arena).expect("actor frame");
        let choice = model.sample(&frame, &space, &mut random).expect("actor");
        let (requests, samples) = history_samples(
            &mut arena,
            &choice,
            &space,
            10089000,
            SeedNamespace::Training,
            &mut history,
        )
        .expect("samples");
        assert_eq!(
            samples[1].frame().global[61],
            previous.map_or(0.0, |kind: ActionKind| (kind.index() + 1) as f32 / 16.0)
        );
        assert_eq!(samples[0].frame().global[61], 0.0);
        if requests[1].is_some() {
            previous = Some(samples[0].teacher_action().kind());
        }
        advance_interval(&mut arena, requests, 3).expect("step");
    }
}

struct HistoryGame {
    environment: TrainingEnvironment,
    corpora: [Option<TailCorpus>; 2],
    history: SentOrderHistory,
    random: PpoRng,
    seed: u64,
    namespace: SeedNamespace,
    finished: bool,
    transcript: Option<std::io::BufWriter<std::fs::File>>,
    counts: [u32; ActionKind::COUNT],
}

fn history_games() -> Vec<HistoryGame> {
    (0..8)
        .map(|index| {
            let seed = 10089000 + index / 2;
            HistoryGame {
                environment: observed_teacher_game(seed, index as usize % 2),
                corpora: std::array::from_fn(|_| Some(TailCorpus::new(10089120 + index))),
                history: SentOrderHistory::default(),
                random: PpoRng::new(10089100 + index),
                seed,
                namespace: if seed < 10089002 {
                    SeedNamespace::Training
                } else {
                    SeedNamespace::Validation
                },
                finished: false,
                transcript: None,
                counts: [0; ActionKind::COUNT],
            }
        })
        .collect()
}

#[test]
fn matched_contract_history_teachers_observe_before_any_retained_row_or_request() {
    let games = history_games();
    assert_eq!(games.len(), 8);
    for game in games {
        let expert = &game.environment.seats[1 - game.environment.policy_seat];
        assert!(expert.persistence.last_sequence().is_none());
        assert!(
            expert.order_bookkeeping.is_neural(),
            "History must observe from start"
        );
        assert!(!expert.order_bookkeeping.is_candidate());
    }
}

fn new_history_pool() -> ImitationPool {
    let namespaces = SeedNamespaces::new(
        vec![10089000, 10089001],
        vec![10089002, 10089003],
        vec![10089010],
    )
    .expect("namespaces");
    ImitationPool::new(
        16384,
        10089200,
        namespaces,
        TrainingScope::new(MapId(0), IMITATION_RULES_AUDIT_VERSION).expect("scope"),
    )
    .expect("pool")
}

fn collect_history_pools(
    model: &PolicyModel,
    output: Option<&Path>,
) -> ([ImitationPool; 2], [TeacherCoverage; 2]) {
    let started = Instant::now();
    let mut pools = std::array::from_fn(|_| new_history_pool());
    let mut coverage = std::array::from_fn(|_| TeacherCoverage::new());
    let mut games = history_games();
    if let Some(output) = output {
        pilot::open_transcripts(&mut games, output);
    }
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(1200),
            "history collection budget"
        );
        let active: Vec<_> = (0..8).filter(|&index| !games[index].finished).collect();
        if active.is_empty() {
            break;
        }
        let completed = history_batch(model, &mut games, &active);
        for pair in completed.into_iter().flatten() {
            for (index, samples) in pair.into_iter().enumerate() {
                for sample in samples {
                    if sample.identity().namespace() == SeedNamespace::Validation {
                        coverage[index]
                            .record_represented_for(&sample)
                            .expect("coverage");
                    }
                    assert!(pools[index].push(sample).expect("push").is_none());
                }
            }
        }
    }
    assert!(games.iter().all(|game| game.finished));
    assert!(
        started.elapsed() < Duration::from_secs(1200),
        "history collection budget"
    );
    assert_eq!(pools[0].statistics().teacher, pools[1].statistics().teacher);
    pilot::assert_matched_pools(&pools);
    eprintln!(
        "history_dataset statistics={:?} seconds={:.3}",
        pools[0].statistics(),
        started.elapsed().as_secs_f64()
    );
    (pools, coverage)
}

fn history_batch(
    model: &PolicyModel,
    games: &mut [HistoryGame],
    active: &[usize],
) -> Vec<Option<[Vec<ImitationSample>; 2]>> {
    assert_eq!(games.len(), 8);
    assert!(!active.is_empty());
    assert!(active.len() <= games.len());
    let prepared = parallel::ordered_active(
        games,
        active.iter().map(|&index| (index, ())).collect(),
        |_, game, ()| prepare_policy_sample(&mut game.environment),
    )
    .expect("prepare");
    let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let mut random: Vec<_> = active
        .iter()
        .map(|&index| games[index].random.clone())
        .collect();
    let choices = model
        .sample_batch(&frames, &spaces, &mut random)
        .expect("frozen M14 initialization opponent");
    for (&index, state) in active.iter().zip(random) {
        games[index].random = state;
    }
    let jobs = active
        .iter()
        .zip(choices)
        .zip(spaces)
        .map(|((&index, choice), space)| (index, (choice, space)))
        .collect();
    parallel::ordered_active(games, jobs, |_, game, (choice, space)| {
        history_round(game, &choice, &space)
    })
    .expect("round")
}

fn history_round(
    game: &mut HistoryGame,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
) -> Result<Option<[Vec<ImitationSample>; 2]>, PpoError> {
    assert!(!game.finished);
    assert!(game.environment.arena.tick() < 108900);
    let (requests, samples) = history_samples(
        &mut game.environment,
        choice,
        space,
        game.seed,
        game.namespace,
        &mut game.history,
    )?;
    pilot::record_history_round(game, &requests, &samples);
    for (corpus, sample) in game.corpora.iter_mut().zip(samples) {
        corpus.as_mut().expect("corpus").retain(sample);
    }
    let ticks = 3.min(108900 - game.environment.arena.tick());
    let step = advance_interval(&mut game.environment, requests, ticks)?;
    reject_production_rejection(&game.environment, "history probe")?;
    game.finished = step.winner.is_some() || game.environment.arena.tick() == 108900;
    if !game.finished {
        return Ok(None);
    }
    let side = 1 - game.environment.policy_seat;
    let expert_won = step.winner == Some(game.environment.seats[side].tracker.team());
    pilot::finish_transcript(game);
    eprintln!(
        "history_game seed={} expert_side={side} won={expert_won} tick={} frequencies={:?}",
        game.seed,
        game.environment.arena.tick(),
        game.counts
    );
    Ok(Some(std::array::from_fn(|index| {
        game.corpora[index]
            .take()
            .expect("corpus")
            .finish(expert_won)
    })))
}

fn zero_probe_input_weights(model: &PolicyModel) -> Vec<f32> {
    let mut parameters = model.export_parameters().expect("parameters");
    let mut offset = 0;
    let mut cleared = false;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if name == "trunk.0.weight" {
            assert_eq!(shape, [2589, 512]);
            parameters[offset + 59 * 512..offset + 64 * 512].fill(0.0);
            cleared = true;
        }
        offset += shape.iter().product::<usize>();
    }
    assert!(cleared);
    assert_eq!(offset, parameters.len());
    parameters
}

#[test]
fn zeroed_reserved_input_rows_preserve_predictions_before_history_training() {
    let model = PolicyModel::fresh(10089201).expect("model");
    let mut arena =
        build_environment(10089000, 1, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let (frame, space) = prepare_policy_sample(&mut arena).expect("frame");
    let original = model.choose(&frame, &space).expect("original");
    model
        .import_parameters(&zero_probe_input_weights(&model))
        .expect("zero unused rows");
    let mut enriched = frame.clone();
    enriched.global[59..64].copy_from_slice(&[1.0, 0.02, 0.5, 0.5, 0.0625]);
    assert_eq!(
        model.choose(&enriched, &space).expect("enriched").action,
        original.action
    );
    assert_eq!(
        model.evaluate(&frame).expect("baseline"),
        model.evaluate(&enriched).expect("enriched")
    );
}

fn fit_history_variant(
    model: &PolicyModel,
    pool: &ImitationPool,
    coverage: &TeacherCoverage,
    enriched: bool,
) {
    let mut trainer = BehavioralTrainer::new(
        64,
        10089151,
        AdamConfig {
            learning_rate: 3e-5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        },
        model,
        pool,
    )
    .expect("optimizer");
    let started = Instant::now();
    for epoch in 0..=8 {
        if epoch > 0 {
            trainer.train_epoch(model, pool).expect("epoch");
        }
        let metrics = OfflineEvaluation::evaluate_validation(model, pool, coverage.clone())
            .expect("validation");
        for (side, values) in [
            ("Radiant", &metrics.metrics().radiant),
            ("Dire", &metrics.metrics().dire),
        ] {
            let continued = values.families[ActionKind::Continue.index()];
            eprintln!(
                "history_fit enriched={enriched} epoch={epoch} side={side} full={}/{} noncontinue={}/{}",
                values.full.matching,
                values.full.total,
                values.full.matching - continued.matching,
                values.full.total - continued.total
            );
        }
        assert!(started.elapsed() < Duration::from_secs(600));
    }
    assert_eq!(trainer.counters().epoch, 8);
}

#[test]
#[ignore = "Test-only reserved-channel history hypothesis; enriched weights must never be deployed as F11"]
fn probe_previous_sent_order_information_without_schema_migration() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let actor = PolicyModel::fresh_on(10089200, device).expect("actor");
    TrainingArtifact::load_runtime_weights(
        &actor,
        &root.join("artifacts/temp/input-facts-m12-u10-init"),
    )
    .expect("opponent");
    let (pools, coverage) = collect_history_pools(&actor, None);
    TrainingArtifact::load_runtime_weights(
        &actor,
        &root.join("artifacts/temp/neural-reset-20260908/finish-skill-alpha-050"),
    )
    .expect("initialization");
    let parameters = zero_probe_input_weights(&actor);
    eprintln!(
        "history_probe reserved_channels=59..63 diagnostic_only=true no_weight_export=true input=previous_sent_presence_age_kind_body_slot not_actual_spell_completion=true"
    );
    for index in 0..2 {
        let model = PolicyModel::fresh_on(10089200, device).expect("learner");
        model
            .import_parameters(&parameters)
            .expect("matched initialization");
        fit_history_variant(&model, &pools[index], &coverage[index], index == 1);
    }
}
