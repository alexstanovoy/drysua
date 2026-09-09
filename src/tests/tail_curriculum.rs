use super::*;
use crate::ImitationSample;

pub(super) fn requests_with_expert_label(
    environment: &mut TrainingEnvironment,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
    seed: u64,
    namespace: SeedNamespace,
) -> Result<(Vec<Option<Request>>, ImitationSample), PpoError> {
    let actor = environment.policy_seat;
    assert!(actor < 2);
    assert_eq!(environment.seats.len(), 2);
    let candidate = policy_request_in_space(&mut environment.seats[actor], choice, space)?;
    let expert = &mut environment.seats[1 - actor];
    let (frame, _) = prepare_neural_observer_sample(expert)?;
    let (action, expert_space) = expert
        .teacher
        .decide(&expert.tracker, &expert.persistence, &expert.readiness)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let identity = SampleIdentity::from_frame(namespace, seed, 0, expert_space.tick(), &frame)
        .map_err(imitation_error)?;
    let sample = ImitationSample::teacher(frame, &expert_space, action, identity)
        .map_err(imitation_error)?;
    expert
        .local
        .note_decision(expert_space.tick(), action.kind())
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let issued = expert_space
        .decode(action)
        .map_err(|error| PpoError::Model(error.to_string()))?;
    let request = issue_request(expert, issued, &expert_space, action.kind(), true)?;
    let mut requests = vec![None, None];
    requests[actor] = candidate;
    requests[1 - actor] = request;
    Ok((requests, sample))
}

#[test]
fn expert_label_recording_preserves_both_controllers_requests_and_observations() {
    let model = PolicyModel::fresh(10089000).expect("model");
    for side in 0..2 {
        let mut source = build_environment(10089000, 10, MapId(0), side, 0, OpponentSpec::Teacher)
            .expect("source");
        let mut target = build_environment(10089000, 10, MapId(0), side, 0, OpponentSpec::Teacher)
            .expect("target");
        let mut source_random = PpoRng::new(10089100);
        let mut target_random = source_random.clone();
        for _ in 0..16 {
            let choice =
                sample_policy(&model, &mut source_random, &mut source).expect("source choice");
            let expected = requests_for_decision(&mut source, &choice).expect("source requests");
            let (frame, space) = prepare_policy_sample(&mut target).expect("target frame");
            let choice = model
                .sample(&frame, &space, &mut target_random)
                .expect("target choice");
            let (requests, sample) = requests_with_expert_label(
                &mut target,
                &choice,
                &space,
                10089000,
                SeedNamespace::Training,
            )
            .expect("recording");
            assert!(target.seats[1 - side].order_bookkeeping.is_neural());
            assert!(!target.seats[1 - side].order_bookkeeping.is_candidate());
            assert_eq!(
                sample.side(),
                if side == 0 {
                    ImitationSide::Dire
                } else {
                    ImitationSide::Radiant
                }
            );
            assert_eq!(expected, requests);
            assert_eq!(
                source.seats[1 - side].teacher,
                target.seats[1 - side].teacher
            );
            assert_eq!(
                source.seats[1 - side].persistence,
                target.seats[1 - side].persistence
            );
            assert_eq!(
                source.seats[1 - side].readiness,
                target.seats[1 - side].readiness
            );
            advance_interval(&mut source, expected, 3).expect("source step");
            advance_interval(&mut target, requests, 3).expect("target step");
            assert_eq!(source_random, target_random);
            for index in 0..2 {
                assert_eq!(
                    source.seats[index].tracker.latest_summary(),
                    target.seats[index].tracker.latest_summary()
                );
            }
        }
    }
}

pub(super) struct TailCorpus {
    opening: Vec<ImitationSample>,
    tail: VecDeque<ImitationSample>,
    rare: [Vec<ImitationSample>; ActionKind::COUNT],
    counts: [u64; ActionKind::COUNT],
    random: PpoRng,
}

impl TailCorpus {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            opening: Vec::with_capacity(32),
            tail: VecDeque::with_capacity(1200),
            rare: std::array::from_fn(|_| Vec::with_capacity(16)),
            counts: [0; ActionKind::COUNT],
            random: PpoRng::new(seed),
        }
    }

    pub(super) fn retain(&mut self, sample: ImitationSample) {
        let kind = sample.teacher_action().kind().index();
        assert!(self.counts.iter().sum::<u64>() < 36300);
        assert!(self.tail.len() <= 1200);
        self.counts[kind] += 1;
        if self.opening.len() < 32 {
            self.opening.push(sample.clone());
        }
        if kind != ActionKind::Continue.index() {
            let index = if self.rare[kind].len() < 16 {
                self.rare[kind].len() as u64
            } else {
                self.random.below(self.counts[kind]).expect("reservoir RNG")
            };
            if index < 16 {
                if index as usize == self.rare[kind].len() {
                    self.rare[kind].push(sample.clone());
                } else {
                    self.rare[kind][index as usize] = sample.clone();
                }
            }
        }
        if self.tail.len() == 1200 {
            self.tail.pop_front();
        }
        self.tail.push_back(sample);
    }

    pub(super) fn finish(self, expert_won: bool) -> Vec<ImitationSample> {
        let mut samples = self.opening;
        samples.extend(self.rare.into_iter().flatten());
        if expert_won {
            samples.extend(self.tail);
        }
        assert!(samples.len() <= 32 + 16 * 16 + 1200);
        samples.sort_unstable_by_key(ImitationSample::identity);
        for pair in samples.windows(2) {
            if pair[0].identity() == pair[1].identity() {
                assert_eq!(pair[0].teacher_action(), pair[1].teacher_action());
                assert!(pair[0].frame() == pair[1].frame());
            }
        }
        samples.dedup_by_key(|sample| sample.identity());
        assert!(samples.len() <= 1488);
        samples
    }
}

struct TailGame {
    environment: TrainingEnvironment,
    corpus: Option<TailCorpus>,
    random: PpoRng,
    seed: u64,
    namespace: SeedNamespace,
    finished: bool,
    expert_won: bool,
}

fn tail_games() -> Vec<TailGame> {
    (0..8)
        .map(|index| {
            let seed = 10089000 + index / 2;
            TailGame {
                environment: observed_teacher_game(seed, index as usize % 2),
                corpus: Some(TailCorpus::new(10089120 + index)),
                random: PpoRng::new(10089100 + index),
                seed,
                namespace: if seed < 10089002 {
                    SeedNamespace::Training
                } else {
                    SeedNamespace::Validation
                },
                finished: false,
                expert_won: false,
            }
        })
        .collect()
}

pub(super) fn observed_teacher_game(seed: u64, actor: usize) -> TrainingEnvironment {
    assert!((10089000..10089004).contains(&seed));
    assert!(actor < 2);
    let mut environment =
        build_environment(seed, 10089110, MapId(0), actor, 0, OpponentSpec::Teacher).expect("game");
    observe_neural_seat_orders(&mut environment.seats[1 - actor])
        .expect("Teacher observer from trajectory start");
    environment
}

#[test]
fn matched_contract_tail_teachers_observe_before_any_retained_row_or_request() {
    let games = tail_games();
    assert_eq!(games.len(), 8);
    for game in games {
        let expert = &game.environment.seats[1 - game.environment.policy_seat];
        assert!(expert.persistence.last_sequence().is_none());
        assert!(
            expert.order_bookkeeping.is_neural(),
            "Teacher annotation must observe from start"
        );
        assert!(!expert.order_bookkeeping.is_candidate());
    }
}

fn collect_tail_pool(model: &PolicyModel) -> (ImitationPool, TeacherCoverage) {
    let namespaces = SeedNamespaces::new(
        vec![10089000, 10089001],
        vec![10089002, 10089003],
        vec![10089010],
    )
    .expect("split");
    let mut pool = ImitationPool::new(
        16384,
        10089140,
        namespaces,
        TrainingScope::new(MapId(0), IMITATION_RULES_AUDIT_VERSION).expect("scope"),
    )
    .expect("pool");
    let mut games = tail_games();
    let mut coverage = TeacherCoverage::new();
    let started = Instant::now();
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(1200),
            "tail collection budget"
        );
        let active: Vec<_> = (0..games.len())
            .filter(|&index| !games[index].finished)
            .collect();
        if active.is_empty() {
            break;
        }
        let prepared = parallel::ordered_active(
            &mut games,
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
            .expect("frozen opponent");
        for (&index, state) in active.iter().zip(random) {
            games[index].random = state;
        }
        let jobs = active
            .iter()
            .zip(choices)
            .zip(spaces)
            .map(|((&index, choice), space)| (index, (choice, space)))
            .collect();
        let completed = parallel::ordered_active(&mut games, jobs, |_, game, (choice, space)| {
            collect_tail_round(game, &choice, &space)
        })
        .expect("tail round");
        for samples in completed.into_iter().flatten() {
            for sample in samples {
                if sample.identity().namespace() == SeedNamespace::Validation {
                    coverage
                        .record_represented_for(&sample)
                        .expect("validation coverage");
                }
                assert!(pool.push(sample).expect("pool sample").is_none());
            }
        }
    }
    assert!(games.iter().all(|game| game.finished));
    for side in 0..2 {
        assert!(
            games
                .iter()
                .any(|game| game.namespace == SeedNamespace::Training
                    && 1 - game.environment.policy_seat == side
                    && game.expert_won),
            "no winning training tail for side {side}; do not train a one-sided corpus"
        );
    }
    eprintln!(
        "tail_collection statistics={:?} seconds={:.3}",
        pool.statistics(),
        started.elapsed().as_secs_f64()
    );
    (pool, coverage)
}

fn collect_tail_round(
    game: &mut TailGame,
    choice: &PpoPolicyChoice,
    space: &ActionSpace,
) -> Result<Option<Vec<ImitationSample>>, PpoError> {
    assert!(!game.finished);
    assert!(game.environment.arena.tick() < 108900);
    let (requests, sample) = requests_with_expert_label(
        &mut game.environment,
        choice,
        space,
        game.seed,
        game.namespace,
    )?;
    game.corpus.as_mut().expect("active corpus").retain(sample);
    let ticks = 3.min(108900 - game.environment.arena.tick());
    let step = advance_interval(&mut game.environment, requests, ticks)?;
    reject_production_rejection(&game.environment, "tail curriculum")?;
    game.finished = step.winner.is_some() || game.environment.arena.tick() == 108900;
    if !game.finished {
        return Ok(None);
    }
    let side = 1 - game.environment.policy_seat;
    let expert_won = step.winner == Some(game.environment.seats[side].tracker.team());
    game.expert_won = expert_won;
    let samples = game.corpus.take().expect("corpus").finish(expert_won);
    eprintln!(
        "tail_game seed={} expert_side={side} expert_won={expert_won} tick={} retained={}",
        game.seed,
        game.environment.arena.tick(),
        samples.len()
    );
    Ok(Some(samples))
}

#[test]
#[ignore = "Bounded real winning-tail BC pilot; no model architecture or Teacher changes"]
fn probe_real_winning_tail_imitation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root.join("artifacts/temp/neural-reset-20260908/tail-bc-001");
    assert!(!output.exists());
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let model = PolicyModel::fresh_on(10089150, device).expect("model");
    TrainingArtifact::load_runtime_weights(
        &model,
        &root.join("artifacts/temp/input-facts-m12-u10-init"),
    )
    .expect("frozen neural opponent");
    eprintln!(
        "tail_profile opponent=input-facts-m12-u10-init initialization=finish-skill-alpha050 teacher_unchanged=true"
    );
    let (pool, coverage) = collect_tail_pool(&model);
    TrainingArtifact::load_runtime_weights(
        &model,
        &root.join("artifacts/temp/neural-reset-20260908/finish-skill-alpha-050"),
    )
    .expect("training initialization");
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
        &model,
        &pool,
    )
    .expect("fresh optimizer");
    std::fs::create_dir(&output).expect("output");
    let started = Instant::now();
    for epoch in 1..=8 {
        let trained = trainer.train_epoch(&model, &pool).expect("epoch");
        let metrics = OfflineEvaluation::evaluate_validation(&model, &pool, coverage.clone())
            .expect("validation");
        eprintln!(
            "tail_bc epoch={epoch} loss={} validation={:?}",
            trained.average_loss,
            metrics.metrics()
        );
        if epoch == 4 || epoch == 8 {
            let checkpoint = output.join(format!("epoch-{epoch:03}"));
            std::fs::create_dir(&checkpoint).expect("checkpoint");
            TrainingArtifact::save_runtime_weights(&model, &checkpoint).expect("weights");
            std::fs::write(checkpoint.join("PROVENANCE.txt"), format!("real_seat_frames=true\nwinning_tail_decisions=1200\nopening=32\nrare_cap_per_kind=16\ncontinue_ratio_not_forced=true\nlearning_rate=3e-5\nepoch={epoch}\n{trained:?}\n{:?}\nqualified=false\n", metrics.metrics())).expect("provenance");
        }
        assert!(started.elapsed() < Duration::from_secs(600));
    }
}

#[test]
fn tail_corpus_keeps_only_a_bounded_winning_tail_and_preserves_opening_on_loss() {
    let mut arena =
        build_environment(10089000, 10, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let (frame, space) = prepare_policy_sample(&mut arena).expect("frame");
    let mut winning = TailCorpus::new(10089180);
    let mut losing = TailCorpus::new(10089180);
    for trajectory in 0..1600 {
        let identity = SampleIdentity::from_frame(
            SeedNamespace::Training,
            10089000,
            trajectory,
            space.tick(),
            &frame,
        )
        .expect("identity");
        let sample = ImitationSample::teacher(
            frame.clone(),
            &space,
            crate::StructuredAction::Continue,
            identity,
        )
        .expect("fixture sample");
        winning.retain(sample.clone());
        losing.retain(sample);
    }
    let won = winning.finish(true);
    let lost = losing.finish(false);
    assert_eq!(won.len(), 1232);
    assert_eq!(lost.len(), 32);
    assert!(
        lost.iter()
            .all(|sample| sample.identity().trajectory() < 32)
    );
}
