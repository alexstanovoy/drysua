use super::*;
use bota_proto::Target;

#[path = "history_probe_replay_codec.rs"]
mod codec;

#[test]
#[ignore = "Recover the same eight recorded trajectories without any model sampling or new cohort"]
fn replay_recorded_history_and_fit_matched_models() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let original = root.join("artifacts/temp/neural-reset-20260908/training-contract-bc");
    let output = std::env::var_os("DRYSUA_HISTORY_REPLAY_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| original.join("replay-fit"));
    assert_eq!(output.parent(), Some(original.as_path()));
    assert!(output.join("manifest.json").is_file());
    assert!(!original.join("baseline-epoch004").exists());
    assert!(!original.join("baseline-epoch008").exists());
    let (pools, coverage) = replay_recorded_pools(&original, &output);
    let started = Instant::now()
        .checked_sub(Duration::from_secs(4))
        .expect("previous failed audit allowance");
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let initial = root.join(INITIAL);
    assert_eq!(
        file_sha256(&initial.join("drysua.weights.safetensors")),
        INITIAL_SHA256
    );
    let parent = PolicyModel::fresh_on(10089200, device).expect("unchanged INITIAL parent");
    TrainingArtifact::load_runtime_weights(&parent, &initial).expect("M14 initialization");
    let parameters = zero_probe_input_weights(&parent);
    assert_eq!(
        parameter_sha256(&parameters),
        "c23b5357787f67ab2d64af4282f2147841ba61635673ecdf809dfa75957c99be"
    );
    let models: [_; 2] = std::array::from_fn(|_| {
        let model = PolicyModel::fresh_on(10089200, device).expect("fresh learner");
        model
            .import_parameters(&parameters)
            .expect("matched zero-row INITIAL");
        model
    });
    write_retained_rows(&pools, &output);
    assert_eq!(
        file_sha256(&output.join("rows.tsv")),
        file_sha256(&original.join("replay-fit/rows.tsv")),
        "same recovered sample identities, targets and all frame tensors"
    );
    assert_initial_outputs(&models, &pools);
    fit_matched_models(&models, &pools, &coverage, &output, &original, started);
    eprintln!(
        "history_complete fit_seconds={:.3} enriched_exported=false automatic_promotion=false",
        started.elapsed().as_secs_f64()
    );
}

fn replay_recorded_pools(
    original: &Path,
    output: &Path,
) -> ([ImitationPool; 2], [TeacherCoverage; 2]) {
    let started = Instant::now();
    let mut games = history_games();
    let records: Vec<Vec<_>> = games
        .iter()
        .map(|game| read_transcript(&transcript_path(original, game)))
        .collect();
    open_transcripts(&mut games, output);
    let mut pools = std::array::from_fn(|_| new_history_pool());
    let mut coverage = std::array::from_fn(|_| TeacherCoverage::new());
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(933),
            "remaining collection/replay budget"
        );
        let active: Vec<_> = (0..8)
            .filter(|&index| !games[index].finished)
            .map(|index| {
                (
                    index,
                    &records[index][games[index].counts.iter().sum::<u32>() as usize],
                )
            })
            .collect();
        if active.is_empty() {
            break;
        }
        let completed = parallel::ordered_active(&mut games, active, |_, game, record| {
            replay_round(game, record)
        })
        .expect("exact recorded round");
        for pair in completed.into_iter().flatten() {
            for (index, samples) in pair.into_iter().enumerate() {
                for sample in samples {
                    if sample.identity().namespace() == SeedNamespace::Validation {
                        coverage[index]
                            .record_represented_for(&sample)
                            .expect("coverage");
                    }
                    assert!(pools[index].push(sample).expect("matched pool").is_none());
                }
            }
        }
    }
    for (game, rows) in games.iter().zip(records) {
        assert!(game.finished);
        assert_eq!(game.counts.iter().sum::<u32>() as usize, rows.len());
        let source = transcript_path(original, game);
        let target = transcript_path(output, game);
        assert_eq!(file_sha256(&source), file_sha256(&target));
        eprintln!(
            "history_replay seed={} expert_side={} transcript_sha256={} exact_original_requests_labels_previous_features=true newly_sampled_games=0",
            game.seed,
            1 - game.environment.policy_seat,
            file_sha256(&target)
        );
    }
    assert_matched_pools(&pools);
    assert_eq!(pools[0].len(), 10783);
    assert!(started.elapsed() < Duration::from_secs(933));
    eprintln!(
        "history_dataset rows={} replay_seconds={:.3} original_collection_seconds=167.543",
        pools[0].len(),
        started.elapsed().as_secs_f64()
    );
    (pools, coverage)
}

fn read_transcript(path: &Path) -> Vec<String> {
    assert!(path.metadata().expect("original transcript").len() < 64 * 1024 * 1024);
    let text = std::fs::read_to_string(path).expect("original transcript");
    let rows: Vec<_> = text.lines().skip(1).map(str::to_owned).collect();
    assert!(rows.len() <= 36300);
    assert!(!rows.is_empty());
    rows
}

fn transcript_path(root: &Path, game: &HistoryGame) -> std::path::PathBuf {
    assert!((10089000..10089004).contains(&game.seed));
    assert!(game.environment.policy_seat < 2);
    root.join(format!(
        "trajectory-{}-side{}.tsv",
        game.seed,
        1 - game.environment.policy_seat
    ))
}

fn replay_round(
    game: &mut HistoryGame,
    record: &str,
) -> Result<Option<[Vec<ImitationSample>; 2]>, PpoError> {
    replay_round_captured(game, record, None).map(|result| result.completed)
}

struct ConditioningReplay {
    completed: Option<[Vec<ImitationSample>; 2]>,
    captured: Option<(ConditioningRow, ActionSpace)>,
}

fn replay_round_captured(
    game: &mut HistoryGame,
    record: &str,
    capture: Option<usize>,
) -> Result<ConditioningReplay, PpoError> {
    assert!(record.len() < 2048);
    assert!(!game.finished);
    let columns: Vec<_> = record.split('\t').collect();
    assert_eq!(columns.len(), 5);
    let tick: u32 = columns[0].parse().expect("recorded tick");
    assert_eq!(game.environment.arena.tick(), tick);
    let requests = [
        codec::parse_request(columns[2]).expect("recorded Radiant request"),
        codec::parse_request(columns[3]).expect("recorded Dire request"),
    ];
    let actor = game.environment.policy_seat;
    replay_candidate_send(&mut game.environment, requests[actor]);
    let capture = capture.map(|index| {
        let expert = &game.environment.seats[1 - actor];
        let space = ActionSpace::from_tracker_with_readiness(&expert.tracker, &expert.readiness)
            .expect("original pre-send Teacher space");
        (index, space)
    });
    let (samples, target_context) = replay_teacher_observation(game, &requests);
    let captured = capture.map(|(index, space)| {
        let mut row = ConditioningRow::new(index, samples[0].clone(), &space);
        row.target_summary = active_target::summary(target_context, &space, row.raw.frame());
        (row, space)
    });
    assert_eq!(format!("{:?}", samples[0].teacher_action()), columns[1]);
    assert_eq!(
        format!("{:?}", &samples[1].frame().global[59..64]),
        columns[4]
    );
    record_history_round(game, &requests, &samples);
    for (corpus, sample) in game.corpora.iter_mut().zip(samples) {
        corpus.as_mut().expect("corpus").retain(sample);
    }
    let step = advance_interval(
        &mut game.environment,
        requests.to_vec(),
        3.min(108900 - tick),
    )?;
    reject_production_rejection(&game.environment, "exact transcript replay")?;
    game.finished = step.winner.is_some() || game.environment.arena.tick() == 108900;
    if !game.finished {
        return Ok(ConditioningReplay {
            completed: None,
            captured,
        });
    }
    let side = 1 - game.environment.policy_seat;
    let won = step.winner == Some(game.environment.seats[side].tracker.team());
    assert!(won, "original eight games all ended in Teacher wins");
    finish_transcript(game);
    eprintln!(
        "history_game seed={} expert_side={side} won={won} tick={} frequencies={:?}",
        game.seed,
        game.environment.arena.tick(),
        game.counts
    );
    Ok(ConditioningReplay {
        completed: Some(std::array::from_fn(|index| {
            game.corpora[index].take().expect("corpus").finish(won)
        })),
        captured,
    })
}

struct ConditioningGame {
    game: HistoryGame,
    rows: Vec<(ConditioningRow, ActionSpace)>,
    wanted: std::collections::BTreeMap<u32, (usize, String, String)>,
}

pub(super) fn reconstruct_conditioning_rows(
    original: &Path,
    output: &Path,
) -> (Vec<ConditioningRow>, Vec<ActionSpace>) {
    let started = Instant::now();
    let mut games = history_games();
    let records: Vec<_> = games
        .iter()
        .map(|game| read_transcript(&transcript_path(original, game)))
        .collect();
    open_transcripts(&mut games, output);
    let mut games: Vec<_> = games
        .into_iter()
        .map(|game| ConditioningGame {
            game,
            rows: Vec::with_capacity(1488),
            wanted: Default::default(),
        })
        .collect();
    conditioning_wanted_rows(&mut games, original, output);
    for _ in 0..36300 {
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "conditioning reconstruction budget"
        );
        let active: Vec<_> = (0..8)
            .filter(|&index| !games[index].game.finished)
            .map(|index| {
                (
                    index,
                    &records[index][games[index].game.counts.iter().sum::<u32>() as usize],
                )
            })
            .collect();
        if active.is_empty() {
            break;
        }
        parallel::ordered_active(&mut games, active, |_, state, record| {
            conditioning_replay_round(state, record)
        })
        .expect("one exact replay");
    }
    let mut rows = Vec::with_capacity(10783);
    for (state, records) in games.into_iter().zip(records) {
        assert!(state.game.finished);
        assert_eq!(
            state.game.counts.iter().sum::<u32>() as usize,
            records.len()
        );
        assert_eq!(state.rows.len(), state.wanted.len());
        assert_eq!(
            file_sha256(&transcript_path(original, &state.game)),
            file_sha256(&transcript_path(output, &state.game))
        );
        rows.extend(state.rows);
    }
    rows.sort_unstable_by_key(|(row, _)| row.index);
    assert_eq!(rows.len(), 10783);
    for (index, (row, _)) in rows.iter().enumerate() {
        assert_eq!(index, row.index);
    }
    assert!(started.elapsed() < Duration::from_secs(600));
    eprintln!(
        "conditioning_reconstruction rows={} seconds={:.3} replay_executions=8 newly_sampled_games=0 all_transcript_and_raw_frame_hashes_match=true",
        rows.len(),
        started.elapsed().as_secs_f64()
    );
    rows.into_iter().unzip()
}

fn conditioning_replay_round(state: &mut ConditioningGame, record: &str) -> Result<(), PpoError> {
    let tick = state.game.environment.arena.tick();
    let selected = state.wanted.get(&tick).map(|entry| entry.0);
    let result = replay_round_captured(&mut state.game, record, selected)?;
    if let Some((row, space)) = result.captured {
        let expected = state.wanted.get(&tick).expect("preselected original row");
        assert_eq!(frame_sha256(row.raw.frame()), expected.1);
        assert_eq!(format!("{:?}", row.raw.teacher_action()), expected.2);
        assert!(state.rows.len() < 1488);
        state.rows.push((row, space));
    }
    assert!(state.rows.len() <= state.wanted.len());
    Ok(())
}

fn conditioning_wanted_rows(games: &mut [ConditioningGame], original: &Path, output: &Path) {
    let path = original.join("replay-fit-002/rows.tsv");
    assert_eq!(
        file_sha256(&path),
        "9d456eef81bbdd4daa34dcb567e56069c91899b85ccc8dbafd9c08ff955de6f3"
    );
    let text = std::fs::read_to_string(&path).expect("original retained manifest");
    let mut count = 0;
    for (index, line) in text.lines().skip(1).enumerate() {
        assert!(index < 10783);
        let columns: Vec<_> = line.split('\t').collect();
        assert_eq!(columns.len(), 10);
        assert_eq!(columns[0].parse::<usize>().expect("index"), index);
        let seed: u64 = columns[1].parse().expect("seed");
        assert!((10089000..10089004).contains(&seed));
        let side = match columns[3] {
            "Radiant" => 0,
            "Dire" => 1,
            _ => panic!("original side"),
        };
        let game = &mut games[(seed - 10089000) as usize * 2 + 1 - side];
        let tick: u32 = columns[2].parse().expect("tick");
        assert!((1..108900).contains(&tick));
        assert!(
            game.wanted
                .insert(tick, (index, columns[8].to_owned(), columns[6].to_owned()))
                .is_none()
        );
        count += 1;
    }
    assert_eq!(count, 10783);
    let mut copy = new_writer(&output.join("rows.tsv"));
    copy.write_all(text.as_bytes())
        .expect("identical original retained manifest");
    copy.flush().expect("retained manifest");
}

fn replay_candidate_send(environment: &mut TrainingEnvironment, recorded: Option<Request>) {
    let actor = environment.policy_seat;
    assert!(actor < 2);
    let (_, space) = prepare_neural_seat_policy_sample(&mut environment.seats[actor])
        .expect("explicit current Candidate replay");
    assert!(environment.seats[actor].order_bookkeeping.is_candidate());
    if let Some(recorded) = recorded {
        let kind = sent_kind(recorded.order);
        environment.seats[actor]
            .local
            .note_decision(space.tick(), kind)
            .expect("actual sent kind");
        let issued = crate::IssuedOrder {
            unit: recorded.unit,
            order: recorded.order,
        };
        let actual = issue_request(
            &mut environment.seats[actor],
            Some(issued),
            &space,
            kind,
            false,
        )
        .expect("Candidate replay transport");
        assert_eq!(
            actual,
            Some(recorded),
            "current Candidate suppression and sequence must match original"
        );
    }
}

fn replay_teacher_samples(
    game: &mut HistoryGame,
    requests: &[Option<Request>; 2],
) -> [ImitationSample; 2] {
    replay_teacher_observation(game, requests).0
}

pub(super) fn replay_teacher_observation(
    game: &mut HistoryGame,
    requests: &[Option<Request>; 2],
) -> ([ImitationSample; 2], active_target::TargetContext) {
    let expert = 1 - game.environment.policy_seat;
    let seat = &mut game.environment.seats[expert];
    let (_, teacher_space) =
        prepare_neural_observer_sample(seat).expect("original history preparation");
    let target_context = active_target::TargetContext::capture(seat);
    let previous = game.history.features(teacher_space.tick());
    let (frame, _) = prepare_neural_observer_sample(seat).expect("original label preparation");
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .expect("unchanged Teacher");
    let identity = SampleIdentity::from_frame(game.namespace, game.seed, 0, space.tick(), &frame)
        .expect("original identity");
    let baseline =
        ImitationSample::teacher(frame.clone(), &space, action, identity).expect("original label");
    seat.local
        .note_decision(space.tick(), action.kind())
        .expect("Teacher decision");
    let actual = issue_request(
        seat,
        space.decode(action).expect("original decode"),
        &space,
        action.kind(),
        true,
    )
    .expect("legacy Teacher transport");
    assert_eq!(actual, requests[expert]);
    assert!(!seat.order_bookkeeping.is_candidate());
    let mut enriched = frame;
    assert_eq!(enriched.global[59..64], [0.0; 5]);
    enriched.global[59..64].copy_from_slice(&previous);
    let enriched = ImitationSample::teacher(enriched, &teacher_space, action, identity)
        .expect("matched enrichment");
    game.history.sent(actual, action.kind(), space.tick());
    ([baseline, enriched], target_context)
}

fn sent_kind(order: Order) -> ActionKind {
    match order {
        Order::Move {
            target: Target::None,
        } => ActionKind::Stop,
        Order::Move {
            target: Target::Pos(_),
        } => ActionKind::MovePoint,
        Order::Move {
            target: Target::Unit(_),
        } => ActionKind::FollowUnit,
        Order::Attack {
            target: Target::None,
        } => ActionKind::Hold,
        Order::Attack {
            target: Target::Pos(_),
        } => ActionKind::AttackMovePoint,
        Order::Attack {
            target: Target::Unit(_),
        } => ActionKind::AttackUnit,
        Order::Cast { .. } => ActionKind::Cast,
        Order::Use { .. } => ActionKind::Use,
        Order::Put {
            target: Target::Unit(_),
            ..
        } => ActionKind::PutUnit,
        Order::Put { .. } => ActionKind::PutPoint,
        Order::Take { .. } => ActionKind::Take,
        Order::Buy { .. } => ActionKind::Buy,
        Order::Sell { .. } => ActionKind::Sell,
        Order::Swap { .. } => ActionKind::Swap,
        Order::Learn { .. } => ActionKind::Learn,
    }
}

#[test]
fn matched_contract_replay_recovers_identical_teacher_frames_from_actual_requests() {
    let mut replay = history_games().remove(0);
    let mut original = observed_teacher_game(10089000, 0);
    let model = PolicyModel::fresh(10089200).expect("instrument");
    let mut random = PpoRng::new(10089100);
    let mut history = SentOrderHistory::default();
    for _ in 0..16 {
        let (frame, space) = prepare_policy_sample(&mut original).expect("original frame");
        let choice = model
            .sample(&frame, &space, &mut random)
            .expect("original choice");
        let (requests, expected) = history_samples(
            &mut original,
            &choice,
            &space,
            10089000,
            SeedNamespace::Training,
            &mut history,
        )
        .expect("original samples");
        replay_candidate_send(&mut replay.environment, requests[0]);
        let actual = replay_teacher_samples(&mut replay, &[requests[0], requests[1]]);
        for index in 0..2 {
            assert_eq!(actual[index].frame(), expected[index].frame());
            assert_eq!(actual[index].identity(), expected[index].identity());
            assert_eq!(actual[index].target(), expected[index].target());
        }
        advance_interval(&mut original, requests.clone(), 3).expect("original advance");
        advance_interval(&mut replay.environment, requests, 3).expect("recorded advance");
    }
}

#[test]
fn conditioning_replay_capture_preserves_exact_frames_targets_and_pre_send_spaces() {
    for side in 0..2 {
        let mut replay = history_games().remove(side);
        let mut original = observed_teacher_game(10089000, side);
        let model = PolicyModel::fresh(10089500).expect("instrument");
        let mut random = PpoRng::new(10089500);
        let mut history = SentOrderHistory::default();
        for index in 0..8 {
            let (frame, space) = prepare_policy_sample(&mut original).expect("source frame");
            let choice = model
                .sample(&frame, &space, &mut random)
                .expect("source choice");
            let (requests, samples) = history_samples(
                &mut original,
                &choice,
                &space,
                10089000,
                SeedNamespace::Training,
                &mut history,
            )
            .expect("source samples");
            let record = format!(
                "{}\t{:?}\t{:?}\t{:?}\t{:?}",
                space.tick(),
                samples[0].teacher_action(),
                requests[0],
                requests[1],
                &samples[1].frame().global[59..64]
            );
            let captured =
                replay_round_captured(&mut replay, &record, Some(index)).expect("captured replay");
            let (row, teacher_space) = captured.captured.expect("selected row");
            assert_eq!(row.raw.frame(), samples[0].frame());
            assert_eq!(row.raw.identity(), samples[0].identity());
            assert_eq!(row.raw.target(), row.alternate.target());
            assert!(teacher_space.allows(row.raw.teacher_action()));
            assert!(row.alternate.frame().matches_action_space(&teacher_space));
            advance_interval(&mut original, requests, 3).expect("source advance");
        }
    }
}
