use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use super::*;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn worker_error_arbitration_preserves_real_failure_in_both_completion_orders() {
    let failure = invalid("simulator rejected an order");
    for errors in [
        [TacticalSearchError::Deadline, failure.clone()],
        [failure.clone(), TacticalSearchError::Deadline],
    ] {
        let mut selected = None;
        for error in errors {
            record_worker_error(&mut selected, error);
        }
        assert_eq!(selected, Some(failure.clone()));
        assert_eq!(
            selected.expect("error").to_string(),
            "simulator rejected an order"
        );
    }
}

#[test]
fn confirmation_only_deadline_sets_the_report_stop_flag() {
    let mut budget = SearchBudget::new(Duration::ZERO);
    let mut stopped = false;
    let confirmation = confirm_selection(
        &TacticalSearchConfig::default(),
        Path::new("unused"),
        &TacticalPolicy::default(),
        &mut budget,
        &mut stopped,
    )
    .expect("recoverable deadline");
    assert!(confirmation.is_none());
    assert!(stopped);
}

#[test]
fn mixed_fitness_prefers_worst_opponent_wins_over_pooled_wins() {
    let policy = TacticalPolicy::default();
    let cohort = TacticalCohort::new(9_203_000, 2, 30_000)
        .expect("cohort")
        .with_opponent(Some(&policy));
    let balanced = mixed_fitness(cohort, [3, 3]);
    let pooled = mixed_fitness(cohort, [4, 2]);
    assert_eq!(balanced.wins, pooled.wins);
    assert_eq!(
        balanced.compare(&pooled).expect("same opponents"),
        Ordering::Greater
    );
    let lopsided = mixed_fitness(cohort, [4, 1]);
    let weaker_total = mixed_fitness(cohort, [2, 2]);
    assert_eq!(
        weaker_total.compare(&lopsided).expect("same opponents"),
        Ordering::Greater
    );
}

#[test]
fn mixed_fitness_rejects_another_opponent_or_game_identity() {
    let policy = TacticalPolicy::default();
    let cohort = TacticalCohort::new(9_203_000, 2, 30_000)
        .expect("cohort")
        .with_opponent(Some(&policy));
    let other = TacticalCohort::new(9_203_000, 2, 30_000)
        .expect("cohort")
        .with_opponent(Some(&policy.mutated(7, 0.2).expect("other opponent")));
    assert_eq!(
        mixed_fitness(cohort, [2, 2])
            .compare(&mixed_fitness(other, [2, 2]))
            .expect_err("wrong opponent")
            .to_string(),
        "tactical fitness comparison requires identical seed, side and tick-limit cohorts"
    );
    let mut games = mixed_games(cohort, [2, 2]);
    games[4].opponent_hash = None;
    assert_eq!(
        TacticalFitness::from_games(cohort, &games)
            .expect_err("missing opponent identity")
            .to_string(),
        "tactical game opponent does not match its cohort identity"
    );
}

#[test]
fn fixed_v004_on_both_seats_has_identical_orders_and_opposite_candidate_results() {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/v0.0.4/drysua.tactical.bin"),
    )
    .expect("released opponent");
    let policy = TacticalPolicy::from_bytes(&bytes).expect("V1 artifact");
    let settings = TacticalMatchConfig {
        seed: 9_203_000,
        candidate_seat: 0,
        tick_limit: 30_000,
    };
    let radiant =
        evaluate_tactical_match_against(Some(&policy), Some(&policy), settings).expect("Radiant");
    let dire = evaluate_tactical_match_against(
        Some(&policy),
        Some(&policy),
        TacticalMatchConfig {
            candidate_seat: 1,
            ..settings
        },
    )
    .expect("Dire");
    assert_eq!(radiant.order_fingerprints, dire.order_fingerprints);
    assert_eq!(radiant.ticks, dire.ticks);
    assert_eq!(radiant.opponent_hash, Some(policy_digest(&policy)));
    assert!(matches!(
        (radiant.outcome, dire.outcome),
        (TacticalMatchOutcome::Win, TacticalMatchOutcome::Loss)
            | (TacticalMatchOutcome::Loss, TacticalMatchOutcome::Win)
            | (TacticalMatchOutcome::Timeout, TacticalMatchOutcome::Timeout)
    ));
}

#[test]
fn default_neural_opponent_matches_the_pure_teacher_path() {
    let policy = tactical_search_founders()[1].clone();
    let settings = TacticalMatchConfig {
        seed: 9_203_001,
        candidate_seat: 0,
        tick_limit: 6_001,
    };
    let teacher = evaluate_tactical_match(Some(&policy), settings).expect("Teacher");
    let neural =
        evaluate_tactical_match_against(Some(&policy), Some(&TacticalPolicy::default()), settings)
            .expect("default");
    assert_eq!(teacher.order_fingerprints, neural.order_fingerprints);
    assert_eq!(teacher.outcome, neural.outcome);
}

fn mixed_games(cohort: TacticalCohort, wins: [usize; 2]) -> Vec<TacticalMatchReport> {
    (0..cohort.game_count())
        .map(|index| {
            let opponent = index / (cohort.pairs * 2);
            let settings = cohort.match_settings(index);
            let outcome = if index % (cohort.pairs * 2) < wins[opponent] {
                TacticalMatchOutcome::Win
            } else {
                TacticalMatchOutcome::Loss
            };
            let mut game = game(settings.seed, settings.candidate_seat, outcome);
            game.opponent_hash = if opponent == 0 {
                None
            } else {
                cohort.opponent_hash
            };
            game
        })
        .collect()
}

fn mixed_fitness(cohort: TacticalCohort, wins: [usize; 2]) -> TacticalFitness {
    TacticalFitness::from_games(cohort, &mixed_games(cohort, wins)).expect("mixed fitness")
}

#[test]
fn live_tactical_wire_orders_match_training_on_the_same_visible_stream() {
    for candidate_seat in 0..2 {
        let policy = tactical_search_founders()[TacticalMode::Recover.index()].clone();
        let (messages, expected) = recorded_stream(&policy, candidate_seat);
        let mut wire = ReplayWire {
            messages: messages.into(),
            tick: 0,
            orders: 0,
            orders_hash: DefaultHasher::new(),
        };
        let seated = crate::Seated {
            player: bota_proto::PlayerId(0),
            slot: SlotId(candidate_seat),
            tick_rate: 30,
            mode: bota_proto::TickMode::Lockstep,
        };

        let outcome = crate::play_tactical_on(&mut wire, seated, Some(6_001), &policy)
            .expect("replayed deployment");

        assert_eq!(wire.orders_hash.finish(), expected);
        assert_eq!(outcome.orders, wire.orders);
        assert!(outcome.decisions > 0);
    }
}

#[test]
fn deadline_discards_incomplete_comparison_without_replacing_archived_champion() {
    let root = directory("deadline");
    std::fs::create_dir(&root).expect("root");
    let policy = TacticalPolicy::default();
    let cohort = TacticalCohort::new(9_200_100, 1, 30_000).expect("cohort");
    let evaluated = evaluation(
        cohort,
        vec![
            game(cohort.first_seed, 0, TacticalMatchOutcome::Win),
            game(cohort.first_seed, 1, TacticalMatchOutcome::Loss),
        ],
    )
    .expect("evaluation");
    let archive =
        save_tactical_archive(&root, "selected-000", &policy, &evaluated).expect("archive");
    update_best_pointer(&root, &archive).expect("best pointer");
    let mut state = SearchState {
        center: policy.clone(),
        policy: policy.clone(),
        selection: evaluated,
        archive,
        completed_generations: 0,
        stopped_for_deadline: false,
    };
    let mut budget = SearchBudget::new(Duration::ZERO);

    let error = search_generation(
        &TacticalSearchConfig::default(),
        &root,
        0,
        &mut state,
        &mut budget,
    )
    .expect_err("expired budget");

    assert_eq!(
        error.to_string(),
        "tactical monotonic wall-time budget exhausted"
    );
    assert_eq!(state.policy, policy);
    assert_eq!(
        std::fs::read_to_string(root.join("best.txt")).expect("best pointer"),
        "selected-000\n"
    );
    assert!(!root.join("generation-001").exists());
    std::fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn paired_confidence_diagnostic_does_not_treat_seats_as_independent() {
    let cohort = TacticalCohort::new(9_200_100, 16, 30_000).expect("cohort");
    let games = (0..32)
        .map(|index| {
            game(
                cohort.match_settings(index).seed,
                (index % 2) as u8,
                TacticalMatchOutcome::Win,
            )
        })
        .collect::<Vec<_>>();
    let perfect = TacticalFitness::from_games(cohort, &games).expect("all wins");

    assert_eq!(perfect.paired_sweeps, 16);
    assert!(perfect.paired_sweep_lower_percent() > 80.0);
    assert!(perfect.paired_sweep_lower_percent() < 81.0);
}

#[test]
fn fitness_rejects_a_terminal_win_at_the_deployment_tick_limit() {
    let cohort = TacticalCohort::new(9_200_100, 1, 30_000).expect("cohort");
    let mut games = [
        game(cohort.first_seed, 0, TacticalMatchOutcome::Win),
        game(cohort.first_seed, 1, TacticalMatchOutcome::Loss),
    ];
    games[0].ticks = cohort.tick_limit;

    assert_eq!(
        TacticalFitness::from_games(cohort, &games)
            .expect_err("terminal at limit")
            .to_string(),
        "tactical outcome must agree with the deployment tick-limit boundary"
    );
}

#[test]
fn terminal_wins_outrank_every_bounded_surrogate_advantage() {
    let cohort = TacticalCohort::new(9_200_000, 1, 30_000).expect("cohort");
    let winner = fitness(
        cohort,
        [TacticalMatchOutcome::Win, TacticalMatchOutcome::Loss],
        -2,
        -500,
    );
    let loser = fitness(cohort, [TacticalMatchOutcome::Loss; 2], 2, 500);

    assert_eq!(
        winner.compare(&loser).expect("same cohort"),
        Ordering::Greater
    );
}

#[test]
fn timeouts_are_not_wins_and_lose_terminal_ties() {
    let cohort = TacticalCohort::new(9_200_000, 1, 30_000).expect("cohort");
    let terminal = fitness(
        cohort,
        [TacticalMatchOutcome::Win, TacticalMatchOutcome::Loss],
        -2,
        -500,
    );
    let timeout = fitness(
        cohort,
        [TacticalMatchOutcome::Win, TacticalMatchOutcome::Timeout],
        2,
        500,
    );

    assert_eq!(terminal.wins, timeout.wins);
    assert_eq!(
        terminal.compare(&timeout).expect("same cohort"),
        Ordering::Greater
    );
}

#[test]
fn fitness_rejects_comparison_across_seed_or_horizon_cohorts() {
    let original = TacticalCohort::new(9_200_000, 1, 30_000).expect("cohort");
    let current = fitness(original, [TacticalMatchOutcome::Win; 2], 0, 0);
    for cohort in [
        TacticalCohort::new(9_200_001, 1, 30_000).expect("other seed"),
        TacticalCohort::new(9_200_000, 1, 29_999).expect("other horizon"),
    ] {
        let other = fitness(cohort, [TacticalMatchOutcome::Win; 2], 0, 0);

        assert_eq!(
            current
                .compare(&other)
                .expect_err("different cohorts")
                .to_string(),
            "tactical fitness comparison requires identical seed, side and tick-limit cohorts"
        );
    }
}

#[test]
fn fitness_rejects_duplicate_missing_or_rejected_games() {
    let cohort = TacticalCohort::new(9_200_000, 1, 30_000).expect("cohort");
    let mut games = [
        game(cohort.first_seed, 0, TacticalMatchOutcome::Win),
        game(cohort.first_seed, 1, TacticalMatchOutcome::Loss),
    ];
    assert_eq!(
        TacticalFitness::from_games(cohort, &games[..1])
            .expect_err("missing game")
            .to_string(),
        "tactical cohort has 1 games; expected 2"
    );
    games[1].settings.candidate_seat = 0;
    assert_eq!(
        TacticalFitness::from_games(cohort, &games)
            .expect_err("duplicate seat")
            .to_string(),
        "tactical cohort games must match each paired seed and both seats in order"
    );
    games[1].settings.candidate_seat = 1;
    games[0].rejections[0] = 1;
    assert_eq!(
        TacticalFitness::from_games(cohort, &games)
            .expect_err("rejected order")
            .to_string(),
        "tactical fitness requires zero rejected orders from both seats"
    );
}

#[test]
fn development_seed_namespace_and_cohort_bounds_fail_closed() {
    for (seed, pairs, ticks, expected) in [
        (
            9_199_999,
            1,
            30_000,
            "tactical seeds must remain in development namespace 9200000..9300000",
        ),
        (
            9_299_999,
            2,
            30_000,
            "tactical seeds must remain in development namespace 9200000..9300000",
        ),
        (
            9_200_000,
            0,
            30_000,
            "tactical paired seeds must be in 1..=16",
        ),
        (
            9_200_000,
            17,
            30_000,
            "tactical paired seeds must be in 1..=16",
        ),
        (
            9_200_000,
            1,
            30_001,
            "tactical tick limit must be in 2..=30000",
        ),
    ] {
        assert_eq!(
            TacticalCohort::new(seed, pairs, ticks)
                .expect_err("invalid cohort")
                .to_string(),
            expected
        );
    }
}

#[test]
fn default_policy_matches_teacher_full_game_orders_on_both_seats() {
    let settings = TacticalMatchConfig {
        seed: 9_200_003,
        candidate_seat: 0,
        tick_limit: 30_000,
    };
    let baseline = evaluate_tactical_match(None, settings).expect("Teacher mirror");
    assert_ne!(baseline.outcome, TacticalMatchOutcome::Timeout);
    for candidate_seat in 0..2 {
        let candidate = evaluate_tactical_match(
            Some(&TacticalPolicy::default()),
            TacticalMatchConfig {
                candidate_seat,
                ..settings
            },
        )
        .expect("default residual");

        assert_eq!(candidate.ticks, baseline.ticks);
        assert_eq!(candidate.order_fingerprints, baseline.order_fingerprints);
        assert_eq!(candidate.orders, baseline.orders);
        assert_eq!(candidate.rejections, [0, 0]);
        assert_eq!(candidate.effective_overrides, 0);
        assert!(candidate.sampled_decisions > 0);
    }
}

#[test]
fn pregame_and_tick_cap_do_not_create_decisions_or_terminal_wins() {
    let game = evaluate_tactical_match(
        Some(&TacticalPolicy::default()),
        TacticalMatchConfig {
            seed: 9_200_000,
            candidate_seat: 0,
            tick_limit: 901,
        },
    )
    .expect("bounded game");

    assert_eq!(game.ticks, 901);
    assert_eq!(game.decisions, [0, 0]);
    assert_eq!(game.outcome, TacticalMatchOutcome::Timeout);
}

#[test]
fn population_mutations_are_reproducible_preserve_anchor_and_explore_only_head_initially() {
    let parent = TacticalPolicy::default();
    let first = population(&parent, 16, 1, 9_200_000).expect("population");
    let repeated = population(&parent, 16, 1, 9_200_000).expect("population");

    assert_eq!(first, repeated);
    assert_eq!(first.len(), 16);
    assert_eq!(first[0], parent);
    assert_eq!(first[1], TacticalPolicy::default());
    for policy in &first[2..] {
        assert_eq!(
            &policy.parameters()[..OUTPUT_OFFSET],
            &parent.parameters()[..OUTPUT_OFFSET]
        );
        assert!(TacticalPolicy::from_parameters(policy.parameters()).is_ok());
    }
    assert!(first[2..].iter().any(|policy| policy != &parent));
}

#[test]
fn search_rejects_overlapping_selection_and_training_seeds() {
    let config = TacticalSearchConfig {
        selection_seed: 9_200_000,
        ..TacticalSearchConfig::default()
    };

    assert_eq!(
        validate_search(&config).expect_err("overlap").to_string(),
        "tactical training, selection and confirmation seed ranges must be disjoint"
    );
}

#[test]
fn tiny_search_is_deterministic_across_worker_counts_and_saves_round_trip_policy() {
    let first_directory = directory("search-one");
    let other_directory = directory("search-two");
    let mut config = TacticalSearchConfig {
        population: 4,
        generations: 2,
        pairs: 1,
        selection_pairs: 1,
        workers: 1,
        tick_limit: 904,
        ..TacticalSearchConfig::default()
    };
    let first = run_tactical_search(&config, &first_directory, None).expect("first search");
    config.workers = 2;
    let other = run_tactical_search(&config, &other_directory, None).expect("second search");

    assert_eq!(first.policy, other.policy);
    assert_eq!(first.selection, other.selection);
    assert_eq!(first.confirmation, other.confirmation);
    assert_eq!(first.evaluated_games, other.evaluated_games);
    let bytes =
        std::fs::read(first.archive.join(TACTICAL_SEARCH_POLICY_FILE)).expect("saved policy");
    assert_eq!(
        TacticalPolicy::from_bytes(&bytes).expect("archive round trip"),
        first.policy
    );
    let metadata = std::fs::read_to_string(first.archive.join("report.json")).expect("metadata");
    assert!(metadata.contains("\"first_seed\":9200100"));
    assert!(metadata.contains("\"policy_sha256\""));
    std::fs::remove_dir_all(first_directory).expect("cleanup first");
    std::fs::remove_dir_all(other_directory).expect("cleanup other");
}

#[test]
fn archive_rejects_overwrite_and_preserves_policy_bytes() {
    let root = directory("archive");
    std::fs::create_dir(&root).expect("root");
    let cohort = TacticalCohort::new(9_200_100, 1, 30_000).expect("cohort");
    let games = vec![
        game(cohort.first_seed, 0, TacticalMatchOutcome::Win),
        game(cohort.first_seed, 1, TacticalMatchOutcome::Loss),
    ];
    let evaluation = evaluation(cohort, games).expect("evaluation");
    let policy = TacticalPolicy::default();
    let archive =
        save_tactical_archive(&root, "selected-000", &policy, &evaluation).expect("archive");
    let before = std::fs::read(archive.join(TACTICAL_SEARCH_POLICY_FILE)).expect("before");

    assert_eq!(
        save_tactical_archive(&root, "selected-000", &policy, &evaluation)
            .expect_err("overwrite")
            .to_string(),
        "tactical archive already exists: selected-000"
    );
    assert_eq!(
        std::fs::read(archive.join(TACTICAL_SEARCH_POLICY_FILE)).expect("after"),
        before
    );
    assert_eq!(
        TacticalPolicy::from_bytes(&before).expect("round trip"),
        policy
    );
    std::fs::remove_dir_all(root).expect("cleanup");
}

fn fitness(
    cohort: TacticalCohort,
    outcomes: [TacticalMatchOutcome; 2],
    deaths: i32,
    farm: i64,
) -> TacticalFitness {
    let mut games = outcomes
        .into_iter()
        .enumerate()
        .map(|(seat, outcome)| game(cohort.first_seed, seat as u8, outcome))
        .collect::<Vec<_>>();
    for game in &mut games {
        game.settings.tick_limit = cohort.tick_limit;
        game.final_summary.enemy.deaths = deaths.max(0) as u64;
        game.final_summary.allied.deaths = (-deaths).max(0) as u64;
        game.final_summary.allied.last_hits = farm.max(0) as u64;
        game.final_summary.enemy.last_hits = (-farm).max(0) as u64;
    }
    TacticalFitness::from_games(cohort, &games).expect("fitness")
}

fn game(seed: u64, candidate_seat: u8, outcome: TacticalMatchOutcome) -> TacticalMatchReport {
    TacticalMatchReport {
        opponent_hash: None,
        settings: TacticalMatchConfig {
            seed,
            candidate_seat,
            tick_limit: 30_000,
        },
        outcome,
        ticks: if outcome == TacticalMatchOutcome::Timeout {
            30_000
        } else {
            1_000
        },
        decisions: [33; 2],
        orders: [3; 2],
        rejections: [0; 2],
        order_fingerprints: [0; 2],
        sampled_decisions: 0,
        effective_overrides: 0,
        final_summary: crate::GlobalSummary::default(),
    }
}

struct ReplayWire {
    messages: std::collections::VecDeque<ServerMsg>,
    tick: u32,
    orders: u32,
    orders_hash: DefaultHasher,
}

impl crate::Wire for ReplayWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        let message = self.messages.pop_front();
        if let Some(ServerMsg::Snapshot { view }) = &message {
            self.tick = view.tick;
        }
        Ok(message)
    }
    fn order(
        &mut self,
        unit: Option<bota_proto::EntityId>,
        order: bota_proto::Order,
    ) -> std::io::Result<u32> {
        assert!(self.orders < 2_000);
        assert!(self.tick > 900);
        self.orders += 1;
        (self.tick, unit, order).hash(&mut self.orders_hash);
        Ok(self.orders)
    }
    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        assert_eq!(tick, self.tick);
        Ok(())
    }
}

fn recorded_stream(policy: &TacticalPolicy, candidate_seat: u8) -> (Vec<ServerMsg>, u64) {
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 9_200_400,
    })
    .expect("arena");
    let mut seats = [
        new_seat(0, &start.messages[0]).expect("Radiant"),
        new_seat(1, &start.messages[1]).expect("Dire"),
    ];
    let candidate = usize::from(candidate_seat);
    let mut messages = start.messages[candidate].clone();
    for _ in 1..6_001 {
        let mut requests = [None, None];
        if arena.tick() > 900 && (arena.tick() - 901).is_multiple_of(3) {
            for (index, seat) in seats.iter_mut().enumerate() {
                requests[index] = seat_request(
                    seat,
                    if index == candidate {
                        Some(policy)
                    } else {
                        None
                    },
                )
                .expect("request");
            }
        }
        let step = arena.step(&requests).expect("step");
        messages.extend(step.messages[candidate].clone());
        let mut terminal = false;
        for (seat, stream) in seats.iter_mut().zip(&step.messages) {
            terminal |= observe_messages(seat, stream).expect("stream").is_some();
        }
        if terminal {
            break;
        }
    }
    assert!(messages.len() <= 18_004);
    assert!(seats[candidate].effective_overrides > 0);
    (messages, seats[candidate].orders_hash.finish())
}

fn directory(label: &str) -> PathBuf {
    let index = NEXT_DIRECTORY.fetch_add(1, AtomicOrdering::Relaxed);
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .join(format!(
            "tactical-test-{label}-{}-{index}",
            std::process::id()
        ))
}
