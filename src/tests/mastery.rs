use crate::{MasteryConfig, MasteryProgress, MasteryStage, TrainingGameOutcome as Outcome};

#[test]
fn mastery_counter_upper_boundary_rejects_next_game_without_mutation() {
    let config = MasteryConfig::new(1, 100, &[]).expect("config");
    let mut state = MasteryProgress::restore(
        MasteryStage::Weak,
        crate::MAX_TRAINING_COUNTER,
        vec![false],
        config,
    )
    .expect("maximum counter");
    let before = state.clone();
    assert_eq!(
        state.record_batch(config, &[Outcome::Win]),
        Err("mastery game counter exceeds bound")
    );
    assert_eq!(state, before);
    assert_eq!(
        MasteryProgress::restore(
            MasteryStage::Weak,
            crate::MAX_TRAINING_COUNTER + 1,
            vec![false],
            config
        ),
        Err("mastery game counter exceeds bound")
    );
    assert_eq!(
        MasteryConfig::new(1024, 1, &[])
            .expect("maximum window")
            .window(),
        1024
    );
}

#[test]
fn mastery_codec_is_retained_with_exact_reward5_linked_identities_and_appended_inputs() {
    assert_eq!(
        (crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH),
        (5, 10_658_390_830_565_586_343)
    );
    assert_eq!(
        (crate::FEATURE_SCHEMA_VERSION, crate::FEATURE_SCHEMA_HASH),
        (20, 9_233_114_641_639_769_206)
    );
    assert_eq!(
        (crate::MODEL_SCHEMA_VERSION, crate::MODEL_SCHEMA_HASH),
        (22, 4_891_874_295_003_631_291)
    );
    assert_eq!(
        (crate::PPO_SCHEMA_VERSION, crate::PPO_SCHEMA_HASH),
        (35, 13_569_352_384_922_890_857)
    );
    assert_eq!(
        (crate::LEAGUE_SCHEMA_VERSION, crate::LEAGUE_SCHEMA_HASH),
        (35, 7_630_384_836_954_837_061)
    );
    assert_eq!(
        (
            crate::CHECKPOINT_SCHEMA_VERSION,
            crate::CHECKPOINT_SCHEMA_HASH
        ),
        (10, 2_382_613_649_322_819_763)
    );
    assert_eq!(
        (
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH
        ),
        (6, 1_084_583_101_075_978_392)
    );
    assert_eq!(crate::MODEL_PARAMETER_COUNT, 1_700_020);
    assert_eq!(crate::GLOBAL_FEATURES, 92);
    assert_eq!(crate::UNIT_FEATURES, 84);
}

#[test]
fn mastery_completed_batch_order_is_terminal_tick_then_stream_not_arrival() {
    let mut batch = crate::CompletedTrainingEpisodes::default();
    batch
        .record(30, 1, Outcome::Win)
        .expect("later stream first");
    batch
        .record(29, 0, Outcome::Loss)
        .expect("earlier terminal second");
    assert_eq!(batch.ordered_outcomes(), [Outcome::Loss, Outcome::Win]);
    let config = MasteryConfig::new(1, 100, &[]).expect("one-game window");
    let mut state = MasteryProgress::default();
    state
        .record_batch(config, &batch.ordered_outcomes())
        .expect("batch boundary");
    assert_eq!(state.stage(), MasteryStage::Teacher);
    assert_eq!(state.games(), 0);
    let mut tied = crate::CompletedTrainingEpisodes::default();
    tied.record(30, 1, Outcome::Loss).expect("stream1");
    tied.record(30, 0, Outcome::Win).expect("stream0");
    assert_eq!(tied.ordered_outcomes(), [Outcome::Win, Outcome::Loss]);
    let mut state = MasteryProgress::default();
    state
        .record_batch(config, &tied.ordered_outcomes())
        .expect("evaluate after both games");
    assert_eq!(state.stage(), MasteryStage::Weak);
    assert_eq!(
        tied.record(30, 0, Outcome::Win),
        Err(crate::PpoError::InvalidTransition(
            "duplicate completed episode stream"
        ))
    );
    assert_eq!(tied.ordered_outcomes(), [Outcome::Win, Outcome::Loss]);
}

#[test]
fn mastery_forty_nine_wins_do_not_fill_a_fifty_game_window() {
    let config = MasteryConfig::default();
    let mut state = MasteryProgress::default();
    for batch in [Outcome::Win; 49].chunks(6) {
        state.record_batch(config, batch).expect("valid batch");
    }
    assert_eq!(state.stage(), MasteryStage::Weak);
    assert_eq!(state.recent().len(), 49);
    assert_eq!(state.wins(), 49);
}

#[test]
fn mastery_requires_forty_of_fifty_and_clears_only_on_stage_advance() {
    let config = MasteryConfig::default();
    for wins in [39, 40] {
        let mut state = MasteryProgress::default();
        let mut results = vec![Outcome::Win; wins];
        results.extend(vec![Outcome::Draw; 50 - wins]);
        for batch in results.chunks(6) {
            state.record_batch(config, batch).expect("batch");
        }
        if wins == 39 {
            assert_eq!(state.stage(), MasteryStage::Weak);
            assert_eq!(state.wins(), 39);
        } else {
            assert_eq!(state.stage(), MasteryStage::Teacher);
            assert_eq!(state.recent().len(), 0);
        }
    }
}

#[test]
fn mastery_uses_latest_ordered_window_not_all_time_or_consecutive_wins() {
    let config = MasteryConfig::new(5, 100, &[]).expect("config");
    let mut state = MasteryProgress::default();
    state
        .record_batch(
            config,
            &[
                Outcome::Win,
                Outcome::Win,
                Outcome::Win,
                Outcome::Win,
                Outcome::Loss,
            ],
        )
        .expect("partial success");
    assert_eq!(state.wins(), 4);
    state
        .record_batch(config, &[Outcome::TimeCap])
        .expect("evict oldest win");
    assert_eq!(state.wins(), 3);
    state
        .record_batch(config, &[Outcome::Win; 5])
        .expect("last five win");
    assert_eq!(state.stage(), MasteryStage::Teacher);
    state
        .record_batch(config, &[Outcome::Win; 5])
        .expect("finish Teacher");
    assert!(state.completed());
    assert_eq!(state.recent().len(), 5);
    let before = state.clone();
    assert_eq!(
        state.record_batch(config, &[Outcome::Loss]),
        Err("mastery already completed")
    );
    assert_eq!(state, before);
}

#[test]
fn mastery_nondivisible_window_uses_integer_ceiling_and_batch_boundary() {
    let config = MasteryConfig::new(7, 80, &[]).expect("config");
    let mut state = MasteryProgress::default();
    state
        .record_batch(
            config,
            &[
                Outcome::Loss,
                Outcome::Loss,
                Outcome::Win,
                Outcome::Win,
                Outcome::Win,
                Outcome::Win,
            ],
        )
        .expect("six");
    state
        .record_batch(config, &[Outcome::Win])
        .expect("five of seven");
    assert_eq!(state.stage(), MasteryStage::Weak);
    state
        .record_batch(config, &[Outcome::Win])
        .expect("six of seven");
    assert_eq!(state.stage(), MasteryStage::Teacher);
    let mut state = MasteryProgress::default();
    for _ in 0..8 {
        state
            .record_batch(MasteryConfig::default(), &[Outcome::Win; 6])
            .expect("batch");
        assert_eq!(state.stage(), MasteryStage::Weak);
    }
    state
        .record_batch(MasteryConfig::default(), &[Outcome::Win; 6])
        .expect("54 games boundary");
    assert_eq!(state.stage(), MasteryStage::Teacher);
    assert_eq!(state.recent().len(), 0);
}

#[test]
fn mastery_configuration_resolves_overrides_canonically_and_rejects_bad_inputs() {
    assert_eq!(
        "weak=00000000000000000000000000000000000080".parse::<crate::OpponentWinPercent>(),
        Err("opponent win percent is too long")
    );
    let weak = "weak=90".parse().expect("weak override");
    let teacher = "teacher=70".parse().expect("teacher override");
    let left = MasteryConfig::new(50, 80, &[weak, teacher]).expect("config");
    let right = MasteryConfig::new(50, 80, &[teacher, weak]).expect("same config");
    assert_eq!(left, right);
    assert_eq!(left.threshold(MasteryStage::Weak), 90);
    assert_eq!(left.threshold(MasteryStage::Teacher), 70);
    assert_eq!(
        MasteryConfig::new(0, 80, &[]),
        Err("mastery window must be in 1..=1024")
    );
    assert_eq!(
        MasteryConfig::new(1025, 80, &[]),
        Err("mastery window must be in 1..=1024")
    );
    for percent in [0, 101] {
        assert_eq!(
            MasteryConfig::new(50, percent, &[]),
            Err("mastery win percent must be in 1..=100")
        );
    }
    assert_eq!(
        MasteryConfig::new(50, 80, &[weak, weak]),
        Err("duplicate opponent win percent")
    );
    for text in [
        "unknown=80",
        "weak=80=90",
        "weak=80.5",
        "weak=-1",
        "weak=0",
        "teacher=101",
        "weak",
        "weak= 80",
    ] {
        assert!(text.parse::<crate::OpponentWinPercent>().is_err(), "{text}");
    }
}

#[test]
fn mastery_corrupt_state_and_invalid_batch_do_not_mutate_progress() {
    let config = MasteryConfig::new(5, 80, &[]).expect("config");
    assert_eq!(
        MasteryProgress::restore(MasteryStage::Weak, 6, vec![false; 6], config),
        Err("mastery window exceeds configured capacity")
    );
    assert_eq!(
        MasteryProgress::restore(MasteryStage::Completed, 4, vec![true; 4], config),
        Err("completed mastery requires a qualifying Teacher window")
    );
    assert_eq!(
        MasteryProgress::restore(MasteryStage::Weak, 0, vec![false], config),
        Err("mastery game counter/window mismatch")
    );
    let mut state = MasteryProgress::default();
    let before = state.clone();
    assert_eq!(
        state.record_batch(config, &[]),
        Err("mastery batch must contain 1..=6 completed games")
    );
    assert_eq!(
        state.record_batch(config, &[Outcome::Win; 7]),
        Err("mastery batch must contain 1..=6 completed games")
    );
    assert_eq!(state, before);
}
