use super::*;

const TEST_SEED: u64 = 7_700_000_101;

fn config(
    environments: usize,
    warmup_rounds: usize,
    pipeline_groups: usize,
) -> TrainingCollectionSliceConfig {
    TrainingCollectionSliceConfig {
        seed: TEST_SEED,
        environments,
        update: 0,
        warmup_rounds,
        pipeline_groups,
    }
}

#[test]
fn collection_slice_window_rejects_zero_and_past_ceiling() {
    assert_eq!(
        validate_slice_window(0, 0)
            .expect_err("empty window")
            .to_string(),
        "invalid PPO config field: collection slice rounds"
    );
    assert_eq!(
        validate_slice_window(0, TRAINING_COLLECTION_SLICE_MAX_ROUNDS + 1)
            .expect_err("window past the cap")
            .to_string(),
        "invalid PPO config field: collection slice rounds"
    );
    assert_eq!(
        validate_slice_window(TRAINING_COLLECTION_SLICE_MAX_ROUNDS + 1, 1)
            .expect_err("warmup past the cap")
            .to_string(),
        "invalid PPO config field: collection slice warmup"
    );
    assert_eq!(validate_slice_window(0, 4).expect("four rounds"), ());
    assert_eq!(
        validate_slice_window(TRAINING_COLLECTION_SLICE_MAX_ROUNDS, 4).expect("warmup at the cap"),
        ()
    );
}

#[test]
fn collection_slice_settings_bound_environments_and_warmup() {
    for environments in [0, TRAINING_MAX_ENVIRONMENTS + 1] {
        assert_eq!(
            collection_slice_settings(config(environments, 0, 1))
                .expect_err("environment count outside the maximum")
                .to_string(),
            "invalid PPO config field: collection slice environments"
        );
    }
    for environments in [3, 5] {
        assert_eq!(
            collection_slice_settings(config(environments, 0, 1))
                .expect_err("odd paired count")
                .to_string(),
            "invalid PPO config field: collection slice paired environment count"
        );
    }
    assert_eq!(
        collection_slice_settings(config(1, TRAINING_COLLECTION_SLICE_MAX_ROUNDS + 1, 1))
            .expect_err("warmup past the cap")
            .to_string(),
        "invalid PPO config field: collection slice warmup"
    );
    for environments in [1, 2, TRAINING_MAX_ENVIRONMENTS] {
        let settings =
            collection_slice_settings(config(environments, 64, 1)).expect("bounded slice settings");
        assert!(settings.complete_episodes);
        assert_eq!(settings.ppo.environments, environments);
        assert_eq!(settings.seed, TEST_SEED);
        assert_eq!(
            settings.opponent_schedule,
            crate::TrainingOpponentSchedule::MasteryV1
        );
        assert!(settings.mastery_config.is_some());
    }
}

#[test]
fn serial_collection_slice_reports_one_decision_per_round() {
    let mut slice = TrainingCollectionSlice::new(config(1, 0, 1), PolicyDevice::Cpu)
        .expect("serial collection slice");
    let report = slice.run(16).expect("sixteen serial decisions");

    assert_eq!(report.environments, 1);
    assert_eq!(report.warmup_rounds, 0);
    assert_eq!(report.rounds, 16);
    assert_eq!(report.decisions, 16);
    assert_eq!(report.ticks, 48);
    assert_eq!(report.start_tick, 1);
    assert_eq!(report.end_tick, 49);
    assert!((1..=2).contains(&report.retained_samples));
}

#[test]
fn parallel_collection_slice_reports_one_decision_per_stream_round() {
    let mut slice = TrainingCollectionSlice::new(config(2, 8, 1), PolicyDevice::Cpu)
        .expect("paired collection slice");
    let report = slice.run(16).expect("sixteen paired decisions");

    assert_eq!(report.environments, 2);
    assert_eq!(report.warmup_rounds, 8);
    assert_eq!(report.rounds, 16);
    assert_eq!(report.decisions, 32);
    assert_eq!(report.ticks, 96);
    assert_eq!(
        report.start_tick,
        1 + 8 * crate::MAP2_DECISION_INTERVAL_TICKS
    );
    assert_eq!(
        report.end_tick,
        report.start_tick + 16 * crate::MAP2_DECISION_INTERVAL_TICKS
    );
    assert!(report.retained_samples > 0);
}

#[test]
fn identical_collection_slices_run_identical_windows() {
    let mut first = TrainingCollectionSlice::new(config(2, 8, 1), PolicyDevice::Cpu)
        .expect("first collection slice");
    let first_summary = first.environments[0].seats[0]
        .tracker
        .latest_summary()
        .expect("first snapshot");
    let mut second = TrainingCollectionSlice::new(config(2, 8, 1), PolicyDevice::Cpu)
        .expect("second collection slice");
    let second_summary = second.environments[0].seats[0]
        .tracker
        .latest_summary()
        .expect("second snapshot");

    assert_eq!(first_summary, second_summary);
    assert_eq!(
        first.run(16).expect("first window"),
        second.run(16).expect("second window")
    );
}

#[test]
fn serial_slice_starts_from_the_paired_stream_zero_world() {
    let serial = TrainingCollectionSlice::new(config(1, 0, 1), PolicyDevice::Cpu)
        .expect("serial collection slice");
    let paired = TrainingCollectionSlice::new(config(2, 0, 1), PolicyDevice::Cpu)
        .expect("paired collection slice");

    assert_eq!(
        serial.environments[0].seats[0].tracker.latest_summary(),
        paired.environments[0].seats[0].tracker.latest_summary()
    );
}

#[test]
#[should_panic(expected = "collection slice is single-use")]
fn collection_slice_rejects_a_second_window() {
    let mut slice = TrainingCollectionSlice::new(config(1, 0, 1), PolicyDevice::Cpu)
        .expect("serial collection slice");
    slice.run(8).expect("first window");
    let _ = slice.run(8);
}

#[test]
fn pipelined_collection_slices_report_every_stream_round() {
    for groups in [2, 4] {
        let mut slice = TrainingCollectionSlice::new(config(8, 8, groups), PolicyDevice::Cpu)
            .expect("pipelined collection slice");
        let report = slice.run(16).expect("sixteen grouped decisions");

        assert_eq!(report.environments, 8);
        assert_eq!(report.warmup_rounds, 8);
        assert_eq!(report.rounds, 16);
        assert_eq!(report.decisions, 128);
        assert_eq!(report.ticks, 384);
        assert_eq!(
            report.end_tick,
            report.start_tick + 16 * crate::MAP2_DECISION_INTERVAL_TICKS
        );
        assert!(report.retained_samples > 0);
    }
}

#[test]
fn pipelined_collection_slices_run_identical_windows() {
    for groups in [2, 4] {
        let mut first = TrainingCollectionSlice::new(config(8, 8, groups), PolicyDevice::Cpu)
            .expect("first pipelined slice");
        let mut second = TrainingCollectionSlice::new(config(8, 8, groups), PolicyDevice::Cpu)
            .expect("second pipelined slice");
        assert_eq!(
            first.run(16).expect("first pipelined window"),
            second.run(16).expect("second pipelined window"),
            "groups={groups}"
        );
    }
}

#[test]
fn phased_collection_slice_accounts_for_every_pool_phase() {
    let mut slice = TrainingCollectionSlice::new(config(4, 8, 2), PolicyDevice::Cpu)
        .expect("phased collection slice");
    let (report, phases) = slice.run_phased(16).expect("phased window");

    assert_eq!(report.environments, 4);
    assert_eq!(report.decisions, 64);
    assert!(phases.prepare_ns > 0);
    assert!(phases.advance_ns > 0);
    assert!(phases.forward_ns > 0);
    assert!(phases.barrier_ns > 0);
    assert!(phases.apply_ns >= phases.flush_wait_ns);
    assert!(phases.evaluator_ns > 0);
}

#[test]
fn serial_collection_slice_rejects_phase_accounting() {
    let mut slice = TrainingCollectionSlice::new(config(1, 0, 1), PolicyDevice::Cpu)
        .expect("serial collection slice");
    assert_eq!(
        slice.run_phased(8).expect_err("serial phases").to_string(),
        "invalid PPO config field: collection slice phases"
    );
}

#[test]
fn collection_slice_settings_reject_impossible_group_counts() {
    for (environments, groups) in [(2, 2), (4, 4), (8, 3)] {
        assert_eq!(
            collection_slice_settings(config(environments, 0, groups))
                .expect_err("impossible group count")
                .to_string(),
            if groups == 3 {
                "invalid PPO config field: pipeline groups"
            } else {
                "invalid PPO config field: pipeline groups exceed paired environments"
            },
            "environments={environments} groups={groups}"
        );
    }
}
