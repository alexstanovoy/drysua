use super::*;
use std::time::Duration;

fn baseline(update: u64) -> TrainingSnapshot {
    TrainingSnapshot {
        scope: [7; 32],
        checkpoint: [update as u8; 32],
        completed_updates: update,
        updates_target: 100,
        samples: update * 10,
        optimizer_steps: update * 2,
        start_update: update,
        parallel: 2,
        games_per_update: 2,
        heartbeat: 123,
        ..TrainingSnapshot::default()
    }
}

fn next_snapshot(base: &TrainingSnapshot) -> TrainingSnapshot {
    let mut value = base.clone();
    value.completed_updates += 1;
    value.checkpoint = [value.completed_updates as u8; 32];
    value.samples += 10;
    value.optimizer_steps += 2;
    value.games[0] += 1;
    value.last_update_games = [1, 0, 0, 0];
    value.generation = Some(value.completed_updates);
    value.scale_bp = Some(9000);
    value.losses = Some([0.5, -0.5, 0.25, 0.0]);
    for histogram in &mut value.durations {
        histogram.observe(Duration::from_secs(1)).unwrap();
    }
    value.heartbeat += 1;
    value
}

fn refresh_checksum(bytes: &mut [u8]) {
    assert_eq!(bytes.len(), RECORD_BYTES);
    let checksum = Sha256::digest(&bytes[..PAYLOAD_BYTES]);
    bytes[PAYLOAD_BYTES..].copy_from_slice(&checksum);
}

#[test]
fn startup_lock_succeeds_after_transient_contention_with_exact_attempts_and_pauses() {
    for busy_attempts in 0..=3 {
        let mut attempts = 0;
        let mut pauses = 0;
        let result = lock_writer_with_retry(
            || {
                attempts += 1;
                if attempts <= busy_attempts {
                    Err(TryLockError::WouldBlock)
                } else {
                    Ok(())
                }
            },
            |duration| {
                assert_eq!(duration, Duration::from_millis(10));
                pauses += 1;
            },
        );
        result.unwrap();
        assert_eq!(attempts, busy_attempts + 1);
        assert_eq!(pauses, busy_attempts);
    }
}

#[test]
fn startup_lock_busy_writer_fails_after_four_attempts_and_three_bounded_pauses() {
    let mut attempts = 0;
    let mut pauses = 0;
    let mut total_pause = Duration::ZERO;
    let error = lock_writer_with_retry(
        || {
            attempts += 1;
            Err(TryLockError::WouldBlock)
        },
        |duration| {
            assert_eq!(duration, Duration::from_millis(10));
            pauses += 1;
            total_pause += duration;
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(error.to_string(), "metrics writer is already active");
    assert_eq!(attempts, 4);
    assert_eq!(pauses, 3);
    assert_eq!(total_pause, Duration::from_millis(30));
}

#[test]
fn startup_lock_io_error_is_propagated_without_retry_or_pause() {
    let mut attempts = 0;
    let mut pauses = 0;
    let error = lock_writer_with_retry(
        || {
            attempts += 1;
            Err(TryLockError::Error(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "metrics lock I/O failed",
            )))
        },
        |_| pauses += 1,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "metrics lock I/O failed");
    assert_eq!(attempts, 1);
    assert_eq!(pauses, 0);
}

#[test]
fn startup_lock_io_error_after_contention_stops_without_another_pause() {
    let mut attempts = 0;
    let mut pauses = 0;
    let error = lock_writer_with_retry(
        || {
            attempts += 1;
            if attempts == 1 {
                Err(TryLockError::WouldBlock)
            } else {
                Err(TryLockError::Error(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "metrics lock I/O failed after contention",
                )))
            }
        },
        |_| pauses += 1,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(
        error.to_string(),
        "metrics lock I/O failed after contention"
    );
    assert_eq!(attempts, 2);
    assert_eq!(pauses, 1);
}

#[test]
fn binary_encoding_is_fixed_bounded_and_roundtrips_every_field() {
    for value in [baseline(0), next_snapshot(&baseline(1))] {
        let bytes = encode_snapshot(&value).unwrap();
        assert_eq!(bytes.len(), 858);
        assert!(bytes.len() <= MAX_STATE_BYTES);
        assert_eq!(&bytes[..8], b"DRYMET01");
        assert_eq!(&bytes[8..12], &1_u32.to_le_bytes());
        assert_eq!(decode_snapshot(&bytes).unwrap(), value);
        assert_eq!(encode_snapshot(&value).unwrap(), bytes);
    }
}

#[test]
fn binary_encoding_rejects_checksum_corruption() {
    let mut bytes = encode_snapshot(&baseline(0)).unwrap();
    bytes[20] ^= 1;
    assert_eq!(
        decode_snapshot(&bytes).unwrap_err().to_string(),
        "metrics snapshot checksum mismatch"
    );
}

#[test]
fn binary_encoding_rejects_unknown_magic_and_version_even_with_valid_checksum() {
    for offset in [0, 8] {
        let mut bytes = encode_snapshot(&baseline(0)).unwrap();
        bytes[offset] ^= 2;
        refresh_checksum(&mut bytes);
        assert_eq!(
            decode_snapshot(&bytes).unwrap_err().to_string(),
            "metrics snapshot format or version is unsupported"
        );
    }
}

#[test]
fn binary_encoding_rejects_truncated_trailing_and_oversized_records() {
    let bytes = encode_snapshot(&baseline(0)).unwrap();
    for length in 0..RECORD_BYTES {
        assert_eq!(
            decode_snapshot(&bytes[..length]).unwrap_err().to_string(),
            "metrics snapshot has invalid length"
        );
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert_eq!(
        decode_snapshot(&trailing).unwrap_err().to_string(),
        "metrics snapshot has invalid length"
    );
    assert_eq!(
        decode_snapshot(&[0; MAX_STATE_BYTES + 1])
            .unwrap_err()
            .to_string(),
        "metrics snapshot exceeds 16384 bytes"
    );
}

#[test]
fn binary_encoding_rejects_invalid_optional_tags_and_nonzero_absent_payloads() {
    for (offset, byte) in [(196, 2), (197, 1), (205, 1), (209, 2), (210, 1)] {
        let mut bytes = encode_snapshot(&baseline(0)).unwrap();
        bytes[offset] = byte;
        refresh_checksum(&mut bytes);
        assert_eq!(
            decode_snapshot(&bytes).unwrap_err().to_string(),
            "metrics snapshot optional value is not canonical"
        );
    }
}

#[test]
fn binary_encoding_rejects_nan_losses_and_invalid_histograms_after_checksum() {
    let cases = [
        (210, f64::NAN.to_bits(), "metrics losses must be finite"),
        (
            330,
            f64::INFINITY.to_bits(),
            "metrics duration sum must be finite and nonnegative",
        ),
        (
            322,
            u64::MAX,
            "metrics duration count exceeds 9007199254740991",
        ),
        (
            242,
            2,
            "metrics duration buckets must be cumulative and at most count",
        ),
    ];
    for (offset, bits, expected) in cases {
        let mut bytes = encode_snapshot(&next_snapshot(&baseline(0))).unwrap();
        bytes[offset..offset + 8].copy_from_slice(&bits.to_le_bytes());
        refresh_checksum(&mut bytes);
        assert_eq!(decode_snapshot(&bytes).unwrap_err().to_string(), expected);
    }
}

#[test]
fn binary_encoding_normalizes_negative_zero_and_rejects_noncanonical_wire_zero() {
    let mut value = baseline(0);
    value.losses = Some([-0.0; 4]);
    value.durations[0].sum_seconds = -0.0;
    let mut bytes = encode_snapshot(&value).unwrap();
    let decoded = decode_snapshot(&bytes).unwrap();
    assert_eq!(decoded.losses.unwrap()[0].to_bits(), 0);
    assert_eq!(decoded.durations[0].sum_seconds.to_bits(), 0);
    bytes[210..218].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
    refresh_checksum(&mut bytes);
    assert_eq!(
        decode_snapshot(&bytes).unwrap_err().to_string(),
        "metrics snapshot floating-point zero is not canonical"
    );
}

#[test]
fn invalid_snapshot_cannot_be_encoded() {
    let mut value = baseline(0);
    value.parallel = 0;
    assert_eq!(
        encode_snapshot(&value).unwrap_err().to_string(),
        "metrics parallel must be in 1..=40"
    );
}

#[test]
fn optional_present_zero_values_remain_distinct_from_absent_values() {
    let absent = baseline(0);
    let mut present = absent.clone();
    present.generation = Some(0);
    present.scale_bp = Some(0);
    present.losses = Some([0.0; 4]);
    let bytes = encode_snapshot(&present).unwrap();
    assert_ne!(bytes, encode_snapshot(&absent).unwrap());
    assert_eq!(decode_snapshot(&bytes).unwrap(), present);
}

#[cfg(not(unix))]
#[test]
fn writer_reports_unsupported_instead_of_claiming_atomic_replacement() {
    let error = MetricsStore::open(&std::env::temp_dir()).err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    assert_eq!(
        error.to_string(),
        "metrics atomic replacement and directory sync require Unix"
    );
}

#[cfg(unix)]
mod journal {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let index = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "drysua-prometheus-journal-{}-{index}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }

        fn open(&self) -> MetricsStore {
            MetricsStore::open(&self.0).unwrap()
        }

        fn write(&self, name: &str, value: &TrainingSnapshot) {
            fs::write(self.0.join(name), encode_snapshot(value).unwrap()).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn first_coverage_is_persisted_without_inventing_historic_outcomes() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let actual = baseline(20);
        let restored = store.restore(actual.clone(), true).unwrap();
        assert_eq!(restored, actual);
        assert_eq!(restored.start_update, 20);
        assert_eq!(restored.games, [0; 4]);
        assert_eq!(read_snapshot(&fixture.0).unwrap(), restored);
    }

    #[test]
    fn first_coverage_rejects_fabricated_history() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let mut actual = baseline(20);
        actual.start_update = 0;
        assert_eq!(
            store.restore(actual, true).unwrap_err().to_string(),
            "metrics first coverage must start at the checkpoint with zero outcomes and durations"
        );
        assert!(!fixture.0.join(STATE_FILE).exists());
    }

    #[test]
    fn restart_every_update_preserves_cumulative_metrics_and_coverage() {
        let fixture = Fixture::new();
        let mut expected = baseline(5);
        fixture.open().restore(expected.clone(), true).unwrap();
        for update in 6..=9 {
            let store = fixture.open();
            assert_eq!(store.restore(baseline(update - 1), true).unwrap(), expected);
            expected = next_snapshot(&expected);
            store.prepare(&expected).unwrap();
            assert_eq!(store.commit(update, expected.checkpoint).unwrap(), expected);
        }
        assert_eq!(fixture.open().restore(baseline(9), true).unwrap(), expected);
        assert_eq!(expected.games, [4, 0, 0, 0]);
        assert_eq!(expected.durations[0].count, 4);
        assert_eq!(expected.start_update, 5);
    }

    #[test]
    fn exporter_reads_only_committed_state_while_pending_is_speculative() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        store.prepare(&next_snapshot(&base)).unwrap();
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        fs::write(fixture.0.join(PENDING_FILE), b"corrupt pending").unwrap();
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn matching_pending_is_promoted_exactly_once_after_checkpoint_restart() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let pending = next_snapshot(&store.restore(baseline(0), false).unwrap());
        store.prepare(&pending).unwrap();
        drop(store);
        let store = fixture.open();
        assert_eq!(store.restore(baseline(1), true).unwrap(), pending);
        assert!(!fixture.0.join(PENDING_FILE).exists());
        assert_eq!(store.restore(baseline(1), true).unwrap(), pending);
        assert_eq!(store.commit(1, pending.checkpoint).unwrap(), pending);
    }

    #[test]
    fn speculative_pending_is_discarded_only_when_actual_matches_base() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        store.prepare(&next_snapshot(&base)).unwrap();
        drop(store);
        let restored = fixture.open().restore(baseline(0), true).unwrap();
        assert_eq!(restored, base);
        assert!(!fixture.0.join(PENDING_FILE).exists());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn recovery_handles_crash_after_state_replacement_before_pending_removal() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let pending = next_snapshot(&store.restore(baseline(0), false).unwrap());
        store.prepare(&pending).unwrap();
        fixture.write(STATE_FILE, &pending);
        drop(store);
        let restored = fixture.open().restore(baseline(1), true).unwrap();
        assert_eq!(restored, pending);
        assert!(!fixture.0.join(PENDING_FILE).exists());
    }

    #[test]
    fn fresh_run_cannot_reuse_existing_state_even_with_the_same_scope() {
        let fixture = Fixture::new();
        let store = fixture.open();
        store.restore(baseline(0), false).unwrap();
        assert_eq!(
            store.restore(baseline(0), false).unwrap_err().to_string(),
            "metrics state already exists for a fresh run"
        );
    }

    #[test]
    fn restore_rejects_scope_rollback_exact_mismatch_and_gap_without_mutation() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(2), true).unwrap();
        let mut other_scope = baseline(2);
        other_scope.scope[0] ^= 1;
        let mut other_identity = baseline(2);
        other_identity.checkpoint[0] ^= 1;
        let mut other_samples = baseline(2);
        other_samples.samples += 1;
        let mut other_steps = baseline(2);
        other_steps.optimizer_steps += 1;
        for (actual, expected) in [
            (other_scope, "metrics scope does not match the run"),
            (
                baseline(1),
                "metrics committed state is ahead of the checkpoint",
            ),
            (
                other_identity,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                other_samples,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                other_steps,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                baseline(3),
                "metrics checkpoint is neither committed nor pending",
            ),
        ] {
            assert_eq!(
                store.restore(actual, true).unwrap_err().to_string(),
                expected
            );
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        }
    }

    #[test]
    fn pending_gap_and_pending_progress_mismatch_are_not_silently_discarded() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        store.prepare(&next_snapshot(&base)).unwrap();
        let mut wrong_progress = baseline(1);
        wrong_progress.samples += 1;
        for actual in [baseline(2), wrong_progress] {
            assert_eq!(
                store.restore(actual, true).unwrap_err().to_string(),
                "metrics checkpoint is neither committed nor pending"
            );
            assert!(fixture.0.join(PENDING_FILE).exists());
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        }
    }

    #[test]
    fn missing_state_with_pending_never_invents_a_baseline() {
        let fixture = Fixture::new();
        let store = fixture.open();
        fixture.write(PENDING_FILE, &next_snapshot(&baseline(0)));
        assert_eq!(
            store.restore(baseline(1), true).unwrap_err().to_string(),
            "metrics committed state is missing while pending exists"
        );
        assert!(!fixture.0.join(STATE_FILE).exists());
    }

    #[test]
    fn corrupt_state_or_pending_is_an_error_not_a_new_coverage_window() {
        for name in [STATE_FILE, PENDING_FILE] {
            let fixture = Fixture::new();
            let store = fixture.open();
            store.restore(baseline(0), false).unwrap();
            fs::write(fixture.0.join(name), b"broken").unwrap();
            assert_eq!(
                store.restore(baseline(0), true).unwrap_err().to_string(),
                "metrics snapshot has invalid length"
            );
            assert_eq!(fs::read(fixture.0.join(name)).unwrap(), b"broken");
        }
    }

    #[test]
    fn prepare_and_commit_fail_if_committed_state_disappeared_while_active() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let next = next_snapshot(&store.restore(baseline(0), false).unwrap());
        store.prepare(&next).unwrap();
        fs::remove_file(fixture.0.join(STATE_FILE)).unwrap();
        assert_eq!(
            store.prepare(&next).unwrap_err().to_string(),
            "metrics committed state is missing"
        );
        assert_eq!(
            store.commit(1, next.checkpoint).unwrap_err().to_string(),
            "metrics committed state is missing"
        );
    }

    #[test]
    fn same_update_may_increase_target_and_change_heartbeat_only() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let mut value = store.restore(baseline(0), false).unwrap();
        value.updates_target = 200;
        value.heartbeat = 1;
        store.prepare(&value).unwrap();
        assert_eq!(store.commit(0, value.checkpoint).unwrap(), value);
        value.games[0] = 1;
        assert_eq!(
            store.prepare(&value).unwrap_err().to_string(),
            "metrics same-update snapshot changes committed metrics"
        );
    }

    #[test]
    fn resumed_target_can_decrease_without_changing_any_committed_counters() {
        let fixture = Fixture::new();
        let mut base = next_snapshot(&baseline(99));
        base.updates_target = 1000;
        fixture.write(STATE_FILE, &base);
        let store = fixture.open();
        let mut actual = baseline(100);
        actual.updates_target = 200;
        let mut candidate = store.restore(actual.clone(), true).unwrap();
        assert_eq!(candidate, base);
        candidate.updates_target = actual.updates_target;

        store.prepare(&candidate).unwrap();
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        assert_eq!(store.commit(100, candidate.checkpoint).unwrap(), candidate);
        drop(store);
        let mut restored = fixture.open().restore(actual, true).unwrap();
        assert_eq!(restored.updates_target, 200);
        restored.updates_target = base.updates_target;
        assert_eq!(restored, base);
    }

    #[test]
    fn same_update_target_can_equal_completed_but_cannot_go_below_it() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let mut candidate = store.restore(baseline(20), true).unwrap();
        candidate.updates_target = 20;
        store.prepare(&candidate).unwrap();
        assert_eq!(store.commit(20, candidate.checkpoint).unwrap(), candidate);

        candidate.updates_target = 19;
        assert_eq!(
            store.prepare(&candidate).unwrap_err().to_string(),
            "metrics updates must satisfy start <= completed <= target <= 1000000"
        );
        assert!(!has_pending(&fixture.0).unwrap());
        assert_eq!(read_snapshot(&fixture.0).unwrap().updates_target, 20);
    }

    #[test]
    fn advancing_update_can_also_lower_the_target_gauge() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let mut candidate = next_snapshot(&base);
        candidate.updates_target = 50;
        store.prepare(&candidate).unwrap();
        assert_eq!(store.commit(1, candidate.checkpoint).unwrap(), candidate);
    }

    #[test]
    fn first_coverage_checkpoint_rebind_prepares_commits_and_resumes_without_outcomes() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(20), true).unwrap();
        let mut rebound = base.clone();
        rebound.checkpoint = [99; 32];
        rebound.updates_target = 50;
        rebound.heartbeat = 124;

        store.prepare(&rebound).unwrap();
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        assert!(has_pending(&fixture.0).unwrap());
        assert_eq!(store.commit(20, rebound.checkpoint).unwrap(), rebound);
        drop(store);
        let mut actual = baseline(20);
        actual.checkpoint = rebound.checkpoint;
        assert_eq!(fixture.open().restore(actual, true).unwrap(), rebound);
        assert!(!has_pending(&fixture.0).unwrap());
        rebound.checkpoint = base.checkpoint;
        rebound.updates_target = base.updates_target;
        rebound.heartbeat = base.heartbeat;
        assert_eq!(rebound, base);
    }

    #[test]
    fn first_coverage_rebind_pending_recovery_is_idempotent_across_both_crash_windows() {
        for state_replaced in [false, true] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let mut rebound = store.restore(baseline(20), true).unwrap();
            rebound.checkpoint = [99; 32];
            store.prepare(&rebound).unwrap();
            if state_replaced {
                fixture.write(STATE_FILE, &rebound);
            }
            drop(store);
            assert!(has_pending(&fixture.0).unwrap());

            let store = fixture.open();
            let mut actual = baseline(20);
            actual.checkpoint = rebound.checkpoint;
            assert_eq!(store.restore(actual.clone(), true).unwrap(), rebound);
            assert!(!has_pending(&fixture.0).unwrap());
            assert_eq!(store.restore(actual, true).unwrap(), rebound);
            assert_eq!(store.commit(20, rebound.checkpoint).unwrap(), rebound);
        }
    }

    #[test]
    fn first_coverage_rebind_is_discarded_when_actual_checkpoint_still_matches_base() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(20), true).unwrap();
        let mut rebound = base.clone();
        rebound.checkpoint = [99; 32];
        store.prepare(&rebound).unwrap();
        drop(store);

        assert_eq!(fixture.open().restore(base.clone(), true).unwrap(), base);
        assert!(!has_pending(&fixture.0).unwrap());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn first_coverage_rebind_rejects_an_actual_checkpoint_matching_neither_record() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(20), true).unwrap();
        let mut rebound = base.clone();
        rebound.checkpoint = [99; 32];
        store.prepare(&rebound).unwrap();
        let mut actual = base.clone();
        actual.checkpoint = [88; 32];

        assert_eq!(
            store.restore(actual, true).unwrap_err().to_string(),
            "metrics committed checkpoint identity or progress does not match"
        );
        assert!(has_pending(&fixture.0).unwrap());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn first_coverage_rebind_cannot_change_other_metrics_at_the_same_update() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(20), true).unwrap();
        for index in 0..6 {
            let mut candidate = base.clone();
            candidate.checkpoint = [99; 32];
            match index {
                0 => candidate.samples += 1,
                1 => candidate.optimizer_steps += 1,
                2 => {
                    candidate.games[0] = 1;
                    candidate.last_update_games[0] = 1;
                }
                3 => candidate.durations[0].observe(Duration::ZERO).unwrap(),
                4 => candidate.losses = Some([0.0; 4]),
                _ => {
                    candidate.generation = Some(20);
                    candidate.scale_bp = Some(9000);
                }
            }
            assert_eq!(
                store.prepare(&candidate).unwrap_err().to_string(),
                "metrics same-update snapshot changes committed metrics"
            );
            assert!(!has_pending(&fixture.0).unwrap());
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        }
    }

    #[test]
    fn first_coverage_rebind_cannot_change_the_existing_scope() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(20), true).unwrap();
        let mut candidate = base.clone();
        candidate.scope = [8; 32];
        candidate.checkpoint = [99; 32];
        assert_eq!(
            store.prepare(&candidate).unwrap_err().to_string(),
            "metrics scope does not match the run"
        );
        assert_eq!(
            store.restore(candidate, true).unwrap_err().to_string(),
            "metrics scope does not match the run"
        );
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn same_update_rebind_requires_empty_coverage_not_just_unchanged_counters() {
        for index in 0..4 {
            let fixture = Fixture::new();
            let mut base = baseline(20);
            match index {
                0 => base.start_update = 19,
                1 => base.games[0] = 1,
                2 => {
                    base.games[0] = 1;
                    base.last_update_games[0] = 1;
                }
                _ => base.durations[5].observe(Duration::ZERO).unwrap(),
            }
            fixture.write(STATE_FILE, &base);
            let store = fixture.open();
            let mut candidate = base.clone();
            candidate.checkpoint = [99; 32];
            assert_eq!(
                store.prepare(&candidate).unwrap_err().to_string(),
                "metrics same-update snapshot changes committed metrics"
            );
            assert!(!has_pending(&fixture.0).unwrap());
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        }
    }

    #[test]
    fn observed_coverage_same_update_checkpoint_conflicts_fail_prepare_and_recovery() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let mut value = next_snapshot(&base);
        store.prepare(&value).unwrap();
        store.commit(1, value.checkpoint).unwrap();
        value.checkpoint = [99; 32];
        assert_eq!(
            store.prepare(&value).unwrap_err().to_string(),
            "metrics same-update snapshot changes committed metrics"
        );
        fixture.write(PENDING_FILE, &value);
        assert_eq!(
            store.restore(value, true).unwrap_err().to_string(),
            "metrics same-update snapshot changes committed metrics"
        );
    }

    #[test]
    fn prepare_rejects_regressions_in_every_cumulative_counter() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = next_snapshot(&baseline(0));
        fixture.write(STATE_FILE, &base);
        for index in 0..7 {
            let mut value = next_snapshot(&base);
            match index {
                0 => value.completed_updates = 0,
                1 => value.samples = 0,
                2 => value.optimizer_steps = 0,
                3 => {
                    value.games[0] = 0;
                    value.last_update_games = [0; 4];
                }
                4 => value.durations[0] = DurationHistogram::default(),
                5 => value.generation = Some(0),
                _ => {
                    value.generation = None;
                    value.scale_bp = None;
                }
            }
            assert_eq!(
                store.prepare(&value).unwrap_err().to_string(),
                "metrics cumulative counters must not regress"
            );
        }
        assert!(!fixture.0.join(PENDING_FILE).exists());
    }

    #[test]
    fn prepare_rejects_independent_bucket_and_sum_regressions() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = next_snapshot(&baseline(0));
        fixture.write(STATE_FILE, &base);
        for index in 0..6 {
            let mut value = next_snapshot(&base);
            value.durations[index].sum_seconds = 0.5;
            assert_eq!(
                store.prepare(&value).unwrap_err().to_string(),
                "metrics cumulative counters must not regress"
            );
            value = next_snapshot(&base);
            value.durations[index].buckets[3] = 0;
            assert_eq!(
                store.prepare(&value).unwrap_err().to_string(),
                "metrics cumulative counters must not regress"
            );
        }
    }

    #[test]
    fn prepare_cannot_replace_unresolved_pending_but_exact_retry_is_allowed() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let next = next_snapshot(&store.restore(baseline(0), false).unwrap());
        store.prepare(&next).unwrap();
        store.prepare(&next).unwrap();
        assert_eq!(
            store
                .prepare(&next_snapshot(&next))
                .unwrap_err()
                .to_string(),
            "metrics pending snapshot must be resolved before another prepare"
        );
        assert_eq!(store.commit(1, next.checkpoint).unwrap(), next);
    }

    #[test]
    fn prepare_rejects_scope_coverage_and_configuration_changes() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(2), true).unwrap();
        let mut wrong_scope = next_snapshot(&base);
        wrong_scope.scope[0] ^= 1;
        assert_eq!(
            store.prepare(&wrong_scope).unwrap_err().to_string(),
            "metrics scope does not match the run"
        );
        for index in 0..3 {
            let mut candidate = next_snapshot(&base);
            match index {
                0 => candidate.start_update = 0,
                1 => candidate.parallel = 3,
                _ => candidate.games_per_update = 0,
            }
            assert_eq!(
                store.prepare(&candidate).unwrap_err().to_string(),
                "metrics coverage or configuration changed within the run"
            );
        }
    }

    #[test]
    fn advancing_update_cannot_reuse_the_old_checkpoint_identity() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let mut candidate = next_snapshot(&base);
        candidate.checkpoint = base.checkpoint;
        assert_eq!(
            store.prepare(&candidate).unwrap_err().to_string(),
            "metrics advancing update requires a new checkpoint identity"
        );
    }

    #[test]
    fn restored_pending_must_share_scope_even_when_actual_checkpoint_matches() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let mut pending = next_snapshot(&base);
        pending.scope = [99; 32];
        fixture.write(PENDING_FILE, &pending);
        assert_eq!(
            store.restore(baseline(1), true).unwrap_err().to_string(),
            "metrics scope does not match the run"
        );
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn commit_requires_exact_prepared_identity_and_never_claims_wrong_success() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let next = next_snapshot(&base);
        store.prepare(&next).unwrap();
        for (update, checkpoint) in [(2, next.checkpoint), (1, [99; 32])] {
            assert_eq!(
                store.commit(update, checkpoint).unwrap_err().to_string(),
                "metrics commit does not match a prepared snapshot"
            );
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        }
        store.commit(1, next.checkpoint).unwrap();
        assert_eq!(
            store.commit(2, next.checkpoint).unwrap_err().to_string(),
            "metrics commit does not match a prepared snapshot"
        );
    }

    #[test]
    fn replacement_io_failure_preserves_committed_state_and_pending_for_retry() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let next = next_snapshot(&base);
        store.prepare(&next).unwrap();
        fs::create_dir(fixture.0.join(STATE_TEMP)).unwrap();
        let error = store.commit(1, next.checkpoint).unwrap_err();
        assert_eq!(
            error.to_string(),
            "metrics path must be a non-symlink regular file: .metrics.state.tmp"
        );
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        assert!(fixture.0.join(PENDING_FILE).exists());
        fs::remove_dir(fixture.0.join(STATE_TEMP)).unwrap();
        assert_eq!(store.commit(1, next.checkpoint).unwrap(), next);
    }

    #[test]
    fn prepare_io_failure_does_not_create_pending_or_publish_candidate() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        fs::create_dir(fixture.0.join(PENDING_TEMP)).unwrap();
        assert_eq!(
            store
                .prepare(&next_snapshot(&base))
                .unwrap_err()
                .to_string(),
            "metrics path must be a non-symlink regular file: .metrics.pending.tmp"
        );
        assert!(!fixture.0.join(PENDING_FILE).exists());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
    }

    #[test]
    fn stale_crash_temps_are_removed_only_after_exclusive_lock_acquisition() {
        let fixture = Fixture::new();
        let store = fixture.open();
        for name in [STATE_TEMP, PENDING_TEMP] {
            fs::write(fixture.0.join(name), b"incomplete").unwrap();
        }
        assert_eq!(
            MetricsStore::open(&fixture.0).err().unwrap().to_string(),
            "metrics writer is already active"
        );
        assert!(fixture.0.join(STATE_TEMP).exists());
        assert!(fixture.0.join(PENDING_TEMP).exists());
        drop(store);
        let _store = fixture.open();
        assert!(!fixture.0.join(STATE_TEMP).exists());
        assert!(!fixture.0.join(PENDING_TEMP).exists());
    }

    #[test]
    fn writer_liveness_is_the_os_lock_and_missing_lock_is_not_created() {
        let fixture = Fixture::new();
        assert_eq!(
            writer_active(&fixture.0).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert!(!fixture.0.join(LOCK_FILE).exists());
        let store = fixture.open();
        assert!(writer_active(&fixture.0).unwrap());
        drop(store);
        assert!(!writer_active(&fixture.0).unwrap());
        let _store = fixture.open();
        assert!(writer_active(&fixture.0).unwrap());
    }

    #[test]
    fn open_requires_an_existing_real_directory() {
        let fixture = Fixture::new();
        let missing = fixture.0.join("missing");
        assert_eq!(
            MetricsStore::open(&missing).err().unwrap().to_string(),
            "metrics directory must be an existing non-symlink directory"
        );
        let alias = fixture.0.join("alias");
        symlink(&fixture.0, &alias).unwrap();
        assert_eq!(
            MetricsStore::open(&alias).err().unwrap().to_string(),
            "metrics directory must be an existing non-symlink directory"
        );
    }

    #[test]
    fn directory_symlink_cannot_be_hidden_by_a_trailing_separator_or_dot() {
        let fixture = Fixture::new();
        let alias = fixture.0.join("alias");
        symlink(&fixture.0, &alias).unwrap();
        for path in [
            PathBuf::from(format!("{}/", alias.display())),
            alias.join("."),
        ] {
            assert_eq!(
                MetricsStore::open(&path).err().unwrap().to_string(),
                "metrics directory must be an existing non-symlink directory"
            );
        }
        assert!(!fixture.0.join(LOCK_FILE).exists());
    }

    #[test]
    fn all_journal_paths_reject_directories_instead_of_regular_files() {
        for name in [
            LOCK_FILE,
            STATE_FILE,
            PENDING_FILE,
            STATE_TEMP,
            PENDING_TEMP,
        ] {
            let fixture = Fixture::new();
            fs::create_dir(fixture.0.join(name)).unwrap();
            assert_eq!(
                MetricsStore::open(&fixture.0).err().unwrap().to_string(),
                format!("metrics path must be a non-symlink regular file: {name}")
            );
        }
    }

    #[test]
    fn exporter_does_not_fall_back_to_pending_when_committed_state_is_missing() {
        let fixture = Fixture::new();
        fixture.write(PENDING_FILE, &next_snapshot(&baseline(0)));
        let error = read_snapshot(&fixture.0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(error.to_string(), "metrics committed state is missing");
    }

    #[test]
    fn pending_removed_between_metadata_and_open_is_absent_without_a_read_error() {
        let fixture = Fixture::new();
        let committed = baseline(0);
        fixture.write(STATE_FILE, &committed);
        fixture.write(PENDING_FILE, &next_snapshot(&committed));
        let mut attempts = 0;

        let pending = read_optional_with_open(&fixture.0, PENDING_FILE, |path| {
            attempts += 1;
            fs::remove_file(path).unwrap();
            File::open(path)
        })
        .expect("concurrent pending removal is not an I/O failure");

        assert_eq!(pending, None);
        assert_eq!(attempts, 1);
        assert_eq!(read_snapshot(&fixture.0).unwrap(), committed);
    }

    #[test]
    fn optional_open_not_found_retries_can_succeed_on_each_bounded_attempt() {
        for missing_attempts in 0..=2 {
            let fixture = Fixture::new();
            let expected = baseline(0);
            fixture.write(PENDING_FILE, &expected);
            let mut attempts = 0;

            let snapshot = read_optional_with_open(&fixture.0, PENDING_FILE, |path| {
                attempts += 1;
                if attempts <= missing_attempts {
                    return Err(io::Error::new(io::ErrorKind::NotFound, "open race"));
                }
                File::open(path)
            })
            .expect("bounded open retry");

            assert_eq!(snapshot, Some(expected));
            assert_eq!(attempts, missing_attempts + 1);
        }
    }

    #[test]
    fn optional_open_not_found_races_fail_after_exactly_three_attempts() {
        let fixture = Fixture::new();
        fixture.write(PENDING_FILE, &baseline(0));
        let mut attempts = 0;

        let error = read_optional_with_open(&fixture.0, PENDING_FILE, |_| {
            attempts += 1;
            Err(io::Error::new(io::ErrorKind::NotFound, "open race"))
        })
        .unwrap_err();

        assert_eq!(attempts, 3);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "metrics file changed while opening");
    }

    #[test]
    fn optional_open_other_errors_propagate_without_retry() {
        for kind in [
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::Interrupted,
            io::ErrorKind::Other,
        ] {
            let fixture = Fixture::new();
            fixture.write(PENDING_FILE, &baseline(0));
            let mut attempts = 0;

            let error = read_optional_with_open(&fixture.0, PENDING_FILE, |_| {
                attempts += 1;
                Err(io::Error::new(kind, "metrics snapshot open failed"))
            })
            .unwrap_err();

            assert_eq!(attempts, 1);
            assert_eq!(error.kind(), kind);
            assert_eq!(error.to_string(), "metrics snapshot open failed");
        }
    }

    #[test]
    fn pending_indicator_is_false_when_absent_and_creates_no_files() {
        let fixture = Fixture::new();
        assert!(!has_pending(&fixture.0).unwrap());
        assert!(fs::read_dir(&fixture.0).unwrap().next().is_none());
    }

    #[test]
    fn pending_indicator_reports_active_and_abandoned_transactions_without_publishing() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        store.prepare(&next_snapshot(&base)).unwrap();
        assert!(has_pending(&fixture.0).unwrap());
        assert!(writer_active(&fixture.0).unwrap());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        drop(store);

        assert!(has_pending(&fixture.0).unwrap());
        assert!(!writer_active(&fixture.0).unwrap());
        assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
        assert_eq!(fixture.open().restore(base.clone(), true).unwrap(), base);
        assert!(!has_pending(&fixture.0).unwrap());
    }

    #[test]
    fn pending_indicator_validates_records_even_without_committed_state() {
        let fixture = Fixture::new();
        fixture.write(PENDING_FILE, &next_snapshot(&baseline(0)));
        assert!(has_pending(&fixture.0).unwrap());
        assert!(!fixture.0.join(STATE_FILE).exists());
        assert!(!fixture.0.join(LOCK_FILE).exists());
    }

    #[test]
    fn pending_indicator_rejects_checksum_corruption_and_preserves_the_record() {
        let fixture = Fixture::new();
        let mut bytes = encode_snapshot(&baseline(0)).unwrap();
        bytes[20] ^= 1;
        fs::write(fixture.0.join(PENDING_FILE), &bytes).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot checksum mismatch"
        );
        assert_eq!(fs::read(fixture.0.join(PENDING_FILE)).unwrap(), bytes);
    }

    #[test]
    fn pending_indicator_rejects_invalid_metrics_even_with_a_valid_checksum() {
        let fixture = Fixture::new();
        let mut bytes = encode_snapshot(&baseline(0)).unwrap();
        bytes[180..188].copy_from_slice(&0_u64.to_le_bytes());
        refresh_checksum(&mut bytes);
        fs::write(fixture.0.join(PENDING_FILE), &bytes).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics parallel must be in 1..=40"
        );
        assert_eq!(fs::read(fixture.0.join(PENDING_FILE)).unwrap(), bytes);
    }

    #[test]
    fn pending_indicator_rejects_truncation_trailing_bytes_and_oversized_files() {
        let fixture = Fixture::new();
        let bytes = encode_snapshot(&baseline(0)).unwrap();
        fs::write(fixture.0.join(PENDING_FILE), &bytes[..RECORD_BYTES - 1]).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot has invalid length"
        );
        let mut trailing = bytes;
        trailing.push(0);
        fs::write(fixture.0.join(PENDING_FILE), trailing).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot has invalid length"
        );
        fs::write(fixture.0.join(PENDING_FILE), [0; MAX_STATE_BYTES + 1]).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot exceeds 16384 bytes"
        );
    }

    #[test]
    fn pending_indicator_rejects_symlinks_without_reading_or_changing_the_target() {
        let fixture = Fixture::new();
        let target = fixture.0.join("unrelated");
        fs::write(&target, b"unchanged").unwrap();
        symlink(&target, fixture.0.join(PENDING_FILE)).unwrap();
        assert_eq!(
            has_pending(&fixture.0).unwrap_err().to_string(),
            "metrics path must be a non-symlink regular file: metrics.pending"
        );
        assert_eq!(fs::read(target).unwrap(), b"unchanged");
    }

    #[test]
    fn all_journal_paths_reject_symlinks_without_touching_the_target() {
        for name in [
            LOCK_FILE,
            STATE_FILE,
            PENDING_FILE,
            STATE_TEMP,
            PENDING_TEMP,
        ] {
            let fixture = Fixture::new();
            let target = fixture.0.join("unrelated");
            fs::write(&target, b"unchanged").unwrap();
            symlink(&target, fixture.0.join(name)).unwrap();
            assert_eq!(
                MetricsStore::open(&fixture.0).err().unwrap().to_string(),
                format!("metrics path must be a non-symlink regular file: {name}")
            );
            assert_eq!(fs::read(target).unwrap(), b"unchanged");
        }
    }

    #[test]
    fn reads_and_replacements_reject_paths_changed_to_symlinks_after_open() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let target = fixture.0.join("unrelated");
        fs::write(&target, encode_snapshot(&base).unwrap()).unwrap();
        fs::remove_file(fixture.0.join(STATE_FILE)).unwrap();
        symlink(&target, fixture.0.join(STATE_FILE)).unwrap();
        assert_eq!(
            read_snapshot(&fixture.0).unwrap_err().to_string(),
            "metrics path must be a non-symlink regular file: metrics.state"
        );
        assert_eq!(
            store
                .prepare(&next_snapshot(&base))
                .unwrap_err()
                .to_string(),
            "metrics path must be a non-symlink regular file: metrics.state"
        );
    }

    #[test]
    fn read_size_limit_is_checked_before_decoding_and_never_truncates_successfully() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join(STATE_FILE), [0; MAX_STATE_BYTES + 1]).unwrap();
        assert_eq!(
            read_snapshot(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot exceeds 16384 bytes"
        );
        fs::write(fixture.0.join(STATE_FILE), [0; MAX_STATE_BYTES]).unwrap();
        assert_eq!(
            read_snapshot(&fixture.0).unwrap_err().to_string(),
            "metrics snapshot has invalid length"
        );
    }

    #[test]
    fn journal_uses_only_fixed_files_with_private_creation_permissions() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let mut value = store.restore(baseline(0), false).unwrap();
        for _ in 0..3 {
            value = next_snapshot(&value);
            store.prepare(&value).unwrap();
            store
                .commit(value.completed_updates, value.checkpoint)
                .unwrap();
        }
        let mut names: Vec<_> = fs::read_dir(&fixture.0)
            .unwrap()
            .take(6)
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, [LOCK_FILE, STATE_FILE].map(std::ffi::OsString::from));
        for name in [LOCK_FILE, STATE_FILE] {
            let metadata = fs::metadata(fixture.0.join(name)).unwrap();
            assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        }
    }
}
