use super::*;

#[test]
#[ignore = "read-only accepted U428 journal qualification through the exclusive runner"]
fn accepted_u428_metrics_bind_the_preserved_checkpoint_and_coverage() {
    let checkpoint = std::path::PathBuf::from(
        std::env::var_os("DRYSUA_ACCEPTED_CHECKPOINT").expect("accepted checkpoint"),
    );
    let directory = std::path::PathBuf::from(
        std::env::var_os("DRYSUA_ACCEPTED_METRICS").expect("accepted metrics"),
    );
    let artifact = crate::TrainingArtifact::load(&checkpoint).unwrap();
    let snapshot = read_snapshot(&directory).unwrap();
    assert_eq!(
        std::fs::metadata(directory.join(STATE_FILE)).unwrap().len(),
        858
    );
    let digest: [u8; 32] =
        Sha256::digest(std::fs::read(checkpoint.join("checkpoint.meta")).unwrap()).into();
    assert_eq!(snapshot.checkpoint, digest);
    assert_eq!(
        snapshot.scope,
        crate::checkpoint::metrics_scope_identity(artifact.run(), artifact.config()).unwrap()
    );
    assert_eq!(snapshot.completed_updates, 428);
    assert_eq!(snapshot.start_update, 135);
    assert_eq!(snapshot.samples, 8_613_414);
    assert_eq!(snapshot.optimizer_steps, 17_696);
    assert_eq!(snapshot.games, [3_038, 8_640, 42, 0]);
    assert_eq!(snapshot.last_update_games, [11, 28, 1, 0]);
    assert!(!has_pending(&directory).unwrap());
}
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
fn version_one_wire_contract_preserves_858_bytes_and_canonical_optionals() {
    let mut present = baseline(0);
    present.generation = Some(0);
    present.scale_bp = Some(0);
    present.losses = Some([-0.0; 4]);
    present.durations[0].sum_seconds = -0.0;
    for value in [baseline(0), present, next_snapshot(&baseline(1))] {
        let bytes = encode_snapshot(&value).unwrap();
        assert_eq!(bytes.len(), 858);
        assert_eq!(&bytes[..12], b"DRYMET01\x01\0\0\0");
        let decoded = decode_snapshot(&bytes).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(encode_snapshot(&decoded).unwrap(), bytes);
        if let Some(losses) = decoded.losses {
            assert_ne!(losses[0].to_bits(), (-0.0_f64).to_bits());
        }
    }
}

#[test]
fn checksummed_records_reject_noncanonical_and_invalid_payloads() {
    for (offset, bits, message) in [
        (210, f64::NAN.to_bits(), "metrics losses must be finite"),
        (
            210,
            (-0.0_f64).to_bits(),
            "metrics snapshot floating-point zero is not canonical",
        ),
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
    ] {
        let mut bytes = encode_snapshot(&next_snapshot(&baseline(0))).unwrap();
        bytes[offset..offset + 8].copy_from_slice(&bits.to_le_bytes());
        refresh_checksum(&mut bytes);
        assert_eq!(decode_snapshot(&bytes).unwrap_err().to_string(), message);
    }
    for (offset, byte) in [
        (0, 0),
        (8, 2),
        (196, 2),
        (197, 1),
        (205, 1),
        (209, 2),
        (210, 1),
    ] {
        let mut bytes = encode_snapshot(&baseline(0)).unwrap();
        bytes[offset] = byte;
        refresh_checksum(&mut bytes);
        let message = if offset < 12 {
            "metrics snapshot format or version is unsupported"
        } else {
            "metrics snapshot optional value is not canonical"
        };
        assert_eq!(decode_snapshot(&bytes).unwrap_err().to_string(), message);
    }
}

#[cfg(not(unix))]
#[test]
fn unsupported_platform_never_claims_atomic_replacement() {
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
    type InvalidSnapshot = (fn(&mut TrainingSnapshot), &'static str);
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
    fn pending_recovery_promotes_or_discards_idempotently_across_crash_windows() {
        for rebind in [false, true] {
            for (checkpoint_saved, state_replaced) in [(false, false), (true, false), (true, true)]
            {
                let fixture = Fixture::new();
                let store = fixture.open();
                let base = store.restore(baseline(20), true).unwrap();
                let mut pending = if rebind {
                    base.clone()
                } else {
                    next_snapshot(&base)
                };
                pending.checkpoint = [99; 32];
                store.prepare(&pending).unwrap();
                store.prepare(&pending).unwrap();
                assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
                if state_replaced {
                    fixture.write(STATE_FILE, &pending);
                }
                drop(store);
                assert!(has_pending(&fixture.0).unwrap());
                assert!(!writer_active(&fixture.0).unwrap());
                let expected = if checkpoint_saved { pending } else { base };
                let mut actual = baseline(expected.completed_updates);
                actual.checkpoint = expected.checkpoint;
                for _ in 0..2 {
                    let store = fixture.open();
                    assert_eq!(store.restore(actual.clone(), true).unwrap(), expected);
                    assert_eq!(
                        store
                            .commit(expected.completed_updates, expected.checkpoint)
                            .unwrap(),
                        expected
                    );
                    assert_eq!(read_snapshot(&fixture.0).unwrap(), expected);
                    assert!(!has_pending(&fixture.0).unwrap());
                }
            }
        }
    }

    #[test]
    fn recovery_rejects_wrong_identity_progress_scope_or_gap_without_mutation() {
        let cases: [InvalidSnapshot; 6] = [
            (
                |value| value.scope[0] ^= 1,
                "metrics scope does not match the run",
            ),
            (
                |value| value.completed_updates -= 1,
                "metrics committed state is ahead of the checkpoint",
            ),
            (
                |value| value.checkpoint[0] ^= 1,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                |value| value.samples += 1,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                |value| value.optimizer_steps += 1,
                "metrics committed checkpoint identity or progress does not match",
            ),
            (
                |value| value.completed_updates += 2,
                "metrics checkpoint is neither committed nor pending",
            ),
        ];
        for (change, message) in cases {
            let fixture = Fixture::new();
            let store = fixture.open();
            let base = store.restore(baseline(20), true).unwrap();
            let pending = next_snapshot(&base);
            store.prepare(&pending).unwrap();
            let mut actual = baseline(20);
            change(&mut actual);
            actual.start_update = actual.completed_updates;
            assert_eq!(
                store.restore(actual, true).unwrap_err().to_string(),
                message
            );
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
            assert_eq!(
                read_optional(&fixture.0, PENDING_FILE).unwrap(),
                Some(pending)
            );
        }
    }

    #[test]
    fn publication_failures_preserve_committed_state_and_allow_exact_retry() {
        for temporary in [PENDING_TEMP, STATE_TEMP] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let base = store.restore(baseline(0), false).unwrap();
            let next = next_snapshot(&base);
            if temporary == STATE_TEMP {
                store.prepare(&next).unwrap();
            }
            fs::create_dir(fixture.0.join(temporary)).unwrap();
            let error = if temporary == PENDING_TEMP {
                store.prepare(&next).unwrap_err()
            } else {
                store.commit(1, next.checkpoint).unwrap_err()
            };
            assert_eq!(
                error.to_string(),
                format!("metrics path must be a non-symlink regular file: {temporary}")
            );
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
            assert_eq!(has_pending(&fixture.0).unwrap(), temporary == STATE_TEMP);
            fs::remove_dir(fixture.0.join(temporary)).unwrap();
            store.prepare(&next).unwrap();
            assert_eq!(
                store
                    .prepare(&next_snapshot(&next))
                    .unwrap_err()
                    .to_string(),
                "metrics pending snapshot must be resolved before another prepare"
            );
            assert_eq!(
                store.commit(1, [99; 32]).unwrap_err().to_string(),
                "metrics commit does not match a prepared snapshot"
            );
            assert_eq!(store.commit(1, next.checkpoint).unwrap(), next);
        }
    }

    #[test]
    fn preparation_rejects_mutated_history_but_allows_target_changes() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        let base = next_snapshot(&base);
        store.prepare(&base).unwrap();
        store.commit(1, base.checkpoint).unwrap();
        for index in 0..13 {
            let mut candidate = next_snapshot(&base);
            let message = invalidate_transition(&mut candidate, index);
            assert_eq!(store.prepare(&candidate).unwrap_err().to_string(), message);
            assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
            assert!(!has_pending(&fixture.0).unwrap());
        }
        for target in [200, 50, 1] {
            let mut candidate = base.clone();
            candidate.updates_target = target;
            store.prepare(&candidate).unwrap();
            assert_eq!(store.commit(1, candidate.checkpoint).unwrap(), candidate);
        }
    }

    fn invalidate_transition(value: &mut TrainingSnapshot, index: usize) -> &'static str {
        match index {
            0 => value.samples = 0,
            1 => value.optimizer_steps = 0,
            2 => {
                value.games = [0; 4];
                value.last_update_games = [0; 4];
            }
            3 => value.durations[0] = DurationHistogram::default(),
            4 => value.durations[1].sum_seconds = 0.5,
            5 => value.durations[2].buckets[3] = 0,
            6 => value.generation = Some(0),
            7 => value.scope[0] ^= 1,
            8 => value.start_update = 1,
            9 => value.parallel = 3,
            10 => value.games_per_update = 0,
            11 => value.checkpoint = [1; 32],
            12 => value.updates_target = 1,
            _ => unreachable!(),
        }
        match index {
            0..=6 => "metrics cumulative counters must not regress",
            7 => "metrics scope does not match the run",
            8..=10 => "metrics coverage or configuration changed within the run",
            11 => "metrics advancing update requires a new checkpoint identity",
            _ => "metrics updates must satisfy start <= completed <= target <= 1000000",
        }
    }

    #[test]
    fn same_update_rebind_is_only_allowed_for_empty_coverage() {
        for observed in [false, true] {
            let fixture = Fixture::new();
            let store = fixture.open();
            let base = store.restore(baseline(20), true).unwrap();
            let base = if observed {
                let next = next_snapshot(&base);
                store.prepare(&next).unwrap();
                store.commit(21, next.checkpoint).unwrap()
            } else {
                base
            };
            for mutate_history in [false, true] {
                let mut candidate = base.clone();
                candidate.checkpoint = [99; 32];
                if mutate_history {
                    candidate.samples += 1;
                }
                if observed || mutate_history {
                    assert_eq!(
                        store.prepare(&candidate).unwrap_err().to_string(),
                        "metrics same-update snapshot changes committed metrics"
                    );
                } else {
                    store.prepare(&candidate).unwrap();
                    store.restore(base.clone(), true).unwrap();
                }
                assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
            }
        }
    }

    #[test]
    fn corrupt_or_oversized_journal_records_are_never_repaired_or_published() {
        for name in [STATE_FILE, PENDING_FILE] {
            for (length, message) in [
                (0, "metrics snapshot has invalid length"),
                (857, "metrics snapshot has invalid length"),
                (859, "metrics snapshot has invalid length"),
                (MAX_STATE_BYTES, "metrics snapshot has invalid length"),
                (MAX_STATE_BYTES + 1, "metrics snapshot exceeds 16384 bytes"),
                (858, "metrics snapshot checksum mismatch"),
            ] {
                let fixture = Fixture::new();
                let store = fixture.open();
                let base = store.restore(baseline(0), false).unwrap();
                let mut bytes = encode_snapshot(&base).unwrap();
                bytes[20] ^= 1;
                bytes.resize(length, 0);
                fs::write(fixture.0.join(name), &bytes).unwrap();
                assert_eq!(
                    store.restore(baseline(0), true).unwrap_err().to_string(),
                    message
                );
                assert_eq!(fs::read(fixture.0.join(name)).unwrap(), bytes);
                if name == PENDING_FILE {
                    assert_eq!(has_pending(&fixture.0).unwrap_err().to_string(), message);
                    assert_eq!(read_snapshot(&fixture.0).unwrap(), base);
                }
            }
        }
    }

    #[test]
    fn missing_committed_state_never_falls_back_to_pending_or_fabricates_history() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let next = next_snapshot(&store.restore(baseline(0), false).unwrap());
        store.prepare(&next).unwrap();
        fs::remove_file(fixture.0.join(STATE_FILE)).unwrap();
        assert_eq!(
            store.restore(baseline(1), true).unwrap_err().to_string(),
            "metrics committed state is missing while pending exists"
        );
        for error in [
            read_snapshot(&fixture.0).unwrap_err(),
            store.prepare(&next).unwrap_err(),
            store.commit(1, next.checkpoint).unwrap_err(),
        ] {
            assert_eq!(error.to_string(), "metrics committed state is missing");
        }
        assert!(!fixture.0.join(STATE_FILE).exists());
    }

    #[test]
    fn journal_paths_reject_symlinks_and_directories_without_touching_targets() {
        for name in [
            LOCK_FILE,
            STATE_FILE,
            PENDING_FILE,
            STATE_TEMP,
            PENDING_TEMP,
        ] {
            for link in [false, true] {
                let fixture = Fixture::new();
                let target = fixture.0.join("unrelated");
                fs::write(&target, b"unchanged").unwrap();
                if link {
                    symlink(&target, fixture.0.join(name)).unwrap();
                } else {
                    fs::create_dir(fixture.0.join(name)).unwrap();
                }
                assert_eq!(
                    MetricsStore::open(&fixture.0).err().unwrap().to_string(),
                    format!("metrics path must be a non-symlink regular file: {name}")
                );
                assert_eq!(fs::read(target).unwrap(), b"unchanged");
            }
        }
        let fixture = Fixture::new();
        let alias = fixture.0.join("alias");
        symlink(&fixture.0, &alias).unwrap();
        for path in [
            fixture.0.join("missing"),
            alias.clone(),
            alias.join("."),
            PathBuf::from(format!("{}/", alias.display())),
        ] {
            assert_eq!(
                MetricsStore::open(&path).err().unwrap().to_string(),
                "metrics directory must be an existing non-symlink directory"
            );
        }
        assert!(!fixture.0.join(LOCK_FILE).exists());
    }

    #[test]
    fn live_writer_excludes_cleanup_and_created_files_are_private_and_bounded() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let base = store.restore(baseline(0), false).unwrap();
        for name in [STATE_TEMP, PENDING_TEMP] {
            fs::write(fixture.0.join(name), b"incomplete").unwrap();
        }
        assert!(writer_active(&fixture.0).unwrap());
        assert_eq!(
            MetricsStore::open(&fixture.0).err().unwrap().to_string(),
            "metrics writer is already active"
        );
        for name in [STATE_TEMP, PENDING_TEMP] {
            assert_eq!(fs::read(fixture.0.join(name)).unwrap(), b"incomplete");
        }
        drop(store);
        let store = fixture.open();
        let next = next_snapshot(&base);
        store.prepare(&next).unwrap();
        store.commit(1, next.checkpoint).unwrap();
        let mut names: Vec<_> = fs::read_dir(&fixture.0)
            .unwrap()
            .take(6)
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, [LOCK_FILE, STATE_FILE].map(std::ffi::OsString::from));
        for name in [LOCK_FILE, STATE_FILE] {
            assert_eq!(
                fs::metadata(fixture.0.join(name))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn pending_removed_between_metadata_and_open_is_not_a_read_error() {
        let fixture = Fixture::new();
        fixture.write(PENDING_FILE, &baseline(0));
        let pending = read_optional_with_open(&fixture.0, PENDING_FILE, |path| {
            fs::remove_file(path).unwrap();
            File::open(path)
        })
        .unwrap();
        assert_eq!(pending, None);
        assert!(!has_pending(&fixture.0).unwrap());
    }

    #[test]
    fn first_coverage_and_live_path_revalidation_fail_closed() {
        let fixture = Fixture::new();
        let store = fixture.open();
        assert_eq!(
            store
                .restore(next_snapshot(&baseline(0)), true)
                .unwrap_err()
                .to_string(),
            "metrics first coverage must start at the checkpoint with zero outcomes and durations"
        );
        assert!(!fixture.0.join(STATE_FILE).exists());
        let base = store.restore(baseline(0), false).unwrap();
        assert_eq!(
            store.restore(base.clone(), false).unwrap_err().to_string(),
            "metrics state already exists for a fresh run"
        );
        let target = fixture.0.join("unrelated");
        fs::write(&target, b"unchanged").unwrap();
        fs::remove_file(fixture.0.join(STATE_FILE)).unwrap();
        symlink(&target, fixture.0.join(STATE_FILE)).unwrap();
        for error in [
            read_snapshot(&fixture.0).unwrap_err(),
            store.prepare(&next_snapshot(&base)).unwrap_err(),
        ] {
            assert_eq!(
                error.to_string(),
                "metrics path must be a non-symlink regular file: metrics.state"
            );
        }
        assert_eq!(fs::read(target).unwrap(), b"unchanged");
    }
}
