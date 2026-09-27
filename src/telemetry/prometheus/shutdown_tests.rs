#[cfg(all(unix, target_has_atomic = "8"))]
mod unix {
    use super::super::*;

    #[test]
    fn ownership_lifecycle_preserves_requests_and_restores_saved_handlers() {
        for during_install in [false, true] {
            let state = SignalState::new();
            state.requested.store(true, Ordering::Relaxed);
            let mut installed = Vec::new();
            let originals = install_with(
                &state,
                &action(0),
                |signal, _| {
                    installed.push(signal);
                    assert!(state.owned.load(Ordering::Acquire));
                    if during_install {
                        state.requested.store(true, Ordering::Relaxed);
                    }
                    Ok(action(signal))
                },
                |_, _| panic!("unexpected rollback"),
            )
            .unwrap();
            assert_eq!(installed, [libc::SIGINT, libc::SIGTERM]);
            assert_eq!(state.requested.load(Ordering::Relaxed), during_install);
            let error = install_with(
                &state,
                &action(0),
                |_, _| panic!("second owner"),
                |_, _| panic!("rollback"),
            );
            let Err(error) = error else {
                panic!("duplicate installation succeeded")
            };
            assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
            assert_eq!(
                error.to_string(),
                "metrics shutdown signal ownership is already held or restoration is incomplete"
            );
            assert!(restore_with(
                &state,
                &originals,
                |_, _| Ok(action(0)),
                |_, _| panic!("restore failed")
            ));
        }
    }

    fn action(marker: libc::c_int) -> libc::sigaction {
        let mut action = signal_action().unwrap();
        action.sa_flags = marker as _;
        action
    }

    #[test]
    fn direct_sigint_and_sigterm_requests_are_sticky_without_installing_handlers() {
        // Only this test accesses STATE. Empty originals keep Drop from touching
        // process dispositions; all other tests use local state and injected I/O.
        let guard = ShutdownSignal {
            originals: [None; 2],
        };
        for signal in [libc::SIGINT, libc::SIGTERM] {
            STATE.requested.store(false, Ordering::Relaxed);
            assert!(!guard.requested());

            request_shutdown(signal);

            assert!(guard.requested());
            request_shutdown(signal);
            assert!(guard.requested());
        }
        drop(guard);
        STATE.requested.store(false, Ordering::Relaxed);
    }

    #[test]
    fn install_failures_restore_only_installed_signals_and_retain_ownership_if_rollback_fails() {
        for (failed_signal, name, rollback_fails) in [
            (libc::SIGINT, "SIGINT", false),
            (libc::SIGTERM, "SIGTERM", false),
            (libc::SIGTERM, "SIGTERM", true),
        ] {
            let state = SignalState::new();
            let mut calls = Vec::new();
            let mut reported = Vec::new();
            let error = match install_with(
                &state,
                &action(0),
                |signal, replacement| {
                    assert!(calls.len() < 3);
                    calls.push((signal, replacement.sa_flags));
                    if calls.len() == 3 && rollback_fails {
                        Err(io::Error::other("restore denied"))
                    } else if signal == failed_signal {
                        Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
                    } else {
                        Ok(action(11))
                    }
                },
                |name, error| reported.push((name, error.to_string())),
            ) {
                Ok(_) => panic!("installation must fail"),
                Err(error) => error,
            };
            let expected = if failed_signal == libc::SIGINT {
                vec![(libc::SIGINT, 0)]
            } else {
                vec![(libc::SIGINT, 0), (libc::SIGTERM, 0), (libc::SIGINT, 11)]
            };
            assert_eq!(calls, expected);
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
            let suffix = if rollback_fails {
                "; rollback incomplete; shutdown signal ownership retained"
            } else {
                ""
            };
            assert_eq!(
                error.to_string(),
                format!("cannot install metrics shutdown handler for {name}: denied{suffix}")
            );
            assert_eq!(
                reported,
                if rollback_fails {
                    vec![("SIGINT", "restore denied".to_owned())]
                } else {
                    vec![]
                }
            );
            assert_eq!(state.owned.load(Ordering::Acquire), rollback_fails);
        }
    }

    #[test]
    fn restoration_replays_both_saved_dispositions_and_releases_ownership_only_on_success() {
        for failures in [[false, false], [true, false], [false, true], [true, true]] {
            let state = SignalState::new();
            state.owned.store(true, Ordering::Release);
            let originals = [Some(action(11)), Some(action(22))];
            let mut calls = Vec::new();
            let mut reported = Vec::new();

            let restored = restore_with(
                &state,
                &originals,
                |signal, original| {
                    assert!(state.owned.load(Ordering::Acquire));
                    assert!(calls.len() < 2);
                    calls.push((signal, original.sa_flags));
                    if failures[usize::from(signal == libc::SIGINT)] {
                        Err(io::Error::other("restore denied"))
                    } else {
                        Ok(action(0))
                    }
                },
                |name, error| reported.push((name, error.to_string())),
            );

            assert_eq!(restored, !failures.contains(&true));
            assert_eq!(calls, [(libc::SIGTERM, 22), (libc::SIGINT, 11)]);
            let expected: Vec<_> = ["SIGTERM", "SIGINT"]
                .into_iter()
                .zip(failures)
                .filter(|(_, failed)| *failed)
                .map(|(name, _)| (name, "restore denied".to_owned()))
                .collect();
            assert_eq!(reported, expected);
            assert_eq!(
                state.owned.load(Ordering::Acquire),
                failures.contains(&true)
            );
        }
    }
}

#[cfg(not(all(unix, target_has_atomic = "8")))]
#[test]
fn unsupported_platform_rejects_standalone_signal_installation() {
    let error = match super::install() {
        Ok(_) => panic!("standalone signal handling must be unsupported"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert_eq!(
        error.to_string(),
        "standalone metrics shutdown requires Unix with lock-free 8-bit atomics"
    );
}
