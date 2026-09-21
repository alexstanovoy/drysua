#[cfg(all(unix, target_has_atomic = "8"))]
mod unix {
    use super::super::*;

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
            assert!(guard.requested());
            request_shutdown(signal);
            assert!(guard.requested());
        }
        drop(guard);
        STATE.requested.store(false, Ordering::Relaxed);
    }

    #[test]
    fn signal_action_uses_the_store_only_handler_and_restarts_interrupted_calls() {
        let replacement = signal_action().unwrap();

        assert_eq!(replacement.sa_flags, action(libc::SA_RESTART).sa_flags);
        assert_eq!(
            replacement.sa_sigaction,
            request_shutdown as *const () as libc::sighandler_t
        );
    }

    #[test]
    fn installation_saves_both_dispositions_and_clears_a_stale_request() {
        let state = SignalState::new();
        state.requested.store(true, Ordering::Relaxed);
        let replacement = action(0);
        let mut calls = Vec::new();

        let originals = install_with(
            &state,
            &replacement,
            |signal, _| {
                assert!(calls.len() < 2);
                calls.push(signal);
                Ok(action(signal))
            },
            |_, _| panic!("successful installation must not report restoration errors"),
        )
        .unwrap();

        assert_eq!(calls, [libc::SIGINT, libc::SIGTERM]);
        assert_eq!(
            originals[0].as_ref().unwrap().sa_flags,
            action(libc::SIGINT).sa_flags
        );
        assert_eq!(
            originals[1].as_ref().unwrap().sa_flags,
            action(libc::SIGTERM).sa_flags
        );
        assert!(state.owned.load(Ordering::Acquire));
        assert!(!state.requested.load(Ordering::Relaxed));
    }

    #[test]
    fn request_received_during_installation_is_not_cleared_afterward() {
        let state = SignalState::new();

        install_with(
            &state,
            &action(0),
            |signal, _| {
                if signal == libc::SIGINT {
                    state.requested.store(true, Ordering::Relaxed);
                }
                Ok(action(signal))
            },
            |_, _| panic!("successful installation must not report restoration errors"),
        )
        .unwrap();

        assert!(state.requested.load(Ordering::Relaxed));
    }

    #[test]
    fn duplicate_installation_preserves_the_request_and_never_exchanges_dispositions() {
        let state = SignalState::new();
        state.owned.store(true, Ordering::Release);
        state.requested.store(true, Ordering::Relaxed);

        let error = match install_with(
            &state,
            &action(0),
            |_, _| panic!("a second owner must not exchange dispositions"),
            |_, _| panic!("a second owner has nothing to restore"),
        ) {
            Ok(_) => panic!("duplicate installation must fail"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            error.to_string(),
            "metrics shutdown signal ownership is already held or restoration is incomplete"
        );
        assert!(state.requested.load(Ordering::Relaxed));
        assert!(state.owned.load(Ordering::Acquire));
    }

    #[test]
    fn first_install_failure_releases_ownership_without_attempting_rollback() {
        let state = SignalState::new();
        let mut calls = 0;

        let error = match install_with(
            &state,
            &action(0),
            |signal, _| {
                calls += 1;
                assert_eq!(signal, libc::SIGINT);
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
            },
            |_, _| panic!("no disposition was installed"),
        ) {
            Ok(_) => panic!("installation must fail"),
            Err(error) => error,
        };

        assert_eq!(calls, 1);
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(
            error.to_string(),
            "cannot install metrics shutdown handler for SIGINT: denied"
        );
        assert!(!state.owned.load(Ordering::Acquire));
    }

    #[test]
    fn second_install_failure_restores_sigint_and_releases_ownership() {
        let state = SignalState::new();
        let mut calls = Vec::new();

        let error = match install_with(
            &state,
            &action(0),
            |signal, replacement| {
                assert!(calls.len() < 3);
                calls.push((signal, replacement.sa_flags));
                if signal == libc::SIGTERM {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
                } else {
                    Ok(action(11))
                }
            },
            |_, _| panic!("rollback must succeed"),
        ) {
            Ok(_) => panic!("installation must fail"),
            Err(error) => error,
        };

        assert_eq!(
            calls,
            [(libc::SIGINT, 0), (libc::SIGTERM, 0), (libc::SIGINT, 11)]
        );
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(
            error.to_string(),
            "cannot install metrics shutdown handler for SIGTERM: denied"
        );
        assert!(!state.owned.load(Ordering::Acquire));
    }

    #[test]
    fn rollback_failure_reports_the_signal_and_retains_ownership() {
        let state = SignalState::new();
        let mut calls = 0;
        let mut reported = Vec::new();

        let error = match install_with(
            &state,
            &action(0),
            |_, _| {
                calls += 1;
                match calls {
                    1 => Ok(action(11)),
                    2 => Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied")),
                    3 => Err(io::Error::other("restore denied")),
                    _ => panic!("rollback must not retry indefinitely"),
                }
            },
            |name, error| reported.push((name, error.to_string())),
        ) {
            Ok(_) => panic!("installation must fail"),
            Err(error) => error,
        };

        assert_eq!(calls, 3);
        assert_eq!(reported, [("SIGINT", "restore denied".to_owned())]);
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(
            error.to_string(),
            "cannot install metrics shutdown handler for SIGTERM: denied; rollback incomplete; shutdown signal ownership retained"
        );
        assert!(state.owned.load(Ordering::Acquire));
    }

    #[test]
    fn restoration_replays_saved_dispositions_in_reverse_order_before_releasing_ownership() {
        let state = SignalState::new();
        state.owned.store(true, Ordering::Release);
        let originals = [Some(action(11)), Some(action(22))];
        let mut calls = Vec::new();

        let restored = restore_with(
            &state,
            &originals,
            |signal, original| {
                assert!(state.owned.load(Ordering::Acquire));
                assert!(calls.len() < 2);
                calls.push((signal, original.sa_flags));
                Ok(action(0))
            },
            |_, _| panic!("restoration must succeed"),
        );

        assert!(restored);
        assert_eq!(calls, [(libc::SIGTERM, 22), (libc::SIGINT, 11)]);
        assert!(!state.owned.load(Ordering::Acquire));
    }

    #[test]
    fn restoration_attempts_both_signals_even_when_either_or_both_fail() {
        for failures in [[true, false], [false, true], [true, true]] {
            let state = SignalState::new();
            state.owned.store(true, Ordering::Release);
            let originals = [Some(action(11)), Some(action(22))];
            let mut calls = Vec::new();
            let mut reported = Vec::new();

            let restored = restore_with(
                &state,
                &originals,
                |signal, _| {
                    assert!(calls.len() < 2);
                    calls.push(signal);
                    if failures[usize::from(signal == libc::SIGINT)] {
                        Err(io::Error::other("restore denied"))
                    } else {
                        Ok(action(0))
                    }
                },
                |name, error| reported.push((name, error.to_string())),
            );

            assert!(!restored);
            assert_eq!(calls, [libc::SIGTERM, libc::SIGINT]);
            let expected: Vec<_> = ["SIGTERM", "SIGINT"]
                .into_iter()
                .zip(failures)
                .filter(|(_, failed)| *failed)
                .map(|(name, _)| (name, "restore denied".to_owned()))
                .collect();
            assert_eq!(reported, expected);
            assert!(state.owned.load(Ordering::Acquire));
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
