use std::io;
#[cfg(all(unix, target_has_atomic = "8"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
#[path = "shutdown_tests.rs"]
mod tests;

#[cfg(all(unix, target_has_atomic = "8"))]
const SIGNALS: [(libc::c_int, &str); 2] = [(libc::SIGINT, "SIGINT"), (libc::SIGTERM, "SIGTERM")];
#[cfg(all(unix, target_has_atomic = "8"))]
const _: () = assert!(libc::SIGINT != libc::SIGTERM);
#[cfg(all(unix, target_has_atomic = "8"))]
static STATE: SignalState = SignalState::new();

#[cfg(all(unix, target_has_atomic = "8"))]
struct SignalState {
    owned: AtomicBool,
    requested: AtomicBool,
}

#[cfg(all(unix, target_has_atomic = "8"))]
impl SignalState {
    const fn new() -> Self {
        Self {
            owned: AtomicBool::new(false),
            requested: AtomicBool::new(false),
        }
    }
}

/// Owns standalone SIGINT/SIGTERM dispositions until server cleanup completes.
/// Install once during standalone startup, before starting the server; training
/// must not install this guard. Other code must not replace these dispositions
/// during its lifetime. Requests are sticky and do not escalate repeated signals.
#[must_use = "keep the signal guard alive until the metrics server has shut down"]
pub(super) struct ShutdownSignal {
    #[cfg(all(unix, target_has_atomic = "8"))]
    originals: [Option<libc::sigaction>; 2],
}

pub(super) fn install() -> io::Result<ShutdownSignal> {
    #[cfg(all(unix, target_has_atomic = "8"))]
    {
        let action = signal_action()?;
        let originals = install_with(&STATE, &action, exchange_action, log_restore_error)?;
        Ok(ShutdownSignal { originals })
    }
    #[cfg(not(all(unix, target_has_atomic = "8")))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "standalone metrics shutdown requires Unix with lock-free 8-bit atomics",
        ))
    }
}

impl ShutdownSignal {
    pub(super) fn requested(&self) -> bool {
        #[cfg(all(unix, target_has_atomic = "8"))]
        {
            STATE.requested.load(Ordering::Relaxed)
        }
        #[cfg(not(all(unix, target_has_atomic = "8")))]
        {
            false
        }
    }
}

impl Drop for ShutdownSignal {
    fn drop(&mut self) {
        #[cfg(all(unix, target_has_atomic = "8"))]
        restore_with(&STATE, &self.originals, exchange_action, log_restore_error);
    }
}

#[cfg(all(unix, target_has_atomic = "8"))]
extern "C" fn request_shutdown(_signal: libc::c_int) {
    // Rust guarantees AtomicBool is lock-free on targets exposing 8-bit atomics.
    // No payload is published: this handler must only perform the flag store.
    STATE.requested.store(true, Ordering::Relaxed);
}

#[cfg(all(unix, target_has_atomic = "8"))]
fn install_with(
    state: &SignalState,
    action: &libc::sigaction,
    mut exchange: impl FnMut(libc::c_int, &libc::sigaction) -> io::Result<libc::sigaction>,
    mut report: impl FnMut(&'static str, &io::Error),
) -> io::Result<[Option<libc::sigaction>; 2]> {
    state
        .owned
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "metrics shutdown signal ownership is already held or restoration is incomplete",
            )
        })?;
    // Clear only before installing either handler, never after a signal can run.
    state.requested.store(false, Ordering::Relaxed);
    let mut originals = [None; 2];
    for (index, (signal, name)) in SIGNALS.into_iter().enumerate() {
        match exchange(signal, action) {
            Ok(original) => originals[index] = Some(original),
            Err(error) => {
                let restored = restore_with(state, &originals, &mut exchange, &mut report);
                let suffix = if restored {
                    ""
                } else {
                    "; rollback incomplete; shutdown signal ownership retained"
                };
                return Err(io::Error::new(
                    error.kind(),
                    format!("cannot install metrics shutdown handler for {name}: {error}{suffix}"),
                ));
            }
        }
    }
    assert!(originals[0].is_some());
    assert!(originals[1].is_some());
    Ok(originals)
}

#[cfg(all(unix, target_has_atomic = "8"))]
fn restore_with(
    state: &SignalState,
    originals: &[Option<libc::sigaction>; 2],
    mut exchange: impl FnMut(libc::c_int, &libc::sigaction) -> io::Result<libc::sigaction>,
    mut report: impl FnMut(&'static str, &io::Error),
) -> bool {
    let mut restored = true;
    for ((signal, name), original) in SIGNALS.into_iter().zip(originals).rev() {
        if let Some(original) = original
            && let Err(error) = exchange(signal, original)
        {
            restored = false;
            report(name, &error);
        }
    }
    if restored {
        state.owned.store(false, Ordering::Release);
    }
    // Retain ownership after any failure: a later install must not save our own
    // remaining handler as the original disposition and silently lose the real one.
    restored
}

#[cfg(all(unix, target_has_atomic = "8"))]
fn signal_action() -> io::Result<libc::sigaction> {
    // SAFETY: sigaction's integer, pointer, and signal-set fields admit zeroed
    // storage. The mask is initialized with sigemptyset before passing it to OS.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = request_shutdown as *const () as libc::sighandler_t;
    action.sa_flags = libc::SA_RESTART as _;
    // SAFETY: sa_mask is aligned, writable storage for one sigset_t.
    if unsafe { libc::sigemptyset(&mut action.sa_mask) } != 0 {
        let error = io::Error::last_os_error();
        return Err(io::Error::new(
            error.kind(),
            format!("cannot initialize metrics shutdown signal mask: {error}"),
        ));
    }
    Ok(action)
}

#[cfg(all(unix, target_has_atomic = "8"))]
fn exchange_action(signal: libc::c_int, action: &libc::sigaction) -> io::Result<libc::sigaction> {
    assert!(SIGNALS.iter().any(|(supported, _)| *supported == signal));
    let mut original = std::mem::MaybeUninit::<libc::sigaction>::zeroed();
    // SAFETY: Both pointers refer to separate, aligned sigaction storage valid
    // throughout the call. New handlers have the C ABI and static code lifetime;
    // restored actions came from an earlier successful sigaction call.
    if unsafe { libc::sigaction(signal, action, original.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: sigaction initialized the previous disposition; any platform
    // extension fields it did not write retain valid zeroed storage.
    Ok(unsafe { original.assume_init() })
}

#[cfg(all(unix, target_has_atomic = "8"))]
fn log_restore_error(signal: &'static str, error: &io::Error) {
    use std::io::Write;

    // A failed stderr write has no secondary reporting channel. Do not panic in
    // Drop (including during unwinding) or skip the other signal's restoration.
    let _ = writeln!(
        io::stderr().lock(),
        "WARN metrics_shutdown signal={signal}: cannot restore original disposition: {error}; shutdown signal ownership retained"
    );
}
