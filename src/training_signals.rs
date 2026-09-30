//! Graceful stop requests for one long-lived training process.
//!
//! The first SIGINT or SIGTERM only raises a flag: the trainer finishes and
//! commits the in-flight update, then returns normally. Each handler is
//! one-shot, so a repeated signal falls back to the default action and
//! terminates at once; the committed checkpoint stays valid either way.

use std::sync::atomic::{AtomicBool, Ordering};

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Installs the one-shot stop handlers for SIGINT and SIGTERM.
#[cfg(unix)]
pub(crate) fn install() -> std::io::Result<()> {
    for signal in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: `sigaction` is plain data; zero is a valid empty value.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = request_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
        action.sa_flags = libc::SA_RESETHAND | libc::SA_RESTART;
        // SAFETY: Both pointers are valid for the duration of each call, and the
        // handler only performs an async-signal-safe atomic store.
        let installed = unsafe {
            libc::sigemptyset(&mut action.sa_mask) == 0
                && libc::sigaction(signal, &action, std::ptr::null_mut()) == 0
        };
        if !installed {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Without POSIX signals the default console handling terminates at once.
#[cfg(not(unix))]
pub(crate) fn install() -> std::io::Result<()> {
    Ok(())
}

/// Whether a stop was requested; the trainer checks it after every update.
pub(crate) fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

#[cfg(unix)]
extern "C" fn request_stop(_: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}
