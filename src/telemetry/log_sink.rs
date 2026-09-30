//! The process's one diagnostic stream: lines are buffered and reach stderr at
//! most every [`FLUSH_INTERVAL`] (and on an explicit flush at stop or exit), so
//! a long training run does not turn every log line into a write.

#[cfg(feature = "builtin")]
use std::fmt;
use std::io::{self, BufWriter, Write};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

const FLUSH_INTERVAL: Duration = Duration::from_secs(2);
const BUFFER_BYTES: usize = 256 * 1024;

struct Sink {
    writer: BufWriter<io::Stderr>,
    dirty: bool,
}

static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();

fn sink() -> MutexGuard<'static, Sink> {
    let sink = SINK.get_or_init(|| {
        // Without the flusher a quiet phase would hold the last lines back indefinitely.
        let _detached = std::thread::Builder::new()
            .name("drysua-log-flush".to_owned())
            .spawn(|| {
                loop {
                    std::thread::sleep(FLUSH_INTERVAL);
                    flush_log();
                }
            });
        Mutex::new(Sink {
            writer: BufWriter::with_capacity(BUFFER_BYTES, io::stderr()),
            dirty: false,
        })
    });
    sink.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Appends one line; a diagnostic write never fails the caller.
#[cfg(feature = "builtin")]
pub(crate) fn write_log_line(line: fmt::Arguments<'_>) {
    let mut sink = sink();
    let _ignored = writeln!(sink.writer, "{line}");
    sink.dirty = true;
}

/// Pushes buffered lines to stderr now; called by the flusher, on stop and at exit.
pub(crate) fn flush_log() {
    let mut sink = sink();
    if sink.dirty {
        let _ignored = sink.writer.flush();
        sink.dirty = false;
    }
}

/// Flushes the sink when dropped, so every exit path of a command drains it.
#[cfg(feature = "builtin")]
pub(crate) struct FlushLogOnDrop;

#[cfg(feature = "builtin")]
impl Drop for FlushLogOnDrop {
    fn drop(&mut self) {
        flush_log();
    }
}

/// A [`Write`] view of the sink for writers that produce whole lines.
pub(crate) struct LogSinkWriter;

impl Write for LogSinkWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut sink = sink();
        sink.writer.write_all(bytes)?;
        sink.dirty = true;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        flush_log();
        Ok(())
    }
}

/// Formats one line into the buffered diagnostic stream.
#[cfg(feature = "builtin")]
macro_rules! log_line {
    ($($argument:tt)*) => {
        $crate::telemetry::write_log_line(format_args!($($argument)*))
    };
}
#[cfg(feature = "builtin")]
pub(crate) use log_line;
