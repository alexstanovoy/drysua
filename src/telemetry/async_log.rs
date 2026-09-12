use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const LOG_LINE_CAPACITY: usize = 4096;
const LOG_QUEUE_CAPACITY: usize = 8;
const LOG_DRAIN_LIMIT: Duration = Duration::from_millis(50);
static LOG_PUBLISHER: OnceLock<Option<LogPublisher>> = OnceLock::new();

#[derive(Clone)]
struct LogPublisher {
    sender: mpsc::SyncSender<LogCommand>,
    dropped: Arc<AtomicU64>,
}

// Inline frames bound queue storage without allocating a buffer for each log record.
#[allow(clippy::large_enum_variant)]
enum LogCommand {
    Line(LogLine),
    Barrier(mpsc::SyncSender<bool>),
}

struct LogLine {
    bytes: [u8; LOG_LINE_CAPACITY],
    length: usize,
}

impl Default for LogLine {
    fn default() -> Self {
        Self {
            bytes: [0; LOG_LINE_CAPACITY],
            length: 0,
        }
    }
}

impl Write for LogLine {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert!(self.length <= LOG_LINE_CAPACITY);
        if bytes.len() > LOG_LINE_CAPACITY - self.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "performance log line exceeds 4096 bytes",
            ));
        }
        self.bytes[self.length..self.length + bytes.len()].copy_from_slice(bytes);
        self.length += bytes.len();
        assert!(self.length <= LOG_LINE_CAPACITY);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// One bounded line builder backed by the process-wide, non-blocking diagnostic queue.
pub(crate) struct AsyncLogWriter {
    publisher: Option<LogPublisher>,
    line: LogLine,
}

impl Default for AsyncLogWriter {
    fn default() -> Self {
        let publisher = LOG_PUBLISHER.get_or_init(|| {
            spawn_log_writer(io::stderr())
                .ok()
                .map(|(publisher, _worker)| publisher)
        });
        Self {
            publisher: publisher.clone(),
            line: LogLine::default(),
        }
    }
}

impl Write for AsyncLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let publisher = self.publisher.as_ref().ok_or_else(log_disconnected)?;
        if let Some(body) = bytes.strip_suffix(b"\n") {
            self.line.write_all(body)?;
            writeln!(self.line, " dropped_logs={}", publisher.dropped())?;
            let line = std::mem::take(&mut self.line);
            match publisher.sender.try_send(LogCommand::Line(line)) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(_)) => publisher.note_drop(),
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(log_disconnected()),
            }
        } else {
            self.line.write_all(bytes)?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.publisher
            .as_ref()
            .ok_or_else(log_disconnected)?
            .flush(LOG_DRAIN_LIMIT)
    }
}

impl LogPublisher {
    #[cfg(test)]
    fn writer(&self) -> AsyncLogWriter {
        AsyncLogWriter {
            publisher: Some(self.clone()),
            line: LogLine::default(),
        }
    }

    fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn note_drop(&self) {
        let _previous = self
            .dropped
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                Some(count.saturating_add(1))
            });
    }

    fn flush(&self, timeout: Duration) -> io::Result<()> {
        let (sender, receiver) = mpsc::sync_channel(1);
        match self.sender.try_send(LogCommand::Barrier(sender)) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "performance log queue is full",
                ));
            }
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(log_disconnected()),
        }
        match receiver.recv_timeout(timeout.min(LOG_DRAIN_LIMIT)) {
            Ok(true) => Ok(()),
            Ok(false) => Err(io::Error::other("performance log sink failed")),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "performance log drain deadline exceeded",
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(log_disconnected()),
        }
    }
}

fn log_disconnected() -> io::Error {
    io::Error::new(
        io::ErrorKind::BrokenPipe,
        "performance log worker is unavailable",
    )
}

fn spawn_log_writer(
    writer: impl Write + Send + 'static,
) -> io::Result<(LogPublisher, JoinHandle<()>)> {
    let (sender, receiver) = mpsc::sync_channel(LOG_QUEUE_CAPACITY);
    let publisher = LogPublisher {
        sender,
        dropped: Arc::new(AtomicU64::new(0)),
    };
    let worker = thread::Builder::new()
        .name("drysua-telemetry".to_owned())
        .spawn(move || drain_logs(receiver, writer))?;
    Ok((publisher, worker))
}

fn drain_logs(receiver: mpsc::Receiver<LogCommand>, mut writer: impl Write) {
    // The process-wide event loop waits for bounded work until all publishers disconnect.
    while let Ok(command) = receiver.recv() {
        let result = match command {
            LogCommand::Line(line) => writer.write_all(&line.bytes[..line.length]),
            LogCommand::Barrier(sender) => {
                let result = writer.flush();
                // A timed-out caller may already have dropped its acknowledgement receiver.
                let _delivered = sender.try_send(result.is_ok());
                result
            }
        };
        if result.is_err() {
            return;
        }
    }
}

#[cfg(feature = "builtin")]
pub(crate) struct FlushPerformanceLogs;

#[cfg(feature = "builtin")]
impl Drop for FlushPerformanceLogs {
    fn drop(&mut self) {
        if let Some(Some(publisher)) = LOG_PUBLISHER.get() {
            // A blocked diagnostic consumer must not prevent training shutdown.
            let _drained = publisher.flush(LOG_DRAIN_LIMIT);
        }
    }
}

#[cfg(test)]
#[path = "log_tests.rs"]
mod tests;
