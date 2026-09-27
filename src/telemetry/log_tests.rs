use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::*;

struct GatedWriter {
    entered: mpsc::SyncSender<()>,
    release: Option<mpsc::Receiver<()>>,
    writes: Arc<AtomicUsize>,
}

impl Write for GatedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(release) = self.release.take() {
            self.entered.send(()).unwrap();
            release.recv().unwrap();
        }
        self.writes.fetch_add(1, Ordering::Relaxed);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn blocked_sink_bounds_queue_loss_and_final_flush_without_blocking_producer() {
    for queued in [0, LOG_QUEUE_CAPACITY] {
        let (entered, waiting) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let writes = Arc::new(AtomicUsize::new(0));
        let (publisher, worker) = spawn_log_writer(GatedWriter {
            entered,
            release: Some(gate),
            writes: writes.clone(),
        })
        .unwrap();
        let mut output = publisher.writer();
        writeln!(output, "first").unwrap();
        waiting.recv().unwrap();
        for _ in 0..queued {
            writeln!(output, "queued").unwrap();
        }
        if queued != 0 {
            writeln!(output, "dropped").unwrap();
        }
        assert_eq!(publisher.dropped(), u64::from(queued != 0));
        let error = publisher.flush(Duration::ZERO).unwrap_err();
        let (kind, message) = if queued == 0 {
            (
                io::ErrorKind::TimedOut,
                "performance log drain deadline exceeded",
            )
        } else {
            (io::ErrorKind::WouldBlock, "performance log queue is full")
        };
        assert_eq!(error.kind(), kind);
        assert_eq!(error.to_string(), message);
        if queued != 0 {
            publisher.dropped.store(u64::MAX, Ordering::Relaxed);
            writeln!(output, "also dropped").unwrap();
            assert_eq!(publisher.dropped(), u64::MAX);
        }
        release.send(()).unwrap();
        drop(output);
        drop(publisher);
        worker.join().unwrap();
        assert_eq!(writes.load(Ordering::Relaxed), queued + 1);
    }
}

#[test]
fn failed_sink_disconnects_the_publisher_without_retrying_gameplay() {
    let (publisher, worker) = spawn_log_writer(io::Cursor::new([0_u8; 0])).unwrap();
    let mut output = publisher.writer();
    writeln!(output, "first record").unwrap();
    worker.join().unwrap();
    let error = writeln!(output, "not retried").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(error.to_string(), "performance log worker is unavailable");
}

#[test]
fn log_line_boundary_preserves_prefix_and_rejects_growth() {
    let mut line = LogLine::default();
    line.write_all(&[b'x'; LOG_LINE_CAPACITY]).unwrap();
    let error = line.write_all(b"y").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "performance log line exceeds 4096 bytes");
    assert_eq!(line.length, LOG_LINE_CAPACITY);
    assert_eq!(line.bytes[LOG_LINE_CAPACITY - 1], b'x');
}
