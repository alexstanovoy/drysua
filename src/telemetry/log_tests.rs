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
fn blocked_sink_has_a_fixed_queue_and_overflow_drops_without_blocking_producer() {
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
    for _ in 0..LOG_QUEUE_CAPACITY {
        writeln!(output, "queued").unwrap();
    }
    writeln!(output, "dropped").unwrap();
    assert_eq!(publisher.dropped(), 1);
    let error = publisher.flush(Duration::ZERO).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(error.to_string(), "performance log queue is full");
    release.send(()).unwrap();
    drop(output);
    drop(publisher);
    worker.join().unwrap();
    assert_eq!(writes.load(Ordering::Relaxed), LOG_QUEUE_CAPACITY + 1);
}

#[test]
fn final_flush_deadline_does_not_wait_for_a_blocked_sink() {
    let (entered, waiting) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let (publisher, worker) = spawn_log_writer(GatedWriter {
        entered,
        release: Some(gate),
        writes: Arc::new(AtomicUsize::new(0)),
    })
    .unwrap();
    let mut output = publisher.writer();
    writeln!(output, "first").unwrap();
    waiting.recv().unwrap();
    let error = publisher.flush(Duration::ZERO).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert_eq!(error.to_string(), "performance log drain deadline exceeded");
    release.send(()).unwrap();
    drop(output);
    drop(publisher);
    worker.join().unwrap();
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

#[test]
fn dropped_log_counter_saturates_at_its_fixed_bound() {
    let (publisher, worker) = spawn_log_writer(io::sink()).unwrap();
    publisher.dropped.store(u64::MAX, Ordering::Relaxed);
    publisher.note_drop();
    assert_eq!(publisher.dropped(), u64::MAX);
    drop(publisher);
    worker.join().unwrap();
}
