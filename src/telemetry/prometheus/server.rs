use super::resources::ResourceCollector;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;

const BODY_LIMIT: usize = RESPONSE_LIMIT - 256;
const CLIENT_LIMIT: usize = 8;
const COLLECTION_INTERVAL: Duration = Duration::from_secs(5);
const HEADER_LIMIT: usize = 64;
const IO_DEADLINE: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const REQUEST_LIMIT: usize = 4096;
const RESPONSE_LIMIT: usize = 128 * 1024;
const _: () = assert!(CLIENT_LIMIT <= 8);
const _: () = assert!(BODY_LIMIT + 256 == RESPONSE_LIMIT);

/// Loopback-only exporter. The callback must be bounded, nonblocking, and return
/// at most a bounded snapshot; std cannot interrupt a caller-supplied closure.
/// The callback must not declare the resource metric families owned by resources.
/// Shutdown wakes the worker immediately, closes clients, and reaps its child.
/// Kernel stalls in process creation/reaping or procfs reads cannot be time-bounded
/// by std; socket progress and child polling have explicit absolute deadlines.
pub(crate) struct MetricsServer {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<io::Result<()>>>,
    #[cfg(test)]
    local_addr: SocketAddr,
}

impl MetricsServer {
    /// Serves on the calling thread until the bounded, nonblocking predicate
    /// requests shutdown. The predicate is reread between work and after each
    /// 20 ms idle park; it need not wake the thread itself. OS scheduling and
    /// blocking callbacks retain the shutdown caveats documented on this type.
    pub(crate) fn run(
        listen: SocketAddr,
        render: impl FnMut() -> io::Result<String>,
        requested: impl Fn() -> bool,
    ) -> io::Result<()> {
        let listener = bind_listener(listen)?;
        serve(
            listener,
            Arc::new(AtomicBool::new(false)),
            render,
            requested,
        )
    }

    #[cfg(any(feature = "builtin", test))]
    pub(crate) fn start(
        listen: SocketAddr,
        render: impl FnMut() -> io::Result<String> + Send + 'static,
    ) -> io::Result<Self> {
        let listener = bind_listener(listen)?;
        #[cfg(test)]
        let local_addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("drysua-metrics".into())
            .spawn(move || {
                finish_worker(
                    serve(listener, worker_stop, render, || false),
                    super::record_server_failure,
                )
            })?;
        Ok(Self {
            stop,
            worker: Some(worker),
            #[cfg(test)]
            local_addr,
        })
    }

    #[cfg(test)]
    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    #[cfg(any(feature = "builtin", test))]
    pub(crate) fn shutdown(mut self) -> io::Result<()> {
        self.stop_and_join()
    }

    fn stop_and_join(&mut self) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        worker.thread().unpark();
        worker
            .join()
            .map_err(|_| io::Error::other("metrics server worker panicked"))?
    }
}

impl Drop for MetricsServer {
    fn drop(&mut self) {
        // Explicit shutdown reports errors; destructors must not double-panic.
        let _ = self.stop_and_join();
    }
}

#[cfg(any(feature = "builtin", test))]
fn finish_worker(result: io::Result<()>, report: impl FnOnce(&io::Error)) -> io::Result<()> {
    if let Err(error) = &result {
        report(error);
    }
    result
}

fn serve(
    listener: TcpListener,
    stop: Arc<AtomicBool>,
    mut render: impl FnMut() -> io::Result<String>,
    requested: impl Fn() -> bool,
) -> io::Result<()> {
    let should_stop = || stop.load(Ordering::Acquire) || requested();
    if should_stop() {
        return Ok(());
    }
    let mut resources = ResourceCollector::new();
    let mut schedule = CollectionSchedule::new(Instant::now());
    let mut pending = None;
    let mut cached = response(Status::ServiceUnavailable, "metrics not available\n");
    let mut clients: [Option<Client<TcpStream>>; CLIENT_LIMIT] = std::array::from_fn(|_| None);
    // This is the sole event loop; every tick services a fixed number of clients.
    while !should_stop() {
        let now = Instant::now();
        if pending.is_none() && schedule.take_due(now) {
            pending = Some(render());
            if should_stop() {
                break;
            }
            resources.collect(Instant::now());
        }
        if should_stop() {
            break;
        }
        if pending.is_some() && resources.poll(Instant::now()) {
            let rendered = pending.take().expect("pending collection was checked");
            cached = make_snapshot(rendered, |output| resources.append_exposition(output));
            schedule.finish_collection(Instant::now());
        }
        if should_stop() {
            break;
        }
        for slot in &mut clients {
            if slot.is_none() {
                *slot = accept_client(&listener, Instant::now())?;
            }
            if let Some(client) = slot
                && !client.advance(Instant::now(), &cached)
            {
                *slot = None;
            }
        }
        thread::park_timeout(POLL_INTERVAL);
    }
    drop(clients);
    resources.shutdown()
}

fn bind_listener(listen: SocketAddr) -> io::Result<TcpListener> {
    if !listen.ip().is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "metrics listener must bind a loopback address",
        ));
    }
    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;
    let local_addr = listener.local_addr()?;
    assert!(local_addr.ip().is_loopback());
    assert_ne!(local_addr.port(), 0);
    Ok(listener)
}

fn accept_client(listener: &TcpListener, now: Instant) -> io::Result<Option<Client<TcpStream>>> {
    match listener.accept() {
        Ok((stream, peer)) => {
            if !peer.ip().is_loopback() {
                return Ok(None);
            }
            // A per-client setup failure must not stop metrics for other clients.
            if stream.set_nonblocking(true).is_err() {
                return Ok(None);
            }
            Ok(Some(Client::new(stream, now)))
        }
        Err(error) if transient(&error) || error.kind() == io::ErrorKind::ConnectionAborted => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

struct CollectionSchedule {
    next: Instant,
}

impl CollectionSchedule {
    fn new(now: Instant) -> Self {
        Self { next: now }
    }

    fn take_due(&mut self, now: Instant) -> bool {
        if now < self.next {
            return false;
        }
        // Never catch up missed intervals with a burst of resource probes.
        self.next = now + COLLECTION_INTERVAL;
        true
    }

    fn finish_collection(&mut self, now: Instant) {
        // Include callback and resource-read time so variable work cannot shorten
        // the gap between consecutive resource collections.
        self.next = now + COLLECTION_INTERVAL;
    }
}

struct Client<Stream> {
    stream: Stream,
    request: [u8; REQUEST_LIMIT + 1],
    received: usize,
    response: Option<Arc<[u8]>>,
    written: usize,
    deadline: Instant,
}

impl<Stream: Read + Write> Client<Stream> {
    fn new(stream: Stream, now: Instant) -> Self {
        Self {
            stream,
            request: [0; REQUEST_LIMIT + 1],
            received: 0,
            response: None,
            written: 0,
            deadline: now + IO_DEADLINE,
        }
    }

    fn advance(&mut self, now: Instant, cached: &Arc<[u8]>) -> bool {
        assert!(self.received <= self.request.len());
        assert!(cached.len() <= RESPONSE_LIMIT);
        if self.response.is_some() {
            return now < self.deadline && self.write_response();
        }
        if now >= self.deadline {
            self.prepare(Status::RequestTimeout, now, cached);
            return true;
        }
        match self.stream.read(&mut self.request[self.received..]) {
            Ok(0) => return false,
            Ok(length) => self.received += length,
            Err(error) => return transient(&error),
        }
        if let RequestState::Complete(status) = parse_request(&self.request[..self.received]) {
            self.prepare(status, now, cached);
        }
        true
    }

    fn prepare(&mut self, status: Status, now: Instant, cached: &Arc<[u8]>) {
        assert!(self.response.is_none());
        assert_eq!(self.written, 0);
        self.response = Some(if status == Status::Ok {
            Arc::clone(cached)
        } else {
            response(status, "request rejected\n")
        });
        self.deadline = now + IO_DEADLINE;
    }

    fn write_response(&mut self) -> bool {
        // Reject extra bytes already queued after the complete request. Bytes
        // arriving after a response starts cause close, never a second response.
        match self.stream.read(&mut [0; 1]) {
            Ok(0) => {}
            Ok(_) if self.written == 0 => {
                self.response = Some(response(Status::BadRequest, "request rejected\n"));
            }
            Ok(_) => return false,
            Err(error) if transient(&error) => {}
            Err(_) => return false,
        }
        let output = self
            .response
            .as_ref()
            .expect("write phase requires a response");
        assert!(output.len() <= RESPONSE_LIMIT);
        assert!(self.written < output.len());
        match self.stream.write(&output[self.written..]) {
            Ok(0) => false,
            Ok(length) => {
                self.written += length;
                self.written < output.len()
            }
            Err(error) => transient(&error),
        }
    }
}

fn transient(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestState {
    Incomplete,
    Complete(Status),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Ok,
    BadRequest,
    NotFound,
    MethodNotAllowed,
    RequestTimeout,
    HeadersTooLarge,
    ServiceUnavailable,
    VersionNotSupported,
}

impl Status {
    fn line(self) -> &'static str {
        match self {
            Self::Ok => "200 OK",
            Self::BadRequest => "400 Bad Request",
            Self::NotFound => "404 Not Found",
            Self::MethodNotAllowed => "405 Method Not Allowed",
            Self::RequestTimeout => "408 Request Timeout",
            Self::HeadersTooLarge => "431 Request Header Fields Too Large",
            Self::ServiceUnavailable => "503 Service Unavailable",
            Self::VersionNotSupported => "505 HTTP Version Not Supported",
        }
    }
}

fn parse_request(input: &[u8]) -> RequestState {
    if input.len() > REQUEST_LIMIT {
        return RequestState::Complete(Status::HeadersTooLarge);
    }
    let Some(end) = input.windows(4).position(|part| part == b"\r\n\r\n") else {
        return if input.len() == REQUEST_LIMIT {
            RequestState::Complete(Status::HeadersTooLarge)
        } else {
            RequestState::Incomplete
        };
    };
    if end + 4 != input.len() || !input.is_ascii() {
        return RequestState::Complete(Status::BadRequest);
    }
    let Ok(text) = std::str::from_utf8(&input[..end]) else {
        return RequestState::Complete(Status::BadRequest);
    };
    RequestState::Complete(validate_request(text).unwrap_or_else(|status| status))
}

fn validate_request(text: &str) -> Result<Status, Status> {
    assert!(text.len() <= REQUEST_LIMIT);
    assert!(text.is_ascii());
    let mut lines = text.split("\r\n");
    let mut parts = lines.next().ok_or(Status::BadRequest)?.split(' ');
    let method = parts.next().ok_or(Status::BadRequest)?;
    let target = parts.next().ok_or(Status::BadRequest)?;
    let version = parts.next().ok_or(Status::BadRequest)?;
    if parts.next().is_some()
        || !token(method)
        || target.is_empty()
        || !target.bytes().all(|byte| (b'!'..=b'~').contains(&byte))
    {
        return Err(Status::BadRequest);
    }
    let mut headers = Headers::default();
    for (index, line) in lines.enumerate() {
        if index >= HEADER_LIMIT {
            return Err(Status::HeadersTooLarge);
        }
        headers.validate(line)?;
    }
    if version == "HTTP/1.1" && !headers.host {
        return Err(Status::BadRequest);
    }
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(Status::VersionNotSupported);
    }
    if method != "GET" {
        return Err(Status::MethodNotAllowed);
    }
    if target != "/metrics" {
        return Err(Status::NotFound);
    }
    Ok(Status::Ok)
}

#[derive(Default)]
struct Headers {
    host: bool,
    content_length: bool,
}

impl Headers {
    fn validate(&mut self, line: &str) -> Result<(), Status> {
        let (name, value) = line.split_once(':').ok_or(Status::BadRequest)?;
        if !token(name)
            || !value
                .bytes()
                .all(|byte| byte == b'\t' || (b' '..=b'~').contains(&byte))
        {
            return Err(Status::BadRequest);
        }
        let value = value.trim_matches([' ', '\t']);
        if name.eq_ignore_ascii_case("host") {
            if self.host || !valid_host(value) {
                return Err(Status::BadRequest);
            }
            self.host = true;
        }
        if name.eq_ignore_ascii_case("content-length") {
            if self.content_length || value != "0" {
                return Err(Status::BadRequest);
            }
            self.content_length = true;
        }
        if ["transfer-encoding", "expect", "upgrade"]
            .iter()
            .any(|name_match| name.eq_ignore_ascii_case(name_match))
        {
            return Err(Status::BadRequest);
        }
        Ok(())
    }
}

fn valid_host(value: &str) -> bool {
    let port = if let Some(address) = value.strip_prefix('[') {
        let Some((address, suffix)) = address.split_once(']') else {
            return false;
        };
        if address.parse::<std::net::Ipv6Addr>().is_err() {
            return false;
        }
        if suffix.is_empty() {
            return true;
        }
        let Some(port) = suffix.strip_prefix(':') else {
            return false;
        };
        Some(port)
    } else {
        let (host, port) = value
            .split_once(':')
            .map_or((value, None), |(host, port)| (host, Some(port)));
        if host.is_empty()
            || !host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".-".contains(&byte))
        {
            return false;
        }
        port
    };
    port.is_none_or(|port| {
        !port.is_empty()
            && port.bytes().all(|byte| byte.is_ascii_digit())
            && port.parse::<u16>().is_ok()
    })
}

fn token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
}

fn make_snapshot(rendered: io::Result<String>, append: impl FnOnce(&mut String)) -> Arc<[u8]> {
    let Ok(mut body) = rendered else {
        return response(Status::ServiceUnavailable, "metrics not available\n");
    };
    if body.len() > BODY_LIMIT {
        return response(
            Status::ServiceUnavailable,
            "metrics snapshot exceeds limit\n",
        );
    }
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    append(&mut body);
    if body.len() > BODY_LIMIT {
        return response(
            Status::ServiceUnavailable,
            "metrics snapshot exceeds limit\n",
        );
    }
    response(Status::Ok, &body)
}

fn response(status: Status, body: &str) -> Arc<[u8]> {
    assert!(body.len() <= BODY_LIMIT);
    let allow = if status == Status::MethodNotAllowed {
        "Allow: GET\r\n"
    } else {
        ""
    };
    let message = format!(
        "HTTP/1.1 {}\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n{allow}\r\n{body}",
        status.line(),
        body.len(),
    );
    assert!(message.len() <= RESPONSE_LIMIT);
    Arc::from(message.into_bytes())
}
