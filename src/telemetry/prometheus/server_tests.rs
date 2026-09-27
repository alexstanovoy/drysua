use super::*;
use std::cell::Cell;

// Fixed snapshots isolate the socket contract from host probes and collection timing.
fn exchange(request: &[u8], cached: Arc<[u8]>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut socket = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let timeout = Some(Duration::from_secs(5));
    socket.set_read_timeout(timeout).unwrap();
    socket.set_write_timeout(timeout).unwrap();
    socket.write_all(request).unwrap();
    socket.shutdown(std::net::Shutdown::Write).unwrap();
    let worker = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(timeout).unwrap();
        stream.set_write_timeout(timeout).unwrap();
        let now = Instant::now();
        let mut client = Client::new(stream, now);
        for _ in 0..REQUEST_LIMIT + 2 {
            if !client.advance(now, &cached) {
                return;
            }
        }
        panic!("request failed to complete within its byte budget");
    });
    let mut bytes = Vec::new();
    let read = socket
        .take((RESPONSE_LIMIT + 1) as u64)
        .read_to_end(&mut bytes);
    worker.join().unwrap();
    if let Err(error) = read {
        assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
    }
    assert!(bytes.len() <= RESPONSE_LIMIT);
    String::from_utf8(bytes).unwrap()
}

#[test]
fn http_rejections_never_leak_cached_metrics() {
    for (request, status) in [
        ("GET /metrics?x=1 HTTP/1.0\r\n\r\n", 404),
        ("GET /metrics/ HTTP/1.0\r\n\r\n", 404),
        ("HEAD /metrics HTTP/1.0\r\n\r\n", 405),
        ("POST /metrics HTTP/1.0\r\n\r\n", 405),
        ("GET /metrics HTTP/2.0\r\n\r\n", 505),
        ("GET  /metrics HTTP/1.0\r\n\r\n", 400),
        ("GET /metrics HTTP/1.1\r\n\r\n", 400),
        ("GET /metrics HTTP/1.0\r\nContent-Length: 1\r\n\r\nx", 400),
        (
            "GET /metrics HTTP/1.0\r\n\r\nGET /metrics HTTP/1.0\r\n\r\n",
            400,
        ),
    ] {
        assert_rejected(request, status);
    }
    for header in [
        "Host: a\r\nHOST: b",
        " folded: value",
        "Bad Name: value",
        "Name : value",
        "Missing-colon",
        "X: a\0b",
        "X: a\nb",
        "Content-Length: 1",
        "Content-Length: +0",
        "Content-Length: 0\r\nContent-Length: 0",
        "Transfer-Encoding: chunked",
        "Expect: 100-continue",
        "Upgrade: websocket",
    ] {
        assert_rejected(&format!("GET /metrics HTTP/1.0\r\n{header}\r\n\r\n"), 400);
    }
    for host in [
        "",
        "a b",
        ":80",
        "[]",
        "[not-ip]",
        "[::1",
        "localhost:",
        "localhost:65536",
        "localhost:80:90",
    ] {
        assert_rejected(
            &format!("GET /metrics HTTP/1.1\r\nHost: {host}\r\n\r\n"),
            400,
        );
    }
}

fn assert_rejected(request: &str, status: u16) {
    let reply = exchange(
        request.as_bytes(),
        response(Status::Ok, "secret_metric 1\n"),
    );
    assert!(
        reply.starts_with(&format!("HTTP/1.1 {status} ")),
        "{request:?}: {reply}"
    );
    assert!(!reply.contains("secret_metric"));
    if status == 405 {
        assert!(reply.contains("Allow: GET\r\n"));
    }
}

#[test]
fn http_header_bounds_accept_exact_limits_and_reject_overflow() {
    let prefix = "GET /metrics HTTP/1.0\r\nX: ";
    let exact = format!(
        "{prefix}{}\r\n\r\n",
        "a".repeat(REQUEST_LIMIT - prefix.len() - 4)
    );
    let headers = |count| format!("GET /metrics HTTP/1.0\r\n{}\r\n", "X: a\r\n".repeat(count));
    for (request, status) in [
        (exact, 200),
        ("a".repeat(REQUEST_LIMIT), 431),
        ("a".repeat(REQUEST_LIMIT + 1), 431),
        (headers(HEADER_LIMIT), 200),
        (headers(HEADER_LIMIT + 1), 431),
    ] {
        let reply = exchange(request.as_bytes(), response(Status::Ok, "metric 1\n"));
        // An overflow split at the cap may instead be rejected as trailing bytes.
        assert!(
            reply.starts_with(&format!("HTTP/1.1 {status} "))
                || (request.len() > REQUEST_LIMIT && reply.starts_with("HTTP/1.1 400 "))
        );
        assert_eq!(reply.contains("\r\n\r\nmetric 1\n"), status == 200);
    }
}

#[cfg(unix)]
impl super::super::DirectoryReader {
    pub(in crate::telemetry::prometheus) fn scrape_for_test(&mut self, request: &str) -> String {
        let reply = exchange(request.as_bytes(), make_snapshot(self.render(), |_| {}));
        let (headers, body) = reply.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(headers.contains(&format!("Content-Length: {}", body.len())));
        assert!(headers.contains("Connection: close"));
        assert!(headers.contains("Content-Type: text/plain; version=0.0.4; charset=utf-8"));
        body.to_owned()
    }
}

#[test]
fn slow_clients_do_not_block_other_scrapes_or_shutdown() {
    for explicit in [false, true] {
        let server =
            MetricsServer::start("127.0.0.1:0".parse().unwrap(), || Ok("metric 1\n".into()))
                .unwrap();
        let mut slow = TcpStream::connect(server.local_addr()).unwrap();
        slow.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        slow.write_all(b"G").unwrap();
        let mut ready = TcpStream::connect(server.local_addr()).unwrap();
        ready
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        ready.write_all(b"GET /metrics HTTP/1.0\r\n\r\n").unwrap();
        ready.shutdown(std::net::Shutdown::Write).unwrap();
        let mut reply = String::new();
        ready
            .take(RESPONSE_LIMIT as u64)
            .read_to_string(&mut reply)
            .unwrap();
        assert!(reply.starts_with("HTTP/1.1 200 ") || reply.starts_with("HTTP/1.1 503 "));
        if explicit {
            server.shutdown().unwrap();
        } else {
            drop(server);
        }
        let mut closed = String::new();
        if let Err(error) = slow.take(RESPONSE_LIMIT as u64).read_to_string(&mut closed) {
            assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
        }
        assert!(closed.is_empty() || closed.starts_with("HTTP/1.1 408 Request Timeout\r\n"));
    }
}

#[test]
fn caller_thread_shutdown_precedes_collection_even_after_render_errors() {
    for (already_requested, fails) in [(true, false), (false, false), (false, true)] {
        let requested = std::rc::Rc::new(Cell::new(already_requested));
        let caller = thread::current().id();
        let mut renders = 0;
        MetricsServer::run(
            "127.0.0.1:0".parse().unwrap(),
            || {
                assert_eq!(thread::current().id(), caller);
                renders += 1;
                requested.set(true);
                if fails {
                    Err(io::Error::other("render unavailable"))
                } else {
                    Ok("metric 1\n".into())
                }
            },
            || requested.get(),
        )
        .unwrap();
        assert_eq!(renders, usize::from(!already_requested));
    }
    for address in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0"] {
        let error = MetricsServer::run(
            address.parse().unwrap(),
            || panic!("render"),
            || panic!("poll"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "metrics listener must bind a loopback address"
        );
    }
}

#[test]
fn snapshots_fail_closed_and_collection_never_catches_up() {
    for rendered in [
        Err(io::Error::other("private detail")),
        Ok("x".repeat(BODY_LIMIT + 1)),
    ] {
        let reply = exchange(
            b"GET /metrics HTTP/1.0\r\n\r\n",
            make_snapshot(rendered, |_| panic!("must not collect")),
        );
        assert!(reply.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
        assert!(!reply.contains("private detail"));
    }
    assert!(
        make_snapshot(Ok("x".repeat(BODY_LIMIT)), |text| text.push('x'))
            .starts_with(b"HTTP/1.1 503 ")
    );
    assert!(
        make_snapshot(Ok("training 1".into()), |text| text
            .push_str("resource 1\n"))
        .ends_with(b"training 1\nresource 1\n")
    );
    let now = Instant::now();
    let mut schedule = CollectionSchedule::new(now);
    assert!(schedule.take_due(now));
    let finished = now + COLLECTION_INTERVAL * 10;
    schedule.finish_collection(finished);
    assert!(!schedule.take_due(finished + COLLECTION_INTERVAL - Duration::from_nanos(1)));
    assert!(schedule.take_due(finished + COLLECTION_INTERVAL));
    assert!(!schedule.take_due(finished + COLLECTION_INTERVAL));
}

struct PartialStream {
    input: &'static [u8],
    output: Vec<u8>,
    write_limit: usize,
    error: Option<io::ErrorKind>,
}

impl Read for PartialStream {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if let Some(kind) = self.error.take() {
            return Err(kind.into());
        }
        if self.input.is_empty() {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        self.input.read(bytes)
    }
}

impl Write for PartialStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let length = bytes.len().min(self.write_limit);
        self.output.extend_from_slice(&bytes[..length]);
        Ok(length)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn fragmented_and_transient_io_cannot_extend_absolute_client_deadlines() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    for kind in [io::ErrorKind::WouldBlock, io::ErrorKind::Interrupted] {
        let stream = PartialStream {
            input: b"GET /met",
            output: Vec::new(),
            write_limit: 1,
            error: Some(kind),
        };
        let mut client = Client::new(stream, now);
        assert!(client.advance(now, &cached));
        assert!(client.advance(now + IO_DEADLINE / 2, &cached));
        assert!(client.stream.output.is_empty());
        client.stream.input = b"rics HTTP/1.0\r\n\r\n";
        assert!(client.advance(now + IO_DEADLINE / 2, &cached));
        assert!(client.advance(now + IO_DEADLINE, &cached));
        assert!(!client.advance(now + IO_DEADLINE + IO_DEADLINE / 2, &cached));
        assert_eq!(client.stream.output, b"H");
        let stream = PartialStream {
            input: b"G",
            output: Vec::new(),
            write_limit: RESPONSE_LIMIT,
            error: None,
        };
        let mut client = Client::new(stream, now);
        assert!(client.advance(now + IO_DEADLINE / 2, &cached));
        assert!(client.advance(now + IO_DEADLINE, &cached));
        assert!(!client.advance(now + IO_DEADLINE, &cached));
        assert!(
            client
                .stream
                .output
                .starts_with(b"HTTP/1.1 408 Request Timeout\r\n")
        );
    }
}
