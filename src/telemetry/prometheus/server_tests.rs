use super::*;
use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::mpsc;

#[test]
fn worker_accept_failure_is_reported_before_join_and_keeps_its_error() {
    let reported = Cell::new(false);
    let error = finish_worker(
        Err(io::Error::other("accept: too many open files")),
        |error| {
            assert_eq!(error.to_string(), "accept: too many open files");
            reported.set(true);
        },
    )
    .unwrap_err();
    assert!(reported.get());
    assert_eq!(error.to_string(), "accept: too many open files");
}

#[test]
fn graceful_worker_shutdown_does_not_report_failure() {
    finish_worker(Ok(()), |_| panic!("graceful shutdown is not an error")).unwrap();
}

#[test]
fn run_with_shutdown_already_requested_skips_rendering_and_resource_collection() {
    let caller = thread::current().id();
    let checks = Cell::new(0);

    MetricsServer::run(
        "127.0.0.1:0".parse().unwrap(),
        || panic!("shutdown must precede rendering and resource collection"),
        || {
            assert_eq!(thread::current().id(), caller);
            checks.set(checks.get() + 1);
            true
        },
    )
    .unwrap();

    assert_eq!(checks.get(), 1);
}

#[test]
fn run_supports_borrowed_non_send_callbacks_and_rechecks_shutdown_after_render() {
    let caller = thread::current().id();
    let requested = Rc::new(Cell::new(false));
    let mut renders = 0;

    MetricsServer::run(
        "127.0.0.1:0".parse().unwrap(),
        || {
            assert_eq!(thread::current().id(), caller);
            renders += 1;
            requested.set(true);
            Ok("training 1\n".into())
        },
        || requested.get(),
    )
    .unwrap();

    assert_eq!(renders, 1);
    assert!(requested.get());
}

#[test]
fn run_observes_static_atomic_shutdown_after_a_render_error_without_collecting() {
    static REQUESTED: AtomicBool = AtomicBool::new(false);
    REQUESTED.store(false, Ordering::Relaxed);
    let mut renders = 0;

    MetricsServer::run(
        "127.0.0.1:0".parse().unwrap(),
        || {
            renders += 1;
            REQUESTED.store(true, Ordering::Relaxed);
            Err(io::Error::other("mock render unavailable"))
        },
        || REQUESTED.load(Ordering::Relaxed),
    )
    .unwrap();

    assert_eq!(renders, 1);
    assert!(REQUESTED.load(Ordering::Relaxed));
}

#[test]
fn run_rejects_nonloopback_before_calling_either_callback() {
    for address in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0"] {
        let error = MetricsServer::run(
            address.parse().unwrap(),
            || panic!("invalid bind must not render"),
            || panic!("invalid bind must not enter the event loop"),
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
fn complete_metrics_requests_accept_only_supported_versions() {
    for request in [
        &b"GET /metrics HTTP/1.0\r\n\r\n"[..],
        &b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n"[..],
        &b"GET /metrics HTTP/1.1\r\nhOsT: [::1]:9100\r\nContent-Length: 0\r\n\r\n"[..],
    ] {
        assert_eq!(parse_request(request), RequestState::Complete(Status::Ok));
    }
}

#[test]
fn incomplete_headers_never_trigger_success() {
    let request = b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n";
    for length in 0..request.len() {
        assert_eq!(parse_request(&request[..length]), RequestState::Incomplete);
    }
}

#[test]
fn routing_rejects_queries_other_paths_methods_and_versions() {
    for (request, expected) in [
        ("GET /metrics? HTTP/1.0\r\n\r\n", Status::NotFound),
        ("GET /metrics?x=1 HTTP/1.0\r\n\r\n", Status::NotFound),
        ("GET /metrics/ HTTP/1.0\r\n\r\n", Status::NotFound),
        ("GET / HTTP/1.0\r\n\r\n", Status::NotFound),
        ("HEAD /metrics HTTP/1.0\r\n\r\n", Status::MethodNotAllowed),
        ("POST /metrics HTTP/1.0\r\n\r\n", Status::MethodNotAllowed),
        ("GET /metrics HTTP/2.0\r\n\r\n", Status::VersionNotSupported),
    ] {
        assert_eq!(
            parse_request(request.as_bytes()),
            RequestState::Complete(expected)
        );
    }
}

#[test]
fn malformed_headers_bodies_and_pipelining_are_rejected() {
    for request in [
        "GET  /metrics HTTP/1.0\r\n\r\n",
        "GET /metrics HTTP/1.1\r\n\r\n",
        "GET /metrics HTTP/1.1\r\nHost: \r\n\r\n",
        "GET /metrics HTTP/1.1\r\nHost: a\r\nHOST: b\r\n\r\n",
        "GET /metrics HTTP/1.1\r\nHost: a b\r\n\r\n",
        "GET /metrics HTTP/1.0\r\n folded: value\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nBad Name: value\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nName : value\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nMissing-colon\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nX: a\0b\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nX: a\nb\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nContent-Length: 1\r\n\r\nx",
        "GET /metrics HTTP/1.0\r\nContent-Length: 1\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nContent-Length: +0\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nExpect: 100-continue\r\n\r\n",
        "GET /metrics HTTP/1.0\r\nUpgrade: websocket\r\n\r\n",
        "GET /metrics HTTP/1.0\r\n\r\nGET /metrics HTTP/1.0\r\n\r\n",
    ] {
        assert_eq!(
            parse_request(request.as_bytes()),
            RequestState::Complete(Status::BadRequest)
        );
    }
}

#[test]
fn host_authority_rejects_malformed_addresses_and_ports() {
    for host in [
        ":80",
        "[]",
        "[not-ip]",
        "[::1",
        "localhost:",
        "localhost:65536",
        "localhost:80:90",
    ] {
        let request = format!("GET /metrics HTTP/1.1\r\nHost: {host}\r\n\r\n");
        assert_eq!(
            parse_request(request.as_bytes()),
            RequestState::Complete(Status::BadRequest)
        );
    }
    for host in [
        "localhost",
        "localhost:9100",
        "127.0.0.1",
        "[::1]",
        "[::1]:9100",
    ] {
        let request = format!("GET /metrics HTTP/1.1\r\nHost: {host}\r\n\r\n");
        assert_eq!(
            parse_request(request.as_bytes()),
            RequestState::Complete(Status::Ok)
        );
    }
}

#[test]
fn header_byte_limit_accepts_exact_boundary_and_rejects_overflow() {
    let prefix = "GET /metrics HTTP/1.0\r\nX: ";
    let request = format!(
        "{prefix}{}\r\n\r\n",
        "a".repeat(REQUEST_LIMIT - prefix.len() - 4)
    );
    assert_eq!(request.len(), REQUEST_LIMIT);
    assert_eq!(
        parse_request(request.as_bytes()),
        RequestState::Complete(Status::Ok)
    );
    assert_eq!(
        parse_request(&[b'a'; REQUEST_LIMIT]),
        RequestState::Complete(Status::HeadersTooLarge)
    );
    assert_eq!(
        parse_request(&[b'a'; REQUEST_LIMIT + 1]),
        RequestState::Complete(Status::HeadersTooLarge)
    );
}

#[test]
fn header_count_limit_is_enforced_independently_of_bytes() {
    let allowed = format!(
        "GET /metrics HTTP/1.0\r\n{}\r\n",
        "X: a\r\n".repeat(HEADER_LIMIT)
    );
    assert_eq!(
        parse_request(allowed.as_bytes()),
        RequestState::Complete(Status::Ok)
    );
    let rejected = format!(
        "GET /metrics HTTP/1.0\r\n{}\r\n",
        "X: a\r\n".repeat(HEADER_LIMIT + 1)
    );
    assert_eq!(
        parse_request(rejected.as_bytes()),
        RequestState::Complete(Status::HeadersTooLarge)
    );
}

#[derive(Default)]
struct MockStream {
    incoming: VecDeque<Vec<u8>>,
    outgoing: Vec<u8>,
    write_limit: Option<usize>,
    read_error: Option<io::ErrorKind>,
    write_error: Option<io::ErrorKind>,
    eof: bool,
}

impl Read for MockStream {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if let Some(kind) = self.read_error.take() {
            return Err(io::Error::from(kind));
        }
        let Some(input) = self.incoming.front_mut() else {
            return if self.eof {
                Ok(0)
            } else {
                Err(io::ErrorKind::WouldBlock.into())
            };
        };
        let length = input.len().min(output.len());
        output[..length].copy_from_slice(&input[..length]);
        drop(input.drain(..length));
        if input.is_empty() {
            self.incoming.pop_front();
        }
        Ok(length)
    }
}

impl Write for MockStream {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if let Some(kind) = self.write_error.take() {
            return Err(io::Error::from(kind));
        }
        let length = self.write_limit.unwrap_or(input.len()).min(input.len());
        self.outgoing.extend_from_slice(&input[..length]);
        Ok(length)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn mock_client(request: &[u8], now: Instant) -> Client<MockStream> {
    let mut stream = MockStream::default();
    stream.incoming.push_back(request.to_vec());
    Client::new(stream, now)
}

#[test]
fn fragmented_request_uses_cached_response_and_closes_after_write() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(b"GET /met", now);
    assert!(client.advance(now, &cached));
    client
        .stream
        .incoming
        .push_back(b"rics HTTP/1.0\r\n\r\n".to_vec());
    assert!(client.advance(now, &cached));
    assert!(Arc::ptr_eq(client.response.as_ref().unwrap(), &cached));
    assert!(!client.advance(now, &cached));
    assert_eq!(client.stream.outgoing, cached.as_ref());
}

#[test]
fn absolute_read_deadline_does_not_extend_with_progress() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(b"G", now);
    assert!(client.advance(now + IO_DEADLINE / 2, &cached));
    assert!(client.advance(now + IO_DEADLINE, &cached));
    assert!(!client.advance(now + IO_DEADLINE, &cached));
    assert!(
        client
            .stream
            .outgoing
            .starts_with(b"HTTP/1.1 408 Request Timeout\r\n")
    );
}

#[test]
fn absolute_write_deadline_does_not_extend_with_partial_writes() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
    client.stream.write_limit = Some(1);
    assert!(client.advance(now, &cached));
    assert!(client.advance(now + IO_DEADLINE / 2, &cached));
    assert!(!client.advance(now + IO_DEADLINE, &cached));
    assert_eq!(client.stream.outgoing.len(), 1);
}

#[test]
fn delayed_pipeline_is_rejected_before_any_success_bytes() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
    assert!(client.advance(now, &cached));
    client
        .stream
        .incoming
        .push_back(b"GET /metrics HTTP/1.0\r\n\r\n".to_vec());
    assert!(!client.advance(now, &cached));
    assert!(
        client
            .stream
            .outgoing
            .starts_with(b"HTTP/1.1 400 Bad Request\r\n")
    );
}

#[test]
fn disconnected_and_broken_clients_are_removed_without_retries() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = Client::new(
        MockStream {
            eof: true,
            ..MockStream::default()
        },
        now,
    );
    assert!(!client.advance(now, &cached));
    let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
    assert!(client.advance(now, &cached));
    client.stream.write_error = Some(io::ErrorKind::BrokenPipe);
    assert!(!client.advance(now, &cached));
}

#[test]
fn half_closed_valid_request_still_receives_one_response() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
    client.stream.eof = true;
    assert!(client.advance(now, &cached));
    assert!(!client.advance(now, &cached));
    assert_eq!(client.stream.outgoing, cached.as_ref());
}

#[test]
fn oversized_socket_read_receives_431_without_using_cached_metrics() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut client = mock_client(&[b'x'; REQUEST_LIMIT + 1], now);
    assert!(client.advance(now, &cached));
    assert!(!client.advance(now, &cached));
    assert!(
        client
            .stream
            .outgoing
            .starts_with(b"HTTP/1.1 431 Request Header Fields Too Large\r\n")
    );
}

#[test]
fn blocked_client_does_not_prevent_other_slot_progress() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    let mut blocked = mock_client(b"G", now);
    let mut ready = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
    for client in [&mut blocked, &mut ready] {
        assert!(client.advance(now, &cached));
    }
    assert!(blocked.advance(now, &cached));
    assert!(!ready.advance(now, &cached));
    assert_eq!(ready.stream.outgoing, cached.as_ref());
}

#[test]
fn transient_io_errors_preserve_deadline_without_spinning() {
    let now = Instant::now();
    let cached = response(Status::Ok, "metric 1\n");
    for kind in [io::ErrorKind::WouldBlock, io::ErrorKind::Interrupted] {
        let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
        client.stream.read_error = Some(kind);
        assert!(client.advance(now, &cached));
        assert_eq!(client.deadline, now + IO_DEADLINE);
        assert!(client.advance(now, &cached));
        client.stream.write_error = Some(kind);
        assert!(client.advance(now, &cached));
        assert!(!client.advance(now + IO_DEADLINE, &cached));
    }
}

#[test]
fn responses_have_exact_length_close_and_prometheus_content_type() {
    let bytes = response(Status::Ok, "metric 1\n");
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("Content-Length: 9\r\n"));
    assert!(text.contains("Connection: close\r\n"));
    assert!(text.contains("Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"));
    assert!(text.ends_with("\r\n\r\nmetric 1\n"));
    assert!(
        response(Status::MethodNotAllowed, "")
            .windows(12)
            .any(|part| part == b"Allow: GET\r\n")
    );
}

#[test]
fn render_failures_and_oversize_snapshots_fail_closed() {
    let failed = make_snapshot(Err(io::Error::other("private detail")), |_| {
        panic!("must not append")
    });
    assert!(failed.starts_with(b"HTTP/1.1 503 Service Unavailable\r\n"));
    assert!(
        !std::str::from_utf8(&failed)
            .unwrap()
            .contains("private detail")
    );
    let oversized = make_snapshot(Ok("x".repeat(BODY_LIMIT + 1)), |_| {
        panic!("must not append")
    });
    assert!(oversized.starts_with(b"HTTP/1.1 503 Service Unavailable\r\n"));
    let combined = make_snapshot(Ok("x".repeat(BODY_LIMIT)), |text| text.push('x'));
    assert!(combined.starts_with(b"HTTP/1.1 503 Service Unavailable\r\n"));
}

#[test]
fn render_error_response_is_reused_without_recollecting_or_exposing_error_details() {
    let now = Instant::now();
    let cached = make_snapshot(Err(io::Error::other("private render failure")), |_| {
        panic!("failed rendering must not append resource samples")
    });

    for _ in 0..CLIENT_LIMIT {
        let mut client = mock_client(b"GET /metrics HTTP/1.0\r\n\r\n", now);
        assert!(client.advance(now, &cached));
        assert!(Arc::ptr_eq(client.response.as_ref().unwrap(), &cached));
        assert!(!client.advance(now, &cached));
        assert_eq!(client.stream.outgoing, cached.as_ref());
    }

    assert!(cached.starts_with(b"HTTP/1.1 503 Service Unavailable\r\n"));
    assert!(
        !std::str::from_utf8(&cached)
            .unwrap()
            .contains("private render failure")
    );
}

#[test]
fn resource_exposition_is_appended_with_line_boundary_and_response_cap() {
    let snapshot = make_snapshot(Ok("training 1".into()), |text| {
        text.push_str("resource 1\n")
    });
    assert!(snapshot.ends_with(b"training 1\nresource 1\n"));
    let maximum = make_snapshot(Ok("x".repeat(BODY_LIMIT - 1)), |_| {});
    assert!(maximum.starts_with(b"HTTP/1.1 200 OK\r\n"));
    assert!(maximum.len() <= RESPONSE_LIMIT);
}

#[test]
fn collection_schedule_never_catches_up_or_amplifies_client_load() {
    let now = Instant::now();
    let mut schedule = CollectionSchedule::new(now);
    assert!(schedule.take_due(now));
    for _ in 0..100 {
        assert!(!schedule.take_due(now));
    }
    assert!(!schedule.take_due(now + COLLECTION_INTERVAL - Duration::from_nanos(1)));
    assert!(schedule.take_due(now + COLLECTION_INTERVAL));
    assert!(schedule.take_due(now + COLLECTION_INTERVAL * 10));
    assert!(!schedule.take_due(now + COLLECTION_INTERVAL * 10));
}

#[test]
fn slow_collection_cannot_shorten_next_collection_interval() {
    let now = Instant::now();
    let mut schedule = CollectionSchedule::new(now);
    assert!(schedule.take_due(now));
    let finished = now + Duration::from_secs(1);
    schedule.finish_collection(finished);
    assert!(!schedule.take_due(now + COLLECTION_INTERVAL));
    assert!(schedule.take_due(finished + COLLECTION_INTERVAL));
}

#[test]
fn explicit_nonloopback_binds_fail_before_starting_worker_or_resources() {
    for address in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0"] {
        let result = MetricsServer::start(address.parse().unwrap(), || panic!("must not render"));
        let error = match result {
            Ok(_) => panic!("nonloopback must fail"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "metrics listener must bind a loopback address"
        );
    }
}

fn parked_server() -> (MetricsServer, mpsc::Receiver<Arc<AtomicBool>>) {
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let (sender, receiver) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        thread::park();
        sender.send(worker_stop).unwrap();
        Ok(())
    });
    let server = MetricsServer {
        stop,
        worker: Some(worker),
        local_addr: "127.0.0.1:9100".parse().unwrap(),
    };
    (server, receiver)
}

#[test]
fn explicit_shutdown_signals_unparks_and_joins_worker() {
    let (server, receiver) = parked_server();
    assert_eq!(server.local_addr(), "127.0.0.1:9100".parse().unwrap());
    server.shutdown().unwrap();
    assert!(receiver.try_recv().unwrap().load(Ordering::Acquire));
}

#[test]
fn drop_signals_unparks_and_joins_worker() {
    let (server, receiver) = parked_server();
    drop(server);
    assert!(receiver.try_recv().unwrap().load(Ordering::Acquire));
}

#[test]
fn worker_panic_is_reported_by_shutdown() {
    let server = MetricsServer {
        stop: Arc::new(AtomicBool::new(false)),
        worker: Some(thread::spawn(|| panic!("mock worker panic"))),
        local_addr: "127.0.0.1:9100".parse().unwrap(),
    };
    let error = server.shutdown().unwrap_err();
    assert_eq!(error.to_string(), "metrics server worker panicked");
}
