use super::*;

fn exposition(collector: &ResourceCollector) -> String {
    let mut text = String::new();
    collector.append_exposition(&mut text);
    assert!(text.len() <= MAX_EXPOSITION_BYTES);
    assert!(!text.contains("NaN"));
    assert!(!text.contains("inf"));
    text
}

#[test]
fn resource_graphs_publish_units_and_remove_stale_devices() {
    let mut collector = ResourceCollector::new();
    collector.update_cpu(parse_cpu(
        b"cpu 0 0 0 0 0 0 0 0\ncpu0 10 10 10 10 10 10 10 10 999 999\n",
    ));
    collector.update_cpu(parse_cpu(b"cpu0 20 20 20 20 20 20 20 20 99999 99999\n"));
    collector.memory =
        Some(parse_memory(b"MemTotal: 8 kB\nIgnored: 4 kB\nMemAvailable: 3 kB\n").unwrap());
    collector.update_gpu(
        parse_gpu(b"1, 25, 1024, 2048, 55\n0, 100, 0, 4096, 0\n"),
        "0,1",
    );
    let text = exposition(&collector);
    for sample in [
        "drysua_host_cpu_utilization_ratio{cpu=\"0\"} 0.75\n",
        "drysua_host_memory_available_bytes 3072\n",
        "drysua_host_memory_total_bytes 8192\n",
        "drysua_gpu_utilization_ratio{gpu=\"1\"} 0.25\n",
        "drysua_gpu_memory_used_bytes{gpu=\"1\"} 1073741824\n",
        "drysua_gpu_memory_total_bytes{gpu=\"1\"} 2147483648\n",
        "drysua_gpu_temperature_celsius{gpu=\"1\"} 55\n",
    ] {
        assert!(text.contains(sample), "{sample}");
    }
    for failed in [Err("unavailable"), parse_gpu(b"0, 25, 1, 2, 50\n")] {
        collector.update_gpu(failed, "0,1");
        collector.update_cpu(Err("unavailable"));
        collector.memory = None;
        let text = exposition(&collector);
        for family in ["host_cpu", "host_memory", "gpu"] {
            assert!(text.contains(&format!("drysua_{family}_collection_available 0\n")));
        }
        assert!(!text.contains("{gpu="));
        assert!(!text.contains("{cpu="));
        assert!(!text.contains("drysua_host_memory_available_bytes "));
    }
    collector.update_cpu(parse_cpu(b"cpu0 30 30 30 30 30 30 30 30\n"));
    assert!(collector.cpu.iter().all(Option::is_none));
}

#[test]
fn cpu_graphs_omit_missing_reset_and_overflow_deltas() {
    let previous = parse_cpu(b"cpu0 10 0 0 10 10 0 0 0\ncpu1 1 0 0 1 0 0 0 0\n").unwrap();
    assert!(cpu_ratios(None, &previous).iter().all(Option::is_none));
    for input in [
        "cpu0 10 0 0 10 10 0 0 0\n",
        "cpu0 9 0 0 99 10 0 0 0\n",
        "cpu0 20 0 0 20 9 0 0 0\n",
        "cpu0 18446744073709551615 99 0 20 10 0 0 0\n",
        "cpu2 10 0 0 10 0 0 0 0\n",
    ] {
        let current = parse_cpu(input.as_bytes()).unwrap();
        assert!(
            cpu_ratios(Some(&previous), &current)
                .iter()
                .all(Option::is_none),
            "{input}"
        );
    }
    let zero = parse_cpu(b"cpu0 0 0 0 0 0 0 0 0\ncpu255 0 0 0 0 0 0 0 0\n").unwrap();
    let busy = parse_cpu(b"cpu0 0 0 0 1 1 0 0 0\ncpu255 1 1 1 0 0 1 1 1\n").unwrap();
    assert_eq!(cpu_ratios(Some(&zero), &busy)[0], Some(0.0));
    assert_eq!(cpu_ratios(Some(&zero), &busy)[255], Some(1.0));
    let mut collector = ResourceCollector::new();
    for input in [
        "cpu0 1 0 0 1 0 0 0 0\ncpu1 1 0 0 1 0 0 0 0\n",
        "cpu0 2 0 0 2 0 0 0 0\n",
        "cpu0 3 0 0 3 0 0 0 0\ncpu1 9 0 0 9 0 0 0 0\n",
    ] {
        collector.update_cpu(parse_cpu(input.as_bytes()));
    }
    assert_eq!(collector.cpu[0], Some(0.5));
    assert_eq!(collector.cpu[1], None);
}

type Parser = fn(&[u8]) -> Result<(), &'static str>;
const PARSERS: [(Parser, usize, &str); 4] = [
    (
        |bytes| parse_cpu(bytes).map(|_| ()),
        PROC_LIMIT,
        "resource input exceeds 65536 bytes",
    ),
    (
        |bytes| parse_memory(bytes).map(|_| ()),
        PROC_LIMIT,
        "resource input exceeds 65536 bytes",
    ),
    (
        |bytes| parse_gpu(bytes).map(|_| ()),
        GPU_OUTPUT_LIMIT,
        "GPU output exceeds 4096 bytes",
    ),
    (
        |bytes| parse_gpu_ids(bytes).map(|_| ()),
        GPU_OUTPUT_LIMIT,
        "GPU output exceeds 4096 bytes",
    ),
];

#[test]
fn parsers_reject_oversize_before_encoding_and_invalid_fields() {
    for (parse, limit, message) in PARSERS {
        assert_eq!(parse(&[255]), Err("resource input is not UTF-8"));
        for byte in [255, b' '] {
            assert_eq!(parse(&vec![byte; limit + 1]), Err(message));
        }
    }
    for (parse, inputs, message) in INVALID_INPUTS {
        for input in *inputs {
            assert_eq!(parse(input.as_bytes()), Err(*message), "{input}");
        }
    }
    let prefix = "cpu0 0 0 0 0 0 0 0 0\n";
    assert!(
        parse_cpu(format!("{prefix}{}", " ".repeat(PROC_LIMIT - prefix.len())).as_bytes()).is_ok()
    );
    assert_eq!(
        parse_memory(b"MemAvailable: 0 kB\nMemTotal: 1 kB\n")
            .unwrap()
            .available,
        0
    );
    assert_eq!(parse_gpu_ids(b"1\n0\n").unwrap(), "0,1");
    assert_eq!(
        parse_gpu_ids(b"0\n1\n2\n3\n4\n5\n6\n7\n").unwrap(),
        "0,1,2,3,4,5,6,7"
    );
}

const INVALID_INPUTS: &[(Parser, &[&str], &str)] = &[
    (
        PARSERS[0].0,
        &[
            "cpu0 1 2 3\n",
            "cpu0 x 0 0 0 0 0 0 0\n",
            "cpu0 -1 0 0 0 0 0 0 0\n",
            "cpu0 18446744073709551616 0 0 0 0 0 0 0\n",
            "cpu256 1 0 0 0 0 0 0 0\n",
            "cpuX 1 0 0 0 0 0 0 0\n",
            "cpucpu0 1 0 0 0 0 0 0 0\n",
            "cpu0 1 0 0 0 0 0 0 0\ncpu0 2 0 0 0 0 0 0 0\n",
            "intr 1 2 3\n",
        ],
        "invalid /proc/stat CPU sample",
    ),
    (
        PARSERS[1].0,
        &[
            "MemTotal: 8 kB\n",
            "MemTotal: 8 kB\nMemAvailable: 9 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 8 MB\nMemAvailable: 1 kB\n",
            "MemTotal: 8 kB extra\nMemAvailable: 1 kB\n",
            "MemTotal: 8 kB\nMemTotal: 8 kB\nMemAvailable: 1 kB\n",
            "MemTotal: 8 kB\nMemAvailable: 1 kB\nMemAvailable: 1 kB\n",
            "MemTotal: 18446744073709551615 kB\nMemAvailable: 1 kB\n",
            "MemTotal: -8 kB\nMemAvailable: 1 kB\n",
        ],
        "invalid /proc/meminfo memory sample",
    ),
    (
        PARSERS[2].0,
        &[
            "",
            "0, 1, 1, 2\n",
            "0, 1, 1, 2, 3, 4\n",
            "8, 1, 1, 2, 3\n",
            "-1, 1, 1, 2, 3\n",
            "0, NaN, 1, 2, 3\n",
            "0, inf, 1, 2, 3\n",
            "0, -1, 1, 2, 3\n",
            "0, 101, 1, 2, 3\n",
            "0, 1, 3, 2, 3\n",
            "0, 1, 0, 0, 3\n",
            "0, 1, 1.5, 2, 3\n",
            "0, 1, 1, 18446744073709551615, 3\n",
            "0, 1, 1, 2, NaN\n",
            "0, 1, 1, 2, -1\n",
            "0, 1, 1, 2, 201\n",
            "0, 1, 1, 2, 3\n0, 1, 1, 2, 3\n",
            "0, [N/A], 1, 2, 3\n",
        ],
        "invalid nvidia-smi GPU sample",
    ),
    (
        PARSERS[3].0,
        &["", "0\n0\n", "8\n", "00\n", "-1\n", "x\n"],
        "invalid nvidia-smi GPU indices",
    ),
];

#[test]
fn maximum_device_graphs_have_unique_families_and_bounded_exposition() {
    let mut rows = (0..GPU_LIMIT)
        .map(|index| format!("{index}, 100, 1, 2, 200\n"))
        .collect::<String>();
    let mut collector = ResourceCollector::new();
    collector.gpu = parse_gpu(rows.as_bytes()).unwrap();
    assert!(collector.gpu.iter().all(Option::is_some));
    rows.push_str("0, 100, 1, 2, 200\n");
    assert_eq!(
        parse_gpu(rows.as_bytes()).unwrap_err(),
        "invalid nvidia-smi GPU sample"
    );
    collector.cpu = [Some(0.5); CPU_LIMIT];
    collector.memory = Some(MemorySample {
        available: u64::MAX,
        total: u64::MAX,
    });
    let text = exposition(&collector);
    let mut families = std::collections::BTreeSet::new();
    for declaration in text.lines().filter_map(|line| line.strip_prefix("# TYPE ")) {
        let name = declaration.strip_suffix(" gauge").unwrap();
        assert!(families.insert(name), "duplicate family {name}");
        assert_eq!(text.matches(&format!("# HELP {name} ")).count(), 1);
    }
    assert!(text.contains("drysua_host_cpu_utilization_ratio{cpu=\"255\"} 0.5\n"));
    assert!(text.contains("drysua_gpu_temperature_celsius{gpu=\"7\"} 200\n"));
    collector.cpu = [cpu_ratio([0; 8], [1, 0, 0, u64::MAX - 1, 0, 0, 0, 0]); CPU_LIMIT];
    collector.gpu = [Some(GpuSample {
        utilization: f64::from_bits(1),
        used: u64::MAX,
        total: u64::MAX,
        temperature: f64::from_bits(1),
    }); GPU_LIMIT];
    exposition(&collector);
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::{cell::Cell, io::Cursor, os::unix::process::ExitStatusExt};

    const TIMEOUT: &str = "nvidia-smi collection deadline exceeded";

    struct TimedRead<'a>(&'a Cell<Instant>, Instant, Option<io::ErrorKind>);
    impl Read for TimedRead<'_> {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.0.set(self.1);
            self.2
                .map_or(Ok(0), |kind| Err(io::Error::new(kind, "read denied")))
        }
    }

    #[test]
    fn gpu_deadline_covers_read_and_wait_while_preserving_reaped_status() {
        let now = Instant::now();
        let deadline = now + GPU_DEADLINE;
        let before = deadline - Duration::from_nanos(1);
        for (read_at, wait_at, exited, expected_polls, success) in [
            (deadline, now, true, 0, false),
            (now, deadline, true, 1, false),
            (now, deadline, false, 1, false),
            (before, before, true, 1, true),
        ] {
            let clock = Cell::new(now);
            let mut output = GpuOutput::new(now);
            let mut eof = false;
            let mut status = None;
            let mut polls = 0;
            let result = poll_gpu(
                &mut output,
                &mut eof,
                &mut status,
                &mut TimedRead(&clock, read_at, None),
                || {
                    polls += 1;
                    clock.set(wait_at);
                    Ok(exited.then(|| ExitStatus::from_raw(0)))
                },
                || clock.get(),
            );
            assert_eq!(result, if success { Ok(true) } else { Err(TIMEOUT) });
            assert_eq!(polls, expected_polls);
            assert_eq!(status.is_some(), polls == 1 && exited);
            assert_eq!(output.deadline, deadline);
        }
    }

    #[test]
    fn reaped_child_is_not_polled_again_while_draining_or_checking_expiry() {
        let now = Instant::now();
        let mut output = GpuOutput::new(now);
        let mut eof = false;
        let mut status = None;
        let mut input = Cursor::new(b"0, 25, 1, 2, 50\n");
        let mut polls = 0;
        for (at, expected) in [
            (now, Ok(false)),
            (now, Ok(true)),
            (now + GPU_DEADLINE, Err(TIMEOUT)),
        ] {
            let result = poll_gpu(
                &mut output,
                &mut eof,
                &mut status,
                &mut input,
                || {
                    polls += 1;
                    assert_eq!(polls, 1, "double reap");
                    Ok(Some(ExitStatus::from_raw(0)))
                },
                || at,
            );
            assert_eq!(result, expected);
        }
        assert_eq!(output.bytes(), b"0, 25, 1, 2, 50\n");
        assert_eq!(polls, 1);
    }

    #[test]
    fn gpu_child_failures_preserve_diagnostics() {
        let now = Instant::now();
        for (waited, message) in [
            (
                Ok(Some(ExitStatus::from_raw(256))),
                "nvidia-smi child exited unsuccessfully",
            ),
            (
                Err(io::Error::other("wait denied")),
                "cannot poll nvidia-smi child",
            ),
        ] {
            let mut waited = Some(waited);
            assert_eq!(
                poll_gpu(
                    &mut GpuOutput::new(now),
                    &mut true,
                    &mut None,
                    &mut Cursor::new([]),
                    || waited.take().expect("one poll per tick"),
                    || now
                ),
                Err(message)
            );
        }
    }

    #[test]
    fn resource_reads_bound_bytes_preserve_errors_and_never_extend_deadlines() {
        let now = Instant::now();
        let clock = Cell::new(now);
        let mut output = GpuOutput::new(now);
        for (kind, expected) in [
            (io::ErrorKind::WouldBlock, Ok(false)),
            (io::ErrorKind::Interrupted, Ok(false)),
            (
                io::ErrorKind::ConnectionReset,
                Err("cannot read nvidia-smi output"),
            ),
        ] {
            let result = output.read(&mut TimedRead(&clock, now, Some(kind)), now);
            assert_eq!(result, expected);
            assert_eq!(output.deadline, now + GPU_DEADLINE);
        }
        for length in [GPU_OUTPUT_LIMIT, GPU_OUTPUT_LIMIT + 1] {
            let mut output = GpuOutput::new(now);
            let result = output.read(&mut Cursor::new(vec![b'x'; length]), now);
            assert_eq!(
                result,
                if length == GPU_OUTPUT_LIMIT {
                    Ok(false)
                } else {
                    Err("GPU output exceeds 4096 bytes")
                }
            );
            if length == GPU_OUTPUT_LIMIT {
                assert_eq!(output.read(&mut Cursor::new([]), now), Ok(true));
                assert_eq!(output.bytes().len(), length);
            }
        }
        assert_eq!(
            output.read(&mut Cursor::new([]), now + GPU_DEADLINE),
            Err(TIMEOUT)
        );
        let mut buffer = [0; PROC_LIMIT + 1];
        assert_eq!(
            read_bounded(&mut Cursor::new(vec![0; PROC_LIMIT]), &mut buffer).unwrap(),
            PROC_LIMIT
        );
        assert_eq!(
            read_bounded(&mut Cursor::new(vec![0; PROC_LIMIT + 1]), &mut buffer)
                .unwrap_err()
                .to_string(),
            "resource input exceeds 65536 bytes"
        );
        let error = read_bounded(
            &mut TimedRead(&clock, now, Some(io::ErrorKind::PermissionDenied)),
            &mut buffer,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(error.to_string(), "read denied");
    }

    #[test]
    fn expired_work_is_not_started_and_late_owned_results_are_dropped() {
        struct Owned<'a>(&'a Cell<usize>);
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let now = Instant::now();
        let deadline = now + GPU_DEADLINE;
        let clock = Cell::new(now);
        let drops = Cell::new(0);
        assert_eq!(
            with_gpu_deadline::<()>(deadline, || deadline, || panic!("expired work")),
            Err(TIMEOUT)
        );
        let result = with_gpu_deadline(
            deadline,
            || clock.get(),
            || {
                clock.set(deadline);
                Ok::<_, io::Error>(Owned(&drops))
            },
        );
        assert_eq!(result.err(), Some(TIMEOUT));
        assert_eq!(drops.get(), 1);
        let missing = io::Error::new(io::ErrorKind::NotFound, "missing nvidia-smi");
        let error = with_gpu_deadline(deadline, || now, || Err::<(), _>(missing))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(error.to_string(), "missing nvidia-smi");
    }

    #[test]
    fn late_parsing_discards_even_previously_available_gpu_samples() {
        let now = Instant::now();
        let deadline = now + GPU_DEADLINE;
        let mut collector = ResourceCollector::new();
        collector.finish_gpu(b"0, 25, 1, 2, 60\n", "0", deadline, || now);
        assert!(exposition(&collector).contains("drysua_gpu_temperature_celsius{gpu=\"0\"} 60\n"));
        let checks = Cell::new(0);
        collector.finish_gpu(b"0, 50, 1, 2, 70\n", "0", deadline, || {
            let check = checks.replace(checks.get() + 1);
            if check == 0 { now } else { deadline }
        });
        let text = exposition(&collector);
        assert!(text.contains("drysua_gpu_collection_available 0\n"));
        assert!(!text.contains("{gpu="));
    }
}
