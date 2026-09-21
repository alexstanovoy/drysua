use super::*;
#[cfg(target_os = "linux")]
use std::cell::Cell;
#[cfg(target_os = "linux")]
use std::os::unix::process::ExitStatusExt;

#[cfg(target_os = "linux")]
struct AdvancingEof<'a> {
    clock: &'a Cell<Instant>,
    completed: Instant,
}

#[cfg(target_os = "linux")]
impl Read for AdvancingEof<'_> {
    fn read(&mut self, _output: &mut [u8]) -> io::Result<usize> {
        self.clock.set(self.completed);
        Ok(0)
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_rejects_eof_at_deadline_without_polling_the_child() {
    let now = Instant::now();
    let clock = Cell::new(now);
    let mut output = GpuOutput::new(now);
    let mut eof = false;
    let mut status = None;
    let mut reader = AdvancingEof {
        clock: &clock,
        completed: output.deadline,
    };

    let result = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut reader,
        || panic!("expired read must not be followed by child polling"),
        || clock.get(),
    );

    assert_eq!(result, Err("nvidia-smi collection deadline exceeded"));
    assert!(!eof);
    assert!(status.is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_rejects_late_success_and_remembers_that_child_was_reaped() {
    let now = Instant::now();
    let clock = Cell::new(now);
    let mut output = GpuOutput::new(now);
    let deadline = output.deadline;
    let mut eof = true;
    let mut status = None;
    let mut polls = 0;

    let result = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut ErrorReader(io::ErrorKind::Other),
        || {
            polls += 1;
            clock.set(deadline);
            Ok(Some(ExitStatus::from_raw(0)))
        },
        || clock.get(),
    );

    assert_eq!(result, Err("nvidia-smi collection deadline exceeded"));
    assert_eq!(polls, 1);
    assert!(status.is_some_and(|status| status.success()));
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_rejects_pending_child_when_try_wait_crosses_deadline() {
    let now = Instant::now();
    let clock = Cell::new(now);
    let mut output = GpuOutput::new(now);
    let deadline = output.deadline;
    let mut eof = false;
    let mut status = None;

    let result = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut ErrorReader(io::ErrorKind::WouldBlock),
        || {
            clock.set(deadline + Duration::from_nanos(1));
            Ok(None)
        },
        || clock.get(),
    );

    assert_eq!(result, Err("nvidia-smi collection deadline exceeded"));
    assert!(status.is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_accepts_success_just_before_deadline_and_retains_original_budget() {
    let now = Instant::now();
    let clock = Cell::new(now);
    let mut output = GpuOutput::new(now);
    let deadline = output.deadline;
    let mut eof = false;
    let mut status = None;
    let mut reader = AdvancingEof {
        clock: &clock,
        completed: deadline - Duration::from_nanos(2),
    };

    let result = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut reader,
        || {
            clock.set(deadline - Duration::from_nanos(1));
            Ok(Some(ExitStatus::from_raw(0)))
        },
        || clock.get(),
    );

    assert_eq!(result, Ok(true));
    assert_eq!(output.deadline, deadline);
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_rechecks_clock_even_when_output_and_exit_status_are_cached() {
    let now = Instant::now();
    let mut output = GpuOutput::new(now);
    let deadline = output.deadline;
    let mut eof = true;
    let mut status = Some(ExitStatus::from_raw(0));

    let result = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut ErrorReader(io::ErrorKind::Other),
        || panic!("already reaped child must not be polled"),
        || deadline,
    );

    assert_eq!(result, Err("nvidia-smi collection deadline exceeded"));
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_preserves_child_failure_messages_before_deadline() {
    let now = Instant::now();
    for (waited, message) in [
        (
            Ok(Some(ExitStatus::from_raw(256))),
            "nvidia-smi child exited unsuccessfully",
        ),
        (
            Err(io::Error::other("mock wait failure")),
            "cannot poll nvidia-smi child",
        ),
    ] {
        let mut output = GpuOutput::new(now);
        let mut eof = true;
        let mut status = None;
        let mut waited = Some(waited);

        let result = poll_gpu(
            &mut output,
            &mut eof,
            &mut status,
            &mut ErrorReader(io::ErrorKind::Other),
            || waited.take().expect("at most one child poll per tick"),
            || now,
        );

        assert_eq!(result, Err(message));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_poll_waits_for_eof_after_reaping_without_polling_child_again() {
    let now = Instant::now();
    let mut output = GpuOutput::new(now);
    let mut eof = false;
    let mut status = None;
    let mut input = std::io::Cursor::new(b"0, 25, 1, 2, 50\n");

    let first = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut input,
        || Ok(Some(ExitStatus::from_raw(0))),
        || now,
    );
    let second = poll_gpu(
        &mut output,
        &mut eof,
        &mut status,
        &mut input,
        || panic!("reaped child must not be polled again"),
        || now,
    );

    assert_eq!(first, Ok(false));
    assert_eq!(second, Ok(true));
    assert_eq!(output.bytes(), b"0, 25, 1, 2, 50\n");
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_deadline_guard_skips_expired_operations() {
    let deadline = Instant::now();
    let result = with_gpu_deadline::<()>(
        deadline,
        || deadline,
        || panic!("expired budget must not start an operation"),
    );
    assert_eq!(result, Err("nvidia-smi collection deadline exceeded"));
}

#[cfg(target_os = "linux")]
struct DropCounter<'a>(&'a Cell<usize>);

#[cfg(target_os = "linux")]
impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_deadline_guard_drops_owned_result_when_start_finishes_late() {
    let now = Instant::now();
    let deadline = now + GPU_DEADLINE;
    let clock = Cell::new(now);
    let drops = Cell::new(0);

    let result = with_gpu_deadline(
        deadline,
        || clock.get(),
        || {
            clock.set(deadline);
            Ok::<_, io::Error>(DropCounter(&drops))
        },
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("late operation result must be discarded"),
    };

    assert_eq!(error, "nvidia-smi collection deadline exceeded");
    assert_eq!(drops.get(), 1);
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_deadline_guard_preserves_start_error_before_expiry() {
    let now = Instant::now();
    let result = with_gpu_deadline(
        now + GPU_DEADLINE,
        || now,
        || {
            Err::<(), _>(io::Error::new(
                io::ErrorKind::NotFound,
                "mock missing nvidia-smi",
            ))
        },
    );

    let error = result.unwrap().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
    assert_eq!(error.to_string(), "mock missing nvidia-smi");
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_completion_discards_samples_if_parsing_crosses_collection_deadline() {
    let now = Instant::now();
    let deadline = now + GPU_DEADLINE;
    let checks = Cell::new(0);
    let mut collector = ResourceCollector::new();
    collector.update_gpu(parse_gpu(b"0, 10, 1, 2, 50\n"), "0");

    collector.finish_gpu(b"0, 25, 1, 2, 60\n", "0", deadline, || {
        let check = checks.get();
        checks.set(check + 1);
        if check == 0 { now } else { deadline }
    });
    let mut text = String::new();
    collector.append_exposition(&mut text);

    assert!(collector.gpu.iter().all(Option::is_none));
    assert!(text.contains("drysua_gpu_collection_available 0\n"));
    assert!(!text.contains("drysua_gpu_temperature_celsius{"));
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_completion_publishes_valid_samples_within_collection_deadline() {
    let now = Instant::now();
    let mut collector = ResourceCollector::new();

    collector.finish_gpu(b"0, 25, 1, 2, 60\n", "0", now + GPU_DEADLINE, || now);

    assert_eq!(collector.gpu[0].unwrap().utilization, 0.25);
    assert_eq!(collector.gpu[0].unwrap().temperature, 60.0);
}

#[test]
fn cpu_delta_uses_all_first_eight_fields_without_double_counting_guest() {
    let previous =
        parse_cpu(b"cpu 0 0 0 0 0 0 0 0\ncpu0 10 10 10 10 10 10 10 10 999 999\n").unwrap();
    let current = parse_cpu(b"cpu0 20 20 20 20 20 20 20 20 99999 99999\n").unwrap();
    let ratios = cpu_ratios(Some(&previous), &current);
    assert_eq!(ratios[0], Some(0.75));
    assert!(ratios[1..].iter().all(Option::is_none));
}

#[test]
fn first_zero_reset_and_hotplug_cpu_samples_are_omitted() {
    let previous = parse_cpu(b"cpu0 10 0 0 10 0 0 0 0\ncpu1 1 0 0 1 0 0 0 0\n").unwrap();
    assert!(cpu_ratios(None, &previous).iter().all(Option::is_none));
    assert!(
        cpu_ratios(Some(&previous), &previous)
            .iter()
            .all(Option::is_none)
    );
    let reset = parse_cpu(b"cpu0 9 0 0 99 0 0 0 0\ncpu2 10 0 0 10 0 0 0 0\n").unwrap();
    assert!(
        cpu_ratios(Some(&previous), &reset)
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn cpu_iowait_regression_and_counter_sum_overflow_are_omitted() {
    let previous = parse_cpu(b"cpu0 10 0 0 10 10 0 0 0\n").unwrap();
    let current = parse_cpu(b"cpu0 20 0 0 20 9 0 0 0\n").unwrap();
    assert_eq!(cpu_ratios(Some(&previous), &current)[0], None);
    let zero = parse_cpu(b"cpu0 0 0 0 0 0 0 0 0\n").unwrap();
    let overflow = parse_cpu(b"cpu0 18446744073709551615 1 0 0 0 0 0 0\n").unwrap();
    assert_eq!(cpu_ratios(Some(&zero), &overflow)[0], None);
}

#[test]
fn cpu_idle_and_busy_boundaries_are_finite() {
    let zero = parse_cpu(b"cpu0 0 0 0 0 0 0 0 0\ncpu255 0 0 0 0 0 0 0 0\n").unwrap();
    let current = parse_cpu(b"cpu0 0 0 0 1 1 0 0 0\ncpu255 1 1 1 0 0 1 1 1\n").unwrap();
    let ratios = cpu_ratios(Some(&zero), &current);
    assert_eq!(ratios[0], Some(0.0));
    assert_eq!(ratios[255], Some(1.0));
}

#[test]
fn malformed_duplicate_and_out_of_range_cpu_ids_fail_closed() {
    for input in [
        "cpu0 1 2 3\n",
        "cpu0 x 0 0 0 0 0 0 0\n",
        "cpu0 -1 0 0 0 0 0 0 0\n",
        "cpu0 18446744073709551616 0 0 0 0 0 0 0\n",
        "cpu256 1 0 0 0 0 0 0 0\n",
        "cpuX 1 0 0 0 0 0 0 0\n",
        "cpucpu0 1 0 0 0 0 0 0 0\n",
        "cpu0 1 0 0 0 0 0 0 0\ncpu0 2 0 0 0 0 0 0 0\n",
        "intr 1 2 3\n",
    ] {
        assert_eq!(
            parse_cpu(input.as_bytes()).unwrap_err(),
            "invalid /proc/stat CPU sample"
        );
    }
}

#[test]
fn proc_parsers_enforce_byte_limits_and_encoding() {
    assert_eq!(
        parse_cpu(&[b' '; PROC_LIMIT + 1]).unwrap_err(),
        "resource input exceeds 65536 bytes"
    );
    assert_eq!(
        parse_memory(&[b' '; PROC_LIMIT + 1]).unwrap_err(),
        "resource input exceeds 65536 bytes"
    );
    assert_eq!(
        parse_cpu(&[255]).unwrap_err(),
        "resource input is not UTF-8"
    );
    let prefix = "cpu0 0 0 0 0 0 0 0 0\n";
    let exact = format!("{prefix}{}", " ".repeat(PROC_LIMIT - prefix.len()));
    assert!(parse_cpu(exact.as_bytes()).is_ok());
}

#[test]
fn all_resource_parsers_report_encoding_errors_before_parsing_fields() {
    let input = [255];
    assert_eq!(
        parse_cpu(&input).unwrap_err(),
        "resource input is not UTF-8"
    );
    assert_eq!(
        parse_memory(&input).unwrap_err(),
        "resource input is not UTF-8"
    );
    assert_eq!(
        parse_gpu(&input).unwrap_err(),
        "resource input is not UTF-8"
    );
    assert_eq!(
        parse_gpu_ids(&input).unwrap_err(),
        "resource input is not UTF-8"
    );
}

#[test]
fn resource_size_errors_take_precedence_over_encoding_errors() {
    let proc_input = [255; PROC_LIMIT + 1];
    let gpu_input = [255; GPU_OUTPUT_LIMIT + 1];
    assert_eq!(
        parse_cpu(&proc_input).unwrap_err(),
        "resource input exceeds 65536 bytes"
    );
    assert_eq!(
        parse_memory(&proc_input).unwrap_err(),
        "resource input exceeds 65536 bytes"
    );
    assert_eq!(
        parse_gpu(&gpu_input).unwrap_err(),
        "GPU output exceeds 4096 bytes"
    );
    assert_eq!(
        parse_gpu_ids(&gpu_input).unwrap_err(),
        "GPU output exceeds 4096 bytes"
    );
}

#[test]
fn failed_cpu_collection_resets_baseline_and_never_bridges_gaps() {
    let sample = b"cpu0 10 0 0 10 0 0 0 0\n";
    let mut collector = ResourceCollector::new();
    collector.update_cpu(parse_cpu(sample));
    collector.update_cpu(Err("unavailable"));
    collector.update_cpu(parse_cpu(b"cpu0 20 0 0 20 0 0 0 0\n"));
    assert!(collector.cpu.iter().all(Option::is_none));
    assert!(collector.previous_cpu.is_some());
}

#[test]
fn removed_then_returning_cpu_requires_a_new_baseline() {
    let mut collector = ResourceCollector::new();
    collector.update_cpu(parse_cpu(b"cpu0 1 0 0 1 0 0 0 0\ncpu1 1 0 0 1 0 0 0 0\n"));
    collector.update_cpu(parse_cpu(b"cpu0 2 0 0 2 0 0 0 0\n"));
    collector.update_cpu(parse_cpu(b"cpu0 3 0 0 3 0 0 0 0\ncpu1 9 0 0 9 0 0 0 0\n"));
    assert_eq!(collector.cpu[0], Some(0.5));
    assert_eq!(collector.cpu[1], None);
}

#[test]
fn memory_parser_converts_kibibytes_and_accepts_zero_available() {
    assert_eq!(
        parse_memory(b"MemTotal: 8 kB\nIgnored: 4 kB\nMemAvailable: 3 kB\n").unwrap(),
        MemorySample {
            available: 3072,
            total: 8192,
        }
    );
    assert_eq!(
        parse_memory(b"MemAvailable: 0 kB\nMemTotal: 1 kB\n")
            .unwrap()
            .available,
        0
    );
}

#[test]
fn memory_parser_rejects_missing_duplicate_units_overflow_and_inversion() {
    for input in [
        "MemTotal: 8 kB\n",
        "MemTotal: 8 kB\nMemAvailable: 9 kB\n",
        "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 8 MB\nMemAvailable: 1 kB\n",
        "MemTotal: 8 kB extra\nMemAvailable: 1 kB\n",
        "MemTotal: 8 kB\nMemTotal: 8 kB\nMemAvailable: 1 kB\n",
        "MemTotal: 8 kB\nMemAvailable: 1 kB\nMemAvailable: 1 kB\n",
        "MemTotal: 18446744073709551615 kB\nMemAvailable: 1 kB\n",
        "MemTotal: -8 kB\nMemAvailable: 1 kB\n",
    ] {
        assert_eq!(
            parse_memory(input.as_bytes()).unwrap_err(),
            "invalid /proc/meminfo memory sample"
        );
    }
}

#[test]
fn gpu_parser_uses_reported_ids_and_converts_percent_and_mib() {
    let samples = parse_gpu(b"1, 25, 1024, 2048, 55\n0, 100, 0, 4096, 0\n").unwrap();
    assert_eq!(
        samples[1].unwrap(),
        GpuSample {
            utilization: 0.25,
            used: 1_073_741_824,
            total: 2_147_483_648,
            temperature: 55.0,
        }
    );
    assert_eq!(samples[0].unwrap().utilization, 1.0);
    assert_eq!(samples[0].unwrap().used, 0);
    assert!(samples[2..].iter().all(Option::is_none));
}

#[test]
fn gpu_parser_rejects_invalid_ranges_nonfinite_values_and_duplicates() {
    for input in [
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
    ] {
        assert_eq!(
            parse_gpu(input.as_bytes()).unwrap_err(),
            "invalid nvidia-smi GPU sample"
        );
    }
}

#[test]
fn gpu_discovery_is_bounded_and_rejects_duplicate_and_noncanonical_ids() {
    assert_eq!(parse_gpu_ids(b"1\n0\n").unwrap(), "0,1");
    assert_eq!(
        parse_gpu_ids(b"0\n1\n2\n3\n4\n5\n6\n7\n").unwrap(),
        "0,1,2,3,4,5,6,7"
    );
    for input in ["", "0\n0\n", "8\n", "00\n", "-1\n", "x\n"] {
        assert_eq!(
            parse_gpu_ids(input.as_bytes()).unwrap_err(),
            "invalid nvidia-smi GPU indices"
        );
    }
    assert_eq!(
        parse_gpu(&[b' '; GPU_OUTPUT_LIMIT + 1]).unwrap_err(),
        "GPU output exceeds 4096 bytes"
    );
    assert_eq!(
        parse_gpu_ids(&[b' '; GPU_OUTPUT_LIMIT + 1]).unwrap_err(),
        "GPU output exceeds 4096 bytes"
    );
}

#[test]
fn eight_gpu_samples_are_accepted_but_a_ninth_row_is_rejected() {
    let mut input = String::new();
    for index in 0..GPU_LIMIT {
        std::fmt::Write::write_fmt(&mut input, format_args!("{index}, 100, 1, 2, 200\n")).unwrap();
    }
    assert!(
        parse_gpu(input.as_bytes())
            .unwrap()
            .iter()
            .all(Option::is_some)
    );
    input.push_str("0, 100, 1, 2, 200\n");
    assert_eq!(
        parse_gpu(input.as_bytes()).unwrap_err(),
        "invalid nvidia-smi GPU sample"
    );
}

#[test]
fn unavailable_resources_expose_availability_without_fabricated_samples() {
    let collector = ResourceCollector::new();
    let mut text = String::new();
    collector.append_exposition(&mut text);
    for name in [
        "drysua_host_cpu_collection_available",
        "drysua_host_memory_collection_available",
        "drysua_gpu_collection_available",
    ] {
        assert!(text.contains(&format!("# TYPE {name} gauge\n{name} 0\n")));
    }
    assert!(!text.contains("drysua_host_cpu_utilization_ratio{"));
    assert!(!text.contains("drysua_host_memory_available_bytes "));
    assert!(!text.contains("drysua_gpu_utilization_ratio{"));
}

#[test]
fn populated_exposition_has_one_declaration_per_family_and_bounded_size() {
    let mut collector = ResourceCollector::new();
    collector.cpu = [Some(0.5); CPU_LIMIT];
    collector.memory = Some(MemorySample {
        available: 1024,
        total: 2048,
    });
    collector.gpu = [Some(GpuSample {
        utilization: 1.0,
        used: 1024,
        total: 2048,
        temperature: 50.0,
    }); GPU_LIMIT];
    let mut text = String::new();
    collector.append_exposition(&mut text);
    assert!(text.len() <= MAX_EXPOSITION_BYTES);
    for name in [
        "drysua_host_cpu_utilization_ratio",
        "drysua_host_memory_available_bytes",
        "drysua_host_memory_total_bytes",
        "drysua_gpu_utilization_ratio",
        "drysua_gpu_memory_used_bytes",
        "drysua_gpu_memory_total_bytes",
        "drysua_gpu_temperature_celsius",
    ] {
        assert_eq!(text.matches(&format!("# HELP {name} ")).count(), 1);
        assert_eq!(text.matches(&format!("# TYPE {name} gauge\n")).count(), 1);
    }
    assert!(text.contains("drysua_host_cpu_utilization_ratio{cpu=\"255\"} 0.5\n"));
    assert!(text.contains("drysua_gpu_temperature_celsius{gpu=\"7\"} 50\n"));
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_output_reader_is_bounded_and_deadline_is_absolute() {
    use std::io::Cursor;
    let now = Instant::now();
    let mut output = GpuOutput::new(now);
    assert_eq!(
        output.read(&mut Cursor::new(vec![b'x'; GPU_OUTPUT_LIMIT + 1]), now),
        Err("GPU output exceeds 4096 bytes")
    );
    let mut output = GpuOutput::new(now);
    assert_eq!(output.read(&mut Cursor::new(b"0\n"), now), Ok(false));
    assert_eq!(
        output.read(&mut Cursor::new(b""), now + GPU_DEADLINE),
        Err("nvidia-smi collection deadline exceeded")
    );
}

#[cfg(target_os = "linux")]
#[test]
fn exact_gpu_output_limit_requires_eof_and_accepts_no_extra_byte() {
    use std::io::Cursor;
    let now = Instant::now();
    let mut output = GpuOutput::new(now);
    assert_eq!(
        output.read(&mut Cursor::new(vec![b'x'; GPU_OUTPUT_LIMIT]), now),
        Ok(false)
    );
    assert_eq!(output.read(&mut Cursor::new(b""), now), Ok(true));
    assert_eq!(output.bytes().len(), GPU_OUTPUT_LIMIT);
}

#[cfg(target_os = "linux")]
struct ErrorReader(io::ErrorKind);

#[cfg(target_os = "linux")]
impl Read for ErrorReader {
    fn read(&mut self, _output: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(self.0, "mock resource read failure"))
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_transient_read_errors_do_not_reset_deadline_or_retry_inline() {
    let now = Instant::now();
    let mut output = GpuOutput::new(now);
    for kind in [io::ErrorKind::WouldBlock, io::ErrorKind::Interrupted] {
        assert_eq!(output.read(&mut ErrorReader(kind), now), Ok(false));
        assert_eq!(output.deadline, now + GPU_DEADLINE);
    }
    assert_eq!(
        output.read(&mut ErrorReader(io::ErrorKind::ConnectionReset), now),
        Err("cannot read nvidia-smi output")
    );
}

#[cfg(target_os = "linux")]
#[test]
fn proc_read_failure_preserves_error_instead_of_fabricating_empty_sample() {
    let mut buffer = [0; PROC_LIMIT + 1];
    let error = read_bounded(
        &mut ErrorReader(io::ErrorKind::PermissionDenied),
        &mut buffer,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "mock resource read failure");
}

#[cfg(target_os = "linux")]
#[test]
fn proc_reader_accepts_exact_limit_and_rejects_one_extra_byte() {
    use std::io::Cursor;
    let mut buffer = [0; PROC_LIMIT + 1];
    let length = read_bounded(&mut Cursor::new(vec![b'x'; PROC_LIMIT]), &mut buffer).unwrap();
    assert_eq!(length, PROC_LIMIT);
    let error =
        read_bounded(&mut Cursor::new(vec![b'x'; PROC_LIMIT + 1]), &mut buffer).unwrap_err();
    assert_eq!(error.to_string(), "resource input exceeds 65536 bytes");
}

#[test]
fn gpu_samples_must_match_discovered_indices_exactly() {
    let samples = parse_gpu(b"0, 1, 1, 2, 3\n").unwrap();
    assert!(gpu_matches_ids(&samples, "0"));
    assert!(!gpu_matches_ids(&samples, "0,1"));
    assert!(!gpu_matches_ids(&samples, "1"));
}

#[test]
fn failed_or_incomplete_gpu_update_removes_previous_samples() {
    let mut collector = ResourceCollector::new();
    collector.update_gpu(parse_gpu(b"0, 25, 1, 2, 50\n"), "0");
    assert!(collector.gpu[0].is_some());
    collector.update_gpu(Err("GPU query failed"), "0");
    assert!(collector.gpu.iter().all(Option::is_none));
    collector.update_gpu(parse_gpu(b"0, 25, 1, 2, 50\n"), "0,1");
    assert!(collector.gpu.iter().all(Option::is_none));
    let mut text = String::new();
    collector.append_exposition(&mut text);
    assert!(text.contains("drysua_gpu_collection_available 0\n"));
    assert!(!text.contains("drysua_gpu_temperature_celsius{"));
}

#[test]
fn exposition_bound_includes_maximum_integer_and_smallest_float_widths() {
    let mut collector = ResourceCollector::new();
    let smallest_cpu = cpu_ratio([0; 8], [1, 0, 0, u64::MAX - 1, 0, 0, 0, 0]);
    collector.cpu = [smallest_cpu; CPU_LIMIT];
    collector.memory = Some(MemorySample {
        available: u64::MAX,
        total: u64::MAX,
    });
    collector.gpu = [Some(GpuSample {
        utilization: f64::from_bits(1),
        used: u64::MAX,
        total: u64::MAX,
        temperature: f64::from_bits(1),
    }); GPU_LIMIT];
    let mut text = String::new();
    collector.append_exposition(&mut text);
    assert!(text.len() <= MAX_EXPOSITION_BYTES);
    assert!(!text.contains("NaN"));
    assert!(!text.contains("inf"));
}
