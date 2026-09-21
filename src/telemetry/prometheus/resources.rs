use std::fmt::Write as _;
use std::io;
use std::time::Instant;

#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::fd::OwnedFd;
#[cfg(target_os = "linux")]
use std::os::unix::net::UnixStream;
#[cfg(target_os = "linux")]
use std::process::{Child, Command, ExitStatus, Stdio};
#[cfg(target_os = "linux")]
use std::time::Duration;

#[cfg(test)]
#[path = "resources_tests.rs"]
mod tests;

const CPU_LIMIT: usize = 256;
const GPU_LIMIT: usize = 8;
#[cfg(any(target_os = "linux", test))]
const GPU_OUTPUT_LIMIT: usize = 4096;
#[cfg(any(target_os = "linux", test))]
const PROC_LIMIT: usize = 64 * 1024;
#[cfg(target_os = "linux")]
const GPU_DEADLINE: Duration = Duration::from_secs(1);
const _: () = assert!(CPU_LIMIT <= 256);
const _: () = assert!(GPU_LIMIT <= 8);
const _: () = assert!(MAX_EXPOSITION_BYTES <= 48 * 1024);

#[cfg(any(target_os = "linux", test))]
type CpuSnapshot = [Option<[u64; 8]>; CPU_LIMIT];
type GpuSnapshot = [Option<GpuSample>; GPU_LIMIT];

pub(super) const MAX_EXPOSITION_BYTES: usize = 48 * 1024;

/// Collection is driven only by the server's five-second schedule, never a scrape.
/// Unsupported platforms expose availability zero and no resource samples.
pub(super) struct ResourceCollector {
    #[cfg(any(target_os = "linux", test))]
    previous_cpu: Option<CpuSnapshot>,
    cpu: [Option<f64>; CPU_LIMIT],
    memory: Option<MemorySample>,
    gpu: GpuSnapshot,
    #[cfg(target_os = "linux")]
    gpu_work: Option<GpuWork>,
}

impl ResourceCollector {
    pub(super) fn new() -> Self {
        Self {
            #[cfg(any(target_os = "linux", test))]
            previous_cpu: None,
            cpu: [None; CPU_LIMIT],
            memory: None,
            gpu: [None; GPU_LIMIT],
            #[cfg(target_os = "linux")]
            gpu_work: None,
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) fn collect(&mut self, now: Instant) {
        assert!(self.gpu_work.is_none());
        self.update_cpu(
            read_proc("/proc/stat", parse_cpu).map_err(|_| "CPU collection unavailable"),
        );
        self.memory = read_proc("/proc/meminfo", parse_memory).ok();
        self.gpu = [None; GPU_LIMIT];
        // Discover actual indices before passing -i: absent indices make a query
        // for a hard-coded 0..7 fail even on a healthy two-GPU machine.
        self.gpu_work = GpuProcess::start(None, now + GPU_DEADLINE)
            .ok()
            .map(|process| GpuWork {
                process,
                indices: None,
            });
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn collect(&mut self, _now: Instant) {}

    #[cfg(target_os = "linux")]
    pub(super) fn poll(&mut self, now: Instant) -> bool {
        let Some(mut work) = self.gpu_work.take() else {
            return true;
        };
        match work.process.poll(now) {
            Ok(false) => {
                self.gpu_work = Some(work);
                return false;
            }
            Err(_) => return true,
            Ok(true) => {}
        }
        let deadline = work.process.output.deadline;
        if let Some(indices) = work.indices {
            self.finish_gpu(
                work.process.output.bytes(),
                &indices,
                deadline,
                Instant::now,
            );
            return true;
        }
        let Ok(Ok(indices)) = with_gpu_deadline(deadline, Instant::now, || {
            parse_gpu_ids(work.process.output.bytes())
        }) else {
            return true;
        };
        let Ok(process) = GpuProcess::start(Some(&indices), deadline) else {
            return true;
        };
        self.gpu_work = Some(GpuWork {
            process,
            indices: Some(indices),
        });
        false
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn poll(&mut self, _now: Instant) -> bool {
        true
    }

    pub(super) fn append_exposition(&self, output: &mut String) {
        let start = output.len();
        availability(
            output,
            "drysua_host_cpu_collection_available",
            self.cpu.iter().any(Option::is_some),
        );
        availability(
            output,
            "drysua_host_memory_collection_available",
            self.memory.is_some(),
        );
        availability(
            output,
            "drysua_gpu_collection_available",
            self.gpu.iter().any(Option::is_some),
        );
        self.append_cpu(output);
        self.append_memory(output);
        self.append_gpu(output);
        assert!(output.len() >= start);
        assert!(output.len() - start <= MAX_EXPOSITION_BYTES);
    }

    pub(super) fn shutdown(&mut self) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        if let Some(mut work) = self.gpu_work.take() {
            return work.process.stop();
        }
        Ok(())
    }

    #[cfg(any(target_os = "linux", test))]
    fn update_cpu(&mut self, sample: Result<CpuSnapshot, &'static str>) {
        match sample {
            Ok(current) => {
                self.cpu = cpu_ratios(self.previous_cpu.as_ref(), &current);
                self.previous_cpu = Some(current);
            }
            Err(_) => {
                self.cpu = [None; CPU_LIMIT];
                self.previous_cpu = None;
            }
        }
    }

    #[cfg(any(target_os = "linux", test))]
    fn update_gpu(&mut self, samples: Result<GpuSnapshot, &'static str>, indices: &str) {
        self.gpu = samples
            .ok()
            .filter(|samples| gpu_matches_ids(samples, indices))
            .unwrap_or([None; GPU_LIMIT]);
    }

    #[cfg(target_os = "linux")]
    fn finish_gpu(
        &mut self,
        input: &[u8],
        indices: &str,
        deadline: Instant,
        clock: impl Fn() -> Instant,
    ) {
        let result = with_gpu_deadline(deadline, clock, || {
            self.update_gpu(parse_gpu(input), indices);
        });
        if result.is_err() {
            self.gpu = [None; GPU_LIMIT];
        }
    }

    fn append_cpu(&self, output: &mut String) {
        if !self.cpu.iter().any(Option::is_some) {
            return;
        }
        family(
            output,
            "drysua_host_cpu_utilization_ratio",
            "Non-idle fraction of CPU counter deltas.",
        );
        for (index, value) in self.cpu.iter().enumerate() {
            if let Some(value) = value {
                assert!(value.is_finite());
                assert!((0.0..=1.0).contains(value));
                writeln!(
                    output,
                    "drysua_host_cpu_utilization_ratio{{cpu=\"{index}\"}} {value}"
                )
                .expect("writing to String cannot fail");
            }
        }
    }

    fn append_memory(&self, output: &mut String) {
        let Some(memory) = self.memory else {
            return;
        };
        assert!(memory.total > 0);
        assert!(memory.available <= memory.total);
        scalar(
            output,
            "drysua_host_memory_available_bytes",
            "Available host memory in bytes.",
            memory.available,
        );
        scalar(
            output,
            "drysua_host_memory_total_bytes",
            "Total host memory in bytes.",
            memory.total,
        );
    }

    fn append_gpu(&self, output: &mut String) {
        if !self.gpu.iter().any(Option::is_some) {
            return;
        }
        family(
            output,
            "drysua_gpu_utilization_ratio",
            "GPU utilization fraction.",
        );
        family(
            output,
            "drysua_gpu_memory_used_bytes",
            "Used GPU memory in bytes.",
        );
        family(
            output,
            "drysua_gpu_memory_total_bytes",
            "Total GPU memory in bytes.",
        );
        family(
            output,
            "drysua_gpu_temperature_celsius",
            "GPU temperature in Celsius.",
        );
        for (index, sample) in self.gpu.iter().enumerate() {
            if let Some(sample) = sample {
                sample.append(output, index);
            }
        }
    }
}

fn family(output: &mut String, name: &str, help: &str) {
    writeln!(output, "# HELP {name} {help}\n# TYPE {name} gauge")
        .expect("writing to String cannot fail");
}

fn availability(output: &mut String, name: &str, available: bool) {
    scalar(
        output,
        name,
        "Whether the latest resource collection has usable samples.",
        u64::from(available),
    );
}

fn scalar(output: &mut String, name: &str, help: &str, value: u64) {
    family(output, name, help);
    writeln!(output, "{name} {value}").expect("writing to String cannot fail");
}

#[cfg(any(target_os = "linux", test))]
fn input_text<'a>(
    input: &'a [u8],
    limit: usize,
    overflow: &'static str,
) -> Result<&'a str, &'static str> {
    if input.len() > limit {
        return Err(overflow);
    }
    std::str::from_utf8(input).map_err(|_| "resource input is not UTF-8")
}

#[cfg(any(target_os = "linux", test))]
fn parse_cpu(input: &[u8]) -> Result<CpuSnapshot, &'static str> {
    let text = input_text(input, PROC_LIMIT, "resource input exceeds 65536 bytes")?;
    let mut snapshot = [None; CPU_LIMIT];
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        if name == "cpu" || !name.starts_with("cpu") {
            continue;
        }
        let index = decimal(
            name.strip_prefix("cpu")
                .ok_or("invalid /proc/stat CPU sample")?,
        )
        .and_then(|value| usize::try_from(value).ok())
        .filter(|index| *index < CPU_LIMIT)
        .ok_or("invalid /proc/stat CPU sample")?;
        if snapshot[index].is_some() {
            return Err("invalid /proc/stat CPU sample");
        }
        let mut counters = [0; 8];
        for counter in &mut counters {
            *counter = fields
                .next()
                .and_then(decimal)
                .ok_or("invalid /proc/stat CPU sample")?;
        }
        // Guest and guest_nice are already included in user and nice counters.
        snapshot[index] = Some(counters);
    }
    if snapshot.iter().all(Option::is_none) {
        return Err("invalid /proc/stat CPU sample");
    }
    Ok(snapshot)
}

#[cfg(any(target_os = "linux", test))]
fn cpu_ratios(previous: Option<&CpuSnapshot>, current: &CpuSnapshot) -> [Option<f64>; CPU_LIMIT] {
    let Some(previous) = previous else {
        return [None; CPU_LIMIT];
    };
    std::array::from_fn(|index| cpu_ratio(previous[index]?, current[index]?))
}

#[cfg(any(target_os = "linux", test))]
#[allow(
    clippy::float_arithmetic,
    reason = "Bounded nonzero u64 counter deltas produce a finite ratio in [0, 1]."
)]
fn cpu_ratio(previous: [u64; 8], current: [u64; 8]) -> Option<f64> {
    let mut delta = [0_u64; 8];
    for (index, value) in delta.iter_mut().enumerate() {
        *value = current[index].checked_sub(previous[index])?;
    }
    let total = delta
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))?;
    let idle = delta[3].checked_add(delta[4])?;
    if total == 0 {
        return None;
    }
    let busy = total.checked_sub(idle)?;
    let ratio = busy as f64 / total as f64;
    assert!(ratio.is_finite());
    assert!((0.0..=1.0).contains(&ratio));
    Some(ratio)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemorySample {
    available: u64,
    total: u64,
}

#[cfg(any(target_os = "linux", test))]
fn parse_memory(input: &[u8]) -> Result<MemorySample, &'static str> {
    let text = input_text(input, PROC_LIMIT, "resource input exceeds 65536 bytes")?;
    let error = "invalid /proc/meminfo memory sample";
    let mut available = None;
    let mut total = None;
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        let slot = match fields.next() {
            Some("MemAvailable:") => &mut available,
            Some("MemTotal:") => &mut total,
            _ => continue,
        };
        if slot.is_some() {
            return Err(error);
        }
        *slot = Some(
            fields
                .next()
                .and_then(decimal)
                .and_then(|value| value.checked_mul(1024))
                .ok_or(error)?,
        );
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(error);
        }
    }
    let sample = MemorySample {
        available: available.ok_or(error)?,
        total: total.ok_or(error)?,
    };
    if sample.total == 0 || sample.available > sample.total {
        return Err(error);
    }
    Ok(sample)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GpuSample {
    utilization: f64,
    used: u64,
    total: u64,
    temperature: f64,
}

impl GpuSample {
    fn append(self, output: &mut String, index: usize) {
        assert!(index < GPU_LIMIT);
        assert!(self.utilization.is_finite());
        assert!((0.0..=1.0).contains(&self.utilization));
        assert!(self.temperature.is_finite());
        assert!((0.0..=200.0).contains(&self.temperature));
        assert!(self.used <= self.total);
        assert!(self.total > 0);
        writeln!(
            output,
            "drysua_gpu_utilization_ratio{{gpu=\"{index}\"}} {}",
            self.utilization
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "drysua_gpu_memory_used_bytes{{gpu=\"{index}\"}} {}",
            self.used
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "drysua_gpu_memory_total_bytes{{gpu=\"{index}\"}} {}",
            self.total
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "drysua_gpu_temperature_celsius{{gpu=\"{index}\"}} {}",
            self.temperature
        )
        .expect("writing to String cannot fail");
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_gpu(input: &[u8]) -> Result<GpuSnapshot, &'static str> {
    let text = input_text(input, GPU_OUTPUT_LIMIT, "GPU output exceeds 4096 bytes")?;
    let error = "invalid nvidia-smi GPU sample";
    let mut samples = [None; GPU_LIMIT];
    for (count, line) in text.lines().enumerate() {
        if count >= GPU_LIMIT {
            return Err(error);
        }
        let mut fields = line.split(',').map(str::trim);
        let index = fields.next().and_then(gpu_index).ok_or(error)?;
        if samples[index].is_some() {
            return Err(error);
        }
        let sample = parse_gpu_fields(&mut fields).ok_or(error)?;
        if fields.next().is_some() {
            return Err(error);
        }
        samples[index] = Some(sample);
    }
    if samples.iter().all(Option::is_none) {
        return Err(error);
    }
    Ok(samples)
}

#[cfg(any(target_os = "linux", test))]
#[allow(
    clippy::float_arithmetic,
    reason = "Finite utilization is checked in [0, 100] before conversion to a ratio."
)]
fn parse_gpu_fields<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<GpuSample> {
    let utilization = bounded_float(fields.next()?, 100.0)? / 100.0;
    let used = decimal(fields.next()?)?.checked_mul(1024 * 1024)?;
    let total = decimal(fields.next()?)?.checked_mul(1024 * 1024)?;
    let temperature = bounded_float(fields.next()?, 200.0)?;
    if total == 0 || used > total {
        return None;
    }
    assert!(utilization.is_finite());
    assert!((0.0..=1.0).contains(&utilization));
    Some(GpuSample {
        utilization,
        used,
        total,
        temperature,
    })
}

#[cfg(any(target_os = "linux", test))]
fn parse_gpu_ids(input: &[u8]) -> Result<String, &'static str> {
    let text = input_text(input, GPU_OUTPUT_LIMIT, "GPU output exceeds 4096 bytes")?;
    let error = "invalid nvidia-smi GPU indices";
    let mut present = [false; GPU_LIMIT];
    for (count, line) in text.lines().enumerate() {
        if count >= GPU_LIMIT {
            return Err(error);
        }
        let index = gpu_index(line.trim()).ok_or(error)?;
        if present[index] {
            return Err(error);
        }
        present[index] = true;
    }
    let indices = gpu_ids(present);
    if indices.is_empty() {
        return Err(error);
    }
    Ok(indices)
}

#[cfg(any(target_os = "linux", test))]
fn gpu_matches_ids(samples: &GpuSnapshot, expected: &str) -> bool {
    gpu_ids(std::array::from_fn(|index| samples[index].is_some())) == expected
}

#[cfg(any(target_os = "linux", test))]
fn gpu_ids(present: [bool; GPU_LIMIT]) -> String {
    let mut indices = String::with_capacity(GPU_LIMIT * 2);
    for (index, present) in present.into_iter().enumerate() {
        if present {
            if !indices.is_empty() {
                indices.push(',');
            }
            write!(indices, "{index}").expect("writing to String cannot fail");
        }
    }
    assert!(indices.len() < GPU_LIMIT * 2);
    indices
}

#[cfg(any(target_os = "linux", test))]
fn gpu_index(text: &str) -> Option<usize> {
    // The configured limit makes a canonical index exactly one ASCII digit.
    if text.len() != 1 {
        return None;
    }
    decimal(text)
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| *index < GPU_LIMIT)
}

#[cfg(any(target_os = "linux", test))]
fn decimal(text: &str) -> Option<u64> {
    if text.is_empty() || text.len() > 20 || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

#[cfg(any(target_os = "linux", test))]
fn bounded_float(text: &str, maximum: f64) -> Option<f64> {
    if text.is_empty() || text.len() > 32 {
        return None;
    }
    let value: f64 = text.parse().ok()?;
    (value.is_finite() && (0.0..=maximum).contains(&value)).then_some(value)
}

#[cfg(target_os = "linux")]
fn read_proc<T>(path: &str, parse: impl FnOnce(&[u8]) -> Result<T, &'static str>) -> io::Result<T> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = [0; PROC_LIMIT + 1];
    let length = read_bounded(&mut file, &mut buffer)?;
    parse(&buffer[..length]).map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
}

#[cfg(target_os = "linux")]
fn read_bounded(input: &mut impl Read, buffer: &mut [u8; PROC_LIMIT + 1]) -> io::Result<usize> {
    let mut length = 0;
    // Even a reader returning one byte at a time gets a fixed call budget.
    for _ in 0..=PROC_LIMIT {
        let count = input.read(&mut buffer[length..])?;
        if count == 0 {
            return Ok(length);
        }
        length += count;
        if length > PROC_LIMIT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "resource input exceeds 65536 bytes",
            ));
        }
    }
    Err(io::Error::other("resource read call budget exhausted"))
}

#[cfg(target_os = "linux")]
fn with_gpu_deadline<T>(
    deadline: Instant,
    clock: impl Fn() -> Instant,
    operation: impl FnOnce() -> T,
) -> Result<T, &'static str> {
    if clock() >= deadline {
        return Err("nvidia-smi collection deadline exceeded");
    }
    // Retain ownership until the post-check so a late spawned child is dropped
    // and reaped, rather than installed as an apparently successful result.
    let result = operation();
    if clock() >= deadline {
        return Err("nvidia-smi collection deadline exceeded");
    }
    Ok(result)
}

#[cfg(target_os = "linux")]
fn poll_gpu(
    output: &mut GpuOutput,
    eof: &mut bool,
    status: &mut Option<ExitStatus>,
    input: &mut impl Read,
    try_wait: impl FnOnce() -> io::Result<Option<ExitStatus>>,
    clock: impl Fn() -> Instant,
) -> Result<bool, &'static str> {
    let deadline = output.deadline;
    with_gpu_deadline(deadline, &clock, || {
        if !*eof {
            *eof = with_gpu_deadline(deadline, &clock, || output.read(input, clock()))??;
        }
        if status.is_none() {
            // Preserve a reaped status even if the outer post-check expires;
            // cleanup must not signal a PID that the kernel may have reused.
            *status = try_wait().map_err(|_| "cannot poll nvidia-smi child")?;
        }
        if status.as_ref().is_some_and(|status| !status.success()) {
            return Err("nvidia-smi child exited unsuccessfully");
        }
        Ok(*eof && status.is_some())
    })?
}

#[cfg(target_os = "linux")]
struct GpuWork {
    process: GpuProcess,
    indices: Option<String>,
}

#[cfg(target_os = "linux")]
struct GpuProcess {
    child: Child,
    socket: UnixStream,
    output: GpuOutput,
    status: Option<ExitStatus>,
    eof: bool,
}

#[cfg(target_os = "linux")]
impl GpuProcess {
    fn start(indices: Option<&str>, deadline: Instant) -> io::Result<Self> {
        with_gpu_deadline(deadline, Instant::now, || Self::spawn(indices, deadline))
            .map_err(|message| io::Error::new(io::ErrorKind::TimedOut, message))?
    }

    fn spawn(indices: Option<&str>, deadline: Instant) -> io::Result<Self> {
        let (socket, writer) = UnixStream::pair()?;
        socket.set_nonblocking(true)?;
        let query = if indices.is_some() {
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu"
        } else {
            "--query-gpu=index"
        };
        let mut command = Command::new("nvidia-smi");
        command.args([query, "--format=csv,noheader,nounits"]);
        if let Some(indices) = indices {
            command.args(["-i", indices]);
        }
        let child = command
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .stdout(Stdio::from(OwnedFd::from(writer)))
            .spawn()?;
        Ok(Self {
            child,
            socket,
            output: GpuOutput {
                deadline,
                buffer: [0; GPU_OUTPUT_LIMIT + 1],
                length: 0,
            },
            status: None,
            eof: false,
        })
    }

    fn poll(&mut self, now: Instant) -> Result<bool, &'static str> {
        if now >= self.output.deadline {
            return Err("nvidia-smi collection deadline exceeded");
        }
        let child = &mut self.child;
        poll_gpu(
            &mut self.output,
            &mut self.eof,
            &mut self.status,
            &mut self.socket,
            || child.try_wait(),
            Instant::now,
        )
    }

    fn stop(&mut self) -> io::Result<()> {
        if self.status.is_some() {
            return Ok(());
        }
        // Always reap, including after kill errors (the child may already have
        // exited). std offers no deadline for wait under a stalled kernel.
        let killed = self.child.kill();
        let waited = self.child.wait();
        match waited {
            Ok(status) => {
                self.status = Some(status);
                Ok(())
            }
            Err(error) => Err(killed.err().unwrap_or(error)),
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for GpuProcess {
    fn drop(&mut self) {
        // Availability records failed collections; shutdown reports reap errors.
        let _ = self.stop();
    }
}

#[cfg(target_os = "linux")]
struct GpuOutput {
    deadline: Instant,
    buffer: [u8; GPU_OUTPUT_LIMIT + 1],
    length: usize,
}

#[cfg(target_os = "linux")]
impl GpuOutput {
    #[cfg(test)]
    fn new(now: Instant) -> Self {
        Self {
            deadline: now + GPU_DEADLINE,
            buffer: [0; GPU_OUTPUT_LIMIT + 1],
            length: 0,
        }
    }

    fn read(&mut self, input: &mut impl Read, now: Instant) -> Result<bool, &'static str> {
        assert!(self.length <= GPU_OUTPUT_LIMIT);
        if now >= self.deadline {
            return Err("nvidia-smi collection deadline exceeded");
        }
        match input.read(&mut self.buffer[self.length..]) {
            Ok(0) => Ok(true),
            Ok(count) => {
                self.length += count;
                if self.length > GPU_OUTPUT_LIMIT {
                    return Err("GPU output exceeds 4096 bytes");
                }
                Ok(false)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(_) => Err("cannot read nvidia-smi output"),
        }
    }

    fn bytes(&self) -> &[u8] {
        assert!(self.length <= GPU_OUTPUT_LIMIT);
        &self.buffer[..self.length]
    }
}
