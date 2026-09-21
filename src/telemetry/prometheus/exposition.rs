use super::snapshot::{BUCKETS, DurationHistogram, TrainingSnapshot};
use std::fmt::{self, Write};
use std::io;

#[cfg(test)]
#[path = "exposition_tests.rs"]
mod tests;

const MAX_EXPOSITION_BYTES: usize = 48 * 1024;
const OUTCOMES: [&str; 4] = ["win", "loss", "draw", "time_cap"];
const SCOPE_LABELS: [&str; 3] = [
    "scope=\"session_initialization\"",
    "scope=\"checkpoint_capture_save_runtime_export\"",
    "scope=\"resume_runtime_export\"",
];
const STAGE_LABELS: [&str; 5] = [
    "stage=\"rollout_initialization\"",
    "stage=\"collection\"",
    "stage=\"batch_preparation\"",
    "stage=\"optimization\"",
    "stage=\"finalization\"",
];
const _: () = assert!(MAX_EXPOSITION_BYTES <= 48 * 1024);
const _: () = assert!(BUCKETS.len() == 10);

pub(super) fn render(
    snapshot: Option<&TrainingSnapshot>,
    active: bool,
    healthy: bool,
    heartbeat: u64,
    scopes: Option<&[DurationHistogram; 3]>,
) -> io::Result<String> {
    if let Some(snapshot) = snapshot {
        snapshot.validate()?;
    }
    let available = snapshot.is_some() && healthy;
    if available && let Some(scopes) = scopes {
        validate_scopes(scopes)?;
    }
    let mut output = Exposition {
        text: String::with_capacity(MAX_EXPOSITION_BYTES),
    };
    append_status(&mut output, active, available, healthy, heartbeat)
        .and_then(|()| {
            if let Some(snapshot) = snapshot.filter(|_| healthy) {
                append_progress(&mut output, snapshot)?;
                append_outcomes(&mut output, snapshot)?;
                append_optional(&mut output, snapshot)?;
                append_durations(&mut output, snapshot, scopes)?;
            }
            Ok(())
        })
        .map_err(|_| io::Error::other("training metrics exposition exceeds 49152 bytes"))?;
    assert!(output.text.len() <= MAX_EXPOSITION_BYTES);
    assert!(output.text.ends_with('\n'));
    Ok(output.text)
}

struct Exposition {
    text: String,
}

impl Write for Exposition {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        assert!(self.text.len() <= MAX_EXPOSITION_BYTES);
        if text.len() > MAX_EXPOSITION_BYTES - self.text.len() {
            return Err(fmt::Error);
        }
        self.text.push_str(text);
        assert!(self.text.len() <= MAX_EXPOSITION_BYTES);
        Ok(())
    }
}

fn append_status(
    output: &mut Exposition,
    active: bool,
    available: bool,
    healthy: bool,
    heartbeat: u64,
) -> fmt::Result {
    for (name, help, value) in [
        ("active", "Whether a training writer is active.", active),
        (
            "metrics_available",
            "Whether a healthy committed training snapshot is available.",
            available,
        ),
        (
            "metrics_state_healthy",
            "Whether the training metrics state is healthy.",
            healthy,
        ),
    ] {
        scalar(output, name, help, "gauge", u8::from(value))?;
    }
    if heartbeat > 0 {
        scalar(
            output,
            "last_heartbeat_timestamp_seconds",
            "Last supplied training heartbeat as Unix timestamp seconds.",
            "gauge",
            heartbeat,
        )?;
    }
    Ok(())
}

fn append_progress(output: &mut Exposition, snapshot: &TrainingSnapshot) -> fmt::Result {
    for (name, help, kind, value) in [
        (
            "updates_completed",
            "Absolute completed training updates, including restored progress.",
            "gauge",
            snapshot.completed_updates,
        ),
        (
            "updates_target",
            "Target number of completed training updates.",
            "gauge",
            snapshot.updates_target,
        ),
        (
            "samples_total",
            "Absolute training samples, including restored totals.",
            "counter",
            snapshot.samples,
        ),
        (
            "optimizer_steps_total",
            "Absolute optimizer steps, including restored totals.",
            "counter",
            snapshot.optimizer_steps,
        ),
        (
            "metrics_start_update",
            "Baseline update for durable outcome and duration coverage.",
            "gauge",
            snapshot.start_update,
        ),
        (
            "parallel_worlds",
            "Configured number of parallel training worlds.",
            "gauge",
            snapshot.parallel,
        ),
        (
            "games_per_update",
            "Configured games per update; zero denotes window mode.",
            "gauge",
            snapshot.games_per_update,
        ),
    ] {
        scalar(output, name, help, kind, value)?;
    }
    Ok(())
}

fn append_outcomes(output: &mut Exposition, snapshot: &TrainingSnapshot) -> fmt::Result {
    for (name, help, kind, values) in [
        (
            "games_total",
            "Committed training games by mutually exclusive outcome since coverage began.",
            "counter",
            snapshot.games,
        ),
        (
            "last_update_games",
            "Games by mutually exclusive outcome in the last committed update.",
            "gauge",
            snapshot.last_update_games,
        ),
    ] {
        family(output, name, help, kind)?;
        for (outcome, value) in OUTCOMES.into_iter().zip(values) {
            writeln!(
                output,
                "drysua_training_{name}{{outcome=\"{outcome}\"}} {value}"
            )?;
        }
    }
    Ok(())
}

fn append_optional(output: &mut Exposition, snapshot: &TrainingSnapshot) -> fmt::Result {
    if let Some(losses) = snapshot.losses {
        for ((name, help), value) in [
            ("policy_loss", "Latest committed PPO policy loss."),
            ("value_loss", "Latest committed PPO value loss."),
            ("entropy", "Latest committed PPO entropy."),
            (
                "approximate_kl",
                "Latest committed PPO approximate KL divergence.",
            ),
        ]
        .into_iter()
        .zip(losses)
        {
            assert!(value.is_finite());
            scalar(output, name, help, "gauge", value)?;
        }
    }
    if let (Some(generation), Some(scale)) = (snapshot.generation, snapshot.scale_bp) {
        scalar(
            output,
            "generation",
            "Actual zero-based environment generation.",
            "gauge",
            generation,
        )?;
        scalar(
            output,
            "environment_scale_ratio",
            "Environment scale as a fraction of full scale.",
            "gauge",
            scale_ratio(scale),
        )?;
    }
    Ok(())
}

#[allow(
    clippy::float_arithmetic,
    reason = "Prometheus represents basis-point environment scale as a floating-point ratio"
)]
fn scale_ratio(scale: u32) -> f64 {
    assert!(scale <= 10_000);
    let ratio = f64::from(scale) / 10_000.0;
    assert!((0.0..=1.0).contains(&ratio));
    ratio
}

fn append_durations(
    output: &mut Exposition,
    snapshot: &TrainingSnapshot,
    scopes: Option<&[DurationHistogram; 3]>,
) -> fmt::Result {
    family(
        output,
        "update_duration_seconds",
        "Committed training update durations in seconds since coverage began.",
        "histogram",
    )?;
    histogram(
        output,
        "update_duration_seconds",
        "",
        &snapshot.durations[0],
    )?;
    family(
        output,
        "stage_duration_seconds",
        "Committed training stage durations in seconds since coverage began.",
        "histogram",
    )?;
    assert_eq!(STAGE_LABELS.len() + 1, snapshot.durations.len());
    for (label, duration) in STAGE_LABELS.into_iter().zip(&snapshot.durations[1..]) {
        histogram(output, "stage_duration_seconds", label, duration)?;
    }
    if let Some(scopes) = scopes {
        family(
            output,
            "scope_duration_seconds",
            "Current invocation scope durations in seconds; not durable across invocations.",
            "histogram",
        )?;
        assert_eq!(SCOPE_LABELS.len(), scopes.len());
        for (label, duration) in SCOPE_LABELS.into_iter().zip(scopes) {
            histogram(output, "scope_duration_seconds", label, duration)?;
        }
    }
    Ok(())
}

fn histogram(
    output: &mut Exposition,
    name: &'static str,
    label: &'static str,
    duration: &DurationHistogram,
) -> fmt::Result {
    assert!(duration.sum_seconds.is_finite());
    assert!(duration.buckets[BUCKETS.len() - 1] <= duration.count);
    // Labels come only from the fixed stage/scope arrays, never snapshot identities.
    assert!(label.is_empty() || STAGE_LABELS.contains(&label) || SCOPE_LABELS.contains(&label));
    let comma = if label.is_empty() { "" } else { "," };
    for (upper, count) in BUCKETS.into_iter().zip(duration.buckets) {
        writeln!(
            output,
            "drysua_training_{name}_bucket{{{label}{comma}le=\"{upper}\"}} {count}"
        )?;
    }
    writeln!(
        output,
        "drysua_training_{name}_bucket{{{label}{comma}le=\"+Inf\"}} {}",
        duration.count,
    )?;
    let (open, close) = if label.is_empty() {
        ("", "")
    } else {
        ("{", "}")
    };
    writeln!(
        output,
        "drysua_training_{name}_count{open}{label}{close} {}",
        duration.count,
    )?;
    writeln!(
        output,
        "drysua_training_{name}_sum{open}{label}{close} {}",
        duration.sum_seconds,
    )
}

fn family(output: &mut Exposition, name: &str, help: &str, kind: &str) -> fmt::Result {
    assert!(matches!(kind, "counter" | "gauge" | "histogram"));
    assert!(!name.is_empty());
    writeln!(
        output,
        "# HELP drysua_training_{name} {help}\n# TYPE drysua_training_{name} {kind}"
    )
}

fn scalar(
    output: &mut Exposition,
    name: &'static str,
    help: &'static str,
    kind: &'static str,
    value: impl fmt::Display,
) -> fmt::Result {
    family(output, name, help, kind)?;
    writeln!(output, "drysua_training_{name} {value}")
}

fn validate_scopes(scopes: &[DurationHistogram; 3]) -> io::Result<()> {
    // The snapshot owns its private histogram validator; invocation scopes are separate.
    for histogram in scopes {
        if !histogram.sum_seconds.is_finite() || histogram.sum_seconds < 0.0 {
            return Err(invalid(
                "metrics scope duration sum must be finite and nonnegative",
            ));
        }
        if histogram.count >= 1_u64 << 53 {
            return Err(invalid(
                "metrics scope duration count exceeds 9007199254740991",
            ));
        }
        let mut previous = 0;
        for count in histogram.buckets {
            if count < previous || count > histogram.count {
                return Err(invalid(
                    "metrics scope duration buckets must be cumulative and at most count",
                ));
            }
            previous = count;
        }
        if histogram.count == 0 && histogram.sum_seconds != 0.0 {
            return Err(invalid(
                "metrics empty scope duration histogram must have zero sum",
            ));
        }
        assert!(histogram.sum_seconds.is_finite());
        assert!(previous <= histogram.count);
    }
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
