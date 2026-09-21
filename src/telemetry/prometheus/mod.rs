//! Bounded committed training telemetry; no policy-path networking or accounting.

mod exposition;
#[cfg(test)]
mod options_tests;
mod registry;
mod resources;
mod server;
mod shutdown;
mod snapshot;
mod state;

use std::io;
use std::net::SocketAddr;
#[cfg(any(feature = "builtin", test))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "builtin")]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use registry::Registry;
use server::MetricsServer;
#[cfg(feature = "builtin")]
use snapshot::TrainingSnapshot;
#[cfg(any(feature = "builtin", test))]
use state::MetricsStore;

static ENABLED: AtomicBool = AtomicBool::new(false);
static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

#[derive(Clone, Debug, Default, clap::Args)]
pub(crate) struct MetricsOptions {
    /// Existing private shared metrics directory, separate from checkpoints.
    #[arg(long)]
    pub(crate) metrics_directory: Option<PathBuf>,
    /// Optional loopback-only in-process Prometheus listener (for example 127.0.0.1:9464).
    #[arg(long)]
    pub(crate) metrics_listen: Option<SocketAddr>,
}

#[cfg(any(feature = "builtin", test))]
impl MetricsOptions {
    pub(crate) fn validate_checkpoint_directory(&self, checkpoint: &Path) -> io::Result<()> {
        if let Some(directory) = &self.metrics_directory {
            let directory = directory.canonicalize()?;
            let checkpoint = checkpoint.canonicalize()?;
            if directory.starts_with(&checkpoint) || checkpoint.starts_with(&directory) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "metrics and checkpoint directories must be separate, non-nested directories",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn start(&self) -> io::Result<MetricsGuard> {
        if self.metrics_directory.is_none() && self.metrics_listen.is_none() {
            return Ok(MetricsGuard {
                installed: false,
                server: None,
            });
        }
        validate_listener(self.metrics_listen)?;
        let mut registry = REGISTRY.lock().map_err(|_| poisoned())?;
        if registry.is_some() {
            return Err(io::Error::other(
                "metrics registry already has a training owner",
            ));
        }
        let store = self
            .metrics_directory
            .as_deref()
            .map(MetricsStore::open)
            .transpose()?;
        *registry = Some(Registry::new(store));
        ENABLED.store(true, Ordering::Release);
        drop(registry);
        let mut guard = MetricsGuard {
            installed: true,
            server: None,
        };
        if let Some(listen) = self.metrics_listen {
            guard.server = Some(MetricsServer::start(listen, render_registry)?);
        }
        eprintln!(
            "level=INFO event=training_metrics_started persistent={} direct_listener={}",
            self.metrics_directory.is_some(),
            self.metrics_listen.is_some()
        );
        Ok(guard)
    }
}

#[cfg(any(feature = "builtin", test))]
pub(crate) struct MetricsGuard {
    installed: bool,
    server: Option<MetricsServer>,
}

#[cfg(any(feature = "builtin", test))]
impl MetricsGuard {
    pub(crate) fn finish(mut self) -> io::Result<()> {
        self.close()
    }

    fn close(&mut self) -> io::Result<()> {
        if !self.installed {
            return Ok(());
        }
        self.installed = false;
        let server_result = self.server.take().map(MetricsServer::shutdown).transpose();
        let mut registry = REGISTRY.lock().map_err(|_| poisoned())?;
        let health = registry.as_ref().map(Registry::check_health).transpose();
        *registry = None;
        ENABLED.store(false, Ordering::Release);
        eprintln!("level=INFO event=training_metrics_stopped");
        server_result?;
        health?;
        Ok(())
    }
}

#[cfg(any(feature = "builtin", test))]
impl Drop for MetricsGuard {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("level=ERROR event=training_metrics_shutdown_failed error={error:?}");
        }
    }
}

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

pub(crate) fn serve_directory(directory: PathBuf, listen: SocketAddr) -> io::Result<()> {
    validate_listener(Some(listen))?;
    let signal = shutdown::install()?;
    let mut reader = DirectoryReader::new(directory);
    eprintln!("level=INFO event=metrics_exporter_started listen={listen}");
    let result = MetricsServer::run(listen, move || reader.render(), || signal.requested());
    eprintln!("level=INFO event=metrics_exporter_stopped");
    result
}

#[cfg(any(feature = "builtin", test))]
fn render_registry() -> io::Result<String> {
    let registry = REGISTRY.lock().map_err(|_| poisoned())?;
    let Some(registry) = registry.as_ref() else {
        return exposition::render(None, false, false, 0, None);
    };
    exposition::render(
        registry.committed.as_ref(),
        enabled(),
        registry.failure.is_none(),
        registry
            .committed
            .as_ref()
            .map_or(0, |snapshot| snapshot.heartbeat),
        Some(&registry.scopes),
    )
}

#[cfg(all(test, feature = "builtin"))]
pub(crate) fn render_directory_for_test(directory: &Path) -> io::Result<String> {
    DirectoryReader::new(directory.to_owned()).render()
}

struct DirectoryReader {
    directory: PathBuf,
    failure: Option<String>,
}

impl DirectoryReader {
    fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            failure: None,
        }
    }

    fn render(&mut self) -> io::Result<String> {
        let snapshot = state::read_snapshot(&self.directory);
        let active = state::writer_active(&self.directory);
        let pending = state::has_pending(&self.directory);
        let failure = snapshot
            .as_ref()
            .err()
            .or(active.as_ref().err())
            .or(pending.as_ref().err())
            .map(ToString::to_string)
            .or_else(|| {
                (pending.as_ref().is_ok_and(|pending| *pending)
                    && active.as_ref().is_ok_and(|active| !active))
                .then(|| {
                    "metrics pending checkpoint requires training resume reconciliation".to_owned()
                })
            });
        if failure != self.failure {
            match &failure {
                Some(error) => {
                    eprintln!("level=ERROR event=metrics_state_unavailable error={error:?}")
                }
                None => eprintln!("level=INFO event=metrics_state_available"),
            }
            self.failure = failure;
        }
        let heartbeat = snapshot.as_ref().map_or(0, |snapshot| snapshot.heartbeat);
        exposition::render(
            snapshot.as_ref().ok(),
            active.unwrap_or(false),
            self.failure.is_none(),
            heartbeat,
            None,
        )
    }
}

fn validate_listener(listen: Option<SocketAddr>) -> io::Result<()> {
    if listen.is_some_and(|listen| !listen.ip().is_loopback()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "metrics listener must bind a loopback address",
        ));
    }
    Ok(())
}

fn with_registry(operation: impl FnOnce(&mut Registry) -> io::Result<()>) -> io::Result<()> {
    if !enabled() {
        return Ok(());
    }
    let mut registry = REGISTRY.lock().map_err(|_| poisoned())?;
    let registry = registry
        .as_mut()
        .ok_or_else(|| io::Error::other("metrics registry is absent"))?;
    let result = registry.check_health().and_then(|()| operation(registry));
    if let Err(error) = &result
        && registry.failure.is_none()
    {
        eprintln!("level=ERROR event=training_metrics_failed error={error:?}");
        registry.failure = Some(error.to_string().chars().take(1024).collect());
    }
    result
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn poisoned() -> io::Error {
    io::Error::other("metrics registry mutex was poisoned")
}

#[cfg(any(feature = "builtin", test))]
fn record_server_failure(error: &io::Error) {
    eprintln!("level=ERROR event=metrics_server_failed error={error:?}");
    let _ = with_registry(|_| Err(io::Error::other(format!("metrics server failed: {error}"))));
}

pub(crate) fn prepare_checkpoint(
    run: &crate::CheckpointRun,
    config: crate::PpoConfig,
    update: u64,
    identity: [u8; 32],
) -> io::Result<()> {
    with_registry(|registry| {
        let scope =
            crate::checkpoint::metrics_scope_identity(run, config).map_err(io::Error::other)?;
        registry.prepare(scope, update, identity, timestamp())
    })
}

pub(crate) fn commit_checkpoint(update: u64, identity: [u8; 32]) -> io::Result<()> {
    with_registry(|registry| registry.commit(update, identity))
}

#[cfg(feature = "builtin")]
pub(crate) struct TrainingMetricsStart {
    pub completed_updates: u64,
    pub samples: u64,
    pub optimizer_steps: u64,
    pub updates_target: u64,
    pub parallel: usize,
    pub games_per_update: usize,
}

#[cfg(feature = "builtin")]
pub(crate) fn begin_training(
    run: &crate::CheckpointRun,
    config: crate::PpoConfig,
    directory: &Path,
    resume: bool,
    start: TrainingMetricsStart,
) -> io::Result<()> {
    with_registry(|registry| {
        let scope =
            crate::checkpoint::metrics_scope_identity(run, config).map_err(io::Error::other)?;
        let checkpoint = if resume {
            crate::checkpoint::metrics_checkpoint_identity(directory).map_err(io::Error::other)?
        } else {
            [0; 32]
        };
        let baseline = TrainingSnapshot {
            scope,
            checkpoint,
            completed_updates: start.completed_updates,
            updates_target: start.updates_target,
            samples: start.samples,
            optimizer_steps: start.optimizer_steps,
            start_update: start.completed_updates,
            parallel: start.parallel as u64,
            games_per_update: start.games_per_update as u64,
            heartbeat: timestamp(),
            ..TrainingSnapshot::default()
        };
        registry.begin(baseline, resume)?;
        let coverage = registry
            .committed
            .as_ref()
            .expect("begin establishes coverage")
            .start_update;
        eprintln!(
            "level=INFO event=training_metrics_coverage start_update={coverage} restored_update={}",
            start.completed_updates
        );
        Ok(())
    })
}

#[cfg(feature = "builtin")]
pub(crate) fn observe_training_update(report: &crate::TrainingCheckpointReport) -> io::Result<()> {
    with_registry(|registry| {
        registry.observe(registry::UpdateObservation {
            completed_updates: report.completed_updates,
            samples: report.rollout_samples,
            optimizer_steps: report.optimizer_step,
            games: [
                report.terminal_wins,
                report.terminal_losses,
                report.terminal_draws,
                report.episode_timeouts,
            ],
            losses: [
                report.policy_loss,
                report.value_loss,
                report.entropy,
                report.approximate_kl,
            ],
        })
    })
}

#[cfg(feature = "builtin")]
pub(crate) fn record_update_timing(
    update_index: u64,
    elapsed: Duration,
    stages: [Option<Duration>; 5],
    valid: bool,
) {
    // The next checkpoint/update fails closed if an infallible timer hook detects a fault.
    let _ = with_registry(|registry| registry.timing(update_index, elapsed, stages, valid));
}

#[cfg(feature = "builtin")]
pub(crate) fn record_scope_timing(scope: usize, elapsed: Duration, complete: bool) {
    if complete {
        let _ = with_registry(|registry| {
            registry
                .scopes
                .get_mut(scope)
                .ok_or_else(|| io::Error::other("metrics timing scope is invalid"))?
                .observe(elapsed)
        });
    }
}

#[cfg(feature = "builtin")]
pub(crate) fn set_generation(generation: u64, scale_bp: u32) {
    let _ = with_registry(|registry| registry.generation(generation, scale_bp));
}
