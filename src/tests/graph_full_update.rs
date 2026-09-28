//! One fresh full update per process; retained artifacts qualify exact cross-process parity.

use super::*;
use crate::model::cuda_graph_probe::ActorGraphStats;
use std::ffi::OsStr;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

const RECORD_FILE: &str = "graph-full-update.json";
const MAX_RECORD_BYTES: u64 = 64 * 1024;

#[test]
fn full_update_gate_requires_an_explicit_graph_mode() {
    assert_eq!(
        required_graph_mode(None),
        Err("DRYSUA_PROBE_ACTOR_GRAPH must be explicitly set to 0 or 1".to_owned())
    );
}

#[test]
fn full_update_gate_requires_nonempty_initial_weights() {
    for value in [None, Some(OsStr::new(""))] {
        assert_eq!(
            required_weights(value),
            Err("DRYSUA_PROBE_WEIGHTS must name the fixed CREDITu10 initial-weights directory")
        );
    }
    let path = OsStr::new("read-only-CREDITu10");
    assert_eq!(required_weights(Some(path)), Ok(PathBuf::from(path)));
}

#[test]
fn actor_layout_defaults_preserve_scope_and_group_two_uses_normal_scope() {
    assert_eq!(parse_actor_layout(None, None), Ok((40, 1)));
    let scope = |batch, groups| {
        let options = gate_settings(batch, groups);
        annealed_run(
            &options,
            PolicyDevice::Cpu,
            options.ppo,
            AnnealedHarness::default(),
            None,
        )
        .expect("actor layout scope")
        .command_line
    };
    let expected = concat!(
        "train-annealed --updates 1 --games 40 --parallel 40 --generation-games 40 ",
        "--zero-updates 0 --epochs 4 --minibatch 2048 --seed 9001 --map 2 --device cpu ",
        "--sample-budget annealed-v1 --opponent teacher --reuse-actor-values ",
        "--training-microbatch 256"
    );
    assert_eq!(scope(40, 1), expected);
    let grouped = expected.replace("--parallel 40", "--parallel 20").replace(
        "--training-microbatch 256",
        "--actor-pipeline-groups 2 --training-microbatch 256",
    );
    assert_eq!(scope(20, 2), grouped);
}

#[test]
fn actor_layout_accepts_only_one_forty_world_collection_scope() {
    for (batch, groups, expected) in [("40", "1", (40, 1)), ("20", "2", (20, 2))] {
        assert_eq!(
            parse_actor_layout(Some(OsStr::new(batch)), Some(OsStr::new(groups))),
            Ok(expected)
        );
    }
    for (batch, groups) in [
        (Some("20"), Some("1")),
        (Some("40"), Some("2")),
        (Some("20"), None),
        (None, Some("2")),
    ] {
        assert_eq!(
            parse_actor_layout(batch.map(OsStr::new), groups.map(OsStr::new)),
            Err("actor batch/groups must be (40, 1) or (20, 2)")
        );
    }
}

#[test]
fn actor_controls_reject_noncanonical_and_out_of_range_values() {
    for value in ["", "0", "1", "32", "64", "020", " 20", "20 ", "+20", "-1"] {
        assert_eq!(
            parse_actor_batch(Some(OsStr::new(value))),
            Err("DRYSUA_PROBE_ACTOR_BATCH must be 20 or 40")
        );
    }
    for value in ["", "0", "3", "01", " 1", "1 ", "+1", "-1"] {
        assert_eq!(
            parse_actor_groups(Some(OsStr::new(value))),
            Err("DRYSUA_PROBE_ACTOR_GROUPS must be 1 or 2")
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(&[255]);
        assert_eq!(
            parse_actor_batch(Some(invalid)),
            Err("DRYSUA_PROBE_ACTOR_BATCH must be 20 or 40")
        );
        assert_eq!(
            parse_actor_groups(Some(invalid)),
            Err("DRYSUA_PROBE_ACTOR_GROUPS must be 1 or 2")
        );
    }
}

#[test]
fn graph_statistics_accept_two_slots_and_an_unused_singleton() {
    for main_batch in [20, 40] {
        for graph in [false, true] {
            let mut stats = graph_stats_fixture(main_batch, graph);
            assert_eq!(validate_graph_stats(&stats, graph, main_batch), Ok(()));
            stats.calls_by_batch[1] = 0;
            stats.hits_by_batch[1] = 0;
            assert_eq!(validate_graph_stats(&stats, graph, main_batch), Ok(()));
        }
    }
}

#[test]
fn graph_statistics_reject_wrong_shapes_and_capture_counts() {
    let mut stats = graph_stats_fixture(40, true);
    stats.main_batch = 20;
    assert_eq!(
        validate_graph_stats(&stats, true, 40),
        Err("actor graph statistics do not match the two fixed shapes")
    );
    stats.main_batch = 40;
    stats.shapes = [40, 40];
    assert_eq!(
        validate_graph_stats(&stats, true, 40),
        Err("actor graph statistics do not match the two fixed shapes")
    );
    for graph in [false, true] {
        let mut stats = graph_stats_fixture(40, graph);
        stats.captures = if graph { 1 } else { 2 };
        assert_eq!(
            validate_graph_stats(&stats, graph, 40),
            Err("actor graph capture count must be two for graph mode and zero for eager")
        );
    }
}

#[test]
fn graph_statistics_accept_bound_and_reject_empty_or_excess_calls() {
    let maximum = 2 * ANNEALED_EPISODE_DECISIONS as u64;
    for calls in [0, maximum + 1, u64::MAX] {
        let mut stats = graph_stats_fixture(40, false);
        stats.calls_by_batch = [0; 65];
        stats.calls_by_batch[40] = calls;
        let expected = if calls == 0 {
            "actor graph statistics require at least one actor call"
        } else {
            "actor graph call count exceeds the full-update bound"
        };
        assert_eq!(validate_graph_stats(&stats, false, 40), Err(expected));
    }
    let mut stats = graph_stats_fixture(40, false);
    stats.calls_by_batch = [0; 65];
    stats.calls_by_batch[40] = maximum;
    assert_eq!(validate_graph_stats(&stats, false, 40), Ok(()));
    stats.calls_by_batch[39] = 1;
    assert_eq!(
        validate_graph_stats(&stats, false, 40),
        Err("actor graph call count exceeds the full-update bound")
    );
}

#[test]
fn graph_statistics_reject_zero_batch_and_invalid_hit_histograms() {
    for (case, graph, expected) in [
        (0, true, "actor graph batch-zero counters must be zero"),
        (1, true, "actor graph batch-zero counters must be zero"),
        (2, true, "actor graph hits exceed calls"),
        (
            3,
            true,
            "actor graph hits require graph mode and a captured shape",
        ),
        (
            4,
            false,
            "actor graph hits require graph mode and a captured shape",
        ),
        (5, true, "actor graph mode requires a main-shape replay"),
    ] {
        let mut stats = graph_stats_fixture(40, graph);
        match case {
            0 => stats.calls_by_batch[0] = 1,
            1 => stats.hits_by_batch[0] = 1,
            2 => stats.hits_by_batch[40] = stats.calls_by_batch[40] + 1,
            3 => stats.hits_by_batch[39] = 1,
            4 => stats.hits_by_batch[40] = 1,
            5 => stats.hits_by_batch[40] = 0,
            _ => unreachable!("bounded statistics cases"),
        }
        assert_eq!(validate_graph_stats(&stats, graph, 40), Err(expected));
    }
}

#[test]
fn graph_history_comparison_ignores_mode_specific_replays_and_timings() {
    let eager = graph_stats_fixture(40, false);
    let mut graph = graph_stats_fixture(40, true);
    graph.setup_ns = 1;
    graph.retirement_ns = 2;
    let source = serde_json::json!({"graph_stats": graph_stats_record(&eager)});
    let target = serde_json::json!({"graph_stats": graph_stats_record(&graph)});

    assert_matching_graph_histories(&source, &target);
}

#[test]
#[should_panic(expected = "cross-process graph workload calls_by_batch")]
fn graph_history_comparison_rejects_changed_batch_calls() {
    let eager = graph_stats_fixture(40, false);
    let mut graph = graph_stats_fixture(40, true);
    graph.calls_by_batch[40] += 1;
    let source = serde_json::json!({"graph_stats": graph_stats_record(&eager)});
    let target = serde_json::json!({"graph_stats": graph_stats_record(&graph)});

    assert_matching_graph_histories(&source, &target);
}

#[test]
#[ignore = "one fresh full CUDA update per process; requires the owner's bounded runner"]
fn actor_graph_full_update_gate_cuda() {
    let graph = required_graph_mode(std::env::var_os("DRYSUA_PROBE_ACTOR_GRAPH").as_deref())
        .expect("explicit graph/eager gate mode");
    let (main_batch, actor_groups) = parse_actor_layout(
        std::env::var_os("DRYSUA_PROBE_ACTOR_BATCH").as_deref(),
        std::env::var_os("DRYSUA_PROBE_ACTOR_GROUPS").as_deref(),
    )
    .expect("one forty-world actor collection scope");
    let initial = required_weights(std::env::var_os("DRYSUA_PROBE_WEIGHTS").as_deref())
        .expect("fixed CREDITu10 initial weights");
    assert!(
        initial.is_dir(),
        "initial weights must be an existing directory"
    );
    let options = gate_settings(main_batch, actor_groups);
    let harness = AnnealedHarness::default();
    validate_annealed(&options, harness).expect("full workload configuration");
    let directory = test_directory("actor-graph-full-update");
    eprintln!(
        "graph-full-update-start {}",
        serde_json::json!({
            "graph": graph,
            "directory": directory.display().to_string(),
            "initial_weights": initial.display().to_string(),
            "actor_counts_enabled": std::env::var_os("DRYSUA_PROBE_MODE").is_some(),
            "workload": workload_record(&options),
        })
    );
    let started = Instant::now();
    let report = run_annealed_job_harnessed(
        options.clone(),
        harness,
        PolicyDevice::Cuda { ordinal: 0 },
        &directory,
        false,
        Some(&initial),
        |_| {},
    )
    .expect("fresh complete graph/eager update");
    let elapsed = started.elapsed();
    finish_gate(graph, &directory, &initial, &options, &report, elapsed);
}

#[test]
#[ignore = "requires eager and graph gate directories; read-only CPU artifact comparison"]
fn actor_graph_full_update_artifacts_match_exactly() {
    let eager = comparison_directory("DRYSUA_GRAPH_EAGER_DIRECTORY");
    let candidate = comparison_directory("DRYSUA_GRAPH_CANDIDATE_DIRECTORY");
    assert_ne!(
        eager, candidate,
        "comparison requires two distinct fresh runs"
    );
    let source = read_record(&eager);
    let target = read_record(&candidate);
    assert_eq!(source["graph"], serde_json::json!(false));
    assert_eq!(target["graph"], serde_json::json!(true));
    for record in [&source, &target] {
        assert_eq!(record["accepted"], serde_json::json!(true));
        assert_eq!(record["report"]["games"], serde_json::json!(40));
        assert_eq!(record["report"]["steps"], serde_json::json!(40));
    }
    for key in [
        "workload",
        "report",
        "actor_trace",
        "run_scope",
        "checkpoint_tensor_sha256",
        "checkpoint_digests",
    ] {
        assert_eq!(source[key], target[key], "cross-process {key}");
    }
    assert_matching_graph_histories(&source, &target);
    for (directory, record) in [(&eager, &source), (&candidate, &target)] {
        assert_eq!(
            serde_json::json!(checkpoint_digests(directory)),
            record["checkpoint_digests"],
            "record must describe the retained checkpoint"
        );
    }
    assert_artifact_bytes(&eager, &candidate);
    assert_trajectory_equal(&eager, &candidate);
    eprintln!(
        "graph-full-update-comparison {}",
        serde_json::json!({
            "exact": true,
            "eager_directory": eager.display().to_string(),
            "candidate_directory": candidate.display().to_string(),
            "eager_elapsed_ns": source["elapsed_ns"],
            "candidate_elapsed_ns": target["elapsed_ns"],
            "checkpoint_tensor_sha256": source["checkpoint_tensor_sha256"],
        })
    );
}

fn required_graph_mode(value: Option<&OsStr>) -> Result<bool, String> {
    crate::model::cuda_graph_probe::parse_graph_mode(value)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "DRYSUA_PROBE_ACTOR_GRAPH must be explicitly set to 0 or 1".to_owned())
}

fn required_weights(value: Option<&OsStr>) -> Result<PathBuf, &'static str> {
    value
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or("DRYSUA_PROBE_WEIGHTS must name the fixed CREDITu10 initial-weights directory")
}

fn parse_actor_batch(value: Option<&OsStr>) -> Result<usize, &'static str> {
    match value {
        None => Ok(40),
        Some(value) if value == "40" => Ok(40),
        Some(value) if value == "20" => Ok(20),
        Some(_) => Err("DRYSUA_PROBE_ACTOR_BATCH must be 20 or 40"),
    }
}

fn parse_actor_groups(value: Option<&OsStr>) -> Result<usize, &'static str> {
    match value {
        None => Ok(1),
        Some(value) if value == "1" => Ok(1),
        Some(value) if value == "2" => Ok(2),
        Some(_) => Err("DRYSUA_PROBE_ACTOR_GROUPS must be 1 or 2"),
    }
}

fn parse_actor_layout(
    batch: Option<&OsStr>,
    groups: Option<&OsStr>,
) -> Result<(usize, usize), &'static str> {
    let layout = (parse_actor_batch(batch)?, parse_actor_groups(groups)?);
    match layout {
        (40, 1) | (20, 2) => Ok(layout),
        _ => Err("actor batch/groups must be (40, 1) or (20, 2)"),
    }
}

fn gate_settings(main_batch: usize, actor_groups: usize) -> AnnealedJobConfig {
    assert!(matches!((main_batch, actor_groups), (40, 1) | (20, 2)));
    let mut options = settings(9001, 1);
    options.games_per_update = 40;
    options.parallel_worlds = main_batch;
    options.games_per_generation = 40;
    options.opponent = AnnealedOpponent::Teacher;
    options.ppo.environments = 40;
    options.ppo.sample_budget = crate::PpoSampleBudget::Annealed;
    options.ppo.epochs = 4;
    options.ppo.minibatch = 2048;
    options.execution = crate::TrainingExecutionOptions {
        actor_pipeline_groups: actor_groups,
        reuse_actor_values: true,
        training_microbatch: 256,
        host_math_workers: 1,
        balanced_minibatches: false,
    };
    options
}

fn workload_record(options: &AnnealedJobConfig) -> serde_json::Value {
    serde_json::json!({
        "seed": options.seed,
        "updates": options.updates,
        "games": options.games_per_update,
        "parallel_worlds": options.parallel_worlds,
        "generation_games": options.games_per_generation,
        "opponent": "Teacher",
        "actor_pipeline_groups": options.execution.actor_pipeline_groups,
        "reuse_actor_values": options.execution.reuse_actor_values,
        "training_microbatch": options.execution.training_microbatch,
        "host_math_workers": options.execution.host_math_workers,
        "balanced_minibatches": options.execution.balanced_minibatches,
        "epochs": options.ppo.epochs,
        "effective_minibatch": options.ppo.minibatch,
        "episode_decisions": ANNEALED_EPISODE_DECISIONS,
    })
}

fn finish_gate(
    graph: bool,
    directory: &Path,
    initial: &Path,
    options: &AnnealedJobConfig,
    report: &AnnealedJobReport,
    elapsed: Duration,
) {
    let stats = crate::model::cuda_graph_probe::actor_graph_stats_for_test()
        .expect("checked pre-PPO graph retirement statistics");
    let accepted = validate_full_workload(report)
        .and_then(|_| validate_graph_stats(&stats, graph, options.parallel_worlds));
    let artifact = TrainingArtifact::load(directory).expect("completed gate artifact");
    let expected_scope = annealed_run(
        options,
        PolicyDevice::Cuda { ordinal: 0 },
        options.ppo,
        AnnealedHarness::default(),
        None,
    )
    .expect("graph-independent run scope");
    assert_eq!(artifact.run(), &expected_scope);
    let digests = checkpoint_digests(directory);
    let record = serde_json::json!({
        "accepted": accepted.is_ok(),
        "graph": graph,
        "directory": directory.display().to_string(),
        "initial_weights": initial.display().to_string(),
        "elapsed_ns": u64::try_from(elapsed.as_nanos()).expect("bounded elapsed duration"),
        "workload": workload_record(options),
        "report": report_record(report),
        "actor_trace": trace_record(),
        "graph_stats": graph_stats_record(&stats),
        "run_scope": artifact.run().command_line,
        "checkpoint_tensor_sha256": digests[1].1,
        "checkpoint_digests": digests,
    });
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(RECORD_FILE))
        .expect("new retained gate record");
    serde_json::to_writer(file, &record).expect("write gate record");
    eprintln!("graph-full-update-result {record}");
    accepted.expect("a reduced workload cannot qualify as a graph speedup");
}

fn validate_full_workload(report: &AnnealedJobReport) -> Result<(), &'static str> {
    if report.games != 40 || report.completed_updates != 1 {
        return Err("full-update gate requires exactly 40 games and one completed update");
    }
    if report.optimizer_step != 40 || report.latest.optimizer_step != 40 {
        return Err("full-update gate requires exactly 40 applied Adam steps");
    }
    if report.latest.minibatches != 40 || report.latest.epochs_completed != 4 {
        return Err("full-update gate requires all 40 minibatches and four epochs");
    }
    if report.latest.stopped_for_kl || report.latest.samples_rejected != 0 {
        return Err("full-update gate forbids a reduced workload from KL rejection");
    }
    if report.rollout_samples == 0 || report.elapsed_ticks == 0 || report.latest.update != 1 {
        return Err("full-update gate requires nonempty completed gameplay");
    }
    if (report.latest.samples_optimized as u64) != report.rollout_samples * 4 {
        return Err("full-update gate requires four complete passes over the rollout");
    }
    Ok(())
}

fn validate_graph_stats(
    stats: &ActorGraphStats,
    graph: bool,
    main_batch: usize,
) -> Result<(), &'static str> {
    if !matches!(main_batch, 20 | 40)
        || stats.main_batch != main_batch
        || stats.shapes != [main_batch, 1]
    {
        return Err("actor graph statistics do not match the two fixed shapes");
    }
    let expected_captures = if graph { 2 } else { 0 };
    if stats.captures != expected_captures {
        return Err("actor graph capture count must be two for graph mode and zero for eager");
    }
    if stats.calls_by_batch[0] != 0 || stats.hits_by_batch[0] != 0 {
        return Err("actor graph batch-zero counters must be zero");
    }
    let maximum = 2 * ANNEALED_EPISODE_DECISIONS as u64;
    let mut total = 0u64;
    for batch in 1..=64 {
        let calls = stats.calls_by_batch[batch];
        let hits = stats.hits_by_batch[batch];
        total = total
            .checked_add(calls)
            .filter(|count| *count <= maximum)
            .ok_or("actor graph call count exceeds the full-update bound")?;
        if hits > calls {
            return Err("actor graph hits exceed calls");
        }
        if hits != 0 && (!graph || (batch != main_batch && batch != 1)) {
            return Err("actor graph hits require graph mode and a captured shape");
        }
    }
    if total == 0 {
        return Err("actor graph statistics require at least one actor call");
    }
    if graph && stats.hits_by_batch[main_batch] == 0 {
        return Err("actor graph mode requires a main-shape replay");
    }
    Ok(())
}

fn graph_stats_record(stats: &ActorGraphStats) -> serde_json::Value {
    serde_json::json!({
        "main_batch": stats.main_batch,
        "shapes": stats.shapes.as_slice(),
        "calls_by_batch": stats.calls_by_batch.as_slice(),
        "hits_by_batch": stats.hits_by_batch.as_slice(),
        "captures": stats.captures,
        "setup_ns": stats.setup_ns,
        "retirement_ns": stats.retirement_ns,
    })
}

fn assert_matching_graph_histories(source: &serde_json::Value, target: &serde_json::Value) {
    // Compare workload history, not mode-specific replay accounting or timings.
    for key in ["main_batch", "shapes", "calls_by_batch"] {
        assert!(
            !source["graph_stats"][key].is_null(),
            "missing graph statistic: {key}"
        );
        assert_eq!(
            source["graph_stats"][key], target["graph_stats"][key],
            "cross-process graph workload {key}"
        );
    }
    assert_eq!(
        source["graph_stats"]["calls_by_batch"]
            .as_array()
            .expect("call histogram")
            .len(),
        65
    );
}

fn graph_stats_fixture(main_batch: usize, graph: bool) -> ActorGraphStats {
    assert!(matches!(main_batch, 20 | 40));
    let mut calls_by_batch = [0; 65];
    calls_by_batch[main_batch] = 7;
    calls_by_batch[1] = 3;
    calls_by_batch[main_batch - 1] = 2;
    let mut hits_by_batch = [0; 65];
    if graph {
        hits_by_batch[main_batch] = 7;
        hits_by_batch[1] = 3;
    }
    ActorGraphStats {
        main_batch,
        shapes: [main_batch, 1],
        calls_by_batch,
        hits_by_batch,
        captures: if graph { 2 } else { 0 },
        setup_ns: 0,
        retirement_ns: 0,
    }
}

fn trace_record() -> serde_json::Value {
    let trace =
        crate::model::cuda_graph_probe::actor_trace_for_test().expect("pre-PPO actor trace");
    serde_json::json!({
        "hashes": trace.hashes.as_slice(),
        "random": trace.random.as_slice(),
        "decisions": trace.decisions.as_slice(),
        "retained": trace.retained.as_slice(),
    })
}

fn assert_artifact_bytes(source: &Path, target: &Path) {
    // CUDA-scoped restore correctly rejects a CPU model. Validate and compare
    // the complete serialized state without migrating it or creating a device.
    let source_artifact = TrainingArtifact::load(source).expect("validated eager artifact");
    let target_artifact = TrainingArtifact::load(target).expect("validated graph artifact");
    assert_eq!(source_artifact.config(), target_artifact.config());
    assert_eq!(source_artifact.run(), target_artifact.run());
    assert_eq!(source_artifact.progress(), target_artifact.progress());
    for name in [
        "checkpoint.meta",
        "checkpoint.safetensors",
        "drysua.weights.safetensors",
    ] {
        let source = bounded_artifact_bytes(&source.join(name));
        let target = bounded_artifact_bytes(&target.join(name));
        assert!(source == target, "complete checkpoint bytes differ: {name}");
    }
}

fn bounded_artifact_bytes(path: &Path) -> Vec<u8> {
    const MAX_BYTES: u64 = crate::MODEL_PARAMETER_COUNT as u64 * 12 + 64 * 1024;
    let file = std::fs::File::open(path).expect("checkpoint file");
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .expect("bounded checkpoint read");
    assert!(!bytes.is_empty(), "checkpoint must be nonempty");
    assert!(
        bytes.len() as u64 <= MAX_BYTES,
        "checkpoint exceeds its fixed tensor bound"
    );
    bytes
}

fn report_record(report: &AnnealedJobReport) -> serde_json::Value {
    let latest = &report.latest;
    serde_json::json!({
        "initial_fingerprint": format!("{:016x}", report.starting_policy_fingerprint),
        "completed_updates": report.completed_updates,
        "games": report.games,
        "samples": report.rollout_samples,
        "steps": report.optimizer_step,
        "ticks": report.elapsed_ticks,
        "generations": report.generations,
        "episode_timeouts": report.episode_timeouts,
        "terminal_wins": report.terminal_wins,
        "terminal_losses": report.terminal_losses,
        "terminal_draws": report.terminal_draws,
        "map2_reward": format!("{:?}", report.map2_reward),
        "latest": {
            "policy_loss_bits": latest.policy_loss.to_bits(),
            "value_loss_bits": latest.value_loss.to_bits(),
            "entropy_bits": latest.entropy.to_bits(),
            "approximate_kl_bits": latest.approximate_kl.to_bits(),
            "rejected_kl_bits": latest.rejected_kl.to_bits(),
            "clip_fraction_bits": latest.clip_fraction.to_bits(),
            "gradient_norm_bits": latest.gradient_norm.to_bits(),
            "applied_scale_bits": latest.applied_scale.to_bits(),
            "samples_optimized": latest.samples_optimized,
            "samples_rejected": latest.samples_rejected,
            "minibatches": latest.minibatches,
            "epochs_completed": latest.epochs_completed,
            "stopped_for_kl": latest.stopped_for_kl,
            "optimizer_step": latest.optimizer_step,
            "update": latest.update,
        },
    })
}

fn comparison_directory(name: &str) -> PathBuf {
    let value = std::env::var_os(name).expect(name);
    assert!(
        !value.is_empty(),
        "{name} must name a completed gate directory"
    );
    let directory = std::fs::canonicalize(value).expect("existing gate directory");
    assert!(directory.is_dir(), "{name} must be a directory");
    directory
}

fn read_record(directory: &Path) -> serde_json::Value {
    let file = std::fs::File::open(directory.join(RECORD_FILE)).expect("retained gate record");
    let mut bytes = Vec::new();
    file.take(MAX_RECORD_BYTES + 1)
        .read_to_end(&mut bytes)
        .expect("read gate record");
    assert!(
        bytes.len() as u64 <= MAX_RECORD_BYTES,
        "bounded gate record"
    );
    let record: serde_json::Value = serde_json::from_slice(&bytes).expect("gate JSON");
    assert!(record.is_object(), "gate record must be an object");
    record
}
