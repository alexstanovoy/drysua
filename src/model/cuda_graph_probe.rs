//! Bounded owner-thread capture experiments, never a production cache.
//! Run this ignored test alone: a capture/backend error terminates the owned process.
//! Each admitted shape gets two warmups and one tensor-only capture: one shape in
//! the encoder probe, two in the actor scope. Replays never enter those caches. Dynamic
//! data is always uploaded outside the guard. One irreversible admission bounds
//! both this probe and the actor experiment; capture failures retire the process.
use super::*;
use candle_core::cuda_backend::cudarc::driver::{CudaGraph, CudaStream, sys};
use std::sync::Arc;
use std::time::Instant;

#[path = "cuda_graph_actor.rs"]
mod actor;
#[path = "../tests/cuda_graph_actor.rs"]
mod actor_tests;
pub(super) use actor::sample_selection;
pub(crate) use actor::{
    ActorGraphStats, actor_graph_stats_for_test, check_owner, parse_graph_mode,
    with_actor_graph_for_test,
};
#[cfg(feature = "builtin")]
pub(crate) use actor::{ActorTrace, actor_trace_for_test, record_actor_trace};

const BATCH: usize = 40;
const TIMED_REPLAYS: usize = 20;
const _: () = assert!(3 + 2 * TIMED_REPLAYS <= 64);

#[test]
#[ignore = "one-shot CUDA encoder graph experiment; requires the owner's bounded runner"]
fn cuda_encoder_graph_matches_changing_inputs_weights_and_times_replay() {
    let model = required(
        PolicyModel::fresh_on(9001, PolicyDevice::Cuda { ordinal: 0 }),
        "model",
    );
    required(actor::admit(&model), "single process admission");
    let stream = probe_stream(&model);
    let frames = [frames(0.0), frames(0.25)];
    let sources = frames
        .each_ref()
        .map(|frames| required(stage_frame_buffer(frames, model.tensor_device()), "stage"));
    assert_eq!(sources[0].1, sources[1].1);
    let input = required(Var::from_tensor(&sources[0].0), "fixed input");
    let inputs = required(
        EncoderInputs::from_buffer(&input.as_detached_tensor(), &sources[0].1, BATCH),
        "input views",
    );
    let outputs = output_buffers(model.tensor_device());
    let expected = warm_encoders(&model, &inputs, &input, &outputs, &sources, &stream, true);
    assert!(
        expected[0] != expected[1],
        "changed features must affect encoder output"
    );
    required(input.set(&sources[0].0), "capture input");
    let graph = capture_once(&model, &inputs, &outputs, &stream);
    for index in 0..2 {
        required(input.set(&sources[index].0), "parity input");
        replay(&model, &graph, &stream);
        assert_outputs(snapshot(&outputs, &stream), &expected[index]);
    }
    change_weights(&model);
    let changed = eager_snapshot(&model, &inputs, &outputs, &stream);
    assert!(
        changed != expected[1],
        "weight mutation must affect the captured encoder"
    );
    replay(&model, &graph, &stream);
    assert_outputs(snapshot(&outputs, &stream), &changed);
    time_paths(&model, &inputs, &input, &outputs, &graph, &stream, &frames);
    required(stream.synchronize(), "before graph destruction");
    drop(graph);
    required(stream.context().check_err(), "graph destruction");
    eprintln!(
        "encoder-graph parity=true batch={BATCH} warmups=2 captures=1 replays={} event_tracking=true production_qualified=false",
        3 + 2 * TIMED_REPLAYS
    );
}

fn probe_stream(model: &PolicyModel) -> Arc<CudaStream> {
    let cuda = required(model.tensor_device().as_cuda_device(), "CUDA backend");
    let stream = cuda.cuda_stream();
    assert!(
        stream.context().has_async_alloc(),
        "graph allocation nodes require async allocation"
    );
    assert!(cuda.is_event_tracking(), "never disable lifetime tracking");
    assert!(
        !stream.context().is_in_multi_stream_mode(),
        "single owner/PTDS only"
    );
    stream
}

fn warm_encoders(
    model: &PolicyModel,
    inputs: &EncoderInputs,
    input: &Var,
    outputs: &[Var; 3],
    sources: &[(Tensor, Vec<usize>); 2],
    stream: &CudaStream,
    cached: bool,
) -> [[Vec<u32>; 3]; 2] {
    let cuda = required(model.tensor_device().as_cuda_device(), "warmup device");
    std::array::from_fn(|index| {
        required(input.set(&sources[index].0), "warmup input");
        let _lock = required(model.read_parameter_lock(), "warmup parameter lock");
        if cached {
            required(
                actor::authorize_shape(model, inputs.batch, false),
                "shape warmup admission",
            );
            let _cache = cuda.enable_cuda_graph_htod_cache();
            required(eager_into(model, inputs, outputs), "cache warmup");
        } else {
            required(eager_into(model, inputs, outputs), "uncached warmup");
        }
        snapshot(outputs, stream)
    })
}

fn frames(shift: f32) -> Vec<FeatureFrame> {
    shape_frames(BATCH, shift)
}

fn shape_frames(batch: usize, shift: f32) -> Vec<FeatureFrame> {
    assert!(matches!(batch, 1 | 20 | 40));
    (0..batch)
        .map(|row| {
            let mut frame = FeatureFrame::new();
            frame.global[0] = row as f32 / batch as f32 + shift;
            frame.units[0][unit_feature::TOKEN_PRESENT] = 1.0;
            frame.units[0][unit_feature::KIND_TOKEN] = if shift == 0.0 { 1.0 } else { 2.0 };
            frame.points[0][point_feature::TOKEN_PRESENT] = 1.0;
            frame.points[0][1] = shift;
            frame.items[0][item_feature::TOKEN_PRESENT] = 1.0;
            frame.items[0][2] = shift;
            assert!(frame.is_finite());
            frame
        })
        .collect()
}

fn output_buffers(device: &Device) -> [Var; 3] {
    shape_output_buffers(device, BATCH)
}

fn shape_output_buffers(device: &Device, batch: usize) -> [Var; 3] {
    assert!(matches!(batch, 1 | 20 | 40));
    [
        required(
            Var::zeros((batch, TRUNK_WIDTH), DType::F32, device),
            "trunk output",
        ),
        required(
            Var::zeros(
                (batch, UNIT_FEATURE_TOKENS, UNIT_EMBEDDING),
                DType::F32,
                device,
            ),
            "unit output",
        ),
        required(
            Var::zeros(
                (batch, POINT_FEATURE_TOKENS, TOKEN_EMBEDDING),
                DType::F32,
                device,
            ),
            "point output",
        ),
    ]
}

fn eager_into(
    model: &PolicyModel,
    inputs: &EncoderInputs,
    outputs: &[Var; 3],
) -> Result<(), ModelError> {
    let state = model.forward_encoder_inputs(inputs)?;
    outputs[0].set(&state.trunk)?;
    outputs[1].set(&state.current_units)?;
    outputs[2].set(&state.points)?;
    // All captured intermediates, including the forward autograd graph, die in this scope.
    Ok(())
}

fn capture_once(
    model: &PolicyModel,
    inputs: &EncoderInputs,
    outputs: &[Var; 3],
    stream: &Arc<CudaStream>,
) -> CudaGraph {
    let cuda = required(model.tensor_device().as_cuda_device(), "capture device");
    let _lock = required(model.read_parameter_lock(), "capture parameter lock");
    required(
        actor::authorize_shape(model, inputs.batch, true),
        "shape capture admission",
    );
    let _cache = cuda.enable_cuda_graph_htod_cache();
    required(stream.synchronize(), "warmup completion");
    required(
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
        "begin capture",
    );
    let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        eager_into(model, inputs, outputs)
    }));
    // End capture even after a body error/panic; never retry an invalid or unknown stream.
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if let Err(error) = &ended {
        eprintln!(
            "encoder-graph capture_body={body:?} end_capture={error}; no retry; terminating owned process"
        );
        std::process::exit(1);
    }
    let graph =
        required(ended, "end capture").unwrap_or_else(|| fatal("end capture returned no graph"));
    let status = required(stream.capture_status(), "capture cleanup status");
    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        fatal("stream remained in capture after end_capture");
    }
    match body {
        Ok(result) => required(result, "capture encoder body"),
        Err(_) => fatal("encoder body panicked during capture; no replay"),
    }
    required(graph.upload(), "graph upload");
    required(stream.synchronize(), "graph upload completion");
    graph
}

fn replay(model: &PolicyModel, graph: &CudaGraph, stream: &CudaStream) {
    let _lock = required(model.read_parameter_lock(), "replay parameter lock");
    required(graph.launch(), "graph replay");
    required(stream.synchronize(), "locked replay completion");
}

fn snapshot(outputs: &[Var; 3], stream: &CudaStream) -> [Vec<u32>; 3] {
    required(stream.synchronize(), "output completion");
    outputs.each_ref().map(|output| {
        let flat = required(output.flatten_all(), "output flatten");
        let values = required(flat.to_vec1::<f32>(), "output readback");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "nonfinite encoder output"
        );
        values.into_iter().map(f32::to_bits).collect()
    })
}

fn assert_outputs(actual: [Vec<u32>; 3], expected: &[Vec<u32>; 3]) {
    for (field, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "encoder output {field} length"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual, expected, "encoder output {field} bits at {index}");
        }
    }
}

fn eager_snapshot(
    model: &PolicyModel,
    inputs: &EncoderInputs,
    outputs: &[Var; 3],
    stream: &CudaStream,
) -> [Vec<u32>; 3] {
    let _lock = required(model.read_parameter_lock(), "eager parameter lock");
    required(
        eager_into(model, inputs, outputs),
        "changed-weight reference",
    );
    snapshot(outputs, stream)
}

fn change_weights(model: &PolicyModel) {
    let before = required(model.policy_identity(), "identity before import");
    let mut parameters = required(model.export_parameters(), "parameter export");
    let mut offset = 0;
    let mut changed = false;
    for (name, shape) in required(model.parameter_schema(), "parameter schema") {
        let end = offset + shape.iter().product::<usize>();
        if name == "trunk.2.bias" {
            for value in &mut parameters[offset..end] {
                *value += 0.25;
            }
            changed = true;
        }
        offset = end;
    }
    assert!(changed);
    assert_eq!(offset, parameters.len());
    required(
        model.import_parameters(&parameters),
        "in-place parameter import outside capture/cache",
    );
    assert_eq!(
        required(model.policy_identity(), "identity after import").revision(),
        before.revision() + 1
    );
}

fn time_paths(
    model: &PolicyModel,
    inputs: &EncoderInputs,
    input: &Var,
    outputs: &[Var; 3],
    graph: &CudaGraph,
    stream: &Arc<CudaStream>,
    frames: &[Vec<FeatureFrame>; 2],
) {
    let _lock = required(model.read_parameter_lock(), "timing parameter lock");
    let expected = snapshot(outputs, stream);
    measure("eager_device_resident", stream, |_| {
        required(eager_into(model, inputs, outputs), "eager encoder")
    });
    assert_outputs(snapshot(outputs, stream), &expected);
    measure("graph_device_resident", stream, |_| {
        required(graph.launch(), "timed graph")
    });
    assert_outputs(snapshot(outputs, stream), &expected);
    measure("eager_staging_upload_included", stream, |index| {
        let state = required(
            model.forward_frames(&frames[index % 2]),
            "eager staged encoder",
        );
        required(outputs[0].set(&state.trunk), "eager trunk sink");
        required(outputs[1].set(&state.current_units), "eager unit sink");
        required(outputs[2].set(&state.points), "eager point sink");
    });
    assert_outputs(snapshot(outputs, stream), &expected);
    measure("graph_staging_upload_included", stream, |index| {
        let (uploaded, _) = required(
            stage_frame_buffer(&frames[index % 2], model.tensor_device()),
            "dynamic upload outside cache",
        );
        required(input.set(&uploaded), "fixed input copy");
        required(graph.launch(), "graph with changed inputs");
    });
    assert_outputs(snapshot(outputs, stream), &expected);
    eprintln!(
        "encoder-graph buffers input_bytes={} output_bytes={} metadata_cache_scope=2_warmups_plus_1_capture",
        input.elem_count() * 4,
        outputs
            .iter()
            .map(|output| output.elem_count() * 4)
            .sum::<usize>()
    );
}

fn measure(label: &str, stream: &Arc<CudaStream>, mut operation: impl FnMut(usize)) {
    required(stream.synchronize(), "timing start boundary");
    let start = required(
        stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)),
        "start event",
    );
    let wall = Instant::now();
    for index in 0..TIMED_REPLAYS {
        operation(index);
    }
    let end = required(
        stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)),
        "end event",
    );
    required(end.synchronize(), "timing completion");
    let wall_ns = wall.elapsed().as_nanos();
    let stream_ms = required(start.elapsed_ms(&end), "event interval");
    // Stream intervals include enqueue gaps, not an isolated sum of GPU kernel durations.
    eprintln!(
        "encoder-graph timing={label} iterations={TIMED_REPLAYS} wall_ns={wall_ns} stream_interval_ms={stream_ms}"
    );
}

fn required<T, E: std::fmt::Display>(result: Result<T, E>, stage: &str) -> T {
    result.unwrap_or_else(|error| fatal(&format!("{stage}: {error}")))
}

fn fatal(message: &str) -> ! {
    eprintln!(
        "encoder-graph failed: {message}; no fallback or retry; terminating owned test process"
    );
    // Instantiation failure can leak one upstream raw graph; process exit is the experiment bound.
    std::process::exit(1)
}
