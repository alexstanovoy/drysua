//! One bounded owner-thread experiment. Never installed by production builds.
use super::*;
use std::cell::RefCell;
use std::ffi::OsStr;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU8},
};
use std::thread::ThreadId;

const MAX_ACTOR_CALLS: usize = 2 * crate::MAP2_ACTOR_DECISIONS;
const _: () = assert!(MAX_ACTOR_CALLS <= 20_000);
const _: () = assert!(BATCH == 40);
const _: () = assert!(BATCH <= MODEL_TRAINING_BATCH);
// Admission includes warmup, and is never returned even if initialization fails.
static USED: AtomicBool = AtomicBool::new(false);
static OWNER: OnceLock<Owner> = OnceLock::new();
static STATS: OnceLock<ActorGraphStats> = OnceLock::new();
#[cfg(feature = "builtin")]
static TRACE: OnceLock<ActorTrace> = OnceLock::new();
thread_local! {
    static ACTIVE: RefCell<Option<ActorWorkspace>> = const { RefCell::new(None) };
}

struct Owner {
    lineage: NonZeroU64,
    thread: ThreadId,
    device: Device,
    shapes: [usize; 2],
    warmups: [AtomicU8; 2],
    captured: AtomicU8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActorGraphStats {
    pub(crate) main_batch: usize,
    pub(crate) shapes: [usize; 2],
    pub(crate) calls_by_batch: [u64; 65],
    pub(crate) hits_by_batch: [u64; 65],
    pub(crate) captures: usize,
    pub(crate) setup_ns: u64,
    pub(crate) retirement_ns: u64,
}

pub(crate) fn actor_graph_stats_for_test() -> Result<ActorGraphStats, ModelError> {
    STATS.get().copied().ok_or(ModelError::InvalidModelState(
        "actor graph stats were not recorded",
    ))
}

#[cfg(feature = "builtin")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActorTrace {
    pub(crate) hashes: [u64; BATCH],
    pub(crate) random: [(u64, u64); BATCH],
    pub(crate) decisions: [usize; BATCH],
    pub(crate) retained: [u32; BATCH],
}

#[cfg(feature = "builtin")]
pub(crate) fn record_actor_trace(trace: ActorTrace) -> Result<(), ModelError> {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return Err(ModelError::InvalidModelState(
            "actor graph trace requires an active scope",
        ));
    }
    TRACE
        .set(trace)
        .map_err(|_| ModelError::InvalidModelState("actor graph trace already recorded"))
}

#[cfg(feature = "builtin")]
pub(crate) fn actor_trace_for_test() -> Result<ActorTrace, ModelError> {
    TRACE.get().copied().ok_or(ModelError::InvalidModelState(
        "actor graph trace was not recorded",
    ))
}

pub(crate) fn parse_graph_mode(value: Option<&OsStr>) -> Result<Option<bool>, ModelError> {
    match value {
        None => Ok(None),
        Some(value) if value == "0" => Ok(Some(false)),
        Some(value) if value == "1" => Ok(Some(true)),
        Some(_) => Err(ModelError::InvalidModelState(
            "DRYSUA_PROBE_ACTOR_GRAPH must be 0 or 1",
        )),
    }
}

pub(super) fn admit(model: &PolicyModel) -> Result<(), ModelError> {
    admit_shapes(model, [BATCH, 0])
}

fn admit_shapes(model: &PolicyModel, shapes: [usize; 2]) -> Result<(), ModelError> {
    assert!(matches!(shapes[0], 20 | 40));
    assert!(shapes[1] <= 1);
    if USED.load(Ordering::Acquire) {
        return Err(ModelError::InvalidModelState(
            "actor graph admission already consumed",
        ));
    }
    if !model.tensor_device().is_cuda() {
        return Err(ModelError::InvalidModelState("actor graph requires CUDA"));
    }
    if model.creation_thread != std::thread::current().id() {
        return Err(ModelError::InvalidModelState(
            "actor graph owner thread mismatch",
        ));
    }
    let _lock = model.write_parameter_lock()?;
    if model.foreign_thread_used.load(Ordering::Acquire) {
        return Err(ModelError::InvalidModelState(
            "actor graph model was used by another thread",
        ));
    }
    let cuda = model.tensor_device().as_cuda_device()?;
    let stream = cuda.cuda_stream();
    if !cuda.is_event_tracking()
        || !stream.context().has_async_alloc()
        || stream.context().is_in_multi_stream_mode()
    {
        return Err(ModelError::InvalidModelState(
            "actor graph requires tracked single-owner async CUDA",
        ));
    }
    USED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| ModelError::InvalidModelState("actor graph admission already consumed"))?;
    OWNER
        .set(Owner {
            lineage: model.lineage,
            thread: std::thread::current().id(),
            device: model.tensor_device().clone(),
            shapes,
            warmups: std::array::from_fn(|_| AtomicU8::new(0)),
            captured: AtomicU8::new(0),
        })
        .map_err(|_| ModelError::InvalidModelState("actor graph owner already bound"))?;
    Ok(())
}

pub(super) fn authorize_shape(
    model: &PolicyModel,
    batch: usize,
    capture: bool,
) -> Result<(), ModelError> {
    let owner = OWNER.get().ok_or(ModelError::InvalidModelState(
        "actor graph has no admission",
    ))?;
    if owner.thread != std::thread::current().id()
        || owner.lineage != model.lineage
        || !owner.device.same_device(model.tensor_device())
    {
        return Err(ModelError::InvalidModelState(
            "actor graph cache owner mismatch",
        ));
    }
    let index = owner
        .shapes
        .iter()
        .position(|shape| *shape == batch && batch != 0)
        .ok_or(ModelError::InvalidModelState(
            "actor graph shape was not admitted",
        ))?;
    if !capture {
        return owner.warmups[index]
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < 2).then(|| count + 1)
            })
            .map(|_| ())
            .map_err(|_| ModelError::InvalidModelState("actor graph warmup budget exceeded"));
    }
    if owner.warmups[index].load(Ordering::Acquire) != 2 {
        return Err(ModelError::InvalidModelState(
            "actor graph requires two shape warmups",
        ));
    }
    let bit = 1u8 << index;
    if owner.captured.fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        return Err(ModelError::InvalidModelState(
            "actor graph shape capture already consumed",
        ));
    }
    Ok(())
}

pub(crate) fn check_owner(model: &PolicyModel) -> Result<(), ModelError> {
    if model.creation_thread != std::thread::current().id() {
        model.foreign_thread_used.store(true, Ordering::Release);
    }
    if OWNER.get().is_some_and(|owner| {
        owner.lineage == model.lineage && owner.thread != std::thread::current().id()
    }) {
        return Err(ModelError::InvalidModelState(
            "actor graph owner thread mismatch",
        ));
    }
    Ok(())
}

/// The callback cannot obtain the workspace or its tensor aliases. All retained
/// parameter tensors own their storage, and teardown precedes returning to PPO.
pub(crate) fn with_actor_graph_for_test<T>(
    model: &PolicyModel,
    enabled: bool,
    main_batch: usize,
    operation: impl FnOnce() -> T,
) -> Result<T, ModelError> {
    if !matches!(main_batch, 20 | 40) {
        return Err(ModelError::InvalidModelState(
            "actor graph main batch must be 20 or 40",
        ));
    }
    let shapes = [main_batch, 1];
    admit_shapes(model, shapes)?;
    let workspace = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ActorWorkspace::prepare(model, enabled, shapes)
    }))
    .unwrap_or_else(|_| fatal("actor graph initialization panicked"));
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none());
        *slot = Some(workspace);
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    let workspace = ACTIVE
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_else(|| fatal("actor graph workspace disappeared"));
    workspace.finish();
    match result {
        Ok(value) => Ok(value),
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

pub(in crate::model) fn sample_selection(
    model: &PolicyModel,
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    random: &mut [PpoRng],
) -> Result<Option<Vec<BatchSelection>>, ModelError> {
    ACTIVE.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| ModelError::InvalidModelState("actor graph recursive sampling"))?;
        let Some(workspace) = slot.as_mut() else {
            return Ok(None);
        };
        workspace.sample(model, frames, spaces, random).map(Some)
    })
}

pub(super) fn counts_for_test() -> Option<(usize, usize)> {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|workspace| (workspace.calls, workspace.replays))
    })
}

struct ActorWorkspace {
    slots: [GraphSlot; 2],
    // Graph nodes hold raw addresses, so pin every original parameter allocation.
    _parameters: [Tensor; MODEL_PARAMETER_TENSORS],
    device: Device,
    stream: Arc<CudaStream>,
    lineage: NonZeroU64,
    owner: ThreadId,
    calls: usize,
    replays: usize,
    stats: ActorGraphStats,
}

impl ActorWorkspace {
    fn prepare(model: &PolicyModel, enabled: bool, shapes: [usize; 2]) -> Self {
        let started = Instant::now();
        let stream = probe_stream(model);
        let parameters = model.parameters();
        assert_eq!(parameters.len(), MODEL_PARAMETER_TENSORS);
        let parameters = std::array::from_fn(|index| parameters[index].value.as_tensor().clone());
        let slots = shapes.map(|batch| GraphSlot::prepare(model, enabled, batch, &stream));
        let stats = ActorGraphStats {
            main_batch: shapes[0],
            shapes,
            calls_by_batch: [0; 65],
            hits_by_batch: [0; 65],
            captures: slots.iter().filter(|slot| slot.graph.is_some()).count(),
            setup_ns: u64::try_from(started.elapsed().as_nanos()).expect("bounded setup"),
            retirement_ns: 0,
        };
        eprintln!(
            "actor-graph-setup enabled={enabled} shapes={shapes:?} warmups=4 captures={} elapsed_ns={} metadata_entries_bound=274 scalar_entries_bound=8 cache_payload_bound_bytes=26336",
            stats.captures, stats.setup_ns
        );
        Self {
            slots,
            _parameters: parameters,
            device: model.tensor_device().clone(),
            stream,
            lineage: model.lineage,
            owner: std::thread::current().id(),
            calls: 0,
            replays: 0,
            stats,
        }
    }

    fn validate(&self, model: &PolicyModel) -> Result<(), ModelError> {
        if self.owner != std::thread::current().id() {
            return Err(ModelError::InvalidModelState(
                "actor graph owner thread mismatch",
            ));
        }
        if self.lineage != model.lineage {
            return Err(ModelError::InvalidModelState("actor graph model mismatch"));
        }
        if !self.device.same_device(model.tensor_device()) {
            return Err(ModelError::InvalidModelState("actor graph device mismatch"));
        }
        if self.calls >= MAX_ACTOR_CALLS {
            return Err(ModelError::InvalidModelState(
                "actor graph call budget exceeded",
            ));
        }
        Ok(())
    }

    fn sample(
        &mut self,
        model: &PolicyModel,
        frames: &[FeatureFrame],
        spaces: &[ActionSpace],
        random: &mut [PpoRng],
    ) -> Result<Vec<BatchSelection>, ModelError> {
        self.validate(model)?;
        assert!(!frames.is_empty());
        assert!(frames.len() <= MODEL_TRAINING_BATCH);
        self.calls += 1;
        self.stats.calls_by_batch[frames.len()] += 1;
        let result = if let Some(slot) = self
            .slots
            .iter()
            .find(|slot| slot.batch == frames.len() && slot.graph.is_some())
        {
            let state = slot.forward(frames, &self.device);
            self.replays += 1;
            self.stats.hits_by_batch[frames.len()] += 1;
            model.selection_from_state_locked(state, spaces, Some(random))
        } else {
            model.selection_batch_locked(frames, spaces, Some(random))
        };
        // Raw graph launch does not update Candle's allocation events. Finish all
        // consumers while sample_batch still owns its read lock and staged RNGs.
        required(self.stream.synchronize(), "actor heads completion");
        if let Err(ModelError::Backend(error)) = &result {
            fatal(error);
        }
        result
    }

    fn finish(mut self) {
        assert_eq!(self.owner, std::thread::current().id());
        assert!(self.replays <= self.calls);
        let started = Instant::now();
        let stream = self.stream.clone();
        required(stream.synchronize(), "actor graph retirement boundary");
        if required(stream.capture_status(), "actor retirement capture status")
            != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            fatal("actor stream remained in capture at retirement");
        }
        for slot in &mut self.slots {
            drop(slot.graph.take());
            required(stream.context().check_err(), "actor graph destruction");
        }
        let (calls, replays) = (self.calls, self.replays);
        let mut stats = self.stats;
        drop(self);
        required(stream.synchronize(), "actor buffer retirement");
        required(stream.context().check_err(), "actor retirement status");
        stats.retirement_ns =
            u64::try_from(started.elapsed().as_nanos()).expect("bounded retirement");
        assert_eq!(stats.calls_by_batch.iter().sum::<u64>(), calls as u64);
        assert_eq!(stats.hits_by_batch.iter().sum::<u64>(), replays as u64);
        STATS
            .set(stats)
            .unwrap_or_else(|_| fatal("actor graph stats already recorded"));
        eprintln!(
            "actor-graph-retired calls={calls} replays={replays} eager_calls={} elapsed_ns={} before_ppo=true",
            calls - replays,
            started.elapsed().as_nanos()
        );
        eprintln!(
            "actor-graph-histogram calls_by_batch={:?} hits_by_batch={:?}",
            stats.calls_by_batch, stats.hits_by_batch
        );
    }
}

struct GraphSlot {
    batch: usize,
    graph: Option<CudaGraph>,
    input: Var,
    _inputs: EncoderInputs,
    outputs: [Var; 3],
}

impl GraphSlot {
    fn prepare(model: &PolicyModel, enabled: bool, batch: usize, stream: &Arc<CudaStream>) -> Self {
        assert!(matches!(batch, 1 | 20 | 40));
        let sources = [0.0, 0.25].map(|shift| {
            required(
                stage_frame_buffer(&shape_frames(batch, shift), model.tensor_device()),
                "actor warmup staging",
            )
        });
        assert_eq!(sources[0].1, sources[1].1);
        let input = required(Var::from_tensor(&sources[0].0), "actor fixed input");
        let inputs = required(
            EncoderInputs::from_buffer(&input.as_detached_tensor(), &sources[0].1, batch),
            "actor fixed views",
        );
        let outputs = shape_output_buffers(model.tensor_device(), batch);
        warm_encoders(model, &inputs, &input, &outputs, &sources, stream, enabled);
        required(input.set(&sources[0].0), "actor capture input");
        let graph = enabled.then(|| capture_once(model, &inputs, &outputs, stream));
        eprintln!(
            "actor-graph-slot batch={batch} input_bytes={} output_bytes={} captured={enabled}",
            input.elem_count() * std::mem::size_of::<f32>(),
            outputs
                .iter()
                .map(|output| output.elem_count() * std::mem::size_of::<f32>())
                .sum::<usize>()
        );
        Self {
            batch,
            graph,
            input,
            _inputs: inputs,
            outputs,
        }
    }

    fn forward(&self, frames: &[FeatureFrame], device: &Device) -> ForwardState {
        assert_eq!(frames.len(), self.batch);
        let (uploaded, lengths) =
            required(stage_frame_buffer(frames, device), "actor dynamic upload");
        assert_eq!(lengths.len(), 20);
        assert_eq!(uploaded.elem_count(), self.input.elem_count());
        required(self.input.set(&uploaded), "actor input copy");
        required(
            self.graph.as_ref().expect("captured slot").launch(),
            "actor replay",
        );
        ForwardState {
            trunk: self.outputs[0].as_detached_tensor(),
            current_units: self.outputs[1].as_detached_tensor(),
            points: self.outputs[2].as_detached_tensor(),
        }
    }
}
