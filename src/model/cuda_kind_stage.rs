//! Single-admission B20 kind-prefix experiment, not a production sampler or graph cache.
use super::*;
use std::marker::PhantomData;
use std::rc::Rc;

#[path = "../tests/cuda_kind_stage.rs"]
mod tests;

const STAGE_BATCH: usize = 20;
const PACKED_ELEMENTS: usize = STAGE_BATCH * 24;
const MAX_REPLAYS: usize = 128;
const _: () = assert!(STAGE_BATCH <= MODEL_TRAINING_BATCH);
const _: () = assert!(PACKED_ELEMENTS == 480);

struct StageBuffers {
    trunk: Var,
    kinds: Var,
    sides: Var,
    output: Var,
}

struct KindStage<'model> {
    model: &'model PolicyModel,
    graph: Option<CudaGraph>,
    buffers: StageBuffers,
    stream: Arc<CudaStream>,
    _parameters: [Tensor; MODEL_PARAMETER_TENSORS],
    _owner: PhantomData<Rc<()>>,
    replays: usize,
}

impl<'model> KindStage<'model> {
    fn new(model: &'model PolicyModel) -> Result<Self, ModelError> {
        actor::admit_kind_prefix(model)?;
        Ok(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::prepare(model)))
                .unwrap_or_else(|_| fatal("kind stage preparation panicked")),
        )
    }

    fn prepare(model: &'model PolicyModel) -> Self {
        let started = Instant::now();
        let device = model.tensor_device();
        let stream = probe_stream(model);
        let buffers = StageBuffers {
            trunk: required(
                Var::zeros((STAGE_BATCH, TRUNK_WIDTH), DType::F32, device),
                "kind stage trunk",
            ),
            kinds: required(
                Var::zeros(STAGE_BATCH, DType::U32, device),
                "kind stage indices",
            ),
            sides: required(
                Var::zeros((STAGE_BATCH, 1), DType::U8, device),
                "kind stage mask",
            ),
            output: required(
                Var::zeros(PACKED_ELEMENTS, DType::F32, device),
                "kind stage output",
            ),
        };
        let parameters = model.parameters();
        assert_eq!(parameters.len(), MODEL_PARAMETER_TENSORS);
        let parameters = std::array::from_fn(|index| parameters[index].value.as_tensor().clone());
        let mut stage = Self {
            model,
            graph: None,
            buffers,
            stream,
            _parameters: parameters,
            _owner: PhantomData,
            replays: 0,
        };
        for warmup in 0..2 {
            let trunk = required(
                Tensor::full(warmup as f32, (STAGE_BATCH, TRUNK_WIDTH), device),
                "kind warmup trunk",
            );
            let kinds = std::array::from_fn::<_, STAGE_BATCH, _>(|row| {
                ((row + warmup) % MODEL_KIND_HEAD) as u32
            });
            let sides = std::array::from_fn::<_, STAGE_BATCH, _>(|row| ((row + warmup) % 2) as u8);
            required(stage.upload(&trunk, &kinds, &sides), "kind warmup inputs");
            let _lock = required(model.read_parameter_lock(), "kind warmup lock");
            required(
                actor::authorize_kind_prefix(model, false),
                "kind warmup budget",
            );
            {
                let _cache =
                    required(device.as_cuda_device(), "kind CUDA").enable_cuda_graph_htod_cache();
                required(stage_into(model, &stage.buffers), "kind warmup body");
            }
            required(stage.stream.synchronize(), "kind warmup completion");
        }
        required(
            actor::authorize_kind_prefix(model, true),
            "kind capture budget",
        );
        stage.graph = Some(capture_tensor_body(model, &stage.stream, || {
            stage_into(model, &stage.buffers)
        }));
        eprintln!(
            "kind-stage setup_ns={} batch={STAGE_BATCH} captures=1 warmups=2 fixed_buffer_bytes=22500 metadata_keys_bound=5 metadata_payload_bound_bytes=240 host_data_cache_entries=0 production_qualified=false",
            started.elapsed().as_nanos()
        );
        stage
    }

    fn upload(&self, trunk: &Tensor, kinds: &[u32], sides: &[u8]) -> Result<(), ModelError> {
        validate_inputs(kinds, sides)?;
        self.validate_trunk(trunk)?;
        let device = self.model.tensor_device();
        // All changing host values are uploaded before entering any cache guard.
        self.buffers.trunk.set(trunk)?;
        self.buffers
            .kinds
            .set(&Tensor::from_slice(kinds, STAGE_BATCH, device)?)?;
        self.buffers
            .sides
            .set(&Tensor::from_slice(sides, (STAGE_BATCH, 1), device)?)?;
        Ok(())
    }

    fn validate_trunk(&self, trunk: &Tensor) -> Result<(), ModelError> {
        actor::check_owner(self.model)?;
        if trunk.dims() != [STAGE_BATCH, TRUNK_WIDTH] || trunk.dtype() != DType::F32 {
            return Err(ModelError::InvalidModelState(
                "kind stage trunk shape or dtype",
            ));
        }
        let device = self.model.tensor_device();
        if !trunk.device().same_device(device) {
            return Err(ModelError::InvalidModelState("kind stage device mismatch"));
        }
        Ok(())
    }

    fn run_locked(
        &mut self,
        trunk: &Tensor,
        kinds: &[u32],
        sides: &[u8],
    ) -> Result<SamplingKindLogits, ModelError> {
        validate_inputs(kinds, sides)?;
        self.validate_trunk(trunk)?;
        let prefixes = kinds
            .iter()
            .map(|kind| {
                TrainingPrefix::new(
                    ActionKind::from_index(*kind as usize).expect("validated kind"),
                    None,
                    None,
                )
            })
            .collect::<Vec<_>>();
        let needed = sampling::kind_needed(&prefixes);
        if !needed.contains(&true) {
            return Ok(SamplingKindLogits::default());
        }
        if self.replays >= MAX_REPLAYS {
            return Err(ModelError::InvalidModelState(
                "kind stage replay budget exceeded",
            ));
        }
        if let Err(error) = self.upload(trunk, kinds, sides) {
            if let ModelError::Backend(message) = &error {
                fatal(message);
            }
            return Err(error);
        }
        required(
            self.graph.as_ref().expect("captured kind stage").launch(),
            "kind stage replay",
        );
        self.replays += 1;
        let values = required(
            self.buffers.output.to_vec1::<f32>(),
            "kind stage packed readback",
        );
        required(self.stream.synchronize(), "kind stage consumer completion");
        unpack(&values, needed)
    }

    fn replays(&self) -> usize {
        self.replays
    }

    #[cfg(feature = "builtin")]
    fn sample_batch(
        &mut self,
        frames: &[FeatureFrame],
        spaces: &[ActionSpace],
        rngs: &mut [PpoRng],
    ) -> Result<Vec<PpoPolicyChoice>, ModelError> {
        let result = self.sample_batch_inner(frames, spaces, rngs);
        if let Err(ModelError::Backend(message)) = &result {
            fatal(message);
        }
        result
    }

    #[cfg(feature = "builtin")]
    fn sample_batch_inner(
        &mut self,
        frames: &[FeatureFrame],
        spaces: &[ActionSpace],
        rngs: &mut [PpoRng],
    ) -> Result<Vec<PpoPolicyChoice>, ModelError> {
        validate_policy_batch(frames, spaces)?;
        validate_sampling_rng_count(frames.len(), rngs.len())?;
        if frames.len() != STAGE_BATCH {
            return Err(ModelError::InvalidModelState("kind stage actor batch"));
        }
        let model = self.model;
        let mut staged = rngs.to_vec();
        let mut random = Some(staged.as_mut_slice());
        let _guard = model.read_parameter_lock()?;
        let state = model.forward_frames(frames)?;
        let routing = ActorRouting::new(frames, model.tensor_device(), false)?;
        let base = model.sampling_base_logits(&state, &routing)?;
        let mut rows = initialize_sampling_rows(&base, spaces, &mut random)?;
        let kinds = rows
            .iter()
            .map(|row| row.prefix.kind().index() as u32)
            .collect::<Vec<_>>();
        let sides = frames
            .iter()
            .map(|frame| u8::from(frame.global[crate::global_feature::SIDE_RADIANT] == 1.0))
            .collect::<Vec<_>>();
        let kind = self.run_locked(&state.trunk, &kinds, &sides)?;
        select_sampling_units(&mut rows, &kind, spaces, &mut random)?;
        let unit = model.sampling_unit_logits(&state, &sampling_prefixes(&rows), &routing)?;
        select_sampling_slots(&mut rows, &unit, spaces, &mut random)?;
        let slot = model.sampling_slot_logits(&state, &sampling_prefixes(&rows), &routing)?;
        let selected = decode_batch_rows(
            spaces,
            &mut random,
            rows,
            SamplingLogits {
                base,
                kind,
                unit,
                slot,
            },
        )?;
        let choices =
            finish_sampled_choices(frames, spaces, selected, model.policy_identity_locked())?;
        required(
            self.stream.synchronize(),
            "kind stage sampling before RNG commit",
        );
        rngs.clone_from_slice(&staged);
        Ok(choices)
    }

    fn finish(mut self) {
        required(
            actor::check_owner(self.model),
            "kind stage retirement owner",
        );
        let stream = self.stream.clone();
        required(stream.synchronize(), "kind stage retirement boundary");
        if required(stream.capture_status(), "kind stage capture status")
            != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            fatal("kind stage capture remained active");
        }
        drop(self.graph.take());
        required(stream.context().check_err(), "kind stage graph destruction");
        let replays = self.replays;
        drop(self);
        required(stream.synchronize(), "kind stage buffer retirement");
        required(
            stream.context().check_err(),
            "kind stage retirement completion",
        );
        eprintln!(
            "kind-stage retired=true replays={replays} readbacks={replays} cache_bound_bytes=240"
        );
    }
}

impl Drop for KindStage<'_> {
    fn drop(&mut self) {
        if self.graph.is_some() {
            fatal("kind stage dropped without checked finish; retiring owned test process");
        }
    }
}

fn stage_into(model: &PolicyModel, buffers: &StageBuffers) -> Result<(), ModelError> {
    let kind = model.kind_embedding_from_indices(&buffers.kinds.as_detached_tensor())?;
    let context = model.kind_context_from_embedding(&buffers.trunk.as_detached_tensor(), &kind)?;
    let controlled = model.raw_actor_pair(ActorHead::Controlled, &context)?;
    let learn = model.raw_actor_pair(ActorHead::Learn, &context)?;
    let mask = buffers.sides.as_detached_tensor();
    let selected_controlled = mask
        .broadcast_as(controlled[0].shape())?
        .where_cond(&controlled[0], &controlled[1])?;
    let selected_learn = mask
        .broadcast_as(learn[0].shape())?
        .where_cond(&learn[0], &learn[1])?;
    let parts = [
        &controlled[0],
        &controlled[1],
        &learn[0],
        &learn[1],
        &selected_controlled,
        &selected_learn,
    ]
    .map(Tensor::flatten_all)
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let packed = Tensor::cat(&parts, 0)?;
    assert_eq!(packed.elem_count(), PACKED_ELEMENTS);
    assert_eq!(packed.layout().start_offset(), 0);
    buffers.output.set(&packed)?;
    Ok(())
}

fn validate_inputs(kinds: &[u32], sides: &[u8]) -> Result<(), ModelError> {
    if kinds.len() != STAGE_BATCH || sides.len() != STAGE_BATCH {
        return Err(ModelError::InvalidModelState("kind stage input counts"));
    }
    if kinds.iter().any(|kind| *kind >= MODEL_KIND_HEAD as u32) {
        return Err(ModelError::InvalidModelState("kind stage kind index"));
    }
    if sides.iter().any(|side| *side > 1) {
        return Err(ModelError::InvalidModelState("kind stage side mask"));
    }
    Ok(())
}

fn unpack(values: &[f32], needed: [bool; 2]) -> Result<SamplingKindLogits, ModelError> {
    if values.len() != PACKED_ELEMENTS {
        return Err(ModelError::InvalidModelState("kind stage output count"));
    }
    for (required, start, width, field) in [
        (needed[0], 0, 2, "radiant.controlled"),
        (needed[0], 40, 2, "dire.controlled"),
        (needed[1], 80, 6, "radiant.learn"),
        (needed[1], 200, 6, "dire.learn"),
        (needed[0], 320, 2, "controlled"),
        (needed[1], 360, 6, "learn"),
    ] {
        if required {
            let part = &values[start..start + STAGE_BATCH * width];
            if let Some(index) = part.iter().position(|value| !value.is_finite()) {
                return Err(ModelError::NonFiniteOutput {
                    field,
                    batch: index / width,
                    index: index % width,
                });
            }
        }
    }
    Ok(SamplingKindLogits {
        controlled: needed[0].then(|| {
            values[320..360]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| chunk.to_vec())
                .collect()
        }),
        learn: needed[1].then(|| {
            values[360..480]
                .as_chunks::<6>()
                .0
                .iter()
                .map(|chunk| chunk.to_vec())
                .collect()
        }),
    })
}
