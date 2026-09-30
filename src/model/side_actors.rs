//! Full-batch actor routing; the shared encoder and value head never branch.
use super::*;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub(super) enum ActorHead {
    Kind,
    Controlled,
    Ability,
    Item,
    Swap,
    Learn,
    Shop,
    Loot,
    TargetMode,
    PutMode,
    EntityQuery,
    PointQuery,
}

impl ActorHead {
    const ALL: [Self; 12] = [
        Self::Kind,
        Self::Controlled,
        Self::Ability,
        Self::Item,
        Self::Swap,
        Self::Learn,
        Self::Shop,
        Self::Loot,
        Self::TargetMode,
        Self::PutMode,
        Self::EntityQuery,
        Self::PointQuery,
    ];

    fn fields(self) -> [&'static str; 2] {
        [
            ["radiant.kind", "dire.kind"],
            ["radiant.controlled", "dire.controlled"],
            ["radiant.ability", "dire.ability"],
            ["radiant.item", "dire.item"],
            ["radiant.swap", "dire.swap"],
            ["radiant.learn", "dire.learn"],
            ["radiant.shop", "dire.shop"],
            ["radiant.loot", "dire.loot"],
            ["radiant.target_mode", "dire.target_mode"],
            ["radiant.put_mode", "dire.put_mode"],
            ["radiant.entity_query", "dire.entity_query"],
            ["radiant.point_query", "dire.point_query"],
        ][self as usize]
    }

    fn width(self) -> usize {
        [16, 2, 8, 15, 15, 6, 64, 16, 3, 2, 128, 64][self as usize]
    }
}

pub(super) struct ActorHeads([Linear; 12]);

impl ActorHeads {
    pub(super) fn fresh(generator: &mut Initializer, device: &Device) -> Result<Self, ModelError> {
        let mut heads = Vec::with_capacity(12);
        for head in ActorHead::ALL {
            let input = if matches!(head, ActorHead::Kind) {
                TRUNK_WIDTH
            } else {
                DECODER_CONTEXT
            };
            heads.push(Linear::fresh(input, head.width(), generator, device)?);
        }
        assert_eq!(heads.len(), 12);
        Ok(Self(heads.try_into().map_err(|_| {
            ModelError::InvalidModelState("side actor head count")
        })?))
    }

    pub(super) fn parameters<'a>(&'a self, output: &mut Vec<NamedParameter<'a>>) {
        let names = [
            ("dire.kind.weight", "dire.kind.bias"),
            ("dire.controlled.weight", "dire.controlled.bias"),
            ("dire.ability_head.weight", "dire.ability_head.bias"),
            ("dire.item_head.weight", "dire.item_head.bias"),
            ("dire.swap_head.weight", "dire.swap_head.bias"),
            ("dire.learn_head.weight", "dire.learn_head.bias"),
            ("dire.shop_head.weight", "dire.shop_head.bias"),
            ("dire.loot_head.weight", "dire.loot_head.bias"),
            ("dire.target_mode.weight", "dire.target_mode.bias"),
            ("dire.put_mode.weight", "dire.put_mode.bias"),
            ("dire.entity_query.weight", "dire.entity_query.bias"),
            ("dire.point_query.weight", "dire.point_query.bias"),
        ];
        assert_eq!(self.0.len(), names.len());
        for (head, names) in self.0.iter().zip(names) {
            head.parameters(names, output);
        }
    }
}

impl PolicyModel {
    pub(super) fn raw_actor_pair(
        &self,
        head: ActorHead,
        input: &Tensor,
    ) -> Result<[Tensor; 2], ModelError> {
        Ok([
            self.radiant_head(head).forward(input)?,
            self.dire.0[head as usize].forward(input)?,
        ])
    }

    fn radiant_head(&self, head: ActorHead) -> &Linear {
        match head {
            ActorHead::Kind => &self.kind,
            ActorHead::Controlled => &self.controlled,
            ActorHead::Ability => &self.ability_head,
            ActorHead::Item => &self.item_head,
            ActorHead::Swap => &self.swap_head,
            ActorHead::Learn => &self.learn_head,
            ActorHead::Shop => &self.shop_head,
            ActorHead::Loot => &self.loot_head,
            ActorHead::TargetMode => &self.target_mode,
            ActorHead::PutMode => &self.put_mode,
            ActorHead::EntityQuery => &self.entity_query,
            ActorHead::PointQuery => &self.point_query,
        }
    }
}

pub(super) struct ActorRouting {
    mask: Tensor,
    /// Host copy of `mask`: one means the radiant head, zero the dire head.
    sides: Vec<u8>,
    training: bool,
    raw: RefCell<[Option<[Tensor; 2]>; 12]>,
}

/// Queued raw radiant and dire outputs of one actor head inside a [`StageRequest`].
#[derive(Clone, Copy)]
pub(super) struct QueuedPair {
    head: ActorHead,
    radiant: usize,
    dire: usize,
}

/// Row-major `[batch, width]` tensors of one sampling stage, read back with one device
/// synchronization instead of one readback and one finiteness check per head.
pub(super) struct StageRequest {
    batch: usize,
    tensors: Vec<Tensor>,
}

/// Host values of one [`StageRequest`], in queue order.
pub(super) struct StageValues {
    batch: usize,
    values: Vec<f32>,
    ranges: Vec<(usize, usize)>,
}

impl StageRequest {
    pub(super) fn new(batch: usize) -> Self {
        assert!((1..=MODEL_PPO_MAX_MICROBATCH).contains(&batch));
        Self {
            batch,
            tensors: Vec::with_capacity(2 * ActorHead::ALL.len() + 3),
        }
    }

    /// Queues one `[batch, width]` tensor and returns its index.
    pub(super) fn push(&mut self, tensor: Tensor) -> Result<usize, ModelError> {
        let (rows, width) = tensor.dims2()?;
        assert_eq!(rows, self.batch);
        assert!((1..=UNIT_EMBEDDING).contains(&width));
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(self.tensors.len() < 2 * ActorHead::ALL.len() + 3);
        self.tensors.push(tensor);
        Ok(self.tensors.len() - 1)
    }

    pub(super) fn read(self) -> Result<StageValues, ModelError> {
        assert!(!self.tensors.is_empty());
        let mut ranges = Vec::with_capacity(self.tensors.len());
        let mut total = 0usize;
        let mut flat = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let count = tensor.elem_count();
            ranges.push((total, tensor.dim(1)?));
            total += count;
            flat.push(tensor.detach().flatten_all()?);
        }
        assert!(total <= MODEL_PPO_MAX_MICROBATCH * 1_200);
        let packed = if flat.len() == 1 {
            flat.pop().expect("one tensor").force_contiguous()?
        } else {
            Tensor::cat(&flat, 0)?
        };
        let values = packed.to_vec1::<f32>()?;
        assert_eq!(values.len(), total);
        Ok(StageValues {
            batch: self.batch,
            values,
            ranges,
        })
    }
}

impl StageValues {
    fn slice(&self, index: usize) -> (&[f32], usize) {
        let (offset, width) = self.ranges[index];
        (&self.values[offset..offset + self.batch * width], width)
    }

    /// One queued tensor split into host rows.
    pub(super) fn rows(&self, index: usize) -> Vec<Vec<f32>> {
        let (values, width) = self.slice(index);
        values.chunks_exact(width).map(<[f32]>::to_vec).collect()
    }

    /// One queued `[batch, 1]` tensor as a column.
    pub(super) fn column(&self, index: usize) -> Vec<f32> {
        let (values, width) = self.slice(index);
        assert_eq!(width, 1);
        values.to_vec()
    }

    fn validate(&self, index: usize, field: &'static str) -> Result<(), ModelError> {
        let (values, width) = self.slice(index);
        match values.iter().position(|value| !value.is_finite()) {
            Some(position) => Err(ModelError::NonFiniteOutput {
                field,
                batch: position / width,
                index: position % width,
            }),
            None => Ok(()),
        }
    }
}

impl ActorRouting {
    pub(super) fn new(
        frames: &[FeatureFrame],
        device: &Device,
        training: bool,
    ) -> Result<Self, ModelError> {
        assert!(!frames.is_empty());
        assert!(frames.len() <= MODEL_PPO_MAX_MICROBATCH);
        validate_sides(frames)?;
        let rows = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| side_row(frame, index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            mask: Tensor::from_slice(&rows, (frames.len(), 1), device)?,
            sides: rows,
            training,
            raw: RefCell::new(std::array::from_fn(|_| None)),
        })
    }

    /// Queues both raw outputs of one actor head for host-side validation and selection.
    pub(super) fn queue_pair(
        &self,
        model: &PolicyModel,
        head: ActorHead,
        input: &Tensor,
        request: &mut StageRequest,
    ) -> Result<QueuedPair, ModelError> {
        assert!(!self.training);
        let [radiant, dire] = model.raw_actor_pair(head, input)?;
        assert_eq!(radiant.dims(), dire.dims());
        Ok(QueuedPair {
            head,
            radiant: request.push(radiant)?,
            dire: request.push(dire)?,
        })
    }

    /// Queues one pointer head: its raw query pair for validation and the scores of the
    /// device-selected query against `tokens`.
    pub(super) fn queue_pointer(
        &self,
        model: &PolicyModel,
        head: ActorHead,
        input: &Tensor,
        tokens: &Tensor,
        request: &mut StageRequest,
    ) -> Result<(QueuedPair, usize), ModelError> {
        assert!(!self.training);
        let [radiant, dire] = model.raw_actor_pair(head, input)?;
        assert_eq!(radiant.dims(), dire.dims());
        assert_eq!(radiant.dim(0)?, self.mask.dim(0)?);
        let query = self
            .mask
            .broadcast_as(radiant.shape())?
            .where_cond(&radiant, &dire)?
            .unsqueeze(1)?;
        let scores = scaled_pointer_dot(tokens, &query)?;
        let pair = QueuedPair {
            head,
            radiant: request.push(radiant)?,
            dire: request.push(dire)?,
        };
        Ok((pair, request.push(scores)?))
    }

    /// Validates one queued pair exactly like `forward` and returns the side-selected rows.
    pub(super) fn select(
        &self,
        values: &StageValues,
        pair: QueuedPair,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.validate(values, pair)?;
        let (radiant, width) = values.slice(pair.radiant);
        let (dire, _) = values.slice(pair.dire);
        assert_eq!(radiant.len(), self.sides.len() * width);
        Ok(self
            .sides
            .iter()
            .zip(radiant.chunks_exact(width).zip(dire.chunks_exact(width)))
            .map(|(&side, (radiant, dire))| if side == 1 { radiant } else { dire }.to_vec())
            .collect())
    }

    /// Validates one queued pair in the radiant-then-dire order of `forward`.
    pub(super) fn validate(
        &self,
        values: &StageValues,
        pair: QueuedPair,
    ) -> Result<(), ModelError> {
        let names = pair.head.fields();
        values.validate(pair.radiant, names[0])?;
        values.validate(pair.dire, names[1])
    }

    pub(super) fn forward(
        &self,
        model: &PolicyModel,
        head: ActorHead,
        input: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let [radiant, dire] = model.raw_actor_pair(head, input)?;
        assert_eq!(radiant.dims(), dire.dims());
        assert_eq!(radiant.dim(0)?, self.mask.dim(0)?);
        if !self.training {
            let names = head.fields();
            validate_named(&[(names[0], &radiant), (names[1], &dire)])?;
        }
        let selected = self
            .mask
            .broadcast_as(radiant.shape())?
            .where_cond(&radiant, &dire)?;
        if self.training {
            let mut raw = self.raw.borrow_mut();
            assert!(raw[head as usize].is_none());
            raw[head as usize] = Some([radiant, dire]);
        }
        Ok(selected)
    }

    pub(super) fn into_raw(self) -> Result<[[Tensor; 2]; 12], ModelError> {
        let mut raw = Vec::with_capacity(12);
        for pair in self.raw.into_inner() {
            raw.push(pair.ok_or(ModelError::InvalidModelState("missing side actor raw head"))?);
        }
        assert_eq!(raw.len(), 12);
        raw.try_into()
            .map_err(|_| ModelError::InvalidModelState("side actor raw head count"))
    }
}

pub(super) fn validate_sides(frames: &[FeatureFrame]) -> Result<(), ModelError> {
    assert!(frames.len() <= MODEL_MAX_BATCH);
    for (index, frame) in frames.iter().enumerate() {
        if !frame.is_finite() {
            return Err(ModelError::NonFiniteFrame { index });
        }
    }
    for (index, frame) in frames.iter().enumerate() {
        side_row(frame, index)?;
    }
    Ok(())
}

fn side_row(frame: &FeatureFrame, index: usize) -> Result<u8, ModelError> {
    let radiant = frame.global[crate::global_feature::SIDE_RADIANT];
    let dire = frame.global[crate::global_feature::SIDE_DIRE];
    if radiant.to_bits() == 1.0f32.to_bits() && dire.to_bits() & 0x7fff_ffff == 0 {
        return Ok(1);
    }
    if radiant.to_bits() & 0x7fff_ffff == 0 && dire.to_bits() == 1.0f32.to_bits() {
        return Ok(0);
    }
    Err(ModelError::InvalidSideOneHot {
        index,
        radiant_bits: radiant.to_bits(),
        dire_bits: dire.to_bits(),
    })
}

pub(super) fn validate_training(output: &PolicyTensorTensors) -> Result<(), ModelError> {
    let mut named = Vec::with_capacity(27);
    named.push(("value", &output.value));
    for (head, pair) in ActorHead::ALL.into_iter().zip(&output.side_raw) {
        let names = head.fields();
        named.push((names[0], &pair[0]));
        named.push((names[1], &pair[1]));
    }
    named.push(("entity pointer", &output.entity_pointer));
    named.push(("point pointer", &output.point_pointer));
    assert_eq!(named.len(), 27);
    validate_named(&named)
}

fn validate_named(named: &[(&'static str, &Tensor)]) -> Result<(), ModelError> {
    assert!(!named.is_empty());
    assert!(named.len() <= 27);
    let batch = named[0].1.dim(0)?;
    assert!((1..=MODEL_PPO_MAX_MICROBATCH).contains(&batch));
    let mut total = 0usize;
    for (_, tensor) in named {
        let (rows, width) = tensor.dims2()?;
        assert_eq!(rows, batch);
        assert!((1..=UNIT_EMBEDDING).contains(&width));
        assert_eq!(tensor.dtype(), DType::F32);
        total += tensor.elem_count();
    }
    assert!(total <= MODEL_PPO_MAX_MICROBATCH * 823);
    if named[0].1.device().is_cpu() {
        for &(field, tensor) in named {
            validate_tensor_finite(field, tensor)?;
        }
        return Ok(());
    }
    let flat = named
        .iter()
        .map(|(_, tensor)| tensor.detach().flatten_all())
        .collect::<Result<Vec<_>, _>>()?;
    let packed = if flat.len() == 1 {
        flat[0].force_contiguous()?
    } else {
        Tensor::cat(&flat, 0)?
    };
    assert_eq!(packed.elem_count(), total);
    assert_eq!(packed.layout().start_offset(), 0);
    let values = packed.to_vec1::<f32>()?;
    assert_eq!(values.len(), total);
    let mut offset = 0;
    for &(field, tensor) in named {
        let end = offset + tensor.elem_count();
        if let Some(index) = values[offset..end]
            .iter()
            .position(|value| !value.is_finite())
        {
            let width = tensor.dim(1)?;
            return Err(ModelError::NonFiniteOutput {
                field,
                batch: index / width,
                index: index % width,
            });
        }
        offset = end;
    }
    assert_eq!(offset, total);
    Ok(())
}

#[cfg(test)]
thread_local! { static ENCODER_FORWARDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
pub(super) fn record_encoder_forward() {
    ENCODER_FORWARDS.set(
        ENCODER_FORWARDS
            .get()
            .checked_add(1)
            .expect("bounded encoder counter"),
    );
}

#[cfg(test)]
pub(crate) fn take_encoder_forwards_for_test() -> usize {
    ENCODER_FORWARDS.replace(0)
}
