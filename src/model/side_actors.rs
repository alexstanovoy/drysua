//! Full-batch actor routing; the shared encoder and value head never branch.
use super::*;
#[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
pub(super) struct ActorHeads([Linear; 12]);

#[cfg(feature = "side-actors")]
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
    #[cfg(feature = "side-actors")]
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
    #[cfg(feature = "side-actors")]
    mask: Tensor,
    #[cfg(feature = "side-actors")]
    training: bool,
    #[cfg(feature = "side-actors")]
    raw: RefCell<[Option<[Tensor; 2]>; 12]>,
}

impl ActorRouting {
    pub(super) fn new(
        frames: &[FeatureFrame],
        device: &Device,
        training: bool,
    ) -> Result<Self, ModelError> {
        #[cfg(feature = "side-actors")]
        {
            assert!(!frames.is_empty());
            assert!(frames.len() <= MODEL_PPO_MAX_MICROBATCH);
            validate_sides(frames)?;
            let rows = frames
                .iter()
                .enumerate()
                .map(|(index, frame)| side_row(frame, index))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                mask: Tensor::from_vec(rows, (frames.len(), 1), device)?,
                training,
                raw: RefCell::new(std::array::from_fn(|_| None)),
            })
        }
        #[cfg(not(feature = "side-actors"))]
        {
            let _ = (frames, device, training);
            Ok(Self {})
        }
    }

    pub(super) fn forward(
        &self,
        model: &PolicyModel,
        head: ActorHead,
        input: &Tensor,
    ) -> Result<Tensor, ModelError> {
        #[cfg(not(feature = "side-actors"))]
        let radiant = model.radiant_head(head).forward(input)?;
        #[cfg(feature = "side-actors")]
        {
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
        #[cfg(not(feature = "side-actors"))]
        {
            Ok(radiant)
        }
    }

    #[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
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

#[cfg(feature = "side-actors")]
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

/// Expands a separately authenticated M24 vector without changing any source bit.
#[cfg(feature = "side-actors")]
pub(crate) fn expand_m24_side_actor_parameters(source: &[f32]) -> Result<Vec<f32>, ModelError> {
    if source.len() != LEGACY_MODEL_PARAMETER_COUNT {
        return Err(ModelError::ParameterLength {
            actual: source.len(),
            expected: LEGACY_MODEL_PARAMETER_COUNT,
        });
    }
    if let Some(index) = source.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteParameter { index });
    }
    const KIND: std::ops::Range<usize> = 1_586_113..1_590_225;
    const OTHER: std::ops::Range<usize> = 1_591_169..1_700_020;
    const _: () = assert!(
        MODEL_PARAMETER_COUNT - LEGACY_MODEL_PARAMETER_COUNT
            == (KIND.end - KIND.start) + (OTHER.end - OTHER.start)
    );
    let mut target = Vec::with_capacity(MODEL_PARAMETER_COUNT);
    target.extend_from_slice(source);
    target.extend_from_slice(&source[KIND]);
    target.extend_from_slice(&source[OTHER]);
    assert_eq!(target.len(), MODEL_PARAMETER_COUNT);
    assert_eq!(MODEL_PARAMETER_COUNT, 1_812_983);
    Ok(target)
}

#[cfg(all(test, feature = "side-actors"))]
thread_local! { static ENCODER_FORWARDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(all(test, feature = "side-actors"))]
pub(super) fn record_encoder_forward() {
    ENCODER_FORWARDS.set(
        ENCODER_FORWARDS
            .get()
            .checked_add(1)
            .expect("bounded encoder counter"),
    );
}

#[cfg(all(test, feature = "side-actors"))]
pub(crate) fn take_encoder_forwards_for_test() -> usize {
    ENCODER_FORWARDS.replace(0)
}
