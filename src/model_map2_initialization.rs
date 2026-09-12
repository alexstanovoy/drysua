use super::{DType, MODEL_PARAMETER_COUNT, ModelError, PolicyModel, Tensor};

pub(super) const M14_PARAMETER_COUNT: usize = 1_689_076;
const M14_LAYOUT: [(&str, &[usize]); 62] = [
    ("unit.0.weight", &[73, 64]),
    ("unit.0.bias", &[64]),
    ("unit.1.weight", &[64, 128]),
    ("unit.1.bias", &[128]),
    ("unit.2.weight", &[128, 128]),
    ("unit.2.bias", &[128]),
    ("ability.0.weight", &[24, 64]),
    ("ability.0.bias", &[64]),
    ("ability.1.weight", &[64, 64]),
    ("ability.1.bias", &[64]),
    ("item.0.weight", &[28, 64]),
    ("item.0.bias", &[64]),
    ("item.1.weight", &[64, 64]),
    ("item.1.bias", &[64]),
    ("point.0.weight", &[32, 64]),
    ("point.0.bias", &[64]),
    ("point.1.weight", &[64, 64]),
    ("point.1.bias", &[64]),
    ("projectile.0.weight", &[20, 64]),
    ("projectile.0.bias", &[64]),
    ("projectile.1.weight", &[64, 64]),
    ("projectile.1.bias", &[64]),
    ("loot.0.weight", &[16, 64]),
    ("loot.0.bias", &[64]),
    ("loot.1.weight", &[64, 64]),
    ("loot.1.bias", &[64]),
    ("trunk.0.weight", &[2576, 512]),
    ("trunk.0.bias", &[512]),
    ("trunk.1.weight", &[512, 256]),
    ("trunk.1.bias", &[256]),
    ("trunk.2.weight", &[256, 256]),
    ("trunk.2.bias", &[256]),
    ("value.weight", &[256, 1]),
    ("value.bias", &[1]),
    ("kind.weight", &[256, 16]),
    ("kind.bias", &[16]),
    ("kind_embedding.weight", &[16, 32]),
    ("unit_embedding.weight", &[2, 32]),
    ("ability_embedding.weight", &[8, 16]),
    ("item_embedding.weight", &[15, 16]),
    ("controlled.weight", &[336, 2]),
    ("controlled.bias", &[2]),
    ("ability_head.weight", &[336, 8]),
    ("ability_head.bias", &[8]),
    ("item_head.weight", &[336, 15]),
    ("item_head.bias", &[15]),
    ("swap_head.weight", &[336, 15]),
    ("swap_head.bias", &[15]),
    ("learn_head.weight", &[336, 6]),
    ("learn_head.bias", &[6]),
    ("shop_head.weight", &[336, 64]),
    ("shop_head.bias", &[64]),
    ("loot_head.weight", &[336, 16]),
    ("loot_head.bias", &[16]),
    ("target_mode.weight", &[336, 3]),
    ("target_mode.bias", &[3]),
    ("put_mode.weight", &[336, 2]),
    ("put_mode.bias", &[2]),
    ("entity_query.weight", &[336, 128]),
    ("entity_query.bias", &[128]),
    ("point_query.weight", &[336, 64]),
    ("point_query.bias", &[64]),
];

const M14_GLOBAL_FEATURES: usize = 72;
const M14_UNIT_FEATURES: usize = 73;
const GLOBAL_INSERTED: usize = super::GLOBAL_FEATURES - M14_GLOBAL_FEATURES;
const UNIT_INSERTED: usize = super::UNIT_FEATURES - M14_UNIT_FEATURES;
const _: () = assert!(super::GLOBAL_FEATURES > M14_GLOBAL_FEATURES);
const _: () = assert!(super::UNIT_FEATURES > M14_UNIT_FEATURES);
const _: () = assert!(super::GLOBAL_FEATURES == 85);
const _: () = assert!(super::UNIT_FEATURES == 84);
const _: () = assert!(MODEL_PARAMETER_COUNT == 1_696_436);
const _: () = assert!(
    MODEL_PARAMETER_COUNT == M14_PARAMETER_COUNT + UNIT_INSERTED * 64 + GLOBAL_INSERTED * 512
);

impl PolicyModel {
    /// Imports all M16 parameter bits with the same audited names, ordering and shapes.
    pub(crate) fn initialize_m16_navigation_parameters(
        &self,
        source: &[f32],
    ) -> Result<(), ModelError> {
        super::validate_parameter_values(source)?;
        let schema = self.parameter_schema()?;
        Self::validate_map2_parameter_schema(&schema)?;
        assert_eq!(schema.len(), 62);
        assert_eq!(source.len(), 1_696_436);
        self.import_parameters(source)
    }

    /// Checks the entire destination layout before interpreting an immutable M14 flat payload.
    pub(crate) fn validate_map2_parameter_schema(
        schema: &[(&str, Vec<usize>)],
    ) -> Result<(), ModelError> {
        if schema.len() != M14_LAYOUT.len() {
            return Err(ModelError::InvalidModelState("Map2 parameter tensor count"));
        }
        let mut count = 0;
        for ((name, shape), (old_name, old_shape)) in schema.iter().zip(M14_LAYOUT) {
            if *name != old_name {
                return Err(ModelError::InvalidModelState(
                    "Map2 parameter name or order",
                ));
            }
            let expected = match old_name {
                "unit.0.weight" => &[super::UNIT_FEATURES, 64][..],
                "trunk.0.weight" => &[super::TRUNK_INPUT, 512][..],
                _ => old_shape,
            };
            if shape != expected {
                return Err(ModelError::InvalidModelState("Map2 parameter shape"));
            }
            count += expected.iter().product::<usize>();
        }
        assert_eq!(count, MODEL_PARAMETER_COUNT);
        assert_eq!(schema.len(), 62);
        Ok(())
    }

    /// Pure Candle F32 row insertion; installs nothing and performs no arithmetic on source bits.
    pub(crate) fn widen_m14_input_parameters(
        &self,
        source: &[f32],
    ) -> Result<Vec<f32>, ModelError> {
        if source.len() != M14_PARAMETER_COUNT {
            return Err(ModelError::ParameterLength {
                actual: source.len(),
                expected: M14_PARAMETER_COUNT,
            });
        }
        if let Some(index) = source.iter().position(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteParameter { index });
        }
        Self::validate_map2_parameter_schema(&self.parameter_schema()?)?;
        // Initialization is infrequent: CPU copies preserve even subnormal and signed-zero bits.
        let device = super::Device::Cpu;
        let flat = Tensor::from_slice(source, source.len(), &device)?;
        let mut tensors = Vec::with_capacity(M14_LAYOUT.len());
        let mut offset = 0;
        for (name, shape) in M14_LAYOUT {
            let count = shape.iter().product::<usize>();
            let tensor = flat.narrow(0, offset, count)?.reshape(shape)?;
            let padded = match name {
                "unit.0.weight" => insert_rows(&tensor, M14_UNIT_FEATURES, UNIT_INSERTED)?,
                "trunk.0.weight" => insert_rows(&tensor, M14_GLOBAL_FEATURES, GLOBAL_INSERTED)?,
                _ => tensor,
            };
            tensors.push(padded.flatten_all()?);
            offset += count;
        }
        assert_eq!(offset, M14_PARAMETER_COUNT);
        let target: Vec<f32> = Tensor::cat(&tensors, 0)?.to_vec1()?;
        super::validate_parameter_values(&target)?;
        assert_eq!(target.len(), MODEL_PARAMETER_COUNT);
        Ok(target)
    }

    // Old ignored artifact utilities must fail closed rather than silently acquire Map2 semantics.
    #[cfg(test)]
    pub(crate) fn widen_m11_input_parameters(
        &self,
        source: &[f32],
    ) -> Result<Vec<f32>, ModelError> {
        if source.len() != 1_684_724 {
            return Err(ModelError::ParameterLength {
                actual: source.len(),
                expected: 1_684_724,
            });
        }
        if let Some(index) = source.iter().position(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteParameter { index });
        }
        Err(ModelError::InvalidModelState(
            "M11 initialization retired; use pinned M14 Map2 initialization",
        ))
    }

    #[cfg(test)]
    pub(crate) fn validate_m12_parameter_schema(
        _schema: &[(&str, Vec<usize>)],
    ) -> Result<(), ModelError> {
        Err(ModelError::InvalidModelState(
            "M12 initialization retired; use pinned M14 Map2 initialization",
        ))
    }
}

fn insert_rows(tensor: &Tensor, prefix: usize, inserted: usize) -> Result<Tensor, ModelError> {
    let (rows, columns) = tensor.dims2()?;
    assert!(prefix <= rows);
    assert!(inserted > 0);
    let leading = tensor.narrow(0, 0, prefix)?;
    let zeros = Tensor::zeros((inserted, columns), DType::F32, tensor.device())?;
    if prefix == rows {
        return Ok(Tensor::cat(&[&leading, &zeros], 0)?);
    }
    let trailing = tensor.narrow(0, prefix, rows - prefix)?;
    Ok(Tensor::cat(&[&leading, &zeros, &trailing], 0)?)
}
