use super::{MODEL_PARAMETER_COUNT, MODEL_SCHEMA_VERSION, ModelError, PolicyModel};

const M11_PARAMETERS: usize = 1_684_724;
const M11_LAYOUT: [(&str, &[usize]); 62] = [
    ("unit.0.weight", &[69, 64]),
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
    ("trunk.0.weight", &[2568, 512]),
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

const _: () = assert!(MODEL_SCHEMA_VERSION == 14);
const _: () = assert!(MODEL_PARAMETER_COUNT == M11_PARAMETERS + 4 * 64 + 8 * 512);

impl PolicyModel {
    /// The M12 flat artifact has this exact named layout, with only two widths changed from M11.
    pub(crate) fn validate_m12_parameter_schema(
        schema: &[(&str, Vec<usize>)],
    ) -> Result<(), ModelError> {
        if schema.len() != M11_LAYOUT.len() {
            return Err(ModelError::InvalidModelState(
                "selected M12 parameter tensor count",
            ));
        }
        let mut count = 0;
        for ((name, shape), (expected_name, old_shape)) in schema.iter().zip(M11_LAYOUT) {
            if *name != expected_name {
                return Err(ModelError::InvalidModelState(
                    "selected M12 parameter name or order",
                ));
            }
            let expected_shape = match expected_name {
                "unit.0.weight" => &[73, 64][..],
                "trunk.0.weight" => &[2576, 512][..],
                _ => old_shape,
            };
            if shape != expected_shape {
                return Err(ModelError::InvalidModelState(
                    "selected M12 parameter shape",
                ));
            }
            count += expected_shape.iter().product::<usize>();
        }
        assert_eq!(count, 1_689_076);
        assert_eq!(count, MODEL_PARAMETER_COUNT);
        Ok(())
    }

    pub(crate) fn widen_m11_input_parameters(
        &self,
        source: &[f32],
    ) -> Result<Vec<f32>, ModelError> {
        if source.len() != M11_PARAMETERS {
            return Err(ModelError::ParameterLength {
                actual: source.len(),
                expected: M11_PARAMETERS,
            });
        }
        if let Some(index) = source.iter().position(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteParameter { index });
        }
        let schema = self.parameter_schema()?;
        assert_eq!(schema.len(), M11_LAYOUT.len());
        let mut target = vec![0.0; MODEL_PARAMETER_COUNT];
        let mut source_offset = 0;
        let mut target_offset = 0;
        for ((name, shape), (old_name, old_shape)) in schema.iter().zip(M11_LAYOUT) {
            assert_eq!(*name, old_name);
            let old_count = old_shape.iter().product::<usize>();
            let count = shape.iter().product::<usize>();
            let (prefix, inserted) = match *name {
                "unit.0.weight" => {
                    assert_eq!(shape, &[73, 64]);
                    (69 * 64, 4 * 64)
                }
                "trunk.0.weight" => {
                    assert_eq!(shape, &[2576, 512]);
                    (64 * 512, 8 * 512)
                }
                _ => {
                    assert_eq!(shape, old_shape);
                    (old_count, 0)
                }
            };
            assert_eq!(count, old_count + inserted);
            let original = &source[source_offset..source_offset + old_count];
            let replacement = &mut target[target_offset..target_offset + count];
            replacement[..prefix].copy_from_slice(&original[..prefix]);
            replacement[prefix + inserted..].copy_from_slice(&original[prefix..]);
            source_offset += old_count;
            target_offset += count;
        }
        assert_eq!(source_offset, source.len());
        assert_eq!(target_offset, target.len());
        Ok(target)
    }
}
