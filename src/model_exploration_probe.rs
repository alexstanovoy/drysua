use super::*;

const ACTOR_LAYERS: [&str; 12] = [
    "kind",
    "controlled",
    "ability_head",
    "item_head",
    "swap_head",
    "learn_head",
    "shop_head",
    "loot_head",
    "target_mode",
    "put_mode",
    "entity_query",
    "point_query",
];

pub(crate) struct ActorRange {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
    pub selected: bool,
}

pub(crate) struct ActorSoftening {
    pub parameters: Vec<f32>,
    pub ranges: Vec<ActorRange>,
}

impl PolicyModel {
    pub(crate) fn softened_actor_parameters(&self) -> Result<ActorSoftening, ModelError> {
        assert_eq!(MODEL_SCHEMA_VERSION, 22);
        let _guard = self.read_parameter_lock()?;
        let tensors = self.parameters();
        assert_eq!(tensors.len(), 62);
        let mut parameters = self.export_parameters_locked()?;
        validate_parameter_values(&parameters)?;
        let mut offset = 0;
        let mut ranges = Vec::with_capacity(62);
        let expected = exploration_parameter_names();
        let actual: std::collections::BTreeSet<_> = tensors
            .iter()
            .map(|tensor| tensor.name.to_owned())
            .collect();
        assert_eq!(actual, expected);
        for tensor in tensors {
            let end = offset + tensor.value.elem_count();
            assert!(end <= parameters.len());
            let selected = ACTOR_LAYERS.iter().any(|layer| {
                tensor.name == format!("{layer}.weight") || tensor.name == format!("{layer}.bias")
            });
            if selected {
                for value in &mut parameters[offset..end] {
                    *value = half_actor_value(*value)?;
                }
            }
            ranges.push(ActorRange {
                name: tensor.name,
                shape: tensor.value.dims().to_vec(),
                begin: offset,
                end,
                selected,
            });
            offset = end;
        }
        assert_eq!(offset, MODEL_PARAMETER_COUNT);
        assert_eq!(ranges.iter().filter(|range| range.selected).count(), 24);
        validate_parameter_values(&parameters)?;
        Ok(ActorSoftening { parameters, ranges })
    }
}

fn exploration_parameter_names() -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    for (prefix, layers) in [
        ("unit", 3),
        ("ability", 2),
        ("item", 2),
        ("point", 2),
        ("projectile", 2),
        ("loot", 2),
        ("trunk", 3),
    ] {
        for index in 0..layers {
            for suffix in ["weight", "bias"] {
                names.insert(format!("{prefix}.{index}.{suffix}"));
            }
        }
    }
    for prefix in ACTOR_LAYERS.into_iter().chain(["value"]) {
        names.insert(format!("{prefix}.weight"));
        names.insert(format!("{prefix}.bias"));
    }
    for prefix in ["kind", "unit", "ability", "item"] {
        names.insert(format!("{prefix}_embedding.weight"));
    }
    assert_eq!(names.len(), 62);
    names
}

fn half_actor_value(value: f32) -> Result<f32, ModelError> {
    if !value.is_finite() {
        return Err(ModelError::InvalidModelState("actor softening nonfinite"));
    }
    let changed = value * 0.5;
    if value != 0.0
        && (changed == 0.0
            || changed.is_subnormal()
            || (changed * 2.0).to_bits() != value.to_bits())
    {
        return Err(ModelError::InvalidModelState("actor softening underflow"));
    }
    assert!(changed.is_finite());
    assert_eq!((changed * 2.0).to_bits(), value.to_bits());
    Ok(changed)
}

#[test]
fn half_actor_value_preserves_exact_normal_bits_and_signed_zero() {
    for value in [0.0_f32, -0.0, 0.25, -12.5, f32::MAX] {
        let changed = half_actor_value(value).unwrap();
        assert_eq!((changed * 2.0).to_bits(), value.to_bits());
        assert_eq!(changed.to_bits(), (value * 0.5).to_bits());
    }
}

#[test]
fn half_actor_value_rejects_nonfinite_and_underflow() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            half_actor_value(value).unwrap_err().to_string(),
            "model produced invalid actor softening nonfinite"
        );
    }
    for value in [f32::from_bits(1), f32::MIN_POSITIVE, -f32::MIN_POSITIVE] {
        assert_eq!(
            half_actor_value(value).unwrap_err().to_string(),
            "model produced invalid actor softening underflow"
        );
    }
}

#[test]
fn actor_softening_selects_exact_24_of_62_tensors_and_never_mutates_source() {
    let model = PolicyModel::fresh_on(10104200, PolicyDevice::Cpu).unwrap();
    let identity = model.policy_identity().unwrap();
    let original = model.export_parameters().unwrap();
    let softened = model.softened_actor_parameters().unwrap();
    assert_eq!(softened.ranges.len(), 62);
    assert_eq!(
        softened
            .ranges
            .iter()
            .filter(|range| range.selected)
            .count(),
        24
    );
    for range in &softened.ranges {
        if range.name.starts_with("value.") || range.name.contains("embedding") {
            assert!(!range.selected);
        }
        for (source, target) in original[range.begin..range.end]
            .iter()
            .zip(&softened.parameters[range.begin..range.end])
        {
            let expected = if range.selected {
                source * 0.5
            } else {
                *source
            };
            assert_eq!(target.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(model.export_parameters().unwrap(), original);
    assert_eq!(model.policy_identity().unwrap(), identity);
}
