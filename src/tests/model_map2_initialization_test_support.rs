use super::*;

impl PolicyModel {
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
