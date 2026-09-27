use super::*;

#[cfg(test)]
impl RaggedFeatureHeader {
    pub(crate) fn corrupt_unit_offset_for_test(&mut self) {
        self.units.offset = u32::MAX;
    }
}
